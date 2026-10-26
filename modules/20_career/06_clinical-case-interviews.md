# Clinical Case Interviews: Structuring an Ambiguous Problem Out Loud

> Module 20 — Career, Portfolio, and Interviews
> Topic 06 of 10
> Estimated study time: 3 hours
> Prerequisites: 04, 05
> Version 2026.3

## 1. Why this topic exists

The clinical case interview is the stage where a biomedical employer finds out whether you can be
trusted with an ambiguous request. The prompt is deliberately underspecified: "A hospital wants to
predict readmissions. How would you approach it?" There is no data dictionary, no defined user,
and no stated capacity. The interviewer is not testing whether you know an algorithm. They are
testing whether you ask the questions that a competent analyst would ask before building anything.

The format exists because the job looks like the prompt. A clinical leader arrives with a problem
that has been through three committees, a vendor has already demoed something, and the actual
decision is unclear. The analyst who starts with a model produces something impressive and
unusable. The analyst who starts with the decision produces something modest and adopted.

### 1.1 What the interviewer is listening for

Five signals, in roughly this order:

1. **Framing.** Do you name the decision, the actor, and the moment before you name a method?
2. **Harm awareness.** Do you consider what happens when the model is wrong, and who is harmed?
3. **Workflow and capacity.** Do you ask who acts on the output and how many actions the system
   can absorb?
4. **Evidence discipline.** Do you distinguish discrimination from calibration, and validation
   from deployment?
5. **Equity and monitoring.** Do you consider subgroups and what would make you stop?

A candidate who covers three of the five and is honest about the rest usually passes. A candidate
who covers none and is fluent about gradient boosting usually fails, and often does not find out
why.

### 1.2 The failure that looks like expertise

The classic weak answer begins: "I would use XGBoost with features from the electronic health record (EHR), split 80/20, and
optimise area under the receiver operating characteristic curve." It is technically coherent and answers a question nobody asked. It does not name
the decision, the user, the outcome definition, the intervention, the capacity, or the harm. The
interviewer's note is usually "went straight to the model" or "no clinical framing."

The second failure is the opposite: a candidate who discusses clinical workflow for 35 minutes and
never commits to a design. The interviewer cannot tell whether the candidate can build anything.
The case needs both: framing first, then a specific, defensible design.

### 1.3 Why the clinical case is different from a product case

A product case asks what to build and for whom. A clinical case adds three constraints that change
the answer:

- **The intervention may not exist.** Predicting a deterioration is useless if no team can
  respond. The case often hinges on whether the action is available.
- **The harm is asymmetric.** A false negative in a sepsis alert can kill; a false positive can
  consume a nurse's shift. The threshold is a clinical and ethical choice, not a statistical one.
- **The evidence standard is external.** A model that improves an internal metric may still fail a
  regulator, a payer, or a clinician's trust. The case expects you to know which standard applies.

### 1.4 What this topic produces

A repeatable structure for answering a clinical case aloud, a worked example with the arithmetic
of capacity and threshold selection, and a runnable calculation that shows how the operating point
changes when capacity changes. It also gives you the vocabulary to discuss harm and equity without
sounding like a compliance manual.

### 1.5 The case is a rehearsal of the first stakeholder meeting

The format is artificial, but the skill is not. In the first month of a clinical data role, someone
will ask for a readmission model in a meeting where the decision, the data, and the capacity are
all unstated. The person who asks three questions before agreeing to a design is the person who
will still be trusted in month twelve.

Practise the case as a meeting, not as an exam. Take notes, draw the chain, and ask the
interviewer questions as you would ask a programme manager. The interviewer is playing a
stakeholder, and treating them as one is part of the assessment.

### 1.6 What this topic does not cover

It does not teach clinical medicine. You are not expected to diagnose; you are expected to ask the
right questions and to know when to defer to a clinician. It does not cover system design (Topic
07), which is the next stage in many loops, or the technical implementation (Topic 05). The
regulatory material in Section 8 is orientation, not legal advice.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Open a clinical case by naming the decision, the actor, and the moment | Exercise 1; self-check 1 |
| 2 | Define the cohort, index event, outcome, and time zero without prompting | Exercise 2; self-check 2 |
| 3 | State the label's limitations and the direction of the resulting bias | Exercise 3; self-check 3 |
| 4 | Choose an operating threshold from capacity and harm, not from a default | Exercise 4; self-check 4 |
| 5 | Compute alerts per week, positive predictive value, and sensitivity at a given capacity | Exercise 5; self-check 5 |
| 6 | Describe the workflow, the alert burden, and the fallback when the model is wrong | Exercise 6; self-check 6 |
| 7 | Identify equity concerns specific to the outcome and the data source | Exercise 7; self-check 7 |
| 8 | State a monitoring plan and a stop criterion before deployment | Exercise 8; interview question 9 |

Outcomes 4 and 5 are where most candidates are weakest, because they require arithmetic rather
than vocabulary. Outcome 8 is the one that signals seniority.

## 3. The core mental model

A clinical case is a chain from a request to a monitored decision. Each link can break, and the
interviewer is listening for whether you know where the links are.

```text
  REQUEST          "Predict readmissions."
      |
      v
  DECISION         Who decides what, at which moment, with what alternatives?
      |
      v
  POPULATION       Who is eligible, from when, followed for how long?
      |
      v
  LABEL            What counts as the outcome, and what does the label miss?
      |
      v
  MODEL            What estimate supports the decision, validated how?
      |
      v
  THRESHOLD        What operating point fits the capacity and the harm?
      |
      v
  WORKFLOW         Who sees the output, what do they do, what if they disagree?
      |
      v
  OUTCOME          What changes for patients, and how would you know?
      |
      v
  MONITOR          What would make you recalibrate, pause, or withdraw?
```

The chain is read top to bottom. A candidate who starts at MODEL has skipped four links and will
be asked to go back. A candidate who walks the chain in order can stop at any link and go deep
when the interviewer probes.

### 3.1 The ten questions of the case spine

These are the questions you ask, in this order, in the first five to eight minutes.

| # | Question | Why it matters |
|---:|---|---|
| 1 | What decision does this change, and who makes it? | Without a decision there is no product |
| 2 | What do they do today, and how well does it work? | The baseline defines the value |
| 3 | What is the intervention, and does it have capacity? | A prediction without an action is a report |
| 4 | Who is eligible, and when does follow-up start? | Defines cohort and time zero |
| 5 | How is the outcome defined, and what does it miss? | Defines the label and its bias |
| 6 | What would a good model add over the current rule? | Defines the comparator and the claim |
| 7 | How would you validate it here, and in another site? | Defines the evidence standard |
| 8 | What operating point fits the capacity and the harm? | Defines the threshold |
| 9 | Who is harmed when it is wrong, and are the harms equal across groups? | Defines equity and safety |
| 10 | What would make you stop or withdraw it? | Defines monitoring and judgment |

You will not ask all ten in every case. You should know which ones you have not asked, and say so:
"I have not covered monitoring yet; I will come back to it."

### 3.2 Three layers of failure

| Layer | Failure | Example | Detection |
|---|---|---|---|
| Decision | The model answers a question no one acts on | Predicting readmission with no programme to enrol patients | Ask who acts and what they do |
| Estimate | The number is wrong or misleading | AUC 0.72 but calibration slope 0.6 at the threshold used | Calibration and decision curve |
| Implementation | The output never reaches the decision | Alert buried in a tab, or fires after discharge | Workflow mapping and silent-mode pilot |

Most candidates spend their preparation on the estimate layer. The interviewer usually spends most
of the case on the decision and implementation layers, because that is where deployments fail.

### 3.3 Prediction is not intervention

The single most useful distinction in a clinical case is between the probability of an outcome and
the effect of acting on it.

```text
  P(readmission | features)          -> who is at risk
  P(readmission | do X, features)    -> whether X helps this patient
```

A model estimates the first. The programme's value depends on the second. A case answer that
conflates them will recommend enrolling the highest-risk patients even when the intervention works
best for a different group, which is a common and expensive error.

If the interviewer asks "does the model reduce readmissions," the correct answer is that the model
does not; the programme does, and the model only selects who receives it. The evaluation design
must therefore compare outcomes between enrolled and not-enrolled patients, or run a stepped-wedge
or randomised pilot, not compare model-positive with model-negative patients.

### 3.4 The case as a written memo

Many loops ask for a one-page written version after the verbal case. The memo has the same spine
and a different order: recommendation first, then the decision, the evidence, the operating point,
the risks, and the next step. Write the recommendation in the first two sentences, because the
reader may stop there.

A memo is not a transcript of the case. It is the decision document the case was rehearsing, and
the interviewer reads it for whether you can commit to a recommendation after exploring the
trade-offs.

### 3.5 Thresholds are a capacity decision

A threshold is not a statistical constant. It is the point where the number of patients flagged
matches the number the system can act on, adjusted for the relative harm of a miss and a false
alarm.

```text
  expected alerts per period = eligible patients per period x flag rate at threshold
  capacity constraint        = alerts must be <= actions available
  choice                     = lowest threshold that fits capacity, given harm asymmetry
```

If capacity is 60 actions per month and the eligible population is 800 per month, the flag rate
cannot exceed 7.5 percent. Any threshold that flags more produces alerts that no one can act on,
which trains staff to ignore the system. This is the arithmetic that separates a candidate who has
deployed something from one who has only trained models.

> **Note:** The capacity constraint assumes the intervention is scarce. If the action is cheap and
> scalable, such as sending an educational message, the constraint is the harm of false positives
> rather than staff time. Ask which situation you are in.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Index admission | The admission that starts the follow-up window | Confused with any admission | Definition |
| Time zero | The moment at which prediction is made and features must be available | Set at admission when the decision is at discharge | Timestamp |
| Label | The outcome definition used to train and evaluate | Assumed to be the clinical outcome of interest | Definition |
| Care gap | The difference between recommended and delivered care | Confused with a risk score | Proportion |
| Intervention | The action triggered by the prediction | Omitted from the case answer | Action |
| Capacity | The number of actions the system can deliver per period | Assumed to be unlimited | Actions per period |
| Positive predictive value (PPV) | The share of flagged patients who have the outcome | Confused with sensitivity | Proportion |
| Sensitivity | The share of outcome cases that are flagged | Confused with PPV | Proportion |
| Number needed to treat (NNT) | Patients who must receive an intervention to prevent one event | Confused with number needed to screen | Count |
| Alert burden | Alerts per clinician per shift or per patient-day | Reported as a total without a denominator | Rate |
| Override rate | The share of alerts dismissed without the recommended action | Treated as evidence the alert is useless | Proportion |
| Alert fatigue | Desensitisation from excessive alerts | Used loosely for any annoyance | Construct |
| Clinical decision support (CDS) | Software that provides patient-specific assessments or recommendations | Confused with any dashboard | Regulatory category |
| Human in the loop | A clinician reviews or approves the output before action | Assumed to make the system safe | Design |
| Shadow mode | Running a model without showing its output, to compare with practice | Confused with a pilot | Deployment mode |
| Silent mode | Running the model and logging alerts without delivering them | Same as shadow mode in some usage | Deployment mode |
| Stepped-wedge | A trial design in which sites switch from control to intervention in sequence | Confused with a before-after study | Design |
| Risk-standardised rate | A rate adjusted for case mix under a specified model | Confused with a crude rate | Rate |
| Excess readmission ratio (ERR) | Predicted rate divided by expected rate under the CMS measure | Confused with the raw readmission rate | Ratio |
| Social risk | Social and economic factors affecting outcomes | Confused with clinical risk | Construct |
| Escalation | The path when a clinician disagrees with the output | Omitted from workflow descriptions | Process |
| Stop criterion | A pre-specified condition for pausing or withdrawing the model | Confused with a performance target | Condition |
| Subgroup harm | Differential false negatives or false positives across groups | Reduced to "we check fairness" | Analysis |
| Intended use | The population, user, and decision the model supports | Assumed to be obvious | Statement |

## 5. The mechanics, step by step

The procedure is a 35- to 45-minute answer. It is written as steps because the order is the
signal, but in a real case you will interleave them. The interviewer will interrupt; when they do,
answer the interruption and return to the step you left.

### Step 1. Restate the request and bound the case

**Input:** the prompt.
**Output:** one sentence that restates the problem and one sentence that bounds it.
**Rationale:** restating confirms understanding and gives you three seconds to think. Bounding
prevents the case from expanding into "improve the health system."

Example: "The request is to reduce readmissions using prediction. I will bound it to adult
medical-surgical discharges from this hospital, a 30-day window, and a transitional care
programme as the intervention. If the real scope is different, tell me and I will adjust."

### Step 2. Name the decision, the actor, and the moment

**Input:** the bounded problem.
**Output:** a decision statement.
**Rationale:** every subsequent choice depends on this. A model that supports discharge planning
has a different time zero from one that supports a post-discharge call.

```text
[Role] decides [choice] at [moment] using [output], and a wrong choice costs [consequence].
```

Example: "The transitional care team decides which patients to enrol in the 30-day programme at
discharge, using a risk estimate, and a wrong choice costs either a preventable readmission or a
slot that could have gone to someone else."

If there are several actors, name the primary one and say how the others interact. "The navigator
enrols, the attending can override, and the programme manager owns the capacity. I will optimise
for the navigator's decision because that is where the model is used." A case with several actors
is not a reason to avoid committing; it is a reason to say which decision you are supporting.

### Step 2b. Separate the decision from the metric

**Input:** the decision statement.
**Output:** the metric the decision-maker is accountable for, distinct from the model metric.
**Rationale:** the decision-maker is accountable for readmissions, cost, or throughput, not for
AUROC. A case answer that optimises the model metric without connecting it to the accountable
metric reads as junior.

Ask: "What is the programme manager measured on?" If the answer is "readmissions per 1,000
discharges," the model's value is in moving that number, and the evaluation must be designed
around it. If the answer is "slots filled," the model's value is in fill rate and appropriateness,
which is a different problem.

### Step 3. Ask what happens today and how well it works

**Input:** the decision statement.
**Output:** the current rule, its performance, and its cost.
**Rationale:** the baseline is the comparator. Without it, "the model has an AUC of 0.72" is
unfalsifiable.

Questions: What rule is used now? Who applies it? How many patients does it flag? What is the
current readmission rate for flagged and unflagged patients? Has anything been tried and
abandoned?

If the interviewer says there is no current rule, the baseline is "flag everyone" or "flag
nobody," and the case becomes a capacity and harm problem rather than a discrimination problem.

### Step 4. Define the population, index event, time zero, and outcome

**Input:** the decision and the workflow.
**Output:** four definitions, each with an inclusion and an exclusion.
**Rationale:** these definitions determine the denominator and the label. Most case errors are
definition errors disguised as modelling errors.

| Element | Question to ask | Example answer |
|---|---|---|
| Population | Which admissions are eligible? | Adults 18+, medical-surgical, discharged alive, continuous enrolment |
| Index event | What starts the clock? | Discharge from the index admission |
| Time zero | When is the prediction made? | 24 hours before discharge, when the programme decides |
| Outcome | What counts, over what window? | Unplanned all-cause readmission within 30 days, excluding planned |
| Exclusions | What is out of scope? | Transfers to another acute hospital, deaths, discharges against advice |

State the direction of each choice. "Including observation stays would raise the rate by about
two percentage points and would change the denominator" is stronger than listing definitions
without consequences.

### Step 5. Interrogate the label

**Input:** the outcome definition.
**Output:** three limitations and the direction of the bias each creates.
**Rationale:** the label is a proxy. A candidate who names its limitations before the interviewer
does is demonstrating the judgment the case is designed to test.

| Label issue | Mechanism | Direction of bias |
|---|---|---|
| Only same-system readmissions are observed | Patients readmitted elsewhere are counted as not readmitted | Understates risk for patients who move or travel |
| Planned readmissions excluded by a flag | The flag is incomplete for some payers | Overstates the unplanned rate in those groups |
| 30-day window truncates at insurance changes | Disenrolment removes observed events | Understates risk for churned populations |
| Diagnosis codes are used to define the cohort | Coding varies by site and by payer | Differential misclassification across sites |
| The outcome is recorded at discharge | Post-discharge events are invisible | Understates the outcome for short stays |

The sentence to practise: "The label understates readmissions for patients who leave the system,
which is likely correlated with social risk, so the model may be least accurate for the patients
who need the programme most."

### Step 6. State the claim and the comparator

**Input:** the baseline and the label.
**Output:** one falsifiable claim with a comparator and a metric.
**Rationale:** the claim determines the validation design and the evidence the interviewer will
ask for.

Weak claim: "The model predicts readmission." Strong claim: "At the capacity of 60 enrolments per
month, the model identifies patients with a 30-day readmission risk at least 15 percentage points
higher than the current rule, with calibration slope between 0.9 and 1.1."

The claim should be falsifiable at the operating point, not at an abstract threshold.

### Step 7. Design the validation

**Input:** the claim.
**Output:** a split strategy, an external validation plan, and a leakage audit.
**Rationale:** a clinical case expects temporal validation at minimum, and external validation if
the model will be used at another site.

Say: development on the earliest 24 months, temporal validation on the next 12, and a prospective
silent-mode evaluation for three months before any alert is shown. Name the leakage risks: features
recorded after time zero, patients in both sets, site-specific coding, and preprocessing fitted on
the full data.

### Step 8. Choose the metrics the decision needs

**Input:** the claim.
**Output:** a metric set with a reason for each.
**Rationale:** the metric set is the evidence the decision-maker will use. AUROC alone answers a
ranking question, not a capacity question.

| Metric | What it answers | Why the case needs it |
|---|---|---|
| AUROC | Can the model rank? | Necessary but not sufficient |
| AUPRC | Does ranking hold at low prevalence? | Readmission prevalence is often 10 to 15 percent |
| Calibration | Are the probabilities right? | The threshold uses the probability |
| Decision curve | Is there net benefit at plausible thresholds? | Connects to the capacity constraint |
| Subgroup metrics | Does it hold across groups? | Equity and safety |
| Alert burden | How many alerts per clinician? | Workflow feasibility |

### Step 9. Set the threshold from capacity and harm

**Input:** the capacity, the eligible population, and the harm asymmetry.
**Output:** a threshold, the expected alerts per period, PPV, and sensitivity.
**Rationale:** this is the step most candidates skip, and the one that most clearly separates
someone who has deployed from someone who has trained.

The arithmetic:

```text
eligible per month           = 800 discharges
capacity                     = 60 enrolments per month
maximum flag rate            = 60 / 800 = 7.5%
choose the threshold whose
cumulative flag rate <= 7.5% = the top 60 patients by score
report                       = alerts/month, PPV, sensitivity, NNT
```

Then discuss the harm asymmetry. If a missed high-risk patient is much worse than a wasted slot,
you might accept a lower threshold and a larger queue, with a triage step. If the programme is
expensive and slots are scarce, a higher threshold with a smaller queue is better. The threshold
is a policy choice informed by the model, not a property of the model.

### Step 10. Describe the workflow and the alert burden

**Input:** the threshold.
**Output:** a workflow description with the actor, the channel, the timing, and the fallback.
**Rationale:** the output has to reach a person at the moment of decision, and that person has to
be able to act.

Cover: who sees the list (role, not team), when (24 hours before discharge, in the discharge
huddle), in what system (EHR worklist, not email), what they do (enrol, defer with a reason, or
decline), what happens if they disagree (escalation to the attending), and what happens if the
model is unavailable (the current rule).

Alert burden: 60 per month is about 3 per working day across the team, which is manageable. 260 per
month is about 13 per working day, which is not, and will produce override behaviour within weeks.

### Step 11. Address equity explicitly

**Input:** the population and the label.
**Output:** two specific equity risks and a monitoring plan for each.
**Rationale:** generic statements about fairness are not scored. Specific mechanisms are.

Examples: the label under-captures readmissions for patients who use other systems, which is
correlated with insurance churn; the features include prior utilisation, which is lower for
patients with limited access, so the model may systematically under-flag them; the programme
requires a phone and a stable address, so enrolment itself may be inequitable.

For each, state the check: subgroup sensitivity at the chosen threshold, enrolment rates by
group, and a review of the features that proxy for access.

### Step 12. State monitoring and the stop criterion

**Input:** the deployment plan.
**Output:** a monitoring plan with a frequency and a stop rule.
**Rationale:** this is the closing signal. A candidate who names what would make them withdraw the
model is demonstrating that they understand deployment as an ongoing commitment.

Monitor: calibration monthly, alert volume weekly, override rate weekly, subgroup performance
quarterly, and outcome (readmission rate among enrolled) quarterly. Stop or pause if calibration
slope falls outside 0.8 to 1.2, if the override rate exceeds 60 percent for two consecutive
months, or if a subgroup's sensitivity falls below half the overall rate.

### Step 13. Close with what you would need and what would change your mind

**Input:** the whole case.
**Output:** three data needs and one condition that would reverse the recommendation.
**Rationale:** the close is what the interviewer writes down. It should be short and specific.

Example: "I would need the discharge-level dataset with 24 months of history, the programme's
actual capacity and enrolment criteria, and a clinical owner for the workflow. I would not deploy
if the temporal validation showed calibration slope below 0.8 at the operating point, or if the
programme could not guarantee the 60 slots."

### Step 14. Handle interruptions without losing the thread

**Input:** an interviewer who interrupts.
**Output:** a short answer to the interruption and an explicit return to the case.
**Rationale:** interruptions are usually probes, not hostility. They test whether you can hold a
structure while being redirected, which is what a real stakeholder meeting feels like.

Three patterns:

1. **The depth probe.** "Tell me more about the calibration." Answer for 60 to 90 seconds, then
   say "coming back to the threshold" and continue.
2. **The challenge.** "I do not think that outcome definition is right." Ask what they would use
   and why, incorporate it, and state how the answer changes. Do not defend the original choice.
3. **The hypothetical.** "What if the programme had no capacity?" Answer the hypothetical briefly,
   then return: "that is the Section 6.7 variation; in the base case, capacity is 60."

If you lose the thread, say so and restate the chain in one sentence: "The decision is enrolment
at discharge, the constraint is 60 slots, and I was choosing the threshold. Let me continue
there." Interviewers score recovery, not flawlessness, and a candidate who names the recovery aloud is
easier to score than one who silently restarts.

### Career-context branches

**From academia.** You may know the methods and lack the workflow. Shadow a clinical team for 20 hours
before the interview, and say what you observed.

**No clinical background.** State the boundary early: "I would confirm the clinical workflow with the
care team." Then reason from the decision, which you can do without a licence.

**Non-traditional degree.** The case rewards structure and arithmetic more than credentials. Lead with
the decision and the capacity calculation.

**Visa constraints.** No effect on the case; the process timeline may be longer, so schedule the case
preparation later than the technical preparation.

**Remote interviews.** Draw the chain on a shared screen and narrate it. The visual structure
substitutes for the whiteboard.

**Return after a gap.** Refresh the domain vocabulary and one public dataset; the structure is durable,
the context is not.

**Over-levelled target.** Expect questions about monitoring, change control, and cost, not only about
the model. Prepare a stop criterion.

### Branch points and edge cases

**There is no intervention.** If no programme can act on the prediction, the case changes: the
honest answer is that a prediction alone will not reduce readmissions, and the useful artifact may
be a care-gap report rather than a model. Saying this is a strong answer, not a refusal.

**Capacity is unknown.** Assume a range and show how the threshold moves. "If capacity is 20, the
threshold rises and sensitivity falls to about 15 percent; if it is 100, the threshold falls and
PPV drops. The decision depends on which the programme can staff."

**Prevalence is very low.** At 2 percent prevalence, even a good model has low PPV at a usable
threshold. Say the arithmetic and propose a different decision (screening with a second test, or a
rule-based trigger) rather than forcing a model.

**A clinician on the panel is hostile to the model.** Do not defend the model; defend the decision.
Ask what would make them trust it, and propose the evidence that would answer that: silent-mode
comparison, a local validation, or a human-review step. Hostility is usually about being
overruled, not about statistics.

**The outcome is paediatric or rare.** The adult evidence does not transfer. Say so, and propose a
different validation and a smaller scope.

**The data are unavailable at time zero.** This is common: the best predictor is recorded after
the decision. Name the constraint and choose a weaker but available feature set, then state the
performance cost.

**The model would be a regulated device.** If the output gives a patient-specific treatment
recommendation and the clinician cannot independently review the basis, the software may meet the
device definition. Say that the intended use determines the pathway and that you would involve
regulatory affairs before deployment.

### Four prompts and what each is really testing

| Prompt | Surface question | What the interviewer wants |
|---|---|---|
| "Predict readmissions" | Which patients are at risk | Capacity, intervention, threshold, evaluation |
| "Detect sepsis earlier" | Which patients are deteriorating | Time zero, alert burden, harm of false negatives, escalation |
| "Predict no-shows" | Who will miss an appointment | Overbooking policy, equity, cost of a wasted slot |
| "Triage imaging" | Which scan to read first | Queueing, turnaround time, radiologist trust, failure modes |

The prompts differ in vocabulary and in the harm structure, but the spine is the same. Practise
one prompt per category rather than ten variations of the same one.

### Decision table: deployment mode

| Evidence available | Risk | Suggested mode |
|---|---|---|
| Retrospective validation only, new model, clinical decision affected | High | Shadow mode for 3 months, then a silent-mode comparison |
| External validation plus local calibration, workflow agreed | Moderate | Limited pilot with human review and a stop rule |
| Prospective evaluation, calibrated, monitoring in place | Lower | Live with a human in the loop and monthly review |
| Model used for research or quality improvement only | Low | No alert; report and review quarterly |

### Decision table: threshold choice under capacity pressure

| Capacity relative to eligible population | Threshold | Expected consequence |
|---|---|---|
| Capacity >= 20% of eligible | Low threshold is affordable | High sensitivity, moderate PPV, manageable alerts |
| Capacity 5–20% of eligible | Threshold at capacity | Moderate sensitivity, higher PPV, alerts match capacity |
| Capacity < 5% of eligible | High threshold, consider a second-stage screen | Low sensitivity, high PPV, many missed cases |
| Capacity unknown | Model the range and present the trade-off | Decision deferred to the programme owner |

## 6. Worked example

Aisha is 30, a health data scientist at a 400-bed community hospital. She is in a 45-minute
clinical case interview for a senior clinical data scientist role. The prompt is: "Our
readmission rate is above the national benchmark. How would you reduce it?"

### 6.1 What she does in the first six minutes

She does not mention a model. She asks:

1. Which readmissions are we talking about, and over what window? (All-cause, 30 days, adult
   medical-surgical.)
2. Who would act on a prediction, and what would they do? (A transitional care programme with
   nurse navigators; enrolment in a 30-day programme.)
3. How many patients can the programme take? (60 per month.)
4. What happens today? (A nurse reviews discharges informally; there is no rule.)
5. What is the readmission rate now? (11.1 percent, about 89 of 800 monthly discharges.)

Then she states the decision: "The programme manager decides which 60 patients to enrol at
discharge, and the cost of a wrong choice is either a preventable readmission or a slot taken from
someone who needed it."

The interviewer's note at this point, as later relayed, was "framed it before modelling."

### 6.2 Diagnostic 1: the first draft opened with the model

In her preparation, Aisha had written an answer that began: "I would build a gradient-boosted model
with prior utilisation, comorbidity, and lab features, validate temporally, and report AUROC." A
mock interviewer stopped her at 90 seconds and asked, "Who does what differently?" She could not
answer without inventing a user.

The fix was structural, not technical: she rewrote the opening as a decision statement and
practised it until it came first. In the real case, the first six minutes contained no method at
all. The method appeared only after the decision, capacity, and baseline were established.

### 6.3 Diagnostic 2: the threshold ignored capacity

Her second draft proposed a threshold of 0.20 because "that is where sensitivity crosses 50
percent." At 0.20, the model would flag 310 patients per month, more than five times the
programme's capacity.

The consequence is predictable: 200 alerts per month that no one can act on, a queue that grows,
and navigators who learn to ignore the list. The model's discrimination would be fine and the
deployment would fail.

The corrected arithmetic, using the score distribution in Section 7:

| Threshold | Alerts per month | PPV | Sensitivity | Fits capacity (60) |
|---|---:|---:|---:|---|
| 0.70 | 20 | 70.0% | 15.7% | Yes, with unused capacity |
| 0.60 | 40 | 57.5% | 25.8% | Yes |
| 0.50 | 60 | 51.7% | 34.8% | Yes, exactly |
| 0.40 | 110 | 38.2% | 47.2% | No |
| 0.20 | 310 | 21.6% | 75.3% | No, by 5.2 times |

The choice is 0.50: 60 alerts, 52 percent PPV, 35 percent sensitivity. About 31 of the 89 monthly
readmissions occur in flagged patients. The other 58 are missed, and the case must say so.

### 6.4 The intervention arithmetic

Suppose the programme reduces readmission risk by 20 percent relative among enrolled patients. The
assumption is illustrative; the actual effect must come from the programme's own evidence.

```text
enrolled per month            = 60
baseline risk among flagged   = 51.7%
absolute reduction            = 51.7% x 20% = 10.3 percentage points
readmissions prevented/month  = 60 x 0.103 = 6.2
readmissions prevented/year   = 6.2 x 12 = 74.4
NNT                           = 1 / 0.103 = 9.7, about 10 enrolled patients per readmission prevented
```

This is the number the programme manager cares about: about ten enrolments per prevented
readmission, and about 74 prevented per year. It also exposes the cost: 720 enrolments per year to prevent 79
events, which may or may not be worth the programme's cost.

### 6.5 The evaluation design

Aisha proposes a stepped-wedge pilot across four units over six months, because randomising
individual patients is ethically and operationally difficult when the programme is believed to
help. The primary outcome is 30-day readmission among eligible discharges, with the unit as the
clustering level. The analysis accounts for time trends, which are substantial in readmission
data.

She explicitly rejects the comparison of model-positive with model-negative patients: those groups
differ in ways the model does not capture, so the comparison would overstate the programme's
effect. The interviewer's follow-up was "why not compare flagged and unflagged," and her answer
was the strongest moment of the case.

### 6.6 The workflow and the stop criterion

The list is delivered to the discharge huddle at 08:00, 24 hours before expected discharge, as an
EHR worklist with three actions: enrol, defer with a reason, or decline with a reason. The
navigator can escalate to the attending if they disagree. If the model is unavailable, the current
informal review continues.

Monitoring: alert volume weekly, override rate weekly, calibration monthly, subgroup sensitivity
quarterly. Stop criteria: calibration slope outside 0.8 to 1.2 for two consecutive months, override
rate above 60 percent for two months, or sensitivity for any major subgroup below half the overall
rate.

### 6.7 Vary one assumption

Change one input: the programme can take only 20 patients per month.

The decision does not disappear, but the design changes.

1. **Threshold rises to 0.70.** Twenty alerts, PPV 70 percent, sensitivity 15.7 percent. The
   programme now serves the highest-risk patients and misses 84 percent of readmissions.
2. **The claim changes.** At this capacity, the model is not a population strategy; it is a
   targeted programme for the highest-risk patients. The evaluation should measure effect in that
   narrow group, where the baseline risk is high enough for a 20 percent relative reduction to be
   clinically meaningful.
3. **A second-stage screen becomes attractive.** With 20 slots, a rule-based pre-screen can
   reduce the modelling problem to a smaller, richer cohort. The model may not be the best use of
   the capacity at all.
4. **The honest recommendation may be not to deploy.** If the programme cannot expand and the
   model only reaches 15 percent of readmissions, the better investment may be a care-gap report
   that identifies missed follow-up appointments, which is cheaper and acts on a larger share of
   the population.

The general lesson: capacity does not just change the threshold. At low enough capacity, it
changes whether a prediction model is the right intervention at all.

### 6.8 The follow-up questions, and what they were testing

| Follow-up | Her answer | What it tested |
|---|---|---|
| "Why not just enrol everyone?" | 800 discharges and 60 slots; enrolling everyone is not a decision, it is a queue | Whether the capacity constraint was real |
| "How do you know the model works at another hospital?" | External validation on a second site, recalibration if the case mix differs | Whether she distinguishes validation from deployment |
| "What if the navigator disagrees?" | Defer with a reason, escalate to the attending, log the disagreement | Whether the workflow has an exception path |
| "Is this fair?" | The label under-captures out-of-network readmissions, which correlates with churn; prior utilisation under-flags patients with poor access | Whether equity is specific or decorative |
| "What would make you pull it?" | Calibration slope outside 0.8 to 1.2, override rate above 60 percent, subgroup sensitivity below half | Whether monitoring has a rule |
| "How would you know it reduced readmissions?" | Stepped-wedge across units, not a comparison of flagged and unflagged | Whether prediction and intervention were conflated |

The last question is the one that separates candidates. Most answer "compare readmission rates
between flagged and unflagged patients," which is confounded by everything the model used. The
correct answer is a design that varies the intervention, not the prediction.

### 6.9 What she would do with ten minutes of case time

If the case were 15 minutes instead of 45, she would compress to five moves:

1. Decision and actor (1 minute).
2. Capacity and the arithmetic of the threshold (2 minutes).
3. Cohort, time zero, and label limitation (2 minutes).
4. Validation and the one metric the decision needs (2 minutes).
5. Workflow, harm, and stop criterion (3 minutes).

The order is the same; the depth is not. A short case is not a different case, and a candidate who
compresses by dropping the decision will be scored as if the case were long.

### 6.10 The sentence she practised most

"At 60 slots, the threshold is 0.50; that flags 60 patients a month, 52 percent of whom are
readmitted, and it reaches 35 percent of all readmissions."

She practised it until it took 12 seconds, because it contains the whole case in one line: the
constraint, the operating point, the yield, and the coverage. A closing sentence like this is what
the interviewer writes down.

## 7. Code, query, or calculation

The script implements the capacity arithmetic from Section 5, Step 9. It takes a score distribution
with outcome counts per bin, computes the operating characteristics at every cumulative threshold
from the top down, selects the lowest threshold that fits the stated capacity, and shows how the
choice changes as capacity changes.

Assumptions:

- The score bins and counts are synthetic, constructed for teaching. They are not from a real
  deployment.
- The intervention effect of 20 percent relative reduction is an illustrative assumption, not a
  published estimate for any specific programme.
- Capacity is treated as a hard monthly constraint. In practice it varies with staffing, and the
  choice should be re-run when it does.

```python
"""Threshold and capacity analysis, standard library only.

Run: python threshold_capacity.py
Given a score distribution and an outcome count per score bin, computes the
operating characteristics of every cumulative threshold from the top down and
selects the lowest threshold that fits a stated capacity.
"""

# (score bin lower bound, patients per month, readmissions per month)
BINS = [
    (0.0, 310, 10),
    (0.1, 180, 12),
    (0.2, 120, 13),
    (0.3, 80, 12),
    (0.4, 50, 11),
    (0.5, 20, 8),
    (0.6, 20, 9),
    (0.7, 12, 7),
    (0.8, 6, 5),
    (0.9, 2, 2),
]

CAPACITY_PER_MONTH = 60


def cumulative_from_top(bins):
    """Yield thresholds from the highest bin downward with cumulative stats."""
    total_patients = sum(n for _, n, _ in bins)
    total_events = sum(e for _, _, e in bins)
    patients = 0
    events = 0
    rows = []
    for lower, n, e in reversed(bins):
        patients += n
        events += e
        rows.append(
            {
                "threshold": lower,
                "alerts": patients,
                "tp": events,
                "ppv": events / patients,
                "sensitivity": events / total_events,
                "flag_rate": patients / total_patients,
            }
        )
    return rows, total_patients, total_events


def choose(rows, capacity):
    """Lowest threshold whose alert volume does not exceed capacity."""
    eligible = [r for r in rows if r["alerts"] <= capacity]
    if not eligible:
        return None
    return min(eligible, key=lambda r: r["threshold"])


def main():
    rows, total_patients, total_events = cumulative_from_top(BINS)
    prevalence = total_events / total_patients
    print(f"eligible patients per month: {total_patients}")
    print(f"readmissions per month:      {total_events}")
    print(f"prevalence:                  {prevalence:.1%}")
    print(f"capacity:                    {CAPACITY_PER_MONTH}")
    print(f"maximum flag rate:           {CAPACITY_PER_MONTH / total_patients:.1%}")
    print()
    print(f"  {'threshold':>9s} {'alerts':>7s} {'TP':>4s} {'PPV':>7s} {'sens':>7s} {'flag rate':>10s} {'fits':>6s}")
    for r in sorted(rows, key=lambda x: x["threshold"], reverse=True):
        fits = "yes" if r["alerts"] <= CAPACITY_PER_MONTH else "no"
        print(f"  {r['threshold']:9.1f} {r['alerts']:7d} {r['tp']:4d} "
              f"{r['ppv']:7.1%} {r['sensitivity']:7.1%} {r['flag_rate']:10.1%} {fits:>6s}")

    chosen = choose(rows, CAPACITY_PER_MONTH)
    print()
    print("CHOSEN OPERATING POINT")
    print(f"  threshold: {chosen['threshold']:.2f}")
    print(f"  alerts per month: {chosen['alerts']}")
    print(f"  PPV: {chosen['ppv']:.1%}")
    print(f"  sensitivity: {chosen['sensitivity']:.1%}")
    print(f"  readmissions reached: {chosen['tp']} of {total_events}")

    relative_effect = 0.20
    absolute = chosen["ppv"] * relative_effect
    nnt = 1 / absolute if absolute else float("inf")
    prevented = chosen["alerts"] * absolute * 12
    print()
    print("INTERVENTION ARITHMETIC (assuming a 20% relative reduction)")
    print(f"  absolute reduction among enrolled: {absolute:.1%}")
    print(f"  number needed to treat: {nnt:.1f}")
    print(f"  readmissions prevented per year: {prevented:.1f}")

    print()
    print("CAPACITY SENSITIVITY")
    for capacity in (20, 60, 100, 200):
        pick = choose(rows, capacity)
        if pick is None:
            print(f"  capacity {capacity:3d}: no threshold fits")
            continue
        print(f"  capacity {capacity:3d}: threshold {pick['threshold']:.1f}, "
              f"PPV {pick['ppv']:.1%}, sensitivity {pick['sensitivity']:.1%}")


if __name__ == "__main__":
    main()
```

Expected output, confirmed by running the script:

```text
eligible patients per month: 800
readmissions per month:      89
prevalence:                  11.1%
capacity:                    60
maximum flag rate:           7.5%

  threshold  alerts   TP     PPV    sens  flag rate   fits
        0.9       2    2  100.0%    2.2%       0.2%    yes
        0.8       8    7   87.5%    7.9%       1.0%    yes
        0.7      20   14   70.0%   15.7%       2.5%    yes
        0.6      40   23   57.5%   25.8%       5.0%    yes
        0.5      60   31   51.7%   34.8%       7.5%    yes
        0.4     110   42   38.2%   47.2%      13.8%     no
        0.3     190   54   28.4%   60.7%      23.8%     no
        0.2     310   67   21.6%   75.3%      38.8%     no
        0.1     490   79   16.1%   88.8%      61.3%     no
        0.0     800   89   11.1%  100.0%     100.0%     no

CHOSEN OPERATING POINT
  threshold: 0.50
  alerts per month: 60
  PPV: 51.7%
  sensitivity: 34.8%
  readmissions reached: 31 of 89

INTERVENTION ARITHMETIC (assuming a 20% relative reduction)
  absolute reduction among enrolled: 10.3%
  number needed to treat: 9.7
  readmissions prevented per year: 74.4

CAPACITY SENSITIVITY
  capacity  20: threshold 0.7, PPV 70.0%, sensitivity 15.7%
  capacity  60: threshold 0.5, PPV 51.7%, sensitivity 34.8%
  capacity 100: threshold 0.5, PPV 51.7%, sensitivity 34.8%
  capacity 200: threshold 0.3, PPV 28.4%, sensitivity 60.7%
```

Reading the output:

1. **The threshold is the last row that fits capacity.** At 0.50 the cumulative alert count is
   exactly 60. At 0.40 it is 110, which exceeds capacity by 83 percent. The chosen point is
   therefore the lowest threshold that does not overrun the programme.
2. **PPV rises and sensitivity falls as the threshold rises.** This is the trade-off the case is
   about. At 0.70 the programme reaches 16 percent of readmissions with 70 percent yield; at 0.50
   it reaches 35 percent with 52 percent yield.
3. **The number needed to treat follows from PPV and the assumed effect.** At 51.7 percent baseline
   risk and a 20 percent relative reduction, about ten enrolments are needed per readmission
   prevented. The number is sensitive to the effect assumption; halve the effect and the NNT
   doubles.
4. **Capacity sensitivity is discrete.** Capacity 60 and capacity 100 produce the same threshold,
   because the next bin down would add 50 alerts. This is a property of the bin structure, and it
   is a useful reminder that thresholds move in steps, not smoothly.
5. **The output is a negotiating document, not an answer.** The case answer should present the
   trade-off and recommend a point, not simply report the point the script selected. The programme
   owner may prefer a different balance of reach and yield.

To use the script on a real model, replace `BINS` with the observed score distribution and outcome
counts from the validation set, and set `CAPACITY_PER_MONTH` from the programme's staffing. Re-run
after any change in capacity, case mix, or intervention effect.

## 8. Data, standards, and regulatory context

Clinical cases are assessed against real specifications, and an answer that invents a definition
is marked down by any interviewer who knows the measure. This section covers the specifications
most likely to appear: readmission, clinical decision support, and privacy.

### 8.1 The readmission measure

The Centers for Medicare and Medicaid Services (CMS) Hospital Readmissions Reduction Program
(HRRP) uses condition- and procedure-specific 30-day risk-standardised unplanned readmission
measures. Key features, from the CMS measure specifications:

- The readmission must be unplanned and within 30 days of discharge from the index admission, to
  the same or another applicable acute care hospital, regardless of principal diagnosis.
- Planned readmissions are excluded using the CMS Planned Readmission Algorithm, versioned
  annually (version 4.0 for the 2026 report year, according to state specification documents).
- The performance metric is the excess readmission ratio (ERR): the predicted rate for the
  hospital divided by the expected rate for a hospital with similar case mix. An ERR below 1.0 is
  better than expected.
- Hospitals with fewer than 25 eligible discharges for a measure are excluded from that measure's
  contribution.
- Payment adjustment runs from a factor of 0.97 to 1.0, applied to Medicare fee-for-service base
  operating DRG payments, with peer grouping by dual-eligible proportion since FY2019.
- From FY2027, the measures expand to include Medicare Advantage beneficiaries. From 1 October
  2026, the performance period shortens from three years to two.

The hospital-wide all-cause readmission measure is a separate measure in the Hospital Inpatient
Quality Reporting program and is not part of HRRP. Conflating the two is a common error in case
answers.

### 8.2 Clinical decision support and device status

In the United States, whether clinical decision support software is a regulated device depends on
its intended use and on the statutory criteria for the clinical decision support exclusion. The
Food and Drug Administration's guidance on clinical decision support software sets out criteria
including whether the software displays, analyses, or directs the user to a basis for a
recommendation, and whether the user can independently review that basis.

For a case answer, the practical rule is: state the intended use, state who reviews the output,
and say that the regulatory pathway depends on those facts. Do not assert that an alert is "just
analytics" or that it is "definitely a device." Both claims are unsupported without the intended
use and the review workflow.

### 8.3 Privacy and data governance

A readmission model uses admission, discharge, and claims data, which are protected health
information in the United States. The case does not require a legal analysis, but it does require
two sentences: the data are used under the covered entity's treatment, payment, and operations
provisions or an institutional review board approval, and the model output is displayed only to
staff with a treatment relationship. In the European Union, the equivalent statement references
the GDPR lawful basis and the retention schedule.

### 8.4 Equity and the measurement

Two measurement issues recur in readmission work. First, risk-standardised measures adjust for
case mix, and the choice of adjustment variables affects the comparison between hospitals that
serve different populations. Second, social risk factors such as housing instability and insurance
churn are correlated with both the outcome and the likelihood of being observed in the data.

A strong case answer names one mechanism and one check. For example: "Patients who are readmitted
out of network are not observed, and out-of-network care is more likely for patients with unstable
insurance. I would check the subgroup sensitivity at the chosen threshold and compare enrolment
rates by payer."

### 8.5 What the specifications do not settle

They do not tell you which intervention to use, how to allocate scarce slots, or what threshold is
ethical. They define the outcome and the payment consequence. The rest is a clinical and
operational decision, and the case is testing whether you know the difference.

### 8.6 What documentation the case expects

A clinical case answer is often followed by "what would you hand over?" The expected set is small:
an intended-use statement, a data dictionary, a validation report, a model card, and a monitoring
plan. You do not need to produce them in the interview, but naming them shows that you know
deployment is a documentation exercise as much as a modelling one.

The intended-use statement is the one to practise. It is two sentences: who uses the output, for
what decision, in what population, and what it must not be used for. Everything else in the
documentation set derives from it, including the regulatory pathway.

## 9. Industry practice and case evidence

### 9.1 Documented: a widely deployed sepsis model underperformed externally

Wong and colleagues externally validated the Epic Sepsis Model at Michigan Medicine, publishing in
JAMA Internal Medicine in 2021. The study included 27,697 patients and 38,455 hospitalisations
between December 2018 and October 2019, with a sepsis incidence of about 7 percent.

The headline result: hospitalisation-level AUROC was 0.63 (95 percent CI 0.62 to 0.64), compared
with the vendor's reported 0.76 to 0.83. At the alert threshold of 6 or higher, sensitivity was 33
percent, specificity 83 percent, positive predictive value 12 percent, and negative predictive
value 95 percent. Alerts fired on 18 percent of hospitalisations, and the model failed to identify
67 percent of sepsis patients. The number needed to evaluate was 8 at the hospitalisation level.

The authors also showed that including scores up to three hours after sepsis onset raised the AUC
to 0.80, illustrating how a timing assumption can inflate apparent performance. A correction was
issued in August 2021 affecting Table 2 and the number-needed-to-evaluate figure.

This is the single most useful public case for a clinical case interview. It demonstrates, with
numbers, why external validation, alert burden, and PPV at the operating threshold matter more than
a vendor's discrimination estimate.

### 9.2 Documented: alert override rates are high, and most overrides are appropriate

A prospective study in six Brigham and Women's Hospital intensive care units (Wong et al., BMJ
Quality and Safety 2018) examined 24,231 medication-related alerts. The overall override rate was
88.5 percent, ranging from 2.3 percent for geriatric alerts to 97.1 percent for renal alerts.
Among 2,448 overridden alerts reviewed, 81.6 percent of overrides were judged appropriate, and
inappropriate overrides were associated with a roughly six-fold higher risk of an adverse drug
event (OR 6.14, 95 percent CI 4.63 to 7.71) and about 2.25 additional days in intensive care.

A separate study of passive EHR alerts in an intensive care unit (Kizzier-Carnahan et al., Journal
of Patient Safety) reported a mean of 64.1 alerts per patient per day, of which only 4.5 percent
were panic values.

The lesson for a case answer is not that alerts are useless. It is that a high override rate is
expected, that most overrides can be clinically correct, and that the relevant measure is whether
the alerts that matter are acted on. A monitoring plan based on override rate alone will
misclassify a well-designed alert as failing.

### 9.3 What the sepsis case means for a candidate

The sepsis model case is useful in an interview for three reasons. First, it is public and
specific, so citing it demonstrates that you read the literature rather than repeating a
principle. Second, it separates vendor-reported performance from local performance with numbers:
0.76 to 0.83 reported, 0.63 observed, 33 percent sensitivity at the alert threshold, 12 percent
PPV. Third, it shows why alert burden matters: alerts fired on 18 percent of hospitalisations,
which is the mechanism that produces fatigue.

Use it carefully. The study is from one site and one period, and the vendor disputed some
interpretations. Say "in one external validation" rather than "the model does not work," and
connect it to the design decision you would make: local validation before deployment, and a
silent-mode period before alerts.

### 9.4 Inference, labelled

- **Inference:** in a clinical case interview, the interviewer weights the decision and workflow
  layers more heavily than the model layer, because that is where the public failures occur.
- **Inference:** candidates who volunteer the capacity arithmetic are read as having deployed
  something, because the arithmetic is only learned in operations.
- **Inference:** naming a specific public failure, such as the sepsis model validation, is more
  persuasive than describing a generic validation principle, because it shows the principle was
  applied to a real case.
- **Inference:** the strongest closing move is a stop criterion, because it signals that you
  expect the model to be wrong eventually and have planned for it.

> **Note:** The Epic Sepsis Model results are from one external site and one period. They do not
> establish that the model fails everywhere, and the vendor disputed some interpretations. They do
> establish that a vendor-reported AUC is not evidence of local performance.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Model-first answer | The first sentence names an algorithm | Training rewards methods | Interviewer stops listening for framing | Practise a decision-first opening until it is automatic |
| No actor | You cannot name who acts on the output | The prompt is abstract and you accepted it | Case reads as academic | Ask "who does what differently" in the first minute |
| No capacity arithmetic | You propose a threshold without alerts per week | Thresholds taught as statistical choices | Deployment fails on workload | Compute alerts, PPV, and sensitivity at the operating point |
| Prediction confused with intervention | You compare flagged with unflagged patients | The two questions sound similar | Overstated effect and a flawed evaluation | Say explicitly that the programme, not the model, changes outcomes |
| Label taken as truth | You never mention what the outcome misses | Codes feel objective | Bias hidden and equity claims unsupported | Name three label limitations and their direction |
| Ignoring prevalence | You quote PPV from another setting | PPV depends on prevalence | The yield at your site is worse | Recompute at the local prevalence |
| No human factors | The alert goes "to the care team" | Workflow treated as someone else's problem | No one acts, or everyone is interrupted | Name role, channel, timing, and exception path |
| Equity as a slogan | "We would check for bias" | Fear of saying the wrong thing | No specific mechanism or check | Name one mechanism and one subgroup metric |
| No stop criterion | Monitoring described as "ongoing review" | Deployment treated as an endpoint | A degrading model stays live | Pre-specify numerical pause conditions |
| Over-scoping | You propose a multi-year research programme | Ambition mistaken for rigor | The interviewer cannot see a deliverable | Bound the case to one decision and one site |
| Under-scoping | You propose a dashboard and stop | Caution mistaken for humility | No evidence of modeling judgment | Include one falsifiable claim and a validation design |
| Defensiveness under challenge | You repeat the original design | Challenge read as attack | Low coachability score | Ask what concern the challenge reflects, then adjust |
| Ignoring regulation | You claim the model is "not a device" | Uncertainty avoided | Regulatory risk unaddressed | State that intended use determines the pathway |
| No baseline | You never ask what happens today | The model feels like the starting point | Improvement cannot be measured | Ask for the current rule and its yield |
| Arithmetic errors | The alert count exceeds capacity by a factor of four | Mental maths skipped | The design is infeasible | Compute flag rate before proposing a threshold |
| Recommendation avoided | The case ends with "it depends" | Fear of committing | Interviewer cannot assess judgment | Commit to a conditional recommendation |

Two red flags are about the case itself. A prompt with no decision and no data, where the
interviewer refuses to clarify anything, is usually a stress test rather than a realistic case;
answer the structure anyway and state your assumptions. A case that changes definitions mid-way
without acknowledging it is testing composure; note the change and restate the chain.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The case has a decision behind it | The whole structure depends on it | Some prompts are exploratory | Ask who acts; if no one, say so |
| Capacity is the binding constraint | It determines the threshold | The intervention may be scalable | Ask what limits enrolment |
| The label is a proxy | It justifies the limitations section | Some outcomes are measured directly | Ask how the outcome is captured |
| The local prevalence is known | PPV depends on it | New programmes lack baseline data | Ask for the current rate or bound it |
| A 20 percent relative effect is plausible | It enables the NNT arithmetic | Effects vary widely by programme | Label it as an assumption and test sensitivity |
| The model will be used with human review | It shapes workflow and regulation | Some deployments are automated | Ask who reviews the output |
| Subgroup data are available | It enables equity checks | Small subgroups have wide intervals | Report n and intervals, not just points |
| The interviewer wants a recommendation | It justifies committing | Some cases test exploration only | Offer a recommendation with conditions |
| The arithmetic can be approximate | It enables mental calculation | The interviewer expects precision | State the rounding and the direction of error |
| The interviewer knows the local context | It justifies asking about baseline and capacity | Some cases are generic by design | Ask, and proceed with a stated assumption if the answer is "unknown" |

The largest limit is that the case rewards a structure that may not fit every employer. Some
health systems want a research proposal; some vendors want a product plan. Ask at the start what
decision the case is meant to support, and adjust the depth rather than the order. The spine holds;
the emphasis changes.

## 12. Exercises

**Exercise 1. Decision-first opening (application, 20 minutes).**
Deliverable: three different openings for the readmission prompt, each under 45 seconds and each
naming a decision, an actor, and a consequence.
Acceptance criteria: no method is mentioned; each opening ends with a clarifying question; a
listener can repeat the decision back.

**Exercise 2. Definitions (application, 30 minutes).**
Deliverable: population, index event, time zero, outcome, and exclusions for one prompt, each with
one exclusion and one consequence.
Acceptance criteria: all five elements present; each choice states its effect on the denominator
or the label; no element is left to the interviewer.

**Exercise 3. Label audit (critique, 30 minutes).**
Deliverable: three label limitations with the direction of bias and one check for each.
Acceptance criteria: each limitation is specific to the outcome; the direction is stated; at least
one limitation relates to equity.

**Exercise 4. Threshold arithmetic (application, 45 minutes).**
Deliverable: a hand calculation of alerts per month, PPV, sensitivity, and NNT at a stated
capacity, checked against the Section 7 script.
Acceptance criteria: the arithmetic is reproducible; the capacity constraint is explicit; the NNT
assumption is labelled; the result is within 5 percent of the script's output.

**Exercise 5. Capacity variation (application, 30 minutes).**
Deliverable: the same case answered at capacities of 20, 60, and 200 per month, with the
recommendation for each.
Acceptance criteria: the threshold changes with capacity; the recommendation changes at least
once; the trade-off is stated in one sentence per capacity.

**Exercise 6. Workflow map (design, 45 minutes).**
Deliverable: a one-page workflow with actor, channel, timing, actions, exception path, and
fallback.
Acceptance criteria: every action has an owner; the override path is explicit; the fallback when
the model is unavailable is stated.

**Exercise 7. Equity review (critique, 30 minutes).**
Deliverable: two equity mechanisms specific to the outcome, with a metric and a review cadence for
each.
Acceptance criteria: mechanisms are causal, not generic; metrics are computable from available
data; at least one mechanism concerns who is observed in the data.

**Exercise 8. Stop criteria (design, 20 minutes).**
Deliverable: three numerical stop or pause conditions with a measurement frequency.
Acceptance criteria: each condition is testable; each has a threshold and a window; at least one
concerns subgroup performance.

## 13. Self-check answers

**1. What are the first three things to say in a clinical case?**
The decision, the actor, and the moment. Then the capacity constraint, because it determines the
threshold. The method comes after.

**2. Define the readmission case in five elements.**
Adults discharged alive from medical-surgical units, index event is discharge, time zero is 24
hours before discharge, outcome is unplanned all-cause readmission within 30 days excluding
planned, exclusions include transfers and deaths.

**3. Name three label limitations and their direction.**
Out-of-network readmissions are unobserved, understating risk for mobile or churned patients;
planned-readmission flags are incomplete for some payers, overstating the unplanned rate there;
coding variation across sites misclassifies the cohort differentially.

**4. How do you choose a threshold?**
Compute the cumulative flag rate at each candidate threshold; choose the lowest threshold whose
alert volume fits capacity; then adjust for harm asymmetry. Report PPV, sensitivity, and NNT at
the chosen point.

**5. At 800 eligible per month, 60 slots, and 89 readmissions, what does the chosen threshold
produce?**
Threshold 0.50, 60 alerts, PPV 51.7 percent, sensitivity 34.8 percent, 31 of 89 readmissions
reached. With a 20 percent relative effect, the absolute reduction is 10.3 percentage points, NNT
is 9.7, and about 74 readmissions are prevented per year.

**6. What is the workflow for the output?**
A worklist delivered to the discharge huddle 24 hours before expected discharge; the navigator
enrols, defers with a reason, or declines with a reason; disagreement escalates to the attending;
if the model is unavailable, the current informal review continues.

**7. Name two equity mechanisms.**
Out-of-network readmissions are unobserved and correlate with insurance churn; prior utilisation
features under-flag patients with limited access. Checks: subgroup sensitivity at the threshold,
and enrolment rates by payer.

**8. What would make you stop the model?**
Calibration slope outside 0.8 to 1.2 for two consecutive months; override rate above 60 percent for
two months; a major subgroup's sensitivity below half the overall rate.

**Rubric for Exercise 3.** Full marks: three limitations specific to the outcome; direction of
bias stated for each; one check per limitation that could be computed from available data; at least
one limitation linked to equity.

**Rubric for Exercise 4.** Full marks: capacity converted to a maximum flag rate; cumulative
counts computed from the top; PPV and sensitivity computed at the chosen threshold; NNT derived
from PPV and a labelled effect assumption; the result matches the script.

**Rubric for Exercise 6.** Full marks: every step has an owner and a timing; the exception path is
explicit; the fallback is stated; the alert volume per clinician per shift is computed.

## 14. Interview questions

**1. "How would you reduce readmissions here?"**
Strong: decision, actor, capacity, label, validation, threshold, workflow, stop rule. Weak: a model
and a metric.

**2. "Why not just use a vendor model?"**
Strong: external validation, local calibration, PPV at the operating point, alert burden, and
contract terms for monitoring. Weak: says vendor models are bad.

**3. "What is the difference between AUROC and PPV, and which matters here?"**
Strong: ranking versus yield at a threshold, prevalence dependence, capacity. Weak: says AUROC is
the standard metric.

**4. "How would you evaluate whether the programme worked?"**
Strong: a design that varies the intervention (stepped-wedge or randomised pilot), not a comparison
of flagged and unflagged patients. Weak: before-after comparison with no control.

**5. "What would you do if the model's calibration drifted?"**
Strong: detect via monitoring, recalibrate on recent data, re-evaluate the threshold, pause if
needed. Weak: retrain immediately without diagnosis.

**6. "Who is harmed by a false positive?"**
Strong: names the patient who receives unnecessary intervention and the patient displaced from a
scarce slot, plus the staff time. Weak: says false positives are harmless.

**7. "How do you handle missing data at prediction time?"**
Strong: mechanism, availability at time zero, imputation with missingness indicators, sensitivity
analysis. Weak: impute the mean.

**8. "Is this a regulated device?"**
Strong: intended use, user review of the basis, and a statement that the pathway depends on those
facts. Weak: asserts yes or no without the intended use.

**9. "What would make you recommend against deploying?"**
Strong: a numerical condition tied to calibration, subgroup performance, or capacity. Weak: says
they would monitor it.

**10. "What did you miss in this case?"**
Strong: names a specific gap, such as cost or a stakeholder, and says how they would address it.
Weak: says nothing was missed.

## 15. Key terms

- **Alert burden** — alerts per clinician per shift or per patient-day.
- **Alert fatigue** — desensitisation from excessive alerts.
- **Capacity** — actions the system can deliver per period.
- **Care gap** — difference between recommended and delivered care.
- **Clinical decision support (CDS)** — software providing patient-specific assessments or
  recommendations.
- **Escalation** — the path when a clinician disagrees with the output.
- **Excess readmission ratio (ERR)** — predicted divided by expected rate under the CMS measure.
- **Human in the loop** — a clinician reviews or approves the output.
- **Index admission** — the admission that starts the follow-up window.
- **Intended use** — the population, user, and decision a model supports.
- **Label** — the outcome definition used to train and evaluate.
- **Number needed to treat (NNT)** — patients who must receive an intervention to prevent one event.
- **Override rate** — share of alerts dismissed without the recommended action.
- **Positive predictive value (PPV)** — share of flagged patients who have the outcome.
- **Risk-standardised rate** — rate adjusted for case mix under a specified model.
- **Sensitivity** — share of outcome cases that are flagged.
- **Shadow mode** — running a model without showing its output.
- **Social risk** — social and economic factors affecting outcomes.
- **Stepped-wedge** — trial design with sequential site crossover.
- **Stop criterion** — pre-specified condition for pausing or withdrawing.
- **Subgroup harm** — differential error across groups.
- **Time zero** — the moment of prediction.

## 16. Sources

All URLs accessed 26 October 2026.

1. Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
   sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*
   2021;181(8):1065–1070. https://doi.org/10.1001/jamainternmed.2021.2626
2. Habib AR, Lin AL, Grant RW. Importance of external validation in sepsis. *JAMA Internal
   Medicine* 2021. (Accompanying editorial.)
3. Wong A, Amato MG, Seger DL, et al. Prospective evaluation of medication-related clinical
   decision support over-rides in the intensive care unit. *BMJ Quality & Safety* 2018.
   https://doi.org/10.1136/bmjqs-2017-007531
4. Kizzier-Carnahan V, et al. Frequency of passive EHR alerts in the ICU: another form of alert
   fatigue? *Journal of Patient Safety*. https://pmc.ncbi.nlm.nih.gov/articles/PMC5820226/
5. Centers for Medicare & Medicaid Services. *Hospital Readmissions Reduction Program.*
   https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/hospital-readmissions-reduction-program-hrrp
6. QualityNet. *Hospital Readmissions Reduction Program measures.*
   https://qualitynet.cms.gov/inpatient/hrrp/measures
7. Minnesota Department of Health. *HRRP Excess Readmission Rate Specifications, 2026 Report
   Year.* https://www.health.mn.gov/data/hcquality/measures/docs/err26.pdf
8. U.S. Food and Drug Administration. *Clinical Decision Support Software: Guidance for Industry
   and Food and Drug Administration Staff.*
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
9. U.S. Department of Health and Human Services. *HIPAA Privacy Rule.*
   https://www.hhs.gov/hipaa/
10. Agency for Healthcare Research and Quality. *Clinical Decision Support.*
    https://www.ahrq.gov/cpi/about/otherwebsites/clinical-decision-support/index.html
11. HealthIT.gov. *Clinical Decision Support and EHR integration.*
    https://www.healthit.gov/topic/safety/clinical-decision-support
12. Institute for Healthcare Improvement. *Transitional care and readmission reduction.*
    https://www.ihi.org/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_technical-interviews.md`](05_technical-interviews.md)
- Next topic: [`07_system-design.md`](07_system-design.md)
- Related topic: [`02_portfolio.md`](02_portfolio.md)
- Survey file: [`../../course/20_career_portfolio_interviews.md`](../../course/20_career_portfolio_interviews.md)
- Labs: [`../../labs/`](../../labs/)
