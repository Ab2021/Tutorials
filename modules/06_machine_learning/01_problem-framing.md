# Problem Framing

> **Module 06 — Machine Learning Foundations**
> Topic 01 of 13
> Estimated study time: 3 hours
> Prerequisites: Module 04
> Version 2026.3

## 1. Why this topic exists

Most machine-learning projects in biomedical organisations do not fail because the algorithm is
weak. They fail because the project was never a decision problem in the first place. A team is
asked to "build a model for readmissions", spends four months on feature engineering, reports an
area under the receiver operating characteristic curve (AUROC) of 0.78, and then discovers that
nobody has agreed who will act on the output, at what threshold, with what capacity, or what
counts as success. The model is technically adequate and organisationally inert.

The cost of that failure is not only the engineering time. It is the opportunity cost of the
intervention that was never targeted properly, the clinicians who lose trust in the next model
because the first one produced unusable alerts, and the regulatory work that has to be redone
when the intended use is finally clarified and turns out to require a different validation study.
Problem framing is the cheapest part of the lifecycle and the part that determines whether every
later part is worth doing.

Three patterns recur across biomedical settings.

**The prediction has no action attached.** A model estimates the probability of a rare event, but
no pathway exists to change that probability. A 30-day readmission risk score shown to a discharge
planner who has no additional home-care capacity does not reduce readmissions. The output is
information without an actuator. Before modelling, the team must be able to name the action, the
person who takes it, and the mechanism by which the action changes the outcome.

**The action has no capacity.** A sepsis alert that fires on 18 percent of admissions will be
ignored on a ward that can review four patients per shift. The model may be accurate and the
workflow may be sound, but the operating point is impossible. Capacity is a first-class constraint
on the prediction task, not an implementation detail to be solved after validation.

**The metric measures the model, not the decision.** AUROC is a property of the ranking. The
decision depends on the probability scale (calibration), the threshold, the prevalence in the
deployed population, and the relative harm of the two error types. A team can improve AUROC while
making the decision worse, for example by improving discrimination in a region of the risk
distribution where nobody acts. Framing forces the team to write down the decision rule before
seeing model results, which is the only reliable defence against choosing a metric after the fact
because it flatters the model.

Regulators have converged on the same point from a different direction. The United States Food and
Drug Administration (FDA), Health Canada, and the United Kingdom Medicines and Healthcare products
Regulatory Agency (MHRA) joint guidance on good machine learning practice (GMLP) begins with the
need to define the intended use and the clinical workflow before data collection. The TRIPOD+AI
statement (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or
Diagnosis, AI extension, 2024) requires the intended use, target population, prediction horizon,
and planned model output to be reported before results. DECIDE-AI (Developmental and Exploratory
Clinical Investigations of DEcision support systems driven by Artificial Intelligence) adds the
human-in-the-loop workflow as a reporting item. These are reporting conventions, not statutes, but
they encode the same operational lesson: a prediction model is a component of a decision system,
and describing it without the decision system is incomplete.

This topic teaches the framing step as a procedure with artefacts. The output is a one-page
prediction task specification that a clinician, an engineer, a biostatistician, and a compliance
lead can all sign. Everything later in the module assumes that specification exists. If it does
not, the validation design, the metric choice, the threshold, and the monitoring plan have no
anchor.

> **Note:** Framing is not a one-time meeting. It is revisited when prevalence shifts, when the
> intervention changes, or when the deployment site changes. A specification that is not versioned
> is not a specification.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a prediction task from a decision problem, and state the action, actor, and
   mechanism that connect them.
2. **Write** a prediction task specification containing time zero, horizon, eligible population,
   label definition, feature availability window, output scale, action threshold, and capacity.
3. **Derive** an action threshold from an explicit benefit and harm ratio, and **explain** why the
   threshold is not 0.5 by default.
4. **Compute** the capacity implied by a threshold and the threshold implied by a capacity, and
   **identify** when the two are inconsistent.
5. **Map** a proposed task to the reporting items in TRIPOD+AI and the workflow items in DECIDE-AI,
   and **name** which items cannot yet be completed.
6. **Critique** a project brief that specifies a model but not a decision, using a structured
   checklist.
7. **Design** a pre-registration note that fixes the primary metric, threshold rule, and subgroup
   analyses before model development begins.
8. **Recognise** framing failures in published case studies, including label proxy choices and
   alert-burden mismatches.

Each outcome is assessed by at least one exercise in Section 12 or one self-check item in Section
13. Outcomes 3, 4, and 7 are the ones most often missing from junior portfolios and the ones most
probed in interviews.

## 3. The core mental model

The model is one box in a chain that starts with a decision and ends with a monitored outcome.
Every arrow in the chain is a place where the project can fail, and most of those failures are
decided before any model is fitted.

```text
   Decision          Target            Label             Features            Split
   (who acts,   ->   (population,  ->  (definition,  ->  (available at   ->  (matches
    what action)      time zero,        source,           time zero)          deployment)
                      horizon)          error structure)
      |                 |                  |                 |                  |
      v                 v                  v                 v                  v
   Threshold   <-    Calibration   <-    Model      <-    Preprocessing  <-   Validation
   (harm ratio,      (does p mean        (family,          (fit inside        (nested,
    capacity)         what it says)       complexity)       folds)             grouped)
      |
      v
   Utility  ->  Deployment  ->  Monitoring  ->  Revisit framing
   (net benefit)  (workflow)     (drift, harm)
```

The chain is read left to right as design, and right to left as evidence. Design asks: given the
decision, what must the model output, and what must be true for that output to change the action?
Evidence asks: given the validation results, which claims about the decision are supported?

Three properties of the chain matter more than the rest.

**Time zero is the load-bearing element.** Time zero is the instant at which the prediction is
made and the feature set is frozen. Every feature must be observable at or before time zero, and
the label must be ascertained strictly after it. If the team cannot state time zero in one
sentence, with a clinical event and a timestamp, they cannot prevent temporal leakage. "At
discharge" is not a time zero; "at the moment the discharge order is signed" is. "On admission"
is not a time zero for a 48-hour deterioration model unless the model is recomputed continuously
and the team says so.

**The decision rule is part of the task, not an afterthought.** A risk model plus a threshold plus
a capacity defines the action. The threshold comes from the relative harm of a false positive and
a false negative, adjusted for prevalence and constrained by capacity. If the team cannot state the
harm ratio, they cannot choose a threshold, and any threshold they report is arbitrary. This does
not mean the threshold must be fixed forever; it means the rule for setting it must be fixed
before evaluation.

**The deployment claim determines the validation design.** A claim that the model "stratifies risk
in the development health system" needs internal validation. A claim that it "improves outcomes at
a new site" needs external validation, calibration assessment, and ideally a prospective
evaluation. A claim that it "reduces readmissions" needs a trial or a strong quasi-experimental
design, because a prediction alone does not change outcomes. The claim must be written in the
specification, in the same language that will appear in the model card and the regulatory
submission.

A useful discipline is to write the specification as a set of sentences that must each be true.

```text
At [time zero], for [eligible population], we predict [outcome] within [horizon].
The output is a probability. A patient is enrolled in [action] when predicted risk exceeds
[threshold rule]. The action is delivered by [actor] with capacity [n per period].
Success means [net benefit / outcome metric] improves relative to [comparator] without
increasing [harm] beyond [limit], including in [subgroups].
```

If any bracket cannot be filled with a specific value or a named owner, the project is not ready
for model development. It may still be ready for a feasibility study, which is a different
deliverable with a different design.

> **Note:** The chain is drawn as linear for teaching. In practice, framing iterates: a capacity
> constraint may force a narrower population, which changes prevalence, which changes the
> threshold. Record the iteration and the reason, because the final specification will otherwise
> look arbitrary to a reviewer.

## 4. Definitions and vocabulary

The vocabulary below is used throughout the module. Definitions are operational: they describe
what the term means when it appears in a specification or a validation report, not its broadest
statistical sense.

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Decision | A choice between actions that differ in outcome or cost | Calling any prediction a decision | Categorical action set |
| Prediction task | A mapping from information available at time zero to a probability or score | Treating the task as the model | Structured specification |
| Time zero | The instant the prediction is made and features are frozen | "On admission" for a model updated hourly | Timestamp with source |
| Prediction horizon | The interval from time zero to the outcome ascertainment boundary | Confusing horizon with follow-up duration | Days, hours, or visits |
| Target population | The set of patients for whom the prediction is intended and validated | The set present in the database | Inclusion and exclusion criteria |
| Eligible population | Target population minus those for whom the action is contraindicated or unavailable | Using eligibility as a modelling convenience | Count and flow diagram |
| Label | The observed outcome used as the supervised target | Assuming the label is the clinical truth | Definition plus source code |
| Proxy label | A measured variable used in place of the outcome of interest | Presenting a proxy as the outcome | Named proxy plus validation evidence |
| Feature | A variable available at or before time zero | Any column in the table | Name, type, window, source |
| Action | The intervention triggered by a positive prediction | The model output itself | Named workflow step |
| Actor | The person or system that executes the action | "The care team" | Role and team |
| Capacity | The number of actions the system can deliver per period | Ignored until deployment | Patients per week |
| Threshold | The predicted risk above which the action is taken | 0.5 by convention | Probability |
| Harm ratio | The cost of a false positive relative to the benefit of a true positive | Assumed to be 1 | Dimensionless ratio |
| Net benefit | True positive rate weighted by prevalence minus false positive rate weighted by the odds of the threshold | Confused with profit | Per patient, dimensionless |
| Utility | The expected value of the decision rule under a stated utility model | Treated as intrinsic to the model | Currency or quality-adjusted units |
| Calibration | Agreement between predicted probabilities and observed frequencies | Replaced by discrimination | Slope, intercept, curve |
| Discrimination | Ability to rank positives above negatives | Treated as sufficient | AUROC, AUPRC |
| AUROC | Area under the receiver operating characteristic curve; probability a random positive ranks above a random negative | Reported without prevalence or calibration | 0.5 to 1.0 |
| AUPRC | Area under the precision-recall curve; sensitive to prevalence | Compared across different prevalences | 0 to 1 |
| Prevalence | Proportion of the eligible population with the label in the deployment period | Taken from the development cohort without checking | Proportion |
| PPV | Positive predictive value; proportion of flagged patients who have the outcome | Assumed constant across sites | Proportion |
| NPV | Negative predictive value; proportion of unflagged patients without the outcome | Assumed constant across sites | Proportion |
| Sensitivity | Proportion of positives flagged | Called "recall" in some literatures | Proportion |
| Specificity | Proportion of negatives not flagged | Ignored when capacity binds | Proportion |
| Alert burden | Alerts generated per unit of clinical work | Measured only after launch | Alerts per 100 admissions |
| Intended use | Regulated statement of what the device is for, in whom, and by whom | Written by regulatory affairs without the clinical team | Free text with claims |
| Deployment claim | The performance or outcome claim the team intends to defend | Broader than the evidence supports | Written claim plus evidence |
| External validation | Evaluation in data not used for development, from a different site or period | Called external when it is a random split | Study design label |
| Prospective evaluation | Evaluation after deployment, with the model live in workflow | Confused with a silent trial | Study design label |
| Silent trial | Prospective scoring without exposing the output to users | Called a prospective trial | Study design label |
| Model card | Structured documentation of intended use, data, performance, and limits | A marketing one-pager | Document |
| Predetermined change control plan | Pre-authorised specification of model changes and their validation | Treated as permission for unconstrained learning | Regulatory document |
| Decision curve analysis | Net benefit plotted across threshold probabilities | Read as a discrimination curve | Curve plus prevalence |
| Capacity-constrained threshold | The threshold that flags exactly the number of patients the system can handle | Derived after seeing scores without disclosure | Probability plus count |
| Time-zero leakage | Use of information created after time zero | Detected only by inspection | Audit finding |
| Label leakage | Presence of the outcome or its cause among features | Confused with class imbalance | Audit finding |
| Base rate | Synonym for prevalence in the deployment population | Assumed stationary | Proportion |
| Case mix | Distribution of severity and comorbidity in the population | Ignored when comparing sites | Description plus measure |
| Actionability | Degree to which the action changes the outcome for a flagged patient | Assumed for any plausible action | Evidence grade |
| Number needed to treat | Patients who must receive the action for one to benefit | Reported without the action's effect size | Count |
| Opportunity cost | Value forgone when capacity is used on one patient rather than another | Omitted from harm calculations | Currency |

Two definitions deserve more than a table row.

**Intended use versus deployment claim.** Intended use is the regulated description of the
device's purpose. The deployment claim is what the team asserts about performance in a specific
setting. They must be consistent, but they are not identical: a device may have a broad intended
use and a narrow, well-evidenced claim for one population. Confusing the two produces submissions
that overclaim and model cards that underinform. Write both, and write the evidence that supports
each.

**Utility versus net benefit.** Utility is the expected value of a decision under a model of
benefits and harms. Net benefit is a specific, transparent utility model used in decision curve
analysis, in which the harm of a false positive is expressed relative to the benefit of a true
positive through the threshold probability. Net benefit is useful because it requires only one
assumption (the harm ratio) and can be compared with treat-all and treat-none strategies. It is
not a complete health economic evaluation: it ignores costs that accrue after the decision, and it
assumes the benefit of a true positive is constant across patients. State that when reporting it.

> **Note:** Every metric in this module is reported with its prevalence context. An AUROC of 0.78
> at 3 percent prevalence and an AUROC of 0.78 at 30 percent prevalence describe different
> clinical situations even though the number is identical, because PPV, net benefit, and capacity
> all differ.

## 5. The mechanics, step by step

Framing is a procedure with fourteen steps and two artefacts: the prediction task specification
and the pre-registration note. The procedure is written for a working session of 90 to 120 minutes
with the clinical owner, the data lead, and the evaluation lead in the room. Steps 1 through 3 can
be completed before any data are extracted. Steps 4 through 9 require a data dictionary. Steps 10
through 12 require a small amount of arithmetic and, ideally, pilot data. Steps 13 and 14 are
sign-off.

Each step states its input, its output, and the reason it cannot be skipped.

### Step 1. Name the decision and the decision owner

**Input:** a project brief, a complaint from operations, or a clinical guideline gap.

**Output:** one sentence of the form "On [day or event], [role] decides whether to [action] for
[patient group]." The role is a named individual or a named committee, not "the team".

**Rationale:** A decision without an owner is not a decision; it is a recommendation that will be
ignored at the point of care. Naming the owner also identifies who can supply the capacity number
and who will be accountable when the model is wrong. In regulated settings the owner is often the
"intended user" in the labelling, and the FDA's intended-use language requires the user to be
specified.

**Branch point:** if more than one role acts on the output, the task may need more than one
specification. A score that triggers both a nurse call and a physician review has two actions,
two capacities, and possibly two thresholds. Split it.

### Step 2. Specify the action and its mechanism

**Input:** the decision sentence from Step 1.

**Output:** a description of the action, its dose or intensity, its timing relative to time zero,
and the causal mechanism by which it is expected to change the outcome. One paragraph.

**Rationale:** Prediction is only useful if the action changes the probability of the outcome. If
the mechanism is implausible, no amount of model accuracy will produce a benefit. The mechanism
also determines which patients are eligible: an intervention that requires a home visit cannot be
delivered to patients without a fixed address.

**Branch point:** if the action is "review the patient", the mechanism is clinician attention, and
the evidence base is different from a pharmacological or device intervention. Attention-based
actions are real but have a capacity curve that saturates quickly; flag them for the capacity step.

### Step 3. Grade the evidence for the action

**Input:** the mechanism paragraph, plus literature or local pilot data.

**Output:** an effect estimate with a range, a source, and an evidence grade (trial, observational,
expert opinion). If no estimate exists, an explicit assumption with a sensitivity range.

**Rationale:** The threshold calculation in Step 10 needs the benefit of a true positive. That
benefit is the action's absolute risk reduction multiplied by the value of the avoided outcome,
minus the cost of delivering the action. An intervention with a 5 percent relative reduction and
one with a 30 percent relative reduction imply very different thresholds.

**Branch point:** if the evidence is absent, the project can still proceed as an exploratory
study, but the specification must label the benefit as an assumption and the pre-registration note
must include a sensitivity analysis over its plausible range. Do not present a threshold derived
from an assumed effect as if it were evidence-based.

### Step 4. Define the eligible population

**Input:** the clinical pathway, the action's contraindications, and the available data.

**Output:** inclusion and exclusion criteria with counts, plus a flow diagram from all discharges
to the eligible cohort.

**Rationale:** The target population determines prevalence, case mix, and the applicable
regulatory claim. It also determines whether the model can be deployed at all: a model developed
on adults with complete laboratory data may not be applicable to the 30 percent of admissions
transferred from outside facilities with missing labs.

**Branch point:** if exclusions remove more than roughly 20 percent of the candidate population,
check whether they are informative. Excluding deaths and hospice discharges from a readmission
model is standard, but excluding patients with prior readmissions may remove the highest-risk
group and change the decision entirely. Report the excluded fraction and the reason.

### Step 5. Fix time zero

**Input:** the clinical workflow and the action's timing.

**Output:** a named event with a timestamp field in the source system, plus the feature cut-off
rule (for example, "all data with a timestamp strictly before the discharge order signature").

**Rationale:** Time zero is the boundary that separates prediction from leakage. Every feature
must be available before it, and every label must be ascertained after it. A specification that
says "at discharge" without naming the event leaves the boundary ambiguous, and ambiguous
boundaries are where leakage enters.

**Branch point:** if the model is intended to update over time (for example, hourly sepsis risk),
there is not one time zero but a series of prediction times. The specification must state the
update cadence, the minimum data available at each update, and how the evaluation handles
correlated predictions from the same admission. That is a different design from a single-shot
discharge model and should be labelled as such.

### Step 6. Set the prediction horizon

**Input:** the action's mechanism and the outcome's natural history.

**Output:** a horizon in days or hours, with the ascertainment rule at the boundary (for example,
"readmission within 30 days of discharge, where day 0 is the discharge date").

**Rationale:** The horizon determines the label, the prevalence, and the clinical meaning of a
positive. A 30-day readmission and a 90-day readmission are different outcomes with different
actionability. The horizon also determines the observation window: a patient discharged on the
last day of the data extract has incomplete follow-up and must be excluded or censored.

**Branch point:** if the horizon is longer than the follow-up available for a meaningful fraction
of the cohort, the label is systematically missing for recent patients. Shorten the horizon,
restrict the cohort to discharges with complete follow-up, or model the censoring explicitly.

### Step 7. Define the label and its ascertainment

**Input:** the outcome of interest, the available coding systems, and the clinical record.

**Output:** a computable label definition with source table, code set, inclusion and exclusion
rules, and a validation sample against chart review.

**Rationale:** The label is the supervision signal, and its error structure propagates into every
metric. A diagnosis-code label misses readmissions to outside hospitals; a lab-threshold label
captures testing intensity as much as disease; a chart-review label is accurate but expensive and
may not scale. The specification must state which error direction is likely and how it was
checked.

**Branch point:** if the label is a proxy (cost, utilisation, clinician concern), the specification
must state the construct it stands for and the evidence that the proxy tracks it. The Obermeyer
study is the canonical example of a proxy that tracked cost rather than need, with racial
consequences in a care-management programme.

### Step 8. Inventory features available at time zero

**Input:** the data dictionary and the source-system timestamps.

**Output:** a feature table with name, type, source, lookback window, missingness rate, and
availability at time zero (yes, no, conditional).

**Rationale:** Feature availability is a framing constraint, not a modelling detail. A feature
that is only available for patients who received a test creates a selection mechanism that the
model will learn. A feature that is recorded after time zero is leakage. Listing features before
modelling prevents the common pattern of discovering at review that the strongest predictor is a
post-discharge note.

**Branch point:** if a feature is available only for a subgroup (for example, echocardiogram for
patients who received one), decide explicitly whether to include it with a missingness indicator,
exclude it, or restrict the population. Each choice changes the deployment claim.

### Step 9. Specify the output and its scale

**Input:** the decision rule and the action.

**Output:** a statement of whether the model outputs a probability, a score, a class, or a
ranking, and the scale on which the threshold is defined.

**Rationale:** A threshold of 0.24 is meaningless for a score that ranges from 0 to 100 or for a
ranking without a probability scale. If the decision requires a probability (for example, for
shared decision-making), the model must be calibrated. If the decision requires only a ranking
(for example, allocating a fixed number of slots), calibration is less important than ranking
stability, but the capacity-constrained threshold must still be expressed as a percentile or a
count.

**Branch point:** if the downstream user will see a probability, the specification must include a
calibration requirement and a recalibration plan. If the user will see only "high risk" versus
"not high risk", the specification must define the display and the threshold that produces it.

### Step 10. Derive the harm ratio and the provisional threshold

**Input:** the benefit of a true positive, the cost of a false positive, and the prevalence.

**Output:** a threshold probability, with the arithmetic shown, plus a sensitivity range over
plausible benefit and cost values.

**Rationale:** The decision-theoretic threshold is the risk at which the expected benefit of
acting equals the expected benefit of not acting. If `B` is the net benefit of a true positive and
`C` is the net cost of a false positive, act when:

```text
p * B > (1 - p) * C
p > C / (B + C)
```

The threshold is not 0.5 unless `B` equals `C`. In screening and prevention, `C` is often much
smaller than `B`, so the threshold is well below 0.5. In high-cost, high-harm interventions, it
can be above 0.5. The arithmetic is simple; the discipline is stating the inputs.

**Branch point:** if `B` or `C` cannot be estimated, do not invent a threshold. Use a range and
report the decision across the range. If the decision flips inside the plausible range, the
project needs better economic evidence before deployment, not a better model.

### Step 11. Reconcile the threshold with capacity

**Input:** the provisional threshold, the score distribution from pilot or development data, and
the action's capacity per period.

**Output:** either a confirmed threshold or a capacity-constrained threshold, with the number
flagged per period and the implied sensitivity and specificity.

**Rationale:** A threshold that flags more patients than the system can treat is not an operating
point; it is a waiting list. The number flagged at threshold `t` is
`N * (sens(t) * prevalence + (1 - spec(t)) * (1 - prevalence))`. Compare that with capacity. If it
exceeds capacity, either raise the threshold (accepting lower sensitivity) or expand capacity
(bearing the cost). Both are legitimate; leaving the mismatch unresolved is not.

**Branch point:** ties in scores at the capacity boundary are common with coarse risk scores. The
specification must state a tie-break rule (for example, clinical priority, then earliest time
zero) and must report how many patients are affected. If ties affect more than a small fraction
of the boundary, the score is too coarse for the capacity constraint and needs more resolution.

The four outcomes of this step are summarised below.

| Condition | Finding | Action |
|---|---|---|
| Flagged count at `t*` is below capacity | Capacity is not binding | Use `t*`; report spare capacity and consider lowering the threshold if harms allow |
| Flagged count at `t*` exceeds capacity | Capacity binds | Raise threshold to the capacity-constrained value, or fund additional capacity; report both options |
| Flagged count equals capacity within rounding | Boundary case | Use the capacity threshold; document tie-breaking and expected queue |
| Capacity unknown or contested | Cannot reconcile | Treat as a blocking issue; run the model as a silent stratification tool until capacity is agreed |

### Step 12. Fix the primary metric and the evaluation plan

**Input:** the decision rule, the threshold rule, and the deployment claim.

**Output:** one primary metric, a small set of secondary metrics, the validation design, the
subgroup list, and the decision rule for "go / no-go".

**Rationale:** Choosing the primary metric after seeing results is a form of analyst degrees of
freedom. Fix it in advance. For a decision aid with a capacity constraint, net benefit at the
operating threshold is a defensible primary metric, with calibration and AUROC as secondary
evidence. For a ruling-out test, sensitivity at a fixed negative predictive value may be primary.
For a ranking tool with fixed capacity, precision at `k` may be primary.

**Branch point:** if the primary metric is net benefit, the threshold must be pre-specified,
because net benefit is a function of the threshold. Reporting the maximum net benefit over all
thresholds is a different (and optimistic) quantity; report the curve and the pre-specified point.

The decision table for metric selection is below.

| Decision structure | Primary metric | Required companion metrics |
|---|---|---|
| Act on everyone above a risk threshold | Net benefit at the pre-specified threshold | Calibration slope and intercept; AUROC; PPV at threshold |
| Rank and treat top `k` | Precision at `k`; recall at `k` | Calibration at the cut; stability of ranking across folds |
| Rule out (avoid a test or admission) | Sensitivity at a required NPV | Specificity; NPV; calibration at the low end |
| Rule in (confirm and treat) | PPV at a required sensitivity | Specificity; calibration at the high end |
| Compare two models for the same decision | Difference in net benefit, with interval | Calibration comparison; subgroup differences |

### Step 13. Write the deployment claim and the monitoring plan

**Input:** the evaluation plan and the intended workflow.

**Output:** a claim sentence, a list of settings where the claim is asserted, and a monitoring
plan with metrics, cadence, and owners.

**Rationale:** The deployment claim determines what evidence is needed. "The model discriminates
readmission risk in adults discharged from this hospital" is supportable with internal validation.
"The model reduces readmissions at partner hospitals" requires external validation and an
outcome study. Writing the claim early prevents the common pattern of running an internal
validation and then writing a marketing claim that the evidence cannot carry.

**Branch point:** if the deployment site differs from the development site in case mix, coding
practice, or workflow, the claim must be conditional on local validation. State the conditions
rather than assuming transportability.

### Step 14. Sign, version, and register

**Input:** the completed specification and pre-registration note.

**Output:** a versioned document with named approvers, a date, and a registration identifier if
the study is prospective.

**Rationale:** The specification is a contract between the clinical, technical, and compliance
functions. Versioning makes later changes visible. Registration (for example, on a clinical trial
registry for prospective evaluations) reduces the risk of post-hoc rewriting. TRIPOD+AI items 18a
through 18f cover protocol, registration, data sharing, and code sharing; the specification should
be written so those items can be completed without reconstruction.

**Branch point:** if the specification changes after model development begins, record the change,
the reason, and whether it was made before or after seeing validation results. A change made after
seeing results is not necessarily wrong, but it must be disclosed.

### Framing edge cases

**Perfect separation at first fit.** If a simple model separates the classes almost perfectly, the
first hypothesis should be leakage, not brilliance. A post-time-zero feature, a label-derived
code, or a duplicate record is more likely than a genuinely near-deterministic clinical
relationship. Return to Steps 5 and 8 before celebrating.

**Prevalence shift.** A model framed at 21 percent prevalence and deployed where prevalence is 12
percent will flag fewer true positives and more false positives at the same threshold. The
specification should name the prevalence assumed, the acceptable range, and the recalibration
trigger. Prevalence shift is a framing failure when it was foreseeable and unstated.

**Tiny subgroups.** A claim about performance in a subgroup with 40 events cannot be supported by
a subgroup AUROC alone. Framing should state the minimum subgroup size for a claim (commonly at
least 100 events and 100 non-events, though this is a convention rather than a rule) and should
pre-specify that smaller subgroups are reported descriptively.

**Single-level categorical variables.** A feature that takes one value in the development cohort
carries no information and will break some encoders. At framing, record the expected levels and
the plan for levels seen only at deployment. A site-specific category (for example, a local
transfer pathway) is a warning that the feature set is not transportable.

**Zero-variance features.** A constant feature is a data-dictionary error or a variable that was
defined after the fact. It should be removed before modelling, but its presence is also a signal
to audit the extraction.

**Empty folds.** In a site-held-out design, a small site may contribute no positive cases to a
fold. The framing decision is whether that site can support a site-level claim at all. If it
cannot, say so before the validation, not after.

**Multiple simultaneous decisions from one score.** If the same score triggers a nurse call, a
pharmacist review, and a physician alert, the capacity and harm ratio differ for each. Frame each
decision separately, even if the underlying model is shared.

## 6. Worked example

### The setting

A 620-bed community hospital wants to reduce 30-day unplanned readmissions after heart failure
discharge. The quality director has funding for a transitional care programme: a nurse navigator
call within 48 hours, a home visit within 7 days, and a pharmacist medication review. The
programme can enrol 25 patients per week. The quality director asks the data science team to
"build a model that predicts readmission".

The team reframes the request using the fourteen steps.

### Step 1 to 3: decision, action, evidence

The decision sentence is: "At discharge, the heart failure nurse navigator decides whether to
enrol a patient in the transitional care programme." The action is enrolment, delivered within 48
hours, with the mechanism of improved self-care, medication reconciliation, and early detection of
decompensation.

The evidence grade is moderate: a local pilot of 180 enrolled patients showed a 25 percent
relative reduction in 30-day readmission compared with matched concurrent controls, with a wide
confidence interval. The team treats 25 percent as the base case and tests 15 percent and 35
percent in sensitivity analysis. The average readmission cost is estimated at USD 16,200 from
finance data; the programme costs USD 640 per enrolled patient.

The net benefit of a true positive is therefore:

```text
B = 0.25 * 16,200 - 640 = 4,050 - 640 = 3,410 USD
C = 640 USD (the cost of enrolling a patient who would not have been readmitted)
```

### Step 4 to 6: population, time zero, horizon

The candidate population is 5,410 heart failure discharges in calendar year 2025. Exclusions:
in-hospital death (214), discharge to hospice (96), left against medical advice (58), out-of-area
residence with no access to the programme (222). Eligible cohort: 4,820 discharges. The flow
diagram records each exclusion with a count.

Time zero is the timestamp of the discharge order signature. The horizon is 30 days from the
discharge date. Readmissions to the index hospital and to two partner hospitals are captured
through a shared admission feed; readmissions outside the network are not captured. That
limitation is recorded in the label definition.

### Step 7: the label

The label is an unplanned inpatient readmission within 30 days, identified from the admission feed
with `admit_type` in {emergency, urgent} and excluding planned procedures and transfers to
rehabilitation. In a chart-review sample of 200 discharges, the label had sensitivity 0.94 and
specificity 0.97 against clinician adjudication. The known error direction is under-ascertainment
for out-of-network readmissions, estimated at 6 percent of true events from a regional claims
comparison. Prevalence in the eligible cohort is:

```text
1,032 / 4,820 = 0.2141, or 21.4 percent
```

### Step 8 and 9: features and output

The feature inventory lists 18 variables available before the discharge order: age, sex, ejection
fraction, NYHA class, creatinine, sodium, haemoglobin, length of stay, prior admissions in 12
months, discharge disposition, and nine medication and comorbidity indicators. Features with
conditional availability (echocardiogram) are flagged. The output is a calibrated probability of
readmission within 30 days.

### Step 10: the provisional threshold

```text
p* = C / (B + C) = 640 / (3,410 + 640) = 640 / 4,050 = 0.158
```

The provisional threshold is 15.8 percent. The team notes that this is far below 0.5 because the
cost of a false positive (USD 640) is much smaller than the benefit of a true positive (USD
3,410). A threshold of 0.5 would be a decision to treat only the most extreme patients, which is
not what the programme intends.

### Step 11: capacity reconciliation

A logistic regression model is developed on 2022-2024 discharges (n = 9,640) and validated
temporally on the 2025 cohort (n = 4,820). AUROC is 0.68 (95 percent CI 0.66 to 0.70) at 21.4
percent prevalence. At the provisional threshold of 0.158, sensitivity is 0.71 and specificity is
0.68. The flagged fraction is:

```text
0.71 * 0.2141 + 0.32 * 0.7859 = 0.1520 + 0.2515 = 0.4035
4,820 * 0.4035 = 1,945 patients per year
```

Capacity is 25 per week times 52 weeks, or 1,300 per year, which is 27.0 percent of 4,820. The
provisional threshold flags 1,945 patients, about 645 more than the programme can absorb.

The team has two options. Raise the threshold until the flagged count equals capacity, or fund the
additional 645 slots. The capacity-constrained operating point is the top 27.0 percent of scores.
At that point, sensitivity is 0.58 and specificity is 0.82, and the implied threshold is
approximately 0.24. The flagged count is:

```text
0.58 * 0.2141 + 0.18 * 0.7859 = 0.1242 + 0.1415 = 0.2657
4,820 * 0.2657 = 1,280 patients per year
```

The small gap between 1,280 and 1,300 is due to rounding in the sensitivity and specificity
estimates; the operational rule is "top 1,300 by score", with ties broken by prior admission
count and then by earliest discharge time. In this cohort, 11 patients sit exactly at the boundary
score, so the tie-break rule affects 11 enrolment decisions per year.

### Step 12: net benefit

Net benefit at threshold `t` is:

```text
NB(t) = sensitivity * prevalence - (1 - specificity) * (1 - prevalence) * t / (1 - t)
```

At the capacity-constrained threshold of 0.24:

```text
Model:     0.58 * 0.2141 - 0.18 * 0.7859 * (0.24 / 0.76)
         = 0.1242 - 0.1415 * 0.3158
         = 0.1242 - 0.0447
         = 0.0795 net true positives per patient

Treat all: 0.2141 - 0.7859 * 0.3158
         = 0.2141 - 0.2482
         = -0.0341
```

At the harm-ratio threshold of 0.158, the model's net benefit is 0.1048, compared with 0.0666 for
treat-all. The capacity constraint therefore costs 0.0253 net true positives per patient, or 25.3
per 1,000 discharges. Over 4,820 discharges that is about 122 net true positives per year, which
is the quantity the programme would recover by funding the additional 645 slots.

Whether that expansion is worth it depends on the marginal cost. The additional 645 slots cost 645
times USD 640, or USD 412,800 per year. At 122 net true positives, the cost per net true positive
is about USD 3,383, against a readmission cost avoided of USD 4,050 per true positive. The margin
is positive but thin, and it depends on the 25 percent relative reduction. At a 15 percent relative
reduction, `B` falls to USD 1,790, the threshold rises to 0.263, and the expansion case weakens
substantially. This is exactly the kind of calculation that belongs in the framing document, not
in a post-hoc appendix.

### Two diagnostics that would reveal a broken analysis

**Diagnostic 1: calibration.** The validation cohort shows a calibration slope of 0.78 and a
calibration intercept of 0.09. A slope below 1 means predictions are overdispersed: low risks are
too low and high risks are too high. An intercept above 0 means the model underpredicts overall.
At a predicted risk of 0.24, the recalibrated risk is:

```text
logit(0.24) = -1.1527
recalibrated logit = 0.09 + 0.78 * (-1.1527) = -0.8091
recalibrated risk = 1 / (1 + exp(0.8091)) = 0.308
```

Patients told they are at 24 percent risk are, on average, at about 31 percent risk. If the
programme uses the nominal threshold in conversations with patients, it understates risk. If it
uses the nominal threshold as an operational cut, the number flagged will not match the planning
estimate. The fix is to recalibrate (Platt scaling or isotonic regression on a held-out set) and
re-derive the operating point, not to reinterpret the existing probabilities.

**Diagnostic 2: the decision curve at the deployment prevalence.** Plotting net benefit against
threshold shows that the model is above treat-all from about 0.08 to 0.62, but the curve has a
steep decline after 0.35, and the confidence band crosses treat-all at about 0.55. A team that
reports only AUROC would not see that the model's advantage is concentrated in the threshold range
where the programme actually operates, and that at very high thresholds the model is no better
than treating everyone. The curve also shows that at the default threshold of 0.5, net benefit is
0.0135, roughly one-sixth of the value at the capacity-constrained point. Reporting AUROC alone
would have hidden a 66 net true positives per 1,000 loss from using the default cut.

### Varying one assumption: prevalence shift

Suppose the programme is so successful that the deployed population's readmission prevalence falls
to 12 percent, while sensitivity and specificity at the fixed threshold of 0.24 remain 0.58 and
0.82. PPV becomes:

```text
PPV = 0.58 * 0.12 / (0.58 * 0.12 + 0.18 * 0.88)
    = 0.0696 / (0.0696 + 0.1584)
    = 0.0696 / 0.2280
    = 0.305
```

PPV falls from 0.467 to 0.305. The programme now enrols 1,280 patients to prevent readmissions in
about 390 of them, rather than 598. Cost per true positive rises from USD 1,370 to USD 2,098.
Net benefit at 0.24 falls from 0.0795 to 0.0196, still positive but much smaller. The decision does
not reverse, but the case for expansion disappears, and the threshold should be revisited. A
framing document that fixed prevalence at 21.4 percent and never revisited it would have missed
this.

### What to report

The primary figure is the decision curve at the deployment prevalence, with the capacity-
constrained operating point marked and the treat-all and treat-none references. The secondary
figure is the calibration plot by decile with the slope and intercept annotated. The primary
metric is net benefit at the pre-specified operating threshold, reported with a bootstrap
confidence interval. AUROC is reported as context, with prevalence, and never alone.

## 7. Code, query, or calculation

The script below implements the arithmetic used in the worked example using only the Python
standard library. It assumes that the validation cohort's confusion matrix is available at the two
operating points and that benefits and costs are constant across patients. Those assumptions are
the ones the specification should state; the code does not remove them.

```python
"""Decision arithmetic for a prediction task specification. Standard library only."""
import math
import random


def threshold_from_costs(benefit, cost):
    """Risk at which acting and not acting have equal expected value."""
    if benefit <= 0:
        raise ValueError("benefit must be positive")
    if cost < 0:
        raise ValueError("cost cannot be negative")
    return cost / (benefit + cost)


def metrics(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    prevalence = (tp + fn) / n
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return {"n": n, "prevalence": prevalence, "sensitivity": sens,
            "specificity": spec, "ppv": ppv, "npv": npv}


def net_benefit(m, threshold):
    """Net true positives per patient at a threshold probability."""
    return (m["sensitivity"] * m["prevalence"]
            - (1 - m["specificity"]) * (1 - m["prevalence"])
            * threshold / (1 - threshold))


def capacity_threshold(scored, capacity):
    """Lowest score admitted when the top `capacity` patients are taken."""
    ordered = sorted(scored, key=lambda pair: pair[0], reverse=True)
    cut = ordered[capacity - 1][0]
    flagged = sum(1 for score, _ in ordered if score >= cut)
    ties = sum(1 for score, _ in ordered if score == cut)
    return cut, flagged, ties


benefit = 0.25 * 16200 - 640          # true positive: avoided cost minus programme cost
cost = 640                            # false positive: programme cost
t_star = threshold_from_costs(benefit, cost)
print(f"B={benefit:.0f} C={cost:.0f} harm-ratio threshold={t_star:.4f}")

harm = metrics(tp=733, fp=1212, tn=2576, fn=299)   # sensitivity 0.71, specificity 0.68
cap = metrics(tp=599, fp=682, tn=3106, fn=433)     # sensitivity 0.58, specificity 0.82
for name, m in (("harm-ratio", harm), ("capacity", cap)):
    print(f"{name}: prev={m['prevalence']:.4f} sens={m['sensitivity']:.4f} "
          f"spec={m['specificity']:.4f} ppv={m['ppv']:.4f} npv={m['npv']:.4f}")

print(f"NB at 0.158 model={net_benefit(harm, 0.158):.4f} "
      f"treat-all={harm['prevalence'] - (1 - harm['prevalence']) * 0.158 / 0.842:.4f}")
print(f"NB at 0.240 model={net_benefit(cap, 0.240):.4f} "
      f"treat-all={cap['prevalence'] - (1 - cap['prevalence']) * 0.240 / 0.760:.4f}")

rng = random.Random(7)
scored = []
for _ in range(4820):
    label = 1 if rng.random() < 0.2141 else 0
    score = rng.betavariate(2.0, 3.0) if label else rng.betavariate(1.3, 4.5)
    scored.append((round(score, 2), label))
cut, flagged, ties = capacity_threshold(scored, 1300)
print(f"capacity cut={cut:.2f} flagged={flagged} boundary_ties={ties}")

p = 0.24
recalibrated = 0.09 + 0.78 * math.log(p / (1 - p))
print(f"recalibrated risk at 0.24 = {1 / (1 + math.exp(-recalibrated)):.4f}")
```

Expected output:

```text
B=3410 C=640 harm-ratio threshold=0.1580
harm-ratio: prev=0.2141 sens=0.7103 spec=0.6800 ppv=0.3769 npv=0.8960
capacity: prev=0.2141 sens=0.5804 spec=0.8200 ppv=0.4676 npv=0.8776
NB at 0.158 model=0.1049 treat-all=0.0666
NB at 0.240 model=0.0796 treat-all=-0.0341
capacity cut=0.36 flagged=1310 boundary_ties=54
recalibrated risk at 0.24 = 0.3081
```

Reading the output:

- `B=3410` is the net benefit of a true positive in US dollars, after subtracting the programme
  cost. `C=640` is the full cost of enrolling a patient who would not have been readmitted.
- The harm-ratio threshold is 0.158. The capacity operating point corresponds to a nominal
  threshold near 0.24, but the code does not estimate that threshold; it takes the confusion
  matrix as given. That is deliberate: the threshold that achieves a capacity is an empirical
  quantile, not a closed-form quantity.
- Net benefit at 0.240 is 0.0796 net true positives per patient. At the harm-ratio threshold it is
  0.1049. The difference, 0.0253 per patient, is the cost of the capacity constraint.
- The capacity function returns 1,310 flagged for a target of 1,300 because 54 patients share the
  boundary score of 0.36. Rounding scores to two decimals creates the tie. With full-precision
  scores, ties would be rarer but still possible with integer risk scores. The tie count is
  reported, not hidden.
- The recalibration print shows the effect of a slope of 0.78 and an intercept of 0.09 at a
  predicted risk of 0.24. The arithmetic is `0.09 + 0.78 * logit(0.24)`, then the inverse logit.

To adapt the script to another project, replace the two confusion matrices with counts from your
validation cohort at the two candidate operating points, replace `benefit` and `cost` with local
values, and replace the synthetic `scored` list with the validation scores. Do not fit the
confusion matrices on the development data; that would defeat the purpose of the exercise.

## 8. Data, standards, and regulatory context

Problem framing sits upstream of most regulated artefacts, but several instruments constrain what
the specification must contain. The table distinguishes binding requirements from conventions.
"Binding" here means a legal or regulatory obligation in the named jurisdiction. A guidance
document is not binding in the same sense as a regulation, even when it is strongly expected.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| FDA intended-use regulation, 21 CFR 801.4 | Binding in the US | The claims a device may make and the population and user named | Does not specify model design or metrics |
| FD&C Act section 515C (FDORA 2022) | Binding in the US | Authorises predetermined change control plans for device software | Does not define what a good PCCP contains |
| FDA PCCP guidance (final December 2024) | Guidance, not binding | Recommends description of modifications, modification protocol, impact assessment | Does not approve a plan without a submission |
| FDA/HC/MHRA GMLP principles (2021) | Guidance, not binding | Ten principles for data, model, and monitoring practice | Does not prescribe thresholds or report formats |
| EU AI Act, Regulation (EU) 2024/1689 | Binding in the EU, phased | Risk classification, transparency, and post-market obligations for AI systems | Does not replace MDR conformity assessment |
| EU MDR, Regulation (EU) 2017/745 | Binding in the EU | Device classification, clinical evaluation, post-market surveillance | Does not address model-specific statistics |
| IMDRF SaMD N41 (2014) | International consensus | Risk categorisation framework for software as a medical device | Not directly enforceable in any jurisdiction |
| ISO 14971:2019 | Voluntary standard, often harmonised | Risk management process for medical devices | Does not specify clinical performance |
| IEC 62304 | Voluntary standard, often harmonised | Software lifecycle processes | Does not cover model validation statistics |
| TRIPOD+AI (2024) | Reporting convention | Items that must be reported for prediction model studies | Does not assess quality or bias |
| PROBAST+AI (2025) | Appraisal tool | Risk of bias and applicability assessment | Does not prescribe a design |
| DECIDE-AI (2022) | Reporting convention | Early live clinical evaluation, including human factors | Does not cover full comparative trials |
| CONSORT-AI and SPIRIT-AI (2020) | Reporting conventions | Trial reports and protocols involving AI interventions | Do not cover model development studies |
| NIST AI Risk Management Framework 1.0 (2023) | Voluntary framework | Govern, map, measure, and manage AI risks | Not sector-specific and not enforceable |

Three consequences for framing follow.

**The intended use must name a user and a population.** A specification that says "clinicians" or
"patients" is not specific enough for a regulated submission. The FDA's intended-use language and
the IMDRF framework both treat the user and the clinical condition as part of the device's
identity. Framing should record the user role, the care setting, and the population, and should
flag any plan to extend them later as a change of intended use rather than a model update.

**The PCCP changes what "frozen model" means.** Under section 515C and the FDA's 2024 guidance, a
manufacturer can describe planned modifications in advance, with a protocol and acceptance
criteria, and then make those modifications without a new marketing submission. Framing therefore
has to decide which parts of the model are allowed to change (for example, recalibration of the
intercept) and which are not (for example, the feature set or the intended use). A model that
silently retrains on local data is not covered by a PCCP unless the plan says so.

**Reporting conventions encode the framing.** TRIPOD+AI items 3a to 3c require the intended use,
target population, and outcomes to be stated; items 5a to 5c require the data sources and study
dates; items 6a to 6d require the outcome definition and timing; item 7 requires the predictor
definitions and measurement timing; item 15 requires the thresholds and how they were chosen.
DECIDE-AI adds the human-in-the-loop workflow, the user training, and the planned response to
errors. A specification written with these items in view can be turned into a report without
reconstruction. One written without them usually cannot.

### Data standards that constrain the feature inventory

Framing is also constrained by the data model that supplies the features. The four standards below
appear repeatedly in biomedical machine-learning projects. None of them guarantees that a feature
is available at time zero; that is a property of the source workflow, not of the standard.

| Standard | What it provides | Framing consequence |
|---|---|---|
| FHIR (Fast Healthcare Interoperability Resources) | Resources such as Patient, Encounter, Observation, Condition, and MedicationRequest, with timestamps | A feature can be traced to a resource and a time; `Observation.effectiveDateTime` is the field that usually decides availability |
| OMOP CDM (Observational Medical Outcomes Partnership Common Data Model) | Relational tables for persons, visits, measurements, conditions, and drugs | Cohorts are defined by concept sets; the concept set version becomes part of the label definition |
| LOINC (Logical Observation Identifiers Names and Codes) | Codes for laboratory and clinical observations | Lab features must specify the LOINC code and unit, not only the local analyte name |
| SNOMED CT and ICD-10/ICD-11 | Clinical terminology and diagnosis classification | Label definitions based on diagnosis codes must state the code set, version, and whether history codes are included |
| CDISC SDTM and ADaM | Tabulation and analysis data models for clinical trials | Trial-based prediction models must map features to SDTM domains and analysis flags to ADaM |

The practical rule is that every feature in the inventory should have a provenance path: source
system, resource or table, code or concept set, timestamp field, and unit. A feature without that
path cannot be audited for time-zero availability, and a feature that cannot be audited cannot be
defended in a validation review. This is a convention rather than a legal requirement, but it is
the convention that makes the difference between a reproducible cohort and a notebook that only
its author can run.

> **Note:** The EU AI Act's obligations are phased. The Regulation entered into force on 1 August
> 2024, with different provisions applying from 2025 through 2027. Medical devices fall under the
> high-risk regime in Annex I, which interacts with MDR conformity assessment. Do not treat a
> single date as the moment all obligations apply; check the current schedule for the specific
> role and system category.

## 9. Industry practice and case evidence

Public cases illustrate framing failures more clearly than any checklist. The cases below are
documented; the lessons drawn from them are inference and are labelled as such.

**Epic Sepsis Model external validation (Wong et al., 2021).** A proprietary sepsis prediction
model was validated at Michigan Medicine in 27,697 patients across 38,455 hospitalisations.
Hospitalisation-level AUROC was 0.63 (95 percent CI 0.62 to 0.64), against vendor-reported values
of 0.76 to 0.83. At the score threshold of 6, the model flagged 18 percent of all hospitalisations
(6,971 of 38,455), missed 67 percent of sepsis cases (1,709 of 2,552), and had a positive
predictive value of 12 percent (sensitivity 33 percent, specificity 83 percent). The authors
reported poor calibration at all horizons. The framing lesson is that the alert threshold had been
set without reference to the local workflow's capacity or to the local prevalence, and the
resulting alert burden was incompatible with clinical review. That lesson is inference from the
reported operating characteristics, not a claim the authors made in those words.

**Readmission models (Kansagara et al., 2011).** A systematic review identified 30 studies of 26
unique models. Of 14 models using retrospective administrative data, the nine tested in large US
populations had c-statistics of 0.55 to 0.65. Models usable at discharge ranged from 0.68 to 0.83,
and only six of 21 studies reporting c-statistics exceeded 0.70. The review concluded that most
models performed poorly and cautioned against using risk-standardised readmission rates for
hospital comparison or payment. The framing lesson is that a model with modest discrimination can
still be useful if the action is effective and targeted, but the specification must state the
action and the expected effect rather than relying on the c-statistic as a proxy for value.

**Cost proxy in a care-management algorithm (Obermeyer et al., 2019).** A widely used population
health algorithm predicted health care costs as a proxy for need. At the same predicted risk
score, Black patients were substantially sicker than White patients, because less money had
historically been spent on their care. The authors estimated that correcting the label would
increase the proportion of Black patients identified for additional help from 17.7 percent to 46.5
percent. The framing lesson is that the choice of label is an ethical decision, and a proxy must be
validated against the construct it stands for, not only against the data it is easy to obtain.

**Autonomous diabetic retinopathy screening (IDx-DR, De Novo DEN180001).** The device was
authorised in 2018 for detecting more than mild diabetic retinopathy in adults with diabetes who
had not been previously diagnosed, using a specific camera in primary care. In the pivotal trial,
sensitivity was 87.2 percent (95 percent CI 81.8 to 91.2) and specificity 90.7 percent (88.3 to
92.7) at a prevalence of 23.8 percent in 819 analysable participants. The framing lesson is that a
narrow intended use, a specified capture device, and a defined user group made the performance
claim defensible. A broader claim ("detects eye disease") would not have been supported by the same
evidence.

**COVID-19 prognostic models (Wynants et al., 2020).** A living systematic review of COVID-19
prediction models found a large number of publications, most at high risk of bias, with few
externally validated and none ready for clinical use at the time of the review. The framing lesson
is that rapid model proliferation is not evidence, and that a specification written after the data
are in hand tends to produce claims that the validation cannot support.

**Transitional care evidence.** The worked example's 25 percent relative reduction is a local
assumption. The published transitional care literature reports effects that vary by programme,
population, and fidelity, and the framing document should carry the range rather than a point
estimate. That is an inference from the general literature, not a claim about a specific trial.

**Technical debt and the unowned model.** Sculley et al. (2015) described the hidden costs of
machine-learning systems: entanglement, undeclared consumers, data dependencies, configuration
debt, and pipeline jungles. In biomedical settings, the most expensive form is an unowned model
that continues to produce scores after the clinical champion changes role. The model does not
fail loudly; it drifts silently until someone audits it. Framing addresses this by naming the
monitoring owner and the review cadence in the specification, not by assuming that the team that
built the model will maintain it.

> **Note:** Vendor-reported performance and independently validated performance can differ
> substantially, as the sepsis case shows. A framing document should record which source supplied
> each performance number and whether the evaluation was independent.

## 10. Failure modes and red flags

The table is ordered roughly by how early the failure can be detected. The first six are framing
failures; the rest typically surface during validation or deployment but originate in framing. A
framing review should be able to mark each row as present, absent, or unknown, and "unknown" should
be treated as a red flag until it is resolved.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Model with no action | The brief says "predict" and never says "do" | Prediction is treated as the deliverable | Output cannot change an outcome | Require an action, actor, and mechanism in the specification |
| Default threshold of 0.5 | The report never mentions a threshold rule | Convention substitutes for decision analysis | Systematic under- or over-treatment | Derive `C/(B+C)` and report the arithmetic |
| Capacity ignored | No one can state how many patients the programme can take | Model evaluated without workflow constraints | Alert fatigue or a waiting list | Reconcile flagged count with capacity before validation |
| Prevalence assumed stationary | Prevalence reported once, from the development cohort | Base rate treated as a model property | PPV and net benefit fall at deployment | State the prevalence range and recalibration trigger |
| Proxy label unvalidated | Label is cost, utilisation, or "concern" | Convenience replaces construct validity | Bias against groups with lower historical spend | Validate proxy against the construct; audit by subgroup |
| Vague time zero | "At admission" or "at discharge" without a timestamped event | Feature boundary is ambiguous | Temporal leakage and unreproducible cohorts | Name the event and the cut-off rule |
| Horizon longer than follow-up | Recent patients have fewer observed outcomes | Administrative censoring is ignored | Label missingness correlated with date | Restrict cohort or model censoring |
| Population defined by data availability | Cohort requires a lab that only some patients receive | Selection mechanism enters the model | Model unusable for the excluded group | State eligibility clinically, then measure missingness |
| One score, several decisions | Same output drives call, review, and alert | Different harms and capacities collapsed | At least one decision is mis-thresholded | Split the specification per decision |
| Metric chosen after results | Primary metric appears only in the results section | Analyst degrees of freedom | Inflated apparent performance | Pre-register the primary metric and threshold |
| Claim broader than evidence | Model card says "improves outcomes" from an AUC study | Outcome claim inferred from discrimination | Regulatory and commercial overreach | Write the claim and the supporting study together |
| Invented harm ratio | `B` and `C` have no source | Threshold arithmetic dressed as evidence | Wrong operating point with false confidence | Label assumptions and run sensitivity ranges |
| Subgroup claim from tiny n | Subgroup AUROC reported without counts | Small-sample noise read as disparity or parity | Misallocated mitigation effort | Pre-specify minimum subgroup size |
| Ties at the capacity boundary | More patients at the cut score than slots | Coarse score or integer risk | Arbitrary enrolment and perceived unfairness | Pre-specify tie-break rule and report tie count |
| Silent trial called prospective | Output not shown to users but called "live" | Evaluation design mislabelled | Claim of workflow benefit unsupported | Use the correct design label and report exposure |
| No named owner | Every meeting has a different "clinical lead" | Accountability diffused | No one supplies capacity or accepts monitoring | Name the owner in the specification and sign-off |

## 11. Assumptions and limits

Every assumption in this table should appear in the specification with an owner and a review date.
An assumption without an owner is a belief, and beliefs are not auditable.

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The action changes the outcome | Without it, prediction has no utility | Intervention is ineffective or poorly delivered | Pilot or trial evidence; fidelity monitoring |
| Benefit of a true positive is constant | Enables a single harm ratio | Effect varies by severity or subgroup | Stratify the effect estimate; test interaction |
| Cost of a false positive is constant | Enables `C/(B+C)` | Downstream costs differ by patient | Micro-costing or finance data by segment |
| Capacity is fixed and known | Enables capacity reconciliation | Staffing varies by season or site | Model capacity as a range; monitor queue |
| Prevalence is stable or predictable | Enables PPV and net benefit planning | Case mix or policy changes | Track prevalence monthly; recalibrate on trigger |
| Label captures the outcome of interest | Supervision requires a target | Coding practice or access differs | Chart-review validation sample |
| Features are available at time zero | Prevents leakage and ensures deployability | Late-arriving labs or notes | Timestamp audit; ablation of suspicious features |
| Development population resembles deployment | Enables transport of performance | Different site, era, or case mix | External validation; subgroup calibration |
| Clinicians act on the output as specified | Workflow benefit depends on adherence | Alert ignored or overridden | Silent trial with response logging |
| The threshold is implementable in the workflow | A decision rule must be executable | Score not displayed at the right moment | Usability test; workflow observation |
| Outcome ascertainment is comparable across groups | Fair comparison requires it | Differential follow-up or access | Audit ascertainment by subgroup |
| The programme budget persists | Capacity depends on funding | Budget cycle or staff turnover | Contractual commitment; monitoring plan |

Two assumptions are easy to overlook because they are about the system rather than the patient.
The first is that patients do not interfere with one another: if the programme enrols the
highest-risk patients first, the remaining patients' risk may change because the care environment
changes. The second is that capacity is fungible across weeks: a programme with 25 slots per week
cannot necessarily absorb a quiet week followed by a busy one, because staff and home-visit
schedules do not store. Both assumptions should be stated when the intervention has a fixed
workforce.

The largest limit is that framing is a prediction of how the model will be used. It is not
evidence of benefit. A specification can be complete and still wrong about the action's effect,
the clinician's response, or the stability of prevalence. The specification should therefore
include a date for revisiting those assumptions, not only a date for signing them.

## 12. Exercises

**Exercise 1. Rewrite a model request as a decision.** Take the sentence "build a model to predict
deterioration" and rewrite it as a decision sentence naming the actor, the action, the time zero,
and the horizon. Deliverable: four sentences, one per element, plus a list of questions you would
ask the clinical owner. Acceptance criteria: all four elements are specific and timestamped where
required; no element is "the care team" or "as needed". Time: 30 minutes.

**Exercise 2. Threshold arithmetic.** For a screening programme with `B = USD 12,000` and
`C = USD 400`, compute the threshold. Repeat for `C = USD 2,000`. Deliverable: the arithmetic and
a one-paragraph interpretation of why the threshold moves. Acceptance criteria: both thresholds
correct to three decimal places; interpretation states the direction and the reason. Time: 30
minutes.

**Exercise 3. Capacity reconciliation.** A cohort has 12,000 eligible patients per year and a
prevalence of 9 percent. At the harm-ratio threshold, the model flags 22 percent of patients. The
programme can treat 1,800 patients per year. Compute the flagged count, compare it with capacity,
and state two options with their consequences. Deliverable: a short memo with the arithmetic and a
recommendation. Acceptance criteria: flagged count correct; both options described; recommendation
names the trade-off. Time: 45 minutes.

**Exercise 4. Map a specification to TRIPOD+AI.** Take the worked example's specification and map
each element to the relevant TRIPOD+AI item (3a to 3c, 4, 5a to 5c, 6a to 6d, 7, 8, 9, 10, 12, 14,
15). Deliverable: a two-column table. Acceptance criteria: at least 12 items mapped; any item that
cannot be completed is marked with the missing information. Time: 60 minutes.

**Exercise 5. Critique a project brief.** Write a 400-word critique of this brief: "We will use
EHR data to build a machine-learning model that predicts readmission. The model will achieve an
AUC above 0.75 and will be deployed to reduce readmissions." Deliverable: the critique, structured
as claims made, claims missing, and questions to ask. Acceptance criteria: identifies at least six
framing gaps, including action, capacity, threshold, prevalence, label, and validation design.
Time: 45 minutes.

**Exercise 6. Write a pre-registration note.** For a project of your choice, write a one-page
pre-registration note containing the decision, population, time zero, horizon, label, features,
output, threshold rule, primary metric, subgroups, and stopping rule. Deliverable: the note.
Acceptance criteria: every element present; primary metric and threshold fixed before results;
subgroup list justified. Time: 90 minutes.

**Exercise 7. Analyse the sepsis case.** Using the Wong et al. numbers (AUROC 0.63, 18 percent
flagged, sensitivity 33 percent, PPV 12 percent, 38,455 hospitalisations), reconstruct the implied
prevalence and explain why the alert burden was incompatible with review capacity. Deliverable: a
calculation and a 300-word analysis. Acceptance criteria: implied prevalence correct within 0.5
percentage points; analysis distinguishes documented fact from inference. Time: 60 minutes.

**Exercise 8. Build and apply a framing checklist.** Create a 12-item checklist for your
organisation based on Sections 5 and 10, then apply it to a real or hypothetical project and score
it. Deliverable: the checklist, the scored project, and a remediation plan for failed items.
Acceptance criteria: each item is binary and observable; scores are justified with evidence; the
remediation plan names an owner and a date. Time: 90 minutes.

## 13. Self-check answers

**Answer to Exercise 1 (model request).** A complete rewrite is: "At the time the ward nurse
reviews the patient at the start of each shift, the nurse decides whether to request a rapid
response review, for adults admitted to the medical ward, using a prediction of deterioration
within the next 8 hours." The four elements are the actor (ward nurse), the action (request rapid
response review), the time zero (start of shift review), and the horizon (8 hours). Questions for
the owner include: what is the current trigger for a rapid response, how many reviews can the team
absorb per shift, and what happens when the score is high but the nurse disagrees.

**Answer to Exercise 2 (threshold arithmetic).** With `B = 12,000` and `C = 400`:
`400 / (12,000 + 400) = 400 / 12,400 = 0.0323`. With `C = 2,000`:
`2,000 / (12,000 + 2,000) = 2,000 / 14,000 = 0.1429`. The threshold rises from 3.2 percent to 14.3
percent because the false positive has become five times more costly relative to the same benefit.
The decision rule becomes more selective.

**Answer to Exercise 3 (capacity).** Eligible patients 12,000; prevalence 9 percent means 1,080
positives and 10,920 negatives. A flagged fraction of 22 percent is 2,640 patients per year.
Capacity is 1,800, so the model flags 840 more patients than the programme can treat, or 47 percent
more than capacity. Option one is to raise the threshold to the top 15 percent of scores, which
reduces sensitivity and misses more true positives. Option two is to fund 840 additional slots,
which increases cost and may not be justified if the marginal net benefit is small. The
recommendation should state which option preserves more net benefit per unit of capacity and
whether the marginal cost per additional true positive is acceptable.

**Answer to Exercise 7 (sepsis case).** The implied prevalence follows from
`PPV = sensitivity * prevalence / flagged fraction`. Rearranging:
`prevalence = PPV * flagged fraction / sensitivity = 0.12 * 0.18 / 0.33 = 0.0655`, or 6.5 percent.
The observed case count gives `2,552 / 38,455 = 0.0664`, or 6.6 percent, which is consistent within
rounding. The alert burden of 18 percent of hospitalisations is incompatible with a review process
that can examine a small number of patients per shift, and the low PPV means most reviewed patients
did not have sepsis. The documented facts are the AUROC, the flagged fraction, the sensitivity,
the PPV, and the counts. The inference is that the threshold was mismatched to capacity; the paper
reports the operating characteristics but does not frame them in capacity terms.

**Rubric for the design exercises (4, 5, 6, 8).** A strong submission (1) names a specific
decision and actor, (2) states time zero with a timestamped event, (3) derives or justifies the
threshold, (4) reconciles capacity, (5) fixes the primary metric and subgroup list in advance, and
(6) distinguishes assumptions from evidence. A competent submission meets four of the six. A weak
submission lists model features and metrics without a decision, uses 0.5 as the threshold without
justification, or claims outcome benefit from a discrimination study.

## 14. Interview questions

**1. What is the difference between a prediction task and a decision problem?** A strong answer
states that a prediction task maps information to a probability, while a decision problem adds an
action, an actor, a threshold, and a capacity, and that the model is only useful if the action
changes the outcome. A weak answer describes model families without mentioning the action.

**2. Why is 0.5 a bad default threshold?** A strong answer derives `C/(B+C)` and explains that the
threshold depends on the relative harm of errors and on prevalence, and that 0.5 is correct only
when the harms are equal. A weak answer says 0.5 is "standard practice" or confuses the threshold
with the decision boundary of a classifier.

**3. How do you choose a threshold when capacity is constrained?** A strong answer computes the
flagged count at the harm-ratio threshold, compares it with capacity, and chooses between raising
the threshold and expanding capacity, reporting the net benefit of both. A weak answer says "use
the top N patients" without acknowledging the sensitivity loss.

**4. What is time zero and why does it matter?** A strong answer defines it as the instant features
are frozen, gives an example with a timestamped clinical event, and connects it to leakage
prevention and label ascertainment. A weak answer says "when the prediction is made" without an
operational definition.

**5. How would you frame a readmission model for a new hospital?** A strong answer asks about the
local action, capacity, prevalence, coding practice, and workflow, and proposes local calibration
and validation before any outcome claim. A weak answer proposes retraining on local data and
reporting AUROC.

**6. What is a proxy label, and what can go wrong?** A strong answer defines it, gives the cost
proxy example, and explains that a proxy can encode historical inequities or differential access.
A weak answer treats any available variable as a valid label.

**7. When is a model with AUROC 0.65 useful?** A strong answer says it can be useful when the
action is effective, the threshold is well chosen, calibration is adequate, and net benefit is
positive, and notes that readmission models in this range have been deployed with mixed results.
A weak answer says AUROC 0.65 is always too low.

**8. What should a prediction task specification contain?** A strong answer lists decision, actor,
action, population, time zero, horizon, label, features, output, threshold rule, capacity,
primary metric, subgroups, and monitoring. A weak answer lists only data and model choices.

**9. How do you handle a stakeholder who wants to skip framing and start modelling?** A strong
answer proposes a short framing workshop with a one-page artefact, shows how framing reduces
rework, and agrees to start exploratory modelling in parallel without making claims. A weak answer
either refuses outright or concedes and builds an unactionable model.

**10. What would make you stop a project at the framing stage?** A strong answer names no
actionable intervention, no capacity, an unvalidated proxy label with equity risk, or a deployment
claim that cannot be supported by any feasible study. A weak answer cannot name a stopping
condition.

## 15. Key terms

- **Action** — the intervention triggered by a positive prediction.
- **Actionability** — the degree to which the action changes the outcome for a flagged patient.
- **Actor** — the person or system that executes the action.
- **Alert burden** — alerts generated per unit of clinical work.
- **AUROC** — area under the receiver operating characteristic curve.
- **AUPRC** — area under the precision-recall curve.
- **Base rate** — prevalence in the deployment population.
- **Benefit of a true positive (`B`)** — net value of correctly flagging a positive.
- **Calibration** — agreement between predicted probabilities and observed frequencies.
- **Capacity** — number of actions deliverable per period.
- **Capacity-constrained threshold** — threshold that flags exactly the deliverable number.
- **Case mix** — distribution of severity and comorbidity in a population.
- **Cost of a false positive (`C`)** — net cost of flagging a patient who would not have the outcome.
- **Decision curve analysis** — net benefit plotted across threshold probabilities.
- **Deployment claim** — performance or outcome claim the team intends to defend.
- **Discrimination** — ability to rank positives above negatives.
- **Eligible population** — target population minus contraindications and access limits.
- **External validation** — evaluation in data not used for development, from another site or era.
- **Harm ratio** — cost of a false positive relative to benefit of a true positive.
- **Intended use** — regulated statement of purpose, population, and user.
- **Label** — observed outcome used as the supervised target.
- **Net benefit** — prevalence-weighted true positive rate minus threshold-weighted false positive rate.
- **Number needed to treat** — patients who must receive the action for one to benefit.
- **PPV** — positive predictive value.
- **Prediction horizon** — interval from time zero to outcome ascertainment.
- **Prevalence** — proportion of the eligible population with the label.
- **Proxy label** — measured variable used in place of the outcome of interest.
- **Silent trial** — prospective scoring without exposing output to users.
- **Threshold** — predicted risk above which the action is taken.
- **Time zero** — instant the prediction is made and features are frozen.
- **Utility** — expected value of a decision rule under a stated utility model.

## 16. Sources

All URLs accessed 25 October 2026.

1. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting
   clinical prediction models that use regression or machine learning methods. *BMJ*
   2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
2. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and
   applicability assessment tool for prediction models using regression or artificial intelligence
   methods. *BMJ* 2025;388:e082505. https://doi.org/10.1136/bmj-2024-082505
3. Vasey B, Nagendran M, Campbell B, et al. Reporting guideline for the early-stage clinical
   evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. *Nature
   Medicine* 2022;28:924-933. https://doi.org/10.1038/s41591-022-01772-9
4. US Food and Drug Administration. Good Machine Learning Practice for Medical Device Development:
   Guiding Principles. 2021.
   https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
5. US Food and Drug Administration. Marketing Submission Recommendations for a Predetermined Change
   Control Plan for Artificial Intelligence-Enabled Device Software Functions. Final guidance,
   December 2024.
   https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
6. US Food and Drug Administration. De Novo classification decision, IDx-DR, DEN180001. 2018.
   https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
7. International Medical Device Regulators Forum. Software as a Medical Device: Possible Framework
   for Risk Categorization and Corresponding Considerations. IMDRF/SaMD WG/N41, 2014.
   https://www.imdrf.org/documents/software-medical-device-samd-possible-framework-risk-categorization-and-corresponding-considerations
8. European Union. Regulation (EU) 2024/1689 laying down harmonised rules on artificial
   intelligence (Artificial Intelligence Act). 2024.
   https://eur-lex.europa.eu/eli/reg/2024/1689/oj
9. National Institute of Standards and Technology. Artificial Intelligence Risk Management
   Framework (AI RMF 1.0). NIST AI 100-1, 2023. https://doi.org/10.6028/NIST.AI.100-1
10. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models.
    *Medical Decision Making* 2006;26(6):565-574. https://doi.org/10.1177/0272989X06295361
11. Vickers AJ, van Calster B, Steyerberg EW. A simple, step-by-step guide to interpreting decision
    curve analysis. *Diagnostic and Prognostic Research* 2019;3:18.
    https://doi.org/10.1186/s41512-019-0064-7
12. Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive
    analytics. *BMC Medicine* 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7
13. Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
    sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*
    2021;181(8):1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
14. Kansagara D, Englander H, Salanitro A, et al. Risk prediction models for hospital readmission:
    a systematic review. *JAMA* 2011;306(15):1688-1698. https://doi.org/10.1001/jama.2011.1515
15. Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to
    manage the health of populations. *Science* 2019;366(6464):447-453.
    https://doi.org/10.1126/science.aax2342
16. Abràmoff MD, Lavin PT, Birch M, Shah N, Folk JC. Pivotal trial of an autonomous AI-based
    diagnostic system for detection of diabetic retinopathy in primary care offices. *npj Digital
    Medicine* 2018;1:39. https://doi.org/10.1038/s41746-018-0040-6
17. Wynants L, Van Calster B, Collins GS, et al. Prediction models for diagnosis and prognosis of
    covid-19: systematic review and critical appraisal. *BMJ* 2020;369:m1328.
    https://doi.org/10.1136/bmj.m1328
18. Sculley D, Holt G, Golovin D, et al. Hidden technical debt in machine learning systems.
    *Advances in Neural Information Processing Systems* 28, 2015.
    https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems
19. Agniel D, Kohane IS, Weber GM. Biases in electronic health record data due to processes within
    the healthcare system: retrospective observational study. *BMJ* 2018;361:k1479.
    https://doi.org/10.1136/bmj.k1479
20. Finlayson SG, Subbaswamy A, Singh K, et al. The clinician and dataset shift in artificial
    intelligence. *New England Journal of Medicine* 2021;385(3):283-286.
    https://doi.org/10.1056/NEJMc2104626

## 17. Links

- Module map: [`_index.md`](_index.md)
- Next topic: [`02_labels-and-outcomes.md`](02_labels-and-outcomes.md)
- Validation design: [`03_splitting-and-validation.md`](03_splitting-and-validation.md)
- Survey file: [`../../06_machine_learning_foundations.md`](../../course/06_machine_learning_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration and decision curve lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
