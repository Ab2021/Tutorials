# The Production Loop

> Module 15 — Deployment and Machine Learning Operations (MLOps)
> Topic 01 of 10
> Estimated study time: 3 hours
> Prerequisites: 03, 06
> Version 2026.3

## 1. Why this topic exists

A model that performs well on a held-out test set has demonstrated one thing: that a particular
function, fitted on a particular sample, generalizes to similar samples drawn from the same
process. It has not demonstrated that the function will still be useful in a hospital on a
Tuesday in March, three years after training, when the laboratory interface has been upgraded,
the case mix has shifted, and the clinician who championed it has changed jobs. The gap between
those two claims is the subject of this module, and the production loop is the structure that
closes it.

Most organizations do not lack modelling skill. They lack a repeatable mechanism for moving a
model from a notebook into a workflow, observing what happens, and changing the model without
breaking the workflow or the regulatory position. The failure is structural. A team is funded to
produce a model; nobody is funded to own it. A pipeline is built to train once; nobody builds the
job that retrains it. A dashboard is built to show uptime; nobody builds the monitor that would
notice the model has become less sensitive while the service is perfectly healthy.

The production loop exists as a topic because the individual stages are taught separately and
the connections between them are where projects die. Data engineering teaches pipelines but not
model behaviour. Machine learning teaches evaluation but not deployment-shaped evaluation.
Software engineering teaches reliability but not the difference between a broken service and a
broken model. Regulatory affairs teaches change control but is usually engaged after the
architecture is fixed. A learner who has all four bodies of knowledge and no picture of the loop
will still ship a system that nobody can operate.

### 1.1 The specific decisions this topic unblocks

The loop is not an abstraction. It answers concrete questions that arise in the first month of
any deployment.

| Decision | Who is blocked | What goes wrong without the loop |
|---|---|---|
| Which metric defines "working" in production | Clinical owner, product manager | Teams argue about AUROC while the service is down, or about uptime while sensitivity collapses |
| When a model change needs revalidation | Regulatory lead, quality manager | Silent coefficient updates create an unaudited change to a regulated device |
| How long the current model can stay in service | Engineering manager, finance | Retraining is either never scheduled or scheduled on a calendar with no evidence |
| Who is paged at 02:00 and what they are allowed to do | On-call engineer, clinical operations | Alerts route to a shared inbox; nobody has authority to roll back |
| Whether the model is still worth its maintenance cost | Budget owner | The model is kept alive because removal requires a decision nobody owns |
| What evidence proves the model works after a vendor change | Auditor, customer | The only evidence is a validation report dated before the change |

Each row is a governance question as much as a technical one. The loop makes the questions
explicit early, when they are cheap to answer, rather than during an incident or an audit.

### 1.2 Why the loop is drawn as a loop

Linear diagrams encourage a linear mental model: build, test, ship, done. That model is wrong in
a way that produces predictable errors. A deployed model is a control system with two feedback
paths running at different speeds.

The fast path runs in seconds to hours: request, prediction, response, error, latency. It is
owned by service reliability. The slow path runs in weeks to months: outcome, label, performance,
drift, retrain. It is owned by the clinical and data science function. Confusing the two paths
produces two classic mistakes. Reliability teams declare a model healthy because the service is
up. Data science teams declare a model healthy because the offline metric has not changed, while
the service has been returning stale features for a month.

The loop also has an outer ring that is easy to forget: retirement. A model that is no longer
used still consumes storage, monitoring attention, security review, and audit scope. Removing it
is a change with its own validation and communication requirements. The loop is only complete if
it includes the exit.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** the production loop from memory, naming the input, output, owner, and failure mode of
   each stage.
2. **Distinguish** service health monitoring from model health monitoring, and give one metric
   that detects each while remaining blind to the other.
3. **Specify** a service level objective (SLO) with a measurement window, an indicator, and an
   error budget, and compute the budget in the unit the on-call engineer actually observes.
4. **Choose** a retraining trigger from evidence (scheduled, drift-triggered, performance-triggered,
   or incident-triggered) and justify the choice against the cost of revalidation.
5. **Diagnose** a deployment that is technically available but clinically degraded, using at least
   two independent signals.
6. **Map** each stage of the loop to the artifact it produces and the change-control obligation it
   creates.
7. **Critique** a proposed deployment plan for missing feedback edges, unnamed owners, and absent
   stop criteria.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13. Outcomes 3 and
5 are the calculation-heavy ones; outcomes 1 and 6 are the recall-and-structure ones; outcome 7
is the critique exercise.

## 3. The core mental model

The production loop has seven stages and four feedback edges. The stages are the work. The edges
are what make it a loop rather than a release process.

```text
                        ┌──────────────────────────────────────────────┐
                        │                                              │
                        ▼                                              │
   ┌─────────┐   ┌───────────┐   ┌─────────┐   ┌────────────┐   ┌─────┴─────┐
   │  Data   │──▶│ Features  │──▶│  Model  │──▶│ Prediction │──▶│ Workflow  │
   └────┬────┘   └─────┬─────┘   └────┬────┘   └──────┬─────┘   └─────┬─────┘
        │              │              │               │               │
        │              │              │               │               ▼
        │              │              │               │         ┌───────────┐
        │              │              │               │         │   Action  │
        │              │              │               │         └─────┬─────┘
        │              │              │               │               │
        │              │              │               │               ▼
        │              │              │               │         ┌───────────┐
        │              │              │               │         │  Outcome  │
        │              │              │               │         └─────┬─────┘
        │              │              │               │               │
        │              ▼              ▼               ▼               ▼
        │        ┌────────────────────────────────────────────────────────┐
        └───────▶│  Feedback: quality, drift, performance, safety, cost   │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                    ┌────────────────────────┴───────────────────────┐
                    ▼                                                ▼
             ┌─────────────┐                                  ┌─────────────┐
             │  Retrain    │                                  │   Retire    │
             └──────┬──────┘                                  └─────────────┘
                    │
                    └──────────────▶ back to Model (new version)
```

### 3.1 The two loops

Read the diagram as two nested control systems.

**Inner loop, minutes to hours.** Data arrives, features are computed, the model scores, the
workflow acts. Sensors here are request rate, error rate, latency, input validation failures,
prediction distribution, and alert volume. Actuators are the deployment switch, the feature flag,
the threshold, and the fallback path. This loop is owned by the service team and is measured in
minutes. Its failure signature is a service that is down, slow, or returning errors.

**Outer loop, weeks to months.** Outcomes accumulate, labels arrive, performance is recomputed,
drift statistics are compared against a reference, and the model is retrained or recalibrated.
Sensors here are calibration, discrimination, subgroup performance, outcome rates, override
rates, and data quality. Actuators are a new model version, a recalibration, a threshold change,
or a retirement. This loop is owned jointly by data science and clinical operations. Its failure
signature is a service that is perfectly available and clinically wrong.

The most important sentence in this topic: **the inner loop cannot detect an outer-loop failure,
and the outer loop is too slow to protect patients from an inner-loop failure.** You need both,
instrumented independently, with different owners and different escalation paths.

### 3.2 Why each stage exists, in causal order

Each stage exists to reduce a specific uncertainty before the next, more expensive stage is
attempted.

- **Data** reduces uncertainty about whether the inputs exist, at the right time, with the right
  meaning. Most deployment failures originate here and surface later.
- **Features** reduce uncertainty about whether the training-time computation can be reproduced
  at serving time, from the same source, at the same moment in the workflow.
- **Model** reduces uncertainty about whether a fitted relationship is stable enough to be useful
  outside the training sample.
- **Prediction** reduces uncertainty about whether the model output can be produced within the
  latency and reliability budget of the decision.
- **Workflow** reduces uncertainty about whether the output reaches the person who acts, at the
  moment they act, in a form they can use.
- **Action** reduces uncertainty about whether the recommendation changes what happens.
- **Outcome** reduces uncertainty about whether the change was beneficial, harmful, or neutral.

The order matters because a failure at stage N is usually observed at stage N+2. A feature
computed with the wrong time window (stage 2) produces a plausible prediction (stage 4) and a
wrong action (stage 6). By the time the outcome (stage 7) reveals it, the causal distance is
large enough that teams misdiagnose it as drift.

### 3.3 The evidence ladder

Each stage should produce an artifact that a stranger can inspect. The artifacts form a ladder
from "we intended" to "we observed."

| Rung | Artifact | Proves | Does not prove |
|---|---|---|---|
| 1 | Intended-use statement | What the model is for | That it works |
| 2 | Data contract and quality report | Inputs were as expected at build time | Inputs will stay as expected |
| 3 | Offline validation report | Performance on a held-out sample | Performance in the workflow |
| 4 | Shadow evaluation | Performance on live traffic without action | Performance under clinical use |
| 5 | Canary or stepped rollout | Performance under partial action | Performance at full scale |
| 6 | Production monitoring record | What happened, when, to whom | That the model caused it |
| 7 | Post-market surveillance | Whether benefits persist | That no unmeasured harm exists |

The ladder is a communication device. When a vendor says "validated," ask which rung. When an
internal team says "it works in production," ask for rung 6 evidence with a date range. Most
disagreements about model quality are disagreements about which rung is being discussed.

### 3.4 Where the loop typically breaks

Four edges carry most of the failure.

1. **Data to Features.** The pipeline that builds training features is not the pipeline that
   builds serving features. The two drift apart silently. This is training-serving skew.
2. **Prediction to Workflow.** The model is available but the integration is fragile: the alert
   fires in a tab nobody opens, or the latency exceeds the time the clinician has.
3. **Workflow to Outcome.** The action is not recorded, so the outcome cannot be attributed. The
   loop loses its ability to learn.
4. **Outcome to Retrain.** Labels arrive late or never, so retraining is triggered by calendar or
   by drift proxy rather than by measured performance.

If you instrument only one edge, instrument the first and the fourth. Skew tells you the model is
being fed something different from what it learned; label latency tells you how long you are blind.

## 4. Definitions and vocabulary

The vocabulary of MLOps is unstable across vendors and organizations. The definitions below are
the ones used in this module. Where a term has a common misuse, it is named.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Production loop | The closed set of stages from data to outcome plus the feedback edges that change the model or the workflow | Used to mean "the CI/CD pipeline" | None |
| Deployment | Making a model available at the point of decision, with an owner and a rollback path | Used to mean "uploaded an artifact" | None |
| Serving | The runtime that accepts a request and returns a prediction | Confused with the model itself | Requests per second |
| Inference | One execution of the model on one input | Used loosely to mean "the whole prediction service" | Milliseconds per call |
| Model artifact | The serialized parameters plus preprocessing, config, and metadata needed to reproduce a prediction | The weights file alone | Version identifier |
| Model version | An immutable identifier for an artifact and its training provenance | A mutable "latest" tag | Semantic version or content hash |
| Feature | A named, typed input computed from source data at a defined time | Any column in a table | Unit depends on feature |
| Feature view | A versioned definition of how a feature is computed and when it is valid | A SQL query saved in a repository | Version and owner |
| Training-serving skew | Any difference between the data or transformation used in training and that used at serving | Used only for preprocessing bugs | Measured as distribution distance or parity test |
| Shadow mode | The model scores live traffic but its output is not shown or acted on | Running the model on historical data | Predictions per day |
| Canary | A limited, time-bound exposure of a change to a subset of traffic with pre-specified evaluation | A small deployment with no evaluation | Traffic fraction, duration |
| Drift | A change over time in the input distribution, the label distribution, or the input-output relationship | Used as a synonym for "model is wrong" | Population stability index (PSI), KS statistic, or performance delta |
| Service level indicator (SLI) | A quantitative measure of service behaviour, such as the fraction of requests served under 800 ms | A dashboard panel with no threshold | Ratio or percentile |
| Service level objective (SLO) | A target range for an SLI over a stated window | A vague aspiration such as "high availability" | Ratio over 28 days |
| Error budget | The amount of allowed failure implied by the SLO: 1 minus the SLO, times the window | Treated as a suggestion rather than a budget | Failed requests or minutes |
| Retraining | Fitting a new model on new or additional data | Re-running the same training job on the same data | Job identifier |
| Recalibration | Adjusting the output mapping without refitting the feature relationships | Used interchangeably with retraining | Calibration slope, intercept |
| Revalidation | Re-establishing that the model meets its acceptance criteria in the intended population | Confused with retraining | Validation report identifier |
| Predetermined change control plan (PCCP) | A pre-authorized description of modifications a device may undergo, with a protocol and impact assessment | Treated as a general permission to update | Submission identifier |
| Model card | A structured summary of intended use, data, performance, limitations, and monitoring | A marketing one-pager | Document version |
| Model registry | A versioned store of artifacts with metadata and stage or alias transitions | A shared network folder | Registry URI |
| Runbook | The procedure an on-call engineer follows for a named alert | A wiki page last edited two years ago | Document version |
| Incident | An unplanned event that degrades a service, a model, or the safety of a workflow | Any alert that fires | Incident identifier |
| Rollback | Restoring the previous known-good combination of image, artifact, config, and schema | Redeploying the same image with a changed config | Time to restore, minutes |
| Champion/challenger | A pattern in which a current model (champion) and a candidate (challenger) are compared under controlled exposure | A/B test with no pre-specified decision rule | Traffic split |
| Point-in-time correctness | The property that a feature used at time t reflects only data available at or before t | Assumed to be automatic in a data warehouse | Event time versus processing time |
| Label latency | The delay between the prediction and the availability of the outcome used to evaluate it | Ignored in monitoring design | Days |
| Silent failure | A failure that produces no error signal because the service is responding normally with degraded outputs | Detected only by outcome review | Detection time, days |

Two definitions deserve emphasis because they are routinely conflated.

**Deployment is not serving.** A model can be deployed (an artifact exists in an environment) and
not serving (no traffic reaches it). A model can be serving and not deployed in any controlled
sense (a notebook cron job writing scores into a shared table). The loop requires both: traffic
and control.

**Drift is not harm.** A change in input distribution may be benign, expected, or even desirable.
A model can drift and remain well calibrated. Treating drift as a synonym for degradation
produces retraining churn, which is itself a risk in a regulated setting.

> **Note:** "Model" in this module means a fitted statistical function plus everything required to
> turn raw input into its output. That includes preprocessing, encoding, imputation, and threshold
> logic. A bug in any of those is a model bug, even if the weights are unchanged.

## 5. The mechanics, step by step

The procedure below is the loop as an operator would run it, not as a diagram presents it. Each
step names its input, its output, and the reason it exists. Steps 1 to 3 are design work that
happens before any model is trained. Steps 4 to 7 produce and verify the artifact. Steps 8 to 10
are the live operation. Steps 11 to 13 close the loop and end it.

A team that skips a step does not remove the work; it moves the work to an incident, an audit
finding, or a retraining cycle that cannot be justified.

### Step 1 — Frame the decision and the moment of use

**Input.** A clinical or operational problem statement, a named decision owner, and a description
of the current workflow.

**Output.** A one-page decision frame: the decision, the actor, the moment, the action taken
today, the action the model would change, and the cost of each error type.

**Rationale.** Every downstream choice depends on the decision. The latency budget, the
acceptable false positive rate, the label definition, and the deployment pattern all follow from
"who acts, when, and on what." A model without a decision frame cannot be evaluated, because
there is no threshold at which its output becomes useful or harmful.

The frame must name a person and a moment, not a role and a phase. "Clinicians will use it during
rounds" is not a frame. "The night-shift nurse reviews the deterioration list at 06:30 before
handover, and either escalates to the rapid response team or documents why not" is a frame. The
difference determines whether the model needs to run at 06:00 in batch or at 06:29 in real time.

Branch point: if the decision is already made by an algorithm, the frame must describe the
existing algorithm and the proposed change to it. Replacing a rule with a model is a different
project from adding a model where no rule exists, and carries different validation obligations.

### Step 2 — Specify the operating envelope and acceptance criteria

**Input.** The decision frame and the intended patient or member population.

**Output.** A written envelope: inclusion and exclusion criteria, sites, care settings, device
types, languages, payer mix, age range, and the performance floor below which the model must not
be used.

**Rationale.** A model is valid inside an envelope, not in general. The envelope is what allows a
later reviewer to decide whether a new site or a new population is inside the original evidence
or requires local validation. Without it, every expansion becomes an argument.

Acceptance criteria must be stated before validation, in units the decision owner understands.
"AUROC at least 0.78" is a statistical criterion. "At a threshold of 0.20, at least 60 percent of
patients who deteriorate within 12 hours are flagged, and no more than 5 percent of
non-deteriorating patients are flagged" is a clinical criterion. Both can coexist, but the second
is the one that determines whether the ward can absorb the alert volume.

Branch point: if the envelope cannot be bounded because the deployment targets "all patients,"
the acceptance criteria must include subgroup performance floors. An unbounded envelope with an
aggregate metric is a hidden equity risk.

### Step 3 — Freeze the data contract

**Input.** Source systems, interface specifications, update schedules, and the fields required by
the features.

**Output.** A versioned data contract: field names, types, units, allowed ranges, null semantics,
arrival-time guarantees, and an owner for each source.

**Rationale.** Most silent model failures are data-contract failures wearing a model costume. The
contract makes the assumption testable. It also assigns the fix: when the laboratory feed changes
its units from mg/dL to µmol/L, the contract owner is the laboratory interface team, not the data
science team.

The contract must distinguish event time from arrival time. A creatinine measured at 07:00 and
arrived at 09:15 is not available to a model that scores at 08:00. A contract that guarantees
only "daily delivery" cannot support a real-time decision.

Branch point: if a source cannot commit to an arrival-time guarantee, the feature must be marked
as delayed and the model must either exclude it or use a time-shifted definition that matches
what will actually be available.

### Step 4 — Build the training pipeline as a reproducible job

**Input.** The data contract, the feature definitions, and the label definition.

**Output.** A versioned pipeline that produces a model artifact from a named data snapshot, with
logged parameters, environment, and data hash.

**Rationale.** Reproducibility is not a research virtue; it is the precondition for rollback and
for audit. If the training job cannot be re-run to produce the same artifact, then a later
investigation cannot distinguish a data change from a code change from a numerical difference in
a library.

Reproducibility has three levels, and teams routinely claim the first while needing the third:
the code is in version control; the environment is pinned; the data snapshot is content-addressed
and the artifact is reproducible bit-for-bit or statistically. The third level is what a
regulated change control process usually needs.

Branch point: if exact bit-for-bit reproduction is impossible because of nondeterministic
hardware kernels, record the seed, the library versions, and the distributional tolerance within
which the artifact is considered equivalent. State that tolerance in the model card.

### Step 5 — Validate against deployment-shaped splits

**Input.** The artifact, the envelope, and a validation protocol.

**Output.** A validation report with discrimination, calibration, subgroup performance, and
decision-curve analysis at the proposed threshold, using splits that mimic deployment.

**Rationale.** A random split answers "does this generalize to another sample from the same
process." Deployment asks "does this generalize to a later time, a different site, a different
documentation practice, and a different case mix." Temporal splits, site-held-out splits, and
retrospective replay answer the deployment question more honestly.

Validation must include the threshold. A model with good AUROC can have unacceptable alert volume
at the operating point. Reporting discrimination without the operating point hides the decision
the ward will actually experience.

Branch point: if the only available validation is a random split from a single site, the report
must say so and the rollout must include a shadow period that collects site-specific evidence.
The absence of external validation is a limit, not a disqualification, provided it is stated.

### Step 6 — Package artifact, preprocessing, and metadata together

**Input.** The trained artifact, the preprocessing code, the configuration, and the model card.

**Output.** An immutable package: container image or equivalent, model file, feature definitions,
threshold, dependency lockfile, model card, and content digest.

**Rationale.** A model file without its preprocessing is not deployable. A container without a
pinned base image is not reproducible. A package without a model card cannot be reviewed by a
clinician or an auditor. Packaging is the step where the artifact becomes a thing that can be
versioned, signed, and rolled back as a unit.

The package must include the threshold as configuration, not as a constant buried in code. A
threshold change is a model change in effect, and it needs to be visible, reviewable, and
reversible.

Branch point: if the serving environment cannot run the training environment's dependencies, the
package must include a serving-specific runtime and a parity test that compares its outputs
against the reference implementation on a fixed input set.

### Step 7 — Verify before exposure: replay, shadow, and contract tests

**Input.** The package, a historical or live traffic stream, and the acceptance criteria.

**Output.** A verification record: output parity, latency distribution, error rate under load,
schema rejection rate, and a go/no-go recommendation.

**Rationale.** Offline validation uses curated data. Replay and shadow use the messy traffic the
service will actually see, including missing fields, out-of-range values, duplicate requests, and
the long tail of malformed input. This is where training-serving skew becomes visible.

Shadow mode is not free. It doubles inference cost, requires logging that does not expose
protected health information, and can create a false sense of safety if the shadow traffic is not
representative. A shadow period over a quiet weekend proves little about Monday morning.

Branch point: if shadow traffic cannot be logged safely, use replay on de-identified historical
requests and state the limitation. Do not log identifiers to make the evaluation easier.

### Step 8 — Roll out in controlled stages

**Input.** The verified package, a rollout plan, and pre-specified stop criteria.

**Output.** A sequence of exposure stages with recorded decisions at each gate.

**Rationale.** A single switch from zero to full traffic concentrates risk. Staged exposure
allows the team to detect a failure while the blast radius is small, and it generates evidence at
each stage. The stages are not ceremonial; each one answers a different question: does it run,
does it serve, does it help, does it hold at scale.

The plan must state who can stop the rollout, on what signal, and what "stop" means (pause,
rollback, or disable the model while keeping the service up). A stop criterion that requires a
committee meeting is not a stop criterion.

Branch point: if the change is a threshold adjustment rather than a new model, the rollout can be
faster, but it still needs a stop rule and a communication plan because the clinical effect can
be immediate and large.

### Step 9 — Operate the inner loop: service health

**Input.** Telemetry from the serving runtime.

**Output.** SLO compliance, error budget consumption, and alerts routed to an owner.

**Rationale.** The inner loop protects the workflow from unavailability, latency, and errors. It
must be measured at the point the user experiences, not only at the process boundary. A service
that returns 200 in 40 ms but computes features from a stale cache is healthy by inner-loop
metrics and broken in fact.

Every alert in this loop needs a stated precision and an owner. Precision is the fraction of
alerts that correspond to a real, actionable condition. An alert with 10 percent precision
consumes nine investigations for every real event; at three in the morning, that is how on-call
engineers learn to ignore alerts.

Branch point: if traffic is too low to distinguish a real error-rate increase from noise, use a
minimum-failure-count condition or synthetic traffic. Do not lower the threshold until the alert
fires; that converts a precision problem into a recall problem without improving either.

### Step 10 — Operate the outer loop: model and outcome health

**Input.** Predictions, delayed labels, override records, subgroup attributes, and data-quality
metrics.

**Output.** A periodic model health review with calibration, discrimination, subgroup gaps,
alert burden, and a disposition.

**Rationale.** The outer loop answers a different question: is the model still useful for the
decision it was built for. It must be scheduled even when nothing appears wrong, because the
absence of alerts is not evidence of model health. A monthly or quarterly review with a fixed
agenda is the minimum.

Label latency determines how long the team is blind. If the outcome is available in 24 hours, the
outer loop can run weekly. If the outcome is available in 90 days, the team needs proxy signals
and an audit sample, and must say so in the model card.

Branch point: if labels never arrive, the model cannot be performance-monitored directly. The
review must rely on proxies (prediction distribution, override rate, alert volume, data
completeness) and must state that a proxy is not the outcome.

### Step 11 — Decide whether to retrain, recalibrate, revise, or retire

**Input.** The model health review, drift statistics, incident history, and the cost of
revalidation.

**Output.** A decision record with the chosen action, the evidence, the owner, and the regulatory
assessment.

**Rationale.** Retraining is one of five responses, and it is often the wrong one. The decision
must follow the diagnosis. Input drift with stable calibration may need no model change. Label
shift may need recalibration. Concept drift may need retraining. A workflow change may need a
human-factors fix. A cost change may need retirement.

The decision table below is the core of this step. It has more than two outcomes because the
evidence pattern, not the severity alone, determines the response.

| Evidence pattern | Most likely cause | First response | Model change required | Regulatory assessment |
|---|---|---|---|---|
| Input distribution shift, calibration stable, performance stable | Population or coding change with preserved relationship | Investigate source, document, continue monitoring | Usually no | Document as post-market surveillance |
| Input shift plus calibration drift, discrimination stable | Case-mix change or prevalence change | Recalibrate on recent labelled data | Yes, recalibration | Assess whether covered by PCCP or change protocol |
| Discrimination falls on delayed labels, inputs stable | Concept drift or label definition change | Retrain and revalidate | Yes, new version | New validation evidence, change control |
| Alert volume falls with stable census and stable inputs | Upstream timing or missing data, not model drift | Fix pipeline, backfill, re-run verification | No model change, data fix | Incident record, possible field action assessment |
| Subgroup gap widens while aggregate stable | Population change or measurement bias | Stratified analysis, local validation | Possibly | Equity review, updated labelling |
| Service error budget exhausted by latency | Capacity or dependency issue | Scale, optimize, or roll back | No | Operational incident |

Branch point: if the evidence is ambiguous, the default is not "retrain." The default is
"investigate and hold," with an explicit date by which the investigation must conclude. Retraining
under ambiguity adds a change to a system whose behaviour is not understood.

### Step 12 — Revalidate and record the change

**Input.** The new or recalibrated artifact and the change description.

**Output.** A change record, updated model card, updated validation report, and a deployment
decision.

**Rationale.** A model change in a regulated context is a change to a device. The record must
allow a reviewer to reconstruct what changed, why, what evidence supported it, and who approved
it. In a non-regulated context, the same record is what allows the next engineer to avoid
repeating an investigation.

The record must cover the whole package, not only the weights. A dependency upgrade, a threshold
change, a feature-definition change, and a base-image change are all changes that can alter
output.

Branch point: if the change is not described in an authorized PCCP or equivalent change protocol,
treat it as requiring regulatory review until counsel says otherwise. The cost of asking is
small; the cost of an unauthorized change to a cleared device is not.

### Step 13 — Retire the model and the service

**Input.** A decision that the model is no longer used, no longer effective, or no longer
affordable.

**Output.** A retirement record: date, reason, replacement, data retention decision, and
communication to users.

**Rationale.** Retirement is a change with safety implications. A clinician who relied on an alert
must know it has stopped. An auditor must know when the model left service. A data retention
obligation may require keeping predictions and audit logs beyond the service life.

Retirement also frees monitoring attention and security review capacity. A portfolio of zombie
models is a hidden cost and a hidden risk: an unmaintained service with access to clinical data
is an attack surface with no owner.

Branch point: if the model is replaced, run the replacement in shadow or canary before switching
off the old one. Do not create a gap in coverage. If the model is withdrawn without replacement,
the workflow must revert to its pre-model process, and that process must be documented and
staffed.

### 5.14 Edge cases the procedure must handle

The following conditions are not exceptions to the procedure; they are the conditions under
which the procedure is actually run. Each changes a step's output and must be planned for.

**Schema drift.** A source adds, removes, renames, or retypes a field. The data contract test
should fail before the model sees the data. If the model serves anyway, it must reject the
request or fall back explicitly, never silently impute a default that was never validated.

**Late data.** A feature arrives after the scoring moment. The model must use the value available
at decision time, not the value that would have been available in a retrospective extract. Late
data also breaks monitoring that assumes complete windows; use event-time windows and watermark
the incompleteness.

**Partial rollout.** Some instances run version A and some version B. Monitoring must be
segmented by version, or the aggregate will hide a regression in the new version. Rollback must
be able to target the version, not the whole deployment.

**Model and feature skew.** The serving pipeline computes a feature differently from training.
Detect with parity tests on fixed inputs, distribution comparisons between training and serving
logs, and a canary that compares outputs from the reference implementation. A single parity test
at deployment is not enough; skew can be introduced by an upstream dependency later.

**Clock skew.** Device time, application time, and database time disagree. A prediction may be
attributed to the wrong hour, a feature window may include future data, or an audit trail may
show an impossible sequence. Record event time, ingestion time, and decision time separately, and
monitor their offsets.

**Quota exhaustion.** An application programming interface (API) rate limit or cloud quota is reached. The service may degrade,
queue, or fail. Capacity planning must include the quota, not only the CPU. Alerts must
distinguish "we are at the limit" from "we are broken."

**Dependency rotation.** A credential, certificate, or API key expires or is rotated. The failure
often appears as an authentication error in a downstream call, not as a model error. Rotation must
be tested before the expiry date, with an owner and a calendar entry.

**Regional outage.** A cloud region or availability zone fails. The service must fail over, queue,
or degrade according to a documented plan. A model that requires a single-region feature store
cannot fail over, regardless of how many replicas the API has.

### 5.15 Control points and who owns them

| Control point | Question answered | Owner | Evidence produced |
|---|---|---|---|
| Decision frame | Who acts, when, on what | Clinical product owner | One-page frame |
| Data contract | What arrives, when, in what unit | Data platform owner | Versioned contract and tests |
| Training pipeline | Can the artifact be reproduced | ML engineer | Run record and data hash |
| Validation | Does it work in the deployment population | Data scientist and clinical lead | Validation report |
| Packaging | Is the artifact complete and immutable | Platform engineer | Digest, software bill of materials (SBOM), model card |
| Pre-exposure verification | Does it work on real traffic | ML engineer and QA | Shadow or replay report |
| Rollout | Is exposure controlled and reversible | Release manager | Stage log and decisions |
| Inner-loop monitoring | Is the service healthy | SRE or platform on-call | SLO dashboard and alert history |
| Outer-loop monitoring | Is the model still useful | Data science and clinical owner | Model health review |
| Change control | Is the change authorized | Regulatory and quality | Change record and approval |
| Retirement | Is the model safely out of service | Product owner | Retirement record |

The table is deliberately short. If a control point has no named owner, it has no control. A
committee is not an owner; a committee can approve a decision but cannot be paged.

## 6. Worked example

A 400-bed hospital deploys a deterioration alert. The model scores every admitted adult once per
hour and writes a risk to a queue that the night-shift nurse reviews at 06:30. Patients above a
threshold of 0.18 are placed on the review list. The service is a containerized API behind the
hospital's integration engine, with two replicas, a readiness probe, and a Prometheus endpoint.

The numbers below are a constructed but internally consistent case. Every arithmetic step is
shown so that you can substitute your own volume and costs.

### 6.1 The operating numbers

| Quantity | Value | Source of the value |
|---|---:|---|
| Eligible admissions per day | 1,150 | Admission census minus obstetrics and day surgery |
| Scoring frequency | Hourly | Workflow requirement |
| Predictions per 28-day window | 32,200 | 1,150 × 28 |
| Outcome prevalence in the window | 3.8 percent | 1,224 deterioration events |
| Sensitivity at threshold 0.18 | 0.63 | Local validation, temporal holdout |
| Specificity at threshold 0.18 | 0.94 | Local validation, temporal holdout |
| Availability SLO | 99.5 percent over 28 days | Service owner |
| Latency SLO | 99 percent of requests under 900 ms | Integration engine timeout budget |
| Nurse review time per alert | 4.5 minutes | Time study, 40 observations |
| Fully loaded nurse cost | 95 US dollars per hour | Finance estimate |

### 6.2 Error budget arithmetic

The availability SLO is 99.5 percent over a 28-day window. The error budget is the complement:
0.5 percent of requests may fail.

```text
budget_requests = 32,200 × 0.005 = 161 failed requests
budget_minutes  = 28 × 24 × 60 × 0.005 = 201.6 minutes of full outage equivalent
```

The budget is expressed in two units because two audiences need it. The service owner needs
failed requests. The clinical operations lead needs minutes, because the ward experiences an
outage as a period during which the review list is stale.

Suppose the first seven days produce a 0.6 percent 5xx rate. That is 8,050 requests and 48.3
failures.

```text
burn_rate = observed_error_rate / (1 - SLO) = 0.006 / 0.005 = 1.2
time_to_exhaustion = 28 / 1.2 = 23.3 days
```

A burn rate of 1.2 is not an emergency. It means that if the rate holds, the budget is exhausted
three days before the window ends. Under the multiwindow burn-rate scheme described by the Google
Site Reliability Engineering (SRE) Workbook, a page at 14.4 times burn over one hour would not
fire, and a ticket at 1 times burn over three days would fire only if the rate persisted. The
correct response is a ticket to investigate the 5xx source, not a page.

Now vary the SLO. If the service owner promises 99.9 percent instead of 99.5 percent, the budget
falls to 32.2 requests and the same 0.6 percent error rate is a burn rate of 6.0. The budget is
exhausted in 4.7 days. The same observed behaviour moves from "ticket" to "page" without any
change in the service. This is why an SLO is a product decision, not an engineering preference.

### 6.3 Alert precision and the cost of a false page

At 3.8 percent prevalence, 32,200 predictions imply 1,224 true deterioration events and 30,976
non-events.

```text
true_positives  = 1,224 × 0.63 = 771
false_negatives = 1,224 − 771  = 453
false_positives = 30,976 × 0.06 = 1,859
total_alerts    = 771 + 1,859  = 2,630
precision       = 771 / 2,630  = 0.293
```

Precision is 29.3 percent. Put differently, 71 percent of the nurse's alert-driven reviews find
no deterioration within the window. That is not necessarily a bad model; it is the arithmetic of
a 3.8 percent base rate with 94 percent specificity. It is, however, the number the ward
experiences, and it determines whether the alert survives.

The alert burden is 2,630 / 28 = 93.9 alerts per day, or 81.7 per 1,000 predictions. If the ward
has 12 nurses on the night shift and each review takes 4.5 minutes, the list consumes 422 minutes
of nursing time per night, or 35 minutes per nurse. That is the real capacity constraint, not the
API.

The review cost is:

```text
cost_per_alert   = 4.5 min × (95 USD / 60 min) = 7.125 USD
monthly_review   = 2,630 × 7.125 = 18,735 USD
cost_per_scored  = 18,735 / 32,200 = 0.582 USD
```

Compare that with the serving cost. Two replicas at 0.25 virtual CPU (vCPU) and 0.25 GiB each,
priced at 0.031 USD per vCPU-hour and 0.004 USD per GiB-hour:

```text
cpu_month  = 2 × 0.25 × 0.031 × 730 = 11.32 USD
mem_month  = 2 × 0.25 × 0.004 × 730 =  1.46 USD
platform   = 11.32 + 1.46 + 18 (load balancer) + 40 (monitoring) + 10 (registry) = 80.78 USD
cost_per_inference = 80.78 / 32,200 = 0.0025 USD
```

The clinical review cost is 233 times the compute cost per prediction. Any optimization that
reduces compute while leaving alert volume unchanged is optimizing 0.4 percent of the cost. This
is the single most common economic error in model serving: teams instrument the GPU bill and not
the human review bill.

### 6.4 Retrain cadence and drift threshold

The team monitors population stability index (PSI) on four features. For creatinine, the
reference window is the training cohort. PSI has been rising at approximately 0.035 per month
since go-live and stood at 0.06 at the last review.

```text
months_to_threshold = (0.20 − 0.06) / 0.035 = 4.0 months
```

A PSI of 0.20 is the team's review threshold, not a universal rule. The threshold was chosen
because it corresponds historically to a creatinine distribution shift large enough to move the
calibration slope below 0.90 in local audits. That is a defensible, local, evidence-based
justification. A threshold copied from a blog post is not.

Retraining and revalidation take six weeks of elapsed time: two weeks to prepare data and train,
two weeks for clinical review and validation, two weeks for quality and regulatory sign-off. If
the team waits until PSI reaches 0.20 to start, the model will be operating past the threshold
for six weeks. Therefore the trigger is set at 0.14, not 0.20:

```text
trigger = 0.20 − (6 weeks / 4.3 weeks per month) × 0.035 = 0.20 − 0.049 = 0.151
```

The team rounds to 0.14 to leave margin. This is the general shape of a retrain trigger: the
monitoring threshold minus the drift expected during the retraining lead time. A trigger set at
the harm threshold guarantees that the model is past the harm threshold for the entire
remediation.

The annual cost of the cadence is four cycles at 15 person-days each, at 1,900 USD per loaded
person-day:

```text
annual_retrain = 4 × 15 × 1,900 = 114,000 USD
```

That is six times the annual compute cost and six times the annual review cost. It is the
dominant maintenance cost, and it is the one most often left out of a business case.

### 6.5 Diagnostic one: the alert volume falls

At week 19, alert volume falls 31 percent week over week while the admission census is flat. The
initial reaction is relief: the ward is less burdened.

Two checks contradict the relief.

First, the prediction distribution has shifted left. The mean risk fell from 0.21 to 0.17, and
the share of scores above 0.18 fell from 12.4 percent to 8.6 percent. A genuine improvement in
population health would not happen in one week with a flat census.

Second, the creatinine missingness rate rose from 0.4 percent to 11.2 percent, and the p95
arrival delay for the creatinine result moved from 22 minutes to 3 hours 40 minutes after a
laboratory interface upgrade. The model imputes the cohort median when creatinine is missing, and
the median is below the values of the patients who would have triggered.

An audit sample of 200 recent admissions found sensitivity had fallen from 0.63 to 0.51. This is
not concept drift. It is late data producing training-serving skew at serving time. The correct
response is to fix the interface, backfill the missing values where possible, re-run verification,
and communicate the detection gap to the clinical owner. Retraining on the degraded data would
teach the model to compensate for a pipeline defect and would embed the defect in the artifact.

### 6.6 Diagnostic two: precision falls, sensitivity holds

At week 31, precision falls from 29.3 percent to 19.9 percent while sensitivity remains at 0.63.
The alert volume rises to 3,869 per month. The cause is a laboratory calibration change that
lowered reported creatinine values by a mean of 0.12 mg/dL across the health system. With the
same sensitivity and a specificity of 0.90 instead of 0.94:

```text
false_positives = 30,976 × 0.10 = 3,098
total_alerts    = 771 + 3,098  = 3,869
precision       = 771 / 3,869  = 0.199
extra_reviews   = 3,869 − 2,630 = 1,239
extra_cost      = 1,239 × 7.125 = 8,829 USD per month
```

The model did not change. The measurement system did. The response is a data-contract fix and a
recalibration decision, not an immediate retrain. If the calibration change is permanent, the
model's coefficient on creatinine is now mis-scaled and recalibration is required; that
recalibration is a model change with a change record.

### 6.7 Vary one assumption: review time

Suppose the time study was wrong and review takes 2.5 minutes rather than 4.5, because the nurse
uses a structured checklist.

```text
cost_per_alert = 2.5 × (95 / 60) = 3.96 USD
monthly_review = 2,630 × 3.96 = 10,408 USD
cost_per_scored = 10,408 / 32,200 = 0.323 USD
```

The review cost falls by 44 percent, and the break-even precision for the programme changes. If
the clinical owner's willingness to pay is 0.40 USD per scored patient, the programme is now
viable at a precision of about 10 percent rather than 15 percent. The model did not change; the
workflow did. This is why cost and capacity work (Topic 10) belongs inside the production loop
rather than after it.

The same variation changes the threshold decision. A lower review cost makes a lower threshold
affordable, which raises sensitivity and alert volume. The operating point is an economic choice
as much as a clinical one, and it must be re-evaluated when either the model or the workflow
changes.

### 6.8 What the example establishes

Four numbers, not one, determine whether this deployment is healthy: the error budget burn rate,
the alert precision, the retrain lead time relative to drift rate, and the cost per decision. Each
is computed from a different data source, owned by a different function, and fails in a different
way. A dashboard that shows only the first is an availability dashboard, not a model health
dashboard.

The example also shows why the loop needs both diagnostics. Diagnostic one was invisible in every
service metric: latency, error rate, and uptime were normal. Diagnostic two was visible in alert
volume but would have been misread as model drift without the laboratory calibration record. In
both cases, the causal evidence lived outside the model.

## 7. Code, query, or calculation

The first script reproduces every number in Section 6 from the inputs. It uses only the standard
library, so it runs in the same environment as the reference service without installing anything.

```python
"""Production-loop arithmetic for the worked example. Standard library only."""

from math import ceil

predictions_per_day = 1_150
window_days = 28
requests = predictions_per_day * window_days
availability_slo = 0.995
error_budget = requests * (1 - availability_slo)
budget_minutes = window_days * 24 * 60 * (1 - availability_slo)

observed_error_rate = 0.006
burn_rate = observed_error_rate / (1 - availability_slo)
days_to_exhaustion = window_days / burn_rate

prevalence = 0.038
sensitivity = 0.63
specificity = 0.94
positives = requests * prevalence
negatives = requests - positives
true_positives = positives * sensitivity
false_negative = positives - true_positives
false_positives = negatives * (1 - specificity)
alerts = true_positives + false_positives
precision = true_positives / alerts

review_minutes = 4.5
loaded_hourly = 95.0
cost_per_alert = review_minutes * loaded_hourly / 60
monthly_review = alerts * cost_per_alert
cost_per_scored = monthly_review / requests

replicas, vcpu_each, gib_each = 2, 0.25, 0.25
vcpu_hour, gib_hour, hours = 0.031, 0.004, 730
platform = (replicas * vcpu_each * vcpu_hour * hours
            + replicas * gib_each * gib_hour * hours
            + 18 + 40 + 10)
cost_per_inference = platform / requests

psi_now, psi_threshold, psi_per_month = 0.06, 0.20, 0.035
lead_time_months = 6 / 4.3
trigger = psi_threshold - lead_time_months * psi_per_month

burst_minutes = 20
arrival_rate = predictions_per_day / (burst_minutes * 60)
service_time = 0.045
offered_load = arrival_rate * service_time
capacity_per_replica = 1 / service_time
replicas_needed = ceil(offered_load / (0.70 * capacity_per_replica * 1))

print(f"requests_28d         {requests}")
print(f"error_budget         {error_budget:.0f} requests")
print(f"budget_minutes       {budget_minutes:.1f}")
print(f"burn_rate            {burn_rate:.2f}")
print(f"days_to_exhaustion   {days_to_exhaustion:.1f}")
print(f"alerts               {alerts:.0f}")
print(f"precision            {precision:.3f}")
print(f"alerts_per_day       {alerts / window_days:.1f}")
print(f"cost_per_alert       {cost_per_alert:.2f} USD")
print(f"monthly_review       {monthly_review:.0f} USD")
print(f"cost_per_scored      {cost_per_scored:.3f} USD")
print(f"platform_month       {platform:.2f} USD")
print(f"cost_per_inference   {cost_per_inference:.4f} USD")
print(f"retrain_trigger_psi  {trigger:.3f}")
print(f"arrival_rate         {arrival_rate:.3f} req/s")
print(f"offered_load         {offered_load:.4f} erlangs")
print(f"replicas_needed      {max(2, replicas_needed)}")

assert abs(error_budget - 161.0) < 1e-9
assert abs(burn_rate - 1.2) < 1e-9
assert abs(precision - 0.293) < 0.001
assert abs(cost_per_scored - 0.582) < 0.001
assert abs(trigger - 0.151) < 0.001
```

Expected output, confirmed on Python 3.11:

```text
requests_28d         32200
error_budget         161 requests
budget_minutes       201.6
burn_rate            1.20
days_to_exhaustion   23.3
alerts               2629
precision            0.293
alerts_per_day       93.9
cost_per_alert       7.12 USD
monthly_review       18735 USD
cost_per_scored      0.582 USD
platform_month       80.78 USD
cost_per_inference   0.0025 USD
retrain_trigger_psi  0.151
arrival_rate         0.958 req/s
offered_load         0.0431 erlangs
replicas_needed      2
```

The `7.12` is the display rounding of 7.125. The capacity result shows that 2 replicas are
sufficient: the offered load is 0.043 erlangs, and a single replica could serve about 15.5
requests per second. Review time, not compute, is the binding constraint.

A canary gate applies the same arithmetic to a release decision: compare the treatment
error rate with the control error rate, require a minimum sample before deciding, and treat
insufficient data as a distinct state rather than as approval. Exercise 12.7 asks you to build
and test one.

The reference service in `labs/mlops/` implements the same loop in miniature. `app/main.py`
exposes `/healthz`, `/readyz`, `/predict`, and `/metrics`; `monitoring/prometheus-rules.yaml`
contains the error-rate and latency alerts; and `monitoring/drift.py` computes PSI for a numeric
feature. The lab's alert thresholds (2 percent 5xx over 10 minutes, p99 above 2 seconds) are
teaching defaults, not tuned production values. A real service derives them from an SLO and
measures their precision over time, as Section 6 does.

## 8. Data, standards, and regulatory context

The production loop is described in engineering language, but in a regulated product several of
its steps are also regulated activities. This section separates what is binding from what is
convention, because teams routinely over-apply voluntary frameworks and under-apply binding ones.

### 8.1 The instruments that touch the loop

| Instrument | Status | What it constrains in the loop | What it does not constrain |
|---|---|---|---|
| FDA Good Machine Learning Practice (GMLP), 2021 | Non-binding guiding principles, jointly issued with Health Canada and the UK MHRA | Expects multidisciplinary expertise, representative data, independent test sets, clinically relevant testing, user information, and post-deployment monitoring | Does not prescribe architecture, metrics, thresholds, or a specific monitoring cadence |
| FDA PCCP guidance, final December 2024, reissued August 2025 | Guidance implementing FD&C Act section 515C, added by FDORA in December 2022 | Describes the modifications, the modification protocol, and the impact assessment that may be pre-authorized | Does not authorize changes outside the described protocol; does not eliminate the need for a submission for a new intended use |
| FDA AI-enabled device software functions lifecycle guidance, draft January 2025 | Draft, not for implementation until finalized | Recommends a postmarket performance monitoring plan, risk assessment, data management, validation, and cybersecurity content in a submission | Draft status means it is not a final statement of FDA expectations |
| IEC 62304:2006/Amd 1:2015 | Binding when invoked by the quality system or a regulator | Software lifecycle processes, software safety classification, maintenance process (clause 6), risk management of changes (7.4), configuration management and change control (8.2) | Does not specify how to monitor a deployed model's statistical performance |
| ISO 14971:2019 | Binding when invoked; the risk management standard for medical devices | Hazard identification, risk control, and production and post-production monitoring (clause 10) | Does not prescribe statistical methods or alert thresholds |
| ISO/IEC 42001:2023 | Voluntary management system standard, certifiable | AI management system: context, leadership, risk and impact assessment, operational control, internal audit, improvement | Certification is not a substitute for device regulation or clinical evidence |
| NIST AI Risk Management Framework 1.0, 2023 | Voluntary | Govern, Map, Measure, Manage functions; a vocabulary for AI risk across the lifecycle | Not a compliance regime and not specific to medical devices |
| EU AI Act, Regulation (EU) 2024/1689 | Binding in the EU, with a phased timetable | Risk classification, transparency, post-market monitoring, and quality management for high-risk systems | Medical device overlap is handled through sectoral law; timelines have been amended, so verify the current dates |
| HIPAA Security Rule, 45 CFR 164.312(b) | Binding for covered entities and business associates in the US | Audit controls that record and examine activity in systems containing electronic protected health information | Does not specify log format, retention beyond the documentation rule, or model-specific events |
| 21 CFR Part 11 | Binding for electronic records in FDA-regulated activities when predicate rules apply | Secure, computer-generated, time-stamped audit trails; records must not obscure prior information | FDA has stated enforcement discretion on some audit-trail specifics, but predicate rules still apply |

### 8.2 What the PCCP actually permits

The PCCP is the instrument most often misunderstood in deployment planning. It is not a general
permission to update a model. It is a pre-authorized description of specific modifications,
executed under a protocol, with an impact assessment. The final guidance describes three
interrelated parts: the description of modifications, the modification protocol, and the impact
assessment.

For the production loop, this means that Step 11 (decide whether to retrain) and Step 12
(revalidate and record) must be written before deployment if the team intends to update the model
without a new marketing submission. The protocol must specify the data management, retraining
practices, performance evaluation, and update procedures. A team that discovers this requirement
after building an automated retraining pipeline has built a pipeline it may not be allowed to
use.

The scope of a PCCP is a legal and regulatory determination. The table in Section 5.11 is an
engineering heuristic, not a legal opinion. The conservative rule from the survey file still
applies: if the change is not described in the PCCP, assume it needs review.

### 8.3 What IEC 62304 adds to the engineering picture

IEC 62304 classifies software by the severity of harm that a failure could cause, from class A
(no injury possible) to class C (death or serious injury possible). The classification drives the
rigour of the lifecycle processes. For a deployed model, the practical consequences are:

- The maintenance process in clause 6 requires a maintenance plan, problem and modification
  analysis, and modification implementation. A retraining event is a modification.
- Clause 7.4 requires risk management of software changes. A model update that changes the output
  distribution is a change that must be risk-assessed, even if the code is unchanged.
- Clause 8.2 requires configuration management and change control. The model artifact, the
  preprocessing, the container, and the configuration must be under configuration control as a
  set.

None of this says how to compute PSI. That is the point: the standard constrains process, not
statistics. A team can be fully compliant with IEC 62304 and still deploy a model with no
performance monitoring, provided its risk file says the residual risk is acceptable. The reverse
is also true: excellent monitoring does not make an undocumented change compliant.

### 8.4 What the audit and records rules require

HIPAA's audit control standard requires mechanisms that record and examine activity in systems
that contain or use electronic protected health information. The HHS audit protocol tests whether
the entity determined what would be audited, selected tools, wrote a review policy, and wrote
standard operating procedures. For a model service, the practical implication is that prediction
events need an audit record sufficient to reconstruct who or what was scored, when, with which
model version, and what was returned, without recording more identifiers than necessary.

Part 11 imposes a stronger standard where it applies: audit trails must be secure, computer
generated, time stamped, and must not obscure prior information. The record must be retained at
least as long as the underlying record and be available to the agency. A model service that
overwrites a prediction row when a score is recalculated fails this requirement in substance,
even if the database is technically append-only at the storage layer.

Documentation retention under HIPAA is six years from creation or last effective date, whichever
is later (45 CFR 164.316). Model cards, validation reports, and change records should be scheduled
against that clock, not against the engineering team's retention preference.

### 8.5 What is convention rather than requirement

Several widely repeated practices are conventions with no binding force:

- A PSI threshold of 0.1 or 0.25. These are credit-scoring rules of thumb. Yurdakul and Naranjo
  show that PSI's distribution depends on sample size and bin count, and that the conventional
  cutoffs have no known error properties. A threshold is defensible only when tied to a locally
  observed harm.
- A calibration slope floor of 0.8. This is a sensible review trigger, not a regulatory limit.

> **Note:** Regulatory status changes. The EU AI Act timetable in particular was amended in 2026,
> and the FDA lifecycle guidance was still a draft as of October 2026. Any deployment plan should
> record the version and date of each instrument it relies on, and re-check before submission.

## 9. Industry practice and case evidence

This section separates what is documented in public sources from what is inference. The
distinction matters because deployment folklore is abundant and evidence is not.

### 9.1 The Epic Sepsis Model: documented facts

Wong and colleagues published an external validation of the Epic Sepsis Model in JAMA Internal
Medicine in 2021. The study covered 27,697 patients across 38,455 hospitalizations at Michigan
Medicine between December 2018 and October 2019. Sepsis occurred in 2,552 hospitalizations, or
7 percent.

The documented findings:

- Hospitalization-level area under the receiver operating characteristic curve (AUC) was 0.63
  (95 percent confidence interval 0.62 to 0.64), against 0.76 to 0.83 reported by the vendor.
- At the alert threshold of 6, alerts fired for 6,971 of 38,455 hospitalizations, or 18 percent.
- At that threshold, sensitivity was 33 percent, specificity 83 percent, positive predictive value
  12 percent, and negative predictive value 95 percent.
- The model failed to identify 1,709 of 2,552 sepsis cases, or 67 percent.
- The number needed to evaluate was 8 at the hospitalization level.

These are facts about one external validation at one health system. They are not a claim that the
model is ineffective everywhere, and they are not a claim about all sepsis models.

### 9.2 The Epic Sepsis Model: what the case demonstrates about the loop

The case is a natural experiment in what happens when the outer loop is missing. The vendor's
reported performance and the deploying system's observed performance diverged, and the divergence
was discovered by an independent retrospective study rather than by the deployment's own
monitoring. The alert burden of 18 percent of hospitalizations is an inner-loop and workflow
signal that was visible in operations but was not treated as evidence about the model.

Three inferences follow. They are inferences, not findings of the study:

1. A vendor's internal validation is rung 3 evidence on the ladder in Section 3.3. It cannot
   substitute for rung 4 or 5 evidence at the deploying site.
2. Alert burden is a model performance metric, not only an operational annoyance. At 18 percent,
   the positive predictive value was 12 percent, which is the arithmetic consequence of a low
   base rate and imperfect specificity.
3. A deployment without a scheduled external validation has no mechanism to detect this class of
   divergence. The failure is structural, not a matter of a better model.

### 9.3 IDx-DR: documented facts and the workflow lesson

IDx-DR received De Novo authorization in 2018 (DEN180001) as a system that analyzes retinal
images to detect diabetic retinopathy in primary care. The authorization described a defined
workflow, including image-quality checks that cause the system to reject unusable images rather
than return a result.

The documented lesson is that the image-quality gate is part of the product, not a preprocessing
detail. The inference for the production loop is that a model's operating envelope can be
enforced in software: rejecting an out-of-envelope input is a valid output, and it must be
monitored as a distinct outcome. A service that silently scores poor-quality input has removed a
safety control that the validation assumed.

### 9.4 NHS deployment guidance: documented position

NHS England's AI guidance and the associated assurance material emphasize local validation,
integration with clinical workflow, workforce support, and ongoing monitoring. This is a
documented policy position, not a study result. Its relevance to the loop is that a national
payer explicitly treats deployment as a sociotechnical process rather than a procurement event.

The inference is that the outer loop has more stakeholders than the model owner: the trust, the
clinical safety officer, and the patients.

### 9.5 Reliability practice: error budgets and canaries

The Google SRE Workbook documents multiwindow, multi-burn-rate alerting and canary release
practice. The specific published starting parameters for a 99.9 percent SLO are a page at 14.4
times burn over one hour and five minutes, a page at 6 times burn over six hours and thirty
minutes, and a ticket at 1 times burn over three days and six hours. These are documented
recommendations from one organization's practice, not a standard.

The inference for clinical AI is that the arithmetic transfers but the thresholds do not. A
clinical service with a 99.5 percent availability SLO and 32,200 requests per month has an error
budget of 161 requests; the same burn-rate structure applies, but a page at 14.4 times burn may
be too slow if the workflow has no fallback. The transferable part is the method: define the
budget, compute burn, and set alert windows from the cost of delay.

## 10. Failure modes and red flags

The failures below are ordered roughly by how often they are detected late. Each row names the
signal that appears first, because the early signal is usually not the one the team is watching.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Silent false-negative increase | Alert volume falls while census is stable | A feature arrives after the scoring cutoff, so the model scores an incomplete picture | Missed deteriorations with no service error | Contract test on arrival time, feature-freshness monitor, audit sample with recomputed sensitivity |
| Alert fatigue | Override rate rises before precision falls | Specificity is too low for the base rate, or the threshold was set from AUROC rather than from workflow capacity | Clinicians ignore the alert, including true positives | Set the threshold against review capacity, monitor precision and override rate, revisit quarterly |
| Training-serving skew | Offline and online feature distributions diverge | Different code path or time window computes the feature in training and serving | Predictions are systematically shifted, often plausibly | Parity tests on fixed inputs, shared feature definitions, distribution comparison on logs |
| Stale feature cache | Feature age p99 rises while latency stays flat | A cache refresh job fails and the service serves the last known value | Model scores old physiology as if current | Freshness metric per feature, cache age alert, readiness check that includes data age |
| Unowned alert | Alerts route to a shared inbox or a departed employee | Ownership was never assigned or was lost in a reorg | Detection exists but response does not | Named owner per alert, escalation policy, quarterly ownership review |
| Incompatible rollback | Previous image requires a schema the database no longer has | Image and model artifact are versioned separately, or a migration is irreversible | Rollback fails during the incident | Version the package as a unit, test rollback quarterly, make migrations backward compatible |
| Silent model update | Model version changes without a change record | Auto-retraining writes a new artifact to the serving path | An unaudited change to a regulated device | Immutable artifacts, registry promotion, deployment gate that rejects unapproved versions |
| Schema drift accepted silently | 422 rate falls because validation was loosened | A new field or unit is coerced rather than rejected | Wrong units produce plausible but wrong scores | Strict schema with explicit unit fields, contract tests, reject-and-alert on unknown units |
| Late-label blindness | No performance review has run for two quarters | Outcome data arrives after the review cadence, so the review is skipped | Drift is undetected for months | Proxy monitoring with stated limits, scheduled audit sample, label-latency metric |
| Proxy fixation | Team reports override rate as if it were accuracy | A proxy is convenient and the outcome is hard to obtain | The proxy improves while the outcome worsens | State what each proxy can and cannot detect, pair with periodic outcome audit |
| PHI in logs | Log volume grows after a debugging change | Structured logging includes request bodies or identifiers | Privacy breach and audit finding | Log schema review, redaction tests, egress monitoring |
| Calendar retraining | Retrain runs on schedule with no evidence | The cadence is a habit rather than a response to drift | Churn, revalidation cost, and change fatigue with no benefit | Evidence-triggered retraining, documented decision record |
| Orphaned model | No one can name the owner of a deployed model | Reorg, departure, or pilot that was never closed | Unpatched service with data access and no accountability | Model inventory with owner, annual recertification, retirement process |
| Unbounded envelope | New sites adopt without local validation | The model was validated at one site but sold as general | Performance loss in new populations, equity harm | Envelope in the model card, site-specific validation gate, subgroup monitoring |
| Quota exhaustion | 429 rate rises at peak | Capacity plan counted CPU but not API quotas | Outage at the busiest hour | Quota in capacity model, load test at peak, alert on quota utilization |

The red flags that deserve escalation even without a metric:

- A dashboard with no owner and no threshold.
- A model card whose evaluation section predates the last data-source change.
- A rollback procedure that has never been executed outside production.
- A retraining job whose output is promoted automatically.
- An alert whose runbook says "investigate" and nothing else.
- A vendor claim of validation with no statement of the validation population.

## 11. Assumptions and limits

Every monitoring design embeds assumptions. When an assumption fails, the monitor can remain
green while the model degrades. The ledger below makes the assumptions testable.

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Outcomes will arrive within the review cadence | Performance monitoring needs labels | Claims lag, outpatient follow-up, data-sharing delays | Measure label latency distribution monthly; compare to cadence |
| The deployed population stays inside the validated envelope | Performance evidence transfers only inside the envelope | New site, new payer, new age range, new language | Compare case-mix distributions; gate new sites on local validation |
| Service telemetry reflects the user experience | SLOs are computed from service metrics | Caching, queueing, integration-engine retries, stale data | Correlate service latency with clinician-reported delay; synthetic probes end to end |
| Alert precision can be estimated | Tuning requires knowing the false-positive cost | No outcome data, no override capture | Audit a random sample of alerts; record override reasons |
| Retraining on recent data improves performance | The retrain response assumes the relationship is learnable from recent data | Concept drift with rare events, label noise, feedback loops | Validate the retrained model on a future window, not a random split |
| Rollback restores the previous behaviour | Incident response depends on reversibility | Irreversible migration, external dependency change, cached state | Quarterly rollback drill with prediction comparison |
| Versioning covers the whole package | Reproducibility and audit depend on it | Config in a separate system, secrets rotated, base image floating | Rebuild from the record and compare outputs |
| Cost estimates include human review | The dominant cost is often clinical time | Finance counts compute only | Time study of review; compare to compute cost |
| Monitoring does not change behaviour | Metrics are treated as observations | Alerts change clinician behaviour, creating a feedback loop | Compare alert and no-alert periods; monitor intervention rates |
| The vendor's change control matches the contract | Third-party components can change underneath | Silent model or firmware update | Contractual notification clause, version pinning, change detection |
| Logs are representative of all traffic | Sampling is used to control cost | Sampling by status code or by site | Compare sampled and unsampled aggregates; stratify samples |
| The error budget maps to clinical harm | SLOs prioritize engineering work | A 0.5 percent outage may hit the sickest patients | Map outages to workflow impact; weight by criticality |
| Drift statistics are comparable across windows | Trend monitoring assumes a stable reference | Reference window updated silently, bin edges recomputed | Version the reference distribution; freeze bin edges |
| The team can detect a regression within one review cycle | The loop assumes timely feedback | Low event rate, slow label arrival | Compute statistical power for the expected effect size |
| Regulatory status is unchanged | The change plan depends on it | New guidance, new classification, new market | Regulatory watch process with dated review |

## 12. Exercises

Eight exercises, ordered from recall to design. Each states a deliverable, acceptance criteria,
and an estimated time. Exercises 2, 3, and 5 have numeric answers in Section 13; the design
exercises are assessed against the rubric in Section 13.6.

### Exercise 12.1 — Draw and annotate the loop

Take a model you have seen deployed, or the reference service in `labs/mlops/`. Draw the seven
stages and four feedback edges from memory. For each stage, write the artifact it produces, the
owner, and one metric that would reveal a failure at that stage.

**Deliverable.** A one-page diagram with a table of stage, artifact, owner, metric.

**Acceptance criteria.** All seven stages present; at least three feedback edges labelled with
the signal they carry; every stage has a named owner role (not a team name); at least one metric
distinguishes service health from model health.

**Time.** 45 minutes.

### Exercise 12.2 — Compute an error budget and burn rate

A service scores 2,400 patients per day. The availability SLO is 99.0 percent over a 30-day
window. In the first six days, 1.4 percent of requests return a 5xx error.

Compute the request count for the window, the error budget in requests, the budget in minutes of
full outage, the current burn rate, and the projected days to exhaustion. State whether a page at
14.4 times burn over one hour would fire.

**Deliverable.** Five numbers and a one-sentence alert decision, with the arithmetic shown.

**Acceptance criteria.** Units stated; budget computed from the window, not from a month of 28
days; burn rate computed against the SLO complement; the alert decision follows from the
arithmetic rather than from intuition.

**Time.** 30 minutes.

### Exercise 12.3 — Set a retrain trigger

A feature's PSI is 0.08 today and has been rising 0.04 per month. The review threshold is 0.25.
Retraining and revalidation take eight weeks of elapsed time. Compute the trigger value at which
the team should start retraining so that the new model is ready before the threshold is crossed.

**Deliverable.** The trigger value and a short justification of the lead-time conversion.

**Acceptance criteria.** Weeks converted to months consistently; trigger below the threshold;
the answer explains why starting at the threshold is too late.

**Time.** 25 minutes.

### Exercise 12.4 — Critique a monitoring plan

A deployment plan contains the following monitoring section: "We will monitor uptime, CPU, memory,
and error rate. Alerts will fire if uptime falls below 99 percent. The model will be retrained
annually. A dashboard will be reviewed monthly by the data science team."

Write a critique of at most 600 words. Identify what the plan can detect, what it cannot detect,
and the three changes with the highest expected value.

**Deliverable.** A structured critique with a table of signal, detects, does not detect.

**Acceptance criteria.** Distinguishes service from model monitoring; names at least two model
failure modes the plan cannot detect; each recommended change has an owner and a cost; no
recommendation is "add more metrics" without a decision it supports.

**Time.** 50 minutes.

### Exercise 12.5 — Diagnose a silent degradation

Alert volume at a hospital falls 22 percent over ten days. Census is flat. Latency and error rate
are normal. The mean predicted risk fell from 0.19 to 0.15. The missingness rate for one laboratory
feature rose from 0.6 percent to 9.8 percent, and its p95 arrival delay rose from 18 minutes to
2 hours 50 minutes after an interface upgrade. An audit of 150 admissions found sensitivity fell
from 0.61 to 0.49.

Name the failure, the mechanism, the first diagnostic you would run, and the response. State
explicitly whether retraining is appropriate.

**Deliverable.** A 400-word diagnosis with a causal chain diagram.

**Acceptance criteria.** Correctly identifies late data rather than concept drift; distinguishes
the model from the pipeline; response includes a data fix and a communication step; explains why
retraining would embed the defect.

**Time.** 40 minutes.

### Exercise 12.6 — Build the ownership table

For a model that predicts no-shows at outpatient clinics, write the control-point table from
Section 5.15 with named roles for a 200-clinic health system. Include the data contract owner for
the scheduling system, the clinical owner for the outreach workflow, and the on-call owner for the
batch job.

**Deliverable.** An eleven-row table with role, decision authority, and evidence produced.

**Acceptance criteria.** Every row has one accountable role; no row is owned by "the team"; the
rollback authority is explicit; the retirement row exists.

**Time.** 45 minutes.

### Exercise 12.7 — Design and test a canary gate

Implement a canary decision function with two modifications: a minimum effect
size of 0.5 percentage points and a requirement that the treatment sample be at least 10 percent
of the control sample. Test it on three cases: a clean release, a regression of 0.8 percentage
points, and a release with 40 treatment requests.

**Deliverable.** The function, three test cases, and the printed decisions.

**Acceptance criteria.** The function returns "hold" for insufficient sample; the regression case
returns "rollback"; the clean case does not return "promote" unless the improvement exceeds the
effect-size floor; the code runs with the standard library only.

**Time.** 60 minutes.

### Exercise 12.8 — Map stages to change-control obligations

Choose one change: adding a new laboratory feature to an existing model. For each stage of the
loop, state what artifact changes, whether the change is covered by a typical PCCP, and what
evidence a reviewer would need. Conclude with a recommendation on whether to proceed under the
existing authorization or seek review.

**Deliverable.** A stage-by-stage table plus a 200-word recommendation.

**Acceptance criteria.** Identifies that a new input feature is usually outside a PCCP; names the
validation and risk-analysis evidence; states the residual uncertainty; recommendation is
conditional on counsel review rather than asserted as legal advice.

**Time.** 55 minutes.

## 13. Self-check answers

### 13.1 Exercise 12.1

A complete answer has seven stages (data, features, model, prediction, workflow, action, outcome),
four feedback edges (quality, drift, performance, safety), and an ownership table. The most common
omission is the outcome stage: learners draw the loop ending at prediction. The second most common
is an owner named as a team, such as "the data team," which is not an accountable role.

A metric that distinguishes service from model health: p99 latency detects a broken service and is
blind to a model whose sensitivity has fallen; calibration slope detects a broken model and is
blind to stale features.

### 13.2 Exercise 12.2

```text
requests        = 2,400 × 30 = 72,000
error_budget    = 72,000 × (1 − 0.99) = 720 requests
budget_minutes  = 30 × 24 × 60 × 0.01 = 432 minutes
burn_rate       = 0.014 / 0.01 = 1.4
days_to_exhaust = 30 / 1.4 = 21.4 days
```

A page at 14.4 times burn over one hour would not fire. The observed rate is 1.4 times the
sustainable rate, which is a ticket-level condition. If the rate persists, the budget is exhausted
on day 21 of 30, leaving nine days with no budget for a second incident. That is the argument for
acting on a 1.4 burn rate even though it does not page: the budget is a shared resource, and
spending 70 percent of it in 40 percent of the window leaves the service exposed.

### 13.3 Exercise 12.3

```text
lead_time_months = 8 / 4.3 = 1.86 months
drift_in_lead    = 1.86 × 0.04 = 0.0744
trigger          = 0.25 − 0.0744 = 0.176
```

Start retraining at a PSI of about 0.18, not 0.25. Starting at the threshold means the model is
already past the review threshold for the entire eight weeks of retraining, and any additional
drift during that period is unmonitored by the retrain process. The 4.3 weeks per month conversion
is an approximation; a team using calendar months should use 0.92 months (8 weeks / 4.35 weeks per
month) and get 0.177, which rounds to the same trigger.

### 13.4 Exercise 12.4

A strong critique contains a table with rows for uptime, CPU, memory, error rate, and the annual
retrain. Uptime detects process and network availability; it does not detect a model returning
plausible but degraded scores. CPU and memory detect capacity pressure; they do not detect data
staleness. Error rate detects failed requests; it does not detect successful requests with wrong
inputs. Annual retraining detects nothing between cycles.

The three highest-value changes are usually: a feature-freshness and missingness monitor with a
contract test; a scheduled performance review on delayed labels with an audit sample; and a named
owner with rollback authority. A rubric for the critique is in Section 13.6.

### 13.5 Exercise 12.5

The failure is late data producing training-serving skew, not concept drift. The causal chain is:
interface upgrade changes arrival timing, creatinine arrives after the feature cutoff, the model
imputes the cohort median, the imputed value is lower than the patient's true value, the risk score
falls, the alert does not fire, and sensitivity falls.

The first diagnostic is a feature-freshness and missingness panel segmented by hour and by source,
joined to the interface change log. The second is a replay of 200 recent admissions using only
data available at the scoring timestamp, compared with a replay using complete data.

Retraining is not appropriate. The training data still reflects the physiological relationship;
the serving pipeline has stopped delivering the inputs. Retraining on the degraded feed would teach
the model to compensate for missing creatinine, embedding the interface defect and making the model
worse when the interface is fixed.

The response is: fix or roll back the interface, backfill where possible, re-run verification,
notify the clinical owner of the detection gap, and add a contract test on arrival time. If patients
were exposed during the gap, assess whether the event is reportable under the applicable quality
system.

### 13.6 Rubric for the design exercises

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Problem framing | 20% | Restates the prompt | Names the decision and the affected population | Quantifies the cost of each error type |
| Evidence | 25% | Asserts conclusions | Cites a metric and a window | Distinguishes measured, inferred, and assumed |
| Failure analysis | 25% | Lists generic risks | Identifies mechanisms and early signals | Ranks by detection latency and designs a test |
| Operational realism | 20% | Ignores ownership and cost | Names owners and a rollback path | Quantifies review load and time to restore |
| Communication | 10% | Dense prose | Clear structure | A busy clinician or auditor can act on it |

## 14. Interview questions

**1. What is the difference between a broken service and a broken model?**
Strong answers separate availability, latency, and error rate from calibration, discrimination,
and subgroup performance, and give an example where one is healthy and the other is not. Weak
answers treat uptime as evidence that the model works.

**2. How would you set an SLO for a clinical prediction service?**
Strong answers start from the workflow's tolerance, name the measurement point (client or server),
state the window, compute the error budget, and explain what the budget funds. Weak answers quote
99.9 percent because it is common.

**3. What is an error budget and what would you spend it on?**
Strong answers define it as 1 minus the SLO times the window, in requests or minutes, and describe
a trade-off such as a risky release or a dependency migration. Weak answers describe it as a
monitoring threshold.

**4. When would you retrain, and when would you not?**
Strong answers distinguish covariate, label, and concept drift, and say that input drift with
stable calibration may need no retrain. Weak answers say "when PSI exceeds 0.2."

**5. How do you monitor a model whose labels arrive after 90 days?**
Strong answers use proxies with stated limits, a scheduled audit sample, prediction distribution,
override rate, and data completeness, and they state the detection delay. Weak answers claim
drift detection substitutes for outcome monitoring.

**6. What belongs in a rollback plan?**
Strong answers include the artifact, config, schema, feature definitions, and secrets as a unit,
a tested procedure, a decision authority, and a time-to-restore target. Weak answers say "redeploy
the previous image."

**7. How would you detect training-serving skew?**
Strong answers propose parity tests on fixed inputs, shared feature definitions, distribution
comparison between training and serving logs, and canary output comparison. Weak answers rely on
a single test at deployment.

**8. What is the first thing you check when alert volume drops?**
Strong answers check census, data freshness, missingness, and the arrival-time distribution before
concluding the population got healthier. Weak answers treat the drop as good news.

## 15. Key terms

- **Artifact** — the immutable package of weights, preprocessing, config, and metadata needed to reproduce a prediction.
- **Burn rate** — observed error rate divided by the SLO's allowed error rate; 1 means the budget lasts exactly the window.
- **Canary** — a limited, time-bound exposure of a change to a subset of traffic with pre-specified evaluation.
- **Champion/challenger** — a current model compared against a candidate under controlled exposure.
- **Change control** — the documented process for approving, recording, and reversing a change.
- **Concept drift** — a change in the relationship between inputs and outcome.
- **Covariate shift** — a change in the input distribution with the input-output relationship unchanged.
- **Data contract** — a versioned specification of fields, types, units, ranges, and arrival guarantees.
- **Drift** — any change over time in inputs, labels, or the input-output relationship.
- **Error budget** — the allowed failure implied by an SLO over a window.
- **Feature freshness** — the age of the data used to compute a feature at scoring time.
- **Label latency** — the delay between a prediction and the availability of its outcome.
- **Model card** — a structured summary of intended use, data, performance, and limits.
- **Model registry** — a versioned store of artifacts with metadata and promotion state.
- **Point-in-time correctness** — the property that a feature reflects only data available at the decision time.
- **Population stability index (PSI)** — a binned divergence measure between a reference and a current distribution.
- **Predetermined change control plan (PCCP)** — a pre-authorized description of permitted device modifications with a protocol and impact assessment.
- **Proxy metric** — a signal used in place of an unavailable outcome, with stated limits.
- **Recalibration** — adjusting the output mapping without refitting the feature relationships.
- **Retraining** — fitting a new model on new or additional data.
- **Revalidation** — re-establishing that the model meets acceptance criteria in the intended population.
- **Rollback** — restoring the previous known-good package as a unit.
- **Runbook** — the procedure for responding to a named alert.
- **Service level indicator (SLI)** — a quantitative measure of service behaviour.
- **Service level objective (SLO)** — a target for an SLI over a stated window.
- **Shadow mode** — scoring live traffic without acting on the output.
- **Silent failure** — a failure with no error signal because outputs remain plausible.
- **Training-serving skew** — any difference between training-time and serving-time data or transformation.

## 16. Sources

All URLs accessed 26 October 2026.

Regulatory and standards

- US Food and Drug Administration, Health Canada, and UK MHRA. *Good Machine Learning Practice for Medical Device Development: Guiding Principles*, October 2021. https://www.fda.gov/media/153486/download
- US Food and Drug Administration. *Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions*. Final guidance, December 2024; reissued August 2025. https://www.fda.gov/media/166704/download
- US Food and Drug Administration. *Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations*. Draft guidance, January 2025. https://www.fda.gov/media/184856/download
- IEC 62304:2006/AMD1:2015. *Medical device software — Software life cycle processes — Amendment 1*. https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019. *Medical devices — Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
- ISO/IEC 42001:2023. *Information technology — Artificial intelligence — Management system*. https://www.iso.org/standard/42001
- National Institute of Standards and Technology. *AI Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, January 2023. https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
- Council of the European Union. *AI Act timeline*. https://www.consilium.europa.eu/en/policies/artificial-intelligence-act/timeline-artificial-intelligence/
- European Commission. *AI Act Service Desk: When does enforcement start?* https://ai-act-service-desk.ec.europa.eu/en/ai-act/faq/when-does-enforcement-start
- 45 CFR 164.312, Technical safeguards, HIPAA Security Rule. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312
- US Department of Health and Human Services. *HIPAA Audit Protocol*. https://www.hhs.gov/hipaa/for-professionals/compliance-enforcement/audit/protocol-edited/index.html
- 21 CFR 11.10, Controls for closed systems. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11/subpart-B/section-11.10

Reliability and engineering practice

- Google. *The Site Reliability Workbook*, Chapter 5, Alerting on SLOs. https://sre.google/workbook/alerting-on-slos/
- Google. *The Site Reliability Workbook*, Chapter 16, Canarying Releases. https://sre.google/workbook/canarying-releases/
- Prometheus. *Histograms and summaries*. https://prometheus.io/docs/practices/histograms/

Peer-reviewed and case evidence

- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*. 2021;181(8):1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
- Yurdakul B, Naranjo J. Statistical properties of the population stability index. *Journal of Risk Model Validation*. 2020;14(4):89-100. https://doi.org/10.21314/JRMV.2020.227
- Gama J, Žliobaitė I, Bifet A, Pechenizkiy M, Bouchachia A. A survey on concept drift adaptation. *ACM Computing Surveys*. 2014;46(4):44. https://doi.org/10.1145/2523813
- US Food and Drug Administration. De Novo classification request DEN180001 (IDx-DR), 2018. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
- NHS England. *Artificial intelligence in health and care*. https://www.england.nhs.uk/ai/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/15_deployment_mlops.md`](../../course/15_deployment_mlops.md)
- Next topic: [`02_deployment-patterns.md`](02_deployment-patterns.md)
- Later topic: [`05_monitoring.md`](05_monitoring.md)
- Related topic: [`06_drift.md`](06_drift.md)
- Labs overview: [`../../labs/`](../../labs/)
- Reference service: [`../../labs/mlops/`](../../labs/mlops/)
- Service code: [`../../labs/mlops/app/main.py`](../../labs/mlops/app/main.py)
- Prometheus rules: [`../../labs/mlops/monitoring/prometheus-rules.yaml`](../../labs/mlops/monitoring/prometheus-rules.yaml)
- Drift script: [`../../labs/mlops/monitoring/drift.py`](../../labs/mlops/monitoring/drift.py)
- Runbook: [`../../labs/mlops/RUNBOOK.md`](../../labs/mlops/RUNBOOK.md)
