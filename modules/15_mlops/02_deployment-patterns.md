# Deployment Patterns

> Module 15 — Deployment and Machine Learning Operations (MLOps)
> Topic 02 of 10
> Estimated study time: 3 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

The deployment pattern is the contract between a model and the workflow it serves. It determines
when the model runs, how long the answer takes, where the computation happens, what happens when
the computation fails, and who sees the output. Teams routinely choose the pattern by fashion
rather than by fit: a real-time application programming interface (API) because it sounds modern, a nightly batch because the team
already has Airflow, an embedded model because a vendor promised offline operation. The wrong
choice is expensive in a way that model quality cannot fix.

A pattern mismatch produces failures that look like model failures but are not. A batch model
that scores at 02:00 and delivers a list at 06:00 cannot support a decision at 14:00 about a
patient in the emergency department. A real-time model with a 2.4-second p99 cannot support a
sepsis alert that the electronic health record (EHR) must render inside a 1.5-second page load. An embedded model with no
update channel cannot be corrected when a calibration error is discovered. In each case the model
may be accurate and the deployment still wrong.

The pattern also determines the cost structure and the monitoring design. Batch scoring
concentrates compute into a window and produces a finite, inspectable output file. Real-time
serving spreads compute across the day, requires capacity for the peak, and produces a stream of
events that must be sampled to be understood. Streaming sits between them and adds state,
ordering, and late-arrival problems. Edge and embedded add device fleets, update signing, and
physical failure modes. Choosing the pattern is choosing which set of problems you will spend the
next two years solving.

### 1.1 The decisions this topic unblocks

| Decision | Who is blocked | Consequence of choosing badly |
|---|---|---|
| Whether the model scores on a schedule, on an event, or on request | Product owner, integration lead | The output arrives after the decision it was meant to inform |
| Where the model runs | Platform architect, security | Data leaves the trust boundary, or latency exceeds the workflow budget |
| How the output reaches the actor | Clinical informatics, UX | The score exists but nobody sees it at the moment of action |
| How much capacity to buy | Finance, SRE | Peak-hour outages or idle spend for 20 hours a day |
| How to roll out and roll back | Release manager | A bad model cannot be withdrawn without taking down the workflow |
| What to monitor | SRE, data science | Service metrics are instrumented, model behaviour is not |

Each decision is cheap at design time and expensive after integration. The integration contract is
the hardest thing to change later, because it is embedded in another team's workflow, another
vendor's product, or a medical device's certified behaviour.

### 1.2 Why "just use an API" is not an answer

A real-time API is the default recommendation in much MLOps writing because it is easy to
demonstrate and easy to instrument. It is a poor default for biomedical work. Many clinical
decisions are batch by nature: a daily outreach list, a monthly population health review, a
weekly quality report. Many others are embedded: the model is a feature of the EHR or the device,
not a service the clinician calls. Others are edge: the computation must happen on the patient's
device because connectivity is unreliable or data cannot leave the premises.

The correct question is not "which pattern is best" but "which pattern matches the trigger, the
latency tolerance, and the action locus of this decision." Section 5 turns that question into a
procedure with a decision table.

### 1.3 The failure that motivates the topic

A health system deployed a readmission-risk model as a nightly batch. The model was accurate and
the list was delivered to care managers at 06:00. Within three months the care managers had
stopped using it. The reason was not accuracy. The list ranked patients by risk, but the care
managers' workflow started with the discharge list at 10:00, by which time the highest-risk
patients had already left. The model was answering a question at the wrong time. Changing the
pattern, not the model, fixed it: scoring moved to the discharge event, with a 90-second budget,
and the output appeared in the discharge workflow.

The lesson is that the pattern is part of the decision. A model that is right and late is a model
that is wrong.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a decision by trigger, latency tolerance, and action locus, and name the pattern
   that fits.
2. **Compare** batch, real-time, streaming, edge, embedded, shadow, and human-in-the-loop patterns
   on latency, cost, failure mode, and update path.
3. **Compute** the capacity and queueing consequences of a pattern choice, including utilization
   and expected wait.
4. **Design** the interface contract for a pattern, including input schema, output schema, error
   semantics, and idempotency.
5. **Choose** a fallback behaviour for each pattern and justify it against the cost of a wrong
   answer versus no answer.
6. **Diagnose** a pattern mismatch from operational symptoms such as stale outputs, timeouts, or
   backlog.
7. **Critique** a proposed architecture for hidden coupling between the pattern and the workflow.

## 3. The core mental model

A deployment pattern is defined by three coordinates: when the model runs, how long the answer may
take, and where the answer is consumed.

```text
TRIGGER                     LATENCY TOLERANCE              ACTION LOCUS
schedule ──────────────┐    days/hours ──────────┐         population list
event (new data) ──────┤    minutes ─────────────┤         clinician screen
request (user action) ─┤    seconds ─────────────┤         device actuator
continuous stream ─────┘    milliseconds ─────────┘         patient app

        │                          │                            │
        ▼                          ▼                            ▼
   ┌─────────┐              ┌─────────────┐              ┌─────────────┐
   │  Batch  │              │  Real-time  │              │  Embedded   │
   │ Stream  │              │  API        │              │  Edge       │
   │ Shadow  │              │  Human-loop │              │  Federated  │
   └─────────┘              └─────────────┘              └─────────────┘
```

The coordinates are not independent. A request-triggered decision almost always has a
sub-second-to-seconds budget, because a user is waiting. A schedule-triggered decision can
tolerate hours, which allows batch compute and cheaper infrastructure. An event-triggered decision
inherits the latency of the event pipeline plus the model, which is often minutes. A continuous
stream has a per-event budget that may be milliseconds, but the action may be delayed by
aggregation windows.

### 3.1 The request path

For a real-time pattern, draw the path from the actor's action to the rendered answer. Every
component on that path consumes budget.

```text
clinician clicks "open patient"
        │
        ▼
EHR page ──▶ integration engine ──▶ auth ──▶ feature service ──▶ model API
   │              │                    │            │                │
   │ 200 ms       │ 150 ms             │ 80 ms      │ 300 ms         │ 120 ms
   ▼              ▼                    ▼            ▼                ▼
render ◀──── assemble ◀────────────── response ◀── features ◀──── score
        total budget 1,500 ms; p99 target 1,200 ms
```

The model's own latency is often a minority of the total. A model that scores in 80 ms can still
miss a 1.5-second page budget if the feature lookup takes 900 ms. Pattern design is end-to-end
latency design, not model benchmarking.

### 3.2 The batch path

For a batch pattern, the path is a schedule, a data snapshot, a compute job, and a delivery step.

```text
02:00 snapshot ──▶ feature build ──▶ scoring ──▶ validation ──▶ delivery ──▶ 06:00 list
      │                 │                │            │             │
   completeness      join + impute    vectorized   row counts,    file or API
   check             (2.1 h)          (0.4 h)      range checks   to care team
```

The batch pattern's failure modes are different: a late upstream file delays the whole run; a
partial snapshot produces a plausible but incomplete list; a job that overruns its window
delivers after the workflow has started. Batch is not simpler than real-time; it moves complexity
from concurrency to data completeness.

### 3.3 The streaming path

Streaming computes on events as they arrive, often with state. It introduces event time versus
processing time, watermarks, late arrivals, and exactly-once or at-least-once semantics.

```text
device ──▶ ingest ──▶ validate ──▶ window/state ──▶ model ──▶ alert
   │          │           │              │             │
 event      queue      schema         late-data      per-window
 time       lag        check          handling       inference
```

A streaming model that aggregates a 30-minute window cannot produce an answer until the window
closes, even if each event arrives in milliseconds. The latency tolerance must be stated for the
decision, not for the transport.

### 3.4 Shadow and human-in-the-loop are overlays, not alternatives

Shadow mode is a pattern overlay: the model runs in the production path but its output is
suppressed. It can be applied to batch, real-time, or streaming. Human-in-the-loop is also an
overlay: the model's output is shown to a person who decides. Both change the action locus without
changing the trigger or the compute pattern.

Treating shadow as a separate pattern leads to a common error: teams build a shadow pipeline that
shares nothing with the production pipeline, then discover that the shadow results do not predict
production behaviour because the feature computation differs. Shadow should use the same package
and the same feature path as production, with the action suppressed at the last step.

### 3.5 The pattern determines the failure semantics

| Pattern | If the model is unavailable | If the model is wrong | Recovery unit |
|---|---|---|---|
| Batch | The list is late or missing; the previous list may still be usable | A cohort is mis-ranked for a day | Re-run the job for a date partition |
| Real-time API | The request times out; the workflow must have a fallback | One decision is wrong, immediately | Roll back the deployment |
| Streaming | Events queue; alerts are delayed; state may be lost | A window of alerts is wrong | Replay from the event log |
| Edge | The device uses a local fallback or stops | The patient sees a wrong result with no server check | Update the model package over the air |
| Embedded | The host application may fail | The clinician sees a wrong recommendation in the host UI | Vendor-issued patch |
| Shadow | No user impact; evaluation data is lost | Nothing, until the shadow is analysed | Restart the shadow job |

The recovery unit is the most useful column. It tells you what a rollback actually means, and it
determines whether the pattern can be reversed in minutes or days.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Batch scoring | Scoring a population on a schedule, producing a finite output | Called "offline" even when it feeds a live workflow | Rows per run, run duration |
| Real-time serving | Scoring on request, with a latency budget | Called "streaming" when the trigger is a request | Requests per second, p99 ms |
| Streaming | Scoring events continuously, often with state and windows | Assumed to mean low latency end to end | Events per second, watermark lag |
| Edge inference | Scoring on a device near the patient | Confused with embedded | Milliseconds, device fleet size |
| Embedded model | A model inside another product, such as an EHR or device | Used for any integrated model | Host application version |
| Shadow deployment | Running the model in production without acting on its output | Running on historical data | Predictions per day |
| Human-in-the-loop | A person reviews and decides on the model output | Used loosely for any alert | Review rate, override rate |
| Federated learning | Training across sites without centralizing data | Confused with federated inference | Sites, rounds |
| Event time | The time an event occurred | Confused with arrival time | ISO 8601 timestamp with offset |
| Processing time | The time the system handled an event | Treated as event time | ISO 8601 timestamp |
| Watermark | A bound on how late events may arrive before a window closes | Treated as a deadline for all events | Seconds of lateness |
| Stateful service | A service whose output depends on prior events | Any service with a database | State size, checkpoint interval |
| Stateless service | A service whose output depends only on the request | Any service without a cache | None |
| Cold start | The delay before a new instance can serve | Ignored in latency budgets | Seconds |
| Throughput | Work completed per unit time | Confused with latency | Requests per second |
| Latency | Time from trigger to answer | Reported as a mean | p50, p95, p99 in ms |
| Concurrency | Requests in flight at once | Confused with requests per second | Count |
| Backpressure | A mechanism that slows producers when consumers are saturated | Assumed to be automatic | Queue depth, rejection rate |
| Idempotency | The property that repeating a request has the same effect | Assumed for all APIs | Idempotency key |
| At-least-once delivery | A message may be delivered more than once | Treated as exactly-once | Duplicate rate |
| Exactly-once semantics | Processing each event once, in effect | Claimed without evidence | Duplicate rate, checkpoint design |
| Point-of-care | The moment and place a clinical decision is made | Used to mean "in the hospital" | Workflow step |
| Population health | Scoring a denominator of members or patients | Used for any aggregate report | Members per cycle |
| Cold path | Computation that is not latency-sensitive | Used for anything batch | Hours |
| Hot path | Computation on the critical path of a user action | Used for anything fast | Milliseconds |

Two definitions carry most of the confusion.

**Streaming is not the same as real-time.** A streaming system can have a 30-minute window and a
five-minute watermark, which makes it unsuitable for a decision that must complete in two seconds.
A real-time API can be built on a batch-computed feature store, which makes it fast at serving
time but stale in its inputs. The trigger and the latency are separate coordinates.

**Edge is not the same as embedded.** Edge inference runs on hardware near the patient, often
with a network connection for updates and telemetry. Embedded inference runs inside another
product's runtime, often with no separate deployment path at all. The update and rollback
mechanisms are completely different.

## 5. The mechanics, step by step

The procedure below selects a pattern and then specifies it. Steps 1 to 6 choose. Steps 7 to 12
make the choice implementable. The output of the procedure is not an architecture diagram; it is
a set of decisions with written justifications that a reviewer can challenge.

### Step 1 — Characterize the trigger

**Input.** A description of the workflow and the event that creates the need for a prediction.

**Output.** One of four trigger classes: schedule, data event, user request, continuous stream.

**Rationale.** The trigger determines the maximum feasible freshness of the output and the
infrastructure that can be used. A schedule cannot produce a fresher answer than its period. A
user request cannot be satisfied by a job that runs at 02:00. A data event inherits the latency of
the event pipeline. A continuous stream requires state and windowing.

The trigger must be described in the workflow's language, not the system's. "When the discharge
order is signed" is a trigger. "When the interface engine emits an ADT A03 message" is the
technical representation of that trigger, and it may fire at a different moment. The gap between
the clinical event and the technical event is a source of staleness that is invisible in the
architecture diagram.

Branch point: if a decision has more than one trigger (for example, a risk score refreshed nightly
and also computed on admission), the pattern must support both. That is usually a batch job plus
a real-time endpoint sharing the same feature definitions, not two independent models.

### Step 2 — State the latency budget end to end

**Input.** The workflow step, the user's tolerance, and the technical path.

**Output.** A latency budget with a percentile target, decomposed by component.

**Rationale.** "Fast" is not a budget. A budget is a number with a percentile and a scope: "p99 of
the complete page render is under 1,500 ms, of which the model may consume 250 ms." Without a
decomposition, each component optimizes its own mean and the end-to-end p99 fails.

The budget must be stated at the point the actor experiences, not at the service boundary. A model
API with a p99 of 400 ms can still produce a 3-second page if the feature service is slow. The
decomposition in Section 3.1 is the template.

Branch point: if the latency budget cannot be met with the available data sources, the pattern
must change (precompute features, move to batch, or move to edge) rather than the model being made
smaller. A smaller model rarely fixes a feature-lookup bottleneck.

### Step 3 — Locate the action locus

**Input.** The workflow, the actor, and the system in which the action occurs.

**Output.** Server, device, host application, or human reviewer.

**Rationale.** The action locus determines where the output must physically arrive and who is
allowed to see it. A prediction consumed by a clinician in the EHR must be rendered in the EHR; a
prediction consumed by an insulin pump must be on the device; a prediction consumed by a care
manager may be a list in a case-management system.

The locus also determines the trust boundary. Sending data to a central service is a disclosure;
computing on the device avoids it but limits model size and update frequency. This is a privacy
and security decision as much as an architectural one.

Branch point: if the action locus is a regulated device, the pattern is embedded or edge, and the
update path must satisfy the device's change control. A cloud API cannot be the action locus for
a real-time actuator without a safety analysis of network failure.

### Step 4 — Map data availability at decision time

**Input.** The features required by the model and the source systems that produce them.

**Output.** A table of feature, source, event time, arrival time, and availability at the decision
moment.

**Rationale.** The most common pattern failure is a model that requires data that is not available
when the pattern runs. A real-time model that needs a laboratory result arriving two hours later
is a batch model in disguise. A batch model that needs an event-stream feature is a streaming
model in disguise.

The table must use event time and arrival time separately. A feature measured at 14:00 but
available at 16:00 is not available to a decision at 14:30, even if the database contains it by
the time the analyst queries retrospectively.

Branch point: if a required feature is unavailable at decision time, three options exist: remove
it, replace it with a time-shifted proxy, or move the decision later. The choice must be recorded
in the model card because it changes the model's meaning.

### Step 5 — Estimate volume, burst, and seasonality

**Input.** Historical volumes, workflow schedules, and known peaks.

**Output.** Average rate, peak rate, burst shape, and growth assumption.

**Rationale.** Capacity is sized for the peak, not the average. A service that handles 9,000
requests per day may still need multiple replicas if 40 percent of requests arrive in a two-hour
morning window. A batch job that runs in four hours at current volume may overrun when the member
population grows by 30 percent.

Seasonality matters in biomedical work: influenza season changes both volume and prevalence; a
new clinic changes the case mix; a software rollout changes the request pattern. The estimate must
include a growth factor and a seasonal factor, not a single average.

Branch point: if the burst cannot be served within the latency budget, the pattern needs
precomputation, queueing with a stated wait, or admission control. Dropping requests silently is
not a capacity plan.

### Step 6 — Select the pattern

**Input.** The trigger, latency budget, action locus, data availability, and volume.

**Output.** A selected pattern with a written rationale and rejected alternatives.

**Rationale.** The selection is a constrained choice, not a preference. The decision table below
encodes the constraints. It has more than two outcomes because the trigger and the latency
tolerance interact.

| Trigger | Latency tolerance | Data available at trigger | Action locus | Pattern | Notes |
|---|---|---|---|---|---|
| Schedule | Hours to days | Complete snapshot after batch window | Population list, report | Batch | Cheapest; staleness equals period |
| Data event | Minutes | Event payload plus reference data | Clinician inbox, workflow queue | Event-driven service or micro-batch | Inherits pipeline latency |
| User request | Sub-second to seconds | Precomputed features or fast lookup | EHR screen, app | Real-time API with feature store | End-to-end budget dominates |
| Continuous stream | Milliseconds to seconds per event | Windowed state | Monitor, alarm, actuator | Streaming with state | Late data and ordering are the hard parts |
| User request | Milliseconds, no network | On-device sensors | Device actuator, patient app | Edge inference | Update and fleet management dominate |
| Any | Any, but inside a host product | Host-provided data | Host UI or device | Embedded | Vendor release cycle governs change |
| Any | Not applicable, no action | Any | None, evaluation only | Shadow overlay | Must share the production feature path |
| Any | Any, human decides | Any | Human reviewer | Human-in-the-loop overlay | Measures override rate and burden |

The table is a starting point, not an oracle. Two rows can both be feasible; the choice then turns
on cost, operability, and regulatory burden. A batch pattern is usually cheaper to operate and
easier to audit. A real-time pattern is usually more expensive and harder to test, but it is the
only way to influence a decision at the moment it is made.

Branch point: if the table yields no feasible pattern, the decision must be redesigned. A model
cannot fix a workflow whose latency tolerance is shorter than the data availability.

### Step 7 — Define the interface contract

**Input.** The selected pattern, the input schema, and the output semantics.

**Output.** A versioned contract: fields, types, units, required versus optional, error codes,
idempotency behavior, and timeout behavior.

**Rationale.** The interface is what other teams integrate against. A contract that changes
silently breaks the workflow even if the model is unchanged. The contract must be explicit about
units (mg/dL versus µmol/L), about missing values (reject versus impute), and about the meaning of
the output (probability versus decision versus rank).

For a real-time API, the contract includes the timeout and retry semantics. For a batch job, it
includes the file format, the partition key, and the delivery guarantee. For a stream, it includes
the event schema, the watermark, and the duplicate semantics.

Branch point: if the consumer cannot handle the contract's error semantics, the contract must be
negotiated before build. A consumer that treats a timeout as "low risk" has inverted the meaning
of the output.

### Step 8 — Choose the state and feature strategy

**Input.** The feature table from Step 4 and the pattern.

**Output.** A decision on precomputation, caching, state, and freshness.

**Rationale.** Real-time patterns usually depend on precomputed features to meet the latency
budget. That creates a freshness problem: the feature may be minutes or hours old. The strategy
must state the maximum acceptable age per feature and how staleness is detected.

Batch patterns can compute features from the snapshot, which is simpler but produces a fixed
vintage. Streaming patterns carry state, which introduces checkpointing, recovery, and exactly-once
questions. Edge patterns must carry the feature computation to the device, which limits model
complexity.

Branch point: if a feature cannot be precomputed within the freshness requirement, the pattern
must either accept the staleness (and say so) or move the computation into the request path (and
pay the latency).

### Step 9 — Define failure and fallback semantics

**Input.** The pattern, the workflow's tolerance for no answer, and the cost of a wrong answer.

**Output.** A fallback decision per failure mode: fail closed, fail open, serve stale, queue, or
degrade to a rule.

**Rationale.** Every pattern fails differently, and the correct fallback depends on the decision.
For a sepsis alert, a missing score may be safer than a stale score presented as current. For a
care-gap list, a one-day-old list may be perfectly usable. For an insulin dose recommendation, no
answer is safer than a wrong answer.

The fallback must be explicit in the contract and visible to the consumer. A service that returns
a stale score with a 200 status and no staleness flag has hidden the failure from the workflow.

Branch point: if the workflow has no fallback, the pattern must include one, or the deployment
must be blocked. "The clinician will notice" is not a fallback.

### Step 10 — Design the release and rollback path for the pattern

**Input.** The pattern and the organization's deployment tooling.

**Output.** A release procedure with the unit of rollback and the time to restore.

**Rationale.** Batch, real-time, streaming, edge, and embedded have different rollback units. A
batch job rolls back by re-running a partition with the previous package. A real-time service
rolls back by switching traffic to the previous deployment. A streaming job rolls back by
restoring a checkpoint and replaying. An edge fleet rolls back by an over-the-air update that may
take days to reach every device.

A release procedure that assumes the real-time model will not work for an edge fleet. The time to
restore must be measured, not estimated, and it must be shorter than the workflow's tolerance for
the degraded state.

Branch point: if the rollback time exceeds the tolerance, the pattern needs a canary or a
parallel-run strategy so that the bad version never reaches the full population.

### Step 11 — Test under realistic load and clock conditions

**Input.** The package, a load profile, and a test plan.

**Output.** Test results for latency, throughput, error rate, and time behavior.

**Rationale.** A model tested at 10 requests per second tells you nothing about a service that
receives 400 requests in a minute at 07:00. A batch job tested on a full snapshot tells you
nothing about a snapshot with a missing partition. A streaming job tested with in-order events
tells you nothing about late arrivals.

Clock conditions belong in the test plan. Test with a device clock that is five minutes fast, a
server clock that is two minutes slow, and a daylight-saving transition. Many "model" failures in
biomedical systems are timestamp failures.

Branch point: if the test environment cannot reproduce production volume, use a production
shadow or a replay against recorded traffic, and state the residual uncertainty.

### Step 12 — Instrument pattern-specific signals and revisit the choice

**Input.** The deployed pattern and its telemetry.

**Output.** A monitoring set that includes the pattern's characteristic failure signals.

**Rationale.** Each pattern fails in its own way. Batch fails silently through late or incomplete
outputs. Real-time fails through timeouts and error codes. Streaming fails through lag and late
data. Edge fails through fleet version skew and connectivity. Embedded fails through host
application changes.

The monitoring must include the signal that would reveal the pattern's characteristic failure, not
only the model's performance. A batch job with perfect model metrics and a 40 percent incomplete
snapshot is a failed deployment.

Branch point: if the pattern's characteristic signal is not measurable with current telemetry, add
it before go-live. Retrofitting telemetry after an incident is how detection gaps become chronic.

### 5.13 Edge cases the procedure must handle

**Schema drift.** A batch input file gains a column or changes a type; a real-time request gains a
field; a stream event changes its schema. The contract must version the schema, and the consumer
must reject unknown versions rather than coerce them.

**Late data.** A batch snapshot arrives after the job starts; a stream event arrives after its
window closes. The pattern must define lateness handling: wait, recompute, or accept and flag.

**Partial rollout.** A real-time service runs two versions behind a load balancer; an edge fleet
runs three firmware versions. Monitoring must segment by version, and the rollback must target the
version rather than the whole deployment.

**Model and feature skew.** The batch feature job and the real-time feature service compute the
same feature differently. Parity tests must run on the same input across both paths, not only
within one path.

**Clock skew.** Event time, ingestion time, and decision time disagree. The pattern must record
all three and use event time for windows. A stream that uses processing time will misattribute
late events.

**Quota exhaustion.** A real-time service hits an API rate limit for a feature lookup; a batch job
hits a cloud quota for workers. The capacity plan must include quotas, and the fallback must be
defined for quota exhaustion specifically.

**Dependency rotation.** A credential or certificate expires. Batch jobs often fail at the start
of a run; real-time services fail on the next request. The rotation must be tested in the pattern's
actual runtime, not only in a staging environment with different credentials.

**Regional outage.** A single-region batch job cannot run; a single-region API cannot serve. The
pattern must state whether the workflow can tolerate a delay (batch) or needs a failover (real-time
and streaming).

### 5.14 Pattern comparison summary

| Dimension | Batch | Real-time API | Streaming | Edge | Embedded |
|---|---|---|---|---|---|
| Freshness | Period-bound | Request-time | Event-time | Sensor-time | Host-update-bound |
| Latency | Hours | 100 ms to 2 s | Per-event, window-bound | Milliseconds | Host-dependent |
| Cost driver | Compute hours | Peak concurrency | Throughput and state | Device fleet | Vendor release |
| Hardest failure | Incomplete input | Timeout and cold start | Late and out-of-order data | Version skew | Host change |
| Rollback unit | Job partition | Deployment | Checkpoint and replay | OTA package | Vendor patch |
| Audit evidence | Output file and run log | Request log | Event log and checkpoint | Device telemetry | Host audit log |
| Best fit | Population health, claims | EHR screen, triage | ICU monitoring, wearables | Patient-worn sensors | Regulated device workflow |

### 5.15 A decision walk-through

Three decisions from the same hospital illustrate how the coordinates combine. The first is a
population health outreach list for 180,000 members with diabetes. The trigger is a monthly
schedule, the tolerance is days, the data are complete after the claims run, and the action is a
mailed or telephoned outreach. The pattern is batch. The hardest problem is not latency; it is
keeping the denominator complete and the list reproducible for audit.

The second is a deterioration alert on the ward. The trigger is a new observation or laboratory
result, the tolerance is minutes, the action is a clinician review before the next rounding
cycle, and the data are available in the EHR at the moment of the event. The pattern is an
event-driven service with a real-time feature lookup. The hardest problem is the arrival-time
contract for laboratory data, not the model.

The third is a closed-loop insulin recommendation. The trigger is a continuous glucose reading,
the tolerance is seconds, the action is a pump adjustment, and the data are on the device. The
pattern is edge inference with a local safety rule and a bounded fallback. The hardest problem is
the update path and the failure behavior when the sensor is unreliable, not the inference latency.

In each case, the first three steps produce a different answer, and the remaining steps inherit
it. The pattern is not a matter of taste; it is the consequence of the decision's coordinates.

## 6. Worked example

A regional health plan operates a care-gap model for 180,000 members with diabetes. The model
predicts the probability that a member has an open care gap that would benefit from outreach. The
plan is choosing between a nightly batch list and a real-time score shown to a care manager when
a member contacts the plan or arrives at a participating clinic.

The numbers below are constructed for teaching. They are internally consistent, and the
arithmetic is reproduced in Section 7.

### 6.1 The two candidates

| Dimension | Batch list | Real-time score |
|---|---|---|
| Trigger | Nightly schedule at 02:00 | Request from care-management application |
| Population per cycle | 180,000 members | 9,000 requests per month |
| Delivery | File to the care-management system by 06:00 | JSON response in the application |
| Latency budget | Complete by 06:00; 4-hour compute window | p99 1,200 ms end to end |
| Flag rate | 2.1 percent | 1.8 percent |
| Precision at the operating threshold | 24 percent | 41 percent |
| Review time per flagged case | 6 minutes | 6 minutes |
| Loaded reviewer cost | 38 US dollars per hour | 38 US dollars per hour |
| Outreach capacity | 600 cases per week | 220 cases per day |

The batch list has lower precision because it scores everyone, including members who would never
have contacted the plan. The real-time score has higher precision because the request itself is
evidence of engagement. The comparison is not therefore "real-time is better." The batch list has
a much larger reach, and reach may matter more than precision for a population health programme.

### 6.2 Batch capacity and the queue that never drains

The batch model flags 2.1 percent of 180,000 members:

```text
flagged_per_month = 180,000 × 0.021 = 3,780
flagged_per_week  = 3,780 / 4.33 = 873
review_capacity   = 600 per week
utilization       = 873 / 600 = 1.455
```

A utilization above 1.0 means the queue grows without bound. Each week adds 273 cases to the
backlog:

```text
weekly_growth = 873 − 600 = 273 cases
monthly_growth = 273 × 4.33 = 1,182 cases
```

By the end of the first month, the list has 1,182 unreviewed cases. By the end of the quarter, the
oldest cases are more than eight weeks old, and the care gap may have closed or the member may
have disenrolled. The model is accurate; the pattern is not operable at the chosen capacity.

The reachable true gaps are limited by capacity, not by flags:

```text
reviewed_per_month = 600 × 4.33 = 2,598
true_gaps_reached  = 2,598 × 0.24 = 624
review_cost        = 2,598 × 6 min × (38 / 60) = 9,872 USD
cost_per_true_gap  = 9,872 / 624 = 15.82 USD
```

### 6.3 Real-time capacity, latency, and error budget

The real-time service receives 9,000 requests per month. The availability service level objective (SLO) is 99.5 percent over
30 days.

```text
requests_30d    = 9,000
error_budget    = 9,000 × 0.005 = 45 failed requests
budget_minutes  = 30 × 24 × 60 × 0.005 = 216 minutes
```

The latency budget is decomposed across the request path:

| Component | p99 budget (ms) | Measured p99 (ms) |
|---|---:|---:|
| EHR page initiation | 200 | 180 |
| Integration engine | 150 | 120 |
| Authentication | 80 | 65 |
| Feature lookup | 300 | 290 |
| Model inference | 120 | 110 |
| Response render | 200 | 190 |
| Contingency | 150 | not used |
| **Total** | **1,200** | **955** |

The measured total is 955 ms, leaving 245 ms of headroom against the 1,200 ms target. The
contingency line is not padding; it is the budget for the components that will grow after the
first integration change.

The peak load is not the daily average. Forty-five percent of requests arrive between 07:00 and
10:00, and 12 percent of that window arrives in a five-minute burst:

```text
peak_window_requests = 9,000 × 0.45 = 4,050
burst_requests       = 4,050 × 0.12 = 486
burst_rate           = 486 / 300 s = 1.62 requests per second
service_time         = 0.045 s
offered_load         = 1.62 × 0.045 = 0.073 erlangs
capacity_per_replica = 1 / 0.045 = 22.2 requests per second
replicas_at_70pct    = 1
```

One replica is sufficient by CPU. The deployment still runs two, because cold start is eight
seconds and the readiness probe must keep traffic away until the model is loaded. The second
replica is an availability decision, not a throughput decision.

The alert economics are:

```text
flagged_per_month = 9,000 × 0.018 = 162
true_gaps         = 162 × 0.41 = 66
review_cost       = 162 × 6 min × (38 / 60) = 615 USD
cost_per_true_gap = 615.60 / 66.42 = 9.27 USD
```

The real-time pattern reaches 66 true gaps per month at 9.27 USD each. The batch pattern reaches
624 at 15.82 USD each, but with an eight-week delay and an unbounded queue. Neither dominates. The
choice depends on whether the programme values reach or timeliness, and on whether the outreach
capacity can be expanded.

### 6.4 Diagnostic one: latency rises while the model is unchanged

At month four, the end-to-end p99 rises from 955 ms to 1,850 ms. The model's own p99 is unchanged
at 110 ms. The feature lookup p99 has risen from 290 ms to 1,100 ms after a feature-service
release.

The cause is a classic pattern failure: the feature service now makes one downstream call per
feature instead of one batched call. Six features produce six sequential calls at roughly 180 ms
each. The model is not slower; the pattern's data access is.

The first diagnostic is a trace-level latency breakdown by span, not an aggregate service
dashboard. The second is a comparison of the feature-service call count per request before and
after the release. The response is to restore the batched call, add a latency SLO on the feature
service itself, and add a contract test that fails if the call count per request exceeds one.

This failure would be invisible in model performance metrics. Predictions are correct, just late.

### 6.5 Diagnostic two: the batch queue is the failure

In the batch pattern, the model's precision is stable at 24 percent, but the care-management team
reports that the list is "not useful anymore." The list age p95 has risen from 2 days to 19 days,
and the outreach yield, the fraction of contacted members who accept an appointment, has fallen
from 31 percent to 14 percent.

The utilization arithmetic explains it. At 1.455, the queue grows by 273 cases per week. The
oldest cases are not wrong; they are stale. A member who had an open gap in January may have
closed it by March, so the reviewer spends time on cases that no longer need action.

The first diagnostic is list age, not model precision. The second is the correlation between list
age and outreach yield, segmented by week. The response is to reduce the flag rate, increase
review capacity, or change the pattern. Retraining the model would not change the queue.

### 6.6 Vary one assumption: review capacity

Suppose the plan hires enough reviewers to handle 900 cases per week instead of 600.

```text
utilization = 873 / 900 = 0.970
mean_wait   = 1 / (900 − 873) weeks = 0.037 weeks = 0.26 days
```

The queue collapses from an unbounded backlog to a wait of about six hours. The batch pattern now
delivers a list that is at most one day old, and the outreach yield should recover toward its
previous level. The cost per true gap changes because more cases are reviewed:

```text
reviewed_per_month = 900 × 4.33 = 3,897
true_gaps_reached  = 3,897 × 0.24 = 935
review_cost        = 3,897 × 6 × (38 / 60) = 14,808 USD
cost_per_true_gap  = 14,808 / 935 = 15.84 USD
```

The cost per true gap is essentially unchanged, because both numerator and denominator scale with
capacity. What changes is the reach and the timeliness. This is the key insight of the example:
capacity does not change unit economics much; it changes whether the pattern is operable at all.

If the plan cannot expand capacity, the alternative is to lower the flag rate to 1.4 percent,
which brings weekly flags to 582 and utilization to 0.97 at the original capacity. Precision rises
because the threshold is higher, but recall falls. That trade-off is a clinical and economic
decision, not an engineering one.

### 6.7 What the example establishes

The pattern choice was not settled by model quality. The batch model had higher reach and lower
precision; the real-time model had higher precision and lower reach. The batch pattern failed
operationally because its review capacity was below its flag rate, and the failure appeared as
stale lists rather than as bad predictions. The real-time pattern failed in latency when a
dependency changed its call pattern, and the failure appeared as timeouts rather than as wrong
answers.

Both failures are pattern failures. Both are detectable with pattern-specific signals: list age
for batch, span-level latency for real-time. Neither is detectable with a model performance
dashboard.

## 7. Code, query, or calculation

The script below computes the capacity, queue, latency, and cost numbers from Section 6. It uses
only the standard library. The queue calculation is an M/M/1 approximation; it is adequate for
showing whether a queue is stable, not for sizing a service to the millisecond.

```python
"""Capacity and latency arithmetic for the deployment-pattern example."""

from math import ceil

members = 180_000
flag_rate = 0.021
weeks_per_month = 4.33
capacity_per_week = 600
precision = 0.24
review_minutes = 6
hourly_cost = 38.0

flagged_month = members * flag_rate
flagged_week = flagged_month / weeks_per_month
utilization = flagged_week / capacity_per_week
weekly_growth = flagged_week - capacity_per_week
reviewed_month = capacity_per_week * weeks_per_month
true_gaps = reviewed_month * precision
review_cost = reviewed_month * review_minutes * hourly_cost / 60
cost_per_gap = review_cost / true_gaps

capacity_900 = 900
util_900 = flagged_week / capacity_900
wait_weeks = 1 / (capacity_900 - flagged_week)
reviewed_900 = capacity_900 * weeks_per_month
gaps_900 = reviewed_900 * precision
cost_900 = reviewed_900 * review_minutes * hourly_cost / 60
gap_cost_900 = cost_900 / gaps_900

flag_low = 0.014
flagged_low_week = members * flag_low / weeks_per_month
util_low = flagged_low_week / capacity_per_week

requests_month = 9_000
availability_slo = 0.995
error_budget = requests_month * (1 - availability_slo)
budget_minutes = 30 * 24 * 60 * (1 - availability_slo)
observed_error = 0.009
burn = observed_error / (1 - availability_slo)
days_to_exhaust = 30 / burn

peak_share, burst_share, burst_seconds = 0.45, 0.12, 300
peak_requests = requests_month * peak_share
burst_requests = peak_requests * burst_share
burst_rate = burst_requests / burst_seconds
service_time = 0.045
offered_load = burst_rate * service_time
capacity_per_replica = 1 / service_time
replicas = max(1, ceil(offered_load / (0.70 * capacity_per_replica)))

latency = [200, 150, 80, 300, 120, 200, 150]
latency_total = sum(latency)
measured = [180, 120, 65, 290, 110, 190]
measured_total = sum(measured)

rt_flag_rate, rt_precision = 0.018, 0.41
rt_flagged = requests_month * rt_flag_rate
rt_gaps = rt_flagged * rt_precision
rt_cost = rt_flagged * review_minutes * hourly_cost / 60
rt_gap_cost = rt_cost / rt_gaps

print(f"batch_flagged_month   {flagged_month:.0f}")
print(f"batch_flagged_week    {flagged_week:.0f}")
print(f"batch_utilization     {utilization:.3f}")
print(f"batch_weekly_growth   {weekly_growth:.0f}")
print(f"batch_reviewed_month  {reviewed_month:.0f}")
print(f"batch_true_gaps       {true_gaps:.0f}")
print(f"batch_cost_per_gap    {cost_per_gap:.2f} USD")
print(f"capacity900_util      {util_900:.3f}")
print(f"capacity900_wait_days {wait_weeks * 7:.2f}")
print(f"capacity900_gap_cost  {gap_cost_900:.2f} USD")
print(f"low_flag_utilization  {util_low:.3f}")
print(f"error_budget          {error_budget:.0f} requests")
print(f"budget_minutes        {budget_minutes:.0f}")
print(f"burn_rate             {burn:.2f}")
print(f"days_to_exhaustion    {days_to_exhaust:.1f}")
print(f"burst_rate            {burst_rate:.2f} req/s")
print(f"offered_load          {offered_load:.4f} erlangs")
print(f"replicas              {replicas}")
print(f"latency_budget_total  {latency_total} ms")
print(f"measured_total        {measured_total} ms")
print(f"latency_headroom      {latency_total - measured_total} ms")
print(f"rt_alerts_month       {rt_flagged:.0f}")
print(f"rt_true_gaps          {rt_gaps:.0f}")
print(f"rt_cost_per_gap       {rt_gap_cost:.2f} USD")
```

Expected output, confirmed on Python 3.11:

```text
batch_flagged_month   3780
batch_flagged_week    873
batch_utilization     1.455
batch_weekly_growth   273
batch_reviewed_month  2598
batch_true_gaps       624
batch_cost_per_gap    15.83 USD
capacity900_util      0.970
capacity900_wait_days 0.26
capacity900_gap_cost  15.83 USD
low_flag_utilization  0.970
error_budget          45 requests
budget_minutes        216
burn_rate             1.80
days_to_exhaustion    16.7
burst_rate            1.62 req/s
offered_load          0.0729 erlangs
replicas              1
latency_budget_total  1200 ms
measured_total        955 ms
latency_headroom      245 ms
rt_alerts_month       162
rt_true_gaps          66
rt_cost_per_gap       9.27 USD
```

Three readings of the output matter.

First, the utilization of 1.455 is the whole batch failure. A utilization above 1.0 means no
queue discipline can keep up; the only fixes are fewer arrivals, more capacity, or a different
pattern. The wait-time formula is undefined in this regime, which is why the script does not
print one.

Second, the capacity-900 case has a utilization of 0.970 and a mean wait of 0.26 days. This is a
stable queue, but it is close to the knee. At 95 percent utilization, small increases in arrival
rate produce large increases in wait. A capacity plan that sits at 0.97 has no room for a
seasonal peak.

Third, the real-time service needs one replica by CPU but runs two for cold-start and readiness
reasons. The capacity calculation and the availability decision are separate. Confusing them
produces either over-provisioning or a service that cannot survive a single pod restart.

The reference service in `labs/mlops/` is a real-time pattern. `app/main.py` exposes a `/predict`
endpoint with a 422 on invalid input and a 503 when the model is not loaded; `k8s/deployment.yaml`
sets readiness and liveness probes; `k8s/hpa.yaml` scales on CPU between two and six replicas.
The lab demonstrates the mechanics of the pattern but not the latency decomposition, because it
has no downstream feature service. Adding one would make the end-to-end budget exercise real.

## 8. Data, standards, and regulatory context

Deployment patterns are shaped by interface standards more than by modelling choices. The
standards below are the ones that determine what a pattern can receive, how it returns an answer,
and what evidence it must keep. Binding requirements are separated from conventions.

### 8.1 Interface standards by pattern

| Pattern | Standard or specification | What it defines | Binding status |
|---|---|---|---|
| Real-time API | HL7 Fast Healthcare Interoperability Resources (FHIR) RESTful API | Resource-oriented HTTP interface, search parameters, operations | Binding when the buyer or regulator requires FHIR; otherwise a convention |
| Event-driven | FHIR Subscriptions R5, or the R5 backport for R4 and R4B | Topic-based notification when a resource changes | Convention; the backport guide defines a standard subset for older servers |
| Batch | FHIR Bulk Data Access (`$export`) | Asynchronous export of a population to files | Convention, widely required in US interoperability programmes |
| Embedded clinical decision support | HL7 CDS Hooks | Invocation of a decision service at a defined point in the clinician workflow | Convention; adopted by major EHR vendors |
| Event-driven clinical data | HL7 version 2 ADT and ORU messages | Admission, discharge, transfer, and observation result events | Widely deployed; message semantics vary by site |
| Streaming device data | IHE Patient Care Device (PCD) profiles; IEEE 11073 | Device observation and alert semantics | Convention plus device standards; varies by device class |
| Imaging | DICOM | Image objects, worklists, and network services | Binding in imaging workflows |
| Service telemetry | OpenTelemetry semantic conventions | Metric and trace names, units, and attributes | Convention; the reference service uses Prometheus text format instead |

The FHIR Subscriptions specification is the clearest example of a standard that changes the
pattern. Before topic-based subscriptions, a client polled for changes; after, the server notifies
on a topic. The pattern shifts from schedule to event, and the latency budget shifts with it. A
team that builds polling because it is familiar is choosing a different pattern from the one the
interface now supports.

### 8.2 What the FDA expects of a deployed pattern

The Good Machine Learning Practice principles, published in 2021, include a principle on testing
under clinically relevant conditions. That principle constrains the pattern indirectly: a model
tested only in a batch retrospective cannot demonstrate performance in a real-time workflow. The
test conditions must match the deployment pattern's population, environment, and users.

The January 2025 draft guidance on AI-enabled device software functions recommends a postmarket
performance monitoring plan. For a real-time pattern, the plan must include request-level
telemetry and error semantics; for a batch pattern, it must include completeness and delivery
checks; for an edge pattern, it must include fleet version and device-level performance. A single
generic monitoring plan does not fit all three.

The predetermined change control plan (PCCP) guidance constrains the update path. An edge fleet's over-the-air update process is part
of the modification protocol if the model can change after authorization. A batch job that
retrains nightly and promotes automatically is a continuous change that the PCCP must describe, or
it is an unauthorized change.

### 8.3 What IEC 62304 and ISO 14971 add

IEC 62304's software safety classification applies to the software that implements the pattern,
not only to the model. A real-time service that fails closed and blocks a clinical workflow may
have a higher safety class than a batch report that is reviewed by a human. The classification
drives the rigour of the architecture, the testing, and the change control.

ISO 14971 requires risk control for the failure modes of the pattern. A network failure in an
edge device, a timeout in a real-time API, and a late batch file are hazards with different
severity and probability. The risk file should name them and state the control: a local fallback,
a fail-closed response, or a delivery deadline with an alert.

### 8.4 Records and privacy

The pattern determines what must be logged. HIPAA's audit control standard requires recording and
examining activity in systems that contain electronic protected health information. A real-time
request log must record enough to reconstruct the decision without storing more identifiers than
necessary. A batch output file is itself a record and must be access-controlled, retained, and
disposed of according to policy.

Part 11 applies when the pattern produces electronic records in an FDA-regulated activity. A batch
scoring run that supports a clinical trial may be a record; a real-time score shown to a clinician
may also be. The audit trail must be computer-generated, time-stamped, and must not obscure prior
values.

### 8.5 What is convention

- Choosing REST over gRPC. Both can meet a latency budget; the choice turns on tooling and
  organizational familiarity.
- Requiring a message queue for every event. A synchronous call with a timeout may be simpler and
  safer for a low-volume workflow.
- Using Kubernetes for every service. A single virtual machine with a supervised process can be
  the right pattern for a low-volume batch job.
- Two-week shadow periods. The duration should follow event volume and label latency.

> **Note:** Interface standards define what is possible, not what is deployed. A site that sends
> HL7 v2 messages with locally customized fields may be nominally conformant and practically
> unique. The data contract, not the standard name, is what the model depends on.

## 9. Industry practice and case evidence

### 9.1 Embedded decision support: IDx-DR

IDx-DR received De Novo authorization in 2018 for detecting diabetic retinopathy from retinal
images in primary care. The authorization described a workflow in which the system captures an
image, checks quality, and returns a result or a rejection. The pattern is embedded: the model is
part of a device used by a primary care operator, not a service called by a developer.

The documented lesson is that the quality gate is part of the pattern. A rejection is a valid
output, and the workflow must handle it. The inference is that an embedded pattern needs an
explicit output space that includes "cannot answer," and the host application must render that
state without treating it as a system error.

### 9.2 Embedded prediction in an EHR: the Epic Sepsis Model

The Epic Sepsis Model was embedded in the EHR and scored patients every 15 minutes. Wong and
colleagues' external validation at Michigan Medicine found a hospitalization-level AUC of 0.63,
alerts for 18 percent of hospitalizations, and a positive predictive value of 12 percent at the
alert threshold.

The pattern is relevant to the finding. An embedded model that fires inside the EHR has no
separate deployment cadence; it changes when the vendor updates the host. The deploying
organization may have little ability to canary a new version, segment traffic by version, or roll
back independently. The inference is that embedded patterns concentrate change-control risk in the
vendor relationship, and the contract must include notification and version visibility.

### 9.3 Edge and device patterns

Connected insulin pens, continuous glucose monitors, and wearable arrhythmia detectors are edge
patterns. The published evidence on these devices focuses on the algorithm and the device's
regulatory authorization; operational evidence on fleet update success and version skew is rarely
public. The inference, clearly labelled, is that the operational risk of an edge pattern is
dominated by update distribution and sensor reliability rather than by model accuracy.

### 9.4 Reliability practice: canary and progressive delivery

The Google SRE Workbook describes canary release as a partial, time-limited deployment with an
evaluation process integrated into the release. It emphasizes that the canary population must be
representative and that the evaluation must use metrics that indicate problems.

For a real-time clinical service, the canary is usually a traffic fraction, not a patient
fraction. That distinction matters: traffic-based canaries can concentrate on certain users or
times of day, so the canary population may not represent the full case mix. A pattern-aware
canary segments by site, shift, and case type.

### 9.5 Hidden technical debt in pattern choice

Sculley and colleagues' 2015 paper on hidden technical debt describes how machine learning systems
accumulate debt at the system level: glue code, pipeline jungles, undeclared consumers, and
configuration debt. The paper's often-cited figure shows that the model code is a small fraction
of the surrounding infrastructure.

The relevance to deployment patterns is direct. Each pattern brings its own glue code and its own
undeclared consumers. A batch output file may be consumed by three teams the model owner has never
met; a real-time API may be called by a mobile app that was not in the integration plan. The
pattern choice determines how many undeclared consumers can appear and how hard they are to find.

### 9.6 What the evidence does not show

There is no public benchmark that compares deployment patterns on clinical outcomes. The
literature compares models, not patterns. Claims that real-time deployment improves outcomes are
usually inferences from workflow logic, not results from controlled studies. A team should treat
pattern selection as a design decision grounded in latency, reach, and operability, and should
measure the outcome after deployment rather than assume it.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Batch list arrives after the workflow starts | Delivery time drifts from 06:00 to 07:40 | The feature join grew or an upstream file arrived late | The team works from yesterday's list | Delivery deadline alert, runtime budget, partial-run fallback |
| Batch snapshot is incomplete | Row count is 8 percent below the trailing median | A source partition is missing but the job succeeds | A whole cohort is absent from the list | Row-count and denominator checks that fail the run |
| Real-time request times out | p99 crosses the page budget before the mean moves | A downstream dependency added latency | The clinician sees a blank panel or a stale score | Span-level latency budget, batched feature fetch, circuit breaker |
| Cold start causes first-request failures | 503s cluster after a deploy or scale-out | The model loads for eight seconds but readiness passes early | Errors during the morning peak | Startup probe, minimum replicas, preloaded model |
| Streaming events arrive late | Watermark lag rises while throughput is flat | Network or producer delay; window closes before events arrive | Alerts are missed or attributed to the wrong window | Allowed lateness, window recomputation, late-event metric |
| Stream state is lost | Recovery time after a restart exceeds the window | Checkpoint interval is too long or state is not durable | Duplicate or missing alerts after failover | Checkpoint frequency, exactly-once design, replay test |
| Edge fleet version skew | Two model versions report from the same clinic | An over-the-air update stalled on some devices | Inconsistent behaviour that is hard to reproduce | Fleet version dashboard, update success SLO, minimum version gate |
| Edge device loses connectivity | Telemetry gap for a device that is still in use | Network coverage or credential expiry | Local fallback or silent stop | Offline mode, local audit buffer, connectivity alert |
| Embedded model changes with the host | Output distribution shifts after an EHR upgrade | The vendor updated the model or its inputs | An uncontrolled change with no local record | Vendor notification clause, version visibility, post-upgrade validation |
| Shadow pipeline diverges from production | Shadow and production scores differ on the same input | Shadow uses a separate feature path | The shadow evaluation does not predict production | Shared package and feature path, parity test |
| Human review becomes the bottleneck | Override rate and review backlog rise together | The alert volume exceeds review capacity | Alert fatigue and delayed action | Capacity-matched threshold, queue-age monitor |
| Peak overload | Queue depth spikes at 07:00 and clears by 11:00 | Capacity sized on daily average | Timeouts at the moment of highest clinical need | Peak-based sizing, admission control, precomputation |
| Clock skew misattributes events | Alerts cluster at improbable times | Device and server clocks disagree | Wrong window, wrong patient state, wrong audit order | Record event, ingestion, and decision time; monitor offsets |
| Two patterns drift apart | Batch and real-time scores disagree for the same patient | Separate feature definitions and separate models | Inconsistent advice to the same clinician | One feature definition, one model, two serving paths |

Red flags worth escalating: a batch job with no denominator check; a real-time service whose
latency SLO is measured only at the server; an edge fleet with no version inventory; a shadow
evaluation that does not share the production package; a fallback that returns a stale score
without a staleness flag.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The latency budget is measured at the actor's screen | The workflow experiences end-to-end delay | Server-side only measurement misses integration and render time | Synthetic transaction from the client; compare with server metrics |
| Required features are available at the trigger | The pattern's output is valid at decision time | Late laboratory or interface data | Event-time versus arrival-time audit on a sample |
| Peak volume is estimated correctly | Capacity is sized for the busiest period | New clinic, seasonal surge, marketing campaign | Load test at 1.5 times observed peak |
| Review capacity matches flag volume | A queue is stable only below 100 percent utilization | Hiring freeze, case-mix change, threshold change | Weekly utilization and queue-age review |
| The queue model is adequate | Wait times are estimated from it | Non-stationary arrivals, priority queues, batching | Compare predicted and observed wait |
| Devices remain connected and updated | Edge behaviour depends on fleet state | Coverage gaps, stalled updates, battery failure | Fleet telemetry and update-success rate |
| The vendor notifies before a host change | Embedded patterns cannot be canaried locally | Contract has no notice clause | Contract review; post-change validation |
| Clocks are synchronized | Windows and audit trails depend on time | NTP failure, daylight saving, device drift | Offset monitoring and time-zone tests |
| Shadow traffic is representative | Shadow results are used for decisions | Sampling by site or shift | Compare shadow and production case mix |
| The fallback is acceptable to the workflow | Every pattern fails eventually | Fallback returns a misleading answer | Failure injection with clinicians |
| Standard conformance implies semantic conformance | Interfaces are assumed to mean the same thing | Local coding and unit customization | Contract tests on real messages |
| The load test represents production | Capacity decisions rest on it | Test data lacks the long tail | Replay recorded traffic; production canary |

## 12. Exercises

### Exercise 12.1 — Classify three decisions

For each decision, state the trigger, latency tolerance, action locus, and the pattern you would
choose, with one sentence of justification.

(a) A pharmacist reviews medication interactions for patients discharged tomorrow.
(b) A nurse sees a pressure-injury risk score when opening the patient's chart.
(c) A wearable detects an irregular rhythm and alerts the patient.

**Deliverable.** A four-row table (including the header) with a pattern and justification per row.

**Acceptance criteria.** Trigger and latency are stated separately; the action locus is a place or
a person, not a team; the chosen pattern follows from the coordinates rather than from preference.

**Time.** 30 minutes.

### Exercise 12.2 — Compute batch utilization

A batch model scores 240,000 members monthly and flags 1.8 percent. The outreach team can review
800 cases per week. Use 4.33 weeks per month. Compute monthly flags, weekly flags, utilization,
and weekly backlog growth. State whether the queue is stable.

**Deliverable.** Four numbers and a one-sentence conclusion.

**Acceptance criteria.** Units stated; utilization compared with 1.0; backlog growth signed
correctly; conclusion distinguishes a stable queue from an unstable one.

**Time.** 25 minutes.

### Exercise 12.3 — Decompose a latency budget

A real-time score must render within 1,400 ms at p99. The components measure 250 ms (page start),
120 ms (integration), 90 ms (auth), 400 ms (feature lookup), 150 ms (model), and 180 ms (render).
Compute the total and the headroom. If the feature lookup doubles, what happens, and what would
you change first?

**Deliverable.** The arithmetic and a two-sentence recommendation.

**Acceptance criteria.** Headroom computed against the p99 budget, not the mean; the doubled
feature lookup is shown to exceed the budget; the recommendation targets the largest component
rather than the model.

**Time.** 25 minutes.

### Exercise 12.4 — Design an interface contract

Write the request and response contract for a real-time deterioration score used by an EHR. Include
required fields, units, error codes for invalid input and unavailable model, the meaning of the
score, and the staleness flag.

**Deliverable.** A contract table plus three example requests and responses.

**Acceptance criteria.** Units are explicit; a missing required field is rejected rather than
imputed; the response distinguishes "no score" from "low risk"; the contract states timeout and
retry behavior.

**Time.** 45 minutes.

### Exercise 12.5 — Choose a fallback

For each pattern, choose fail-closed, fail-open, serve-stale-with-flag, or queue, and justify:
(a) sepsis alert in an EHR; (b) nightly care-gap list; (c) insulin dose recommendation on a pump;
(d) readmission risk shown to a discharge planner.

**Deliverable.** A four-row table with the chosen fallback, the cost of a wrong answer, and the
cost of no answer.

**Acceptance criteria.** The fallback follows from the relative cost of the two error types; stale
answers are flagged; the answer does not assume the clinician will notice.

**Time.** 35 minutes.

### Exercise 12.6 — Diagnose a pattern failure

A batch care-gap model has stable precision at 25 percent, but the outreach yield has fallen from
30 percent to 15 percent. List age p95 is 21 days. Compute capacity utilization if 3,600 cases are
flagged monthly and 700 can be reviewed weekly. Name the failure and the first diagnostic.

**Deliverable.** A 350-word diagnosis with the utilization arithmetic.

**Acceptance criteria.** Correctly identifies the queue rather than the model; computes utilization
above 1.0; recommends a capacity, threshold, or pattern change rather than retraining.

**Time.** 30 minutes.

### Exercise 12.7 — Design an edge rollout

A wearable model update must reach 12,000 devices. Design the rollout: staging, canary fraction,
update-signing, rollback, and the criteria for pausing. State what happens to a device that loses
power during the update.

**Deliverable.** A one-page rollout plan with a stage table.

**Acceptance criteria.** Update is signed and verified on device; a failed update leaves the device
in a bootable state; rollback is possible without recalling devices; pause criteria are numeric.

**Time.** 50 minutes.

### Exercise 12.8 — Shadow versus canary

For a real-time API and for a batch list, compare shadow deployment and canary release. State what
each can detect, what it cannot, and which you would use first for a new model version.

**Deliverable.** A comparison table and a recommendation for each pattern.

**Acceptance criteria.** Distinguishes "no action" from "partial action"; identifies that a batch
canary is a partition or site rather than a traffic fraction; states the cost of each approach.

**Time.** 35 minutes.

## 13. Self-check answers

### 13.1 Exercise 12.1

(a) Trigger: schedule, one day before discharge. Tolerance: hours. Locus: pharmacist worklist.
Pattern: batch. The decision is population-shaped and not latency-sensitive within the day.

(b) Trigger: user request when the chart opens. Tolerance: seconds. Locus: EHR screen. Pattern:
real-time API with precomputed features. The clinician is waiting, so the end-to-end budget
dominates.

(c) Trigger: continuous stream from the device. Tolerance: seconds to a minute, depending on the
clinical claim. Locus: patient app or device actuator. Pattern: edge inference with a local rule
and a connectivity fallback.

### 13.2 Exercise 12.2

```text
monthly_flags = 240,000 × 0.018 = 4,320
weekly_flags  = 4,320 / 4.33 = 997.7
utilization   = 997.7 / 800 = 1.247
growth        = 997.7 − 800 = 197.7 cases per week
```

The queue is unstable. At 1.247, the backlog grows by about 198 cases per week, or 856 per month.
No queue discipline fixes this; the arrival rate exceeds the service rate.

### 13.3 Exercise 12.3

```text
total    = 250 + 120 + 90 + 400 + 150 + 180 = 1,190 ms
headroom = 1,400 − 1,190 = 210 ms
doubled  = 1,190 + 400 = 1,590 ms, which is 190 ms over budget
```

The feature lookup is the largest component and the first target. Options are a batched fetch, a
precomputed feature cache with a freshness bound, or moving part of the computation into the
model's input pipeline. Making the model smaller would save at most 150 ms and would not close a
190 ms gap on its own.

### 13.4 Exercise 12.4

A strong contract states units for every numeric field, uses a 422 for malformed input and a 503
for an unavailable model, returns a score in [0,1] with the model version and threshold, and
includes a `data_age_seconds` field. "No score" is a distinct response, not a score of zero. The
timeout is stated in the contract, and the retry policy is idempotent.

### 13.5 Exercise 12.5

| Decision | Fallback | Reasoning |
|---|---|---|
| Sepsis alert | Fail closed, show "score unavailable" | A stale or default score presented as current could delay escalation; absence is visible and actionable |
| Care-gap list | Serve stale with a date and flag | A one-day-old list is usable; the date lets the reviewer judge |
| Insulin recommendation | Fail closed, revert to the local safety rule | A wrong dose is catastrophic; no answer with a known rule is safer |
| Readmission risk | Serve stale with a flag or hide the score | The discharge planner can proceed without it; a flagged stale score is acceptable |

The pattern is consistent: where a wrong answer can cause immediate harm, fail closed. Where the
decision can tolerate delay, serve stale with an explicit flag.

### 13.6 Exercise 12.6

```text
weekly_flags = 3,600 / 4.33 = 831.4
utilization  = 831.4 / 700 = 1.188
growth       = 131.4 cases per week
```

The model is not the failure. The queue is. The first diagnostic is list age and its correlation
with outreach yield, not precision. The response is to raise capacity above 832 per week, lower
the flag rate, or move to an event-driven pattern that scores at the point of contact. Retraining
would leave the utilization unchanged.

### 13.7 Rubric for the design exercises

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Pattern fit | 25% | Chooses by familiarity | Matches trigger and latency | Quantifies the mismatch cost |
| Interface | 20% | Undefined schema | Typed fields and errors | Versioned contract with staleness and units |
| Failure behavior | 25% | Assumes availability | Names a fallback | Chooses fallback from error costs and tests it |
| Operability | 20% | No rollout plan | Canary or staged rollout | Numeric pause criteria and tested rollback |
| Evidence | 10% | Assertions | Cites a metric | Distinguishes measured from inferred |

### 13.8 Exercise 12.8

Shadow detects whether a new model would have produced different or worse outputs on the same
traffic without exposing anyone to the change. It cannot detect the effect of acting on the
output, because no action occurs. Canary detects the effect of partial action but exposes a subset
of users.

For a real-time API, shadow first, then a small traffic canary. For a batch list, shadow means
scoring the same population and comparing lists without delivery; a canary is a site or a
partition, not a traffic fraction. The batch canary is slower and harder to interpret because the
sites differ.

## 14. Interview questions

**1. How do you choose between batch and real-time?**
Strong answers use trigger, latency tolerance, and action locus, and give an example where a
batch model is correct. Weak answers default to real-time.

**2. What is the latency budget for an EHR-embedded score, and how do you decompose it?**
Strong answers state a percentile and an end-to-end path, including integration and render. Weak
answers quote the model's inference time.

**3. Why can a queue be unstable at 100 percent utilization?**
Strong answers explain arrival versus service rate and the growth of backlog above 1.0. Weak
answers talk about CPU.

**4. What is the difference between streaming and real-time?**
Strong answers separate trigger, window, and end-to-end latency. Weak answers treat them as
synonyms.

**5. How would you roll back an edge model?**
Strong answers describe signed over-the-air updates, staged fleet exposure, and a bootable
fallback. Weak answers say "deploy the previous version."

**6. What does a shadow deployment not tell you?**
Strong answers say it cannot measure the effect of action or clinician response. Weak answers
claim it is equivalent to a canary.

**7. How do you handle late-arriving events in a stream?**
Strong answers describe watermarks, allowed lateness, and recomputation. Weak answers say "process
them when they arrive."

**8. What belongs in an interface contract?**
Strong answers include units, required fields, error semantics, idempotency, timeout, and
staleness. Weak answers list only the JSON fields.

**9. When is an embedded pattern the right choice?**
Strong answers cite the host workflow, the vendor release cycle, and the inability to operate a
separate service. Weak answers say it is faster.

**10. How do you monitor a batch pattern?**
Strong answers monitor delivery time, denominator completeness, and list age. Weak answers monitor
job success.

## 15. Key terms

- **At-least-once delivery** — a guarantee that every message is delivered, possibly more than once.
- **Backpressure** — a mechanism that slows producers when consumers are saturated.
- **Batch scoring** — scoring a population on a schedule and producing a finite output.
- **Canary** — a limited exposure of a change to a subset of traffic or sites.
- **Cold start** — the delay before a new instance can serve requests.
- **Concurrency** — the number of requests in flight at one time.
- **Edge inference** — scoring on hardware near the patient.
- **Embedded model** — a model running inside another product's runtime.
- **Event time** — the time an event occurred.
- **Exactly-once semantics** — processing each event once in effect.
- **Federated learning** — training across sites without centralizing data.
- **Hot path** — computation on the critical path of a user action.
- **Human-in-the-loop** — a person reviews and decides on the output.
- **Idempotency** — repeating a request has the same effect as sending it once.
- **Latency** — time from trigger to answer, reported at a percentile.
- **Offered load** — arrival rate multiplied by service time, in erlangs.
- **Point-of-care** — the moment and place a clinical decision is made.
- **Population health** — scoring a denominator of members or patients.
- **Processing time** — the time the system handled an event.
- **Real-time serving** — scoring on request within a latency budget.
- **Shadow deployment** — running the model without acting on its output.
- **Stateful service** — a service whose output depends on prior events.
- **Streaming** — scoring events continuously, often with windows and state.
- **Throughput** — work completed per unit time.
- **Utilization** — arrival rate divided by service rate; above 1.0 the queue is unstable.
- **Watermark** — a bound on lateness before a window closes.

## 16. Sources

All URLs accessed 26 October 2026.

- HL7. *FHIR R5: SubscriptionTopic resource*. https://hl7.org/FHIR/R5/subscriptiontopic.html
- HL7. *Subscriptions R5 Backport Implementation Guide*, STU 1.1.0. https://hl7.org/fhir/uv/subscriptions-backport/
- HL7. *FHIR Bulk Data Access*. https://hl7.org/fhir/uv/bulkdata/
- HL7. *CDS Hooks*. https://cds-hooks.hl7.org/
- IHE. *Patient Care Device domain*. https://www.ihe.net/resources/technical_frameworks/
- DICOM Standards Committee. *DICOM PS3: Network communication support*. https://www.dicomstandard.org/
- OpenTelemetry. *Semantic conventions for HTTP metrics*. https://opentelemetry.io/docs/specs/semconv/http/http-metrics/
- Kubernetes. *Liveness, Readiness, and Startup Probes*. https://kubernetes.io/docs/concepts/workloads/pods/probes/
- Google. *The Site Reliability Workbook*, Chapter 16, Canarying Releases. https://sre.google/workbook/canarying-releases/
- Sculley D, Holt G, Golovin D, et al. Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems* 28, 2015. https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*. 2021;181(8):1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
- US Food and Drug Administration. De Novo classification request DEN180001 (IDx-DR), 2018. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
- US Food and Drug Administration, Health Canada, and UK MHRA. *Good Machine Learning Practice for Medical Device Development: Guiding Principles*, 2021. https://www.fda.gov/media/153486/download
- US Food and Drug Administration. *Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions*, final guidance, December 2024, reissued August 2025. https://www.fda.gov/media/166704/download
- US Food and Drug Administration. *Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations*, draft guidance, January 2025. https://www.fda.gov/media/184856/download
- IEC 62304:2006/AMD1:2015. *Medical device software — Software life cycle processes*. https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019. *Medical devices — Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
- 45 CFR 164.312, Technical safeguards, HIPAA Security Rule. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/15_deployment_mlops.md`](../../course/15_deployment_mlops.md)
- Previous topic: [`01_the-production-loop.md`](01_the-production-loop.md)
- Next topic: [`03_packaging-and-reproducibility.md`](03_packaging-and-reproducibility.md)
- Related topic: [`05_monitoring.md`](05_monitoring.md)
- Related topic: [`08_rollout-and-rollback.md`](08_rollout-and-rollback.md)
- Labs overview: [`../../labs/`](../../labs/)
- Reference service: [`../../labs/mlops/`](../../labs/mlops/)
- Service endpoints: [`../../labs/mlops/app/main.py`](../../labs/mlops/app/main.py)
- Kubernetes manifests: [`../../labs/mlops/k8s/`](../../labs/mlops/k8s/)
