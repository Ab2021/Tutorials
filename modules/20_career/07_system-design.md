# System Design: Designing a Clinical Data Product in 45 Minutes

> Module 20 — Career, Portfolio, and Interviews
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: 04, 06
> Version 2026.3

## 1. Why this topic exists

The system design interview asks you to design a clinical data product on a whiteboard in 45
minutes. The prompt is usually one sentence: "Design a system that alerts nurses when a patient is
deteriorating." The interviewer is not looking for a complete architecture. They are looking for
whether you can move from a clinical decision to a system that delivers it reliably, and whether
you know where such systems break.

Most candidates produce a diagram of boxes: ingest, feature store, model, application programming interface (API), dashboard. The
diagram is not wrong and it is not a design. A design states the user, the decision, the time
zero, the latency budget, the failure behaviour, the privacy boundary, the change-control process,
and the cost. The boxes are the easy part; the constraints are the interview.

### 1.1 What the interviewer is listening for

1. **Requirements before components.** Do you ask who uses it, what decision it supports, and what
   happens if it is unavailable?
2. **Time zero and data availability.** Do you know which data exist at the moment of prediction?
3. **Trade-offs, stated explicitly.** Do you choose batch versus streaming, build versus buy, and
   explain why?
4. **Failure behaviour.** Do you design the degraded mode before the happy path?
5. **Operational and regulatory realism.** Do you know how the model is monitored, changed, and
   governed?

A candidate who covers requirements, time zero, failure, and monitoring usually passes, even if
the diagram is incomplete. A candidate with a beautiful diagram and no failure plan usually fails.

### 1.2 Why clinical systems are different

Clinical systems have four properties that general software design interviews do not emphasise.

- **The cost of unavailability is clinical, not commercial.** If the alerting service is down, the
  safety net is gone, and someone has to know that.
- **The data are fragmented and late.** Labs arrive hours after collection, notes are signed later,
  and interfaces fail silently.
- **Change is regulated.** A model update in a regulated product may require a predetermined change
  control plan; an unannounced update can invalidate the evidence package.
- **The user is interrupted.** A clinician has seconds, not minutes, and every alert competes with
  the patient in front of them.

A design that ignores these is a generic ML system with clinical labels.

### 1.3 The failure that looks like a design

The common weak answer: "We ingest electronic health record data into a data lake, compute features, train a model,
serve it with an API, and show it in a dashboard. We use Kafka for streaming and Kubernetes for
scaling." Every sentence is a technology choice and none is a decision. There is no user, no
latency requirement, no failure mode, and no explanation of why Kafka is needed when the workload
is 1,200 beds.

The strongest answers are often smaller. "The score is computed every four hours for inpatients,
written to a worklist that the charge nurse reviews at shift huddle, with a pager fallback if the
service is down. The peak load is about one request per second, so a single service with a queue
is sufficient. The hard part is the interface timing, not the scale."

### 1.4 What this topic produces

A design procedure you can execute in 45 minutes, a capacity and latency calculator you can run
before the interview, a worked example with real arithmetic, and the standards vocabulary (Fast Healthcare Interoperability Resources (FHIR),
CDS Hooks, the ONC HTI-1 rule, Good Machine Learning Practice) that makes the design legible to a clinical informatics panel.

### 1.5 The 45-minute clock

The interview is timed, and the time allocation is part of the assessment. A workable split:

| Minutes | Activity |
|---:|---|
| 0 to 6 | Users, decision, constraints |
| 6 to 10 | Sizing arithmetic and integration choice |
| 10 to 20 | Data plane and time zero |
| 20 to 28 | Inference plane and delivery |
| 28 to 36 | Failure modes and degraded mode |
| 36 to 42 | Privacy, monitoring, change control |
| 42 to 45 | Recommendation, risks, what was rejected |

If you reach minute 30 without having discussed failure, cut the component detail and move to it.
The interviewer scores what you covered, not how much you drew.

### 1.6 What this topic does not cover

It does not teach cloud infrastructure or Kubernetes. It does not cover the clinical case
structure (Topic 06), which precedes the design in most loops, or model development (Modules 06 and
15). The regulatory content is orientation; a real deployment requires regulatory and legal review.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Elicit users, decisions, and constraints before drawing components | Exercise 1; self-check 1 |
| 2 | Define time zero and the data available at that moment | Exercise 2; self-check 2 |
| 3 | Choose an integration pattern and justify it against the workflow | Exercise 3; self-check 3 |
| 4 | Compute average and peak request rates, storage, and allowed downtime | Exercise 4; self-check 5 |
| 5 | Design the degraded mode and the failure notification | Exercise 5; self-check 6 |
| 6 | Specify monitoring, drift detection, and change control | Exercise 6; self-check 7 |
| 7 | Draw a privacy boundary and name the data protection controls | Exercise 7; self-check 8 |
| 8 | Deliver a 45-minute design that ends with a recommendation and a risk list | Exercise 8; interview question 10 |

Outcomes 4 and 5 are the ones that separate candidates who have operated a system from those who
have only built models.

The outcomes are assessed in a mock design with a timer and a stranger. Self-assessment is
unreliable here because the design feels complete when the diagram is complete, and the missing
parts are the ones you did not think to draw.

## 3. The core mental model

A clinical data product has three planes. Most candidates design only the middle one.

```text
  DATA PLANE                 INFERENCE PLANE              CONTROL PLANE
  ----------                 ---------------              -------------
  Sources                    Feature computation          Monitoring
  - EHR / ADT                - time-zero rules            - latency, errors
  - Labs                     - missingness                - calibration drift
  - Claims                   - caching                    - alert volume
  - Devices                  Model serving                - override rate
      |                      - batch / stream             - subgroup metrics
      v                      - version pinning                 |
  Ingestion                  - fallback score                  v
  - interfaces (HL7v2,           |                        Governance
    FHIR)                        v                        - change control
  - validation               Delivery                     - audit log
  - late arrival             - worklist / pager           - access control
      |                      - CDS Hooks card             - incident review
      v                      - EHR integration                 |
  Storage                         |                             v
  - raw / curated            Feedback                     Cost and capacity
  - feature history          - action taken               - requests/sec
  - audit log                - outcome                    - storage/year
```

The data plane determines whether the input is trustworthy. The inference plane determines whether
the output is timely and correct. The control plane determines whether anyone notices when either
fails. A design that omits the control plane is a demo.

### 3.1 The four constraints

Every clinical design is shaped by four constraints, and the interviewer will probe at least two.

| Constraint | Typical question | Design consequence |
|---|---|---|
| Latency | How long after the lab result must the alert appear? | Determines batch vs stream and the size of the feature cache |
| Availability | What happens if the service is down for an hour? | Determines the fallback, the on-call model, and the redundancy |
| Privacy | Who may see the output, and where may the data live? | Determines the boundary, the access controls, and the audit log |
| Change control | How does a model update reach production? | Determines versioning, approval, and the regulatory pathway |

The constraints interact. A 60-second latency requirement rules out a nightly batch. A
zero-downtime requirement rules out a single service without a queue. A regulated change-control
requirement rules out continuous retraining without a release process.

### 3.2 Back-of-envelope capacity

Clinical workloads are usually small. The interview rewards candidates who know this, because it
prevents over-engineering.

```text
  scores per day   = beds x scores per bed per day
  average QPS      = scores per day / 86,400
  peak QPS         = average QPS x peak factor (often 5 to 10)
  storage per year = patient-hours per year x features x bytes per feature
  allowed downtime = (1 - uptime target) x 525,600 minutes per year
```

For 1,200 beds and six scores per bed per day, the average is 0.083 requests per second and the
peak is under one. That is a single service, not a cluster. Saying so is a strength, not a
weakness: it shows you can size a system to its workload.

The hard problems in clinical systems are not throughput. They are interface timing, missing data,
duplicate messages, code mapping, alert fatigue, and change control.

### 3.3 Failure is the design

The interviewer will ask "what happens when X fails." Prepare a table of failures before the
interview, and use it as a checklist during the design.

| Failure | Detection | User impact | Mitigation |
|---|---|---|---|
| Source interface stops sending | Message volume monitor, freshness check | Scores computed on stale data | Show data age; suppress scores beyond a threshold |
| Lab arrives late | Event timestamp versus receipt time | Score computed without the result | Recompute on arrival; mark the score as provisional |
| Model service unavailable | Health check, error rate | No scores or stale scores | Serve the last score with a timestamp; page the on-call; fall back to the existing rule |
| Feature store unavailable | Cache miss rate | Elevated latency | Serve from a warm cache; degrade to a simpler feature set |
| Duplicate messages | Idempotency key check | Double counting, duplicate alerts | Deduplicate on message ID |
| Model drift | Calibration and alert volume monitors | Wrong risk estimates | Recalibrate, pause, or withdraw |
| Unauthorised access | Audit log review | Privacy breach | Role-based access, minimum necessary, alerting |
| Bad model release | Canary metrics | Wrong scores at scale | Canary, rollback, version pinning |

A design that names the degraded mode for each row is a design that an operations team can run.

### 3.4 A reference architecture is a strawman, not an answer

It helps to have a default architecture in mind, as long as you treat it as a starting point to be
modified by the constraints.

```text
  interfaces -> validation -> raw store -> feature jobs -> feature store
             -> scoring job/service -> score store -> worklist / CDS card
             -> monitoring, audit, and release control
```

The default is deliberately boring. In an interview, you say: "My default is a micro-batch
pipeline behind the EHR worklist. The constraints that would move me off it are sub-minute
latency, no ability to batch, or a requirement to score on demand inside an order-entry workflow."
That sentence shows you have a default and know what changes it.

### 3.5 Build, buy, or integrate

The interview often includes a build-versus-buy question. The answer depends on where the
differentiation is.

- **Buy** when the capability is commoditised and the vendor's evidence is adequate: interfaces,
  terminology services, general-purpose monitoring.
- **Build** when the differentiation is the clinical decision, the local data, or the workflow
  integration.
- **Integrate** when the model exists but the delivery does not: a vendor model surfaced through a
  local worklist.

The wrong answer is "build everything" or "buy everything." The right answer names the component,
the reason, and the risk of the choice.

> **Note:** In a regulated product, "buy" does not remove responsibility. The deploying
> organisation still owns the clinical claim and the local validation.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Service level indicator (SLI) | A measured quantity such as latency or error rate | Confused with the target | Measure |
| Service level objective (SLO) | The target range for an SLI | Confused with a contractual guarantee | Target |
| Service level agreement (SLA) | A contractual commitment with remedies | Confused with an internal objective | Contract |
| p95 latency | The 95th percentile of response time | Reported as an average | Milliseconds |
| Throughput | Requests processed per unit time | Confused with latency | Requests per second |
| Batch inference | Scores computed on a schedule | Assumed to be too slow for clinical use | Mode |
| Stream inference | Scores computed as events arrive | Assumed to be necessary for all clinical use | Mode |
| Feature store | A system that serves feature values for training and inference | Confused with a data warehouse | Component |
| Time zero | The moment at which prediction is made | Set at data extraction rather than decision | Timestamp |
| Idempotency | The property that repeated identical requests produce one effect | Ignored in message processing | Property |
| Message queue | A buffer between producers and consumers | Added without a reason | Component |
| HL7 v2 | A messaging standard widely used for ADT and orders | Confused with FHIR | Standard |
| FHIR | Fast Healthcare Interoperability Resources; a modern API standard | Assumed to be universally deployed | Standard |
| CDS Hooks | A specification for invoking decision support at workflow points | Confused with FHIR itself | Specification |
| SMART on FHIR | An authorisation and launch framework for health apps | Confused with CDS Hooks | Framework |
| Worklist | A list of patients or tasks presented in the EHR | Confused with a dashboard | UI pattern |
| Degraded mode | The behaviour when a component is unavailable | Omitted from designs | Mode |
| Canary release | A release to a small subset before full rollout | Confused with a pilot | Release strategy |
| Rollback | Reverting to a previous version | Assumed to be instant | Operation |
| Drift | Change in input distribution or relationship to outcome | Confused with a bug | Phenomenon |
| Calibration drift | Change in the mapping from score to observed risk | Detected only with outcomes | Phenomenon |
| Audit log | A record of who accessed or changed what | Treated as optional | Record |
| Role-based access control (RBAC) | Permissions assigned by role | Confused with authentication | Control |
| Minimum necessary | The privacy principle of limiting data to what is needed | Applied only to humans, not services | Principle |
| De-identification | Removal or transformation of identifiers | Assumed to make data shareable | Process |
| Predetermined change control plan (PCCP) | A pre-authorised description of anticipated model changes | Confused with any update process | Regulatory device |
| Good Machine Learning Practice (GMLP) | Ten guiding principles for ML-enabled device development | Treated as a regulation | Guidance |
| HTI-1 | ONC rule on certification, algorithm transparency, and information sharing | Confused with a device regulation | Regulation |
| Decision support intervention (DSI) | ONC term for decision support subject to transparency requirements | Confused with any alert | Regulatory category |
| Intended use | The population, user, and decision a system supports | Assumed to be obvious | Statement |
| Incident review | Structured analysis after a failure | Confused with blame assignment | Process |

## 5. The mechanics, step by step

The procedure is a 45-minute sequence. The first ten minutes decide the design; the remaining
thirty-five make it concrete. If you run out of time, sacrifice component detail, never
requirements or failure behaviour.

### Step 1. Clarify the user, the decision, and the moment

**Input:** the prompt.
**Output:** one sentence naming the user, the decision, the trigger, and the consequence.
**Rationale:** every component choice follows from this sentence. Without it, the design is a
generic pipeline with clinical labels.

Ask: Who is the user (role)? What do they do differently because of the output? When do they see
it? What happens today? What is the cost of a wrong output?

Example: "The charge nurse decides whether to escalate a patient to the rapid response team at the
start of each shift and when a new lab result arrives, using a deterioration score. A missed
escalation can delay care; a false alarm consumes a rapid response team that may be needed
elsewhere."

If the interviewer gives a user but not a decision, ask what the user does with the output. If the
answer is "they look at it," the design has no action and the interview will stall. A dashboard is
not a decision, and saying so early saves twenty minutes.

### Step 3b. Decide the availability and redundancy posture

**Input:** the availability constraint.
**Output:** a redundancy design and an on-call model.
**Rationale:** availability is a cost decision. A 99.9 percent target allows about 8.8 hours of
downtime a year, which is achievable with a single region and a tested fallback. A 99.99 percent
target requires redundancy that a clinical worklist rarely justifies.

Ask what the clinical fallback is. If the answer is "the paper screen," the system can be simpler.
If the answer is "there is no fallback," the system needs redundancy and a much higher
availability target, and the cost rises accordingly.

State the trade-off: "I would target 99.9 percent with a documented paper fallback, because the
cost of 99.99 percent is not justified by a worklist that is reviewed at huddle."

### Step 2. Establish the four constraints

**Input:** the decision statement.
**Output:** a stated value or range for latency, availability, privacy, and change control.
**Rationale:** the constraints determine the architecture more than the model does. State them
before drawing anything.

| Constraint | Question | Example answer |
|---|---|---|
| Latency | How long after the triggering event must the output appear? | Within 15 minutes of a new lab result; score refreshed every 4 hours |
| Availability | What happens if the service is down for an hour? | The charge nurse falls back to the paper sepsis screen; the on-call data engineer is paged |
| Privacy | Who may see the output, and where may data be processed? | Clinical staff with a treatment relationship; data stay in the hospital's cloud tenant |
| Change control | How does a model update reach production? | Quarterly release with a validation report and a canary; no silent retraining |

If the interviewer will not specify a constraint, state an assumption and move on. "I will assume
99.9 percent availability, which allows about 8.8 hours of downtime a year, and I will design the
fallback around that."

### Step 3. Size the workload before choosing technology

**Input:** the population and the refresh frequency.
**Output:** average and peak request rates, storage per year, and allowed downtime.
**Rationale:** clinical workloads are small, and the arithmetic prevents over-engineering. It also
gives you a defensible answer to "how would this scale?"

Use the formulas from Section 3.2. Say the numbers aloud. For 1,200 beds and six scores per bed per
day, the average is under 0.1 requests per second; a single service with a queue is sufficient. The
interviewer will notice that you did the arithmetic and that you did not reach for a streaming
platform by reflex.

### Step 4. Map the data sources and the time zero

**Input:** the decision and the workload.
**Output:** a list of sources, each with the field, the latency, and the availability at time zero.
**Rationale:** the most common design failure is a feature that does not exist at prediction time.
The time-zero table prevents it.

| Source | Fields | Typical latency | Available at time zero |
|---|---|---|---|
| ADT feed | Admission, transfer, discharge | Minutes | Yes |
| Lab interface | Result, collection time, units | 30 minutes to hours | Yes, if collected before the trigger |
| Medication administration | Drug, dose, time | Minutes | Yes |
| Notes | Text | Hours to days | No for a real-time score |
| Claims | Diagnoses, utilisation | Weeks to months | No for an inpatient score |
| Device vitals | Heart rate, respiratory rate | Seconds to minutes | Yes, if the device is connected |

For each field, write the timestamp and the condition under which it is available. "Discharge
diagnosis is available at discharge, which is after the prediction, so it cannot be used" is the
sentence that prevents target leakage in production.

### Step 5. Choose the integration pattern

**Input:** the latency constraint and the workflow.
**Output:** batch, micro-batch, stream, or on-demand, with a justification.
**Rationale:** the pattern determines the operational complexity. Most clinical use cases do not
need streaming, and saying so is a strength.

| Pattern | Latency | When to use | Cost |
|---|---|---|---|
| Nightly batch | Hours | Population reports, panel management | Low |
| Micro-batch (5 to 15 minutes) | Minutes | Score refresh on new results, worklists | Moderate |
| Stream | Seconds | Bedside alerts, device-driven deterioration | High |
| On-demand (API) | Milliseconds to seconds | Clinician opens a patient and requests a score | Moderate, spiky |

The common error is choosing stream because the prompt says "real-time." Ask what the decision
requires. A charge nurse who reviews a worklist at shift change needs a refresh every few hours,
not a sub-second stream.

### Step 6. Design ingestion and validation

**Input:** the source list.
**Output:** a pipeline with validation, deduplication, late-data handling, and code mapping.
**Rationale:** clinical data arrive late, twice, and in the wrong coding system. The pipeline's job
is to make those failures visible rather than to hide them.

Include:

- **Schema and range validation** on every message, with a quarantine queue for failures.
- **Idempotency** on message ID, so a duplicate ADT event does not create a duplicate encounter.
- **Late-arrival handling**: recompute the score when a result arrives after the trigger, and mark
  the score as updated.
- **Code mapping** from local codes to standard terminologies (LOINC for labs, RxNorm for drugs,
  SNOMED or ICD for diagnoses), with an unmapped-code report.
- **Freshness checks** per source, exposed to the control plane.

### Step 7. Compute and cache features

**Input:** the validated data.
**Output:** a feature set with a documented computation and a cache.
**Rationale:** feature computation is where time-zero rules and missingness are enforced. It is
also where training-serving skew appears.

Rules:

1. Every feature has a SQL or code definition, a source, and a timestamp rule.
2. Missing values are represented explicitly, with an indicator, not imputed silently.
3. The same code computes features for training and serving; if that is impossible, the difference
   is documented and tested.
4. Features are cached with a time-to-live tied to the refresh frequency.
5. Training data are generated with the same time-zero rule as serving, ideally by replaying the
   feature pipeline over history.

### Step 8. Serve the model with versioning and a fallback

**Input:** the feature set.
**Output:** a serving design with a pinned model version, a fallback score, and a latency budget.
**Rationale:** the model is a component with a version and a failure mode, not a constant.

State: where the model runs (batch job, service, or embedded), how the version is pinned, how the
service behaves when the model is unavailable, and the latency contribution. A fallback can be the
previous model version, the current clinical rule, or the last score with an age label.

### Step 9. Deliver into the workflow

**Input:** the user and the moment.
**Output:** a delivery mechanism with the channel, the timing, and the action.
**Rationale:** a correct score that arrives in the wrong place is not a product. The delivery
choice determines adoption.

| Mechanism | Fits | Does not fit |
|---|---|---|
| EHR worklist | Scheduled review, batch refresh | Moment-of-order decisions |
| CDS Hooks card | Order entry, patient view, real-time | Population review |
| Pager or secure message | Urgent, single patient | High-volume lists |
| Dashboard | Operational oversight, quality review | Individual clinical decisions |
| API for another system | Integration with a downstream tool | Direct clinician use without a UI |

Name the action: enrol, escalate, order, review, or document. If the output does not change an
action, the design is a report.

The delivery choice should be stated with the hook. For CDS Hooks, the relevant workflow points are
`patient-view`, `order-select`, and `order-sign`; a card returned at `order-sign` can block or warn,
while a card at `patient-view` is informational. Naming the hook and the card's indicator is the
level of specificity that signals you have integrated with an EHR before.

### Step 10. Design the degraded mode

**Input:** the component list.
**Output:** a failure table with detection, user impact, and mitigation for each component.
**Rationale:** this is the step most candidates skip and the one that most clearly signals
operational experience.

A degraded mode has three parts: what the user sees, what the system does, and who is notified. A
design that has only the second part is an engineering plan, not an operational one.

Use the table in Section 3.3. For each failure, state what the user sees. "If the service is down,
the worklist shows the last score with its timestamp and a banner; the charge nurse uses the paper
screen" is a design. "We would have high availability" is not.

### Step 11. Draw the privacy boundary and the access controls

**Input:** the data sources and the users.
**Output:** a boundary diagram with the controls at each crossing.
**Rationale:** the boundary determines what can be logged, where data can be processed, and who can
see the output.

State: the data stay in the covered entity's environment; access is role-based and limited to staff
with a treatment relationship; the audit log records every view of a score; de-identified extracts
are used for development where possible; the vendor, if any, is under a business associate
agreement.

### Step 12. Specify monitoring and change control

**Input:** the deployed design.
**Output:** a monitoring plan and a release process.
**Rationale:** the control plane is what makes the system operable and auditable.

Monitoring: latency p95, error rate, data freshness per source, score volume, alert volume,
override rate, calibration (monthly, with outcomes), subgroup performance (quarterly).

Change control: model versions are immutable, releases are canaried to one unit, rollback is
tested, and any change to the model or its inputs is reviewed against the intended use and, where
applicable, the predetermined change control plan.

### Step 13. State the cost, the risks, and the recommendation

**Input:** the whole design.
**Output:** three cost drivers, three risks, and a recommendation.
**Rationale:** the close is what the interviewer remembers. It should be specific and short.

Example: "The cost is dominated by the interface work and the clinical validation, not compute. The
three risks are stale data during interface outages, alert fatigue at the chosen threshold, and a
model update that invalidates the validation. I would start with a micro-batch worklist at two
sites, measure for three months, and only then consider a real-time card."

A useful close is a three-line decision record:

```text
Decision:   micro-batch worklist at huddle, 15-minute freshness, paper fallback.
Rejected:   streaming, because no decision requires sub-minute latency.
Revisit if: the clinical owner requires vitals-triggered alerts within 60 seconds.
```

The record is short, it names the alternative, and it states the condition under which the design
would change. Interviewers remember candidates who can state what they rejected and why.

### Decision table: data freshness and the fallback score

| Data age at scoring | Display | Action |
|---|---|---|
| Under 1 hour | Normal score | Use as designed |
| 1 to 4 hours | Score with timestamp | Use, note the age |
| 4 to 12 hours | Score marked stale | Use only with clinical judgement; log |
| Over 12 hours | Suppress the score | Fall back to the existing rule; alert the data team |

### Career-context branches

**From academia.** You may have built models and never operated one. Learn the failure table before
the interview; it is the part of the design you cannot infer from modelling experience.

**No clinical background.** Map the workflow before the architecture. A design that ignores who acts on
the output is a data pipeline, not a clinical system.

**Non-traditional degree.** Breadth is the advantage. Cover data, inference, control, and governance
rather than going deep on one component.

**Visa constraints.** No effect on the design; the employer's ability to sponsor may affect which
employers are worth designing for.

**Remote roles.** Design for a distributed on-call team: runbooks, escalation, and asynchronous
monitoring matter more when the team is not in one room.

**Return after a gap.** Refresh one cloud platform and one interface standard; the design patterns are
stable, the tooling is not.

**Over-levelled target.** Expect cost, change control, and regulatory questions. A design without
those reads as mid-level.

### Branch points and edge cases

**No cloud permitted.** Some health systems require on-premises processing. The design changes in
three ways: capacity must be provisioned for peak rather than elastic, the model-serving stack must
be supportable by the local team, and the monitoring must work without a vendor SaaS. State these
consequences rather than assuming the cloud.

**No FHIR, only HL7 v2.** This is the common case in United States hospitals. Design around the
ADT and ORU interfaces, with an integration engine that maps to your internal model. FHIR is the
target state, not the current one.

**Multi-site deployment.** Three hospitals with different EHRs and coding practices. Decide
single-tenant versus multi-tenant, and state how local calibration and local validation are
handled. A single model across sites without local calibration is a common failure.

**Low bandwidth or offline units.** Some units lose connectivity. The design needs a local cache
and a clear statement of how stale a score may be before it is suppressed.

**A vendor model.** The model is bought; the integration, validation, and monitoring are yours.
State the contract terms you need: transparency on inputs, notification of updates, performance
reporting, and the right to validate locally.

**Research-only use.** If the output is not used for care, the design can be simpler, but the
boundary must be explicit: no alerts to clinicians, no protected health information leaving the environment, and a
documented intended use that says research.

**A regulated device.** If the output directs a treatment decision, the design must accommodate the
device pathway: intended use, risk management, software lifecycle, and change control. Say that the
pathway depends on the claim and that regulatory review is required before launch.

**The prompt asks for "real-time."** Ask what decision requires it. If the answer is "the clinician
wants to see it when they open the chart," an on-demand API with a cache is sufficient; a streaming
platform is over-engineering.

### Decision table: integration pattern

| Requirement | Batch | Micro-batch | Stream | On-demand |
|---|---|---|---|---|
| Refresh every few hours | Yes | Yes | Unnecessary | No |
| Alert within 15 minutes of a lab | No | Yes | Yes | No |
| Alert within seconds of a vitals change | No | No | Yes | No |
| Score only when the clinician opens the chart | No | No | No | Yes |
| Population reporting | Yes | No | No | No |

### Decision table: build, buy, or integrate

| Component | Differentiating? | Recommended | Risk |
|---|---|---|---|
| EHR interface | No | Buy or use the vendor's interface | Vendor lock-in |
| Terminology mapping | Partly | Buy the service, own the mapping rules | Unmapped codes |
| Feature pipeline | Yes, if local definitions matter | Build | Maintenance burden |
| Model | Sometimes | Build if the decision is local; buy if validated and monitored | Local performance unknown |
| Delivery UI | Yes, workflow-specific | Integrate into the EHR | Interface fragility |
| Monitoring | No | Buy or use platform tooling | Metric definitions drift |

## 6. Worked example

Rosa is 36, a senior data scientist at a health system, interviewing for a clinical AI engineer
role. The prompt is: "Design a system that alerts nurses when an inpatient is deteriorating,
across our three hospitals." She has 45 minutes and a whiteboard.

### 6.1 The first six minutes

She asks and writes:

1. Who acts? The charge nurse on each unit, at shift huddle and when a new lab results.
2. What do they do? Escalate to the rapid response team, increase monitoring, or call the
   attending.
3. What happens today? A paper early-warning score computed manually at huddle, inconsistently.
4. How fast? Within 15 minutes of a new lab result; the score refreshes every 4 hours.
5. How many beds? 1,200 across three hospitals, about 60,000 admissions a year.
6. What is the harm? A missed deterioration delays care; a false alarm consumes a team that may be
   needed elsewhere.

She writes the decision statement on the board and leaves it there for the whole interview.

### 6.2 The sizing arithmetic

```text
beds                        = 1,200
scores per bed per day      = 6
scores per day              = 7,200
average requests per second = 7,200 / 86,400 = 0.083
peak factor                 = 8
peak requests per second    = 0.67
payload per score           = 2 KB
daily payload               = 14.4 MB
feature rows per year       = 1,200 x 24 x 365 = 10.5 million patient-hours
feature storage per year    = 10.5M x 150 features x 8 bytes = 12.6 GB before overhead
uptime 99.9%                = 8.77 hours downtime per year
uptime 99.95%               = 4.38 hours downtime per year
```

She says the conclusion aloud: "This is a small workload. A single service with a queue and a warm
cache is sufficient. The hard parts are interface timing and alert fatigue, not scale."

That sentence changes the interview. The interviewer stops probing for distributed systems and
starts probing for workflow, which is where the role's actual difficulty is.

### 6.3 Diagnostic 1: the first design violated time zero

Her first draft, prepared before the interview, computed features "from the patient record at
prediction time." In a mock, the interviewer asked which fields were available at 06:00 on the day
of the huddle. Her list included the discharge diagnosis, the day's imaging report, and a lab
result collected at 09:00.

The design was leaking the future. In production, those fields would be missing or stale, and the
model's performance would collapse. The offline evaluation had looked excellent for the same
reason.

The fix was a time-zero table, one row per feature with the timestamp at which it becomes
available, and a rule: any feature collected after the trigger is excluded, and any feature that
arrives late triggers a recomputation, not a silent insertion. She redrew the data plane around
that table, and the model's realistic performance dropped by about 4 points of area under the receiver operating characteristic curve points. That drop was
the honest number.

### 6.4 Diagnostic 2: the first design had no degraded mode

Her second draft described a service with "high availability" and no fallback. In the mock, the
interviewer asked: "The service is down for two hours. What does the charge nurse see?"

Her answer was "nothing." That is the worst possible behaviour, because silence is indistinguishable
from "no patients are deteriorating." Nurses would infer safety from an outage.

The revised design:

| Failure | What the user sees | What the system does |
|---|---|---|
| Model service down | Banner: "Scores unavailable since 08:10; use the paper screen" | Pages the on-call engineer; serves the last score with its age |
| Lab interface delayed | Score marked "provisional: 2 labs pending" | Recomputes on arrival; logs the delay |
| Feature store miss | Score computed from the last cached features | Marks the score as stale if the cache is older than 4 hours |
| Duplicate ADT message | No visible change | Deduplicates on message ID |
| Model released with a defect | Canary unit reverts | Rollback to the pinned previous version |

### 6.5 The full design, in the order she presented it

1. **Data plane.** ADT and lab interfaces through the integration engine; validation and
   quarantine; idempotent message handling; terminology mapping to LOINC and local concepts;
   feature computation with a time-zero table; a 4-hour cache with recomputation on late labs.
2. **Inference plane.** A micro-batch job every 5 minutes and on lab arrival; a pinned model
   version; a fallback to the previous version; scores written to a feature-and-score table.
3. **Delivery.** A worklist in the EHR, sorted by unit and score, refreshed at huddle and on lab
   arrival; a pager only for the top band; the action is escalation to the rapid response team.
4. **Control plane.** Freshness monitors per source; latency p95; score and alert volume; override
   rate; monthly calibration with outcomes; quarterly subgroup review; canary releases; a rollback
   that has been tested.
5. **Privacy.** Data stay in the health system's tenant; role-based access limited to clinical
   staff; audit logging on every score view; de-identified extracts for development.
6. **Cost.** Interface work and clinical validation dominate; compute is a few hundred dollars a
   month at this volume.

### 6.6 The close

"The design is a micro-batch worklist, not a streaming platform, because the decision happens at
huddle and within 15 minutes of a lab. The three risks are stale data during interface outages,
alert fatigue if the threshold is set too low, and a model update that invalidates the local
validation. I would pilot at two hospitals for three months with a silent mode comparison, then
turn on the worklist at one unit, measure override rate and time-to-escalation, and expand only if
both hold."

### 6.7 Vary one assumption

Change one input: the chief nursing officer says the alert must appear within 60 seconds of a
vitals change, because the deteriorating patients are identified by respiratory rate on continuous
monitoring.

The design changes in four ways:

1. **The integration pattern moves from micro-batch to stream.** Device vitals arrive as a feed;
   the score is recomputed on each new observation, with a debounce to avoid scoring on every
   heartbeat.
2. **The latency budget is decomposed.** 60 seconds total: 5 seconds for device ingestion, 20 for
   feature computation, 20 for inference, 10 for delivery, 5 for buffer. The feature computation
   moves to a streaming operator with a sliding window.
3. **The fallback changes.** With a 60-second requirement, a 4-hour-old score is not a fallback; it
   is a different product. The degraded mode becomes the existing bedside monitor alarm, and the
   design must state that explicitly.
4. **The cost and operational burden rise.** Streaming infrastructure, on-call for the stream, and
   a new class of failure (late or out-of-order events) enter the design. The interviewer will ask
   whether the clinical benefit justifies it, and the honest answer is that it depends on whether
   the 60-second requirement is real or aspirational.

The general lesson: latency is the constraint that changes the architecture most, and it should be
traced to a clinical decision rather than accepted as a slogan.

### 6.8 The cost estimate she gave

She estimated the first-year cost in three lines, which is more than most candidates offer:

| Cost driver | Estimate | Basis |
|---|---:|---|
| Interface build and validation | 60 to 90 person-days | Three hospitals, two interface types, terminology mapping |
| Clinical validation and safety review | 20 to 30 person-days | Silent-mode analysis, clinician review, governance |
| Compute and storage | Low hundreds of USD per month | 0.67 peak requests per second, 12.6 GB of features per year |
| Operations and on-call | 0.2 FTE | Paging, incident review, quarterly recalibration |

The point is not the precision. It is that the design's cost is in integration and validation, not
in the model, and a candidate who says so understands where clinical AI projects actually spend
their budget.

### 6.9 What the interviewer probed after the design

| Probe | What she said | What it tested |
|---|---|---|
| "Why not streaming?" | The decision is at huddle and within 15 minutes of a lab; micro-batch meets it at a fraction of the operational cost | Whether the pattern followed the requirement |
| "What if the lab interface is down?" | Scores are marked provisional, the data team is paged, and the paper screen remains | Whether the degraded mode was designed |
| "How do you know it works?" | Silent-mode comparison for three months, then override rate and time-to-escalation | Whether evaluation is prospective |
| "How do you update the model?" | Pinned versions, canary at one unit, rollback tested, change reviewed against intended use | Whether change control is real |
| "Who can see the score?" | Clinical staff with a treatment relationship; audit logging; de-identified extracts for development | Whether privacy is a boundary, not a sentence |
| "What does it cost?" | Interface work and validation dominate; compute is a few hundred dollars a month | Whether cost was considered |

### 6.10 What she would add with another week

If the design moved from interview to project, the first week would be spent on three things she
could only name in the interview: the interface inventory (which messages exist, at what latency,
with what error rates), the clinical workflow observation (who actually reads the worklist and
when), and the data access agreement for retrospective development. None of them is modelling, and
all three determine whether the model matters.

### 6.11 What she would do with 15 minutes

1. Decision and user (2 minutes).
2. Latency and availability constraints (2 minutes).
3. Sizing arithmetic (2 minutes).
4. Data plane and time zero (3 minutes).
5. Delivery and degraded mode (3 minutes).
6. Monitoring and close (3 minutes).

The components compress; the constraints and the failure behaviour do not. A 15-minute design that
names the user, the latency, the time zero, and the fallback is stronger than a 45-minute diagram
with no failure plan.

## 7. Code, query, or calculation

The script computes the workload, availability, and latency arithmetic from Sections 3.2 and 5,
Step 3. It is the calculation to run before a system design interview, so that the numbers are
available without mental arithmetic under pressure.

Assumptions:

- Bed count, scores per bed, peak factor, and payload are illustrative for a three-hospital system.
- The latency budget is 900 seconds, matching the 15-minute requirement in Section 6.
- Storage is computed before replication and indexing overhead, which typically multiply it by 2 to
  4.

```python
"""Capacity and latency budget calculator, standard library only.

Run: python capacity_budget.py
Computes request rates, storage, allowed downtime, and a latency budget for a
clinical scoring system. All inputs are stated in one place so they can be
changed and re-run during interview preparation.
"""

BEDS = 1200
SCORES_PER_BED_PER_DAY = 6
PEAK_FACTOR = 8
PAYLOAD_KB = 2
FEATURES = 150
BYTES_PER_FEATURE = 8
UPTIME_TARGET = 0.999
LATENCY_BUDGET_SECONDS = 900  # 15 minutes

# Stage name, seconds allocated.
LATENCY_STAGES = [
    ("device or lab ingestion", 120),
    ("validation and deduplication", 30),
    ("feature computation", 300),
    ("model inference", 60),
    ("delivery to worklist", 60),
]


def main():
    scores_per_day = BEDS * SCORES_PER_BED_PER_DAY
    avg_qps = scores_per_day / 86_400
    peak_qps = avg_qps * PEAK_FACTOR
    daily_payload_mb = scores_per_day * PAYLOAD_KB / 1024
    patient_hours_per_year = BEDS * 24 * 365
    feature_gb_per_year = patient_hours_per_year * FEATURES * BYTES_PER_FEATURE / 1e9
    downtime_minutes = (1 - UPTIME_TARGET) * 525_600

    print("WORKLOAD")
    print(f"  beds: {BEDS}")
    print(f"  scores per day: {scores_per_day:,}")
    print(f"  average requests per second: {avg_qps:.3f}")
    print(f"  peak requests per second: {peak_qps:.2f}")
    print(f"  score payload per day: {daily_payload_mb:.1f} MB")
    print(f"  patient-hours per year: {patient_hours_per_year:,.0f}")
    print(f"  feature storage per year: {feature_gb_per_year:.1f} GB (before overhead)")
    print()
    print("AVAILABILITY")
    print(f"  target: {UPTIME_TARGET:.3%}")
    print(f"  allowed downtime: {downtime_minutes:.1f} minutes per year "
          f"({downtime_minutes / 60:.1f} hours)")

    print()
    print("LATENCY BUDGET")
    print(f"  budget: {LATENCY_BUDGET_SECONDS} seconds")
    total = 0
    for name, seconds in LATENCY_STAGES:
        total += seconds
        print(f"  {name:30s} {seconds:4d} s  ({seconds / LATENCY_BUDGET_SECONDS:5.1%})")
    print(f"  {'total':30s} {total:4d} s")
    slack = LATENCY_BUDGET_SECONDS - total
    print(f"  slack: {slack} s ({slack / LATENCY_BUDGET_SECONDS:.1%})")
    if slack < 0:
        print("  WARNING: the design exceeds its latency budget.")
    elif slack < 0.1 * LATENCY_BUDGET_SECONDS:
        print("  NOTE: less than 10 percent slack; any stage regression breaks the budget.")

    print()
    print("WHAT THE NUMBERS IMPLY")
    print("  A sub-1 QPS peak does not justify a streaming platform.")
    print("  A 15-minute budget is met by a 5-minute micro-batch with recomputation on late labs.")
    print("  The binding constraint is interface freshness, not compute.")


if __name__ == "__main__":
    main()
```

Expected output, confirmed by running the script:

```text
WORKLOAD
  beds: 1200
  scores per day: 7,200
  average requests per second: 0.083
  peak requests per second: 0.67
  score payload per day: 14.1 MB
  patient-hours per year: 10,512,000
  feature storage per year: 12.6 GB (before overhead)

AVAILABILITY
  target: 99.900%
  allowed downtime: 525.6 minutes per year (8.8 hours)

LATENCY BUDGET
  budget: 900 seconds
  device or lab ingestion         120 s  (13.3%)
  validation and deduplication     30 s  ( 3.3%)
  feature computation             300 s  (33.3%)
  model inference                  60 s  ( 6.7%)
  delivery to worklist             60 s  ( 6.7%)
  total                           570 s
  slack: 330 s (36.7%)

WHAT THE NUMBERS IMPLY
  A sub-1 QPS peak does not justify a streaming platform.
  A 15-minute budget is met by a 5-minute micro-batch with recomputation on late labs.
  The binding constraint is interface freshness, not compute.
```

Reading the output:

1. **Peak load is 0.67 requests per second.** A single service with a queue handles this. A
   candidate who proposes Kafka for this workload is either preparing for a different scale or has
   not done the arithmetic.
2. **Feature storage is 12.6 GB per year before overhead.** Even with replication and indexing, the
   data volume is trivial. The design's cost is integration, not infrastructure.
3. **Allowed downtime at 99.9 percent is 8.8 hours per year.** That is the budget for a tested
   fallback, not for heroic recovery. If the clinical owner requires less, the architecture and the
   on-call model change.
4. **The latency budget has 36.7 percent slack.** The slack is the reserve for retries and queueing.
   A design that allocates 100 percent of the budget to stages has no room for a slow lab interface.
5. **The binding constraint is interface freshness.** Feature computation takes five minutes in the
   budget, but the lab may take hours to arrive. The design must handle stale data, not faster
   maths.

To use the script in preparation, change the inputs to your target system and run it before the
interview. Write the four headline numbers on a card: peak QPS, storage per year, allowed downtime,
and the largest latency stage. Those four answers cover most of the quantitative probes.

## 8. Data, standards, and regulatory context

System design in health IT is constrained by interoperability standards and, when the output
influences care, by device regulation. The interviewer will not expect a legal opinion, but they
will expect you to know which standard applies to which part of the design.

### 8.1 Interoperability standards

| Standard | What it defines | Where it fits in the design |
|---|---|---|
| HL7 v2 | Messaging for ADT, orders, and results | The common interface in United States hospitals |
| FHIR R4 | Resources and REST APIs for clinical data | The target API for new integrations; the data model for CDS |
| CDS Hooks 2.0.1 | Invocation of decision support at workflow points | The delivery mechanism for real-time cards |
| SMART on FHIR | Authorisation and app launch | How a clinical app obtains scoped access to FHIR data |
| LOINC | Laboratory and observation codes | Mapping lab results to a common concept |
| SNOMED CT | Clinical terminology | Mapping diagnoses and findings |
| ICD-10-CM/PCS | Diagnosis and procedure coding | Claims and administrative cohorts |
| RxNorm | Drug terminology | Medication features and interactions |
| OMOP CDM | Common data model for analytics | The warehouse layer for training data |

CDS Hooks defines services, hooks, and cards. A CDS service exposes a discovery endpoint and an
invocation endpoint; the EHR calls a hook such as `patient-view` or `order-sign` with context and
optional prefetch data, and the service returns cards with a summary, an indicator (`info`,
`warning`, or `critical`), and optional suggestions. The specification recommends HTTPS, short-lived
OAuth 2.0 tokens, and a prefetch response target of about 500 milliseconds. That target is a useful
number in a design interview: it tells you the delivery step must be fast even when the underlying
scoring is batch.

### 8.2 Transparency and certification

The ONC Health Data, Technology, and Interoperability: Certification Program Updates, Algorithm
Transparency, and Information Sharing final rule (HTI-1), published in the Federal Register on 9
January 2024 and effective 8 February 2024, replaced the clinical decision support certification
criterion with a decision support intervention criterion at 170.315(b)(11).

For predictive decision support interventions, the rule specifies 31 source attributes across nine
categories, including intervention details, purpose, out-of-scope use, development and input
features, fairness process, external validation, quantitative performance, ongoing maintenance, and
an update schedule. It also requires an intervention risk management process covering validity,
reliability, robustness, fairness, intelligibility, safety, security, and privacy, with summary
information publicly posted.

The practical consequence for a design: if the product is certified health IT, the model's
documentation is part of the product, and the design must include the data needed to populate the
source attributes. A model card is not optional decoration; it is a regulatory artifact.

### 8.3 Device regulation and good practice

The FDA, Health Canada, and the UK MHRA jointly published ten guiding principles for Good Machine
Learning Practice in October 2021. They cover multidisciplinary expertise, software engineering and
security, representative data, independent training and test sets, reference standards, intended-use
alignment, human-AI team performance, clinically relevant testing, clear user information, and
post-deployment monitoring.

The principles are guidance, not regulation, but they describe the expectations a device reviewer
will apply. Two are especially relevant to system design: the focus on the performance of the
human-AI team, and the requirement to monitor deployed models and manage retraining risk. Both are
control-plane requirements, which is why a design without monitoring is incomplete.

A predetermined change control plan (PCCP) is the mechanism by which a manufacturer can describe
anticipated modifications in advance. If the design includes continuous retraining, the PCCP is the
relevant regulatory instrument, and the architecture must support the versioning and validation
that the plan promises.

### 8.4 Privacy and security

In the United States, the Health Insurance Portability and Accountability Act Security Rule requires administrative, physical, and technical
safeguards for electronic protected health information, including access control, audit controls,
integrity, and transmission security. In the European Union, the GDPR adds lawful basis, data
minimisation, and records of processing. A design should name the boundary, the access model, and
the audit log, and should state where data are processed.

### 8.5 What the design must expose to governance

Even outside a certified product, a clinical design should expose four things: the intended use,
the model version and its validation date, the input features and their time-zero rules, and the
monitoring results. These are the artifacts a safety committee, an inspector, or an incident review
will ask for.

If the design cannot produce them on demand, the governance process will rely on memory and
email, which is how undocumented model changes reach production.

### 8.6 What is binding and what is convention

Binding: the HIPAA Security Rule for covered entities, the GDPR where applicable, the device
regulations if the product meets the device definition, and the certification criteria if the
product is certified health IT.

Convention: FHIR version choice when both ends support R4, the use of CDS Hooks versus a custom
API, the choice of queue, and the internal data model. These are engineering decisions judged on
fit, not compliance.

The distinction matters in an interview because a candidate who presents a convention as a
requirement sounds inflexible, and a candidate who presents a requirement as a convention sounds
unsafe.

## 9. Industry practice and case evidence

### 9.1 Documented: standards adoption is uneven

A 2021 survey in the Yearbook of Medical Informatics (Taber et al.) reviewed 44 studies of clinical
decision support using modern standards. FHIR appeared in 98 percent, SMART in 50 percent, CDS
Hooks in 18 percent, and CQL in 5 percent. Of the studies, 55 percent were at the design stage, 34
percent piloting, and 11 percent deployed; only 27 percent reported an evaluation.

The sample is a literature review, not a census, and publication bias favours design-stage work.
The direction is still informative: the standards exist and are referenced widely, but deployment
and evaluation lag. A design interview that assumes FHIR and CDS Hooks are everywhere is likely to
meet a hospital that still runs HL7 v2 interfaces and a custom worklist.

### 9.2 Documented: transparency requirements are specific

The HTI-1 final rule's 31 source attributes for predictive decision support interventions are
published in the Federal Register and summarised in ONC fact sheets. The rule also requires that
certified health IT allow a limited set of users to access, record, and change the source attribute
information, and that developers review and update it under maintenance of certification.

This is a concrete example of a regulatory requirement shaping a design: the system must store and
expose model documentation, not just model outputs.

### 9.3 Documented: the failure modes are public

The Epic Sepsis Model external validation (Wong et al., JAMA Internal Medicine 2021), discussed in
Topic 06, is also a system design case. The model was deployed at scale and generated alerts on 18
percent of hospitalisations, with 12 percent PPV at the alert threshold. The design implication is
that alert volume and PPV at the operating point are architecture-level decisions, not
afterthoughts.

### 9.4 Vendor platforms and EHR-native tools

Most health systems do not build the whole stack. They buy an EHR-native risk score, a
vendor worklist, or a cloud health platform, and build only the local validation and the workflow.
That changes the design interview's emphasis: the questions become about contract terms, local
validation, data access, and exit.

The contract questions worth raising in an interview: does the vendor provide the input features
and their update schedule; will it notify you of model changes; does it provide subgroup
performance; can you export scores and outcomes; and what happens to your data if you leave.
A candidate who asks these is read as someone who has managed a vendor rather than only used one.

### 9.5 Inference, labelled

- **Inference:** most clinical AI designs in interviews are over-engineered on compute and
  under-engineered on interfaces and failure handling, because candidates practise modelling more
  than operations.
- **Inference:** the strongest differentiator in a system design interview is a named degraded mode,
  because it is the part of the design that cannot be copied from a blog post.
- **Inference:** interviewers from health systems weight integration and workflow more heavily than
  interviewers from vendors, who weight scalability and model serving more heavily. Ask who is on
  the panel and adjust the emphasis.
- **Inference:** a candidate who cites HTI-1 or GMLP accurately is read as having worked in a
  governed environment, because those names are learned on the job rather than in a course.

> **Note:** No uptime, adoption, or cost benchmark is quoted as a market fact. The numbers in
> Section 6 are illustrative for one hypothetical three-hospital system.

The absence of a benchmark is itself a finding: clinical system design has few public, comparable
operational datasets, so designs are judged on internal consistency and on whether the constraints
were named, not against an industry baseline.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Components before requirements | The first sentence names Kafka | Technology-first training | Design answers no stated decision | Ask user, decision, latency, availability first |
| No time zero | Features include post-decision data | Offline evaluation hides it | Performance collapses in production | Time-zero table per feature |
| Over-engineering | A cluster for sub-1 QPS | Scale assumed, not computed | Cost and operational burden without benefit | Do the arithmetic aloud |
| No degraded mode | "High availability" and nothing else | Availability treated as a property | Silent outage looks like safety | Design what the user sees when it fails |
| Ignoring interface latency | The lab is assumed to arrive instantly | Data treated as real-time | Scores computed on stale data | Freshness monitors and provisional scores |
| Duplicate messages | Encounter counts drift upward | No idempotency key | Wrong denominators and duplicate alerts | Deduplicate on message ID |
| No model versioning | "We retrain weekly" | Continuous change without control | Evidence invalidated silently | Pin versions; canary; rollback |
| Privacy boundary vague | "Data are secure" | No named controls | Compliance failure | Named boundary, RBAC, audit log, BAAs |
| Alert fatigue unaddressed | Volume never computed | Threshold chosen statistically | Override behaviour within weeks | Compute alerts per clinician per shift |
| No monitoring | "We would monitor it" with no metric | Control plane omitted | Drift undetected for months | Named metrics, frequency, thresholds |
| No cost estimate | The design ends at the diagram | Cost treated as someone else's problem | Business case fails late | Three cost drivers with orders of magnitude |
| Ignoring the EHR reality | Assumes FHIR everywhere | Standards assumed | Integration slips by months | Ask which interfaces exist today |
| No clinical owner | No one named for the workflow | The design is a data project | No adoption | Name the role and the action |
| Vendor lock-in ignored | No exit plan or data export | Procurement treated as the end | Cannot change vendors or validate | Ask about export, notification, and feature documentation |
| No capacity for the alert load | Nurses receive 40 alerts per shift | Volume never computed | Workflow abandoned | Compute alerts per clinician per shift |

Two red flags are about the prompt. A design question with no user and no decision, where the
interviewer refuses to clarify, is a stress test; state your assumptions and proceed. A prompt that
keeps adding requirements every five minutes is testing whether you can hold a design under change;
restate the constraints and say what the new requirement breaks.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The decision is known or askable | The design depends on it | Exploratory prompts | Ask who acts; state an assumption if refused |
| Latency can be traced to a clinical need | It prevents over-engineering | "Real-time" is a stakeholder slogan | Ask what decision fails at a slower refresh |
| Capacity is small in clinical settings | It justifies simple architecture | Multi-region research platforms | Compute from beds and refresh rate |
| 99.9 percent is an acceptable target | It bounds redundancy cost | Life-critical closed-loop systems | Ask what the clinical fallback is |
| Interfaces are the hard part | It focuses the design | Greenfield FHIR-native sites | Ask which interfaces exist today |
| The model will be monitored with outcomes | It justifies the control plane | No outcome data available | Ask how outcomes are captured |
| Cloud processing is permitted | It simplifies the design | On-premises requirements | Ask about data residency early |
| A human reviews the output | It shapes the regulatory pathway | Fully automated action | Ask who can override |
| The cost is dominated by integration | It sets expectations | Very large scale or custom hardware | Compare person-days with compute cost |
| The interviewer values breadth over completeness | It justifies the time split | Some panels want depth in one area | Ask which part they want to go deeper on |

The largest limit is that a 45-minute design cannot be complete. The interviewer knows this. The
goal is to demonstrate that you know which parts are missing and why, not to produce a buildable
specification. Saying "I have not covered cost or the terminology mapping; I would address both
before committing" is stronger than pretending the design is finished.

## 12. Exercises

**Exercise 1. Requirements interview (application, 20 minutes).**
Deliverable: ten questions you would ask before drawing anything, with the design consequence of
each answer.
Acceptance criteria: at least three questions concern the decision or user; at least two concern
failure; each question has a stated consequence.

**Exercise 2. Time-zero table (application, 30 minutes).**
Deliverable: a table of 12 features with source, timestamp, and availability at time zero.
Acceptance criteria: every feature has a timestamp rule; at least two features are excluded because
they are unavailable; the exclusion is explained.

**Exercise 3. Integration choice (application, 30 minutes).**
Deliverable: a recommendation of batch, micro-batch, stream, or on-demand for three scenarios, with
a justification and a rejected alternative.
Acceptance criteria: each choice follows from a stated latency requirement; the rejected
alternative is named; the operational cost is mentioned.

**Exercise 4. Sizing (application, 30 minutes).**
Deliverable: a hand calculation of average QPS, peak QPS, annual storage, and allowed downtime for
a system you know.
Acceptance criteria: assumptions are stated; arithmetic is correct; the conclusion about
architecture follows from the numbers; the result is within 10 percent of the Section 7 script.

**Exercise 5. Failure table (design, 45 minutes).**
Deliverable: eight failures with detection, user impact, and mitigation.
Acceptance criteria: each failure has a user-visible consequence; at least two are interface
failures; at least one is a model failure; the mitigation is testable.

**Exercise 6. Monitoring plan (design, 30 minutes).**
Deliverable: a one-page plan with metrics, frequency, thresholds, and owner.
Acceptance criteria: at least one leading indicator and one outcome indicator; thresholds are
numeric; the owner is a role, not a team.

**Exercise 7. Privacy boundary (application, 30 minutes).**
Deliverable: a boundary diagram with controls at each crossing and a data retention statement.
Acceptance criteria: access is role-based; audit logging is specified; the processing location is
named; retention is stated.

**Exercise 8. 45-minute mock design (design, 45 minutes).**
Deliverable: a recorded design with the Section 5 order, ending in a recommendation and three
risks.
Acceptance criteria: requirements before components; time zero defined; one degraded mode per
critical component; monitoring and change control included; the close names what was rejected.

## 13. Self-check answers

**1. What do you ask before drawing a component?**
Who uses the output, what decision it changes, when they see it, what happens today, the latency
requirement, the availability requirement, the privacy boundary, and how the model is updated.

**2. Why is time zero the most important design decision?**
Because it determines which features exist. A feature recorded after the decision is unavailable in
production and inflates offline performance. The time-zero table prevents leakage and defines the
feature set.

**3. When is streaming justified?**
When a clinical decision requires sub-minute response to an event, such as a device-driven
deterioration alert. It is not justified by the word "real-time" in a prompt, and it is rarely
justified by a worklist reviewed at huddle.

**4. Compute the peak QPS for 1,200 beds and six scores per bed per day with a peak factor of 8.**
7,200 scores per day divided by 86,400 seconds is 0.083 average; multiplied by 8 is 0.67 peak
requests per second.

**5. What is the allowed downtime at 99.9 percent?**
About 525.6 minutes per year, or 8.8 hours. At 99.95 percent it is about 4.4 hours.

**6. What does the user see when the model service is down?**
The last score with its timestamp and a banner stating that scores are unavailable, plus the
fallback instruction. The on-call engineer is paged. Silence is the wrong behaviour because it is
indistinguishable from safety.

**7. What belongs in the monitoring plan?**
Latency p95, error rate, source freshness, score and alert volume, override rate, calibration with
outcomes, subgroup performance, and the thresholds and owners for each.

**8. What is the difference between a canary and a pilot?**
A canary is a small-scope release of a version to detect defects before full rollout. A pilot is a
limited deployment to evaluate clinical and operational value. A canary can last hours; a pilot
lasts months. Confusing the two produces either a pilot that never measures value or a canary that
is treated as evidence.

**Rubric for Exercise 3.** Full marks: three scenarios; each choice traces to a latency
requirement; a rejected alternative is named for each; operational cost is mentioned; at least one
scenario correctly rejects streaming.

**Rubric for Exercise 5.** Full marks: eight failures; user-visible impact for each; at least two
interface failures and one model failure; mitigations are testable; at least one failure has no
perfect mitigation and says so.

**Rubric for Exercise 8.** Full marks: requirements before components; arithmetic for scale; time
zero; degraded modes; monitoring and change control; a recommendation with three risks and one
rejected alternative; finishes within 45 minutes.

## 14. Interview questions

**1. "Design a deterioration alerting system."**
Strong: user, decision, latency, time zero, delivery, failure, monitoring. Weak: a pipeline diagram
with no user.

**2. "Why did you choose micro-batch over streaming?"**
Strong: the decision needs 15-minute freshness, the workload is under 1 QPS, and streaming adds
operational cost without clinical benefit. Weak: says micro-batch is simpler, with no requirement
trace.

**3. "What happens if the lab interface goes down?"**
Strong: provisional scores, freshness banner, page the data team, fallback to the paper screen.
Weak: says the interface is reliable.

**4. "How do you prevent training-serving skew?"**
Strong: the same feature code for both paths, time-zero replay, and a parity test. Weak: says they
use a feature store.

**5. "How would you scale this to 50 hospitals?"**
Strong: multi-tenancy, local calibration, per-site validation, data residency, and a support model.
Weak: says "add more servers."

**6. "Where does PHI live in your design?"**
Strong: names the boundary, the processing location, RBAC, audit logging, and the BAA. Weak: says
the data are encrypted.

**7. "How do you update the model?"**
Strong: pinned versions, validation, canary, rollback, change review, and a PCCP if regulated.
Weak: says they retrain weekly.

**8. "What is the biggest risk in this design?"**
Strong: names a specific risk with a mechanism and a mitigation. Weak: says security.

**9. "How would you know the alert is working?"**
Strong: override rate, time-to-escalation, calibration, and a prospective comparison. Weak: says
they would monitor accuracy.

**10. "How would you decommission this system?"**
Strong: export the scores and outcomes, notify users, keep the audit log, and define the fallback.
Weak: says they would turn it off.

**11. "What did you leave out, and why?"**
Strong: names the omission and the reason, and says what would be needed before building. Weak:
claims the design is complete.

## 15. Key terms

- **Audit log** — record of who accessed or changed what.
- **Batch inference** — scores computed on a schedule.
- **Calibration drift** — change in the score-to-risk mapping.
- **Canary release** — release to a small subset before full rollout.
- **CDS Hooks** — specification for invoking decision support at workflow points.
- **Degraded mode** — behaviour when a component is unavailable.
- **Drift** — change in input distribution or outcome relationship.
- **Feature store** — system serving feature values for training and inference.
- **FHIR** — Fast Healthcare Interoperability Resources.
- **GMLP** — Good Machine Learning Practice guiding principles.
- **HL7 v2** — messaging standard for ADT, orders, and results.
- **HTI-1** — ONC rule on certification and algorithm transparency.
- **Idempotency** — repeated identical requests produce one effect.
- **Intended use** — population, user, and decision a system supports.
- **Micro-batch** — frequent small batches, typically minutes.
- **Minimum necessary** — privacy principle of limiting data to what is needed.
- **p95 latency** — 95th percentile response time.
- **PCCP** — predetermined change control plan.
- **RBAC** — role-based access control.
- **Rollback** — reverting to a previous version.
- **SLA** — contractual service commitment.
- **SLI** — measured service quantity.
- **SLO** — target range for an SLI.
- **SMART on FHIR** — authorisation and launch framework for health apps.
- **Stream inference** — scores computed as events arrive.
- **Throughput** — requests per unit time.
- **Time zero** — moment at which prediction is made.
- **Worklist** — list of patients or tasks in the EHR.

## 16. Sources

All URLs accessed 26 October 2026.

1. U.S. Food and Drug Administration, Health Canada, and UK MHRA. *Good Machine Learning Practice
   for Medical Device Development: Guiding Principles*, October 2021.
   https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
2. Office of the National Coordinator for Health Information Technology. *HTI-1 Final Rule.*
   https://healthit.gov/regulations/hti-rules/hti-1-final-rule/
3. Federal Register. *Health Data, Technology, and Interoperability: Certification Program
   Updates, Algorithm Transparency, and Information Sharing*, 9 January 2024.
   https://www.federalregister.gov/documents/2024/01/09/2023-28857/
4. ONC. *HTI-1 Decision Support Interventions Fact Sheet.*
   https://healthit.gov/wp-content/uploads/2023/12/HTI-1_DSI_fact-sheet_508.pdf
5. HL7. *CDS Hooks v2.0.1.* https://cds-hooks.hl7.org/
6. HL7. *FHIR R4: Clinical Reasoning and CDS on FHIR.*
   https://hl7.org/FHIR/R4/clinicalreasoning-cds-on-fhir.html
7. HL7. *FHIR R4 specification.* https://hl7.org/fhir/R4/
8. Taber P, et al. New standards for clinical decision support: a survey of the state of
   implementation. *Yearbook of Medical Informatics* 2021.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8416232/
9. Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
   sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*
   2021;181(8):1065–1070. https://doi.org/10.1001/jamainternmed.2021.2626
10. U.S. Department of Health and Human Services. *HIPAA Security Rule.*
    https://www.hhs.gov/hipaa/for-professionals/security/index.html
11. European Commission. *General Data Protection Regulation (EU) 2016/679.*
    https://eur-lex.europa.eu/eli/reg/2016/679/oj
12. LOINC. https://loinc.org/ and RxNorm. https://www.nlm.nih.gov/research/umls/rxnorm/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_clinical-case-interviews.md`](06_clinical-case-interviews.md)
- Next topic: [`08_behavioural-interviews.md`](08_behavioural-interviews.md)
- Related topic: [`05_technical-interviews.md`](05_technical-interviews.md)
- Survey file: [`../../course/20_career_portfolio_interviews.md`](../../course/20_career_portfolio_interviews.md)
- Labs: [`../../labs/`](../../labs/)
