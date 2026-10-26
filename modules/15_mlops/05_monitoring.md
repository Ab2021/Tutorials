# Monitoring

> Module 15 — Deployment and Machine Learning Operations (MLOps)
> Topic 05 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01, 02
> Version 2026.3

## 1. Why this topic exists

Monitoring is the part of deployment that determines whether anyone finds out that the model
stopped working. It is also the part most often reduced to a single dashboard showing uptime and
CPU, which answers a question nobody asked: whether the process is alive.

A clinical prediction service can be perfectly alive and clinically broken. It can return 200 in
80 milliseconds while scoring every patient from a stale feature cache. It can serve every request
while its sensitivity has fallen twelve points. It can have zero errors while its alert volume has
doubled and the ward has stopped reading the alerts. None of these conditions produces an
exception, a failed health check, or a red panel on an infrastructure dashboard.

The topic exists because the signals that detect a broken service and the signals that detect a
broken model are different, are owned by different teams, and operate on different time scales. A
monitoring plan that instruments only the first is not a monitoring plan for a model. It is a
monitoring plan for a web server that happens to compute a probability.

### 1.1 The decisions this topic unblocks

| Decision | Who is blocked | Consequence of weak monitoring |
|---|---|---|
| When to intervene in a live service | On-call engineer | Intervention is triggered by a complaint, not a signal |
| Whether a model is still fit for use | Clinical owner, data science | The review happens annually, or after harm |
| Whether an alert is worth paging for | SRE, clinical operations | Alert fatigue or missed events |
| How to distinguish data, model, and workflow failures | Incident responder | The wrong team is paged and the wrong fix is applied |
| How much monitoring is enough | Finance, engineering leadership | Either blind spots or an unmaintainable telemetry bill |
| What evidence to retain for audit | Quality, regulatory | The record cannot reconstruct what happened |

### 1.2 The four layers

Monitoring for a deployed model has four layers, and each answers a different question.

1. **Service health**: is the process serving requests within its latency and error budget.
2. **Data health**: are the inputs present, fresh, in range, and in the expected schema.
3. **Model health**: is the model's output distribution, calibration, discrimination, and subgroup
   performance still acceptable.
4. **Outcome health**: are patients or operations better off, unchanged, or harmed.

The layers are ordered by detection latency, not by importance. Service health is detected in
seconds. Data health in minutes to hours. Model health in days to weeks, depending on label
latency. Outcome health in weeks to months, and often never in a usable form.

A monitoring plan that covers only the first layer has a detection gap measured in weeks. A plan
that tries to cover only the fourth has a detection gap measured in months and depends on data
that may not exist.

### 1.3 The cost of a false page

Every alert has a precision, whether or not the team measures it. Precision is the fraction of
alerts that correspond to a real, actionable condition. An alert with 80 percent precision costs
one unnecessary investigation for every four real events. An alert with 10 percent precision costs
nine.

The cost is not only engineer time. A false page at 02:00 teaches the on-call engineer that the
alert is noise. After a few repetitions, the engineer acknowledges without investigating, and the
next real event is missed. This is the mechanism by which a monitoring system fails: not by being
absent, but by being untrusted.

### 1.4 Why dashboards are not monitoring

A dashboard is a display. Monitoring is a system that detects, routes, and triggers a response. The
difference is the consequence. A dashboard with no alert and no owner is a screen that someone
looks at during a meeting. A monitoring system has a signal, a threshold, a route, an owner, and a
runbook.

Many teams build dashboards and call the work done. The test is simple: name the last time a
dashboard panel caused someone to take an action. If the answer is "never," the dashboard is
decorative.

## 2. Learning outcomes

By the end of this topic you can:

1. **Separate** service, data, model, and outcome monitoring, and state the detection latency of
   each.
2. **Define** service level indicators (SLIs) and service level objectives (SLOs) with windows and error budgets for a model service.
3. **Design** data-quality checks that catch schema, freshness, range, and completeness failures.
4. **Monitor** model performance when labels are delayed, using proxies with stated limits.
5. **Set** alert thresholds from a target precision and the cost of a false page.
6. **Route** alerts to owners with runbooks and escalation paths.
7. **Audit** an existing monitoring plan for blind spots and alert fatigue.

Outcomes 1 and 2 are assessed by Exercises 12.1 to 12.3; outcomes 3 and 4 by 12.4 and
12.5; outcomes 5 to 7 by 12.6 to 12.8. The calculation exercises are the ones that separate a
monitoring plan from a dashboard: a plan states a budget, a precision, and a detection delay.

## 3. The core mental model

Monitoring is a stack of sensors, each with a different time constant. Draw it as four horizontal
layers over the request path, with the detection latency increasing downward.

```text
 request ──▶ service ──▶ features ──▶ model ──▶ output ──▶ action ──▶ outcome
               │            │           │          │          │         │
               ▼            ▼           ▼          ▼          ▼         ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ L1 SERVICE   error rate, latency p50/p95/p99, saturation, restarts      │  seconds
 ├─────────────────────────────────────────────────────────────────────────┤
 │ L2 DATA      schema, freshness, missingness, ranges, duplicates        │  minutes
 ├─────────────────────────────────────────────────────────────────────────┤
 │ L3 MODEL     score distribution, calibration, discrimination, subgroups│  days–weeks
 ├─────────────────────────────────────────────────────────────────────────┤
 │ L4 OUTCOME   alert burden, override rate, time to treatment, harm      │  weeks–months
 └─────────────────────────────────────────────────────────────────────────┘
```

### 3.1 What each layer can and cannot see

| Layer | Detects | Blind to |
|---|---|---|
| Service | Process down, slow, erroring, saturated | A fast, correct-looking answer computed from stale data |
| Data | Missing, late, out-of-range, wrong schema | A correctly formed value that is clinically wrong |
| Model | Distribution shift, calibration change, performance loss on labels | Whether the action changes outcomes |
| Outcome | Benefit, harm, workflow effect | Which component caused it |

The blindness is structural. No amount of service telemetry can detect a model that is confidently
wrong; no amount of model telemetry can detect a workflow in which nobody reads the output.

### 3.2 Detection latency and the label problem

The most important design constraint is label latency: the delay between a prediction and the
outcome that would tell you whether it was right.

| Label latency | What is possible | What is not |
|---|---|---|
| Minutes | Near-real-time performance monitoring | Reliable estimates for rare outcomes |
| Hours | Daily performance review | Detection within a shift |
| Days | Weekly calibration and discrimination | Detection within an incident |
| Weeks to months | Periodic audit and recalibration | Any timely detection |
| Never | Proxy monitoring only | Direct performance monitoring |

When labels are delayed, the monitoring plan must say so and must use proxies. A proxy is a signal
that correlates with the outcome. It is not the outcome. The plan must state what the proxy can
detect and what it cannot.

Common proxies in clinical settings:

- Prediction distribution and score entropy.
- Alert volume per 1,000 patients.
- Clinician override rate and override reasons.
- Data completeness and feature freshness.
- Downstream actions such as orders, referrals, and escalations.
- Complaint and safety-event rates.

### 3.3 Leading and lagging indicators

A leading indicator changes before the outcome. A lagging indicator changes after it. Monitoring
needs both, because leading indicators give time to act and lagging indicators confirm that the
action mattered.

For a deterioration model, a leading indicator is the missingness rate for a key laboratory
feature. A lagging indicator is the sensitivity computed on an audit sample two weeks later. The
missingness rate is actionable now; the sensitivity is evidence that the action was needed.

### 3.4 The alert contract

Every alert should have a written contract with five fields:

```text
signal      what is measured, from which source, at what aggregation
threshold   the value and duration that trigger the alert
precision   the expected fraction of alerts that are real, with the measurement method
owner       the role that receives the alert and can act
action      the first three steps in the runbook, and the escalation path
```

An alert without a precision estimate is an experiment. An alert without an owner is a
notification. An alert without an action is a dashboard with a pager attached.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Monitoring | A system that detects, routes, and triggers response to a condition | A dashboard | None |
| Observability | The ability to infer internal state from external signals | A synonym for monitoring | None |
| Telemetry | Metrics, logs, and traces emitted by a system | Metrics only | Volume per second |
| SLI | A quantitative measure of service behaviour | A dashboard panel | Ratio or percentile |
| SLO | A target for an SLI over a window | An aspiration | Ratio over days |
| Error budget | The allowed failure implied by the SLO | A suggestion | Failed requests or minutes |
| Golden signals | Latency, traffic, errors, saturation | Any four metrics | Per signal |
| p99 | The value below which 99 percent of observations fall | The maximum | Milliseconds |
| Saturation | The degree to which a resource is fully utilized | CPU only | Ratio |
| Data freshness | The age of the data used at decision time | The time since ingestion | Seconds |
| Completeness | The fraction of expected records present | The row count | Ratio |
| Schema validation | Checking types, fields, and constraints | Checking a sample | Pass or fail |
| Drift | A change in inputs, labels, or relationships | Any change in a metric | population stability index (PSI), KS, or delta |
| PSI | A binned divergence between reference and current distributions | A universal threshold | Unitless |
| KS statistic | The maximum distance between two empirical distributions | A p-value only | Unitless |
| Calibration | Agreement between predicted probabilities and observed frequencies | Discrimination | Slope, intercept |
| Discrimination | The ability to rank positives above negatives | Accuracy | AUROC |
| Brier score | Mean squared error of probabilistic predictions | A calibration measure alone | Unitless, 0 to 1 |
| Subgroup performance | Performance within a defined population | Overall performance | Metric per group |
| Alert precision | The fraction of alerts that are real and actionable | The alert rate | Ratio |
| Alert recall | The fraction of real events that generate an alert | Precision | Ratio |
| Page | An alert that requires immediate human response | Any notification | Count per week |
| Ticket | An alert that requires action during business hours | A page | Count per week |
| Runbook | The procedure for responding to an alert | A wiki page | Document version |
| Label latency | The delay between prediction and outcome | The data refresh interval | Days |
| Proxy metric | A signal used in place of an unavailable outcome | The outcome | Unit depends on proxy |
| Sentinel metric | A single metric chosen to detect a broad failure | Any metric | Unit depends |
| Cardinality | The number of distinct label combinations in a metric | The number of metrics | Count |
| Sampling | Recording a subset of events to control cost | Dropping data silently | Ratio |
| Trace | A record of a request across components | A log line | Trace identifier |
| Span | A timed operation within a trace | A metric | Milliseconds |

Two distinctions carry the most weight.

**Monitoring is not observability, and observability is not monitoring.** Observability is the
property that lets you ask new questions of the telemetry. Monitoring is the set of questions you
decided to ask in advance and the responses you wired to them. A system can be observable and
unmonitored, or monitored and poorly observable.

**Calibration is not discrimination.** A model can rank patients correctly and assign probabilities
that are wrong by a factor of two. If the workflow uses the probability, calibration matters more
than ranking. Van Calster and colleagues describe calibration as the Achilles heel of predictive
analytics for exactly this reason.

## 5. The mechanics, step by step

The procedure builds a monitoring plan from failure modes, not from available metrics. Starting
from metrics produces dashboards full of things that are easy to measure and irrelevant to the
decision.

### Step 1 — Enumerate decisions and failure modes

**Input.** The decision frame from Topic 01, the architecture, and the incident history.

**Output.** A list of failure modes, each with a layer (service, data, model, outcome), a
consequence, and a detection question.

**Rationale.** Monitoring exists to detect failures, so the failure list is the requirement. A
generic list such as "the service is down" produces generic monitoring. A specific list such as
"the laboratory interface delivers creatinine after the scoring cutoff" produces a specific,
testable check.

The list should include failures that have happened, failures that nearly happened, and failures
that are plausible given the architecture. The incident history is the best source; the
architecture review is the second best.

Branch point: if a failure mode has no observable signal, it cannot be monitored directly. It must
be detected through a proxy, or accepted as a residual risk with a stated rationale.

### Step 2 — Define service SLIs and SLOs

**Input.** The workflow's tolerance and the request path.

**Output.** SLIs with a measurement point, an aggregation, a window, and an SLO with an error
budget.

**Rationale.** An SLO converts a vague expectation into a budget that can be spent and tracked. The
measurement point matters: server-side latency excludes network and render time, which may be the
largest part of the user's experience.

A workable starting set for a real-time service:

| SLI | Definition | Measurement point | SLO | Window |
|---|---|---|---|---|
| Availability | Fraction of requests returning a valid response | Client or integration engine | 99.5 percent | 28 days |
| Latency | Fraction of requests under 1,200 ms | Client | 99 percent | 28 days |
| Freshness | Fraction of predictions using features younger than 15 minutes | Service | 99 percent | 28 days |
| Correctness | Fraction of responses with a schema-valid body | Service | 99.9 percent | 28 days |

The freshness SLI is the one most often missing and the one that catches the most dangerous
failures. A service can meet its availability and latency SLOs while serving stale data.

Branch point: if the service has no client-side measurement, use a synthetic probe that exercises
the full path. State that the probe is a sample, not the population.

### Step 3 — Choose measurement points and instrumentation

**Input.** The SLIs and the architecture.

**Output.** A telemetry design: metrics, logs, traces, sampling, and retention.

**Rationale.** Each SLI needs a source. A latency SLI measured only at the server will miss the
integration and render time. An error SLI measured only at the load balancer will miss application
errors returned as 200 with an error body.

Traces are the best tool for latency decomposition because they show the time spent in each span.
Metrics are the best tool for aggregates and alerting because they are cheap to store and query.
Logs are the best tool for individual events and audit, and the most expensive to retain at scale.

Branch point: if full tracing is too expensive, sample traces at a rate that preserves the ability
to diagnose the p99, and use metrics for the aggregate SLO.

### Step 4 — Define data-quality checks

**Input.** The data contract and the feature definitions.

**Output.** Checks for schema, type, unit, range, missingness, duplicates, and referential
integrity, each with a severity and an owner.

**Rationale.** Data failures are the most common cause of silent model degradation. The checks must
run before the model sees the data, not after. A check that runs after inference and only logs a
warning does not prevent a bad prediction.

Breck and colleagues describe production data validation as a combination of schema constraints,
distribution checks, and training-serving skew detection. Their observation that distribution
distance metrics can produce false positives at large sample sizes is relevant: a check should be
tuned to the sample size and to the cost of a false alarm.

Branch point: if a check cannot run in the request path because of latency, run it in the feature
pipeline and gate the model on the result. A check that runs in a dashboard is a report, not a
control.

### Step 5 — Define freshness SLIs per feature

**Input.** The feature table with source, event time, and arrival time.

**Output.** A freshness metric per feature, with a threshold and an owner.

**Rationale.** Freshness is the difference between a correct model and a confident wrong one. A
model that uses a creatinine value from six hours ago may be clinically reasonable; one that uses a
value from six days ago is not.

Freshness must be measured at the moment of inference, not at the moment of ingestion. A feature
that was ingested recently may still be old if the source event was old.

Branch point: if a feature is legitimately delayed, its freshness threshold must reflect the
clinical meaning of the delay. A discharge summary that arrives two days later is normal; a vital
sign that arrives two hours later is not.

### Step 6 — Define model-output monitoring

**Input.** The prediction log and the model version.

**Output.** Distribution metrics for scores, decisions, and model versions.

**Rationale.** Output monitoring is available immediately and requires no labels. A shift in the
score distribution is not proof of degradation, but it is a leading indicator that something
changed. A sudden change in the share of `review` decisions changes the workload even if accuracy
is unchanged.

The output log must include the model version, the threshold, and the input feature vector or a
hash of it. Without the version, a distribution change cannot be attributed.

Branch point: if logging the full feature vector is a privacy risk, log a hash and aggregate
statistics. Do not log identifiers to make the analysis easier.

### Step 7 — Define performance monitoring given label latency

**Input.** The outcome definition, the label source, and the expected latency.

**Output.** A monitoring design that states what can be measured, when, and with what uncertainty.

**Rationale.** Performance monitoring is the only direct evidence that the model is still useful.
It is also the slowest. The design must match the cadence to the label latency and the event rate,
not to the calendar.

A useful structure:

| Label latency | Metric | Cadence | Minimum sample |
|---|---|---|---|
| 24 hours | Sensitivity, PPV at threshold | Weekly | 50 events |
| 7 days | Calibration slope, AUROC | Monthly | 100 events |
| 90 days | Subgroup performance, decision curve | Quarterly | 50 events per group |
| Never | None directly; use proxies | Continuous | Not applicable |

The minimum sample column is a statistical requirement, not a preference. A sensitivity estimate
from 12 events has a confidence interval wide enough to be uninformative.

Branch point: if the event rate is too low for the cadence, lengthen the window or aggregate
across sites. Do not report a metric whose uncertainty makes it uninterpretable.

### Step 8 — Define proxy metrics with stated limits

**Input.** The available signals and the label gap.

**Output.** A small set of proxies, each with a definition, a correlation evidence, and a stated
blind spot.

**Rationale.** Proxies are necessary when labels are delayed and dangerous when they are treated as
outcomes. The plan must state what each proxy can detect. Override rate can detect a change in
clinician agreement; it cannot detect a change in patient outcome. Alert volume can detect a shift
in the workload; it cannot distinguish a true increase in events from a false increase in flags.

The correlation between a proxy and the outcome should be established on historical data where
both are available. If it has not been established, the proxy is a hypothesis.

Branch point: if a proxy has never been validated against the outcome, label it as exploratory and
do not use it as a stop criterion for a release.

### Step 9 — Define subgroup monitoring

**Input.** The subgroup attributes, the pre-specified floors, and the sample sizes.

**Output.** A subgroup dashboard and alert thresholds, with confidence intervals and minimum
sample rules.

**Rationale.** An aggregate metric can hide a group whose performance has fallen. Subgroup
monitoring makes the gap visible. It also produces many comparisons, which increases the chance of
a false alarm. The design must pre-specify which groups are monitored and how multiple comparisons
are handled.

Branch point: if a group is too small to monitor statistically, aggregate over a longer window or
across sites, and state the limitation. Do not silently omit the group.

### Step 10 — Define outcome and workflow monitoring

**Input.** The workflow data, the action records, and the outcome definitions.

**Output.** Metrics for alert burden, override rate, time to action, and the clinical outcome.

**Rationale.** The outcome layer is the only one that can detect harm. It is also the layer most
dependent on data the model team does not own. The plan must name the data owner and the cadence.

Alert burden is a model metric as well as a workflow metric. At a fixed sensitivity, a fall in
specificity increases the number of alerts and the review cost, which changes the workflow even if
the model's ranking is unchanged.

Branch point: if the outcome data cannot be linked to predictions, the outcome layer cannot be
monitored. Record the gap and use the workflow layer as the best available proxy.

### Step 11 — Set alert thresholds from precision targets

**Input.** The failure modes, the cost of a false page, and the cost of a missed event.

**Output.** A threshold, a duration, and a target precision for each alert.

**Rationale.** A threshold chosen because it looks reasonable produces an unknown precision. A
threshold chosen from a target precision can be evaluated and tuned. The target should reflect the
cost ratio: if a false page costs one hour of engineer time and a missed event costs a patient
harm, the target precision should be low enough to prioritize recall.

The duration (`for:` in Prometheus terms) is part of the threshold. A 2 percent error rate for one
minute is noise; for ten minutes it is a condition.

Branch point: if the alert has never been observed in production, its precision is unknown. Run it
in shadow mode (log only) for a month, measure precision, then decide whether to page.

### Step 12 — Route alerts with owners, severities, and runbooks

**Input.** The alert list and the on-call structure.

**Output.** A routing table: alert, severity, owner, escalation, runbook link.

**Rationale.** An alert that reaches the wrong team wastes the most expensive resource in an
incident: time to the right responder. The routing table makes the path explicit. The runbook
makes the first three steps possible without a meeting.

Severity should be defined by consequence and urgency, not by metric magnitude. A 5 percent error
rate on a batch report may be a ticket; a 1 percent error rate on a sepsis alert may be a page.

Branch point: if no team owns an alert, it should not page. Either assign an owner or convert it to
a ticket routed to a named person.

### Step 13 — Design dashboards by audience

**Input.** The audiences: on-call, clinical operations, data science, quality, leadership.

**Output.** One dashboard per audience, each with a decision it supports.

**Rationale.** A single dashboard for all audiences is either too detailed for leadership or too
shallow for on-call. Each audience needs a different question answered. The on-call engineer needs
current state and recent changes. The clinical owner needs alert burden and outcome trends. The
quality reviewer needs evidence of monitoring and response.

A dashboard should state its question at the top. "Is the service healthy right now" and "is the
model still fit for use this quarter" are different dashboards.

Branch point: if an audience has no decision to make, it does not need a dashboard. It needs a
report at the cadence of its decision.

### Step 14 — Test, review, and retire alerts

**Input.** The alert set and its history.

**Output.** A monthly review of precision, response time, and relevance, with adjustments.

**Rationale.** Alerts decay. A threshold that was appropriate at launch may be noise after a
population change, or too loose after a workflow change. The review is the feedback loop for the
monitoring system itself.

The review should compute precision for every alert that fired, classify each firing as true or
false, and decide: keep, adjust, downgrade, or retire. An alert that has not fired in a year is
either unnecessary or broken.

Branch point: if an alert fires frequently and is always true, it may be a condition that should
be fixed rather than monitored. A chronic alert is a permanent tax on attention.

### 5.15 Alert severity and response

| Severity | Criteria | Response | Precision target | Example |
|---|---|---|---|---|
| Page | User-visible harm or imminent risk; no workaround | Immediate, 24/7 | At least 70 percent | Prediction application programming interface (API) returning 5xx during ward rounds |
| Urgent ticket | Degraded but usable; workaround exists | Same business day | At least 60 percent | Freshness SLO breached for one feature |
| Ticket | Trend or quality issue; no immediate harm | Within the sprint | At least 50 percent | Calibration slope drifting toward the floor |
| Log only | Exploratory or low signal | Reviewed weekly | Not applicable | A new proxy metric with no validated threshold |

The precision targets are starting points. They should be set from the cost ratio and revised with
observed data. A page with 70 percent precision means three in ten pages are unnecessary; a page
with 30 percent precision means the on-call engineer is mostly doing noise triage.

### 5.16 Edge cases the procedure must handle

**Schema drift.** A field changes type or name. The data check must fail closed and name the field.
If the service continues, it must return a clearly marked degraded response, not a default value.

**Late data.** A feature arrives after the scoring moment. The freshness SLI must capture it, and
the service must either wait, use a time-shifted value, or refuse to score.

**Partial rollout.** Two model versions are live. Every metric must be segmented by version, or the
aggregate will hide a regression in the new one.

**Model and feature skew.** Training and serving compute a feature differently. A parity check on a
golden input catches it at release; a distribution comparison between training and serving logs
catches drift after release.

**Clock skew.** Event time and ingestion time disagree. Freshness computed from the wrong clock is
wrong. Record both and monitor the offset.

**Quota exhaustion.** A metrics backend hits a cardinality or ingestion quota. Alerts stop firing
silently. Monitor the monitoring system: a heartbeat alert that fires when no data arrives.

**Dependency rotation.** A credential for a data source expires. The freshness check fails, but the
service may continue with cached data. The alert must distinguish "no new data" from "no data at
all."

**Regional outage.** A region fails and traffic shifts. The remaining region's latency and error
rate change. Dashboards must be segmented by region, and the SLO must be defined globally or
per-region explicitly.

## 6. Worked example

A 400-bed hospital runs the deterioration service from Topic 01. It scores 1,150 admissions per day,
or 32,200 per 28-day window, at a threshold of 0.18. The outcome is deterioration within 12 hours,
which is available from the electronic health record (EHR) with a median lag of 14 days.

The monitoring plan has four layers and three alerts. The arithmetic below shows how the plan is
sized and where it fails.

### 6.1 Service layer

| SLI | SLO | Window | Budget | Observed |
|---|---|---|---:|---:|
| Availability | 99.5 percent | 28 days | 161 failed requests | 122 failures |
| Latency under 900 ms | 99 percent | 28 days | 322 slow requests | 240 slow requests |
| Freshness under 15 min | 99 percent | 28 days | 322 stale requests | 410 stale requests |

The availability budget is 32,200 × 0.005 = 161. Observed failures are 122, so 39 remain, or 24
percent of the budget. The burn rate is 122 / 161 = 0.76, which is below 1 and does not page.

The freshness SLO is breached: 410 stale requests against a budget of 322. The burn rate is
410 / 322 = 1.27. This is the only service-layer signal that is telling the truth about a
developing problem, and it is the one most teams do not have.

### 6.2 Model layer

The model is evaluated monthly on the full labelled cohort. At 1,150 admissions per day and a
14-day label lag, a monthly review has about 28 days of predictions and roughly 1,224 events at
3.8 percent prevalence.

The standard error of a sensitivity estimate with 1,224 events at 0.60 is:

```text
SE = sqrt(0.60 × 0.40 / 1,224) = sqrt(0.000196) = 0.014
```

A 5-point drop is 3.6 standard errors, which is detectable. A 2-point drop is 1.4 standard errors,
which is not. The review is powered to detect the changes that matter, and not the ones that do
not.

The detection delay is the label lag plus the review cadence:

```text
best case  = 14 days label lag + 1 day review = 15 days
typical    = 14 days + 30 days to the next review = 44 days
```

If the label lag were 90 days, the typical detection delay would be 120 days. At 1,150 predictions
per day and 3.8 percent prevalence, that window contains 5,244 events. A 12-point sensitivity loss
would mean about 629 missed deteriorations before the review detected the change.

### 6.3 Alert economics

Three alerts are configured. Their precision is measured from the last quarter's firing history.

| Alert | Firings | True | Precision | Cost per false firing | False-alarm cost |
|---|---:|---:|---:|---:|---:|
| Error rate above 2 percent for 10 minutes | 6 | 4 | 0.67 | 1.5 h × 95 = 142.50 USD | 285 USD |
| Freshness breach above 1 percent of requests | 9 | 7 | 0.78 | 1.5 h × 95 = 142.50 USD | 285 USD |
| Calibration slope below 0.85 at monthly review | 4 | 2 | 0.50 | 12 h × 95 = 1,140 USD | 2,280 USD |
| **Total** | **19** | **13** | **0.68** | | **2,850 USD** |

The calibration alert is the most expensive false alarm because investigating it requires a data
scientist for a day and a half. Its precision of 50 percent is arguably too low for a monthly
review that consumes 12 hours per firing, but the cost of missing a real calibration failure is
higher. The team's decision is to keep the alert, add a second signal (a shift in the score
distribution) to the investigation trigger, and review precision after two more quarters.

The overall precision is 13 / 19 = 0.68. That is acceptable for a mix of pages and tickets. The
error-rate alert, which pages, has 67 percent precision, meaning one in three pages is
unnecessary. The team sets a target of 75 percent and adds a minimum-traffic condition to reduce
night-time false pages.

### 6.4 Diagnostic one: the service is green and the model is failing

In week 22, every service dashboard is green. Availability is 99.7 percent, latency p99 is 780 ms,
and the error rate is 0.2 percent. The freshness alert fires for the third time in two weeks.

The first diagnostic is a feature-age panel segmented by feature. Creatinine p95 age has risen from
8 minutes to 3 hours 20 minutes. The service is returning 200 responses in 80 ms, computed from
old data.

The second diagnostic is a replay of 200 recent admissions using only data available at the
scoring timestamp. Sensitivity on the replay is 0.51, against 0.60 on the complete-data replay. The
model is not degrading; the inputs are late.

The response is to fix the interface, backfill where possible, and add a hard gate that refuses to
score when a required feature is older than its threshold. The service-layer SLOs never detected
this. The freshness SLI did.

### 6.5 Diagnostic two: the calibration alert is noisy

The calibration alert fires four times in a quarter and is a true model problem twice. The two
false alarms were caused by a seasonal prevalence change: the calibration slope fell because the
case mix changed, not because the model's relationship changed.

The first diagnostic is a plot of calibration by month with the prevalence overlaid. The slope
tracks prevalence, which is a signature of a case-mix change rather than a coefficient change.

The second diagnostic is a recalibration on the recent cohort. If recalibration restores the slope
without changing discrimination, the model's ranking is intact and the issue is the intercept.

The response is to recalibrate rather than retrain, and to add prevalence to the review context so
that the next reviewer can distinguish the two causes. The alert remains, but the runbook now
requires the prevalence check before escalation.

### 6.6 Vary one assumption: label latency

Suppose the outcome data arrived with a 90-day lag instead of 14 days. The monthly review would
still run, but it would evaluate predictions from three months earlier. The detection delay would
be at least 120 days, and the 629 missed events computed in Section 6.2 would accumulate before
anyone knew.

The response changes in three ways. First, the review cadence is no longer the binding constraint;
the label lag is. Shortening the review to weekly does not help. Second, the plan must rely on
proxies: freshness, score distribution, alert volume, and override rate. Third, the plan must state
explicitly that a 12-point sensitivity loss could persist for four months, and the clinical owner
must accept that residual risk or fund a faster label source.

This is the central trade-off in monitoring. The most direct signal is the slowest. A plan that
does not state its detection delay is a plan that has not been thought through.

### 6.7 What the example establishes

The service layer was green while the model was degraded. The freshness SLI, not the availability
SLO, detected the failure. The calibration alert had 50 percent precision and consumed the largest
investigation budget, but it was still worth keeping because the cost of a missed calibration
failure was higher.

The arithmetic also shows why alert precision must be measured, not assumed. A team that had never
counted its firings would not know that one in three pages was unnecessary, and would not know
which alert to tune.

### 6.8 The plan as a single table

The monitoring plan fits on one page. Each row is a signal with an owner and a response.

| Signal | Layer | Threshold and duration | Severity | Owner | First action |
|---|---|---|---|---|---|
| 5xx rate | Service | Above 2 percent for 10 minutes | Page | Platform on-call | Check `/readyz`, rollback if new version |
| p99 latency | Service | Above 900 ms for 10 minutes | Page | Platform on-call | Check feature service and database |
| Feature age | Data | Above 15 minutes for 1 percent of requests | Urgent ticket | Data platform | Check interface lag, backfill |
| Missingness | Data | Above 5 percent for any required feature | Urgent ticket | Data platform | Check source and schema |
| Score distribution | Model | Mean shift above 0.05 from reference | Ticket | Data science | Compare feature distributions |
| Calibration slope | Model | Below 0.85 at monthly review | Ticket | Data science | Check prevalence, recalibrate |
| Sensitivity | Model | Below 0.55 on monthly audit | Urgent ticket | Clinical owner and data science | Investigate data and model, consider rollback |
| Alert volume | Outcome | Above 120 per day for three days | Ticket | Clinical operations | Review threshold and staffing |
| Override rate | Outcome | Above 40 percent for two weeks | Ticket | Clinical owner | Review reasons, usability |

The table is the plan. Everything else, including this topic, is explanation.

Two rows deserve comment. The score-distribution row has a low severity because a distribution
shift is not harm; it is a reason to look. The override-rate row has a clinical owner because a
high override rate is a workflow problem that the data science team cannot fix alone.

The table also makes gaps visible. If a failure mode from Step 1 has no row, it is either
undetectable or undetected. The team should be able to point to the row that would catch each
failure it claims to monitor, and to the runbook that follows the alert.

A plan with more than about a dozen signals is usually a plan that has not prioritized. The table
should fit on one page so that the on-call engineer can read it during an incident.

## 7. Code, query, or calculation

The script computes the quantities a monitoring review needs every month: alert precision and
recall, burn rates against two SLOs, detection delay, the number of events exposed during that
delay, and the cost of false alarms. It uses only the standard library.

```python
"""Alert quality, burn rate, and detection delay. Standard library only."""


def alert_quality(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def burn(observed, allowed):
    return observed / allowed


def false_alarms_allowed(tp, target_precision):
    """Maximum false positives that still meet a precision target."""
    return tp * (1 - target_precision) / target_precision


def main() -> None:
    tp, fp, fn = 13, 6, 1
    precision, recall = alert_quality(tp, fp, fn)
    print(f"precision            {precision:.3f}")
    print(f"recall               {recall:.3f}")
    print(f"false_alarms_allowed {false_alarms_allowed(tp, 0.75):.1f}")

    availability = burn(122, 161)
    freshness = burn(410, 322)
    print(f"availability_burn    {availability:.2f}")
    print(f"freshness_burn       {freshness:.2f}")

    label_lag_days = 14
    review_cadence_days = 30
    detection = label_lag_days + review_cadence_days
    print(f"detection_days       {detection}")

    predictions_per_day = 1_150
    prevalence = 0.038
    long_lag = 90
    window_days = long_lag + review_cadence_days
    events = predictions_per_day * window_days * prevalence
    missed = events * 0.12
    print(f"events_in_window     {events:.0f}")
    print(f"missed_at_12_points  {missed:.0f}")

    false_page_hours = 1.5
    hourly = 95
    calibration_hours = 12
    cost = 4 * false_page_hours * hourly + 2 * calibration_hours * hourly
    print(f"false_alarm_cost     {cost:.0f} USD")

    assert abs(precision - 0.684) < 0.001
    assert abs(recall - 0.929) < 0.001
    assert abs(availability - 0.758) < 0.001
    assert abs(freshness - 1.273) < 0.001
    assert abs(missed - 629) < 1
    assert abs(cost - 2850) < 1


if __name__ == "__main__":
    main()
```

Expected output, confirmed on Python 3.11:

```text
precision            0.684
recall               0.929
false_alarms_allowed 4.3
availability_burn    0.76
freshness_burn       1.27
detection_days       44
events_in_window     5244
missed_at_12_points  629
false_alarm_cost     2850 USD
```

Four readings matter.

First, precision and recall are both needed. A service that fires on everything has perfect recall
and useless precision. A service that fires only on the most obvious cases has high precision and
misses events. The pair describes the alert; either alone can be gamed.

Second, `false_alarms_allowed` translates a precision target into an operational limit. With 13
true alerts and a 75 percent target, at most 4.3 false alarms are allowed. The observed 6 means the
team needs to remove two false alarms per quarter, which is a concrete goal rather than a vague
aspiration.

Third, the burn comparison shows the only SLO that is actually in trouble. The availability burn is
0.76, which is healthy. The freshness burn is 1.27, which is not. A dashboard that showed only
availability would report "green."

Fourth, the detection delay is the product of two decisions: how long labels take and how often
the review runs. Reducing the review cadence from monthly to weekly would cut 23 days from the
delay, but only if the event count in a week is large enough to estimate sensitivity. At 1,150
admissions per day and 3.8 percent prevalence, a week contains about 306 events, which is enough
for a 5-point change but not a 2-point change.

The reference service in `labs/mlops/` exposes a Prometheus endpoint with prediction counts, HTTP
status counts, and latency sum and count. The accompanying `monitoring/prometheus-rules.yaml` has
three alerts: error rate above 2 percent for 10 minutes, p99 latency above 2 seconds for 10
minutes, and no predictions for 30 minutes. All three are service-layer alerts. None of them can
detect stale features, a calibration change, or a fall in sensitivity. That is the gap this topic
is about.

## 8. Data, standards, and regulatory context

Monitoring is where post-market obligations become engineering work. The instruments below
determine what must be monitored, what must be recorded, and what must be reported.

### 8.1 The instruments

| Instrument | Status | What it requires of monitoring | What it does not specify |
|---|---|---|---|
| FDA GMLP principle 10, 2021 | Non-binding guidance | Monitor deployed models for performance and manage retraining risks | Metrics, thresholds, or cadence |
| FDA AI-enabled device software functions guidance, draft January 2025 | Draft, not for implementation | A postmarket performance monitoring plan in the submission | The statistical design of that plan |
| FDA predetermined change control plan (PCCP) guidance, 2024, reissued 2025 | Final guidance implementing FD&C 515C | The monitoring and update protocol described in the PCCP | Internal dashboards or alerting tools |
| ISO 14971:2019 clause 10 | Binding when invoked | Production and post-production monitoring of risk control effectiveness | Which metrics to use |
| IEC 62304:2006/Amd 1:2015 clauses 6 and 9 | Binding when invoked | Maintenance process, problem resolution, and change control | Monitoring architecture |
| ISO/IEC 42001:2023 clause 9 | Voluntary, certifiable | Performance evaluation, internal audit, management review | Technical metrics |
| NIST AI RMF 1.0, MEASURE and MANAGE | Voluntary | Measurement of AI risks and management of responses | Specific thresholds |
| HIPAA Security Rule, 45 CFR 164.312(b) | Binding for covered entities | Audit controls that record and examine activity | Log format or retention period beyond the documentation rule |
| 21 CFR Part 11 | Binding where predicate rules apply | Secure, time-stamped audit trails that do not obscure prior records | Monitoring metrics |
| OpenTelemetry semantic conventions | Voluntary convention | Metric names, units, and attributes | Whether to monitor model behaviour |
| Prometheus alerting practices | Voluntary convention | Alert structure and expression patterns | Clinical thresholds |

### 8.2 What a post-market monitoring plan must contain

The FDA draft guidance on AI-enabled device software functions recommends a postmarket performance
monitoring plan as part of a marketing submission. The plan should describe what will be monitored,
how often, with what methods, and what actions follow a breach. The guidance is a draft, so it is
not a final statement of expectations, but it indicates the direction.

For the four-layer model, a plan that covers only service health would be incomplete. The plan
should cover data quality, model performance with delayed labels, subgroup performance, and
workflow effects. It should also state the label latency and the resulting detection delay, because
that is the plan's most important limitation.

### 8.3 What ISO 14971 clause 10 adds

Clause 10 requires monitoring of production and post-production information, including whether risk
control measures remain effective. For a model, the risk control may be a threshold that limits
alert volume or a human review step. Monitoring must show that the control is still working, not
only that the model is still accurate.

This is a subtle but important distinction. A model can remain accurate while a risk control fails,
for example if the review step is removed from the workflow or the alert is routed to a queue
nobody reads. The monitoring plan must include the control, not only the model.

### 8.4 What the audit rules require

HIPAA's audit control standard requires mechanisms that record and examine activity in systems
containing electronic protected health information. For a model service, the audit record should
allow reconstruction of a prediction: when it was made, for which patient or encounter, with which
model version, and what was returned. It should not record more identifiers than necessary.

Part 11 adds requirements where it applies: audit trails must be secure, computer-generated, time
stamped, and must not obscure prior values. A monitoring system that overwrites a prediction record
when a score is recomputed fails this requirement in substance.

### 8.5 What is convention

- The four golden signals (latency, traffic, errors, saturation). They are a useful starting point
  for service monitoring, not a complete model monitoring plan.
- A PSI threshold of 0.1 or 0.2. These are conventions with no universal statistical justification.
- A monthly model review. The cadence should follow label latency and event rate.
- Prometheus and Grafana. The tools are interchangeable; the signals and the responses are not.
- A particular alert duration. The duration should be set from the noise characteristics of the
  signal, not copied from another service.

> **Note:** A monitoring plan is not a substitute for clinical evidence. It detects change; it does
> not establish that the original model was effective. The two are separate obligations.

### 8.6 The monitoring record

The technical telemetry and the quality record are different datasets with different retention
rules. Telemetry is sampled, aggregated, and retained for weeks or months to control cost. The
quality record is complete, linked to the model version, and retained for the audit period.

A workable split is: metrics retained for 13 months for trend comparison; traces sampled at 5
percent and retained for 30 days; prediction events retained as an audit record with identifiers
removed and a link to the encounter; alert firings and dispositions retained for the audit period.

The split matters because an incident review often needs a record that the telemetry system has
already deleted. The quality record should contain enough to reconstruct the prediction and the
model version, even if it does not contain the full request.

## 9. Industry practice and case evidence

### 9.1 The four golden signals and SLO alerting

Google's Site Reliability Engineering practice describes four golden signals for user-facing
services: latency, traffic, errors, and saturation. The SRE Workbook extends this with alerting on
SLOs and multiwindow, multi-burn-rate alerts, which are designed to balance precision, recall,
detection time, and reset time.

The documented practice is about software services. The inference for clinical AI is that the
method transfers, but the fourth layer does not. A service can satisfy all four golden signals and
still produce clinically wrong output. The golden signals are necessary for the first layer and
silent on the other three.

### 9.2 The Epic Sepsis Model and alert burden

Wong and colleagues' external validation at Michigan Medicine found that the Epic Sepsis Model
fired alerts for 18 percent of hospitalizations at the alert threshold, with a positive predictive
value of 12 percent and sensitivity of 33 percent. The model had been embedded in the EHR and
scored every 15 minutes.

The documented finding is about model performance in one health system. The inference for
monitoring is that alert burden is a first-class model metric. At 18 percent of hospitalizations,
the alert volume alone changes the workflow, regardless of accuracy. A monitoring plan that tracks
only discrimination would not have surfaced the operational failure.

### 9.3 Data validation at production scale

Breck and colleagues describe a data validation system used across many teams at Google, with
schema checks, distribution checks, and training-serving skew detection. They note that
distribution distance metrics can produce false positives at large sample sizes, which is why they
choose L-infinity and L1 distances over chi-squared or KL divergence in their setting.

The documented lesson is that a data check has a statistical design, not just a threshold. The
inference for clinical monitoring is that a PSI threshold copied from credit scoring may fire
constantly on a large cohort or never on a small one. The threshold should be set from the sample
size and the cost of a false alarm.

### 9.4 Calibration as the Achilles heel

Van Calster and colleagues argue that calibration is under-assessed relative to discrimination and
can make a model misleading even when ranking is good. They describe four levels of calibration,
from calibration-in-the-large to strong calibration, and recommend against the Hosmer-Lemeshow test
as a summary.

The documented argument supports monitoring calibration, not only AUROC. The inference is that a
monthly review should report calibration slope and intercept alongside discrimination, and should
distinguish a prevalence-driven intercept shift from a relationship change.

### 9.5 Vendor defaults are not evidence

Evidently, an open-source monitoring library, documents default drift methods and thresholds: PSI
with a default threshold of 0.1, Wasserstein distance with 0.1, KS with a p-value threshold of
0.05, and automatic method selection based on reference sample size. These are documented defaults,
not validated clinical thresholds.

The inference is that a team adopting a library should treat its defaults as starting points and
calibrate them against local outcomes. A default that produces 40 percent false alarms in a
clinical service is not usable, regardless of its provenance.

### 9.6 What the evidence does not show

There is little published evidence on the precision of model-monitoring alerts in clinical
settings, the rate of false retraining triggered by drift, or the effect of monitoring on patient
outcomes. Vendor documentation describes capabilities. Case reports describe failures. The base
rates are local and unpublished.

A team should measure its own alert precision, detection delay, and response time, and should
treat those numbers as the evidence for its monitoring design. This is the same conclusion as
Topic 01, applied to the monitoring layer.

A related gap is the absence of published base rates for clinical model incidents. Teams know
that incidents happen; they rarely know how often, of what kind, or how long they last. Without
that base rate, a monitoring investment cannot be justified by expected loss, only by the
plausibility of harm. Local measurement is the substitute.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Green service, degraded model | Freshness or score distribution moves while error rate is flat | Service metrics are the only instrumented layer | Missed events for weeks | Add data and model layers; make freshness an SLI |
| Stale feature cache | Feature age p95 rises | Refresh job fails or a dependency is slow | Confident predictions from old data | Per-feature freshness metrics and a hard gate |
| Alert fatigue | Override or acknowledgement rate rises | Too many low-precision alerts | Real events are ignored | Measure precision; tune or retire alerts |
| Unowned alert | No one can name the responder | Ownership was never assigned | Detection without response | Routing table with a named role |
| Monitoring the monitor gap | A dashboard panel is empty | Metrics pipeline or quota failed | Silent blindness | Heartbeat alert on telemetry absence |
| Cardinality explosion | Metrics cost rises sharply | Per-patient or per-request labels | Budget overrun or dropped series | Aggregate labels; sample high-cardinality dimensions |
| Sampling bias | Sampled and unsampled aggregates diverge | Sampling by status or site | Wrong conclusions about the population | Stratified sampling and periodic reconciliation |
| Proxy treated as outcome | A proxy improves while outcomes worsen | Convenience and label lag | False confidence | State proxy limits; audit against outcomes |
| Label lag ignored | Review cadence shorter than label availability | Calendar-driven review | The review evaluates stale predictions | Match cadence to label latency |
| Aggregate hides a subgroup | Overall metric stable, subgroup gap widens | No stratified monitoring | Equity harm | Pre-specified subgroup panels and floors |
| Copied threshold | Alert fires constantly or never | Threshold from another service or blog | Noise or blindness | Calibrate from local data and cost ratio |
| Dashboard without a decision | No one remembers acting on it | Metrics chosen for availability | Decorative monitoring | Tie each panel to a decision and an owner |
| Metric definition drift | A metric changes without a code change | Dashboard query edited or source changed | Trend comparisons invalid | Version metric definitions |
| Retention too short | Incident review cannot reconstruct the window | Log retention set by cost alone | No evidence for root cause | Retention aligned with audit period |
| PHI in logs | Log volume grows after debugging | Request bodies logged | Privacy breach | Log schema review and redaction tests |

Red flags: a monitoring plan that fits on a slide but has no owner; an alert whose runbook says
"investigate"; a dashboard with no threshold; a model review that reports only AUROC; a metrics
pipeline with no heartbeat.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Service metrics reflect the user experience | SLOs are computed from them | Caching, retries, integration delays | Synthetic end-to-end probe |
| Labels arrive within the review cadence | Performance monitoring depends on it | Claims lag, outpatient follow-up | Measure label latency monthly |
| The event rate supports the cadence | Estimates need enough events | Rare outcomes, small sites | Compute confidence intervals before reporting |
| Proxies correlate with outcomes | They substitute for labels | Workflow changes, measurement changes | Validate on historical data with both |
| Subgroup attributes are complete and accurate | Equity monitoring depends on them | Missing or misclassified attributes | Compare attribute distributions to known totals |
| Alert precision is measurable | Tuning requires feedback | No record of firings or outcomes | Log every firing and its disposition |
| The telemetry pipeline is reliable | Monitoring depends on it | Quota, outage, misconfiguration | Heartbeat and synthetic canary |
| Thresholds are locally calibrated | Defaults do not transfer | Copying from another service | Measure false-positive rate after a month |
| Metrics are stable over time | Trends are comparable | Query or source changes | Version definitions and annotate changes |
| Retention matches the audit period | Evidence must be retrievable | Cost-driven deletion | Retrieve a record from 12 months ago |

## 12. Exercises

### Exercise 12.1 — Inventory the four layers

For a model service you know, list one signal at each of the four layers, its measurement point,
its detection latency, and its owner. Identify the layer with the weakest coverage.

**Deliverable.** A four-row table plus a one-paragraph gap statement.

**Acceptance criteria.** Signals are distinct across layers; the detection latency is stated in
units; the owner is a role; the gap is specific.

**Time.** 30 minutes.

### Exercise 12.2 — Compute alert quality

An alert fired 24 times in a quarter. Fifteen firings were true, and 16 real events occurred.
Compute precision, recall, the number of false positives and false negatives, and the maximum
number of false positives allowed at a precision target of 0.80.

**Deliverable.** Five numbers with the arithmetic.

**Acceptance criteria.** Precision and recall computed from the confusion matrix; the allowed
false positives derived from the target; the interpretation distinguishes noise from missed
events.

**Time.** 25 minutes.

### Exercise 12.3 — Design SLOs

A service receives 8,000 requests per day. Define an availability SLO of 99.8 percent over 30 days
and a latency SLO of 99 percent under 600 ms. Compute the error budgets in requests and minutes.

**Deliverable.** Two SLOs with windows and budgets, and the arithmetic.

**Acceptance criteria.** Budgets computed from the window; units stated; the latency budget is in
slow requests, not minutes, unless the conversion is justified.

**Time.** 25 minutes.

### Exercise 12.4 — Design data-quality checks

For a model using age, systolic blood pressure, creatinine, and prior admissions, specify checks
for schema, type, unit, range, missingness, duplicates, and freshness. For each, state the
severity and the owner.

**Deliverable.** A seven-row table with check, threshold, severity, and owner.

**Acceptance criteria.** Each check is testable in the pipeline; severe checks fail closed; the
freshness check uses event time, not ingestion time.

**Time.** 40 minutes.

### Exercise 12.5 — Design proxy monitoring

For a model whose outcome labels arrive after 90 days, design a proxy monitoring plan. For each
proxy, state what it can detect, what it cannot, and the evidence that it correlates with the
outcome.

**Deliverable.** A four-row table plus a statement of the residual detection delay.

**Acceptance criteria.** Proxies are not presented as outcomes; the correlation evidence is stated
or explicitly missing; the plan states what would trigger an investigation.

**Time.** 45 minutes.

### Exercise 12.6 — Critique a dashboard

A dashboard shows uptime, CPU, memory, request count, and average latency, refreshed every minute.
No panel has a threshold. Write a critique and propose three changes.

**Deliverable.** A 350-word critique with a revised panel list.

**Acceptance criteria.** Identifies the missing layers; notes that averages hide tail latency;
each proposed change names a decision and an owner.

**Time.** 30 minutes.

### Exercise 12.7 — Compute detection delay and exposure

A model scores 900 patients per day. Outcome labels arrive after 21 days, and the review runs
monthly. Compute the typical detection delay and the number of events in that window at 4.2
percent prevalence. If sensitivity falls by 10 points, how many events are missed before detection?

**Deliverable.** Three numbers with the arithmetic.

**Acceptance criteria.** Delay includes both lag and cadence; events computed from the full window;
missed events computed as a fraction of events.

**Time.** 25 minutes.

### Exercise 12.8 — Design alert routing

For five alerts (error rate, latency, freshness, calibration, alert volume), assign severity,
owner, escalation path, and the first three runbook steps. State the precision target for each.

**Deliverable.** A five-row routing table with runbook excerpts.

**Acceptance criteria.** Severity follows consequence, not metric magnitude; every alert has one
owner; the runbook steps are actionable without a meeting; precision targets are numeric.

**Time.** 45 minutes.

## 13. Self-check answers

### 13.1 Exercise 12.1

A typical answer: service layer, p99 latency at the load balancer, seconds, platform on-call; data
layer, feature age at inference, minutes, data platform; model layer, calibration slope at monthly
review, weeks, data science; outcome layer, override rate, days, clinical owner. The weakest layer
is usually the outcome layer, because the data is owned by another team and the cadence is long.

### 13.2 Exercise 12.2

```text
precision = 15 / 24 = 0.625
recall    = 15 / 16 = 0.9375
false_positives = 24 − 15 = 9
false_negatives = 16 − 15 = 1
allowed_fp_at_0.80 = 15 × (1 − 0.80) / 0.80 = 3.75
```

The alert is sensitive and noisy. It catches almost every event but produces nine false alarms per
quarter. To reach 80 percent precision, the team must remove about five false alarms, either by
raising the threshold, adding a confirming condition, or shortening the evaluation window.

### 13.3 Exercise 12.3

```text
requests_30d = 8,000 × 30 = 240,000
availability_budget = 240,000 × 0.002 = 480 failed requests
budget_minutes = 30 × 24 × 60 × 0.002 = 86.4 minutes
latency_budget = 240,000 × 0.01 = 2,400 slow requests
```

The latency budget is in slow requests because a slow request is not an outage. Converting it to
minutes requires an assumption about how many slow requests constitute an outage, which is a
different SLO.

### 13.4 Exercise 12.4

A strong answer includes: schema check for the four fields, fail closed; type check for numeric,
fail closed; unit check for mg/dL versus µmol/L on creatinine, fail closed; range check using
clinically plausible bounds, fail closed for impossible values and warn for implausible ones;
missingness check on each required field with a 5 percent threshold, fail closed; duplicate check
on the encounter identifier, fail closed; freshness check using event time with a 15-minute
threshold, fail closed.

### 13.5 Exercise 12.5

A strong plan uses four proxies: feature freshness, score distribution, alert volume, and override
rate. Freshness detects data-delivery failures; it cannot detect a change in the relationship. Score
distribution detects a shift in the model's output; it cannot distinguish a population change from
a model change. Alert volume detects workload changes; it cannot distinguish true from false
positives. Override rate detects clinician disagreement; it is confounded by staffing and usability.

The residual detection delay is at least 90 days plus the review cadence, and the plan should state
that a performance loss could persist for a quarter.

### 13.6 Exercise 12.6

The critique should note that uptime, CPU, memory, and request count are all service-layer signals,
and average latency hides the tail that users experience. The three changes: replace average
latency with p95 and p99 against an SLO; add a feature-freshness panel with a threshold and an
owner; add a model-output panel segmented by version. Each panel should state the decision it
supports.

### 13.7 Exercise 12.7

```text
detection_delay = 21 + 30 = 51 days
events = 900 × 51 × 0.042 = 1,927.8
missed = 1,927.8 × 0.10 = 192.8
```

About 1,928 events occur in the window, and a 10-point sensitivity loss means about 193 are missed
before the review detects the change. The number is an upper bound if the loss begins at the start
of the window and a lower bound if it begins later.

### 13.8 Rubric for the design exercises

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Layer coverage | 25% | Service only | Service and data | All four layers with detection latency |
| Thresholds | 25% | Copied defaults | Local calibration | Precision targets tied to cost |
| Ownership | 20% | Team name | Named role | Escalation and backup |
| Response | 20% | "Investigate" | Three concrete steps | Tested runbook with a rollback path |
| Honesty | 10% | Claims full coverage | States limits | Quantifies the residual detection gap |

## 14. Interview questions

**1. What is the difference between service monitoring and model monitoring?**
Strong answers separate availability and latency from calibration, discrimination, and drift, and
give an example where one is healthy and the other is not. Weak answers treat uptime as model
health.

**2. How do you monitor a model without labels?**
Strong answers use proxies with stated limits and a scheduled audit sample. Weak answers claim
drift detection is a substitute for outcomes.

**3. What makes an alert good?**
Strong answers name precision, recall, ownership, actionability, and cost. Weak answers say it
fires when something is wrong.

**4. How do you set an alert threshold?**
Strong answers start from the cost of a false page and a missed event, measure precision, and tune.
Weak answers copy a threshold from a vendor.

**5. What is an error budget and how does it change alerting?**
Strong answers define it as the complement of the SLO over a window and use burn rates. Weak
answers describe it as a dashboard number.

**6. How do you detect stale features?**
Strong answers measure feature age at inference, not ingestion, and gate scoring on a threshold.
Weak answers check the data pipeline's success status.

**7. Why is calibration important if AUROC is good?**
Strong answers explain that ranking and probability are different, and that workflows using the
probability depend on calibration. Weak answers say AUROC is sufficient.

**8. What is the detection delay of your monitoring plan?**
Strong answers compute label latency plus review cadence and state the exposure. Weak answers say
"continuous."

**9. How do you avoid alert fatigue?**
Strong answers measure precision, retire or downgrade noisy alerts, and use severity by
consequence. Weak answers add more alerts.

**10. How do you monitor the monitoring system?**
Strong answers describe heartbeats, synthetic canaries, and checks on telemetry freshness. Weak
answers assume the monitoring stack is reliable.

## 15. Key terms

- **Alert precision** — the fraction of alerts that are real and actionable.
- **Alert recall** — the fraction of real events that generate an alert.
- **Burn rate** — observed error rate divided by the SLO's allowed error rate.
- **Calibration** — agreement between predicted probabilities and observed frequencies.
- **Cardinality** — the number of distinct label combinations in a metric.
- **Completeness** — the fraction of expected records present.
- **Data freshness** — the age of the data used at decision time.
- **Detection delay** — the time from a failure's onset to its detection.
- **Discrimination** — the ability to rank positives above negatives.
- **Error budget** — the allowed failure implied by an SLO over a window.
- **Golden signals** — latency, traffic, errors, and saturation.
- **Label latency** — the delay between a prediction and its outcome.
- **Monitoring** — a system that detects, routes, and triggers a response.
- **Observability** — the ability to infer internal state from external signals.
- **Page** — an alert requiring immediate human response.
- **p99** — the value below which 99 percent of observations fall.
- **Proxy metric** — a signal used in place of an unavailable outcome.
- **PSI** — a binned divergence between reference and current distributions.
- **Rolling window** — a moving time period used for an SLI.
- **Runbook** — the procedure for responding to an alert.
- **Sampling** — recording a subset of events to control cost.
- **Saturation** — the degree to which a resource is fully utilized.
- **Sentinel metric** — a single metric chosen to detect a broad failure.
- **SLI** — a quantitative measure of service behaviour.
- **SLO** — a target for an SLI over a window.
- **Span** — a timed operation within a trace.
- **Subgroup performance** — performance within a defined population.
- **Telemetry** — metrics, logs, and traces emitted by a system.
- **Ticket** — an alert requiring action during business hours.
- **Trace** — a record of a request across components.

## 16. Sources

All URLs accessed 26 October 2026.

- Google. *Site Reliability Engineering*, Chapter 6, Monitoring Distributed Systems. https://sre.google/sre-book/monitoring-distributed-systems/
- Google. *The Site Reliability Workbook*, Chapter 5, Alerting on SLOs. https://sre.google/workbook/alerting-on-slos/
- Prometheus. *Alerting practices*. https://prometheus.io/docs/practices/alerting/
- Prometheus. *Histograms and summaries*. https://prometheus.io/docs/practices/histograms/
- OpenTelemetry. *Semantic conventions for HTTP metrics*. https://opentelemetry.io/docs/specs/semconv/http/http-metrics/
- Breck E, Polyzotis N, Roy S, Whang SE, Zinkevich M. Data validation for machine learning. *Proceedings of MLSys 2019*. https://proceedings.mlsys.org/paper_files/paper/2019/file/928f1160e52192e3e0017fb63ab65391-Paper.pdf
- Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Medicine*. 2019;17:230. https://doi.org/10.1186/s12916-019-1466-7
- Van Calster B, Nieboer D, Vergouwe Y, et al. A calibration hierarchy for risk models was defined. *Journal of Clinical Epidemiology*. 2016;74:167-176. https://doi.org/10.1016/j.jclinepi.2015.12.005
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*. 2021;181(8):1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
- Evidently AI. *Data drift documentation*. https://docs.evidentlyai.com/metrics/explainer_drift
- US Food and Drug Administration, Health Canada, and UK MHRA. *Good Machine Learning Practice for Medical Device Development: Guiding Principles*, 2021. https://www.fda.gov/media/153486/download
- US Food and Drug Administration. *Artificial Intelligence-Enabled Device Software Functions: Lifecycle Management and Marketing Submission Recommendations*, draft guidance, January 2025. https://www.fda.gov/media/184856/download
- ISO 14971:2019. *Medical devices — Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
- IEC 62304:2006/AMD1:2015. *Medical device software — Software life cycle processes*. https://webstore.iec.ch/en/publication/22794
- ISO/IEC 42001:2023. *Information technology — Artificial intelligence — Management system*. https://www.iso.org/standard/42001
- NIST. *AI Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, 2023. https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
- 45 CFR 164.312, Technical safeguards. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312
- 21 CFR 11.10, Controls for closed systems. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11/subpart-B/section-11.10

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/15_deployment_mlops.md`](../../course/15_deployment_mlops.md)
- Previous topic: [`04_ci-cd-ct.md`](04_ci-cd-ct.md)
- Next topic: [`06_drift.md`](06_drift.md)
- Related topic: [`07_incident-response.md`](07_incident-response.md)
- Labs overview: [`../../labs/`](../../labs/)
- Reference service: [`../../labs/mlops/`](../../labs/mlops/)
- Prometheus rules: [`../../labs/mlops/monitoring/prometheus-rules.yaml`](../../labs/mlops/monitoring/prometheus-rules.yaml)
- Metrics endpoint: [`../../labs/mlops/app/main.py`](../../labs/mlops/app/main.py)
- Runbook: [`../../labs/mlops/RUNBOOK.md`](../../labs/mlops/RUNBOOK.md)
