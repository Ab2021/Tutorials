# Observability

> Module 26 — Cloud, Kubernetes, and Streaming Production Labs
> Topic 06 of 09
> Estimated study time: 3.5 hours
> Prerequisites: 03
> Version 2026.3

## 1. Why this topic exists

A production service is observed through three kinds of signal: metrics, logs, and traces. Metrics
tell you that something is wrong and roughly where. Traces tell you which call in a request path is
slow. Logs tell you what the process thought at the moment it failed. A team that has only logs is
debugging with a flashlight and no map. A team that has only dashboards can see the problem but not
the cause.

The biomedical constraint makes observability harder than in most domains. You cannot log the
request body, because it may contain protected health information (PHI). You cannot attach a patient
identifier as a metric label, because it creates unbounded cardinality and because it is an
identifier that should not be in a metrics store. You must still be able to answer an auditor's
question about who accessed what and when, which requires an audit log that is separate from the
debug log and retained differently.

The reference stack provides almost no observability. `labs/cloud/k8s/03-deployment.yaml` exposes
`/healthz` and `/readyz`, and nothing emits metrics or traces. The network policy in
`labs/cloud/k8s/08-networkpolicy.yaml` allows ingress from a `monitoring` namespace on port 8000,
which anticipates a Prometheus scrape but does not configure one. This topic fills that gap
conceptually and shows what a production configuration looks like.

### 1.1 The questions observability must answer

| Question | Signal | Typical failure without it |
|---|---|---|
| Is the service meeting its objective | SLO dashboard | Debate instead of evidence |
| Why is this request slow | Trace | Guesswork across five services |
| Which dependency is failing | Error rate by dependency | Blame shifting |
| Did the last deploy cause this | Change marker on a dashboard | Slow rollback decision |
| Who accessed this record | Audit log | Cannot answer the regulator |
| Is the queue backing up | Consumer lag | Discovery at the SLA breach |
| Is the node about to evict pods | Saturation metric | Surprise eviction |

Each row maps to a section below.

### 1.2 Why "add more dashboards" does not work

A dashboard is a view, not a control. The failure mode of observability in most organisations is not
a lack of charts but a lack of agreement about which numbers matter, what threshold constitutes a
problem, and who acts when it is crossed. A service with forty dashboards and no SLO has more data
and no more decision-making capacity than a service with none.

The discipline is to start from the user-visible outcome, define one indicator for it, set an
objective, and only then choose the instruments.

### 1.3 The cost dimension

Observability is not free. Metrics are billed by series and retention, logs by ingested volume and
retention, traces by spans and retention. A cardinality mistake can multiply a metrics bill by fifty
without changing a single dashboard. Section 6 quantifies it.

### 1.4 One definition, written properly

The difference between a useful observability document and a wish is a definition that a second
engineer can implement without asking a question. Compare:

- Weak: "The API should be highly available and fast."
- Strong: "99.9 percent of FHIR read requests measured at the ingress return a 2xx within 500 ms,
  over a rolling 30-day window; the error budget is 43.2 minutes and page-level alerts fire at a
  burn rate above 14.4 sustained for one hour."

The strong version names the indicator, the threshold, the measurement point, the window, the
budget, and the alert. It can be implemented, tested, and disputed with evidence. Every section
below builds toward writing definitions of that kind.

> **Note:** The lab stack is a teaching scaffold. It does not ship a metrics pipeline, a log
> pipeline, or a trace collector, and it is not a compliant deployment.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** a service level indicator, objective, and error budget for a clinical API, with a
   measurement point and a window.
2. **Compute** the error budget in minutes and in failed requests for a stated request rate.
3. **Apply** multi-window burn-rate alerting and explain why a single-threshold alert is either
   noisy or slow.
4. **Choose** between a counter, gauge, histogram, and summary for a given measurement, and explain
   why summary quantiles cannot be aggregated.
5. **Instrument** an HTTP service with OpenTelemetry semantic conventions and state the metric name,
   unit, and attributes.
6. **Design** a logging policy that supports debugging and audit without recording PHI.
7. **Estimate** the cardinality and cost consequence of adding a label.
8. **Diagnose** an alert that fires too often and a dashboard that reports a misleading percentile.

Outcomes 1 to 3 are assessed by Exercises 12.1 to 12.3. Outcomes 4 and 5 by 12.4 and 12.5.
Outcomes 6 and 7 by 12.6 and 12.7. Outcome 8 by 12.8.

## 3. The core mental model

Observability has four layers, and each layer answers a different question. Confusing them is the
most common design error.

```text
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. Instrumentation: what the process emits                  │
   │    counters, gauges, histograms, spans, structured logs     │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 2. Pipeline: collection, aggregation, storage               │
   │    scrape or push, relabel, sample, retain                  │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 3. Indicators: SLI queries over a window                    │
   │    success ratio, latency percentile, lag, freshness        │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 4. Decisions: SLOs, error budgets, alerts, runbooks         │
   └─────────────────────────────────────────────────────────────┘
```

A team can have excellent instrumentation and no decisions, or a decision with no reliable
instrumentation. Both fail. The layers must be designed together.

### 3.1 The four golden signals

The four signals from Google's Site Reliability Engineering practice are latency, traffic, errors,
and saturation. They are a useful checklist because each maps to a different failure:

| Signal | Question | Example metric | Failure it detects |
|---|---|---|---|
| Latency | How long does a request take | `http_server_request_duration_seconds` histogram | Slow dependency, CPU throttling |
| Traffic | How much demand is there | `http_server_requests_total` counter | Capacity mismatch, attack |
| Errors | How many requests fail | Counter partitioned by status code | Dependency outage, bad deploy |
| Saturation | How full is the constrained resource | CPU throttling ratio, queue depth | Imminent degradation |

Saturation is the leading indicator; latency and errors are the lagging ones. A dashboard that shows
only latency and errors tells you about a problem after users experience it.

### 3.2 Percentiles are not averages

An average latency of 200 ms can hide a p99 of 8 seconds affecting one percent of requests. In a
clinical workflow, one percent can be every request from a particular ward during a shift change.
Report percentiles, and report them from histograms so they can be aggregated across instances.

### 3.3 The error budget is the bridge between reliability and delivery

An SLO of 99.9 percent over 30 days permits 43.2 minutes of unavailability. That permission is the
error budget. It converts an abstract argument about reliability into a quantity that can be spent,
tracked, and exhausted. When the budget is gone, releases stop and reliability work starts. Without
a budget, every outage becomes a negotiation.

### 3.4 Cost is a first-class constraint

Every label on a metric multiplies the number of time series. Every log line is ingested, indexed,
and stored. Every span is a record. An observability design that ignores cost will be cut during the
first budget review, usually at the moment it is most needed.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Observability | Ability to infer internal state from external outputs | Having dashboards | Property |
| Metric | Numeric measurement over time | A log line with a number | Name, labels, value |
| Counter | Monotonically increasing metric | A gauge that goes up and down | Cumulative total |
| Gauge | Metric that can rise and fall | A counter | Current value |
| Histogram | Bucketed distribution of observations | A summary | Buckets and counts |
| Summary | Client-side quantile estimate | A histogram | Quantile and value |
| Label | Key-value dimension on a metric | A free-text field | String |
| Cardinality | Number of distinct label combinations | Number of metrics | Count |
| Exemplar | Reference from a metric to a trace | A log line | Trace ID |
| Span | Timed operation within a trace | A log entry | Span ID |
| Trace | Directed graph of spans for one request | A log stream | Trace ID |
| Sampling | Keeping a fraction of traces | Dropping logs | Ratio |
| SLI | Service level indicator | An SLO | Ratio or duration |
| SLO | Service level objective | A guarantee | Percentage over a window |
| SLA | Service level agreement, a contract with consequences | An SLO | Contract |
| Error budget | Allowed violation within a window | Downtime allowance | Minutes or requests |
| Burn rate | Speed of budget consumption relative to the SLO | Error rate | Multiple |
| Window | Period over which an SLI is computed | Retention | Duration |
| Alert | Notification that requires action | Any threshold breach | Rule |
| Precision | Fraction of alerts that are real | Recall | Ratio |
| Recall | Fraction of real incidents that alert | Precision | Ratio |
| Runbook | Documented response to an alert | A wiki page | Document |
| Golden signals | Latency, traffic, errors, saturation | Any four metrics | Framework |
| RED method | Rate, errors, duration | A vendor product | Framework |
| USE method | Utilization, saturation, errors | A dashboard tool | Framework |
| Audit log | Record of access and actions for accountability | Debug log | Event record |
| PHI | Protected health information | Any health data | Regulatory category |
| Cardinality explosion | Unbounded growth of series from labels | High traffic | Failure mode |
| Recording rule | Precomputed query stored as a metric | A dashboard query | Rule |
| Scrape interval | How often metrics are collected | Retention | Seconds |
| Retention | How long signals are kept | Sampling | Duration |

Three distinctions carry the weight.

**Metric versus log.** A metric is cheap to store and expensive to make specific. A log is specific
and expensive to store. Use metrics for rates and distributions, logs for discrete events with
context.

**Precision versus recall.** An alert that fires for everything has perfect recall and terrible
precision. An alert that fires only for certain outages has the reverse. The SLO determines which
error is more expensive.

**Monitoring versus observability.** Monitoring answers questions you anticipated. Observability
lets you answer questions you did not. A production system needs both, and the budget usually funds
only the first.

## 5. The mechanics, step by step

The procedure designs an observability stack from the user-visible outcome backwards. It assumes
Kubernetes for compute and Prometheus-compatible metrics, but the reasoning transfers.

### Step 1 — Name the user-visible outcome

**Input.** The service and its consumers.

**Output.** One sentence describing what a user experiences when the service works.

**Rationale.** "The risk score is available in the ward dashboard within five seconds of the
observation" is an outcome. "CPU is below 70 percent" is not. Indicators derived from the outcome
survive refactoring; indicators derived from internals do not.

**Branch point.** If the service has several distinct consumers, it needs several outcomes and
possibly several SLOs. Do not average a batch consumer and an interactive clinician into one number.

### Step 2 — Define the indicator and the measurement point

**Input.** The outcome and the request path.

**Output.** An SLI expression with explicit numerator, denominator, and measurement location.

**Rationale.** "Success ratio of HTTP requests measured at the ingress, over five minutes" is a
definition. "Availability" is not. The measurement point determines what is included: an ingress
measurement includes the load balancer, a pod measurement does not.

**Decision table.** More than two outcomes, so the choice is tabulated.

| SLI type | Numerator | Denominator | Good for | Blind to |
|---|---|---|---|---|
| Availability | Non-5xx responses | All responses | APIs | Slow responses |
| Latency | Requests under threshold | All responses | Interactive APIs | Errors that are fast |
| Throughput | Processed events | Expected events | Batch and streaming | Latency |
| Freshness | Events within max age | All events | Data pipelines | Correctness |
| Correctness | Valid records | All records | Validation layers | Downstream effects |

**Edge case.** A latency SLI that counts only successful requests hides a service that fails fast.
Pair latency with availability, or count errors as failures in the latency indicator.

**Where to measure, concretely.** For a request through an ingress, there are at least four
candidate measurement points: the load balancer, the ingress controller, the service pod, and the
downstream dependency. Each includes a different part of the path:

| Point | Includes | Excludes | Best for |
|---|---|---|---|
| Load balancer | Client to edge, TLS termination | Ingress routing, application | External availability |
| Ingress controller | Routing, upstream timeout | Application processing | Routing errors |
| Service pod | Application processing | Client network, ingress | Application latency |
| Dependency client | Downstream call | Queueing before the call | Blame isolation |

Pick one as the SLO measurement point and use the others for diagnosis. A dashboard that mixes them
will show four different p99 values for the same request.

### Step 3 — Set the objective and the window

**Input.** The indicator and the consumer's tolerance.

**Output.** A percentage and a rolling window.

**Rationale.** A 30-day rolling window is the common choice for a monthly operational review. A
7-day window reacts faster and is noisier. A calendar month aligns with reporting but resets
abruptly.

Compute the budget:

```text
budget_minutes = window_days × 24 × 60 × (1 − SLO)
```

For 99.9 percent over 30 days, that is 43.2 minutes. For 99.95 percent, 21.6 minutes.

**Branch point.** If the service is safety-critical, the objective must be agreed with clinical
leadership, not derived from the previous quarter's uptime. A number chosen to avoid work is worse
than no number.

### Step 4 — Choose the metric type

**Input.** The measurement and the queries it must support.

**Output.** A metric type and, for histograms, bucket boundaries.

**Rationale.** Counters answer "how many per second" after a `rate()`. Gauges answer "what is the
current value". Histograms answer "what is the distribution" and can be aggregated across instances.
Summaries answer quantiles per instance and cannot be aggregated.

The aggregation difference is the decisive one:

```text
histogram: histogram_quantile(0.95, sum by (le) (rate(http_server_request_duration_seconds_bucket[5m])))
summary:   avg(http_server_request_duration_seconds{quantile="0.95"})   -- mathematically wrong
```

Averaging per-instance quantiles does not produce the fleet quantile. It produces a number with no
defined meaning.

**Edge case.** A histogram with poorly chosen buckets cannot estimate a quantile accurately. If the
p99 is 800 ms and the highest bucket boundary is 500 ms, the estimate is bounded by the bucket, not
measured.

### Step 5 — Instrument with semantic conventions

**Input.** The service and its HTTP framework.

**Output.** Metrics, spans, and logs that follow OpenTelemetry semantic conventions.

**Rationale.** Conventions make dashboards portable and alerts comparable across services. The
stable HTTP server metric is `http.server.request.duration`, a histogram in seconds, with
recommended explicit bucket boundaries of 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1,
2.5, 5, 7.5, and 10 seconds. Required attributes include `http.request.method` and `url.scheme`;
conditionally required attributes include `http.response.status_code` and `error.type`.

A custom metric named `api_latency_ms` is not wrong. It is unportable, and every dashboard that uses
it must be rewritten when the service is replaced.

**Branch point.** If the instrumentation library predates the stable conventions, it may emit
`http.server.duration` in milliseconds. The environment variable `OTEL_SEMCONV_STABILITY_OPT_IN`
selects the behaviour. Mixing old and new names in one dashboard produces empty panels.

### Step 6 — Design the log policy before writing any log line

**Input.** The debugging questions and the audit obligations.

**Output.** A schema, a retention policy, and an explicit list of prohibited fields.

**Rationale.** Logs are the highest-risk signal for PHI. A single `logger.info(request.body)` can put
a patient record into a log aggregator with different access controls from the database.

A workable policy:

| Field | Allowed | Example |
|---|---|---|
| Request identifier | Yes | UUID |
| Route template | Yes | `/patients/{id}/risk` |
| Status code | Yes | 503 |
| Duration | Yes | 42 ms |
| Model version | Yes | 0.1.0 |
| Patient identifier | Hashed only | `sha256:ab12...` |
| Request body | No | — |
| Response body | No | — |
| Free-text clinical note | No | — |

**Branch point.** Audit logs are not debug logs. They record who accessed what, are append-only,
and have a retention set by policy. Mixing them into the application log makes both worse: the audit
record becomes deletable with the log, and the debug log fills with access events.

**An enforceable schema.** A structured logger with an allowlist is stronger than a denylist,
because new fields are rejected by default rather than logged by accident:

```python
ALLOWED = {"request_id", "route", "method", "status", "duration_ms",
           "model_version", "patient_hash", "error_type"}

def log_event(logger, **fields):
    rejected = set(fields) - ALLOWED
    if rejected:
        raise ValueError(f"fields not permitted in logs: {sorted(rejected)}")
    logger.info("request", extra=fields)
```

The allowlist makes the policy executable. A developer who tries to log `patient_name` gets an
exception in a test rather than a breach in production. The cost is that every new field requires a
review, which is the point.

### Step 7 — Decide sampling for traces

**Input.** Request rate and the questions traces must answer.

**Output.** A head-based sampling ratio and a tail-based policy if needed.

**Rationale.** At 20 requests per second, 100 percent sampling produces 1.7 million spans per day for
a single service. Head-based sampling at 1 percent is cheap and misses rare failures. Tail-based
sampling keeps all errors and a sample of successes, which is usually the right trade.

**Edge case.** Sampling must propagate the decision through the trace context, or downstream
services will make independent decisions and produce broken traces.

### Step 8 — Build a small number of dashboards

**Input.** The SLOs and the golden signals.

**Output.** One service dashboard with four rows and one SLO dashboard.

**Rationale.** A dashboard is a decision aid. The first row answers "are we meeting the objective".
The second answers "where is the latency". The third answers "what is failing". The fourth answers
"what is saturated". Anything else belongs in an investigation view.

**Branch point.** If a dashboard has more than about twenty panels, nobody reads it during an
incident. Split by question, not by subsystem.

### Step 9 — Design alerts from burn rate

**Input.** The SLO and the response capability.

**Output.** A small set of multi-window, multi-burn-rate alerts with runbooks.

**Rationale.** A single-threshold alert on error rate either fires constantly or fires after the
budget is gone. Multi-window burn-rate alerting compares the current error rate to the rate that
would exhaust the budget over the window.

```text
burn_rate = error_ratio ÷ (1 − SLO)
```

For a 99.9 percent SLO, an error ratio of 1 percent is a burn rate of 10. Sustained for 72 hours, it
exhausts the 30-day budget.

A common pair:

| Severity | Long window | Short window | Burn threshold | Budget consumed |
|---|---|---:|---:|---|
| Page | 1 hour | 5 minutes | 14.4 | 2 percent in 1 hour |
| Ticket | 6 hours | 30 minutes | 6 | 5 percent in 6 hours |

The short window confirms the problem is still active, which suppresses alerts that resolved during
the long window.

**Edge case.** Burn-rate alerting assumes a stable request rate. A service that receives ten
requests per hour cannot use a five-minute window meaningfully; the denominator is too small.

**The expression, in one place.** A page rule needs both windows in the same expression, because
either alone is wrong:

```promql
(
  sum(rate(http_server_requests_total{job="risk-service",status=~"5.."}[1h]))
  / sum(rate(http_server_requests_total{job="risk-service"}[1h]))
) > 14.4 * 0.001
and
(
  sum(rate(http_server_requests_total{job="risk-service",status=~"5.."}[5m]))
  / sum(rate(http_server_requests_total{job="risk-service"}[5m]))
) > 14.4 * 0.001
```

The long window detects a sustained problem. The short window suppresses the alert when the problem
has already resolved, which prevents a page for an incident that ended during the evaluation delay.

### Step 10 — Write the runbook before the alert

**Input.** The alert and the system it observes.

**Output.** A runbook with a diagnosis path, a mitigation, and an escalation.

**Rationale.** An alert without a runbook wakes someone who has to rediscover the system at 02:00.
The runbook is part of the alert definition, not a separate document.

**Branch point.** If the first action in the runbook is "restart the service", the alert is probably
measuring a symptom that a restart will not fix.

### Step 11 — Separate audit from observability

**Input.** The regulatory obligation and the access paths.

**Output.** An append-only audit stream with its own retention and access control.

**Rationale.** HIPAA requires audit controls at 45 CFR 164.312(b). An application log with a
seven-day retention does not satisfy a six-year retention policy for access records. The audit
stream also needs integrity protection, so that an administrator cannot edit it.

### Step 12 — Review cost and cardinality monthly

**Input.** The metrics, logs, and traces bill.

**Output.** A list of the top series by cardinality and the top log sources by volume.

**Rationale.** Cardinality grows silently. A new label with a request identifier multiplies series
by the number of distinct values. A monthly review catches it before the bill arrives.

### 5.1 Edge cases that change the answer

**Cardinality explosion.** Adding `patient_id` as a label on a request counter creates one series
per patient per status code per method. With 50,000 patients, that is hundreds of thousands of
series, most of which are never queried. Use exemplars or traces to carry the identifier, not
labels.

**Counter resets.** A pod restart resets a counter to zero. `rate()` handles this if the metric is a
true counter and the scrape interval is shorter than the reset window. A dashboard that plots the
raw counter shows a sawtooth that looks like an incident.

**Clock skew.** Event-time latency computed from a device clock can be negative or hours long. Use
the ingest timestamp for pipeline latency and the event timestamp only for clinical freshness, and
state which one a panel shows.

**Sampling bias.** Tail sampling that keeps all errors makes the trace sample unrepresentative of
normal traffic. Latency percentiles computed from traces will be biased high. Compute latency from
metrics, and use traces for causation.

**Missing metrics.** A pod that fails to scrape produces a gap, not a zero. An alert on "error rate
above 5 percent" will not fire during a total outage if the metric is absent. Add a
`up == 0` alert or an absent-metric alert.

**Log volume spike during an incident.** The debug logging that helps also fills the pipeline at the
moment the system is least able to handle it. Rate-limit logs and keep a sampling path.

**Retention mismatch.** Metrics retained for 15 days cannot support a 30-day SLO. Either extend
retention or use recording rules that persist the aggregated indicator.

**Multi-tenant label leakage.** A shared metrics backend with a `tenant` label can expose one
tenant's series to another if access control is not configured. Treat metrics as data with an
access policy.

### 5.2 A decision table for alert severity

| Condition | Severity | Response | Window | Runbook owner |
|---|---|---|---|---|
| Budget burn 14.4x, 1 hour | Page | Immediate | 1 h and 5 m | On-call engineer |
| Budget burn 6x, 6 hours | Ticket | Next business day | 6 h and 30 m | Service owner |
| Single dependency error above 10 percent | Page | Immediate | 5 m | On-call engineer |
| Disk above 80 percent | Ticket | This week | 1 h | Platform team |
| Certificate expiring in 21 days | Ticket | This month | 1 d | Platform team |
| Consumer lag above 10 minutes | Page | Immediate | 10 m | Data on-call |

### 5.3 An instrumentation checklist

Before declaring a service observable, check each row. The list is short enough to use in review.

| Check | Pass condition | Consequence if failed |
|---|---|---|
| Request counter with status | Labelled by method, route, status | Cannot compute availability |
| Duration histogram | Buckets cover the SLO threshold | Quantile estimate is bounded by buckets |
| In-flight gauge | Present | Cannot see saturation or queueing |
| Dependency latency | One histogram per dependency | Cannot isolate blame |
| Structured logs | JSON with an allowlist schema | PHI risk, unparseable logs |
| Trace propagation | Context passed to dependencies | Broken traces |
| Resource attributes | Service name, version, environment | Cannot separate deployments |
| Exemplars | At least on the duration histogram | Cannot jump from metric to trace |
| Audit stream | Separate, append-only, retained | Compliance gap |
| Change marker | Deploy annotation on dashboards | Cannot correlate change and effect |

The last row is the one most often skipped. A dashboard without a deploy marker forces the on-call
engineer to ask in chat when the last release was, which is exactly the moment nobody is answering.

## 6. Worked example

A clinical risk API serves 20 requests per second at peak. The team proposes a 99.9 percent
availability SLO over a 30-day rolling window. The current alerting is a single rule: page when the
five-minute error rate exceeds 1 percent.

### 6.1 Inputs

| Parameter | Value | Source |
|---|---|---|
| Request rate | 20 requests/s | Load test |
| SLO | 99.9 percent availability | Proposed |
| Window | 30 days rolling | Proposed |
| Current alert | 5-minute error rate above 1 percent | Existing configuration |
| Metrics retention | 15 days | Platform default |
| Series per pod | 20 | Instrumentation estimate |
| Pods | 200 across services | Cluster inventory |

### 6.2 Arithmetic: the error budget

Minutes:

```text
30 × 24 × 60 = 43,200 minutes
43,200 × 0.001 = 43.2 minutes
```

Requests:

```text
20 requests/s × 86,400 s/day = 1,728,000 requests/day
1,728,000 × 30 = 51,840,000 requests in 30 days
51,840,000 × 0.001 = 51,840 failed requests
```

The budget is 43.2 minutes or 51,840 failed requests. At a steady 20 requests per second, a total
outage exhausts the request budget in:

```text
51,840 ÷ 20 = 2,592 seconds = 43.2 minutes
```

The two expressions agree exactly, which is the expected result at a constant request rate. They
diverge when the rate varies: a service that is down overnight has few requests during the outage
and can spend the minute budget without spending the request budget.

### 6.3 Arithmetic: burn rate

The burn rate for an error ratio of 1 percent against a 99.9 percent SLO:

```text
1% ÷ 0.1% = 10
```

Sustained, it exhausts the 30-day budget in:

```text
720 hours ÷ 10 = 72 hours
```

At 5 percent errors, the burn rate is 50 and the budget lasts 14.4 hours. At 10 percent, it lasts
7.2 hours.

### 6.4 Arithmetic: why the existing alert is bad

A 1 percent error rate over five minutes is a burn rate of 10. That is a real problem, but the alert
does not distinguish a brief spike from a sustained fault. Suppose it fires 30 times in a month and
6 are genuine incidents. Precision:

```text
6 ÷ 30 = 20 percent
```

Twenty-four false pages per month is roughly one every 30 hours. On-call fatigue follows, and the
next real page is more likely to be acknowledged without investigation.

Multi-window burn-rate alerting at 14.4x over one hour and five minutes would fire only when the
budget is being consumed at 14.4 times the sustainable rate, which for a 99.9 percent SLO is an error
ratio of 1.44 percent sustained for an hour. That is rarer and more actionable.

### 6.5 Arithmetic: histogram quantile

A latency histogram with cumulative buckets:

| Upper bound (s) | Cumulative count |
|---:|---:|
| 0.01 | 900 |
| 0.025 | 950 |
| 0.05 | 980 |
| 0.1 | 995 |
| 0.25 | 999 |
| 0.5 | 1000 |
| 1.0 | 1000 |
| 2.5 | 1000 |

The p95 estimate:

```text
rank = 0.95 × 1000 = 950
bucket = 0.025 with cumulative 950
interpolation: 0.01 + (950 − 900)/(950 − 900) × (0.025 − 0.01) = 0.025 s
```

That lands exactly on the bucket boundary, which is a warning: the estimate is constrained by the
bucket layout. The p99:

```text
rank = 990
bucket = 0.1 with cumulative 995
0.05 + (990 − 980)/(995 − 980) × (0.1 − 0.05) = 0.0833 s = 83.3 ms
```

Both estimates are only as good as the buckets. A p99 of 83 ms with a highest bucket of 2.5 s is
plausible; a true p99 above 2.5 s would be reported as approximately 2.5 s.

### 6.6 Arithmetic: cardinality

Base series with 200 pods and 20 series per pod:

```text
200 × 20 = 4,000 series
```

Add one label with five distinct values:

```text
4,000 × (5 + 1) = 24,000 series
```

Add one label with fifty distinct values:

```text
4,000 × 51 = 204,000 series
```

If the label is a patient identifier with 50,000 values, the multiplication is not survivable. Most
managed metrics services bill per series per month, so the difference between 4,000 and 204,000
series is the difference between a negligible line item and a budget conversation.

### 6.7 Two diagnostics that reveal a misconfiguration

**Diagnostic 1: the p99 dashboard disagrees with the trace data.**

The dashboard uses `avg(http_server_request_duration_seconds{quantile="0.95"})` from a summary. The
trace data, sampled from real requests, shows a much higher tail. The average of per-instance
quantiles is not the fleet quantile, and it systematically understates the tail because slow
instances are diluted by fast ones. Confirm by comparing the panel with a histogram query:

```promql
histogram_quantile(0.95,
  sum by (le) (rate(http_server_request_duration_seconds_bucket[5m])))
```

If the two disagree materially, the summary is the problem.

**Diagnostic 2: an outage produces no alert.**

The error-rate alert never fired during a five-minute total outage. The cause is that the metric
disappeared rather than rising: no requests reached the service, so no request counter was
incremented, and the alert expression `rate(errors[5m]) / rate(requests[5m])` evaluated to no data.
The fix is an availability alert on `up` or on a synthetic probe, not a lower error threshold.

A third diagnostic is a metrics bill that triples after a release. Diff the series count by metric
name before and after the deploy. A new label on a high-traffic metric is almost always the cause.

### 6.8 Vary one assumption

Change the SLO from 99.9 to 99.95 percent. The budget halves:

```text
43,200 × 0.0005 = 21.6 minutes
25,920,000 × 0.0005 = 12,960 failed requests
```

The burn-rate thresholds change accordingly. A 14.4x burn now corresponds to an error ratio of:

```text
14.4 × 0.0005 = 0.72 percent
```

The same 1 percent error rate that was a 10x burn under 99.9 percent is now a 20x burn under 99.95
percent, and the existing alert threshold becomes more sensitive without anyone changing it. That is
the correct behaviour, and it is a reason to define alerts in terms of burn rate rather than a fixed
error percentage: the alert follows the objective.

## 7. Code, query, or calculation

The following standard-library Python computes error budgets, burn rates, histogram quantiles, and
cardinality. It reproduces Section 6.

```python
"""SLO, burn-rate, histogram quantile, and cardinality model. Stdlib only."""
from __future__ import annotations

import math


def budget_minutes(slo: float, days: int = 30) -> float:
    return days * 24 * 60 * (1 - slo)


def budget_requests(slo: float, rps: float, days: int = 30) -> float:
    return rps * days * 86_400 * (1 - slo)


def burn_rate(error_ratio: float, slo: float) -> float:
    """How many times faster than the budget the current error rate consumes it."""
    return error_ratio / (1 - slo)


def time_to_exhaust_hours(error_ratio: float, slo: float, days: int = 30) -> float:
    rate = burn_rate(error_ratio, slo)
    if rate <= 0:
        return math.inf
    return (days * 24) / rate


def histogram_quantile(q: float, buckets: list[tuple[float, float]]) -> float:
    """buckets: sorted (upper_bound, cumulative_count). Linear interpolation within a bucket."""
    total = buckets[-1][1]
    if total == 0:
        return float("nan")
    rank = q * total
    previous_bound = 0.0
    previous_count = 0.0
    for bound, count in buckets:
        if count >= rank:
            if count == previous_count:
                return bound
            fraction = (rank - previous_count) / (count - previous_count)
            return previous_bound + fraction * (bound - previous_bound)
        previous_bound, previous_count = bound, count
    return buckets[-1][0]


def series_count(pods: int, series_per_pod: int, extra_labels: int = 0) -> int:
    return pods * series_per_pod * (extra_labels + 1)


def main() -> None:
    print("error budgets, 30-day window")
    for slo in (0.99, 0.995, 0.999, 0.9995):
        print(f"  SLO {slo:7.3%}  {budget_minutes(slo):7.1f} min  "
              f"{budget_requests(slo, 20):12,.0f} requests at 20 rps")

    print()
    print("burn rate and time to exhaust the 99.9% budget")
    for ratio in (0.001, 0.005, 0.01, 0.05, 0.10):
        br = burn_rate(ratio, 0.999)
        tte = time_to_exhaust_hours(ratio, 0.999)
        print(f"  error_ratio={ratio:6.3%}  burn={br:6.2f}x  exhaust_in={tte:8.2f} h")

    print()
    print("multi-window alert thresholds (99.9% SLO)")
    print("  page:   1h burn > 14.4x and 5m burn > 14.4x")
    print("  ticket: 6h burn > 6x and 30m burn > 6x")
    print("  These consume 2% and 5% of the 30-day budget respectively when sustained.")

    print()
    print("histogram_quantile(0.95) from cumulative buckets")
    buckets = [(0.01, 900), (0.025, 950), (0.05, 980), (0.1, 995),
               (0.25, 999), (0.5, 1000), (1.0, 1000), (2.5, 1000)]
    for q in (0.5, 0.9, 0.95, 0.99):
        print(f"  q={q:4.2f}  {histogram_quantile(q, buckets)*1000:8.2f} ms")

    print()
    print("series cardinality")
    for label_cost in (0, 5, 50):
        print(f"  extra high-cardinality labels={label_cost:2d}  "
              f"series={series_count(200, 20, label_cost):,}")

    print()
    print("alert precision")
    for fired, real in ((30, 6), (30, 24), (8, 8), (50, 5)):
        print(f"  fired={fired:2d} real={real:2d}  precision={real/fired:5.1%}  "
              f"false_pages={fired-real}")


if __name__ == "__main__":
    main()
```

Actual output:

```text
error budgets, 30-day window
  SLO 99.000%    432.0 min       518,400 requests at 20 rps
  SLO 99.500%    216.0 min       259,200 requests at 20 rps
  SLO 99.900%     43.2 min        51,840 requests at 20 rps
  SLO 99.950%     21.6 min        25,920 requests at 20 rps
```

The 99.9 percent row matches Section 6.2: 51,840 failed requests and 43.2 minutes. The check that
catches an error in this arithmetic is dimensional: requests per second times seconds equals
requests, and at a constant rate the minute budget and the request budget must agree.

The full output continues:

```text
burn rate and time to exhaust the 99.9% budget
  error_ratio= 0.100%  burn=  1.00x  exhaust_in=  720.00 h
  error_ratio= 0.500%  burn=  5.00x  exhaust_in=  144.00 h
  error_ratio= 1.000%  burn= 10.00x  exhaust_in=   72.00 h
  error_ratio= 5.000%  burn= 50.00x  exhaust_in=   14.40 h
  error_ratio=10.000%  burn=100.00x  exhaust_in=    7.20 h

multi-window alert thresholds (99.9% SLO)
  page:   1h burn > 14.4x and 5m burn > 14.4x
  ticket: 6h burn > 6x and 30m burn > 6x
  These consume 2% and 5% of the 30-day budget respectively when sustained.

histogram_quantile(0.95) from cumulative buckets
  q=0.50      5.56 ms
  q=0.90     10.00 ms
  q=0.95     25.00 ms
  q=0.99     83.33 ms

series cardinality
  extra high-cardinality labels= 0  series=4,000
  extra high-cardinality labels= 5  series=24,000
  extra high-cardinality labels=50  series=204,000

alert precision
  fired=30 real= 6  precision=20.0%  false_pages=24
  fired=30 real=24  precision=80.0%  false_pages=6
  fired= 8 real= 8  precision=100.0%  false_pages=0
  fired=50 real= 5  precision=10.0%  false_pages=45
```

The following Prometheus rule group is YAML. It is validated by parsing, not loaded into a running
Prometheus.

```yaml
groups:
  - name: risk-service-slo
    interval: 30s
    rules:
      - record: risk_service:request_error_ratio5m
        expr: |
          sum(rate(http_server_requests_total{job="risk-service",status=~"5.."}[5m]))
          /
          sum(rate(http_server_requests_total{job="risk-service"}[5m]))
      - alert: RiskServiceErrorBudgetBurnPage
        expr: |
          risk_service:request_error_ratio5m > (14.4 * 0.001)
          and
          risk_service:request_error_ratio5m offset 1h > (14.4 * 0.001)
        for: 2m
        labels:
          severity: page
        annotations:
          summary: "risk-service is burning the 99.9% error budget at more than 14.4x"
          runbook: "https://runbooks.example.org/risk-service/error-budget"
      - alert: RiskServiceDown
        expr: up{job="risk-service"} == 0
        for: 2m
        labels:
          severity: page
        annotations:
          summary: "risk-service has no scrape target"
```

The `RiskServiceDown` rule exists because the ratio rule cannot fire when the denominator is absent.
That is the Diagnostic 2 failure in rule form.

## 8. Data, standards, and regulatory context

### 8.1 HIPAA audit controls

45 CFR 164.312(b) requires hardware, software, and procedural mechanisms that record and examine
activity in information systems containing electronic PHI. It does not specify a format, a
retention period, or a technology. It does require that the records exist and can be examined.

An application debug log with a seven-day retention is not an audit control. An audit stream with
who, what, when, and the outcome, retained per policy and protected from modification, is. The two
can share a pipeline only if the retention and access controls of the stricter one apply to both.

### 8.2 The minimum necessary standard

45 CFR 164.502(b) requires reasonable efforts to limit PHI to the minimum necessary. For
observability, this constrains what may be logged and labelled. A trace that records the full
request body violates the standard even if the trace store is access-controlled, because the body
was not necessary for the operational purpose.

### 8.3 NIST SP 800-207 monitoring tenets

The fifth and seventh tenets of zero trust concern monitoring the integrity and security posture of
assets and collecting information to improve posture. Observability is where those tenets become
operational: an anomaly in authentication failures or egress volume is a security signal, and it
belongs in the same pipeline as the reliability signals.

### 8.4 OpenTelemetry semantic conventions

The OpenTelemetry project publishes semantic conventions that define metric names, units, and
attributes. The HTTP conventions were declared stable, and the current server metric is
`http.server.request.duration` in seconds. Following the conventions is not a regulatory
requirement. It is an interoperability requirement, and it determines whether a dashboard can be
reused across services and vendors.

### 8.5 Prometheus and aggregation

The Prometheus documentation is explicit that summary quantiles cannot be aggregated across
instances, and that histograms are the appropriate type when aggregation is needed. Native
histograms, which store a sparse bucket structure rather than a fixed set of series, are stable in
Prometheus 3.8 and are the recommended choice where supported.

### 8.6 What the standards do not settle

They do not define an SLO, a retention period, an alert threshold, or a dashboard layout. Those are
operational decisions informed by clinical risk, contract, and budget.

## 9. Industry practice and case evidence

### 9.1 Documented practice

**Multi-window burn-rate alerting is the published SRE practice.** The Google SRE Workbook describes
alerting on burn rate with a long window and a short window, and gives thresholds such as 14.4 for a
one-hour page and 6 for a six-hour ticket against a 30-day budget. The numbers in Section 6 come
from that guidance.

**The four golden signals are a documented framework.** Latency, traffic, errors, and saturation, as
described in Google's Site Reliability Engineering book, remain the standard checklist for a service
dashboard.

**OpenTelemetry conventions are stable and specific.** The HTTP server metric name, unit, and bucket
boundaries are published and versioned. The rename from `http.server.duration` in milliseconds to
`http.server.request.duration` in seconds is a documented breaking change with a migration
environment variable.

**Prometheus native histograms are stable.** The Prometheus documentation states that native
histograms became stable in version 3.8, and that scraping requires `scrape_native_histograms: true`.
They reduce series count and improve quantile accuracy compared with classic histograms.

### 9.2 Case evidence

1. **The Google SRE Workbook's alerting chapter** is the primary public source for burn-rate
   alerting and is used across the industry. It is a book, not a peer-reviewed study, and its
   thresholds are guidance rather than measured optima.
2. **The OpenTelemetry HTTP semantic convention change** affected every instrumented service that
   upgraded. It is a concrete example of why metric names should not be invented locally.
3. **Public cloud billing incidents caused by metrics cardinality** are widely reported by
   practitioners, though rarely with named organisations. The mechanism is well understood: a label
   with unbounded values multiplies series, and series are the billing unit.

### 9.3 Inference

The following are inferences, labelled as such:

- Most clinical services have adequate latency metrics and inadequate freshness metrics. The
  question "how old is the data behind this number" is rarely instrumented, and it is the question
  clinicians ask.
- Alert fatigue is more often caused by too many low-value alerts than by too few high-value ones.
  Removing alerts is harder politically than adding them, which is why the count grows.
- Teams that define SLOs before choosing a monitoring vendor end up with simpler stacks than teams
  that choose the vendor first.

A further inference concerns clinical data freshness. Most pipelines instrument whether the job ran,
not whether the data is current. A job that exits zero after writing zero rows is a success by the
former measure and a failure by the latter. Adding a `max(event_time)` gauge per table, and alerting
on its age, catches an entire class of silent failures that no job-level alert detects.

The cost of that gauge is negligible: one series per table. The benefit is that the question "how
old is the number on this dashboard" has an answer. In a clinical setting that question is asked
during every data quality review, and a pipeline that cannot answer it is trusted less than one that
can.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No SLO | Reliability debated anecdotally | No agreed target | Cannot prioritise | Define SLI and SLO |
| Alert on every error | Pages several times a day | No burn-rate logic | On-call fatigue | Multi-window burn rate |
| Averaging summary quantiles | Dashboard disagrees with traces | Quantiles not aggregable | Wrong latency picture | Histograms |
| Cardinality explosion | Metrics bill rises after deploy | High-cardinality label | Cost, slow queries | Label review, exemplars |
| PHI in logs | Free-text field in log schema | Request body logged | Breach risk | Prohibited-field list, redaction |
| No audit stream | Cannot answer access questions | Debug log used as audit | Compliance gap | Separate append-only audit |
| Missing `up` alert | Total outage produces no page | Absent metric, not zero | Undetected downtime | Availability alert |
| Retention shorter than SLO window | 30-day SLO, 15-day metrics | Data deleted | Cannot compute objective | Recording rules or longer retention |
| Clock skew in latency | Negative latency values | Device clock used | Nonsense panels | Use ingest time for pipeline |
| Trace sampling bias | Trace p99 above metric p99 | Errors oversampled | Wrong prioritisation | Metrics for latency |
| Log volume spike | Pipeline lag during incident | Debug logging amplified | Lost logs when needed | Rate limits, sampling |
| Dashboard sprawl | Nobody opens the dashboard | Too many panels | No shared view | One dashboard per question |
| Alert without runbook | On-call asks in chat | No documented response | Slow mitigation | Runbook before alert |
| Metric name drift | Empty panels after upgrade | Convention change | Blind spots | Semantic conventions |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Request rate is roughly constant | Request-based budget arithmetic | Burst traffic, batch jobs | Compare hourly rates |
| 30-day window matches the review cycle | SLO definition | Weekly operational cadence | Ask the service owner |
| Scrape interval is shorter than the shortest alert window | Rate accuracy | 60 s scrape, 1 m window | Check scrape configuration |
| Histogram buckets cover the tail | Quantile accuracy | p99 above the top bucket | Compare with traces |
| Metrics retention covers the SLO window | Objective computation | 15-day retention, 30-day SLO | Query the oldest sample |
| Logs can be redacted before storage | PHI control | Unstructured free text | Review the log schema |
| Traces carry no PHI attributes | Privacy | Framework auto-instrumentation | Inspect a sample span |
| Series count is the billing unit | Cost model | Vendor bills by ingestion | Read the contract |
| The on-call engineer can act on the alert | Alert value | No access, no runbook | Drill the response |
| Audit retention is set by policy | Compliance | Policy not documented | Obtain the retention schedule |

## 12. Exercises

### 12.1 Write an SLO

**Deliverable.** An SLI, SLO, window, and error budget for a FHIR read API used by a clinical
viewer, with the measurement point named.

**Acceptance criteria.** Numerator and denominator are explicit. Budget is expressed in minutes and
in requests. Measurement point is stated.

**Time.** 30 minutes.

### 12.2 Compute budgets

**Deliverable.** Error budget in minutes and requests for 99.9 and 99.95 percent over 30 days at 50
requests per second.

**Acceptance criteria.** Correct arithmetic. States whether the two expressions agree and why.

**Time.** 25 minutes.

### 12.3 Design burn-rate alerts

**Deliverable.** A page rule and a ticket rule for the SLO in 12.1, with windows, thresholds, and the
budget consumed when sustained.

**Acceptance criteria.** Uses two windows per severity. Justifies the thresholds. Includes an
availability rule for the missing-metric case.

**Time.** 40 minutes.

### 12.4 Choose metric types

**Deliverable.** For request count, in-flight requests, request latency, queue depth, and record age,
choose counter, gauge, histogram, or summary, with a reason.

**Acceptance criteria.** Latency is a histogram, not a summary. Gauge and counter are distinguished
correctly.

**Time.** 25 minutes.

### 12.5 Instrument an endpoint

**Deliverable.** A code or configuration sketch that emits `http.server.request.duration` with the
required attributes, plus a trace span around the handler.

**Acceptance criteria.** Metric name and unit follow the convention. No PHI attributes. Sampling
decision stated.

**Time.** 45 minutes.

### 12.6 Write a log policy

**Deliverable.** A prohibited-field list, a redaction approach, and a retention schedule that
separates audit from debug.

**Acceptance criteria.** Names at least five prohibited fields. Distinguishes audit retention from
debug retention. Describes how a hashed identifier is produced.

**Time.** 35 minutes.

### 12.7 Estimate cardinality

**Deliverable.** For 100 pods emitting 15 series each, compute series with zero, three, and twenty
extra label values, and state the cost implication in general terms.

**Acceptance criteria.** Correct multiplication. Identifies which labels are safe (bounded
enumerations) and which are not.

**Time.** 20 minutes.

### 12.8 Diagnose a silent outage

**Deliverable.** A written explanation of why a ratio-based error alert does not fire during a total
outage, and two alert rules that would.

**Acceptance criteria.** Names the absent-metric mechanism. Provides an `up` rule and a synthetic
probe rule.

**Time.** 30 minutes.

## 13. Self-check answers

### 13.1 SLO example

SLI: proportion of FHIR read requests returning a 2xx response within 500 ms, measured at the ingress
over a rolling 30-day window. SLO: 99.9 percent. Budget: 43.2 minutes or 0.1 percent of requests,
whichever is exhausted first.

### 13.2 Budget arithmetic

At 50 requests per second over 30 days:

```text
50 × 30 × 86,400 = 129,600,000 requests
99.9%  → 129,600 failed requests
99.95% →  64,800 failed requests
```

In minutes: 43.2 and 21.6. The two expressions agree when the request rate is constant, because
`50 × 43.2 × 60 = 129,600`.

### 13.3 Burn-rate alerts

Page: one-hour burn above 14.4 and five-minute burn above 14.4. Ticket: six-hour burn above 6 and
thirty-minute burn above 6. At 14.4x, one hour consumes `14.4 / 720 = 2 percent` of the 30-day
budget. At 6x, six hours consumes `6 × 6 / 720 = 5 percent`.

### 13.4 Metric types

Request count: counter. In-flight requests: gauge. Request latency: histogram. Queue depth: gauge.
Record age: gauge, or a histogram if the distribution matters.

### 13.5 Instrumentation

The handler records a histogram observation with attributes `http.request.method`,
`url.scheme`, `http.response.status_code`, and `http.route`. It starts a span named after the route
and sets the same attributes. The trace context is propagated to downstream calls. Sampling is
tail-based, keeping all errors and 1 percent of successes.

### 13.6 Log policy

Prohibited: request body, response body, patient name, medical record number, date of birth, free-
text note, access token. Redaction: a structured logger with an allowlist of fields, not a denylist.
Audit retention: per the organisation's HIPAA policy, commonly six years. Debug retention: 30 days.

### 13.7 Cardinality

```text
100 × 15 = 1,500 series
three extra values:  1,500 × 4  = 6,000
twenty extra values: 1,500 × 21 = 31,500
```

Safe labels are bounded enumerations such as status code, method, and region. Unsafe labels are
identifiers, timestamps, and free text.

### 13.8 Silent outage

The ratio `errors / requests` has no value when both are absent, so the comparison is false and no
alert fires. An `up{job="risk-service"} == 0` rule fires after two minutes. A synthetic probe that
requests a known endpoint and alerts on failure catches the case where the service is up but not
serving.

## 14. Interview questions

**1. What is the difference between an SLI and an SLO?**
A strong answer says an SLI is a measurement and an SLO is a target for it over a window. A weak
answer treats them as synonyms.

**2. Why can you not average p99 values from different instances?**
A strong answer explains that a quantile is not a linear statistic and that averaging them has no
defined meaning. A weak answer says it is inaccurate.

**3. When would you use a summary instead of a histogram?**
A strong answer says when per-instance quantiles are sufficient and aggregation is not needed, and
notes the cost. A weak answer says summaries are more accurate.

**4. What is a burn rate?**
A strong answer defines it as the error ratio divided by the budget ratio and connects it to time
to exhaustion. A weak answer says it is the error rate.

**5. Why does an error-rate alert fail during a total outage?**
A strong answer explains that the metric disappears rather than rising, so the ratio has no value.
A weak answer says the threshold is too high.

**6. How do you log without PHI?**
A strong answer describes an allowlist schema, hashed identifiers, and a separate audit stream. A
weak answer says "do not log patient data".

**7. What is cardinality and why does it cost money?**
A strong answer explains that each label combination is a time series and that managed services bill
per series. A weak answer says it is the number of metrics.

**8. How would you instrument data freshness?**
A strong answer computes the difference between the current time and the event time at the serving
layer, as a histogram or a gauge of the oldest record. A weak answer says to log the timestamp.

**9. What belongs in an audit log but not a debug log?**
A strong answer names access events, actor identity, and outcome, and explains the retention and
immutability difference. A weak answer says both should contain everything.

**10. How do you decide the SLO for a clinical service?**
A strong answer starts from the clinical workflow and the consequence of delay, involves clinical
owners, and states that the number is a decision, not a measurement. A weak answer copies 99.9
percent from a template.

## 15. Key terms

- **Alert** — Notification requiring action.
- **Audit log** — Append-only record of access and actions.
- **Burn rate** — Budget consumption speed relative to the SLO.
- **Cardinality** — Number of distinct label combinations.
- **Counter** — Monotonically increasing metric.
- **Error budget** — Allowed SLO violation in a window.
- **Exemplar** — Metric reference to a trace.
- **Gauge** — Metric that can rise and fall.
- **Golden signals** — Latency, traffic, errors, saturation.
- **Histogram** — Bucketed distribution.
- **Ingest time** — Time a record enters the platform.
- **Label** — Metric dimension.
- **Metric** — Numeric measurement over time.
- **Observability** — Ability to infer internal state from outputs.
- **OpenTelemetry** — Vendor-neutral instrumentation standard.
- **PHI** — Protected health information.
- **Precision** — Fraction of alerts that are real.
- **Prometheus** — Metrics system and query language.
- **Quantile** — Value below which a fraction of observations fall.
- **Recording rule** — Precomputed query stored as a metric.
- **RED method** — Rate, errors, duration.
- **Recall** — Fraction of incidents that alert.
- **Retention** — How long signals are kept.
- **Runbook** — Documented response to an alert.
- **Sampling** — Keeping a fraction of traces.
- **Saturation** — How full a constrained resource is.
- **SLA** — Contractual service level agreement.
- **SLI** — Service level indicator.
- **SLO** — Service level objective.
- **Span** — Timed operation in a trace.
- **Summary** — Client-side quantile estimate.
- **Trace** — Graph of spans for one request.
- **USE method** — Utilization, saturation, errors.

## 16. Sources

All sources accessed 26 October 2026.

1. Google. *Site Reliability Engineering*, Chapter 6, Monitoring Distributed Systems.
   https://sre.google/sre-book/monitoring-distributed-systems/
2. Google. *The Site Reliability Workbook*, Chapter 5, Alerting on SLOs.
   https://sre.google/workbook/alerting-on-slos/
3. OpenTelemetry. *Semantic conventions for HTTP metrics*.
   https://opentelemetry.io/docs/specs/semconv/http/http-metrics/
4. Prometheus. *Histograms and summaries*.
   https://prometheus.io/docs/practices/histograms/
5. Prometheus. *Native Histograms*.
   https://prometheus.io/docs/specs/native_histograms/
6. Prometheus. *Metric types*.
   https://prometheus.io/docs/concepts/metric_types/
7. HHS. *HIPAA Security Rule*.
   https://www.hhs.gov/hipaa/for-professionals/security/index.html
8. Cornell Legal Information Institute. *45 CFR 164.312*.
   https://www.law.cornell.edu/cfr/text/45/164.312
9. Cornell Legal Information Institute. *45 CFR 164.502*.
   https://www.law.cornell.edu/cfr/text/45/164.502
10. NIST. *SP 800-207: Zero Trust Architecture*. https://csrc.nist.gov/pubs/sp/800/207/final
11. Kubernetes. *Metrics for Kubernetes system components*.
    https://kubernetes.io/docs/concepts/cluster-administration/system-metrics/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_streaming.md`](05_streaming.md)
- Next topic: [`07_security.md`](07_security.md)
- Related topic: [`03_kubernetes.md`](03_kubernetes.md)
- Survey file: [`../../course/26_cloud_streaming_production_labs.md`](../../course/26_cloud_streaming_production_labs.md)
- Lab root: [`../../labs/`](../../labs/)
- Reference stack: [`../../labs/cloud/`](../../labs/cloud/)
- Deployment probes: [`../../labs/cloud/k8s/03-deployment.yaml`](../../labs/cloud/k8s/03-deployment.yaml)
- Network policy: [`../../labs/cloud/k8s/08-networkpolicy.yaml`](../../labs/cloud/k8s/08-networkpolicy.yaml)
