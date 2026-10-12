# Observability & Monitoring — Playbook

> **Purpose:** make the internal state of a running system answerable from the outside, without SSH, without asking the author, and without guessing. Monitoring tells you *that* something is wrong; observability lets you ask *why* — including questions nobody anticipated.

> **Rule of thumb:** if you can only answer a question that you pre-built a dashboard for, you have monitoring, not observability. Incidents rarely ask the questions you planned for.

---

## 1. Scope

| In scope | Out of scope |
|---|---|
| Metrics, logs, traces and profiles | Application business logic |
| Dashboards, alerts, runbooks, on-call | Incident command process (see [SLOs & incident response](03-slos-error-budgets-and-incident-response.md)) |
| Instrumentation standards and libraries | Vendor procurement |
| Observability for classical services | — |
| Observability for AI/LLM services (§8) | Model training observability (a separate discipline) |
| Retention, cost and cardinality governance | — |

---

## 2. The three pillars (and the two that are now mandatory)

| Pillar | Answers | Cost profile | Cardinality |
|---|---|---|---|
| **Metrics** | "Is it healthy? How is it trending?" | Cheap to store, cheap to query | **Low** — must be bounded |
| **Logs** | "What exactly happened in this case?" | Expensive; grows with traffic | **High** |
| **Traces** | "Where in the distributed request did the time go?" | Moderate; sampling-controlled | **Very high** |
| **Profiles** (4th) | "Which function is burning CPU/memory?" | Moderate, periodic | Low per sample |
| **Evals/quality** (5th, AI) | "Is the output *good*?" | Per-sample cost | Per-request |

> **The maturity ladder:** metrics-only (you know *that*), + logs (you know *what*), + traces (you know *where*), + profiles (you know *which code*), + evaluations (you know *whether it's any good* — the AI-specific rung).

---

## 3. SLIs and the questions they must answer

Every service must be able to answer these six questions **from the outside**:

| Question | The SLI that answers it | Where it lives |
|---|---|---|
| Is it up? | Availability (success ratio) | Dashboard + alert |
| Is it fast? | Latency p50/p95/p99 | Dashboard + alert |
| Is it correct? | Error rate by type | Dashboard + alert |
| Is it saturated? | Utilization / queue depth | Dashboard + alert |
| Is it working *for the business*? | Business KPI (orders, signups, tokens) | Dashboard |
| **Is the AI output good?** | Quality/eval score, groundedness, retrieval hit rate | Dashboard + alert |

> **The rule:** technical health ≠ business health. A service can be 100% available and serving 100% wrong answers. Both must be instrumented.

---

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        SERVICES (instrumented)                       │
│   OpenTelemetry SDK:  metrics · traces · logs · (GenAI attributes)   │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ OTLP
                            ▼
                  ┌───────────────────┐
                  │   COLLECTOR       │  receive · process · sample ·
                  │  (OTel Collector) │  redact PII · route · batch
                  └────┬────┬────┬────┘
        ┌──────────────┘    │    └──────────────┐
        ▼                   ▼                   ▼
 ┌────────────┐     ┌────────────┐      ┌────────────┐
 │  METRICS   │     │   LOGS     │      │  TRACES    │
 │ Prometheus │     │ Loki/ES    │      │ Tempo/Jaeger│
 │ /Mimir     │     │            │      │            │
 └─────┬──────┘     └─────┬──────┘      └─────┬──────┘
       └──────────────────┼──────────────────┘
                          ▼
                 ┌────────────────────┐
                 │   GRAFANA / UI     │  dashboards · explore · SLOs
                 └────────┬───────────┘
                          ▼
                 ┌────────────────────┐
                 │  ALERT MANAGER     │  routes → PagerDuty/Opsgenie
                 └────────┬───────────┘
                          ▼
                 ┌────────────────────┐
                 │  RUNBOOKS          │  every alert links to one
                 └────────────────────┘
```

**AI/LLM add-on:** a parallel path for LLM traces — prompt/response, token counts, cost, model version, retrieval results, guardrail decisions, eval scores — typically to a dedicated tool (LangSmith, Langfuse, Arize Phoenix, Braintrust, or a self-hosted equivalent).

### Component choices

| Layer | Options | Notes |
|---|---|---|
| **Instrumentation** | OpenTelemetry (default), vendor SDKs | **Use OTel** — vendor-neutral, avoids lock-in |
| **Collector** | OTel Collector | The right place to redact PII and control sampling |
| **Metrics store** | Prometheus (small), Mimir/Thanos (scale), Datadog | Prometheus is the default; long-term needs remote storage |
| **Logs** | Loki (cheap, label-indexed), Elasticsearch (rich search) | Loki is far cheaper for high volume |
| **Traces** | Tempo, Jaeger, Datadog APM | Tail-based sampling matters at scale |
| **Profiles** | Pyroscope, Parca | Underused; finds CPU hogs fast |
| **AI traces** | LangSmith, Langfuse, Arize, Braintrust | Capture prompts, tokens, cost, evals |
| **Alerting** | Alertmanager → PagerDuty/Opsgenie | Routing, dedup, grouping, escalation |

---

## 5. Instrumentation standards

### 5.1 The RED method — for every request-serving service

| Signal | Definition | Alert on |
|---|---|---|
| **R**ate | Requests per second | Anomalous drop or spike |
| **E**rrors | Error rate (5xx / total) | SLO threshold |
| **D**uration | Latency p50/p95/p99 | SLO threshold |

### 5.2 The USE method — for every resource

| Signal | Definition | Alert on |
|---|---|---|
| **U**tilization | % of capacity busy | > 80% sustained |
| **S**aturation | Queue depth / waiting work | Any sustained queue growth |
| **E**rrors | Resource-level errors | > 0 |

**Use RED for services and USE for resources.** Between them they cover most of what pages you.

### 5.3 Structured logging rules

| Rule | Why |
|---|---|
| **JSON, not free text** | Parseable, queryable, indexable |
| **Always include a trace ID** | Joins a log line to a distributed trace |
| Include service, version, environment, request ID | Enables slicing during incidents |
| **Log at boundaries**, not everywhere | Cost and noise |
| Use levels consistently: ERROR = needs a human, WARN = degraded, INFO = state change, DEBUG = off in prod | Level inflation destroys signal |
| **Never log secrets, tokens, credentials or raw PII** | Legal liability; redact at the collector too |
| Include the decision, not just the event | "Rejected request: quota exceeded" beats "quota check failed" |

### 5.4 Trace standards

| Rule | Why |
|---|---|
| Propagate context across **every** service boundary | A broken trace chain is useless |
| One span per outbound call (DB, cache, HTTP, model) | Shows which hop is slow |
| Span attributes: `db.system`, `http.status_code`, `peer.service` | Standardised queries |
| Sample intelligently: head sampling for volume, **tail sampling for errors** | Never lose the trace of a failed request |
| **Always retain 100% of errors and 100% of slow requests** | The two you actually need |

### 5.5 Metric naming and cardinality

| Rule | Why |
|---|---|
| Names: `service_operation_unit` (`http_requests_total`, `llm_tokens_total`) | Consistent, queryable |
| **Never put unbounded values in labels** — no user IDs, no request IDs, no full URLs, no prompt text | Cardinality explosion = cost blowout + a monitoring outage |
| Bound label sets: status class (2xx/4xx/5xx), route template (`/users/{id}` not `/users/123`), model name, region | Keeps cardinality in the hundreds, not millions |
| Cap: a metric should have **< ~10,000 active series** | Above that, question it |

> **The cardinality trap:** a single `user_id` label on a request counter at 1M users creates 1M time series — which will not only cost a fortune but can take down the metrics backend itself, so **your observability outage blinds you during the outage you're trying to debug.**

---

## 6. SOPs and runbooks

### 6.1 The dashboard standard

Every service gets **one primary dashboard**, ordered top-down as the on-call reads it:

```
┌──────────────────────────────────────────────────────────┐
│ 1. SLO status — error budget remaining, burn rate        │  ← "are we in trouble?"
├──────────────────────────────────────────────────────────┤
│ 2. RED — request rate, error rate, latency p50/p95/p99   │  ← "how bad, and where?"
├──────────────────────────────────────────────────────────┤
│ 3. Dependencies — DB, cache, queue, third parties        │  ← "is it us or them?"
├──────────────────────────────────────────────────────────┤
│ 4. Saturation — CPU, memory, connections, queue depth    │  ← "what's the ceiling?"
├──────────────────────────────────────────────────────────┤
│ 5. Business KPI — orders, signups, tokens/sec            │  ← "are users affected?"
├──────────────────────────────────────────────────────────┤
│ 6. Deploys & changes — annotate every deploy             │  ← "what changed?"
└──────────────────────────────────────────────────────────┘
```

**Deploy annotations are non-negotiable.** The first question in every incident is "what changed?" — the dashboard must answer it without leaving the page.

### 6.2 The alerting philosophy

| Principle | Rule |
|---|---|
| **Alert on symptoms, not causes** | Alert on "users are seeing errors", not "CPU is 80%" |
| **Every alert is actionable** | If there's nothing to do, it's a dashboard item, not an alert |
| **Every alert links to a runbook** | 100% coverage |
| **Every alert has a clear owner** | Routed to a team, not "everyone" |
| **Prefer burn-rate alerts to threshold alerts** | A static threshold either misses slow burns or fires on noise |
| **Two windows per SLO** | Fast burn (1 h / 5 min) = page. Slow burn (6 h / 30 min) = ticket |
| **Alerts must be testable** | Fire them deliberately in a game day |

### 6.3 Alert severity tiers

| Tier | Meaning | Response | Channel |
|---|---|---|---|
| **P1 / Page** | User-visible impact, or burn rate critical | Immediate, 24/7 | Pager |
| **P2 / Urgent** | Imminent risk or partial impact | Within business hours, or on-call if worsening | Pager or chat |
| **P3 / Ticket** | Degradation, no user impact yet | Next working day | Ticket |
| **P4 / Dashboard-only** | Informational trend | No action | Dashboard |

**Anti-pattern:** everything at P1. If everything pages, nothing pages — the on-call mutes the channel and the real outage is missed.

### 6.4 The alert catalogue (thresholds)

| Alert | Condition | Severity | Runbook says |
|---|---|---|---|
| **Availability burn (fast)** | Error budget burn rate > 14.4× over 1 h | P1 | Check recent deploys; roll back if correlated |
| **Availability burn (slow)** | Burn rate > 6× over 6 h | P3 | Investigate trend; check dependency latency |
| **Latency SLO breach** | p99 > target for 10 min | P1/P2 | Check DB, cache hit rate, downstream |
| **Saturation** | CPU/memory > 85% for 15 min | P2 | Scale out; check for a runaway query |
| **Queue depth growth** | Consumer lag rising for 15 min | P2 | Check consumer health; scale consumers |
| **DB connections** | > 85% of max | P1 | Check pool sizes, long-running queries |
| **Cache hit rate drop** | < 80% of baseline for 10 min | P2 | Check evictions, key distribution, node health |
| **Certificate expiry** | < 14 days | P3 | Renew |
| **Error-budget exhausted** | 100% burned in window | P1 | Enter reliability mode per policy |
| **Cost anomaly** | > 20% week-over-week | P3 | Check for runaway jobs, cardinality, retries |
| **AI: provider errors** | Provider 5xx/timeout > 2% | P1 | Failover to fallback model |
| **AI: token spend spike** | > 30% above baseline/hour | P2 | Check prompt regressions, retry loops |
| **AI: quality regression** | Eval score < baseline − threshold | P1 | Roll back prompt/model version |
| **AI: guardrail spike** | Guardrail blocks > 5× baseline | P2 | Investigate attack or a false-positive regression |

### 6.5 The runbook template

```markdown
# Runbook: <Alert name>
## What this means
<One paragraph: what the user experiences, why the alert fires>
## Impact
<Blast radius: which users, which functions>
## Immediate steps (first 5 minutes)
1. Open the dashboard: <link>
2. Check recent deploys: <link>
3. Check dependency health: <link>
## Diagnosis decision tree
- If <X> → do <A>
- If <Y> → do <B>
## Mitigation options (fastest first)
1. Roll back: <command/link>
2. Scale out: <command/link>
3. Shed load:<command/link>
4. Failover: <command/link>
## Escalation
- Primary on-call → Secondary → Service owner → Vendor support
## Verification
<How you know it's actually fixed>
## Post-incident
<Who writes the PIR, by when>
```

---

## 7. Failure modes of observability itself

| Failure mode | Symptom | Fix |
|---|---|---|
| **Cardinality explosion** | Metrics backend slow or down; bill spike | Label discipline; drop high-cardinality labels; cardinality limits |
| **Log volume explosion** | Storage cost spikes; queries time out | Sampling, level discipline, retention tiers |
| **Blind during an outage** | Observability goes down with the system | Separate the observability plane; buffer locally; don't depend on the thing you're observing |
| **Broken trace chain** | Traces stop at a service boundary | Context propagation audits |
| **Alert fatigue** | On-call mutes the channel | Cut alerts with no action; tier severity; tune thresholds |
| **Missing alerts** | Users report outages before monitoring does | Add synthetic checks and business-KPI alerts |
| **Alert storms** | One root cause fires 200 alerts | Grouping, deduplication, dependency suppression |
| **Stale dashboards** | Panels reference dead metrics | Dashboard ownership; review after architecture changes |
| **Sampling drops errors** | Failed requests have no trace | Tail-based sampling that keeps 100% of errors |
| **PII in logs** | Sensitive data in a log store | Redaction at the collector; log-scanning audits |
| **Time skew** | Traces and logs don't line up | NTP; monotonic clocks; store UTC |
| **Symptom blindness** | "Everything is green" while users suffer | Business-KPI and synthetic browser checks |
| **Shadow metrics** | Two sources of truth for the same number | Single definition; document ownership |
| **No deploy annotations** | "What changed?" is unanswerable | Auto-annotate from CI |
| **AI: no prompt/response capture** | Can't reproduce a bad output | Log prompts, versions, params with the trace |
| **AI: quality unmonitored** | Service green, outputs degrading | Production eval sampling + alerting |

---

## 8. AI/LLM observability — the additional layer

Classical observability is necessary but **insufficient**: an LLM service can be 100% available, fast, and returning confidently wrong answers.

### 8.1 What to capture per LLM request

| Attribute | Why |
|---|---|
| **Trace ID, session ID, user (hashed)** | Correlate across a multi-turn conversation |
| **Model name and version** | The equivalent of a deploy version |
| **Prompt template version** | Reproducibility — a prompt change is a deploy |
| **Input/output token counts** | Cost attribution and capacity |
| **Cost per request** | The unit economic metric |
| **Latency split: TTFT vs total** | Time-to-first-token is the UX metric for streaming |
| **Retrieved document IDs and scores** | Diagnoses bad answers as retrieval failures |
| **Guardrail decisions** (input and output) | Safety posture; false-positive detection |
| **Tool calls and arguments** | Agent debugging |
| **Evaluation score** where computed | Quality over time |
| **Finish reason** (stop, length, content filter) | Truncation and safety signals |

### 8.2 OpenTelemetry GenAI semantic conventions

Instrument LLM calls with the standardised attributes so traces are portable across tools:

| Attribute | Example |
|---|---|
| `gen_ai.system` | `anthropic` |
| `gen_ai.request.model` | model identifier |
| `gen_ai.request.max_tokens` | 1024 |
| `gen_ai.request.temperature` | 0.2 |
| `gen_ai.usage.input_tokens` | 1840 |
| `gen_ai.usage.output_tokens` | 312 |
| `gen_ai.response.finish_reasons` | `["stop"]` |

**Why it matters:** without a convention you get vendor-specific traces, and switching tools means re-instrumenting everything. Follow the standard and instrumentation becomes portable.

### 8.3 The AI dashboard

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Traffic — requests/min, tokens/min, by route and model   │
├─────────────────────────────────────────────────────────────┤
│ 2. Errors — provider errors, timeouts, rate limits,         │
│    guardrail blocks, schema-validation failures             │
├─────────────────────────────────────────────────────────────┤
│ 3. Latency — TTFT p50/p95, total latency p50/p95/p99        │
├─────────────────────────────────────────────────────────────┤
│ 4. Cost — spend/hour, cost/request, cost by route           │
├─────────────────────────────────────────────────────────────┤
│ 5. QUALITY — eval scores, groundedness, retrieval hit rate, │
│    no-answer rate, human-feedback score                     │
├─────────────────────────────────────────────────────────────┤
│ 6. Retrieval — recall proxy, index freshness, reranker lag  │
├─────────────────────────────────────────────────────────────┤
│ 7. Safety — guardrail hits by category, injection attempts  │
└─────────────────────────────────────────────────────────────┘
```

### 8.4 Agent observability

For agentic systems, a trace must show the **whole loop**, not just the final call:

```
Trace: "support-ticket-resolution"
├─ span: plan (model call, 1.2 s, 800 tokens)
├─ span: tool.fetch_order (DB query, 45 ms)
├─ span: tool.check_refund_policy (retrieval, 120 ms)
├─ span: reason (model call, 2.4 s, 1500 tokens)
├─ span: tool.issue_refund (write, 300 ms)  ← irreversible!
├─ span: verify (model call, 900 ms)
└─ span: respond (streamed, TTFT 400 ms)
   Total: 5.4 s · 2,600 tokens · $0.021 · 5 steps
```

**Agent-specific signals:** steps per task (rising = looping), tool-error rate per tool, step-budget exhaustion rate, human-handoff rate, task success rate, and cost per completed task (not per call).

---

## 9. Cost, retention and governance

| Control | Default | Notes |
|---|---|---|
| Metrics retention | 15–30 days raw; 1–13 months downsampled | Downsample aggressively |
| Log retention | 7–30 days hot; 90+ days cold | Most logs are never read |
| Trace retention | 7–14 days, sampled | Keep 100% of errors |
| AI trace retention | 30–90 days, with PII redaction | Consider a separate, access-controlled store |
| Sampling — traces | 1–10% head, 100% errors + slow | Tail-based |
| Error-budget-relevant data | **Never sampled** | You need every error to measure an SLO |
| Cardinality cap | < 10,000 series per metric | Enforce in the collector |
| Cost allocation | Per-team labels on all telemetry | Without this you can't charge back |
| PII redaction | At the collector, before storage | Never rely on a single control |

**The cost reality:** observability is typically **5–15% of infrastructure spend**, and log/trace volume is usually the fastest-growing line. Treat retention as a design decision, not a default.

---

## 10. Q&A

**Q1. What's the difference between monitoring and observability?**
Monitoring tells you *that* something is wrong — you pre-defined the signals and the thresholds. Observability lets you ask *why*, including questions you never anticipated, by capturing high-dimensional context you can slice after the fact. You need both; monitoring alone leaves you blind to novel failures.

**Q2. What are the three pillars?**
Metrics (is it healthy, how is it trending — cheap, low cardinality), logs (what exactly happened — expensive, high cardinality), and traces (where in the distributed request did time go — high cardinality, sampled). Profiles and AI-quality evaluations are increasingly treated as a fourth and fifth.

**Q3. What's the RED method? What's USE?**
RED is for services: Rate, Errors, Duration. USE is for resources: Utilization, Saturation, Errors. Use RED for every request-serving service and USE for every resource.

**Q4. Why alert on symptoms rather than causes?**
Because causes are many and symptoms are few. "CPU is 80%" might be fine; "users are seeing 500s" never is. Symptom-based alerting catches failure modes you never anticipated, and it removes the noise of causes that don't actually matter.

**Q5. What's a burn-rate alert and why is it better than a threshold?**
It measures how fast you're consuming the error budget relative to the rate that would exhaust it within the SLO window. Two windows — fast (1 h, pages) and slow (6 h, tickets) — catch both sudden outages and slow degradation. A static threshold either fires on harmless spikes or misses a slow burn entirely.

**Q6. Why is cardinality dangerous?**
Because every unique label combination is a separate time series. A `user_id` label on a request counter at 1M users creates 1M series — which costs a fortune and can take the metrics backend down, so your observability goes dark exactly during the incident you needed it for. Keep labels bounded and never put unbounded values in them.

**Q7. How do you sample traces without losing the important ones?**
Head sampling for volume, tail sampling for importance — keep 100% of errors and 100% of slow requests, sample the rest. Never sample anything that feeds an SLO, because you can't measure an error rate from a sampled subset of errors.

**Q8. What must appear on every service dashboard?**
SLO status and error budget first, then RED, then dependency health, then saturation, then the business KPI, and finally deploy annotations — because "what changed?" is the first question in every incident and the dashboard should answer it without leaving the page.

**Q9. What's the rule for whether something should be an alert?**
If there's no action to take, it's a dashboard, not an alert. Every alert must be actionable, have an owner, and link to a runbook. Alert fatigue is itself a cause of outages.

**Q10. How does observability change for an LLM service?**
Classical observability is necessary but insufficient, because an LLM service can be 100% available, fast, and returning confidently wrong answers. I'd add per-request capture of model and prompt version, token counts, cost, TTFT, retrieved document IDs and scores, guardrail decisions and eval scores — and I'd instrument with the OpenTelemetry GenAI semantic conventions so the data stays portable.

**Q11. What's different about observing an agent?**
The trace must show the whole loop, not just the final model call — each plan step, each tool call with its arguments, and the verify step. The agent-specific signals are steps per task (rising means looping), tool-error rate per tool, step-budget exhaustion, human-handoff rate, and **cost per completed task**, which is the metric that actually matters rather than cost per call.

**Q12. How do you observe quality, not just uptime?**
Sample production traffic continuously and score it — groundedness, relevance, task success, or an LLM-as-judge with human calibration on a subset. Track the score as a time series and alert on regression against a baseline. That's the only way to catch a prompt or model change that silently degrades output while every technical metric stays green.

**Q13. *Pushback:* "Observability costs too much."**
Cost is a design decision. Retention tiering, sampling, cardinality caps and redaction at the collector typically cut the bill by an order of magnitude without losing what you need. And the alternative — mean-time-to-diagnose in hours instead of minutes — costs far more in downtime.

**Q14. *Pushback:* "We only need logs."**
Logs alone can't answer "where in the request path did the time go" without manual correlation, and they can't give you a cheap, alertable time series. You'll end up grepping during incidents, which is slow and error-prone. Metrics for alerting, traces for diagnosis, logs for detail.

**Q15. What does good look like after an incident?**
The PIR can answer, from telemetry alone: what happened, when it started, what the blast radius was, which component was the cause, what changed, and how long detection and mitigation took. If any of those required someone's memory, the observability has a gap to close.

---

## 11. Cheat sheet

**Three pillars + two:** Metrics (is it healthy) · Logs (what happened) · Traces (where's the time) · Profiles (which code) · Evals (is the output good)

**RED (services):** Rate · Errors · Duration
**USE (resources):** Utilization · Saturation · Errors

**Alerting:** symptom-based · every alert actionable, owned, and runbook-linked · burn-rate over thresholds · fast window (1 h) pages, slow window (6 h) tickets · severity P1–P4, not everything at P1

**Alert thresholds**

| Signal | Threshold |
|---|---|
| Burn rate (fast) | > 14.4× / 1 h → P1 |
| Burn rate (slow) | > 6× / 6 h → P3 |
| p99 latency | breach for 10 min → P1/P2 |
| CPU/memory | > 85% / 15 min → P2 |
| DB connections | > 85% of max → P1 |
| Cache hit rate | < 80% of baseline → P2 |
| Cert expiry | < 14 days → P3 |
| Cost anomaly | > 20% WoW → P3 |
| **AI provider errors** | > 2% → P1 |
| **AI token spend** | > 30% above baseline → P2 |
| **AI quality score** | < baseline − threshold → **P1** |

**Cardinality:** never put unbounded values in labels · route templates not raw paths · < 10,000 series per metric · cardinality limits in the collector

**Retention:** metrics 15–30 d raw · logs 7–30 d hot · traces 7–14 d sampled · **never sample SLO data**

**Trace sampling:** head for volume, tail for importance · **100% of errors and slow requests**

**Dashboard order:** SLO → RED → dependencies → saturation → business KPI → deploy annotations

**AI capture per request:** model + prompt version · input/output tokens · cost · TTFT · retrieved doc IDs + scores · guardrail decisions · tool calls · eval score · finish reason

**OTel GenAI attributes:** `gen_ai.system` · `gen_ai.request.model` · `gen_ai.usage.input_tokens` · `gen_ai.usage.output_tokens` · `gen_ai.response.finish_reasons`

**Agent signals:** steps per task · tool-error rate per tool · budget exhaustion · human handoff · **cost per completed task**

**Rule:** if you can only answer questions you pre-built a dashboard for, you have monitoring, not observability.

---

## 12. Cross-links

- Previous: [Operational readiness checklist](01-operational-readiness-checklist.md)
- Next: [SLOs, error budgets & incident response](03-slos-error-budgets-and-incident-response.md)
- Related playbooks: [Deployment & release](04-deployment-release-and-cicd.md) · [Capacity planning](05-capacity-planning-and-autoscaling.md) · [Cost & FinOps](06-cost-and-finops.md) · [Reliability patterns](07-reliability-patterns.md) · [LLMOps](10-llmops-for-ai-systems.md)
- AI quality measurement: [RAG evaluation & observability](../HLD/05-ai-systems/03-rag-evaluation-and-observability.md)
- Foundations: [Availability numbers & SLOs](../HLD/01-foundations/06-availability-numbers-and-slos.md)
- Drill: [LLMOps & operationalisation questions](../Interview-Questions/06-llmops-and-operationalisation-questions.md) · [Cheat sheet](../Cheat-Sheets/06-operationalisation-cheatsheet.md)
