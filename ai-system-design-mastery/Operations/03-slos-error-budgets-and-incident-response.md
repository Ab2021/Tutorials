# SLOs, Error Budgets & Incident Response — Playbook

> **Purpose:** convert "reliability" from an argument into a measurable currency, then define exactly what happens when the currency runs out — and what everyone does during an incident.

> **The two ideas in one line each:**
> **Error budget:** the amount of unreliability you've decided to tolerate, spent as a currency that buys release velocity.
> **Incident response:** a rehearsed process that shortens MTTR — which, at scale, is the cheapest availability you can buy.

---

## 1. Purpose & scope

| In scope | Out of scope |
|---|---|
| Defining SLIs, SLOs and error budgets | Choosing the architecture (see foundations) |
| Burn-rate alerting | Alert delivery tooling (see [Observability](02-observability-and-monitoring.md)) |
| Error-budget policy and governance | Product roadmap |
| Incident severity, roles, lifecycle | Post-incident disciplinary process — it must stay blameless |
| Communications during incidents | Legal/PR crisis comms (escalation only) |
| Post-incident reviews and action tracking | — |
| On-call health and game days | — |

> **Foundations counterpart:** [Availability numbers & SLOs](../HLD/01-foundations/06-availability-numbers-and-slos.md) covers the *definitions and arithmetic* for design interviews. This file covers *running* SLOs in production.

---

## 2. SLI, SLO, SLA — the working definitions

| Term | Definition | Owner | Example |
|---|---|---|---|
| **SLI** | A precisely defined measurement of service behaviour, from the user's perspective | Service team | "Proportion of `GET /search` requests returning 200 in < 300 ms, measured at the load balancer" |
| **SLO** | The internal target for an SLI over a window | Service team + product | "≥ 99.9% over a rolling 30 days" |
| **SLA** | A contractual commitment with consequences | Legal + business | "≥ 99.5%, else service credits" |
| **Error budget** | `1 − SLO` — the permitted unreliability | The team | "0.1% of 30 days = 43.8 minutes" |

### The iron rule

> **SLO must be strictly stricter than SLA.** If the SLA is 99.5%, set the SLO at 99.9%. The gap is the warning zone — the region where you must act before you owe anyone money.

### Choosing a good SLI

| Property | Good SLI | Bad SLI |
|---|---|---|
| **User-centric** | "Successful page loads in < 1 s" | "CPU below 70%" |
| **Precisely defined** | Exact endpoint, status codes, and percentile | "API latency" |
| **Measured where the user is** | Load balancer, edge, synthetic probe | Deep inside one service |
| **Simple enough to explain** | One sentence | Needs a paragraph of caveats |
| **Has a natural denominator** | valid requests (excluding client 4xx) | all requests including bots |
| **Actionable** | The team can move it | Depends entirely on a vendor with no alternatives |

### The SLI catalogue

| Category | SLI | Typical SLO |
|---|---|---|
| **Availability** | success ratio | 99.9–99.99% |
| **Latency** | % of requests under threshold | 99% < 300 ms |
| **Throughput** | requests served per unit time | ≥ 95% of forecast peak |
| **Correctness** | % of responses that are semantically right | 99.9% |
| **Freshness** | % of reads within acceptable staleness | 99% < 5 s |
| **Durability** | % of writes surviving a period | 99.999999% |
| **Coverage** | % of requests actually served (not shed) | 99.9% |
| **AI quality** | eval/groundedness score vs baseline | ≥ baseline − 2% |
| **AI retrieval** | % of queries with a relevant result | ≥ 95% |

### Error budget windows

| Window | Best for | Trade-off |
|---|---|---|
| **Rolling 30 days** ✅ | Default for most services | Slow to forgive a bad day; smooth |
| Rolling 7 days | Fast-moving services | Jittery; punishes a single incident heavily |
| Calendar month | Reporting/compliance | Resets abruptly; gaming at month end |
| Rolling 90 days | Stable, slow-changing services | Very forgiving |

**Default: rolling 30 days.** It's long enough to be statistically meaningful and short enough to be actionable.

---

## 3. Architecture — where the SLI is measured

An SLO is only as honest as the point at which you measure it. The same service can be 99.99% or 98% depending on where the counter sits, and the difference is not pedantry — it is the difference between an alert that fires before your users notice and one that fires after.

```
   MEASUREMENT POINTS                                        PIPELINE
 ┌───────────────┐   ┌────────────────────┐   ┌───────────────────────┐   ┌──────────────────┐
 │ Client / RUM  │   │  Edge / CDN / LB   │   │  Service + sidecar    │   │  collector       │
 │ browser, app  │──▶│  5xx counted here  │──▶│  RED metrics, traces, │──▶│   → TSDB         │
 │ SDK, probe    │   │  TLS, routing      │   │  structured logs      │   │   → SLO evaluator│
 └───────────────┘   └────────────────────┘   └───────────────────────┘   └────────┬─────────┘
        ▲                                                                            │
        │  synthetic probes (independent of real traffic)                            │
        └────────────────────────────────────────────────────────────────────────────┤
                                                                                     ▼
                                                          ┌──────────────────────────────────┐
                                                          │ burn-rate alerts  → page / ticket │
                                                          │   14.4× fast · 6× slow · 1× ticket│
                                                          ├──────────────────────────────────┤
                                                          │ error-budget policy → RELEASE GATE│
                                                          │   budget exhausted ⇒ freeze risky │
                                                          │   changes until it recovers       │
                                                          └──────────────────────────────────┘
```

### The four measurement points, and what each one hides

| Where | What it captures | What it hides |
|---|---|---|
| **Client / RUM** | The user's actual experience — DNS, TLS, network, device, render | Cannot say *why* it was slow; absent for non-instrumented clients; ad-blockers and privacy modes silently drop beacons |
| **Edge / CDN / LB** | Every request that reached your infrastructure | Everything that never arrived — DNS failures, client-side errors, requests shed upstream of you |
| **Service (in-app RED)** | Per-endpoint rates, errors and durations, with attribution | Retries look like extra successful traffic and *inflate* your success rate; the load balancer's own failures are invisible |
| **Synthetic probes** | Availability measured independently of real traffic, on a schedule | Does not reflect the real user mix or volume; a probe on one path can miss a broken feature entirely |

**Choose one as primary and say which.** The common failure is quoting the most flattering number in a review and the least flattering one during an incident. Pick the point that matches the promise you are making: if the SLO is a user-facing promise, measure at the client or edge; if it is an internal contract between services, measure in-app and be explicit that it excludes upstream failure.

### The three decisions that make the number honest

1. **Do retries count as success?** They should not, at least not alone. A request that succeeds on the third attempt still made the user wait. Count the *first* attempt for the SLI, and track retries separately — a rising retry rate with a flat success rate is an outage forming before it shows in the SLO.
2. **What counts as "valid"?** Exclude 4xx the client caused (400, 401, 404 from bad input) — but *include* timeouts and 5xx, and include 429s, because a rate-limited user did not get served. Decide this once, write it down, and keep it identical between the SLO, the alert and the dashboard, or the three will disagree during an incident.
3. **Is the denominator requests or time?** Request-based is the default for request-serving systems because a low-traffic hour should not cost the same budget as a peak hour. Time-based is the right choice for a *binary* capability — "was the service reachable" — where the question is genuinely about minutes.

### The release gate — what makes an error budget real

A dashboard that nobody acts on is not an error budget. The mechanism only does work if it is wired to a decision:

```
 budget remaining > 50%   → normal releases
 budget remaining 0–50%   → releases allowed; reliability work scheduled
 budget exhausted         → feature releases frozen; only fixes and reliability work ship
                            (and the exception path requires a named approver, in writing)
```

Wire it to the CI/CD pipeline rather than to a wiki page: a release check that reads the current budget and requires an override with a reason. That is the entire difference between "we track SLOs" and "we use SLOs" — and it is the part most candidates omit when asked how they would operate a system.

---

## 4. Error budgets — the mechanism

### The arithmetic

```
Error budget = (1 − SLO) × window

99.9%  over 30 days = 0.1% × 43,200 min = 43.8 minutes
99.95% over 30 days = 0.05% × 43,200 min = 21.6 minutes
99.99% over 30 days = 0.01% × 43,200 min = 4.3 minutes
```

**The request-based form is usually better:**

```
Budget = (1 − SLO) × total valid requests in window
99.9% of 100M requests = 100,000 failed requests permitted
```

### Why request-based beats time-based

| | Time-based | Request-based |
|---|---|---|
| Brief outage at peak | Same cost as at 3 a.m. | **Costs more** — correctly, since more users were hurt |
| Definition | "Is it up?" | "What fraction of users succeeded?" |
| Partial failure | Awkward — is a shard down "down"? | Natural — it's just a ratio |
| Gaming | Redefine "up" | Harder |

### Burn rate

**Burn rate = how fast you're spending the budget relative to the rate that exhausts it in the window.**

```
Burn rate of 1    → budget exhausts exactly at the end of the window (exactly on target)
Burn rate of 14.4 → budget exhausts in 1/14.4 of the window
```

**The multi-window, multi-burn-rate table (the standard pattern):**

| Burn rate | Long window | Short window | Budget consumed | Action | Severity |
|---|---|---|---|---|---|
| **14.4×** | 1 hour | 5 minutes | 2% | **Page** | P1 |
| **6×** | 6 hours | 30 minutes | 5% | **Page** | P1/P2 |
| **1×** | 3 days | 6 hours | 10% | **Ticket** | P3 |
| **0.5×** | 7 days | 12 hours | — | Ticket / review | P4 |

**Why two windows:** the long window confirms it's a real trend; the short window confirms it's still happening *now*. Alerting on the long window alone keeps firing after recovery; the short window alone fires on transient noise.

### Error-budget policy — write it before you need it

```markdown
# Error Budget Policy — <service>

## Budget
SLO: 99.9% availability, rolling 30 days.
Budget: 43.8 minutes / 100,000 failed requests per window.

## When budget remains (healthy)
- Normal release cadence.
- Risky changes allowed with canary + rollback ready.

## When > 50% of budget is consumed in a window
- Review in the weekly reliability sync.
- Risky changes require an explicit rollback plan.

## When the budget is exhausted (< 0%)
- CHANGE FREEZE except for reliability fixes and security patches.
- All engineering effort shifts to reliability until the budget recovers above 25%.
- Post-incident reviews completed for every incident in the window.
- Service owner reports weekly to engineering leadership.

## Exceptions
- Emergency security patches and customer-critical fixes are exempt.
- Any exemption is recorded with an owner and an expiry date.
```

> **The point of the policy:** it removes the argument. When the budget is gone, the freeze is automatic — not a negotiation between product and engineering at the worst possible moment.

### Anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| SLO with no policy | Budget is measured and then ignored |
| Budget never spent | SLO is too loose; you're buying reliability you don't need |
| Budget always spent | SLO is too strict, or the service needs investment — decide which, explicitly |
| 99.99% on a fast-moving service | Freezes releases permanently; either loosen it or invest in canaries |
| SLO defined on backend metrics only | Users can suffer while every backend number is green |
| Per-service SLOs, no user journey | Ten green services can still deliver a broken journey |

---

## 5. Incident response

### 4.1 Severity definitions

| Sev | Definition | Response | Comms |
|---|---|---|---|
| **Sev-1** | Major user-facing outage, data loss/corruption risk, or security breach | Immediate, all hands, exec notified | Status page + customer comms |
| **Sev-2** | Significant degradation, or a critical function broken for some users | Immediate, on-call + owner | Status page if user-visible |
| **Sev-3** | Minor degradation, workaround exists | Business hours | Internal |
| **Sev-4** | No user impact; risk or hygiene issue | Next sprint | Internal |

**The classification rule:** classify by **user impact**, not by technical drama. A spectacular-looking failure with no user impact is a Sev-3. A quiet wrong-answer bug affecting payments is a Sev-1.

### 4.2 Incident roles

| Role | Responsibility | Notes |
|---|---|---|
| **Incident Commander (IC)** | Owns the response; makes decisions; does **not** debug | The single most important role — without it, everyone debugs and nobody decides |
| **Operations Lead** | Directs the technical work; hands on keyboard | Can be the IC in small orgs, but separation is better |
| **Communications Lead** | Status page, stakeholder updates, customer comms | Frees the IC from comms |
| **Scribe** | Timestamped timeline of actions and decisions | Invaluable for the PIR |
| **Subject Matter Experts** | Debug and fix their domain | Time-boxed; rotate out to avoid burnout |

> **The cardinal rule:** **the IC does not debug.** The moment the IC opens a terminal, the response loses coordination. In a small team one person may hold both roles, but name it out loud.

### 4.3 The incident lifecycle

```
  DETECT ──▶ TRIAGE ──▶ MITIGATE ──▶ RESOLVE ──▶ REVIEW ──▶ PREVENT
    │          │            │            │          │          │
  alerts,    severity,   stop the     confirm    blameless   actions
  synthetic, assess      bleeding     recovery   PIR         tracked
  customers  impact      FIRST                    (5 days)    to done
```

**Mitigate before you diagnose.** Restore service first, understand second. Roll back, fail over, scale out, shed load, disable a feature — any of these is valid before you know the cause.

### 4.4 The first five minutes — the checklist

```
1. ACK the page.                       (Never leave an alert unacknowledged.)
2. Declare: "I am IC" (or find one).   (Announce the role out loud.)
3. Classify severity by user impact.
4. Open the dashboard.                 (SLO → RED → dependencies → saturation.)
5. Check "what changed."               (Deploys, config, feature flags, infra.)
6. Start a timeline doc + comms channel.
7. Decide the first mitigation.        (Roll back is the default if a deploy correlates.)
8. Communicate: status page if user-visible.
```

### 4.5 Mitigation ladder — ordered by speed

| # | Mitigation | Time | Reversible | When |
|---|---|---|---|---|
| 1 | **Roll back the last deploy** | Minutes | Yes | Any correlation with a recent change |
| 2 | **Disable a feature flag** | Seconds | Yes | The fault is isolated to one feature |
| 3 | **Fail over** (AZ/region/DB) | Seconds–minutes | Costly | Infrastructure failure |
| 4 | **Scale out** | Minutes | Yes | Saturation is the cause |
| 5 | **Shed load / rate limit harder** | Seconds | Yes | Overload; protect the core |
| 6 | **Degrade gracefully** | Minutes | Yes | Serve a reduced experience rather than nothing |
| 7 | **Drain / take a node out of rotation** | Seconds | Yes | One bad node |
| 8 | **Restart** | Minutes | Partly | Suspected leak or stuck state |
| 9 | **Restore from backup** | Hours | **No** | Data loss — last resort, highest authority needed |

> **The default instinct should be rollback.** Most outages are change-induced, and rollback is the cheapest, fastest, most reversible action available. Make rollback a one-click operation *before* you need it.

### 4.6 Communication standards

| Audience | Cadence | Content |
|---|---|---|
| Status page | Within 15 min, then every 30 min | Impact, affected functions, workaround, next update time |
| Internal stakeholders | Every 30 min | The same, plus an ETA if known |
| Customer support | At declaration | What to tell users who ask |
| Executives | Sev-1 only, at declaration and resolution | Impact, customer count, revenue exposure |

**Rules:** never speculate publicly about cause; never promise an ETA you don't believe; always give the *next update time* even if there's no news — silence is read as incompetence.

### 4.7 Recovery and closure

An incident is not resolved when the symptoms stop. It's resolved when:

- [ ] Metrics are back within SLO, sustained for **at least 30 minutes**
- [ ] Any manual mitigations are reverted (or consciously kept, with an owner)
- [ ] Backlogged work is drained (queues, retries, delayed jobs)
- [ ] Data integrity is verified after any write-path incident
- [ ] Dependent teams are informed
- [ ] The timeline is complete and the PIR is scheduled

---

## 6. Post-incident reviews

### Principles

| Principle | Meaning |
|---|---|
| **Blameless** | The goal is system improvement, not individual fault. People don't cause outages — conditions do |
| **Timely** | Held within **5 business days**, before memory fades |
| **Specific** | "Add monitoring" is not an action; "alert when consumer lag > 10k for 5 min" is |
| **Actioned** | Every action has an owner and a date, tracked like any other work |
| **Shared** | Published org-wide so others learn the same lesson |

### The PIR template

```markdown
# Post-Incident Review: <title>  (<date>, Sev-N)

## Summary
<2–3 sentences: what happened, who was affected, for how long.>

## Impact
- Users affected: <number / %>
- Functions affected: <list>
- Revenue: <estimate or "none">
- Error budget consumed: <minutes / %>

## Timeline (all times UTC)
| Time | Event |
|------|-------|
| 14:02 | Deploy of service v2.14.0 completed |
| 14:07 | Alert: availability burn rate > 14.4x |
| 14:09 | IC declared; Sev-2 assigned |
| 14:14 | Rollback initiated |
| 14:19 | Error rate normal |
| 14:49 | Incident resolved after 30 min of stable metrics |

## Detection
- How detected: <alert / synthetic / customer report>
- Time to detect: <N min>
- **If a customer reported it first, this is finding #1.**

## Root cause
<The technical cause, stated as a system condition, not a person.>

## Contributing factors
<The conditions that made it possible or worse. Usually 2–4.>

## What went well
<Genuinely useful — and it makes the review safe.>

## What went poorly
<Honest. Include the things that were luck, not skill.>

## Action items
| # | Action | Type | Owner | Due | Ticket |
|---|--------|------|-------|-----|--------|
| 1 | Add alert on consumer lag | Detect | @a | date | LINK |
| 2 | Add timeout on outbound call | Prevent | @b | date | LINK |
| 3 | Add rollback drill to quarterly calendar | Respond | @c | date | LINK |

## Lessons for other teams
<What applies beyond this service.>
```

### Action types — balance all four

| Type | Purpose | Example |
|---|---|---|
| **Detect** | Find it faster | Add an alert, add a synthetic check |
| **Prevent** | Make it not happen | Add a timeout, a test, a constraint |
| **Mitigate** | Make it smaller next time | Circuit breaker, blast-radius limiting, canary |
| **Respond** | Recover faster | Runbook update, one-click rollback, drill |

> **The common mistake:** every action is "prevent". But you cannot prevent everything, and at scale **detection and response improvements buy more availability per hour of effort than prevention does.** Balance them roughly equally.

---

## 7. On-call health and drills

| Practice | Why |
|---|---|
| **Primary + secondary rotation** | Never single on-call; it's a burnout and coverage risk |
| **Max 1 week on-call, with a week off** | Sustained alertness degrades |
| **Alert volume target: ≤ 2 actionable pages per 12-hour shift** | More than that is unsustainable |
| **Handover document each rotation** | Ongoing risks, fragile areas, in-flight work |
| **Runbook for every alert** | No thinking required at 3 a.m. |
| **Compensation or time-in-lieu** | On-call is work |
| **No-penalty escalation** | Escalating has no cost; failing to escalate does |
| **Game days / chaos drills** | Test detection and response *before* production does |
| **Quarterly DR drill** | Restore and failover, measured, timed |
| **Rotation health review** | Ask the humans how it's going |

### The drill calendar

| Drill | Cadence | Measured outcome |
|---|---|---|
| Rollback | Monthly | Time to roll back (< 5 min) |
| Restore from backup | Quarterly | Time to restore, data loss window |
| Failover (DB / AZ / region) | Quarterly | Time to failover, error spike during |
| Chaos experiment | Monthly | Was it detected? In how long? |
| Game day (full scenario) | Quarterly | MTTR vs last quarter |
| Alert validation | Monthly | Every alert fires when it should |

---

## 8. Failure modes of this whole system

| Failure mode | Symptom | Fix |
|---|---|---|
| **SLO with no policy** | Budget measured, ignored | Write the policy; get product sign-off |
| **Aspirational SLO** | Constant freeze | Derive the target from the cost of downtime |
| **Vanity SLI** | Green while users suffer | Measure at the user boundary; add synthetic and business KPIs |
| **Alert fatigue** | On-call mutes the channel | Cut non-actionable alerts; two-window burn rates |
| **No IC** | Everyone debugs, nobody decides | Declare the role out loud, every time |
| **IC debugs** | Coordination collapses | Enforce separation; name the rule |
| **Diagnosis before mitigation** | Outage lengthens | Mitigate first; roll back on correlation |
| **Untested rollback** | It doesn't work when needed | Drill monthly |
| **Blame in the PIR** | Information is withheld next time | Enforce blamelessness; name it in the charter |
| **"Add monitoring" actions** | No measurable change | Require a specific, testable action with an owner |
| **All actions are prevention** | Slow detection and recovery persist | Balance detect/mitigate/respond |
| **Actions not tracked** | The same incident recurs | Track in the normal backlog, with dates |
| **Sev inflation** | Everything is Sev-1 | Classify strictly by user impact |
| **Status-page silence** | Trust collapses faster than the service | Give the next update time even with no news |
| **Per-service SLOs only** | Journeys break while all services are green | Add user-journey SLOs |
| **AI quality SLO missing** | Service green, answers wrong | Evaluate quality as a first-class SLI |

---

## 9. AI/LLM specifics

| Classical | AI/LLM addition |
|---|---|
| Availability SLI | **Quality SLI** — eval/groundedness score vs baseline |
| Error rate | **Guardrail block rate, schema-validation failures, provider errors** |
| Latency p99 | **TTFT p95 and total latency** (streaming changes the UX metric) |
| Saturation | **Token throughput, provider quota utilization, GPU utilization** |
| Cost per request | **Cost per completed task** (agents) |
| Deploy rollback | **Prompt, model and index version rollback** |
| Correctness | **Hallucination/groundedness, retrieval hit rate, citation accuracy** |
| Data freshness | **Index freshness lag** |

### The AI incident types

| Incident | Looks like | Mitigation |
|---|---|---|
| **Provider outage** | Errors spike, latency spikes | Failover to secondary model; queue and replay |
| **Quality regression** | Service green, outputs worse | Roll back the prompt or model version |
| **Cost runaway** | Spend spikes 10× | Kill retry loops; enforce budgets; rate limit |
| **Prompt injection** | Guardrail hits spike; data exfiltration attempt | Block patterns; tighten tool scopes; human review |
| **Retrieval failure** | "I don't know" rate spikes | Check index health, embedding version, ACL filters |
| **Agent loop** | Steps per task climb, cost climbs | Enforce step/time/token budgets; hard stop |
| **Silent data leak** | Wrong user's data in output | Kill the path; verify ACL at retrieval; notify per policy |

> **The key point:** for AI systems, **"is it up?" and "is it working?" are different questions.** A quality regression can be a Sev-1 with every classical metric green — which is why a quality SLI must exist before you can detect it.

---

## 10. Numbers, defaults & thresholds

| Parameter | Default |
|---|---|
| SLO window | Rolling 30 days |
| SLO vs SLA buffer | ≥ 0.05% |
| Consumer availability target | 99.9% |
| Critical-path target | 99.99% |
| Internal-tool target | 99% or none |
| Burn-rate fast window | 1 h / 5 min at **14.4×** → page |
| Burn-rate slow window | 6 h / 30 min at **6×** → page/ticket |
| Ticket-level burn | 3 d / 6 h at **1×** |
| 99.9% / 30 days budget | **43.8 minutes** |
| PIR deadline | **5 business days** |
| Rollback target | **< 5 minutes** |
| Recovery confirmation | 30 min of stable metrics |
| Status-page first update | Within 15 min |
| Status-page cadence | Every 30 min |
| Actionable pages per 12 h shift | ≤ 2 |
| On-call rotation | 1 week on / 1 week off, primary + secondary |
| Restore drill | Quarterly, timed |
| Failover drill | Quarterly, timed |
| Chaos/game day | Monthly / quarterly |

---

## 11. Q&A

**Q1. What is an SLO and why not just use an SLA?**
An SLO is an internal target for a measured indicator; an SLA is a contract with penalties. You need both, and the SLO must be stricter so you get a warning zone before you breach the contract and owe money.

**Q2. What is an error budget?**
`1 − SLO`, expressed as permitted downtime or permitted failed requests — 0.1% of 30 days is 43.8 minutes. It converts reliability into a currency that buys release velocity, which is what stops reliability and feature work from being a perpetual argument.

**Q3. Why prefer request-based SLIs to time-based ones?**
Because a brief outage at peak traffic hurts far more users than the same outage at 3 a.m., and request-based measurement reflects that automatically. It also handles partial failure naturally — one shard down is a ratio, not a philosophical question about whether the system is "up".

**Q4. What's a burn rate?**
How fast you're consuming the error budget relative to the rate that would exhaust it exactly at the end of the window. A burn rate of 1 exhausts the budget at the window boundary; 14.4× exhausts it in about 1/14th of the window.

**Q5. Why use two windows for burn-rate alerts?**
The long window confirms a real trend and prevents noise-driven pages; the short window confirms it's still happening, so the alert clears when the problem does. Alerting on the long window alone keeps firing after recovery; the short window alone fires on transient blips.

**Q6. What happens when the error budget is exhausted?**
Per the written policy: a change freeze except for reliability fixes and security patches, with engineering effort redirected to reliability until the budget recovers. The value is that it's automatic and pre-agreed — not a negotiation during an outage.

**Q7. What's the most important role in an incident?**
The Incident Commander — and the rule that matters is that the IC does not debug. The moment the IC is hands-on-keyboard, coordination collapses. In a small team one person may hold both roles, but the role must be named out loud.

**Q8. What's the first thing you do in an incident?**
Acknowledge and declare, classify by user impact, check what changed — and then **mitigate before you diagnose.** Rollback if there's any correlation with a recent change. You restore service first and understand it afterwards.

**Q9. Why is rollback the default mitigation?**
Because most outages are change-induced, and rollback is the fastest, cheapest, most reversible action available. That's why it must be a one-click, tested operation *before* you need it — an untested rollback is a hope, not a plan.

**Q10. What makes a post-incident review good?**
Blamelessness, timeliness (within five business days), specific and testable actions with owners and dates, and a balance across detect, prevent, mitigate and respond. Reviews where every action is "prevent" miss the fact that at scale, faster detection and faster recovery buy more availability per hour than prevention does.

**Q11. *Pushback:* "Error budgets just give engineers an excuse to stop shipping."**
The opposite — the budget is what *lets* you ship. With a healthy budget you ship faster with less ceremony; the freeze only happens when reliability has genuinely degraded. Without a budget, every release is an argument about risk instead of a measurement.

**Q12. *Pushback:* "We're too small for formal SLIs."**
Start with one SLI on the one user journey that matters, one SLO, and a burn-rate alert. That's a few hours of work and it will find problems nothing else surfaces. Formality scales later; the measurement matters from day one.

**Q13. How do you set an SLO target?**
From the cost of downtime, not from aspiration. If an hour of downtime costs £X, buy the next nine only when the downtime avoided is worth more than the yearly cost of achieving it. Then check that the architecture can actually deliver it — compute the series product of your dependencies first.

**Q14. How does incident response differ for an AI system?**
The process is the same, but the incident types differ: provider outage (fail over to a secondary model), quality regression (roll back the prompt or model version), cost runaway (kill retry loops, enforce budgets), prompt injection, retrieval failure, agent loops, and data leakage through the model. The critical addition is a **quality SLI**, because a quality regression can be a Sev-1 while every classical metric is green.

**Q15. *Pushback:* "Can't we just alert when the service is down?"**
That's the alert that fires *after* your users already know. Burn-rate alerting fires while the budget is being consumed, not once it's gone — and quality alerting catches the failures that never show up as downtime at all.

---

## 12. Cheat sheet

**Definitions:** SLI = measured · SLO = internal target · SLA = contract · **SLO stricter than SLA**

**Budget:`(1 − SLO) × window`**

| SLO | Budget / 30 days |
|---|---|
| 99% | 7.3 hours |
| 99.9% | 43.8 minutes |
| 99.95% | 21.6 minutes |
| 99.99% | 4.3 minutes |

**Burn-rate alerting**

| Burn | Long / Short | Consumed | Action |
|---|---|---|---|
| 14.4× | 1 h / 5 min | 2% | **Page** |
| 6× | 6 h / 30 min | 5% | **Page/ticket** |
| 1× | 3 d / 6 h | 10% | Ticket |

**Incident roles:** IC (does **not** debug) · Ops Lead · Comms Lead · Scribe · SMEs

**First five minutes:** ACK → declare IC → classify severity → open dashboard → check what changed → start timeline → pick a mitigation → communicate

**Mitigation ladder (fastest first):** rollback → flag off → failover → scale out → shed load → degrade → drain → restart → restore

**Sev by user impact:** Sev-1 major outage / data loss / breach · Sev-2 significant degradation · Sev-3 minor with workaround · Sev-4 no user impact

**PIR:** blameless · within 5 business days · specific testable actions · balanced across **detect / prevent / mitigate / respond** · tracked with owners and dates

**Key numbers:** 30-day window · PIR in 5 days · rollback < 5 min · 30 min stable to declare resolved · status page within 15 min then every 30 · ≤ 2 pages per 12 h shift · primary + secondary on-call

**AI additions:** quality SLI (eval/groundedness) · provider failover · prompt/model/index version rollback · token & cost budgets · step budgets for agents · cost per completed task

**The one-liner:** *"Is it up?" and "is it working?" are different questions — instrument both.*

---

## 13. Cross-links

- Previous: [Observability & monitoring](02-observability-and-monitoring.md) · [Operational readiness](01-operational-readiness-checklist.md)
- Foundations: [Availability numbers & SLOs](../HLD/01-foundations/06-availability-numbers-and-slos.md) · [Capacity estimation](../HLD/01-foundations/05-capacity-estimation-playbook.md)
- Related playbooks: [Reliability patterns](07-reliability-patterns.md) · [DR, failover & backup](09-dr-failover-and-backup.md) · [Deployment & release](04-deployment-release-and-cicd.md) · [Cost & FinOps](06-cost-and-finops.md) · [Security, guardrails & compliance](08-security-guardrails-and-compliance.md) · [LLMOps](10-llmops-for-ai-systems.md)
- AI quality: [RAG evaluation & observability](../HLD/05-ai-systems/03-rag-evaluation-and-observability.md) · [Guardrails & safety](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md)
- Drill: [LLMOps & operationalisation questions](../Interview-Questions/06-llmops-and-operationalisation-questions.md) · [Behavioural & project questions](../Interview-Questions/07-behavioral-and-project-questions.md) · [Cheat sheet](../Cheat-Sheets/06-operationalisation-cheatsheet.md)
