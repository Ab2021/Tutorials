# Availability Numbers & SLOs — The Nines, the Math, and What They Cost

> **Source of truth:** *System Design Interview Prep | Day 1 — Foundations, Framework & Capacity Estimation* (Section on availability, MTBF/MTTR and reliability vs availability) plus the availability anchor table used throughout the Day 1 estimation section.

> **Companion file:** this is the *definitions and arithmetic* of availability for the design interview. For running SLOs in production — burn rates, alerting policy, error-budget governance and incident response — see [SLOs, error budgets & incident response](../../Operations/03-slos-error-budgets-and-incident-response.md).

---

## TL;DR

**Availability is `MTBF ÷ (MTBF + MTTR)`** — the fraction of time a system is serving. It's expressed in "nines," and each nine costs roughly an order of magnitude more money and engineering effort than the last: 99% is 3.65 days of downtime a year, 99.9% is 8.77 hours, 99.99% is 52.6 minutes, and 99.999% is 5.26 minutes. The two levers are **make failures rarer** (raise MTBF — redundancy, testing, hardware quality) and **make recovery faster** (lower MTTR — automation, canaries, runbooks). At scale, **MTTR reduction is usually the cheaper lever**, because you cannot eliminate failure but you can shrink it. Availability is also *not* reliability: availability is "can I reach it now," reliability is "does it do the right thing consistently over time." And availability composes dangerously — components in series multiply, components in parallel improve, so a request touching ten 99.9% services in series is 99%, not 99.9%.

> **60-second interview answer:** "Availability is mean time between failures divided by mean time between failures plus mean time to repair. I express it in nines — 99.9% allows 8.77 hours of downtime a year, 99.99% allows 52.6 minutes, and each additional nine is roughly an order of magnitude more cost. There are two levers: raise MTBF with redundancy and testing, or lower MTTR with automation and canaries — at scale, lowering MTTR is usually cheaper because you can't eliminate failure, only shrink it. Crucially, availability composes: components in series multiply, so a request fanning out to ten 99.9% services is 99%, not 99.9%. And I distinguish availability from reliability — reachable now versus correct consistently over time."

---

## 1. Definition & mental model

### The core formula

```
                 MTBF
Availability = ─────────────────
               MTBF + MTTR
```

| Term | Meaning | How you improve it |
|---|---|---|
| **MTBF** — Mean Time Between Failures | How long the system runs before something breaks | Redundancy, better hardware, testing, change control, reducing complexity |
| **MTTR** — Mean Time To Repair | How long it takes to restore service once broken | Automation, health checks, failover, runbooks, canaries, observability |

Read the formula as a **ratio of good time to total time**. A system that runs 99 hours then takes 1 hour to fix has availability 99/100 = 99%.

### The two-lever mental model

```
   FAILURE HAPPENS
         │
    ┌────┴────┐
    ▼         ▼
  RAISE     LOWER
  MTBF      MTTR
  (rarer)   (shorter)
    │         │
  Redundancy, Automation,
  quality,    failover,
  testing,    observability,
  simpler     runbooks,
  systems     canaries
```

Think of availability as **how much of the year your shop's lights are on**. You can (a) stop the bulbs from burning out, or (b) learn to swap a bulb in ten seconds instead of an hour. Option (b) is nearly always more effective per dollar, because bulbs *will* burn out.

### Availability vs reliability — the distinction that gets asked

| | **Availability** | **Reliability** |
|---|---|---|
| Question answered | *Can I reach it right now?* | *Does it do the right thing, consistently, over time?* |
| Time orientation | Instantaneous | Over a period |
| Measured by | Uptime %, MTBF/MTTR | Failure rate, error rate, correctness over time |
| Example failure | Service returns 503 | Service returns 200 with **wrong data** |
| Can you have one without the other? | **Yes, both ways** | |

**The two asymmetric cases — know both:**

| Scenario | Available? | Reliable? | Why |
|---|---|---|---|
| Serving 200 OK with **stale or corrupted data** | ✅ Yes | ❌ No | Reachable, but wrong |
| Service is **down but has never served a wrong answer** | ❌ No | ✅ Yes | Correct, but unreachable |

This is why "the site is up" and "the site is correct" are separate SLOs. A cache serving stale prices is 100% available and 0% reliable for a pricing system.

### The nines table — memorise this

| Availability | "Nines" | Downtime/year | Downtime/month | Downtime/day |
|---|---|---|---|---|
| 90% | one nine | **36.5 days** | 3 days | 2.4 hours |
| 99% | two nines | **3.65 days** | 7.3 hours | 14.4 min |
| 99.9% | three nines | **8.77 hours** | 43.8 min | 1.44 min |
| 99.99% | four nines | **52.6 minutes** | ~26 sec | 8.6 sec |
| 99.999% | five nines | **5.26 minutes** | ~2.6 sec | 0.86 sec |
| 99.9999% | six nines | **31.5 seconds** | ~0.26 sec | 86 ms |

### The two facts that make this table matter

1. **Each nine is roughly 10× harder and more expensive than the previous.** Going from 99.9% to 99.99% is not a 0.09% improvement — it's a 10× reduction in allowed downtime, and it demands automation, multi-AZ, tested failover and often multi-region.
2. **You cannot achieve nines you haven't architected for.** 99.99% (52.6 min/year) rules out any process that requires a human to notice, page, log in and act — each of those steps takes minutes. Four nines is an *automation* target, not a diligence target.

---

## 2. Why it exists — the problem it solves

### What happens without an availability target

| Without a target | Consequence |
|---|---|
| You can't decide how much redundancy to buy | You either over-build (waste) or under-build (outages) |
| You can't answer "is this incident bad?" | Every incident feels equally urgent; alert fatigue |
| You can't have a cost conversation | No way to tie spend to a business outcome |
| You can't say no to risky changes | No error budget to spend |
| You can't compare designs | "Multi-AZ vs multi-region" is unresolvable without a target |

### The cost curve — why "just make it five nines" is not an answer

```
Cost
 ▲
 │                                        ╱ 99.999%
 │                                    ╱
 │                             ╱  99.99%
 │                      ╱
 │              ╱  99.9%
 │       ╱
 │  ╱  99%
 └──────────────────────────────────────────▶ Availability
```

Each step right costs roughly an order of magnitude more in infrastructure, engineering time and operational complexity. **A 99.99% target for an internal admin dashboard is malpractice** — and a 99% target for a payment rail is negligence. The target must come from the business impact of downtime, not from aspiration.

### The framework: SLA vs SLO vs SLI

| Term | Stands for | Definition | Example | Consequence of breach |
|---|---|---|---|---|
| **SLI** | Service Level **Indicator** | The raw measurement | "Fraction of requests with latency < 200 ms" | None — it's a number |
| **SLO** | Service Level **Objective** | Your internal target for the SLI | "99.9% of requests < 200 ms over 30 days" | Error budget burns; action required |
| **SLA** | Service Level **Agreement** | The *contract* with the customer, with penalties | "99.9% uptime or 10% service credit" | Money, legal liability |

**The relationship, in one line:** SLI is what you measure, SLO is what you promise yourself, SLA is what you promise the customer — and **your SLO must always be stricter than your SLA**, because you need margin to detect and fix a problem before it becomes a contractual breach.

```
   SLI ──measured──▶ SLO ──must be stricter than──▶ SLA
 (the number)     (internal target)                (customer contract)
```

**The rule:** if the SLA is 99.9%, set the internal SLO at 99.95%. The gap is your early-warning buffer.

---

## 3. Internal mechanics

## 3.1 Where MTBF and MTTR come from

**MTBF** is a property of the *component and the process*:

| Driver of MTBF | Effect |
|---|---|
| Hardware quality | Fewer physical failures |
| Redundancy (N+1, N+2) | A component failure is not a *system* failure |
| Testing and canaries | Catches bad changes before they reach everyone |
| Complexity | **More moving parts = more failure modes** — the biggest MTBF lever is often deletion |
| Change volume | Most outages are change-induced; smaller, more frequent changes are safer |

**MTTR** is a property of the *operations*:

| Driver of MTTR | Effect |
|---|---|
| Automated failover | Seconds instead of minutes |
| Health checks and readiness probes | Detect and remove bad instances automatically |
| Observability | Diagnose in minutes, not hours |
| Runbooks | No thinking required at 3 a.m. |
| Rollback capability | Undo instead of fix-forward |
| Blast-radius limits (canary, cell-based) | Smaller incidents by construction |

### The MTTR insight worth saying out loud

> *"At scale I'd invest more in MTTR than MTBF, because failures are inevitable — hardware fails, dependencies fail, and every deploy is a risk. You can't make failure rare enough to reach four nines; you can make recovery fast enough."*

This is the correct senior instinct and it is a strong interview moment.

### Illustrative arithmetic

| MTBF | MTTR | Availability | Nines |
|---|---|---|---|
| 1,000 h | 10 h | 99.0% | two nines |
| 1,000 h | 1 h | 99.9% | three nines |
| 1,000 h | 6 min | 99.99% | four nines |
| 10,000 h | 6 min | 99.999% | five nines |
| 100 h | 1 h | 99.0% | two nines |
| **10,000 h** | **36 s** | **99.9999%** | six nines |

**Read the table's lesson:** to reach four nines with an MTBF of 1,000 hours (roughly a failure every six weeks), your MTTR must be **six minutes** — which means fully automated detection and failover. A human in the loop cannot do it.

---

## 3.2 How availability composes — the part candidates get wrong

### Components in series (all must work)

```
A ──▶ B ──▶ C
A = 99.9% each
Total = 0.999 × 0.999 × 0.999 = 99.7%
```

**Availability multiplies downward.** Three 99.9% services in a request path give 99.7% — almost **9 hours** of downtime a year instead of 8.77.

### The fan-out amplification table — why microservices hurt availability

| Services in the critical path | Each at 99.9% | Resulting availability | Downtime/year |
|---|---|---|---|
| 1 | 99.9% | 99.9% | 8.77 h |
| 3 | 99.9% | 99.7% | 26.3 h |
| 5 | 99.9% | **99.5%** | 43.8 h |
| 10 | 99.9% | **99.0%** | 3.65 days |
| 20 | 99.9% | **98.0%** | 7.3 days |
| 50 | 99.9% | **95.1%** | 17.9 days |
| 10 | 99.99% | 99.9% | 8.77 h |
| 20 | 99.99% | 99.8% | 17.5 h |

**The lesson:** a ten-service request path built from 99.9% components delivers only 99% — two nines. To get four nines end-to-end, every one of twenty services must be 99.999%. This is why aggressive microservice decomposition *without* degradation paths destroys reliability.

**The mitigation — and the sentence to say:**

> *"Ten services in series at 99.9% gives 99%, so I'd either reduce the critical path or make every non-essential dependency degrade gracefully — serve a default, hide the section, or return partial results — so it's out of the availability path."*

### Components in parallel (redundancy)

```
        ┌──▶ A (99%) ──┐
Input ──┤              ├──▶ Output
        └──▶ B (99%) ──┘
Total = 1 − (1 − 0.99)² = 99.99%
```

**Availability improves with redundancy.** Two 99% components in parallel give 99.99%.

| Redundancy | Each at 99% | Combined | Each at 99.9% | Combined |
|---|---|---|---|---|
| 2 replicas | 99% | **99.99%** | 99.9% | 99.9999% |
| 3 replicas | 99% | 99.9999% | 99.9% | 99.999999% |

**But the caveat that makes it real:**

> **Parallel redundancy only helps if the failures are independent.** Two replicas sharing a rack, a power supply, an AZ, a database, a config deployment or a DNS provider are *not* independent — they fail together. This is why "multi-AZ" exists (independent power/network) and why "multi-region" exists (independent everything, at high cost).

**Correlated failure examples:**
- Both replicas in the same AZ → one AZ outage takes both.
- Both replicas deployed from the same bad config → both die at once.
- Both replicas behind the same load balancer → the LB is a single point of failure.
- Both replicas querying the same DB → the DB is the real availability floor.

**The rule:** *redundancy is only as good as the independence of its failure domains.*

### Availability of a whole system

```
End-to-end = (product of series components) × (parallel redundancy benefit)
```

**Worked example — a 4-tier web system:**

```
Client ─▶ LB (99.99%) ─▶ App tier (2 replicas @ 99.9%) ─▶ DB (primary + standby, 99.95%)
                          parallel: 1-(0.001)² = 99.9999%

Series:  0.9999 × 0.999999 × 0.9995
      = 0.9994  ≈ 99.94%
```

**Conclusion to state:** *"The database is the availability floor at 99.95% — adding a third app server buys nothing, because the series product is dominated by the weakest link. Money spent on the DB's failover path is where the availability comes from."*

**The general rule:** in a series system, **improving anything above the weakest link is wasted money.** Find the floor first.

---

## 3.3 The availability design toolkit

| Technique | Raises MTBF or lowers MTTR? | Availability gain | Cost |
|---|---|---|---|
| **Health checks + auto-restart** | Lowers MTTR | Moderate | Low |
| **Load balancer over replicas** | Both | High | Low–medium |
| **N+1 redundancy** | Lowers MTTR | High | +1× capacity |
| **Multi-AZ** | Both (independent failure domains) | High | ~2× |
| **Multi-region active-passive** | Both | High | ~2–3× + latency |
| **Multi-region active-active** | Both | Highest | High + consistency complexity |
| **Graceful degradation** | Raises effective availability | High | Medium |
| **Circuit breakers** | Lowers MTTR (stops cascades) | High | Low |
| **Canary deploys** | Raises MTBF | High | Low |
| **Automated rollback** | Lowers MTTR | High | Medium |
| **Chaos engineering** | Raises MTBF (finds latent issues) | Medium | Engineering time |
| **Rate limiting / load shedding** | Raises effective availability | Medium | Low |
| **Queues/async** | Decouples; absorbs bursts | Medium | Medium |
| **Idempotency + retries** | Masks transient failures | Medium | Low |
| **Read replicas** | Improves read availability | Medium | +replica cost |
| **Backups + tested restore** | Handles the catastrophic case | Existential | Low–medium |
| **Removing components** | Raises MTBF | Often highest | Negative cost |

---

## 3.4 The availability budgets of common dependencies

Useful for reasoning about a system's realistic ceiling.

| Dependency | Typical availability | Notes |
|---|---|---|
| Single VM | ~99.5–99.9% | Hardware + hypervisor |
| Single AZ | ~99.9–99.99% | |
| Managed DB (multi-AZ) | ~99.95–99.99% | Failover is minutes for many engines |
| Managed DB (single-AZ) | ~99.5–99.9% | Maintenance windows hurt |
| Object storage (S3-class) | 99.99% availability, 99.999999999% durability | Note availability ≠ durability |
| Load balancer (managed) | 99.99% | |
| DNS | 99.99%+ | But **TTL and caching** mean recovery is slow |
| CDN | 99.9–99.99% | |
| Third-party API | 99.9% (if good) | **Out of your control — the hardest case** |
| Kubernetes control plane | 99.95% (managed) | |
| LLM provider API | 99.5–99.9% | Rate limits and capacity are the real risk |
| Internet / BGP | ~99.9% | Regional outages happen |

**The third-party rule:** *you cannot exceed the availability of your critical dependencies.* If your LLM provider is 99.5% and inference is in the critical path, your ceiling is 99.5% unless you have a fallback model, a queue, or a degraded mode.

---

## 4. Variants & taxonomy

### Availability measurement variants

| Variant | Definition | When it's the right SLI |
|---|---|---|
| **Time-based** | % of wall-clock time the service responded | Infrastructure SLOs; simple systems |
| **Request-based** | % of requests that succeeded | **Preferred** — weights by traffic; a brief blip during high traffic costs more than a quiet-night outage |
| **Probe-based** | % of synthetic probes that succeeded | Third-party/black-box measurement |
| **Capacity-based** | % of time with headroom above threshold | Preventing saturation |
| **Correctness-based** | % of responses that were *right* | Data pipelines, pricing, ML predictions |
| **Freshness-based** | % of reads within an acceptable staleness | Caches, replicas, analytics |

### Availability target tiers — what to pick

| Tier | Target | Typical system | Architecture required |
|---|---|---|---|
| **Best-effort** | 99% | Internal tools, prototypes, batch jobs | One server, backups |
| **Standard** | 99.9% | Most consumer apps, B2B SaaS | Multi-instance, LB, health checks, monitoring |
| **High** | 99.99% | Payments, auth, core APIs | Multi-AZ, automated failover, canaries, on-call |
| **Critical** | 99.999% | Emergency services, exchanges, telecom core | Multi-region, automated everything, formal change control |

### Availability in different system types

| System | Realistic target | Dominant constraint |
|---|---|---|
| URL shortener | 99.99% | Read path; cache availability |
| Chat/messaging | 99.99% | Delivery + durable storage |
| Payment system | 99.999% | Consistency + auditability |
| Social feed | 99.9% | Acceptable to degrade to a stale feed |
| Video streaming | 99.9% | CDN absorbs the load |
| Search | 99.9% | Index availability |
| Ride-hailing | 99.99% | Real-time matching |
| LLM/agent API | 99.5–99.9% | **Upstream model provider** |
| Analytics pipeline | 99% | Hours of lag are tolerable |

---

## 5. Pros / Cons

### Pros of setting availability targets

| Pros | Cons / caveats |
|---|---|
| Turns "reliability" into a testable number | A target with no measurement is theatre |
| Enables the error budget → velocity trade-off | Requires real observability investment first |
| Forces the redundancy conversation to be quantitative | Targets can be gamed by redefining the SLI |
| Makes cost/benefit explicit (nines vs dollars) | Per-service SLOs can hide the user-facing experience |
| Lets you say no to risky changes legitimately | Can be used to block necessary risk-taking |
| Comparable across teams and services | Composition across services is usually unmeasured |

### Pros / Cons of each lever

| Lever | Pros | Cons |
|---|---|---|
| **Raise MTBF (redundancy)** | Simple to reason about; buys headroom too | Costly; **does not protect against bad deploys or correlated failure** |
| **Lower MTTR (automation)** | Cheaper per nine at scale; helps every failure mode | Requires real investment in observability and tooling first |
| **Multi-AZ** | Big availability jump for moderate cost | Doubles infra; cross-AZ latency + data transfer cost |
| **Multi-region** | Survives regional outage | High cost; consistency complexity; only worth it for global products or extreme targets |
| **Graceful degradation** | Cheap; often the biggest real-world win | Requires designing every feature to have a "reduced" mode |
| **Reduce components** | Improves MTBF *and* cost simultaneously | Conflicts with org structure and team autonomy |

---

## 6. Exceptions, edge cases & failure modes

| Exception / failure mode | Why the naive model breaks | How to handle |
|---|---|---|
| **Correlated failures** | "Two replicas = 99.99%" assumes independence; shared racks, AZs, configs and dependencies break it | Enumerate the shared failure domains; separate power, network, AZ, and deploy pipelines |
| **Cascading failure** | One slow dependency exhausts thread pools everywhere | Circuit breakers, timeouts, bulkheads, load shedding |
| **Retry storms** | Clients retrying a struggling service multiply load 3–5× at the worst moment | Retry budgets, exponential backoff with jitter, server-side shedding |
| **Thundering herd on recovery** | Everyone reconnects/refetches at once when a cache or service returns | Jittered backoff, connection draining, warm-up, request coalescing |
| **Maintenance windows** | Planned downtime counts against you if the SLO is wall-clock | Exclude maintenance from the SLO, or design zero-downtime deploys |
| **The SLI doesn't measure the user** | Server-side 200s while users see a broken page (JS error, CDN miss, bad data) | Add synthetic/browser-level SLIs for critical journeys |
| **Availability ≠ durability** | Object storage is 99.99% available and 99.999999999% durable — different nines for different things | Separate SLOs for availability, durability, correctness, freshness |
| **Availability ≠ consistency** | A read replica is highly available and served stale data | Choose per-path: availability-first reads, consistency-first writes |
| **Partial failure** | One shard down = 1/n of requests failing, not "the system is down" | Define the SLI per-shard or as a success ratio, not binary up/down |
| **The weakest link dominates** | Improving a strong component in series wastes money | Compute the series product; invest only at the floor |
| **Third-party ceiling** | You can't beat your dependencies | Fallbacks, degradation, multi-provider, queue-and-replay |
| **DNS TTL delays recovery** | Failover "completes" but clients cache the old IP for minutes | Low TTLs, anycast, client-side retry to alternates |
| **Availability during deploys** | Most outages are change-induced | Canary, blue/green, feature flags, progressive rollout |
| **Autoscaling lag** | Scaling takes minutes; the spike is seconds | Pre-warm, over-provision headroom, queue the excess |
| **Clock/timestamp skew** | Affects measurement and event ordering | NTP; logical clocks; monotonic timers |
| **The target is unachievable at the chosen architecture** | 99.99% with a single-AZ DB is impossible | Do the series math *before* promising the number |
| **Availability cost at the tail** | The last nine can dwarf all prior spend | Ask the business what an hour of downtime costs; buy nines up to that value |
| **Measuring only the happy path** | Excluding a failing endpoint from the SLI hides the outage | Define which endpoints are in scope *before* an incident |
| **Error budget too small to ship** | 99.99% on a fast-moving service freezes releases | Either loosen the target or invest in canaries; don't silently ignore it |

---

## 7. When to use / When NOT to use

### When a formal availability target is essential

| Situation | Why |
|---|---|
| Revenue-critical path | Directly maps to money |
| Contractual SLA with penalties | Legal exposure |
| Regulated systems | Compliance requirement |
| Multi-team dependencies | Need a shared number to design against |
| Anything with on-call | Determines alerting and escalation |
| Cost justification for redundancy | The only way to argue for it credibly |

### When a formal target is over-engineering

| Situation | Instead |
|---|---|
| Internal tool, handful of users | Best-effort; monitor, don't target |
| Prototype / spike | Best-effort |
| Batch job on a nightly schedule | Completion SLO, not uptime |
| A dev environment | None |
| A system whose downtime costs less than the redundancy | Take the downtime |

### The decision rule

> **Set the target from the cost of downtime, not from aspiration.** If an hour of downtime costs £X and the next nine costs £Y/year to buy, buy it only when `(hours of downtime avoided) × X > Y`.

---

## 8. Numbers, defaults & tuning

### The numbers table

| Item | Value |
|---|---|
| Seconds/year | 31,536,000 (~3.15 × 10⁷) |
| Seconds/month | ~2,592,000 (~2.6 × 10⁶) |
| 99% | 3.65 days/yr · 7.3 h/mo |
| 99.9% | 8.77 h/yr · 43.8 min/mo |
| 99.99% | 52.6 min/yr · ~26 s/mo |
| 99.999% | 5.26 min/yr · ~2.6 s/mo |
| Series composition | `A₁ × A₂ × … × Aₙ` |
| Parallel composition | `1 − (1 − A)ⁿ` |
| Default consumer target | 99.9% |
| Default critical-path target | 99.99% |
| Default internal tool | 99% or none |
| SLO buffer over SLA | SLO ≥ SLA + 0.05% |
| Typical SLO window | 30 days (rolling) |
| Typical multi-AZ cost | ~2× |
| Typical multi-region cost | ~2–3× + latency |
| Typical MTTR — manual | 30–120 min |
| Typical MTTR — automated failover | seconds–minutes |
| Typical MTTR — with canary + rollback | minutes |

### Tuning guidance

| If… | Then… |
|---|---|
| You're at 99.9% and need 99.99% | Automate detection and failover; MTTR must drop below ~6 min |
| Your SLI is time-based and flappy | Switch to request-based |
| Your error budget is always exhausted | Either the target is too strict or the service needs investment — decide explicitly |
| Your error budget is never touched | The target is too loose; tighten it and buy velocity |
| One dependency caps your availability | Add a fallback, a queue, or a degraded mode |
| You have 20 services in series | Reduce the critical path or make non-essentials async/degradable |

---

## 9. Interview Q&A

**Q1. What's the formula for availability?**
`Availability = MTBF ÷ (MTBF + MTTR)` — mean time between failures over mean time between failures plus mean time to repair. It's the ratio of good time to total time.

**Q2. What are the nines?**
99% is 3.65 days of downtime a year; 99.9% is 8.77 hours; 99.99% is 52.6 minutes; 99.999% is 5.26 minutes. Each nine is roughly an order of magnitude more cost and effort.

**Q3. Where does 99.99% come from — the number, not the aspiration?**
52.6 minutes a year is about 26 seconds a month. That's below human reaction time for detection, paging, logging in and acting, so four nines is only achievable with automated detection and failover. It's an automation target, not a diligence target.

**Q4. How do you improve availability?**
Two levers: raise MTBF — redundancy, testing, canaries, simpler systems — or lower MTTR — automated failover, health checks, observability, runbooks, rollback. At scale I invest more in MTTR, because failures are inevitable and you can't make them rare enough to reach four nines; you can make recovery fast enough.

**Q5. Availability versus reliability — what's the difference?**
Availability is "can I reach it right now." Reliability is "does it do the right thing consistently over time." They're independent in both directions: a service returning stale data is available but unreliable; a service that's down but has never served a wrong answer is reliable but unavailable.

**Q6. How does availability compose across services?**
In series it multiplies — ten services at 99.9% in the request path give 99%, which is 3.65 days of downtime instead of 8.77 hours. In parallel it improves — `1 − (1 − A)ⁿ` — but only if the failures are genuinely independent. Two replicas in the same AZ fail together, so redundancy is only as good as the independence of the failure domains.

**Q7. What's the availability of a system with a load balancer at 99.99%, two app servers at 99.9%, and a multi-AZ database at 99.95%?**
The app tier in parallel is 1 − 0.001² = 99.9999%. Series: 0.9999 × 0.999999 × 0.9995 ≈ 99.94%. And the conclusion is that the database is the floor — adding a third app server buys nothing because the series product is dominated by the weakest link.

**Q8. What are SLI, SLO and SLA?**
SLI is the measured indicator — "fraction of requests under 200 ms." SLO is the internal objective — "99.9% over 30 days." SLA is the customer contract with penalties. Your SLO must be stricter than your SLA so you get early warning before breaching the contract.

**Q9. What's an error budget?**
The downtime an SLO permits — 0.1% of 30 days is about 43 minutes. It converts reliability into a currency: if you have budget, ship faster; if you've exhausted it, freeze changes and invest in reliability. It's the mechanism that stops reliability and velocity from being an argument.

**Q10. Why is multi-AZ not the same as multi-region?**
Multi-AZ gives independent power and network within a region — it survives a datacentre failure but not a regional service, control-plane, or DNS failure. Multi-region gives independent everything, which is the only way to survive a regional outage, but it costs 2–3× and forces you to solve cross-region consistency.

**Q11. *Pushback:* "Isn't 99.99% always better than 99.9%?"**
No — it costs roughly 10× and only matters if an hour of downtime is expensive. For an internal dashboard, the right answer is 99% and the money is better spent elsewhere. I'd set the target from the cost of downtime: if an hour costs £X, buy the next nine only when the downtime avoided is worth more than the yearly cost of buying it.

**Q12. *Pushback:* "You added replicas, so why didn't availability improve?"**
Because redundancy only works across independent failure domains. If both replicas share an AZ, a rack, a deploy pipeline, a config, or an upstream database, they fail together. Improving the app tier buys nothing when the database — the series floor — is at 99.95%. I'd find the floor first and invest there.

**Q13. How do you measure availability for a sharded system?**
I'd use a request-based SLI rather than binary up/down, because one shard down means 1/n of requests fail rather than the system being "down." Request-based weighting also reflects reality: a blip during peak traffic costs more users than one at 4 a.m.

**Q14. What's the ceiling on your availability if you depend on a third-party LLM API at 99.5%?**
99.5%, unless the dependency has a fallback. That's why for a critical AI path I'd add a second provider, a smaller local model for degraded mode, or a queue-and-replay path — the same reasoning as any third-party dependency in the critical path.

**Q15. How do you choose between improving MTBF and MTTR for a specific system?**
I look at where the downtime actually comes from. If MTTR dominates — outages are long because detection and recovery are slow — invest in automation, observability and rollback. If failures are frequent — many separate incidents — invest in MTBF: testing, canaries, simpler architecture. Most teams find MTTR dominates, and it's usually the cheaper lever.

---

## 10. Common mistakes candidates make

1. **Confusing availability with reliability.** They're different SLOs.
2. **Promising nines without doing the series math.** A ten-service path can't deliver 99.99% if each service is 99.9%.
3. **Assuming redundancy helps without checking failure-domain independence.**
4. **Ignoring that the weakest link in a series dominates.** Investing above the floor is wasted money.
5. **Forgetting that your availability ceiling is your dependencies' availability.**
6. **Ignoring change-induced outages** — the largest real-world MTBF lever is safer deploys.
7. **Choosing an aspiration target rather than one derived from the cost of downtime.**
8. **Using binary up/down SLIs for sharded or partial-failure systems.**
9. **Skipping the SLO-over-SLA buffer.**
10. **Treating availability and durability as the same metric.**
11. **Forgetting DNS TTL in recovery time** — failover isn't complete when the client still caches the old IP.
12. **Not accounting for retry storms**, which amplify load exactly when the system is weakest.
13. **Ignoring autoscaling lag** — scaling takes minutes; spikes take seconds.
14. **Never measuring the SLI**, making the target decorative.
15. **Treating maintenance windows as free** when the SLO is wall-clock based.

---

## 11. Cheat-sheet summary

**The formula**

```
Availability = MTBF ÷ (MTBF + MTTR)
```

**The nines**

| Target | Down/year | Down/month |
|---|---|---|
| 99% | 3.65 days | 7.3 h |
| 99.9% | 8.77 h | 43.8 min |
| 99.99% | 52.6 min | ~26 s |
| 99.999% | 5.26 min | ~2.6 s |

**Composition**

| Arrangement | Formula | Effect |
|---|---|---|
| Series | `A₁ × A₂ × … × Aₙ` | Availability **decreases** |
| Parallel | `1 − (1 − A)ⁿ` | Availability **increases** — *only if failures are independent* |

**Fan-out reality check (each at 99.9%)**

| In series | Result |
|---|---|
| 1 | 99.9% |
| 5 | 99.5% |
| 10 | 99.0% |
| 20 | 98.0% |

**Two levers**

| Raise MTBF | Lower MTTR |
|---|---|
| Redundancy, quality, testing, canaries, fewer components | Automation, health checks, failover, observability, runbooks, rollback |

**SLI / SLO / SLA**

| | What | Example |
|---|---|---|
| SLI | Measured indicator | % requests < 200 ms |
| SLO | Internal target | 99.9% / 30 days |
| SLA | Customer contract | 99.9% or service credit |

**Rule:** SLO must be stricter than SLA.

**Availability ≠ reliability ≠ durability ≠ consistency** — separate SLOs for each.

**The sentence to say:** *"Availability in series multiplies, so a request touching ten 99.9% services is 99% — which is why I either shorten the critical path or make non-essential dependencies degrade gracefully."*

**The senior instinct:** *"At scale I invest more in MTTR than MTBF — failures are inevitable, but recovery speed is a choice."*

---

## 12. Cross-links

- Previous: [Capacity estimation playbook](05-capacity-estimation-playbook.md)
- Back to: [The five-step framework](04-five-step-interview-framework.md) · [Six goals & trade-offs](02-six-goals-and-tradeoffs.md)
- Production counterpart: [SLOs, error budgets & incident response](../../Operations/03-slos-error-budgets-and-incident-response.md)
- Ops playbooks: [Observability & monitoring](../../Operations/02-observability-and-monitoring.md) · [Reliability patterns](../../Operations/07-reliability-patterns.md) · [DR, failover & backup](../../Operations/09-dr-failover-and-backup.md)
- Architecture that delivers availability: [Load balancers](../03-scaling/01-load-balancers.md) · [Sharding & replication](../03-scaling/05-sharding-replication-and-consistency.md) · [Reliability patterns](../../Operations/07-reliability-patterns.md)
- Drill: [Foundations & capacity questions](../../Interview-Questions/01-foundations-and-capacity-questions.md)
