# The Six Goals of System Design — And Why They Fight Each Other

> **Source of truth:** *System Design Interview Prep | Day 1 — Foundations, Framework & Capacity Estimation* (Section: "What are you solving for?").

---

## TL;DR

Every system design balances **six goals: scalability, availability, reliability, performance, consistency and cost**. These goals **conflict with each other** — making a system more consistent makes it slower, making it more available costs you consistency, and maximum performance costs money. Your job in an interview is not to maximise all six; it is to **pick the right balance for the given use case** and defend it. A chat system prioritises reliability and availability (never lose a message); a stock exchange prioritises consistency (the price must be accurate everywhere); an Instagram feed prioritises performance (a feed that is a few seconds stale is fine).

> **60-second interview answer:** "Every design is a balance across six goals — scalability, availability, reliability, performance, consistency and cost — and they conflict. Making a system more consistent makes it slower; making it more available can cost you consistency; chasing maximum performance costs money. So I'd first pin down which two or three goals actually matter for *this* product. For a messaging system I'd prioritise reliability and availability, because losing a message is unacceptable. For a trading system I'd prioritise consistency, because two users seeing different prices is a correctness bug. Everything else follows from that."

---

## 1. Definition & mental model

### The framing from Day 1

> "Every system design balances these goals. And these goals **fight each other**. Your job is picking the right balance."

Two halves, both essential:

1. **The goals are a finite set of six.** Learn them and you can evaluate any design — yours or the interviewer's.
2. **They are in tension.** Any statement of the form "this design is better" is meaningless unless you say *better at what, and at the cost of what*.

### The mental model: the six dials on a mixing desk

Picture a mixing desk with six faders. Pushing one up pulls others down. There is no position where all six sit at maximum — that desk does not exist. The interview is about choosing the fader positions for the specific song you were asked to play.

```
   SCALABILITY   AVAILABILITY   RELIABILITY   PERFORMANCE   CONSISTENCY   COST
       ▲              ▲              ▲             ▲             ▲          ▲
       │              │              │             │             │          │
   ────┼──────────────┼──────────────┼─────────────┼─────────────┼──────────┼────
       │              │              │             │             │          │
       ▼              ▼              ▼             ▼             ▼          ▼
   More users     Always up      Never lose    Fast p99     Everyone     Affordable
   tomorrow       (uptime)       data          responses    sees same    at scale
                                                             thing
                    ◀────────── conflict ──────────▶
```

### The three product archetypes that anchor the trade-off

Memorise these three; they are the transcript's own examples and they appear constantly in interviews.

| Product | Prioritises | Sacrifices | The one-line reason |
|---|---|---|---|
| **WhatsApp** | **Reliability + Availability** | Latency, some consistency | *You can never lose a message.* A message that arrives 3 seconds late is fine; a message that never arrives is a product failure. |
| **Stock exchange / payments** | **Consistency** | Latency, availability during partitions | *The price must be accurate everywhere.* Two users seeing different prices is not a UX problem, it is a correctness bug with financial consequences. |
| **Instagram feed** | **Performance** | Freshness (weak consistency) | *Load fast.* A feed that is a few seconds stale is completely acceptable; a slow feed is not. |

> **Interview move:** when you're handed a product, immediately name which archetype it is closest to, out loud. "This is a WhatsApp-class system, so I'll bias toward reliability and availability, and accept eventual consistency on the read path." That one sentence signals seniority faster than any diagram.

---

## 2. Why it exists — the problem it solves

Without this framing, three specific failures happen:

1. **You optimise everything and finish nothing.** Six goals at maximum is an infinite project. Naming the two that matter collapses the design space.
2. **Your design decisions become unfalsifiable opinions.** "I'd use Cassandra" is a preference. "I'd accept eventual consistency here, which means Cassandra is a good fit" is a decision tied to a goal.
3. **The interviewer cannot follow your reasoning.** The goals give you a shared vocabulary. Saying "this trade-off buys availability at the cost of consistency" is instantly legible to any senior engineer.

It also solves the *"why do you need a cache?"* problem. Every component you add to a design should be traceable to one of the six goals. If you cannot trace it, you probably don't need it.

| Component you want to add | Goal it serves |
|---|---|
| Load balancer, autoscaling, sharding | Scalability |
| Replicas, multi-AZ, health checks, failover | Availability |
| Durable WAL, replication, backups, idempotency | Reliability |
| Cache, CDN, index, connection pooling, async | Performance |
| Transactions, quorum reads/writes, single-leader writes | Consistency |
| Right-sizing, caching, tiered storage, spot instances | Cost |

---

## 3. Internal mechanics — each goal defined precisely

### 3.1 Scalability

> **Can the system handle 10× more users tomorrow?**

The question is not "does it work now" but "does it still work when the load multiplies". Day 1's phrasing: *"Can the system handle 10× more users?"*

Two axes:

| Axis | Meaning | Ceiling | Typical mechanism |
|---|---|---|---|
| **Vertical (scale up)** | Bigger machine — more CPU, RAM, faster disk | Hard physical/cost ceiling; single point of failure remains | Larger instance type |
| **Horizontal (scale out)** | More machines | Effectively unbounded, but requires statelessness or partitioning | Load balancer + N servers, sharding |

**The non-negotiable prerequisite for horizontal scaling is statelessness.** If a server holds session state in local memory, requests must keep returning to it (sticky sessions), which limits your ability to add/remove capacity and makes failure recovery lossy. Push state to Redis or a database; then any server can serve any request.

**Key sub-question the interviewer will ask:** *"Does it scale linearly?"* Usually no. Adding a second database node does not double write throughput (coordination overhead), and adding cache nodes does not double hit rate (hot keys concentrate). Say so explicitly.

### 3.2 Availability

> **Is the system always up?**

Availability is **uptime as a fraction of total time**, usually expressed as a percentage of the year/month. It is measured, targeted and paid for.

Formally:

```
Availability = MTBF / (MTBF + MTTR)
```
where MTBF = mean time between failures, MTTR = mean time to recovery.

**The crucial insight: you raise availability faster by reducing MTTR than by increasing MTBF.** Preventing every failure is impossible; recovering in 30 seconds instead of 30 minutes is achievable engineering. This is why health checks, fast failover, and graceful degradation matter more than heroic prevention.

| Availability | Downtime / year | Downtime / month | Typical system |
|---|---|---|---|
| 99% ("two nines") | 3.65 days | 7.2 hours | Internal tools |
| 99.9% ("three nines") | 8.77 hours | 43.8 minutes | Standard consumer SaaS |
| 99.95% | 4.38 hours | 21.9 minutes | Business-critical SaaS |
| 99.99% ("four nines") | **52 minutes** | **~26 seconds** | Gmail-class (cited in Day 1) |
| 99.999% ("five nines") | ~5 minutes | ~2.6 seconds | Payments, telecom core |

> **Detail worth having ready:** the difference between 99.99% and 99.999% is roughly *one incident per year*. Going from four nines to five nines typically multiplies infrastructure cost, not by 10×, but by several-fold — and usually requires multi-region active-active.

**Common availability failure modes** (name at least two in any design):

- Single point of failure (one load balancer, one primary DB, one NAT gateway, one DNS provider).
- Cascading failure — a slow dependency consumes all threads, then the caller dies, then *its* callers die.
- Thundering herd on recovery — 10,000 clients retry simultaneously and re-kill the just-recovered service.
- Deployment-caused outage — the most common cause of downtime in mature orgs is a *change*, not hardware.
- Dependency outage — your provider (LLM API, FCM, payment gateway) is down and you have no fallback.

### 3.3 Reliability

> **Does it never lose your data? Does it do what it promised, every time?**

Availability and reliability are *different* and the distinction is a classic probe:

| | Availability | Reliability |
|---|---|---|
| Question | Is it up? | Does it behave correctly? |
| Failure mode | Requests fail fast (500/503), user sees an error | Requests succeed but are wrong, or data silently vanishes |
| Example | The service returns 503 for 10 minutes | The service returns 200 but dropped a write |
| Fix | Redundancy, failover, health checks | Replication + durability, transactions, checksums, idempotency, verification |

A system can be **available but unreliable** (returns answers quickly; they're wrong) or **reliable but unavailable** (correct data, but you can't reach it). Both are bad in different ways. The nastiest incidents are reliability failures because there is no alarm — the system looks healthy while corrupting data.

Concretely, reliability engineering means:

- **Durability:** a write is on stable storage (and replicated) before you acknowledge it.
- **Correctness under concurrency:** transactions, isolation levels, or optimistic version checks.
- **Exactly-once *effects*:** idempotency keys, dedup tables, at-least-once delivery + idempotent consumers.
- **Verification:** checksums, consistency scans, reconciliation jobs.

### 3.4 Performance

> **Does it respond under 200 ms?**

Day 1's anchor: *"Does it respond under 200 milliseconds?"* Performance is latency, throughput and their variance. Interviewers care about **percentiles, not averages**.

| Metric | Meaning | Why it matters |
|---|---|---|
| **p50 (median)** | Half of requests are faster | The "typical" experience |
| **p95 / p99** | Tail latency | Your best customers hit this constantly at scale |
| **Average** | Arithmetic mean | **Misleading** — one 30-second outlier hides 999 fast requests |
| **Throughput** | Requests/sec the system sustains | Capacity planning unit |
| **TTFT** | Time to first token (streaming/LLM) | Perceived responsiveness for AI systems |
| **TPOT / ITL** | Time per output token | Reading speed of a streamed answer |

> **The tail-amplification rule:** if one user request fans out to 100 backend calls, and each has a 1% chance of being slow, then ~63% of user requests hit at least one slow backend. Tail latency is *multiplied* by fan-out. This is why p99 matters more than p50 in microservice and multi-agent architectures.

**Sources of latency to name:** network round-trips (speed of light is a hard floor), serialisation, queueing, lock contention, GC pauses, cold caches, cold starts, and *queue depth* (see [capacity planning](../../Operations/05-capacity-planning-and-autoscaling.md) — latency explodes non-linearly as utilisation approaches 100%).

### 3.5 Consistency

> **Do all users see the same output? Is the data accurate everywhere?**

Day 1's framing: *"How accurate is it? Does every user see the same output? It shouldn't be that I see one output and later see something different."*

The spectrum:

| Level | Guarantee | Cost |
|---|---|---|
| **Strong / linearizable** | Every read sees the latest committed write, globally | High latency, coordination, reduced availability during partitions |
| **Read-your-writes** | *You* see your own writes immediately | Cheap-ish; needs sticky routing or session tokens |
| **Monotonic reads** | You never see time go backwards | Cheap |
| **Bounded staleness** | Reads are at most N seconds behind | Tunable, good middle ground |
| **Eventual consistency** | Replicas converge *eventually* | Cheapest, highest availability |

**Where inconsistency becomes a *bug*, not a UX issue:**

- Reading your own write (post a comment → it isn't there → user posts again).
- Double-spend / double-booking (two users buy the last seat).
- Inventory oversell.
- Feed showing a deleted/hidden post.

> **The read-replica trap (extremely common in interviews):** "I'll add read replicas for scale." Follow-up: *"A user updates their profile and immediately reloads — what do they see?"* With naive async replication they see the old value. Standard fix: route **read-your-writes** traffic to the primary for a short window after a write (or use a session-stickiness token, or read from primary if `last_write_ts > now - lag_bound`).

### 3.6 Cost

> **Can the company afford the return?**

Day 1's phrasing: *"What is its cost? It shouldn't be that what I built costs far too much. Can the company afford the return?"*

Cost is a **first-class design goal**, not an afterthought. Every redundancy, cache, replica and region multiplies spend. In AI systems this is sharper than anywhere else: at scale, **inference cost and egress cost dominate**, and a design that ignores them is a design that does not ship.

Cost levers to name:

| Lever | Effect |
|---|---|
| Right-sizing and autoscaling | Pay for peak only when peak exists |
| Caching (every layer) | Converts expensive compute into cheap memory reads |
| Tiered/object storage | Cold data at 1/10 the price |
| Spot/preemptible instances | 60–90% cheaper for interruptible work |
| Batching and async processing | Higher utilisation per unit |
| Compression and format choice | Directly reduces storage and egress |
| Multi-tenancy and resource pooling | Amortise fixed costs |
| Model routing/tiering (AI) | Cheap model for easy queries, expensive for hard ones |

**The "cost per unit of business value" reframe:** don't say "this costs $X/month". Say "this costs $X per 1,000 requests, which is Y% of the revenue per request" — that's what a staff engineer would say.

---

## 4. Variants & taxonomy — how the conflicts actually manifest

### 4.1 The named conflicts

| Conflict | Mechanism | Classic resolution |
|---|---|---|
| **Consistency ↔ Performance** | Strong consistency requires coordination (locks, quorums, round trips) before you can answer | Accept bounded staleness on the read path; keep writes strongly consistent |
| **Consistency ↔ Availability** | During a network partition you must choose: refuse to answer (CP) or answer with possibly-stale data (AP) | *CAP theorem* — choose per operation, not per system |
| **Availability ↔ Consistency** (restated) | Replicas that can accept writes independently of each other will diverge | Single-leader writes + follower reads |
| **Performance ↔ Cost** | The fastest hardware, most replicas and most regions cost the most | Cache aggressively; scale the hot path only |
| **Scalability ↔ Consistency** | More nodes = more coordination = harder consistency and higher tail latency | Partition data so cross-node coordination is rare |
| **Reliability ↔ Cost** | Every extra durable copy and every extra region is money | Tier durability by data value (hot vs cold, critical vs derived) |
| **Availability ↔ Reliability** | A system that always answers *something* can answer *wrongly*; a system that refuses when unsure is less available | Fail closed for safety-critical, fail open for availability-critical — decide explicitly |
| **Performance ↔ Reliability** | Caching improves performance by serving slightly stale data (a mild reliability/consistency hit) | Bound the staleness; invalidate on critical events |
| **Latency ↔ Consistency (geo)** | Serving a global user from the nearest replica means reading possibly-stale data | Region-local reads, global writes, or per-region data ownership |

### 4.2 CAP and PACELC — the formal statement

**CAP:** in the presence of a network **P**artition, a distributed system can guarantee at most two of **C**onsistency and **A**vailability.

The two common misreadings to avoid:

1. **"Pick 2 of 3."** Partitions are not optional — they *happen*. So you do not choose whether to have P; you choose how to behave **during** a partition. The real choice is CP or AP.
2. **"CAP applies all the time."** It only applies during a partition. The rest of the time, both C and A are achievable.

**PACELC** therefore extends it: **P**artition → **A** or **C**; **E**lse → **L**atency or **C**onsistency. This is the more useful framing because it covers the normal case, where the real choice is *latency vs consistency*.

| System | During partition | Else | Classification |
|---|---|---|---|
| DynamoDB, Cassandra | Available | Latency | PA/EL |
| MongoDB (default) | Consistent | Latency | PC/EL |
| BigTable/HBase | Consistent | Consistency | PC/EC |
| Postgres single-primary | Consistent | Consistency | PC/EC |
| Spanner | Consistent | Consistency (with latency cost) | PC/EC (via TrueTime) |

### 4.3 Trade-offs as a decision table (the interview artefact)

This is how you should present trade-offs on the whiteboard:

| Decision | Buys you | Costs you | Choose it when |
|---|---|---|---|
| Cache in front of DB | Latency ↓, DB load ↓ | Staleness, invalidation complexity | Read-heavy, staleness tolerable |
| Read replicas | Read throughput ↑, availability ↑ | Replication lag, read-your-writes bugs | Read:write ratio > 5:1 |
| Sharding | Write throughput ↑, unbounded storage | Cross-shard queries, rebalancing pain | Write QPS or storage exceeds one primary |
| Async queue (Kafka/SQS) | Decoupling, burst absorption, resilience | Eventual consistency, ordering complexity, DLQ ops | Producer faster than consumer, or spikes |
| Strong consistency (quorum) | Correctness | Latency, availability under partition | Money, inventory, auth |
| Multi-region active-active | Latency ↓, regional survival | Conflict resolution, cost, complexity | Global user base, high availability target |
| Precomputed feeds (fan-out on write) | Read latency ↓ | Write amplification, hot-user problem | Read-heavy timeline product |
| Serverless | Zero ops at low scale | Cold starts, cost cliff at high scale, less control | Spiky/low traffic |

---

## 5. Pros / Cons

### Pros of reasoning explicitly in these six goals

| Pros | Cons / caveats |
|---|---|
| Gives every design decision a justification — no more "I'll use Redis" with no *why* | Six goals is a simplification; real systems have more (security, compliance, operability, developer velocity) |
| Instantly legible to senior interviewers; shows you think like a staff engineer | Naming a priority without defending it is worse than not naming one |
| Prevents over-engineering: goals you *haven't* prioritised justify *not* building things | Can become a checklist recited without thought — interviewers notice |
| Translates directly into the trade-off section every strong answer needs | The "right" balance depends on requirements you must first elicit |
| Transfers across every domain, including AI/LLM systems | Cost and security are underweighted by most candidates; both matter enormously in 2026 AI systems |

### Pros / Cons of the three archetypes

| Archetype | Pros | Cons |
|---|---|---|
| **Reliability-first (WhatsApp)** | Users trust the product with irreplaceable data; failures are survivable | Expensive (durability everywhere); higher write latency; complex dedup |
| **Consistency-first (exchange)** | No correctness bugs; auditable; regulator-friendly | Slow; unavailable during partitions; hard to scale geographically |
| **Performance-first (feed)** | Best UX per dollar; cheap (caches everywhere); degrades gracefully | Stale reads confuse users occasionally; cache invalidation complexity |

---

## 6. Exceptions, edge cases & failure modes

| Exception | What it looks like | How to handle / what to say |
|---|---|---|
| **The interviewer changes the priority mid-interview** ("now make it strongly consistent") | Your whole design must bend | State what must change: single-leader writes, quorum reads, drop cache-aside on the affected path, accept higher latency. Show the delta, don't redraw everything. |
| **Different operations need different guarantees** | One system cannot be "CP" or "AP" globally | Choose per operation: auth and payments strongly consistent; profile view and feeds eventually consistent. Say this explicitly — it is a strong signal. |
| **Availability measured against the wrong denominator** | 99.99% for the whole product when one dependency is 99.9% | Multiply dependency availabilities: two 99.9% dependencies in series → 99.8%. Add redundancy or a fallback to break the chain. |
| **"Reliability" used as a synonym for "availability"** | Interviewer pushes back on sloppy usage | Distinguish: availability = is it up; reliability = is it correct / did it lose data. |
| **Cost ignored until asked** | Candidate designs multi-region active-active for a 1k-DAU internal tool | Proactively say "for this scale, multi-region is not justified; here's the milestone at which it would be." |
| **Caching silently converts availability into staleness** | Redis is down → all traffic hits the DB → DB dies | This is a *cache-availability* dependency. Mitigate with: local in-process cache as L1, request coalescing, circuit breaker on the DB, graceful degradation to stale-while-error. |
| **The availability/consistency choice is made implicitly** | Design behaves as AP during a partition by accident | Decide and state: "during a partition I choose availability here, and I reconcile later via X." |
| **Performance optimisations that hurt reliability** | Fire-and-forget async writes lose data on crash | Use durable queues (write to log/disk before ack), not in-memory channels. |
| **99.99% target with a 99.9% database** | Arithmetic does not work | Redundancy (multi-AZ failover) or a design that tolerates DB unavailability (serve from cache, queue writes). |
| **Tail latency ignored** | p50 is 20 ms, p99 is 4 s — "it's fast" | Report p50/p95/p99; explain fan-out amplification; add timeouts + hedged requests. |
| **Regulatory constraints override all six** | PCI/HIPAA/GDPR/data-residency | Name them early: they can force single-region storage, encryption at rest, and audit logging — which then constrains every other goal. |

---

## 7. When to use / When NOT to use

### Which goal to lead with (decision table)

| If the product is… | Lead with | Then optimise | Accept weakness in |
|---|---|---|---|
| Chat / messaging (WhatsApp) | Reliability, Availability | Cost | End-to-end latency, cross-device read consistency |
| Payments / trading / inventory | Consistency, Reliability | Cost | Latency, availability under partition |
| Social feed / media (Instagram, YouTube) | Performance, Scalability | Cost | Freshness, read consistency |
| Search | Performance, Scalability | Cost | Index freshness (seconds to minutes stale is fine) |
| Internal tool / low traffic | Cost, Reliability | Simplicity | Scalability (can be added later) |
| Healthcare / regulated data | Consistency, Reliability, **Security** | Cost | Global distribution |
| Real-time collaboration (docs) | Consistency (convergent), Availability | Performance | Strict linearizability |
| AI inference at scale | Cost, Performance | Reliability | Freshness of model/embedding index |

### When NOT to optimise a goal

- **Don't chase five-nines for a system whose users don't notice 10 minutes of downtime.** The cost is enormous; the benefit is zero.
- **Don't add strong consistency to a read path where staleness is invisible.** Profile views, like counts, feed ordering.
- **Don't shard before one primary is saturated.** Sharding adds cross-shard query and rebalancing complexity that you will pay for daily.
- **Don't multi-region before you have users in multiple regions.** It multiplies cost and introduces conflict resolution.
- **Don't cache before you've estimated.** No estimation → no justification → the cross-question lands.
- **Don't add a component for a goal you haven't been asked to hit.** Every box on your diagram must earn its place.

---

## 8. Numbers, defaults & tuning

| Goal | Metric | Typical production target | Notes |
|---|---|---|---|
| Scalability | Headroom | Run at **50–70%** peak utilisation | Above ~80%, queueing latency explodes |
| Scalability | Scale-out step | Add **20–50%** capacity at a time | Avoid huge jumps; they cause cold-cache storms |
| Availability | SLO | 99.9% consumer, 99.99% critical | See [availability numbers](06-availability-numbers-and-slos.md) |
| Availability | MTTR target | **< 5 min** for tier-1 | Recovery speed beats prevention |
| Reliability | Durability | 11 nines for object storage (99.999999999%) | S3-class |
| Reliability | Data loss tolerance | **RPO = 0** for payments/chat | Choose explicitly; RPO=0 is expensive |
| Performance | p50 | 50–100 ms API | |
| Performance | p99 | **< 200 ms** for interactive APIs (Day 1 anchor) | |
| Performance | LLM TTFT | < 500 ms perceived, < 2 s acceptable | Streaming hides total latency |
| Performance | LLM TPOT | 20–50 ms/token ≈ 20–50 tok/s | Human reading speed ≈ 5–10 tok/s, so faster feels instant |
| Consistency | Replication lag | < 100 ms intra-region, 100 ms–1 s cross-region | Above this, read-your-writes breaks visibly |
| Consistency | Cache TTL | 1 s (hot), 5 min (warm), 1 h+ (cold) | Trade freshness for hit rate |
| Cost | GPU utilisation | **> 60%** for self-hosted inference to beat API pricing | Below that, API providers are cheaper |
| Cost | Cache hit rate | **> 80%** for the cache to be worth its complexity | Below 60%, revisit key/TTL design |
| Cost | Egress | Design to keep data in-region | Cross-region egress is a top-3 cloud bill line |

---

## 9. Interview Q&A

**Q1. Name the six goals of system design.**
Scalability, availability, reliability, performance, consistency and cost.

**Q2. Why can't you maximise all six?**
They are structurally in tension. Strong consistency needs coordination, which costs latency and reduces availability during partitions. Maximum performance needs the fastest hardware and replicas, which costs money. Higher durability means more replication, which costs money and adds write latency. Every gain is paid for somewhere.

**Q3. Give a concrete example of two goals conflicting.**
Making a system more consistent makes it slower — a quorum read must contact multiple replicas and wait for a majority. Making a system more available can reduce consistency — the classic is accepting writes on a partitioned replica, which then diverges.

**Q4. Which goals would you prioritise for WhatsApp, and why?**
Reliability and availability. You can never lose a message, so every accepted write must be durably stored and replicated before acknowledgement, and the service must accept messages even when parts of the infrastructure are degraded. I'd accept higher end-to-end latency and eventual consistency on read receipts and multi-device sync.

**Q5. Which for a stock exchange?**
Consistency and reliability. The price must be accurate everywhere, and an incorrect match is a financial and legal failure. I'd accept higher latency and reduced availability during partitions — a CP system. That means single-leader or quorum writes, and refusing reads rather than serving a stale price.

**Q6. Which for an Instagram feed?**
Performance. A feed that's a few seconds stale is fine; a slow feed is not. So I'd cache aggressively, precompute fan-out on write, serve from edge/Redis, and accept eventual consistency and some staleness in the ranking.

**Q7. What's the difference between availability and reliability?**
Availability is *is it up?* — a fraction of time the system answers requests. Reliability is *is it correct, and did it keep my data?* A system returning 503 is unavailable; a system returning 200 while silently dropping writes is unreliable. Reliability failures are worse because they're invisible.

**Q8. How do you improve availability?**
Two levers: increase MTBF (redundancy, health checks, canaries) or decrease MTTR (fast failover, graceful degradation, automated rollback). In practice MTTR is the cheaper lever — you can't prevent every failure, but you can recover in seconds instead of hours.

**Q9. Why is p99 more important than p50?**
At scale, a large fraction of users hit the tail regularly, and tail latency amplifies with fan-out: if a request calls 100 services each with a 1% slow chance, ~63% of requests are slow. Averages also hide outliers entirely. Interviewers expect p50/p95/p99, not a mean.

**Q10. Explain the read-replica consistency trap.**
Async replication lags. A user writes to the primary and immediately reads from a replica, seeing stale data — often their own write missing. Fixes: route read-your-writes traffic to the primary for a short window, use session stickiness with a write-timestamp token, or track replication lag and fall back to the primary when lag exceeds a bound.

**Q11. Is CAP "pick two of three"?**
No — that's the common misreading. Partitions happen whether you like it or not, so you don't choose whether to have P. You choose how to behave *during* a partition: CP (refuse to answer, stay correct) or AP (answer, possibly stale). Outside a partition you can have both C and A — which is what PACELC captures, adding the latency-vs-consistency choice in the normal case.

**Q12. *Pushback:* "You said consistency matters, but you also added a cache — isn't that contradictory?"**
No, because I choose consistency per operation, not per system. Auth, payments and inventory writes go to the primary with strong consistency. Feed reads and profile views are served from cache with a bounded TTL because staleness there is invisible to users. A cache is not a blanket weakening of consistency — it's a scoped one.

**Q13. *Pushback:* "Your design is expensive. Defend it."**
Here's the cost breakdown: the cost is dominated by X (replication/egress/inference) at $Y per 1,000 requests, which is Z% of the revenue per request. If that ratio is unacceptable, the levers are: cache more aggressively to cut the expensive path, tier storage, reduce replication factor on derived data, or route a fraction of traffic to a cheaper model. I'd instrument cost per request before optimising, because the dominant term is usually not the one you'd guess.

**Q14. How do the six goals apply to an LLM system?**
Same six with AI-specific metrics. Scalability = tokens/sec of the GPU pool. Availability = provider failover across OpenAI/Anthropic/self-hosted. Reliability = no dropped user requests, idempotent tool calls, no silent model regression. Performance = TTFT and TPOT rather than just p99. Consistency = index freshness and identical answers to identical prompts where required. Cost = cost per 1,000 queries, dominated by context length.

**Q15. What's the biggest mistake candidates make with trade-offs?**
Stating a choice without a cost. "I'll use eventual consistency" is incomplete — the complete sentence is "I'll use eventual consistency here, which means a user might briefly see stale data, and that's acceptable because the read path is a recommendation feed." Naming the cost is what proves you understand the trade-off.

---

## 10. Common mistakes candidates make

1. **Reciting all six goals without picking.** The interviewer learns nothing about your judgement.
2. **Confusing availability with reliability** — the single most common definitional slip.
3. **Claiming "high availability" without a number or a mechanism.** Say 99.99% and say *how*.
4. **Ignoring cost entirely.** In 2026 AI interviews, cost blindness is disqualifying.
5. **Treating CAP as "pick two".** Signals shallow understanding immediately.
6. **Global consistency decisions instead of per-operation.** Real systems are mixed.
7. **Using averages for latency.** Never report a mean without percentiles.
8. **Forgetting that adding a cache creates an availability dependency.** Redis down → DB stampede.
9. **Over-engineering for a goal nobody asked for.** Multi-region for 1k users.
10. **Not stating the sacrifice.** Every trade-off sentence needs a "which costs me…".
11. **Forgetting regulatory/security goals.** Data residency can override everything.
12. **Failing to revisit priorities when the interviewer changes them.** Adapt visibly.

---

## 11. Cheat-sheet summary

| Goal | The question it answers | Primary levers | What it costs you |
|---|---|---|---|
| **Scalability** | Can it handle 10× more users? | Horizontal scaling, statelessness, sharding, queues, autoscaling | Coordination complexity, cross-shard queries |
| **Availability** | Is it always up? | Redundancy, multi-AZ, health checks, fast failover, graceful degradation | Money; and availability often forces weaker consistency |
| **Reliability** | Does it never lose data / always behave correctly? | Durable WAL, replication, transactions, idempotency, verification | Write latency, storage cost, complexity |
| **Performance** | Does it answer under 200 ms? | Caching, CDN, indexing, connection pooling, async, batching | Staleness (via cache), money (via hardware) |
| **Consistency** | Do all users see the same thing? | Transactions, quorums, single-leader writes, sync replication | Latency, availability under partition |
| **Cost** | Can the company afford the return? | Right-sizing, caching, tiered storage, spot, batching, model routing | Some performance/durability headroom |

**Archetypes to memorise:**

| Product | Leads with | Accepts weakness in |
|---|---|---|
| WhatsApp | Reliability + Availability | Latency, read consistency |
| Stock exchange | Consistency | Latency, availability under partition |
| Instagram feed | Performance | Freshness |

**Conflicts in one line each:**
- Consistency ↑ → Performance ↓
- Availability ↑ → Consistency ↓
- Performance ↑ → Cost ↑
- Scalability ↑ → Coordination ↑ (consistency harder)
- Reliability ↑ → Cost ↑

**Numbers:** 50–70% target utilisation · p99 < 200 ms · MTTR < 5 min · replication lag < 100 ms intra-region · cache hit rate > 80% · GPU utilisation > 60% to beat API pricing.

**The sentence that wins points:** *"Every design balances these goals, and they fight each other — so my job is picking the right balance for the given use case, and stating what I'm giving up."*

---

## 12. Cross-links

- Previous: [What is system design and why](01-what-is-system-design-and-why.md)
- Next: [HLD vs LLD](03-hld-vs-lld.md) → [The five-step interview framework](04-five-step-interview-framework.md)
- Quantifying these goals: [Capacity estimation playbook](05-capacity-estimation-playbook.md) · [Availability numbers & SLOs](06-availability-numbers-and-slos.md)
- Mechanisms for each goal: [Load balancers](../03-scaling/01-load-balancers.md) · [Caching](../03-scaling/02-caching-strategies-and-layers.md) · [Sharding, replication & consistency](../03-scaling/05-sharding-replication-and-consistency.md) · [Reliability patterns](../../Operations/07-reliability-patterns.md)
- In production: [SLOs, error budgets & incident response](../../Operations/03-slos-error-budgets-and-incident-response.md) · [Cost & FinOps](../../Operations/06-cost-and-finops.md)
