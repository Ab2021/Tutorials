# Cheat Sheet — Foundations, Framework, Estimation & Availability

> **Print this. It is the whole of Day 1 on two pages.** Everything here is sourced from the *AI System Design Roadmap 2026*, *Day 1 — Foundations, Framework & Capacity Estimation*, and the *URL shortener* walkthrough.

---

## 1. The interview in one frame

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. CLARIFY      2. ESTIMATE     3. API       4. HLD     5. DEEP    │
│  requirements    capacity        contracts    diagram    DIVE      │
│                                                                     │
│  functional +    QPS, peak,      endpoints +  components  trade-offs│
│  non-functional  storage, BW     payloads     + arrows    defended  │
│                                                                     │
│  6 min           5 min           5 min        15 min      14 min    │
│                        (times for a 45-minute round)                │
└─────────────────────────────────────────────────────────────────────┘
                 NEVER START DRAWING. ASK FIRST.
```

**The rule:** "Design Twitter" could mean 100 different systems. Guess the wrong scope and you've wasted the interview.

**The second round.** The five steps above are the **HLD** round. There is a separate **LLD / machine-coding** round with its own seven steps — *clarify semantics → entities → diagram → interfaces → code → concurrency → extend* — and its own cheat sheet: [Cheat-Sheets/07](07-lld-cheatsheet.md). The five steps below do **not** transfer wholesale: LLD has no capacity estimation and no multi-region topology, and its proof of quality is complexity, not QPS. Tell the two apart by prompt: a *product* or *"at scale"* is HLD; *"classes"*, *"thread-safe"*, *"implement"* or *"extensible"* is LLD.

---

## 2. Definitions on the tip of your tongue

| Term | One-line answer |
|---|---|
| **System design** | Deciding how to build a large system that serves millions without breaking or slowing down |
| **Your role** | The architect, not the builder — you are not writing code here |
| **HLD** | Components and connections — boxes with labelled arrows |
| **LLD** | Classes, methods, interfaces, data structures, complexity |
| **Scalability** | How well performance holds as load grows |
| **Availability** | Can I reach it right now? |
| **Reliability** | Does it do the right thing consistently over time? |
| **Performance** | How fast is one request, now |
| **Consistency** | Do all readers see the same data? |
| **CAP** | During a Partition, choose Consistency or Availability |
| **PACELC** | …Else, trade Latency against Consistency |
| **MTBF / MTTR** | Mean time between failures / mean time to repair |
| **p50 / p95 / p99** | Median / 95th / 99th percentile latency — the tail is what hurts |
| **SLI / SLO / SLA** | Measured indicator / internal target / customer contract |

---

## 3. The four problems that make systems hard

| Problem | The naive failure | The fix |
|---|---|---|
| 1. Server crashes under load | One server, too many users | Load balancer + stateless replicas |
| 2. DB slow under millions of queries | Every read hits the database | Cache + read replicas |
| 3. Data lost when a machine fails | Single node, no copies | Replication + tested backups |
| 4. Two users update the same record | Last write silently wins | Locks, conditional writes, versioning |

**McDonald's mapping:** multiple counters → **load balancer** · pre-made patties → **cache** · backup supplier → **consistent API** · manager watching orders → **monitoring**.

---

## 4. The six goals — and the conflicts

| Goal | The question it answers | Key insight |
|---|---|---|
| **Scalability** | Does performance hold at 10×? | Requires **statelessness** to scale out |
| **Availability** | Is it reachable now? | `MTBF ÷ (MTBF + MTTR)` |
| **Reliability** | Is it correct over time? | ≠ availability |
| **Performance** | How fast? | Judge on **p99**, not average |
| **Consistency** | Do all readers agree? | A spectrum, not a boolean |
| **Cost** | What does it cost to run? | The goal that makes the others finite |

### The conflicts, memorised

| Tension | Because |
|---|---|
| Consistency ↔ Availability | CAP: during a partition you can't have both |
| Consistency ↔ Latency | Quorum reads/writes cost round trips |
| Performance ↔ Consistency | Caches and replicas are fast because they're stale |
| Availability ↔ Cost | Redundancy costs ~2–3× |
| Scalability ↔ Simplicity | Distribution introduces coordination |
| Everything ↔ Time-to-market | The perfect design ships too late |

### Product archetypes

| Product | Lead with | Sacrifice |
|---|---|---|
| **WhatsApp** | Reliability + availability | Eventual consistency (a second late is invisible) |
| **Stock exchange** | Consistency | Latency, and it's fine to halt trading |
| **Instagram feed** | Performance | Consistency (staleness is invisible) |
| **Payments** | Consistency + auditability | Latency |
| **Analytics** | Cost + throughput | Freshness (hours of lag ok) |

---

## 5. Capacity estimation — the formulas

```
requests/day = DAU × requests per user per day
QPS          = requests/day ÷ 86,400        (≈ ÷ 10⁵)
peak QPS     = QPS × 3
storage      = requests/day × record size
bandwidth    = QPS × record size

STORAGE, EXTENDED:
  annual      = daily × 365
  with index  = annual × 1.5          ← MOST PEOPLE FORGET
  replicated  = indexed × 3           ← MOST PEOPLE FORGET
  total       = replicated × retention years
```

### The anchors

| Anchor | Value |
|---|---|
| Seconds/day | **86,400 ≈ 10⁵** |
| **1M requests/day** | **≈ 12 QPS** |
| **1B requests/day** | **≈ 12,000 QPS** |
| **1 KB × 1M** | **1 GB** |
| 1 KB × 1B | 1 TB |
| Peak multiplier | **3×** |
| Single DB reads | **≈ 5,000 QPS** |
| Single DB writes | ≈ 1,000–5,000 QPS |
| App server | ≈ 1,000 RPS |
| Redis node | ≈ 100,000 ops/sec |
| Kafka broker | ≈ 100,000 msgs/sec |
| Node storage | 1–10 TB |
| Replication factor | 3× |
| Index overhead | 1.5× |

### Default assumptions when the interviewer won't say

| Parameter | Default |
|---|---|
| DAU | 1M small · 10M medium · 100M large |
| Requests/user/day | 1–10 |
| Read:write | 100:1 feed · 10:1 typical · 1:1 chat |
| Record size | text 1–2 KB · URL 500 B · image 200 KB–2 MB · video 5–50 MB |
| Peak | 3× |
| Retention | 1–5 years |
| Availability target | 99.99% consumer · 99% internal |

### Estimate → architecture (the money table)

| If the number is… | Then you need… | Because |
|---|---|---|
| Read QPS > 5,000 | **Cache** | A DB can't absorb it |
| Read:write > 10:1 | **Cache-first design** | Reads dominate |
| Write QPS > 5,000 | **Shard / write queue** | One primary saturates |
| Storage > 10 TB | **Distributed store** | One node can't hold it |
| Fan-out > 1,000/event | **Async queue + hybrid push/pull** | Sync fan-out is impossible |
| Peak:avg > 5:1 | **Queue for bursts** | Provisioning for peak wastes money |
| Bandwidth > 1 GB/s | **CDN** | Egress cost + origin protection |
| Global + tight p99 | **Multi-region / edge** | Speed of light |
| Any storage number | **× 3 for replicas** | Replication isn't free |

### The professional sentence

> *"I measured **[METRIC]** at **[NUMBER]**, and **[RESOURCE]** handles about **[CAPACITY]**, so we need **[COMPONENT]** to **[EFFECT]**, bringing **[METRIC]** from **[BEFORE]** to **[AFTER]**."*

**Example:** *"Read QPS is 50,000 and a single DB handles 5,000, so a cache absorbing 90% of reads is required, bringing latency from 50 ms to 5 ms."*

### The rules of delivery

✅ State assumptions **before** numbers · write it on the board · round aggressively · say units · finish with the architectural conclusion · name the **crossover point** where the design would change
❌ Silent math · false precision · stopping at the number · designing for the average · forgetting replication

---

## 6. Availability — the numbers

```
Availability = MTBF ÷ (MTBF + MTTR)
```

| Target | Down/year | Down/month |
|---|---|---|
| 90% | 36.5 days | 3 days |
| 99% | **3.65 days** | 7.3 hours |
| 99.9% | **8.77 hours** | 43.8 min |
| 99.99% | **52.6 min** | ~26 sec |
| 99.999% | 5.26 min | ~2.6 sec |

### Composition

| Arrangement | Formula | Effect |
|---|---|---|
| **Series** (all must work) | `A₁ × A₂ × … × Aₙ` | Availability **decreases** |
| **Parallel** (redundancy) | `1 − (1 − A)ⁿ` | Availability **increases** — *only if failures are independent* |

### The fan-out reality check (each service at 99.9%)

| Services in the request path | End-to-end availability | Downtime/year |
|---|---|---|
| 1 | 99.9% | 8.77 h |
| 3 | 99.7% | 26.3 h |
| 5 | 99.5% | 43.8 h |
| **10** | **99.0%** | **3.65 days** |
| 20 | 98.0% | 7.3 days |

> **Say this:** *"Ten services in series at 99.9% gives 99%, so I'd either shorten the critical path or make non-essential dependencies degrade gracefully."*

### The two levers

| Raise MTBF (rarer failures) | Lower MTTR (faster recovery) |
|---|---|
| Redundancy (N+1) · better hardware · testing · canaries · **fewer components** | Automated failover · health checks · observability · runbooks · rollback |
| **Costly; doesn't stop bad deploys** | **Cheaper per nine at scale — usually the better investment** |

### Design toolkit

| Technique | Lever | Cost |
|---|---|---|
| Multi-AZ | Both | ~2× |
| Multi-region active-active | Both | ~3× + consistency work |
| Graceful degradation | Effective availability | Low — **often the biggest real win** |
| Circuit breakers | MTTR | Low |
| Canary + automated rollback | MTBF + MTTR | Low |
| Rate limiting / load shedding | Effective availability | Low |
| Removing components | MTBF | **Negative cost** |

### Target selection rule

> **Set the target from the cost of downtime, not from aspiration.** Buy the next nine only when `(hours of downtime avoided) × (cost per hour) > (yearly cost of buying it)`.

| Tier | Target | Architecture required |
|---|---|---|
| Best-effort | 99% | One server + backups |
| Standard | 99.9% | Multi-instance, LB, health checks |
| High | 99.99% | Multi-AZ, automated failover, canaries, on-call |
| Critical | 99.999% | Multi-region, automated everything |

---

## 7. The URL shortener — the flagship, in one block

**Requirements:** create · redirect · custom alias · expiration · click tracking · delete
**Non-functional:** 100M DAU · 1k writes/s · 100k reads/s · 99.9% · unique **and unpredictable** · scale to 1B

```
Writes:  1,000/s  →  86.4M URLs/day
Reads:   100,000/s → 8.64B/day        RATIO = 100:1 READ-HEAVY
Gap:     100k reads vs 5k DB capacity = 60×  →  CACHE IS MANDATORY
Storage: 43 GB/day → 15.8 TB/yr → ~355 TB over 5 yrs (replicated)
Codes:   62⁷ = 3.5 × 10¹²  ·  collision ≈ 1 in 22,000
```

```
Client → CDN → LB → API Gateway → Rate Limiter
                                    ├──▶ URL Service ──▶ NoSQL (+ replica)
                                    └──▶ Redirect Service ──▶ Redis (+replica)
                                                                 └─ miss ─▶ NoSQL
                                             └─▶ Kafka ─▶ Stream ─▶ Analytics DB
```

**Key decisions**

| Decision | Answer | Why |
|---|---|---|
| Code generation | Distributed ID → base62 | No coordination, unpredictable, 3.5T codes |
| Collisions | **Conditional write**, regenerate on failure | Race-free, no read-before-write |
| Redirect code | **302**, not 301 | Browsers cache 301 forever → analytics break |
| Storage | NoSQL, partition key = `shortCode` | Pure key lookup, uniform distribution |
| Analytics | Kafka, fire-and-forget | Sync would raise writes 3k → 103k/s |
| Alias consistency | Strong — conditional write, 409 | Two users could claim the same alias |
| Biggest risk | **Cache tier down → DB gets 20× its capacity** | Circuit breaker + shedding + serve-stale |
| Key metric | **Cache hit rate** | It *is* the design |

---

## 8. Trade-off phrases that score

| Instead of… | Say… |
|---|---|
| "I'll use a cache." | "Read QPS is 50,000 and the DB handles 5,000, so a cache absorbing 90% of reads is required — p99 goes from 50 ms to 5 ms, at the cost of ≤1 s of staleness, which a feed tolerates." |
| "I'll use NoSQL." | "Access is pure key lookup with no joins at 355 TB, so a key-value store scales horizontally where SQL would need manual sharding — at the cost of ad-hoc queries, which we don't need." |
| "I'll add a queue." | "Peak is 3× average, so a queue absorbs the burst and decouples delivery — at the cost of eventual consistency in analytics, which tolerates seconds of lag." |
| "I'll replicate." | "Three-way replication means 15.8 TB/year becomes ~48 TB/year, so replication is part of the storage estimate, not an afterthought." |
| "It's highly available." | "The DB at 99.95% is the series floor, so a third app server buys nothing — availability comes from the DB's failover path." |

---

## 9. The 15 mistakes that cost interviews

1. Drawing before asking
2. Asking 20 clarifying questions
3. Skipping estimation → "why do you need that?"
4. Estimating silently
5. Designing for average, not peak
6. Forgetting replication (3×)
7. Forgetting index overhead (1.5×)
8. Estimating input volume, not derived/fan-out volume
9. Assuming bandwidth is the bottleneck
10. Chasing precision over order of magnitude
11. Losing units (KB vs Kb, 1000 vs 1024)
12. Stopping at the number without the conclusion
13. Declaring components instead of deriving them
14. No stated trade-offs
15. Confusing availability with reliability

---

## 10. The close

> *"So the design trades **[what you gave up]** for **[what you gained]**. If the requirement changed to **[Y]**, I'd change **[Z]** and accept **[new cost]**."*

**Never leave the interview without saying what you sacrificed.** An unfinished diagram with a stated trade-off beats a finished diagram with none.

---

## Cross-links

- Full detail: [What is system design](../HLD/01-foundations/01-what-is-system-design-and-why.md) · [Six goals](../HLD/01-foundations/02-six-goals-and-tradeoffs.md) · [HLD vs LLD](../HLD/01-foundations/03-hld-vs-lld.md) · [Five-step framework](../HLD/01-foundations/04-five-step-interview-framework.md) · [Capacity estimation](../HLD/01-foundations/05-capacity-estimation-playbook.md) · [Availability & SLOs](../HLD/01-foundations/06-availability-numbers-and-slos.md)
- Other cheat sheets: [Networking](02-networking-cheatsheet.md) · [Scaling, caching & DB](03-scaling-caching-db-cheatsheet.md) · [RAG & LLM](04-rag-and-llm-cheatsheet.md) · [Agentic AI](05-agentic-ai-cheatsheet.md) · [Operationalisation](06-operationalisation-cheatsheet.md) · [**LLD**](07-lld-cheatsheet.md)
- The other round: [What is LLD](../LLD/01-what-is-lld-and-how-to-answer.md) · [machine-coding framework](../LLD/06-machine-coding-interview-framework.md)
- Practice: [Foundations questions](../Interview-Questions/01-foundations-and-capacity-questions.md) · [LLD questions](../Interview-Questions/08-lld-questions.md) · [URL shortener case study](../Case-Studies/HLD/01-url-shortener.md)
