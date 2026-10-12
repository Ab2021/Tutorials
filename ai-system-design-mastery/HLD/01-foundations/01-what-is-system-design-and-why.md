# What Is System Design — And Why Does It Exist?

> **Source of truth:** *System Design Interview Prep | Day 1 — Foundations, Framework & Capacity Estimation* (AI System Design 2026 series), plus the roadmap video *Complete AI System Design Roadmap 2026*.

---

## TL;DR

System design is the discipline of **deciding how to build a large software system so that it keeps working for millions of users without breaking and without slowing down** — and of solving scale problems *at the architecture level, before they happen in production*. In an interview you are not writing code; you are acting as an **architect**, choosing components, drawing the connections between them, and defending every choice with reasoning and numbers. Roughly 60–70% of modern AI-role interview loops now contain a system design round, and performance in that round is often the single biggest determinant of your level and compensation band.

> **60-second interview answer:** "System design is the discipline of solving scale problems before they happen, at the architecture level. Writing code for 10 users is easy; writing it for 10 million concurrent users is very hard, because at scale servers crash under load, databases slow down, data is lost when a machine fails, and two users can update the same record at the same time. Since every system has to balance conflicting goals — scalability, availability, reliability, performance, consistency and cost — the job is never to find the perfect answer, it is to find the *right balance* for the given use case."

---

## 1. Definition & mental model

### The plain definition (as taught in Day 1)

> **System design is simply deciding how to build a large software system so it works for millions of users — without breaking, and without slowing down.**

Three words in that definition carry all the weight:

| Word | What it actually demands |
|---|---|
| **Deciding** | It is a decision-making discipline, not a coding discipline. Deliverable = a diagram + justified choices. |
| **Millions** | Scale is the whole point. Everything that follows (caching, sharding, queues, replicas) exists *because of* scale. |
| **Without breaking / slowing** | The two failure modes you are hired to prevent: **unavailability** (breaks) and **degradation** (slows). |

### The one rule that reframes the entire interview

> **You are the architect, not the builder. You are not writing code here.**

This is stated explicitly in Day 1 and it is the most common reason strong coders fail the round. In a system design interview:

- You **do not** write implementation code (except a short snippet if the interviewer explicitly asks, e.g. "implement an LRU cache").
- You **do** talk about components, protocols, data models, capacity numbers, failure modes and trade-offs.
- You **do** draw: boxes for components, arrows for data flow, labels for the protocol on each arrow (HTTP, gRPC, Kafka topic, SQL, replication stream).

### The mental model that makes it click: *the restaurant*

A single person cooking, taking orders and cleaning tables can serve a handful of customers. Now imagine serving **70 million customers a day, globally** — which is roughly what McDonald's does. No single human, and no single server, can absorb that. The system that makes it possible is built from a small number of repeatable tricks:

| McDonald's | Software equivalent | Why it works |
|---|---|---|
| **Multiple counters** — never one single counter, because *a single counter is the bottleneck* | **Load balancer** in front of many identical servers | Spreads concurrent work across parallel resources |
| **Pre-prepared patties** — food is already ready before you order | **Cache / Redis** — data is in memory before the user asks | Turns a slow read into a fast read |
| **Backup supplier** — if one supplier fails, another covers | **Consistent API contract** — same response format from every server | Any replica can serve any request; no server is irreplaceable |
| **Manager tracking orders** — someone always knows what is in flight | **Monitoring & observability** — every request is traced, every metric is recorded | You cannot fix what you cannot see |

That table is worth memorising. It is the whole of system design compressed into four rows: **parallelise the work, pre-compute the answer, make every node replaceable, and watch everything.**

---

## 2. Why it exists — the problem it solves

### The central asymmetry

> **Writing code for 10 users is an easy task. Writing the same code for 10 million users at the same time is very, very hard.**

The code that works for 10 users is *the same code*. Nothing about the logic is wrong. What breaks is everything **around** the logic: the machine it runs on, the database it queries, the network between them, and the assumptions about who else is doing the same thing at the same time.

### The four canonical problems that appear at scale

These are the four problems Day 1 names explicitly. Every large system hits at least one; most hit all four.

| # | Problem | What it looks like in production | First-line architectural answer |
|---|---|---|---|
| 1 | **The server crashes under too much load** | A single box receives more concurrent requests than it has workers/memory/CPU for. Latency rises, then the process dies, and now *all* traffic is failing. | Horizontal scaling + load balancer + autoscaling + admission control |
| 2 | **The database becomes slow when millions of users query it** | A single primary DB handles ~5,000 read QPS comfortably. At 50,000 read QPS queries queue, connection pools exhaust, p99 explodes. | Cache in front of the DB, read replicas, sharding, query optimisation |
| 3 | **Data gets lost when a machine fails** | The node holding the only copy of a write dies; the write was in memory or on a disk that is gone. Data loss is usually *permanent and unrecoverable*. | Replication, durable write-ahead logging, backups, multi-AZ |
| 4 | **Two users update the same record simultaneously — who wins?** | Lost update, double-booking, double-spend, torn read of a half-applied transaction. | Transactions + isolation levels, optimistic concurrency (version numbers), pessimistic locks, idempotency keys |

> **The definition that follows from these four:** System design is **the discipline of solving these problems *before* they happen, at the architecture level.**

### A fifth problem the source also names: distance

Day 1 lists a fifth failure that the table above does not cover, and it is easy to miss because it is not about load at all:

> *"The users in different countries are seeing slow responses."* (T2, [5:00])

This is **physics, not capacity**. Light in fibre takes roughly 60–70 ms to cross the Atlantic and back, before any processing happens. A user in Singapore hitting a Virginia database pays that round trip *per query*, and no amount of horizontal scaling removes it — adding servers in Virginia makes the problem worse, because now there are more hops in the same distant region.

The architectural answers are all about *moving the data closer to the user*, not the user closer to the data:

| Mechanism | What it moves | Trade it makes |
|---|---|---|
| **CDN / edge cache** | Static and cacheable content | Freshness — edge copies are stale by the TTL |
| **Regional read replicas** | Read traffic | Replication lag; a read-your-own-writes guarantee needs care |
| **Multi-region active-active** | Writes too | Conflict resolution, and the consistency model becomes a design decision |
| **Geo-routing at DNS/LB** | The request itself | Complexity; cross-region failover needs its own story |

**Why it belongs in the foundations file:** it is the cleanest example of a problem where **the fix is a different category of solution, not more of the same**. Scaling is "add more of the thing"; distance is "put the thing somewhere else." Candidates who have internalised only the first reflex will answer "add more servers" to a latency problem that more servers cannot touch — and that is exactly the follow-up question this problem is designed to expose.

**The interview move:** when a requirement mentions a global user base, say the number out loud — *"cross-region round trip is 60–70 ms of pure physics, so if the p99 target is 100 ms, a synchronous cross-region read is already most of my budget."* That converts a vague "we need a CDN" into a derived decision.

That word *before* is the entire job. Anyone can add a cache after the database melts. The architect adds the cache because the arithmetic said the database would melt.

### Why the company is really asking

Day 1 lists the four things the interviewer is actually measuring when they ask you to design something:

1. **Can you think at scale-level, not just for 10 users?**
2. **Can you make trade-offs with proper reasoning?** (Not "I will use Redis" but "I will use Redis *because* read QPS is 50,000 and a single DB handles 5,000, so the cache absorbs 90% of reads.")
3. **Can you communicate clearly under pressure?** The interview is a conversation, not a monologue.
4. **Do you know the right tools for the right job?** Postgres vs Cassandra vs Redis vs Kafka — each has a reason to exist.

And the business consequence, stated bluntly in the roadmap video:

> System design performance is the **main deciding factor for your salary band**. A candidate with *less* experience but a strong design process can be offered a better band than a candidate with *more* experience and a weak design process.

---

## 3. Who asks this, and why it spread beyond backend roles

System design used to be limited to software-developer profiles. That is no longer true.

| Company category | Examples | What they probe |
|---|---|---|
| **Frontier AI labs** | OpenAI, Anthropic, Google DeepMind, Meta AI, xAI | Depth of knowledge plus "evaluate how you reason" — they want your *thinking*, not a memorised diagram |
| **FAANG / big tech** | Google, Amazon, Apple, Microsoft, Netflix, Meta | The classic architecture-design process: requirements → estimation → HLD → deep dive |
| **AI-first startups & unicorns** | Fireworks, Cursor, LangChain, Perplexity, Sierra | Applied AI system design: RAG, agents, inference cost, guardrails |

If you are switching into an **AI profile** — data scientist, ML engineer, GenAI engineer, agent engineer — you must still learn system design. Modern loops include *both* a classical system design question and an AI-specific one, frequently in the same round.

**Scale of the change:** 60–70% of modern AI interviews now include generative-AI-specific design questions. The examples named in the roadmap video:

- *Design a system for an LLM responding to a user query.*
- *Design a customer support chatbot on top of an LLM.*
- *Design a safeguard for an AI agent that takes actions for a user.*
- *Distribute a huge model weight file across thousands of machines.*

> **None of these have a "perfect" answer.** The interviewer is watching the *process*. That is exactly why the next file exists — a fixed five-step framework that turns any ambiguous prompt into a structured, defensible answer.

---

## 4. The two shapes of system design question

Day 1 makes a hard distinction that candidates routinely blur. You must know which one you are being asked.

### 4.1 High-Level Design (HLD)

> **The big picture: which components exist, and how they are connected.**

You draw boxes and arrows. You decide *what* exists — API gateway, load balancer, service, cache, queue, database — and *how* they talk (HTTP, gRPC, Kafka, replication).

**Worked example from the transcript — "Design Instagram":**

```
User phone ──▶ API Gateway ──▶ User/Upload service ──▶ S3 bucket (media)
                    │
                    └────────▶ Feed service ──▶ Redis cache ──▶ DB
```

That is HLD. Different components, connected with particular arrows, each arrow labelled with a protocol. Nothing about classes or methods.

### 4.2 Low-Level Design (LLD)

> **The internals of one component: classes, methods, interfaces, data structures — the object-oriented thinking.**

**Worked example from the transcript — "Design a parking lot":**

```java
class ParkingLot {
    List<Floor> floors;
    ParkingSpot findAvailableSpot(VehicleType type);
    Ticket       reserveSpot(ParkingSpot spot, Vehicle vehicle);
    void         releaseSpot(Ticket ticket);
}

class Floor {
    List<ParkingSpot> spots;
}
```

Now you *are* deciding class boundaries, method signatures, and how the objects interact — the OOP thinking you already use when you write a class, a constructor and its methods.

### 4.3 HLD vs LLD at a glance

| Dimension | HLD | LLD |
|---|---|---|
| Question it answers | *What components exist and how do they connect?* | *How is one component implemented?* |
| Artefact | Architecture diagram, capacity numbers, API contracts | Class diagram, interfaces, method signatures, DB schema |
| Typical prompt | "Design Twitter", "Design a URL shortener" | "Design a parking lot", "Implement an LRU cache" |
| Scale focus | Millions of users, multiple machines, network | Concurrency, complexity, extensibility, correctness |
| Failure focus | Node failure, region failure, thundering herd | Race conditions, deadlock, memory leaks, null handling |
| Typical interviewer | Senior/staff engineer, EM | Senior engineer, paired-coding style |

**The exam-day rule:** if the prompt is a *product* ("design WhatsApp"), you are in HLD territory — start with requirements and capacity. If the prompt is a *mechanism* ("design a seat-locking service", "implement LFU"), you are in LLD territory — start with the data structures and the concurrency model.

---

## 5. Pros / Cons

### Pros of learning system design as a discipline

| Pros | Cons / honest caveats |
|---|---|
| It is the highest-leverage interview round for level and compensation | It is *not* memorisable — memorised architectures are detected within two follow-up questions |
| The vocabulary transfers across backend, ML, data and now GenAI roles | Depth required is broad: networking, storage, distributed systems, and now LLM serving |
| The framework forces clarity — you stop rambling and start structuring | Real fluency takes practice; reading is not enough (the course's own advice: draw 10–15 designs by hand) |
| Trade-off reasoning is genuinely useful on the job, not just in interviews | "Best practice" lists go stale fast (e.g. "always use MongoDB" advice aged badly) |
| Numbers (capacity estimation) make your arguments unfalsifiable-proof | Estimation is approximate; being paralysed by precision is a common failure |

### Pros / Cons of the *interview format itself*

| Pros | Cons |
|---|---|
| Open-ended by design — rewards structured thinking over trivia | No single correct answer makes feedback hard; candidates often don't know why they failed |
| Simulates real architecture review conversations | Time-boxed (35–60 min) which punishes over-exploration |
| Lets strong candidates with less experience demonstrate seniority | Punishes silent thinking — you must narrate |

---

## 6. Exceptions, edge cases & failure modes

These are the ways this *topic itself* — the framing of system design — goes wrong.

| Exception / failure mode | What happens | How to handle it |
|---|---|---|
| **"Design Twitter" is really 100 different systems.** | You guess the wrong scope and waste the entire interview building something the interviewer did not want. | Never start drawing. Ask clarifying questions first — see [the five-step framework](04-five-step-interview-framework.md). |
| **The prompt is deliberately ambiguous.** | Candidates freeze or pick arbitrarily. | Ambiguity is the *point*. Reduce it with questions, then state the scope you are assuming out loud. |
| **The interviewer says "scale it" mid-answer.** | You panic and bolt on a load balancer with no justification. | Have the estimation numbers ready so scaling decisions are *derived*, not asserted. |
| **It's a fake-AI question.** ("Design an LLM chatbot" = actually just a stateless API + vector DB + cache.) | You over-engineer with agents and fine-tuning. | Ask "what is the actual hard part here?" — often the answer is retrieval quality or cost, not the model. |
| **The interview is LLD but you deliver HLD.** | You draw an architecture diagram for "design a parking lot" and never define a class. | Detect the shape early (§4). Ask: "do you want the architecture, or the class-level design?" |
| **You optimise the wrong constraint.** | You add sharding to a system whose bottleneck is *write contention on a single row*, not volume. | Identify the bottleneck before choosing the fix. Name it: "the bottleneck here is X, so I'd…" |
| **"Perfect answer" trap.** | You chase an optimal design and never converge, so the interviewer sees no completed architecture. | Day 1, verbatim: *"In an interview you are never looking for the perfect answer. You are looking for the right balance for the given use case."* |
| **Interviewer's company context is ignored.** | You design like it's a greenfield startup when they said "we have an existing system". | Day 1's notification-system example explicitly asks: *new system, or extending an existing one?* Always clarify. |

### The genuine exception to "you are the architect, not the builder"

Some FAANG rounds (and most LLD rounds) **do** expect code. The distinction:

- **HLD round:** no code. Diagram + numbers + trade-offs.
- **LLD round:** class/method design, possibly pseudocode, possibly a real implementation of a data structure (LRU cache is the classic).
- **Coding round embedded in a design loop:** real code, but scoped to a component (the seat-locking function, the rate limiter).

Prepare for all three. The "architect not builder" rule applies to HLD, not universally.

---

## 7. When to use / When NOT to use

### When the five-step framework applies (i.e. almost always)

| Situation | Use the framework? | Notes |
|---|---|---|
| "Design <product>" | ✅ Fully | Requirements → estimation → API → HLD → deep dive |
| "How would you scale <existing system>?" | ✅ Partially | Skip API design; go requirements → *current* bottleneck → estimation → changed components |
| "Design the schema for <X>" | ⚠️ Compressed | Requirements → access patterns → schema → indexes → scaling |
| "Implement LRU cache" | ❌ Not really | This is LLD: clarify semantics → data structures → complexity → concurrency → tests |
| "What is CAP theorem?" | ❌ No | This is a knowledge question, not a design question — answer it and bridge to a design example |
| "Design an ML/LLM system" | ✅ With additions | Add: data pipeline, training/eval loop, model serving, drift, cost per inference |

### When NOT to use a heavy architecture

Not every problem deserves a distributed system. Recognising this *is* senior signal.

- **Low traffic + strong consistency + simple queries** → one Postgres instance, one app server, a nightly backup. Adding Kafka "for future scale" is a liability: more failure modes, more on-call, more cost.
- **Internal tool, <100 users** → a monolith and a managed DB. No microservices.
- **The bottleneck is organisational, not technical** → no amount of sharding fixes an unclear product requirement.
- **You have not measured** → premature optimisation. Day 1 pushes estimation hard precisely because *"without estimation, the interviewer immediately asks a cross-question: why do you need a cache?"*

> **Rule of thumb to state out loud:** "For the current scale of X QPS, a single primary with a read replica is sufficient. I'd add sharding only when write QPS exceeds what one primary can absorb — and I'd flag that as a milestone trigger, not a day-one decision."

---

## 8. Numbers, defaults & tuning

These are the anchors from Day 1 that make the whole discipline quantitative. Full derivations live in [05-capacity-estimation-playbook.md](05-capacity-estimation-playbook.md).

| Number | Value | Why it matters |
|---|---|---|
| Seconds in a day | **86,400 ≈ 10⁵** | The divisor for every QPS calculation. Use 10⁵ for mental math. |
| 1 million requests/day | ≈ **12 QPS** | The single most useful conversion in the whole interview. |
| 1 billion requests/day | ≈ **12,000 QPS** | 1B ÷ 10⁵. |
| Peak multiplier | **3×** average QPS | Traffic is not uniform; interviews expect a peak factor. |
| Comfortable read QPS per single DB | **~5,000** | The justification anchor used in the transcript's "why a cache" answer. |
| Storage sanity check | 1 KB × 1M records = **1 GB** | Lets you convert record counts into bytes instantly. |
| Availability 99% | **3.65 days/year** downtime | "Two nines" is weak. |
| Availability 99.9% | **8.77 hours/year** | "Three nines" = standard consumer app. |
| Availability 99.99% | **52 min/year**, ~26 s/month | Gmail-class. Explicitly cited in Day 1. |
| Availability 99.999% | **~5 min/year** | Payments/telecom class; very expensive. |

---

## 9. Interview Q&A

**Q1. What is system design, in one sentence?**
Deciding how to build a large software system so that it works for millions of users without breaking and without slowing down. It is a decision-making and architecture discipline, not a coding exercise.

**Q2. Why is coding for 10 users easy but for 10 million users hard?**
The business logic is identical; what changes is the environment it runs in. At scale you hit four problems: servers crash under concurrent load, a single database saturates under millions of queries, data is lost when a machine fails, and concurrent writers to the same record conflict. System design is the discipline of solving those before they happen.

**Q3. What is the difference between HLD and LLD?**
HLD decides *which components exist and how they connect* — boxes and labelled arrows, plus capacity numbers and API contracts. LLD decides the *internals of one component* — classes, methods, interfaces and data structures, using OOP thinking. HLD answers "design Instagram"; LLD answers "design a parking lot".

**Q4. Why do companies ask system design instead of just more coding rounds?**
Because coding rounds cannot test scale reasoning, trade-off justification, communication under ambiguity, or whether you know which tool fits which job. System design is the closest interview analogue to the architecture-review conversations you will actually have on the job.

**Q5. Is there a "perfect" answer to a system design question?**
No — and chasing one is a mistake. Every design balances six conflicting goals. You are looking for the *right balance for the given use case*. Two candidates can give opposite answers and both be right, if they justify them against different requirements.

**Q6. A candidate immediately draws an architecture. What's wrong with that?**
They skipped requirement clarification and probably guessed the wrong scope. "Design Twitter" could mean the timeline, the tweet ingestion path, the follow graph, the search, or the notification fan-out — five different systems. If you guess wrong you waste the interview. Always ask first.

**Q7. How much coding should I write in a system design interview?**
In an HLD round: none, unless explicitly asked. In an LLD round: class/method signatures and pseudocode are expected, and you may be asked to implement a data structure such as an LRU cache. Ask "do you want the architecture or the class-level design?" to find out which round you're in.

**Q8. What is the single most important habit that separates strong candidates?**
Justifying every component with a number or a requirement. "I'll use Redis as a cache" invites the cross-question "why?". "Read QPS is 50,000 and a single database handles about 5,000, so a cache in front absorbing 90% of reads is necessary" ends the debate.

**Q9. How do you handle a prompt you've never seen before?**
With the framework rather than with memory: clarify functional and non-functional requirements, estimate capacity, define the API, draw the HLD, then deep-dive two areas and state the trade-offs. The framework is deliberately problem-agnostic.

**Q10. Why has system design spread from backend roles to AI roles?**
Because AI systems are systems. A production RAG pipeline has ingestion queues, vector indexes, caches, gateways, fallbacks and cost budgets. AI-first companies now run a full system design round, and 60–70% of modern AI interviews include a generative-AI-specific design question.

**Q11. *Pushback:* "You said you're an architect, not a builder — so you can't code?"**
No — the distinction is about the *deliverable of this round*, not your ability. In HLD the deliverable is a design; writing code there wastes the time budget. In LLD and coding rounds I write real code. I'd rather show design judgement here and code there.

**Q12. *Pushback:* "Isn't system design just memorising architectures for common products?"**
Memorised architectures fail at the first follow-up question because they're not tied to requirements. What transfers between problems is the *method* — requirements, estimation, API, HLD, trade-offs — plus a small set of reusable primitives: load balancer, cache, queue, replica, shard, circuit breaker. Products change; primitives persist.

---

## 10. Common mistakes candidates make

1. **Starting to draw immediately.** Skipping requirement clarification is the #1 failure and it is explicitly called out in Day 1.
2. **Thinking like a builder, not an architect.** Diving into implementation detail in an HLD round.
3. **Asserting components with no justification.** "I'll use Kafka" with no answer to "why not a database table?".
4. **Ignoring the four canonical failure modes.** Never mentioning what happens when a node, a region, or a provider dies.
5. **Treating non-functional requirements as optional.** Latency, availability and consistency targets are what make the design decidable.
6. **Not writing numbers down.** Estimation is a *visible* artefact; doing it silently scores nothing.
7. **Confusing HLD with LLD.** Delivering a class diagram for "design Uber" or an architecture diagram for "design a parking lot".
8. **Designing for 10 million users when the interviewer said 10,000.** Over-engineering is as wrong as under-engineering.
9. **Never converging.** Running out of time with no complete architecture on the board.
10. **Not stating trade-offs.** Every decision has a cost; naming the cost is what makes it a decision rather than a preference.

---

## 11. Cheat-sheet summary

| Concept | One-line answer |
|---|---|
| **System design** | Deciding how to build a large software system that works for millions of users without breaking or slowing. |
| **Your role** | Architect, not builder. (In HLD; in LLD you do design classes.) |
| **Core asymmetry** | 10 users = easy. 10M users = very hard. Same logic, different environment. |
| **Four scale problems** | Server crash under load · DB slow under millions of queries · data lost on machine failure · concurrent updates to the same record. |
| **True definition** | The discipline of solving those problems *before* they happen, at the architecture level. |
| **Four-scale-answer analogy** | McDonald's: many counters → load balancer; pre-made patties → cache; backup supplier → consistent API; manager tracking orders → monitoring. |
| **HLD** | Components + connections + labelled arrows. |
| **LLD** | Classes + methods + data structures + interfaces. |
| **Six conflicting goals** | Scalability · Availability · Reliability · Performance · Consistency · Cost. |
| **Interview truth** | Never the perfect answer — the *right balance for the given use case*. |
| **Who asks** | Frontier labs (reasoning depth), FAANG (design process), AI startups (applied AI design). |
| **Salary impact** | Design performance is the main deciding factor for level and band. |
| **Key conversions** | 86,400 s/day ≈ 10⁵ · 1M req/day ≈ 12 QPS · peak = 3× · 1 DB ≈ 5,000 read QPS. |
| **Availability anchors** | 99% = 3.65 d/yr · 99.9% = 8.77 h/yr · 99.99% = 52 min/yr. |

---

## 12. Cross-links

- Next: [The six goals and their trade-offs](02-six-goals-and-tradeoffs.md) — what you are actually optimising.
- Then: [HLD vs LLD](03-hld-vs-lld.md) → [The five-step interview framework](04-five-step-interview-framework.md) → [Capacity estimation playbook](05-capacity-estimation-playbook.md) → [Availability numbers & SLOs](06-availability-numbers-and-slos.md).
- Worked example in the course's house style: [URL shortener case study](../../Case-Studies/HLD/01-url-shortener.md).
- The primitives referenced above: [Load balancers](../03-scaling/01-load-balancers.md) · [Caching](../03-scaling/02-caching-strategies-and-layers.md) · [Redis](../03-scaling/03-redis-deep-dive.md) · [Sharding & replication](../03-scaling/05-sharding-replication-and-consistency.md).
- Keeping it alive in production: [Operational readiness checklist](../../Operations/01-operational-readiness-checklist.md) · [Observability](../../Operations/02-observability-and-monitoring.md).
