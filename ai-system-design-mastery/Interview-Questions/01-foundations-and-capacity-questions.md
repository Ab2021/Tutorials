# Interview Questions — Foundations, Frameworks, Capacity & Availability

> **Sources:** *AI System Design Roadmap 2026* · *System Design Interview Prep Day 1 — Foundations, Framework & Capacity Estimation* · *Most Asked FAANG System Design Question (URL shortener)*.

> **How to use this file:** cover the answer, say yours out loud, then compare. The model answers are written the way you'd actually speak them — first person, with the number or the trade-off stated. Anything marked **⚡** is a question that separates senior from mid-level candidates.

---

## Part A — What system design is (and what it isn't)

### Q1. What is system design?
Deciding how to build a large software system that works for millions of users without breaking or slowing down. It's architecture, not implementation — you're deciding which components exist, how they connect, and what you're trading away.

### Q2. ⚡ What's the difference between a system designer and a developer in this interview?
A developer asks "how do I write this function?" A system designer asks "which components, what protocols, how does it fail, what does it cost?" Day 1 puts it plainly: *you are the architect, not the builder — you are not writing code here.*

### Q3. Why does a system break as it grows? Name the four problems.
(1) The server crashes under load. (2) The database becomes slow under millions of queries. (3) Data is lost when a machine fails. (4) Two users update the same record at the same time and one write disappears.

### Q4. Give the McDonald's analogy for system design.
Multiple counters → **load balancer** (spread customers across servers). Pre-prepared patties → **cache/Redis** (avoid the expensive work on every request). Backup supplier → **consistent API** (a stable contract that hides internal change). Manager tracking orders → **monitoring/observability** (know what's happening in real time). Same product, same demand, but organised so it doesn't collapse at noon.

### Q5. What's the difference between high-level and low-level design?
HLD is components and connections — boxes with labelled arrows: client, load balancer, service, cache, database, queue. LLD is inside one box — classes, methods, interfaces, data structures and their complexity. HLD answers "what are the parts"; LLD answers "how does this part work."

### Q6. Who asks system design questions?
Frontier AI labs (OpenAI, Anthropic, Google DeepMind, Meta AI, xAI), FAANG, and AI startups. Roughly 60–70% of modern AI interviews include GenAI system design questions.

### Q7. ⚡ Why does system design matter more than coding for your salary band?
Because system design performance is the main determinant of the level you're hired at, and level determines the band. A candidate who writes flawless code but can't defend an architecture gets a mid-level offer; a candidate who designs and reasons about trade-offs gets a senior one.

### Q8. What are the four things you must always state in a design?
Requirements (functional and non-functional), the numbers (capacity), the contracts (API), and the trade-offs (what you gave up). A design missing any of the four is incomplete.

### Q9. What does "it works at scale" actually mean?
It means the system survives 10× growth in users, traffic and data without redesign, degrading gracefully rather than failing — and at a cost that still makes sense.

### Q10. What's the difference between scaling up and scaling out?
Scaling up (vertical) means a bigger machine — simple, no code change, but a hard ceiling and a single point of failure. Scaling out (horizontal) means more machines — no ceiling, needs statelessness and load balancing, and introduces coordination problems.

---

## Part B — The six goals and their conflicts

### Q11. Name the six goals of a system.
Scalability, availability, reliability, performance, consistency and cost.

### Q12. ⚡ Why can't you maximise all six?
Because they conflict pairwise. Strong consistency costs latency and availability (CAP). High availability costs money. Aggressive caching buys performance and costs consistency. Every real design is a *selection*, and the selection is the answer.

### Q13. Map a product to the goal it should lead with: WhatsApp.
Reliability and availability. Losing a message is unacceptable and the service must always be reachable. Consistency can be eventual — a message arriving a second late is invisible.

### Q14. …a stock exchange?
Consistency. A trade executed against a stale price is a real financial loss. You sacrifice latency and availability (it's acceptable to halt trading) rather than allow divergent state.

### Q15. …an Instagram feed?
Performance. The feed must feel instant. Eventual consistency is perfectly acceptable — seeing a post one second late costs nothing.

### Q16. What's the difference between scalability and performance?
Performance is how fast one request is *now*. Scalability is how well performance holds as load grows. A system can be fast and unscalable (a single well-tuned server) or slow and scalable.

### Q17. What's the difference between reliability and availability?
Availability is "can I reach it right now?" Reliability is "does it do the right thing consistently over time?" A service returning stale data is available but unreliable; a service that's down but never served a wrong answer is reliable but unavailable.

### Q18. What's the availability formula?
`Availability = MTBF ÷ (MTBF + MTTR)` — mean time between failures divided by mean time between failures plus mean time to repair.

### Q19. ⚡ At scale, would you invest more in MTBF or MTTR?
MTTR. Failures are inevitable — hardware dies, dependencies fail, every deploy is a risk. You cannot make failure rare enough to reach four nines, but you can make recovery fast enough. Automation, health checks, canaries and rollback beat trying to eliminate failure.

### Q20. What's the consistency spectrum?
Strong (linearizable) → sequential → causal → read-your-writes → monotonic reads → eventual. Most systems don't need strong consistency globally; they need it on *specific paths* — a balance update, a seat reservation, a custom alias claim.

### Q21. What's PACELC and how does it extend CAP?
CAP says during a **P**artition you choose **A**vailability or **C**onsistency. PACELC adds: **E**lse (no partition), you still trade **L**atency against **C**onsistency. It matters because most of the time there is no partition, and the latency/consistency trade-off is the one you live with daily.

### Q22. What are the two most common misreadings of CAP?
(1) That you pick two of three permanently — actually CA is impossible in a distributed system, so the real choice is C-or-A *during a partition*. (2) That CAP applies to a whole system — it applies to a specific operation on a specific data item.

### Q23. Give three ways to reduce cost without hurting users.
Cache hit rate (every hit is a database query you didn't pay for), tiered storage (hot vs cold data), and right-sizing (autoscaling instead of provisioning for peak). Prompt caching is the LLM equivalent of the first.

### Q24. What's tail latency and why does it amplify?
p99 latency is the slowest 1% of requests. It amplifies because a request that fans out to 100 services waits for the slowest of the 100 — so if each service has a 1% chance of being slow, the probability that *at least one* is slow is nearly 63%, not 1%. Tail latency is a fan-out problem.

### Q25. What's the single most important trade-off in caching?
Freshness vs latency and load. A cache buys a large latency and cost win in exchange for staleness — so the question is never "should we cache" but "how stale can this specific path tolerate being?"

---

## Part C — HLD vs LLD

### Q26. What goes in an HLD diagram?
Client, entry point (DNS/CDN/LB), API gateway, services, cache, database with read/write split, queues, and external dependencies — with the protocol labelled on every arrow.

### Q27. What goes in an LLD?
Classes, methods, attributes, interfaces, data structures, inheritance, and the time/space complexity of the key operations.

### Q28. ⚡ Walk me through an LRU cache in LLD terms.
A hash map plus a doubly linked list. The map gives O(1) lookup of a node; the linked list maintains recency order, with the most recently used at the head and the least recently used at the tail. `get` moves the node to the head — O(1). `put` inserts at the head, and if capacity is exceeded, evicts the tail — O(1). Both operations are O(1) because the map holds a direct pointer to the list node.

### Q29. Why do you need the doubly linked list if you have the hash map?
The map alone can't tell you which key is least recently used. You need an ordering structure, and you need removal from the middle in O(1) — which requires the backward pointer, hence doubly linked.

### Q30. How do you tell which round you're in?
If you're asked for classes, methods, interfaces or complexity, it's LLD. If you're asked for components, protocols, scaling or trade-offs, it's HLD. Signal phrases: "implement", "design a class", "what data structure" → LLD. "Design Twitter", "how would you scale", "what happens at 10×" → HLD.

### Q31. What's the boundary question for a URL shortener?
HLD: generate a short code, redirect, cache, store, analytics. LLD: the encoding algorithm, the collision-handling function signature, the cache class. The service boundary is HLD; the function boundary is LLD.

### Q32. What's the boundary for "design an AI agent"?
HLD: the orchestration loop, tool registry, memory store, model gateway, guardrails, evaluation. LLD: the ReAct loop implementation, the tool interface (name, description, JSON schema, handler), the context-window packing function.

### Q33. What's the time split in a 45-minute interview?
Roughly 6 minutes clarifying, 5 estimating, 5 on the API, 15 on the HLD, and 14 on deep dives and trade-offs. For an LLD round, most of the time goes on entities, interfaces, and then coding + edge cases.

---

## Part D — The five-step framework

### Q34. What are the five steps?
Clarify requirements → capacity estimation → define the API → high-level design → deep dive and trade-offs.

### Q35. Why can't you start drawing?
Because "design Twitter" could mean a hundred different systems, and guessing the wrong scope wastes the entire interview. Clarifying first also shows the interviewer how you think before you act.

### Q36. What are the two categories of requirement?
Functional — what the system does, user-facing, and explicitly what's *out* of scope. Non-functional — how well it must perform, stated in technical and measurable terms: DAU, latency, durability, geography, availability.

### Q37. ⚡ Give a clarifying question that changes the whole architecture.
For a notification system: "Is delivery speed critical, or is a few seconds' delay acceptable?" Seconds acceptable → a durable queue with batching: cheaper, more resilient. Sub-second → persistent connections and a different delivery path. Same product name, different system.

### Q38. Why is capacity estimation worth five minutes?
Because it makes architecture defensible. Without it, "I'll add a cache" invites "why?" and you have nothing. With it — "read QPS is 50,000 and a DB handles 5,000, so a cache is required to absorb 90% of reads" — the number ends the debate.

### Q39. Why define the API before drawing components?
It forces clarity about what the system actually does, and it shows you think in interfaces rather than boxes. It also surfaces requirements you missed — if you can't write the endpoint cleanly, the data model behind it is unclear.

### Q40. What's the structure of a good trade-off statement?
Four parts: the specific choice, the numeric reason, the benefit, and the cost. *"I'll use Redis as a cache because read QPS is 50,000 and the DB handles 5,000; it absorbs 90% of reads and cuts p99 from 50 ms to 5 ms, at the cost of up to a second of staleness, which is invisible on a feed."*

### Q41. What does "go deeper" mean?
The interviewer wants depth on one or two areas — usually scale, failure modes, consistency, cost or alternatives. Have three areas prepared and know the numbers for each.

### Q42. ⚡ How do you close a design?
State what you sacrificed and what would change it. *"This design trades read staleness for a large latency and cost win and accepts eventual consistency in analytics. If consistency were tightened, I'd drop the cache on that path, use quorum reads, and accept a higher p99."*

### Q43. How does the framework change for an LLM system?
Steps 1–3 are identical. Step 2 adds token-cost and embedding-storage estimation. Step 4 adds retrieval, an inference/gateway layer, guardrails and an evaluation pipeline. Step 5 adds prompt/model/index versioning, and **cost per query** becomes a first-class trade-off alongside latency and quality.

---

## Part E — Capacity estimation

### Q44. Give the template.
Requests/day = DAU × requests per user per day. QPS = requests/day ÷ 86,400. Peak = QPS × 3. Storage = requests/day × record size. Bandwidth = QPS × record size.

### Q45. What's the most useful conversion?
One million requests per day ≈ 12 QPS. It converts product-scale numbers into hardware-scale numbers instantly.

### Q46. Why divide by 86,400?
Because there are 86,400 seconds in a day — so requests/day ÷ seconds/day gives requests/second. I use 10⁵ for mental math and say so.

### Q47. Why multiply by 3 for peak?
Traffic isn't uniform — it has a diurnal shape. Designing for the average means designing for failure. I state the 3× explicitly so it can be challenged.

### Q48. Estimate a messaging system: 50M DAU, 10 messages each, 1 KB per message.
500M messages/day → 5,000 write QPS → 15,000 at peak. Storage 500 GB/day → 180 TB/year. Bandwidth ~5 MB/s average. Conclusion: write throughput and storage are the bottlenecks → a write queue and a sharded store are required, because a single primary saturates at peak.

### Q49. ⚡ Why do you include replication in the storage number?
Because replication is not free. 180 TB/year raw becomes ~270 TB with index overhead at 1.5× and ~820 TB with three-way replication. That changes the answer from "one big node" to "a sharded cluster with a tiering policy."

### Q50. Estimate a URL shortener with 100M creations/day and a 100:1 read ratio.
1,000 write QPS (3,000 peak); 10B reads/day → 100,000 read QPS → 300,000 at peak. That's 60× a database's capacity, so caching isn't optional — the design is a cache in front of a key-value store.

### Q51. What's typically the bottleneck?
Usually not bandwidth. The real bottlenecks are write throughput, fan-out volume, storage growth, hot keys, or the tail latency of a dependency chain.

### Q52. How do you estimate a system with fan-out?
Estimate the *derived* write count. A feed with 100M posts/day and 200 average followers produces 20B timeline writes/day — 200,000 QPS, which is 20–40× a database's capacity. That number is the argument for hybrid push/pull fan-out and an async queue.

### Q53. ⚡ What are the three assumptions that most often wreck an estimate?
Forgetting fan-out (the write count is rarely what the input suggests), forgetting replication and index multipliers (understates storage 3–6×), and designing for the average instead of peak.

### Q54. What's a server count estimate?
Peak QPS ÷ per-server capacity, ×1.5 for headroom. At 300,000 peak read QPS and ~1,000 RPS per app server, that's ~300 servers — but a cache absorbing 90% cuts the app-tier requirement, which is the more important architectural point.

### Q55. How do you estimate an LLM system?
Requests first, then two extra axes: tokens per request, which sizes the GPU fleet, and cost per query, which becomes a first-class design constraint. KV-cache memory grows linearly with concurrency and context length; embedding storage is chunks × dimensions × 4 bytes plus 20–100% for the index; prompt caching can remove 50–90% of input token cost.

### Q56. *Pushback:* "Your numbers are off by 2×."
Probably — they're order-of-magnitude estimates with stated assumptions. What matters is whether the conclusion is robust. At 2× or 0.5×, does the design change? If read QPS is 100,000 or 50,000, the cache is still mandatory. If it were 4,000, it wouldn't be — so let me state the crossover, which is the useful engineering answer.

### Q57. When would you skip estimation?
When the system is obviously small — an internal tool at thousands of requests a day — I'd say so in one sentence and spend the time on the interesting part. Also if time is short: state DAU and QPS in one breath and draw.

### Q58. Estimate storage for 500M photos/day at 200 KB each, 3× replicated, 3-year retention.
500M × 200 KB = 100 TB/day. × 365 = 36.5 PB/year. × 1.5 index/metadata ≈ 55 PB. × 3 replicated ≈ 164 PB/year. × 3 years ≈ **~490 PB**. Conclusion: this is a blob-storage problem, not a database problem, and the retention policy is a cost decision worth hundreds of petabytes.

---

## Part F — Availability, SLOs and failure

### Q59. State the nines.
99% = 3.65 days/year. 99.9% = 8.77 hours/year. 99.99% = 52.6 minutes/year (~26 seconds/month). 99.999% = 5.26 minutes/year.

### Q60. ⚡ Why is 99.99% fundamentally different from 99.9%?
52.6 minutes a year is about 26 seconds a month — below human reaction time for detect, page, log in and act. Four nines is only achievable with automated detection and failover. It's an automation target, not a diligence target.

### Q61. How does availability compose across services?
In series it multiplies: three 99.9% services in a request path give 99.7%. In parallel it improves — `1 − (1 − A)ⁿ` — but only if failures are independent.

### Q62. What's the availability of ten 99.9% services in series?
99.0% — a full 3.65 days of downtime a year instead of 8.77 hours. This is why aggressive microservice decomposition without degradation paths destroys reliability.

### Q63. ⚡ Why doesn't adding replicas always help?
Because redundancy only works across independent failure domains. Two replicas in the same AZ, sharing a rack, a config, a deploy pipeline or an upstream database fail together. Redundancy is only as good as the independence of its failure domains.

### Q64. Availability of: LB 99.99%, two app servers at 99.9%, multi-AZ DB at 99.95%?
App tier in parallel = 1 − 0.001² = 99.9999%. Series: 0.9999 × 0.999999 × 0.9995 ≈ 99.94%. The conclusion is that the **database is the floor** — a third app server buys nothing, because the series product is dominated by the weakest link.

### Q65. What are SLI, SLO and SLA?
SLI is the measured indicator ("% of requests under 200 ms"). SLO is the internal objective ("99.9% over 30 days"). SLA is the customer contract with penalties. The SLO must be stricter than the SLA so you get early warning before a contractual breach.

### Q66. What's an error budget and why does it exist?
The downtime an SLO permits — 0.1% of 30 days is about 43 minutes. It converts reliability into a currency: with budget, ship faster; with the budget spent, freeze changes and invest in reliability. It stops reliability and velocity from being an argument.

### Q67. What is MTTR and why does it matter more than people think?
Mean time to repair. At scale it's usually the cheaper lever, because you can't eliminate failure but you can shrink it. To reach four nines with an MTBF of 1,000 hours, MTTR must be under six minutes — which means automation, not heroics.

### Q68. What's a cascading failure and how do you stop it?
One slow dependency exhausts thread pools across the fleet until everything dies. Stop it with circuit breakers, timeouts, bulkheads, load shedding, and retry budgets with jitter.

### Q69. What's a retry storm?
Clients retrying a struggling service multiply load 3–5× at exactly the moment it's weakest, turning a partial failure into a total one. Mitigation: exponential backoff with jitter, retry budgets, and server-side shedding.

### Q70. What's the availability ceiling if your LLM provider is 99.5%?
99.5%, unless the dependency has a fallback. For a critical AI path I'd add a second provider, a smaller local model for degraded mode, or a queue-and-replay path.

### Q71. What's the difference between availability and durability?
Availability is whether you can reach the data now. Durability is whether it survives. S3-class storage is 99.99% available and 99.999999999% durable — different nines for different things, and they need separate SLOs.

### Q72. *Pushback:* "Isn't 99.99% always better?"
No — it costs roughly 10× and only matters if downtime is expensive. For an internal dashboard the right answer is 99% and the money belongs elsewhere. Set the target from the cost of downtime: buy the next nine only when the downtime avoided is worth more than the yearly cost of buying it.

### Q73. What determines whether your failover actually works?
Whether it's *tested*, and whether the failure domains are truly independent. DNS TTL matters too — failover isn't complete when clients still cache the old IP for minutes. And most outages are change-induced, so the deploy path is part of the availability design.

---

## Part G — Estimation drills (cover the answers)

| # | Problem | Answer |
|---|---|---|
| G1 | 10M DAU, 5 requests/user/day | 50M req/day → **~500 QPS**, 1,500 peak |
| G2 | 1B requests/day | **~12,000 QPS**, 36,000 peak |
| G3 | 1M records × 2 KB | **2 GB** |
| G4 | 100M users × 10 KB profile | **1 TB** |
| G5 | 5,000 QPS × 1.5 KB | **7.5 MB/s** |
| G6 | 10B reads/day | **~100,000 QPS** |
| G7 | 1 TB/month sustained | **~0.4 MB/s** |
| G8 | 50M DAU × 20 reads/day | 1B reads/day → **~10,000 QPS**, 30,000 peak |
| G9 | 200M photos/day × 300 KB | **60 TB/day** → 22 PB/year |
| G10 | 500 GB/day, 3 replicas, 1 yr | 180 TB × 3 = **~540 TB** |
| G11 | 300,000 peak QPS, 1,000 RPS/server | **300 servers**, 450 with headroom |
| G12 | 10M concurrent WebSocket connections | **Connections problem, not QPS** — 10M ÷ ~100k/node ≈ 100 nodes |
| G13 | 1M chunks × 1536 dims × 4 bytes | **~6 GB** raw; ~1.5 GB at int8 + index overhead |
| G14 | 99.9% over 30 days | **~43.8 minutes** of budget |
| G15 | Ten services at 99.9% in series | **99.0%** — 3.65 days/year |
| G16 | Two components at 99% in parallel | **99.99%** |
| G17 | 1M LLM requests/day × 2,000 tokens | 2B tokens/day → **~23,000 tokens/sec** average |
| G18 | Cache hit rate 95% at 100k QPS | DB sees **5,000 QPS** — exactly at capacity, so the cache *is* the design |

---

## Part H — Communication and behaviour

### Q74. What do you do when you don't know something?
Say so and reason from what I do know. *"I haven't used that specific system, but the access pattern is a key lookup at this volume, so I'd reach for X, and I'd validate Y by benchmarking."* Bluffing gets exposed by the next follow-up.

### Q75. What if the interviewer is silent?
Silence is usually evaluation, not disapproval. Narrate everything and check in every few minutes: "Shall I go deeper here or move on?"

### Q76. ⚡ What if you realise mid-design that an early assumption was wrong?
Say it out loud and correct it: "Revisiting my assumption about X, I need to change Y — here's what that affects." Recovering visibly is a positive signal; quietly papering over it is not.

### Q77. What if the interviewer changes the constraints halfway?
Re-run only the delta — the affected estimates and components. Don't redraw the world.

### Q78. How do you handle a prompt you've never seen?
The framework is the answer. I don't need to have seen the product; I need to elicit, quantify, specify, synthesise and defend. The primitives — load balancer, cache, queue, replica, shard — are reusable across every problem.

### Q79. ⚡ How do you show seniority in a design interview?
Three ways: derive components from numbers instead of declaring them, state the trade-off and its cost explicitly, and name the failure modes of your own design before being asked.

### Q80. How do you handle a design question you think is over-specified?
Ask whether the extra requirement matters. If a feature doesn't change the architecture, say so: "I'd note that X adds a field, not a component — I'll include it but it doesn't change the design."

### Q81. What's the most common way candidates fail?
Skipping requirement clarification and drawing immediately. Everything downstream — wrong scope, unjustified components, incoherent API — traces back to it.

---

## Cheat-sheet — the answers to memorise

| Topic | The sentence |
|---|---|
| Definition | Deciding how to build a large system that works for millions without breaking or slowing |
| Role | Architect, not builder — you're not writing code here |
| Four scale problems | Server crash, DB slow, data lost, concurrent write conflict |
| Six goals | Scalability, availability, reliability, performance, consistency, cost — and they conflict |
| Archetypes | WhatsApp → availability+reliability · Exchange → consistency · Instagram → performance |
| Five steps | Clarify → estimate → API → HLD → deep dive + trade-offs |
| Estimation | `DAU × req/user/day` → `÷86,400` → `×3 peak` → `×record size` for storage/bandwidth |
| Anchors | 10⁵ s/day · 1M req/day ≈ 12 QPS · 1 KB × 1M = 1 GB · DB ≈ 5,000 QPS · 3× peak |
| Nines | 99% = 3.65 d · 99.9% = 8.77 h · 99.99% = 52.6 min · 99.999% = 5.26 min |
| Availability math | `MTBF ÷ (MTBF+MTTR)` · series multiplies · parallel = `1−(1−A)ⁿ` |
| SLI/SLO/SLA | Measure / internal target / customer contract — SLO stricter than SLA |
| Trade-off formula | "I'll use [X] because [number]. It gives me [benefit], at the cost of [sacrifice]." |
| The close | "If the requirement changed to [Y], I'd change [Z] and accept [new cost]." |

---

## Cross-links

- Theory: [What is system design](../HLD/01-foundations/01-what-is-system-design-and-why.md) · [Six goals & trade-offs](../HLD/01-foundations/02-six-goals-and-tradeoffs.md) · [HLD vs LLD](../HLD/01-foundations/03-hld-vs-lld.md) · [Five-step framework](../HLD/01-foundations/04-five-step-interview-framework.md) · [Capacity estimation](../HLD/01-foundations/05-capacity-estimation-playbook.md) · [Availability & SLOs](../HLD/01-foundations/06-availability-numbers-and-slos.md)
- Worked answers: [URL shortener](../Case-Studies/HLD/01-url-shortener.md)
- Other question banks: [Networking & API](02-networking-and-api-questions.md) · [Scaling, caching & DB](03-scaling-caching-db-questions.md) · [LLM, RAG & GenAI](04-llm-rag-genai-questions.md) · [Agentic AI](05-agentic-ai-questions.md) · [LLMOps](06-llmops-and-operationalisation-questions.md) · [Behavioural](07-behavioral-and-project-questions.md)
- Cheat sheet: [Foundations & estimation](../Cheat-Sheets/01-foundations-and-estimation-cheatsheet.md)
