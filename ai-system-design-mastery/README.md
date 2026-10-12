# AI System Design Mastery

**A complete, transcript-grounded study system for AI system design interviews — organised by interview round: HLD, LLD, case studies, operations, question banks and cheat sheets.**

Built from 7 video transcripts (6,880 lines) on AI system design and interviews, plus **the agent-engineering video series** (~6,700 further lines — see [the source map](00-meta/02-source-topic-map.md)) and reference repositories on agentic AI, LLM engineering and production GenAI interview prep.

**100 files · ~92,700 lines · 1,759 interview questions and pushbacks with model answers.**

---

## The structure mirrors the interview

The source roadmap defines the course in five phases, and three of them are literally **High-Level Design**, **Low-Level Design** and a **capstone**. The top level of this repository is that structure, because that is how you actually revise for a loop — you do not revise "networking", you revise *the HLD round*.

| Round you're facing | Go to |
|---|---|
| **"Design <a product>"** — boxes, arrows, scale, trade-offs | **[`HLD/`](HLD/)** |
| **"Design a parking lot" / "implement an LRU cache"** — classes, methods, concurrency | **[`LLD/`](LLD/)** |
| **A full end-to-end design, worked** | **[`Case-Studies/`](Case-Studies/)** |
| **"How would you operate this?"** — SLOs, deploys, incidents, cost | **[`Operations/`](Operations/)** |
| **Rapid revision the night before** | **[`Cheat-Sheets/`](Cheat-Sheets/)** |
| **Practising answers out loud** | **[`Interview-Questions/`](Interview-Questions/)** |
| **Running a full mock and a capstone project** | **[`00-meta/03`](00-meta/03-capstone-project-and-mock-interview.md)** |

**The one rule that governs every file: never start drawing — clarify requirements first.**

---

## The 30-second summary of the whole field

System design is the discipline of deciding how to build a large system that works for millions of users without breaking or slowing down. You are the architect, not the builder. Four problems make it hard at scale: servers crash under load, databases slow under millions of queries, data is lost when a machine fails, and two users update the same record at once — plus a fifth that is physics rather than load, distance. Six goals pull against each other — scalability, availability, reliability, performance, consistency and cost — so every design is a *selection*, justified by numbers. The interview has five steps: **clarify requirements → estimate capacity → define the API → draw the high-level design → deep-dive and state trade-offs.** Then there is a second, lower-level round: **classes, interfaces, concurrency and extensibility** — the HLD answer and the LLD answer to the same product are different artefacts and are examined separately. In 2026 the same discipline extends to LLM systems, where retrieval, evaluation, guardrails, inference economics and agent orchestration are the new primitives — and where **cost per query** and **output quality** are first-class constraints alongside latency and availability.

---

## Repository map

**New here?** Start with [`00-meta/01-how-to-use-and-template.md`](00-meta/01-how-to-use-and-template.md) — how the repository is organised, the section template every file follows, and the reading paths by time available.

```
00-meta/                     How to use it · source & topic map · capstone + mock interview
│
HLD/                         ── the high-level design round ──
├── 01-foundations/          The framework and the arithmetic      ← START HERE
├── 02-networking/           DNS · CDN · proxy · API gateway
├── 03-scaling/              Load balancing · caching · Redis · sharding
├── 04-data/                 SQL · NoSQL · vector DB · blob · selection
└── 05-ai-systems/           RAG · agents · LLM serving · guardrails · evals
│
LLD/                         ── the low-level design round ──
├── 01–06                    OOP/SOLID · patterns · concurrency · UML · framework
└── problems/                16 class-level design problems
│
Case-Studies/                ── worked designs, both levels ──
├── HLD/                     14 end-to-end system designs
└── LLD/                     5 systems taken down to classes
│
Operations/                  Running it in production (10 playbooks)
Interview-Questions/         8 banks of model answers
Cheat-Sheets/                7 dense revision sheets
```

---

## HLD — the high-level design round

### HLD/01-foundations

*The framework and the arithmetic. Read these first — everything else assumes them.*

| File | What it covers |
|---|---|
| [01. What is system design & why it matters](HLD/01-foundations/01-what-is-system-design-and-why.md) | The definition · architect vs builder · the four scale problems **plus distance** · the McDonald's analogy · who asks · the numbers anchor table |
| [02. The six goals & their trade-offs](HLD/01-foundations/02-six-goals-and-tradeoffs.md) | Scalability · availability · reliability · performance · consistency · cost · product archetypes · CAP & PACELC · the conflict matrix |
| [03. HLD vs LLD](HLD/01-foundations/03-hld-vs-lld.md) | Components and connections vs classes and complexity · the parking-lot LLD · how to tell which round you're in · **the bridge to the [`LLD/`](LLD/) pillar** |
| [04. The five-step interview framework](HLD/01-foundations/04-five-step-interview-framework.md) | Clarify → estimate → API → HLD → deep dive · functional vs non-functional · the "right way vs wrong way" answer pattern |
| [05. Capacity estimation playbook](HLD/01-foundations/05-capacity-estimation-playbook.md) | The five formulas · the anchors · three worked examples · the estimate→architecture mapping · the professional answer pattern |
| [06. Availability numbers & SLOs](HLD/01-foundations/06-availability-numbers-and-slos.md) | MTBF/MTTR · the nines · series and parallel composition · fan-out reality check · SLI/SLO/SLA · the cost curve of nines |

### HLD/02-networking

| File | What it covers |
|---|---|
| [01. DNS deep dive](HLD/02-networking/01-dns-deep-dive.md) | Resolution path · record types · TTL and propagation · failure modes · DNS in failover |
| [02. CDN deep dive](HLD/02-networking/02-cdn-deep-dive.md) | Edge caching · push vs pull · origin shield · invalidation · bandwidth play vs latency play |
| [03. Forward & reverse proxy](HLD/02-networking/03-forward-and-reverse-proxy.md) | The distinction · TLS termination · caching proxies · load-balancing proxies · security roles |
| [04. API gateway & rate limiting](HLD/02-networking/04-api-gateway-and-rate-limiting.md) | Gateway responsibilities · token bucket, leaky bucket, sliding window · distributed rate limiting · 429 semantics |

### HLD/03-scaling

| File | What it covers |
|---|---|
| [01. Load balancers](HLD/03-scaling/01-load-balancers.md) | L4 vs L7 · algorithms · health checks · session affinity · global load balancing · failure modes |
| [02. Caching strategies & layers](HLD/03-scaling/02-caching-strategies-and-layers.md) | Client/edge/application/DB caching · cache-aside, write-through, write-back · eviction · hit-rate economics |
| [03. Redis deep dive](HLD/03-scaling/03-redis-deep-dive.md) | Data structures · persistence (RDB/AOF) · clustering · replication · use cases and anti-patterns |
| [04. Cache invalidation & consistency](HLD/03-scaling/04-cache-invalidation-and-consistency.md) | TTLs · write-through vs invalidate · staleness budgets · stampede and thundering herd |
| [05. Sharding, replication & consistency](HLD/03-scaling/05-sharding-replication-and-consistency.md) | Shard keys · rebalancing · replication topologies · sync vs async · consistency models · quorums |

### HLD/04-data

| File | What it covers |
|---|---|
| [01. SQL deep dive](HLD/04-data/01-sql-deep-dive.md) | Indexes · transactions and isolation · query planning · replication · sharding · when SQL hits its ceiling |
| [02. NoSQL deep dive](HLD/04-data/02-nosql-deep-dive.md) | Key-value, document, wide-column, graph · CAP positioning · modelling for access patterns |
| [03. Vector databases](HLD/04-data/03-vector-databases.md) | Embeddings · HNSW and IVF · hybrid search and RRF · filtering · recall/latency/memory trade-offs |
| [04. Blob & object storage](HLD/04-data/04-blob-and-object-storage.md) | Object vs block vs file · tiering · lifecycle policies · CDN integration · cost model |
| [05. Choosing a database — decision guide](HLD/04-data/05-choosing-a-database-decision-guide.md) | The decision tree · access pattern first · the questions to ask before choosing |

### HLD/05-ai-systems

*The 2026 core. These files extend every classical primitive with three new variables: generated output, per-request cost, and probabilistic correctness.*

| File | What it covers |
|---|---|
| [01. Production RAG system](HLD/05-ai-systems/01-production-rag-system.md) | Ingestion and query pipelines · chunking · embeddings · hybrid retrieval · reranking · grounded generation |
| [02. Advanced RAG patterns](HLD/05-ai-systems/02-advanced-rag-patterns.md) | HyDE · contextual retrieval · GraphRAG · RAPTOR · self-RAG · CRAG · when each earns its complexity |
| [03. RAG evaluation & observability](HLD/05-ai-systems/03-rag-evaluation-and-observability.md) | recall@k · faithfulness · RAGAS · LLM-as-judge · production sampling · the eval pipeline |
| [04. Guardrails & safety for AI systems](HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md) | Input and output guardrails · prompt injection · OWASP LLM Top 10 · defence in depth |
| [05. LLM serving, inference & gateway](HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md) | Prefill/decode · KV cache · continuous batching · paged attention · speculative decoding · TTFT/TPOT · gateways |
| [06. Agentic system design](HLD/05-ai-systems/06-agentic-system-design.md) | ReAct · planning · durable execution · step/token/time budgets · tool design · failure modes |
| [07. MCP, tool use & multi-agent](HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md) | MCP · A2A · tool schemas · topologies (supervisor, swarm, hierarchical) · coordination costs |
| [08. Memory & context engineering](HLD/05-ai-systems/08-memory-and-context-engineering.md) | Short vs long-term memory · summarisation · retrieval-augmented context · compaction · context budgets |
| [09. Fine-tuning vs RAG vs prompting](HLD/05-ai-systems/09-fine-tuning-vs-rag-vs-prompting.md) | The decision framework · LoRA/QLoRA · data requirements · cost comparison · combining approaches |
| [10. Distributing a huge model weight file](HLD/05-ai-systems/10-distributed-model-weight-serving.md) | Sharded checkpoints · cache hierarchy · P2P distribution · manifest + barrier · version skew · TTFS · the egress arithmetic |
| [11. The agent engineering hands-on stack](HLD/05-ai-systems/11-agent-engineering-hands-on-stack.md) | `uv` · LangChain `create_agent` · tools · LangGraph checkpointers & `thread_id` · DynamoDB saver · Pydantic structured output · LangFuse tracing · middleware · supervisor multi-agent |

---

## LLD — the low-level design round

> **The source calls this "ये सबसे ज़्यादा इंपॉर्टेंट है आपका" — the most important part for you.** It is also the pillar this repository previously lacked almost entirely. The source names its syllabus explicitly: OOPS concepts, coding problems, class diagrams, **parking lot**, **LRU/LFU**, **hotel management system**, and **BookMyShow seat locking with concurrent booking**.
>
> **Attribution note:** the reference repositories contain no OOP/LLD material, and the transcripts *name* these topics without teaching them in depth — with the exception of the parking-lot walkthrough, which is genuinely in the Day 1 transcript. So the LLD files here are **supplementary**: standard interview canon, clearly marked as such, with the source-named topics flagged.

### LLD/01–06 — the concept files

| File | What it covers |
|---|---|
| [01. What is LLD & how to answer it](LLD/01-what-is-lld-and-how-to-answer.md) | The source's own definition · the parking-lot walkthrough from the transcript · how LLD differs from HLD and from machine coding · what interviewers score |
| [02. OOP principles & SOLID](LLD/02-oop-principles-and-solid.md) | Encapsulation, inheritance, polymorphism, abstraction · the five SOLID principles with a violation→fix example each · composition over inheritance · interface vs abstract class |
| [03. Design patterns](LLD/03-design-patterns.md) | Creational, structural, behavioural — each with intent, a use case from *this repo's own systems*, and when **not** to use it |
| [04. Concurrency & threading in LLD](LLD/04-concurrency-and-threading-in-lld.md) | Race conditions, locks, atomicity, deadlock/livelock · thread-safety strategies · optimistic vs pessimistic locking — the bridge to seat locking |
| [05. UML & class diagrams](LLD/05-uml-and-class-diagrams.md) | Class/sequence/state diagrams · association vs aggregation vs composition · cardinality · drawing legibly under time pressure |
| [06. The machine-coding interview framework](LLD/06-machine-coding-interview-framework.md) | The LLD equivalent of the five steps: clarify → entities → diagram → interfaces → code → extend → test |

### LLD/problems — 16 class-level problems

*Each follows a fixed shape: requirement clarification → core entities → class diagram → interfaces → code → concurrency & thread-safety → extensibility ("add feature X") → failure modes → trade-offs → Q&A.*

**The four the source names** — do these first:

| File | What it covers |
|---|---|
| [01. Parking lot](LLD/problems/01-parking-lot.md) | **Source-named, with the transcript's own class structure.** Floors, spots, vehicle types, tickets, pricing |
| [02. LRU / LFU cache](LLD/problems/02-lru-lfu-cache.md) | **Source-named.** Hash map + doubly-linked list, O(1); LFU frequency buckets; thread-safety |
| [03. Hotel management system](LLD/problems/03-hotel-management.md) | **Source-named.** Room inventory, reservations, check-in/out, pricing, availability search |
| [04. BookMyShow seat locking](LLD/problems/04-bookmyshow-seat-locking.md) | **Source-named.** Seat locking, concurrent booking, the double-booking problem, lock TTL and expiry |

**Then the standard canon:**

| File | | File | |
|---|---|---|---|
| [05. Elevator system](LLD/problems/05-elevator-system.md) | Scheduling, state machine | [11. Rate limiter](LLD/problems/11-rate-limiter.md) | Token bucket, sliding window |
| [06. Splitwise](LLD/problems/06-splitwise.md) | Graph simplification, settle-up | [12. Logger framework](LLD/problems/12-logger-framework.md) | Levels, appenders, async |
| [07. ATM machine](LLD/problems/07-atm-machine.md) | State machine, cash dispensing | [13. Pub-sub system](LLD/problems/13-pub-sub-system.md) | Topics, subscribers, delivery |
| [08. Library management](LLD/problems/08-library-management.md) | Lending, fines, reservations | [14. Tic-tac-toe](LLD/problems/14-tic-tac-toe.md) | Board, win detection, extensibility to N×N |
| [09. Chess game](LLD/problems/09-chess-game.md) | Pieces, moves, legality, check | [15. Snake and ladder](LLD/problems/15-snake-and-ladder.md) | Board, dice, players, turn order |
| [10. Vending machine](LLD/problems/10-vending-machine.md) | State machine, inventory, change | [16. Food delivery](LLD/problems/16-food-delivery.md) | Orders, assignment, tracking |

**Where a problem overlaps HLD**, the LLD file owns the *class and code* treatment and cross-links to the HLD file for the distributed view — no duplication. Rate limiter ↔ [HLD/02-networking/04](HLD/02-networking/04-api-gateway-and-rate-limiting.md); LRU ↔ [HLD/01-foundations/03](HLD/01-foundations/03-hld-vs-lld.md); pub-sub ↔ the Kafka material in the [URL shortener](Case-Studies/HLD/01-url-shortener.md).

---

## Case-Studies — worked designs at both levels

### Case-Studies/HLD — 14 end-to-end designs

*Each runs the five-step framework and ends with defended trade-offs.*

| File | What it covers |
|---|---|
| [01. URL shortener](Case-Studies/HLD/01-url-shortener.md) | **The flagship.** 100:1 read-heavy · base62 vs hashing vs distributed IDs · 301 vs 302 · cache-first architecture · async click analytics |
| [02. WhatsApp chat system](Case-Studies/HLD/02-whatsapp-chat-system.md) | Real-time delivery · WebSockets · message ordering · delivery guarantees · offline handling · group fan-out |
| [03. Notification system](Case-Studies/HLD/03-notification-system.md) | Multi-channel (push/email/SMS) · fan-out · deduplication · preferences · at-least-once delivery |
| [04. Instagram news feed](Case-Studies/HLD/04-instagram-news-feed.md) | Push vs pull fan-out · hybrid model for celebrities · timeline storage · ranking |
| [05. YouTube upload & recommendation](Case-Studies/HLD/05-youtube-upload-and-recommendation.md) | Chunked upload · transcoding pipeline · CDN delivery · recommendation architecture |
| [06. Uber ride-hailing](Case-Studies/HLD/06-uber-ride-hailing.md) | Geospatial indexing · matching · real-time location updates · surge pricing · consistency needs |
| [07. BookMyShow seat booking](Case-Studies/HLD/07-bookmyshow-seat-booking.md) | Concurrency control · seat locking · inventory consistency · the double-booking problem |
| [08. Twitter/X — timeline, trends & search](Case-Studies/HLD/08-twitter-x-feed-and-search.md) | Fan-out at celebrity scale · the hybrid timeline model · trending topics · tweet search (inverted index) · the write:read asymmetry |
| [09. Production RAG case study](Case-Studies/HLD/09-production-rag-case-study.md) | The full RAG system end-to-end: ingestion, retrieval, generation, evaluation, operations, cost |
| [10. AI customer support agent](Case-Studies/HLD/10-ai-customer-support-agent.md) | Retrieval + tools + escalation · human handoff · guardrails · quality monitoring |
| [11. Agentic coding assistant](Case-Studies/HLD/11-agentic-coding-assistant.md) | Repo context · tool loops · sandboxing · verification · budget control |
| [12. Netflix — streaming & recommendation](Case-Studies/HLD/12-netflix-recommendation-and-streaming.md) | **Source-named.** Open Connect CDN · per-title encoding · the recommendation pipeline · the two-row problem · peak egress arithmetic |
| [13. Fraud detection system](Case-Studies/HLD/13-fraud-detection-system.md) | **Source-named.** The 0.1% base rate · the specificity arithmetic · label delay · selective-label bias · three-way decisions |
| [14. Autocomplete & typeahead](Case-Studies/HLD/14-autocomplete-and-typeahead.md) | **Source-named.** Precomputed top-k over a trie · prefix-range sharding · Zipf and caching · the layered freshness index |

### Case-Studies/LLD — 5 systems taken down to classes

*The bridge between the two rounds: the same systems as the HLD studies, one level down. Each shows how an HLD box becomes classes, and references [`LLD/problems/`](LLD/problems/) for technique rather than repeating it.*

| File | The HLD box it opens up |
|---|---|
| [01. URL shortener LLD](Case-Studies/LLD/01-url-shortener-lld.md) | The ID-generation and redirect classes behind the HLD design |
| [02. WhatsApp LLD](Case-Studies/LLD/02-whatsapp-lld.md) | Message, conversation and delivery-status models; ordering |
| [03. Uber matching LLD](Case-Studies/LLD/03-uber-matching-lld.md) | Driver, rider, trip, and the matching/assignment classes |
| [04. Instagram feed LLD](Case-Studies/LLD/04-instagram-feed-lld.md) | Post, follow-graph and feed-assembly classes |
| [05. Notification system LLD](Case-Studies/LLD/05-notification-system-lld.md) | Channel abstraction, template and preference classes |

BookMyShow is deliberately **not** duplicated here: its HLD view is [Case-Studies/HLD/07](Case-Studies/HLD/07-bookmyshow-seat-booking.md) and its LLD view is [LLD/problems/04](LLD/problems/04-bookmyshow-seat-locking.md), cross-linked.

---

## Operations — running it in production

*Playbook-shaped: purpose, scope, SLIs, SOPs, failure modes, capacity, alerts, change management, compliance.*

| File | What it covers |
|---|---|
| [01. Operational readiness review](Operations/01-operational-readiness-checklist.md) | The go-live gate · 9 checklist groups · critical vs standard items · the failure rehearsal · AI/LLM gates |
| [02. Observability & monitoring](Operations/02-observability-and-monitoring.md) | Metrics/logs/traces/profiles/evals · RED & USE · cardinality · sampling · alerting philosophy · agent observability |
| [03. SLOs, error budgets & incident response](Operations/03-slos-error-budgets-and-incident-response.md) | SLIs · budget policy · burn-rate alerting · incident roles · mitigation ladder · blameless PIR |
| [04. Deployment, release & CI/CD](Operations/04-deployment-release-and-cicd.md) | Pipelines · canary · blue/green · feature flags · eval-gated deployment for AI · rollback |
| [05. Capacity planning & autoscaling](Operations/05-capacity-planning-and-autoscaling.md) | Load forecasting · autoscaling policies · tokens/sec and queue-depth scaling · Little's law · time-to-scale |
| [06. Cost & FinOps](Operations/06-cost-and-finops.md) | Unit economics · prompt caching · batch APIs · model routing · GPU break-even · cost anomaly detection |
| [07. Reliability patterns](Operations/07-reliability-patterns.md) | Circuit breakers · bulkheads · timeouts · retries with jitter · load shedding · degradation ladders |
| [08. Security, guardrails & compliance](Operations/08-security-guardrails-and-compliance.md) | AuthN/AuthZ · secrets · data classification · SOC2/HIPAA/GDPR/EU AI Act · audit · AI-specific security |
| [09. DR, failover & backup](Operations/09-dr-failover-and-backup.md) | RTO/RPO · backup strategies · restore drills · multi-region · index rebuild vs snapshot |
| [10. LLMOps for AI systems](Operations/10-llmops-for-ai-systems.md) | The LLMOps lifecycle · prompt/model/index versioning · eval-gated release · provider management · quality SLOs |

---

## Interview-Questions — 8 banks of model answers

*Written the way you'd actually speak them. ⚡ marks the senior-separating questions.*

| File | Contents |
|---|---|
| [01. Foundations, framework, capacity & availability](Interview-Questions/01-foundations-and-capacity-questions.md) | Definitions · the six goals · HLD/LLD · the framework · estimation · availability · estimation drills · communication |
| [02. Networking & API](Interview-Questions/02-networking-and-api-questions.md) | DNS · CDN · proxies · gateways · rate limiting · REST vs GraphQL · idempotency · versioning |
| [03. Scaling, caching & databases](Interview-Questions/03-scaling-caching-db-questions.md) | Load balancing · caching · Redis · invalidation · sharding · replication · consistency · SQL vs NoSQL · vector DBs |
| [04. LLM, RAG & GenAI](Interview-Questions/04-llm-rag-genai-questions.md) | RAG pipelines · chunking · embeddings · hybrid search · reranking · advanced patterns · evaluation · serving · guardrails |
| [05. Agentic AI](Interview-Questions/05-agentic-ai-questions.md) | ReAct · planning · memory · tool design · MCP · A2A · topologies · budgets · failure modes · evaluation |
| [06. LLMOps & operationalisation](Interview-Questions/06-llmops-and-operationalisation-questions.md) | SLOs · observability · release engineering · capacity · cost · reliability · security · DR · the LLMOps lifecycle |
| [07. Behavioural & project questions](Interview-Questions/07-behavioral-and-project-questions.md) | STAR stories · incident narratives · trade-off justifications · project deep-dives · leadership signals |
| [08. LLD questions](Interview-Questions/08-lld-questions.md) | OOP & SOLID · design patterns · concurrency · UML · the class-level problems · the "add feature X" follow-ups |

---

## Cheat-Sheets — 7 dense revision sheets

*Use them to revise, not to learn.*

| File | Contents |
|---|---|
| [01. Foundations & estimation](Cheat-Sheets/01-foundations-and-estimation-cheatsheet.md) | The five steps · definitions · the six goals · estimation anchors · the nines · trade-off phrases · 15 mistakes |
| [02. Networking](Cheat-Sheets/02-networking-cheatsheet.md) | DNS · CDN · proxies · API gateway · rate limiting algorithms · latency budgets |
| [03. Scaling, caching & databases](Cheat-Sheets/03-scaling-caching-db-cheatsheet.md) | LB algorithms · caching strategies · Redis · invalidation · sharding · replication · DB selection |
| [04. RAG & LLM](Cheat-Sheets/04-rag-and-llm-cheatsheet.md) | The RAG pipeline · chunking · hybrid search · reranking · advanced patterns · evaluation · serving · guardrails |
| [05. Agentic AI](Cheat-Sheets/05-agentic-ai-cheatsheet.md) | Agent loop · tool schema · memory types · MCP · topologies · budgets · failure modes |
| [06. Operationalisation](Cheat-Sheets/06-operationalisation-cheatsheet.md) | SLOs and burn rates · alert thresholds · incident roles and ladder · deployment · capacity · cost levers · security · DR |
| [07. LLD](Cheat-Sheets/07-lld-cheatsheet.md) | SOLID · the pattern catalogue · concurrency primitives · UML notation · the machine-coding framework · the 16 problems compressed |

---

## The 10 rules that apply to every question

1. **Never start drawing.** Clarify requirements first — functional *and* non-functional.
2. **Estimate before you design.** Numbers justify architecture.
3. **Derive components; don't declare them.** Every box traces to a requirement or a number.
4. **Label every arrow** with its protocol or purpose.
5. **State trade-offs in four parts:** the choice, the numeric reason, the benefit, the cost.
6. **Name your own failure modes** before you're asked.
7. **Distinguish availability from reliability** — "is it up?" vs "is it working?"
8. **Remember the multipliers:** replication ×3, index overhead ×1.5, fan-out, peak ×3.
9. **Protect the last five minutes** for trade-offs.
10. **Close with what you'd change if the requirement changed.**

---

## The 60-second answer (memorise this before any interview)

> "I follow five steps. First I clarify requirements — functional and non-functional — because 'design Twitter' could mean the feed, the tweet path, the follow graph or the notifications, and guessing wrong wastes the interview. Second, capacity estimation, because numbers justify architecture: I can't say 'I'll add a cache' without knowing read QPS versus what a database can absorb. Third, I define the API endpoints and their payloads, because that forces precision about what the system actually does. Fourth, I draw the high-level design with labelled arrows. Fifth, I go deep on one or two areas and state the trade-offs explicitly — including what I gave up and what I'd change if the requirement changed."

---

## Key numbers

| | |
|---|---|
| Seconds/day | 86,400 ≈ 10⁵ |
| 1M requests/day | ≈ 12 QPS |
| 1B requests/day | ≈ 12,000 QPS |
| 1 KB × 1M records | 1 GB |
| Peak multiplier | 3× average |
| Single DB reads | ≈ 5,000 QPS |
| Replication factor | 3× |
| Index overhead | 1.5× |
| Cross-region round trip | 60–70 ms of pure physics |
| 99% availability | 3.65 days downtime/year |
| 99.9% | 8.77 hours |
| 99.99% | 52.6 minutes |
| 99.999% | 5.26 minutes |

---

## Study paths

| Time available | Path |
|---|---|
| **20 minutes** | [Cheat sheet 01](Cheat-Sheets/01-foundations-and-estimation-cheatsheet.md) |
| **2 days** | Foundations (all) → [URL shortener](Case-Studies/HLD/01-url-shortener.md) → all cheat sheets |
| **1 week** | Foundations → Networking → Scaling → Data → cheat sheets → question banks |
| **2 weeks** | Everything in order, case studies last each day, with the whiteboard drill |
| **AI-focused** | [Foundations 04](HLD/01-foundations/04-five-step-interview-framework.md) + [05](HLD/01-foundations/05-capacity-estimation-playbook.md) → all of [HLD/05-ai-systems](HLD/05-ai-systems/) → [Cheat sheets 04 & 05](Cheat-Sheets/) |
| **Ops-focused** | All of [Operations](Operations/) → [Cheat sheet 06](Cheat-Sheets/06-operationalisation-cheatsheet.md) → [Questions 06](Interview-Questions/06-llmops-and-operationalisation-questions.md) |
| **An LLD/machine-coding round** | [LLD/01](LLD/01-what-is-lld-and-how-to-answer.md) → [LLD/06](LLD/06-machine-coding-interview-framework.md) → the four source-named problems ([01](LLD/problems/01-parking-lot.md), [02](LLD/problems/02-lru-lfu-cache.md), [03](LLD/problems/03-hotel-management.md), [04](LLD/problems/04-bookmyshow-seat-locking.md)) → [Cheat sheet 07](Cheat-Sheets/07-lld-cheatsheet.md) |
| **Not ready yet** | [The capstone project and the mock interview](00-meta/03-capstone-project-and-mock-interview.md) — invent a system, design it at both levels, then mock it |

---

## Sources, coverage and honesty notes

Seven roadmap video transcripts (6,880 lines), plus **the agent-engineering video series** (~6,700 further lines) and reference repositories on agentic AI, LLM engineering and production GenAI interview prep. Full traceability — every topic mapped to its source — is in [`00-meta/02-source-topic-map.md`](00-meta/02-source-topic-map.md).

Five things worth stating plainly:

1. **Two of the seven transcripts are Hindi in Devanagari script** (the roadmap video and Day 1 Foundations). English keyword search over them silently returns nothing, so they were mined by reading. Where they are quoted in this repository, the original is given alongside a translation.
2. **The LLD pillar is largely supplementary.** The transcripts *name* the LLD syllabus — OOPS concepts, parking lot, LRU/LFU, hotel management, BookMyShow seat locking — but teach only the parking-lot example in depth. Every LLD file says so at the top rather than implying transcript grounding.
3. **Figures are labelled.** Assumptions are marked `[A]` and derived arithmetic `[D]` in the newer case studies, because a capacity number presented without its assumption is indistinguishable from a measurement.
4. **Where the transcripts' numbers are garbled, this repository says so and uses the correct ones.** The clearest case is the Day 1 messaging-system estimation, whose extracted transcript renders the peak as *"500 × 3 = 50,000"* and the bandwidth as *"1 MB/s"* — both wrong, and both corrected (15,000/s and 5 MB/s) with a note explaining why the *method* is still sound. The full note is in [`HLD/01-foundations/05`](HLD/01-foundations/05-capacity-estimation-playbook.md) §3.3. The same care applies in the other direction: where a diagram goes beyond the source, it says which parts are the source's and which are this repository's (`LLD/01` §1.2 is the example).

5. **The agent-engineering series is admitted as a source, and its `archive/` sub-folder is not.** The `nitmonk` folder was originally excluded by instruction; that was lifted on 2026-10-03, and the four English transcripts outside `nitmonk/archive/` now supply the whole build layer beneath the AI-systems pillar — context/state/memory, the six verbs of context engineering, compaction vs compression, the memory taxonomy, MCP progressive discovery, and the LangChain/LangGraph/LangFuse/DynamoDB stack ([`HLD/05-ai-systems/11`](HLD/05-ai-systems/11-agent-engineering-hands-on-stack.md)). **`nitmonk/archive/` remains excluded** — it holds superseded duplicates and the Hindi-origin recordings of material available in English outside it. They are catalogued as **T8–T11** in the source map — deliberately numbered after the seven roadmap transcripts (T1–T7) so that the two sets are never confused for one another.
