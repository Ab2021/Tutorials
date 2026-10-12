# Source & Topic Map — Transcript → File Traceability

> **Purpose:** every file in this knowledge base traces back to a source. This is the mapping, so you can verify a claim, find the original context, or extend the material without duplicating it.

---

## 1. The source inventory

### Primary sources — video transcripts (7 files, 6,880 lines)

> **Language disclosure — important for anyone extending this repo.** **Two of the seven transcripts are Hindi written in Devanagari script**, not English: **T1** and **T2**. Roughly 100% of T2's 1,128 lines are Devanagari. This has a concrete consequence: **an English keyword search over these two files silently returns nothing** — `grep -i "scalability"` finds zero matches in a file that discusses scalability extensively. They were therefore mined by *reading*, not searching. Any future extension of this repository must do the same, or it will appear that the source never covered a topic that it in fact covers in detail.

| # | Transcript | Lines | Language | Covers |
|---|---|---|---|---|
| **T1** | `Complete_AI_System_Design_Roadmap_2026_Crack_Any_AI_Interview` | 344 | **Hindi (Devanagari)** | The five-phase roadmap, the interview landscape, the company taxonomy, what to study in what order |
| **T2** | `System_Design_Interview_Prep_Day_1_Foundations_Framework_Capacity_Estimation` | 1,128 | **Hindi (Devanagari)** | Definitions, the scale problems, the McDonald's analogy, the six goals, HLD vs LLD, the five-step framework, capacity estimation, availability arithmetic |
| **T3** | `AI_System_Design_Interview_Most_Asked_FAANG_System_Design_Question` | 771 | English | The full URL shortener walkthrough — requirements, estimation, architecture, tool diagram, trade-offs |
| **T4** | `AI_System_Design_Networking_DNS_CDN_Forward_Reverse_Proxy_API_Gateway` | 671 | English | DNS resolution, CDN mechanics, forward vs reverse proxy, API gateway, rate limiting |
| **T5** | `The_Ultimate_Guide_to_AI_System_Design_Load_Balancers_Redis_and_Cache_Invalidation` | 1,418 | English | Load balancing algorithms and layers, health checks, Redis data structures and persistence, caching strategies, invalidation |
| **T6** | `AI_System_Design_SQL_NoSQL_Vector_DB_Blob_Storage_Complete_Guide` | 1,007 | English | SQL internals and scaling, NoSQL families, vector databases and ANN indexes, blob storage, selection criteria |
| **T7** | `How_to_Design_a_Production_RAG_System` | 1,541 | English | The full RAG pipeline — ingestion, chunking, embeddings, retrieval, reranking, generation, evaluation, operations |

---

## 1b. The source's five-phase curriculum — and how this repo mirrors it

T1 defines the course in five phases, and gives them names. The repository's top level is that structure, so a reader revising for a specific round can go straight to it.

| Phase | The source's name for it | What the source says it contains | Where it lives here |
|---|---|---|---|
| **1** | **System Design Foundation** | "What is system design? What is networking? DNS? CDN? Forward proxy? Load balancer? Caching? How do we store data and retrieve it fast? How does cache invalidation work? How does vector DB store data? SQL, NoSQL, effective DB choice?" | [`HLD/01-foundations`](../HLD/01-foundations/) · [`HLD/02-networking`](../HLD/02-networking/) · [`HLD/03-scaling`](../HLD/03-scaling/) · [`HLD/04-data`](../HLD/04-data/) |
| **2** | **High-Level Design** | YouTube (upload, pipeline, CI/CD, git, recommendation), Uber, Twitter, **autocomplete**, data storage | [`Case-Studies/HLD`](../Case-Studies/HLD/) · [`HLD/03-scaling`](../HLD/03-scaling/) · [`HLD/04-data`](../HLD/04-data/) |
| **3** | **Low-Level Design** | "OOPS patterns, coding problems. How is BookMyShow built? How do you do seat locking? How does concurrent booking work? What is LRU? What is LFU? How is a hotel management system built? Booking?" — *"ये सबसे ज़्यादा इंपॉर्टेंट है आपका"* (the most important part for you) | **[`LLD/`](../LLD/)** — the whole pillar, plus [`Case-Studies/HLD/07`](../Case-Studies/HLD/07-bookmyshow-seat-booking.md) for the HLD view |
| **4** | **ML, Generative AI & Agentic AI Systems** | Guardrails, MCP, LLM interfaces, ML practice design, **the Netflix system**, **the fraud detection system**, RAG, sharding, vector DB, retrieval, evaluation, indexing, fine-tuning vs RAG vs prompt engineering, multi-agent systems, AI guardrails | [`HLD/05-ai-systems`](../HLD/05-ai-systems/) · [`Case-Studies/HLD/12`](../Case-Studies/HLD/12-netflix-recommendation-and-streaming.md) · [`Case-Studies/HLD/13`](../Case-Studies/HLD/13-fraud-detection-system.md) |
| **5** | **Capstone + mock interview** | "We'll create a complete design — a project that doesn't exist yet, applying LLD and HLD, everything we've learned." Plus a one-to-one mock interview | [`00-meta/03-capstone-project-and-mock-interview.md`](03-capstone-project-and-mock-interview.md) |

**Two structural facts worth noting about the source's own sequencing.** Phase 3 (LLD) sits *between* HLD and the AI material, and the source explicitly says why: *"I've put module four a little later so that your aptitude mind is completely ready."* The AI material is deliberately last because it assumes everything before it. And the source says it carries **42 videos**, with **a practice session after every four episodes** — because the stated goal is not theory but *"एपटीट्यूड माइंड"*, an aptitude mind.

### Secondary sources — reference repositories

| Repository | Used for |
|---|---|
| `ai-system-design-guide-main/` (~150 md files, 19 numbered sections) | Structure validation, topic coverage, glossary/patterns/radar material, case-study breadth |
| `AI-System-Design-main/` | Roadmap notes, URL shortener design code |
| `AmanAI-Lab-Agentic-AI-Interview-Series-main/` (Day 1–5 PDFs) | Agentic-AI interview framing |
| `LLM-Coding-Series-main/` | Runnable reference implementations: `rag.py`, `Agentic_RAG.py`, `react_agent.py`, `multi_tool_agent.py`, `langgraph_agent.py`, `langgraph_memory.py`, `multi_agent.py`, `guardrail_agent.py`, `human_approval.py` |
| `Production-level-Generative-AI-Interview-Questions-main/` (18 PDFs) | Question banks: advanced RAG (30), agentic AI (30), advanced AI agents (30), LLMOps (21), LLM (30), agentic RAG (10) |

### Tertiary source — the agent-engineering video series

| Source | Used for |
|---|---|
| `zip_reference_file/nitmonk/` — *24 Hours of AI Agent Engineering* Parts 1–2 (T8, T9 below), *23+ Context Engineering & AI Agent Memory Techniques* (T10), *LangChain + LangFuse — Build an AI Agent From Scratch* Parts 1–3, *AI Agents LIVE* Parts 1–2 (T11) | The **build layer** beneath the AI-systems pillar: the context/state/memory framing, the six verbs of context engineering, compaction vs compression, the full memory taxonomy (including *environment* and *project* memory), MCP progressive discovery and meta-tools, the agent harness, AWS AgentCore, middleware, and the hands-on LangChain / LangGraph / LangFuse / DynamoDB stack |

**Status: admitted 2026-10-03.** This folder was previously excluded by explicit instruction; that instruction has been lifted and it is now a source. Two boundaries remain:

- **`nitmonk/archive/` is excluded.** It holds a much larger duplicate of Part 2 (1.49 MB against 633 KB) and the Hindi-origin recordings. The English transcripts *outside* `archive/` are the ones drawn upon.
- **Two files are byte-identical duplicates** — `AI Agents live/…Part 2 - AI Agents LIVE…` and `AI AGENT FROM SCRATCH/…Part 2 - AI Agents LIVE…` have the same MD5, so that session is counted once, not twice.

### Explicitly excluded

| Source | Reason |
|---|---|
| `zip_reference_file/nitmonk/archive/` | Excluded by explicit instruction (2026-10-03). Superseded duplicates and Hindi-origin recordings of material already available in English outside `archive/`. |

---

## 2. Transcript → file mapping

### T1 — AI System Design Roadmap 2026

| Topic in the transcript | Where it lives |
|---|---|
| What system design is, who asks it, why it determines the salary band | [HLD/01-foundations/01](../HLD/01-foundations/01-what-is-system-design-and-why.md) |
| The classical topic roadmap | [00-meta/01](01-how-to-use-and-template.md) §3, reading paths |
| The AI/GenAI topic roadmap | [HLD/05-ai-systems](../HLD/05-ai-systems/) — all files |
| Frontier labs vs FAANG vs AI startups | [HLD/01-foundations/01](../HLD/01-foundations/01-what-is-system-design-and-why.md) |
| 60–70% of AI interviews include GenAI design | [Interview-Questions/01](../Interview-Questions/01-foundations-and-capacity-questions.md) · [00-meta/01](01-how-to-use-and-template.md) §7 |

### T2 — Day 1 Foundations

| Topic in the transcript | Where it lives |
|---|---|
| Definition of system design; "architect, not builder" | [HLD/01-foundations/01](../HLD/01-foundations/01-what-is-system-design-and-why.md) |
| The four scale problems (server crash, slow DB, data loss, concurrent writes) | [HLD/01-foundations/01](../HLD/01-foundations/01-what-is-system-design-and-why.md) |
| The McDonald's analogy (counters, patties, supplier, manager) | [HLD/01-foundations/01](../HLD/01-foundations/01-what-is-system-design-and-why.md) |
| HLD vs LLD; the parking-lot LLD example | [HLD/01-foundations/03](../HLD/01-foundations/03-hld-vs-lld.md) |
| The six goals and their conflicts | [HLD/01-foundations/02](../HLD/01-foundations/02-six-goals-and-tradeoffs.md) |
| Product archetypes (WhatsApp, exchange, Instagram) | [HLD/01-foundations/02](../HLD/01-foundations/02-six-goals-and-tradeoffs.md) |
| CAP, PACELC, consistency spectrum | [HLD/01-foundations/02](../HLD/01-foundations/02-six-goals-and-tradeoffs.md) |
| The five-step framework | [HLD/01-foundations/04](../HLD/01-foundations/04-five-step-interview-framework.md) |
| Functional vs non-functional requirement split | [HLD/01-foundations/04](../HLD/01-foundations/04-five-step-interview-framework.md) |
| Notification-system clarify questions (bad vs good candidate) | [HLD/01-foundations/04](../HLD/01-foundations/04-five-step-interview-framework.md) |
| The estimation template and the messaging-system worked example | [HLD/01-foundations/05](../HLD/01-foundations/05-capacity-estimation-playbook.md) |
| The "professional answer" pattern (why estimate) | [HLD/01-foundations/04](../HLD/01-foundations/04-five-step-interview-framework.md) · [05](../HLD/01-foundations/05-capacity-estimation-playbook.md) |
| The "right way vs wrong way" cache/NoSQL answers | [HLD/01-foundations/04](../HLD/01-foundations/04-five-step-interview-framework.md) |
| MTBF/MTTR, reliability vs availability, the nines | [HLD/01-foundations/06](../HLD/01-foundations/06-availability-numbers-and-slos.md) |
| Availability anchor numbers used in estimation | [HLD/01-foundations/05](../HLD/01-foundations/05-capacity-estimation-playbook.md) · [06](../HLD/01-foundations/06-availability-numbers-and-slos.md) |

### T3 — The FAANG question (URL shortener)

| Topic in the transcript | Where it lives |
|---|---|
| Functional requirements (create, redirect, alias, expiry, click count, delete) | [Case-Studies/HLD/01](../Case-Studies/HLD/01-url-shortener.md) |
| Non-functional requirements (100M DAU, 1k writes/s, 100k reads/s, uniqueness + unpredictability) | [Case-Studies/HLD/01](../Case-Studies/HLD/01-url-shortener.md) |
| The 86.4M/day write calculation | [Case-Studies/HLD/01](../Case-Studies/HLD/01-url-shortener.md) §Step 2 |
| The full architecture: CDN → LB → API Gateway → rate limiter → URL/redirect services → Redis → NoSQL → replica → Kafka → streaming → analytics | [Case-Studies/HLD/01](../Case-Studies/HLD/01-url-shortener.md) §Step 4 |
| Redis replica for cache fault tolerance | [Case-Studies/HLD/01](../Case-Studies/HLD/01-url-shortener.md) · [HLD/03-scaling/03](../HLD/03-scaling/03-redis-deep-dive.md) |
| Kafka click events → streaming → analytics DB | [Case-Studies/HLD/01](../Case-Studies/HLD/01-url-shortener.md) §Deep dive D |
| The trade-off enumeration: hashing, base62, distributed IDs, sequential IDs | [Case-Studies/HLD/01](../Case-Studies/HLD/01-url-shortener.md) §Deep dive A |
| API design (POST + GET + analytics) | [Case-Studies/HLD/01](../Case-Studies/HLD/01-url-shortener.md) §Step 3 |

### T4 — Networking

| Topic in the transcript | Where it lives |
|---|---|
| DNS resolution, record types, TTL, propagation | [HLD/02-networking/01](../HLD/02-networking/01-dns-deep-dive.md) |
| CDN mechanics, edge caching, origin shield, invalidation | [HLD/02-networking/02](../HLD/02-networking/02-cdn-deep-dive.md) |
| Forward vs reverse proxy, TLS termination, caching proxy | [HLD/02-networking/03](../HLD/02-networking/03-forward-and-reverse-proxy.md) |
| API gateway responsibilities, rate limiting algorithms | [HLD/02-networking/04](../HLD/02-networking/04-api-gateway-and-rate-limiting.md) |

### T5 — Load balancers, Redis & cache invalidation

| Topic in the transcript | Where it lives |
|---|---|
| L4 vs L7 load balancing, algorithms, health checks, session affinity | [HLD/03-scaling/01](../HLD/03-scaling/01-load-balancers.md) |
| Caching layers, strategies (cache-aside, write-through, write-back), eviction | [HLD/03-scaling/02](../HLD/03-scaling/02-caching-strategies-and-layers.md) |
| Redis data structures, persistence, clustering, use cases | [HLD/03-scaling/03](../HLD/03-scaling/03-redis-deep-dive.md) |
| Invalidation strategies, TTLs, staleness, stampede, consistency | [HLD/03-scaling/04](../HLD/03-scaling/04-cache-invalidation-and-consistency.md) |
| Sharding, replication, consistency models (supporting material) | [HLD/03-scaling/05](../HLD/03-scaling/05-sharding-replication-and-consistency.md) |

### T6 — SQL, NoSQL, Vector DB, Blob Storage

| Topic in the transcript | Where it lives |
|---|---|
| SQL internals, indexing, transactions, replication, sharding | [HLD/04-data/01](../HLD/04-data/01-sql-deep-dive.md) |
| NoSQL families (key-value, document, wide-column, graph), trade-offs | [HLD/04-data/02](../HLD/04-data/02-nosql-deep-dive.md) |
| Vector databases, embeddings, ANN indexes (HNSW, IVF), hybrid search | [HLD/04-data/03](../HLD/04-data/03-vector-databases.md) |
| Blob/object storage, tiering, CDN integration, lifecycle | [HLD/04-data/04](../HLD/04-data/04-blob-and-object-storage.md) |
| The selection decision guide | [HLD/04-data/05](../HLD/04-data/05-choosing-a-database-decision-guide.md) |

### T7 — Production RAG

| Topic in the transcript | Where it lives |
|---|---|
| The ingestion pipeline (load → chunk → embed → index) | [HLD/05-ai-systems/01](../HLD/05-ai-systems/01-production-rag-system.md) |
| The query pipeline (embed → retrieve → rerank → generate) | [HLD/05-ai-systems/01](../HLD/05-ai-systems/01-production-rag-system.md) |
| Chunking strategies | [HLD/05-ai-systems/01](../HLD/05-ai-systems/01-production-rag-system.md) |
| Hybrid search, RRF, reranking | [HLD/05-ai-systems/01](../HLD/05-ai-systems/01-production-rag-system.md) · [02](../HLD/05-ai-systems/02-advanced-rag-patterns.md) |
| Advanced patterns (HyDE, contextual retrieval, GraphRAG, RAPTOR, self-RAG, CRAG) | [HLD/05-ai-systems/02](../HLD/05-ai-systems/02-advanced-rag-patterns.md) |
| Evaluation (recall@k, faithfulness, RAGAS) and observability | [HLD/05-ai-systems/03](../HLD/05-ai-systems/03-rag-evaluation-and-observability.md) |
| Guardrails, prompt injection | [HLD/05-ai-systems/04](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md) |
| The end-to-end production case study | [Case-Studies/HLD/09](../Case-Studies/HLD/09-production-rag-case-study.md) |

### T8–T11 — the nitmonk agent-engineering series

Admitted 2026-10-03. These four sources sit **beneath** the AI-systems pillar: where T7 teaches *what* a RAG system is, this series teaches the agent-engineering layer — how context, state and memory are actually managed, and what you assemble to build one. They are numbered T8–T11 to keep them distinct from the seven roadmap transcripts above.

| # | Source | Covers |
|---|---|---|
| **T8** | *24 Hours of AI Agent Engineering — Part 1 (Complete Course)* · 2,459 lines | Tokens/BPE, embeddings, context window, temperature, prompt anatomy, prompt injection, chunking, ANN indexes, BM25/hybrid/rerank, graph/agentic/corrective/adaptive RAG, hallucination and grounding, RAG security |
| **T9** | *24 Hours of AI Agent Engineering — Part 2 (Hands-On, MCP, Production & AI Interviews)* · 3,361 lines | Long-term memory horizons, AWS Bedrock AgentCore, production agent security, MCP deep dive, progressive discovery, middleware, the LangChain/LangGraph/LangFuse hands-on build, multi-agent, enterprise cost |
| **T10** | *23+ Context Engineering & AI Agent Memory Techniques* · 257 lines | Context vs state vs memory, the RAM/disk/OS model, the six verbs, compaction vs compression, the memory taxonomy, memory failure modes and the poisoning taxonomy |
| **T11** | *LangChain + LangFuse — Build an AI Agent From Scratch* Parts 1–3 and *AI Agents LIVE* Parts 1–2 | The concrete stack: `uv`, `create_agent`, `@tool`, Pydantic `response_format`, checkpointers, `thread_id`, DynamoDB saver, IAM, tracing, middleware, supervisor multi-agent |

| Topic in the source | Where it lives |
|---|---|
| Context vs state vs memory; the RAM/disk/OS analogy | [HLD/05-ai-systems/08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) §1.1–1.2 |
| The six verbs of context engineering; the real window's budget | [HLD/05-ai-systems/08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) §3.9–3.10 |
| Compaction vs compression; structured notes | [HLD/05-ai-systems/08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) §3.11 |
| Refresh, isolation, prune, order; just-in-time memory | [HLD/05-ai-systems/08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) §3.12–3.13 |
| The short-lived/long-lived memory taxonomy; environment and project memory | [HLD/05-ai-systems/08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) §4.1 |
| "Context grows and then it rots" — the decay modes | [HLD/05-ai-systems/08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) §6.0 |
| False memory · retrieval failure · retrieval pollution | [HLD/05-ai-systems/08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) §6.17–6.19 |
| The memory/RAG/tool poisoning taxonomy | [HLD/05-ai-systems/08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) §6.1 |
| The agent harness; agent vs workflow; unbounded loops | [HLD/05-ai-systems/06](../HLD/05-ai-systems/06-agentic-system-design.md) |
| MCP host/client/server; T/P/R primitives; **progressive discovery**; meta-tools | [HLD/05-ai-systems/07](../HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md) |
| Tool poisoning; production agent security; AgentCore policies | [HLD/05-ai-systems/04](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md) |
| Tokens/BPE; embeddings; temperature; prompt anatomy and injection | [HLD/05-ai-systems/09](../HLD/05-ai-systems/09-fine-tuning-vs-rag-vs-prompting.md) |
| Chunking; ANN indexes; hybrid/rerank; RAG variants | [HLD/05-ai-systems/01](../HLD/05-ai-systems/01-production-rag-system.md) · [02](../HLD/05-ai-systems/02-advanced-rag-patterns.md) |
| Hallucination; citation and grounding; RAG eval | [HLD/05-ai-systems/03](../HLD/05-ai-systems/03-rag-evaluation-and-observability.md) |
| **The whole build stack** — `uv`, `create_agent`, tools, checkpointers, `thread_id`, DynamoDB, LangFuse, middleware, supervisor agents | **[HLD/05-ai-systems/11](../HLD/05-ai-systems/11-agent-engineering-hands-on-stack.md)** |

> **A note on transcript quality, because it changes how you should read the numbers.** These are NoteGPT auto-transcripts of live coding sessions: unpunctuated, no speaker labels, and the instructor debugs aloud while building — so a failed run appears in the text next to the fix. Figures are sometimes garbled. Every `[S]` number in the files above was sanity-checked before being labelled; where the transcript's rendering was wrong or ambiguous, the file says so or omits it. This is the same discipline applied to T2's estimation example — see [HLD/01-foundations/05](../HLD/01-foundations/05-capacity-estimation-playbook.md) §3.3.

Four specific caveats, recorded here so they need not be rediscovered:

- **Prompt-caching reduction is stated as both ~80% and ~90%** across the series (T8 says 90%, T9 says 80% in one passage and 90% in another). Read it as *80–90% of the cached prefix portion*, never as a discount on the whole bill. Both files that cite it ([09](../HLD/05-ai-systems/09-fine-tuning-vs-rag-vs-prompting.md), [11](../HLD/05-ai-systems/11-agent-engineering-hands-on-stack.md)) say so inline.
- **Part 1 announces a segment it never delivers.** Its agenda lists "enterprise RAG architecture" (line 854) but the recording ends without it — presumably deferred to Part 2, which does not cover it under that name either. It is therefore **not** claimed as covered anywhere in this repository.
- **Vendor and product names are unreliable** in these auto-transcripts. The typed-decision product appears under several different short names; DeepEval is rendered "deep"; RAGAS's fourth metric is rendered "correction" (the canonical names are context precision / context recall). The files describe the *pattern* and tell the reader not to cite the name.
- **The OS/RAM/disk/filesystem/CPU-cache analogy is stated only in T10** — not in T9, which has the drawer/desk split and the in-memory-saver/DynamoDB one but never names the metaphor. [HLD/05-ai-systems/06](../HLD/05-ai-systems/06-agentic-system-design.md) §1.1 and [08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) §1.2 both attribute it to T10.

### T1 + T2 → the LLD pillar (phase 3)

The transcripts **name** the LLD syllabus without teaching it, with one exception. This table is the honest mapping: it says for each LLD topic whether the source *teaches* it or merely *names* it.

| Topic named in the source | Which transcript | Transcript treatment | Where it lives |
|---|---|---|---|
| HLD vs LLD, the definitions | **T2** (Day 1), [23:04]–[26:46] | **Taught** — T2 gives the HLD/LLD distinction in full | [HLD/01-foundations/03](../HLD/01-foundations/03-hld-vs-lld.md) · [LLD/01](../LLD/01-what-is-lld-and-how-to-answer.md) |
| **Parking lot** — `ParkingLot` → `List<Floor>`; `Floor` → `List<ParkingSpot>`; `findAvailableSpot() / reserveSpot() / releaseSpot()` | **T2** (Day 1), [26:48]–[27:38] | **Taught** — the only LLD example the source actually walks through | [LLD/problems/01](../LLD/problems/01-parking-lot.md) — and the class structure there is the transcript's own |
| LRU / LFU | **T1** (Roadmap), [9:20] | **Named only** — *"एलआरयू क्या होता है? एलएफयू क्या होता है?"*, a syllabus item | [LLD/problems/02](../LLD/problems/02-lru-lfu-cache.md) · brief worked example in [HLD/01-foundations/03](../HLD/01-foundations/03-hld-vs-lld.md) |
| BookMyShow — seat locking, concurrent booking | **T1** (Roadmap), [9:15]–[9:18] | **Named only** — three questions, no answer | [LLD/problems/04](../LLD/problems/04-bookmyshow-seat-locking.md) (classes) · [Case-Studies/HLD/07](../Case-Studies/HLD/07-bookmyshow-seat-booking.md) (distributed) |
| Hotel management system, booking | **T1** (Roadmap), [9:22] | **Named only** | [LLD/problems/03](../LLD/problems/03-hotel-management.md) |
| "OOPS patterns", coding problems | **T1** (Roadmap), [9:09]–[9:13] | **Named only** — the phrase is used, no pattern is taught | [LLD/02](../LLD/02-oop-principles-and-solid.md) · [LLD/03](../LLD/03-design-patterns.md) |
| Class diagrams, methods, constructors | **T2** (Day 1), [26:02]–[26:46] | **Named only** — mentioned as what LLD consists of | [LLD/05](../LLD/05-uml-and-class-diagrams.md) · [LLD/06](../LLD/06-machine-coding-interview-framework.md) |
| Concurrency (implied by "concurrent booking") | **T1** (Roadmap), [9:18] | **Named only** | [LLD/04](../LLD/04-concurrency-and-threading-in-lld.md) |

> **Read this as a warning, not a reassurance.** Seven of these eight rows say *"named only"*. The source calls LLD *"ये सबसे ज़्यादा इंपॉर्टेंट है आपका"* — the most important part for you — at **T1 [9:37]**, and then teaches one example of it. That is precisely why the [`LLD/`](../LLD/) pillar is the largest piece of **supplementary** material in this repository — it fills a gap the source identified but left open. Every LLD file says so at the top.

> **Note the split across two transcripts, which is easy to get wrong.** The *syllabus* is announced in **T1**, the roadmap video, in a forty-second stretch at [9:09]–[9:27]. The only *teaching* is in **T2**, the Day 1 lecture. So "the source names BookMyShow" and "the source teaches LLD" are statements about two different videos — and anyone who searches only Day 1 for BookMyShow will find nothing.

### T1 → the source-named case studies added to close coverage gaps

| Source-named design | Transcript treatment | Where it lives |
|---|---|---|
| **Netflix system** (T1 phase 4: *"ML practice design — the Netflix system"*) | **Named only** | [Case-Studies/HLD/12](../Case-Studies/HLD/12-netflix-recommendation-and-streaming.md) |
| **Fraud detection system** (T1 phase 4) | **Named only** | [Case-Studies/HLD/13](../Case-Studies/HLD/13-fraud-detection-system.md) |
| **Autocomplete** (T1 phase 2: *"Twitter, autocomplete, data storage"*) | **Named only** | [Case-Studies/HLD/14](../Case-Studies/HLD/14-autocomplete-and-typeahead.md) |
| **"Distribute a huge model weight file across thousands of machines"** (T1, example interview question) | **Named only** — posed as an example question, never answered | [HLD/05-ai-systems/10](../HLD/05-ai-systems/10-distributed-model-weight-serving.md) |
| The capstone project and the mock interview (T1 phase 5) | **Described** in outline | [00-meta/03](03-capstone-project-and-mock-interview.md) |

---

## 3. Topics with no transcript coverage

These come from the **reference repositories** and standard industry practice, not from the video transcripts. They are marked here so you know which material is transcript-grounded and which is supplementary.

Three distinct kinds of supplementary material live in this table, and they are not equally distant from the source:

- **Source-listed but never designed.** The roadmap names these systems (Netflix, fraud detection, autocomplete) as topics the course covers. The files here fill in what the source promised. They are the *least* speculative supplement, because at least the topic choice is the source's own.
- **Source-named but never taught.** The whole LLD pillar. The source calls LLD the most important part and then teaches only the parking lot. Everything else in [`LLD/`](../LLD/) is standard interview canon.
- **Beyond the source entirely.** LLM serving internals, MCP, memory engineering, the ops playbooks — topics a modern interview asks about that the seven transcripts never reach.

| Topic | Source | File |
|---|---|---|
| LLM serving internals (prefill/decode, KV cache, continuous batching, paged attention, speculative decoding, quantization) | Reference repos + industry practice | [HLD/05-ai-systems/05](../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md) |
| Agentic design (ReAct, durable execution, step budgets) | `LLM-Coding-Series` code + reference repos | [HLD/05-ai-systems/06](../HLD/05-ai-systems/06-agentic-system-design.md) |
| MCP, A2A, multi-agent topologies | Reference repos | [HLD/05-ai-systems/07](../HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md) |
| Memory and context engineering | `langgraph_memory.py` + reference repos | [HLD/05-ai-systems/08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) |
| Fine-tuning vs RAG vs prompting | Reference repos | [HLD/05-ai-systems/09](../HLD/05-ai-systems/09-fine-tuning-vs-rag-vs-prompting.md) |
| WhatsApp, notification, Instagram, YouTube, Uber, BookMyShow case studies | Reference repos (`16-case-studies`) | [Case-Studies/HLD/02–08](../Case-Studies/HLD/) |
| AI support agent, agentic coding assistant | Reference repos | [Case-Studies/HLD/10–11](../Case-Studies/HLD/) |
| **The entire LLD pillar** — OOP/SOLID, design patterns, concurrency, UML, the machine-coding framework, and all 16 problems | **Supplementary.** The source names these topics (see §2) but does not teach them, apart from the parking lot. The content follows the standard interview canon. | [LLD/](../LLD/) |
| **LLD case studies** — URL shortener, WhatsApp, Uber matching, Instagram feed, notifications, taken down to classes | Supplementary — these are bridges from the HLD designs, written for this repo | [Case-Studies/LLD](../Case-Studies/LLD/) |
| **Netflix, fraud detection, autocomplete case studies** | **Source-named, supplementary content.** The source lists all three as topics to be covered and never designs any of them. See §2. | [Case-Studies/HLD/12–14](../Case-Studies/HLD/) |
| **Distributing a huge model weight file** | Source-named — T1 poses it verbatim as an example interview question and does not answer it | [HLD/05-ai-systems/10](../HLD/05-ai-systems/10-distributed-model-weight-serving.md) |
| **The capstone project and the mock-interview guide** | Source-described in outline (T1 phase 5) | [00-meta/03](03-capstone-project-and-mock-interview.md) |
| All operationalisation playbooks | Industry practice + reference repos | [Operations](../Operations/) |
| Behavioural and project questions | Reference repos | [Interview-Questions/07](../Interview-Questions/07-behavioral-and-project-questions.md) |
| OWASP LLM Top 10 | Industry standard | [HLD/05-ai-systems/04](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md) · [Operations/08](../Operations/08-security-guardrails-and-compliance.md) |
| EU AI Act, SOC2, HIPAA, GDPR | Industry standard | [Operations/08](../Operations/08-security-guardrails-and-compliance.md) |

---

## 4. The complete topic inventory

Every topic named in the roadmap video (T1) and where it is covered.

### Classical

| Topic | File | Covered |
|---|---|---|
| URL shortener | [Case-Studies/HLD/01](../Case-Studies/HLD/01-url-shortener.md) | ✅ Full case study |
| WhatsApp / chat | [Case-Studies/HLD/02](../Case-Studies/HLD/02-whatsapp-chat-system.md) | ✅ Full case study |
| Notification system | [Case-Studies/HLD/03](../Case-Studies/HLD/03-notification-system.md) | ✅ Full case study |
| YouTube | [Case-Studies/HLD/05](../Case-Studies/HLD/05-youtube-upload-and-recommendation.md) | ✅ Full case study |
| Twitter/X | [Case-Studies/HLD/08](../Case-Studies/HLD/08-twitter-x-feed-and-search.md) | ✅ Full case study |
| Instagram | [Case-Studies/HLD/04](../Case-Studies/HLD/04-instagram-news-feed.md) | ✅ Full case study |
| Uber | [Case-Studies/HLD/06](../Case-Studies/HLD/06-uber-ride-hailing.md) | ✅ Full case study |
| BookMyShow + seat booking | [Case-Studies/HLD/07](../Case-Studies/HLD/07-bookmyshow-seat-booking.md) (distributed) · [LLD/problems/04](../LLD/problems/04-bookmyshow-seat-locking.md) (classes) | ✅ Both levels |
| **Parking lot** | [LLD/problems/01](../LLD/problems/01-parking-lot.md) | ✅ **Taught in the source** — the only LLD example it walks through |
| LRU / LFU | [LLD/problems/02](../LLD/problems/02-lru-lfu-cache.md) · [HLD/01-foundations/03](../HLD/01-foundations/03-hld-vs-lld.md) | ✅ Full LLD problem + worked example |
| Hotel management | [LLD/problems/03](../LLD/problems/03-hotel-management.md) | ✅ Full LLD problem |
| Autocomplete | [Case-Studies/HLD/14](../Case-Studies/HLD/14-autocomplete-and-typeahead.md) | ✅ **Full case study** (was previously only "in context") |
| Rate limiting | [HLD/02-networking/04](../HLD/02-networking/04-api-gateway-and-rate-limiting.md) (distributed) · [LLD/problems/11](../LLD/problems/11-rate-limiter.md) (classes) | ✅ Both levels |
| Sharding | [HLD/03-scaling/05](../HLD/03-scaling/05-sharding-replication-and-consistency.md) | ✅ Deep dive |
| Circuit breaker | [Operations/07](../Operations/07-reliability-patterns.md) | ✅ Deep dive |
| Kafka | [Case-Studies/HLD/01](../Case-Studies/HLD/01-url-shortener.md) · [Operations/07](../Operations/07-reliability-patterns.md) · [LLD/problems/13](../LLD/problems/13-pub-sub-system.md) | ✅ In context |
| REST | [HLD/01-foundations/04](../HLD/01-foundations/04-five-step-interview-framework.md) §Step 3 · [HLD/02-networking/04](../HLD/02-networking/04-api-gateway-and-rate-limiting.md) | ✅ |
| GraphQL | [HLD/02-networking/04](../HLD/02-networking/04-api-gateway-and-rate-limiting.md) | ✅ Comparison |
| CI/CD | [Operations/04](../Operations/04-deployment-release-and-cicd.md) | ✅ Playbook |
| Recommendation systems | [Case-Studies/HLD/05](../Case-Studies/HLD/05-youtube-upload-and-recommendation.md) · [Case-Studies/HLD/12](../Case-Studies/HLD/12-netflix-recommendation-and-streaming.md) | ✅ Two full case studies |
| Fraud detection | [Case-Studies/HLD/13](../Case-Studies/HLD/13-fraud-detection-system.md) | ✅ **Full case study** (was previously only inside the Uber study) |
| Netflix-style ML systems | [Case-Studies/HLD/12](../Case-Studies/HLD/12-netflix-recommendation-and-streaming.md) | ✅ **Full case study** (was previously only "patterns") |
| Distributing a model weight file | [HLD/05-ai-systems/10](../HLD/05-ai-systems/10-distributed-model-weight-serving.md) | ✅ Full design |

### AI / GenAI

| Topic | File | Covered |
|---|---|---|
| GenAI / agentic AI | [HLD/05-ai-systems/06](../HLD/05-ai-systems/06-agentic-system-design.md) | ✅ Deep dive — deepened from T9 (the harness, loop engineering, long-horizon execution, reflection/critique/debate, routing patterns, enterprise cost) |
| MCP | [HLD/05-ai-systems/07](../HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md) | ✅ Deep dive — deepened from T9 (the JSON-RPC wire, the per-tool approval gate, code mode, enterprise server patterns, WebMCP) |
| A2A | [HLD/05-ai-systems/07](../HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md) | ✅ Deep dive |
| Guardrails | [HLD/05-ai-systems/04](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md) | ✅ Deep dive — deepened from T9 (the injection-surface table, the four-rung defence ladder, the poisoning taxonomy, RBAC/ACL isolation, AWS AgentCore) |
| LLM inference | [HLD/05-ai-systems/05](../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md) | ✅ Deep dive |
| Multi-agent systems | [HLD/05-ai-systems/07](../HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md) | ✅ Deep dive |
| Fine-tuning vs RAG vs prompt engineering | [HLD/05-ai-systems/09](../HLD/05-ai-systems/09-fine-tuning-vs-rag-vs-prompting.md) | ✅ Deep dive — deepened from T8 (tokens/BPE, embeddings, the context window and sampling, prompt anatomy, output contracts, prompt caching, reasoning strategies, injection) |
| RAG pipeline | [HLD/05-ai-systems/01](../HLD/05-ai-systems/01-production-rag-system.md) | ✅ Deep dive — deepened from T8 (the chunking catalogue, the ANN index mechanics, BM25-vs-dense, vector-database selection, the failure cascade, the stage-latency profile) |
| Advanced RAG patterns | [HLD/05-ai-systems/02](../HLD/05-ai-systems/02-advanced-rag-patterns.md) | ✅ Deep dive — from T8 (the retrieval-strategy catalogue, the variant taxonomy by trigger, graph RAG mechanics, the effort-by-difficulty ladder) |
| RAG evaluation and observability | [HLD/05-ai-systems/03](../HLD/05-ai-systems/03-rag-evaluation-and-observability.md) | ✅ Deep dive — from T8 (rank-before-generation, the hallucination taxonomy, the four-part framework, the observe/evaluate/regress split, citation and grounding) |
| Context engineering (the six verbs, window rot) | [HLD/05-ai-systems/08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) | ✅ Deep dive — from T10 |
| Agent memory taxonomy (incl. environment / project) | [HLD/05-ai-systems/08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) | ✅ Deep dive — from T10 |
| Agent harness (the OS analogy: context = RAM, memory = disk) | [HLD/05-ai-systems/06](../HLD/05-ai-systems/06-agentic-system-design.md) §1.1 · [HLD/05-ai-systems/08](../HLD/05-ai-systems/08-memory-and-context-engineering.md) §1.2 | ✅ Deep dive — from T10 (stated explicitly there); T9 gives only the drawer/desk split |
| Progressive discovery for large tool catalogues | [HLD/05-ai-systems/07](../HLD/05-ai-systems/07-mcp-tool-use-and-multi-agent.md) | ✅ Deep dive — from T9 |
| The build stack (LangChain/LangGraph/LangFuse/DynamoDB) | [HLD/05-ai-systems/11](../HLD/05-ai-systems/11-agent-engineering-hands-on-stack.md) | ✅ **Full walkthrough** — from T11 |
| LLMOps | [Operations/10](../Operations/10-llmops-for-ai-systems.md) | ✅ Playbook |

---

## 5. How to extend this knowledge base

If you want to add a topic, follow the same discipline the sources did:

1. **Find the transcript section** that covers it (if any) and cite it at the top of the file.
2. **Use the 13-section template** from [00-meta/01](01-how-to-use-and-template.md) §4.
3. **Weight the exceptions section heaviest** — it is where interview value lives.
4. **Include ≥ 12 Q&A with spoken answers**, and at least three pushbacks.
5. **Add concrete numbers** — defaults, thresholds, anchors — never abstract advice.
6. **Add the file to the [README](../README.md) index** and to the relevant cheat sheet.
7. **Cross-link both ways** — from the new file and from at least one existing file.
8. **Label your numbers.** Mark every figure as `[S]` (stated in a source), `[A]` (an assumption you are making), or `[D]` (derived by arithmetic here). A file full of unlabelled numbers is indistinguishable from a file full of inventions, and in an interview the difference is the whole answer.

**The quality bar:** if a section could have been written without reading the source, it probably doesn't belong in this repository.

**The honesty bar, which is separate and stricter:** a section that *could not* have been written from the source belongs here all the same — but it must say so. The LLD pillar, the three added case studies and the model-weight design are all material the source names and does not teach. They are here because the gap is real; they are labelled because the gap is real. Neither half of that sentence is optional.

---

## 6. Q&A

**Q1. Which transcripts are the most important?**
T2 (Day 1 foundations) is the framework everything else hangs on; T3 (URL shortener) is the flagship worked example; T7 (production RAG) is the deepest AI-specific one. If you only had three, those are the three.

**Q2. Is `nitmonk` still excluded?**
No — that instruction was lifted on 2026-10-03. It is now a source, and it is the origin of the entire agent-engineering build layer: the context/state/memory framing, the six verbs of context engineering, compaction vs compression, the memory taxonomy, MCP progressive discovery, and the hands-on LangChain/LangGraph/LangFuse/DynamoDB stack. See T8–T11 in §2 and the tertiary-source table in §1. One sub-folder stays excluded: **`nitmonk/archive/`**, which holds superseded duplicates and the Hindi-origin recordings of material already available in English outside it.

**Q3. How do I verify a claim?**
Use §2 — each file maps back to the transcript section and topic that produced it. Then read the transcript directly.

**Q4. Which content is transcript-grounded and which is supplementary?**
§2 lists the transcript-grounded material; §3 lists topics that come from the reference repositories and industry practice because the transcripts don't cover them. Knowing which is which matters if you're asked to defend a claim in an interview.

**Q5. What's missing that I might expect?**
Two things, and both are now stated rather than hidden. **First**, the LLD pillar is *supplementary*: the source names the LLD syllabus — parking lot, LRU/LFU, hotel management, BookMyShow seat locking, OOPS patterns, class diagrams — and teaches only the parking lot, in about eighty seconds. Everything else in [`LLD/`](../LLD/) comes from the standard interview canon, not from the transcripts. **Second**, on the AI side, formal ML-training-pipeline design (feature stores, training/serving skew, retraining cadence) is covered at the patterns level in [`HLD/05-ai-systems`](../HLD/05-ai-systems/) rather than as a full end-to-end pipeline design — the closest thing to a full treatment is [fraud detection](../Case-Studies/HLD/13-fraud-detection-system.md), which walks the label-delay and train/serve-skew problems in detail. Everything else the roadmap video names by topic now has a dedicated file.

**Q5b. How do I tell a sourced number from a made-up one?**
Every figure in a case study or design file is labelled. **`[S]`** = stated in a transcript or a reference repo; **`[A]`** = an assumption this repo makes, chosen to be defensible and stated so you can change it; **`[D]`** = derived — arithmetic done here from `[S]` and `[A]` inputs. In an interview, quote `[S]` freely, say `[A]` out loud as an assumption, and show the `[D]` working. The labels exist so that you never present this repo's arithmetic as if the source had done it.

**Q6. Can I add my own case studies?**
Yes — follow §5 and keep to the 13-section template, so the new file matches everything else.

**Q7. Why include reference-repo material at all if it's not from the transcripts?**
Because the transcripts name the topics (LLM serving, MCP, agentic design) without covering them in the depth an interview requires. Leaving those gaps would make the roadmap incomplete.

**Q8. How do I cite a file in my own notes?**
Use the relative paths shown in the cross-links section of every file — they resolve correctly from anywhere in the repository.

---

## 7. Cheat sheet

| | |
|---|---|
| **Transcripts** | 7 roadmap transcripts, 6,880 lines |
| **Agent-engineering series** | 4 sources (T8–T11), ~6,700 further lines — admitted 2026-10-03 |
| **Most important** | T2 (foundations) · T3 (URL shortener) · T7 (production RAG) · T10 (context & memory) |
| **Excluded** | `nitmonk/archive/` — by instruction |
| **Transcript-grounded** | §2 — foundations, networking, scaling/caching, data, RAG, URL shortener |
| **Supplementary** | §3 — the whole LLD pillar, LLM serving, agents, MCP, memory, ops playbooks, the Netflix/fraud/autocomplete studies |
| **Figure labels** | `[S]` sourced · `[A]` assumed · `[D]` derived here. Quote `[S]`, declare `[A]`, show `[D]` |
| **Topic coverage** | §4 — the full roadmap inventory with ✅ |
| **To extend** | 13-section template, cite the source, weight exceptions, ≥ 12 Q&A, concrete numbers, cross-link both ways, label the numbers |

---

## 8. Cross-links

- [How to use this knowledge base](01-how-to-use-and-template.md)
- [README — master index](../README.md)
- Foundations: [01 what is system design](../HLD/01-foundations/01-what-is-system-design-and-why.md) → [06 availability & SLOs](../HLD/01-foundations/06-availability-numbers-and-slos.md)
- Flagship case study: [URL shortener](../Case-Studies/HLD/01-url-shortener.md)
- Deepest AI file: [Production RAG system](../HLD/05-ai-systems/01-production-rag-system.md)
