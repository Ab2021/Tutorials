# Case Study: Enterprise Knowledge Base RAG (10M docs, 100k DAU)

**TL;DR** — This is a full worked system design interview answer for the most common GenAI system design question: *"Design an enterprise knowledge assistant over 10 million internal documents serving 100,000 daily active users."* It follows the course's mandatory five-step framework — **clarify requirements → capacity estimation → define the API → high-level design → deep dive and trade-offs** — and it uses the standard estimation template (`QPS = requests/day ÷ 86,400`, `peak = 3× QPS`, `storage = requests/day × record size`, `bandwidth = QPS × record size`). The headline numbers: **9.3 QPS average, 28 QPS peak, ~80M chunks, ~490 GB of raw vectors (~640 GB with HNSW), ~$0.008 per query all-in, ~$6,400/day at full traffic**, of which generation is roughly 80%. The three deep dives that win the interview are **hybrid retrieval with RRF and reranking**, **permission-aware retrieval**, and **freshness/staleness** — because those are the three things that actually break an enterprise RAG system, and they are the three most candidates never mention.

> **60-second interview answer:** "Let me clarify requirements first. Functionally: users ask natural-language questions, get grounded answers with citations, over documents they already have permission to see, with freshness under 15 minutes. Non-functionally: 100k DAU, p95 under 3 seconds, 99.9% availability, answers must be citeable and permission-correct. Capacity: at 8 queries per user per day that's 800k requests per day, so 9.3 QPS average and 28 QPS at 3x peak. Ingestion of 10M documents at roughly 5,000 tokens each and 8 chunks per document gives 80 million chunks, which at 1536 dimensions and 4 bytes per float is about 490 GB of raw vectors, roughly 640 GB with HNSW overhead — that forces a sharded index and pushes me toward Matryoshka truncation or binary quantization. The API is a POST /v1/query taking the question plus a conversation ID, returning an answer, a citation list, and a confidence flag. High level: two pipelines — an ingestion pipeline and a query pipeline — with a gateway, a semantic cache, hybrid retrieval, a reranker, and generation. For the deep dive I'd pick permission-aware retrieval, because the failure mode there is a data breach, not a bad answer."

---

## 1. The question, unpacked

**The prompt a candidate receives:**

> "Design an enterprise knowledge assistant. The company has 10 million internal documents — wikis, PDFs, tickets, design docs, Slack exports. 100,000 employees use it daily. They ask questions in natural language and expect answers grounded in company documents."

**The trap:** this looks like "build RAG." It is actually three problems wearing one coat:

1. **A retrieval problem** — 10M documents, 80M chunks, 28 QPS at peak with a 3-second budget.
2. **A permissions problem** — employees have different access rights, and a retrieval miss is a quality bug while a permission miss is a breach.
3. **A data-engineering problem** — documents change, get deleted, get re-uploaded, and have 20 different formats.

Most candidates solve only the first. The interview is won on the other two.

---

## 2. Step 1 — Clarify requirements

**Never draw first.** The interviewer is grading whether you scope before you build. Ask, then state your assumptions explicitly so the interviewer can correct them.

### 2.1 Functional requirements

| # | Requirement | Clarifying question | Assumed answer |
|---|---|---|---|
| F1 | Natural-language Q&A over internal documents | Is it Q&A, or also summarisation and browsing? | Q&A with citations |
| F2 | Grounded answers with citations | Must every claim be traceable to a source? | **Yes — mandatory** |
| F3 | Permission-aware | Do answers respect the asker's existing access rights? | **Yes — hard requirement** |
| F4 | Multi-turn | Is there conversation context? | Yes, last 5 turns |
| F5 | Freshness | How quickly must a new document be searchable? | **< 15 minutes** |
| F6 | Sources | Which systems? | Wikis (Confluence), drives, PDFs, tickets (Jira), Slack |
| F7 | Feedback | Can users rate answers? | Yes — thumbs plus optional comment |
| F8 | Languages | Monolingual or multilingual? | English first, i18n later |
| F9 | Not in scope | What are we explicitly not doing? | No actions/writes; read-only; no web search; no code execution |

**The "what's in and what's out" move.** Stating F9 explicitly is worth real points: it shows you manage scope. "Read-only, no actions" removes an entire class of authorisation and guardrail complexity from the design.

### 2.2 Non-functional requirements

| # | Requirement | Target | Why this target |
|---|---|---|---|
| N1 | Scale | 100k DAU, 8 queries/user/day | Given |
| N2 | Latency | **p50 < 1.5 s, p95 < 3 s** | Interactive chat tolerance; 3 s is the point where users rephrase |
| N3 | Availability | **99.9%** (43m 12s downtime/month) | Internal tool, not payments; 99.99% is not worth the cost |
| N4 | Reliability | No data loss on ingestion; no dropped queries | Ingestion is idempotent and replayable |
| N5 | Consistency | **Eventual** for the index (< 15 min); **strong** for permissions | Permissions must never be stale |
| N6 | Cost | **< $0.02 per query** blended | Must be cheaper than the human time it saves |
| N7 | Security | No cross-tenant or cross-user leakage; full audit log | Non-negotiable |
| N8 | Freshness | New/changed docs searchable in < 15 min | Matches the wiki publish cadence |

### 2.3 The six goals, and which ones win here

The course's six goals — **scalability, availability, reliability, performance, consistency, cost** — fight each other. Your job is the *right balance for the use case*, not the perfect answer.

| Goal | Priority here | Reasoning |
|---|---|---|
| **Scalability** | High | 10M docs growing ~10%/year; 100k users |
| **Availability** | Medium | 99.9% is fine — an internal tool, and users have fallbacks |
| **Reliability** | High | Losing an ingested document is a silent, permanent quality bug |
| **Performance** | High | p95 < 3 s is the adoption threshold; slower and people stop using it |
| **Consistency** | Split | Eventual for content (15 min), **strong for permissions** |
| **Cost** | High | Per-query cost must be justifiable at 800k queries/day |

**The conflict to name explicitly:** *"Making the index more consistent costs latency and money — I could re-embed synchronously on every document change, but that pushes ingestion cost up and adds nothing, because a 15-minute delay on a wiki page is invisible to users. But permissions are the opposite: I will pay latency and complexity for strongly consistent permission checks, because a stale permission is a data breach."*

**Compare to the canonical examples** (this shows breadth): WhatsApp prioritises reliability and availability — never lose a message. A stock exchange prioritises consistency — the price must be identical everywhere. An Instagram feed prioritises performance — a slightly stale feed is fine if it loads fast. **This system prioritises reliability of permissions and cost-efficiency of answers, and accepts eventual consistency on content.**

---

## 3. Step 2 — Capacity estimation

Memorise the template. It is the difference between "I would use Redis" and "read QPS is 9.3, peak 28, a single Postgres node handles ~5,000 QPS, so the database is not the bottleneck — but the vector index at 80M chunks is."

### 3.1 The template

```
  ┌─────────────────────────────────────────────────────────────┐
  │  1. DAU                     = given                          │
  │  2. Requests/day            = DAU × requests per user per day│
  │  3. QPS                     = Requests/day ÷ 86,400          │
  │  4. Peak QPS                = QPS × 3                        │
  │  5. Storage                 = Requests/day × size per record │
  │  6. Bandwidth               = QPS × size per record          │
  └─────────────────────────────────────────────────────────────┘

  Useful constants:
    1 day    = 86,400 s ≈ 10^5 s
    1M req/day  ≈ 12 QPS
    1B req/day  ≈ 12,000 QPS
    1 KB × 1M records = 1 GB
```

### 3.2 Query-side estimation

| Step | Formula | Value |
|---|---|---|
| DAU | given | 100,000 |
| Queries per user per day | given | 8 |
| **Requests/day** | 100,000 × 8 | **800,000** |
| **Average QPS** | 800,000 ÷ 86,400 | **9.3 QPS** |
| **Peak QPS** | 9.3 × 3 | **28 QPS** |
| Peak concurrent streams | 28 × 2.5 s avg duration | **~70 in flight** |

**Sanity check — this is a small system by web standards.** 28 QPS peak is trivial for a web tier. The bottleneck is *not* request throughput; it is **latency budget** and **vector index size**. Saying this out loud is a strong signal: you have identified where the difficulty actually is.

### 3.3 Latency budget (the real constraint)

| Stage | p50 | p95 | Notes |
|---|---|---|---|
| Gateway + auth + permission resolution | 20 ms | 60 ms | Permission set may need a cache |
| Semantic cache lookup | 10 ms | 30 ms | Fail-open on timeout |
| Query rewrite (LLM, small model) | 120 ms | 250 ms | Skippable for simple queries |
| Embedding the query | 20 ms | 50 ms | Single short input |
| Hybrid retrieval (BM25 + ANN, parallel) | 60 ms | 120 ms | ANN dominates; BM25 is ~15 ms |
| RRF fusion | 5 ms | 10 ms | In-process |
| Rerank 50 → 5 | 90 ms | 180 ms | Cross-encoder on GPU |
| Context assembly + permission re-check | 15 ms | 40 ms | |
| **Generation (first token)** | **400 ms** | **900 ms** | Prefill over ~4k tokens |
| **Generation (full answer)** | **1,800 ms** | **2,700 ms** | ~350 output tokens at ~200 tok/s |
| Citation verification | 50 ms | 120 ms | Parallel with streaming |
| **End-to-end** | **~1.4 s** | **~2.9 s** | **Within the 3 s p95 target** |

**The point to make:** the generation step consumes ~60% of the p95 budget. Everything upstream must be fast because there is almost no headroom left. This is why you cannot add a second retrieval round or a multi-hop decomposition without a design change.

### 3.4 Index-side estimation

| Step | Formula | Value |
|---|---|---|
| Documents | given | 10,000,000 |
| Average tokens per document | 10 pages × 500 tok | ~5,000 tokens |
| Chunk size | 500–800 tokens, 12% overlap | ~650 effective |
| Chunks per document | 5,000 ÷ 650 | **~8** |
| **Total chunks** | 10M × 8 | **80,000,000** |
| Raw vector size (1536d, fp32) | 1536 × 4 B | **6,144 B/chunk** |
| **Raw vector storage** | 80M × 6,144 B | **~491 GB** |
| HNSW graph overhead (M=32) | ~1.3× | **~640 GB** |
| Extracted text storage | 80M × ~2.6 KB | **~208 GB** |
| Metadata + permissions | 80M × ~400 B | **~32 GB** |
| BM25 inverted index | ~25–30% of text | **~60 GB** |
| **Total index footprint** | | **~940 GB** |

**Immediate architectural consequence:** 940 GB does not fit in RAM on one node. That single number drives the whole storage design — sharding, and quantisation.

### 3.5 Vector index options at 80M chunks

| Configuration | Memory | Recall@10 | Notes |
|---|---|---|---|
| Flat (exact) | 491 GB | 1.00 | Impossible at 28 QPS |
| HNSW fp32, M=32 (2× sharded) | ~640 GB | 0.97 | 320 GB/node — needs 2× large-memory nodes |
| HNSW + **MRL 256-dim** | ~107 GB | ~0.955 | **Matryoshka truncation: ~6x reduction, <2% recall loss** |
| HNSW + **binary quantisation** (256-d) | **~3.3 GB** | ~0.92 pre-rerank | 32x reduction; Hamming distance ~10x faster |
| IVF-PQ (m=64, nbits=8) | ~10 GB | ~0.90 | Cheapest; most recall loss |

**The chosen configuration:** **MRL-256 embeddings with HNSW (M=32, ef_construction=256), plus binary quantisation for the first-pass candidate scan and fp32 rescoring of the top 500.**

Reasoning to state out loud: *"Binary quantisation alone loses about 5 points of recall, which I can afford because I rerank anyway — the vector search is a candidate generator, not the final ranker. Doing a binary prefilter at 3.3 GB, then rescoring the top 500 in fp32, then reranking the top 50, gets me most of the recall at a fraction of the memory. And the retrieval budget is 120 ms p95, which Hamming-distance search at 10x the speed comfortably meets."*

**Total working set after quantisation: ~130 GB of vectors + ~270 GB of text/BM25/metadata ≈ 400 GB across shards.** That fits comfortably on 4 shards of 128 GB.

### 3.6 Ingestion-side estimation

| Step | Value |
|---|---|
| Initial corpus | 10M docs → 80M chunks |
| Steady-state change rate | 3% of documents per day = 300k docs/day |
| Chunks to re-embed | 300k × 8 = 2.4M chunks/day |
| Embedding throughput needed | 2.4M ÷ 86,400 = **~28 chunks/s** average, ~85/s peak |
| Embedding cost (batch API, small model) | 2.4M × 650 tok × $0.02/1M = **~$31/day** |
| Contextual enrichment (Haiku-class, $0.0003/chunk) | 2.4M × $0.0003 = **~$720/day** |
| **Catch-up capacity** | Full recrawl of 80M chunks at 500 chunks/s = **~44 hours** |

**The design consequence:** contextual enrichment is 23x the cost of embedding. That is fine at 3%/day but catastrophic on a full recrawl — so enrichment must be **a tiered investment**, applied at ingest for high-value sources and skipped for bulk low-value content. This is the kind of specific, quantitative judgement that separates senior answers.

### 3.7 Cost per query

| Component | Tokens / units | Unit price | Cost |
|---|---|---|---|
| Query rewrite (small model) | 300 in / 60 out | $0.15 / $0.60 per 1M | $0.0001 |
| Query embedding | 60 tokens | $0.02 per 1M | ~$0 |
| Vector search (28 QPS on 4 shards) | — | amortised | $0.0002 |
| Rerank (50 docs, 100k tok) | 100k tok | $2.00 per 1M | $0.0002 |
| **Generation** | **3,500 in (85% cached) / 400 out** | **$3.00 / $15.00 per 1M** | **$0.0073** |
| Citation check (small model) | 500 in / 40 out | $0.15 / $0.60 | $0.0001 |
| **Total per query** | | | **~$0.008** |

**Daily and annual cost:**

| | Value |
|---|---|
| Raw per query | $0.008 |
| Cache hit rate (semantic cache) | 25% → effective $0.006 |
| **Daily query cost** | 800k × $0.006 = **~$4,800** |
| Daily ingestion cost | ~$750 |
| **Daily total** | **~$5,550** |
| **Annual** | **~$2.0M** |

**The 80/15/2/1 split — state this explicitly:**

```
   Generation  ████████████████████████████████████████  ~80%
   Reranking   ████████                                   ~15%
   Vector DB   █                                           ~2%
   Embedding   ▌                                           ~1%
```

**And the three levers that change the number most:**

| Lever | Saving | Cost |
|---|---|---|
| Prompt caching on the static prefix | ~35% of generation cost | Requires strict static-first ordering |
| Semantic cache (25% hit rate) | ~25% of total | Risk of stale/wrong cached answers |
| Route 50–60% of queries to a small model | ~40% of generation cost | Requires a router + quality monitoring |

**Combined, these take $0.008 to roughly $0.003 — a 2.7x reduction.** But this is only safe if you build the quality gates first.

### 3.8 Bandwidth and storage growth

| Metric | Formula | Value |
|---|---|---|
| Egress bandwidth | 28 QPS × ~14 KB response | **~0.4 MB/s** — negligible |
| Ingress (ingestion) | 28 chunks/s × 2.6 KB | ~0.07 MB/s — negligible |
| Storage growth | 3%/day compounding | ~1.6x/year |
| Year-1 end state | 940 GB × 1.6 | **~1.5 TB** |
| 3-year projection | | **~3 TB** |

**Bandwidth is not a constraint. Storage growth and index memory are.** Naming which dimension matters is itself the answer.

---

## 4. Step 3 — Define the API

Before drawing boxes, define the contract. It forces clarity about what the system actually does and shows the interviewer you think in interfaces.

### 4.1 Query endpoint

```http
POST /v1/query
Authorization: Bearer <user-jwt>
Content-Type: application/json

{
  "question": "What is our data retention policy for customer PII?",
  "conversation_id": "conv_8f3a...",     // optional; enables multi-turn
  "filters": {                            // optional; user-applied
    "source_types": ["confluence", "pdf"],
    "updated_after": "2025-01-01"
  },
  "options": {
    "stream": true,
    "max_citations": 5,
    "answer_mode": "grounded"             // grounded | extractive
  }
}
```

### 4.2 Query response

```json
{
  "request_id": "req_01HX...",
  "answer": "Customer PII is retained for 90 days after account closure...",
  "abstained": false,
  "confidence": 0.87,
  "citations": [
    {
      "index": 1,
      "doc_id": "doc_7712",
      "title": "Data Retention Policy v4.2",
      "url": "https://wiki.internal/policies/retention",
      "chunk_id": "doc_7712#c3",
      "span": "…retained for 90 days following…",
      "score": 0.91,
      "updated_at": "2026-02-14T09:12:00Z"
    }
  ],
  "usage": {
    "input_tokens": 3512, "cached_tokens": 2980,
    "output_tokens": 412, "latency_ms": 1840
  },
  "trace_id": "tr_9c2f..."
}
```

### 4.3 Why each field exists

| Field | Why it is in the contract |
|---|---|
| `abstained` | The system must be able to say "I don't know." A first-class field makes abstention a designed behaviour, not an accident |
| `confidence` | Drives UI treatment — a low-confidence answer gets a warning banner |
| `citations[].span` | Enables highlighting the exact supporting text; makes verification possible |
| `citations[].updated_at` | Freshness is visible to the user; a 3-year-old citation is a hint |
| `usage.cached_tokens` | Separates cache savings from real input cost — necessary for FinOps |
| `trace_id` | Joins the response to the retrieval trace for debugging |

### 4.4 Supporting endpoints

| Endpoint | Purpose | Notes |
|---|---|---|
| `POST /v1/feedback` | Thumbs up/down plus comment | Feeds the golden set and the eval pipeline |
| `GET /v1/conversations/{id}` | Conversation history | Last 5 turns used for rewriting |
| `POST /v1/admin/reindex` | Trigger a targeted reindex | Requires admin scope |
| `GET /v1/health` | Liveness and index freshness | Reports index lag in seconds |
| `POST /v1/documents` (internal) | Ingestion webhook | Called by source connectors |

### 4.5 The permission contract — the part people miss

```json
{
  "question": "...",
  "principal": {
    "user_id": "u_4412",
    "groups": ["eng", "eng-platform", "oncall"],
    "clearance": "internal",
    "region": "eu"
  }
}
```

**The critical rule:** the permission set is resolved **server-side from the authenticated token**, never accepted from the client. The API accepts the question and the token; it derives the principal itself.

**And the second critical rule:** permissions are applied as a **filter inside the vector search**, not as a post-filter on results. Post-filtering means you searched over documents the user cannot see — which is both a correctness bug (you may return fewer than top-k after filtering) and, if anything leaks into logs or traces, a breach.

---

## 5. Step 4 — High-level design

### 5.1 The two pipelines

```
════════════════════ INGESTION PIPELINE (asynchronous) ════════════════════

  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────────────┐
  │ Confluence│  │  Drive   │  │  Jira + Slack│  │  Change events  │
  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └────────┬────────┘
       └─────────────┴───────────────┴───────────────────┘
                             │
                    ┌────────▼─────────┐
                    │  INGEST QUEUE    │  Kafka; keyed by doc_id
                    │  (idempotent)    │  → ordering per document
                    └────────┬─────────┘
                             ▼
              ┌──────────────────────────────┐
              │  PARSE + NORMALISE           │  PDF/HTML/DOCX → text
              │  layout-aware OCR fallback   │  tables preserved atomically
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  ACL EXTRACTION              │  ← source ACLs → doc_ids
              │  (group + user principals)   │     written to a fast store
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  CHUNK                       │  recursive 500-800 tok,
              │  parent-child for long docs  │  12% overlap, AST for code
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  ENRICH (tiered)             │  context header + metadata
              │  high-value sources only     │  $0.0003/chunk
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  EMBED                       │  batch; MRL 1536→256
              └──────────────┬───────────────┘
                             ▼
        ┌────────────────────┴────────────────────┐
        ▼                    ▼                    ▼
  ┌───────────┐     ┌──────────────┐     ┌─────────────┐
  │ VECTOR    │     │ BM25 /       │     │ METADATA +  │
  │ SHARDS ×4 │     │ OpenSearch   │     │ ACL STORE   │
  │ 80M chunks│     │ 80M chunks   │     │ (Postgres)  │
  └───────────┘     └──────────────┘     └─────────────┘
```

```
════════════════════ QUERY PIPELINE (synchronous) ════════════════════════

  ┌──────────┐
  │  Client  │
  └────┬─────┘
       ▼
  ┌──────────────────┐
  │  API GATEWAY     │  authN · rate limit · request ID
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │  PRINCIPAL       │  resolve groups from token (cached, 5 min TTL)
  │  RESOLUTION      │  → permission filter object
  └────────┬─────────┘
           ▼
  ┌──────────────────┐        HIT ──────────────────────────┐
  │  SEMANTIC CACHE  │  keyed by (normalised query,         │
  │  (Redis + ANN)   │  permission_hash, corpus_version)    │
  └────────┬─────────┘                                      │
           │ MISS                                           │
           ▼                                                │
  ┌──────────────────┐                                      │
  │  QUERY REWRITE   │  small model: decontextualise from   │
  │  (+ route)       │  history; skip if standalone         │
  └────────┬─────────┘                                      │
           ▼                                                │
     ┌─────┴─────┐  parallel, 120 ms budget                 │
     ▼           ▼                                          │
 ┌────────┐  ┌─────────┐                                    │
 │  BM25  │  │ ANN     │  ← ACL filter INSIDE the search    │
 │ top-50 │  │ top-50  │                                    │
 └───┬────┘  └────┬────┘                                    │
     └─────┬──────┘                                         │
           ▼                                                │
  ┌──────────────────┐                                      │
  │  RRF FUSION      │  Σ 1/(60 + rank + 1)                 │
  │  → top-50        │                                      │
  └────────┬─────────┘                                      │
           ▼                                                │
  ┌──────────────────┐                                      │
  │  RERANK          │  cross-encoder, 50 → 5               │
  │  (GPU)           │  180 ms p95                          │
  └────────┬─────────┘                                      │
           ▼                                                │
  ┌──────────────────┐                                      │
  │  PERMISSION      │  defence in depth: re-verify post-   │
  │  RE-CHECK        │  filter before content enters context│
  └────────┬─────────┘                                      │
           ▼                                                │
  ┌──────────────────┐                                      │
  │  CONTEXT ASSEMBLY│  static prefix first → cache hit;    │
  │  ~3.5k tokens    │  evidence + question at the end      │
  └────────┬─────────┘                                      │
           ▼                                                │
  ┌──────────────────┐                                      │
  │  GENERATE        │  stream tokens; temperature 0.1      │
  │  (LLM)           │  citation markers required           │
  └────────┬─────────┘                                      │
           ▼                                                │
  ┌──────────────────┐                                      │
  │  POST-CHECK      │  citation validity ≥0.995 · PII scan │
  │  + CACHE WRITE   │  · abstention check → cache on pass  │
  └────────┬─────────┘                                      │
           ▼                                                │
  ┌──────────────────┐
  │  OBSERVABILITY   │  trace: retrieval, rerank, tokens,   │
  │                  │  cost, citation verdict              │
  └──────────────────┘
```

### 5.2 Component-by-component justification

| Component | Choice | Why — with numbers |
|---|---|---|
| **Ingest queue** | Kafka, keyed by `doc_id` | Guarantees per-document ordering; replayable after a bad deploy |
| **Parser** | Layout-aware parser + OCR fallback | 10M docs, ~15% scanned PDFs; tables must stay atomic |
| **Chunker** | Recursive 500–800 tok, 12% overlap | 8 chunks/doc; the 12% overlap costs 12% more vectors and prevents boundary loss |
| **Enricher** | Contextual headers, tiered | 5.7% → 1.9% retrieval failure, but at $0.0003/chunk it is applied only to high-value sources |
| **Embedder** | MRL 1536 → 256 indexed | 491 GB → 107 GB, < 2% recall loss — the single biggest memory lever |
| **Vector store** | 4 HNSW shards, binary prefilter | 80M chunks do not fit one node; binary quantisation gives a 32x memory cut |
| **Lexical index** | OpenSearch/BM25 | Handles exact identifiers, error codes, acronyms that dense retrieval misses |
| **Fusion** | RRF, k=60 | No score normalisation needed across two incomparable scales |
| **Reranker** | Cross-encoder, 50 → 5 | +0.10–0.13 NDCG; 180 ms p95 fits the budget |
| **Cache** | Redis + semantic ANN | Keyed by permission hash and corpus version — **never** by query alone |
| **ACL store** | Postgres + Redis hot cache | Must be strongly consistent; 5-min TTL is the max staleness |
| **LLM** | Mid-tier, routed | 50–60% of queries to the small model cuts generation cost ~40% |

### 5.3 The ACL filter — the design detail that matters most

```
   Query: "What is our data retention policy?"

   ┌─────────────────────────────────────────────────────────┐
   │  Principal: u_4412                                      │
   │  groups: [eng, eng-platform, oncall]                    │
   │  permission_hash: 0x8f3a…  ← used in the cache key      │
   └────────────────────────┬────────────────────────────────┘
                            ▼
   ┌─────────────────────────────────────────────────────────┐
   │  VECTOR SEARCH with filter:                             │
   │    allowed_groups ∩ principal.groups ≠ ∅                │
   │    OR allowed_users ∋ principal.user_id                 │
   │    AND classification ≤ principal.clearance             │
   │                                                         │
   │  Implemented as:                                        │
   │   • pre-filter (exact, high-recall) for small groups    │
   │   • partition-by-group for large groups + oversample    │
   └────────────────────────┬────────────────────────────────┘
                            ▼
   ┌─────────────────────────────────────────────────────────┐
   │  POST-FILTER RE-CHECK (defence in depth)                │
   │  Every chunk about to enter the context is re-verified  │
   │  against the live ACL store. Belt and braces.           │
   └─────────────────────────────────────────────────────────┘
```

**Why pre-filter, not post-filter:**

| | Post-filter | Pre-filter |
|---|---|---|
| Recall | Broken — top-50 filtered might leave 3 | Correct — top-50 are all visible |
| Security | Documents the user cannot see were read and scored | Never touched |
| Logs/traces | Leak candidate doc IDs | Safe |
| Cost | Same search, useless results | Filters narrow the search |

**The group-explosion problem, and its fix.** A user in a large group (`everyone@`) can generate a filter with millions of principals. The fix is **partition-by-group**: index large groups as separate partitions and query only the partitions the user belongs to, oversampling 2–3x to compensate for per-partition recall variation.

**The permission-cache key must include the permission hash.** If you cache an answer under the query text alone, user B receives user A's answer — with user A's citations. **This is the failure mode that ends careers, and it is worth saying out loud in the interview.**

### 5.4 Index sharding strategy

| Strategy | Chunks/shard | Vectors/shard (MRL-256, binary) | Notes |
|---|---|---|---|
| By source system | 20M | ~0.8 GB | Simple; uneven growth |
| By document hash | 20M | ~0.8 GB | Even; every query fans out to all 4 |
| **By tenant/org unit** | varies | varies | **Isolation win; uneven** |
| **Chosen: hash-sharded with org partitioning on top** | 20M | ~0.8 GB | Fans out to 4 shards, each filtered by org |

**Query fan-out cost:** 4 shards × ~35 ms each in parallel = ~40 ms wall-clock. Acceptable within the 120 ms retrieval budget. State this — it shows you checked that sharding did not break the latency target.

---

## 6. Step 5 — Deep dive and trade-offs

The interviewer picks one or two areas. Prepare all four; lead with permissions.

### 6.1 Deep dive A — Permission-aware retrieval

**The threat model:**

| Failure | Consequence | Severity |
|---|---|---|
| Answer cites a doc the user cannot open | Information disclosure | **Critical** |
| Answer *contains* content from an inaccessible doc | Breach, possibly regulatory | **Critical** |
| Answer cached under a query key without permission | Cross-user leak | **Critical** |
| Retrieval misses a doc the user *can* see | Quality bug | Medium |
| Permission change takes 10 min to propagate | Access revocation gap | High |

**The design:**

1. **Resolve the principal server-side** from the JWT. Never trust a client-supplied group list.
2. **Extract ACLs at ingest** from the source system and store them as `allowed_groups[]` / `denied_users[]` per chunk, denormalised onto the chunk record so the filter is a single predicate.
3. **Pre-filter inside the ANN search.** Use group partitioning for large groups.
4. **Re-check post-filter** against a live ACL store before content enters the context. This catches the window between ingest and a permission change.
5. **Cache key includes the permission hash**, not just the query.
6. **Purge on revocation.** When a user is removed from a group, invalidate their cached answers — a targeted cache purge keyed by principal.
7. **Audit every retrieval** with the principal, the returned doc IDs, and the filter applied.

**Trade-off to state:** *"Strong permission checks cost me about 40 ms p95 — the pre-filter, plus permission resolution, plus the post-filter re-check. I pay it without hesitation, because the alternative is a data breach. But I refuse to pay it twice: the post-filter re-check only runs on the top-5 chunks that will actually enter the context, not on the top-50, because the top-50 are already pre-filtered and re-checking all 50 costs 3x for no additional safety."*

**The test that proves it works:** a CI job that creates a document visible only to user A, runs a query as user B, and asserts that neither the answer nor the citations nor the trace contain any reference to it. Run it on every build.

### 6.2 Deep dive B — Hybrid retrieval, fusion, and reranking

**Why hybrid is not optional here.** Enterprise corpora are full of things dense retrieval is bad at:

| Query type | Dense | BM25 | Winner |
|---|---|---|---|
| "What's our PTO policy?" | Strong | Weak | Dense |
| "Error code E-4021" | Weak | **Exact match** | BM25 |
| "SOC2 Type II report" | Medium | **Precise** | BM25 |
| "How do I request time off?" | **Semantic match** | Fails on "PTO" | Dense |
| "incident postmortem 2025-11-03" | Weak | **Exact** | BM25 |

**The tuned configuration:**

| Parameter | Value | Reasoning |
|---|---|---|
| BM25 top-k | 50 | Recall-oriented |
| Dense top-k | 50 | Recall-oriented |
| BM25 `k1` / `b` | 1.2 / 0.75 | Standard defaults; `b` tuned to 0.6 for long chunks |
| Fusion | RRF, k=60 | Score-scale-free |
| Weighting (technical corpora) | BM25 bias α≈0.4–0.5 | Identifiers and codes matter more than prose |
| Weighting (general policy) | α≈0.5–0.6 | Balanced |
| **Hybrid vs dense-only** | **+8–12 points recall@10** | The measured reason to do the work |
| Rerank depth | 50 → 5 | **NDCG@10 0.65 → 0.78** |
| Rerank depth K=20/50/100 | NDCG 0.72 / 0.76 / 0.77 | **K=50 is the knee** — K=100 costs 2x latency for +0.01 |
| Final context | 5 chunks, ~3.5k tokens | Evidence budget |

**The RRF formula:**

```
   RRF_score(d) = Σ_{r ∈ retrievers}  1 / (k + rank_r(d) + 1)

   with k = 60 (the standard constant)

   Why k=60:  it flattens the contribution of the top few ranks
              so a single retriever's #1 does not dominate the
              fused list. Lower k → sharper, more top-heavy;
              higher k → flatter, more consensus-driven.
```

**The trade-off to name:** *"Reranking 50 documents costs ~180 ms p95 and about $0.0002 per query — roughly 15% of my total cost for a 13-point NDCG gain. That is the best value in the pipeline. But going from K=50 to K=100 doubles the rerank latency for +0.01 NDCG, which is not worth 180 ms out of a 3-second budget. So I cap at 50 and spend the remaining time budget on generation quality instead."*

**The recall-vs-latency ladder:**

| Configuration | Recall@10 | Retrieval + rerank p95 | Verdict |
|---|---|---|---|
| Dense only, top-20, no rerank | 0.72 | 60 ms | Too lossy |
| Hybrid, top-50, RRF | 0.85 | 120 ms | Good |
| Hybrid, top-50, RRF + rerank | **0.91** | 300 ms | **Chosen** |
| Hybrid, top-200, RRF + rerank | 0.93 | 620 ms | Latency budget blown |
| + multi-query expansion | 0.94 | 900 ms | +2 LLM calls; breaks the p95 target |

### 6.3 Deep dive C — Freshness and index staleness

**The problem:** a document is updated and the assistant keeps citing the old version. Users lose trust faster from a stale answer than from a "no results" answer.

**The design:**

| Change type | Detection | Latency to searchable | Path |
|---|---|---|---|
| New document | Webhook | < 2 min | Normal ingest |
| Edited document | Webhook | < 5 min | Re-chunk changed sections only |
| Deleted document | Webhook + nightly reconcile | < 1 min | Tombstone, then async purge |
| Permission change | Source-system poll (5 min) | **< 5 min** | ACL update only — no re-embedding |
| Bulk import | Queue depth monitor | < 4 hours | Backpressure, lower priority |
| Full recrawl | Scheduled monthly | ~44 hours | Off-peak |

**The tombstone pattern.** Deletion is the hardest case: you cannot "re-embed" a deletion. Write a tombstone first (a soft delete that immediately removes the doc from all retrieval filters), then purge vectors and text asynchronously. **Tombstone latency is under one minute; purge latency can be hours.** This gives deletion correctness without blocking on an expensive index operation.

**Index versioning.** Every ingest writes to the live index and bumps a `corpus_version`. Queries record the version they ran against. This gives you:
- Reproducible debugging ("this answer came from corpus v4412").
- A safe rollback path if a bad bulk import corrupts the index.
- A cache-invalidation trigger — a corpus version bump invalidates cached answers for affected document IDs.

**The freshness SLO and its alert:**

```yaml
- name: index_freshness_slo
  metric: max(now - document.updated_at) for documents pending ingest
  warning:  > 600s   # 10 min
  critical: > 900s   # 15 min — SLO breach
  action: page_ingest_oncall
```

**Trade-off to state:** *"I chose 15 minutes rather than near-real-time because near-real-time would mean synchronous embedding on every wiki save, which is expensive and unnecessary — a wiki edit is not a payment. But I hold deletion and permission changes to a much tighter bound, under 5 minutes, because those are correctness and security properties rather than freshness conveniences."*

### 6.4 Deep dive D — Cost control

**The cost stack and its levers:**

| Lever | Mechanism | Saving | Risk |
|---|---|---|---|
| **Prompt caching** | Static prefix first (system + tools + policy) | **~35% of generation** | Must be byte-stable |
| **Semantic cache** | Redis ANN, keyed by query + permission + corpus version | **~25% of total** | Wrong-answer risk; needs a similarity threshold (0.93) and a freshness check |
| **Model routing** | Classify difficulty; 50–60% to the small model | ~40% of generation | Quality regression on hard queries |
| **Context discipline** | 5 chunks, 3.5k tokens, not 20 chunks | ~30% of generation | Recall risk |
| **Batch ingestion** | Batch API for embeddings and enrichment | ~50% on ingestion | Latency acceptable for async |
| **Rerank at K=50** | Not K=100 | ~50% of rerank | +0.01 NDCG forgone |

**The semantic cache design in detail — because the naive version is dangerous:**

```
   cache_key = hash(
       normalised_query,          ← lowercase, stopwords stripped
       permission_hash,           ← CRITICAL — never omit
       corpus_version,            ← invalidates on ingest
       answer_mode
   )

   lookup:
     1. exact match on the hash         → 15 ms, highest confidence
     2. ANN over cached embeddings      → similarity ≥ 0.93
     3. verify the cached answer's cited
        documents still exist AND the
        user still has permission        → else MISS
     4. verify corpus_version unchanged  → else MISS
```

**The threshold trade-off:** 0.93 is conservative. Lowering to 0.90 raises the hit rate from ~25% to ~32% but starts returning answers to subtly different questions — which is worse than a cache miss, because the user cannot tell it happened. **Alert on cache-hit answer quality separately from uncached quality**; if the cached path scores lower on the golden set, the threshold is too low.

**The routing rule:**

```
   Classify the query (small model, ~50 ms):
     • "lookup"     — single fact, clear entity      → small model
     • "synthesis"  — multi-document reasoning        → mid model
     • "analytical" — comparison, calculation, chains → frontier model

   Measured distribution: 55% lookup · 35% synthesis · 10% analytical
   Blended generation cost: ~40% below all-frontier
```

**And the guard:** route decisions are logged with the query class, and the golden set is scored **per class**. If the lookup class scores below the gate on the small model, the router is demoted for that class.

### 6.5 Trade-off summary table

Present this unprompted. It is the single highest-signal artefact in the whole answer.

| Decision | Chosen | Alternative | Why | What I gave up |
|---|---|---|---|---|
| Vector index | HNSW + MRL-256 + binary prefilter | HNSW fp32 | 491 GB → ~3 GB prefilter, <5% recall loss recovered by rerank | ~3 points of raw recall |
| Retrieval | Hybrid BM25 + dense + RRF | Dense only | Exact identifiers and error codes | ~40 ms and an extra system |
| Rerank depth | K=50 → 5 | K=100 → 5 | 180 ms vs 400 ms for +0.01 NDCG | 0.01 NDCG |
| Permissions | Pre-filter + post-filter re-check | Post-filter only | Post-filter is a breach | ~40 ms p95 |
| Freshness | 15 min content, 5 min permissions | Near-real-time | Cost; a wiki edit is not a payment | Freshness on content |
| Deletion | Tombstone then async purge | Synchronous purge | Correctness without blocking | Eventual storage reclamation |
| Consistency | Eventual (content), strong (permissions) | Strong everywhere | Cost and latency | 15 min staleness window |
| Availability | 99.9% | 99.99% | Internal tool; cost not justified | 4x the downtime budget |
| Cache | Semantic, threshold 0.93 | Exact-match only | 25% hit rate | Wrong-answer risk requiring monitoring |
| Generation model | Routed (55% small) | All frontier | ~40% of generation cost | Router complexity + per-class monitoring |
| Abstention | First-class, confidence-gated | Always answer | Trust; a wrong confident answer is worse than "I don't know" | ~8% abstention rate |

---

## 7. Failure modes and how this design handles them

| Failure | Symptom | Mitigation in this design |
|---|---|---|
| **Retrieval miss** | "I don't have information on that" for a doc that exists | Hybrid retrieval + contextual enrichment; alert when `context_recall < 0.80` |
| **Hallucination** | Confident claim with no supporting citation | Citation validity check ≥ 0.995; abstention when confidence is low |
| **Context poisoning** | Retrieved doc contains injected instructions | Retrieved content is untrusted; delimit and label; never let it change tool sets or permissions |
| **Cross-user leak** | User B sees user A's content | Pre-filter + permission-hash cache key + post-filter re-check + CI probe |
| **Stale citation** | Answer cites a superseded policy | Freshness SLO 15 min; `updated_at` surfaced in the citation; tombstone on delete |
| **Index corruption** | Bad bulk import degrades recall | `corpus_version` + atomic index swap + rollback path |
| **Embedding drift** | Recall degrades after an embedding model change | Pin the model; re-embed only on a versioned migration with a shadow index |
| **Cost explosion** | Daily spend spikes 3x | Alert on hourly cost > 2x trailing mean; per-tenant budget caps; degradation ladder |
| **Latency cliff** | p95 crosses 3 s | Per-stage budgets with timeouts; drop reranking and then retrieval breadth under load |
| **Abstention failure** | Answers questions it has no evidence for | Confidence gate; abstention is a designed, measured behaviour |
| **Permission propagation lag** | Revoked user still sees content | 5-min ACL poll + post-filter re-check against the live store |
| **Silent quality degradation** | All green, answers worse | Golden-set eval on every deploy; judge-drift alert separate from system-drift alert |
| **Poisoned memory** | N/A here — read-only, no memory writes | Not applicable, and worth stating: no write path means no memory poisoning |

---

## 8. Evaluation and operations

### 8.1 Release gates

| Gate | Threshold | Action |
|---|---|---|
| Recall@10 | ≥ 0.85 | Block |
| Faithfulness | ≥ 0.80 | Block |
| Citation validity | ≥ 0.995 | Block |
| Context precision | ≥ 0.70 | Warn |
| Abstention correctness | ≥ 0.90 | Warn |
| p95 latency | ≤ 3.0 s | Block |
| Permission probe | 0 leaks | **Block, always** |
| Cost per query | ≤ $0.012 | Warn |

### 8.2 Alerts

```yaml
- name: retrieve_p99_latency
  warning: > 500ms   critical: > 2000ms
- name: rerank_p95_latency
  warning: > 300ms   critical: > 600ms
- name: ann_bench_recall
  warning: < 0.90    critical: < 0.85
- name: semantic_cache_hit_rate
  warning: < 0.15    note: falling hit rate may mean query drift
- name: hourly_cost
  warning: > 2× trailing 24h mean   critical: > 4×
- name: abstention_rate
  warning: > 15%     note: retrieval or confidence-gate problem
- name: citation_validity
  warning: < 0.995   critical: < 0.98
- name: index_freshness
  warning: > 600s    critical: > 900s
- name: permission_probe_failures
  critical: > 0      action: page_security
- name: judge_frozen_items_drift
  critical: > 0      note: distinguishes judge drift from system drift
```

### 8.3 Degradation ladder

Under load or provider failure, shed features in this order — never fail outright:

```
   1. Disable semantic cache writes (reads still serve)      ~0 ms saved
   2. Drop multi-turn query rewriting                          ~250 ms
   3. Reduce rerank depth 50 → 20                              ~90 ms
   4. Reduce retrieval k 50 → 25                               ~30 ms
   5. Route everything to the small model                     ~500 ms
   6. Serve retrieve-only with extractive snippets            ~1,500 ms
   7. Queue and show "high demand, try again" — last resort
```

Steps 1–4 are invisible to users. Step 6 is visible but still useful: showing the five most relevant passages with their titles is a legitimate degraded product.

---

## 9. Interview Q&A on this case

**Q1. Why hybrid retrieval instead of dense-only?**
Because enterprise corpora are full of exact-match queries that dense retrieval is weak on: error codes like `E-4021`, document identifiers, acronyms, dates, and version numbers. Dense retrieval embeds those into a semantic neighbourhood where `E-4021` looks similar to `E-4022`. BM25 catches them exactly. In practice hybrid adds 8–12 points of recall@10 over dense-only on enterprise data, and it costs about 40 ms because BM25 is fast and runs in parallel with the ANN search. RRF with k=60 fuses them without needing score normalisation, which matters because BM25 scores and cosine similarities are on incomparable scales.

**Q2. How do you handle permissions at 80 million chunks?**
Four layers. First, resolve the principal server-side from the JWT — never accept a group list from the client. Second, denormalise the ACL onto each chunk at ingest so the filter is a single predicate inside the ANN search, as a pre-filter, not a post-filter — post-filtering means I searched over documents the user cannot see and returned fewer than top-k, and it can leak document IDs into traces. Third, partition by group for large groups, because a user in `everyone@` generates a filter with millions of principals; querying only the relevant partitions with 2–3x oversampling solves it. Fourth, a post-filter re-check against the live ACL store for the five chunks that will actually enter the context — that catches the window between ingest and a permission revocation. Total cost about 40 ms p95.

**Q3. What's the worst failure mode in this system?**
A cross-user memory or cache leak. If I cache an answer under the query text alone, user B gets user A's answer with user A's citations — a breach, not a bug. So the cache key includes a permission hash and a corpus version, and I run a CI probe on every build: create a document visible only to user A, query as user B, assert that the answer, the citations, and the trace contain no reference to it. That test never gets skipped.

**Q4. Walk me through your capacity estimation.**
100k DAU at 8 queries each is 800k requests per day. Divided by 86,400 that's 9.3 QPS average; at 3x peak it's 28 QPS. That's small for a web tier — the bottleneck is not throughput. On the index side: 10M documents at roughly 5,000 tokens each, chunked at 500–800 tokens with 12% overlap gives about 8 chunks per document, so 80 million chunks. At 1536 dimensions and 4 bytes per float that's about 491 GB of raw vectors and roughly 640 GB with HNSW overhead. That number is what drives the design — it does not fit one node, which forces sharding, and it pushes me to Matryoshka truncation to 256 dimensions, which is about 107 GB with under 2% recall loss, plus binary quantisation for the candidate scan.

**Q5. What's your latency budget breakdown?**
p95 target is 3 seconds. Auth and permission resolution 60 ms, cache lookup 30 ms, query rewrite 250 ms, embedding 50 ms, hybrid retrieval 120 ms, RRF 10 ms, rerank 180 ms, context assembly 40 ms, and generation — first token at 900 ms, full answer at 2,700 ms. So generation is about 60% of the budget. That's the key constraint: everything upstream has to be fast because there's almost no headroom, which is why I refuse multi-query expansion at 900 ms and why I cap rerank at K=50 rather than K=100.

**Q6. How do you keep the index fresh?**
15 minutes for content, 5 minutes for permissions — deliberately different. Content freshness at 15 minutes matches the wiki publish cadence and lets me batch embedding and enrichment asynchronously, which is much cheaper than synchronous embedding on every save. Permissions are different because they're a security property, so I poll source ACLs every 5 minutes and I also re-check the live ACL store before content enters the context. Deletion is the hard case: I can't re-embed a deletion, so I write a tombstone that immediately removes the document from all retrieval filters — under a minute — and purge vectors and text asynchronously, which can take hours. Deletion correctness without blocking on an expensive index operation.

**Q7. How do you control cost?**
Three levers, in order. Prompt caching on the static prefix — system prompt, tool schemas, policy text — which is worth about 35% of generation cost, and requires keeping the prefix byte-identical with static content first. A semantic cache at a 0.93 similarity threshold with a 25% hit rate, keyed by query plus permission hash plus corpus version. And model routing, where a classifier sends about 55% of queries to a small model, cutting generation cost roughly 40%. Together those take me from $0.008 per query to about $0.003. Generation is 80% of the cost, which is why every lever targets it.

**Q8. Why did you choose 99.9% availability and not 99.99%?**
Because it's an internal tool and the four-fold increase in downtime budget — 43 minutes a month versus 4.4 — is not worth the cost of the redundancy it would require. 99.99% means multi-region active-active with synchronized vector indexes, which is a very large engineering investment. Compare that to WhatsApp, where you cannot lose a message, so reliability and availability dominate. Here the priority ordering is reliability of ingestion, cost efficiency of answers, and performance, with availability third.

**Q9. What would you change if the corpus were 100x larger — 1 billion chunks?**
Three things. First, I'd push harder on quantisation — at a billion chunks, binary quantisation plus fp32 rescoring of a few hundred candidates becomes mandatory, and I'd probably move to a disk-based index like DiskANN, which has reported 90–95% RAM reduction with acceptable recall. Second, I'd stop hash-sharding and partition by domain, because fan-out to 40 shards breaks the latency budget even in parallel. Third, I'd reconsider whether all billion chunks need to be in one searchable index at all — a router that selects a subset of corpora before retrieval is cheaper and often more accurate than searching everything.

**Q10. How do you know the answer is grounded?**
Three mechanisms. The generation prompt requires inline citation markers and I validate that every marker resolves to a retrieved chunk — citation validity, gated at 0.995. A separate faithfulness check on the golden set, gated at 0.80. And an abstention path: if the top reranked score is below the similarity floor, or if the citation check fails, the system says it does not have enough information rather than answering. The abstention rate is monitored — above 15% means retrieval is failing, below 3% means the confidence gate is too permissive.

**Q11 (pushback). Isn't this over-engineered for 28 QPS peak?**
The throughput is easy; the latency and index size are not. 28 QPS is trivial for the web tier, and I said so. But 80 million chunks is 491 GB of raw vectors, which does not fit one node at any QPS, so sharding and quantisation are forced by the index size, not the traffic. And the 3-second p95 with generation consuming 60% of the budget means every upstream stage has to earn its milliseconds. So the complexity here is buying latency headroom and memory efficiency, not throughput.

**Q12 (pushback). Wouldn't fine-tuning solve the quality problems more simply than all this retrieval engineering?**
No, and this is the case where that distinction is clearest. Fine-tuning is a behaviour store — it can make formatting and tone consistent, but it cannot tell me what page 47 of the 2026 retention policy says, and it cannot cite it. The requirements here are freshness under 15 minutes, mandatory citations, and permission-filtered retrieval. All three are RAG properties. A fine-tuned model would need retraining every time a policy changes, which is daily, and it could not tell the user which document it was drawing from. I might fine-tune a small model for the *routing* and *query rewriting* behaviours, which is a genuine behaviour task — but the knowledge stays in the index.

**Q13. What's the first thing you'd build?**
The evaluation set and the permission probe. Before any retrieval tuning, I need a golden set of 300–500 real questions with known-correct source documents so I can measure whether a change helped, and I need the cross-user leak probe because that failure is unrecoverable. Everything else — hybrid retrieval, reranking, quantisation — is an optimisation on top of a system that works. Building retrieval tuning before you can measure it is how teams spend six months and ship something worse.

**Q14. If you had to cut the design in half, what stays?**
Hybrid retrieval with reranking, permission-aware pre-filtering, citations, and abstention. Those four are the product. What goes: the semantic cache (saves 25% of cost but adds a leak surface and a wrong-answer mode), model routing (saves 40% of generation but needs per-class monitoring), contextual enrichment at ingest (the biggest ingestion cost), and multi-turn rewriting (nice, not essential). The system would be slower and more expensive but still correct and safe — and correctness and safety are the parts that must never be the thing you cut.

**Q15. What single number would you monitor above all others?**
Permission probe failures — which must be zero. Everything else is degradable: latency can rise, cost can rise, recall can dip, and the product still works, just worse. A single cross-user leak is a breach with legal and regulatory consequences, and it destroys user trust in a way that no quality improvement recovers from.

---

## 10. What a strong answer looks like — checklist

| Phase | Signal | Weak answer | Strong answer |
|---|---|---|---|
| Clarify | Scopes before drawing | Immediately draws boxes | Asks functional + non-functional, states "in and out of scope" |
| Clarify | Mentions the goal conflicts | Ignores trade-offs | Names which of the six goals wins and why |
| Estimate | Uses the template | "Some thousands of QPS" | 9.3 QPS avg, 28 peak, 80M chunks, 491 GB |
| Estimate | Derives an architectural consequence | Numbers are decoration | "491 GB does not fit one node → sharding + MRL + BQ" |
| API | Defines the contract | Skips to architecture | POST /v1/query with citations, abstention, usage |
| API | Permission contract | Omits | Principal resolved server-side, never client-supplied |
| HLD | Two pipelines | One blob | Ingestion (async) and query (sync) drawn separately |
| HLD | Names the ACL filter placement | Says "filter results" | Pre-filter inside search + post-filter re-check |
| Deep dive | Quantifies the trade | "Reranking improves quality" | NDCG 0.65 → 0.78 for 180 ms and $0.0002 |
| Deep dive | Surfaces the security failure | Never mentions leakage | Cache key includes permission hash; CI probe |
| Deep dive | Freshness with a rationale | "It's real-time" | 15 min content, 5 min permissions, tombstone on delete |
| Close | Trade-off table | Trails off | Explicit chosen-vs-alternative-vs-cost table |

---

## 11. Cheat-sheet summary

| Dimension | Value |
|---|---|
| **Corpus** | 10M documents · ~5,000 tokens each · ~15% scanned |
| **Chunks** | 80M at 500–800 tokens, 12% overlap, ~8/doc |
| **DAU / queries** | 100k DAU × 8 = 800k requests/day |
| **QPS** | 9.3 average · **28 peak** |
| **Latency** | p50 1.5 s · **p95 < 3 s** |
| **Availability** | 99.9% (43 min/month) |
| **Consistency** | Eventual for content (15 min) · **strong for permissions (5 min)** |
| **Raw vectors (1536d fp32)** | 491 GB |
| **HNSW fp32** | ~640 GB |
| **MRL-256 + HNSW** | ~107 GB (< 2% recall loss) |
| **Binary quantisation prefilter** | ~3.3 GB · 32x reduction |
| **Total index** | ~940 GB → ~1.5 TB after year 1 |
| **Vector index** | HNSW M=32, ef_construction=256, ef_search=64–128 |
| **Retrieval** | BM25 top-50 ∥ dense top-50 · **RRF k=60** |
| **Rerank** | Cross-encoder 50 → 5 · NDCG 0.65 → **0.78** · 180 ms |
| **Context** | 5 chunks · ~3.5k tokens |
| **Generation** | temperature 0.1 · routed 55% small / 35% mid / 10% frontier |
| **Abstention** | First-class field · target rate 5–10% |
| **Citation validity gate** | ≥ 0.995 |
| **Faithfulness gate** | ≥ 0.80 |
| **Recall@10 gate** | ≥ 0.85 |
| **Cache** | Semantic, threshold **0.93**, key = query + permission hash + corpus version |
| **Cache hit rate** | ~25% |
| **Cost per query** | $0.008 raw → **~$0.006 after cache** |
| **Cost split** | Generation 80% · rerank 15% · vector 2% · embed 1% |
| **Daily cost** | ~$5,550 (queries $4,800 + ingestion $750) |
| **Prompt caching saving** | ~35% of generation |
| **Routing saving** | ~40% of generation |
| **Ingestion rate** | 300k docs/day (3%) → 2.4M chunks/day → 28/s avg |
| **Full recrawl** | ~44 hours at 500 chunks/s |
| **Freshness SLO** | Content < 15 min · permissions < 5 min · deletion tombstone < 1 min |
| **Permission controls** | Server-side principal · pre-filter · group partitioning · post-filter re-check · permission-hash cache key · audit log |
| **The one non-negotiable test** | Cross-user leak probe, every build, must return zero |
| **Degradation ladder** | Cache writes → rewrite → rerank depth → k → small model → extractive → queue |

---

## 12. Cross-links

- [Production RAG System](../../HLD/05-ai-systems/01-production-rag-system.md) — the reference architecture this case instantiates
- [Advanced RAG Patterns](../../HLD/05-ai-systems/02-advanced-rag-patterns.md) — contextual retrieval, HyDE, routing, and when each earns its cost
- [RAG Evaluation and Observability](../../HLD/05-ai-systems/03-rag-evaluation-and-observability.md) — the golden set, the triad, and the gates used here
- [Guardrails and Safety for AI Systems](../../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md) — citation checks, PII scanning, and untrusted retrieved content
- [LLM Serving, Inference and Gateway](../../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md) — routing, caching, and the generation cost model
- [Memory and Context Engineering](../../HLD/05-ai-systems/08-memory-and-context-engineering.md) — conversation state, prefix caching, and context assembly
- [Fine-Tuning vs RAG vs Prompting](../../HLD/05-ai-systems/09-fine-tuning-vs-rag-vs-prompting.md) — why the knowledge stays in the index
- [AI Customer Support Agent](10-ai-customer-support-agent.md) — the same framework applied to an agentic system
- [Agentic Coding Assistant](11-agentic-coding-assistant.md) — the same framework with tools and sandboxes
- [Capacity Planning and Autoscaling](../../Operations/05-capacity-planning-and-autoscaling.md)
- [Cost and FinOps](../../Operations/06-cost-and-finops.md)
- [Security, Guardrails and Compliance](../../Operations/08-security-guardrails-and-compliance.md)
- [LLMOps for AI Systems](../../Operations/10-llmops-for-ai-systems.md)
- [RAG and LLM Cheatsheet](../../Cheat-Sheets/04-rag-and-llm-cheatsheet.md)
