# RAG and LLM Cheatsheet

**How to use this** — One page per subsystem, each designed to be readable in 90 seconds before an interview or during an incident. Every number here is a defensible starting point, not a law; the reasoning behind it is in the linked chapter. If you remember only three things: **hybrid retrieval plus a reranker is the highest-return quality change**, **`retrieval failure` and `generation failure` have different fixes and you must separate them**, and **chunk size is tuned against your corpus, never accepted as a default**.

---

## 1. The RAG pipeline at a glance

```
INGESTION (offline, or on change)
  source docs
      │
      ▼
  ┌─────────────┐   parse PDFs/HTML/Office, OCR scans, extract tables
  │  PARSE      │   → quality gate: parse error rate per source type
  └──────┬──────┘
         ▼
  ┌─────────────┐   structure-aware split on headings/sections
  │  CHUNK      │   fallback: fixed 500–800 tokens, 10–15% overlap
  └──────┬──────┘
         ▼
  ┌─────────────┐   + context sentence per chunk (contextual retrieval)
  │  ENRICH     │   + ACL and tenant_id baked into the chunk
  └──────┬──────┘
         ▼
  ┌─────────────┐   pinned embedding model version, normalised vectors
  │  EMBED      │   → NEVER mix two embedding models in one collection
  └──────┬──────┘
         ▼
  ┌─────────────┐   vector index (HNSW)  +  lexical index (BM25)
  │  INDEX      │   + metadata payload incl. chunk text + ACL
  └─────────────┘

QUERY (online, per request)
  user query
      │
      ▼
  ┌─────────────┐   ~50 ms, resolve pronouns from history, expand acronyms
  │  REWRITE    │   log original + rewritten side by side
  └──────┬──────┘
         ▼
  ┌─────────────┐   ~30 ms
  │  EMBED      │
  └──────┬──────┘
         ▼
  ┌───────────────────────────┐   ~100 ms
  │  HYBRID RETRIEVE          │   dense top-50  ‖  BM25 top-50
  │  WHERE tenant_id = $1     │   ← ACL enforced HERE, not in the prompt
  │    AND acl_group = ANY($2)│
  └──────┬────────────────────┘
         ▼
  ┌─────────────┐   ~10 ms   score = Σ 1/(60 + rank)
  │  FUSE (RRF) │
  └──────┬──────┘
         ▼
  ┌─────────────┐   ~250 ms  cross-encoder, 50 → 5
  │  RERANK     │   ← the single highest-return quality change
  └──────┬──────┘
         ▼
  ┌─────────────┐   below threshold → ABSTAIN, don't answer
  │  THRESHOLD  │   target abstention band 8–15% where corpus has gaps
  └──────┬──────┘
         ▼
  ┌─────────────┐   ~2000 ms
  │  GENERATE   │   instruction: answer ONLY from context; cite sources
  └──────┬──────┘
         ▼
  ┌─────────────┐   ~400 ms
  │  GUARDRAILS │   input ~160 ms + output ~330 ms
  └──────┬──────┘
         ▼
     response + citations
```

**End-to-end budget (typical interactive RAG):**

| Stage | p50 | Notes |
|---|---|---|
| Rewrite | 50 ms | optional; skip for short queries |
| Embed | 30 ms | cache embeddings of frequent queries |
| Hybrid retrieve | 100 ms | dense and sparse in parallel |
| RRF fuse | 10 ms | negligible |
| Rerank | 250 ms | cap candidates at 50, not 200 |
| Generate | 2,000 ms | dominates; stream it |
| Guardrails | 400 ms | keep in-path stack < 500 ms |
| **Total** | **~2.8 s** | TTFT is what the user perceives — stream early |

---

## 2. Chunking decision table

| Corpus shape | Chunk size | Overlap | Method | Why |
|---|---|---|---|---|
| Technical docs / API reference | 256–512 tokens | 15% | structure-aware on headings | dense, self-contained concepts |
| Narrative / policy documents | 512–800 tokens | 10–12% | structural, fallback fixed | arguments span paragraphs |
| Legal contracts | 400–600 tokens | 20% | clause-level, keep clause numbers | citation precision matters |
| Code | function/class boundaries | — | AST-aware | never split mid-function |
| Tables | one table per chunk | — | extract to markdown + NL summary | flattening loses row/column meaning |
| Chat / tickets | one thread per chunk | — | conversation-turn bounded | context is the thread |
| Unknown / mixed | 512 tokens | 12% | fixed window, then tune | defensible default |

**Tuning procedure** — never accept a default:

1. Build 100–200 labelled (query → relevant chunk) pairs from real traffic
2. Sweep 256 / 512 / 768 / 1024 with everything else held constant
3. Compare `recall@10` and `nDCG@10`
4. Re-check after adding hybrid retrieval and a reranker — the optimum moves

**Contextual retrieval** — prepend an LLM-generated context sentence per chunk before embedding. Published effect: retrieval failure **5.7% → 1.9%** combined with hybrid + rerank. Cost: one LLM call per chunk at ingest (a few thousand dollars and several hours for 10M chunks). One-time per corpus version, not per query.

---

## 3. Embedding models and vector space

| Model class | Dims | When to use | Notes |
|---|---|---|---|
| Small open (e.g. 384-dim) | 384 | cost-sensitive, high volume, simple domain | weakest on paraphrase |
| Standard open (e.g. 768–1024) | 768–1024 | general default | best cost/quality balance |
| Large hosted (e.g. 1536) | 1536 | accuracy-critical retrieval | the common production default |
| MRL-truncatable | 1536 → 256 | memory-bound at scale | one model, two index sizes |
| Domain fine-tuned bi-encoder | inherits base | vocabulary mismatch is the bottleneck | needs a full re-index |

**Matryoshka Representation Learning (MRL)** lets you truncate a 1536-dim vector to 256 dims with modest recall loss — a 6x memory reduction per vector. Useful when memory is the binding constraint and re-training an embedder isn't on the table.

**Binary quantisation** — 1 bit per dimension instead of 32, giving roughly **32x** memory reduction. Recall loss is real but often acceptable as a first-stage filter before reranking.

**The two absolute rules:**

- Changing the embedding model **requires a full re-index**. Vectors from two models are not comparable; similarity scores become meaningless.
- The **embedding model version is part of the index identity** — record it in the index snapshot metadata and in every trace.

**Storage arithmetic** (the number to know cold):

```
10M chunks × 1536 dims × 4 bytes           = 61.4 GB   raw vectors
× 1.5 (HNSW graph overhead)                = 92.2 GB
+ metadata + chunk text (~500 B/chunk)     =  ~5.0 GB
                                    TOTAL  ≈ 97 GB per replica
```

---

## 4. Vector index comparison

| Index | Memory | Build | Query | Recall | Use when |
|---|---|---|---|---|---|
| **Flat** (brute force) | 1x | instant | slow at scale | 100% | < 1M vectors, or as ground truth |
| **HNSW** | ~1.5x | slow | fastest | excellent | up to ~10–50M vectors, latency-sensitive |
| **IVF-Flat** | ~1x | fast | fast | good, tunable | very large corpora, memory matters |
| **IVF-PQ** | ~0.1x | fast | fast | moderate | 100M+ vectors, memory-bound |
| **Binary quantised** | ~0.03x | fast | fastest | reduced | first-stage filter before rerank |

**HNSW parameters to know:**

| Parameter | Default | Effect of raising |
|---|---|---|
| `M` | 32 | more connections → better recall, more memory, slower build |
| `ef_construction` | 256 | better graph quality, slower build, no query cost |
| `ef_search` | 64–128 | better recall, linear latency increase at query time |

`ef_search` is the one you tune in production — it's the only knob with no build cost. Going from the default to 128 can lift recall@10 from ~0.71 to ~0.89 at roughly +40 ms.

**Index maintenance:**

| Job | Cadence | Threshold / note |
|---|---|---|
| Freshness reconciliation | daily | index vs source: orphaned vectors, missing docs |
| Compaction | weekly or on trigger | above **20% fragmentation** |
| Stale re-embedding | on embedder change | full rebuild, never partial |
| Delete propagation | continuous | orphans are the classic bug — deleted docs still cited |
| Read replica lag check | continuous | alert on lag > freshness SLO |

**Multi-tenancy models:**

| Model | Isolation | Cost | Use for |
|---|---|---|---|
| **Silo** — index per tenant | strongest | highest | enterprise / regulated |
| **Pool** — shared + `tenant_id` filter | weakest | lowest | SMB SaaS at scale |
| **Bridge** — silo top N, pool rest | mixed | moderate | typical production answer (e.g. 20 siloed, 480 pooled) |

The tenant predicate goes **in the retrieval query**, never in the system prompt. A CI test must seed tenant B's documents and assert tenant A never retrieves them. Cross-tenant retrieval is a **paging alert, not a warning**.

---

## 5. Retrieval fusion and reranking

**Reciprocal Rank Fusion:**

```
score(d) = Σ over lists L containing d of  1 / (k + rank_L(d))

k = 60   (smoothing; TREC-derived default)
```

Why it works: it uses **rank only**, so it never needs a cosine similarity (0.82) and a BM25 score (14.3) to be on the same scale. A document appearing in *both* lists gets both contributions — that consensus is the signal.

Tuning `k`: smaller `k` makes rank 1 dominate (aggressive, brittle); larger `k` flattens toward uniform (loses signal). Raise `k` if one retriever is dominating and you want the other to have more influence.

**Typical hybrid gain:** 8–15 points of `recall@10` over dense-only on mixed query workloads.

**Two-stage retrieval is a compute trade-off:**

| Stage | Model | Cost | Accuracy | Runs over |
|---|---|---|---|---|
| Retrieve | bi-encoder (independent embeddings) | cheap | approximate | millions of docs |
| Rerank | cross-encoder (query + doc together) | expensive | much higher | 50–100 candidates |

**Reranker effect:** `nDCG@10` **0.65 → 0.78**, ~+250 ms, a fraction of a cent. If you have a quality problem and no reranker, add one before touching anything else.

**Candidate set sizing:** 50 is the sweet spot. At 200 you pay 4x the rerank latency for a small recall gain and a larger precision risk.

**Late interaction (ColBERT)** — one vector *per token*, MaxSim scoring:

| | Bi-encoder | ColBERT | Cross-encoder |
|---|---|---|---|
| Vectors per doc | 1 | 1 per token | n/a |
| Storage | 1x | 10–100x | n/a |
| Latency @ 10M docs | ~10 ms | ~50–100 ms (PLAID) | infeasible |
| Accuracy | good | near cross-encoder | best |

Use ColBERT at roughly 1M–50M documents where accuracy is paramount and storage is affordable.

---

## 6. Advanced RAG patterns

| Pattern | What it does | Use when | Cost |
|---|---|---|---|
| **Contextual retrieval** | context sentence prepended per chunk | orphan chunks hurt retrieval | 1 LLM call/chunk at ingest |
| **Multi-query** | 3–5 paraphrases, retrieve each, fuse | short/ambiguous queries, vocabulary mismatch | 3x retrieval |
| **Query decomposition** | split into sub-questions, retrieve each | genuine multi-part questions | 1 LLM call + N retrievals |
| **RAPTOR** | recursive clustering + summarisation tree | global/"what are the themes" questions | ~$2,000 + 6–10 h for 10M chunks |
| **GraphRAG** | entity/relation graph, traverse | multi-hop and relational questions | LLM extraction per chunk + rebuild |
| **Agentic RAG** | model controls retrieve/reformulate/loop | exploratory, multi-hop | 3–10 round trips |
| **Semantic caching** | embed query, return prior answer above threshold | high-repetition traffic | 20–40% hit rate typical |
| **HyDE** | generate hypothetical answer, embed that | query and doc phrasing differ sharply | 1 LLM call + 1 embed |

**Conditional routing beats blanket application.** Multi-query is worth it only for short queries, low initial top-score, or known vocabulary mismatch. Agentic RAG should be gated behind a classifier that sends simple queries down the fixed path.

**Semantic cache — the dimensions that must be in the key:**

```
tenant_id + user_acl + locale + plan_tier + release_id + prompt_version
```

Missing `release_id` is the classic bug: the cache serves answers from the previous prompt for days. Target a **conservative threshold** — a 20% hit rate with a 0.2% false-hit rate beats a 40% hit rate with 3%. False hits are invisible by construction, so **sample cache hits and re-run them fresh** to measure the rate.

---

## 7. RAG failure modes — symptom → cause → fix

| Symptom | Likely cause | Fix | Signal to watch |
|---|---|---|---|
| Confident wrong answer, source exists in corpus | retrieval miss | hybrid + reranker; check chunk coverage | `recall@k` |
| Confident wrong answer, source NOT in corpus | no threshold → forced answer | score threshold + abstention prompt | abstention rate below 8% |
| Answer ignores the retrieved context | generation problem | prompt: answer ONLY from context; verify faithfulness | faithfulness < 0.85 |
| Answer blends two contradictory sources | contradiction not handled in prompt | prompt: state both and note disagreement | contradiction cases in eval set |
| Good answers, then a cliff after a deploy | index or prompt change | check `release_id` diff, roll back | score by release |
| Retrieval works, quality drifts slowly | embedding drift or corpus drift | re-embed, corpus reconciliation | score distribution shift |
| Stale answers | index staleness / orphaned vectors | freshness SLO + daily reconciliation | freshness lag |
| Cross-tenant content appears | ACL filtered at query time, not ingest | ACL in chunk at write; tenant predicate in query | cross-tenant retrieval count = 0 |
| Cost spikes with flat traffic | cache-hit collapse | check cache key stability, TTL, routing | cache hit rate vs baseline |
| Quality score moves, nothing changed | **judge drifted** | recalibrate judge against human labels | Cohen's kappa |
| Long-context migration goes wrong | cache invalidation + attention dilution | hybrid: long context for small static corpora | cost per request, p95 |

**The diagnostic order, always:**
`was the answer in the context?` → if no, retrieval problem → `was precision bad?` → if no, generation problem.
Labelling 200 failures this way takes a day and saves weeks of tuning the wrong stage.

---

## 8. Evaluation metrics

**Retrieval:**

| Metric | Measures | Target |
|---|---|---|
| `recall@k` | did we get the relevant docs at all | ceiling on everything downstream |
| `precision@k` | how much retrieved context is relevant | drives context pollution |
| `MRR` | is the best doc near the top | — |
| `nDCG@k` | graded relevance with position | **0.65 → 0.78** with reranker |

**Generation:**

| Metric | Definition | Target |
|---|---|---|
| Faithfulness | every claim supported by the context | ≥ 0.85 |
| Answer relevance | does it address the question | — |
| Completeness | did it cover the necessary parts | — |
| Hallucination rate | claims not supported by any source | ≤ 0.03 |
| Schema-valid rate | structured output parses | ≥ 0.995 |
| Abstention correctness | did it abstain when it should | band 8–15% |

**Judge design:**

| Bias | Symptom | Fix |
|---|---|---|
| Position | prefers whichever came first | swap and re-run; inconsistent → tie |
| Length | longer looks better | instruct to ignore length |
| Self-preference | favours its own model family | judge with a **different** family than the model under test |
| Format | markup influences the score | diverse calibration examples |

**Judge calibration:** Cohen's kappa against human labels — **target κ ≥ 0.6, alert < 0.4**. An uncalibrated judge is worse than no judge: it produces confident wrong numbers.

**Eval set construction:**

- 200–500 cases; stratified by query type (factual, multi-hop, no-answer, ambiguous, adversarial)
- 10–20% double-annotated for agreement
- Rubric written with 3–5 examples per score level
- **Versioned with its own review** — a release that can edit its exam isn't tested
- Held-out slice you look at rarely
- Replace 10–20% per quarter with recent production failures; keep a frozen core

**CI gate pass criteria:** no slice drops > threshold; faithfulness ≥ 0.85; hallucination ≤ 0.03; p95 latency growth ≤ 15%; cost per request growth ≤ 20%.

---

## 9. Guardrails

| Layer | Budget | What it catches |
|---|---|---|
| Input normalisation | ~10 ms | zero-width chars, homoglyphs, HTML comments, base64 |
| Injection patterns + classifier | ~60 ms | direct injection, jailbreak attempts |
| Input PII scan | ~40 ms | PII entering the system |
| Topic / relevance | ~50 ms | off-topic, scope violation |
| **Input total** | **~160 ms** | |
| Grounding / faithfulness check | ~120 ms | unsupported claims |
| Output PII scan | ~80 ms | **egress — BLOCK, never silently redact** |
| Toxicity / safety | ~80 ms | harmful content |
| Schema validation | ~50 ms | malformed structured output |
| **Output total** | **~330 ms** | |

**Total in-path: ~490 ms, ~$0.0007 per request** (~$2.70 per 1M requests).

**Instruction hierarchy:** system > developer > user > tool content. Never treat tool or retrieved content as instruction-bearing.

**The lethal trifecta:** private data + untrusted content + outbound channel. Remove any one and exfiltration becomes structurally impossible. Guardrails act on none of the three — the four architectural controls do: least privilege, no credentials in context, egress allowlist, out-of-band confirmation.

**Action risk tiers:**

| Tier | Examples | Gate |
|---|---|---|
| T0 | read-only lookup | auto |
| T1 | log, tag | auto + audit record |
| T2 | internal email, create ticket | auto + notify |
| T3 | customer-facing, financial, access change | confirmation required |
| T4 | delete, revoke, bulk operation | confirmation + typed scope acknowledgement |

The gate runs in **code**, in order: permission → schema verification → risk tier → suspicious flags → velocity → budget. Confirmation is bound to the **canonicalised arguments**, is single-use, has a ~60 s TTL, and is delivered out of band.

**Sandbox requirements for code execution:** loopback-only egress, `169.254.169.254` blocked, no host mounts, no secrets in env, ~300 s wall clock, ephemeral container.

---

## 10. LLM serving and inference

**KV cache size:**

```
KV bytes = 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per_param

≈ 0.31 MB per token  for a 70B model
≈ 0.131 MB per token for an 8B model
```

Worked example: A100 80 GB, 8B model in FP16 (16 GB weights) → ~62 GB for KV → at 4k average sequence length, roughly **118 concurrent sequences**. This — not FLOPs — is what usually caps concurrency.

**Prefill vs decode:**

| | Prefill | Decode |
|---|---|---|
| Processes | the whole prompt, in parallel | one token at a time |
| Bound by | compute | memory bandwidth |
| Determines | **TTFT** | inter-token latency |
| Optimised by | chunked prefill, caching | continuous batching, quantisation |

A GPU can be compute-saturated on prefill and bandwidth-idle on decode — which is why a naive token-throughput capacity estimate can be off by 4x.

**Optimisation toolbox:**

| Technique | Gain | Cost |
|---|---|---|
| Prompt caching | 40–60% input cost reduction on stable prefixes | requires stable, static-first prompt order |
| Continuous batching | large throughput gain | slight per-request latency variance |
| Chunked prefill | smooths TTFT under long prompts | implementation complexity |
| Speculative decoding | 2–3x latency, needs ≥70% acceptance | more total compute |
| KV quantisation | memory reduction, more concurrency | small quality risk |
| Cascade routing | 45–85% cost saving at ~95% quality | quality risk on misrouted queries |
| Batch API | ~50% off list | latency — offline only |
| Semantic caching | 20–40% hit rate | false-hit risk (correctness) |

**Cost structure to remember:**

- Output is **3–5x** the price of input per token
- But **input usually dominates the bill** for RAG/agents (e.g. 8,000 in / 300 out)
- System prompts + tool definitions can be **~69% of all input tokens** and are identical every request
- Only **~28%** of calls use caching in measured systems — the biggest untapped lever
- Reasoning models multiply output tokens **3–15x**
- Inference is **80–90%** of GenAI spend

**Self-host break-even:** below ~**4,000–8,000 calls/day**, self-hosting never wins. Apply a **2.5–3x true-cost multiplier** to the GPU line (cluster, people, idle capacity). For a 70B-class tier, the planning anchor is roughly **500M tokens/month**. Target utilisation 60–70% for latency-sensitive workloads.

**Latency reduction order:** cap output length → stream (perceived latency is TTFT) → prompt caching → cascade route simple queries → reduce rerank candidate set → index quantisation. Guardrails on a sample, not in-path at full cost.

---

## 11. Memory and context engineering

**Context ordering rule:** static first, volatile last — so prompt caching hits on the stable prefix.

**Context budget:** decide the token allocation per component (system, tools, retrieval, history, memory) and enforce it. Irrelevant context measurably degrades answers; it's not just a cost.

**Conversation history techniques:**

| Technique | Keeps | Loses | Cost |
|---|---|---|---|
| Sliding window (last N turns) | recent flow | early context | free |
| Summarisation | the gist | detail; adds its own errors | 1 LLM call |
| Retrieval over history | relevant detail | — | index + latency |
| Structured state extraction | the facts that change the answer | nuance | 1 LLM call |

**Recommended combination:** structured state for must-not-lose facts + sliding window for recent flow + summarisation for the middle.

**Agent memory — two paths, both hard:**

| Path | Hard part | Control |
|---|---|---|
| Write | deciding what's worth remembering | extraction quality bar + dedupe + **validation** |
| Read | recalling the right thing without flooding context | relevance recall + count budget |

**Memory poisoning** — if an injected instruction can write to memory, the compromise persists across sessions. Controls: provenance tracking, reject writes containing instructions rather than facts, tenant/user scope binding, periodic audit.

**Per-operation memory evals:** extraction / update / QA measured separately. A 5% extraction error compounds — a wrong memory generates future wrong behaviour.

---

## 12. Fine-tuning vs RAG vs prompting

| | Prompting | RAG | Fine-tuning |
|---|---|---|---|
| Gives the model | instructions | knowledge | behaviour |
| Changes | how it responds | what it knows | how it acts |
| Update cost | seconds | minutes (re-index) | days–weeks (retrain) |
| Attribution | none | **source citations for free** | none |
| Data needed | 0 | corpus | 500–1,000+ curated pairs |
| Best at | format, persona, constraints | facts, freshness, citation | tone, format compliance, narrow tasks |
| Fails at | missing knowledge | behaviour problems | stale knowledge, confabulation |

**The decision test:** *If I gave the model the right document in the context, would it answer correctly?* Yes → retrieval problem. No → behaviour problem → fine-tune.

**LoRA:**

```
h = Wx + (BA)x · (alpha / r)
r = 8–64 typically        alpha ≈ 2r        < 1% of parameters trained
```

Gives a 10–200 MB adapter you can swap per request; QLoRA adds 4-bit base quantisation.

**Fine-tuning failure modes:** catastrophic forgetting (evaluate on a general benchmark before *and* after), format overfitting (strong eval, poor production), data leakage (establish the holdout first), staleness, and **silent regression on untested slices** (multi-turn, non-English).

**Fine-tune an embedder instead when:** retrieval quality is the bottleneck due to domain vocabulary. Needs a few thousand (query, relevant-doc) pairs — but check hybrid + reranker first, and note it triggers a **full re-index**, so it's a scheduled change with a dual-index comparison before the alias flips.

---

## 13. Numbers to remember

| Number | Value |
|---|---|
| Chunk size | 500–800 tokens, 12% overlap (tuned, not defaulted) |
| Contextual retrieval effect | 5.7% → 1.9% retrieval failure |
| HNSW | `M=32`, `ef_construction=256`, `ef_search=64–128` |
| RRF | `k = 60` |
| Rerank | 50 → 5 candidates |
| Reranker nDCG@10 | 0.65 → 0.78 |
| Hybrid gain over dense-only | +8–15 pts recall@10 |
| MRL truncation | 1536 → 256 dims |
| Binary quantisation | ~32x memory reduction |
| Index size, 10M × 1536d | ~97 GB (61.4 raw × 1.5 HNSW + 5 metadata) |
| KV cache | 0.31 MB/token (70B), 0.131 MB/token (8B) |
| Concurrency, A100 80GB / 8B / 4k | ~118 sequences |
| Speculative decoding | 2–3x speedup at ≥70% acceptance |
| Cascade routing saving | 45–85% at ~95% quality |
| Batch API discount | ~50% |
| Output vs input price | 3–5x |
| System prompt share of input | ~69% |
| Calls using caching | ~28% |
| Reasoning token multiplier | 3–15x |
| Inference share of GenAI spend | 80–90% |
| Self-host break-even | 4,000–8,000 calls/day |
| True-cost multiplier for self-host | 2.5–3x |
| Self-host utilisation target | 60–70% |
| Semantic cache hit rate | 20–40% |
| Abstention band | 8–15% |
| Faithfulness target | ≥ 0.85 |
| Hallucination ceiling | ≤ 0.03 |
| Schema-valid rate | ≥ 0.995 |
| Judge kappa | target ≥ 0.6, alert < 0.4 |
| Guardrail budget | ~490 ms, ~$0.0007/request |
| Eval gold set size | 200–500 cases |
| Eval stack share of inference cost | 2–4% |

---

## 14. Interview one-liners

- "Prompting gives instructions, RAG gives knowledge, fine-tuning gives behaviour."
- "Chunk size is tuned against the corpus, never accepted as a default."
- "Hybrid retrieval because dense and sparse fail differently; RRF because cosine and BM25 scores aren't comparable."
- "If you have a quality problem and no reranker, add one before touching anything else."
- "Retrieval failure and generation failure have different fixes — label 200 failures before tuning either."
- "Long context has eaten some RAG use cases. The decision is corpus size and query frequency, not ideology."
- "A cache is a correctness component that looks like a cost component."
- "The tenant filter goes in the query, not in the prompt."
- "Perceived latency is TTFT, not total."
- "A model that works is not a feature that works."
- "Fairness only becomes real when it's a number in a gate that can fail."

---

## Cross-links

- [Production RAG System](../HLD/05-ai-systems/01-production-rag-system.md)
- [Advanced RAG Patterns](../HLD/05-ai-systems/02-advanced-rag-patterns.md)
- [RAG Evaluation and Observability](../HLD/05-ai-systems/03-rag-evaluation-and-observability.md)
- [Guardrails and Safety for AI Systems](../HLD/05-ai-systems/04-guardrails-and-safety-for-ai-systems.md)
- [LLM Serving, Inference and Gateway](../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md)
- [Memory and Context Engineering](../HLD/05-ai-systems/08-memory-and-context-engineering.md)
- [Fine-Tuning vs RAG vs Prompting](../HLD/05-ai-systems/09-fine-tuning-vs-rag-vs-prompting.md)
- [LLM, RAG and GenAI Questions](../Interview-Questions/04-llm-rag-genai-questions.md)
- [Agentic AI Cheatsheet](05-agentic-ai-cheatsheet.md)
- [Operationalisation Cheatsheet](06-operationalisation-cheatsheet.md)
- [Production RAG Case Study](../Case-Studies/HLD/09-production-rag-case-study.md)
