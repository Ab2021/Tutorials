# 05 — SYSTEM DESIGN ROUND PLAYBOOK

> **TL;DR.** The design round is scored on **process, not the final diagram**. Five phases: clarify → estimate → draw → deep-dive → self-critique. Ask **one question per dial** (quality, cost, latency, safety) before you draw anything. Add a box only when a constraint you surfaced justifies it — extra sequential components *multiply* error (0.9³ = 0.73). This file gives the framework, the arithmetic, the component-by-component reference architecture, the deep-dive material, and **three worked prompts including the image-anomaly design they asked last time**.

**60-second interview answer (if asked "how do you approach a design question?"):**

> *"I never draw first. I spend the first few minutes on four questions — one per dial: the accuracy bar and cost of a wrong answer, the latency target, the per-unit cost budget, and whether the output is regulated. Then I do quick capacity math so the sizing is grounded. Then I sketch left-to-right and justify every box against a constraint. Then I let you pick the subsystem to go deep on. And I finish by naming what I've left out, because that's usually more useful than claiming it's complete."*

---

## 1. The five phases and their time budget

Scale to the ~15–20 minutes the round gives you (the candidate feedback shows Part 3 is one of four).

| Phase | Time | You are doing | What they score |
|---|---|---|---|
| **1. Clarify** | 3 min | One question per dial + scale + freshness | Do you ask about scale, latency, cost **and safety**? |
| **2. Estimate** | 2 min | QPS, peak, tokens/sec, storage, cost | Do you reason from numbers, not assumptions? |
| **3. Draw** | 6 min | Left-to-right boxes, labelled, justified | Is every box traceable to a constraint? |
| **4. Deep-dive** | 6 min | The subsystem they name | Retrieval tuning, retries, streaming, serving |
| **5. Self-critique** | 2 min | Name what you left out + what you'd monitor | Maturity |

> **If you get 10 minutes instead of 20:** collapse to clarify (2) → draw (5) → one deep-dive (3). Never skip clarify.

---

## 2. Phase 1 — clarifying questions (the four dials)

Ask exactly one per dial, then the three context questions. Write them down as you ask.

| Dial | The question | Why the answer changes the design |
|---|---|---|
| **Quality** | "What's the accuracy bar, and what does one wrong answer cost?" | Decides the model tier, whether you need reranking, whether the human is mandatory |
| **Cost** | "Is there a per-decision or per-conversation budget?" | Decides model tiering, caching strategy, batch vs real-time |
| **Latency** | "What's the P95 target, and what does the user actually tolerate?" | Decides streaming vs batch, cache placement, whether an agent loop is affordable |
| **Safety** | "Is the output regulated, and who is accountable for the decision?" | Decides human-in-the-loop, explainability, audit logging, on-prem vs API |

Plus three:

| Context | Question |
|---|---|
| **Scale** | "How many [documents / requests / users] per day, and what's the peak multiplier?" |
| **Freshness** | "How often does the source data change?" (weekly → RAG mandatory; live → the index is a cache and must not be the source of truth) |
| **Constraints** | "Data residency, tenancy, or any restriction on third-party providers?" |

**Line to say while writing them down:** *"I'll take these as the constraints, and every box I add has to earn its place against one of them."*

---

## 3. Phase 2 — capacity arithmetic (do it out loud)

### 3.1 The formulas

```
QPS          = daily_requests / 86,400
Peak QPS     = QPS × peak_multiplier          (2–3× typical)
Prefill tput = avg_input_tokens  × Peak QPS   (tokens/sec)
Decode tput  = avg_output_tokens × Peak QPS   (tokens/sec)
In-flight    = Peak QPS × duration_seconds    (Little's Law)
KV cache     = in_flight × avg_context_tokens × KV_bytes_per_token
Monthly cost = total_tokens × token_price
Storage      = num_chunks × dims × 4 bytes (+ index overhead ≈ 1.5–2×)
```

### 3.2 Worked example A — a support assistant

> 200,000 chats/day · 500 input tokens (200 system + 200 history + 100 retrieved) · 300 output tokens · P95 12 s

- Median QPS = 200,000 / 86,400 ≈ **2.3**; peak ≈ **7** (3×)
- Peak prefill = 7 × 500 = **3,500 tok/s**; peak decode = 7 × 300 = **2,100 tok/s**
- In-flight = 7 × 12 s = **84 concurrent**
- KV cache ≈ 84 × 500 × 0.5 MB/1k ≈ **21 GB** → this is the sizing constraint, not FLOPS
- Cost: input ≈ $0.50/M, output ≈ $1.50/M (output ≈ 3× input)
  → per conversation ≈ (500×0.5 + 300×1.5)/1M ≈ **$0.0007** at cheap-tier pricing; at frontier pricing it is 10–30× that → **this is why tiering matters**

### 3.3 Worked example B — the 100× trap (say this, it's memorable)

> *"A 20-token request and an 8,000-token request are both 'one request' to a rate limiter, but they differ ~100× in cost — roughly $0.000055 versus $0.0055. That's why RPS-only rate limiting is inadequate for LLM workloads: you need a token-based limit as well, and separate input and output budgets, because output is ~3× the price of input."*

### 3.4 Worked example C — document ingestion (insurance)

> 1,000,000 multi-page PDFs/day · avg 20 pages · ~1,200 tokens/page

- Pages/day = **20 M**; if chunked at 600 tokens with a 520-token stride → ~3 chunks/page → **60 M chunks/day** — that is not a one-shot job; it must be a queue with backpressure and a batch embedding path
- Storage: 60 M × 768 dims × 4 B ≈ **184 GB** raw vectors; with HNSW graph overhead ≈ **1.5–2×** → ~300 GB RAM-resident if fully in-memory → **requires sharding or an IVF-PQ/disk-backed index**
- *"The estimate people miss is the stride: 600-token chunks with 80-token overlap means a 520 stride, so a 1,200-token page produces 3 chunks, not 2. That 50% error is the difference between a design that fits and one that doesn't."*

### 3.5 Latency decomposition (use this when something is slow)

| Stage | Typical | Note |
|---|---|---|
| Queue | 10–50 ms | Spikes silently under load |
| Retrieval (ANN + filter) | 100–300 ms | Pre-filtering is the main lever at scale |
| Rerank (cross-encoder, top-50→5) | 100–200 ms | The quality you pay for |
| Prompt build | 10–30 ms | — |
| Prefill | 150–500 ms | Scales with input length |
| **Decode** | **dominant** | output_tokens × per-token time (e.g. 350 × 18 ms ≈ **6.3 s**) |
| Guardrails / validation | 50–100 ms | — |

**The lesson to state:** *"When a RAG endpoint is slow, the decode term usually dominates — optimising retrieval from 120 ms to 60 ms buys you almost nothing against a 6.3 s decode. You only know that if you decompose before you optimise."*

**TTFT for a RAG system** = retrieval + rerank + prefill ≈ **0.5–1.2 s**. That is the number the user *feels*; total generation is not.

---

## 4. Phase 3 — the reference architecture (draw this from memory)

```
Client
  │
  ▼
Load Balancer          ── distributes, removes SPOF
  │
  ▼
API Gateway            ── auth · rate limit (RPS + tokens) · cancellation propagation
  │
  ▼
Intent Router          ── classify complexity → small / mid / frontier
  │
  ├──────────────► RAG Module     ── vector DB + BM25 + metadata pre-filter + reranker
  ├──────────────► Session Memory ── last N turns; long-term store
  └──────────────► Tool Registry  ── external APIs, validated server-side
  │
  ▼
LLM Service            ── prefill + decode, streamed (SSE)
  │
  ▼
Output Validator       ── schema · entailment · PII · confidence
  │
  ▼
Human-in-the-loop gate ── only for irreversible / high-value actions
  │
  ▼
Tracing & Eval Layer   ── every request = a trace; golden set gates deploys
  │
  ▼
Response
```

### 4.1 What each component prevents (say this, not just the name)

| Component | The failure it prevents |
|---|---|
| Load balancer | Single point of failure; uneven load |
| API gateway | Unauthenticated access; cost runaway (**token-based** limits, not just RPS) |
| Intent router | Paying frontier prices for FAQ traffic |
| RAG module | Stale knowledge; ungrounded answers |
| Session memory | Amnesia across turns |
| Tool registry | Unvalidated side effects |
| Streaming (SSE) | Perceived latency — the *user-visible* fix |
| Output validator | Malformed or ungrounded output reaching a workflow |
| HITL gate | Irreversible damage |
| Tracing/eval | Silent quality regression |

**The justification line:** *"Let me justify each box against the constraints you gave me: the router exists because you have a cost budget; the validator exists because the output is regulated; the human gate exists because binding risk is irreversible. If you took the cost budget away, I'd delete the router."*

---

## 5. Phase 4 — the deep dives (know all five; they pick one)

### 5.1 Retrieval — the funnel

| Step | Choice | Numbers |
|---|---|---|
| Chunk | Fixed / structure-aware / parent-child / semantic / propositions | FAQ 100–300; general 300–500; legal 500–1,000; code 50–200. Overlap **10–20%**. |
| Embed | Model choice | ada-002 1536·d; text-embedding-3-large 3072·d; BGE-M3 1024·d (multilingual + hybrid); E5-large 1024·d |
| Index | HNSW (fast, memory-heavy) vs IVF-Flat vs IVF-PQ (compressed, lossy) | HNSW `M` and `efSearch` trade recall for latency; 1M→HNSW was ~**392× cheaper queries for ~3 points of recall** |
| Retrieve | Hybrid | BM25 + dense, fused by **RRF** (`Σ 1/(k+rank)`, k=60) |
| Filter | Metadata **pre-filter** | tenant, LOB, date, authority — shrinks the search space before the ANN scan |
| Rerank | Cross-encoder top-50 → top-5 | Usually the **single highest-leverage quality win** |
| Assemble | Most relevant chunk **first** | "Lost in the middle" — the model attends poorly to mid-context |

**Parent-child, stated crisply:** *"Small child chunks (100–200 tokens) are indexed because they give precise embeddings; when one is retrieved I fetch its parent (500–1,000 tokens) to give the model enough context. The children point at the parent by ID and the parents are not in the ANN index at all."*

**Freshness (they asked this of another candidate at Razorpay/ICICI):** *"Hash each chunk's content and re-embed only changed chunks; subscribe to webhooks/CDC rather than polling; upsert by deterministic chunk ID and delete vectors for removed sections; keep the write path off the read path with a queue and an atomic index swap; and track index lag per document with an alert. Freshness is an incremental-sync problem, not a re-index schedule."*

**The live-value trap (a great answer if you get the chance):** *"If the question is 'what is my balance?', RAG is the wrong shape — that's a live value and must come from the system of record. The index is a cache of documents, not a database of state. Retrieving a statement to answer a balance question gives a fluent, confident, stale answer."*

### 5.2 Streaming and perceived latency

| Fact | Number |
|---|---|
| Human reading speed | ~5.5 tokens/sec |
| Model decode | 25–50 tokens/sec |
| Buffered: 0.9 s TTFT + 380 tokens @ 25 tok/s | **16.1 s** before anything appears → **38% abandonment** |
| Doubling the model to 50 tok/s, still buffered | ~8.5 s → abandonment only falls to **22%** |
| **Streaming** | Text at 0.9 s → the perceived-latency fix |

**The layers that must all be un-buffered** (a favourite follow-up):

> *"Streaming at the provider isn't enough. I have to disable buffering at the assistant service, in the API gateway config, at the reverse proxy — Nginx `proxy_buffering off` — and **Gzip/Brotli must be off on streaming endpoints**, because compression waits to accumulate bytes and silently re-introduces the latency I just removed."*

**SSE over WebSockets:** one-way is all you need, standard HTTP traverses proxies and CDNs, native `Last-Event-ID` for resume without sticky sessions. WebSockets are bidirectional and need sticky sessions or a pub/sub bus.

**Event ledger + cancellation:** *"I keep a short log of emitted event IDs so a dropped mobile connection resumes from `Last-Event-ID` instead of restarting the generation — which also prevents duplicate tool calls. And when the client disconnects, I propagate a cancellation signal all the way to the provider, otherwise the GPU keeps generating tokens for a dead connection."*

### 5.3 Reliability — timeouts, retries, idempotency

The three artefacts they will probe:

| Problem | Answer |
|---|---|
| **Timeout mismatches** | Downstream timeouts must be **shorter** than upstream. Deadline propagation: browser 10 s > gateway 8 s > service 6 s > provider 5 s, propagating cancellation inward. Otherwise you get zombie work nobody is waiting for. |
| **Retry amplification** | Browser ×3 · gateway ×3 · service ×3 = up to **27** calls per user click. Fix: **one layer retries**, exponential backoff with **full jitter** (`random(0, min(cap, base·2^attempt))`), and a **retry budget** — a token bucket capping retries at ~10% of request rate. Amplification drops to ~1.1×. |
| **Duplicate side effects** | Idempotency key per **intent** (not per HTTP request), stored in Redis with `in-flight` status; a retry with the same key waits or returns the cached result. Reads are safe to retry freely; **writes always gate**. |

**The duplicate-refund story (use it — it makes the point concrete):** *"A user clicked refund, the UI spun for 16 seconds, they clicked three more times — and the system issued the refund twice. The root cause wasn't the UI; it was that each retry was treated as an independent request. Idempotency keyed on the intent fixes it."*

### 5.4 Serving & inference (vLLM, batching, quantisation)

| Topic | What to say |
|---|---|
| **Prefill vs decode** | Prefill processes the whole prompt in parallel and builds the KV cache — fast and cheap (~$0.50/M). Decode is sequential (token *n* needs tokens 1…*n*−1) and **memory-bandwidth bound** — ~$1.50/M, ~3× the price. |
| **KV cache** | `2 × layers × heads × head_dim × seq_len × bytes`. A 7B fp16 ≈ **0.5 MB per 1,000 tokens**; a 200K context ≈ **100 GB**. This, not FLOPS, is what runs out. |
| **Continuous batching** | vLLM/TGI insert new requests into the running batch at each step instead of waiting for the slowest — the biggest throughput win. Static batching makes every request as slow as the longest. |
| **PagedAttention** | Stores the KV cache in non-contiguous pages, removing the fragmentation that wastes 60–80% of VRAM in naive serving. |
| **Speculative decoding** | A small draft model proposes N tokens; the target verifies them in one parallel pass — **2–3× decode speedup with the same output distribution**. Latency win, not quality change. |
| **Quantisation** | 7B fp16 = 14 GB, int8 = 7 GB, int4 = 3.5 GB. fp16 lowest loss; int8 usually safe; **int4 (GPTQ/AWQ) is noticeably lossy on complex reasoning** — I'd A/B it rather than assume. |
| **TTFT vs ITL/TPOT** | TTFT = retrieval + rerank + prefill → what the user feels first. ITL = per-token time → total duration. Optimise TTFT for perceived speed, ITL for completion time. |
| **GPU utilisation** | A GPU at 30% costs the same as at 90% but serves 3× fewer. Continuous batching and speculative decoding raise *effective* utilisation without changing the hardware bill. That's why cutting p95 halved a real GPU bill. |
| **When self-hosting wins** | High steady volume, data residency, or a hard "no third-party provider" constraint. Below that, managed APIs win on everything except control. |

### 5.5 Cost control — the five levers, in order

| Lever | Typical effect |
|---|---|
| **Model tiering** | 75% small / 20% mid / 5% frontier hits a $0.02/conversation budget where all-frontier costs 10–30× more |
| **Semantic cache** | 20–40% hit rate on repetition-heavy traffic; cosine ≥ 0.95; **must be keyed on context** (dates, tenant, version) and invalidated on the write path |
| **Provider prompt caching** | Cache reads at ~10% of input price — 80–90% off the repeated prefix; one changed token invalidates it |
| **Output-length control** | Output is ~3× input; `max_tokens`, JSON instead of prose, "direct answer first" |
| **Batch API** | ~50% cheaper for non-interactive work, at the cost of minutes-to-hours latency |

**The trap to name:** *"A cheaper model is only cheaper per *resolved task*. If the weaker model retries, needs longer prompts, or escalates more, the all-in cost per resolution can be higher than the expensive model. I've seen an AI line item go up six months after a model downgrade with usage flat — that's the mechanism."*

---

## 6. The three worked prompts (one is almost certainly yours)

### 6.1 ⭐ The image-anomaly design (asked verbatim last time)

**Prompt:** *"Draw a system design diagram — given you have images and you need to run these images through a model for checking anomalies."* This is Brit's post-catastrophe property-damage use case, so treat it as a **vision pipeline with a human decision** and it lands perfectly.

**Clarify first (always):**
- *"Are these ultra-high-resolution aerial/satellite images, or ground photos?"*
- *"What's the accuracy bar, and which error is worse — calling damage where there is none, or missing damage?"*
- *"Is a human reviewing every flagged image, or only the model's positives?"*
- *"What's the volume — thousands or millions of images, and is it bursty after a catastrophe?"*
- *"What latency — real time, or overnight batch is fine?"*

**Then draw:**

```
Image source (aerial/drone/satellite, very large tiles)
   │
   ▼
Ingest queue (burst-tolerant — post-cat events are spiky)   ← backpressure, not sync
   │
   ▼
Tiling / preprocessing   ── tile huge rasters into model-sized patches *with overlap*
   │                        store tile→image mapping for traceability
   ▼
Model service (batched inference on GPU)
   ├── detection model (e.g. YOLO-family) → bounding boxes of candidate damage
   └── or segmentation model → pixel masks
   │
   ▼
Geospatial post-processing  ── stitch tiles back, de-duplicate boxes across overlaps,
   │                           map to the property/policy record
   ▼
Decision policy (deterministic)  ── confidence threshold + severity rules in CODE, not prompt
   │
   ├── high confidence  → auto-flag for accelerated handling
   ├── low confidence   → human adjuster queue  ← HITL, mandatory
   └── model unavailable → full manual path     ← regulated service must degrade, not fail
   │
   ▼
Evidence store  ── original image, tiles, boxes, model version, confidence, reviewer decision
   │                ← this is the audit trail an auditor asks for
   ▼
Metrics: precision at the auto-flag threshold · recall in the human queue · review time saved
```

**The five things to say that make it a Lead answer:**
1. **"The burst is the design problem."** Post-catastrophe volume arrives in hours; the pipeline must be queue-based with backpressure, not a synchronous API.
2. **"Tile with overlap and store the mapping."** Huge rasters don't fit; overlap prevents edge-cutting damage; the tile→image→property mapping is what makes the output traceable.
3. **"The threshold is a policy decision in code, not a model output."** Deterministic rules decide auto-flag vs human — this is the "model proposes, policy disposes" line.
4. **"Precision protects the insurer; recall protects the customer."** *"I'd set a high threshold for auto-flagging because a false damage call sends an adjuster out for nothing, and I'd keep recall high in the human queue so no genuine claim is missed. And in a regulated market I'd write down which one I'm trading."*
5. **"The fallback is manual, not an error."** Claims service must continue when the model is down.

**Bonus if they push on "how do you know the model is good?"** → *"A labelled set built with adjusters, per-class precision/recall (not just accuracy, because damage classes are imbalanced), and drift monitoring — new imagery types, new regions, seasonal changes. And I'd evaluate on *properties*, not images, because the business decision is per property."*

### 6.2 The document-ingestion / submission-triage design (the JD's core use case)

**Prompt shape:** *"Millions of multi-page PDFs daily, strict accuracy guarantees, deterministic fallback."*

```
Broker/email ingestion → object store (raw, immutable)
   → queue → OCR/layout extraction (Document Intelligence) 
   → field extraction (constrained JSON schema) 
   → deterministic validator (Pydantic: types, enums, ranges, cross-field rules)
   ├── valid   → system of record (write-back)
   ├── invalid → human review queue with the failing field highlighted
   └── low confidence → human review
   → dedupe by content hash + submission ID (idempotent upsert)
   → trace: page, box, span for every field
```

**The three lines that matter:** *"Immutable raw storage so I can always re-extract without asking the broker again. Server-side schema validation so a malformed extraction is an explicit failure, not a subtly wrong number. And field-level traceability — every extracted value points to a page and a box, because an underwriter has to be able to check it."*

### 6.3 The support-copilot design (from the mock-interview prompt)

Use the reference architecture verbatim, then land on: **P95 < 200 ms is impossible for a generative response** — so split the budget: *"If the SLA is 200 ms, that's an intent-classifier + retrieval + cached-response budget, not a generation budget. A streamed generative answer has TTFT ~0.5–1.2 s minimum. So either the 200 ms applies to the routing/ack path and generation streams after, or we're talking about a retrieval-and-template system, not a generative one. Which is it?"* — **questioning an unrealistic constraint is a Lead behaviour, not a failure.**

---

## 7. Phase 5 — self-critique (the two minutes that win the room)

Say, unprompted:

> *"Things I haven't covered that I'd firm up: the **cost ceiling** and what degrades first when we hit it; **failure modes** — what happens when the vector DB is down, when the model returns invalid JSON, when the same request arrives twice; **drift monitoring** post-launch; and the **adoption** plan — how the underwriter's day actually changes, which is usually the part that decides whether any of this was worth building."*

**Name the trade-offs explicitly** (this is the list they reward):

| Choice | You gain | You pay |
|---|---|---|
| More context | Better accuracy | Higher token cost, higher latency |
| Higher recall | Less missed evidence | More noise, more reranking cost |
| More agents | Flexibility | More failure nodes, harder to debug |
| Central gateway | Governance | Single point of failure |
| Stateless + Redis | Easy scaling | Network hop, new critical dependency |
| Semantic chunking | Better boundaries | Compute cost to embed every sentence |
| Reranking | Big quality lift | 100–200 ms + cost per query |

---

## 8. Common mistakes in the design round

| Mistake | Fix |
|---|---|
| Drawing before clarifying | The four dials, always |
| Adding boxes that no constraint justifies | "Every box must earn its place" |
| Using one model for everything | Tier it, and state the cost difference |
| Ignoring the human | In insurance, name where the human decides |
| Not stating TTFT vs total latency | Split them; the user feels TTFT |
| Treating a live value as a RAG question | The index is a cache, not a system of record |
| Claiming it's complete | Self-critique is scored |
| Not asking about data residency | Brit is UK-regulated; residency and third-party constraints are real |
| Answering a 200 ms generative SLA as if it were achievable | Challenge the constraint, then offer the split |

---

## 8.5 Serving numbers to have on the tip of your tongue

Use **at most two** of these in a design answer. More reads as recitation.

**KV cache — the formula, then the factor:**

```
KV_bytes/token = 2 × layers × KV_heads × head_dim × dtype_bytes
GPU = weights + activations + KV cache + ~1–2 GB framework overhead
```

| 70B-class model | KV per token | Factor |
|---|---|---|
| MHA BF16 | 2.5 MB | 1× |
| **GQA BF16** | **0.320 MB** | **8×** |
| **GQA + FP8 KV** | **0.160 MB** | **16×** (<0.3% quality loss) |
| MLA (DeepSeek-V3 style) | 0.00113 MB | 57× |

- **H100, FP8 weights, TP=2 (~40 GB KV headroom):** 4K ctx → ~62 users; 8K → ~31; 32K → ~8; **128K → ~1–2**.
- **Decode is memory-bandwidth-bound:** a 70B FP8 on 2×H100 ≈ **10.4 ms/token → ~96 tok/s single-stream**. Batching to 64 lifts aggregate throughput to ~3,855 tok/s but worsens ITL.
- **PagedAttention** (vLLM default block=16) reclaims the **60–80%** of HBM wasted by contiguous pre-allocation. **Prefix/RadixAttention at 80% hit → ~71% fewer prefill tokens → ~3.5× better TTFT.**

**Quantisation — the average quality drop vs BF16 (EMNLP 2025; 5 methods × 5 models):**

| Method | Avg drop |
|---|---|
| **FP8 W8A8** | **−0.2%** |
| GPTQ int8 | −0.8% |
| **AWQ int4** | **−1.8%** |
| GPTQ int4 | −2.7% |
| **BNB-nf4** (HF/vLLM 4-bit default) | **−6.9%** |

- Degradation **worsens with context**: 4-bit up to **−23% at 128K**; worst case **−59%** (Llama-3.1-70B, BNB-nf4, long context). Non-English degrades up to **5× more**.
- **AWQ generally beats GPTQ**; a 4-bit *larger* model can beat a smaller FP16 one.
- **The line to quote:** *"Activation quantisation — W4A4 — costs about 30% on multi-step reasoning versus about 4% for weight-only. Reasoning is far more quant-sensitive than perplexity suggests, so I'd quantise aggressively for retrieval and summarisation but never for the underwriting reasoning step without re-running the task evals."*

**Embeddings:** dims sweet spot **768–1,024** (384→1,024 gains ~5–10%; 1,024→3,072 only ~1–3%). OpenAI `text-embedding-3-large` 3,072 dims (Matryoshka-truncatable to 256/1,536, ~92–95% quality retained); `3-small` 1,536; Cohere embed-v4 1,024/128K tokens/multimodal; **BAAI BGE-M3** 1,024 and **hybrid dense+sparse+ColBERT, self-hostable on one A10G**. 1M vectors @1,024 float32 ≈ **4.1 GB**. → *"MTEB retrieval scores cluster tightly — a 2-point gap rarely survives contact with your own domain, so I'd benchmark on our data, and switching models means a full re-embed with a dual-index migration."*

**Chunking empirically:** quality **plateaus by ~32 tokens** and **collapses below ~16**; semantic chunking beats fixed-size by ~0.09 nDCG at small sizes. Practical default **256–512 tokens**.

**Reranking:** two-stage (fast ANN → cross-encoder) improves **precision@1 by 5–15%** for ~100 ms. **Rule: add a reranker when P@1 < 0.85.**

**Semantic caching (2025):** Redis + HNSW at cosine **0.8** → **up to 68.8% fewer API calls**, hit rate 61.6–68.8%, positive-hit accuracy >97%; **~31% of ChatGPT queries are semantically similar to a prior one**. **vCache (Berkeley)** uses per-embedding **learned** thresholds — up to **12.5× higher hit rate and 26× lower error** than static GPTCache, ~1.5 ms overhead. → *"Static cosine thresholds are unreliable; the state of the art is a learned per-embedding threshold — and you still verify the cached answer against the source before serving it."*

**Engine choice:** vLLM = peak throughput and tail stability (wins single-request TTFT on 15/16 models); **SGLang = low-concurrency TTFT and prefix reuse via RadixAttention**; TensorRT-LLM = raw throughput and long-context prefill; TGI = deployment simplicity. *[Benchmark figures are indicative — harnesses disagree.]*

---

## 9. Cheat-sheet summary


- **Clarify (4 dials) → estimate → draw → deep-dive → self-critique.** Never draw first.
- **Math:** QPS = daily/86,400; peak ×2–3; in-flight = QPS × duration (Little's Law); output ≈ 3× input cost; KV ≈ 0.5 MB/1k tokens (7B fp16).
- **Reference architecture:** LB → gateway → router → (RAG + memory + tools) → LLM → validator → HITL → tracing.
- **Retrieval funnel:** chunk → embed → ANN → hybrid (RRF, k=60) → metadata pre-filter → rerank top-50→5 → most relevant first.
- **Perceived latency = streaming.** Buffered 16.1 s / 38% abandonment → streaming fixes it; a 2× faster model doesn't.
- **Reliability = deadline propagation + one retrying layer + full jitter + retry budget + idempotency keys.**
- **Compounding error:** 0.9³ = 0.73 — simplicity is a seniority signal.
- **Vision use case:** queue for bursts, tile with overlap, threshold in code, human queue, manual fallback, trace to property.
- **Self-critique always:** cost ceiling, failure modes, drift, adoption.

## 10. Cross-links

- Evaluation depth for rung 4: [`07-agent-evaluation-and-guardrails.md`](07-agent-evaluation-and-guardrails.md)
- Reliability code patterns: [`06-coding-exploration-playbook.md`](06-coding-exploration-playbook.md) §2
- Domain constraints (FCA/PRA, model risk, residency): [`11-insurance-lloyds-domain-primer.md`](11-insurance-lloyds-domain-primer.md)
- The mock prompt that mirrors this round: [`08-applied-ai-mock-interview-prompt.md`](08-applied-ai-mock-interview-prompt.md)
- Repo-wide AI-systems theory: [`../HLD/05-ai-systems/`](../../HLD/05-ai-systems/)
