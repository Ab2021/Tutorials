# 08 — APPLIED AI MOCK-INTERVIEW PROMPT & ALIGNED ANSWERS

> **TL;DR.** This file contains the **interviewer prompt** to paste into an LLM for a realistic Applied-AI system-design drill, plus the **pre-built answers** you should give against each of its four phases. The prompt is deliberately aggressive: it drills orchestration choice, serving and latency metrics, evaluation and guardrails, and data/storage. Everything in files `05` and `07` is what you feed into it.

**How to use it:** paste §1 into a fresh chat, answer the opening question out loud (not typed), then paste §2–§5 as reference *only when you get stuck*. The point is to be surprised.

---

## 1. The interviewer prompt (paste this)

```
You are a Principal Applied AI Engineer acting as an interviewer at a top-tier tech
company. You are interviewing me for a Lead Applied AI Engineer role.

We will conduct a realistic, interactive Applied AI System Design Interview. The focus is
strictly on AI application orchestration, scaling production pipelines, performance
optimization, and reliable system architecture — NOT on training models from scratch.

THE APPLIED AI CHALLENGE (pick one, or surprise me):
1. A real-time, context-aware AI Customer Support Copilot for an e-commerce giant that
   integrates live data, orchestrates multi-turn agent workflows, and operates under a
   <200ms latency budget.
2. A high-throughput, automated legal document ingestion and metadata extraction engine
   handling millions of multi-page PDFs daily with strict accuracy guarantees and
   deterministic fallback systems.
3. A personalized, multi-modal content recommendation and generation pipeline that
   dynamically creates tailored ad copy and images based on live user interaction graphs.

INTERVIEW RULES:
1. INTERACTIVE DIALOGUE ONLY: Do not output the solution. Present the problem and wait
   for my response. Treat this as a real, back-and-forth conversation.
2. DRILL DOWN ON APPLIED AI METRICS: Challenge me aggressively on engineering trade-offs:
   - Orchestration (LangChain/LangGraph vs custom state machines, event-driven loops).
   - Serving & Latency (vLLM concurrency, speculative decoding, Time-to-First-Token vs
     Inter-Token Latency).
   - Evaluation & Guardrails (Llama-Guard, programmatic semantic validators,
     cost-efficient LLM-as-a-judge setups).
   - Data & Storage (Vector DB hybrid search, metadata filtering, hybrid caching layouts).
3. STRUCTURE: Guide us sequentially through:
   - Phase 1: Applied Requirements & SLA Definition (Latency, Cost, Accuracy boundaries).
   - Phase 2: Orchestration Topology & Multi-Agent Design.
   - Phase 3: Concrete Infra Selection & Serving Strategy (Quantization, Chunking
     strategy, Embedding caching).
   - Phase 4: Production Fail-safes, Drift Monitoring, and Evaluation.

To begin, introduce yourself, state the exact Applied AI challenge you have selected for
me, and ask the opening question to kick off Phase 1.
```

### 1.1 Two variants that sharpen the drill for *this* interview

**Variant A — make it Brit-shaped.** After the challenge list, add:

```
CONTEXT: I am interviewing at Brit, a Lloyd's of London specialty insurer. Where it fits
naturally, frame challenges as underwriting, submission and claims problems, and hold me
to regulated-industry standards (explainability, auditability, human-in-the-loop, model
risk). Also ask me people-leadership questions about the team as we go — I am applying
for a Lead role that includes managing engineers.
```

**Variant B — force the four-part Stage-1 shape.** Add:

```
Run this as a 60-minute Stage-1 loop with four parts, in order:
(1) Introductions — ask me to introduce myself, then interrogate my background.
(2) Project drill-down — pick whichever project I describe and drill four levels deep.
(3) System design — the Applied AI challenge below.
(4) Coding exploration — show me a small RAG function and ask me to explain it and
    improve it; then ask me a Git question about linear history with a team.
Do not move on until each part is done, and interrupt me if I ramble.
```

---

## 2. Phase 1 — Applied Requirements & SLA Definition

**What the interviewer wants:** you convert a vague prompt into numbers *before* designing, and you name which dial you are sacrificing.

### 2.1 Your opening response (template — say this, don't type it)

> *"Before I design anything, four questions, one per dial.*
> ***Quality:** what's the accuracy bar, and what does one wrong answer cost? For a support copilot I'd assume a wrong refund is expensive and a wrong FAQ answer is cheap, so the bar may differ per action.
> ***Cost:** is there a per-conversation or per-decision budget? If it's the classic $0.02 target, that shapes the whole design, because a flat frontier-model approach costs 10–30× that.
> ***Latency:** you said sub-200ms. I want to flag that **a generative response cannot start in 200 ms** — retrieval plus rerank plus prefill is 0.5–1.2 s at best. So is the 200 ms the budget for the **routing and acknowledgement path**, with generation streamed after it? That distinction changes the architecture completely.
> ***Safety:** are there actions with real consequences — refunds, account changes, sending messages? Those need a human gate and idempotency, not a confident model.*
> *On context: what's the daily volume and peak multiplier, how often does the source data change, and is there any data-residency or third-party-provider restriction?"*

**Why this scores:** it asks one question per dial, it **challenges an unrealistic SLA with arithmetic**, and it pre-commits the safety gate.

### 2.2 The numbers you should volunteer here

| Quantity | Value | Source of the value |
|---|---|---|
| Human reading speed | ~5.5 tok/s | Streaming rationale |
| Buffered response, 380 tok @ 25 tok/s | 16.1 s → 38% abandonment | Why streaming wins |
| TTFT for RAG | retrieval 200–500 + rerank 100–200 + prefill 200–500 = **0.5–1.2 s** | The honest latency floor |
| Output vs input token price | **~3×** | Why output length is a cost lever |
| Cost ratio, 20-token vs 8,000-token request | **~100×**, both "1 request" | Why RPS-only limits fail |
| Tiering split | 75% small / 20% mid / 5% frontier | Hits $0.02/conversation |
| Semantic cache | cosine ≥ 0.95, 20–40% hit rate | Cost lever |

---

## 3. Phase 2 — Orchestration Topology & Multi-Agent Design

**The interviewer will attack:** *why an agent at all?* framework vs custom state machine? how do you stop loops? how do agents share state?

### 3.1 The escalation-ladder answer (use before defending an agent)

> *"Let me justify the complexity rather than assume it. Prompting first — if the task is static, that's it. RAG if the knowledge updates. Tools and an agent loop only if the task requires **acting** — API calls, writes, multi-step iteration where the next step depends on what the last returned. And fine-tuning only if the problem is *style*, not knowledge. For this challenge, the branching is genuine — refund path vs exchange path vs escalation depends on what the order lookup returns — so an agent is justified. If it weren't, I'd say so and delete it, because every extra sequential component multiplies error: 0.9 × 0.9 × 0.9 = 0.73."*

### 3.2 Framework vs custom state machine (the exact trade-off)

| Option | When it wins | The cost |
|---|---|---|
| **LangGraph / LangChain** | You want the graph, checkpointing, retries and tracing out of the box; the team already knows it | Framework churn, abstraction leakage, harder to debug a bad step |
| **Custom state machine** | A small, fixed, high-volume path where you want full control and zero surprises | You rebuild retries, tracing, resumability — which is real work |
| **Event-driven loop** | Long-running, async, resumable workflows; work arrives from a queue | Harder to reason about; needs idempotency everywhere |

**The judgement line:** *"I pick a framework when the *topology is uncertain* and I'll iterate on it, and a state machine when the topology is *known and hot*. The decision is about how often the graph changes, not about which is more modern."*

### 3.3 Multi-agent — when and how

> *"I only add a second agent when the sub-tasks need **different tools, different prompts, or different models**. Otherwise it's one agent with more tools, and that's cheaper and easier to debug — more agents means more failure nodes and a harder trace. When I do split them, I keep the handoff explicit: a typed state object, not free-form text passing between agents, and I put the shared state in Redis or a queue with a single writer per key so I don't get lost updates."*

### 3.4 Loop prevention (they will ask)

> *"Loops are a state problem, so I detect them outside the model. Fingerprint each tool call — name plus normalised arguments — and flag repeats. But the better signal is **progress**: if no new information or state change in three turns, it's stuck. Then I force a *different* strategy rather than just cutting it off — feed back 'that query already returned nothing' and force a reformulation or a narrower scope. Plus a hard budget: max steps, max tokens, max cost per task, as a circuit breaker in code. A budget stops the bill, not the loop — the agent still fails, just sooner."*

### 3.5 Memory (four layers, name them)

| Layer | Lives in | Contains |
|---|---|---|
| **Short-term** | context window | current objective, recent tool outputs, intermediate reasoning |
| **Long-term** | Redis / DB / vector store | user preferences, past decisions, account history |
| **Shared** | Redis / queue | cross-agent task progress and intermediate outputs |
| **Episodic** | append-only log | past execution traces, for debugging and few-shot selection |

**Staleness rule:** *"Every memory record carries a timestamp and a confidence; I validate before use and archive low-confidence or outdated entries. Stale memory is worse than no memory, because it's confidently wrong."*

---

## 4. Phase 3 — Concrete Infra Selection & Serving Strategy

**The interviewer will attack:** quantisation, chunking, embedding caching, vLLM concurrency, TTFT vs ITL, hybrid caching.

### 4.1 The serving answer (memorise the metrics vocabulary)

> *"Two different latency numbers and they need different fixes. **TTFT** — time to first token — is retrieval + rerank + prefill, and it's what the user *feels*. **ITL / TPOT** — inter-token latency — is the per-token decode time, and it sets total completion time. For perceived speed I optimise TTFT and stream; for completion I optimise ITL. On serving: vLLM with **continuous batching**, so new requests join the running batch instead of waiting for the slowest request — that's the biggest throughput win. **PagedAttention** to stop the KV cache fragmenting 60–80% of VRAM. **Speculative decoding** with a small draft model gives 2–3× decode speedup at identical output distribution — a latency win, not a quality change. And I'd watch **KV cache**, because a 7B fp16 is about 0.5 MB per 1,000 tokens — at long contexts the KV cache is what runs out, not FLOPS."*

### 4.2 Quantisation

> *"fp16 → int8 → int4 is roughly 14 GB → 7 GB → 3.5 GB for a 7B. fp16 has the lowest loss, int8 is usually safe, and int4 (GPTQ/AWQ) is visibly lossy on complex reasoning — so I'd A/B it on the golden set rather than assume. And a GPU at 30% utilisation costs the same as one at 90% but serves 3× fewer requests, so batching and speculative decoding are how you actually reduce the GPU bill without changing hardware."*

### 4.3 Chunking (their favourite depth probe)

| Strategy | Numbers | Use when |
|---|---|---|
| Fixed-size | FAQ 100–300; general 300–500; legal 500–1,000; code 50–200; overlap **10–20%** | Baseline; homogeneous text |
| Structure-aware | split on headers/sections/paragraphs, recursive | Documents with real structure |
| Parent-child | index children **100–200**; send parents **500–1,000** | Long documents, legal/technical |
| Semantic | split where adjacent-sentence cosine drops below ~0.4 (tune 0.3–0.6) | Topic-shifting prose; costs an embedding per sentence |
| Propositions | atomic facts, 2–5 per 100-token passage | Legal/finance/medical, high stakes |

**The trade-off line:** *"Small chunks give precise embeddings but starve the model of context; large chunks give good context but noisy embeddings. Parent-child is the standard fix: index the small one, retrieve it, send its parent."*

### 4.4 Hybrid caching (the "three cache layers" answer)

> *"Three caches at different levels. **Exact cache** on the normalised query — cheapest, safest. **Semantic cache** on the embedding, cosine ≥ 0.95, for paraphrases — 20–40% hit rate on repeat-heavy traffic, but it must be keyed on **context** (tenant, date, product version), not just the text, and invalidated on the write path, otherwise you serve stale answers confidently. And **prompt/prefix caching** at the provider for the shared system prompt — cache reads at ~10% of input price, though one changed token invalidates it. Plus **embedding caching** at ingestion so I never re-embed an unchanged chunk — hash the content and re-embed only what changed."*

---

## 5. Phase 4 — Production Fail-safes, Drift Monitoring & Evaluation

**The interviewer will attack:** guardrails, LLM-as-judge cost, drift, what runs in CI.

### 5.1 The eval answer (full version in file `07`; compressed for here)

> *"Three layers. Component — retrieval and generation measured separately, so I know *which* stage failed. Trajectory — for the agent, tool selection, arguments, order, step count and recovery, because a right answer by an unsafe path is still a failure. Online — task completion, escalation rate, human-override rate, grounded-answer rate, cost per resolved task. And a frozen golden set of ~120 expert-labelled cases in CI gating every merge."*

### 5.2 Guardrails (deterministic first)

> *"The model proposes, the validator disposes. Structured outputs with a Pydantic schema, server-side validation before anything is acted on, an entailment check against retrieved context for grounding, and a human gate on anything irreversible. **Llama-Guard or a content classifier handles the safety class of problem — harmful content, PII — but it is not the control that stops a bad refund.** For that, the control is a policy check in code plus a human approval. A prompt is a suggestion; permissions are enforcement."*

### 5.3 LLM-as-judge on a budget (they asked about *cost-efficient* judging)

> *"Three cost rules. One — don't spend a judge where a deterministic check works: if a schema or a unit test can verify it, use that. Two — sample rather than judge everything: judge a rolling sample of production plus 100% of the golden set. Three — cache judge verdicts keyed on the input hash so a re-run is free. And calibrate the judge against a few hundred human labels first, because a judge with a 10% false-positive rate reports a true 85% as about 80% — usable for direction, not for an absolute number."*

### 5.4 Drift

> *"Four drift types: corpus (documents change — fix with incremental re-index by content hash), query (users ask new things — cluster production queries and grow the eval set), model (provider updates with no code change — pin versions, run the golden set on every release before it becomes default, track behavioural metrics like refusal rate and length), and embedding (index and query encoders diverge — version the embedding model and re-index on upgrade). The provider one is the nastiest because nothing breaks — the answers just quietly get worse."*

### 5.5 Fail-safes

| Failure | Response |
|---|---|
| Vector DB down | Degrade to keyword-only BM25, or to a template answer, or to human queue — never a 500 to the customer |
| Model returns invalid JSON | Validator catches it, one retry with the error appended, then human review |
| Timeout mid-workflow | Resume from the checkpoint, don't restart (prevents duplicate side effects) |
| Provider 429 | Backoff with jitter, retry budget, then shed load with a clear message |
| Duplicate request | Idempotency key on the intent returns the cached result |
| Model unavailable entirely | Manual path — a regulated service must degrade, not stop |

---

## 6. Scoring your own mock (after each run)

Rate 1–5 and re-run the weak ones:

| # | Criterion | 1 | 5 |
|---|---|---|---|
| 1 | Did I clarify before drawing? | Drew immediately | Asked one question per dial |
| 2 | Did I use numbers? | None | QPS, tokens, cost, latency decomposition |
| 3 | Did I justify every component? | Listed tools | Every box tied to a constraint |
| 4 | Did I name the trade-off? | No | "X over Y because Z, and the cost is W" |
| 5 | Did I handle the human? | Forgot | Named the gate and the audit trail |
| 6 | Did I self-critique? | Claimed complete | Named what I left out |
| 7 | Did I challenge a bad constraint? | Accepted it | "That SLA can't hold for generation — here's the split" |
| 8 | Filler words / rambling | Yes | None; paused instead |

**Target:** ≥ 4 on 1, 2, 3, 4 and 6 before you stop rehearsing.

---

## 7. Cheat-sheet summary

- **Paste §1, answer out loud, and let it surprise you.** Add Variant B to force the Stage-1 four-part shape.
- **Phase 1:** one question per dial; **challenge the 200 ms SLA with arithmetic**; pre-commit the human gate.
- **Phase 2:** use the escalation ladder *before* defending an agent; framework when the topology changes, state machine when it's hot; loops are a state problem.
- **Phase 3:** TTFT vs ITL; continuous batching + PagedAttention + speculative decoding; quantisation A/B'd; parent-child chunking; three cache layers.
- **Phase 4:** three eval layers + CI gate; deterministic validators before LLM judges; four drift types; degrade, never 500.
- **Score yourself on eight criteria.** ≥4 on five of them before you stop.

## 8. Cross-links

- The material to feed it: [`05-system-design-round-playbook.md`](05-system-design-round-playbook.md) · [`07-agent-evaluation-and-guardrails.md`](07-agent-evaluation-and-guardrails.md)
- How to run the mock as a drill: [`09-mock-interview-strategies-and-drills.md`](09-mock-interview-strategies-and-drills.md)
- Question bank for filler questions: [`10-question-bank-with-model-answers.md`](10-question-bank-with-model-answers.md)
