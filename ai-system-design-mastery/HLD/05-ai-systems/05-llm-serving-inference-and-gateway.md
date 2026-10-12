# LLM Serving, Inference and Gateway

**TL;DR** — LLM inference has two phases with *opposite* bottlenecks: **prefill** (processing the prompt) is compute-bound and sets **TTFT**; **decode** (generating tokens) is memory-bandwidth-bound and sets **TPOT/ITL** and throughput. Almost every serving optimization targets one phase or the other — FlashAttention and FP8 for prefill, quantization/GQA/speculative decoding for decode, and **continuous batching plus paged attention** for throughput across both. Capacity is planned in **tokens per second**, not requests per second, and GPU memory is dominated by the **KV cache**, not the weights. In front of the model sits the **LLM gateway**: a single egress point that owns routing, multi-provider fallback, retries with idempotency, caching, budget enforcement, key management, and observability. The gateway is where the reliability and FinOps requirements of the whole system actually get implemented, which is why it is a first-class architectural component rather than a library call.

> **60-second interview answer:** "Inference splits into prefill, which is compute-bound and determines TTFT — target under 200 ms, optimized with FlashAttention, FP8, and prefix caching — and decode, which is memory-bandwidth-bound and determines TPOT, target under 30 ms per token, optimized with 4-bit quantization, GQA, and speculative decoding. Throughput comes from continuous batching and paged attention, which keep the GPU busy by admitting new requests every step and paging the KV cache instead of requiring contiguous memory. Capacity planning is in tokens per second: I compute the KV cache size with `2 × layers × heads × head_dim × seq_len × batch × bytes`, subtract it from GPU memory after weights, and size the fleet from that — not from nominal GPU count. Then the LLM gateway does the operational work: routing by query complexity, provider fallback, retries with idempotency keys, semantic and prompt caching, hard budget caps, and centralized key management, so no service ever holds a provider key or implements failover itself."

---

## 1. Definition & mental model

**Plain-English definition.** *LLM serving* is the discipline of running a model efficiently enough to meet latency and cost targets under concurrent load — batching, memory management, quantization, and scheduling. *An LLM gateway* is the control plane in front of one or more model endpoints that centralizes routing, fallback, caching, budgets, keys, and telemetry.

**The analogy.** Serving is a **restaurant kitchen**. Prefill is reading the whole order ticket and gathering ingredients — the more items on the ticket, the longer it takes, but the work is parallel and it is all *thinking*. Decode is plating and serving one dish at a time, in order — it is fast per dish but strictly sequential, and the limiting factor is how quickly you can move things (memory bandwidth), not how hard you can think. Continuous batching is the maître d' seating new tables the moment a table finishes instead of waiting for a whole seating to clear. Paged attention is not requiring each table to have one giant contiguous tablecloth.

The **gateway** is the **reception desk and switchboard**: it decides which kitchen a ticket goes to, redirects to the backup kitchen when the first is on fire, has a stack of pre-made dishes for the orders that repeat, enforces the nightly budget, and holds all the kitchen's vendor accounts so individual waiters never carry them.

**The mental model that governs every optimization decision:**

| Phase | Bound by | Determines | Optimize with |
|---|---|---|---|
| **Prefill** | Compute (FLOPs) | **TTFT** | FlashAttention, FP8/FP16, tensor parallelism, prefix/prompt caching |
| **Decode** | Memory bandwidth | **TPOT / ITL**, throughput | Quantization, GQA/MQA, batching, speculative decoding |
| **Scheduling** | Both | Goodput, cost/token | Continuous batching, paged attention, chunked prefill |

**The "memory wall" insight worth naming:** as models grew, compute (TFLOPS) scaled faster than memory bandwidth (HBM). That is why decode — which must load the entire weight matrix from VRAM to produce a single token — is the production bottleneck, and why 4-bit quantization (which quarters the bytes moved) is such a large win.

**The one-sentence framing:** *Prefill is a math problem, decode is a moving problem, and the gateway is the operational problem.*

---

## 2. Why it exists — the problem it solves

### What breaks without a serving layer

| Naive approach | Symptom |
|---|---|
| One request per model call, no batching | GPU utilization 5–15%; cost per token 5–10x optimal |
| Static batching (wait for N requests, process, return all) | Head-of-line blocking; a 5-token request waits for a 2,000-token request; p99 explodes |
| Contiguous KV cache allocation | Memory fragmentation; cannot fit long-context requests even with free memory; OOM |
| FP16 everywhere | Half the throughput of FP8 with no measurable quality gain on modern hardware |
| No prefix caching | Every request with the same system prompt re-prefills 3,000 tokens |
| Direct provider SDK calls from every service | Provider keys scattered across services; no central budget; inconsistent retry behaviour; every team re-implements failover badly |
| No gateway | A provider outage takes down the product because there is no fallback and no one owns it |
| No token accounting | Cost is unknown until the invoice; no per-tenant chargeback; no anomaly detection |

### What each serving technique buys

| Technique | Effect | Typical gain |
|---|---|---|
| **Continuous / dynamic batching** | New requests admitted every decode step | 5–20x throughput vs. no batching |
| **Paged attention** | KV cache in fixed-size pages, non-contiguous | Near-zero fragmentation; larger effective batch; 2–4x throughput |
| **Chunked prefill** | Split long prefills into chunks interleaved with decode | Smoother TTFT under mixed load; fewer stalls |
| **Prefix / prompt caching** | Reuse KV for a shared prefix | ~90% cost reduction and large TTFT reduction on repeated prefixes |
| **FP8** | 2x throughput vs FP16, < 0.1% quality loss on H100/B200-class | 2x throughput |
| **INT4/INT8 weight quantization** | 4x / 2x less memory traffic on decode | ~2–4x decode throughput; enables bigger models per GPU |
| **GQA / MQA** | Fewer KV heads → smaller KV cache | 4–8x KV memory reduction |
| **Speculative decoding** | Draft model proposes N tokens, target verifies | 1.5–3x decode speedup |
| **Tensor parallelism** | Split each layer across GPUs | Reduces per-GPU memory; needed for large models |
| **Prefill/decode disaggregation** | Separate pools for compute-bound and memory-bound work | Better utilization for mixed workloads |

### Why the gateway specifically exists

It solves five problems that otherwise get solved five different ways:

1. **Provider concentration risk.** Any single provider will have an outage. Fallback is a system property, not a per-call try/except.
2. **Cost control.** Budgets, caps, and per-tenant attribution require a central chokepoint.
3. **Key management.** Provider keys in every service is a security incident waiting to happen. One place, one rotation policy, one audit trail.
4. **Consistency.** Retry policy, timeout policy, caching policy, and observability should be identical for every caller.
5. **Change velocity.** Swapping a model or adding a provider should be a config change at the gateway, not a deployment of ten services.

---

## 3. Internal mechanics — how it actually works

### 3.1 Where things sit

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            CLIENTS                                       │
│         (chat UI, RAG service, agent runtime, batch jobs)                │
└───────────────────────────────┬──────────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        LLM GATEWAY                                       │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Auth + key   │  │ Rate limit + │  │ Cache:       │  │ Observability│ │
│  │ management   │  │ budget caps  │  │ exact/semantic│  │ traces, cost │ │
│  │ (no keys in  │  │ (per tenant) │  │ /prompt      │  │ tokens, TTFT │ │
│  │  services)   │  │              │  │              │  │              │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  ROUTER — classify, then pick a target                             │ │
│  │   simple → small model   complex → frontier   long → long-context  │ │
│  │   code → code model      vision → multimodal                       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  RELIABILITY — timeout → retry(w/ jitter + idempotency key) →      │ │
│  │   circuit breaker → provider failover → degradation ladder         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└───────┬───────────────────────┬──────────────────────────┬───────────────┘
        ▼                       ▼                          ▼
┌───────────────┐      ┌────────────────┐         ┌────────────────────┐
│ API provider A│      │ API provider B │         │ SELF-HOSTED POOL   │
│ (frontier)    │      │ (secondary)    │         │ vLLM / TGI / TRT-LLM│
└───────────────┘      └────────────────┘         │  ┌──────────────┐  │
                                                  │  │ Scheduler    │  │
                                                  │  │ (continuous  │  │
                                                  │  │  batching)   │  │
                                                  │  └──────┬───────┘  │
                                                  │         ▼          │
                                                  │  ┌──────────────┐  │
                                                  │  │ KV cache     │  │
                                                  │  │ (paged)      │  │
                                                  │  └──────────────┘  │
                                                  └────────────────────┘
```

### 3.2 The two phases, mechanically

**Prefill.** The model processes all input tokens in a single forward pass. Every token attends to every previous token, so the work is a set of large, highly parallel matrix multiplications. Time complexity is O(N) in prompt length but heavily parallelized — so the wall-clock cost is *linear but with a small constant* when compute is the limit, and it is dominated by FLOPs. Optimizations: FlashAttention (avoids materializing the full attention matrix in HBM), FP8 (2x the FLOPs per second on supported hardware), tensor parallelism (split the matmul across GPUs), and prefix caching (skip prefill entirely for a known prefix — the single biggest win when prompts share a long system preamble).

**Decode.** Tokens are generated one at a time and each depends on all previous ones, so it is strictly sequential. For each token the model must load the entire weight matrix from VRAM, plus the growing KV cache. The arithmetic intensity is tiny — gigabytes of weights moved to produce a few bytes of output — so the limit is memory bandwidth, not FLOPs. Optimizations: quantization (fewer bytes to move), GQA/MQA (smaller KV cache to move), batching (amortize the weight load across many sequences — this is the key insight behind continuous batching), and speculative decoding (verify several tokens per weight load).

**Why batching is the master optimization:** in decode, the weight load is the same whether you are serving one sequence or thirty-two. Batching turns a memory-bound problem into a much better amortized one. That is the entire economic argument for a serving framework.

### 3.3 Batching and scheduling

```
STATIC BATCHING (bad)
  batch of 4 arrives ──▶ process all 4 to completion ──▶ return all 4
  ├─ seq A needs 5 tokens   (finishes in 5 steps, then GPU idles for it)
  ├─ seq B needs 200 tokens
  ├─ seq C needs 40 tokens
  └─ seq D needs 900 tokens  ◀── everyone waits 900 steps

CONTINUOUS / DYNAMIC BATCHING (good)
  step 1: [A, B, C, D]      A finishes after 5 steps
  step 6: [B, C, D, E]      ◀── E admitted immediately; no idle slots
  step 41:[B, D, E, F, G]   C finished; F, G admitted
  step 201:[D, F, G, ...]   B finished; new work keeps arriving
```

Continuous batching is what every production serving engine (vLLM, TGI, TensorRT-LLM) implements by default. It is typically a 5–20x throughput improvement over no batching, and it is the reason GPU utilization above 60% is achievable at all.

**Chunked prefill** solves the interference problem: a 32k-token prefill can stall all in-flight decode for hundreds of milliseconds. Chunked prefill splits it into smaller pieces and interleaves them with decode steps, keeping TTFT for the long request reasonable without destroying TPOT for everyone else.

### 3.4 Paged attention and the KV cache

The KV cache is the memory of everything the model has attended to. Its size is:

```
KV_bytes = 2                       # one for keys, one for values
         × n_layers
         × n_kv_heads
         × head_dim
         × seq_len
         × batch_size
         × bytes_per_element       # 2 for FP16/BF16, 1 for FP8/INT8
```

**Worked example — Llama-3-70B-class, FP16:**

```
n_layers      = 80
n_kv_heads    = 8      (GQA with 64 attention heads → 8 KV heads)
head_dim      = 128
bytes         = 2      (FP16)

KV per token  = 2 × 80 × 8 × 128 × 2 = 327,680 bytes ≈ 0.31 MB/token
```

| Context length | KV per sequence | Batch 1 | Batch 16 | Batch 64 |
|---|---|---|---|---|
| 4k tokens | 1.25 GB | 1.25 GB | 20 GB | 80 GB |
| 8k tokens | 2.50 GB | 2.50 GB | 40 GB | 160 GB |
| 32k tokens | 10.0 GB | 10 GB | 160 GB | 640 GB |
| 128k tokens | 40.0 GB | 40 GB | 640 GB | — |

**Read that table carefully — it is the single most useful piece of intuition in LLM serving.** Weights for a 70B model in FP16 are ~140 GB; at 8k context and batch 16 the KV cache is 40 GB, i.e. ~29% overhead. At 32k context and batch 16 it is 160 GB — **larger than the weights**. Long context is a memory problem before it is a compute problem.

**Without GQA** (64 KV heads instead of 8) every number in that table multiplies by 8. Batch 16 at 8k context would need 320 GB of KV cache. That is why GQA exists.

**Paged attention** (vLLM's core contribution) stores the KV cache in fixed-size blocks with a block table mapping logical to physical positions — essentially virtual memory for the KV cache. Consequences: near-zero internal fragmentation (you allocate in 16-token pages, not whole sequences), memory sharing across sequences with a common prefix (copy-on-write), and much larger effective batch sizes.

**FP8/INT8 KV cache** halves or quarters every number above at a small quality cost, and is the go-to lever when KV memory is the binding constraint.

### 3.5 Quantization

| Format | Bits | Memory vs FP16 | Quality impact | Notes |
|---|---|---|---|---|
| FP16 / BF16 | 16 | 1x | baseline | Reference |
| FP8 | 8 | 2x smaller | < 0.1% on H100/B200 | Native hardware support; **the default for modern serving** |
| INT8 | 8 | 2x | ~0.5–2% | Needs calibration |
| INT4 (AWQ / GPTQ) | 4 | 4x | 1–3% typical | Big decode win; needs per-task eval |
| INT2–3 | 2–3 | 8x | Large, task-dependent | Research; rarely production |

**Principal-level nuance to mention:** modern serving stacks use **dynamic FP8 scaling**, adjusting quantization scales per layer so activation outliers do not degrade the whole model. Naive per-tensor scaling is where most FP8 quality loss comes from.

**The rule:** quantization decisions are *empirical*. Measure on your task, not on a leaderboard. A 2% perplexity increase can be a 10-point drop on a structured-extraction task.

### 3.6 Speculative decoding

A small draft model proposes the next N tokens; the target model verifies them in a single forward pass and accepts the longest correct prefix.

```
Draft model proposes:  ["the", "refund", "window", "is", "thirty"]   (5 tokens, cheap)
Target verifies all 5 in one pass: accepts ["the", "refund", "window", "is"]
                                   rejects "thirty", emits correct token
Result: 4 tokens produced for the cost of ~1 target forward pass
```

Typical speedup 1.5–3x on decode with **identical output distribution** (it is exact, not approximate), provided the draft model is well-matched. It helps most when the target model is large and the output is predictable (summarization, code completion, structured output) and helps least on highly creative or high-entropy generation.

### 3.7 The metrics that define the service

| Metric | Full name | What it controls | Target |
|---|---|---|---|
| **TTFT** | Time To First Token | Perceived responsiveness | **< 200 ms** (chat), < 500 ms acceptable |
| **TPOT / ITL** | Time Per Output Token / Inter-Token Latency | Reading speed, conversational flow | **< 30 ms** (~33 tok/s, faster than reading) |
| **E2E latency** | Total request time | Task completion | < 2 s for a single agent turn |
| **Throughput** | Aggregate tokens/sec | Cost per token | Maximize |
| **Goodput** | Tokens/sec meeting the SLO | The metric that matters | Maximize under SLO |
| **GPU utilization** | Compute busy fraction | Efficiency | > 60%; > 80% excellent |
| **KV cache utilization** | Fraction of KV memory used | Headroom | 70–90%; > 95% → preemption risk |

**The decomposition that interviewers want:**

```
E2E latency ≈ TTFT + (output_tokens − 1) × TPOT
```

A 300-token answer at TTFT 200 ms and TPOT 25 ms = 0.2 + 299 × 0.025 = **7.7 seconds**. That is why answer length is a latency lever: capping output at 150 tokens instead of 300 nearly halves the wall-clock time. **`max_output_tokens` is a latency control, not just a cost control.**

### 3.8 The LLM gateway, component by component

**1. Authentication and key management.** Callers authenticate to the gateway with an internal token; the gateway holds provider credentials. Rotation, scoping, and revocation happen in one place. No service ever embeds a provider key.

**2. Routing.** A classifier (or cheap heuristics) picks a target:

```python
class ModelRouter:
    def __init__(self):
        self.models = {
            "simple":       SmallModel(),        # classification, extraction, short answers
            "complex":      FrontierModel(),     # reasoning, synthesis
            "code":         CodeModel(),
            "long_context": LongContextModel(),  # > 100k tokens
            "vision":       MultimodalModel(),
        }

    async def route(self, request):
        request_type = self.classifier.classify(request)   # ~20-50 ms
        try:
            return await self.models[request_type].generate(request)
        except RateLimitError:
            return await self.fallback(request, request_type)
```

Routing signals: query length, presence of code, required output schema complexity, latency SLO of the caller, tenant tier, and a small classifier's judgment. Typical outcome: 50–70% of traffic served by a small model at 1/10 the cost.

**3. Caching.** Four distinct caches, each with its own invalidation rule:

| Cache | Key | Savings | Invalidation |
|---|---|---|---|
| Exact response | hash(normalized prompt + params) | ~100% on hit | TTL / explicit |
| Semantic response | embedding similarity ≥ 0.95 | ~68% cost on hit | TTL + index version |
| Prompt / prefix (provider-side) | stable prefix hash | ~90% on cached input tokens | Provider-managed TTL |
| Provider prompt cache for retrieval context | shared document prefix | high on repeated context | Document version |

**4. Budget and quota enforcement.** Per request, per user, per tenant, per day. Enforced as *degradation* before *failure*: at 80% of budget, route to a smaller model; at 100%, queue non-critical work or return a degraded response; only at a hard cap do you reject.

**5. Reliability chain.** In order: connection timeout (5 s) → total timeout (30–60 s) → retry with exponential backoff and jitter (max 3 attempts, only on retryable errors) → circuit breaker per provider (5 failures → open for 30 s → half-open for 3 probes → 2 successes → closed) → provider failover → degradation ladder (big model → small model → cached answer → static fallback).

```python
class RetryConfig:
    max_retries = 3
    base_delay = 1.0
    max_delay = 60.0
    exponential_base = 2.0
    jitter = 0.5          # ±50% to prevent thundering herd

    def get_delay(self, attempt):
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        jitter_range = delay * self.jitter
        return max(0, delay + random.uniform(-jitter_range, jitter_range))
```

**Retryable:** rate limit (429), timeout, 503, connection error, and 5xx generally.
**Not retryable:** 401/403 auth, 400 bad request, context length exceeded, content policy violation. Retrying these wastes budget and latency for a guaranteed failure.

**6. Idempotency.** Every gateway request carries an idempotency key. On retry, the same key returns the cached result rather than re-invoking the model. This is what makes retries safe for non-idempotent operations, and it matters enormously when the model call is part of a workflow with side effects.

**7. Observability.** Per-request record: model + version, input/output tokens, TTFT, TPOT, total latency, cache hit/miss, provider, retry count, cost, and the final status. Aggregated by tenant, route, and model for dashboards and chargeback.

### 3.9 Provider failover and hedging

```python
class ProviderManager:
    def __init__(self):
        self.providers = {"primary": ..., "secondary": ..., "tertiary": ...}
        self.health = {name: True for name in self.providers}

    async def generate(self, request):
        for name in self.priority_order:
            if not self.health[name]:
                continue
            try:
                return await self.providers[name].generate(request)
            except RetryableError:
                self.health[name] = False
                asyncio.create_task(self._recheck_later(name))
                continue
        raise AllProvidersUnavailableError()
```

**Hedging** is the latency-focused variant: send to the primary, and if it has not responded within `hedge_delay` (typically p95 of normal latency, ~2 s), fire the same request at the secondary and take whichever returns first. It costs money (you may pay for two calls) and should be reserved for latency-critical, low-volume paths.

**Cross-provider caveats worth stating:** providers differ in tokenizers (token counts are not comparable), in structured-output support, in system-prompt handling, and in safety behaviour. A fallback provider is a *degraded* mode and must be evaluated as such — run the golden set against the fallback path and record its quality separately.

---

## 4. Variants & taxonomy

### 4.1 Self-hosted serving engines

| Engine | Complexity | Performance | Ecosystem | Best for |
|---|---|---|---|---|
| **vLLM** | Medium | Excellent | Broadest; PagedAttention originator | **Default for production self-hosting** |
| **TGI (HuggingFace)** | Medium | Very good | HuggingFace-native | Teams already in the HF ecosystem |
| **TensorRT-LLM** | High | Best on NVIDIA | NVIDIA-only, CUDA-locked | Maximum performance, NVIDIA committed |
| **SGLang** | Medium | Excellent | Strong on structured output + RadixAttention | Heavy prefix sharing, structured generation |
| **Ollama** | Low | Good | Simple, local | Development, demos, single-user, edge |
| **llama.cpp** | Low | Good | CPU/edge, GGUF | CPU inference, laptops, embedded |

### 4.2 Deployment model comparison

| Factor | API providers | Self-hosted |
|---|---|---|
| Setup time | Minutes | Days to weeks |
| Operational burden | None | Significant (1–2 dedicated engineers) |
| Cost at low volume | Lower | Higher (fixed GPU cost) |
| Cost at high volume | Higher | Lower (scale economics) |
| Latency control | Limited | Full |
| Data privacy | Data leaves your infra | Data stays |
| Model choice | Provider's catalogue | Any open weights |
| Fine-tuning control | Via API | Full |
| Latest models | Immediate | Weeks-to-months behind |
| Elasticity | Instant | Slow scale-out; cold start |

**Rough crossover:** API is usually right below ~1M queries/month; self-hosting becomes compelling above ~10M queries/month, or immediately when data residency, sub-100 ms p99, or custom weights are hard requirements. Most mature teams run **hybrid**: self-host the high-volume predictable workload, use APIs for spikes and specialised models, and keep an API as the fallback for self-hosted failures.

### 4.3 GPU pool sizing

| Scale | GPUs | Typical setup |
|---|---|---|
| Dev/test | 1 | Single A10G / L4 / consumer GPU |
| Small prod | 2–4 | 2× A100/H100 with tensor parallelism |
| Medium prod | 4–8 | 4× H100 tensor-parallel, multiple replicas |
| Large prod | 8+ | Multi-node with pipeline + tensor parallelism |

### 4.4 Gateway deployment patterns

| Pattern | Description | When |
|---|---|---|
| **Sidecar / library** | Gateway logic in-process in each service | Low latency, no extra hop; but policy is duplicated |
| **Central service** | One gateway deployment, all traffic through it | **Default.** One policy, one key store, one budget |
| **Mesh / ingress-integrated** | Gateway as an ingress or service-mesh filter | Platform teams with existing mesh |
| **Two-tier** | Regional gateways behind a global one | Multi-region, data residency |

### 4.5 Inference optimization techniques by target

| Target | Techniques |
|---|---|
| **Reduce TTFT** | Prefix caching, FlashAttention-3, FP8, chunked prefill, tensor parallelism, prefill/decode disaggregation |
| **Reduce TPOT** | INT4/FP8 weights, GQA/MQA, speculative decoding, larger batches |
| **Increase throughput** | Continuous batching, paged attention, larger batch, quantization |
| **Reduce memory** | Quantization, GQA, FP8 KV cache, paged attention, KV eviction/offload |
| **Reduce cost** | All of the above + routing + caching + batch API for async work |

---

## 5. Pros / Cons

| Pros | Cons |
|---|---|
| Continuous batching turns a memory-bound problem into an amortized one — 5–20x throughput | Adds a scheduler to debug; per-request behaviour is no longer trivially reproducible |
| Paged attention eliminates KV fragmentation and enables prefix sharing | Framework-specific; migrating engines means re-tuning and re-validating |
| FP8 gives 2x throughput for < 0.1% quality loss on modern hardware | Requires modern GPUs; older hardware gains much less |
| Quantization to INT4 cuts decode cost 2–4x | Quality loss is task-dependent and must be measured, not assumed |
| Speculative decoding is exact — same output distribution as the target model | Needs a well-matched draft model; helps least on high-entropy generation |
| A gateway centralizes routing, keys, budgets, and retries | Extra network hop (1–10 ms) and a potential single point of failure — must itself be HA |
| Routing sends 50–70% of traffic to a small model at 1/10 cost | The router is a model with its own error rate; misrouting is silent quality loss |
| Provider fallback removes single-vendor risk | Fallback providers have different tokenizers, safety behaviour, and quality — it is degraded mode, not equivalent |
| Gateway-level observability makes per-tenant chargeback possible | Centralizing all prompts in one place is also a concentration of sensitive data |
| Self-hosting gives full control over latency, versions, and data residency | 1–2 dedicated engineers, slow scale-out, cold start, and a model-update treadmill |
| Prompt caching can cut input cost ~90% | Cache TTL is provider-controlled and can change; correctness depends on prefix stability |

---

## 6. Exceptions, edge cases & failure modes

### 6.1 Latency cliffs from prefill interference

**Symptom:** p50 TTFT is 150 ms, p99 is 4 seconds. Under mixed load, a few long prompts stall everything.
**Cause:** a 32k-token prefill monopolizes the GPU for hundreds of milliseconds while decode requests wait.
**Mitigation:** enable **chunked prefill**; cap max input tokens per request at the gateway; route long-context requests to a separate pool or a separate provider; admit long requests with a concurrency limit (a bulkhead) so they cannot starve interactive traffic.

### 6.2 KV cache exhaustion and preemption

**Symptom:** requests are evicted and recomputed mid-generation; TTFT for affected requests spikes; throughput degrades non-linearly.
**Cause:** KV cache utilization above ~95% forces the scheduler to preempt sequences.
**Mitigation:** monitor `kv_cache_utilization` and alert at 90%; cap max context and max batch; enable FP8 KV cache to double capacity; add replicas before saturation; reject or queue rather than thrash. **Rejecting cleanly is better than degrading everything.**

### 6.3 Retry storm / thundering herd

**Symptom:** a provider has a 5-second blip and the fleet doubles its request rate for a minute, turning a blip into an outage.
**Mitigation:** exponential backoff **with jitter** (never fixed-interval retries); a retry budget (cap total retries at ~10% of request volume over a window); circuit breakers so a failing provider stops receiving traffic entirely; idempotency keys so retries do not duplicate work.

### 6.4 Retrying non-retryable errors

**Symptom:** context-length-exceeded errors retry 3 times, tripling cost and latency for a request that can never succeed.
**Mitigation:** explicit retryable/non-retryable classification. Retryable: 429, timeout, 503, connection errors, 5xx. Non-retryable: 401/403, 400, context length exceeded, content policy.

### 6.5 Cache poisoning and stale answers

**Symptom:** a user gets an answer citing a policy that was superseded three weeks ago, because the semantic cache returned a cached response.
**Mitigation:** include the **index version** in every cache key so a reindex invalidates everything; include the model version; use short TTLs for anything backed by mutable data (15 min–24 h); never cache responses that depend on user-specific ACLs under a global key — key by tenant at minimum; provide an explicit cache-bypass flag for compliance queries.

### 6.6 Semantic cache false hits

**Symptom:** the cache returns an answer to a subtly different question because the embeddings were 0.96 similar. "What's the refund window in the EU?" returns the US answer.
**Mitigation:** a high threshold (0.95) plus a cheap verification step — either a small model that confirms equivalence, or entity/number matching that rejects a cache hit if the numbers or named entities differ. **Numbers and negation are where semantic caches fail**, so check them explicitly.

### 6.7 Provider tokenizer mismatch breaking cost and context math

**Symptom:** the fallback provider truncates requests because the token count was computed with the primary provider's tokenizer.
**Mitigation:** compute token counts per provider, not globally; keep a per-provider context limit and a safety margin (use at most 80% of the advertised window); test the fallback path with the golden set and record its quality separately.

### 6.8 Model version drift on the provider side

**Symptom:** quality changes overnight with no deployment on your side. Cost per query changes because tokenization changed.
**Mitigation:** pin model versions explicitly where the provider allows it (dated snapshots); record the model version in every trace; run the golden set nightly so a provider-side change is detected within 24 hours; keep a small self-hosted model as an escape hatch so you are never fully exposed to a provider's release cadence.

### 6.9 Gateway as a single point of failure

**Symptom:** the gateway goes down and every AI feature in the company goes down with it.
**Mitigation:** run the gateway as a horizontally scaled stateless service behind a load balancer; keep cache and budget state in Redis with its own HA; **define a fail-open path** for the gateway itself (e.g. a read-only mode allowing direct provider calls with a shared key) with a documented, tested procedure; make the gateway's own SLO explicit and monitor it as a tier-0 service.

### 6.10 Cost explosion from unbounded output and context

**Symptom:** a single endpoint's traffic drives a 5x bill increase. The cause is usually an unbounded `top_k` in a RAG pipeline or a missing `max_output_tokens`.
**Mitigation:** hard caps at the gateway — max input tokens, max output tokens, max cost per request, max requests per tenant per minute. Alert on cost-per-request percentiles, not just totals, so a per-request regression is visible even when volume is flat.

### 6.11 Cold start and scale-out latency

**Symptom:** autoscaling adds a replica and it takes 90 seconds to become ready, during which the queue keeps growing.
**Mitigation:** keep a warm pool sized for the p95 ramp rather than the average; pre-load weights onto fast local storage or bake them into the image; measure and monitor `model_load_time` as a first-class metric; prefer scale-out on **queue depth** with lead time over reactive CPU-based scaling; consider a small always-on tier that absorbs the ramp.

### 6.12 Silent quality regression from quantization

**Symptom:** a 4-bit quantization was adopted for a 3x throughput win and the structured-output success rate dropped from 99% to 94%. Nobody noticed because nobody measured.
**Mitigation:** run the task golden set against the quantized model before adopting it, not a perplexity benchmark; compare on structured-output validity, JSON conformance, and the domain metrics; keep the FP8/FP16 path available so you can roll back.

### 6.13 Multi-provider behavioural divergence

**Symptom:** the fallback path technically works but produces answers in a different format, ignores the system prompt's style rules, or refuses things the primary handles.
**Mitigation:** treat the fallback as a separate system with its own prompt variant and its own eval results; run the golden set against every provider; alert when the fraction of traffic served by fallback exceeds a threshold (e.g. > 5%) because that is a quality incident even though error rates are zero.

### 6.14 Hedging doubling cost

**Symptom:** a hedging configuration sends every request to two providers and the bill doubles.
**Mitigation:** hedge only on latency-critical, low-volume paths; set `hedge_delay` at roughly the p95 of normal latency so only genuinely slow requests hedge; cap the hedge rate as a fraction of traffic; cancel the loser immediately (and confirm the provider cancels billing, which not all do).

---

## 7. When to use / When NOT to use

### Self-host vs. API — the decision table

| Requirement | API | Self-host | Hybrid |
|---|---|---|---|
| Time to market urgent | ✅ | ❌ | |
| < 1M queries/month | ✅ | ❌ | |
| > 10M queries/month | ❌ | ✅ | ✅ |
| Data cannot leave your infra | ❌ | ✅ | ✅ |
| p99 latency < 100 ms | ❌ | ✅ | |
| Custom/fine-tuned weights | ⚠️ (limited) | ✅ | ✅ |
| Need the newest frontier model immediately | ✅ | ❌ | ✅ |
| Team has no GPU ops expertise | ✅ | ❌ | |
| Highly variable / spiky workload | ✅ | ❌ | ✅ |
| Predictable high-volume workload | ❌ | ✅ | ✅ |
| Regulatory data residency | ❌ | ✅ | ✅ |

### Per-technique usage

| Technique | Use when | Do NOT use when |
|---|---|---|
| Continuous batching | Always, in any self-hosted engine | Never skip |
| Paged attention | Always in vLLM-class engines | Single-request dev servers |
| Chunked prefill | Mixed short/long workload | Pure uniform short prompts |
| Prefix caching | Long shared system prompts or documents | Every prompt is unique |
| FP8 | H100/B200-class hardware | Older GPUs; validate quality first |
| INT4 quantization | Decode-bound, memory-constrained | Quality-critical structured extraction without eval |
| Speculative decoding | Large target model, predictable output | High-entropy creative generation; small target |
| GQA | Model supports it — it is near-free | Legacy MHA-only checkpoints |
| Tensor parallelism | Model does not fit on one GPU, or TTFT matters | Model fits comfortably; adds communication overhead |
| Prefill/decode disaggregation | Large-scale mixed workload with dedicated SRE | Small deployments; complexity not worth it |
| Gateway routing | > ~10k queries/day and > 1 model | Single-model prototype |
| Provider fallback | Any production system | Never skip in production |
| Hedging | Latency-critical, low volume | High volume (cost); non-urgent work |

### When NOT to build a gateway

- You have one model, one provider, low volume, and no compliance requirement — a shared client library with a common configuration is sufficient.
- You are pre-product-market-fit and the gateway would be the most complex component in the system.
- You already have a service mesh or API gateway with policy plugins and the LLM-specific logic fits as a plugin.

**But even then, centralize key management and token accounting from day one** — retrofitting cost attribution across ten services is far more painful than building the gateway early.

---

## 8. Numbers, defaults & tuning

### 8.1 Latency and throughput targets

| Metric | Target | Acceptable | Alarm |
|---|---|---|---|
| TTFT (chat) | < 200 ms | < 500 ms | > 800 ms |
| TTFT (batch/async) | < 2 s | < 5 s | > 10 s |
| TPOT / ITL | < 30 ms (~33 tok/s) | < 50 ms | > 80 ms |
| E2E (single turn) | < 2 s | < 4 s | > 6 s |
| GPU utilization | > 70% | > 50% | < 30% (over-provisioned) |
| KV cache utilization | 70–85% | < 92% | > 95% (preemption) |
| Goodput (tokens/s within SLO) | maximize | — | — |

### 8.2 Timeouts, retries, circuit breakers

| Parameter | Default | Range | Notes |
|---|---|---|---|
| Connection timeout | 5 s | 2–10 s | — |
| Read / total timeout | 30 s | 15–120 s | 60 s for reasoning models |
| Max retries | 3 | 1–5 | Only retryable errors |
| Base delay | 1 s | 0.5–2 s | — |
| Max delay | 60 s | 30–120 s | — |
| Jitter | ±50% | 0–100% | Always non-zero in a fleet |
| Circuit breaker failure threshold | 5 | 3–10 | Per provider |
| Circuit recovery timeout | 30 s | 10–60 s | — |
| Half-open probe count | 3 | 1–5 | — |
| Successes to close | 2 | 2–5 | — |
| Retry budget | 10% of requests | 5–20% | Prevents retry amplification |
| Hedge delay | p95 of normal latency (~2 s) | 1–5 s | Latency-critical paths only |

### 8.3 Batching and scheduling defaults

| Parameter | Default | Notes |
|---|---|---|
| Max batch size (tokens) | 2,048–8,192 | Bounded by KV memory, not by a fixed number of requests |
| Max concurrent sequences | 128–256 | Engine-dependent |
| Max model length | 8k–32k | Cap it; do not expose the model's full window by default |
| KV cache block size | 16 tokens | vLLM default |
| Prefix cache hit target | > 40% for RAG with a fixed system prompt | — |
| Chunked prefill chunk size | 512–2,048 tokens | Tune for TTFT smoothness |

### 8.4 GPU memory budgeting — the worked example interviewers ask for

**Task:** serve a 70B model (FP16 weights) with 8k context, batch 16, on H100 80GB.

```
Weights (70B × 2 bytes)                     = 140 GB   ◀── already > 1 GPU
```

So the model is split across GPUs. On **2× H100 80GB = 160 GB total**:

```
Weights (tensor-parallel across 2 GPUs)     = 140 GB   (70 GB per GPU)
KV cache @ 8k, batch 16, FP16               =  40 GB   (20 GB per GPU)
Activations / workspace                     =  ~6 GB
--------------------------------------------------------------
Total                                       = 186 GB   ✗ does not fit
```

**Levers, in the order you should apply them:**

| Lever | New KV | New total | Fits in 160 GB? |
|---|---|---|---|
| Baseline FP16 | 40 GB | 186 GB | ✗ |
| Quantize weights to FP8 | 40 GB | 116 GB | ✓ (44 GB headroom) |
| …or quantize KV to FP8 | 20 GB | 166 GB | ✗ (close) |
| …or reduce batch to 8 | 20 GB | 166 GB | ✗ |
| …or reduce max context to 4k | 20 GB | 166 GB | ✗ |
| FP8 weights **and** FP8 KV | 20 GB | 96 GB | ✓ (64 GB headroom) |
| FP8 weights + 3× H100 | 40 GB | 116 GB / 240 GB | ✓ comfortable |

**The interview point:** the answer is not "add more GPUs." It is a sequence — quantize weights first (cheapest, biggest win, smallest quality risk), then quantize KV, then reduce batch or context, and only then add hardware. And you must state the quality caveat: FP8 weights and KV quantization each need a task-level eval before adoption.

**Rule of thumb for capacity:** `usable_KV = total_GPU_memory − weights − activations − framework_overhead(~10–15%)`, then `max_batch × context ≤ usable_KV / KV_per_token`.

### 8.5 Capacity planning in tokens per second

Requests per second is the wrong unit. Tokens per second is the right one, because cost and GPU time scale with tokens.

```
tokens_per_request = input_tokens + output_tokens
required_tokens_per_sec = peak_QPS × tokens_per_request
GPUs_needed = required_tokens_per_sec / measured_tokens_per_sec_per_GPU
```

**Worked example.** 1M queries/day → 11.6 QPS average → 35 QPS at 3× peak. Average request: 3,000 input + 300 output = 3,300 tokens.

```
required = 35 × 3,300 = 115,500 tokens/sec
```

A well-tuned H100 running a 7B model in FP8 with continuous batching can sustain on the order of 3,000–8,000 output tokens/sec aggregate (prompt-processing throughput is far higher, often 20,000+ tokens/sec). Using a blended assumption of ~4,000 tokens/sec per GPU:

```
GPUs = 115,500 / 4,000 ≈ 29 GPUs
```

Then apply headroom: **+30–50% for spikes, +1 GPU per replica minimum for HA, and never run a pool at more than ~70% steady-state utilization** because queueing latency is non-linear near saturation (see file 14 on Little's law). That gives roughly **40–45 H100-equivalents**, or the API-provider equivalent.

**Cost sanity check:** at typical H100 rental of $2–4/GPU-hour, 42 GPUs is ~$100–170/hour ≈ **$2,500–4,000/day** — which is roughly the same order as the API cost for the same volume. **Self-hosting is not automatically cheaper.** The break-even is driven by utilization: at 20% utilization, self-hosting loses badly; above ~50% sustained, it usually wins.

### 8.6 Gateway defaults

| Parameter | Default | Notes |
|---|---|---|
| Max input tokens | 32k (per route) | Cap below the model's window; use 80% of advertised |
| Max output tokens | 1,024 | A latency and cost cap |
| Max cost per request | $0.10 | Hard cap; degrade before rejecting |
| Requests per minute per user | 20 | Tiered: free 5 QPS / pro 50 / enterprise 200 |
| Daily budget per tenant | Configurable | Alert at 80%, degrade at 100% |
| Semantic cache threshold | 0.95 cosine | Plus entity/number verification |
| Cache TTL | 15 min–24 h | Shorter for mutable data |
| Gateway SLO | 99.9% availability, < 10 ms added latency | The gateway is tier-0 |
| Route small-model share | 50–70% | The primary cost lever |

### 8.7 Cost reference points

| Model tier | Input $/1M tokens | Output $/1M tokens | Route share |
|---|---|---|---|
| Small (mini/haiku-class) | ~$0.15–0.30 | ~$0.60–1.25 | 50–70% |
| Mid (sonnet-class) | ~$2.50–3.00 | ~$10–15 | 25–45% |
| Frontier (opus-class) | ~$10–15 | ~$40–75 | 5–15% |

**Note the asymmetry: output tokens cost 3–5x input tokens.** That is why `max_output_tokens` is one of the highest-leverage cost controls, and why "be concise" in the system prompt is a FinOps decision, not just a UX one. Prompt caching discounts input by ~90%, and batch APIs discount everything by ~50% for non-interactive work.

---

## 9. Interview Q&A

**Q1. Why is LLM generation slower than classification?**
Classification is prefill-only: it processes the entire input in one parallel pass and produces a single output, so it is compute-optimal. Generation is autoregressive — each token depends on the previous one, forcing a sequential decode loop. Each step in that loop is **memory-bandwidth-bound**: you load gigabytes of weights from VRAM to produce a few bytes of output, so the GPU spends its time waiting on memory transfers rather than doing math. That is the fundamental asymmetry, and it is why quantization and batching help generation so much more than they help classification.

**Q2. Explain prefill vs. decode and what each optimizes.**
Prefill processes the whole prompt in parallel — it is compute-bound, O(N) in prompt length but heavily parallelized, and it sets **TTFT**. Optimize with FlashAttention, FP8, tensor parallelism, chunked prefill, and prefix caching. Decode generates one token at a time sequentially — it is memory-bandwidth-bound and sets **TPOT**. Optimize with quantization, GQA, speculative decoding, and larger batches. The key structural point is that batching amortizes the weight load across sequences, which is why continuous batching is the master throughput optimization.

**Q3. How do you optimize TTFT and TPOT separately?**
For TTFT I attack prefill: FlashAttention-3, FP8 precision, tensor parallelism to get more compute, chunked prefill so long prompts do not stall the queue, and most importantly **prefix caching** — if every request shares a 3,000-token system prompt, caching it eliminates that prefill entirely and drops TTFT dramatically. For TPOT I attack memory bandwidth during decode: 4-bit or FP8 weight quantization reduces the bytes moved per token, GQA reduces the KV cache that must be read, and speculative decoding produces multiple tokens per weight load. And I cap `max_output_tokens`, because end-to-end latency is `TTFT + (output_tokens − 1) × TPOT` — answer length is a latency lever.

**Q4. What is paged attention and what problem does it solve?**
Without it, the KV cache for each sequence must occupy contiguous GPU memory, so you must reserve a worst-case block per sequence and you fragment badly — you can OOM while having plenty of free memory. Paged attention stores the KV cache in fixed-size blocks (vLLM uses 16 tokens) with a block table mapping logical positions to physical ones, exactly like virtual memory in an OS. The benefits are near-zero internal fragmentation, much larger effective batch sizes, and the ability to **share blocks across sequences with a common prefix** using copy-on-write. It is the core reason vLLM-class engines achieve the throughput they do.

**Q5. Walk me through GPU memory sizing for a 70B model at 8k context, batch 16.**
FP16 weights are 70B × 2 bytes = 140 GB, so it does not fit on one 80GB H100 — it needs at least two, and 2× 80 = 160 GB is tight. The KV cache is `2 × layers × kv_heads × head_dim × seq_len × batch × bytes`. For a 70B model with 80 layers, 8 KV heads, 128 head_dim at FP16: 2 × 80 × 8 × 128 × 2 = ~0.31 MB per token, so 8k tokens is 2.5 GB per sequence and batch 16 is 40 GB. Add activations and framework overhead and you are at roughly 186 GB — it does not fit. So I apply levers in order: quantize weights to FP8 first (140 → 70 GB, biggest win, smallest risk), then quantize the KV cache to FP8 (40 → 20 GB), and only then reduce batch or context or add a third GPU. And I would eval the quantized model on the task golden set before shipping it, not assume.

**Q6. What is continuous batching and how much does it buy?**
Instead of assembling a batch and running it to completion, continuous batching admits new requests at every decode step, so a slot freed by a finished sequence is immediately refilled. Static batching wastes the GPU on the longest sequence in the batch — a 5-token request and a 900-token request in the same batch both occupy a slot for 900 steps. Continuous batching is typically a 5–20x throughput improvement over no batching and is the baseline behaviour in vLLM, TGI, and TensorRT-LLM. It is what makes >60% GPU utilization achievable.

**Q7. Explain speculative decoding and its caveats.**
A small draft model proposes the next N tokens and the target model verifies them in a single forward pass, accepting the longest correct prefix. It is exact — the output distribution is identical to the target model alone, unlike approximate methods — because verification guarantees correctness. Typical speedup is 1.5–3x on decode. The caveats: it needs a well-matched draft model (a bad draft wastes the verification pass), it helps most on predictable output like summarization and structured extraction and least on high-entropy creative generation, and it adds memory because you are running two models.

**Q8. Why do you need an LLM gateway?**
Five reasons. First, provider concentration risk — every provider has outages, and fallback must be a system property rather than a try/except repeated in ten services. Second, cost control — budgets, hard caps, and per-tenant attribution require a central chokepoint. Third, key management — provider credentials scattered across services is a breach waiting to happen; one store, one rotation policy, one audit trail. Fourth, consistency — retry, timeout, cache, and observability policy should be identical for every caller. Fifth, change velocity — swapping a model or adding a provider should be a config change, not a deployment of every service. The costs are an extra hop of 1–10 ms and the fact that the gateway is now tier-0 and must itself be HA.

**Q9. How do you handle a provider outage?**
A chain: per-provider circuit breakers so a failing provider stops receiving traffic after 5 failures and is probed in half-open after 30 seconds. Retries with exponential backoff and ±50% jitter, capped at 3 attempts, and only for retryable errors — 429, timeouts, 503, connection errors. Idempotency keys so a retry does not duplicate work. Then provider failover across at least two providers, each with its own circuit breaker and its own eval baseline, because a fallback provider is degraded mode, not equivalent — different tokenizer, different safety behaviour, different quality. Then the degradation ladder: big model → small model → cached answer → static fallback message → human escalation. And I alert when fallback serves more than ~5% of traffic, because that is a quality incident even though error rates look fine.

**Q10. How do you decide self-host vs. API?**
By volume, control requirements, and expertise. API is right below roughly 1M queries/month, when time-to-market matters, when you want the newest models immediately, and when your workload is spiky. Self-host becomes compelling above roughly 10M queries/month, or immediately when data cannot leave your infrastructure, when you need sub-100 ms p99, or when you need custom weights. And I always check the arithmetic: at $2–4 per GPU-hour, a 40-GPU fleet costs $2,500–4,000/day — roughly the same order as the API bill for the same volume. **Self-hosting is not automatically cheaper**; it is cheaper only above roughly 50% sustained utilization. Most mature teams go hybrid: self-host the high-volume predictable workload, use APIs for spikes and specialised models, and keep an API as the fallback.

**Q11. What is prompt caching and how much does it save?**
The provider (or your own engine, as prefix caching) stores the KV cache for a stable prefix so subsequent requests reuse it instead of recomputing prefill. Savings on cached input tokens are roughly 90%, and TTFT drops substantially. The requirements are that the prefix be byte-identical and stable — so a system prompt with a timestamp in it destroys the cache — and that the prefix be long enough to exceed the provider's minimum. In a RAG system this is a large win because the retrieved context and system preamble are often identical across a session.

**Q12 (pushback). Isn't continuous batching plus a big batch always better?**
No, for two reasons. First, latency: bigger batches increase TPOT because each step processes more tokens, so a user-visible request gets slower. There is a real throughput-vs-latency trade-off, which is why serving engines let you cap batch size and why goodput-under-SLO is the right metric rather than raw throughput. Second, memory: batch size is bounded by KV cache capacity, and pushing KV utilization above ~95% causes preemption and recomputation, which *reduces* effective throughput non-linearly. So the answer is to tune batch size to the SLO, not to maximize it, and to monitor KV cache utilization as the real constraint.

**Q13. How do you prevent a retry storm?**
Jitter is the first requirement — fixed-interval retries synchronize the fleet and turn a blip into an outage. Then a retry budget capping total retries at around 10% of request volume over a window, so the fleet cannot amplify load more than that. Then circuit breakers so a genuinely failing provider is removed from rotation rather than being hammered. Then correct retryable/non-retryable classification, so we never retry a context-length-exceeded error. And idempotency keys so a retry that succeeds after the original actually completed does not duplicate the work.

**Q14. What do you monitor on an inference service?**
Two tiers. Service tier: TTFT p50/p95/p99, TPOT p50/p95, E2E latency, tokens/sec, goodput, error rate by class, retry rate, circuit breaker state, fallback share, cache hit rate, and cost per request and per tenant. Resource tier: GPU utilization, GPU memory, **KV cache utilization** (alert at 90%), batch size distribution, queue depth, and model load time during scale-out. The two panels that catch the most incidents are KV cache utilization — because it degrades non-linearly near saturation — and fallback share, because a silent fallback is a quality regression that error rates never reveal.

**Q15. What's the most common mistake you see in LLM serving design?**
Planning capacity in requests per second. Cost and GPU time scale with tokens, not requests, and a workload of 1,000 requests at 100 tokens each is completely different from 1,000 requests at 4,000 tokens each. The second most common is forgetting the KV cache in the memory budget — teams size for weights and then are shocked when long context OOMs. The third is running a global retry policy without jitter and without a retry budget, which converts provider blips into self-inflicted outages.

---

## 10. Common mistakes candidates make

| Mistake | Why it is wrong | Do this instead |
|---|---|---|
| Capacity planning in QPS | Cost and GPU time scale with tokens | Plan in tokens/sec: peak QPS × tokens/request ÷ measured tok/s per GPU |
| Forgetting the KV cache in memory math | Long context makes KV larger than the weights | Compute `2 × layers × kv_heads × head_dim × seq × batch × bytes` explicitly |
| Treating prefill and decode as one thing | They have opposite bottlenecks and different fixes | Separate TTFT and TPOT; optimize each phase independently |
| No `max_output_tokens` | Output is 3–5x the price of input and dominates latency | Cap it — it is both a cost and a latency lever |
| Fixed-interval retries | Synchronizes the fleet; turns blips into outages | Exponential backoff with ±50% jitter + a retry budget |
| Retrying non-retryable errors | Triples cost for requests that cannot succeed | Classify retryable (429/5xx/timeout) vs not (400/401/context/policy) |
| No circuit breaker | Every request waits through all retries during an outage | Per-provider breaker: 5 failures → open 30 s → half-open probes |
| Assuming a fallback provider is equivalent | Different tokenizer, safety behaviour, and quality | Treat fallback as degraded mode with its own golden-set eval |
| No gateway; providers called directly everywhere | Keys scattered, budgets unknown, retries inconsistent | Central gateway: keys, budgets, routing, retries, telemetry |
| Gateway with no HA story | It becomes the single point of failure for every AI feature | Stateless, horizontally scaled, with a documented fail-open mode |
| Semantic cache with no verification | 0.96-similar questions can have different answers (numbers, negation) | Verify entities/numbers or use a small equivalence check |
| Quantizing without task-level eval | Perplexity hides structured-output regressions | Run the task golden set; compare JSON validity and domain metrics |
| Ignoring KV cache utilization | Above 95% the scheduler preempts and throughput collapses | Alert at 90%; scale or cap before saturation |
| No per-tenant cost attribution | Cannot charge back; cannot detect an abusive tenant | Tag every request with tenant + route; expose cost per tenant |
| Sizing the fleet at 100% utilization | Queueing latency is non-linear near saturation | Plan for ~70% steady-state, +30–50% spike headroom |

---

## 11. Cheat-sheet summary

| Dimension | Answer |
|---|---|
| **Two phases** | Prefill (compute-bound → TTFT) and decode (memory-bound → TPOT) |
| **Latency identity** | `E2E ≈ TTFT + (output_tokens − 1) × TPOT` |
| **TTFT target** | < 200 ms chat; optimize with FlashAttention, FP8, prefix caching, chunked prefill, TP |
| **TPOT target** | < 30 ms (~33 tok/s); optimize with quantization, GQA, speculative decoding, batching |
| **Master throughput trick** | Continuous batching (5–20x); amortizes the decode weight load |
| **Memory trick** | Paged attention — fixed 16-token blocks, zero fragmentation, prefix sharing |
| **KV formula** | `2 × layers × kv_heads × head_dim × seq_len × batch × bytes` |
| **KV anchor number** | 70B/80-layer/8-KV-head/128-dim FP16 ≈ **0.31 MB per token** |
| **Memory-order levers** | FP8 weights → FP8 KV → smaller batch/context → more GPUs |
| **FP8** | 2x throughput vs FP16, < 0.1% quality loss on H100/B200 |
| **INT4** | 4x memory reduction, 2–4x decode speedup, 1–3% quality impact (measure it) |
| **Speculative decoding** | 1.5–3x decode, **exact** output distribution, needs a matched draft model |
| **GQA** | 8x KV reduction for a 64-head/8-KV-head model — near-free |
| **Engines** | vLLM (default), TGI, TensorRT-LLM (max NVIDIA perf), SGLang (prefix-heavy), Ollama (dev), llama.cpp (CPU) |
| **Capacity unit** | **Tokens per second**, not requests per second |
| **Capacity formula** | `GPUs = (peak_QPS × tokens_per_request) ÷ measured_tok/s_per_GPU` |
| **Headroom** | ~70% steady state; +30–50% for spikes; +1 replica for HA |
| **Self-host break-even** | Roughly 1M queries/month to switch, 10M to clearly win, > ~50% utilization |
| **GPU cost anchor** | $2–4/GPU-hour → 40 GPUs ≈ $2.5k–4k/day |
| **Gateway owns** | Keys, routing, budgets, caching, retries, failover, telemetry |
| **Retry defaults** | 3 attempts, 1 s base, 60 s max, **±50% jitter**, 10% retry budget |
| **Circuit breaker** | 5 failures → open 30 s → 3 half-open probes → 2 successes → closed |
| **Retryable** | 429, timeout, 503, connection, 5xx. **Not:** 400/401/403, context length, policy |
| **Routing outcome** | 50–70% of traffic to a small model at ~1/10 the cost |
| **Cost asymmetry** | Output tokens cost 3–5x input tokens — cap `max_output_tokens` |
| **Caching** | Exact (~100% on hit), semantic ≥0.95 (~68%), prompt/prefix (~90% input discount) |
| **Cache key must include** | Model version + index version + tenant |
| **Degradation ladder** | Big model → small model → cache → static → human |
| **Alert thresholds** | KV util > 90%, TTFT p99 > 800 ms, TPOT > 80 ms, fallback share > 5%, hourly cost > 2× trailing mean |
| **Biggest design mistake** | Planning in QPS and forgetting the KV cache |

---

## 12. Cross-links

- [Production RAG System](01-production-rag-system.md) — the generator this serves, and prompt caching in context
- [Advanced RAG Patterns](02-advanced-rag-patterns.md) — the router pattern that lives in the gateway
- [RAG Evaluation and Observability](03-rag-evaluation-and-observability.md) — tracing TTFT/TPOT per stage
- [Guardrails and Safety for AI Systems](04-guardrails-and-safety-for-ai-systems.md) — gateway-level rate limits and budget caps
- [Agentic System Design](06-agentic-system-design.md) — agents multiply model calls; step and cost budgets
- [Memory and Context Engineering](08-memory-and-context-engineering.md) — prompt caching economics and KV reuse
- [Capacity Planning and Autoscaling](../../Operations/05-capacity-planning-and-autoscaling.md) — tokens/sec, queueing theory, autoscale on queue depth
- [Cost and FinOps](../../Operations/06-cost-and-finops.md) — routing, tiering, and the 15 optimisation plays
- [Reliability Patterns](../../Operations/07-reliability-patterns.md) — timeouts, breakers, bulkheads, DLQs
- [RAG and LLM Cheatsheet](../../Cheat-Sheets/04-rag-and-llm-cheatsheet.md)
