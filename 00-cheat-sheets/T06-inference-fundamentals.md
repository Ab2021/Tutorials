# Cheat Sheet: Prefill/Decode, Roofline & Latency Metrics

> `T06` · **Transcript coverage:** primary · [Case study](../01-case-studies/T06-inference-fundamentals.md) · [Blueprint](../03-design-blueprints/T06-inference-fundamentals/HLD.md) · [Interview bank](../02-interview-questions/T06-inference-fundamentals.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| Llama 3.1 405B | layers **126**, hidden **16384**, **8 GQA KV heads**, 128k context | `[T]` CMU lecture 1 |
| Llama 3.1 70B / 8B | layers **80** / **32**, hidden **8192** / **4096** | `[T]` CMU lecture 1 |
| MLP width | **≈3.5×** hidden dim | `[T]` CMU lecture 1 |
| One traced request | **1.2 s** total = retrieval **90 ms** + generation **>1 s** | `[T]` LLMOps observability |
| Enterprise SLA example | **16 req/s sustained** on half an H100 | `[T]` NextGen |
| Pipeline planning | **256 pages/sec** end-to-end | `[T]` NextGen |
| Max concurrency, 1P1D | **~20,000** | `[T]` ROCm/WideEP |

---

## The one-table summary — the two phases

| | **Prefill** | **Decode** |
|---|---|---|
| What it does | processes the **whole prompt** at once | generates **one token at a time** |
| Parallelism | fully parallel over prompt tokens | inherently sequential |
| Bound by | **compute** (FLOPs) | **memory bandwidth** |
| Determines | **TTFT** | **ITL / TPOT** |
| Bottleneck resource | tensor cores | HBM bandwidth |
| Batching effect | amortises well | amortises *very* well (weights read once for the whole batch) |
| Hardware implication | wants FLOPs | wants memory bandwidth — this is why some silicon targets decode specifically `[T]` |
| Optimisations | chunked prefill, prefix caching, SP | KV cache, quantization, spec decoding, MTP |

> **The single most useful sentence in the corpus:** *prefill is compute-bound and sets TTFT;
> decode is memory-bandwidth-bound and sets ITL.* Almost every serving optimisation follows from
> which side of this line it targets. `[T]`

> **And the reframe** `[T]` NextGen: *"decode is a solved problem — the hard part is now producing
> the first token cheaply."* Agentic workloads are ~98% prefill tokens, so TTFT is where the
> problem moved.

---

## The metrics that matter

| Metric | Definition | Use for |
|---|---|---|
| **TTFT** | time to first token | interactive chat, perceived responsiveness |
| **ITL** | inter-token latency | streaming smoothness |
| **TPOT** | time per output token | the same thing, averaged |
| **End-to-end latency** | `TTFT + ITL × output_tokens` | the user-visible number |
| **Throughput** | tokens/sec across all requests | cost efficiency |
| **Goodput** | requests **meeting the SLO** / total requests | **the metric that actually matters** |
| **Request latency** | full request duration | **agentic workloads** — the right unit `[T]` |
| **Session / program completion time** | whole multi-turn agent program | agentic workloads `[T]` |
| **KV cache hit rate** | cached prefix tokens / total | agentic workloads `[T]` |
| **Cache hit rate** | as above, service-wide | cost |
| **Cost / 1k requests** | unit economics | FinOps |

**Why goodput and not throughput** `[D]`: throughput counts every token; goodput counts only tokens
delivered inside the SLO. A configuration can double throughput while halving goodput by batching
so aggressively that latency blows out. **Optimise goodput.**

**Agentic workloads invert the metric set** `[T]`: TTFT/ITL matter for interactive chat, but for
agents use **request latency and program completion time**, plus **KV hit rate per session**. An
agent that is fast per call but cannot reuse KV will lose.

---

## Formulas

**Latency budget**
```
E2E = TTFT + ITL × N_out
```
Worked: TTFT 200 ms, ITL 20 ms, 300 output tokens ⇒ `0.2 + 0.02×300 = 6.2 s`. To hit a 3 s budget
you must cut ITL to ~9 ms **or** cut output length — TTFT is not the lever at this length.

**FLOPs per token (dense, forward pass)** — the standard approximation:
```
FLOPs/token ≈ 2 × N_params          (multiply + accumulate)
Prefill FLOPs ≈ 2 × N × N_prompt
Decode  FLOPs ≈ 2 × N               (per generated token, per sequence)
```
The `2N` rule is why decode is cheap in FLOPs and *expensive in memory traffic* — you read all `N`
weights to produce one token.

**Arithmetic intensity / roofline**
```
intensity = FLOPs / bytes_moved
if intensity < machine_balance:  → memory-bound
else:                            → compute-bound
```
`machine_balance` = peak FLOPs ÷ memory bandwidth (e.g. an H100-class part is roughly a few hundred
FLOPs per byte). **Decode sits far below this line; prefill sits above it.** That single comparison
is the whole roofline story for LLM inference.

**Attention is quadratic, MLPs are linear** `[T]`:
```
attention FLOPs ∝ N_ctx²  ·  d
MLP FLOPs       ∝ N_ctx   ·  d²
```
At **short** context the MLP dominates. At **long** context attention takes over — which is why
long-context serving is a different engineering problem, and why sparse/linear attention
architectures exist.

**KV cache size per token**
```
bytes/token = 2 (K and V) × n_layers × n_kv_heads × head_dim × dtype_bytes
```
Llama 3.1 405B, fp16, 128k context: `2 × 126 × 8 × 128 × 2 ≈ 516 KB/token` ⇒ **~66 GB for a single
128k-context sequence.** This is the arithmetic that motivates GQA, KV quantization and paging `[D]`.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| TTFT high, ITL fine | prefill-bound — long prompts, no prefix cache | prompt length distribution, cache hit rate |
| TTFT fine, ITL high | decode-bound — KV pressure or low batch | batch size, KV utilisation |
| Throughput up, users complain | batching hurting latency | measure **goodput**, not throughput |
| Throughput collapses at long context | attention quadratic term + KV eviction | context length histogram |
| Same output differs run to run at `temperature=0` | float non-associativity, MoE argmax ties, batch-shape-dependent kernels | expected `[T]` |
| Latency fine in load test, bad in prod | load test used uniform prompt lengths | real traffic is skewed; check P95 not mean |
| KV cache OOM mid-run | long-context sequences not accounted for | `bytes/token × max_ctx × concurrency` |

---

## Gotchas

- **Mean latency hides everything.** Always report P50/P95/P99 and slice by prompt-length bucket.
- **`2N` FLOPs/token ignores attention.** It is accurate at short context and wrong at long context.
- **Goodput ≠ throughput.** Batching past the latency SLO makes throughput look good while the
  service gets worse.
- **TTFT is a prefill problem and prefill is ~98% of agentic tokens** `[T]` — for agent workloads
  optimise TTFT and KV reuse, not decode speed.
- **Decode is memory-bandwidth-bound, so quantizing weights helps decode far more than it helps
  prefill** `[D]`.
- **Speculative decoding helps decode, not prefill** — it attacks the sequential token loop.
- **Report the metric set per workload class.** Chat → TTFT/ITL. Agent → request latency, session
  completion, KV hit rate. Batch → throughput and cost. Using one dashboard for all three is a
  common and expensive mistake `[T]`.

---

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_1_Introduction_to_Language_Models_and_Inference.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_7_Chain_of_Thought_and_Intermediate_Steps.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt`
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt`
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/LLM_Observability_Traces_Spans_OpenTelemetry_for_AI_Apps.txt`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/01-inference-fundamentals.md` `[R]`
