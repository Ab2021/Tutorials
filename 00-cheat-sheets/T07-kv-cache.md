# Cheat Sheet: KV Cache, Paged Attention & Tiering

> `T07` · **Transcript coverage:** partial · **The most cross-cutting topic in the corpus** — appears in 8+ transcripts
> [Case study](../01-case-studies/T07-kv-cache.md) · [Blueprint](../03-design-blueprints/T07-kv-cache/HLD.md) · [Interview bank](../02-interview-questions/T07-kv-cache.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| **CPU KV offload, session return** | **~5× TTFT improvement** | `[T]` llm-d talk |
| KV eviction in agentic workloads | frequent enough that the router consumes **per-request** create *and* evict events, and an offload tier + retention API exist to manage it `[T]`; that this is the steady state for long agent sessions is an inference `[D]` | `[T]`/`[D]` llm-d |
| Saturation threshold example | KV cache **80% full** | `[T]` llm-d talk |
| KV utilisation as a deployment knob | fill to **90% of VRAM** | `[T]` NextGen |
| Paged attention block size | **16 tokens** per block | `[R]` guide / llm-inference-engineering |
| Memory waste, naive contiguous | **~60–80%** | `[R]` guide |
| Memory waste, paged | **<4%** | `[R]` guide |
| Rack-scale pool demo | **4 servers**, one pooled memory box mid-rack | `[T]` Jongryool Kim, SK hynix |
| Transfer latency ranking | pooled memory **< RDMA << TCP/IP** | `[T]` Jongryool Kim |
| KV recomputation cliff | at **28k input tokens** KV cache recomputation happened every time | `[T]` ROCm/WideEP talk |

---

## The one-table summary

| Technique | What it buys | Cost | Use when |
|---|---|---|---|
| **Paged attention** | ~4% waste vs ~60–80%; enables sharing | block-table indirection | **always** — it is table stakes |
| **Prefix caching** | skip re-prefill of a shared prefix | radix/block hash lookup | stable system prompts, multi-turn, RAG with fixed preamble |
| **Block sharing / COW** | one copy for N identical prefixes | refcount bookkeeping | fan-out (best-of-N), shared system prompt |
| **KV offload → CPU DRAM** | frees HBM; ~5× TTFT win on session return | PCIe transfer on evict/restore | multi-turn sessions with idle gaps |
| **KV offload → SSD/object store** | effectively unbounded retention | slow restore, must be async | long-lived agent sessions |
| **P2P KV sharing (GPU↔GPU)** | avoid recompute *and* queueing | fabric bandwidth | middle ground between queue-on-busy and recompute-on-cold `[T]` |
| **KV quantization** (fp8/int8) | ~2× more KV in the same HBM | accuracy risk | memory-bound, accuracy headroom |
| **GQA / MQA** | divides KV size by group factor | model must be trained that way | architectural — decided at training |
| **Linear / hybrid attention (KDA)** | **fixed-size state** per sequence | different per-layer behaviour | very long context `[T]` |
| **Retention API + session metadata** | router knows which session owns a block | needs a metadata plane | agentic serving `[T]` |

---

## Formulas

**KV bytes per token**
```
bytes/token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes
```
Llama 3.1 70B, fp16: `2 × 80 × 8 × 128 × 2 = 327 KB/token` ⇒ **~43 GB at 128k context**, for one
sequence. This is why GQA exists: with full MHA (`n_kv_heads = n_heads`) it would be many times
larger.

**Paged allocation**
```
n_blocks_needed = ceil(seq_len / block_size)          # block_size = 16 typical
waste_frac      ≈ (block_size − 1) / (2 × seq_len)     # expected internal fragmentation
```
At `seq_len = 2048`, `block=16`: waste ≈ 0.4% — versus up to 80% for contiguous allocation sized to
the max sequence `[R]`.

**KV cache capacity**
```
max_concurrency ≈ (HBM_for_KV) / (bytes/token × avg_context_len)
```
Worked: 40 GB for KV, 327 KB/token, 4k average context ⇒ `40e9 / (327e3 × 4000) ≈ 30` concurrent
sequences. **This single division is what your concurrency limit actually is** `[D]`.

**Prefix cache hit benefit**
```
saved_prefill = hit_tokens × prefill_cost_per_token
```
Cache reads are **~10× cheaper** than fresh tokens `[T]` LLMOps cost talk.

---

## Configuration

```bash
# vLLM — the flags that matter
vllm serve <model> \
  --gpu-memory-utilization 0.90 \        # the KV-utilisation knob [T] NextGen
  --block-size 16 \                      # paged attention block
  --enable-prefix-caching \              # radix prefix cache
  --swap-space 16 \                      # CPU offload capacity (GB)
  --max-model-len 32768 \                # caps bytes/token × ctx
  --kv-cache-dtype fp8                   # halve KV memory, accuracy risk
```

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| TTFT spikes on repeat turns | prefix cache miss; routing to a cold pod | per-pod KV hit rate, router affinity |
| Throughput collapses at long context | KV eviction + **recomputation** | eviction rate; the 28k-input cliff `[T]` |
| OOM under load | `bytes/token × max_ctx × concurrency` unaccounted | lower `--max-model-len` or `--gpu-memory-utilization` |
| Cache hit rate "atrocious" in Prometheus | naive LB sending follow-ups to random pods | prefix-aware routing `[T]` |
| Sessions slow after an idle gap | KV evicted during the gap | CPU/SSD offload tier `[T]` |
| Accuracy dropped after enabling `fp8` KV | KV quantization | rerun evals — never assume |
| GPU idle but requests queueing | prefill/decode imbalance | split pools ([T12](../01-case-studies/T12-disaggregation-kv-transfer.md)) |
| Memory waste high | contiguous allocation somewhere | confirm paging is actually on |

---

## Gotchas

- **Treat eviction as routine for agents, not as an exception** `[D]`. The corpus's supporting
  evidence is indirect but strong `[T]`: the router is fed create **and** evict events *per request*,
  and both a CPU offload tier and a retention API exist specifically to manage KV movement. Agentic
  sessions have long idle gaps and large prefixes, so design for the cache being gone rather than
  assuming it survives.
- **Hash-based prefix routing is not good enough.** Precise **per-block KV events** (creation and
  eviction) beat approximate hashing, which suffers consistency problems. This is the core argument
  for llm-d's event-based affinity `[T]`.
- **Prefix caching only helps a *stable* prefix.** Cache something volatile — a timestamp, a
  per-request user blob early in the prompt — and you gain nothing `[T]`.
- **Context condensation reduces cache effectiveness.** Summarising history replaces the cached
  tokens with new ones, so you trade compute savings for cache hits `[T]` CMU agents lecture.
- **Hybrid attention breaks static memory partitioning.** With mixed full-attention and
  linear-attention (KDA) layers, the optimal split depends on batch size and context length, which
  change at runtime. You need **dynamic partitioning over one shared pool with per-attention-type
  allocators and automatic rebalancing** `[T]` Kwon.
- **Linear attention keeps a fixed-size state per sequence** — radically different memory behaviour
  per layer. You cannot reason about the model with one number `[T]`.
- **Pooled/rack-scale memory beats RDMA and destroys TCP/IP.** Sharing a physical memory box across
  nodes also removes GPU-side and PCIe contention — and it survives prefill OOM because the KV is
  already in the pool `[T]` Jongryool Kim.
- **Block size is a real tradeoff.** Larger blocks ⇒ less table overhead, more internal
  fragmentation, worse sharing granularity `[D]`.
- **KV cache is what makes prefill/decode disaggregation necessary** — if there were no KV to move,
  the two phases could stay together `[D]`.

---

## When to use what

| Situation | Do |
|---|---|
| Any production serving | paged attention + prefix caching (non-negotiable) |
| Stable system prompt / fixed RAG preamble | prefix caching, and put the volatile part **last** |
| Multi-turn chat | CPU offload tier |
| Agent sessions with hour-long idle gaps | offload to DRAM/SSD + retention metadata |
| Long-context, memory-bound | KV quantization (after evaluating accuracy) |
| Fan-out (best-of-N, beam) | block sharing with copy-on-write |
| Multi-replica serving | **prefix-cache-aware routing** — not round-robin |
| Very long context, architectural freedom | hybrid/linear attention |

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Jongryool_Kim_-_Disaggregated_LLM_Serving_with_Shared_Memory_KV_Cache_at_Rack_Sc.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Woosuk_Kwon_-_vLLM_Building_Open_and_Efficient_Inference_for_Agents.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_11_Agents_and_Multi-Agent_Communication.txt`
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` `[R]`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/05-paged-attention.md` `[R]`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/02-kv-cache-and-context-caching.md` `[R]`
