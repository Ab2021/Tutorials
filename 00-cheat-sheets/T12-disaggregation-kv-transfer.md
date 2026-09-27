# Cheat Sheet: Prefill/Decode Disaggregation & KV Transfer

> `T12` · **Transcript coverage:** primary · [Case study](../01-case-studies/T12-disaggregation-kv-transfer.md) · [Blueprint](../03-design-blueprints/T12-disaggregation-kv-transfer/HLD.md) · [Interview bank](../02-interview-questions/T12-disaggregation-kv-transfer.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| **PD ratio used in the ROCm/llm-d demo** | **1 prefill : 1 decode (1P1D)** | `[T]` ROCm/WideEP |
| Max concurrency on that 1P1D pair | **~20,000** | `[T]` ROCm/WideEP |
| A second config, explicitly flagged preliminary | **2P4D** — *"not to be trusted"* | `[T]` ROCm/WideEP |
| Rack-scale demo topology | **4 servers**, one pooled memory box mid-rack | `[T]` Jongryool Kim, SK hynix |
| **Transfer latency ranking** | **pooled memory < RDMA << TCP/IP** | `[T]` Jongryool Kim |
| KV offload win on session return | **~5× TTFT** | `[T]` llm-d |
| The cliff that forces the split | **28k inputs** ⇒ KV recomputation every time | `[T]` ROCm/WideEP |
| Why move KV at all | prefill is **compute-bound**, decode is **memory-bandwidth-bound** | `[T]`/`[D]` |

---

## The one-table summary

| Topology | What it separates | Cost | Use when |
|---|---|---|---|
| **Colocated (no PD)** | nothing | none | small scale, uniform traffic, simplest ops |
| **PD disaggregation** | prefill pool from decode pool | KV transfer on every request | long prompts + high concurrency; TTFT and ITL both SLO-bound |
| **EPD** | **encode** (vision) + prefill + decode | an extra hop | multimodal — image/video encode is its own resource profile |
| **KV offload tiering (no split)** | HBM from DRAM/SSD | transfer on evict/restore | session reuse; keeps one pool |
| **Rack-scale pooled memory** | the *storage medium* from the GPU | specialised hardware | you control the datacentre — best transfer latency `[T]` |
| **P2P GPU↔GPU KV** | nothing; peers share | fabric bandwidth | middle ground: avoid recompute without a pool `[T]` |
| **NIXL** | not a topology — the **transfer library** | integration work | the plumbing for any of the above `[R]` |

**The core argument** `[T]`/`[D]`: prefill and decode want opposite things. Prefill is one big matmul
— compute-bound, wants large batches of prompt tokens, saturates tensor cores. Decode is a
token-at-a-time loop — memory-bandwidth-bound, reads all weights per token, wants large *sequence*
batches. Colocating them means each phase's optimal batch shape starves the other. Separating them
lets you size and scale each pool independently — at the cost of moving the KV cache between them.

---

## Formulas

**Why the split can win — phase arithmetic** `[D]`:
```
prefill_time ≈ 2 × N_params × prompt_tokens / (FLOPs_peak × MFU)
decode_time  ≈ bytes_per_token_read × tokens / HBM_BW
```
Prefill scales with **prompt tokens** at compute peak; decode scales with **weights read** at
bandwidth. No single batch size optimises both, and the optimum shifts with the traffic mix.

**KV transfer volume per request**
```
transfer_bytes = bytes_per_token × prompt_len
```
Where `bytes/token = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes` (see [T07](T07-kv-cache.md)).
For Llama-3.1-70B fp16 that is ~327 KB/token, so a **28k-token prompt moves ~9 GB** before decode
starts. **This is the number that decides whether PD disaggregation is viable on your fabric** `[D]`.

**The break-even condition** `[D]`:
```
transfer_time = transfer_bytes / fabric_BW
split_wins  ⟺  (prefill_gain + decode_gain) > transfer_time + queueing + failure_risk
```
On NVLink/NVSwitch this is easy; over TCP/IP it is usually a loss; over pooled memory it is the
cheapest `[T]`.

**Pool sizing** `[T]`:
```
P:D ratio ≈ (aggregate prefill token rate) / (aggregate decode token rate)
```
The ROCm demo ran **1:1** and reached ~20k concurrency; a **2:4** variant was tried and explicitly
marked preliminary. **Measure your own ratio** — it is workload-dependent `[T]`.

---

## Configuration

```bash
# vLLM — disaggregated prefill and decode as separate instances
# (1) prefill instance
vllm serve <model> --port 8100 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# (2) decode instance
vllm serve <model> --port 8200 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

# (3) proxy in front, routing by phase
python -m vllm.entrypoints.openai.api_server --port 8000   # or llm-d's EPP
```

```yaml
# llm-d — the higher-level way: declare the role, let the scheduler place it
# (Helm values, abbreviated)
decode:
  replicas: 4
prefill:
  replicas: 2
  kvTransfer:
    connector: NixlConnector
```

Key connectors `[R]`: **NIXL** (NVIDIA Inference Xfer Library — the general RDMA-capable transfer
layer), **LMCache** (a KV store/cache front-end), **Mooncake** (the KVCache-centric disaggregated
architecture). The **connector abstraction** is what lets you swap the transport without redesigning
the pools.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Split made TTFT *worse* | transfer latency exceeds the prefill saving | fabric bandwidth vs `KV_bytes / BW` |
| Decode pool idle, prefill pool saturated | wrong P:D ratio | re-measure the token-rate ratio |
| Goodput worse than colocated | added queueing hop | try colocated as the baseline before believing the split |
| Works at 8k context, fails at 28k | KV volume too large for the fabric | the documented 28k cliff `[T]` |
| Transfer errors under load | NIXL/RDMA misconfiguration or NIC contention | connector logs, RDMA counters |
| Encode-heavy traffic stalls prefill | no separate encode stage | **EPD** instead of PD `[D]` |
| Long tail on P99 | one slow transfer blocks the request | async transfer + timeout/fallback to recompute |
| Cost went *up* | you doubled the pool count without raising utilisation | the split needs high load to pay off `[D]` |

---

## Gotchas

- **PD disaggregation is not free and not always a win.** It buys independent scaling and phase-pure
  batches; it costs a KV transfer on every request plus a second failure domain. **Benchmark against
  a colocated baseline** — the corpus's own 2P4D number was flagged as untrustworthy `[T]`.
- **The KV transfer is the whole engineering problem.** Once you accept that, the design reduces to:
  which fabric, which connector, and how to hide the latency `[D]`.
- **Pooled memory beats RDMA, which beats TCP/IP** `[T]`. If you are choosing a transport and have
  the option of physical shared memory, that is the fastest path — and it also removes GPU-side and
  PCIe contention.
- **The rack-scale pool also fixes prefill OOM**, because the KV already lives in the pool rather
  than in a GPU's HBM `[T]` Jongryool Kim. That is a second, independent benefit beyond latency.
- **EPD is the multimodal generalisation.** Image/video encoding is neither prefill nor decode and
  deserves its own pool — `[T]` the corpus mentions encode as a distinct stage for vision models.
- **The ratio is empirical, not theoretical.** Prefill and decode costs move with context length,
  output length and model. Re-derive it when any of those change `[D]`.
- **At low load, the split loses.** Two half-idle pools are worse than one busy pool. It is a
  high-utilisation technique `[D]`.
- **Disaggregation changes your failure domains.** A prefill-pool outage now fails decode-capable
  requests too. Plan the degradation path: fall back to colocated or to KV recomputation `[D]`.
- **KV offload tiering is the lower-risk cousin.** If you want the session-reuse benefit without
  splitting pools, offload to CPU/SSD first and measure `[T]`.

---

## When to use what

| Situation | Do |
|---|---|
| Small scale, uniform traffic | **colocated** — do not split |
| Long prompts + high concurrency, both TTFT and ITL SLO-bound | PD disaggregation |
| Multimodal, encode-heavy | **EPD** |
| Multi-turn sessions with idle gaps | KV offload tier first (cheaper) |
| You control the datacentre and the network is the bottleneck | **rack-scale pooled memory** `[T]` |
| You need the split but have no shared memory | NIXL over RDMA |
| You have TCP/IP only | probably do not split — measure first |
| Fabric bandwidth unknown | compute `KV_bytes / BW` before designing anything `[D]` |

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Jongryool_Kim_-_Disaggregated_LLM_Serving_with_Shared_Memory_KV_Cache_at_Rack_Sc.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Woosuk_Kwon_-_vLLM_Building_Open_and_Efficient_Inference_for_Agents.txt`
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` `[R]`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md` `[R]`
