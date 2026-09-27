# Cheat Sheet: Parallelism, MoE & WideEP

> `T11` · **Transcript coverage:** primary · [Case study](../01-case-studies/T11-parallelism-moe.md) · [Blueprint](../03-design-blueprints/T11-parallelism-moe/HLD.md) · [Interview bank](../02-interview-questions/T11-parallelism-moe.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| **Parallelism types in vLLM** | **7** — tensor, pipeline, data, expert, sequence, context, decode-context (DCP) | `[T]` Kwon |
| Kimi K3 | **896 experts** | `[T]` ROCm/WideEP |
| DeepSeek V3 | **256 experts** | `[T]` ROCm/WideEP |
| GLM 5.1 | 78 layers, **256 experts**, top-k **8**, sparse attention + MLA | `[T]` ROCm/WideEP |
| EP sizing | EP8 ⇒ **32 experts/GPU**; EP32 (4 nodes × 8 GPUs) ⇒ **8 experts/GPU** | `[T]` ROCm/WideEP |
| AMD MI300 | **192 GB** HBM/GPU | `[T]` ROCm/WideEP |
| AMD MI355 | **288 GB** HBM/GPU | `[T]` ROCm/WideEP |
| Naive MoE → fused | **2 all-to-all + 6 kernels → 3 kernels** | `[T]` ROCm/WideEP |
| NIAH validation | 10 needles × concurrency × **3 shapes = 72 configs**, all passed | `[T]` ROCm/WideEP |
| Max concurrency, 1P1D pair | **~20,000** | `[T]` ROCm/WideEP |
| **The cliff** | at **28k inputs**, KV recomputation every time ⇒ throughput drop at concurrency 256 | `[T]` ROCm/WideEP |
| The worked config | naive **8-way TP (single host)** loses to a tuned mix using TP+PP+SP+EP across **16 GPUs/replica** | `[T]` Kwon (exact factorisation ASR-garbled) |

---

## The one-table summary

| Strategy | Splits | Communication | Use when |
|---|---|---|---|
| **Data parallel (DP)** | the batch across replicas | none (embarrassingly parallel) | always the outer layer — replicas |
| **Tensor parallel (TP)** | each layer's matrices | **all-reduce every layer** — very chatty | fits within a node over NVLink; **keep it small** |
| **Pipeline parallel (PP)** | layers into stages | point-to-point between stages | across nodes; adds bubbles |
| **Expert parallel (EP)** | MoE experts across GPUs | **all-to-all** | MoE models — this is the main axis |
| **Sequence parallel (SP)** | the sequence dim | all-gather/reduce-scatter | long prefill — **overlaps comm with compute** |
| **Context parallel (CP)** | context across devices | attention-level comm | very long context |
| **Decode context parallel (DCP)** | context **around the KV cache** | KV-level | long-context decode `[T]` |
| **WideEP** | **DP attention + EP MoE** | all-to-all for experts only | MoE on large clusters — the current default `[T]` |

**The headline config result** `[T]` Kwon: for **DeepSeek prefill on B200 in a disaggregated
(prefill-only) pool**, a naive single-host **8-way tensor parallel** deployment is *worse* than a
tuned mix deployed **across 16 GPUs per model replica** — with much lower TTFT and much higher
throughput per GPU.

The transcript credits four mechanisms: **pipeline parallelism** parallelises chunks of the long
prefill sequence, **sequence parallelism** overlaps communication with compute, **expert
parallelism** gives better GEMM shapes, and **tensor parallelism stays at 2** (not 8). The exact
middle of the config is garbled by ASR in the source — the speaker's phrasing runs "two-way tensor
parallel … two-way pipeline parallel … tensor parallel plus sequence parallelism … across 16 GPUs"
`[T]`. **Treat the four mechanisms and the 16-GPU figure as reliable; do not quote a precise
factorisation.** The load-bearing claim is the contrast with 8-way TP, not the exact product.

> **"there's no universal winner"** Parallelism must be chosen per **model architecture, cluster
> setup, and workload shape** `[T]` Kwon. Anyone offering a universal rule is wrong.

---

## Formulas

**MoE parameter arithmetic** — the sizing that determines your cluster:
```
experts_per_GPU = total_experts / EP_degree
```
GLM 5.1 at EP8: `256/8 = 32 experts/GPU`. At EP32: `256/32 = 8 experts/GPU` — freeing VRAM for KV
at the cost of more cross-node all-to-all `[T]`.

**Why MoE needs EP at all** `[T]`: an MoE model's total parameters are far larger than its *active*
parameters per token. You cannot fit the weights (so DP alone fails), but you only compute a few
experts per token (so TP on the whole thing is wasteful). EP matches the computation to the sparsity.

**Communication volume, all-to-all**
```
bytes ≈ tokens × top_k × hidden_dim × dtype_bytes × 2   (dispatch + combine)
```
Every MoE layer performs a dispatch all-to-all and a combine all-to-all. **This is why MoE serving is
a network problem** `[T]`: *"distributed inference is becoming more of a communication and memory
problem, not a computation problem."*

**Kernel fusion win** `[T]`: naive MoE layer = 2 all-to-all + **6 kernels** → fused to **3 kernels**
(dispatch A2A, combine A2A, fused MoE). The 6-stage fused MoE is: top-k permute → grouped GEMMs →
unpermute → reduction/scale. Removing GPU launch overhead is the point.

**Pipeline bubbles**
```
bubble_fraction ≈ (PP_degree − 1) / (microbatches + PP_degree − 1)
```
More microbatches amortise the bubble — which is why PP pairs well with high concurrency `[D]`.

---

## Configuration

```bash
# WideEP / MoE on AMD — the ROCm path
VLLM_USE_ROCM=1 vllm serve <moe-model> \
  --tensor-parallel-size 1 \              # note: TP stays 1 [T]
  --data-parallel-size 16 \               # DP covers attention
  --enable-expert-parallel \              # EP covers MoE
  --enable-chunked-prefill

# The tuned NVIDIA prefill config from the talk: TP+PP+SP+EP over 16 GPUs/replica.
# Exact factorisation is ASR-garbled in the source - see the table above; TP is 2, not 8. [T]
vllm serve <model> --tensor-parallel-size 2 --pipeline-parallel-size 2 \
  --enable-expert-parallel --enable-chunked-prefill
```

**In WideEP, tensor parallelism deliberately stays at 1** `[T]`. DP handles attention, EP handles
MoE. Do not reach for TP out of habit on MoE models.

**But note the nuance the ROCm team states for their own target configuration** `[T]`: they run
**TP8 + 2P2D + EP8 (intra-node, "shallow EP") + DP16** — that is, TP *is* used, but kept
**intra-node at 8**, with the *wide* dimension carried by DP16 and EP8. So the accurate statement is
**not** "TP is always 1"; it is that **TP stays inside the node and the MoE/attention sharding is
done by EP and DP**. The single-node EP8 arrangement is what they call *shallow* EP; EP32 across
four nodes is the *wide* case `[T]`.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| All-to-all fabric saturated | EP degree too high across nodes | EP topology vs node boundary |
| Throughput collapses at ~28k context | **KV recomputation** — the documented cliff `[T]` | KV cache capacity vs context length |
| TP scales badly past 8 | cross-node all-reduce every layer | keep TP intra-node over NVLink |
| PP idle time | bubble with too few microbatches | raise concurrency |
| Experts unevenly loaded | poor routing balance | expert load histogram |
| MoE slow on AMD vs NVIDIA | WideEP enablement gaps / pending PRs | check ROCm upstream status `[T]` |
| Kernel launch overhead dominates | unfused MoE path | ensure the fused kernel is in use |
| New hardware bring-up takes months | the whole stack must be re-taken from the ground up | expect it; coding agents help but do not remove it `[T]` |

---

## Gotchas

- **"TP stays 1" is stated for the WideEP attention case, and it is easy to over-generalise.**
  The same team's own target config uses **TP8 intra-node** alongside EP8 and DP16 `[T]`. The
  durable rule is *TP stays inside the node; EP and DP carry the wide dimension* — not "TP is
  always 1".
- **MoE routing is not done at the serving-router layer** — it happens inside the LM `[T]`. Do not try
  to influence expert choice from llm-d.
- **EP moves the bottleneck from compute to network.** Past a certain EP degree, adding GPUs makes
  things worse because all-to-all dominates `[D]`.
- **The 28k-token cliff is a KV capacity problem, not a compute problem.** It appears as a
  throughput drop at high concurrency because KV is recomputed every time `[T]`.
- **NIAH (needle-in-a-haystack) is your long-context correctness regression test.** The ROCm team
  runs a 10-needle × concurrency × 3-shape matrix — 72 configurations — with a ≥7-needle pass
  threshold. Copy this pattern `[T]`.
- **Bringing up new hardware requires re-taking the whole inference stack.** Coding agents make it
  easier but it is still a from-scratch effort — and vLLM's plugin structure with a shared core is
  what makes >10 backends tractable `[T]`.
- **Benchmark your parallel plan before committing.** The talk's own 2P4D result was explicitly
  flagged *preliminary — not to be trusted*. Configurations interact in ways that are not
  predictable from first principles `[T]`.
- **Pipeline parallelism needs high concurrency to be worth it.** At low load the bubble dominates `[D]`.

---

## When to use what

| Situation | Plan |
|---|---|
| Dense model fits on a node | DP replicas + TP within the node (TP ≤ 8) |
| Dense model spans nodes | TP intra-node + PP inter-node |
| MoE, large cluster | **DP attention + EP MoE (WideEP)**, TP = 1 |
| Long-context prefill, TTFT-bound | add **SP** to overlap communication |
| Long-context decode | **DCP** around the KV cache |
| Very long context, single sequence | CP + DCP |
| Unknown | **measure** — there is no universal winner `[T]` |

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Woosuk_Kwon_-_vLLM_Building_Open_and_Efficient_Inference_for_Agents.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt`
- `refs/gpu-perf-engineering-resources-main/gpu-perf-engineering-resources-main/README.md` `[R]`
