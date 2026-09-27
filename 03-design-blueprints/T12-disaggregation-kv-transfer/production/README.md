# T12 — production reference artifacts

> `T12` · **Transcript coverage:** primary · [HLD](../HLD.md) · [LLD](../LLD.md)

**Every file here is REFERENCE-GRADE and was NOT executed in this environment.** There is
no GPU, no RDMA fabric and no pooled-memory device present. These are the artifacts a
platform team would commit; they are written to be correct-on-inspection.

| File | What it is |
|---|---|
| `pd-topology.yaml` | the prefill pool, the decode pool, and the KV transfer between them |
| `kv-tier-policy.yaml` | placement, retention and eviction policy as configuration |
| `transfer-engine.md` | why the transfer library is a swappable seam |

## The three things these files encode

1. **Prefill and decode are separate pools, independently scalable** `[T]`. The ratio
   between them is configuration, because it tracks the workload.
2. **Every KV create and every KV evict is an event a router consumes** `[T]`. A design
   that emits only creates gives the router a view that silently goes stale.
3. **Eviction is a normal lifecycle stage.** The alarm is on *thrash* — blocks evicted
   after proving reusable — not on eviction itself.

## What is deliberately absent

A pinned 2P4D ratio. The corpus presents 2P4D as the best-looking number in a set the
speaker explicitly says not to read performance from `[T]`. Sizing on it would be exactly
the error that disclaimer exists to prevent.

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — KV events as router input; CPU KV offload; precise prefix routing; the retention API PR.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — P/D pools tuned independently, KV transfer over RDMA, and the 2P4D preliminary caveat.
