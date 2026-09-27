# T12 — the transfer engine is a seam, not the design

> `T12` · **Transcript coverage:** primary

**REFERENCE-GRADE, NOT EXECUTED HERE.** No transfer was performed in this environment.

## The mistake this file exists to prevent

It is easy to read a description of a deployment that uses a particular transfer library
and conclude that the library *is* the architecture. It is not. The corpus names NIXL as
one supported option among several on ROCm, alongside a vendor's own GPU-initiated
transfer engine with read and write modes `[T]`. What is being designed is the **placement
and lifecycle policy**; the library only moves bytes.

| Layer | Is it the design? | Evidence |
|---|---|---|
| the fabric (RDMA NICs) | no | commodity; the deployment assumes it, does not define it |
| the transfer library | **no** — a seam | one of several supported `[T]` |
| the tier ordering | yes | decides whether offload beats recompute at all |
| the block lifecycle | yes | decides what the router can know |
| the retention policy | yes | decides whether a returning session re-prefills |

## The interface the engine must satisfy

```
transfer(block, src_tier, dst_tier) -> handle
await(handle)                          -> {bytes, elapsed, achieved_bandwidth}
```

Four properties, and they are the whole contract:

1. **The engine never chooses src or dst.** Placement is the policy's job. An engine that
   picks destinations cannot be swapped without changing behaviour.
2. **It reports achieved bandwidth.** The break-even in `sim/transfer.py` assumes a
   bandwidth; if the achieved value differs, the offload decision is being made on a
   number that is not true.
3. **Failure is an outcome, not an exception.** A failed transfer means the block is gone,
   which is an `evicted` event the router must receive.
4. **It does not emit events.** The tier manager owns the ledger and therefore owns the
   event stream. Two components emitting lifecycle events is two sources of truth.

## The pooled-memory tier, stated precisely

The corpus's rack-scale demo moves KV through a physically disaggregated pooled memory that
multiple servers share as one address space, using store/load rather than a network
protocol, and the speaker reports it as faster than RDMA `[T]`. That is a **vendor report
from a talk describing a system that vendor built**, and the speaker calls the performance
number very initial. It is recorded here at that strength and no higher.

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — KV transfer over RDMA; a GPU-initiated transfer engine with read/write modes; NIXL
  described as one supported library among several; 2P4D preliminary.
- `refs/Agentic_AI_Infra_transcripts_2/Jongryool_Kim_-_Disaggregated_LLM_Serving_with_Shared_Memory_KV_Cache_at_Rack_Sc.txt`
  — **vendor talk**: pooled-memory modes, rack-scale shared-address-space KV movement,
  store/load instead of a network protocol, "very initial" performance framing.
