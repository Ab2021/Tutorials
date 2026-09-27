# T11 — fabric tuning the topology depends on

> `T11` · **Transcript coverage:** primary

**REFERENCE-GRADE — NOT EXECUTED HERE.** These are node-bootstrap settings, applied before
any container starts. They are documented because the chosen topology *assumes* them: a
WideEP shape whose all-to-all rides a misconfigured fabric does not fail loudly, it just
runs slower, and the regression is easy to misattribute to the model.

## Why this file exists in a design blueprint

A blueprint that says "EP=16 across two nodes" without saying what the link between those
nodes must do has hidden its most fragile assumption. The all-to-all is not a constant of
the design — it is a dependency, and it is the first thing to degrade `[D]`.

## The three settings that matter

| Setting | Effect if wrong | Why the topology cares |
|---|---|---|
| GPUDirect RDMA (`NCCL_NET_GDR_LEVEL`) | every all-to-all bounce goes through host memory | the MoE layer's all-to-all is on the critical path once per layer `[D]` |
| Jumbo frames / MTU | per-message overhead rises with message count, and EP raises message count | all-to-all peers scale with EP degree `[D]` |
| Adaptive routing on the leaf-spine | a hot expert's traffic pins one uplink | expert load is skewed; top-k routing concentrates traffic `[D]` |

## The invariant to test, not assume

```
intra-node  bandwidth  >>  inter-node bandwidth
```

Tensor parallelism lives entirely inside this first tier. If a TP group is ever scheduled
across a node boundary, its collective is promoted onto the second tier and the replica's
latency stops tracking the GPU — which is why the enumerator in `sim/planner.py` cannot
emit `tp > gpus_per_node`, and why the topology ConfigMap carries
`requireNodeLocalTpGroup: "true"`.

**What the corpus says, precisely:** tensor parallelism stays *inside the node*; "TP stays
1" is **not** the rule `[T]`. The ranked shape in `run.py` §3b uses TP=8 and is legal.

## Verification before promotion

1. Bandwidth test between two nodes hosting one replica's EP group.
2. Bandwidth test between two GPUs in the same node.
3. Assert the ratio exceeds the design's assumption; if it does not, the topology is
   wrong, not the fabric.
4. Re-run the long-context sweep (`docs/SEQUENCES.md`, flow 4) after any fabric change —
   the all-to-all is inside the latency path the sweep measures.

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — WideEP, expert parallelism, all-to-all placement, and the TP-inside-the-node rule.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — llm-d's treatment of the fabric as the scarce shared resource.
