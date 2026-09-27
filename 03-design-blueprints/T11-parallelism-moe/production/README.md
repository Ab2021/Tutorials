# T11 — production reference artifacts

> `T11` · **Transcript coverage:** primary · [HLD](../HLD.md) · [LLD](../LLD.md)

**Every file in this directory is REFERENCE-GRADE and has NOT been executed in this
environment.** There is no GPU here, no ROCm/CUDA runtime, and no Kubernetes cluster. These
are the artifacts a platform team would commit, written to be correct-on-inspection rather
than shown-working, and they are the shape the corpus's material describes `[T]`.

| File | What it is | Where it would live |
|---|---|---|
| `vllm-args-moe.sh` | the launch flags for a WideEP MoE replica | the container entrypoint / a StatefulSet arg list |
| `topology-configmap.yaml` | the chosen sharding as a ConfigMap, so a replica can be re-shaped without a rebuild | cluster config |
| `llmd-values.yaml` | llm-d Helm values for an EP-aware, DP-attention deployment | `helm install` |
| `nic-tuning.md` | the fabric settings the topology depends on | node bootstrap |

## The shape these files encode

```
attention : DP (replicated weights, no collective across the wide dimension)   [T]
experts   : EP  (experts_per_gpu = total_experts / EP_degree)                 [T]
tensor    : confined to the node -- TP <= gpus_per_node                        [T]
pipeline  : 2                                                                  [T]
MoE path  : fused (top-k permute -> grouped GEMMs -> unpermute)                [T]
```

## Why the ConfigMap rather than baked-in flags

The HLD's §9 degradation ladder depends on the topology being *reconfigurable at runtime*.
If the sharding is a compiled constant, a fabric degradation is an outage rather than a
narrower shape. Keeping it in a ConfigMap — and having the replica derive its buffer sizes
from it (LLD §8) — is what makes the ladder real.

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — WideEP, expert parallelism, the fused kernel path, TP-inside-the-node, and the
  2P2D/2P4D configurations with the preliminary flag.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/`
  — the engine configuration surface behind these flags.
