# T13 — production reference artifacts

> `T13` · **Transcript coverage:** primary · [HLD](../HLD.md) · [LLD](../LLD.md)

**Every file here is REFERENCE-GRADE and was NOT executed in this environment.** No GPU,
no engine process and no model checkpoint exists here. These are the artifacts a platform
team would commit; they are written to be correct-on-inspection.

| File | What it is |
|---|---|
| `engine-config.yaml` | the engine's own tuning surface, with the values the HLD justifies |
| `kv-connector.yaml` | the connector to external KV storage, and the retention posture |
| `benchmark-plan.md` | how you would actually decide the parallelism configuration |

## What these files encode

1. **The pooling and scheduling values are configuration, not constants** — the engine
   should be re-tunable without a rebuild.
2. **`admissions_refused_total` is emitted split by reason.** Aggregate rejection cannot
   distinguish "add memory" from "change the admission policy".
3. **Parallelism is chosen by measurement, not inherited.** The corpus's own worked example
   replaced the naive 8-way TP baseline with a tuned mix and won on both TTFT and
   throughput per GPU `[T]` — but the same talk says there is **no universal winner**, so
   the configuration is a result, not a starting point.

## What is deliberately not here

A shipped parallelism configuration. Copying the corpus's tuned mix into a different
cluster, model and workload is exactly the error the "no universal winner" conclusion warns
against `[T]`. `benchmark-plan.md` describes the sweep that produces one instead.

## Sources

- `refs/Agentic_AI_Infra_transcripts_2/Woosuk_Kwon_-_vLLM_Building_Open_and_Efficient_Inference_for_Agents.txt`
  — the parallelism surface, the tuned-vs-naive comparison, the dynamic partitioning of a
  shared memory pool between attention types, the KV connector, and the hardware plug-in
  structure.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — the same engine on a second backend; 1P1D/2P2D policies and the nightly CI that guards
  them.
