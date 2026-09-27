# T13 — How you actually choose the parallelism configuration

> `T13` · **Transcript coverage:** primary

**REFERENCE-GRADE, NOT EXECUTED HERE.** No benchmark in this document was run; there is no
GPU in this environment. This is the plan, not the result.

## Why this file exists

The corpus's most quotable result for this topic is a tuned configuration beating a naive
one: for a large MoE prefill on B200 in a disaggregated setup, a mix of tensor, pipeline,
sequence and expert parallelism across 16 GPUs per replica beat the straightforward 8-way
tensor-parallel single-host baseline on **both** time-to-first-token and throughput per GPU
`[T]`. Each dimension earned its place for a stated reason — pipeline gives parallelism
between chunks of a long prefill sequence, sequence parallelism gives more
communication/computation overlap, expert parallelism gives better GEMM shapes than 8-way
TP `[T]`.

The same talk draws the opposite conclusion from that result: the parallelism must be
selected for the target model architecture, the target cluster and the target workload
shape, and **there is no universal winner** `[T]`. So the correct artifact to ship is not
that configuration. It is this plan.

## The sweep

| Step | What varies | Held fixed | Read |
|---|---|---|---|
| 1 | TP degree, 1..gpus_per_node | PP=1, EP=1 | the baseline; find where it stops improving |
| 2 | PP depth | best TP | where the bubble stops being hidden |
| 3 | EP degree | best TP×PP | whether the model even fits without it, then GEMM shape |
| 4 | sequence parallelism on/off | best of the above | the overlap it buys |
| 5 | the full mix across N GPUs | — | against the 8-way TP baseline from step 1 |

Steps 1–4 exist to make step 5 a shortlist rather than a search of the whole space.
`sim/planner.py` in `T11-parallelism-moe/` enumerates structurally valid shardings and
ranks them; use it to generate the shortlist, then measure.

## The three gates before promotion

1. **Correctness.** A parallel configuration that changes outputs is not a configuration.
   Compare against a single-device reference on a fixed prompt set.
2. **Long-context validation at the target shape.** The sweep must include the longest
   context the deployment will serve — a configuration that wins at 4k and collapses at
   128k is a regression, not a tuning.
3. **The workload's own mix.** Prefill-heavy and decode-heavy workloads favour different
   configurations; the corpus's own P:D comparison found the answer *dependent on the
   workload*, with prefill-heavy work wanting more prefill `[T]`.

## What to write down

For every configuration promoted, record:

- the model, cluster and workload shape it was tuned for (all three, because it is tuned
  for all three);
- the configuration it replaced and the measured delta;
- the sweep cells *not* run, so the next person knows what was assumed.

That last line is the one teams skip, and it is the one that turns a tuning result into a
folklore result six months later.

## Sources

- `refs/Agentic_AI_Infra_transcripts_2/Woosuk_Kwon_-_vLLM_Building_Open_and_Efficient_Inference_for_Agents.txt`
  — the 8-way TP baseline versus the tuned 16-GPU mix on B200 prefill; why pipeline,
  sequence and expert parallelism each contributed; "there's no universal winner" and the
  requirement for a performance model; seven kinds of parallelism including decode context
  parallelism around the KV cache.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — the same tuning problem on a second backend; nightly CI covering 1P1D and 2P2D; the
  2P4D result presented as preliminary and explicitly not to be sized on.
