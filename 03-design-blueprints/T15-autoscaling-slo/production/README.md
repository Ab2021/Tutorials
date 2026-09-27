# T15 — Production reference: autoscaling and SLOs

> `T15` · **Transcript coverage:** primary · [HLD](../HLD.md) · [LLD](../LLD.md)

**REFERENCE-GRADE, NOT EXECUTED HERE.** Nothing in this folder was applied to a cluster.
There is no Kubernetes API and no autoscaler in this environment. These are the artifacts
the design calls for, written as they would ship.

## Files

| File | What it is |
|---|---|
| `autoscaling.yaml` | per-variant autoscaling policy with derived cooldowns |
| `hpa-antipattern.yaml` | the CPU-at-70% HPA from the corpus's File A, annotated — kept as a teaching artifact |
| `slo-definitions.yaml` | SLO classes with stage ownership and shed accounting |
| `alerts.yaml` | the alerts covering the failures a self-referential loop hides |

## The three things to read first

**1. `hpa-antipattern.yaml` is in this folder deliberately.** The corpus ships an HPA
scaling an LLM service on CPU at 70% `[R]`, and another file saying to scale on KV cache
utilisation instead `[R]`. Rather than silently picking the winner, the anti-pattern is
committed with annotations showing what each line costs. `run.py` §2 puts the gap at 25
points of SLO.

It is a **valid** manifest. It applies cleanly, it will not error, and it will run in
production for months. That is exactly why it is worth keeping where someone will read it.

**2. `down_cooldown_s: derived`.** The scale-down cooldown is computed from the observed
cold start, never hard-coded. `run.py` §4: a symmetric cooldown converts 8.7% boot waste
into 29.7% and costs 6.6 points of SLO, because the controller cancels replicas that are
still booting. A hard-coded value drifts away from the boot time it exists to exceed — the
boot time changes when the model, the image or the weight store changes.

**3. `alert_if_boot_p99_exceeds_s: 120`.** Past a long enough cold start, autoscaling is
**negative-value**: `run.py` §5 shows a static peak-sized fleet beating every autoscaler on
both SLO and cost once the boot exceeds the burst duration. This alert is the tripwire that
says "stop autoscaling and pin the floor", and it is the alert most likely to be argued
with and most likely to be right.

## Why the artifacts are shaped this way

**Per-variant, never a single replica count.** Prefill and decode have different bottlenecks
and the corpus's instruction is to identify the bottlenecked side and scale that variant
`[T]`. The P:D ratio is workload-dependent `[T]`, so one number cannot serve both.

**Every SLO names the stage that owns it.** TTFT is queueing plus prefill plus transfer plus
scheduling. An end-to-end number alone cannot be acted on, because the fix differs by stage.

**Shed counters ship beside the SLOs they bias.** A percentile computed over a shedding
class describes a biased population (T14 §5), where mean wait *improves* during starvation.
The schema requires both or neither.

**Cost is a first-class signal, not a finance report.** An autoscaler is part of its own
input: scaling up shortens the queue, which makes the queue signal look healthy whether or
not the fleet is now five times too large. Cost per served request is the only metric that
sees it, which is why `alerts.yaml` alerts on it.

## What is deliberately absent

**No scale-to-zero policy.** The corpus raises zero-to-one and the speaker declines to
answer it `[T]`. With a 15–20 second cold start against sub-second TTFT budgets, the
arithmetic says it is not viable for interactive traffic — HLD §10. The config ships a warm
floor and a comment saying why, rather than a feature that would fail the first cold
request after an idle period.

**No KEDA/Karpenter node-scaling configuration.** Node provisioning adds a second dead time
on top of the pod boot, and modelling it would require cluster-specific numbers this
environment does not have. Its absence is noted so a reader does not assume pod autoscaling
is the whole loop.

## Sources

- `refs/ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/01-llm-infrastructure.md`
  — **File A**: the CPU-at-70% HPA reproduced in `hpa-antipattern.yaml`, plus the
  queue-based async architecture for high-throughput workloads.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md`
  — **File B**: KV-cache-based autoscaling; cold booting from un-quantized base images with
  weights on a high-speed mount, minutes to 15–20 seconds; heterogeneous clusters.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — saturation-based autoscaling; the workload variant autoscaler; the zero-to-one question
  and the refusal to answer it; scale the bottlenecked variant.
- `refs/gpu-perf-engineering-resources-main/gpu-perf-engineering-resources-main/README.md`
  — goodput under latency SLOs (Etalon); DistServe; ServeGen and BurstGPT.
