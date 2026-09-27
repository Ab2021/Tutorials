# T15 — Autoscaling and SLOs: High-Level Design

> `T15` · **Transcript coverage:** primary · [LLD](LLD.md) · [Sequences](docs/SEQUENCES.md) · [Production](production/README.md)

How a fleet of inference replicas decides its own size, and what it promises while doing
it. The subject looks like capacity planning and is actually control theory with an
unusually long dead time: **a replica that starts now serves nothing for the next fifteen
seconds to several minutes**, and every design decision below follows from that.

This is the system-level view. The [LLD](LLD.md) is the component-level view. `run.py`
models the control loop end to end.

---

## 1. The corpus contradicts itself, and that is the finding

Two guide files in the corpus give incompatible autoscaling advice.

**File A** — `11-infrastructure-and-mlops/01-llm-infrastructure.md` — ships a Kubernetes
HorizontalPodAutoscaler for an LLM service scaling on **CPU utilisation at a 70% target**
and requests-per-second `[R]`.

**File B** — `04-inference-optimization/06-serving-infrastructure.md` — says the opposite
in one line: autoscaling should be based on **KV cache utilisation rather than CPU or
standard memory usage** `[R]`.

This is recorded as a contradiction, not resolved. Both files are in the corpus, both are
prescriptive, and an operator following the more detailed and more concrete of the two
gets the worse outcome.

The model quantifies the gap. At the corpus's own best-case cold start of 15–20 seconds:

| signal | source | SLO attainment | replica-steps |
|---|---|---|---|
| CPU at 70%, the guide's signal | File A | 74.6% | 2,320 |
| queue/KV-shaped | File B | **100.0%** | 4,804 |

**A 25-point SLO gap, from the choice of signal alone.** But the interesting part is *why*,
because there are two independent causes and they have different fixes:

**Cause 1 — compression.** In LLM serving the CPU orchestrates (tokenisation, batching,
HTTP) while the accelerator does the work. CPU utilisation therefore rises slowly and
saturates well below 100%, only loosely coupled to whether requests are meeting their
budget. In the model it tops out at 0.85 even when the fleet is hopelessly behind, so a
0.70 target is reached only at ~2.5× overload `[D]`.

**Cause 2 — the control law.** The HPA computes `desired = ceil(current × signal / target)`.
That moves **geometrically**: it approaches the setpoint rather than jumping to it. A fleet
far from its setpoint is slow even with a perfect signal.

These are separable, and the model separates them: `cpu_ideal` is a CPU signal that tracks
load linearly with no lag and no ceiling. It reaches 100% SLO — and costs **1.98× the
static baseline**, with 13.7% of the bill spent on replicas that were never ready. **Fixing
the signal does not fix the loop.** That is the finding File A's readers need and cannot
get from either guide.

## 2. The mechanism: a control loop with dead time

```
        ┌──────────────┐   signal   ┌───────────┐  desired  ┌──────────────┐
        │   workload   │───────────▶│ controller │──────────▶│  provisioning │
        └──────┬───────┘            └───────────┘           └──────┬───────┘
               │                                                   │
               │              ┌──────────────┐                     │
               └─────────────▶│   replicas   │◀────────────────────┘
                              └──────┬───────┘
                                     │  BOOT DELAY
                                     └──▶ capacity arrives
                                          C steps late
```

Every quantity that matters is a delay:

| Delay | What it is | Corpus figure |
|---|---|---|
| **cold start** | replica created → able to serve | 15–20 s from un-quantized base images with weights on a fast mount, down from minutes `[R]` |
| **signal lag** | load rises → the metric reflects it | the metric's own averaging window |
| **control period** | the loop's tick interval | the HPA's sync period |
| **cooldown** | hysteresis preventing flapping | policy, and §4 shows it is the one that bites |

The corpus adds a further complication that changes the shape of the problem: the fleet is
**not one pool**. With prefill/decode disaggregation, the operator must "identify where the
bottleneck is — whether your prefill is bottlenecked or the decoder is bottlenecked — and
accordingly scale up either of these variants" `[T]`. A single replica count cannot serve a
workload whose P:D ratio moves, and the corpus's own P:D study found the answer **dependent
on the workload**, with prefill-heavy work wanting more prefill `[T]`.

## 3. Signals: what to scale on

| Signal | Coupling to user-visible pain | Notes |
|---|---|---|
| CPU utilisation | poor | compressed and lagged; the File A choice |
| GPU utilisation | poor | saturates at 100% under any load that queues |
| requests/second | moderate | ignores request *cost*; a 4k-token and a 400k-token request count the same |
| **queue depth / waiting requests** | **direct** | the thing the SLO is actually about |
| **KV cache utilisation** | **direct** | the corpus's recommendation `[R]` |
| **time-to-first-token** | **direct** | the user-visible metric itself |

The corpus's deployment exposes saturation-based autoscaling as a feature — the
**workload variant autoscaler** — which deploys and scales Kubernetes pods up and down
against usage in real time `[T]`. Note the framing: *saturation*-based. The threshold is
the same operator-defined constant T14 uses for flow control, and it is the same design
object seen from the other side.

**The signal and the symptom must be the same thing.** If the scaling signal is KV
occupancy and the workload is prefill-heavy with fast turnover, the cache may never fill
while the fleet queues. This failure is silent and it is the reason the metric choice is a
design decision rather than a tuning parameter.

## 4. The rule that matters most: scale-down must outlast the cold start

The model's most operationally valuable output is a rule, not a measurement.

**If scale-down is as fast as scale-up, the controller cancels replicas that are still
booting.** They were paid for, they never served, and the capacity the signal asked for
never arrives. The controller eats its own tail.

`run.py` §4 holds the policy, signal and workload fixed and varies only the down-direction
hysteresis:

| scale-down hysteresis | SLO attainment | boot waste |
|---|---|---|
| naive (same as scale-up) | 93.4% | 29.7% |
| derived (2 × cold start) | **100.0%** | **8.7%** |

The cost goes *up* slightly (4,304 → 4,804 replica-steps) because the derived cooldown
holds capacity longer. That is the honest trade, and it identifies the right metric:
**total spend is the wrong lens; spend per *served* request is the right one.**

The general form of the rule: **the scale-down cooldown must exceed the cold start.**
Deriving it from the measured boot time rather than hard-coding it is the single most
valuable line in the design.

## 5. When autoscaling loses to overprovisioning

The model's sharpest result, and the one most likely to be argued with:

| cold start | best autoscaler | static overprovision |
|---|---|---|
| 5 steps | 94.2% SLO at 2,015 | 100% at 2,700 |
| 20 steps | 100% at 4,804 | 100% at 2,700 |
| 60 steps | 76.8% at 9,653 | **100% at 2,700** |
| 180 steps | 37.1% at 17,266 | **100% at 2,700** |

**Past a long enough cold start, a statically overprovisioned fleet beats every autoscaler
on both SLO and cost.** The static fleet's replicas were already running when the burst
arrived; the autoscalers pay for replicas that arrive after the need has gone and are then
billed anyway.

This is not fixable by a better policy — it is structural. An autoscaler's value
proposition is that it adds capacity when load rises. **If the load rises and falls inside
the boot window, the capacity arrives after the need.**

The design rule: **autoscaling only pays when the cold start is short relative to the burst
duration.** Which reframes cold-start reduction — the corpus's un-quantized base images and
fast weight mounts cutting startup from minutes to 15–20 seconds `[R]` — as a
**prerequisite for autoscaling**, not a separate optimisation. File A's HPA is not merely
on the wrong signal; it is a control loop installed on a plant with a long dead time.

## 6. What an SLO is, for this workload

Latency SLOs for generative serving are per-request and two-dimensional:

- **TTFT** — time to first token. Dominated by prefill and by queueing.
- **ITL / TPOT** — inter-token latency. Dominated by decode batch composition.
- A **goodput** objective — requests meeting both, per unit time — is the standard framing
  in the literature the corpus points at (Etalon, DistServe) `[R]`.

Three properties make these different from a typical web SLO:

**A latency SLO here is a *composition*.** TTFT is queueing plus prefill plus transfer;
ITL is scheduling plus decode. Hitting the end-to-end number tells you nothing about which
stage to fix, which is why T13's and T14's per-stage metrics are load-bearing.

**Percentiles hide the failure that matters.** Under a shedding policy the surviving
requests are not a random sample (T14 §5). **Survivorship bias is the default, not the
exception**, and any percentile reported beside a shed counter is describing a biased
population.

**SLO attainment and cost must be read together.** Every table in this blueprint has an
SLO column and a cost column, because either alone is trivially gameable: overprovision
and the SLO is perfect; scale to the floor and the cost is minimal.

## 7. The fleet is not one pool

Replica count is a scalar, and the fleet it describes is not. Two independent
decompositions are already in play by the time this blueprint is reached, and both make
"how many replicas?" the wrong question.

**Prefill and decode scale separately.** §2: the corpus's instruction is to identify which
side is bottlenecked and scale that variant `[T]`. They have different bottlenecks
(prefill is compute-bound, decode is memory-bandwidth-bound), different optimal hardware
ratios, and a ratio that moves with the workload. A single replica count forces the
operator to size for the worse of the two at all times.

**Model tiers scale separately, and the router decides the mix.** The corpus describes
heterogeneous clusters mixing frontier accelerators with small ones in the same cluster
`[R]`, and T14's semantic router classifies requests into simple/medium/reasoning before a
model is chosen `[T]`. So the fleet is a *set* of pools with different per-replica costs,
and the arrival rate at each pool is an output of the classifier, not of the workload.

That makes autoscaling hierarchical, and it introduces a failure the single-pool model
cannot express: **if the difficulty classifier's output distribution drifts, the tier that
receives the traffic is not the tier that was scaled for.** The autoscaler sees each pool's
own signal and reacts correctly to a demand shift that is really a classification change.
The symptom is one pool permanently saturated and another permanently idle, while every
individual scaling decision looks right.

The design consequence is that the tier mix belongs on the dashboard next to the replica
counts. A per-pool autoscaler cannot detect a routing change, and T14's `view_staleness`
has no equivalent here.

## 8. Observability and the autoscaling feedback path

The dangerous property of an autoscaler is that **it is part of its own input**. A scaling
decision changes queue depth, which changes the signal, which changes the next decision.
Two consequences:

- **An autoscaler can hide its own failure.** If it scales up enough to keep the queue
  short, the queue signal looks healthy — even if the fleet is now five times too large.
  Cost is the only metric that reveals it.
- **The loop can be driven by its own oscillation.** Flapping replicas change the signal
  that decides the next replica count. Damping (cooldown, hysteresis, rate limits) is a
  stability requirement, not an optimisation.

Required metrics, beyond the SLO set:

| Metric | Why |
|---|---|
| `replica_steps_total` | cost, counting **booting** replicas — the number a ready-count hides |
| `replica_boot_waste_ratio` | share of spend on replicas that never served §4 |
| `scale_events_total{direction}` | flapping; a rising count with flat load is instability |
| `cold_start_seconds` | the term that dominates every decision in this blueprint |
| `queue_depth_p99` | the leading indicator, before TTFT moves |
| `slo_attainment{class}` + `shed_total{class}` | read together, always (T14 §5) |

## 9. Scope of the model, and what it does not claim

`sim/autoscaler.py` is a control-loop model with one workload, one replica class and one
boot delay. **No cluster scaled and no replica booted.** Every figure is a simulation
output.

Two assumptions are load-bearing and are stated in the source: CPU utilisation is modelled
as a compressed, lagged function of load, and the HPA's control law is modelled as the
geometric `ceil(current × signal / target)`. `run.py` §2 runs the variant where the CPU
signal is perfect, so the conclusion that *the loop, not just the signal, is slow* does not
depend on the compression assumption. The §3 and §5 findings do not depend on it at all.

## 10. Design decisions worth defending

1. **Scale on queue depth or KV occupancy, never on CPU.** File B over File A, with the
   model's 25-point gap as the reason.
2. **Fix the cold start before tuning the autoscaler.** §5: without it, autoscaling is
   negative-value.
3. **Derive the scale-down cooldown from the boot time.** §4.
4. **Scale prefill and decode independently.** The corpus's bottleneck-identification
   framing `[T]`; one replica count cannot serve a moving P:D ratio.
5. **Predict across the boot window rather than reacting.** `run.py` §3: 53% cheaper than
   reacting to the current queue, and the finding holds for any signal.
6. **Report cost and SLO together, and report sheds.** §6.
7. **Keep a warm floor.** Scale-to-zero is not solved — §10.

## 11. Scale to zero: an honest gap

The corpus contains a practitioner asking exactly the right question and a speaker
declining to answer it `[T]`:

> "How do you handle the zero to one like we do autoscaling right? … we cannot scale from
> 0 to 1, only from 1 to n."
>
> "I'm not very much aware of that path, the autoscaling path… I'm not an expert at that so
> I can't answer."

**The gap is recorded as a gap.** No answer is fabricated. What the model contributes is
why it is hard, as arithmetic anyone can check `[D]`: with a cold start of 15–20 seconds,
the first request after idle pays 15–20 seconds before its first token, while interactive
TTFT budgets are sub-second to a few seconds. **Scale-to-zero is viable only when the cold
start is under the TTFT budget — which for interactive traffic it is not.**

The honest position is therefore not "implement scale-to-zero" but:

- scale to a **warm floor** sized for the p99 of the idle-period arrival rate;
- treat zero as a **batch-tier** option only, where no TTFT promise exists;
- if zero must be offered, put a queue in front of it and promise throughput, not latency.

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — saturation-based autoscaling exposed as a feature of llm-d; the workload variant
  autoscaler scaling Kubernetes pods up and down on real-time usage; the zero-to-one
  question and the speaker's explicit refusal to answer it; identify the prefill-vs-decode
  bottleneck and scale that variant; autoscaling as one of the reasons the problem becomes
  cluster-level rather than node-level; the P:D ratio being workload-dependent.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/01-llm-infrastructure.md`
  — **File A**: the HorizontalPodAutoscaler scaling on CPU utilisation at 70% and
  requests_per_second. Reproduced as the contradiction in §1, not as guidance.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md`
  — **File B**: "Scaling based on KV Cache utilization rather than CPU or standard memory
  usage"; cold booting from un-quantized base images with weights on a high-speed mount
  taking startup from minutes to 15–20 seconds; heterogeneous clusters mixing H100s and
  L4s; the inference gateway's auth/rate-limiting, model-router, context-tracker and
  output-filter responsibilities; Layer 7 load balancers that understand the
  end-of-sequence token and rebalance between user turns.
- `refs/gpu-perf-engineering-resources-main/gpu-perf-engineering-resources-main/README.md`
  — Etalon (TTFT, TPOT, goodput and latency SLOs), DistServe (separate prefill and decode
  workers optimised for goodput under latency constraints), ServeGen and BurstGPT (workload
  generation preserving production-trace properties), MLPerf Endpoints.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — the autoscaling path exercised against a second backend.
