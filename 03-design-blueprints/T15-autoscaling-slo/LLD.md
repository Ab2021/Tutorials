# T15 — Autoscaling and SLOs: Low-Level Design

> `T15` · **Transcript coverage:** primary · [HLD](HLD.md) · [Sequences](docs/SEQUENCES.md) · [Production](production/README.md)

Component-level design of the autoscaling controller and the SLO contract it serves: the
loop, the signals, the damping, the per-variant decomposition, the SLO objects, and the
observability that makes a self-referential control loop legible.

Everything here is a design. **No cluster scaled and no replica booted**, and no figure
below is a measurement.

---

## 1. Component inventory

| Component | Responsibility | Stateful? |
|---|---|---|
| `MetricSource` | expose the candidate signals per pool | no |
| `Controller` | evaluate the policy, emit desired replica counts | yes (history, cooldowns) |
| `Policy` | signal → desired count, per pool | no (pure) |
| `Actuator` | create/delete replicas, own the boot lifecycle | yes (in-flight boots) |
| `Damping` | cooldowns, rate limits, stability guards | yes (last-scale timestamps) |
| `SLOEvaluator` | attainment per class, with shed accounting | yes (windows) |
| `TierMixMonitor` | observed traffic mix across model tiers | yes (windows) |

`Controller` is **per pool, not global**. HLD §7: prefill, decode and each model tier scale
independently, and a global controller cannot express a bottleneck that lives in one of
them.

## 2. The control loop, concretely

One tick:

```python
def tick(pool, now):
    # 1. OBSERVE -- from the pool's MetricSource
    sig = pool.metrics.sample(now)

    # 2. DECIDE -- pure function of signal + policy state
    desired = pool.policy.decide(sig, pool.state, pool.config)

    # 3. DAMP -- asymmetric: up fast, down slow (section 4)
    direction = "up" if desired > pool.current else "down"
    if not pool.damping.permits(direction, now):
        return
    delta = pool.damping.rate_limit(desired - pool.current)
    if delta == 0:
        return

    # 4. ACT -- the actuator owns the boot lifecycle, not the controller
    pool.actuator.reconcile(pool.current + delta, now)
    pool.damping.record(direction, now)
```

Three properties that the design depends on:

**The policy is pure.** `decide` takes signal, state and config and returns a number. This
is what makes a policy testable against a replayed trace without a cluster, and it is why
every policy in `sim/autoscaler.py` shares one signature.

**The actuator owns the boot lifecycle.** The controller reasons about *desired* replicas;
only the actuator knows which are still booting. HLD §5's central finding — booting
replicas cost money and serve nothing — is only representable if that state lives
somewhere, and it belongs with the thing that creates pods.

**Damping sits between decide and act, not inside the policy.** Every policy needs the same
hysteresis, and the damping rule differs by *direction* — which a per-policy implementation
would get wrong independently.

## 3. Signals, and the two failure shapes

```
Signal {
  name        : "cpu" | "gpu" | "queue_depth" | "kv_used_fraction" | "ttft_p99"
  value       : float
  window      : seconds        # the averaging window -- a delay
  saturation  : float          # the value at which the pool is deemed full
}
```

Two failure shapes, and they are different problems:

**Compression — the signal cannot reach its own target.** `run.py` §2: CPU modelled with a
0.85 ceiling against a 0.70 target is reached only at ~2.5× overload. A signal whose range
does not span the target range cannot regulate; the loop will sit at its setpoint while the
fleet is failing. Detected by checking whether the signal ever *approaches* its target in
production, which is a month-one question nobody asks.

**Misalignment — the signal moves for reasons other than the thing being promised.** GPU
utilisation pins at 100% under any queuing load, so it cannot distinguish "healthy and
busy" from "collapsing". Requests/second ignores request cost, so a 4k-token request and a
400k-token request count the same — and the corpus's own work is entirely about long
context `[T]`.

The corpus's recommendation, KV cache utilisation `[R]`, is neither compressed nor
misaligned for this workload: every in-flight request holds KV for its whole life, so the
signal tracks the queue.

**The signal must be per pool, and the corpus's own recommendation has a caveat.**
Prefill and decode hold KV differently, and a prefill-heavy deployment with fast turnover
may never fill the cache while the fleet queues. HLD §3: the signal and the symptom must be
the same thing.

## 4. Damping: the rule that this blueprint exists to state

```
Damping {
  up_cooldown   : max(15s, 1 x boot_p50)      # scale up promptly
  down_cooldown : max(60s, 3 x boot_p99)      # NEVER shorter than the cold start
  max_up_step   : 2x current                  # the HPA's own geometric limit
  max_down_step : 25% of current              # down is the dangerous direction
  min_replicas  : the warm floor (section 7)
}
```

**`down_cooldown` is derived from the cold start, never hard-coded.** `run.py` §4 holds
policy, signal and workload fixed and varies only this term:

| scale-down hysteresis | SLO | boot waste |
|---|---|---|
| naive (same as up) | 93.4% | 29.7% |
| derived (2 × boot) | 100.0% | 8.7% |

The mechanism: with a short down-cooldown, the controller cancels replicas **that are still
booting**. They were paid for, they never served, and the capacity the signal asked for
never arrives. Total spend rises slightly under the derived rule (4,304 → 4,804
replica-steps) because capacity is held longer — which is why the metric has to be *spend
per served request*, not spend.

**Why the asymmetry is not just a bigger number.** A symmetric cooldown large enough to
protect scale-down would also delay scale-up, which is the direction where delay costs SLO.
One number cannot serve both directions, which is why the config surface has two.

## 5. Policies

| Policy | Signal | Lookahead | Where it fails |
|---|---|---|---|
| `static_min` | none | none | SLO is a function of the workload, not the fleet |
| `static_peak` | none | sized for peak | pays peak cost permanently — **and wins anyway when the cold start is long** |
| `cpu` | CPU | none | compressed; the HPA's own geometric control law |
| `queue` | queue + inflight | none | always `boot_steps` late; oscillates |
| `queue_predict` | queue + inflight | **+ boot window** | overshoots if the ramp estimate is noisy |

**The predictor is the one to ship when the cold start is short.** `run.py` §3: same signal,
same workload, projected forward across the boot window using the observed arrival ramp —
**53% cheaper** than reacting to the current queue (2,235 vs 4,804 replica-steps), at 85.6%
versus 100% SLO. It is the cheapest policy in the model that tracks the workload at all.

**The ramp estimate must be sampled, not differenced.** The implementation detail that took
two attempts in `sim/autoscaler.py` and is worth carrying into production: a ramp computed
as the difference between two adjacent samples is a **pulse** — large for exactly one
window during a step change, then gone. A controller built on it over-scales, then
immediately reverts and cancels the replicas it just started booting. The model samples the
arrival rate every `window` steps and fits the trend across several samples.

**`static_peak` is a real contender, not a straw man.** HLD §5: past a long enough cold
start it beats every autoscaler on both SLO and cost. A design review that treats
overprovisioning as obviously wasteful has not read the boot-time column.

## 6. The SLO contract

```
SLOClass {
  name        : "premium" | "best_effort"
  ttft_p99_ms : int
  itl_p99_ms  : int
  budget      : which stage owns the budget   # queue / prefill / transfer / decode
  shed_policy : reference to the flow-control class in T14
}
```

Three rules the schema enforces:

**Every SLO declares which stage owns it.** TTFT is queueing plus prefill plus transfer
(T12) plus scheduling (T13). An SLO that names only the end-to-end number cannot be acted
on, because the fix differs by stage and the number does not say which stage is failing.

**Every SLO that can be shed carries its shed counter.** T14 §5: under priority bands the
surviving requests are not a random sample, and best-effort's mean wait *improves* while it
is being starved. **Survivorship bias is the default.** A percentile reported without a shed
rate beside it is describing a biased population.

**Goodput, not throughput, is the capacity unit.** The literature the corpus points at
frames it exactly this way — requests meeting their latency constraint per unit time
(Etalon, DistServe) `[R]`. Throughput without a latency constraint is trivially maximised by
never admitting anything.

## 7. The warm floor

```
min_replicas = max(
    warm_floor_for_p99_idle_arrival,     # sized from the idle-period arrival p99
    replicas_needed_for_one_request,     # a fleet of 0 cannot serve 1
)
```

`min_replicas` is the answer to scale-to-zero, and it is a deliberate refusal rather than a
feature. HLD §10: with a 15–20 second cold start the first request after idle pays 15–20
seconds before its first token, against interactive TTFT budgets of sub-second to a few
seconds. **Scale-to-zero is viable only when the cold start is under the TTFT budget.**

Sizing rule, with its assumption visible `[D]`: over the idle period, size for the p99 of
the per-step arrival rate rather than the mean, because the mean floor is exceeded by
definition half the time and the cost of exceeding it is a full cold start.

## 8. Per-variant decomposition

```
Variant {
  name          : "prefill" | "decode" | "tier-small" | "tier-frontier"
  replicas      : independent
  policy        : independent
  signal        : independent        # prefill: queue/prefill-tokens; decode: kv/active
  boot_seconds  : independent        # weights differ; a small tier boots faster
}
```

**Prefill and decode must not share a replica count.** The corpus's instruction is to
identify the bottlenecked side and scale that variant `[T]`, and the P:D study found the
right ratio **workload-dependent** `[T]`.

**Tiers must not share one either, and their traffic mix is not stable.** HLD §7: with a
semantic router choosing between model tiers (T14 §6), the arrival rate at each pool is an
output of the classifier. If the classifier's output distribution drifts, one pool
saturates and another idles while every individual scaling decision is correct.

The design consequence: **`tier_mix` is a first-class metric**, and a divergence between
the mix the fleet was scaled for and the mix it is receiving is an alert. A per-pool
autoscaler structurally cannot detect a routing change.

## 9. Failure handling

| Failure | Detection | Response |
|---|---|---|
| cold start longer than the burst | `boot_waste_ratio` climbing | stop autoscaling; pin to a static peak-sized floor |
| flapping | `scale_events_total` rate | widen cooldowns; check `down_cooldown > boot_p99` |
| signal never reaches target | saturation signal max over 24h | the signal is compressed or misaligned; change it |
| queue grows, signal healthy | queue p99 vs signal value | the signal measures something else than the symptom |
| tier mix drift | `tier_mix` vs scaled-for mix | the classifier changed, not the workload |
| autoscaler masks its own over-scale | cost per served request rising, SLO healthy | read cost and SLO together; SLO alone cannot see it |

The last row is the one that is unique to this component: **an autoscaler is part of its
own input**. Scaling up shortens the queue, which makes the queue signal look healthy
regardless of whether the fleet is now five times too large. Cost is the only thing that
reveals it, which is why every table in HLD carries a cost column beside the SLO column.

## 10. Configuration surface

```yaml
autoscaling:
  variants: [prefill, decode]        # NEVER a single replica count for both
  pools:
    - name: prefill
      signal: {name: queue_depth, window_s: 15}
      policy: queue_predict
      min_replicas: 4                # the warm floor, section 7
      max_replicas: 48
    - name: decode
      signal: {name: kv_used_fraction, window_s: 15}
      policy: queue
      min_replicas: 4
      max_replicas: 64
  damping:
    up_cooldown_s: 15
    down_cooldown_s: derived         # = 3 x observed boot_p99. Do not hard-code.
    max_down_step_fraction: 0.25
  cold_start:
    boot_p50_s: 18                   # measured, not assumed
    boot_p99_s: 45
    alert_if_boot_p99_exceeds_s: 120 # beyond this, autoscaling is negative-value
  slo:
    classes: [premium, best_effort]
    report: [attainment, shed_rate, cost_per_served_request]
```

The comment on `down_cooldown_s` is the one that must not be edited out by a later reader:
**it is derived from the cold start because a hard-coded value drifts away from the boot
time it is supposed to exceed.**

## 11. Observability

| Metric | Type | Why |
|---|---|---|
| `slo_attainment{class}` | gauge | with the shed rate, never alone |
| `shed_total{class,reason}` | counter | T14 §5; unaggregated |
| `replica_steps_total{pool}` | counter | cost, **including booting replicas** |
| `replica_boot_waste_ratio{pool}` | gauge | the share of spend that never served — §4 |
| `scale_events_total{pool,direction}` | counter | flapping; rising with flat load is instability |
| `cold_start_seconds{pool}` | histogram | the term that dominates every decision here |
| `queue_depth_p99{pool}` | gauge | the leading indicator, before TTFT moves |
| `tier_mix{observed}` vs `tier_mix{scaled_for}` | gauge | §8; a routing change a per-pool loop cannot see |
| `cost_per_served_request` | gauge | the only metric that reveals §9's last row |

## 12. What the model does and does not show

`sim/autoscaler.py` is a control-loop model: one workload, one replica class, one boot
delay, Poisson arrivals, a plateau burst.

Two assumptions are load-bearing and are stated in the source rather than buried:

- **CPU is modelled as compressed and lagged.** The rationale is that the CPU orchestrates
  while the accelerator works. `run.py` §2 runs `cpu_ideal` — a CPU signal that tracks load
  linearly with no lag and no ceiling — so the conclusion that *the loop, not only the
  signal, is slow* does not depend on this.
- **The HPA control law is modelled as `ceil(current × signal / target)`.** This is the
  published HPA behaviour, and it is the reason a far-from-setpoint fleet is slow even with
  a perfect signal.

The §3 (prediction) and §5 (overprovisioning crossover) findings do not depend on either
assumption. The absolute numbers depend on the workload shape and should be read as
directional. Nothing here was tuned to produce a preferred answer: the CPU signal was
initially modelled so compressed that it never fired at all, which made the comparison
degenerate and was corrected to a signal that fires late but does fire.

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — saturation-based autoscaling as a feature; the workload variant autoscaler scaling
  Kubernetes pods on real-time usage; the zero-to-one question and the explicit refusal to
  answer it; identify the prefill-vs-decode bottleneck and scale that variant; autoscaling
  as a reason the problem becomes cluster-level.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/06-serving-infrastructure.md`
  — "Scaling based on KV Cache utilization rather than CPU or standard memory usage"; cold
  booting from un-quantized base images with weights on a high-speed mount, minutes to
  15–20 seconds; heterogeneous clusters mixing H100s and L4s; the inference gateway's four
  responsibilities; Layer 7 load balancers that understand the end-of-sequence token.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/01-llm-infrastructure.md`
  — the CPU-at-70% HorizontalPodAutoscaler, cited as the contradictory position (HLD §1),
  not as guidance.
- `refs/gpu-perf-engineering-resources-main/gpu-perf-engineering-resources-main/README.md`
  — Etalon (TTFT, TPOT, goodput, latency SLOs); DistServe (prefill/decode split optimised
  for goodput under latency constraints); ServeGen and BurstGPT (production-trace-shaped
  workload generation); MLPerf Endpoints.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — the autoscaling path against a second backend.
