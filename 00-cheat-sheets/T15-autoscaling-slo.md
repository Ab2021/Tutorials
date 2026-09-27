# Cheat Sheet: Capacity, Autoscaling & SLOs

> `T15` · **Transcript coverage:** primary · [Case study](../01-case-studies/T15-autoscaling-slo.md) · [Blueprint](../03-design-blueprints/T15-autoscaling-slo/HLD.md) · [Interview bank](../02-interview-questions/T15-autoscaling-slo.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| Concurrency on a tuned 1P1D pair | **~20,000** | `[T]` ROCm/WideEP |
| Saturation threshold example | KV cache **80% full** | `[T]` llm-d |
| Second saturation signal | **average active requests > 8** | `[T]` llm-d |
| KV fill target in production | fill to **90% of VRAM** | `[T]` NxtGen |
| The cliff to plan around | **28k inputs** ⇒ KV recomputation ⇒ throughput drop at concurrency 256 | `[T]` ROCm/WideEP |
| Prefill share of agentic tokens | **~98%** | `[T]` llm-d |
| Fairness win from session-level scheduling | request latencies cut **2–3×** | `[T]` llm-d |
| GPU utilisation ceiling you can actually plan for | well below 100% — KV is the binding constraint, not FLOPs | `[D]` |

---

## The one-table summary

| Scaling approach | Signal | Works for LLM serving? | Notes |
|---|---|---|---|
| **CPU/GPU utilisation HPA** | device utilisation | **poorly** | a busy LLM pod sits near 100% util while still healthy `[D]` |
| **Queue-depth / in-flight autoscaling** | pending requests | **yes** | the closest analogue to a real load signal `[D]` |
| **KV-saturation autoscaling** | KV used % | **yes — best signal** | 80% is the corpus's example `[T]` |
| **Goodput-driven autoscaling** | SLO-attainment rate | **yes, best objective** | scale on the metric that pays the bills `[D]` |
| **KEDA / event-driven 0→1** | queue/event source | yes, but cold start is brutal | weight loading dominates the scale-up `[D]` |
| **Predictive / scheduled scaling** | traffic pattern | yes for known peaks | agents have bursty, long-lived sessions `[T]` |
| **Gang scheduling** | — | **required for multi-node replicas** | partial allocation deadlocks `[D]` |
| **Admission control / load shedding** | saturation | yes — pair with autoscaling always | degrade deliberately, not accidentally `[T]` |
| **Overprovision + queue** | — | the honest default | LLM autoscaling is slower than traffic changes `[D]` |

**The core asymmetry** `[D]`: LLM pods take **minutes** to become useful (download and load tens of
GB of weights, warm the cache, compile kernels) while traffic can spike in **seconds**. Autoscaling
alone cannot save you; you need admission control and a queue in front. **Plan capacity for the
peak, and autoscale for the trend.**

---

## Formulas

**SLO definitions you must fix before any capacity math** `[D]`:
```
TTFT_p95 ≤ T_t   (time to first token — prefill-bound)
ITL_p95  ≤ T_i   (inter-token latency — decode-bound)
goodput  = |{r : TTFT(r) ≤ T_t AND ITL(r) ≤ T_i}| / |requests|
```
Two separate SLOs because two different resources drive them. A single "latency" SLO hides which
pool to scale — a mistake that shows up as scaling the wrong tier.

**Little's Law — the capacity planning workhorse**
```
concurrency = throughput × latency
```
Worked: an SLO of 20k concurrent sessions `[T]` at ~10 s per session ⇒
`throughput = 20,000 / 10 = 2,000 sessions/s` — the number you actually buy hardware for.

**KV-bound concurrency ceiling** — the real limit `[D]`:
```
max_concurrency = HBM_for_KV / (bytes_per_token × avg_context_len)
```
See [T07](T07-kv-cache.md) for `bytes_per_token`. **You scale GPU count when this ceiling binds —
not when FLOPs do.** Worked: 40 GB KV, 327 KB/token, 4k context ⇒ ~30 sequences per replica.
At 2,000 sessions/s you need ~67 replicas of that shape.

**Autoscaling headroom**
```
replicas = ceil(peak_concurrency / max_concurrency_per_replica) × safety_factor
safety_factor ≈ 1.3–1.5   (covers skew + scale-up lag)
```
Use the *peak* concurrency, not the mean — sessions are long-lived and bursty `[T]`.

**Scale-up latency budget** `[D]`:
```
T_ready ≈ T_schedule + T_image + T_weights + T_warm
```
For a large model on a cold node, `T_weights + T_warm` dominates and can be minutes. Compare
`T_ready` against your traffic's rate of change; if `T_ready` > spike duration, autoscaling cannot
help and you must have pre-warmed capacity.

---

## Configuration

```yaml
# KEDA — queue-driven scaling on a real load signal, not device utilisation
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
spec:
  scaleTargetRef: {name: vllm-decode}
  minReplicaCount: 2          # never 0 in prod — cold start loses the SLO
  maxReplicaCount: 64
  triggers:
    - type: prometheus
      metadata:
        query: avg(vllm:num_requests_waiting)
        threshold: "8"        # the corpus's example saturation point [T]
```

```yaml
# Autoscale on the signal that matters: KV pressure, then goodput
- type: prometheus
  metadata:
    query: avg(vllm:gpu_cache_usage_perc)
    threshold: "0.80"         # KV 80% full -> add capacity [T]
```

**Two-tier scaling** `[D]`: scale the **decode** pool on ITL/goodput and the **prefill** pool on
TTFT/queue depth. They respond to different signals; a single policy couples them and mis-scales.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Pod at 100% GPU but healthy, no scale-out | HPA on utilisation is blind for LLMs | switch to queue-depth or KV-pressure `[D]` |
| Scale-out helps nothing | the constraint is KV per replica, not replica count | check `bytes/token × ctx` |
| Thrash — scale up/down repeatedly | threshold too tight, metric laggy | hysteresis, `stabilizationWindow` `[D]` |
| SLO missed during spikes only | `T_ready` > spike duration | pre-warm; raise `minReplicaCount` |
| Multi-node pods stuck Pending | no gang scheduling | K8s scheduler config `[D]` |
| P99 explodes as load rises | no admission control | add saturation gate + queue `[T]` |
| One tenant consumes the fleet | no per-tenant quotas | gateway quotas `[T]` |
| Scale-down kills in-flight agent sessions | long-running requests treated as idle | track in-flight work, not just QPS `[T]` |
| Throughput looks great, users unhappy | optimising throughput, not goodput | measure SLO attainment `[D]` |
| Scale-up lag measured in minutes | weight load + warmup | pre-pull images/weights to nodes `[D]` |

---

## Gotchas

- **Do not autoscale on GPU utilisation.** LLM inference saturates the device by design; a healthy
  pod looks like an overloaded one. Use **queue depth** or **KV pressure** `[D]`.
- **The binding constraint is KV memory, not compute.** Capacity math starts from
  `HBM_for_KV / (bytes_per_token × context)`, and that number is what you scale against `[D]`.
- **Set `minReplicaCount` above zero in production.** Cold start is minutes; a 0→1 scale-up under
  load is an outage, not an autoscale `[D]`.
- **Goodput is the only autoscaling objective that respects the SLO.** Throughput scaling maximises
  the wrong thing `[D]`.
- **Two SLOs, two pools.** TTFT scales prefill; ITL scales decode. Coupling them mis-sizes both `[D]`.
- **Gang scheduling is a hard requirement for multi-node replicas** — without it, partial
  allocations deadlock and you pay for capacity that can never run. **`[D]`, not `[T]`:** this is
  standard Kubernetes practice, but the corpus never states it. The word "gang" appears in none of
  the named transcripts — verified by search. Treat it as sound engineering judgement, not as
  something a talk in this corpus asserts.
- **Long-lived agent sessions break naive scale-down.** A pod serving one 40-minute agent looks
  idle to a QPS-based scaler. Track in-flight requests and session state `[T]`.
- **The 28k-input cliff is a capacity planning input, not a surprise.** Above it, KV gets
  recomputed and throughput drops at concurrency 256 — so size for the context distribution you
  actually see `[T]`.
- **Load shedding is not failure — it is a feature.** A system that degrades gracefully serves more
  goodput than one that accepts everything and misses every SLO `[T]`.
- **Publish the SLO with the dashboard.** A latency graph with no SLO line cannot tell anyone
  whether to act `[D]`.

---

## When to use what

| Situation | Do |
|---|---|
| Steady load, known peak | provision for peak; autoscale for the trend |
| Bursty, unpredictable | queue + admission control + generous `minReplicaCount` |
| Multi-node replicas | **gang scheduling** `[D]` |
| Latency-critical interactive | scale on TTFT + KV pressure, two pools |
| Batch / best-effort traffic | separate pool with lower priority `[T]` |
| Agentic, long sessions | track in-flight sessions; never scale down mid-session `[T]` |
| Cost-sensitive | scale on goodput; measure the cost per *served* request, not per GPU-hour |
| Cold start unavoidable | pre-pull weights; keep a warm floor `[D]` |

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Tim_Hockin_-_Is_Kubernetes_Good_for_Agents_Infrastructure_Solutions_for_Agent_Sh.txt`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/01-llm-infrastructure.md` `[R]`
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` `[R]`
