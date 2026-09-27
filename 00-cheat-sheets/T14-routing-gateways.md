# Cheat Sheet: Routing, Gateways & Semantic Dispatch

> `T14` · **Transcript coverage:** primary · [Case study](../01-case-studies/T14-routing-gateways.md) · [Blueprint](../03-design-blueprints/T14-routing-gateways/HLD.md) · [Interview bank](../02-interview-questions/T14-routing-gateways.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| Cache reads vs fresh tokens | **~10× cheaper** | `[T]` LLMOps cost talk |
| Cache-hit contribution to the cost ladder | **26 → 11** (the final step) | `[T]` LLMOps cost talk |
| Symptom of naive load balancing | prefix cache hit rate **"atrocious"** in Prometheus | `[T]` llm-d |
| What beats hash routing | **per-block KV events** (creation + eviction) | `[T]` llm-d |
| FCFS queueing | **"adds nothing"** | `[T]` llm-d |
| Saturation signals used for routing | KV cache **80% full**, or **active requests > 8** | `[T]` llm-d |
| The routing layer's name in llm-d | **EPP** — Endpoint Picker | `[T]` llm-d |
| Semantic router's job | match a request's *meaning* to a target, not its path | `[T]` meetup |

---

## The one-table summary

| Routing strategy | Signal used | Good at | Bad at |
|---|---|---|---|
| **Round-robin / random** | none | nothing for caching; fine for stateless models | destroys prefix locality |
| **Least-connections** | in-flight count | rough load balance | ignores KV state entirely |
| **Hash routing** | request/prefix hash | approximate locality, no state needed | **consistency problems** — hash ≠ actual block residency `[T]` |
| **Prefix-cache-aware (KV events)** | per-block create/evict events | **the correct answer** — routes to where KV actually is | needs an event plane from the engine `[T]` |
| **Performance / load-aware (EPP)** | queue depth, KV saturation | protecting SLOs under skew | needs accurate, low-lag metrics `[T]` |
| **Semantic routing** | embedding similarity to intents | cost tiering, model choice, domain dispatch | extra latency + an embedding model to run `[T]` |
| **Rule-based** | headers, tenant, token count | predictability, easy debugging | does not adapt |
| **Cascade / fallback** | confidence, error, budget | quality-per-rupee; resilience | two paths to maintain `[D]` |

**The ordering that works** `[D]`: **rule-based first** (cheap, deterministic policy — tenant,
quota, model alias) → **semantic** (which model/tier does this request deserve?) → **performance /
prefix-aware** (which *replica* holds this prefix and has capacity?). Doing them in the opposite
order means you pay embedding latency before you have even applied policy.

---

## Formulas

**Routing's effect on cache economics** `[D]`:
```
effective_cost = hit_rate × cached_cost + (1 − hit_rate) × fresh_cost
```
With cache reads ~10× cheaper `[T]`, moving hit rate from 0.2 to 0.8 cuts the per-token cost by
roughly `0.6 × 0.9 = 54%` — **which is exactly the final 26 → 11 step on the cost ladder** `[T]`.
Routing is not a plumbing concern; it is the last big cost lever.

**Prefix-aware placement** `[T]`:
```
score(replica) = prefix_match_blocks(replica) × w₁ − queue_depth(replica) × w₂ − kv_pressure(replica) × w₃
```
The llm-d insight is that `prefix_match_blocks` must come from **actual block events**, not from
hashing the prompt. A hash tells you where a prefix *would* be if the cache were perfect; events
tell you where it *is* `[T]`.

**Saturation gate**
```
route_away_if: kv_used_fraction > 0.80  OR  active_requests > 8
```
These are the corpus's example thresholds `[T]`. They are examples, not law — pick from your own
goodput curve — but the *pattern* (route on pressure, not just on load) is what keeps P99 sane.

---

## Configuration

```yaml
# llm-d / Gateway API Inference Extension — the declarations that matter
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata: {name: llm}
spec:
  rules:
    - matches: [{path: {value: /v1/chat/completions}}]
      backendRefs:
        - group: inference.networking.x-k8s.io
          kind: InferencePool
          name: my-pool
---
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferencePool
metadata: {name: my-pool}
spec:
  endpointPickerConfig:
    extensionRef: {name: epp}       # the EPP decides the endpoint
```

```python
# Rule + semantic + performance, in the order that works [D]
def route(req):
    if t := tenant_policy(req):            # 1. rules: quota, allowed models
        return t
    tier = semantic_classify(req)          # 2. meaning -> model/tier
    return epp.pick(tier, req.prefix)      # 3. prefix + pressure aware
```

**Fallback is part of the router** `[D]`: on 5xx, timeout, or saturation, fall back to
(a) another replica, (b) a smaller/cheaper model, (c) a cached answer, in that order. Decide the
order deliberately — it determines your degradation behaviour under load.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Cache hit rate near zero in Prometheus | naive LB ignoring prefix locality | router affinity `[T]` |
| TTFT spikes on turn 2+ | follow-up went to a cold pod | prefix-aware routing |
| Good routing on average, terrible P99 | no saturation signal | add KV-pressure gating `[T]` |
| Traffic piles onto one replica | hash routing collided | switch to event-based `[T]` |
| Semantic router slow | embedding model in the hot path | cache classifications; move it off the critical path |
| Routing flap — requests oscillate | metrics too noisy / too slow | smooth the signal, hysteresis `[D]` |
| Cost not falling despite caching | cache is on but routing defeats it | this is the 26 → 11 gap `[T]` |
| Bad fallback behaviour | no defined degradation order | define replica → model → cache `[D]` |
| Prefixes route wrong after a restart | events lost, router state stale | resync from engine state `[D]` |

---

## Gotchas

- **A load balancer is not a router.** For LLM serving, "least loaded" and "least likely to have
  your prefix" are different questions, and the second one dominates cost `[T]`.
- **Hash-based prefix routing has consistency problems.** The corpus is explicit: precise per-block
  KV events beat hashing `[T]`. If you cannot get events, hashing is a fallback, not the design.
- **Routing is the last big cost lever on the ladder.** Caching gets you the memory; routing gets
  you the *hit rate*. **26 → 11** `[T]`.
- **Semantic routing costs a model call.** Budget it, cache it, and keep it off the critical path
  where possible `[D]`.
- **Route on pressure, not just queue depth.** KV saturation at 80% or >8 active requests are the
  corpus's example signals `[T]`; the pattern generalises.
- **FCFS "adds nothing"** `[T]` — do not ship it and call it a policy.
- **Failover must be ordered and tested.** An untested fallback path is not a fallback; it is a
  second outage waiting for the first `[D]`.
- **The gateway is a control point for governance too** — quotas, tenancy, audit, PII policy. See
  [T18](T18-guardrails-security.md) and [T19](T19-finops-sovereignty.md).
- **Cache-aware routing requires the engine to talk to the router.** Budget the integration: vLLM
  emits block events; the router must consume them. This is a cross-team interface `[T]`.

---

## When to use what

| Situation | Route by |
|---|---|
| Single replica, dev | round-robin is fine |
| Multi-replica, shared system prompts | **prefix-cache-aware** — mandatory |
| Mixed tenants with quotas | rules first, then prefix-aware |
| Multiple models / cost tiers | semantic routing to pick the tier |
| Latency-critical under skew | performance routing (EPP) with saturation gating |
| Regulated / audited traffic | rule-based + gateway policy, deterministic |
| Provider outage | ordered fallback: replica → model → cache |
| You want the cost ladder's final step | fix routing before buying more GPUs `[T]` |

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Inside_vLLM_Semantic_Router.txt`
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/03-ai-gateways-and-model-routing.md` `[R]`
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` `[R]`
