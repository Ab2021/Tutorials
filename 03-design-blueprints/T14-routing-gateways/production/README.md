# T14 — Production reference: the routing layer

> `T14` · **Transcript coverage:** primary · [HLD](../HLD.md) · [LLD](../LLD.md)

**REFERENCE-GRADE, NOT EXECUTED HERE.** Nothing in this folder was applied to a cluster.
There is no gateway and no Kubernetes API in this environment. These are the artifacts the
design calls for, written as they would ship.

## Files

| File | What it is |
|---|---|
| `inference-pool.yaml` | the `InferencePool` + gateway wiring for the Inference Extension |
| `router-config.yaml` | the endpoint picker's policies, flow control and thresholds |
| `slo-policy.yaml` | priority bands, SLO classes, and the shed accounting that goes with them |
| `alerts.yaml` | the alert rules that cover the silent failures HLD §8 names |

## The one thing to read first

`router-config.yaml` contains this line:

```yaml
consume: [create, evict, offload]   # dropping "evict" is the approximate policy [T]
```

It is a one-word config change with a measured consequence. `run.py` §2/§6: dropping evicts
raises the cache hit rate by 4.5 points and cuts usable cluster capacity from 5.48 to 3.61
instances of 8 — a 34% loss of the fleet the flow controller believes it is protecting.

It is not a mistake anyone makes by typing the wrong thing. It is a mistake made by
starting with creates only because they are simpler, and never revisiting it.

## Why the artifacts are shaped this way

**The `InferencePool` is a separate object from the gateway.** The corpus's deployment
plugs into an existing Gateway API Inference Extension rather than replacing the ingress —
GKE, Istio or Envoy `[T]`. The pool says *what* the fleet is; the route says *how* it is
addressed. Keeping them separate means the fleet can be re-labelled without a gateway
change.

**Saturation thresholds sit next to the capacity they assume.** `run.py` §6 is the reason:
the threshold is a constant while the reachable capacity is a variable the routing policy
controls. The config file states the assumed fleet size beside the threshold so the two can
be compared during an incident.

**Priority bands ship with their shed counter, in the same file.** A policy that can refuse
work must count the refusals where the operator will see them — HLD §5. In
`run.py` §5 the shedding policy *improves* the best-effort mean wait (98.42 → 27.34) while
shedding 1,324 requests. Any dashboard showing one number and not the other reports an
incident as an improvement.

**The alerts are all about staleness, not error rates.** The router's characteristic
failure is silently believing something false. Error-rate alerts are structurally blind to
it; `view_staleness_seconds` and `misroutes_total` are not.

## What is deliberately absent

No autoscaling policy. The corpus contains two guide files that contradict each other on
whether the router should scale the fleet on saturation signals, and that contradiction is
a first-class edge case rather than something to resolve silently — it is recorded in
T12/T16 rather than decided here. A production router artifact that quietly picked a side
would be asserting a resolution the corpus does not support.

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — the Gateway API Inference Extension; hooking into GKE/Istio/Envoy; the `InferencePool`
  and `InferenceObjective` shapes; the single router deployment between pods and gateway;
  create and evict event consumption; the filter/rank policy composition; operator-defined
  saturation at 80% KV or 8 mean active requests; FCFS as the policy that "doesn't add
  anything extra"; priority bands under saturation; the TTFT / ITL / pod-saturation /
  queueing metric set.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — the same layer against a second backend.
