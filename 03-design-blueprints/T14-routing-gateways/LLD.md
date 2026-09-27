# T14 — Routing and Gateways: Low-Level Design

> `T14` · **Transcript coverage:** primary · [HLD](HLD.md) · [Sequences](docs/SEQUENCES.md) · [Production](production/README.md)

Component-level design of the endpoint picker: the data it holds, the event feed it
consumes, the pipeline it runs per request, the policy composition, the flow-control
state machine, and the observability that makes its failures visible. Interfaces are given
as concrete shapes; the runnable model of the two central policies is `run.py` + `sim/`.

Everything here is a design. **No gateway, cluster or engine was run**, and no figure below
is a measurement.

---

## 1. Component inventory

| Component | Responsibility | Stateful? |
|---|---|---|
| `Proxy` (gateway) | terminate client connection, apply the Inference Extension filter | no |
| `EndpointPicker` | the router proper — view, pipeline, policies | yes (the view) |
| `EventIngest` | consume create/evict events from engines, apply to the view | yes (offsets) |
| `ViewStore` | per-instance prefix residency + load, the filter's input | yes (the hot path) |
| `SaturationSensor` | evaluate the operator's threshold | yes (windows) |
| `Queue` | pending admissions while saturated | yes |
| `MetricsSink` | the counters in §11 | no |

**Why `EndpointPicker` is one deployment and not a sidecar.** HLD §2. The filter step
needs the whole fleet's residency map; a per-pod component can only ever see its own.

## 2. Wire contracts

### 2.1 The event feed — the router's only source of truth

The corpus states that the router consumes the per-request **create** event and the **evict**
event `[T]`. Both carry the block identity the engine already assigns.

```
KVEvent {
  instance_id : str          # which engine emitted it
  block_hash  : bytes        # the prefix identity (engine-computed, not router-computed)
  event       : "create" | "evict"
  seq         : uint64       # per-instance monotonic; detects gaps and reordering
  ts          : timestamp
  tier        : "hbm" | "cpu" | "disk"      # T12's tiers — offload is not eviction
}
```

Three contract details that are easy to get wrong and expensive to get wrong:

**`seq` is per-instance and must be gap-checked.** A dropped event is a permanently wrong
belief until that block is re-created. Without gap detection the router cannot tell "no
events" from "no events arrived".

**`tier` distinguishes offload from eviction.** T12's tiering means a block can leave HBM
without ceasing to exist. A router that treats offload as eviction loses the ability to
route to a peer that can restore — and the corpus's middle ground between queueing and
recomputing is exactly that peer fetch.

**`block_hash` is engine-computed, not router-computed.** If the router hashes prefixes
itself, it must reproduce the engine's tokenisation and block boundaries exactly; any
divergence produces a router that is confidently wrong. The corpus's approximate policy is
hash-based precisely because it lacks this feed `[T]`.

### 2.2 The request context the router needs

```
RouteRequest {
  request_id   : str
  prefix_hashes: []bytes     # the block hashes of the shared prefix, in order
  phase        : "prefill" | "decode"
  priority     : "premium" | "best_effort"
  est_tokens   : uint32
}
```

`phase` is load-bearing: the corpus routes decode by an **active-request scorer** rather
than prefix identity, because a decode pulling KV from a prefill worker has different needs
`[T]`. One scorer for both is a design error that presents as a tuning problem.

## 3. The view

```
InstanceView {
  instance_id   : str
  resident      : Set[block_hash]   # believed-resident prefixes
  last_seq      : uint64            # gap detection
  active        : uint32            # in-flight requests (decode-aware)
  queued        : uint32
  saturated     : bool              # sensor output, §7
}
```

`resident` is the whole point of the component. Its size is bounded by the fleet's KV cache
in blocks, and its update path is O(1) per event.

**Applying an event:**

```
on create(inst, hash, seq):   expect seq == last_seq+1  else mark view degraded
                              resident.add(hash); last_seq = seq
on evict(inst, hash, seq):    resident.discard(hash);  last_seq = seq
on offload(inst, hash, tier): resident.discard(hash); mark hash as tier-resident
```

**The approximate policy, modelled explicitly.** It applies creates only and ignores evicts.
`run.py` §2/§4 quantifies it: highest hit rate of any policy modelled (81.8%), zero
misroutes in a stationary fleet, and a peak/mean load of 2.22 that leaves only 3.61 of 8
instances usable. Keeping evicts is not free — it costs 4.5 points of hit rate — and it buys
52% more usable cluster.

## 4. The per-request pipeline

The corpus's shape: **build a view → filter → rank** `[T]`. Concretely:

```python
def route(req, fleet, view):
    # 1. FILTER by phase-appropriate identity
    candidates = fleet
    if req.phase == "prefill":
        cached = [i for i in candidates if any(h in view[i].resident
                                               for h in req.prefix_hashes)]
        if cached:
            candidates = cached          # prefer affinity, do not require it
    # 2. RANK what survives
    if req.phase == "prefill":
        return min(candidates, key=lambda i: (i.active, i.queued))
    else:
        # decode: active-request scorer, NOT prefix identity
        return min(candidates, key=lambda i: (i.active))
```

Four properties of this that the design depends on:

**Filter is a preference, not a constraint.** If no instance holds the prefix, `candidates`
falls back to the full fleet. A filter that can empty the candidate set turns a cache miss
into a request failure.

**Phase changes the scorer, not the filter.** Prefill ranks on load *after* filtering for
affinity. Decode skips the affinity filter entirely — the corpus's active-request scorer
`[T]`.

**Ranking alone is the naive policy.** `run.py` §2: round robin and least-loaded misroute
24.9–29.4% of a Zipf workload. A quarter of requests re-prefill something the fleet holds.

**The filter step is why the component is centralised.** §1.

## 5. Policy composition

The corpus gives a worked composition — a prefill filter, then a prefix-cache-identity
filter, then a token-load scorer `[T]`. Generalising that into a config surface:

```yaml
policies:
  - name: prefill-affinity
    phase: prefill
    kind: filter
    on_empty: pass_through        # never fail a request on a cache miss
  - name: prefix-identity
    phase: prefill
    kind: filter
    on_empty: pass_through
  - name: token-load
    phase: prefill
    kind: rank
    weight: 1.0
  - name: active-requests
    phase: decode
    kind: rank
    weight: 1.0
```

Two rules the surface enforces:

**A `filter` stage must declare `on_empty`.** Either `pass_through` or `fail`. Making the
author type it is the difference between a considered decision and an outage.

**Filters precede ranks.** A rank-then-filter pipeline silently discards ranking work and,
worse, makes the ordering of independent stages matter in a way nobody will remember.

## 6. The second middleware tier: semantic routing and guardrails

The corpus describes a **second** routing layer that sits in front of the one designed
above, and conflating the two is a design error worth naming `[T]`.

The endpoint picker answers *which instance of this model*. The semantic router answers
*which model at all* — and whether the request should be served at all. It is described as a
middleware layer whose pipeline is **signal → partition → difficulty score → decision** `[T]`:

| Stage | What it does | Corpus detail |
|---|---|---|
| **signal** | read the request | ~50–60 headers are the source of truth for which model was chosen and its confidence `[T]` |
| **partition** | decide the route | which model class, which tier |
| **difficulty score** | classify the request | simple / medium / reasoning, from a fine-tuned encoder classifier retrained and republished on a regular cadence `[T]` |
| **decision** | dispatch | named algorithms: confidence, rem, fusion, workflow loops `[T]` |

**The headers contract is the interface between the two tiers.** The ~50–60 headers record
which model was chosen and with what confidence `[T]`. That is what makes the decision
auditable after the fact, and it is what lets the endpoint picker below route on a decision
it did not make. A design that passes a bare model name and nothing else loses the
confidence, and with it any ability to ask whether the classifier was sure.

**Guardrails are in this layer, not bolted on beside it.** The corpus's list: input
filtering, semantic classification, policy routing, PII and jailbreak detection, and fact
check `[T]`. Two deployment cases drive the design:

- A **bank** case where sensitive data must not leave the environment. The constraint is a
  routing input, not a filter applied afterwards — the request must be classified before a
  model is chosen, because the model choice is what would leak it `[T]`.
- An **internal-agent** case where routine queries should not reach frontier models. This is
  a cost control expressed as a routing rule, and it is only enforceable if the difficulty
  classifier is good enough to be trusted with it `[T]`.

**Where the two tiers' failure modes differ.** The endpoint picker's failure is *silent
misplacement* — 200s with a bad choice behind them. The semantic router's failure is
*silent misclassification*: a request classified as simple that needed a reasoning model
returns a confident, wrong answer, and nothing in the request path errors. That is why the
confidence value belongs in the header rather than in a log line, and why a low-confidence
classification should have a defined fallback rather than being dispatched on the score.

**What to keep separate.** A single component doing both jobs would have to hold the fleet's
residency map *and* a classifier, and would then be on the critical path for every
admission decision at both tiers. The corpus keeps them as distinct layers `[T]`; the design
above keeps the endpoint picker's hot path free of classification for the same reason.

## 7. Flow control

### 7.1 Saturation

The corpus's examples: KV cache 80% full, or average active requests above eight `[T]`.
Both are operator-typed constants. Design consequence from `run.py` §6: the reachable
capacity those constants describe is *changed by the routing policy above them*.

```
saturated = (kv_used_fraction > cfg.kv_threshold)
         or (mean_active_over_window > cfg.active_threshold)
```

Report `effective_capacity` beside the threshold in the config and the dashboard. A
threshold guarding a fleet that is 45% reachable is a signal about the wrong thing.

### 7.2 Queue policies

```
admit_none      : no flow control; the fleet simply falls behind
fcfs            : a queue with no policy in it
priority_bands  : under saturation, dispatch premium first; bound the best-effort queue
```

`run.py` §5's result, which is the section to read twice:

| policy | premium SLO | best-effort SLO | dispatched | shed | mean wait |
|---|---|---|---|---|---|
| none | 2.9% | 11.1% | 4772 | 0 | 98.42 |
| fcfs | 2.9% | 11.1% | 4772 | 0 | 98.42 |
| priority bands | 99.3% | 12.5% | 3448 | **1324** | 27.34 |

`admit_none` and `fcfs` are **identical rows**. The corpus's judgement that FCFS "doesn't
add anything extra" `[T]` is, in this model, exactly literal.

### 7.3 The state machine

```
        accept ─────────────────────────────► DISPATCHED
          │                                       ▲
          │ not saturated                         │ capacity available
          ▼                                       │
       ADMITTED ────── saturated ──────► QUEUED ─┘
                                            │
                                            │ queue bound exceeded
                                            ▼
                                          SHED ──► shed_total{priority}++
```

`SHED` is a terminal state that must be counted. Under priority bands in the model, 1,324
requests take it, and the class they come from sees its **mean wait improve** while its
attainment does not. Mean wait is a lying metric under a shedding policy.

## 8. Failure handling

| Failure | Detection | Response |
|---|---|---|
| stalled event feed | `view_staleness_seconds` per instance | degrade to load-only routing, alert |
| event gap (`seq` jump) | gap check in §3 | mark instance degraded, stop trusting its residency |
| instance leaves unannounced | health probe vs view membership | remove from fleet, drop residency |
| queue unbounded growth | queue depth vs bound | engage the queue bound, count sheds |
| saturation never engages | queue depth high while `saturated == false` | the threshold and the symptom are different things; re-derive |

The last row is the subtlest: if the saturation signal is derived from KV occupancy and the
deployment is prefill-heavy with fast turnover, the cache may never reach its threshold
while the fleet is queueing. **The signal and the symptom must be the same thing**, and
picking a merely-correlated one is a silent failure.

## 9. Concurrency and consistency

**View updates are per-instance sequential.** The `seq` field makes each instance's stream
ordered by construction; cross-instance ordering does not matter because residency sets are
disjoint per instance. This is what keeps the hot path lock-free per instance.

**Reads are snapshot-tolerant.** A route that reads a residency set one event stale is
correct-but-imperfect; it is never *wrong*, because a stale belief degrades the choice
rather than invalidating it. That is why the filter falls back rather than failing.

**The view is rebuildable.** On restart the router has no residency and routes as
load-only until creates arrive. That is a degraded mode, not an outage, and the recovery
curve is a metric worth emitting.

## 10. Configuration surface

```yaml
router:
  view:
    consume: [create, evict, offload]     # dropping "evict" is the approximate policy
    staleness_alert_seconds: 30
  policies: [...]                          # §5
  flow_control:
    enabled: true
    policy: priority_bands
    saturation:
      kv_used_fraction: 0.80               # operator-defined [T]
      mean_active: 8                       # operator-defined [T]
    queue:
      max_depth_per_instance: 32
      on_overflow: shed
  metrics:
    exclude: []                            # §11 counters are not optional
```

The one comment worth keeping in the file: `# dropping "evict" is the approximate policy`.
It is a one-word config change with a 52%-of-cluster consequence.

## 11. Observability

The corpus's list for this layer is TTFT, ITL, pod saturation and queueing `[T]`. The model
in `run.py` demands four more, because each covers a failure the first four cannot see:

| Metric | Type | Why |
|---|---|---|
| `misroutes_total` | counter | requests placed where the prefix was absent while resident elsewhere — the cost of a stale view, invisible in per-instance hit rate |
| `shed_total{priority}` | counter | the hidden half of priority bands |
| `view_staleness_seconds{instance}` | gauge | the event feed degrading before anything else does |
| `effective_capacity` | gauge | n ÷ (peak/mean) — the fleet the operator thinks they have vs the one they reach |
| `queue_depth{instance}` | gauge | saturation lead indicator |
| `ttft_seconds`, `itl_seconds` | histogram | the corpus's own list `[T]` |

`misroutes_total` and `shed_total` are the two that convert an invisible policy error into
a number. Without them, a stale feed and an aggressive queue policy both present as
"latency is bad sometimes" — and they have opposite fixes.

## 12. What the model does and does not show

`sim/router.py` and `sim/flowcontrol.py` are models of two policies. The workload is
stationary, instances are homogeneous, and the prefix distribution is synthetic.

Where the model agrees with the corpus it corroborates a mechanism: FCFS adds nothing;
priority bands protect premium at best-effort's expense; routing without cache awareness
wastes prefill; a hash-only router cannot follow a moving fleet.

Where it does **not** agree, the disagreement is recorded rather than tuned out:

- **Hash routing wins the hit-rate column** (81.8% vs 77.3%), against the corpus's report
  that the precise approach performs much better. The corpus's comparison is from a
  production deployment under KV saturation with continuously moving cache state; this
  model has a fixed fleet and a stationary workload. The precise view's advantage appears
  here in *usable capacity* (5.48 vs 3.61 of 8), which is the metric saturation cares about.
- **The scale-out penalty is small** (1.8% misroutes) and a consistent-hash ring would
  shrink it further.

Neither was adjusted to produce a preferred answer.

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt`
  — the build-a-view → filter → rank pipeline; consumption of KV create AND evict events;
  the composed policy of prefill filter, prefix-cache-identity filter and token-load scorer;
  the active-request scorer for decode; the single router deployment between pods and
  gateway; the Gateway API Inference Extension; operator-defined saturation (KV cache 80%
  full, average active requests above 8); FCFS "doesn't add anything extra to your
  policies"; priority bands under saturation; TTFT, ITL, pod saturation and queueing as the
  metric set.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Inside_vLLM_Semantic_Router.txt`
  — the router as middleware; the ~50–60 headers recording which model was chosen and its
  confidence; the signal → partition → difficulty score → decision pipeline; the named
  algorithms (confidence, rem, fusion, workflow); synthetic data generation for the
  simple/medium/reasoning split; guardrails — input filtering, semantic classification,
  policy routing, PII and jailbreak detection, fact check; the bank case where sensitive
  data must not leave and the internal-agent case where routine queries must not reach
  frontier models. **ASR garble unresolved:** the encoder classifier is rendered "ambert 32
  model"; ModernBERT and DeBERTa are both plausible and no name is asserted.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
  — the routing layer against a second backend and a WideEP topology.
