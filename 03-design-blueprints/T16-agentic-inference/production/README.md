# T16 — Production Artifacts (reference-grade)

> **Not executed by this repo.** These files document the deployment shape implied by
> [../HLD.md](../HLD.md) and [../LLD.md](../LLD.md). No GPU and no vendor price appears anywhere:
> the budget is denominated in **GPU-seconds and tokens**, never currency, because the corpus this
> blueprint draws on asserts no price (HLD §9). `[T]` transcript · `[R]` repo · `[D]` derived.

```
production/
  README.md                     this file
  harness.yaml                  the agent harness: context assembly, tools, fan-out, budget
  tool-registry.yaml            tool schema placement and per-task retrieval
  trajectory-stream.yaml        durable per-turn trajectory emission (the unrecoverable failure)
  prefix-affinity-routing.yaml  scope_id as a placement input alongside T14's prefix identity
  agent-substrate.yaml          the Kubernetes shape for agent workloads
  otel-collector.yaml           the two metric groups, deliberately on separate axes
```

---

## 1. `harness.yaml` — the harness contract

```yaml
# The harness is the source of every cost property in HLD §3. Its context assembly policy
# decides whether prefix caching works at all.
harness:
  context:
    # LLD §3. If false, the cache is decorative — see the prefix-stability violation table.
    stable_prefix_required: true
    # HLD §4.1: a 6,000-token tool catalogue is 19.1% of all prompt tokens across 40 turns.
    tool_schema_placement: stable_prefix      # stable_prefix | per_turn_text
    ordering: [system, tool_schemas, memory_scope, history, tool_results]  # [D] fixed, never shuffled
    compaction:
      trigger_fraction: 0.80                  # [D] of the model window
      stop_reason: compacted                  # must be emitted so the cache-miss spike is attributable
    # No timestamps, UUIDs, request-ids or dict-ordering artefacts inside the prefix. Each one
    # invalidates the cache from its own offset onward. This is the most common self-inflicted
    # cause of "we enabled prefix caching and saw no improvement". [D]

tools:
  # LLD §4: a GLOBAL concurrency limit starves tasks. Per-task, workload-derived.
  per_task_concurrency: workload_derived
  default_deadline_s: 30                      # [D] policy-set, never agent-set
  # Orphan reaping: a lease alive with an owning task that is terminal is killed by task_id.
  lease_reaper_interval_s: 15                 # [D]

fanout:
  # LLD §12: the ~5.6-turn crossover from run.py §5. Below this, a FRESH scope is cheaper
  # than sharing the parent's — because the shared prefix is not yet large enough to pay
  # for the sub-agent's own cold start. This constant is DERIVED, not copied: recompute it
  # for a real workload rather than trusting 5.
  shared_scope_max_turns: 5
  spawn_limit: budget_derived                 # LLD §7 clamp; HLD §5's 100,000-sub-agent case

budget:
  # HLD §9. Denominated in GPU-seconds and tokens — NEVER in currency, because the corpus
  # asserts no vendor price and a currency budget would require a fabricated one.
  denomination: gpu_seconds
  per_task:
    gpu_seconds_limit: 600                    # [D]
    tokens_limit: 4_000_000                   # [D]
    wall_clock_limit_s: 1800                  # [D]
    retry_limit: 3                            # charged against the budget, per LLD §5
  per_tenant:
    # LLD §9: a per-task budget stops a runaway agent. It does NOT stop a thousand
    # well-behaved agents, which is the normal case and the one that produces the
    # overnight surprise. The tenant budget is the one that bounds the fleet. [D]
    gpu_seconds_per_hour: 50_000              # [D] the number T19 develops further
  enforcement: admission                      # LLD §9 — on the request path, not scraped after

trajectory:
  # LLD §8. The ONLY setting in this file whose violation is unrecoverable: everything
  # else costs money or time; a lost trajectory costs the training signal the harness
  # exists to produce. [T]
  emit: per_turn_durable
```

**The three lines that carry the blueprint.** `stable_prefix_required: true` (or the cache does
nothing), `tools.per_task_concurrency` (a global cap starves tasks), and `trajectory.emit:
per_turn_durable` (the only unrecoverable failure). Everything else is tuning.

---

## 2. `tool-registry.yaml` — the tool-schema tax, as configuration

```yaml
# HLD §4.1. Tool schemas are re-sent EVERY TURN, and their share is a function of the number
# of TURNS, not of the amount of work — so the tax grows as agents get longer, which is the
# regime the field is moving into. [T] "massive tool volume and token prefill"
registry:
  total_schemas: 240                          # [D] illustrative catalogue
  total_schema_tokens: 6000                   # [D] ~19.1% of prompt tokens at 40 turns (HLD §4.1)

  placement:
    mode: stable_prefix                       # never per_turn_text — a per-turn re-send
                                              # cannot be cached if it is not byte-stable (LLD §3)
  selection:
    # Three design consequences from HLD §4.1, and none is about prompt quality:
    #   1. schemas belong in the cached prefix;
    #   2. tool SELECTION is a cache-and-cost optimisation with a serving payoff;
    #   3. the tax is invisible at low load and becomes the whole bill at high load,
    #      because it is exactly the per-token cost batching cannot amortise away.
    enabled: true
    mode: retrieve_relevant                   # ship only the schemas this task needs
    max_schemas_per_task: 12                  # [D]
    # Retrieval MUST be deterministic given the task, or the prefix changes and the cache dies.
    deterministic: true

  # Verification: on a fixed task, the schema block must be byte-identical across turns.
  # This is the integration test LLD §13 says almost nobody writes.
  byte_stability_test: required
```

---

## 3. `trajectory-stream.yaml` — durability

```yaml
# LLD §8 / §11 last row. The agent's decisions cannot be replayed without re-running the
# task, so a lost trajectory is unrecoverable — unlike every other failure in the table,
# which costs only money or time.
trajectory:
  emit: per_turn_durable
  sink:
    type: append_only_log
    fsync: per_turn                           # [D] the point of the setting
    retention_days: 30                        # [D]
  record:
    fields:
      - task_id
      - turn_index
      - prompt_token_count                    # feeds agent_prompt_tokens_per_turn
      - cache_hit_prefix_tokens               # §3's stability signal
      - scope_id                              # §6 — the placement input
      - tool_calls                            # id, args digest, latency, outcome
      - stop_reason                           # absent on a crashed turn -> task is "failed"
      - gpu_seconds_delta                     # charged incrementally against the budget
  on_task_crash:
    # A turn emitted without a terminal stop_reason means the task crashed mid-turn.
    mark_task: failed
    emit_partial: true                        # the partial trajectory is still emitted
```

---

## 4. `prefix-affinity-routing.yaml` — scope as a placement input

```yaml
# LLD §10. The router is T14's; this blueprint CONSUMES it and adds one requirement:
# scope_id becomes a placement input alongside the prefix identity T14 already uses.
router:
  inputs:
    - prefix_block_identity                   # T14's existing signal
    - scope_id                                # T16's addition (LLD §6)
  # A sub-agent placed on a replica that does not hold its parent's scope pays FULL cold
  # prefill. Per run.py §5 that is the difference between 1,924 and 2,418 GPU-seconds per
  # 512 sub-agents at 90% reuse — a 20% swing decided entirely by placement.
  affinity:
    scope_locality_weight: 1.0                # [D]
    fallback: least_loaded                    # only when no replica holds the scope
  on_scope_eviction:
    # The router must consume EVICT events, not only CREATE events, or it will route to
    # a replica that has already dropped the scope. [D]
    subscribe: [kv.block.created, kv.block.evicted]
```

---

## 5. `agent-substrate.yaml` — the Kubernetes shape

```yaml
# The corpus's own question is "is Kubernetes good for agents?" — the answer it gives is
# that it is the substrate, with agent-specific concerns layered on: long idle gaps, many
# short-lived children, and a durable log per task. [T] Hockin, Kubernetes for agents
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-harness
spec:
  replicas: 12
  template:
    spec:
      containers:
        - name: harness
          env:
            - name: HARNESS_CONFIG
              value: /etc/agent/harness.yaml
          resources:
            # Agents idle 99.999% of the time [T]. Request small, and let the FLEET be
            # sized for the fan-out bursts rather than the steady state.
            requests: {cpu: "500m", memory: "1Gi"}
            limits:   {cpu: "2",    memory: "4Gi"}
          volumeMounts:
            - {name: trajectory, mountPath: /var/agent/trajectory}
```

**Why the small request matters.** The corpus states agents are idle **99.999%** of the time `[T]`
Hockin. Provisioning for the peak of a fan-out burst means paying for capacity that is idle almost
always; the design is a modest per-pod reservation with the burst absorbed by the *fleet* — which is
precisely why `fanout.spawn_limit` is budget-derived rather than a pod-count.

---

## 6. `otel-collector.yaml` — two groups, two axes

```yaml
# LLD §10. The metric set is split into two groups that are NEVER graphed on the same axis,
# because HLD §3's finding is that a single dashboard produces a wrong decision.
processors:
  attributes:
    actions:
      - key: agent.group     # "latency" | "cost" — the split is enforced, not conventional
        action: upsert
      # latency group (what the user feels)
      - key: agent.task_wall_seconds
      - key: agent.tool_wall_seconds
      - key: agent.model_wall_seconds
      - key: agent.turns_per_task
      # cost group (what the fleet spends)
      - key: agent.prefill_tokens_total          # the number the whole blueprint is about
      - key: agent.prompt_tokens_per_turn
      - key: agent.cache_hit_prefix_tokens
      - key: agent.gpu_seconds_per_task
      - key: agent.fanout_ratio                  # total GPU-s / parent GPU-s; the clamp trigger
```

**The one graph that separates the two causes of an expensive fleet** (LLD §10):
`agent_prompt_tokens_per_turn` plotted against turn index. A **linear** slope with a large
intercept is the tool-schema tax — fix it by moving schemas into the stable prefix. A
**superlinear** slope is missing prefix reuse — fix it with cache work and placement. Same symptom
on the bill, completely different fix.

---

## Sources

- `refs/Agentic_AI_Infra_transcripts_2/Panel_Agentic_AI_Infrastructure_Platform.txt` — fan-out to 100,000 sub-agents; "massive tool volume and token prefill"
- `refs/Agentic_AI_Infra_transcripts_2/Tim_Hockin_-_Is_Kubernetes_Good_for_Agents_Infrastructure_Solutions_for_Agent_Sh.txt` — agents idle 99.999%; the substrate question
- `refs/Agentic_AI_Infra_transcripts_2/Jerry_Tworek_-_Opportunities_and_Challenges_for_Long_Horizon_Agents.txt` — long-horizon cost growth
- `refs/Agentic_AI_Infra_transcripts_2/Jianfeng_Gao_-_Agentic_Modeling_via_Internalizing_Agent_Harnesses.txt` — harness internalisation
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` — ~70% agentic traffic, ~98% of tokens are prefill

**All configuration here is `[D]`** except where a comment carries `[T]`. Every numeric default is
an illustrative parameter, not a measurement; the corpus supplies the mechanisms, the workload
shape and the 99.999% idle figure, and supplies no configuration file.
