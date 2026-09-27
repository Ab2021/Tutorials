# T07 — Production Artifacts (reference-grade)

> **Not executed by this repo.** These are real artifacts an operator would use, written to be
> copy-adapted rather than copy-pasted. Every bandwidth, TTL and capacity here is `[D]` and must be
> replaced with measured values. `[T]` marks corpus-sourced facts. `[R]` marks repo-sourced ones.

```
production/
  README.md                this file
  engine-kv.yaml           vLLM launch config: block size, cache capacity, offload tiers
  retention-policy.yaml    per-session retention driven by re-arrival, not a fleet TTL
  cache-routing.yaml       llm-d EPP: prefix-affinity placement so fleet reuse survives scaling
  tier-fabric.yaml         the measured-media inventory the tiering decision reads
  queries.promql           refcount accounting, hit rate (both measures), tier occupancy
```

---

## 1. `engine-kv.yaml` — the engine's KV configuration

```yaml
# vLLM-style launch configuration. Field names follow the engine's own vocabulary; values are
# illustrative.
model: meta-llama/Llama-3.1-70B-Instruct
dtype: float16

kv_cache:
  # Block size is the ONE knob. See HLD 2.1 for the curve.
  #  16 = the corpus-typical default [R] and the knee of the curve
  #   8 = use when the traffic is dominated by prompts under ~64 tokens
  #  32+ = use only for long-context-only workloads (64k+)
  block_size: 16                 # [R]
  dtype: float16                 # fp8 doubles the concurrency ceiling (HLD 7.1)
  gpu_memory_utilization: 0.90   # [T] NextGen: size to fill 90% of VRAM
  swap_space_gb: 64              # host DRAM tier; see tier-fabric.yaml for the decision

  # CRITICAL: bound the prefix cache BELOW the pool. A cache with an unbounded footprint will
  # consume every block and then a live sequence cannot be admitted -- the pool-exhaustion
  # MemoryError arrives while the cache holds thousands of blocks nobody is reading.
  # This was a real defect in the runnable core: PrefixCache defaults to the whole pool.
  prefix_cache:
    enabled: true
    max_blocks: 0.60             # [D] fraction of the pool; the rest is live-sequence headroom
    hash: chained_blake2b        # MUST chain to the parent block (HLD 4.1)
    key_extra:                   # everything that makes identical tokens non-interchangeable
      - lora_adapter_id
      - multimodal_hash
      - system_prompt_version
    reuse_partial_block: false   # MUST be false -- the tail's KV was never computed

  # The offload decision is driven by measured media, not by a blanket setting.
  offload:
    enabled: true
    target: host_dram
    policy_file: retention-policy.yaml
    # DO NOT enable tiering across a fabric where transfer > recompute. Price it first:
    #   sim.tiering.breakeven(8000, 327680, medium, prefill_tps)
    # On TCP/IP this returns recompute for an 8k session (HLD 6.1).
```

**Why `block_size: 16` is annotated rather than asserted.** The default is right for a general
workload and wrong for a short-prompt one. Leaving the reason beside the value is what lets an
operator change it deliberately rather than by guess.

**Why `max_blocks: 0.60` is the most important line in the file.** The runnable core's
`PrefixCache` defaults its capacity to the whole pool. That is fine for a simulation and
catastrophic in production: the cache will fill the pool with prefixes nobody is reading, and the
first live sequence after that gets a `MemoryError`. Bounding the cache is what keeps the refcount
accounting honest.

**Why `reuse_partial_block: false` is written out.** It is the engine's default and it is also one
of the two settings whose corruption is *silent*. Writing it explicitly means a future reader who
flips it has to disagree with a comment rather than with nothing.

---

## 2. `retention-policy.yaml` — per-session, not a fleet TTL

```yaml
# The corpus's contract [T] (llm-d): the retention API orchestrates KV movement at the engine
# layer and attaches SESSION METADATA to the cache so the system knows which session a request
# belongs to. The policy below is [D] -- the corpus supplies the mechanism, not the thresholds.
#
# The failure this file exists to prevent: a single fleet-wide TTL evicts the agent prefix that
# returns in milliseconds AND retains the batch job's 5 GB that never returns. See HLD 6.5.

session_classes:
  - name: agent_tool_call
    # Re-arrival is MILLISECONDS. A chat-sized TTL destroys the ~5x TTFT win [T].
    expected_re_arrival_s: 0.5
    ttl_s: 30
    action: retain_in_dram
    source: harness_tool_latency_p95     # the harness knows this (T16); a timer cannot

  - name: chat_session
    expected_re_arrival_s: 45
    ttl_s: 300
    action: retain_in_dram
    source: session_gap_histogram

  - name: one_shot_batch
    expected_re_arrival_s: null          # never returns
    ttl_s: 0
    action: evict
    source: request_metadata

  - name: unknown
    # The default must be the CONSERVATIVE one. An unknown session class must not be allowed to
    # consume a tier it may never release.
    expected_re_arrival_s: null
    ttl_s: 60
    action: recompute_on_return
    source: default

tier_capacity:
  dram_gb: 512
  # When the tier is full, evict the session with the SOONEST expiry -- which is the one
  # LEAST likely to be needed again. Do not evict LRU: recency and re-arrival are different
  # signals, and the agent prefix that just went on a long tool call is the least recent and
  # the most likely to return.
  eviction_order: soonest_expiry

events:
  # The router consumes BOTH event types [T]. A hit count alone cannot tell "this tier is
  # working" from "this tier is full of sessions that will never come back".
  emit: [create, evict]
  sink: otel
  attributes: [session_id, session_class, bytes_kv, reason, expected_re_arrival_s]
```

**Why `unknown` defaults to `recompute_on_return`.** The safe default for a class you have not
characterised is the one that consumes no tier capacity. A default of `retain` lets an
unclassified workload monopolise DRAM — and because the symptom is a *capacity* problem, it looks
like an engine bug rather than a policy error.

**Why eviction is `soonest_expiry`, not LRU.** These are different signals and the difference is
the agentic case. An agent prefix that has just gone on a 30-second tool call is the **least
recently used** and the **most likely to return**. LRU evicts precisely the wrong session. This is
the same distinction the corpus draws between "which session is active" and "which session has a
tool call coming back" `[T]`.

---

## 3. `cache-routing.yaml` — the fleet-scale multiplier

```yaml
# Without prefix-aware routing, scaling from 1 replica to 10 makes caching WORSE: each replica
# starts cold and dilutes the fleet-wide hit rate, while every per-pod dashboard stays green.
# This is the failure llm-d's prefix-cache-aware routing exists to fix [T].

routing:
  scorer: prefix_affinity
  inputs:
    - kv_cache_events        # create AND evict, from the engine [T]
    - prefix_hash_index      # which replica holds which block hash
    - kv_utilisation         # load balancing still applies
  weights:
    prefix_affinity: 0.7
    load_balance: 0.3
  # When affinity and balance conflict, affinity wins up to a utilisation ceiling -- past that
  # the replica is saturated and admitting a request there costs more than the prefill saved.
  affinity_ceiling: 0.85     # [T] saturation gate is KV ~80% full

scopes:
  # Agents carry a scope_id so all turns of one session land on one replica (T16).
  - name: agent_session
    key: scope_id
    sticky: true
  - name: chat
    key: prefix_hash
    sticky: false
```

**Why affinity must have a ceiling.** Perfect affinity plus a hot prefix is a load-balancing
catastrophe: every request for that prefix queues behind the one replica that holds it. The
ceiling is what makes affinity a *scorer* rather than a *constraint*, and the corpus's own
saturation gate (KV ~80%) `[T]` is the natural place to set it.

---

## 4. `tier-fabric.yaml` — the measured media the tiering decision reads

```yaml
# THE DECISION DEPENDS ENTIRELY ON THESE NUMBERS. The corpus supplies the ORDERING
# (pooled memory < RDMA << TCP/IP) [T] and no figures. These are placeholders -- replacing them
# with the operator's own measured round-trip times is the single most important step in
# enabling tiering, because the offload-vs-recompute crossover happens inside this range.

media:
  - name: pooled_memory
    bandwidth_bytes_s: 900000000000
    latency_s: 0.000002
    corpus_rank: 0            # [T]
  - name: rdma
    bandwidth_bytes_s: 50000000000
    latency_s: 0.000005
    corpus_rank: 1            # [T]
  - name: nvme_local
    bandwidth_bytes_s: 7000000000
    latency_s: 0.000020
    corpus_rank: null         # NOT ranked by the corpus
  - name: tcp
    bandwidth_bytes_s: 3000000000
    latency_s: 0.000200
    corpus_rank: 3            # [T]

decision:
  # At an 8k session on a 70B fp16 (2.62 GB of KV) with re-prefill at 12k tok/s:
  #   pooled  5.8 ms   -> offload  (~115x better)
  #   rdma    105 ms   -> offload  (~6.4x -- the corpus's ~5x, reproduced)
  #   nvme    749 ms   -> RECOMPUTE
  #   tcp     1748 ms  -> RECOMPUTE (2.6x WORSE than re-prefilling)
  # DO NOT ENABLE TIERING ON A MEDIUM WHERE THIS TABLE SAYS RECOMPUTE.
  verify_before_enable: true
  measurement: round_trip_p95_over_10gb
```

**Why `corpus_rank: null` for NVMe is the honest entry.** The corpus ranks three media `[T]`.
NVMe is in the model because real deployments use it, and marking it as unranked is what stops a
reader from believing the corpus endorsed a position it never took.

**Why `verify_before_enable: true`.** The TCP row is a latency *regression* recorded as an
optimisation. That is the failure this file exists to prevent, and it is invisible in every
dashboard except TTFT on restored sessions.

---

## 5. `queries.promql` — refcount accounting and the two hit rates

```promql
# ---- THE REFCOUNT INVARIANT ------------------------------------------------------------
# THE most important query in this file. Every block is either free or referenced, so the sum
# of live references must equal total minus free. A gap means a LEAK (too many) or an
# UNDERFLOW (too few). An underflow is the silent-corruption case: a live sequence is reading
# blocks that were reallocated to another request, and the symptom is fluent wrong output.
(llm_kv_blocks_total - llm_kv_blocks_free) != llm_kv_refcount_sum

# ---- SATURATION -- the corpus's gate ---------------------------------------------------
# [T] alert at KV 80% full; the engine is configured to fill 90% of VRAM.
llm_kv_cache_usage_ratio > 0.80

# ---- THE TWO HIT RATES -- they answer different questions ------------------------------
# Request hit rate: how often a request found a cached prefix.
sum(rate(llm_prefix_cache_hits_total[5m])) / sum(rate(llm_prefix_cache_lookups_total[5m]))
# Token reuse fraction: how much PREFILL WORK was avoided. This is the one the router uses [T].
sum(rate(llm_prefix_cache_hit_tokens_total[5m])) / sum(rate(llm_prompt_tokens_total[5m]))

# ---- CACHE-BLIND ROUTING -- per-pod healthy, fleet-wide not ---------------------------
# If per-pod hit rate is high while fleet-wide reuse is low, the ROUTER is the problem,
# not the cache. This pair is the diagnostic.
max by (pod) (rate(llm_prefix_cache_hit_tokens_total[5m]))
  /
sum(rate(llm_prompt_tokens_total[5m]))

# ---- SESSION METADATA -- the retention API's observable surface [T] -------------------
# Sessions held in a tier that have NOT returned. A growing count means the tier is filling
# with sessions nobody is waiting for, i.e. the policy is wrong.
llm_kv_retained_sessions{tier="dram"} and time() - llm_kv_session_last_seen_seconds > 600

# ---- TIER REGRESSION -- tiering that made things slower -------------------------------
# Compare TTFT for restored vs recomputed sessions. If restored is SLOWER, the medium is
# wrong and tiering should be turned off for it (HLD 6.1).
histogram_quantile(0.95, sum by (le, path) (rate(llm_ttft_seconds_bucket{path=~"restored|recompute"}[5m])))
```

**Why the refcount query is first.** It is the only query here that detects a *correctness*
failure. Every other query detects a performance or capacity problem, which the user experiences as
slowness. An underflow is experienced as wrong answers, and by the time anyone notices, the
correlation with the KV subsystem has been lost.

**Why the two hit rates are adjacent.** They are routinely conflated in dashboards, and a fleet
with many tiny requests plus one long one has a *high* token-reuse fraction and a *low* request hit
rate. Watching the wrong one sizes the cache wrong in the wrong direction.

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` — CPU KV offload (~5× TTF, session return); the router consuming create and evict events; the retention API and session metadata
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt` — KV saturation at 80%; fill to 90% of VRAM
- `refs/Agentic_AI_Infra_transcripts_2/Jongryool_Kim_-_Disaggregated_LLM_Serving_with_Shared_Memory_KV_Cache_at_Rack_Sc.txt` — the pooled-memory tier and the transfer-medium ordering
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` — block-table and paged-attention structure; the block-size default

**All configuration, all bandwidths and all TTLs here are `[D]`/illustrative** except where a line
carries `[T]`. The corpus supplies the mechanisms, the ordering, the retention/session-metadata
contract and the saturation gates; it supplies no configuration file, no bandwidth and no TTL.
