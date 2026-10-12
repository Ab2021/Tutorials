# Capacity Planning and Autoscaling for AI Systems

**TL;DR** — Classical capacity planning scales on CPU. AI capacity planning scales on **memory, concurrency, and tokens per second** — and the bottleneck is almost never CPU. This playbook covers the capacity model that actually predicts LLM throughput (Little's Law plus KV-cache arithmetic), why CPU-based autoscaling fails for inference, what to autoscale on instead (queue depth and concurrency, not latency), the cold-start problem that makes GPU autoscaling fundamentally different (3–10 minute node provisioning), how to plan against provider rate limits rather than server capacity, and the load-shedding policy that keeps a saturated system useful. The organising insight: **at 70% utilisation a latency-sensitive system is already in trouble — the queueing curve is non-linear, and the last 20% of utilisation costs you 4x the latency.**

> **60-second interview answer:** "The first thing I'd say is that capacity planning for an LLM system is not a CPU problem. Two things dominate: memory, because concurrent request count is bounded by KV cache size, and tokens per second, because that's the actual unit of work. So I compute capacity in three steps. First, throughput per replica — for an 8B model on an 80 GB GPU with FP16 weights I have about 62 GB left for KV cache, at 0.13 MB per token and a 4,000-token average sequence that's about 118 concurrent sequences, and with continuous batching that's roughly 20 requests per second per GPU. Second, I divide the peak load by that. Third, and this is the part people get wrong, I autoscale on queue depth or concurrency rather than CPU or even latency — because latency is a lagging indicator and CPU isn't the bottleneck. And I keep utilisation at 60 to 70 percent, not 90, because the queueing curve is non-linear: going from 70 to 90 percent utilisation roughly quadruples the wait."

---

## 1. Purpose

This playbook answers three questions, in order:

1. **How much capacity do I need?** — derived from demand, not guessed.
2. **How do I get more, fast, when demand rises?** — autoscaling and provisioning.
3. **What do I do when I cannot get more?** — load shedding and degradation.

It exists because LLM capacity planning has three properties that break the standard approach:

| Standard approach | Why it fails for LLMs |
|---|---|
| Scale on CPU utilisation | CPU is not the bottleneck — memory bandwidth and KV cache are |
| Scale on request rate | Requests vary hugely in cost (100 tokens vs 100,000) |
| Scale on latency | Latency is a *lagging* indicator — you are already degrading |
| Assume fast scale-up | GPU node provisioning is 3–10 minutes, not 30 seconds |
| Assume unlimited backend | Provider rate limits (RPM and TPM) are the real ceiling |

**In scope:** demand forecasting, per-replica throughput modelling, autoscaling policy, headroom policy, provider quota planning, load shedding, and capacity incidents.

**Out of scope:** cost optimisation levers (see [cost and FinOps](06-cost-and-finops.md)), reliability patterns under failure (see [reliability patterns](07-reliability-patterns.md)), and release capacity (see [deployment and CI/CD](04-deployment-release-and-cicd.md)).

---

## 2. Scope and units

### 2.1 The units that matter

Getting the units right is half the work. **Requests per second is a bad unit for LLM capacity.**

| Unit | Use it for | Why |
|---|---|---|
| Requests/s | Gateway and routing capacity | Cheap, uniform work |
| **Concurrency** (in-flight requests) | **The primary capacity unit** | Bounded by KV cache memory |
| **Prefill tokens/s** | Compute-bound input processing | Scales with FLOPS |
| **Decode tokens/s** | Memory-bandwidth-bound generation | Scales with HBM bandwidth |
| **KV cache bytes/token** | Concurrency ceiling | The hard limit |
| **Queue depth** | Autoscaling signal | Leading indicator |

**The rule:** *"I plan in concurrency and tokens per second, and I report in requests per second only when talking to people outside the team. A request is not a unit of work — a 200-token answer and a 20,000-token answer are the same request and 100x the work."*

### 2.2 The six goals as capacity constraints

| Goal | Capacity implication |
|---|---|
| **Performance** | Drives the 60–70% utilisation ceiling |
| **Availability** | Drives N+1 headroom and multi-region |
| **Reliability** | Drives the load-shedding policy and the degradation ladder |
| **Cost** | Drives the utilisation floor — idle capacity is wasted money |
| **Scalability** | Drives the autoscaling design and its limits |
| **Consistency** | Drives the warm-pool requirement for cache-hit stability |

**The central conflict:** *"Performance and cost pull in opposite directions here, and the resolution is the utilisation target. At 60% utilisation my p95 is comfortable and I'm paying for 40% idle. At 90% my p95 blows out — going from 70 to 90 percent utilisation roughly quadruples the wait time for a memory-bandwidth-bound decoder. So I target 60–70% for anything latency-sensitive and only push higher for batch work that has no latency SLO."*

---

## 3. SLIs and SLOs

### 3.1 Capacity-specific SLIs

| SLI | Definition | Target | Alert |
|---|---|---|---|
| **Saturation** | max(gpu_kv_cache_used, gpu_compute_used) / capacity | ≤ 0.70 | > 0.85 for 10 m |
| **Queue depth** | Requests waiting for a slot | ≤ 3× replica count | > 5× for 5 m |
| **Queue wait time** | p95 time from arrival to first token started | ≤ 250 ms | > 500 ms for 5 m |
| **Scale-response time** | From breach to additional capacity serving | ≤ 90 s (API), ≤ 8 min (self-hosted) | > 12 min |
| **Headroom** | (capacity − peak load) / capacity | ≥ 30% | < 15% |
| **Rejected requests** | Shed or 429'd / total | ≤ 0.5% | > 2% for 5 m |
| **Provider quota utilisation** | RPM and TPM consumed / quota | ≤ 70% | > 85% for 10 m |
| **Cold-start rate** | Requests hitting a cold replica / total | ≤ 5% | > 15% |

### 3.2 System SLOs that capacity must protect

| SLI | SLO | Capacity driver |
|---|---|---|
| Availability | 99.9% | N+1 minimum, multi-AZ |
| **TTFT p95** | ≤ 800 ms | Queue wait + prefill time |
| **TPOT p95** | ≤ 60 ms/token | Decode batch size |
| **End-to-end p95** | ≤ 3 s (RAG), ≤ 90 s (agent) | All of the above |
| Throughput floor | ≥ 50 req/s sustained | The thing you scale for |
| Error rate | ≤ 0.5% | Shedding policy |

### 3.3 Error budget as a capacity signal

```
   Monthly budget (99.9%)                       = 43 min 12 s

   Saturation-induced latency breach:
     p95 > 2x SLO for 6 min = 6 min of budget burnt
     → one such event costs ~14% of the monthly budget

   Three saturation events in a month = 42% of budget.
   At 60% consumed → capacity freeze: no new launches,
   only scaling work. This is how capacity debt gets visible.
```

---

## 4. Architecture

```
══════════════════ DEMAND SIDE ══════════════════

   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │  Users   │  │  Batch   │  │ Agents   │
   │ (bursty) │  │ (steady) │  │ (long)   │
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        └─────────────┴─────────────┘
                      ▼
        ┌───────────────────────────────┐
        │  PRIORITY CLASSIFIER          │
        │  P0 interactive · P1 standard │
        │  P2 background · P3 bulk      │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  ADMISSION CONTROL            │
        │  per-tenant quota · RPM/TPM   │
        │  → 429 or queue by priority   │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  QUEUE (per priority class)   │
        │  depth = THE autoscaling      │
        │  signal                       │
        └───────────────┬───────────────┘
                        ▼
══════════════════ SERVING SIDE ══════════════════

   ┌────────────────────────────────────────────┐
   │  AUTOSCALER                                │
   │  inputs: queue depth, KV util, tokens/s    │
   │  NOT: CPU, NOT raw latency                 │
   │                                            │
   │  ┌──────────────┐   ┌──────────────────┐   │
   │  │ API replica  │   │ GPU node pool    │   │
   │  │ pool         │   │ (self-hosted)    │   │
   │  │ scale 30-90s │   │ scale 3-10 min   │   │
   │  └──────────────┘   └──────────────────┘   │
   └───────────────┬────────────────────────────┘
                   ▼
   ┌────────────────────────────────────────────┐
   │  MODEL SERVER (vLLM / TGI / TensorRT-LLM)  │
   │                                            │
   │   weights  │  KV CACHE  │  CUDA graphs     │
   │   ← the concurrency ceiling is HERE →      │
   │                                            │
   │   continuous batching · PagedAttention     │
   └───────┬────────────────┬───────────────────┘
           ▼                ▼
   ┌──────────────┐  ┌──────────────────────┐
   │  SEMANTIC    │  │  PROVIDER POOL       │
   │  CACHE       │  │  (fallback / burst)  │
   │  hit = 0 GPU │  │  RPM + TPM quota     │
   └──────────────┘  └──────────────────────┘

══════════════════ HEADROOM BUDGET ══════════════════

   Total capacity 100%
   ├── 60-70%  steady-state serving
   ├──  15%    surge / diurnal peak
   ├──  10%    N+1 failure headroom
   └──   5%    deploy headroom (shadow + canary)
```

---

## 5. Standard operating procedures

### SOP-1 — Capacity estimation (do this on the back of an envelope)

```
STEP 1  DEMAND
   DAU × requests/user/day = requests/day
   QPS = requests/day ÷ 86,400
   Peak QPS = QPS × 3                    (5x for consumer bursty)
   Peak concurrency = Peak QPS × avg_service_time_s     [Little's Law]

STEP 2  WORK PER REQUEST
   avg input tokens   (prefill — compute-bound)
   avg output tokens  (decode — memory-bandwidth-bound)
   P50 and P95 of the above. Plan on P95.

STEP 3  PER-REPLICA CEILING (the KV cache calculation)
   kv_bytes_per_token = 2 × layers × kv_heads × head_dim × dtype_bytes
   concurrency_max    = free_vram_for_kv ÷ (kv_bytes_per_token × avg_seq_len)
   throughput_req_s   ≈ concurrency_max ÷ avg_service_time_s
                      (with continuous batching; verify empirically)

STEP 4  REPLICAS
   replicas = ceil(peak_concurrency ÷ concurrency_max)
   + N+1 failure headroom
   + deploy headroom

STEP 5  VERIFY
   □ Does measured throughput/s match the model within 20%?
   □ Is the bottleneck actually KV, or is it prefill FLOPS?
   □ Does the provider quota allow the burst?
   □ Is the cold-start time inside the scale-response SLO?
```

**Worked example — a RAG service at 800k requests/day:**

```
STEP 1  Demand
   800,000 req/day ÷ 86,400          = 9.3 QPS average
   Peak = 9.3 × 3                    = 28 QPS
   Service time p50 = 1.8 s
   Peak concurrency = 28 × 2.6 s     = ~73 in flight (using p95 service time)

STEP 2  Work per request
   Input:  p50 3,500 · p95 14,000 tokens
   Output: p50 180 · p95 520 tokens

STEP 3  Per-replica ceiling — 8B model, A100 80 GB, FP16
   Weights                                = 16 GB
   Activations + CUDA graphs + overhead   ≈ 2 GB
   Free for KV cache                      ≈ 62 GB
   
   kv per token = 2 × 32 layers × 8 kv_heads × 128 head_dim × 2 B
                = 131,072 B = 0.131 MB/token
   
   At 4,000-token average context:
     0.131 MB × 4,000 = 524 MB per sequence
     62 GB ÷ 0.524 GB ≈ 118 concurrent sequences
   
   Decode: with continuous batching, ~2,800 aggregate output tok/s
     → 2,800 ÷ 180 avg output = ~15.5 req/s
   Prefill: 14,000-token p95 × 28 QPS = 392k prefill tok/s needed
     → this is often the REAL bottleneck, not decode

STEP 4  Replicas
   Peak concurrency 73 ÷ 118 per replica = 1 replica for concurrency
   BUT prefill-bound: 2 replicas needed for prefill throughput
   + 1 for N+1                              = 3 replicas
   + deploy headroom (shadow)               = 4 replicas

STEP 5  Verify
   Measured: 4 replicas sustain 44 req/s at p95 2.9 s ✓ (28 needed)
   Headroom at peak = (44 − 28) / 44 = 36% ✓
```

**The lesson from step 3:** *"The naive answer is 1 replica because concurrency fits. The correct answer is 4 because the bottleneck is prefill, not decode — a RAG system with 14,000-token contexts is compute-bound on the input side, and the KV-cache calculation alone will under-provision you by 4x. **Always compute both and take the larger.**"*

### SOP-2 — Choosing the autoscaling signal

```
   SIGNAL QUALITY, ranked:

   ✓✓✓  QUEUE DEPTH (per priority class)
        Leading — rises before latency does.
        Scale target: keep depth ≤ 3 × replica count.

   ✓✓   CONCURRENCY (in-flight requests / max concurrency)
        Leading-ish. Directly tied to the KV limit.
        Scale target: 0.65 of max concurrency.

   ✓✓   KV CACHE UTILISATION
        The actual saturation metric for self-hosted inference.
        Scale target: ≤ 0.75.

   ✓    TOKENS/S vs CAPACITY
        Good but noisy; needs a rolling window.

   ✗    CPU UTILISATION
        Wrong for LLM inference. A saturated GPU server often
        sits at 30-50% CPU. You will scale too late or never.

   ✗✗   LATENCY (p95 TTFT)
        LAGGING. If p95 TTFT is high, you are ALREADY degraded.
        By the time this fires, users have had a bad minute.

   ✗✗✗ REQUEST RATE (QPS)
        Misleading — a request can be 100 or 100,000 tokens.
```

**The practical policy:**

```yaml
autoscaling:
  # ── API / orchestration replicas (cheap, fast) ──────────
  api_pool:
    min_replicas: 4
    max_replicas: 40
    metrics:
      - type: queue_depth
        target: 30                # items in queue
        scale_up_stabilization: 30s
        scale_down_stabilization: 300s
      - type: cpu
        target: 70                # secondary guard only
    scale_up_policy:
      step: 50%                   # +50% per step
      period: 30s
      max_step: 10                # cap to avoid thundering herd
    scale_down_policy:
      step: 10%
      period: 120s

  # ── Self-hosted GPU pool (expensive, slow) ──────────────
  gpu_pool:
    min_replicas: 3               # N+1 for a 2-replica baseline
    max_replicas: 24
    warm_pool: 1                  # 1 idle replica, always ready
    metrics:
      - type: kv_cache_utilization
        target: 0.70
      - type: queue_wait_p95_ms
        target: 250
    scale_up_policy:
      step: 2                     # ADD 2 NODES, never 1 — the
      period: 180s                # provisioning cost is per-node,
    scale_down_policy:            # not per-node-second
      step: 1
      period: 900s
      cooldown: 1800s             # never scale down within 30 min
                                  # of a scale-up
    predictive:
      enabled: true               # pre-scale on known patterns
      rules:
        - cron: "0 7 * * 1-5"     # weekdays 07:00
          target_replicas: 8      # pre-warm before the 09:00 peak
```

**Three rules embedded in that config, each worth stating:**

1. **Scale up fast, scale down slow.** Asymmetric stabilisation. Scaling up late costs latency; scaling down late costs a little money. The asymmetry is deliberate.
2. **Add N nodes at a time, not 1.** GPU provisioning is 3–10 minutes. If you add one node every 3 minutes and demand needs four, you spend 12 minutes degraded. Adding 2–4 at once overshoots by a little and recovers in one provisioning cycle. **Overshoot is cheap; slow recovery is not.**
3. **Never scale down within 30 minutes of scaling up.** Oscillation is worse than over-provisioning, because it defeats the cache and the warm pool.

### SOP-3 — Handling cold start

GPU cold start has four distinct components and each needs a different fix.

```
   t=0      ┌─────────────────────────────────────────────────┐
            │ 1. Autoscaler decision                    ~5 s  │
            ├─────────────────────────────────────────────────┤
            │ 2. Node provisioning (K8s + GPU driver) 180-480s│ ← the killer
            ├─────────────────────────────────────────────────┤
            │ 3. Image pull + container start          30-90 s│
            ├─────────────────────────────────────────────────┤
            │ 4. Model weight load into VRAM            20-60 s│
            ├─────────────────────────────────────────────────┤
            │ 5. CUDA graph capture / warmup            10-40 s│
            └─────────────────────────────────────────────────┘
   TOTAL: 4-11 minutes
```

| Component | Mitigation | Effect |
|---|---|---|
| Node provisioning | **Over-provisioned node pool** (nodes running, no pods) | **Removes 3–8 min** — the single biggest win |
| Node provisioning | Predictive pre-scaling on known patterns | Removes it entirely for predictable peaks |
| Image pull | Pre-pulled images, node image cache | −30–60 s |
| Weight load | **Local NVMe model cache**, not S3 each time | −30–50 s |
| Weight load | `--load-format=safetensors` with mmap | Faster start |
| CUDA graphs | Capture at one batch size, not a sweep | −20–30 s |
| All | **Warm pool of 1 replica** | Instant for the first surge step |

**The over-provisioned node pool is the key design.** *"I keep 2 GPU nodes idle and ready. That costs roughly $4–8 per hour per node, so maybe $250 a day. In exchange, a scale-up takes 45 seconds instead of 8 minutes. If you have a p95 latency SLO and a diurnal peak, that is the cheapest latency insurance in the system. Teams almost always reject it on cost, and then spend the next year firefighting."*

**The warm-pool sizing rule:**

```
   warm_replicas ≈ peak_arrival_rate_per_s × cold_start_s
   
   Example: 1.5 req/s arriving, 45 s to serve, 300 s cold start
     A warm replica absorbs 1/45 = 0.022 req/s of service
     To absorb a 10-req/s burst for 300 s you need
       (10 × 300) / 45 ≈ 67 replica-seconds
     → 2 warm replicas would cover the first 90 seconds
   
   Practical: warm pool = ceil(expected_scale_step × cold_start / 60)
   For a 4-node step and 5-minute cold start → 1 warm node
```

### SOP-4 — Provider quota planning

When you use an API provider, **your capacity ceiling is their rate limit, not their servers.** Two quotas bite in different ways.

| Quota | Fails how | Mitigation |
|---|---|---|
| **RPM** (requests/min) | Hard 429 at the boundary | Batch small requests; cache aggressively |
| **TPM** (tokens/min) | Hard 429; the one that surprises people | Cap context; enable prompt caching |
| **Concurrent requests** | 429 or queueing | Bound with a semaphore |
| **Daily/monthly spend cap** | Hard stop — total outage | Alert at 60%; negotiate a higher cap |
| **Provisioned throughput** | Reserved capacity | Buy for the floor, burst for the peak |

**Quota arithmetic — do this before launch:**

```
   Peak QPS                     28
   Avg tokens/request (in+out)  3,700  (in 3,500 + out 200)
   Peak TPM = 28 × 3,700 × 60 = 6.2M tokens/min

   If the provider quota is 4M TPM → YOU CANNOT SERVE THE PEAK.
   Options:
     a) Raise the quota (lead time: days to weeks)
     b) Enable prompt caching → 70% of input cached at a
        ~90% discount → effective TPM ≈ 28 × (1,050 + 3,500×0.1 + 200) × 60
                                    ≈ 2.7M TPM  ✓ fits
     c) Route 40% to a second provider
     d) Shed or queue

   ALWAYS compute peak TPM before launch. This is the number
   that causes day-one outages on provider-based systems.
```

**The multi-provider capacity plan:**

| Provider | Role | Quota share | Failover |
|---|---|---|---|
| Primary | 70% of traffic | Provisioned throughput for the floor | — |
| Secondary | 25% (steady, keeps the path warm) | On-demand | Takes primary's share |
| Tertiary | 5% + emergency burst | On-demand | Last resort |

**Why the secondary runs 25% at all times and not 0%:** *"A failover provider you have never sent traffic to is a provider you have never tested. You will discover the tokeniser difference, the tool-call format difference, and the rate limit at 3 a.m. during an incident. Running 25% steady keeps the path exercised and the differences known. The cost of that redundancy is real and it is the cheapest reliability you will ever buy."*

### SOP-5 — Load shedding

**When you cannot add capacity, you must choose who gets served.** Shedding is a product decision implemented as a technical policy, and it must be decided before the incident.

```
   PRIORITY CLASSES

   P0  Interactive, paid, latency-SLO bound
       → never shed. Degrade quality instead.
   P1  Interactive, standard
       → shed last. Queue up to 10 s.
   P2  Background (summarisation, enrichment, eval)
       → shed first. Queue to 60 s, then drop.
   P3  Bulk (backfill, re-embed, batch eval)
       → shed immediately. Pause and resume later.

   SHEDDING LADDER (in order)

   1. Pause all P3 work                              invisible
   2. Disable proactive/speculative features         invisible
   3. Reduce semantic cache TTL (more hits)          invisible
   4. Lower retrieval k 50 → 20                      minor quality
   5. Cap output tokens 1,024 → 512                  shorter answers
   6. Route P1 to a smaller model                    quality dip
   7. Queue P2 instead of executing                  visible delay
   8. Return 429 to P3 with Retry-After              visible
   9. Shed P1 at the edge with 429                   visible, last resort

   NEVER in the ladder:
     ✗ Disable safety guardrails
     ✗ Disable PII filtering
     ✗ Bypass the action gate
     Capability degrades. Safety does not.
```

**The two shedding modes and when to use each:**

| Mode | Behaviour | Use when |
|---|---|---|
| **Queue** | Hold the request; the user waits | Wait is tolerable (< 10 s) and the work is valuable |
| **Reject** | 429 with `Retry-After` | Wait would exceed the client timeout anyway |
| **Degrade** | Serve a cheaper/smaller answer | Quality loss is acceptable, availability is not |

**The decision rule:** *"Rejecting is honest; degrading is silent. I strongly prefer rejecting a background job over silently serving a worse answer to an interactive user, because the background job can be retried and the user cannot tell that they got the degraded one."*

**Shedding must be automatic and pre-authorised.** *"If shedding requires a human decision at 3 a.m., it will not happen and the system will fall over for everyone instead of degrading for some. So the ladder is encoded, the thresholds are set, and the on-call's job is to verify it fired correctly — not to decide whether to fire it."*

### SOP-6 — Capacity incident runbook

```
SYMPTOM: p95 latency breach, queue depth climbing, saturation > 0.85

T+0    CONFIRM
       □ Is it demand (traffic up) or capacity (replicas down)?
         → traffic dash vs replica count dash, overlaid
       □ Is it one model, one region, or global?
       □ Did a deploy happen in the last 60 min?  → roll back first
       □ Is a provider rate-limiting us? → check 429 rate by provider

T+2m   IMMEDIATE ACTIONS (in order)
       □ 1. Engage the shedding ladder from the top
       □ 2. Scale the API pool manually to max
       □ 3. Force GPU pool scale-up (+4 nodes)
       □ 4. Drain the semantic cache TTL to 0 (max hit rate)
       □ 5. If provider-limited: shift traffic share to secondary

T+5m   IF STILL DEGRADED
       □ 6. Enable the degradation ladder (smaller model)
       □ 7. Reject P3 and P2 traffic at the edge
       □ 8. Page the platform lead

T+10m  IF STILL DEGRADED
       □ 9. Consider regional failover (see DR playbook)
       □ 10. Notify stakeholders with an ETA range

T+RESOLVED
       □ Ramp back up SLOWLY — remove shedding steps one at a time,
         with 5 min between steps. Removing them all at once
         re-creates the surge.
       □ Post-incident: was this a capacity shortfall or a
         capacity-planning failure? Different fixes.
```

**The most important line in that runbook:** *"Ramp back up one step at a time with 5-minute gaps. The single most common second incident is a thundering herd created by restoring full service instantly — every queued and retried request hits at once."*

---

## 6. Failure modes and mitigations

| # | Failure | Symptom | Root cause | Mitigation |
|---|---|---|---|---|
| 1 | **CPU-based autoscaling never fires** | Latency collapses, replicas flat | CPU is not the bottleneck | Scale on queue depth / KV utilisation |
| 2 | **Scale too late** | p95 bad for 10 min before replicas arrive | Scaling on latency (lagging) | Scale on queue depth (leading) + predictive |
| 3 | **Cold start exceeds SLO** | 8 min degraded after every scale event | Node provisioning | Over-provisioned node pool + warm replica |
| 4 | **Autoscaler oscillation** | Replicas flap every few minutes | Symmetric stabilisation, tight cooldown | Asymmetric: fast up, slow down, 30 min cooldown |
| 5 | **TPM quota exhaustion** | 429s from the provider at peak | Planned in QPS, not tokens | Compute peak TPM; prompt caching; multi-provider |
| 6 | **Thundering herd on recovery** | Second incident right after the first | All requests retried at once | Jittered retries; ramp shedding off one step at a time |
| 7 | **KV cache OOM** | Server crash under load | Concurrency ceiling exceeded | Admission control tied to KV utilisation |
| 8 | **Cache stampede on release** | Cost and latency spike post-deploy | Cold cache after `release_id` key change | Pre-warm top 1,000 queries before the flip |
| 9 | **Shadow doubles load** | Latency regression blamed on the release | Shadow mirrors 100% of traffic | Shadow 30% for 8 h, or size for 2x |
| 10 | **Diurnal peak missed** | Every Monday 09:00 is bad | Reactive-only autoscaling | Predictive pre-scale on a cron |
| 11 | **Zombie capacity** | Bill grows, utilisation falls | Scale-down blocked by a stuck metric | Alert on replica-count vs load divergence |
| 12 | **No N+1** | One node loss = outage | Capacity planned to exactly peak | N+1 minimum enforced in the autoscaler floor |
| 13 | **Shedding requires a human** | Total outage instead of partial | No pre-authorised policy | Encoded ladder with automatic thresholds |
| 14 | **Shedding disabled safety** | Guardrails off under load | Ladder included safety bypass | Safety is never in the ladder; test this in chaos drills |
| 15 | **Quota raised without notice** | Provider 429 at the new ceiling | Assumed quota scaled with traffic | Recompute quota headroom monthly |
| 16 | **Long-context requests starve short ones** | p50 fine, p95 terrible | One 100k-token request blocks a batch | Separate pools or separate queues per context-length class |
| 17 | **Agent loops consume capacity** | Sudden capacity collapse | Runaway agents at 40 steps each | Per-task step and cost budgets; agent pool isolated from RAG pool |
| 18 | **Capacity plan never revisited** | Plan from 6 months ago | No review cadence | Monthly capacity review against actuals |

### 6.1 The long-context starvation problem — expanded

This one is subtle and worth a paragraph.

```
   A batch mixes a 500-token request and a 100,000-token request.
   With continuous batching, the short request's decode is
   interleaved with the long one — so TTFT for the short request
   is fine but its TPOT suffers, and the batch holds far more
   KV than planned.

   Fixes:
     1. SEPARATE POOLS: route by context length class
        (< 8k · 8k-32k · > 32k) to different replicas.
        Costs capacity fragmentation.
     2. CHUNKED PREFILL: split long prefill into chunks so short
        requests interleave. Free, and the right default.
     3. ADMISSION CONTROL: cap concurrent long-context requests
        per replica (e.g. 4 at > 32k).
     4. PREEMPTION: vLLM can preempt and recompute long sequences.

   Prefer (2) then (3). (1) is a last resort because it wastes
   capacity — the 8k pool sits idle while the 32k pool is full.
```

### 6.2 The agent capacity trap

Agents break capacity models in a specific way: **a single request can consume 40 sequential model calls over 90 seconds.** A capacity plan built on "requests × service time" assumes one call per request.

```
   RAG:    1 request = 1 LLM call  = ~2.7 s  = 1 concurrency slot
   Agent:  1 request = 5-40 calls  = 30-90 s = 5-40 concurrency slots

   → An agent fleet and a RAG fleet MUST have separate capacity
     pools. If they share, a single agent surge consumes the
     entire fleet and takes RAG down with it.

   → Agent concurrency must be planned as:
        peak_agent_concurrency × avg_steps_per_run
     not peak_agent_concurrency.
```

---

## 7. Capacity and cost

### 7.1 Self-hosted vs API — the break-even

| Factor | API | Self-hosted |
|---|---|---|
| Time to first capacity | Minutes | Days to weeks |
| Marginal cost at low volume | Low | **High** (fixed GPU cost) |
| Marginal cost at high volume | High | **Low** |
| Scale-up latency | 0 (provider absorbs) | 3–10 min |
| Scale-to-zero | Instant | Impossible (warm pool needed) |
| Data residency | Provider-dependent | Full control |
| Ops burden | ~0 | 1–2 dedicated engineers |
| **Break-even** | | **~4,000–8,000 calls/day** at $2–4/GPU-hour vs $0.012/call |

**The hybrid that actually works in production:**

| Workload | Placement | Why |
|---|---|---|
| Steady baseline (floor traffic) | Self-hosted | Cheapest per token |
| Diurnal peak | API burst | No idle GPU |
| Batch / bulk | API batch endpoint (50% off) | No latency SLO |
| Spikes and failover | API secondary | Infinite ceiling |
| Long-context (rare) | API | Not worth a dedicated pool |

**The economics of the warm pool, stated as a number:** *"A warm GPU node costs about $2.50 an hour, so $60 a day. It saves 5–8 minutes of degraded latency on every scale event, and at 3 scale events a day that is 15–24 minutes of p95 breach avoided. Fifteen minutes of p95 breach costs about 35% of my monthly error budget. So $60 a day buys me most of a month's budget. That is an obviously correct trade and it is still the one teams reject first."*

### 7.2 The capacity cost model

```
   Monthly cost = Σ (replicas × hours × unit_cost) + provider spend

   Self-hosted GPU pool:
     baseline 3 nodes × 730 h × $2.50      = $5,475
     warm pool 1 node  × 730 h × $2.50     = $1,825
     peak surge: 200 node-hours × $2.50    = $500
     over-provisioned nodes 2 × 730 × $2.50= $3,650
     ─────────────────────────────────────────────
     total                                 = $11,450

   API provider:
     peak 6.2M TPM × 30% duty cycle
       → ~2.7M effective TPM after caching
       → ~$14,000/month at current rates

   TOTAL INFRASTRUCTURE CAPACITY COST      ≈ $25,450/month

   Idle-capacity cost embedded above:      ≈ $5,475 (21%)
   → This is the price of the latency SLO. Report it explicitly
     so it is a decision, not a surprise.
```

### 7.3 Provisioning lead times — plan backwards from these

| Resource | Lead time | Implication |
|---|---|---|
| API quota increase | **2 days – 3 weeks** | Plan 4 weeks ahead of a launch |
| Cloud GPU quota (new region) | 1–4 weeks | Start early |
| Reserved GPU capacity | 1–6 months | Commit only for the floor |
| On-demand spot GPU | Minutes, may be unavailable | Do not depend on it |
| New region enablement | 4–12 weeks | Region expansion is a quarter-long project |
| Model snapshot access | Immediate | — |

**The rule that follows:** *"If your launch needs a quota increase, the capacity work starts four weeks before the launch, not the week of. Almost every day-one AI outage I've seen was a quota that nobody requested in time."*

---

## 8. Metrics and alerts

### 8.1 Alerts

```yaml
capacity_alerts:
  # ── Saturation (leading) ─────────────────────────────────
  - name: kv_cache_saturation
    condition: gpu_kv_cache_utilization > 0.85
    window: 10m
    severity: critical
    action: scale_up, page
    note: this is the real saturation signal for inference

  - name: queue_depth_growth
    condition: queue_depth > 5 * replicas
    window: 5m
    severity: critical
    action: scale_up

  - name: queue_wait_breach
    condition: queue_wait_p95_ms > 500
    window: 5m
    severity: warning
    action: scale_up

  - name: headroom_low
    condition: (capacity - load) / capacity < 0.15
    window: 15m
    severity: warning
    note: act before saturation, not after

  # ── Scaling health ───────────────────────────────────────
  - name: scale_up_latency
    condition: time_from_breach_to_serving > 12m
    severity: warning
    note: the warm pool is undersized or the node pool is exhausted

  - name: autoscaler_oscillation
    condition: replica_changes > 8 in 60m
    severity: warning
    action: widen stabilisation windows

  - name: scale_up_failure
    condition: scale_up_requested AND NOT replicas_increased within 10m
    window: 10m
    severity: critical
    action: page_platform
    note: usually a cloud quota or a scheduling failure

  - name: warm_pool_depleted
    condition: warm_available_replicas < 1
    severity: warning

  # ── Provider quota ───────────────────────────────────────
  - name: provider_rpm_utilization
    condition: rpm_consumed / rpm_quota > 0.85
    window: 10m
    severity: warning

  - name: provider_tpm_utilization
    condition: tpm_consumed / tpm_quota > 0.85
    window: 10m
    severity: critical
    note: TPM is the quota that surprises people

  - name: provider_429_rate
    condition: provider_429_rate > 0.01
    window: 5m
    severity: critical
    action: shift_traffic_to_secondary

  - name: provider_spend_cap_approaching
    condition: provider_spend / spend_cap > 0.60
    severity: warning
    note: a spend cap breach is a total outage, not a bill

  # ── Efficiency ───────────────────────────────────────────
  - name: capacity_cost_per_request
    condition: capacity_cost_per_request > 2 * 30d_median
    window: 1h
    severity: warning

  - name: idle_capacity_excess
    condition: utilization < 0.30 for 24h
    severity: info
    action: review_pool_size
    note: the opposite failure — paying for nothing

  - name: tokens_per_request_drift
    condition: mean_tokens_per_request > 1.3 * 30d_median
    window: 24h
    severity: warning
    note: usually a prompt change, a new feature, or a retry loop
```

### 8.2 Dashboards

**Dashboard 1 — Capacity (on-call's first screen)**

| Panel | Content |
|---|---|
| Traffic | QPS overlaid with peak forecast |
| **Concurrency** | In-flight vs max concurrency per pool |
| **KV cache** | Utilisation per replica, heatmap |
| Queue | Depth and p95 wait, by priority class |
| Replicas | Current, desired, warm-available |
| Saturation | max(KV, compute) as a single % with the 70% line |
| Provider | RPM and TPM vs quota, side by side |

**Dashboard 2 — Capacity planning (monthly review)**

| Panel | Content |
|---|---|
| Demand vs actual | Forecast from last month overlaid on actuals |
| Headroom trend | 90 days, with the 15% floor marked |
| Scale events | Count, duration, and time-to-recover |
| Cost per 1k requests | Trend, with release markers |
| Idle capacity | $ per day spent on unused capacity |
| Quota utilisation | 90-day trend and time to breach |

### 8.3 The monthly capacity review — the checklist

```
□ Measured peak vs planned peak — variance > 25%?
□ Headroom stayed above 15% at all times?
□ How many scale events, and did any breach the SLO?
□ Time-to-recover distribution per scale event
□ Provider quota utilisation trend — when do we breach?
□ Cost per 1k requests — up or down, and why?
□ Any capacity-related incident? Root cause: planning or execution?
□ Forecast next month from the growth trend, not from last month
□ Launch calendar for the next 90 days — any quota requests needed?
```

---

## 9. Change management

| Change | Review | Notice |
|---|---|---|
| Autoscaling thresholds | Platform lead | 1 day |
| Pool min/max | Platform lead | 1 day |
| Warm-pool size | Platform lead + finance | 1 day |
| Shedding ladder order | **Product + platform** | **1 week** |
| Priority-class assignment | Product | 1 week |
| Provider quota increase | Platform | **4 weeks lead** |
| New region | Platform + finance | 1 quarter |
| Reservation / commitment | Finance + platform | 1 quarter |

**The shedding-ladder change is the one that needs product sign-off**, because reordering it changes *who* gets degraded. That is a product decision disguised as a config change.

---

## 10. Audit and compliance

| Record | Contents | Retention |
|---|---|---|
| Capacity plan | Forecast, model, assumptions, per month | 3 years |
| Scale events | Trigger, time, replicas added, time-to-recover | 1 year |
| Shedding events | Priority class, count shed, duration, who approved | 3 years |
| Quota records | Provider, quota, utilisation, increase requests | 3 years |
| Reservation commitments | What was committed and for how long | 7 years |
| Incident → capacity link | Which incidents were capacity-related | 3 years |

**The audit question to be able to answer:** *"During the incident on the 14th, which requests were shed and under whose authority?"* — This requires shedding decisions to be logged with the policy version that authorised them, not just the count.

---

## 11. Interview Q&A

**Q1. How do you plan capacity for an LLM system?**
Three steps. First, demand: DAU times requests per user, divide by 86,400 for average QPS, multiply by 3 for peak, then multiply by service time for peak concurrency — that's Little's Law. Second, per-replica ceiling, and this is where it differs from normal systems: I compute the KV cache budget. An 8B model in FP16 takes 16 GB of weights on an 80 GB GPU, leaving about 62 GB for KV. At 0.13 MB per token and a 4,000-token average sequence, that's 524 MB per sequence, so about 118 concurrent sequences per replica. Third, replicas: peak concurrency divided by per-replica concurrency, plus N+1 and deploy headroom. But I always compute the prefill side too, because a RAG system with 14,000-token contexts is compute-bound on input, and in my worked example that changed the answer from 1 replica to 4.

**Q2. What do you autoscale on?**
Queue depth, primarily. It's the leading indicator — it rises before latency does. Then KV cache utilisation, which is the actual saturation metric for inference, with a target around 0.70 to 0.75. I explicitly do not scale on CPU, because a saturated GPU inference server often sits at 30 to 50% CPU, so CPU-based autoscaling either never fires or fires far too late. And I don't scale on latency, because by the time p95 TTFT is high, users have already had a bad minute — it's a lagging indicator that tells you about a problem you already have.

**Q3. Why not run at 90% utilisation and save money?**
Because the queueing curve is non-linear and it bends hard. For a memory-bandwidth-bound decoder, going from 70% to 90% utilisation roughly quadruples the wait time — the relationship isn't linear, it's closer to hyperbolic as you approach saturation, because variability in service time means the queue grows without bound near 100%. So the last 20% of utilisation costs 4x the latency. I target 60 to 70% for anything latency-sensitive and only push to 85 or 90% for batch work that has no latency SLO and can absorb queueing.

**Q4. How do you handle GPU cold start?**
It's four components and the biggest one by far is node provisioning, which is 3 to 8 minutes on top of 30 to 90 seconds for image pull, 20 to 60 seconds to load weights into VRAM, and 10 to 40 seconds for CUDA graph capture. So 4 to 11 minutes total. The single highest-leverage fix is an over-provisioned node pool — nodes running and ready with no pods scheduled. That costs roughly $60 a day per node and it removes the 3-to-8-minute component entirely. Plus a warm replica so the first surge step is instant. Teams usually reject the idle node cost and then firefight for a year, which is a bad trade.

**Q5. Why scale up fast and scale down slow?**
It's an asymmetric cost. Scaling up late costs latency, which is visible to users and burns error budget. Scaling down late costs a little money, which is invisible. So the stabilisation windows are asymmetric — 30 seconds up, 300 seconds down — and there's a 30-minute cooldown after any scale-up before scaling down is allowed. Oscillation is the thing I most want to avoid, because flapping defeats the warm pool and the cache, and it's worse than simply over-provisioning.

**Q6. Why add multiple nodes at once rather than one at a time?**
Because the provisioning cost is per-node, not per-node-second. If I need four nodes and I add one every three minutes, I spend twelve minutes degraded. If I add two at once and overshoot by one, I recover in one provisioning cycle and pay a little extra for a few minutes. Overshoot is cheap; slow recovery is expensive. So the scale-up step is 2 to 4 nodes with a max-step cap to avoid a thundering herd on the control plane.

**Q7. What's the capacity ceiling when you use an API provider?**
Their rate limit, not their servers — and there are two quotas that fail differently. RPM is requests per minute, which you can usually manage by batching and caching. TPM is tokens per minute, and that's the one that surprises people, because a system planned in QPS can be fine on RPM and blow through TPM at peak. The arithmetic: 28 peak QPS at 3,700 tokens per request is 6.2 million TPM. If the quota is 4 million, you cannot serve the peak, full stop. The fixes are to raise the quota, which has a two-day-to-three-week lead time, or enable prompt caching so 70% of the input bills at a 90% discount, which brings the effective rate to about 2.7 million and fits. I always compute peak TPM before launch — a quota that nobody requested in time is the most common cause of a day-one outage on a provider-based system.

**Q8. What is load shedding and how do you design it?**
It's the policy for when you cannot add capacity, and it's a product decision implemented as a technical policy. I define priority classes — P0 interactive paid that is never shed, P1 standard, P2 background, P3 bulk — and a ladder of nine steps from least to most visible. Pausing P3 work is invisible; shedding P1 with a 429 is a last resort. The ladder is encoded and pre-authorised, because if shedding requires a human decision at 3am it won't happen and the system falls over for everyone rather than degrading for some. And the hard rule is that safety is never in the ladder — I never disable guardrails, PII filtering, or an action gate to recover capacity. Capability degrades, safety doesn't.

**Q9. How do you decide between queueing and rejecting?**
Rejecting is honest, degrading is silent. I prefer rejecting a background job over silently serving a worse answer to an interactive user, because the background job can be retried and the user can't tell they got the degraded version. So the rule is: queue if the predicted wait is under the client timeout and the work is valuable, reject with a `Retry-After` if the wait would exceed the timeout anyway, and degrade only when quality loss is acceptable and availability matters more.

**Q10. What's the biggest capacity mistake you've seen?**
Sharing a capacity pool between agent workloads and single-shot workloads. A RAG request is one LLM call and about 2.7 seconds; an agent request is 5 to 40 sequential calls over 30 to 90 seconds. So one agent consumes 5 to 40 concurrency slots where one RAG request consumes one. If they share a pool, a single agent surge takes down the RAG service. Agent concurrency has to be planned as peak agent concurrency times average steps per run, and the two pools have to be separate.

**Q11. How do you handle a huge spike — say 20x for an hour?**
Four mechanisms in order. Predictive pre-scaling if the spike is predictable, like a Monday morning or a product launch. The over-provisioned node pool for unexpected spikes, which absorbs the first few minutes. API burst capacity, which has effectively infinite ceiling as long as the quota covers it — that's the main reason to keep a secondary provider running at 25% steady state. And then load shedding, which caps the damage. What I don't rely on is self-hosted GPU autoscaling, because 4 to 11 minutes of cold start means it cannot respond to a spike shorter than about 15 minutes.

**Q12. How do you plan for a launch?**
Backwards from the lead times. Provider quota increases are 2 days to 3 weeks, cloud GPU quota for a new region is 1 to 4 weeks, new region enablement is 4 to 12 weeks. So if the launch needs a quota increase, the capacity work starts four weeks before launch. Then I load-test at 2x the forecast peak, run the shedding ladder deliberately in a game day to confirm it fires, and verify the scale-response SLO by actually triggering a scale-up. The thing I check that people skip is the prefill bottleneck — load tests at low context lengths pass and then production fails because real contexts are ten times longer.

**Q13 (pushback). Isn't 30% headroom just wasted money?**
It's not wasted, it's priced. The idle-capacity cost is about 21% of the infrastructure spend, roughly $5,500 a month on the numbers I used, and in exchange I keep p95 inside its SLO and I absorb one node failure without an outage. If I ran at 95% utilisation to eliminate it, I'd spend that saving on error budget instead — and the queueing math means I'd spend more. The honest framing is that headroom is the price of the latency SLO and the availability target, and it should be reported as a line item so it's a decision rather than a surprise. What I do watch for is the *opposite* failure — utilisation below 30% for a sustained period, which means I'm paying for capacity I'm not using, and that's a pool-size review.

**Q14. How do you know your capacity model is right?**
I validate it against measurement every month. The model predicts throughput per replica; I compare that against actual sustained throughput and I expect agreement within 20%. When it diverges, the usual cause is that I modelled decode and the real bottleneck is prefill, or the average context length drifted, or the batch composition changed. I also track the forecast-versus-actual peak variance — above 25% and the forecast method needs revisiting, not just the number. And I watch tokens-per-request drift on a 24-hour window, because that metric moving 30% is usually a prompt change or a retry loop hiding as a capacity problem.

**Q15. What would you do first if you inherited an AI system with capacity problems?**
Instrumentation, before any scaling change. I need to know which of the three it is: demand, capacity, or efficiency. Queue depth, KV utilisation, tokens per second, and replica counts, all on one dashboard, overlaid. Nine times out of ten the answer is that they're autoscaling on CPU and it never fires, or they're scaling on latency and it fires too late. That's a one-line config change with a huge effect. After that, the warm pool and the over-provisioned nodes, because those fix cold start. Then separate the agent pool from the RAG pool. And only then look at buying more capacity.

---

## 12. Cheat-sheet summary

| Dimension | Value |
|---|---|
| **Primary capacity unit** | **Concurrency**, not requests/s |
| **Secondary units** | Prefill tokens/s · decode tokens/s · KV bytes/token |
| **Little's Law** | concurrency = arrival rate × service time |
| **Estimation template** | DAU × req/day ÷ 86,400 → QPS → ×3 peak → × service time → concurrency |
| **Constants** | 1M req/day ≈ 12 QPS · 1 KB × 1M = 1 GB · 1 day ≈ 10^5 s |
| **KV per token (70B GQA)** | ~0.31 MB/token |
| **KV per token (8B GQA)** | ~0.13 MB/token |
| **KV formula** | 2 × layers × kv_heads × head_dim × dtype_bytes |
| **A100 80 GB, 8B FP16** | 16 GB weights → ~62 GB KV → ~118 seqs at 4k |
| **Utilisation target** | **60–70%** latency-sensitive · 85% batch |
| **Why not 90%** | 70→90% utilisation ≈ **4x the wait** |
| **Autoscale on** | **Queue depth** · KV utilisation · concurrency |
| **Never autoscale on** | CPU · raw latency · raw QPS |
| **Scale-up stabilisation** | 30 s |
| **Scale-down stabilisation** | 300 s · 30 min cooldown |
| **Scale-up step** | 2–4 nodes at once (provisioning is per-node) |
| **Cold start breakdown** | decision 5 s · node 180–480 s · image 30–90 s · weights 20–60 s · warmup 10–40 s |
| **Cold start total** | 4–11 min (self-hosted GPU) |
| **Over-provisioned node pool** | ~$60/node/day → removes 3–8 min |
| **Warm pool** | `ceil(step × cold_start / 60)` → typically 1 replica |
| **Warm node cost** | ~$2.50/GPU-hour, ~$60/day |
| **Headroom budget** | 60–70% serving · 15% surge · 10% N+1 · 5% deploy |
| **Provider quotas** | **RPM and TPM** — always compute peak TPM |
| **Quota increase lead time** | 2 days – 3 weeks |
| **Break-even self-host** | ~4,000–8,000 calls/day |
| **Priority classes** | P0 never shed · P1 · P2 · P3 shed first |
| **Shedding ladder** | 9 steps, P3 pause → 429 at the edge |
| **Safety in the ladder** | **Never** — capability degrades, safety does not |
| **Recovery rule** | Ramp back **one step per 5 min** — avoid the herd |
| **Error budget link** | 6 min of p95 breach ≈ 14% of a monthly 99.9% budget |
| **Freeze rule** | 60% budget consumed → capacity-only work |
| **Review cadence** | Monthly, against actuals |
| **Forecast tolerance** | ±25% peak variance |
| **Model validation tolerance** | ±20% throughput vs predicted |
| **The rule that never breaks** | **Compute both prefill and decode limits; take the larger.** |

---

## 13. Cross-links

- [Deployment, Release and CI/CD](04-deployment-release-and-cicd.md) — shadow and canary capacity requirements
- [Cost and FinOps](06-cost-and-finops.md) — the cost side of the utilisation trade-off
- [Reliability Patterns](07-reliability-patterns.md) — circuit breakers, retries, and degradation
- [DR, Failover and Backup](09-dr-failover-and-backup.md) — regional capacity and failover headroom
- [LLMOps for AI Systems](10-llmops-for-ai-systems.md) — the operational lifecycle this sits inside
- [LLM Serving, Inference and Gateway](../HLD/05-ai-systems/05-llm-serving-inference-and-gateway.md) — PagedAttention, continuous batching, and the KV cache
- [Agentic System Design](../HLD/05-ai-systems/06-agentic-system-design.md) — why agent pools must be separate
- [Capacity Estimation Playbook](../HLD/01-foundations/05-capacity-estimation-playbook.md) — the base estimation template
- [Availability Numbers and SLOs](../HLD/01-foundations/06-availability-numbers-and-slos.md)
- [Operationalisation Cheatsheet](../Cheat-Sheets/06-operationalisation-cheatsheet.md)
