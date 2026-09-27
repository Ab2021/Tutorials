# T08 — Production Artifacts (reference-grade)

> **Not executed by this repo.** These are real artifacts an operator would use, written to be
> copy-adapted rather than copy-pasted. Every threshold, slot count and SLO here is `[D]` and must
> be replaced with measured values. `[T]` marks corpus-sourced facts. `[R]` marks repo-sourced ones.

```
production/
  README.md                this file
  engine-scheduler.yaml    vLLM-style launch config: slots, token budget, chunking, preemption
  flow-control.yaml        llm-d EPP: the saturation gate, bands, per-program fairness
  slo-policy.yaml          the objective -- SLOs that BIND, and the goodput they define
  capacity-plan.yaml       the sizing arithmetic: slots from the ceiling + headroom for recovery
  queries.promql           the five signals that make a silent scheduling failure visible
```

**Read them in that order.** The engine limits set the ceiling (§1), flow control decides who
reaches it (§2), the SLO decides whether reaching it was worth anything (§3), capacity planning
sizes the fleet against that SLO (§4), and §5 is how you find out whether any of it is true.

---

## 1. `engine-scheduler.yaml` — the engine's limits

```yaml
# vLLM-style launch configuration. Field names follow the engine's own vocabulary; values are
# illustrative placeholders sized for the experiment-6 traffic of run.py (800 prompt + 200 output).
#
# THE TWO LIMITS ARE NOT INDEPENDENT, which is the first thing to get right.
#   max_num_seqs           the slot count -- how many sequences may be IN FLIGHT at once
#   max_num_batched_tokens the per-step token budget -- the roofline ceiling for one iteration
# A deep batch shares the token budget, so raising max_num_seqs without raising the budget
# produces a deeper queue served at the same rate: more latency, identical throughput (HLD 8).

model: <your-model>
tensor_parallel_size: 1

# ---- slots ----------------------------------------------------------------------------
max_num_seqs: 32
#   Not 128. Past the goodput knee, extra slots only redistribute a fixed budget more thinly:
#   run.py exp 6 reaches the 800 tok/step ceiling at 8 slots and P99 rises monotonically after.
#   Provisioned concurrency and served throughput are DECOUPLED past the knee.
#   START at 8, raise only while goodput at the binding SLO improves.

# ---- the token budget -----------------------------------------------------------------
max_num_batched_tokens: 8192
#   This is the bandwidth-bound ceiling AND the chunking budget. It sets the prefill chunk:
#       prefill_chunk_size ~= max_num_batched_tokens - (concurrent_decode_seqs x 1)
#   At 16 decoding sequences the chunk is 8,176 tokens (run.py exp 3). Sizing this from decode
#   throughput alone -- without the concurrent-decode subtraction -- is the standard config error.

# ---- chunked prefill ------------------------------------------------------------------
enable_chunked_prefill: true
#   ON here because this deployment has a prompt-length TAIL. The trade, measured (exp 3):
#     a 32k prompt unchunked blocks every stream for ~328 decode-steps; chunked, the spike is
#     one chunk and the prompt finishes 4 steps later.
#   TTFT gets slightly WORSE, ITL gets dramatically BETTER. A team dashboarding on TTFT alone
#   will turn this off and be right about the metric and wrong about the system (HLD 4).
#   TURN IT OFF if the prompt distribution is genuinely uniform and short -- then it is pure
#   scheduling overhead and buys nothing.

# ---- preemption -----------------------------------------------------------------------
preemption_mode: recompute
#   recompute | swap.  recompute is free and scales with context length; swap needs a tier and a
#   fabric (T07) but has a bounded restore cost. Choose by comparing re-prefill cost against the
#   tier's round trip -- at long contexts recompute is quadratic and swap wins.
#
#   THIS MUST NOT BE OFF. A pool filled to 100% with no preemption cannot recover, because the
#   request that would free space is the one that cannot be admitted. run.py exp 5 demonstrates
#   it: at an 88,000-token cap the run NEVER completes (14,880 refusals). See HLD 5.

# ---- the recovery slack ---------------------------------------------------------------
gpu_memory_utilization: 0.90
#   The engine fills this fraction of VRAM. Note the gap between it and the 0.80 KV gate below:
#   that 10-20% is not waste, it is the slack the preemption path needs to operate. A pool at
#   1.00 has no room to evict INTO.
```

---

## 2. `flow-control.yaml` — the router's admission and fairness policy

```yaml
# llm-d EPP-style flow control. The corpus places admission and placement AT THE ROUTER [T]
# (llm-d), not in the engine -- because fairness and bands are FLEET-wide properties that no
# single engine can arbitrate. The engine limits in §1 remain the backstop.

flowControl:
  # ---- THE SATURATION TEST -- corpus values -------------------------------------------
  saturation:
    kvThreshold: 0.80            # [T] llm-d: "when the KV cache is 80% full I declare the
                                 # cluster is saturated". Not 1.0. The headroom is the design.
    activeRequestThreshold: 8    # [T] "the number of active requests ... on average is more
                                 # than eight". Either condition alone triggers -- alternatives,
                                 # not a conjunction.
    onSaturation: queue          # queue | refuse
    #   QUEUE. The corpus: "the EP starts queuing the request in the flow control" [T]. A
    #   queued request preserves the load AND the evidence of it; the queue depth is what the
    #   autoscaler (T15) consumes. Refusing converts a capacity problem into a user error and
    #   destroys the signal at the same time.

  # ---- PRIORITY BANDS ------------------------------------------------------------------
  bands:
    premium:      [interactive, chat, copilot]     # dispatched under saturation
    bestEffort:   [batch, eval, embedding-backfill]  # waits for capacity, then retries
    #   The corpus's contract [T]: under saturation "only the premium traffic is dispatched
    #   and the rest are not dispatched until you have capacity ... to make sure that your
    #   customer-facing or interactive workloads don't suffer while your batch processing
    #   workloads can wait and then retry later".
    #
    #   RESERVE AND RELEASE, not hard reservation. Measured (run.py exp 7, 4 slots):
    #     ungated (FCFS)      premium 1.8   best-effort 1.9
    #     gated (bands)       premium 1.0   best-effort 2.2   idle 0
    #     hard reservation    premium 1.0   best-effort 3.8   idle 12
    #   Hard reservation protects premium best AND BURNS 12 SLOTS while premium is idle.
    #   Whatever premium does not use must go back to everyone else.
    maxReservedFraction: 0.50
    #   Ceiling on the premium share under saturation, so a flooding premium tenant cannot
    #   take the whole fleet. Without it, "premium" is a promise with no bound on its cost.

  # ---- FAIRNESS ------------------------------------------------------------------------
  fairness:
    policy: leastAttained        # fcfs | priorityBand | leastAttained | turnPriority
    scope: program               # program | request
    #   PROGRAM, always. The corpus calls this "agentic program aware fairness" [T]. A
    #   per-REQUEST policy rewards the program that issues the most requests -- exactly the
    #   program that needs throttling -- because a wide fan-out refills its own queue the
    #   instant any turn retires and is therefore permanently at the head of it.
    key: absoluteService         # absoluteService | shareOfDemand
    #   ABSOLUTE SERVICE RECEIVED. This is the line that decides whether the policy works.
    #   shareOfDemand is PERMANENTLY SMALLEST FOR THE LARGEST PROGRAM (its ratio stays near
    #   zero for a long time), so it is permanently served first and reproduces the exact
    #   starvation the policy exists to fix. Absolute service needs no knowledge of program
    #   size -- which is precisely why it protects SHORT sessions.
    #
    #   DIRECTION WARNING. The corpus's scenario has the LARGE session monopolising dispatch
    #   ("a small session and a [large] session ... takes all the dispatch cycles ... the
    #   shorter sessions are starving" [T]), so least-attained service protects the SMALL
    #   sessions. It is NOT "protect the long agent from short requests". The reverse reading
    #   is the most commonly inverted claim in this topic.

  # ---- TURN PRIORITY -- a DIFFERENT resource, gated ------------------------------------
  turnPriority:
    enabled: true
    activateAboveKv: 0.70
    #   Turn priority finishes near-done programs so their KV is EVICTED [T]: "at turn 100
    #   most probably it is likely to finish faster ... if you prioritize those agents, those
    #   agents will finish and then evict their KV cache".
    #
    #   IT MANAGES MEMORY, NOT COMPUTE. The corpus is explicit that the two mechanisms
    #   address different bottlenecks: "one addresses the compute saturation, other addresses
    #   the KV cache saturation" [T]. They are NOT two settings in one slot.
    #
    #   GATE IT. With no memory pressure, turn priority is pure starvation of long-horizon
    #   agents -- run.py exp 5, no cap: the long program finishes at cycle 121 against 81.
    #   Below this threshold it must be OFF.
```

---

## 3. `slo-policy.yaml` — the objective

```yaml
# THE SLO IS PART OF THE SCHEDULER'S CONFIGURATION, not a dashboard setting. The batch size that
# maximises goodput is a function of the SLO as much as of the engine (HLD 8), so tuning the
# scheduler without a binding SLO is tuning against nothing.

objective:
  primary: goodput               # goodput | throughput
  #   GOODPUT, NOT THROUGHPUT. Throughput rises monotonically with batch size and therefore can
  #   NEVER tell you that you have batched too far -- it goes quiet at the ceiling, which looks
  #   like success. Goodput peaks and falls, which is the signal you need.
  #
  #   THE EXCEPTION: for genuinely offline batch traffic with no SLO, throughput IS the right
  #   objective and the peak-goodput slot count under-provisions. The metric follows the
  #   workload, not the other way round.

  slos:
    ttft_ms: 800                 # time to first token -- PRE-FILL bound (compute)
    itl_ms: 50                   # inter-token latency -- DECODE bound (bandwidth)
    e2e_ms: 15000
  #   Two SLOs, because the two phases have different bottlenecks. A single end-to-end SLO
  #   cannot tell you which lever to pull: the corpus's rule is "trim the prompt for prefill or
  #   batch harder for decode" [T], and you can only apply it if the two are measured apart.

  # ---- THE ONE THING TO VERIFY BEFORE TUNING ANYTHING -----------------------------------
  requireBinding: true
  #   An SLO every configuration meets is not a constraint, it is decoration. run.py exp 6 is
  #   the demonstration: at a 25-step SLO EVERY batch size from 4 to 128 slots conforms, so
  #   goodput@25 is 100%-ish everywhere and discriminates between nothing. At SLO 10 and 15 the
  #   curve peaks at 8 slots and falls to 0% by 128.
  #
  #   THIS FAILURE IS THE ONE TEAMS ACTUALLY HIT, because the SLO is usually set AFTER looking
  #   at the latency distribution -- which guarantees it is achievable, which means the
  #   subsequent tuning targets a constraint that no longer constrains.
  #
  #   VERIFY: at your current configuration, is goodput at the primary SLO comfortably below
  #   100%? If it is ~100%, TIGHTEN THE SLO until it bites, then tune.

  reportAtSlos: [10, 15, 25]
  #   Report goodput at SEVERAL SLOs. A single-SLO column invites the reader to treat the peak
  #   as a property of the hardware, which it is not -- it moves with both the traffic mix and
  #   the SLO.
```

---

## 4. `capacity-plan.yaml` — the sizing arithmetic

```yaml
# How many slots should this deployment run? The answer is counter-intuitive and it is the
# single largest cost lever in this blueprint.

planning:
  # ---- STEP 1: the engine's ceiling -----------------------------------------------------
  engineDecodeCeilingTps: 800    # decode tokens/step at saturation -- measure this, do not guess
  prefillRateTps: 8000           # prefill tokens/step

  # ---- STEP 2: the smallest slot count that REACHES the ceiling -------------------------
  slotsAtCeiling: 8
  #   run.py exp 6: at 4 slots the engine is capped at 400 tok/step -- exactly half its ceiling,
  #   because the batch is too shallow to saturate the bandwidth. At 8 slots it reaches 800 and
  #   STOPS there. Every slot beyond 8 adds latency and no throughput.

  # ---- STEP 3: headroom for the recovery path -------------------------------------------
  headroomFraction: 0.20
  #   The corpus's 80% KV gate [T] expressed as reserved capacity. This is NOT conservatism:
  #   it is the slack preemption needs to evict INTO, and the space the deadlock in HLD 5
  #   requires to avoid. A pool provisioned to 100% cannot recover, because the request that
  #   would free space is the one that cannot be admitted.

  # ---- STEP 4: burst headroom ------------------------------------------------------------
  burstMultiplier: 1.5
  #   Provisioned concurrency x 1.5 for arrival bursts. Cheap, because past the knee the
  #   marginal slot costs latency rather than hardware.

  # ---- THE RESULT ------------------------------------------------------------------------
  recommendedMaxNumSeqs: 12
  #   8 (ceiling) x 1.5 (burst). Compare against the intuitive answer of 128: a 10x difference
  #   in provisioned concurrency for IDENTICAL throughput, with P99 24 instead of 34 (exp 6).
  #
  #   WHY THIS IS THE COST LEVER: provisioned batch depth drives KV occupancy, which drives
  #   HBM, which drives the GPU count. Teams that size concurrency from a peak-throughput
  #   target rather than from the goodput knee buy roughly an order of magnitude more hardware
  #   than the workload requires.

  # ---- DO NOT AUTOSCALE ON UTILISATION ----------------------------------------------------
  autoscalingSignal: queueDepth
  #   NEVER gpuUtilisation. A continuous batcher KEEPS UTILISATION HIGH BY DESIGN [T] -- that is
  #   the entire point of the mechanism -- so a utilisation-driven autoscaler sees a healthy
  #   fleet right up until the queue explodes. Scale on:
  #     - saturation-gate state (kv / active)
  #     - flow-control queue depth, by band
  #     - goodput at the binding SLO
  #   The gate exists partly to EXPOSE these; a gate that queues without exporting the queue
  #   depth is only half built (HLD 7).
```

---

## 5. `queries.promql` — the five signals that make the silent failures visible

```promql
# ---- 1. SLOT IDLE FRACTION -- the admission-discipline signal --------------------------
# THE most important query in this file. Static batching is not slower in aggregate; it is
# unable to use capacity it already has. run.py exp 2: identical makespan, and the only
# observable difference is that the static run has 0% idle while finishing four short
# sequences at step 9 instead of step 3.
#
# A throughput or makespan dashboard shows NOTHING here -- which is the whole of the
# "continuous batching didn't help" story. This is the gauge that shows it.
sum(rate(vllm:num_running_seqs_sum[1m]))
  / (vllm:num_running_seqs_limit * count(vllm:num_running_seqs_limit))
#   Sustained well below 1.0 with a non-empty queue = slots are idling while work waits.
#   THAT COMBINATION IS THE BUG. Idle slots with an empty queue are just a small deployment.

# ---- 2. THE QUEUE, BY BAND -- the autoscaling signal (T15) -----------------------------
# The saturation gate converts a latency collapse into a queue; this reads the queue. A gate
# that queues without exporting depth has removed the load AND the evidence.
sum by (band) (llmd_flowcontrol_queue_depth)
#   premium climbing  = the fleet is genuinely short of capacity -> scale
#   bestEffort only   = the gate is working as designed -> do NOT scale, that is the point
#   Distinguishing these two is why the band label is not optional.

# ---- 3. TTFT AND ITL SEPARATELY -- which phase is the bottleneck ------------------------
# The corpus's rule: "trim the prompt for prefill or batch harder for decode" [T]. You can
# only apply it if the two phases are measured apart. A single e2e latency can tell you that
# something is slow and never which lever to pull.
histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
histogram_quantile(0.99, sum by (le) (rate(vllm:time_per_output_token_seconds_bucket[5m])))
#   TTFT p95 up with ITL flat  -> PREFILL is the bottleneck: chunking, prompt trimming, or
#                                 the token budget is too small for the concurrent batch
#   ITL p99 up with TTFT flat  -> DECODE is the bottleneck: the batch is past the knee
#   BOTH up                    -> the engine is saturated; apply the gate (HLD 7)
#
#   AND CORRELATE ITL AGAINST PROMPT LENGTH. An ITL spike that tracks long prompts is
#   head-of-line blocking and chunked prefill is the fix (exp 3); an ITL spike that tracks
#   BATCH DEPTH is over-batching and the slot count is the fix (exp 6). Same symptom,
#   opposite levers.

# ---- 4. GOODPUT AT THE BINDING SLO -- the objective -------------------------------------
# Throughput goes flat at the ceiling and looks like success. This is the metric that tells
# you the batch is too deep.
sum(rate(vllm:request_success_total[5m]))
  /
sum(rate(vllm:request_total[5m]))
#   ...restricted to requests meeting the SLO. Report against SEVERAL SLOs (exp 6 peaks at 8
#   slots for SLO 10 and 15, and never peaks at all for SLO 25).
#
#   IF THIS IS ~100% AT EVERY CONFIGURATION, THE SLO IS DECORATION. Tighten it before reading
#   the chart -- otherwise you are tuning against a constraint that does not exist.

# ---- 5. PER-TENANT LATENCY -- the starvation signal -------------------------------------
# Starvation is NOT an error. Nothing raises, nothing logs, the fleet mean barely moves while
# one tenant's latency grows 12x (exp 4). A fleet-wide p99 CANNOT see it.
#
# ALERT ON THE RATIO, NOT ON THE ABSOLUTE. A slow tenant may simply be a heavy tenant.
histogram_quantile(0.95, sum by (tenant, le) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))
  /
histogram_quantile(0.50, sum by (tenant, le) (rate(vllm:e2e_request_latency_seconds_bucket[5m])))
#   A tenant whose p95/p50 ratio is climbing while the fleet is stable is being starved by
#   another program's fan-out. Fix the KEY, not the capacity: least-attained service on
#   ABSOLUTE service, scoped to the PROGRAM (HLD 6).
#
#   AND TRACK SERVICE SHARE. service_tokens by tenant / demand by tenant. A tenant's share
#   falling below its demand share while it is queueing is the same finding, one layer up.

# ---- 6. KV REFUSALS -- the deadlock's early warning ------------------------------------
rate(llmd_flowcontrol_kv_refused_total[5m])
#   Should be ~0 in a healthy deployment. A CLIMBING COUNT means the pool is over-committed
#   and the recovery path is being exercised. In the deadlock of exp 5 the count reaches
#   14,880 -- and it is the ONLY numeric evidence the run produced, because nothing raised.
#   Pair it with headroom: kv_usage sustained above 0.90 leaves no room to evict into.
```

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` — Pravin (IBM Research): flow control at the EP; the 80% KV and >8 active-request saturation tests; queuing on saturation; premium/best-effort bands; agentic-program-aware fairness on attained service; turn priority and KV eviction; "one addresses the compute saturation, other addresses the KV cache saturation".
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — continuous batching; the prefill/decode split and "trim the prompt for prefill or batch harder for decode"; vLLM's built-in continuous batching needing "no manual batching logic"; cost per thousand requests alongside p95 latency and TTFT as the tracking set.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/04-batching-strategies.md` — chunked prefill and the "stall"; in-flight batching; continuous-batching throughput range.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/05-paged-attention.md` — the pool whose exhaustion is the preemption trigger in §1.

Every threshold in these files is `[D]` illustrative. The corpus supplies the mechanisms and three
numbers (0.80, 8, and the band contract); everything else must be replaced with measured values.
Measured figures quoted in the comments are reproducible from [`../run.py`](../run.py).
