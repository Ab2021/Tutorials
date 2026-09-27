# T04 — Production Artifacts (reference-grade)

> **These are not executed by this repo.** They document the shape of a real deployment so the
> blueprint is buildable, not aspirational. Nothing here has been run against a GPU; every number
> in a comment is either a corpus citation `[T]`, a repo citation `[R]`, or marked `[D]`.
> See [../HLD.md](../HLD.md) for the capacity model and [../LLD.md](../LLD.md) §8 for the
> configuration surface these files instantiate.

```
production/
  README.md                     this file
  controller.yaml               the test-time-compute controller: budgets, thresholds, routing
  vllm-sampling.yaml            engine-side: how to actually serve `n > 1` sampling cheaply
  length-policy.yaml            the always-leave-room-for-the-answer invariant, as config
  eval-gate.yaml                the CI gate: any change to the above reruns evals
  otel-collector.yaml           the spans that make a budget decision auditable
  k8s-canary.yaml               rollout shape for changing a sampling policy in production
```

---

## 1. `controller.yaml` — the decision surface

```yaml
# T04 test-time-compute controller.
# Every knob here maps to a field in LLD §8. Provenance is marked per line.
adaptive:
  enabled: true
  alpha: 3.0            # [T] CMU lecture 9 — the corpus states this Dirichlet prior
  threshold: 0.95       # [T] CMU lecture 9 — P(leader beats runner-up) to stop
  batch: 2              # [D] samples drawn between stopping checks
  cap: 16               # [D] hard ceiling; CAP_HIT beyond this escalates
  min_samples: 4        # [D] floor; without it a unanimous wrong run stops after 1 draw
  fixed_n_fallback: 32  # [T] CMU lecture 12 — the lecturer's example sizing (fits one batch)

routing:
  # Which requests get test-time compute at all. This is the cost control.
  easy:   {strategy: greedy,            samples: 1}    # [D]
  medium: {strategy: adaptive,          samples: null} # adaptive decides
  hard:   {strategy: fixed,             samples: 32}   # [T] the batch-fitting size
  # Escalation on CAP_HIT — the LLD §7 contract requires this be explicit.
  on_cap_hit: escalate_to_stronger_model   # accept | escalate_to_stronger_model | escalate_to_human

self_correction:
  # OFF by default. This is the design decision, not a default we forgot to change.
  # LLD §3.3: correction helps iff f_w/f_c > acc/(1-acc), a ratio that grows without bound.
  enabled: false       # [D] derived from the corpus's negative result [T] CMU lecture 8
  f_c: 0.20            # [D] illustrative P(correct -> wrong) per round
  f_w: 0.30            # [D] illustrative P(wrong -> correct) per round
  rounds: 2
  # Enable only where breaks_even() returns True for the MEASURED (acc, f_c, f_w).
  oracle_checker: false  # if true, f_c = 0 structurally (external verifier, not intrinsic)

budget:
  max_length: 8192       # [D]
  room_for_answer: 1024  # [D] reserved; never spent on reasoning — LLD §2.4 invariant
  # The exceed-rate crash (HLD §5.6) is prevented by the reservation, not by hope.
```

**The two lines that matter most.** `min_samples: 4` and `on_cap_hit: escalate_to_stronger_model`.
The first is a mitigation for the HLD §5.4 blind spot — the stopping rule reads *agreement*, not
correctness, so it fires fastest on the problems where everyone is confidently wrong. The second is
the contract that a `CAP_HIT` result is never silently accepted as a plurality answer. Silently
accepting it is how a system comes to report confidence it does not have.

---

## 2. `vllm-sampling.yaml` — serving `n > 1` without paying `n×` naively

The controller's cost is `samples × per-sample cost`. The engine-side job is to make that multiplier
as small as honesty allows.

```yaml
# Engine-side guidance for high-fan-out sampling. Reference-grade; values are illustrative.
engine:
  # All n samples of one prompt share a prefix. Prefix caching is the single biggest lever:
  # the prompt is prefilled ONCE and the n chains diverge after it. [T] vLLM
  enable_prefix_caching: true

  # Chunked prefill keeps a long shared prompt from blocking other requests' decode. [T] vLLM
  enable_chunked_prefill: true

  # Continuous batching: the n siblings of one request batch with everyone else's tokens. [T]
  max_num_seqs: 256

  # KV pressure is the real constraint at fan-out n. n samples of one prompt = n KV chains.
  gpu_memory_utilization: 0.90
  max_model_len: 8192        # must equal controller.budget.max_length [D]

sampling:
  # If using a reasoning model, the "thinking budget" is the length policy in disguise.
  # [T] CMU lecture 9 — thinking length is where R1-style gains come from, and where the
  # exceed-rate crash lives.
  temperature: 1.0           # [T] CMU lecture 9 — reasoning models are trained at temp 1.0
  # Do NOT set temperature 0 and expect determinism: it is still non-deterministic. [T] CMU lec 2
```

**Why `temperature: 1.0` and not 0.** Two corpus facts collide here. Reasoning-model gains are
realised at the temperature the model was trained at `[T]` CMU lecture 9, and temperature 0 is
*still non-deterministic* on real hardware `[T]` CMU lecture 2. Setting 0 therefore buys neither
determinism nor the reasoning gains — it buys an illusion of the first and the loss of the second.

---

## 3. `length-policy.yaml` — the reservation, spelled out

```yaml
length_policy:
  # LLD §2.4 invariant: used + room_for_answer <= max_length, always.
  mode: reserve_tail        # [D]
  max_length: 8192
  reserve_for_final_answer: 1024

  # If the reasoning chain is still running when `max_length - reserve` is reached:
  on_budget_pressure:
    action: force_conclude   # [D] truncate with a "therefore, the answer is" prompt scaffold
    # alternatives:
    #   hard_truncate   -> the exceed-rate crash: no room left to emit an answer
    #   continue        -> blows the budget and returns nothing
    #   escalate        -> hand to a bigger-budget path

  # Training-side lever, if you own the model. [T] CMU lecture 9 — this is a real intervention.
  reward_shaping:
    enabled: false
    # cosine_reward(correct, length): shape toward a target length, multiplied by correctness.
    # LLD §3.4 — this trades accuracy against brevity. It is not a free win; sweep it.
    target_length: 2048
```

**The exceed-rate crash, precisely.** A run that spends its entire budget on reasoning and is then
cut off has no room to emit an answer, so it returns nothing — not a worse answer, *nothing*. The
reservation converts a hard failure into a soft one. This is the single highest-value line in the
file.

---

## 4. `eval-gate.yaml` — the rule the corpus states most often

```yaml
# CI gate. Any change to controller.yaml, vllm-sampling.yaml or length-policy.yaml triggers it.
gate:
  trigger_paths:
    - production/controller.yaml
    - production/vllm-sampling.yaml
    - production/length-policy.yaml
  must_pass:
    - name: accuracy-regression
      # Reason: "always rerun your evals after quantizing" [T] LLMOps cost talk.
      # The same rule applies to every knob above, not only quantization.
      threshold: no_regression_vs_baseline
    - name: cost-regression
      # Adaptive sampling's whole claim is a LOWER cost multiplier. Gate it.
      metric: mean_samples_per_request
      threshold: <= baseline * 1.10
    - name: cap-hit-rate
      # A rising CAP_HIT rate means the threshold is unreachable. That is a routing bug,
      # not a compute shortage — LLD §4.
      metric: cap_hit_fraction
      threshold: <= 0.05
  # Assert properties, never exact strings: outputs vary at temperature 0. [T] CMU lecture 2
  assertion_style: property_based
```

---

## 5. `otel-collector.yaml` — making a budget decision auditable

```yaml
# One span per sampling fan-out. The attributes below are what let you answer
# "why did this request cost 16 samples and still get it wrong?" — the HLD §5.4 question.
processors:
  attributes:
    actions:
      - key: ttc.strategy            # adaptive | fixed | greedy
        action: upsert
      - key: ttc.samples             # how many were actually drawn
      - key: ttc.stopped_early       # bool — the adaptive win
      - key: ttc.leader_wins         # the stopping statistic at the moment of the decision
      - key: ttc.state               # CONVERGED | CAP_HIT  (LLD §4)
      - key: ttc.truncated           # bool — did the budget pressure the run
      - key: ttc.correction_enabled  # bool — was self-correction on for this request
```

**`ttc.leader_wins` at the moment of decision is the attribute that makes the blind spot visible.**
A high `leader_wins` with a wrong answer is the §5.4 failure, and without this attribute it is
invisible in aggregate — you see accuracy dip and cannot tell whether the samples were too few or
the agreement signal was lying. It was lying.

---

## 6. `k8s-canary.yaml` — changing a sampling policy safely

```yaml
# A sampling-policy change is a behaviour change. Roll it out like code.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ttc-controller
spec:
  replicas: 10
  strategy:
    canary:
      steps:
        - setWeight: 5      # 5% of traffic sees the new policy
        - pause: {duration: 30m}
        - setWeight: 25
        - pause: {duration: 1h}
        - setWeight: 100
      # Promote only if BOTH hold. A policy can win on cost and lose on quality.
      analysis:
        - name: cost-per-request
          args: {metric: ttc.samples, threshold: "not worse than baseline"}
        - name: accuracy-on-shadow-set
          args: {threshold: "no regression"}
```

---

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_8_Self-Refine_and_Self-Correction_Methods.txt` — the self-correction negative result
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_9_Reasoning_Models.txt` — `alpha`, `threshold`, thinking-length, temperature 1.0
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts/CMU_LLM_Inference_12_Reward_Models_and_Best-of-N.txt` — `n = 32`, batch-fitting
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt` — non-determinism at temperature 0
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — the eval-rerun rule, batching and prefix caching levers
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/LLM_Observability_Traces_Spans_OpenTelemetry_for_AI_Apps.txt` — span attributes

**All configuration in this directory is `[D]` unless a line carries `[T]` or `[R]`.** The corpus
states the mechanisms and the two threshold values; it states no configuration file.
