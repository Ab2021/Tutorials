# T06 — Production Artifacts (reference-grade)

> **Not executed by this repo.** This blueprint is a *model*, not a service — so its "production
> artifacts" are the inputs it consumes and the decisions it gates, not manifests. `[T]` transcript ·
> `[R]` repo · `[D]` derived.

```
production/
  README.md            this file
  hardware-inventory.yaml   peak FLOPs (DENSE) and bandwidth per part — the model's hardware input
  slo-templates.yaml        two-part SLOs; goodput, not throughput
  traffic-profile.yaml      the prompt/output distributions that size everything
  config-gate.yaml          the check that runs BEFORE a config change is applied
  queries.promql            goodput, TTFT/ITL split, KV saturation
```

---

## 1. `hardware-inventory.yaml` — the input that is most often wrong

```yaml
# CRITICAL: `peak_flops` MUST be the DENSE figure at the working precision.
# Vendor datasheets quote sparse and low-precision peaks that are 2-4x what your kernel
# achieves. Substituting one moves machine_balance by an order of magnitude and flips the
# classification -- silently. sim/roofline.assert_dense_peak() exists to catch this.
parts:
  - name: h100-class
    peak_flops_dense_fp16: 989.5e12    # [D] ILLUSTRATIVE -- verify against your own part
    memory_bw: 3.35e12                 # bytes/s
    memory_capacity: 80e9
    # machine_balance = 989.5e12 / 3.35e12 = 295.4 FLOPs/byte  <- compute this yourself
  - name: mi300-class
    # AMD parts matter here: the corpus's WideEP talk reports 192 GB (MI300) and 288 GB
    # (MI355) HBM per GPU [T], which changes the KV budget far more than the FLOPs do.
    memory_capacity: 192e9
```

**Why the note is at the top of the file.** The single most common error in this whole topic is
comparing a workload's intensity against a *quoted* peak rather than the achievable one. It produces
a confident, wrong answer, and it produces it silently.

---

## 2. `slo-templates.yaml` — two parts, always

```yaml
# A one-part SLO cannot express streaming quality. A request with excellent TTFT and terrible
# ITL fails, and so does the reverse. Both parts are required.
slos:
  interactive-chat:
    ttft_ms: 400          # [D] prefill-bound
    itl_ms: 25            # [D] decode-bound
    unit: goodput         # NOT throughput
  agentic:
    # Agentic traffic is ~98% PREFILL tokens [T], so the chat SLO is the wrong instrument.
    request_latency_ms: 30000        # [D] the whole request
    program_completion_s: 600        # [D] the whole multi-turn program
    kv_hit_rate_min: 0.70            # [D] per session — an agent that cannot reuse KV loses
  batch-offline:
    throughput_only: true            # the only case where throughput is the right unit
```

**Why agentic gets a different template.** The corpus is explicit that TTFT/ITL are the chat metrics
and that agents need **request latency**, **program completion time** and **KV hit rate per session** `[T]`.
Applying the chat SLO to an agent fleet measures the wrong thing at both ends: it over-weights
per-call latency and completely misses cache reuse.

---

## 3. `traffic-profile.yaml` — P95 is the sizing input

```yaml
# Both percentiles are MANDATORY. Sizing on the mean is the most common modelling error, and
# its symptom is "latency fine in the load test, bad in production" -- because a load test
# with uniform prompt lengths has no skew.
profile:
  prompt_tokens:
    p50: 1200
    p95: 8000        # <- this is what sizes the KV budget
    p99: 32000
  output_tokens:
    p50: 180
    p95: 900         # <- this is what sizes the latency budget
  qps: 40
  notes: |
    Real traffic is skewed. A load generator that samples prompt length uniformly will
    under-provision KV by roughly the ratio p95/p50 and will never reproduce the production
    P95 latency. Profile from the gateway (T14) or the traces (T17), not from a harness.
```

---

## 4. `config-gate.yaml` — the check that runs before a change

```yaml
# The model's highest-value use: run it at CHANGE TIME, not incident time. A config change
# that moves a workload across the machine-balance line makes an optimisation inapplicable,
# and the symptom is "the technique did not help" rather than an error.
gates:
  - name: dtype-change
    trigger: [quantization, dtype]
    checks:
      - id: classification-unchanged
        # KEY POINT: quantizing weights halves bytes_moved and therefore decode TIME, but
        # does NOT change the classification. Decode was memory-bound and remains memory-bound.
        rule: "classify(phase=decode) must remain memory-bound"
      - id: kv-budget-recomputed
        # KV quantization and weight quantization are DIFFERENT changes. This one moves the
        # concurrency ceiling; the other does not.
        rule: "recompute max_concurrency with the new kv dtype_bytes"

  - name: context-window-change
    trigger: [max_model_len, context]
    checks:
      - id: attention-crossover
        # Past n_ctx = hidden the dominant term changes from MLP to attention. An 8k-validated
        # model deployed at 128k fails because the PROBLEM changed, not because of a bug.
        rule: "attention_mlp_ratio(n_ctx, hidden) reported; flag if it crosses 1.0"
      - id: kv-per-sequence
        rule: "kv_total(n_ctx) reported in GB; flag if max_concurrency drops below 1"

  - name: batch-policy-change
    trigger: [max_num_seqs, batch]
    checks:
      - id: goodput-not-throughput
        # The check that prevents the classic mistake. A larger batch raises throughput
        # monotonically and can lower goodput.
        rule: "goodput(batch, slo) evaluated; block if goodput falls"
```

**The three gates are ordered by how often they fire in practice**: dtype changes are routine,
context changes are occasional, batch changes are constant — and the batch gate is the one guarding
the mistake teams actually make.

---

## 5. `queries.promql` — measure the two phases separately

```promql
# GOODPUT -- the headline metric. Not throughput.
# Fraction of requests meeting BOTH parts of the SLO, over a rolling window.
sum(rate(llm_request_duration_seconds_bucket{le="0.4", phase="ttft"}[5m]))
  /
sum(rate(llm_requests_total[5m]))

# THE PHASE SPLIT -- the diagnostic that tells you which dashboard to open.
# TTFT high + ITL normal  -> prefill path: prompt length, prefix cache hit rate
# TTFT normal + ITL high  -> decode path: KV utilisation, batch, quantization
histogram_quantile(0.95, sum(rate(llm_ttft_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(llm_itl_seconds_bucket[5m])) by (le))

# KV SATURATION -- the capacity constraint, not the compute constraint.
# The corpus's saturation gates are KV ~80% and >8 active requests [T].
llm_kv_cache_usage_ratio > 0.80
llm_num_requests_running > 8

# PREFIX CACHE HIT RATE -- the lever on the prefill path, and the signal for bad routing [T].
sum(rate(llm_prefix_cache_hit_tokens_total[5m]))
  /
sum(rate(llm_prompt_tokens_total[5m]))

# THE DIVERGENCE -- throughput rising while goodput falls is the failure this blueprint exists
# to prevent. Two panels, NEVER on the same axis.
sum(rate(llm_output_tokens_total[5m]))                       # throughput
avg(llm_goodput_ratio)                                       # goodput
```

**Why the last two panels must not share an axis.** Their ranges are unrelated, and plotting them
together creates the visual impression that they move together — which is exactly the misreading the
goodput metric exists to prevent.

---

## Sources

- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_1_Introduction_to_Language_Models_and_Inference.txt` — model dimensions, GQA
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` — saturation gates (KV 80%, >8 active), prefix cache hit rate per pod, agentic metric set
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt` — MI300/MI355 HBM capacities
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt` — "decode is solved; the hard part is the first token"
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — "optimization without measurement is just guessing"

**All configuration and all hardware parameters here are `[D]`/illustrative** except where a line
carries `[T]`. The corpus supplies the mechanisms, the two-phase framing and the saturation gates; it
supplies no configuration file and no hardware specification.
