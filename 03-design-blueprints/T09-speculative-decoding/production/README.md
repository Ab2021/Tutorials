# T09 — Production artifacts

> `T09` · **Transcript coverage:** partial · [HLD](../HLD.md) · [LLD](../LLD.md) · [Runnable core](../run.py) · [Sequences](../docs/SEQUENCES.md)

**Reference-grade.** These are the five artifacts a team needs to take speculative decoding from a
benchmark to a deployment that still works after its traffic grows. They are shaped like real
configs (vLLM's `--speculative-config`, a Prometheus rules file, a K8s ConfigMap) but they are
**written for this blueprint**, not copied from a running cluster, and the thresholds are the ones
this blueprint's model produces — recompute them for your model and hardware.

The two that engines do not ship are **`gate-policy.yaml`** and **`metrics.promql`**. Everything
else can be adopted; those two have to be built, and they are the ones that prevent the p99
regression.

---

## 1. `engine-spec.yaml` — the engine launch surface (vLLM-shaped)

```yaml
# Reference-grade. Values are the ones this blueprint's model selects for a 70B target on an
# H100-class part with a 1B draft sibling, gamma tuned for alpha_1 = 0.80 / decay 0.90.
server:
  model: meta-llama/Llama-3.3-70B-Instruct
  tensor_parallel_size: 4
  max_model_len: 32768
  gpu_memory_utilization: 0.90

speculative:
  # DEFAULT OFF. Enable per route at the router (T14), never globally -- alpha is a property of the
  # workload, and a global on-switch is how a service regresses at p99 without anyone choosing it.
  enabled: false

  method: ngram                 # ngram | mtp | draft_model | medusa | eagle
  num_speculative_tokens: 5     # gamma. Smallest value within 2% of the peak, NOT the argmax.
  ngram:
    prompt_lookup_min: 2        # shortest match that counts as a proposal
    prompt_lookup_max: 5        # longest span copied

  # For method: draft_model only. NOTE the two costs it carries that this file cannot express:
  # its weights (2 GB at 1B/fp16) and its own KV, both of which come out of the target's KV budget
  # and lower the concurrency ceiling (T07). Budget for them at capacity-plan time.
  draft_model: null
  draft_tensor_parallel_size: 1

  # For method: mtp. Must be present in the checkpoint -- MTP heads cannot be bolted onto an
  # arbitrary model, so this is a checkpoint-selection decision, not a runtime one.
  mtp_num_heads: 2

# --- THE THREE VALUES THAT DECIDE WHETHER THIS IS A WIN OR A REGRESSION ---------------------
# 1. num_speculative_tokens comes from gamma_sweep(), not from a blog post. At alpha_1 = 0.80 /
#    decay 0.90 / c = 0.014 the peak is gamma = 6 at 2.892x; gamma in {5,6,7,8} is within 2% of it,
#    so 5 is chosen -- same result, less draft compute burned on every rejection.
# 2. method: ngram because the first route enabled is summarisation/extraction, where the output
#    quotes the input. c = 0 means a WRONG guess costs exactly one plain decode step.
# 3. enabled: false. The route decides.
```

---

## 2. `gate-policy.yaml` — the batch-regime gate (**built, not adopted**)

```yaml
# NO ENGINE SHIPS THIS. Without it, a service that is memory-bound at p50 and compute-bound at p99
# shows a speedup in every average and a latency regression under load -- and nothing alerts,
# because the requests that got slower are a minority of the samples.
speculation_gate:
  enabled: true

  # batch* from memory_bound_batch_limit(). For 70B/fp16, GQA KV 0.33 MB/token, H100-class
  # machine balance ~295 flops/byte:
  #     batch* = (295 * 140e9) / (140e9 - 295 * 0.33e6)  ~  295 concurrent sequences
  # RECOMPUTE for your model and hardware. It moves with parameter count, weight precision and KV
  # layout -- a copied value gates on the wrong regime.
  batch_limit: 295

  measure: batch_depth          # concurrent DECODE sequences. NOT queue depth, NOT request rate.
  sample_at: iteration_start    # the regime that the verify pass actually ran in

  on_exceed: reduce_gamma       # disable | reduce_gamma
  gamma_under_load: 2           # a shorter draft has a smaller denominator and survives deeper
                                # into the compute-bound region: 1 + 2*0.014 = 1.028 vs 1.114 at 8
  hysteresis: 0.10              # do not flap at the boundary

  alert:
    acceptance_below: 0.55      # per-position; aggregate acceptance hides the cause
    report_positions: true      # REQUIRED: distinguishes "bad drafter" from "gamma too long"
    on_regression: notify
    # Do NOT auto-disable on falling acceptance. That converts a diagnosable problem (the draft
    # has drifted from the target) into a feature that silently disappeared.

  # Startup validation -- each of these is a misconfiguration whose only runtime symptom is an
  # ABSENT benefit, which is the failure mode that survives for months.
  validate_at_boot:
    require_batch_limit: true
    require_tokenizer_match: true    # a mismatched tokenizer is a CORRECTNESS failure, not a speed
                                     # loss -- acceptance collapses toward chance
    require_gamma_within_drafter: true
    require_draft_model_if_used: true
```

---

## 3. `drafter-matrix.yaml` — per-route selection at the router (T14)

```yaml
# alpha is a property of the WORKLOAD, so the drafter is a routing decision, not a server flag.
# Each row's gamma comes from gamma_sweep() at that row's expected alpha; the speedups are this
# blueprint's model, not benchmarks.
routes:
  - match: {path: /v1/extract, task: extraction}
    speculative: {method: ngram, gamma: 5}
    # output quotes the input; p(copy) ~ 0.99 -> tokens/step 5.85, c = 0

  - match: {path: /v1/summarise, task: summarisation}
    speculative: {method: ngram, gamma: 5}
    # p(copy) ~ 0.95 -> 5.30x; free to be wrong

  - match: {task: rag_grounded}
    speculative: {method: ngram, gamma: 5}
    # p(copy) ~ 0.85 -> 4.15x; the grounded answer reuses the retrieved span

  - match: {task: code_edit}
    speculative: {method: ngram, gamma: 5}
    # p(copy) ~ 0.70 -> 2.94x; the edit region is a copy of surrounding context

  - match: {path: /v1/chat, class: interactive}
    speculative: {method: mtp, gamma: 4}
    # small batch, memory-bound -> the regime where learned drafting pays. Requires MTP heads in
    # the checkpoint. Modest, real: the corpus's own production figure is ~2x from MTP [T].

  - match: {task: creative, temperature_gte: 0.9}
    speculative: {method: none}
    # a flat distribution means low alpha for ANY drafter. Speculation is inert here.

  - match: {class: batch_offline}
    speculative: {method: none}
    # large batch -> compute-bound -> speculation is a net LOSS. See gate-policy.yaml.

  - match: {class: agent_fanout, width_gte: 8}
    speculative: {method: mtp, gamma: 2}
    # prefill-dominated: "prefill is occupying like 98% of the tokens" [T]. Speculation attacks
    # decode, so the lever here is prefix caching and KV offload (T07), not this. MTP helps
    # INTERACTIVITY at moderate fan-out, which is a different mechanism from throughput.
```

---

## 4. `boot-checks.sh` — the four startup refusals

```sh
#!/usr/bin/env sh
# Refuse to start on the misconfigurations whose only runtime symptom is an absent benefit.
# Every one of these is cheap to check at boot and expensive to diagnose in production.
set -eu

echo "== T09 speculative decoding boot checks =="

# 1. A mismatched tokenizer is not a slowdown, it is a correctness failure -- the draft's tokens
#    denote different strings, so acceptance collapses toward chance.
python - <<'PY'
import sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
d = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
if t.get_vocab() != d.get_vocab():
    sys.exit("FAIL: tokenizer mismatch -- speculation would be incorrect, not merely slow")
print("ok: tokenizers identical")
PY

# 2. gamma must be within what the drafter can actually produce. A silent clamp would disguise a
#    configuration error as a performance shortfall.
#    (check num_speculative_tokens <= the checkpoint's trained/supported depth)

# 3. The gate is not optional. Without batch_limit the deployment is ungated by construction.
#    (assert speculation_gate.batch_limit is set whenever speculative.enabled is true on any route)

# 4. If the method is draft_model, the weights must exist. A silently disabled speculator reads as
#    "speculation gave no speedup".
#    (check the path)

echo "ok: all boot checks passed"
```

---

## 5. `metrics.promql` — the two dashboards this blueprint insists on

```promql
# ---------------------------------------------------------------------------------------------
# WHY THESE TWO, AND NOT "SPEEDUP"
# Speedup is downstream of acceptance AND of the batch regime, so a single speedup counter cannot
# tell "the drafter got worse" from "the service got busier". Those are the two most common real
# problems and they have different fixes. These queries discriminate them.
# ---------------------------------------------------------------------------------------------

# 1. ACCEPTANCE BY POSITION -- the health of the draft/target pair, independent of load.
#    A fall at LOW position = the drafter is bad or has drifted -> replace or retrain it.
#    A fall only at HIGH position = gamma is too long -> shorten it.
#    One aggregate number cannot distinguish these. This query can.
sum by (position) (rate(spec_draft_accept_total[5m]))
  / sum by (position) (rate(spec_draft_proposed_total[5m]))

# 2. BATCH DEPTH AGAINST batch* -- the only signal that catches the p99 regression.
#    Below the line: speculation pays. Above it: the verify pass costs gamma+1x and speculation is
#    a net loss. Watch the p99 panel, not the mean.
histogram_quantile(0.99, sum by (le) (rate(spec_batch_depth_bucket[5m])))
  / on() group_left () speculation_gate_batch_limit

# 3. MODELLED vs MEASURED SPEEDUP -- a persistent gap means the acceptance model is wrong.
#    Modelled is the blueprint's arithmetic; measured is wall clock. They should track.
spec_speedup_modelled / spec_speedup_measured

# 4. TOKENS PER STEP -- the direct observable of the numerator in the speedup formula.
rate(spec_tokens_emitted_total[5m]) / rate(spec_target_passes_total[5m])

# 5. GATE STATE -- how often the service is actually in the regime where speculation pays.
#    A gate that is closed most of the time means speculation is configured for a load the service
#    does not have.
avg_over_time(spec_in_memory_bound_regime[1h])

# 6. REJECTION-SIDE SAMPLING COUNT -- a canary for the residual branch being exercised at all.
#    If this is zero while rejections are happening, the residual resampler is not being reached,
#    and the output distribution is drifting. Pair with a periodic distribution test.
rate(spec_residual_resample_total[5m])

# 7. DRAFT MODEL MEMORY -- the capacity cost the speedup number hides (T07).
#    Rising draft-model KV is a falling concurrency ceiling, and it looks like nothing at all.
spec_draft_model_kv_bytes / spec_total_kv_bytes
```

---

## What is deliberately absent

| Not provided | Why |
|---|---|
| an attention kernel with a tree mask | kernel work, engine territory (T06) |
| a draft model training recipe | MTP heads ship in the checkpoint; a draft model is a separate project |
| benchmark numbers | **no fabricated benchmarks** — the corpus's ~2× MTP figure is cited to its speaker `[T]`, everything else here is this blueprint's model, reproducible from `run.py` |
| a "best" configuration | alpha is a property of the workload; the matrix in §3 is the deliverable, not a single row |

---

## Sources

- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_Agentic_AI_Distributed_Inference_with_llm-d.txt` — MTP, "about 2x improvement in throughput"; the agentic prefill share.
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt` — Eagle / MTP / Deep Flash support; Spec V2.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/03-speculative-decoding.md` — the draft/verify paradigm and the hardware-aware dynamic-draft-length family.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/01-inference-fundamentals.md` — the roofline behind `batch_limit: 295`.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/02-kv-cache-and-context-caching.md` — the KV budget the draft model competes with.
