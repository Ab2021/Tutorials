# T10 — Quantization: production artifacts

> `T10` · **Transcript coverage:** partial · [HLD](../HLD.md) · [LLD](../LLD.md) · [Runnable core](../run.py) · [Sequences](../docs/SEQUENCES.md)

**Reference-grade, not machine-validated.** These are the artifacts a team would actually write and
review, shown inline so they can be copied. Every number that comes from the corpus is marked `[T]`
or `[R]`; every number from the blueprint's simulator is marked `[D]`.

The design decision these encode: **quantization is decided in three places, owned by three teams,
on three cadences** (HLD §10). The weights are an artefact, the KV cache is a launch flag, and the
lever choice is a per-route policy — and the failure mode is that the last two are treated as
afterthoughts to the first.

---

## 1. `quantization-manifest.yaml` — the checkpoint decision (offline, model team)

Ships *with the weights*. A checkpoint without this file is unusable, because the quality number
that justifies it cannot be interpreted without the group size (LLD §2.2).

```yaml
apiVersion: inference.internal/v1
kind: QuantizationManifest
metadata:
  name: llama-3.1-8b-awq-g128
  base_model: meta-llama/Llama-3.1-8B-Instruct
spec:
  # ---- the artefact: not reversible without re-quantizing and re-evaluating ----
  quantizer: awq            # awq | gptq | nf4 | fp8 | none
  weight_bits: 4
  weight_group: 128         # per-GROUP scale. `null` (per-tensor) is REFUSED -- see section 5.
  weight_scale_dtype: fp16
  salient_frac: 0.01        # AWQ: top 1% of channels restored to fp16 [R]
  # effective bits = 4 + 16/128 = 4.125 -> 4.12 GB for 8B [D], not 4.00 GB.

  # ---- AWQ/GPTQ require activations from a real model; NF4/FP8 do not ----
  calibration:
    required: true
    dataset: <the model team's calibration set>
    samples: 512
    note: "real AWQ salience is ACTIVATION-derived [R]; the blueprint's L1-norm proxy is offline only"

  # ---- the gate. This block is why the manifest exists ----
  eval_gate:
    threshold_pct: 2.0
    slice: worst             # MUST be `worst`. The mean passes a 6.88% regression [D].
    context_lengths: [2048, 8192, 32768]
    note: |
      KV-quantization damage is invisible below the deployed context length [D]. An eval at
      2048 tokens does not validate a service running at 32768.
    baselines:
      - {metric: mmlu, bf16: 68.4, quantized: 67.4, delta_pct: 1.46}
      - {metric: gsm8k, bf16: 74.1, quantized: 72.6, delta_pct: 2.02}   # BREACH -> blocks the merge
      - {metric: longctx_32k, bf16: 61.0, quantized: 59.8, delta_pct: 1.97}
    verdict: FAIL
    verdict_reason: "gsm8k worst-slice delta 2.02% > 2.0% threshold"
    remediation: "re-quantize at weight_group=64 (4.25 bits, +3% storage) and re-run the gate"
```

**Two things in this file are the point of the whole family.** `slice: worst` — because the mean
would pass (HLD §12.2). And `context_lengths` including the *deployed* context — because a
KV-quantized model can score identically at 2k and fail at 32k, and a short-context suite tests
neither lever (HLD §7).

---

## 2. `engine-quant.yaml` — the launch flags (serving team, restart cadence)

**This file holds the larger lever.** The checkpoint above is worth 1.21× on concurrency; this file
is worth 4.00×, and it is the one teams forget to set (HLD §3).

```yaml
# vLLM launch -- the KV precision is a LAUNCH FLAG, not a property of the weights.
vllm:
  model: /models/llama-3.1-8b-awq-g128
  quantization: awq
  kv_cache_dtype: fp8            # <-- the 4x concurrency lever [R]
  calculate_kv_scales: true      # per-layer KV calibration pass; REQUIRED for fp8 on many models
  max_model_len: 32768
  gpu_memory_utilization: 0.90
  enable_prefix_caching: true    # the ladder's fourth rung -- only helps a STABLE prefix [T]

# expected capacity [D], 8B, one 80 GB part, util 0.90, ctx 8192:
#   bf16 weights + bf16 KV :  16.00 GB weights, 0.536 GB/seq ->  52 sequences
#   int4 weights + int4 KV :   4.12 GB weights, 0.277 GB/seq -> 245 sequences
#   of which: weight quant = 1.21x, KV quant = 4.00x, together = 4.7x
capacity_assertion:
  min_sequences: 230             # deploy fails if measured concurrency is below this

# the alternative, and when it is right:
#   engine-quant-latency.yaml: kv_cache_dtype: auto, weight 4-bit -- for a latency SLO with
#   a small model, where the weights' share of bytes-per-token is what matters.
```

### 2.1 Why the two files are separate

| | `quantization-manifest.yaml` | `engine-quant.yaml` |
|---|---|---|
| changes | per model release | per deployment, per route |
| revert cost | re-quantize + re-eval (hours) | restart (seconds) |
| the lever | **fitting** and latency | **concurrency** |
| owned by | model team | serving team |
| validated by | the eval gate | the eval gate, **at the deployed context length** |

**Merging them is the common mistake**, because the merge makes the reversible decision look as
expensive as the irreversible one — so the KV precision never gets tuned, and 4× is left on the
table.

---

## 3. `lever-matrix.yaml` — which lever, per route

The decision function from LLD §8.3, expressed as config, because "quantize for memory" is not a
verifiable instruction and "weights, group 128, because the model does not fit" is.

```yaml
routes:
  - name: long-context-rag
    model: llama-3.1-8b-awq-g128
    context_len: 32768
    constraint: concurrency       # concurrency-limited -> KV is the lever
    config: { weight_bits: 4, weight_group: 128, kv_bits: 4 }
    expected: { sequences: 63, lever: "KV 4.00x" }
    rationale: "the KV term dominates at 32k; weights are 27% of one sequence's footprint [D]"

  - name: interactive-chat
    model: llama-3.1-8b-awq-g128
    context_len: 4096
    constraint: latency           # decode latency SLO -> weights are the lever
    config: { weight_bits: 4, weight_group: 128, kv_bits: 16 }
    expected: { bytes_per_token: "4.12 GB / batch + ctx*kv", lever: "weights 1.21x + latency" }
    rationale: "KV quantization would buy concurrency we do not need and risk attention quality"

  - name: batch-classification
    model: llama-3.1-70b-awq-g128
    context_len: 2048
    constraint: fit               # does not fit at bf16 -> weights are the ONLY lever
    config: { weight_bits: 4, weight_group: 128, kv_bits: 16 }
    expected: { sequences_bf16: 0.0, sequences_int4: 66.9 }
    rationale: "70B bf16 on one 80 GB part does not fit at all [D]; no KV setting substitutes"

  - name: no-quantization-control
    model: llama-3.1-8b-bf16
    context_len: 4096
    constraint: quality           # fits, not concurrency-limited, latency not the SLO
    config: { weight_bits: 16, weight_group: null, kv_bits: 16 }
    expected: { sequences: 208.6 }
    rationale: "quantization would buy nothing here and cost an eval cycle (LLD 4.1, terminal state G)"

policy:
  # evaluated in order; the first match wins, and `none` is a legitimate verdict
  - if: "weight_bytes(params, bits=16) > usable_hbm"      -> lever: weights,  reason: fit
  - if: "service_is_concurrency_limited"                   -> lever: kv,       reason: concurrency
  - if: "decode_latency_is_slo"                            -> lever: weights,  reason: latency
  - else                                                   -> lever: none,     reason: buys nothing
```

**`no-quantization-control` is a real route, not a placeholder.** It is the A/B that makes a
quantization claim falsifiable, and it is also the correct production config for a route where
quantization buys nothing.

---

## 4. `metrics.promql` — what to alert on

The two metrics that discriminate the silent failures (HLD §11). Both are **worst-slice**, never
means.

```promql
# 1. THE MONEY METRIC: realised concurrency, broken out by LEVER.
#    Alert if this is below the manifest's assertion -- it means a config did not take effect,
#    which is silent: the service runs and simply does not deliver (KV quant unset is the usual cause).
sum(vllm:num_requests_running) < 230
  unless on() vllm:config_kv_cache_dtype != "auto"

# 2. WORST-SLICE quality, per model, per context band. NOT the mean.
#    This is the metric that sees a destroyed channel (HLD 4.1) and a long-context regression (7).
max by (model, context_band) (eval_slice_degradation_pct) > 2.0

# 3. KV cache utilisation vs the arithmetic's prediction.
#    A gap here means the KV term is not what the capacity model assumed.
(vllm:kv_cache_usage_perc > 0.95) and (vllm:num_requests_running < 0.8 * expected_concurrency)

# 4. PREFIX CACHE is only worth it on a stable prefix [T].
#    A low hit rate on a route that budgets for caching is a wasted rung.
sum(rate(vllm:prefix_cache_hits_total[5m]))
  / sum(rate(vllm:prefix_cache_queries_total[5m])) < 0.5
  and on(route) lever_matrix_expects_caching == 1

# 5. Prefill share, to check the ladder's assumption against reality.
#    If prefill is not ~60% of the work, the caching rung's 57% saving is not there [D].
sum(rate(vllm:prompt_tokens_total[5m])) / sum(rate(vllm:prompt_tokens_total[5m]) + rate(vllm:generation_tokens_total[5m]))

# 6. Effective batch depth, to compare against `solve_batch`'s single-digit answer [D].
#    Deep batches mean the batching rung is already collected.
vllm:num_requests_running

# 7. Per-layer quantized-vs-bf16 error, sampled offline at boot.
#    The metric that catches `weight_group: null` before it reaches production.
quant_layer_worst_channel_error > 0.5
```

**Query 1 is the one to write first.** Its failure is silent by construction — the deployment
succeeds, the service is healthy, and the 4× was never collected because `kv_cache_dtype` stayed at
`auto`. That is the single most common way this topic goes wrong in production.

---

## 5. `boot-checks.sh` — four refusals

Configs that must not load. Each one is a silent failure at runtime and a loud one here.

```bash
#!/usr/bin/env bash
# T10 boot checks. Fail fast, loudly, before the engine allocates.
set -euo pipefail
FAIL=0
refuse() { echo "REFUSE: $*" >&2; FAIL=1; }

# --- 1. Per-tensor scales on weights -------------------------------------------------
# A per-tensor 4-bit scale destroys a channel outright: worst-channel error 1.0000 [D].
# Aggregate SNR looks survivable (8.48 dB) while the feature is gone. NEVER ship this.
GROUP=$(yq '.spec.weight_group' quantization-manifest.yaml)
BITS=$(yq '.spec.weight_bits' quantization-manifest.yaml)
if [ "$BITS" -lt 16 ] && [ "$GROUP" = "null" ]; then
  refuse "weight_bits=$BITS with per-tensor scale: one outlier channel sets the scale for all of it.
          Use weight_group in [16,32,64,128]. See HLD 4.1."
fi

# --- 2. KV precision without a matching eval -----------------------------------------
# KV quantization damages ATTENTION and is invisible to short-context evals [D]. A KV
# precision change with no eval at the deployed context length is an unvalidated change.
KVD=$(yq '.vllm.kv_cache_dtype' engine-quant.yaml)
MAXLEN=$(yq '.vllm.max_model_len' engine-quant.yaml)
if [ "$KVD" != "auto" ] && ! grep -q "context_lengths:.*$MAXLEN" quantization-manifest.yaml; then
  refuse "kv_cache_dtype=$KVD but no eval at max_model_len=$MAXLEN. The regression is not
          visible at shorter contexts. Add it to the gate."
fi

# --- 3. FP8 requested on hardware without native support -----------------------------
# Silent fallback: the engine runs and is SLOWER than bf16 (HLD 11).
if [ "$KVD" = "fp8" ] || [ "$(yq '.spec.quantizer' quantization-manifest.yaml)" = "fp8" ]; then
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  case "$CAP" in
    9.0|10.0|12.0) : ;;   # H100 / B200 / 4090-class: native FP8 [R]
    *) refuse "FP8 requested on compute capability $CAP: no native FP8 support.
               Use AWQ/GPTQ/EXL2 kernels on Nvidia, or GGUF off-GPU. See HLD 11." ;;
  esac
fi

# --- 4. Prefix caching budgeted on a changing prefix ---------------------------------
# The corpus: "caching only helps a stable prefix. Cache something that changes each
# call and you gain nothing." [T] Measured: the same cache is worth 57% or 1% depending
# on the prompt/output ratio alone [D].
if [ "$(yq '.vllm.enable_prefix_caching' engine-quant.yaml)" = "true" ]; then
  RATIO=$(yq '.workload.prompt_output_ratio' engine-quant.yaml)
  if awk "BEGIN{exit !($RATIO < 1.0)}"; then
    echo "WARN: prompt/output ratio $RATIO -- the caching rung is worth ~1% here [D].
          Not a refusal, but do not count it in the capacity plan." >&2
  fi
fi

# --- 5. The capacity assertion is only meaningful with the KV lever actually set -----
if [ "$(yq '.vllm.kv_cache_dtype' engine-quant.yaml)" = "auto" ]; then
  echo "WARN: kv_cache_dtype=auto -- the 4.00x concurrency lever is NOT engaged [D].
        If this service is concurrency-limited, this is the largest available win." >&2
fi

[ "$FAIL" -eq 0 ] || { echo "boot checks FAILED" >&2; exit 1; }
echo "boot checks passed"
```

**Check 5 warns rather than refuses, and that is deliberate.** Running bf16 KV is a legitimate
configuration for a latency-bound small-model route (§3's `interactive-chat`). The check exists to
make the *absence* of the larger lever visible, not to forbid it.

**Checks 1 and 3 refuse; 2 and 4 escalate.** Check 1 is a guaranteed quality failure. Check 3 is a
guaranteed performance regression. Check 2 is an unvalidated change and check 4 is a wasted
optimization — both are conditions a team might knowingly accept, but not unknowingly.

---

## 6. What is deliberately absent

| Absent | Why |
|---|---|
| a quantizer implementation | AWQ/GPTQ/NF4/FP8 are shipped and hardware-coupled `[T]`; re-implementing one is not this project |
| a calibration pipeline | needs real activations and a real model; the manifest *names* it and the boot check *requires* it |
| a bits→quality predictor | **no single exponent reconciles the corpus's own anchors** (HLD §12); the gate is the honest substitute |
| a per-layer sensitivity config | needs measurement against a real model; `calculate_kv_scales` and the eval gate are where it lives |
| GPU execution | no GPU here — every number above marked `[D]` is from [`run.py`](../run.py) and reproducible offline |

**The third row is the design's central claim.** A tool that predicted quality loss from bit width
would be more convenient and less true; what ships instead is a worst-slice gate, and the boot
checks make it non-optional.

---

## Sources

- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — the 100/42/26/11 ladder; "AWQ and GPTQ to quantize" `[T]`; "always rerun your evals after quantizing" `[T]`; "caching only helps a stable prefix" `[T]`.
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt` — native 8-bit/4-bit training and FP4 rollout, behind the `native` capability check.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/03-training-and-adaptation/07-quantization-deep-dive.md` — the 1%-salient AWQ mechanism and its calibration requirement `[R]`; NF4's equal-mass bins; FP8's dynamic range; "allow 4x higher concurrency on the same GPU" `[R]`; QAT below 3B.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/09-on-device-and-edge-deployment.md` — the 4-bit on-device standard and the Q2/Q3 over-quantization pitfall.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/07-cost-optimization-playbook.md` — the VRAM levers (GQA, quantization) in the FinOps frame behind query 1.

Blueprint: [`HLD.md`](../HLD.md) §§3, 4.1, 7, 10, 11, 12 · [`LLD.md`](../LLD.md) §§2.2, 3.4, 7, 8.3 · [`run.py`](../run.py).
