# Cheat Sheet: Quantization

> `T10` · **Transcript coverage:** partial · [Case study](../01-case-studies/T10-quantization.md) · [Blueprint](../03-design-blueprints/T10-quantization/HLD.md) · [Interview bank](../02-interview-questions/T10-quantization.md)

---

## Numbers to know

| Quantity | Value | Source |
|---|---|---|
| **Quantization's contribution to the cost ladder** | **42 → 26** (from continuous-batching baseline) | `[T]` LLMOps cost talk |
| Precision journey | **16-bit → 8-bit → 4-bit** | `[T]` LLMOps cost talk |
| Named methods | **AWQ**, **GPTQ** | `[T]` LLMOps cost talk |
| The mandatory rule | **"always rerun **your** evals after quantizing"** | `[T]` LLMOps cost talk |
| Non-determinism effect | quantization **worsens** temperature-0 non-determinism | `[T]` CMU lecture 2 |
| Security side effect | quantization is **fingerprintable** — inference attacks can infer a provider's scheme | `[T]` CMU lecture 2 |
| Low-precision training | 8-bit and 4-bit, **FP4 native rollout**, quantization-aware training | `[T]` Zhu, SGLang |

---

## The one-table summary

| Method | Bits | Calibration needed | Use when |
|---|---|---|---|
| **FP8 (W8A8)** | 8 | minimal | **default for modern GPUs** — near-lossless, hardware-accelerated |
| **INT8 (W8A8)** | 8 | yes | older hardware; smooth-quant style needed |
| **AWQ** | 4 (weights) | yes, small | weight-only 4-bit; protects salient channels — **good default** |
| **GPTQ** | 4 (weights) | yes | weight-only 4-bit; mature, wide support |
| **GGUF (k-quants)** | 2–8 | yes | **CPU / llama.cpp / edge** — the format, not a method |
| **BitsAndBytes NF4** | 4 | no | QLoRA **fine-tuning**, quick experiments |
| **KV-cache quantization** | 8 | no | memory-bound serving; separate decision from weight quant |
| **QAT (quantization-aware training)** | 4 | trained in | you control training and need 4-bit quality |

**Two independent decisions:** *weight* quantization and *KV-cache* quantization. They have different
accuracy profiles and different failure modes. Do not conflate them `[D]`.

---

## Formulas

**Memory footprint**
```
weights_GB ≈ N_params × bits / 8 / 1e9
```
70B model: fp16 ⇒ `70e9 × 16/8 / 1e9 = 140 GB`. INT4 ⇒ **35 GB**. That is the difference between
*needs two 80 GB GPUs* versus *fits on one* `[D]`.

**Quality risk is not uniform.** Weight quantization error concentrates in **outlier channels**.
AWQ's insight is that protecting ~1% of salient channels preserves most of the quality — which is
why 4-bit AWQ/GPTQ often loses very little while naive round-to-nearest at the same bit width
collapses `[R]`.

**KV cache quantization is the bigger win for long context**, because KV grows linearly with context
while weights are constant:
```
KV_GB = 2 × n_layers × n_kv_heads × head_dim × seq_len × concurrency × bits/8
```
Halving KV bits doubles your concurrency at fixed HBM — often a larger practical win than weight
quantization `[D]`.

---

## Configuration

```bash
# Weight quantization, 4-bit AWQ (the common production choice)
vllm serve <model>-AWQ --quantization awq --dtype float16

# FP8 weights + activations on modern GPUs — near-lossless default
vllm serve <model> --quantization fp8

# KV-cache quantization — a SEPARATE decision
vllm serve <model> --kv-cache-dtype fp8

# Calibrate an AWQ/GPTQ checkpoint yourself
python -m awq.entry --model_path <model> --w_bit 4 --q_group_size 128 \
                    --calib_dataset <your-domain-data> --export_path <out>
```

**Calibrate on in-domain data.** Calibration sets shape the quantization; a generic calibration set
can measurably underperform on your domain `[D]`.

---

## Failure signatures

| Symptom | Likely cause | First check |
|---|---|---|
| Format/schema compliance dropped | quantization degraded structured output | rerun the format eval — it is the first thing to break |
| Math/reasoning accuracy dropped | long reasoning chains amplify small errors | rerun reasoning evals; consider 8-bit |
| Output differs run to run more than before | **expected** — quantization worsens non-determinism `[T]` | not a bug |
| Quality fine on English, bad on another language | calibration set unrepresentative | recalibrate in-domain |
| OOM still | KV cache, not weights, is the constraint | quantize KV or raise `--gpu-memory-utilization` |
| Provider fingerprinting concern | quantization scheme is inferable from outputs `[T]` | a security consideration, not a perf one |
| Accuracy fine at 8-bit, broken at 4-bit | you crossed the quality cliff | stay at 8-bit, or use AWQ/GPTQ with good calibration |
| Eval passes offline, prod quality complaints | your eval set is too easy | the tail breaks first |

---

## Gotchas

- **Quantization is not free quality-wise — it is a trade you must measure.** The corpus states the
  rule flatly: **"always rerun **your** evals after quantizing"** `[T]`. This is the single most-skipped step.
- **The failures are in the tail, not the mean.** A 1% average accuracy loss can be a 20% loss on
  the specific task you care about. Always inspect worst cases `[D]`.
- **Reasoning and structured output break before prose does.** Quantization noise compounds over long
  chains, and logit margins that a grammar mask depends on can shrink `[T]`/`[D]`.
- **Quantization makes non-determinism worse** and creates a **fingerprint** `[T]`. If you care about
  reproducibility or about not disclosing your stack, this matters.
- **8-bit is close to free; 4-bit is not.** If you have any accuracy headroom concern at all, take
  FP8 and spend your effort on KV cache or batching instead `[D]`.
- **KV-cache quantization is a separate decision with a separate eval.** Weight quant ≠ KV quant.
- **QAT beats post-training quantization at 4-bit**, but only if you can train. If you cannot,
  AWQ/GPTQ with in-domain calibration is the practical ceiling `[T]`.
- **Low-precision *rollout* is a distinct thing from low-precision serving.** Zhu's talk describes
  FP4 native rollout during RL training, which is a training-efficiency technique, not a deployment
  decision `[T]`.

---

## When to use what

| Situation | Do |
|---|---|
| Modern GPU, want memory savings with minimal risk | **FP8** |
| One GPU instead of two, quality tolerable | **AWQ** or GPTQ 4-bit, in-domain calibration |
| CPU / edge / llama.cpp | **GGUF** k-quants |
| Fine-tuning on one GPU | **QLoRA / NF4** |
| Long context, memory-bound | **KV-cache fp8** — often the bigger win |
| Reasoning-critical, no accuracy headroom | **stay at 16-bit** and optimise elsewhere |
| You control training and need 4-bit quality | **QAT** |

---

## Sources

- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt`
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt`
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt`
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt`
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/03-training-and-adaptation/07-quantization-deep-dive.md` `[R]`
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` `[R]`
