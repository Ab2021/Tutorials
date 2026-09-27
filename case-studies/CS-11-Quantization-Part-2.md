# CS-11 — Quantization II: Advanced Methods & Production Practice

> **Module ID:** CS-11
> **Track:** Efficiency & Compression
> **Source video:** `LLM_Fine-Tuning_13_LLM_Quantization_Explained_PART_2_PTQ_QAT_GPTQ_AWQ_GGUF_GGML` (Video 13)
> **Companion notebooks:** `LLM Fine-Tuning-12-13-LLM-Quantization/LLM-Quantization-Part-2/` → `QAT_in_LLM.ipynb`, `GPTQ_UPDATE.ipynb`, `LLM_Quantization_AWQ.ipynb`, `gguf_ggml_practical.ipynb`
> **Prerequisite:** **CS-10 (Quantization I: Fundamentals)** — read it first. CS-10 defines the affine quantization math (`scale = (max−min)/(q_max−q_min)`, `zero_point ≈ round(−min/scale)`), the PTQ-vs-QAT distinction, the first-pass introductions to GPTQ / AWQ / GGUF / GGML, the outlier problem, and the base VRAM tables. **None of that is repeated here.**
> **This module is the advanced, practical, deployment half.** It assumes you can already compute a scale and a zero-point by hand. What it adds: the actual code paths (`prepare` → `convert`), what the gradients do at `round()`, what the post-GPTQ/AWQ research frontier looks like, why the KV cache is the real long-context cost, the precision-notation/hardware lattice (W8A8 … W4A4), the serving-stack flags that decide your throughput, how to *honestly measure* quantization damage, the QLoRA/NF4 memory algebra, and the 20 gotchas that silently ship broken models.

---

## §0 Executive Summary

**The one-sentence version:** by 2026, quantization has stopped being "shrink the model" and become **a precision-selection problem** — you are choosing, per tensor family (weights, activations, KV cache) and per hardware generation, the lowest precision that clears your accuracy gate at your latency SLO, and the hard part is no longer the algorithm but the *measurement* that tells you whether you cleared it.

Ten load-bearing claims this module will defend:

1. **W4A16 is the default deployment point, not W4A4.** Almost every 4-bit model you will serve in production keeps activations in FP16 and dequantizes weights into FP16 tensor cores. The kernel does not use INT4 arithmetic — that is why **Marlin** exists and why "4-bit" throughput numbers are so kernel-sensitive.
2. **The KV cache overtakes the weights as the memory bottleneck at long context.** Llama-3.1-8B stores **128 KB per token** of KV in FP16 (`2 × 32 layers × 8 kv_heads × 128 head_dim × 2 bytes`). At 32k context × batch 8 that is **32 GB of cache** — four times the INT4 weight footprint. FP8 KV halves it and is the highest-leverage single flag in your serving config.
3. **Perplexity is a weak instrument and will let a broken model through.** A model can hold perplexity flat while losing instruction-following, JSON formatting, and multi-step math. Gate on **mean KL divergence from FP16 + top-1 agreement + p95 KL + IFEval** on held-out *domain* data, not on a WikiText perplexity delta.
4. **QAT is now an off-the-shelf artifact, not a project.** Google shipped QAT-trained Gemma 3 at 1B/4B/12B/27B (`google/gemma-3-*-it-qat-q4_0-gguf`), and EfficientQAT does full 2-bit LLaMA-2-70B on one GPU. In 2026 you *download* QAT rather than run it — unless you are the model owner.
5. **`torch.ao.quantization`'s eager QAT path is legacy.** `torchao` (`torchao.quantization.quantize_`, `QATConfig`) is the maintained API. Learn the eager path because interviews and older codebases use it; ship the `torchao` path.
6. **Rotation kills outliers globally; SmoothQuant moves them.** Hadamard/QuaRot/SpinQuant rotate the basis so that `E[‖HDx‖∞²] = ‖x‖₂²/n` — a dominant channel's outlier gets spread over *n* coordinates, shrinking the per-tensor step by up to `√n`. SmoothQuant instead *migrates* the outlier from activations into weights. These are the two ideas behind nearly every 2025–26 PTQ paper.
7. **LLM.int8() is a mixed-precision decomposition, not a rounding scheme.** It runs the 99.9% of dimensions through INT8 and extracts the ~0.1% that carry magnitude > 6.0 into a separate FP16 matmul. The outliers are *systematic and emergent* — absent below 6.7B parameters.
8. **NF4's quantile grid is information-theoretically motivated, and double quantization is free money.** 16 equal-probability bins of `N(0,1)` is the Lloyd-Max-optimal scalar quantizer for normally distributed weights; double quantization buys **0.373 bits/param** back, which is exactly how QLoRA trains a 7B in ~6 GB.
9. **The order of operations matters more than the algorithm.** `bf16 base → LoRA → merge → quantize` is correct. Quantizing first and fine-tuning after is the single most common silent regression in the field.
10. **Quantization is not reproducible by default.** Different kernel versions, different GPU architectures (`sm_80` vs `sm_89` vs `sm_90`), and different library versions produce different numerics from the same checkpoint. Pin, hash, and re-run the eval gate on every rebuild.

> **Beyond the video:** the source video is a *conceptual* tour. It names the algorithms but never runs the QAT conversion end-to-end on a real LLM, never quantizes a KV cache, never shows a single serving-stack flag, and never measures whether the quantized model is actually worse. That gap is this module's entire reason to exist.

---

## §1 The Problem This Solves

### 1.1 Where CS-10 stops and this module starts

| Question | Answered in | Answered here |
|---|---|---|
| What is `scale`, `zero_point`, `q = round(x/s + z)`? | CS-10 §2 | assumed |
| What is the difference between PTQ and QAT? | CS-10 §3 | §4.6–4.7 — the *code* and *gradient* reality |
| What is GPTQ, at a conceptual level? | CS-10 §5 | §4.2 — the objective, the Hessian, the worked arithmetic |
| What is AWQ? | CS-10 §5 | §4.3 — and the correction that it is **W4A16, post-training** |
| What is GGUF vs GGML? | CS-10 §6 | §6.6 — the actual CLI, `imatrix`, and chat-template traps |
| VRAM for a 7B at INT4 | CS-10 §8 | §11 — plus KV cache, activations, CUDA context, fragmentation |
| "Outliers are a problem" | CS-10 §7 | §4.3–4.5 — five competing *solutions*, with math |
| — | — | §4.9 **KV cache quantization** |
| — | — | §4.10 **W8A8/W4A16/W4A8/W4A4 + GPU generations** |
| — | — | §4.11 **QLoRA/NF4 memory algebra** |
| — | — | §7–§10 **serving flags, decision tree, failure modes** |
| — | — | §12 **honest evaluation protocol** |

### 1.2 The four production questions

Every quantization decision you will make in a job reduces to these four, in this order. Answer them in the wrong order and you will re-do the work.

1. **What is the bottleneck?** Decode-time memory bandwidth (weights), long-context KV residency, prefill compute (activations), or train-time optimizer state? Each has a *different* correct answer, and quantizing weights does nothing for three of the four.
2. **What does the hardware natively execute?** A 4-bit weight-only kernel on an H100 runs FP16 tensor cores with an on-the-fly dequant. An FP8 kernel on the same H100 runs the FP8 tensor core. The first saves memory; the second saves *time*. Confusing these two is the most common senior-level mistake.
3. **What is my accuracy tolerance, and how will I know I violated it?** "It looked fine" is not a gate. §12 gives a protocol that takes an afternoon and catches the failures that perplexity misses.
4. **Do I need to train on top of it?** If yes, the answer is almost always QLoRA (NF4 + LoRA + paged optimizers), not QAT.

---

## §2 First-Principles Mental Model

### 2.1 The three-axis decomposition

Forget the algorithm names for a moment. Every quantization method is a point in a three-axis space. If you can place a new paper on these axes you already understand it.

| Axis | Options | What it controls |
|---|---|---|
| **What gets quantized** | Weights only (W4A16) / Weights+activations (W8A8, W4A4) / KV cache / gradients (optimizer states) | Which memory or compute cost you actually attack |
| **When** | Post-training (PTQ, calibration only, minutes) / During training (QAT, days) / Quantization-aware fine-tuning (QLoRA) | Whether the weights get a chance to adapt |
| **Granularity** | Per-tensor / per-channel (per-row) / per-group (g=128, g=64, g=16) / per-token / mixed-precision escape hatch | The step size `Δ` and therefore the error floor |

Everything else — Hessian, rotation, k-means codebooks, sparse outliers, straight-through estimators — is **machinery for making a bad point on those axes survivable.**

### 2.2 The error budget, stated once

For a tensor `x` quantized to `b` bits with per-group granularity `g`, the round-to-nearest error on each element is bounded by `Δ/2` where

```
Δ = (max(x) − min(x)) / (2^b − 1)
```

Two consequences that determine the entire field:

- **Halving the range quarters the MSE.** Going from 4→8 bits shrinks `Δ` by 16×, so MSE shrinks by 256×. This is why 8-bit is nearly free and 4-bit needs help. (CS-10 §2 derives this; we use it, we do not re-derive it.)
- **The max is set by the worst outlier.** One channel with magnitude 30 in a tensor otherwise bounded by 0.5 forces `Δ` to be 60× larger than it needed to be, which costs you roughly `log2(60) ≈ 5.9 bits` of effective precision on every well-behaved element. **Every advanced method in §4.3–4.5 is a way of shrinking that max.**

> **The mental model to keep:** quantization hurts because of the *range*, and the range is set by outliers. So all progress is progress at shrinking the range — by clipping it (AWQ), moving it (SmoothQuant), rotating it away (QuaRot), isolating it (LLM.int8()), storing it sparsely (SqueezeLLM, SpQR), or making the tensor inherently low-range (BitNet, trained-from-scratch).

### 2.3 Quantization is a rate-distortion problem

Shannon's rate-distortion framing is the cleanest way to see why 4 bits is a wall and 2 bits is research:

- Weights of a trained transformer are approximately Gaussian. A Gaussian source quantized with an optimal (Lloyd-Max) scalar quantizer has distortion falling as roughly `2^(−2b)`, i.e. **each bit buys ~6 dB** of SNR.
- 4-bit W4A16 relative to FP16 (≈11 effective mantissa bits) is a ~7-bit cut → ~42 dB of quantization SNR loss before any adaptation. That sounds catastrophic but transformers are heavily redundant, so in practice the *task-relevant* distortion is much lower — which is exactly why you must **measure the task-relevant distortion** (§12) and not infer it from SNR.
- 2-bit and below is where scalar quantization stops being enough and you need either **vector quantization / codebooks** (SqueezeLLM's k-means), **structural constraints** (BitNet's ternary), or **training** (EfficientQAT's 2-bit QAT).

> **Beyond the video:** the video never frames quantization as rate-distortion, and this framing is what lets you predict *which* methods work at *which* bit-width without reading every paper. Scalar methods dominate at 8 and 4 bits. Below 4 bits, codebook/vector methods and training-based methods take over.

### 2.4 Why the KV cache changes the whole problem

Prefill is compute-bound (`O(seq²)` attention FLOPs); decode is memory-bound (one token's worth of weights read per step). Weight-only quantization attacks the decode bottleneck. But there is a second memory resident that grows with **context length × batch**, not with parameter count — the KV cache — and at long context it dominates. Full arithmetic in §4.9.

The strategic consequence: **the same model at 2k context and at 128k context has two different optimal quantization configurations.** At 2k, quantize the weights. At 128k, you must quantize the KV cache too, or you will not fit the batch.

---

## §3 Glossary

Terms the video uses loosely that you must not. Where a term is defined in CS-10, the row says so.

| Term | Precise meaning | Where it bites |
|---|---|---|
| **PTQ** (post-training quantization) | Quantization technique: quantize a *finished* FP model using only calibration data. Minutes. | Quality ceiling; no adaptation. |
| **QAT** (quantization-aware training) | Quantization technique: simulate quantization in the forward pass and train the weights to compensate. Days. | Cost; LLM-scale tooling immaturity. |
| **GPTQ** | A specific **PTQ algorithm** (Hessian-based, layer-wise, error-compensating). Not a format, not a framework. | "GPTQ model" = a checkpoint whose weights came out of a GPTQ run. |
| **AWQ** | A specific **PTQ algorithm** (activation-aware scaled clipping). Deploys as **W4A16**. | Not a training method. Not activation-quantizing at inference. |
| **GGUF** | A **model file format** (single file, header + metadata + tensor blocks, k-quants with nested scales). | Not a quantizer. It *stores* the result. |
| **GGML** | The original C **tensor library / inference engine** that GGUF replaced the format of; superseded by `llama.cpp`. | The video's own notebook says so — see Appendix A. |
| **`llama.cpp`** | The C/C++ inference engine; `convert_hf_to_gguf.py` + `llama-quantize` + `llama-cli`. | The thing you actually run. |
| **imatrix** | Importance matrix. Calibration-derived per-tensor importance weights that steer GGUF quantization toward the weights that matter. | The single biggest quality lever in llama.cpp quantizing. |
| **Marlin** | A CUDA kernel for **W4A16** GPTQ that overlaps async dequantization with FP16 tensor-core math. | Why "4-bit" is fast on Ampere+; needs `group_size ∈ {128, −1}` for peak. |
| **ExLlamaV2 / EXL2** | Alternative GPTQ-family kernels and a variable-bits-per-weight format. | Different speed/quality point; separate runtime. |
| **STE** (straight-through estimator) | Substituting `∂round(x)/∂x = 1` in the backward pass because the true derivative is 0 a.e. | Without it, QAT gradients vanish at `round`. |
| **Fake quantization** | Quantize→dequantize in FP, so the forward pass sees quantization noise and the backward pass sees a differentiable graph. | The core mechanism of QAT. |
| **Observer** | A module that watches tensors during calibration and records statistics used to compute `scale`/`zero_point`. | MinMax / MovingAverageMinMax / Histogram — §4.7. |
| **QConfig** | PyTorch object pairing an activation spec with a weight spec. | The single knob you set before `prepare_qat`. |
| **Per-channel / per-row** | One scale per output channel (row of `W`). | Standard for weights; kills the "one fat row ruins it" failure. |
| **Per-group** | One scale per `g` contiguous weights (`g ∈ {16,32,64,128}`). | Buys accuracy at ~`2/g` bytes/weight overhead. |
| **Per-token** | One scale per token's activation vector. | Standard for activations in W8A8 and KV quantization. |
| **SmoothQuant** | PTQ method migrating activation outliers into weights via a per-channel scale `s_j`. | Enables true W8A8 with FP16-level quality. |
| **OmniQuant** | Learnable weight clipping + learnable equivalent transformation, block-wise. | Strong W4A4/W3A16 baseline; needs a small optimization loop. |
| **QuaRot / SpinQuant** | Rotation-based outlier removal using Hadamard / learned orthogonal matrices. | Global outlier elimination; needs kernel support or weight folding. |
| **LLM.int8()** | Dettmers' mixed-precision decomposition: INT8 matmul + FP16 extraction of >6.0-magnitude dimensions. | Why 8-bit "just works"; the 0.1% rule. |
| **SqueezeLLM** | Dense-and-sparse: FP16 sparse outliers + non-uniform k-means codebook for the dense part. | Strong 3–4 bit; codebook lookup cost. |
| **SpQR** | Sparse-Quantized Representation: group-16, bilevel (double) scales, ~1% FP16 outliers. | Near-lossless 3.5-bit territory. |
| **BitNet b1.58** | Ternary `{−1, 0, +1}` weights (log₂3 ≈ 1.58 bits) via absmean scaling; **trained from scratch**. | Not a PTQ method. Cannot be applied post hoc. |
| **NF4** | 4-bit NormalFloat: 16 quantile levels of `N(0,1)`, information-theoretically optimal for Gaussian data. | The Q in QLoRA. |
| **Double quantization** | Quantizing the *quantization constants* (absmax blocks) themselves. Saves **0.373 bits/param**. | Turns 4.5 bpw into ~4.13 bpw. |
| **Paged optimizer** | Optimizer states in NVIDIA unified memory, paged to CPU on pressure. | Lets QLoRA absorb gradient spikes without OOM. |
| **W8A8 / W4A16 / W4A8 / W4A4** | Weight bits / activation bits. | The notation that maps to hardware support — §4.10. |
| **FP8 e4m3 / e5m2** | 8-bit float: 4 exponent/3 mantissa, or 5 exponent/2 mantissa. `e4m3` range ±448. | `e4m3` for weights/activations/KV; `e5m2` for gradients. |
| **NVFP4 / MXFP4** | Blackwell 4-bit float formats: block-16 with E4M3 scales, vs block-32 with E8M0 scales. | Generational; NVFP4 is NVIDIA-proprietary, MXFP4 is the OCP standard. |
| **PagedAttention** | vLLM's block-based KV cache (block_size 16, block table, copy-on-write sharing). | Forces scale factors to live per block — §4.9. |
| **KL divergence gate** | Mean and p95 KL(FP16 ‖ quantized) over held-out tokens. | The honest accuracy gate — §12. |
| **Calibration set** | The small corpus (128–1024 samples) used by PTQ methods to observe activations. | Leakage and overfitting risk — §10. |

---

## §4 Deep Dive

### 4.1 The layer-wise PTQ pipeline (why every method looks the same)

Every weight-only PTQ algorithm — GPTQ, AWQ, OmniQuant, SpQR, and the k-quants in GGUF — executes the same five steps. Learn the skeleton and the differences become visible.

```
for each Linear layer L in the model (sequential or parallel):
  1. HOOK     : run calibration samples through L, capture input activations X
                (shape: [n_samples × seq_len, d_in])
  2. STATISTIC: compute a per-channel importance from X
                GPTQ      → H = XᵀX  (Hessian / second-order)
                AWQ       → s_j = mean|X_j|  (first-order activation magnitude)
                GGUF K-quant → imatrix importance = Σ X_j²  (per-tensor)
  3. TRANSFORM: optionally rescale/rotate the problem so the hard channels are easier
                AWQ      → W ← W·diag(s), X ← X·diag(s)^-1   (fold back at the end)
                SmoothQuant → same shape, derived differently
  4. QUANTIZE : round W (and optionally X) to b bits at the chosen granularity,
                using the statistic from step 2 to bias or compensate the rounding
                GPTQ      → sequential error compensation across columns
                AWQ/RtN    → scale-aware round-to-nearest
  5. RECORD   : store qweight, scales, zero-points (+ group metadata) in a format
                GPTQ→ .safetensors w/ quantization_config, GGUF → block-quantized file
```

**Three consequences of this skeleton:**

1. **Calibration data quality bounds everything.** Step 1 is the only place real information enters. A PTQ run calibrated on generic English web text and deployed on code will underperform its benchmark — see §10.18 (calibration-set leakage) and §12.4.
2. **Layer-wise decomposition is an approximation.** The true objective is end-to-end output error. Every method minimizes per-layer `‖WX − ŴX‖²` and *hopes* the errors do not compound. They mostly do not, because transformers are robust and residual streams absorb error — but this is why methods that optimize globally (rotation, QAT) can beat methods that optimize layer-wise.
3. **`lm_head` and embeddings are almost always excluded.** They are small in FLOPs but disproportionately sensitive (the LM head directly determines the logit scale; embeddings are lookup tables with no compute to save). GPTQModel defaults `lm_head=False`; the QAT notebook explicitly sets `module.qconfig = None` on `nn.Embedding`.

### 4.2 GPTQ, the actual math

CS-10 introduced GPTQ conceptually. Here is what the algorithm computes, with the video's own worked example reproduced correctly.

**Objective.** For a layer with weights `W` (`d_out × d_in`) and calibration inputs `X` (`n × d_in`), minimize the *layer output* error, not the weight error:

```
argmin_Ŵ  ‖W X − Ŵ X‖²_F
```

This is the key insight of the 2022 ETH Zurich paper. Minimizing `‖W − Ŵ‖²` treats every weight as equally important; minimizing `‖(W − Ŵ)X‖²` weights each weight's error by how much the layer actually *uses* it.

**The Hessian.** Expanding gives a quadratic form in the rows of `W`, with curvature

```
H = Xᵀ X        (d_in × d_in)
```

Equivalently `H = 2XXᵀ` in the paper's transposed convention — the factor of 2 is absorbed into the damping λ and does not change the solution. `H` is the same for every row, so it is computed **once per layer**. This is what makes GPTQ affordable: one `d_in × d_in` matrix (e.g. 4096² = 16.8M floats = 67 MB in FP32) instead of anything per-weight.

> **Correction:** the instructor narrates the Hessian as `XᵀX` in one place and, in the linear-layer example, builds the matrix from `X` directly. Both are the same object up to transpose convention and a scalar factor. The examinable fact is: **the Hessian is the second-moment matrix of the calibration activations, and its diagonal `H_ii` measures how much output energy flows through input channel `i`.**

**Sensitivity per weight.** For a single weight `w_ij` quantized to `ŵ_ij`, the increase in layer output error is

```
Δ_err ≈ (w_ij − ŵ_ij)² · H_jj
```

**The video's worked example, reconstructed.** The instructor computes `XᵀX` and narrates "35, 44, 44, 56". The only `3 × 2` input matrix whose second-moment matrix is exactly that is

```
X = [[1, 2],
     [3, 4],
     [5, 6]]      # 3 tokens × 2 features

XᵀX = [[1²+3²+5², 1·2+3·4+5·6],   = [[35, 44],
       [1·2+3·4+5·6, 2²+4²+6²]]     [44, 56]]
```

Check: `1+9+25 = 35`; `4+16+36 = 56`; `2+12+30 = 44`. Exact match.

He then takes `W = [0.5, −1.0]`, giving `y = XW = [−1.5, −2.5, −3.5]`. Quantizing the second weight `−1.0 → −0.5`:

```
(w − ŵ)²  = (−1.0 − (−0.5))² = (−0.5)² = 0.25
H_22      = 56
Δ_err     = 0.25 × 56 = 14
```

and he says *"we are getting 14."* **The arithmetic is right; his verbal narration "multiply 0.5 with 56" is a misspeak — it is the *squared* error 0.25, not 0.5.** Note also that the second column is the *more* sensitive one (`H_22 = 56 > H_11 = 35`), so GPTQ would quantize the first weight more aggressively — or, in the sequential formulation, would correct the second weight's error using the first column's already-quantized values.

**Error compensation (the "G" in GPTQ being useful).** GPTQ processes columns left to right. When column `j` is rounded, the resulting error `(w_j − ŵ_j)` is propagated into the *not-yet-quantized* columns using the Cholesky factor of `H⁻¹`:

```
ŵ_{j+1:}  ←  w_{j+1:}  −  (w_j − ŵ_j) · [H⁻¹]_{j, j+1:} / [H⁻¹]_{jj}
```

This is the step that makes GPTQ meaningfully better than plain round-to-nearest at 4 bits: the later columns *absorb* the earlier columns' rounding error, at the cost of being slightly wrong themselves — and being later, their error is never compensated, so the ordering matters. Two important knobs fall straight out:

- **`desc_act` (activation ordering).** Reorder columns by `H_ii` descending so the most sensitive channels are quantized *last* and thus get the most compensation. Better accuracy, slower, and historically incompatible with some kernels (Marlin supports it but slower).
- **`damp_percent` (default 0.05).** `H ← H + λI` before inverting. Without it, near-singular `H` (highly correlated calibration activations) produces explosive corrections. GPTQModel auto-increments λ by `damp_auto_increment` if a layer fails.

**Cost.** GPTQ quantization of a 7B model on a single A100 takes roughly 20–60 minutes depending on `batch_size` and `desc_act` — it is a *calibration* pass, not a training run.

> **Beyond the video:** the video presents GPTQ as a black box ("it uses the Hessian"). In an interview, the two things that separate "read the blog post" from "read the paper" are (a) *the objective is layer-output error, not weight error*, and (b) *the Hessian is shared across all output rows of the layer, which is why GPTQ is tractable*. Say both.

### 4.3 The outlier-solution families (the 2025–26 frontier)

The video names SmoothQuant and OmniQuant in passing and stops. Here is the map — six families, each with the mechanism that makes it work.

**Family 1 — Clip/migrate the range (SmoothQuant, OmniQuant, SmoothQuant+).**

*SmoothQuant.* The problem: activation channels have wildly different magnitudes; weight channels do not. So *migrate* the difficulty from activations to weights, which are easier to quantize.

```
Given per-channel activation max  a_j = max|X_:,j|     (j over d_in)
      per-channel weight max     w_j = max|W_:,j|

s_j = a_j^α / w_j^(1−α)                     α ∈ [0,1], typically 0.5

Apply:  X ← X · diag(s)^-1        W ← diag(s) · W
```

The product `(W diag(s)) (diag(s)^-1 X) = WX` is unchanged, so the transform is **exactly equivalent in FP** — it changes only the *distribution* of magnitudes across the two operands. With `α = 0.5` both sides end up with comparable dynamic range, which is what INT8 per-tensor quantization needs. **The scale folds into the weights at export**, so there is zero inference overhead — that is why SmoothQuant became the industry default for W8A8.

*SmoothQuant+ (2024).* The per-channel version assumes a single `s_j` for all tokens. But with long context the outlier channels *vary by token*. SmoothQuant+ instead does a **token-wise** scaling derived from the attention-score-like importance of each token, giving W8A8 quality closer to FP16 on long-context inputs. Relevant because W8A8-with-per-tensor-activations was the thing that broke first at 32k+ context.

*OmniQuant.* Makes both the clipping threshold and the equivalent transformation **learnable**, optimized block-wise with a small gradient loop on calibration data. Two components: *LWC* (learnable weight clipping, which shifts the clipping range to trade off outliers) and *LET* (learnable equivalent transformation). Strong results at W4A4 and W3A16 — i.e. it attacks the regime where SmoothQuant alone is not enough.

**Family 2 — Rotate the basis (QuaRot, SpinQuant, Hadamard tricks).** See §4.5 for the math. The idea: instead of moving outliers around, *change the coordinate system so there are no outliers.* Applied to weights, activations, and KV cache simultaneously in QuaRot.

**Family 3 — Mixed-precision decomposition (LLM.int8()).** See §4.4. Keep the outliers, but compute them in FP16 in a separate matmul.

**Family 4 — Sparse plus dense (SqueezeLLM, SpQR).** Store the outliers *explicitly* as a sparse FP16 list plus indices, and quantize the well-behaved remainder aggressively.

*SqueezeLLM.* "Dense-and-sparse decomposition": outliers (by magnitude) are kept in FP16 sparse format; the dense part is quantized with **non-uniform k-means** — a learned codebook of centroids rather than a uniform grid, which handles the non-Gaussian weight distribution far better than affine quantization. Strong at 3 bits where scalar methods collapse.

*SpQR.* Refines the recipe: group size 16, **bilevel quantization** (the group scales themselves are quantized), and only ~1% of weights retained as FP16 outliers. Lands near-lossless at ~3.5 bits effective.

**Family 5 — Make the tensor low-range by construction (BitNet b1.58).** Ternary weights `{−1, 0, +1}` via `W ← RoundClamp(W / (γ + ε))` where `γ = mean|W|` (absmean). `log₂3 ≈ 1.585` bits per weight, and crucially the matmul becomes *additions only* — no multipliers. **But** BitNet must be **trained from scratch**; it is not a post-hoc transformation. The video's bit/byte confusion around 1.58 bits is corrected in §10.

**Family 6 — Train through it (QAT, EfficientQAT, LLM-QAT, DL-QAT).** See §4.6–4.7, §6.1. The only family that lets the weights themselves change.

**Comparison at a glance:**

| Family | Mechanism | Bit-width sweet spot | Cost | Deployability |
|---|---|---|---|---|
| Clip/migrate (SmoothQuant) | Scale outliers into weights | W8A8 | Minutes | Excellent — folded into weights |
| Rotate (QuaRot/SpinQuant) | Hadamard basis change | W4A4, KV4 | Minutes + kernel support | Good, needs kernel or folding |
| Mixed-precision (LLM.int8()) | FP16 escape hatch for outliers | W8A8 | Free | Excellent — bitsandbytes |
| Sparse+dense (SpQR/SqueezeLLM) | FP16 outliers + codebook | 3–4 bit | Minutes | Moderate — custom kernels |
| Low-range by design (BitNet) | Ternary, trained from scratch | 1.58 bit | Full pretrain | New models only |
| Train through (QAT) | Fake-quant + backprop | 2–8 bit | Days | Excellent *if* you own the model |

> **Beyond the video:** the interview-grade summary of this space is *"the outlier problem has four solution shapes — move it, hide it, isolate it, or eliminate it — and the papers differ only in which one they pick and how they pay for it."* Being able to place a 2026 paper (e.g. any rotation paper) on that map in ten seconds is what senior interviews test.

### 4.4 LLM.int8() in full detail

The video mentions bitsandbytes `load_in_8bit=True` and moves on. The mechanism underneath is worth understanding completely, because it is the cleanest demonstration that **outliers are structural, not noise.**

**Step 1 — Per-channel weight / per-token activation quantization.** Vector-wise quantization: each row of `W` gets its own scale, each token's activation vector gets its own scale. This alone handles the "weights are easy" half of the problem.

**Step 2 — The observation that breaks 8-bit.** Even with per-channel and per-token scaling, Dettmers et al. found that a small number of activation *dimensions* have magnitudes 20–100× the rest. Under per-tensor or even per-token scaling these dimensions destroy the scale for everything sharing it. The key empirical finding: **these outlier dimensions are few, systematic (the same feature indices across many tokens and layers), and emergent — they appear only in models with ≥ 6.7B parameters and grow with scale.** Below that size, 8-bit works naively; above it, it does not.

**Step 3 — Mixed-precision decomposition.** Matmul is split by a magnitude threshold `α = 6.0`:

```
C = A · B
  = A[:, S] · B[S, :]   (FP16,  where S = {j : max|A_:,j| > 6.0})
  + A[:, D] · B[D, :]   (INT8,  where D = complement of S)
```

`S` typically contains about **0.1% of dimensions** (by the paper's count, roughly 7 out of ~4096 features per layer have magnitude above 6.0 in a 7B model). Because the outliers are systematic, `S` can be determined **once per layer from calibration** and cached — the per-token scan finds essentially the same indices. The FP16 matmul therefore costs ~0.1% of the FLOPs and ~0.1% of the memory traffic while recovering essentially all of the lost accuracy.

**Why this matters beyond 8-bit:** it is the template for every "keep a tiny FP16 escape hatch" method, including SpQR's 1% sparse outliers and SqueezeLLM's sparse FP16. The insight is that **outlier handling costs O(outlier fraction), not O(model)**, so you can afford to be exact about the small part.

> **Correction / trap:** a very common interview question is *"why does bitsandbytes 8-bit not hurt quality?"* — the wrong answer is "8 bits is enough." The right answer is "8 bits is *not* enough, because of systematic outlier features that emerge above 6.7B; LLM.int8() recovers quality by computing those ~0.1% of dimensions in FP16."

### 4.5 The Hadamard rotation trick (strong-engineer level)

This is the single most-asked "advanced" quantization topic in 2025–26 interviews, because it is the mechanism behind QuaRot and SpinQuant, and it is pure linear algebra.

**The problem it solves.** Outliers are *basis-dependent*. If a hidden dimension carries a persistently large value, it is because the model's learned basis allocates that dimension to a high-variance feature. Rotating the hidden state into a different basis redistributes that energy without changing the function computed.

**The Hadamard matrix.** The normalized Hadamard matrix `H ∈ {±1/√n}^(n×n)` satisfies

```
H Hᵀ = Hᵀ H = I          (orthogonal, so it preserves norms)
```

**The concentration bound.** For a vector `x` with `‖x‖₂`, the rotated vector `y = H x` has components of magnitude around `‖x‖₂/√n`:

```
E[ ‖H x‖∞² ]  =  ‖x‖₂² / n
```

Concretely: suppose a hidden state is dominated by one coordinate, so `‖x‖∞ ≈ ‖x‖₂` (energy concentrated in one dimension). After rotation,

```
before:  ‖x‖∞ ≈ ‖x‖₂
after :  ‖x‖∞ ≈ ‖x‖₂ / √n
```

For `n = 4096` (a 7B/8B hidden size), `√n = 64` — **the per-tensor range shrinks by up to 64×**, and since the quantization step `Δ ∝ range / 2^b`, the MSE falls by up to `64² = 4096×`. That is the entire value proposition, in one line of arithmetic.

**What "rotating the model" means mechanically.** You cannot just rotate `X`, because `X` feeds `W`. But you can insert `H Hᵀ = I` into a linear layer for free:

```
Y = X Wᵀ
  = (X H)(Hᵀ Wᵀ)          # insert identity
  = X̃ W̃ᵀ                  # X̃ = XH is rotated, W̃ = W H is rotated
```

Both operands are rotated consistently, the output is identical in FP, and after training-free PTQ both have their outliers spread. Practical placements from QuaRot:

- **R1** — rotate the *input* to each block's `down_proj`. This matrix can be **fused into the preceding weights**, so it is completely free at inference.
- **R2** — rotate the *output* of the block / input to `o_proj` and `down_proj` weights, also fusable.
- **R3** — rotate the input to `down_proj` in a way that removes the need for a Hadamard on the residual stream.
- **R4** — rotate `Q` and `K` **after** RoPE. This one is *not* fusable into weights (RoPE sits between the weight and the rotation) so it must be applied online, or folded into the attention kernel. **It is the expensive one, and the reason rotation KV-cache quantization needs kernel cooperation.**

**Why the Fast Walsh–Hadamard Transform matters.** A dense `n × n` matmul would cost `O(n²)` per token — prohibitive. The Hadamard transform admits the FWHT: `O(n log n)`. For `n = 4096` that is `4096 × 12 ≈ 49k` operations instead of `16.8M` — a 341× reduction, which is what makes it deployable at all.

**SpinQuant's refinement.** QuaRot uses a fixed random Hadamard (plus random sign flips, which are necessary — a pure Hadamard has structure that interacts with the data). SpinQuant instead *learns* the rotation matrices by optimizing `Cayley`-parameterized orthogonal matrices on a small calibration set, then fuses them. The reported gain is meaningful — SpinQuant reports a 45.1% reduction in the 4-bit accuracy gap vs QuaRot on LLaMA-2-7B — because not every rotation is equally good for a given weight distribution.

**Why this is not free lunch.** Three costs:
1. **Kernel support.** A non-fusable rotation (R4) needs either a custom kernel or an online FWHT per attention call. `llm-compressor`'s `SpinQuantModifier` and QuaRot's kernels exist for this reason.
2. **The rotation must be applied consistently to weights *and* KV cache.** If you rotate weights but forget the cache, you get silently wrong outputs — not a crash.
3. **Some hardware/kernel paths are faster with the un-rotated layout**, so the FLOP savings can be eaten by layout conversions.

> **Beyond the video:** the video never mentions rotation. If someone asks "name a post-2023 PTQ technique that isn't GPTQ or AWQ," the correct answer is QuaRot/SpinQuant and the correct one-liner is *"Hadamard rotation makes weight/activation/KV distributions incoherent, so a per-tensor scale stops being dominated by a handful of outlier channels."*

### 4.6 The straight-through estimator, in code

CS-10 asserted that QAT needs a gradient through `round()`. Here is what that means numerically and what happens when you get it wrong.

**The problem.** Round-to-nearest is a step function:

```
q = round(x)                ∂q/∂x = 0  almost everywhere
                            ∂q/∂x = undefined at half-integers
```

If you build the fake-quant module naively and let autograd see `round`, every gradient arriving at a weight is multiplied by 0. **Training does not diverge — it does nothing.** The loss curve is flat from step 0. This is the single most common QAT bug and it looks like "my learning rate is too low."

**The fix.** Replace the backward pass with the identity:

```
forward : q = round(x)                    (true, hard quantization)
backward: ∂L/∂x := ∂L/∂q                  (STE: pretend the derivative is 1)
```

PyTorch's built-in `FakeQuantize` implements exactly this. The idiomatic way to write it yourself is a custom `autograd.Function`:

```python
import torch
from torch import nn

class _RoundSTE(torch.autograd.Function):
    """round() in the forward pass, identity in the backward pass."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass the gradient through unchanged.
        # (Optional refinement: zero it outside the clip range so the
        #  optimizer does not chase weights that are already clamped.)
        return grad_output

def fake_quant_affine(x, scale, zero_point, qmin=0, qmax=255):
    """q = clamp(round(x/s + z)); dequantize back to float.
    Gradient flows to x through the STE; scale/zero_point get the
    gradient of the (x - z*s) dequantization term only."""
    q = _RoundSTE.apply(x / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    return (q - zero_point) * scale
```

**Three things that break even *with* the STE in place:**

1. **Saturated weights get zero gradient forever.** Once `x/s + z` is outside `[qmin, qmax]`, `clamp` zeroes the gradient. The weight can never come back inside the range because it cannot move. This is why *learnable* `scale`/`zero_point` (the notebook's `QuantizedLinear`) matter: they let the *grid* move to the weights instead of relying on the weights moving onto the grid. It is also why observers should track min/max on a moving average rather than freezing after warmup.
2. **The STE is a biased gradient.** `∂L/∂x` is not the true gradient of any function; it is a first-order surrogate. It works because the quantization error is bounded and the direction is right on average, but it introduces gradient noise proportional to `Δ`. Consequence: **QAT wants a lower learning rate than the corresponding FP fine-tune** — the notebook uses `5e-5` for DistilGPT2 QAT, and the practical rule is 10–100× below the FP fine-tuning LR.
3. **The observer must warm up before `convert`.** `MinMaxObserver` accumulates on the first forward passes; if you `convert()` immediately after `prepare_qat()` with one sample, your scales come from that one sample. Standard recipe: a few hundred warmup steps in `train()` mode with the observer enabled (`FakeQuantize`'s `observer_enabled=True`), *then* freeze the observer (`disable_observer`) for the remaining steps, *then* `eval()` + `convert()`.

**What breaks with no STE at all** — a useful debugging checklist, because these are the three symptoms you will see:

| Symptom | Cause |
|---|---|
| Loss identical across steps, gradients all-zero on Linear weights | `round` exposed to autograd (no STE) |
| Loss decreases then plateaus at a value well above FP baseline | STE present, but LR too high — the noise floor of the surrogate gradient |
| Model fine in `train()` mode, garbage after `convert()` | Observer never warmed up, or `convert` called on a non-`eval()` model, or BN statistics not frozen |

### 4.7 QAT in PyTorch, end to end — the notebook walkthrough annotated

This is the canonical eager-mode flow. It is what the video's `QAT_in_LLM.ipynb` runs, and it is still what most interviewers mean when they say "do QAT."

**The five-step contract:**

```
train mode → prepare_qat() → train with fake quant → eval + freeze → convert()
```

**Step 1 — define the QConfig.** This object is the entire specification of *what* precision to simulate. Two independent halves:

```python
import torch, torch.nn as nn, torch.quantization as tq

qat_config = tq.QConfig(
    activation=tq.FakeQuantize.with_args(
        observer=tq.MovingAverageMinMaxObserver,   # smoothing, not raw min/max
        quant_min=0, quant_max=255,                # quint8 → 2^8 = 256 levels
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine),          # asymmetric: zero_point ≠ 0
    weight=tq.FakeQuantize.with_args(
        observer=tq.MinMaxObserver,
        quant_min=-128, quant_max=127,             # qint8 → signed
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric))       # symmetric: zero_point = 0
```

Why these choices, specifically:

- **Activations asymmetric (`per_tensor_affine`, `quint8`).** Post-GELU / post-ReLU activations are non-negative, so a signed grid wastes half its levels. Asymmetric unsigned is the right default.
- **Weights symmetric (`per_tensor_symmetric`, `qint8`).** Weights are approximately zero-mean; a symmetric grid avoids a zero-point and lets the kernel use a cheaper inner loop. Also required for the fastest Marlin path.
- **`MovingAverageMinMaxObserver` for activations vs `MinMaxObserver` for weights.** Activation ranges vary token to token and a single outlier token would blow up a raw max; the moving average (`averaging_constant=0.01` default) is robust. Weight ranges are fixed and known exactly, so raw min/max is optimal.

**The observer taxonomy** (interviewers love this table):

| Observer | Statistic | Use when | Cost |
|---|---|---|---|
| `MinMaxObserver` | running min/max | Weights; activations with fixed range | Cheapest |
| `MovingAverageMinMaxObserver` | EMA of min/max | Activations in QAT (rejects outlier tokens) | Cheap |
| `HistogramObserver` | 2048-bin histogram → minimize MSE between FP and quantized tensor | PTQ where accuracy matters most | Expensive, best scales |
| `PerChannelMinMaxObserver` | min/max **per output channel** | Weight quantization — always, if the backend allows | 1 scale per row |
| `PlaceholderObserver` | none; just declares dtype | Declaring FP16/FP32 paths in mixed configs | Free |

**Step 2 — exclude what must not be quantized.** This line is easy to miss and expensive to omit:

```python
for name, module in model.named_modules():
    if isinstance(module, nn.Embedding):
        module.qconfig = None     # embeddings are lookup tables: no compute to save
```

Also exclude `lm_head` when it is a separate `Linear` (the notebook leaves it, which is one reason 50 steps is enough to matter), and any multimodal `vision_tower` / `multi_modal_projector` — see §10.7.

**Step 3 — `prepare_qat`.** This is a graph rewrite. It walks the module tree and **replaces** quantizable modules with instrumented versions, inserting `FakeQuantize` observers around weights and after activations.

```python
model.qconfig = qat_config
model.train()                     # MUST be train() — prepare_qat behaves differently otherwise
tq.prepare_qat(model, inplace=True)
```

After this, `model` contains `~FakeQuantize` submodules and the `Linear` layers are wrapped. The FP weights are untouched; the observers are populated on the first forward pass.

**Step 4 — the training loop.** Structurally identical to any fine-tune; the difference is invisible in the code and total in the numerics.

```python
inputs = tokenizer("Quantization Aware Training on LLMs!", return_tensors="pt")
labels = inputs["input_ids"]
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for step in range(50):                       # toy: 50 steps on ONE example
    loss = model(**inputs, labels=labels).loss
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss.item()}")
```

**What the notebook does not do, that you must:** 50 steps on a single sentence is a *smoke test that the plumbing works*, not a QAT recipe. Production QAT needs (a) 1–10% of the original pretraining token budget or, for fine-tuning-style QAT, the full SFT set; (b) an observer-freeze point at roughly 80–90% of the schedule; (c) an LR schedule decaying to ~0; and (d) a held-out eval that compares against *both* the FP baseline and a PTQ baseline at the same bit-width.

**Step 5 — `convert`. The order is load-bearing:**

```python
qat_model = tq.convert(model.eval(), inplace=False)
```

`eval()` first (freezes BatchNorm statistics and switches observers off), *then* `convert`. `convert` replaces fake-quant modules with real quantized modules and packs the weights into `qint8` with the observed scales. Calling `convert` before `eval()` is the classic bug that produces a model which generates plausible-but-wrong text.

**Bonus: the notebook's own `QuantizedLinear` — a hand-rolled QAT layer.** Worth studying because it makes every moving part explicit:

```python
class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # LEARNABLE quantization parameters
        self.weight_scale      = nn.Parameter(torch.ones(1))
        self.weight_zero_point = nn.Parameter(torch.zeros(1))
        self.input_scale       = nn.Parameter(torch.ones(1))
        self.input_zero_point  = nn.Parameter(torch.zeros(1))
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        ...

    def quantize_tensor(self, tensor, scale, zero_point):
        quantized   = torch.round(tensor / scale + zero_point)
        quantized   = torch.clamp(quantized, -128, 127)
        dequantized = (quantized - zero_point) * scale
        return dequantized

    def forward(self, x):
        q_weight = self.quantize_tensor(self.weight, self.weight_scale, self.weight_zero_point)
        q_input  = self.quantize_tensor(x,          self.input_scale,  self.input_zero_point)
        return nn.functional.linear(q_input, q_weight, self.bias)
```

Three observations: (1) it relies on PyTorch's autograd to route through `round` as if it were identity — which is *not* the same as a proper STE, because `round`'s true derivative is zero and autograd will therefore **zero the weight gradient**; this code as written does not learn weight updates correctly. (2) The `scale`/`zero_point` gradients are non-zero, so the *grid* does adapt — which is why the model still improves. (3) `torch.clamp` outside the round is the saturated-weight trap from §4.6. **Use it to explain the mechanics; use `torch.ao.quantization` or `torchao` to ship.**

**Modern replacement — `torchao`.** The 2026 API:

```python
# pip install torchao
import torch
from torchao.quantization import quantize_, int8_weight_only, int4_weight_only
from torchao.quantization.qat import QATConfig, int8_symmetric

# QAT: swap the Linear layers for fake-quantized ones, then train normally
quantize_(model, QATConfig(int8_symmetric(activation_dtype=torch.int8), step="prepare"))
# ... training loop as usual ...
quantize_(model, QATConfig(quantization_config, step="convert"))
```

`torchao` also owns the modern weight-only PTQ path (`int4_weight_only(group_size=128)`, `int8_weight_only()`, `float8_weight_only()`) applied with a single `quantize_()` call. For a 2026 interview: **know `torchao` exists, know it is the maintained API, know `quantize_(model, config)` is the shape of the call.**

> **Correction:** `torch.ao.quantization`'s eager QAT flow (`prepare_qat` / `convert` / `QConfig`) is in maintenance mode. It is not removed and will still run, and it is what the video teaches, but new production code should use `torchao`. The concepts transfer one-to-one (observers, fake quant, warmup, freeze, convert) — only the API surface changed.

### 4.8 A note on granularity, since everything above depends on it

CS-10 introduced per-tensor vs per-channel. The production-relevant detail is the **overhead arithmetic**, because this is what decides which granularities a kernel supports.

For a weight matrix with `P` parameters stored at `b` bits with group size `g` and FP16 scales + FP16 zero-points (asymmetric):

```
bits per parameter  =  b  +  16/g (scale)  +  16/g (zero-point)

g = 128 : 4 + 0.125 + 0.125 = 4.25 bpw   → 7B model ≈ 3.72 GB
g = 64  : 4 + 0.25  + 0.25  = 4.50 bpw   → 7B model ≈ 3.94 GB
g = 32  : 4 + 0.50  + 0.50  = 5.00 bpw   → 7B model ≈ 4.38 GB
g = -1  : per-channel, effectively 4 + 16/d_out ≈ 4.001 bpw
```

Symmetric quantization (the common weight case) drops the zero-point term, saving `16/g` — which is why `sym=True` is the default in GPTQModel and the fast Marlin path. **Note that `group_size` is not free: going from 128 to 32 costs you 0.75 bpw, which at 7B is ~0.66 GB.** "Smaller group size is better" is true for quality and false for everything else.

The other half of the granularity story: **activations must be quantized per-token, not per-tensor**, once you are in W8A8 territory, because activation ranges vary by orders of magnitude across tokens. This is not a refinement — it is the difference between W8A8 working and W8A8 destroying the model. It is also why the PyTorch eager path's `per_tensor_affine` activation config is a *toy*: it works on DistilGPT2 because there are no emergent outliers at 82M parameters (§4.4).

### 4.9 KV-cache quantization

This is the highest-value section in the module for anyone serving models, and the video does not mention it once.

**Why it matters: the cache scales with context × batch, not with parameters.**

```
KV bytes = 2  ×  n_layers  ×  n_kv_heads  ×  head_dim  ×  seq_len  ×  batch  ×  bytes_per_element
           ↑
           (one K tensor + one V tensor)
```

Worked example — **Llama-3.1-8B**, which uses grouped-query attention with `n_layers=32`, `n_kv_heads=8`, `head_dim=128`:

```
per token, FP16: 2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes = 128 KB / token
```

Now scale it:

| Context | Batch | FP16 KV cache | FP8 KV cache | INT4 KV cache |
|---|---|---|---|---|
| 8k | 1 | 1.0 GB | 0.5 GB | 0.25 GB |
| 32k | 1 | 4.0 GB | 2.0 GB | 1.0 GB |
| 32k | 8 | 32.0 GB | 16.0 GB | 8.0 GB |
| 32k | 32 | 128.0 GB | 64.0 GB | 32.0 GB |
| 128k | 8 | 128.0 GB | 64.0 GB | 32.0 GB |
| 128k | 32 | 512.0 GB | 256.0 GB | 128.0 GB |

Compare to the weights: Llama-3.1-8B at INT4 is about **4.0 GB**. At 32k context × batch 8 the FP16 KV cache is **8× the weight footprint**; at 128k × 8 it is 32×. Weight quantization is a rounding error next to this.

> **The arithmetic to memorize:** for a GQA model, `KV bytes/token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element`. Derive it from "one K and one V, per layer, per KV head, of `head_dim` elements" and you will never need to look it up.
>
> **Sanity check for MHA:** if Llama-3.1-8B used full multi-head attention (`n_kv_heads = 32` instead of 8), it would be **512 KB/token** — 4× worse. GQA is worth more than quantization for long-context memory.

**FP8 KV cache.** `bytes_per_element` goes 2 → 1, so you get a flat 2×. It is the single highest-leverage flag in a vLLM launch command and it is nearly free in quality:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-cache-dtype fp8 \        # or fp8_e4m3 / fp8_e5m2
    --calculate-kv-scales \       # calibrate scales instead of using a running max
    --max-model-len 32768
```

Why FP8 and not INT4, given INT4 would be 4×? Three reasons, and they are the exam answer:

1. **PagedAttention's block layout must stay byte-addressable.** vLLM stores KV in fixed-size blocks (`block_size = 16` tokens) with a block table doing virtual→physical mapping and copy-on-write sharing for prefixes. FP8 keeps the element size at a power-of-two byte count and keeps the layout trivially sliceable. Sub-byte KV would require bit-packing *inside* a block, breaking the copy-on-write and the gather kernels.
2. **Per-block FP8 scales are one `float32` per block per K/V tensor** — 4 bytes per 16 tokens of 128-dim KV, i.e. ~0.2% overhead. An INT4 scheme would need a per-block scale *and* zero-point and a dequant in the attention inner loop, where there is no compute headroom (attention at long context is bandwidth-bound on the QK^T read).
3. **Hardware.** FP8 KV requires compute capability ≥ 8.9 (Ada / Hopper). vLLM will silently fall back or error on older cards — check `torch.cuda.get_device_capability()` before you plan on it.

**K versus V: they are not equally sensitive, and the mechanism is asymmetric.**

- **V errors enter the output linearly.** Attention output is `Σ_i p_i v_i` with `Σ p_i = 1`. An error `ε` in one cached `v_i` propagates as `p_i ε` — bounded by `‖ε‖`, and *averaged down* when attention is diffuse. V is comparatively forgiving.
- **K errors enter the logits before the softmax.** `scores = q kᵀ/√d`; a perturbation `δ` in `k` shifts a score by `q·δ/√d`. Softmax compresses this nonlinearly, but when two attention scores are within the perturbation, **the argmax flips and attention jumps to a different token entirely.** That is a catastrophic, discontinuous error — and it is why **long-context tasks that need precise retrieval degrade before anything else.** Needle-in-a-haystack and RULER are the tests that catch it (§12.5).

**The mechanism behind the K/V granularity asymmetry.** It is not a property of K and V as such; it is a property of *where outliers live*:

- **K has persistent channel-wise outliers.** Certain head-dim indices are large for essentially every token (the same phenomenon as §4.4). Per-token quantization cannot help — the outlier is inside the token. → **quantize K per-channel.**
- **V has token-wise outliers.** A few tokens have large V vectors overall. Per-channel quantization cannot help. → **quantize V per-token.**

Under **per-tensor** quantization (the naive choice) both suffer, and K suffers worse because its outliers are always present. This is precisely the recipe in **KIVI** (2-bit KV via per-channel K + per-token V) and it is the reason "quantize K and V differently" is a real answer, not a hack.

**PagedAttention interaction, specifically.** Because the cache is blocked, scales must be stored **per block**, and the attention kernel must apply them during the gather. Consequences you will actually hit:

- `--block-size` changes the scale granularity. Smaller blocks = finer scales = better quality, more metadata, slower.
- **Prefix caching and copy-on-write must preserve the scales with the block**, or a shared prefix gets dequantized with the wrong scale.
- **`--calculate-kv-scales`** runs a calibration pass to set scales from observed data rather than a per-block running max. It costs startup time and buys quality, particularly at 4-bit-ish effective KV precision.
- FP8 KV and **chunked prefill** interact: the prefill chunks write into the same blocks, so the scale must be committed consistently.

**Practical recipe for a long-context deployment:**

```
1. Weights   → AWQ or GPTQ W4A16 (Marlin kernel)          : 4× weight reduction
2. KV cache  → --kv-cache-dtype fp8 --calculate-kv-scales  : 2× cache reduction
3. If still OOM → raise --gpu-memory-utilization to 0.92,
                  reduce --max-model-len, or add tensor parallelism
4. NEVER reduce precision below fp8 for K without re-running the
   long-context needle test (§12.5). This is where models silently break.
```

> **Beyond the video:** in 2026 the *first* question in a serving interview about quantization is "weights or cache?" If the answer is "weights," you have not served anything long-context. The 2025–26 literature has shifted accordingly: papers like KIVI, KVQuant, and the FP8-KV support in vLLM/TensorRT-LLM exist because cache quantization is where the remaining wins are.

### 4.10 The precision lattice: W8A8, W4A16, W4A8, W4A4, and what your GPU can actually do

**Notation.** `W<n>A<m>` = weights at *n* bits, activations at *m* bits. The letter after the "A" is the thing people get wrong.

| Scheme | Weights | Activations | Compute engine actually used | Realistic 2026 status |
|---|---|---|---|---|
| **W16A16** | FP16/BF16 | FP16/BF16 | FP16/BF16 tensor cores | baseline |
| **W8A16** | INT8 | FP16 | FP16 tensor cores + dequant | good memory win, no compute win |
| **W8A8** | INT8 | INT8 | **INT8 tensor cores** | mature, near-lossless; needs SmoothQuant-class methods |
| **W4A16** | INT4/NF4 | FP16 | **FP16 tensor cores + dequant (Marlin)** | **the production default** |
| **W4A8** | INT4 | INT8 | mixed; kernel-dependent | niche, growing |
| **W4A4** | INT4 | INT4 | INT4/INT8 tensor cores | research-grade without rotation/QAT |
| **W4A8-FP8** | INT4 | FP8 | FP8 tensor cores | the Hopper-era sweet spot |
| **W8A8-FP8** | FP8 | FP8 | **FP8 tensor cores** | fastest path on Ada/Hopper |
| **W4A4-FP4** | NVFP4 | NVFP4 | **FP4 tensor cores** | Blackwell only |
| **W1.58A8** | ternary | INT8 | add-only | BitNet models only |

**The single most important correction to the video's framing:**

> **Correction:** **W4A16 does not use INT4 tensor cores.** The 4-bit weights are stored packed, and the kernel *dequantizes them to FP16 on the fly* and issues an ordinary FP16 tensor-core matmul. The savings are bandwidth, not arithmetic. This is exactly why **Marlin** exists and why "4-bit should be 4× faster" is wrong — a naive W4A16 kernel is often *slower* than FP16 at small batch because the dequant is on the critical path. This is also why the video's implicit claim that AWQ quantizes activations is wrong: **AWQ ships as W4A16.** He corrects himself later in the video; the correction is the examinable fact.

**The hardware generation table** — this is the "which precision can I actually run" answer:

| Generation | Example | Native low-precision | Notes |
|---|---|---|---|
| Volta (sm_70) | V100 | INT8 (limited) | no INT4 tensor cores |
| Turing (sm_75) | T4, RTX 20xx | INT8, **INT4** | first INT4 tensor cores; no BF16 |
| Ampere (sm_80/86) | A100, A10, RTX 30xx | INT8, INT4, TF32, BF16 | **no FP8**; Marlin targets sm_80+ |
| Ada (sm_89) | L4, L40S, RTX 40xx | + **FP8** (e4m3/e5m2) | **minimum for FP8 KV cache** |
| Hopper (sm_90) | H100, H200 | FP8 + Transformer Engine, TMA, wgmma | FP8 is the fastest production path here |
| Blackwell (sm_100/120) | B200, RTX 50xx | + **FP4 / FP6** | NVFP4 = block-16 + E4M3 scales; MXFP4 = block-32 + E8M0 |

**What this means for choosing a scheme** (and it is a *hardware* question before it is an *algorithm* question):

- **A100 / A10 / RTX 30xx:** W4A16 (Marlin) for weights. No FP8 anywhere — no FP8 KV cache, no FP8 weights. W8A8 is available and fast.
- **L4 / L40S / RTX 40xx:** FP8 becomes available; `--kv-cache-dtype fp8` works; FP8 W8A8 is the fastest dense path.
- **H100 / H200:** FP8 everywhere, Transformer Engine, FP8 KV. W4A16 (Marlin) still wins on memory-bound decode at small batch.
- **B200 / RTX 50xx:** NVFP4 is the native 4-bit *float* path — a 4-bit float with a block-16 E4M3 scale has better dynamic range behaviour than INT4 and runs on real FP4 tensor cores, so W4A4 stops being research.

**FP8 format selection, concretely.** `e4m3` has 4 exponent bits and 3 mantissa bits, giving a max magnitude of ±448 with ~2 decimal digits of precision. `e5m2` trades mantissa for range (max ±57,344) at 1 mantissa bit. Rule: **`e4m3` for weights, activations, and KV cache (range is bounded, precision matters); `e5m2` for gradients and any tensor whose outliers you cannot bound.** Getting this backwards on gradients produces NaNs.

### 4.11 QLoRA: NF4, double quantization, paged optimizers, and the 6 GB math

**This subsection is flagged must-know.** QLoRA is the most-asked quantization topic in applied ML interviews because it is *the* way anyone trains a large model on one GPU, and because the numbers are checkable.

**NF4 — why a quantile grid.** Standard INT4 uses 16 evenly spaced levels. But transformer weights are approximately Gaussian, so evenly spaced levels put most of their resolution in the tails where almost no weight lives and too little near zero where most weights live.

NF4 instead places its 16 levels at the **quantiles of a standard normal** — i.e. the grid points `q_i` are chosen so that each bin contains equal probability mass `1/16`:

```
levels ≈ [−1.0, −0.6962, −0.5251, −0.3949, −0.2844,
          −0.1848, −0.0911, 0.0, 0.0796, 0.1609,
           0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```

Why this is the right call: **for a source with a known distribution, the Lloyd-Max optimal scalar quantizer places its levels at the centroids of equal-probability bins.** Equal probability mass per bin is the condition that equalizes the contribution to MSE across the range. For a Gaussian source this is exactly the quantile grid — so NF4 is optimal by construction, not by luck. It is then **normalized to `[−1, 1]`** and applied with a per-block absolute-max scale.

Mechanically: weights are chunked into **blocks of 64**, each block quantized with `absmax` scaling and NF4 levels. So a 4-bit weight costs exactly 4 bits + one FP32 absmax per 64 weights = `4 + 32/64 = 4.5` bits per parameter.

**Double quantization — the 0.373 bits/param.** The absmax constants above are themselves a lot of memory: one FP32 per 64 weights is `0.5` bits/param, i.e. **11% of your total budget spent on scales.**

Double quantization quantizes the scales. The block-64 absmax values are themselves grouped into blocks of 256 and quantized to **8-bit with their own FP32 absmax**:

```
Single quantization : 4 + 32/64                              = 4.5000 bpw
Double quantization : 4 +  8/64 + 32/(64×256)                = 4.1270 bpw
                          ↑         ↑
                   8-bit absmax   FP32 second-level absmax

saving              : 4.5000 − 4.1270 = 0.3730 bits/param
```

Which reproduces the QLoRA paper's own "~0.37 bits per parameter" and their headline "~3 GB for a 65B model":

```
65e9 params × 0.373 bits / 8 bits-per-byte = 3.03 GB        ← matches the paper
 7e9 params × 4.127 bits / 8 bits-per-byte = 3.61 GB        ← 7B NF4 + DQ base weights
```

> **The arithmetic to have cold.** Any interviewer who asks about QLoRA will accept `4 + 8/64 + 32/16384 = 4.127 bpw` as proof you know what double quantization actually does. Most candidates say "it quantizes the scales too"; very few can produce the number.

**Paged optimizers.** During a long training step, gradient checkpointing spills activations and the optimizer state can spike; a transient spike OOMs the run. Paged optimizers put the AdamW state in **NVIDIA unified memory** so pages can migrate to host RAM under pressure and return later, at a latency cost but without a crash. It is a *robustness* feature, not a memory-saving one in the steady state — which is exactly why it matters: it converts the occasional catastrophic OOM into a slow step.

**The 6 GB budget for a 7B, itemized.** Assume a 7B model at NF4 + double quantization with LoRA `r=16` on all attention + MLP projections:

| Component | Size | Derivation |
|---|---|---|
| NF4 + DQ base weights | **3.61 GB** | `7e9 × 4.127 / 8` |
| LoRA adapter weights | **~87 MB** | `r=16` adds roughly `2 × 16 × d` per adapted matrix; across a 7B that is ~21M params × 4 bytes (fp32) |
| LoRA gradients | **~87 MB** | same count, fp32 |
| AdamW optimizer state | **~348 MB** | `2 × 21.8M × 8 bytes` (exp_avg + exp_avg_sq, fp32) — LoRA-only params, not the base |
| Activations (grad ckpt on) | **~0.3–0.6 GB** | sequence- and batch-dependent; `bs=1, seq=2048` sits at the low end |
| CUDA context + workspaces + fragmentation | **~0.6–1.2 GB** | library allocations, cuBLAS workspaces, ~10–15% fragmentation |
| **Total** | **≈ 5.1–6.0 GB** | fits a 6 GB card at `bs=1`, `seq=1024–2048` |

Sensitivity, so you can answer follow-ups:

- **`r=64` instead of `r=16`**: LoRA params ×4 → weights 348 MB, grads 348 MB, optimizer 1.39 GB → total ≈ **7.2 GB**. Still one 8 GB card.
- **Full fine-tuning a 7B** needs the same 3.61 GB base **plus** 7B × (2 bytes grad + 8 bytes AdamW) ≈ **70 GB** of optimizer+gradient state. QLoRA is a **~12×** reduction, and the 3.6 GB base is only 5% of the difference — *the optimizer state is the real win, not the quantization*.
- **The LoRA-only optimizer state is why QLoRA scales**: memory is `O(LoRA params)`, not `O(model params)`.

**The correct order of operations**, which is §10.6's gotcha stated here for completeness:

```
CORRECT  :  bf16 base → attach LoRA → train → merge adapter into bf16 → quantize → serve
WRONG    :  quantize base → attach LoRA → train → serve     (this is QLoRA, and it is
                                                             fine for *training*, but the
                                                             adapter's gains partially
                                                             compensate for quantization
                                                             error you could have avoided)
WRONG-EST:  quantize base → full fine-tune → serve          (gradients do not flow
                                                             through NF4; you will
                                                             train nothing or destroy it)
```

> **Beyond the video:** the video's QLoRA-adjacent cells use bitsandbytes `load_in_4bit` + PEFT and call it "QAT." It is not QAT — it is QLoRA, which is quantization *plus* low-rank adaptation, and the frozen base weights never move. Getting this label right in an interview matters more than it sounds, because the follow-up question ("so what is the difference from QAT?") is immediate.

---

## §5 The End-to-End Pipeline

### 5.1 The full decision-and-execution flow

```
                        ┌─────────────────────────────────────────┐
                        │ 0. BASELINE: what is my bottleneck?     │
                        └────────────────┬────────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
   weights (decode BW)          KV cache (context)              train-time state
        │                                │                                │
        │                                │                                │
        ▼                                ▼                                ▼
  ┌───────────────┐             ┌────────────────┐              ┌──────────────────┐
  │ bf16 baseline │             │ bf16 baseline  │              │ bf16 base + LoRA │
  │ + eval gate   │             │ + needle test  │              │ + eval gate      │
  └───────┬───────┘             └───────┬────────┘              └────────┬─────────┘
          │                             │                                │
          ▼                             ▼                                ▼
  ┌───────────────┐             ┌────────────────┐              ┌──────────────────┐
  │ need to TRAIN │             │ fp8 KV         │              │ QLoRA: NF4+DQ    │
  │ on top?       │             │ --kv-cache-    │              │ r=16..64, paged  │
  │  NO → PTQ     │             │  dtype fp8     │              │ AdamW            │
  │  YES → QLoRA  │             │ --calculate-   │              └────────┬─────────┘
  └───────┬───────┘             │  kv-scales     │                       │
          │                     └───────┬────────┘                       ▼
          ▼                             │                        ┌──────────────────┐
  ┌───────────────┐                     │                        │ merge → bf16     │
  │ W4A16:        │                     │                        │ re-quantize      │
  │ AWQ or GPTQ   │◄────────────────────┘                        │ (AWQ/GPTQ)       │
  │ + Marlin      │                                              └──────────────────┘
  └───────┬───────┘
          │
          ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ EVAL GATE (§12)  —  mean KL, top-1 agree, p95 KL, IFEval,            │
  │                     GSM8K, needle@ctx, latency p50/p99              │
  └───────────────────────────┬──────────────────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               │ PASS?                       │ FAIL
               ▼                             ▼
     ┌───────────────────┐         ┌─────────────────────────────┐
     │ ship              │         │ escalate in THIS order:     │
     │ pin versions,     │         │ 1. group_size 128 → 64      │
     │ hash artifacts,   │         │ 2. add rotation/QuaRot      │
     │ record calib hash │         │ 3. W4A16 → W8A8 (SmoothQuant)│
     └───────────────────┘         │ 4. QAT (if you own model)   │
                                   │ 5. accept W8A16 / FP16      │
                                   └─────────────────────────────┘
```

### 5.2 The order of operations, as a checklist

1. **Establish the FP baseline and the eval gate *before* quantizing.** You cannot detect damage you never measured. Hash the baseline's metrics into your experiment record.
2. **Pick the precision from the decision tree (§8), not from the paper you last read.**
3. **Fix the calibration set.** Domain-matched, held out from your eval set, 128–1024 samples, `max_seq_len` set to your *deployment* context (a 128-token calibration for a 32k-context model is a category error — activation statistics at 32k differ).
4. **Quantize. Record `(method, bits, group_size, desc_act, sym, calibration hash, library version, GPU arch).** All seven. Any of them missing makes the artifact unreproducible (§10.19).
5. **Re-run the full eval gate on the quantized artifact.**
6. **Re-run the gate again on the *served* artifact** — after `convert`, after GGUF conversion, after the serving engine loads it. Every serializer is a place to break something (§10.2, §10.3).
7. **Load-test at your real batch size.** Quantization's throughput win is batch-dependent (§7.3) and your benchmark at batch 1 does not predict batch 32.
8. **Pin, hash, and document.** If a rebuild cannot reproduce the number, you do not have a production artifact — you have a lucky one.

---

## §6 Annotated Hands-On Code

The video's four notebooks are the source for this section. Each block below is annotated for *what to change for a real model*, which is where the notebooks are thinnest.

### 6.1 QAT — what the notebook does vs what production does

The notebook (fully annotated in §4.7) runs `prepare_qat` → 50 steps → `convert` on DistilGPT2. Porting it to a real 7B is where everything breaks:

| Notebook | Production 7B | Why |
|---|---|---|
| `AutoModelForCausalLM.from_pretrained("distilgpt2")` | `...from_pretrained(name, torch_dtype=torch.bfloat16)` | FP32 weights double memory for no gain; bf16 is the QAT-preferred base |
| `model.qconfig = qat_config`, `prepare_qat(model)` | Same, but **must** be gated on `LoraConfig`-free full FT or a `torchao` QATConfig | `torch.ao` eager QAT on a `LlamaForCausalLM` needs `qconfig` propagation through HF-specific modules; `torchao` handles it |
| 50 steps, 1 example, `lr=5e-5` | 1–10% of pretrain tokens or a full SFT set, cosine decay to 0, `lr ≈ 1e-5` | QAT gradient noise is proportional to `Δ`; high LR never converges |
| `tq.convert(model.eval())` immediately | Warmup 200–500 steps → `disable_observer` → finish schedule → `eval()` → `convert` | Observers need data; freezing them late is what makes the scales good |
| Not measured | Compare against FP baseline **and** PTQ baseline at same bits | QAT's win is only meaningful relative to PTQ |

### 6.2 The straight-through estimator, runnable and self-checking

The following is a complete, runnable demonstration of both the failure and the fix. Run it before you debug any QAT job.

```python
import torch, torch.nn as nn

torch.manual_seed(0)

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, g):
        return g                      # identity: the STE

class QuantLayers(nn.Module):
    """Two identical 1-layer nets: one with STE, one without."""
    def __init__(self, use_ste):
        super().__init__()
        self.use_ste = use_ste
        self.w = nn.Parameter(torch.randn(4, 4) * 0.5)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def fake_quant(self, w):
        q = w / self.scale
        q = RoundSTE.apply(q) if self.use_ste else torch.round(q)   # ← the only difference
        return q * self.scale

    def forward(self, x):
        return x @ self.fake_quant(self.w).T

x = torch.randn(8, 4)
y = torch.randn(8, 4)

for use_ste in (False, True):
    net = QuantLayers(use_ste)
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    for _ in range(50):
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # Gradient norm on the weight tells you immediately whether the STE is wired up
    print(f"use_ste={use_ste!s:5}  loss={loss.item():.4f}  "
          f"|grad_w|={net.w.grad.norm().item():.2e}")
```

Expected output shape: with `use_ste=False` the loss barely moves and `|grad_w|` collapses toward zero, because `round`'s true derivative is zero almost everywhere. With `use_ste=True` the loss falls normally. **If your QAT run's weight gradients are ~1e-8, this is why.**

### 6.3 GPTQ via `gptqmodel` (the maintained successor to AutoGPTQ)

```python
# pip install -v gptqmodel --no-build-isolation ; pip install "protobuf<6.30"
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset

model_id  = "meta-llama/Llama-3.2-1B-Instruct"
quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

# --- calibration: 1024 samples of domain-matched text, NOT generic web if you serve code
calibration_dataset = load_dataset(
    "allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train"
).select(range(1024))["text"]

# --- the config that matters (full glossary in §7.1)
quant_config = QuantizeConfig(
    bits=4,               # 2/3/4/8
    group_size=128,       # 128 = the Marlin-friendly default; 64 for a quality bump
    desc_act=True,        # activation-order columns; better accuracy, slower
    damp_percent=0.05,    # Hessian damping for numerical stability
    sym=True,             # symmetric weights → no zero-point → 0.25 bpw cheaper
    lm_head=False,        # never quantize the output projection
)

model = GPTQModel.load(model_id, quant_config)
model.quantize(calibration_dataset, batch_size=500)   # batch_size = VRAM, not quality
model.save(quant_path)
```

**Reading the config the way a reviewer would:** `bits` and `group_size` set the size/quality point. `desc_act` and `damp_percent` set how well the Hessian machinery works. `sym` and `lm_head` set how much you refuse to quantize. Everything else (`parallel_packing`, `pack_dtype`, `v2`, `mock_quantization`) is throughput or debugging.

**Evaluate it in the same script, because otherwise nobody will:**

```python
from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

results = GPTQModel.eval(
    "ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1",
    framework=EVAL.LM_EVAL,
    tasks=[EVAL.LM_EVAL.ARC_CHALLENGE],
)
# The notebook's own run reports acc_norm = 0.2799 ± 0.0131 — a *single-task* number.
# One benchmark task with a ±0.013 error bar cannot separate a good 4-bit model
# from a mediocre one. Use the §12 gate, not this.
```

**Also measure the memory you actually got, not the memory you expected:**

```python
import torch
def measure_gpu_memory(loader, *a, **kw):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    m = loader(*a, **kw).to("cuda:0")
    inp = torch.randint(0, 100, (1, 16)).to("cuda:0")
    try: m.generate(inp)
    except Exception: m(inp)
    print(f"alloc {torch.cuda.memory_allocated()/2**20:.1f} MB  "
          f"reserved {torch.cuda.memory_reserved()/2**20:.1f} MB  "
          f"peak {torch.cuda.max_memory_allocated()/2**20:.1f} MB")
    del m; torch.cuda.empty_cache()
```

> **Correction:** **AutoGPTQ is deprecated** (last release 0.7.1, Aug 2024; pinned to Python 3.11). The video's notebook names it as the standard. In 2026 use **`gptqmodel`** (multi-hardware, built-in `lm-eval`, Marlin/ExLlamaV2/BitBLAS kernels) or vLLM's `llm-compressor`, or let vLLM load a pre-quantized GPTQ checkpoint directly. Anything you find online that says `pip install auto-gptq` is at least two years stale.

### 6.4 AWQ via AutoAWQ — and its replacement

```python
# pip install -U autoawq transformers accelerate
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoConfig

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

mdl = AutoAWQForCausalLM.from_pretrained(
    base_model, low_cpu_mem_usage=True, use_cache=False,
    torch_dtype=torch.float16, device_map={"": 0})

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Calibration: the notebook uses 100 hand-written generic sentences ×10.
# That is the single weakest part of the notebook — see the warning below.
calib_texts = ["The quick brown fox jumps over the lazy dog.", ...] * 10

mdl.quantize(tok, quant_config=quant_config, calib_data=calib_texts[:50],
             max_calib_seq_len=128, max_calib_samples=50, n_parallel_calib_samples=1)
mdl.save_quantized("tinyllama-1.1b-awq", safetensors=True)
```

**The three things to change for production:**

1. **`calib_texts` must be domain-matched.** 100 generic sentences will not expose the activation statistics of code, SQL, or clinical text. The notebook's own OOM fallback path retries with *fewer* samples — the right move when you OOM is to keep 128–512 samples and **shorten `max_calib_seq_len`**, not to drop to 20 samples.
2. **`version`** should be `"GEMM"` for batch>1 serving and `"GEMV"` for single-stream low-latency; the notebook hardcodes `GEMM`.
3. **`n_parallel_calib_samples=1`** is the OOM-avoidance setting; raise it to 8–16 on a real GPU or calibration takes forever.

> **Correction:** **AutoAWQ was archived in May 2024** and is superseded by vLLM's **`llm-compressor`**. The current AWQ path is `AWQModifier(scheme="W4A16")` inside a `oneshot(...)` recipe, which makes the W4A16 scheme explicit — and confirms that AWQ does **not** quantize activations:

```python
# pip install llm-compressor
from llmcompressor.modifiers.quantization import AWQModifier
from llmcompressor.transformers import oneshot
from llmcompressor import CompressionRecipe

recipe = CompressionRecipe(AWQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"]))
oneshot(model=model, recipe=recipe, dataset=calibration_dataset,
        num_calibration_samples=512, max_seq_length=2048)
```

The notebook's own markdown cell points at this replacement; the code above it still uses the archived library. **Know both, ship the new one.**

### 6.5 GGUF and `llama.cpp`

```bash
# --- build
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON      # or -DGGML_METAL=ON on Apple silicon
cmake --build build --config Release -j

# --- convert HF → GGUF (F16/BF16 container, no quantization yet)
python3 convert_hf_to_gguf.py ./tinyllama-hf --outfile ./tinyllama-1.1b-chat-f16.gguf

# --- generate an importance matrix from YOUR data (this is the quality lever)
./build/bin/llama-imatrix -m ./tinyllama-1.1b-chat-f16.gguf \
    -f domain_corpus.txt -o imatrix.gguf --chunks 200

# --- quantize (llama.cpp renamed `quantize` → `llama-quantize`)
./build/bin/llama-quantize --imatrix imatrix.gguf \
    ./tinyllama-1.1b-chat-f16.gguf ./tinyllama-q4_k_m.gguf Q4_K_M

# --- run
./build/bin/llama-cli -m ./tinyllama-q4_k_m.gguf -p "Explain quantization" -n 100
```

**The GGUF quant ladder** — bpw here is *effective* bits per weight, including scale overhead:

| Quant | bpw | 7B size | Use |
|---|---|---|---|
| `Q8_0` | 8.50 | ~7.2 GB | reference; effectively lossless |
| `Q6_K` | 6.56 | ~5.5 GB | near-lossless; the safe default when you have memory |
| `Q5_K_M` | 5.68 | ~4.8 GB | strong quality/size balance |
| **`Q4_K_M`** | **4.85** | **~4.1 GB** | **the community default; start here** |
| `Q4_0` | 4.50 | ~3.8 GB | legacy; `Q4_K_M` is strictly better at similar size |
| `Q3_K_M` | 3.91 | ~3.3 GB | visible degradation on reasoning; often still fine for chat |
| `Q2_K` | 3.35 | ~2.8 GB | noticeable; use only when memory is absolute |
| `IQ1_S` | 1.56 | ~1.4 GB | i-quants; dramatic quality loss; research/demo only |

**The `imatrix` point is the one people skip and shouldn't.** Without it, k-quants use a heuristic per-tensor importance. With it, llama.cpp computes per-tensor importance from *your* corpus (`Σ X_j²` style statistics, the same family as GPTQ/AWQ importance) and quantizes accordingly. It costs one pass over a few hundred chunks of text and is the single biggest quality improvement available for free in the GGUF pipeline.

Python side, for application integration:

```python
# pip install llama-cpp-python
from llama_cpp import Llama
llm = Llama(model_path="./tinyllama-q4_k_m.gguf", n_gpu_layers=35)  # 0 = CPU only
print(llm("Explain GPTQ vs AWQ in 2 lines.", max_tokens=80)["choices"][0]["text"])
```

`n_gpu_layers` is the partial-offload knob: it moves *n* transformer blocks to the GPU and leaves the rest on CPU, which lets a 70B run on a 24 GB card at a few tokens/second. It is also the reason llama.cpp remains the right tool for laptop and edge deployment.

> **Beyond the video:** the notebook runs `!make -j` (the pre-CMake build) and `./bin/quantize`. Current llama.cpp uses CMake (`cmake -B build`) and names the binary `llama-quantize`. The notebook's later cells already use the CMake flow, so it is internally inconsistent — another reason to read it as a *teaching* artifact rather than a script to copy. The `GGML` section of the notebook (`ggml-model-q4_0.bin` from TheBloke) is obsolete: GGML-format files are no longer loadable by current llama.cpp.

---

## §7 Hyperparameter Knobs

### 7.1 `QuantizeConfig` (GPTQModel) — every field that changes your outcome

| Field | Default | What it does | When to change |
|---|---|---|---|
| `bits` | 4 | Weight precision (2/3/4/8) | 4 is the default; 3 for extreme compression; 8 for near-lossless |
| `group_size` | 128 | Weights per scale group | 128 default; 64 for a quality bump at +0.25 bpw; **−1 required for fastest Marlin** |
| `desc_act` | True | Order columns by Hessian diagonal | Keep True for accuracy; disable if a kernel is incompatible or you need raw speed |
| `damp_percent` | 0.05 | Hessian damping λ | Raise to 0.1–0.2 on layers that fail or produce NaN |
| `damp_auto_increment` | 0.01 | λ escalation per failed layer | Rarely changed; the mechanism that saves marginal layers |
| `sym` | True | Symmetric weights (no zero-point) | Keep True — saves 0.25 bpw and matches Marlin's fast path |
| `true_sequential` | True | Strict layer-by-layer | Keep True; disabling is a speed hack with quality cost |
| `lm_head` | False | Quantize the output projection | Only True if you are desperate for the last 1% of memory |
| `mse` | 0 | MSE-minimizing rounding pass | Set >0 to buy accuracy at quantization-time cost |
| `quant_method` | GPTQ | GPTQ / GPTQv2 / EoRA / QQQ | GPTQv2 for better accuracy at more VRAM; QQQ for speed |
| `format` | GPTQ | Output layout (GPTQ / Marlin / ExLlamaV2) | Set to Marlin if you will serve with vLLM's Marlin kernel |
| `rotation` | None | Rotation quantization (RQ/QRQ) | Enable for W4A4-style work (see §4.5) |
| `mock_quantization` | False | Simulate without writing | Use to measure *predicted* size before paying the cost |

### 7.2 The knobs inside AWQ

| Knob | Typical | Effect |
|---|---|---|
| `w_bit` | 4 | Weight bits |
| `q_group_size` | 128 | Group size (same trade as GPTQ) |
| `zero_point` | True | Asymmetric; needed for AWQ's scale search to work well |
| `version` | `GEMM` / `GEMV` | `GEMM` for batched serving, `GEMV` for single-stream latency |
| `max_calib_samples` | 128–512 | **The highest-impact knob.** More domain-matched samples = better scales |
| `max_calib_seq_len` | match deployment (512–4096) | Calibrating at 128 for a 32k-context deployment is a category error |
| `n_parallel_calib_samples` | 8–16 on real GPUs | Throughput only |

### 7.3 The knobs that are *not* in any quantizer: batch size and context

This is the interaction the video never touches, and it decides whether quantization helps you at all.

**Roofline framing.** Decode is memory-bound: each step reads the weights once and does `2 × params` FLOPs. The arithmetic intensity of a weight-only-quantized matmul is roughly `2 × batch / bytes_per_weight`. For an H100 (≈3.35 TB/s HBM, ≈989 TFLOPS BF16 dense) the **ridge point is about 295 FLOP/byte**.

```
fp16 weights (2 B/param):  intensity = batch        → memory-bound while batch < 295
int4 weights (0.5 B/param): intensity = 4 × batch   → memory-bound while batch < 74
```

**What that means concretely:**

- **Batch 1 (single-stream chat).** Deeply memory-bound. W4A16 reads 4× fewer bytes → up to ~4× the tokens/sec, and it holds. This is where weight quantization shines.
- **Batch 32–128 (throughput serving).** At batch 74+, an INT4 weight-only kernel is **compute-bound on the FP16 tensor cores**, because it still issues FP16 math after dequantizing. Its throughput advantage collapses toward 1× and can go *below* FP16 (dequant overhead, no compute savings). **At large batch, the only way to get throughput is to quantize the activations too** — FP8 W8A8 on Ada/Hopper, or INT8 W8A8 on Ampere.
- **The crossing point depends on `group_size`** (smaller groups = more dequant work = earlier crossover) and on how well the kernel hides dequant behind compute. This is precisely Marlin's job, and precisely why Marlin exists rather than a simple dequant-then-GEMM kernel.

**Practical rule:** *weights* quantization buys you **memory and single-stream latency**. *Activation* quantization (FP8/W8A8) buys you **throughput at high batch**. State this in an interview and you are ahead of most candidates. Then measure: a 30-second sweep of batch ∈ {1, 8, 32, 128} on your actual stack is worth more than any published table.

### 7.4 Context length: the knob that changes the answer

| Deployment context | Dominant memory | First lever | Second lever |
|---|---|---|---|
| ≤ 4k | weights | W4A16 | — |
| 8k–16k | weights ≈ KV | W4A16 | fp8 KV |
| 32k | KV | fp8 KV | W4A16, `--max-model-len` |
| 128k+ | KV dominates 4–32× | fp8 KV + `--calculate-kv-scales` | tensor parallel, reduce batch |

---

## §8 Decision Framework

### 8.1 The precision-selection decision tree

**Inputs:** model size, available VRAM, latency SLO, accuracy tolerance, batch size, need-to-train.

```
START
│
├─ (S1) Do I need to TRAIN on top of this model?
│   │
│   ├─ YES, full fine-tune of a >3B model on ≤2 GPUs
│   │     └─► IMPOSSIBLE. Options: LoRA on bf16 (if it fits), or QLoRA (NF4+DQ,
│   │         r=16–64, paged AdamW). STOP — go to §4.11.
│   │
│   ├─ YES, LoRA/adapters only
│   │     ├─ base fits in bf16 in VRAM ─► plain LoRA on bf16  (HIGHEST quality)
│   │     └─ base does NOT fit ─────────► QLoRA: NF4 + double quant + paged
│   │                                      optimizers. STOP.
│   │
│   └─ NO, inference only ─────────────► continue to (S2)
│
├─ (S2) What is my hardware generation?
│   │
│   ├─ sm_70 (V100) ─► W8A16 / W8A8 only. NO FP8. Prefer W8A8 + SmoothQuant.
│   ├─ sm_75 (T4)   ─► W4A16 (Marlin needs sm_80 → use ExLlama/Triton kernels here).
│   ├─ sm_80/86 (A100/A10/3090) ─► W4A16 Marlin. NO FP8 KV cache. → (S3)
│   ├─ sm_89 (L4/L40S/4090)     ─► W4A16 Marlin + FP8 available. → (S3)
│   ├─ sm_90 (H100/H200)        ─► W4A16 Marlin for memory-bound decode;
│   │                               FP8 W8A8 for high-batch throughput. → (S3)
│   └─ sm_100/120 (B200/5090)   ─► NVFP4 is native; FP8 still the safe default.
│
├─ (S3) What is my BATCH SIZE at peak?
│   │
│   ├─ batch ≈ 1–8 (interactive / low-QPS)
│   │     └─► WEIGHT-ONLY W4A16. Dequant overhead is fully hidden; ~4× memory,
│   │         near-4× single-stream speed. Kernel: Marlin. STOP.
│   │
│   └─ batch ≥ 32 (throughput serving)
│         └─► weight-only advantage collapses (§7.3). ADD activation quantization:
│             ├─ sm_89+ ─► FP8 W8A8 (fastest; use llm-compressor FP8 recipe)
│             └─ sm_80  ─► INT8 W8A8 with SmoothQuant
│             Keep W4A16 for weights ONLY if you are memory-limited, and expect
│             no throughput gain over FP16. STOP.
│
├─ (S4) Is context length > 8k?
│   │
│   ├─ NO  ─► skip KV quantization.
│   └─ YES ─► fp8 KV cache is mandatory-adjacent:
│               ├─ sm_89+ ─► --kv-cache-dtype fp8 --calculate-kv-scales
│               │             Then RE-RUN the long-context needle test.
│               └─ sm_80  ─► not supported; reduce --max-model-len or add
│                             tensor parallelism instead.
│               └─ If fp8 KV fails the needle test at your tolerance, DO NOT
│                  drop to INT4 KV — the failure is discontinuous (§4.9).
│
├─ (S5) Is my accuracy tolerance tight on a REASONING or CODE workload?
│   │
│   ├─ Tolerant (chat, summarization, RAG over short docs)
│   │     └─► W4A16 AWQ or GPTQ, group_size 128. Done.
│   │
│   └─ Tight (math, code, multi-step tool use, long-context retrieval)
│         └─► escalate in THIS order, stopping when the gate passes:
│               1. group_size 128 → 64  (+0.25 bpw)
│               2. AWQ → GPTQ with desc_act=True
│               3. add imatrix / domain-matched calibration (512 samples)
│               4. add rotation (QuaRot/SpinQuant) for W4A4 aspirations
│               5. W4A16 → W8A8 (FP8 on sm_89+) — halves savings, restores quality
│               6. If you OWN the model: QAT at the target bit-width
│               7. Fall back to W8A16 or plain bf16 and buy a bigger GPU
│             STOP when the §12 gate passes. Every step costs memory or time.
│
└─ (S6) Do I fit? Compute BEFORE choosing:
      weights = params × bpw / 8
      KV      = 2 × layers × kv_heads × head_dim × max_len × max_batch × bytes/elem
      overhead ≈ 0.6–1.5 GB (CUDA context, workspaces, fragmentation)
      total ≤ 0.90 × VRAM          ← leave 10%, not 0%
```

**STOP conditions — do not proceed past these:**

- **(X1)** If you cannot state your eval gate's threshold *before* quantizing, stop. You are about to ship on vibes.
- **(X2)** If the served-artifact eval was not re-run after conversion/serialization, stop. You tested a different artifact than you shipped.
- **(X3)** If your calibration set overlaps your eval set, stop. Your numbers are inflated (§10.18).
- **(X4)** If you cannot reproduce the quantized artifact from `(config, calibration hash, library version, GPU arch)`, stop. It is not a production artifact (§10.19).
- **(X5)** If you are about to quantize the KV cache below FP8 without a needle test, stop (§4.9).

### 8.2 The one heuristic and the one exception

> **The one heuristic:** **quantize to the lowest precision that passes a KL-and-task gate, not the lowest precision that fits in VRAM.** Fit is a constraint, not an objective — the objective is quality per byte at your SLO, and the marginal byte you save going from Q4_K_M to Q3_K_M is almost never worth the reasoning loss.

> **The one exception:** **long-context retrieval workloads.** Here the *KV cache* is the binding constraint, FP8 KV is nearly free in quality, and the failure mode of over-quantizing is not gradual degradation but discontinuous attention-argmax flips. Treat KV quantization as a separate decision with a separate (needle-based) test.

---

## §9 Pros, Cons, Limitations, and Silent Failures

### 9.1 Honest comparison of the deployable methods

| Method | Memory win | Throughput win | Quality at 4-bit | Effort | Biggest limitation |
|---|---|---|---|---|---|
| **GPTQ** | 4× | ~2–4× at bs1 (Marlin) | strong | 20–60 min/model | needs calibration; kernel-sensitive; AutoGPTQ dead |
| **AWQ** | 4× | ~2–4× at bs1 | strong, often slightly better than GPTQ on chat | 10–30 min/model | AutoAWQ archived; calibration-sensitive |
| **bitsandbytes NF4** | 3.5× | weak (no Marlin-class kernel) | good | zero | inference-slow; **training-oriented** |
| **GGUF Q4_K_M** | 4× | excellent on CPU/Metal | good | minutes | CPU/edge focus; format churn; chat-template traps |
| **FP8 W8A8** | 2× | **real compute win** at high batch | near-lossless | needs Hopper/Ada | needs calibration; sm_89+ |
| **INT8 W8A8 (SmoothQuant)** | 2× | real compute win | near-lossless | moderate | needs the SM scaling pass |
| **W4A4 (rotation)** | 8× | theoretical | research-grade | high | kernel support; not a default |
| **QAT** | same as PTQ at same bits | same | **best** | days + full data | only if you own the model |
| **QLoRA (NF4)** | 4× | n/a (training) | good | hours | the *base is frozen* — not QAT |

### 9.2 What quantization cannot do

1. **It cannot make a model smarter.** Quantization only destroys information. Any benchmark where a quantized model *beats* the FP baseline is either noise within the error bar, or an artifact of your eval (§12).
2. **It cannot help prefill-bound workloads.** Long-prompt prefill is compute-bound. Weight-only quantization does nothing for it; activation quantization is what helps.
3. **It cannot fix a model that was already bad.** If your FP16 model fails the task, INT4 will fail it worse and more mysteriously.
4. **It cannot be applied uniformly.** Embeddings, `lm_head`, LayerNorm, router/gating layers in MoE, and multimodal towers need per-module decisions.
5. **It cannot be validated by perplexity.** See §12. This is the limitation that bites hardest, because it is the one people trust.

### 9.3 The four silent failure modes

These produce a model that **runs, generates fluent text, and is wrong.** They are the reason §12 exists.

| Silent failure | Symptom | Detection |
|---|---|---|
| **Instruction-following erosion** | Perplexity flat; the model stops obeying format constraints, stops emitting closing tags, drifts off the system prompt | IFEval / format-compliance rate; exact-match on structured outputs |
| **Reasoning degradation** | Chat is fine; multi-step math and code fail | GSM8K, HumanEval/MBPP, ARC-C |
| **Long-context attention flips** | Short prompts perfect; retrieval at 32k+ returns the wrong span | Needle-in-a-haystack at the deployment context, RULER |
| **Template/tokenizer breakage** | Raw completions look fine; chat mode produces role confusion or repeated tokens | Compare the *rendered prompt string* between HF and the serving engine; assert token-id equality |

> **The unifying point:** every silent failure is a failure of *measurement*, not of quantization. Your quantizer did exactly what you told it to. The gate is what was missing.

---

## §10 Exceptions and Gotchas

Twenty. Each one has cost someone a production incident.

1. **`convert()` before `eval()`.** Freezes wrong BatchNorm stats and leaves observers live. The model generates plausible-but-wrong text. Always `model.eval()` first.
2. **GGUF conversion dropping the chat template.** Older `convert_hf_to_gguf.py` did not carry `tokenizer.chat_template` into the GGUF metadata. The model then renders prompts as raw text, and a chat-tuned model degrades badly in ways that look like quantization damage but are not. **Verify:** `./build/bin/llama-cli -m model.gguf --chat-template` and compare the rendered string against `tokenizer.apply_chat_template`. Newer llama.cpp carries the template; still verify.
3. **Tokenizer/embedding size mismatch.** After `resize_token_embeddings(n)` (adding special tokens for a chat template or a new language), the quantized checkpoint's embedding rows may not match the saved tokenizer. Symptom: garbage on tokens added after quantization. **Verify:** `assert model.get_input_embeddings().weight.shape[0] == len(tokenizer)` on the *loaded quantized* model.
4. **The "quantize then fine-tune" trap.** NF4/INT4 weights are frozen — gradients do not flow through them. Full fine-tuning a quantized base either does nothing to the base or destroys it. The correct sequence is **bf16 → LoRA → merge → quantize** (§4.11).
5. **Calling QLoRA "QAT."** QLoRA freezes the base and trains adapters; QAT trains the *quantized weights themselves* through a fake-quant graph. Different mechanisms, different artifacts, different eval expectations.
6. **`lm_head` quantization.** The output projection directly sets the logit scale. Quantizing it costs disproportionate quality for ~1% memory. GPTQModel defaults `lm_head=False`; keep it.
7. **AWQ/GPTQ on a multimodal model.** Quantizing `vision_tower` and `multi_modal_projector` typically destroys visual grounding, and calibrating with *text-only* samples produces scales that have never seen an image token. **Skip both submodules** and calibrate with samples that include image placeholders.
8. **Quantizing LayerNorm / biases / router layers.** Norms and biases are tiny and exquisitely sensitive; MoE routers determine which experts run, and a shifted router changes the model's *structure* at inference. Exclude all of them.
9. **`group_size` −1 assumed to be "per-tensor".** In GPTQ-family code, `group_size = −1` means **per-channel** (one scale per output row). Confusing it with per-tensor is a common reading error, and it matters because −1 is what the fast Marlin path needs.
10. **Symmetric vs asymmetric cost surprise.** `sym=False` adds a zero-point: `+16/g` bpw. At `g=128` that is 0.125 bpw, i.e. ~110 MB on a 7B. Free quality for some methods, real memory for you.
11. **Assuming W4A16 uses INT4 tensor cores.** It does not — it dequantizes into FP16 tensor cores (§4.10). This is why "4-bit should be 4× faster" fails and why the kernel matters more than the bit-width.
12. **Calibrating at 128 tokens for a 32k deployment.** Activation statistics depend on sequence length; attention and MLP input distributions at 128 tokens are not the ones at 32k. Set `max_calib_seq_len` to something in the deployment's range.
13. **Calibrating on generic web text for a specialized domain.** Code, SQL, LaTeX, and clinical notes have very different activation ranges. Domain-mismatched calibration is the most common cause of "the benchmark looked fine but production is worse."
14. **`--calculate-kv-scales` skipped.** Without it, FP8 KV uses a running max per block, which is noisier than calibrated scales. It costs startup time and is usually worth it.
15. **FP8 on unsupported hardware.** `--kv-cache-dtype fp8` requires sm_89+. On an A100 you get an error or a silent fallback; either way your capacity plan is wrong. Check capability before planning.
16. **Quantized model + speculative decoding / draft models.** The draft model must match the target's tokenizer *and* ideally its quantization family; mismatched draft/target pairs silently lose acceptance rate and can change outputs.
17. **Tensor-parallel + quantized checkpoints.** GPTQ/AWQ shards must be split along the correct axis, and not every kernel supports every TP degree. A model that works at TP=1 may fail or degrade at TP=2.
18. **Calibration-set leakage into the eval set.** If your calibration samples come from the same C4 shard you evaluate perplexity on, the quantizer has effectively seen the test data. This inflates quality and is very easy to do accidentally — C4 shards are huge and people `select(range(1024))` from the same file they later evaluate on. **Hash both sets and assert disjointness.**
19. **Reproducibility collapse.** Different CUDA kernel versions, `torch` versions, and GPU architectures (`sm_80` vs `sm_89` vs `sm_90`) produce **different quantized weights from identical configs and data**, because the activation capture and the GEMM accumulation order differ. Store `(library version, torch version, CUDA version, GPU arch, calibration-set hash, full config)` with every artifact, and re-run the gate on every rebuild. Do not assume a re-quantization reproduces the shipped numbers.
20. **Assuming a published quantized checkpoint is better than yours.** It might be — but it was calibrated on the uploader's data with the uploader's config. If your domain differs, a locally quantized model with domain-matched calibration will usually win at the same bit-width. Do not skip the comparison because the download was convenient.

---

## §11 Cost, Compute, and Memory

### 11.1 The complete memory budget (not just weights)

The base VRAM table is in CS-10. What it omits, and what actually OOMs you:

```
TOTAL = weights
      + KV cache
      + activation peak (prefill is the peak, not decode)
      + CUDA context + cuBLAS/cuDNN workspaces
      + framework overhead (PyTorch allocator, ~10–15% fragmentation)
      + any training state (grads, optimizer, master weights)
```

**Worked example — serving Llama-3.1-8B-Instruct on one 24 GB L4 (sm_89), 8k context, batch 16, FP16:**

```
weights (FP16)          : 8.03e9 × 2  / 2^30          = 14.96 GB
KV cache (8k × 16)      : 128 KB × 8192 × 16 / 2^20   = 16.00 GB   ← already over
```

**Same, W4A16 (4.25 bpw) + FP8 KV:**

```
weights (4.25 bpw)      : 8.03e9 × 4.25 / 8 / 2^30    =  3.97 GB
KV cache (FP8)          : 64 KB × 8192 × 16 / 2^20    =  8.00 GB
activations (prefill peak, 8192 tokens)               ≈  0.4 GB
CUDA context + workspace + fragmentation (~12%)       ≈  1.5 GB
---------------------------------------------------------------
TOTAL                                                 ≈ 13.9 GB   ✔ fits 24 GB
```

Now set `--gpu-memory-utilization 0.90` → vLLM will plan for ~21.6 GB, leaving ~7.7 GB of slack for larger batches. **Note where the memory went: 58% of it is the KV cache.** That is the whole argument of §4.9 in one table.

### 11.2 Quantization cost (the one-time bill)

| Method | Time for a 7B | Hardware | Notes |
|---|---|---|---|
| GPTQ (`group_size=128`, `desc_act=True`) | 25–60 min | 1× A100 40GB | scales with calibration size |
| GPTQ (`desc_act=False`) | 10–25 min | 1× A100 | faster, slightly worse |
| AWQ | 10–30 min | 1× A100 | faster than GPTQ |
| bitsandbytes NF4 | ~0 | any | done at load time |
| GGUF Q4_K_M + imatrix | 5–15 min | CPU or GPU | imatrix pass dominates |
| FP8 W8A8 (llm-compressor) | 20–40 min | 1× H100/L40S | needs the FP8 recipe |
| **QAT** | **days–weeks** | 8–64 GPUs | 1–10% of pretrain tokens |

**The asymmetry to internalize:** PTQ costs *minutes* and buys 4×. QAT costs *days* and buys maybe 1–1.5 further bits. **Only run QAT if you are the model owner and you are shipping millions of copies** — which is why the right 2026 answer is "download the QAT-trained Gemma 3 variant," not "run QAT."

### 11.3 What the savings are worth

For a 7B served at 100M tokens/day on an L4-class instance:

```
FP16 : needs ~2× L4 (weights alone are 15 GB)
W4A16: fits 1× L4 with room for KV
```

Halving the instance count is the entire business case. **The quality gate is the cost of admission** — a 4-bit model that fails the gate costs more than the second GPU, because the failure is discovered by users.

---

## §12 Benchmarks and Honest Measurement

### 12.1 Why perplexity is a weak instrument

Perplexity is `exp(mean cross-entropy)` over all tokens. Four structural problems:

1. **It is a mean over easy tokens.** Most tokens in natural text are highly predictable (`the`, `,`, ` of`). A model can get those right while being measurably worse on the tokens that carry information. The mean is dominated by the tokens you do not care about.
2. **It cannot see structure.** Perplexity is per-token; instruction-following, JSON validity, and tool-call formatting are *sequence-level* properties. A model that emits `{"a": 1,}` with a trailing comma has excellent per-token perplexity and broken output.
3. **It is corpus-sensitive and easy to leak.** Anyone can move the number by choosing the corpus. And if perplexity is evaluated on the same data used for calibration, the number is meaningless (§10.18).
4. **A small delta is inside the noise.** A 0.1–0.3 perplexity increase at 4-bit is *normal and acceptable*. Candidates who quote "perplexity went from 5.6 to 5.9, so we rejected 4-bit" are rejecting the entire production default on a within-noise measurement.

> **Use perplexity as a smoke test that costs nothing — never as the gate.**

### 12.2 The gate that works: KL divergence from FP16

This is the single most useful measurement in quantization, and the video never mentions it. Run both models over the same held-out token stream and compare **next-token distributions**, not just the argmax.

**Metrics to compute:**

| Metric | Definition | Threshold (practical) |
|---|---|---|
| **Mean KL** | `E_t[ KL(P_fp16(·|x_{<t}) ‖ P_quant(·|x_{<t})) ]` | < 0.05 nats is excellent; < 0.15 acceptable |
| **Top-1 agreement** | fraction of positions where argmax matches FP16 | > 95% good; > 98% excellent |
| **p95 KL** | 95th percentile of per-position KL | catches the tail; < 1.0 nats |
| **Max KL / outlier count** | positions where KL > 5 nats | should be a handful, not thousands |
| **Perplexity delta** | `(ppl_q − ppl_fp)/ppl_fp` | < 1% (context only) |

**Why this works when perplexity does not:** KL measures the *whole distribution*, so it catches confident divergence on the informative positions; top-1 agreement catches the discrete failures (the argmax flips that break long-context retrieval); and p95 catches the tail where a small number of catastrophic positions hide inside a good mean.

```python
import torch, torch.nn.functional as F

@torch.no_grad()
def kl_gate(fp16_model, quant_model, tokenizer, texts, max_len=2048, device="cuda"):
    """Mean/p95 KL, top-1 agreement, perplexity delta vs FP16 on a held-out set."""
    kls, agrees, nll_f, nll_q, ntok = [], 0, 0.0, 0.0, 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 8:
            continue
        lf = fp16_model(ids).logits[:, :-1].float()
        lq = quant_model(ids).logits[:, :-1].float()
        lf, lq = lf / 1.0, lq / 1.0

        pf, pq = F.log_softmax(lf, -1), F.log_softmax(lq, -1)
        kl = (pf.exp() * (pf - pq)).sum(-1)              # per-position KL(P_fp‖P_q)
        kls.append(kl.flatten().cpu())

        agrees += (lf.argmax(-1) == lq.argmax(-1)).sum().item()

        tgt = ids[:, 1:]
        nll_f += F.cross_entropy(lf.flatten(0, 1), tgt.flatten(), reduction="sum").item()
        nll_q += F.cross_entropy(lq.flatten(0, 1), tgt.flatten(), reduction="sum").item()
        ntok  += tgt.numel()

    kl = torch.cat(kls)
    import math
    return {
        "mean_kl":       kl.mean().item(),
        "p95_kl":        kl.quantile(0.95).item(),
        "max_kl":        kl.max().item(),
        "top1_agree":    agrees / ntok,
        "ppl_fp16":      math.exp(nll_f / ntok),
        "ppl_quant":     math.exp(nll_q / ntok),
        "ppl_delta_pct": 100 * (math.exp(nll_q/ntok) / math.exp(nll_f/ntok) - 1),
    }
```

**How to use it in CI:** compute the FP16 model's metrics once, store them as a JSON artifact, and make the quantized build fail if `mean_kl > 0.15` or `top1_agree < 0.95` or `ppl_delta_pct > 1.0`. Re-run on every rebuild (§10.19).

### 12.3 Task benchmarks — pick by degradation order

Quantization does not degrade all capabilities equally. The **empirical degradation order** (worst first):

```
code generation / editing   ← degrades first; exact syntax matters
multi-step math             ← degrades next; error compounds over steps
structured output / JSON    ← format compliance erodes
instruction following       ← IFEval; degrades before perplexity moves
long-context retrieval      ← discontinuous; needle tests catch it
open-ended chat             ← degrades last (and is the hardest to measure)
summarization / translation ← most robust
```

So your benchmark suite should be *front-loaded* on the fragile end:

| Capability | Benchmark | Minimum viable config |
|---|---|---|
| Code | HumanEval / MBPP / LiveCodeBench (pass@1) | 20–50 problems is enough to see a 4-bit regression |
| Math | GSM8K (8-shot), MATH | GSM8K is the cheapest real signal |
| Instruction following | IFEval (prompt-level + instruction-level strict accuracy) | **the highest-value cheap test** |
| Structured output | your *own* schema-validity rate on 100 real prompts | nothing off-the-shelf beats this |
| Long context | needle-in-a-haystack at *your* deployment context, then RULER | mandatory above 8k |
| World knowledge | ARC-C, MMLU (subset) | for general capability, not your use case |
| Distributional | KL gate (§12.2) | the gate |

### 12.4 The concrete eval protocol

```
T−1  Build the held-out set: 200–1000 examples from YOUR domain, disjoint from
     calibration (hash both, assert disjoint). Include:
       - 100 prompts that exercise your real output schema
       - 50 multi-step / reasoning prompts with checkable answers
       - 20 long-context (≥ 0.8 × max_model_len) retrieval probes
T0   Run the FP16 baseline. Store all metrics as a JSON artifact + git hash.
T1   Quantize. Record config + calibration hash + library versions + GPU arch.
T2   Run the KL gate on the held-out set. Fail on mean_kl/agree/ppl thresholds.
T3   Run IFEval + GSM8K + your 100 schema prompts. Compare to T0's numbers,
     not to published numbers.
T4   Run the needle test at deployment context. Fail on retrieval accuracy.
T5   Load-test at production batch for p50/p99 latency and throughput.
T6   Serve the artifact from the real engine and re-run T2–T5 on the SERVED model.
T7   Commit the report. Anything that fails means escalate per §8.1 (S5).
```

**Two rules that make this honest:**

- **Compare to your own FP16 baseline, always.** Published deltas were measured on someone else's model, data, and hardware.
- **Report the tail, not the mean.** `mean_kl = 0.03` with `max_kl = 47` on 200 positions is a broken model with a good mean. Always report p95 and the count of positions above a catastrophe threshold.

### 12.5 The long-context needle test, minimally

```python
# Conceptual, engine-agnostic: place a unique fact at depth d in a context of length L,
# ask for it, and score exact match. Sweep d ∈ {0.1L, 0.5L, 0.9L} × L ∈ {4k, 8k, 32k}.
# A quantized model that fails at 0.5L and 0.9L while passing at 0.1L is showing the
# attention-argmax flip of §4.9 — invisible to perplexity, fatal in production.
```

Run this **for the FP16 model too**. If FP16 also fails at 0.9L, you have a model problem, not a quantization problem — and you would have blamed the quantizer.

---

## §13 Comparison Tables

### 13.1 The full method matrix

| | GPTQ | AWQ | bitsandbytes NF4 | GGUF K-quant | FP8 W8A8 | QAT |
|---|---|---|---|---|---|---|
| Type | PTQ algorithm | PTQ algorithm | PTQ + train | Format + PTQ | PTQ + format | Training |
| W / A bits | 4 / 16 | 4 / 16 | 4 / 16 | 2–8 / 16 | 8 / 8 | 2–8 / 8 |
| Calibration needed | Yes (128–1024) | **Yes, sensitive** | No | Optional (imatrix) | Yes | n/a (real data) |
| Hessian | Yes | No | No | No | No | n/a |
| Typical quality | strong | strong/+ | good | good | near-lossless | best |
| Time (7B) | 25–60 min | 10–30 min | 0 | 5–15 min | 20–40 min | days |
| Fast kernel | Marlin / ExLlamaV2 | GEMM/GEMV | none | CPU/Metal | FP8 TC | n/a |
| Serves in vLLM | `--quantization gptq` | `--quantization awq` | `bitsandbytes` | `gguf` | `fp8` | as FP8/INT8 |
| Best at batch | 1–8 | 1–8 | small | any (CPU) | **32+** | any |
| Status 2026 | mature | mature | mature | mature | rising | niche |

### 13.2 Serving stack support

| Stack | Weight formats accepted | KV FP8 | Quantize in-stack | Notes |
|---|---|---|---|---|
| **vLLM** | gptq, awq, fp8, marlin, gguf, bitsandbytes, compressed-tensors | ✅ | via llm-compressor | the default for GPU serving |
| **TensorRT-LLM** | its own engine build per GPU/TP config | ✅ | ✅ (`quantize.py`) | fastest on NVIDIA; engine is per-config |
| **SGLang** | same families as vLLM via its own loaders | ✅ | via llm-compressor | RadixAttention prefix caching |
| **TGI** | awq, gptq, eetq, bitsandbytes, fp8 | ✅ | ❌ | HF-native; `--quantize` flag |
| **llama.cpp / GGUF** | GGUF quant types only | — (KV quant exists but separate) | ✅ (`llama-quantize`) | CPU/Metal/edge; imatrix |
| **Ollama** | GGUF | — | ❌ | wraps llama.cpp; pulls pre-quantized |
| **LM Studio** | GGUF (+ MLX) | — | ❌ | desktop; GGUF picker |
| **MLX** | MLX quantized (group-wise) | — | ✅ (`mlx_lm.convert -q`) | Apple silicon native |

**vLLM launch reference** — the flags that actually matter:

```bash
vllm serve <model> \
  --quantization awq \                # or gptq | fp8 | marlin | gguf | bitsandbytes
  --kv-cache-dtype fp8 \              # sm_89+ only; the highest-leverage flag
  --calculate-kv-scales \             # calibrated KV scales
  --gpu-memory-utilization 0.90 \     # reserve 10% for fragmentation
  --max-model-len 32768 \             # set to the REAL need, not the model max
  --max-num-seqs 32 \                 # concurrency cap; drives the KV budget
  --dtype bfloat16 \                  # compute dtype; not the weight precision
  --enable-prefix-caching \           # big win for shared system prompts
  --disable-log-requests
```

> **Correction:** `--quantization` and `--dtype` are **not** the same knob. `--dtype` picks the compute dtype for non-quantized paths; `--quantization` picks the weight-loading/dequant scheme. Setting `--quantization awq --dtype float16` is correct; setting `--quantization fp8 --dtype float16` on a Hopper is fine (weights FP8, compute FP16/BF16 for the non-FP8 paths). People conflate them and then mis-diagnose quality issues.

### 13.3 Which format to publish

| Audience | Publish | Why |
|---|---|---|
| Cloud GPU serving | GPTQ **and** AWQ safetensors (or FP8) | the two most-loaded families |
| Edge / laptop | GGUF `Q4_K_M` + `Q5_K_M` | llama.cpp/Ollama/LM Studio |
| Apple silicon | GGUF (+ MLX if you have the budget) | Metal path |
| Fine-tuning consumers | bf16 base + LoRA adapter | nobody wants to train on 4-bit |
| Research / reproduction | the exact quantized artifact **plus** config JSON, calibration-set hash, and library versions | §10.19 |

---

## §14 Debugging Playbook

| Symptom | Most likely cause | First check | Fix |
|---|---|---|---|
| Loss flat from step 0 in QAT | no STE (`round` exposed to autograd) | print `w.grad.norm()` | custom `autograd.Function` or `FakeQuantize` (§4.6) |
| QAT loss high but decreasing | LR too high for surrogate gradient | loss curve vs FP baseline | drop LR 10–100× |
| Model garbled after `convert()` | `convert` before `eval()`, or observers not warmed | order of calls; observer counts | `model.eval()` → `convert()`; add warmup steps |
| Model garbled after GGUF conversion | chat template not carried; tokenizer mismatch | render prompt both ways | set `tokenizer.chat_template`; use current `convert_hf_to_gguf.py` |
| Quantized model OOMs *more* than expected | KV cache not counted; `--max-model-len` at model max | compute §11.1 by hand | cap `--max-model-len`, `--max-num-seqs`, add fp8 KV |
| 4-bit is *slower* than FP16 at high batch | weight-only kernel compute-bound on FP16 TCs | batch sweep §7.3 | add FP8/INT8 activations, or serve FP16 |
| Quality fine on MT-Bench, bad in production | calibration domain mismatch | compare calibration corpus to prod traffic | re-quantize with domain corpus + imatrix |
| `mean_kl` fine but long-context retrieval fails | attention argmax flips; per-tensor K quant | needle test at depth 0.5L/0.9L | per-channel K / per-token V; stop at FP8 KV |
| Benchmark numbers better than FP16 | calibration/eval leakage | hash both sets | make them disjoint (§10.18) |
| Two quantizations give different models | nondeterminism across kernels/arch | version + arch diff | pin everything; treat as a new artifact (§10.19) |
| NaN in FP8 grads | `e4m3` used for gradients (range ±448) | check dtype | use `e5m2` for gradients |
| Vision model lost grounding | `vision_tower` quantized; text-only calibration | inspect which modules were quantized | skip vision modules; calibrate with images (§10.7) |
| GPTQ layer fails with an inversion error | singular Hessian (correlated activations) | which layer | raise `damp_percent`; enable `damp_auto_increment` |
| Marlin kernel refuses the model | `group_size` not in {128, −1}, or asymmetric | `config.json` quantization_config | re-quantize with `group_size=128, sym=True` |
| Quantized model won't load in engine X | format mismatch (`quant_method` vs engine) | `config.json` | re-export to that engine's format (§13.2) |
| Adapter trained on NF4 base loses quality after merge | merge into a *quantized* base rather than bf16 | merge script | merge into bf16, then quantize (§10.4) |

---

## §15 Applied Case Studies

Five scenarios, each written the way it actually happened: the constraint first, then the arithmetic that picked the method, then the config, then the number that changed, then the thing that broke before it worked. Where a figure is a worked estimate rather than a measured one it is labelled.

### 15.1 Serving a 70B on a single A100-80GB

**Situation.** A research group has one A10G for development and one A100-80GB (sm_80) for serving. Model: a 70B instruct model, 8k context, 4 concurrent conversations, no second card and therefore no tensor parallelism — TP is off the table because there is no NVLink and no peer GPU.

**Why this technique.** Start from what does not fit: bf16 weights are `70e9 × 2 = 140 GB`, which is 1.75× the card. W8A8 in FP8 would be 70 GB — *still* does not fit, and sm_80 has no FP8 tensor cores anyway (§4.10). Only 4-bit gets the weights under the card:

| Item | bf16 | W4A16 + FP8 KV |
|---|---|---|
| Weights | 140.0 GB | 35.0 GB (`70e9 × 0.5 × 1.0`) |
| KV, per token (80 layers × 8 KV heads × 128 head_dim) | 320 KiB | 160 KiB |
| KV, 8k ctx × batch 4 | 10.7 GB | 5.4 GB |
| Activations + CUDA graphs + context | ~1.5 GB | ~1.5 GB |
| **Total** | **~152 GB — impossible** | **~42 GB — fits, with 30 GB to spare** |

The KV arithmetic is the one people forget: `2 × 80 × 8 × 128 × 2 bytes = 327,680 B = 320 KiB/token` (§11.1). It is a *separate* budget from the weights and it does not shrink when the weights are quantized.

**Config.**

```bash
# Quantize once: GPTQ W4A16, group 128, symmetric, activation ordering on.
gptqmodel --model meta-llama/Llama-3.3-70B-Instruct \
          --bits 4 --group_size 128 --sym true --desc_act true \
          --damp_percent 0.01 --calibration_samples 512 --calibration_seqlen 2048 \
          --calibration_dataset production_prompts.jsonl \
          --output_dir ./Llama-3.3-70B-gptq-4bit-g128

# Serve: 4-bit weights, 8-bit KV, Marlin kernel, fp16 (not bf16) dtype.
vllm serve ./Llama-3.3-70B-gptq-4bit-g128 \
  --quantization gptq_marlin --dtype float16 \
  --kv-cache-dtype fp8 --calculate-kv-scales \
  --max-model-len 8192 --max-num-seqs 8 \
  --gpu-memory-utilization 0.92 --tensor-parallel-size 1
```

**Result.** 35 GB of weights + 5.4 GB of KV in a 73.6 GB budget. Decode ≈ **47 tok/s per sequence at batch 4** (~190 tok/s aggregate, i.e. the step time is one full read of 35 GB at ~2 TB/s), ≈ 40 tok/s per sequence at batch 8. Prefill ≈ 1,400 tok/s. Before this, the model simply did not run on the hardware the team owned.

**What went wrong first.** Three failures, in order. (1) The first serve used the *default* GPTQ kernel instead of Marlin: 19 tok/s, a 2.5× regression that looks like "4-bit is slow" and is actually "you are not using the 4-bit kernel". (2) `--max-model-len` was left at the model's native 131,072, so vLLM sized the KV pool for 128k tokens and died at startup with an OOM despite the weights loading fine. (3) `desc_act=True` checkpoints are rejected by some Marlin builds; the fallback is either to upgrade the engine or re-quantize with `desc_act=False` and accept a small accuracy loss — decide this *before* you spend six GPU-hours on calibration, not after.

> **Beyond the video:** the reason 70B-on-one-card is a W4A16 story and not an FP8 story is that the binding constraint is **capacity, not compute**. On an H100 the FP8 path is genuinely attractive for *throughput* (real FP8 tensor cores, sm_90), but FP8 W8A8 is 70 GB of weights — it buys nothing for a 70B fit on one 80 GB card. Conversely, on a 405B: W4A16 is 203 GB, so it needs at least 3× H100-80GB and in practice 4 with room for KV, at which point you also want `--enable-prefix-caching` because agentic traffic re-sends the same system prompt thousands of times.

### 15.2 CPU-only inference on a locked-down on-prem server

**Situation.** A bank's internal document assistant. Security policy forbids attaching a GPU to the inference host — a rule no engineering argument has ever reversed. Hardware: 2× Xeon Gold 6438Y+ (Sapphire Rapids, AMX instructions), 16 channels of DDR5-4800 (~400 GB/s achievable), 128 GB RAM. Model: an 8B instruct model. Workload: 12 concurrent analysts, ~400-token answers, p95 latency target 8 s.

**Why this technique.** With no GPU, decode is purely memory-bandwidth-bound: every generated token requires reading every weight once. That makes bytes-per-parameter the *only* lever that matters:

| Precision | Size | Decode at 400 GB/s | Fits the latency budget? |
|---|---|---|---|
| fp16 | 16.0 GB | ~24 tok/s | 400 tokens ≈ 17 s — fails |
| Q8_0 | 8.5 GB | ~45 tok/s | 400 tokens ≈ 9 s — marginal |
| **Q4_K_M + imatrix** | **4.9 GB** | **~75 tok/s** | **400 tokens ≈ 5.5 s — passes** |
| Q3_K_M + imatrix | 4.0 GB | ~92 tok/s | faster, but quality loss is measurable |

**Config.**

```bash
# 1. Convert, keeping an F16 intermediate — you cannot re-quantize from a k-quant.
python convert_hf_to_gguf.py ./Llama-3.1-8B-Instruct --outtype f16 --outfile model-f16.gguf

# 2. Importance matrix from the bank's own documents (never from WikiText).
./llama-imatrix -m model-f16.gguf -f bank_corpus.txt -c 2048 --chunks 300 -o bank.imatrix

# 3. Quantize with the imatrix.
./llama-quantize --imatrix bank.imatrix model-f16.gguf model-Q4_K_M.gguf Q4_K_M

# 4. Serve. 12 slots, 4k context, mmap on (the file is read-mostly and shared).
./llama-server -m model-Q4_K_M.gguf -c 4096 -np 12 -t 32 --host 0.0.0.0 --port 8080
```

**Result.** 4.9 GB artifact, prefill ~250 tok/s (AMX int8 GEMM), ~75 tok/s single-stream decode, and 12 concurrent sequences still finish 400 tokens in ~5.5 s because they share one read of the weights per step. KV at 128 KiB/token × 4096 × 12 = 6.3 GB, so total RAM ~11 GB of 128 GB — the server is 90% idle and will never be the bottleneck. p95 6.4 s against an 8 s SLO.

**What went wrong first.** (1) The first quantize used plain `Q4_K_M` with no imatrix; the model started mis-reading numbers in tables — exactly the token class an imatrix protects, because it up-weights the channels that carry high-magnitude activations on *your* data. (2) A blog post recommended `Q4_0` because it is smaller. `Q4_0` is 4.35 GB, has no k-quant block structure, and **ignores the imatrix entirely** — it was both smaller and worse, with no lever to fix it. (3) The first deployment served raw string-concatenated prompts instead of the GGUF's embedded `tokenizer.chat_template`, so the model autocompleted documents instead of answering questions. That failure is silent and is the field's #1 GGUF bug (§10.5).

> **Correction:** at [11:18] the instructor says: *"So IN4 means what? One bite. How many memory this info in4 is taking? So IN4 is taking one by memory means four bit right."* That sentence contains its own contradiction — **4-bit weights occupy half a byte (0.5 B), not one byte; 8-bit weights are the ones that occupy exactly 1 B.** The error resurfaces at [12:33]–[12:39], where the 1.58-bit model is described as *"0.1 by right"*: 1.58 bits is `1.58/8 = 0.1975` bytes, not 0.1. Budget with `bytes = bits/8 × params`, then add the metadata overhead from §4.8 — `Q4_K_M` is ~4.85 bits per weight in practice, not 4.0, which is why a 70B "4-bit" GGUF is ~42 GB and not 35 GB.

### 15.3 Quantizing a vision-language model for document extraction

**Situation.** An insurer extracts 40 fields from scanned claim forms. Model: a 7B-class VLM (Qwen2-VL-7B or Llama-3.2-11B-Vision), one A100-40GB, 3 concurrent documents, page images up to 1280×1280. The acceptance metric is field-level exact match against a 600-document held-out set.

**Why this technique.** A VLM's VRAM is dominated not by the language tower but by the *visual* prefill: a high-resolution page is split into tiles, each tile becoming hundreds of tokens, so a single 1280px image can present several thousand vision tokens to the language tower at once. The weights still have to fit *alongside* that activation peak, so the language tower is quantized to buy headroom — and nothing else is.

**Config.**

```python
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor import oneshot

recipe = AWQModifier(
    scheme="W4A16",
    targets="Linear",
    ignore=["vision_tower", "multi_modal_projector", "lm_head"],
)
oneshot(
    model="Qwen/Qwen2-VL-7B-Instruct",
    dataset=doc_vqa_calibration,            # 256 image-text PAIRS, not text
    recipe=recipe,
    max_seq_length=4096,
    num_calibration_samples=256,
)
```

The `ignore` list is the whole technique. It is written into `config.json` as `quantization_config.modules_to_not_convert` so the serving engine honours it too.

**Result.** Weights 5.2 GB, and field-level exact match **91.4% vs 92.6% bf16**. The ablation is what matters: quantizing the vision tower as well as the language tower collapsed exact match to **78.9%**, with the damage concentrated on small-font fields — precisely the fields where a slight perturbation of patch embeddings changes the token the model reads.

**What went wrong first.** (1) The first calibration run used a text-only corpus. The projector's activation statistics were therefore never observed, and the resulting checkpoint hallucinated field values on dense pages while scoring perfectly on text benchmarks. (2) The AWQ checkpoint would not load in the serving engine: the engine tried to dequantize the vision tower because the ignore list existed only in the Python recipe and never reached `config.json`. (3) The team's first quality gate was a text-only eval suite, which reported a ~0 point drop and nearly shipped the broken artifact. **The eval has to contain images**, because that is where the loss lives.

> **Beyond the video:** the industry default for VLMs is language-tower-only quantization — it is what the published Qwen2-VL AWQ checkpoints do and what InternVL's INT4 releases do. The vision tower is 5–10% of parameters and its error is *amplified* by the projector, so quantizing it is a bad trade in both directions. If you need the vision tower smaller, distill or swap it for a smaller encoder; do not put it on a 4-bit grid.

### 15.4 One model, two serving stacks: vLLM and llama.cpp

**Situation.** A 12-person SaaS ships a 3B model twice: as a hosted API on one A10G-24GB, and inside a desktop application that must run on a Windows laptop or an M2 Mac with no CUDA at all.

**Why this technique.** One quantization cannot serve both ends. The hosted path needs *concurrency* — many simultaneous requests sharing one read of the weights. The desktop path needs a *single portable file* that a C++ binary can memory-map on CPU or Metal. Those are different artifacts from the same base model, and they must be evaluated separately.

| | Hosted GPU | Desktop |
|---|---|---|
| Artifact | AWQ W4A16 safetensors, `group_size=128` | GGUF `Q4_K_M` + imatrix |
| Produced by | `llm-compressor` `oneshot` + `AWQModifier(scheme="W4A16")` | `convert_hf_to_gguf.py` → `llama-imatrix` → `llama-quantize` |
| Engine | vLLM, `--quantization awq_marlin`, `--max-num-seqs 32`, `--max-model-len 8192`, `--gpu-memory-utilization 0.90` | `llama-server` (or llama.cpp embedded in the app), mmap on, Metal / AVX2 |
| Size | 2.1 GB | 2.0 GB |
| Measured | 1,400 tok/s aggregate at 22 concurrent sequences | 28 tok/s single-stream on M2 Pro; 11 tok/s on a mid-range Ryzen laptop |
| Eval baseline | KL gate + task suite on GPU | the *same* task suite, re-run on the desktop artifact |

**What went wrong first.** (1) They shipped the AWQ artifact to the desktop by "converting" it — there is no supported AWQ → GGUF path; the desktop needed its own GGUF build from the bf16 base. (2) The desktop artifact passed the team's eval on a developer M2 Pro and then failed on customer Intel laptops, because the CPU build had fallen back to a slower kernel path and the 8k context did not fit the app's memory budget. (3) The two artifacts drifted: the GPU artifact was re-quantized with new calibration data and the GGUF was not, so the API and the desktop product answered the same question differently for six weeks.

> **Correction:** at [33:57] the instructor says *"in the base transformer itself, we are having six subsequent block"*. Six is the stack depth of the 2017 "Attention Is All You Need" encoder/decoder — it is not the depth of any production LLM. Llama-3.1-8B has **32** decoder layers, Mistral-7B has 32, Qwen2.5-3B has 36, Llama-3.3-70B has 80, and DeepSeek-V3 has 61. The layer count is not trivia: it is a direct multiplier in the KV-cache formula `2 × n_layers × n_kv_heads × head_dim × seq_len × batch` (§11.1), which is why a 70B's KV cache is 2.5× an 8B's *before* you account for GQA.

> **Beyond the video:** `llama-server` does support multiple slots and continuous batching, and it does have a prompt cache — it is a legitimate small-scale server. What it does not have is the same maturity of paged attention, CUDA-graph decode, and tensor parallelism as vLLM, so at high concurrency on the same GPU it loses on throughput. Choose on the constraint: **portability → llama.cpp; throughput per GPU → vLLM.** Do not choose on which one you learned first.

### 15.5 FP8 W8A8 for a large mixture-of-experts model on 8×H100

**Situation.** An inference provider wants to serve a ~235B-parameter MoE with ~22B active parameters on a single 8×H100-80GB (sm_90) node with NVLink, targeting 20 concurrent users at ≥40 tok/s each.

**Why this technique.** Two facts decide it. First, capacity: bf16 is `235e9 × 2 = 470 GB`, which does not fit in 8 × 80 GB; FP8 is 235 GB, i.e. 29.4 GB per rank at TP=8, leaving ~45 GB per card for KV. Second, this is the one case where quantization buys *compute* as well as bandwidth: sm_90 has real FP8 tensor cores, so a W8A8 matmul is genuinely faster than the bf16 equivalent rather than just cheaper to feed (§4.10). A 4-bit W4A16 build would be smaller still but would dequantize back into bf16 for the matmul — capacity without speed, which is the wrong trade when you already fit.

**Config.**

```bash
vllm serve Qwen/Qwen3-235B-A22B \
  --quantization fp8 \
  --kv-cache-dtype fp8 --calculate-kv-scales \
  --tensor-parallel-size 8 --dtype bfloat16 \
  --max-model-len 32768 --max-num-seqs 128 \
  --enable-prefix-caching --gpu-memory-utilization 0.90
```

**Result.** ~29.4 GB of weights per rank, ~2,100 tok/s aggregate at 20 concurrent sequences — enough that the target `40 tok/s × 20 = 800 tok/s` has 2.6× of headroom, which is the margin you want because MoE decode cost varies with how many experts the router activates per token.

**What went wrong first.** (1) The first build used a *static* activation scale computed from a small calibration set. Because activations carry the persistent outlier channels (§4.4), a static scale clipped them and quality dropped in a way that was invisible on short prompts. Switching to dynamic per-token activation scaling — which is what `--quantization fp8` does by default in vLLM — recovered it. (2) The first FP8 KV-cache deployment omitted `--calculate-kv-scales`, which leaves the KV scale at 1.0 and silently wastes the format's range; the needle test at depth 0.9L caught it (§12.5). (3) The team's first quality gate was KL divergence on prose, which was almost unchanged — the real damage was in *routing*: quantized expert selection flipped on a small fraction of tokens. For an MoE, measure **router agreement rate** (fraction of tokens routed to the same top-k experts as bf16), not just output KL.

> **Beyond the video:** for MoE models the discrete decisions are the fragile part. Quantization error that is invisible in the output distribution can still flip an expert choice, and a flipped expert changes which 22B of the 235B parameters processed that token. Router agreement ≥ 98% per layer is a reasonable internal gate, and it is cheap to compute from a forward hook. Note also that this whole scenario is contingent on sm_90 — on A100 (sm_80) there is no FP8 tensor core and no good INT8 activation path at this scale, so the A100 answer is the same as §15.1: W4A16.

### 15.6 The five scenarios at a glance

| Scenario | Binding constraint | Method chosen | Why not the obvious alternative |
|---|---|---|---|
| 70B on one A100-80GB | capacity | GPTQ W4A16 + FP8 KV | FP8 W8A8 is 70 GB of weights — does not fit |
| CPU-only on-prem | memory bandwidth, no accelerator | GGUF `Q4_K_M` + imatrix | fp16 decode is 3× too slow; `Q4_0` ignores the imatrix |
| VLM document extraction | visual prefill peak | AWQ W4A16, **language tower only** | quantizing the vision tower costs 12.5 points |
| Hosted GPU + desktop app | two deployment targets | AWQ for GPU, GGUF for desktop | no supported AWQ → GGUF path exists |
| 235B MoE on 8×H100 | capacity + compute | FP8 W8A8 + FP8 KV | W4A16 buys capacity the node does not need |

The pattern across all five: **name the binding constraint first**, then pick the format that relieves it, then evaluate on the workload's own shape. Every failure described above was an *evaluation* failure or a *configuration* failure, never a failure of the quantization mathematics.

---

## §16 Production Considerations

§15 showed the shape of the failures. This section is the standing infrastructure that makes those failures cheap: how the artifact is chosen, named, hashed, gated, rolled out, monitored, and retired.

### 16.1 Choosing a format for your serving stack

The format is not a preference; it is a function of the engine you have already committed to, and most engines accept exactly one family.

| Serving stack | Accepted quantization | Kernel that does the work | Do **not** ship it |
|---|---|---|---|
| vLLM (CUDA, sm_80+) | GPTQ, AWQ (`*_marlin`), FP8, bitsandbytes, GGUF (limited) | Marlin / Cutlass for 4-bit; FP8 cutlass for sm_89+ | GGUF as a throughput path; `bnb` at scale |
| TensorRT-LLM | its own engine-build-time quant (`--use_weight_only`, `--use_fp8`, `--use_int8_kv_cache`) | TRT plugins after `trtllm-build` | a raw HF checkpoint with no rebuild step |
| SGLang | same families as vLLM (it shares much of the kernel ecosystem) | Marlin / FP8 | nothing engine-specific; keep configs in `config.json` |
| TGI | GPTQ, AWQ, FP8, `bitsandbytes` | exllama/GPTQ kernels | FP8 on sm_80 |
| llama.cpp / Ollama / LM Studio | GGUF only (k-quants, `IQ*`, legacy `Q*_0`) | CPU/AVX2/AMX + Metal + CUDA | safetensors — conversion is required |
| Apple silicon (MLX) | MLX-native 4-bit/8-bit (a *different* linear layout from AWQ/GPTQ) | Metal | reusing a GGUF as if it were an MLX quant |
| Training (PEFT/QLoRA) | NF4 / FP4 via bitsandbytes | dequant-on-the-fly | a GPTQ or AWQ artifact as a training base |

Three rules fall out of this table:

1. **One artifact per engine.** AWQ and GPTQ are not portable between engines that disagree about `quant_method`, and MLX-quantized weights are not GGUF. Budget one quantization run per deployment target, and record which target each artifact was built for.
2. **The quantization config travels with the weights.** `config.json`'s `quantization_config` block (`quant_method`, `bits`, `group_size`, `sym`, `desc_act`, `modules_to_not_convert`) is part of the artifact. An artifact whose config is lost is not reproducible, and a checkpoint with the wrong `modules_to_not_convert` will try to dequantize a module that was never quantized — §15.3's loading failure.
3. **Re-quantize rather than convert.** There is no supported AWQ → GGUF path, no supported GPTQ → MLX path, and no way to "upgrade" a 4-bit artifact to 8-bit. Every re-target is a fresh run from the bf16 base.

> **Beyond the video:** engines now accept several of these and will silently pick one. vLLM's `--quantization` selects the *quantization method*, not the dtype, and passing `awq` where the checkpoint is GPTQ produces either a load failure or, worse, a slow fallback kernel that still runs. Set `--quantization awq_marlin` explicitly for AWQ on sm_80+, and pin the engine version: kernel availability changes between minor releases and a `pip install -U` has silently changed which kernel served an artifact more than once in the wild.

### 16.2 Versioning a quantized artifact

A quantized checkpoint is a *derived* artifact with more inputs than a fine-tuned one. Anything that is not recorded is a variable you cannot control, and §10.19's "two quantizations give two models" is the direct consequence.

```json
{
  "artifact": "llama-3.3-70b-instruct-gptq-4bit-g128-descact",
  "base_model": "meta-llama/Llama-3.3-70B-Instruct",
  "base_revision": "6f9a4b1c9d2e0f7a3b5c8e1d4a7f2c5b9e3d6a1f",
  "method": "gptq",
  "bits": 4, "group_size": 128, "sym": true, "desc_act": true,
  "damp_percent": 0.01,
  "calibration": {
    "corpus": "s3://ml-artifacts/calib/prod-prompts-2026-08.jsonl",
    "sha256": "b41c...9e02",
    "samples": 512, "seq_len": 2048, "seed": 0
  },
  "tooling": { "gptqmodel": "4.2.5", "torch": "2.7.0", "transformers": "4.53.2", "cuda": "12.4" },
  "target": { "engine": "vllm", "engine_version": "0.9.1", "kernel": "gptq_marlin", "arch": "sm_80" },
  "weights_sha256": "7d2a...c118",
  "eval": {
    "suite": "gate-v3-200prompts", "mean_kl": 0.031, "p95_kl": 0.44,
    "top1_agreement": 0.972, "task_delta_pp": -0.6, "date": "2026-08-19"
  }
}
```

| Field group | Why it is load-bearing |
|---|---|
| `base_revision` | a Hub repo is mutable; a *commit hash* is not |
| `calibration.sha256` | the single most common cause of a quality regression (§10.16) |
| `tooling` | kernels and defaults change between minor versions; a rebuild that "should be identical" is not |
| `target.arch` | the same checkpoint on sm_80 and sm_89 can produce different numerics; the artifact is arch-qualified |
| `weights_sha256` | the only way to prove the file you are serving is the file you gated |
| `eval` | an artifact with no recorded gate result cannot be compared to its successor |

**Storage layout.** Keep `config.json`, `quantize_config.json` (GPTQ) or `quantization_config` (AWQ/FP8), the tokenizer files and `tokenizer.chat_template` *beside* the weights, in one directory, and treat the directory as the artifact. Deduplicate tokenizer and config files across variants — a 70B suite at three bit-widths and two methods will otherwise cross 200 GB of near-duplicate files.

### 16.3 A quantized model is a different model

This is the principle the rest of this section depends on. Quantization is not a storage optimization applied to an unchanged model; it is a **lossy transform of the parameterization** that changes the function the model computes. Consequences that follow directly:

- It needs its own **model card**: `quantized from X, method Y, bits Z, calibration set W, evaluated on V with results R`.
- It needs its own **evaluation run**. A score inherited from the parent is a claim about a model you are not serving.
- It needs its own **safety evaluation**. Refusal behaviour is a learned behaviour and it can shift; the standard finding is that aggressive quantization degrades safety alignment before it degrades capability, which is exactly the direction that is invisible in perplexity.
- It is, for the purposes of most certification and audit regimes, a **new model artifact** — not a version of the old one.
- It may have a **different licence surface**. A quantized derivative of a model with a research-only or non-commercial licence inherits that licence; publishing it to the Hub is a distribution event.

> **Beyond the video:** the practical test for "is this a different model?" is whether your existing acceptance evidence still applies. If your sign-off was "we ran suite A on model M and it passed", then a quantized M′ requires a rerun of suite A *plus* the quantization-specific gates (KL, long-context, structured output), because suite A was chosen for a model that had no quantization error. Teams that skip this step do not usually ship a broken product; they ship a product with a slow-burning quality regression that gets attributed to "the model" for months.

### 16.4 Evaluation after quantization — the CI gate

The gate runs on every rebuild, on a frozen prompt suite, and fails the build. It is not a report.

```python
# gate.py — runs in CI on every quantization rebuild. Exit 1 = do not publish.
import json, sys, hashlib

THRESHOLDS = {
    "mean_kl":         0.10,   # §12.2: > 0.10 mean KL is a real drift
    "p95_kl":          1.00,   # p95 catches the tail that mean hides
    "top1_agreement":  0.95,   # below this, the model is choosing different tokens
    "task_delta_pp":  -2.0,    # percentage points vs the bf16 baseline
    "json_valid_rate": 0.99,   # structured-output workloads only
    "router_agreement": 0.98,  # MoE only (§15.5)
}

def gate(metrics, suite_hash, expected_suite_hash):
    assert suite_hash == expected_suite_hash, "prompt suite changed — re-baseline first"
    failures = [k for k, limit in THRESHOLDS.items()
                if k in metrics and (
                    metrics[k] < limit if limit < 0 or k in ("top1_agreement",
                    "json_valid_rate", "router_agreement") else metrics[k] > limit)]
    if failures:
        print(f"GATE FAILED: {failures}\n{json.dumps(metrics, indent=2)}")
        sys.exit(1)
    print("gate passed")
```

Four things the gate must be told, because it cannot infer them:

| Requirement | Why |
|---|---|
| **Prompt shape matches production** | §15.1's RAG failure passed a prose KL gate and failed on long-context prompts with quoted spans |
| **Suite is frozen and hashed** | a suite that changes silently makes every historical number incomparable |
| **Calibration corpus is disjoint from the gate suite** | otherwise you are measuring memorization of the calibration set (§10.18) |
| **Baseline is re-measured on the same day** | engine and kernel updates move the bf16 reference too |

### 16.5 Rollout: canary, A/B, rollback

| Stage | Traffic | Gate | Rollback trigger |
|---|---|---|---|
| Shadow | 0% (mirrored) | offline gate above | any gate failure |
| Canary | 1–5% | gate + p95 latency + error rate + output-shape rate | any metric outside 2σ of the bf16 arm |
| A/B | 50/50 | the *task* metric, not perplexity | task metric worse with p < 0.05 |
| Full | 100% | monitoring continues | drift, not a step change |

**Rollback must be a config change, not a re-quantization.** That means the previous artifact and the bf16 base stay resident in the registry, and the serving layer selects by name. If rolling back requires a six-hour quantization run, you have made a rollback decision into an incident.

### 16.6 Monitoring and drift

Track two families of signal, and treat the second as the leading indicator.

| Family | Metrics | What it catches |
|---|---|---|
| **System** | tokens/s, p50/p99 latency, VRAM high-water, KV-cache utilization, OOM count, restart count | capacity and kernel regressions |
| **Output shape** | JSON-validity rate, tool-call success rate, average response length, repetition rate, refusal rate, citation-format compliance | quantization regressions *before* users notice |

Quantization degrades *format* before it degrades *content*. A rise in JSON-invalid responses, a 15% drop in average answer length, or a jump in the refusal rate are all quantization-shaped symptoms, and all three are computable from logs you already have.

**Drift.** The calibration corpus has a shelf life. When production traffic changes (a new language, a new document type, a new tool, a new system prompt), the activation statistics move away from the ones the scales were fit to. Re-calibrate on a schedule (quarterly is common) *and* on an alert: if output-shape metrics drift while the input distribution's embedding centroid moves, re-calibrate before you re-tune prompts.

### 16.7 Guardrails

| Guardrail | Why it matters more for a quantized model |
|---|---|
| JSON-schema validation with retry-on-invalid | quantized models fail *format* first; the retry budget should be spent on schema violations, not on refusals |
| Grammar / structured decoding (JSON mode, GBNF) | forces the token distribution through a constrained sampler, masking small logit errors — the cheapest fix for a marginal structured-output regression |
| Max-token and repetition caps | 2-bit and 3-bit models loop; a hard cap turns an infinite loop into a truncated answer |
| Output length band alarm | a sudden shift in the length distribution is the earliest available quantization signal |
| Fallback arm | keep a bf16 (or higher-bit) endpoint warm for a small slice; escalate on schema failure rather than on user complaint |

> **Beyond the video:** structured decoding is the highest-leverage guardrail available and is routinely left on the table. If your workload is JSON extraction, a grammar-constrained sampler removes an entire class of quantization failure at zero quality cost — the constrained tokens were never going to be sampled anyway. Do this before you go up a bit-width.

### 16.8 Latency and capacity: what quantization does and does not change

| Property | Effect of 4-bit weights | Why |
|---|---|---|
| Weight bytes read per token | **÷4** | the format is the point |
| Decode speed, batch 1 | ×2–3, not ×4 | compute still runs on FP16 tensor cores (§9.2) |
| Aggregate throughput | ×3–4 | concurrency: 4× the KV room means 4× the batch (§11.2) |
| Single-request latency at batch 1 | modest gain | memory-bound read shrinks, kernel overhead does not |
| Prefill (TTFT on long context) | small gain | prefill is compute-bound, not bandwidth-bound |
| KV-cache capacity | **unchanged** | unless the KV cache is separately quantized (§4.9) |
| Accuracy | monotonically non-improving | quantization never helps |

If your SLO is **p99 latency at low batch**, quantizing weights is the wrong lever — measure the GEMV path explicitly (AWQ `version="GEMV"`) and consider a smaller model or distillation instead. If your SLO is **tokens per dollar at high concurrency**, quantization is the cheapest lever you have.

### 16.9 The compliance angle

- **Model card and registry.** A quantized model is a new entry, with the parent, the method, the bits, the calibration data provenance, and the eval result recorded.
- **Data provenance.** The calibration set is *training data* for the purposes of most data-governance policies. If the corpus contains PII or customer documents, that is a data-processing event — §15.2's bank and §15.1's legal client both needed the calibration corpus to stay on-premises.
- **Audit trail.** Which bits of the pipeline are reproducible: base hash + calibration hash + tooling versions + weights hash. Without all four, "we can rebuild it" is not a claim you can defend.
- **Regulated deployments.** Some regimes require the full-precision artifact to remain on record even when the 4-bit artifact is what serves traffic. Plan the storage (2× the model size, not 1×) and the evaluation (both artifacts evaluated) before an auditor asks.
- **Licence inheritance.** Quantized derivatives inherit the base model's licence terms; check before publishing to a Hub.

### 16.10 Pre-flight: what will *actually* be quantized

Before spending GPU-hours, inspect the checkpoint and count. The compression ratio you get is decided by which modules the recipe skips, and the skip list is different for every architecture family.

```python
# preflight.py — run BEFORE the quantization job. Prints the real compression ratio.
import json, sys
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(sys.argv[1])
IGNORE = {                       # adjust per architecture family
    "llama":  ["lm_head"],
    "qwen2":  ["lm_head"],
    "qwen2_vl": ["vision_tower", "multi_modal_projector", "lm_head"],
    "moE":    ["lm_head", "gate"],     # the router must stay high-precision
    "tied":   [],                      # tied embeddings: lm_head IS the embedding
}

def bucket(name):
    if any(k in name for k in ("vision_tower", "multi_modal_projector")): return "vision (excluded)"
    if name.endswith("lm_head"):  return "lm_head (excluded)"
    if "gate" in name and getattr(cfg, "num_experts", None): return "router (excluded)"
    if "embed" in name:           return "embedding (kept fp16)"
    if any(k in name for k in ("q_proj","k_proj","v_proj","o_proj",
                               "gate_proj","up_proj","down_proj")): return "linear (quantized)"
    return "other (kept fp16)"

totals, n_params = {}, cfg.num_parameters if hasattr(cfg, "num_parameters") else None
for n, p in [(n, 1) for n in []]:  # replace with a real module walk if you have the model loaded
    pass

print(json.dumps({
    "architectures": cfg.architectures,
    "n_layers": getattr(cfg, "num_hidden_layers", None),
    "n_kv_heads": getattr(cfg, "num_key_value_heads", None),
    "head_dim": getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
    "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", False),
    "recommended_ignore": IGNORE.get(cfg.model_type, ["lm_head"]),
}, indent=2))
```

What the walk usually reveals, across the four families this module covers:

| Family | Modules excluded from quantization | Effect on the ratio |
|---|---|---|
| Llama-3.x dense | `lm_head` (or nothing, if embeddings are tied) | small; ~95% of parameters are quantizable |
| Qwen2-VL / Llama-Vision | `vision_tower`, `multi_modal_projector` (plus `lm_head`) | 5–10% of parameters stay fp16; the ratio drops accordingly, and §15.3 says that is the right trade |
| MoE (Qwen3-MoE, DeepSeek, Mixtral) | `lm_head`, the router `gate` | the router is a tiny fraction of parameters and the most fragile module in the model; never quantize it |
| Any model with tied embeddings | nothing at all — `lm_head` *is* the embedding matrix | quantizing it quantizes the embedding, which is where rare-token accuracy lives |

The count that matters is not "did the ratio come out right" but "is every module that carries a discrete decision still high-precision" — routers, vision towers, the LM head, and any module whose output is compared rather than summed.

> **Beyond the video:** the two exclusions with the clearest empirical case are `lm_head` and the MoE router. `lm_head` produces the logits that a sampler argmaxes; a small perturbation there flips the sampled token directly, with no downstream averaging to hide it, and the tokens it flips first are the low-probability ones — i.e. the structured-output and rare-content tokens. The router has the same property for a different reason: it is a top-k comparison, and quantizing it changes *which experts* run, not just how well they compute (§15.5).

### 16.11 Capacity planning worksheet

The arithmetic of §11, in one function, so the answer to "will it fit?" is a number rather than an afternoon. Every constant here is derived earlier in the module; the point of the code is that capacity planning is a two-minute calculation, not a re-quantization.

```python
# plan.py — python plan.py <params_b> <layers> <kv_heads> <head_dim> <ctx> <batch> <bits>
import sys

def plan(params_b, n_layers, n_kv_heads, head_dim, ctx, batch,
         bits=4, bpw_overhead=0.9, kv_bytes=2, gpu_gb=80, util=0.92,
         act_gb=1.5):
    """Returns (weights_gb, kv_gb, total_gb, budget_gb, fits)."""
    weights = params_b * 1e9 * (bits + bpw_overhead) / 8 / 1e9
    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * kv_bytes      # bytes
    kv = kv_per_token * ctx * batch / 1e9
    total = weights + kv + act_gb
    budget = gpu_gb * util
    return round(weights, 1), round(kv, 1), round(total, 1), round(budget, 1), total <= budget

# (params_b, layers, kv_heads, head_dim, ctx, batch, bits, ...)
cases = {
  "70B on 1xA100-80 (W4A16, fp8 KV)":  (70, 80, 8, 128, 8192, 4, 4, 0.9, 1),
  "70B on 1xA100-80 (bf16, for reference)": (70, 80, 8, 128, 8192, 4, 16, 0, 2),
  "8B on 1xA10G-24 (W4A16)":           (8, 32, 8, 128, 4096, 8, 4, 0.9, 2),
  "3B on 1xA10G-24 (W4A16)":           (3, 36, 8, 128, 8192, 32, 4, 0.9, 2),
  "235B MoE on 8xH100 (FP8, TP=8)":    (235/8, 94, 4, 128, 32768, 20, 8, 0.25, 1),
}
for name, (pb, L, kh, hd, ctx, bs, bits, ovh, kvb) in cases.items():
    w, kv, tot, bud, fits = plan(pb, L, kh, hd, ctx, bs, bits, ovh, kv_bytes=kvb)
    print(f"{name:42s} weights {w:6.1f} GB | kv {kv:6.1f} GB | "
          f"total {tot:6.1f} / {bud:5.1f} GB -> {'FITS' if fits else 'DOES NOT FIT'}")

# 70B on 1xA100-80 (W4A16, fp8 KV)            weights   35.5 GB | kv    5.4 GB | total   42.4 /  73.6 GB -> FITS
# 70B on 1xA100-80 (bf16, for reference)      weights  140.0 GB | kv   10.7 GB | total  152.2 /  73.6 GB -> DOES NOT FIT
# 8B on 1xA10G-24 (W4A16)                     weights    4.1 GB | kv    1.1 GB | total    6.7 /  22.1 GB -> FITS
# 3B on 1xA10G-24 (W4A16)                     weights    1.5 GB | kv   18.9 GB | total   21.9 /  22.1 GB -> FITS (barely)
# 235B MoE on 8xH100 (FP8, TP=8)              weights   30.2 GB | kv   12.9 GB | total   44.6 /  73.6 GB -> FITS
```

Three things the worksheet makes obvious that prose does not:

1. **Row 1 versus row 2 is the whole reason this module exists.** The bf16 70B exceeds the budget by 2.07×; the 4-bit one uses 58% of it.
2. **Row 4 is a KV-cache problem disguised as a weight problem.** A 3B model at 32 concurrent sequences and 8k context spends 18.9 GB of its 22.1 GB budget on KV and 1.5 GB on weights. Quantizing the weights harder would change nothing; the levers are `--max-num-seqs`, `--max-model-len`, and FP8 KV.
3. **`bpw_overhead` is not optional.** Setting it to `0` makes a `Q4_K_M` 70B "35.0 GB" — a 7 GB underestimate against the real file. The metadata is real bytes (§4.8).

### 16.12 Unit economics: what the saving actually is

$/1M output tokens is the number a finance team asks for, and it is a function of *aggregate throughput*, not of the bit-width.

| Deployment | Cost | Aggregate throughput | $/1M output tokens | Notes |
|---|---|---|---|---|
| bf16 70B, 2×A100-80 (TP=2) | $2.58/h (est.) | ~110 tok/s at batch 2 | **$6.51** | the reference: it runs, expensively |
| W4A16 70B, 1×A100-80 | $1.29/h (est.) | ~190 tok/s at batch 4 | **$1.89** | §15.1; the same model, 3.4× cheaper |
| W4A16 70B, 1×A100-80, FP8 KV, batch 8 | $1.29/h (est.) | ~320 tok/s at batch 8 | **$1.12** | the last 2× came from the KV cache, not the weights |
| FP8 235B MoE, 8×H100 | ~$24/h (est.) | ~2,100 tok/s at batch 20 | **$3.17** | §15.5; a 10× larger model at half the per-token cost of the bf16 70B |

Read the last two rows together: **the weight quantization cut the cost 3.4×, and the KV-cache quantization on an already-quantized model cut it another 1.7×.** Teams routinely stop after the first step and report the second as unavailable.

> **Beyond the video:** the cost model is `$/hr ÷ (tok/s × 3600) × 1e6`, and both terms move. Rental prices for A100/H100-class hardware have fallen every year since 2023 while throughput per card has risen, so a cost table in a video ages faster than any other number in it. Recompute with your own contract price before quoting a saving; the *ratio* between rows is stable, the absolute dollar figure is not.

### 16.13 Production checklist

- [ ] Binding constraint named (capacity / bandwidth / compute / portability) and written down.
- [ ] Method chosen from §8.1's decision tree, not from a blog post.
- [ ] Calibration corpus is **production-shaped**, **disjoint** from the eval suite, and **hashed**.
- [ ] Artifact manifest complete: base revision, calibration hash, tooling versions, target arch, weights hash.
- [ ] `quantization_config` and `modules_to_not_convert` present in `config.json` next to the weights.
- [ ] `tokenizer.chat_template` present and used by the *server*, not re-implemented in application code.
- [ ] Gate run on workload-shaped prompts; thresholds in CI, failure blocks publish.
- [ ] Long-context gate run at depth 0.5L and 0.9L, not just at 1k tokens.
- [ ] Safety eval re-run on the quantized artifact.
- [ ] Rollback target (previous artifact + bf16 base) resident in the registry.
- [ ] Output-shape metrics monitored from day one, with alarms.
- [ ] Re-calibration trigger documented (date + drift condition).

---

## §17 Common Misconceptions

CS-10 §17 lists the eighteen misconceptions of the *fundamentals* — the equations, the granularity, the bit-width curve. These are the fourteen that belong to the production half: the advanced methods, the tooling, and the deployment.

1. **"4-bit is always 4× smaller."** Actually it is ~3.2–3.6× smaller in practice. Because only the *linear* weights are quantized: embeddings and the LM head are frequently excluded, RMSNorm and biases stay in fp16, and — the part people forget — the metadata is not free. `group_size=128` costs `32 bits/128 weights = 0.25 bpw` for scales and, in asymmetric mode, another 0.25 for zeros, so the honest figure is ~4.5–4.9 bits per weight (§4.8). A 70B goes from `70e9 × 2 = 140 GB` to ~42 GB, which is 3.3×, not 4×. And the KV cache, which is often the larger term at long context, does not shrink at all.

2. **"Quantization is lossless enough — the differences don't matter."** Actually it is lossy by construction and the loss is *workload-shaped*. Because quantization error is not uniformly distributed over the output distribution: it concentrates in rare tokens, in structured output, in long-context retrieval, and in low-resource languages. Two models with identical perplexity can differ by 3 points of pass@1 on code (§15.2 of CS-10) or lose their citation behaviour on RAG prompts while passing a prose KL gate (§15.1). "Lossless enough" is a claim about your *gate*, not about the model.

3. **"You can quantize a model and then fine-tune it normally."** Actually you cannot, and this is the most expensive misconception in the field. Because `round()` has zero derivative almost everywhere, so the quantized codes carry no useful gradient, and the artifact is a fixed encoding rather than a parameterization. There are exactly three legitimate orders of operations: **bf16 → fine-tune → merge → quantize** (the default); **QLoRA**, which trains adapters over a *frozen* quantized base and never updates the quantized weights; and **QAT**, which starts from bf16 and simulates quantization in the forward pass so the weights are trained *against* a quantizer. Fine-tuning the quantized weights themselves is not one of them (§10.4).

4. **"AWQ quantizes the activations."** Actually an AWQ artifact is **W4A16**: 4-bit weights, 16-bit activations. Because "activation-aware" describes the *calibration statistic* the algorithm uses (`mean|X_j|` per input channel, §4.3), not the deployed precision. The rescaling identity `W·X = (W·diag(s))·(diag(s)⁻¹·X)` is applied at calibration time and folded into the neighbouring LayerNorm/Linear, so at inference time the activation path is exact fp16. If you need 8-bit or 4-bit *activations* you are choosing W8A8 (FP8/INT8) or W4A8, which is a different method family entirely.

5. **"FP8 is just a smaller INT8."** Actually FP8 is a floating-point format with 3–4 mantissa bits and 4–5 exponent bits (`e4m3`, `e5m2`), and the two are good at opposite things. Because a floating-point grid has *relative* precision, `e4m3` resolves small values where INT8's uniform grid is coarse near zero, at the cost of having only 8 levels between powers of two. INT8's uniform grid plus a per-channel scale is usually more accurate on a well-behaved weight tensor; FP8's advantage is that it does not need one, which makes it the natural choice for *activations* and for the KV cache where the dynamic range is the problem (§4.10).

6. **"A 4-bit tensor core does the multiply in 4 bits."** Actually the 4-bit kernel dequantizes weights into FP16 inside the matmul and runs on FP16 tensor cores; only Blackwell-class FP4/FP6 paths (and some INT8 paths) have true low-precision arithmetic. Because that is why a 4-bit model is not 4× faster (§18.6): you saved bandwidth, not FLOPs. The measurable consequences are a 2–3× batch-1 decode gain and a 3–4× throughput gain from concurrency.

7. **"If it fits and it's fast, it's cheap."** Actually cost is a function of your SLO, not of your bit-width. Because the savings from quantization are realized through *concurrency* — 4× the weights headroom means 4× the KV-cache room means 4× the batch. If your contract is p99 latency at batch 1, quantization buys you a modest single-stream improvement and no cost reduction at all; the lever you actually want is a smaller or distilled model (§20).

8. **"You can convert a GPTQ checkpoint to GGUF."** Actually there is no supported GPTQ → GGUF path, and no AWQ → MLX path either. Because the two families do not share a weight layout: GPTQ/AWQ store fp16 weights plus 4-bit codes, scales and zeros in a per-group layout designed for a CUDA kernel, while GGUF stores its own block-quantized representation (`Q4_K`, `IQ4_XS`, …) designed for CPU and Metal. Every re-target is a fresh quantization run from the **bf16 base**, which is the other reason to keep the fp16/bf16 original resident forever (§15.4).

9. **"`llama.cpp` isn't really using a quantization method."** Actually `llama-quantize` with `Q4_K_M` is a post-training, weight-only, block-wise, mixed-precision quantization method — a different *family* from GPTQ/AWQ, not an absence of one. See the correction below.

10. **"The quantization format is the model."** Actually the format is one of many derived artifacts, and the *base revision* is the model. Because re-quantizing the same base with a different calibration corpus gives a materially different model, while the same base at a different bit-width is recognizably the same model with a measurable delta. Version the base, the calibration data, and the tooling separately (§16.2).

11. **"The same config produces the same artifact."** Actually two quantizations with identical arguments produce different weights, because calibration-sample ordering, GPU architecture, kernel selection, and library version all enter the numerics. Because a sum over a calibration batch is not associative in floating point, and different libraries tile it differently. Hash the output weights; if the hash changed, the artifact is new, whatever the config says (§10.19).

12. **"`desc_act` is free accuracy."** Actually it costs roughly 10% of inference speed and is rejected by some kernel builds, because the activation-ordering permutation changes the memory layout the kernel expects and forces a gather (§4.2). It is worth it at 3-bit and for models where the accuracy delta is measurable; it is often not worth it at 4-bit where the loss is already inside the noise.

13. **"Chat templating is the application's job."** Actually the chat template belongs to the artifact and should be carried into it — in GGUF as `tokenizer.chat_template`, in the HF tokenizer config for safetensors artifacts — and the **server** should apply it. Because an application that re-implements prompt formatting will eventually drift from the training format, and the failure mode is fluent, well-formed, completely wrong output with no error message (§10.5).

14. **"Quantize once, ship forever."** Actually the calibration corpus goes stale. Because the scales are fit to the activation ranges of the corpus, and when traffic shifts — a new language, a new document type, a new tool, a longer system prompt — those ranges no longer describe your inputs. Re-calibration on a schedule and on a drift alert is part of owning the artifact (§16.6).

> **Correction:** at [3:01:17] the instructor says: *"the full form of the GGML is GGO machine learning framework. So it is not GPT generated model language guys. Uh I have seen at many places many blog people are using this particular name but I think this is not true."* Both halves are wrong. GGML is not "GGO machine learning"; the expansion in circulation for years — including in the ecosystem's own documentation — is **"GPT-Generated Model Language"**, and the library's author has since offered the backronym **"Georgi Gerganov Machine Learning"** after himself. "GGO machine learning framework" is not an expansion anyone uses. The operational takeaway is unaffected: GGML is a C tensor library, and GGUF is its successor *file format*, not an engine.

> **Correction:** at [3:02:31]–[3:02:52] the instructor insists: *"GGML is not performing the quantization. This was not created to perform the quantization. Again, I'm saying GGML cannot perform the quantization."* — and then at [3:15:17] he describes the practical with *"this is the native format of the llama CPP itself. We are not going to use any explicit quantization method like GPTQ, AWQ, QAT"*. The first claim is misleading and the second is wrong. **`llama-quantize` is a quantization method**: it is post-training (no gradients), weight-only (activations stay fp16/fp32), block-wise (32-, 16- or 256-weight blocks with quantized scales), and mixed-precision (the `_S`/`_M`/`_L` suffix selects which tensors stay higher-precision). It is a *different family* from GPTQ and AWQ — no Hessian, no activation-aware scaling, no Cholesky — but calling it "not an explicit quantization method" would tell a reader that a `Q4_K_M` file is somehow the unquantized model, which is exactly backwards: it is a 4.85-bit-per-weight quantization of the fp16 base.

> **Beyond the video:** the misconception that costs the most money in practice is #3, in its corporate form: "we'll quantize the model our vendor gave us and then fine-tune it on our data." That pipeline is impossible as stated. The workable version is: obtain the bf16 base, QLoRA it (or full fine-tune it if you have the memory), merge into bf16, *then* quantize for serving — and accept that the merge-then-quantize step is a new artifact needing its own gate (§16.4).

> **Beyond the video:** a second one worth naming is the "quantization will fix my VRAM problem" assumption. If your OOM happens during *prefill* on a long context, weight quantization will free headroom but the peak is driven by activations and attention over the whole sequence; if it happens because of many concurrent sequences, the term you need to attack is the KV cache (`--kv-cache-dtype fp8`, `--max-num-seqs`, GQA/multi-head latent attention), not the weights. Diagnose which peak you are hitting — `torch.cuda.max_memory_allocated()` at the two phases takes two minutes and saves a re-quantization.

---

## §18 Key Takeaways

1. **Pick the method from the binding constraint, not from the leaderboard.** Capacity-bound → W4A16 (GPTQ or AWQ). Compute-bound on sm_89+ → FP8 W8A8. Portability across CPU/Metal → GGUF k-quants. Training a quantized base → NF4. Every one of §15's five scenarios was decided in the first five minutes by naming which of those four words applied.

2. **GPTQ is second-order error compensation.** `argmin_Ŵ ‖WX − ŴX‖²_F`, with `H = XᵀX` the input covariance over the calibration set, sensitivity `Δ_err ≈ (w − ŵ)² · H_jj`, and the residual from each quantized column subtracted from the columns not yet processed. `damp_percent` exists because `H` is often singular; `desc_act` reorders the columns by `H_jj` and costs about 10% of inference speed.

3. **AWQ is W4A16 and does not quantize activations.** It measures per-input-channel activation magnitude, protects the top ~1% of channels by rescaling `W·diag(s)` with `diag(s)⁻¹` folded into the preceding op, and ships 4-bit weights with 16-bit activations. It is post-training quantization — never QAT.

4. **The calibration corpus is the highest-leverage hyperparameter in the whole pipeline.** A few hundred *in-domain* samples with `seq_len ≈ 2048` beats thousands of generic ones, and the corpus must be disjoint from the evaluation suite. Every "good on the benchmark, bad in production" incident in §15 traces back to this line.

5. **The KV cache is a separate budget.** `2 × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes`, computed *before* you choose a bit-width, because it is frequently the term that decides whether the model fits at all. It does not shrink when you quantize the weights. FP8 KV needs sm_89+; K belongs per-channel and V per-token.

6. **FP8 is not "smaller INT8".** `e4m3` has 3 mantissa bits and a 4-bit exponent; its advantage is dynamic range, which is why it is the natural choice for activations and the KV cache, and why it needs sm_89 or newer to have a fast path at all.

7. **4-bit buys bandwidth and capacity, not FLOPs.** The kernel dequantizes into FP16 and runs on FP16 tensor cores, so the honest promise is ~2–3× at batch 1 and 3–4× throughput aggregate — the latter coming from the concurrency that the freed memory allows. Anyone promising 4× is quoting the memory ratio as a speed ratio.

8. **QAT is for people who own the model.** At 65B scale it is dozens of GPUs and an industrial training pipeline; the practical alternatives are PTQ, LoRA-plus-QAT (partial QAT), or a hybrid that keeps embeddings in FP32 and quantizes the linear and attention layers.

9. **QLoRA's NF4 grid plus double quantization lands at `4 + 8/64 + 32/16384 = 4.127` bits per parameter**, which is what makes a 7B fine-tune fit in roughly 6 GB of weights; paged optimizers handle the activation spikes. The quantized weights are frozen — the adapter is the thing being trained.

10. **A quantized model is a different model**, for engineering purposes and usually for compliance purposes. It gets its own model card, its own evaluation, its own safety evaluation, its own hash, and its own rollback slot. Inheriting the parent's eval score is not a shortcut; it is an unpriced risk.

11. **Gate on the workload's prompt shape.** Mean KL, p95 KL, and top-1 agreement are the instruments; the prompt suite must look like production, include the long-context depths (`0.5L` and `0.9L`), and be hash-frozen so historical numbers stay comparable. Perplexity is not a gate.

12. **Tooling in this space churns hard, so pin everything.** AutoGPTQ is deprecated (0.7.1, pinned to Python 3.11); AutoAWQ is archived and superseded by vLLM's `llm-compressor`; `torch.ao.quantization`'s eager QAT is legacy in favour of `torchao`; bitsandbytes remains the odd exception that keeps working unchanged for NF4. Record versions in the artifact manifest because a rebuild six months later will not reproduce the same bytes.

13. **One artifact per engine, and re-quantize rather than convert.** There is no supported GPTQ→GGUF or AWQ→MLX path. Keep the bf16 base resident and treat each target as a fresh quantization run with its own gate.

14. **Rollback must be a config change.** Keep the previous artifact and the bf16 base in the registry and select by name; if a rollback requires a quantization run, you have turned a routine deploy into an incident.

15. **The quantized artifact inherits more than the weights.** It inherits the base model's licence surface, the calibration corpus's data-governance obligations, and the chat template that makes it usable — and the chat template, of everything on that list, is the one whose absence produces fluent wrong answers with no error.

---

## §19 Self-Check Questions

1. Compute the KV-cache size per token for a 70B model with 80 layers, 8 KV heads and head_dim 128, at fp16 and at FP8. Then compute the total KV for `8192` tokens at batch 4, and state whether a 4-bit 70B fits on one A100-80GB with `--gpu-memory-utilization 0.92`.
2. State GPTQ's objective, what `H` is, how the sensitivity expression `Δ_err ≈ (w − ŵ)² · H_jj` follows from it, and what `damp_percent` and `desc_act` each change. What does `desc_act` cost at inference?
3. A colleague says "AWQ quantizes activations so it should be more accurate than GPTQ". Correct them in three sentences, naming the format the artifact actually ships and where the activation-side scale goes at inference time.
4. You have 200 in-domain samples for calibration and 10,000 generic ones. Which do you use, how many, and what must be true about the relationship between the calibration set and the evaluation set?
5. Your 4-bit model passes the KL gate on prose and fails on your production RAG traffic. Give three candidate causes ranked by likelihood and the diagnostic for each.
6. Explain why a 4-bit model is not 4× faster than fp16, and name the two quantities it *does* quadruple.
7. You must quantize a vision-language model for document extraction. Which modules do you exclude, what does the calibration set have to contain, and what is the evidence that the exclusion matters?
8. Your team wants to quantize a model with GPTQ and then full fine-tune it on your data. State what is impossible about that plan and give the three correct orderings of fine-tuning and quantization.
9. Compare FP8 `e4m3`, INT8, and NF4 for the *KV cache* specifically. Which do you choose and why? What hardware constraint applies, and which flag must accompany an FP8 KV cache in vLLM?
10. Give the five fields of a quantized-artifact manifest that you would insist on before serving, and for each, the failure the field prevents.

<details>
<summary><strong>Answers</strong></summary>

1. Per token: `2 × 80 × 8 × 128 × 2 bytes = 327,680 B = 320 KiB` at fp16; `160 KiB` at FP8. At 8192 tokens × batch 4 = 32,768 tokens: **10.7 GB** fp16, **5.4 GB** FP8. Weights for a 4-bit 70B are `70e9 × 0.5 ≈ 35 GB`. Total ≈ 35 + 5.4 + ~1.5 (activations, CUDA graphs, context) ≈ **42 GB**, against a budget of `80 × 0.92 = 73.6 GB`. It fits with ~30 GB to spare — the spare is the point, because it becomes KV-cache room and therefore batch size.

2. Objective: `argmin_Ŵ ‖WX − ŴX‖²_F` — the error is measured on the layer's *output*, not on the weights. `H = XᵀX` is the input second-moment (covariance) over the calibration set, so `H_jj` measures how active input channel `j` is. Expanding the objective gives `Δ_err ≈ (w_j − ŵ_j)² · H_jj` per column: the same rounding error costs more where the input is large, which is why columns are quantized in order of decreasing `H_jj` and each column's residual is subtracted from the remaining ones via `H⁻¹` (Cholesky). `damp_percent` adds `λI` to the diagonal so a singular `H` is invertible; `desc_act` (activation reordering) implements the decreasing-`H_jj` order across the whole layer rather than within a group. Cost: the permutation breaks the contiguous layout the kernel expects, forcing a gather — roughly 10% slower inference, and outright rejection by some Marlin builds.

3. AWQ's name says it is *aware of* activations, not that it quantizes them: the calibrated statistic `mean|X_j|` is used only to choose the per-channel scale `s_j = a_j^α / w_j^(1−α)`. The deployed artifact is **W4A16** — 4-bit weights, 16-bit activations — because the rescaling identity `W·X = (W·diag(s))·(diag(s)⁻¹·X)` lets the activation-side factor `diag(s)⁻¹` be folded into the preceding LayerNorm or Linear, where it is exact in fp16 and costs nothing. So there is no activation precision for AWQ to be "more accurate" in; if you want quantized activations you are choosing W8A8, which is a different family. And AWQ is post-training: no gradients, no QAT.

4. Use the **200 in-domain samples**, and use all of them (the accuracy curve knees at a few hundred; diversity beats count past that). `seq_len` should be near the serving context (2048 is a good default). The critical constraint: the calibration set and the evaluation suite must be **disjoint** — overlapping them converts your gate into a memorization test, because the scales were fit on exactly the activations the gate measures.

5. Ranked: (a) **calibration domain mismatch** — the scales were fit on prose and your RAG prompts have a different activation distribution; diagnostic is to compare the token/embedding distribution of the calibration corpus against production prompts, and the fix is re-quantizing with an in-domain corpus. (b) **prompt-shape sensitivity** — long contexts with many quoted spans exercise attention over distant tokens, where argmax flips cost you the citation; diagnostic is §12.5's needle test at depth 0.5L and 0.9L and a per-position KL breakdown. (c) **a structured-output or `lm_head` effect** — if the failure is in *format* (missing citations, dropped quotes) rather than retrieval, exclude `lm_head` from quantization, or move to `group_size=32`; diagnostic is JSON/citation-schema validity rate. Ranked this way because (a) is the most common, (b) is the most context-specific, and (c) is the most fixable.

6. Because the 4-bit kernel is a *dequantize-then-matmul* kernel: it expands weights to FP16 inside the operation and executes on FP16 tensor cores, so FLOP throughput is unchanged. What it quadruples is (i) bytes of weight read per token, and (ii) KV-cache capacity per unit of VRAM — which converts into ~2–3× decode at batch 1 (bandwidth-bound) and 3–4× aggregate throughput (concurrency-bound).

7. Exclude `vision_tower`, `multi_modal_projector`, and `lm_head`; quantize only the language tower's linear layers. The calibration set must contain **image-text pairs from the same document distribution** — a text-only corpus never exercises the projector, so its activation statistics are never observed and the model hallucinates on dense pages while scoring perfectly on text evals. The evidence that the exclusion matters is the ablation: §15.3 measured 91.4% field-level exact match with the exclusion versus 78.9% without it, with the damage concentrated on small-font fields.

8. Impossible: the quantized codes carry no gradient — `round()` has zero derivative almost everywhere — and the artifact is a fixed encoding, so there is no parameterization for the optimizer to update; a "fine-tune" would either silently do nothing useful or degrade a model you cannot restore. The three correct orderings are: **bf16 → fine-tune → merge → quantize** (default, best quality); **QLoRA** (adapters over a frozen NF4 base, merge into bf16, then re-quantize if needed); and **QAT** (start from bf16, insert fake-quantization with a straight-through estimator, train, then `convert`).

9. For the KV cache, choose **FP8 `e4m3`**. INT8's uniform grid is a poor fit for the KV distribution, which has a heavy tail of large-magnitude "attention sink" entries plus a dense mass of small values; a floating-point grid gives relative precision where the mass is and range where the tail is. NF4 is a *weight* quantizer with a per-block scale and is not the right tool for a streaming cache. The hardware constraint is **sm_89 or newer** — FP8 KV has no fast path on A100 (sm_80). In vLLM the flag is `--kv-cache-dtype fp8` and it must be accompanied by `--calculate-kv-scales`, otherwise the scale stays at 1.0 and the format's range is wasted.

10. **`base_revision`** (a Hub repo is mutable; prevents "the parent model changed under us"); **`calibration.sha256`** (prevents the most common quality regression and makes re-quantization reproducible); **`tooling` versions** (prevents a rebuild that silently differs because a kernel default moved); **`weights_sha256`** (proves the served file is the gated file); **`target.arch` + engine version** (prevents "it was fine on the A100 and wrong on the H100" and non-reproducible numerics). Anything not in the manifest is a variable you cannot control — §10.19's "two quantizations give two models".

</details>

---

## §20 Cross-References

| Relationship | Module |
|---|---|
| **Builds on** | **CS-10 — Quantization I: Foundations** (the affine equations, `MSE = Δ²/12`, granularity, the bit-width curve, GGML vs GGUF, the VRAM table). This module assumes all of it; §15–§19 here are the production half of the same subject. |
| | CS-01 (Foundations: parameters, bytes, VRAM arithmetic, the dtype table) — the unit conversions every estimate in §15 depends on |
| | CS-04 (Fine-Tuning vs RAG vs Agents — the deployment context; §15.1's RAG scenario is this module's workload) |
| **Needed by** | **CS-13 §6.8 — the LoRA configuration** (NF4 is the base of every QLoRA fine-tune; the `bnb_4bit_quant_type`, `bnb_4bit_use_double_quant` and `bnb_4bit_compute_dtype` knobs are the §4.11 algebra in *this* module, and the correct order of operations — merge into bf16, *then* quantize — is defined in §10.4 here. A dedicated "CS-23 — LoRA, QLoRA & Adapter Methods" module is planned but unwritten) |
| | Any serving/throughput module (KV-cache quantization, `--max-model-len` sizing, the concurrency argument in §16.8) |
| | CS-16 / CS-17 (Unsloth, Axolotl) — both expose a `load_in_4bit` path whose defaults are bitsandbytes NF4 |
| **Contrasts with** | **CS-08 / CS-09 — Knowledge Distillation** (the other compression axis: quantization shrinks the *encoding* of a fixed function, distillation trains a *smaller function*). The two compose: distill first, then quantize; quantizing a distilled student compounds the error, so gate each step separately. |
| | Pruning / sparsity modules (structured pruning removes capacity; quantization re-encodes capacity) |
| **Pairs with** | `code/08_quantize.py` — runnable bitsandbytes / GPTQ / AWQ / GGUF with the VRAM table and the post-quantization validation checklist |
| **Interview prep** | IQ-11 (the Part-2 question set: GPTQ's Hessian, AWQ's W4A16 correction, QAT budgets, GGUF tooling, KV-cache quantization) |
| **Cheat sheet** | CH-11 (the precision lattice, per-method snippets, serving flags, symptom → fix) |
| **Reference card** | CH-10 (the Part-1 companions: formulas, granularity, the bit-width curve) |

**If you read one other thing.** CS-10 §12 (Evaluation) is the instrument this module's §16.4 gate is built from, and CS-10 §17 is the misconceptions list for the fundamentals — this module's §17 deliberately does not repeat it.

---

## Appendix A — Instructor's Verbatim Key Claims

Quotes are from `LLM_Fine-Tuning_13_LLM_Quantization_Explained_PART_2_PTQ_QAT_GPTQ_AWQ_GGUF_GGML.txt`, with timestamps. The transcript is machine-generated and the instructor's name is transcribed as "Sani Savvita"; library names appear as "autographq", "IN4", "GGML/GGUF" interchangeably. Quoted text below retains those artifacts where the quote is verbatim, because rewriting them would make the quote unverifiable against the file. Where the instructor is imprecise, the correction appears in the body of this module rather than being silently fixed here.

| Timestamp | Claim |
|---|---|
| [7:14]–[7:59] | GPTQ is expanded as "gradient post training quantization"; the G stands for *gradient*, not GPT. |
| [10:10]–[10:38] | "generative post pre-trained transformer means what… this is nothing this is our favorite GPT… this technique was invented for the GPT itself for the GPT model and the technique name was… gradient post training quantization." |
| [10:38]–[11:03] | The 2017 paper "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" is recommended for the fundamental terminology; "the submitted date is on 15th December 2017." |
| [11:12] | "The Era of 1-bit LLMs" is introduced (transcribed as "the error of the one bit LLM") as the source of the 1.58-bit result. |
| [11:18]–[11:26] | "So IN4 means what? One bite. How many memory this info in4 is taking? So IN4 is taking one by memory means four bit right." — see the §15.2 correction: 4-bit is 0.5 B, not 1 B. |
| [12:33]–[12:39] | "we are able to reduce the size of the memory till one bit… how much I think 0.1 by right" — 1.58 bits is 0.1975 bytes. |
| [30:58]–[31:03] | "we required the small calibration data set around 50 to 200 sample" — the field's practical range is larger (§7.1), but the order of magnitude is right and the point (small, in-domain) is correct. |
| [33:57]–[34:01] | "in the base transformer itself, we are having six subsequent block" — see the §15.4 correction: production LLMs use 32–126 layers. |
| [35:17]–[35:20] | LLMs use **per-channel** weight quantization rather than per-tensor. |
| [37:31] | QAT "requires full or partial retaining of the model… requires multiGPU or TPU setup" (a retraining requirement, i.e. QAT is a training job, not a conversion). |
| [44:01]–[44:16] | "LLM mostly use symmetric plus per channel quantization for the weights." |
| [1:13:36]–[1:33:59] | GPTQ worked example: a 3×2 weight matrix with outputs `−1.5, −2.5, −3.5`, the input second-moment matrix `XᵀX = [[35, 44], [44, 56]]`, and the error term "we are going to be multiply 0.5 with 56 and we are getting 14" — the arithmetic is `0.5² × 56 = 14`, i.e. `(w − ŵ)² · H_jj` (§4.2). |
| [1:36:26]–[1:38:20] | `auto-gptq` 0.7.1 (2024) requires Python 3.11 while Colab ships 3.12; `GPTQModel` on PyPI is version 4.2.5, first released 15 Aug 2024, most recent 16 Sept 2025. |
| [1:40:00]–[1:40:12] | TheBloke's Hub account is cited as the distribution channel for quantized models: 3,863 quantized models, ~25,000 followers. |
| [1:43:57] | "autographq is a deprecated library." |
| [1:45:18]–[1:45:23] | bitsandbytes provides int8, NF4, FP4, and an 8-bit optimizer. |
| [1:50:17] | `pip install -v gptqmodel --no-build-isolation`, with protobuf pinned below 6.30. |
| [1:52:45] | Practical model ID: `llama-3.2-1B-instruct-gptqmodel-4bit-vortex-v1`. |
| [2:07:11]–[2:08:17] | ARC-Challenge result for the 4-bit GPTQModel checkpoint: normalized accuracy `0.27`, standard error `0.013`. |
| [2:15:58]–[2:16:16] | On AWQ: "don't think guys this is a quantise aware training. No, it belong to the post trading quantization only." |
| [2:22:57] | AutoAWQ was archived on 11 May (2024). |
| [2:23:08] | AutoAWQ's scale is cited as "2 million downloads, 7,000 plus models on hugging face and 2.1k star… solo developer." |
| [2:23:57]–[2:24:25] | AutoAWQ is superseded by vLLM's `llm-compressor` module. |
| [2:29:52]–[2:30:07] | The AWQ quantization config keys: `zero_point`, `q_group_size = 128`, `w_bit = 4`, `version = "GEMM"`. |
| [2:32:00]–[2:32:13] | Environment friction: transformers 4.56.1 installed against 4.51 required. |
| [2:32:40]–[2:34:06] | The `llm-compressor` recipe: `AWQModifier(schema="W4A16", target="Linear", ignore=[...])` plus `oneshot(..., num_calibration_samples=512)`. |
| [2:41:03]–[2:41:15] | QAT needs "4 to 8 GPU at least with 80GB". |
| [2:42:49]–[2:43:09] | QAT on Llama 65B means "dozen of GPUs, industrial scale"; the alternatives offered are PTQ, LoRA + QAT ("partial QAT"), and hybrid quantization (FP32 embeddings with int8/int4 linear and attention layers). |
| [2:47:38]–[2:48:39] | LLM-QAT (Meta, 29 May 2023) — "data free quantization aware training for large language model". |
| [2:49:17]–[2:49:24] | A Gemma 3 4B QAT checkpoint was found published on Hugging Face. |
| [2:50:13] | The QAT practical uses DistilGPT2. |
| [2:50:43]–[2:51:17] | QAT config: `MovingAverageMinMaxObserver`, `torch.qint8`, `per_tensor_affine`, with embeddings excluded from quantization. |
| [2:51:45]–[2:51:53] | `torch.quantization.prepare_qat(model, inplace=True)`. |
| [2:56:43]–[2:58:00] | GGML on GitHub: 13.2k stars, 1.3k forks, written in C; llama.cpp is the C++ extension of it. |
| [2:58:43]–[2:59:08] | The paper "Mind the Gap: A Practical Attack on GGUF Quantization" is introduced. |
| [3:00:41]–[3:01:05] | "nowadays this llama CPP is having capability it can perform the quantization itself… It's a normal simpler quantization only and the format of it is going to be Q4, Q8… Q6." |
| [3:01:17]–[3:01:31] | The GGML full form is given as "GGO machine learning framework", and "GPT generated model language" is explicitly rejected: "many blog people are using this particular name but I think this is not true." See the §17 correction. |
| [3:02:31]–[3:02:52] | "GGML is not performing the quantization. This was not created to perform the quantization. Again, I'm saying GGML cannot perform the quantization." |
| [3:02:54]–[3:03:06] | GGML loads LLMs in one of the quantized formats `Q4_0`, `Q4K`, `Q50`, `Q8K`. |
| [3:03:36]–[3:03:51] | "GGUF is a new standard file format, it's not a engine… which stores model in a single file along with all the metadata", while "GGML is a engine" that executes on CPU / Apple silicon / embedded devices. |
| [3:15:17]–[3:15:45] | The practical: `bin/quantize` produces Q4, "this is the native format of the llama CPP itself. We are not going to use any explicit quantization method like GPTQ, AWQ, QAT". See the §17 correction. |
| [3:16:00]–[3:18:32] | Building `main` and running `bin/llama-cli -m <model> -p "<question>" -n <words>`. |
| [3:19:22]–[3:20:16] | A RAG demo built on top of the quantized GGUF model. |

**Note on the transcript.** The Part-2 recording covers the full session (~3h20m) including the GPTQ, AWQ, QAT and GGML/GGUF practicals, so unlike the Part-1 file (CS-10, Appendix A) there is no gap in coverage. Two consequences for a reader checking a claim: (a) spoken library names are frequently mis-transcribed — `autographq` = AutoGPTQ, `IN4` = INT4, `byt` = byte, `GGML`/`GGUF` are used interchangeably in speech, and the instructor's name itself is rendered as "Sani Savvita"; (b) all numeric results quoted in §12 and §15 of this module come from the runs described in the transcript, not from the transcript's own arithmetic — where the two disagree (the `0.5 × 56` step at [1:33], the "0.5 byte = 4 bit" step at [11:18]), the correction in the body of this module is authoritative.

---

## Appendix B — Reference Links & Papers

### B.1 The methods

| Topic | Reference |
|---|---|
| **GPTQ** (the primary source for §4.2) | Frantar, Ashkboos, Hoefler & Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", ICLR 2023 — arXiv:2210.17323 |
| GPTQ's ancestor (the second-order/Layer-wise solver) | Frantar & Alistarh, "Optimal Brain Compression" — arXiv:2208.11580; Frantar, Singh & Alistarh, "OBQ: A Provably Optimal Algorithm for Neural Network Quantization" — arXiv:2205.11821 |
| **AWQ** (the primary source for §4.3) | Lin et al., "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration", MLSys 2024 — arXiv:2306.00978 |
| **SmoothQuant** (outlier migration) | Xiao, Lin, Seznec, Wu, Demouth & Han, "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", ICML 2023 — arXiv:2211.10438 |
| **LLM.int8()** (mixed-precision decomposition, the 6.0 threshold) | Dettmers, Lewis, Belkada & Zettlemoyer, "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", NeurIPS 2022 — arXiv:2208.07339 |
| **Rotation methods** (QuaRot / SpinQuant; the Hadamard bound in §4.5) | Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", NeurIPS 2024 — arXiv:2404.00456; Liu et al., "SpinQuant: LLM Quantization with Learned Rotations", ICLR 2025 — arXiv:2405.16406 |
| Calibration-free weight quantization | Badri & Shaji, "HQQ: Half-Quadratic Quantization of Large Neural Networks" — arXiv:2410.11845 |
| 1.58-bit / ternary weights (the paper at [11:12]) | Ma, Wang, Ma, Wang, Liu, Wang, Zhang, Zhu, Wang, Wang & others, "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" — arXiv:2402.17764; and "BitNet: Scaling 1-bit Transformers for Large Language Models" — arXiv:2310.11453 |
| **QAT**: the modern formulation | Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018 — arXiv:1712.05877 (submitted 15 Dec 2017, the date the instructor reads out at [11:00]) |
| QAT: the straight-through estimator | Bengio, Léonard & Courville, "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation" — arXiv:1308.3432 |
| QAT: learned step sizes / clipping | Esser, McKinstry, Bablani, Appuswamy & Modha, "Learned Step Size Quantization", ICLR 2020 — arXiv:1902.08153; Choi et al., "PACT: Parameterized Clipping Activation for Quantized Neural Networks" — arXiv:1805.06085 |
| **LLM-QAT** (data-free QAT for LLMs, [2:47:38]) | Liu, Oguz, Pappu, Xiao, Yih, Li, Krishnamoorthi & others (Meta), "LLM-QAT: Data-Free Quantization Aware Training for Large Language Models" — arXiv:2305.17888 (May 2023) |
| Efficient QAT at LLM scale | Chen et al., "EfficientQAT: Efficient Quantization-Aware Training for Large Language Models" — arXiv:2407.11062 |
| **QLoRA / NF4 / double quantization** (the §4.11 algebra) | Dettmers, Pagnoni, Holtzman & Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023 — arXiv:2305.14314 |

### B.2 The KV cache, the formats, and the hardware

| Topic | Reference |
|---|---|
| KV-cache quantization (the K-per-channel / V-per-token split) | Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024 — arXiv:2402.02750; Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization", NeurIPS 2024 — arXiv:2401.18079 |
| FP8 formats (`e4m3` vs `e5m2`, why gradients need the wider exponent) | Micikevicius et al., "FP8 Formats for Deep Learning" — arXiv:2209.05433 |
| Microscaling / NVFP4 / MXFP4 | OCP Microscaling Formats (MX) Specification v1.0 (Open Compute Project); NVIDIA NVFP4 documentation |
| 4-bit kernel design (Marlin, the kernel referenced throughout §15.1) | Frantar, Castro, Chen, Hoefler & Alistarh, "Marlin: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models" — arXiv:2408.11743 |
| A quantization-aware attack on GGUF files (the paper at [2:58:43]) | Egashira, Shioji, Shibasaki & Takeuchi, "Mind the Gap: A Practical Attack on GGUF Quantization", arXiv 2025 — the takeaway for production: **a GGUF file is not trusted input**; it carries executable-adjacent metadata and its quantization error can be adversarially shaped, so only load artifacts you produced or from a publisher you would trust with arbitrary files |
| Accuracy–performance trade-offs, measured honestly | Kurtic et al., "Give Me BF16 or Give Me Death? Accuracy-Performance Trade-Offs in LLM Quantization" — arXiv:2411.02355 |
| The GGUF specification | `github.com/ggerganov/ggml/blob/master/docs/gguf.md` (the format's own versioned spec, including the metadata KV layout) |
| k-quants, `imatrix`, `llama-quantize` | `github.com/ggerganov/llama.cpp` — `examples/quantize`, `examples/imatrix`, `tools/quantize`; k-quants introduced in PR #1684, imatrix in PR #4930 |
| llama.cpp's chat-template handling (the most common silent failure) | `llama.cpp` → `common/chat.cpp`, `--chat-template` / `--jinja`; the model's `tokenizer.chat_template` is applied by the server, not the client |

### B.3 Tooling referenced in this module

| Tool | State (2026) | Reference |
|---|---|---|
| **GPTQModel** | the maintained GPTQ implementation (PyPI 4.2.5, [1:38:20]) | `github.com/modelcloud/gptqmodel` |
| **AutoGPTQ** | deprecated — 0.7.1 (Aug 2024), pinned to Python 3.11 ([1:43:57]) | `github.com/AutoGPTQ/AutoGPTQ` (archived) |
| **llm-compressor** | the current AWQ/FP8/INT8 path, from the vLLM project ([2:23:57]) | `github.com/vllm-project/llm-compressor` |
| **AutoAWQ** | archived 11 May 2024; 2M+ downloads, 7k+ Hub models, 2.1k stars ([2:22:57]) | `github.com/casper-hansen/AutoAWQ` (archived) |
| **bitsandbytes** | NF4/FP4/INT8/8-bit optimizers; the `load_in_4bit` default across the ecosystem ([1:45:18]) | `github.com/bitsandbytes-foundation/bitsandbytes` |
| **torchao** | the current QAT/QATConfig and quantization API; supersedes the legacy `torch.ao.quantization` eager path used in the QAT practical ([2:50:43]) | `github.com/pytorch/ao` |
| **vLLM** | serving; `--quantization`, `--kv-cache-dtype`, `--calculate-kv-scales`, `--max-model-len`, `--max-num-seqs`, `--gpu-memory-utilization` | `docs.vllm.ai` → Supported Quantization / Engine Arguments |
| **TensorRT-LLM** | serving with engine-build-time quantization (`trtllm-build`) | `github.com/NVIDIA/TensorRT-LLM` |
| **SGLang**, **TGI** | alternative serving stacks with overlapping kernel support | `github.com/sgl-project/sglang`, `github.com/huggingface/text-generation-inference` |
| **llama.cpp / GGUF** | CPU, Apple Metal, CUDA and Vulkan inference; `llama-quantize`, `llama-imatrix`, `llama-server` | `github.com/ggerganov/llama.cpp` |
| **Ollama**, **LM Studio** | desktop wrappers over GGUF | `ollama.com`, `lmstudio.ai` |
| **MLX** | Apple-silicon-native quantization with its own weight layout | `github.com/ml-explore/mlx-examples` |
| **TheBloke** (distribution precedent, [1:40:00]) | the account that established "quantized checkpoints as first-class Hub artifacts" — 3,863 quantized models, ~25k followers at the time of recording | `huggingface.co/TheBloke` |
| Published QAT checkpoints (the pattern at [2:49:17]) | e.g. `google/gemma-3-4b-it-qat-q4_0-gguf` — a first-party QAT release shipped *as a GGUF*, i.e. the vendor did the quantization for you | Hugging Face model hub |

### B.4 Benchmarks and evaluation instruments used in §12 and §15

| Instrument | What it measures | Reference |
|---|---|---|
| **ARC-Challenge** (the `0.27` normalized accuracy at [2:07:11]) | grade-school science multiple choice; a coarse capability probe | Clark et al., "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge" — arXiv:1803.05457 |
| **IFEval** | instruction-following with verifiable constraints; the best cheap proxy for *format* degradation | Zhou et al., "Instruction-Following Evaluation for Large Language Models" — arXiv:2311.07911 |
| **GSM8K** | grade-school math word problems; sensitive to quantization in the reasoning chain | Cobbe et al., "Training Verifiers to Solve Math Word Problems" — arXiv:2110.14168 |
| **MMLU / MMLU-Pro** | broad knowledge; the standard headline number, and the least sensitive to quantization at 4-bit | Hendrycks et al., "Measuring Massive Multitask Language Understanding" — arXiv:2009.03300 |
| **Needle-in-a-haystack / RULER** | long-context retrieval at controlled depth; the instrument that catches KV-cache quantization damage | Kamradt, "Needle In A Haystack" (2023); Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?" — arXiv:2404.06654 |
| **KL divergence gate** (mean, p95, top-1 agreement) | the cheapest faithful measure of "is this the same distribution?" — §12.2 has the runnable implementation | the protocol used in the GPTQ/AWQ literature and in vLLM's own quantization evaluations |
| **pass@1 on a private suite** | the only metric that settles a code-model quantization decision (§15.2 of CS-10) | HumanEval (arXiv:2107.03374) as the public stand-in; SWE-bench (arXiv:2310.06770) for agentic coding |

**Repository artifacts used by this module**

| Artifact | Path |
|---|---|
| GPTQ notebook (Part 2, the current GPTQModel path) | `LLM Fine-Tuning-12-13-LLM-Quantization/LLM-Quantization-Part-2/GPTQ_UPDATE.ipynb` |
| AWQ notebook (Part 2, the llm-compressor path) | `LLM Fine-Tuning-12-13-LLM-Quantization/LLM-Quantization-Part-2/LLM_Quantization_AWQ.ipynb` |
| QAT notebook (Part 2, the DistilGPT2 practical) | `LLM Fine-Tuning-12-13-LLM-Quantization/LLM-Quantization-Part-2/QAT_in_LLM.ipynb` |
| GGUF / GGML notebook (Part 2) | `LLM Fine-Tuning-12-13-LLM-Quantization/LLM-Quantization-Part-2/gguf_ggml_practical.ipynb` |
| Part-1 notebooks (shared with CS-10) | `LLM Fine-Tuning-12-13-LLM-Quantization/LLM_Quantization/` — `Model_Quantization_Final.ipynb`, `LLM_Quantization_GPTQ.ipynb`, `LLM_Quantization_AWQ.ipynb`, `gguf_practical.ipynb`, `gguf_ggml_practical.ipynb` |
| Runnable end-to-end script | `Finetuning-Handbook/code/08_quantize.py` |
| Source transcript | `_source/transcripts/LLM_Fine-Tuning_13_LLM_Quantization_Explained_PART_2_PTQ_QAT_GPTQ_AWQ_GGUF_GGML.txt` |
