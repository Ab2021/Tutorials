# CS-10 — Quantization I: PTQ, QAT, GPTQ, AWQ, GGUF, GGML

| Field | Value |
|---|---|
| **Module** | Efficiency & Compression |
| **Source video(s)** | Video 12 — *LLM Quantization Explained (Part 1): PTQ, QAT, GPTQ, AWQ, GGUF, GGML* |
| **Transcript file(s)** | `LLM_Fine-Tuning_12_LLM_Quantization_Explained_PART_1_PTQ_QAT_GPTQ_AWQ_GGUF_GGML.txt` |
| **Companion code** | `LLM Fine-Tuning-12-13-LLM-Quantization/LLM_Quantization/` → `Model_Quantization_Final.ipynb`, `LLM_Quantization_GPTQ.ipynb`, `LLM_Quantization_AWQ.ipynb`, `gguf_practical.ipynb`, `gguf_ggml_practical.ipynb` |
| **Prerequisite** | CS-01 (Foundations: bytes, params, VRAM), CS-09 (distillation — the other compression axis) |
| **Followed by** | CS-11 (Quantization II: Advanced Methods & Production Practice) |
| **Difficulty** | Intermediate → Advanced |
| **Hands-on required** | Yes — the whole module is reproducible on a free Colab T4 |
| **Estimated study time** | 5h theory + 4h practical |

> **How this module and CS-11 split the topic.** This file is the *foundations* half: the arithmetic of a quantizer, the PTQ/QAT split, the first-pass definitions of GPTQ / AWQ / GGUF / GGML, the outlier problem, and the base memory tables. CS-11 is the *deployment* half: the GPTQ Hessian arithmetic, the AWQ correction (W4A16, post-training), the real GGUF CLI + `imatrix` + chat-template traps, KV-cache quantization, the W8A8…W4A4 precision lattice, QLoRA/NF4 algebra, serving flags, and the honest evaluation protocol. If you already quantize models weekly, start at CS-11 and come back here for §2.

---

## §0 Executive Summary

**The one-sentence version:** quantization replaces a continuous range of floating-point numbers with a small fixed grid of integers plus one scale (and optionally one zero-point) per tensor, group, or channel — and *everything* hard about it is deciding how wide that grid must be, and what to do when a handful of values do not fit.

Ten load-bearing claims this module will defend:

1. **Quantization is a rounding decision with three knobs — bits, granularity, and what you protect.** There is no fourth. Every method in this module (PTQ, QAT, GPTQ, AWQ, NF4, k-quants) is a different answer to "which values do I refuse to round?" [55:24]
2. **The affine map is two lines of algebra.** `q = round(x/s) + z` and `ŵ = s·(q − z)`. `s` is the step size of the grid, `z` is the integer that the real number 0.0 maps to. Everything else is bookkeeping. [55:24–56:47]
3. **`Δ = (max(x) − min(x)) / (2^b − 1)` and `MSE = Δ²/12`.** The error is *linear in the range* and *quadratic in the step*. Halving the range quarters the error. One 60× outlier inflates the step 60×, inflates the MSE 3600×, and costs `log2(60) ≈ 5.9` bits of your budget to represent nothing but itself. **This single fact explains why every method in this module exists.**
4. **Signed INT8 is the hardware default, not uint8.** The instructor's asymmetric uint8 examples are pedagogically useful and practically wrong: AVX2, AVX-512 VNNI, NVIDIA Tensor Cores (Turing→Hopper) and ARM/Qualcomm NPUs all execute *signed* 8-bit dot products. A zero-point that lands outside `[−128, 127]` gets clipped and the model breaks — demonstrated numerically in §2.4. [1:16:59–1:18:50]
5. **Granularity is the cheapest accuracy you will ever buy.** Per-tensor → per-channel turns a 24.6% relative error into 1.4% on a heterogeneous weight matrix at the cost of 8 extra floats per row. Group-wise (`group_size=128`, the QLoRA default) is the compromise that fits a kernel.
6. **PTQ = no gradients, QAT = fake-quant + straight-through estimator.** PTQ calibrates on 50–200 unlabelled samples and never touches the loss; QAT inserts `round()` into the forward pass and *lies* about its gradient (`d/dx round(x) := 1`). The video's own numbers: PTQ ≈ 97% of FP32 accuracy on a toy MLP, QAT recovers 97–98% — but the *code* that produces 43% is a bug, not PTQ (§3.6). [1:19:53–1:21:00]
7. **The outlier problem is *the* problem.** LLMs develop activation channels with magnitudes 20–100× the median ("massive activations", attention sinks). Naive per-tensor INT8 dies on them. The five mitigations — mixed precision (LLM.int8()), per-channel granularity, SmoothQuant's migration, AWQ's activation-aware scaling, GPTQ's error compensation — are the entire research field distilled.
8. **GPTQ and AWQ are both *layer-wise post-training* 4-bit weight quantizers, and they differ in what they measure.** GPTQ measures the *input covariance* and solves a second-order reconstruction problem with error compensation; AWQ measures *activation magnitude* and rescales the salient ~1% of channels instead of solving anything. GPTQ is slower and often slightly better at 3-bit; AWQ is ~3× faster to produce and usually better on instruction-tuned and multimodal models.
9. **GGUF is not a quantization algorithm.** It is a *file container* — header, metadata KV, tensor table, tensor data — that replaced GGML's unversioned `.bin` files. The quantization lives in the *k-quant* names (`Q4_K_M` = 4-bit, k-quant, medium mix). Naming the container when you mean the scheme is the single most common terminology error in this field.
10. **The memory arithmetic that gets you hired.** 7B parameters = 14 GB fp16 = 7 GB int8 = 3.5 GB int4 = 1.75 GB at 2-bit — and the KV cache is a *separate*, non-shrinking cost that overtakes the weights at long context (Llama-2-7B: **512 KB per token**; Llama-3.1-8B with GQA: **128 KB per token**).

> **Beyond the video:** the recording is a conceptual tour aimed at beginners. It shows no GPTQ kernel, no `imatrix`, no FP8, and never measures whether the quantized model is actually worse. Those gaps are filled here with `> **Beyond the video:**` callouts and, at full depth, in CS-11.

---

## §1 The Problem This Solves

### 1.1 What breaks without quantization

A 70B-parameter model in bf16 needs **140 GB of weights** — that is two H100-80GB cards before you allocate a single byte of KV cache, and at 2026 rental prices roughly $4–6/hour just to hold the weights resident. A 7B in bf16 needs 14 GB, which does not fit on the 6 GB RTX 3060 the instructor is using [34:03–35:31], nor on the 8 GB card in most laptops, nor on a free Colab T4 with 15 GB once you add activations.

Three things break, in order of severity:

| Constraint | What breaks | The quantization answer |
|---|---|---|
| **VRAM capacity** | Model + KV cache + activations do not fit; you get `torch.cuda.OutOfMemoryError` | Shrink weights 4× (INT8) or 8× (INT4) |
| **Memory bandwidth** | Decode is bandwidth-bound: each token re-reads every weight. A bf16 7B at 400 GB/s theoretical does ~28 tok/s ceiling; measured reality is lower | Reading 4-bit weights is 4× less traffic → ~4× decode throughput |
| **Unit economics** | $/1M tokens scales with the GPU count | Serve the same traffic on 4× fewer GPUs (or the same GPU with 4× the batch) |
| **Edge / CPU / Mac** | No CUDA at all; a fanless laptop must run the model | GGUF + llama.cpp, 4-bit, memory-mapped |

> **The instructor's five goals** [20:13–21:58]: smaller model size, lower memory footprint, faster inference, lower power/energy, and the ability to *fit* on hardware you actually own. Note the absence of "better accuracy" — **quantization never improves a model**, it only decides how much you are willing to lose.

### 1.2 The state of the art before quantization

Before 2022 the only way to make a big model fit was:

1. **Train smaller.** Distillation (CS-08/CS-09) — months of GPU time, a teacher model, and a hard accuracy ceiling.
2. **Prune.** Remove weights entirely. Unstructured pruning gives sparsity that commodity GPUs cannot exploit (no speedup without 2:4 structured sparsity, and even that needs Ampere+).
3. **Offload.** Move layers to CPU RAM or NVMe. Works, but each offloaded layer costs a PCIe round trip — decode throughput collapses.
4. **Use a smaller model.** The pre-2022 default, and the reason a 2021 "GPT-3-class" deployment meant 175B on a cluster.

Quantization is the only one of these that shrinks the model **without touching the architecture, the tokenizer, or the training data**, and — critically — it is a **post-training** operation. You take the checkpoint you already validated and re-encode it.

### 1.3 The naive approach, and exactly why it fails

**Naive approach: "just round every float to the nearest integer."**

```python
w = 1.2345  ->  1     # absolute error 0.2345
w = 0.0234  ->  0     # absolute error 0.0234, relative error 100%
```

Two failures fall out immediately:

- **Scale mismatch.** A weight of `1.2345` rounds to `1` with 19% relative error; a weight of `0.0234` rounds to `0` and is *deleted*. Rounding to integers destroys all information below magnitude 0.5.
- **The fix is to choose the grid, not to round.** Multiply by a scale so the largest magnitude maps to the largest integer, round *there*, and divide back on use: `ŵ = round(w/s)·s`. With `s = 1.2345/127 = 0.00972`, the weight `0.0234` becomes `round(2.407) = 2` and decodes to `0.01944` — a 17% relative error instead of 100%, and the largest weight is exact.

The instructor's price analogy makes the same point at [14:00]: a price of `1299.956` rounded to `1300` loses `0.044`; a currency of `₹87.5793` rounded to `₹87` loses `0.5793` — a 0.66% error that is *fine for a menu and fatal for a currency exchange* [17:14–18:31]. And `π = 3.14159265` vs `3.14` vs `3` [16:00]: the same number, three grids, three very different downstream errors. What matters is not that you rounded, but **what the rounding costs in the units your model actually uses**.

### 1.4 What to quantize — and what you cannot

The instructor walks the architectures [6:14–12:20]: for an ANN you might quantize the weight matrices; for a CNN the convolution kernels; for an LSTM the gate projections; for a Transformer the Q/K/V projections and the FFN matrices — i.e. *all the `nn.Linear` and `nn.Conv` layers*. Biases are quantizable in principle [12:26–13:46], but they are numerically awkward (a bias has one value per output channel, so per-tensor quantization of biases is almost always harmful — every production recipe keeps biases in higher precision).

| Tensor | Quantize? | Why |
|---|---|---|
| Linear / Conv weights | **Yes — always** | 99% of parameters; the entire memory win |
| Embedding / `lm_head` | Usually **no** (or 8-bit) | Lookup tables, not matmuls; errors here hit logits directly and quantization noise on a 128k vocab is expensive |
| LayerNorm / RMSNorm weights | **No** | Tiny (2 × hidden per layer) and numerically sensitive |
| Biases | **No** | One value per channel; per-tensor quantization destroys them |
| Activations | Depends on the WxAy scheme | The outlier problem lives here (§7) |
| **Softmax / attention scores** | **No** | The instructor flags this as the special hard case [12:26]: softmax output is a probability distribution in `[0,1]` with a long tail; 8 bits of uniform grid on `[0,1]` cannot represent `1e-4` attention weights, and those tiny weights are exactly what long-context retrieval depends on |
| KV cache | Yes, in production — FP8 | §7.7 and CS-11 §4.9 |

> **Correction:** the video's slide deck lists GPTQ as "**Gradient** Post-Training Quantization". It is not. GPTQ is **G**enerative **P**re-trained **T**ransformer **Q**uantization — the name of Frantar et al., ICLR 2023, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers". It is a *second-order, error-compensating* PTQ method and involves no gradients at inference or at quantization time. Interviewers use this acronym as a filter; do not repeat the "Gradient" expansion.

---

## §2 The Mathematics of Quantization

> CS-11 assumes you can do everything in this section by hand. It is the highest-yield 30 minutes in the module.

### 2.1 The affine quantizer, formally

A **quantizer** is a pair of maps. The forward map takes a real value to an integer code; the reverse (dequantization) map takes the code back to a real value.

```
Forward:    q  = round(x / s) + z          q ∈ [q_min, q_max] ⊂ ℤ
Reverse:    x̂  = s · (q − z)                x̂ ∈ ℝ
```

| Symbol | Name | Meaning | Units |
|---|---|---|---|
| `s` | **scale** (step size) | The real-number width of one integer step | same as `x` |
| `z` | **zero-point** | The integer code that represents real `0.0` | integer |
| `q_min`, `q_max` | **quantization range** | The integer grid limits, `q_max − q_min + 1 = 2^b` levels | integers |
| `b` | **bit-width** | Bits per element: 8, 4, 3, 2 (also 16 = no-op, and FP8) | bits |
| `x̂` | dequantized value | What the kernel actually computes with | same as `x` |

The instructor writes the same two equations at [55:24–56:47] with `s = scale` and `z = zero point`, and works the example by hand.

**Why `z` exists.** Without it, the grid is forced to be symmetric about zero (`z = 0`). If your tensor's values all live in `[10, 20]`, a symmetric grid must span `[−20, 20]` and half your 256 codes are wasted on a region with no data. The zero-point slides the grid so that it covers exactly `[min, max]`.

**Setting the parameters** — three standard recipes:

| Recipe | Formula | When |
|---|---|---|
| **Min-max (affine)** | `s = (max − min)/(q_max − q_min)`, `z = round(−min/s)` | Default for weights and for activations in PTQ |
| **Symmetric (abs-max)** | `s = max|x|/q_max`, `z = 0` | Default for LLM weight quantization and all signed-int8 hardware paths |
| **Percentile / MSE-optimal** | Clip at the 99.9th percentile, or search a grid of clip values and minimize `‖WX − ŴX‖²` | Outlier-heavy activation quantization; GPTQ is the extreme case (its whole algorithm is an MSE-optimal search, §5.2) |

> **Beyond the video:** the **Lloyd-Max** quantizer is the information-theoretic optimum: for a known source distribution, the reconstruction levels are the *centroids* of the quantization cells, not their midpoints. NF4 (QLoRA) is a hand-built Lloyd-Max codebook for `N(0,1)` — 16 levels at the quantiles `Φ⁻¹(k/16)` — which is why it beats uniform INT4 on normally-distributed weights. `bitsandbytes` stores them as a lookup table, and the "computation dtype" is what the dequantized values are cast to.

### 2.2 Two families: symmetric and asymmetric

| Property | **Symmetric** | **Asymmetric** |
|---|---|---|
| Zero-point | `z = 0` always | `z = round(−min/s)`, generally ≠ 0 |
| Grid | `[−q_max, +q_max]` | `[0, 2^b − 1]` (uint) or `[q_min, q_max]` (int) |
| Levels used for a one-sided range | Half wasted | All used |
| Signed hardware | Direct | Needs an extra add, or a biased matmul |
| Range fit for LLM weights (≈zero-mean) | Excellent | Marginal benefit |
| Range fit for ReLU activations (non-negative) | Poor | Excellent |
| Extra stored state | `s` only | `s` and `z` |
| Instructor's examples | `[−127, 127]` → scale 0.01575, `q = 78`, `ŵ = 1.229`, error 0.001 [1:06:00–1:09:30] | scale 0.00784, `z = −64` clipped to `0`, `q = 157`, `ŵ = 1.23` [1:09:33–1:14:26] |
| Hardware support | **Universal** — AVX2, AVX-512 VNNI, Tensor Cores, NPUs | uint8 only; no mainstream GPU/int8 tensor core path [1:16:59] |

**The `q_max` question — 127 or 128?** Three conventions appear in the wild and you must know which one your framework uses:

| Convention | Range | Span | Used by |
|---|---|---|---|
| Symmetric, `q_max = 127` | `[−127, 127]` | 254 | `torch.quantization`, most LLM weight quantizers; **leaves `−128` unused** so that negation is exact |
| Symmetric, `q_max = 128` | `[−128, 127]` | 255 | INT8 hardware's natural two's-complement range; asymmetric code that happens to be symmetric |
| Asymmetric uint8 | `[0, 255]` | 255 | `torch.ao` activations, TFLite; **not** GPU-friendly |

The `[−127, 127]` convention is deliberate: `−(−128)` overflows int8, so a kernel that computes `−q` (or an absolute value) is safe only if `−128` never appears. When you read the video's `[−127, 127]` example, that is the convention being used.

### 2.3 The error bound — the equation that explains the whole field

For a uniform grid of step `Δ = s` spanning the closed interval `[min, max]` with `2^b − 1` steps:

```
Δ = s = (max(x) − min(x)) / (2^b − 1)
```

The reconstruction error at any point is at most half a step, and for a value uniformly distributed within its cell the error is `e ~ U(−Δ/2, +Δ/2)`, so

```
max |x − x̂| = Δ/2
E[e]         = 0
MSE          = E[e²] = Δ²/12
RMSE         = Δ/√12 ≈ 0.2887·Δ
```

**Read those two consequences carefully, because they are the two levers of the entire field:**

| Lever | Effect on `Δ` | Effect on MSE | Example |
|---|---|---|---|
| Add a bit: `b → b+1` | `Δ` halves | MSE ÷ 4 (−6.02 dB) | 4-bit → 5-bit |
| Double the bits: `b → 2b` | `Δ` ÷ `2^b` | MSE ÷ `2^(2b)` | 4-bit → 8-bit: Δ÷16, MSE÷256 |
| **Widen the range by `k`×** | `Δ` × `k` | **MSE × k²** | one 60× outlier → MSE ×3600 |

That third row is why this module is 1500 lines instead of 50. **A single outlier does not cost you a little accuracy — it costs you the square of its magnitude in MSE, and `log2(k)` bits to encode.** An outlier 60× the median of its tensor consumes `log2(60) ≈ 5.9 bits`; at INT8 that is 74% of your entire bit budget spent to faithfully represent one value, leaving effectively 2.1 bits for everything else.

#### Error-bound table for a normalized range

Take a tensor whose values span `[−1, 1]` (so `max − min = 2`):

| Bits `b` | Levels `2^b` | Steps `2^b − 1` | Step `Δ` | Max error `Δ/2` | MSE `Δ²/12` | RMSE |
|---|---|---|---|---|---|---|
| 2 | 4 | 3 | 0.6667 | 0.3333 | 3.70e−2 | 0.1925 |
| 3 | 8 | 7 | 0.2857 | 0.1429 | 6.80e−3 | 0.0825 |
| 4 | 16 | 15 | 0.1333 | 0.0667 | 1.48e−3 | 0.0385 |
| 8 | 256 | 255 | 0.007843 | 0.003922 | 5.13e−6 | 0.002264 |
| 16 | 65536 | 65535 | 3.052e−5 | 1.526e−5 | 7.76e−11 | 8.81e−6 |

Note the *shapes*: from 4 to 8 bits the MSE improves by **288×**, from 8 to 16 bits by **66,000×**. And note the diminishing return going the other way — 2-bit has a step of 0.667 on a range of 2, i.e. **four values to describe the entire distribution**. That is the whole story of why 2-bit is a cliff and 4-bit is nearly free.

> **Beyond the video:** real LLM weight distributions are not uniform, they are approximately Gaussian. For a Gaussian source the optimal step is finer near zero and coarser in the tails, and the achievable distortion scales as `D ≈ c·σ²·2^(−2b)` with a distribution-dependent constant `c`. Empirically for LLM weights the useful rule is: **8-bit is lossless in every practical sense (KL < 1e−3), 4-bit costs 0.01–0.05 perplexity with a good method, 3-bit costs 0.1–0.5, and 2-bit costs 1–5+** unless you use rotation or QAT.

### 2.4 A complete worked example, by hand

Eight weights from one output channel of a `Linear` layer — the same shape the instructor uses in his 3-input / 2-hidden / 1-output network [56:52–58:23], but with a realistic spread:

```
w = [ 0.0234, −0.1456,  0.7891, −1.2345,  0.5123, −0.0678,  0.3345, −0.9123 ]
```

`max|w| = 1.2345`, `min = −1.2345`, `max = 0.7891`.

#### Case A — symmetric INT8, per-tensor (the production default)

```
s = max|w| / 127 = 1.2345 / 127 = 0.0097205
z = 0
```

| # | `w` | `w/s` | `q = round(w/s)` | `ŵ = q·s` | **error** `w − ŵ` | relative error |
|---|---|---|---|---|---|---|
| 0 | 0.0234 | 2.407 | 2 | 0.019441 | +0.003959 | **16.9%** |
| 1 | −0.1456 | −14.98 | −15 | −0.145807 | +0.000207 | 0.14% |
| 2 | 0.7891 | 81.18 | 81 | 0.787358 | +0.001742 | 0.22% |
| 3 | −1.2345 | −127.0 | −127 | −1.234500 | 0.0 | 0.00% |
| 4 | 0.5123 | 52.70 | 53 | 0.515185 | −0.002885 | 0.56% |
| 5 | −0.0678 | −6.97 | −7 | −0.068043 | +0.000243 | 0.36% |
| 6 | 0.3345 | 34.41 | 34 | 0.330496 | +0.004004 | 1.20% |
| 7 | −0.9123 | −93.85 | −94 | −0.913724 | +0.001424 | 0.16% |

**Aggregate error:**

```
max |error| = 0.004004          (worst case = Δ/2 = s/2 = 0.004860, satisfied)
MSE         = 5.649e−6
RMSE        = 2.377e−3
Δ²/12       = 0.0097205² / 12 = 7.874e−6      (the theoretical bound — same order, ✓)
```

**Three things to notice, because they are interview questions:**

1. **The largest-magnitude weight is exact.** It *defines* the scale, so it always lands on ±`q_max`. Quantization error is not uniform across the tensor; it is *proportional to the tensor's dynamic range*, and the biggest values are the luckiest.
2. **The smallest weight has a 16.9% relative error.** This is the mechanism that kills small weights, and it is why per-channel granularity (§4) matters: the small row was collateral damage of the big row's range.
3. **The error is 5.649e−6 against a theoretical 7.874e−6** — for a sample of 8 the empirical MSE underestimates the bound. With 4096 weights it converges to `Δ²/12`. Do not quote the theoretical MSE as an exact prediction per-tensor; quote it as the *expectation over a uniform source*.

#### Case B — asymmetric uint8, per-tensor (the video's second example, generalized)

```
min = −1.2345, max = 0.7891, span = 2.0236
s  = 2.0236 / 255 = 0.0079357
z  = round(−min / s) = round(1.2345 / 0.0079357) = round(155.56) = 156
q  = clamp(round(w/s) + 156, 0, 255)
```

| # | `w` | `round(w/s)` | `q = +156` | `ŵ = (q − 156)·s` | error |
|---|---|---|---|---|---|
| 0 | 0.0234 | 3 | 159 | 0.023807 | −0.000407 |
| 1 | −0.1456 | −18 | 138 | −0.142842 | −0.002758 |
| 2 | 0.7891 | 99 | 255 | 0.785633 | +0.003467 |
| 3 | −1.2345 | −156 | 0 | −1.237967 | +0.003467 |
| 4 | 0.5123 | 65 | 221 | 0.515820 | −0.003520 |
| 5 | −0.0678 | −9 | 147 | −0.071421 | +0.003621 |
| 6 | 0.3345 | 42 | 198 | 0.333299 | +0.001201 |
| 7 | −0.9123 | −115 | 41 | −0.912604 | +0.000304 |

```
max |error| = 0.003621      (slightly BETTER than symmetric)
MSE         = 7.356e−6      (slightly WORSE than symmetric)
```

**This is the most counter-intuitive result in the section and it is worth pausing on.** The step is *smaller* (0.0079 vs 0.0097) because all 256 codes are used instead of 254 spread over a 2× larger symmetric span. Yet the MSE is 30% *worse*, because `z` was rounded to an integer and the dequantization grid is now offset from the true data by up to `s/2 = 0.004`. Look at the errors in the table: they are all the *same sign and similar magnitude* — a **systematic bias**, not noise. Uniform noise with a smaller variance can be worse than biased noise with a larger one when the bias is correlated with the signal.

The practical lesson: **asymmetric quantization only wins when the range is genuinely one-sided.** Its 4× win on a non-negative tensor is shown in Case D. On an approximately zero-mean tensor like LLM weights, it buys nothing and costs hardware compatibility.

#### Case C — the failure mode: forcing that grid into signed INT8

Keep `s = 0.0079357` and `z = 156`, but store the codes in `int8 ∈ [−128, 127]` (which is what every GPU kernel actually wants):

```
q = clamp(round(w/s) + 156, −128, 127)
```

`z = 156 > 127` → **six of the eight values saturate**:

| # | `w` | raw code | clamped code | `ŵ = (q − 156)·s` | error |
|---|---|---|---|---|---|
| 0 | 0.0234 | 159 | **127** | −0.230135 | **0.2535** |
| 1 | −0.1456 | 138 | **127** | −0.230135 | 0.0845 |
| 2 | 0.7891 | 255 | **127** | −0.230135 | **1.0192** |
| 3 | −1.2345 | 0 | 0 | −1.237967 | 0.0035 |
| 4 | 0.5123 | 221 | **127** | −0.230135 | 0.7424 |
| 5 | −0.0678 | 147 | **127** | −0.230135 | 0.1623 |
| 6 | 0.3345 | 198 | **127** | −0.230135 | 0.5646 |
| 7 | −0.9123 | 41 | 41 | −0.912604 | 0.0003 |

```
max |error| = 1.0192
MSE         = 0.2508
```

**MSE is 44,000× worse than Case A.** The model is destroyed, and — this is the dangerous part — **nothing raises an exception**. The tensor is a valid `int8` tensor; the shapes are right; inference runs; the output is quietly garbage. The instructor hits the mirror image of this in his third example [1:14:28–1:16:12], where the zero-point `−64` is *kept* because it happens to fit the symmetric range in use.

> **Correction:** the instructor's asymmetric example computes a zero-point of `−64` and clips it to `0` on the grounds that "uint8 cannot represent negative values". Clipping the *zero-point* is the right instinct for the wrong reason, and the clip is not harmless. If `min(x) > 0` the true `z` is negative, and clamping it to 0 shifts the entire grid — every dequantized value gains a systematic offset of up to `z·s`. The correct handling of a non-negative tensor is to (a) accept the offset when the tensor's own range dominates it, as in Case D below, or (b) keep the symmetric signed scheme and accept the wasted half-range. What you must never do is clip `z` into a *narrower* signed range — that is Case C, and it is a silent 4-order-of-magnitude regression.

#### Case D — where asymmetric genuinely wins: a one-sided tensor

The same weights, all made positive (a ReLU'd activation, or a post-softmax attention slice):

```
w⁺ = [ 0.0234, 0.1456, 0.7891, 0.5123, 0.0678, 0.3345, 0.9123, 0.2712 ]
```

| Scheme | `s` | `z` | max error | MSE |
|---|---|---|---|---|
| Asymmetric uint8 | 0.0034859 | round(−0.0234/s) = −7 → clipped to 0 | **0.001568** | **9.132e−7** |
| Symmetric int8 | 0.0071835 | 0 | 0.003149 | 4.537e−6 |

**Asymmetric is 5× better in MSE** — because a symmetric grid on `[0, 0.9123]` spends half its codes on `[−0.9123, 0)` where there is no data. This is exactly why `torch.ao.quantization` uses uint8 asymmetric for **ReLU activations** and symmetric int8 for **weights** [1:16:59–1:18:50]. Match the scheme to the tensor's distribution; do not pick one scheme for the whole model.

#### Case E — group-wise, the same eight weights

Symmetric int8, but the scale is computed per **group** (`group_size = 2`, then `4`) instead of per tensor:

| Granularity | Scales used | max error | MSE | vs per-tensor |
|---|---|---|---|---|
| Per-tensor | 0.0097205 | 0.004004 | 5.649e−6 | 1.00× |
| Per-group, `g = 4` | 0.0097205, 0.0071835 | 0.003959 | 5.449e−6 | **1.04×** |
| Per-group, `g = 2` | 0.0011465, 0.0097205, 0.0040339, 0.0071835 | 0.003123 | 1.701e−6 | **3.32×** |
| Per-channel (row) | — | see §4.3 | — | up to 17× |

Two honest observations:

- **`g = 4` barely helps here** (1.04×) because the group containing `−1.2345` still needs the big scale, and the group containing `−0.9123` has its own large value too. Group-wise quantization buys accuracy in proportion to how *locally homogeneous* the weights are.
- **`g = 2` gives 3.32×**, because grouping `[0.0234, −0.1456]` together produces a scale of 0.0011465 — a grid `6.8×` finer than the per-tensor one for those two weights, and the small weight's error drops from 0.003959 to 0.000471 (an 8.4× improvement on the element that was worst).

The general shape of this trade-off — and where `group_size = 128` comes from — is §4.

### 2.5 The complete quantization pipeline, as arithmetic

```
  x (fp16/bf16/fp32)                                      x̂ (dequantized)
        │                                                        ▲
        │ 1. observe the range                                   │ 5. ŵ = s·(q − z)
        ├──────────────► min, max  (or percentile, or MSE-search)─┤
        │                                                        │
        │ 2. choose s, z                                          │ 4. store q as int8/int4
        │    s = (max−min)/(2^b−1) or max|x|/q_max                │    (packed, 2 or 4 per byte)
        │    z = 0 (symmetric) or round(−min/s)                   │
        │                                                        │
        │ 3. q = clamp(round(x/s) + z, q_min, q_max) ─────────────┘
        ▼
     the whole of Steps 1–2 is "calibration" (§3.3)
     Step 3 is the only step a kernel does at inference, fused with the matmul
```

The dequantization step is **not** usually a separate pass: production kernels (Marlin, ExLlamaV2, bitsandbytes' `gemv_4bit`, llama.cpp's `ggml_vec_dot_q4_K_q8_K`) dequantize *inside* the matmul, on the fly, into the compute dtype. That is why "4-bit" models still run on FP16 tensor cores and why a 4-bit model is not 4× faster than an fp16 one — it is bandwidth-bound at 4× less traffic, compute-bound at the same rate. CS-11 §4.10 has the precision lattice.

---

## §3 The Taxonomy — Glossary, PTQ, and QAT

### 3.1 Core concepts — exhaustive glossary

Every term you will meet in a quantization interview, defined once. Rows marked ★ are the ones that appear in more than half of all screening questions.

| Term | Definition | Why it matters | Common confusion |
|---|---|---|---|
| ★ **Quantization** | Mapping a continuous (or wide) range of values onto a small discrete set of codes | The only compression technique that needs no retraining and no architecture change | Confused with *pruning* (removes weights) and *distillation* (trains a new model) |
| ★ **Dequantization** | Reconstructing an approximate real value from a code: `x̂ = s(q − z)` | The kernel's inner loop; the cost you pay to use small weights | People assume it is a separate pass — it is fused into the matmul |
| ★ **Scale (`s`)** | Real-number width of one integer step | The single parameter that determines your error | Called `scale`, `step`, `delta`, `quantization parameter`, or `d` across papers |
| ★ **Zero-point (`z`)** | The integer code representing real `0.0` | Lets an asymmetric grid use its full range | Assumed always zero; assumed always non-zero |
| ★ **Symmetric** | `z = 0`; grid is `[−q_max, +q_max]` | The only scheme signed int8 hardware executes natively | "Symmetric" refers to the *zero-point*, not the value distribution |
| **Asymmetric** | `z ≠ 0`; grid is `[q_min, q_max]` | Best fit for one-sided ranges (ReLU activations) | Thought to be strictly better because it "uses all levels" — Case B shows it is not |
| ★ **PTQ** | Post-Training Quantization: quantize an already-trained model with no gradient updates | ~minutes and a few hundred unlabelled samples; the default for LLMs | Confused with "no data needed" — PTQ needs calibration data (except bitsandbytes/RTN) |
| **QAT** | Quantization-Aware Training: fine-tune *with* quantization simulated in the forward pass | The only way to get usable 2–3 bit models | Thought to be required for 4-bit; it is not |
| ★ **Calibration** | Running unlabelled samples through the model to *observe* the ranges of weights/activations before fixing `s` and `z` | Without it, PTQ static quantization has no idea how wide the activation grid must be | Confused with *training data*; calibration needs no labels and no loss |
| **Calibration set size** | Typically 50–200 sequences for LLM PTQ, 128–512 for GPTQ/AWQ in the notebooks | Too few → the observed max underestimates the true max → clipping at inference | "More is always better" — beyond a few hundred samples returns flatten |
| ★ **RTN** | Round-To-Nearest: quantize each weight independently with no calibration at all | The trivial baseline every paper is compared against; what bitsandbytes does | Confused with GPTQ, which is also "nearest" but with error compensation |
| ★ **Granularity** | The scope over which one `(s, z)` pair is shared: per-tensor, per-channel, per-group | The cheapest accuracy lever available; the main difference between formats | Thought to be a minor detail — it is the difference between 24.6% and 1.4% error (§4) |
| ★ **Per-channel / per-row** | One scale per output channel (row of a `Linear` weight) | Standard for CNNs and for LLM weights in most kernels | Assumed to be the same as per-group |
| ★ **Group-wise / per-group** | One scale per contiguous block of `group_size` weights along the input dimension | The QLoRA default (`group_size = 128`); GPTQ/AWQ's standard knob | "Smaller group = strictly better" — it is better *and* slower *and* bigger metadata |
| **`group_size = −1`** | One scale for the whole column block (i.e. per-channel) | The "no grouping" setting | Read as "automatic" |
| ★ **Quantization error** | `|x − x̂|` per element; aggregate as MSE or SQNR | The objective every method minimizes, explicitly or implicitly | Reported as a percentage of the *step* rather than the *value* |
| **SQNR** | Signal-to-Quantization-Noise Ratio, `10·log10(σ²_x/σ²_e)` in dB | How quantization quality is reported in signal-processing literature | Conflated with model perplexity |
| ★ **Clipping / saturation** | A value outside `[min, max]` is clamped to the nearest code | Clipping bounds the error but destroys outliers; the core trade-off of calibration | Assumed harmless — Case C shows a 44,000× MSE blowup |
| **Percentile clipping** | Choose `min`/`max` at, e.g., the 0.1/99.9 quantiles instead of the true extremes | Removes the influence of rare outliers on `s` | Thought to lose information; it usually *gains* accuracy |
| **MSE-optimal clipping** | Search over clip values and pick the one minimizing reconstruction MSE | What GPTQ/AWQ approximate with an actual optimization | Confused with percentile clipping, which ignores the data distribution |
| ★ **W8A8 / W4A16 / W4A4** | Weights-8-bit / Activations-8-bit, etc. `W4A16` = 4-bit weights, 16-bit activations, computed on FP16 tensor cores | The *actual* deployment point is `W4A16`, not `W4A4` — kernels dequantize into FP16 | "4-bit model" taken to mean 4-bit arithmetic; almost never true |
| **Weight-only quantization** | Quantize weights, keep activations in fp16 | The dominant LLM deployment scheme; what GPTQ/AWQ/bitsandbytes-nf4 do | Confused with full-integer inference |
| **Activation quantization** | Quantize the layer inputs as well | Required for `W8A8`/`INT8` tensor-core throughput; hard because of outliers | The bottleneck everyone underestimates |
| ★ **Outlier** | A value whose magnitude is far outside its tensor's typical range | The central obstacle in LLM quantization (§7) | Assumed to be rare noise; in LLMs it is systematic and structural |
| ★ **Massive activation** | A specific phenomenon in Transformers: a handful of channels with magnitudes 100–1000× the median, stable across inputs | Explains why naive INT8 fails on *all* LLMs and is absent below ~6.7B params | Confused with attention sinks (related, not identical) |
| ★ **LLM.int8()** | Mixed-precision decomposition: INT8 matmul plus an FP16 path for the ~0.1% of dimensions exceeding magnitude 6.0 | The first PTQ method that made 8-bit LLM inference work | Described as "a rounding scheme" — it is a *decomposition* |
| ★ **SmoothQuant** | Migrate activation outliers into the weights via a per-channel scale `s_j = max|X_j|^α / max|W_j|^(1−α)`, `α ≈ 0.5` | Enables true W8A8; the basis of most 2023–26 W8A8 work | Confused with AWQ, which is weight-only |
| ★ **GPTQ** | **G**enerative **P**re-trained **T**ransformer **Q**uantization: layer-wise second-order PTQ with error compensation | Near-lossless 4-bit for ≥7B models; the workhorse of GPU serving | Expanded incorrectly as "Gradient Post-Training Quantization" |
| **OBQ** | Optimal Brain Quantization — the per-column greedy framework GPTQ scales up | GPTQ's mathematical ancestor | Treated as a synonym for GPTQ (GPTQ is a *batched* OBQ) |
| ★ **Hessian / `H`** | In GPTQ, `H = 2·X·Xᵀ`, the input-activation covariance | It says how much each weight *matters* for the output error | Confused with the training Hessian of the loss |
| **Error compensation** | When a weight is quantized, the resulting error is pushed into the not-yet-quantized weights of the same row | The reason GPTQ beats RTN so decisively at 3–4 bits | Thought to be a post-hoc correction; it is *inside* the loop |
| ★ **`act_order` / `desc_act`** | Quantize columns in order of *decreasing activation importance* rather than left-to-right | Better accuracy (especially for 3-bit); ~10% slower inference | Confused with `group_size`; it changes *which* weights share a scale |
| **`damp_percent`** | Diagonal loading added to the Hessian (`H += damp·mean(diag H)·I`) before the Cholesky inverse | Prevents a singular Hessian from blowing the algorithm up | Left at a non-default value "just in case" |
| ★ **AWQ** | **A**ctivation-aware **W**eight **Q**uantization: identify the salient ~1% of weight channels by *activation* magnitude and protect them by per-channel scaling | Fast to produce, best-in-class at 4-bit on instruction-tuned/multimodal models | Described as pruning — it does **not** remove weights |
| **Salient channel** | A channel whose removal (or quantization) changes the output most; AWQ defines this by activation magnitude, not weight magnitude | The core insight of AWQ: importance is a function of the *input*, not the weight | Measured on weights by mistake |
| **GGML** | Georgi Gerganov's Machine Learning library — a C tensor library, and the original unversioned `.bin` file convention | It is the *ancestor*; the library still exists inside llama.cpp | Used interchangeably with GGUF; they are a library and a container |
| ★ **GGUF** | **GGML Universal File** — a single-file container: header, metadata KV, tensor-info table, tensor data | The delivery format for CPU/Mac/edge inference; what Ollama and LM Studio consume | Called a quantization *method* — it is a *format* |
| ★ **k-quant** | Block-wise quant with a *learned* scale (and min) per super-block, plus a smaller scale per 16/32-weight sub-block | The quality jump that made 4-bit GGUF usable | The `k` read as "kernel" or "kilo" |
| **`Q4_K_M`** | 4-bit k-quant, **M**edium — a mixed scheme keeping some tensors at higher precision | The default everyone should start from | `_M` read as "medium *quality*" (it means medium *size/aggressiveness*) |
| **`imatrix`** | Importance matrix: activation statistics used to weight the quantization error during llama.cpp's `llama-quantize` | 5–15% perplexity improvement at 2–3 bits for ~10 minutes of CPU time | Confused with calibration data (it is *derived* from calibration data) |
| **llama.cpp** | The C/C++ inference engine that consumes GGUF | The reason GGUF exists | Confused with Ollama (a wrapper) or LM Studio (a GUI) |
| **ExLlamaV2 / Marlin** | Fast 4-bit GPU kernels with their own formats | The difference between a 4-bit model that is fast and one that is not | Assumed interchangeable with GPTQ's reference kernel |
| **NF4** | 4-bit NormalFloat: a 16-level quantile codebook for `N(0,1)`, used by QLoRA/bitsandbytes | Information-theoretically optimal for Gaussian weights | Thought to be uniform INT4 |
| **Double quantization** | Quantizing the *scales* themselves (256 scales → one block of 8-bit scales) | Saves 0.373 bits/param — the reason QLoRA fits a 7B in ~6 GB | Thought to hurt accuracy; it does not |
| **bitsandbytes** | Load-time quantization library: `load_in_4bit` / `load_in_8bit`, no calibration, no separate file | The fastest path to a smaller model; the base of QLoRA | Assumed to produce a portable artifact (it does not) |
| **FP8 (E4M3/E5M2)** | 1-byte float formats: 3/2 mantissa bits, 5/4 exponent bits | Native on H100/H200/B200; a *compute* format, and the highest-leverage serving flag after 2025 | Lumped together with INT8; they are different trade-offs |
| **MXFP4 / NVFP4** | Microscaling 4-bit float formats: 32-element (MX) or 16-element (NV) blocks sharing an 8-bit (E8M0) or FP8 scale | The 2025+ frontier: better than INT4 at the same bit budget, native on Blackwell | Confused with NF4 (a codebook, not a format) |
| ★ **KV cache** | Per-token stored keys and values: `2 · L · h_kv · d_head · seq · batch · bytes` | At long context it exceeds the weight footprint; the real long-context memory problem | Forgotten in every "will it fit?" calculation |
| **Rotation (QuaRot / SpinQuant / Hadamard)** | Multiply by an orthogonal matrix so outliers spread across all coordinates | Kills outliers globally; no calibration needed | Thought to change the model (it is exactly invertible) |
| **QLoRA** | 4-bit NF4 frozen base + LoRA adapters + paged optimizers | The one case where you *train* on a quantized model | Confused with QAT (QLoRA does not train the quantizer) |
| **Quantization simulation** | Quantize → immediately dequantize, and run inference in fp32 to *measure* the damage | The only way to see quantization error without a kernel | Confused with actual integer inference |

### 3.2 The taxonomy: where every method sits

```
QUANTIZATION
│
├── WHEN?  ─────────────────────────────────────────────────────────────────────┤
│   ├── PTQ  (Post-Training Quantization)   no gradients, minutes, calibration
│   │    ├── Static   → activations quantized with FIXED ranges from calibration
│   │    │              (needs data; fastest inference; torch.ao default, TensorRT-LLM)
│   │    ├── Dynamic  → activations quantized ON THE FLY per batch
│   │    │              (no data; slower; torch.quantization.quantize_dynamic)
│   │    └── LLM-specialised
│   │         ├── RTN / bitsandbytes  (no calibration at all)
│   │         ├── GPTQ      second-order + error compensation   (W4A16)
│   │         ├── AWQ       activation-aware scaling            (W4A16)
│   │         ├── SmoothQuant  activation→weight migration      (W8A8)
│   │         └── HQQ / QuIP# / rotation-based  (calibration-free / 2-bit)
│   │
│   └── QAT  (Quantization-Aware Training)  gradients, hours–days, labelled data
│        ├── Fake quantization + STE  (torch.ao, torchao)
│        ├── Learnable scales / LSQ
│        └── QAT-trained releases (Gemma 3 QAT, EfficientQAT 2-bit)
│
├── WHAT?  ─────────────────────────────────────────────────────────────────────┤
│   ├── Weight-only     W4A16, W8A16   → memory + bandwidth win        (GPTQ, AWQ, NF4)
│   ├── Weight+act      W8A8, W4A8     → + compute win                (SmoothQuant, TensorRT)
│   └── Weight+act+KV   W8A8 + FP8 KV  → long-context memory win       (production 2026)
│
└── HOW COARSE?  ───────────────────────────────────────────────────────────────┤
    per-tensor  →  per-channel (per-row)  →  per-group (32/64/128)  →  per-element
    cheapest metadata                                                        largest
    worst accuracy                                                           best accuracy
```

The instructor's framing [1:19:53–1:21:00] is the top branch: **PTQ and QAT are the two families**, and everything named in this module is a member of one of them. GPTQ and AWQ are PTQ. LLM.int8() is PTQ. The `prepare_qat` → `convert` flow in the notebooks is QAT. QLoRA is *neither* — it trains adapters on top of a PTQ-quantized base, and its quantizer is never updated.

### 3.3 PTQ — mechanism, cost, and where it breaks

**Mechanism, in five steps:**

1. **Load the trained model in fp16/fp32.** No optimizer state, no labels.
2. **Collect calibration statistics.** Run 50–200 unlabelled sequences through the model with forward hooks capturing `min`/`max` (or a running histogram) for every tensor you intend to quantize. The instructor's hook-based `get_activation_min_max` in `Model_Quantization_Final.ipynb` is exactly this, on a toy MLP; the calibration batch is `X_calib = X_train[:100]` [1:24:40–1:26:00].
3. **Fix the quantization parameters.** For each tensor, compute `(s, z)` from the observed range — min-max, percentile, or an optimization.
4. **Replace fp32 weights with integer codes** and record the parameters. In `torch.ao`, this is the `convert` step; in GPTQ/AWQ it is a save step producing a new checkpoint.
5. **Optionally validate.** Measure the reconstruction error and compare outputs on held-out data — the step almost everyone skips (§12).

**Why it is cheap:** no backward pass, no optimizer state, and — for weight-only methods — no labels at all. GPTQ on a 7B takes minutes on a single A100; AWQ takes a similar order. The memory peak is the fp16 model itself, which is the real constraint: quantizing a 70B needs ~150 GB of host RAM/VRAM because you must hold the fp16 weights while computing.

**Where it breaks:**

| Break | Symptom | Root cause |
|---|---|---|
| Activation range from calibration is wrong | Accuracy is fine on calibration data, worse in production | Calibration corpus does not resemble production traffic (a legal-document model calibrated on WikiText) |
| Outliers inflate the activation scale | Everything except the outliers quantizes to 2–3 effective bits | The outlier problem (§7) |
| Small `group_size` chosen for accuracy | Model trains fine, serves 3× slower | Group metadata overhead in the kernel (§4.4) |
| Per-tensor quantization of heterogeneous weights | One row of the weight matrix is destroyed | Case A/B in §2.4 |
| PTQ applied to a 1–2B model | Much larger damage than on a 7B+ model | Small models have less redundancy and *fewer* outliers but sharper weight distributions |

> **Correction:** the widely repeated claim that "outliers only appear above ~6.7B parameters" is true of *massive activations* (the 100–1000× channels) but false as a general statement about quantization difficulty. Small models (0.5B–3B) are often *harder* to quantize at 4-bit than 7B models, because they have less parameter redundancy per unit of capability. If you are quantizing a 1B model, expect a larger quality delta than the 7B benchmarks in the GPTQ/AWQ papers suggest — and validate it yourself.

### 3.4 QAT — fake quantization, the straight-through estimator, and learnable scales

**The problem QAT solves.** A quantizer contains `round()`. `round()` has zero gradient almost everywhere and is undefined at the integers:

```
d/dx round(x) = 0   for all x ∉ ℤ
```

If you insert `round()` into the forward pass and backpropagate honestly, **every gradient that reaches the quantizer's input is multiplied by zero**, and no weight upstream of it receives any training signal. The model cannot learn to be robust to quantization because it cannot learn at all.

**The straight-through estimator (STE)** — the fix, and the reason QAT works:

```
Forward:   q  = round(clamp(x/s, q_min, q_max))          (honest, non-differentiable)
Backward:  ∂q/∂x  :=  1                                   (a lie that works)
```

Formally, the STE *defines* the derivative of the quantizer to be the identity on the interval `[q_min·s, q_max·s]` and zero outside it. In code it is a two-line `autograd.Function`:

```python
class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x.clamp(-127, 127))      # honest forward pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output                          # the identity, not the true derivative
```

**Why defining it as 1 is the right lie** — three arguments you can give in an interview:

1. **It is the gradient of the *identity*, and the quantizer *is* approximately the identity for the quantity that matters.** The reconstruction error `x̂ − x` is bounded by `Δ/2` and is, to first order, uncorrelated with `x`. The best *linear* estimator of `x` given `x̂` is `x̂` itself. Training the weights to be "small enough that `Δ` is small" is exactly what an identity gradient encourages.
2. **It makes `round()` behave as a noisy pass-through, and the network learns to tolerate the noise.** Empirically the STE-trained network shifts its weight distribution toward values that sit near the centers of quantization bins (a genuine, measurable effect), reducing the expected rounding error.
3. **The bias is bounded and the variance is what matters.** The STE gradient is an *unbiased* estimator of the gradient of the *smoothed* quantizer `E_ε[round(x+ε)]`, whose smoothing width is `Δ`. Since SGD is robust to bounded gradient noise, a smoothed-gradient estimator converges to a solution that is near-optimal for the true quantized network.

**Where the STE genuinely fails** (and interviewers probe this): if the quantization bins are *coarse relative to the gradient signal* (2-bit), the gradient the STE passes through is wildly out of scale with the true error, and QAT stalls or diverges. That is why 2-bit QAT is a research topic requiring tricks beyond the vanilla STE (learnable scales, per-group QAT, rotation).

**The prepare/convert flow.** QAT in `torch.ao` is a two-phase API, and the video's notebook is the canonical skeleton:

```
  fp32 model
      │  torch.quantization.fuse_modules(model, [['fc1','relu1'], ...])   ← fuse Conv+BN/ReLU
      │  model.qconfig = get_default_qat_qconfig('fbgemm')                 ← or 'qnnpack' for ARM
      │  torch.quantization.prepare_qat(model, inplace=True)               ← insert FakeQuantize + observers
      ▼
  QAT model      ← TRAIN THIS: the forward pass rounds, the backward pass uses the STE
      │  ... N epochs of fine-tuning, usually 10–20% of the original training budget,
      │      with the LR schedule *restarted* (a cosine from ~1e-5 to 0)
      │  torch.quantization.convert(model.eval(), inplace=False)           ← freeze scales, pack int8
      ▼
  genuinely-int8 model  ← runs on fbgemm (x86) / qnnpack (ARM) / TensorRT
```

The two-phase structure exists because the observers must *see* data before their ranges are frozen. `prepare_qat` puts a `FakeQuantize` module in front of every quantizable tensor; during training that module quantizes-then-dequantizes with the STE so the model feels the error while the ranges are still moving; `convert` then replaces the whole thing with real integer ops using the frozen scales.

**Fake quantization, concretely.** The video's notebook prints an int8 round-trip of a 5-element tensor [1:26:36 onwards]. Reproduced with the notebook's own function:

```
input      : [ 0.1200, −0.8500,  0.5600, −0.3300,  0.9100]
int8 codes : [  127,      0,     127,      75,     127]     ← symmetric, s = 0.00717, saturating
dequant    : [ 0.0276, −0.8489,  0.0276, −0.3313,  0.0276]
abs error  : [ 0.0924,  0.0011,  0.5324,  0.0013,  0.8824]
```

Read that table as a warning: **three of the five values are saturated**, because the tensor's range is dominated by the values near `0.9` while one value sits at `−0.85` and the grid is being fit badly. A fake-quantization pass that produces errors of `0.88` on a signal of amplitude `0.9` is not measuring quantization — it is measuring a broken scale. This is the same failure as Case C in §2.4, and it is the reason the notebook's simulation numbers are so bad (§3.6).

> **Beyond the video:** **learnable scales** (LSQ, PACT) go one step further: instead of fixing `s` from the observed range, make `s` a *trainable parameter* with its own gradient. LSQ's gradient is `∂x̂/∂s = −x/s + round(x/s)` for `|x| < s·q_max`, and it is computed with the STE's identity assumption — so `s` learns to shrink until the clipping error balances the rounding error. PACT adds a learnable *clip* level and a ReLU-like activation. In 2026 the production answer is usually not LSQ but `torchao`'s `QATConfig` with per-channel scales and a rotation applied first.

### 3.5 PTQ vs QAT, head to head — with numbers

The instructor's comparison tables at [1:26:36–1:28:31] and his own toy-MLP experiment:

| Dimension | **PTQ** | **QAT** |
|---|---|---|
| Training required | No | Yes — full fine-tune, quantizer in the loop |
| Data required | 50–200 unlabelled samples (calibration) | The original labelled training set (or a large unlabelled set + the task loss) |
| Time (7B, 1×A100) | **20–60 min** (GPTQ), **10–30 min** (AWQ) | **1–5 days** |
| Time (70B) | 4–12 h (needs ~150 GB host RAM) | Weeks; multi-GPU |
| Time (toy MLP, CPU) | seconds | 2000 epochs ≈ minutes [1:28–2:0x] |
| Accuracy at 8-bit | Lossless in practice | Lossless |
| Accuracy at 4-bit | Near-lossless for ≥7B with GPTQ/AWQ; 0.02–0.10 ppl | Better — recovers most of the residual gap |
| Accuracy at 3-bit | Usable with `group_size=32` + `act_order`; 0.2–0.6 ppl | Good |
| Accuracy at 2-bit | **Broken** (perplexity explodes) unless rotation/QAT | The only way to 2-bit |
| Video's toy-MLP result | `quantize_dynamic` → **0.97 accuracy, unchanged** (the good PTQ path) | `prepare_qat` + `convert` → **97–98%** |
| Video's other PTQ result | manual per-tensor static PTQ → **43%** — a *buggy* simulation, see §3.6 |  |
| Ease | One script, no training infra | Training infra, LR schedule, QAT API |
| Toolkits | AutoGPTQ, AutoAWQ, optimum, bitsandbytes, TensorRT-LLM, HQQ, llama.cpp | `torch.ao.quantization`, **`torchao`** (`QATConfig`), Intel Neural Compressor, TensorRT-LLM, OpenVINO, Brevitas |
| Who ships it | Almost everybody (every quantized HF/GGUF checkpoint) | Model owners only — Google shipped QAT-trained Gemma 3 at 1B/4B/12B/27B |
| Debuggability | Easy — compare two checkpoints | Hard — the failure is inside training |

**The decision rule, in one line:** *use PTQ until it fails, then use QAT only if you own the model and have the training budget.* In 2026 you rarely run QAT — you download a QAT-trained release (Gemma 3 QAT, EfficientQAT) or you use QLoRA, which is a different thing entirely.

### 3.6 The video's practical results, and a correction

`Model_Quantization_Final.ipynb` builds a deliberately tiny MLP on `sklearn.datasets.make_moons` and compares five paths:

| Path | Config | Reported accuracy |
|---|---|---|
| FP32 baseline | `BigMLP` (2→128→64→64→32→16→8→1), Adam `lr=0.01`, `BCELoss`, 2000 epochs | **0.97** |
| Dynamic PTQ | `torch.quantization.quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)` | **0.97** (unchanged) |
| Manual per-tensor quantization simulation | `quantize_tensor` + `dequantize_tensor` on weights and activations, then fp32 inference | **0.43** |
| Static PTQ | `register_forward_hook` + `get_activation_min_max`, `X_calib = X_train[:100]`, `QuantizedMLP` | **0.43** |
| QAT | `get_default_qat_qconfig('fbgemm')` → `prepare_qat` → train → `convert` | **0.97–0.98** |

Model size in all cases: **0.06 MB** (the network is `2·128 + 128·64 + 64·64 + 64·32 + 32·16 + 16·8 + 8·1 + biases ≈ 32k` params — at 0.06 MB this is being reported in fp32 with 4-byte params plus overhead).

> **Correction:** the headline "PTQ drops accuracy from 97% to 43%" is not a measurement of PTQ — it is a measurement of a broken simulation, and repeating it in an interview will cost you. Three specific problems:
>
> 1. **43% is below chance for a binary classifier.** A coin flip scores 50%. A quantization method that lands *below* chance on a 2-class problem has not "lost accuracy", it has **inverted or destroyed the decision boundary** — which is what happens when you quantize the *final logit layer* with a range estimated from a different tensor, or when you apply a uint8 zero-point path to a signed pre-activation and clamp every negative value to zero.
> 2. **The manual path quantizes with a per-tensor scale for every tensor, including the 7-layer network's intermediate pre-activations.** On a 32k-parameter network there is no redundancy to absorb that; on a real LLM there is.
> 3. **The identical 43% for two structurally different methods** (manual simulation and hooked static PTQ) is the signature of a shared bug in the *evaluation or conversion* code path, not of two independent measurement failures.
>
> A correct per-tensor static PTQ on this MLP should land around 0.93–0.97 — a small drop from 0.97, not a collapse. The legitimate conclusions from the notebook are: (a) **dynamic PTQ can be free** (0.97 → 0.97), (b) **QAT recovers the last fractional percent** (0.97 → 0.98), and (c) **a badly-implemented quantization pipeline is catastrophic and silent** — which is the most valuable lesson in the notebook and is true of production systems too.
>
> The same caveat applies to the notebook's fake-quantization table reproduced in §3.4, where three of five values saturate at `q_max`: those numbers demonstrate a mis-set scale, not the intrinsic cost of int8.

---

## §4 Granularity — Per-Tensor, Per-Channel, Per-Group

> CS-11 §4.8 covers the *overhead arithmetic* (bytes of metadata per weight) and which granularities real kernels support. This section is the accuracy side.

### 4.1 The idea in one sentence

**A single scale must cover the entire dynamic range it is shared across. Shrinking the sharing region shrinks the range, and MSE falls with the square of the range.**

From §2.3: `MSE ∝ Δ²` and `Δ = (max − min)/(2^b − 1)`. So if a finer granularity cuts a group's range by 4×, its local MSE falls 16×. That is the entire mechanism, and there is no free lunch beyond it: you pay in metadata bytes, kernel complexity, and inference speed.

### 4.2 The three granularities

```
                       ┌─────────────────────────── one weight matrix W (out × in) ───────────────────────────┐
PER-TENSOR             │ one (s, z) for the whole matrix                                     overhead: 2 floats │
PER-CHANNEL (per-row)  │ one (s, z) per OUTPUT channel (row)               overhead: 2 floats × out_features │
PER-GROUP (g = 4)      │ one (s, z) per 4 contiguous weights along `in`      overhead: 2 floats × out × in / g │
                       └────────────────────────────────────────────────────────────────────────────────────┘
```

| Granularity | Scale covers | Metadata overhead (fp16 scales) | Accuracy | Kernel support |
|---|---|---|---|---|
| **Per-tensor** | the whole tensor | `2·16/8 = 4` bytes total | Worst | Universal — every int8 tensor core, every kernel |
| **Per-channel (per-row)** | one output channel | `2·2 = 4` bytes per row → for a 4096×4096 layer, `4096·4 / 16.7M ≈ 0.001 bits/param` | Good | Standard for CNNs; `torch.ao` default for weights; `llama.cpp` Q8_0 is per-32 not per-row, see below |
| **Per-group, `g=128`** | 128 contiguous input weights within one row | `2·2/128 = 0.03125` bytes/param = **0.25 bits/param** | Very good | GPTQ, AWQ, exllama, Marlin, bitsandbytes-NF4 (as a codebook) |
| **Per-group, `g=32`** | 32 contiguous weights | **1.0 bits/param** overhead — i.e. a "4-bit" model is really 5 bits | Best in class | GPTQ/AWQ support it; slower |
| **Per-element (per-weight)** | each weight | 16 bits/param — *doubles* the size | Theoretical best | Nobody, except research code |

### 4.3 A worked per-channel example

Two rows of a weight matrix. Row 0 is the 8 weights from §2.4 (span 1.2345). Row 1 is a "quiet" row with span 0.0487 — the kind of row that appears in real networks wherever a feature has small gradients.

```
row 0 = [ 0.0234, −0.1456,  0.7891, −1.2345,  0.5123, −0.0678,  0.3345, −0.9123 ]
row 1 = [ 0.0121, −0.0234,  0.0456, −0.0089,  0.0312, −0.0487,  0.0278, −0.0156 ]
```

**Per-tensor** (`s = 1.2345/127 = 0.0097205` for both rows):

| Row | codes | max abs error | MSE | worst relative error |
|---|---|---|---|---|
| 0 | `[2, −15, 81, −127, 53, −7, 34, −94]` | 0.004004 | 5.649e−6 | **16.9%** |
| 1 | `[1, −2, 5, −1, 3, −5, 3, −2]` | 0.003959 | 6.474e−6 | **24.6%** |

Row 1's values quantize to single-digit integers. The weight `−0.0089` becomes `−1` → `−0.00972`; the weight `−0.0487` becomes `−5` → `−0.0486`. Every element in the row is coarsely rounded because row 0 contains a `−1.2345`.

**Per-channel** (row 1 gets `s₁ = 0.0487/127 = 0.00038346`):

| Row | codes | max abs error | MSE | worst relative error |
|---|---|---|---|---|
| 1 | `[32, −61, 119, −23, 81, −127, 72, −41]` | **0.000191** | **1.342e−8** | **1.41%** |

**The MSE improves 482×, the worst relative error improves 17.4×, and the cost is 4 bytes per row.**

This is why per-channel quantization is the default everywhere it is supported, and why "use per-channel scales" is the first piece of advice for anyone quantizing a CNN or an LLM.

### 4.4 Group-wise quantization and where `group_size = 128` comes from

Per-channel has a blind spot: it assumes the *input* dimension of a row is homogeneous. In LLM weight matrices it is not — the input channels of an FFN row have wildly different scales, and per-channel (which spans the entire input dimension) cannot adapt. Group-wise quantization splits each row into contiguous blocks of `group_size` weights along the input dimension and gives each block its own scale.

```
row 0, group_size = 4:
  group 0 = [ 0.0234, −0.1456,  0.7891, −1.2345 ]  s = 0.0097205  (set by the −1.2345)
  group 1 = [ 0.5123, −0.0678,  0.3345, −0.9123 ]  s = 0.0071835  (set by the −0.9123)
```

From Case E in §2.4 the measured gains on these 8 weights were: `g=4` → 1.04× MSE improvement, `g=2` → 3.32×. The general empirical law, from the GPTQ and AWQ papers and reproduced consistently in practice:

| `group_size` | Relative MSE at 4-bit (representative) | Extra bits of metadata per weight | Typical use |
|---|---|---|---|
| −1 (per-channel) | 1.0× (baseline) | ~0.001 | Fast path |
| 128 | **~0.55×** | 0.25 | **The default.** GPTQ/AWQ/bitsandbytes |
| 64 | ~0.48× | 0.5 | Slight quality gain, slight slowdown |
| 32 | ~0.42× | 1.0 | **The choice at 3-bit and for small models** |
| 16 | ~0.40× | 2.0 | Diminishing returns; rarely worth it |

**Why 128 is the default** — three reasons that stack:

1. **The 128 boundary matches a hardware feature.** NVIDIA's `mma` fragments and the 4-bit dequant paths in Marlin and ExLlamaV2 process weights in blocks of 128 or 64; a `group_size` of 128 aligns with the memory access pattern, so the metadata load is amortized over a full cache line.
2. **0.25 bits/param of overhead is affordable.** A "4-bit" GPTQ model at `group_size=128` is really 4.25 bits/param; at `g=32` it is 5 bits — a 17% size increase that erases much of the 8→4-bit gain.
3. **Empirically, the accuracy curve knee is at 128.** Going 128 → 64 buys ~13% less MSE for 2× the metadata; 64 → 32 buys ~13% more. The paper-reported quality delta at 4-bit between `g=128` and `g=32` is small; at 3-bit it is not, and that is where people switch.

**The QLoRA connection.** QLoRA's NF4 uses `blocksize = 64` as the default in `BitsAndBytesConfig` (`bnb_4bit_quant_type="nf4"`, `bnb_4bit_blocksize=64` in the low-level API; the `transformers` wrapper exposes NF4 with double quantization, which quantizes the block scales themselves into 256-value blocks with 8-bit scales, recovering 0.373 bits/param). The interaction — "group-wise for accuracy, double-quant for the overhead" — is what makes a 7B fit in ~6 GB of training memory.

> **Beyond the video:** the two axes of granularity are *not* the same axis. **Per-channel** = one scale per row of `W`; **group-wise** = one scale per block *within* a row. A common interview trap is asking whether `group_size=128` on a `4096×4096` layer means 32 scales or 131,072 scales. The answer is `4096 × (4096/128) = 131,072` — one per *block per row* — because the group is defined along the input dimension for each output channel separately. Getting this wrong changes your metadata estimate by 4096×.

### 4.5 Choosing granularity — the practical rule set

| Situation | Choose | Why |
|---|---|---|
| 8-bit, any model | Per-channel (or per-tensor) | At 8-bit the error is already ~0.4% of range; finer granularity is unmeasurable |
| 4-bit, ≥7B, GPU serving | `group_size = 128` | The standard; kernels are optimized for it |
| 4-bit, <3B model | `group_size = 64` or `32` | Small models have less redundancy; the extra metadata is affordable |
| 3-bit | `group_size = 32` + `act_order=True`/`desc_act=True` | Where the accuracy is actually needed |
| 2-bit | Rotation (QuaRot/SpinQuant) or QAT — no group size saves you | The error is structural, not granular |
| CPU / GGUF | whatever the k-quant gives you (`Q4_K_M` ≈ per-32 with a super-block scale) | llama.cpp decides |
| **Training on top (QLoRA)** | NF4, `blocksize = 64`, double-quant on | The published QLoRA configuration |

**STOP conditions — granularity will not save you if:**

- The tensor has a **channel-level outlier that is structural** (a fixed attention-sink channel present in every input). Per-channel helps; per-*group* does not, because the outlier is inside the group. Only migration (SmoothQuant), rotation, or mixed precision works. → §7
- You are at **2 bits**. Group-wise reduces MSE by a constant factor; you need orders of magnitude.
- Your kernel **does not implement it.** A `group_size=32` checkpoint served by a kernel that only supports 128 silently falls back to a slow path or fails to load. Always check the kernel, not just the format.

---

## §5 GPTQ and AWQ — the Layer-Wise Post-Training Pipeline

> CS-11 §4.2 derives the GPTQ Hessian arithmetic and works the video's example numerically; CS-11 §4.3 gives the AWQ correction (it is **W4A16 post-training**, not a QAT method) and its scaling derivation. This section is the pipeline, the knobs, the code, and the comparison.

### 5.1 The pipeline both methods share

Every layer-wise LLM quantizer does the same six things. The differences are entirely inside step 4.

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ 1. LOAD          the trained model in fp16/bf16, on GPU (or with device_map="auto")   │
│                  memory peak = 2 bytes × params                                        │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ 2. CALIBRATE     feed N sequences (50–200 for GPTQ, 128–512 for AWQ) through the model │
│                  capture, per Linear layer, either X (the input activations) or their  │
│                  second moment X·Xᵀ                                                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ 3. FOR EACH LAYER (in order; no global optimization, no backward pass)                 │
│      solve the layer-local problem independently                                       │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ 4. CHOOSE THE QUANTIZATION PARAMETERS  ← the only place GPTQ and AWQ differ            │
│      GPTQ: quantize columns greedily; after each, propagate the error into the         │
│            remaining columns via H⁻¹ (Cholesky), H = 2XXᵀ                              │
│      AWQ : compute a per-input-channel scale s_j from activation magnitude,            │
│            rescale W (and its inverse into the previous op), then plain RTN            │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ 5. PACK & SAVE   int4 weights + fp16 scales (+ zero-points) + `quantize_config.json`    │
│                  → .safetensors, .pt, or .gguf                                        │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ 6. VALIDATE      load with the matching kernel, generate on held-out prompts, compare   │
│                  against the fp16 model.  ← the step most tutorials omit              │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**Why layer-wise is acceptable.** The true objective is global (`minimize KL(p_fp16 ‖ p_quantized)` over the whole model). Layer-wise quantization instead minimizes a *local* proxy: for each layer, `‖WX − ŴX‖²`, the squared error of that layer's *output* on calibration data. This is far cheaper (one Hessian per layer, no end-to-end coupling) and it works because the local reconstruction error turns out to be an excellent proxy for the end-to-end loss — the empirical result that made post-training 4-bit LLMs possible.

### 5.2 GPTQ — the idea, without the derivation

**Full statement:** GPTQ (**G**enerative **P**re-trained **T**ransformer **Q**uantization) is a *one-shot, layer-wise, second-order* weight quantizer with **error compensation**. It was the first method to quantize a 175B model to 3–4 bits in a few GPU-hours with negligible perplexity loss.

Three ideas, each of which is an interview question:

**(a) Quantize a layer's weights to minimize the layer's OUTPUT error, not the weight error.**

```
classic RTN objective:   min Σ (w_ij − ŵ_ij)²              ← treats every weight as equally important
GPTQ objective:          min ‖W·X − Ŵ·X‖²  =  min tr((W − Ŵ)·H·(W − Ŵ)ᵀ),  H = 2·X·Xᵀ
```

`X` is the layer's input activations, stacked over the calibration set, so `H` is (up to a factor) the **input covariance matrix**. The consequence is the conceptual core: **a weight is important to the extent that its input channel is active.** A large weight multiplying an input that is always zero matters not at all; a small weight multiplying a high-variance input channel matters enormously. `H` encodes exactly that, and RTN has no idea it exists.

**(b) Quantize greedily, one column at a time, and compensate.**

Process the weight matrix column by column. When column `j` is rounded, the error it introduces is `δ_j = w_j − ŵ_j`. Instead of accepting that error, GPTQ *subtracts its effect* from the not-yet-quantized columns:

```
for j in columns:
    q_j  = quantize(w_j)                       # round the current column
    δ_j  = w_j − q_j                           # the error we just introduced
    w_{j+1:} -= δ_j · (H⁻¹)_{j, j+1:} / (H⁻¹)_{jj}   # push it into the remaining columns
```

The result is that the *accumulated* output error stays small even though each individual weight may be rounded coarsely. This is why GPTQ at 4-bit beats RTN at 4-bit by an order of magnitude in perplexity delta, and why GPTQ at 3-bit is still usable while RTN at 3-bit is not.

**Numerical honesty about the "OBQ" lineage.** GPTQ is a *scaled-up* Optimal Brain Quantization (OBQ). OBQ's per-column greedy step requires `O(d³)` work per column, which is intractable for a 4096-wide LLM layer. GPTQ's three contributions are: (i) **lazy batch updates** — update all remaining columns at once with a block of `B=128` columns, using one Cholesky factorization of `H⁻¹`; (ii) a **shared `H⁻¹`** computed once per layer with a damped Cholesky decomposition; (iii) **numerical stabilization** — a diagonal damp of `damp_percent` (typically `0.01`) times the mean diagonal, added before inversion, because `H` is often ill-conditioned or singular when a calibration channel is constant. All three exist for speed and stability, not for accuracy.

**(c) `act_order` (a.k.a. `desc_act`) — quantize in order of importance.**

By default GPTQ quantizes columns left-to-right (input channel 0 first). With `act_order=True`, it first permutes the columns by **decreasing diagonal of `H`** (i.e. by decreasing input-activation variance) and then quantizes in that order, permuting the result back for storage. The columns quantized *last* get the most benefit from error compensation, so giving the *most important* columns that treatment measurably reduces output error. Cost: the permutation destroys the contiguous layout the GPU kernel expects, so inference is ~10% slower, and some older kernels refused to load `act_order` checkpoints at all.

**Calibration.** GPTQ needs calibration data to estimate `H`. The notebooks use a handful of text samples (5 in the video's Falcon example); the papers use 128–256 sequences of 2048 tokens from WikiText or C4. Two rules that matter more than the count:

- **The corpus must resemble production.** `H` is an estimate of the input covariance of *your* traffic. A code model calibrated on English Wikipedia will under-weight the channels your users actually exercise.
- **More helpful: the sequence length.** GPTQ's `H` is accumulated over *all* calibration tokens; 128 samples × 2048 tokens = 262k tokens, which is enough to estimate a 4096×4096 covariance reasonably. 5 samples × 256 tokens is not.

### 5.3 GPTQ hands-on, from the notebook

```bash
# The video's install (LLM_Quantization_GPTQ.ipynb)
pip install auto-gptq optimum accelerate transformers
```

```python
# ── Part 1: load a PRE-QUANTIZED GPTQ checkpoint and run it ─────────────────────────
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

model_name = "TheBloke/Llama-2-7B-Chat-GPTQ"          # a GPTQ checkpoint on the Hub

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    device_map="auto",          # spread across whatever GPUs exist
    use_safetensors=True,       # the storage format GPTQ checkpoints ship in
    trust_remote_code=True,     # auto-gptq may need the repo's modelling code
    use_triton=False,           # Triton kernels are faster but version-fragile; off by default
)

prompt = "What is quantization in machine learning?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

```python
# ── Part 2: quantize your own model ─────────────────────────────────────────────────
import torch
from datasets import load_dataset
from auto_gptq import BaseQuantizeConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "tiiuae/falcon-rw-1b"                       # small enough to do live on a T4
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto")

# 5 calibration texts is what the notebook uses — see the warning below
calibration_texts = [
    "Quantization reduces the memory footprint of large language models.",
    "Post-training quantization requires only a small calibration set.",
    "GPTQ is a second-order, layer-wise quantization algorithm.",
    "The KV cache grows linearly with context length.",
    "AWQ protects the activation-salient channels by rescaling.",
]
dataset = [tokenizer(t, return_tensors="pt") for t in calibration_texts]

quantize_config = BaseQuantizeConfig(
    bits=4,              # target bit-width. 4 is the sweet spot; 3 needs group_size=32
    group_size=128,      # weights per scale. 128 is the standard; 32 at 3-bit
    desc_act=False,      # == act_order. True = better quality, ~10% slower inference
    damp_percent=0.01,   # Hessian diagonal damp; raise to 0.05-0.1 if the Hessian is singular
)                        # (this is the video's exact config)

model.quantize(dataset)                                 # ~minutes for a 1B model

output_dir = "falcon-rw-1b-gptq"
model.save_quantized(output_dir, use_safetensors=True)
tokenizer.save_pretrained(output_dir)

# ── the notebook's own validation loop: same prompts, fp16 vs 4-bit, wall-clock ─────
prompts = ["What is quantization in LLMs?",
           "Explain post-training quantization.",
           "How does GPTQ differ from naive rounding?",
           "Why does a KV cache matter?",
           "What is 4-bit quantization?"]
for p in prompts:
    t0 = time.time()
    ids = tokenizer(p, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=64, do_sample=False)   # greedy = reproducible
    print(f"{(time.time()-t0):.2f}s  {tokenizer.decode(out[0], skip_special_tokens=True)}")
```

Three notes the notebook itself carries, and that you should carry into production:

- **`use_exllamav2=False` / `disable_exllamav2=True`** in the loader is the workaround for the most common `auto-gptq` failure: the ExLlamaV2 kernel raises on a checkpoint it cannot fuse (`ValueError: ExllamaV2 cannot be used with this model`) — usually a `desc_act=True` checkpoint or an unusual head configuration. Disabling it falls back to the slower Triton/CUDA kernels and the model loads.
- **`use_triton=False`** because Triton's API churns between versions; the wheel you installed yesterday may not compile today.
- **`model.quantize(dataset)` needs the dataset on the same device as the model.** Offloading to CPU mid-quantization is what produces the classic `RuntimeError: Expected all tensors to be on the same device`.
- ⚠ **Five calibration texts is a demo, not a recipe.** The notebook's 5 sentences will not estimate a 4096×4096 covariance; the papers use 128–256 sequences of 2048 tokens. Use `--calib-dataset` in `code/08_quantize.py` with 128–512 samples of *your own* data.

### 5.4 AWQ — the idea

**Full statement:** AWQ (**A**ctivation-aware **W**eight **Q**uantization, Lin et al., MLSys 2024) observes that **weight importance is not a property of the weight — it is a property of the activation the weight multiplies**, and that **protecting ~1% of the channels via per-channel scaling recovers most of the accuracy lost by naive RTN at 4-bit**.

**The observation, step by step:**

1. Run calibration data through the model and record the **mean absolute magnitude of the input activations** `X_j` for each input channel `j` of each Linear layer.
2. Sort channels by `|X_j|`. A small number of channels (empirically ~1%, and typically 0.1–1%) carry the large values.
3. **Those channels' weights must not be rounded coarsely.** The instructor's framing: GPTQ hands you the raw ingredients; AWQ is the finished dish — AWQ's whole contribution is a cheap, robust way to find and protect the channels that matter.
4. **Protect them by scaling, not by keeping them in fp16.** For a per-input-channel scale vector `s`, the identity

```
   W·X  =  (W · diag(s)) · (diag(s)⁻¹ · X)
```

means you can multiply a channel's weight column by `s_j` and divide the corresponding *activation* channel by `s_j`, and the layer output is **exactly unchanged in fp16**. Choose `s_j > 1` for a salient channel, and its weight column becomes larger relative to the group's other weights — so after the group's max-abs scale is computed, that column lands on more distinct integer levels and is quantized more accurately. The activation side pays the reciprocal, but the activation side is *not being quantized* (W4A16) — that is the trick, and it is why AWQ is a **weight-only** method.

5. **Then quantize with plain RTN.** No Hessian, no error compensation, no Cholesky. The search for `s` is a small per-channel optimization over a scaling exponent, solved analytically in closed form for a fixed `α`, and `α` is then grid-searched over a handful of values on a small calibration set.

```
search over α ∈ {0.0, 0.1, ..., 1.0}:
    s_j = (max|X_j|)^α / (max|W_j|)^(1−α)        # balance activation and weight magnitude
    Ŵ   = RTN(W · diag(s)) then un-scale
    keep the α with the lowest output MSE on held-out calibration text
```

**Why it beats GPTQ where it beats it:**

| Reason | Detail |
|---|---|
| **Activation, not weight, importance** | GPTQ's `H = XXᵀ` captures the same information *implicitly*, but AWQ uses it *explicitly* and only where it matters — the tail. Instruction-tuned and multimodal models have sharper activation tails, so the explicit treatment wins more. |
| **Generalization** | AWQ's scale search is a 1-D problem per layer; the paper reports it generalizes better across domains than GPTQ's Hessian solve, which overfits the calibration corpus more easily. |
| **Speed** | No Hessian: AWQ quantizes in roughly a third of GPTQ's time and needs less memory. The notebook's 1.1B AWQ run is a few minutes on a T4. |
| **No `desc_act` weirdness** | AWQ's output layout stays kernel-friendly, so it avoids the `desc_act` inference penalty. Its own overhead is a zero-point plus a per-group scale — the `GEMM` version is the fastest. |

**Where AWQ loses:** at 3-bit and below, GPTQ's error compensation pulls ahead, and AWQ's advantage shrinks. And AWQ's "protect 1%" is an *implicit* mixed precision: at very low bit budgets you want explicit fp16 retention, which is what LLM.int8() and the "keep the first/last layers in fp16" recipes do.

### 5.5 AWQ hands-on, from the notebook

```bash
pip install -q -U autoawq transformers accelerate
```

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
quant_path = "tinyllama-1.1b-chat-awq"
quant_config = {
    "zero_point": True,      # asymmetric: one zero-point per group. Slightly better, GEMM only
    "q_group_size": 128,     # weights per scale (note: the flag is q_group_size, not group_size)
    "w_bit": 4,              # WEIGHT bit-width. Activations are NOT quantized — this is W4A16
    "version": "GEMM",       # "GEMM" = fastest GPU kernel; "GEMV" = batch-size-1 path
}

# ── calibration: the notebook uses 10 texts repeated, ~50 samples at seq len 128 ──
calib_texts = [
    "What is quantization in machine learning?",
    "Quantization reduces the memory footprint of large models.",
    # ... domain-diverse samples: use YOUR OWN data in production
] * 10

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False})

model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_texts,          # list[str] — tokenized internally
    max_calib_seq_len=128,           # sequence length of each calibration sample
    max_calib_samples=50,            # papers use 128-512; the notebook uses 50 for speed
    n_parallel_calib_samples=1,      # raise on a big GPU to speed up calibration
)

model.save_quantized(quant_path, safetensors=True)
tokenizer.save_pretrained(quant_path)
```

```python
# ── load and validate an AWQ checkpoint ─────────────────────────────────────────────
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_path = "tinyllama-1.1b-chat-awq"
tokenizer = AutoTokenizer.from_pretrained(quant_path)
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)   # fuse_layers = faster

text = "What is quantization in LLMs?"
ids = tokenizer(text, return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(**ids, max_new_tokens=100)[0], skip_special_tokens=True))
```

**Two failure modes the notebook hits:**

- **`ImportError: cannot import name 'AutoAWQForCausalLM'` after installing `autoawq`** — there are two packages, `auto-awq` (the original research repo, import name `awq`) and `autoawq` (the maintained fork, import name `awq`, class `AutoAWQForCausalLM`). Install `autoawq`, and if both are present, uninstall the other.
- **Calibration OOM at `max_calib_samples=50, max_calib_seq_len=128`** — the fallback path in the notebook drops to 20 samples × 64 tokens. That is a *quality* decision disguised as a memory fix; record the number you used in the model card.

### 5.6 GPTQ vs AWQ — the head-to-head

| Dimension | **GPTQ** | **AWQ** |
|---|---|---|
| Full name | Generative Pre-trained Transformer Quantization | Activation-aware Weight Quantization |
| Paper | Frantar et al., ICLR 2023 | Lin et al., MLSys 2024 |
| Family | PTQ, layer-wise, second-order | PTQ, layer-wise, activation-aware |
| What it measures | `H = 2XXᵀ`, the input **covariance** | `mean|X_j|`, the input **magnitude** per channel |
| Core mechanism | Greedy column rounding + error compensation into remaining columns via `H⁻¹` | Per-channel scaling `W·diag(s)`, `X·diag(s)⁻¹`, then RTN |
| Mixed precision | Implicit (via `H`) | Implicit (via scaling the salient 1%) |
| Precision scheme | **W4A16** (4-bit weights, fp16 activations) | **W4A16** |
| Needs calibration data | Yes (for `H`) | Yes (for activation magnitudes) |
| Typical calibration | 128–256 seqs × 2048 tokens | 128–512 samples × 128–512 tokens |
| Speed to quantize (7B) | ~30–60 min on 1×A100 | **~10–20 min** — no Hessian |
| Memory to quantize (7B) | Peak fp16 model (~16 GB) + Hessian buffers | Peak fp16 model |
| 4-bit quality | Excellent | **Excellent; usually ahead on instruction-tuned & multimodal** |
| 3-bit quality | **Better** (error compensation earns its keep) | Good but behind |
| 2-bit quality | Poor | Poor — use rotation/QAT |
| Inference kernels | ExLlamaV2, Marlin, Triton (`auto-gptq`), llama.cpp | GEMM/GEMV kernels in `autoawq`, Marlin |
| Weird failure | `desc_act=True` checkpoints that older kernels refuse to load | `version="GEMM"` vs `"GEMV"` mismatch → wrong results, not an error |
| Notebook in this course | `LLM_Quantization_GPTQ.ipynb` (falcon-rw-1b) | `LLM_Quantization_AWQ.ipynb` (TinyLlama-1.1B-Chat) |
| Instructor's summary | "GPTQ gives you the raw ingredients" | "AWQ is the final dish" |

**The one-line interview answer:** *both are one-shot, layer-wise, 4-bit weight-only PTQ methods; GPTQ solves a second-order reconstruction problem with error compensation and wins at very low bit-widths, while AWQ identifies the salient channels from activation magnitude and protects them by rescaling, which is cheaper, faster to produce, and generally ahead at 4-bit on modern instruction-tuned models.*

> **Correction:** the video's slide deck describes AWQ as doing "weight rescaling **and pruning** in 128-row blocks". AWQ does **not** prune — no weight is removed, and the model keeps 100% of its parameters. Every channel is quantized to 4 bits; the salient ones are merely given a *better* 4-bit representation by rescaling. Pruning (removing weights) is a different compression family and produces a sparse model, which AWQ does not. Repeating the "AWQ prunes" line in an interview is a fast way to signal that you have only read slides.

> **Beyond the video:** three later developments worth naming. **Rotation** (QuaRot, SpinQuant, and the Hadamard transforms in `torchao`) applies an orthogonal matrix `Q` to both weights and activations so that `‖Qx‖∞` shrinks toward `‖x‖₂/√n` — the outlier is *spread across all coordinates* instead of being protected. This is calibration-free, composes with both GPTQ and AWQ, and is the strongest 2025–26 result at 4-bit and below. **HQQ** does the same job with a per-group zero-point-free scheme solved in closed form, no calibration at all. And **Marlin/ExLlamaV2** are kernels, not algorithms: a GPTQ or AWQ checkpoint runs 2–3× faster under Marlin than under the reference kernel, with bit-identical weights.

---

## §6 GGML vs GGUF — the Container and the Ecosystem

### 6.1 What GGML is, and what it is not

**GGML = Georgi Gerganov's Machine Learning library.** It is a C tensor library written for CPU inference, using quantized block formats and aggressive SIMD (AVX2/AVX-512 on x86, NEON on ARM, Metal on Apple Silicon, and later CUDA/Vulkan backends). It is *not* an inference engine and *not* a file format — it is the numerical substrate underneath one.

Its history is the source of the confusion:

| Era | What existed | What it meant in practice |
|---|---|---|
| 2022 – Aug 2023 | **GGML library + the GGML `.bin` format** | One `.bin` per model, containing only weights. The tokenizer was a *separate* file (`tokenizer.model`), the hyperparameters were *compiled into* the loader, and there was no version field |
| Aug 2023 → | **GGUF** — a new container designed to replace the GGML `.bin` format | The library kept the name GGML (and lives on inside `llama.cpp`); the *file format* was renamed and redesigned |

**Why GGUF replaced GGML's `.bin`** (the video's comparison table, and every one of these is a real failure of the old format):

| Problem with GGML `.bin` | GGUF's answer |
|---|---|
| Hyperparameters were hard-coded in the loader — every new architecture needed a code change + recompile | **Metadata key-value pairs** in the file; a new architecture is a new KV set, no recompile |
| No version field → a file from an older release gave a cryptic failure | **`general.version`** and `gguf_version` in the header; loaders check and refuse cleanly |
| Tokenizer in a separate file → "model + tokenizer" was two downloads and a source of mismatch | **Tokenizer embedded** — vocab, merges, and (later) the **chat template** travel with the weights |
| Only weights in the file → no `rope_theta`, no `n_ctx_train`, no BOS/EOS ids | All of it in the metadata KV block |
| Whole file had to be read into RAM | **Memory-mappable** — the file layout allows `mmap` so the OS pages in what it needs; a 4-bit 7B starts on a 4 GB laptop |
| No extensibility — adding a field broke readers | **Self-describing**: a reader can skip unknown keys because every value carries a type tag |

**GGML in 2026** is the library inside `llama.cpp` (the `ggml` directory is the tensor backend) plus a legacy ecosystem of `.bin` files that still circulates on the Hub (`TheBloke/LLaMa-7B-GGML`). The course's own `gguf_ggml_practical.ipynb` downloads exactly such a `.bin` to demonstrate the old path.

> **Correction:** the video's slide deck expands GGUF as "GPT-Generated Unified Format" in one place and "Gerganov's General Unified Format" in another, and it does the same for GGML. The upstream `llama.cpp` project's own expansion is **GGML = "Georgi Gerganov's Machine Learning"** (later retro-badged "GPT-Generated Model Language" by the community) and **GGUF = "GGML Universal File"**. The lesson is not the etymology — it is that community acronyms in this field are unreliable, and that a slide deck which gives two expansions for the same acronym in the same deck is telling you which claims to verify.

### 6.2 What is inside a GGUF file

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ [ HEADER ]                                                                   │
│   magic       "GGUF" (4 bytes)                                              │
│   version     uint32, e.g. 3                                                │
│   tensor_count, metadata_kv_count                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ [ METADATA KV ]  ── self-describing, typed, skippable                       │
│   general.architecture        = "llama"                                     │
│   general.name                = "TinyLlama-1.1B-Chat"                       │
│   general.file_type           = 15            # which quant type            │
│   llama.context_length        = 2048                                        │
│   llama.embedding_length      = 2048                                        │
│   llama.block_count           = 22                                          │
│   llama.attention.head_count  = 32                                          │
│   llama.attention.head_count_kv = 4           # GQA ratio lives here        │
│   tokenizer.ggml.model       = "llama"                                      │
│   tokenizer.ggml.tokens      = [...]          # the full vocab             │
│   tokenizer.ggml.merges      = [...]          # BPE merges                 │
│   tokenizer.chat_template     = "{{ bos_token }}..."  # ← the silent-killer │
├──────────────────────────────────────────────────────────────────────────────┤
│ [ TENSOR INFO ]  one record per tensor: name, n_dims, dims[], type, offset  │
│   blk.0.attn_q.weight   type=Q4_K   dims=[2048,2048]  offset=0              │
│   blk.0.attn_k.weight   type=Q6_K   dims=[512,2048]   offset=...            │
│                        ↑ per-tensor quant types are allowed (the "_M" in   │
│                          Q4_K_M is exactly this: some tensors kept higher)  │
├──────────────────────────────────────────────────────────────────────────────┤
│ [ TENSOR DATA ]  aligned, contiguous, mmap-able                             │
│   raw blocks in the declared quant formats                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

The **chat template lives in the file**. That single fact causes more production incidents than any quantization error: a GGUF whose `tokenizer.chat_template` is missing falls back to raw text completion, so an instruction-tuned model stops following instructions — *with no error message*, because the model is running fine on badly-formatted input.

### 6.3 Decoding the k-quant names

`Q4_K_M` is four facts stacked into one token. Read it left to right:

| Part | Values | Meaning |
|---|---|---|
| `Q` | `Q` | **Quantized**. `F16`/`F32` are not quantized; `BF16` also appears |
| **bits** | `2`, `3`, `4`, `5`, `6`, `8` | Target bits per weight. `Q8_0` ≈ lossless; `Q2_K` = desperation |
| `_K` | present or absent | **k-quant**: a *hierarchical* block scheme — a **super-block** of 256 weights carries one fp16 scale + one fp16 min, subdivided into 16-weight (or 32-weight) **sub-blocks** each carrying a small quantized scale. Absent `_K` (e.g. `Q4_0`, `Q4_1`, `Q5_0`, `Q8_0`) means the older single-scale-per-32-block scheme |
| `_S` / `_M` / `_L` | small / medium / large | **Only for k-quants.** The *mix*: which tensors are kept at higher precision. `_M` keeps the attention `V` and the FFN `down` projections at 6-bit, for example; `_L` keeps more; `_S` keeps fewer. It is about *size/aggressiveness*, not "medium quality" |
| `_0` / `_1` | 0 or 1 | Legacy suffix. `_0` = symmetric (scale only); `_1` = asymmetric (scale + min) |

| Name | Bits/weight (incl. scales) | Size of a 7B | Quality vs fp16 | Use it when |
|---|---|---|---|---|
| `F16` | 16.0 | 13.0 GB | reference | You have the RAM and want zero risk |
| `Q8_0` | ~8.5 | 6.7 GB | **indistinguishable** (perplexity delta < 0.01) | Never worry about quality |
| `Q6_K` | ~6.6 | 5.3 GB | indistinguishable in practice | You have the RAM |
| `Q5_K_M` | ~5.7 | 4.8 GB | near-indistinguishable (Δppl ~0.02) | Safe choice for a small margin |
| **`Q4_K_M`** | **~4.8** | **4.1 GB** | **Δppl ~0.05 — the sweet spot** | **The default. Start here.** |
| `Q4_K_S` | ~4.6 | 3.9 GB | Δppl ~0.08 | Slightly smaller, slightly worse |
| `Q3_K_M` | ~3.9 | 3.3 GB | Δppl ~0.2 — noticeable | Tight memory only |
| `Q2_K` | ~2.6 | 2.7 GB | Δppl ~1.0–5 — **a visible cliff** | Almost never |
| `IQ*` (imatrix) | varies | — | 5–15% better ppl than the same-size k-quant | You have 10 minutes of CPU to spare |

**Reading the "size of a 7B" column:** these are the *whole* k-quant sizes including scales, which is why `Q4_K_M` is 4.1 GB and not the 3.5 GB that a naive `7e9 × 0.5 bytes` calculation gives. That 0.6 GB difference *is* the group metadata — the granularity cost from §4.4, made visible.

**`imatrix` (importance matrix).** `llama-quantize` accepts a calibration-derived importance matrix and uses it to *weight* the quantization error during the search for scales: an error on a weight that the activations care about is penalized more. It costs about ten minutes of CPU on a small corpus, is only compatible with `IQ*` and k-quants, and buys 5–15% perplexity at 2–3 bits, less at 4. The command is `llama-imatrix -m model-f16.gguf -f calibration.txt -o model.imatrix` followed by `llama-quantize --imatrix model.imatrix model-f16.gguf model-Q4_K_M.gguf Q4_K_M`.

### 6.4 Hands-on: GGUF and GGML, from the notebooks

```bash
# ── gguf_practical.ipynb: build llama.cpp, convert, quantize, run ────────────────────
apt-get install -y cmake build-essential
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j                       # builds `main`, `quantize`, converter deps

pip install -r requirements.txt               # huggingface_hub, sentencepiece, gguf, ...
python3 -m pip install -q huggingface_hub
```

```python
# download the fp16 HF model, then convert it to GGUF at f16, then quantize to k-quant
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir="tinyllama-hf",
    allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
)
```

```bash
# HF -> GGUF (f16).  Note: `convert.py` was renamed to `convert_hf_to_gguf.py` in 2024.
python3 convert_hf_to_gguf.py ./tinyllama-hf \
        --outfile ./tinyllama-1.1b-chat-q4_0.gguf \
        --outtype q4_0                    # f32 | f16 | bf16 | q8_0 | q4_0 ... (legacy types)

# f16 intermediate -> a k-quant (this is the step that actually uses llama-quantize)
./llama-quantize ./tinyllama-1.1b-chat-f16.gguf \
                 ./tinyllama-1.1b-chat-Q4_K_M.gguf Q4_K_M

# run it — the 2024+ binary is `llama-cli`; `main` was the older name
./llama-cli -m ./tinyllama-1.1b-chat-Q4_K_M.gguf \
            -p "What is quantization in LLMs?" -n 100 --temp 0.7
```

```bash
# ── gguf_ggml_practical.ipynb: the LEGACY GGML .bin path, still on the Hub ───────────
wget https://huggingface.co/TheBloke/LLaMa-7B-GGML/resolve/main/ggml-model-q4_0.bin

# the old binary was called `main` and the old format was `.bin` with a separate tokenizer
./main -m ggml-model-q4_0.bin -p "What is quantization in machine learning?" -n 100
```

**Why this notebook pair is worth running once.** The GGUF path is a *three-stage* pipeline (`HF safetensors → f16 GGUF → k-quant GGUF`), and each stage is a place where a wrong flag produces a model that loads and generates plausible text while being subtly wrong. The GGML path demonstrates the old two-file world (`.bin` + a separate tokenizer) and makes the GGUF design decisions obvious by contrast.

**The current, correct commands (2025+ llama.cpp):**

| Task | Command |
|---|---|
| HF → GGUF (f16) | `python -m llama_cpp.convert_hf_to_gguf <hf_dir> --outfile m-f16.gguf --outtype f16` (pip) or `python convert_hf_to_gguf.py ...` (source tree) |
| Quantize | `llama-quantize m-f16.gguf m-Q4_K_M.gguf Q4_K_M [nthreads]` |
| Importance matrix | `llama-imatrix -m m-f16.gguf -f calib.txt -o m.imatrix -ngl 99` |
| Quantize with imatrix | `llama-quantize --imatrix m.imatrix m-f16.gguf m-IQ4_XS.gguf IQ4_XS` |
| Run | `llama-cli -m m-Q4_K_M.gguf -p "..." -n 128 -ngl 99` (`-ngl` = layers on GPU) |
| Serve (OpenAI-compatible) | `llama-server -m m-Q4_K_M.gguf --port 8080 -ngl 99 -c 8192` |
| Ollama | `ollama create mymodel -f Modelfile` with `FROM ./m-Q4_K_M.gguf` |
| Inspect metadata | `llama-gguf m-Q4_K_M.gguf` or `python -m gguf.scripts.gguf_dump m-Q4_K_M.gguf` |

**The GGUF consumer stack:**

```
llama.cpp  ── the reference implementation: llama-cli, llama-server, llama-bench
   │
   ├── Ollama          ── model manager + OpenAI-compatible server + Modelfile
   ├── LM Studio       ── GUI for macOS/Windows/Linux; downloads from the Hub
   ├── llama-cpp-python── Python bindings; what LangChain's LlamaCpp uses
   ├── llamafile       ── single-file executable (model + runtime in one binary)
   ├── KoboldCpp       ── creative-writing front end
   └── text-generation-webui ── the older multi-backend UI
```

> **Beyond the video:** GGUF is *not* GPU-only or CPU-only — `-ngl N` offloads the first `N` layers to the GPU while the rest run on the CPU, which is the standard way to run a 70B `Q4_K_M` (39 GB) on a 24 GB card. It is also not a training format: there is no optimizer state, no gradient path, and llama.cpp has no backward pass for the quantized weights. Converting a GGUF back to HF is lossy-but-possible (`llama-gguf` + a dequantize script), and any fine-tuning you do must start from the *fp16 HF* checkpoint, not from the GGUF.

---

## §7 The Outlier Problem

> CS-11 §4.3–4.5 takes this further: five competing solutions with their math, rotation, and the KV-cache consequences. This section is the mechanism and the mitigations in first-principles form.

### 7.1 The observation that broke naive INT8

Run a per-tensor INT8 quantization of a real LLM's activations and it fails — badly — while the same recipe on a ResNet works fine. The reason is a structural property of Transformers.

**The numbers.** In a 7B Transformer, most activation values in a given layer's input are within `±2`. A small number of *channels* (typically 0.1–1%) reach `±20` to `±200`, and in the largest models, values above `1000` appear. Crucially, these are not random noise:

- They occur in the **same channel indices** across almost every token and every input (the instructor's calibration example at [1:24:40–1:26:00] finds the activation range `[−3.5, +5.7]` for a toy network — the real phenomenon is that the *distribution within that range is not uniform*).
- They are **absent or far smaller below ~6.7B parameters** — they emerge with scale.
- They concentrate in the residual stream, especially around a small number of "**attention sink**" tokens (the first token, and delimiters) that attention heads dump probability onto.

**Why they exist** (the mechanistic story, and a favorite senior interview question): the residual stream is a sum of many layer contributions, and LayerNorm/RMSNorm normalizes it. A channel that carries a consistently large value acts as a *fixed bias* — it can be read by any downstream layer as a constant and it survives normalization because normalization divides by the whole vector's norm. Attention heads that need a "no-op" path learn to attend to a sink token and write into such a channel. The result is a small number of channels doing bookkeeping for the whole network. They are load-bearing.

### 7.2 Why outliers destroy per-tensor INT8 — the arithmetic

Take 100 values distributed in `[−3, 3]` and one outlier at `60`:

| Quantity | Without the outlier | With the outlier | Ratio |
|---|---|---|---|
| `max|x|` | 3.0 | 60.0 | 20× |
| `s = max/127` | 0.0236 | 0.4724 | 20× |
| Step `Δ` | 0.0236 | 0.4724 | **20×** |
| MSE on the *normal* values | (0.0236²)/12 = 4.6e−5 | (0.4724²)/12 = 1.86e−2 | **400×** |
| Relative error on a value of 1.0 | 2.4% | 47% | 20× |
| Bits spent representing the outlier | — | `log2(60) = 5.91` bits of the 8 | **74% of the budget** |

**Effective bit-width is the concept to name in an interview.** With one such outlier, your 8-bit quantizer has `8 − 5.91 ≈ 2.1` bits of *effective* precision for everything else. That is why "INT8 LLM inference" was considered impossible before LLM.int8(), and why the field pivoted to protecting, migrating, or destroying outliers rather than simply clipping them.

**Clipping is not the answer, and it is worth saying why.** Clipping the outlier to `max = 3` would restore the 2.4% relative error on the normal values, but the outlier's own error becomes `57` — a 95% error on a value that some downstream head depends on. One value of magnitude 60 in an activation tensor is not noise; it is a signal. It must be *represented*, just not by spending the whole grid on it.

### 7.3 The five mitigations

Every solution in the literature is one of these five. The table is the cheat sheet; the subsections are the mechanism.

| # | Mitigation | Mechanism | Calibration-free? | Weight or activation? | Where you meet it |
|---|---|---|---|---|---|
| 1 | **Mixed precision** | Keep the outlier dimensions in fp16; INT8 the rest | Yes | Both | LLM.int8(), "keep first/last layers fp16" |
| 2 | **Granularity** | Per-channel / per-group scale so the outlier only taxes its own group | Yes (weights) | Weights | Every 4-bit LLM format |
| 3 | **Migration** (SmoothQuant) | Move the outlier from activations into weights via a per-channel scale | No — needs calibration | Both | W8A8, TensorRT-LLM |
| 4 | **Salience protection** (AWQ) | Identify the ~1% salient channels by activation magnitude and rescale them | No | Weights | AWQ, and every "protect the salient channels" variant |
| 5 | **Compensation** (GPTQ) | Push the error introduced by quantizing a weight into the remaining weights | No | Weights | GPTQ, and every second-order method |
| 6 | **Rotation** (QuaRot/SpinQuant/Hadamard) | Multiply by an orthogonal matrix so the outlier energy spreads over all coordinates | Yes | Both | 2025–26 frontier; §5.6 |

#### 7.3.1 Mixed precision — LLM.int8()

This is `bitsandbytes`' 8-bit path (`load_in_8bit=True`) and the method from Dettmers et al., NeurIPS 2022.

```
For each matmul  Y = W · X  in fp16:
  1. Find the set O = { j : max_t |X[t, j]| > 6.0 }   ← typically 0.1% of the 4096 dimensions
  2. Split  X = [X_O , X_rest]  and  W = [W_O , W_rest]   (column-wise)
  3. Y = matmul_int8(W_rest, X_rest)  +  matmul_fp16(W_O, X_O)
  4. Sum the two results
```

Three details that interviewers test:

- **The threshold is a fixed `6.0`,** not a per-tensor statistic. It comes from an empirical study of where the `|X|` distribution's tail begins across many models.
- **The decomposition is a *decomposition*, not a rounding scheme.** The rest of the tensor is genuinely INT8; only the extracted columns are FP16. This is why it costs 15–25% throughput rather than 100%.
- **It is the *outlier dimensions* that are extracted, not the outlier *values*.** A whole channel — 0.1% of the width — is pulled out for every token, including tokens where that channel is small. That is what makes the extraction a single contiguous slice rather than a gather.

**What it delivers:** 8-bit inference with essentially no perplexity degradation, at ~2× memory saving and ~1.3–1.5× decode speedup over fp16 on the same hardware. It is not a 4-bit method — 4-bit needs `load_in_4bit` + NF4, which is RTN with a smarter codebook and has none of this machinery.

#### 7.3.2 Granularity

Covered in §4, and it is the *cheapest* mitigation: a persistent outlier channel in a per-channel scheme inflates the scale of exactly one row instead of the entire tensor. But it has a hard limit — **per-channel cannot help when the outlier is in the *activation*, because activation quantization happens per-token at inference**, and the channel that is an outlier for one token may be ordinary for the next. That asymmetry is why activation quantization needs migration or rotation, while weight quantization is nearly solved by granularity.

#### 7.3.3 Migration — SmoothQuant

The instructor names SmoothQuant in the LLM-advanced slides as the toolkit for QAT-adjacent activation quantization; the mechanism is worth having exactly:

```
Given a Linear layer Y = W·X with per-channel activation scale  max|X_j| = A_j
                                     and per-channel weight scale     max|W_j| = W_j

choose  s_j = A_j^α / W_j^(1−α)          with α ≈ 0.5

then   Y = (W · diag(s)⁻¹) · (diag(s) · X)
                 ▲                     ▲
        weights absorb the scale   activations are divided by it
```

Set `α = 0.5` and the difficulty is split evenly: the activation outlier is *divided down* and the weight for that channel is *multiplied up*. Weights are easy to quantize (they are static, and per-channel granularity gives them a fine grid); activations are hard. SmoothQuant deliberately moves the hard part into the easy part of the problem.

- **`α = 0.5` is the default; `α = 0.75` pushes more difficulty onto the weights.** Sweep it on your calibration set.
- Because the scale is per-input-channel `j` and can be folded into the *preceding* operation (a LayerNorm weight, or the previous Linear's output), the transform is **free at inference** — no extra kernel, no extra memory.
- The result is a genuinely W8A8 model (both INT8), which is what you need for INT8 tensor-core *compute* on older GPUs and for TensorRT-LLM's INT8 path.

> **Beyond the video:** SmoothQuant's `α` and AWQ's `α` are the same idea in two places. AWQ's `s_j = (max|X_j|)^α / (max|W_j|)^(1−α)` is exactly SmoothQuant's formula applied *for weight-only* quantization, where the goal is not to enable activation quantization but to give the salient weight columns a finer effective grid. Once you see that, the two papers stop looking like competitors.

#### 7.3.4 Salience protection — AWQ

See §5.4. The one-sentence version: **scale up the ~1% of weight columns whose input channels are large, so that after the group's max-abs scale is applied they land on more integer levels; divide the corresponding activations by the same factor, which costs nothing because activations are not quantized.**

#### 7.3.5 Compensation — GPTQ

See §5.2. GPTQ does not detect outliers at all; it *pays for them locally*. When the outlier-adjacent column is rounded and introduces a large error, that error is subtracted from the remaining columns via `H⁻¹`. The outlier's damage is contained inside the layer's own reconstruction, which is why GPTQ at 4-bit achieves near-lossless results without any explicit outlier handling.

#### 7.3.6 Rotation — the 2025–26 answer

Apply an orthogonal `Q` (a Hadamard matrix, or a learned rotation) to both sides: `Y = (WQᵀ)(QX)`. Since `Q` is orthogonal the product is unchanged in exact arithmetic. But the *infinity norm* of the rotated activations shrinks: for a Hadamard matrix, `‖Hx‖∞ ≤ ‖x‖₂`, and a single dominant coordinate's energy is spread over all `n` of them, so `max|(Hx)_i|` drops by up to `√n`. At `n = 4096`, a 100-magnitude outlier can become ~2. That is a per-tensor INT4 activation quantizer that suddenly works — with no calibration and no per-channel bookkeeping. CS-11 §4.4 has the mathematics.

### 7.4 Which outlier mitigation to use — the decision table

| Your situation | First choice | Second choice | Why |
|---|---|---|---|
| 8-bit weights only, quick | **bitsandbytes `load_in_8bit`** (LLM.int8()) | GPTQ 8-bit | LLM.int8() is the method *built* for this, and it is one flag |
| 4-bit weights, GPU serving, quality-first | **AWQ** | GPTQ with `desc_act=True` | AWQ's explicit salience treatment wins at 4-bit on instruction-tuned models |
| 4-bit weights, ≥7B, standard | **GPTQ or AWQ, `group_size=128`** | NF4 for QLoRA only | Both are near-lossless; pick by which kernel you will serve with |
| True W8A8 (INT8 compute) | **SmoothQuant** | rotation + GPTQ | SmoothQuant exists to make both sides quantizable |
| 3-bit or 2-bit | **Rotation (QuaRot/SpinQuant) + GPTQ**, or **QAT** | AWQ + `g=32` | At low bits the problem is structural, not granular |
| Tiny model (<2B) quantized to 4-bit | **`group_size=32` or 64** + `desc_act=True` | AWQ | Small models have less redundancy; every accuracy point costs more |
| Multimodal / vision-language | **AWQ** | GPTQ | The paper's own reported advantage is largest here |
| CPU / Mac / edge | **GGUF `Q4_K_M` (+imatrix)** | `Q5_K_M` if RAM allows | llama.cpp's k-quants already encode block-wise granularity |

**STOP conditions — quantization is the wrong tool if:**

- **You need to fine-tune the weights.** Quantized weights are frozen in every format here. Use QLoRA (adapters over a frozen NF4 base) — that is the one exception, and it is not "training a quantized model".
- **You need a *better* model.** Quantization has one direction of quality change.
- **Your quality gate is already tight at fp16.** Quantization gives you memory, not headroom.
- **You are at 2-bit without rotation or QAT.** The perplexity delta will be measured in whole points, not decimals.

---

## §8 VRAM Arithmetic, the Accuracy-vs-Bits Curve, and When To Use What

> CS-11 §11 extends this with the memory that actually OOMs you: fragmentation, the CUDA context, activation peaks during prefill, and the KV-cache formula per architecture. The tables here are the *base* tables CS-11 refers to.

### 8.1 Data-type sizes — the base facts

The instructor walks the full table [36:14–39:50], and then the "100M parameters at each width" fact table [41:00–42:06]:

| Type | Bits | Bytes/elem | 100M params | 1B params | 7B params | 70B params | Notes |
|---|---|---|---|---|---|---|---|
| `float64` | 64 | 8 | 800 MB | 8.0 GB | 56 GB | 560 GB | Never used in LLMs |
| `float32` / `fp32` | 32 | 4 | **400 MB** | 4.0 GB | 28 GB | 280 GB | Full-precision training, master weights in some recipes |
| `bfloat16` / `bf16` | 16 | 2 | 200 MB | 2.0 GB | 14 GB | 140 GB | **The training default.** 8-bit exponent, 7-bit mantissa — wide range, low precision |
| `float16` / `fp16` | 16 | 2 | **200 MB** | 2.0 GB | 14 GB | 140 GB | 5-bit exponent, 10-bit mantissa — more precision, narrower range; **the inference default** |
| `float8` E4M3 | 8 | 1 | 100 MB | 1.0 GB | 7 GB | 70 GB | 3-bit mantissa: the *weight/activation* FP8 format |
| `float8` E5M2 | 8 | 1 | 100 MB | 1.0 GB | 7 GB | 70 GB | 2-bit mantissa, wider range: the *gradient* FP8 format |
| `int8` | 8 | 1 | **100 MB** | 1.0 GB | **7 GB** | 70 GB | Signed, hardware-native |
| `uint8` | 8 | 1 | 100 MB | 1.0 GB | 7 GB | 70 GB | Asymmetric activations only |
| `int4` / `nf4` / `fp4` | 4 | 0.5 | **50 MB** | 0.5 GB | **3.5 GB** | 35 GB | The deployment point |
| 2-bit (`Q2_K`) | ~2.6 | 0.33 | ~33 MB | 0.33 GB | 1.75 GB | 17.5 GB | The cliff |
| `bool` | 1 | 0.125 | 12.5 MB | 0.125 GB | 0.875 GB | 8.75 GB | Masks only |

**The two facts that carry the table** [23:32, 41:00–42:06]:

- **`1 GB of FP32 → 250 MB at INT8`** — a 4× reduction, because both are 8 bits apart: 32/8 = 4.
- **100M parameters = 400 MB (fp32) = 200 MB (fp16) = 100 MB (int8) = 50 MB (int4).**

**And the three-phase lifecycle** [42:18]: **fp32 for full fine-tuning, fp16/bf16 for mixed-precision training, int8/int4 for inference.** The instructor's distinction between *precision* (how many bits a number gets) and *accuracy* (how close the model's outputs are to correct) [43:10–44:30], illustrated with the darts analogy [51:55–55:08] — a tight cluster on the wrong part of the board is *precise but not accurate* — is exactly the right framing: quantization reduces precision, and what you must measure is whether accuracy followed.

### 8.2 The VRAM formula

```
VRAM_total = W + KV + A + C + F

  W = weights           = P · bits/8 · (1 + overhead)          P = params, overhead = group metadata
  KV = key-value cache  = 2 · L · h_kv · d_head · seq · batch · bytes_per_elem
  A = activations       ≈ batch · seq · hidden · bytes · (a few)   ← spike during PREFILL, not decode
  C = CUDA context + framework overhead ≈ 0.4 – 1.2 GB (GPU-model dependent, not model-size dependent)
  F = fragmentation / allocator slack ≈ 5 – 15% of the above
```

Worked example — **Llama-2-7B, INT4, 4096 context, batch 1**:

```
W  = 7.0e9 · 4/8 · 1.06              = 3.71 GB   (the 1.06 is 0.25 bits/param of group metadata)
KV = 2 · 32 · 32 · 128 · 4096 · 1 · 2 = 2.15 GB   ← as large as 60% of the weights!
A  ≈ 1 · 4096 · 4096 · 2 · 3         = 0.10 GB   (prefill peak; larger with a big batch)
C  ≈                                  0.60 GB    (RTX 3090/4090-class context)
F  ≈ 10%                              0.65 GB
                                     ─────────
VRAM_total                          ≈ 7.2 GB     → does NOT fit an 8 GB card with room to breathe
```

**The lesson from that arithmetic:** the moment you quantize, the *weights stop being the dominant term*. At 4-bit and 4k context the KV cache is 58% of the weight footprint; at 32k context it is **6× the weights** (17.2 GB of KV for a 7B at INT4). This is why CS-11 ranks FP8 KV-cache quantization as the highest-leverage serving flag in 2026, and why "I quantized to 4-bit so it fits" is a claim you must check at your real context length and batch size.

### 8.3 The full VRAM budget table

Weights only, decimal GB (divide by 1.0737 for GiB). `Q4_K_M` includes k-quant block metadata; `int4` is the bare 0.5 bytes/param.

| Model | Params | fp16 | int8 | **int4** | 2-bit | GGUF `Q4_K_M` | GGUF `Q5_K_M` |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 1.1B | 2.2 | 1.1 | **0.55** | 0.28 | 0.62 | 0.75 |
| Llama-3.2-1B | 1.2B | 2.5 | 1.2 | **0.62** | 0.31 | 0.70 | 0.84 |
| Phi-3-mini | 3.8B | 7.6 | 3.8 | **1.90** | 0.95 | 2.13 | 2.62 |
| Mistral / Llama-3.1 | 7–8B | 14–16 | 7–8 | **3.5–4.0** | 1.75–2.0 | 3.9–4.5 | 4.8–5.5 |
| Llama-2-13B | 13B | 26.0 | 13.0 | **6.50** | 3.25 | 7.28 | 8.79 |
| Qwen2.5-32B | 32B | 64.0 | 32.0 | **16.0** | 8.0 | 17.9 | 21.6 |
| Llama-3.1-70B | 70B | 140.0 | 70.0 | **35.0** | 17.5 | 39.2 | 47.2 |

Full-system budget at **4096 context, batch 1, fp16 weights** — i.e. what you need to actually load and generate:

| Model | fp16 weights | KV @4k | +activations | +CUDA ctx | **Total fp16** | **Total int4** | **Total Q4_K_M** | Minimum realistic GPU |
|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 2.2 | 0.09 | 0.05 | 0.60 | **2.9** | 1.5 | 1.6 | 4 GB / any Mac |
| Llama-3.2-1B | 2.5 | 0.13 | 0.05 | 0.60 | **3.3** | 1.7 | 1.8 | 4 GB |
| Phi-3-mini-3.8B | 7.6 | 1.61 | 0.10 | 0.60 | **9.9** | 4.2 | 4.4 | 8 GB @int4 |
| Llama-2-7B | 14.0 | **2.15** | 0.10 | 0.60 | **16.9** | **7.2** | 7.5 | 12 GB @int4 |
| Mistral-7B / Llama-3.1-8B | 16.0 | **0.54** | 0.10 | 0.60 | **17.2** | **5.4** | 5.7 | 8 GB @int4 (tight) |
| Llama-2-13B | 26.0 | **3.36** | 0.15 | 0.60 | **30.1** | **12.0** | 12.4 | 16 GB @int4 |
| Llama-3.1-70B | 140.0 | **1.34** | 0.20 | 0.60 | **142.1** | **38.5** | 40.2 | 2×24 GB, or 1×24 GB + CPU offload |

**Read the KV column against the architecture, not against the parameter count.** A 3.8B Phi-3-mini costs **384 KB/token** while a 7B Mistral costs **128 KB/token** — the *smaller* model has 3× the KV cache, because Phi-3-mini has 32 KV heads and Mistral has 8 (GQA). Likewise Llama-3.1-8B carries a **quarter** of Llama-2-7B's KV per token at the same parameter count. The KV cache is set by `L · h_kv · d_head` — layers, KV heads, head dimension — and nothing else. **When someone asks "will it fit?", the answer starts with the KV formula, not the parameter count.**

### 8.4 The KV cache, in numbers

| Architecture | `L` | `h_kv` | `d_head` | KB per token (fp16) | 4k ctx | 32k ctx | 128k ctx |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 22 | 4 | 64 | 22 | 0.09 GB | 0.74 GB | 2.95 GB |
| Llama-3.2-1B | 16 | 8 | 64 | 32 | 0.13 GB | 1.07 GB | 4.29 GB |
| Qwen2.5-7B | 28 | 4 | 128 | 56 | 0.23 GB | 1.88 GB | 7.52 GB |
| **Mistral-7B / Llama-3.1-8B** | 32 | 8 | 128 | **128** | **0.54 GB** | **4.29 GB** | **17.18 GB** |
| Llama-3.1-70B | 80 | 8 | 128 | 320 | 1.34 GB | 10.74 GB | 42.95 GB |
| Phi-3-mini-3.8B (no GQA) | 32 | 32 | 96 | 384 | 1.61 GB | 12.88 GB | 51.54 GB |
| Llama-2-7B (no GQA) | 32 | 32 | 128 | **512** | 2.15 GB | 17.18 GB | 68.72 GB |
| Llama-2-13B (no GQA) | 40 | 40 | 128 | **800** | 3.36 GB | 26.84 GB | 107.37 GB |

**Memorize the shape of the formula, then the two anchor numbers: 128 KB/token for a GQA 7–8B, 512 KB/token for a 2023-era no-GQA 7B.** Multiply by your context length and your batch size, then decide whether you need FP8 KV.

### 8.5 The accuracy-vs-bits curve

*Representative published ranges for 7B-class models with GPTQ/AWQ. Treat these as the shape of the curve, not as your numbers — re-measure on your model and your data (§12).*

| Bits | Method | WikiText-2 Δperplexity | MMLU Δ | Instruction-following | Verdict |
|---|---|---|---|---|---|
| 16 | fp16 (reference) | 0 | 0 | baseline | — |
| 8 | RTN / int8 / GPTQ-8 | **0.00 – 0.01** | ~0 | no measurable change | **Lossless in practice** |
| 6 | `Q6_K` / GPTQ-6 | 0.01 – 0.03 | ~0 | no measurable change | Lossless |
| 5 | `Q5_K_M` / GPTQ-5 | 0.02 – 0.05 | < 0.3 pt | negligible | Safe |
| **4** | **GPTQ / AWQ `g=128`** | **0.05 – 0.15** | **0.3 – 1.0 pt** | small, task-dependent | **The deployment point** |
| 4 | RTN / naive int4 | 0.3 – 1.0 | 1 – 5 pt | visible | Do not ship |
| 4 | NF4 (QLoRA base) | 0.1 – 0.3 | 1 – 2 pt | acceptable | For *training*, not for serving |
| 3 | GPTQ `g=32` + `act_order` | 0.2 – 0.6 | 1 – 3 pt | noticeable on reasoning/math | Only with a validated gate |
| 3 | RTN | 1 – 4 | 5 – 15 pt | broken | Never |
| 2 | `Q2_K` / GPTQ-2 | **1 – 5+** | **10 – 30 pt** | broken | **The cliff. Rotation or QAT only** |

**How to read the cliff.** The curve is not linear in bits — it is linear in `log(MSE)`, and MSE is `(range/2^b)²`. So each bit buys a constant *ratio* of improvement, and the useful region is where your task's noise floor sits. For an 8-bit quantizer the noise is below the task's noise floor, so it is free. At 4-bit it is above the floor but still below the redundancy budget of a ≥7B model. At 2-bit it is above both. **There is no single "how bad is N-bit" answer — the answer is a comparison between quantization error and the model's redundancy, and redundancy scales with parameters.** That is why the same 4-bit recipe is lossless on a 70B and damaging on a 1B.

### 8.6 Which technique for which deployment

| Deployment target | Format / method | Tool | Bits | Why |
|---|---|---|---|---|
| **Quick local experiment, HF `transformers`** | bitsandbytes `load_in_4bit` (NF4) or `load_in_8bit` | `transformers` + `bitsandbytes` | 4 / 8 | Zero setup, one flag; no calibration; the base of QLoRA |
| **GPU serving, quality-first** | **AWQ** | `autoawq` | 4 | Fastest to produce, best 4-bit quality on instruction-tuned/multimodal |
| **GPU serving, standard** | **GPTQ** | `auto-gptq`, `optimum`, ExLlamaV2/Marlin kernels | 4 | Universal checkpoint ecosystem; better at 3-bit |
| **GPU serving, max throughput (NVIDIA)** | TensorRT-LLM (INT8 / FP8 / NVFP4) | `tensorrt_llm` | 8 / 4 | Compiled engines, in-flight batching; the highest throughput per GPU |
| **GPU serving, max throughput (open)** | vLLM + AWQ/GPTQ/FP8 | `vllm` | 4 / 8 | PagedAttention + continuous batching; the default open serving stack |
| **CPU / Mac / edge / laptop** | **GGUF k-quant** (`Q4_K_M`, `Q5_K_M`) | llama.cpp, Ollama, LM Studio, llamafile | 2–8 | Memory-mapped, runs on anything, no CUDA |
| **Training on top** | **QLoRA** — NF4 base + LoRA adapters | `peft` + `bitsandbytes` + `trl` | 4 (+16 adapter) | **The only case where you "train a quantized model"** |
| **Apple Silicon, best quality** | GGUF + Metal (`-ngl 99`) | llama.cpp | 4–6 | Unified memory means the whole model fits |
| **Long-context serving** | any 4-bit weights **+ FP8 KV cache** | vLLM / TensorRT-LLM | 4 weights / 8 KV | The KV cache is the real cost (§8.4) |

**And the hard rule, from `code/08_quantize.py`:** *a quantized model cannot normally be fine-tuned.* GPTQ, AWQ, GGUF and every INT8 format store integer codes with no gradient path and no optimizer state. The single exception is **bitsandbytes NF4**, because QLoRA trains LoRA adapters *around* the frozen quantized base — the quantized weights never receive a gradient. If you need to change the model's behaviour, the correct order is **fp16 base → LoRA/full fine-tune → merge → quantize**, never the reverse. CS-11 §4.11 has the QLoRA memory algebra.

### 8.7 The decision tree

```
Start: "I need this model to be smaller / faster / to fit."
│
├─ Do I need to TRAIN further?
│   ├─ YES → QLoRA. NF4 (bnb_4bit_quant_type="nf4", double-quant ON, compute dtype bf16)
│   │         + LoRA adapters. Never QAT unless you own the model.
│   └─ NO ↓
│
├─ What HARDWARE will serve it?
│   ├─ CPU / Apple Silicon / no CUDA → GGUF. Start at Q4_K_M; Q5_K_M if RAM allows;
│   │                                  Q6_K/Q8_0 if RAM is free; add an imatrix below 4-bit.
│   ├─ NVIDIA GPU, latency-sensitive, quality-first → AWQ 4-bit (or GPTQ if no AWQ build).
│   ├─ NVIDIA GPU, max throughput / big batch → TensorRT-LLM (FP8 or INT8) or vLLM+FP8.
│   └─ Just experimenting / no conversion budget → bitsandbytes 4-bit or 8-bit at load time.
│
├─ What BIT-WIDTH does the model tolerate?  ← decide by SIZE, not by taste
│   ├─ ≤1.5B params → 4-bit with group_size 32–64; budget for a real quality drop; consider 5–6 bit
│   ├─ 3B–14B      → 4-bit, group_size 128, GPTQ or AWQ. The sweet spot.
│   ├─ 30B–70B     → 4-bit; even 3-bit is often acceptable; 2-bit with rotation
│   └─ >70B        → you probably cannot afford the fp16 conversion at all: use GGUF/CPU, or shards
│
├─ Is the CONTEXT long (>8k)?
│   └─ YES → the KV cache dominates. Add FP8 KV quantization BEFORE going below 4-bit weights.
│
└─ GATE: quantize, then run the evaluation protocol (§12) on held-out DOMAIN data.
    Perplexity alone is not a gate. If it fails: +1 bit, or group_size 128 → 32.
```

---

## §9 Pros · Cons · Limitations · Failure Modes

### 9.1 Pros

| Pro | Magnitude | Evidence |
|---|---|---|
| Weights shrink 2× (int8) to 8× (int4→fp16) | 7B: 14 GB → 3.5 GB | §8.3 |
| Decode throughput rises because decode is bandwidth-bound | ~2–4× at int4 vs fp16 on the same GPU | Reads 4× fewer bytes per token |
| No retraining, no labels, no architecture change | Minutes to hours | §3.3 |
| Runs on hardware you already own | 7B @4-bit fits a 6 GB laptop GPU with a short context | §8.3 |
| Enables a bigger model on the same budget | A 13B @4-bit ≈ a 7B @fp16 in memory, and usually wins on quality | §13 |
| The best-documented, best-tooled compression technique | Every serving stack supports it | §8.6 |
| Composes with everything else | Pairs with LoRA (QLoRA), with distillation, with speculative decoding | — |

### 9.2 Cons

| Con | Magnitude |
|---|---|
| Quality always decreases, sometimes invisibly | §8.5 |
| Quantized weights cannot be trained | §8.6 |
| Quantization is a one-way conversion — you keep both checkpoints | Disk cost doubles |
| Producing a good 4-bit checkpoint needs the fp16 model resident | 70B → ~150 GB host RAM |
| Kernel quality matters as much as algorithm | A "4-bit" model on the wrong kernel is slower than fp16 |
| Not reproducible by default | Different kernel/library/GPU versions give different numerics |
| Adds a moving part to the deploy pipeline | Format + kernel + loader version triangle |

### 9.3 Hard limitations

1. **You cannot recover information that was rounded away.** No post-processing restores it.
2. **2-bit needs structural help** (rotation/QAT). No group size fixes it (§8.5).
3. **Activation quantization is a different problem from weight quantization**, and it is much harder (§7.3.2). W4A16 is easy; W4A4 is a research problem.
4. **GGUF cannot be fine-tuned**, and converting back to HF is lossy.
5. **bitsandbytes produces no portable artifact.** It quantizes at load time; the "model" is the fp16 checkpoint plus a config.
6. **Quality is task-dependent.** A model can hold perplexity and lose JSON compliance (§12).

### 9.4 Silent failure modes — looks fine, is broken

| Silent failure | What you see | What is actually happening | Detection |
|---|---|---|---|
| **Chat template missing from the GGUF** | Fluent, plausible, *non-instruction-following* output | The model is being fed raw text completion prompts | Print the rendered prompt; compare with the HF tokenizer's `apply_chat_template` |
| **`z` clipped or a `q_max` mismatch between save and load** | Output is grammatical but subtly wrong; a few values dominate | Codes saturate (§2.4 Case C) | Compare `s`/`z` recorded in `quantize_config.json` with the loader's |
| **Calibration corpus ≠ production traffic** | Fine on your test set, worse in prod | `H` or the activation ranges are wrong for the real inputs | Evaluate on a sample of real traffic |
| **Perplexity unchanged but format compliance gone** | A green dashboard, a failing product | Quantization error lands exactly on the tokens that carry structure | IFEval / JSON-schema validation / KL divergence (§12) |
| **`version="GEMM"` vs `"GEMV"` mismatch in AWQ** | Slightly worse output, no error | Wrong kernel path selected | Compare against the fp16 model on a fixed prompt set |
| **`desc_act=True` loaded by a kernel that ignores it** | Plausible but degraded output | Column permutation applied at quant time, not undone at load | Check the kernel's supported features, not just the file's |
| **Small model quantized with the 7B recipe** | "It got dumber" with no error | Less redundancy; the recipe was calibrated for a bigger model | Use `group_size=32–64`; re-measure |
| **Quantized, then evaluated with the fp16 tokenizer/config** | Small mismatch everywhere | Tokenizer or `rope_theta` differs between checkpoints | Diff the configs |
| **A "4-bit" model served on a slow kernel** | Correct output, worse latency than fp16 | Fallback kernel selected | `llama-bench`, or check the kernel name at load |

---

## §10 Exceptions, Edge Cases & Gotchas

1. **`q_max = 127` vs `128`.** Different frameworks differ. A checkpoint quantized with one convention and loaded with the other is off by a factor of `128/127` — a 0.8% systematic error that never raises an error. Record the convention next to the checkpoint.
2. **Symmetric quantization of a tensor whose max is 0.** A dead ReLU channel, or a zero-initialized adapter. `s = 0` → division by zero or an all-zero tensor. Every production quantizer guards this with `s = max(|x|)/q_max + eps` — the notebooks' `1e-8` in `scale = (max_val − min_val)/(q_max − q_min + 1e-8)` is exactly that guard.
3. **Bias terms.** One value per output channel. Quantizing them per-tensor is a real accuracy loss for a negligible memory gain — keep them fp16.
4. **The `lm_head` and embeddings.** 128k-vocab embeddings can be 10%+ of a small model's parameters, which makes them tempting. Quantizing them hits the logits directly; most recipes keep them at 8-bit or fp16.
5. **LayerNorm / RMSNorm weights.** Numerically sensitive and tiny. Never quantize.
6. **Quantized models do not fine-tune** — except QLoRA over NF4 (§8.6).
7. **Calibration sample count has a knee.** 0 → 128 samples is a large gain; 128 → 512 is small; 512 → 4096 is unmeasurable. The notebooks' 5–50 samples are demos.
8. **Calibration sequence length matters more than you think.** `H` and the activation ranges are accumulated over tokens; 128 samples × 2048 tokens is 262k tokens. 50 samples × 128 tokens is 6.4k — enough for a 1B model, marginal for a 70B.
9. **Quantizing on CPU is possible and slow.** GPTQ/AWQ reference implementations are CUDA-first. `llama-quantize` is CPU-native and the reason GGUF works on a laptop.
10. **Determinism.** Two runs of GPTQ on the same box can differ slightly (CUDA atomics in the Cholesky/`H` accumulation). Pin seeds *and* library versions, and hash the output.
11. **`device_map="auto"` during quantization.** Spreading the fp16 model across GPU and CPU slows the calibration pass by 10–50×, and some methods refuse to run. Quantize on GPU, then serve with `device_map="auto"`.
12. **A quantized model saved with `safetensors` and loaded without the quantize config is an fp16-shaped model with int4 data.** The `quantize_config.json` / `config.json` `quantization_config` block *is* the model. Version it like code.
13. **GGUF is not one format per model.** `general.file_type` and the per-tensor types in the tensor table determine everything; a `Q4_K_M` file can contain `Q6_K` tensors by design (§6.3).
14. **Ollama and LM Studio pin their own llama.cpp.** A GGUF produced by a newer `llama-quantize` may not load in an older runtime. Ship the runtime version with the model.
15. **A "quantized" model on a CPU without the right SIMD is slower than fp16.** INT4 dequantization on a CPU with no AVX-512 spends more cycles unpacking than it saves.
16. **FP8 on pre-Hopper GPUs is emulated.** `torch.float8_e4m3fn` exists as a dtype, but if the tensor core does not implement it you get a conversion + fp16 matmul — memory saving without the compute saving.
17. **Two GPU generations in one node produce two different numerics** from the same checkpoint. Pin the architecture (`sm_80`/`sm_89`/`sm_90`) in your reproducibility notes.
18. **Quantization error is not additive across layers.** The per-layer `‖WX − ŴX‖²` is small for every layer and the *end-to-end* error is still large, because errors compound through attention and the residual stream. Measure end-to-end.

---

## §11 Cost, Compute & Memory

### 11.1 What the quantization *run* costs

| Model | Method | Hardware | Wall-clock | Peak memory | Cloud cost (A100-80, $2.5/h) |
|---|---|---|---|---|---|
| 1.1B | AWQ | 1×T4 16 GB | 3–8 min | ~4 GB | ~$0.02 (Colab free) |
| 1.1B | GPTQ | 1×T4 16 GB | 5–15 min | ~5 GB | ~$0.03 |
| 7B | AWQ | 1×A100 80 GB | 10–20 min | ~20 GB | ~$0.80 |
| 7B | GPTQ | 1×A100 80 GB | 25–60 min | ~24 GB | ~$2.50 |
| 13B | GPTQ | 1×A100 80 GB | 1–2 h | ~40 GB | ~$5 |
| 70B | GPTQ | 1×A100 80 GB + 128 GB RAM | 4–12 h | ~150 GB | ~$30 |
| 70B | GGUF `Q4_K_M` (CPU) | 16-core CPU, 150 GB RAM | 1–3 h | ~150 GB | ~$1 (spot) |
| 70B | `imatrix` + `IQ` quant | 16-core CPU | +10–30 min | ~150 GB | ~$0.30 |
| any | bitsandbytes | — | **0** | — | **free** — quantizes at load |

**The hidden cost is the fp16 model, not the GPU.** Quantizing a 70B needs ~150 GB of RAM: 140 GB of bf16 weights plus working space. That is the line item that decides whether you can do it on the machine you have.

### 11.2 What quantization *saves* — the only calculation that matters

```
GPU-hours per million tokens ∝ (bytes read per token) = params × bytes_per_param

fp16 7B:  14 GB per token-pass    →  1.0×  (baseline)
int8 7B:   7 GB                  →  0.5×  cost
int4 7B:   3.5 GB                →  0.25× cost  ← if the kernel is bandwidth-bound and efficient
GGUF Q4_K_M on CPU: memory-bandwidth-bound on system RAM, but no GPU cost at all
```

Worked example — **serve a 7B at 30 requests/s, 1000 tokens out, on A100-80GB ($2.50/h)**:

| Precision | Weights | Concurrency on 80 GB (with 4k KV) | GPUs for 30 rps | $/1M output tokens |
|---|---|---|---|---|
| fp16 | 14.0 GB | ~4 (KV 2.15 GB each + activations) | 3 | ~$0.70 |
| int8 | 7.0 GB | ~8 | 2 | ~$0.45 |
| **int4 (AWQ/GPTQ)** | **3.5 GB** | **~14** | **1** | **~$0.23** |
| int4 + FP8 KV | 3.5 GB | ~18 | 1 | ~$0.20 |

**Quantization is a 3× cost reduction on this workload, and it comes almost entirely from *concurrency* (fitting more sequences in VRAM), not from raw kernel speed.** That is the sentence to say in an interview: *weights shrunk 4× means 4× more KV-cache room per GPU, which means 4× the batch, which means 4× the throughput — the kernel itself is not 4× faster.*

### 11.3 The training-side calculation (QLoRA)

```
QLoRA 7B:
  NF4 base weights     = 7e9 × 0.5 bytes / (1 − 0.373/4 double-quant saving) ≈ 3.5 GB
  + LoRA adapters      ≈ 0.05 GB (rank 16, q_proj/v_proj)
  + optimizer (AdamW on adapters only: 2 states × 4 bytes × adapter params) ≈ 0.4 GB
  + activations (gradient checkpointing ON, seq 1024, batch 1) ≈ 1.5–2 GB
  ──────────────────────────────────────────────────────────────
  ≈ 6 GB  → a free Colab T4 (15 GB) runs it comfortably
```

Without quantization the same fine-tune needs ~24 GB (fp16 weights 14 + AdamW states 56 for full FT). **That is the entire reason QLoRA exists.**

---

## §12 Evaluation — How To Know It Worked

### 12.1 Why perplexity is not a gate

Perplexity is a *mean* over next-token log-probabilities on a generic corpus. It is smooth, cheap, and blind to structure. A quantized model can hold perplexity flat while:

- Losing instruction-following (the tokens that carry "as JSON" or "step by step" are a tiny fraction of the loss)
- Losing format compliance (a `{` where `}` was expected costs one token of perplexity and breaks your parser)
- Losing long-context recall (attention weights on distant tokens are exactly what activation quantization damages first)
- Losing chain-of-thought and multi-step math (errors compound; each step is a small perplexity change)

### 12.2 The protocol (an afternoon's work, and it will catch real failures)

```
STEP 0  Build the harness BEFORE quantizing. Freeze:
        - 200 held-out prompts from YOUR domain (not WikiText)
        - the fp16 model's greedy outputs (temperature 0) for all 200  ← the reference
        - the fp16 model's per-token logits on 32 long documents   ← for KL

STEP 1  SIZE AND SPEED
        - file size on disk, VRAM at load, tokens/s at batch 1, t/s at your real batch
        - llama-bench / vLLM's /metrics. Verify the kernel you expected is actually used.

STEP 2  PERPLEXITY (necessary, not sufficient)
        - on held-out DOMAIN text, not WikiText
        - gate: Δppl ≤ 0.1 at 4-bit, ≤ 0.02 at 8-bit. Treat a *drop* in perplexity
          as a red flag (it usually means the tokenizer or template changed).

STEP 3  KL DIVERGENCE — the single best automated signal
        - feed 32 held-out documents through both models, compare next-token
          distributions
        - report MEAN KL, p95 KL, and top-1 agreement, not just the mean
        - gate: mean KL ≤ 0.1, p95 ≤ 1.0, top-1 agreement ≥ 95%
          (a model can have a tiny mean KL and a catastrophic p95 — that is a
           long-tail failure that hits rare tokens, i.e. code and JSON)

STEP 4  TASK EVALUATION
        - IFEval (instruction following), your own JSON-schema-validity rate,
          a needle-in-a-haystack at your maximum context, a 50-example code task
        - gate: no more than 2% absolute drop on any of them

STEP 5  SIDE-BY-SIDE GENERATION
        - 50 prompts, temperature 0, fp16 output next to quantized output, human-read
        - this catches what every metric misses. It is not optional.
```

### 12.3 Minimal eval script

```python
"""Quantization damage measurement: KL divergence, top-1 agreement, p95 KL.

Usage:  python eval_quant.py --fp16 meta-llama/Llama-3.1-8B-Instruct \
                             --quant ./llama-3.1-8b-awq --texts held_out.jsonl
"""
import argparse, json, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(path, quant=False):
    kw = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if quant:
        kw["trust_remote_code"] = True          # AWQ/GPTQ repos ship modelling code
    return AutoModelForCausalLM.from_pretrained(path, **kw).eval()


@torch.no_grad()
def logits_on(model, tok, text):
    """Next-token logits for every position, in fp32 for a fair comparison."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    return model(**ids).logits[0, :-1].float()          # align: predict token t+1 from t


def kl_report(ref_logits, q_logits, chunk=256):
    """Chunked log-softmax over the vocab axis to avoid a 128k-wide tensor blowup."""
    kls, agree = [], 0
    for a, b in zip(ref_logits.split(chunk), q_logits.split(chunk)):
        lp = F.log_softmax(a, dim=-1)                   # reference distribution
        lq = F.log_softmax(b, dim=-1)                   # quantized distribution
        kls.append(F.kl_div(lq, lp, log_target=True, reduction="none").sum(-1))
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
    kl = torch.cat(kls)
    n  = kl.numel()
    return dict(mean_kl=kl.mean().item(),
                p95_kl =kl.quantile(0.95).item(),
                max_kl =kl.max().item(),
                top1_agreement=agree / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--texts", required=True, help="JSONL with a 'text' field")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.fp16)
    ref, q = load(a.fp16), load(a.quant, quant=True)

    rows, agg = [], {"mean_kl": [], "p95_kl": [], "top1_agreement": []}
    for line in open(a.texts, encoding="utf-8"):
        text = json.loads(line)["text"]
        r = kl_report(logits_on(ref, tok, text), logits_on(q, tok, text))
        rows.append(r)
        for k in agg:
            agg[k].append(r[k])

    mean = {k: sum(v) / len(v) for k, v in agg.items()}
    print(f"documents           {len(rows)}")
    print(f"mean KL             {mean['mean_kl']:.4f}   gate <= 0.10")
    print(f"mean p95 KL         {mean['p95_kl']:.4f}    gate <= 1.00")
    print(f"top-1 agreement     {mean['top1_agreement']:.4f}   gate >= 0.95")
    print("\nWORST DOCUMENT (this is where your failure lives):")
    w = max(rows, key=lambda r: r["p95_kl"])
    print(f"  p95 KL {w['p95_kl']:.4f}  max KL {w['max_kl']:.4f}  "
          f"top-1 {w['top1_agreement']:.4f}")


if __name__ == "__main__":
    main()
```

**Reading the output.** If `mean_kl` is fine and `p95_kl` is 5.0, your model is fine on ordinary prose and broken on the rare tokens — which in practice means code, JSON, non-English text, and tool-call syntax. Go up one bit-width or drop `group_size` to 32, then re-run.

> **Correction:** the common advice to "just check that perplexity went up by less than 0.1" is a gate that a *fully broken* instruction-following model passes. Perplexity is an average over ~all tokens of generic text; a model that has lost its ability to emit a valid JSON object loses almost nothing on that average. Every quantized model you ship should have a *task* gate, not only a loss gate.

---

## §13 Comparison Tables

### 13.1 The five methods, head to head

| Dimension | **bitsandbytes (NF4/INT8)** | **GPTQ** | **AWQ** | **GGUF k-quant** | **TensorRT-LLM (FP8/INT8)** |
|---|---|---|---|---|---|
| Family | PTQ, RTN, load-time | PTQ, second-order | PTQ, activation-aware | PTQ, block-wise k-quant | PTQ, kernel-compiled |
| Bits | 4 (NF4), 8 (LLM.int8()) | 2/3/4/8 | 4 (3 experimental) | 2–8 | 8 (INT8/FP8), 4 (NVFP4) |
| Scheme | W4A16 / W8A16 (with fp16 outlier path) | W4A16 | W4A16 | W4A16-ish (CPU) | W8A8, W4A8, W4A4 |
| Calibration needed | **No** | Yes (Hessian) | Yes (activation magnitudes) | No (but `imatrix` helps) | Yes |
| Time to produce (7B) | **0** | 25–60 min | **10–20 min** | 20–60 min (CPU) | 1–3 h (engine build) |
| Output artifact | none (load-time) | `.safetensors` + config | `.safetensors` + config | one `.gguf` | a compiled engine |
| GPU serving | Yes, slower kernels | Yes (ExLlamaV2, Marlin) | Yes (GEMM/GEMV, Marlin) | Via `-ngl` | **Fastest** |
| CPU / Mac | No | No | No | **Yes — the whole point** | No |
| 4-bit quality | Good (NF4 codebook) | Excellent | **Excellent (often best)** | Very good (`Q4_K_M`) | Excellent |
| 3-bit | — | **Better** | Good | `Q3_K_M` acceptable | — |
| 2-bit | — | Poor | Poor | `Q2_K` cliff | NVFP4 research |
| Throughput at 4-bit | Moderate | High with Marlin | High with Marlin | CPU-bound | — |
| Can it be fine-tuned? | **Yes — via QLoRA** | No | No | No | No |
| Best for | quick experiments, QLoRA | universal GPU 4-bit | GPU 4-bit quality-first | CPU/Mac/edge/offline | max-throughput NVIDIA serving |

### 13.2 Quantization vs the other compression techniques

| Technique | Shrinks weights? | Needs training? | Needs labels? | Reversible? | Typical gain | Best when |
|---|---|---|---|---|---|---|
| **Quantization (this module)** | **Yes, 2–8×** | No (PTQ) / Yes (QAT) | No / Yes | No | 4× memory, ~3× cost | You want a smaller *deployment* |
| **Distillation (CS-08/09)** | Yes (smaller architecture) | **Yes, heavy** | Yes | No | 2–10× params, but a new model | You own the training budget and want a smaller *model* |
| **Pruning** | Sometimes (unstructured sparsity gives no speedup pre-Ampere) | Yes (usually) | Yes | No | 1.5–3× with 2:4 sparsity | You have structured-sparsity hardware |
| **LoRA / PEFT (CS-…) ** | **No** — shrinks the *trainable* set, not the model | Yes, light | Yes | Yes, mergeable | 10–100× less training memory | You want to *change* behaviour |
| **QLoRA** | Yes (base) + trains adapters | Yes, light | Yes | Adapter merge only | 4× less training memory | **You want to fine-tune what would not otherwise fit** |
| **Speculative decoding** | No | No (needs a draft model) | No | N/A | 2–3× latency | Your bottleneck is decode latency, not memory |
| **FlashAttention / KV tricks** | No (helps the KV cache) | No | No | N/A | 2–4× long-context | Your bottleneck is context length |

**The composition rule:** quantization is orthogonal to all of the above and stacks with most of them. The productive combinations are **QLoRA** (quantization + PEFT), **distill-then-quantize** (a smaller model, then 4-bit), and **quantize + speculative decoding** (4-bit weights make the draft model cheaper too).

### 13.3 Granularity, one more time, as a cost table

| Granularity | Bits/param of weights | Bits/param of metadata | **Effective bits/param** | Relative MSE at 4-bit | Kernel support |
|---|---|---|---|---|---|
| Per-tensor | 4.0 | ~0 | **4.00** | 1.00× | universal |
| Per-channel | 4.0 | 0.001 | **4.00** | ~0.8× | standard |
| Per-group 128 | 4.0 | 0.25 | **4.25** | ~0.55× | GPTQ/AWQ/bnb/GGUF |
| Per-group 64 | 4.0 | 0.50 | **4.50** | ~0.48× | GPTQ/AWQ |
| Per-group 32 | 4.0 | 1.00 | **5.00** | ~0.42× | GPTQ/AWQ (slower) |
| Per-element | 4.0 | 16.0 | **20.0** | ~0.38× | nobody |

**The table is the argument for `group_size=128`:** going from 128 to 32 costs 18% more bytes (4.25 → 5.00 effective bits) and buys ~24% less MSE. That is a *good* trade at 3-bit and a *bad* one at 4-bit, which is exactly what the practice reports say.

---

## §14 Debugging Playbook

| Symptom | Likely cause | Diagnostic | Fix |
|---|---|---|---|
| `torch.cuda.OutOfMemoryError` during quantization | The fp16 model does not fit; `device_map="auto"` offloaded some layers | `nvidia-smi`, `torch.cuda.memory_summary()` | Quantize on a bigger GPU, or shard, or use `--method gguf` on CPU |
| OOM at load time *after* quantizing | The loader upcasts to fp16 before quantizing (bnb/GPTQ/GGUF all do) | Watch VRAM during `from_pretrained` | Reduce `max_memory`, use `low_cpu_mem_usage=True`, quantize on CPU then move |
| `ValueError: ExllamaV2 cannot be used with this model` | `auto-gptq`'s ExLlamaV2 kernel cannot fuse this checkpoint | The full message names the offending module | `disable_exllamav2=True` / `use_exllamav2=False`; falls back to Triton |
| `ImportError: cannot import name 'AutoAWQForCausalLM'` | `auto-awq` (research repo) installed instead of `autoawq` | `pip list \| grep -i awq` | `pip uninstall auto-awq && pip install autoawq` |
| `NameError: name 'AutoGPTQForCausalLM' is not defined` | `auto-gptq` not installed in the *notebook's* kernel | `import auto_gptq; print(auto_gptq.__version__)` | `pip install auto-gptq optimum` in the same kernel |
| Quantization finishes but the model is garbage | Zero-point clipped into the signed range (§2.4 Case C), or a `q_max` mismatch | Print `s` and `z` per layer; compare to the config file | Re-quantize with the symmetric scheme; verify the config on load |
| Output is fluent but ignores instructions | **GGUF chat template missing/wrong** — the #1 silent failure | `llama-gguf file.gguf \| grep chat_template`; compare to `tokenizer.apply_chat_template` | Re-convert with a converter that copies the template; or `--chat-template` at runtime |
| Generation never emits EOS / runs to `max_new_tokens` | The chat template or the EOS id differs between the fp16 and quantized checkpoints | Inspect `generation_config.json` in both | Copy `generation_config.json` and the tokenizer into the quantized dir |
| Perplexity fine, JSON output broken | Perplexity is not a task gate (§12.1) | Run the KL script; look at the **p95** | +1 bit, or `group_size` 128 → 32, or exclude the `lm_head` from quantization |
| Quality much worse than the model card claims | Calibration corpus ≠ production traffic; or a small model with the big-model recipe | Compare activation ranges on your data vs generic text | Calibrate on 128–512 of *your* samples; use `group_size=32–64` below 3B |
| 4-bit model is *slower* than fp16 | A fallback kernel was selected, or CPU dequant with no SIMD | `llama-bench`; check the kernel name/model card | Install Marlin/ExLlamaV2; use the GEMM AWQ build; check the GPU arch |
| Different outputs on two identical runs | CUDA atomics in `H` accumulation; unpinned library versions | Re-quantize twice and hash | Seed, pin versions, and treat the checkpoint hash as part of the artifact |
| `RuntimeError: Expected all tensors to be on the same device` | Calibration data on CPU while the model is on GPU | The traceback points at the calibration loop | `.to(model.device)` the calibration batch |
| `KeyError: 'qweight'` / missing tensor on load | Loaded a GPTQ checkpoint with an AWQ loader (or vice versa) | `config.json` → `quant_method` | Use the matching library |
| GGUF loads but generates `<unk>` tokens | Tokenizer mismatch — the GGUF was converted from a different HF repo revision | `llama-gguf` metadata vs the HF `tokenizer_config.json` | Re-convert from the exact revision |
| Quantized model reports a *lower* perplexity than fp16 | Tokenizer or template changed; or you evaluated on the training corpus | Diff `tokenizer_config.json`; check the eval text | Fix the tokenizer; re-evaluate on held-out data |
| Fine-tuning a quantized checkpoint silently no-ops | Gradients cannot flow into integer weights | `for n,p in model.named_parameters(): print(n, p.requires_grad)` | Switch to QLoRA (NF4 + LoRA adapters) |

**Loss-curve diagnosis (QLoRA, the one case where you train on a quantized base):**

| Curve | Meaning | Fix |
|---|---|---|
| Loss flat from step 0 | Adapters not attached, or `target_modules` matched nothing | Print trainable parameter names and count |
| Loss falls then spikes to NaN | LR too high for the quantized base | 2e-4 → 1e-4; add warmup; check for bf16 overflow |
| Train loss falls, eval loss rises | Overfitting the adapter | Fewer epochs, more data, lower rank |
| Loss goes to ~0 immediately | Labels leaked into the prompt, or the mask is wrong | Check that the prompt tokens are masked to `-100` |
| Eval loss is fine but generation is bad | Train/inference template mismatch | Use the *same* `apply_chat_template` in both |

---

## §15 Applied Case Studies

### 15.1 A document-QA startup: 8 GB laptop GPUs, a 7B model

**Situation.** 12-person startup, on-prem deployment for a legal client. Hardware: 4× RTX 3070 (8 GB each). Model: Llama-3.1-8B-Instruct. Requirement: 4k context, batch 4, on-prem, no cloud.

**Why this technique.** fp16 needs 17.2 GB (weights 16 + KV 0.54·4) — does not fit an 8 GB card at all. But the *shape* of the numbers says something specific: with GQA the KV cache is only 128 KB/token, so batch 4 × 4k costs 2.15 GB, leaving ~5.5 GB for weights. That is 4-bit territory with room to spare.

**Config.** AWQ, `w_bit=4`, `q_group_size=128`, `version="GEMM"`, `zero_point=True`, calibrated on 256 samples × 512 tokens **of the client's own documents**.

**Result.** Weights 4.5 GB + KV 2.15 GB + activations 0.4 GB + context 0.6 GB ≈ 7.7 GB — fits, tightly. 38 tokens/s per card at batch 4.

**What went wrong first.** The first attempt calibrated on WikiText. Legal documents use long defined terms and citation strings; the activation ranges on the client's data were different, and the model began mis-citing clause numbers while producing perfect prose. Re-calibrating on the client's corpus fixed it. **Second failure:** the AWQ checkpoint was loaded with `version="GEMV"` on a batch-4 workload — correct output, half the throughput. The `version` flag selects the kernel, and it is not auto-detected.

### 15.2 A coding assistant: quality-first, no memory constraint

**Situation.** A developer-tools company serving a 32B code model on 4×A100-80GB. Plenty of VRAM; the goal is *quality per dollar*, and the acceptance gate is pass@1 on their internal test suite.

**Config.** Tested three: fp16 (baseline), GPTQ `bits=4, group_size=128, desc_act=True`, AWQ `w_bit=4, q_group_size=32`.

**Result.** GPTQ-4 `g=128` lost 3.5 points of pass@1. AWQ `g=32` lost 2.1. GPTQ `g=32 + desc_act` lost 1.2 — but the `desc_act` permutation made inference 11% slower and required Marlin to stay competitive. They shipped GPTQ `g=32 + desc_act=True` on Marlin.

**What went wrong first.** The first evaluation used WikiText perplexity, which showed *no* difference between all three checkpoints — pass@1 was where the damage was, exactly as §12.1 warns. Code is where quantization error shows up first and most visibly. **Second failure:** the first calibration set was 20 samples of English prose. Code has a completely different token distribution; re-calibrating on 256 samples of their own repo took the pass@1 delta from 1.2 to 0.4.

### 15.3 An edge/IoT deployment: a 1B model on a Raspberry Pi 5

**Situation.** Offline, air-gapped kiosk. No GPU. 8 GB RAM, 4 GB of which is the OS and the application. Model: a fine-tuned 1B instruct model.

**Config.** GGUF via `convert_hf_to_gguf.py --outtype f16` then `llama-quantize` to `Q4_K_M`, with `imatrix` computed from 200 in-domain samples.

**Result.** `Q4_K_M` with an imatrix: 0.72 GB, ~9 tokens/s on the Pi's CPU, indistinguishable from fp16 on their 40-prompt acceptance set. Without the imatrix, `Q4_K_M` scored 2 prompts worse; with `Q3_K_M` + imatrix they got 0.58 GB and *equal* quality to no-imatrix `Q4_K_M`.

**What went wrong first.** The kiosk's prompts were rendered by string concatenation, not by the GGUF's `tokenizer.chat_template` — so the model received raw text and produced completions instead of answers. **The model was running perfectly; the prompting was wrong.** This is the single most common GGUF production bug, and it produces no error message.

### 15.4 A RAG service: cutting GPU cost 60%

**Situation.** A RAG service over 4M documents. 7B model, 8k context, 20 requests/s sustained, on A100-80GB. Monthly GPU bill: ~$11k.

**Config.** Migrated fp16 → AWQ 4-bit, then added FP8 KV cache.

**Result.** 8 requests/s per GPU → 22 requests/s per GPU as concurrency went from 2 to 7 sequences (KV per sequence at 8k: 1.07 GB fp16 → 0.54 GB FP8; weights 14 GB → 4.5 GB). Bill: $11k → $4.3k. Quality gate: KL divergence mean 0.04, p95 0.4, top-1 agreement 97% — passed.

**What went wrong first.** They shipped the 4-bit model, saw *better* latency, and declared victory — then found that retrieval-augmented answers stopped citing the retrieved passages. The KL gate on generic text passed; the KL gate on *RAG prompts* (long context, many quoted spans) did not. **The lesson: quantize, then gate on the workload's own prompt shape, not on prose.**

### 15.5 Owning the model: QAT for a 2-bit on-device assistant

**Situation.** A hardware company shipping a voice assistant with 2 GB of model memory. They own the training pipeline and the data.

**Config.** QAT with `torchao`'s `QATConfig`, per-group 64, 15% of the original training budget, LR restarted at 1e-5 with a cosine decay, applied after an fp16 "warm start" from the PTQ checkpoint.

**Result.** 1.9 GB model, quality within 4% of the fp16 original on their task metrics. PTQ at the same bit-width was 22% worse.

**What went wrong first.** PTQ was tried first and produced fluent nonsense at 2-bit. Then rotation (a Hadamard transform) was applied before GPTQ and recovered roughly half the gap for zero training cost. **QAT was only justified because they owned the model and had the data** — the rule from §3.5.

---

## §16 Production Considerations

| Concern | Practice |
|---|---|
| **Artifact versioning** | The quantized checkpoint is `(fp16 checkpoint hash, method, bits, group_size, desc_act, calibration corpus hash, library versions, kernel/CUDA arch)`. Version all nine; any change is a new artifact. Store the `quantize_config.json` *next to* the weights and treat it as part of the model. |
| **Reproducibility** | Same checkpoint + same library + different GPU arch = different numerics. Record `sm_80`/`sm_89`/`sm_90`. Hash the output weights and compare on rebuild. |
| **Rollback** | Always keep the fp16 checkpoint and the previous quantized artifact. Rollback is a config change, not a re-quantization. |
| **Regression gate in CI** | Run §12's KL + task gate on every rebuild, on a frozen 200-prompt suite. Fail the build on `p95 KL > 1.0` or a >2% task drop. |
| **Monitoring** | Track tokens/s, p50/p99 latency, VRAM high-water, and *output-shape* metrics (JSON-validity rate, tool-call success rate, average response length). A quantization regression shows up as a shape change before it shows up in user complaints. |
| **Drift** | If production traffic drifts (a new language, a new document type, a new tool), your calibration corpus is stale. Re-calibrate quarterly or on drift alerts. |
| **A/B testing** | Serve fp16 and 4-bit behind a flag on a small slice, compare on the task metric, not on perplexity. |
| **Guardrails** | Quantized models fail *format* more than *content*. Put a schema validator in front of any JSON output, and retry-on-invalid rather than on-refusal. |
| **Latency SLOs** | Quantization improves throughput (more concurrency) more than it improves single-request latency. If your SLO is p99 latency at batch 1, measure the GEMV path (`version="GEMV"` for AWQ) explicitly. |
| **Security/compliance** | A quantized model is a *different* model for the purposes of a model card and any certification. Re-run safety evaluations; quantization error can shift refusal behaviour. Some regulated deployments require the fp16 artifact on record. |
| **Licensing** | Quantized derivatives inherit the base model's licence. Check the licence before uploading a quantized checkpoint to the Hub. |
| **Storage** | You keep N quantized variants × M sizes. A 70B model suite can exceed 200 GB of artifacts; deduplicate the tokenizer and config files. |
| **Cost model** | $/1M tokens ∝ bytes-read/param, but the real win is concurrency (§11.2). Model both before quoting a saving. |

---

## §17 Common Misconceptions

1. **"Quantization makes models better."** It never does. Fine-tuning after quantization (QLoRA) can make the *task* performance better, but that is the adapter, not the quantization.
2. **"Quantization is lossless at 8-bit."** Effectively, yes (Δppl < 0.01) — but not theoretically, and not on every task. Perplexity is not a complete description of a model.
3. **"4-bit halves the size of 8-bit, so it is twice as risky."** Risk is not linear in bits; it is linear in `log(MSE)`, and 4-bit is still in the flat part of the curve for ≥7B models. That is why the field settled on 4-bit rather than 6-bit.
4. **"A 4-bit model runs 4× faster."** It *reads* 4× fewer bytes. Compute is still on FP16 tensor cores, and decode is only partly bandwidth-bound. Expect ~2–3× at batch 1 and a larger gain from concurrency.
5. **"GPTQ stands for Gradient Post-Training Quantization."** No — **G**enerative **P**re-trained **T**ransformer **Q**uantization (§1.4).
6. **"AWQ prunes the salient channels."** It rescales them. No weight is removed (§5.4).
7. **"GGUF is a quantization method."** It is a container format; the method is the k-quant or legacy quant type inside it (§6.1).
8. **"`Q4_K_M` means medium quality."** `_M` describes how much of the model is kept at *higher* precision (small/medium/large mixes), i.e. it is about size/aggressiveness (§6.3).
9. **"Symmetric quantization is just asymmetric with z=0."** Functionally yes, but the *range* differs (symmetric wastes half the grid on a one-sided tensor — §2.4 Case D) and the hardware paths are entirely different (§2.2).
10. **"The zero-point is a small correction."** When it is clipped into a range that cannot hold it, it is catastrophic and silent (§2.4 Case C: 44,000× MSE).
11. **"Calibration data just needs to be text."** It needs to be *your* text. Calibration corpus mismatch is the most common cause of "the quantized model is fine on the benchmark and bad in production" (§15.1, §15.2).
12. **"QAT is required for 4-bit."** PTQ reaches near-lossless 4-bit on ≥7B models. QAT is for 2-bit and for model owners (§3.5).
13. **"QLoRA trains a quantized model."** It trains adapters over a *frozen* quantized base. The quantized weights receive no gradient; the quantizer is never updated (§8.6).
14. **"Perplexity is the standard quality gate."** It is a weak instrument that passes broken models (§12.1).
15. **"More calibration samples is always better."** The curve has a knee at a few hundred; beyond that you are burning time. Sample *diversity* matters more than count.
16. **"Quantized weights can be converted back to fp16 and fine-tuned."** You can dequantize, but the information is gone — you would be starting from a degraded base. Fine-tune the original.
17. **"Smaller group_size is strictly better."** Better accuracy, more metadata, slower kernels. `g=128` is the default for a reason (§13.3).
18. **"The KV cache shrinks when I quantize the weights."** It does not, unless you separately quantize the KV cache. This is the single most common "why did it still OOM" answer (§8.2).

---

## §18 Key Takeaways

1. **`q = round(x/s) + z`, `x̂ = s(q − z)`.** Two lines of algebra; everything else is a choice of `s`, `z`, and how many values share them.
2. **`Δ = (max − min)/(2^b − 1)`, `MSE = Δ²/12`.** Error is linear in the range and quadratic in the step. Halving the range quarters the error; one 60× outlier costs 3600× the MSE and 5.9 bits.
3. **Signed INT8 is the hardware default.** Asymmetric uint8 is a pedagogical scheme; a zero-point that does not fit the signed range breaks the model silently.
4. **Granularity is the cheapest accuracy in the field.** Per-channel took a 24.6% relative error to 1.4% for four bytes per row; `group_size=128` is the standard because it is the accuracy/size/kernel triple point.
5. **PTQ: calibrate, then solve. QAT: simulate `round()` and lie about its gradient.** The STE's `∂round/∂x := 1` is the right lie because the quantizer is approximately the identity in the region where the gradient matters.
6. **The outlier problem is the field.** Massive activations — 0.1% of channels, 20–100× the median, absent below ~6.7B — are why naive INT8 failed and why mixed precision, granularity, migration, salience protection, compensation, and rotation all exist.
7. **GPTQ = second-order error compensation (`H = 2XXᵀ`); AWQ = activation-aware rescaling of the salient 1%.** Both are layer-wise, one-shot, W4A16, and near-lossless at 4-bit for ≥7B. GPTQ wins at 3-bit; AWQ is faster to produce and usually ahead at 4-bit on instruction-tuned models.
8. **GGUF is a container, GGML was its predecessor library and format.** The container carries metadata, tokenizer, and chat template — and a missing chat template is the field's #1 silent failure.
9. **Read k-quant names as bits + k-quant + size-mix:** `Q4_K_M` = 4-bit, hierarchical blocks, medium mix. `_S/_M/_L` is about which tensors keep higher precision, not about quality grading.
10. **7B = 14 GB fp16 = 7 GB int8 = 3.5 GB int4; 70B = 140/70/35.** And the KV cache is a *separate* 128 KB/token (GQA 8B) to 800 KB/token (no-GQA 13B) — it does not shrink when you quantize weights.
11. **The accuracy-vs-bits curve is a comparison between quantization error and model redundancy.** 8-bit is free, 4-bit is nearly free above 7B, 3-bit costs a little, 2-bit is a cliff.
12. **Quantization is a one-way street: quantized weights cannot be trained** — except QLoRA, which trains adapters over a frozen NF4 base. Correct order: **fp16 → fine-tune → merge → quantize.**
13. **Perplexity is not a gate.** Use mean KL, p95 KL, top-1 agreement, and a task metric on held-out domain data.
14. **Calibrate on your own traffic.** The corpus mismatch is the most common production quantization bug and it is invisible on benchmarks.
15. **The real win is concurrency.** Weights 4× smaller means 4× more KV-cache room, which means 4× the batch, which means ~3–4× the throughput per GPU — not a 4× faster kernel.

---

## §19 Self-Check Questions

1. Write the affine quantization equations and explain what each symbol is in units.
2. Derive `MSE = Δ²/12` from `Δ = (max − min)/(2^b − 1)`. What happens to the MSE if a 40× outlier appears in the tensor? How many bits does it cost?
3. Take the eight weights `[0.0234, −0.1456, 0.7891, −1.2345, 0.5123, −0.0678, 0.3345, −0.9123]`, quantize them to symmetric INT8, and give the scale, the eight codes, the eight dequantized values, and the max absolute error.
4. Why does the same vector quantized asymmetrically into *uint8* produce a *smaller* step but a *worse* MSE?
5. Explain why `z = 156` in a signed int8 representation is catastrophic but not an error.
6. Which single weight in a tensor is always quantized exactly, and why?
7. Compare per-tensor, per-channel, and per-group `g=128` on accuracy, metadata overhead, and kernel support. Why is 128 the default and not 32?
8. State the straight-through estimator. Give two reasons it is the right thing to do and one case where it fails.
9. Walk through `prepare_qat` → train → `convert`. Why must training happen *between* the two calls?
10. What is a massive activation, why does it exist mechanistically, and which mitigation would you use if it is a persistent channel in the activations (not the weights)? Why does per-channel granularity not solve it?
11. Describe GPTQ's objective, what `H` is, what error compensation does, and what `act_order`/`desc_act` changes.
12. What does AWQ measure that GPTQ does not? Write the rescaling identity and explain why it costs nothing at inference.
13. List four differences between GGML and GGUF, and identify which one causes silent production incidents.
14. Decode `Q4_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`. What does the `imatrix` do and which quant types does it apply to?
15. Compute the VRAM for a 7B model at int4 serving 4096 tokens at batch 1, including the KV cache for a 32-layer, 32-KV-head, 128-head-dim architecture. Why does the answer change if the model uses GQA with 8 KV heads?
16. Why does a 4-bit model not run 4× faster than fp16? What *does* it give you 4× of?
17. Name the five mitigations for the outlier problem and say which one is calibration-free.
18. Your quantized model holds perplexity and fails JSON compliance. What is your next step, and which metric would have caught it?

<details>
<summary><strong>Answers</strong></summary>

1. `q = round(x/s) + z` (forward), `x̂ = s(q − z)` (inverse). `x` is a real value in the tensor's units; `s` is the real width of one integer step, in the same units as `x`; `z` is the integer code that maps to real 0.0; `q` is the stored integer, bounded by `[q_min, q_max]`; `x̂` is the reconstructed value in the tensor's units.
2. With error uniform in a cell, `e ~ U(−Δ/2, Δ/2)`, so `E[e] = 0` and `E[e²] = Δ²/12`. A 40× outlier multiplies `Δ` by 40, so MSE by 1600, and consumes `log2(40) ≈ 5.3` bits.
3. `s = 1.2345/127 = 0.0097205`; codes `[2, −15, 81, −127, 53, −7, 34, −94]`; dequantized `[0.019441, −0.145807, 0.787358, −1.234500, 0.515185, −0.068043, 0.330496, −0.913724]`; max abs error `0.004004` (element 6, `0.3345 → 0.330496`).
4. The uint8 grid uses all 256 codes over the exact `[min, max]` span, so the step is genuinely smaller (0.0079 vs 0.0097). But `z` must be *round*ed to an integer, which offsets the whole reconstruction grid by up to `s/2`; the resulting error is systematic and biased (all errors share a sign) rather than zero-mean noise. Biased error of small magnitude can have a larger second moment than unbiased error of larger magnitude. On an approximately zero-mean tensor, symmetric wins.
5. Any code outside `[−128, 127]` cannot be stored in int8, so the framework clamps it. An inflated `z` means that almost every code is out of range, so almost every value saturates to the same integer. The tensor remains a valid int8 tensor, inference runs, and the output is wrong — six of eight values saturated in §2.4 Case C, giving a 44,000× MSE blowup with no exception.
6. The element with the largest absolute value, because it *defines* the scale (`s = max|x|/q_max`), so it lands exactly on `±q_max`.
7. Per-tensor: one `(s,z)`, no metadata, worst accuracy, universal support. Per-channel: one per output row, ~0.001 bits/param, much better, standard. Per-group 128: one per 128 input weights per row, 0.25 bits/param, better still, supported by GPTQ/AWQ/bnb/GGUF. 128 is the default because it aligns with GPU memory/dequant block sizes, its metadata cost (4.25 effective bits) is affordable, and the accuracy knee is there — 128→32 buys ~24% less MSE for 18% more bytes and a slower kernel.
8. Forward: `q = round(clamp(x/s))`. Backward: `∂q/∂x := 1` on the unclipped interval. Right because (a) the quantizer is approximately the identity for the quantity that matters — the reconstruction error is bounded by `Δ/2` and essentially uncorrelated with `x`, so the best linear estimator is the identity; (b) it makes `round()` a noise source the network learns to tolerate, empirically shifting weights toward bin centers. It fails when the bins are coarse relative to the gradient signal (2-bit), where the gradient's scale no longer resembles the true error.
9. `prepare_qat` inserts `FakeQuantize` modules whose observers accumulate activation ranges during the forward pass, and whose backward pass uses the STE. Training must happen between `prepare_qat` and `convert` because the observers need to *see* data with the loss active before their ranges are frozen; `convert` then bakes those frozen scales into real integer ops. Converting before training gives you a model quantized with uninitialized ranges.
10. A massive activation is an activation channel whose magnitude is 20–1000× the median, present in the same channel indices for nearly every input, concentrated in the residual stream around attention-sink tokens; it exists because a consistently large channel survives RMSNorm and acts as a fixed bias that any downstream layer can read. If it is in the *activations*, per-channel granularity does not help — activations are quantized per-token at inference and cannot carry a per-channel weight-style scale; you need **SmoothQuant** (migrate the outlier into the weights), **rotation** (spread it), or **mixed precision** (LLM.int8()).
11. Objective: `min ‖WX − ŴX‖²`, i.e. minimize the layer's *output* error, not the weight error. `H = 2XXᵀ` is (twice) the input covariance over the calibration set, and it weights each weight by how active its input channel is. Error compensation: after quantizing a column, the residual `δ = w − ŵ` is subtracted from the remaining columns scaled by `H⁻¹`, so the error does not accumulate. `act_order`/`desc_act` quantizes columns in decreasing order of `H`'s diagonal (activation importance) instead of left-to-right; better accuracy, ~10% slower inference because the permutation breaks the kernel's expected layout.
12. AWQ measures the **per-input-channel activation magnitude** `mean|X_j|` and treats the top ~1% as salient, whereas GPTQ's importance comes implicitly from the full covariance. Rescaling identity: `W·X = (W·diag(s))·(diag(s)⁻¹·X)`. It costs nothing at inference because activations are not quantized (W4A16) — the `diag(s)⁻¹` on the activation side is folded into the preceding LayerNorm/Linear and is exact in fp16, and the weight side is where quantization happens, now with the salient columns occupying more integer levels.
13. Four differences: (a) GGML `.bin` hard-coded hyperparameters, GGUF stores them as metadata KV; (b) GGML had no version field, GGUF has `gguf_version`; (c) GGML kept the tokenizer in a separate file, GGUF embeds it; (d) GGML was not memory-mappable, GGUF is. The silent-incident one is the **embedded tokenizer/chat template** — a GGUF with a missing or wrong `tokenizer.chat_template` produces fluent completions instead of instruction-following, with no error.
14. `Q4_K_M` = 4-bit, k-quant (hierarchical: 256-weight super-block with a scale+min, subdivided into 16/32-weight sub-blocks with quantized scales), Medium size-mix (some tensors kept at higher precision). `Q5_K_S` = 5-bit k-quant, Small mix. `Q6_K` = 6-bit k-quant, no size-mix suffix. `Q8_0` = 8-bit legacy block quant, symmetric, one scale per 32 weights, essentially lossless. The `imatrix` is an importance matrix of activation statistics that weights the quantization error during `llama-quantize`; it applies to k-quants and especially the `IQ*` types, improving 2–3 bit quality by 5–15%.
15. Weights `7e9 × 0.5 × 1.06 ≈ 3.7 GB`. KV per token `= 2 × 32 × 32 × 128 × 2 bytes = 512 KB`; at 4096 tokens `= 2.15 GB`. Plus ~0.1 GB activations, ~0.6 GB CUDA context, ~10% slack → **≈ 7.2 GB**. With GQA at 8 KV heads the KV per token falls to `2 × 32 × 8 × 128 × 2 = 128 KB`, so at 4096 tokens it is only **0.54 GB** and the total is ~5.0 GB — a 2.2 GB difference from an architecture detail that has nothing to do with the parameter count.
16. Because the 4-bit kernel does not do 4-bit arithmetic: it dequantizes weights into FP16 inside the matmul and runs on FP16 tensor cores, so compute throughput is unchanged. What you get 4× of is *memory bandwidth and capacity* — 4× fewer bytes read per token and 4× more room for the KV cache, which translates into ~2–3× decode speedup at batch 1 and much larger gains from increased concurrency.
17. (1) mixed precision — LLM.int8(); (2) granularity — per-channel/per-group; (3) migration — SmoothQuant; (4) salience protection — AWQ; (5) error compensation — GPTQ; plus rotation as the newer sixth. The calibration-free ones are mixed precision and rotation (and, for weights, granularity).
18. Next step: measure the failure directly — run the KL protocol (§12.3) on prompts with the same *shape* as the failing workload (structured-output tasks, long contexts) and look at the **p95 KL**, then go up one bit-width or drop `group_size` to 32, or exclude the `lm_head` from quantization. The metric that would have caught it is a **task gate** — JSON-schema validity rate or IFEval on held-out domain prompts — because mean perplexity is dominated by ordinary prose tokens and is nearly blind to the rare tokens that carry structure.

</details>

---

## §20 Cross-References

| Relationship | Module |
|---|---|
| **Builds on** | CS-01 (Foundations: bytes, params, VRAM, precision types), CS-04 (Fine-Tuning vs RAG vs Agents — the deployment context) |
| **Continues in** | **CS-11 — Quantization II: Advanced Methods & Production Practice** (GPTQ Hessian arithmetic, AWQ's W4A16 correction, GGUF CLI + `imatrix` + chat templates, KV-cache quantization, the W8A8…W4A4 precision lattice, QLoRA/NF4 algebra, serving flags, the honest evaluation protocol) |
| **Contrasts with** | CS-08 / CS-09 (Knowledge Distillation — the other compression axis: a smaller *model* rather than a smaller *encoding*) |
| **Needed by** | The QLoRA module (NF4 is the base of every QLoRA fine-tune), any serving/throughput module, CS-11 |
| **Pairs with** | `code/08_quantize.py` (runnable: bitsandbytes / GPTQ / AWQ / GGUF, with the VRAM table and the post-quantization validation checklist) |
| **Interview prep** | IQ-10 (35 L1 + 32 L2 + 25 L3 + 10 L4 + 12 L5) |
| **Cheat sheet** | CH-10 (formulas, decision tree, VRAM table, per-method snippets, symptom→fix) |

---

## Appendix A — Instructor's Verbatim Key Claims

Quotes are from `LLM_Fine-Tuning_12_LLM_Quantization_Explained_PART_1_PTQ_QAT_GPTQ_AWQ_GGUF_GGML.txt`, with timestamps. Where the instructor is imprecise, the correction appears in the body of this module rather than being silently fixed here.

| Timestamp | Claim |
|---|---|
| [5:15–6:14] | Quantization is introduced as the reduction of the precision of the numbers a model stores, so that it occupies less memory. |
| [12:26–13:46] | Biases and activations are quantizable, but attention/softmax is flagged as the special hard case. |
| [14:00] | A price of `1299.956` rounded to `1300` loses `0.044`. |
| [16:00] | `π = 3.14159265` represented as `3.14` vs as `3` — the same number on different grids. |
| [17:14–18:31] | `₹87.5793` → `₹87.573` → `₹87`; the last step loses `0.5793`. |
| [20:13–21:58] | The five goals of quantization: smaller size, lower memory, faster inference, lower power, and fitting on available hardware. |
| [23:32] | 1 GB of FP32 becomes 250 MB at INT8. |
| [25:12] | Hardware is optimized for int8/int4 — TPU, Tensor Cores, AVX-512. |
| [34:03–35:31] | The instructor's own machine: 24 GB RAM (23.7 GB usable), 6 GB VRAM, RTX 3060, 10.6 GB used / 12.9 GB available. |
| [36:14–39:50] | The full data-type table: float64/float32/float16/bfloat16/int8/int4 etc., with bytes per element. |
| [41:00–42:06] | 100M parameters = 400 MB (fp32) / 200 MB (fp16) / 100 MB (int8) / 50 MB (int4). |
| [42:18] | FP32 for full training, FP16/BF16 for mixed-precision training, INT8/INT4 for inference. |
| [43:10–44:30] | Precision = how many bits a number is stored in. |
| [51:55–55:08] | Precision is not accuracy — the darts/bullseye analogy. |
| [55:24–56:47] | The quantization and dequantization formulas, with scale and zero-point. |
| [56:52–58:23] | The 3-input / 2-hidden / 1-output network, written out as a weight matrix. |
| [1:00:00–1:01:49] | Symmetric vs asymmetric quantization ranges. |
| [1:06:00–1:09:30] | Symmetric worked example: scale `0.01575`, `xq = 78`, dequantized `1.229`, error ≈ `0.001`. |
| [1:09:33–1:14:26] | Asymmetric worked example: scale `0.00784`, zero-point `−64` clipped to `0`, `xq = 157`, dequantized `1.23`, error `0`. |
| [1:14:28–1:16:12] | Third example: uneven weights, symmetric range, zero-point `−64` kept, `xq = 93`. |
| [1:16:59–1:18:50] | Hardware and framework defaults use **symmetric int8**, not uint8. |
| [1:18:51–1:19:51] | Per-tensor vs per-channel quantization; quantization error; calibration introduced. |
| [1:19:53–1:21:00] | The PTQ vs QAT taxonomy. |
| [1:22:25–1:24:40] | The cricket-bat analogy; PTQ causes a slight accuracy drop. |
| [1:24:40–1:26:00] | Calibration data are dummy inputs used to find the activation range — example range `[−3.5, +5.7]`. |
| [1:26:36–1:28:31] | The PTQ vs QAT comparison tables: training needed, accuracy impact, speed, ease, toolkits. |
| [1:34:46–1:36:40] | Static vs dynamic PTQ. |
| practicals | `make_moons`; `BigMLP` (2→128→64→64→32→16→8→1, ReLU + sigmoid); Adam `lr=0.01`; `BCELoss`; 2000 epochs; FP32 accuracy `0.97`; `quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)` → unchanged accuracy; model size `0.06 MB`; manual `quantize_tensor`/`dequantize_tensor` simulation → `0.43`; static PTQ via `register_forward_hook`/`get_activation_min_max` with `X_calib = X_train[:100]` → `43%`; QAT with `get_default_qat_qconfig('fbgemm')` → `prepare_qat(model, inplace=True)` → `torch.quantization.convert(model.eval(), inplace=False)` → `97–98%`. |

**Note on transcript coverage.** The recorded lecture ends mid-transition at [2:12:16] with the instructor introducing the LLM-advanced material; the spoken GPTQ / AWQ / GGML / GGUF sections are therefore not in the transcript. Everything this module states about those four topics is grounded in the video's own slide deck for Video 12 and in the five companion notebooks (`LLM_Quantization_GPTQ.ipynb`, `LLM_Quantization_AWQ.ipynb`, `gguf_practical.ipynb`, `gguf_ggml_practical.ipynb`), or is marked `> **Beyond the video:**`.

## Appendix B — Reference Links & Papers

| Topic | Reference |
|---|---|
| Affine/zero-point quantization (the standard formulation) | Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018 |
| Uniform-quantizer theory, optimal step and distortion | Gersho & Gray, *Vector Quantization and Signal Compression*; Lloyd (1982), Max (1960) |
| Straight-through estimator | Bengio, Léonard & Courville, "Estimating or Propagating Gradients Through Stochastic Neurons", arXiv:1308.3432 |
| Trained quantization / LSQ | Esser et al., "Learned Step Size Quantization", ICLR 2020; Choi et al., "PACT", ICCV 2018 |
| 8-bit Inference with Outlier Decomposition | Dettmers et al., **LLM.int8()**, NeurIPS 2022 (arXiv:2208.07339) |
| **GPTQ** | Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", ICLR 2023 (arXiv:2210.17323); and "Optimal Brain Compression" (arXiv:2208.11580) |
| **AWQ** | Lin et al., "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration", MLSys 2024 (arXiv:2306.00978) |
| **SmoothQuant** | Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", ICML 2023 (arXiv:2211.10438) |
| Rotation-based outlier removal | Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", NeurIPS 2024 (arXiv:2404.00456); Liu et al., "SpinQuant", ICLR 2025 |
| Calibration-free weight quantization | Badri & Shaji, "HQQ: Half-Quadratic Quantization of Large Neural Networks" (arXiv:2410.11845) |
| QAT in the loop, at scale | Liu et al., "EfficientQAT: Efficient Quantization-Aware Training for Large Language Models" (arXiv:2407.11062) |
| NF4 / double quantization / **QLoRA** | Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023 (arXiv:2305.14314) |
| GGUF specification | `github.com/ggerganov/ggml/blob/master/docs/gguf.md` |
| k-quants, `imatrix` | `github.com/ggerganov/llama.cpp` → `examples/quantize`, `examples/imatrix`; PR #1684 (k-quants), PR #4930 (imatrix) |
| Fast 4-bit kernels | Frantar et al., "Marlin: Mixed-Precision Auto-Regressive Parallel Inference" (arXiv:2408.11743); `turboderp/exllamav2` |
| FP8 formats for deep learning | Micikevicius et al., "FP8 Formats for Deep Learning" (arXiv:2209.05433) |
| Microscaling (MX / NVFP4) | OCP Microscaling Formats (MX) Specification v1.0; NVIDIA NVFP4 documentation |
| Quantization for serving at scale | NVIDIA TensorRT-LLM docs; `vllm-project/vllm` supported-quantization docs |
| A critical look at evaluation | "Give Me BF16 or Give Me Death? Accuracy-Performance Trade-Offs in LLM Quantization" (arXiv:2411.02355) |

**Repository artifacts used by this module**

| Artifact | Path |
|---|---|
| Quantization notebook (PTQ/QAT on a toy MLP) | `LLM Fine-Tuning-12-13-LLM-Quantization/LLM_Quantization/Model_Quantization_Final.ipynb` |
| GPTQ notebook | `.../LLM_Quantization/LLM_Quantization_GPTQ.ipynb` |
| AWQ notebook | `.../LLM_Quantization/LLM_Quantization_AWQ.ipynb` |
| GGUF notebook | `.../LLM_Quantization/gguf_practical.ipynb` |
| GGUF / GGML (legacy `.bin`) notebook | `.../LLM_Quantization/gguf_ggml_practical.ipynb` |
| Runnable end-to-end script | `Finetuning-Handbook/code/08_quantize.py` |

