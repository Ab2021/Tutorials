# IQ-10 — Quantization Interview Questions (PTQ, QAT, GPTQ, AWQ, GGUF, GGML)

| Field | Value |
|---|---|
| **Module** | Efficiency & Compression |
| **Pairs with** | CS-10 (Quantization I — Fundamentals), CH-10 (Cheat Sheet) |
| **Total questions** | **114** (35 L1 + 32 L2 + 25 L3 + 10 L4 + 12 L5) |
| **Levels covered** | L1 screen · L2 working engineer · L3 senior · L4 staff / system design · L5 debugging |
| **Plus** | 20-row rapid-fire true/false, 8 coding/whiteboard tasks, 30 numbers to memorize, answers to CS-10 §19 |
| **Advanced follow-ups** | IQ-11 covers the GPTQ Hessian arithmetic, AWQ's correction, KV-cache quantization, and the W8A8…W4A4 lattice |

---

## How To Use This File

| Level | Who asks it | What it tests | Time per question |
|---|---|---|---|
| **L1** | Recruiter screen, or the first 20 minutes of a technical screen | Vocabulary. Can you define terms *precisely* and not confuse them? | 30–60 s |
| **L2** | A working engineer on the team | Can you do the arithmetic and write the code? Do you know the defaults and the flags? | 2–4 min |
| **L3** | A senior engineer or the hiring manager | Do you know *why*, and do you know where the method breaks? | 5–10 min |
| **L4** | Staff / system design round | Can you size a deployment, choose a format, and defend the accuracy/latency/cost trade-off? | 20–40 min |
| **L5** | Debugging round, or an "tell me about a failure" behavioural | Can you diagnose a broken quantized model from symptoms, without a stack trace? | 10–20 min |

**How to answer at each level.** L1: one sentence, correct vocabulary, no hedging. L2: the number, the flag, and the default. L3: mechanism, then the failure mode, then the mitigation. L4: the formula, the numbers, and the trade-off you chose. L5: the *silent* failure first, then the diagnostic, then the fix.

**Company-style tags.** Where a question is asked in a recognisable style, the tag is given: `[OpenAI infra]`, `[Anthropic serving]`, `[NVIDIA/TensorRT]`, `[Meta/OSS release]`, `[HF ecosystem]`, `[startup/on-prem]`, `[quant-research]`. Tags are indicative, not official.

### The five facts that carry this entire module

If you retain nothing else:

1. **`q = round(x/s) + z`, `x̂ = s(q − z)`** — and `Δ = (max − min)/(2^b − 1)`, `MSE = Δ²/12`.
2. **7B = 14 GB fp16 = 7 GB int8 = 3.5 GB int4** — and the **KV cache is separate**: 128 KB/token (GQA 8B) to 800 KB/token (no-GQA 13B).
3. **PTQ ≠ QAT.** PTQ calibrates and solves; QAT inserts `round()` and fakes its gradient with the STE (`∂round/∂x := 1`).
4. **GPTQ = second-order error compensation with `H = 2XXᵀ`; AWQ = activation-aware rescaling of the salient ~1%.** Both layer-wise, one-shot, W4A16.
5. **GGUF is a container, not a method.** `Q4_K_M` = 4-bit, k-quant, Medium size-mix. A missing chat template is the #1 silent production failure.

---

## Level 1 — Screen (35 questions)

**Q1. What is quantization, in one sentence?**

- **Answer:** Quantization is the replacement of a continuous or wide range of numeric values with a small fixed grid of integer codes plus one scale (and optionally one zero-point) per tensor, group, or channel — trading numeric precision for memory and bandwidth.
- **Why asked:** It separates people who have *used* a quantized model from people who know what happened to it. "Makes the model smaller" is a weak answer at L1.
- **Trap:** Saying "it reduces accuracy". Quantization reduces *precision*; whether accuracy follows is an empirical question, and at 8-bit the answer is usually "no".

---

**Q2. Write the affine quantization equations and name every symbol.**

- **Answer:** `q = round(x/s) + z` and `x̂ = s(q − z)`. `x` is the real value in the tensor's units; `s` is the **scale**, the real width of one integer step, in the same units as `x`; `z` is the **zero-point**, the integer code that represents real `0.0`; `q` is the stored integer in `[q_min, q_max]`; `x̂` is the reconstructed real value.
- **Why asked:** The single most-asked L1 question in every quantization interview. It is a filter.
- **Trap:** Getting the signs backwards on the inverse — it is `s(q − z)`, not `s·q + z`. And forgetting that `z` is an *integer*.

---

**Q3. What is the difference between symmetric and asymmetric quantization?**

- **Answer:** Symmetric forces `z = 0` and uses a grid `[−q_max, +q_max]`; asymmetric computes `z = round(−min/s)` and uses the full `[q_min, q_max]` range. Symmetric is what signed-int8 hardware executes natively; asymmetric fits one-sided distributions (ReLU activations) better in theory but is uint8-only in practice.
- **Why asked:** The follow-up is always "which do frameworks use, and why?" — see Q4.
- **Trap:** Claiming asymmetric is "strictly better because it uses all the levels". It rounds `z`, which introduces a systematic offset; on a zero-mean tensor symmetric often wins on MSE (§2.4 of CS-10).

---

**Q4. Which is the default in real frameworks — symmetric or asymmetric — and why?**

- **Answer:** **Symmetric int8** for weights, because AVX2/AVX-512 VNNI, NVIDIA tensor cores and ARM/Qualcomm NPUs all execute signed 8-bit dot products directly; an asymmetric scheme needs an extra add or a biased matmul. `torch.ao.quantization` uses symmetric int8 for weights and asymmetric uint8 for ReLU activations, because activations are genuinely non-negative.
- **Why asked:** Tests whether you have read the hardware constraints or only the textbook formulation.
- **Trap:** Extrapolating "asymmetric for activations" to "asymmetric for LLM weights". It is not, and it costs 15–25% throughput.

---

**Q5. What is a zero-point, physically?**

- **Answer:** It is the integer code that the real number `0.0` maps to. Equivalently, it is the integer offset that slides the integer grid so it covers exactly `[min, max]` of the tensor instead of being centred on zero.
- **Why asked:** Tests whether `z` is understood as a *shift* rather than a magic constant.
- **Trap:** Saying "zero-point is where zero is stored" without noting that when `z` falls outside the representable integer range it must be **clipped**, and that clipping it silently destroys the model (§2.4 Case C).

---

**Q6. Give the formula for the scale and the zero-point in min-max affine quantization.**

- **Answer:** `s = (max − min)/(q_max − q_min)` and `z = round(−min/s)`. With a `2^b`-level closed integer interval, `q_max − q_min = 2^b − 1`.
- **Why asked:** It is the arithmetic that everything else is built on.
- **Trap:** Using `2^b` in the denominator instead of `2^b − 1`. The number of *steps* between `q_min` and `q_max` is `2^b − 1`, not `2^b`.

---

**Q7. How many levels does a `b`-bit quantizer have, and what is the step size?**

- **Answer:** `2^b` codes, `2^b − 1` steps, so `Δ = (max − min)/(2^b − 1)`. For int8 with the `[−127, 127]` convention there are 255 codes and a span of 254.
- **Why asked:** It is the arithmetic in Q6, asked differently — and `[−127,127]` vs `[−128,127]` is a real distinction.
- **Trap:** Not knowing that `[−127, 127]` deliberately leaves `−128` unused so that negation and absolute value are safe in int8.

---

**Q8. Define PTQ and QAT.**

- **Answer:** PTQ (Post-Training Quantization) quantizes an already-trained model using calibration statistics and no gradient updates. QAT (Quantization-Aware Training) fine-tunes the model with quantization simulated in the forward pass, so the weights adapt to the quantization error.
- **Why asked:** The second-most-asked L1 question.
- **Trap:** Saying PTQ "needs no data". Weight-only PTQ needs no data; static PTQ needs a calibration set; GPTQ/AWQ need calibration data for `H` or the activation magnitudes.

---

**Q9. What is calibration data, and how much do you need?**

- **Answer:** Calibration data are unlabelled inputs run through the model to *observe* the ranges of weights and activations before fixing the quantization parameters. There are no labels and no loss. Typical: 50–200 sequences for LLM PTQ, 128–512 samples for GPTQ/AWQ, with a knee at a few hundred.
- **Why asked:** Tests whether you know the data requirement is real but small.
- **Trap:** "More is always better." Beyond a few hundred samples the returns flatten; *domain match* matters far more than count.

---

**Q10. What is quantization error, and how do you aggregate it?**

- **Answer:** Per element, `|x − x̂|`. Aggregated as MSE (`E[(x − x̂)²]`), RMSE, or SQNR in dB (`10·log10(σ²_x/σ²_e)`). For a uniform grid with a uniform source, `MSE = Δ²/12` and `max|error| = Δ/2`.
- **Why asked:** Sets up the "MSE is quadratic in the range" insight they will ask next.
- **Trap:** Reporting error as a fraction of the step (which is always ≤ 0.5) instead of as a fraction of the *value*, which is what a model actually experiences.

---

**Q11. What is the KV cache and how big is it?**

- **Answer:** The cached keys and values for every token in the context, `2 · L · h_kv · d_head · seq · batch · bytes`. Anchors: **128 KB per token** for a GQA 7–8B model (32 layers, 8 KV heads, head_dim 128) and **512 KB per token** for a 2023-era no-GQA 7B (32 KV heads).
- **Why asked:** It is the memory nobody remembers to include, and it decides whether a deployment fits.
- **Trap:** Assuming the KV cache shrinks when you quantize the weights. It does not — unless you quantize the KV cache separately.

---

**Q12. How much memory does a 7B model need at fp16, int8, and int4?**

- **Answer:** 14 GB, 7 GB, 3.5 GB. Rule: `params × bytes_per_param`. The 2-bit figure is 1.75 GB. These are weights only — add the KV cache, activations and a ~0.6–1.2 GB CUDA context.
- **Why asked:** The single most useful number in the field, and interviewers ask it as a warm-up before the real sizing question.
- **Trap:** Quoting GiB and GB interchangeably. 14 GB ≈ 13.0 GiB; the difference is large enough to matter when you are 1 GB from an OOM.

---

**Q13. What is the difference between precision and accuracy?**

- **Answer:** Precision is how many bits a number is stored in. Accuracy is how close the model's *outputs* are to correct. High precision does not guarantee accuracy — a tight cluster of darts on the wrong part of the board is precise and inaccurate.
- **Why asked:** It is the instructor's own framing and it tests conceptual hygiene.
- **Trap:** Using them interchangeably in an answer — which is exactly the confusion the question is designed to expose.

---

**Q14. What is a floating-point format's structure, and what makes bf16 different from fp16?**

- **Answer:** A float is sign + exponent + mantissa. fp16 has 5 exponent and 10 mantissa bits; bf16 has 8 exponent and 7 mantissa bits. bf16 has the same dynamic range as fp32 (so it does not overflow) at the cost of mantissa precision; fp16 has more precision in a narrower range.
- **Why asked:** Explains why bf16 is the training default and fp16 the inference default.
- **Trap:** Saying "fp16 is more accurate so training should use fp16". bf16's range is what stops gradient overflow; precision at 7 mantissa bits is adequate because gradients are noisy anyway.

---

**Q15. Name three things quantization makes smaller or faster.**

- **Answer:** Model size on disk (up to 8× from fp16 to int4), memory footprint at load, and decode throughput (because decode is memory-bandwidth-bound and reads fewer bytes per token). Indirectly: cost per token, via higher concurrency.
- **Why asked:** Checks that the candidate knows it is about bandwidth, not FLOPs.
- **Trap:** "4× smaller means 4× faster." It does not — the kernel dequantizes into FP16 and runs on FP16 tensor cores.

---

**Q16. What is GGUF?**

- **Answer:** A single-file model container: header, metadata key-value pairs, a tensor-info table and the tensor data. It carries the hyperparameters, the tokenizer and the chat template alongside the weights, and is memory-mappable. It replaced GGML's unversioned `.bin` format in August 2023.
- **Why asked:** Tests whether GGUF is understood as a *format*.
- **Trap:** Calling GGUF a quantization method. The method is the k-quant or legacy quant type *inside* the container.

---

**Q17. What was GGML?**

- **Answer:** Georgi Gerganov's Machine Learning library — a C tensor library for CPU inference — and the name of the old `.bin` file format it used. The library survives inside `llama.cpp` as its tensor backend; the `.bin` format is obsolete.
- **Why asked:** The "what came before" question; the follow-up is always "why was it replaced?"
- **Trap:** Claiming GGML is dead. The library is alive and is what `llama.cpp` runs on; only the file format was retired.

---

**Q18. Decode the name `Q4_K_M`.**

- **Answer:** `Q` = quantized; `4` = 4 bits per weight; `_K` = k-quant, a hierarchical block scheme with a super-block scale plus per-sub-block scales; `_M` = Medium size-mix, i.e. how much of the model is kept at higher precision. Not "medium quality".
- **Why asked:** It is a favourite screening question because it has four parts and most candidates know two.
- **Trap:** Reading `_M` as a quality grade. `_S/_M/_L` describe the *mix*, and they only exist for k-quants.

---

**Q19. What are `Q8_0` and `Q2_K`, and when would you use each?**

- **Answer:** `Q8_0` is 8-bit legacy block quantization — essentially lossless (Δppl < 0.01) and about half the fp16 size; use it when quality must not move. `Q2_K` is ~2.6 bits/weight and is a visible quality cliff (Δppl 1–5+); use it only when nothing else fits, and add an `imatrix` or use a higher bit-width instead.
- **Why asked:** Checks that the candidate has a sense of the accuracy-vs-bits curve.
- **Trap:** Calling `Q2_K` "half of `Q4_K_M`, so twice as small and a bit worse". It is worse in a different regime, not proportionally.

---

**Q20. What is `bitsandbytes` and what does it do at load time?**

- **Answer:** A library that quantizes a model *as it is loaded* — `load_in_4bit` (NF4) or `load_in_8bit` (LLM.int8()). No calibration, no conversion step, no output artifact; it also provides the NF4 base that QLoRA trains adapters on.
- **Why asked:** It is the path 90% of practitioners touch first, and its limitation is the real question.
- **Trap:** Expecting it to produce a portable quantized checkpoint. It does not — the "quantized model" is the fp16 checkpoint plus a config, quantized on every load.

---

**Q21. What is QLoRA?**

- **Answer:** 4-bit NF4 quantization of a frozen base model plus LoRA adapters trained in bf16, with paged optimizers and double quantization. It is the one case where you "train a quantized model" — and strictly, you train *adapters around* a frozen quantized base.
- **Why asked:** It is the bridge between quantization and fine-tuning, and it comes up in every fine-tuning interview.
- **Trap:** Saying QLoRA modifies the quantized weights. It does not; the base never receives a gradient.

---

**Q22. Can you fine-tune a GPTQ or AWQ model?**

- **Answer:** No. The stored integer codes have no gradient path and no optimizer state. You must fine-tune the fp16 base and re-quantize: **fp16 → fine-tune → merge → quantize**. The only training-on-quantized path is QLoRA over a bitsandbytes NF4 base.
- **Why asked:** This is the most common *pipeline order* mistake in practice.
- **Trap:** "You can dequantize it back and fine-tune." You can dequantize, but the information is gone; you would be starting from a degraded model.

---

**Q23. What does `group_size` control?**

- **Answer:** How many contiguous weights along the input dimension share one `(s, z)` pair. Smaller = finer grid = better accuracy and more metadata; `128` is the standard default, `32` is common at 3-bit.
- **Why asked:** It is a knob every practitioner touches and few can explain.
- **Trap:** The metadata arithmetic — `group_size=128` costs `2 × 2 bytes / 128 = 0.03125` bytes/param = **0.25 bits/param**, so a "4-bit" model is really 4.25 bits.

---

**Q24. What is per-channel quantization and why does it help?**

- **Answer:** One scale per output channel (row) instead of one per tensor. It confines an outlier to its own row instead of inflating the grid for the whole tensor — in CS-10's worked example, a 24.6% worst-case relative error drops to 1.4%, an MSE improvement of ~480×, for four bytes per row.
- **Why asked:** It is the cheapest accuracy win in the field, and the number makes the point.
- **Trap:** Confusing per-channel (one scale per row) with per-group (one scale per block *within* each row).

---

**Q25. What is the outlier problem in LLM quantization?**

- **Answer:** LLM activations contain a small number (0.1–1%) of channels whose magnitudes are 20–100× the median — and in the largest models, over 1000. A per-tensor scale must cover them, so the step grows by the same factor, all other values lose precision, and `log2(60) ≈ 5.9` bits of an 8-bit budget are spent on the outlier itself.
- **Why asked:** It is the central problem of the field. If a candidate does not know it, nothing above L2 is reachable.
- **Trap:** Saying outliers are rare random noise. They are systematic, stable across inputs, and absent below ~6.7B parameters — a structural property, not noise.

---

**Q26. What is LLM.int8()?**

- **Answer:** A mixed-precision decomposition (Dettmers et al., 2022, in `bitsandbytes`): for each matmul, dimensions whose activation magnitude exceeds `6.0` — about 0.1% of them — are extracted and computed in FP16, while the remaining 99.9% run in INT8. The two results are summed.
- **Why asked:** It is the canonical answer to "how do you handle outliers?"
- **Trap:** Calling it a rounding scheme. It is a decomposition, and that is why it keeps ~85% of the throughput instead of collapsing.

---

**Q27. What is NF4?**

- **Answer:** 4-bit NormalFloat: a 16-level codebook whose levels are the quantiles `Φ⁻¹(k/16)` of a standard normal distribution. It is the Lloyd-Max-optimal scalar quantizer for normally distributed weights, and it is what QLoRA quantizes its frozen base with.
- **Why asked:** It is the one 4-bit scheme that is *not* uniform, and it is where QLoRA's memory story comes from.
- **Trap:** Describing it as "4-bit like int4". It is a lookup-table codebook with a per-block scale (and, with double quantization, quantized scales).

---

**Q28. What is double quantization?**

- **Answer:** Quantizing the *quantization scales themselves*. NF4 uses blocks of 64 weights, each with an fp32/fp16 scale; double quantization stores those scales as 8-bit values with their own second-level scales, recovering about **0.373 bits per parameter**.
- **Why asked:** It is the specific trick that makes a 7B QLoRA fine-tune fit in ~6 GB.
- **Trap:** Assuming it costs accuracy. It is free money in the memory budget.

---

**Q29. What is the difference between static and dynamic PTQ?**

- **Answer:** Static PTQ computes activation ranges from calibration data once, at conversion time, and bakes them in. Dynamic PTQ computes activation ranges *on the fly*, per batch, at inference — no calibration data needed, but a per-batch range computation adds latency.
- **Why asked:** It is the taxonomy behind `torch.quantization.quantize_dynamic`, which is in the course's notebook.
- **Trap:** Assuming one is strictly better. Dynamic is the pragmatic default for LSTMs/MLPs; static is what you need for maximum throughput, and it is what needs the calibration set.

---

**Q30. What is the difference between W8A8 and W4A16?**

- **Answer:** `WxAy` = weights in `x` bits, activations in `y` bits. W8A8 quantizes both and can use INT8 tensor-core *compute* (throughput win). W4A16 keeps activations in FP16 and dequantizes 4-bit weights into FP16 inside the matmul (memory and bandwidth win only).
- **Why asked:** It is the notation that separates people who have read about serving from people who have done it.
- **Trap:** Calling a GPTQ or AWQ model "a 4-bit model" and implying 4-bit arithmetic. It is W4A16 — 4-bit *storage*.

---

**Q31. What is a quantization-aware training "fake quantize" module?**

- **Answer:** A module inserted into the forward pass that quantizes a tensor and immediately dequantizes it, so the network computes with real fp values but *feels* the quantization error. It carries observers that record the ranges during training, and its backward pass uses the straight-through estimator.
- **Why asked:** It is the mechanism behind `prepare_qat`, and the next question is always about the gradient.
- **Trap:** Thinking fake quantization changes the model's weights to integers. It never stores integers; `convert` is what packs them.

---

**Q32. What is the straight-through estimator?**

- **Answer:** The convention that defines the derivative of `round()` (and of the whole quantizer) to be the identity: forward `q = round(x/s)`, backward `∂q/∂x := 1` on the unclipped interval. It lets gradients pass through a non-differentiable operation.
- **Why asked:** It is the single most technical L1/L2 boundary question in this module.
- **Trap:** Saying "we ignore the gradient of rounding" — if you did, the gradient would be *zero*, and the model would not train at all. The STE does not ignore it; it *replaces* it with 1.

---

**Q33. What is the accuracy-vs-bits rule of thumb?**

- **Answer:** 8-bit is effectively free (Δppl < 0.01); 6- and 5-bit are free in practice; **4-bit costs 0.05–0.15 perplexity with a good method and is the deployment point**; 3-bit costs 0.2–0.6 and needs `group_size=32`; 2-bit is a cliff (Δppl 1–5+) unless you use rotation or QAT.
- **Why asked:** It is a calibration question — do you know what is safe to ship?
- **Trap:** Quoting a single number as universal. The curve is a comparison between quantization error and the model's redundancy, and redundancy scales with parameters — a 4-bit 70B is lossless while a 4-bit 1B is not.

---

**Q34. Which is the standard format for CPU inference, and which for GPU serving?**

- **Answer:** CPU/Mac/edge: **GGUF** consumed by llama.cpp / Ollama / LM Studio. GPU serving: **GPTQ** or **AWQ** over `auto-gptq`/`autoawq` with ExLlamaV2, Marlin or the GEMM kernels, or a compiled TensorRT-LLM engine for maximum throughput. **bitsandbytes** for experiments and QLoRA.
- **Why asked:** Deployment-mapping question. It tests whether the candidate has shipped anything.
- **Trap:** Recommending GPTQ for a Mac. There is no GPU-kernel path there; GGUF + Metal is the answer.

---

**Q35. What is the imatrix in llama.cpp?**

- **Answer:** An importance matrix — activation statistics collected from a calibration corpus with `llama-imatrix`, then passed to `llama-quantize` so that the quantization error on weights the activations care about is penalized more. It costs about ten minutes of CPU and buys 5–15% perplexity at 2–3 bits.
- **Why asked:** It is the highest-value ten minutes in GGUF quantization and most candidates have never heard of it.
- **Trap:** Confusing it with calibration data. The imatrix is *derived from* calibration data, not the data itself.

---

## Level 2 — Working Engineer (32 questions)

**Q36. Quantize `[0.0234, −0.1456, 0.7891, −1.2345, 0.5123, −0.0678, 0.3345, −0.9123]` to symmetric INT8. Show the scale, the codes, and the max error.**

- **Answer:** `max|w| = 1.2345`, `s = 1.2345/127 = 0.0097205`. Codes: `[2, −15, 81, −127, 53, −7, 34, −94]`. Dequantized: `[0.019441, −0.145807, 0.787358, −1.234500, 0.515185, −0.068043, 0.330496, −0.913724]`. Max absolute error `0.004004` (on `0.3345`), MSE `5.649e−6`. The largest-magnitude weight is exact because it defines the scale.
- **Why asked:** It is *the* whiteboard task for this module. They want to watch you compute a scale and a round without a calculator.
- **Trap:** Forgetting that `w = −1.2345` maps to `−127`, not `−128` — the `[−127,127]` convention. And rounding `w/s` correctly for negative values (round-half-to-even is not what you want; round-half-away-from-zero is what `torch.round` does for `.5` — check your framework).

---

**Q37. Now quantize the same vector asymmetrically to uint8. Why is the step smaller but the MSE worse?**

- **Answer:** `s = (0.7891 − (−1.2345))/255 = 0.0079357`, `z = round(1.2345/0.0079357) = 156`, codes `[159, 138, 255, 0, 221, 147, 198, 41]`, MSE `7.356e−6` — worse than symmetric despite the finer step. Because `z` must be an integer, the reconstruction grid is offset from the data by up to `s/2`, and the errors come out *systematic* (all the same sign) rather than as zero-mean noise. A biased small error can have a larger second moment than an unbiased larger one.
- **Why asked:** It is the discriminator between "I read the formula" and "I understand what rounding the zero-point does".
- **Trap:** Claiming the smaller step must produce a smaller error. It is the *total* error that matters, and quantization error has a bias term most people never write down.

---

**Q38. What happens if you take that uint8 scheme (`z = 156`) and store the codes as signed int8?**

- **Answer:** Six of the eight values saturate at `127`, every dequantized value collapses toward `−0.230`, max error becomes `1.019`, and MSE becomes `0.2508` — **44,000× worse**. Nothing raises an exception; the tensor is a valid int8 tensor and inference runs.
- **Why asked:** It tests whether you know that a zero-point can be *out of range* and what the failure looks like.
- **Trap:** Assuming a framework would catch it. Most do not; the clip happens inside the casting op and produces a plausible-looking tensor.

---

**Q39. Compute the error bound for a tensor spanning `[−1, 1]` at 4 bits and at 8 bits. What is the MSE ratio?**

- **Answer:** 4 bits: `Δ = 2/15 = 0.1333`, `MSE = Δ²/12 = 1.48e−3`. 8 bits: `Δ = 2/255 = 0.007843`, `MSE = 5.13e−6`. The ratio is `(255/15)² = 289×`. Going from 4 to 8 bits shrinks the step 17× and the MSE 289×.
- **Why asked:** It is the derivation behind the whole "4-bit is nearly free, 2-bit is a cliff" intuition.
- **Trap:** Saying "twice the bits halves the error". Four more bits is a 17× smaller step and a 289× smaller MSE.

---

**Q40. A tensor has one outlier 60× larger than everything else. What does that cost you?**

- **Answer:** The step grows 60× (`s = max|x|/q_max`), so the MSE on the ordinary values grows `60² = 3600×`, and `log2(60) ≈ 5.9` bits of an 8-bit budget are spent representing the outlier. Your effective precision on everything else is ~2.1 bits.
- **Why asked:** It is the arithmetic that motivates every outlier mitigation.
- **Trap:** "Just clip it." Clipping a magnitude-60 value to 3 gives it a 95% error — and that value may be a load-bearing attention sink.

---

**Q41. Write the min-max quantization function in PyTorch, including the guard everyone forgets.**

- **Answer:**
```python
def quantize_tensor(t, num_bits=8):
    qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1        # [-128, 127]
    min_val, max_val = t.min(), t.max()
    # the epsilon guard: a dead ReLU channel makes min == max == 0, and s would be 0/0
    scale = (max_val - min_val) / (qmax - qmin + 1e-8)
    zero_point = torch.round(-min_val / scale).to(torch.int32)
    q = torch.clamp(torch.round(t / scale) + zero_point, qmin, qmax).to(torch.int8)
    return q, scale, zero_point

def dequantize_tensor(q, scale, zero_point):
    return (q.float() - zero_point) * scale
```
- **Why asked:** It is the course's own notebook code, and the `1e-8` is the detail that separates a working implementation from a `NaN`.
- **Trap:** Omitting the epsilon. A tensor whose activations are all zero (a dead channel, a zero-initialized adapter) gives `scale = 0` and then `0/0`.

---

**Q42. Your quantized tensor has a scale of 0 after running that function. What happened?**

- **Answer:** The tensor's `max == min`. Either the tensor is constant (an all-zero dead ReLU channel, or a weight tensor at initialization), or your `min`/`max` were computed over the wrong axis (an empty slice, or a reduction bug).
- **Why asked:** It is the first real bug everyone hits when they implement quantization by hand.
- **Trap:** Only adding the epsilon but not checking whether the *axis* is right. An epsilon turns a loud `NaN` into a silently wrong per-tensor scale.

---

**Q43. Write the straight-through estimator as a PyTorch `autograd.Function`.**

- **Answer:**
```python
import torch

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x.clamp(-127, 127))

    @staticmethod
    def backward(ctx, grad_output):
        # the STE: the derivative of a quantizer is the identity where it is not clipping
        return grad_output
```
- **Why asked:** It is the single most-tested piece of QAT code, and it is four lines.
- **Trap:** Returning `grad_output * (x.abs() <= 127)` — the clipping mask is a legitimate variant, but if you write it without stating your convention, an interviewer will ask which one the framework you named uses. Also: `ctx` must be the first argument even when unused.

---

**Q44. What does `torch.quantization.prepare_qat` do, and what must happen between it and `convert`?**

- **Answer:** `prepare_qat` fuses modules, inserts `FakeQuantize` modules with observers, and switches the model to training mode. Between `prepare_qat` and `convert` you must **train**: the observers need to see data with the loss active before their ranges are frozen, and the weights need to adapt to the simulated quantization error. `convert(model.eval())` then bakes the frozen scales into real integer ops.
- **Why asked:** The two-phase API is the part people get wrong when they copy a QAT script.
- **Trap:** Calling `convert` immediately — you get a model quantized with uninitialized ranges. And forgetting `.eval()` before `convert`, which is a documented requirement for the observers to freeze.

---

**Q45. What qconfig would you pick for a server CPU versus a Raspberry Pi?**

- **Answer:** `get_default_qat_qconfig('fbgemm')` for x86 servers (Facebook GEMM, uses AVX2/AVX-512) and `'qnnpack'` for ARM (mobile/embedded, which for Raspberry Pi means the ARM CPU path). The engine string must match the hardware, or you get slow fallback kernels or a load failure.
- **Why asked:** It is a two-value fact that shows you have actually run QAT rather than read about it.
- **Trap:** Assuming the qconfig is portable. It is not — `fbgemm` artifacts do not run on ARM, and `qnnpack` does not use AVX.

---

**Q46. How do you load a 4-bit model with bitsandbytes, and what do the four config flags mean?**

- **Answer:**
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,                       # quantize at load; no artifact is produced
    bnb_4bit_quant_type="nf4",               # NF4 codebook (vs "fp4"): better for Gaussian weights
    bnb_4bit_compute_dtype=torch.bfloat16,   # what dequantized values are cast to for the matmul
    bnb_4bit_use_double_quant=True,          # quantize the block scales too: -0.373 bits/param
)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct",
                                             quantization_config=bnb, device_map="auto")
```
- **Why asked:** These four flags are the QLoRA defaults and the most-copied config in the ecosystem.
- **Trap:** `bnb_4bit_compute_dtype=torch.float16` on a model trained in bf16 — it can overflow on long sequences. Use bf16 unless you have a reason not to.

---

**Q47. Why does bitsandbytes produce no file you can ship?**

- **Answer:** Because it quantizes at load time from the original fp16 checkpoint. There is no conversion pass, no calibration, and no serialized quantized weights — the "model" is the fp16 checkpoint plus the config. It also means the load is slower and the first forward pass pays a dequantization cost.
- **Why asked:** It is the practical consequence of the design, and it decides whether bnb is a fit for your deployment.
- **Trap:** Trying to `save_pretrained` a bnb model and expecting the 4-bit weights. You get the fp16 weights back (or an error), not a 4-bit artifact.

---

**Q48. Write the GPTQ configuration for a 7B model and explain each field.**

- **Answer:**
```python
from auto_gptq import BaseQuantizeConfig
cfg = BaseQuantizeConfig(
    bits=4,              # weight bits. 4 = the deployment point; 3 needs group_size=32
    group_size=128,      # weights per scale. 128 standard; 32 at 3-bit; -1 = per-channel
    desc_act=False,      # == act_order: quantize columns by decreasing activation importance
                         #    True = better quality, ~10% slower inference, breaks some kernels
    damp_percent=0.01,   # diagonal loading on H before Cholesky. Raise to 0.05-0.1 if singular
    sym=True,            # symmetric (z=0). The default and the hardware-friendly choice
    true_sequential=True # per-layer sequential quantization using activations from prior layers
)
```
- **Why asked:** The `damp_percent` and `desc_act` fields are where candidates reveal whether they have only copied a tutorial.
- **Trap:** Setting `desc_act=True` "because it is better quality" and then failing to load it in an older ExLlamaV2/Marlin build. Check the kernel before flipping the flag.

---

**Q49. What is `damp_percent` and when do you change it?**

- **Answer:** It adds `damp_percent × mean(diag(H))` to the diagonal of the Hessian before inversion. `H` is often ill-conditioned or singular — a calibration channel that is constant, or a layer with fewer calibration tokens than input dimensions, makes it exactly singular — and the Cholesky factorization fails or produces garbage. The default `0.01` handles most cases; raise it to `0.05–0.1` if you see a singular-Hessian error or a suspiciously bad result on one layer.
- **Why asked:** It is the "do you know what your library is doing" question for GPTQ.
- **Trap:** Raising it blindly. Too much damping biases the solution toward plain RTN and you lose the error compensation that makes GPTQ worth using.

---

**Q50. Write the AWQ quantization config and explain what is *not* being quantized.**

- **Answer:**
```python
quant_config = {
    "zero_point": True,    # one zero-point per group (asymmetric). Slightly better; GEMM only
    "q_group_size": 128,   # weights per scale — note the flag is q_group_size, not group_size
    "w_bit": 4,            # WEIGHT bits
    "version": "GEMM",     # "GEMM" = batched GPU kernel; "GEMV" = batch-size-1 path
}
```
The **activations** are not quantized — this is W4A16. Dequantization happens inside the matmul, which is exactly why the per-channel rescaling trick costs nothing at inference.
- **Why asked:** It tests both the config and the concept, and `GEMM` vs `GEMV` is a real production bug.
- **Trap:** Leaving `version="GEMV"` on a batched serving workload. It is not auto-detected, it produces *correct output at half the throughput*, and nothing in the logs tells you.

---

**Q51. You must serve a 7B on a single RTX 4090 (24 GB) at 8k context, batch 8. Will it fit at 4-bit? Show the arithmetic.**

- **Answer:** Weights 4-bit ≈ `7e9 × 0.5 × 1.06 ≈ 3.7 GB`. KV for a GQA 7B (32 layers, 8 KV heads, 128 head_dim) is `2·32·8·128·2 = 131,072 B = 128 KB/token`; 8 sequences × 8k tokens = 65,536 tokens × 128 KB = **8.4 GB**. Activations ~0.5 GB, CUDA context ~0.6 GB, slack 10% ~1.3 GB. Total ≈ **14.5 GB** — fits in 24 GB with 9.5 GB to spare.
- **Why asked:** It tests whether you do the KV-cache arithmetic, which is the whole answer.
- **Trap:** Stopping at "3.7 GB of weights, easily fits". The KV cache is 2.3× the weights here, and at 32k context it would be 33 GB — an OOM on the same card with the same model.

---

**Q52. Same question, but the model is Llama-2-7B (no GQA). What changes?**

- **Answer:** `2·32·32·128·2 = 512 KB/token` — 4× the KV. At 8k × batch 8 that is **33.6 GB**, which does not fit. You must reduce the batch to 2 (8.4 GB) or use FP8 KV (16.8 GB) or drop to 4k context.
- **Why asked:** Same formula, different architecture, opposite conclusion. It checks that you used the formula rather than a memorized number.
- **Trap:** Assuming a 7B is a 7B. The `h_kv` difference between Llama-2 and Mistral/Llama-3 is the single largest architectural factor in long-context memory.

---

**Q53. What is the exact command to convert an HF model to GGUF and quantize it to `Q4_K_M`, with an imatrix?**

- **Answer:**
```bash
# 1. HF -> GGUF f16
python -m llama_cpp.convert_hf_to_gguf ./llama-hf --outfile m-f16.gguf --outtype f16
#    (from a llama.cpp source tree: python3 convert_hf_to_gguf.py ... — the old name
#     convert.py is deprecated)

# 2. build the importance matrix from YOUR OWN calibration text
llama-imatrix -m m-f16.gguf -f calibration.txt -o m.imatrix -ngl 99

# 3. quantize, using the imatrix
llama-quantize --imatrix m.imatrix m-f16.gguf m-Q4_K_M.gguf Q4_K_M

# 4. run it
llama-cli -m m-Q4_K_M.gguf -p "What is quantization?" -n 128 -ngl 99
```
- **Why asked:** Command-level fluency. It is a copy-paste question with a version trap.
- **Trap:** Using `convert.py` (renamed to `convert_hf_to_gguf.py` in 2024) or `./main` (renamed to `llama-cli`/`llama-server`). Both still appear in older tutorials and both will fail on a current checkout.

---

**Q54. What does `-ngl` do in llama.cpp?**

- **Answer:** It sets how many layers to offload to the GPU. `-ngl 99` offloads all of them; `-ngl 0` runs purely on CPU. Partial offload is how you run a model that does not fit in VRAM — the GPU handles the first N layers and the CPU the rest, at the cost of a PCIe transfer per token.
- **Why asked:** It is the flag that makes "GGUF runs on a 24 GB card with a 40 GB model" true.
- **Trap:** Assuming offload is free. Every offloaded boundary costs bandwidth per token, and decode throughput drops sharply with the number of CPU-resident layers.

---

**Q55. How do you know which quantization to pick for a 7B GGUF?**

- **Answer:** `Q4_K_M` is the default — ~4.8 bits/weight including block metadata, ~4.1 GB for a 7B, Δppl ≈ 0.05. Move to `Q5_K_M` or `Q6_K` if RAM is available (both are effectively lossless), `Q8_0` if RAM is free and you want zero doubt, and only go below `Q4_K_M` (to `Q3_K_M`, or `Q2_K` + imatrix) when the memory budget forces it.
- **Why asked:** It is the practical decision every GGUF user makes, and the answer is a specific ordering.
- **Trap:** Choosing `Q4_K_S` to save 0.2 GB. The whole point of `_M` is to spend those 0.2 GB where the accuracy is; the `_S` variants exist for a memory-constrained edge case, not as a default.

---

**Q56. How do you measure whether a quantized model actually got worse?**

- **Answer:** Not with perplexity alone. Compare against the fp16 model on: (1) **mean KL divergence** of next-token distributions on held-out domain text, (2) **p95 KL** (catches long-tail/long-context failures the mean hides), (3) **top-1 agreement**, (4) a **task gate** — IFEval, JSON-schema validity, needle-in-a-haystack at your max context, and (5) a **50-prompt side-by-side generation** read by a human.
- **Why asked:** The most common real-world mistake after quantization is shipping on a green perplexity number.
- **Trap:** Reporting only the mean KL. A model can have mean KL 0.02 and p95 KL 8.0 — that is a model that is fine on prose and broken on code, JSON and rare tokens.

---

**Q57. Write the KL-divergence comparison for a quantized model in ten lines.**

- **Answer:**
```python
import torch, torch.nn.functional as F

@torch.no_grad()
def kl(ref_logits, q_logits):                 # both fp32, shape [seq, vocab]
    lp = F.log_softmax(ref_logits, dim=-1)
    lq = F.log_softmax(q_logits,  dim=-1)
    kl = F.kl_div(lq, lp, log_target=True, reduction="none").sum(-1)   # per position
    return kl.mean().item(), kl.quantile(0.95).item(), (ref_logits.argmax(-1)
                                                        == q_logits.argmax(-1)).float().mean().item()
```
- **Why asked:** It is the one diagnostic that catches real quantization damage automatically.
- **Trap:** Materializing a `[seq, 128k]` log-softmax in one shot on a long sequence and OOMing. Chunk over the sequence, and cast both to fp32 — comparing a bf16 and a bf16 log-softmax gives you bf16 noise, not quantization damage. Also: `F.kl_div`'s argument order is `(input, target)` where `input` is *already* log-softmaxed — mixing it up gives a negative KL that looks fine.

---

**Q58. What does `torch.quantization.quantize_dynamic` do, and why did the course's MLP keep its accuracy with it?**

- **Answer:** It converts `nn.Linear`/`nn.LSTM` weights to `qint8` and computes the activation scale dynamically per batch, with no calibration and no model modification. On the course's toy MLP it left accuracy at 0.97 because the network is small and over-parameterized for `make_moons`; the weights quantize to 8 bits and the per-batch activation range is estimated on the actual input.
- **Why asked:** It is the course's own headline result, and the follow-up is "why did the manual PTQ fail then?" (answer: it was buggy — see Q59).
- **Trap:** Concluding "dynamic PTQ is lossless" in general. It is lossless *on that problem*, and 8-bit losslessness does generalize reasonably — but the 0.43 result from the same notebook is a bug, not a counterexample.

---

**Q59. The course's notebook reports PTQ dropping accuracy from 97% to 43%. Is that real?**

- **Answer:** No. 43% is below chance for a binary classifier, which means the decision boundary was destroyed, not merely perturbed — and the identical 43% appears for two structurally different code paths, which indicates a shared bug rather than two independent measurement failures. The likely causes: a per-tensor scale for every tensor including the final logit layer, and a uint8 zero-point path applied to signed pre-activations (clamping every negative value to zero). A correct per-tensor static PTQ on that MLP should land at ~0.93–0.97.
- **Why asked:** It is an adversarial question about whether you blindly repeat a course result.
- **Trap:** Defending the number, or dismissing the whole notebook. The notebook's *legitimate* findings — dynamic PTQ free, QAT recovering the last point, and a bad pipeline being catastrophic and silent — are all correct and valuable.

---

**Q60. What is the difference between quantizing weights and quantizing activations, in terms of difficulty?**

- **Answer:** Weights are static — you can compute a per-channel or per-group scale once, offline, with full knowledge of the distribution, and get an excellent grid. Activations are *runtime* and *per-token*: the scale must be fixed before you see the token (static) or computed on the fly (dynamic), a per-channel scale cannot be applied the same way, and the outliers are input-dependent. That asymmetry is why weight-only 4-bit (W4A16) is solved and 8-bit activation quantization (W8A8) still needed SmoothQuant.
- **Why asked:** It is the conceptual reason the field settled on W4A16, and it sets up every "why not W4A4" follow-up.
- **Trap:** "Activations are just another tensor." They are not — you cannot per-channel-quantize a tensor you have not seen yet, and that is exactly where the outlier problem bites.

---

**Q61. Name the four knobs you can turn when a quantized model is not good enough.**

- **Answer:** (1) **Bits** — 4 → 5 or 6 (biggest, most expensive change). (2) **Group size** — 128 → 64 → 32 (accuracy for metadata and speed). (3) **`act_order`/`desc_act`** — on (better quality, ~10% slower, kernel-dependent). (4) **Calibration** — more samples and, more importantly, samples that look like production. Plus a fifth that is not a knob: exclude the `lm_head`/embeddings from quantization, or switch method (GPTQ ↔ AWQ).
- **Why asked:** It is the practical answer to "make it better", and it tests whether you have actually debugged a quality regression.
- **Trap:** Jumping straight to more bits. Calibration corpus mismatch is a more common cause than insufficient precision, it is free to fix, and it is invisible on a generic benchmark.

---

**Q62. How much slower is GPTQ's quantization pass than AWQ's, and why?**

- **Answer:** Roughly 2–3× slower. GPTQ accumulates and inverts an input covariance matrix (`H = 2XXᵀ`, a `d×d` matrix per layer) with a damped Cholesky factorization and then does blocked error-compensation updates across the weight matrix. AWQ computes only per-channel activation magnitudes and does a small 1-D scale search — no matrix inversion anywhere.
- **Why asked:** It is the practical reason teams pick AWQ even when the quality is comparable.
- **Trap:** Assuming faster means worse. AWQ is both faster to produce *and* frequently ahead at 4-bit on instruction-tuned models; GPTQ's advantage appears at 3-bit, where the error compensation earns its cost.

---

**Q63. What does `zero_point: True` vs `False` do in AWQ?**

- **Answer:** `True` stores an asymmetric zero-point per group (scale + zero-point), which fits a one-sided or offset distribution better; `False` is symmetric, scale only. Empirically `True` is slightly better and is the notebook's setting, but it is only supported by the GEMM kernel path.
- **Why asked:** It is a small flag with a real accuracy effect and a kernel constraint.
- **Trap:** Setting `zero_point=True` and running the GEMV path, or vice versa. The kernel/flag mismatch produces plausible-but-wrong output.

---

**Q64. Your AWQ model loads but generates slightly worse text than the fp16 model, and there is no error. What do you check?**

- **Answer:** In order: (1) `version` — `"GEMM"` vs `"GEMV"` must match your batch shape; (2) the calibration data used — was it your domain, and how many samples; (3) whether the tokenizer/`generation_config.json` was copied into the quantized directory (a missing one changes the prompt format, not the weights); (4) the kernel actually selected at load; (5) `q_group_size` — was it 128 when 32 was needed.
- **Why asked:** It is the exact shape of a real production regression, and it has no stack trace.
- **Trap:** Re-quantizing immediately. Three of the five causes are configuration, not quantization, and re-quantizing with the same config reproduces the same problem.

---

**Q65. How do you quantize a model that does not fit on your GPU?**

- **Answer:** Options, in rough order of preference: (1) quantize on a bigger rented GPU for an hour (cheapest in wall-clock); (2) shard with `device_map="auto"` across several GPUs, accepting the slowdown in the calibration pass; (3) for GPTQ, use an implementation that streams or offloads layers to CPU RAM (slower but possible, requires ~2 bytes/param of host RAM); (4) for GGUF, do the whole thing on CPU with `llama-quantize`, which is CPU-native and needs ~2 bytes/param of RAM; (5) use bitsandbytes, which never materializes a separate pass at all.
- **Why asked:** It is a real constraint on any model above 13B, and the CPU/RAM answer (GGUF) is the one people forget.
- **Trap:** Assuming a 70B can be quantized on a 24 GB card. The fp16 model must be resident somewhere — that is ~140 GB — and that is the binding constraint, not the GPU.

---

**Q66. What is the difference between `bnb_4bit_quant_type="nf4"` and `"fp4"`?**

- **Answer:** `nf4` uses the NormalFloat codebook — 16 levels at the quantiles of `N(0,1)` — which is information-theoretically optimal for approximately Gaussian weights and is what QLoRA validated. `fp4` uses a standard 4-bit float layout (1 sign, 2 exponent, 1 mantissa), which has a wider range but much coarser resolution near zero. NF4 wins for LLM weights.
- **Why asked:** It is a one-flag decision that most people copy without knowing.
- **Trap:** Choosing `fp4` because "float is more expressive". For a zero-mean approximately-normal weight distribution, the quantile codebook strictly dominates.

---

**Q67. A colleague quantizes a model to `Q4_K_M`, uploads it, and a user reports the model "ignores instructions". What is your first hypothesis?**

- **Answer:** The **chat template**. A GGUF whose `tokenizer.chat_template` metadata is missing or wrong falls back to raw text completion, so the instruction-tuned model receives an unformatted prompt and continues the text instead of answering. It generates fluently and never errors. Verify by printing the exact rendered prompt from the GGUF runtime and comparing it with `tokenizer.apply_chat_template` from the original HF repo.
- **Why asked:** It is the single most common GGUF production failure and it produces *no* error message.
- **Trap:** Immediately re-quantizing at a higher bit-width. The weights are almost certainly fine; a template mismatch looks exactly like a broken model.

---

## Level 3 — Senior Engineer (25 questions)

**Q68. Derive why quantization MSE is quadratic in the range and linear in `2^−b`.**

- **Answer:** For a uniform grid with closed interval `[q_min, q_max]` spanning `2^b − 1` steps, `Δ = (max − min)/(2^b − 1)`. A value falling uniformly within its cell gives an error `e ~ U(−Δ/2, Δ/2)`, so `E[e] = 0` and `E[e²] = Δ²/12`. Substituting: `MSE = (max − min)² / (12(2^b − 1)²)`. So MSE is *quadratic in the range* (an outlier of factor `k` costs `k²`) and *inversely quadratic in the code count* (each extra bit divides MSE by ~4, i.e. −6.02 dB). That is the whole shape of the accuracy-vs-bits curve: constant multiplicative improvement per bit, versus catastrophic multiplicative degradation per unit of range.
- **Why asked:** It is the derivation that turns "4-bit is worse than 8-bit" into a quantitative statement, and it is the basis of every method in the field.
- **Trap:** Presenting `Δ²/12` as an exact per-tensor prediction. It is the expectation for a *uniform* source; real LLM weights are approximately Gaussian and the true constant differs — Gaussian sources have `D ≈ c·σ²·2^{−2b}`, which is why the *shape* is right and the *constant* is not.

---

**Q69. Why is the largest-magnitude weight in a tensor always quantized exactly, and why is that a misleading comfort?**

- **Answer:** Because it defines the scale: `s = max|x|/q_max`, so `x_max/s = q_max` exactly. It is a comfort because the error is not uniform — it is *proportional to the tensor's dynamic range*, so the largest values are lucky and the smallest values absorb all the damage. In the worked example the largest weight had 0.00% error and the smallest had 16.9%. That asymmetry is the reason per-channel and per-group granularity matter: the small values in a row are collateral damage of the big values sharing the same grid.
- **Why asked:** It tests whether you think about the *distribution* of the error or only its mean.
- **Trap:** Assuming relative error is bounded. Absolute error is bounded by `Δ/2`; relative error is bounded by `Δ/(2|x|)` and explodes as `x → 0`. A weight of `0.001` in a tensor with `max = 1.2` has a relative error of up to 480%.

---

**Q70. Explain the outlier problem mechanically — where do massive activations come from?**

- **Answer:** The residual stream is a running sum of every layer's contribution, and RMSNorm normalizes it by its norm. A channel that consistently carries a large value therefore survives normalization and acts as a **fixed bias** that any downstream layer can read without needing a weight to produce it. Attention heads that need a "write nothing useful" path learn to attend to a small number of **sink** tokens (the first token, delimiters) and dump probability there, and those heads write into such a channel. The result is a handful of channels doing bookkeeping for the network — stable across inputs, concentrated in channel indices, and *absent below ~6.7B parameters* because smaller models have less capacity to develop that structure.
- **Why asked:** It is the deepest "why" in the module and a favourite senior question, because it distinguishes people who know the *phenomenon* from people who know the *mechanism*.
- **Trap:** Calling them "outliers in the statistical sense" (rare, random, removable). They are structural, deterministic given the input distribution, and load-bearing — clipping them breaks the model.

---

**Q71. Why does per-channel granularity solve the outlier problem for weights but not for activations?**

- **Answer:** For weights, the outlier is a property of a *static* matrix: if input channel `j` is an outlier, the weights in column `j` are the ones that need care, and you can give each row (or each group within a row) its own scale *before* deployment. For activations, the quantization happens at inference on a tensor you have not seen yet: you cannot compute a per-channel scale for the current token without a reduction over the whole sequence, which is exactly the kernel you were trying to avoid, and a per-channel activation scale must be *shared across tokens* to be usable — which is the "static" PTQ scheme and reintroduces the outlier's effect on every other channel. So activation outliers need **migration** (SmoothQuant moves the difficulty into the weights), **rotation** (spread it), or **mixed precision** (LLM.int8()).
- **Why asked:** It is the pivot from "quantization is easy" to "activation quantization is a research problem", and it is where the W8A8 vs W4A16 split becomes principled rather than arbitrary.
- **Trap:** "Just use per-channel for activations too." In W8A8 static quantization a per-token dynamic scale is possible (`torch.ao` calls it dynamic), but a per-*channel* scale over the sequence dimension is not what the kernels implement, and per-token scaling does not help when the outlier is in the same channel for every token.

---

**Q72. Explain the straight-through estimator and give two rigorous justifications and one failure mode.**

- **Answer:** The STE defines the quantizer's backward pass to be the identity: forward `q = round(clamp(x/s))`, backward `∂q/∂x := 1` on the unclipped interval. **Justifications:** (1) the reconstruction error `x̂ − x` is bounded by `Δ/2` and essentially uncorrelated with `x`, so the identity is the best *linear* estimator of `x` given `x̂` — training with it encourages the weights to stay in a regime where `Δ` is small; (2) the STE's gradient is an unbiased estimator of the gradient of the *smoothed* quantizer `E_ε[round(x+ε)]` with smoothing width `Δ`, and SGD tolerates bounded gradient noise, so it converges near a solution that is good for the true quantized network; (3) empirically it makes the weight distribution migrate toward bin centres, measurably reducing expected rounding error. **Failure mode:** when the bins are coarse relative to the gradient signal (2-bit), the passed-through gradient is wildly out of scale with the true error, and QAT stalls or diverges.
- **Why asked:** Any senior quantization interview contains this. The "why is 1 the right lie" part is what separates a memorized definition from understanding.
- **Trap:** Saying "we approximate the gradient as 1 because the derivative of round is 0 or undefined." That is the *set-up*, not the justification. The interviewer wants to hear why defining it as 1 is better than, say, skipping the layer or using a soft rounding function — the answer is the smoothing/unbiasedness argument.

---

**Q73. What exactly does GPTQ's `H` measure, and why does that make it better than rounding nearest?**

- **Answer:** `H = 2XXᵀ` is (twice) the covariance of the layer's *input activations* over the calibration set. The objective `min ‖WX − ŴX‖² = min tr((W − Ŵ)H(W − Ŵ)ᵀ)` weights each weight's error by how active its input channel is. RTN treats every weight as equally important; GPTQ knows that a large weight multiplying an always-zero input matters not at all, and that a modest weight multiplying a high-variance channel matters enormously. This is why GPTQ-4bit is dramatically closer to fp16 than RTN-4bit, and it is the same information AWQ uses (activation magnitude) in a cheaper, explicit form.
- **Why asked:** It is the conceptual core of the most widely deployed 4-bit method.
- **Trap:** Calling `H` "the Hessian of the loss". It is the Hessian of the *layer-reconstruction* objective, i.e. the input second-moment matrix — a different object from the training Hessian, and computable from forward passes only.

---

**Q74. Walk through GPTQ's greedy loop and explain where the error compensation enters.**

- **Answer:** For each column `j` of `W`: (1) round the column to `q_j`; (2) compute the residual `δ_j = w_j − q_j`; (3) update all remaining columns `w_{k>j} -= δ_j · (H⁻¹)_{j,k} / (H⁻¹)_{jj}`. The update is what "compensates": the output error introduced by quantizing column `j` is redistributed into the columns that have not yet been quantized, so the *accumulated* output error stays small even though individual weights are rounded coarsely. GPTQ's engineering contribution is doing this in blocks of `B=128` columns with one Cholesky factorization of `H⁻¹` per layer, plus diagonal damping for numerical stability — OBQ's exact per-column greedy is `O(d³)` per column and intractable at `d = 4096`.
- **Why asked:** It is the algorithm question, and the compensation step is the part candidates cannot fake.
- **Trap:** Describing GPTQ as "RTN with a better scale". It is a fundamentally different loop; the compensation step is what makes 3-bit usable at all.

---

**Q75. What does `act_order`/`desc_act` change, and what does it cost?**

- **Answer:** It permutes the columns by decreasing diagonal of `H` (i.e. by decreasing input-activation variance) and quantizes in that order, then permutes back for storage. The columns quantized *last* receive the most benefit from error compensation, so handing the most important columns the last position measurably reduces output error. The cost: the permutation destroys the contiguous layout a GPU kernel expects, so inference is ~10% slower, and some older ExLlamaV2/Marlin builds refuse to load such checkpoints at all.
- **Why asked:** It is a real flag with a real cost, and it is a favourite "do you know what you enabled?" question.
- **Trap:** Enabling it and forgetting to check kernel support — a `desc_act=True` checkpoint that the loader ignores is *plausible but degraded*, not an error.

---

**Q76. Explain AWQ's rescaling identity and why it is free at inference.**

- **Answer:** `W·X = (W·diag(s))·(diag(s)⁻¹·X)`. Scaling input channel `j`'s weight column up by `s_j` and dividing the corresponding activation channel down by `s_j` leaves the layer output unchanged in exact arithmetic. Choose `s_j > 1` for the salient channels (those with large `|X_j|`), and after the group's max-abs scale is computed, that column occupies more integer levels and is quantized more finely. It is **free** because activations are not quantized in W4A16 — the `diag(s)⁻¹` on the activation side can be folded into the preceding LayerNorm or Linear and is exact in FP16, and only the weight side ever sees an integer grid. That is the entire reason AWQ is a weight-only method.
- **Why asked:** It is the core insight of a method that is otherwise easy to describe vaguely as "AWQ protects the important channels".
- **Trap:** Thinking AWQ is "mixed precision done cheaply" in the sense of keeping some channels in fp16. It does not — every channel is 4-bit; the salient ones just get a better-positioned 4-bit grid.

---

**Q77. AWQ's `s_j = (max|X_j|)^α/(max|W_j|)^(1−α)`. What is `α` doing, and what is its relationship to SmoothQuant?**

- **Answer:** `α` balances how much of the dynamic range is carried by the activations versus the weights. `α = 1` puts all of it on the activation side (max protection of salient channels, no weight-side adjustment); `α = 0` ignores activations entirely; `α ≈ 0.5` splits it evenly. AWQ grid-searches `α` over a handful of values on a small held-out calibration set and keeps the one with the lowest output MSE. This is *identical* to SmoothQuant's formula — SmoothQuant uses the same `s_j` to make activations quantizable in a W8A8 setting, and AWQ uses it to give salient weight columns a finer grid in a W4A16 setting. They are the same idea in two regimes.
- **Why asked:** It is the "do you see the unified picture" question, and it is where a senior candidate stands out.
- **Trap:** Treating AWQ and SmoothQuant as competing methods. They target different schemes (W4A16 vs W8A8) and share the mechanism; a candidate who says "AWQ is SmoothQuant for weight-only quantization" is exactly right.

---

**Q78. Compare GPTQ and AWQ on the four axes that matter: quality, production cost, inference cost, and robustness.**

- **Answer:**
  - **Quality:** at 4-bit, comparable, with AWQ usually ahead on instruction-tuned and multimodal models; at 3-bit, GPTQ is clearly ahead because error compensation pays off as the grid coarsens; at 2-bit both fail.
  - **Production cost:** AWQ is ~2–3× faster (no Hessian, no Cholesky); GPTQ needs more peak memory for the covariance buffers.
  - **Inference cost:** equivalent once both are on Marlin/ExLlamaV2; GPTQ with `desc_act=True` pays ~10% for the permutation unless the kernel handles it.
  - **Robustness:** AWQ's 1-D per-layer scale search generalizes better across domains; GPTQ's Hessian solve depends on the calibration corpus more strongly and can overfit it — but GPTQ degrades more gracefully when the calibration set is small.
- **Why asked:** It is the practical decision table, asked as a comparison rather than as a definition.
- **Trap:** "AWQ is strictly better." At 3-bit and at very low calibration budgets, GPTQ wins, and the honest answer names the regime.

---

**Q79. What is the difference between `H = 2XXᵀ` in GPTQ and the Hessian in optimal brain pruning, and why does the distinction matter?**

- **Answer:** Both are second-derivative objects of the form `∂²E/∂w²`, but GPTQ's is the second derivative of a *layer-local reconstruction objective* `E = ‖WX − ŴX‖²` with respect to the weights, evaluated on the calibration input — which is exactly `2XXᵀ` and depends only on forward passes. The pruning/OBQ Hessian in the original literature is usually written for the *network* loss, which would require a backward pass and end-to-end coupling. The distinction matters because it is why GPTQ is a *post-training, forward-only* method that a practitioner can run in an hour with no labels — and why its objective is a proxy for the true loss rather than the true loss itself.
- **Why asked:** It is the "do you know which Hessian" question, and it is where candidates who memorized `H = 2XXᵀ` without understanding it get caught.
- **Trap:** Describing GPTQ as "gradient-based" or "needing a backward pass". Neither is true; the acronym confusion (Gradient Post-Training Quantization) usually precedes this mistake, and interviewers use it as a filter.

---

**Q80. Why is layer-wise quantization acceptable when the true objective is global?**

- **Answer:** Because the local layer-reconstruction error `‖WX − ŴX‖²` turns out to be an excellent proxy for the end-to-end loss — the empirical result that made post-training 4-bit LLMs possible. The intuition is that a Transformer's layers are approximately sequentially decomposable for the purpose of *error propagation*: the quantization error injected at layer `l` is small relative to the residual stream, so it propagates roughly linearly and the total damage is close to the sum of the per-layer damages. It is an approximation, not a theorem — and it is precisely why per-layer error should not be assumed additive across layers (errors compound through attention) and why you must measure end-to-end.
- **Why asked:** It tests whether you know the *limitation* of the method you are recommending, which is the definition of seniority in this topic.
- **Trap:** Claiming optimality ("layer-wise is optimal because of the chain rule"). It is a proxy that happens to work extremely well in practice, and it breaks down at very low bit-widths and in models with strong layer coupling (some MoE and multimodal architectures).

---

**Q81. Why does 4-bit work on a 70B and damage a 1B?**

- **Answer:** Because the damage is a *ratio* between the quantization error and the model's redundancy. The error is set by the weight distribution's dynamic range and the bit budget — roughly constant per parameter. The model's ability to absorb that error scales with its capacity and with the number of parameters sharing the burden of any given computation. A 70B has more redundant pathways, more heads to average over, and a smoother loss landscape, so a fixed per-weight perturbation produces a smaller output perturbation. A 1B model has little slack, and the same 4-bit recipe produces a visible quality drop. Practical corollary: **use `group_size=32–64` and expect a real delta below ~3B parameters**, and validate before shipping rather than assuming the model card's 7B numbers transfer.
- **Why asked:** It is a real and frequently-surprising effect, and it separates people who quantize 7B models from people who have shipped a small model.
- **Trap:** "Smaller models have fewer outliers, so they are easier." They have fewer *massive activations* but less redundancy — and empirically the redundancy effect dominates at 4-bit.

---

**Q82. What is the difference between GGUF and a "quantization format" like GPTQ?**

- **Answer:** They operate at different layers of the stack. GGUF is a **container** — header, metadata KV, tensor table, tensor data — that can hold weights in many quantization schemes (`Q4_K_M`, `Q8_0`, `F16`, and even `IQ*` imatrix-derived types). GPTQ is a **quantization algorithm** plus a checkpoint layout convention. The correct pairing is "GPTQ as a method, `.safetensors` or GGUF as a container" and "k-quant as a method, GGUF as its container". The confusion is so common that saying "I quantized it to GGUF" in an interview is read as a vocabulary failure.
- **Why asked:** It is the terminology question that most cleanly separates reading from understanding.
- **Trap:** Answering "GGUF is 4-bit". GGUF is not any bit-width; `general.file_type` and the per-tensor types in the tensor table determine everything.

---

**Q83. Why did GGUF replace GGML's `.bin` format? Name four design failures it fixed.**

- **Answer:** (1) **Hard-coded hyperparameters** — GGML loaders compiled `n_layer`, `n_head` etc. into the code, so a new architecture meant a code change and a recompile; GGUF stores them as metadata KV. (2) **No versioning** — no field to check, so an old file produced a cryptic failure; GGUF has `gguf_version`. (3) **A separate tokenizer file** — two downloads and a mismatch risk; GGUF embeds the vocab, merges and chat template. (4) **Not memory-mappable** — the whole file had to be read; GGUF's layout allows `mmap` so a 4-bit 7B starts on a 4 GB laptop. Plus extensibility: GGUF is self-describing, so a reader can skip keys it does not know.
- **Why asked:** It is the "why does this format exist" question, and the *chat template* consequence is the practically important one.
- **Trap:** Treating it as a pure size/quality change. The container redesign is about *extensibility and correctness*, not about compression.

---

**Q84. When would you choose GGUF over AWQ, and when would that be a mistake?**

- **Answer:** Choose GGUF when the target is CPU, Apple Silicon, a laptop, an air-gapped kiosk, or an environment where you cannot control the GPU stack — its advantages are memory-mapping, single-file delivery, partial GPU offload via `-ngl`, and broad runtime support (llama.cpp, Ollama, LM Studio, llamafile). It is a **mistake** when the target is a high-throughput GPU serving fleet: GPTQ/AWQ on Marlin/ExLlamaV2, or TensorRT-LLM, will be substantially faster per GPU, with proper continuous batching and paged attention. A second mistake: using GGUF when you intend to **fine-tune** afterwards — GGUF cannot be trained on.
- **Why asked:** It is a deployment-mapping question and it is the one where practitioners' habits (rather than their knowledge) show.
- **Trap:** "GGUF is just 4-bit GPTQ for CPU." The quantization schemes, the kernels, the granularity and the container are all different.

---

**Q85. A model is quantized to 4-bit and quality drops sharply on code and math but not on chat. What is happening?**

- **Answer:** Code and math are the **long-tail** tasks: they depend on rare tokens (`{`, `}`, `;`, `<`, digits, operators) and on precise multi-step composition where each step's small error compounds. Quantization error is roughly uniform in *absolute* terms across the vocabulary, but the loss is dominated by high-frequency tokens in ordinary prose — so perplexity barely moves while the probability mass on rare structural tokens shifts enough to break syntax and arithmetic. The diagnostic is **p95 KL**, not mean KL: run the KL script on code prompts and look at the tail. The fix is +1 bit, or `group_size` 128 → 32, or excluding the `lm_head`.
- **Why asked:** It is the "perplexity is not a gate" lesson in its most concrete form, and it is the most common real-world quantization regression.
- **Trap:** Concluding "the model is fine because perplexity moved by 0.03". The mean is the wrong statistic; you must look at the tail and at a task metric.

---

**Q86. Explain the difference between quantization error and *accumulated* quantization error in a deep network.**

- **Answer:** Per-layer quantization error is bounded (`Δ/2` per weight) and roughly zero-mean. Accumulated error is what happens when those perturbations propagate: each layer's output error becomes the *next* layer's input error, and through attention and the residual stream the contributions combine — sometimes averaging out (if independent), sometimes compounding (if correlated with the signal). This is why per-layer MSE can look excellent while end-to-end quality is poor, and it is why quantization error is **not** additively decomposable across layers. Practical consequence: never validate with a per-layer reconstruction metric; validate end-to-end with KL and a task gate.
- **Why asked:** It is the reason layer-wise methods work *well enough* and the reason they are not exact — a nuanced answer that only comes from having debugged a real regression.
- **Trap:** Summing per-layer MSEs and calling it the model's error.

---

**Q87. What is an "effective bit-width", and how do you compute it for a GPTQ `group_size=128` checkpoint?**

- **Answer:** Effective bit-width is the total bits stored per parameter, including all metadata. For `group_size=128` with fp16 scales and (symmetric) no zero-points: `4 + 16/128 = 4 + 0.125 = 4.125` bits/param. With a zero-point as well: `4 + 32/128 = 4.25`. Scale to 32: `4 + 32/32 = 5.0`. This matters because a "4-bit" checkpoint is never 4.0 bits, and the difference — 4.0 vs 5.0 — is a 20% larger file and a real kernel slowdown, which is exactly why `group_size=32` is reserved for 3-bit and for small models.
- **Why asked:** It is the arithmetic that decides whether a deployment fits, and it is where the granularity discussion becomes quantitative.
- **Trap:** Forgetting the scales entirely, or using `2 bytes` when the implementation stores them as fp32 (4 bytes). Check the config, not the label.

---

**Q88. What problem does rotation (QuaRot, SpinQuant, Hadamard) solve that AWQ and GPTQ do not?**

- **Answer:** AWQ and GPTQ both *work around* outliers: AWQ protects the salient channels by rescaling, GPTQ compensates for the error locally. Rotation **removes the outlier structure itself**. Multiplying weights and activations by an orthogonal `Q` (`Y = (WQᵀ)(QX)`) leaves the exact output unchanged, but spreads a single dominant coordinate's energy across all `n` coordinates — for a Hadamard matrix, `‖Hx‖∞ ≤ ‖x‖₂` and a magnitude-100 outlier at `n = 4096` can drop to ~2. Crucially it is **calibration-free** and it composes with GPTQ/AWQ, which is why it is the strongest 2025–26 result at 4-bit and below: it makes per-tensor INT4 *activation* quantization viable, which is the thing AWQ and GPTQ never even attempt.
- **Why asked:** It is the frontier question and it tests whether the candidate reads beyond the two 2023 papers.
- **Trap:** "Rotation loses information." It is exactly orthogonal — in exact arithmetic it is a no-op on the output. The only cost is that it must be fused into adjacent ops (and into the embedding/LayerNorm boundaries) or it adds a matmul.

---

**Q89. Is the quantization of a model reproducible? What would you pin?**

- **Answer:** Not by default. The sources of nondeterminism are: CUDA atomics in the Hessian accumulation and Cholesky (GPTQ), differences in kernel selection across GPU architectures (`sm_80` vs `sm_89` vs `sm_90`), library and kernel versions (`auto-gptq`, `autoawq`, `bitsandbytes`, `llama.cpp` build), and any nondeterministic calibration ordering. Pin all of them: the fp16 checkpoint hash, the quantization config, the library versions, the CUDA/driver version, the GPU architecture, and a seed. Then **hash the produced weights** and treat that hash as the artifact identity. In practice two runs on the same machine usually match and two runs on different architectures often do not.
- **Why asked:** It is a production-maturity question, and most candidates have never thought about it.
- **Trap:** Assuming `torch.manual_seed` is sufficient. It is not — the nondeterminism is in CUDA kernels and library versions, not in the RNG.

---

**Q90. You can have 8-bit weights with no quality loss, or 4-bit weights with a 0.5% task drop, on the same GPU. Which do you ship, and how do you decide?**

- **Answer:** It depends on what the 4× memory buys, expressed in the deployment's own currency. The memory saving translates into **concurrency** (more KV-cache room → more simultaneous sequences), which is the actual throughput and cost lever. So the question is: does fitting 2× or 4× more sequences per GPU cut the fleet size, and does that saving exceed the cost of a 0.5% task drop? Frame it as a business decision with numbers (see Q101 for the arithmetic pattern), not as a quality preference. The one non-negotiable: **measure the 0.5% on the task metric, not on perplexity**, and confirm it is uniformly distributed rather than concentrated in one user-visible capability (JSON compliance, tool calls, a language).
- **Why asked:** It is the staff-level framing question in disguise, asked at senior level.
- **Trap:** Answering "4-bit, it is nearly free". At 4-bit, "nearly free" is a claim about the *mean*; the failure mode is concentrated in the tail, and that is what costs you the customer.

---

**Q91. What are the hard limits of weight-only 4-bit quantization?**

- **Answer:** (1) **It does not help compute-bound workloads.** Prefill at large batch is compute-bound; 4-bit weights dequantize into FP16 and run on FP16 tensor cores, so prefill throughput barely moves. (2) **It does not shrink the KV cache**, which dominates at long context. (3) **It cannot go below ~3 bits without structural help** (rotation/QAT). (4) **It is one-way** — the quantized checkpoint cannot be further trained. (5) **It does not reduce activation memory**, which is what OOMs you during prefill with a long prompt. (6) **It adds a kernel dependency** that can cost you more throughput than it saves if the wrong kernel is selected.
- **Why asked:** It is the counterweight to the enthusiasm the rest of the interview rewards, and senior candidates are expected to volunteer limits.
- **Trap:** Listing only "some quality loss". The structural limits (1)-(5) matter more than the accuracy delta in most real deployments.

---

**Q92. A colleague says "we quantized to 4-bit so the model is 4× cheaper". Rewrite that claim correctly.**

- **Answer:** "Weights shrank 4×, which reduces memory bandwidth per token and VRAM per model. Decode speedup is ~2–3× at batch 1 (the kernel still dequantizes into FP16 and computes on FP16 tensor cores), and the real saving is *concurrency*: 4× less weight memory means roughly 4× more KV-cache room per GPU, so we can hold ~4× as many concurrent sequences and get close to a 3–4× cost reduction per million tokens — provided the workload is decode-bound and the KV cache, not the weights, is what was limiting our batch size. And the KV cache is unchanged per token, so at long context the gain shrinks."
- **Why asked:** It is the single most common overclaim in the field, and correcting it precisely is a strong senior signal.
- **Trap:** Correcting it to "it is only 2× cheaper". The concurrency effect is real and is usually the dominant one — the mistake is attributing the win to kernel speed rather than to batching.

---

## Level 4 — Staff / System Design (10 questions)

**Q93. Design the compression and serving strategy for a 70B model that must serve 50 requests/s at 8k context, on-prem, with a hard requirement of no more than a 1% drop on your internal task metric. Give the hardware count and the reasoning.** `[staff / system design]`

- **Answer:**
  - **Bit-width:** 4-bit, W4A16. 70B at 4-bit ≈ 35 GB + 6% metadata ≈ 37 GB; at fp16 it is 140 GB, which makes the whole design uneconomic. 3-bit (≈27 GB) is worth *testing* but only if the 1% gate passes, which it usually will not.
  - **Method:** AWQ first (fastest to produce, best 4-bit quality at this scale); fall back to GPTQ `g=128` if the AWQ build fails on the architecture. Validate both against the gate; they differ by tenths of a point, so pick on kernel availability.
  - **KV cache:** Llama-3.1-70B has 80 layers × 8 KV heads × 128 head_dim × 2 bytes = **320 KB/token**. At 8k tokens that is **2.6 GB per sequence**. Twenty concurrent sequences = 52 GB of KV — larger than the weights. **FP8 KV quantization is mandatory**, halving that to 26 GB and roughly doubling the achievable batch.
  - **Hardware:** 4×H100-80GB (or 4×A100-80GB). Per GPU: 37/4 ≈ 9.3 GB of weights + 26/4 ≈ 6.5 GB of KV (20 seqs) + ~2 GB activations + 1 GB context ≈ 19 GB — comfortable. Throughput: 70B at 4-bit on an H100 is in the low hundreds of tokens/s per GPU with batching; 50 rps × ~500 output tokens = 25k tokens/s, needing ~3–4 GPUs with continuous batching, so 4 GPUs with headroom. That is ~$12–16/h on-demand, or ~$0.15–0.20 per million output tokens.
  - **Serving stack:** vLLM (paged attention + continuous batching) with the AWQ kernel, or TensorRT-LLM if the team can own a compiled-engine pipeline. FP8 KV on both.
  - **Gate:** mean KL ≤ 0.1, p95 KL ≤ 1.0, top-1 agreement ≥ 95% *on 8k-context prompts in your domain*, plus the internal task metric within 1%.
- **Why asked:** It is the full stack in one question: bits, method, KV arithmetic, GPU count, serving stack, and a validation protocol. Staff candidates are expected to volunteer the KV cache before being asked.
- **Trap:** Sizing on weights alone. At 8k context with a real batch, the KV cache is as large as the weights and it is the term that decides the GPU count. The second-most-common trap is quoting 4× and forgetting that you also need the fp16 model resident somewhere to produce the quantized artifact (≈280 GB of host RAM).

---

**Q94. You have a 3B model that must run on a 4 GB device with a 2048-token context, offline. Choose the format, the bit-width, and the runtime. Justify every choice.** `[startup/on-prem]`

- **Answer:**
  - **Format: GGUF.** Offline, no CUDA, no container control, and the runtime must be embeddable. GGUF's memory-mapping means the 4 GB device does not have to hold the whole file plus a copy.
  - **Bit-width: `Q4_K_M` at first, `Q3_K_M` + imatrix if it does not fit.** A 3B at `Q4_K_M` is ~1.9 GB, leaving ~1.5 GB for the KV cache and the OS. KV for a 3B GQA model (e.g. 28 layers, 8 KV heads, 128 head_dim) is 112 KB/token → 2048 tokens = 0.23 GB per sequence, so batch 4 fits in ~0.9 GB. Tight but workable.
  - **Runtime: llama.cpp** (`llama-cli` embedded, or `llama-cpp-python` if it is a Python service), with `-ngl 0` on a pure-CPU device or `-ngl 99` if there is a small GPU.
  - **Quality safeguard:** build an `imatrix` from 200 in-domain samples and quantize `IQ4_XS` or `Q4_K_M` with it — 10 minutes of CPU, and at 3B every accuracy point counts.
  - **Validation:** a 40-prompt acceptance set, run on the device itself (device throughput is part of the acceptance criteria), plus an explicit check that the **chat template is embedded** in the GGUF and is being used by the runtime.
- **Why asked:** It is the small-model, small-device corner where every default is wrong: 7B recipes do not transfer, the KV cache is not negligible relative to 4 GB, and the chat template is a live risk.
- **Trap:** Recommending `Q4_K_M` with the standard 7B `group_size=128` reasoning and stopping there. At 3B you should expect a real quality delta and should plan for `Q3_K_M`+imatrix or `Q4_K_S` as the fallback, and you should validate on-device.

---

**Q95. Half your fleet is A100 (`sm_80`), half is H100 (`sm_90`). Design the quantization and artifact strategy.** `[NVIDIA/TensorRT]`

- **Answer:**
  - **One algorithm, two artifact variants if necessary.** GPTQ/AWQ 4-bit works on both architectures, but the *fastest kernel* differs: H100 supports FP8 natively and has better Marlin/NVFP4 paths; A100 has no FP8 tensor core (FP8 is emulated) and relies on Marlin for 4-bit. So: 4-bit W4A16 (AWQ or GPTQ) as the common artifact, and FP8 variants (weights and KV) **only** for the H100 pool.
  - **Pin and hash.** Nondeterminism across architectures is real (§Q89): record `sm_80`/`sm_90` with the artifact, hash the produced weights, and run the eval gate on each architecture — not on one and assuming the other.
  - **Two eval runs, one gate.** The same checkpoint can produce different numerics on the two cards; a gate that passes on H100 does not certify A100.
  - **Version the kernel, not just the weights.** Marlin/ExLlamaV2/vLLM versions differ; a rebuild of the serving image can change numerics. Treat the serving image digest as part of the artifact identity.
  - **Routing:** if the accuracy gate is tight, route the sensitive workload to the H100 pool where you can afford FP8/W8A8 rather than INT4, and keep INT4 for the throughput-tolerant traffic.
- **Why asked:** It is the heterogeneous-fleet reality inside almost every large company, and it exposes whether a candidate understands that "the same checkpoint" is not the same computation on different silicon.
- **Trap:** Producing one artifact and declaring it validated. The correct answer includes per-architecture validation and an explicit statement that FP8 is a *no-op on A100 in compute terms*.

---

**Q96. Your team wants to save money by quantizing a fine-tuned 13B to 3-bit instead of 4-bit. Design the decision process, including the gate and the failure criteria.** `[staff / system design]`

- **Answer:**
  - **Compute the actual saving first.** 13B at 4-bit ≈ 6.9 GB; at 3-bit ≈ 5.2 GB. That is a 25% weight reduction — and since you still need the same KV cache and activations, the *total* VRAM reduction is smaller, maybe 15%. Convert that into GPUs-per-fleet only if it crosses a bin-packing boundary (e.g. 2 GPUs → 1 GPU, or 4 → 3). If the fleet stays the same size, the saving is zero and the proposal is dead on arrival. **This is the first thing to say.**
  - **If it does cross a boundary:** produce the 3-bit checkpoint with `group_size=32`, `desc_act=True`, and 256 calibration samples of production-like data. Expect a real quality cost — published ranges put 3-bit at 0.2–0.6 perplexity and 1–3 points of MMLU, with reasoning and math degrading first.
  - **Gate:** mean KL ≤ 0.15, **p95 KL ≤ 1.5**, top-1 ≥ 94%, plus task metrics with an explicit per-capability budget (e.g. no more than 2% on JSON validity, no more than 3% on the reasoning suite) — because a mean-based gate will pass a model whose reasoning collapsed.
  - **Failure criteria, written down in advance:** if p95 KL exceeds the threshold, or any single capability drops more than its budget, revert to 4-bit. No "let's see if users notice".
  - **Fallback ladder:** 3-bit `g=32`+`desc_act` → rotation + 3-bit (which is where the real 3-bit wins live) → 4-bit `g=128` → 4-bit with FP8 KV (a different, often larger saving for free).
  - **And check the cheapest option first:** FP8 KV cache usually recovers more memory than 4→3-bit weights, at a much smaller quality cost, if the workload is long-context.
- **Why asked:** It is a proposal-evaluation question dressed as a quantization question. Staff-level answers start with "what does the saving actually buy?" rather than with the quantization recipe.
- **Trap:** Jumping to the recipe. Also: assuming the 4-bit → 3-bit saving is linear and that fleet size scales continuously with VRAM — it does not; it scales in whole GPUs.

---

**Q97. Design the CI/CD gate for quantized model artifacts.** `[staff / system design]`

- **Answer:**
  - **Artifact identity:** `(fp16 checkpoint hash, method, bits, group_size, desc_act, calibration corpus hash, library versions, CUDA/driver, GPU arch, output weights hash)`. All ten are part of the version string.
  - **Stage 1 — build:** quantize from a pinned checkpoint in a pinned container; assert the output hash matches either the recorded one or a freshly approved one (a changed hash without a changed config is a red flag).
  - **Stage 2 — mechanical:** assert the config on load (`quant_method`, `bits`, `group_size`, `q_max` convention), assert the chat template is present and renders identically to the fp16 tokenizer's `apply_chat_template`, assert the file size is within ±2% of expectation.
  - **Stage 3 — statistical:** the KL protocol on a **frozen 200-prompt suite** (mean KL, p95 KL, top-1 agreement) with hard thresholds; fail the build on breach.
  - **Stage 4 — task:** IFEval, JSON-schema validity, needle-in-a-haystack at max context, and a code/math smoke suite. Thresholds per capability, not just an average.
  - **Stage 5 — performance:** tokens/s at production batch size, VRAM high-water, p50/p99 latency. A quantization change that halves throughput is a regression even if quality holds.
  - **Stage 6 — shadow deploy:** serve the new artifact on 1% of traffic for 24 h; compare output *shape* metrics (JSON validity, average length, tool-call success) — these move before user complaints do.
  - **Rollback:** always keep the previous artifact and the fp16 checkpoint; rollback is a config change.
- **Why asked:** It is the "can you operationalize this" question, and the artifact-identity and frozen-suite parts are where candidates are thin.
- **Trap:** Gating on perplexity, or on a suite you regenerated this week. A frozen suite and a recorded artifact hash are what make the gate meaningful.

---

**Q98. You must serve both a 7B chat model and a 70B reasoning model from the same cluster, with a shared tokenizer and shared tooling. Design the format strategy.** `[staff / system design]`

- **Answer:**
  - **One method where possible, two where necessary — but one *tooling* path.** Use GPTQ or AWQ 4-bit for both (the same `auto-gptq`/`autoawq` + vLLM pipeline, one calibration harness, one eval harness), rather than picking GGUF for one and AWQ for the other. Shared tooling is worth more than a marginal quality win.
  - **The 7B stays on GPU** (it is cheap and latency-sensitive); the 70B is the memory-bound one, so it gets the FP8 KV cache and, if capacity is tighter than latency, could run a 3-bit variant for the lowest-priority traffic class.
  - **Two artifacts per model, one harness:** fp16 reference + quantized serving artifact, both registered in the same model registry with the same 10-field identity string.
  - **One eval suite, per-model thresholds.** A 7B will not meet the same KL thresholds as a 70B at 4-bit; thresholds are per-artifact, recorded in the registry, not global constants.
  - **One serving runtime** (vLLM) with per-model config, so the KV-cache accounting, batching and observability are identical.
  - **Chat template handling:** store the jinja template with the artifact and inject it at serve time, so a template regression is caught in one place.
- **Why asked:** It is the "one platform, two workloads" question, and the right answer optimizes for shared tooling rather than for the best individual choice.
- **Trap:** Choosing the theoretically best format per model and ending up with two pipelines, two eval harnesses and two failure modes.

---

**Q99. A product team asks for "the best possible quality at 8 bits". What do you tell them about the memory they are actually getting?** `[staff / system design]`

- **Answer:** 8-bit gives 2× memory reduction, not 4× or 8× — and at 8-bit you are almost certainly leaving quality on the table in exchange for very little. Three things to say: (1) **8-bit is effectively lossless** (Δppl < 0.01), so there is no quality argument for it over 5–6-bit, which is 30% smaller for a Δppl of ~0.02 — the honest recommendation is `Q5_K_M`/`Q6_K` or GPTQ-6. (2) **The KV cache does not shrink**, so at long context the memory saving is far less than 2×; compute the real number. (3) **If the goal is throughput, 8-bit W8A8 (INT8 tensor core) is a genuinely different proposition from 8-bit W8A16** — it saves *time*, not just memory, and it needs SmoothQuant-style handling of activation outliers. Ask which of the two they want, because the answers diverge.
- **Why asked:** It is a requirements-clarification question. The correct staff answer is to reframe the ask rather than to fulfil it literally.
- **Trap:** Implementing W8A16 because "they asked for 8 bits" when the requirement was actually throughput (which needs W8A8) or memory (which 6-bit serves better).

---

**Q100. Design a quantization evaluation harness that a non-ML engineer can run before every release.** `[staff / system design]`

- **Answer:** Package it as one command with a config file and a pass/fail report:
  ```
  quant-eval --fp16 <path> --quant <path> --suite eval/frozen_v3.jsonl \
             --thresholds eval/thresholds.yaml --out report.html
  ```
  The suite is a frozen JSONL of 200 prompts, each tagged with a capability (`chat`, `json`, `code`, `long_context`, `multilingual`). Thresholds are per capability. The report is a table: capability, mean KL, p95 KL, top-1 agreement, task metric, fp16 baseline, delta, PASS/FAIL — plus the five worst prompts with both outputs side by side. Wire it into CI with a non-zero exit on failure, and make the artifact identity string (the ten fields from Q97) the report's header. The critical design constraints: **the suite is versioned and frozen**, **the thresholds live in the repo, not in someone's head**, and **the report is readable by a non-specialist** (green/red per capability, with a one-line explanation of what each capability means if it fails).
- **Why asked:** It is the "make the right thing the easy thing" question. Staff engineers are judged on whether the organization can execute their design without them.
- **Trap:** Building a dashboard nobody runs. The design must be a *gate in CI*, not a report on a wiki.

---

**Q101. Build the business case: quantize to 4-bit and cut GPU spend, or keep fp16 and buy more GPUs. Show the arithmetic.** `[staff / system design]`

- **Answer:** Take a concrete workload — 7B, 1000 output tokens per request, 30 requests/s, 4k context, A100-80GB at $2.50/h.
  - **fp16:** weights 14 GB + KV 2.15 GB/sequence (no-GQA 7B at 4k) + ~1 GB overhead = ~17.2 GB per sequence. On 80 GB: 4 concurrent sequences. Serving 30 rps with ~4 s per request means ~120 in flight — so ~30 GPUs.
  - **int4 + FP8 KV:** weights 3.7 GB (4-bit, 0.25 bits/param metadata) + KV 1.07 GB + 1 GB = ~5.8 GB per sequence. On 80 GB: **13 concurrent sequences** — 3.3× the concurrency, so ~9–10 GPUs.
  - **Saving:** ~20 GPUs × $2.50/h × 730 h ≈ **$36k/month**, against a one-time quantization cost of ~1 GPU-hour per artifact and a validation afternoon.
  - **Cost of the quality risk:** quantify it. If the 4-bit model loses 0.5% of a task metric that maps to a 0.5% revenue impact, and the fleet costs $36k/month *less*, the trade is obvious — **provided** the loss is uniformly distributed and not concentrated in a capability whose failure costs more (e.g. malformed tool calls breaking an integration).
  - **Also state what is not saved:** prefill throughput at large batch is compute-bound and barely moves; so if the workload is prompt-heavy (RAG with long contexts), the saving is smaller than the concurrency calculation suggests, and the FP8 KV flag is doing most of the work.
- **Why asked:** It is the staff-level version of "does this actually help?", and the arithmetic is checkable.
- **Trap:** Quoting a 4× saving from the bit-width alone. The saving comes from concurrency, it is bounded by what the KV cache allows, and it is smaller for prompt-heavy workloads.

---

**Q102. Your company must ship a model to a customer's air-gapped environment with a documented, auditable process. How does quantization change your compliance story?** `[staff / system design]`

- **Answer:**
  - **A quantized model is a different model.** For a model card, a safety evaluation, or a certification, quantization is not an implementation detail — it changes the weights. Re-run safety evaluations and refusal-rate checks on the quantized artifact; quantization error can shift refusal behaviour at the margin.
  - **Provenance:** record the fp16 source checkpoint (hash + licence + revision), the exact quantization config, the calibration corpus (and its provenance — this is customer data in many cases, which may itself be a compliance question), the toolchain versions, and the operator.
  - **Reproducibility for audit:** an auditor will ask "show me that this artifact came from that checkpoint". You need the ten-field identity string and a re-runnable build, which means a pinned container, not a notebook.
  - **Data handling for calibration:** calibrating on customer data is *processing* that data. Confirm the contract allows it, or use a synthetic/representative corpus and document that choice.
  - **Deliverable format:** GGUF or a self-contained safetensors directory with the tokenizer and config included — avoid formats that require fetching anything at load time in an air-gapped network. And ship a **pinned runtime** (the llama.cpp build or the serving image digest), because GGUF compatibility is a function of the runtime version.
  - **Licence:** the quantized derivative inherits the base model's licence terms; check whether redistribution of a quantized form is permitted.
- **Why asked:** It is the question that separates engineers who have shipped to regulated customers from those who have not, and it is increasingly asked at staff level.
- **Trap:** Treating quantization as a build step with no compliance surface.

---

## Level 5 — Debugging (12 questions)

**Q103. Your 4-bit GPTQ model produces fluent, grammatical output that ignores the instruction. The fp16 model follows it. What do you check, in order?**

- **Answer:**
  - (1) **The prompt format.** If the serving path renders the prompt differently from the fp16 path — wrong template, missing special tokens, a `system` block dropped — the model receives a prompt it was not trained on. Print both rendered prompts as raw strings and diff them. This is the most common cause and it is not a quantization problem at all.
  - (2) **`generation_config.json` and EOS.** If the EOS id differs, the model can run to `max_new_tokens` and appear to "ignore" the end of its answer.
  - (3) **The tokenizer files in the quantized directory.** If `tokenizer_config.json` was not copied, `transformers` silently falls back to a default chat template.
  - (4) **The `q_max`/zero-point convention.** Load the config and compare `s`/`z` with the values recorded at quantization time.
  - (5) **A per-layer reconstruction check** — compare `‖WX − ŴX‖²` per layer against a reference to find a layer that is anomalously bad.
  - If all five pass, quantize at 8-bit and re-test. If 8-bit is also broken, it was never quantization.
- **Why asked:** It is the highest-frequency real bug in this field and the answer's *order* is what is being tested — you check the free explanations before the expensive one.
- **Trap:** Re-quantizing immediately, or jumping to "the model is too small for 4-bit". Three of the five causes are configuration.

---

**Q104. Your quantized model's perplexity is unchanged but your JSON-output validity rate fell from 99.2% to 91%. Diagnose and fix.**

- **Answer:**
  - **Diagnosis:** perplexity is a mean over ordinary prose tokens; the tokens that carry JSON structure (`{`, `}`, `"`, `:`, `,`) are a tiny fraction of that average and are individually rare in the pretraining distribution — so quantization error concentrated in the long tail shows up as a large task regression and a negligible perplexity change. Confirm by measuring **p95 KL** on JSON-generating prompts, not mean KL: the tail will be visibly worse. Also check **top-1 agreement restricted to the structural tokens**, which is a much sharper diagnostic than a global mean.
  - **Fix, cheapest first:** (1) constrain decoding — a grammar/JSON-schema-constrained sampler eliminates the failure at inference for zero quality cost; (2) exclude the `lm_head` (and embeddings) from quantization — structural tokens' logits come straight from the head and 8-bit or fp16 there is cheap; (3) `group_size` 128 → 32; (4) +1 bit (5-bit or 6-bit); (5) re-quantize with AWQ and domain-matched calibration.
  - **Prevention:** add a JSON-validity gate to CI (§Q97).
- **Why asked:** It is the canonical "perplexity lies" scenario, and it is asked because almost everyone has been burned by it.
- **Trap:** Going straight to 8-bit. Constrained decoding and excluding the head are cheaper and usually sufficient.

---

**Q105. A quantized model works on your A100 dev box and fails on the customer's H100. Same artifact, same container. What is happening?**

- **Answer:** The kernels differ. Likely causes, in order: (1) **kernel selection** — the serving stack picked a different quantization kernel (Marlin vs ExLlamaV2 vs a Triton fallback) on the new architecture, and one of them has a bug or a feature gap (e.g. `desc_act=True` support); (2) **FP8 emulation** — if the artifact or config requests FP8 and the path differs, numerics differ; (3) **library/driver mismatch** — a different `torch` or `vllm` build in the customer's image; (4) **`sm` capability differences** in the kernel's fast paths. Diagnose by printing the selected kernel name, the library versions and the CUDA arch on both machines, then pinning them. Validate on the target architecture before declaring success — "same artifact" is not "same computation" (§Q95).
- **Why asked:** It tests reproducibility discipline, which is a real production skill and a common blind spot.
- **Trap:** Assuming the artifact is deterministic across hardware. It is not.

---

**Q106. Quantization of a 70B failed halfway with `CUDA out of memory`, and after reducing batch the model quantized but the quality is terrible on one layer. What went wrong?**

- **Answer:** Two separate failures, and the second is the interesting one. (1) The OOM is the fp16 model plus the per-layer Hessian buffers not fitting; `device_map="auto"` was spreading layers across devices, and a layer whose calibration data was on a different device produced a degenerate Hessian. (2) The "terrible on one layer" outcome is the signature of an **ill-conditioned or singular `H`** — when a calibration channel is constant (a padding token repeating, a zero-variance feature) or when there are fewer calibration tokens than the layer's input dimension, `H` is singular and the Cholesky factorization either fails or produces a garbage `H⁻¹`, so the error-compensation updates explode on that layer. Fix: raise `damp_percent` (0.01 → 0.05 or 0.1), increase the number of calibration tokens so `H` is over-determined, and quantize with the model and calibration data on the same device.
- **Why asked:** It is a real GPTQ failure mode that candidates who have only run tutorials have never seen, and `damp_percent` is the knob they do not know exists.
- **Trap:** Blaming quantization in general. One bad layer is a numerical-stability bug, not a bit-width problem.

---

**Q107. Ollama loads your `Q4_K_M` GGUF and emits `<unk>` tokens. Diagnose.**

- **Answer:** A **tokenizer mismatch**. The GGUF was converted from a different HF revision (or a different repo) than the tokenizer it shipped with, so `tokenizer.ggml.tokens` does not correspond to the model's expected vocabulary. Diagnose by dumping the GGUF metadata and comparing the vocab size and a few token strings against the original HF `tokenizer.json`/`tokenizer_config.json`, and by checking `general.basename`/`general.name` against the source repo. Fix: re-convert from the exact pinned revision, and record the revision in the artifact identity. Related failure with a different symptom: a **`rope_theta`/context-length mismatch** in the metadata, which produces plausible-but-degraded long-context output rather than `<unk>`.
- **Why asked:** It is a concrete, checkable GGUF failure that requires knowing the container's contents.
- **Trap:** Assuming the quantization is at fault and re-quantizing at a higher bit-width — which reproduces the identical `<unk>` output and wastes an hour.

---

**Q108. After quantizing, generation never stops and always hits `max_new_tokens`. The fp16 model stops normally. What is wrong?**

- **Answer:** The EOS token. Either the quantized directory is missing `generation_config.json` (so `eos_token_id` falls back to the tokenizer default, which can be `None` with a chat template in play), or the GGUF metadata's EOS id differs from the one the model was trained with, or the chat template is not being applied so the model never reaches the assistant-turn terminator it learned. Check `generation_config.json` and `tokenizer_config.json` in both directories, diff the EOS/PAD ids, and print the rendered prompt. Fix by copying the generation config and tokenizer into the quantized artifact.
- **Why asked:** It is a small, unambiguous bug with a clean diagnostic — a good filter for whether a candidate debugs systematically or guesses.
- **Trap:** Increasing `max_new_tokens` or blaming sampling parameters.

---

**Q109. Your quantized model is *slower* than fp16 at batch 1, though it is smaller. Explain and fix.**

- **Answer:** At batch 1, decode is bandwidth-bound *if the kernel is efficient* — but the dequantization path costs compute per weight, and if the runtime selected a slow fallback kernel (a Triton or CUDA reference implementation instead of Marlin/ExLlamaV2), the dequantization overhead exceeds the bandwidth saving. Other causes: (1) **AWQ `version="GEMV"` vs `"GEMM"`** — GEMV is the batch-1 path and GEMM the batched path; using the wrong one costs up to 2×; (2) **`desc_act=True`** without kernel support, forcing a slow path; (3) **CPU offload** sneaking in via `device_map="auto"` so some layers are on the CPU; (4) **CPU-only dequant** with no SIMD support for the block format. Fix: install/select the fast kernel, match the AWQ version flag to the batch shape, ensure the whole model is on the GPU, and benchmark with the real batch size — a 4-bit model can *lose* at batch 1 and *win* at batch 16.
- **Why asked:** It is counter-intuitive and it is the reason "4-bit is faster" is a claim you must measure.
- **Trap:** Concluding quantization is broken. The weights are fine; the kernel path is wrong.

---

**Q110. You quantize a fine-tuned model to 4-bit and it loses 6 points on your task metric, while the *base* model at 4-bit loses only 0.5. What is different about the fine-tuned model?**

- **Answer:** Fine-tuning sharpens the weight distribution and narrows the model's behaviour: an instruction-tuned or RLHF'd model has *low-entropy*, peaked outputs, and the tokens that carry its behaviour are exactly the ones with small margins between the top-1 and top-2 logits. Quantization error of a fixed magnitude therefore flips more decisions — the model's decision boundaries are closer together. Additionally, fine-tuning on a narrow domain *reduces the redundancy* that absorbs quantization error, moving probability mass into a small number of directions that a coarse grid handles badly. Diagnose by measuring KL on *domain* prompts for both the base and the fine-tuned model: the fine-tuned one will show a much worse tail. Fix: 8-bit or 6-bit for the fine-tuned model (the fine-tuning is what you paid for — do not quantize it away), or `group_size=64`/`32`, or QLoRA instead of post-hoc quantization.
- **Why asked:** It is a subtle, empirical, and very common finding, and it explains why "the paper says 4-bit is lossless" does not transfer to your fine-tune.
- **Trap:** Assuming the quantization is buggy. It is behaving correctly; the fine-tuned model is simply more sensitive — and the correct response is a different bit budget for a different artifact.

---

**Q111. During QAT, the loss falls normally for 500 steps and then the quantized model's accuracy after `convert` is far worse than the simulated accuracy during training. What happened?**

- **Answer:** The classic QAT calibration mismatch. Causes: (1) **`convert` was called without `model.eval()`**, so the observers' ranges were still updating and the frozen scales are wrong; (2) **the calibration data distribution at convert time differs from training** — e.g. the observers were fed augmented or shuffled data and froze on a shifted range; (3) **a module was not fused** (Conv+BN+ReLU) so the BN statistics drift between the fake-quant and integer paths; (4) **the qconfig at prepare differs from the qconfig at convert**; (5) **skipping `model.eval()` at inference** so BN keeps updating on the quantized path. Fix, in order: call `convert(model.eval())` on the *same* qconfig, fuse before `prepare_qat`, and re-run the training-time calibration pass in eval mode before converting.
- **Why asked:** QAT conversion bugs are the hardest to see because the model trains fine and only breaks at the end.
- **Trap:** Re-training longer. The problem is in the conversion, not the optimization.

---

**Q112. A model quantized with GPTQ has a *lower* perplexity on your eval set than the fp16 model. What do you conclude?**

- **Answer:** Treat it as a bug, not a win. Quantization cannot add information, so a genuine improvement is impossible. Ranked hypotheses: (1) **the tokenizer, template, or `rope_theta` changed** between the two evaluations, so you are not comparing the same computation; (2) **the eval text overlaps the calibration corpus** — the quantized model was tuned toward it via the Hessian; (3) **a different evaluation path** (e.g. the quantized model is evaluated with a sliding window and the fp16 one is not); (4) **the quantized model is more peaked**, which reduces perplexity while degrading task performance — a real and documented effect that makes perplexity an actively misleading metric; (5) a `float32`/`bfloat16` difference in the loss computation. Fix: diff the tokenizer and configs, hold out calibration data strictly, and gate on a task metric.
- **Why asked:** It is the trap-of-traps in this module: an apparently good number that means something is wrong.
- **Trap:** Shipping it. Any "quantization improved the model" result is a measurement bug until proven otherwise.

---

**Q113. A 4-bit model that was perfect on your 4k-context tests degrades at 16k. Quantization was validated at 4k. What is happening?**

- **Answer:** Two distinct effects. (1) **The KV cache, not the weights.** At 16k the cache is 4× larger; if the serving stack silently truncates context or spills, or if FP8 KV quantization was enabled for memory and never validated at 16k, that is where the damage is — and FP8 KV error compounds with sequence position. (2) **Attention-score quantization error compounds over distance.** Long-range retrieval depends on small differences between attention weights that a coarse grid flattens; the model can look fine on local coherence while losing needle-in-a-haystack retrieval. Diagnose with a needle-in-a-haystack test at your maximum context, and with KL measured *bucketed by position* — the tail (positions 12k–16k) will show more damage than the head. Fix: validate at the real context length, disable FP8 KV if it was the cause, or use 8-bit KV for the sensitive layers.
- **Why asked:** It is the failure that appears only in production, because validation is always done at a shorter context than traffic.
- **Trap:** Blaming the weight quantization. The weights are context-independent; the KV cache and the attention path are not.

---

**Q114. You are paged at 2 a.m.: the quantized model is emitting repeated tokens. Yesterday it was fine; nothing was deployed except a serving-image rebuild. What do you do?**

- **Answer:**
  - (1) **Roll back first, diagnose second.** Revert the serving image to the previous digest and confirm the symptom disappears. This is a one-command fix and it establishes causality immediately.
  - (2) **Then diff the image:** the kernel library versions (`vllm`, `exllamav2`, `marlin`, `bitsandbytes`), the CUDA/driver version, and the selected kernel. A repeated-token loop is a classic symptom of a **wrong kernel path or a degenerate dequantization** — the model is producing a plausible-but-degenerate distribution, which is what a numerically broken weight layout looks like.
  - (3) **Check whether the artifact is actually being loaded as expected** — a rebuilt image may have changed the `quant_method` dispatch and be running the checkpoint through the wrong dequantizer.
  - (4) **Verify the previous artifact hash still reproduces** — if the same artifact on the previous image is fine, the weights are exonerated.
  - (5) **Add a smoke test to the image build** — a 20-prompt greedy generation with a repeated-token detector — so this class of regression cannot reach production again.
- **Why asked:** It is a full incident-response question that happens to be about quantization. The grading is on the *order*: roll back, then isolate the variable, then prevent.
- **Trap:** Starting with the artifact and re-quantizing at 2 a.m. The only variable that changed is the image; changing a second variable destroys your ability to learn what broke.

---

## Rapid-Fire True / False (20)

Answer out loud in under five seconds each. The *reason* is the real answer.

| # | Statement | T/F | Why |
|---|---|---|---|
| 1 | Quantization can improve a model's accuracy. | **F** | It can only reduce precision; any apparent improvement is a measurement artifact (Q112). |
| 2 | 8-bit quantization is lossless in practice. | **T** | Δperplexity < 0.01 for LLMs; not theoretically lossless, but effectively so. |
| 3 | A 4-bit model runs 4× faster than fp16. | **F** | It reads 4× fewer bytes; compute is still FP16 tensor cores. ~2–3× at batch 1 (Q92). |
| 4 | Symmetric quantization always has a zero-point of 0. | **T** | That is the definition. |
| 5 | Asymmetric quantization always beats symmetric on MSE. | **F** | Rounding `z` introduces a bias; on zero-mean tensors symmetric usually wins (Q37). |
| 6 | The largest-magnitude weight in a tensor is quantized exactly. | **T** | It defines the scale, so it lands on `±q_max` (Q69). |
| 7 | GPTQ stands for Gradient Post-Training Quantization. | **F** | Generative Pre-trained Transformer Quantization (Q79). |
| 8 | AWQ removes the least important weights. | **F** | It rescales salient channels; nothing is pruned (§5.4 of CS-10). |
| 9 | GPTQ and AWQ both need calibration data. | **T** | GPTQ for `H`, AWQ for activation magnitudes. |
| 10 | bitsandbytes produces a portable 4-bit checkpoint. | **F** | It quantizes at load time; there is no artifact (Q47). |
| 11 | GGUF is a quantization method. | **F** | It is a container format (Q82). |
| 12 | `Q4_K_M` means 4-bit, medium quality. | **F** | `_M` is the size-mix — how much of the model keeps higher precision (Q18). |
| 13 | The `imatrix` improves low-bit GGUF quality. | **T** | 5–15% better perplexity at 2–3 bits for ~10 min of CPU (Q35). |
| 14 | Per-channel quantization uses one scale per output channel. | **T** | One per row of the weight matrix. |
| 15 | `group_size=128` means a 4096×4096 layer stores 32 scales. | **F** | It stores `4096 × (4096/128) = 131,072` — one per block per row. |
| 16 | The KV cache shrinks when you quantize the weights. | **F** | It is entirely unaffected. Quantize it separately (Q11, Q51). |
| 17 | QAT is required to get good 4-bit models. | **F** | PTQ reaches near-lossless 4-bit for ≥7B; QAT is for 2-bit and for model owners. |
| 18 | QLoRA trains the quantized weights. | **F** | It trains adapters over a frozen NF4 base (Q21). |
| 19 | The straight-through estimator makes the gradient of `round` equal 1. | **T** | On the unclipped interval, by definition (Q43, Q72). |
| 20 | A quantized model is a different model for compliance purposes. | **T** | The weights changed; re-run safety evaluations and re-version the model card (Q102). |

---

## Coding / Whiteboard Tasks (8)

**Task 1 — Implement affine quantization and dequantization from scratch.** (15 min)
Write `quantize_tensor(t, num_bits)` and `dequantize_tensor(q, scale, zp)` for signed int8, handling the `min == max` case. Verify with the eight weights from Q36: assert the max absolute error is ≤ `s/2`.
*What they are watching for:* the `1e-8` guard, the `clamp`, the `.int8` cast, and whether you compute `scale` from `max − min` (asymmetric) or `max|x|` (symmetric) — and whether you *say which* you chose.

**Task 2 — Compute the error bound table.** (10 min)
For `b ∈ {2, 3, 4, 8}` on a tensor spanning `[−1, 1]`, produce a table of `Δ`, `Δ/2`, and `Δ²/12`. Then state the MSE ratio between 4 and 8 bits and explain it in one sentence.
*What they are watching for:* whether you use `2^b − 1` or `2^b` in the denominator, and whether you can explain *why* the ratio is quadratic.

**Task 3 — Write the straight-through estimator.** (10 min)
Implement `RoundSTE` as a `torch.autograd.Function` and demonstrate with a one-layer network that gradients flow through it. Then explain what happens if you use plain `torch.round` instead.
*What they are watching for:* `ctx` as the first argument, the clamp, `grad_output` returned unchanged, and the answer to "what happens otherwise" being "the gradient is zero and nothing upstream trains".

**Task 4 — Write a QAT pipeline for a small model.** (25 min)
Take a 2-layer MLP, train it to convergence in fp32, then: fuse, set `get_default_qat_qconfig('fbgemm')`, `prepare_qat`, train 10% more epochs, `convert(model.eval())`. Report accuracy for fp32, PTQ-dynamic, and QAT.
*What they are watching for:* fusing *before* `prepare_qat`, `.eval()` before `convert`, and whether they explain that `fbgemm` is x86-specific.

**Task 5 — Compute a full VRAM budget.** (15 min)
For a 7B model with 32 layers, 8 KV heads, `head_dim=128`, at 4-bit weights, 8k context, batch 6: give weights, KV cache, activations, CUDA context, and the total. Then say whether it fits a 24 GB card.
*What they are watching for:* the KV formula written out (`2·L·h_kv·d_head·seq·batch·bytes`), and whether they include the metadata overhead (`×1.06`).

**Task 6 — Write the KL-divergence evaluation.** (20 min)
Load an fp16 model and its 4-bit counterpart, run 32 held-out documents through both, and report mean KL, p95 KL, and top-1 agreement. Handle the memory correctly.
*What they are watching for:* fp32 casting of both logits, chunking over the sequence (a `[seq, 128k]` log-softmax is a memory bomb), the correct `F.kl_div` argument order, and reporting the **p95**.

**Task 7 — Write the GPTQ quantization script.** (20 min)
Take `tiiuae/falcon-rw-1b`, load it in fp16, build a `BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)`, quantize with 128 calibration samples from a dataset, save with safetensors, then load it back and generate with greedy decoding.
*What they are watching for:* calibration data on the model's device, `use_safetensors=True`, `trust_remote_code=True`, and whether they verify the reload (the step most candidates skip).

**Task 8 — Write the GGUF pipeline.** (15 min, may be described rather than executed)
Convert a small HF model to GGUF f16, build an `imatrix` from a calibration file, quantize to `Q4_K_M`, and run it. Give the exact commands and say what you would check first if the output looked wrong.
*What they are watching for:* `convert_hf_to_gguf.py` (not `convert.py`), `llama-quantize`, `llama-cli` (not `./main`), and the answer to the last part being **the chat template**.

---

## Numbers To Memorize (30)

| # | Number | What it is |
|---|---|---|
| 1 | `q = round(x/s) + z`, `x̂ = s(q − z)` | The affine quantizer |
| 2 | `Δ = (max − min)/(2^b − 1)` | The step size |
| 3 | `max error = Δ/2`, `MSE = Δ²/12` | Uniform quantization error |
| 4 | `MSE ∝ 1/(2^b)²` | Each bit divides MSE by ~4 (−6.02 dB) |
| 5 | 4→8 bits: Δ ÷ 16, MSE ÷ 256 | The quadratic-in-bits relationship |
| 6 | `log2(60) ≈ 5.9` bits | The cost of a 60× outlier |
| 7 | `[−127, 127]` | The symmetric int8 convention (−128 deliberately unused) |
| 8 | `[−128, 127]` | Two's-complement int8 |
| 9 | `[0, 255]` | Asymmetric uint8 |
| 10 | 8 weights → `s = 0.0097205`, max error `0.004004` | The worked example |
| 11 | `0.25` bits/param | Group metadata at `group_size=128` |
| 12 | `0.373` bits/param | What double quantization recovers (NF4/QLoRA) |
| 13 | `s = max|X_j|^α / max|W_j|^(1−α)`, `α ≈ 0.5` | SmoothQuant and AWQ's scale |
| 14 | `6.0` | LLM.int8()'s outlier threshold |
| 15 | `0.1%` | The fraction of dimensions LLM.int8() extracts to FP16 |
| 16 | `~1%` | The fraction of channels AWQ treats as salient |
| 17 | `H = 2XXᵀ` | GPTQ's Hessian (the input covariance) |
| 18 | `damp_percent = 0.01` | GPTQ's default diagonal damping |
| 19 | `group_size = 128` | GPTQ/AWQ/bnb default; 32 at 3-bit |
| 20 | `blocksize = 64` | NF4's block size in QLoRA |
| 21 | `0.97 → 0.97 → 0.98` | The course's dynamic-PTQ / QAT MLP accuracies |
| 22 | `400 / 200 / 100 / 50 MB` | 100M params at fp32 / fp16 / int8 / int4 |
| 23 | `14 / 7 / 3.5 / 1.75 GB` | 7B at fp16 / int8 / int4 / 2-bit |
| 24 | `140 / 70 / 35 / 17.5 GB` | 70B at fp16 / int8 / int4 / 2-bit |
| 25 | `128 KB/token` | KV for a GQA 7–8B (32L, 8 KV heads, d_head 128) |
| 26 | `512 KB/token` | KV for a no-GQA 7B (32L, 32 KV heads, d_head 128) |
| 27 | `0.6–1.2 GB` | CUDA context + framework overhead |
| 28 | `Δppl 0.05 / 0.2–0.6 / 1–5+` | At 4-bit / 3-bit / 2-bit |
| 29 | `50–200` and `128–512` | Calibration samples: PTQ and GPTQ/AWQ respectively |
| 30 | `~4.8 bits` | The *effective* width of `Q4_K_M` including k-quant metadata |

---

## Answers To CS-10's Self-Check Questions

Compact recall versions — full derivations are in CS-10 §19.

| # | CS-10 question | One-line answer |
|---|---|---|
| 1 | Write the affine equations and define each symbol | `q = round(x/s) + z` forward, `x̂ = s(q − z)` back; `s` = step size in the tensor's units, `z` = the integer code for real 0.0 |
| 2 | Derive `MSE = Δ²/12`; cost of a 40× outlier | Error is uniform in the cell → `E[e²] = Δ²/12`. A 40× outlier multiplies `Δ` by 40 → MSE ×1600, and costs `log2(40) ≈ 5.3` bits |
| 3 | Quantize the eight weights to symmetric INT8 | `s = 0.0097205`; codes `[2, −15, 81, −127, 53, −7, 34, −94]`; max error `0.004004` |
| 4 | Why does uint8 give a smaller step but a worse MSE? | Smaller step (finer grid) but `z` is rounded, so the whole reconstruction is offset — a systematic bias beats a smaller variance in MSE |
| 5 | Why is `z = 156` in signed int8 catastrophic but not an error? | Codes outside `[−128,127]` are clamped, so six of eight values saturate to one code; the tensor stays valid int8 and nothing raises |
| 6 | Which weight is always exact, and why? | The maximum-magnitude one — it defines `s = max|x|/q_max`, so it lands on `±q_max` |
| 7 | Per-tensor vs per-channel vs per-group; why 128? | Metadata 0 / ~0.001 / 0.25 bits per param; 128 aligns with GPU block sizes, its metadata cost is affordable, and the accuracy knee is there |
| 8 | State the STE, two reasons for it, one failure | Forward `round(clamp(x/s))`, backward `:= 1`. (a) The quantizer is approximately the identity in the region that matters; (b) its gradient is an unbiased estimator of the smoothed quantizer's. Fails at 2-bit, where bins are coarse relative to the gradient |
| 9 | Why must training happen between `prepare_qat` and `convert`? | The observers must see data with the loss active to freeze correct ranges, and the weights must adapt to the simulated error |
| 10 | Massive activation: mechanism and mitigation | A consistently large residual channel survives RMSNorm and acts as a fixed bias, fed by attention-sink heads. Per-channel granularity cannot fix *activations*; use SmoothQuant (migrate), rotation (spread), or LLM.int8() (mixed precision) |
| 11 | GPTQ's objective, `H`, compensation, `act_order` | `min ‖WX − ŴX‖²`; `H = 2XXᵀ` (input covariance); the residual from each quantized column is subtracted from the remaining columns via `H⁻¹`; `act_order` quantizes columns in decreasing activation importance |
| 12 | What AWQ measures; the identity | Per-channel **activation** magnitude; `W·X = (W·diag(s))·(diag(s)⁻¹·X)`; free because activations are not quantized (W4A16) |
| 13 | Four GGML→GGUF differences; which is silent | Metadata KV replaces hard-coded hyperparameters; versioning; embedded tokenizer; mmap. The chat template is the silent one |
| 14 | Decode `Q4_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`; what is `imatrix`? | 4/5/6/8-bit; `_K` = hierarchical k-quant; `_S/_M` = size-mix. `imatrix` weights the quantization error by activation importance from calibration data |
| 15 | VRAM for a 7B at int4, 4k, batch 1 (no GQA) | Weights 3.7 GB + KV 2.15 GB + activations 0.1 + context 0.6 + slack ≈ **7.2 GB**. With GQA (8 KV heads) the KV drops to 0.54 GB → ~5.5 GB |
| 16 | Why not 4× faster? What *is* 4×? | The kernel dequantizes into FP16 and computes on FP16 tensor cores. What is 4× is bytes read and (therefore) KV-cache room per GPU → concurrency |
| 17 | Five outlier mitigations; which is calibration-free? | Mixed precision, granularity, migration (SmoothQuant), salience protection (AWQ), compensation (GPTQ). Calibration-free: mixed precision, granularity (for weights), and rotation |
| 18 | Perplexity holds but JSON breaks — next step and metric | Measure **p95 KL** on JSON-shaped prompts; fix by constrained decoding, excluding `lm_head`, or `group_size=32`. The metric that would have caught it is JSON-schema validity per release |

---

## What To Read Next

| If you want | Go to |
|---|---|
| The GPTQ Hessian arithmetic, the video's worked example reproduced, AWQ's correction, KV-cache quantization, the W8A8…W4A4 precision lattice, QLoRA/NF4 algebra, serving flags | **CS-11 — Quantization II: Advanced Methods & Production Practice**, and its question bank **IQ-11** |
| The formulas, the decision tree, the VRAM table, per-method copy-paste snippets | **CH-10 — Quantization Cheat Sheet** |
| The full derivation and the notebook reproductions | **CS-10 — Quantization I: Fundamentals** |
| The other compression axis (a smaller model rather than a smaller encoding) | **CS-08 / CS-09 — Knowledge Distillation**, and **IQ-08 / IQ-09** |
| The one case where you train on a quantized base | The QLoRA module, and CS-11 §4.11 |
| A runnable end-to-end script | `Finetuning-Handbook/code/08_quantize.py` |

