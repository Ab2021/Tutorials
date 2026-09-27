# IQ-11 — Interview Questions: Quantization II — GPTQ, AWQ, GGUF & the Production Lattice

| Field | Value |
|---|---|
| **Module** | IQ-11 — Quantization Part 2: advanced methods, formats, deployment |
| **Pairs with** | `case-studies/CS-11-Quantization-Part-2.md`, `cheat-sheets/CH-11-Quantization-II-GPTQ-AWQ-GGUF.md`, `code/08_quantize.py` |
| **Total questions** | 54 across Levels 1–5, plus 12 rapid-fire and 2 whiteboard tasks |
| **Levels covered** | L1 screen · L2 working engineer · L3 senior · L4 staff/system design · L5 debugging |
| **Source material** | CS-11 §4–§19 (every derivation and number), CH-11 §1–§12, and the verified `--help` surface of `code/08_quantize.py` |

**Ground truth used throughout:** the GPTQ objective `argmin_Ŵ ‖WX − ŴX‖²_F` with `H = XᵀX` and the compensation term from `H⁻¹`; the AWQ scale `s_j = a_j^α / w_j^(1−α)` and the rescaling identity `W·X = (W·diag(s))·(diag(s)⁻¹·X)`; the precision lattice W8A8 / W4A16 / W4A8 / W4A4; the GGUF k-quant naming scheme and the importance matrix; the gate `mean KL ≤ 0.10`, `p95 KL ≤ 1.00`, `top-1 agreement ≥ 0.95`, measured against **your own fp16 baseline**.

**Relationship to IQ-10 — read this first.** IQ-10 is the Part-1 bank: the affine quantiser, `Δ`, `MSE = Δ²/12`, symmetric vs asymmetric, per-tensor vs per-channel vs per-group, PTQ vs QAT, the outlier problem, LLM.int8(), NF4, QLoRA, the GGML→GGUF history, the bit-width curve, and "which precision does my model fit in". **IQ-11 repeats none of it.** This bank covers only Part-2 territory: the *algorithms* behind GPTQ and AWQ, the *artifacts* they produce, the *serving lattice*, the *gate*, and — the part that decides real projects — **what each format does to a fine-tuning workflow.** If a question could be answered from IQ-10, it was cut.

---

## How To Use This File

- **L1 — phone screen (60–90 s per answer).** Vocabulary and orientation: what GPTQ optimises, what W4A16 means, why GGUF is a container. Do not let a candidate bluff past "AWQ quantizes activations."
- **L2 — working engineer (3–5 min).** Has quantised something real and hit the tooling. Flags, defaults, calibration, the artifact chain, merge-then-quantise ordering. The `--help` surface is fair game here.
- **L3 — senior (5–10 min).** Internals: why the Hessian, why the compensation term, why activation-aware *scaling* beats keeping channels in fp16, why W4A4 is a different problem from W4A16, what a rotation does to an outlier.
- **L4 — staff / system design (10–15 min).** Budgets with two terms that move independently, artifacts per serving stack, manifests, the cost of a re-quantisation, the compliance surface of a calibration corpus.
- **L5 — debugging (5–10 min).** Symptoms with no error message. Name what you look at first, what each observation rules in or out, and the fix.
- **The meta-rule.** "It depends" is a failure unless it immediately says *what* it depends on and then picks a default. "4-bit or 8-bit? Depends on whether the workload is reasoning-heavy — for chat I'd ship W4A16 group 128 and gate it; for maths/code I'd start at FP8 W8A8" is a pass. "Depends on your use case" is a fail.
- **The three answers that get people hired in this module.** (1) *"The format is chosen by where the model runs, not by which benchmark table looks best — and the calibration set, not the algorithm, is the highest-leverage knob in the pipeline."* (2) *"AWQ is W4A16. Activation-aware names the statistic used to pick the scale, not the deployed precision."* (3) *"A quantized model usually cannot be fine-tuned. The three legal orderings are bf16→finetune→merge→quantize, QLoRA over a frozen NF4 base, and QAT."*

---

## Level 1 — Fundamentals & Vocabulary (screening)

**Q1. State GPTQ's objective in one line. What quantity is the error measured on?**
- **Answer:** `argmin_Ŵ ‖WX − ŴX‖²_F` — the error is measured on the layer's **output**, not on the weights. Weight-space error is the wrong proxy: a rounding error in a weight that multiplies a near-zero activation costs nothing, while the same error on a busy channel is expensive.
- **Why asked / trap:** Tests whether they know GPTQ is *layer-wise output matching* rather than round-to-nearest. Saying "it minimises the weight error" describes plain RTN — which is what bitsandbytes NF4 does.

**Q2. What is `H` in GPTQ, what shape, and how much memory for a 4096-wide layer?**
- **Answer:** `H = XᵀX`, the second moment of the calibration activations — one `d_in × d_in` matrix per layer, computed once. For `d_in = 4096` in fp32 that is `4096² × 4 B = 67 MB` per layer (CH-11 §2).
- **Why asked / trap:** `H` is the whole method and its memory is why GPTQ needs the fp16 model resident. The trap is calling it the Hessian *of the loss* — it is the input covariance; loss curvature is never computed.

**Q3. Write GPTQ's per-column sensitivity and say which term is squared.**
- **Answer:** `Δ_err ≈ (w_j − ŵ_j)² · H_jj`. The **weight error is squared**; `H_jj` enters linearly. The instructor's worked example is `0.5² × 56 = 14` (CS-11 App. A).
- **Why asked / trap:** The square is load-bearing — doubling the rounding error quadruples its cost, which is why column *order* matters. The trap is squaring `H_jj` too, double-counting channel activity.

**Q4. What does `damp_percent` do, and what breaks in each direction?**
- **Answer:** It adds `λI` to `H`'s diagonal before the Cholesky factorisation, so a rank-deficient or ill-conditioned `H` stays invertible. Too low → singular `H`, NaN, or one destroyed layer. Too high → the solution degrades toward plain RTN and you paid GPTQ's cost for RTN's quality.
- **Why asked / trap:** It is the one knob you actually touch when a single layer fails. The trap is reading it as a regulariser against overfitting; it is a numerical stabiliser for one layer's linear solve.

**Q5. What does `desc_act` change, what does it cost, and why does it break kernels?**
- **Answer:** It quantises columns in order of decreasing `H_jj` across the whole layer rather than within a group — better accuracy, for roughly **10% slower inference**, because the resulting permutation breaks the contiguous layout the kernel expects and forces a gather; some Marlin builds refuse the checkpoint outright.
- **Why asked / trap:** The classic "free accuracy that is not free." The trap is recommending `desc_act=True` unconditionally — at 4-bit the gain is often inside the noise; it pays at 3-bit (CS-11 §17.12).

**Q6. Write AWQ's scale and say what each term is measured over.**
- **Answer:** `s_j = a_j^α / w_j^(1−α)` with `α ≈ 0.5`, where `a_j = max|X_:,j|` is the **maximum absolute activation** on input channel `j` over the calibration set and `w_j = max|W_:,j|` the corresponding weight magnitude. Applied as `W·diag(s)` and `X·diag(s)⁻¹`.
- **Why asked / trap:** The formula is the method. The trap is using the *mean* activation where the formula needs the max — the aggregation is what makes published numbers reproducible.

**Q7. AWQ is called "activation-aware". What precision are the activations in the artifact you ship?**
- **Answer:** **fp16.** The artifact is W4A16. "Activation-aware" names the *statistic* the algorithm consults, and the activation-side factor `diag(s)⁻¹` is folded into the preceding LayerNorm or Linear where it is exact in fp16 and costs nothing (CH-11 §1, CS-11 §4.3, §17.4).
- **Why asked / trap:** The highest-frequency misconception in the subject and a fast read on whether the candidate has opened a real config. The trap: "AWQ quantizes activations so it beats GPTQ on accuracy" — quantised activations means W8A8/W4A8, a different family.

**Q8. Is GGUF a quantization method, or a container? What about `Q4_K_M`?**
- **Answer:** **Both, at different layers.** GGUF is a *container file format* — one portable file carrying tensors, metadata and tokenizer. `Q4_K_M` is a *method's output*: post-training, weight-only, block-wise, mixed-precision, produced by `llama-quantize` (CS-11 §17.9, CH-11 §9.1).
- **Why asked / trap:** Separates the file from the quantiser, which is what makes people ask whether they can "convert a GPTQ to GGUF". The trap is "GGUF is just a wrapper, the real quantisation is GPTQ" — there is no GPTQ anywhere in that path.

**Q9. Decode `Q4_K_M`, `Q3_K_S`, `Q6_K`, `Q8_0`.**
- **Answer:** `Q<bits>_K_<S|M|L>`: block-wise quant, `K` = k-quant (nested scales per super-block), suffix = how much of the model stays higher-precision, `_S` < `_M` < `_L`. `Q6_K` is 6-bit k-quant with no size suffix; `Q8_0` is legacy (non-K) 8-bit, essentially lossless.
- **Why asked / trap:** You cannot pick a GGUF without reading the name. The trap is assuming `Q4_K_M` is 4.00 bpw — it is about **4.85 bpw**, which is why a "4-bit" 70B GGUF is ~42 GB, not 35 GB.

**Q10. What is an importance matrix, when do you build one, and what does it change?**
- **Answer:** `llama-imatrix` computes per-tensor activation-importance statistics over a corpus you supply; `llama-quantize --imatrix` then weights the quantization error by how much each channel matters. Optional, but **the single biggest free quality lever in GGUF**, and effectively mandatory below 4-bit (CH-11 §4.3, §5.3).
- **Why asked / trap:** It is the GGUF equivalent of a calibration set and the step everyone skips. The trap is building it from WikiText and then blaming the model for getting worse on your domain — the imatrix is fit to a corpus, so make it yours.

**Q11. Name the four cells of the precision lattice and say which each method occupies.**
- **Answer:** W8A8 / W4A16 / W4A8 / W4A4 (weights × activations). GPTQ, AWQ and GGUF k-quants are all **W4A16**; bitsandbytes NF4 is W4A16 too, but a *training* format rather than a serving one. W8A8 is the FP8/INT8 family (`llm-compressor`, SmoothQuant); W4A8 and W4A4 are early production and need rotation-style methods (CS-11 §4.10, CH-11 §9.4).
- **Why asked / trap:** The lattice is the mental model that makes every "which method?" question answerable. The trap is putting AWQ in a different cell from GPTQ — they reach the same deployment point by different mathematics.

**Q12. Name the three gate metrics and their thresholds.**
- **Answer:** Against **your own fp16 baseline**, on held-out data disjoint from calibration: **mean KL ≤ 0.10**, **p95 KL ≤ 1.00**, **top-1 agreement ≥ 0.95**. Read `p95` and `max`, not just the mean — `mean_kl = 0.03` with `max_kl = 47` on 200 positions is a broken model with a good average (CH-11 §5.6).
- **Why asked / trap:** If a candidate cannot state a threshold *before* quantising, they are shipping on vibes (CH-11 §3 stop condition X1). The trap is substituting perplexity, which is not a gate.

**Q13. What does an FP8 KV cache buy, what hardware does it need, and which second flag must accompany it?**
- **Answer:** It halves KV-cache bytes — the term that dominates at long context and that weight quantization never touches — roughly doubling concurrency at a fixed VRAM budget. Requires **sm_89 or newer** (L4/L40S/4090/H100); there is no fast path on A100 (sm_80). In vLLM the flag is `--kv-cache-dtype fp8` and it must be accompanied by `--calculate-kv-scales`, or the scale stays at 1.0 and the format's range is wasted (CS-11 §19.9).
- **Why asked / trap:** The highest-leverage serving flag, and the hardware constraint is what teams get wrong. The trap is quantising K below FP8 without a needle test — K errors flip the attention argmax, a discontinuous failure.

**Q14. Where does the chat template live for a GGUF artifact, and what breaks if it is missing?**
- **Answer:** In the GGUF **metadata**. If it is missing or wrong the model degrades **silently** — fluent, well-formed, wrong. `code/08_quantize.py` prints this warning at the end of the GGUF branch and tells you to re-test with the same prompts you used pre-quantisation.
- **Why asked / trap:** It is the #1 GGUF production failure. The trap is re-implementing prompt formatting in the application — the template belongs to the artifact and the server should apply it, or it drifts from the training format.

---

## Level 2 — Applied & Implementation

**Q15. You run `python code/08_quantize.py --model X --method bnb --bits 3`. What exactly does it print?**
- **Answer:** It exits before doing anything with `--method bnb supports --bits 4 or 8, not 3.` followed by `  For 2/3-bit, use --method gguf (Q2_K/Q3_K) or --method gptq.` The guard exists because otherwise `--bits 3` becomes `load_in_3bit=True` and fails deep inside `transformers`.
- **Why asked / trap:** Tests whether they read the tool's own contract. The trap is answering "it quantises to 3-bit" — bitsandbytes offers 4-bit and 8-bit and nothing else.

**Q16. Why does `--method` not offer HQQ?**
- **Answer:** Because it used to, and the branch fell through to AWQ: `--method hqq` silently wrote an **AWQ checkpoint into a directory named `hqq-4bit`**. The flag's help text says so verbatim. Removing an option is the correct fix when it cannot be honoured — a wrong artifact with a plausible name is worse than a missing feature.
- **Why asked / trap:** A mature-engineer question about footguns, and the best example of why a format is not a label. The acceptable answer if you want it back: re-add it *with* a post-write assertion that the artifact's `quantization_config` names the method you asked for.

**Q17. `--group-size` from 128 to 32. What does that cost on a 7B? Show the arithmetic.**
- **Answer:** Asymmetric effective bpw is `b + 32/g`. At `g = 128`: `4 + 0.25 = 4.25`. At `g = 32`: `4 + 1.0 = 5.0`. Delta **+0.75 bpw** = `7e9 × 0.75 / 8 = 0.656 GB` ≈ **0.66 GB**. Accuracy bought with real bytes.
- **Why asked / trap:** "Smaller is better" is true for quality and false for everything else; they must price it. The trap is dropping to 32 by reflex at 4-bit — the script's own help says 32 is *for 3-bit*.

**Q18. When would you pass `--desc-act`?**
- **Answer:** At 3-bit, or where the accuracy delta is measurable, and only after confirming your serving kernel accepts the reordered layout. The help prices it: "Better quality, ~10% slower." At 4-bit with `group_size=128` on a modern chat model the honest answer is usually *not*.
- **Why asked / trap:** A flag with a real asymmetric cost. The trap is passing it and then serving with a Marlin build that rejects it — a load-time failure on the server, long after the quantisation run.

**Q19. What is `--calib-samples`'s default, what does the script warn about, and what is the built-in fallback for?**
- **Answer:** Default **256**; help says "128 minimum, 256–512 preferred." Without `--calib-dataset`, `build_calibration` repeats a **16-item generic English list** to reach `n` and prints `⚠  These are generic English prompts. For production, pass --calib-dataset with data that matches your traffic.` That fallback is a **smoke test, not a calibration set** (CH-11 §4.4).
- **Why asked / trap:** The calibration set is the highest-leverage knob in the pipeline. The trap is treating 256 generic prompts as equal to 256 in-domain samples — the scales fit the wrong distribution and the failure is "benchmark fine, production bad."

**Q20. The AWQ branch passes `calib_data=calib` explicitly. Why does that matter?**
- **Answer:** Because AutoAWQ otherwise falls back to its own generic English corpus, which would make the script's "calibrate on YOUR OWN data" warning a lie and tune the artifact on the wrong distribution. The same code comment flags that AutoAWQ was archived in 2024 and that new work should prefer llm-compressor.
- **Why asked / trap:** Tests whether they know a library default can silently contradict a stated protocol. The trap is believing a default is neutral — a default calibration corpus is a *choice of distribution*, made by someone who has never seen your traffic.

**Q21. Walk the GGUF path in `code/08_quantize.py`. What are the two calls, and what happens if the second fails?**
- **Answer:** Step 1: `python -m llama_cpp.convert_hf_to_gguf <model> --outfile model-f16.gguf --outtype f16`. Step 2: `python -m llama_cpp.llama_quant model-f16.gguf model-<TYPE>.gguf <TYPE>`. On failure it **keeps the f16 intermediate** and prints the retry command plus the note that `llama-cpp-python` may not ship a runnable `llama_cpp.llama_quant`, in which case use the binary `llama-quantize model-f16.gguf model-<TYPE>.gguf <TYPE>`, along with the tail of the tool output.
- **Why asked / trap:** A previous version ran step 2 with `check=False`, deleted the f16, and printed "GGUF saved" unconditionally — leaving an empty directory, no error, and a success message. The trap is deleting the f16 before confirming success: re-converting a 70B is hours, re-quantising from f16 is minutes.

**Q22. You fine-tuned a LoRA on bf16 and want to serve AWQ 4-bit. Give the exact artifact chain.**
- **Answer:** **train** (bf16 base + LoRA adapter) → **merge** (adapter + base → full bf16 weights, `code/09_merge_and_export.py`) → **quantise** the *merged bf16* model (`code/08_quantize.py --model ./out/merged --method awq`) → **serve** the quantised artifact. Merge is lossless for a bf16 LoRA; the quantisation is the only lossy step, and it is the one you ship (CH-11 §5.5).
- **Why asked / trap:** The most common production incident is serving the wrong artifact from this chain. The trap is quantising the adapter, or merging into a *quantised* base — generic `merge_and_unload()` on an NF4-loaded model bakes the reconstruction error into the merged weights (CS-11 §10 item 4, §17.3).

**Q23. Which can you serve directly: `out/sft-lora/`, `out/merged/`, `out/serve/awq-4bit/`, `out/serve/gguf-Q4_K_M/`?**
- **Answer:** The last two. `out/sft-lora/` is an adapter needing a base; `out/merged/` is servable but expensive and slow. Only the AWQ directory (with its `quantization_config` and `qweight`) and the single `.gguf` file are *serving artifacts* — and **neither can be trained**.
- **Why asked / trap:** "Which artifact am I holding?" resolves most incidents in this module. The trap is assuming `out/merged/` is the thing to ship; it is the thing to *quantise*.

**Q24. `--quant-type Q2_K` — when, if ever?**
- **Answer:** Almost never — the script's own `--quant-type` help says `Q2_K = do not use unless you are truly desperate; it is a visible cliff.` The right response to "nothing fits at 4-bit" is tensor parallelism, a smaller or distilled model, or KV-cache quantization — not a lower bit-width.
- **Why asked / trap:** Tests whether the candidate has a *floor* on the bit-width axis or just keeps turning the dial down. The trap is reaching for 2-bit to fit a 70B on one card and discovering the model cannot follow instructions after the date is promised.

**Q25. What must be true of the relationship between the calibration set and the evaluation set?**
- **Answer:** They must be **disjoint**, and you should assert it — hash both and compare, as CH-11 §12's check 3 does. If calibration leaks into eval, your scales were fit on exactly the activations the gate measures, and a model that *beats* fp16 is a **bug**, not a win.
- **Why asked / trap:** One assertion separates a real gate from a memorisation test. The trap is splitting by chunk rather than by document — chunk-level splits leak near-duplicates and inflate the result.

**Q26. Your model is at `--bits 8`. What is the honest quality claim?**
- **Answer:** At 8-bit, weight-only quantization is close to lossless on most models and the artifact is ~2× smaller than bf16 — and 4-bit is usually the better trade if you can gate it. The honest framing is not "8-bit is safer" but "8-bit costs double the bytes for a delta you may not be able to measure" — and at weights-only 8-bit you have bought **no compute win at all**.
- **Why asked / trap:** Tests whether bit-width is a justified dial or a safety blanket. The trap: "we ship 8-bit because it's safer," with no gate, no measurement, and no statement of what was given up.

**Q27. A 3B, W4A16, 8k context, batch 32, on a 22 GB budget. Walk the fit.**
- **Answer:** Weights ≈ 1.5 GB. KV comes to **18.9 GB** in CH-11 §7.1's worked row; with ~1.5 GB overhead that is 21.9 of 22.1 GB. The binding constraint is the **KV cache**, not the weights: 18.9 of 22.1 GB is cache, so quantising the weights harder would change nothing. The levers are `--max-num-seqs`, `--max-model-len`, and FP8 KV (sm_89+ only).
- **Why asked / trap:** The question that catches people who think of VRAM as a weights problem. The trap is proposing 3-bit or 2-bit — you would be shrinking 1.5 GB of a 22 GB budget.

**Q28. A 70B on one A100-80 at `--gpu-memory-utilization 0.92`, 8k context, batch 4. Does it fit, and what do you do with what is left over?**
- **Answer:** Budget `80 × 0.92 = 73.6 GB`. KV: `2 × 80 × 8 × 128 × 2 B = 327,680 B = 320 KiB` per token fp16; at `8192 × 4 = 32,768` tokens, **10.7 GB**. Weights `70e9 × 0.5 = 35 GB`. Total `35 + 10.7 + ~1.5 ≈ 42 GB` — it fits with ~30 GB spare. **The spare is the point:** it becomes KV room and therefore batch, or you spend it on FP8 KV to halve the cache term (CS-11 §19.1).
- **Why asked / trap:** Two terms that do not move together, plus what the headroom is *for*. The trap is reporting "it fits" and stopping — quantisation buys concurrency, not just fit.

---

## Level 3 — Advanced, Internals & Theory

**Q29. Derive why `Δ_err ≈ (w_j − ŵ_j)² · H_jj` follows from the objective.**
- **Answer:** Expand `‖WX − ŴX‖²_F` column by column. Isolating column `j`'s contribution gives `(w_j − ŵ_j)ᵀ XᵀX (w_j − ŵ_j) = (w_j − ŵ_j)² H_jj` when only that column is perturbed, because `H = XᵀX` is exactly the quadratic form's matrix. The cost of rounding column `j` is its squared error scaled by how active input channel `j` is over the calibration set.
- **Why asked / trap:** The derivation is the reason the algorithm exists. The trap is treating `H_jj` as a heuristic weight — it falls out of the objective in one line and is not a design choice.

**Q30. Why is the compensation term `δ_j · (H⁻¹)_jk / (H⁻¹)_jj` rather than simply `δ_j`?**
- **Answer:** The error just introduced in column `j` must be *undone in the not-yet-quantised columns*, and the cheapest direction to push it is not the identity direction — it is the one that costs least under the same quadratic form. Solving that minimisation gives the `H⁻¹`-weighted update; pushing `δ_j` verbatim into every later column would over-correct.
- **Why asked / trap:** Separates "knows GPTQ has error compensation" from "knows what it is solving." The trap is describing it as "spreading the error," which loses the point: it is the *minimum-cost* redistribution.

**Q31. AWQ applies `W·diag(s)` and `X·diag(s)⁻¹`. Why is that not lossy?**
- **Answer:** It is an identity: `W·X = (W·diag(s))·(diag(s)⁻¹·X)` exactly, in exact arithmetic. Both factors are diagonal, so they commute with the matmul and cancel. Practically the activation-side factor is folded into the preceding LayerNorm/Linear at export, so there is **zero inference overhead** — which is why AWQ can claim an activation-side correction without ever quantising an activation.
- **Why asked / trap:** The elegance of the method is the identity, and it is why W4A16 suffices. The trap is "it's an approximation that works because s is small" — the rescaling is exact; the approximation is the search over `s` and `α`.

**Q32. Why is W4A4 hard when W4A16 is easy? Name the two solution families.**
- **Answer:** Because activations have **outliers** — a handful of channels 10–100× the median — and a 4-bit activation grid must either clip them or waste its levels on them. Weights are smooth; activations are not. Family (a) **migrate or mask the outlier**: SmoothQuant moves it into the weights, LLM.int8() keeps the top ~0.1% of dimensions in fp16. Family (b) **rotate it away**: Hadamard-style rotations (QuaRot, SpinQuant) spread a single large coordinate across all of them (CS-11 §4.3, §4.5).
- **Why asked / trap:** W4A4 is where the field is going. The trap is assuming 4-bit weights imply 4-bit activations — they are independent axes and almost nothing in production quantises both at 4.

**Q33. Quantitatively, what does a Hadamard rotation do to an outlier?**
- **Answer:** It takes a vector whose norm is concentrated in one coordinate and spreads it so the max-to-norm ratio goes from `‖x‖∞ : ‖x‖₂` toward `‖x‖₂/√n`. For `n = 4096` that is up to **64× smaller range** (CH-11 §2). The price is that the rotation must be folded into the surrounding weights or supported by the kernel — and a rotation applied *after* RoPE is not fusable (CH-11 §9.4).
- **Why asked / trap:** The current answer to W4A4, with both the win and the constraint. The trap is forgetting the post-RoPE Q/K case, which is exactly the part that cannot be folded.

**Q34. Why is a 4-bit model ~2–3× faster at batch 1, ~3–4× in aggregate, and never 4×?**
- **Answer:** The 4-bit kernel is *dequantize-then-matmul*: it expands weights to FP16 inside the operation and executes on **FP16 tensor cores**, so FLOP throughput is unchanged. What you saved is **bytes of weight read**, which binds at batch 1 → ~4× fewer bytes, ~2–3× faster in practice because kernel overhead does not shrink. In aggregate the freed weight memory becomes KV room, which becomes batch, giving ~3–4× (CH-11 §7.4, CS-11 §17.6).
- **Why asked / trap:** The most common wrong expectation in capacity planning. The trap is quoting 4× — the only true 4× is the weight-bytes term.

**Q35. What is the roofline ridge point, and what happens past it?**
- **Answer:** The arithmetic intensity at which a kernel stops being memory-bound and becomes compute-bound — roughly **295 FLOP/byte on an H100**. At batch 1, decode sits far below it and weight quantization is a pure win; as batch grows past roughly 74, a weight-only int4 kernel becomes compute-bound on the FP16 tensor cores and the gain collapses toward 1× — and can go **below** fp16, because you now pay dequantisation with no bandwidth saving.
- **Why asked / trap:** The quantitative form of "it depends on your batch." The trap is measuring at batch 1 and deploying at batch 64 — the two regimes have opposite conclusions.

**Q36. Marlin wants `group_size ∈ {128, −1}` and `sym=True`. Why?**
- **Answer:** Its inner loop is written against a specific memory layout — a fixed group stride and no zero-point table to load. `group_size = −1` means per-channel, the layout the fastest path prefers; `sym=True` removes the zero-point tensor and its `+16/g` bpw. An asymmetric or odd-group checkpoint still runs, but on a generic dequant path, and the failure looks like "it loaded and then ran slowly with no error."
- **Why asked / trap:** The difference between quantising for *a* kernel and for *your* kernel. The trap is discovering this after the run — `format` exists so you choose the kernel at quantisation time, not at load time.

**Q37. Perplexity is not a gate. Give three concrete losses it misses.**
- **Answer:** (1) **Structured output** — JSON/schema validity can drop while perplexity is flat, because the failure lives in sequence-level constraints, not next-token likelihood. (2) **Long-context retrieval** — a K-quantisation error flips the attention argmax, so a needle at depth 0.9L stops being found while average token loss barely moves. (3) **Code and maths**, which degrade first and most visibly, dominated by rare tokens the corpus averages away. The gate is mean/p95 KL, top-1 agreement, and a task metric (CS-11 §12.1–12.4).
- **Why asked / trap:** A candidate still gating on perplexity has not shipped a quantised model to real traffic. The trap is "perplexity is a lower bound so it's conservative" — it is neither a bound nor conservative, and it is blind to precisely the modes that break.

**Q38. Two quantisation runs with identical arguments produced different weights. Explain.**
- **Answer:** A sum over a calibration batch is **not associative in floating point**, and different libraries, kernel versions and GPU architectures tile and accumulate it differently. Calibration ordering, GPU arch, kernel selection and library version all enter the numerics. The consequence is procedural: pin library + torch + CUDA + arch, **hash the output weights**, and if the hash changed treat it as a new artifact whatever the config says (CS-11 §10 item 19, §17.11).
- **Why asked / trap:** The reproducibility question, and the answer is a manifest, not a seed. The trap is setting a random seed and calling it deterministic — this nondeterminism comes from reduction order, not sampling.

**Q39. A MoE model's output KL looks fine but quality dropped. What is a bad detector, and what should you measure?**
- **Answer:** Output KL is a bad detector because it averages over the whole vocabulary while the damage is concentrated in **which expert runs** — quantisation error can flip the router's top-k, changing the computation entirely while the mean next-token distribution stays close. Measure **router agreement rate** (top-k expert selection vs the fp16 model) and hold it to **≥98%**.
- **Why asked / trap:** A concrete, non-obvious failure mode with a specific mitigation. The trap is concluding "KL is fine, so it must be the data" — MoE needs its own gate.

**Q40. `llama-quantize` has no Hessian, no Cholesky and no activation-aware scaling. Why is it still a quantization method?**
- **Answer:** Because a `Q4_K_M` file **is a ~4.85-bit-per-weight quantisation of the fp16 base** — post-training (no gradients), weight-only (activations stay fp16/fp32), block-wise (32/16/256-weight blocks with quantised scales), and mixed-precision (the `_S`/`_M`/`_L` suffix decides which tensors stay higher-precision). It is a *different family* from GPTQ/AWQ, not the absence of one.
- **Why asked / trap:** The misconception CS-11 §17.9 corrects explicitly, and it decides whether the candidate can reason about GGUF quality at all. The trap is agreeing with the video's "GGML cannot perform the quantization", which would imply a `Q4_K_M` file is somehow the unquantised model.

---

## Level 4 — System Design & Scenario

> These are 15-minute whiteboard questions. Answer in the fixed shape: **requirements → constraints → design → trade-offs → failure modes**.

**Q41. CPU-only on-prem, 32 GB RAM, a 13B, shipping this week.**
- **Answer:** GGUF `Q4_K_M` with an imatrix built from the customer's own corpus. Roughly `13e9 × 4.85 / 8 ≈ 7.9 GB`, comfortably inside 32 GB with room for KV. There is no CUDA in the building, so GPTQ and AWQ are not options at all, and `llama.cpp` is the only runtime that turns this into a service (`llama-server` is OpenAI-compatible). If the corpus is specialised, the imatrix is the difference between domain-native and not.
- **Why asked / trap:** Tests whether the format decision is driven by the *machine* rather than the benchmark table. The trap is Q8_0 "for safety" on a 13B, leaving no room for context — or proposing a conversion from a GPTQ checkpoint, for which no path exists.

**Q42. A VLM for document extraction. What do you exclude, what must the calibration set contain, and what is the evidence the exclusion matters?**
- **Answer:** Exclude `vision_tower`, `multi_modal_projector` and `lm_head`; quantise only the language tower's linear layers. The calibration set must contain **image-text pairs from the same document distribution** — a text-only corpus never exercises the projector, so its activation statistics are never observed, and the model hallucinates on dense pages while scoring perfectly on text evals. The ablation in CS-11 §15.3: **91.4%** field-level exact match against a **92.6%** bf16 reference, collapsing to **78.9%** when the vision tower is quantised too — damage concentrated on small-font fields.
- **Why asked / trap:** The `ignore` list is the whole technique for VLMs and MoE, and it is the field most often omitted from a config. The trap is text-only calibration plus a text eval.

**Q43. The same model must serve vLLM on A100s and an Apple laptop. How many artifacts, produced how?**
- **Answer:** **Two**, both from the bf16 base — a GPTQ or AWQ safetensors checkpoint for vLLM (with the matching `*_marlin` kernel selected explicitly) and a GGUF `Q4_K_M` for the laptop. There is no supported AWQ→GGUF or GPTQ→GGUF path: the weight layouts share no representation, so every re-target is a fresh quantisation run from bf16 — which is the other reason to keep the bf16 original resident forever (CS-11 §17.8).
- **Why asked / trap:** Tests whether they know the conversion matrix is mostly empty and plan the inventory rather than discovering it. The trap is "convert it with a script" — there is no script.

**Q44. Compliance asks you to prove which quantised model is in production. Name five manifest fields and the failure each prevents.**
- **Answer:** `base_revision` (a Hub repo is mutable — prevents "the parent changed under us"); `calibration.sha256` (prevents the most common quality regression and makes a re-quantisation reproducible); **tooling versions** (prevents a rebuild that silently differs because a kernel default moved); `weights_sha256` (proves the served file is the gated file); `target.arch` + engine version (prevents "fine on the A100, wrong on the H100"). Anything not in the manifest is a variable you cannot control (CS-11 §19.10).
- **Why asked / trap:** The difference between a checkpoint and a governed artifact. The trap is listing "the config file" — the config records intent, the hash records what was produced, and only the second is auditable.

**Q45. A quantised model is serving well and the base model gets a security patch. What is the full re-release cost?**
- **Answer:** Not a re-quantisation. It is: re-pull the base at the new revision, re-run the gate on the new base against the old one, re-calibrate and re-quantise (the activation statistics moved with the weights), re-run the full gate, re-run the task evals, re-derive the needle test if KV is quantised, and re-issue the manifest with a new `base_revision` and `weights_sha256`. Budget it as a release, not a build step — and the merge step also has to be re-run if the artifact came from a LoRA.
- **Why asked / trap:** It is the maintenance cost behind every "can we just quantise it once?" plan. The trap is assuming the old calibration still applies — calibration is a function of the weights, and the weights changed.

**Q46. A 70B must serve 32k context and you have one H100-80. Design it.**
- **Answer:** Weights at W4A16 `= 70e9 × 0.5 = 35 GB`. KV fp16 at 32k: `327,680 B × 32,768 = 10.7 GB`; FP8 halves it to ~5.4 GB. Total ≈ 42 GB of 73.6 GB with FP8 KV, leaving ~30 GB — which is the batch headroom. Decisions: FP8 KV (`--kv-cache-dtype fp8 --calculate-kv-scales`; sm_89 means H100 is eligible), `--max-model-len 32768` set to the *real* need rather than the model maximum, `--max-num-seqs` chosen from the spare, and a needle test at 0.5L and 0.9L as the acceptance criterion, since K precision is what fails non-gracefully.
- **Why asked / trap:** Three terms, one flag with a hardware gate, and an acceptance test that must be named. The trap is setting `--max-model-len` to the model's native maximum and OOMing at startup (Q48).

---

## Level 5 — Debugging & Incident Response

> Answer in a fixed shape: **what I look at first, what each observation rules in or out, and the fix.**

**Q47. Your 4-bit model passes the KL gate on prose and fails on production RAG traffic.**
- **Answer:** First compare the calibration corpus's token/embedding distribution against real production prompts. Ranked causes: (a) **calibration domain mismatch** — scales fit on the wrong activation distribution; fix by re-quantising on 128–512 of *your* samples. (b) **prompt-shape sensitivity** — long contexts with many quoted spans exercise attention over distant tokens, where argmax flips cost you the citation; diagnose with a needle test at 0.5L and 0.9L plus a per-position KL breakdown. (c) **a structured-output or `lm_head` effect** — if the failure is in the format (dropped quotes) rather than retrieval, exclude `lm_head` or drop to `group_size=32`; diagnose with schema-validity rate.
- **Why asked / trap:** The archetypal "benchmark fine, production bad" incident, and why the calibration rule exists. The trap is going up a bit-width first — that buys ~0.5 bits of headroom against a problem about distribution, not precision.

**Q48. vLLM OOMs at startup with `--max-model-len` even though the weights loaded fine.**
- **Answer:** The engine sized the **KV pool** for the requested context, not the weights. Check `torch.cuda.get_device_capability()` for FP8 eligibility, cap `--max-model-len` to the *real* need rather than the model's native maximum, cap `--max-num-seqs`, and add tensor parallelism if neither is enough. Recompute the KV term from `2 · L · h_kv · d_head · seq · batch · bytes` before picking a number.
- **Why asked / trap:** The failure that turns "the weights fit" into "the service does not start", and it is arithmetic you can do in advance. The trap is reducing `--gpu-memory-utilization` — that changes the fraction of a too-large number.

**Q49. 4-bit is slower than fp16 at batch 64.**
- **Answer:** Look at the kernel name in the engine logs first. If it is a Marlin/ExLlama kernel the run is compute-bound on FP16 tensor cores: past the roofline ridge you pay dequantisation with no bandwidth saving, so weight-only quantisation can be *slower* than fp16. Fix: add activation quantisation (FP8 W8A8 on sm_89+, INT8 on sm_80) or serve fp16. If the kernel name is a generic dequant path, the checkpoint did not match the fast-layout requirements (`group_size ∈ {128,−1}`, `sym=True`) and the fix is a re-quantisation, not a config change.
- **Why asked / trap:** Two distinct causes with two distinct fixes, separated by one observation. The trap is assuming 4-bit is always the faster option — at high batch it is a different trade.

**Q50. A GGUF model is fluent, well-formed, and answers the wrong question.**
- **Answer:** Inspect the GGUF metadata (`llama-gguf model.gguf | head -40`) and compare the rendered prompt against `apply_chat_template` on the original tokenizer. The most likely cause is a **missing or wrong chat template in the metadata**, which raises no error and degrades output silently. Fix: re-convert from the correct base, or supply the template, then re-test with the same prompts used pre-quantisation.
- **Why asked / trap:** "Fluent and wrong" is the signature of a *format* problem, not a precision problem. The trap is dropping to Q8_0 — precision was never the issue, and you have doubled the file size for nothing.

**Q51. The quantised model *beats* the fp16 baseline on your gate.**
- **Answer:** Treat it as a **bug**, not a win: quantisation only destroys information, so a quantised model cannot be better. The overwhelmingly likely cause is that the **calibration corpus leaked into the evaluation set** — the scales were fit on exactly the activations the gate measures. Diagnose by hashing both sets and asserting disjointness. Secondary candidates: the eval set is too small to distinguish the two models, or the "fp16 baseline" arm is misconfigured.
- **Why asked / trap:** A strong prior with a specific, checkable cause. The trap is shipping it because the numbers look great and the deadline is Friday.

**Q52. A needle test passes at depth 0.1L and fails at 0.9L.**
- **Answer:** This is a **K-quantisation failure**, not a weight failure: K errors flip the attention argmax, so the failure is discontinuous and depth-dependent — exactly this signature. Check whether K was quantised below FP8. Fixes in order: stop at FP8 KV (`--kv-cache-dtype fp8 --calculate-kv-scales`, sm_89+); if already at FP8, revert K to fp16 while keeping V quantised (per-channel K, per-token V); if neither is possible, cap `--max-model-len`. Never ship a KV quantisation without a needle test at 0.5L and 0.9L (CH-11 §3 stop condition X5).
- **Why asked / trap:** The depth signature *is* the diagnosis. The trap is re-quantising the weights at a higher bit-width — the weights were never the problem.

**Q53. A rebuild six months later is measurably worse, config unchanged.**
- **Answer:** Two candidates. (a) **Tooling defaults moved** — a kernel default, library default or `damp_percent` changed between versions, so the same config no longer means the same artifact. (b) **The calibration corpus went stale** — traffic shifted (new language, new document type, longer system prompt), so the activation ranges the scales were fit to no longer describe your inputs. Fix: pin the toolchain and record it in the manifest, **re-run the gate on every rebuild**, and re-calibrate on a schedule or on a drift alert. A hash mismatch on the output weights identifies (a) before you serve it.
- **Why asked / trap:** The maintenance question behind "quantise once, ship forever" (CS-11 §17.14). The trap is assuming the base model changed — possible, and what `base_revision` is for, but the two causes above are more common.

**Q54. A colleague quantised a model with GPTQ and now wants to full fine-tune it. The loss "doesn't move."**
- **Answer:** The plan is impossible as stated. `round()` has zero derivative almost everywhere, so the quantised codes carry no useful gradient, and the artifact is a fixed encoding rather than a parameterisation — there is nothing for the optimizer to update, and a "fine-tune" either does nothing or degrades a model you cannot restore. Diagnostic: print `sum(p.requires_grad for p in model.parameters())` and the weight delta after one step. The three legal orderings: **bf16 → fine-tune → merge → quantize** (default), **QLoRA** over a frozen NF4 base (merge into bf16, re-quantise if needed), and **QAT** (fake-quantization with a straight-through estimator from a bf16 start).
- **Why asked / trap:** The most expensive misconception in this module in its corporate form, and the fix is a workflow change, not a hyperparameter. The trap is lowering the learning rate or unfreezing more layers — neither addresses it.

---

## Rapid Fire — True / False / One-Liner

| # | Statement | Answer | One-line why |
|---|---|---|---|
| 1 | AWQ quantises activations. | **False** | The artifact is W4A16; "activation-aware" names the statistic that picks the scale. |
| 2 | `Q4_K_M` is 4 bits per weight. | **False** | ~4.85 bpw including k-quant block metadata. |
| 3 | You can convert a GPTQ checkpoint to GGUF. | **False** | Different weight layouts; re-quantise from the bf16 base. |
| 4 | 4-bit weights run on 4-bit tensor cores. | **False** | The kernel dequantises to FP16 and runs FP16 maths. |
| 5 | `desc_act=True` is free accuracy. | **False** | ~10% slower inference, and some kernels reject it. |
| 6 | Perplexity is a sufficient quality gate. | **False** | It misses JSON, maths and long-context recall entirely. |
| 7 | Quantising the weights shrinks the KV cache. | **False** | Separate term; quantise KV separately (FP8, sm_89+). |
| 8 | bitsandbytes NF4 writes an artifact you can serve. | **False** | It quantises at load; there is no portable file. |
| 9 | GGUF is a container and `Q4_K_M` is a method. | **True** | One file format, many block-quant schemes inside it. |
| 10 | Two identical quantisation configs give identical weights. | **False** | FP reduction order varies by kernel, library and GPU arch. |
| 11 | A quantised model can beat its fp16 baseline on your gate. | **Only as a bug** | Calibration leaked into eval. Hash both, assert disjoint. |
| 12 | `group_size=32` is the right call at 4-bit. | **False** | It is a 3-bit lever; at 4-bit you pay +0.75 bpw for noise. |

---

## Coding / Whiteboard Tasks

### Task 1 — Write the gate

Write the function that decides whether a quantised checkpoint ships. It receives an fp16 model, a quantised model, a tokenizer and a list of held-out texts, and returns the metrics and a pass/fail.

**Expected solution sketch**

```python
import math, torch, torch.nn.functional as F

@torch.no_grad()
def kl_gate(fp16_model, quant_model, tokenizer, texts, max_len=2048, device="cuda"):
    kls, agrees, nll_f, nll_q, ntok = [], 0, 0.0, 0.0, 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 8:
            continue
        lf = fp16_model(ids).logits[:, :-1].float()
        lq = quant_model(ids).logits[:, :-1].float()
        pf, pq = F.log_softmax(lf, -1), F.log_softmax(lq, -1)
        kls.append((pf.exp() * (pf - pq)).sum(-1).flatten().cpu())   # KL(P_fp || P_q)
        agrees += (lf.argmax(-1) == lq.argmax(-1)).sum().item()
        tgt = ids[:, 1:]
        nll_f += F.cross_entropy(lf.flatten(0, 1), tgt.flatten(), reduction="sum").item()
        nll_q += F.cross_entropy(lq.flatten(0, 1), tgt.flatten(), reduction="sum").item()
        ntok  += tgt.numel()
    kl = torch.cat(kls)
    return {"mean_kl": kl.mean().item(), "p95_kl": kl.quantile(0.95).item(),
            "max_kl": kl.max().item(), "top1_agree": agrees / ntok,
            "ppl_delta_pct": 100 * (math.exp(nll_q / ntok) / math.exp(nll_f / ntok) - 1)}
```

- **The decisions being graded:** the KL direction (`KL(P_fp ‖ P_q)`, not the reverse); reading **p95 and max** as well as the mean; comparing against **your own** fp16 baseline rather than a paper; excluding very short sequences; adding a *task* gate (JSON-validity rate, IFEval, GSM8K, needle@0.5L and @0.9L) on top of the distributional one.
- **Grading:** pass thresholds `mean_kl ≤ 0.10 AND p95_kl ≤ 1.00 AND top1_agree ≥ 0.95`. Stating them *before* running anything is the discipline being tested (CH-11 §3 stop condition X1).
- **Fail condition:** returning only `mean_kl`; gating on perplexity alone; or accepting a run where `mean_kl = 0.03` but `max_kl = 47` without comment.

### Task 2 — Size the deployment and choose the KV flag

An 8B GQA model (32 layers, 8 KV heads, head_dim 128) must serve 8k context at batch 8 on one L40S (48 GB). Weights are W4A16. Does it fit, is FP8 KV available, and what does it buy?

**Expected solution sketch**

```python
L, h_kv, d_head, seq, batch = 32, 8, 128, 8192, 8
kv = 2 * L * h_kv * d_head * seq * batch * 2        # one K + one V per layer, fp16
print(f"{kv/1e9:.1f} GB fp16  |  {kv/2/1e9:.1f} GB fp8")

import torch
print(torch.cuda.get_device_capability())           # L40S = (8, 9) -> sm_89, FP8 KV available
```

- **The decisions being graded:** writing the KV formula with **both K and V** (the `2 ·`); using `h_kv` (the GQA head count) and not the attention head count; checking `torch.cuda.get_device_capability()` rather than assuming the card supports FP8; and noticing that at this size the answer is *not* "turn it on" — the cache is about **1.1 GB fp16 / 0.5 GB fp8** out of a 48 GB budget, so the flag buys ~0.5 GB and adds a K-precision risk with no capacity benefit.
- **Grading:** a candidate who computes the number and then **declines** the flag, with the reason, is stronger than one who enables every flag that exists.
- **Fail condition:** using the full attention head count; omitting the `×2`; or enabling FP8 KV because "it is free" — it is a K-precision change requiring a needle test at 0.5L and 0.9L.

---

## Cheat Sheet of Numbers To Memorize

| Quantity | Value | Why it matters |
|---|---|---|
| AWQ artifact precision | **W4A16** | It does not quantise activations |
| `Q4_K_M` real bpw | **~4.85** | A 70B "4-bit" GGUF is ~42 GB, not 35 GB |
| Honest 4-bit size reduction | **3.2–3.6×**, not 4× | Embeddings, `lm_head`, norms and metadata |
| `group_size` 128 → 32 | **+0.75 bpw** = **+0.66 GB** at 7B | Accuracy bought with real bytes |
| `desc_act=True` | **~10% slower** inference | And rejected by some kernel builds |
| `damp_percent` | `0.01` (optimum) / `0.05` (gptqmodel) | Sweep 0.01–0.2 when a layer fails |
| Calibration samples | **128 min, 256–512 preferred**, in-domain | The highest-leverage knob in the pipeline |
| Gate | `mean KL ≤ 0.10`, `p95 KL ≤ 1.00`, `top1 ≥ 0.95` | Against your own fp16 baseline |
| `08_quantize.py` escalation rule | `KL > ~0.1` mean → up a bit-width or `group_size=32` | The script's own threshold |
| Quantise-run peak | `params × 2 B + 2–4 GB` | 7B → ~16–20 GB; 70B → ~150 GB |
| 70B KV per token | **320 KiB** fp16 | `2 × 80 × 8 × 128 × 2 B` |
| H100 roofline ridge | **~295 FLOP/byte** | int4 goes compute-bound around batch 74 |
| Batch-1 decode / aggregate | **~2–3× / ~3–4×** | Never 4× |
| FP8 KV | **2×** cache reduction, **sm_89+** only | Needs `--calculate-kv-scales` in vLLM |
| MoE router agreement | **≥ 98%** | Output KL is blind to expert flips |
| VLM ablation | **91.4% → 78.9%** field-level exact match | With vs without excluding the vision modules |

---

## Answers To The Self-Check Questions From CS-11

- **Q1 — 70B KV and the A100 fit.** Per token `2 × 80 × 8 × 128 × 2 B = 327,680 B = 320 KiB` fp16, `160 KiB` FP8. At `8192 × 4 = 32,768` tokens: **10.7 GB** fp16, **5.4 GB** FP8. Weights at 4-bit ≈ `70e9 × 0.5 = 35 GB`; total `35 + 5.4 + ~1.5 ≈ 42 GB` against `80 × 0.92 = 73.6 GB` — it fits with ~30 GB spare, and the spare *is* the point because it becomes KV room and therefore batch. (CS-11 §19.1) — **trap:** Two budget terms that do not move together, plus what the headroom is for. The trap is answering "it fits" and stopping there.
- **Q2 — GPTQ's objective, `H`, `Δ_err`, `damp_percent`, `desc_act`.** Objective `argmin_Ŵ ‖WX − ŴX‖²_F`; `H = XᵀX` is the input second moment over calibration, so `H_jj` measures channel activity; expanding gives `Δ_err ≈ (w_j − ŵ_j)²·H_jj` per column, which is why columns are quantised in decreasing `H_jj` and each residual subtracted via `H⁻¹`. `damp_percent` adds `λI` so a singular `H` inverts; `desc_act` implements the decreasing order across the layer. ~10% slower inference, and some Marlin builds reject it. (CS-11 §19.2) — **trap:** The full chain in one answer. The trap is describing `H` as the loss Hessian.
- **Q3 — "AWQ quantizes activations."** Correct them: the name says AWQ is *aware of* activations, not that it quantises them. `max|X_j|` only chooses the per-channel scale; the deployed artifact is W4A16 because `W·X = (W·diag(s))·(diag(s)⁻¹·X)` lets `diag(s)⁻¹` be folded into the preceding LayerNorm or Linear where it is exact fp16 and free. Quantised activations means W8A8, a different family — and AWQ is post-training, with no gradients. (CS-11 §19.3) — **trap:** The highest-frequency misconception in the module. The trap is correcting the precision but not the reason (the fold).
- **Q4 — 200 in-domain vs 10,000 generic samples.** Use the **200 in-domain**, all of them — the accuracy curve knees at a few hundred and diversity beats count past that. `seq_len` near the serving context; 2048 is a good default. The critical constraint: calibration and evaluation must be **disjoint**, or the gate measures memorisation. (CS-11 §19.4) — **trap:** Tests the *hierarchy* of calibration properties: domain, then size, then length. The trap is blending the two corpora "to be safe", which dilutes the only signal that mattered.
- **Q5 — Gate passes on prose, fails on RAG traffic.** Ranked: (a) calibration domain mismatch, diagnosed by comparing the calibration token/embedding distribution against production prompts; (b) prompt-shape sensitivity, diagnosed with a needle test at 0.5L/0.9L and a per-position KL breakdown; (c) a structured-output or `lm_head` effect, diagnosed with schema-validity rate. (CS-11 §19.5) — **trap:** Ranked diagnosis with a *diagnostic for each* is the skill. The trap is listing causes without saying how to tell them apart.
- **Q6 — Why not 4× faster, and what *is* quadrupled?** The kernel dequantises to FP16 and executes on FP16 tensor cores, so FLOP throughput is unchanged. What quadruples is (i) the bytes of weight read per token and (ii) KV-cache capacity per unit VRAM — giving ~2–3× batch-1 decode (bandwidth-bound) and ~3–4× aggregate throughput (concurrency-bound). (CS-11 §19.6) — **trap:** The capacity-planning number everyone gets wrong. The trap is naming FLOPs as one of the quadrupled quantities.
- **Q7 — Quantising a VLM for document extraction.** Exclude `vision_tower`, `multi_modal_projector`, `lm_head`; quantise only the language tower's linear layers. Calibration must contain image-text pairs from the same document distribution — text-only never exercises the projector, so the model hallucinates on dense pages while scoring perfectly on text evals. Evidence: **91.4% vs 78.9%** field-level exact match, damage concentrated on small-font fields. (CS-11 §19.7) — **trap:** The `ignore` list plus calibration modality is the entire VLM technique. The trap is text-only calibration, then trusting a text eval.
- **Q8 — GPTQ then full fine-tune.** Impossible: quantised codes carry no gradient (`round()` has zero derivative almost everywhere) and the artifact is a fixed encoding with no parameterisation to update. The three correct orderings: **bf16 → fine-tune → merge → quantize**; **QLoRA** over a frozen NF4 base (merge into bf16, re-quantise if needed); **QAT** from a bf16 start with fake-quantization and a straight-through estimator. (CS-11 §19.8) — **trap:** The most expensive misconception in the field, in its corporate form. The trap is offering QLoRA as "fine-tuning the quantised model" — the quantised weights never move.
- **Q9 — FP8 `e4m3`, INT8, or NF4 for the KV cache.** Choose **FP8 `e4m3`**: the KV distribution has a heavy tail of large-magnitude attention-sink entries plus a dense mass of small values, so a floating-point grid gives relative precision where the mass is and range where the tail is; INT8's uniform grid fits that poorly, and NF4 is a *weight* quantiser with a per-block scale, wrong for a streaming cache. Hardware: **sm_89 or newer** — no fast path on A100 (sm_80). vLLM flag `--kv-cache-dtype fp8`, which must be accompanied by `--calculate-kv-scales` or the scale stays at 1.0. (CS-11 §19.9) — **trap:** Three formats, one right answer, one hardware constraint, one companion flag. The trap is forgetting `--calculate-kv-scales`, which silently wastes the format's range.
- **Q10 — Five manifest fields.** `base_revision`, `calibration.sha256`, tooling versions, `weights_sha256`, `target.arch` + engine version — preventing "the parent changed", the most common quality regression, a rebuild that silently differs, "the served file is not the gated file", and "fine on the A100, wrong on the H100" respectively. (CS-11 §19.10) — **trap:** It distinguishes a checkpoint from a governed artifact. The trap is listing the config, which records intent rather than outcome.

---

## Cross-References

| Module | Relationship to IQ-11 |
|---|---|
| **IQ-10 — Quantization I** | The prerequisite bank. All of IQ-11 assumes it and none repeats it: the affine quantiser, granularity, PTQ/QAT, LLM.int8(), NF4, QLoRA, the bit-width curve, the GGML→GGUF history. |
| `case-studies/CS-11-Quantization-Part-2.md` | The source of truth. §4 is the derivations, §12 the evaluation protocol, §17 the misconception list this bank's trickiest questions come from. |
| `case-studies/CS-10-Quantization-Part-1.md` | The foundations CS-11 §1.1 explicitly does not repeat. Read it before this bank. |
| `cheat-sheets/CH-11-Quantization-II-GPTQ-AWQ-GGUF.md` | The one-page version: decision tree, per-method snippets, the gate, the symptom → fix table. |
| `code/08_quantize.py` | The runnable implementation of every branch this bank asks about; its `--help` is the flag authority for Q15–Q21. |
| `code/09_merge_and_export.py` | Step 2 of the artifact chain — the merge that must happen *before* the quantisation. |
| **IQ-13 — Instruction Fine-Tuning** | The stage that produces the model you are about to quantise, and the masking discipline that decides whether it is worth quantising. |
| **IQ-12 — Domain-Adaptive Continued Pretraining** | The other half of "adapt then serve": CPT changes the distribution, quantisation re-encodes it. Both end at the same gate. |
| **IQ-16 — Unsloth** | Where the bf16 base you quantise is actually produced on one GPU, and where the merge-then-quantise ordering is easiest to get wrong. |

*End of IQ-11. Companion artifacts: `case-studies/CS-11-Quantization-Part-2.md`, `cheat-sheets/CH-11-Quantization-II-GPTQ-AWQ-GGUF.md`, `code/08_quantize.py`.*
