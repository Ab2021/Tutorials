# Interview Bank: Quantization

> `T10` · **Transcript coverage:** partial · [Cheat sheet](../00-cheat-sheets/T10-quantization.md) · [Case study](../01-case-studies/T10-quantization.md) · [Design blueprint](../03-design-blueprints/T10-quantization/HLD.md)

## How to use this bank

Levels are **L3** (working competence — you have shipped with this), **L4** (senior practitioner — you own the tradeoff), **L5** (staff/architect — you own the decision and its blast radius). Every answer here is a *model* answer, not a script: it shows the shape and the numbers a strong candidate reaches for, and none of it should be recited. Numbers carry provenance — `[T]` for a transcript statement with the speaker or talk named, `[R]` for a supporting-repo path, `[D]` for arithmetic derived here with assumptions shown. Where the corpus has no figure — **KV-cache accuracy**, **BitNet quality**, **the DeepSeek pricing, the FP4-rollout claim and the ASR-garbled AITER passage** — the answer says so rather than inventing one.

Questions are ordered to read as one interview: foundations and arithmetic, then the methods, then the KV lever, then the tradeoffs, then debugging, then design at scale.

---

### Foundations

#### T10-Q1 · What does quantization actually buy?
**Difficulty:** L3 · **Depth expected:** 2 min
**Question:** Forget formats and bit widths for a moment. What does quantization change in a serving system, and why is it worth doing at all?

**Model answer:** It changes two things at once, and that is what makes it unusual. The LLMOps cost walkthrough states it directly: quantization "stores weights in eight or four bits instead of 16. Together, they attack the two things you pay for: **the memory a request occupies and the compute each token costs**" `[T]`. Every other common lever touches one of those terms. Continuous batching improves throughput by sharing a weight read across sequences but does nothing about the memory each sequence's KV holds. Prompt caching skips prefill but leaves decode memory traffic unchanged. Quantization is the only lever in the standard set that reduces both the bytes you must hold and the bytes you must move per token `[T]`. In the same source's cost ladder it is the strongest single rung: 100 cost units naive 16-bit → ~42 with continuous batching → **26 with 4-bit quantization** → ~11 with prompt caching `[T]` — a further **38%** off the batched baseline `[D]`. The catch is stated just as plainly: "four-bit quantization is nearly free on many models, but on some it **quietly drops accuracy**. So **always rerun your evals after quantizing**" `[T]`. *Quietly* is the operative word — the output still reads as fluent, plausible structured data.

**Signal:** Names both terms of the saving (memory per request, compute per token) rather than just "smaller model", and volunteers the eval requirement before being asked.

**Follow-ups:**
- *Why does it reduce compute, not just memory?* — In decode, per-token memory traffic is the weight bytes; fewer bytes read is less time per token.
- *What is the saving worth in the ladder?* — 42 → 26 units; Q20 sequences it against the other levers.
- *What is the failure mode?* — Silent, non-uniform accuracy loss; Q22.

**Red flags:** Says "quantization makes the model smaller" and stops; treats it as purely a memory optimization; never mentions re-running evals.

---

#### T10-Q2 · The precision table for an 8B model
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** Give me the standard precision tradeoff for an 8B model — size, quality loss and hardware compatibility at each level.

**Model answer:** Four rows, and the fourth is the one that matters `[R]`:

| Precision | Bits | Weight size (8B) | Quality loss | GPU compatibility |
|---|---|---|---|---|
| **BF16** | 16 | 16 GB | 0% (baseline) | All modern |
| **FP8** | 8 | 8 GB | < 1% | H100 / B200 / RTX 4090 |
| **4-bit (NF4)** | 4 | 5 GB | 1–2% | All modern |
| **2-bit** | 2 | 2.5 GB | **10–15%** | Research / specialized |

Three things to read off it. **The quality column is not linear in bits** — going 8 → 4 costs one to two points; going 4 → 2 costs ten to fifteen. There is a cliff, and it sits between 4-bit and 2-bit. **The compatibility column is not uniform either**: FP8 is the constrained row, because it is a *native hardware format* rather than a compressed one and needs silicon that executes it. **The sizes are not the formula's sizes** — the naive `params × bits / 8` gives 16, 8, 4, 2 GB and the table says 16, 8, **5**, **2.5** (Q3). Note also what is *absent*: the source mentions "even 1.5-bit (BitNet)" in prose but gives it no row in the tradeoff table and **no quality figure at all** `[R]`. It is a research direction, not an option — and an answer that quotes a BitNet accuracy number is quoting something the corpus does not contain.

**Signal:** Recites the table accurately *and* names both structural features — the cliff at 2-bit and the missing BitNet row.

**Follow-ups:**
- *Why is FP8 the constrained row?* — It is hardware-native; Q5.
- *What happens below 4-bit?* — 10–15% loss and research-grade support only; the case study's §5.1 says "never".
- *Do those figures transfer to your task?* — No — they are stated for a generic model on generic tasks and must be measured `[T]`.

**Red flags:** Says 2-bit is "half of 4-bit, so half the loss"; invents a BitNet figure; presents the table as a guarantee rather than a starting point.

---

#### T10-Q3 · Why "4-bit" is not 4 bits
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** The formula says a 70B at 4-bit is 35 GB. Someone sized a box for that and the model would not fit. What did they miss?

**Model answer:** They missed the scales and zeros. The rule of thumb is `VRAM (GB) ≈ params × bits / 8` `[R]`, and it counts only the weight *values*. A sub-byte format also stores a scale per block — NF4 keeps one per block of 64 weights, GPTQ one per group of 128 `[R]` — plus zeros, and those metadata bytes are not free. The case study derives the overhead from the source's own table `[R]`: the 8B model is quoted at 16, 8, **5** and **2.5** GB against formula values of 16, 8, 4 and 2. The 8- and 16-bit rows carry **no** overhead; the sub-byte rows carry a consistent **~25%** `[D]`. So for a 70B:

| Precision | Nominal (formula) | Real (×1.25 for sub-byte) |
|---|---|---|
| BF16 | 140 GB | 140 GB |
| FP8 | 70 GB | 70 GB |
| 4-bit | 35 GB | **43.75 GB** |
| 2-bit | 17.5 GB | 21.9 GB |

Against a 96 GB box with a 15% runtime reserve — 81.6 GB usable `[D]` — that is the difference between a comfortable fit and razorthin. **The planning rule: for sub-byte formats, multiply the nominal ratio by 1.25.** The wider lesson: a rule of thumb that omits metadata under-sizes the box, and the error always shows up at load time, not at design time.

**Signal:** Produces the 1.25 factor and can say *where* the missing 25% physically lives — group scales and zeros — rather than reciting a fudge factor.

**Follow-ups:**
- *Why do the 8- and 16-bit rows carry no overhead?* — FP8 and BF16 have hardware-native scaling; the metadata is not a separate per-block artefact `[D]`.
- *What does this do to a capacity plan?* — It changes the weight term in every table; Q13.
- *Does the factor vary by format?* — Yes — it is a function of block size, so smaller blocks cost more `[D]`.

**Red flags:** Treats 35 GB as correct; knows the 1.25 factor but not what it is for; applies it to FP8 and BF16 as well.

---

#### T10-Q4 · Where does the memory actually go at long context?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** A 70B at 4-bit is 43.75 GB and the box has 96 GB. Why is there still an OOM?

**Model answer:** Because weights are neither the only tenant nor the growing one. The on-device rule of thumb is that you size "weights, then add the KV cache (it grows with context length times concurrent requests) plus roughly 10–20% runtime overhead" `[R]`. Weights are constant in context; KV is linear in context × concurrency. The case study's arithmetic for a 70B with 80 layers, 8 KV heads and head dim 128 at BF16 gives

```
KV per token = 2 × 80 × 8 × 128 × 2 bytes = 327,680 B ≈ 0.3125 MB   [D]
0.3125 MB × 60,000 tokens = 18.75 GB per bundle                     [D]
```

**One bundle's cache is a quarter of the model.** At FP8 KV it is 9.4 GB; at Int4 KV, 4.7 GB. So the realistic budget is 43.75 GB of weights *plus* 18.75 GB of KV *plus* a reserve — and the reserve is what turns "fits" into "fits exactly once". The source states the general case: in long-context work "the **KV Cache** often consumes more VRAM than the model weights themselves" `[R]`. The consequence is that **sizing a precision decision on weight bytes alone systematically under-provisions**, and the error scales with the context length of your *longest* surface rather than its average.

**Signal:** Reaches for KV before being prompted, and reproduces the 0.3125 MB/token term from its shape (2 × layers × KV heads × head dim × bytes).

**Follow-ups:**
- *What is the 2 for?* — Keys and values, one tensor each.
- *What is the cheapest way to halve it?* — KV dtype, or fewer KV heads via GQA/MQA; Q13.
- *Which surface hits this first?* — The long-context one; Q13 and Q26.

**Red flags:** Blames fragmentation or the framework; does not know KV grows with context × concurrency; sizes on the median document rather than the longest.

---

#### T10-Q5 · FP8 versus the 4-bit methods: is this the same decision?
**Difficulty:** L3 · **Depth expected:** 4 min
**Question:** Is choosing FP8 the same kind of decision as choosing AWQ at 4-bit? What is actually different?

**Model answer:** No, and the difference is categorical rather than one of degree. **FP8 is a native hardware format** — the case study quotes it as "hardware-native quantization supported by Nvidia's Transformer Engine… It provides the **speed of Int8 with the dynamic range of Float16**, making it stable for both training and inference" `[R]`. The GPU executes FP8 directly. **The 4-bit methods are compressed representations the hardware must decompress**: the weights are unpacked to a compute dtype before the matmul, so you save memory and memory traffic but you do not get a native 4-bit arithmetic path on most hardware. Three consequences follow. **Quality:** FP8's loss is quoted as "< 1%" in the quantization chapter `[R]` and as "negligible (<0.1%)" in the repo's own inference-fundamentals chapter `[R]`. **That is an internal inconsistency in the corpus** — the honest answer flags it rather than quoting whichever figure is flattering. **Compatibility:** FP8 needs H100/B200/4090-class silicon `[R]`; the 4-bit methods run on "all modern" `[R]`. **Compression:** FP8 gives 2x, where 4-bit gives roughly 4x. So FP8 is the low-risk, low-compression option you take when the hardware has it and the accuracy bar is high; 4-bit is the high-compression option you take when the bar allows it or the hardware is older.

**Signal:** Names native-versus-decompressed as the organising difference, and volunteers the <1%/<0.1% conflict instead of quoting one side.

**Follow-ups:**
- *Which do you pick with no accuracy headroom?* — FP8 if the GPU supports it; otherwise stay at BF16 and optimize elsewhere.
- *Why does the hardware check come first?* — FP8 falls back or fails if unsupported `[R]`; Q23.
- *What is dynamic FP8 scaling for?* — Per-layer scales that stop outlier channels from degrading the whole model `[R]`.

**Red flags:** Treats FP8 as "just another bit width"; quotes one FP8 accuracy figure with no awareness of the conflict; ignores the hardware constraint.

---

### Methods and mechanism

#### T10-Q6 · NF4 and the entropy-optimal bins
**Difficulty:** L4 · **Depth expected:** 4 min
**Question:** Why does NF4 exist? A 4-bit integer grid is also 16 values — what does NF4 do differently?

**Model answer:** It chooses *where the 16 values sit* to match the distribution the weights actually have. Standard Float4 has "a fixed grid that doesn't map well to the actual distribution of LLM weights, which typically follow a zero-centered normal distribution. NF4 is mathematically optimized so that **each quantization bin contains an equal number of values from the normal distribution**. This prevents 'clustering' of weights and ensures the model preserves as much information (entropy) as possible" `[R]`. The intuition: a uniform grid spends equal resolution in the tails, where almost no weights live, and too little near zero, where almost all of them live. Equal-probability-mass bins spend resolution where the density is — dense near zero, sparse in the tails. That is the standard scalar-quantizer result: for a *known* source distribution, the minimum-distortion partition gives each bin equal probability mass `[D]`. Its role in practice is specific — NF4 is "the gold standard for fine-tuning (QLoRA)" and is the QLoRA standard `[R]` — so its natural home is a workflow that includes QLoRA, not a serving-only pipeline. On the case study's server surfaces NF4 competes with AWQ at the same bit width, and AWQ's claim is stronger for a numeric task because it is *activation*-aware rather than distribution-shaped: it protects the specific weights the data says matter. And note the pairing — NF4 stores a scale per block of 64 weights `[R]`, which is where Q3's 25% overhead comes from.

**Signal:** Explains equal-mass binning as a rate-distortion choice rather than "NF4 is a better 4-bit", and places it correctly in the QLoRA workflow.

**Follow-ups:**
- *Why does the normal assumption hold?* — Trained weights are empirically near zero-centred normal; a distribution mismatch degrades NF4's advantage `[D]`.
- *When would you pick NF4 over AWQ?* — When the plan includes QLoRA fine-tuning `[R]`.
- *Where do block scales show up in planning?* — The 25% footprint overhead; Q3.

**Red flags:** "NF4 has better precision than Int4" with no mechanism; thinks NF4 needs calibration data; never connects it to fine-tuning.

---

#### T10-Q7 · AWQ: what "activation-aware" buys
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Explain AWQ. Why does protecting 1% of the weights matter, and when does it actually beat GPTQ?

**Model answer:** AWQ's premise is that quantization error is not uniform in *importance*, even when it is roughly uniform in magnitude. "Instead of quantizing all weights equally, AWQ identifies the **1% of 'salient' weights** that are most important for quality and keeps them in higher precision" `[R]`, where salience is derived from "the actual activation values seen during a small calibration run" `[R]`. The mechanism is that a weight's effect on the output is its magnitude *multiplied by* the activation magnitude it multiplies: a large weight on a channel that is almost always near zero is harmless, and a moderate weight on a channel that is always large is not. So AWQ requires calibration data — and that is also its failure mode, because **the salience it computes is only as representative as the calibration set**. The comparative claim is exact: "AWQ achieves better perplexity than GPTQ, **especially for smaller models or more aggressive quantization (e.g., 3-bit)**" `[R]`. Read the scope carefully: the advantage is largest where quantization is hardest and narrows at 4-bit on a large model. That is why the case study picks it — a 70B is large, but the quantization is aggressive and the task is numeric-heavy, "exactly the regime where AWQ's salient-weight preservation is claimed to win" `[D]`. The honest caveat: this is a corpus claim, not a measurement on your model, which is the same reason the eval gate exists.

**Signal:** Derives salience as magnitude × activation, names calibration-dependence as the failure mode, and reads the "especially at 3-bit and small models" scope rather than treating AWQ as uniformly better.

**Follow-ups:**
- *What breaks AWQ?* — A calibration set that does not resemble production traffic `[R]`; recalibrate in-domain.
- *Why 1%?* — The corpus's figure for the salient fraction; the protected fraction is a knob, not a law `[R]`.
- *Does AWQ help at 8-bit?* — Much less; the mechanism matters when the bit budget is tight `[D]`.

**Red flags:** "AWQ is more accurate" as the whole answer; does not know calibration is required; claims AWQ beats GPTQ at every bit width and model size.

---

#### T10-Q8 · GPTQ, AWQ, RTN: how do you choose?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** Three routes to 4-bit. Which do you ship, and what is the second choice?

**Model answer:** The choice is driven by two questions: do you have representative calibration data, and how aggressive is the bit width? **RTN (round-to-nearest)** is the baseline — trivial, no calibration — and has the "worst quality at a given bit width" `[R]`; it breaks first on small models and at 3-bit. It is defensible only where nothing better exists. **GPTQ** is "a 'Layer-wise' quantization method that minimizes the mean squared error of the weights" `[R]`; it is the older baseline and still the more widely available in tooling, which is its actual advantage — ecosystem reach, not quality. **AWQ** is activation-aware and keeps the **1% salient weights** in higher precision `[R]`, and it "achieves better perplexity than GPTQ, especially for smaller models or more aggressive quantization (e.g., 3-bit)" `[R]`. So the default is **AWQ at 4-bit with in-domain calibration**, and GPTQ is the compatibility fallback when a serving engine or checkpoint format does not support AWQ. The case study's decision table says exactly that, marking GPTQ "legacy compatibility" and AWQ the choice for "4-bit and below, numeric-heavy tasks" `[D]`. Two caveats to volunteer: the AWQ/GPTQ gap narrows at 4-bit on large models, so do not promise a difference you have not measured; and a vendor's pre-quantized checkpoint is neither — you still re-run the eval `[T]`.

**Signal:** Chooses on calibration availability and engine support rather than on a quality ranking, and knows GPTQ's real advantage is availability.

**Follow-ups:**
- *When is RTN acceptable?* — Essentially never when a calibrated method is available `[D]`.
- *What if the engine only supports GPTQ?* — Take GPTQ and eval; availability beats a quality edge you cannot deploy.
- *How does calibration change the choice?* — No representative data removes AWQ's mechanism entirely; Q7.

**Red flags:** Picks a method by name with no criteria; believes the AWQ/GPTQ gap is large at every setting; ignores engine support.

---

#### T10-Q9 · Where the extra bits go, and why block size matters
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Same model, same bit width, two configurations with different footprints. Beyond the bit width, what actually moves the footprint and the quality?

**Model answer:** Three knobs that trade against each other. **Block (group) size.** A scale and a zero point are stored per block — NF4 per block of 64 weights, GPTQ per group of 128 `[R]`. Smaller blocks mean finer scales, which track local weight ranges better and improve quality, and more metadata, which costs footprint. That is the mechanism behind the ~25% sub-byte overhead in Q3: it is not a constant of nature, it is a function of block size, so a 32-weight block costs more than a 128-weight block `[D]`. **What is quantized.** Weight-only quantization compresses the weights and leaves activations wider; W8A8 quantizes both. Weight-only is the safer default because activation outliers are the harder problem, and it is what "4-bit" usually means in the serving stacks `[D]`. **Whether KV is quantized too.** The case study is emphatic that this is a *separate* decision with a separate evidence base, and the runbook's instruction is to "quantize weights and KV **separately and in that order of evidence**, not together — otherwise a regression cannot be attributed" `[D]`. The practical rule: choose the bit width from the accuracy budget, the block size from the footprint budget, and never change two of these in the same release.

**Signal:** Names block size as a real knob with a footprint consequence rather than treating "4-bit" as one configuration, and insists on moving KV and weight precision separately.

**Follow-ups:**
- *Which direction does a smaller block push quality?* — Up, at a footprint cost `[D]`.
- *Why not quantize activations too?* — Activation outliers are harder; W8A8 needs smooth-quant-style handling `[D]`.
- *Why separate the weight and KV changes?* — Attribution; the runbook order is KV first, weights later `[D]`.

**Red flags:** Treats "4-bit" as one thing; changes weight and KV precision in the same release; has no notion of block size.

---

#### T10-Q10 · GGUF quant levels: Q4_K_M or Q5_K_M?
**Difficulty:** L4 · **Depth expected:** 4 min
**Question:** A 3B model has to run on a laptop under ~4 GB of app RAM. Which GGUF quant level do you pick, and why not the one most people reach for?

**Model answer:** Q5_K_M, and the reason is the job the model does rather than the size it occupies. The deployment guidance is unusually specific `[R]`: **Q4_K_M** is "the practical sweet spot (roughly **1–3% quality loss** versus FP16 at about a quarter of the size)"; **Q5_K_M** is "noticeably better for **code and reasoning** at under ~1% loss"; **Q8_0** is effectively lossless at about half FP16; **Q2/Q3** "degrade **math and reasoning by 5–10% or more**." The field adjuster's job includes reading and reconciling numbers — claim amounts, dates, procedure codes — which is precisely the capability the guidance says Q4_K_M compromises and Q5_K_M protects. So the popular default is wrong *for this surface*, and the cost of being right is ~1.25x the file size, which is affordable at 3B. Q2/Q3 are not candidates: the "5–10% or more" figure lands directly on the numeric capability that is the surface's entire purpose. A second decision stacks on this one — the case study chooses **QAT** here, because the source claims QAT is "**mandatory for models smaller than 3B parameters to remain useful at 4-bit**" `[R]`, and that claim is stated without qualification or measurement, so it is treated as a hypothesis that gates the release rather than a law.

**Signal:** Picks against the common default for a stated task reason, quotes both GGUF figures accurately, and knows the below-3B QAT claim is unverified.

**Follow-ups:**
- *What if RAM forces Q4_K_M?* — Then numeric capability is what you are spending; measure it before accepting `[D]`.
- *Why not Q8_0?* — Effectively lossless but roughly double Q4_K_M's size `[R]`.
- *Does the QAT claim apply here?* — Unverified; build with QAT and check whether the non-QAT baseline passes `[R]`; Q18.

**Red flags:** Defaults to Q4_K_M because it is popular; recommends Q2/Q3 for footprint; repeats the QAT-below-3B claim as fact.

---

#### T10-Q11 · GGUF and EXL2: is this a quantization choice or a format choice?
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** GGUF and EXL2 both appear next to "4-bit". Are they quantization methods? What is the actual decision?

**Model answer:** They are **containers**, not methods. The corpus is clear `[R]`: **GGUF** (llama.cpp) is for "CPU + GPU offloading", with "cross-platform (Mac, Linux, Windows), single file, highly portable" as its pros and "slower than pure GPU formats" as its con. **EXL2** (ExLlamaV2) is "GPU-only (Nvidia)" and is "the **fastest 4-bit format on Nvidia GPUs**", with "inflexible (Nvidia only)" as the cost. So the decision is portable-and-CPU-capable versus fastest-and-Nvidia-locked. The case study's surfaces split cleanly: **Surface C is GGUF**, because the laptop may be either platform and portability dominates; **Surfaces A and B are server-side and use neither**, running FP8 or AWQ safetensors under a serving engine `[D]`. The trap is conflating the container with the compression — a GGUF file has a quant level (Q4_K_M, Q5_K_M) *inside* it and EXL2 has its own bit settings, so "we use GGUF" tells you nothing about precision and "we use EXL2" tells you nothing about footprint. The interview-relevant point: the choice is made by deployment target first — what hardware, what OS, what offload — and only then by quant level.

**Signal:** Separates the container from the compression, and derives the surface split from portability rather than from a quality ranking.

**Follow-ups:**
- *Why not EXL2 on the server surfaces?* — Those use safetensors under a serving engine, not a local-inference format `[D]`.
- *What does "fastest 4-bit format" compare?* — Kernel efficiency on Nvidia, not accuracy `[R]`.
- *What does a GGUF quant level not tell you?* — Anything about accuracy on your task; Q10.

**Red flags:** Calls GGUF a quantization algorithm; assumes EXL2 is higher quality because it is faster; ignores the hardware lock.

---

### The KV cache lever

#### T10-Q12 · Why is KV quantization a separate decision?
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** You have already chosen 4-bit weights. Why touch the KV cache at all — is that not the same decision twice?

**Model answer:** No. Different scaling law, different evidence, different reversibility. **Different scaling:** weights are constant in context, KV is linear in it. At a 60k bundle the case study's 70B holds 18.75 GB of KV against 43.75 GB of 4-bit weights — a quarter of the model — and the on-device source states the general case, that in long-context work "the **KV Cache** often consumes more VRAM than the model weights themselves" `[R]`. **Different evidence:** the corpus publishes a *quality* column for weight precision and **no accuracy figure at all for KV quantization** `[R]` — VRAM numbers only. **Different reversibility:** KV dtype is an engine flag; a weight precision change is a different model artefact with a different hash. That asymmetry produces the case study's rule, worth stating verbatim: compress KV first and weights second, "**not because KV is proven safer, but because the weight risk is quantified and the KV risk is unknown, and an unknown risk you can revert is preferable to a measured one you cannot**" `[D]`. Practically, KV is where the concurrency is — moving FP8 → Int4 KV takes a 4-bit/FP8-KV configuration from 3 concurrent 60k bundles to 7 `[D]` — and it is the move you can walk back in a config change. The gate is an eval, because the evidence is missing.

**Signal:** Justifies KV-first on the *unknown-versus-measured* risk argument rather than claiming KV is safe, and knows the accuracy evidence does not exist.

**Follow-ups:**
- *What is the actual KV saving?* — Halves at FP8, quarters at Int4 `[D]`.
- *Is there any KV accuracy figure in the corpus?* — None `[R]`; say so and gate on your own eval.
- *What else changes KV cost for free?* — Fewer KV heads (MQA/GQA); Q13.

**Red flags:** Treats KV as "the same thing at a different tensor"; asserts KV quantization is lossless; changes weight and KV precision together.

---

#### T10-Q13 · Do the corpus's KV numbers hold up?
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** Here is the source's illustration: "BF16 KV Cache: 2M tokens ≈ 32GB VRAM (on 8B model). FP8/Int4 KV Cache: 2M tokens ≈ 8GB - 16GB VRAM." Is that right?

**Model answer:** It is right — and it is right *for one specific configuration*, which is the part worth deriving. Check it against the formula. 32 GB over 2M tokens is **16 KB per token**. For an 8B model at ~32 layers and head dim 128, per-token KV is `2 × layers × kv_heads × head_dim × bytes`; solving 16 KB/token at 2 bytes per element gives **one KV head** — MQA, not the GQA 8 that most 8B models actually ship `[D]`. Under GQA 8 the same 2M tokens need **250 GB**, an 8x difference `[D]`. So the figure is exact for an MQA-configured model, and the quoted "8GB – 16GB" range reconciles perfectly with that base: FP8 halves 32 → 16 GB, Int4 quarters it → 8 GB `[D]`. The number is usable once you know which configuration it belongs to. The transferable lesson is the one [T07](../01-case-studies/T07-kv-cache.md) makes: **KV cost is a function of the GQA/MQA choice at least as much as of the dtype** `[D]`. It also yields a free optimization — 8 KV heads to 4 halves KV per token, which the case study's sensitivity table notes is the same effect as a KV dtype change, for free `[D]`. The failure to name: quoting a KV number without its head configuration is how a capacity plan goes wrong by 8x.

**Signal:** Derives the implied KV head count from the figure instead of accepting or rejecting it, and reconciles the FP8/Int4 range against that base.

**Follow-ups:**
- *What if your model is GQA 8?* — 250 GB for 2M tokens; the illustration does not transfer `[D]`.
- *Which is the cheaper win, dtype or heads?* — Heads is architectural, dtype is a flag — but both halve the term `[D]`.
- *What does this do to an FP8 concurrency plan?* — It is the term that decides it; Q14.

**Red flags:** Accepts the figure at face value; rejects it as simply wrong; cannot state the per-token formula; does not distinguish MQA from GQA.

---

#### T10-Q14 · The asymmetry between weight and KV evidence
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** The corpus quantifies weight-quantization quality loss and gives no accuracy figure for KV quantization. How do you make a decision with that gap?

**Model answer:** You make it explicitly, and you make the gap part of the decision. The asymmetry is real: weight precision has a published quality column, while "the source gives VRAM figures for KV quantization and **no accuracy figure at all**" `[R]`. Three responses, in order. **First, say so.** An answer that supplies a KV accuracy number — "FP8 KV costs about 0.5%" — is inventing a figure the corpus does not carry, and that is a worse failure than admitting the gap. **Second, buy the revertibility.** Compress the thing you can undo: KV dtype is an engine flag; the weight artefact is not. Hence the runbook's tuning order — KV dtype first, then context length, then weight precision, then method, then QAT `[D]`. **Third, build the gate.** Because the evidence is missing, the eval converts an unknown into a *managed* unknown: per-field-type exact match on a held-out bundle set, with the KV change isolated so a regression is attributable. Note the failure direction — a KV regression shows up first on the *longest* contexts, so the eval must cover the actual length distribution rather than the median `[D]`. The general principle transfers well beyond quantization: when one risk is measured and the other is not, your ordering is set by which one you can walk back, not by which one you suspect is smaller.

**Signal:** Refuses to fabricate the missing number, orders the changes by revertibility, and names the eval's length-distribution requirement.

**Follow-ups:**
- *What if the eval shows KV degradation?* — Revert KV before reverting weights; KV is the reversible one `[D]`.
- *Why does aggregating over lengths hide it?* — The tail breaks first; Q22.
- *Does this survive a strict audit?* — Yes, provided the manifest records which precision produced the output; Q24.

**Red flags:** Invents a KV accuracy figure; declares KV quantization safe because "everyone does it"; changes KV and weights together.

---

#### T10-Q15 · Streaming quantization and the 4x claim
**Difficulty:** L3 · **Depth expected:** 3 min
**Question:** What is streaming quantization, and what does "4x higher concurrency" actually mean?

**Model answer:** Streaming quantization compresses the KV cache **on the fly** rather than only at write time. The corpus presents it as a capability of modern serving frameworks — vLLM, SGLang, TensorRT-LLM — where "the KV cache is compressed on-the-fly, allowing **4x higher concurrency on the same GPU**" `[R]`. Read that as a **memory** claim, and it is consistent with the arithmetic: concurrency at fixed HBM is bounded by bytes held per sequence, so quartering the KV term raises the number of resident sequences by roughly four — the same direction as Int4 KV in the case study's capacity table, where 4-bit weights plus Int4 KV reach 7 concurrent 60k bundles against 3 at FP8 KV `[D]`. Two honest caveats. **It is engine-dependent** — the corpus names frameworks that support it, so support is a deployment check rather than an assumption `[R]`. And **it carries the same evidence gap as every other KV compression**: the source gives the concurrency figure and no accuracy figure `[R]`, so on a regulated surface it is an eval-gated option, not a free win. The framing to volunteer: it is a *capacity* lever whose quality cost is unmeasured, which is exactly the profile that belongs behind the per-field eval gate rather than in a default config.

**Signal:** Reads the 4x as a memory/concurrency claim, checks it against the capacity arithmetic, and attaches the same evidence caveat as any KV compression.

**Follow-ups:**
- *Is 4x the same as Int4 KV?* — Same direction and magnitude; Int4 quarters the KV bytes `[D]`.
- *What must you verify before adopting it?* — Engine support and your own eval `[R]`.
- *Does it change the weight decision?* — No — it relieves concurrency pressure, which is the KV axis `[D]`.

**Red flags:** Treats 4x as a throughput or latency guarantee; assumes universal engine support; adopts it with no eval because "it is just KV".

---

### Tradeoffs and decisions

#### T10-Q16 · One precision or per surface?
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** Three surfaces share a base model: a long-context audited extractor, a latency-sensitive chat surface, and an offline laptop tool. One precision or three?

**Model answer:** Three, and the case study's constraint says why: "the three surfaces have quality bars that differ by more than 4x; a single config is wrong for at least two of them" `[D]`. The decisions diverge for structural reasons, not preference. **The extractor is context-bound and accuracy-bound** — a 60k bundle at BF16 KV costs 18.75 GB, so KV precision is forced, and the accuracy bar pushes weight precision up: **FP8 weights + FP8 KV**, which fits with one concurrent bundle `[D]`. **The chat surface is concurrency-bound and latency-bound** — short context, many concurrent requests, lower stakes — so **4-bit AWQ weights + FP8 KV**, reaching 3 concurrent bundles at 60k and far more at chat lengths `[D]`. **The laptop is footprint-bound** — **GGUF Q5_K_M with QAT**, under 4 GB `[D]`. Three artefacts, three evals, three manifests, and that cost is not small: the eval surface area grows with the number of precision points, which is exactly why the case study's 10x section says the *eval* becomes the constraint rather than quantization `[D]`. The rejected options: one global precision (fails the audit or the footprint), per-request dynamic switching (a reload costs the latency SLO), and per-layer mixed precision (research-grade tooling — it is what AWQ already approximates internally).

**Signal:** Derives each surface's precision from its binding constraint — context, concurrency, footprint — rather than from a quality ranking, and prices the multi-artefact cost honestly.

**Follow-ups:**
- *What breaks first at scale?* — The eval surface, not the precision matrix; Q26.
- *When does a single precision become right again?* — When the fleet collapses to one surface with one bar `[D]`.
- *What would you measure per surface?* — Its own binding metric: numeric exact match, P95 latency, footprint.

**Red flags:** Picks one precision "to keep it simple"; chooses per surface but cannot say which constraint drives which; ignores the eval-count cost.

---

#### T10-Q17 · Where should the precision decision live?
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Who decides the bit width — the researcher who quantizes the checkpoint, the platform team that serves it, or the product team that owns the surface?

**Model answer:** The decision has three components and they belong in different places, which is why it goes wrong when one team owns all of it. **The artefact** — which checkpoint, which method, which calibration set, which block size — belongs to whoever produces it, and it is pinned by hash, because a quantized model is not "the 70B"; it is "this checkpoint, this method, this calibration set, this engine" `[D]` (the discipline [T08](../01-case-studies/T08-batching-scheduling.md) applies to engine version). **The selection** — which artefact a surface runs — belongs to the serving platform as a versioned policy, because that is the only place the eval gate can be enforced as a *release* gate rather than a suggestion, and because a surface swap must force an eval rather than silently inherit one. **The accuracy bar** — how much loss this surface tolerates — belongs to the product owner, because it is a business judgement, and it is the input the other two consume. The anti-patterns are specific: a researcher shipping a quantized checkpoint straight to production (no eval gate, no manifest), and a platform team choosing a bit width by memory arithmetic alone — the 4-bit-versus-FP8 call is an *accuracy* call made with a capacity argument, which is exactly the error the case study's §5.1 catches. The discipline that makes the split work: every precision change is a measured change, and the manifest is the artefact that proves which configuration produced an output.

**Signal:** Splits artefact, selection and accuracy bar across owners, and names capacity-driven precision selection as the specific anti-pattern.

**Follow-ups:**
- *What is the enforcement mechanism?* — The eval gate as a release gate, plus the per-output manifest `[D]`.
- *What happens on a model upgrade?* — A re-quantize plus a re-eval; treat it as a new artefact `[D]`.
- *How is this like the sampling policy service?* — Same shape — versioned, replayable, centrally gated — with a checkpoint instead of a parameter set.

**Red flags:** "The ML team decides"; picks precision by memory arithmetic alone; no versioning or manifest story.

---

#### T10-Q18 · PTQ or QAT?
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Post-training quantization is hours of work; quantization-aware training is weeks. When is the training pipeline actually justified?

**Model answer:** The corpus gives one hard criterion, and it is narrow. PTQ is "quantizing a model *after* it's trained"; QAT "simulates quantization *during* the training process… The model learns to compensate for the lost precision" `[R]`. The claim that motivates QAT is precisely scoped: it is "**mandatory for models smaller than 3B parameters to remain useful at 4-bit**" `[R]`. Note two things about it — it is about *post-training* quantization of *small* models, and it is stated "without qualification or supporting measurement" `[R]`. So the case study's response is the right one to copy: **treat it as a hypothesis that gates the release.** Build the 3B with QAT; if the non-QAT baseline passes the eval anyway, the claim does not apply to this model and the cheaper path wins `[D]`. On the server surfaces PTQ is the choice because it is hours not weeks, and its documented risk — "quietly drops accuracy" `[T]` — is managed by the eval rather than by training. Two escape hatches sit between them. **QLoRA fine-tune after PTQ** recovers task accuracy, with NF4 as the QLoRA standard `[R]`, at the cost of a fine-tune to maintain per model version. **Distillation to a smaller model instead** avoids quantization as a technique entirely, at the cost of a different pipeline with different failure modes. Finally, the direction of travel in training infrastructure: one vendor reports native 8-bit and 4-bit training with the rollout in lower precision `[T]` — but that transcript renders "FP4" as "VIP 4" and the partner name is garbled, so the attribution is soft and it is a vendor claim about the vendor's own stack.

**Signal:** Reads the below-3B claim's scope precisely, converts it into a gating experiment rather than a belief, and prices both escape hatches.

**Follow-ups:**
- *How do you test the QAT requirement?* — Build both, eval both on the surface's own metric; the cheaper path wins if it passes `[D]`.
- *When do you reach for QLoRA?* — When PTQ alone misses the numeric floor `[R]`.
- *What does the corpus actually support about QAT in training infra?* — A vendor claim with an ASR-garbled attribution, cited softly `[T]`.

**Red flags:** Accepts "QAT is mandatory below 3B" as fact; uses QAT on a 70B "to be safe"; cannot name the QLoRA escape hatch.

---

#### T10-Q19 · The accuracy-versus-hardware arithmetic
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** Finance will fund either more accuracy or more hardware. How do you decide, and what does the answer look like written down?

**Model answer:** You price the accuracy in the same units as the hardware, with the assumptions visible. Take the source's optimistic 4-bit figure of **1%** `[R]` and apply it to the surface where it bites:

| Item | Value |
|---|---|
| Claims/day | 6,000 |
| Numeric-field error rate induced | 1% `[R]` |
| Claims affected | 60/day |
| Adjuster rework | 20 min @ $45/hr = $15 |
| Daily rework cost | **$900** |
| Annualised (260 working days) | **$234,000** |

Against that, the hardware alternative: a second 2-GPU node at $30,000 capex over 3 years is $10,000/yr, plus an assumed $8,000/yr for power, cooling, ops and spares — **$18,000/yr all-in**, and two extra nodes to match the 4-bit concurrency `[D]`. That is **$36,000/yr against $234,000/yr — a 6.5x ratio in favour of buying the hardware** `[D]`. The conclusion is *not* "run BF16", because 140 GB of weights do not fit a 96 GB box `[D]`. It is a four-part decision: do not spend weight precision to buy concurrency; spend KV precision instead, because it is reversible and it is where the concurrency is; buy the third node to give the accuracy-critical surface the concurrency the low-stakes surface needs; and keep 4-bit and Q5_K_M where the bars are genuinely lower `[D]`. The sensitivity to state out loud: at a 5-minute rework estimate the ratio falls to 1.6x and the decision gets close `[D]` — so the answer is a function of the rework model, and the rework model is the first input to challenge.

**Signal:** Prices accuracy in dollars with assumptions shown, and refuses the naive conclusion the arithmetic does not support.

**Follow-ups:**
- *Which input would you challenge first?* — The rework estimate; the 6.5x is sensitive to it `[D]`.
- *Why not spend it all on accuracy?* — BF16 does not fit; capacity precedes quality here `[D]`.
- *What does the corpus say about the saving side?* — 42 → 26 cost units — vivid, but it says nothing about the accuracy term `[T]`.

**Red flags:** Quotes a ratio with no arithmetic; concludes "quantize less" while ignoring the footprint; treats the vendor's cost ladder as a measured bill.

---

#### T10-Q20 · Where does quantization sit among the levers, and in what order?
**Difficulty:** L5 · **Depth expected:** 7 min
**Question:** You have batching, caching, KV compression and quantization available. Give me the ordering and defend it.

**Model answer:** Two orderings matter and they are different — the *cost* ordering and the *tuning* ordering. **Cost ordering** is the corpus's ladder: 100 units naive 16-bit → ~42 with continuous batching → **26 with 4-bit quantization** → ~11 with prompt caching `[T]`. Quantization is the third rung: a 38% cut on the batched baseline, behind batching's 58% and ahead of caching's further cut. But the ladder is an "illustrative video example, not a measured bill" `[T]`, so it orders levers rather than predicting your bill. The structural reason batching comes first is that prefill is compute-bound and therefore batches, so three concurrent requests cost far less than three sequential ones — the case study's step 2 shows the fleet is short even at 4-bit at batch 1, which means "the answer is not precision at all, it is concurrency" `[D]`. **Tuning ordering** is the runbook's: KV dtype first (where the concurrency is, and reversible), then context length, then weight precision per surface, then method (AWQ over GPTQ, Q5_K_M over Q4_K_M), then QAT only if PTQ misses the floor `[D]`. The defense of that order is revertibility and attribution: move the cheapest-to-undo, highest-leverage lever first, and never move two at once. The senior point to volunteer: quantization looks like the best lever because it is the only one that cuts memory *and* compute simultaneously `[T]` — and it is exactly that apparent free-ness that makes the eval gate non-optional.

**Signal:** Separates the cost ladder from the tuning order, knows the ladder is illustrative, and justifies the tuning order by revertibility rather than by magnitude.

**Follow-ups:**
- *Why is the ladder not a bill?* — The source presents it as illustrative cost units `[T]`.
- *Why KV before weights?* — Revertibility plus the evidence asymmetry; Q12.
- *What if batching is already optimal?* — Then quantization is the next rung, and the eval gate is its price `[D]`.

**Red flags:** Treats 100 → 11 as a measured saving; tunes weight precision before KV; changes several levers in one release.

---

#### T10-Q21 · Quantize, fine-tune, or distill?
**Difficulty:** L5 · **Depth expected:** 7 min
**Question:** Your task is narrow, high-volume and accuracy-critical. Quantization is one option. Argue for and against it against the alternatives.

**Model answer:** Four options, and the case for quantization is narrower than it first appears. **PTQ** is hours of work and no training pipeline, and its documented failure is silent: 4-bit "quietly drops accuracy" `[T]`. On a narrow numeric task, "narrow" is the problem — a 1–2% loss `[R]` concentrated on the field family that moves money is not a 1–2% business problem (Q19's arithmetic). **QAT** is the answer when accuracy must hold at low bit width and you control training; it is claimed mandatory below 3B at 4-bit `[R]`, which is the regime where the training pipeline is cheapest to justify. **QLoRA fine-tune after PTQ** is the middle path — recover task accuracy on a quantized base, with NF4 as the QLoRA standard `[R]` — at the cost of a fine-tune to maintain per model version, and with a mandatory re-eval afterwards because the fine-tune interacts with the quantized weights `[D]`. **Distillation to a smaller model instead** sidesteps quantization entirely: a smaller dense model at BF16 or FP8 may beat a large model at 4-bit on a narrow task, and it fails differently. The decision rule I would use: quantize when the accuracy cost is smaller than the hardware saving *and* the loss is uniform across the fields you care about; fine-tune when the loss is not uniform but the task is stable; distill when the task is narrow enough that capacity is not the constraint. The shared prerequisite for all three is the per-field eval, because the aggregate is the last place the loss shows.

**Signal:** Argues quantization's limits rather than its merits, and makes a per-field eval the shared prerequisite of every path.

**Follow-ups:**
- *Which is cheapest to reverse?* — Precision, if the unquantized checkpoint is kept; a fine-tune and a distillation are new artefacts `[D]`.
- *Why does "narrow" cut against quantization?* — It concentrates the loss on the fields that matter instead of averaging it away `[D]`.
- *What does the corpus support for distillation?* — It is named in the decision table as avoiding quantization; **the corpus gives no figures for it** `[R]`.

**Red flags:** Treats quantization as the only lever for a memory problem; ignores the per-field eval prerequisite; assumes a smaller model is automatically worse.

---

### Debugging and failure modes

#### T10-Q22 · Aggregate quality is fine, the audit is not
**Difficulty:** L4 · **Depth expected:** 6 min
**Question:** You quantized, your headline eval passed, and the audit failed on numeric fields. Walk me through it.

**Model answer:** This is the central risk of the topic, and the diagnosis is that you measured the wrong thing. "Quietly drops accuracy" `[T]` means the aggregate is the last place the loss shows — the model still produces fluent, plausible structured data, and the failures concentrate in a slice that averaging hides. Two corpus facts explain the *direction*. **Degradation is not uniform**: the case study's requirement section is explicit that the audit floor is per-field "because those are the ones that move money", and that "degradation is not uniform" `[D]`. **Numeric fields degrade first**: the GGUF guidance says Q2/Q3 "degrade math and reasoning by 5–10% or more" `[R]`, so the capability family that fails first is arithmetic and multi-step reasoning — which is the extraction task. The procedure: re-run the eval **sliced by field type** rather than in aggregate; if the numeric slice is the one that fell, you have confirmed the mechanism. Then attribution — which change preceded it? If weight and KV precision moved in the same release you cannot attribute it, which is exactly why the runbook insists they move separately `[D]`. The fix is to revert the most recent precision change and re-eval per field type; the structural fix is to make per-field numeric exact match the release gate. The failure to name out loud: a 0.1% aggregate move looks like noise, which is why the gate must be per field.

**Signal:** Refuses the aggregate, names non-uniformity plus the math-and-reasoning-first ordering as the mechanism, and diagnoses by slicing rather than by guessing.

**Follow-ups:**
- *Which slice first?* — Numeric exact match; it is where the floor is and where the loss concentrates.
- *Why can you not attribute it?* — Weights and KV moved together `[D]`.
- *What is the structural fix?* — Per-field eval as a release gate, not a dashboard `[D]`.

**Red flags:** Retunes the sampler or blames the prompt; treats the aggregate as ground truth; changes several variables at once.

---

#### T10-Q23 · It does not fit, at the size it was supposed to fit
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** A 70B quantized to 4-bit was sized at 35 GB and the box planned for it rejects it. What happened, and what else do you check before changing the plan?

**Model answer:** Two checks, in order. **First, the arithmetic.** 35 GB is the nominal `params × bits / 8` number; the real figure is **43.75 GB**, because sub-byte formats also store per-block scales and zeros. The case study derives the overhead from the source's own table: the 8B model shows 16, 8, **5**, **2.5** GB against formula values of 16, 8, 4, 2 — a consistent **~25% overhead on the sub-byte rows and none on the 8- and 16-bit rows** `[D]`. The planning rule is ×1.25 for sub-byte formats, and it is "the difference between a model fitting and not" `[D]`. **Second, whether the weights are even the binding term.** The next question is the KV cache, which at long context can exceed the weights — 18.75 GB for one 60k bundle at the case study's 70B configuration — and the on-device source states that in long-context work the KV cache "often consumes more VRAM than the model weights themselves" `[R]`. So before touching the quantization configuration, establish whether the OOM is **at load** (weights) or **at runtime under context** (KV). The remedies differ: at load, recompute with the 1.25 factor and either accept a coarser weight format or move nodes; at runtime, compress KV or shorten context, which is the cheaper and more reversible fix `[D]`. Third, check the reserve — the case study uses 15%, and a plan that assumed all HBM as usable is wrong by that margin alone `[D]`.

**Signal:** Gets the 1.25 factor right *and* immediately asks load-time versus runtime, because the two OOMs have different fixes.

**Follow-ups:**
- *Which is more likely?* — Load-time if the sizing came from the formula; runtime if the surface is long-context `[D]`.
- *What is the fix at load time?* — A coarser format or more memory; the reserve removes the margins `[D]`.
- *What does the corpus say about the reserve?* — 10–20% runtime overhead `[R]`; the case study uses 15% `[D]`.

**Red flags:** Blames fragmentation; applies the 1.25 factor to FP8; never asks whether the OOM is at load or under context.

---

#### T10-Q24 · Reproducibility: quantization broke it
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** The audit requires that a disputed claim can be re-run and give the same answer. Your quantized fleet cannot. What do you do?

**Model answer:** Start by naming the mechanism, because the naive answer — "pin the seed, set temperature 0" — does not survive it. The CMU lecture raises exactly this in a student exchange: asked whether quantization makes temperature-0 non-determinism worse, the answer is "**yeah, it will because you'll have more rounding errors**" `[T]`; asked whether it compounds across GPUs, the answer is that it "certainly would", because "**one GPU is more likely to have a larger difference than the threads within the GPU**" `[T]`. Kestrel's fleet runs quantized weights across **two GPUs** — the configuration the lecture identifies as compounding it `[D]`. So the reproducibility requirement is in direct tension with the fleet's existence, and the tension is invisible in every dashboard. The response has two tiers, matching the case study's §5.6. **Default: a precision manifest per output** — dtype, method, engine version, GPU count, batch shape. It does not make output reproducible, only *explainable*, and it is cheap; it is what makes the audit question answerable at all. **Escalation: a BF16 single-GPU re-run** for contested claims — genuinely reproducible, at the cost of hardware the fleet may not have and of a path that diverges from production, which is its own risk. Plus the standing discipline of pinning dtype, engine version, GPU count and batch shape, which reduces variation to a known set without achieving byte-equality. There is a security corollary worth volunteering: precision configuration is **fingerprintable from output behaviour** — the lecture's student suggests inferring "what kind of quantization" a provider uses, and the professor's reply is that it is "a good idea… I'll give you an A+ on the project if you can demonstrate that" `[T]`. The manifest you keep for the audit is also information you should not expose externally.

**Signal:** Quotes the rounding-error mechanism and the multi-GPU compounding, offers manifest-plus-escalation with the escalation's cost stated, and cannot be pushed into promising byte-equality.

**Follow-ups:**
- *Why is the manifest not sufficient?* — It explains, it does not reproduce `[D]`.
- *What does the BF16 re-run cost?* — Hardware the fleet may lack, plus path divergence; §5.6's stated exception `[D]`.
- *What is the security angle?* — Fingerprintability; do not expose the manifest externally `[T]`; Q28.

**Red flags:** Promises bit-exact reproduction from a quantized two-GPU config; thinks a seed fixes it; has no manifest tier.

---

#### T10-Q25 · Quality fell after a fine-tune, and a vendor's quant is suspect
**Difficulty:** L4 · **Depth expected:** 5 min
**Question:** Two reports, one afternoon. Your quantized model drifted after a LoRA fine-tune, and a colleague wants to adopt a vendor's pre-quantized checkpoint. Same problem or different?

**Model answer:** Different mechanisms, one shared rule. **The fine-tune case:** the fine-tune interacted with the quantized weights — the adapter was trained against quantized weights, and the merged artefact is neither the base nor the adapter. The incident runbook gives the action directly: "re-quantize from the new checkpoint rather than fine-tuning a quantized artefact" `[D]`. The related nuance is that NF4 is the QLoRA standard `[R]`, so quantized fine-tuning is a legitimate workflow — it is the *order* that matters: fine-tune then quantize, rather than quantize, fine-tune, ship. And re-eval after any fine-tune regardless, because the interaction is not predictable from the aggregate. **The vendor checkpoint:** the rule is flat and it is the same rule — a vendor's quant is not your eval `[D]`. "Always rerun your evals after quantizing" `[T]` applies to a quantization you did not perform at least as much as to one you did, and more so, because you do not know the calibration set, the block size, the method or the tooling version. The failure mode is specific: a vendor quant carries an unknown quality profile *on your task* while looking entirely legitimate — same architecture, same parameter count, plausible file size. **The shared rule:** the quantized artefact is not the model. It is a specific compression of a specific checkpoint under a specific calibration set, and every one of those is a variable you have to hold fixed and evaluate.

**Signal:** Separates the ordering error from the provenance error, and applies the same eval rule to both rather than treating the vendor case as a trust question.

**Follow-ups:**
- *What is the right order for QLoRA?* — Fine-tune then quantize; or fine-tune a quantized base with NF4 and re-eval `[R]`.
- *What must you know about a vendor quant?* — Calibration set, method, block size, tooling version — usually none of it is disclosed `[D]`.
- *What is the monitoring signal?* — Precision manifest comparison; a silent artefact swap otherwise looks like a quality change with no deploy `[D]`.

**Red flags:** Adopts the vendor quant because the vendor is reputable; blames the LoRA rank; re-evals only the fine-tune case.

---

### Scale and design

#### T10-Q26 · 10x: what changes in the precision design?
**Difficulty:** L5 · **Depth expected:** 7 min
**Question:** Your estate grows 10x. Which parts of this precision design survive, which break, and what inverts?

**Model answer:** **What survives is the arithmetic**: the 1.25x sub-byte factor, the non-uniformity of quality loss, the eval-after-quantizing rule, and the accuracy-versus-hardware comparison — the case study calls these "structural" `[D]`. **What breaks first is the eval, not the quantization.** Every precision change needs a per-field-type eval, and at 10x three surfaces become dozens of task classes, so the number of precision points grows and the eval volume grows with it; "eval throughput becomes an ops problem in its own right" `[D]`. That is the same shape as the sampling estate's constraint: the platform gate, not the mechanism, is the bottleneck. **What stops scaling is the hand-maintained table.** Per-surface precision becomes per-workload-class precision, derived from a policy — accuracy bar plus context length plus concurrency need — rather than a table someone edits. **What gets harder, not easier, is reproducibility.** The lecture's multi-GPU aggregation point scales with GPU count `[T]`, so at 10x the audit's reproducibility requirement is met by a dedicated deterministic path rather than by the production fleet `[D]` — the manifest tier stops being sufficient and the BF16 single-GPU re-run becomes a standing service. **What inverts:** today the 70B does not fit and quantization is mandatory; at 10x the fleet is larger, a BF16 tier becomes affordable for the audit path, and **precision becomes a choice again — made per path rather than per model** `[D]`. The first thing I would build at 10x is not a bigger quantization pipeline; it is the eval harness.

**Signal:** Identifies the eval as the scaling constraint rather than the quantization, and names the precision-becomes-optional inversion.

**Follow-ups:**
- *Why is the eval the constraint?* — Its volume scales with the number of precision points `[D]`.
- *What happens to determinism at 10x?* — It gets worse in proportion to fleet size; the audit path separates out `[T]`.
- *What replaces the per-surface table?* — A policy function over accuracy bar, context length and concurrency `[D]`.

**Red flags:** "Add GPUs and keep the config"; assumes quantization gets cheaper to manage at scale; no view on the eval or the deterministic path.

---

#### T10-Q27 · Design the precision plan for a new estate
**Difficulty:** L5 · **Depth expected:** 8 min
**Question:** Here is a fleet, three surfaces and a per-field accuracy floor. Give me the whole plan: what you quantize, in what order, and how you know it worked.

**Model answer:** Six steps, and the ordering is the answer. **1. Measure the baseline.** Run the per-field numeric eval on BF16 first, so the quantized delta is measured against something — otherwise you have a quality number with no referent `[D]`. **2. Size with the 1.25 factor**, not the nominal ratio, and include KV at the longest context the eval covers: at 60k tokens a single bundle's cache is 18.75 GB against 43.75 GB of 4-bit weights, so the KV term is not a rounding error `[D]`. **3. Choose weight precision per surface from the binding constraint** — FP8 for the context-bound accuracy-bound surface; 4-bit AWQ for the concurrency-bound lower-stakes one; GGUF Q5_K_M with QAT for the edge surface `[D]`. **4. Move KV first, weights second, never together**, because KV is where the concurrency is, it is the reversible lever, and moving them together destroys attribution `[D]`. **5. Gate every change on the numeric slice** of the eval, not the aggregate — degradation is non-uniform and quiet `[T]`. **6. Record the manifest** — dtype, method, engine version, GPU count, batch shape — on every audited output, and pin the quantized artefact by hash, because a quantized model is "this checkpoint, this method, this calibration set, this engine" `[D]`. Three things to state up front as constraints rather than choices: never below 4-bit for weights (2-bit is 10–15% loss `[R]`); never treat a vendor's quant as validated; and never choose precision by memory arithmetic alone — that is an accuracy decision made with a capacity argument. The single number to carry into the room is the accuracy-versus-hardware ratio, because it settles the argument that actually stalls these projects.

**Signal:** Produces an ordered plan whose steps are justified by attribution and revertibility, and states the anti-patterns as constraints rather than preferences.

**Follow-ups:**
- *Why eval the BF16 baseline first?* — Without it the quantized number has no referent `[D]`.
- *What if the floor cannot be met at 4-bit?* — QLoRA-recover, or buy the node; the 6.5x arithmetic `[D]`; Q19.
- *What is the one number to lead with?* — The accuracy-versus-hardware ratio.

**Red flags:** Jumps to a bit width with no baseline; sizes on the formula; changes weight and KV precision in one step; no manifest.

---

#### T10-Q28 · The manifest, and what it leaks
**Difficulty:** L5 · **Depth expected:** 6 min
**Question:** You record a precision manifest on every audited output. What goes in it, and what risk have you just created?

**Model answer:** **What goes in it:** the fields the case study's requirement table names — "precision + engine + GPU count + batch shape" `[D]`. Concretely: the weight dtype and method (FP8, or AWQ-4bit with its block size), the KV dtype, the engine and its version, the GPU count and class, the batch shape, and the artefact hash. Why each: dtype and method identify the compression; engine version matters because a model is not X, it is X on this engine at this version on this hardware `[D]` — the rule [T08](../01-case-studies/T08-batching-scheduling.md) applies to batching; GPU count and batch shape are there because both change the output; and the hash is what detects a silent artefact swap, which otherwise "looks like a precision change and is not" `[D]`. **What it buys:** it does not make output reproducible, only *explainable* — the distinction the case study's §5.6 table draws explicitly — and it is the cheapest artefact that makes the audit question answerable. **The risk it creates:** precision configuration is fingerprintable from behaviour. The lecture's student asks whether this could be used to "infer what kind of quantization" a provider is using, and the professor's response is that it is "a good idea… I'll give you an A+ on the project if you can demonstrate that" `[T]`. So the manifest is simultaneously the audit's evidence and a disclosure of your serving stack, and the case study's edge-case table states the mitigation: "do not expose the manifest externally" `[D]`. The design point: keep the manifest internal, expose a reference id on the response, and resolve the manifest behind an authenticated audit endpoint rather than embedding the configuration in the payload.

**Signal:** Lists the fields with a reason for each, states the explain-versus-reproduce distinction, and volunteers the disclosure risk with a concrete mitigation.

**Follow-ups:**
- *Does the manifest make output reproducible?* — No — explainable only; the BF16 single-GPU path is the reproducible one `[D]`.
- *Which field detects a silent swap?* — The artefact hash `[D]`.
- *Why is engine version in a precision manifest?* — It changes kernel paths, and therefore output `[D]`.

**Red flags:** Records only the bit width; believes the manifest gives reproducibility; exposes the manifest in the public response.

---

## Whiteboard exercises

### Exercise 1 — Size a quantized fleet against a per-field accuracy floor
**Prompt.** A 70B-class model must be served from two 48 GB GPUs (96 GB total, 15% runtime reserve). Two surfaces: **(a)** a 60,000-token document surface with an audited *numeric-field* exact-match floor and a requirement of at least 3 in-flight documents; **(b)** an interactive short-context Q&A surface at P95 < 2 s. Size the weight and KV precision for both, show the footprint arithmetic, and state the single change you would make first and why.

**What to produce.** The per-surface precision configuration; the arithmetic with the sub-byte factor and the reserve applied; the concurrency achieved per configuration; and the first change, with its justification.

**Expected whiteboard.**

```
Assumptions [D]
  sub-byte overhead      x1.25            (from the source's own 8B table)
  usable HBM             96 x 0.85 = 81.6 GB
  KV per token (70B, 80L, 8 KVH, d128, BF16) = 0.3125 MB
  KV per 60k bundle      = 18.75 GB   (FP8: 9.375   Int4: 4.69)

Configuration                Weights    KV/bundle   Total, 1    Concurrency
  BF16 weights             140 GB     18.75 GB    158.75 GB   DOES NOT FIT
  FP8 w + BF16 KV           70 GB     18.75 GB     88.75 GB   none
  FP8 w + FP8 KV            70 GB      9.375 GB    79.4 GB    1
  4-bit w + BF16 KV         43.75 GB  18.75 GB     62.5 GB    2 (razorthin)
  4-bit w + FP8 KV          43.75 GB   9.375 GB    53.1 GB   3
  4-bit w + Int4 KV         43.75 GB   4.69 GB     48.4 GB   7

Surface (a): FP8 weights + FP8 KV  -> 1 bundle  -> DOES NOT MEET the >=3 requirement
Surface (b): 4-bit AWQ + FP8 KV    -> 3 bundles -> meets it

First change: KV dtype, not weight dtype.  [D]
  rationale: KV is where the concurrency is (2 -> 3 -> 7) and it is a flag,
             revertible without changing the model artefact; the weight risk is
             the documented one, the KV risk is the unmeasured one.
```

**Grading rubric.**
- Applies **×1.25 to the sub-byte rows only** and shows 43.75 GB, not 35 GB — and does not apply the factor to FP8 or BF16.
- Reproduces the 18.75 GB per-bundle KV figure and uses it as the term that decides concurrency, not as a footnote.
- Notices that Surface (a) at FP8 + FP8 KV cannot meet a 3-bundle concurrency requirement, and says so rather than hiding it — the honest answer is the third node or a KV-precision move (Q19).
- Names **KV dtype** as the first change and justifies it by revertibility plus the evidence asymmetry (documented weight risk, undocumented KV risk).

### Exercise 2 — Diagnose a silent accuracy regression after a precision change
**Prompt.** Two weeks ago the extraction surface moved from FP8 to 4-bit AWQ weights *and* from BF16 to FP8 KV in the same release, to raise concurrency. The headline eval is unchanged at 96.4% aggregate accuracy. The audit has now failed on numeric fields, and the failure rate is 3.1% on the longest decile of documents and 0.2% on the shortest. Reproduce the diagnosis, name the mechanism, and give the fix and the structural change.

**What to produce.** The ranked hypothesis list with the measurement that kills each; the mechanism; the attribution problem; the fix with its rollback; and the structural change that prevents a recurrence.

**Expected whiteboard.**

```
Hypotheses (ranked by likelihood x cheapness to test)
  1. Weight-quantization loss concentrated on numeric fields   test: numeric slice, 4-bit vs FP8 artefact, same KV
  2. KV-quantization loss at long context                      test: error rate vs document length, KV dtype held fixed
  3. Calibration set unrepresentative of real bundles          test: re-run AWQ with in-domain calibration
  4. Something else (prompt, template, engine)                 test: manifest diff vs the previous release

The length signal is the tell [D]:
  error 3.1% on the longest decile, 0.2% on the shortest
     -> error scales with CONTEXT, not with document content
     -> the KV term is the suspect, because KV cost scales with context
        and weight-quantization error does not

The attribution problem [D]:
  weights AND KV moved in one release -> neither can be blamed from the data
  -> this is the runbook's "never move two at once" rule

Fix:        revert KV first (BF16 KV), keep 4-bit weights, re-eval the numeric slice
Rollback:   engine config flag; no new artefact, no re-quantize
Structural: (i) move weight and KV precision in separate releases
            (ii) numeric exact match as a per-field RELEASE GATE
            (iii) eval set must cover the real length distribution, not the median
```

**Grading rubric.**
- Reads the **length gradient** as the decisive evidence and argues from it that KV, not weights, is the suspect — a weight-quantization error would not scale with context `[D]`.
- Names the attribution failure explicitly: two precision variables moved in one release, so the data cannot separate them.
- Fixes it by reverting **KV** — the reversible lever — rather than reverting the weight artefact.
- Gives three structural changes, including a per-field numeric release gate and an eval set covering the length distribution.

### Exercise 3 — Write the precision policy for a multi-surface estate
**Prompt.** Four surfaces: **(a)** an audited 60k-token extractor with a numeric-field floor; **(b)** a bursty interactive Q&A surface; **(c)** a 3B offline laptop tool that reconciles figures; **(d)** a model-quality dashboard that must measure the model as it is. Write the precision policy: one row per surface, with the artefact, the KV dtype, the rejected alternative, and the signal that tells you the choice was wrong.

**What to produce.** The policy table, the artefact/hash and manifest rules that apply to all four, and the ordering in which you would make the changes on a first deployment.

**Expected whiteboard.**

| Surface | Artefact | KV dtype | Reject | Wrong-if signal |
|---|---|---|---|---|
| (a) Extractor | FP8 weights | FP8 KV | 4-bit AWQ — 1–2% loss is above the floor | numeric-field exact match < floor |
| (b) Q&A | 4-bit AWQ, in-domain calibration | FP8 KV | FP8 — 2x compression is not enough at this concurrency | P95 > 2 s or OOM under burst |
| (c) Laptop 3B | GGUF Q5_K_M, QAT-trained | n/a | Q4_K_M — compromises code and reasoning; Q2/Q3 — 5–10%+ on math | on-device numeric eval < threshold |
| (d) Dashboard | the production artefacts, **unmodified** | production dtype | any quantized-to-be-cheaper variant | reported quality diverges from the untruncated reference |

```
Rules that apply to all four [D]:
  - quantize weights and KV SEPARATELY, KV first     (attribution + revertibility)
  - size with x1.25 for sub-byte formats, plus a 15% reserve
  - never below 4-bit for weights  (2-bit is 10-15% loss)
  - every artefact pinned by hash; manifest on every audited output
  - every precision change is an eval-gated release, per FIELD TYPE
  - vendor-provided quants are re-evaluated, not trusted  [T]

Order on first deployment:
  BF16 baseline eval -> KV dtype -> weight precision per surface -> method -> QAT only if needed
```

**Grading rubric.**
- Gives surface (a) **FP8**, not 4-bit, on the grounds that the 1–2% figure is above a per-field numeric floor — the case study's §5.1 exception, not a preference.
- Chooses **Q5_K_M over Q4_K_M** for (c) with the code-and-reasoning reason, and rejects Q2/Q3 on the 5–10% math figure.
- Leaves (d) **unquantized** and explains that a measurement surface must measure the model rather than a compressed version of it.
- States the shared rules: separate weight/KV changes with KV first, the 1.25 factor, the hash pin plus manifest, and the per-field-type eval gate.

## Sources

- `refs/ai-system-design-guide-main/ai-system-design-guide-main/03-training-and-adaptation/07-quantization-deep-dive.md` — the BF16/FP8/4-bit/2-bit tradeoff table with sizes, quality loss and GPU compatibility (the basis of Q2, Q3 and the 1.25 factor); the NF4, AWQ and FP8 method descriptions including the 1% salient-weight mechanism and the "better perplexity than GPTQ, especially for smaller models or more aggressive quantization (e.g., 3-bit)" claim; the NF4 equal-bin interview answer; GPTQ as layer-wise MSE minimisation; the GGUF and EXL2 format comparison; the 2M-token KV illustration and streaming quantization's 4x concurrency; and QAT with the below-3B claim.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/09-on-device-and-edge-deployment.md` — the `params × bits / 8` VRAM rule of thumb, the "weights, then KV, plus 10–20% runtime overhead" sizing rule, and the GGUF quant-level quality figures (Q4_K_M 1–3%, Q5_K_M under ~1% and better on code and reasoning, Q8_0 lossless at half, Q2/Q3 5–10%+ on math and reasoning).
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/01-inference-fundamentals.md` — the FP8 description as the native format on H100/B200 and its "negligible (<0.1%) accuracy loss" figure, which **conflicts with the quantization chapter's "< 1%"** — the internal inconsistency Q5 must flag rather than resolve.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/07-cost-optimization-playbook.md` — quantization's place in the cost ladder and VRAM as a cost driver; also the **DeepSeek V4 Flash/Pro pricing figures**, a corpus claim carried with its own "verify on the pricing page before committing" caveat and used here only as an example of a claim requiring verification.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/04-finops-and-token-economics.md` — the FinOps framing behind the build-versus-buy comparison in Q19.
- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — the 100 → 42 → 26 → 11 cost ladder with 4-bit quantization as the third rung; "the memory a request occupies and the compute each token costs"; "always rerun your evals after quantizing" and the "quietly drops accuracy" warning; AWQ and GPTQ named as the tools.
- `refs/CMU_Inference_Algorithms_for_Language_Modeling_Fall_2025_transcripts_2/CMU_LLM_Inference_2_Probability_Review_and_Code_Examples.txt` — the instructor's answer that quantization worsens temperature-0 non-determinism through rounding errors, the multi-GPU aggregation compounding ("one GPU is more likely to have a larger difference than the threads within the GPU"), and the student's quantization-fingerprinting suggestion with the professor's reply. **This is lecture 2, in the `_2` directory.**
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt` — low-precision training with quantization-aware training, native 8-bit and 4-bit training, and the FP4 rollout claim. **The transcript is ASR-garbled at this passage** ("VIP 4" for FP4; the partner name unusable), cited as a soft attribution and as a vendor claim.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt` — quantization as one of AITER's optimized operator classes on AMD, enabled by a single environment variable. **The transcript renders both "AITER" and the variable name garbled**, and no figure is attributed to it; cited only as evidence that the vendor/corpus claim exists and needs verification.
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` — topic inventory covering quantization. **This file is a table of contents and contains no figures**; cited for coverage only.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/16-case-studies/01-enterprise-rag.md` — house style reference.
