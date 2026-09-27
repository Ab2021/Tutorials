# T10 — Quantization: high-level design

> `T10` · **Transcript coverage:** partial · [LLD](LLD.md) · [Cheat sheet](../../00-cheat-sheets/T10-quantization.md) · [Case study](../../01-case-studies/T10-quantization.md) · [Interview bank](../../02-interview-questions/T10-quantization.md) · [Runnable core](run.py) · [Production](production/README.md) · [Sequences](docs/SEQUENCES.md)

Quantization is presented as a single trade — fewer bits, less memory, small quality loss — and it
is four separable things: an **accounting chain** with no choices in it, a set of **quantizers**
whose error depends far more on the tensor than on the bit width, a **lever-selection problem**
where weights and KV answer different questions, and an **evaluation problem** that has no formula.
The corpus's own framing is the honest one: 4-bit is "nearly free on many models, but on some it
quietly drops accuracy. So always rerun your evals after quantizing" `[T]`.

`[T]` transcript · `[R]` repo · `[D]` derived. Every modelled number is reproducible from
[`run.py`](run.py); every corpus number is cited to its source.

---

## 1. System context

Quantization is not a serving component — it is a **property of every tensor the engine touches**,
decided partly offline (weights) and partly at configuration time (KV cache). It therefore appears
in three places in the stack, and they are usually owned by different people.

| Where | What it decides | Who usually owns it | Reversible at runtime? |
|---|---|---|---|
| **Checkpoint** | weight precision, quantizer, group size | the model team | no — it is the artefact |
| **Engine config** | KV cache precision, per-layer scales | the serving team | yes, at restart |
| **Hardware** | which precisions are native (FP8 on H100/B200/4090 `[R]`) | procurement | no |

```mermaid
graph TB
    subgraph offline["Offline — the artefact"]
        CK[Checkpoint<br/>AWQ / GPTQ / NF4 / FP8] --> W[Weight precision<br/>+ group size + metadata]
    end

    subgraph runtime["Runtime — the engine config"]
        KVQ[KV cache precision<br/>fp8 / int4 / streaming]
        QSL[Serving layer<br/>vLLM / TGI / SGLang / TensorRT-LLM]
    end

    subgraph hw["Hardware — fixed"]
        FP8N[Native FP8?]
        FP4N[Blackwell Ultra FP4 kernels]
    end

    W --> BUD[HBM budget]
    KVQ --> BUD
    BUD --> CONC[Concurrency ceiling<br/>= free / (kv_per_token x ctx)]
    FP8N --> QSL
    FP4N --> QSL
    QSL --> LAT[Latency: bytes moved per token]
    CONC --> THR[Throughput]
    LAT --> THR

    W --> QUAL[Output quality]
    KVQ --> QUAL
    QUAL --> EVAL[Evals — the only real check]
    EVAL -->|fails on the worst slice| REJ[Reject the merge]
```

**Two edges carry the design.** `HBM budget → Concurrency` is where quantization converts into
capacity, and it is a *division*, which is why the two levers multiply rather than add.
`Quality → Evals` is where it converts into risk, and it is the edge with no formula behind it —
§8 shows why there cannot be one.

---

## 2. The accounting chain — bits are not bytes

The one part of this topic with no choices in it, and the part most often got wrong in both
directions.

```
weight_bytes = params x effective_bits / 8
effective_bits(bits, group) = bits + (scale_bits + zero_point_bits) / group
kv_bytes_per_token = 2 x layers x kv_heads x head_dim x effective_bits / 8
concurrency = (usable_hbm - weight_bytes) / (kv_bytes_per_token x ctx_len)
```

**The metadata is real.** A 4-bit weight with a per-group fp16 scale costs
`4 + 16/group_size` bits, not 4. At group 128 that is **4.125**; at group 64, **4.25**.

| precision | bits | effective bits | weights (8B) | concurrency @ 2k |
|---|---|---|---|---|
| bf16 | 16 | 16.000 | 16.00 GB | 208.6 |
| fp8 | 8 | 8.125 | 8.12 GB | 238.4 |
| int8 | 8 | 8.000 | 8.00 GB | 238.4 |
| int4 (g128) | 4 | **4.125** | 4.12 GB | 253.3 |
| nf4 (g64) | 4 | **4.250** | 4.25 GB | 253.3 |
| int2 (g32) | 2 | 2.500 | 2.50 GB | 260.8 |

Measured for an 8B on one 80 GB part at util 0.90, GQA 8 KV heads, ctx 2048.

**Two consequences.** A "4-bit model" is *not* a quarter of bf16 — it is 4.125/16 = 25.8%. And a
smaller group quantizes better while costing more storage, which is the only genuine trade in the
whole method (§4).

**GQA is already a quantization.** `kv_heads < heads` divides the KV size *before* any precision
change, and the corpus lists both under the same heading `[R]`. They multiply, and they are the two
cheapest levers on the largest term in long-context serving.

---

## 3. Which lever — weights or KV?

The finding this blueprint leads with, and it is not the one the field's emphasis suggests. The two
levers act on different terms of the same division:

```
concurrency = (usable - W) / (kv_per_token x ctx)
                 ^^^^          ^^^^^^^^^^^^^^^
              weights: the      KV: the denominator
              subtrahend
```

Measured — both multipliers are **constant in context**, because context cancels out of the ratio:

| ctx | bf16/bf16 | int4 weights | int4 KV | int4 both | weight gain | KV gain |
|---|---|---|---|---|---|---|
| 2048 | 208.6 | 252.9 | 834.5 | 1011.4 | 1.21× | 4.00× |
| 8192 | 52.2 | 63.2 | 208.6 | 252.9 | 1.21× | 4.00× |
| 32768 | 13.0 | 15.8 | 52.2 | 63.2 | 1.21× | 4.00× |
| 131072 | 3.3 | 4.0 | 13.0 | 15.8 | 1.21× | 4.00× |

**A degenerate metric to avoid.** The obvious framing — "what fraction of memory is KV?" — is
useless: concurrency absorbs whatever is left, so `kv_per_seq × sequences ≡ usable − weights`, and
KV is 100% of the footprint at every context length. The informative quantities are the
**per-sequence** comparison and the **marginal** cost:

| ctx | KV per sequence | vs weights | concurrency |
|---|---|---|---|
| 2048 | 0.27 GB | 2% | 208.6 |
| 8192 | 1.07 GB | 7% | 52.2 |
| 32768 | 4.29 GB | 27% | 13.0 |
| 131072 | 17.18 GB | **107%** | 3.3 |

One sequence's KV overtakes the weights at **~122k tokens**, much further out than the "long
context" intuition. **The number that matters is the marginal one: each additional 1k tokens of
context costs 0.131 GB *per concurrent sequence*, so at 200 sequences that is 26.2 GB per 1k tokens
— more than the weights, at any context.** Context and concurrency are the same budget spent two
ways, and that is the sentence to carry into a capacity review.

### 3.1 The decision table

| Choice | Pros | Cons | Use when | Exception |
|---|---|---|---|---|
| **KV quant** (fp8/int4) | a consistent ~4× concurrency at any context; the only lever that makes very long context serveable | degrades attention, so it is invisible to short-context evals | context ≥ 8k, or the service is concurrency-limited | the model is already attention-quality-limited |
| **Weight quant** | shrinks latency (bytes per token) and is the *fit* lever | only ~1.21× concurrency on a large part | the model does not fit, or decode latency is the SLO | the part is small enough that weights are not the constraint |
| **Both** | the effects multiply — 4.86× here | two independent quality risks to evaluate | long context + tight memory | evaluate them separately; a combined eval hides which one hurt |
| **Neither** | no quality risk; no re-validation cost | a 70B in bf16 does not fit on one 80 GB part at all | the model fits and quality is untouchable | — |
| **Weight-free, KV-only** | attacks the term that grows with context | leaves decode latency untouched | long-context RAG with a small model | — |

**The two levers answer different questions.** KV quantization is a **concurrency** lever, worth a
consistent 4×. Weight quantization is largely a **fitting** lever — only 1.21× on an 80 GB part,
because the weights are a small share of the budget there:

| configuration | bf16 | int4 | verdict |
|---|---|---|---|
| 8B on 24 GB | 10.4 seq | 32.5 seq | fits either way |
| **70B on 80 GB** | **0.0 seq** | **66.9 seq** | int4 is the difference between a deployment and none |

For a 70B in bf16 on one 80 GB card, weight quantization is not an optimization. **No amount of KV
quantization substitutes for it**, and a plan that says "we'll just quantize the KV cache" is a plan
that does not fit.

---

## 4. Where the error comes from — granularity and outliers

The bit width is the least interesting variable. What decides quality is **where the scale is
fitted**, and the fact that weight matrices have heavy-tailed channel magnitudes.

Measured on a 96×64 tensor, zero-centred normal with 4% of channels at 12× magnitude, 4 bits:

| method | effective bits | SNR | worst channel | bytes |
|---|---|---|---|---|
| per-tensor | 4.000 | 8.48 dB | **1.0000** | 3074 |
| per-channel | 4.000 | 19.87 dB | 0.1714 | 3264 |
| group 32 | 4.500 | 20.56 dB | 0.1366 | 3456 |
| group 16 | 5.000 | 21.97 dB | 0.1088 | 3840 |
| group 8 | 6.000 | 23.23 dB | 0.0948 | 4608 |
| nf4 group 64 | 4.250 | 20.82 dB | 0.1174 | 3264 |

**`effective bits` and `SNR` move in opposite directions — that is the entire trade.** Group 8 beats
group 32 on quality and pays for it in bytes. The right group size is wherever the marginal dB stops
being worth the marginal byte, and it is workload-specific.

**But the `worst channel` column decides whether the model still works.** Per-tensor 4-bit destroys
a channel outright: one outlier row sets the scale for everything, so every small weight rounds to
zero. A worst-channel relative error of **1.0000** means an output that is pure error. Aggregate SNR
looks survivable at 8.48 dB; the worst channel does not. **A mean-only report hides exactly the
failure that matters** — the same tail-over-mean discipline as T08's goodput and T09's p99.

### 4.1 SNR's blind spot

The sharpest finding in this blueprint. Sweeping outlier severity at a **fixed 4 bits**:

| severity | per-tensor SNR | per-channel | group 32 | worst channel (per-tensor) |
|---|---|---|---|---|
| 1.0 | 14.76 | 19.28 | 20.22 | 0.2333 |
| 4.0 | 9.16 | 19.53 | 20.36 | 0.5760 |
| 8.0 | 7.25 | 19.77 | 20.50 | 0.9277 |
| 12.0 | **8.48** | 19.87 | 20.56 | 1.0000 |
| 24.0 | **12.93** | 19.95 | 20.60 | 1.0000 |
| 48.0 | **16.61** | 19.97 | 20.61 | 1.0000 |

**Per-tensor SNR is non-monotone — it improves as the tensor gets more extreme.** Read alone, that
says the quantization got better. It did not: SNR is signal-energy over noise-energy, and the
outlier channels contribute to both. At high severity the signal grows faster than the error, so the
ratio rises *while* the small weights are being destroyed. The worst-channel column is monotone
(0.23 → 1.00) and is the truth.

**This is the same pattern as T08 and T09** — a metric that improves while the thing being measured
collapses. There it was a p99 regression hidden by a better mean; here it is a destroyed feature
hidden by a better SNR.

**And the flatness is the design criterion.** Per-channel (19.28 → 19.97) and group-wise (20.22 →
20.61) barely move across a 48× change in severity, precisely because each scale is fitted locally.
**That flatness, not the absolute SNR, is what makes them the default** — a per-tensor scale's
quality is a property of the *model*, so it cannot be validated once and reused.

### 4.2 The granularity decision table

| Choice | Pros | Cons | Use when | Exception |
|---|---|---|---|---|
| per-tensor | zero metadata; simplest kernel | destroyed by one outlier channel | never for weights | activation tensors, where outliers are less severe |
| per-channel | flat across severity; ~2 bytes/channel | scale per row only | **the default for INT8 weights** | very small tensors |
| group-wise (16–128) | best error/storage ratio; the 4-bit standard | metadata overhead; kernel support varies | **the default for 4-bit weights** | tiny groups waste more on scales than they save in error |
| codebook (NF4) | exploits the weight distribution (§5) | assumes normality; lookup cost | 4-bit fine-tuning, QLoRA `[R]` | weights that are not normally distributed |
| mixed precision (AWQ) | spends bytes where the error is (§6) | needs a calibration set | aggressive bit widths, small models | a quantizer that is already flat |

---

## 5. NF4 — the grid shape is a modelling assumption

Both NF4 and INT4 are 4 bits with 16 levels. They are not equivalent, and the reason is a bet about
the weight distribution.

```
NF4  (normal quantiles): -1.000 -0.696 -0.525 -0.395 -0.284 -0.185 -0.091  0.000 ...
INT4 (uniform grid)    : -1.000 -0.867 -0.733 -0.600 -0.467 -0.333 -0.200 -0.067 ...
```

NF4's levels are **dense near zero and sparse in the tails**. LLM weights are zero-centred normal,
so most of the mass is near zero — exactly where NF4 has the most resolution and INT4 wastes it.
Measured at group 64: **NF4 20.82 dB with a worst channel of 0.1174, against uniform INT4's
19.87 dB and 0.1714.**

The corpus states the mechanism precisely `[R]`: each NF4 bin "contains an equal number of values
from the normal distribution. This prevents 'clustering' of weights and ensures that the model
preserves as much information (entropy) as possible". The practical effect is a grid shaped like the
data.

**The honest caveat: NF4 is a bet, and on weights that are not normally distributed it narrows or
vanishes.** It is a good bet for transformer weights — which is why it is the QLoRA standard `[R]`
— but it is a bet, not a theorem.

---

## 6. AWQ — mixed precision as a policy, not a format

AWQ is not a quantizer. It is a *policy on top of* one: quantize everything, then restore the most
salient channels at full precision. The corpus states the mechanism exactly `[R]`: AWQ "identifies
which weights are the most 'salient' based on the actual activation values seen during a small
calibration run. By preserving only these important weights (usually 1%) in higher precision and
quantizing the rest, AWQ achieves better perplexity than GPTQ".

Measured, 4-bit group 32, protecting the top-N weight-L1 channels:

| salient fraction | channels | SNR | gain | storage cost |
|---|---|---|---|---|
| 0.00 | 0 | 20.56 dB | — | 0% |
| **0.01** | 1 | **21.67 dB** | **+1.12** | **3%** |
| 0.02 | 2 | 23.03 dB | +2.47 | 6% |
| 0.05 | 5 | 28.73 dB | +8.17 | 15% |
| 0.10 | 10 | 29.08 dB | +8.52 | 30% |
| 0.25 | 24 | 29.90 dB | +9.34 | 75% |

**The first 1% recovers a disproportionate share, and 25% is barely better than 10%.** That is the
empirical justification for the corpus's stated 1%, and the reason AWQ beats GPTQ at aggressive
widths: it spends a few percent of storage exactly where the error is concentrated.

**The storage cost is the other half of the trade.** Protecting 1% at 16 bits against 4 bits
elsewhere costs `0.01 × (16−4)/4 = 3%` more bytes. The ratio of dB recovered to bytes spent is what
makes this worth doing.

**The caveat, stated plainly:** real salience comes from **activation** statistics on a calibration
set, which the corpus states explicitly `[R]`. The blueprint's offline stand-in is the weight row's
L1 norm — correlated with activation salience, not equal to it. A deployment that uses a calibration
set is doing something this model approximates rather than reproduces.

---

## 7. KV cache quantization — a different kind of damage

The KV cache is quantized at *runtime*, by a config flag, and its failure mode is different from
weight quantization's in a way that matters operationally.

| | Weight quantization | KV cache quantization |
|---|---|---|
| decided | offline, in the artefact | at engine config time |
| reversible | no | at restart |
| damages | the model's weights | the **attention** over a sequence |
| visible in evals | on any benchmark | **only on long-context benchmarks** |
| scales with | model size | context length × concurrency |

**The last two rows are the operational trap.** A model can score identically on a short-context
benchmark and fail at 64k, because KV-quantization errors accumulate over positions the benchmark
never exercises. The corpus notes the payoff is real — KV quantization enables "4x higher
concurrency on the same GPU" `[R]` — and this blueprint's arithmetic reproduces it exactly at 4.00×
(§3). What the arithmetic cannot tell you is whether *your* model survives it at *your* context
length, and that is an eval question with no shortcut.

**Streaming quantization** — compressing the cache on the fly — is now supported in vLLM, SGLang and
TensorRT-LLM `[R]`. It is the same trade with a different implementation: the precision is a config
value, and the eval requirement is identical.

---

## 8. The cost ladder — four rungs, four terms

The corpus's headline `[T]`:

> "Start with naive 16-bit serving one request at a time as 100 cost units. Turn on continuous
> batching and you are near **42**. Quantize to 4-bit and you reach **26**. Cache the stable system
> prompt and you land around **11**."

Reproduced from the mechanism (4000-token prompt, 100-token output, 8B):

| rung | model units | corpus | delta | prefill share | implies |
|---|---|---|---|---|---|
| naive bf16 | 100.0 | 100 | +0.0 | 12% | batch 1 |
| continuous batching | 35.9 | 42 | −6.1 | 32% | effective batch ~4 |
| 4-bit weights | 20.0 | 26 | −6.0 | 58% | 4-bit weights, bf16 KV, group 128 |
| prefix cached | 9.0 | 11 | −2.0 | 6% | cache hit ~95% |

**This is a mechanism, not a fit, and the residual column is reported rather than hidden.** The
order and the shape are right; two rungs land within a few units. The 4-bit rung is the least
accurate — the model credits weight quantization a little more than the corpus's composite does.
Claiming a perfect reproduction would be the dishonest move here.

**The rungs act on different terms of one equation:**

```
cost = output_len x (W/batch + ctx x kv_per_token) / W_ref  +  prompt_len x (1 - cache_hit) x prefill
                    ^^^^^^^^   ^^^^^^^^^^^^^^^^^^              ^^^^^^^^^^^^^^^^^^^^
                    batching    KV quantization                prefix caching
                    and weight quantization                (removes prefill entirely)
```

**The batching rung's implied batch is single digits.** That is the most useful number in the table:
the corpus's "you are near 42" is already mostly captured at a depth a modest deployment reaches —
the batching win is not something you must reach a large batch to collect.

**The 4-bit rung is the smallest single step**, at a factor of 0.62 rather than the 0.25 the words
suggest. Two reasons, both real: the weights are only part of what is read per token (the KV term is
untouched), and group-wise 4-bit carries scale overhead, so the true ratio is 4.125/16, not 4/16.

**The caching rung is the largest — and it is entirely a property of the workload's shape:**

| prompt | output | ratio | prefill share | cache step saves |
|---|---|---|---|---|
| 100 | 500 | 0.2 | 1% | **1%** |
| 500 | 500 | 1.0 | 4% | 4% |
| 4000 | 100 | 40.0 | 60% | **57%** |
| 16000 | 100 | 160.0 | 82% | **78%** |

The corpus's own caveat, quantified `[T]`: "caching only helps a stable prefix. Cache something
that changes each call and you gain nothing." **The same cache implementation is worth 57% on one
workload and 1% on another**, and the difference is the prompt-to-output ratio, not the code.

**Which is the transferable lesson of the ladder: which rung is worth climbing is a property of the
workload's shape.** A ladder measured on one shape does not transfer to another.

---

## 9. Capacity and cost model — worked

**Concurrency.** For an 8B on one 80 GB part at util 0.90, ctx 8192, int4 weights and int4 KV:

```
  usable            = 80 x 0.90                      = 72.00 GB
  weights (4.125 b) = 8e9 x 4.125 / 8                =  4.12 GB
  kv_per_token      = 2 x 32 x 8 x 128 x 4.125/8     =  3.38e4 B
  kv per sequence   = 3.38e4 x 8192                  =  0.277 GB
  concurrency       = (72.00 - 4.12) / 0.277         =  245 sequences

  against bf16 weights and bf16 KV:  52 sequences     -- 4.7x
```

**Cost per million output tokens.** The headline number teams quote, and it is dominated by a term
nobody tunes:

```
  requests/hour  = 10,000,  prompt 4000, output 100 -> 41M tokens moved/hour
  at 3,000 tokens/s  -> 3.80 GPU-hours  -> $9.50 at $2.50/GPU-hour
  cost per million OUTPUT tokens = 9.50 / (10,000 x 100 / 1e6) = $9.50
  the same service with a 100-token prompt      = $3.40 per million output tokens
```

**The prompt/output ratio is a bigger cost driver than the quantization.** Two services with
identical per-token pricing differ by ~3× per *useful* token because one re-reads a 4000-token
system prompt on every call. That is T19's arithmetic, and it is worth stating in T10 because the
fix — prefix caching — is the fourth rung of the ladder, not a quantization at all.

**Which numbers are corpus and which are modelled.** The ladder's four figures, the 4× concurrency
claim, the quality-loss column, and the 1%-salient AWQ mechanism are the corpus's `[T]`/`[R]`. Every
SNR, error, byte count and cost unit here is **this blueprint's model**, reproducible from `run.py`,
and is not a measurement of any real model. No corpus figure is asserted as an output of the
simulator.

---

## 10. Deployment topology — where quantization is decided

| Stage | Decision | Tooling | Reversible |
|---|---|---|---|
| **Checkpoint** | quantizer, bits, group size | AWQ / GPTQ `[T]`, NF4 for fine-tuning `[R]` | no — re-quantize and re-eval |
| **Engine launch** | KV precision, per-layer overrides | vLLM / SGLang / TensorRT-LLM flags, `--kv-cache-dtype` | at restart |
| **Runtime** | streaming quantization, dynamic scales | engine internals `[R]` | yes, but it is a code path |
| **Hardware** | FP8 native, FP4 kernels | H100/B200/4090 for FP8; Blackwell Ultra for FP4 `[R]` | no |

**The pipeline order matters and it is the wrong way round in most teams.** The KV decision is made
at engine config time — which is *after* the checkpoint is chosen — but it is the larger lever for
concurrency (§3). A team that picks a 4-bit checkpoint and runs bf16 KV has spent its effort on the
1.21× and left the 4× on the table.

**The evaluation must follow the same order.** A weight-quantization change is visible on any
benchmark; a KV-quantization change is visible only on long-context ones. Evaluating the combined
configuration with a short-context suite tests neither.

---

## 11. Failure domains

| Failure | Symptom | Silent? | Detection | Mitigation |
|---|---|---|---|---|
| **SNR looks fine while a channel is destroyed** | quality loss with no metric movement | **yes** | worst-channel error, not aggregate SNR (§4.1) | per-channel or group-wise scales |
| per-tensor scale on an outlier-heavy tensor | catastrophic, model-specific | no, but attributed to the bit width | worst-channel check per layer | never ship per-tensor for weights |
| **quality regression only on some models** | "4-bit was fine in our test" | **yes** | evals on the *worst* slice, per model (§12) | eval gate; per-model validation |
| **KV quantization invisible to short-context evals** | degrades only at long context | **yes** | long-context eval at the deployed ctx | an eval at production context length |
| group size too small | storage grows past the nominal bit width | **yes** | `effective_bits` accounting (§2) | budget the metadata, not the nominal bits |
| **weights quantized, KV not** | only 1.21× of an available 4.86× collected | **yes** | the concurrency arithmetic | quantize both, or state which you chose |
| model does not fit after "quantizing for memory" | OOM at load | no | fit check before deploy | weight quantization is the fit lever (§3) |
| FP8 requested on hardware without native support | falls back, slower than bf16 | **yes** | hardware capability check at boot | check the capability matrix `[R]`; use EXL2/AWQ kernels on Nvidia, GGUF off-GPU |
| prefix cache configured on a changing prefix | no gain, cache churn | **yes** | cache hit rate × prompt/output ratio (§8) | cache only a stable prefix `[T]` |

**Six of the nine are silent**, and their silence has a common shape: the system runs correctly and
simply does not deliver the benefit it was configured for, or delivers it and damages something
nobody measured. The two dashboards that discriminate them are **worst-channel (or worst-slice)
quality** and **the concurrency arithmetic broken out by lever** — the same tail-over-mean
discipline this knowledge base applies in T08 and T09.

---

## 12. The eval gate — turning advice into a merge check

The corpus's rule is "always rerun your evals after quantizing" `[T]`. This blueprint's addition is
that **the gate must be on the worst slice, and no formula substitutes for it.**

Fitting a power law to the corpus's own three anchors (`fp8 <1%`, `4-bit 1–2%`, `2-bit 10–15%` `[R]`),
driven by the SNR the quantizers actually produce:

| precision | measured SNR | corpus | fitted | error |
|---|---|---|---|---|
| fp8 | 44.63 dB | 0.5% | 0.40% | −19.3% |
| int4 | 20.56 dB | 1.5% | 2.51% | **+67.2%** |
| int2 | 3.33 dB | 12.5% | 9.26% | −25.9% |

**The fit is bad, and that is the finding rather than a defect to tune away.** A power law in SNR
cannot pass through the corpus's own anchors, because the local slopes between them are
inconsistent: fp8→int4 implies an exponent of **0.40**, int4→int2 implies **1.07**. If quality loss
were a smooth function of reconstruction error, those would be equal. **No single exponent
reconciles them, and that is the quantitative form of the corpus's advice — there is no formula
that will tell you in advance.**

### 12.1 The gate's value depends on the quantizer

| quantizer | severity 1 | severity 4 | severity 12 | severity 48 |
|---|---|---|---|---|
| per-tensor | 3.89% | 5.95% | 6.27% | 3.38% |
| group 32 | 2.57% | 2.54% | 2.51% | 2.50% |

With group-wise quantization the prediction barely moves — severity is irrelevant, as §4.1 showed —
so the gate is nearly a formality. With a per-tensor scale it moves across the whole budget, and the
gate is doing real work. **A quantizer that is flat across tensors does not need a gate; a quantizer
whose quality is a property of the checkpoint does, and nothing substitutes for it.**

### 12.2 Mean versus worst

Per-tensor at 4 bits across severity: **mean 5.29%, worst 6.88%.** A threshold of 6% sits between
them on purpose:

```
  gate on the MEAN  ->  5.29% < 6%  -> would PASS
  gate on the WORST ->  6.88% > 6%  -> REJECT
```

**Same numbers, opposite decisions.** The mean is what a summary reports; the worst is what a user
experiences. A gate on the mean ships exactly the regression the corpus warns about.

---

## 13. Build vs buy

| Component | Build | Adopt | Recommendation |
|---|---|---|---|
| the quantizer | — | AWQ / GPTQ / NF4 / FP8 `[T]` | **adopt.** Kernels are hardware-coupled and the implementations are mature |
| group size / scheme | the decision | — | **build the decision.** It is workload-specific and it is the trade in §4 |
| KV cache precision | — | engine flag | **adopt**, but evaluate at production context length |
| the eval gate | the gate and its threshold | — | **build.** No toolkit ships "block the merge on the worst slice" |
| worst-channel instrumentation | per-layer error reporting | — | **build.** It is the metric that sees §4.1 |
| the cost model | the lever decomposition | — | **build.** Which lever matters is a property of your workload's shape (§8) |

**Same shape as T07–T09.** The mechanisms are solved and shipped; the *decisions* — which lever,
which group size, which precision at which layer, and whether the worst slice survives — are not,
and they are the deliverable.

---

## 14. What changes at 10× scale

| At 1× | At 10× | Why it changes |
|---|---|---|
| one quantization config for the fleet | per-model and per-route configs | severity varies by checkpoint (§4.1); a config validated on one model is unvalidated on another |
| weights quantized to fit | KV quantized for concurrency | the constraint moves from fitting to throughput (§3) |
| evals on a sample | evals per model, per context length | the failure is model-specific **and** context-specific (§7, §12) |
| one GPU class | a mixed fleet | FP8 is native on H100/B200/4090 and not everywhere `[R]`; a config is a hardware decision |
| nominal bit widths in the plan | effective bits, with metadata | at fleet scale the 3% metadata overhead is 3% of the bill (§2) |
| cost per token | cost per *useful* token | the prompt/output ratio dominates at scale (§9) |
| quality checked once | a merge gate on the worst slice | manual validation does not scale (§12) |

**The one that bites first is per-model validation.** At 1× a team validates the quantization on the
model it deploys. At 10× there are several models, several context lengths and several GPU classes,
and the property being validated — "this quantizer is flat across tensors" — is exactly the one that
does not transfer.

---

## 15. Six things to carry away

1. **Bits are not bytes.** A 4-bit weight with a per-group fp16 scale costs 4.125 bits. A "4-bit
   model" is 25.8% of bf16, not 25%, and at fleet scale that gap is on the bill (§2).
2. **Weight quantization and KV quantization answer different questions.** Weights is a *fitting*
   lever (1.21× concurrency here, but the difference between fitting and not for a 70B on 80 GB);
   KV is a *concurrency* lever, a consistent 4× at every context length (§3).
3. **Error comes from where the scale is fitted, not from the bit width.** Per-tensor 4-bit destroys
   a channel outright; per-channel and group-wise are flat across a 48× change in outlier severity,
   and that flatness is why they are the default (§4).
4. **Aggregate SNR can improve while the model gets worse.** SNR is non-monotone in outlier
   severity because the outliers inflate the signal too. Report the worst channel, not the mean —
   the same tail-over-mean discipline as T08's goodput and T09's p99 (§4.1).
5. **The ladder's rungs act on different terms, so which one is worth climbing is a property of the
   workload's shape.** The corpus's caching win is 57% on a prompt-heavy workload and 1% on an
   output-heavy one, and its batching win is mostly captured at a batch depth in the single digits
   (§8).
6. **There is no formula from bits to quality, and that is a finding rather than a gap.** The
   corpus's own three anchors cannot be reconciled by any single exponent, which is why its advice
   is to run your evals — and why the gate must be on the worst slice, since the mean would ship the
   regression (§12).

---

## Sources

Corpus (transcripts under `refs/`):

- `refs/LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` — the 100 → 42 → 26 → 11 cost ladder; "Four-bit quantization is nearly free on many models, but on some it quietly drops accuracy. So always rerun your evals after quantizing"; "AWQ and GPTQ to quantize"; "caching only helps a stable prefix".
- `refs/Agentic_AI_Infra_transcripts_2/Banghua_Zhu_-_Building_Frontier_Inference_and_Training_Infra_for_Agent_A_Case_St.txt` — native "8-bit and also 4-bit training"; "FP4 native rollout … without any performance loss"; quantization-aware training with a lower-precision rollout stage.
- `refs/vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Distributed_Inference_on_ROCm_with_WideEP_on_vLLM_llm-d.txt` — quantization techniques as first-class kernels in AMD's ROCm library.

Supporting repos:

- `refs/ai-system-design-guide-main/ai-system-design-guide-main/03-training-and-adaptation/07-quantization-deep-dive.md` — the precision/quality table (the fit anchors), NF4/AWQ/FP8, GGUF vs EXL2, KV-cache quantization and the "4x higher concurrency" claim, QAT for sub-3B models.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/01-inference-fundamentals.md` — the prefill/decode asymmetry and the FP8/dynamic-scaling notes behind §8's cost terms.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/09-on-device-and-edge-deployment.md` — the 4-bit on-device standard, the "over-quantizing (Q2/Q3 hurts reasoning)" pitfall, and the sub-4 GB mobile RAM budget.
- `refs/ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/07-cost-optimization-playbook.md` — VRAM levers (GQA, quantization) in the FinOps frame.
- `refs/llm-inference-engineering-main/llm-inference-engineering-main/README.md` — the KV cache → paged attention → engine → hardware reading order.

Runnable: [`run.py`](run.py) and [`sim/`](sim/) — `precision.py`, `quantize.py`, `quality.py`, `cost.py`, `experiments.py`. Stdlib-only, offline, no GPU; `python run.py` exits 0 and prints every figure quoted above.
