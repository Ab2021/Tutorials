# Quantization

> Module 14B — Efficient AI and On-Device Inference
> Topic 05 of 11
> Estimated study time: 4 hours
> Prerequisites: 02, 03
> Version 2026.3

## 1. Why this topic exists

Quantization is the most widely deployed compression method in edge machine learning, and the one most often applied without understanding. A team converts a model to INT8, measures accuracy on the test set, observes a small drop, and ships. The small drop is real on average and can be large for a subgroup, a rare class, or an input outside the calibration distribution. The method works, and its failure modes are invisible in the metric most teams report.

The purpose of this topic is to make the failure modes visible. Quantization is a mapping from real numbers to a finite set of integers. The mapping has a scale, a zero point, a rounding rule, and a range. Every property of the mapping determines which values survive and which are destroyed. A team that can derive the mapping can predict which layers will fail, which data will expose the failure, and which remedy will fix it.

Three properties make quantization different from the other methods in this module.

**It changes every value.** Distillation changes the architecture and pruning removes weights, but quantization keeps the same weights and alters their representation. The alteration is small per value and accumulates through the network. A layer with a 30 dB signal-to-quantization-noise ratio (SQNR) can be acceptable in isolation and catastrophic after twenty layers of accumulation.

**It is data-dependent in a way that is easy to get wrong.** The scales for weights are computed from the weights, which are fixed and available. The scales for activations are computed from calibration data, which is a sample. If the sample is not representative, the ranges are wrong, and the error appears in the population the sample missed. Calibration is a data-governance activity disguised as a numerical one.

**It interacts with the hardware and the runtime.** A quantized model is only fast if the target has integer kernels for its operators, at its precision, with its layout. A model that is numerically quantized and runs on float kernels is slower and less accurate than the original. The conversion succeeding is not evidence of acceleration.

The clinical stakes are direct. Quantization can shift the operating threshold, widen a subgroup gap, increase false negatives for a rare condition, and change the alarm rate per hour. It can also make a device feasible that would otherwise be too large, too slow, or too hot. The difference between the two outcomes is the quality of the quantization work, and the quality is measurable.

### 1.1 Where quantization pays

Quantization is not uniformly beneficial. It pays in four situations and is neutral or harmful in three others.

| Situation | Why it pays | Evidence to collect |
|---|---|---|
| Weight bytes dominate deployment | 4× fewer weight bytes at 8 bits | Weight traffic, flash budget |
| Integer kernels are faster than float | More MACs per cycle | Target latency at both precisions |
| Memory bandwidth is the bottleneck | Fewer bytes per activation | Roofline placement |
| Model size limits the product | Smaller flash or RAM part | BOM and memory high-water mark |
| Model is compute-bound and integer kernels are equal | MAC count unchanged | Latency unchanged |
| Activations dominate and only weights are quantized | Activation bytes unchanged | Activation byte count |
| No integer kernel for a key operator | Fallback adds latency | Graph partition |

The table is a filter. A team that cannot state which row applies should not start with quantization; it should finish the profiling in Topic 02.

### 1.2 The three questions a quantization review asks

1. **What is the scheme and granularity per tensor?** "INT8" is not an answer. Symmetric or affine, per-tensor or per-channel, which layers excluded.
2. **What data set the activation ranges?** Size, composition, dates, and whether it is disjoint from the test set.
3. **What changed in the outputs?** Accuracy, calibration at the threshold, subgroup performance, and saturation rate, each with an interval.

A quantization report that answers these three can be reviewed. One that reports a single accuracy number cannot.

The questions are deliberately ordered. The scheme determines what the model computes; the data determines the ranges; the evaluation determines whether the change is acceptable. A weakness in the first cannot be compensated by strength in the third.

## 2. Learning outcomes

By the end of this topic you can:

1. **Derive** affine and symmetric scales and zero points by hand for a weight or activation tensor.
2. **Quantize** and dequantize a tensor, and compute SQNR, maximum error, and cosine similarity.
3. **Choose** a granularity and justify the metadata cost.
4. **Select** a calibration method and a calibration set, and explain the bias each introduces.
5. **Diagnose** outlier-driven scale collapse and per-channel skew from layer statistics.
6. **Explain** the straight-through estimator and why quantization-aware training needs it.
7. **Evaluate** a quantized model on accuracy, calibration, subgroups, latency, and energy.
8. **Distinguish** weight-only from weight-and-activation quantization and predict which will help a given layer.

| Outcome | Practised in | Evidenced by |
|---:|---|---|
| 1 | Exercise 1 and Section 6 | Correct scale, zero point, endpoints |
| 2 | Exercise 2 and Section 7 | SQNR and error metrics |
| 3 | Exercise 3 | A granularity choice with byte cost |
| 4 | Exercise 4 | A calibration plan with a bias analysis |
| 5 | Exercise 5 | Two diagnostics and a remedy |
| 6 | Exercise 6 | An STE explanation with a gradient argument |
| 7 | Exercise 7 | An evaluation table with intervals |
| 8 | Exercise 8 | A prediction tied to bytes and MACs |

## 3. The core mental model

Quantization maps a real interval onto a finite set of integers. The mapping has three parts: a range, a scale, and an offset.

```text
   real value x
        │
        ▼
   ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
   │  clamp to   │────►│ divide by S  │────►│ round and add│────► integer q
   │  [lo, hi]   │     │              │     │ zero point Z │
   └─────────────┘     └──────────────┘     └──────────────┘
        │                                           │
        │                                           ▼
        │                                    ┌──────────────┐
        └───────────────────────────────────►│ x_hat = S(q-Z)│
                                             └──────────────┘
```

The forward mapping is `q = clamp(round(x/S) + Z, q_min, q_max)`. The inverse is `x_hat = S(q - Z)`. The error is `x - x_hat`, and it has two components: rounding error, which is bounded by S/2 when the value is inside the range, and clipping error, which can be arbitrarily large when the value is outside.

### 3.1 Affine and symmetric

**Affine** quantization uses both a scale and a zero point. It maps the observed minimum to `q_min` and the observed maximum to `q_max`, so the full integer range is used even when the data is asymmetric.

```text
S = (x_max - x_min) / (q_max - q_min)
Z = round(-x_min / S)
```

**Symmetric** quantization sets `Z = 0` and maps `[-amax, amax]` onto `[-q_max, q_max]`. It wastes part of the range when the data is asymmetric, but it removes the zero-point term from the arithmetic and is the standard choice for weights.

```text
S = amax / q_max
```

The choice matters for two reasons. First, affine uses the range more efficiently, which matters when the data has a large offset or a one-sided distribution. Second, symmetric makes the integer arithmetic simpler: the zero point cancels in many operations, and the accumulation can be done with a single scale. Most runtimes use symmetric weights and asymmetric activations for exactly this reason.

### 3.2 Granularity

A scale can be shared by a whole tensor, by an output channel, or by a group of weights.

| Granularity | One scale per | Metadata | Accuracy | Typical use |
|---|---|---|---|---|
| Per-tensor | Whole tensor | 5 bytes | Lowest | Activations, simple kernels |
| Per-channel | Output channel | 5 bytes × channels | High | Convolution and linear weights |
| Per-group | Group of weights (for example, 128) | 5 bytes × groups | Very high | Large language model weights |
| Per-token | Input token, computed at runtime | 5 bytes × tokens | High for activations | Transformer activations |
| Per-block (microscaling) | Shared exponent per block | Format-specific | High | FP4/FP8 accelerator formats |

Per-channel is the default for weights because it costs little and recovers most of the accuracy lost to per-tensor skew. The cost is 5 bytes per channel with a float32 scale and an int8 zero point, or 3 bytes if the scale is float16 and the zero point is omitted. For a layer with 512 output channels and 64 weights each, the metadata is 2,560 bytes against 32,768 bytes of payload, or 7.8%.

### 3.3 Where the error enters

Quantization error enters at three points, and they behave differently.

- **Weight quantization** changes the function the layer computes. The error is fixed at conversion time and can be analyzed offline.
- **Activation quantization** changes the input to each layer at runtime. The error depends on the input distribution and can be larger for out-of-distribution inputs.
- **Accumulation** is usually done in a wider integer (int32) and requantized to int8. The requantization step introduces another rounding, and the accumulator width bounds the largest dot product that can be represented without overflow.

A model can have excellent weight SQNR and poor end-to-end accuracy because the activation ranges are wrong. Conversely, it can have mediocre weight SQNR and acceptable accuracy because the network is robust to weight noise. The two must be measured separately.

### 3.4 The fake-quantization view

Quantization-aware training (QAT) inserts a fake quantizer into the forward pass: the tensor is quantized and immediately dequantized, so the values seen by the next layer are the quantized values, but the graph remains differentiable in the usual sense.

```text
x ──► quantize ──► dequantize ──► next layer
      (integer)     (float)
```

The backward pass needs a gradient through the rounding operation, which is zero almost everywhere. The straight-through estimator (STE) substitutes the identity:

```text
∂x_hat/∂x ≈ 1
```

This is not the true gradient; it is a useful approximation that lets the network learn weights that are robust to quantization. The approximation is biased, and QAT can converge to a different solution than float training, which is why QAT models must be evaluated, not assumed equivalent.

### 3.5 What quantization does to a distribution

A useful way to see the effect is to plot the quantizer's step against the data.

```text
   count
     ▲
     │        ███
     │      ███████
     │    ███████████
     │  ███████████████
     └──┴──┴──┴──┴──┴──┴──► weight value
        ▲              ▲
        │              └── outlier sets the scale
        └── most values get few levels
```

When one value is far from the bulk, the scale is set by that value, and the bulk is represented by a small number of levels. The aggregate SQNR can still look good because the outlier dominates the signal energy, while the bulk is coarsely quantized. This is the single most important diagnostic in quantization, and Section 6 works through it numerically.

### 3.6 Integer arithmetic and requantization

A quantized convolution does not compute in real numbers. It computes an integer dot product and then converts the result back to the output scale.

```text
q_out = round( (S_in × S_w / S_out) × (Σ q_in × q_w) ) + Z_out
```

The term `S_in × S_w / S_out` is the requantization multiplier. It is a real number, and multiplying an int32 accumulator by it requires either floating-point arithmetic or a fixed-point approximation. Most runtimes use a fixed-point multiplier of the form `M × 2^-n`, where M is an integer and n is a shift, because integer multiplication and shifting are cheap on the target.

Three consequences follow.

- **The multiplier's precision is part of the model.** A coarse approximation changes the output by more than the quantization error, especially when the scale ratio is far from 1.
- **The accumulator must be wide enough.** A dot product of 1,024 int8 values can reach about 1,024 × 127 × 127 ≈ 16.5 million, which fits in int32 with room to spare; a much larger reduction may not.
- **The zero point must be handled consistently.** In affine quantization, the input zero point contributes a term `-Z_in × Σ q_w` that must be subtracted or folded into the bias. A kernel that omits it produces a constant offset, which is a common source of a small, uniform accuracy loss.

The integer view explains why symmetric weights are popular: with `Z_w = 0`, the cross term vanishes, and the inner loop is a pure dot product plus a bias.

### 3.7 What quantization does not change

Quantization changes precision, not architecture. It does not:

- Reduce the number of MACs. A quantized convolution performs the same multiply-accumulates in a narrower format.
- Reduce the number of parameters. The same weights exist, at lower precision.
- Remove operators or fuse the graph. Those are separate transformations.
- Improve a model that is memory-latency-bound rather than bandwidth- or compute-bound.
- Guarantee a speedup. The runtime must have an integer kernel that the target can execute efficiently.

The list matters because teams often expect quantization to solve a latency problem that is caused by operator fallback or graph overhead. Those causes require different remedies, and no bit width fixes them.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Quantization | Mapping real values to a finite set of integers | Used to mean compression generally | Method |
| Dequantization | Mapping integers back to real values | Assumed exact | Method |
| Scale (S) | Real units per integer step | Confused with learning rate | Real units/step |
| Zero point (Z) | Integer code representing real zero | Assumed always zero | Integer |
| Affine | Quantization with a non-zero zero point | Called asymmetric | Scheme |
| Symmetric | Quantization with zero point zero | Assumed lossless | Scheme |
| Bit width | Bits per quantized value | Confused with storage width | Bits |
| qmin, qmax | Representable integer range | Assumed 0–255 for signed types | Integer |
| Per-tensor | One scale for a tensor | Assumed accurate enough | Granularity |
| Per-channel | One scale per output channel | Confused with per-group | Granularity |
| Per-group | One scale per group of weights | Confused with block floating point | Granularity |
| Per-token | One scale per input token | Assumed free | Granularity |
| Calibration | Selecting ranges from data | Confused with probability calibration | Activity |
| Calibration set | Data used to select ranges | Confused with test set | Dataset |
| Min-max calibration | Use observed min and max | Assumed optimal | Method |
| Percentile calibration | Clip a tail percentile | Assumed to reduce error always | Method |
| MSE calibration | Choose scale to minimize squared error | Confused with SQNR maximization | Method |
| KL calibration | Minimize divergence between distributions | Assumed to be the default | Method |
| Entropy calibration | Histogram-based calibration used by TensorRT | Confused with KL calibration | Method |
| SQNR | Signal-to-quantization-noise ratio | Reported without the signal definition | dB |
| Clipping error | Error from values outside the range | Ignored in SQNR | Real |
| Rounding error | Error from mapping to the nearest code | Assumed uniform | Real |
| Saturation | Fraction of values clipped | Ignored in accuracy reports | Proportion |
| PTQ | Post-training quantization | Assumed to require no data | Method |
| QAT | Quantization-aware training | Assumed to be a runtime feature | Method |
| Static quantization | Scales fixed from calibration | Confused with PTQ | Method |
| Dynamic quantization | Scales computed at runtime | Confused with QAT | Method |
| Weight-only quantization | Weights quantized, activations float | Assumed to speed up compute | Method |
| W8A8 | 8-bit weights and activations | Assumed universal support | Format |
| W4A16 | 4-bit weights, 16-bit activations | Assumed to reduce latency | Format |
| Mixed precision | Different precisions per tensor | Confused with dynamic quantization | Method |
| STE | Straight-through estimator | Assumed exact | Method |
| Fake quantization | Quantize-dequantize in the forward pass | Confused with real quantization | Method |
| Batch normalization folding | Folding BN parameters into the preceding convolution | Assumed to be exact at low precision | Transform |
| Bias correction | Correcting the mean shift after quantization | Confused with recalibration | Method |
| Requantization | Converting int32 accumulators back to int8 | Ignored in error analysis | Step |
| Accumulator | Wider integer register for dot products | Assumed unlimited | Integer |
| Overflow | Accumulator exceeding its range | Assumed impossible | Failure |
| Outlier | Value far from the bulk of the distribution | Removed without analysis | Value |
| Scale collapse | Scale dominated by an outlier, leaving few levels for the bulk | Confused with low SQNR | Failure |
| Channel skew | Large variation in per-channel ranges | Ignored in per-tensor quantization | Failure |
| Zero-point drift | Zero point no longer representing real zero after a distribution shift | Invisible in accuracy metrics | Failure |
| SmoothQuant | Migration of activation outliers to weights by per-channel scaling | Confused with clipping | Method |
| GPTQ | Post-training weight quantization using second-order information | Assumed to be lossless | Method |
| AWQ | Activation-aware weight quantization | Confused with SmoothQuant | Method |
| KV cache quantization | Quantizing stored attention keys and values | Assumed free of accuracy cost | Method |
| Group size | Number of weights sharing a scale | Confused with block size | Count |
| Packing | Storing multiple low-bit values per byte | Ignored in memory accounting | Layout |
| Codebook | Table of representable values | Confused with scale and zero point | Table |

Two definitions deserve emphasis.

**SQNR is not accuracy.** It measures the ratio of signal energy to quantization noise energy on a tensor. A layer can have 40 dB SQNR and still fail clinically if the noise is concentrated in the feature that distinguishes a rare class. SQNR is a diagnostic, not an acceptance criterion.

**Calibration is two different activities.** In this topic it means selecting ranges from data. In clinical statistics it means the agreement between predicted probability and observed frequency. A sentence such as "we calibrated the model on 500 records" is ambiguous; say "we selected quantization ranges on 500 records" or "we assessed probability calibration on 500 records".

## 5. The mechanics, step by step

The procedure produces a quantized model with evidence. It assumes a trained model, a target runtime, and a held-out evaluation set.

### Step 1. Decide which tensors to quantize and at what precision

**Input:** The model graph, the target runtime's supported precisions, and the constraint ledger.

**Output:** A per-tensor precision plan.

**Rationale:** Not every tensor should be quantized. The input and output layers often need higher precision, some normalization layers are numerically sensitive, and some operators have no quantized kernel. A plan prevents both over-quantization and a false sense of uniformity.

A typical plan:

| Tensor group | Precision | Reason |
|---|---|---|
| Convolution and linear weights | INT8 symmetric, per-channel | Mature kernels, high accuracy |
| Convolution and linear activations | INT8 asymmetric, per-tensor | Kernel support, simple arithmetic |
| Bias | INT32 | Prevents overflow and preserves accuracy |
| Layer normalization | FP32 or FP16 | Sensitive to small changes |
| Softmax output | FP32 or INT8 with care | Probability calibration |
| Input preprocessing | FP32 | Sensor scaling and filtering |
| KV cache (if present) | INT8 per-token, evaluated separately | Memory-bound decode |

**Branch point:** If the runtime lacks an integer kernel for an operator, either keep that operator in float or replace it. Keeping it in float creates a partition; replacing it changes the model. Both are decisions that must be recorded.

**Edge case:** A model with dynamic shapes may not support static activation scales. Use dynamic quantization for those tensors or fix the shapes.

### Step 2. Choose the granularity

**Input:** The precision plan and the tensor shapes.

**Output:** A granularity per weight and activation tensor.

**Rationale:** Granularity trades metadata for accuracy. Per-channel is almost always worth its cost for weights; per-tensor is usually necessary for activations because the kernels expect it.

The decision:

| Tensor | Per-tensor | Per-channel | Recommended |
|---|---|---|---|
| Conv weights, few channels | Adequate | Better | Per-channel |
| Conv weights, many channels, similar ranges | Adequate | Marginal gain | Per-tensor if metadata matters |
| Linear weights, large | Poor if skewed | Much better | Per-channel or per-group |
| Activations | Kernel requirement | Rarely supported | Per-tensor |
| Transformer activations | Poor due to outliers | Per-token helps | Per-token if supported |
| Embedding tables | Poor | Per-row | Per-row or per-group |

**Branch point:** If the metadata exceeds about 10% of the payload, evaluate whether per-channel is still worth it. It usually is for accuracy, but on a very small model the bytes matter.

**Edge case:** Some runtimes quantize per-channel weights symmetrically only. A plan that assumes asymmetric per-channel weights may not be implementable.

### Step 3. Select the calibration set

**Input:** Candidate data, the deployment population description, and the subgroup definitions.

**Output:** A calibration set with documented composition and a held-out audit set.

**Rationale:** Activation ranges are only as good as the data they are computed from. A calibration set that misses a site, a sensor, or a demographic group produces ranges that are wrong for that group, and the error is invisible in the aggregate.

The set must:

- Be disjoint from the test set and from the final evaluation set.
- Cover every site, device revision, and subgroup in the intended use.
- Include the tails: motion artifact, poor perfusion, lead-off, noisy segments.
- Be large enough to estimate the tails; a few hundred windows is typical for a small model, more for a transformer.
- Be versioned and described in the design record.

**Branch point:** If the calibration set cannot cover a subgroup, either collect more data or exclude the subgroup from the intended use. Do not calibrate on the majority and hope.

**Edge case:** Calibration data collected during a different season, firmware version, or sensor batch may not represent the deployed device. Record the collection conditions.

### Step 4. Choose the range method

**Input:** The calibration set and the tensor statistics.

**Output:** A range per tensor, with a method and a parameter.

**Rationale:** The range determines the scale, and the scale determines the error. Min-max preserves the full range but is sensitive to outliers. Percentile clipping reduces the scale at the cost of saturating a tail. MSE and KL methods choose the range that minimizes a distributional error.

| Method | Rule | Strength | Weakness |
|---|---|---|---|
| Min-max | `lo = min(x)`, `hi = max(x)` | No clipping | Outlier-sensitive |
| Percentile | Clip at the p-th percentile | Robust to outliers | Saturates the tail |
| MSE | Minimize squared reconstruction error | Balances clipping and rounding | Computationally heavier |
| KL divergence | Minimize D_KL(P || Q) over histograms | Preserves distribution shape | Needs enough samples |
| Entropy | TensorRT's histogram method | Works well for CNNs | Implementation-specific |
| Min-max for activations | Common in NLP | Preserves outliers | Can waste range |

**Branch point:** If the tensor has a heavy tail, percentile or MSE is usually better than min-max. If the tensor is bounded by construction (for example, a post-ReLU activation), min-max is appropriate.

**Edge case:** A percentile that is too aggressive can remove the signal that distinguishes a rare class. Evaluate the clipping fraction on held-out data, not only the SQNR.

### Step 5. Compute scales and zero points

**Input:** The ranges, the bit width, and the scheme.

**Output:** A scale and, for affine, a zero point per tensor or channel.

**Rationale:** This is the arithmetic core. It must be done with the same rounding rule as the hardware, because a mismatch between the calibration arithmetic and the kernel arithmetic produces a systematic error.

For 8-bit affine:

```text
S = (hi - lo) / 255
Z = round(-lo / S)          clamped to [0, 255]
```

For 8-bit symmetric:

```text
S = max(|lo|, |hi|) / 127
Z = 0
```

For a bit width b:

```text
affine:     S = (hi - lo) / (2^b - 1)
symmetric:  S = amax / (2^(b-1) - 1)
```

**Branch point:** If `hi == lo`, the tensor is constant. Use a scale of 1 and a zero point of 0, and record the degenerate case.

**Edge case:** The rounding rule matters at the endpoints. `round(-lo/S)` must produce exactly `q_min` when `lo` is the observed minimum. A floating-point implementation that rounds half-to-even can produce `q_min - 1` and a clamp, which shifts the effective range. Use round-half-away and verify the endpoints.

### Step 6. Insert quantize-dequantize nodes and convert

**Input:** The model, the scales, and the runtime's quantization format.

**Output:** A quantized model in the runtime's format, with a graph that shows where quantization happens.

**Rationale:** The format determines what the runtime can fuse and which operators remain in float. A QDQ representation (QuantizeLinear followed by DequantizeLinear) is explicit and portable; a QOperator representation (QLinearConv and similar) is more compact and may be faster.

The conversion checklist:

- The quantized operators match the runtime's supported list.
- The scales are attached to the correct tensors and axes.
- The bias is int32 with scale equal to input scale times weight scale.
- The graph has no unnecessary float islands.
- The model loads and runs on the target, not only on the host.

**Branch point:** If the converter inserts a float fallback for an operator, decide whether to replace the operator, keep the fallback, or exclude the whole subgraph from quantization.

**Edge case:** A converter can silently change the rounding or the layout. Compare the quantized model's outputs with a reference implementation on a fixed input before measuring accuracy.

### Step 7. Handle outliers

**Input:** The layer statistics and the error analysis.

**Output:** A remedy per failing layer.

**Rationale:** Outliers are the dominant cause of quantization accuracy loss. The remedy depends on where the outlier is and whether it can be moved.

| Outlier location | Remedy | Cost |
|---|---|---|
| A few weight channels | Per-channel scales | Metadata |
| A few activation channels, persistent | SmoothQuant-style migration | A per-channel scaling at inference, often fused |
| A few activation values, sporadic | Clipping at a percentile | Tail saturation |
| Many channels, heavy tail | Mixed precision for the layer | Memory and complexity |
| Outlier in the input | Keep the input layer in float | Small memory cost |

**Branch point:** If the outlier is persistent across inputs and confined to a few channels, per-channel or migration is effective. If it is sporadic, clipping is the only option, and the clipping fraction must be evaluated.

**Edge case:** Clipping can improve mean error while worsening the error on the tail that matters clinically. Evaluate both the bulk metric and the worst-case metric.

### Step 8. Use QAT if PTQ fails

**Input:** A PTQ model that misses the accuracy or calibration requirement.

**Output:** A QAT model with recovered accuracy and a training record.

**Rationale:** PTQ chooses scales from data and does not adapt the weights. QAT lets the weights move to compensate for quantization, which recovers accuracy especially at low bit widths.

The QAT procedure:

1. Insert fake quantizers at the positions from the precision plan.
2. Initialize from the PTQ or float model.
3. Fold batch normalization into the preceding convolution where possible.
4. Train with the STE, using a lower learning rate than the original training.
5. Freeze the quantizer scales after an initial warm-up, or learn them with care.
6. Evaluate on held-out data at each checkpoint.
7. Export the quantized model and re-evaluate after export.

**Branch point:** If QAT does not recover the accuracy, the bit width or the granularity is wrong, or the model is capacity-limited. More QAT epochs will not fix either.

**Edge case:** Batch normalization folding is exact in real arithmetic but not in quantized arithmetic. Fold before quantization and verify the outputs; a fold that changes the rounding can cost accuracy.

### Step 9. Evaluate accuracy, calibration, and subgroups

**Input:** The quantized model, the held-out set, and the clinical threshold.

**Output:** A comparison table with confidence intervals.

**Rationale:** The quantized model is a different model. It must be evaluated on the same evidence as the baseline, with the same threshold rule and the same subgroup definitions.

The evaluation includes:

- Sensitivity and specificity at the operating threshold.
- Calibration: ECE, slope, Brier score.
- Subgroup performance with confidence intervals.
- Performance on tails and edge cases.
- The saturation rate per layer, as a diagnostic.
- Agreement with the float model, as a diagnostic.

**Branch point:** If the quantized model passes overall and fails a subgroup, the decision is not to average the two. The subgroup result is the deployment risk.

**Edge case:** A quantized model can have better calibration than the float model by accident, because quantization smooths extreme probabilities. A better ECE does not imply a better model; check sensitivity and the threshold.

### Step 10. Measure on target, export, and document

**Input:** The quantized model and the target.

**Output:** Target latency, energy, and memory, plus a model card and a change record.

**Rationale:** A quantized model that runs on float kernels is slower and less accurate. Only a target measurement establishes the benefit.

The record includes:

- Quantization scheme, bit width, granularity, and rounding.
- Calibration set version, size, and composition.
- Range method and parameters.
- Per-layer scales and zero points, or a hash of them.
- QAT configuration, if used.
- Accuracy, calibration, and subgroup results.
- Target latency, energy, and memory.
- Runtime, converter, and target versions.
- Change classification.

**Branch point:** If the target latency does not improve, the kernel is not using the integer path. Inspect the graph partition before tuning anything else.

**Edge case:** A model can be faster on the target and slower in the product because of preprocessing changes or a different input pipeline. Measure the pathway, not only the model.

### 5.1 Decision table for quantization outcomes

| Observation | Likely cause | Next action |
|---|---|---|
| Accuracy drops < 0.5 pp, calibration acceptable | Normal PTQ loss | Ship with documented margin |
| Accuracy drops 0.5–2 pp | Outliers or per-tensor scales | Per-channel; percentile; re-evaluate |
| Accuracy drops > 2 pp | Range method or bit width | QAT; mixed precision; exclude a layer |
| One layer dominates the error | Outlier channel | Per-channel or SmoothQuant |
| Calibration shifts, accuracy stable | Activation clipping | Re-tune threshold; recalibrate |
| Subgroup degrades, overall stable | Calibration-set bias | Stratified calibration; re-evaluate |
| Latency unchanged | Float fallback | Inspect partition; replace operators |
| Latency worse | Layout or fallback | Check kernel selection; re-export |
| Memory lower, energy unchanged | Memory-bound model | Combine with compute reduction |
| Model fails to load | Unsupported operator or opset | Check runtime version and operator list |

### 5.2 Edge cases in quantization

**Outlier channels.** A channel whose range is several times the others forces a per-tensor scale that leaves the bulk with few levels. Diagnose with the ratio of the largest channel range to the smallest, and with the number of distinct levels used.

**Per-tensor versus per-channel scales.** Per-tensor is cheaper and sometimes required by the kernel; per-channel is more accurate and costs metadata. The choice is a property of the tensor and the runtime, not a global preference.

**Activation clipping.** Clipping reduces the scale and improves resolution for the bulk, at the cost of saturating the tail. It can lower mean absolute error while raising root-mean-square error, because the clipped values contribute large errors. Choose the metric that matches the clinical consequence.

**Operator fallback.** An unsupported quantized operator runs in float on the CPU. The model is still numerically quantized in part, but the latency benefit is lost and the boundary may add a copy.

**Unsupported operators.** Some operators have no quantized kernel at any precision. The model may fail to load or may run a float fallback. Check the operator list for the exact opset and precision before conversion.

**Zero-point drift.** In asymmetric activation quantization, the zero point represents real zero. If the activation distribution shifts after deployment, the calibrated range may no longer contain the bulk, and the zero point no longer corresponds to the same real value. Saturation rate is the early warning; accuracy is the late one.

**Calibration-set bias.** A calibration set drawn from one site produces ranges tuned to that site. The quantized model's error is larger elsewhere, and the aggregate metric hides it. Stratify the calibration set and evaluate per stratum.

**Thermal throttling.** Quantization reduces energy per inference, which can raise the sustained clock and change the thermal profile. Measure the quantized model under sustained load; do not assume the baseline's thermal behavior carries over.

### 5.3 QAT details that decide the outcome

The QAT loop is short, but four details decide whether it recovers accuracy.

**Initialization.** Starting from the float model usually converges faster than starting from the PTQ model, because the float weights are a better optimum. Starting from PTQ can work if the PTQ loss is small. Record which was used.

**Scale learning.** Scales can be fixed from calibration or learned during training. Learned scales can adapt to the weights, but they can also drift to degenerate values, especially at low bit width. A common compromise is to learn the scales for a warm-up period and then freeze them.

**Batch normalization.** Folding BN into the preceding convolution is standard, but the fold changes the weight range. Fold before inserting quantizers, then recalibrate. Folding after QAT invalidates the scales.

**Learning rate and schedule.** QAT fine-tuning uses a much lower learning rate than the original training, often one or two orders of magnitude lower, with a short cosine or step decay. A full training schedule on a quantized graph usually diverges or overfits.

A QAT run that does not improve over PTQ is diagnostic: either the bit width is too low for the architecture, the granularity is wrong, or the model is at its capacity limit. More epochs will not fix any of the three.

### 5.4 A short checklist before conversion

- The calibration set is disjoint from the test set and covers every subgroup.
- The range method and its parameters are recorded.
- The scales are computed with the same rounding as the target kernel.
- The endpoints map to the representable minimum and maximum.
- Bias is int32 with the correct scale.
- The operator list has been checked against the runtime's quantized kernels.
- A reference input has been run through both the float and quantized graphs and the outputs compared.
- The exported model has been loaded on the target, not only on the host.
- The evaluation threshold was fixed before the quantized results were seen.

The last item is the one most often skipped under schedule pressure, and the one that most often turns a result into an invalid comparison.

### 5.5 How large a calibration set needs to be

The calibration set must estimate the tails of the activation distribution, because the maximum sets the scale. The sample maximum is a biased estimator of the population maximum: with n samples from a distribution, the expected maximum approaches the true maximum slowly, roughly in proportion to 1/log(n) for a Gaussian tail. A small set underestimates the maximum and produces a scale that clips in deployment.

A practical approach:

- Start with a few hundred samples per stratum for a small model, and a few thousand for a transformer.
- Plot the range estimate against sample size; when the estimate stabilizes, the set is large enough.
- Hold out an audit set and measure the saturation rate at the chosen range. A rate above the planned clipping fraction means the range is too tight.
- If the saturation rate is sensitive to the sample size, the set is too small.

This is an empirical procedure, not a formula. The formula would require knowing the tail distribution, which is what the calibration set is trying to estimate.

## 6. Worked example

This is a constructed case. The model and measurements are illustrative, and all arithmetic is reproduced in Section 7.

### 6.1 The layer and the model

The model is the ECG classifier from earlier topics: 41,216 parameters, 4,096,000 MACs, 45.7 ms per inference in INT8 on the Cortex-M4F at 64 MHz, and 1.124 mJ at 3.0 V and 8.2 mA. The baseline FP32 model runs at 138 ms and 3.395 mJ, with sensitivity 0.942 and ECE 0.021.

The layer under study is the fully connected layer: 512 output channels, 64 inputs, 32,768 weights. It is the largest single tensor in the model and the one most sensitive to per-tensor quantization because its channel ranges vary widely.

### 6.2 Scale and zero point by hand

The weights have a minimum of -1.029 and a maximum of 1.2912. The median absolute weight is 0.0610.

**Affine, 8 bits:**

```text
range = 1.2912 - (-1.029) = 2.3202
S     = 2.3202 / 255 = 0.00909882
Z     = round(1.029 / 0.00909882) = round(113.09) = 113
```

Endpoint checks:

```text
q(-1.029)  = round(-113.09) + 113 = -113 + 113 = 0
q(1.2912)  = round(141.91) + 113 = 142 + 113 = 255
```

**Symmetric, 8 bits:**

```text
amax = 1.2912
S    = 1.2912 / 127 = 0.01016693
Z    = 0
q(1.2912)  = 127
q(-1.029)  = round(-101.21) = -101
```

The symmetric scale is 11.7% coarser than the affine scale, and the negative side of the range is underused because the data reaches only -1.029 while the representable minimum is -1.2912.

### 6.3 What the choice does to the weights

| Scheme | Scale | Step / median | Zeroed weights | Distinct levels | SQNR |
|---|---:|---:|---:|---:|---:|
| Symmetric per-tensor | 0.01016693 | 0.167 | 1,411 of 32,768 | 102 of 255 | 30.00 dB |
| Affine per-tensor | 0.00909882 | 0.149 | 1 | 110 of 255 | 30.97 dB |
| Symmetric per-channel | per channel | — | — | — | 44.67 dB |

Three results stand out.

First, 1,411 weights (4.3%) round to zero under the symmetric per-tensor scale, because the step is 16.7% of the median weight. The zeros are not "sparsity" in any useful sense; they are quantization error concentrated in the smallest weights.

Second, affine avoids the zeros almost entirely because it uses the asymmetric range. Its SQNR gain is 0.97 dB, which is modest compared with the per-channel gain.

Third, per-channel quantization gains 14.67 dB over per-tensor symmetric, roughly 2.4 bits of effective precision, at a metadata cost of 2,560 bytes.

### 6.4 Memory before and after

```text
FP32:                32,768 × 4 = 131,072 B = 128.0 KiB
INT8 per-tensor:     32,768 + 5 = 32,773 B = 32.0 KiB
INT8 per-channel:    32,768 + 512 × 5 = 35,328 B = 34.5 KiB
```

The per-channel metadata is 2,560 bytes, or 7.8% of the payload. The compression ratio against FP32 is 4.00 for per-tensor and 3.71 for per-channel. For the whole model at 41,216 parameters and 120 channels, the INT8 size is 41,216 + 600 = 41,816 bytes, a 3.94× reduction.

Activations are a separate budget. A fully connected layer with a batch of one moves 64 input values and 512 output values per inference, which is negligible in bytes but non-negligible in latency because the layer's arithmetic intensity is about 1 MAC per byte at batch 1. This is the weight-bound case from Topic 02.

### 6.5 Activation calibration and clipping

The layer's input activations were collected over 5,000 calibration windows. The maximum is 4.538; the 99.9th percentile is 3.879.

| Method | Range | Scale | MAE | RMSE | SQNR | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| Min-max | 0 to 4.538 | 0.017796 | 0.0044 | 0.0051 | 47.91 dB | 0% |
| 99.9th percentile | 0 to 3.879 | 0.015212 | 0.0040 | 0.0127 | 39.97 dB | 0.1% |

The percentile method lowers mean absolute error by 9% and lowers SQNR by 7.9 dB, because the clipped values produce large errors that dominate the squared-error metric. This is the clipping trade-off in its purest form: better average behavior, worse tail behavior. Which is preferable depends on whether the clinical consequence is driven by typical cases or by the tail. For an arrhythmia detector, the tail often contains the events of interest, so the min-max range is the safer default unless the clipped fraction is shown to be benign.

### 6.6 Latency and energy

The model-level INT8 latency is 45.7 ms, against 138 ms in FP32, a 3.02× speedup. The speedup comes from the integer kernels, not from the byte reduction: the model is compute-bound, and the integer pipeline performs more MACs per cycle.

```text
E_INT8 = 3.0 V × 0.0082 A × 0.0457 s = 1.124 mJ
E_FP32 = 3.0 V × 0.0082 A × 0.138 s = 3.395 mJ
ratio  = 3.02
```

The energy ratio equals the latency ratio because the board current is assumed constant. In reality the integer pipeline may draw less current per cycle, so the energy gain can exceed the latency gain; measure it rather than assume it.

### 6.7 Accuracy and calibration cost

| Model | Sensitivity | Specificity | ECE | Slope | Worst subgroup |
|---|---:|---:|---:|---:|---:|
| FP32 baseline | 0.942 | 0.961 | 0.021 | 0.97 | 0.931 |
| INT8 per-tensor PTQ | 0.934 | 0.956 | 0.036 | 0.88 | 0.921 |
| INT8 per-channel PTQ | 0.938 | 0.958 | 0.029 | 0.93 | 0.926 |
| INT8 per-channel QAT | 0.941 | 0.960 | 0.024 | 0.96 | 0.930 |
| INT4 weight-only, per-channel | 0.929 | 0.952 | 0.039 | 0.86 | 0.915 |

The per-tensor model passes the overall floor of 0.925 with 0.9 points of margin, but its slope falls to 0.88, which means the predicted probabilities are too extreme. The per-channel model recovers 0.4 points of sensitivity and 0.007 of ECE. QAT recovers almost everything at the cost of a training run.

The slope column deserves separate attention. A slope of 0.88 means that a one-unit increase in predicted log-odds corresponds to 0.88 units in observed log-odds, so the model's probabilities are too extreme. At the operating threshold, this can move the effective sensitivity even when the ranking is unchanged, because the threshold is a probability, not a rank. That is why the threshold must be re-selected on validation data for each quantized candidate and then frozen for the held-out evaluation.

The INT4 weight-only model is the least accurate and, on this compute-bound model, no faster than INT8. It is included to show that a lower bit width is not automatically better: it costs accuracy and buys memory that the model does not need.

### 6.8 Diagnostic one: outlier-driven scale collapse

The per-tensor symmetric step is 0.01016693, which is 16.7% of the median absolute weight (0.0610). Any weight below 0.00508 rounds to zero. In this tensor, 1,411 weights (4.3%) do so, and only 102 of 255 levels are used.

The diagnostic pair is:

```text
step / median(|w|) = 0.01016693 / 0.0610 = 0.167
distinct levels used = 102 / 255
```

A step above 10% of the median, or fewer than half the levels in use, indicates collapse. The remedy is per-channel quantization, which raises SQNR from 30.00 to 44.67 dB. Affine quantization helps only marginally (30.97 dB) because the outlier is positive and the negative side is not the problem.

### 6.9 Diagnostic two: per-channel skew

The 512 output channels have ranges from 0.2756 to 2.3202, a ratio of 8.4. Channel 7 is the outlier: its weights were scaled by six and one weight is 0.62, so its range is 2.3202 while a typical channel has 0.28.

Under a per-tensor scale, a typical channel gets about 28 levels, which is roughly 4.8 bits of effective resolution. Under per-channel scales, each channel gets the full 255 levels. The SQNR gain is 14.67 dB.

The metadata cost is 2,560 bytes. On a 35,328-byte layer, that is 7.8%; on the 41,816-byte model, 1.4%. The diagnostic is the channel range ratio; above about 4, per-tensor will cost material accuracy.

### 6.10 Varying one assumption

**Calibration set: 512 windows to 64.** The activation maximum is undersampled, the range shrinks by about 6%, and clipping increases. Sensitivity falls a further 0.4 points to 0.930 for the per-channel model, and ECE rises to 0.033. The weight quantization is unchanged; only the activation scales move. The lesson is that calibration set size is an accuracy variable, and a small set can silently shift the operating point.

**Bit width: 8 to 4 for weights, per-channel.** The layer's weight bytes fall to 16,384 + 512 × 5 = 18,944 bytes, and the model to 20,608 + 600 = 21,208 bytes (20.7 KiB). The SQNR of the weight tensor falls from 44.67 to 4.87 dB, cosine similarity to 0.866, and sensitivity to about 0.929. On this compute-bound model, latency does not improve. INT4 is a memory tool for weight-bound models, not a universal upgrade.

**Batch size: 1 to 16.** The layer's weight traffic is amortized over 16 inputs, so its arithmetic intensity rises from about 1 MAC per byte to about 10. The layer becomes compute-bound, and the benefit of weight-only quantization disappears. This is why a quantization scheme validated at batch 1 must be re-evaluated at the batch the product uses.

**Layer excluded: the first convolution kept in FP32.** Adding 120 bytes (0.29% of the model) recovers about 0.2 points of sensitivity, because the first layer's quantization error propagates through every later layer. The trade is favorable for a small layer and unfavorable for a large one.

**Clipping percentile: 99.9 to 99.0.** The activation range shrinks further, the scale falls by about 4%, and the saturation rate rises to 1%. Sensitivity falls by 0.3 points in the constructed case. Aggressive clipping is a bet that the tail does not matter; for an arrhythmia detector, that bet needs clinical evidence.

### 6.11 The decision

The per-channel, per-tensor-activation INT8 model meets the overall floor at 0.938 with 1.3 points of margin, and its worst subgroup at 0.926 clears a 0.915 subgroup floor by 1.1 points. Its latency of 45.7 ms is above the 40 ms model budget used in Topic 03, so it does not meet the latency constraint without distillation or pruning.

The decision is therefore not "quantize or not" but "quantize and then reduce MACs". Quantization alone reduces memory from 161.0 KiB to 40.8 KiB and latency from 138 ms to 45.7 ms; it does not reach 40 ms because the model remains compute-bound. The next method is structural, not numerical: distillation or structured pruning, as in Topic 03.

This is the practical lesson of the topic. Quantization is necessary but not sufficient for a compute-bound model. It removes precision cost and memory cost; it does not remove arithmetic.

### 6.12 What the case does not show

- **The rest of the model's layers.** The fully connected layer is 32,768 of 41,216 parameters but a small share of MACs. Its quantization error is not the model's error.
- **The activation quantization of earlier layers.** Each layer has its own ranges and its own sensitivity.
- **The interaction with the runtime's requantization.** The case assumes exact requantization; fixed-point multipliers introduce additional error.
- **The effect of the exported graph.** Fusion and layout changes can alter the outputs after the scales are fixed.
- **The subgroup structure beyond the worst group.** A single worst-group number hides the shape of the distribution.

The case is a method demonstration. The model-level conclusion requires the full evaluation in Step 9.

It also does not show the interaction between quantization and the other methods. A quantized model that is then distilled must be re-quantized, and the scales from the first quantization are not valid for the second. That interaction is the subject of Topic 03's composition rules.

## 7. Code, query, or calculation

The script implements affine and symmetric quantization, per-channel scales, activation clipping, memory accounting, and error metrics. It reproduces Section 6.

```python
import math
import random


def round_half_away(value):
    """Round as integer kernels do: halves away from zero."""
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def symmetric_scale(values, bits=8):
    amax = max(abs(v) for v in values)
    qmax = 2 ** (bits - 1) - 1
    return amax / qmax if amax else 1.0


def affine_params(values, bits=8):
    lo, hi = min(values), max(values)
    qmin, qmax = 0, 2 ** bits - 1
    scale = (hi - lo) / (qmax - qmin)
    if scale == 0:
        scale = 1.0
    zero_point = min(qmax, max(qmin, round_half_away(-lo / scale)))
    return scale, zero_point


def quantize_affine(values, scale, zero_point, bits=8):
    qmax = 2 ** bits - 1
    return [min(qmax, max(0, round_half_away(v / scale) + zero_point))
            for v in values]


def quantize_symmetric(values, scale, bits=8):
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    return [min(qmax, max(qmin, round_half_away(v / scale))) for v in values]


def dequantize(codes, scale, zero_point=0):
    return [(q - zero_point) * scale for q in codes]


def error_metrics(original, reconstructed):
    n = len(original)
    mae = sum(abs(a - b) for a, b in zip(original, reconstructed)) / n
    mse = sum((a - b) ** 2 for a, b in zip(original, reconstructed)) / n
    signal = sum(a * a for a in original)
    noise = sum((a - b) ** 2 for a, b in zip(original, reconstructed))
    dot = sum(a * b for a, b in zip(original, reconstructed))
    norm = math.sqrt(signal) * math.sqrt(sum(b * b for b in reconstructed))
    return {"mae": mae, "rmse": math.sqrt(mse),
            "sqnr_db": 10 * math.log10(signal / noise) if noise else float("inf"),
            "cosine": dot / norm if norm else 0.0}


def weight_bytes(params, bits=8, channels=None):
    payload = params * bits / 8
    metadata = channels * 5 if channels else 0
    return payload + metadata
```

Build the tensor and quantize per tensor:

```python
random.seed(11)
channels, fan_in = 512, 64
weights = [[round(random.gauss(0, 0.09), 4) for _ in range(fan_in)]
           for _ in range(channels)]
weights[7] = [round(v * 6.0, 4) for v in weights[7]]
weights[7][3] = 0.62
flat = [v for row in weights for v in row]

s_sym = symmetric_scale(flat, 8)
q_sym = quantize_symmetric(flat, s_sym, 8)
print("min %.4f max %.4f median %.4f"
      % (min(flat), max(flat), sorted(abs(v) for v in flat)[len(flat) // 2]))
print("symmetric scale %.8f" % s_sym)
print("zeros", sum(1 for q in q_sym if q == 0),
      "levels", len(set(q_sym)))
print("symmetric", {k: round(v, 4)
                    for k, v in error_metrics(flat, dequantize(q_sym, s_sym)).items()})

s_aff, zp = affine_params(flat, 8)
q_aff = quantize_affine(flat, s_aff, zp, 8)
print("affine scale %.8f zero point %d" % (s_aff, zp))
print("affine zeros", sum(1 for q in q_aff if q == 0),
      "levels", len(set(q_aff)))
print("affine", {k: round(v, 4)
                 for k, v in error_metrics(flat, dequantize(q_aff, s_aff, zp)).items()})
```

Expected output:

```text
min -1.0290 max 1.2912 median 0.0610
symmetric scale 0.01016693
zeros 1411 levels 102
symmetric {'mae': 0.0025, 'rmse': 0.0029, 'sqnr_db': 29.9977, 'cosine': 0.9995}
affine scale 0.00909882 zero point 113
affine zeros 1 levels 110
affine {'mae': 0.0023, 'rmse': 0.0026, 'sqnr_db': 30.9728, 'cosine': 0.9996}
```

Per-channel:

```python
per_channel = []
for row in weights:
    scale = symmetric_scale(row, 8)
    per_channel.extend(dequantize(quantize_symmetric(row, scale, 8), scale))
print("per-channel", {k: round(v, 4)
                      for k, v in error_metrics(flat, per_channel).items()})
print("fp32 bytes", channels * fan_in * 4)
print("int8 per-tensor bytes", weight_bytes(channels * fan_in, 8))
print("int8 per-channel bytes", weight_bytes(channels * fan_in, 8, channels))
print("ratio per-channel", round(channels * fan_in * 4
                                 / weight_bytes(channels * fan_in, 8, channels), 3))
```

Expected output:

```text
per-channel {'mae': 0.0005, 'rmse': 0.0005, 'sqnr_db': 44.6718, 'cosine': 1.0}
fp32 bytes 131072
int8 per-tensor bytes 32773.0
int8 per-channel bytes 35328.0
ratio per-channel 3.71
```

Activation clipping:

```python
random.seed(5)
activations = [round(abs(random.gauss(0.8, 1.0)), 3) for _ in range(5000)]
ordered = sorted(activations)
p999 = ordered[int(0.999 * len(ordered))]

for name, high in (("minmax", max(activations)), ("p99.9", p999)):
    scale = high / 255
    codes = quantize_affine(activations, scale, 0, 8)
    metrics = error_metrics(activations, dequantize(codes, scale, 0))
    print(name, "high %.3f scale %.6f" % (high, scale),
          {k: round(v, 4) for k, v in metrics.items()})
```

Expected output:

```text
minmax high 4.538 scale 0.017796 {'mae': 0.0044, 'rmse': 0.0051, 'sqnr_db': 47.9071, 'cosine': 1.0}
p99.9 high 3.879 scale 0.015212 {'mae': 0.004, 'rmse': 0.0127, 'sqnr_db': 39.9729, 'cosine': 1.0}
```

Assumptions: the weights are synthetic Gaussian values with one scaled channel; the activation sample is synthetic; rounding is half-away-from-zero; metadata is 5 bytes per channel; and the error metrics use the full tensor without weighting by class or subgroup.

## 8. Data, standards, and regulatory context

Quantization is defined by runtime specifications, not by ISO standards. The specifications constrain what a quantized model means; the regulators constrain what evidence is needed before a quantized model can be used.

### 8.1 The TensorFlow Lite / LiteRT 8-bit specification

Google's 8-bit quantization specification defines the arithmetic that kernels implement:

- `real_value = (int8_value - zero_point) × scale`.
- Activations are int8 in [-128, 127] with a zero point in [-128, 127], asymmetric.
- Weights are int8 in [-127, 127] with zero point 0, symmetric.
- Per-axis quantization is supported along a specified `quantized_dimension`, typically the output-channel dimension of convolution weights.
- Bias is int32, with per-axis scale equal to input scale times weight scale.

The specification is binding for the runtime: a model that does not follow it will be rejected or will fall back to float. It is not a regulatory standard, but it determines whether the quantized model runs at all.

### 8.2 The ONNX Runtime quantization formats

ONNX Runtime supports two representations. QOperator uses operators such as `QLinearConv` and `MatMulInteger`. QDQ inserts `QuantizeLinear` and `DequantizeLinear` around the original operators, with scales and zero points as initializers.

The documented details matter for implementation:

- Per-channel weight quantization is supported for static weights, with the axis determined by the operator (0 for Conv, 1 for MatMul) or by an override.
- Per-channel scale and zero-point initializers are one-dimensional with length equal to the channel count; per-tensor values are scalar.
- Per-channel weights are quantized symmetrically.
- Bias is int32 with zero point 0 and scale equal to input scale times weight scale.
- Opset 10 is the minimum for QDQ; opset 12 is required for some operators; 16-bit and 4-bit types require opset 21 or the `com.microsoft` domain.

A model quantized with the wrong opset or the wrong axis will load and produce wrong results, which is worse than failing to load.

### 8.3 Hardware documentation

Arm's CMSIS-NN implements int8 and int16 kernels bit-exact with the TensorFlow Lite reference for Cortex-M, with int4 weights and int8 activations for some operators. It selects among pure C, DSP-extension, and M-Profile Vector Extension implementations based on feature flags. The bit-exactness claim is important: it means a model validated with the reference kernels should produce the same outputs on the device, which is not true of every accelerator.

TensorRT's documentation describes entropy calibration (`IInt8EntropyCalibrator2`) and min-max calibration (`IInt8MinMaxCalibrator`), and notes that implicit quantization is deprecated in recent versions in favor of explicit Q/DQ quantization. A team following an older tutorial may implement a workflow that the current release no longer supports.

### 8.4 What regulators require

IEC 62304, "Medical device software — Software life cycle processes", Edition 1.1 (2006 plus Amendment 1:2015), recognized by the United States Food and Drug Administration (FDA) under recognition number 13-79, requires the quantization step to be a documented design activity with verification against the requirements.

ISO 14971, "Medical devices — Application of risk management to medical devices", requires the hazards introduced by quantization to be identified: subgroup degradation, calibration shift, saturation on out-of-distribution inputs, and threshold drift.

The FDA's final guidance on Predetermined Change Control Plans, issued on 3 December 2024 with a Federal Register notice on 4 December 2024, provides the pathway for planned model changes. Quantization may be covered if the plan anticipated it and the protocol's acceptance criteria are met. The guidance is non-binding, but the statutory pathway it implements is binding.

In the European Union, Regulation (EU) 2017/745 governs the device and Regulation (EU) 2024/1689, the Artificial Intelligence Act, adds obligations for high-risk systems. The Annex I applicability date was originally 2 August 2027; the AI Omnibus, adopted on 19 November 2025 with political agreement reported on 7 May 2026, moves it to 2 August 2028. Verify the current date before relying on it.

### 8.5 What is binding and what is convention

| Item | Status | Consequence |
|---|---|---|
| TFLite 8-bit specification | Runtime-binding | Model must follow the arithmetic or fall back |
| ONNX QDQ specification | Runtime-binding | Opset and axis determine correctness |
| CMSIS-NN bit-exactness | Vendor claim, documented | Justifies host-side validation for supported ops |
| TensorRT calibration methods | Vendor documentation | Method choice affects accuracy |
| IEC 62304 | Binding where required | Quantization is a design and verification activity |
| ISO 14971 | Binding where required | Quantization hazards enter the risk file |
| FDA PCCP guidance | Non-binding guidance; statute binding | Change must be pre-authorized or submitted |
| "INT8" as a description | Convention | Must state scheme, granularity, and per-tensor/per-channel |
| SQNR thresholds | Convention | No standard defines an acceptable SQNR |

### 8.6 The documentation a quantized model needs

| Record | Content |
|---|---|
| Precision plan | Per-tensor bit width, scheme, granularity |
| Calibration record | Data source, size, dates, subgroup composition |
| Range method | Min-max, percentile, MSE, or KL, with parameters |
| Scale record | Per-layer scales and zero points, or a hash |
| QAT record | Initialization, learning rate, epochs, STE configuration |
| Evaluation record | Accuracy, calibration, subgroups, threshold rule |
| Target record | Latency, energy, memory, runtime and board versions |
| Change record | Classification and PCCP reference |

### 8.7 What "model equivalence" means after quantization

A quantized model is not numerically equivalent to its float baseline, and no standard defines how close it must be. A team must define equivalence itself, and the definition should have three parts.

- **Distributional equivalence:** the accuracy, calibration, and subgroup metrics are within pre-specified margins on a held-out set.
- **Decision equivalence:** at the fixed operating threshold, the sensitivity and the alarm rate per hour are within margins that are clinically justified.
- **Behavioral equivalence on tails:** on a pre-specified set of edge cases, the quantized model does not fail in a way the float model does not.

A definition that covers only the first part is an accuracy check, not an equivalence claim. A definition that covers none is a hope.

## 9. Industry practice and case evidence

### 9.1 The foundational result

Jacob, Kligys, Chen, Zhu, Tang, Howard, Adam, and Kalenichenko published "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" in 2017, with the CVPR version in 2018 (arXiv:1712.05877). The paper established the affine quantization scheme that most runtimes still use, showed how to perform inference with integer arithmetic only, and introduced a quantization-aware training procedure that simulates quantization in the forward pass.

Two details from the paper remain visible in every runtime. The bias is stored as int32 with a scale that is the product of the input and weight scales, which prevents overflow and preserves accuracy. The requantization from the int32 accumulator to the int8 output uses a fixed-point multiplier, which is why the scale ratio is represented as an integer times a power of two rather than as a float.

### 9.2 The PTQ and QAT pipelines

Nagel et al., "A White Paper on Neural Network Quantization" (arXiv:2106.08295, 2021), is the canonical description of the two pipelines.

The PTQ pipeline:

1. Set ranges with min-max, MSE, cross-entropy, or batch-normalization-based methods.
2. Apply cross-layer equalization to reduce activation range variation between layers.
3. Absorb bias into the next layer where possible.
4. Correct the bias to compensate for the mean shift introduced by quantization.
5. Apply AdaRound to choose rounding directions that minimize the loss.

The QAT pipeline:

1. Insert fake quantizers.
2. Use the straight-through estimator for gradients.
3. Learn or freeze the scales with a warm-up.
4. Fold batch normalization.
5. Fine-tune with a lower learning rate.

The paper reports results for W8A8 and W4A8 on ResNet-18/50, MobileNetV2, InceptionV3, EfficientNet-lite, DeepLabV3, EfficientDet-D1, and BERT-base. The general finding is that 8-bit quantization is close to floating point with a good pipeline, and that 4-bit requires QAT and careful handling.

### 9.3 Outliers in large language models

Dettmers et al., "LLM.int8()" (arXiv:2208.07339, 2022), showed that transformer activations develop systematic outliers in a small number of channels beyond about 6.7 billion parameters, with magnitudes roughly 100 times the typical value. The paper's mixed-precision decomposition keeps the outlier dimensions in FP16 and quantizes the rest, preserving accuracy close to FP16. The cost is runtime outlier detection and scatter-gather, which the paper reports as often slower than FP16: 237 ms versus 152 ms on OPT-13B at sequence length 256.

SmoothQuant (Xiao et al., arXiv:2211.10438, ICML 2023) addresses the same problem by migrating the quantization difficulty from activations to weights with a mathematically equivalent per-channel scaling: `Y = (X diag(s)^-1)(diag(s) W)`, with `s_j = max|X_j|^α / max|W_j|^(1-α)`. The paper reports α = 0.5 for OPT and BLOOM, 0.75 for GLM-130B, and 0.8–0.9 for LLaMA, and reports up to 1.56× speedup and about 2× memory reduction with FasterTransformer. Those are the authors' measurements on their hardware.

GPTQ (Frantar et al., arXiv:2210.17323) and AWQ (Lin et al., arXiv:2306.00978) are post-training weight quantization methods for large models. Both target weight-only quantization, where activations remain at higher precision, and both are evaluated primarily on language-model perplexity and task accuracy. They are relevant to on-device language models, not to a small convolutional classifier.

### 9.4 What the evidence supports

| Claim | Status |
|---|---|
| Affine int8 quantization with int32 bias is the standard scheme | Established by Jacob et al. and implemented in TFLite and ONNX Runtime |
| A good PTQ pipeline gives near-float accuracy at 8 bits for CNNs | Nagel et al.; model- and pipeline-dependent |
| 4-bit weights require QAT or advanced PTQ for accuracy | Nagel et al.; GPTQ/AWQ for LLMs |
| Activation outliers are systematic in large transformers | Dettmers et al.; SmoothQuant |
| Per-channel weight quantization recovers most skew loss | Widely reported; magnitude model-dependent |
| INT8 is always faster | Not supported; depends on kernels and whether the model is compute- or memory-bound |
| A 30 dB SQNR layer is clinically safe | Not supported; SQNR is not accuracy |

### 9.5 Why medical quantization differs

Three differences from benchmark quantization matter.

**The calibration population is the deployment population.** A benchmark model can calibrate on the training set. A medical device must calibrate on data that represents every site, sensor, and subgroup in the intended use, and must document that composition.

**The operating threshold is part of the claim.** Quantization shifts probabilities, and a shift at the threshold changes sensitivity and the alarm rate. A benchmark reports accuracy at a threshold chosen after quantization; a device must hold the threshold fixed or justify the change.

**The subgroup is the risk.** A model that loses 0.6 points overall may lose 1.5 points in a small subgroup. The aggregate metric is reassuring and the subgroup is the deployment question.

### 9.6 Three rules from the case evidence

**Rule 1: match the method to the model size.** Per-tensor 8-bit quantization is close to lossless for a small convolutional model with moderate ranges. The same method fails on a transformer with systematic outliers. The literature's successes and failures are both real; they apply to different regimes.

**Rule 2: prefer per-channel for weights.** The evidence is consistent across models and runtimes: per-channel weight scales recover most of the loss from per-tensor skew, at a metadata cost that is small relative to the payload. The exceptions are kernels that do not support per-channel and models so small that metadata dominates.

**Rule 3: never accept a single aggregate number.** Every published quantization result that reports only overall accuracy leaves the deployment question unanswered. The medical case requires calibration, subgroups, and target measurement, and no benchmark substitutes for them.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Outlier-driven scale collapse | Few distinct levels, many zeroed weights | Max sets the scale | Accuracy loss concentrated in small weights | Per-channel; clipping |
| Per-tensor on skewed channels | One channel dominates the range | Shared scale | Small channels lose resolution | Per-channel scales |
| Calibration on the test set | Test accuracy suspiciously stable | Leakage | Invalid evidence | Disjoint calibration set |
| Calibration set misses a subgroup | Aggregate passes, subgroup fails | Ranges tuned to majority | Equity harm | Stratified calibration |
| Activation clipping too aggressive | Saturation rate rises | Tail removed | Rare events missed | Percentile sweep on held-out |
| Zero-point drift | Saturation rises after deployment | Distribution shift | Silent error | Monitor saturation; recalibrate |
| Float fallback | Latency unchanged | Unsupported op | No speedup | Inspect partition; replace op |
| Unsupported operator | Model fails to load | No kernel | Schedule slip | Operator-list audit |
| Wrong opset or axis | Outputs differ from reference | Converter mismatch | Silent wrong results | Compare with reference on fixed input |
| Bias not int32 | Accuracy drops at accumulation | Overflow or scale mismatch | Layer error | Follow the runtime spec |
| Weight-only on compute-bound model | Memory falls, latency flat | MACs unchanged | Wasted effort | Check intensity first |
| QAT after pruning without recalibration | Accuracy below prediction | Scales mismatch | Additional loss | Recalibrate after structure change |
| Thermal profile changed | Sustained latency worse | Higher clock, more heat | Deadline miss | Soak test |
| Unrecorded scales | Model not reproducible | Missing metadata | Evidence invalid | Version the scale record |
| Accumulator overflow | Outputs saturate on long dot products | int32 overflow | Wrong outputs | Check accumulator width |
| Coarse requantization multiplier | Small uniform bias | Fixed-point approximation | Systematic error | Verify multiplier precision |
| Calibration set too small | Range estimate unstable | Undersampled tail | Clipping in deployment | Range-versus-size plot |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Calibration data represents deployment | Scales generalize | Site, sensor, or demographic shift | Stratified evaluation |
| Rounding matches the hardware | Predicted error matches measured | Different rounding mode | Compare outputs on fixed input |
| Per-channel metadata is 5 bytes | Byte accounting | float16 scales, shared zero points | Inspect exported model |
| Weights dominate bytes in linear layers | Weight-only benefit | Large batch, activation-bound | Compute intensity |
| Integer kernels exist for all operators | Speedup | Float fallback | Partition inspection |
| SQNR predicts accuracy | Diagnostics | Error concentrated in a rare feature | Subgroup evaluation |
| Clipping improves the relevant metric | Range selection | Metric is tail-sensitive | Compare MAE and RMSE |
| Bias is int32 | Accuracy and overflow | int8 bias in a custom kernel | Check the specification |
| Accumulator is wide enough | No overflow | Very large dot products | Check kernel accumulator width |
| QAT recovers PTQ loss | Recovery | Capacity limit or wrong granularity | QAT experiment |
| Activation ranges are stable after deployment | No drift | Distribution shift | Monitor saturation |
| Export preserves semantics | Accuracy transfer | Fusion or layout change | Compare outputs |
| Requantization multiplier is accurate enough | Predicts output error | Coarse fixed-point approximation | Compare with reference |
| The operating threshold is fixed across candidates | Fair comparison | Threshold re-tuned per candidate | Check the protocol |
| Saturation rate is monitored | Detects drift | No runtime instrumentation | Add a counter |

The central limit is that quantization error is not uniform across the input space. It is larger where the activation distribution differs from calibration, and it is larger for rare classes whose features are represented by few weights. Aggregate metrics average over the input space and hide both. The only defense is stratified evaluation: by subgroup, by site, by class, and by input quality.

A second limit is that the quantized model's behavior depends on the runtime. The same scales, applied by two kernels with different accumulation orders, can produce different outputs. The runtime is part of the model definition for validation purposes, which is why it must be versioned.

## 12. Exercises

**Exercise 1 — Scale and zero point (calculation, 30 minutes).**
A weight tensor has min -0.35 and max 0.82. Compute the 8-bit affine scale and zero point, verify both endpoints, and compute the symmetric scale. State which scheme uses the range better.
*Deliverable:* hand calculation.
*Acceptance criteria:* endpoints map to 0 and 255; zero point correct; comparison quantified.
*Time:* 30 minutes.

**Exercise 2 — SQNR (calculation, 35 minutes).**
For the tensor in Exercise 1, compute the quantization error for a value of 0.10 under both schemes, and compute SQNR for a three-value tensor [-0.35, 0.10, 0.82].
*Deliverable:* a table of values, codes, reconstructions, and errors.
*Acceptance criteria:* errors correct to four decimals; SQNR formula stated; units in dB.
*Time:* 35 minutes.

**Exercise 3 — Granularity (analysis, 30 minutes).**
A convolution has 256 output channels and 3×3×64 weights each. Compute the metadata for per-tensor, per-channel, and per-group (32 weights) quantization at 8 bits, and state which you would choose and why.
*Deliverable:* byte counts and a recommendation.
*Acceptance criteria:* metadata included; recommendation tied to accuracy and bytes.
*Time:* 30 minutes.

**Exercise 4 — Calibration plan (design, 40 minutes).**
Design a calibration set for a wearable ECG model deployed across three sites and two sensor revisions. State the size, stratification, inclusion of tails, and the audit set.
*Deliverable:* a one-page plan.
*Acceptance criteria:* every site and revision represented; tails included; held-out set disjoint; versioning described.
*Time:* 40 minutes.

**Exercise 5 — Diagnostics (analysis, 45 minutes).**
Given a tensor with 40,000 weights, median absolute value 0.05, max 0.9, and 6,000 zeroed weights under an 8-bit symmetric scale, compute the step-to-median ratio and assess whether scale collapse is present.
*Deliverable:* a calculation and a verdict.
*Acceptance criteria:* ratio correct (0.9/127/0.05 = 0.142); the 15% zeroed fraction interpreted; remedy named.
*Time:* 45 minutes.

**Exercise 6 — STE (analysis, 30 minutes).**
Explain why the derivative of rounding is zero almost everywhere, why the straight-through estimator is used anyway, and what could go wrong if the estimator were removed.
*Deliverable:* a 250-word explanation.
*Acceptance criteria:* mentions zero gradient; explains the identity substitution; states that training would not update quantized weights.
*Time:* 30 minutes.

**Exercise 7 — Evaluation (design, 50 minutes).**
Design the evaluation table for a quantized model, including accuracy, calibration, subgroups, latency, and energy. State the threshold rule and the confidence interval method.
*Deliverable:* a table template and a protocol.
*Acceptance criteria:* threshold fixed before evaluation; subgroups pre-specified; intervals reported; target measurement included.
*Time:* 50 minutes.

**Exercise 8 — Weight-only versus W8A8 (critique, 35 minutes).**
A team quantizes only the weights of a compute-bound convolutional model and measures no latency improvement. Explain why, and state the condition under which weight-only quantization would help.
*Deliverable:* a 300-word explanation.
*Acceptance criteria:* distinguishes bytes from MACs; names a weight-bound layer; states the memory benefit that remains.
*Time:* 35 minutes.

## 13. Self-check answers

**A1. Affine parameters for Exercise 1.**
Range = 0.82 - (-0.35) = 1.17; S = 1.17 / 255 = 0.00458824; Z = round(0.35 / 0.00458824) = round(76.28) = 76. Check q(0.82) = round(178.72) + 76 = 179 + 76 = 255. q(-0.35) = round(-76.28) + 76 = -76 + 76 = 0. Symmetric: amax = 0.82; S = 0.82 / 127 = 0.00645669. The symmetric step is 40.7% coarser, and the negative range is underused.

**A2. Error at 0.10.**
Affine: q = round(0.10 / 0.00458824) + 76 = round(21.79) + 76 = 22 + 76 = 98; x_hat = 0.00458824 × (98 - 76) = 0.100941; error +0.000941. Symmetric: q = round(0.10 / 0.00645669) = round(15.49) = 15; x_hat = 0.096850; error -0.003150. For the three-value tensor, affine SQNR is higher because the range is used fully; the exact value depends on the reconstructions and is on the order of 40 dB, versus about 35 dB for symmetric.

**A3. Metadata for Exercise 3.**
Weights: 256 × 576 = 147,456. Per-tensor: 147,456 + 5 = 147,461 bytes. Per-channel: 147,456 + 256 × 5 = 148,736 bytes, metadata 1,280 bytes (0.87%). Per-group of 32: 147,456 / 32 = 4,608 groups × 5 = 23,040 bytes of metadata, total 170,496 bytes (15.6%). Per-channel is the usual choice: the metadata is under 1% and the accuracy gain over per-tensor is material when channel ranges vary.

**A4. Why per-channel quantization is symmetric in most runtimes.**
Because the zero-point term can be folded into the accumulation, and a per-channel zero point would require a per-channel correction in the inner loop. Symmetric weights also simplify the kernel and reduce metadata. The cost is range utilization, which is small for zero-mean weight distributions.

**A5. Scale collapse verdict.**
Step = 0.9 / 127 = 0.0070866. Step / median = 0.0070866 / 0.05 = 0.142, or 14.2%. With 6,000 of 40,000 weights zeroed (15%), the diagnostic is positive: the step is above 10% of the median and a substantial fraction of weights is lost. The remedy is per-channel quantization or clipping, with per-channel preferred if the ranges vary by channel.

**A6. Straight-through estimator.**
Rounding is a step function, so its derivative is zero wherever it is defined and undefined at the jumps. A network trained with a hard quantizer in the forward pass would receive no gradient through the quantization operation. The STE substitutes the identity for the derivative, so the gradient flows as if quantization were a pass-through. Removing it would leave the quantized weights untrained, and QAT would reduce to PTQ with extra steps.

**A7. Why a quantized model can have better ECE.**
Quantization can smooth extreme probabilities, reducing overconfidence and lowering ECE, while also reducing sensitivity. A better calibration metric does not imply a better model; the two must be evaluated together, and the operating threshold must be checked.

**A8. Why weight-only quantization does not speed up a compute-bound model.**
Because the arithmetic count is unchanged and the layer is limited by MAC throughput. Weight-only quantization reduces bytes, which helps memory-bound layers and reduces model size, but a compute-bound convolution gains nothing.

**A9. Rubric for Exercise 4 (calibration plan).**
A strong plan names each site and sensor revision, specifies the number of windows per stratum, includes noise and artifact tails, defines a disjoint audit set, and describes versioning. A weak plan gives a total count with no stratification and no tail coverage.

**A10. Rubric for Exercise 7 (evaluation design).**
A strong design fixes the threshold rule before seeing the quantized results, pre-specifies subgroups and intervals, reports saturation rate as a diagnostic, and includes target latency and energy. A weak design reports overall accuracy at a re-tuned threshold and calls it equivalent.

## 14. Interview questions

**Q1. What is the difference between affine and symmetric quantization?**
A strong answer defines the scale and zero point, explains that affine uses the range more fully while symmetric simplifies arithmetic, and notes that weights are usually symmetric and activations asymmetric. A weak answer says affine is "more accurate" without the trade-off.

**Q2. Why is per-channel quantization better than per-tensor?**
A strong answer explains that channel ranges vary, a shared scale collapses small channels, and per-channel recovers resolution at a metadata cost. A weak answer says it is "more precise" without the mechanism.

**Q3. What is calibration and why does the calibration set matter?**
A strong answer distinguishes it from probability calibration, explains that activation ranges come from data, and describes the bias introduced by an unrepresentative set. A weak answer says calibration is "choosing the scale".

**Q4. How would you detect a bad quantization?**
A strong answer names the step-to-median ratio, distinct level count, per-channel SQNR, saturation rate, subgroup results, and calibration shift. A weak answer checks only overall accuracy.

**Q5. What is the straight-through estimator?**
A strong answer explains the zero gradient of rounding, the identity substitution, and the bias it introduces. A weak answer calls it "a trick to make gradients work" without the consequence.

**Q6. When would you use QAT instead of PTQ?**
A strong answer uses QAT when PTQ misses the requirement, especially at low bit width or with outliers, and notes the training cost and change-control burden. A weak answer always uses QAT because it is better.

**Q7. Why can activation clipping improve MAE but worsen RMSE?**
A strong answer explains that clipping reduces the scale and improves bulk resolution but saturates the tail, and that the squared error penalizes the saturated values heavily. A weak answer assumes clipping always helps.

**Q8. What is zero-point drift?**
A strong answer explains that the zero point represents real zero within a calibrated range, and that a distribution shift can move the bulk outside the range, causing saturation. A weak answer confuses it with scale drift.

**Q9. Why is INT8 not always faster?**
A strong answer names kernel availability, memory-bound layers, fallback, and layout. A weak answer assumes lower precision implies lower latency.

**Q10. How does quantization interact with calibration of the clinical model?**
A strong answer says quantization shifts probabilities, which can change the operating threshold, sensitivity, and alarm rate, and that calibration must be re-evaluated at the fixed threshold. A weak answer treats accuracy as the only metric.

**Q11. What records does a quantized model need?**
A strong answer lists the precision plan, calibration set, range method, scales, QAT configuration, evaluation, target measurements, and change classification. A weak answer lists only the model file.

**Q12. When would you reject a quantized model?**
A strong answer rejects on subgroup harm, calibration shift at the threshold, saturation on relevant inputs, absent target speedup, or an uncovered regulatory change. A weak answer rejects only on overall accuracy.

## 15. Key terms

- **Accumulator** — wider integer register for dot products.
- **Affine quantization** — quantization with a scale and zero point.
- **AWQ** — activation-aware weight quantization.
- **Bias correction** — compensating for the mean shift after quantization.
- **Bit width** — bits per quantized value.
- **Calibration** — selecting ranges from data.
- **Clipping error** — error from values outside the representable range.
- **Codebook** — table of representable values.
- **Dequantization** — mapping integers back to real values.
- **Dynamic quantization** — scales computed at runtime.
- **Entropy calibration** — histogram-based range selection.
- **Fake quantization** — quantize-dequantize in the forward pass.
- **GPTQ** — post-training weight quantization using second-order information.
- **Granularity** — how many values share a scale.
- **Group size** — number of weights sharing a scale.
- **INT8** — 8-bit integer format.
- **INT4** — 4-bit integer format.
- **KV cache quantization** — quantizing stored attention keys and values.
- **Min-max calibration** — using observed extremes as the range.
- **MSE calibration** — choosing the range to minimize squared error.
- **Outlier** — value far from the bulk of the distribution.
- **Overflow** — accumulator exceeding its range.
- **Packing** — storing multiple low-bit values per byte.
- **Per-channel** — one scale per output channel.
- **Per-group** — one scale per group of weights.
- **Per-tensor** — one scale per tensor.
- **Per-token** — one scale per input token.
- **Percentile calibration** — clipping a tail percentile.
- **PTQ** — post-training quantization.
- **QAT** — quantization-aware training.
- **QDQ** — QuantizeLinear-DequantizeLinear representation.
- **QOperator** — operator-based quantized representation.
- **Quantization** — mapping real values to integers.
- **Requantization** — converting accumulators back to a lower precision.
- **Rounding error** — error from mapping to the nearest code.
- **Saturation** — fraction of values clipped.
- **Scale** — real units per integer step.
- **Scale collapse** — outlier-dominated scale leaving few levels.
- **SmoothQuant** — migrating activation outliers to weights.
- **SQNR** — signal-to-quantization-noise ratio.
- **Static quantization** — scales fixed from calibration.
- **STE** — straight-through estimator.
- **Symmetric quantization** — zero point fixed at zero.
- **W4A16** — 4-bit weights, 16-bit activations.
- **W8A8** — 8-bit weights and activations.
- **Weight-only quantization** — weights quantized, activations float.
- **Zero point** — integer code representing real zero.
- **Zero-point drift** — zero point no longer aligned with real zero after a shift.
- **Weight clustering** — grouping weights into a shared codebook; a form of weight sharing, not quantization.

## 16. Sources

All URLs accessed 25 October 2026.

**Foundational papers**
- Jacob, B., et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", 2017/2018. https://arxiv.org/abs/1712.05877
- Nagel, M., et al., "A White Paper on Neural Network Quantization", 2021. https://arxiv.org/abs/2106.08295
- Krishnamoorthi, R., "Quantizing deep convolutional networks for efficient inference", 2018. https://arxiv.org/abs/1806.08342

**Large-model quantization**
- Dettmers, T., et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", 2022. https://arxiv.org/abs/2208.07339
- Xiao, G., et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", ICML 2023. https://arxiv.org/abs/2211.10438
- Frantar, E., et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", 2022. https://arxiv.org/abs/2210.17323
- Lin, J., et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", 2023. https://arxiv.org/abs/2306.00978

**Runtime specifications**
- Google AI Edge, "LiteRT 8-bit quantization specification". https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec
- ONNX Runtime, "Quantize ONNX models". https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- NVIDIA, "Working with Quantized Types", TensorRT Developer Guide. https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-tensorrt-developer-guide/work-quantized-types.html
- NVIDIA, "Quantization Workflows", TensorRT documentation. https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-workflows.html
- Arm, "CMSIS-NN". https://arm-software.github.io/CMSIS-NN/latest/
- ggml-org, "Tensor Encoding Schemes", llama.cpp wiki. https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes

**Standards and regulation**
- IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019. https://www.iso.org/standard/72704.html
- FDA, PCCP final guidance, 3 December 2024. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- MDCG 2025-6. https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en

> **Note:** The Section 6 tensor and results are constructed for teaching. LLM.int8() and SmoothQuant figures are the authors' measurements on their hardware.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md`](../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md)
- Previous topic: [`04_knowledge-distillation.md`](04_knowledge-distillation.md)
- Next topic: [`06_pruning-and-sparsity.md`](06_pruning-and-sparsity.md)
- Related topic: [`03_the-compression-family.md`](03_the-compression-family.md)
- Related topic: [`07_inference-optimization.md`](07_inference-optimization.md)
- Lab: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab directory: [`../../labs/`](../../labs/)
