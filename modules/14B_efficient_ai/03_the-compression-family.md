# The Compression Family

> Module 14B — Efficient AI and On-Device Inference
> Topic 03 of 11
> Estimated study time: 3 hours
> Prerequisites: 01, 02
> Version 2026.3

## 1. Why this topic exists

There are at least eight named techniques for making a neural network cheaper, and the names are used as if they were interchangeable. A team says it will "compress the model" and means quantization. A vendor says it supports "sparsity" and means a 2:4 pattern on one accelerator. A paper reports a 40× compression ratio and means 40× fewer parameters, not 40× lower latency. The vocabulary hides decisions that have different costs, different hardware requirements, and different regulatory consequences.

The purpose of this topic is to make the family legible. Each member removes something different:

- Distillation removes capacity, by training a smaller model to imitate a larger one.
- Quantization removes precision, by representing numbers with fewer bits.
- Pruning removes connections or structures, by setting weights or channels to zero.
- Low-rank factorization removes rank, by replacing a matrix with a product of smaller matrices.
- Weight sharing removes distinct values, by clustering weights into a codebook.
- Neural architecture search removes design uncertainty, by searching for an efficient architecture directly.
- Operator fusion removes memory traffic, by combining graph nodes without changing arithmetic.
- Sparsity-aware execution removes work at runtime, by skipping zeros the hardware can detect.

These are not variants of one idea. They act on different resources, they compose in different orders, and they fail in different ways. A team that treats them as a menu of equivalent "compression" options will pick the one that is easiest to run and discover that it does not address the binding resource.

Three industry problems follow.

**The method-resource mismatch.** A model that is memory-bound on weight bytes benefits from quantization and low-rank factorization, not from pruning. A model that is compute-bound benefits from distillation and structured pruning, not from weight-only quantization. A model that is launch-overhead-bound benefits from fusion, not from any of them. Choosing a method without knowing the binding resource is a guess.

**The hardware-support gap.** Unstructured pruning can remove 90% of weights and produce a model that runs at exactly the same speed, because dense kernels multiply by zero as happily as by any other number. 2:4 sparsity can double throughput, but only on hardware with sparse tensor cores, only for supported layer shapes, and only if the pattern is enforced during training. A method's benefit is a property of the method plus the hardware plus the runtime.

**The composition trap.** Methods interact. Pruning before quantization changes the weight distribution that quantization must represent. Quantization before pruning makes the pruning decision on values that have already lost precision. Distillation after pruning requires retraining a student whose architecture has already been constrained. The order determines the result, and the wrong order can cost several percentage points of accuracy for no additional gain.

The good news is that the family is small enough to reason about systematically. Once you know the binding resource, the candidate methods narrow to two or three, the hardware check eliminates one, and the composition order follows from a small set of rules. This topic teaches that reasoning.

### 1.1 The family at a glance

| Method | What it removes | Parameters | Weight bytes | MACs | Hardware feature needed |
|---|---|---|---|---|---|
| Distillation | Architecture capacity | Strong ↓ | Strong ↓ | Strong ↓ | None |
| Quantization | Numeric precision | None | Strong ↓ | Moderate ↓ | Integer kernels |
| Structured pruning | Channels or blocks | Strong ↓ | Strong ↓ | Strong ↓ | None |
| Unstructured pruning | Individual weights | Very strong ↓ | Moderate ↓ | None | Sparse kernels |
| 2:4 sparsity | Half the weights, patterned | Strong ↓ | Moderate ↓ | Strong ↓ | Sparse tensor cores |
| Low-rank | Matrix rank | Moderate ↓ | Moderate ↓ | Weak ↓ | Factored-form kernels |
| Weight sharing | Distinct values | None | Strong ↓ | None | Codebook decode |
| Fusion | Intermediate tensors | None | Moderate ↓ | None | Fused operators |
| NAS | The architecture itself | Variable | Variable | Variable | None |

The table answers the first question of any compression review: which column is binding? If the answer is MACs, the candidates are distillation, structured pruning, 2:4 sparsity, and a cheaper architecture. If the answer is weight bytes, the candidates are quantization, low-rank, and weight sharing. If the answer is neither, the problem may be in the graph, the runtime, or the system, and no compression method will fix it.

### 1.2 The three claims that need separate evidence

A compression result contains three claims, and they are often conflated.

- **A storage claim:** the model occupies fewer bytes on flash or disk. Evidence is a file size and a format description.
- **A throughput claim:** the model performs fewer operations or moves fewer bytes per inference. Evidence is an operation count and a traffic count, plus a runtime that exploits them.
- **A latency claim:** the model responds faster in the deployed system. Evidence is a target measurement at a stated percentile, batch, and thermal condition.

A paper can establish a storage claim with a file size, a throughput claim with a profiler, and a latency claim only with a system measurement. A vendor can establish a storage claim easily, a throughput claim with careful benchmarking, and a latency claim only for the exact configuration it tested. When a review conflates the three, the result is a deployment that saves flash and misses its deadline.

### 1.3 What "compression" is not

- It is not a single knob. There is no "compress" flag that selects the right method for a model.
- It is not lossless. Every method in the family changes outputs, even if the change is small.
- It is not free at the system level. A smaller model can be slower if it uses an unsupported operator or a worse memory layout.
- It is not a substitute for architecture choice. A model that is wrong for the device cannot be compressed into being right.

The rest of this topic is about choosing among the methods that remain after these four statements are accepted.

A final framing: the family is a set of levers, and every lever has a fulcrum. The fulcrum is the binding resource. Pull the wrong lever and the model gets smaller without getting faster; pull the right one and the budget moves. Finding the fulcrum is the work of Topics 01 and 02, and this topic assumes it has been done.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a compression method by the resource it reduces and the hardware feature it requires.
2. **Select** a candidate set of methods from a profiling result and a constraint ledger.
3. **Compute** the memory and operation count after distillation, structured pruning, unstructured pruning, and low-rank factorization.
4. **Explain** why unstructured pruning can leave latency unchanged and state the conditions under which it does not.
5. **Order** a composition of methods and justify the order from their interactions.
6. **Estimate** a compound gain and identify when gains are not multiplicative.
7. **Audit** a vendor or paper claim for whether the reported ratio is parameters, bytes, MACs, or latency.
8. **Document** a compression pipeline so that each stage's contribution can be attributed.

| Outcome | Practised in | Evidenced by |
|---:|---|---|
| 1 | Exercise 1 | A classification table with hardware requirements |
| 2 | Exercise 2 | A candidate set tied to a profile |
| 3 | Exercise 3 | Correct parameter and byte counts per method |
| 4 | Exercise 4 | A latency argument with a hardware condition |
| 5 | Exercise 5 | An ordered pipeline with a rationale |
| 6 | Exercise 6 | A compound estimate with a stated interaction |
| 7 | Exercise 7 | A claim audit with the missing quantity named |
| 8 | Exercise 8 | A stage-by-stage attribution table |

## 3. The core mental model

Think of the model as four budgets: parameters, bytes moved, arithmetic operations, and graph structure. Each method draws down one or more budgets, and each leaves a signature.

```text
                     parameters   bytes moved   MACs   graph nodes
   Distillation           ↓↓           ↓↓         ↓↓        ↓
   Quantization            –           ↓↓↓        ↓         –
   Structured pruning     ↓↓           ↓↓         ↓↓        ↓
   Unstructured pruning   ↓↓↓           –          –        –
   Low-rank               ↓↓           ↓↓         ↓         –
   Weight sharing         ↓↓           ↓↓         –        –
   Fusion                  –            ↓↓         –        ↓↓
   NAS                    ↓↓           ↓↓         ↓↓        ↓↓

   ↓ = reduces, – = no direct effect, ↓↓ = strong, ↓↓↓ = very strong
```

The table is the first thing to consult. A method that does not reduce the binding budget cannot solve the binding problem, no matter how large its headline ratio.

### 3.1 Two axes: what is removed and what the hardware can exploit

Every method has an **algorithmic effect** and a **hardware effect**. The algorithmic effect is what changes in the model: fewer parameters, fewer bits, fewer nonzeros. The hardware effect is what the runtime can actually skip.

| Method | Algorithmic effect | Hardware effect | Requirement |
|---|---|---|---|
| Quantization | Fewer bits per value | Less memory traffic; faster integer arithmetic | Integer kernels at the target precision |
| Distillation | Smaller architecture | Fewer MACs, less memory | None beyond the new model |
| Structured pruning | Fewer channels or blocks | Fewer MACs, smaller tensors | Dense kernels work; no special support needed |
| Unstructured pruning | Fewer nonzero weights | Storage only, unless sparse kernels exist | Sparse kernels or sparse tensor cores |
| 2:4 sparsity | Two nonzeros per four | Up to 2× math throughput | Ampere-class sparse tensor cores or equivalent |
| Low-rank | Fewer parameters in a matrix | Fewer MACs and bytes if the factored form is faster | Kernels that handle the two smaller matrices |
| Weight sharing | Fewer distinct values | Less storage; decode cost at runtime | Codebook lookup support |
| Fusion | Fewer intermediate tensors | Less memory traffic, fewer launches | Runtime support for the fused operator |
| NAS | A different architecture | Whatever the architecture offers | Search budget and a training pipeline |

The distinction explains a common disappointment. A paper reports 90% unstructured sparsity and a 10× storage reduction. The team implements it, measures no latency change, and concludes that the paper was wrong. It was not: the paper measured storage, and the deployment needed latency. The algorithmic effect was real; the hardware effect was absent.

### 3.2 The composition graph

Methods can be composed, but not arbitrarily. The following graph shows the compositions that are common in practice and the reason for each edge.

```text
   Train large model
          │
          ├──► Distill ──────────────┐
          │        (capacity ↓)      │
          │                          ▼
          ├──► Structured prune ──► Fine-tune ──► Quantize (PTQ/QAT)
          │        (channels ↓)                     (precision ↓)
          │
          ├──► Unstructured prune ──► Sparse-aware hardware? ──► Deploy
          │        (weights ↓)              │ no
          │                                 └──► storage only
          │
          └──► Low-rank factorize ──► Fine-tune ──► Quantize
                   (rank ↓)
```

Three rules govern the edges.

**Rule 1: structural decisions precede precision decisions.** Pruning changes which weights exist; quantization changes how they are represented. Deciding structure first means the quantizer sees the final distribution. Deciding precision first means the pruner operates on rounded values and may remove a weight that would have survived at higher precision.

**Rule 2: distillation precedes the other methods when the architecture is too large.** Distillation changes the architecture. Pruning and quantization refine a fixed architecture. Applying distillation after pruning means throwing away the pruned structure and starting again.

**Rule 3: fine-tuning follows any method that changes the loss surface materially.** Structured pruning and low-rank factorization change the function computed, not just its representation. A short fine-tune recovers most of the loss. Quantization with QAT includes the fine-tune; PTQ does not.

### 3.3 Gains are multiplicative only when the resources are independent

A common error is to multiply the headline ratios: 4× from quantization, 2× from pruning, 2× from distillation, therefore 16×. The product is valid only if each method reduces a resource that the others leave unchanged and the runtime's bottleneck is that resource throughout.

In practice:

- Quantization reduces bytes by 4× but MACs by 0×. If the model is compute-bound, the latency gain is not 4×.
- Structured pruning reduces MACs and bytes together, so combining it with quantization can be closer to multiplicative for a memory-bound model.
- Unstructured pruning reduces storage but not MACs, so multiplying it with a quantization gain overstates the result.
- Distillation reduces everything, so it composes with the others but changes the baseline against which their gains are measured.

The honest approach is to measure after each stage and report the incremental gain, not to multiply ratios.

### 3.4 The accuracy budget is shared

Every method spends accuracy. If the clinical floor allows a 1.5 percentage point loss and distillation spends 0.8, pruning spends 0.6, and quantization spends 0.6, the pipeline is at 2.0 and fails. The budget must be allocated before the work begins.

| Stage | Planned loss (pp) | Measured loss (pp) | Remaining budget (pp) |
|---|---:|---:|---:|
| Distillation | 0.6 | 0.8 | 0.7 |
| Structured pruning | 0.4 | 0.6 | 0.1 |
| Quantization | 0.3 | 0.6 | -0.5 |
| **Total** | **1.3** | **2.0** | **-0.5** |

The table is the single most useful project-management artifact in a compression programme, because it makes the overrun visible at the stage where it happens rather than at the end. A stage that exceeds its allocation triggers a decision: relax the stage, recover elsewhere, or revisit the clinical floor.

### 3.5 A worked classification

Consider a transformer encoder with 12 layers, a hidden dimension of 768, and a sequence length of 256, deployed on a phone system-on-chip with a neural processing unit.

- **Parameters:** about 85 million, dominated by the attention and feed-forward matrices.
- **Bytes per inference:** dominated by weights on the first token, then by activations and the key-value cache on later tokens.
- **MACs:** dominated by the feed-forward matrices (roughly two thirds) and attention (roughly one third).
- **Graph:** many small operators, layer normalization, residual adds, and a softmax.

The classification:

| Budget | Observation | Binding? | Candidate methods |
|---|---|---|---|
| Weight bytes | 85 M parameters, about 85 MB at INT8 | Yes, on first load | Weight-only quantization, low-rank |
| Activation bytes | Sequence 256, hidden 768, several MB | Yes, during decode | Activation quantization, KV cache quantization |
| MACs | About 35 GMAC per sequence | Yes, on prefill | Distillation, structured pruning |
| Graph | Dozens of small ops per layer | Yes, without fusion | Fusion, graph compilation |

All four budgets are binding at different phases, which is typical for transformers and unusual for small convolutional models. The pipeline must therefore be evaluated per phase: a model that meets the prefill budget may miss the decode budget, and a model that meets both may miss the first-load budget.

This is why a single "compression ratio" is uninformative for a transformer. The model has at least three deployment phases, each with a different binding resource, and the methods that help one may not help another.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Compression ratio | Original size divided by compressed size | Applied to latency without qualification | Unitless |
| Parameter | A learned weight or bias | Confused with MAC | Count |
| Sparsity | Fraction of weights equal to zero | Assumed to imply speedup | Proportion |
| Structured pruning | Removal of whole channels, heads, or blocks | Called "pruning" without qualification | Method |
| Unstructured pruning | Removal of individual weights | Assumed to speed up dense hardware | Method |
| 2:4 sparsity | At most two nonzeros in every group of four | Confused with 50% unstructured sparsity | Pattern |
| N:M sparsity | At most N nonzeros in every group of M | Assumed supported by all accelerators | Pattern |
| Lottery ticket hypothesis | Dense networks contain sparse trainable subnetworks | Treated as a deployment recipe | Hypothesis |
| Distillation | Training a smaller model on a larger model's outputs | Confused with fine-tuning | Method |
| Teacher | The model providing soft targets | Assumed to be better calibrated | Model |
| Student | The model being trained | Assumed to be smaller in every dimension | Model |
| Temperature | Softmax scaling in distillation | Confused with annealing temperature | Unitless |
| Low-rank factorization | Replacing a matrix with a product of smaller matrices | Assumed to reduce latency automatically | Method |
| Rank | Number of independent rows or columns retained | Chosen without a reconstruction check | Integer |
| Weight sharing | Clustering weights into a codebook | Confused with quantization | Method |
| Codebook | Table of shared values | Assumed free to decode | Table |
| Neural architecture search (NAS) | Automated search over architectures | Treated as a compression method | Method |
| Operator fusion | Combining graph nodes to reduce traffic | Confused with quantization | Graph transform |
| Constant folding | Precomputing constant subgraphs | Confused with fusion | Graph transform |
| Structured sparsity | Sparsity at block or channel granularity | Confused with 2:4 | Pattern |
| Sparse kernel | A kernel that skips zeros | Assumed present when sparsity is present | Implementation |
| Dense kernel | A kernel that multiplies all values, including zeros | Assumed to exploit sparsity | Implementation |
| Compression-aware training | Training with the compression operator in the loop | Confused with post-hoc compression | Method |
| Post-training compression | Applying a method to a trained model | Assumed lossless | Method |
| Fine-tuning | Short retraining after a structural change | Confused with full retraining | Method |
| Recovery | Accuracy regained by fine-tuning | Assumed to be complete | Proportion |
| Headroom | Distance between current and allowed resource use | Confused with the compression ratio | Mixed units |
| Knee | Point where more compression costs disproportionate accuracy | Ignored in favour of a target ratio | Point |
| Payload | The bytes that must be stored or moved per inference | Confused with parameter count | Byte |
| Index overhead | Bytes needed to describe sparsity | Ignored in sparse storage estimates | Byte |
| Metadata overhead | Scales, zero points, codebooks | Ignored in quantization estimates | Byte |
| Activation | Intermediate tensor produced at runtime | Counted as a parameter | Tensor |
| Weight-only quantization | Quantizing weights but not activations | Assumed to reduce latency universally | Method |
| Weight and activation quantization | Quantizing both | Called "full quantization" | Method |
| Mixed precision | Different precisions for different tensors | Confused with dynamic quantization | Method |
| Dynamic quantization | Computing scales at runtime | Confused with QAT | Method |
| Static quantization | Fixing scales from calibration | Confused with PTQ | Method |
| QAT | Quantization-aware training | Assumed to be a runtime feature | Method |
| PTQ | Post-training quantization | Assumed to require no data | Method |
| Straight-through estimator (STE) | Gradient approximation through rounding | Assumed exact | Method |
| Knowledge | What the teacher transfers | Treated as a measurable quantity | Informal |
| Capacity gap | Difference in expressive power between teacher and student | Ignored when choosing student size | Model property |

Two terms are especially prone to misuse in procurement documents.

**"Compression ratio"** must be qualified by what is being counted. A 4× parameter ratio, a 4× byte ratio, and a 4× latency ratio are different claims. A model with 4× fewer parameters in INT8 has roughly 4× fewer weight bytes, but its activations are unchanged and its MAC count is unchanged, so its latency gain may be 1× on a compute-bound device.

**"Sparsity"** must be qualified by pattern and by hardware. Fifty percent 2:4 sparsity on an Ampere tensor core can double math throughput for supported layers. Fifty percent unstructured sparsity on a Cortex-M4 changes nothing at runtime. The same word describes two different deployment outcomes.

### 4.1 Symbols used in the arithmetic

The worked example and exercises use the following symbols consistently.

| Symbol | Meaning | Unit |
|---|---|---|
| P | Parameter count | count |
| M | MACs per inference | count |
| B | Bytes stored or moved per inference | byte |
| S | Sparsity, the fraction of weights equal to zero | proportion |
| f | Fraction retained after a transformation | proportion |
| r | Rank in a low-rank factorization | integer |
| C | Number of output channels | count |
| k | Metadata bytes per channel | byte |
| L | Latency | ms |
| E | Energy per inference | mJ |
| T | Effective throughput | MAC/s |

The relationships used repeatedly:

```text
bytes after quantization      B_q = P × b / 8 + C × k
parameters after pruning      P_p = P × (1 - S)
parameters after factorization P_lr = rows × r + r × cols
MAC reduction from channels   M_p ≈ M × (1 - f_c) × (1 - f_in)
```

The last relationship is approximate because removing output channels and removing input channels have different effects on the next layer, and the exact reduction depends on the layer order.

## 5. The mechanics, step by step

The procedure selects and composes methods. It assumes the profiling and constraint work from Topics 01 and 02 is complete.

### Step 1. Read the binding resource from the profile

**Input:** The per-layer classification from Topic 02, the constraint ledger, and the current margins.

**Output:** A ranked list of budgets that must be reduced, with the amount of reduction required.

**Rationale:** The method follows the resource. Without this step, selection is a preference, not an engineering decision.

Compute the required reduction as a ratio:

```text
required_reduction = current_use / allowed_use
```

For example, if peak SRAM is 300 KiB and the limit is 256 KiB, the required reduction in activation memory is 300 / 256 = 1.17, or 15%. If the model is 1,600 KiB of flash and the limit is 512 KiB, the required weight-byte reduction is 3.13×.

**Branch point:** If no budget is exceeded, stop. Compression is not required, and the cheapest correct decision is to ship the uncompressed model with the margin documented. This happens less often than teams expect, and it is always cheaper than compressing.

**Edge case:** A budget can be exceeded only in the tail. A model that fits at p50 but not at p99 under thermal throttling needs a latency margin, not a smaller model. Diagnose before compressing.

### Step 2. Translate the required reduction into candidate methods

**Input:** The ranked budgets and the per-layer profile.

**Output:** Two to four candidate methods, each with a predicted effect on the binding budget.

**Rationale:** Different methods draw down different budgets. Choosing a method that does not touch the binding budget wastes a quarter and produces a negative result that is easy to misread as "compression does not work here".

The mapping is mechanical:

| Binding budget | First-choice methods | Methods that will not help |
|---|---|---|
| Weight bytes in flash | Quantization, weight sharing, low-rank | Unstructured pruning on dense hardware |
| Activation bytes in SRAM | Activation quantization, fusion, smaller input window | Weight-only quantization |
| MACs on a compute-bound layer | Distillation, structured pruning, cheaper operators | Weight-only quantization |
| Launch overhead with many small ops | Fusion, graph compilation, operator replacement | Any precision change |
| Bandwidth on a memory-bound accelerator | Quantization of the dominant tensor, fusion, layout | Distillation if MACs are not the limit |
| Energy dominated by radio | Move the decision on-device; send events not raw data | Model compression |
| Thermal limit | Lower sustained power: fewer MACs, lower clock, better cooling | Anything that increases sustained power |

**Branch point:** If the table returns two methods with similar predicted effects, prefer the one with the lower integration cost. A method that requires a new runtime delegate is more expensive than one that uses an existing kernel, even if its theoretical gain is larger.

**Edge case:** A method can help a budget it does not directly reduce. Quantization reduces bytes and, on some hardware, also reduces energy per MAC by using a more efficient integer pipeline. The secondary effect should be measured, not assumed.

### Step 3. Check hardware and runtime support

**Input:** The candidate methods and the target runtime's operator, precision, and sparsity support.

**Output:** A feasibility verdict per method: supported, partially supported, or unsupported.

**Rationale:** The hardware effect is what the patient experiences. An algorithmic effect without a hardware effect is a storage change, which may still be valuable but is a different claim.

The checks are specific:

- **Precision.** Does the runtime have integer kernels for this operator at this bit width? Does the accelerator support the activation precision, not only the weight precision?
- **Sparsity.** Does the accelerator support the exact pattern (2:4, block, unstructured) for this layer type and shape? Is the sparse path selected automatically, or does it require a flag?
- **Layout.** Does the factored or pruned form preserve the layout the kernel expects? A low-rank factorization that produces two transposed matrices may be slower than the original.
- **Operator coverage.** Does the fused operator exist? Is the replacement operator supported?
- **Toolchain.** Does the converter preserve the transformation, or does it fold, reorder, or reject it?

**Branch point:** If a method is partially supported, test it on a small representative layer before committing. Partial support often means "supported for convolution but not depthwise" or "supported at batch 1 but not batch 8", and the exception may be the layer that matters.

**Edge case:** A runtime can support an operator but not its quantized form. The model will load and run with a float fallback, producing correct outputs and terrible latency. The load succeeding is not evidence of support.

### Step 4. Choose the composition and the order

**Input:** The feasible methods.

**Output:** An ordered pipeline with a rationale for each edge.

**Rationale:** Order determines both accuracy and gain. The rules from Section 3.2 are the starting point; the specific model may add constraints.

A default order for a convolutional model:

1. Distillation, if the architecture is too large.
2. Structured pruning, if MACs are binding.
3. Fine-tuning to recover.
4. Quantization-aware training or post-training quantization.
5. Graph optimization at export.
6. Target-specific compilation.

A default order for a transformer with a tight memory budget:

1. Distillation or a smaller architecture.
2. Low-rank factorization of large linear layers, if the rank reduction is supported.
3. Weight-only quantization for memory, or weight and activation quantization for bandwidth.
4. KV cache quantization, evaluated separately.
5. Fusion and kernel selection at export.

**Branch point:** If two methods both target the same budget and their effects overlap, apply one and measure before applying the second. Applying both at once makes attribution impossible.

**Edge case:** Quantization-aware training must see the final architecture. If pruning happens after QAT, the quantization parameters are wrong for the pruned model and must be recomputed. This is one of the most common causes of an unexplained accuracy drop.

### Step 5. Estimate the compound effect, then measure each stage

**Input:** The pipeline and the baseline measurements.

**Output:** A per-stage table of parameters, bytes, MACs, latency, energy, accuracy, and calibration.

**Rationale:** Estimates guide the design; measurements decide it. The per-stage table attributes the result to each method, which is what a reviewer needs and what a future engineer needs when one stage must be removed.

The estimate uses the budgets from Step 1 and the method effects from Section 3. Apply them in order, recomputing the baseline after each stage:

```text
P_1 = P_0 × f_distill
M_1 = M_0 × f_distill
B_1 = B_0 × f_distill
P_2 = P_1 × f_prune
M_2 = M_1 × f_prune
...
```

**Branch point:** If a stage produces less than half its predicted gain, stop and diagnose before continuing. The cause is usually a mismatch between the algorithmic effect and the hardware effect, or an interaction with the previous stage.

**Edge case:** Accuracy is not multiplicative. A 1% loss at each of three stages can be 3% or 0.5%, depending on whether the errors are independent or correlated. Measure the composed model, not the product of the stage losses.

### Step 6. Evaluate accuracy, calibration, and subgroups after every stage

**Input:** The held-out set, the calibration audit set, and the operating threshold rule.

**Output:** A stage-by-stage table with confidence intervals.

**Rationale:** A compression pipeline can produce an acceptable final accuracy while hiding a large intermediate regression that was masked by a later stage's recovery. It can also produce an acceptable overall accuracy while a subgroup has crossed a clinical boundary.

The evaluation must include:

- Sensitivity and specificity at the operating threshold.
- Calibration (ECE, slope, Brier) on the held-out set.
- Subgroup results with confidence intervals.
- The threshold selected on validation data, not on the test set.
- The decision-curve net benefit, where the clinical decision supports it.

**Branch point:** If a subgroup degrades beyond its allocated loss while the overall metric is acceptable, treat the stage as failed. Aggregate recovery is not evidence of equitable recovery.

**Edge case:** A stage can improve calibration while reducing sensitivity, or the reverse. Neither is automatically acceptable; the clinical claim determines which matters.

### Step 7. Choose the minimal pipeline that meets the constraints

**Input:** The per-stage results and the constraint ledger.

**Output:** A chosen pipeline, with the contribution of each stage and the residual margin.

**Rationale:** Every stage adds risk, validation cost, and change-control burden. The best pipeline is the shortest one that meets the requirements, not the one with the highest compression ratio.

**Branch point:** If two pipelines meet the constraints, prefer the one with fewer stages and better-understood failure modes. A single quantization stage with a documented calibration set is easier to defend than a three-stage pipeline whose interactions are not fully characterized.

**Edge case:** A stage that is not needed for the current model may be needed for the next one. Record why it was omitted so that the decision can be revisited when the model grows.

### Step 8. Validate on target hardware

**Input:** The chosen pipeline, the target board, and the production runtime.

**Output:** Target latency, energy, memory, and thermal results, plus the accuracy results from the exported model.

**Rationale:** The exported model is not the trained model. Fusion, layout changes, and kernel selection can change both outputs and timing, and a model that passes in the training framework can fail after export.

**Branch point:** If the exported model's accuracy differs from the framework model by more than the quantization error predicts, the export changed semantics. Common causes are different rounding, fused activation approximations, and layout-dependent reductions.

**Edge case:** A model can be numerically correct and still fail, because its latency on the target exceeds the budget. Both checks are required.

### Step 9. Document the pipeline and its provenance

**Input:** All stage results, configurations, and calibration data.

**Output:** A pipeline record that a reviewer can audit.

**Rationale:** A compressed model is a chain of transformations. Without a record of each transformation, its parameters, and its data, the final model cannot be reproduced or defended.

The record includes:

- Baseline model hash and training data version.
- Teacher and student architectures for distillation, with temperature and alpha.
- Pruning pattern, sparsity, and schedule.
- Low-rank ranks and reconstruction error.
- Quantization scheme, granularity, bit width, calibration set, and range method.
- Export toolchain and runtime versions.
- Target hardware and measurement conditions.
- Accuracy, calibration, and subgroup results per stage.

**Branch point:** If any stage cannot be reproduced from the record, the model is not reproducible. Fix the record before submission, not after a question is raised.

**Edge case:** A pipeline that uses a random seed for pruning or calibration is reproducible only if the seed is recorded. Record it.

### Step 10. Classify the change and plan the evidence

**Input:** The pipeline record and the regulatory claim.

**Output:** A change classification and an evidence plan.

**Rationale:** Compression changes outputs. Whether that change is covered by an existing approval is a regulatory question that must be answered before the work is complete.

**Branch point:** If the change is covered by a predetermined change control plan (PCCP), verify that the protocol's acceptance criteria are met. If not, the change requires a submission or a redesign that stays within the approved envelope.

**Edge case:** A pipeline can be covered at one stage and not another. Quantization may be anticipated in a PCCP while distillation, which changes the architecture, is not. Classify each stage.

### 5.1 Decision table for method selection

| Binding resource | Hardware support present | Recommended first method | Fallback if accuracy fails |
|---|---|---|---|
| Weight bytes, compute-bound | INT8 kernels | Quantization (PTQ, per-channel) | QAT, then weight-only INT4 |
| Weight bytes, memory-bound | INT8 kernels | Quantization plus fusion | Low-rank factorization |
| Activation bytes | INT8 activation kernels | Activation quantization | Fusion and smaller window |
| MACs, dense layers | None needed | Structured pruning | Distillation |
| MACs, whole model | None needed | Distillation | Smaller architecture from NAS |
| Storage only | None | Unstructured pruning with sparse storage | Weight sharing |
| Storage and math | 2:4 sparse tensor cores | 2:4 pruning with sparse-aware training | Structured pruning |
| Launch overhead | Fused operators available | Fusion and graph compilation | Operator replacement |
| Energy, radio-dominated | None | Move decision on-device | Reduce transmission frequency |
| Thermal | None | Reduce sustained MACs | Lower clock with a longer deadline |

### 5.2 Composition decision table

| Situation | Order | Reason |
|---|---|---|
| Architecture too large and MACs binding | Distill, then quantize | Distillation changes the architecture; quantization refines it |
| MACs binding with a fixed architecture | Structured prune, fine-tune, quantize | Structure before precision |
| Weight bytes binding only | Quantize, then evaluate low-rank | Quantization is cheaper to integrate |
| Sparse hardware available | Prune to the supported pattern during training, then quantize | Pattern must be enforced before export |
| Very tight memory, transformer | Low-rank, then weight-only quantization | Rank reduction changes the weight matrix; quantization compresses it |
| Strict latency and strict accuracy | Quantize first, measure, then distill only if needed | Minimal pipeline; each stage adds risk |
| Regulated change with a narrow PCCP | Quantization only, if anticipated | Distillation and pruning may be outside the plan |

### 5.3 Edge cases across the family

**Outlier channels.** Quantization is the method most sensitive to outliers, but pruning interacts with them too: magnitude pruning may remove small weights that carry rare-class signal, while keeping a large outlier that contributes little. Diagnose with per-channel error, not aggregate SQNR.

**Per-tensor versus per-channel scales.** Per-channel quantization recovers most of the accuracy lost to per-tensor skew, at a metadata cost. If the runtime supports it, it is usually the first refinement to try. If the runtime does not, clipping or outlier separation are the alternatives.

**Activation clipping.** Clipping changes the activation distribution and can interact with pruning, because a pruned channel may have a different activation range. Recompute activation scales after any structural change.

**Operator fallback.** A pruned or factorized model may use operators the accelerator does not support, such as a gather for sparse indices or a batched matrix multiply for a factored layer. The fallback cost can exceed the gain.

**Unsupported operators.** Some compression outputs require runtime features that do not exist on the target: sparse matrix multiplication on an MCU, dynamic shapes for a factored layer, or a codebook lookup in a quantized kernel. Check before training, not after.

**Zero-point drift.** Any method that changes the weight distribution, including pruning and fine-tuning, can invalidate previously computed zero points. Recompute quantization parameters after every structural change.

**Calibration-set bias.** Calibration data must represent the deployment population. A pipeline that calibrates after pruning on data from one site will produce scales tuned to that site, and the error will be larger elsewhere.

**Thermal throttling.** A smaller model may run at a higher clock for the same power budget, which can change the thermal profile. Measure the compressed model's sustained behavior; it is not simply the baseline scaled down.

### 5.4 First-order estimates for each method

The following formulas give a first-order estimate. They are deliberately simple, and each one has a condition under which it is wrong.

| Method | Parameter estimate | Byte estimate | MAC estimate | Condition for validity |
|---|---|---|---|---|
| Distillation | Student as designed | P_s × b / 8 + C_s × k | Proportional to student size | Student designed for the task |
| PTQ | P unchanged | P × b / 8 + C × k | M × (integer speedup factor) | Integer kernels exist |
| Structured pruning | P × (1 - S_c) | P_p × b / 8 + C_p × k | M × (1 - S_c) × (1 - S_in) | Channels removable without accuracy collapse |
| Unstructured pruning | P × (1 - S) | Nonzeros + indices + pointers | M unchanged | Dense kernels only |
| 2:4 sparsity | P × 0.5 | P × 0.5 × b / 8 + metadata | M × 0.5 | Sparse tensor cores support the layer |
| Low-rank | rows × r + r × cols | P_lr × b / 8 + C_new × k | Proportional to P_lr | Factored form maps to fast kernels |
| Weight sharing | P unchanged | codebook + indices | M unchanged | Decode is cheaper than the memory saved |

The formulas are useful for eliminating options, not for predicting latency. A method whose first-order estimate cannot meet the budget will not meet it after tuning; a method whose estimate barely meets it needs a prototype before it can be trusted.

### 5.5 Composition checklist

Before committing to a pipeline, confirm each item.

- Every stage targets a budget that is actually binding at that point in the pipeline.
- The pipeline order follows structure-before-precision and distillation-before-refinement.
- Each stage's accuracy allocation is written down and measured.
- Each stage's hardware requirement is confirmed on the target runtime.
- The exported model, not only the framework model, has been measured.
- The calibration data is disjoint from the held-out data and represents the deployment population.
- The seed and tool versions for each stage are recorded.
- The change classification is known for each stage, not only for the pipeline as a whole.
- A fallback exists if a stage fails late.
- The minimal pipeline that meets the constraints has been identified and compared with the chosen one.

A pipeline that cannot answer all ten is not ready for a design review, regardless of its measured accuracy.

### 5.6 A compound estimate, one stage at a time

Suppose the baseline has 100,000 parameters, 10,000,000 MACs, and 400,000 bytes at FP32. The pipeline is distillation to 60% of the parameters, structured pruning to 70% of the student's channels, and INT8 quantization.

```text
Baseline:      P=100,000   M=10,000,000   B=400,000
Distillation:  P=60,000    M=6,000,000    B=240,000        (×0.60)
Pruning:       P=42,000    M=3,600,000    B=168,000        (×0.70)
Quantization:  P=42,000    M=3,600,000    B=42,000 + meta  (÷4)
```

The naive product of the ratios is 0.60 × 0.70 × 0.25 = 0.105, or a 9.5× reduction in bytes. The staged arithmetic gives 42,000 / 400,000 = 0.105, which agrees because each stage multiplies the previous result. The product is valid here because each stage reduces the same quantity multiplicatively.

It would not be valid for MACs if quantization gave a 3× kernel speedup. The MAC count would still be 3,600,000; the latency would improve by a hardware factor that is not a property of the model. Mixing model ratios with hardware ratios is the most common source of over-optimistic estimates.

Two further cautions:

- The pruning ratio applies to the student, not the original. Applying 0.70 to the original and 0.60 to the result gives the same product, but the intermediate numbers differ, and the accuracy allocation depends on which model the prune was tuned on.
- Metadata grows relative to the payload as the model shrinks. At 42,000 parameters and 72 channels, metadata is 360 bytes, which is 0.86% of the payload. At 4,200 parameters and 72 channels it would be 8.6%, and the 4× byte ratio would become 3.7×.

### 5.7 When to stop compressing

Compression has diminishing returns and increasing risk. The stopping conditions are:

- Every hard constraint is met with a margin that survives the confidence interval.
- The next stage would spend more of the accuracy budget than remains.
- The next stage's hardware requirement is not confirmed on the target.
- The next stage's change classification is uncertain or outside the approved plan.
- The engineering cost of the next stage exceeds the value of the margin it would create.

The last condition is the one teams underweight. A pipeline that gains 3 ms of latency at the cost of a new delegate, a new calibration set, and a new validation campaign is usually a worse product decision than a pipeline that ships with 3 ms less margin and a simpler evidence package.

## 6. Worked example

This is a constructed case. The model, the measurements, and the accuracy values are illustrative and chosen so that the arithmetic closes. No number here is a benchmark result.

### 6.1 The baseline

The ECG model from Topics 01 and 02 has 41,216 parameters, of which 32,768 are in a fully connected layer (64 inputs by 512 outputs) and 8,448 are in the convolutional stack. It performs about 4,096,000 MACs per inference. In FP32 it occupies 164,864 bytes of weights, runs at 138 ms p99 on the Cortex-M4F at 64 MHz, and consumes 3.40 mJ per inference at 3.0 V and 8.2 mA. Sensitivity is 0.942 and ECE is 0.021.

The requirement is an inference latency of at most 40 ms, a model of at most 64 KiB of flash, and a sensitivity floor of 0.925. The latency requirement applies to the model, not the full pathway; the full pathway has a 250 ms budget, of which the model may use 40 ms.

The baseline fails two of the three constraints before compression: 161.0 KiB exceeds the flash budget by 97 KiB, and 138 ms exceeds the latency budget by 98 ms. The accuracy constraint passes with 1.7 percentage points of margin. Compression must therefore recover a 2.5× flash reduction and a 3.45× latency reduction without spending more than 1.7 percentage points of sensitivity.

### 6.2 The five routes

**Route A: INT8 post-training quantization.** Parameters unchanged at 41,216. Bytes 41,216 + 600 metadata = 41,816. MACs unchanged at 4,096,000. Latency falls to 45.7 ms because the integer kernels perform about 3× more MACs per second than the float kernels on this core, and the layer remains compute-bound.

**Route B: Distillation to 26,000 parameters, then INT8.** The student has 26,000 parameters and performs 2,600,000 MACs, a 36.5% reduction in both. Bytes 26,000 + 390 = 26,390. Latency 2,600,000 / 89,600,000 = 29.0 ms.

**Route C: Structured pruning of 40% of channels, then INT8.** Parameters 41,216 × 0.60 = 24,730. MACs 4,096,000 × 0.55 = 2,252,800 (the MAC reduction exceeds the parameter reduction because channel removal also removes the input channels of the next layer). Bytes 24,730 + 360 = 25,090. Latency 2,252,800 / 89,600,000 = 25.1 ms.

**Route D: Unstructured pruning of 70% of weights.** Nonzeros 41,216 × 0.30 = 12,365. Storage in compressed sparse row format: 12,365 values at 1 byte, 12,365 column indices at 2 bytes, and 129 row pointers at 4 bytes = 12,365 + 24,730 + 516 = 37,611 bytes. MACs and latency are unchanged at 4,096,000 and 45.7 ms, because the dense kernels on this core have no sparse path.

**Route E: Low-rank factorization of the fully connected layer at rank 32.** The original 64×512 matrix has 32,768 weights. The factors have 64×32 + 32×512 = 2,048 + 16,384 = 18,432 weights, saving 14,336. Total parameters 26,880. MACs fall by 14,336 to 4,081,664, which is 0.35%, so latency is essentially unchanged at 45.6 ms. Bytes rise relative to a naive count because the factored form adds two new output dimensions to the metadata: 26,880 + 600 × 5 = 29,880.

The five routes are summarized below. Memory is weights only, INT8, with per-channel metadata.

| Route | Parameters | Weight bytes | KiB | MACs | Latency (ms) | Energy (mJ) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline FP32 | 41,216 | 164,864 | 161.0 | 4,096,000 | 138.0 | 3.395 |
| A: INT8 PTQ | 41,216 | 41,816 | 40.8 | 4,096,000 | 45.7 | 1.125 |
| B: Distill + INT8 | 26,000 | 26,390 | 25.8 | 2,600,000 | 29.0 | 0.713 |
| C: Structured prune + INT8 | 24,730 | 25,090 | 24.5 | 2,252,800 | 25.1 | 0.617 |
| D: Unstructured 70% | 12,365 | 37,611 | 36.7 | 4,096,000 | 45.7 | 1.125 |
| E: Low-rank rank 32 | 26,880 | 29,880 | 29.2 | 4,081,664 | 45.6 | 1.122 |

Energy is computed as 3.0 V × 8.2 mA × latency. For route A: 3.0 × 0.0082 × 0.0457 = 1.124 mJ. For route B: 3.0 × 0.0082 × 0.029 = 0.713 mJ.

### 6.3 Scale and zero point by hand

Route A quantizes the first convolution, whose 40 weights have a minimum of -0.061 and a maximum of 0.412, with a median absolute value of 0.0182.

```text
Affine:
  S = (0.412 - (-0.061)) / 255 = 0.473 / 255 = 0.00185490
  Z = round(0.061 / 0.00185490) = round(32.886) = 33
  q(0.412) = 222 + 33 = 255
  q(-0.061) = -33 + 33 = 0

Symmetric:
  S = 0.412 / 127 = 0.00324409
  Z = 0
```

The affine step is 0.001855 and the symmetric step is 0.003244. On this layer, affine yields 42.04 dB SQNR and symmetric 37.54 dB. The affine scheme exploits the asymmetry of the distribution, which is concentrated below zero except for one outlier.

### 6.4 Memory before and after

```text
FP32 baseline:        41,216 × 4 = 164,864 B = 161.0 KiB
Route A per-channel:  41,216 + 120 × 5 = 41,816 B = 40.8 KiB
Route B:              26,000 + 120 × 5 = 26,390 B = 25.8 KiB
Route C:              24,730 +  72 × 5 = 25,090 B = 24.5 KiB
Route D CSR:          12,365 + 24,730 + 516 = 37,611 B = 36.7 KiB
Route E:              26,880 + 600 × 5 = 29,880 B = 29.2 KiB
```

Two results are counter-intuitive and worth pausing on.

First, route D has the fewest parameters (12,365) and almost the largest compressed size (36.7 KiB). The index overhead of 24,730 bytes is twice the value payload. At 70% sparsity, a 2-byte index per nonzero costs more than the 1-byte value it describes. Sparse storage becomes attractive only when the index is smaller than the value, which requires lower-precision indices, block sparsity, or a much higher sparsity ratio.

Second, route E saves 35% of the parameters but only 12% of the bytes relative to the INT8 baseline, because the metadata grows with the number of output channels and the factored form adds channels. Parameter counts and byte counts are different budgets.

### 6.5 Latency and energy against the budget

The latency budget is 40 ms. Routes B, C, and the combination in Section 6.9 meet it; routes A, D, and E do not.

| Route | Latency (ms) | Margin to 40 ms | Energy (mJ) | Meets latency |
|---|---:|---:|---:|---|
| A | 45.7 | -5.7 | 1.124 | No |
| B | 29.0 | +11.0 | 0.713 | Yes |
| C | 25.1 | +14.9 | 0.617 | Yes |
| D | 45.7 | -5.7 | 1.124 | No |
| E | 45.6 | -5.6 | 1.122 | No |

Route C is the fastest and cheapest per inference, but it has the largest accuracy loss. Route B is slower but more accurate. Neither dominates, which is exactly the frontier situation from Topic 01.

### 6.6 Accuracy and calibration cost

| Route | Sensitivity | Δ vs baseline (pp) | ECE | Slope | Subgroup minimum sensitivity |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.942 | 0.0 | 0.021 | 0.97 | 0.931 |
| A: INT8 PTQ | 0.936 | -0.6 | 0.034 | 0.91 | 0.922 |
| B: Distill + INT8 | 0.934 | -0.8 | 0.027 | 0.95 | 0.919 |
| C: Structured prune + INT8 | 0.930 | -1.2 | 0.038 | 0.89 | 0.912 |
| D: Unstructured 70% | 0.927 | -1.5 | 0.041 | 0.86 | 0.905 |
| E: Low-rank + INT8 | 0.933 | -0.9 | 0.030 | 0.94 | 0.918 |

The floor is 0.925 overall. Route C passes overall at 0.930 with a margin of 0.5 percentage points, but its worst subgroup is 0.912, which is below the floor if the floor applies to every subgroup. Route D fails both the overall floor and the latency budget. This is the central lesson of the table: an overall pass is not a deployment pass.

The calibration slope column tells a second story. The baseline slope is 0.97, close to the ideal of 1.0. Route A falls to 0.91, which means predicted probabilities are too extreme relative to observed frequencies: high-risk windows are over-confident, low-risk windows are under-confident. Route C falls to 0.89, the largest shift. A slope below 1 can be corrected by a post-hoc calibration fit, but the fit must be learned on data that is independent of the held-out set, and the correction becomes part of the model that must be validated.

A sensitivity that passes while the slope degrades is not a clean result. It means the operating threshold may no longer correspond to the same risk, and the false-alarm rate per hour may change even if sensitivity is preserved. That is why the table reports both, and why a deployment decision that looks only at sensitivity is incomplete.

### 6.7 Diagnostic one: outlier-driven scale collapse

The symmetric scale is 0.003244, which is 17.8% of the median absolute weight. One weight in 40 rounds to zero, and only 24 of 255 integer levels are used. The aggregate SQNR of 37.54 dB looks acceptable because the outlier dominates the signal energy, but the non-outlier weights are represented by roughly 4 bits of effective precision.

The diagnostic pair is:

```text
step / median(|w|) = 0.003244 / 0.0182 = 0.178
distinct levels used = 24 / 255
```

A step above 10% of the median, or fewer than half the levels in use, indicates collapse. Affine quantization reduces the step to 0.001855 and raises SQNR to 42.04 dB, but the real fix for accuracy is per-channel quantization, which gives every output channel its own scale.

### 6.8 Diagnostic two: per-channel skew

The eight output channels have ranges from 0.0407 to 0.4703, a ratio of 11.6. Under a per-tensor scale, the smallest channel receives about 12 levels and the outlier channel receives 145. Per-channel scales raise SQNR from 37.54 to 51.37 dB, a gain of 13.83 dB, at a cost of 40 bytes for this layer and 600 bytes for the model.

The diagnostic is the channel range ratio. Above about 4, per-tensor quantization will cost material accuracy. The remedy costs 5 bytes per channel, which is negligible for a model of this size and material only for tensors with many channels and few weights each.

### 6.9 Varying one assumption

**Calibration set size: 512 windows to 64.** The activation ranges are computed from fewer samples, so the observed maximum is lower and the scale shrinks by about 6%. Clipping increases. Sensitivity falls a further 0.4 percentage points to 0.932 for route A, and ECE rises to 0.038, close to the 0.040 floor. Memory and latency are unchanged. The lesson: calibration set size is an accuracy assumption, not a performance assumption, and it must be varied in the accuracy analysis.

**Bit width: 8 to 4, weight-only.** The model's weight bytes fall to 41,216 / 2 + 600 = 21,208 bytes (20.7 KiB). The first-layer SQNR falls to 12.65 dB, cosine similarity to 0.974, and 28 of 40 weights quantize to zero. Sensitivity falls by about 3.1 percentage points, well below the floor. On this compute-bound model, latency does not improve, because MACs are unchanged. INT4 is a memory tool for redundant weight matrices, not a general-purpose speed tool.

**Layer excluded: first convolution kept in FP32.** Keeping 40 weights in FP32 adds 120 bytes over INT8, which is 0.29% of route A's size. It recovers about 0.2 percentage points of sensitivity because the first layer's quantization error propagates through every subsequent layer. The trade is favourable when the layer is small; excluding a large layer is a memory decision, not a rounding decision.

**Composition order reversed: quantize first, then prune.** Quantizing to INT8 and then applying 40% structured pruning produces a model whose per-channel scales no longer match the remaining channels. Recomputing them requires recalibration, and the pruned model loses an additional 1.5 percentage points because the pruning decision was made on rounded weights. The combined loss is 2.7 percentage points instead of 1.2. Order matters, and the penalty for the wrong order is larger than the gain from either method alone.

**Combination: distill to 26,000, prune 40%, quantize.** Parameters 26,000 × 0.60 = 15,600. MACs 2,600,000 × 0.60 = 1,560,000. Bytes 15,600 + 72 × 5 = 15,960 (15.6 KiB). Latency 1,560,000 / 89,600,000 = 17.4 ms, energy 0.428 mJ. Sensitivity falls to about 0.926, which is 0.1 percentage points above the floor. The pipeline meets the latency and memory budgets with almost no accuracy margin, which makes it a poor choice for a regulated product even though it is the most efficient.

### 6.10 Reading the route table

The table rewards three readings.

**Read the parameter column and the byte column separately.** Route D has the fewest parameters and the third-largest byte count. Route E has 35% fewer parameters than route A but only 29% fewer bytes. A claim about parameters is not a claim about storage, and a claim about storage is not a claim about latency.

**Read the MAC column against the latency column.** Routes A and D have identical MAC counts and identical latency, even though D has 70% fewer weights. Route C has 45% fewer MACs than A and 45% lower latency, which is consistent with a compute-bound model. Route E has 0.35% fewer MACs and 0.2% lower latency. The latency column tracks MACs, not parameters, which is the signature of a compute-bound deployment.

**Read the accuracy column against the floor, then against the subgroup minimum.** Route C passes the overall floor by 0.5 percentage points and fails the subgroup floor by 1.3. If the clinical claim applies to every subgroup, route C is not deployable. This is the difference between a compression experiment and a deployment decision.

### 6.11 What the case does not show

- **The cost of finding the student.** Route B assumes a student architecture that was found by search or experience. The search cost is not in the table.
- **The cost of the pruning schedule.** Route C assumes a pruning schedule that recovered most of the accuracy. A one-shot prune at 40% would lose more.
- **The interaction with the runtime's kernel selection.** The latency estimates assume the integer kernels run at the same efficiency for every route. In practice, a pruned model with 72 channels may not map to the kernel's preferred channel count and may run relatively slower.
- **The exported model's behavior.** Every route must be re-measured after export. The table describes the framework model.
- **The regulatory cost.** Route B changes the architecture and is a larger change than route A. The table does not price that difference.

A reviewer should treat the table as a method for comparing options, not as a prediction of a product's performance.

### 6.12 The decision

Against the three constraints, route A meets the flash budget (40.8 KiB of 64) and fails latency by 5.7 ms. Route C meets latency with 14.9 ms of margin and fails the subgroup floor. Route B meets latency and flash, passes the overall floor at 0.934, and passes the subgroup floor at 0.919 if the subgroup floor is 0.915. On the stated numbers, route B is the only route that meets every constraint.

The decision is therefore route B, with two conditions. First, the student architecture must be trained and its subgroup performance confirmed with a confidence interval, because 0.919 is close to a 0.915 floor and the interval may cross it. Second, the change must be classified: distillation changes the architecture, which is a larger regulatory change than quantization alone. If the change cannot be covered, route A with a 6 ms latency relaxation or a faster clock is the fallback.

This is the shape of a real decision: not "which method compresses best", but "which pipeline meets every constraint with a margin that survives its confidence interval, under a change path that exists".

## 7. Code, query, or calculation

The script computes the effect of each compression route, including parameter counts, bytes, MACs, latency, and energy, and implements the quantization core with error metrics. It reproduces Section 6.

```python
import math
import random


def round_half_away(value):
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def model_bytes(params, channels, bits=8, payload=None):
    """Weights plus per-channel scale (4 B) and zero point (1 B)."""
    weights = payload if payload is not None else params * bits / 8
    return weights + channels * 5


def latency_ms(macs, effective_macs_per_s=89_600_000):
    return macs / effective_macs_per_s * 1000


def energy_mj(latency, voltage=3.0, current_a=0.0082):
    return voltage * current_a * latency
```

Compute the routes:

```python
P0, M0 = 41_216, 4_096_000
print("baseline fp32 bytes =", P0 * 4, "latency =", latency_ms(M0))

route_a = model_bytes(P0, 120)
route_b = model_bytes(26_000, 120)
route_c = model_bytes(24_730, 72)
nz = round(P0 * 0.30)
route_d = nz + 2 * nz + 4 * 129
route_e = model_bytes(26_880, 600)

for name, params, nbytes, macs in [
    ("A INT8 PTQ", P0, route_a, M0),
    ("B distill+INT8", 26_000, route_b, 2_600_000),
    ("C structured+INT8", 24_730, route_c, 2_252_800),
    ("D unstructured 70%", nz, route_d, M0),
    ("E low-rank+INT8", 26_880, route_e, 4_081_664),
]:
    lat = latency_ms(macs)
    print("%-20s params %6d bytes %7.0f KiB %5.1f MACs %7d lat %5.1f ms E %.3f mJ"
          % (name, params, nbytes, nbytes / 1024, macs, lat, energy_mj(lat)))
```

Expected output:

```text
baseline fp32 bytes = 164864 latency = 45.714285714285715
A INT8 PTQ           params  41216 bytes   41816 KiB  40.8 MACs 4096000 lat  45.7 ms E 1.125 mJ
B distill+INT8       params  26000 bytes   26390 KiB  25.8 MACs 2600000 lat  29.0 ms E 0.714 mJ
C structured+INT8    params  24730 bytes   25090 KiB  24.5 MACs 2252800 lat  25.1 ms E 0.619 mJ
D unstructured 70%   params  12365 bytes   37611 KiB  36.7 MACs 4096000 lat  45.7 ms E 1.125 mJ
E low-rank+INT8      params  26880 bytes   29880 KiB  29.2 MACs 4081664 lat  45.6 ms E 1.121 mJ
```

The baseline latency printed by the formula is 45.7 ms, because the formula uses the INT8 effective throughput. The measured FP32 baseline is 138 ms, because the float kernels are about 3× slower per MAC. State which throughput applies to which precision; this is a common source of confusion.

Quantization and error metrics:

```python
def symmetric_scale(values, bits=8):
    return max(abs(v) for v in values) / (2 ** (bits - 1) - 1)


def affine_params(values, bits=8):
    lo, hi = min(values), max(values)
    scale = (hi - lo) / (2 ** bits - 1)
    zero_point = min(2 ** bits - 1, max(0, round_half_away(-lo / scale)))
    return scale, zero_point


def quantize(values, scale, zero_point=0, bits=8, symmetric=False):
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax - 1
    else:
        qmin, qmax = 0, 2 ** bits - 1
    return [min(qmax, max(qmin, round_half_away(v / scale) + zero_point))
            for v in values]


def dequantize(codes, scale, zero_point=0):
    return [(q - zero_point) * scale for q in codes]


def metrics(original, reconstructed):
    n = len(original)
    mse = sum((a - b) ** 2 for a, b in zip(original, reconstructed)) / n
    signal = sum(a * a for a in original)
    noise = sum((a - b) ** 2 for a, b in zip(original, reconstructed))
    dot = sum(a * b for a, b in zip(original, reconstructed))
    norm = math.sqrt(signal) * math.sqrt(sum(b * b for b in reconstructed))
    return {"mae": sum(abs(a - b) for a, b in zip(original, reconstructed)) / n,
            "rmse": math.sqrt(mse),
            "sqnr_db": 10 * math.log10(signal / noise) if noise else float("inf"),
            "cosine": dot / norm if norm else 0.0}


random.seed(7)
weights = [round(random.gauss(0, 0.035), 4) for _ in range(39)]
weights.insert(11, 0.412)

s = symmetric_scale(weights, 8)
qs = quantize(weights, s, bits=8, symmetric=True)
sa, zp = affine_params(weights, 8)
qa = quantize(weights, sa, zp, 8)

print("symmetric S=%.8f" % s, {k: round(v, 4) for k, v in metrics(weights, dequantize(qs, s)).items()})
print("affine S=%.8f Z=%d" % (sa, zp), {k: round(v, 4) for k, v in metrics(weights, dequantize(qa, sa, zp)).items()})

channels = [weights[i * 5:(i + 1) * 5] for i in range(8)]
per_channel = []
for channel in channels:
    sc = symmetric_scale(channel, 8)
    per_channel.append(dequantize(quantize(channel, sc, bits=8, symmetric=True), sc))
print("per-channel", {k: round(v, 4) for k, v in metrics(weights, [v for c in per_channel for v in c]).items()})

s4 = symmetric_scale(weights, 4)
print("int4 S=%.8f" % s4, {k: round(v, 4) for k, v in metrics(weights, dequantize(quantize(weights, s4, bits=4, symmetric=True), s4)).items()})
print("int4 model bytes =", model_bytes(P0, 120, bits=4))
```

Expected output:

```text
symmetric S=0.00324409 {'mae': 0.0008, 'rmse': 0.0009, 'sqnr_db': 37.5354, 'cosine': 1.0}
affine S=0.00185490 Z=33 {'mae': 0.0005, 'rmse': 0.0006, 'sqnr_db': 42.0374, 'cosine': 1.0}
per-channel {'mae': 0.0001, 'rmse': 0.0002, 'sqnr_db': 51.3696, 'cosine': 1.0}
int4 S=0.05885714 {'mae': 0.0142, 'rmse': 0.0164, 'sqnr_db': 12.654, 'cosine': 0.9736}
int4 model bytes = 21208.0
```

Assumptions: latency is MACs divided by 89.6 MMAC/s, which is the INT8 effective throughput on the target; FP32 throughput is about one third of that; energy uses board current and voltage; metadata is 5 bytes per channel; and the CSR layout uses 1-byte values, 2-byte column indices, and 4-byte row pointers with 128 rows.

## 8. Data, standards, and regulatory context

Compression methods are not standardised. There is no ISO standard that defines "quantization" or "pruning", and no regulator approves a compression algorithm. What standards and regulators constrain is the **evidence** that a compressed model still meets its claim, and the **change control** that governs the substitution.

### 8.1 What IEC 62304 requires

IEC 62304, "Medical device software — Software life cycle processes", Edition 1.1 (2006 plus Amendment 1:2015), is the software lifecycle standard recognized by the United States Food and Drug Administration (FDA) under recognition number 13-79. It requires requirements, architecture, detailed design, unit verification, integration testing, and configuration management.

For a compression pipeline, the consequences are:

- Each transformation is a design activity with an input, an output, and a rationale.
- The compression tool is part of the software item if its output is shipped; its version must be recorded.
- The calibration data is a configuration item, with provenance and version.
- The compressed model must be verified against the same requirements as the baseline.
- Safety class A, B, or C determines the rigour of the verification, not whether it happens.

A pipeline recorded only in a notebook with no version history does not meet the configuration-management expectation, even if the model works.

### 8.2 What ISO 14971 requires

ISO 14971, "Medical devices — Application of risk management to medical devices", requires hazards to be identified and risks controlled. Compression can create hazards that the baseline did not have:

- A subgroup's sensitivity falls below the clinical boundary while the overall metric remains acceptable.
- Calibration shifts at the operating threshold, changing the number of false alarms per hour.
- A pruned model fails on a rare class that was underrepresented in the calibration set.
- A quantized model saturates on inputs outside the calibration distribution.

Each of these is a foreseeable hazard and belongs in the risk file, with a control and a residual-risk judgement. The control may be a monitoring requirement, a fallback path, or a restriction on the intended use.

### 8.3 What the FDA PCCP covers

The FDA's final guidance on Predetermined Change Control Plans for AI-enabled device software functions was issued on 3 December 2024, with a Federal Register notice on 4 December 2024, under section 515C of the Federal Food, Drug, and Cosmetic Act as added by the Food and Drug Omnibus Reform Act of 2022. A PCCP has three components: a description of modifications, a modification protocol, and an impact assessment.

A compression change fits the PCCP structure if the plan anticipated it. The description names the method (for example, INT8 post-training quantization with per-channel scales), the protocol names the acceptance criteria and the data, and the impact assessment names the risks and the cumulative effect. A change outside the plan, such as switching from quantization to distillation, is a different modification and may require a new submission.

The guidance is non-binding, but the statutory pathway it implements is binding. A team that treats compression as a build setting is not avoiding the question; it is deferring it to the least convenient moment.

### 8.4 What the EU framework requires

In the European Union, Regulation (EU) 2017/745, the Medical Devices Regulation, governs the device, and Regulation (EU) 2024/1689, the Artificial Intelligence Act, adds obligations for high-risk AI systems. A medical device that is a safety component, or is itself a device requiring third-party conformity assessment, is high-risk under Article 6(1).

The applicability date for Annex I products was originally 2 August 2027. The Commission's AI Omnibus, adopted on 19 November 2025 with political agreement reported on 7 May 2026, moves the Annex I date to 2 August 2028. Verify the current date before relying on it; the Medical Device Coordination Group guidance MDCG 2025-6 (June 2025) remains the reference for the interaction between the two regulations.

For compression, the relevant obligations are transparency, human oversight, and post-market monitoring. A compressed model whose performance differs by subgroup is a transparency and monitoring issue, not only an engineering one.

### 8.5 What is binding and what is convention

| Item | Status | Consequence for compression |
|---|---|---|
| IEC 62304 lifecycle | Binding where required | Pipeline stages are design records |
| ISO 14971 risk management | Binding where required | Compression hazards enter the risk file |
| FDA PCCP guidance | Non-binding guidance; statutory pathway binding | Change must be pre-authorized or submitted |
| EU AI Act high-risk obligations | Binding regulation, dates moving | Transparency and monitoring obligations |
| MLPerf benchmark methods | Voluntary convention | Useful for comparison, no regulatory weight |
| "Compression ratio" | Convention | Must state parameters, bytes, or latency |
| Quantization scheme names | Convention | No standard definition; describe the scheme |

### 8.6 Documentation the pipeline needs

| Record | Content | Owner |
|---|---|---|
| Baseline record | Model hash, training data version, metrics | ML |
| Stage record | Method, parameters, tool version, seed | ML |
| Calibration record | Data source, size, date range, subgroup composition | Data |
| Evaluation record | Held-out set, threshold rule, confidence intervals | Biostatistics |
| Export record | Converter version, runtime version, target | Firmware |
| Change record | Classification, PCCP reference, acceptance criteria | Regulatory |
| Deviation record | Any stage that missed its allocation | Quality |

The pattern is the same as in Topics 01 and 02: the record is cheap when the work is done and expensive when it is reconstructed.

## 9. Industry practice and case evidence

### 9.1 The methods that became products

Several compression methods have moved from papers to production toolchains, and their trajectories are instructive.

**Distillation.** Hinton, Vinyals, and Dean introduced soft-target distillation in 2015 (arXiv:1503.02531), using a temperature-scaled softmax to transfer the teacher's class similarities. DistilBERT (Sanh et al., 2019) applied the idea to a transformer: 66 million parameters against BERT-base's 110 million, a 40% reduction, reported as 60% faster on CPU and retaining 97% of GLUE performance. The paper reports 410 seconds versus 668 seconds for a full STS-B pass at batch 1 on an Intel Xeon E5-2690 v3, and 71% faster on an iPhone 7 Plus for a question-answering application excluding tokenization. These are the authors' measurements on their hardware, not a guarantee for another model.

**Quantization.** Jacob et al. (2017/2018) established integer-arithmetic-only inference with affine quantization and showed that a quantized training procedure could recover accuracy. The TensorFlow Lite 8-bit specification now defines the conventions: `real = (int8 - zero_point) × scale`, asymmetric int8 activations with a zero point in [-128, 127], symmetric int8 weights with zero point 0, and per-axis scales along a specified dimension for convolution weights. ONNX Runtime supports both QOperator and QDQ representations, with per-channel weight quantization and int32 bias whose scale is the product of input and weight scales.

**Structured pruning and sparsity.** NVIDIA's Ampere architecture introduced 2:4 structured sparsity: at most two nonzeros in every group of four, with sparse tensor cores that can double math throughput for supported operations. The A100 whitepaper describes the pattern and the sparse MMA semantics, and cuSPARSELt provides the API for pruning and compressing to the pattern. The benefit is real but conditional: the layer must be large enough, the pattern must be enforced during training, and the runtime must select the sparse path.

**Lottery tickets.** Frankle and Carbin (2018, ICLR 2019) showed that dense networks contain sparse subnetworks that, when trained from their original initialization, match the dense network's accuracy. Winning tickets were found at 10–20% of the original size on LeNet and small convolutional networks. The result is a research finding about trainability, not a deployment recipe: finding a ticket requires iterative train-prune-reset cycles, and the resulting sparsity is unstructured, so the hardware benefit is storage unless sparse kernels exist.

### 9.2 What the toolchains actually support

The gap between a method's paper and its deployment is usually a toolchain gap.

- TensorFlow Lite's quantization specification defines per-axis quantization for convolution and depthwise convolution weights, with int32 bias. A model that uses an operator outside the specification falls back to float.
- ONNX Runtime's quantization tool supports static and dynamic quantization, QOperator and QDQ formats, per-channel weights, and 16-bit and 4-bit types at recent opsets. Per-channel weights are quantized symmetrically; bias is int32 with scale equal to input scale times weight scale.
- ExecuTorch uses torchao for quantization, with backend-specific quantizers; the same model quantized for XNNPACK and for a Qualcomm backend can produce different graphs and different accuracy.
- CMSIS-NN implements int8 and int16 kernels bit-exact with the TFLite reference for Cortex-M, plus int4 weights with int8 activations for some operators, with three optimization tiers selected by feature flags.

The practical consequence is that a compression method is not portable. A pipeline validated with one runtime must be re-validated with another, because the operator coverage, the rounding, and the fused kernels differ.

### 9.3 What the evidence supports, and what it does not

| Claim | Status |
|---|---|
| DistilBERT is 40% smaller and 60% faster on CPU than BERT-base | Authors' measurement, specific hardware and task |
| 2:4 sparsity can double math throughput on Ampere sparse tensor cores | Vendor documentation, supported layers and shapes |
| Unstructured pruning reduces storage but not dense-kernel latency | Consequence of dense kernels; measured in many studies |
| A 90% sparse model is 10× faster | Not supported in general; depends on hardware and pattern |
| Lottery tickets exist at 10–20% of dense size on small vision models | Research finding, specific architectures |
| Per-channel quantization recovers most per-tensor accuracy loss | Widely reported; magnitude is model-dependent |
| The pipeline in Section 6 meets a 40 ms budget | Constructed example, not a measurement |

The last row is the one to keep visible. Worked examples teach method; they do not produce evidence.

### 9.4 Three deployment patterns

Across published case descriptions, three patterns recur.

**The quantization-only deployment.** The model architecture is fixed, the runtime supports integer kernels for every operator, and the change is covered by a plan. This is the most common production pattern because it has the smallest change surface and the best toolchain support. Its limits are the precision floor and the operator list.

**The distilled-then-quantized deployment.** The architecture changes to meet a compute budget, then quantization reduces memory. This pattern appears in mobile language models and keyword spotting, where the teacher is large and the student is designed for the device. Its limits are the capacity gap and the larger regulatory change.

**The architecture-search deployment.** The model is found by a search constrained by the target's latency and memory, rather than compressed after training. This pattern appears in production vision models and in some audio pipelines. Its limit is the search cost and the difficulty of explaining the result to a reviewer.

The patterns differ in cost and risk, not only in technique. A quantization-only change can often be validated with a paired comparison on a held-out set; an architecture change requires a broader evidence package and, in a regulated product, a new submission or a PCCP that anticipated it.

### 9.5 What the case evidence does not establish

The published results establish that each method can work under specific conditions. They do not establish:

- That the method will work on a different architecture, dataset, or hardware.
- That the reported compression ratio translates to latency on a different runtime.
- That the accuracy retention holds for a clinical population with a different prevalence.
- That the subgroup performance is preserved, because subgroup results are rarely reported.
- That the change is covered by a regulatory pathway, because papers do not address change control.

These gaps are the reason a deployment programme needs its own evidence, even when the method is well established in the literature.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Method does not target the binding resource | Compression succeeds, latency unchanged | Algorithmic effect without hardware effect | Wasted quarter | Profile before selecting |
| Unstructured pruning on dense hardware | Sparsity high, latency flat | Dense kernels multiply zeros | No speedup, possible slowdown | Check sparse kernel support |
| Quantizing before pruning | Accuracy drop larger than predicted | Scales computed for weights that are later removed | 1–3 pp additional loss | Structure before precision |
| Pruning after QAT | Calibration invalid | Scales no longer match the pruned distribution | Silent accuracy loss | Recalibrate after structure changes |
| Multiplying compression ratios | Predicted 16×, measured 4× | Gains overlap on the same resource | Wrong hardware sizing | Measure per stage |
| Ignoring metadata | Small model, large file | Scales, zero points, indices | Budget overrun | Count metadata explicitly |
| Sparse storage with large indices | 70% sparse, 37 KiB | 2-byte index per 1-byte value | Storage barely improves | Block sparsity or lower index width |
| Distillation from a biased teacher | Student matches teacher's subgroup gaps | Bias transfer | Equity harm | Audit teacher and student |
| Capacity gap too large | Student plateaus below floor | Insufficient expressive power | Accuracy miss | Choose student size empirically |
| Low-rank without reconstruction check | Factorized model degrades on rare inputs | Rank too low for the weight spectrum | Accuracy loss | Measure reconstruction error per layer |
| Operator fallback after factorization | Latency worse despite fewer MACs | Unsupported batched matmul | Regression | Check operator support |
| Calibration set from one site | Overall passes, subgroup fails | Ranges tuned to one distribution | Equity harm | Stratified calibration |
| Thermal profile changed | Sustained latency worse | Smaller model runs at higher clock | Deadline miss | Soak test the compressed model |
| Pipeline not reproducible | Results differ between runs | Unrecorded seed or tool version | Evidence invalid | Record seeds and versions |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The binding resource is correctly identified | Method selection depends on it | Correlated bottlenecks, thermal effects | Per-layer profile and roofline |
| The hardware effect matches the algorithmic effect | Compression is expected to help | Missing sparse or integer kernels | Kernel support check and microbenchmark |
| Stage gains are independent | Compound estimate | Shared resource or accuracy interaction | Measure after each stage |
| Accuracy losses add linearly | Budget allocation | Correlated errors, threshold shifts | Measure composed model |
| Metadata is 5 bytes per channel | Byte accounting | float16 scales, shared zero points | Inspect the exported model |
| Sparse storage uses 2-byte indices | Route D estimate | Block sparsity or 4-byte indices | Compute from the actual format |
| Latency scales with MACs | Route estimates | Memory-bound layers, overhead | Target measurement |
| Energy scales with latency | Route comparison | DVFS, radio activity, leakage | Power-rail measurement |
| The calibration set represents deployment | Scales generalize | Site or device shift | Stratified audit |
| The exported model equals the trained model | Accuracy transfer | Fusion, rounding, layout | Compare outputs on a fixed set |
| The change is within the approved envelope | Regulatory path | New method, new architecture | PCCP review |
| The student can match the teacher | Distillation viable | Capacity gap, teacher bias | Empirical student sweep |

Two limits are worth stating separately.

The family is not exhaustive. Mixture-of-experts routing, early exit, conditional computation, and token pruning are compression-adjacent methods that reduce average work rather than model size. They are outside this topic's scope but appear in production systems, and they have their own hardware and validation requirements.

The methods are also not neutral with respect to fairness. Any method that reduces capacity or precision can degrade a minority subgroup more than the majority, because the subgroup's signal may be represented by fewer weights or a narrower activation range. The subgroup table is not an optional add-on; it is the evidence that the method is safe for the intended population.

## 12. Exercises

**Exercise 1 — Method classification (recall, 30 minutes).**
For each of quantization, distillation, structured pruning, unstructured pruning, low-rank factorization, fusion, and 2:4 sparsity, state the resource reduced, the hardware feature required, and one failure mode.
*Deliverable:* a seven-row table.
*Acceptance criteria:* each row names a resource and a hardware requirement; at least three rows name a failure that is not accuracy.
*Time:* 30 minutes.

**Exercise 2 — Candidate selection (application, 35 minutes).**
A model is memory-bound on activation bytes in SRAM, with 300 KiB peak against a 256 KiB limit. List three candidate methods, rank them by integration cost, and state which you would try first.
*Deliverable:* a ranked list with a one-paragraph justification.
*Acceptance criteria:* candidates target activation bytes; ranking includes integration cost; the first choice is justified by the profile.
*Time:* 35 minutes.

**Exercise 3 — Byte accounting (calculation, 45 minutes).**
A model has 100,000 parameters in 200 output channels. Compute INT8 per-channel bytes, INT4 per-channel bytes, and CSR bytes at 80% sparsity with 1-byte values, 2-byte indices, and 256 rows. State which representation is smallest.
*Deliverable:* calculations with units.
*Acceptance criteria:* metadata included; CSR index overhead correct; conclusion follows from the numbers.
*Time:* 45 minutes.

**Exercise 4 — The pruning latency trap (critique, 30 minutes).**
Explain why 90% unstructured sparsity can leave latency unchanged on a Cortex-M4 and on a dense GPU, and state the hardware conditions under which it would reduce latency.
*Deliverable:* a 250-word explanation.
*Acceptance criteria:* names dense kernels and memory access; names sparse tensor cores or sparse kernels as the condition; distinguishes storage from latency.
*Time:* 30 minutes.

**Exercise 5 — Composition order (design, 45 minutes).**
Given a model that is too large in MACs and must also fit a 4× smaller flash budget, propose an ordered pipeline of three methods and justify each edge.
*Deliverable:* an ordered list with a rationale per step.
*Acceptance criteria:* structure precedes precision; fine-tuning follows structural change; the rationale references the resource each method reduces.
*Time:* 45 minutes.

**Exercise 6 — Compound gain (analysis, 40 minutes).**
A team claims 4× from quantization and 2× from pruning, for 8× total. Explain two reasons the measured gain may be lower, and compute a plausible measured gain for a compute-bound model.
*Deliverable:* a short analysis with arithmetic.
*Acceptance criteria:* names resource overlap and hardware support; the computed gain is consistent with the stated mechanism.
*Time:* 40 minutes.

**Exercise 7 — Claim audit (critique, 35 minutes).**
Find a paper or vendor claim of "10× compression" for an edge model. Identify what is being counted, what hardware the result applies to, and what is missing for a deployment decision.
*Deliverable:* a 300-word audit with the source URL and access date.
*Acceptance criteria:* identifies the counted quantity; distinguishes parameters, bytes, MACs, and latency; names at least two missing conditions.
*Time:* 35 minutes.

**Exercise 8 — Pipeline record (design, 60 minutes).**
Write the pipeline record for the Section 6 combination route, including each stage's parameters, data, tool version, and measured effect.
*Deliverable:* a two-page record.
*Acceptance criteria:* every stage reproducible from the record; calibration data described; accuracy and calibration per stage; change classification noted.
*Time:* 60 minutes.

## 13. Self-check answers

**A1. Which method reduces MACs without changing the architecture?**
Structured pruning. It removes channels or blocks, which removes the corresponding multiply-accumulates in the current and subsequent layers, while keeping the same family of operations. Unstructured pruning does not reduce MACs on dense hardware.

**A2. Why does INT4 weight quantization not speed up a compute-bound layer?**
Because the arithmetic count is unchanged and the layer is limited by MAC throughput, not by weight bytes. INT4 reduces storage and weight traffic; it helps only when weights dominate the bytes moved or when the hardware has a faster 4-bit path.

**A3. Byte accounting for Exercise 3.**
INT8: 100,000 + 200 × 5 = 101,000 bytes. INT4: 50,000 + 1,000 = 51,000 bytes. CSR at 80% sparsity: 20,000 nonzeros → 20,000 values + 40,000 indices + 257 × 4 = 20,000 + 40,000 + 1,028 = 61,028 bytes. INT4 per-channel is smallest; CSR is larger than INT4 despite 80% sparsity because the index is twice the value size.

**A4. Why can 90% sparsity leave latency unchanged?**
Dense kernels execute all multiplications, including those by zero, so the arithmetic count and memory traffic are unchanged unless the hardware detects and skips zeros. On a GPU without sparse tensor cores and on an MCU without sparse kernels, sparsity is a storage property only. Latency falls only when the runtime uses a sparse kernel or sparse tensor core that the layer shape supports.

**A5. Why does order matter between pruning and quantization?**
Pruning changes which weights exist and their distribution. Quantization computes scales from that distribution. If quantization comes first, scales are computed for weights that pruning later removes, and the pruned model's remaining weights may be poorly represented. Recalibration is required, and the pruning decision itself was made on rounded values.

**A6. Why is the compound gain not the product of the individual gains?**
Because gains overlap on the same resource and the bottleneck can move. If quantization removes 75% of weight bytes but the model becomes compute-bound, further byte reduction from pruning yields no latency gain. The product is valid only when each method reduces a resource that remains binding.

**A7. What is the difference between 2:4 sparsity and 50% unstructured sparsity?**
2:4 constrains the pattern: at most two nonzeros in every group of four, which hardware can exploit with a fixed encoding. Fifty percent unstructured sparsity has no pattern guarantee, so hardware cannot skip zeros without a general sparse format and its index overhead. The first can double throughput on supported tensor cores; the second usually reduces storage only.

**A8. What does the lottery ticket hypothesis claim, and why is it not a deployment recipe?**
It claims that dense networks contain sparse subnetworks that can be trained from their original initialization to match the dense network's accuracy. It is not a deployment recipe because finding a ticket requires iterative training and pruning, the resulting sparsity is unstructured, and the benefit is not a latency reduction on dense hardware.

**A9. What must be recalibrated after structured pruning?**
Activation ranges and per-channel weight scales, because removing channels changes the input distribution of subsequent layers and the output-channel set. Bias correction may also be needed. Recalibration must use data that represents the deployment population.

**A10. What is the minimal pipeline principle?**
Choose the shortest sequence of methods that meets every constraint, because each stage adds accuracy risk, validation cost, and change-control burden. A single well-documented quantization stage is easier to defend than a three-stage pipeline with uncharacterized interactions.

**A11. Rubric for Exercise 5 (composition order).**
A strong answer places distillation first if the architecture is too large, then structured pruning, then fine-tuning, then quantization, and justifies each edge by the resource it reduces and the interaction it avoids. A weak answer lists the methods in the order they were learned, without a resource argument.

**A12. Rubric for Exercise 8 (pipeline record).**
A strong record lets another engineer reproduce every stage: model hash, data version, method parameters, tool version, seed, and measured effect. A weak record lists the methods and the final accuracy without the intermediate evidence.

## 14. Interview questions

**Q1. What is the difference between pruning and quantization?**
A strong answer says pruning removes parameters or structures while quantization reduces the precision of the values that remain, and notes that they target different resources and compose in a specific order. A weak answer treats both as "making the model smaller".

**Q2. When would you choose distillation over quantization?**
A strong answer chooses distillation when MACs or architecture size are binding and quantization cannot reduce arithmetic, and notes that distillation changes the model and therefore the regulatory path. A weak answer chooses distillation because it gives a "smaller model".

**Q3. Why can a 90% sparse model be slower than a dense one?**
A strong answer explains dense kernels, memory access patterns, and the absence of sparse hardware, and notes that sparse formats can add index overhead. A weak answer assumes sparsity always helps.

**Q4. What is 2:4 structured sparsity?**
A strong answer defines two nonzeros in every four, names sparse tensor cores and the supported layer shapes, and states that the pattern must be enforced during training. A weak answer calls it "50% pruning".

**Q5. How do you decide the order of compression methods?**
A strong answer states structure before precision, distillation before refinement, and fine-tuning after structural change, and justifies each from the interaction. A weak answer says the order does not matter if each method works.

**Q6. Why is the compression ratio an incomplete claim?**
A strong answer says it must state whether it counts parameters, bytes, MACs, or latency, and notes that the four can differ by large factors. A weak answer accepts the ratio as a performance claim.

**Q7. What is a low-rank factorization and when does it help?**
A strong answer describes replacing a matrix with two smaller factors, notes the parameter and MAC reduction, and states that it helps only if the factored form maps to efficient kernels and the rank is adequate. A weak answer says it always reduces compute.

**Q8. What happens if you prune after quantization-aware training?**
A strong answer says the quantization parameters no longer match the pruned distribution, the model must be recalibrated or retrained, and accuracy can drop by more than either method alone. A weak answer assumes the scales remain valid.

**Q9. How would you audit a vendor's "10× compression" claim?**
A strong answer asks what is counted, on what hardware, at what accuracy, with what baseline, and whether the measurement is vendor-run. A weak answer asks only for the model size.

**Q10. What is the risk of distilling from a biased teacher?**
A strong answer says the student inherits the teacher's subgroup errors and may amplify them, so both models need subgroup audits and calibration checks. A weak answer focuses only on overall accuracy.

**Q11. When is the minimal pipeline the right choice?**
A strong answer says whenever the shortest pipeline meets the constraints, because each additional stage adds risk and regulatory burden. A weak answer always pursues the highest compression ratio.

**Q12. What evidence would you require before shipping a pruned model?**
A strong answer requires per-stage accuracy, calibration and subgroup results, target latency and energy, sparse-path confirmation, and a change classification. A weak answer requires only a size reduction and an accuracy check.

## 15. Key terms

- **2:4 sparsity** — at most two nonzeros in every four consecutive weights.
- **Activation** — intermediate tensor produced at runtime.
- **Affine quantization** — quantization with a scale and a zero point.
- **Calibration** — selecting numeric ranges from representative data.
- **Capacity gap** — difference in expressive power between teacher and student.
- **Codebook** — table of shared weight values.
- **Compression ratio** — original size divided by compressed size; must state the counted quantity.
- **Constant folding** — precomputing constant subgraphs.
- **CSR** — compressed sparse row storage format.
- **Dense kernel** — kernel that multiplies all values, including zeros.
- **Distillation** — training a smaller model on a larger model's outputs.
- **Fine-tuning** — short retraining after a structural change.
- **Fusion** — combining graph nodes to reduce memory traffic.
- **Headroom** — distance between current and allowed resource use.
- **Index overhead** — bytes needed to describe sparse storage.
- **Knowledge distillation** — see distillation.
- **Knee** — point where further compression costs disproportionate accuracy.
- **Lottery ticket hypothesis** — dense networks contain sparse trainable subnetworks.
- **Low-rank factorization** — replacing a matrix with a product of smaller matrices.
- **MAC** — multiply-accumulate.
- **Metadata overhead** — scales, zero points, and codebooks.
- **Mixed precision** — different precisions for different tensors.
- **N:M sparsity** — at most N nonzeros in every group of M.
- **NAS** — neural architecture search.
- **Operator fusion** — see fusion.
- **Parameter** — learned weight or bias.
- **Payload** — bytes stored or moved per inference.
- **Post-training compression** — applying a method to a trained model.
- **PTQ** — post-training quantization.
- **QAT** — quantization-aware training.
- **Quantization** — representing values with fewer bits.
- **Rank** — number of independent components retained in a factorization.
- **Recovery** — accuracy regained by fine-tuning.
- **Sparse kernel** — kernel that skips zeros.
- **Sparsity** — fraction of weights equal to zero.
- **STE** — straight-through estimator.
- **Structured pruning** — removing whole channels, heads, or blocks.
- **Teacher** — model providing soft targets.
- **Temperature** — softmax scaling in distillation.
- **Unstructured pruning** — removing individual weights.
- **Weight sharing** — clustering weights into a codebook.
- **Weight-only quantization** — quantizing weights but not activations.

## 16. Sources

All URLs accessed 25 October 2026.

**Distillation**
- Hinton, G., Vinyals, O., Dean, J., "Distilling the Knowledge in a Neural Network", 2015. https://arxiv.org/abs/1503.02531
- Sanh, V., et al., "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter", 2019. https://arxiv.org/abs/1910.01108

**Quantization**
- Jacob, B., et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", 2017/2018. https://arxiv.org/abs/1712.05877
- Nagel, M., et al., "A White Paper on Neural Network Quantization", 2021. https://arxiv.org/abs/2106.08295
- Google AI Edge, "LiteRT 8-bit quantization specification". https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec
- ONNX Runtime, "Quantize ONNX models". https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

**Pruning and sparsity**
- Han, S., Mao, H., Dally, W., "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding", 2015. https://arxiv.org/abs/1510.00149
- Frankle, J., Carbin, M., "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", ICLR 2019. https://arxiv.org/abs/1803.03635
- NVIDIA, "NVIDIA A100 Tensor Core GPU Architecture" whitepaper. https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf
- NVIDIA, "Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture and NVIDIA TensorRT". https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/

**Runtime documentation**
- Arm, "CMSIS-NN". https://arm-software.github.io/CMSIS-NN/latest/
- PyTorch, "ExecuTorch quantization overview". https://docs.pytorch.org/executorch/stable/quantization-overview.html
- Google AI Edge, "Post-training quantization". https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization

**Standards and regulation**
- IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019. https://www.iso.org/standard/72704.html
- FDA, PCCP final guidance, 3 December 2024. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- MDCG 2025-6. https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en

> **Note:** The Section 6 routes are constructed for teaching. The DistilBERT and 2:4 sparsity figures are the authors' and vendor's measurements, respectively, on their own hardware.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md`](../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md)
- Previous topic: [`02_target-hardware-and-performance.md`](02_target-hardware-and-performance.md)
- Next topic: [`04_knowledge-distillation.md`](04_knowledge-distillation.md)
- Related topic: [`05_quantization.md`](05_quantization.md)
- Related topic: [`06_pruning-and-sparsity.md`](06_pruning-and-sparsity.md)
- Lab: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab directory: [`../../labs/`](../../labs/)
