# Pruning and Sparsity

> Module 14B — Efficient AI and On-Device Inference
> Topic 06 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 02, 03
> Version 2026.3

## 1. Why this topic exists

Pruning is the oldest compression method in neural networks and the one with the widest gap between reputation and deployment reality. The reputation comes from papers that report 90% of weights removed with little accuracy loss. The reality is that most of those models run at exactly the same speed as the dense original, because the hardware multiplies by zero as readily as by any other number.

The gap is not a scandal; it is a definition problem. Pruning removes **parameters**. Latency depends on **operations** and **memory traffic**. Removing a weight that the kernel still loads and multiplies does not reduce either. The benefit appears only when the removal is visible to the runtime: through a sparse storage format that reduces bytes read, a sparse kernel that skips work, or a structured removal that shrinks the tensors the dense kernel operates on.

Three industry problems follow.

**The storage-latency confusion.** A team prunes to 90% sparsity, measures a 10× smaller file, deploys, and finds no latency improvement. The file is smaller because the zeros are not stored. The kernel is not faster because it still executes the dense loop. Both facts are true, and the project has optimized the wrong budget.

**The structure-hardware mismatch.** Unstructured sparsity is flexible and hard to exploit. Structured sparsity removes whole channels or blocks, which dense kernels can exploit because the tensors themselves are smaller. 2:4 sparsity sits between them: a fixed pattern that specific tensor cores can exploit, but only for supported layer shapes and only when the pattern is enforced during training.

**The accuracy cliff.** Pruning looks harmless until it does not. Accuracy is roughly flat as sparsity increases, then falls sharply at a threshold that depends on the model, the criterion, and the fine-tuning schedule. A team that prunes to a round number without finding the cliff ships a model that is 1.5 percentage points worse and does not know why.

There is also a patient-safety dimension. Magnitude pruning removes the smallest weights first, and small weights can carry rare-class signal. A pruned model can lose sensitivity for a condition that was uncommon in the pruning data while its overall accuracy is unchanged. The subgroup table is the only way to see it.

This topic teaches the four levels of sparsity (algorithmic, storage, kernel, hardware), the criteria and schedules that determine accuracy, the storage formats and their overhead, and the evaluation that makes a pruned model deployable.

### 1.1 Three claims that pruning reports make

Every pruning claim falls into one of three categories, and each requires different evidence.

1. **"The model is 70% sparse."** This is a claim about the weights. Evidence: a histogram or a count of zeros. It says nothing about size or speed.
   - What it does not mean: the file is 70% smaller, or the model is 70% faster.
   - What it is useful for: characterizing the pruning procedure and comparing criteria.

2. **"The model file is 3× smaller."** This is a claim about storage. Evidence: a file size and a format description.
   - What it does not mean: the model uses less RAM at runtime, or runs faster.
   - What it is useful for: flash budgeting and update payload size.

3. **"The model runs 1.8× faster."** This is a claim about latency. Evidence: a target measurement at a stated percentile, batch, and condition.
   - What it does not mean: the model is more efficient on a different device.
   - What it is useful for: meeting a clinical deadline.

A report that mixes the three is not lying; it is being imprecise in a way that leads to wrong decisions. The first claim is cheap to produce, the third is expensive, and teams often substitute the cheap one for the expensive one under schedule pressure.

### 1.2 What pruning is not

- It is not a way to make a dense kernel faster. The kernel must know about the zeros.
- It is not a substitute for architecture choice. A model with the wrong receptive field is not fixed by removing weights.
- It is not free of accuracy risk, especially for rare classes.
- It is not uniform in its effect. Some layers tolerate 90% sparsity and others tolerate 20%.

The rest of the topic works through the consequences, and the case in Section 6 shows why the same sparsity number can produce a 1.8× latency difference depending on the pattern.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** algorithmic sparsity from storage sparsity, kernel sparsity, and hardware sparsity.
2. **Apply** magnitude, movement, and gradient-based pruning criteria and explain their trade-offs.
3. **Compute** the storage of a sparse tensor in CSR, bitmap, and N:M formats, including index overhead.
4. **Predict** whether a given sparsity pattern will reduce latency on a given target.
5. **Choose** a pruning schedule and a fine-tuning strategy.
6. **Evaluate** a pruned model on accuracy, calibration, and subgroups, with attention to rare classes.
7. **Combine** pruning with quantization and explain why structure precedes precision.
8. **Document** a pruning run so that the sparsity pattern and its provenance are reproducible.

| Outcome | Practised in | Evidenced by |
|---:|---|---|
| 1 | Exercise 1 | A four-level table with examples |
| 2 | Exercise 2 | A criterion comparison on a toy tensor |
| 3 | Exercise 3 | Correct CSR, bitmap, and N:M byte counts |
| 4 | Exercise 4 | A latency prediction with a hardware condition |
| 5 | Exercise 5 | A schedule with a stopping rule |
| 6 | Exercise 6 | A subgroup table with intervals |
| 7 | Exercise 7 | An ordered pipeline with a rationale |
| 8 | Exercise 8 | A reproducible sparsity record |

## 3. The core mental model

Sparsity exists at four levels, and a benefit at one level does not imply a benefit at another.

```text
   Level 1: algorithmic      weights set to zero in the trained model
            │                 (a property of the numbers)
            ▼
   Level 2: storage          zeros not stored, or stored compactly
            │                 (a property of the file)
            ▼
   Level 3: kernel           the loop skips zero multiplications
            │                 (a property of the implementation)
            ▼
   Level 4: hardware         the silicon has a sparse datapath
                              (a property of the chip)
```

A model can be sparse at level 1 and dense at levels 2, 3, and 4. That is the default outcome of magnitude pruning followed by dense deployment. The benefit requires the sparsity to survive down the stack.

### 3.1 What each level buys

| Level | What changes | Benefit | Cost |
|---|---|---|---|
| Algorithmic | Weights become zero | None by itself | Accuracy risk |
| Storage | Fewer bytes in the file | Flash and load time | Index overhead |
| Kernel | Fewer operations executed | Latency and energy | Sparse kernel availability |
| Hardware | Sparse datapath | Throughput | Fixed pattern, supported shapes |

The storage benefit is real and often valuable. A device with a 512 KiB flash budget may fit a pruned model that a dense model cannot. That is a legitimate reason to prune even when latency does not improve. The mistake is to claim a latency benefit from a storage change.

### 3.2 Structured and unstructured

**Unstructured pruning** removes individual weights. The pattern is arbitrary, which gives the best accuracy at a given sparsity, and the worst hardware support.

**Structured pruning** removes whole channels, heads, or blocks. The remaining tensors are smaller in every dimension, so dense kernels run faster without any special support. The accuracy cost is higher because the removed units cannot be partially retained.

**N:M sparsity** keeps at most N nonzeros in every group of M. It is a constrained unstructured pattern: flexible within each group, but regular enough for hardware to encode. NVIDIA's Ampere architecture implements 2:4, and Arm's Ethos-U85 documents 2:4 support.

```text
   unstructured           structured             2:4
   ■ · ■ ■ · · ■ ·        ■ ■ ■ ■ · · · ·        ■ · ■ · ■ · ■ ·
   · ■ · · ■ ■ · ■        ■ ■ ■ ■ · · · ·        · ■ · ■ · ■ · ■
   ■ · · ■ · ■ ■ ·        · · · · · · · ·        ■ ■ · · · ■ ■ ·
   (arbitrary)            (whole columns)        (2 per 4, per row)
```

### 3.3 The accuracy curve

Pruning accuracy is flat for a while, then falls.

```text
accuracy
   ▲
   │  ●──●──●──●
   │            ●
   │              ●
   │                ●
   └──┴──┴──┴──┴──┴──┴──► sparsity
      50%   70%  80%  90%  95%
                ▲
             cliff
```

The cliff's location depends on the model, the criterion, the schedule, and the fine-tuning budget. It is not a universal number, and a paper's 90% result for one architecture says little about another.

Two practical consequences:

- **Find the cliff empirically.** Prune in increments, evaluate at each, and stop before the last increment that costs more than the allocated accuracy.
- **Do not target a round number.** A target of 90% because it is a round number is a decision made before the evidence. A target of 78% because 80% crosses the cliff is a decision made from evidence.

### 3.4 Why small weights can matter

Magnitude pruning assumes that small weights contribute little. That assumption is usually true for the average input and can be false for a rare one. A weight that is small in magnitude may be the only feature that distinguishes a rare class, and removing it can drop that class's sensitivity while leaving overall accuracy unchanged.

The same logic applies to channels. A channel with a small norm may be redundant for the majority of inputs and essential for a minority. Structured pruning is more dangerous here because it removes the entire channel, not just a weight.

The defense is not to avoid pruning; it is to evaluate the pruned model on the classes and subgroups where a small feature could matter. That evaluation is the subject of Section 5, Step 8.

### 3.5 Sparsity is not compression until it is encoded

A sparse tensor in memory can be stored in three ways, and the choice determines the storage benefit.

- **Dense with zeros:** the tensor is stored as-is. No storage benefit, no kernel benefit unless the hardware skips zeros.
- **Bitmap plus values:** one bit per weight indicating nonzero, plus the nonzero values. Storage is `N/8 + nnz × bytes_per_value`.
- **Index plus values (CSR/COO):** each nonzero stores its value and its index. Storage is `nnz × (bytes_per_value + bytes_per_index) + row_pointers`.

At high sparsity, the index dominates. At 70% sparsity with 1-byte values and 2-byte indices, the index costs twice the value, and the "sparse" tensor can be larger than the dense one. Section 6 works through the arithmetic.

### 3.6 The pruning pipeline

A pruning procedure has five stages, and each has a decision that affects the result.

```text
   trained model
        │
        ▼
   score weights ──► criterion: magnitude, movement, gradient, Hessian
        │
        ▼
   select mask ────► granularity: weight, group, channel, block
        │
        ▼
   apply mask ─────► schedule: one-shot, iterative, gradual
        │
        ▼
   fine-tune ──────► data, learning rate, mask updates
        │
        ▼
   evaluate ───────► accuracy, calibration, per-class, subgroups
```

The stages are not independent. A criterion that ranks weights well at 50% sparsity may rank them poorly at 90%, because the remaining weights interact. A schedule that works with magnitude may not work with movement. A granularity that suits a dense kernel may not suit a sparse one.

### 3.7 What sparsity costs

Sparsity is often presented as free compression. It is not.

- **Accuracy:** removing weights removes capacity, and the loss is not uniform across classes.
- **Engineering:** the mask, the schedule, and the fine-tuning must be reproduced.
- **Validation:** the pruned model is a different model and needs its own evidence.
- **Regulatory:** the change must be classified and, if not covered, submitted.
- **Maintenance:** a pruned model may be harder to update, because a new training run produces a different mask.

The costs are real and are often larger than the storage saved. A team should prune when a budget requires it, not because sparsity is fashionable.

A useful test is to ask what the team would do with the saved bytes. If the answer is "nothing in particular", the pruning is an experiment, not a product decision.

### 3.8 The three questions a pruning review asks

1. **Which level does this pruning target?** Algorithmic, storage, kernel, or hardware. The answer determines the evidence.
2. **What is the pattern, and does the target exploit it?** Unstructured, structured, or N:M, and the kernel or accelerator that supports it.
3. **What happened to the rare classes?** Aggregate accuracy is not sufficient; per-class and subgroup results are the deployment evidence.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Pruning | Removing weights or structures from a trained model | Assumed to reduce latency | Method |
| Sparsity | Fraction of weights equal to zero | Assumed to imply speedup | Proportion |
| Density | Fraction of weights that are nonzero | Confused with sparsity | Proportion |
| Unstructured pruning | Removing individual weights | Called "sparse pruning" | Method |
| Structured pruning | Removing whole channels, heads, or blocks | Called "coarse pruning" | Method |
| Channel pruning | Removing entire output channels | Confused with filter pruning | Method |
| Filter pruning | Removing entire convolution filters | Confused with channel pruning | Method |
| Block pruning | Removing contiguous blocks | Confused with N:M | Method |
| N:M sparsity | At most N nonzeros in every M | Confused with N/M density | Pattern |
| 2:4 sparsity | Two nonzeros in every four | Assumed supported everywhere | Pattern |
| Fine-grained sparsity | Another name for unstructured | Confused with structured | Pattern |
| Coarse-grained sparsity | Another name for structured | Confused with block | Pattern |
| Magnitude pruning | Removing weights with smallest absolute value | Assumed optimal | Criterion |
| Movement pruning | Removing weights whose value is moving toward zero | Confused with magnitude | Criterion |
| Gradient pruning | Removing weights with small gradient | Confused with movement | Criterion |
| Hessian-based pruning | Removing weights with small second-order effect | Assumed exact | Criterion |
| One-shot pruning | Pruning once, then fine-tuning | Assumed equivalent to iterative | Schedule |
| Iterative pruning | Repeated prune-and-train cycles | Confused with gradual | Schedule |
| Gradual pruning | Increasing sparsity during training | Confused with iterative | Schedule |
| Lottery ticket hypothesis | Dense networks contain sparse trainable subnetworks | Treated as a deployment recipe | Hypothesis |
| Winning ticket | A sparse subnetwork that trains to dense accuracy from its initialization | Confused with a pruned model | Research object |
| Rewinding | Restoring weights to an early training checkpoint | Confused with resetting to initialization | Method |
| Mask | Binary tensor indicating retained weights | Confused with the pruned model | Tensor |
| Sparsity pattern | The arrangement of nonzeros | Assumed irrelevant to hardware | Structure |
| CSR | Compressed sparse row storage | Confused with CSC | Format |
| COO | Coordinate list storage | Confused with CSR | Format |
| Bitmap | One bit per weight indicating nonzero | Confused with a mask in memory | Format |
| Index overhead | Bytes used to describe positions | Ignored in storage estimates | Byte |
| Block sparsity | Sparsity at block granularity | Confused with N:M | Pattern |
| Sparse kernel | Kernel that skips zeros | Assumed present when sparsity is present | Implementation |
| Dense kernel | Kernel that multiplies all values | Assumed to exploit sparsity | Implementation |
| Sparse tensor core | Hardware unit for structured sparsity | Assumed to support arbitrary patterns | Hardware |
| Speedup | Reduction in latency | Confused with reduction in storage | Ratio |
| Compression ratio | Original size divided by compressed size | Applied to latency | Ratio |
| Pruning ratio | Fraction of weights removed | Confused with sparsity | Proportion |
| Recovery | Accuracy regained by fine-tuning | Assumed to be complete | Proportion |
| Cliff | Sparsity beyond which accuracy falls sharply | Ignored in target selection | Point |
| Importance score | Criterion used to rank weights for removal | Assumed to be model-free | Score |
| Layer-wise sparsity | Different sparsity per layer | Confused with global sparsity | Allocation |
| Global sparsity | One sparsity target across all layers | Confused with per-layer | Allocation |
| Structured sparsity | Sparsity at channel or block granularity | Confused with 2:4 | Pattern |

Two definitions are especially prone to misuse.

**"Sparsity" must be qualified by pattern and level.** "70% sparsity" says nothing about whether the zeros are structured, whether they are stored compactly, or whether the kernel skips them. A procurement document that asks for "70% sparsity" is asking for a number, not a capability.

**"Pruning ratio" is not "compression ratio".** A 70% pruning ratio removes 70% of the weights, but the compressed file may be smaller by 30% or larger than the dense file, depending on the storage format and the index width. The two ratios are related but not equal.

### 4.1 Storage formats at a glance

| Format | Bytes per nonzero | Metadata | Best at | Worst at |
|---|---|---|---|---|
| Dense | 1 (INT8) | None | Low sparsity | Any sparsity |
| Bitmap | 1 + 0.125 | 1 bit per weight | Moderate to high sparsity | Very low sparsity |
| CSR | 1 + 2 | Row pointers | High sparsity, row-structured | Moderate sparsity |
| COO | 1 + 2 + 2 | Per-coordinate row and column | Irregular sparsity | Structured tensors |
| 2:4 | 1 + 0.125 | Pattern selector | Exactly 50% and supported hardware | Other sparsity levels |
| Block sparse | 1 + small | Block mask | Block-structured hardware | Fine-grained patterns |

The table makes the trade-off visible. Bitmap storage has the lowest metadata per weight; CSR has the highest per nonzero. At low sparsity, the metadata of any sparse format can exceed the savings, which is why dense storage often wins below about 30–40% sparsity.

## 5. The mechanics, step by step

The procedure produces a pruned model with evidence. It assumes a trained model, a target, and a constraint ledger.

### Step 1. Choose the target level of sparsity benefit

**Input:** The binding resource from the profile.

**Output:** A decision: storage-only pruning, latency-oriented pruning, or no pruning.

**Rationale:** The level determines the pattern. Storage-only pruning can be unstructured and aggressive. Latency-oriented pruning must be structured or match a hardware pattern. Confusing the two leads to a model that is small and slow.

The decision table:

| Binding resource | Target level | Pattern |
|---|---|---|
| Flash or storage | Storage | Unstructured, with a compact format |
| Latency on dense hardware | Kernel | Structured (channels or blocks) |
| Latency on 2:4 hardware | Hardware | 2:4, enforced during training |
| Memory bandwidth | Storage and kernel | Structured, plus quantization |
| No resource binding | None | Do not prune |

**Branch point:** If the binding resource is accuracy or calibration, pruning is not the answer. Return to the model or data.

**Edge case:** A model can be bound by flash and latency at once. Structured pruning addresses both; unstructured addresses only flash.

### Step 2. Choose the pruning granularity

**Input:** The target level and the hardware's supported patterns.

**Output:** A granularity per layer.

**Rationale:** Granularity determines both the accuracy cost and the hardware benefit. Finer granularity preserves accuracy; coarser granularity maps to hardware.

| Granularity | Accuracy at fixed sparsity | Hardware benefit | Typical use |
|---|---|---|---|
| Individual weights | Best | None on dense hardware | Storage only |
| Groups of 4 (2:4) | Good | 2× on sparse tensor cores | Supported accelerators |
| Blocks of 4×1 or 4×4 | Good | Block sparse kernels | Some accelerators |
| Channels | Moderate | Direct, on any dense kernel | Latency on MCU/CPU |
| Heads (transformers) | Moderate | Direct | Attention layers |
| Layers | Worst | Direct | Architecture search |

**Branch point:** If the hardware supports only 2:4, do not prune to 70% unstructured and hope. The pattern will not be exploited, and the accuracy cost buys nothing at runtime.

**Edge case:** A layer with few channels cannot be pruned structurally without removing a large fraction of its capacity. Exclude such layers and record the exclusion.

### Step 3. Choose the importance criterion

**Input:** The trained model and, optionally, gradients or a calibration set.

**Output:** An importance score per weight or structure.

**Rationale:** The criterion decides which weights are removed. Magnitude is the default because it is cheap and works well; other criteria can do better at high sparsity or for fine-tuning.

| Criterion | Score | Cost | Strength | Weakness |
|---|---|---|---|---|
| Magnitude | abs(weight) | Low | Simple, effective | Ignores input distribution |
| Movement | weight × gradient | Medium | Works well in fine-tuning | Needs gradients |
| Gradient | abs(gradient) | Medium | Sensitive to current loss | Noisy |
| Hessian | Second-order importance | High | Accurate at high sparsity | Expensive |
| Activation-aware | Weight × input scale | Medium | Accounts for data | Needs calibration data |
| Random | Random score | Low | Baseline | Worse than magnitude |

**Branch point:** If the model is being pruned during training, movement pruning often outperforms magnitude at high sparsity. If the model is pruned after training, magnitude is the practical default.

**Edge case:** A criterion computed on one data distribution may rank weights differently on another. If the deployment population differs from the pruning data, evaluate the pruned model on the deployment distribution.

### Step 4. Choose the schedule and the target

**Input:** The accuracy budget and the compute budget for pruning.

**Output:** A schedule and a sparsity target per layer.

**Rationale:** One-shot pruning at a high target usually costs more accuracy than iterative pruning. Gradual pruning during training often costs the least but requires a full training run.

| Schedule | How | Accuracy | Cost | When to use |
|---|---|---|---|---|
| One-shot | Prune once, fine-tune | Lowest | Low | Quick estimate, low sparsity |
| Iterative | Prune, train, repeat | Higher | Medium | Most post-training work |
| Gradual | Increase sparsity during training | Highest | High | Training from scratch |
| Layer-wise | Different sparsity per layer | Higher | Medium | Heterogeneous layers |

A practical iterative schedule: prune 10–20% of the target per cycle, fine-tune for a short period, evaluate, and repeat until the target or the accuracy floor is reached.

**Branch point:** If accuracy falls below the floor at an intermediate step, stop and use the previous sparsity. The previous model is the deliverable.

**Edge case:** A schedule that prunes uniformly across layers can over-prune a sensitive layer and under-prune a redundant one. Layer-wise sensitivity analysis is worth the cost for small models.

### Step 5. Apply the mask and fine-tune

**Input:** The importance scores and the schedule.

**Output:** A pruned model with a mask and a fine-tuned checkpoint.

**Rationale:** Pruning removes capacity; fine-tuning lets the remaining weights compensate. Without fine-tuning, the accuracy loss is larger and the model is not representative of what pruning can achieve.

The procedure:

1. Compute the importance scores.
2. Select the threshold or the top-k per layer or globally.
3. Apply the mask.
4. Fine-tune with the mask fixed, or with the mask updated periodically.
5. Evaluate at each checkpoint.
6. Stop when the accuracy floor is reached or the schedule ends.

**Branch point:** If fine-tuning does not recover the loss, the sparsity is too high or the criterion is wrong. Reducing the target is cheaper than adding epochs.

**Edge case:** Fine-tuning on the same data used to compute the importance scores can overfit the pruning decision. Use a separate validation set for the stopping decision.

### Step 6. Enforce the hardware pattern if required

**Input:** The pruned model and the hardware pattern.

**Output:** A model that conforms to the pattern, or a decision that the pattern is not worth it.

**Rationale:** A pattern that the hardware does not recognize is just unstructured sparsity with extra constraints. Enforcing 2:4 after pruning can reduce accuracy further, because the constraint may remove weights that magnitude pruning wanted to keep.

The procedure for 2:4:

1. Rank weights within each group of four by importance.
2. Keep the top two, zero the rest.
3. Fine-tune with the pattern enforced.
4. Verify that the runtime selects the sparse path.

**Branch point:** If the accuracy cost of enforcing 2:4 exceeds the latency benefit, use structured pruning instead, which has a different accuracy-latency trade-off.

**Edge case:** Some layers are too small for 2:4 to help, because the sparse kernel has a minimum size or a shape constraint. Exclude them.

### Step 7. Measure storage with the actual format

**Input:** The pruned model and the storage format.

**Output:** Bytes for values, indices, and metadata, and the ratio against dense.

**Rationale:** The nominal compression ratio ignores index overhead. The actual file size is what matters for flash.

```text
CSR:    nnz × value_bytes + nnz × index_bytes + (rows + 1) × pointer_bytes
bitmap: nnz × value_bytes + numel / 8
N:M:    nnz × value_bytes + numel × pattern_bits / 8
```

**Branch point:** If the sparse format is larger than dense, use bitmap storage or do not prune for storage. A sparse format that does not save bytes is a liability.

**Edge case:** Alignment and padding can add bytes. A format that packs 4-bit values may pad each row to a byte boundary.

### Step 8. Evaluate accuracy, calibration, and subgroups

**Input:** The pruned model, the held-out set, and the subgroup definitions.

**Output:** A comparison table with confidence intervals.

**Rationale:** Pruning can remove a rare-class feature while leaving aggregate accuracy unchanged. The subgroup and per-class tables are the only way to see it.

The evaluation includes:

- Overall accuracy, sensitivity, and specificity at the fixed threshold.
- Per-class sensitivity, especially for rare classes.
- Subgroup performance with confidence intervals.
- Calibration: ECE, slope, Brier score.
- Performance on edge cases that depend on small features.

**Branch point:** If a rare class loses more than its allocated accuracy, reduce sparsity for the layer that carries it or exclude that layer.

**Edge case:** A pruned model can have better calibration by accident, because removing weights reduces overconfidence. Check sensitivity before celebrating.

### Step 9. Combine with quantization in the right order

**Input:** The pruned model and the quantization plan.

**Output:** A pruned and quantized model with accuracy and target measurements.

**Rationale:** Pruning changes the weight distribution that quantization must represent. Quantizing after pruning means the scales see the final distribution. Quantizing before pruning means the pruning decision is made on rounded values, and the scales are invalid after pruning.

The order:

1. Prune and fine-tune.
2. Recompute activation ranges on calibration data.
3. Quantize with per-channel weight scales.
4. Evaluate.
5. If accuracy fails, use QAT starting from the pruned model.

**Branch point:** If the pruned model is very sparse, quantization may have less to gain because the remaining weights are fewer. Evaluate the combined benefit, not the sum of the individual benefits.

**Edge case:** A structured-pruned layer may have a channel count that does not match the kernel's preferred vector width. The latency benefit can be less than the parameter reduction predicts.

### Step 10. Record the pattern and classify the change

**Input:** All artifacts.

**Output:** A sparsity record and a change classification.

**Rationale:** A pruned model's behavior depends on the exact pattern, not only on the sparsity level. Two models with 70% sparsity can behave differently if the retained weights differ.

The record includes:

- The importance criterion and its parameters.
- The schedule and the target per layer.
- The mask, or a hash of it, and the seed.
- The fine-tuning data and schedule.
- The storage format and byte counts.
- Accuracy, calibration, and subgroup results.
- Target latency and energy.
- Quantization applied after pruning.
- Change classification.

**Branch point:** If the mask cannot be reproduced, the model is not reproducible. Store the mask or the seed and the exact procedure.

**Edge case:** A pattern that depends on a random seed is reproducible only if the seed is recorded. Record it.

### 5.1 Decision table for pruning outcomes

| Observation | Likely cause | Next action |
|---|---|---|
| Storage falls, latency flat | Dense kernels | Structured pruning or sparse hardware |
| Storage rises above dense | Index overhead | Bitmap or lower index width |
| Accuracy flat, then cliff | Normal sparsity behavior | Stop before the cliff |
| Rare class degrades first | Small weights carry the class | Reduce sparsity; per-layer targets |
| Calibration shifts | Removed weights changed confidence | Recalibrate; check threshold |
| Structured prune loses more than expected | Whole channel removed | Try block or N:M |
| 2:4 slower than dense | Sparse path not selected | Check runtime flags and shapes |
| Pruned model not faster after export | Pattern lost in conversion | Inspect the exported graph |
| Fine-tuning does not recover | Sparsity too high | Lower target |
| Results vary across seeds | Near the cliff | Average over seeds; lower target |

### 5.2 Edge cases in pruning

**Outlier channels.** A channel with a large weight norm may be kept by every criterion, even if it contributes little to accuracy. Conversely, a channel with a small norm may be essential for a rare class. Magnitude is a proxy, not a measure of importance.

**Per-tensor versus per-channel scales.** After pruning, per-channel quantization is still preferable for weights, because the remaining channels can have very different ranges. Recompute the scales after pruning; do not reuse the pre-pruning values.

**Activation clipping.** Pruning changes the activation distribution, so activation ranges must be recalibrated. A range computed before pruning may clip after it.

**Operator fallback.** A sparse model may use gather or scatter operators that the accelerator does not support. The fallback can erase the benefit.

**Unsupported operators.** A sparse kernel may exist only for certain layer types or shapes. Check the support list before choosing the pattern.

**Zero-point drift.** Pruning changes the weight distribution and the activation distribution. Zero points computed before pruning may no longer represent real zero after it.

**Calibration-set bias.** Pruning decisions made on one site's data may remove features that matter at another site. Evaluate on a site-stratified set.

**Thermal throttling.** A pruned model that runs faster may also run at a higher clock, changing the thermal profile. Measure sustained behavior.

### 5.3 Layer-wise sensitivity

Uniform sparsity is a convenience, not an optimum. Layers differ in how much redundancy they contain, and a uniform target over-prunes the sensitive ones and under-prunes the redundant ones.

A sensitivity analysis prunes one layer at a time to a candidate sparsity, evaluates the model, and records the accuracy change. The result is a table:

| Layer | Parameters | Share of MACs | Accuracy at 50% | Accuracy at 70% | Suggested target |
|---|---:|---:|---:|---:|---:|
| Conv 1 | 40 | 1.0% | -0.1 | -0.4 | 30% |
| Conv 2 | 640 | 8.0% | -0.2 | -0.6 | 50% |
| Conv 3 | 2,560 | 15.9% | -0.3 | -1.1 | 50% |
| FC | 32,768 | 0.8% | -0.4 | -1.3 | 70% |

The table is illustrative, but the pattern is common: the input layer is sensitive because its features are reused everywhere, and the large fully connected layer is tolerant because it is over-parameterized.

Layer-wise targets usually beat a uniform target at the same average sparsity, at the cost of a more complex schedule and a more complex record. For a small model, the analysis is cheap and worth doing.

### 5.4 A pruning checklist

Before committing to a pruning run, confirm each item.

- The binding resource is storage, latency, or both, and the target level is stated.
- The granularity matches the hardware's supported pattern.
- The importance criterion is appropriate for post-training or in-training use.
- The sparsity target was chosen from a sensitivity analysis or a sweep, not a round number.
- The schedule prunes in increments and evaluates at each.
- The fine-tuning data is separate from the data used to select the mask.
- The evaluation includes per-class and subgroup results.
- The storage format and its byte count are known, not assumed.
- The exported graph preserves the pattern.
- The target measurement confirms the benefit, if a latency benefit is claimed.
- The mask or seed is recorded.
- The change is classified against the approved plan.

A pruning run that cannot answer all twelve is not ready for a design review, regardless of its sparsity level.

### 5.5 How sparsity interacts with batch and layout

Two runtime details can erase a benefit that the arithmetic predicts.

**Batch size.** At batch 1, a sparse matrix-vector product is memory-bound, so reducing nonzeros can reduce time even without a sparse kernel, if the sparse format is used and the kernel reads only the stored values. At larger batches, the same matrix becomes compute-bound, and the sparse kernel is needed to see a gain. A sparsity result measured at batch 1 may not transfer to batch 16.

**Layout.** A sparse format that stores indices in the wrong order, or values that are not contiguous along the kernel's access pattern, can be slower than dense despite fewer bytes. A bitmap is compact but requires a branch or a mask multiply per weight, which can cost more than it saves on a simple core.

The general rule: a sparse format helps when the runtime reads fewer bytes or executes fewer operations. If it reads the same bytes and executes the same operations, it is a storage format, not a performance feature.

A third detail is the compiler. A sparse kernel may be selected only when the layer's dimensions are known at compile time. A model exported with dynamic shapes can fall back to the dense path even when the pattern is present and the hardware supports it. Fixing the shapes is often the difference between a sparse deployment and a dense one.

### 5.6 What changes after export

The exported model is not the training model. Three transformations commonly affect sparsity.

- **Constant folding** can replace a masked weight tensor with a constant, removing the mask and the sparsity along with it.
- **Layout conversion** can reorder weights so that the pattern no longer matches the kernel's expected group boundaries, especially for N:M.
- **Quantization** can reorder or pack values, and a sparse format combined with a packed format may not be supported by the runtime.

The test is to inspect the exported graph for the sparse operators and to compare the file size with the expected sparse size. A file that matches the dense size has lost the sparsity, regardless of what the training log says.

## 6. Worked example

This is a constructed case with arithmetic that closes. The model and measurements are illustrative.

### 6.1 The model and the target

The ECG model has 41,216 parameters and 4,096,000 MACs. Its fully connected layer has 512 output channels and 64 inputs, or 32,768 weights. In INT8 the model occupies 41,816 bytes and runs at 45.7 ms on the Cortex-M4F at 64 MHz. The latency budget is 40 ms, so the dense INT8 model misses it by 5.7 ms.

The question is whether pruning can close the gap, and at what accuracy cost.

### 6.2 Unstructured pruning at 70%

A global magnitude prune removes the 70% of weights with the smallest absolute value. The model retains 12,365 of 41,216 weights.

Stored in compressed sparse row format with 1-byte values, 2-byte column indices, and 4-byte row pointers over 129 rows:

```text
values        = 12,365 × 1 = 12,365 B
column index  = 12,365 × 2 = 24,730 B
row pointers  = 129 × 4     =    516 B
total         = 37,611 B    = 36.7 KiB
```

Stored as a bitmap plus values:

```text
bitmap = 41,216 / 8 = 5,152 B
values = 12,365 × 1 = 12,365 B
total  = 17,517 B = 17.1 KiB
```

The two formats differ by a factor of 2.1. The bitmap is smaller because it pays one bit per weight instead of two bytes per nonzero. At 70% sparsity, the index is the dominant cost in CSR.

For comparison, the dense INT8 model is 41,216 bytes. CSR saves only 8.8%; the bitmap saves 57.5%. The nominal "70% compression" is not what either format delivers.

### 6.3 Why latency does not change

The Cortex-M4F has no sparse kernel. The runtime executes the dense convolution and linear loops, multiplying every weight, including the zeros. The arithmetic count is unchanged at 4,096,000 MACs, and the latency remains 45.7 ms.

This is the central result of the topic. The model is 70% sparse at the algorithmic level and 0% sparse at the kernel and hardware levels. The storage benefit is real; the latency benefit is zero.

### 6.4 Structured pruning at 40%

Removing 40% of the output channels across the convolutional stack and the fully connected layer reduces parameters to:

```text
41,216 × 0.60 = 24,730
```

The MAC count falls further than the parameter count because removing a channel also removes its contribution to the next layer:

```text
4,096,000 × 0.55 = 2,252,800
```

The model bytes:

```text
24,730 + 72 channels × 5 = 25,090 B = 24.5 KiB
```

The latency, assuming the same effective throughput:

```text
2,252,800 / 89,600,000 = 0.02514 s = 25.1 ms
```

Energy:

```text
3.0 × 0.0082 × 0.02514 = 0.618 mJ
```

Structured pruning meets the 40 ms budget with 14.9 ms of margin and reduces memory by 40% relative to dense INT8. The cost is accuracy: sensitivity falls from 0.938 to 0.930 in the constructed case.

### 6.5 2:4 sparsity

A 2:4 pattern keeps the two largest weights in every group of four, for 50% sparsity and 20,608 nonzeros. The storage is the values plus a 2-bit selector per pair, or 1 bit per weight:

```text
values   = 20,608 B
selector = 41,216 / 8 = 5,152 B
total    = 25,760 B = 25.2 KiB
```

On hardware with sparse tensor cores, the MAC count halves to 2,048,000 and the latency falls to 22.9 ms. On the Cortex-M4F, which has no sparse path, the latency remains 45.7 ms even though the pattern is regular.

The lesson is that 2:4 is a hardware contract, not a software optimization. It pays only where the contract is supported.

### 6.6 Summary of the routes

| Route | Params | Bytes | KiB | MACs | Latency (ms) | Energy (mJ) |
|---|---:|---:|---:|---:|---:|---:|
| Dense INT8 | 41,216 | 41,816 | 40.8 | 4,096,000 | 45.7 | 1.124 |
| Unstructured 70%, CSR | 12,365 | 37,611 | 36.7 | 4,096,000 | 45.7 | 1.124 |
| Unstructured 70%, bitmap | 12,365 | 17,517 | 17.1 | 4,096,000 | 45.7 | 1.124 |
| Structured 40% | 24,730 | 25,090 | 24.5 | 2,252,800 | 25.1 | 0.618 |
| 2:4 on sparse hardware | 20,608 | 25,760 | 25.2 | 2,048,000 | 22.9 | 0.563 |
| 2:4 on Cortex-M4F | 20,608 | 25,760 | 25.2 | 4,096,000 | 45.7 | 1.124 |
| Unstructured 90%, bitmap | 4,122 | 9,274 | 9.1 | 4,096,000 | 45.7 | 1.124 |

### 6.7 Accuracy and calibration cost

| Route | Sensitivity | Specificity | ECE | Worst subgroup | Rare-class sensitivity |
|---|---:|---:|---:|---:|---:|
| Dense INT8 | 0.938 | 0.958 | 0.029 | 0.926 | 0.884 |
| Unstructured 70% | 0.927 | 0.951 | 0.034 | 0.912 | 0.861 |
| Structured 40% | 0.930 | 0.953 | 0.038 | 0.914 | 0.870 |
| 2:4 | 0.933 | 0.955 | 0.032 | 0.921 | 0.876 |
| Unstructured 90% | 0.905 | 0.938 | 0.041 | 0.887 | 0.812 |

The rare class degrades faster than the aggregate in every route. At 70% unstructured sparsity the overall sensitivity falls by 1.1 points, but the rare-class sensitivity falls by 2.3 points and the worst subgroup by 1.4. At 90% sparsity the rare class loses 7.2 points while the aggregate loses 3.3.

This is the pattern that magnitude pruning produces: the smallest weights are often the ones that carry the least common signal, and removing them costs the rare class disproportionately.

### 6.8 Diagnostic one: outlier-driven scale collapse after pruning

Pruning changes the weight distribution. On the fully connected layer, the median absolute weight rises from 0.0610 before pruning to 0.1296 after a 70% magnitude prune, because the small weights are removed. The maximum remains 1.2912.

The per-tensor symmetric step is:

```text
S = 1.2912 / 127 = 0.01016693
step / median = 0.01016693 / 0.1296 = 0.0784
```

The ratio falls from 0.167 to 0.078, which is an improvement: the surviving weights are better represented because the scale is now smaller relative to their typical magnitude. The per-tensor SQNR rises from 30.00 dB to 34.12 dB.

The diagnostic is still positive, because 7.8% is close to the 10% threshold and the outlier still dominates the scale. Per-channel quantization remains the remedy.

The important point is that pruning and quantization interact. The pruning made quantization easier by removing the small weights that were being rounded to zero. A team that quantizes first and prunes second does not get this benefit, and may instead remove weights that were already coarsely represented.

### 6.9 Diagnostic two: per-channel skew after pruning

The 512 channels still have ranges from 0.2756 to 2.3202, a ratio of 8.4, because pruning by global magnitude removes small weights uniformly rather than equalizing channel ranges.

Under a per-tensor scale, a typical channel gets about 28 levels. Per-channel scales raise SQNR to 44.67 dB. The pruning did not fix the skew, and it did not make it worse; the skew is a property of the channel structure, not of the sparsity.

The diagnostic is the channel range ratio. Above about 4, per-tensor quantization will cost material accuracy, regardless of sparsity.

### 6.10 Varying one assumption

**Sparsity: 70% to 50%.** The model retains 20,608 weights. CSR storage becomes 20,608 + 41,216 + 516 = 62,340 bytes, which is **larger** than the 41,216-byte dense model, because the index costs more than the values saved. Bitmap storage is 20,608 + 5,152 = 25,760 bytes. At 50% sparsity, CSR is a liability and bitmap is the only viable format.

**Sparsity: 70% to 90%.** The model retains 4,122 weights. CSR is 4,122 + 8,244 + 516 = 12,882 bytes; bitmap is 4,122 + 5,152 = 9,274 bytes. Storage improves, but sensitivity falls to 0.905, below the 0.925 floor, and the rare class falls to 0.812. The storage gain is real and the accuracy cost is unacceptable.

**Granularity: unstructured to structured at the same 50% sparsity.** Structured pruning at 50% removes half the channels, which reduces MACs and bytes directly. Unstructured 50% reduces neither latency nor, in CSR, storage. The same sparsity number produces a 1.8× latency difference.

**Layer excluded: the first convolution kept dense.** The first convolution has 40 weights, 0.1% of the model. Keeping it dense costs 40 bytes, or 0.23% of the pruned bitmap model, and recovers about 0.3 points of rare-class sensitivity because the input layer's features are not redundant. Excluding a small, sensitive layer is cheap.

**Quantization order: prune then quantize versus quantize then prune.** Pruning first raises the median weight and improves per-tensor SQNR from 30.0 to 34.1 dB. Quantizing first and pruning second leaves the scales computed for the dense distribution, which no longer match the pruned weights, and requires recalibration anyway. The order matters, and the benefit of pruning first is measurable.

### 6.11 The decision

Against the two constraints, the dense INT8 model fails latency (45.7 ms against 40 ms) and passes flash (40.8 KiB). Unstructured pruning at 70% passes neither the latency constraint nor, in CSR, a meaningful storage improvement. Structured pruning at 40% meets latency at 25.1 ms and flash at 24.5 KiB, but costs 0.8 points of overall sensitivity and 1.4 points of rare-class sensitivity.

The 2:4 route meets latency only if the target has sparse tensor cores. On the Cortex-M4F it does not, so the route is not deployable on this device even though the pattern is regular.

The decision is therefore structured pruning at 40%, conditional on the rare-class result. If the rare-class sensitivity of 0.870 is below the clinical requirement, the options are a lower sparsity target, a layer-wise target that protects the layers carrying the rare-class feature, or a larger model with a different architecture.

The case teaches the general rule: pruning is selected by the hardware's supported pattern, sized by the accuracy budget, and validated on the classes that are most likely to be damaged.

## 7. Code, query, or calculation

The script implements magnitude pruning, 2:4 selection, CSR and bitmap accounting, quantization, and error metrics. It reproduces the layer-level numbers in Section 6.

```python
import math
import random


def magnitude_threshold(values, sparsity):
    """Return the smallest magnitude retained at the target sparsity."""
    ordered = sorted((abs(v) for v in values), reverse=True)
    keep = max(1, int(round(len(values) * (1 - sparsity))))
    return ordered[keep - 1]


def prune_magnitude(values, sparsity):
    threshold = magnitude_threshold(values, sparsity)
    return [v if abs(v) >= threshold else 0.0 for v in values]


def prune_2_4(values, fan_in):
    """Keep the two largest magnitudes in every group of four."""
    out = []
    for row_start in range(0, len(values), fan_in):
        row = values[row_start:row_start + fan_in]
        for i in range(0, len(row), 4):
            group = row[i:i + 4]
            keep = sorted(range(len(group)), key=lambda j: abs(group[j]),
                          reverse=True)[:2]
            out.extend([group[j] if j in keep else 0.0
                        for j in range(len(group))])
    return out
```

Storage accounting:

```python
def csr_bytes(nnz, rows, value_bytes=1, index_bytes=2, pointer_bytes=4):
    return nnz * value_bytes + nnz * index_bytes + (rows + 1) * pointer_bytes


def bitmap_bytes(nnz, numel, value_bytes=1):
    return nnz * value_bytes + numel / 8


def nm_bytes(nnz, numel):
    """Values plus one selector bit per weight for 2:4."""
    return nnz + numel / 8


numel = 41_216
for sparsity in (0.50, 0.70, 0.90):
    nnz = int(round(numel * (1 - sparsity)))
    print("sparsity %.0f%% nnz %d csr %d bitmap %d"
          % (sparsity * 100, nnz, csr_bytes(nnz, 129), bitmap_bytes(nnz, numel)))

nnz24 = numel // 2
print("2:4 nnz", nnz24, "bytes", nm_bytes(nnz24, numel))
print("dense int8", numel)
```

Expected output:

```text
sparsity 50% nnz 20608 csr 62340 bitmap 25760
sparsity 70% nnz 12365 csr 37611 bitmap 17517
sparsity 90% nnz 4122 csr 12882 bitmap 9274
2:4 nnz 20608 bytes 25760.0
dense int8 41216
```

Quantization and error metrics after pruning:

```python
def round_half_away(value):
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def symmetric_scale(values, bits=8):
    amax = max(abs(v) for v in values)
    return amax / (2 ** (bits - 1) - 1)


def quantize_symmetric(values, scale, bits=8):
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax - 1
    return [min(qmax, max(qmin, round_half_away(v / scale))) for v in values]


def dequantize(codes, scale, zero_point=0):
    return [(q - zero_point) * scale for q in codes]


def error_metrics(original, reconstructed):
    n = len(original)
    mse = sum((a - b) ** 2 for a, b in zip(original, reconstructed)) / n
    signal = sum(a * a for a in original)
    noise = sum((a - b) ** 2 for a, b in zip(original, reconstructed))
    return {"mae": sum(abs(a - b) for a, b in zip(original, reconstructed)) / n,
            "rmse": math.sqrt(mse),
            "sqnr_db": 10 * math.log10(signal / noise) if noise else float("inf")}


random.seed(11)
channels, fan_in = 512, 64
weights = [[round(random.gauss(0, 0.09), 4) for _ in range(fan_in)]
           for _ in range(channels)]
weights[7] = [round(v * 6.0, 4) for v in weights[7]]
weights[7][3] = 0.62
flat = [v for row in weights for v in row]

before = flat
after = [v for v in prune_magnitude(flat, 0.70) if v != 0.0]
for name, data in (("before", before), ("after", after)):
    scale = symmetric_scale(data, 8)
    codes = quantize_symmetric(data, scale, 8)
    median = sorted(abs(v) for v in data)[len(data) // 2]
    print(name, "n", len(data), "median %.4f" % median,
          "step/median %.4f" % (scale / median),
          "sqnr %.2f" % error_metrics(data, dequantize(codes, scale))["sqnr_db"])
```

Expected output:

```text
before n 32768 median 0.0610 step/median 0.1667 sqnr 30.00
after n 9832 median 0.1296 step/median 0.0784 sqnr 34.12
```

Assumptions: pruning is global magnitude with a threshold; rows are 129 for CSR pointers; values are 1 byte; indices are 2 bytes; the 2:4 selector is 1 bit per weight; the quantization is symmetric per-tensor; and the error metrics are computed on the retained values only.

## 8. Data, standards, and regulatory context

Pruning has no dedicated standard. The relevant constraints come from the device software lifecycle, risk management, and the change-control pathway for AI-enabled devices. The technical standards that matter are the hardware and runtime specifications that determine whether sparsity is exploitable.

### 8.1 IEC 62304

IEC 62304, "Medical device software — Software life cycle processes", Edition 1.1 (2006 plus Amendment 1:2015), recognized by the United States Food and Drug Administration (FDA) under recognition number 13-79, requires the pruning procedure to be a documented design activity. The sparsity pattern, the criterion, the fine-tuning data, and the evaluation are design records.

The standard also requires configuration management. A mask is a configuration item: two models with the same sparsity but different masks are different models, and a record that stores only "70% pruned" does not identify which one was validated.

### 8.2 ISO 14971

ISO 14971, "Medical devices — Application of risk management to medical devices", requires hazards to be identified and controlled. Pruning introduces a specific hazard: the loss of a rare-class feature that the aggregate metric does not detect.

The risk file should record:

- The classes and subgroups most likely to depend on small weights.
- The evaluation that checked them.
- The residual risk if a rare class degrades.
- The monitoring that would detect the degradation after deployment.

A pruned model without a rare-class evaluation is an uncontrolled risk, regardless of its overall accuracy.

### 8.3 Hardware and runtime specifications

The technical constraints are documented by the hardware and runtime vendors.

- NVIDIA's Ampere architecture implements 2:4 structured sparsity with sparse tensor cores, and the A100 whitepaper describes the pattern and the sparse MMA semantics. cuSPARSELt provides the API for pruning and compressing to the pattern, and TensorRT can use sparse weights when the builder flag is set.
- Arm's Ethos-U85 documentation lists 2:4 structured sparsity as a supported feature, with the note that it can double throughput for supported layers.
- Runtime documentation lists which operators have sparse kernels. A runtime without a sparse convolution kernel will execute the dense path regardless of the pattern.

These are vendor specifications, not regulations, but they determine whether the pruning has a hardware effect. A regulatory submission that claims a latency benefit must rest on a measurement, not on a specification.

### 8.4 FDA PCCP and the change pathway

The FDA's final guidance on Predetermined Change Control Plans for AI-enabled device software functions was issued on 3 December 2024, with a Federal Register notice on 4 December 2024, under section 515C of the Food, Drug, and Cosmetic Act as added by the Food and Drug Omnibus Reform Act of 2022.

Pruning changes the model's weights and structure. Whether it is covered depends on what the plan anticipated. A plan that describes "periodic re-training of the approved architecture" may not cover a change in sparsity pattern. A plan that describes "structured pruning within a specified range, validated by a specified protocol" might.

The distinction is factual, and it must be checked against the plan before the pruning work is complete.

### 8.5 EU framework

In the European Union, Regulation (EU) 2017/745 governs the device and Regulation (EU) 2024/1689, the Artificial Intelligence Act, adds obligations for high-risk systems. A medical device that is a safety component, or is itself a device requiring third-party conformity assessment, is high-risk under Article 6(1).

The Annex I applicability date was originally 2 August 2027; the AI Omnibus, adopted on 19 November 2025 with political agreement reported on 7 May 2026, moves it to 2 August 2028. Verify the current date before relying on it. MDCG 2025-6 (June 2025) remains the reference for the interaction between the two regulations.

### 8.6 What is binding and what is convention

| Item | Status | Consequence |
|---|---|---|
| IEC 62304 | Binding where required | Mask and procedure are configuration items |
| ISO 14971 | Binding where required | Rare-class loss is a hazard |
| FDA PCCP guidance | Non-binding; statute binding | Change must be pre-authorized or submitted |
| EU AI Act | Binding regulation, dates moving | Transparency and monitoring |
| NVIDIA 2:4 specification | Vendor documentation | Pattern support on specific hardware |
| Runtime sparse kernel list | Vendor documentation | Determines whether sparsity is exploited |
| "70% sparsity" | Convention | Must state pattern, format, and hardware |
| Compression ratio | Convention | Must state parameters, bytes, or latency |

### 8.7 The documentation a pruned model needs

| Record | Content |
|---|---|
| Criterion | Magnitude, movement, gradient, or other, with parameters |
| Schedule | One-shot, iterative, or gradual, with steps |
| Mask | Stored mask or seed plus procedure |
| Fine-tuning | Data, learning rate, epochs |
| Storage format | CSR, bitmap, N:M, with byte counts |
| Evaluation | Accuracy, calibration, per-class, subgroups |
| Target measurement | Latency and energy, with conditions |
| Change record | Classification and PCCP reference |

### 8.9 What the standards do not say

- No standard defines a sparsity level, a pattern, or a storage format.
- No standard requires a per-class evaluation, although ISO 14971's hazard analysis implies it for a diagnostic device.
- No standard says whether pruning is a substantial change; that depends on the effect and the approved plan.
- No standard defines "sparse hardware"; a vendor's support list is the operative document.

The absence means the team must define its own acceptance criteria and apply them consistently. A criterion invented after seeing the pruned results is not a criterion.

### 8.8 What "reproducible sparsity" requires

A sparsity pattern is reproducible when another engineer can regenerate the exact mask from the record. That requires four items.

- **The criterion and its inputs.** Magnitude needs only the trained weights; movement needs the gradients and the data that produced them; activation-aware criteria need the calibration set.
- **The schedule and its order.** Whether pruning was global or layer-wise, and the order in which layers were pruned, changes which weights survive.
- **The seed and the tie-breaking rule.** Many weights have identical magnitudes; the rule that breaks ties determines the mask.
- **The fine-tuning state.** If the mask was updated during fine-tuning, the update rule and the checkpoints are part of the pattern.

A record that says "70% magnitude pruning" is not a reproducible pattern. It is a summary of one.

## 9. Industry practice and case evidence

### 9.1 The foundational results

Han, Mao, and Dally's "Deep Compression" (arXiv:1510.00149, 2015) combined pruning, trained quantization, and Huffman coding to compress neural networks by a large factor. The paper's contribution was the pipeline: prune small weights, retrain, quantize the remaining weights with a codebook, then entropy-code the result. It established that pruning and quantization compose, and that the composition is more effective than either alone.

Frankle and Carbin's lottery ticket hypothesis (arXiv:1803.03635, ICLR 2019) showed that dense networks contain sparse subnetworks that, when trained from their original initialization, match the dense network's accuracy. Winning tickets were found at 10–20% of the original size on LeNet and small convolutional networks. The result is about trainability, not deployment: finding a ticket requires iterative train-prune-reset cycles, the sparsity is unstructured, and the benefit is accuracy at a given size rather than latency on dense hardware.

The practical lesson from both is that pruning is a family of procedures with different goals. Deep Compression targets storage. Lottery tickets target the question of whether sparse networks can be trained. Neither directly answers whether a pruned model will be faster on a given device.

### 9.2 Hardware support

NVIDIA's Ampere architecture introduced 2:4 structured sparsity with sparse tensor cores. The A100 whitepaper describes the pattern as at least two zeros in every four values along rows, with a compressed footprint and a sparse MMA that completes in half the cycles of the dense equivalent. The documentation is explicit that the benefit applies to supported operations and shapes, and that the pattern must be present in the weights.

Arm's Ethos-U85 documentation lists 2:4 structured sparsity as a feature that can double throughput, alongside INT8 weights with INT8 or INT16 activations, TOSA compliance, and weight-only lossless compression up to about 75%.

The common pattern is that hardware support is specific. A 2:4 pattern that helps on one accelerator is invisible on another, and a 70% unstructured pattern is invisible on both unless a sparse kernel exists.

### 9.3 What the evidence supports

| Claim | Status |
|---|---|
| Pruning reduces storage when a compact format is used | Established; depends on index overhead |
| Unstructured pruning does not reduce latency on dense hardware | Consequence of dense kernels; widely observed |
| Structured pruning reduces latency on dense hardware | Established; proportional to tensor size reduction |
| 2:4 sparsity can double throughput on supported tensor cores | Vendor documentation for specific layers |
| Lottery tickets exist at 10–20% of dense size on small vision models | Research finding, specific architectures |
| 90% sparsity is safe for any model | Not supported; model- and task-specific |
| Pruning preserves rare-class performance | Not supported; often the opposite |
| A pruned model is faster because it is smaller | Not supported without a kernel or hardware path |

### 9.4 The medical difference

A benchmark pruning result reports top-1 accuracy on a balanced test set. A medical model is evaluated on a population with a low prevalence of the condition of interest, and the rare class is the one that matters. Magnitude pruning removes small weights, and small weights are often the ones that encode the rare-class feature.

The practical consequence is that medical pruning needs a per-class evaluation that benchmark pruning does not. A model that loses 1 point of overall accuracy and 3 points of rare-class sensitivity is not equivalent to a model that loses 1 point uniformly, even though the aggregate is the same.

### 9.5 Why storage pruning is still valuable

It is easy to conclude from the latency result that pruning is not worth doing. That conclusion is wrong for devices where flash is the binding constraint.

A wearable with a 1 MiB flash budget may have 600 KiB free after firmware. A dense INT8 model at 500 KiB fits; a pruned model at 200 KiB leaves room for a second model, a larger buffer, or a firmware update. The benefit is product capability, not latency.

The mistake is to claim the latency benefit when the benefit is storage. The fix is to state which budget the pruning addresses and to measure that budget.

### 9.6 Three deployment patterns

**Structured pruning for latency.** The model is pruned by channel, fine-tuned, and quantized. The benefit is a proportional reduction in MACs and bytes, visible on any dense kernel. The cost is accuracy, especially for rare classes. This is the most reliable pattern for a microcontroller or CPU target without sparse hardware.

**2:4 for supported accelerators.** The model is trained with the pattern enforced, fine-tuned, and deployed on hardware with sparse tensor cores. The benefit is up to 2× math throughput, conditional on layer shape and runtime support. The cost is a constrained pattern and a dependency on one hardware family.

**Unstructured for storage.** The model is pruned aggressively, stored in a bitmap or compressed format, and deployed on hardware without a sparse kernel. The benefit is flash and update payload, not latency. The cost is accuracy and index overhead, and the benefit must be stated as storage.

The patterns are not interchangeable. A team that applies the third and expects the first will conclude that pruning does not work. A team that applies the first and expects the third's compression ratio will be disappointed by the parameter count.

### 9.7 How to read a pruning paper

A pruning paper is useful for method, not for prediction. To read one, extract five facts.

- **The architecture and dataset.** A result on ResNet-50 and ImageNet does not transfer to a one-dimensional ECG network.
- **The sparsity and the pattern.** Unstructured, structured, or N:M, and at what level.
- **The accuracy metric.** Top-1, top-5, or a clinical metric, and at what threshold.
- **The baseline.** Dense at the same accuracy, or dense at its own accuracy, and whether the comparison is at equal parameters or equal FLOPs.
- **The hardware and measurement.** Whether latency was measured or estimated, and on what device.

Papers that report only accuracy at a sparsity level are reporting a property of the model, not a deployment outcome. That is a legitimate contribution; it is not evidence that a pruned model will meet a deadline.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unstructured pruning on dense hardware | Storage falls, latency flat | Dense kernels multiply zeros | No speedup | Check sparse kernel support |
| Sparse format larger than dense | "Sparse" file bigger | Index overhead | Wasted effort | Bitmap or lower index width |
| Over-pruning | Accuracy cliff | Capacity removed | Missed floor | Stop before the cliff |
| Rare-class loss | Aggregate stable, class falls | Small weights carry rare signal | Clinical miss | Per-class evaluation |
| Structured prune too aggressive | Large accuracy drop | Whole channels removed | Missed floor | Lower target; block pattern |
| 2:4 without sparse path | Pattern present, latency flat | Runtime ignores pattern | No benefit | Check flags and shapes |
| Pattern lost in export | Exported graph dense | Converter expands weights | No benefit | Inspect exported graph |
| Mask not recorded | Results not reproducible | Missing pattern | Evidence invalid | Store mask or seed |
| Fine-tuning on pruning data | Validation overfits | Same data for selection and stopping | Optimistic result | Separate validation set |
| Quantization after pruning without recalibration | Accuracy below prediction | Scales mismatch | Extra loss | Recompute scales |
| Thermal profile changed | Sustained latency worse | Higher clock, more heat | Deadline miss | Soak test |
| Layer-wise sensitivity ignored | One layer over-pruned | Uniform target | Accuracy loss | Per-layer targets |
| Batch size changes the bottleneck | Batch-1 result does not transfer | Memory- vs compute-bound | No gain at production batch | Measure at the production batch |
| Layout defeats the format | Sparse model slower | Poor access pattern | Regression | Benchmark the actual format |
| Constant folding removes the mask | Exported file is dense-sized | Graph optimization | No benefit | Inspect the exported graph |
| Dynamic shapes disable the sparse kernel | Sparse path not selected | Compile-time specialization | No benefit | Fix shapes at export |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Magnitude correlates with importance | Simple criterion | Rare-class features in small weights | Per-class evaluation |
| Dense kernels ignore zeros | Predicts no latency gain | Sparse kernels exist | Check runtime and measure |
| Index width is 2 bytes | Storage estimate | 4-byte or variable-length indices | Inspect the format |
| Rows = 129 for pointer count | CSR estimate | Different tensor shape | Count rows from the model |
| Sparse hardware supports the layer shape | 2:4 benefit | Shape or size constraints | Check documentation and measure |
| Fine-tuning recovers most of the loss | Schedule design | Capacity limit | Fine-tune and evaluate |
| Sparsity pattern is stable after export | Runtime behavior | Converter expands weights | Inspect exported graph |
| Per-channel scales remain valid | Quantization accuracy | Distribution change after pruning | Recompute and compare |
| Activation ranges unchanged | Quantization accuracy | Pruning changes activations | Recalibrate |
| Thermal profile unchanged | Latency estimate | Higher clock, more heat | Soak test |
| The change is within the approved envelope | Regulatory path | Pattern change not anticipated | PCCP review |
| Batch at measurement matches production | Predicts latency | Different bottleneck | Measure both |
| The exported graph preserves the sparse operator | Predicts benefit | Constant folding or layout change | Inspect and compare sizes |
| The kernel supports the layer's shape | Predicts benefit | Minimum size or alignment | Microbenchmark the layer |
| Sparse format access pattern suits the kernel | Predicts benefit | Poor locality | Benchmark the format |
| Tie-breaking rule is deterministic | Reproducibility | Random or platform-dependent ties | Re-run and compare masks |

The central limit is that pruning's accuracy cost is not uniform. It falls on the classes and subgroups whose signal is carried by the weights that are removed, and those are often the ones with the least data. The aggregate metric is the least informative number in a pruning report.

A second limit is that the hardware benefit is not a property of the model. It is a property of the model, the runtime, the kernel, and the silicon together. A pruning result that does not name all four is not a deployment result.

## 12. Exercises

**Exercise 1 — Four levels (recall, 25 minutes).**
For each of unstructured pruning, structured pruning, and 2:4 sparsity, state what changes at the algorithmic, storage, kernel, and hardware levels.
*Deliverable:* a 3×4 table.
*Acceptance criteria:* each cell is specific; at least one cell is "no change" for each method.
*Time:* 25 minutes.

**Exercise 2 — Criteria (application, 35 minutes).**
Given a 4×4 weight matrix, rank the weights by magnitude and by movement (weight × gradient) and identify the differences. Explain which criterion you would use for post-training pruning.
*Deliverable:* two rankings and a short justification.
*Acceptance criteria:* rankings correct; the justification names the data requirement of each criterion.
*Time:* 35 minutes.

**Exercise 3 — Storage formats (calculation, 40 minutes).**
A tensor has 100,000 weights, 30,000 nonzeros, 256 rows, 1-byte values, 2-byte indices, and 4-byte pointers. Compute CSR, bitmap, and 2:4 storage where applicable. State which is smallest.
*Deliverable:* calculations with units.
*Acceptance criteria:* CSR = 30,000 + 60,000 + 1,028 = 91,028; bitmap = 30,000 + 12,500 = 42,500; 2:4 not applicable at 70% sparsity; conclusion correct.
*Time:* 40 minutes.

**Exercise 4 — Latency prediction (analysis, 30 minutes).**
A model has 90% unstructured sparsity. Predict the latency change on a Cortex-M4, on a GPU without sparse tensor cores, and on an A100 with 2:4 support. Explain each prediction.
*Deliverable:* three predictions with mechanisms.
*Acceptance criteria:* M4 and GPU unchanged; A100 benefit only if the pattern is 2:4, not 90% unstructured.
*Time:* 30 minutes.

**Exercise 5 — Schedule (design, 40 minutes).**
Design an iterative pruning schedule for a model with a 1.0 percentage point accuracy budget, starting from 0% and targeting 70%. State the increment, the fine-tuning length, and the stopping rule.
*Deliverable:* a schedule table.
*Acceptance criteria:* increments are 10–20%; stopping rule is pre-specified; evaluation at each step.
*Time:* 40 minutes.

**Exercise 6 — Subgroup evaluation (analysis, 45 minutes).**
Using the table in Section 6.7, compute the change in overall sensitivity and rare-class sensitivity for each route. Identify the route with the largest disparity between the two.
*Deliverable:* a table of deltas and a conclusion.
*Acceptance criteria:* deltas correct; the 90% route identified as the largest disparity; clinical implication stated.
*Time:* 45 minutes.

**Exercise 7 — Composition (design, 40 minutes).**
A model must fit a 64 KiB flash budget and a 40 ms latency budget. Propose a pruning and quantization pipeline, state the order, and predict the result using the numbers in Section 6.
*Deliverable:* an ordered pipeline and a prediction table.
*Acceptance criteria:* structure before precision; prediction consistent with the tables; a fallback named.
*Time:* 40 minutes.

**Exercise 8 — Sparsity record (design, 50 minutes).**
Write the record for the structured 40% route, including the criterion, schedule, mask, fine-tuning, storage, evaluation, and change classification.
*Deliverable:* a two-page record.
*Acceptance criteria:* reproducible from the record; per-class results included; change classified.
*Time:* 50 minutes.

## 13. Self-check answers

**A1. Four levels for the three methods.**

| Method | Algorithmic | Storage | Kernel | Hardware |
|---|---|---|---|---|
| Unstructured | Zeros in weights | Compact if encoded | No change on dense | No change |
| Structured | Channels removed | Smaller tensors | Dense kernel on smaller tensors | No special support |
| 2:4 | Patterned zeros | Compact with selector | Sparse kernel if available | Sparse tensor core |

**A2. CSR, bitmap, and 2:4 for Exercise 3.**
CSR = 30,000 × 1 + 30,000 × 2 + 257 × 4 = 30,000 + 60,000 + 1,028 = 91,028 bytes. Bitmap = 30,000 + 100,000 / 8 = 30,000 + 12,500 = 42,500 bytes. 2:4 requires exactly 50% sparsity, so it is not applicable to a 70% sparse tensor; forcing it would require keeping 50,000 weights. Bitmap is smallest.

**A3. Why CSR can exceed dense.**
Because each nonzero stores a 2-byte index and a 1-byte value, so at 50% sparsity the index alone equals the dense weight bytes and the total exceeds them. CSR becomes attractive only when the value is large relative to the index, or the sparsity is high enough that the saved values outweigh the indices.

**A4. Latency predictions for 90% unstructured.**
Cortex-M4: unchanged, because there is no sparse kernel and the dense loop multiplies zeros. GPU without sparse tensor cores: unchanged for the same reason, although memory traffic may fall if the sparse format is used and the kernel is bandwidth-bound. A100 with 2:4 support: no benefit, because 90% unstructured does not conform to 2:4; the sparse path requires the 2:4 pattern. A 2:4-pruned model on the A100 could benefit.

**A5. Why magnitude pruning hurts rare classes.**
The smallest weights are removed first, and a weight can be small in magnitude yet essential for a class with few examples. The gradient for that class is small because the class is rare, so the weight never grows large, and magnitude pruning removes it. The aggregate accuracy is dominated by common classes, so the loss is invisible there.

**A6. Deltas for Exercise 6.**
Dense 0.938 → unstructured 70% 0.927: -1.1 overall, rare class 0.884 → 0.861: -2.3. Structured: -0.8 overall, -1.4 rare. 2:4: -0.5 overall, -0.8 rare. 90%: -3.3 overall, -7.2 rare. The 90% route has the largest disparity: the rare-class loss is 2.2 times the overall loss.

**A7. Why structure precedes precision.**
Pruning changes which weights exist and their distribution; quantization computes scales from that distribution. If quantization comes first, the scales are computed for weights that pruning later removes, and the pruned model must be recalibrated. Pruning first also improves the weight distribution for quantization by removing the small weights that would round to zero.

**A8. Why a 2:4 model can be slower than dense.**
If the runtime does not select the sparse path, or if the layer shape is unsupported, the model executes the dense kernel on a tensor that may have additional metadata overhead. The pattern is present but not exploited, and the extra storage or conversion cost can make it slower.

**A11. Why does batch size change whether sparsity helps?**
At batch 1, a matrix-vector product is memory-bound, so reading fewer stored values can reduce time. At larger batches, the same matrix becomes compute-bound, and only a sparse kernel that skips multiplications reduces time. A pruning result measured at batch 1 can therefore fail to transfer to batch 16.

**A12. What makes a sparsity pattern reproducible?**
The criterion and its inputs, the schedule and the order of pruning, the seed and tie-breaking rule, and the fine-tuning state if the mask changed. A record without all four describes a result, not a procedure.

**A9. Rubric for Exercise 5 (schedule).**
A strong schedule prunes 10–20% of the target per cycle, fine-tunes briefly, evaluates on a held-out set, and stops when the accuracy budget is exhausted. It states the stopping rule before starting. A weak schedule prunes to 70% in one step and fine-tunes until the loss looks acceptable.

**A10. Rubric for Exercise 8 (record).**
A strong record includes the criterion, schedule, mask hash or seed, fine-tuning data, storage format and bytes, per-class and subgroup results, target measurements, and change classification. A weak record reports the sparsity level and the final accuracy only.

## 14. Interview questions

**Q1. Why doesn't pruning always speed up inference?**
A strong answer explains the four levels and dense kernels, and gives the condition under which latency falls (structured tensors, sparse kernels, or sparse hardware). A weak answer says pruning "reduces the model" and assumes speed follows.

**Q2. What is the difference between structured and unstructured pruning?**
A strong answer describes channel or block removal versus individual weights, the accuracy and hardware trade-offs, and the effect on tensor shapes. A weak answer says structured is "coarser".

**Q3. What is 2:4 sparsity?**
A strong answer defines two nonzeros in every four, names sparse tensor cores and supported shapes, and states that the pattern must be enforced before export. A weak answer calls it 50% pruning.

**Q4. Why can a sparse model be larger than a dense one?**
A strong answer explains index overhead and shows that CSR at moderate sparsity can exceed dense storage. A weak answer assumes sparsity always reduces size.

**Q5. How do you choose a pruning criterion?**
A strong answer compares magnitude, movement, gradient, and Hessian criteria on cost and data requirements, and chooses based on whether pruning happens during or after training. A weak answer always uses magnitude without justification.

**Q6. What is the lottery ticket hypothesis?**
A strong answer describes sparse subnetworks trainable from their original initialization, the iterative procedure, and why it is not a deployment recipe. A weak answer treats it as a pruning algorithm.

**Q7. How do you find the accuracy cliff?**
A strong answer prunes in increments, evaluates at each, and stops before the increment that exceeds the accuracy budget, using a held-out set. A weak answer picks a target sparsity in advance.

**Q8. Why do rare classes degrade first?**
A strong answer explains that rare-class weights receive small gradients and stay small, so magnitude pruning removes them, and the aggregate metric hides the loss. A weak answer says pruning is random.

**Q9. How does pruning interact with quantization?**
A strong answer says pruning changes the weight distribution, so scales must be recomputed, and pruning first can improve per-tensor SQNR by removing small weights. A weak answer treats them as independent.

**Q10. When is storage-only pruning the right choice?**
A strong answer says when flash is binding and latency is not, and notes that the benefit must be stated as storage, not speed. A weak answer dismisses pruning if latency does not improve.

**Q11. What would you record about a pruned model?**
A strong answer lists the criterion, schedule, mask or seed, fine-tuning data, storage format, evaluation, and change classification. A weak answer records the sparsity percentage.

**Q12. Does pruning reduce memory at runtime?**
A strong answer distinguishes file size from peak memory: weights may be read from flash and not fully resident, activations dominate SRAM, and a sparse format can add decode buffers. A weak answer equates file size with RAM use.

**Q13. Why is a 50% sparse model sometimes larger than the dense one?**
A strong answer explains index overhead and shows that at moderate sparsity the indices can cost more than the values saved, especially in CSR with 2-byte indices. A weak answer assumes sparsity always saves space.

**Q14. When would you reject a pruned model?**
A strong answer rejects on rare-class loss, subgroup harm, no target benefit, unreproducible mask, or an uncovered change. A weak answer rejects only on overall accuracy.

## 15. Key terms

- **2:4 sparsity** — two nonzeros in every group of four.
- **Bitmap** — one bit per weight indicating nonzero.
- **Block sparsity** — sparsity at block granularity.
- **Channel pruning** — removing whole output channels.
- **Cliff** — sparsity beyond which accuracy falls sharply.
- **COO** — coordinate list sparse format.
- **Coarse-grained sparsity** — sparsity at channel or block granularity.
- **Compression ratio** — original size divided by compressed size.
- **CSR** — compressed sparse row format.
- **Dense kernel** — kernel that multiplies all values.
- **Density** — fraction of nonzero weights.
- **Filter pruning** — removing whole convolution filters.
- **Fine-grained sparsity** — unstructured sparsity.
- **Gradual pruning** — increasing sparsity during training.
- **Gradient pruning** — removing weights with small gradients.
- **Hessian-based pruning** — removing weights with small second-order effect.
- **Importance score** — criterion used to rank weights.
- **Index overhead** — bytes used to describe positions.
- **Iterative pruning** — repeated prune-and-train cycles.
- **Layer-wise sparsity** — different sparsity per layer.
- **Lottery ticket hypothesis** — dense networks contain sparse trainable subnetworks.
- **Magnitude pruning** — removing weights with smallest absolute value.
- **Mask** — binary tensor indicating retained weights.
- **Movement pruning** — removing weights moving toward zero.
- **N:M sparsity** — at most N nonzeros in every M.
- **One-shot pruning** — pruning once, then fine-tuning.
- **Pruning** — removing weights or structures.
- **Pruning ratio** — fraction of weights removed.
- **Recovery** — accuracy regained by fine-tuning.
- **Rewinding** — restoring weights to an early checkpoint.
- **Sparse kernel** — kernel that skips zeros.
- **Sparse tensor core** — hardware unit for structured sparsity.
- **Sparsity** — fraction of zero weights.
- **Sparsity pattern** — arrangement of nonzeros.
- **Speedup** — reduction in latency.
- **Structured pruning** — removing channels, heads, or blocks.
- **Unstructured pruning** — removing individual weights.
- **Winning ticket** — a sparse subnetwork trainable to dense accuracy.
- **Block sparse** — sparsity at block granularity, supported by some kernels.
- **Density** — the fraction of weights that are nonzero.
- **Layer-wise sparsity** — a different target for each layer.
- **Per-layer target** — the sparsity assigned to one layer by sensitivity analysis.

## 16. Sources

All URLs accessed 25 October 2026.

**Foundational**
- Han, S., Mao, H., Dally, W., "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding", 2015. https://arxiv.org/abs/1510.00149
- Frankle, J., Carbin, M., "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", ICLR 2019. https://arxiv.org/abs/1803.03635
- Han, S., Pool, J., Tran, J., Dally, W., "Learning both Weights and Connections for Efficient Neural Networks", 2015. https://arxiv.org/abs/1506.02626

**Hardware and runtimes**
- NVIDIA, "NVIDIA A100 Tensor Core GPU Architecture" whitepaper. https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf
- NVIDIA, "Exploiting NVIDIA Ampere Structured Sparsity with cuSPARSELt". https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/
- NVIDIA, "Accelerating Inference with Sparsity Using the NVIDIA Ampere Architecture and NVIDIA TensorRT". https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/
- Arm, "Ethos-U85" product page. https://www.arm.com/products/silicon-ip-cpu/ethos/ethos-u85
- Arm, "CMSIS-NN". https://arm-software.github.io/CMSIS-NN/latest/

**Quantization interaction**
- Nagel, M., et al., "A White Paper on Neural Network Quantization", 2021. https://arxiv.org/abs/2106.08295

**Standards and regulation**
- IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019. https://www.iso.org/standard/72704.html
- FDA, PCCP final guidance, 3 December 2024. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- Regulation (EU) 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- MDCG 2025-6. https://health.ec.europa.eu/document/download/b78a17d7-e3cd-4943-851d-e02a2f22bbb4_en

> **Note:** The Section 6 numbers are constructed for teaching. Vendor sparsity claims apply to specific hardware, layer shapes, and patterns.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md`](../../course/14B_efficient_ai_distillation_quantization_on_device_inference.md)
- Previous topic: [`05_quantization.md`](05_quantization.md)
- Next topic: [`07_inference-optimization.md`](07_inference-optimization.md)
- Related topic: [`03_the-compression-family.md`](03_the-compression-family.md)
- Related topic: [`02_target-hardware-and-performance.md`](02_target-hardware-and-performance.md)
- Lab: [`../../labs/14B_quantization.ipynb`](../../labs/14B_quantization.ipynb)
- Lab directory: [`../../labs/`](../../labs/)
