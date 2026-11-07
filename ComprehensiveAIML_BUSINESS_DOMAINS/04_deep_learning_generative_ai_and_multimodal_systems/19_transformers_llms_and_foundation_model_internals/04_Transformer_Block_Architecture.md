---
lesson_id: "FT-M01-02"
lesson_title: "Transformer Block Architecture"
version: "1.0-draft"
created: "2026-11-07"
reviewed: null
status: "draft"
evidence_basis: "primary research papers and constructed architecture calculations"
owner_role: "curriculum maintainer"
review_trigger: "Architecture convention, normalization order, positional mechanism, or implementation contract changes"
---

# 1.2 — Transformer Block Architecture

> **Course location:** Module 19, Transformers, LLMs and Foundation-Model Internals. This lesson maps to `finetuning_course/Module_01_Transformer_Architecture/1.2_Transformer_Block_Architecture.md` in [`Train/Templates/directory.txt`](../../../Templates/directory.txt). The source title is a scope cue; its lesson body was not available for inspection.
>
> **Status:** Draft; calculations are illustrative, not model measurements or implementation validation.

## Learning outcomes

Learners can trace the major computations in a transformer block, distinguish residual and normalization arrangements, derive tensor shapes and dense parameter counts for a conventional block, explain how positional information and masks interact with attention, and identify boundaries between a block-level diagram and an actual model implementation.

## Prerequisites

This lesson assumes the attention operation and its Q/K/V shapes from [1.1 — Attention Mechanism](03_Attention_Mechanism.md), matrix multiplication, vector normalization, and the concept of a residual connection. It does not prescribe one universal architecture: exact ordering, biases, activation, normalization, positional mechanism, and attention variant depend on the model.

## 1. What a block does

A transformer block transforms a sequence of hidden states while retaining a path for information to pass through the layer. A common decoder block contains:

1. A normalization or pre-normalization operation.
2. Self-attention over the allowed token positions.
3. A residual addition that combines the attention result with the block input.
4. A second normalization or other conditioning operation.
5. A position-wise feed-forward network (FFN), often with an expansion and contraction projection and a nonlinear activation.
6. A second residual addition.

The block is repeated in a stack. Each block's output becomes the next block's input. In a language model, a final normalization and vocabulary projection commonly convert final hidden states to token logits. Those head details may be shared or tied with embeddings and are not part of the block count unless explicitly included.

A block diagram is only a topology description. It does not specify tensor layout, parameter sharing, dropout, precision, attention cache, padding policy, checkpoint format, or runtime kernels. Those determine exact behavior and performance.

## 2. Hidden-state and sublayer shapes

Let `X` be a batch of token hidden states with shape `[B,T,d]`, where `B` is batch size, `T` sequence length, and `d=d_model` hidden width. A residual block normally returns `[B,T,d]` so the residual operands are shape-compatible.

Multihead attention may internally reshape projected values to `[B,H,T,d_k]`, compute attention, concatenate heads, and project back to `[B,T,d]`. The FFN is usually applied independently at each token position with shared weights:

```text
X [B,T,d]
  → first linear map [B,T,d_ff]
  → activation/gating [B,T,d_ff] (or related gated layout)
  → second linear map [B,T,d]
  → residual addition [B,T,d]
```

`d_ff` is the intermediate width. A conventional dense FFN uses two learned matrices. Gated FFNs may use an additional input projection and elementwise gate, increasing parameter count and compute. These architectures preserve sequence axis length in the standard case; convolutional or sequence-mixing variants may differ.

### Shape invariants

- Both residual branches must return the residual width and compatible batch/sequence axes.
- Attention query and key dimensions must match for dot products.
- The FFN output width must match `d_model` before its residual addition.
- Position IDs, causal mask, padding mask, and cached K/V positions must describe the same token ordering.
- Residual additions combine values with compatible dtype, device, and shape. Framework broadcasting can hide a missing dimension; assert intended shapes rather than relying on broadcast success.

## 3. Residual paths and normalization order

A residual connection adds a sublayer transformation to a stream that can carry the original signal. It supports information and gradient paths across depth, but does not guarantee stable optimization or make an incorrect branch harmless.

Two broad arrangements are commonly discussed:

### Pre-normalization

The sublayer receives normalized hidden states; its result is then added to the skip path. A simplified block is:

```text
Y = X + Attention(Norm_1(X))
Z = Y + FFN(Norm_2(Y))
```

This arrangement leaves an identity path around each sublayer. It is widely used in modern decoder stacks, but exact normalization type, scaling, and final normalization vary.

### Post-normalization

A simplified post-norm arrangement applies normalization after the residual addition:

```text
Y = Norm_1(X + Attention(X))
Z = Norm_2(Y + FFN(Y))
```

The original Transformer used residual connections followed by layer normalization in its described stack. Other systems use different conventions. The distinction changes gradient paths and may affect optimization stability; do not swap ordering in a checkpoint without retraining or explicit compatibility evidence.

### Normalization is not a cosmetic reorder

Layer normalization normalizes features within a token's feature vector according to learned scale and shift parameters. RMSNorm uses a different statistic and parameterization. Pre-norm and post-norm act on different tensors at different points. Replacing one with another, changing epsilon, or altering the reduced axis changes behavior even when input/output shapes remain identical.

A robust implementation test checks both forward outputs and gradients against a small, explicit reference. Shape equality alone cannot establish functional equivalence.

## 4. Feed-forward network and activation

For a conventional two-linear FFN with hidden width `d` and expansion width `d_ff`, the transformation is:

```text
FFN(x) = W_2 * phi(W_1 * x + b_1) + b_2
```

Here `W_1` maps `d → d_ff`, `W_2` maps `d_ff → d`, `phi` is an activation, and `b_1`, `b_2` are optional biases. Actual storage orientation varies by software convention. An activation such as ReLU, GELU, or a gated construction defines different functions. These are architecture choices, not interchangeable labels.

A gated FFN such as a SwiGLU-style variant generally applies a nonlinearity to one projection and multiplies it elementwise by a second projection before mapping back to `d`. Because it uses additional learned projection weights, parameter counts differ from a two-matrix FFN at the same intermediate width. The activation name alone does not determine the complete layer shape.

The position-wise description means the same FFN weights are applied at every sequence position independently; it does not mean the block as a whole ignores sequence context. Attention has already mixed contextual information into each token representation before the FFN operates.

## 5. Residual block equations and parameter accounting

For one conventional pre-norm dense decoder block with equal Q/K/V projection widths and a two-matrix FFN, ignoring dropout and optional biases:

```text
U = X + MHA(Norm_1(X), causal_mask)
Y = U + FFN(Norm_2(U))
```

The attention sublayer uses Q/K/V and output projections. With each matrix `d × d`, its projection count is `4d^2` weights. The FFN has matrices `d × d_ff` and `d_ff × d`, for `2d*d_ff` weights. Two layer norms add approximately `4d` scale/shift parameters when both use one scale and one bias vector of width `d`; norm variants may use fewer or different parameters.

Validity: dense conventional MHA, equal model and concatenated attention widths, two dense FFN projections, no parameter sharing, no biases counted in the primary total, and no gated extra projection.

Derivation: count each matrix by its row/column product; sum the four attention matrices and two FFN matrices. Biases, normalization, embeddings, output head, positional embeddings, and any shared weights are accounted separately.

```text
FORMULA [BLK-01]: P_dense = 4*d^2 + 2*d*d_ff
WHERE: P_dense = block matrix-weight count [parameters]; d = residual width [features];
       d_ff = FFN intermediate width [features].
CONSTRAINT: the conventional dense topology above; excludes biases, norms, embeddings, output head, and gated projections.
NUMERICAL EXAMPLE: d=512; d_ff=4*d=2048;
                   P_dense=4*512^2 + 2*512*2048
                          =1,048,576 + 2,097,152
                          =3,145,728 matrix weights.
INTERPRETATION: attention projections contribute one-third and the two-matrix FFN two-thirds of this illustrative block-matrix total.
```

Independent check: `512^2=262,144`; `4*262,144=1,048,576`; `512*2048=1,048,576`; twice that is `2,097,152`; sum is `3,145,728`. The ratio `2,097,152 / 3,145,728 = 2/3` follows from `d_ff=4d` under this exact topology.

A parameter count is not a memory estimate. It omits parameter precision, gradients, optimizer states, activations, temporary buffers, tied/shared parameters, sharding, and quantization metadata. Nor is it a runtime estimate: matrix dimensions, batch, sequence, kernel efficiency, and hardware determine throughput.

## 6. Positional information, masks, and residual flow

Self-attention compares content vectors across positions. A permutation-equivariant attention operation without positional information cannot by itself distinguish all sequence orders: if token content is permuted and masks follow that permutation, outputs permute correspondingly. Language tasks require order information, which can enter through learned or fixed positional embeddings, rotary transformations, relative position biases, or other mechanisms.

The position mechanism must agree with attention masks and cache indices. For a decoder, the model must not expose future-token information during training. For an encoder, bidirectional visibility may be intended. For cross-attention, source and target positions can use distinct conventions.

Residual streams carry representations through depth while attention and FFN sublayers modify them. A block is not simply a sequence of independent token classifiers: contextual dependencies arise through attention; residual additions preserve and combine streams; later blocks transform the resulting representations again.

When debugging a positional defect, inspect token IDs, position IDs, attention mask, rotary or relative-position parameters, and cache write/read indices together. Testing only the position embedding tensor misses the full contract.

## 7. Training and inference behavior

During training, the block may compute all token positions in parallel while applying a causal mask to prevent future leakage. During autoregressive inference, a model commonly processes a prompt and then generates tokens incrementally. K/V caching retains prior key/value states so they do not need to be recomputed for every next token, but this introduces cache shape, position, dtype, and lifetime constraints.

Training-time dropout, stochastic layers, and gradient calculation may differ from inference mode. The evaluation and serving paths must set mode correctly and use compatible masks. An inference optimization is valid only if it preserves intended outputs within an approved numerical tolerance and respects authorization, cache isolation, and request boundaries.

A cache belongs to a particular model revision, request/session, positional convention, and prefix. Reusing K/V across incompatible contexts or tenants can produce incorrect outputs and potentially expose information. Cache keys and storage isolation are system-level security properties, not just tensor implementation details.

## 8. Alternatives and selection

| Choice | Consider when | Tradeoff / boundary |
|---|---|---|
| Pre-norm vs post-norm | Select a known architecture or conduct a controlled research change | Different optimization behavior; not a drop-in serving switch |
| Two-layer FFN vs gated FFN | Match the architecture and budget under measured evaluation | Gated form changes parameter count, activation behavior and memory traffic |
| LayerNorm vs RMSNorm | Match checkpoint convention or evaluate a new model design | Reduction and affine parameters differ; checkpoint compatibility matters |
| Learned vs rotary/relative position | Respect pretrained positional convention or test supported extension | Different extrapolation, cache, and parameter properties; no universal best choice |
| Dense vs grouped-query attention | Reduce cache/serving cost if measured constraints justify it | Query-to-KV mapping differs; quality/performance must be evaluated |
| Dropout on/off | Match training regime and inference path | Inference dropout can make outputs stochastic; disabling during training changes regularization |
| Fused vs unfused kernels | Improve a profiled bottleneck | Fusion can change accumulation order and numerical error; validate output/gradient parity |

The safe decision is compatibility first for an existing checkpoint; experimental redesigns require controlled training and evaluation rather than ad hoc layer substitution.

## 9. Failure diagnostics

### Failure A: residual dimension mismatch

**Symptom:** runtime shape error, unintended broadcasting, or quality regression after a projection edit.

**Competing causes:** output projection width wrong; FFN contraction maps to `d_ff`; concatenation order wrong; hidden width differs across branch.

**Discriminating check:** assert both branch outputs exactly match `[B,T,d]`; run a coordinate-coded small tensor and inspect addition axes.

**Fix:** repair projection/reshape at the exact boundary; do not add an unexplained broadcast or reshape just to pass the runtime.

**Verification:** shape assertions plus numerical comparison to a small reference.

### Failure B: unstable deep-stack training

**Symptom:** loss spikes, exploding gradients, or a model that works at shallow depth but fails at intended depth.

**Competing causes:** norm ordering changed; initialization or learning rate mismatch; residual scale issue; mixed-precision overflow; corrupted batch.

**Discriminating check:** compare layerwise activation/gradient norms and replay the first failing batch; verify normalization axes and configuration against the intended architecture.

**Fix:** address confirmed cause; preserve the original architecture contract and run a controlled short reproduction.

**Verification:** finite activations/gradients, stable small-scale run, and held-out behavior before scaling up.

### Failure C: training and serving outputs disagree

**Symptom:** same prefix produces different logits or text across training/evaluation and deployed runtime.

**Competing causes:** train/eval mode; dropout; mask polarity; position IDs; cache mismatch; tokenizer/chat template; precision or fused-kernel rounding.

**Discriminating check:** compare layerwise hidden states with identical token IDs, position IDs, masks, model revision, and precision, first with cache disabled.

**Fix:** align contracts, then separately test caching and performance kernels.

**Verification:** deterministic parity test under declared tolerance and end-to-end regression suite.

### Failure D: parameter estimate disagrees with checkpoint size

**Symptom:** actual parameter count or file size exceeds expected count.

**Competing causes:** biases; embeddings or vocabulary head included; tied weights counted twice; gated FFN; extra experts; quantization scales; checkpoint optimizer state.

**Discriminating check:** enumerate named parameter tensors, shapes, dtypes, sharing, and serialized artifact contents.

**Fix:** revise scope and count distinct trainable/storage tensors separately; do not force equality with a simplified formula.

**Verification:** framework parameter enumeration reconciles with a hand-derived architecture breakdown.

### Failure E: position changes break generation

**Symptom:** cached decoding differs from full-prefix decoding or quality falls after context extension.

**Competing causes:** position index origin; RoPE scaling policy; cache offset; causal mask offset; context truncation.

**Discriminating check:** compare short uncached and cached per-layer outputs at each step; test positions at the trained boundary and target length.

**Fix:** restore checkpoint-compatible position/caching behavior; test extension as a separate adaptation, not a silent config edit.

**Verification:** token-by-token logits and task-level long-context position sweep.

## 10. Complexity and cost interpretation

For sequence length `T`, residual width `d`, and expansion `d_ff`, the dense projection/FFN arithmetic per token is proportional to `d^2 + d*d_ff`; attention pairwise work grows approximately with `T^2*d` per sequence for dense self-attention. Exact multipliers depend on FLOP/MAC convention, attention variant, head widths, and implementation.

During training, activation storage can scale with batch, layers, sequence length, and hidden width; attention intermediates can add sequence-quadratic storage in an unfused implementation. Activation recomputation trades additional arithmetic for reduced retained activation memory. During autoregressive inference, weight storage and KV-cache growth are distinct resource categories.

Use measured peak memory and throughput for the actual workload. A parameter count multiplied by bytes per weight is only a lower-level weight-payload estimate and excludes optimizer, activation, workspace, and allocator allocations. A block's analytical arithmetic cannot predict latency without hardware and kernel information.

## 11. Interview and self-check questions

### Q1. Why must the attention and FFN branches match residual width?
**Answer:** They are added elementwise to the residual stream, so batch, sequence and feature axes must be compatible and intended to match exactly.
**Reasoning:** Broadcasting can make an invalid design execute while producing wrong semantics.
**Follow-up:** What test catches that?
**Answer:** Exact shape assertions and small coordinate-coded forward comparisons.

### Q2. Count conventional attention projection weights for `d=512`.
**Answer:** Four matrices give `4*512^2=1,048,576` weights, excluding biases.
**Reasoning:** Q, K, V, and output projections each have `d × d` dimensions under the assumption.
**Follow-up:** Does grouped-query attention preserve this exact count?
**Answer:** No; K/V projection shapes may change.

### Q3. Count the 4x-expansion FFN weights at `d=512`.
**Answer:** `2*512*2048=2,097,152` matrix weights.
**Reasoning:** The expansion and contraction matrices have dimensions `d × 4d` and `4d × d`.
**Follow-up:** What if the FFN is gated?
**Answer:** Additional projection weights usually change the count; derive from the actual topology.

### Q4. Why does changing pre-norm to post-norm matter?
**Answer:** It changes which representations are normalized and the residual/gradient path through the block.
**Reasoning:** The operations are not algebraically interchangeable in general.
**Follow-up:** Can a deployed checkpoint just flip a config flag?
**Answer:** Not safely without compatibility evidence; it usually changes the function.

### Q5. Does a residual connection guarantee stable training?
**Answer:** No; it provides a skip path but optimization depends on scale, normalization, initialization, depth, data and optimizer.
**Reasoning:** Residual paths do not prevent exploding activations or gradients.
**Follow-up:** What diagnostic would you inspect?
**Answer:** Layerwise activation and gradient norms along with the first failing batch and config.

### Q6. Why include positional information?
**Answer:** Content-only attention has permutation-equivariant behavior and cannot in general encode sequence order by itself.
**Reasoning:** Position must be introduced through an embedding, rotation, bias, or other position mechanism.
**Follow-up:** What must remain consistent at inference?
**Answer:** Position IDs, mask offsets, positional transforms and cache indexes.

### Q7. What is the difference between a parameter count and a memory estimate?
**Answer:** Parameter count measures distinct model parameters; memory includes their storage plus gradients, optimizer state, activations, temporary buffers and allocator overhead.
**Reasoning:** Simultaneous allocations depend on training/inference phase and precision.
**Follow-up:** How do you get a deployable estimate?
**Answer:** Profile the exact model and workload and include concurrency and headroom.

### Q8. Why can training and inference paths disagree?
**Answer:** Different dropout mode, causal masks, position conventions, precision, cache path, tokenizer, or kernel arithmetic can alter outputs.
**Reasoning:** The same weights do not guarantee the same complete computation contract.
**Follow-up:** What is the diagnostic order?
**Answer:** Freeze token IDs/config, disable cache, compare hidden states layerwise, then add cache and optimized kernels one at a time.

### Q9. Is a block output a per-token independent representation?
**Answer:** No. Attention can incorporate information from other allowed positions before the FFN applies shared position-wise transforms.
**Reasoning:** FFN independence by position describes that sublayer alone, not the whole block.
**Follow-up:** What makes causal decoder output position-specific?
**Answer:** Its prefix context and causal mask, along with position handling and learned transformations.

### Q10. A model supports an input length in configuration. Does that establish quality at that length?
**Answer:** No. Acceptance by a tensor shape or runtime is not evidence of useful long-context behavior.
**Reasoning:** Positional adaptation, training support, cache correctness and retrieval quality need separate evaluation.
**Follow-up:** Name a focused test.
**Answer:** A held-out position sweep that moves known evidence through context positions and checks answer accuracy and latency.

## 12. Quick reference card

| Item | Rule |
|---|---|
| Residual stream | Maintain intended `[B,T,d_model]` shape across sublayer additions. |
| Conventional attention projections | `4*d_model^2` matrix weights when Q/K/V/output are square. |
| Two-matrix FFN | `2*d_model*d_ff` weights, excluding biases. |
| 4x expansion example | At `d=512`, attention = 1,048,576; FFN = 2,097,152; total = 3,145,728 matrix weights. |
| Norm order | Pre-norm and post-norm are distinct architectures, not interchangeable configuration details. |
| Position contract | Keep position IDs, masks, cache offsets, and positional mechanism aligned. |
| Parameter vs memory | Add gradients, optimizer, activations, workspace, KV cache and allocator only when applicable and measured. |
| Validation | Compare forward/gradient reference and train/eval/cache paths under explicit shapes and tolerances. |
| Evidence | Constructed calculations only; no model or runtime tested. |
| Avoid | Inferring latency, total memory, safety, or quality from block parameter count alone. |

## Sources and boundaries

- Vaswani et al., [“Attention Is All You Need”](https://arxiv.org/abs/1706.03762), 2017. Primary paper for the original Transformer encoder-decoder layer structure and residual/normalization convention described there.
- Xiong et al., [“On Layer Normalization in the Transformer Architecture”](https://arxiv.org/abs/2002.04745), 2020. Analysis of pre-LN/post-LN optimization behavior; results are scoped to paper assumptions and experiments.
- Shazeer, [“GLU Variants Improve Transformer”](https://arxiv.org/abs/2002.05202), 2020. Gated FFN variants and empirical context; not a universal architecture recommendation.
- [Module 01 source coverage brief](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-transformers), including the illustrative width/head/expansion exercise and validation plan.
- [`Train/Templates/directory.txt`](../../../Templates/directory.txt), lines 860–864: visible Module 01 filenames; source lesson bodies are unavailable.

This lesson describes a conventional block and selected alternatives. It does not prove which topology is used by a particular checkpoint. Inspect the model configuration and implementation, then validate them against versioned source and behavior.
