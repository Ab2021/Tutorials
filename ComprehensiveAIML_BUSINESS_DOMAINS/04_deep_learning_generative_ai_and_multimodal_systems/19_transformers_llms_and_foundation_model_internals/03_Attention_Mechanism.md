---
lesson_id: "FT-M01-01"
lesson_title: "Attention Mechanism"
version: "1.0-draft"
created: "2026-11-07"
reviewed: null
status: "draft"
evidence_basis: "primary research paper and constructed tensor examples"
owner_role: "curriculum maintainer"
review_trigger: "Change to architecture assumptions, attention implementation, or supported sequence limits"
---

# 1.1 — Attention Mechanism

> **Course location:** Module 19, Transformers, LLMs and Foundation-Model Internals. This lesson maps to `finetuning_course/Module_01_Transformer_Architecture/1.1_Attention_Mechanism.md` in [`Train/Templates/directory.txt`](../../../Templates/directory.txt). It is authored from the visible source title and public references; the source lesson body was not inspected.
>
> **Status:** Draft. Numeric values are illustrative, independently recomputed here, and not benchmarks of a specific runtime.

## Learning outcomes

Learners can:

- Derive the scaled dot-product attention operation and explain why the key dimension appears in the scale.
- Track query, key, value, score, probability, and output tensor shapes across batch, sequence, head, and feature axes.
- Distinguish causal self-attention, bidirectional self-attention, and cross-attention by their permitted information flow.
- Calculate score count and a first-order arithmetic estimate for a given batch and sequence length.
- Explain softmax masking, numerical stability, and attention-weight interpretation limits.
- Diagnose shape, mask, positional, memory, and performance failures with discriminating tests.

## 1. Core intuition and notation

Attention is a content-dependent weighted combination. Each query compares against keys; normalized compatibility weights determine how values are combined. In self-attention, queries, keys, and values are derived from the same input sequence. In cross-attention, query vectors come from one stream and key/value vectors from a separate source stream.

For a batch of input hidden states `X`, learned linear projections produce query `Q`, key `K`, and value `V` tensors. For one head, attention first computes pairwise query-key scores, scales them, optionally adds a mask or bias, normalizes each query's scores across permitted keys, and uses the resulting weights to mix values.

The phrase “attention weight” describes a coefficient in this computation. It should not be treated automatically as a causal explanation, a human-interpretable rationale, or proof that a model used a particular source in a semantic sense. Weights are one internal signal; outputs and behavior also depend on value vectors, projections, residual paths, later layers, and model state.

### Symbols and shapes

Assume row-major batched tensors and one conventional multihead implementation:

| Symbol | Meaning | Shape / units | Convention |
|---|---|---|---|
| `B` | Batch size | positive integer | Number of independent sequences in the batch |
| `T_q` | Query sequence length | tokens | Number of query positions |
| `T_k` | Key/value sequence length | tokens | Number of keys available per query |
| `d_model` | Hidden width | scalar feature count | Input and concatenated output width |
| `H` | Attention-head count | positive integer | `d_model` is often divisible by `H` in standard designs |
| `d_k` | Per-head query/key width | scalar feature count | Often `d_model/H`; not required by all variants |
| `d_v` | Per-head value width | scalar feature count | May equal `d_k`; some architectures differ |
| `Q` | Query vectors | `[B,H,T_q,d_k]` | Usually projected from target/input states |
| `K` | Key vectors | `[B,H,T_k,d_k]` | Compared with queries by dot product |
| `V` | Value vectors | `[B,H,T_k,d_v]` | Weighted and aggregated |
| `S` | Unnormalized score matrix | `[B,H,T_q,T_k]` | One score for each query/key pair |
| `A` | Normalized attention weights | `[B,H,T_q,T_k]` | Each valid query row sums to one |
| `O` | Per-head output | `[B,H,T_q,d_v]` | Weighted value mixture |

The token axis and feature axis are different: increasing context length increases `T_q` and/or `T_k`; increasing head width changes `d_k`. Do not interchange attention score count with parameter count.

## 2. Scaled dot-product attention derivation

For one query vector `q` and key vector `k`, the dot product `q · k` grows in magnitude with the number of dimensions if component values have roughly stable variance. If the components are independent, mean-zero, with variance one, the dot product's variance is approximately `d_k`; its standard deviation is approximately `sqrt(d_k)`. Dividing by `sqrt(d_k)` keeps score scale from growing linearly with the square root of the feature count under these assumptions.

This is a scale-control motivation, not a theorem that every trained score has variance exactly one after scaling. Learned projections, normalization, correlations, finite precision, and training dynamics affect actual scores. The scale does not remove the need to inspect logits or apply stable softmax computation.

Validity: standard dense scaled dot-product attention with compatible query/key feature widths and a softmax over the key axis. Let `d_k > 0`.

Derivation: compute the compatibility matrix using matrix multiplication `Q K^T`; divide each score by `sqrt(d_k)`; add any permitted position bias or mask; apply softmax along the key dimension; multiply by `V`.

```text
FORMULA [ATTN-01]: Attention(Q,K,V) = softmax((Q K^T) / sqrt(d_k) + M) V
WHERE: Q [B,H,T_q,d_k]; K [B,H,T_k,d_k]; V [B,H,T_k,d_v];
       M [B,H,T_q,T_k] additive bias/mask; output [B,H,T_q,d_v].
CONSTRAINT: Q and K feature widths match; the softmax key axis includes at least one valid key per query; invalid entries are excluded by the mask convention.
NUMERICAL EXAMPLE: d_k=64, raw dot score=16; scaled score=16/sqrt(64)=2.
INTERPRETATION: scaling reduces score magnitude by 8 in this example; it does not assert any particular attention probability before normalization.
```

Independent check: `sqrt(64)=8`, and `16/8=2`. If all logits are equal across `n` valid keys, softmax gives `1/n` to each; if one key is masked, it must receive zero probability in the ideal masked computation. If every key is masked, softmax is undefined or implementation-dependent and should be treated as an invalid input condition rather than a harmless corner case.

### Softmax and numerical stability

For logits `z_j` over valid keys, the softmax probability is `exp(z_j)/sum_k exp(z_k)`. For finite logits, probabilities are nonnegative and sum to one. A stable implementation replaces `z_j` with `z_j - max_k z_k`; the shared subtraction leaves probabilities unchanged while reducing overflow risk.

An additive mask convention commonly uses zero for allowed positions and a sufficiently negative value for disallowed positions. Some implementations use boolean masks or specialized kernel semantics; mask polarity varies between APIs. Validate the precise convention instead of copying a mask from a different library. If a negative finite sentinel is not sufficiently negative relative to scores, forbidden positions can retain nonzero mass. If every position is assigned negative infinity, subtracting the maximum can produce `-inf - (-inf)`, a NaN; robust code must reject or explicitly handle all-masked rows.

The causal mask for query position `i` permits keys at positions `j <= i` (for zero-based positions in a standard decoder). A padding mask may additionally suppress keys that are padding. Combining causal and padding masks requires the mask composition to preserve both constraints. A single incorrect boolean inversion can expose future tokens or block the entire row.

## 3. Self-attention, cross-attention, and causal attention

### Self-attention

Self-attention draws `Q`, `K`, and `V` from one input sequence. It allows each position to aggregate information from other positions permitted by the mask. Bidirectional encoder attention can see both earlier and later positions; causal decoder attention restricts each position to its prefix. The architecture and objective determine which mask is appropriate.

### Cross-attention

Cross-attention uses queries from a target or decoder stream and keys and values from an encoded source. If source length is `T_k` and target length is `T_q`, scores have shape `[B,H,T_q,T_k]`. The source may be an encoder representation, memory bank, or another modality. Cross-attention is not automatically causal over the source axis; source visibility depends on the task and model design.

### Causal information-flow invariant

For an autoregressive output at position `i`, changing a future token `x_j` with `j>i` must not change the model's representation or logits at position `i` when the causal mask and all other inputs are held fixed. This is a useful invariant test. Training labels may be shifted relative to inputs, so test with the actual position mapping used by the model.

A causal mask prevents one direct path for future information through attention. It does not repair data leakage from precomputed features, incorrect sequence packing, non-causal operations, target misalignment, or evaluation contamination. Validate the full pipeline rather than just the mask tensor.

## 4. Multihead attention and projection shapes

Multihead attention applies several learned projections so heads can compute separate query/key/value subspaces. A common implementation first uses dense projections from hidden width `d_model` to `H*d_k` query and key channels and `H*d_v` value channels, reshapes the result into heads, computes attention independently per head, concatenates head outputs, and applies an output projection.

If `d_model=H*d_k=H*d_v`, Q, K, and V each have projection weights of size `d_model × d_model` (orientation depends on framework convention), plus an output projection of the same size. The four matrices therefore contribute `4*d_model^2` weights before biases. This count is for the stated conventional design only; grouped-query attention, multi-query attention, separate value width, fused implementation, or omitted projections alter it.

The concatenated head output has width `H*d_v`; the output projection maps it back to `d_model` if that is the architecture's residual width. A head split is a reshape/permutation, not a learned parameter by itself. Fusing QKV into one matrix can reduce kernel launches or simplify storage without changing the mathematical projection if the resulting matrix contains the same independent weights.

### Shape example

Use the Module 01 source-plan values: batch `B=2`, sequence length `T=128`, hidden width `d_model=512`, and `H=8`. Under equal head dimensions, `d_k=d_v=512/8=64`.

```text
X: [2,128,512]
Q,K,V after projection and head reshape: [2,8,128,64]
Q K^T scores: [2,8,128,128]
softmax weights: [2,8,128,128]
weighted values: [2,8,128,64]
concatenate heads: [2,128,512]
```

There are `2*8*128*128 = 262,144` score elements per forward pass. This is a tensor-element count, not a byte count: multiply by the actual stored element size for a payload estimate and add any temporary/workspace allocations separately.

Independent parameter count for four square projections is `4*512*512 = 1,048,576` weights, excluding biases. This is not the total transformer layer parameter count because the feed-forward network, normalization, embeddings, biases, and possible architecture-specific components are omitted.

## 5. Compute and memory scaling

For dense attention, score computation and value aggregation each perform a matrix multiplication. Their multiply-accumulate count per layer is approximately `B*H*T_q*T_k*d_k` for QK^T and `B*H*T_q*T_k*d_v` for AV. If one multiply-accumulate (MAC) is counted as two floating-point operations, multiply those counts by two. FLOP conventions differ; state the convention when comparing hardware or papers.

For equal self-attention lengths `T_q=T_k=T`, score storage grows quadratically in sequence length: `B*H*T^2` elements. The two matrix multiplications' arithmetic also grows approximately quadratically in `T` for fixed head dimensions. Projections and feed-forward layers often scale linearly with `T` for fixed width, though their constants and hardware efficiency may dominate at shorter lengths.

Using `B=2`, `H=8`, `T=128`, `d_k=d_v=64`:

- QK^T MACs: `2*8*128*128*64 = 16,777,216` MACs.
- AV MACs: the same, `16,777,216` MACs.
- Combined attention-core work: `33,554,432` MACs, or `67,108,864` FLOPs if two FLOPs/MAC are used.
- Score tensor elements: `262,144` before considering dropout, softmax intermediates, backward activations, or implementation fusion.

This is analytic work accounting, not latency prediction. Actual runtime depends on kernel selection, data types, memory bandwidth, cache, sequence length, occupancy, padding, batch composition, and hardware. A fused exact-attention kernel may avoid materializing the full score matrix in high-bandwidth memory, but it does not remove the mathematical pairwise interactions for dense attention. Different approximations or sparse patterns change the compute and semantic tradeoff.

### Sequence-length sensitivity

Doubling `T` from 128 to 256, while holding `B`, `H`, and head dimensions fixed, multiplies dense score elements and QK/AV MACs by four. The illustrative score count becomes `2*8*256^2 = 1,048,576` elements. This does not imply that measured runtime exactly quadruples: fixed overhead, memory hierarchy, kernel selection, padding, and parallelism can make observed scaling depart from the arithmetic estimate.

A 128-token score tensor with 262,144 elements occupies 0.5 MiB if stored as 2-byte values (`262,144*2 = 524,288 bytes`, divided by `1,048,576`). This is only one tensor payload. Real training can retain multiple activations or recompute them; inference may use fused kernels; allocator/workspace and other layers add memory. Never use this payload as the complete attention memory requirement.

## 6. Geometric interpretation and limits

A query vector points toward features it seeks; a key vector encodes features available at each position. Their scaled dot product is a learned compatibility score. Softmax turns each query's scores into a distribution across valid keys. The weighted sum of value vectors forms a context-dependent representation.

This geometric description helps reason about projections and similarity but can be misleading if treated literally. Vector directions live in learned spaces; basis rotations or parameter transformations can preserve behavior while changing individual coordinates. Attention weights alone omit value content and downstream transformations. A high weight on one key does not necessarily mean the output reproduces that token's meaning.

Attention is content-dependent but not free-form search over arbitrary external information. It operates only over key/value representations supplied to the layer. It does not itself provide information absent from the input or memory. A longer context expands the candidate positions but may increase computation and does not guarantee effective retrieval of every distant fact.

## 7. Design variants and selection

| Variant | Core change | Useful when | Costs or cautions |
|---|---|---|---|
| Full multihead attention | Multiple query/key/value subspaces | General sequence modeling with moderate lengths | Dense pairwise score work; head width and masks must align |
| Causal self-attention | Future-key entries masked | Autoregressive generation | Incorrect masks leak or block future/past information |
| Bidirectional self-attention | Both directions visible | Encoders with full input available | Not directly compatible with left-to-right generation without a different decoding strategy |
| Cross-attention | Query and key/value streams differ | Encoder-decoder or cross-modal fusion | Source-target length product determines score work |
| Multi-query attention | Multiple query heads share key/value heads | Reduce autoregressive KV-cache footprint | Quality and throughput depend on model/runtime; exact head mappings matter |
| Grouped-query attention | Groups of query heads share fewer KV heads | Tradeoff between multihead expressivity and cache size | Must track query-to-KV group assignment and cache layout |
| Local/window attention | Restrict each query to a neighborhood | Long sequences with local structure | Distant dependencies require another path or are unavailable directly |
| Sparse/structured attention | Selected query-key pairs only | Workloads with exploitable sparsity | Pattern design may miss needed interactions; implementation overhead matters |
| Exact fused attention | Compute exact dense attention with less intermediate materialization | Reduce memory traffic and intermediates | Same dense semantics; performance depends on shape, dtype, and kernel support |

Choose based on the actual sequence, latency, memory, quality, and compatibility requirements. Do not select a variant solely from a paper's reported speedup or a model label. Compare the same task and hardware with a correctness oracle and a controlled workload.

## 8. Failure modes and discriminating tests

### Failure A: head reshape silently permutes dimensions

**Symptom:** output shapes are valid, but parity with a small reference computation fails or quality collapses after a kernel change.

**Competing causes:** head axes swapped; sequence axis mixed with head; Q/K projection order mismatch; wrong contiguous layout; incorrect transpose before matrix multiplication.

**Discriminating check:** use tiny dimensions with coordinate-coded input values and compare each projected head and score against an explicit loop or trusted tensor operation. Assert shapes after every reshape and transpose.

**Bounded fix:** correct the reshape/permutation or projection split. Do not compensate by changing learning rate or adding a projection.

**Verification:** exact/close tensor parity on small inputs; then a fixed-seed end-to-end evaluation and performance comparison.

### Failure B: causal mask is inverted or shifted

**Symptom:** a decoder position changes when a future token changes, or later tokens cannot be attended to when they should be visible.

**Competing causes:** boolean-mask polarity mismatch; off-by-one position convention; target/input shift mismatch; incorrect combination of padding and causal masks.

**Discriminating check:** mutate one future token while holding earlier tokens fixed and compare logits for all earlier positions. Inspect the permitted-key matrix for short length, including first and last positions.

**Bounded fix:** correct mask semantics and position mapping; keep the original failing test as regression coverage.

**Verification:** causal-invariance test, padding test, and parity against a simple explicit mask implementation.

### Failure C: attention probabilities contain NaN or become uniform unexpectedly

**Symptom:** NaN outputs, zero gradients, or probability rows that are uniform despite meaningful score differences.

**Competing causes:** all keys masked; softmax over wrong axis; overflow/underflow; low-precision accumulation; mask sentinel insufficiently negative; scores accidentally zeroed.

**Discriminating check:** inspect pre-softmax scores and row-validity, apply max-subtracted softmax in a high-precision reference, and compare axis sums and finite-value checks.

**Bounded fix:** ensure at least one valid key per row; use the correct key axis and stable kernel; match mask contract and accumulation precision.

**Verification:** finite outputs and gradients across boundary rows, masked-padding parity, and stress tests with large-magnitude logits.

### Failure D: measured memory is far above score payload

**Symptom:** a simple `B*H*T^2*bytes` estimate is much smaller than peak device memory.

**Competing causes:** training backward activations; multiple layers; temporary logits/probabilities; retained graphs; workspace; allocator fragmentation; concurrent requests or KV cache.

**Discriminating check:** measure synchronized peak separately for forward, backward, optimizer, and generation phases; inspect tensor lifetimes and allocator snapshots.

**Bounded fix:** optimize the identified live allocation (for example fused kernel, activation checkpointing, or bounded batch) and rerun with unchanged quality conditions. Do not assume a smaller dtype solves a lifetime leak.

**Verification:** compare peak allocation and throughput against baseline on identical shapes; verify output and gradient parity within a declared tolerance.

### Failure E: output differs after cache optimization

**Symptom:** full-sequence and token-by-token decoding disagree.

**Competing causes:** cache position offset; rotary-position mismatch; stale K/V entries; cache page index; mask dimension or causal offset error.

**Discriminating check:** compare per-layer Q/K/V and logits at each step between cached and uncached paths for a short deterministic sequence.

**Bounded fix:** repair position, cache write/read, or mask mapping. Avoid raising tolerance to hide systematic drift.

**Verification:** exact or justified numeric tolerance under greedy decoding for short sequences, plus longer randomized tests and end-of-context boundary tests.

### Failure F: long-context quality degrades although the tensor fits

**Symptom:** a model accepts the intended length but misses information or changes answer quality.

**Competing causes:** training/inference positional convention mismatch; data truncation; retrieval dilution; lost-in-middle behavior; mask or cache error; positional extension unsupported by checkpoint.

**Discriminating check:** measure known-answer recall by position and distance on controlled prompts; inspect tokenized/truncated inputs and cached vs uncached logits.

**Bounded fix:** use a supported context strategy, retrieve a smaller relevant subset, or reduce accepted length until quality gates pass.

**Verification:** held-out position-sweep and task evaluation, not just successful allocation or absence of runtime error.

## 9. Cost, complexity, and resource accounting

For dense self-attention with sequence length `T`, score count is proportional to `B*H*T^2`; QK and AV arithmetic is proportional to `B*H*T^2*d`, where `d` is the head width (assuming equal key/value width). Projection work is proportional to `B*T*d_model^2` for fixed projection count. These expressions describe leading terms, not latency or complete memory.

Memory should be partitioned by lifecycle:

- **Parameters:** projection and output matrices, plus other model components.
- **Forward intermediates:** scores, probabilities, projected Q/K/V, outputs.
- **Backward state:** gradients and saved/recomputed activations during training.
- **Inference KV cache:** keys and values retained for prior tokens, often shared/reused according to architecture.
- **Workspace and allocator:** temporary buffers, kernel workspace, fragmentation and graph capture.
- **Concurrency:** multiple simultaneous sequences, prefills, and generation steps.

For conventional autoregressive inference, KV-cache payload is approximately `2 * layers * batch * T * H_kv * d_head * bytes_per_scalar`, where 2 accounts for keys and values. This estimate assumes dense cache for every token and layer with a common dtype; it excludes weights, metadata, page tables, temporary workspace, allocator overhead, and speculative tokens. It is not the dense attention-score tensor, which is a separate quantity.

Latency may be compute-bound, memory-bandwidth-bound, launch-bound, synchronization-bound, or scheduling-bound depending on prompt length, decode length, batch and hardware. Profile the same workload and report time-to-first-token separately from inter-token latency when serving. Larger context may affect prefill and cache retention differently.

## 10. Results and empirical boundary

The Transformer paper introduced an attention-only sequence-transduction architecture and reported results on translation and parsing tasks in its particular 2017 experimental setting. This lesson uses it as a primary historical reference for scaled dot-product attention and multihead attention, not as evidence that one contemporary implementation is universally superior.

FlashAttention work describes an IO-aware exact attention approach designed to reduce reads/writes between high-bandwidth memory and on-chip SRAM under stated assumptions. That paper motivates memory-traffic-aware kernels; it does not guarantee a speedup for every shape, device, datatype, or application. Reproduction requires matching software versions, hardware, workload, precision, and measurement method.

All specific batch/sequence/width examples in this lesson are constructed. No GPU kernel, model checkpoint, training run, memory profile, or latency benchmark has been executed here.

## 11. Interview and self-check questions

### Q1. Why divide query-key dot products by `sqrt(d_k)`?
**Answer:** Under an idealized independent, unit-variance component model, dot-product variance grows with `d_k`, so division by `sqrt(d_k)` controls score scale before softmax.
**Reasoning:** It is a scale argument under assumptions, not a promise of exact variance for trained vectors.
**Follow-up:** Does this replace stable softmax?
**Answer:** No. Subtracting the maximum logit is a separate numerical-stability technique.

### Q2. With `d_model=512` and 8 equal heads, what is `d_k`?
**Answer:** `512/8=64` features per head.
**Reasoning:** This assumes the conventional equal split and divisibility.
**Follow-up:** Is every architecture required to use this ratio?
**Answer:** No; head widths and query/KV grouping can differ.

### Q3. What is the score tensor shape for `B=2`, `H=8`, `T_q=T_k=128`?
**Answer:** `[2,8,128,128]`.
**Reasoning:** One score for each batch, head, query-position, and key-position combination.
**Follow-up:** How many score elements?
**Answer:** `2*8*128*128=262,144`.

### Q4. What does a causal mask guarantee?
**Answer:** Under correct data alignment and fully causal operations, a position cannot use tokens from later positions through attention.
**Reasoning:** The permitted keys satisfy `j<=i` for zero-based positions.
**Follow-up:** Is this sufficient to prevent all leakage?
**Answer:** No; precomputed features, packing, target shifts, or evaluation contamination can leak future information elsewhere.

### Q5. Why can all-masked rows produce NaNs?
**Answer:** A softmax over no valid entries is undefined; stable max-subtraction can compute negative infinity minus negative infinity.
**Reasoning:** Require at least one valid key or explicitly handle the invalid row.
**Follow-up:** Should one simply replace resulting NaNs with zero?
**Answer:** Not without defining intended semantics; silent replacement can hide a mask bug.

### Q6. If sequence length doubles, how do dense score elements change?
**Answer:** They quadruple when batch and head count remain fixed, because the key-query grid is quadratic in length.
**Reasoning:** `(2T)^2=4T^2`.
**Follow-up:** Does measured latency necessarily quadruple?
**Answer:** No; overhead, kernels, memory and parallelism change observed runtime.

### Q7. When do cross-attention scores have shape `[B,H,T_q,T_k]`?
**Answer:** When the query stream has `T_q` target positions and source key/value stream has `T_k` positions.
**Reasoning:** Each target query compares against each available source key.
**Follow-up:** Is the source axis necessarily causal?
**Answer:** No; visibility is task- and architecture-specific.

### Q8. Count four conventional square projections at width 512.
**Answer:** `4*512^2=1,048,576` weights, excluding biases.
**Reasoning:** Q, K, V, and output matrices each have `512*512` entries under the stated design.
**Follow-up:** Does this count the whole transformer layer?
**Answer:** No; MLP, norms, biases and other components are omitted.

### Q9. Explain why attention weights are not automatically explanations.
**Answer:** The output depends on value vectors and later transformations as well as weights; internal representations can also be reparameterized.
**Reasoning:** A coefficient is an internal computational quantity, not a causal or human-grounded explanation by itself.
**Follow-up:** What evidence is stronger for a model-behavior claim?
**Answer:** Controlled interventions and evaluations tied to the claimed behavior, with limitations stated.

### Q10. A mask change gives faster execution but breaks parity. What next?
**Answer:** Stop the change and compare the permitted-key matrix, mask polarity, axis and boundary positions against a small reference.
**Reasoning:** Performance is irrelevant if semantics changed unexpectedly.
**Follow-up:** What test catches future leakage?
**Answer:** Change a future token and assert earlier-position logits remain invariant.

### Q11. How can exact attention reduce memory without changing dense semantics?
**Answer:** A fused/IO-aware algorithm can tile computation and avoid materializing all score/probability intermediates in high-bandwidth memory.
**Reasoning:** It changes data movement and intermediate storage, not the dense mathematical attention relation.
**Follow-up:** Is it always faster?
**Answer:** No; shape, hardware, datatype, launch and framework support matter.

### Q12. Why are score payload estimates not full GPU memory estimates?
**Answer:** They omit projections, model parameters, multiple layers, backward state, KV caches, workspace, allocator effects, and concurrent work.
**Reasoning:** Peak memory is a lifecycle maximum across simultaneous allocations.
**Follow-up:** How should a real estimate be established?
**Answer:** Profile synchronized peaks across phases under a representative workload and record software/hardware versions.

## 12. Quick reference card

| Item | Rule |
|---|---|
| Core operation | `softmax(QK^T/sqrt(d_k) + mask) V` over the key axis. |
| Shape | Scores `[B,H,T_q,T_k]`; output `[B,H,T_q,d_v]`. |
| Scale | `sqrt(d_k)` controls idealized dot-product variance growth; not a guarantee about trained score distribution. |
| Causal rule | Position `i` can attend to `j<=i`; test future-token invariance. |
| Dense score count | `B*H*T_q*T_k`; equal self-attention length makes this quadratic in `T`. |
| MAC estimate | QK: `B*H*T_q*T_k*d_k`; AV: `B*H*T_q*T_k*d_v`. State whether 1 MAC counts as 1 or 2 FLOPs. |
| Mask guard | Correct polarity, valid key per row, correct axis and boundary offsets. |
| Failure oracle | Tiny explicit implementation with shape, mask and output comparisons. |
| Evidence | Constructed arithmetic; no runtime or GPU result measured. |
| Avoid | Treating weights as explanations or analytic counts as benchmarks. |

## Sources and boundaries

1. Vaswani et al., [“Attention Is All You Need”](https://arxiv.org/abs/1706.03762), 2017. Primary source for scaled dot-product attention, multihead attention, and the original Transformer architecture.
2. Dao et al., [“FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”](https://arxiv.org/abs/2205.14135), 2022. Primary paper for tiled, IO-aware exact attention; speed results are workload- and hardware-specific.
3. [Fine-tuning Module 01 source brief](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-transformers): source inventory, learning outcome, illustrative shape exercise, and validation plan.
4. [`Train/Templates/directory.txt`](../../../Templates/directory.txt), lines 860–864: visible Module 01 filenames; original lesson bodies are not available in this workspace.

Attention is one component of a transformer. The next submodule should connect attention to residual paths, normalization, feed-forward blocks, positional information, and training/inference contracts without treating any single implementation as universal.
