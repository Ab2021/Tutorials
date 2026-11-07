---
lesson_id: "FT-M01-03"
lesson_title: "Scaling and Parameter Counting"
version: "1.0-draft"
created: "2026-11-07"
reviewed: null
status: "draft"
evidence_basis: "primary scaling-law studies and constructed accounting examples"
owner_role: "curriculum maintainer"
review_trigger: "Change to architecture assumptions, empirical scaling evidence, or compute-accounting convention"
---

# 1.3 — Scaling and Parameter Counting

> **Course location:** Module 19, Transformers, LLMs and Foundation-Model Internals. This lesson maps to `finetuning_course/Module_01_Transformer_Architecture/1.3_Scaling_and_Parameter_Counting.md` in [`Train/Templates/directory.txt`](../../../Templates/directory.txt). The source title supplies scope only; its lesson body was not inspected.
>
> **Status:** Draft; calculations use constructed values and no training run or empirical scaling experiment was performed.

## Learning outcomes

Learners can calculate matrix parameters in a specified dense transformer, distinguish parameter count from active parameters and memory, estimate a simplified training-compute budget, explain why empirical scaling laws are conditional fits rather than universal prescriptions, and design small controlled scaling experiments with held-out evaluation.

## 1. Define “scale” before calculating

Scale is not one number. A model or training run may be described by:

- Total distinct learned parameters.
- Parameters activated for one token or one example.
- Training tokens and their mixture.
- Training compute and wall-clock duration.
- Hidden width, depth, head configuration, vocabulary and context length.
- Data quality, diversity, duplication, licensing and temporal freshness.
- Optimizer, schedule, precision, hardware and achieved utilization.
- Inference context, output length, batch, cache and serving concurrency.

Two models with the same parameter count can differ in architecture, data, active computation, memory footprint, quality, latency and license. A model with many total parameters can activate only a subset for each token; a dense model typically uses most of its learned matrices on every token. Report which scale measure supports the claim.

“Large language model” does not define a universal parameter threshold. A specific count can help estimate storage or matrix work after architecture and dtype are known; it does not establish task competence or deployment feasibility by itself.

## 2. Parameter-counting method

Count distinct parameter tensors from an architecture specification. For each tensor, multiply its dimensions; add the results while preventing double-counting shared/tied weights. Distinguish learned parameters from non-parameter buffers such as rotary frequencies, masks, running metadata, or quantization scales, which may still consume memory.

Maintain an auditable inventory with these columns:

| Component | Tensor shape | Distinct count | Shared? | Included in total? |
|---|---:|---:|---|---|
| Token embedding | vocabulary × hidden width | product | possibly tied to output head | yes/no explicitly |
| Attention projections | depends on MHA/GQA/MQA topology | sum of matrix products | often separate per layer | yes |
| Feed-forward | hidden/intermediate dimensions | sum of matrix products | generally per layer | yes |
| Normalization | scale and optional bias vectors | vector lengths | per layer or shared by design | yes |
| Positional embeddings | position count × hidden width if learned | product | architecture-specific | yes |
| Output head | hidden width × vocabulary | product | may share embedding matrix | yes/no explicitly |
| MoE experts/router | per expert plus router | topology-dependent | expert weights distinct | total and active separately |

Parameter count is a property of a declared model structure and counting convention. It is not the serialized file size: storage precision, quantization metadata, compression, tensor sharding, optimizer states, and serialization format affect bytes.

### Dense decoder approximation

Assume a decoder-only transformer with `L` layers, hidden width `d`, vocabulary size `V`, learned input embeddings, untied output projection, and a conventional block with four square attention projections and a two-matrix FFN of width `d_ff`. Exclude biases and normalization parameters initially.

Per layer:

```text
P_layer = 4*d^2 + 2*d*d_ff
```

Embedding plus output projection:

```text
P_embed_head = 2*V*d
```

Approximate total:

```text
P_total = L*(4*d^2 + 2*d*d_ff) + 2*V*d
```

Validity: dense conventional architecture, independent per-layer weights, no MoE, equal square attention projection width, two FFN matrices, untied input/output embeddings, and omitted biases/norms/other parameters.

Derivation: enumerate Q, K, V and output matrices, each with `d^2` elements. Add FFN expansion `d*d_ff` and contraction `d_ff*d`, multiply per-layer block count by `L`, then add two embedding/head matrices of `V*d` each.

```text
FORMULA [SCALE-01]: P_total ≈ L*(4*d^2 + 2*d*d_ff) + 2*V*d
WHERE: P_total [parameter count]; L [layers]; d [hidden features];
       d_ff [intermediate features]; V [vocabulary entries].
CONSTRAINT: untied embeddings, dense attention, two-matrix FFN; all omitted tensors are excluded.
NUMERICAL EXAMPLE: L=24, d=512, d_ff=2048, V=32,000.
                   P_layer=3,145,728.
                   P_layers=24*3,145,728=75,497,472.
                   P_embed_head=2*32,000*512=32,768,000.
                   P_total≈108,265,472 parameters.
INTERPRETATION: this is a structural estimate under the stated simplifications, not a checkpoint count or a quality prediction.
```

Independent arithmetic: per-layer block count is `4*262,144 + 2*1,048,576 = 1,048,576 + 2,097,152 = 3,145,728`; 24 layers contribute 75,497,472; embedding and untied head add 32,768,000; total is 108,265,472. If embeddings are tied, subtract one `V*d` matrix, yielding 91,881,472 before omitted terms.

A real checkpoint should be reconciled against enumerated tensor shapes and shared storage. Fused projection tensors do not necessarily change the number of learned scalars. Biases, gates, normalization, position embeddings, added experts, and vocabulary heads must be included according to the actual model.

## 3. Dense versus active parameter count

For a dense model, nearly all layer matrices participate in the forward computation for each token, apart from conditional features and implementation-specific optimizations. In a sparse mixture-of-experts (MoE) layer, all expert weights exist in the model, but a router selects a subset for each token. Therefore, distinguish:

- **Total parameters:** all distinct learned weights resident in or available to the model.
- **Active parameters per token:** weights used along the selected computation path for a token (definition depends on whether shared trunk, router, embeddings and output head are counted).
- **Stored parameters:** values actually represented in a checkpoint, possibly quantized or sharded.
- **Resident parameters:** values currently loaded into a device or memory tier.

Active parameter count does not determine total storage, communication, router overhead, load imbalance, or quality. It is not valid to compare dense and sparse models by active count alone when the product claim concerns memory or total compute.

## 4. Memory accounting is separate

At inference, raw weight payload under a fixed-width dtype is approximately `P * b`, with `P` parameter scalars and `b` bytes/scalar. This excludes quantization scales, metadata, embeddings when omitted from `P`, temporary activations, runtime workspace, KV cache, fragmentation, graph buffers, and concurrent requests.

At training, add gradients, optimizer states, activations, temporary communication buffers, and any master weights or quantization state required by the training method. The categories can have different dtypes and lifetimes. Peak allocation is the maximum of simultaneous allocations over time, not always the sum of all category maxima measured in separate phases.

Illustrative example: 108,265,472 parameters stored at 2 bytes each need `216,530,944` bytes for the raw parameter payload. In binary units, divide by `1,073,741,824` bytes/GiB to get approximately `0.2016 GiB`. This is not a full training or serving memory estimate.

If the checkpoint uses 4-bit weight codes, the nominal code payload is 0.5 byte per parameter; real storage also contains quantization scales, blocks, alignment and metadata. A 4-bit payload calculation does not imply that all model execution can remain in four-bit arithmetic or that total memory is 0.5 bytes per parameter.

## 5. Compute accounting and conventions

A matrix multiplication with dimensions `m × k` and `k × n` uses `m*n*k` multiply-accumulate operations under a common MAC count. Some FLOP conventions count one multiply and one add as two floating-point operations, so FLOPs are approximately `2*m*n*k`. Always state the convention.

For a dense transformer, a commonly used rough training estimate is about `6*P*N` FLOPs for `P` parameters and `N` training tokens, under assumptions that forward and backward passes are dominated by parameterized dense operations. This is a first-order estimate, not a universal identity. Attention's sequence-length-dependent work, vocabulary projection, checkpointing, padding, MoE routing, recomputation, optimizer updates, precision, and kernel behavior can change the total.

Validity: the rough estimate is for a dense autoregressive transformer training regime where parameter-associated multiply-accumulate work dominates and token count approximates useful training tokens.

Derivation sketch: forward dense matmul work is approximately proportional to two FLOPs per parameter per token. Backpropagation through the linear operations adds roughly comparable matrix work, giving an approximate factor near three times forward matmul work, or roughly six FLOPs per parameter per token under the two-FLOPs-per-MAC convention. Exact operation counts vary by layer and accounting convention.

```text
FORMULA [SCALE-02]: C_train ≈ 6*P*N
WHERE: C_train [FLOPs]; P [distinct dense model parameters under the estimate];
       N [training tokens counted under the chosen convention].
CONSTRAINT: rough dense-training estimate; not a measured hardware time and not exact for every architecture or sequence length.
NUMERICAL EXAMPLE: P=108,265,472; N=1,000,000,000 tokens;
                   C≈6*108,265,472*1,000,000,000
                    =649,592,832,000,000,000 FLOPs
                    ≈6.496e17 FLOPs.
INTERPRETATION: this is nominal arithmetic under the approximation; divide by achieved—not peak—throughput only for a rough compute-time estimate, then add overhead and validate.
```

If an illustrative sustained throughput were 100 TFLOP/s (`1e14 FLOP/s`), the idealized arithmetic time would be `6.496e17 / 1e14 = 6,495.9 seconds`, approximately 1.80 hours. This is an arithmetic quotient, not a training schedule prediction. It excludes data loading, communication, checkpointing, evaluation, failures, warmup, utilization variation, and any mismatch between quoted throughput and the actual precision/kernel mix.

A more accurate budget can be built by counting layer operations for the sequence lengths, batch composition, attention implementation, forward/backward method, and objective. Then compare the estimate to profiler evidence from a smaller representative run. Never translate peak FLOP/s directly to wall-clock without measured utilization and workload compatibility.

## 6. Empirical scaling laws: what the fit says

Scaling-law studies fit relationships between model/data/compute scale and measured loss or other outcomes across specific experiments. A fitted power law can help plan a budget sweep, but coefficients and exponents depend on data, architecture, objective, optimizer, tokenization, evaluation, and the explored range.

Kaplan et al. (2020) reported empirical power-law relationships across model size, dataset size, and compute in their study. Hoffmann et al. (2022) studied compute-optimal allocation and argued that under their experimental setup, many models were undertrained relative to their parameter count. These papers provide important empirical evidence, not a universal recipe for a different dataset, architecture, domain, or objective.

A scaling fit may not extrapolate reliably beyond the measured range. The effective budget may be limited by data quality, duplication, licensing, compute utilization, hardware availability, or downstream constraints. Loss scaling does not automatically predict task utility, safety, factuality, or business value.

A responsible use is to define a range of candidate model/data budgets, run controlled proxy experiments, estimate uncertainty, and validate the selected regime at a larger but still controlled scale. Preserve a fixed holdout and avoid changing multiple factors in a sweep such that causes cannot be separated.

## 7. Compute-optimal allocation as a decision, not a slogan

A training budget can be spent on more parameters, more tokens, higher-quality data, more epochs, better optimization, or experimentation. More parameters can require more memory and inference compute. More tokens can increase exposure to duplicates and low-quality material if data curation is poor. Higher-quality data may be scarce or expensive to license. The right allocation is the one that satisfies the target quality under full constraints, measured by a controlled comparison.

Compare candidates under:

- Equal compute budget when the decision is allocation efficiency.
- Equal token count when studying capacity effects.
- Equal parameter count when isolating data or recipe changes.
- Equal wall-clock or monetary budget when planning a real program, with hardware price and utilization stated.

Do not mix these comparators. A model trained with more compute can improve because of size, tokens, data quality, or optimization; the result cannot identify a scaling relationship without controlled conditions.

### Example of a controlled allocation sweep

Illustrative plan, not a measured experiment: hold tokenizer, data mixture, objective and evaluation fixed; test three model sizes with two token budgets per size; train each configuration with a recorded seed and compute count; preserve identical held-out evaluation; estimate validation loss and downstream task metrics; compare both equal-compute slices and equal-token slices. If resources permit only one seed, report that seed-level uncertainty is unknown and avoid claiming a robust ranking.

A proxy sweep can screen, not certify, large-scale behavior. Hardware efficiency and optimizer dynamics can change with scale. A final candidate still needs a representative run, data governance review, and general/regression evaluation.

## 8. Scaling failure diagnostics

| Symptom | Candidate explanations | Discriminating evidence | Bounded action |
|---|---|---|---|
| Larger model gives no expected loss gain | Undertraining; data bottleneck; optimization instability; bad token count | Matched token/compute runs, loss curves, gradient norms and data audit | Hold factors fixed; verify objective and budget before increasing size |
| Measured run time exceeds estimate | Low utilization; communication; input pipeline; checkpoint overhead; FLOP convention mismatch | Profiler phase breakdown and actual sustained throughput | Recompute with measured throughput and explicit wall-clock overhead |
| Parameter count disagrees with checkpoint | Tied weights; omitted biases/gates; extra heads; sharding or quantization | Enumerate tensors and unique storage, inspect config | Reconcile actual topology, not the simplified formula |
| Quality improves on validation but test stagnates | Repeated tuning on validation; split leakage; distribution mismatch | Audit tuning log, duplicate/group/time splits and task slices | Treat validation as consumed; create a new governed test set if justified |
| Sparse model appears cheaper by active count but costs more | Expert all-to-all communication; imbalance; weight residency; routing overhead | Measure total communication, memory, throughput and token-to-expert histogram | Compare end-to-end at matched workload and quality |
| Quality fit extrapolates poorly | Changed data/architecture regime; outside fitted range; noisy measurements | Re-run nearby scale points and estimate fit uncertainty | Restrict claim to observed range; expand controlled sweep |

A diagnostic should name what observation would weaken the leading hypothesis. Scaling narratives without a falsifiable check are not a resource plan.

## 9. Scaling experiment design

Define the objective and evaluation before a sweep. Record dataset and tokenizer versions, model architecture, initialization, optimizer, schedule, batch/token counting, precision, hardware, parallelism, seed, code, checkpoint policy, and exact FLOP convention. Keep the held-out test inaccessible during selection.

Change one factor at a time for causal interpretability when feasible. Factorial or designed experiments can estimate interactions, but need enough runs and a stated analysis plan. Training seeds, data order, nondeterministic kernels and distributed reductions can create variation; quantify it where the decision value warrants.

A sound scaling analysis presents raw points, uncertainty, fitted range, residuals, alternative fits, and a boundary on extrapolation. Report training loss and held-out metrics, not only a derived law. Include data quality and composition because more tokens are not interchangeable with more informative tokens.

## 10. Model size and system design

Parameter count informs several components but never determines deployment alone:

- **Weight storage:** parameter count times bytes per stored value, plus metadata and format overhead.
- **Training memory:** weights plus gradients, optimizer state, activations and transient buffers.
- **Inference memory:** weights plus KV cache and runtime/context/workspace allocations.
- **Compute:** per-token and training FLOPs depend on architecture and sequence.
- **Communication:** tensor/data/pipeline/expert parallelism changes network transfers and synchronization.
- **Latency:** batch, context, decoding length, kernel shape and serving scheduling matter.
- **Cost:** utilization, device price, power, availability, licensing and operational staff matter.

A model that fits in aggregate cluster memory may still fail because one layer, rank, expert, or communication buffer exceeds a device or topology constraint. Sharding changes placement, not logical parameter count. Quantization changes representation and possibly numerical behavior; verify quality and backend support.

## 11. Interview and self-check questions

### Q1. What does parameter count measure?
**Answer:** The number of distinct learned scalar parameters under a stated counting convention.
**Reasoning:** It does not by itself count memory, FLOPs, active sparse parameters, or performance.
**Follow-up:** What may make checkpoint bytes differ?
**Answer:** Dtype, quantization metadata, serialization, sharding and compression.

### Q2. Derive a dense block estimate for `d=512`, `d_ff=2048`.
**Answer:** `4d²+2d*d_ff=1,048,576+2,097,152=3,145,728` weights, excluding other parameters.
**Reasoning:** Four square attention matrices and two FFN matrices.
**Follow-up:** What is omitted?
**Answer:** Biases, norms, embeddings, output head, gated projections and architecture variants.

### Q3. Why distinguish total and active MoE parameters?
**Answer:** All experts contribute to stored capacity, while only selected experts may execute for a token.
**Reasoning:** Active count does not erase memory, routing, communication or load-balance costs.
**Follow-up:** Which measure is relevant for checkpoint storage?
**Answer:** Total stored parameters and their representation, not only active count.

### Q4. What is the rough `6PN` training-compute estimate based on?
**Answer:** A simplified dense matmul-dominated model where forward and backward work are each proportional to parameters times tokens under a two-FLOPs-per-MAC convention.
**Reasoning:** It is a rough accounting approximation and varies with architecture and sequence length.
**Follow-up:** What does it not give?
**Answer:** Measured wall time, full optimizer/data/communication overhead, or a universal exact FLOP count.

### Q5. One billion tokens at 50,000 tokens/s: active time?
**Answer:** `1e9/50,000=20,000 seconds`, or about 5.56 hours.
**Reasoning:** This assumes the quoted sustained scored-token rate and excludes overhead.
**Follow-up:** Why is that not a schedule estimate?
**Answer:** It omits evaluation, checkpoints, communication, stalls, failure and utilization variation.

### Q6. What does an empirical power law establish?
**Answer:** A fitted relationship over a particular experimental setup and observed scale range.
**Reasoning:** Coefficients and exponents depend on data, architecture, objective and protocol.
**Follow-up:** Can it establish downstream safety?
**Answer:** No; safety requires its own threat model and evaluations.

### Q7. How can a scaling sweep identify whether tokens or parameters are limiting?
**Answer:** Compare controlled budgets varying one factor or using a designed factorial plan while holding data/recipe/evaluation fixed.
**Reasoning:** Uncontrolled simultaneous changes confound attribution.
**Follow-up:** What if only one seed is affordable?
**Answer:** Report that variability is unestimated and avoid robust-ranking claims.

### Q8. What is wrong with comparing models at unequal compute and calling the bigger one more efficient?
**Answer:** Compute is a confound; the larger result may reflect more resources rather than better efficiency.
**Reasoning:** Choose a comparator matching the decision, such as equal compute for allocation efficiency.
**Follow-up:** When is equal-token comparison useful?
**Answer:** When isolating capacity or architecture effects at a fixed data exposure.

### Q9. A 4-bit model uses 0.5 bytes per parameter. Is that the full memory estimate?
**Answer:** No; scales, metadata, alignment, activations, KV cache, workspace and runtime overhead remain.
**Reasoning:** Nominal code payload is only one storage category.
**Follow-up:** Does 4-bit storage imply 4-bit compute?
**Answer:** No; the runtime may dequantize or use another accumulation path.

### Q10. Why must scaling-law extrapolation show residuals and range?
**Answer:** A fit can be noisy or regime-specific; visible fit error and observed domain bound its usefulness.
**Reasoning:** Extrapolating outside measured scales can fail under changed bottlenecks.
**Follow-up:** What is a safe next experiment?
**Answer:** Add controlled points near the proposed use range and preserve holdout data.

## 12. Quick reference card

| Quantity | First-order expression | Main caveat |
|---|---|---|
| Dense block weights | `4d² + 2d*d_ff` | Conventional MHA + two-matrix FFN; excludes biases/norms |
| Dense model estimate | `L*(4d²+2d*d_ff)+2Vd` | Untied embeddings, simplified topology |
| Raw weight bytes | `P * bytes_per_parameter` | Excludes all other live allocations and metadata |
| Dense training compute | approximately `6*P*N` FLOPs | Workload-dependent arithmetic estimate, not wall clock |
| Active MoE parameters | selected path for a token | Not equal to total storage/communication |
| Scaling law | fitted empirical relation | Bound claim to data, setup, fit range, and uncertainty |
| Compute time | estimated FLOPs / sustained FLOP/s | Add full system overhead; peak rate is not sustained rate |
| Decision rule | compare at the budget relevant to the decision | State equal-compute, equal-token, or equal-cost protocol |
| Evidence status | illustrative arithmetic and literature references | No model, run, or benchmark measured here |
| Avoid | treat parameter count as quality or deployment capacity | Profile and evaluate the actual system |

## Sources and boundaries

1. Kaplan et al., [“Scaling Laws for Neural Language Models”](https://arxiv.org/abs/2001.08361), 2020. Empirical fits across model/data/compute scales in the paper's setup; not a universal prescription.
2. Hoffmann et al., [“Training Compute-Optimal Large Language Models”](https://arxiv.org/abs/2203.15556), 2022. Compute-optimal allocation analysis for the reported experimental setup; extrapolation requires fresh validation.
3. Vaswani et al., [“Attention Is All You Need”](https://arxiv.org/abs/1706.03762), 2017. Transformer block architecture reference used for conventional matrix accounting.
4. [Module 01 focused source brief](../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-transformers), including the 4x FFN shape exercise and MoE comparison requirement.
5. [`Train/Templates/directory.txt`](../../../Templates/directory.txt), lines 860–864: visible filenames for Module 01; source lesson bodies unavailable.

All numerical examples are constructed. The formulas are architecture-specific approximations with their assumptions exposed; they are not a training-cost quote, model-size recommendation, quality forecast, or benchmark result.
