---
collection: "finetuning-alignment"
file_id: "01"
topic: "LoRA parameter-efficient LLM fine-tuning"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-07"
reviewed: null
review_trigger: "Base architecture, adapter runtime, data policy, or evaluation population changes"
notation_style: "plain-text"
prerequisites: ["Matrix multiplication and rank", "Transformer projections", "Gradient descent and supervised loss"]
pattern_tags: ["PEFT", "LoRA"]
source_status: "OPEN"
owner_role: "Model engineering"
source_index: "INDEX.md"
---
# LoRA Parameter-Efficient LLM Fine-Tuning
## 1. File Header
**Purpose:** Calculate adapter parameters, reason about memory, and decide whether an adapter is ready to deploy.
**Prerequisites:** Matrix shapes and multiplication, transformer projections, gradient descent, token-level loss, and basic evaluation.
See the [original LoRA paper](https://arxiv.org/abs/2106.09685) for the technique and its assumptions.
**Outcome:** Given model shapes, target modules, rank and precision, derive trainable counts and a bounded memory estimate; design a paired quality gate and rollback.
**Exclusions:** Full preference optimization, model merging algorithms, and a complete multitenant platform design are not taught here.
**Level:** Core methods with production considerations.
**Version/review:** 1.0 / not reviewed.
**Evidence:** DERIVED and ILLUSTRATIVE examples; no production results measured.
**Retrieval cue:** Start at Section 4 for counts, Section 8 for diagnosis, and Section 11 for interview practice.
**Source boundary:** The directory inventory identifies teaching topics, not accessible source lesson contents or benchmark evidence.
**Teaching home:** [Module 21](../INDEX.md); [source plan, PEFT](../../../00_SOURCE_COVERAGE_CUDA_TRAINING.md#ft-peft).
**Takeaway:** A small trainable adapter does not by itself guarantee a small training footprint or an acceptable model.
## 2. Concept Map / Mental Model
```text
Task and acceptance criteria
   -> evaluate prompting / retrieval baseline
   -> select frozen base + target projection matrices
   -> attach trainable low-rank factors
   -> optimize on authorized, split training examples
   -> compare frozen base and adapter on held-out groups
   -> check safety, quality, cost and serving compatibility
   -> gated rollout with versioned rollback
Failure at any gate -> investigate or use baseline, not automatic deployment
```
The base supplies the existing representation.
The adapter modifies selected linear transformations while the base weights remain fixed during adapter training.
Gradients still propagate through the frozen network to reach trainable factors.
The method changes parameter updates, not the need for correct data or evaluation.
**Takeaway:** Treat LoRA as a controlled update to a versioned base, not as a standalone model or isolation boundary.
## 3. Foundational Definitions
| Term | Meaning and boundary |
|---|---|
| Base weight W | Frozen linear map with d_out rows and d_in columns. |
| A | Trainable down-projection, r by d_in. |
| B | Trainable up-projection, d_out by r. |
| r | Positive integer adapter rank; maximum update rank, not necessarily achieved rank. |
| alpha | Dimensionless scaling hyperparameter; convention depends on implementation. |
| Target | A named linear module to which factors are attached. |
| Trainable count | Unique scalar parameters receiving optimization, including any explicitly unfrozen extras. |
| Activation | Intermediate tensor retained or recomputed for backpropagation. |
| Optimizer state | Persistent state maintained for trainable parameters, such as Adam moments. |
| Merge | Add the scaled update to a compatible base weight in an appropriate representation. |
| Held-out group | Independent evaluation unit such as a customer or document, not merely a random row. |
| Adapter artifact | Factors plus configuration and metadata; it requires a compatible base and tokenizer. |
| QLoRA | Training adapters with a quantized frozen base; memory and numerical behavior depend on implementation. |
| Rank versus width | Rank limits the update subspace; it is not the model hidden dimension. |
Symbols are scalar counts unless matrix shapes are stated.
Parameters have units of scalar elements; memory is bytes, MiB or GiB.
One MiB is 1,048,576 bytes; one GiB is 1,073,741,824 bytes.
**Takeaway:** Record shapes and trainability for actual modules, not just the advertised model parameter count.
## 4. Core Formulas with Full Derivation
### 4.1 Base: one projection
Validity: a dense linear map with a rank-r additive update, no biases or other trainable parameters.
Derivation: A has r*d_in entries and B has d_out*r entries; sum their independent storage counts.
The forward map adds the frozen map and the scaled composition of the two trainable maps.
```text
FORMULA [L1]: W_effective = W + (alpha/r) B A
WHERE: W [d_out,d_in], B [d_out,r], A [r,d_in]; alpha scalar; r positive.
CONSTRAINT: Standard alpha/r scaling convention; implementation alternatives exist.
NUMERICAL EXAMPLE: d_in=d_out=4096, r=8, alpha=16; scale=2.
INTERPRETATION: The update has compatible shape [4096,4096] and rank at most 8.
```
A zero initial B makes the initial update zero; a nonzero A allows B to receive gradients on the first step.
Initialization details are implementation-dependent and must be recorded.
The magnitude of the update depends on factors as well as scale; alpha alone does not determine it.
```text
FORMULA [L2]: P_adapter = r*(d_in+d_out)
WHERE: P_adapter is a count of scalar trainable entries in A and B.
CONSTRAINT: One independent pair; exclude biases, embeddings, heads, and extra trainable modules.
NUMERICAL EXAMPLE: 8*(4096+4096)=65,536 entries.
INTERPRETATION: Compare with 4096*4096=16,777,216 entries for a full matrix.
```
The ratio is 65,536/16,777,216=1/256, or 0.390625%.
Boundary: if rank is large enough, factor storage can exceed the full matrix storage.
### 4.2 General: several target modules
Validity: independently allocated factor pairs, with each targeted matrix counted once.
Derivation: sum L2 over each target, using that module's actual input and output widths and rank.
```text
FORMULA [L3]: P_total = sum_i r_i*(d_in_i+d_out_i) + P_extra
WHERE: i indexes distinct target matrices; P_extra counts unique additional trainable scalars.
CONSTRAINT: Do not double count tied/shared parameters; include biases or heads only if unfrozen.
NUMERICAL EXAMPLE: 32 layers, two square 4096 projections each, r=8, P_extra=0.
P_total = 32*2*8*(4096+4096) = 4,194,304 parameters.
INTERPRETATION: This count is for the named targets, not the entire model.
```
The corresponding targeted dense matrices contain 64*16,777,216=1,073,741,824 entries.
The same 1/256 ratio follows because these shapes and ranks match.
Grouped-query attention can make key/value output dimensions different; inspect actual tensor shapes.
### 4.3 Variants: memory and rank choices
Validity: this is an accounting identity for allocated tensors, not a universal runtime formula.
Derivation: sum storage of frozen weights, trainable factors, gradients, optimizer state, activations, temporary buffers and runtime overhead.
```text
FORMULA [L4]: M_peak = M_base + M_adapter + M_grad + M_opt + M_activation + M_temp + M_runtime
WHERE: each M is peak-resident bytes at the same time, adjusted for actual overlap and sharding.
CONSTRAINT: Components must refer to the same device and instant; allocator reservations can differ from tensor allocations.
NUMERICAL EXAMPLE: adapter=4,194,304 entries at 2 bytes each -> 8,388,608 bytes = 8 MiB.
INTERPRETATION: 8 MiB is adapter weights only, not total training or checkpoint memory.
```
Quantized base storage requires scales, metadata and potentially higher-precision compute buffers.
Mixed precision may retain master weights or use optimizer states of different precision.
Gradient checkpointing trades recomputation for activation memory; measure actual peak and throughput.
For QLoRA see [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314); do not assume every 4-bit implementation behaves identically.
### 4.4 Limits and counterexample
Validity: the rank statement follows from composition of finite matrices over real arithmetic.
The image of BA is contained in the image of B, whose dimension is at most r.
Therefore its rank cannot exceed r or either matrix dimension.
A rank-r update cannot express a target update with rank greater than r exactly.
For a 2 by 2 identity update, rank one is insufficient: its image is a line, not the entire plane.
If r=4096 for the square example, the factors contain twice as many entries as the original matrix.
Changing r while holding alpha fixed also changes alpha/r in this convention.
A rank experiment should distinguish rank effects from scale and training-budget effects.
**Takeaway:** Count every targeted matrix and every extra trainable tensor; low rank constrains the update, not the frozen base.
## 5. Worked Numerical Example
**Evidence:** ILLUSTRATIVE and DERIVED; this is not an observed run.
Assume a decoder with 32 blocks and independent query and value matrices in each block.
Each of these 64 matrices is 4096 by 4096, with no tied target parameters.
Choose r=8, alpha=16 and train no biases, heads or embeddings.
By L2 each pair has 8*8192=65,536 parameters.
By L3 64 pairs have 4,194,304 parameters.
Independent arithmetic: 65,536=2^16 and 64=2^6, so the sum is 2^22=4,194,304.
The targeted full matrices hold 64*4096^2=2^30=1,073,741,824 values.
The targeted adapter-to-full ratio is 2^22/2^30=1/256.
Suppose adapter weights and gradients each occupy 2 bytes per parameter.
Suppose two Adam moment tensors each occupy 4 bytes per parameter.
Under only those explicit assumptions, trainable state uses 12 bytes per parameter.
Weights use 8 MiB; gradients use 8 MiB; the two moments use 32 MiB.
Their sum is 48 MiB, excluding any master copy, communication, fragmentation or offload buffers.
An additional FP32 master copy would add 16 MiB to this example.
A hypothetical 7 billion parameter base at exactly two bytes per value uses 14 billion bytes.
That is approximately 13.04 GiB, excluding all metadata and other resident tensors.
Do not infer that this model fits into a 16 GiB accelerator: activation, workspace and allocator peaks remain unknown.
At r=16 with identical targets, trainable parameters and the assumed trainable state double to 8,388,608 and 96 MiB.
The base weight storage does not halve when rank is halved.
For an independent rectangular matrix 4096 input by 1024 output at r=8, L2 gives 40,960 entries.
A square-only shortcut would overestimate that rectangular adapter at 65,536.
**Decision:** Measure real peak memory on representative sequences and compare task quality before selecting rank.
**Takeaway:** The arithmetic is exact under the toy assumptions, whereas actual memory and quality remain unmeasured.
## 6. Visual / Geometric Intuition
```text
input x [d_in] -> A [r,d_in] -> r coordinates -> B [d_out,r] -> update
input x [d_in] -> W [d_out,d_in] ---------------------------> base output
                                      sum scaled update + base output
```
The columns of B span at most r output directions for the update.
A chooses coordinates in that restricted output subspace from the input.
The full output need not be low rank, because the frozen W may be full rank.
Two factorizations can represent the same update, so factor values alone are not unique explanations.
A small update rank is an inductive constraint, not proof that a downstream task is simple.
Different layers and projections may have different useful update subspaces.
**Takeaway:** Visualize LoRA as a narrow residual path beside a potentially full-rank base path.
## 7. Comparison / Selection Table
| Option | Choose when | Disqualifier or boundary | Fallback |
|---|---|---|---|
| Prompt only | Task is controllable using existing knowledge and context | Repeated behavior is unreliable or context is too costly | Test adapter with same evaluation set |
| Retrieval | Facts change or must be cited | Retrieval quality or authorization fails | Fix retrieval and access controls |
| LoRA | Reusable behavioral adaptation with limited trainable state | Quality, base compatibility or safety gates fail | Prompt/retrieval or full fine-tune |
| Full fine-tuning | Data and budget justify unrestricted updates | Memory or governance constraints prohibit it | Adapter or narrower scope |
| Quantized-base adapter | Base memory is the binding constraint | Quantization or runtime degrades required quality | Higher precision or different hardware |
| Merged adapter | One stable adapter per deployed artifact and merge is supported | Need rapid tenant switching or quantized merge harms quality | Serve base plus adapter |
| Unmerged adapter | Many compatible adapters need independent lifecycle | Runtime does not safely pin adapter to request | Isolated model instances |
Retrieval and fine-tuning can be combined; neither replaces authorization.
Select on quality, safety, measured end-to-end cost and lifecycle constraints, not trainable parameter count alone.
**Takeaway:** Choose adaptation only after testing simpler controls and stating disqualifying conditions.
## 8. Failure Modes & Edge Cases
All cases below are HYPOTHETICAL diagnostic exercises; no incident is asserted.
### No improvement
- Symptom: No improvement.
- Candidate cause: Incorrect target selection.
- Competing explanation: Bad data or insufficient optimization.
- Discriminating check: List trainable names and compare held-out loss.
- Bounded action: Select intended modules and fix labels.
- Verification: Verify gradients and paired task score.
- Escalation: retain or restore the last accepted artifact if the signal persists.
- Evidence boundary: this is a hypothetical mechanism, not a reported incident.
### Out of memory before backward
- Symptom: Out of memory before backward.
- Candidate cause: Base or temporary loading peak.
- Competing explanation: Activation peak during training.
- Discriminating check: Measure device allocation by phase.
- Bounded action: Use suitable base precision or placement.
- Verification: Repeat representative load and training step.
- Escalation: retain or restore the last accepted artifact if the signal persists.
- Evidence boundary: this is a hypothetical mechanism, not a reported incident.
### Out of memory during backward
- Symptom: Out of memory during backward.
- Candidate cause: Long-sequence activations.
- Competing explanation: Optimizer state or fragmentation.
- Discriminating check: Profile peak by sequence length and allocation.
- Bounded action: Reduce microbatch or checkpoint activations.
- Verification: Measure full-step peak and throughput.
- Escalation: retain or restore the last accepted artifact if the signal persists.
- Evidence boundary: this is a hypothetical mechanism, not a reported incident.
### Training loss falls but evaluation fails
- Symptom: Training loss falls but evaluation fails.
- Candidate cause: Leakage or overfit.
- Competing explanation: Wrong evaluation formatting.
- Discriminating check: Inspect group overlap and token templates.
- Bounded action: Re-split groups and align formatting.
- Verification: Rerun untouched holdout.
- Escalation: retain or restore the last accepted artifact if the signal persists.
- Evidence boundary: this is a hypothetical mechanism, not a reported incident.
### Adapter has no effect
- Symptom: Adapter has no effect.
- Candidate cause: Wrong base or adapter not loaded.
- Competing explanation: Evaluation uses stale cache.
- Discriminating check: Log immutable base and adapter hashes.
- Bounded action: Pin compatible artifact and invalidate cache.
- Verification: Compare deterministic logits against base.
- Escalation: retain or restore the last accepted artifact if the signal persists.
- Evidence boundary: this is a hypothetical mechanism, not a reported incident.
### Initial model differs from base
- Symptom: Initial model differs from base.
- Candidate cause: Nonzero adapter initialization.
- Competing explanation: Dropout or preprocessing mismatch.
- Discriminating check: Disable dropout and inspect initial update norm.
- Bounded action: Use documented initialization and matching inputs.
- Verification: Compare initial logits within precision tolerance.
- Escalation: retain or restore the last accepted artifact if the signal persists.
- Evidence boundary: this is a hypothetical mechanism, not a reported incident.
### Quality falls after merge
- Symptom: Quality falls after merge.
- Candidate cause: Quantization and rounding.
- Competing explanation: Wrong scale or merge target.
- Discriminating check: Compare unmerged and merged logits before requantization.
- Bounded action: Merge in suitable precision with correct metadata.
- Verification: Evaluate merged artifact on held-out tasks.
- Escalation: retain or restore the last accepted artifact if the signal persists.
- Evidence boundary: this is a hypothetical mechanism, not a reported incident.
### Wrong tenant behavior
- Symptom: Wrong tenant behavior.
- Candidate cause: Adapter routing race.
- Competing explanation: Incorrect tenant authorization.
- Discriminating check: Trace request-to-adapter immutable binding.
- Bounded action: Pin authorized adapter for complete generation.
- Verification: Run concurrent cross-tenant negative tests.
- Escalation: retain or restore the last accepted artifact if the signal persists.
- Evidence boundary: this is a hypothetical mechanism, not a reported incident.
### Checkpoint cannot resume
- Symptom: Checkpoint cannot resume.
- Candidate cause: Missing optimizer or scheduler state.
- Competing explanation: Changed data order or base revision.
- Discriminating check: Inspect checkpoint manifest and state counters.
- Bounded action: Restore complete compatible training state.
- Verification: Compare next-step loss and parameters.
- Escalation: retain or restore the last accepted artifact if the signal persists.
- Evidence boundary: this is a hypothetical mechanism, not a reported incident.
### Toxicity regression
- Symptom: Toxicity regression.
- Candidate cause: Training distribution alters safety behavior.
- Competing explanation: Evaluator or decoding changed.
- Discriminating check: Run fixed safety slices against both models.
- Bounded action: Revise data or reject release.
- Verification: Review blinded outputs and uncertainty.
- Escalation: retain or restore the last accepted artifact if the signal persists.
- Evidence boundary: this is a hypothetical mechanism, not a reported incident.
### Short examples improve but long ones fail
- Symptom: Short examples improve but long ones fail.
- Candidate cause: Sequence truncation.
- Competing explanation: Position or prompt template mismatch.
- Discriminating check: Inspect token lengths and actual masks.
- Bounded action: Correct truncation and evaluate long slices.
- Verification: Check task metrics by length.
- Escalation: retain or restore the last accepted artifact if the signal persists.
- Evidence boundary: this is a hypothetical mechanism, not a reported incident.
### Instability at first updates
- Symptom: Instability at first updates.
- Candidate cause: Excessive learning rate or scale.
- Competing explanation: Nonfinite data or precision overflow.
- Discriminating check: Inspect factor gradients and loss scaling.
- Bounded action: Adjust configuration after isolating cause.
- Verification: Run monitored training with finite checks.
- Escalation: retain or restore the last accepted artifact if the signal persists.
- Evidence boundary: this is a hypothetical mechanism, not a reported incident.
**Takeaway:** Diagnose the competing causes with phase-specific evidence before changing rank or releasing a mitigation.
## 9. Complexity & Cost Analysis
For one token, a dense base projection costs on the order of d_in*d_out multiply-adds.
The unmerged adapter adds work on the order of r*(d_in+d_out), excluding scaling and kernel overhead.
For a batch of N token vectors, both counts scale linearly with N under this dense operation model.
The 4096 square rank-8 example therefore adds 65,536 versus 16,777,216 multiply-adds per token for that projection.
This arithmetic ratio does not predict latency: kernel launches, memory traffic, batching and hardware utilization matter.
A merged weight can avoid a separate adapter path, but requires an extra merged artifact per variant.
If B and A are merged explicitly, constructing BA costs order d_out*r*d_in operations and produces a dense update.
Count temporary merge buffers as well as final model storage.
For unmerged serving, cache capacity must include factors, base models and runtime metadata.
Eviction policy and adapter loading latency depend on workload and are NOT MEASURED here.
Training costs include forward, backward through frozen operations, optimizer updates for trainable parameters, data loading and evaluation.
A frozen matrix does not require a weight gradient but may still require an input gradient.
Distributed runs can add synchronization, sharding and communication overhead.
Measure wall-clock time, accelerator-hours and held-out quality together.
A smaller optimizer state is not evidence of proportionally faster training.
**Takeaway:** Separate parameter arithmetic from actual kernel, memory, communication and lifecycle costs.
## 10. Key Theorems / Results Reference
**DERIVED:** Rank of a matrix product is no greater than the rank of either factor.
Applied to L1, rank(BA) is at most min(r,d_in,d_out).
This is an algebraic upper bound, not a guarantee that training reaches rank r.
**DERIVED:** Parameter count is linear in r for fixed target shapes by L2 and L3.
This conclusion does not imply linear changes in runtime, accuracy, or total memory.
**SOURCED:** The original LoRA paper introduces low-rank adaptation of frozen pretrained weights: [Hu et al., 2021](https://arxiv.org/abs/2106.09685).
Its reported experiments are workload-specific and are not reproduced or claimed here.
**SOURCED:** QLoRA describes training adapters through a quantized frozen base: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314).
This reference does not establish the memory or quality of a different runtime.
Source versions: arXiv landing pages; exact implementation versions and current behavior remain OPEN.
Review trigger: a change to the base checkpoint, adapter implementation or inference runtime.
The [support-domain adaptation case](../Case_Studies/01_support_domain_adaptation.md) applies adapter sizing to a proposed evaluation and deployment decision; its outcomes are not measured.
**Takeaway:** Algebraic bounds travel across implementations; benchmark outcomes do not.
## 11. Interview / Exam Questions
### 1. What is frozen?
**Answer:** The selected base weights remain fixed, while adapter factors are optimized. Other explicitly unfrozen parameters must be included in the trainable count.
**Reasoning:** Apply the stated assumptions and check what is and is not controlled.
**Boundary:** A numeric or operational claim outside those assumptions requires new evidence.
### 2. What is the shape of the update?
**Answer:** For W of shape d_out by d_in, A is r by d_in and B is d_out by r. BA has the same shape as W.
**Reasoning:** Apply the stated assumptions and check what is and is not controlled.
**Boundary:** A numeric or operational claim outside those assumptions requires new evidence.
### 3. Count a rank-8 adapter on a 4096 square projection.
**Answer:** 8*(4096+4096)=65,536 trainable scalar values, excluding bias and any extra trainable tensors.
**Reasoning:** Apply the stated assumptions and check what is and is not controlled.
**Boundary:** A numeric or operational claim outside those assumptions requires new evidence.
### 4. How many parameters for 32 blocks with query and value targeted?
**Answer:** There are 64 independent projections; 64*65,536=4,194,304. Tying or unequal widths changes the answer.
**Reasoning:** Apply the stated assumptions and check what is and is not controlled.
**Boundary:** A numeric or operational claim outside those assumptions requires new evidence.
### 5. Why is rank eight not a guarantee of an eight-dimensional final model?
**Answer:** Only the additive update has rank at most eight; adding it to an arbitrary full-rank base can yield a full-rank effective weight.
**Reasoning:** Apply the stated assumptions and check what is and is not controlled.
**Boundary:** A numeric or operational claim outside those assumptions requires new evidence.
### 6. Why can a small adapter still run out of memory?
**Answer:** Frozen base weights, activations, temporary workspaces, allocator overhead and distributed buffers remain resident; long sequences may dominate peak.
**Reasoning:** Apply the stated assumptions and check what is and is not controlled.
**Boundary:** A numeric or operational claim outside those assumptions requires new evidence.
### 7. How do you distinguish training data leakage from a format bug?
**Answer:** Inspect group overlap and compare actual tokenized prompts, targets and masks across splits. Correct each identified defect and re-evaluate on untouched groups.
**Reasoning:** Apply the stated assumptions and check what is and is not controlled.
**Boundary:** A numeric or operational claim outside those assumptions requires new evidence.
### 8. When should retrieval be preferred?
**Answer:** For changing or attributable facts, test authorized retrieval because weights are not a freshness mechanism. Retrieval itself needs evidence, access control and evaluation.
**Reasoning:** Apply the stated assumptions and check what is and is not controlled.
**Boundary:** A numeric or operational claim outside those assumptions requires new evidence.
### 9. What changes when rank doubles?
**Answer:** For fixed target shapes, factor count doubles. Under alpha/r scaling, holding alpha fixed halves the explicit scale; total memory and quality need not double.
**Reasoning:** Apply the stated assumptions and check what is and is not controlled.
**Boundary:** A numeric or operational claim outside those assumptions requires new evidence.
### 10. How do you validate merging?
**Answer:** Pin base and adapter versions; compare merged and unmerged logits in suitable precision, then run the same held-out quality and safety suite on the deployed artifact.
**Reasoning:** Apply the stated assumptions and check what is and is not controlled.
**Boundary:** A numeric or operational claim outside those assumptions requires new evidence.
### 11. How would you test tenant isolation?
**Answer:** Bind authorized adapter identity for the whole request, test concurrent cross-tenant requests and cache keys, and verify outputs and access logs; factors alone are not security boundaries.
**Reasoning:** Apply the stated assumptions and check what is and is not controlled.
**Boundary:** A numeric or operational claim outside those assumptions requires new evidence.
### 12. What does a paired evaluation require?
**Answer:** Run base and adapted models on the same independent held-out groups with identical prompts and decoding, define metric denominators and uncertainty, and inspect safety and subgroup regressions.
**Reasoning:** Apply the stated assumptions and check what is and is not controlled.
**Boundary:** A numeric or operational claim outside those assumptions requires new evidence.
**Takeaway:** Strong answers distinguish exact algebra, experiment design, and production evidence.
### Production decision drills
Each drill is an ILLUSTRATIVE design exercise; proposed checks are not observed results.
#### Target inventory
- Decision input: projection names.
- Failure under consideration: missing or duplicated module.
- Evidence to collect: unique parameter identities.
- First controlled action: enumerate shapes before attaching factors.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Rank sweep
- Decision input: rank and scaling.
- Failure under consideration: confounded scale change.
- Evidence to collect: fixed task set and training budget.
- First controlled action: record both alpha and effective scale.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Learning rate
- Decision input: optimizer step size.
- Failure under consideration: unstable or stalled convergence.
- Evidence to collect: loss and gradient norms by step.
- First controlled action: compare schedules under equal tokens.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Token masking
- Decision input: supervised target tokens.
- Failure under consideration: learning from prompt or padding.
- Evidence to collect: decoded examples and loss mask.
- First controlled action: validate boundaries with hand-labeled samples.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Data grouping
- Decision input: customer and document identities.
- Failure under consideration: correlated leakage across splits.
- Evidence to collect: group intersection counts.
- First controlled action: split before augmentation.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Temporal evaluation
- Decision input: event and cutoff timestamps.
- Failure under consideration: future information in training.
- Evidence to collect: latest training timestamp by group.
- First controlled action: reserve a later test interval.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Deduplication
- Decision input: content hashes and near duplicates.
- Failure under consideration: test contamination.
- Evidence to collect: cross-split duplicate audit.
- First controlled action: deduplicate with a documented policy.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Rare slices
- Decision input: slice membership.
- Failure under consideration: aggregate hides regression.
- Evidence to collect: per-slice numerator and denominator.
- First controlled action: predefine minimum sample and escalation.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Safety regression
- Decision input: harm categories.
- Failure under consideration: task optimization degrades refusal.
- Evidence to collect: paired blinded review.
- First controlled action: block release on material harm.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Factuality
- Decision input: source-supported claims.
- Failure under consideration: confident unsupported answer.
- Evidence to collect: claim-level evidence review.
- First controlled action: retain retrieval and citations.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Output schema
- Decision input: valid structured responses.
- Failure under consideration: syntactically valid but wrong values.
- Evidence to collect: parse and semantic validation separately.
- First controlled action: check downstream contract.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Quantized base
- Decision input: weight representation.
- Failure under consideration: accuracy or kernel mismatch.
- Evidence to collect: matched precision comparisons.
- First controlled action: record quantization configuration.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Merge precision
- Decision input: merge dtype.
- Failure under consideration: rounding makes merged model differ.
- Evidence to collect: logit comparison before requantization.
- First controlled action: test exact deployment artifact.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Adapter cache
- Decision input: base and adapter versions.
- Failure under consideration: stale adapter reused.
- Evidence to collect: request trace and cache identity.
- First controlled action: key on immutable compatible versions.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Tenant routing
- Decision input: authorization identity.
- Failure under consideration: wrong adapter applied.
- Evidence to collect: concurrent negative tests.
- First controlled action: bind identity to generation session.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Rollback
- Decision input: previous accepted artifact.
- Failure under consideration: missing dependent base.
- Evidence to collect: restore drill and manifest.
- First controlled action: keep compatible base and tokenizer.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Training resume
- Decision input: optimizer and RNG state.
- Failure under consideration: non-equivalent restart.
- Evidence to collect: compare step counter and sample order.
- First controlled action: save full training state.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Context length
- Decision input: tokenized sequence length.
- Failure under consideration: truncation changes labels.
- Evidence to collect: length histogram and actual masks.
- First controlled action: evaluate representative long examples.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Optimizer memory
- Decision input: moments and master weights.
- Failure under consideration: state exceeds assumptions.
- Evidence to collect: dtype inventory at step.
- First controlled action: sum allocated state per device.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Activation memory
- Decision input: microbatch and sequence lengths.
- Failure under consideration: peak occurs in backward.
- Evidence to collect: phase-level peak allocation.
- First controlled action: measure after warmup.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Evaluation uncertainty
- Decision input: independent group count.
- Failure under consideration: false confidence from correlated rows.
- Evidence to collect: group bootstrap or appropriate intervals.
- First controlled action: report interval and sample unit.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Production monitoring
- Decision input: quality proxy and latency.
- Failure under consideration: proxy hides harmful failure.
- Evidence to collect: audited outcome sample.
- First controlled action: define ownership and alert response.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Distribution shift
- Decision input: input and task mix.
- Failure under consideration: offline result does not transfer.
- Evidence to collect: time and segment comparisons.
- First controlled action: collect consented representative labels.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Serving throughput
- Decision input: batch mix and concurrency.
- Failure under consideration: adapter path launch overhead.
- Evidence to collect: end-to-end latency distribution.
- First controlled action: benchmark real adapter mix.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### License and privacy
- Decision input: training data permission.
- Failure under consideration: unauthorized reuse.
- Evidence to collect: dataset lineage and retention audit.
- First controlled action: exclude unapproved records.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Reproducibility
- Decision input: base code and data revisions.
- Failure under consideration: artifact cannot be reconstructed.
- Evidence to collect: immutable manifest completeness.
- First controlled action: record seeds and environment.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Evaluation budget
- Decision input: number of candidate adapters.
- Failure under consideration: overfitting to validation.
- Evidence to collect: predeclared final untouched test.
- First controlled action: limit repeated test access.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Preference leakage
- Decision input: human reviewer identity.
- Failure under consideration: correlated subjective judgments.
- Evidence to collect: reviewer and group stratification.
- First controlled action: use blinded adjudication.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Cross-domain retention
- Decision input: general capability slices.
- Failure under consideration: specialization causes forgetting.
- Evidence to collect: paired general-task comparison.
- First controlled action: set regression guardrails.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
#### Failure fallback
- Decision input: service health and model quality.
- Failure under consideration: bad adapter stays active.
- Evidence to collect: reversion trigger and exercise.
- First controlled action: route to accepted baseline.
- Establish an explicit owner before changing the candidate.
- Record the exact base and adapter revision for the comparison.
- Keep evaluation groups separate from the training inputs.
- Preserve the same prompt and decoding settings for paired comparisons.
- Record the denominator and missing cases for every reported rate.
- Inspect whether one subgroup carries most of the observed change.
- Do not interpret an absent measurement as a passing result.
- If the check fails, retain the prior accepted configuration.
- Re-run the targeted check after the bounded correction.
- Re-run relevant safety and quality guardrails before rollout.
- Record uncertainty and the conditions under which the conclusion transfers.
- An example plan is not evidence that the check was executed.
### Additional worked decisions
These examples are ILLUSTRATIVE and use the assumptions explicitly stated.
#### Unequal projection shapes
Suppose a block has a 4096-by-4096 query projection.
Suppose its value projection maps 4096 inputs to 1024 outputs.
Use rank 8 independently for both projections.
The query adapter uses 65,536 entries by L2.
The value adapter uses 8*(4096+1024)=40,960 entries.
The two adapters together contain 106,496 entries per block.
Across 32 identical blocks they contain 3,407,872 entries.
Independent check: 106,496*2^5=3,407,872.
This excludes trainable biases and any other targets.
Counting both as square would overstate the count.
#### Mixed ranks
Keep the same unequal shapes but choose query rank 4.
Choose value rank 16.
Query then requires 4*8192=32,768 entries.
Value requires 16*5120=81,920 entries.
Their combined count is 114,688 entries per block.
Across 32 blocks the count is 3,670,016.
Independent check: 114,688*2^5=3,670,016.
Higher value rank costs less per rank unit because its output width is smaller.
The count alone cannot prove the mixed-rank allocation has better quality.
Compare candidates using the same data, objective and evaluation.
#### Additional classifier head
Suppose the 4,194,304-entry adapter also trains a classifier.
The classifier maps 4096 features to two outputs.
With a separate bias, its weight has 8192 entries and its bias has two.
The added head contains 8194 entries.
Total trainable parameters become 4,202,498.
Independent check: 4,194,304+8192+2=4,202,498.
The head must be saved and restored with the adapter.
The tokenizer and input formatting remain separate dependencies.
If the head shares a weight, count the unique storage rather than both names.
#### Adapter storage for multiple tasks
Assume ten distinct adapters each have 4,194,304 parameters.
Assume each stored factor uses exactly two bytes.
Each adapter then contains 8,388,608 bytes, or 8 MiB.
Ten adapters require 80 MiB of factor payload.
This sum excludes metadata, padding and serialized checkpoint overhead.
It also excludes a base model and any optimizer state.
If all ten are loaded simultaneously, device cache may need at least that payload.
If only one is loaded at a time, loading and eviction add latency.
Neither option establishes a throughput or latency result without measurement.
The cached artifact identity must include the compatible base revision.
#### Matched paired task evaluation
Assume 100 independent held-out requests, each scored correct or incorrect.
The baseline is correct on 70 requests.
The adapter is correct on 75 requests.
The illustrative absolute difference is five percentage points.
Suppose both are correct on 65 of the same requests.
Then the adapter alone is correct on 10 requests.
The baseline alone is correct on five requests.
The remaining 20 requests are wrong for both.
The paired outcomes sum to 65+10+5+20=100.
The net gain is 10-5=5 requests, matching the marginal difference.
This constructed point estimate does not establish statistical significance.
If requests share customers, resample customers rather than pretending requests are independent.
#### Aggregate and subgroup caution
Suppose 90 common requests and 10 rare requests form the test set.
The baseline answers 72 common and eight rare requests correctly.
The adapter answers 81 common and five rare requests correctly.
Baseline aggregate accuracy is 80/100=80%.
Adapter aggregate accuracy is 86/100=86%.
The common slice rises from 72/90=80% to 81/90=90%.
The rare slice falls from 8/10=80% to 5/10=50%.
Aggregate improvement therefore coexists with a 30-point rare-slice decline.
The small rare denominator makes its estimate uncertain.
Predeclared slice guardrails and expert review determine whether this blocks release.
The numbers describe a constructed evaluation, not measured performance.
#### Missing outcome caution
Suppose 100 evaluation requests are attempted.
For the baseline, 90 are scored and 10 time out.
Of the scored baseline requests, 81 are correct.
Its conditional scored accuracy is 81/90=90%.
If timeouts count as failures, end-to-end success is 81/100=81%.
These two metrics answer different questions.
Report both the score denominator and the attempted-request denominator.
Apply the same timeout policy to the adapter.
Inspect whether missing cases cluster by length or task difficulty.
Do not silently discard failures to make the quality metric look better.
#### Token-weighted loss interpretation
Suppose one example has two supervised tokens with mean loss 1.
A second example has eight supervised tokens with mean loss 3.
The unweighted mean of example means is (1+3)/2=2.
The token-weighted mean is (2*1+8*3)/10=2.6.
Both arithmetic results are correct for their respective estimands.
The second weights the longer example four times as much.
Padding and prompt tokens must be excluded if they are masked.
Report the reduction convention before comparing training runs.
A lower token loss does not automatically imply better task completion.
#### Rollout accounting
Suppose a canary receives 200 eligible requests.
Suppose 180 complete successfully under the declared success contract.
The illustrative completion rate is 180/200=90%.
If 20 requests are excluded after assignment, the denominator changes.
Exclusions need a reason specified independently of the outcome.
Compare with the baseline over a similar period and request mix.
Include latency, quality and safety rather than success alone.
Keep automatic rollback thresholds separate from this invented example.
A canary does not prove performance on rare groups absent from its traffic.
#### Rank and scaling sensitivity
Use a rank-8 adapter with alpha 16.
The standard scaling factor is 16/8=2.
Double rank to 16 while keeping alpha at 16.
The standard scaling factor becomes 16/16=1.
Alternatively double alpha to 32 along with rank.
The resulting standard scaling factor remains 32/16=2.
Neither adjustment makes the optimization trajectories equivalent.
Factor initialization and attainable update subspace also change.
Record the complete configuration and compare repeated runs where feasible.
#### Memory unit sensitivity
Suppose a tensor contains 1,073,741,824 scalar entries.
At two bytes per entry it occupies 2,147,483,648 bytes.
This is exactly 2 GiB under binary units.
It is approximately 2.147 GB under decimal units.
Confusing the two units can conceal a capacity shortfall.
This computation covers the tensor payload, not allocator reservation.
If stored in a quantized format, account for scales and packing.
If sharded, count the local shard and any transient gathered copy.
#### Grouped uncertainty
Suppose 1,000 requests come from only ten customers.
The number of requests is not necessarily the number of independent units.
Requests from the same customer may share templates and subject matter.
A request-level confidence interval can therefore be too narrow.
Use customer-level resampling if the estimand and sample design permit.
Ten customers may still be too few for a stable interval.
Report the number of customers as well as the number of requests.
Recruit more independent groups when the decision requires tighter uncertainty.
**Takeaway:** The production decision depends on the actual estimand, artifact, and population, not a single aggregate count.
## 12. Quick Reference Card
| Item | Value |
|---|---|
| Topic | LoRA parameter-efficient fine-tuning and production selection |
| Core formula | `W_effective = W + (alpha/r)BA`; independent adapter entries are `r*(d_in+d_out)` |
| Key assumption | The chosen implementation uses this scaling; count actual target shapes and additional trainable tensors |
| Complexity | One unmerged dense application adds O(r*(d_in+d_out)) arithmetic per input vector; actual runtime is implementation-dependent |
| Use when | A bounded low-rank update and frozen base satisfy measured quality, deployment and memory constraints |
| Avoid when | Unmeasured activation peaks or assumed quality are treated as proof that the workload fits or succeeds |
| Go deeper | [Collection index](INDEX.md); quantized-base details require their own implementation review |
| Memory | Include simultaneous base, adapter, gradient, optimizer, activations, workspace and allocator overhead |
| Evidence | Counts are illustrative and derived; no actual training or serving result is measured |
