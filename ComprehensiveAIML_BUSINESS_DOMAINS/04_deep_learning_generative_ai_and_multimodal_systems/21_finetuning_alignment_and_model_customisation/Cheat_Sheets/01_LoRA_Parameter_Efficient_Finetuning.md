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
**Source boundary:** [directory.txt Module_08_PEFT_Model_Merging](../../../../Templates/directory.txt) lists 8.2 LoRA and related PEFT topics; it is an inventory, not lesson text or benchmark evidence.
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
### Out of memory before backward
- Symptom: Out of memory before backward.
- Candidate cause: Base or temporary loading peak.
- Competing explanation: Activation peak during training.
- Discriminating check: Measure device allocation by phase.
- Bounded action: Use suitable base precision or placement.
- Verification: Repeat representative load and training step.
### Out of memory during backward
- Symptom: Out of memory during backward.
- Candidate cause: Long-sequence activations.
- Competing explanation: Optimizer state or fragmentation.
- Discriminating check: Profile peak by sequence length and allocation.
- Bounded action: Reduce microbatch or checkpoint activations.
- Verification: Measure full-step peak and throughput.
### Training loss falls but evaluation fails
- Symptom: Training loss falls but evaluation fails.
- Candidate cause: Leakage or overfit.
- Competing explanation: Wrong evaluation formatting.
- Discriminating check: Inspect group overlap and token templates.
- Bounded action: Re-split groups and align formatting.
- Verification: Rerun untouched holdout.
### Adapter has no effect
- Symptom: Adapter has no effect.
- Candidate cause: Wrong base or adapter not loaded.
- Competing explanation: Evaluation uses stale cache.
- Discriminating check: Log immutable base and adapter hashes.
- Bounded action: Pin compatible artifact and invalidate cache.
- Verification: Compare deterministic logits against base.
### Initial model differs from base
- Symptom: Initial model differs from base.
- Candidate cause: Nonzero adapter initialization.
- Competing explanation: Dropout or preprocessing mismatch.
- Discriminating check: Disable dropout and inspect initial update norm.
- Bounded action: Use documented initialization and matching inputs.
- Verification: Compare initial logits within precision tolerance.
### Quality falls after merge
- Symptom: Quality falls after merge.
- Candidate cause: Quantization and rounding.
- Competing explanation: Wrong scale or merge target.
- Discriminating check: Compare unmerged and merged logits before requantization.
- Bounded action: Merge in suitable precision with correct metadata.
- Verification: Evaluate merged artifact on held-out tasks.
### Wrong tenant behavior
- Symptom: Wrong tenant behavior.
- Candidate cause: Adapter routing race.
- Competing explanation: Incorrect tenant authorization.
- Discriminating check: Trace request-to-adapter immutable binding.
- Bounded action: Pin authorized adapter for complete generation.
- Verification: Run concurrent cross-tenant negative tests.
### Checkpoint cannot resume
- Symptom: Checkpoint cannot resume.
- Candidate cause: Missing optimizer or scheduler state.
- Competing explanation: Changed data order or base revision.
- Discriminating check: Inspect checkpoint manifest and state counters.
- Bounded action: Restore complete compatible training state.
- Verification: Compare next-step loss and parameters.
### Toxicity regression
- Symptom: Toxicity regression.
- Candidate cause: Training distribution alters safety behavior.
- Competing explanation: Evaluator or decoding changed.
- Discriminating check: Run fixed safety slices against both models.
- Bounded action: Revise data or reject release.
- Verification: Review blinded outputs and uncertainty.
### Short examples improve but long ones fail
- Symptom: Short examples improve but long ones fail.
- Candidate cause: Sequence truncation.
- Competing explanation: Position or prompt template mismatch.
- Discriminating check: Inspect token lengths and actual masks.
- Bounded action: Correct truncation and evaluate long slices.
- Verification: Check task metrics by length.
### Instability at first updates
- Symptom: Instability at first updates.
- Candidate cause: Excessive learning rate or scale.
- Competing explanation: Nonfinite data or precision overflow.
- Discriminating check: Inspect factor gradients and loss scaling.
- Bounded action: Adjust configuration after isolating cause.
- Verification: Run monitored training with finite checks.
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
### 2. What is the shape of the update?
**Answer:** For W of shape d_out by d_in, A is r by d_in and B is d_out by r. BA has the same shape as W.
### 3. Count a rank-8 adapter on a 4096 square projection.
**Answer:** 8*(4096+4096)=65,536 trainable scalar values, excluding bias and any extra trainable tensors.
### 4. How many parameters for 32 blocks with query and value targeted?
**Answer:** There are 64 independent projections; 64*65,536=4,194,304. Tying or unequal widths changes the answer.
### 5. Why is rank eight not a guarantee of an eight-dimensional final model?
**Answer:** Only the additive update has rank at most eight; adding it to an arbitrary full-rank base can yield a full-rank effective weight.
### 6. Why can a small adapter still run out of memory?
**Answer:** Frozen base weights, activations, temporary workspaces, allocator overhead and distributed buffers remain resident; long sequences may dominate peak.
### 7. How do you distinguish training data leakage from a format bug?
**Answer:** Inspect group overlap and compare actual tokenized prompts, targets and masks across splits. Correct each identified defect and re-evaluate on untouched groups.
### 8. When should retrieval be preferred?
**Answer:** For changing or attributable facts, test authorized retrieval because weights are not a freshness mechanism. Retrieval itself needs evidence, access control and evaluation.
### 9. What changes when rank doubles?
**Answer:** For fixed target shapes, factor count doubles. Under alpha/r scaling, holding alpha fixed halves the explicit scale; total memory and quality need not double.
### 10. How do you validate merging?
**Answer:** Pin base and adapter versions; compare merged and unmerged logits in suitable precision, then run the same held-out quality and safety suite on the deployed artifact.
### 11. How would you test tenant isolation?
**Answer:** Bind authorized adapter identity for the whole request, test concurrent cross-tenant requests and cache keys, and verify outputs and access logs; factors alone are not security boundaries.
### 12. What does a paired evaluation require?
**Answer:** Run base and adapted models on the same independent held-out groups with identical prompts and decoding, define metric denominators and uncertainty, and inspect safety and subgroup regressions.
**Takeaway:** Strong answers distinguish exact algebra, experiment design, and production evidence.
### Production decision drills
Each drill is an ILLUSTRATIVE design exercise; proposed checks are not observed results.
#### Target inventory
- Decision input: projection names.
- Failure under consideration: missing or duplicated module.
- Evidence to collect: unique parameter identities.
- First controlled action: enumerate shapes before attaching factors.
#### Rank sweep
- Decision input: rank and scaling.
- Failure under consideration: confounded scale change.
- Evidence to collect: fixed task set and training budget.
- First controlled action: record both alpha and effective scale.
#### Learning rate
- Decision input: optimizer step size.
- Failure under consideration: unstable or stalled convergence.
- Evidence to collect: loss and gradient norms by step.
- First controlled action: compare schedules under equal tokens.
#### Token masking
- Decision input: supervised target tokens.
- Failure under consideration: learning from prompt or padding.
- Evidence to collect: decoded examples and loss mask.
- First controlled action: validate boundaries with hand-labeled samples.
#### Data grouping
- Decision input: customer and document identities.
- Failure under consideration: correlated leakage across splits.
- Evidence to collect: group intersection counts.
- First controlled action: split before augmentation.
#### Temporal evaluation
- Decision input: event and cutoff timestamps.
- Failure under consideration: future information in training.
- Evidence to collect: latest training timestamp by group.
- First controlled action: reserve a later test interval.
#### Deduplication
- Decision input: content hashes and near duplicates.
- Failure under consideration: test contamination.
- Evidence to collect: cross-split duplicate audit.
- First controlled action: deduplicate with a documented policy.
#### Rare slices
- Decision input: slice membership.
- Failure under consideration: aggregate hides regression.
- Evidence to collect: per-slice numerator and denominator.
- First controlled action: predefine minimum sample and escalation.
#### Safety regression
- Decision input: harm categories.
- Failure under consideration: task optimization degrades refusal.
- Evidence to collect: paired blinded review.
- First controlled action: block release on material harm.
#### Factuality
- Decision input: source-supported claims.
- Failure under consideration: confident unsupported answer.
- Evidence to collect: claim-level evidence review.
- First controlled action: retain retrieval and citations.
#### Output schema
- Decision input: valid structured responses.
- Failure under consideration: syntactically valid but wrong values.
- Evidence to collect: parse and semantic validation separately.
- First controlled action: check downstream contract.
#### Quantized base
- Decision input: weight representation.
- Failure under consideration: accuracy or kernel mismatch.
- Evidence to collect: matched precision comparisons.
- First controlled action: record quantization configuration.
#### Merge precision
- Decision input: merge dtype.
- Failure under consideration: rounding makes merged model differ.
- Evidence to collect: logit comparison before requantization.
- First controlled action: test exact deployment artifact.
#### Adapter cache
- Decision input: base and adapter versions.
- Failure under consideration: stale adapter reused.
- Evidence to collect: request trace and cache identity.
- First controlled action: key on immutable compatible versions.
#### Tenant routing
- Decision input: authorization identity.
- Failure under consideration: wrong adapter applied.
- Evidence to collect: concurrent negative tests.
- First controlled action: bind identity to generation session.
#### Rollback
- Decision input: previous accepted artifact.
- Failure under consideration: missing dependent base.
- Evidence to collect: restore drill and manifest.
- First controlled action: keep compatible base and tokenizer.
#### Training resume
- Decision input: optimizer and RNG state.
- Failure under consideration: non-equivalent restart.
- Evidence to collect: compare step counter and sample order.
- First controlled action: save full training state.
#### Context length
- Decision input: tokenized sequence length.
- Failure under consideration: truncation changes labels.
- Evidence to collect: length histogram and actual masks.
- First controlled action: evaluate representative long examples.
#### Optimizer memory
- Decision input: moments and master weights.
- Failure under consideration: state exceeds assumptions.
- Evidence to collect: dtype inventory at step.
- First controlled action: sum allocated state per device.
#### Activation memory
- Decision input: microbatch and sequence lengths.
- Failure under consideration: peak occurs in backward.
- Evidence to collect: phase-level peak allocation.
- First controlled action: measure after warmup.
#### Evaluation uncertainty
- Decision input: independent group count.
- Failure under consideration: false confidence from correlated rows.
- Evidence to collect: group bootstrap or appropriate intervals.
- First controlled action: report interval and sample unit.
#### Production monitoring
- Decision input: quality proxy and latency.
- Failure under consideration: proxy hides harmful failure.
- Evidence to collect: audited outcome sample.
- First controlled action: define ownership and alert response.
#### Distribution shift
- Decision input: input and task mix.
- Failure under consideration: offline result does not transfer.
- Evidence to collect: time and segment comparisons.
- First controlled action: collect consented representative labels.
#### Serving throughput
- Decision input: batch mix and concurrency.
- Failure under consideration: adapter path launch overhead.
- Evidence to collect: end-to-end latency distribution.
- First controlled action: benchmark real adapter mix.
#### License and privacy
- Decision input: training data permission.
- Failure under consideration: unauthorized reuse.
- Evidence to collect: dataset lineage and retention audit.
- First controlled action: exclude unapproved records.
#### Reproducibility
- Decision input: base code and data revisions.
- Failure under consideration: artifact cannot be reconstructed.
- Evidence to collect: immutable manifest completeness.
- First controlled action: record seeds and environment.
#### Evaluation budget
- Decision input: number of candidate adapters.
- Failure under consideration: overfitting to validation.
- Evidence to collect: predeclared final untouched test.
- First controlled action: limit repeated test access.
#### Preference leakage
- Decision input: human reviewer identity.
- Failure under consideration: correlated subjective judgments.
- Evidence to collect: reviewer and group stratification.
- First controlled action: use blinded adjudication.
#### Cross-domain retention
- Decision input: general capability slices.
- Failure under consideration: specialization causes forgetting.
- Evidence to collect: paired general-task comparison.
- First controlled action: set regression guardrails.
#### Failure fallback
- Decision input: service health and model quality.
- Failure under consideration: bad adapter stays active.
- Evidence to collect: reversion trigger and exercise.
- First controlled action: route to accepted baseline.
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
### Derivation lab: gradients and initialization
For a single vector, let y = Wx + sBAx, with s = alpha/r.
Let g = dL/dy and z = Ax; all vectors are columns.
The loss differential is dL = g^T dy.
Holding x fixed, the B contribution is s g^T(dB)z.
Thus dL/dB = s g z^T, an outer product of shape d_out by r.
The A contribution is s g^T B(dA)x.
Thus dL/dA = s B^T g x^T, of shape r by d_in.
The input gradient is W^T g + s A^T B^T g.
A frozen W has no weight update, but its transpose can still propagate gradients.
If B starts at zero, dL/dA is zero on the first backward pass.
If A and g are nonzero, dL/dB need not be zero at that same pass.
After B changes, a subsequent backward pass can update A as well.
If both factors start at zero, both factor gradients vanish under this loss path.
Regularization or another loss path could alter that conclusion.
Take x=(1,2), A=[1,0], B=(0,0)^T, g=(3,4), s=2.
Here z=1, so dL/dB=(6,8)^T.
The initial dL/dA is [0,0] because B is zero.
With learning rate 0.1 and plain SGD, the new B is (-0.6,-0.8)^T.
At the same x and g, B^T g=-5 and dL/dA=[-10,-20].
This second gradient is conditional on holding g fixed, not a claim about a real network.
Gradient accumulation sums or averages such contributions according to the implementation.
A zero factor gradient on the first step is therefore not by itself a broken adapter.
A persistently zero B gradient instead warrants checking inputs, masks and graph connectivity.
A detached adapter path also blocks learning even if factors are marked trainable.
Inspect gradients before the optimizer clears them.
Weight decay may move parameters even when the data gradient is zero.
The factor gradients are not the same as the gradient of an unconstrained dense update.
The factorization couples their optimization and changes the accessible update trajectory.
### Derivation lab: factor symmetries and rank
For any invertible r by r matrix R, BA = (BR)(R^-1 A).
Consequently individual factor magnitudes are not uniquely determined by the effective update.
For a scalar c other than zero, (cB)(A/c)=BA.
A large norm of B alone therefore does not prove a large functional change.
For example B=(2,0)^T and A=[3,0] produce [[6,0],[0,0]].
Replacing B by (20,0)^T and A by [0.3,0] produces the same matrix.
Their separate squared norms differ, despite equal dense updates.
Factor-wise weight decay can therefore distinguish equivalent factorizations.
Rank is at most r, but linearly dependent columns can make it smaller.
For B=[[1,1],[0,0]] and A=[[1,0],[0,1]], BA has rank one.
The nominal rank is two although the realized update has only one nonzero singular value.
A singular-value inspection of BA can diagnose unused capacity.
For large matrices, use a small-factor method or an approximation rather than materializing BA unnecessarily.
Do not infer a globally optimal rank from one checkpoint spectrum.
A small singular value can still matter on an important input direction.
The update response on an input x depends on Ax, not just on ||BA||.
If x lies in the null space of A, the adapter contributes zero for that x.
The null space can contain relevant examples even when training loss decreases.
For A=[1,0], x=(0,1) is such an input.
For x=(1,0), the same A passes a nonzero coordinate to B.
Evaluation should therefore include examples spanning meaningful input conditions.
A nonlinear stack of adapted layers is not equivalent to a single low-rank end-to-end map.
Intervening attention and nonlinearities alter how layerwise changes compose.
Even one full-rank W plus a rank-one update can have full rank.
With W=I_2 and BA=diag(1,0), the effective matrix diag(2,1) has rank two.
Conversely, a low-rank update can reduce the rank of W.
With W=I_2 and BA=diag(-1,0), the effective matrix has rank one.
Neither example predicts task quality; both only establish algebraic possibilities.
### Derivation lab: target approximation
Consider a desired dense update D=diag(6,3,1).
Its singular values are 6, 3 and 1.
For an unconstrained rank-k approximation in Frobenius norm, truncated SVD keeps the k largest singular values.
At k=1, the optimal approximation is diag(6,0,0).
The squared residual is 3^2+1^2=10.
The residual Frobenius norm is sqrt(10), approximately 3.162.
At k=2, the optimal approximation is diag(6,3,0).
Its squared residual is 1 and its residual norm is 1.
At k=3 the residual is zero.
The squared norm of D is 36+9+1=46.
Rank one captures 36/46, approximately 78.26%, of squared Frobenius energy.
Rank two captures 45/46, approximately 97.83%.
This approximation result assumes the target dense update is already known.
It does not say the low-rank training objective will find the truncated SVD.
Frobenius error also weights all matrix coordinates equally.
A task distribution can concentrate on a discarded direction.
For x=(0,0,1), the rank-two approximation outputs zero while D outputs (0,0,1).
Thus high retained matrix energy can coexist with complete error on that input.
A loss-weighted or data-weighted criterion may lead to different priorities.
If two singular values tie at the cutoff, the best rank-k subspace need not be unique.
This is an offline diagnostic, not a substitute for held-out behavioral evaluation.
The full target update might itself be noisy or overfit.
Comparing spectra requires compatible target shapes and consistent normalization.
The memory required to compute an exact SVD can exceed that of the adapter.
#### Allocation exercise: square attention
Assume one independent 4096 by 4096 dense weight, with no trainable bias.
The base matrix contains 16,777,216 scalar entries.
With rank 8, A has 32768 entries and B has 32768 entries.
The first adapter therefore contains 65,536 trainable entries.
At rank 16, A and B together contain 131,072 entries.
The rank increase adds 65,536 trainable entries per target.
At two bytes per factor entry, the extra factor payload is 131,072 bytes.
This payload excludes gradients, optimizer state, metadata and the frozen weight.
At fixed alpha=16, alpha/r changes from 2.00 to 1.00.
Keeping the original scale instead requires alpha=32 at rank 16.
The factor count comparison does not establish a quality difference.
If the target is shared across layers, count one factor pair per actual allocation.
If every layer owns a separate pair, multiply these counts by the number of layers.
Check that the stored weight orientation matches the runtime's linear convention.
The example is derived arithmetic for an assumed shape, not a measured workload.
#### Allocation exercise: grouped value
Assume one independent 1024 by 4096 dense weight, with no trainable bias.
The base matrix contains 4,194,304 scalar entries.
With rank 8, A has 32768 entries and B has 8192 entries.
The first adapter therefore contains 40,960 trainable entries.
At rank 16, A and B together contain 81,920 entries.
The rank increase adds 40,960 trainable entries per target.
At two bytes per factor entry, the extra factor payload is 81,920 bytes.
This payload excludes gradients, optimizer state, metadata and the frozen weight.
At fixed alpha=16, alpha/r changes from 2.00 to 1.00.
Keeping the original scale instead requires alpha=32 at rank 16.
The factor count comparison does not establish a quality difference.
If the target is shared across layers, count one factor pair per actual allocation.
If every layer owns a separate pair, multiply these counts by the number of layers.
Check that the stored weight orientation matches the runtime's linear convention.
The example is derived arithmetic for an assumed shape, not a measured workload.
#### Allocation exercise: expansion projection
Assume one independent 11008 by 4096 dense weight, with no trainable bias.
The base matrix contains 45,088,768 scalar entries.
With rank 4, A has 16384 entries and B has 44032 entries.
The first adapter therefore contains 60,416 trainable entries.
At rank 8, A and B together contain 120,832 entries.
The rank increase adds 60,416 trainable entries per target.
At two bytes per factor entry, the extra factor payload is 120,832 bytes.
This payload excludes gradients, optimizer state, metadata and the frozen weight.
At fixed alpha=8, alpha/r changes from 2.00 to 1.00.
Keeping the original scale instead requires alpha=16 at rank 8.
The factor count comparison does not establish a quality difference.
If the target is shared across layers, count one factor pair per actual allocation.
If every layer owns a separate pair, multiply these counts by the number of layers.
Check that the stored weight orientation matches the runtime's linear convention.
The example is derived arithmetic for an assumed shape, not a measured workload.
#### Allocation exercise: vocabulary-side head
Assume one independent 32000 by 2048 dense weight, with no trainable bias.
The base matrix contains 65,536,000 scalar entries.
With rank 2, A has 4096 entries and B has 64000 entries.
The first adapter therefore contains 68,096 trainable entries.
At rank 4, A and B together contain 136,192 entries.
The rank increase adds 68,096 trainable entries per target.
At two bytes per factor entry, the extra factor payload is 136,192 bytes.
This payload excludes gradients, optimizer state, metadata and the frozen weight.
At fixed alpha=4, alpha/r changes from 2.00 to 1.00.
Keeping the original scale instead requires alpha=8 at rank 4.
The factor count comparison does not establish a quality difference.
If the target is shared across layers, count one factor pair per actual allocation.
If every layer owns a separate pair, multiply these counts by the number of layers.
Check that the stored weight orientation matches the runtime's linear convention.
The example is derived arithmetic for an assumed shape, not a measured workload.
#### Allocation exercise: bottleneck projection
Assume one independent 128 by 512 dense weight, with no trainable bias.
The base matrix contains 65,536 scalar entries.
With rank 8, A has 4096 entries and B has 1024 entries.
The first adapter therefore contains 5,120 trainable entries.
At rank 16, A and B together contain 10,240 entries.
The rank increase adds 5,120 trainable entries per target.
At two bytes per factor entry, the extra factor payload is 10,240 bytes.
This payload excludes gradients, optimizer state, metadata and the frozen weight.
At fixed alpha=16, alpha/r changes from 2.00 to 1.00.
Keeping the original scale instead requires alpha=32 at rank 16.
The factor count comparison does not establish a quality difference.
If the target is shared across layers, count one factor pair per actual allocation.
If every layer owns a separate pair, multiply these counts by the number of layers.
Check that the stored weight orientation matches the runtime's linear convention.
The example is derived arithmetic for an assumed shape, not a measured workload.
#### Allocation exercise: audio cross-modal projection
Assume one independent 3072 by 768 dense weight, with no trainable bias.
The base matrix contains 2,359,296 scalar entries.
With rank 8, A has 6144 entries and B has 24576 entries.
The first adapter therefore contains 30,720 trainable entries.
At rank 16, A and B together contain 61,440 entries.
The rank increase adds 30,720 trainable entries per target.
At two bytes per factor entry, the extra factor payload is 61,440 bytes.
This payload excludes gradients, optimizer state, metadata and the frozen weight.
At fixed alpha=16, alpha/r changes from 2.00 to 1.00.
Keeping the original scale instead requires alpha=32 at rank 16.
The factor count comparison does not establish a quality difference.
If the target is shared across layers, count one factor pair per actual allocation.
If every layer owns a separate pair, multiply these counts by the number of layers.
Check that the stored weight orientation matches the runtime's linear convention.
The example is derived arithmetic for an assumed shape, not a measured workload.
#### Allocation exercise: expert expansion
Assume one independent 14336 by 4096 dense weight, with no trainable bias.
The base matrix contains 58,720,256 scalar entries.
With rank 4, A has 16384 entries and B has 57344 entries.
The first adapter therefore contains 73,728 trainable entries.
At rank 8, A and B together contain 147,456 entries.
The rank increase adds 73,728 trainable entries per target.
At two bytes per factor entry, the extra factor payload is 147,456 bytes.
This payload excludes gradients, optimizer state, metadata and the frozen weight.
At fixed alpha=8, alpha/r changes from 2.00 to 1.00.
Keeping the original scale instead requires alpha=16 at rank 8.
The factor count comparison does not establish a quality difference.
If the target is shared across layers, count one factor pair per actual allocation.
If every layer owns a separate pair, multiply these counts by the number of layers.
Check that the stored weight orientation matches the runtime's linear convention.
The example is derived arithmetic for an assumed shape, not a measured workload.
### Interview drill: optimizer and batching
**Question:** Does freezing W make backpropagation through W unnecessary?
**Answer:** No; earlier adapters require input gradients through later frozen operations.
For y=Wx, the input gradient is W^T g even when dL/dW is not retained.
A framework may avoid storing a weight gradient while still computing that input gradient.
**Question:** Are eight microbatches equivalent to one batch eight times larger?
**Answer:** Only under compatible loss normalization, optimizer timing and stochastic behavior.
Accumulate gradients for eight microbatches, then perform one optimizer step for a comparable update.
Eight optimizer steps instead change both moments and parameters eight times.
If supervised-token counts vary, average by total supervised tokens for the token-level objective.
Averaging eight microbatch means equally weights microbatches, not necessarily tokens.
Dropout and random data order can prevent bitwise equivalence even with matching normalization.
Activation memory is usually governed by microbatch shape, not merely effective batch size.
**Question:** Is an adapter checkpoint sufficient for exact training resumption?
**Answer:** Not necessarily; restore optimizer, scheduler, step, scaler and random states as applicable.
Also restore data position, sample order and distributed configuration where relevant.
An adapter-only export may be adequate for inference but omits training-state information.
Even with state restored, nondeterministic kernels can prevent bitwise reproduction.
Compare the next training step under a documented reproducibility tolerance.
**Question:** Does the alpha/r ratio alone determine the update size?
**Answer:** No; factor values and input direction also matter.
The same explicit scale applied to different BA matrices gives different residuals.
The same BA can produce different outputs for different x.
Record the scale convention, initialization and optimizer with every rank comparison.
### Interview drill: deployment and safety
**Question:** Why might merged and unmerged inference differ numerically?
**Answer:** Arithmetic ordering and precision change rounding, especially after requantization.
In exact arithmetic, (W+sBA)x = Wx+sB(Ax).
Floating-point matrix multiplication is not generally associative.
A quantized base may need dequantization before adding the update.
Requantizing the resulting dense weight introduces an additional approximation.
Compare logits first at matched inputs and precision, then test behavior on held-out cases.
**Question:** Can one adapter be safely loaded onto any checkpoint of the same architecture?
**Answer:** No; shape compatibility is necessary but not sufficient.
The learned update depends on the particular frozen base and target naming convention.
Tokenizer and prompt-template changes can also alter behavior without changing tensor shapes.
Pin the base revision, adapter configuration and tokenizer as an artifact set.
**Question:** Does an adapter isolate confidential tenant training examples?
**Answer:** No; it can encode or reveal information learned during training.
Access control, data minimization and leakage testing remain necessary.
Binding an adapter to the wrong request is a routing failure, not a rank failure.
Do not switch an adapter mid-generation without an explicitly supported cache policy.
Key-value caches may contain activations produced by the prior model configuration.
A cache must be scoped to compatible model, adapter, prompt and request identity.
**Question:** What if aggregate task quality rises while a critical subgroup regresses?
**Answer:** Apply the prespecified subgroup and safety decision rules rather than averaging the regression away.
Inspect sample size and uncertainty before interpreting the magnitude.
A failed guardrail requires investigation or a fallback, not an automatic release.
All these cases are hypothetical; no production behavior is asserted.
### Worked lab: shared and fused targets
Suppose a block computes query, key and value using one fused weight.
Its input width is 8 and the concatenated output width is 24.
A single rank-2 fused adapter has 2*(8+24)=64 entries.
Partition its B matrix vertically into query, key and value row groups.
All three groups multiply the same A and therefore share an input-side subspace.
Alternatively attach three independent rank-2 adapters to three separate 8 by 8 weights.
Each pair has 2*(8+8)=32 entries.
All three independent pairs contain 96 entries.
Their three A matrices can learn different input-side subspaces.
The fused and independent choices therefore differ in both capacity and count.
A fused rank-2 update has rank at most two across the concatenated output.
Three independent rank-2 updates can have concatenated rank as high as six, subject to input width.
A library that adapts only query and value slices may implement a different masking scheme.
Inspect the actual wrapper and allocated factor shapes rather than guessing from the module name.
Now suppose an embedding and output projection share one parameter object.
Two names do not imply two independent base storages.
If separate adapters are attached at the two uses, however, their factors are distinct.
If the same factor pair is reused at both uses, count that pair once.
Reusing one factor pair also couples its gradients from the two paths.
Shared storage and shared adapter factors are separate design decisions.
A transpose at the output use can make the factor orientation nontrivial.
Check both forward paths against a small explicit matrix example.
Merging into a tied base can unexpectedly change every consumer of that storage.
A reversible unmerged comparison can expose this coupling before artifact creation.
Never assume that a fused QKV name guarantees three identical output widths.
Grouped-query attention commonly has smaller key/value widths than query width.
For input width 8, query output 8 and key/value outputs 2 each, the fused output is 12.
A rank-2 fused pair has 2*(8+12)=40 entries.
Separate rank-2 query and value pairs have 2*(8+8)+2*(8+2)=52 entries.
The second configuration does not modify key, while the first can modify all fused output rows.
Counts alone cannot establish that the two target sets are functionally equivalent.
### Worked lab: finite precision and merging
Let W=[[1,0],[0,1]], A=[1,2], B=[3,4]^T and s=0.5.
The outer product BA is [[3,6],[4,8]].
The scaled update is [[1.5,3],[2,4]].
The exact merged matrix is [[2.5,3],[2,5]].
For x=[2,1]^T, the base output is [2,1]^T.
The adapter coordinate Ax equals 4.
The scaled adapter output is [6,8]^T.
Their sum is [8,9]^T.
The merged matrix applied to x also gives [8,9]^T in exact arithmetic.
This check covers both factor orientation and scale for this tiny case.
If scale is incorrectly applied twice, the adapter output becomes [3,4]^T instead.
If scale is omitted, the adapter output becomes [12,16]^T instead.
Such errors can be obscured by comparing only generated text.
Use deterministic tensor-level inputs and compare intermediate values when diagnosing a merge.
A floating-point tolerance must reflect the chosen dtype and operation ordering.
Absolute tolerance matters near zero, while relative tolerance matters at larger magnitudes.
Large logits can yield small probability differences, or small margins can flip an argmax.
Thus passing a numerical tolerance alone does not replace downstream evaluation.
Consider a toy quantizer that rounds every value to the nearest integer.
Rounding the merged matrix above gives [[2,3],[2,5]] under ties-to-even.
Its output on x is [7,9]^T rather than [8,9]^T.
This deliberately simple example illustrates quantization error, not a production format.
Actual quantizers can use block scales, zero points and other rounding rules.
Merging may require a higher-precision temporary dense matrix.
That temporary allocation can be much larger than the adapter factors.
Measure peak memory during merge separately from inference memory.
Keep an immutable unmerged base and adapter if reversible rollback is required.
Repeated merging into the same weight can accidentally apply the update twice.
A merge operation should check whether its input artifact is already merged.
### Worked lab: loss masks and objective reductions
Suppose two sequences have supervised target counts 2 and 6.
Their individual supervised-token losses sum to 4 and 9 respectively.
Their per-example means are 2 and 1.5.
The example-weighted mean is (2+1.5)/2=1.75.
The token-weighted mean is (4+9)/(2+6)=1.625.
Both are valid objectives, but they weight examples differently.
For the token-weighted objective, a long response contributes more tokens to the numerator.
Masking an instruction token removes it from both numerator and denominator.
Masking a padding token should not change the loss if it was already excluded correctly.
A shifted causal target must align each logit with the following token label.
An off-by-one shift can train the wrong prediction even when tensor shapes match.
Inspect decoded tokens alongside label IDs and the mask at boundary positions.
For multi-turn dialogue, specify whether prior assistant turns contribute to the loss.
Separators and end-of-sequence tokens need an explicit supervision policy.
If truncation removes the entire answer, the sample has zero supervised tokens.
Exclude or handle zero-token examples explicitly to avoid division by zero.
For a packed batch, prevent unintended cross-document attention if isolation is required.
Concatenation alone does not automatically enforce document boundaries.
The training loss can fall while formatting differs from the serving prompt.
Run the same tokenizer and chat-template revision in both paths where intended.
A tokenizer change can invalidate saved token IDs even when text remains the same.
A fixed random split can still leak if near-duplicate documents cross it.
Split by the intended generalization unit before deriving packed training samples.
Audit overlap at document, customer or conversation level as appropriate.
The evaluation loss denominator must be stated before comparing runs.
A token-weighted objective need not align with request-level success.
Include task-specific success criteria and failure analysis in the held-out evaluation.
### Worked lab: memory by training phase
Suppose frozen base payload is 12 GiB and adapter payload is 32 MiB.
Assume adapter gradients consume another 32 MiB.
Assume Adam's two moments together consume 128 MiB.
Assume no master copy and no sharding in this deliberately simplified accounting.
Before activations and workspace, these tensors total 12 GiB plus 192 MiB.
Since 192 MiB is 0.1875 GiB, the subtotal is 12.1875 GiB.
Add a hypothetical 2 GiB of live activations at backward peak.
Add a hypothetical 1 GiB of concurrent temporary workspace.
The resulting tensor subtotal is 15.1875 GiB.
This excludes runtime allocations, fragmentation and any other simultaneous buffers.
A nominal 16 GiB device therefore has only 0.8125 GiB before those exclusions.
These numbers are a constructed budget, not a model-fit guarantee.
If activation checkpointing cuts the assumed live activations to 1 GiB, subtotal becomes 14.1875 GiB.
That calculation does not predict the extra recomputation time.
If an FP32 master copy of the 32 MiB two-byte factors is used, it adds 64 MiB.
Optimizer implementations can allocate their state lazily at the first step.
A run may consequently fit forward and backward but fail on the first optimizer step.
Measure loading, forward, backward, optimizer step and checkpoint saving separately.
A peak that combines maxima from different times is a conservative sum, not necessarily an observed peak.
Conversely, leaving out concurrent temporaries understates the actual peak.
For sharded training, distinguish persistent shards from transient all-gathered tensors.
CPU offload moves some storage and traffic to the host rather than eliminating it.
Long sequences affect activation memory and possibly attention workspace.
A rank reduction can save adapter state without resolving a dominant activation peak.
Record allocated and reserved memory separately when the allocator exposes both.
Checkpoint serialization can create host-side or device-side copies depending on the implementation.
### Worked lab: paired decisions and uncertainty
Suppose 40 independent evaluation groups each receive both configurations.
Both are correct on 24 groups and both are wrong on 6 groups.
The adapter alone is correct on 7 groups; the base alone is correct on 3.
The four paired cells sum to 40.
Baseline accuracy is (24+3)/40=67.5%.
Adapter accuracy is (24+7)/40=77.5%.
The paired point difference is (7-3)/40=10 percentage points.
Discordant pairs determine the observed net change in this binary example.
Aggregate accuracies alone do not reveal how many examples changed direction.
The discordant count is 10, which is limited evidence for a precise improvement estimate.
An exact paired test can use the discordant counts under its stated null assumptions.
A confidence interval for the difference is more useful than a binary significance label alone.
If multiple requests share a customer, the customer may be the appropriate resampling unit.
If groups were sampled nonrandomly, uncertainty from that sampling design must be acknowledged.
Choose a minimum meaningful improvement before viewing the results.
Also define unacceptable subgroup or safety regressions before comparing candidates.
Missing outputs should be counted or separately reported under a prespecified policy.
Changing a policy after inspecting failures can bias the comparison.
Use a development set for selecting rank and an untouched holdout for the final decision.
Repeatedly choosing candidates on the same holdout turns it into a development set.
A new holdout or appropriate nested evaluation can reduce this selection effect.
A small aggregate gain may not justify additional operational complexity.
A quality gain does not establish improved latency or reduced memory.
Those claims require representative measurements of the deployed configuration.
### Worked lab: exact rank-one optimization step
Use a two-dimensional input, a two-dimensional output, rank one and scale s=1.
Set x=(1,2)^T, target t=(0,1)^T and frozen W=I_2.
Initialize A=[1,0] and B=(0,0)^T so the adapter initially contributes zero.
The initial output is y=(1,2)^T and the residual y-t is (1,1)^T.
For L=0.5*||y-t||^2, the initial loss is 1.
The output gradient g=y-t equals (1,1)^T.
The bottleneck activation Ax equals 1.
Therefore dL/dB=g(Ax)^T=(1,1)^T.
Because B is zero, dL/dA=(0,0) on this first step.
With plain SGD at learning rate 0.1, B becomes (-0.1,-0.1)^T.
The same step leaves A=[1,0] unchanged.
On the same input after the step, the adapter output is (-0.1,-0.1)^T.
The new prediction is (0.9,1.9)^T and the residual is (0.9,0.9)^T.
The new loss is 0.5*(0.9^2+0.9^2)=0.81.
Recomputing the gradient at that point gives g=(0.9,0.9)^T.
Now B^T g=-0.18, so dL/dA=(-0.18,-0.36).
The next A update with the same learning rate would produce [1.018,0.036].
This second A update assumes ordinary simultaneous SGD using the current gradients.
The example uses one sample twice only to expose the algebra, not to model generalization.
With a batch, each example contributes an outer product to the factor gradients.
Averaging those contributions before a step differs from sequentially updating after each sample.
If the data loss gradient is zero, an optimizer with decoupled weight decay can still change factors.
An Adam update differs from this worked SGD trajectory because it uses moment estimates and normalization.
### Worked lab: unequal per-layer rank allocation
Consider two independent targets with shapes 1024 by 1024 and 4096 by 1024.
Their per-rank adapter costs are 2048 and 5120 entries, respectively.
Allocating rank 8 to each target requires 8*2048+8*5120=57,344 entries.
At 2 bytes per factor entry, the factor payload is 114,688 bytes.
An alternative allocation of ranks 16 and 8 requires 16*2048+8*5120=73,728 entries.
The alternative adds 16,384 entries and 32,768 bytes of factor payload.
By contrast, increasing only the second target from rank 8 to rank 12 costs 4*5120=20,480 entries.
The second increase costs more despite adding only half as many rank units.
A fixed parameter budget therefore constrains ranks differently for differently shaped matrices.
Suppose the available budget is exactly 57,344 entries.
Ranks 16 and 4 use 16*2048+4*5120=53,248 entries.
The unused remainder is 4096 entries, sufficient for two more rank units on the first target.
Ranks 18 and 4 therefore use the full 57,344-entry budget.
Both allocations meet the same factor-count budget but constrain different update spaces.
Holding count constant does not hold activation memory or per-module kernel overhead constant.
Rank choices should also respect each matrix's minimum dimension and library constraints.
If rank exceeds a matrix dimension, the factorization cannot increase update rank beyond that dimension.
Per-layer selection based on the final holdout would contaminate that holdout.
Use a development set for allocation choices and reserve a separate final evaluation.
A candidate with lower average loss may still regress on a protected task or subgroup.
### Worked lab: accumulation with variable supervised tokens
Take two microbatches containing 2 and 6 supervised tokens.
Suppose their summed losses are 4 and 9, giving means 2 and 1.5.
Averaging the two means gives 1.75, which assigns each microbatch half the weight.
The global token-mean loss is (4+9)/8=1.625.
To accumulate the global token-mean gradient, weight the two mean-loss gradients by 2/8 and 6/8.
Equivalently, accumulate summed-loss gradients and divide once by the total of 8 tokens.
The same principle applies across distributed workers when their token counts differ.
If the distributed framework averages worker gradients, its reduction factor must be included.
For two workers, one with 2 tokens and one with 6, averaging their local means gives the wrong token objective.
Scaling each local mean by twice its fraction of global tokens compensates for a subsequent two-worker average.
Those multipliers are 0.5 for the first worker and 1.5 for the second.
The resulting averaged gradient is 0.25*g1+0.75*g2, the desired token weighting.
This assumes g1 and g2 are gradients of the respective local token means.
If the runtime sums rather than averages worker gradients, these multipliers are not appropriate.
Gradient clipping before accumulation need not equal clipping the final accumulated gradient.
For scalar gradients 3 and 4, clipping each to magnitude 2 and summing gives 4.
Summing first gives 7, which clips to 2 under the same bound.
Record clipping placement when comparing supposedly equivalent effective batch sizes.
Check for an all-masked microbatch before normalizing by its supervised-token count.
A zero-token microbatch contributes zero summed loss and zero token count under this objective.
### Worked lab: rank-one update versus full matrix update
Let the target dense update be D=[[2,0],[0,1]].
For x=(1,0)^T, D produces (2,0)^T.
For x=(0,1)^T, D produces (0,1)^T.
A rank-one update U=diag(2,0) reproduces the first response exactly.
It produces zero on the second input, leaving residual (0,1)^T.
A rank-one update V=diag(0,1) instead reproduces the second response exactly.
It leaves residual (2,0)^T on the first input.
For squared output error with equal probability on the two inputs, U has expected error 0.5.
Under the same distribution, V has expected error 2.
If the first input has probability 0.1 and the second 0.9, U has expected error 0.9.
Under that skewed distribution, V has expected error 0.4.
The preferable rank-one direction therefore depends on the input distribution.
This example compares only two candidate matrices, not every possible rank-one matrix.
A distribution-weighted optimization may select a different factorization or direction.
If training underrepresents a rare critical slice, a rank constraint can intensify the trade-off.
A larger rank can represent D exactly, but that does not establish better held-out quality.
Changing rank also changes the number of learned parameters and may change optimization behavior.
Evaluate the slice and the aggregate rather than treating matrix approximation error as task success.
### Worked lab: interpreting update norms
Suppose ||W||_F=100 and ||sBA||_F=2 for a selected matrix.
The ratio of update norm to base norm is 2%, a dimensionless matrix-level summary.
For input x, the residual is sBAx and obeys ||sBAx||_2 <= ||sBA||_2 ||x||_2.
The spectral norm ||sBA||_2 is no greater than the Frobenius norm ||sBA||_F.
Consequently the displayed Frobenius norm yields an upper bound of 2||x||_2.
A small global norm ratio does not bound the relative output change when Wx is near zero.
For W=diag(100,0.01), the base output on x=(0,1)^T has norm 0.01.
Adding U=diag(0,1) changes that output by 1, one hundred times its original norm.
The Frobenius norm of U is 1 while ||W||_F is approximately 100.
Thus the global update-to-base norm ratio is approximately 1% in this example.
This discrepancy is an input-direction effect, not a contradiction of the norm bound.
Inspect activation-conditioned residuals on representative data when diagnosing large behavioral shifts.
Relative output ratios require care when the baseline output norm is close to zero.
A low update norm does not establish that sensitive data cannot be memorized.
A high update norm does not by itself establish that an adapter is unsafe or defective.
Track per-target norms as a diagnostic alongside task outcomes and gradient behavior.
Factor norms separately can change under equivalent rescalings of A and B.
Compare the effective product or its action on inputs when factorization symmetry matters.
### Worked lab: adapter lifecycle and compatibility
Assume a service hosts one frozen base and two separately trained adapters, X and Y.
A request authorized for X must not inherit Y merely because Y is already cached.
The request binding should include an immutable adapter version, not only a mutable display name.
A compatible base revision must be part of the binding and artifact manifest.
Two adapter files with identical factor shapes may still have been trained on different bases.
Shape validation catches incompatible dimensions but cannot prove semantic compatibility.
A model revision that renames target modules can make a previously valid mapping ambiguous.
Fail closed when the expected target list cannot be resolved exactly.
If factors are loaded asynchronously, keep the request waiting or reject it until the intended version is ready.
A partial load must not silently execute with some adapted modules missing.
For deterministic debugging, compare selected module outputs with adapters disabled and enabled.
A nonzero output difference establishes activation of the path, not task improvement.
For an end-to-end regression, check tokenizer, prompt, adapter and base versions together.
The same generated text does not prove the logits or hidden states are identical.
When merging for serving, retain an auditable relationship to the original base and factors.
A merged artifact for X must not be treated as the unmodified base when loading Y.
Otherwise Y would be applied on top of X unless explicitly designed and evaluated that way.
For a rollback, restore the full compatible artifact set and invalidate incompatible caches.
A key-value cache produced with one adapter should not be reused under another by default.
The model's adapter factors do not replace tenant authorization or input data controls.
Log artifact identifiers without logging sensitive request contents unless authorized.
These are design checks for hypothetical systems, not statements about a particular runtime.
### Worked lab: repeated model selection and reliability
Suppose four rank candidates are evaluated on the same development set.
The highest observed score may benefit from favorable sampling noise.
Choosing that candidate makes its development score an optimistic estimate of final performance.
An independent holdout estimates the selected candidate without repeating the same selection step.
If the holdout is used repeatedly to choose new ranks, it becomes part of development.
Predefine a selection rule combining the primary metric with required safety guardrails.
Report the number of candidates tried, the selection criterion and any excluded runs.
A single seed cannot separate a method difference from training randomness.
Repeated seeds can expose instability, but are not a substitute for independent evaluation groups.
For paired evaluation, use the same cases and scoring procedure for base and adapter.
If generation is stochastic, define how seeds or repeated samples enter the comparison.
Different decoding temperatures can confound the adapter effect.
A per-request score and a per-token loss are different estimands with different denominators.
A mean can conceal a high-severity regression on a small but important subset.
When a subgroup has few cases, report its count and uncertainty rather than claiming precision.
If no examples cover a required risk, the correct status is unmeasured, not passed.
A production canary can measure behavior only for traffic it actually receives.
Routing changes during the canary can alter its case mix and confound the comparison.
Preserve a baseline and specify rollback triggers before deploying a candidate.
No constructed numerical result in this sheet is evidence of measured deployment performance.
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
