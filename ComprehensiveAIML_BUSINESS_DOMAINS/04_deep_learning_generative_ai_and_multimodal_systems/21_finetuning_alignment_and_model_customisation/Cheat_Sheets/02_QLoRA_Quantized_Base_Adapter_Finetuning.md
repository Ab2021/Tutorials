---
collection: "finetuning-alignment-and-model-customisation"
file_id: "02"
topic: "QLoRA and quantized-base adapter fine-tuning"
level: 3
status: "draft"
version: "0.1"
created: "2026-11-07"
reviewed: null
review_trigger: "PEFT, Transformers, bitsandbytes, base checkpoint, workload, or evaluation protocol changes"
notation_style: "plain-text"
prerequisites: ["01_LoRA_Parameter_Efficient_Finetuning.md"]
pattern_tags: ["fine-tuning", "quantization", "memory", "evaluation"]
source_status: "OPEN"
owner_role: "ML platform and model adaptation teams"
source_index: "INDEX.md"
---
# Quantized-Base Low-Rank Adapter Fine-Tuning and QLoRA

## 1. File Header

> **Purpose:** Budget, train, evaluate, and deploy low-rank adapters on a frozen quantized base without confusing weight storage with total memory.
> **Prerequisites:** [LoRA](01_LoRA_Parameter_Efficient_Finetuning.md), matrix multiplication, derivatives, causal language-model loss, and bytes versus bits.
> **Key outcome:** Given matrix shapes, precision and measured runtime overhead, estimate a training footprint and construct an evaluation and release decision.
> **This file does NOT cover:** pretraining, complete quantization-aware training algorithms, RLHF, or safety certification.
> **Level:** 3 — Advanced / Production.
> **Version / reviewed:** 0.1 / not reviewed; draft and unreviewed.
> **Evidence:** `SOURCED` for attributed papers and versioned guides; `DERIVED` for identities; `ILLUSTRATIVE` for fictional examples; `VERIFIED` denotes arithmetic checks only; `OPEN` denotes unmeasured behavior.
> **Retrieval cue:** For an out-of-memory failure use Section 8; for capacity arithmetic use Sections 4–5; for rollout use Section 7.

The visible source index contains the clipped label `8.3_QLoRA_and_Quantization_Aware_Fine_Tuning… [TRUNCATED/UNVERIFIED]`; its missing suffix and contents are unavailable.
The label establishes a topic, not evidence of the original file's contents.
The separate visible `17.3_LoRA_and_QLoRA_Implementation.md` label indicates another related topic, not an accessible source document.
A platform team must decide whether a workload fits and meets quality and operational constraints, rather than merely whether model weights load.
No production hardware result, model score, or financial result in this sheet has been measured.
**Takeaway:** Treat the quantized model, trainable adapter, and whole execution environment as separate objects of evidence.

## 2. Concept Map / Mental Model

```text
Licensed checkpoint + approved examples + held-out evaluation
          |                         |
          v                         v
Frozen 4-bit codes + scales      Tokenize and mask labels
          |                         |
          v                         v
Dequantize for compute -----> forward + adapter branch
                                  |
                                  v
                         masked training loss
                                  |
                  backpropagate through frozen base
                                  |
                 adapter gradients -> optimizer update
                                  |
                 evaluate -> release gate -> monitor
```

Arrows indicate data or computation flow, not a claim of causality beyond the training algorithm.
The central invariant is that the pretrained quantized base is not updated by the adapter optimizer.
A frozen base still participates in forward and backward computations needed to propagate gradients to adapters.
The code representation, compute dtype, adapter dtype, and optimizer-state dtype may differ.
Peak memory is a time-dependent sum; adding component maxima that occur at different times is conservative, not an exact peak.
**Takeaway:** Backpropagating through a weight is not the same as optimizing that weight.

## 3. Foundational Definitions

| Term | Definition and notation | Domain / units | Confusion and operational check |
|---|---|---|---|
| Frozen base | Pretrained weight W is held fixed during adapter training. | W is an m-by-n matrix. | Frozen does not mean absent from the backward graph; inspect trainable names. |
| Quantizer | Q maps weights to codes and metadata; D reconstructs approximate numerical values. | Q(W) is encoded; D(Q(W)) has W's shape. | Reconstruction is generally not the original W; compare layer outputs. |
| NF4 | A four-bit NormalFloat codebook for normally distributed values used in QLoRA. | Sixteen representable code indices before block scaling. | Nonuniform NF4 is not a uniform signed four-bit integer format. |
| Block scale | A per-block factor used in reconstructing values from codes. | One scalar per block, with representation-dependent metadata. | It is not a model parameter count; inspect the implementation. |
| Double quantization | Quantization of quantization constants to reduce metadata footprint. | Extra encoded scales and higher-level metadata. | It is not a second independent full copy of all weights. |
| Adapter | Low-rank matrices A and B form a trainable weight update. | A is r-by-n; B is m-by-r; r is a positive integer. | Low rank applies to the update, not necessarily the effective weight. |
| Alpha | LoRA scaling parameter; in the conventional form scaling is alpha/r. | alpha is a scalar; r > 0. | Other scaling variants exist; record the chosen convention. |
| Compute dtype | Arithmetic representation used by a kernel after or during decoding. | Example BF16; device and kernel support matter. | Four-bit stored weights do not imply four-bit accumulators. |
| Optimizer state | Persistent optimizer-owned values such as Adam first and second moments. | Bytes per trainable element depend on implementation. | Paged storage manages pressure; it does not eliminate state. |
| Activation | Intermediate value retained or recomputed for backward. | Depends on shape, layer and sequence length. | It is not part of the serialized base checkpoint. |
| Gradient checkpointing | Trading recomputation for fewer retained activations. | Implementation- and graph-dependent. | It does not reduce every memory category. |
| Paging | Moving eligible optimizer memory between memory tiers under pressure. | Host/device capacity, transfer bandwidth and framework support apply. | It is neither a speed guarantee nor free GPU memory. |
| QAT | Quantization-aware training models quantization effects during training according to its algorithm. | Quantizer and trainable parameters vary by method. | Frozen quantized-base adapter training is not identical to QAT. |
| Held-out evaluation | A predefined set not used for fitting or model selection when estimating final performance. | Split unit must reflect deployment dependencies. | Repeated tuning against a holdout contaminates it. |
| Peak memory | Maximum live allocation or reserved footprint over a specified interval and device. | Bytes; allocator accounting must be named. | Model-load memory is not training-step peak. |

For NF4, the paper motivates its distribution-specific codebook; actual layer distributions and block normalization need not satisfy its idealized assumptions.
A code is an index and a scale is decoding metadata, not a small floating-point weight that can be interpreted without its format.
The [index glossary](INDEX.md#7-master-symbol-glossary) gives collection-level symbol cross-references.
**Takeaway:** Identify what each precision describes before comparing storage, arithmetic, or quality.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

Validity: one dense linear map with conventional LoRA scaling, fixed decoded base, no bias in the expression, and compatible dimensions.
Derivation: compute the base map and add a factored linear perturbation; matrix multiplication associativity permits either evaluation order.
**FORMULA [E1]:** `y = D(Q(W)) x + (alpha/r) B (A x)`.
**WHERE:** `x` is n-by-1; `W` and its decoded approximation are m-by-n; `A` is r-by-n; `B` is m-by-r; `y` is m-by-1; alpha is dimensionless.
**CONSTRAINT:** `r >= 1`; the actual architecture must use the specified projection and scaling convention.
**NUMERICAL EXAMPLE:** if `m=4`, `n=6`, `r=2`, `alpha=4`, and `A x = [1,2]`, the adapter contribution is `2 B [1,2]`.
With B rows `[1,0]`, `[0,1]`, `[1,1]`, `[1,-1]`, the contribution is `[2,4,6,-2]`.
**INTERPRETATION:** the added contribution has output dimension four and does not replace the frozen base output.
**Boundary:** if B is initialized to zero the adapter output is zero, but the decoded base can already differ from the original unquantized base.

Validity: dense A and B only, no trainable biases or other modules, and no parameter sharing across the matrices being counted.
Derivation: A contains r*n scalars and B contains m*r scalars; add the disjoint counts.
**FORMULA [E2]:** `P_adapter = r*(m+n)` per targeted matrix.
**WHERE:** `P_adapter` is scalar elements; m, n and r are the dimensions defined above.
**CONSTRAINT:** count each distinct parameter once; add biases, embeddings, heads or saved modules separately if trainable.
**NUMERICAL EXAMPLE:** for `m=4`, `n=6`, and `r=2`, `P_adapter=2*(4+6)=20` elements.
**INTERPRETATION:** compare with the 24 elements of W; a low rank is not necessarily a useful compression for a small matrix.
**Boundary:** if `r >= mn/(m+n)`, the adapter can have at least as many elements as W; low rank alone does not ensure a smaller parameter count.

### 4.2 General Case

Validity: L distinct targeted matrices with explicit dimensions and ranks; weights not covered by those matrices must be counted separately.
Derivation: sum E2 across independent matrix parameter sets, correcting for any tied parameters.
**FORMULA [E3]:** `P_ad = sum_i r_i*(m_i+n_i)` for i from 1 to L, absent parameter sharing.
**WHERE:** `m_i`, `n_i`, and `r_i` are matrix dimensions and adapter rank for target i; `P_ad` is elements.
**CONSTRAINT:** weights shared between named modules must not be double counted; target discovery must match the actual model.
**NUMERICAL EXAMPLE:** two matrices 4-by-6 and 6-by-4 at rank 2 have `20+20=40` trainable elements.
**INTERPRETATION:** adapter memory depends on target selection as well as rank.
**Boundary:** an `all-linear` selector is implementation-dependent; inspect the actual targeted modules and excluded output head.

Validity: one device, mutually exclusive memory categories measured or estimated at the same time, and no duplicated accounting for shared storage.
Derivation: physical live storage is the sum of disjoint live storage allocations; maximize that time-dependent sum over the interval.
**FORMULA [E4]:** `M_peak = max_t [M_base(t)+M_meta(t)+M_ad(t)+M_grad(t)+M_opt(t)+M_act(t)+M_temp(t)+M_other(t)]`.
**WHERE:** each M is bytes on the device: base codes, quantization metadata, adapter weights, gradients, optimizer state, retained activations, temporary buffers, and other non-overlapping allocations.
**CONSTRAINT:** do not add allocator reserved bytes to their already included allocated bytes; include unquantized base modules and communication buffers in the appropriate category.
**NUMERICAL EXAMPLE:** simultaneous hypothetical category values `4+0.2+0.1+0.1+0.2+3+0.4+0.5 = 8.5 GiB`.
**INTERPRETATION:** a 4 GiB code payload does not imply a 4 GiB training requirement.
**Boundary:** summing separately observed category peaks can exceed the true concurrent peak; unknown workspaces or fragmentation can make a rough estimate too low.

Validity: P dense quantized scalar weights, exactly four code bits each, no padding, metadata or unquantized weights included.
Derivation: four bits equal half a byte; binary GiB equals 2^30 bytes.
**FORMULA [E5]:** `M_codes = P/2 bytes`.
**WHERE:** P is the count of encoded weights, not necessarily the total model parameter count.
**CONSTRAINT:** add block scales, codebooks, padding and all non-four-bit tensors separately.
**NUMERICAL EXAMPLE:** `P=8,000,000,000` gives `4,000,000,000 bytes ≈ 3.7253 GiB`.
**INTERPRETATION:** this is a payload-only lower-level accounting component, not a promised GPU footprint.
**Boundary:** small or unquantized layers and temporary representations can materially alter actual memory.

### 4.3 Key Special Cases / Variants

| Variant | Change | Assumption | Use | Avoid | Reference |
|---|---|---|---|---|---|
| Full tuning | Optimize base parameters | Optimizer supports trainable model | Maximum update freedom when resources permit | Fit is impossible or data is inadequate | E4 with base gradients and states |
| LoRA | Frozen higher-precision base plus low-rank adapters | Targeted layers support adapters | Isolate adapter benefit without base quantization | Base precision cannot fit the device | E1 without Q/D |
| Quantized LoRA | Frozen quantized base plus adapter | Compatible quantizer and kernel | Lower base payload | Unsupported hardware or unacceptable quality | E1–E5 |
| QLoRA-style | NF4, double quantization and low-rank tuning; paging is a configurable mechanism | Version-compatible stack | Memory-constrained experiments | Assumed universal quality or throughput equivalence | [Paper](https://arxiv.org/abs/2305.14314) |
| Paged optimizer | State can use managed paging | Supported implementation and host resources | Spikes make paging useful after profiling | Host transfers or capacity violate latency/availability | Section 8 |

The original QLoRA paper reports its own scoped experiments and describes NF4, double quantization and paged optimizers; do not extrapolate its reported outcomes to every task or stack.
QLoRA-style use of adapters across linear layers is distinct from a setup that targets only attention projections.
The [PEFT 0.18.0 guide](https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/quantization) documents `target_modules="all-linear"` for QLoRA-style targeting; inspect actual matches and exclusions.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

At fixed matrix shape, adapter elements grow linearly with rank, while base code payload is unchanged.
As rank grows toward full dimension, the update may no longer be parameter-efficient even though factorization remains defined.
As sequence length grows, activations and attention work may dominate; their scaling depends on the actual attention implementation.
With all adapters disabled, the quantized base can still differ from the original base due to reconstruction error.
Quantizing scales does not mean every scale is free: nested metadata remains and must be counted.
A method that fits at initialization can fail when the first optimizer step materializes state.
**Takeaway:** The parameter formula is exact under its stated shape assumptions; runtime memory and quality require measurement.

## 5. Worked Numerical Example

**ILLUSTRATIVE / VERIFIED arithmetic:** a hypothetical support assistant adapts a dense model; no company or benchmark is represented.
**Decision:** decide whether a 12 GiB device has sufficient margin for an experimental run; final permission requires a measured peak.
**Input:** 8,000,000,000 encoded base weights; four-bit codes; metadata and remaining unquantized weights together 0.80 GiB.
**Input:** 32 layers, each with four distinct 4096-by-4096 targeted projections, rank 16, no extra trainable modules.
**Input:** adapter weights and gradients each use two bytes per trainable element; optimizer maintains two four-byte moments per trainable element.
**Input:** there is no separate master parameter copy in this hypothetical optimizer; if one exists, add it explicitly.
**Input:** simultaneous activations, temporaries and runtime reserve are assumed to be 3.00, 0.60 and 0.50 GiB respectively.
These are invented inputs, not a claim that any named model has this architecture or footprint.

| Step | Rule | Calculation | Result | Independent check |
|---|---|---|---|---|
| Quantized payload | E5 | 8,000,000,000 / 2 | 4,000,000,000 bytes ≈ 3.7253 GiB | Divide bytes by 1,073,741,824. |
| One adapter | E2 | 16*(4096+4096) | 131,072 elements | Each factor has 65,536 elements. |
| All adapters | E3 | 32*4*131,072 | 16,777,216 elements | 128 targets times 2^17 gives 2^24. |
| Weights | Count times width | 16,777,216*2 | 32 MiB | 2^24 times 2 bytes = 2^25 bytes. |
| Gradients | Count times width | 16,777,216*2 | 32 MiB | Same count and width as weights. |
| Moments | Count times width | 16,777,216*8 | 128 MiB | Two 4-byte moment tensors. |
| Adapter subtotal | Sum | 32+32+128 | 192 MiB = 0.1875 GiB | 192/1024 GiB. |
| Total | E4, same time | 3.7252903+0.80+0.1875+3.00+0.60+0.50 | ≈ 8.8128 GiB | Sum is below 12 GiB by ≈ 3.1872 GiB. |

The margin is approximately 26.56% of the nominal 12 GiB capacity; device sharing and additional allocations can consume it.
Wrong path: `8 billion * 4 bytes` mistakes bits for bytes and overstates the four-bit payload by a factor of eight.
Another wrong path: calling 3.7253 GiB the entire footprint omits the supplied metadata and training allocations.
The estimate is conditional on the stated simultaneous terms and excludes undocumented external consumers.
The assumed activation size does not predict longer sequences or a different batch size.
Measure the actual peak through a complete warm-up, forward, backward, optimizer update, save and evaluation cycle on the deployment stack.
Report allocated and reserved peaks separately; check process-level and device-level usage, including other processes.
A successful estimate is not proof of task quality, throughput, safety, or even successful allocation under fragmentation.
**Takeaway:** The arithmetic supports a candidate experiment, not a guarantee that a run fits.

## 6. Visual / Geometric Intuition

```text
Parameter space for one m-by-n base matrix (schematic, not to scale)
Original W -------- quantization error --------> D(Q(W))
                                                  |
                                         low-rank update B A
                                                  |
                                                  v
                                       D(Q(W)) + (alpha/r) B A
```

The horizontal arrow describes one realization of base reconstruction error, not an axis with measured distance.
The adapter update is constrained to rank at most r, whereas the quantization error can have rank greater than r.
Therefore an adapter cannot in general exactly cancel arbitrary quantization error at one layer.
Task loss may still be satisfactory because the training objective cares about outputs on the task distribution, not exact recovery of every weight.
Compensating an error in one layer can interact with nonlinearities and errors elsewhere; it is not a guarantee of global correction.
**Takeaway:** Low-rank adaptation has a constrained correction space, while task-level consequences must be tested.

## 7. Comparison / Selection Table

| Method | Preconditions | Quality claim | Cost tradeoff | Prefer when | Disqualifier | Evidence |
|---|---|---|---|---|---|---|
| Full fine-tuning | Trainable base and sufficient capacity | No universal superiority | Base gradients and optimizer states may dominate | Update freedom is needed and capacity permits | Memory, reproducibility or deployment constraints fail | DERIVED/OPEN |
| Higher-precision LoRA | Frozen base, supported target modules | Quality is task-specific | Fewer trainable states; base is not 4-bit | Base fits and quantization error is a concern | Base payload exceeds available memory | DERIVED/OPEN |
| Quantized-base LoRA | Compatible quantizer and adapters | Quality is task-specific | Smaller base payload; decoding and metadata cost | Memory is binding and validation passes | Unsupported kernel or task regression | SOURCED/OPEN |
| QLoRA-style | Compatible NF4, nested scales and chosen targets | Paper findings are experiment-specific | Paging may reduce device pressure but add traffic | Budget requires it and end-to-end tests pass | Unmeasured quality or unsuitable runtime | SOURCED/OPEN |
| Prompt-only base | Frozen inference path and acceptable prompts | No task-specific weight update | No adapter training; prompt/context costs remain | Strong baseline meets requirements | Needs adaptation or prompt violates constraints | OPEN |

Priority: first disqualify data, security, license and compatibility violations.
Next enforce quality and operational acceptance gates; a cheaper model is not acceptable if a critical safety slice regresses beyond the agreed limit.
Then compare measured cost and capacity across methods under equivalent data, base, context and evaluation.
If full tuning is not feasible, mark it infeasible rather than claiming its quality is worse.
If no method meets constraints, collect data, change hardware, narrow the task or escalate; do not manufacture a pass.

### Evaluation design

Split on the unit that prevents real deployment leakage: customer, conversation, document, time or another dependency as appropriate.
Keep test data independent of prompt selection, target-module selection, stopping and hyperparameter sweeps.
Compare the original base, quantized base without adapter, higher-precision LoRA, and quantized-base adapter where feasible.
Use the same held-out examples, prompt template, decoding settings and output budget for paired comparisons.
Record task correctness, format compliance, unsupported assertions, safety failures and latency as separate outcomes.
Measure slices such as long contexts, rare terms, languages, protected or otherwise high-impact groups where lawful and appropriate.
Define each metric's numerator, denominator, missing-output handling and confidence procedure before inspecting results.
Use paired uncertainty estimates at the independent sampling unit; correlated messages from one conversation are not independent samples.
Record multiple random seeds when training variability matters; report the number of attempts and any failed runs.
Review privacy and memorization risks separately; aggregate accuracy cannot establish data protection.
Check deployment quantization and decoding again, because a merged or re-quantized artifact is a different evaluated model.
`OPEN / NOT MEASURED`: no comparable benchmark, training run, latency result or confidence interval is asserted here.
**Takeaway:** Choose on a controlled, task-specific tradeoff, not on paper results detached from their setup.

## 8. Failure Modes & Edge Cases

All cases in this section are hypothetical diagnostics, not claims of observed incidents.
A negative diagnostic rules out only the tested explanation under the test's sensitivity and setup.
Escalate unresolved safety, security, numerical or infrastructure problems to the responsible owner rather than releasing.

| ID | Trigger / competing causes | Discriminating diagnostic | Bounded action and verification | Provenance |
|---|---|---|---|---|
| F-01 | OOM at load: base payload, metadata, another process | Device-level process and per-module allocation before any step | Use a compatible smaller base or device; remeasure load and full step | HYPOTHETICAL |
| F-02 | OOM at backward: activations, temporary buffers, fragmentation | Compare memory snapshots at forward and backward across sequence lengths | Reduce microbatch/context or checkpoint; confirm peak and quality | HYPOTHETICAL |
| F-03 | OOM at first optimizer step: lazy states or copies | Measure before and after first update and inspect optimizer state dtypes | Change optimizer/device or budget; rerun full step | HYPOTHETICAL |
| F-04 | No learning: no matched targets, mask error or zero gradients | Inspect trainable names, supervised token count and per-factor gradients | Correct setup and check nonzero updates after a step | HYPOTHETICAL |
| F-05 | Quality regression: quantization, data or decoding | Compare base, quantized base, LoRA and QLoRA on same paired slices | Fix dominant source and rerun held-out evaluation | HYPOTHETICAL |
| F-06 | Slow step: paging, decoding or input bottleneck | Profile host-device transfers, kernels and loader wait | Address measured bottleneck; check quality and throughput | HYPOTHETICAL |
| F-07 | Reload differs: wrong base revision, tokenizer or quantizer | Compare manifest, quantization config and fixed-input logits | Pin compatible components and test clean reload | HYPOTHETICAL |
| F-08 | Merge regression: extra quantization or unsupported merge | Evaluate separate and merged artifacts on identical examples | Keep separate adapter if merge not validated | HYPOTHETICAL |
| F-09 | NaN loss: dtype overflow, corrupt data or optimizer settings | Identify first nonfinite tensor; compare BF16/FP32 and token IDs | Repair source, lower instability and repeat controlled run | HYPOTHETICAL |
| F-10 | Misleading offline score: leakage, dependent rows or selection | Audit split ancestry and grouped paired intervals | Rebuild split and evaluate untouched holdout | HYPOTHETICAL |

### Diagnostic playbooks

#### F-011: Frozen base unexpectedly trainable

- **Signal:** Full base names appear among requires-grad parameters.
- **Competing explanation:** A saved embedding or head may be deliberately trainable.
- **Discriminating check:** Inspect named parameters and optimizer groups before the first update.
- **Bounded response:** Exclude unintended base tensors from the optimizer.
- **Verification:** Compare base tensor checksums before and after a step.
- **Boundary:** Checksums must refer to the same representation.
- **Prevention:** Record the intended trainable-module manifest.

#### F-012: No adapter targets matched

- **Signal:** The trainable parameter count is zero or unexpectedly small.
- **Competing explanation:** The architecture may use different projection names.
- **Discriminating check:** List module class names and compare against target selectors.
- **Bounded response:** Select actual supported linear modules.
- **Verification:** Print matched paths and expected matrix shapes.
- **Boundary:** A selector matching names does not prove compatible kernels.
- **Prevention:** Review module matching after every model revision.

#### F-013: Too many targets matched

- **Signal:** The adapter count greatly exceeds the planned count.
- **Competing explanation:** Additional heads or feed-forward layers may have been included.
- **Discriminating check:** Compare actual target paths with the architecture map.
- **Bounded response:** Use explicit supported targets if the broad selector is wrong.
- **Verification:** Recompute count from the matched shapes.
- **Boundary:** A broad selector may be intentional for a QLoRA-style run.
- **Prevention:** Version the target list with the checkpoint.

#### F-014: Gradients absent on one factor at initialization

- **Signal:** One LoRA factor has zero gradient on the first backward pass.
- **Competing explanation:** A zero-initialized other factor can legitimately produce this pattern.
- **Discriminating check:** Inspect initialization and compare gradients on subsequent steps.
- **Bounded response:** Avoid declaring failure from one factor alone.
- **Verification:** Check that trainable factors subsequently change and loss is finite.
- **Boundary:** A zero gradient for both factors across batches is different.
- **Prevention:** Include an initialization-aware gradient smoke check.

#### F-015: All labels masked

- **Signal:** Loss is missing or contains no supervised contribution.
- **Competing explanation:** Padding or response-boundary detection can mask every token.
- **Discriminating check:** Count non-ignored labels per sample and per batch.
- **Bounded response:** Fix label construction and token boundary logic.
- **Verification:** Inspect labeled tokens on representative examples.
- **Boundary:** A low supervised-token count can be valid for short responses.
- **Prevention:** Reject or route empty-supervision examples explicitly.

#### F-016: Prompt accidentally supervised

- **Signal:** Training loss decreases but completion style deteriorates.
- **Competing explanation:** The mask may include instruction and system text.
- **Discriminating check:** Render token IDs and supervision mask together.
- **Bounded response:** Mask the intended non-target span consistently.
- **Verification:** Compare supervised token counts with manually annotated samples.
- **Boundary:** Some objectives intentionally supervise the full sequence.
- **Prevention:** Document the objective before treating this as an error.

#### F-017: Padding supervised

- **Signal:** The model learns excessive padding or loss depends on pad length.
- **Competing explanation:** The pad token can also be a valid end-of-sequence token.
- **Discriminating check:** Compare labels at padding positions against attention masks.
- **Bounded response:** Set only actual padding positions to the ignore index.
- **Verification:** Test left and right padding examples with known lengths.
- **Boundary:** Blindly masking all instances of the same token can hide real EOS.
- **Prevention:** Keep length-aware mask tests in the data pipeline.

#### F-018: Truncation removes answers

- **Signal:** The dataset has input tokens but few answer tokens.
- **Competing explanation:** Long prompts may consume the configured context budget.
- **Discriminating check:** Measure answer retention by input-length percentile.
- **Bounded response:** Change truncation policy or filter examples with documented criteria.
- **Verification:** Inspect retained targets for longest examples.
- **Boundary:** Increasing context also changes activation memory.
- **Prevention:** Report truncation rates by evaluation slice.

#### F-019: Packing leaks across examples

- **Signal:** A packed sequence allows attention across unrelated examples.
- **Competing explanation:** Some packers do not enforce isolated attention boundaries.
- **Discriminating check:** Inspect constructed attention masks and position IDs.
- **Bounded response:** Use a packing implementation with the intended isolation semantics.
- **Verification:** Compare packed and unpacked example losses under a controlled test.
- **Boundary:** Cross-example context may be intentional in other objectives.
- **Prevention:** Record packing semantics and test segment boundaries.

#### F-020: Tokenizer drift

- **Signal:** Reloaded adapter produces different token IDs.
- **Competing explanation:** A tokenizer revision or special-token setting changed.
- **Discriminating check:** Hash tokenizer files and compare tokenized fixed examples.
- **Bounded response:** Restore the training tokenizer revision.
- **Verification:** Re-run the fixed-input token and logit checks.
- **Boundary:** Identical text does not imply identical token IDs.
- **Prevention:** Pin tokenizer and chat template alongside the adapter.

#### F-021: Chat template mismatch

- **Signal:** The deployed assistant emits role tags or fails to stop.
- **Competing explanation:** Training and serving may render roles differently.
- **Discriminating check:** Diff rendered token sequences including special tokens.
- **Bounded response:** Use the approved matching template at serving time.
- **Verification:** Test multi-turn and end-of-turn behavior.
- **Boundary:** A template change may be an intentional new model interface.
- **Prevention:** Version templates independently and include them in release gates.

#### F-022: Invalid token IDs

- **Signal:** Embedding lookup fails or inputs map to unintended tokens.
- **Competing explanation:** Special tokens may have been added without resizing embeddings.
- **Discriminating check:** Compare maximum token ID with embedding vocabulary size.
- **Bounded response:** Use compatible tokenizer and embedding weights.
- **Verification:** Run boundary-token lookup and clean reload.
- **Boundary:** Resizing creates new weights whose training policy must be explicit.
- **Prevention:** Validate vocabulary compatibility before training.

#### F-023: Adapter not loaded

- **Signal:** Serving quality resembles the base despite a successful deployment.
- **Competing explanation:** The adapter path may resolve but not be activated.
- **Discriminating check:** Inspect active adapter names and compare logits with adapter enabled and disabled.
- **Bounded response:** Load and explicitly activate the intended adapter.
- **Verification:** Check a fixed input whose logits differ between the two modes.
- **Boundary:** A tiny numerical difference alone does not prove task improvement.
- **Prevention:** Expose artifact identity and active adapter in deployment metadata.

#### F-024: Wrong adapter for tenant

- **Signal:** A response reflects another tenant's domain or policy.
- **Competing explanation:** Cache key collision or missing authorization can select another adapter.
- **Discriminating check:** Trace tenant identity through authorization and adapter lookup.
- **Bounded response:** Fail closed on identity or cache mismatch.
- **Verification:** Run cross-tenant negative tests and audit access.
- **Boundary:** Adapters alone do not isolate data or runtime processes.
- **Prevention:** Bind tenant, base revision and adapter digest in the cache key.

#### F-025: Adapter and base revision mismatch

- **Signal:** Loading fails or yields unexpected outputs after a base upgrade.
- **Competing explanation:** Identical architecture shapes can conceal changed base weights.
- **Discriminating check:** Compare base commit digest in adapter manifest with loaded weights.
- **Bounded response:** Restore the exact base or retrain and revalidate.
- **Verification:** Repeat clean-load and held-out regression checks.
- **Boundary:** Shape compatibility is not behavioral compatibility.
- **Prevention:** Treat the base digest as part of the adapter identity.

#### F-026: Quantization configuration drift

- **Signal:** The same adapter behaves differently across environments.
- **Competing explanation:** NF4, nested scales or compute dtype may differ.
- **Discriminating check:** Compare serialized configs and runtime module types.
- **Bounded response:** Align the deployment configuration with the evaluated one.
- **Verification:** Compare outputs and task metrics using fixed inputs.
- **Boundary:** Kernel nondeterminism can still create small numerical differences.
- **Prevention:** Store quantization settings with evaluation artifacts.

#### F-027: Unexpected BF16 fallback

- **Signal:** Memory or performance differs from the plan.
- **Competing explanation:** The GPU or kernel may not support the requested compute path.
- **Discriminating check:** Inspect runtime dtype, device capability and profiler trace.
- **Bounded response:** Choose a supported dtype or hardware configuration.
- **Verification:** Measure numerical behavior and step throughput again.
- **Boundary:** Changing dtype can affect convergence and kernel availability.
- **Prevention:** Maintain a tested hardware and software compatibility matrix.

#### F-028: Unexpected FP32 trainables

- **Signal:** Adapter memory is larger than the two-byte estimate.
- **Competing explanation:** Preparation or framework policies may retain selected tensors in FP32.
- **Discriminating check:** Print trainable parameter and gradient dtypes.
- **Bounded response:** Recalculate the budget using actual widths.
- **Verification:** Compare estimated state bytes with allocator observations.
- **Boundary:** Casting without stability testing may cause numerical regressions.
- **Prevention:** Generate a dtype inventory for each training run.

#### F-029: Unquantized output head

- **Signal:** Base memory exceeds a simple four-bit payload estimate.
- **Competing explanation:** Embeddings, output heads or norms may remain at higher precision.
- **Discriminating check:** Inventory base tensors by dtype, size and device.
- **Bounded response:** Account for each unquantized tensor in the budget.
- **Verification:** Reconcile inventory totals against model-load memory.
- **Boundary:** Packed wrappers may hide data and metadata from naive tensor counts.
- **Prevention:** Use framework-aware inspection rather than assuming all weights are four-bit.

#### F-030: Padding in packed weights

- **Signal:** A theoretical code count undershoots actual storage.
- **Competing explanation:** Block alignment, scale tensors and wrapper buffers add bytes.
- **Discriminating check:** Compare serialized and in-memory layer storage to code-only estimates.
- **Bounded response:** Reserve explicit metadata and padding allowances.
- **Verification:** Measure the largest representative layers and whole model.
- **Boundary:** On-disk compression and runtime allocation are different quantities.
- **Prevention:** Keep a metadata category separate from the base payload.

#### F-031: Double quantization misunderstood

- **Signal:** A budget assumes scale metadata is zero.
- **Competing explanation:** Nested quantization still needs codes and higher-level constants.
- **Discriminating check:** Inspect quantization state tensors in the installed implementation.
- **Bounded response:** Include all nested metadata and alignment in the estimate.
- **Verification:** Compare nested and non-nested footprints on the same model.
- **Boundary:** Savings vary with block size and implementation.
- **Prevention:** Document all quantization-state tensors and their ownership.

#### F-032: Page faults dominate steps

- **Signal:** GPU compute is intermittently idle while host transfers spike.
- **Competing explanation:** Paged optimizer state can migrate under memory pressure.
- **Discriminating check:** Profile page migration and correlate it with per-step latency.
- **Bounded response:** Reduce pressure or use a larger device or different optimizer.
- **Verification:** Compare throughput and peak memory on matched runs.
- **Boundary:** A paged optimizer does not guarantee migration in every step.
- **Prevention:** Alert on transfer volume and step-time distribution.

#### F-033: Host memory exhausted

- **Signal:** Device usage looks acceptable but the worker is killed.
- **Competing explanation:** Paged state, dataset caches or pinned buffers may consume host RAM.
- **Discriminating check:** Monitor resident host memory and operating-system kill events.
- **Bounded response:** Right-size host capacity and bound caches and loader workers.
- **Verification:** Repeat the longest workload with peak host memory recorded.
- **Boundary:** A container memory limit can be lower than physical host RAM.
- **Prevention:** Set host-memory limits and alerts for training workers.

#### F-034: PCIe contention

- **Signal:** Training slows when other jobs share the host interconnect.
- **Competing explanation:** Paging and input transfers can compete for bandwidth.
- **Discriminating check:** Profile transfers with and without concurrent jobs.
- **Bounded response:** Schedule isolation or reduce transfer volume.
- **Verification:** Compare step latency distributions under the intended concurrency.
- **Boundary:** Bandwidth results are platform-specific and not universal constants.
- **Prevention:** Include host topology and co-tenancy in capacity planning.

#### F-035: First-step memory spike

- **Signal:** The model loads but fails when the optimizer first updates.
- **Competing explanation:** Optimizer moments may be allocated lazily.
- **Discriminating check:** Record memory immediately before and after the first step.
- **Bounded response:** Allocate sufficient headroom or change the optimizer budget.
- **Verification:** Execute multiple complete steps including checkpoint save.
- **Boundary:** Other temporary allocations can also peak at the first step.
- **Prevention:** Make a complete optimizer step part of every fit check.

#### F-036: Save-time memory spike

- **Signal:** Training steps fit but checkpoint creation fails.
- **Competing explanation:** Serialization can materialize copies or gather distributed tensors.
- **Discriminating check:** Observe host and device peaks during an actual save.
- **Bounded response:** Use a supported lower-memory save path or add capacity.
- **Verification:** Restore the resulting artifact in a fresh process.
- **Boundary:** An adapter-only checkpoint may omit optimizer and RNG state.
- **Prevention:** Test both deployable artifacts and resumable checkpoints.

#### F-037: Evaluation OOM

- **Signal:** Training fits but generation fails during evaluation.
- **Competing explanation:** Generation cache, beam count or batch size may dominate.
- **Discriminating check:** Measure cache and peak memory while varying output length and beams.
- **Bounded response:** Limit evaluation batch or output budget without changing the metric silently.
- **Verification:** Run the full evaluation protocol at the chosen settings.
- **Boundary:** Disabling cache can change performance characteristics.
- **Prevention:** Budget evaluation and training as separate workloads.

#### F-038: Fragmented allocator

- **Signal:** Allocation fails despite apparently sufficient free bytes.
- **Competing explanation:** Reserved fragments or external consumers may prevent a large allocation.
- **Discriminating check:** Inspect allocated, reserved and device-global free memory.
- **Bounded response:** Restart an isolated worker or reduce peak allocation sizes.
- **Verification:** Repeat the same workload and inspect allocation history.
- **Boundary:** A restart is a mitigation, not proof of the original cause.
- **Prevention:** Record memory snapshots before retrying failed jobs.

#### F-039: Gradient accumulation misunderstood

- **Signal:** A nominally large batch OOMs or gives unexpected scaling.
- **Competing explanation:** Accumulation does not shrink a microbatch automatically.
- **Discriminating check:** Inspect per-step microbatch size, accumulation count and loss reduction.
- **Bounded response:** Reduce microbatch and set accumulation deliberately.
- **Verification:** Compare token-normalized gradients against a small reference.
- **Boundary:** Unequal token counts can make averaging per microbatch biased.
- **Prevention:** Document effective batch in supervised tokens, not only examples.

#### F-040: Long-context blowup

- **Signal:** Peak memory rises sharply on long sequences.
- **Competing explanation:** Attention and saved activations depend on kernel and sequence length.
- **Discriminating check:** Sweep actual length and record peak and throughput.
- **Bounded response:** Use compatible efficient attention or shorten context based on requirements.
- **Verification:** Measure quality on the long-context slice after the change.
- **Boundary:** A short-sequence fit does not extrapolate reliably.
- **Prevention:** Include worst permitted sequence lengths in fit tests.

#### F-041: Checkpointing slows training

- **Signal:** Memory drops but per-step time increases.
- **Competing explanation:** Recomputed forward operations add work.
- **Discriminating check:** Compare profiler traces with and without activation checkpointing.
- **Bounded response:** Choose the memory-throughput tradeoff appropriate to the run.
- **Verification:** Report peak, tokens per second and final task metrics.
- **Boundary:** Checkpointing may interact with dropout and reproducibility settings.
- **Prevention:** Keep a documented reference run for the chosen configuration.

#### F-042: Dropout during comparison

- **Signal:** A/B logits vary even with identical artifacts.
- **Competing explanation:** Training mode enables dropout while inference mode disables it.
- **Discriminating check:** Inspect model mode and random seed during comparison.
- **Bounded response:** Use evaluation mode for deterministic artifact checks where supported.
- **Verification:** Repeat multiple passes and compare expected variability.
- **Boundary:** Some kernels remain nondeterministic even in evaluation mode.
- **Prevention:** Record execution mode and determinism settings.

#### F-043: Optimizer moments missing on resume

- **Signal:** Loss jumps after restart despite loaded adapter weights.
- **Competing explanation:** A deployment adapter save may not contain optimizer state.
- **Discriminating check:** Inspect the checkpoint manifest for optimizer, scheduler and step.
- **Bounded response:** Resume from a complete training checkpoint.
- **Verification:** Compare next-step behavior with an uninterrupted control.
- **Boundary:** Random and data-loader states are also needed for closer replay.
- **Prevention:** Separate deployment and training checkpoint policies.

#### F-044: Scheduler reset on resume

- **Signal:** The learning rate returns to its initial value after restart.
- **Competing explanation:** The scheduler state or global step may not have been restored.
- **Discriminating check:** Log learning rate immediately before and after resumption.
- **Bounded response:** Restore scheduler state and update count.
- **Verification:** Compare the resumed learning-rate trace with the planned schedule.
- **Boundary:** Distributed or skipped steps may complicate step counting.
- **Prevention:** Save global step and scheduler state atomically.

#### F-045: Random state lost

- **Signal:** A resumed run follows a different data order or dropout path.
- **Competing explanation:** RNG and sampler state may be omitted from checkpoints.
- **Discriminating check:** Compare saved RNG states and sample identifiers across restart.
- **Bounded response:** Restore relevant Python, framework, accelerator and sampler states.
- **Verification:** Run a controlled interruption and continuation drill.
- **Boundary:** Exact reproducibility is not guaranteed across kernels or hardware.
- **Prevention:** Document determinism limitations with each run.

#### F-046: Training data duplicates test data

- **Signal:** Held-out score is unexpectedly high.
- **Competing explanation:** Conversation fragments or templates can span splits.
- **Discriminating check:** Hash normalized content and audit near-duplicates by entity.
- **Bounded response:** Rebuild splits before tuning using the proper grouping unit.
- **Verification:** Evaluate on a newly untouched test set.
- **Boundary:** Perfect deduplication is difficult and semantic overlap can remain.
- **Prevention:** Track data lineage and split assignment at ingestion.

#### F-047: Temporal leakage

- **Signal:** Evaluation appears strong on past examples but fails on new events.
- **Competing explanation:** Future labels or documents entered training or retrieval.
- **Discriminating check:** Audit timestamps and feature availability relative to prediction time.
- **Bounded response:** Rebuild a forward-in-time evaluation set.
- **Verification:** Compare performance on a subsequent untouched window.
- **Boundary:** Time ordering alone does not prevent entity-level duplication.
- **Prevention:** Record data extraction and availability timestamps.

#### F-048: Contaminated model selection

- **Signal:** Repeated experiments improve a supposedly final holdout.
- **Competing explanation:** The holdout has become an implicit tuning set.
- **Discriminating check:** Audit evaluation accesses and selection decisions.
- **Bounded response:** Reserve a new final test and use validation for selection.
- **Verification:** Report number of comparisons and final locked evaluation.
- **Boundary:** A fresh test can still be nonrepresentative.
- **Prevention:** Enforce access controls and a test-use policy.

#### F-049: Biased slice aggregate

- **Signal:** Overall quality rises while a critical subgroup degrades.
- **Competing explanation:** Large slices can hide small but consequential groups.
- **Discriminating check:** Compute per-slice paired outcomes and denominators.
- **Bounded response:** Investigate and remediate material regressions before release.
- **Verification:** Report uncertainty and small-sample limitations for each slice.
- **Boundary:** Multiple subgroup comparisons increase false-alarm risk.
- **Prevention:** Define critical slices and escalation limits in advance.

#### F-050: Overstated uncertainty

- **Signal:** Confidence intervals treat every message as independent.
- **Competing explanation:** Messages from one conversation may be correlated.
- **Discriminating check:** Compare row-level and conversation-clustered resampling.
- **Bounded response:** Use the independent deployment unit for uncertainty.
- **Verification:** Report sample counts at both row and cluster levels.
- **Boundary:** A small number of clusters may still make estimates unstable.
- **Prevention:** Choose the sampling unit during evaluation design.

#### F-051: Judge model bias

- **Signal:** Automated grading favors a particular style or length.
- **Competing explanation:** The judge may share biases with the model under test.
- **Discriminating check:** Blind and randomize paired outputs for independent human review.
- **Bounded response:** Calibrate automated ratings against a relevant expert-labeled set.
- **Verification:** Report disagreement and error categories.
- **Boundary:** Human labels also require training and quality control.
- **Prevention:** Version grading prompts and adjudication rules.

#### F-052: Format compliance hides factual errors

- **Signal:** Outputs match the schema but contain unsupported claims.
- **Competing explanation:** Syntactic validation does not test factual grounding.
- **Discriminating check:** Sample valid outputs and audit their evidentiary support.
- **Bounded response:** Add groundedness checks and human review where needed.
- **Verification:** Measure schema validity and correctness separately.
- **Boundary:** Automated grounding checks can themselves be wrong.
- **Prevention:** Define task-specific quality criteria beyond format.

#### F-053: Unsafe answer hidden by averaging

- **Signal:** A small number of high-impact outputs fail policy.
- **Competing explanation:** Mean accuracy may conceal severity and exposure.
- **Discriminating check:** Review severity-weighted failures and high-risk slices.
- **Bounded response:** Block release or add controls according to the safety policy.
- **Verification:** Re-run adversarial and representative scenarios after changes.
- **Boundary:** No finite test suite proves universal safety.
- **Prevention:** Set independent high-severity release gates.

#### F-054: Unstable training loss

- **Signal:** Loss oscillates or becomes nonfinite.
- **Competing explanation:** Learning rate, data corruption and precision are competing causes.
- **Discriminating check:** Locate the first nonfinite tensor and compare a small stable reference.
- **Bounded response:** Correct the demonstrated cause rather than only suppressing the error.
- **Verification:** Track finite gradients and held-out loss after the fix.
- **Boundary:** A smooth training loss does not prove generalization.
- **Prevention:** Log gradient norms, learning rate and data identifiers.

#### F-055: Gradient overflow

- **Signal:** Nonfinite gradients occur in a narrow range of batches.
- **Competing explanation:** Large activations, loss scaling or numerical kernels may be responsible.
- **Discriminating check:** Capture tensor statistics before the first invalid operation.
- **Bounded response:** Adjust the supported precision or loss-scaling configuration.
- **Verification:** Replay the failing batch and compare a higher-precision reference.
- **Boundary:** A higher-precision result is a diagnostic, not an accuracy guarantee.
- **Prevention:** Set nonfinite alerts and retain failing batch identifiers.

#### F-056: Gradient underflow

- **Signal:** Small updates vanish in a low-precision path.
- **Competing explanation:** Small scale, dtype range and optimizer arithmetic may interact.
- **Discriminating check:** Compare gradient histograms with a supported higher-precision run.
- **Bounded response:** Use an appropriate dtype or scaling strategy after confirming the cause.
- **Verification:** Verify nonzero updates and task quality on a controlled subset.
- **Boundary:** A zero gradient may be mathematically correct for some parameters.
- **Prevention:** Monitor gradient distribution by adapter factor.

#### F-057: Wrong loss normalization

- **Signal:** Training changes when equivalent batches are partitioned.
- **Competing explanation:** Mean-of-means can overweight short microbatches.
- **Discriminating check:** Compare accumulated gradients against a single token-weighted reference.
- **Bounded response:** Normalize by total supervised tokens for the intended objective.
- **Verification:** Check gradient equivalence under two microbatch partitions.
- **Boundary:** Some objectives intentionally weight examples equally.
- **Prevention:** Specify whether examples or tokens receive equal weight.

#### F-058: Incorrect accumulation boundary

- **Signal:** Updates occur too often or never occur.
- **Competing explanation:** Optimizer zeroing and step order may be wrong.
- **Discriminating check:** Log optimizer-step count and parameter deltas over microbatches.
- **Bounded response:** Correct zero-grad, backward and optimizer-step ordering.
- **Verification:** Compare a tiny deterministic run with a reference update.
- **Boundary:** Gradient clipping must be placed consistently with accumulation.
- **Prevention:** Add a step-count invariant to the training loop.

#### F-059: Learning-rate scheduler mismatch

- **Signal:** Run warms up or decays at the wrong pace.
- **Competing explanation:** Scheduler may count microbatches rather than optimizer updates.
- **Discriminating check:** Plot scheduler step against actual optimizer-step count.
- **Bounded response:** Use the intended unit and restore state on resume.
- **Verification:** Check logged learning rates at selected update indices.
- **Boundary:** Skipped updates can complicate scheduler semantics.
- **Prevention:** Document the scheduling unit in the run manifest.

#### F-060: Quantization error localized in a layer

- **Signal:** Quantized base quality drops before adapters are trained.
- **Competing explanation:** A sensitive layer or block may have poor reconstruction.
- **Discriminating check:** Compare per-layer output differences on representative activations.
- **Bounded response:** Try a supported higher-precision or selective configuration.
- **Verification:** Re-evaluate both memory and task metrics.
- **Boundary:** A large local difference need not cause a large task failure.
- **Prevention:** Track layer sensitivity under representative inputs.

#### F-061: Adapter rank too small

- **Signal:** Training plateaus above an acceptable error level.
- **Competing explanation:** Rank, target coverage, data and optimizer are confounded.
- **Discriminating check:** Run a controlled rank sweep with fixed targets and budget reporting.
- **Bounded response:** Increase rank only when evidence supports the tradeoff.
- **Verification:** Compare paired validation outcomes and memory across seeds.
- **Boundary:** Higher rank does not guarantee better generalization.
- **Prevention:** Set a predefined sweep and stopping criterion.

#### F-062: Rank increase causes OOM

- **Signal:** A rank sweep crosses a capacity boundary.
- **Competing explanation:** Adapter weights, gradients and states scale with rank.
- **Discriminating check:** Recompute E3 and measure optimizer-step peaks per candidate.
- **Bounded response:** Reduce rank or target fewer modules when quality allows.
- **Verification:** Check that the revised run completes and meets quality gates.
- **Boundary:** Activation and workspace terms need not scale linearly with rank.
- **Prevention:** Budget the largest candidate before launching the sweep.

#### F-063: Uncovered sensitive modules

- **Signal:** Quality remains poor although rank was increased.
- **Competing explanation:** The affected projection may not have an adapter.
- **Discriminating check:** Inspect matched target paths and layer-level ablations.
- **Bounded response:** Expand coverage to supported modules with an explicit memory estimate.
- **Verification:** Compare quality and footprint with the previous target set.
- **Boundary:** Changing targets changes more than parameter count alone.
- **Prevention:** Log and review target-module maps.

#### F-064: Unexpected output-head training

- **Signal:** A supposedly adapter-only artifact is much larger than expected.
- **Competing explanation:** The output head may be included through modules-to-save or bias settings.
- **Discriminating check:** List all trainable parameters and serialized keys.
- **Bounded response:** Freeze or intentionally include the head according to requirements.
- **Verification:** Reload and verify both artifact size and trainable manifest.
- **Boundary:** Freezing a newly added vocabulary head may make the task infeasible.
- **Prevention:** Declare every non-adapter trainable component.

#### F-065: Bias policy drift

- **Signal:** Training differs from the documented adapter-only configuration.
- **Competing explanation:** A bias option may have been enabled.
- **Discriminating check:** Inspect bias requires-grad flags and optimizer membership.
- **Bounded response:** Set the intended bias policy explicitly.
- **Verification:** Compare parameter counts and clean-reload predictions.
- **Boundary:** Some tasks intentionally train biases.
- **Prevention:** Include bias policy in the adapter manifest.

#### F-066: Tied weights counted twice

- **Signal:** The budget overstates base or optimizer storage.
- **Competing explanation:** Embedding and output tensors can share storage.
- **Discriminating check:** Inspect storage identity and tied-weight configuration.
- **Bounded response:** Count unique allocations while retaining all logical uses.
- **Verification:** Reconcile estimated bytes with actual allocation.
- **Boundary:** Framework conversions can break or recreate ties.
- **Prevention:** Document weight tying after model loading.

#### F-067: Shared storage missed

- **Signal:** A checkpoint round trip changes weight tying.
- **Competing explanation:** Serialization or resizing can create independent copies.
- **Discriminating check:** Check storage identity before save and after reload.
- **Bounded response:** Use a supported save and restore procedure.
- **Verification:** Repeat memory and functional comparisons after reload.
- **Boundary:** Equal numerical values do not prove shared storage.
- **Prevention:** Test tie invariants during artifact validation.

#### F-068: Wrong compute device

- **Signal:** A module unexpectedly runs on CPU and throughput collapses.
- **Competing explanation:** Automatic placement or an unsupported kernel may move layers.
- **Discriminating check:** Inspect device placement for each model module and input.
- **Bounded response:** Choose an explicitly supported placement plan.
- **Verification:** Profile execution and transfer time on the intended topology.
- **Boundary:** Automatic device maps designed for inference may not be suitable for training.
- **Prevention:** Record device placement and supported distributed strategy.

#### F-069: Distributed state mismatch

- **Signal:** Workers disagree on updates or fail during synchronization.
- **Competing explanation:** Adapter wrapping or optimizer construction order can differ by rank.
- **Discriminating check:** Compare trainable names and shapes across workers.
- **Bounded response:** Use a framework-supported wrapping and optimizer sequence.
- **Verification:** Check synchronized adapter parameters after an update.
- **Boundary:** Communication costs depend on the parallelism strategy.
- **Prevention:** Run multi-worker correctness tests before scaling.

#### F-070: Distributed checkpoint incomplete

- **Signal:** A saved adapter misses parameters from some ranks.
- **Competing explanation:** Sharded state may not have been gathered or saved correctly.
- **Discriminating check:** Inspect state-dict keys and per-rank checkpoint protocol.
- **Bounded response:** Use the distributed framework's supported save path.
- **Verification:** Reload in a clean single-process or supported distributed environment.
- **Boundary:** A successful write does not prove a complete artifact.
- **Prevention:** Exercise save and restore under the actual topology.

#### F-071: Distributed imbalance

- **Signal:** One worker repeatedly waits for others.
- **Competing explanation:** Sequence-length skew or data-loading imbalance may dominate.
- **Discriminating check:** Compare per-rank batch lengths, compute time and input wait.
- **Bounded response:** Balance lengths or partition data more evenly.
- **Verification:** Measure end-to-end throughput and slice coverage after rebalancing.
- **Boundary:** Aggressive length grouping can alter the training sequence.
- **Prevention:** Record batch construction and per-rank timing.

#### F-072: Wrong sequence attention mask

- **Signal:** Model attends to padding or ignores valid tokens.
- **Competing explanation:** Mask polarity or shape can vary by framework.
- **Discriminating check:** Inspect the actual mask for small left- and right-padded batches.
- **Bounded response:** Use the documented mask convention for the model.
- **Verification:** Compare batched and single-example logits at valid positions.
- **Boundary:** Position IDs and cache state can also affect the comparison.
- **Prevention:** Add model-specific padding equivalence tests.

#### F-073: Missing EOS supervision

- **Signal:** Generated responses continue until the maximum token limit.
- **Competing explanation:** EOS may be truncated or masked out of labels.
- **Discriminating check:** Inspect end-of-answer labels and template rendering.
- **Bounded response:** Retain and supervise the intended stopping token.
- **Verification:** Measure stop reasons and completion lengths on held-out prompts.
- **Boundary:** Some applications intentionally use alternative stop sequences.
- **Prevention:** Version stop tokens and generation configuration together.

#### F-074: Output budget mismatch

- **Signal:** A model appears less accurate because answers are cut off.
- **Competing explanation:** Evaluation maximum tokens may be shorter than the required response.
- **Discriminating check:** Inspect completion stop reasons and truncated examples.
- **Bounded response:** Set a task-appropriate common output budget.
- **Verification:** Re-score all compared models under identical settings.
- **Boundary:** A longer budget changes cost and can introduce additional errors.
- **Prevention:** Report output limits with every metric.

#### F-075: Sampling mismatch

- **Signal:** Two variants show different scores from different decoding parameters.
- **Competing explanation:** Temperature, seed or top-p can change outputs.
- **Discriminating check:** Compare full generation configs and randomization policies.
- **Bounded response:** Align decoding or evaluate the intended production settings explicitly.
- **Verification:** Run paired comparisons over documented seeds.
- **Boundary:** Deterministic decoding may not reflect production traffic.
- **Prevention:** Treat decoding as part of the evaluated system.

#### F-076: Requantized merge error

- **Signal:** Merged deployment quality is worse than the separate adapter.
- **Competing explanation:** Merging into a quantized base can require another lossy quantization.
- **Discriminating check:** Compare logits and task scores for separate and merged artifacts.
- **Bounded response:** Serve separately or use a supported merge workflow that passes gates.
- **Verification:** Test both quality and peak serving memory after the change.
- **Boundary:** Merge support depends on quantizer and library version.
- **Prevention:** Record the full conversion chain and artifact hashes.

#### F-077: Cache key omits quantization

- **Signal:** A cached base is reused with the wrong adapter runtime.
- **Competing explanation:** Two quantization configurations can share the same model name.
- **Discriminating check:** Compare cache keys against full model and quantization manifests.
- **Bounded response:** Include base digest, quantizer settings and runtime version in cache identity.
- **Verification:** Exercise cache hit and miss paths with conflicting configs.
- **Boundary:** More specific keys increase cache cardinality and memory needs.
- **Prevention:** Invalidate old entries on format or runtime upgrades.

#### F-078: Version incompatibility

- **Signal:** Loading raises an exception or selects a different implementation.
- **Competing explanation:** PEFT, Transformers and bitsandbytes integration may have changed.
- **Discriminating check:** Inspect exact package versions, hardware and the exception location.
- **Bounded response:** Use a tested compatible dependency set.
- **Verification:** Run a clean environment load, step and reload test.
- **Boundary:** The cited PEFT page is versioned but its dependencies are not all pinned by this sheet.
- **Prevention:** Lock tested versions and record driver and device details.

#### F-079: Unsupported hardware path

- **Signal:** A four-bit kernel fails or falls back unexpectedly.
- **Competing explanation:** The installed backend may not support the device or dtype.
- **Discriminating check:** Check official backend compatibility for the installed versions.
- **Bounded response:** Use a supported device or alternative method.
- **Verification:** Repeat numerical and performance checks on the selected platform.
- **Boundary:** Availability on one accelerator does not imply portability to another.
- **Prevention:** Maintain an explicit device support matrix.

#### F-080: Network-dependent checkpoint

- **Signal:** A production worker fails when the model registry is unreachable.
- **Competing explanation:** A deployment may resolve a mutable remote reference at startup.
- **Discriminating check:** Audit artifact fetches and revision pinning.
- **Bounded response:** Stage signed immutable artifacts in an approved local store.
- **Verification:** Perform an offline startup and rollback drill.
- **Boundary:** Offline storage still requires access control and integrity checks.
- **Prevention:** Treat dependencies as release artifacts rather than runtime guesses.

#### F-081: Incomplete adapter artifact

- **Signal:** A checkpoint loads but omits extra trainable modules.
- **Competing explanation:** Saving only low-rank factors may omit deliberately trained heads.
- **Discriminating check:** Compare serialized keys against the full trainable manifest.
- **Bounded response:** Use a supported save configuration that includes required modules.
- **Verification:** Reload and compare fixed-input outputs in a fresh process.
- **Boundary:** A training resume checkpoint needs more state than an inference artifact.
- **Prevention:** Validate artifact completeness before promotion.

#### F-082: Unsafe training corpus

- **Signal:** The adapter memorizes sensitive text or receives unauthorized data.
- **Competing explanation:** Data selection and retention controls may be inadequate.
- **Discriminating check:** Audit dataset lineage, permissions and representative memorization probes.
- **Bounded response:** Remove unauthorized data and retrain under approved controls.
- **Verification:** Repeat privacy review and access tests.
- **Boundary:** A negative probe does not establish absence of memorization.
- **Prevention:** Apply data minimization and retention controls before training.

#### F-083: Prompt injection in training data

- **Signal:** The model learns to follow malicious instructions embedded in examples.
- **Competing explanation:** Untrusted documents may be mislabeled as authoritative instructions.
- **Discriminating check:** Review provenance and examples crossing trust boundaries.
- **Bounded response:** Filter or relabel data according to the task and threat model.
- **Verification:** Test held-out adversarial instructions after retraining.
- **Boundary:** Fine-tuning alone does not enforce runtime permissions.
- **Prevention:** Separate instruction roles and authorize every external action.

#### F-084: Unexpected tool execution

- **Signal:** A tuned assistant requests a dangerous operation.
- **Competing explanation:** The generated text may be treated as an authorization decision.
- **Discriminating check:** Trace tool calls through the permission and approval boundary.
- **Bounded response:** Enforce independent tool authorization and approval.
- **Verification:** Run denied-action and replay tests.
- **Boundary:** A high safety score is not an access-control mechanism.
- **Prevention:** Use least privilege and explicit effect controls.

#### F-085: Non-idempotent tool retry

- **Signal:** A transient failure causes a repeated external action.
- **Competing explanation:** The tool interface may lack an idempotency key.
- **Discriminating check:** Trace request IDs and external side effects across retry paths.
- **Bounded response:** Use idempotent operations or an explicit deduplication contract.
- **Verification:** Simulate timeout after the side effect and verify one effective action.
- **Boundary:** The model cannot guarantee exactly-once execution by itself.
- **Prevention:** Design retry and reconciliation semantics outside the model.

#### F-086: Data distribution drift

- **Signal:** Quality drops after deployment despite passing the original test.
- **Competing explanation:** New vocabulary, workflows or source populations may shift inputs.
- **Discriminating check:** Compare privacy-safe current input slices with the evaluation distribution.
- **Bounded response:** Collect approved labels and reassess or retrain.
- **Verification:** Evaluate the candidate on both new and old slices.
- **Boundary:** Distribution shift metrics alone do not reveal task correctness.
- **Prevention:** Set monitoring and periodic review triggers.

#### F-087: Latency regression

- **Signal:** Adapter serving violates the response-time objective.
- **Competing explanation:** Kernel overhead, cache misses or longer output can be responsible.
- **Discriminating check:** Profile end-to-end latency with actual concurrency and token lengths.
- **Bounded response:** Optimize the measured path or choose a different serving strategy.
- **Verification:** Repeat load tests and task evaluation after the change.
- **Boundary:** Mean latency does not establish a percentile SLO.
- **Prevention:** Monitor prompt and output lengths and relevant latency percentiles.

#### F-088: Throughput claim from memory

- **Signal:** A smaller checkpoint is assumed to run faster.
- **Competing explanation:** Dequantization overhead and hardware utilization may offset storage savings.
- **Discriminating check:** Benchmark tokens per second and latency on the target hardware.
- **Bounded response:** Choose based on the measured cost and quality tradeoff.
- **Verification:** Repeat with warm caches and representative batches.
- **Boundary:** One synthetic batch does not characterize production traffic.
- **Prevention:** Maintain a representative versioned workload suite.

#### F-089: Capacity estimate ignores output

- **Signal:** A long-generation endpoint exceeds serving memory.
- **Competing explanation:** KV cache grows with context and generated tokens.
- **Discriminating check:** Measure serving peak across input and output length grids.
- **Bounded response:** Limit concurrency or output length based on observed capacity.
- **Verification:** Load-test under worst supported admission conditions.
- **Boundary:** KV cache is a serving cost distinct from training optimizer state.
- **Prevention:** Set token and concurrency budgets at the gateway.

#### F-090: Cost accounting omits retries

- **Signal:** A cheaper GPU appears economical per training hour.
- **Competing explanation:** Failed runs and slow paging can raise cost per successful artifact.
- **Discriminating check:** Track all attempts, wall time, storage and human review.
- **Bounded response:** Compare total costs for a defined successful outcome.
- **Verification:** Review cost and quality together over multiple runs.
- **Boundary:** Illustrative hourly prices are not actual procurement quotes.
- **Prevention:** Record currency, date and included charges for each comparison.

#### F-091: Rollback artifact unavailable

- **Signal:** A newly promoted adapter fails and the old one cannot be restored.
- **Competing explanation:** The previous base or tokenizer revision may have been garbage-collected.
- **Discriminating check:** Check availability and hashes for the complete previous bundle.
- **Bounded response:** Retain and stage the prior compatible bundle.
- **Verification:** Perform a full rollback and fixed-input check.
- **Boundary:** Rollback may require separate data or schema compatibility.
- **Prevention:** Define retention and rollback ownership before release.

#### F-092: Adapter swap during session

- **Signal:** A multi-turn session changes behavior halfway through.
- **Competing explanation:** Cache refresh or routing may switch adapters mid-conversation.
- **Discriminating check:** Trace adapter identity for each turn in a session.
- **Bounded response:** Pin an authorized artifact version for the session.
- **Verification:** Test concurrent rollout and active-session continuity.
- **Boundary:** Session pinning does not solve cross-tenant authorization.
- **Prevention:** Include adapter identity in session state and audit logs.

#### F-093: Model registry tag moves

- **Signal:** A reproducible run cannot be recreated from a mutable alias.
- **Competing explanation:** A model tag can later point to another revision.
- **Discriminating check:** Resolve and store immutable checkpoint and dataset digests.
- **Bounded response:** Pin revisions for training, evaluation and deployment.
- **Verification:** Reload the exact manifest in a clean environment.
- **Boundary:** External sources may become unavailable despite a recorded digest.
- **Prevention:** Retain approved copies according to licensing and policy.

#### F-094: Kernel upgrade changes outputs

- **Signal:** A dependency update alters numerical behavior or memory.
- **Competing explanation:** Kernel implementation, precision or ordering may have changed.
- **Discriminating check:** Run fixed-input and representative regression suites before upgrade.
- **Bounded response:** Pin or roll back the dependency while investigating.
- **Verification:** Compare task metrics, peaks and throughput on the candidate stack.
- **Boundary:** Bitwise equality may be unrealistic for all parallel kernels.
- **Prevention:** Use versioned tolerances and behavioral release gates.

#### F-095: Quantizer conversion mismatch

- **Signal:** Offline weights and runtime decoder interpret metadata differently.
- **Competing explanation:** A serialization or library-version mismatch may corrupt scale interpretation.
- **Discriminating check:** Compare a small layer's decoded values between save and reload.
- **Bounded response:** Use a documented compatible serialization path.
- **Verification:** Run layer-level and end-to-end reload comparisons.
- **Boundary:** Matching a few values does not prove every tensor is correct.
- **Prevention:** Round-trip representative modules and a full model.

#### F-096: Zero scale edge case

- **Signal:** A quantization block containing all zeros produces invalid values.
- **Competing explanation:** An implementation may need a defined zero-scale handling rule.
- **Discriminating check:** Quantize and decode an all-zero block with the installed library.
- **Bounded response:** Use a supported implementation with correct zero-block behavior.
- **Verification:** Verify finite decoded values and gradients on the edge case.
- **Boundary:** Do not invent implementation semantics from the abstract NF4 description.
- **Prevention:** Keep zero-block tests in backend qualification.

#### F-097: Outlier-heavy block

- **Signal:** A few extreme values consume much of the block's dynamic range.
- **Competing explanation:** Block scaling may reduce effective resolution for typical entries.
- **Discriminating check:** Compare reconstruction errors by magnitude and block.
- **Bounded response:** Evaluate a supported alternative quantizer or precision for sensitive modules.
- **Verification:** Measure task quality and memory after the change.
- **Boundary:** Outlier statistics from one layer may not represent the whole network.
- **Prevention:** Profile reconstruction error on representative checkpoints.

#### F-098: Non-normal weight distribution

- **Signal:** NF4's distributional motivation may not match a particular tensor.
- **Competing explanation:** Block normalization and layer distributions can differ from the ideal.
- **Discriminating check:** Inspect empirical weight and normalized-block distributions.
- **Bounded response:** Compare quantizers under the same evaluation and memory protocol.
- **Verification:** Report reconstruction and task-level outcomes separately.
- **Boundary:** A distribution test does not determine downstream quality.
- **Prevention:** Revisit the format for substantially changed architectures.

#### F-099: Scale metadata corruption

- **Signal:** Decoded outputs contain extreme or nonfinite values.
- **Competing explanation:** Incorrect scale serialization or memory corruption is possible.
- **Discriminating check:** Compare scale tensor hashes and finite ranges across save and reload.
- **Bounded response:** Restore a verified artifact and investigate the serialization path.
- **Verification:** Repeat fixed-input and layer decode comparisons.
- **Boundary:** A plausible range alone does not prove metadata integrity.
- **Prevention:** Use artifact hashes and controlled format migrations.

#### F-100: Loss does not correlate with utility

- **Signal:** Validation loss improves but user-facing task outcomes decline.
- **Competing explanation:** The optimization objective may not match the operational decision.
- **Discriminating check:** Evaluate task success, safety and operational costs separately.
- **Bounded response:** Choose checkpoints using predefined task-aligned gates.
- **Verification:** Use a locked holdout to confirm the selected checkpoint.
- **Boundary:** Task metrics may be sparse or expensive to label.
- **Prevention:** Maintain a metric hierarchy with clear escalation rules.

#### F-101: Overfitting adapters

- **Signal:** Training loss decreases while held-out performance degrades.
- **Competing explanation:** Small data, leakage or excessive optimization may be involved.
- **Discriminating check:** Plot train and validation metrics by update and inspect split integrity.
- **Bounded response:** Select a validated checkpoint or change data and regularization.
- **Verification:** Confirm on a separate final test.
- **Boundary:** A single noisy validation point does not establish overfitting.
- **Prevention:** Predefine evaluation cadence and model-selection rules.

#### F-102: No improvement over prompt baseline

- **Signal:** Training costs are incurred without a measurable task gain.
- **Competing explanation:** Data quality, target coverage or objective mismatch may be responsible.
- **Discriminating check:** Compare matched prompt-only and adapted systems on paired examples.
- **Bounded response:** Improve the demonstrated bottleneck or retain the simpler baseline.
- **Verification:** Estimate uncertainty and compare operational costs.
- **Boundary:** An inconclusive result is not proof of equivalence.
- **Prevention:** Set a minimum practical improvement before the experiment.

#### F-103: Synthetic labels reinforce errors

- **Signal:** Adapter reproduces inaccurate teacher outputs.
- **Competing explanation:** The label-generation process may not include independent verification.
- **Discriminating check:** Audit a representative sample against independent evidence.
- **Bounded response:** Correct or exclude unreliable examples and rerun training.
- **Verification:** Evaluate on independently labeled held-out data.
- **Boundary:** A second model judge may share the teacher's failure modes.
- **Prevention:** Record label provenance and adjudication procedures.

#### F-104: License or consent violation

- **Signal:** A technically successful run cannot be legally deployed.
- **Competing explanation:** Model or dataset terms may prohibit the intended use.
- **Discriminating check:** Inspect source terms and approvals for every artifact.
- **Bounded response:** Stop deployment and obtain an approved data or model source.
- **Verification:** Complete the organization's documented review.
- **Boundary:** This sheet does not provide legal advice.
- **Prevention:** Track licensing and consent through artifact lineage.

#### F-105: Access-control logging gap

- **Signal:** A sensitive adapter is served without an auditable identity trail.
- **Competing explanation:** Requests and artifacts may be logged separately or incompletely.
- **Discriminating check:** Trace an authorized request from identity through adapter selection.
- **Bounded response:** Add privacy-preserving correlation and enforce access checks.
- **Verification:** Run authorized and denied request tests.
- **Boundary:** Logs themselves can contain sensitive information.
- **Prevention:** Apply retention and access controls to audit records.

#### F-106: Canary traffic not representative

- **Signal:** A rollout appears healthy but misses high-risk traffic.
- **Competing explanation:** Routing may exclude long prompts or important customer segments.
- **Discriminating check:** Compare canary and intended traffic distributions.
- **Bounded response:** Adjust routing or run targeted offline tests before expansion.
- **Verification:** Review quality and latency by predefined slice.
- **Boundary:** A canary cannot ethically expose users to known unsafe behavior.
- **Prevention:** Define representative and safety constraints before rollout.

#### F-107: Rollback metric delayed

- **Signal:** A harmful release remains active because labels arrive late.
- **Competing explanation:** Many task outcomes are only observable after a delay.
- **Discriminating check:** Identify leading signals and independent human review channels.
- **Bounded response:** Use conservative rollout thresholds and escalation procedures.
- **Verification:** Test the rollback trigger with a simulated incident.
- **Boundary:** Leading indicators may be imperfect proxies.
- **Prevention:** Document alert ownership and decision time limits.

#### F-108: Monitoring leaks user data

- **Signal:** Detailed logging exposes prompts or outputs.
- **Competing explanation:** Raw text collection may exceed operational need.
- **Discriminating check:** Audit log fields, access and retention against policy.
- **Bounded response:** Minimize or redact sensitive fields before storage.
- **Verification:** Test redaction on representative sensitive examples.
- **Boundary:** Automated redaction can miss novel patterns.
- **Prevention:** Apply least privilege and periodic access review.

#### F-109: Adapter cache eviction

- **Signal:** Requests fall back to the wrong model or fail after eviction.
- **Competing explanation:** Cache misses may silently select a default adapter.
- **Discriminating check:** Trace cache miss handling and artifact identity.
- **Bounded response:** Fail closed or load the authorized adapter explicitly.
- **Verification:** Test eviction under concurrent tenant requests.
- **Boundary:** A fallback may be acceptable only when explicitly authorized.
- **Prevention:** Define cache capacity and miss behavior per service.

#### F-110: Adapter rollout race

- **Signal:** A request uses metadata from one version and weights from another.
- **Competing explanation:** Artifact update and cache invalidation may not be atomic.
- **Discriminating check:** Record digests at lookup, load and generation.
- **Bounded response:** Use atomic immutable bundle publication.
- **Verification:** Exercise concurrent reads during a rollout.
- **Boundary:** A matching version label is weaker than a matching digest.
- **Prevention:** Bind all runtime components to a single release manifest.

#### F-111: Insufficient benchmark repetitions

- **Signal:** One run seems faster but variance is high.
- **Competing explanation:** Warm-up, contention or small sample size can dominate.
- **Discriminating check:** Inspect per-run distributions under controlled load.
- **Bounded response:** Repeat the planned benchmark and report spread.
- **Verification:** Compare paired workloads over multiple independent runs.
- **Boundary:** Repetition does not cure a systematically unrepresentative workload.
- **Prevention:** Specify warm-up and sampling protocol in advance.

#### F-112: Unreported failed runs

- **Signal:** Only successful configurations appear in the comparison.
- **Competing explanation:** OOMs and numerical failures can bias selection.
- **Discriminating check:** Audit the experiment registry against launched configurations.
- **Bounded response:** Report failures and their resource cost alongside successes.
- **Verification:** Recompute cost per successful result using all attempts.
- **Boundary:** Failure classification can remain uncertain without logs.
- **Prevention:** Retain run manifests and failure evidence.

#### F-113: Checkpoint selected on test

- **Signal:** The highest test score is chosen after many experiments.
- **Competing explanation:** The final test has become a selection mechanism.
- **Discriminating check:** Inspect experiment decisions and access logs.
- **Bounded response:** Use a fresh final holdout after locking the candidate.
- **Verification:** Report the selection procedure and comparison count.
- **Boundary:** A new holdout still needs representative coverage.
- **Prevention:** Separate validation and final-test permissions.

#### F-114: False equivalence claim

- **Signal:** An inconclusive comparison is described as equal performance.
- **Competing explanation:** A wide interval can include both meaningful benefit and harm.
- **Discriminating check:** Inspect paired effect size, uncertainty and predeclared margin.
- **Bounded response:** Report uncertainty rather than asserting equivalence.
- **Verification:** Run a suitably powered equivalence study if justified.
- **Boundary:** Failure to reject a difference is not evidence of equivalence.
- **Prevention:** Set practical margins and sample plans before data collection.

#### F-115: Quantization mistaken for QAT

- **Signal:** Design review assumes the frozen base is trained through fake quantization.
- **Competing explanation:** QLoRA and QAT have different trainable-state and quantizer semantics.
- **Discriminating check:** Inspect which parameters update and how quantization enters the graph.
- **Bounded response:** Use the precise method name and compare the actual algorithms.
- **Verification:** Check the optimizer's parameter groups and training graph.
- **Boundary:** Some hybrid methods combine ideas but require their own specification.
- **Prevention:** Document trainable tensors and quantizer update rules.

#### F-116: Paper benchmark generalized

- **Signal:** A task owner assumes QLoRA always matches full tuning.
- **Competing explanation:** The original result is bounded by its models, tasks and evaluation.
- **Discriminating check:** Read the paper's setup and compare it with the target workload.
- **Bounded response:** Run a task-specific controlled comparison.
- **Verification:** Report both comparable conditions and differences.
- **Boundary:** The paper's reported result is not reproduced here.
- **Prevention:** Keep claim, source and applicability together.

#### F-117: NF4 called four-bit integer

- **Signal:** A reviewer expects equally spaced signed integer levels.
- **Competing explanation:** NF4 uses a nonuniform codebook and block-level scaling.
- **Discriminating check:** Inspect the format definition and decoded representable values.
- **Bounded response:** Describe codes and scale rather than equating them with int4.
- **Verification:** Test decoder outputs for a small controlled block.
- **Boundary:** Exact bit packing and scales are implementation-specific.
- **Prevention:** Document the quantization format and version.

#### F-118: Double quantization called double model compression

- **Signal:** A budget subtracts a second four-bit copy of every base weight.
- **Competing explanation:** Nested quantization applies to quantization constants, not a second whole model.
- **Discriminating check:** Inventory code and scale tensors separately.
- **Bounded response:** Count each encoded object once.
- **Verification:** Reconcile the result against model-load memory.
- **Boundary:** Some implementations store additional metadata in different layouts.
- **Prevention:** Use a disjoint memory accounting worksheet.

#### F-119: Paging treated as extra GPU memory

- **Signal:** A capacity plan assumes optimizer state has no host or transfer cost.
- **Competing explanation:** Paged state still occupies storage and may migrate.
- **Discriminating check:** Measure both device and host memory plus transfer traffic.
- **Bounded response:** Budget both tiers and the interconnect.
- **Verification:** Repeat the longest representative run under contention.
- **Boundary:** OOM can arise from activations or kernels outside pageable state.
- **Prevention:** Specify which allocations the optimizer actually pages.

#### F-120: Quantized storage confused with activation precision

- **Signal:** Memory plan prices all intermediate tensors at four bits.
- **Competing explanation:** Compute, activations and optimizer states can use different dtypes.
- **Discriminating check:** Inspect runtime tensor and kernel dtypes.
- **Bounded response:** Assign a separate width and size to each memory category.
- **Verification:** Compare measured activation peak against the revised model.
- **Boundary:** Activation checkpointing and fusion can change liveness.
- **Prevention:** Avoid a single global bytes-per-parameter assumption.

#### F-121: Quantization error mistaken for data error

- **Signal:** A task regression is blamed on poor labels without isolating the base.
- **Competing explanation:** The frozen quantized base may already perform worse on the task.
- **Discriminating check:** Evaluate original and quantized bases before adapter training.
- **Bounded response:** Fix quantization or choose a different base when that comparison fails.
- **Verification:** Repeat the adapter comparison after the correction.
- **Boundary:** Base degradation and data defects can occur together.
- **Prevention:** Include both base controls in the evaluation matrix.

#### F-122: Data error mistaken for quantization

- **Signal:** The quantized variant is blamed despite malformed training labels.
- **Competing explanation:** The same corruption can hurt higher-precision LoRA.
- **Discriminating check:** Train or inspect a controlled higher-precision adapter baseline.
- **Bounded response:** Repair the demonstrated data issue before changing quantizer.
- **Verification:** Re-evaluate all variants on the same held-out set.
- **Boundary:** A higher-precision baseline may be infeasible on the same device.
- **Prevention:** Keep data validation independent of the model format.

#### F-123: Temporary dequantization ignored

- **Signal:** A step exceeds an estimate based on stored codes.
- **Competing explanation:** A kernel or framework can materialize temporary higher-precision buffers.
- **Discriminating check:** Profile allocation peaks and kernel behavior for the exact stack.
- **Bounded response:** Include observed temporary buffers or choose a supported lower-memory kernel.
- **Verification:** Measure the complete training step after the change.
- **Boundary:** Not every implementation materializes a full dequantized model.
- **Prevention:** Use runtime evidence rather than assuming a universal kernel design.

#### F-124: Model initialization uses excess memory

- **Signal:** Quantization loading fails despite enough space for final weights.
- **Competing explanation:** Loading may temporarily hold original and quantized forms.
- **Discriminating check:** Inspect process peak during the actual loader path.
- **Bounded response:** Use a supported loading strategy or more initialization capacity.
- **Verification:** Run clean-start loading on the target machine.
- **Boundary:** A post-load footprint cannot establish initialization peak.
- **Prevention:** Budget startup separately from steady-state operation.

#### F-125: Seed sweep changes data order

- **Signal:** Variation across runs is attributed only to adapter initialization.
- **Competing explanation:** Seeds may control shuffling, dropout and sampling too.
- **Discriminating check:** Record all RNG sources and data-order identifiers.
- **Bounded response:** Design controlled experiments that vary the intended factors.
- **Verification:** Report seed distribution and fixed versus varying components.
- **Boundary:** Perfect isolation may not be possible with nondeterministic kernels.
- **Prevention:** Define the reproducibility protocol before the sweep.

#### F-126: Adapter parameter count mismatch

- **Signal:** The formula disagrees with framework-reported trainable elements.
- **Competing explanation:** Biases, heads, tied weights or unmatched targets may explain the difference.
- **Discriminating check:** Enumerate each trainable tensor and compare its shape to E3.
- **Bounded response:** Correct the target or budget based on the actual manifest.
- **Verification:** Sum unique trainable elements and reconcile with optimizer groups.
- **Boundary:** Framework wrappers may expose names differently.
- **Prevention:** Persist the parameter manifest with every run.

#### F-127: Gradient count mismatch

- **Signal:** The estimate assumes every trainable tensor always has a gradient.
- **Competing explanation:** Unused branches or conditional execution can leave gradients absent.
- **Discriminating check:** Inspect gradients after representative backward passes.
- **Bounded response:** Budget for the worst supported active path and investigate unexpected gaps.
- **Verification:** Exercise conditional branches and verify updates.
- **Boundary:** A missing gradient can be expected for an unused path.
- **Prevention:** Test representative model paths before production training.

#### F-128: Optimizer state dtype assumption

- **Signal:** A moment tensor consumes more bytes than the estimate.
- **Competing explanation:** Optimizer implementations may use different precision or auxiliary state.
- **Discriminating check:** Inspect each optimizer state tensor after the first update.
- **Bounded response:** Recompute memory using actual state entries.
- **Verification:** Compare state inventory with observed memory growth.
- **Boundary:** Some optimizers allocate state lazily or use quantized state.
- **Prevention:** Record optimizer class, options and state dtypes.

#### F-129: Master weight copy omitted

- **Signal:** Actual training footprint exceeds adapter weights plus moments.
- **Competing explanation:** Some optimizers or mixed-precision stacks retain master parameters.
- **Discriminating check:** Inspect optimizer state and parameter storage identities.
- **Bounded response:** Add the master-copy bytes when present.
- **Verification:** Measure the delta after optimizer initialization and first update.
- **Boundary:** Do not assume a master copy always exists either.
- **Prevention:** Derive the budget from the exact optimizer implementation.

#### F-130: OOM due to concurrency

- **Signal:** A fit test passes alone but fails when the device is shared.
- **Competing explanation:** Other workers consume memory or reserve buffers concurrently.
- **Discriminating check:** Compare device-global memory and process allocations over time.
- **Bounded response:** Isolate the job or admit based on a shared capacity policy.
- **Verification:** Repeat under intended co-tenancy and peak overlap.
- **Boundary:** Allocator reservation may not be visible in a single tensor inventory.
- **Prevention:** Define GPU tenancy and capacity guarantees.

#### F-131: Memory unit confusion

- **Signal:** A budget has unexplained differences near its capacity limit.
- **Competing explanation:** Decimal GB and binary GiB use different denominators.
- **Discriminating check:** Recalculate each reported byte value with explicit units.
- **Bounded response:** Convert all terms to bytes before summing.
- **Verification:** Verify the final result against both unit systems.
- **Boundary:** Hardware marketing capacity and available runtime memory may differ.
- **Prevention:** Print byte counts and units in budget reports.

#### F-132: Counting base gradients

- **Signal:** A LoRA estimate includes gradients for every frozen base weight.
- **Competing explanation:** Only trainable tensors normally require optimizer gradients.
- **Discriminating check:** Inspect requires-grad and optimizer membership for the base.
- **Bounded response:** Remove unsupported assumptions from the estimate.
- **Verification:** Measure actual gradient allocations after backward.
- **Boundary:** The backward graph can still retain activations for input gradients.
- **Prevention:** Separate gradient storage from backward computation.

#### F-133: Missing base gradient path

- **Signal:** Adapter gradients fail in an architecture with frozen inputs.
- **Competing explanation:** A checkpointing or autograd setup may not preserve a required path.
- **Discriminating check:** Inspect the actual computation graph and supported preparation procedure.
- **Bounded response:** Use the documented model preparation and checkpointing settings.
- **Verification:** Verify adapter gradients on representative batches.
- **Boundary:** A specific fix depends on the model and library version.
- **Prevention:** Add an end-to-end gradient check to integration tests.

#### F-134: Static estimate treated as peak

- **Signal:** The sum of individually observed maxima is labeled exact.
- **Competing explanation:** Different memory categories may peak at different times.
- **Discriminating check:** Capture synchronized time-series or memory snapshots.
- **Bounded response:** Report a conservative upper estimate or measured concurrent peak.
- **Verification:** Compare the observed peak with the component timeline.
- **Boundary:** Telemetry overhead and allocator behavior can affect measurements.
- **Prevention:** Specify observation interval and measurement method.

#### F-135: Repeated evaluation consumes memory

- **Signal:** Training OOMs after several validation cycles.
- **Competing explanation:** References to outputs or computation graphs may be retained.
- **Discriminating check:** Inspect allocation growth and evaluation-mode gradient settings.
- **Bounded response:** Release references and disable unnecessary gradient recording.
- **Verification:** Track memory over repeated cycles in a controlled run.
- **Boundary:** Caching can be intentional and should have an explicit bound.
- **Prevention:** Set bounded retention and monitor evaluation cycles.

#### F-136: Versioned docs mistaken for local validation

- **Signal:** A guide example is assumed to have run on the current machine.
- **Competing explanation:** The documented example does not establish local compatibility.
- **Discriminating check:** Record installed package, driver and hardware versions.
- **Bounded response:** Perform a full integration test in the intended environment.
- **Verification:** Capture logs and artifact reload results.
- **Boundary:** This draft has not executed the example.
- **Prevention:** Maintain a verified environment matrix per release.

#### F-137: Misleading adapter-only checkpoint size

- **Signal:** Small artifact size is interpreted as total deployment footprint.
- **Competing explanation:** The base checkpoint and runtime still must be loaded.
- **Discriminating check:** Measure base, adapter and serving cache together.
- **Bounded response:** Report complete per-replica and shared storage requirements.
- **Verification:** Test cold startup and active serving peak.
- **Boundary:** Shared base memory depends on deployment architecture.
- **Prevention:** Publish both incremental adapter and whole-system costs.

#### F-138: Training loss computed on wrong tokens

- **Signal:** Loss appears low because only easy prompt tokens are scored.
- **Competing explanation:** Incorrect masking changes the effective objective.
- **Discriminating check:** Decode supervised positions and inspect sample weights.
- **Bounded response:** Correct the objective and retrain rather than rescaling the metric.
- **Verification:** Compare token-level loss on an independently checked batch.
- **Boundary:** Some prompt tokens may be intentionally supervised.
- **Prevention:** Keep objective examples as executable data checks.

#### F-139: Evaluation uses training mode

- **Signal:** Reported quality changes between runs unexpectedly.
- **Competing explanation:** Dropout and other train-mode behaviors can be active.
- **Discriminating check:** Inspect model mode during evaluation.
- **Bounded response:** Switch to the intended evaluation behavior.
- **Verification:** Repeat fixed-input checks and the evaluation suite.
- **Boundary:** Some stochastic evaluation protocols are intentionally randomized.
- **Prevention:** Record model mode and randomness policy.

#### F-140: Human review lacks independent reference

- **Signal:** Reviewers reward fluent but unsupported answers.
- **Competing explanation:** They may not have access to source evidence or task rules.
- **Discriminating check:** Audit the annotation interface and reference availability.
- **Bounded response:** Provide appropriate independent references and adjudication.
- **Verification:** Measure inter-reviewer agreement and resolve disputed cases.
- **Boundary:** Agreement alone does not prove labels are correct.
- **Prevention:** Version the rubric and sample design.

#### F-141: Retrieval changes alongside adapter

- **Signal:** An improvement is attributed to fine-tuning although context changed.
- **Competing explanation:** Retriever, index or source corpus may also have changed.
- **Discriminating check:** Run an ablation holding retrieval fixed across variants.
- **Bounded response:** Separate retrieval and adapter effects in the evaluation.
- **Verification:** Repeat paired tests with frozen retrieval artifacts.
- **Boundary:** Interactions may require a factorial comparison.
- **Prevention:** Version the retrieval index and context construction.

#### F-142: Tool permissions change alongside adapter

- **Signal:** A success gain may result from broader permissions.
- **Competing explanation:** More tool access can also create new safety risks.
- **Discriminating check:** Compare allowed actions and audit logs across variants.
- **Bounded response:** Hold permission sets fixed for model comparisons.
- **Verification:** Test denied actions and task success separately.
- **Boundary:** The production system may intentionally change both components.
- **Prevention:** Treat tool policy as a versioned evaluation input.

#### F-143: Cost ignores human review

- **Signal:** An apparently cheap model increases manual correction workload.
- **Competing explanation:** Low-quality outputs can impose downstream labor.
- **Discriminating check:** Measure reviewed outcomes and correction time on a representative sample.
- **Bounded response:** Include reviewer labor in cost per successful outcome.
- **Verification:** Compare total cost under the same service-quality target.
- **Boundary:** Labor cost is context- and currency-specific.
- **Prevention:** Record cost assumptions and review dates.

#### F-144: Model output length confounds speed

- **Signal:** One variant appears faster because it generates shorter answers.
- **Competing explanation:** Tokens per second and request latency measure different things.
- **Discriminating check:** Compare matched output-length distributions and task completion.
- **Bounded response:** Report both throughput and latency alongside output quality.
- **Verification:** Repeat at representative prompt and output budgets.
- **Boundary:** A deliberately shorter correct answer can still have real value.
- **Prevention:** Report token counts and stop reasons with latency.

#### F-145: Long-tail failures hidden

- **Signal:** A small evaluation misses rare but severe failure classes.
- **Competing explanation:** Finite random sampling can leave critical scenarios unobserved.
- **Discriminating check:** Add targeted adversarial cases alongside representative sampling.
- **Bounded response:** Treat targeted failures as separate release criteria.
- **Verification:** Track category coverage and remediate confirmed issues.
- **Boundary:** Passing targeted tests does not prove absence of unknown failures.
- **Prevention:** Maintain a threat-informed test inventory.

#### F-146: Escalation route missing

- **Signal:** A diagnostic identifies risk but nobody can authorize a decision.
- **Competing explanation:** Ownership and severity thresholds may not be defined.
- **Discriminating check:** Trace alert routing to an accountable on-call owner.
- **Bounded response:** Assign an owner and escalation path before launch.
- **Verification:** Run an incident tabletop using the actual runbook.
- **Boundary:** A tabletop is not a substitute for live monitoring.
- **Prevention:** Review ownership after team or system changes.

**Takeaway:** Diagnose with measured discriminators before selecting a mitigation; a symptom does not uniquely identify its cause.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic Complexity

Let m and n be output and input widths, r the adapter rank, T the number of processed token vectors, and K the number of training updates.
Let Pq be the number of encoded base weights and Pad the number of trainable adapter elements.
A multiply-add count is an operation model, not a wall-clock latency measurement.

| Operation | Time or work | Additional space | I/O and limitations |
|---|---|---|---|
| Encode a dense base | At least inspect the encoded weights, O(Pq); exact quantizer work depends on block algorithm | Codes and metadata | Read checkpoint and write quantized representation; startup can require more memory. |
| Base linear forward | O(Tmn) arithmetic for a dense map | Kernel-specific temporary storage | Read encoded weights and metadata; decode/fusion costs depend on the backend. |
| Adapter linear forward | O(Tr(m+n)) arithmetic | O(Tr) intermediate in a simple unfused implementation | Fusion or recomputation may change actual temporary allocation. |
| Training | K times per-step forward/backward work plus optimizer operations | Activations, gradients, states and workspaces | Sequence length, attention implementation, microbatch and precision control the dominant cost. |
| Adapter update | O(Pad) per optimizer step for standard elementwise optimizers | O(Pad) gradients and optimizer states | Paging may add host-device traffic and contention. |
| Separate-adapter serving | Base forward plus adapter work per token vector | Adapter weights, activations and generation cache | Throughput and latency must be measured under real concurrency. |
| Resume | Read checkpoint and restore model, optimizer, scheduler and state | Storage and loading buffers | Distributed gathering and conversion can dominate recovery. |

A full transformer also includes attention, normalization, nonlinearities, embeddings, output projection and their backward passes; the linear-layer row alone is not a full model cost.
The operation counts do not assume a particular sparse kernel, tensor core speed, or dequantization implementation.
Distributed runs add gradient communication, synchronization and straggler effects that depend on parallelism and sharding.

### 9b. Back-of-Envelope Sizing

**ILLUSTRATIVE / VERIFIED:** use the hypothetical Section 5 device and 8.8127903 GiB simultaneous footprint.
If the activation term rises from 3.00 GiB to 5.00 GiB and every other term is unchanged, the new estimate is 10.8127903 GiB.
The remaining nominal device margin is 12 - 10.8127903 = 1.1872097 GiB, or about 9.89% of capacity.
This sensitivity calculation assumes the other allocations and their simultaneous timing do not change; that is not guaranteed.
An actual longer context can also alter workspaces, the data pipeline and execution time, so the result is not a measured capacity bound.
If cost matters, compare total currency per accepted artifact or per successful task, including failed runs, storage, inference, review and labor.
`OPEN`: no currency, device price, latency, token throughput, energy or utilization has been measured in this hypothetical.
Record device, driver, library versions, workload distribution, concurrency, warm-up and allocation method for any real benchmark.

### Versioned implementation sketch

**SOURCED:** the [PEFT 0.18.0 quantization guide](https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/quantization) documents this sequence: configure bitsandbytes four-bit NF4 and optional double quantization; load; prepare for k-bit training; configure LoRA; wrap.
The following adapts that guide to an explicitly chosen model and is an implementation sketch, not an executed benchmark or guaranteed environment.
It does not select a paged optimizer; adding one requires a separately verified compatible optimizer configuration and device/host budget.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "mistralai/Mistral-7B-v0.1"
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization,
)
base = prepare_model_for_kbit_training(base)
adapter_config = LoraConfig(
    r=16,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
model = get_peft_model(base, adapter_config)
model.print_trainable_parameters()
```

Before use, pin an immutable model revision and compatible package, driver and hardware versions; review the model license and access requirements.
Verify which modules were wrapped, which other tensors remain trainable, actual storage and compute dtypes, and whether the environment supports BF16 and the quantization backend.
A causal-LM training loop still needs approved tokenized data, correct attention and label masks, optimizer configuration, checkpoints and held-out evaluation.
The guide is versioned at PEFT 0.18.0; its example is not a promise about other versions or every architecture.
**Takeaway:** Count operations for scaling intuition, but measure the complete workload and all relevant memory tiers for capacity or economics.

## 10. Key Theorems / Results Reference

| Result | Statement and essential assumptions | Implication | Does not imply | Source |
|---|---|---|---|---|
| Matrix rank bound | For compatible real matrices A and B, rank(BA) <= min(rank(A), rank(B)) <= r. | A rank-r adapter cannot represent every full-rank matrix update when r is too small. | The task needs a high-rank update or that larger rank always helps. | [Matrix rank, MIT OCW](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) (standard linear-algebra reference). |
| Parameter count | E2 follows from counting entries of two dense factors, with no extra parameters or sharing. | Compare candidate ranks and targets before allocation. | Training memory equals parameter count times a single global byte width. | DERIVED, Section 4. |
| QLoRA paper finding | The authors report competitive full-16-bit fine-tuning performance in their studied settings while using a frozen 4-bit base and adapters. | Motivates a controlled local comparison. | Universal performance parity, no quality loss, or reproduction here. | [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314). |
| PEFT guide | The v0.18.0 guide documents NF4/double-quant configuration and adapter preparation. | Provides a version-scoped implementation path. | Compatibility with every GPU, package combination, model or optimizer. | [PEFT quantization guide v0.18.0](https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/quantization). |

The matrix rank bound follows because the image of BA lies in the image of B, while its rank cannot exceed the rank of A; the inner dimension is r.
The QLoRA finding is an empirical report, not a theorem about future workloads.
The PEFT guide is implementation documentation, not evidence that the code in this draft has been run.
Source registry: paper published 2023-05-23; versioned PEFT guide 0.18.0; both accessed 2026-11-07 for the specific claims above.
Review triggers: paper correction, dependency change, new hardware or a change to the evaluation population.
The visible source label is clipped; its hidden suffix and body remain `OPEN`, not reconstructed.
**Takeaway:** Separate algebraic guarantees, experiment-specific paper findings and version-scoped implementation documentation.

## 11. Interview / Exam Questions

Scoring for every question: 0 = incorrect; 1 = partial; 2 = correct with assumptions; 3 = correct and defends the boundary or alternative.
The prompts below are teaching questions and are not attributed to an employer.

### Level 1 — Definition / Recall

#### Q01 — Recall

**Candidate prompt:** What is frozen, and what receives updates in quantized-base LoRA?

- **Clarification:** Assume no extra trainable head.
- **Answer:** The quantized base weights remain fixed; low-rank adapter parameters are optimized. Backpropagation traverses base operations to reach adapters.
- **Defensible alternative:** A deliberately trainable head changes the list and must be counted.
- **Follow-up:** If embeddings are trained, what changes in the manifest?
- **Trap:** Saying no backward pass uses the base confuses frozen parameters with a frozen graph.
- **Source:** Sections 2–4.

#### Q02 — Recall

**Candidate prompt:** What do NF4 and double quantization encode?

- **Clarification:** Discuss the paper’s scheme rather than every possible backend.
- **Answer:** NF4 uses four-bit nonuniform weight codes motivated by normal distributions; double quantization compresses quantization constants.
- **Defensible alternative:** Another four-bit scheme can use a different codebook or scale layout.
- **Follow-up:** How would outlier-heavy blocks affect the evaluation plan?
- **Trap:** Claiming nested quantization removes all metadata is wrong.
- **Source:** Sections 3 and 10.

#### Q03 — Recall

**Candidate prompt:** Is QLoRA identical to quantization-aware training?

- **Clarification:** Use QAT broadly, without assuming a single QAT algorithm.
- **Answer:** No. QLoRA trains adapters through a frozen quantized base; QAT is a family of methods modeling quantization during training with method-dependent trainable parameters.
- **Defensible alternative:** A hybrid approach needs its own specification and may combine ideas.
- **Follow-up:** Which parameters does the optimizer update?
- **Trap:** Treating all algorithms containing quantization as identical erases the trainable-state distinction.
- **Source:** Sections 3 and 4.

### Level 2 — Derivation / Apply

#### Q04 — Derivation

**Candidate prompt:** How many adapter elements target a 4096-by-4096 matrix at rank 16?

- **Clarification:** Ignore biases, extra modules and sharing.
- **Answer:** A has 16*4096=65,536 elements and B has 4096*16=65,536; together 131,072. Both factors have equal element counts here.
- **Defensible alternative:** A different rank or rectangular matrix uses r*(m+n).
- **Follow-up:** What is the count for four such matrices? 524,288.
- **Trap:** Counting only one factor halves the count.
- **Source:** E2 and Section 5.

#### Q05 — Derivation

**Candidate prompt:** How large are eight billion four-bit codes?

- **Clarification:** Count only packed codes; use binary GiB.
- **Answer:** Eight billion times four bits is 32 billion bits, or four billion bytes, approximately 3.7253 GiB. Metadata and other tensors are separate.
- **Defensible alternative:** A different encoded weight count changes the calculation.
- **Follow-up:** What if 0.8 GiB of metadata and other weights is added? Approximately 4.5253 GiB.
- **Trap:** Multiplying by four bytes mistakes bits for bytes.
- **Source:** E5 and Section 5.

#### Q06 — Derivation

**Candidate prompt:** Why can a rank-r adapter not represent every correction?

- **Clarification:** Consider a single m-by-n linear layer.
- **Answer:** The product BA has rank at most r, so a desired correction of rank greater than r cannot be represented exactly by that single update.
- **Defensible alternative:** The task might not require exact matrix recovery and multi-layer behavior can differ.
- **Follow-up:** What if r reaches min(m,n)? The rank bound no longer rules out a full-rank update.
- **Trap:** Equating low matrix reconstruction error with task success is unsupported.
- **Source:** Section 10.

#### Q07 — Application

**Candidate prompt:** If adapter weights and gradients each use 2 bytes and two moments use 4 bytes each, how many bytes per trainable element?

- **Clarification:** Assume no master copy and no other state.
- **Answer:** Two plus two plus eight equals 12 bytes per element. Multiply by the actual trainable count; activations and base are separate.
- **Defensible alternative:** Actual optimizers may use a master copy or different state representations.
- **Follow-up:** For 16,777,216 elements, what is the subtotal? 192 MiB.
- **Trap:** Assuming the same bytes apply to frozen base parameters overcounts optimizer state.
- **Source:** Section 5.

#### Q08 — Application

**Candidate prompt:** A 12 GiB device has an estimated simultaneous footprint of 8.8128 GiB. What margin remains?

- **Clarification:** Use the rounded estimate; no other consumer is assumed.
- **Answer:** The remaining nominal capacity is 12-8.8128=3.1872 GiB, about 26.56% of 12 GiB. Actual available capacity and peaks still require measurement.
- **Defensible alternative:** A conservative headroom policy may reject even this candidate.
- **Follow-up:** If activations grow by 2 GiB, what remains? About 1.1872 GiB.
- **Trap:** Calling the margin a proof of fit ignores other allocations.
- **Source:** Sections 5 and 9.

### Level 3 — Judgement / Design / Debug

#### Q09 — Design

**Candidate prompt:** When might full tuning or higher-precision LoRA be preferable to QLoRA?

- **Clarification:** The objective has quality, capacity and compliance gates.
- **Answer:** Prefer a feasible method that meets the safety and task criteria with the best measured operational tradeoff; use higher precision if quantization causes material regression and memory permits.
- **Defensible alternative:** Quantized-base tuning is defensible if it fits and passes the same gates.
- **Follow-up:** What if full tuning cannot fit? Report infeasibility rather than a quality ranking.
- **Trap:** Assuming full tuning always wins or QLoRA always matches it is unsupported.
- **Source:** Section 7.

#### Q10 — Debug

**Candidate prompt:** The model loads but runs out of memory at its first optimizer step. What do you check?

- **Clarification:** The load-time device usage alone is known.
- **Answer:** Measure the step-by-step peak and inspect lazy optimizer state, gradients, temporary buffers and concurrent processes; then address the measured source.
- **Defensible alternative:** A different step may expose activation or evaluation peaks.
- **Follow-up:** What if host RAM also climbs? Profile paging and host-resident state.
- **Trap:** Only shrinking the base may not address optimizer-state allocation.
- **Source:** F-03 and Section 8.

#### Q11 — Design

**Candidate prompt:** How do you isolate quantization impact from adapter impact?

- **Clarification:** Comparable models and held-out examples are available.
- **Answer:** Evaluate the original base, quantized base, higher-precision LoRA and quantized-base adapter with fixed data, templates and decoding; use paired uncertainty at the independent sampling unit.
- **Defensible alternative:** When a baseline is infeasible, label the missing comparison and avoid inferring its score.
- **Follow-up:** What if data contains multiple turns per conversation? Cluster or split at conversation level as appropriate.
- **Trap:** Comparing different datasets or prompts confounds the result.
- **Source:** Section 7.

#### Q12 — Debug

**Candidate prompt:** Why might paged optimization make a run slower?

- **Clarification:** The optimizer uses supported pageable state and memory pressure exists.
- **Answer:** State migration can create host-device traffic and contention; profile transfer, step time and host memory before concluding that paging is responsible.
- **Defensible alternative:** If migration is rare, data loading or kernels may dominate.
- **Follow-up:** What if host memory is exhausted? Increase capacity or reduce pressure; paging is not free storage.
- **Trap:** Assuming paging guarantees speed or eliminates memory needs is incorrect.
- **Source:** Section 8.

#### Q13 — Design

**Candidate prompt:** How do you validate a merged or re-quantized deployment artifact?

- **Clarification:** Separate-adapter inference has already been evaluated.
- **Answer:** Treat the converted artifact as a new model: compare clean reload, fixed-input logits within appropriate tolerances, task and safety metrics, memory and serving performance.
- **Defensible alternative:** If conversion is unsupported or fails the gates, retain the separate adapter.
- **Follow-up:** What if logits differ slightly but tasks remain acceptable? Apply the predeclared numerical and behavioral thresholds.
- **Trap:** Assuming the serialized merge is mathematically and numerically identical ignores conversion error.
- **Source:** Sections 7–8.

#### Q14 — Judgment

**Candidate prompt:** A small test reports no significant difference between QLoRA and full tuning. Are they equivalent?

- **Clarification:** No equivalence margin or power analysis was prespecified.
- **Answer:** No; the result may be inconclusive. Report the paired effect, interval and test design, and assess whether meaningful regressions remain plausible.
- **Defensible alternative:** A properly designed equivalence study can support a bounded equivalence conclusion.
- **Follow-up:** What if the confidence interval is narrower than a prespecified practical margin? Interpret it under the chosen equivalence procedure.
- **Trap:** Failure to reject a difference is not proof of equality.
- **Source:** Section 7.

#### Q15 — Design

**Candidate prompt:** How do you safely operate tenant-specific adapters?

- **Clarification:** Multiple tenants share an inference platform.
- **Answer:** Authorize tenant and adapter selection independently, bind cache keys to immutable base and adapter digests, pin sessions, test cross-tenant denial and retain a rollback bundle.
- **Defensible alternative:** Dedicated isolation may be preferable for stricter security requirements.
- **Follow-up:** What if an adapter is missing? Fail closed or use only an explicitly authorized fallback.
- **Trap:** Adapters by themselves are not an access-control boundary.
- **Source:** Section 8.

**Takeaway:** An interview-quality answer names assumptions, computes correctly, and explains what evidence could overturn its conclusion.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Decide and operate frozen quantized-base low-rank adapter tuning. |
| Core formula | `y = D(Q(W))x + (alpha/r)B(Ax)`; conventional scaling and compatible shapes required. |
| Key assumption | Base weights are frozen, targeted modules and actual precision/storage formats are known, and all live allocations are counted without overlap. |
| Complexity | Per targeted dense layer and T token vectors, base work is O(Tmn), adapter work O(Tr(m+n)); runtime depends on kernels and hardware. |
| Use when | Base memory is a binding constraint and controlled task, safety and operational evaluations pass. |
| Avoid when | Quantization causes unacceptable regression, a supported backend is unavailable, or the full training/serving peak exceeds capacity. |
| Memory rule | `max_t(base codes + metadata + adapters + gradients + optimizer + activations + temporaries + other)`; include host paging separately. |
| Diagnostic | Measure load, forward, backward, first optimizer step, save and evaluation peaks; inspect trainable names and actual dtypes. |
| Go deeper | [LoRA sheet](01_LoRA_Parameter_Efficient_Finetuning.md); [paper](https://arxiv.org/abs/2305.14314); [PEFT 0.18.0 guide](https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/quantization). |
