---
collection: "finetuning-alignment-and-model-customisation"
file_id: "03"
topic: "RoPE and position interpolation"
level: 3
status: "draft"
version: "0.1"
created: "2026-11-07"
reviewed: null
review_trigger: "Checkpoint, rotary configuration, kernel, target length, or evaluation distribution changes"
notation_style: "plain-text"
prerequisites: []
pattern_tags: ["transformers", "position", "long-context", "inference", "evaluation"]
source_status: "OPEN"
owner_role: "Model adaptation and serving teams"
source_index: "INDEX.md"
---
# RoPE and Position Interpolation

## 1. File Header

> **Purpose:** Derive rotary relative-position attention, select a context-extension strategy, and make a falsifiable deployment decision.
> **Prerequisites:** Transformer query/key attention, vector dot products, rotations, causal masking, and bytes versus binary units.
> **Key outcome:** Given a checkpoint, rotary convention, length budget and evaluation data, compute position mapping and KV-cache size and propose a controlled extension and release gate.
> **This file does NOT cover:** A complete pretraining recipe, every long-context RoPE variant, or security certification.
> **Level:** 3 — advanced and production.
> **Version / reviewed:** 0.1 / not independently reviewed.
> **Evidence:** DERIVED identities, ILLUSTRATIVE constructed examples, SOURCED paper claims; runtime performance and business outcomes are NOT MEASURED.
> **Retrieval cue:** Use Sections 4–5 for math, 8 for diagnosis, and 9 for sizing.

The source inventory lists Module 11 topics on motivation, RoPE/interpolation, long-context training, inference and evaluation.
The original course files are unavailable; their contents are not reconstructed or attributed here.
This sheet is independently authored from the stated primary papers and explicitly identified assumptions.
A larger accepted input limit is not proof that a model reliably uses the additional tokens.
The practical decision is whether the task-quality benefit justifies training, memory, latency and operational risk.
**Takeaway:** Treat numerical position support, trained length, evaluated performance and serving limit as separate claims.

## 2. Concept Map / Mental Model

```text
Checkpoint + tokenizer + rotary convention + known training length
             |                         |
             v                         v
       choose target length       reserve held-out tasks
             |                         |
       baseline / interpolate          |
             |                         |
    optional long-context training    |
             |                         |
  checkpoint + fixed position policy  |
             |                         |
       prefill -> rotated K cache -> incremental decoding
             |                         |
             +----> compare quality, cost, safety, reliability
                             |
                      gate / rollback
```
Each query and key is rotated at its own logical position before the dot product.
Relative displacement appears in that dot product because rotations compose.
Position interpolation compresses position coordinates and normally accompanies adaptation; it does not compress the actual tokens.
The cache must use the same position mapping as the query that consumes it.
**Takeaway:** The mapping is part of the model-and-cache contract, not a mere UI limit.

## 3. Foundational Definitions

| Term | Definition / units / range | Operational distinction |
|---|---|---|
| Token position m, n | Zero-based logical token indices, nonnegative integers in the normal sequence. | A padded array offset may differ from the logical position. |
| q, k | Query and key vectors for a specified head, dimension d. | Values depend on token content and preceding model layers. |
| Rotary dimension d_r | Even number of query/key coordinates to rotate. | It may be smaller than head dimension d; the remaining coordinates are unrotated. |
| Frequency theta_j | Angle increment in radians per position for rotary pair j. | It is not an attention weight or measured frequency of tokens. |
| R(p) | Block-diagonal orthogonal rotation matrix with pair angles p*theta_j. | A fractional p is valid for the trigonometric operation. |
| L_train | Maximum sequence length used in a specified training phase. | The model may have multiple phases and unknown training-length distribution. |
| L_target | Intended maximum total sequence length, prompt plus generated tokens. | A configured limit is not measured competence. |
| Scale s | Positive mapping factor, here L_target/L_train when extending. | Some implementations define its reciprocal instead. |
| KV cache | Stored per-layer, per-KV-head keys and values for previous tokens. | KV heads can be fewer than query heads in grouped-query attention. |
| Prefill / decode | Processing prompt tokens / generating subsequent tokens. | Their workload shapes and bottlenecks differ. |
| Extrapolation | Evaluating original positional mapping beyond trained range. | It is not the same operation as interpolation and retraining. |
| Interpolation | Mapping target positions to a smaller coordinate range, here p/s. | It does not eliminate long-distance attention or guarantee quality. |
| Context utilization | Successful use of relevant evidence across a stated task and position distribution. | Acceptance of long input alone is not utilization. |
| Packed example | Multiple examples concatenated within a sequence for training. | Proper isolation requires appropriate masks and position semantics. |

The familiar base-frequency schedule is an implementation choice, not part of the abstract rotation identity.
For one common convention, theta_j = base^(-2j/d_r), for j = 0,...,d_r/2-1 and base > 0.
A checkpoint's actual base, pairing order, partial rotary dimension and scaling configuration are authoritative for that checkpoint.
A head's content vectors vary with the whole prefix, so the entire model output is not generally a function only of relative positions.
**Takeaway:** Record the exact checkpoint-specific mapping before reasoning about context extension.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

Validity: one two-dimensional pair, identical frequency and rotation convention for query and key, real coordinates.
Define R(a) = [[cos(a), -sin(a)], [sin(a), cos(a)]].
Multiplying the transpose by the matrix at another angle gives R(a)^T R(b) = R(b-a), using cosine and sine difference identities.
Rotating q by m*theta and k by n*theta therefore yields the following identity.
**FORMULA [E1]:** `(R(m*theta) q)^T (R(n*theta) k) = q^T R((n-m)*theta) k`.
**WHERE:** q and k are real 2-vectors; m,n are position indices; theta is radians per position.
**CONSTRAINT:** The same orthogonal rotation convention and theta must be used on both sides.
**NUMERICAL EXAMPLE:** q=k=[1,0], theta=pi/2, m=0, n=1 gives q^T R(pi/2) k = 0.
**INTERPRETATION:** the position-dependent part of this pair's dot product depends on n-m, not separately on m and n.
**Boundary:** q and k themselves are content-dependent; the identity does not establish translation invariance of the whole network.
The standard attention score then divides the dot product by sqrt(d), followed by masking and softmax.
The causal mask, positional interpolation and q/k normalizations are separate operations.

### 4.2 General Case

Validity: d_r even, matching rotary blocks for q and k, same logical position policy; remaining coordinates may be unrotated.
Stack pair rotations into R(p) = diag(R(p*theta_0), ..., R(p*theta_(d_r/2-1))).
Blockwise multiplication applies E1 to every pair; any unrotated tail contributes its ordinary dot product.
**FORMULA [E2]:** `score(m,n) = [sum_j q_j^T R((n-m)*theta_j) k_j + q_tail^T k_tail] / sqrt(d)` before mask and softmax.
**WHERE:** q_j,k_j are 2-vectors; tail coordinates have dimension d-d_r; d is the head dimension.
**CONSTRAINT:** Use the scaling actually specified by the attention architecture if it differs from sqrt(d).
**NUMERICAL EXAMPLE:** d=4, two rotary pairs, q=k=[1,0,1,0], theta=[pi/2,pi], n-m=1: numerator is 0 + (-1) = -1; score = -1/2.
**INTERPRETATION:** frequencies contribute different phases, and their sum can be positive or negative.
**Boundary:** changing the pairing layout without consistently permuting weights changes the score; interleaved and split-half layouts are not automatically interchangeable.

Validity: uniform linear interpolation at scale s>0, with the same mapping for keys and queries.
Substitute p=m/s and p=n/s into E1; the relative angle becomes (n-m)*theta/s.
**FORMULA [E3]:** `p(m)=m/s`, where for target length L_target and original length L_train, `s=L_target/L_train`.
**WHERE:** lengths count tokens, positions run from zero; p is a real-valued position coordinate.
**CONSTRAINT:** L_train and L_target are positive; use s>=1 for the extension described here.
**NUMERICAL EXAMPLE:** L_train=4096, L_target=16384, s=4; index 16383 maps to 4095.75.
**INTERPRETATION:** the target coordinate lies below 4096, though it exceeds the last original integer index 4095.
**Boundary:** changing coordinates shifts even short-range phases; being inside the original coordinate interval is not proof of unchanged quality or training distribution.
If exact alignment of last integer endpoints is a requirement, (L_train-1)/(L_target-1) is a different convention; do not silently substitute it for E3.

### 4.3 Key Special Cases / Variants

| Variant | Change | Assumption | Use when | Avoid when |
|---|---|---|---|---|
| Original RoPE | s=1 | Matching pretrained convention. | Baseline within known training/evaluation range. | Claiming unmeasured extrapolation is validated. |
| Direct extrapolation | Original phases at positions beyond training. | Runtime accepts the positions. | Diagnostic baseline under controlled tests. | Reliability is mandatory without measured task evidence. |
| Uniform PI | All position coordinates divided by s. | Model uses compatible RoPE, consistent policy. | Adapting a compatible checkpoint for longer sequences. | Kernel or cache cannot preserve mapping consistently. |
| Partial rotary | Rotate d_r<d coordinates. | Architecture explicitly uses partial rotation. | Matching its checkpoint. | Assuming full head rotation. |
| Other frequency scaling | Frequency-dependent changes. | Exact published or model-specific configuration known. | A measured alternative matches workload. | Treating it as identical to uniform PI. |
| Retrieval/chunking | Reduce tokens in each model call. | Relevant content can be retrieved with acceptable recall. | Full-document attention is too costly. | Critical cross-document dependencies cannot be recovered. |

The position-interpolation paper reports adaptation experiments for specific LLaMA models and tasks, not a universal guarantee.
Its abstract describes extension up to 32768 tokens, adaptation within 1000 steps, and preservation of short-context quality in its experiments [P2].
Those are author-reported findings, not measurements reproduced here, and they do not establish the same outcome on another checkpoint or corpus.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

At s=1 the uniform mapping recovers original positions exactly.
As s grows for fixed m and n, their rotary relative angle approaches zero for fixed theta.
Thus large compression can make nearby positions more similar in positional phase; it does not merge their actual token representations by definition.
Adding a common offset c to both m and n leaves the isolated rotary dot product unchanged under the stated matching conditions.
A model with other position-dependent operations, a boundary token, or altered hidden states need not produce identical outputs after a shift.
For q=k=[1,0], theta=pi/2, m=0, n=1, the rotary dot is 0.
If the key mistakenly uses theta=pi instead, that dot becomes -1, not the value implied by the shared-frequency identity.
A discontinuous or length-dependent mapping can invalidate reuse of a key cached under a previous mapping.
**Takeaway:** The relative identity is exact for matching rotations, while long-context quality remains an empirical question.

## 5. Worked Numerical Example

**ILLUSTRATIVE; arithmetic checked:** A team has a RoPE model trained through 4096-token sequences and wants an experimental 16384-token total context.
The scenario is fictional; no model, workload or outcome is measured.
Set s=16384/4096=4 and allocate prompt plus response within the same total limit.
For the first rotary pair use theta=pi/2 only to make the arithmetic hand-checkable; actual checkpoints have their own frequencies.
Let the content query and key each be [1,0], with query m=4 and key n=0.
The causal ordering n<=m holds; actual contents of q and k are assumed fixed in this small algebraic example.
Original positional difference is n-m=-4, giving angle -2*pi and pair dot product 1.
Under PI, the difference in mapped coordinates is -1, giving angle -pi/2 and dot product 0.
Independent check: rotating the query at m/4=1 gives [0,1], while the key at zero is [1,0]; their dot is 0.
These are unscaled pair contributions, not normalized attention probabilities.
For d=2, division by sqrt(2) preserves the zero but changes the original contribution to approximately 0.7071.
A plausible wrong path rotates only the query under PI while keeping a key cached under the old policy.
At m=4 and n=0 this particular key at zero masks the bug; test nonzero key positions too.
For example m=4,n=2 gives consistent PI angle -pi/4 and dot sqrt(2)/2, while a key rotated with unscaled n=2 gives relative angle pi/2 and dot 0.
At production scale, the team must evaluate the full checkpoint, causal mask, packing, positions and memory rather than extrapolate from this pair.
An acceptable release requires measured task quality at multiple lengths and positions, short-context retention, bounded costs and a rollback.
Remaining uncertainty includes training data quality, model-specific phase conventions and the actual serving implementation.
**Takeaway:** PI changes relative phases even for short sequences, so a correct rotation is necessary but not sufficient for quality.

## 6. Visual / Geometric Intuition

```text
Pair plane (schematic; axes are coordinates of one q/k pair)
             y
             ^   R(pi/2)[1,0] = [0,1]
             |
             |
  -----------+----------> x
           [0,0]    [1,0] = R(0)[1,0]
```
The dot of the two indicated unit vectors is zero, reflecting their 90-degree relative angle.
Rotation preserves each pair's Euclidean norm, but it does not preserve a dot product when q and k receive different angles.
Applying the same extra rotation to both vectors preserves their dot, which is the geometric heart of E1.

```text
Token index:  0 ---------------- 4095 ---------------- 16383
Original:     0 ---------------- 4095 ---------------- 16383
PI s=4:       0 --------------- 1023.75 -------------- 4095.75
```
The diagram is schematic and not to scale; all coordinates are positions, not sequence lengths.
PI leaves token count unchanged and spreads more tokens over nearly the original coordinate range.
**Takeaway:** Compression is in angular coordinates, not in the physical number of tokens or attention interactions.

## 7. Comparison / Selection Table

| Method | Preconditions | Quality evidence | Cost and production consequence | Use when | Avoid when |
|---|---|---|---|---|---|
| Original checkpoint, original length | Supported runtime and validated limit. | Existing checkpoint evaluation must be checked. | Lowest change risk; may need multiple calls. | Task fits and baseline meets quality. | Important evidence cannot fit and chunking loses it. |
| Original mapping beyond training | Backend supports longer positions. | Extrapolated quality NOT MEASURED here. | Dense attention remains expensive. | A diagnostic baseline. | Hard quality guarantees are required absent evidence. |
| PI without adaptation | Compatible implementation. | No general quality guarantee. | No training but changed phases. | Measuring a zero-training ablation. | Treating it as the paper's complete adaptation recipe. |
| PI with long-context training | Licensed corpus, compute, matching implementation. | [P2] reports model/task-specific experiments. | Training and evaluation cost; long inference still expensive. | Measured long-context gains justify costs. | Short-task regressions or capacity breach gate. |
| Retrieval plus shorter context | Index and authorized retrieval available. | Recall and grounding must be measured. | Indexing, freshness, access-control and retrieval overhead. | Sparse relevant evidence is identifiable. | Critical information cannot be retrieved reliably. |
| Different native long-context model | Appropriate license, evaluation and serving support. | Compare on the same blind task suite. | Migration and operational risk. | Existing extension fails constraints. | Vendor length claim is the sole evidence. |

Prioritize authorization, safety and hardware feasibility before quality and unit-cost optimization.
If the baseline meets requirements, prefer the simpler supported configuration.
If evidence is distributed across long text, compare PI and a suitable native model against retrieval with matched access and task information.
When tradeoffs conflict, require decision-owner approval on measured quality, latency and cost rather than declaring a universal winner.
**Takeaway:** PI is an adaptation candidate, not a default substitute for a matched baseline and evidence.

## 8. Failure Modes & Edge Cases

All cases below are HYPOTHETICAL diagnostic playbooks, not incidents.
A negative diagnostic narrows a hypothesis; it does not prove that the rest of the system is correct.
### F01 — Accepts long prompt but misses evidence
- **Trigger:** Accepts long prompt but misses evidence.
- **Candidate mechanism:** Position extension is mistaken for demonstrated utilization.
- **Discriminating diagnostic:** Place answer-bearing facts at controlled depths; compare position-stratified accuracy.
- **Bounded correction:** Evaluate multi-hop and natural tasks in addition to single-needle tests.
- **Prevention and verification:** Gate by worst relevant depth and task slice.
- **Competing explanation / limit:** A retrieval failure can resemble a reasoning failure; provide oracle context as a control.
- **Additional control:** Give the same question only the known supporting passage; improvement suggests an assembly or utilization bottleneck.
- **Escalation:** Ask the evaluation owner to inspect whether missing evidence or wrong reasoning dominates the paired failures.
### F02 — Short tasks regress after scaling
- **Trigger:** Short tasks regress after scaling.
- **Candidate mechanism:** Uniform interpolation changes local phase differences.
- **Discriminating diagnostic:** Compare identical short prompts against original checkpoint and PI checkpoint.
- **Bounded correction:** Mix short and long training examples or choose a different evaluated policy.
- **Prevention and verification:** Keep a frozen short-context regression suite.
- **Competing explanation / limit:** A changed tokenizer or prompt also causes apparent regression.
- **Additional control:** Compare base and adapted checkpoints on identical short prompts with the original and interpolated position policies.
- **Escalation:** If regression follows the checkpoint rather than the mapping, review training mix and optimization with the adaptation owner.
### F03 — Cached and uncached outputs diverge
- **Trigger:** Cached and uncached outputs diverge.
- **Candidate mechanism:** Position IDs, scale, mask, or cache position mismatch.
- **Discriminating diagnostic:** Compare per-step logits in deterministic eval with identical tokens.
- **Bounded correction:** Fix mapping and cache construction before rolling out.
- **Prevention and verification:** Add cache parity checks across prefill and decode.
- **Competing explanation / limit:** Tiny floating-point differences alone are not proof of a semantic bug.
- **Additional control:** Compare first divergent logits at positions 0, 1 and a nonzero prefix boundary under fixed decoding.
- **Escalation:** A divergence introduced only after cache reuse warrants inspection of stored key layout and rotation state.
### F04 — First batch works, later batch fails
- **Trigger:** First batch works, later batch fails.
- **Candidate mechanism:** Padding offsets treated as logical positions.
- **Discriminating diagnostic:** Compare an unpadded item alone and in differently padded batches.
- **Bounded correction:** Use architecture-correct logical position IDs and attention masks.
- **Prevention and verification:** Test left padding, right padding and variable lengths.
- **Competing explanation / limit:** Some kernels intentionally use physical positions; check their contract.
- **Additional control:** Hold token IDs constant while varying padding side and batch companions.
- **Escalation:** If logical positions match but logits diverge, inspect the kernel's padding mask and cache offsets.
### F05 — Quality changes at a threshold
- **Trigger:** Quality changes at a threshold.
- **Candidate mechanism:** A dynamic mapping changes after keys were cached.
- **Discriminating diagnostic:** Trace scale and position map at each decode step.
- **Bounded correction:** Pin mapping for session or rebuild all cached keys under the new mapping.
- **Prevention and verification:** Store policy identity with cache metadata.
- **Competing explanation / limit:** Changing scale only for new queries invalidates old-key compatibility.
- **Additional control:** Record scale and cache-construction scale at every step around the observed threshold.
- **Escalation:** If they differ, invalidate that session cache and route it to a compatible policy.
### F06 — Unexpected rotary mismatch after migration
- **Trigger:** Unexpected rotary mismatch after migration.
- **Candidate mechanism:** Interleaved versus split-half pairing or different base.
- **Discriminating diagnostic:** Compare rotated vectors and logits against the originating implementation.
- **Bounded correction:** Use the checkpoint-compatible implementation and layout.
- **Prevention and verification:** Record version, pairing and frequency schedule in artifact manifest.
- **Competing explanation / limit:** A shape match alone cannot establish semantic compatibility.
- **Additional control:** Probe an asymmetric vector such as `[1,2,3,4]` to expose different coordinate pairings.
- **Escalation:** If rotations differ before attention, compare the originating checkpoint's exact layout and phase signs.
### F07 — Training loss improves but held-out long tasks fail
- **Trigger:** Training loss improves but held-out long tasks fail.
- **Candidate mechanism:** Training examples reward local prediction rather than long-range use.
- **Discriminating diagnostic:** Evaluate long-range evidence manipulation and hard distractors.
- **Bounded correction:** Improve training task distribution and retain untouched holdout.
- **Prevention and verification:** Track task, length and evidence-depth distributions.
- **Competing explanation / limit:** Low loss is not sufficient evidence of effective context use.
- **Additional control:** Replace distant evidence while holding local context fixed and measure the change in the correct answer's likelihood.
- **Escalation:** If distant evidence has little influence, inspect the fraction of training examples requiring cross-document dependencies.
### F08 — OOM only on long prefill
- **Trigger:** OOM only on long prefill.
- **Candidate mechanism:** Activation/workspace or attention storage exceeds budget.
- **Discriminating diagnostic:** Capture peak allocation timeline at target batch and length.
- **Bounded correction:** Reduce batch or length, use a validated efficient attention kernel.
- **Prevention and verification:** Capacity-test prefill separately from decode.
- **Competing explanation / limit:** Efficient attention avoids materializing full scores but dense compute remains quadratic.
- **Additional control:** Measure peak prefill allocation with the same prompt at batch sizes one and two.
- **Escalation:** If attention workspace dominates even at batch one, evaluate a supported tiled implementation or lower the length cap.
### F09 — OOM during long generation
- **Trigger:** OOM during long generation.
- **Candidate mechanism:** KV cache grows with sequence and concurrency.
- **Discriminating diagnostic:** Measure cache bytes per layer and live sequence lengths.
- **Bounded correction:** Reserve generation budget and enforce admission control.
- **Prevention and verification:** Alert on cache pressure and prefix lifetime.
- **Competing explanation / limit:** KV memory estimate excludes allocator fragmentation and other live tensors.
- **Additional control:** Plot resident cache pages against generated token count for a single uninterrupted session.
- **Escalation:** If allocation rises after requests end, inspect page reclamation and cancellation paths.
### F10 — More context increases latency dramatically
- **Trigger:** More context increases latency dramatically.
- **Candidate mechanism:** Dense prefill work rises roughly quadratically with length.
- **Discriminating diagnostic:** Profile prefill versus queueing and tokenization under fixed load.
- **Bounded correction:** Use measured length caps, batching and retrieval if task permits.
- **Prevention and verification:** Publish latency by prompt-length and output-length buckets.
- **Competing explanation / limit:** Big-O does not predict a specific latency or kernel throughput.
- **Additional control:** Record queue wait, prefill and decode time separately for matched prompt-length buckets.
- **Escalation:** If queue wait dominates rather than kernel time, investigate admission and batch scheduling before changing RoPE.
### F11 — Passkey score high but real tasks fail
- **Trigger:** Passkey score high but real tasks fail.
- **Candidate mechanism:** Synthetic task is easier than the deployment distribution.
- **Discriminating diagnostic:** Use held-out documents, adversarial distractors and human or task gold.
- **Bounded correction:** Report task-specific metrics and uncertainty.
- **Prevention and verification:** Separate retrieval, reasoning and citation quality.
- **Competing explanation / limit:** One successful retrieval benchmark is not comprehensive.
- **Additional control:** Include examples requiring two separated passages and penalize incomplete evidence use.
- **Escalation:** Have domain reviewers inspect whether the synthetic target matches the actual decision task.
### F12 — Model follows hostile document instructions
- **Trigger:** Model follows hostile document instructions.
- **Candidate mechanism:** Retrieved or supplied text is treated as trusted instructions.
- **Discriminating diagnostic:** Insert benign canary instructions in untrusted context in a controlled test.
- **Bounded correction:** Separate instruction authority and enforce tool permissions externally.
- **Prevention and verification:** Test injection and access-control boundaries before release.
- **Competing explanation / limit:** Longer context does not itself authorize untrusted instructions.
- **Additional control:** Use a benign document canary that requests an unauthorized tool action and verify external policy denial.
- **Escalation:** Disable the affected tool route and involve the security owner if authorization can be bypassed.
### F13 — Truncated source is mistaken for model failure
- **Trigger:** Truncated source is mistaken for model failure.
- **Candidate mechanism:** Input assembly or generation budget silently clips content.
- **Discriminating diagnostic:** Log tokenizer counts and exact truncation decision without sensitive payloads.
- **Bounded correction:** Return explicit limit errors or use an approved truncation policy.
- **Prevention and verification:** Test near-boundary prompts and reserve response tokens.
- **Competing explanation / limit:** Avoid logging private document content.
- **Additional control:** Preserve token counts and source-span inclusion flags at each assembly step without recording sensitive text.
- **Escalation:** If the gold span was removed, repair retrieval or truncation before changing model weights.
### F14 — Loss fluctuates under long-sequence training
- **Trigger:** Loss fluctuates under long-sequence training.
- **Candidate mechanism:** Optimization, packing, precision or data changes are confounded.
- **Discriminating diagnostic:** Inspect gradients, tokens per update and matched controls.
- **Bounded correction:** Isolate one change and restore a stable checkpoint if needed.
- **Prevention and verification:** Save optimizer, RNG, sampler and position configuration.
- **Competing explanation / limit:** No single curve establishes a unique root cause.
- **Additional control:** Compare gradient norm and supervised-token counts at equal optimizer updates and sequence-length distributions.
- **Escalation:** Restore a stable checkpoint if instability persists with a fixed data and precision configuration.
### F15 — Cross-tenant answer contamination
- **Trigger:** Cross-tenant answer contamination.
- **Candidate mechanism:** Cache or retrieval identity is scoped too broadly.
- **Discriminating diagnostic:** Run isolated adversarial tenant and prefix-cache tests.
- **Bounded correction:** Key caches by tenant, authorization, model and exact policy.
- **Prevention and verification:** Enforce isolation at service boundaries and audit access.
- **Competing explanation / limit:** Hash equality does not replace authorization.
- **Additional control:** Test identical prefixes across tenants with different permissions and inspect the cache hit decision.
- **Escalation:** Isolate affected caches and involve the security owner if a tenant can read another tenant's state.
### F16 — Inconsistent outputs across serving replicas
- **Trigger:** Inconsistent outputs across serving replicas.
- **Candidate mechanism:** Different model revision, scaling config or kernel deployed.
- **Discriminating diagnostic:** Compare artifact manifests and logits on fixed probes.
- **Bounded correction:** Pin and roll back incompatible replicas.
- **Prevention and verification:** Validate artifacts before routing live traffic.
- **Competing explanation / limit:** Different low-level precision may need tolerance-based comparison.
- **Additional control:** Compare manifest digests and the first unequal layer output on a fixed token sequence.
- **Escalation:** Remove the divergent replica from routing until its model, kernel and configuration are reconciled.
### F17 — Degradation near target endpoint
- **Trigger:** Degradation near target endpoint.
- **Candidate mechanism:** Training length distribution or evaluation coverage misses the tail.
- **Discriminating diagnostic:** Plot quality against actual length and answer depth near boundary.
- **Bounded correction:** Expand data and test coverage, or reduce advertised limit.
- **Prevention and verification:** Monitor target-length percentile and edge slices.
- **Competing explanation / limit:** The endpoint itself is not a guarantee of sufficient training frequency.
- **Additional control:** Evaluate adjacent length bins while fixing evidence depth as a fraction of the full prompt.
- **Escalation:** Reduce the published limit if endpoint performance fails the predeclared threshold.
### F18 — Answer is correct but citation points elsewhere
- **Trigger:** Answer is correct but citation points elsewhere.
- **Candidate mechanism:** Position tracking or postprocessing offsets are wrong.
- **Discriminating diagnostic:** Compare cited spans against original token-to-document mapping.
- **Bounded correction:** Preserve provenance through concatenation and verify citations.
- **Prevention and verification:** Test repeated strings and document-boundary cases.
- **Competing explanation / limit:** A correct answer can still lack a supporting citation.
- **Additional control:** Trace the answer span through document concatenation, tokenization and rendering.
- **Escalation:** Suppress unverified citations if the character-to-token mapping cannot be reconciled.
### F19 — Long batch gives corrupted document boundaries
- **Trigger:** Long batch gives corrupted document boundaries.
- **Candidate mechanism:** Packed training examples attend across isolation boundaries.
- **Discriminating diagnostic:** Inspect block mask and label masking with synthetic markers.
- **Bounded correction:** Use supported block-diagonal causal masking or separate sequences.
- **Prevention and verification:** Include cross-example leakage checks in data validation.
- **Competing explanation / limit:** Resetting position IDs alone does not enforce attention isolation.
- **Additional control:** Alter a unique marker in an earlier packed example and verify later-example logits remain unchanged.
- **Escalation:** Stop training if the intended independent-example attention mask permits cross-boundary access.
### F20 — Large-position results differ across precisions
- **Trigger:** Large-position results differ across precisions.
- **Candidate mechanism:** Phase computation loses precision or kernel uses a different approximation.
- **Discriminating diagnostic:** Compare sin/cos and logits to a high-precision reference at large positions.
- **Bounded correction:** Use validated phase computation and supported precision.
- **Prevention and verification:** Add extreme-position numerical regression probes.
- **Competing explanation / limit:** No single dtype is guaranteed safe for every frequency and length.
- **Additional control:** Compare high-precision phases and rotary outputs for the slowest and fastest checkpoint frequencies.
- **Escalation:** If error grows with position, retain a validated phase precision or reduce the supported range.
The collection-wide Common Issues artifact is planned; these local IDs are not completed entries in that future artifact.
**Takeaway:** Separate observable symptoms from causal hypotheses and fix only after a discriminating control.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic Complexity

Let B be batch size, T prompt length, G generated tokens, H_q query heads, H_kv key/value heads, d head dimension, and N layers.
For simplicity assume all layers have the same H_q, H_kv and d, dense causal self-attention, and a fixed batch.
The number of allowed causal query-key pairs is T(T+1)/2, still Theta(T^2).
Attention score and value aggregation arithmetic is Theta(B*N*H_q*T^2*d) for dense prefill, omitting constant factors.
Projection and MLP computation also matter and are not included in that attention-only term.
A naive implementation materializes Theta(B*H_q*T^2) scores per layer; memory-efficient kernels can avoid that full matrix without making dense pairwise attention linear-time.
With an existing KV cache, G decode steps attend over a growing prefix: Theta(B*N*H_q*d*(G*T+G*(G-1)/2)) attention work, excluding projections.
Different batching, sliding windows, sparse attention, parallelism and speculative decoding alter the model; assess each implementation separately.
Dense KV storage for a uniform batch is linear in total retained length.
Validity: both keys and values are retained for all layers, no prefix sharing, quantization, eviction or sharding, and b bytes per cached scalar.
Derivation: B sequences times N layers times H_kv heads times T tokens times d coordinates, times two tensors, times b bytes.
**FORMULA [E4]:** `KV_bytes = 2*B*N*H_kv*T*d*b`.
**WHERE:** B,N,H_kv,T,d are positive integer counts; b is bytes per element.
**CONSTRAINT:** Count only resident cache data represented by these dimensions; cache metadata and allocator slack are excluded.
**NUMERICAL EXAMPLE:** B=1,N=32,H_kv=8,T=16384,d=128,b=2 gives 2,147,483,648 bytes = 2 GiB.
**INTERPRETATION:** four concurrent independent sequences at this length require 8 GiB of KV payload under the assumptions.
**Boundary:** grouped-query attention reduces H_kv relative to H_q, while tensor parallel sharding or replication changes per-device allocation.
If T grows from 4096 to 16384, KV payload grows 4x, whereas dense prefill pair count grows approximately 16x.
Exact causal pairs are 4096*4097/2=8,390,656 and 16384*16385/2=134,225,920 per head and sequence.
Their ratio is approximately 15.9971 rather than exactly 16 because of the triangular boundary term.

| Operation | Time model | Additional storage | I/O or communication | Limits |
|---|---|---|---|---|
| Position rotation | Theta(B*N*T*H_q*d_r), plus key rotations. | Frequency tables or on-the-fly phases. | Kernel-specific phase reads. | Heads and partial rotation differ. |
| Dense prefill attention | Theta(B*N*H_q*T^2*d). | Naive scores quadratic; tiled kernels need not retain them. | Loads/stores depend on tiling. | Does not include projections or MLP. |
| Cached generation | Theta(B*N*H_q*d*(G*T+G*(G-1)/2)). | KV grows linearly with T+G. | Reads increasingly long cache. | No sliding-window eviction assumed. |
| Adaptation training | Depends on steps, sequence distribution and model. | Weights, optimizer, activations, temporary buffers. | Distributed gradient and activation traffic. | Do not infer wall-clock from asymptotic order. |
| Recovery | Reload artifacts and reconstruct or replay cache. | Checkpoint and replay buffer. | Storage/network and warmup. | Restored sessions must retain position policy. |

### 9b. Back-of-Envelope Sizing

The E4 case uses fictional BF16-like two-byte cache values and one 32-layer model; device, runtime and costs are unspecified.
At 4096 tokens its KV payload is 512 MiB; at 16384 it is 2048 MiB, with identical other dimensions.
Eight independent maximum-length sequences require 16 GiB of payload, before weights, activations, buffers, fragmentation, or reserve.
Variable lengths require summing per-sequence retained tokens rather than multiplying by the maximum if the allocator permits it.
A budget with 16 GiB available for all runtime allocations cannot accommodate that 16 GiB payload plus any other resident state.
An operator must measure peak allocated and reserved memory, actual batching, cache paging and tail latency on the target hardware.
Use successful task completions as the cost denominator; include training amortization, evaluation, hosting, failed calls and retrieval when estimating economics.
Currency, device prices, energy, throughput and actual cost per task are NOT MEASURED here.
**Takeaway:** A fourfold context extension has linear KV growth but approximately sixteenfold dense prefill attention work at fixed batch.

## 10. Key Theorems / Results Reference

| Result | Statement and hypotheses | Practical implication | Does not imply | Source |
|---|---|---|---|---|
| Rotation composition | Matching 2D rotations satisfy R(a)^T R(b)=R(b-a). | Derive relative positional contribution. | Whole-model translation invariance. | [P1] and direct trigonometric identity. |
| Norm preservation | Orthogonal R has ||R x||_2=||x||_2. | Rotation alone preserves pair norm. | Different rotated vectors keep their mutual dot. | Direct orthogonality. |
| RoPE construction | Rotate q and k with position-dependent phases to encode relative position in their interaction. | Match model implementation. | Perfect long-distance utilization. | [P1]. |
| Uniform PI | Map positions down by an extension scale for RoPE-based models, typically with adaptation. | Candidate for controlled context extension. | Same quality for every model and task. | [P2]. |
| PI experiments | Paper reports its own LLaMA-scale extension experiments. | Motivates a model-specific replication. | Reproduced results or transfer to a new corpus. | [P2], abstract; original experiments not reproduced. |

### Source registry and evidence boundaries

- **P1:** Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), arXiv, submitted 2021-04-20; abstract consulted 2026-11-07.
- **P1 claim:** RoPE uses absolute-position rotations with explicit relative-position dependency in attention; this sheet derives the identity independently.
- **P1 limitation:** The abstract alone is not a complete implementation specification for any particular checkpoint.
- **P2:** Chen et al., [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595), arXiv, submitted 2023-06-27; abstract consulted 2026-11-07.
- **P2 claim:** The authors report position interpolation and particular long-context fine-tuning experiments up to 32768 tokens.
- **P2 limitation:** Training details and empirical results are paper-specific; no independent replication is claimed here.
- **Original inventory:** `finetuning_course/Module_11_Context_Length_Extension` names five topics in `Train/Templates/directory.txt`.
- **Inventory limitation:** Original source contents are unavailable and the directory is not an empirical source.
- **Review trigger:** Recheck primary papers, target implementation and evaluation protocol when the model, scaling policy or evidence changes.
- **Evidence status:** Algebra is DERIVED; numerical examples are ILLUSTRATIVE; runtime measurements and independent technical review are OPEN.
**Takeaway:** Exact algebra and paper-specific empirical results answer different questions.
### Operational design: implementation and training controls

The following control cards describe distinct choices; exact API names are intentionally omitted because they depend on the pinned runtime.
Each card identifies a testable invariant and a practical consequence rather than an assertion that an implementation was exercised.
#### Checkpoint manifest
- Persist checkpoint revision, tokenizer hash and architecture config.
- Include rotary base, rotary dimension, pair layout and scaling policy.
- Compare these values with the training artifact before loading weights.
- A numerical match in parameter shape cannot validate the position convention.
#### Frequency construction
- Construct each frequency from the checkpoint’s actual formula and coordinate dimension.
- For the common schedule, j=0 has theta=1 radian per position.
- Higher j typically gives smaller theta when base exceeds one.
- Never replace the checkpoint schedule with a familiar default without an ablation.
#### Position numbering
- Define whether the first real token is logical position zero.
- Check position arrays after tokenization, padding and packing.
- Use the same convention during training, prefill and incremental decoding.
- A constant shared offset cancels in the isolated rotary dot but not necessarily in the full model.
#### Interleaved pairing
- An interleaved layout rotates coordinates (0,1), (2,3), and so on.
- The cosine and sine dimensions must be broadcast onto those pairs.
- Compare a hand-computed 2D pair against the kernel output.
- A split-half implementation pairs different coordinates unless weights and layouts are consistently transformed.
#### Split-half pairing
- One split-half convention pairs coordinate j with j+d_r/2.
- Verify sign convention and the exact ordering in the trained implementation.
- Use an arbitrary vector with unequal coordinates as a parity probe.
- An all-zero or symmetric vector can hide a pairing mistake.
#### Partial rotary tail
- Rotate only the configured d_r coordinates, where d_r is even.
- Pass the remaining d-d_r coordinates through unchanged.
- Check that the norm of the rotary subvector is preserved within tolerance.
- Using all d coordinates changes the learned model even when tensor dimensions still match.
#### Phase precision
- Compute m*theta and trigonometric functions in a validated numerical precision.
- Compare large-position angles and resulting logits with a higher-precision reference.
- Use the actual checkpoint frequencies and extreme target positions in the comparison.
- A float format with insufficient mantissa precision can collapse distinct integer positions.
#### Causal mask
- At query position m permit only authorized key positions n<=m.
- Test the first token, an interior token and a padded token explicitly.
- In a cache, use the absolute logical query index rather than the local decode-chunk offset.
- A correct rotary phase does not prevent future-token leakage from an incorrect mask.
#### Training loss mask
- Mark padding and intentionally unsupervised prompt tokens as excluded from the loss.
- Keep the attention mask conceptually separate from the label mask.
- Manually inspect a token-level example with prompt and answer spans.
- Masking labels does not prevent one packed example from attending to another.
#### Sequence packing
- Decide whether concatenated examples are independent sequences or a continuous document.
- For independent samples use an appropriate block-diagonal causal attention mask.
- Validate example boundaries, end-of-sequence tokens and position resets together.
- A position reset alone does not enforce independence.
#### Long-document provenance
- Carry document identifiers and source-span offsets through tokenization.
- Track which chunks were selected, ordered and included in each training sequence.
- Filter documents by license and access policy before they enter training.
- A longer sequence increases the scope of a provenance or permission error.
#### Long-context data mix
- Choose a distribution over lengths and tasks, not just a maximum length.
- Log both sampled sequence lengths and the share of supervised tokens by length.
- Reserve examples requiring evidence from distant parts of the sequence.
- Many short examples padded to a long shape do not create long-range supervision.
#### Short-context retention
- Keep a representative fraction of short examples if short usage matters.
- Hold optimizer, budget and data sources fixed in the relevant ablation.
- Compare the adapted model to its original baseline on untouched short tasks.
- A training mix is a mitigation to evaluate, not a guarantee of retained quality.
#### Effective optimization batch
- Record sequences per microbatch, accumulation steps and valid tokens per update.
- Variable sequence length makes sequences per update an incomplete batch description.
- Normalize loss consistently with the intended token or example weighting.
- A change in length distribution can change gradient weighting even at fixed nominal batch size.
#### Training mask correctness
- Construct a tiny sample whose future tokens contain a unique marker.
- Verify earlier logits do not change when only future token content changes.
- Disable dropout or control randomness during this diagnostic.
- This checks causality for the tested shape, not all backend paths.
#### Activation memory
- Budget saved activations and temporary workspaces separately from KV cache.
- Profile forward and backward peaks for the longest target training shape.
- Use validated activation recomputation if the compute tradeoff is acceptable.
- Inference KV sizing does not predict training peak memory.
#### Optimizer state
- Record which tensors are trainable and the precision of each optimizer state.
- Estimate weights, gradients and moments in separate categories.
- Check actual first-step allocation because optimizers can allocate lazily.
- A model loading successfully says nothing about whether its first training update fits.
#### Scale selection
- Define s from the exact original and target lengths used in the experiment.
- Save s with the artifact and make the denominator convention explicit.
- Compare with s=1 under the same tokenizer, corpus and evaluation.
- Selecting a scale based only on the desired marketing window bypasses quality evidence.
#### Adaptation baseline
- Start with the original checkpoint and a fixed training recipe.
- Include a no-adaptation PI arm if distinguishing mapping from training is important.
- Include an original-mapping arm at matched compute if feasible.
- Different data or training budgets make causal interpretation ambiguous.
#### Training continuity
- Save model, optimizer, scheduler, random generator and sampler state.
- Also save position policy, length distribution and packing configuration.
- On resume, compare the next batch positions and learning rate to a continuous run.
- Weights alone are insufficient for exact training continuation.
#### Checkpoint selection
- Use a validation set that was not used for gradient updates.
- Select using both short and long metrics with predeclared thresholds.
- Retain the original checkpoint for rollback and a final untouched test set.
- Repeated changes against the same test suite turn it into a development set.
#### Prefill position IDs
- Assign logical positions to each prompt token before applying rotations.
- For a full unpadded sequence they normally run from zero to T-1.
- Verify the final position and scaling factor on a tiny trace.
- Physical buffer offsets in a paged cache need not equal logical positions.
#### Incremental position IDs
- After prefill of T tokens, the next new token normally receives position T.
- A decode chunk with C new tokens advances logical positions T through T+C-1.
- Compare chunked decode with one-token decode using the same masking.
- Resetting positions to zero on every decode step destroys the intended relative phases.
#### Key-cache convention
- Determine whether the implementation stores already rotated keys or unrotated keys.
- If rotated keys are stored, their rotation policy becomes part of cache identity.
- Inspect the implementation contract and compare cache-enabled with cache-disabled logits.
- Do not rotate cached keys a second time unless the cache format explicitly requires it.
#### Value-cache convention
- Standard RoPE rotates queries and keys, not values, in the construction used here.
- Record architecture-specific departures rather than assuming every model is identical.
- Test value-cache layout and precision separately from key phase.
- A correct key rotation cannot repair a corrupted value cache.
#### Cache reuse
- Reuse a prefix only for identical token IDs, model revision and positional policy.
- Include tenant and authorization scope in the reuse decision.
- Ensure prefix position offsets and attention masks also match.
- A text string alone is not a sufficient cache key.
#### Dynamic position policy
- If scaling depends on final sequence length, determine that length before cache construction.
- Alternatively invalidate and rebuild keys whenever the mapping changes.
- Compare logits immediately before and after any policy transition.
- Updating only new queries or keys creates a mixed-position cache.
#### Prompt output budget
- Treat target length as a total limit for prompt and generated tokens.
- Reserve a maximum generation allowance before accepting input.
- Return a clear limit response when the request cannot fit.
- Silently dropping document text can produce confident unsupported answers.
#### Sliding-window serving
- A local attention window may evict or mask older cache entries.
- Identify whether the checkpoint was trained for that attention pattern.
- Measure tasks needing evidence beyond the window.
- A large configured position index does not restore evicted evidence.
#### Cache precision
- Select key and value storage precision independently of model weight storage where supported.
- Compare long-context output quality with an unquantized cache baseline.
- Measure actual memory and decode throughput under target concurrency.
- Reduced bytes per scalar do not include scale metadata or numerical error.
#### Paged allocation
- Track live tokens, reserved pages, internal fragmentation and sharing.
- Use page counts and page size rather than assuming ideal contiguous allocation.
- Exercise variable-length requests and cancellations.
- Logical KV payload is a lower-bound component, not an allocator peak.
#### Admission control
- Estimate worst-case prompt plus output cache reservation per accepted request.
- Account for in-flight requests, existing sessions and temporary prefill memory.
- Reject or queue work when the budget is exhausted.
- Assuming average request length protects neither peak memory nor tail latency.
#### Batch composition
- Bucket by prompt length where it improves utilization without unacceptable queueing.
- Measure the real distribution of padding, queue wait and kernel shapes.
- Separate throughput from individual-request latency.
- A bigger batch can improve tokens per second while making a deadline miss worse.
#### Long prefill scheduling
- Bound individual prefill work so a single request does not starve decode traffic.
- Measure user-facing time to first token and inter-token latency separately.
- Evaluate chunked prefill if the backend preserves the same causal semantics.
- A scheduling change must not alter logical positions or reuse the wrong cache.
#### Distributed placement
- Document whether KV heads and model layers are sharded or replicated.
- Count per-device memory after applying the actual placement.
- Measure collective communication and interconnect contention under long contexts.
- Dividing total KV bytes by device count is wrong if the cache is replicated.
#### Artifact rollout
- Package checkpoint, tokenizer, code version and position configuration together.
- Verify replicas have matching manifest digests before receiving traffic.
- Route a small controlled cohort and keep an immediate rollback path.
- A server configuration change can invalidate sessions created by the prior policy.
#### Session migration
- Attach artifact and cache-format identity to every live generation session.
- On a backend migration, reconstruct from tokens if the target cannot consume the cache safely.
- Check token offsets, masks and authorization after replay.
- Blindly deserializing an older cache can silently produce incorrect answers.
#### Observability
- Log lengths, version IDs, queue time, prefill time and cache usage without raw sensitive text.
- Report percentiles over named windows and request populations.
- Slice errors by actual prompt length and generated length.
- Aggregate means can hide endpoint degradation and long-request tail latency.
#### Privacy controls
- Minimize stored prompts, cached prefixes and document-derived artifacts.
- Apply tenant-scoped expiration, deletion and access control.
- Ensure diagnostic traces and benchmark corpora follow the same policy.
- A longer prompt can contain more sensitive data even when model weights are unchanged.
#### Tool safety
- Treat document content as untrusted data, even when it appears in the context window.
- Authorize each tool action against external policy and user identity.
- Test that document instructions cannot grant themselves permissions.
- A position encoding change is not a prompt-injection defense.
#### Tokenization drift
- Pin tokenizer revision and special-token behavior to the checkpoint.
- Measure actual token counts after all system and tool wrappers are added.
- Retest endpoint behavior when tokenizer or prompt template changes.
- Character count is not a reliable context-length budget.
#### Numerical comparison
- Use deterministic settings where feasible and set justified tolerance for logits.
- Compare maximum absolute and relative errors as well as final generated tokens.
- Investigate discrepancies that grow with position or decode length.
- Identical generated tokens alone can hide substantial logit drift.
#### Boundary indexing
- For length L, valid zero-based positions end at L-1.
- Under E3, the target last coordinate is (L_target-1)/s.
- Check both T=L_target and a rejected T=L_target+1 request.
- Equating a count with the final index causes off-by-one errors.
#### Frequency coverage
- Inspect the phase changes across representative rotary pairs.
- Fast and slow frequencies respond differently to the same positional distance.
- Plot phase modulo 2*pi only as a visualization, retaining the raw phase in checks.
- An individual pair wraps periodically and cannot identify an arbitrarily large distance uniquely.
#### Quality release gate
- Define minimum acceptable short and long task performance before running the final test.
- Include uncertainty and safety, not just a pooled mean score.
- Predefine what happens when a critical slice has too few examples.
- A pass on one benchmark cannot compensate for a failed mandatory safety constraint.
### Evaluation design: task-specific checks

The entries below specify distinct measurements or controls; none is an observed benchmark result.
Split by document, customer, source or time as appropriate so related material cannot leak across development and final evaluation.
#### Length bins
- Stratify actual tokenized inputs into short, original-window, intermediate and near-target bins.
- Report sample count and quality in each bin rather than only their weighted average.
- A sparse endpoint bin needs more examples or a visibly uncertain conclusion.
#### Evidence depth
- Move identical relevant facts among early, middle and late positions while preserving the rest of the task.
- Record depth as both token index and fraction of actual sequence length.
- Position effects can interact with document structure and should not be interpreted as a universal curve.
#### Distance to answer
- Vary the distance between the query and the supporting fact at fixed total length.
- Keep distractors and question wording controlled when possible.
- This isolates a different factor from merely increasing total length.
#### Multiple needles
- Place several relevant facts in different documents or positions.
- Require a response that uses all facts and score each required element.
- Single-fact success does not establish multi-fact aggregation.
#### Multi-hop reasoning
- Ask a question whose answer requires a chain across separated passages.
- Score supporting facts and the final inference independently.
- An answer may be correct by chance even when the model missed a required step.
#### Contradictory evidence
- Include sources with conflicting statements and timestamps.
- Define the expected precedence rule before scoring.
- A benchmark must not silently reward guessing when evidence is ambiguous.
#### No-answer control
- Include examples where the answer is absent from all provided material.
- Measure unsupported assertions and appropriate abstention separately.
- A model that always abstains can score well on hallucination but poorly on usefulness.
#### Citation correctness
- Require document identifier and exact supporting span for a factual claim.
- Check that the cited span entails the claim rather than merely containing similar words.
- Correct final answers with invalid citations are not fully grounded.
#### Distractor similarity
- Add semantically similar but incorrect facts at non-answer positions.
- Compare against matched tasks without those distractors.
- A relevance improvement must not come from silently reducing task difficulty.
#### Order sensitivity
- Permute document ordering while retaining the same evidence and question.
- Measure both answer changes and citation changes across permutations.
- If temporal order is semantically important, preserve or explicitly encode it.
#### Context assembly
- Save a reproducible manifest of document IDs, order, token counts and truncation.
- Verify the gold supporting span actually survived assembly.
- Otherwise retrieval or preprocessing failure can be mistaken for model reasoning failure.
#### Retrieval recall
- For retrieval-based baselines, measure whether authorized gold evidence enters the prompt.
- Report recall separately from answer accuracy conditioned on evidence presence.
- Oracle context is a diagnostic upper control, not a deployable retrieval score.
#### Reasoning conditional on evidence
- Score model answers only on examples where required evidence is present.
- Compare this conditional result to end-to-end success over all examples.
- Conditioning changes the evaluated population and must be reported explicitly.
#### Document-group split
- Assign related document versions and near-duplicates to the same split.
- Deduplicate using an appropriate combination of metadata and content similarity.
- A random chunk split can leak the same underlying document into train and test.
#### Temporal split
- Train and tune on records available before a cutoff and evaluate on later records.
- Record the document creation and actual availability times separately.
- Future-revised documents can leak facts even if their original creation date is old.
#### Tenant split
- When generalization to new customers is required, hold out whole tenants.
- When measuring within-tenant generalization, explicitly state that narrower estimand.
- A high pooled score can hide a low-resource tenant failure.
#### Contamination audit
- Search benchmark prompts and answers in training and adaptation data where permitted.
- Document limitations of matching and inaccessible pretraining corpora.
- No match found is not proof that the benchmark was never seen.
#### Prompt-template control
- Use the same system instructions and answer format across candidates where compatible.
- Record model-specific required wrappers as potential confounders.
- A performance difference can reflect formatting rather than position scaling.
#### Tokenizer fairness
- Report lengths in each model tokenizer as well as source-document units.
- Keep access to the same underlying documents and facts.
- Equal token counts do not necessarily contain equal text across tokenizers.
#### Decoding control
- Fix sampling settings, generation limits and random seeds where applicable.
- For deterministic comparisons, document backend nondeterminism and precision.
- Different sampling temperatures confound a checkpoint comparison.
#### Paired comparison
- Evaluate both models on the same independent evaluation units.
- Compute per-unit score differences and an interval appropriate to the sampling unit.
- Treating many correlated questions from one document as independent understates uncertainty.
#### Cluster uncertainty
- Bootstrap or otherwise model dependence at the document or customer level.
- Report the number of independent clusters, not just total questions.
- A narrow interval based on token-level observations can be misleading.
#### Slice multiplicity
- Predeclare critical slices and their decision thresholds.
- Show all evaluated slices and account for exploratory multiple comparisons.
- Selecting only the most favorable length or depth slice inflates apparent success.
#### Scoring rubric
- Specify exact match, semantic equivalence, partial credit and abstention rules.
- Use blinded human review or validated task-specific checks for ambiguous answers.
- A judge model can share systematic errors with the evaluated model.
#### Judge calibration
- Compare an automated judge to independent human labels on a stratified sample.
- Measure disagreement by length, language and answer style.
- A single overall agreement number may conceal failures on long answers.
#### Human review reliability
- Define reviewer instructions and double-score a subset of examples.
- Resolve disagreements with an explicit adjudication procedure.
- Annotator agreement is not a substitute for checking source evidence.
#### Perplexity protocol
- Compute held-out token-level negative log likelihood with the stated masking and tokenizer.
- Specify context windows, stride and aggregation so comparisons are reproducible.
- Perplexity is not equivalent to grounded task completion.
#### Short-task regression
- Compare pre-extension and adapted checkpoints on untouched short tasks.
- Use paired examples and publish changes with uncertainty.
- A gain on long retrieval does not cancel a mandatory short-task failure.
#### Latency measurement
- Measure time to first token, inter-token latency and end-to-end latency separately.
- Report percentiles by prompt size, output length, load and concurrency.
- Do not infer p99 latency by summing individual component p99 values.
#### Throughput measurement
- Measure successful requests and generated tokens per second at a defined workload.
- Record hardware, software versions, precision, batching and request mix.
- Throughput under saturation is not a latency guarantee.
#### Memory measurement
- Record peak allocated and reserved device memory for prefill and decode.
- Include allocator slack, runtime workspace and concurrent requests.
- Payload-only E4 is not a complete production memory measurement.
#### Cost per success
- Divide all included serving and amortized training costs by successful task completions.
- Define success using the same task-quality and safety criteria across alternatives.
- Tokens per dollar alone can reward fast but incorrect answers.
#### Stress and cancellation
- Exercise long-running requests, cancellation, retries and concurrent bursts.
- Check that cache pages are reclaimed and duplicate actions are controlled.
- A steady-state isolated request misses resource exhaustion under contention.
#### Access control
- Use synthetic cross-tenant documents and attempt unauthorized retrieval or prefix reuse.
- Verify external authorization before evidence reaches the model.
- A correct model answer on public test data does not establish isolation.
#### Prompt injection
- Place adversarial instructions in a document that should be treated as data.
- Score both instruction resistance and whether authorized evidence is still used.
- Never execute dangerous tool actions merely to assess model behavior.
#### Language and format
- Stratify by language, document structure, code, tables and noisy extraction.
- Inspect whether tokenization and parsing systematically drop important spans.
- English prose performance need not transfer to other representations.
#### Long-document summarization
- Define coverage, factual consistency and attribution against source documents.
- Include salient information at several depths and require human-audited examples.
- A fluent summary is not necessarily complete or faithful.
#### Decision-oriented task
- For a hypothetical contract-review workflow, require identification of every relevant clause.
- Score missed clauses, false alarms and cited support separately.
- Do not present this constructed workflow as a measured legal outcome.
#### Release decision
- Freeze the suite, thresholds and owner before exposing the final holdout.
- Report failures, uncertainty, exclusions and rollback conditions alongside passes.
- If the required sample size is unavailable, state the result as inconclusive.
#### Post-release drift
- Track input length, document type and task mix over time without storing unnecessary content.
- Re-evaluate if the workload moves outside the validated distribution.
- An initially valid benchmark cannot certify a changed environment forever.
### Reference implementation: language-neutral pseudocode

This is an algorithm specification, not a claim that a particular runtime has been tested.
The routine assumes one interleaved rotary pair per adjacent coordinate pair.
Inputs are a real vector x of even rotary dimension, scalar position p, positive scale s and frequency vector theta.
The checkpoint may require split-half pairing, partial rotation, a different frequency schedule or different numerical precision.

```text
function rotate_interleaved(x, p, s, theta):
    assert length(x) is even
    assert length(theta) == length(x) / 2
    assert s > 0
    output = same_shape_as(x)
    for j from 0 to length(theta)-1:
        angle = (p / s) * theta[j]
        c = cos(angle)
        z = sin(angle)
        a = x[2*j]
        b = x[2*j+1]
        output[2*j]   = a*c - b*z
        output[2*j+1] = a*z + b*c
    return output
```

The loop terminates because it visits exactly d_r/2 pairs.
For each pair, both outputs are computed from the original a and b, preventing an in-place overwrite from corrupting the second coordinate.
If modifying x in place, read both a and b into temporaries before either write.
A nonrotary tail should be copied unchanged by the caller or handled explicitly.
The operation count per token/head is linear in d_r, excluding implementation-specific trigonometric cost and lookup-table preparation.
Frequency-table memory depends on whether phases are precomputed for all positions, computed per batch, or fused into a kernel.
A table indexed only by integer positions cannot directly represent p/s unless it stores the correct scaled phases or interpolates according to the implementation contract.
Precomputation must use the same scale and frequency vector as the model consuming the table.
For the worked 2D case, rotate_interleaved([1,0], 4, 4, [pi/2]) returns [0,1] up to floating-point error.
For x=[3,4], p=1, s=1, theta=[pi/2], it returns [-4,3], whose squared norm remains 25.
The inverse rotation is obtained by negating the angle, not by negating the vector.
A robust reference comparison tests different positions, mixed signs, each frequency pair and at least one nonzero key position.
When comparing to fused code, check shapes, strides, offsets, padding and error tolerance independently.
For models using split-half layout this pseudocode is not a drop-in implementation.
**Takeaway:** Checkpoint parity requires matching coordinate pairing, frequency, scale, positions, precision and cache semantics together.

## 11. Interview / Exam Questions

Answers below use the equations and assumptions in this sheet; examples are constructed, not production measurements.
Scoring for each question: 0 incorrect, 1 partial, 2 correct under assumptions, 3 correct with defended tradeoffs or counterexample.
### Q01 — L1 / Concept
- **Prompt:** What does RoPE add to attention?
- **Assumptions:** Query and key use matching rotary frequencies, with content vectors held fixed for the identity.
- **Answer, first step:** For a head, rotate query and key coordinates by position-dependent angles before their dot product.
- **Answer, next step:** Identical frequency blocks make their interaction depend on relative position as in E1.
- **Answer, conclusion:** It does not add a fixed scalar position embedding to the value vector.
- **Follow-up / boundary:** Explain why content-dependent q and k still make the complete network more than a function of distance.
- **Red flag:** Claiming every model using RoPE is globally translation-invariant ignores masks, content and boundaries.
- **Scoring focus:** Full credit distinguishes the rotated pair interaction from the whole network's behavior.
- **Answer source:** E1 and the counterexample in Section 4.4.
### Q02 — L1 / Definition
- **Prompt:** How do trained, configured and evaluated lengths differ?
- **Assumptions:** A checkpoint can have multiple training phases and an independently configured serving limit.
- **Answer, first step:** The trained length describes the sequence regime used during a named training phase.
- **Answer, next step:** The configured limit is the maximum accepted by a particular runtime policy.
- **Answer, conclusion:** The evaluated length has measured performance on a specified task and population.
- **Follow-up / boundary:** A checkpoint may have unknown training distribution even when a configuration file contains a maximum.
- **Red flag:** A successful API call at a length is not evidence that the model can use all of its content.
- **Scoring focus:** Full credit identifies the evaluation task and distribution, not just its maximum length.
- **Answer source:** Trained, target and utilization definitions in Section 3.
### Q03 — L1 / Mapping
- **Prompt:** What is uniform position interpolation?
- **Assumptions:** Positions are zero-based and uniform PI uses the length-ratio convention E3.
- **Answer, first step:** It applies p(m)=m/s to logical position m before the rotary angle is computed.
- **Answer, next step:** For L_train=4096 and L_target=16384, s=4 under E3.
- **Answer, conclusion:** This maps index 16383 to 4095.75, not to the last original integer index 4095.
- **Follow-up / boundary:** The token sequence itself remains 16384 tokens long.
- **Red flag:** Describing PI as token compression incorrectly predicts lower dense attention cost.
- **Scoring focus:** Full credit distinguishes the final coordinate from the original final integer index.
- **Answer source:** E3 and its endpoint boundary.
### Q04 — L2 / Derivation
- **Prompt:** Derive the relative-position dot-product identity.
- **Assumptions:** Both rotations use the same real two-dimensional orthogonal rotation convention.
- **Answer, first step:** Start with (R(m theta)q)^T(R(n theta)k).
- **Answer, next step:** Associativity gives q^T R(m theta)^T R(n theta) k.
- **Answer, conclusion:** Orthogonality and angle addition give R(m theta)^T R(n theta)=R((n-m)theta).
- **Follow-up / boundary:** The result requires the same rotation convention and frequency for the two vectors.
- **Red flag:** Swapping n-m to m-n without also changing rotation convention can reverse sine terms.
- **Scoring focus:** Full credit explains both transpose and angle composition.
- **Answer source:** E1 in Section 4.1.
### Q05 — L2 / Arithmetic
- **Prompt:** Compute the rotated vector [3,4] at a quarter-turn.
- **Assumptions:** The specified positive rotation is counterclockwise in the convention of E1.
- **Answer, first step:** Use R(pi/2)=[[0,-1],[1,0]].
- **Answer, next step:** The result is [-4,3].
- **Answer, conclusion:** The original and rotated squared norms are both 25.
- **Follow-up / boundary:** The answer is a single-pair computation, not the attention output.
- **Red flag:** In-place updating the first coordinate before computing the second may yield the wrong result.
- **Scoring focus:** Full credit computes both coordinates and checks norm preservation.
- **Answer source:** Rotation definition in Section 4.1.
### Q06 — L2 / Attention
- **Prompt:** Compute an unscaled pair score for q=k=[1,0], theta=pi/2, m=4, n=2 with s=4.
- **Assumptions:** Both q and k use the same interpolated coordinate mapping and one rotary pair.
- **Answer, first step:** The relative coordinate is (n-m)/s=-1/2.
- **Answer, next step:** The relative angle is -pi/4.
- **Answer, conclusion:** The dot product is cos(-pi/4)=sqrt(2)/2, approximately 0.7071.
- **Follow-up / boundary:** If this were the only pair of a d=2 head, divide by sqrt(2) to get 0.5 before masking and softmax.
- **Red flag:** Calling 0.7071 a probability is wrong: softmax depends on all unmasked keys.
- **Scoring focus:** Full credit applies the negative relative angle and distinguishes logit from probability.
- **Answer source:** E1, E3 and Section 5.
### Q07 — L2 / Capacity
- **Prompt:** Compute KV payload for B=1, N=32, H_kv=8, T=16384, d=128, b=2.
- **Assumptions:** Every layer uses the same KV shape and both keys and values are resident with no sharing.
- **Answer, first step:** Use E4: 2*1*32*8*16384*128*2 bytes.
- **Answer, next step:** That equals 2,147,483,648 bytes or 2 GiB.
- **Answer, conclusion:** Four independent equal-size requests require 8 GiB payload without sharing.
- **Follow-up / boundary:** Add weights, workspaces and allocator overhead before deciding whether a device fits.
- **Red flag:** Using H_q instead of H_kv overstates cache size for grouped-query attention.
- **Scoring focus:** Full credit includes both KV tensors and converts bytes using 2^30 bytes per GiB.
- **Answer source:** E4 in Section 9.
### Q08 — L2 / Scaling
- **Prompt:** How does fourfold sequence length affect dense causal attention and cache?
- **Assumptions:** Batch, architecture and dense causal attention remain fixed while context length changes.
- **Answer, first step:** The KV payload scales fourfold at fixed batch and architecture.
- **Answer, next step:** Causal pair counts change from 8,390,656 at 4096 to 134,225,920 at 16384.
- **Answer, conclusion:** Their ratio is approximately 15.9971, close to sixteen.
- **Follow-up / boundary:** Efficient attention may avoid quadratic score storage while retaining dense quadratic pairwise work.
- **Red flag:** Predicting sixteenfold latency without measurement ignores kernels, projections, batching and hardware.
- **Scoring focus:** Full credit separates attention-pair complexity, KV capacity and observed latency.
- **Answer source:** Causal pair count and E4 in Section 9.
### Q09 — L3 / Selection
- **Prompt:** When should a team use retrieval instead of extending the window?
- **Assumptions:** Both systems receive authorized access to the same underlying evidence.
- **Answer, first step:** Use retrieval when relevant evidence can be found reliably and a smaller context meets the task requirement.
- **Answer, next step:** Compare authorized evidence recall, grounded answer accuracy, latency and full system cost.
- **Answer, conclusion:** If the task requires interactions across the entire corpus, retrieval misses may be disqualifying.
- **Follow-up / boundary:** A native long-context model and original checkpoint are additional baselines.
- **Red flag:** Calling retrieval universally superior ignores task recall and index freshness.
- **Scoring focus:** Full credit weighs retrieval misses against measured context-extension costs.
- **Answer source:** Selection table and evaluation controls in Sections 7 and 10.
### Q10 — L3 / Debug
- **Prompt:** Cache-enabled decoding disagrees with uncached inference; what next?
- **Assumptions:** Token IDs and generation settings are identical between cached and uncached paths.
- **Answer, first step:** Freeze tokens and decoding settings, then compare stepwise logits and the first divergent position.
- **Answer, next step:** Inspect position IDs, scaling factor, mask, rotary pairing and whether keys were already rotated.
- **Answer, conclusion:** Repeat with a nonzero prefix position and varied batch padding.
- **Follow-up / boundary:** Allow expected small precision differences, but investigate growing position-dependent errors.
- **Red flag:** Retraining the model before checking cache parity fails to isolate the implementation bug.
- **Scoring focus:** Full credit locates the first divergence before proposing a bounded correction.
- **Answer source:** F03 and cache implementation controls.
### Q11 — L3 / Policy transition
- **Prompt:** What happens if a dynamic scale changes halfway through a generation session?
- **Assumptions:** The cached representation contains keys already rotated with the initial scale.
- **Answer, first step:** Previously rotated cached keys retain the old angular mapping.
- **Answer, next step:** Newly rotated queries then interact with keys under inconsistent position policies.
- **Answer, conclusion:** Pin the policy for the session or rebuild all affected keys using a consistent mapping.
- **Follow-up / boundary:** Tag cache entries with model, tokenizer, scale, offset and authorization context.
- **Red flag:** Changing only the new token positions does not retroactively update stored keys.
- **Scoring focus:** Full credit identifies the mixed-policy interaction and a valid cache-rebuild alternative.
- **Answer source:** F05 and dynamic position policy controls.
### Q12 — L3 / Evaluation
- **Prompt:** How would you test a long-context quality claim?
- **Assumptions:** The desired inference concerns deployment tasks rather than only synthetic retrieval.
- **Answer, first step:** Predefine real task outcomes and independent evaluation units, plus short and long length bins.
- **Answer, next step:** Vary evidence position, distance, distractors and multi-hop requirements.
- **Answer, conclusion:** Use paired comparisons and cluster-aware uncertainty at document or customer level.
- **Follow-up / boundary:** Report missing evidence, failure slices, citation accuracy and an untouched final holdout.
- **Red flag:** A single passkey at one position proves only performance on that specific synthetic case.
- **Scoring focus:** Full credit describes independent units, depth slices and a held-out decision gate.
- **Answer source:** Evaluation design controls in Section 10.
### Q13 — L3 / Change constraint
- **Prompt:** What changes if training budget is unavailable?
- **Assumptions:** No additional adaptation training is available, but baseline inference and evaluation are feasible.
- **Answer, first step:** Evaluate the unchanged baseline, a no-training PI ablation, retrieval and an appropriate native model.
- **Answer, next step:** Apply the same task, access and latency requirements to each.
- **Answer, conclusion:** Treat zero-shot PI as an experiment, not as a reproduced adaptation result.
- **Follow-up / boundary:** If all options fail a hard constraint, reduce the advertised supported workload or obtain resources.
- **Red flag:** Silently shipping extrapolation because the model accepts a request bypasses the quality gate.
- **Scoring focus:** Full credit keeps zero-shot PI distinct from the adapted PI paper result.
- **Answer source:** Selection table in Section 7 and P2 evidence registry.
### Q14 — L3 / Safety
- **Prompt:** Does a longer context solve prompt injection?
- **Assumptions:** A retrieved document is untrusted and tool authorization is enforced outside the model.
- **Answer, first step:** No: adversarial instructions in a document remain untrusted data.
- **Answer, next step:** Authorize retrieval and tools outside the model using tenant-scoped policy.
- **Answer, conclusion:** Test canaries, inappropriate tool calls, citation grounding and cross-tenant leakage.
- **Follow-up / boundary:** Maintain minimum logging and retention rules for sensitive text.
- **Red flag:** Treating a document instruction as an authority escalation is a security failure regardless of its position.
- **Scoring focus:** Full credit requires a concrete enforcement or negative-test mechanism.
- **Answer source:** F12 and security controls in Sections 8 and 10.
### Q15 — L3 / Evidence
- **Prompt:** A paper reports a long-window score: what can you conclude?
- **Assumptions:** The cited paper and its evaluation protocol are available for inspection.
- **Answer, first step:** Identify the exact model, data, task, metric, training steps and baseline in the original report.
- **Answer, next step:** Treat the score as author-reported for that setup unless independently reproduced.
- **Answer, conclusion:** Replicate on representative held-out tasks and include short-context regression and cost.
- **Follow-up / boundary:** If the report or setup is unavailable, mark the corresponding result unverified.
- **Red flag:** A paper-specific upper bound or benchmark does not establish universal production reliability.
- **Scoring focus:** Full credit separates author-reported findings from local replication and transfer.
- **Answer source:** Primary-paper registry in Section 10.
### Additional quantitative and operational exercises

Each exercise is ILLUSTRATIVE and applies only under its explicitly stated assumptions.
The calculations are component estimates, not observed production results.
#### Position zero
- Set m=0, s=4 and any finite frequency theta.
- The scaled coordinate is 0/4=0.
- The rotation angle is zero and R(0) is the identity.
- This is why a position-zero-only cache test can miss a scaling mismatch.
- Use nonzero positions to test the rest of the mapping.
#### First token separation
- Take consecutive indices m=10 and n=9 with s=4.
- Their scaled separation is (9-10)/4=-0.25.
- At theta=2 radians per position their relative angle is -0.5 radians.
- Without scaling the relative angle is -2 radians.
- Even nearby tokens receive a different phase under uniform interpolation.
#### Target endpoint
- Let L_train=2048 and L_target=8192, so s=4.
- The last valid target index is 8191.
- Its mapped coordinate is 8191/4=2047.75.
- This remains below the length 2048 but exceeds last integer index 2047.
- The distinction matters when checking an integer-indexed lookup table.
#### Endpoint-preserving variant
- Use original length 5 and target length 9, both greater than one.
- The endpoint-preserving multiplier is (5-1)/(9-1)=0.5.
- Target index 8 maps to original index 4 exactly.
- The E3 multiplier 5/9 instead maps index 8 to 40/9, approximately 4.4444.
- Both are different conventions; checkpoint compatibility determines which is applicable.
#### No extension
- Set L_train=L_target=4096.
- Then s=4096/4096=1.
- Every integer m maps to itself under E3.
- Any difference in output must come from another change or numerical implementation detail.
- This identity case is a useful baseline for position-scaling code.
#### Phase wrap
- Take theta=pi/2 and relative displacement four positions.
- The relative angle is 4*pi/2=2*pi.
- For a unit vector on the first axis, its rotated image returns to [1,0].
- That single pair alone cannot distinguish displacement four from zero.
- Multiple rotary frequencies and token content provide additional information, not a unique-distance guarantee.
#### Common shift
- Let m=3, n=1, theta=pi/2 and shift both positions by c=7.
- The original difference is n-m=-2.
- The shifted difference is (1+7)-(3+7)=-2.
- Both isolated rotary dot products use angle -pi.
- Other model components or changes to content can still change full-model output.
#### Mismatched scale
- Let q=k=[1,0], theta=pi/2, m=4 and n=2.
- With s=4 for both, the relative angle is -pi/4 and the dot is sqrt(2)/2.
- With query scale 4 and key scale 1, the relative angle is pi/2 and the dot is zero.
- The difference arises solely from inconsistent mappings.
- A cached-key parity test should include this kind of nonzero-position case.
#### Two frequencies
- Use q=k=[1,0,1,0], d=4, displacement one and frequencies pi/2 and pi.
- The pair contributions are cos(pi/2)=0 and cos(pi)=-1.
- The unscaled sum is -1.
- Standard sqrt(d) scaling divides by 2, yielding -0.5.
- Negative logits are valid; softmax probabilities depend on all key logits.
#### Nonrotary tail
- Let d=4, d_r=2, q=k=[1,0,2,3], and rotary angle pi/2.
- The rotated-pair contribution is zero for the first two coordinates.
- The unchanged tail contributes 2*2+3*3=13.
- The score before masking is 13/sqrt(4)=6.5.
- Ignoring the unrotated tail would produce a fundamentally different score.
#### Padding invariant
- A two-token example has logical positions 0 and 1.
- If left-padded by three physical slots, its physical array offsets become 3 and 4.
- Its logical positions should still be 0 and 1 when the architecture uses that convention.
- Compare the item alone and in a padded batch using the same valid tokens.
- Masks and checkpoint-specific position semantics must also match.
#### Prompt reservation
- A service caps total sequence length at 16384 tokens.
- If 2048 tokens are reserved for output, the prompt budget is 14336 tokens.
- A prompt of 15000 tokens plus the reservation requires 17048 tokens.
- The request exceeds the cap by 664 tokens.
- Reject or apply a disclosed task-approved reduction rather than silently discarding evidence.
#### Growing cache
- Use the E4 model with one sequence and two-byte cache elements.
- At T=4096, payload is 536870912 bytes, or 512 MiB.
- At T=8192, payload is 1073741824 bytes, or 1 GiB.
- At T=16384, payload is 2147483648 bytes, or 2 GiB.
- The linear relationship assumes all cached entries remain resident and dimensions are fixed.
#### Grouped-query effect
- Keep every E4 dimension fixed except H_kv.
- With H_kv=8, the 16384-token example uses 2 GiB payload.
- With H_kv=4, it uses 1 GiB payload.
- The ratio is exactly one half under identical element precision and cache policy.
- Query-head count and total model computation need not halve.
#### Cache precision effect
- Use the E4 example but assume one byte per cached scalar.
- Its payload becomes 1073741824 bytes, or 1 GiB.
- That is half the two-byte payload before quantization metadata.
- Actual quantization error and memory overhead require measurement.
- The calculation does not recommend a particular cache format.
#### Unequal requests
- Two independent sequences have retained lengths 4096 and 16384.
- Their lengths sum to 20480 tokens.
- The ideal payload is 512 MiB + 2048 MiB = 2560 MiB.
- A fixed rectangular allocation to the maximum could instead reserve 2*2048=4096 MiB.
- The actual allocator determines whether variable-length savings are realized.
#### Page rounding
- Assume one cache page holds 256 tokens for a sequence.
- A 513-token sequence needs ceil(513/256)=3 pages.
- Three pages provide capacity for 768 tokens, leaving 255 token slots unused.
- That is page-level internal fragmentation, not useful sequence content.
- Actual page sharing and metadata depend on the serving implementation.
#### Causal pairs
- For T=4, list causal pair counts per query: 1, 2, 3 and 4.
- Their sum is 10.
- The closed form T(T+1)/2 gives 4*5/2=10.
- For T=8 the corresponding count is 36, not 40.
- The finite-length ratio is 3.6 rather than exactly the asymptotic factor 4.
#### Decode attention
- Assume a cached prompt of T=4 and generate G=3 tokens.
- If each generation step attends to the prefix lengths T, T+1 and T+2, the counts are 4, 5 and 6.
- Their sum is 15.
- The expression G*T+G*(G-1)/2 gives 3*4+3=15.
- Other conventions for when a generated token is inserted require consistent indexing.
#### Training weighting
- Suppose a batch has one 100-token example and one 1000-token example, all supervised.
- A token-average objective gives the longer example 1000/1100 of the total token weight.
- That fraction is approximately 0.9091.
- An equal-example average instead weights each example 0.5.
- Neither objective is automatically correct; document the intended estimand.
#### Short-regression gate
- Suppose a fictional paired test has 100 independent examples.
- The original model answers 80 correctly and the new model answers 76 correctly.
- Their observed accuracy difference is -4/100=-0.04, or -4 percentage points.
- The paired outcomes and uncertainty are still needed to judge the change.
- Aggregate counts alone cannot determine a paired confidence interval.
#### Evidence survival
- A dataset contains 100 questions with one gold support span each.
- If assembly retains support for 90 questions, support inclusion is 90%.
- Suppose the model answers 72 of those 90 correctly.
- Conditional accuracy is 72/90=80%, while end-to-end accuracy is 72/100=72%.
- Both metrics are needed; neither proves performance on another distribution.
#### Cluster dependence
- Consider 100 questions drawn evenly from 10 documents.
- There are ten questions per document but only ten document clusters.
- Document-specific difficulty can correlate answers within each cluster.
- A document-level resampling procedure preserves that clustering structure.
- The number of questions alone is not the effective independent sample size.
#### Document depth
- For a 10000-token prompt, suppose a fact starts at token index 2500.
- Its normalized starting depth is 2500/10000=0.25.
- Moving the same fact to index 7500 gives depth 0.75.
- Keep question wording and other evidence fixed when attributing a change to depth.
- Different tokenizers or prompt wrappers change the measured index.
#### Latency percentiles
- Suppose one request spends 1 second in prefill and 2 seconds decoding.
- Its end-to-end time is 3 seconds if the phases are sequential and no other time is included.
- Across many requests, the p99 of their sums need not equal the sum of component p99s.
- Request-level paired measurements determine the end-to-end distribution.
- Queueing and network time must be included when they affect the user.
#### Cost denominator
- A fictional evaluation spends 100 cost units and has 80 successful tasks.
- Its cost per success is 100/80=1.25 cost units.
- If a faster alternative costs 80 units but succeeds on only 40 tasks, its cost per success is 2.
- This ignores unlisted fixed, training and human-review costs in both alternatives.
- Cost optimization must retain the same success and safety definitions.
#### Access boundary
- Assume two tenants submit identical token prefixes under different permissions.
- The visible token strings may match exactly.
- Their authorization contexts still differ.
- A shared cache hit must be permitted by explicit isolation policy, not inferred from text equality.
- Test revoked and changed permissions as well as initial access.
#### Scale transition
- Suppose cached keys use s=4 through position 100.
- At position 101 a new query is accidentally rotated with s=8.
- For a key at position 100 the angle is based on 100/4-101/8, not (100-101)/4.
- The intended and mixed-policy angles differ for every nonzero frequency.
- Pin the scale or regenerate the affected cache before continuing.
#### Position counter overflow
- Assume an implementation stores logical position in an unsigned 16-bit integer.
- The maximum representable value is 65535.
- Incrementing beyond that value cannot be represented in this type.
- Depending on language and implementation, overflow may wrap or trigger an error.
- Choose a suitable position type and test the actual supported maximum.
#### Sensitivity to concurrency
- For the 2 GiB maximum-length cache payload, consider six independent sessions.
- The ideal payload is 6*2=12 GiB.
- On a hypothetical 16 GiB device, only 4 GiB remains before all other allocations.
- Weights, activations and workspace may already exceed that remainder.
- Admission must use measured peak headroom rather than this payload estimate alone.
#### Softmax and masking trace
- Consider one query with three key logits `[0, 0, 100]`.
- Assume the first two keys are allowed and the third is masked as a future token.
- Masking before softmax replaces the third logit with negative infinity.
- The remaining exponentials are `[1, 1, 0]`, so the attention probabilities are `[0.5, 0.5, 0]`.
- If the third key is accidentally unmasked, its large logit dominates the distribution.
- This shows why a correct rotary implementation alone cannot establish causal correctness.
#### Relative score versus probability
- Suppose a query has two allowed key logits `[0, 0]` after rotary processing and scaling.
- Their softmax probabilities are `[0.5, 0.5]`.
- Increase only the first logit to `ln(3)`; the exponentials become `[3, 1]`.
- The corresponding probabilities are `[0.75, 0.25]`.
- The probability assigned to one key depends on the other eligible key logits.
- Therefore, a single pairwise RoPE score cannot by itself determine an attention probability.
#### Cache prefix sharing
- Suppose four requests share an authorized 4096-token prefix and each has a distinct 4096-token continuation.
- Without sharing, retained token copies total `4*(4096+4096)=32768`.
- With ideal immutable prefix sharing, token copies total `4096+4*4096=20480`.
- Under the same E4 model and precision, that reduces the illustrative payload from 4 GiB to 2.5 GiB.
- Actual savings depend on page boundaries, copy-on-write behavior and metadata.
- Sharing is permitted only when model, tokens, policy and authorization all match.
#### Long-context loss aggregation
- Suppose one evaluation sequence has 100 supervised tokens with mean negative log likelihood 1.
- A second has 900 supervised tokens with mean negative log likelihood 2.
- Token-weighted mean loss is `(100*1+900*2)/1000=1.9`.
- Equal-sequence mean loss is `(1+2)/2=1.5`.
- The corresponding exponentiated quantities differ because the weighting estimands differ.
- Report the averaging convention and tokenization before comparing perplexity across runs.
#### Recovery cost
- Suppose a session has 12000 retained prompt tokens and its cache cannot be transferred safely to a replacement replica.
- A correct fallback can replay the retained token IDs under the original model and position policy.
- The replacement rebuilds the corresponding cache before resuming the next token.
- Replay incurs prefill work and can increase time to the next token.
- A different checkpoint or scaling policy requires a new session or an explicitly validated migration.
- A serialized cache without a compatible artifact manifest is insufficient for safe recovery.
#### Fast and slow phase response
- Set relative displacement `n-m=-8` and interpolation scale `s=4`.
- The mapped displacement is `-8/4=-2`.
- For a fast frequency of `pi/2` radians per coordinate, the relative angle is `-pi`.
- For a slow frequency of `pi/8`, the angle is `-pi/4`.
- Unit-axis pair contributions are respectively `-1` and `sqrt(2)/2`.
- This toy frequency choice illustrates why the same displacement affects rotary pairs differently.
#### Rotation sign and order
- Let `q=[1,0]`, `k=[0,1]`, and the relative angle be `pi/2`.
- Applying `R(pi/2)` to k yields `[-1,0]`.
- The resulting q-dot-rotated-k score is `-1`.
- Reversing the angle gives `R(-pi/2)k=[1,0]` and score `1`.
- Unlike symmetric unit-axis examples, this example exposes a sign reversal.
- Position order and the chosen rotation convention must therefore be tested together.
#### Cache payload per generated token
- Use the E4 architecture with `N=32`, `H_kv=8`, `d=128`, and `b=2`.
- One additional token adds `2*32*8*128*2=131072` bytes for one sequence.
- This is exactly `128 KiB` of ideal KV payload per retained token.
- Generating 1024 additional tokens adds `128 MiB` under the same assumptions.
- For three independent sequences, the same growth adds `384 MiB`.
- The figures omit allocation granularity, metadata and non-cache memory.
#### Distinct prompt and completion lengths
- Consider a 16384-token total capacity and a 12000-token prompt.
- The maximum completion fitting that budget is `16384-12000=4384` tokens.
- A requested 5000-token completion would exceed the cap by `616` tokens.
- Truncating the completion budget is different from removing prompt evidence.
- Which action is permissible depends on the application's explicit contract.
- Compute the budget after adding system, tool and special tokens.
#### Triangular prefill growth
- At a prompt length of 2, causal attention permits `2*3/2=3` pairs.
- At a prompt length of 6, it permits `6*7/2=21` pairs.
- Tripling this short prompt therefore multiplies its pair count by `21/3=7`.
- The asymptotic ninefold approximation ignores the finite triangular boundary.
- At much larger lengths the ratio approaches nine.
- Neither ratio alone specifies kernel latency or total inference cost.
#### Offset error in decode
- Assume a prompt contains four tokens at logical positions `0,1,2,3`.
- The next token should receive position `4`, not the local decode-array index `0`.
- With `theta=pi/2`, that incorrect reset changes the query phase by `2*pi` for this pair.
- The error is invisible in that single pair because this particular phase wraps exactly.
- A pair with `theta=pi/4` instead changes by `pi`, reversing its unit-axis contribution.
- A parity test should use the checkpoint's full frequency set rather than only one wrapped frequency.
#### Paired evaluation outcomes
- Suppose 40 paired examples have both models correct and 30 have both wrong.
- The original alone is correct on 20, while the new model alone is correct on 10.
- Original accuracy is `(40+20)/100=60%`.
- New accuracy is `(40+10)/100=50%`, a difference of `-10` percentage points.
- The 30 discordant examples contain the paired evidence about the change.
- Document grouping and sampling uncertainty still require an appropriate interval.
#### Fixed-context evidence perturbation
- Construct two otherwise identical prompts of 8000 tokens.
- Put the same question at the end and alter a supporting fact at position 1000.
- The evidence-to-question separation is approximately 7000 tokens in both prompts.
- A successful controlled response changes only the answer components entailed by the altered fact.
- A changed answer alone is not sufficient if the new response cites unrelated evidence.
- Keep distractors, mask, tokenizer and generation settings fixed for a meaningful comparison.
#### Cache sharding counterexample
- Suppose the ideal total KV payload of a session is 8 GiB across four devices.
- Perfect equal sharding with no replication would place 2 GiB of payload on each device.
- Full replication instead places 8 GiB on each device, or 32 GiB in aggregate.
- A hybrid scheme lies between these figures depending on which layers and heads are replicated.
- Weight memory and communication buffers are separate from these payload figures.
- Inspect the actual serving topology before deriving per-device headroom.
#### Position-dependent cache key
- Two sessions may share identical token IDs and model weights but have different `s` values.
- At logical position `8`, the scaled coordinate is `2` when `s=4`.
- At the same position it is `1` when `s=8`.
- For a frequency of `pi/2`, the corresponding angles are `pi` and `pi/2`.
- Rotated keys for a general nonzero vector therefore differ between the two policies.
- Reusing a rotated prefix cache across those scales is invalid without rebuilding it.
**Takeaway:** Numeric checks are strongest when paired with an explicit assumption and a counterexample.

## 12. Quick Reference Card

| Field | Answer |
|---|---|
| Topic | RoPE relative-position attention and the decision to use position interpolation for context extension. |
| Core formula | `(R(m theta)q)^T(R(n theta)k)=q^T R((n-m)theta)k`; `p(m)=m/s`, with `s=L_target/L_train`. |
| Key assumption | Matching rotation/frequency/mapping for q and k; checkpoint-compatible layout, masks, positions and cache. |
| Complexity | Dense prefill attention is Theta(B*N*H_q*T^2*d); retained KV payload is `2*B*N*H_kv*T*d*b` bytes under Section 9 assumptions. |
| Use when | A compatible checkpoint needs longer context and measured quality justifies adaptation and serving costs. |
| Avoid when | Required quality or safety fails, cache policy is inconsistent, or capacity cannot support the workload. |
| Go deeper | [RoFormer](https://arxiv.org/abs/2104.09864); [Position Interpolation](https://arxiv.org/abs/2306.15595); [collection index](INDEX.md). |
