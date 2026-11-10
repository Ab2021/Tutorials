---
collection: "module-17-vision"
file_id: "02"
topic: "CNNs, attention, and transfer learning"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Change in input resolution, deployment device, label policy, or pretrained checkpoint license."
notation_style: "plain-text"
prerequisites: ["Basic linear algebra, convolution, train/validation/test separation"]
pattern_tags: ["vision", "representation", "transfer-learning"]
source_status: "OPEN — visible source names inventoried; original bodies unavailable"
owner_role: "Vision ML reviewer"
source_index: "INDEX.md"
---

# CNNs, Attention and Transfer Learning

> **Purpose:** Select a visual representation and adaptation policy without confusing receptive field, spatial resolution, and task evidence.
> **Prerequisites:** Matrix multiplication, supervised-loss minimisation, grouped data splits; see the module [INDEX](../INDEX.md).
> **Key outcome:** Given image size, feature-map shape, label volume and deployment constraints, compute a convolutional layer's output and parameters, then defend a transfer-learning baseline.
> **This file does NOT cover:** detector matching, dense-mask losses, or generative image synthesis.
> **Level:** 1 Foundations.
> **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** DERIVED layer arithmetic; VERIFIED illustrative calculations; SOURCED architecture definitions; no deployment claim.
> **Retrieval cue:** If validation is high but a new camera fails, start at Section 8.

## 1. File Header

**Manuscript status:** incomplete. The extended drills, interview fields and inspection cards contain repeated generic prose; those blocks are not accepted as substantive diagnostic or interview depth.

The decision is not “CNN or transformer?” in isolation. It is whether the available labels, resolution, latency, and expected shift justify training features from scratch, adapting an existing encoder, or collecting evidence before changing architecture. The visible source inventory includes CNN architecture fundamentals, classification/transfer learning, and augmentation titles at [`directory.txt`](../../../../Templates/directory.txt#L549-L565); those names are scope evidence only.

**Takeaway:** An encoder choice is a conditional operational decision, not a leaderboard identity.

## 2. Concept Map / Mental Model

```text
image x [H,W,C] -> resize/normalise -> encoder f_theta -> feature map z [H',W',D]
                                      |                     |
                                      |                     +-> pooling/head -> class score
                                      +-> frozen or adapted weights
train labels -> split by scene/device/subject -> loss -> update only permitted parameters
new camera/site -> slice evaluation -> recalibrate, adapt, or abstain
```

A convolution imposes local connectivity and shared filters; depth grows its effective receptive field. Channel/spatial attention reweights existing features. Transfer learning starts from a checkpoint trained on another distribution, so it exchanges sample efficiency for a possible representation and license mismatch.

**Takeaway:** Keep preprocessing, encoder, head, and decision threshold separately versioned.

## 3. Foundational Definitions

| Term | Meaning | Units / shape | Valid boundary | Nonexample |
|---|---|---|---|---|
| kernel | learned local weight tensor | [K_h,K_w,C_in,C_out] | stride/padding declared | an input image |
| receptive field | input region that can influence one activation | pixels | depends on all prior layers | feature-map width |
| stride s | input displacement between windows | pixels | positive integer | a pooling score |
| transfer learning | initialise from a prior task then freeze/adapt | procedure | source/target shift assessed | merely using augmentation |
| attention gate | data-dependent feature weighting | [H',W',D] or [D] | gate is trained and evaluated | a guarantee of explanation |
| calibration | agreement of predicted probability and event frequency | probability | tested on target population | accuracy alone |

A random image split can put the same product, patient, camera scene, or video sequence into train and test. That evaluates memorisation of a dependency, not deployment generalisation.

**Takeaway:** State the dependency group before reporting any visual-model metric.

## 4. Core Formulas with Full Derivation

### 4.1 Base: convolution output geometry

Validity before use: rectangular input, integer stride, symmetric zero padding, no dilation. Derivation: a valid window starts at offset 0 and the last start is at most H+2P-K; counting stride-spaced starts gives the floor expression.

```text
FORMULA [CNN-01]: H_out = floor((H + 2P - K_h)/S) + 1
WHERE: H input height [pixels]; P padding [pixels/side]; K_h kernel height [pixels];
       S stride [pixels]; H_out output height [activations]. Width follows identically.
CONSTRAINT: H + 2P >= K_h; integer geometry; dilation=1.
NUMERICAL EXAMPLE: H=224, P=3, K_h=7, S=2 gives floor((224+6-7)/2)+1=112.
INTERPRETATION: a 7x7, stride-2 stem produces 112 positions, not 224.
```

Boundary/counterexample: with asymmetric padding or dilation, use the framework's exact convention; copying this expression silently is wrong.

### 4.2 General: parameter count

Validity before use: dense convolution without groups or bias sharing beyond channel sharing. Each output channel owns K_h*K_w*C_in weights and optionally one bias.

```text
FORMULA [CNN-02]: N = C_out*(K_h*K_w*C_in + 1)
WHERE: N trainable scalar parameters; C_in/C_out channel counts; +1 is one bias/output channel.
CONSTRAINT: groups=1; if bias=False, remove +1.
NUMERICAL EXAMPLE: 3x3, C_in=64, C_out=128: N=128*(9*64+1)=73,856 parameters.
INTERPRETATION: doubling output channels doubles this layer's weights and output activation payload.
```

For grouped convolution with g groups, replace C_in by C_in/g when divisible. Depthwise convolution is g=C_in, so it saves parameters but needs pointwise mixing to exchange channels.

### 4.3 Variant: transfer objective

For target examples (x_i,y_i), an adapted encoder minimises the empirical task loss plus an optional penalty that limits drift from initial weights theta_0.

```text
FORMULA [CNN-03]: L(theta,w)= (1/n) sum_i ell(h_w(f_theta(x_i)),y_i) + lambda||theta-theta_0||_2^2
WHERE: theta encoder weights; w head weights; n labelled target examples; lambda dimensionless penalty.
CONSTRAINT: labels and loss match the task; penalty is a design choice, not evidence of preservation.
NUMERICAL EXAMPLE: illustrative mean loss=0.42 and penalty=0.03 with lambda=0.1 gives L=0.423.
INTERPRETATION: the penalty contribution is 0.003, so data loss still dominates this stated objective.
```

### 4.4 Limit / counterexample

Freezing all layers assumes source features remain useful. If target images are thermal while pretraining was natural RGB, the first-layer channel statistics can invalidate that assumption; unfreezing only the head may not repair it.

**Takeaway:** Compute geometry and parameter cost before interpreting “small” or “efficient.”

## 5. Worked Numerical Example

**ILLUSTRATIVE, VERIFIED arithmetic; not a benchmark.** A team has 1,200 labelled 224x224 RGB images from 60 machines and must classify a defect. It proposes a checkpoint encoder plus a linear head. Split by machine: 42/9/9 machines, not random images. A 7x7 stride-2 padded stem yields 112x112 positions by CNN-01. Its first layer has 9,536 parameters: 64*(7*7*3+1). If the final feature is 512-dimensional and there are four classes, a biased linear head has 4*(512+1)=2,052 parameters.

The head is small, but the decision risk lies in 60 correlated machines. If one machine contributes 200 images, image-level accuracy can be dominated by its appearance. Report macro recall by machine and inspect confidence calibration on held-out machines. A 0.90 confidence threshold should be chosen only after counting review capacity and false-negative cost.

Sensitivity: changing stride from 2 to 1 makes the stem 224x224, four times as many spatial positions (50,176 versus 12,544). It can improve small-defect detail but raises downstream activation memory and latency; it is not automatically a quality improvement.

**Takeaway:** The arithmetic identifies spatial and ownership constraints before a training run exists.

## 6. Visual / Geometric Intuition

```text
input pixels:     [ local 7x7 window ] -> one channel response
shift filter:             [ local 7x7 window ] -> next response
stack layers: local edges -> textures -> parts -> task-specific evidence
attention: feature positions/channels receive different learned weights, but retain no causal guarantee
```

Convolutional sharing says a useful local pattern may appear in multiple positions. It does not say location never matters: cropping, positional features, or task framing can be necessary for documents and medical images. A global pooling head sacrifices location; that is desirable for image-level classification and harmful when the output must identify where a defect is.

**Takeaway:** Match retained spatial detail to the required output grain.

## 7. Comparison / Selection Table

| Option | Use when | Strength | Avoid when | Disqualifier | Fallback |
|---|---|---|---|---|---|
| frozen encoder + head | modest target labels; similar acquisition | low update cost | source/target mismatch likely | target probe fails by required slice | unfreeze late blocks |
| partial fine-tuning | some labels and controlled validation | adapts semantics | labels are noisy or split leaks | general-domain regression unacceptable | regularise/replay and gate |
| train from scratch | large, licensed target corpus | no inherited mismatch | label count is small | no credible held-out protocol | transfer baseline |
| CNN with local stages | latency and locality matter | predictable dense geometry | global relation is core | receptive field cannot cover needed context | add attention/global stage |
| attention-heavy encoder | long-range layout matters | global interaction | resolution makes cost infeasible | measured memory/latency exceeds budget | hierarchical/windowed design |

**Takeaway:** Choose the least complex option that passes target-slice and operational gates.

## 8. Failure Modes & Edge Cases

| Failure | Competing causes | Discriminating check | Bounded fix | Verification |
|---|---|---|---|---|
| held-out score collapse | site shift or leakage removal | compare image vs site split; inspect metadata | collect sites; adapt only after split freeze | site-level confidence interval |
| small defects missed | downsampling or label ambiguity | recall by object size and adjudicated sample | retain resolution or change task to detection | size-slice recall gate |
| high confidence wrong | miscalibration or shortcut | reliability plot and counterfactual background test | calibrate on target validation; diversify data | expected calibration error with support |
| adaptation regresses source task | catastrophic forgetting or metric mismatch | evaluate fixed source suite and target suite | freeze more layers or use replay | both promotion gates |
| training instability | learning rate or bad normalisation | loss/gradient trace and input statistics | reduce LR; audit preprocessing | deterministic short rerun |

Do not diagnose “overfitting” from a single accuracy gap: label noise, leakage, a changed preprocessing path, or prevalence shift can give similar symptoms.

**Takeaway:** A failure label is useful only when it identifies a discriminating next check.

## 9. Complexity & Cost Analysis

A dense convolution produces approximately H_out*W_out*C_out*K_h*K_w*C_in multiply-accumulate operations per image; this is an operation count, not observed latency. For the illustrative 112x112 output, 7x7x3 to 64 channels costs 112*112*64*7*7*3 = 118,013,952 MACs: 12,544*64=802,816; 802,816*147=118,013,952 MACs. Activations alone contain 802,816 values. At FP16 that payload is 1,605,632 bytes, about 1.53 MiB, excluding gradients, workspace, and later layers.

Batch size, memory layout, device kernels, I/O, augmentation, and mixed precision determine actual throughput. Measure them on the target environment; no hardware rate is asserted here.

**Takeaway:** Parameter count is not activation memory or end-to-end latency.

## 10. Key Theorems / Results Reference

- Translation equivariance of an ideal convolution follows from shared local filtering; padding, sampling, pooling, and learned downstream heads can break exact equivariance.
- Universal approximation results state existence under conditions, not sample efficiency, optimisation success, calibration, or robustness.
- Transfer learning is an empirical strategy, not a theorem that source features transfer.
- [Krizhevsky, Sutskever, Hinton (2012)](https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) supports the historical CNN architecture example; it does not validate this sheet's scenario.
- [He et al. (2016)](https://arxiv.org/abs/1512.03385) defines residual learning; source review date 2026-11-09.

**Takeaway:** Cite a result for the narrow claim it actually establishes.

## 11. Interview / Exam Questions


### 11.2 Worked diagnostic traces

**Trace A — geometry mismatch.** The training preprocessing produces 224 by 224 images with a 7-by-7, stride-2, padding-3 stem. A serving change preserves 256 by 256 images but leaves the fixed-size downstream positional/feature assumption unchanged. CNN-01 yields 128 rather than 112 stem positions. The first discriminating check is to log the input and every feature-map shape for one immutable fixture in both environments. If the model permits variable shapes, the remaining head and pooling contract still needs inspection. The bounded correction is to pin the preprocessing manifest and reject incompatible inputs, or deliberately qualify a new variable-size path with a frozen evaluation. The verification is an end-to-end fixture trace plus a held-out target evaluation, not merely a successful forward pass.

**Trace B — threshold versus representation.** On a held-out machine, ranking remains reasonable but confidence scores are systematically higher than observed event frequency. Lowering the decision threshold may raise recall and overload reviewers; changing the encoder may be unnecessary. Compare reliability curves and precision-recall curves for the old and new machine, with support counts. If ordering is stable but calibration shifts, fit and validate a calibration method on a representative validation partition that is separate from the final test. If ordering fails too, collect target labels and investigate input shift. Verification requires the same queue-capacity rule used for promotion.

**Trace C — false transfer improvement.** A head-only model reports a higher random-image validation score than a scratch baseline. The source checkpoint was trained on images collected from the same camera vendor. This can be legitimate reuse or hidden overlap. Construct an entity/site provenance table, hash exact duplicates where permitted, and repeat evaluation by machine. If the advantage vanishes only after grouping, report the initial estimate as leakage-sensitive rather than as transfer benefit. Do not solve this by selecting the random split because it is larger.

**Trace D — augmentation damage.** Brightness jitter improves a general class score but reduces recall for an underexposed defect whose darkness is the label signal. Compare the same labelled examples under the proposed transform, record whether expert labels should remain invariant, and inspect the failure slice. The correction is not “turn off augmentation” universally: restrict jitter range, condition it on acquisition state, or split the task. Verify with the affected slice and a human review of transform validity.

### 11.3 Transfer-learning decision ledger

| Decision point | Evidence required before decision | Decision if evidence passes | Decision if evidence fails | Owner |
|---|---|---|---|---|
| checkpoint eligibility | license, input channels, known training scope, artifact hash | permit offline baseline | reject checkpoint | model governance |
| input compatibility | preprocessing fixture and channel/range test | reuse declared stem | add validated adapter or reject | ML engineer |
| frozen probe | grouped validation with support | retain frozen encoder candidate | test limited unfreezing | ML scientist |
| adaptation | target and retained-domain gates | promote candidate to shadow test | rollback to frozen baseline | model owner |
| threshold | cost/review-capacity table and calibration | define operating point | abstain/human-review route | operations owner |
| deployment | target device latency/memory measurement | controlled rollout | optimize or change architecture | platform owner |

The ledger deliberately separates a method choice from a deployment decision. An offline metric cannot prove capacity, privacy, or rollback readiness.

### 11.4 Additional numerical checks

**Grouped-convolution check.** A 3-by-3 convolution from 64 to 128 channels with 8 groups has 128*(3*3*(64/8)+1)=9,344 parameters. The dense result was 73,856, so grouping reduces this layer’s weights by about 87.35%. It also changes cross-channel mixing; it is not a free replacement. The calculation assumes 64 is divisible by 8 and includes one bias per output channel. A pointwise layer may restore mixing and alter the total cost.

**Activation check.** The illustrative stem output has 112*112*64=802,816 values. At FP32, values alone need 3,211,264 bytes, or about 3.06 MiB using 1 MiB=1,048,576 bytes. At FP16 it is half that payload. Training requires saved activations, gradients, optimizer state, workspace and allocator overhead; therefore neither number is a peak-memory estimate.

**Receptive-field recurrence.** For a stack indexed by l with prior jump j_(l-1), kernel k_l and stride s_l, the jump is j_l=j_(l-1)*s_l and receptive field is r_l=r_(l-1)+(k_l-1)*j_(l-1), starting r_0=j_0=1. This follows because adjacent positions at layer l are separated by s_l prior positions, and a k_l-wide window adds k_l-1 such jumps. Two 3-by-3 stride-1 layers produce r_1=3 and r_2=5. A stride-2 3-by-3 next layer gives r_3=7 and j_3=2. Padding affects which input locations are real, so receptive field size does not guarantee useful boundary evidence.

**Sensitivity consequence.** Replacing the stride-2 stem with stride-1 changes output positions from 12,544 to 50,176. If every later layer maintained comparable spatial scale, activation traffic can grow materially; however exact growth must be measured from the complete graph. A reviewer should request the model graph, batch size, dtype, device, and peak-allocation trace before approving a capacity conclusion.

### 11.5 Source and evidence register

| Source ID | Primary reference | Exact supported scope | Publication date | Reviewed | Limitation / recheck trigger |
|---|---|---|---:|---:|---|
| SRC-CNN-01 | [Krizhevsky et al.](https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | historical CNN architecture context | 2012 | 2026-11-09 | does not support target deployment results |
| SRC-CNN-02 | [He et al., ResNet](https://arxiv.org/abs/1512.03385) | residual-learning architecture | 2015 | 2026-11-09 | source setup differs from illustrative workflow |
| SRC-CNN-03 | [scikit-learn pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | training-only preprocessing/leakage teaching | current docs | 2026-11-09 | not vision-specific performance evidence |
| INV-CNN-01 | [`directory.txt` 549–565](../../../../Templates/directory.txt#L549-L565) | visible inventory titles | inventory | 2026-11-09 | original lesson bodies unavailable |

All numeric workload and cost values in this file are ILLUSTRATIVE and hand-checked. No training run, hardware benchmark, clinical outcome, model license decision, or deployment outcome was executed or observed.


### 11.6 Release contracts: evidence required before a visual-model decision

The following contracts make the earlier transfer ledger executable. They are checklists for an illustrative teaching workflow, not evidence that any system passed them.

#### Contract 01 — Input manifest
- **Required declaration:** Record sensor, optics, illumination, resize kernel, crop rule, colour order, range, dtype, orientation, and timestamp.
- **Why it matters:** A tensor may have the expected shape while its channel order or intensity scale is wrong.
- **Discriminating check:** Hash the preprocessing configuration and run one immutable fixture through training and serving.
- **Bounded decision:** Reject a release when the fixture differs at an asserted intermediate tensor.
- **Boundary:** This proves path equivalence for the fixture, not performance.
- **Record:** `CNN-CONTRACT-01` with owner, date, input artifact, result, and unresolved exception.

#### Contract 02 — Label contract
- **Required declaration:** Specify unit, labeler instructions, adjudication, abstentions, ambiguous class policy, and label timestamp.
- **Why it matters:** A class name does not state whether borderline defects count as positive.
- **Discriminating check:** Double-label a sampled disagreement slice and retain the adjudication reason.
- **Bounded decision:** Hold the task definition fixed while comparing models.
- **Boundary:** Agreement does not prove labels match the downstream action.
- **Record:** `CNN-CONTRACT-02` with owner, date, input artifact, result, and unresolved exception.

#### Contract 03 — Split contract
- **Required declaration:** Name the dependency group and assign every group to exactly one partition.
- **Why it matters:** Random images estimate repeated-scene performance when scenes recur.
- **Discriminating check:** Publish counts by group, class, acquisition period, and partition.
- **Bounded decision:** Fail the protocol when a group crosses a partition.
- **Boundary:** Grouping may increase uncertainty because fewer independent groups remain.
- **Record:** `CNN-CONTRACT-03` with owner, date, input artifact, result, and unresolved exception.

#### Contract 04 — Threshold contract
- **Required declaration:** Map scores to action, reviewer capacity, false-negative harm, and abstention policy.
- **Why it matters:** A classifier score is not an instruction to automate.
- **Discriminating check:** Compute the confusion matrix at each candidate threshold on a locked validation set.
- **Bounded decision:** Choose only a threshold that meets the stated guardrail and capacity.
- **Boundary:** A threshold can drift when prevalence or calibration changes.
- **Record:** `CNN-CONTRACT-04` with owner, date, input artifact, result, and unresolved exception.

#### Contract 05 — Augmentation contract
- **Required declaration:** State the hypothesised nuisance, transform distribution, and label-preservation argument.
- **Why it matters:** Visual plausibility is not semantic validity.
- **Discriminating check:** Have a domain reviewer inspect paired original and transformed samples from every class.
- **Bounded decision:** Remove or bound transforms that change a label-defining cue.
- **Boundary:** A valid training transform may still be absent in deployment.
- **Record:** `CNN-CONTRACT-05` with owner, date, input artifact, result, and unresolved exception.

#### Contract 06 — Checkpoint contract
- **Required declaration:** Record checkpoint identifier, license, intended modality, normalization, known source task, and permitted use.
- **Why it matters:** Similar architecture names do not establish equivalent weights or rights.
- **Discriminating check:** Run input-range and feature-shape fixtures before measuring task metrics.
- **Bounded decision:** Disqualify an artifact with incompatible governance or interface terms.
- **Boundary:** Passing compatibility does not establish target transfer.
- **Record:** `CNN-CONTRACT-06` with owner, date, input artifact, result, and unresolved exception.

#### Contract 07 — Head contract
- **Required declaration:** Declare whether output is image class, multi-label vector, box, mask, keypoint, or embedding.
- **Why it matters:** A pooled class head cannot recover a spatial coordinate that was discarded.
- **Discriminating check:** Trace target shape and loss target shape for a batch containing all edge labels.
- **Bounded decision:** Change the output formulation before tuning the encoder when output grain is wrong.
- **Boundary:** Correct shapes do not establish an adequate label ontology.
- **Record:** `CNN-CONTRACT-07` with owner, date, input artifact, result, and unresolved exception.

#### Contract 08 — Calibration contract
- **Required declaration:** Define probability population, binning rule, sample support, and recalibration boundary.
- **Why it matters:** Accuracy can improve while probability estimates become less useful.
- **Discriminating check:** Compare reliability and precision-recall on the same held-out groups.
- **Bounded decision:** Use a separately held-out calibration partition; leave final test untouched.
- **Boundary:** Sparse bins make apparent reliability unstable.
- **Record:** `CNN-CONTRACT-08` with owner, date, input artifact, result, and unresolved exception.

#### Contract 09 — Slice contract
- **Required declaration:** Pre-register acquisition, geography, device, class, size, illumination, and time slices that alter risk.
- **Why it matters:** A high overall score can hide a deployment-critical group.
- **Discriminating check:** Report numerator, denominator, interval method, and missing-slice policy.
- **Bounded decision:** Block promotion on a critical slice that violates its gate.
- **Boundary:** Unobserved slices remain unknown rather than passing by omission.
- **Record:** `CNN-CONTRACT-09` with owner, date, input artifact, result, and unresolved exception.

#### Contract 10 — Rollback contract
- **Required declaration:** Specify current model, artifact location, compatible preprocessing, trigger owner, and safe fallback.
- **Why it matters:** A model package alone cannot restore an end-to-end decision path.
- **Discriminating check:** Exercise a controlled rollback fixture before exposure.
- **Bounded decision:** Route to manual review or the prior approved path when the trigger fires.
- **Boundary:** A successful dry run does not prove incident response timing.
- **Record:** `CNN-CONTRACT-10` with owner, date, input artifact, result, and unresolved exception.

#### Contract 11 — Latency contract
- **Required declaration:** Measure arrival pattern, percentile objective, batch rule, queue limit, hardware, and preprocessing time.
- **Why it matters:** Parameter count is neither p95 latency nor capacity.
- **Discriminating check:** Trace decode, transforms, host-device transfer, inference, postprocess, and response separately.
- **Bounded decision:** Use backpressure or degrade only under a declared user-impact policy.
- **Boundary:** A synthetic batch may conceal queueing and I/O cost.
- **Record:** `CNN-CONTRACT-11` with owner, date, input artifact, result, and unresolved exception.

#### Contract 12 — Memory contract
- **Required declaration:** Specify batch, resolution, dtype, graph, workspace, allocator, and concurrent requests.
- **Why it matters:** Activation memory often dominates weights during training.
- **Discriminating check:** Capture peak allocated and reserved memory on the target graph.
- **Bounded decision:** Reduce resolution, batch, precision, or graph footprint with a new quality gate.
- **Boundary:** One device trace is not a fleet capacity guarantee.
- **Record:** `CNN-CONTRACT-12` with owner, date, input artifact, result, and unresolved exception.

#### Contract 13 — Drift contract
- **Required declaration:** Define reference window, monitored features, score distribution, output action, and label-delay plan.
- **Why it matters:** Feature shift alone does not quantify decision harm.
- **Discriminating check:** Compare current input and score slices to the approved reference and review labeled samples when available.
- **Bounded decision:** Escalate to shadow evaluation or manual review instead of automatic retraining.
- **Boundary:** No-label monitoring cannot establish accuracy decline.
- **Record:** `CNN-CONTRACT-13` with owner, date, input artifact, result, and unresolved exception.

#### Contract 14 — Explanation contract
- **Required declaration:** State the question an explanation must answer and the intervention it supports.
- **Why it matters:** A saliency map is not a causal proof or a clinical rationale.
- **Discriminating check:** Test stability under benign perturbations and inspect with qualified reviewers.
- **Bounded decision:** Use explanation only as a diagnostic input, not as a sole release criterion.
- **Boundary:** Stable maps can still highlight a non-causal correlate.
- **Record:** `CNN-CONTRACT-14` with owner, date, input artifact, result, and unresolved exception.

#### Contract 15 — Privacy contract
- **Required declaration:** Minimize identifiers, define retention, access, redaction, and audit owner before collecting images.
- **Why it matters:** Removing a filename may not remove identifying pixels or metadata.
- **Discriminating check:** Inspect EXIF, overlays, storage access logs, and export paths.
- **Bounded decision:** Block export or route to approved storage when controls are absent.
- **Boundary:** Technical controls do not decide legal applicability.
- **Record:** `CNN-CONTRACT-15` with owner, date, input artifact, result, and unresolved exception.

#### Contract 16 — Release contract
- **Required declaration:** Bind model, preprocessing, threshold, label policy, evaluation report, and rollback plan into one signed release record.
- **Why it matters:** A good notebook result is not a deployable decision contract.
- **Discriminating check:** Resolve every artifact digest from the release record in a staging replay.
- **Bounded decision:** Do not release with an unresolved artifact or critical-slice exception.
- **Boundary:** A complete record does not convert illustrative evidence into measured outcomes.
- **Record:** `CNN-CONTRACT-16` with owner, date, input artifact, result, and unresolved exception.

**Takeaway:** A release is defensible only when model behavior, decision policy, and operational fallback are jointly specified.

### 11.7 Diagnostic drills with expected evidence

Each drill is a constructed scenario. The stated check is an oracle input for review; no run or production incident is asserted.

#### Drill 01 — RGB/BGR reversal
- **Observed symptom (illustrative):** A camera SDK emits BGR but the checkpoint expects RGB.
- **Competing explanations:** channel-order mismatch, checkpoint normalization mismatch, or a codec color-matrix conversion defect.
- **Diagnostic:** Compare per-channel means and a known red/green/blue color-bar fixture before normalization; inspect raw decode separately.
- **Bounded correction:** Swap explicitly at ingress and version the conversion; do not retrain to absorb an accidental channel order.
- **Verification:** Recheck decoded channel values, normalized tensors, logits, and features on the immutable fixture.
- **Escalation:** Preserve the original compressed image and decoded tensor if the color bar cannot localize the stage.
- **Mitigation risk:** applying another swap to an already corrected path reverses valid inputs; bind conversion to source schema.

#### Drill 02 — EXIF orientation loss
- **Observed symptom (illustrative):** A mobile image rotates at display time but tensors ignore orientation metadata.
- **Competing explanations:** EXIF ignored, pixels already transposed with metadata retained, or crop coordinates interpreted before rotation.
- **Diagnostic:** Render source and tensor-side thumbnails with orientation tags and an asymmetric corner marker visible.
- **Bounded correction:** Normalize orientation once before crop and record the source tag and resulting coordinate convention.
- **Verification:** Test each relevant EXIF orientation plus missing metadata; review rotated and unrotated slices separately.
- **Escalation:** Preserve metadata and pixel bytes if display and tensor orientation disagree after normalization.
- **Mitigation risk:** unconditional rotation corrupts correctly oriented inputs and invalidates location labels; condition on verified metadata.

#### Drill 03 — Resizer aliasing
- **Observed symptom (illustrative):** A small defect disappears after downsampling.
- **Competing explanations:** insufficient output resolution, an unsuitable interpolation kernel, an over-tight crop, or incorrect annotation coordinates.
- **Diagnostic:** Overlay source and resized pixels around adjudicated defects and measure their source/output size in pixels.
- **Bounded correction:** Use a task-qualified resolution/interpolation policy or change the output to a feasible localization task.
- **Verification:** Gate recall by source object-size band with support, not just aggregate image accuracy.
- **Escalation:** Preserve source-resolution labels if the minimum action-critical object is not represented after resize.
- **Mitigation risk:** higher resolution raises activation memory/latency and can expose annotation noise; requalify capacity and labels.

#### Drill 04 — Padding shortcut
- **Observed symptom (illustrative):** Black letterbox bars correlate with one class.
- **Competing explanations:** aspect ratio differs by class, acquisition source creates both padding and label, or normalization makes pad color distinctive.
- **Diagnostic:** Mask bars, vary padding color, and change aspect-ratio handling while keeping object content fixed in a review fixture.
- **Bounded correction:** Crop to valid content when label-preserving and diversify acquisition across aspect ratios and pad policies.
- **Verification:** Confirm class performance survives padding counterfactuals on held-out acquisition groups.
- **Escalation:** Reject the shortcut-dependent evaluation if class identity cannot be assessed without acquisition markers.
- **Mitigation risk:** aggressive cropping can remove border defects; check crop content coverage and transform box/mask labels accordingly.

#### Drill 05 — Duplicate near-leakage
- **Observed symptom (illustrative):** Burst captures of one item occupy train and test.
- **Competing explanations:** exact duplicate export, near-duplicate burst, same physical item across sessions, or repeated public web image.
- **Diagnostic:** Cluster content hashes/perceptual similarity with item ID and camera timestamps; inspect cluster membership across partitions.
- **Bounded correction:** Assign all related images to one deployment-meaningful group before splitting and retain an exclusion audit.
- **Verification:** Assert disjoint group/raw-lineage IDs and recompute metrics with support on the cleaned split.
- **Escalation:** Treat unresolved identity matches as a protocol failure rather than assuming independent examples.
- **Mitigation risk:** over-aggressive perceptual clustering can merge distinct items and shrink the evaluation population; audit false merges.

#### Drill 06 — Timestamp leakage
- **Observed symptom (illustrative):** A post-inspection annotation is embedded in the image name.
- **Competing explanations:** metadata explicitly contains outcome, annotation overlay is burned into pixels, or post-decision crop selects a diagnostic region.
- **Diagnostic:** Audit filenames, EXIF, overlays, crops, and derived features against decision and materialization timestamps.
- **Bounded correction:** Remove every post-decision field/overlay from training and serving paths; do not mask only the visible filename.
- **Verification:** Run a provenance fixture containing an obvious future marker and assert rejection in every exported branch.
- **Escalation:** Invalidate evaluations that consumed the contaminated feature set and rebuild immutable as-of inputs.
- **Mitigation risk:** broad metadata removal can discard legitimate acquisition context; classify each field by semantics and availability.

#### Drill 07 — Class prevalence shift
- **Observed symptom (illustrative):** A test set has a much larger positive rate than deployment.
- **Competing explanations:** enriched evaluation sampling, changed operating population, threshold drift, or inconsistent eligibility.
- **Diagnostic:** Compare prevalence, score-conditioned outcome rates, predicted-positive volume, and precision-recall under a justified deployment mix.
- **Bounded correction:** Report prevalence-sensitive metrics and reviewer load; do not transplant precision from the enriched set.
- **Verification:** Recompute confusion counts and actions per 1,000 eligible cases with explicit prevalence and support.
- **Escalation:** Keep threshold/value claims open when the deployment prevalence is unknown or unstable.
- **Mitigation risk:** reweighting assumes stable class-conditional behavior and reliable prevalence; both require sensitivity checks.

#### Drill 08 — Annotation style shift
- **Observed symptom (illustrative):** One labeling vendor outlines defects more tightly.
- **Competing explanations:** different annotation instructions, genuinely different defect mix, coordinate convention mismatch, or unequal review expertise.
- **Diagnostic:** Measure box/mask agreement on a shared double-labeled bridge sample, stratified by object size and defect type.
- **Bounded correction:** Adjudicate disagreements and document canonical extent, truncation, ambiguity, and coordinate policies before merging data.
- **Verification:** Evaluate by annotation source and canonical-policy version on shared groups.
- **Escalation:** Do not interpret a metric difference across vendors as model shift until the bridge sample separates label definition from image mix.
- **Mitigation risk:** harmonizing to one vendor can erase valid alternative interpretations; retain disagreement where the target is ambiguous.

#### Drill 09 — Focal-loss misuse
- **Observed symptom (illustrative):** Rare-class weighting changes gradient emphasis but labels are ambiguous.
- **Competing explanations:** focal loss emphasizes mislabeled hard examples, oversampling changes calibration, or class definitions overlap.
- **Diagnostic:** Inspect confusion, duplicate/disagreement rates, and per-example loss by annotator before changing gamma or alpha.
- **Bounded correction:** Fix target semantics and sampling first; tune focal parameters only under a fixed grouped evaluation protocol.
- **Verification:** Verify rare-class precision, recall, calibration, and support rather than reporting training loss reduction.
- **Escalation:** Withhold the loss-change conclusion while ambiguous labels dominate high-loss examples.
- **Mitigation risk:** emphasizing hard examples can amplify label noise and degrade probability calibration; retain a non-focal baseline.

#### Drill 10 — Frozen batch statistics
- **Observed symptom (illustrative):** A transferred network uses stale normalization statistics.
- **Competing explanations:** training/evaluation mode mismatch, source-domain running statistics, small target batches, or unintended state updates during evaluation.
- **Diagnostic:** Compare layer input/output moments and running-stat buffers in train/eval modes on identical target fixtures.
- **Bounded correction:** Freeze or update normalization statistics deliberately using training-only target data and a declared batch regime.
- **Verification:** Repeat target and retained-domain gates; assert no evaluation batch mutates model state.
- **Escalation:** Stop comparison if evaluation-mode state differs across repeated calls with the same input.
- **Mitigation risk:** target-stat updates can erase useful source behavior and leak across evaluation; preserve a frozen-stat baseline.

#### Drill 11 — Partial unfreezing instability
- **Observed symptom (illustrative):** A late block is unfrozen with a head learning rate.
- **Competing explanations:** excessive pretrained-layer step size, normalization-mode change, random-head gradient shock, or frozen/unfrozen parameter list mismatch.
- **Diagnostic:** Inspect trainable-parameter IDs, gradient/update-to-weight norms, normalization mode, and loss after each optimizer step.
- **Bounded correction:** Warm up the head, use discriminative rates, or re-freeze the unstable block while holding splits/configurations fixed.
- **Verification:** Require a finite stable short run and target/retained-domain improvement over the frozen baseline.
- **Escalation:** Reject runs where the optimizer updates layers declared frozen or where nonfinite gradients appear.
- **Mitigation risk:** lower learning rates can conceal instability while slowing learning; compare actual updates and final utility, not just loss smoothness.

#### Drill 12 — Mixed precision overflow
- **Observed symptom (illustrative):** A loss becomes non-finite only at production resolution.
- **Competing explanations:** activation overflow at larger shapes, underflowed gradients, invalid pixels/normalization, or reduction accumulation in low precision.
- **Diagnostic:** Log finite checks, extrema, loss-scale events, and the first nonfinite tensor/operator for the failing batch and shape.
- **Bounded correction:** Adjust only the implicated precision/reduction/loss-scaling policy after locating the first invalid operation.
- **Verification:** Replay the failing batch and shape deterministically; compare with a higher-precision reference and boundary shapes.
- **Escalation:** Preserve the input and gradient/activation summary if the first nonfinite operator cannot be identified.
- **Mitigation risk:** loss scaling does not fix activation overflow or NaN inputs, and clipping can hide numerical defects; verify actual tensors.

#### Drill 13 — Attention map overclaim
- **Observed symptom (illustrative):** A heatmap is presented as evidence of the object causing a decision.
- **Competing explanations:** attention weights correlate with features, saliency implementation differs, or background shortcut drives both map and output.
- **Diagnostic:** Compare map method/version and occlude highlighted and unhighlighted regions under a declared label-preserving perturbation.
- **Bounded correction:** Use maps as investigation cues, not causal or domain explanations; obtain task-specific evidence for any explanatory claim.
- **Verification:** Measure output sensitivity under controlled perturbations and have a qualified reviewer assess the intended interpretation.
- **Escalation:** Withdraw the causal wording if highlighted-region occlusion does not support the claimed mechanism.
- **Mitigation risk:** occlusion creates out-of-distribution artifacts and can change labels; include control masks and alternative perturbations.

#### Drill 14 — Global pooling mismatch
- **Observed symptom (illustrative):** The requested output is defect location but training labels are image-level.
- **Competing explanations:** output grain is wrong, spatial information was pooled away, or weak localization supervision is unvalidated.
- **Diagnostic:** Compare required action coordinates with target/head shapes and inspect where global aggregation discards position.
- **Bounded correction:** Collect spatial labels and use an appropriate head, or change the workflow to an image-level review action.
- **Verification:** Verify localization error and coverage at the intended coordinate tolerance on independent groups.
- **Escalation:** Block location claims when only image-level labels support the model output.
- **Mitigation risk:** retaining spatial maps increases computation and annotation burden; image-level presence does not prove location accuracy.

#### Drill 15 — Threshold copied across sites
- **Observed symptom (illustrative):** One threshold overloads reviewers at a higher-prevalence site.
- **Competing explanations:** prevalence/volume shift, score calibration shift, different image pipeline, or inadequate reviewer capacity.
- **Diagnostic:** Estimate eligible volume, predicted-positive count, score-conditioned outcomes, and completed review capacity by site.
- **Bounded correction:** Use a governed capacity-aware threshold or abstain; do not simply raise the threshold to hide excess demand.
- **Verification:** Monitor selected counts, deadline completion, false-negative exposure, and calibration by site after rollout.
- **Escalation:** Retain the prior route when a site has no feasible operating point under its review and guardrail constraints.
- **Mitigation risk:** site-specific thresholds can produce unequal service and overfit small samples; require support and policy review.

#### Drill 16 — Calibration-set reuse
- **Observed symptom (illustrative):** The validation set is used repeatedly for tuning and calibration.
- **Competing explanations:** adaptive overfitting, group overlap between partitions, or final-test labels already influenced choices.
- **Diagnostic:** Trace every architecture, checkpoint, threshold, and calibration decision that consumed each partition's labels.
- **Bounded correction:** Reserve independent groups for calibration and a final untouched test; restart the protocol if the test was consumed.
- **Verification:** Publish partition genealogy, disjointness assertions, decision history, and one final evaluation timestamp.
- **Escalation:** Do not call repeatedly inspected data a final unbiased test, even if no explicit training gradient used it.
- **Mitigation risk:** extra partitions reduce training/evaluation support; report uncertainty rather than splitting into statistically useless fragments.

#### Drill 17 — Test-time crop mismatch
- **Observed symptom (illustrative):** Training uses random crops while serving center-crops a document.
- **Competing explanations:** deterministic serving crop excludes border content, training crop distribution lacks full images, or crop coordinates use different orientation/resize order.
- **Diagnostic:** Visualize crop boxes and transformed target coordinates for matched fixtures, including border and corner content.
- **Bounded correction:** Align serving to a qualified deterministic policy or evaluate the new path; do not apply random augmentation at inference.
- **Verification:** Check full-image and crop recall by content location, orientation, and aspect ratio.
- **Escalation:** Block the crop path when action-critical content is outside the retained region.
- **Mitigation risk:** using full-frame serving changes object scale and may increase memory; validate both semantic coverage and geometry.

#### Drill 18 — JPEG quality shift
- **Observed symptom (illustrative):** Recall falls only on low-quality uploads; the same scene succeeds in the archival image.
- **Competing causes:** DCT blocking may erase small edges; a mobile-app resize may be changing geometry; or low-quality uploads may come from a different site with a different class mix.
- **Discriminating diagnostic:** Re-encode a fixed, labeled fixture set at quality 95, 75, and 40 while holding pixels’ dimensions and site constant; separately replay the app resize. A monotone quality-only loss implicates compression rather than sampling.
- **Correction and risk:** Train with the measured codec/quality range or reject unsupported quality. Synthetic JPEG augmentation can teach artifacts not used in production and harm pristine images.
- **Verification:** Keep an independently captured low-quality slice; require its recall and false-positive rate to meet the declared action gate.

#### Drill 19 — Camera firmware shift
- **Observed symptom (illustrative):** Scores change after a firmware update although the product and model digest are unchanged.
- **Competing causes:** Exposure/white-balance behavior changed; lens focus changed; or timestamp/device metadata were joined to the wrong image cohort.
- **Discriminating diagnostic:** Photograph a stable color/geometry target before and after update, compare raw histogram, saturation fraction, sharpness, and EXIF firmware fields, then score paired frames.
- **Correction and risk:** Pin approved firmware or collect labeled post-update data and qualify it as a new input regime. Pinning may defer a security or device-maintenance update; escalation belongs to device operations.
- **Verification:** Make firmware version a mandatory monitored slice and replay the target on each approved version.

#### Drill 20 — Small-object metric masking
- **Observed symptom (illustrative):** Mean overlap looks acceptable while action-critical small defects are missed.
- **Competing causes:** Resolution/stride is insufficient; annotations omit tiny objects; or the matching IoU threshold is inappropriate for annotation uncertainty.
- **Discriminating diagnostic:** Report object recall and AP by area band with counts; inspect false negatives against image resolution and have a second annotator review a sample.
- **Correction and risk:** Increase input resolution, use an appropriate feature pyramid, or route the band to review. Higher resolution raises latency and may reduce batch capacity.
- **Verification:** Release only if the smallest action-critical band, not the aggregate, meets its predeclared gate.

#### Drill 21 — Background confounding
- **Observed symptom (illustrative):** A class score follows a workbench, tray, or hospital marker more than the object.
- **Competing causes:** A real contextual cue may be valid; a visible overlay may leak the label; or object crop quality may differ by site.
- **Discriminating diagnostic:** Score controlled object-preserving background swaps, overlay masks, and tight crops. A score change after background replacement with the object fixed is shortcut evidence, not proof that all context is invalid.
- **Correction and risk:** Collect counterfactual contexts, remove label-bearing overlays, or constrain the crop. Aggressive masking can remove clinically or operationally relevant context.
- **Verification:** Evaluate paired counterfactuals and a held-out site; retain per-site error tables.

#### Drill 22 — Channel-count adapter
- **Observed symptom (illustrative):** A single-channel sensor is copied three times into an RGB-pretrained model without an input contract.
- **Competing causes:** Channel semantics differ; intensity scaling differs; or the adapter order/transposition is wrong.
- **Discriminating diagnostic:** Run fixed black, mid-gray, and white fixtures through the serving transform and compare tensor shape, per-channel mean/std, and first-layer activations to the declared adapter calculation.
- **Correction and risk:** Train a one-channel stem or explicitly initialize a three-channel adapter and validate it. Reusing RGB normalization after replication can create an artificial color prior.
- **Verification:** Test illumination and sensor cohorts with the exact exported preprocessing digest.

#### Drill 23 — Multi-label collapse
- **Observed symptom (illustrative):** Co-occurring findings disappear because targets are encoded as one softmax class.
- **Competing causes:** Labels may be mutually exclusive by definition; annotation policy may suppress secondary findings; or the evaluation decoder may be wrong.
- **Discriminating diagnostic:** Inspect the label schema and co-occurrence matrix, then hand-check examples with two findings. Softmax mass summing to one is incompatible with independent positives.
- **Correction and risk:** Use independent sigmoid heads and per-label thresholds where labels are nonexclusive. Independent thresholds can create impossible combinations, requiring a downstream consistency rule.
- **Verification:** Report per-label precision/recall, calibration, and observed invalid-combination rate.

#### Drill 24 — Abstention blind spot
- **Observed symptom (illustrative):** A confidence threshold suppresses predictions but no owner receives the suppressed work.
- **Competing causes:** Low confidence may indicate ambiguity, distribution shift, calibration error, or merely a conservative score scale.
- **Discriminating diagnostic:** For each threshold compute coverage, error among accepted cases, abstentions per hour, reviewer capacity, and age of the oldest unresolved item.
- **Correction and risk:** Define queue owner, SLA, safe fallback, and overflow rule before enabling abstention. Raising the threshold can improve accepted accuracy while creating an unsafe backlog.
- **Verification:** Simulate a peak-rate interval and demonstrate bounded queue age or deliberate service degradation.

#### Drill 25 — Edge-device quantization
- **Observed symptom (illustrative):** INT8 changes decisions near the action threshold.
- **Competing causes:** Calibration set range is unrepresentative; operator implementation differs; or threshold was copied from float evaluation.
- **Discriminating diagnostic:** On paired device outputs calculate `delta = score_int8 - score_fp`, plot it by score and input slice, and count threshold crossings.
- **Correction and risk:** Recalibrate with representative inputs, use quantization-aware training, retain selected higher-precision operations, and reselect threshold. Higher precision can violate memory or latency limits.
- **Verification:** Requalify action metrics and threshold crossings on the target runtime, not an emulator alone.

#### Drill 26 — Asynchronous preprocessing
- **Observed symptom (illustrative):** Serving uses a newer resize/normalization manifest than the weights were evaluated with.
- **Competing causes:** The model may have regressed; a cache may hold old transforms; or the request trace may misreport the active manifest.
- **Discriminating diagnostic:** Resolve model, transform, and label-map digests from the same request; replay a golden fixture through both artifact tuples.
- **Correction and risk:** Package compatible transforms and weights atomically and reject incompatible tuples. Atomic release may slow independent preprocessing fixes, so use an explicit compatibility migration.
- **Verification:** A deployment test must reject a deliberately mismatched digest pair and preserve the last compatible release.

#### Drill 27 — Human-review feedback loop
- **Observed symptom (illustrative):** Operational reviewer dispositions become training truth without adjudication.
- **Competing causes:** Disagreement can be reviewer variation, ambiguous policy, delayed outcome information, or model-influenced anchoring.
- **Discriminating diagnostic:** Blindly double-label a stratified sample, compare reviewers to later adjudicated outcomes where available, and separate model score from reviewer view.
- **Correction and risk:** Store disposition, rationale, and adjudicated label as distinct fields; sample difficult cases for adjudication. More adjudication increases latency and may reduce the volume of fresh labels.
- **Verification:** Audit a future training batch for field provenance, reviewer agreement, and outcome maturity.

#### Drill 28 — Temporal drift hidden by random split
- **Observed symptom (illustrative):** Random splitting gives high validation scores although deployment occurs after a process or camera change.
- **Competing causes:** Legitimate repeat captures, entity leakage, and a real temporal shift can all produce correlated partitions.
- **Discriminating diagnostic:** Plot capture time and entity/site overlap by split; compare random-split performance with a forward holdout after the release cutoff.
- **Correction and risk:** Use group-aware temporal splits and retain a post-cutoff gate. A strict split can reduce training data and increase uncertainty; it is not evidence of poorer deployment performance by itself.
- **Verification:** Evaluate the untouched post-cutoff interval and disclose support and regime differences.

#### Drill 29 — Source license mismatch
- **Observed symptom (illustrative):** Proposed deployment rights are not established by the checkpoint’s public availability.
- **Competing causes:** The base model, training data, adapter, and serving dependency can have different terms; an internal approval may be expired or narrower than the use.
- **Discriminating diagnostic:** Bind exact artifact digests to their license texts, intended use, distribution path, and approval record; ask the authorized legal owner to resolve ambiguity.
- **Correction and risk:** Replace the artifact or obtain scoped approval before release. Substituting a model changes quality, safety, and evaluation evidence and requires requalification.
- **Verification:** Store the approval scope with the release manifest; do not treat this teaching check as legal advice.

#### Drill 30 — Out-of-distribution false certainty
- **Observed symptom (illustrative):** Unseen materials or layouts receive high class probabilities.
- **Competing causes:** The novelty set may be mislabeled, threshold calibration may be stale, or confidence may simply be non-diagnostic for this shift.
- **Discriminating diagnostic:** Use a labeled, semantically distinct novelty set; compare accepted error, score distribution, and abstention behavior to in-domain data.
- **Correction and risk:** Collect coverage, add a bounded abstention/review route, or narrow the supported input contract. Confidence-only novelty filters can reject valid rare inputs while accepting harmful unknowns.
- **Verification:** Measure novelty handling at the actual action boundary with a declared false-abstention cost.

#### Drill 31 — Resolution-dependent calibration
- **Observed symptom (illustrative):** Identical object categories receive different confidence after a camera resolution change.
- **Competing causes:** Native detail changed, resize interpolation changed, or resolution is proxying for site/illumination.
- **Discriminating diagnostic:** Hold object identity and scene as constant as possible across native resolutions; plot reliability and error by native resolution before resize and by post-resize object pixels.
- **Correction and risk:** Qualify resolution regimes, alter capture requirements, or calibrate only when the calibration set covers each regime. Separate calibrators can overfit sparse slices.
- **Verification:** Preserve a resolution-stratified holdout and check both calibration and action error.

#### Drill 32 — Metric denominator omission
- **Observed symptom (illustrative):** A recall number cannot be reproduced because ignored frames and eligibility are undefined.
- **Competing causes:** Exclusions may be legitimate quality rules, accidental filtering, or a change in the unit from object to image.
- **Discriminating diagnostic:** Reconstruct the cohort table: eligible count, exclusions by reason, TP, FN, and whether each image can contain multiple targets.
- **Correction and risk:** Freeze the eligibility rule before comparison and publish numerator, denominator, and exclusion reasons. Broadening eligibility may lower a metric but improve decision relevance.
- **Verification:** Independently recompute the metric from the frozen cohort manifest.

#### Drill 33 — Capacity claims from FLOPs
- **Observed symptom (illustrative):** A design is admitted from MAC counts despite unknown decode, preprocessing, transfer, and queue costs.
- **Competing causes:** Compute may dominate, memory bandwidth may dominate, or upstream image decode may be the bottleneck.
- **Discriminating diagnostic:** Trace end-to-end request phases on target hardware: decode, transform, host/device transfer, inference, postprocess, and queue wait; report p50/p95 by offered load.
- **Correction and risk:** Use FLOPs for relative model budgeting and size service capacity from measured traces. Optimizing one phase can shift contention to another.
- **Verification:** Run a declared load sweep to saturation with bounded queue policy; label results as environment-specific.

#### Drill 34 — Retraining on predictions
- **Observed symptom (illustrative):** High-confidence pseudo-labels amplify a site-specific shortcut in the next model.
- **Competing causes:** The model may be wrong, the site distribution may legitimately differ, or the pseudo-label selection rule may exclude hard but important examples.
- **Discriminating diagnostic:** Stratify sampled pseudo-labels by site, class, confidence, and predicted outcome; compare to blind human adjudication and inspect confidence-error curves.
- **Correction and risk:** Require adjudicated anchors, cap pseudo-label influence, isolate the experiment, and retain a fixed human-labeled holdout. Filtering to high confidence can entrench majority classes.
- **Verification:** Compare the new model with the frozen baseline on human-adjudicated site slices before promotion.

**Takeaway:** The useful unit of troubleshooting is symptom plus a discriminating observation, not a generic “retrain” response.

### 11.1 Answered interview questions

1. **Why can two 3x3 layers be preferred to one 5x5 layer?**  
   **Answer:** Two stride-one 3x3 layers have a 5x5 effective receptive field and an intervening nonlinearity; for equal channel width, they have `18C²` rather than `25C²` weights (biases omitted). That is neither a latency claim nor a guarantee of better accuracy. **Follow-up:** profile the target graph when launches or memory traffic dominate. **Red flag:** claiming parameter count proves device throughput. **Score 3:** derives the comparison, names the hardware boundary, and proposes a trace.

2. **Derive output width with dilation.**  
   **Answer:** Dilation gives `K_eff=d(K-1)+1`, so `W_out=floor((W+2P-K_eff)/S)+1` under symmetric padding and the declared framework rounding. For `W=32,K=3,d=2,P=2,S=1`, `K_eff=5` and `W_out=32`. **Follow-up:** test an even-sized input and border fixture. **Red flag:** using `K` instead of `K_eff`. **Score 3:** states convention and checks shape.

3. **What evidence justifies partial fine-tuning?**  
   **Answer:** A frozen probe fails a locked target-slice gate; a partial-unfreeze candidate improves that gate without breaking retained-domain and calibration gates on the same split. **Follow-up:** if random-image score rises alone, reject the conclusion. **Red flag:** changing split, augmentation, and unfreezing together. **Score 3:** requires a controlled comparison and rollback criterion.

4. **When is attention a poor first remedy?**  
   **Answer:** When the failure is an input-contract mismatch, insufficient object pixels, label ambiguity, leakage, or an invalid decision target. Attention changes feature interaction, not the truth of the labels. **Follow-up:** inspect fixture tensors, annotation policy, grouped split, and error slices first. **Red flag:** adding attention to “explain” a wrong crop. **Score 3:** names a falsifiable upstream check.

5. **How do you choose augmentation?**  
   **Answer:** Specify the real deployment nuisance, label-invariance argument, bounded parameter range, and harmed slice; inspect transformations with a qualified label owner and compare to a no-transform baseline on the identical split. **Follow-up:** rotation is disqualified if orientation changes class meaning. **Red flag:** applying visually plausible transforms without semantic review. **Score 3:** includes a disqualifier and evaluation protocol.

6. **Why show support with slice recall?**  
   **Answer:** `9/10` and `900/1000` are both 0.9 but support radically different confidence in the estimate. State eligible positives, TP, exclusions, and an interval suited to the protocol. **Follow-up:** do not rank tiny slices as stable. **Red flag:** reporting a percentage without denominator. **Score 3:** distinguishes uncertainty from performance.

7. **What is catastrophic forgetting here?**  
   **Answer:** Adaptation improves target loss while degrading a still-required acquisition regime or task on a frozen comparable suite. Parameter distance does not measure it. **Follow-up:** if the old task is retired, document the scope change rather than hiding regression. **Red flag:** declaring success from target loss alone. **Score 3:** names retained gates and decision scope.

8. **Why is a model card insufficient?**  
   **Answer:** It supplies artifact provenance but does not validate local sensor compatibility, split integrity, threshold, review capacity, rollback, or target performance. **Follow-up:** bind the card to immutable release and evaluation records. **Red flag:** treating public documentation as a local safety result. **Score 3:** separates source evidence from local validation.

9. **How do you investigate a sudden confidence increase?**  
   **Answer:** Compare preprocessing/model digests, input statistics, prevalence, and calibration slices before calling it quality. With delayed labels, use shadow review. **Follow-up:** compare paired fixtures first to isolate a pipeline change. **Red flag:** equating higher softmax scores with accuracy. **Score 3:** proposes both interface and outcome checks.

10. **What changes for a multi-site rollout?**  
    **Answer:** Evaluate the site-aware decision: group/split by site where possible, preregister blocking slices, estimate per-site review load, and state whether thresholds may vary. **Follow-up:** a global mean cannot waive a site failure. **Red flag:** pooling sites before checking capture and label policy differences. **Score 3:** names a site-level promotion gate.

11. **How does depthwise separable convolution change design?**  
    **Answer:** Depthwise spatial filtering has approximately `K²C_in` weights and its pointwise mixer `C_inC_out`, versus `K²C_inC_out` dense weights. It changes connectivity and kernel behavior, not just arithmetic. **Follow-up:** profile the actual target kernel. **Red flag:** omitting the pointwise stage from comparison. **Score 3:** gives both formula and deployment caveat.

12. **What is the first oracle for serving mismatch?**  
    **Answer:** A versioned immutable fixture with input bytes, orientation, transformed tensor shape/range, and expected output. It isolates contract divergence before statistical evaluation. **Follow-up:** matching top-1 labels is too weak; inspect intermediate invariants. **Red flag:** debugging only with live traffic. **Score 3:** specifies immutable artifact identities.

13. **Why hold out calibration data?**  
    **Answer:** Choosing a calibration map on the same examples used to judge it biases apparent reliability. Keep calibration selection separate from final assessment; if data are scarce, select a resampling protocol before inspection. **Follow-up:** calibrate by qualified regime only with adequate support. **Red flag:** using accuracy as calibration evidence. **Score 3:** identifies partition purpose and limitation.

14. **When should the system abstain?**  
    **Answer:** When score, input validity, novelty, or policy evidence falls outside the qualified region and a human or safe fallback can own the case. A queue needs an owner, SLA, capacity, and overflow behavior. **Follow-up:** calculate coverage and queue age at peak rate. **Red flag:** dropping cases without a route. **Score 3:** joins uncertainty to operating capacity.

15. **How does a tighter latency limit alter selection?**  
    **Answer:** Rebudget decode, transform, transfer, inference, postprocess, and queueing; then compare resolution, precision, architecture, batching, and workflow fallback while keeping quality gates fixed. **Follow-up:** measure p95 under offered load. **Red flag:** scaling latency from parameter count. **Score 3:** states end-to-end measurement and non-negotiable quality boundary.

**Takeaway:** Defensible interview answers connect a mechanism to a locked protocol, a discriminating check, and a reversible operational decision.


### 11.9 Formula and mechanism inspection cards

Use these compact cards to audit whether an implementation has supplied the variables needed to make a statement meaningful. They summarize earlier derivations; they do not replace them.

#### Inspection 01 — Stride
- **Definition:** Input displacement between adjacent windows.
- **Mechanism:** Changing stride changes sample locations and feature-map geometry.
- **Oracle input:** Check H_out and W_out from declared padding and dilation.
- **Boundary:** Do not call stride an accuracy parameter without target evidence.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-01` must name the model version and evaluation partition.

#### Inspection 02 — Padding
- **Definition:** Added boundary pixels under an explicit convention.
- **Mechanism:** It changes output geometry and which activations touch artificial values.
- **Oracle input:** Visualize border activations on a fixture.
- **Boundary:** Do not assume padding preserves equivariance at borders.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-02` must name the model version and evaluation partition.

#### Inspection 03 — Dilation
- **Definition:** Spacing between kernel taps.
- **Mechanism:** It expands receptive field without increasing tap count.
- **Oracle input:** Use K_eff=d(K-1)+1 before CNN-01.
- **Boundary:** Do not ignore gridding or framework padding rules.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-03` must name the model version and evaluation partition.

#### Inspection 04 — Kernel area
- **Definition:** K_h times K_w learned spatial positions.
- **Mechanism:** Dense convolution weight count grows with area and channels.
- **Oracle input:** Separate weights from H_out W_out activation count.
- **Boundary:** Do not infer runtime solely from parameters.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-04` must name the model version and evaluation partition.

#### Inspection 05 — Groups
- **Definition:** Independent channel partitions in a convolution.
- **Mechanism:** They reduce cross-channel connectivity and parameter count.
- **Oracle input:** Require C_in divisible by groups.
- **Boundary:** Do not substitute groups without considering pointwise mixing.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-05` must name the model version and evaluation partition.

#### Inspection 06 — Depthwise layer
- **Definition:** One spatial filter per input channel.
- **Mechanism:** It captures within-channel spatial structure but does not itself mix channels.
- **Oracle input:** Count K_h K_w C_in weights plus declared bias.
- **Boundary:** Do not compare it to dense convolution without its pointwise stage.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-06` must name the model version and evaluation partition.

#### Inspection 07 — Pointwise layer
- **Definition:** A 1x1 convolution mixing channels at each location.
- **Mechanism:** It trades spatial kernel area for channel projection.
- **Oracle input:** Count C_in C_out weights plus optional biases.
- **Boundary:** Do not claim it sees neighboring pixels.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-07` must name the model version and evaluation partition.

#### Inspection 08 — Receptive field
- **Definition:** The theoretical input region that can influence an activation.
- **Mechanism:** Layer composition grows it through kernel width and jump.
- **Oracle input:** Use the recurrence in Section 11.4.
- **Boundary:** Do not equate theoretical coverage with useful evidence.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-08` must name the model version and evaluation partition.

#### Inspection 09 — Jump
- **Definition:** Input-pixel separation of adjacent feature positions.
- **Mechanism:** Strides multiply the spacing inherited by later layers.
- **Oracle input:** Track j_l alongside r_l.
- **Boundary:** Do not confuse jump with receptive-field width.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-09` must name the model version and evaluation partition.

#### Inspection 10 — Feature map
- **Definition:** Spatial-by-channel activation tensor after an encoder stage.
- **Mechanism:** It carries location and representation before a head reduces it.
- **Oracle input:** Log shape and dtype at every interface.
- **Boundary:** Do not report its width as a parameter count.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-10` must name the model version and evaluation partition.

#### Inspection 11 — Global pooling
- **Definition:** Aggregation across spatial positions.
- **Mechanism:** It provides location-insensitive summary for image-level outputs.
- **Oracle input:** Compare task output grain with discarded coordinates.
- **Boundary:** Do not use it when location itself is required.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-11` must name the model version and evaluation partition.

#### Inspection 12 — Channel attention
- **Definition:** Data-dependent reweighting of feature channels.
- **Mechanism:** It selects or suppresses learned features conditional on input.
- **Oracle input:** Audit output range and stability under benign perturbations.
- **Boundary:** Do not call weights explanations of cause.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-12` must name the model version and evaluation partition.

#### Inspection 13 — Spatial attention
- **Definition:** Data-dependent reweighting of locations.
- **Mechanism:** It can emphasize locations but inherits upstream representation limits.
- **Oracle input:** Test maps under crop and background changes.
- **Boundary:** Do not use it as localization ground truth.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-13` must name the model version and evaluation partition.

#### Inspection 14 — Frozen encoder
- **Definition:** Checkpoint weights excluded from gradient updates.
- **Mechanism:** It limits adaptation degrees of freedom and compute.
- **Oracle input:** Fit a head with grouped validation.
- **Boundary:** Do not assume source features suit new sensors.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-14` must name the model version and evaluation partition.

#### Inspection 15 — Fine tuning
- **Definition:** Gradient updates to some or all pretrained parameters.
- **Mechanism:** It can adapt representation but risks forgetting and instability.
- **Oracle input:** Compare against frozen baseline on locked splits.
- **Boundary:** Do not change split and architecture together.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-15` must name the model version and evaluation partition.

#### Inspection 16 — Regularization to checkpoint
- **Definition:** Penalty on deviation from initial weights.
- **Mechanism:** It encodes a preference for retention, not a guarantee.
- **Oracle input:** State lambda and compare target/retained gates.
- **Boundary:** Do not report small parameter drift as retained utility.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-16` must name the model version and evaluation partition.

#### Inspection 17 — Calibration
- **Definition:** Agreement of scores with observed frequencies in a defined population.
- **Mechanism:** It changes action interpretation rather than representation ranking alone.
- **Oracle input:** Report bins, support, partition, and outcome window.
- **Boundary:** Do not use accuracy as a calibration proxy.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-17` must name the model version and evaluation partition.

#### Inspection 18 — Threshold
- **Definition:** Policy boundary mapping a score to an action.
- **Mechanism:** It determines false-positive load and false-negative exposure.
- **Oracle input:** Compute confusion counts and queue volume at the candidate value.
- **Boundary:** Do not copy it across populations without validation.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-18` must name the model version and evaluation partition.

#### Inspection 19 — Macro recall
- **Definition:** Average of class-specific recalls.
- **Mechanism:** It prevents frequent classes from dominating the average.
- **Oracle input:** Publish per-class supports and eligibility.
- **Boundary:** Do not hide unacceptable rare-class recall in a macro mean.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-19` must name the model version and evaluation partition.

#### Inspection 20 — Artifact digest
- **Definition:** Immutable identifier for weights, code, or manifest.
- **Mechanism:** It makes a released result traceable to exact bytes.
- **Oracle input:** Resolve every digest in a staging replay.
- **Boundary:** Do not replace it with a mutable filename.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-20` must name the model version and evaluation partition.

#### Inspection 21 — Illustrative result
- **Definition:** Constructed numerical teaching input and derived output.
- **Mechanism:** It teaches mechanism without claiming a measurement.
- **Oracle input:** Label values and show arithmetic.
- **Boundary:** Do not promote it to a benchmark or production fact.
- **Evidence label:** DERIVED for the stated mechanism; any workload conclusion remains ILLUSTRATIVE until measured.
- **Review record:** `CNN-INSPECT-21` must name the model version and evaluation partition.

**Takeaway:** A compact term is useful only when its quantity, mechanism, test, and limit remain visible.

### 11.10 Review evidence fields

- **Decision ID:** Name the decision rather than merely the model experiment.
- **Population:** State the deployment group represented by each evaluation row.
- **Cut-off:** Record the latest event or label time eligible for that row.
- **Input digest:** Preserve bytes or an approved immutable fixture identifier.
- **Preprocess digest:** Bind resize, normalization, crop, and orientation behavior.
- **Weight digest:** Identify the exact checkpoint, not only its architecture family.
- **Head digest:** Identify the output and threshold policy paired with the encoder.
- **Split digest:** Preserve group assignment and seed or deterministic construction rule.
- **Metric contract:** State numerator, denominator, exclusions, aggregation, and units.
- **Slice contract:** State which slice can block promotion and its minimum support rule.
- **Capacity contract:** State reviewer capacity, queue window, and overflow path.
- **Fallback contract:** State the owner and behavior when the decision is unavailable.
- **Evidence status:** Mark each result derived, illustrative, sourced, measured, or open.
- **Review conclusion:** State the bounded decision, uncertainty, and recheck trigger.

**Takeaway:** A review record is actionable when another reader can identify the exact artifact, population, policy, and unresolved boundary.

### 11.11 Worked review packets

#### Packet A — Geometry release check
- **Decision:** Can a 32-pixel-wide fixture traverse a dilated convolution without unexpected shrinkage?
- **Inputs:** `W=32`, `K=3`, `d=2`, `P=2`, `S=1`; units are pixels.
- **Calculation:** `K_eff=2(3-1)+1=5`; `W_out=floor((32+4-5)/1)+1=32`.
- **Expected invariant:** Output width equals input width because padding supplies the two missing pixels at each side under this convention.
- **Failure signal:** Width 30 indicates the serving transform omitted padding or uses an incompatible layer definition.
- **Competing cause:** A correct convolution can still receive a 30-pixel crop; inspect input tensor shape before blaming the kernel.
- **Owner/action:** ML engineer records layer config; serving owner replays the immutable fixture.
- **Evidence:** DERIVED and VERIFIED arithmetic; not a measured latency claim.
- **Boundary:** Asymmetric padding or framework-specific rounding changes the calculation.
- **Promotion rule:** Reject artifact pairing if shape differs from the declared fixture trace.

#### Packet B — Depthwise parameter check
- **Decision:** Does a mobile candidate reduce dense spatial parameters as claimed?
- **Inputs:** `K=3`, `C_in=32`, `C_out=64`, one bias per output channel.
- **Calculation:** Dense weights `=3*3*32*64=18,432`; depthwise-plus-pointwise weights `=3*3*32+32*64=2,336`.
- **Interpretation:** The separable pair uses 16,096 fewer weights before biases but has two operations and different channel mixing.
- **Expected invariant:** Depthwise group count equals `C_in`; pointwise input channels equal depthwise output channels.
- **Failure signal:** A layer configured with groups 1 is not depthwise and invalidates this count.
- **Competing cause:** Export tooling may fuse operations without altering mathematical connectivity.
- **Owner/action:** Architecture reviewer checks graph shape and group field, not an architecture name.
- **Evidence:** DERIVED arithmetic; target runtime remains NOT MEASURED.
- **Boundary:** A depth multiplier or bias changes the count.
- **Promotion rule:** Capacity planning must use target-device traces, not this count alone.

#### Packet C — Threshold queue check
- **Decision:** Is a proposed abstention threshold operationally safe?
- **Inputs:** Illustrative arrival rate 40 cases/hour, abstention rate 0.15, reviewer capacity 4 cases/hour.
- **Calculation:** Expected abstentions/hour `=40*0.15=6`; net backlog growth `=6-4=2` cases/hour.
- **Interpretation:** Even with better accepted-case accuracy, backlog grows 16 cases over an eight-hour shift.
- **Expected invariant:** Queue simulation uses the same population and threshold as the score evaluation.
- **Failure signal:** A report gives coverage but omits arrival and review capacity.
- **Competing cause:** Burstiness can create an SLA breach despite average capacity balance.
- **Owner/action:** Operations owner chooses capacity, threshold, or safe fallback before release.
- **Evidence:** ILLUSTRATIVE, arithmetic VERIFIED; no workload measurement is claimed.
- **Boundary:** Real service times and priority rules must replace the constant-rate assumption.
- **Promotion rule:** Demonstrate bounded queue age at peak offered load.

#### Packet D — Slice-recall denominator check
- **Decision:** Does a 90% recall statement support a site release?
- **Inputs:** Site A has `TP=9,FN=1`; Site B has `TP=900,FN=100`; eligible positives only.
- **Calculation:** Each recall is `TP/(TP+FN)=0.90`, but supports are 10 and 1,000.
- **Interpretation:** Equal point estimates do not imply equal evidential stability or equal missed-case counts.
- **Expected invariant:** Exclusions are recorded before numerator/denominator construction.
- **Failure signal:** A dashboard aggregates images when the action contract is per object.
- **Competing cause:** Low support may reflect rare incidence, not necessarily poor capture.
- **Owner/action:** Evaluation owner publishes support and uncertainty protocol beside each slice.
- **Evidence:** ILLUSTRATIVE and VERIFIED arithmetic.
- **Boundary:** Confidence interval choice depends on sampling/dependence; do not infer it from this example.
- **Promotion rule:** Site-specific minimum support and error gates are declared before inspection.

#### Packet E — Quantized threshold-crossing check
- **Decision:** Can an INT8 export retain a float threshold of 0.70?
- **Inputs:** Illustrative paired scores: float `[0.68,0.71,0.74]`, INT8 `[0.66,0.69,0.75]`.
- **Trace:** At threshold 0.70, the second item changes from accept to abstain; one of three action decisions differs.
- **Interpretation:** Mean score delta is inadequate because policy changes occur near the threshold.
- **Expected invariant:** Pair by identical input bytes and matching postprocessing version.
- **Failure signal:** Comparing separately shuffled float and device outputs.
- **Competing cause:** A threshold crossing may come from output dequantization or changed image preprocessing.
- **Owner/action:** Runtime owner logs paired digests; model owner reselects threshold only after device evaluation.
- **Evidence:** ILLUSTRATIVE and VERIFIED trace; hardware performance is NOT MEASURED.
- **Boundary:** Three examples do not estimate real crossing frequency.
- **Promotion rule:** Set a maximum crossing/error rate on a representative qualified holdout.

#### Packet F — Background shortcut check
- **Decision:** Is a high score driven by object evidence or a label-bearing background?
- **Inputs:** One object crop, its original background, one neutral replacement, and one masked-overlay version.
- **Trace:** Hold object pixels and preprocessing fixed; score all three variants with the same artifact digest.
- **Interpretation:** A large score movement after a background swap is evidence to investigate shortcut reliance, not proof of causal explanation.
- **Expected invariant:** Geometry, scale, and compression must stay constant across the variants.
- **Failure signal:** A replacement changes object boundary or introduces an unnatural edge, confounding the result.
- **Competing cause:** Context may contain legitimate information required by the declared task.
- **Owner/action:** Data owner checks collection policy; model owner evaluates counterfactual and held-out-site slices.
- **Evidence:** Method is DERIVED; outcome is OPEN until run on a documented review set.
- **Boundary:** Synthetic counterfactuals may be out of distribution.
- **Promotion rule:** Do not remove context without showing that the intervention preserves task semantics.

#### Packet G — Temporal split check
- **Decision:** Is a random image split leaking future process conditions?
- **Inputs:** Capture timestamps, entity/site IDs, training cutoff, and proposed deployment date.
- **Trace:** Build a table of min/max time by partition and count entities that appear on both sides.
- **Interpretation:** Future captures in training invalidate a deployment-time claim even if labels are correct.
- **Expected invariant:** The cutoff uses availability time, not a later label-adjudication timestamp.
- **Failure signal:** A high random-split result collapses on a post-cutoff holdout.
- **Competing cause:** Repeated captures can be legitimate only if the decision will see that same entity history.
- **Owner/action:** Evaluation owner freezes the group/time rule; data owner documents late labels.
- **Evidence:** Diagnostic method DERIVED; local impact remains OPEN until evaluated.
- **Boundary:** Temporal order does not itself prove drift.
- **Promotion rule:** Use the untouched post-cutoff interval for final reported evidence.

#### Packet H — Artifact-pair compatibility check
- **Decision:** Are weights and preprocessing safe to release together?
- **Inputs:** Weight digest, transform digest, label-map digest, immutable fixture bytes, expected tensor ranges.
- **Trace:** Resolve all three digests from a single release manifest, replay fixture, then compare shape, dtype, range, and output schema.
- **Interpretation:** This check detects interface drift before aggregate quality metrics hide it.
- **Expected invariant:** A manifest is immutable and names exactly one compatible tuple.
- **Failure signal:** A mutable “latest” transform resolves differently during replay.
- **Competing cause:** A numerical tolerance issue can be hardware-dependent rather than a semantic transform mismatch.
- **Owner/action:** Release owner blocks incompatible tuple; platform owner preserves prior manifest for rollback.
- **Evidence:** Mechanism DERIVED; observed compatibility requires a recorded replay.
- **Boundary:** Passing fixtures cannot prove broad statistical correctness.
- **Promotion rule:** Require both fixture compatibility and locked evaluation gates.
## 12. Quick Reference Card

| Item | Rule |
|---|---|
| Input/output | Declare image grain and class/box/mask requirement |
| Geometry | Use CNN-01 with stated padding, stride, dilation |
| Parameters | Use CNN-02; distinguish weights from activations |
| Transfer | Start with frozen baseline; unfreeze only against grouped evidence |
| Evaluation | Split by deployment dependency; report meaningful slices |
| Avoid | Random correlated image splits and unsupported “attention explains” claims |
| Evidence | All workload values here are illustrative; source bodies unavailable |
