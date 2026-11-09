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

A dense convolution produces approximately H_out*W_out*C_out*K_h*K_w*C_in multiply-accumulate operations per image; this is an operation count, not observed latency. For the illustrative 112x112 output, 7x7x3 to 64 channels costs 112*112*64*7*7*3 = 118,013,? MACs: 12,544*64=802,816; 802,816*147=118,013,952 MACs. Activations alone contain 802,816 values. At FP16 that payload is 1,605,632 bytes, about 1.53 MiB, excluding gradients, workspace, and later layers.

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

1. **Why split by machine rather than image?** Images from one machine share nuisance factors; grouping estimates unseen-machine performance.
2. **Calculate a 3x3, 64-to-128 convolution's weights with bias.** CNN-02 gives 73,856.
3. **When does a frozen encoder win?** When a target validation protocol supports feature reuse and adaptation cost/risk is unjustified.
4. **Why is global pooling risky for defects?** It can remove the location required for a box or mask decision.
5. **Does attention explain a decision?** No; a learned weight is not a causal explanation and needs separate evidence.
6. **A site split drops recall. First action?** Audit split, labels, acquisition metadata and prevalence before changing architecture.
7. **What changes if latency halves?** Re-evaluate resolution, stage depth, batching and operational acceptance; do not extrapolate from parameter count.
8. **Why might image augmentation hurt?** It can alter a label-defining feature or create acquisition states absent at inference.
9. **How test catastrophic forgetting?** Freeze comparable source and target suites before adaptation and require both gates.
10. **What is the wrong response to calibration failure?** Treating score ranking or accuracy as a probability fix; fit/evaluate calibration on representative validation data.

**Takeaway:** Good answers state an assumption, a check, and a decision consequence.

### 11.1 Interview answer rationale and scoring rubric

| Question | Reasoning an excellent answer must show | Follow-up constraint | Acceptable alternative | Red flag | Score |
|---|---|---|---|---|---:|
| Why group by machine? | Correlated acquisition nuisance can make image-level holdout optimistic; the group must exist at deployment. | What if only two machines exist? | A time/site split if it better represents rollout. | “Random split is always unbiased.” | 0–3 |
| Compute layer weights. | Identify kernel, input channels, output channels and bias before multiplying. | Set groups=64. | Explain depthwise plus pointwise decomposition. | Counts activations as parameters. | 0–3 |
| Freeze or unfreeze? | Begin from target task, split, data volume and source-target difference. | Target labels are noisy. | Train a calibrated linear baseline first. | “Always fine-tune all layers.” | 0–3 |
| Why not use attention as explanation? | Attention is a learned route/weight, not an intervention demonstrating cause. | Reviewer demands explanation. | Use input perturbation with explicit limits and human evidence. | Calls heatmap proof. | 0–3 |
| New-site recall fell. | Separate site shift, threshold/calibration, preprocessing, and label quality. | No labels at new site. | Run shadow collection and abstain/human review. | Retrain immediately on predictions. | 0–3 |

Score 0 means an incorrect or unsupported claim; 1 names a relevant term; 2 gives a correct mechanism and check; 3 also states the decision consequence and boundary. The rubric assesses reasoning, not memorisation of a particular architecture.

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
