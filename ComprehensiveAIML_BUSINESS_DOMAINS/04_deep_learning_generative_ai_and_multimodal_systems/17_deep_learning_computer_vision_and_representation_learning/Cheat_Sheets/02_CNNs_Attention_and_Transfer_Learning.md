---
collection: "module-17-vision"
file_id: "02"
topic: "CNNs, attention, and transfer learning"
level: 1
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Change in input resolution, deployment device, label policy, or pretrained checkpoint license."
notation_style: "plain-text fenced"
prerequisites: ["Basic linear algebra, convolution, train/validation/test separation"]
pattern_tags: ["vision", "representation", "transfer-learning"]
source_status: "OPEN — visible source names inventoried; original bodies unavailable"
owner_role: "Vision ML reviewer"
source_index: "INDEX.md"
---

# CNNs, Attention and Transfer Learning

> **Purpose:** Select a visual representation and an adaptation policy from four measurable facts — label volume, resolution, latency budget, and shift profile — instead of from architecture fashion.
> **Prerequisites:** Matrix multiplication, supervised loss minimisation, grouped data splits; see the module [INDEX](../INDEX.md).
> **Key outcome:** Given image size, feature-map shape, label volume and deployment constraints, compute a layer's output geometry, parameters and multiply-accumulates, then defend either a frozen-backbone baseline or a fine-tuned one with a stated failure boundary.
> **This file does NOT cover:** detector matching and dense-mask losses, generative image synthesis, or video temporal modelling.
> **Level:** 1 Foundations.
> **Version / reviewed:** 2.0 / not reviewed.
> **Evidence:** DERIVED layer arithmetic and transfer argument; VERIFIED illustrative calculations (recomputed independently in §5); SOURCED architecture definitions named in §10; no deployment or accuracy claim.
> **Retrieval cue:** If validation accuracy is high but a new camera fails, start at §8 and run checks F-02, F-05 and F-09.

## 1. File Header

The decision this sheet supports is not “CNN or transformer”. It is a four-part operational question. How many independent labelled examples exist, at what resolution, under what latency and memory ceiling, and what kind of distribution shift will the deployed input suffer? Every section below feeds one of those four facts, because a representation choice that ignores any one of them is a guess wearing a benchmark number.

Convolution and attention are both ways of sharing parameters across space. They differ in what they share and in what they are allowed to condition on. A convolutional layer shares one small filter across every position, so its output at a position depends on a fixed local neighbourhood and its weights are independent of the input content. An attention layer computes its weights from the input itself, so it can route information between distant positions but must learn, from data, which positions matter. That single contrast explains almost every practical difference in this file: sample efficiency, resolution flexibility, memory growth, interpretability, and the shape of the failure modes.

**The four artifacts this sheet produces.** Each later section serves one row of the table below; if a row is missing from a project, the decision is unfalsifiable.

| Artifact | Question it freezes | Fails when |
|---|---|---|
| Geometry sheet | What is the spatial size and channel count at every stage for a stated input? | Shapes are recomputed from memory and disagree between author and reviewer |
| Cost sheet | How many parameters, MACs, activations and bytes does one forward pass need? | The model is sized by parameter count alone and blows the activation budget |
| Adaptation contract | Which parameters move, at what learning rate, with which frozen statistics? | Half the team believes the backbone is frozen and half believes it is fine-tuned |
| Evidence gate | Which comparison licenses promotion, on which split, at which threshold? | Two models are compared under different preprocessing or different splits |

**Contract version vector.** Freeze the comparison as a tuple rather than a sentence:

```text
A = (backbone, pretrain_corpus, input_resolution, aug_policy, frozen_set,
     lr_schedule, split_key, epoch_budget, decode_rule)
```

Two accuracy numbers are comparable only when their tuples agree on every component that can move the estimator. “Fine-tuning beat the linear probe by four points” is not a claim about adaptation if the fine-tuned run also used a larger input resolution, because resolution changes the effective receptive field and the token count at the same time.

**How the tuple is used in practice.** Write it into the run manifest before the run starts, not into the report afterwards. A manifest written afterwards is a reconstruction, and reconstructions quietly acquire the settings the author believes were used. The test is mechanical: a reviewer must be able to reproduce the run by reading only the manifest and the commit hash, with no verbal context from the author. Where a component was inherited from a checkpoint rather than chosen — the normalisation statistics and the label vocabulary are the usual two — the manifest must say so, because inherited values cannot be tuned and cannot be compared.

**What a tuple diff looks like when a comparison is invalid.** Two runs that differ in `input_resolution` but nothing else still differ in three derived quantities: the effective object scale, the receptive field measured in input pixels, and the token count. So a single-line manifest diff can hide a three-way change. Treat any diff in a geometric component as a new hypothesis rather than as a variant of the old one, and give it its own comparison against a common reference run.

**Evidence policy for this file.** All workloads, label counts, camera counts, annotator rates, GPU hours and money figures are ILLUSTRATIVE constructions whose arithmetic the reader can recompute. Definitions of convolution, batch normalisation, residual connections, depthwise-separable convolution, scaled dot-product attention and the pretraining-then-fine-tuning protocol are SOURCED conventions from the named primary papers in §10; they are conventions an implementation may legally vary, provided the variation is stated. No value here is a measurement of any product, dataset or device, and the collection's source inventory records visible titles only, not source bodies.

**Ownership split.** The vision reviewer owns the geometry and cost sheets and signs the adaptation contract. The data owner owns split eligibility and label provenance. The deployment owner owns the latency and memory ceiling and therefore owns any veto on resolution. A role may not silently change another role's frozen value; every change to a tuple component forces a new comparison, not an edit to an old one.

**Reading paths.** A reader who needs to size a model today should read §3, §4.1, §5 and §12 in that order; the geometry and cost formulas are self-contained. A reader defending a transfer decision in review should read §3, §7 and §8, because the failure table is where the argument usually breaks. A reader inheriting a model that already underperforms on a new site should start at §8 and use the discriminating checks before touching hyperparameters.

**What changes force a re-read.** Any change to input resolution, channel count, kernel size, stride, dilation or grouping invalidates the geometry sheet. Any change to which parameters are trainable invalidates the adaptation contract. Any change of pretraining corpus, its licence, or its preprocessing invalidates the comparison, because a checkpoint carries the normalisation statistics and the label vocabulary of its own training distribution.

**Scope boundaries against neighbouring sheets.** Detector matching, anchor assignment and mask losses belong to the detection sheet; this file supplies the encoder geometry those methods consume and nothing more. Evaluation algebra — threshold selection, AP interpolation, calibration error, queue sizing — belongs to the evaluation sheet, and this file refers to it rather than duplicating it. Self-supervised pretraining objectives are covered where they are taught as a training method; here they appear only as a source of checkpoints whose inductive bias must be characterised before use.

**What this sheet refuses to answer.** It will not tell you which encoder is more accurate on your data, because that is a measurement only your split can supply. It will not tell you whether attention is “better” than convolution, because the answer depends on the label budget and the shift profile, both of which are yours. It will not predict wall-clock latency from a FLOP count, because compilers, memory layout and quantisation move latency by more than the arithmetic does.

**Maintenance trigger for this file.** Re-derive the arithmetic whenever a named architecture's block definition changes, and re-check the failure table whenever a deployment incident is traced to a mechanism not listed in §8. The failure table is the part of this file that ages fastest, because it is a catalogue of observed mechanisms rather than of mathematical facts, and a mechanism that is missing from it is the one that will surprise the next team.

**Takeaway:** An encoder is a resource-and-evidence contract, not an identity; name the four facts before naming the architecture.

## 2. Concept Map / Mental Model

The discipline is one forward pipeline with two backward loops. Read the map top to bottom. Every arrow that points left is a place where an apparently good result becomes invalid without any weight changing.

```text
pixels x [H,W,C]
   |  resize + normalise (statistics come from the pretraining corpus, not yours)
   v
encoder f_theta
   |-- stage 1: stem conv, high resolution, few channels
   |-- stage 2..k: strided or pooled downsampling, channel growth
   +-- head: pool -> linear -> logits [K]
                 |
                 +--> loss (cross-entropy + augmentation-consistent targets)
                 +--> gradient only into parameters the adaptation contract allows
shifted input (new camera / site / season)
   |
   v
slice evaluation -> recalibrate threshold | adapt weights | abstain | reject the input
```

Three quantities move together under every architecture change, and confusing them is the most common analytical error in this area. Spatial size H' x W' sets the number of positions. Channel count D sets how many numbers describe each position. Depth k sets how many times information can be mixed. A layer that halves spatial size and doubles channels keeps the element count roughly constant while reducing the number of positions fourfold, which is why downsampling is the main lever on both compute and localisation precision.

**Invariant 1 — geometry closure.** For every stage, the stated output size must satisfy the convolution counting formula exactly, with integer arithmetic and no rounding. A stage that “roughly” produces 56 x 56 is a stage whose parameter count and MAC count are both wrong.

**Invariant 2 — channel continuity.** The input channel count of a layer equals the previous layer's output channel count after accounting for groups and concatenation. Residual addition requires equal channel counts or an explicit projection.

**Invariant 3 — parameter independence from resolution.** Convolutional parameter count depends on kernel size, input channels, output channels and groups; it never depends on H or W. MAC count scales linearly with output area. Any cost model that makes parameters grow with resolution has confused a fully connected layer with a convolution.

**Invariant 4 — receptive field monotonicity.** The receptive field of a unit never shrinks with depth. It grows by (k-1)·j at each layer, where j is the accumulated stride. Inserting a stride-2 layer grows the field and reduces the sampling density at the same time, which is the mechanism behind the classic “deep but blind” failure.

**Invariant 5 — frozen statistics are not frozen behaviour.** Freezing weights but leaving batch normalisation in training mode updates running means and variances, so the model changes even though no gradient reaches its kernels. This invariant is violated in a large share of transfer-learning regressions.

**Invariant 6 — adaptation is a constrained optimisation.** Fine-tuning asks which parameters may move, in which order, at which rate. It is not a single switch. A model with a frozen stem and a trainable head has a different hypothesis space from one with all layers trainable at a tenth of the head learning rate.

**Where attention enters the map.** Replace the fixed-weight mixing of a convolution with content-dependent mixing. A convolution computes y[p] = Σ_r w[r]·x[p+r]. Attention computes y[p] = Σ_q softmax(q[p]·k[q]/√d)·v[q], where the weights depend on both p and q. The set of q is usually the whole image, so the cost is quadratic in the number of positions instead of linear in the kernel area.

**Where transfer enters the map.** Pretraining supplies θ, and adaptation chooses a subset S of parameters to update and a learning-rate map over S. Everything else — normalisation, label decoding, threshold — is a separate contract that pretraining does not settle.

**The three questions a reviewer asks, in order.** First, does the stated geometry close, meaning do the shapes multiply out to the claimed parameter and MAC totals? Second, is the comparison tuple identical apart from the variable under test? Third, is the failure mode that would invalidate the result one of the entries in §8, and if so which check was run? A design document that answers all three in one page is reviewable in ten minutes; one that answers none is reviewable only by re-execution.

**The one diagram to draw first.** Draw the channel ladder, not the block diagram. Write the output shape and channel count at every stage, in one column. Ninety percent of geometry errors are visible in that column: a channel count that does not match its consumer, a spatial size that does not halve when the stride says it should, a head that expects a different flattened width. Block diagrams hide these because boxes are drawn with names rather than shapes.

```text
ladder template
stage   op                H'   W'    C     params     MACs
 0      input            224  224     3          0        0
 1      stem 7x7 s2 p3   112  112    64      9,472  117,997,568
 2      pool 3x3 s2 p1    56   56    64          0        0
 3      block x2          56   56    64    147,712  462,422,016
 4      block x2          28   28   128    590,336  462,422,016
 5      block x2          14   14   256  2,360,320  462,422,016
 6      block x2           7    7   512  9,439,232  462,422,016
 7      global pool        1    1   512          0        0
 8      linear             1    1    12      6,156        6,144
```

**Why the ladder makes the cost equality obvious.** In the template above, stages 3 through 6 each cost 462,422,016 MAC. That is not an accident of these numbers; it is what a balanced ladder produces, because each stage quarters the area and quadruples the width. A ladder where one stage is ten times the others tells the designer immediately which stage to change, without any profiling.

**What the map deliberately omits.** It omits the loss, the optimiser and the data loader, because those are protocol artefacts that the tuple in §1 handles separately. It omits augmentation, because augmentation changes the distribution the model sees, not the shape it computes. And it omits the deployment runtime, because a compiler or a quantiser can change latency by more than the architecture choice does — which is exactly why the latency measurement in §9 is on the target device rather than in a FLOP table.

**Takeaway:** Keep four ledgers — shapes, parameters, MACs, trainable set — and update all four whenever anything changes.

## 3. Foundational Definitions

Definitions below are written to be operationally checkable. For each term the “not the same as” column exists because the neighbouring concept is what a reviewer will actually be thinking of when they disagree with you.

| Symbol | Meaning | Units / shape | Valid range |
|---|---|---|---|
| H, W | input height and width | pixels | positive integers |
| C | input channels | count | 1 for grey, 3 for RGB |
| k | kernel (filter) side for a square kernel | pixels | odd by convention, ≥1 |
| s | stride | pixels per step | ≥1 |
| p | padding per side | pixels | ≥0 |
| d | dilation | spacing multiplier | ≥1 |
| g | groups | count | divides C_in and C_out |
| H', W' | output spatial size | pixels | positive integers |
| D | channel depth at a stage | count | ≥1 |
| r | receptive field side | pixels | ≥1 |
| j | accumulated stride (jump) | input pixels per output pixel | ≥1 |
| N | number of spatial positions (tokens) | count | ≥1 |
| MAC | multiply-accumulate | operations | ≥0 |
| θ | encoder parameters | tensor set | — |
| S | trainable subset of θ | set | subset of θ |

**Convolution.** A linear operator that applies the same learned kernel at every spatial position, summing over input channels. It is *not* a correlation in the strict signal-processing sense; every deep-learning framework implements cross-correlation and calls it convolution, so the kernel is not flipped. This matters only when comparing against a hand-written filter library, but it matters absolutely there.

**Kernel size.** The spatial extent of one filter. It is *not* the receptive field of the layer, except for the first layer, where the two coincide. A 3 x 3 kernel inside a stack of twenty layers contributes to a much larger receptive field than 3 pixels.

**Stride.** The step between kernel placements. It is *not* pooling, though both downsample; stride is learned-linear and pooling is fixed-nonlinear, and a strided convolution can therefore learn to preserve information that max-pooling discards.

**Padding.** Extra values placed around the border, usually zeros, used to control output size. It is *not* neutral: zero padding creates a border effect where edge pixels see fabricated black, so a model trained on centre-cropped objects can behave differently on border-touching ones.

**Dilation.** Spacing between kernel taps, which enlarges the receptive field without adding parameters. It is *not* equivalent to a larger kernel: a dilated kernel samples a sparse grid, so fine structure between taps is invisible at that layer.

**Groups.** A partition of input and output channels into independent convolution blocks. It is *not* channel pruning; grouping reduces compute while keeping all channels present, whereas pruning removes them.

**Depthwise-separable convolution.** A depthwise convolution (one filter per input channel, g = C) followed by a 1 x 1 pointwise convolution. It is *not* a low-rank factorisation of a dense kernel in the linear-algebra sense, though its cost saving comes from the same intuition of factorising a large operator into two smaller ones.

**Feature map.** The tensor of activations at a stage, of shape [batch, D, H', W']. It is *not* the same as the token sequence used by a transformer, which is a reshape of the same numbers into [batch, N, D].

**Receptive field.** The region of the input that can influence one output unit. It is *not* the effective receptive field, which is the region that carries non-negligible gradient in practice and is typically much smaller and Gaussian-shaped around the centre.

**Batch normalisation.** A per-channel normalisation using batch statistics during training and running estimates during inference, followed by a learned scale and shift. It is *not* a regulariser by design, though it has regularising side effects; its inference behaviour depends entirely on the running statistics, which is why transfer across domains is fragile.

**Residual connection.** An identity path added to a transformed path, y = x + F(x). It is *not* merely an optimisation trick for gradients; it also makes the default function close to identity, which is why very deep stacks can start near a shallow solution.

**Attention.** A weighted average of values where the weights are computed from a query and a set of keys and normalised with softmax. It is *not* a lookup table, because the weights are soft and every position contributes a little to every output.

**Scaled dot-product attention.** The specific form softmax(QKᵀ/√d_k)V. The √d_k divisor is *not* cosmetic: without it the dot products grow with dimension, the softmax saturates, and gradients vanish.

**Transfer learning.** Reusing parameters learned on one distribution as the initialisation or feature extractor for another. It is *not* domain adaptation in the formal sense, which usually assumes access to unlabelled target data and makes an explicit distribution-matching objective.

**Linear probe.** Training only a new linear head on frozen features. It is *not* a weak baseline to be skipped; it is the cheapest measurement of how linearly separable the pretrained representation already is on your data.

**Fine-tuning.** Updating some or all pretrained parameters with a small learning rate. It is *not* training from scratch with a good initialisation in the loose sense people mean; the learning-rate map and the frozen set are part of the definition.

**Discriminative learning rate.** A per-layer-group learning-rate multiplier, typically smaller for early layers. It is *not* a regulariser; it encodes a prior that early-layer features are more transferable than late-layer ones.

**Label shift.** A change in class prior between training and deployment with unchanged class-conditional distributions. It is *not* covariate shift, which changes the input distribution; the two require different corrections, and threshold recalibration fixes the first far more often than the second.

| Term | Precise meaning here | Not the same as |
|---|---|---|
| Stage | A resolution band with fixed H', W' and D | A layer |
| Head | The task-specific mapping from pooled features to logits | The encoder |
| Backbone | The feature extractor whose output feeds the head | The whole network |
| Frozen | No gradient and no statistic update reaches these parameters | Low learning rate |
| Warm-up | A schedule segment that raises LR from near zero | Pretraining |
| Cut-out | Random erasure of a rectangular region | Occlusion at inference |
| Mixup | Convex combination of inputs and targets | Label smoothing |
| Test-time augmentation | Averaging predictions over transformed inputs | Ensembling distinct models |
| Pooling | A fixed aggregation over a spatial window, usually max or mean | A learned downsampling layer |
| Global average pooling | Mean over all spatial positions, producing one vector per channel | A flatten plus dense layer |
| Stem | The first block that converts pixels to features | The backbone |
| Stage | A run of blocks at constant spatial size | An epoch |
| Stride-2 downsample | A learned linear map that also halves the grid | Max pooling, which is fixed |
| Dropout | Random unit masking during training only | Weight decay |
| Stochastic depth | Random skipping of whole residual blocks in training | Dropout on activations |
| Weight decay | An L2 penalty on parameters, applied by the optimiser | Learning-rate decay |
| Cosine schedule | A learning-rate curve following half a cosine period | Warm-up, which is its leading segment |
| Label smoothing | Soft targets mixing the one-hot with a uniform distribution | Mixup, which mixes inputs too |
| Mixup | Convex combination of two inputs and their targets | Cutout, which erases a region |
| CutMix | Pasting a rectangular region from one image onto another | Mixup, which blends everywhere |
| Class prior | The frequency of each label in the population | The class weight used in a loss |
| Covariate shift | Change in the input distribution with the label rule fixed | Label shift |
| Label shift | Change in the class prior with class-conditional densities fixed | Concept drift |
| Concept drift | Change in the mapping from input to label | Any change in inputs alone |
| Calibration | Agreement between predicted confidence and observed frequency | Accuracy |
| Temperature | A single scalar dividing logits before the softmax during calibration | The softmax inverse temperature used in sampling |
| Expected calibration error | Weighted mean gap between confidence and accuracy over bins | Brier score |
| Threshold | The operating point converting a score into a decision | The decision rule's cost model |
| Checkpoint | A serialised parameter set plus its normalisation statistics | A model definition |
| Feature cache | Stored encoder outputs reused to fit a head cheaply | A distilled model |
| Preprocessing contract | Resize, crop, colour, scaling and channel order, versioned | The model weights |
| Effective batch | Batch times accumulation steps, for normalisation statistics | The physical batch |

**Definitions that carry a decision, not just a name.**

*Pooling versus strided convolution.* Both halve the grid. Max pooling keeps the strongest response in each window and discards the rest; a strided convolution learns a weighted combination over the window. The decision matters when the discriminative signal is a thin structure: max pooling preserves its peak response, while a strided convolution may average it away unless the learned filter happens to be a peak detector. Conversely, max pooling throws away the position within the window, which reduces localisation precision for dense prediction.

*Weight decay versus learning-rate decay.* Weight decay shrinks parameters toward zero each step and controls effective capacity; learning-rate decay reduces the step size over time and controls convergence. A run whose validation loss oscillates late is usually a schedule problem; a run whose training loss is far below validation loss throughout is usually a capacity or augmentation problem. Conflating them leads to sweeps in which neither variable is actually tested.

*Temperature versus threshold.* Temperature changes the shape of the score distribution and therefore the calibration curve; the threshold chooses a point on that curve. Fitting both on the same held-out split produces an optimistic calibration estimate, because the threshold selection consumes the same data that the calibration was fitted on. Split twice: fit temperature on one part, choose the threshold on another.

*Class prior versus class weight.* The class prior is a property of the population; the class weight is a hyperparameter of the loss. Correcting for an unbalanced prior is a decision-theoretic operation on the scores, while reweighting the loss changes what the model optimises. They can produce similar-looking recall on the training distribution and quite different behaviour under a further shift.

*Covariate shift versus label shift versus concept drift.* Covariate shift moves p(x) and leaves p(y|x) alone; label shift moves p(y) and leaves p(x|y) alone; concept drift moves p(y|x). The first is repaired by adaptation of the representation or of the normalisation statistics, the second by prior correction on the scores, the third only by new labels. Misdiagnosing the third as the second produces a threshold change that looks like a fix for a week and then fails.

*Checkpoint versus model definition.* A checkpoint without its preprocessing contract and its normalisation statistics is not reproducible. The most common silent reproducibility failure in vision is a weights file that is loaded correctly into a pipeline that resizes differently, so the numbers differ while the code review passes.

**Scope boundaries of these definitions.** A “stage” is defined by constant spatial size and width, so a block that changes width without changing resolution sits at a stage boundary and must be counted deliberately. A “frozen” parameter is one that receives no gradient and no statistic update; any looser use of the word is not admitted in this file. A “probe” is a linear classifier fitted on cached features with its own regularisation, so two probes with different regularisation strengths are different measurements.

**Takeaway:** Most disagreements in this area are definitional; pin the symbol, its units and its shape before arguing about the value.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

**Result 1 — output geometry of a single convolution.**

```text
FORMULA      H' = floor((H + 2p - d*(k-1) - 1)/s) + 1
WHERE        H  input side (pixels); p padding per side; d dilation;
             k kernel side; s stride; H' output side (pixels)
CONSTRAINT   the numerator must be >= 0, else the kernel never fits;
             H' must be an integer count, hence the floor
NUMERICAL    H=224, p=3, d=1, k=7, s=2
             numerator = 224 + 6 - 1*6 - 1 = 223
             H' = floor(223/2) + 1 = 111 + 1 = 112
INTERPRETATION a 7x7 stride-2 stem halves the side to 112, not 111,
             because padding restores exactly one extra placement
```

Derivation, without the formula. Place the kernel so its first tap is at coordinate -p. The last legal placement is the largest integer t such that t·s + d·(k-1) - p ≤ H - 1. Rearranging gives t ≤ (H + 2p - d(k-1) - 1)/s, and the number of placements is floor of that plus one. The floor is not a rounding convenience; it is a count of discrete placements, and any model that reports a fractional output size has silently changed the input.

**Result 2 — parameter count.**

```text
FORMULA      P = k*k*(C_in/g)*(C_out/g)*g + C_out   (for grouped conv)
             P = k*k*C_in*C_out + C_out             (when g = 1)
WHERE        C_in, C_out channel counts; g groups; +C_out is the bias
CONSTRAINT   g must divide both C_in and C_out; bias may be omitted when
             a normalisation layer follows, since BN's shift subsumes it
NUMERICAL    k=3, C_in=64, C_out=64, g=1
             P = 9*64*64 + 64 = 36,864 + 64 = 36,928
INTERPRETATION a single 3x3 layer at 64 channels already holds more
             parameters than a small classifier head
```

The formula contains no H and no W. This is the single most useful fact in the sheet for capacity planning: parameters are a property of the channel ladder, while compute is a property of area times channels.

**Result 3 — multiply-accumulate count.**

```text
FORMULA      MAC = H'*W' * k*k*(C_in/g)*(C_out/g)*g
WHERE        H', W' output area; the rest is per-position kernel work
CONSTRAINT   FLOPs convention: MAC ~ 2 FLOPs if one counts the add, and
             many papers quote only MACs; state which convention you use
NUMERICAL    56x56 output, k=3, C_in=C_out=64, g=1
             MAC = 3136 * 9 * 64 * 64 = 3136 * 36,864 = 115,605,504
INTERPRETATION roughly 0.116 GMAC for one layer; a 50-layer network at this
             width therefore lives in the single-digit GMAC range
```

Check the multiplication independently: 9·64 = 576; 576·64 = 36,864; 36,864·3,136 = 36,864·3,000 + 36,864·136 = 110,592,000 + 5,013,504 = 115,605,504. The two routes agree, which is the minimum standard for any number printed in a design document.

**Result 4 — receptive field recursion.**

```text
FORMULA      r_0 = 1, j_0 = 1
             r_l = r_{l-1} + (k_l - 1) * j_{l-1}
             j_l = j_{l-1} * s_l
WHERE        r receptive field side (pixels); j jump, i.e. input pixels
             per one output pixel after layer l
CONSTRAINT   valid for a single-path feed-forward chain; branches,
             concatenations and residual sums require taking the maximum
             over paths and then reasoning about the effective field
NUMERICAL    stem k=3 s=2 then four k=3 s=1 layers
             l1: r=1+2*1=3,   j=2
             l2: r=3+2*2=7,   j=2
             l3: r=7+2*2=11,  j=2
             l4: r=11+2*2=15, j=2
             l5: r=15+2*2=19, j=2
INTERPRETATION one stride-2 layer makes every later 3x3 add four pixels of
             field rather than two, because the jump already doubled
```

**Validity condition for §4.1.** These four results hold for a single-path, zero-padded, square-kernel stack on a dense grid. They are wrong for transposed convolutions, for asymmetric padding, for kernels larger than the padded input, and for any architecture where a channel dimension is spliced rather than summed.

### 4.2 General case

**Padding modes and the “same” shortcut.** With s=1 and odd k, choosing p=(k-1)/2 preserves the side exactly. With even k there is no symmetric padding that preserves the side, so a framework must break the tie and it will choose inconsistently between libraries. Record which side receives the extra pixel if you compare implementations.

**Transposed convolution geometry.** The output side of a transposed convolution is (H-1)·s - 2p + d(k-1) + 1. This is the adjoint of the forward convolution's shape map, not its inverse; information destroyed by a strided forward pass is not restored by the transpose.

**Grouped and depthwise forms.** Setting g = C_in = C_out gives depthwise convolution, whose parameter count is k²·C + C. Composing it with a 1 x 1 convolution gives the depthwise-separable block:

```text
FORMULA      P_sep = k*k*C + C + C*C_out + C_out
             P_dense = k*k*C*C_out + C_out
             ratio = (k*k*C + C*C_out) / (k*k*C*C_out)
WHERE        C = C_in = C_out for a square block
CONSTRAINT   the ratio is only this small when C_out is comparable to C;
             a block that expands 32 -> 256 channels saves much less
NUMERICAL    k=3, C=64, C_out=64
             P_sep = 576 + 64 + 4,096 + 64 = 4,800
             P_dense = 36,928
             ratio = 4,800/36,928 = 0.130
             MAC ratio = (576 + 4,096)/(9*4,096) = 4,672/36,864 = 0.127
INTERPRETATION about 8.7x fewer parameters and 7.9x fewer MACs at equal
             channel counts, at the cost of one extra nonlinearity and
             a smaller mixing capacity per block
```

**Bottleneck block.** A 1 x 1 reduction to C/4, a 3 x 3 at C/4, and a 1 x 1 expansion back to C costs 64·16 + 16·16·9 + 16·64 = 1,024 + 2,304 + 1,024 = 4,352 parameters, versus 36,928 for the plain block: a ratio of 0.118. The MAC count at 56 x 56 is 3,136·4,352 = 13,647,872, a ratio of 0.118 as well, because every term scales with area identically. The lesson is that reduction-expansion factorises along channels, while depthwise factorises along space; they compose, and a block that does both is cheaper than either alone.

**Attention cost.** For N positions and depth D:

```text
FORMULA      MAC_attn = 2*N*N*D            (QK^T and the AV product)
             MAC_linear = 4*N*D*D          (Q, K, V, output projections)
             MAC_mlp = 8*N*D*D             (two-layer MLP at 4x expansion)
             crossover: 2*N*N*D = 4*N*D*D  =>  N = 2D
WHERE        N tokens; D model width; the 4x MLP expansion is a convention
CONSTRAINT   ignores the softmax, layer norms, residual adds and memory
             traffic; it is an arithmetic lower bound, not a latency model
NUMERICAL    N=196 (14x14), D=512
             MAC_attn = 2*196*196*512 = 39,337,984/2 = 19,668,992
             MAC_conv equivalent 3x3 same-shape = 196*9*512*512 = 462,422,016
INTERPRETATION at 196 tokens and width 512 attention mixing is ~23x cheaper
             than a dense 3x3 convolution at the same width; the ordering
             inverts once N exceeds 2D, i.e. N > 1024 at this width
```

**Transfer as constrained optimisation.** Let θ be pretrained parameters, S ⊆ θ the trainable subset, and η_l the learning rate for parameter group l. The adapted model is argmin over {θ_s : s ∈ S} of the empirical risk on the target set, with groups outside S held at their pretrained values. Every practical transfer recipe is a choice of S and of the map l → η_l. Common choices, in increasing cost of evidence required: head only (S = head), last block plus head, all layers at a tenth of head LR, all layers at full LR after a warm-up segment.

**Feature-reuse estimate.** If a pretrained encoder has been trained on n_p examples from a distribution whose low-level statistics resemble the target, the usable fraction of its features decays with the distance between distributions and grows with n_p. A practical screening rule: compute a linear probe on 5–10% of the target labels and compare it with a from-scratch model on the same subset. If the probe wins by more than its own confidence interval, the representation carries transferable signal and full fine-tuning is worth attempting; if it loses, the bottleneck is the target labels, not the initialisation.

**Validity condition for §4.2.** Each result is an arithmetic identity under the stated block definition. None of them is a performance prediction. In particular the crossover at N = 2D says nothing about wall-clock time, because attention's memory traffic and softmax cost dominate at the scales where the arithmetic is near parity.

**Result 5 — full block accounting, including normalisation and residual.**

```text
FORMULA      P_block = k*k*C_in*C_out + 2*C_out        (conv + BN affine)
             P_proj  = C_in*C_out + C_out              (1x1 for a channel change)
             P_total = P_conv_part + P_residual_path + P_block_norms
WHERE        each batch-norm layer contributes 2*C parameters (scale and
             shift); bias in the preceding convolution is redundant when a
             normalisation layer follows and should be omitted
CONSTRAINT   the residual path needs a projection only when C_in != C_out
             or the stride is not 1; otherwise it is parameter-free
NUMERICAL    3x3 conv 128 -> 256, stride 2, followed by BN
             conv: 9*128*256 = 294,912
             BN:   2*256 = 512
             residual 1x1: 128*256 + 256 = 33,024
             total = 328,448 for the projection variant, 295,424 without
INTERPRETATION the residual projection is 10% of the block, which is why
             counting "the convolutions" understates a stage by about a
             tenth wherever channel counts change
```

The bias omission is worth 256 parameters here but matters for a different reason: a convolution with a bias followed by batch normalisation has a redundant degree of freedom, and the two can drift in opposite directions during training without changing the function. Removing the bias makes the parameterisation identifiable, which improves optimiser behaviour at the margin.

**Result 6 — activation memory for a training step.**

```text
FORMULA      B_act = sum over saved tensors of B*H'_l*W'_l*D_l*bytes
             B_train ~ B_act * (1 + checkpointing_fraction_unsaved)
WHERE        B batch size; bytes per element (4 for float32, 2 for bf16)
CONSTRAINT   a backward pass needs every tensor whose gradient is required;
             checkpointing recomputes rather than stores, trading ~33% more
             forward arithmetic for a reduction in stored bytes
NUMERICAL    B=32, stage1 tensor 56x56x64 fp32
             per tensor = 32*3136*64*4 = 25,690,112 B = 24.5 MiB
             four comparable tensors at 24.5 MiB = 98 MiB, before the
             backward buffers that hold gradients of the same shape
INTERPRETATION activations, not weights, decide the batch size; a model
             with 48 MiB of weights can still fail at batch 32
```

Check the multiplication independently: 3,136·64 = 200,704; 200,704·32 = 6,422,528; 6,422,528·4 = 25,690,112 bytes. Dividing by 1,048,576 gives 24.5 MiB. The four-tensor total is 98.2 MiB, and adding gradient buffers of the same shape doubles it to 196.4 MiB.

**Result 7 — arithmetic intensity and the balance point.**

```text
FORMULA      I = 2*MAC / bytes_moved
             balance = peak_FLOP_per_s / peak_bytes_per_s
WHERE        I in FLOP per byte; bytes_moved counts the tensors actually
             read and written, not the parameters (which are reused)
CONSTRAINT   a layer is compute-bound only when I exceeds the machine's
             balance point; below it, latency follows bytes, not FLOPs
NUMERICAL    3x3 conv, 64 channels, 56x56 input and output, fp32
             MAC = 115,605,504, FLOP = 231,211,008
             bytes = input 802,816 + output 802,816 = 1,605,632
             I = 231,211,008/1,605,632 = 144 FLOP/byte
             depthwise conv same shape: MAC = 1,806,336, FLOP = 3,612,672
             I = 3,612,672/1,605,632 = 2.25 FLOP/byte
INTERPRETATION the dense 3x3 layer is 64x more intense than the depthwise
             layer at the same shape, which is the whole latency story of
             mobile architectures in one ratio
```

**Result 8 — learning-rate map under partial fine-tuning.**

```text
FORMULA      eta_l = eta_head * alpha^(L - l)      for l = 1..L
WHERE        eta_head the head learning rate; alpha the decay factor
             per layer group; L the number of groups; l counted from the
             input side so early layers get the smallest rate
CONSTRAINT   alpha < 1; with alpha = 1 the map is uniform and the schedule
             is ordinary fine-tuning; alpha must be re-tuned whenever the
             trainable depth changes, because the gradient scale changes
NUMERICAL    eta_head = 1e-3, alpha = 0.75, L = 12
             earliest group eta = 1e-3 * 0.75^11 = 1e-3 * 0.0422 = 4.2e-5
             ratio head-to-stem = 23.7
INTERPRETATION the stem moves 24x slower than the head, which is a
             statement about expected transferability, not about a
             measured optimum; if the target domain differs in colour
             statistics, the stem is exactly the group you should NOT
             slow down
```

**Result 9 — label-shift correction on the scores.**

```text
FORMULA      logit'_k = logit_k + log(pi'_k / pi_k)
WHERE        pi'_k target class prior; pi_k training class prior
CONSTRAINT   exact for a softmax classifier under pure label shift with
             unchanged class-conditional densities; invalid if the
             shift is also covariate, because then p(x|y) moved too
NUMERICAL    training prior 2% for the defect class, deployment prior 10%
             log(0.10/0.02) = log(5) = 1.609
INTERPRETATION the defect logit rises by 1.61 nats, which at a decision
             threshold near zero moves a large mass of borderline cases
             across the boundary; this is a fivefold prior change and is
             common when a line is retooled
```

This correction is the arithmetic behind F-02's discriminating check: if applying it recovers most of the loss on the new camera, the change was label shift; if it does not, the densities moved and the repair must be representational.

**Result 10 — the sample-efficiency shape of transfer.**

```text
FORMULA      L(n) = a * n^(-b) + c
WHERE        L target error; n target labels; a scale; b decay exponent;
             c irreducible error floor
CONSTRAINT   fitted over a range of n and extrapolated only within that
             range; b is dataset-dependent and not a universal constant
NUMERICAL    a = 0.42, b = 0.18, c = 0.09, all ILLUSTRATIVE
             n = 50:    0.42*50^-0.18 + 0.09 = 0.298
             n = 200:   0.252
             n = 1,000: 0.211
             n = 5,000: 0.181
             n = 25,000: 0.158
             solving 0.42*n^-0.18 = 0.11 gives n = 1,708 for a 0.20 error
INTERPRETATION a decade more labels buys about 0.03 error at the top of
             the range and 0.05 at the bottom, so the marginal label is
             worth less each time; the curve, not a single point, is the
             argument for a label budget
```


### 4.3 Variants

**Variant A — convolutional stem before attention.** Replace the first patch embedding with a small stack of strided convolutions, reducing the token count by 4–16x before the first attention block. The token count falls quadratically in the attention term while the convolutional part stays linear in area. At 224 input with a 4x reduction, N falls from 197 to about 50, and the attention term falls by about 16x.

**Variant B — windowed attention.** Restrict each query to a window of M x M positions. The quadratic term becomes 2·N·M²·D. With N = 3,136 (56 x 56), M = 7 and D = 96, windowed attention costs 2·3,136·49·96 = 29,503,488 MAC, versus 2·3,136²·96 = 1,888,223,232 MAC for global attention at the same width. The ratio is exactly N/M² = 3,136/49 = 64, so the saving grows linearly with the grid and quadratically with a smaller window. Check: 1,888,223,232/64 = 29,503,488, matching the direct computation. Mixing across windows then requires a shifting or a merging mechanism, which is the price of the saving.

**Variant C — multi-head split.** With h heads, each head operates at width D/h, so the quadratic term per head is 2·N²·(D/h) and the total is unchanged. Heads do not save compute; they change the rank and the number of independent attention patterns. A claim that more heads improve accuracy is a claim about inductive bias, never about cost.

**Variant D — frozen backbone with retrained normalisation.** Freezing kernels while re-estimating batch-norm statistics on target data is a distinct recipe from full freezing. It costs one forward pass over unlabelled target images and repairs the most common domain-shift failure at almost no label cost, which is why it is the first repair to try in §8.

**Variant E — partial depth freezing.** Unfreezing only the last L blocks changes the number of trainable parameters roughly linearly in L, and changes the effective learning-rate scale more than linearly, because gradients from early layers accumulate through more nonlinearities. When L changes, re-tune the head LR before concluding anything about transferability.

**Variant F — distillation into a convolutional student.** If the deployment ceiling forbids attention, the pretrained attention model can act as a teacher whose soft targets train a convolutional student. The student inherits the teacher's decision boundaries without inheriting its cost, and the contract that must be frozen is the teacher's preprocessing, including its normalisation statistics.

**Validity condition for §4.3.** Each variant is a cost transformation with a stated mechanism. None is claimed to improve accuracy, and each requires its own comparison under the tuple A in §1.

### 4.4 Limits and counterexamples

**Counterexample 1 — the receptive-field formula overstates what the network uses.** A stack with r = 19 nominally sees 19 pixels, but the effective field, measured by gradient magnitude, is concentrated near the centre. A model can therefore have a nominal field covering an object and still fail when the object's discriminative part lies at the border of the field. The discriminating check is a spatial occlusion sweep: move a grey patch across the input and record the prediction change as a function of patch position.

**Counterexample 2 — parameters are not capacity.** Two models with identical parameter counts can differ by an order of magnitude in MACs, because one applies its channels at 14 x 14 and the other at 112 x 112. Capacity arguments that cite parameter count alone cannot explain why the wider-at-low-resolution model trains faster and generalises differently.

**Counterexample 3 — attention is not automatically global.** A windowed model with M = 7 cannot move information between opposite corners in one layer. Claims about “global context” require counting the number of layers needed for the information path to span the image, which for window attention with shifts is logarithmic in the grid size, not one.

**Counterexample 4 — the 2D crossover ignores memory.** At N = 1,024 and D = 512 the arithmetic terms are equal, but the attention matrix is 1,024² = 1,048,576 scores per head per layer. In float32 that is 4 MiB per head per layer before any backward pass, which usually binds before the MAC count does.

**Counterexample 5 — depthwise savings vanish under expansion.** A separable block that expands 32 input channels to 256 output channels costs 9·32 + 32 + 32·256 + 256 = 288 + 32 + 8,192 + 256 = 8,768 parameters, versus 9·32·256 + 256 = 73,984 dense: a ratio of 0.119, still good, but if the expansion is applied at high resolution the depthwise part dominates MACs and the saving over a dense layer shrinks toward the ratio of the depthwise term alone.

**Counterexample 6 — the linear-probe screening rule fails under label shift.** If the target class prior is very different from the pretraining prior, a probe can look strong on accuracy while being useless at the operating threshold. Screen with per-class recall at a fixed false-positive budget, not with accuracy.

**Counterexample 7 — a “constant per-stage budget” ladder is not automatically optimal.** The equality in §5 Step 3 holds when width doubles exactly as area quarters. A ladder that doubles width only every other stage has stages with very different budgets, and the expensive stage is not necessarily the one to cut, because the cheap stages may be the ones carrying the resolution the task needs. Cost equality is a diagnostic for design intent, not a proof of good design.

**Counterexample 8 — the effective batch is not the physical batch.** With gradient accumulation over eight micro-batches of 16, the normalisation statistics are estimated from 16 samples per micro-batch while the gradient is averaged over 128. Any argument that relies on batch statistics being low-variance is therefore wrong under accumulation, and the fix is a normalisation layer that does not use batch composition.

**Counterexample 9 — a receptive-field argument that ignores stride sampling density.** Two models can have the same nominal field and different sampling density: one reads every pixel in its field, the other reads one in four along each axis. The sparser model has the same support and one sixteenth of the samples, so it is more sensitive to aliasing and more vulnerable to high-frequency texture. Field size and density are separate numbers and both belong in the geometry sheet.

**Counterexample 10 — attention with a single head is not a weaker version of multi-head.** One head can express a single soft partition of the input; several heads can express several simultaneously. A single-head model with the same total width has the same cost and a strictly smaller function class, so “we removed the heads to save compute” is a claim about nothing, since heads do not save compute.

**Validity condition for the whole of §4.** Every formula here is an arithmetic identity on a stated block. The moment a block contains a branch, a concatenation, a learned normalisation, or a non-uniform resolution, the geometry must be recomputed per branch and summed; no shortcut survives contact with a real architecture. A second, stronger condition applies to every interpretive sentence in this section: it describes the consequence of the arithmetic, not a measured property of a trained model, and any sentence that would change if the data changed belongs in §8 or §10 rather than here.

## 5. Worked Numerical Example

**The task (ILLUSTRATIVE).** A 12-class defect classifier on 224 x 224 RGB images, deployed on an edge device with a 30 ms p99 budget and 256 MiB of usable memory for weights plus workspace. Four candidate encoders are on the table, and the question is which one to spend the next three weeks on.

**Step 1 — geometry walk of candidate A.** Stem 7 x 7 stride 2 padding 3, then max-pool 3 x 3 stride 2, then four stages that each halve the side twice.

```text
input                224 x 224 x 3
stem  7x7 s2 p3      112 x 112 x 64     (224 -> 112, verified in 4.1)
pool  3x3 s2 p1       56 x 56 x 64     (112 -> 56)
stage1 2 blocks       56 x 56 x 64
stage2 2 blocks       28 x 28 x 128
stage3 2 blocks       14 x 14 x 256
stage4 2 blocks        7 x 7 x 512
global average pool    1 x 1 x 512
head  linear           1 x 1 x 12
```

**Step 2 — parameters per block.** Each stage contains two 3 x 3 convolutions at constant width. Using P = k²·C² + C:

```text
stage1 block: 9*64*64 + 64      = 36,864 + 64   = 36,928  (x2 convs) = 73,856
stage2 block: 9*128*128 + 128   = 147,456 + 128 = 147,584 (x2)      = 295,168
stage3 block: 9*256*256 + 256   = 589,824 + 256 = 590,080 (x2)      = 1,180,160
stage4 block: 9*512*512 + 512   = 2,359,296 + 512 = 2,359,808 (x2)   = 4,719,616
two blocks per stage: stage1 147,712 | stage2 590,336 | stage3 2,360,320 | stage4 9,439,232
stem 7x7: 9? no — 7*7*3*64 + 64 = 147*64 + 64 = 9,408 + 64 = 9,472
head: 512*12 + 12 = 6,144 + 12 = 6,156
total = 147,712 + 590,336 + 2,360,320 + 9,439,232 + 9,472 + 6,156 = 12,553,228
```

Arithmetic re-check by a different route: 9·512² = 2,359,296 is the dominant term; four such convolutions in stage 4 give 9,437,184, plus four biases of 512 gives 9,439,232. The remaining total is 3,113,996, and 12,553,228 = 9,439,232 + 3,113,996. The two routes agree to the unit.

**Step 3 — MAC count per stage.** Using MAC = area · k² · C²:

```text
stage1: 3136 * 9 * 64 * 64   = 3136 * 36,864   = 115,605,504  per conv, x4 convs = 462,422,016
stage2:  784 * 9 * 128 * 128 =  784 * 147,456  = 115,605,504  per conv, x4        = 462,422,016
stage3:  196 * 9 * 256 * 256 =  196 * 589,824  = 115,605,504  per conv, x4        = 462,422,016
stage4:   49 * 9 * 512 * 512 =   49*2,359,296  = 115,605,504  per conv, x4        = 462,422,016
```

This equality is the most useful coincidence in convolutional design: halving the side quarters the area, doubling the width quadruples the per-position work, and the stage cost stays flat. Recognising it lets a designer predict that a balanced ladder has a constant per-stage budget and that a stage which does not match is the one to fix. Total convolutional MAC = 4·462,422,016 = 1,849,688,064, plus stem 112²·49·3·64 = 12,544·9,408 = 117,997,568 and head 512·12 = 6,144. Forward MAC ≈ 1.968 GMAC.

**Step 4 — the same walk for candidate B, a 25.6M-parameter baseline at 4.1 GMAC.** At 224 input this baseline costs 2·4.1e9 = 8.2 GFLOP forward under the two-FLOPs-per-MAC convention. On a stated 45 TFLOP/s effective throughput the arithmetic lower bound is 8.2e9/45e12 = 0.182 ms, and the nominal ceiling is 1/0.000182 = 5,488 images/s. The same device achieving 1,000 images/s implies 8.2e12 FLOP/s of sustained arithmetic, which is 18% of peak — a plausible efficiency for a small model with limited parallelism, and a signal that the bottleneck is not FLOPs.

**Step 5 — memory.** Weights in float32 are 12,553,228·4 = 50,212,912 bytes ≈ 47.9 MiB; in int8, ≈ 12.0 MiB. Per-image activations for training at float32 with a conservative full-retention assumption are dominated by stage 1: 56·56·64·4 = 802,816 bytes ≈ 0.77 MiB per image per tensor, and there are four stem/stage-1 tensors of comparable size, so a batch of 64 needs roughly 64·4·0.77 ≈ 197 MiB before the backward graph's temporary buffers. This is why a model with 48 MiB of weights can fail to train on a 16 GiB card at batch 256 while inference on the same card is trivial.

**Step 6 — the decision under the constraint.** Candidate A meets 30 ms at the arithmetic level with a wide margin and fits the weight budget in int8. But the honest reading is that neither candidate is excluded by arithmetic; the tie-break has to come from the label budget and the shift profile. With 3,000 labelled defects across 12 classes and a second camera arriving next quarter, the screening experiment is a linear probe of candidate A's pretrained weights on 300 held-out examples, compared against A trained from scratch on the same 300. That experiment costs minutes and is the actual decision procedure.

**Independent arithmetic re-check of the whole example.** Parameter total recomputed from the channel ladder: stem 9,472; stage1 147,712; stage2 590,336; stage3 2,360,320; stage4 9,439,232; head 6,156. Summing in a different order: (9,472 + 147,712) = 157,184; (590,336 + 2,360,320) = 2,950,656; (9,439,232 + 6,156) = 9,445,388; total 157,184 + 2,950,656 + 9,445,388 = 12,553,228. Matches Step 2. MAC total recomputed as 4 stages · 4 convs · 115,605,504 plus 117,997,568 plus 6,144 = 1,849,688,064 + 117,997,568 + 6,144 = 1,967,691,776. The earlier rounded 1.968 GMAC is correct at three significant figures.

**Sensitivity note.** The parameter total is dominated by stage 4 (75%). Halving stage-4 width to 256 would cut parameters by about 9,439,232·(1 - 1/4) = 7,079,424, taking the total to 5.47M, and would cut stage-4 MACs fourfold. The accuracy cost of that change is unknown without measurement, but the direction is predictable: the reduction removes the layer with the largest receptive field, so tasks whose labels depend on long-range context will degrade first. Conversely, if the 12 classes are decided by fine texture, halving stage-4 width is nearly free and the saved budget should be spent on stage-2 width. The sensitivity is therefore not symmetric, and the label structure decides which way to spend it.

**Step 7 — the separable candidate C, and why its FLOP advantage does not settle the latency question.** Replace each 3 x 3 convolution in candidate A with a depthwise-separable block at the same input and output channels. At 64 channels the block costs 4,800 parameters against 36,928, a ratio of 0.130, and its MAC count at 56 x 56 is 3,136·4,672 = 14,651,392 against 115,605,504, a ratio of 0.127. Applied to all sixteen convolutions, the convolutional MAC total falls from 1,849,688,064 to about 234,460,000, so the forward pass drops to roughly 0.35 GMAC.

The arithmetic suggests a 5x speedup. The measurement usually shows 1.5–2.5x. The gap is the arithmetic intensity column of the roofline analysis: the depthwise stage moves the same 1,605,632 bytes per layer while doing 16x fewer FLOPs, so its intensity falls from 144 to about 9 FLOP/byte and the layer stops being compute-bound. On a device whose balance point is around 10 FLOP/byte, the layer has already crossed to the memory side, and further FLOP reduction buys nothing.

```text
candidate A stage-3 layer: 115,605,504 MAC -> 231,211,008 FLOP
                           bytes 1,605,632 -> 144 FLOP/byte  (compute-bound)
candidate C stage-3 block: depthwise  3,136*576  =  1,806,336 MAC
                           pointwise  3,136*4,096 = 12,845,056 MAC
                           total      14,651,392 MAC -> 29,302,784 FLOP
                           bytes ~1,605,632 + 802,816 intermediate
                           intensity ~ 29,302,784/2,408,448 = 12 FLOP/byte
```

The lesson for the decision in Step 6 is that candidate C must be benchmarked, not derived. Its parameter advantage is real and matters for storage on an edge device; its latency advantage is contingent on the target hardware's balance point, which is a measurement, not a property of the model.

**Step 8 — what the three candidates imply for the review.** Candidate A is the reference design and the only one with a fully worked geometry sheet. Candidate B is a checkpoint with a published cost that the team cannot verify internally, so it enters only as a pretrained initialisation. Candidate C is an optimisation that must be gated on a device benchmark. Naming which artefact each candidate is — a design, a checkpoint, or an optimisation — prevents the common mistake of comparing them as if they were three points on the same axis.

## 6. Visual / Geometric Intuition

**What one filter sees.** Picture the input as a grid; one 3 x 3 filter is a 3 x 3 magnifying glass slid across it, producing one number per placement. The channel dimension is not spatial: the filter spans all input channels at once, so a 3 x 3 filter on 64 channels holds 576 weights, not 9. Readers who visualise a 3 x 3 patch of a single channel routinely underestimate parameter counts by 64x.

```text
one output pixel of a 3x3 conv, C_in = 4:

      ch1   ch2   ch3   ch4
k = [ . . .   . . .   . . .   . . . ]   3x3x4 = 36 weights
    [ . . .   . . .   . . .   . . . ]   + 1 bias
    [ . . .   . . .   . . .   . . . ]   -> sum -> one number
```

**Receptive field growth as a cone.** Each layer extends the cone by (k-1)·j at the top. A stride doubles the cone's slope, so after a stride every subsequent layer adds twice as much. This is why a network with a single early stride-2 layer and twenty 3x3 layers has a field comparable to a network with no strides and forty layers, at a quarter of the compute.

```text
layer:      1     2     3     4     5
field:      3     7    11    15    19      (k=3, single stride 2 at layer 1)
jump:       2     2     2     2     2
```

**Downsampling is a compression decision.** Going from 56 x 56 to 28 x 28 discards three quarters of the positions. Whether that is lossless depends on the content: a defect of 2 pixels in a 224 image occupies 1 pixel at 56 and a quarter of a pixel at 28, so it is gone. The geometric rule is that the smallest discriminative structure must be at least one pixel wide at the stage where the decision is made, which fixes the maximum permissible total stride.

**Attention as content-based routing.** A convolution's routing is fixed: position p always looks at p+r for the same set of r. Attention's routing is a matrix of similarities, so two positions that look alike exchange information regardless of distance. Picture the attention matrix as a soft, learned adjacency graph that is different for every image. This is the source of both its flexibility and its sample cost: the graph must be inferred from data, and with few labels it is inferred badly.

**Token grid versus pixel grid.** A 224 image with patch 16 becomes a 14 x 14 grid of 196 tokens; each token is a 16 x 16 x 3 = 768-number vector projected to width D. The spatial resolution available to a dense prediction task is therefore 14 x 14 unless the model re-upsamples. This is why pure transformers are competitive on classification and comparatively awkward on segmentation without a hierarchical or convolutional decoder.

**Feature pyramid intuition.** Objects of different sizes need different resolutions. A single-stride model detects large objects well and small ones poorly, because the small ones are sub-pixel at the deepest stage. The pyramid remedy — predict from several stages and fuse — is a geometric answer to a geometric problem, and it costs memory proportional to the sum of the stage areas, which is dominated by the earliest stage.

**Why 3 x 3 stacks beat 7 x 7 filters.** Three 3 x 3 layers have the same receptive field as one 7 x 7 layer (3 + 2 + 2 = 7) but use 3·9 = 27 weights per channel pair instead of 49, and they insert two extra nonlinearities. The extra nonlinearities are the real reason the stack is more expressive: the composite function is piecewise linear with more pieces, not merely a rank-limited 7 x 7 filter.

**Geometric cost of padding.** Zero padding makes the border pixels see fabricated black, so a filter near the edge responds to a partly synthetic signal. If objects are centred in training crops and appear at the border in deployment, the edge statistics are out of distribution. The geometric fix is either to train with border-touching objects or to pad by reflection, and the check is a per-position accuracy map over the image.

**Stride as a sampling lattice.** A stride-2 convolution does not read every other pixel and skip the rest; it reads overlapping windows centred on every other pixel, so a 3 x 3 stride-2 filter still touches all pixels within a 2-pixel neighbourhood. The distinction matters when reasoning about aliasing: a stride-2 max-pool genuinely discards three of four positions, whereas a stride-2 convolution with a 3 x 3 kernel and shift-invariant content is close to a low-pass followed by decimation. When the content has structure at the sampling frequency — a fine mesh, a printed dot pattern, a woven fabric — decimation aliases it into a lower-frequency phantom, and the model learns the phantom. The remedy is anti-aliasing before the stride, either by blurring or by using average pooling at that transition.

```text
sampling lattice, stride 2, 3x3 kernel:
positions at 0,2,4,... read windows {0,1,2},{2,3,4},{4,5,6},...
=> every input pixel is read at least once, but pixel 1 appears only in
   window 0 and window 1, while pixel 0 appears only in window 0
=> borders of the window overlap unevenly, so the response amplitude
   depends on phase; this is the mechanism behind checkerboard artefacts
   in dense prediction
```

**Channel dimension as a coordinate, not a count.** For a 64-channel stage, each spatial position is a point in a 64-dimensional space. Two positions that are similar in the image end up near each other after training, and the head's linear layer is a set of hyperplanes through that space. Depthwise convolution operates within each coordinate separately; a 1 x 1 convolution mixes coordinates at a single position; a dense 3 x 3 mixes coordinates across a neighbourhood. This is the cleanest way to see what depthwise separation gives up: it can no longer learn that “a red edge adjacent to a blue edge” is a single feature, because red and blue are different coordinates and only the pointwise layer can combine them, and only at the same position.

**Attention as a learned adjacency matrix.** Draw an N x N grid where cell (i,j) holds the attention weight from position i to position j. A convolution's matrix is banded: nonzero only near the diagonal, with the band width equal to the kernel area, and identical for every image. An attention layer's matrix is dense and image-specific. Consequences follow directly. Storage is N² per head instead of k². Sharing across positions is lost, so the model must learn from data what a convolution assumes. And the matrix is interpretable: visualising a row shows which regions a query attends to, which is a genuinely useful debugging tool for localisation failures — but only for the layer inspected, and only for one head at a time.

**Multi-scale object geometry.** An object of 40 pixels and an object of 400 pixels in the same 224 image need different feature resolutions. At the stage where the grid is 7 x 7, one cell covers 32 x 32 input pixels, so the 40-pixel object occupies about one cell and its shape is unrecoverable. This is a statement about information, not about model capacity: no head can recover shape that the grid no longer contains. The pyramid remedy fuses a coarse stage with a finer one, and the cost is dominated by the finer stage's area.

```text
object size vs stage grid (224 input, standard ladder)
stage   grid    input px per cell   40px object spans    400px object spans
 56     56x56        4                    10 cells            100 cells
 28     28x28        8                     5 cells             50 cells
 14     14x14       16                   2.5 cells             25 cells
  7      7x7        32                   1.25 cells            12 cells
```

**The cone in three dimensions.** For a 3D convolution on video or volumetric scans, the same recursion holds with kernel side along the third axis and a third jump component. The anisotropy is the practical problem: a 3 x 3 x 3 kernel advances one frame per tap while spatial strides are usually larger, so the temporal field outgrows the spatial field unless strides are chosen deliberately. The check is to compute the three field components separately and compare them with the object's aspect ratio in time and space.

**Residual geometry.** A residual block's output is x + F(x). If F is initialised near zero, the network starts as an identity map and moves away from it as training proceeds. The geometric picture is a set of shallow paths through the network, one per subset of skipped blocks, which is why the gradient can reach the stem through a path of length one. The failure boundary is a residual branch whose output scale is much larger than the identity path: the skip stops being a shortcut and becomes a rounding error in the sum, and the deep-stack advantage disappears.

**Takeaway:** Every architectural claim in this area can be drawn as a grid, a cone or an adjacency matrix; if it cannot be drawn, it is usually mis-stated.

## 7. Comparison / Selection Table

The rows below are adaptation policies for a fixed encoder, ordered by the amount of target evidence they require. The intended reading is a selection procedure: find the row whose evidence conditions you actually meet, not the row whose result you want.

| Policy | Trainable set | Target labels needed | Compute multiplier | Typical failure | Decision rule |
|---|---|---|---|---|---|
| Zero-shot transfer | none | 0 | 1x forward | Domain mismatch in normalisation | Use only to establish a floor and to test the input pipeline |
| Linear probe | head | 100–1,000 | 1x forward + head fit | Probe hides a threshold problem | Run first, always; it is the cheapest evidence |
| Probe plus BN re-estimation | head + BN stats | 0 labels, unlabelled target | 1x forward pass | Statistics drift back after further training | Apply before any labelled fine-tune when the domain shifts |
| Last-block fine-tune | final block + head | 500–5,000 | ~1.2x | Overfits with few labels | Default when labels are scarce but shift is real |
| Full fine-tune, low LR | all | 2,000+ | ~3x (forward + backward) | Catastrophic forgetting of early filters | Use when the target domain differs structurally from the source |
| Progressive unfreezing | all, layer groups in order | 2,000+ | ~3x plus schedule cost | Schedule tuned on one dataset does not transfer | Choose when you can afford 2–3 extra runs to compare |
| Distillation into a new student | student | teacher outputs, few labels | teacher forward once + student training | Student inherits teacher's confident errors | Choose when the deployment ceiling excludes the teacher |
| Train from scratch | all | 10,000+ per class typically | 3x and many epochs | Overfits; fails to converge in budget | Choose only when no in-domain pretrained checkpoint exists |

**Architecture rows for the same task.** These are not alternatives to the policies above; they compose with them.

| Encoder family | Params at comparable accuracy class | Resolution flexibility | Global mixing | Best evidence for choosing it |
|---|---|---|---|---|
| Plain convolutional ladder | moderate | high (fully convolutional) | no, only via depth | Local texture, small data, latency-critical |
| Residual convolutional | moderate | high | no | Default strong baseline for most vision tasks |
| Depthwise-separable mobile | low | high | no | Hard latency and power ceiling, with accuracy floor stated |
| Windowed attention | moderate | medium (needs grid alignment) | limited per layer | High resolution with dense prediction and enough data |
| Pure transformer | high | low (positional interpolation needed) | yes from layer one | Very large pretraining corpus, or a checkpoint you already trust |
| Convolutional stem plus transformer body | moderate to high | medium | yes after tokenisation | Mid-size data where both locality and long range matter |
| Distilled hybrid | low | high | inherited | Deployment ceiling below the model you actually validated |

**How to use the two tables together.** The policy table answers “how much of θ may move”; the architecture table answers “which θ is worth moving”. Combining a high-data architecture row with a zero-label policy row is a valid experiment, but its result is a statement about features, not about the architecture, and the report must say so.

**Anti-patterns in this selection.** First, comparing a linear probe at 224 against a fine-tuned model at 384 and attributing the difference to adaptation. Second, choosing a pure transformer because a published result on a large corpus was strong, while your corpus is three orders of magnitude smaller. Third, choosing a mobile architecture to hit a latency budget that was never measured on the target device, then discovering the budget was met by the compiler rather than by the architecture. Fourth, treating distillation as free accuracy, when it transfers the teacher's errors preferentially on the inputs where the teacher was most confident.

**A concrete selection walk (ILLUSTRATIVE).** Suppose 4,200 labelled images, 12 classes, two cameras, a 30 ms ceiling, and a pretrained residual checkpoint available under a permissive licence. Row “last-block fine-tune” matches the label count, and the residual row matches the evidence. The procedure: linear probe to establish the floor; BN re-estimation with unlabelled images from the second camera; last-block fine-tune on 3,800 images with 400 held out by capture session; then a full fine-tune only if the last-block model's per-class recall is still below target on the second camera. Each step has a stated trigger for proceeding, which is what makes the procedure reviewable.

**Reading the policy table as a cost ordering.** The compute multiplier column is the least important. The label column is the binding constraint: a policy row whose label requirement you do not meet is not available, regardless of its multiplier. The failure column tells you what the second run will cost if the first is mis-specified. A team that picks by expected accuracy rather than by label availability usually ends up running the expensive policy on the small set, overfitting, and then running the cheap policy anyway with three weeks gone.

**When the architecture question is genuinely open.** It is open when the label budget supports both candidates, the latency budget is met by both, and the shift profile is similar for both. That is a narrower condition than it sounds: on most real projects the label budget or the latency ceiling already excludes one side. State which constraint excluded which candidate, because that sentence is the audit trail for the decision and it survives personnel changes better than a benchmark table.

**A tie-break rule for near-equal cost.** When two candidates are within the run-to-run variance on the gate metric, choose on the axis that is cheapest to change later. Preprocessing and threshold are days; the adaptation policy is weeks; the architecture is a quarter. This ordering follows from the recovery costs in §9, and it means the near-equal case should almost always be resolved in favour of the checkpoint with the simpler preprocessing and the clearer licence.

| Constraint that binds | Excluded candidates | Remaining policy |
|---|---|---|
| Fewer than 1,000 labels | Full fine-tune, from scratch | Probe, then last-block fine-tune |
| Latency under 5 ms at batch 1 | Dense-at-high-resolution models, pure attention at high N | Separable or grouped model at reduced resolution |
| New camera with strong colour shift | Zero-shot, probe without statistic repair | BN re-estimation, then small labelled fine-tune |
| Labelling forbidden by policy | Any recipe needing new labels | Probe plus prior correction on existing scores |
| Fixed inference hardware without int8 units | Mobile architectures whose saving is depthwise | Dense model at moderate width with fused blocks |

**Why the table has a “policy plus architecture” structure rather than a single ranking.** A single ranking would have to assume a fixed label count, a fixed device and a fixed shift, which is exactly the information a practitioner lacks at the start. The two-table structure lets a reader enter at the constraint that binds and leave with a shortlist, which is the usable form of the answer.

**A worked contrast between two rows that look similar.** Consider “last-block fine-tune” and “progressive unfreezing” on the same 4,200-image set. Both eventually move most of the network, so their parameter counts at the final epoch are similar. They differ in the order of exposure: last-block fine-tune commits to a small hypothesis space and expands only if the gate fails, while progressive unfreezing expands on a schedule regardless of the gate. With 4,200 labels the first is cheaper because it can stop early and still produce a defensible model; with 42,000 the second is cheaper because the schedule is amortised over a run that will succeed either way. The deciding variable is label volume, not architecture.

**What the table deliberately does not rank.** It does not rank the policies by final accuracy, because no such ordering is stable across datasets. It does not rank the architecture families by accuracy either. What it ranks is the evidence each option requires before its result can be trusted, which is the only ordering that survives contact with a real deadline.



## 8. Failure Modes & Edge Cases

Each entry states the symptom as it appears in a report, the competing causes that produce that symptom, the check that discriminates between them, and the correction. The ordering is roughly by frequency in practice, not by severity.

**F-01 — Validation accuracy is high, deployment accuracy is low, same images.**
Competing causes: preprocessing mismatch between evaluation and serving; colour-space or bit-depth conversion applied only in one path; normalisation statistics taken from the wrong corpus; input resized with a different interpolation kernel.
Discriminating check: save the exact tensor fed to the model in each path for one image and compare hashes; if the hashes differ, the pipeline is the cause and no modelling work will fix it.
Correction: freeze the preprocessing as a versioned artefact with a test that asserts tensor equality across paths.
Prevention: a golden-image test in continuous integration that fails on any byte difference.

**F-02 — A new camera fails while the original cameras are fine.**
Competing causes: covariate shift in colour and exposure; label shift in defect prevalence; a genuinely different defect morphology on the new line; resolution or aspect-ratio change altering the effective object scale.
Discriminating check: compute a linear probe on new-camera labels only, and separately recompute the operating threshold on the new camera. If threshold recalibration alone recovers most of the loss, the cause is label shift; if not, it is covariate shift; if the probe is at chance, the morphology is new.
Correction: BN re-estimation from unlabelled new-camera images first, then threshold recalibration, then a small labelled fine-tune.
Prevention: include a per-camera slice in the promotion gate from the first release.

**F-03 — Training loss falls, validation loss rises after epoch 4.**
Competing causes: overfitting to a small training set; a leak between train and validation via capture session or patient; augmentation that is applied to validation inputs but not to validation targets, or vice versa.
Discriminating check: re-split by the group key and re-run; if the gap closes, the leak was the cause. If the gap persists, plot per-epoch train and validation loss on the same axes with the augmentation pipeline disabled on both.
Correction: group-aware split, stronger augmentation, or a smaller trainable set.
Prevention: the group key is chosen before any training run and recorded in the split manifest.

**F-04 — Fine-tuned model is worse than the frozen probe.**
Competing causes: learning rate too high for pretrained weights; BN in training mode corrupting running statistics on a small target set; head initialised with a large scale; too few epochs so the head has not converged while the backbone has already drifted.
Discriminating check: train with the backbone frozen and only the head moving for the first epoch, then unfreeze; if the model recovers, the cause is early drift. Separately, set BN to evaluation mode throughout and re-run; recovery implicates statistics.
Correction: warm-up segment, lower LR for early layers, BN in evaluation mode for small target sets.
Prevention: record the frozen set and the LR map in the run manifest; a run without them is not comparable.

**F-05 — Loss is NaN within the first hundred steps.**
Competing causes: LR too high with a saturating nonlinearity; division by a near-zero variance in a normalisation layer with a tiny batch; a corrupted label index outside the class range; mixed-precision overflow in an attention score before the softmax.
Discriminating check: run with gradient clipping at a large value and log per-layer gradient norms; if one layer spikes before the loss, that layer's scale is the cause. Separately, assert label range at load time; if the assertion fires, the cause is data.
Correction: gradient clipping, larger effective batch for normalisation, softmax computed in float32, label range assertions at ingest.
Prevention: range assertions and a single-batch overfit test at the start of every project.

**F-06 — Model is accurate on average and useless on the minority class.**
Competing causes: class imbalance with an unweighted loss; the minority class is visually a subset of a majority class; the operating threshold was chosen on the aggregate ROC.
Discriminating check: per-class recall at a fixed false-positive rate, plus a confusion matrix restricted to the minority class's neighbours. If the minority class is confused only with one neighbour, the representation does not separate them; if it is spread across all classes, the loss weighting is the cause.
Correction: per-class thresholds before any retraining; then loss reweighting or resampling; then targeted augmentation.
Prevention: the promotion gate reports per-class recall, never accuracy alone.

**F-07 — Latency budget is missed although the FLOP count is small.**
Competing causes: memory-bound layers at low resolution; depthwise convolutions with poor hardware utilisation; an unoptimised upsampling or reshape that forces a copy; batch size 1 removing all parallelism.
Discriminating check: profile per-layer time and compute arithmetic intensity (FLOPs per byte moved) for the slowest layer; a layer far below the machine's balance point is memory-bound, not compute-bound.
Correction: replace depthwise blocks with grouped or dense blocks at the same parameter budget, fuse operations, or raise batch size if the service allows it.
Prevention: benchmark the actual device at batch 1 before freezing the architecture.

**F-08 — Two runs of the same configuration differ by 3 points.**
Competing causes: nondeterministic kernels and atomics; different data order from a shuffling bug; different augmentation random seeds; different validation-time augmentation.
Discriminating check: fix all seeds and disable nondeterministic kernels; if the gap persists, diff the data order logs. If it disappears, the variance was kernel-level, and the original 3-point difference was inside the noise band.
Correction: report a variance estimate from at least three seeds; compare means with an interval, not single runs.
Prevention: a seeded reference run that is re-executed whenever the environment changes.

**F-09 — Accuracy is stable but the confidence scores are unusable for triage.**
Competing causes: overconfidence from a loss with no calibration term; temperature never tuned on held-out data; confidence computed after test-time augmentation with a different transform set; threshold inherited from a different class prior.
Discriminating check: expected calibration error on a held-out split, with bins chosen before looking at the curve. If ECE exceeds the operational tolerance, calibration is the cause regardless of accuracy.
Correction: fit a single temperature on held-out data, then recompute the threshold; never calibrate and threshold on the same split.
Prevention: calibration is part of the promotion gate and its bin edges are versioned.

**F-10 — The model degrades gradually over months with no code change.**
Competing causes: input drift from sensor ageing or lighting changes; label drift from a changed annotation guideline; data-pipeline drift from a library upgrade in resize or colour conversion.
Discriminating check: re-score a frozen golden set of images stored as raw bytes. If the scores change, the pipeline drifted; if they do not, the inputs drifted.
Correction: pin library versions for preprocessing, re-estimate BN statistics on recent unlabelled data, and re-validate the guideline.
Prevention: a golden set plus a monthly score-delta report on it.

**F-11 — Attention model overfits on 5,000 images while the convolutional baseline does not.**
Competing causes: too few tokens' worth of inductive bias; excessive positional parameters; augmentation too weak to compensate for the missing locality prior; LR schedule tuned for convolutions.
Discriminating check: add a convolutional stem and re-run with everything else fixed. If the gap closes, the cause was patch-level locality; if not, the cause is capacity relative to data.
Correction: convolutional stem, stronger augmentation, or a smaller width with the same depth.
Prevention: state the data volume alongside any architecture recommendation, and never recommend a pure attention model without it.

**F-12 — The frozen backbone turns out not to be frozen.**
Competing causes: normalisation layers left in training mode; a shared module referenced twice and unfrozen through one reference; an optimiser constructed before the freeze call.
Discriminating check: after the freeze call, count parameters with requires_grad true and compare with the intended set; then assert that the running statistics are byte-identical before and after one training step with a nonzero loss.
Correction: set the mode explicitly on every normalisation layer, construct the optimiser last, and log the trainable parameter count at step 0.
Prevention: a printed manifest of trainable names at run start, archived with the checkpoint.

**F-13 — Accuracy drops when the batch size changes between training and serving.**
Competing causes: normalisation statistics estimated over a small training batch; a head whose input scale depends on the pooled variance; a test-time aggregation that assumed a fixed batch composition.
Discriminating check: serve the same images at batch 1 and batch 32 with running statistics forced on both; if the outputs differ, statistics are being computed rather than read. Then re-read the training log for the physical batch size in the last epochs.
Correction: raise the effective batch, freeze statistics at the end of training, or replace batch normalisation with a group or layer variant that does not depend on batch composition.
Prevention: a serving-shape test in the release checklist that runs at batch 1.

**F-14 — A model is retrained with a new augmentation and its calibration silently degrades.**
Competing causes: the augmentation changes the score distribution's spread; the threshold was carried forward from the previous run; temperature fitted on the previous model was reused.
Discriminating check: compute ECE and the reliability diagram on the new model; if accuracy is unchanged while ECE doubled, the augmentation changed the confidence geometry without changing the decision boundary much.
Correction: refit temperature, then recompute the threshold on a disjoint split.
Prevention: calibration artefacts are versioned with the checkpoint, never inherited.

**F-15 — Feature cache is stale after the backbone changes.**
Competing causes: the cache key does not include the checkpoint hash; augmentation was applied at cache time and baked into the features; a preprocessing change was applied for caching but not for serving.
Discriminating check: recompute features for 100 sampled images with the current checkpoint and compare to the cache; any mismatch identifies the key as insufficient.
Correction: key the cache on checkpoint hash plus preprocessing version plus image content hash.
Prevention: cache keys are structured, not string-concatenated, and the key schema is reviewed.

**F-16 — A group split leaks through a derived key.**
Competing causes: the grouping key was built from a field that changes between sessions; near-duplicate frames from a burst were not deduplicated before splitting; a file-level rename broke the mapping from image to session.
Discriminating check: nearest-neighbour search in feature space between train and validation sets; a suspiciously high similarity for a large fraction of validation items indicates the split, not the model, produced the good number.
Correction: rebuild the split from the raw session identifier, deduplicate bursts first.
Prevention: a similarity audit is part of the split manifest's acceptance.

**F-17 — Depthwise model trains but its per-class recall is uneven in a way the dense baseline's is not.**
Competing causes: the pointwise layer must relearn cross-channel combinations from limited labels; the depthwise filter's spatial support is smaller than the dense kernel's at the same nominal k; the normalisation placement differs between the two blocks.
Discriminating check: compare the two models' confusion matrices restricted to the classes whose evidence is a cross-channel pattern; if the depthwise model confuses exactly those, the factorisation is the cause.
Correction: widen the pointwise layer, or replace the first and last depthwise blocks with dense blocks where cost permits.
Prevention: report per-class recall for every architecture compared, not just the aggregate.

**Edge case — kernels larger than the padded input.** If H + 2p < d(k-1) + 1, the numerator in the geometry formula is negative and no valid placement exists. Frameworks differ in whether they raise or silently produce an empty tensor; assert the geometry before the forward pass, because a silent empty tensor can propagate through a residual sum as a broadcast and produce plausible-looking but wrong shapes later.

**Edge case — a single-image batch with batch normalisation.** Batch statistics over one image per channel have variance estimated from one sample, so the normalised output is nearly constant. A model trained at batch 64 and served at batch 1 must use running statistics at inference, and a model that must both train and serve at batch 1 needs a different normalisation strategy.

**Edge case — aspect-ratio change.** A model trained on square crops and served non-square inputs either distorts the content or pads it; both change the effective object scale. Record the resize-and-pad policy in the tuple; it is one of the most common silent causes of a two-camera discrepancy.

**Edge case — checkpoint licence and vocabulary.** A checkpoint's label vocabulary and its preprocessing are part of the artefact. Reusing the weights but remapping the label order without remapping the head produces a model that is wrong in a way no accuracy metric at the class level will reveal until confusion is inspected by name.

**Edge case — a head trained on pooled features cannot localise.** Global average pooling discards position by construction, so a classifier built on it has no spatial output. If a downstream consumer needs a location, the model must expose a dense stage or a separate head; adding attention visualisation to a pooled model produces a plausible-looking map that is not the model's decision path.

**Edge case — resolution below the patch grid.** A patch-based model at an input smaller than the patch size produces fewer tokens than the positional table expects, and frameworks vary between interpolating and raising. A pipeline that resizes to a fixed side will silently mask this until a camera delivers a small crop.

**Edge case — an all-zero or all-constant input batch.** A batch of padded regions in a detection pipeline can be entirely constant, which makes the batch variance zero and the normalisation divide by zero. Clamping the variance hides the symptom; masking the padding before normalisation fixes it.

**Edge case — fine-tuning a model whose pretraining crop policy differs from yours.** A checkpoint pretrained on aggressive random crops sees objects at partial extent, so its features are tuned for partial views. Training on whole objects at full extent is therefore a distribution change for the encoder, not only for the head, and the first epochs can look worse than a frozen probe before recovering.

## 9. Complexity & Cost Analysis

**Four cost axes, not one.** Parameters determine storage and, weakly, capacity. MACs determine the arithmetic lower bound on latency. Activations determine training memory, which usually binds before weights. Data-collection cost determines whether the plan is feasible at all. A design review that cites only one of the four is incomplete by construction.

| Axis | Scaling with H, W, D, k | What it binds | Reduction lever |
|---|---|---|---|
| Parameters | k²·D_in·D_out, independent of area | Weight storage, model download size | Depthwise separation, channel bottlenecks, quantisation |
| MACs | area · k² · D_in·D_out | Arithmetic latency at high utilisation | Downsampling earlier, grouping, pruning |
| Activations | area · D per tensor per layer | Training memory, batch size | Gradient checkpointing, smaller batch, earlier downsampling |
| Tokens (attention) | N²·D per layer | Inference memory and latency at high N | Windowing, stem downsampling, token merging |

**Arithmetic intensity.** A layer's intensity is FLOPs divided by bytes moved. A 3 x 3 convolution at 56 x 56 x 64 reads 802,816 bytes of input and performs 115,605,504 MACs, so its intensity is about 2·115,605,504/802,816 ≈ 288 FLOP/byte — comfortably compute-bound on any modern accelerator, whose balance point is in the single or low double digits. A 1 x 1 convolution at the same shape reads the same bytes and performs 3,136·64·64 = 12,845,056 MACs, an intensity of about 32 FLOP/byte, which is closer to the balance point. A depthwise convolution at 64 channels performs 3,136·576 = 1,806,336 MACs over the same bytes, an intensity of about 4.5 FLOP/byte, which is memory-bound on every accelerator. That single number explains why mobile architectures with many depthwise layers underperform their FLOP count on real hardware.

**Training memory budget.** Weights plus gradients plus optimiser state dominate before activations. For 12.55M parameters: weights 47.9 MiB at float32, gradients another 47.9 MiB, and Adam state two more copies at 95.8 MiB, totalling about 191 MiB, which is small. Activations at batch 64 and 224 input dominate at roughly 200 MiB for stage 1 alone with a naive full-retention graph. Mixed precision halves activation bytes and can double the feasible batch, which is usually a better first move than shrinking the model.

**Annotation cost (ILLUSTRATIVE).** At 12,000 images, 1.5 minutes per image for a bounding label at a loaded rate of 22 currency units per hour, labelling costs 12,000·1.5/60·22 = 6,600 units. A 25% audit at 38 units per hour costs 12,000·0.25·1.0/60·38 ≈ 1,900 units if an audit item takes a minute, so the audit is 29% of the labelling bill. These are constructed rates; the transferable point is that audit cost is a first-class line item and is often the reason a larger label budget is not approved.

**Transfer compute cost.** A linear probe over 90,000 images with a 2,048-dimensional feature needs one forward pass, and the extracted feature table is 90,000·2,048·4 bytes = 703 MiB at float32 — cacheable but not free. Fitting a linear head on cached features solves a 2,048 x 12 least-squares problem, which is negligible. The full fine-tune alternative requires backward passes: with 4.1 GMAC forward and roughly three times that for forward-plus-backward, 90,000 images times 30 epochs costs 90,000·30·3·8.2e9 = 6.64e16 FLOP, which at 15 TFLOP/s effective is about 1.23 hours. That number decides the experiment design: probing is minutes, fine-tuning is hours, so probing comes first.

**Sweep cost.** A five-point learning-rate sweep with three seeds and early stopping at 30 epochs costs fifteen runs, i.e. roughly 18 hours at the above throughput. A sweep whose resolution is coarser than the run-to-run variance (F-08) is wasted money, so estimate the variance from three identical seeds before designing the sweep.

**Inference throughput.** At batch 1 on a stated 45 TFLOP/s effective device, the 1.968 GMAC model needs 2·1.968e9/45e12 = 87 µs of pure arithmetic. Real latency typically lands at 3–10x that for small models because kernel launch, memory traffic and framework overhead are not amortised. The practical rule: below about 2 GFLOP per image, measure before predicting, because the arithmetic no longer explains the runtime.

**Energy.** Energy per inference scales with bytes moved at least as strongly as with FLOPs, which means that for depthwise-heavy models the battery cost is driven by activation traffic rather than MAC count. On a fixed energy budget, dropping input resolution by 2x reduces activation bytes 4x and MACs 4x, and is almost always the largest available saving.

**Cost of a wrong architecture decision.** If an architecture is chosen that misses the latency ceiling, the recovery is a re-architecture plus re-validation, which costs the training compute again plus the evaluation time. If an architecture is chosen that overfits for lack of data, the recovery is data collection, which costs the labelling bill above. The asymmetry argues for establishing the cheapest falsifying experiment first: probe before fine-tune, batch-1 latency before the full benchmark suite.

**Quantisation as a cost lever with a contract.** Post-training quantisation of a 12.55M-parameter model takes weights from 50,212,912 bytes to 12,553,228 bytes, an exact 4x reduction. It does not automatically reduce latency by 4x: on hardware without int8 units the conversion costs time at load, and on hardware with them the gain depends on whether the slow layers are the ones that were quantised. The contract that must accompany it is a re-measured threshold and a re-measured per-slice recall, because quantisation changes the score distribution's tails first.

**Sparsity and pruning as cost levers.** Structured pruning removes whole channels, which reduces both parameters and MACs and therefore produces real latency gains; unstructured pruning produces sparse tensors that most accelerators do not accelerate. A pruning plan stated as “remove 40% of weights” without specifying structured versus unstructured is not a cost plan. The check is to measure latency on the target device after pruning, not to count zeros.

**Data-side cost that dominates.** At 12,000 images and a 25% audit, the audit is 29% of the labelling bill by the constructed rates in this section. Add the cost of re-auditing after a guideline change, which resets the audit on all affected images, and the data side can exceed the compute side entirely. This is why the label budget argument in §5 Step 6 is framed as a curve rather than a total.

**Marginal analysis of the whole plan.** Compare three plans on the same evidence: (a) fine-tune the pretrained residual model on all 12,000 images; (b) probe, then fine-tune the last block on 4,000; (c) train a small convolutional model from scratch on 12,000. Plan (a) costs the most compute and the most labelling, plan (b) costs about a third of the labelling if the audit is scoped to the probe set, and plan (c) avoids the licence question but pays full labelling. The decision criterion is which plan first produces a per-camera recall that clears the gate; any plan whose expected cost of the *first falsifying measurement* is high is the wrong plan to start.

**A second marginal comparison on the augmentation axis.** Augmentation is the cheapest capacity control available and the most frequently mis-set. A plan that adds a 4.8M-parameter block to fix overfitting costs months of latency budget forever; a plan that adds a crop-and-flip policy costs one configuration line and no inference cost. The ordering is therefore: augmentation first, then capacity reduction, then architecture change. Only the third changes the geometry sheet, and any change to the geometry sheet invalidates every comparison made before it.

**Throughput headroom as a safety margin.** A model that uses 60% of the latency budget at nominal load has no headroom for a busier host, a thermal-throttled device, or a larger input from a new camera at higher resolution. A 2x resolution change quadruples MACs and activation bytes, so a model at 60% of budget at 224 becomes infeasible at 448. Record the resolution headroom explicitly: the maximum input side at which the budget is still met, which for a MAC-dominated model scales as the square root of the budget ratio.

```text
max_side = side_0 * sqrt(budget_ms / measured_ms)
measured 18 ms at side 224 into a 30 ms budget:
max_side = 224 * sqrt(30/18) = 224 * 1.291 = 289 px
=> a 256-input camera variant fits; a 384 variant does not
```

**Takeaway:** Cost arguments must name which of the four axes they are optimising and show the measurement that would falsify the claim.

## 10. Key Theorems / Results Reference

This section records results that a practitioner is entitled to rely on, with the precise scope of each and the way it is commonly overstated. No numerical claim about a dataset's accuracy appears here.

**R1 — Translation equivariance of convolution.** A convolution commutes with translation up to boundary effects: shifting the input shifts the output and leaves interior values unchanged. Scope: exact for circular padding, approximate for zero padding near borders. Overstatement: “CNNs are translation invariant.” They are equivariant, and the pooling-plus-head stack is what produces approximate invariance. Practical consequence: a model can be equivariant and still fail on shifted inputs, because the head's global pooling is not the only asymmetry in a real pipeline.

**R2 — Universal approximation for convolutional architectures.** A sufficiently wide convolutional network can approximate any continuous function on a bounded domain to arbitrary accuracy. Scope: existence only; says nothing about the sample complexity, the optimisation path, or the depth required. Overstatement: “a big enough CNN can learn anything, so architecture does not matter.” The theorem is the reason architecture arguments must be made on data efficiency and inductive bias, not on expressiveness.

**R3 — Receptive-field composition (derived in §4.1).** The field of a chain is the recursion r_l = r_{l-1} + (k_l-1)j_{l-1}. Scope: single-path chains; branches require per-path computation and a max. Overstatement: the nominal field is the used field; gradient-based measurements show a much narrower effective field. Practical consequence: localisation accuracy depends on the effective field, so architectural depth alone does not buy context.

**R4 — Batch normalisation and internal covariate shift.** Normalisation stabilises the distribution of layer inputs and permits higher learning rates; the original motivation attributed this to reducing internal covariate shift, and later analyses argued the mechanism is better explained by smoothing the optimisation landscape. Scope: the empirical stabilisation is robust; the mechanism is contested. Overstatement: “BN makes training batch-size independent.” It does the opposite: with very small batches, the statistics are noisy and behaviour degrades.

**R5 — Batch-norm inference depends on running statistics.** At inference the layer uses stored means and variances, so a checkpoint is only valid under the statistics it was trained with. Scope: exact. Overstatement: “freezing the weights freezes the model.” Changing statistics changes outputs with no weight change, which is why §4.3 variant D exists.

**R6 — Residual connections ease optimisation of deep stacks.** Identity shortcuts give gradients a short path to early layers and make the residual branch's default behaviour close to identity, so depth can be increased without the degradation that afflicts plain stacks. Scope: optimisation behaviour, not capacity. Overstatement: “residuals prevent overfitting.” They do not; they change what is easy to optimise.

**R7 — Depthwise-separable factorisation reduces cost by roughly k².** Derived in §4.2: with equal input and output channels the ratio is about (k² + C)/(k²C). Scope: valid for equal channels; degrades under wide expansion. Overstatement: “separable convolutions are strictly better.” They reduce the mixing capacity per parameter and are memory-bound in practice.

**R8 — Scaled dot product and softmax saturation.** Dividing scores by √d_k keeps the score variance roughly constant as dimension grows, preventing the softmax from saturating into a near-one-hot distribution whose gradient vanishes. Scope: assumes approximately unit-variance, independent components. Overstatement: “the scale factor is a minor detail.” With unnormalised inputs it is the difference between learning and not learning.

**R9 — Attention cost is quadratic in token count.** Derived in §4.2, with the crossover at N = 2D against a 4x MLP. Scope: arithmetic only. Overstatement: “transformers are too expensive for images.” At 196 tokens and width 512 the attention term is a small fraction of the block cost, which is why patch-based models are tractable at moderate resolution.

**R10 — Pretrained features are linearly separable to a degree that grows with the pretraining distribution's overlap.** The empirical regularity behind linear probing: frozen features from a large pretraining corpus support a linear classifier that beats a from-scratch model on small target sets. Scope: empirical, dataset-dependent, and sensitive to the probe's regularisation. Overstatement: “probing always beats fine-tuning.” It does not when the target distribution differs structurally from the pretraining distribution.

**R11 — Fine-tuning order matters more than label count near the low-data regime.** When labels are scarce, adapting the last block plus head tends to beat adapting everything, because the early features are both more transferable and more easily destroyed. Scope: empirical, and sensitive to the learning-rate map. Overstatement: “always freeze early layers.” With abundant in-domain labels, full fine-tuning catches up and can exceed it.

**R12 — Scaling behaviour with data and parameters.** Loss decreases approximately as a power law in parameters and in data over wide ranges, with the exponent depending on the distribution. Scope: empirical regularity measured on large corpora, not a law. Overstatement: “more parameters always help.” The regularity holds along a compute-optimal frontier; off it, a model can be parameter-rich and data-poor.

**R13 — Equivalence of a stack of small filters to one large filter.** A composition of linear convolutions without intervening nonlinearities is itself a convolution whose kernel is the full convolution of the component kernels. Scope: exact for linear layers with matched strides; destroyed by any nonlinearity between them. Overstatement: “three 3 x 3 layers equal one 7 x 7 layer, so use the stack to save parameters.” The parameter saving is real, but the equivalence is false in the presence of the nonlinearities that make the stack worth using; the stack is a different function class, not a cheaper encoding of the same one.

**R14 — Variance of batch statistics scales inversely with batch size.** A mean estimated from B samples per channel has standard error σ/√B, so at B = 32 and σ = 1 the error is 0.177, and at B = 8 it is 0.354. Scope: independent samples within a batch; strongly correlated samples from a video burst have an effective B far smaller than the nominal one. Overstatement: “batch normalisation makes the result batch-size independent.” The estimate's noise is exactly what makes small-batch training fragile.

```text
standard error of a batch mean at sigma = 1
B = 128 -> 1/sqrt(128) = 0.088
B = 32  -> 0.177
B = 8   -> 0.354
B = 2   -> 0.707
```

The table is the arithmetic behind F-05 and F-13: at B = 2, the normalised activation is dominated by estimation noise, and the model can learn a function of the batch composition rather than of the input.

**R15 — Effective receptive field is smaller than and differently shaped from the nominal field.** Gradient-based measurement of the effective field shows a Gaussian-like profile concentrated near the centre, with the tails contributing little. Scope: empirical, architecture-dependent, and sensitive to initialisation and training length. Overstatement: “the model sees the whole 19-pixel field.” A consequence a practitioner can use: a model can be locally accurate and globally blind, so the occlusion sweep is the honest measurement, and a nominal-field calculation can only set an upper bound.

**R16 — Transfer benefit decreases with target label volume.** Stated as the shape in §4.2 result 10 rather than as a constant. Scope: the direction is robust across reports; the exponent is not transferable between datasets. Overstatement: “pretraining always wins.” It wins on sample efficiency, which matters when labels are scarce and matters little when labels are abundant and in-domain.

**R17 — Softmax invariance to a constant shift, not to a scaling.** Adding the same constant to all logits leaves the softmax unchanged; multiplying them changes it. Scope: exact algebra. Consequence: a per-class prior correction is a shift and is therefore safe to apply post hoc, while temperature is a scaling and changes the decision boundary — which is why temperature must be fitted before the threshold is chosen, never after.

**R18 — Convolution is not invariant to permutation of channels.** Permuting the input channels and the corresponding filter taps leaves the function unchanged only if the permutation is applied consistently to every filter. Scope: exact but easily broken in practice by a channel-order change in the input pipeline — the BGR/RGB confusion that F-01 is built around.

**How to cite these in a review.** Cite the result and its scope together. “Convolution is translation-equivariant up to border effects, so the border slice must be evaluated separately” is a usable sentence. “CNNs are translation invariant” is not, and the difference is exactly the boundary effect that F-02 and the padding discussion in §6 describe.

**Which of these results a design document should quote.** Quote R3, R8, R9 and R10, because they set the geometry, the numerical stability, the cost and the transfer expectation respectively. Quote R16 in the form of a curve, not a sentence. Do not quote R2 or R12 as support for any decision; they are scope-setting statements whose misuse is the most common way an architecture argument becomes unfalsifiable.

**A scope note on the pretraining literature.** Results about pretraining scale are reported on corpora and label vocabularies that are usually not the reader's. A checkpoint's published accuracy is therefore an upper bound on what a practitioner can expect only if the target distribution overlaps; where it does not, the published number is a statement about a different problem. This is why the screening experiment in §4.2 uses the reader's own 300 held-out examples rather than a published table: the published table cannot know the shift.

## 11. Interview / Exam Questions

**Q1. An input of 227 x 227 passes through a 11 x 11 kernel with stride 4 and no padding. What is the output side, and what changes if padding is 2?**
Answer: (227 - 11)/4 + 1 = 54 + 1 = 55. With padding 2: (227 + 4 - 11)/4 + 1 = 220/4 + 1 = 55 + 1 = 56. The output moves by exactly one placement because the added padding is a whole multiple of the stride.
Why asked: tests whether the candidate uses the counting formula or a memorised halving rule.
Follow-up: with k = 11 and s = 2, what does p = 1 give and what does p = 2 give? The numerator is 227 + 2p - 11 = 216 + 2p. At p = 1 it is 218, so H' = 109 + 1 = 110; at p = 2 it is 220, so H' = 110 + 1 = 111. Two extra pixels of padding add only one placement, which is the general form of the “padding is not free” lesson.

**Q2. A stage takes 56 x 56 x 128 input and produces 28 x 28 x 256 output with a 3 x 3 convolution. Give stride, padding and parameter count.**
Answer: stride 2 is required for the halving. Numerator must satisfy 56 + 2p - 2 - 1 = 53 + 2p ≡ 0 mod 2 with floor to 28, so p = 1 works: (56 + 2 - 3)/2 + 1 = 27 + 1 = 28. Parameters: 9·128·256 + 256 = 294,912 + 256 = 295,168.
Why asked: joins geometry and capacity in one question; most candidates get one of the two.
Follow-up: how many MACs? 784·9·128·256 = 784·294,912 = 231,211,008.

**Q3. Your model uses 12.55M parameters and achieves 1,000 images/s on a device with 45 TFLOP/s peak. Is it compute-bound?**
Answer: forward FLOPs at 1.968 GMAC is 3.936 GFLOP per image, so 1,000 images/s requires 3.94 TFLOP/s, which is 8.7% of peak. At that utilisation the model is not compute-bound; the bottleneck is memory traffic or launch overhead, and the next diagnostic is a per-layer profile with arithmetic intensity.
Why asked: distinguishes candidates who can turn a throughput number into a utilisation figure.
Follow-up: what would change the answer? A larger batch would amortise launches and could raise utilisation; a smaller model would lower the arithmetic requirement and make memory bind harder.

**Q4. A 3 x 3 convolution at 64 channels costs 36,928 parameters. A separable block at the same channels costs 4,800. Where did the capacity go, and when would you refuse the swap?**
Answer: the dense kernel factorises a full 64 x 64 mixing matrix per tap; the separable form restricts the spatial mixing to per-channel filters and the channel mixing to 1 x 1. The lost capacity is cross-channel spatial correlation. Refuse the swap when the discriminative signal is a joint pattern across channels at a spatial offset, and the label budget is too small for the 1 x 1 layer to relearn it.
Why asked: tests mechanical versus conceptual understanding of factorisation.
Follow-up: how would you test that refusal? Train both at equal epochs and compare per-class recall; if the dense version wins only on classes whose evidence is a cross-channel texture, the hypothesis is supported.

**Q5. You have 800 labelled images, a pretrained residual checkpoint, and a new camera. Give the sequence of experiments and the trigger for each step.**
Answer: linear probe first, cost minutes, establishes the floor. Then BN re-estimation from unlabelled new-camera images; trigger for proceeding is that the probe's per-class recall on the new camera is below target. Then last-block fine-tune with 640 train and 160 held out by session; trigger for full fine-tune is that recall remains below target on the new camera while training accuracy is high. Each step changes one component of the tuple.
Why asked: this is the actual job; the answer reveals whether the candidate thinks in triggers.
Follow-up: what if the probe is at chance? Then the representation does not transfer, and the choice is between collecting labels for the new morphology and abandoning the checkpoint.

**Q6. Compute the receptive field of a 5-layer stack where layers 1 and 3 have stride 2 and all have 3 x 3 kernels.**
Answer: l1: r = 1 + 2·1 = 3, j = 2. l2: r = 3 + 2·2 = 7, j = 2. l3: r = 7 + 2·2 = 11, j = 4. l4: r = 11 + 2·4 = 19, j = 4. l5: r = 19 + 2·4 = 27. Final field 27 pixels with a jump of 4.
Why asked: the recursion is easy to state and easy to get wrong once strides interleave.
Follow-up: how many pixels does the field sample densely? One in every four along each axis, so 7 x 7 of the 27 x 27 region are actually read at the deepest layer.

**Q7. Validation accuracy is 0.92 and deployment accuracy is 0.71 on the same images. Name the three most likely causes and the one check that separates them.**
Answer: preprocessing mismatch, normalisation statistics, and channel-order or bit-depth conversion. The separating check is to hash the exact input tensor at both paths for one image; identical bytes rule out the pipeline, different bytes localise it.
Why asked: this failure is common and cheap to fix, and candidates who jump to retraining waste weeks.
Follow-up: if the tensors are identical, what next? Compare the confidence distribution; a uniform shift suggests the model artefact differs, so compare checkpoint hashes next.

**Q8. Explain why a depthwise-separable model can have 8x fewer FLOPs and no latency improvement.**
Answer: depthwise layers have very low arithmetic intensity, roughly 4.5 FLOP/byte in the 64-channel 3 x 3 case, so they are memory-bound; on hardware whose balance point is around 10 FLOP/byte the layer waits on memory rather than on arithmetic. Fused or grouped alternatives raise intensity at the same parameter cost.
Why asked: separates roofline reasoning from FLOP-count reasoning.
Follow-up: what would you measure to confirm? Bytes moved per layer from a profiler, compared with the device's measured balance point.

**Q9. A frozen backbone produces different outputs after a training run in which no parameter was supposed to move. What happened?**
Answer: normalisation layers were in training mode, so running statistics were updated even though gradients were blocked. The check is to compare running-statistic tensors before and after one step; a nonzero difference confirms it. Fix by setting evaluation mode explicitly and by logging the trainable set at step 0.
Why asked: tests whether the candidate understands that freezing is about state, not just gradients.
Follow-up: when is updating those statistics desirable? When the target domain's colour or exposure differs systematically, in which case re-estimation on unlabelled target images is a cheap repair.

**Q10. You must choose between a 12.5M-parameter convolutional model at 1.97 GMAC and a 5.5M-parameter windowed attention model whose token count is 3,136 at width 96. Which question decides it?**
Answer: not the parameter or MAC totals, but whether the discriminative evidence is local texture or long-range layout, and whether the label budget supports the weaker locality prior. Compute the attention model's quadratic term per layer: 2·3,136·49·96 = 29.5 MMAC per windowed block, which is small, so cost does not decide it either. The deciding evidence is a paired experiment on the same split with the same augmentation and resolution.
Why asked: rejects cost-only arguments and demands a stated decision criterion.
Follow-up: what would make the attention model clearly right? Evidence that labels depend on relationships between distant regions, plus enough data for the weaker prior, plus a latency measurement showing windowed attention fits.

**Q11. Your report says “the model is 3 points better”. State the four tuple components that must be equal before that sentence means anything.**
Answer: at minimum the split key, the preprocessing and normalisation, the adaptation policy (frozen set plus LR map), and the evaluation protocol including threshold and test-time augmentation. Any difference in these makes the comparison a comparison of pipelines.
Why asked: this is the review question that catches most invalid claims.
Follow-up: which of them most often differs silently? Input resolution, because it changes both geometry and effective object scale and is often left out of a run manifest.

**Q12. Given 224 input, patch 16, width 768, compute the attention term as a fraction of one block's arithmetic at N = 197.**
Answer: linear projections 4·197·768² = 464,781,312; attention 2·197²·768 = 59,610,624; MLP 8·197·768² = 929,562,624; block total 1,453,954,560. The attention term is 4.1% of the block. So at this resolution, the quadratic term is not the cost driver; the dense projections are.
Why asked: forces the candidate to compute rather than repeat “attention is quadratic and therefore expensive”.
Follow-up: at what resolution does the attention term reach a third of the block? The fraction is N/(6D + N); setting it to 1/3 gives 3N = 6D + N, so N = 3D = 2,304 tokens, about 768 input at patch 16 where N = 2,305.

**Q13. A model trained on square crops is served non-square images. Predict the failure and the diagnostic.**
Answer: the pipeline either distorts aspect ratio or pads, and both change the effective object scale; the failure appears as a scale-dependent recall drop on elongated objects. Diagnostic: bin evaluation by object aspect ratio and by the padding fraction, and inspect the per-bin recall.
Why asked: aspect ratio is the forgotten component of the tuple.
Follow-up: what is the cheapest fix? Train with the deployment resize-and-pad policy applied to training crops, so the model sees the border statistics it will meet.

**Q14. Your linear probe beats from-scratch training by 8 points on 300 images, and by 0.5 points on 3,000. Interpret.**
Answer: the advantage of pretraining shrinks as target labels grow, which is the expected shape of the transfer benefit curve. The 0.5-point gap on 3,000 may be inside run-to-run variance, so the honest conclusion is that pretraining buys sample efficiency, not a permanent accuracy gain, and the decision should weigh the label-saving against checkpoint licence and preprocessing constraints.
Why asked: tests quantitative interpretation of a diminishing effect.
Follow-up: how would you decide with 3,000 labels? Estimate variance from three seeds; if the interval covers zero, choose on operational grounds such as licence and latency.

**Q15. What does an int8 quantisation of a 12.55M-parameter model save, and what can it break?**
Answer: weights fall from 50.2 MB to 12.6 MB, a 4x reduction, and arithmetic throughput typically improves on hardware with int8 units. It can break layers whose dynamic range is wide, especially the first and last, and it can shift calibration, so the operating threshold must be recomputed after quantisation.
Why asked: connects a storage fact to an operational consequence.
Follow-up: which layers would you leave in higher precision? The stem and the head, because they see the raw input range and produce the logits whose scale calibration depends on.

## 12. Quick Reference Card

| Item | Value or rule |
|---|---|
| Output side | H' = floor((H + 2p - d(k-1) - 1)/s) + 1 |
| Parameters | k²·C_in·C_out/g + C_out; independent of H and W |
| MACs | H'·W'·k²·C_in·C_out/g |
| Receptive field | r += (k-1)·j; then j *= s |
| Separable ratio | ≈ (k² + C)/(k²C), about 0.13 at k=3, C=64 |
| Attention crossover | N = 2D against a 4x MLP block |
| Balanced ladder | halve area, double width → constant per-stage MACs |
| Frozen ≠ static | Normalisation statistics move unless the mode is set |
| Cheapest first experiment | Linear probe on frozen features, then BN re-estimation |
| Promotion gate | Per-class recall at a fixed false-positive budget, plus calibration |
