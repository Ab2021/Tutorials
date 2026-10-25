# Architectures

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, 03, and 04
> Version 2026.3

## 1. Why this topic exists

Architecture is the part of an imaging project that looks most like machine learning and is
least often the reason a product succeeds or fails. A U-Net and a transformer can both segment a
lung nodule; the difference between a usable product and a failed one is usually the task
formulation, the label quality, the input geometry, and the evaluation, not the block diagram.

That said, architecture encodes assumptions that matter. A 2D convolutional network assumes that
one slice contains enough evidence. A 3D network assumes that through-plane context is available
and that the spacing is meaningful. A detection network assumes that a bounding box is a useful
representation of the finding. A segmentation network assumes that a voxel-level boundary is the
right output. A vision transformer assumes that global attention is affordable and that the
pretraining distribution matches the deployment distribution.

The industry problem is that architecture choices are often copied from a benchmark without
checking whether the benchmark's assumptions hold. A model that wins a challenge on 1 mm CT may
fail on 5 mm CT because the through-plane receptive field is wrong. A model that wins on a
lesion-centred patch set may fail on a full-volume worklist because the prevalence is different.
A segmentation model with a high Dice may be clinically useless because it misses small lesions
that contribute few voxels to the average.

This topic teaches how to choose an architecture from the task and the data, how to compute the
properties that determine whether it can see the finding, and how to evaluate it at the right
unit. It treats architecture as one component of an evidence package, not as the product.

### 1.1 The decisions this topic unblocks

| Decision | What architecture knowledge provides |
|---|---|
| Output representation | Classification, detection, segmentation, or ordinal score |
| Dimensionality | 2D, 2.5D, or 3D, given spacing and context |
| Receptive field | Whether the network can see the finding and its context |
| Pretraining | Whether ImageNet, radiology-specific, or self-supervised weights transfer |
| Loss | Cross-entropy, Dice, focal, or a combination |
| Imbalance handling | Sampling, weighting, or loss reweighting |
| Post-processing | Thresholds, connected components, non-maximum suppression |
| Compute budget | Parameters, FLOPs, and inference latency |

### 1.2 Architecture is not the claim

A regulated claim is about a clinical task, a population, and a workflow. "A 3D U-Net" is not a
claim. "A concurrent-read device that marks solid lung nodules of 4 mm or more on thin-section CT"
is a claim. The architecture is evidence that the claim can be met, not the claim itself.

### 1.3 What this topic does not argue

It does not argue that architecture is irrelevant. A model without enough receptive field or the
wrong output representation cannot do the task, no matter how good the data. It argues that
architecture is a necessary condition, not a sufficient one, and that the evidence for an
architecture is a controlled comparison at the clinical unit.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** a clinical task to an output representation and a loss function.
2. **Compute** the output shape, receptive field, and parameter count of a convolutional network
   from its layers.
3. **Choose** 2D, 2.5D, or 3D processing from the acquisition spacing and the required context.
4. **Select** a pretraining strategy and justify it by the similarity of the source and target
   distributions.
5. **Handle** class imbalance without distorting the operating point.
6. **Design** a baseline and an ablation that isolate the contribution of the architecture.
7. **Evaluate** a detection or segmentation model at the lesion level and the scan level, with
   the unit and reference standard stated.

## 3. The core mental model

The architecture is a function from an input tensor to an output tensor, trained by a loss that
defines what "correct" means. The task determines the output tensor, the output tensor determines
the loss, and the loss determines the metric. Architecture is downstream of all three.

```text
Clinical task
   │
   ▼
Output representation  ──►  Loss  ──►  Metric
   │                          │           │
   │                          │           └─ unit: patient, lesion, voxel
   │                          └─ what the model is rewarded for
   └─ classification, box, mask, score
   │
   ▼
Input geometry  ──►  Architecture  ──►  Post-processing
(spacing, size,        (CNN, U-Net,       (threshold, NMS,
 channels, context)     transformer)       connected components)
```

### 3.1 The receptive field is the binding constraint

A voxel in the output depends on a region of the input called its receptive field. If the
receptive field is smaller than the finding, the network cannot see the whole finding. If it is
much larger than the finding, the network may dilute the signal with irrelevant context.

For a stack of convolutions with kernel size `k` and stride `s`, the receptive field grows as:

```text
r_out = r_in + (k - 1) × j_in
j_out = j_in × s
```

where `j` is the jump, the distance between two consecutive output centres in input voxels. A
sequence of 3×3 convolutions grows the receptive field by 2 per layer; downsampling grows the
jump and therefore accelerates the growth.

A 3D U-Net with four downsampling stages and 3×3×3 convolutions has a bottleneck receptive field
of several tens of voxels, enough for a nodule and its margin. A shallow 2D network with three
3×3 convolutions has a receptive field of 7 pixels, which may be enough for a microcalcification
and not enough for a mass.

### 3.2 The task-to-output map

| Task | Output | Loss | Metric | Unit |
|---|---|---|---|---|
| Binary classification | Probability | Binary cross-entropy | AUROC, sensitivity, specificity | Patient or study |
| Multi-label classification | Vector of probabilities | Binary cross-entropy per label | Per-label AUROC, AUPRC | Study |
| Ordinal classification | Ordered logits | Ordinal or cross-entropy | Weighted kappa, MAE | Lesion |
| Detection | Boxes plus scores | Classification plus regression | Sensitivity at FP/scan, mAP | Lesion |
| Semantic segmentation | Voxel mask | Dice plus cross-entropy | Dice, IoU, Hausdorff | Voxel |
| Instance segmentation | Masks plus identities | Mask plus box loss | Dice per instance, detection | Lesion |
| Self-supervised pretraining | Embedding | Contrastive or masked reconstruction | Linear probe, fine-tune | Patch |

### 3.3 The bias-variance-compute triangle

Architecture choices trade three quantities:

- **Bias**: the error from assumptions that do not match the data. A 2D model has high bias for a
  3D finding; a 3D model has lower bias but needs more data.
- **Variance**: the error from sensitivity to the training sample. Large models have high variance
  on small datasets.
- **Compute**: the cost of training and inference. 3D convolutions and global attention are
  expensive.

The practical rule is to start with the simplest architecture that can represent the task, then
add capacity only when the learning curve shows underfitting. A small dataset with a large 3D
transformer is a variance problem, not a data problem.

### 3.4 Why 2D still wins in many products

2D models are cheaper, easier to pretrain, and easier to deploy. They work when the finding is
visible on a single slice and the through-plane context is not essential. They fail when the
finding is defined across slices, when the spacing is anisotropic but the model treats slices as
independent, and when adjacent slices leak between splits.

The intermediate option is 2.5D: a 2D network that takes a small stack of adjacent slices as
channels. It captures some through-plane context at a fraction of the cost of 3D. The number of
slices is a hyperparameter, and it must be justified by the through-plane extent of the finding.

### 3.5 Pretraining is a distribution assumption

A pretrained backbone helps when the source and target distributions share low-level features
(edges, textures) and the target dataset is small. It can hurt when the source domain is far from
the target (natural images vs CT) or when the pretraining data overlap the test data.

Radiology-specific pretraining (for example, models trained on chest radiographs or CT) is closer
to the target than ImageNet, but it can still be mismatched by modality, anatomy, and resolution.
Self-supervised pretraining on unlabeled target-domain images is the closest, but it requires the
unlabeled data and the compute. The choice must be validated, not assumed.

### 3.6 Skip connections and multi-scale features

An encoder-decoder network loses spatial detail as it downsamples. Skip connections copy
high-resolution features from the encoder to the decoder, restoring boundary information. This is
why U-Net works on small structures: the decoder has both semantic context from the bottleneck and
fine detail from the encoder.

A feature pyramid network does the same for detection, combining predictions at several scales so
that small and large lesions are handled by different levels. The choice of scales is a statement
about the lesion size distribution. A pyramid with levels at 8, 16, 32, and 64 voxels covers
objects from about 8 to 256 voxels; a lesion outside that range is handled poorly.

The practical implication is that architecture and label distribution must be matched. A dataset
with 3 mm nodules and 30 mm masses needs a multi-scale head or two models. A single-scale model
will be tuned to one end of the distribution and will fail at the other.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Convolution | Local weighted sum with shared weights | Calling any layer a convolution | Operation |
| Kernel size | Spatial extent of the filter | Confusing with receptive field | Voxels |
| Stride | Step between kernel positions | Confusing with downsampling factor | Voxels |
| Padding | Border handling | Assuming zero is neutral | Voxels |
| Receptive field | Input region affecting one output | Confusing with kernel size | Voxels or mm |
| Jump | Distance between output centres in input voxels | Ignoring in receptive field | Voxels |
| Feature map | Output of a convolution | Calling it an image | Tensor |
| Encoder | Downsampling pathway | Confusing with backbone | Network part |
| Decoder | Upsampling pathway | Confusing with head | Network part |
| Skip connection | Direct path from encoder to decoder | Assuming it preserves all detail | Connection |
| U-Net | Encoder-decoder with skip connections | Using as a synonym for segmentation | Architecture |
| FPN | Feature pyramid network | Confusing with U-Net | Architecture |
| Backbone | Feature extractor | Confusing with full model | Network part |
| Head | Task-specific output layers | Confusing with loss | Network part |
| Anchor | Reference box for detection | Assuming anchors fit all lesion sizes | Box |
| Region proposal | Candidate box | Confusing with final detection | Box |
| Non-maximum suppression | Remove overlapping boxes | Using the wrong IoU threshold | Post-processing |
| IoU | Intersection over union | Confusing with Dice | Ratio |
| mAP | Mean average precision | Using for imbalanced detection | Metric |
| FROC | Free-response ROC | Confusing with ROC | Metric |
| Sensitivity at FP/scan | Detection operating point | Omitting the FP rate | Metric |
| Dice loss | Overlap-based loss | Using alone on small lesions | Loss |
| Cross-entropy | Log loss | Using on severe imbalance without weights | Loss |
| Focal loss | Downweight easy examples | Using without tuning | Loss |
| Class imbalance | Unequal class frequencies | Ignoring in accuracy | Ratio |
| Patch sampling | Extracting sub-volumes | Sampling only lesions | Operation |
| 2.5D | 2D network with a slice stack | Calling it 3D | Design |
| 3D convolution | Convolution over volume | Assuming isotropic spacing | Operation |
| ViT | Vision transformer | Assuming it needs less data | Architecture |
| Patch embedding | Split image into tokens | Confusing with patching | Operation |
| Self-attention | Global token interaction | Ignoring quadratic cost | Operation |
| Self-supervised learning | Pretraining without labels | Assuming no bias | Method |
| Linear probe | Frozen features plus linear classifier | Confusing with fine-tuning | Evaluation |
| Fine-tuning | Updating pretrained weights | Assuming it always helps | Training |
| Parameter count | Number of trainable weights | Confusing with compute | Count |
| FLOPs | Floating-point operations | Confusing with latency | Operations |
| Latency | Time per inference | Confusing with throughput | ms |
| Ablation | Remove a component and compare | Changing several things at once | Experiment |

### 4.1 Architecture families by task

| Family | Output | Strength | Weakness | Typical use |
|---|---|---|---|---|
| CNN classifier | Probability vector | Simple, strong baseline | No localization | Triage, diagnosis |
| 2D U-Net | Mask | Cheap, mature | No through-plane context | Slice-wise segmentation |
| 3D U-Net | Volume mask | Context aware | Memory heavy | Nodule, organ segmentation |
| nnU-Net | Volume mask | Self-configuring | Less flexible | Segmentation baseline |
| Faster R-CNN | Boxes | Accurate localization | Two-stage latency | Lesion detection |
| RetinaNet | Boxes | Dense detection | Anchor tuning | Lesion detection |
| YOLO family | Boxes | Fast | Small-lesion recall | Real-time triage |
| Mask R-CNN | Masks and boxes | Instance segmentation | Compute | Lesion instances |
| ViT | Embedding or logits | Global context | Data hungry | Classification |
| Swin transformer | Hierarchical features | Multi-scale | Compute | Detection, segmentation |
| SAM/MedSAM | Promptable mask | General segmentation | Prompt required | Annotation assistance |

### 4.2 Loss functions and what they optimize

| Loss | Optimizes | Behaviour on imbalance | Suitable for |
|---|---|---|---|
| Binary cross-entropy | Calibrated log loss | Dominated by majority class | Balanced classification |
| Weighted cross-entropy | Weighted log loss | Tunable | Imbalanced classification |
| Focal loss | Hard-example focus | Robust | Dense detection |
| Dice loss | Overlap | Sensitive to small objects | Segmentation |
| Dice plus cross-entropy | Overlap and calibration | Common compromise | Segmentation |
| Tversky loss | Weighted FP/FN | Tunable | Small-lesion segmentation |
| Boundary loss | Boundary distance | Improves edges | Segmentation |
| Smooth L1 | Box regression | Outlier robust | Detection |
| Contrastive | Embedding similarity | Needs negatives | Self-supervised |

### 4.3 Metrics by output type

| Output | Primary metric | Secondary metric | Common error |
|---|---|---|---|
| Probability | AUROC | Calibration, sensitivity at fixed specificity | Reporting accuracy at 0.5 |
| Multi-label | Per-label AUROC and AUPRC | Calibration per label | Pooling labels |
| Ordinal | Weighted kappa, MAE | Adjacent-category accuracy | Treating as nominal |
| Boxes | Sensitivity at FP/scan | mAP, localization accuracy | Reporting mAP only |
| Masks | Per-lesion Dice | Hausdorff, surface distance | Pooling voxels |
| Instances | Per-instance Dice | Detection rate | Merging instances |
| Embedding | Linear probe accuracy | Retrieval, clustering | Claiming clinical use |

The primary metric must match the claim. A triage product's claim is about time to review and
missed critical findings, not about AUROC. A segmentation product's claim is about boundary and
volume, not about a single pooled Dice.

## 5. The mechanics, step by step

This procedure selects and validates an architecture for an imaging task. It assumes the data
passed the ingestion and label checks in Topics 02 and 04. Each step states input, output,
rationale, branches, and edge cases.

### Step 1. Fix the output representation and the unit

**Input:** The clinical task and the decision the output supports.

**Output:** One output representation (probability, box, mask, ordinal score) and one unit
(patient, study, lesion, voxel).

**Rationale:** The output representation determines the loss, the architecture head, and the
metric. A detection task cannot be trained with a study-level binary label without a weak-label
strategy, and a segmentation task cannot be evaluated with a study-level AUROC.

**Branch points:**
- Binary decision: probability and threshold.
- Multi-finding: multi-label vector.
- Localization: boxes or points.
- Boundary: mask.
- Severity: ordinal score.

**Edge case:** A task may need two outputs, for example a detection box and a malignancy score.
Those are two heads with two losses and two metrics; combining them into one number hides the
trade-off.

### Step 2. Choose the dimensional handling

**Input:** Spacing, slice thickness, lesion extent, and compute budget.

**Output:** 2D, 2.5D, or 3D processing, with a justification in physical units.

**Rationale:** The dimensionality must match the evidence. A 9 mm nodule on 1.25 mm slices spans
about 7 slices, so 3D or 2.5D can use the through-plane context. A 2 mm microcalcification on
0.1 mm mammography pixels is a 2D object; 3D processing adds cost without evidence.

**Branch points:**
- Isotropic or near-isotropic spacing and a 3D finding: 3D.
- Anisotropic spacing and a slice-visible finding: 2D or 2.5D.
- Very large volumes: patch-based 3D with a defined context.
- Whole-slide pathology: 2D tiles at a stated magnification.

**Edge case:** A 3D convolution on 5 mm slices with a 3×3×3 kernel covers 15 mm through-plane and
2 mm in-plane. That is not a cube; either resample or use anisotropic kernels.

### Step 3. Compute the required receptive field

**Input:** Lesion size, context requirement, and spacing.

**Output:** A minimum receptive field in millimetres and voxels.

**Rationale:** The network must see the finding and enough surrounding tissue to distinguish it
from look-alikes. A nodule detector needs the nodule plus vessel and pleural context; a
segmentation network needs the boundary plus a margin.

**Branch points:**
- Finding-only: receptive field about 2x the lesion diameter.
- Finding plus context: 3-5x the lesion diameter.
- Global context: whole volume or attention.

**Edge case:** A large receptive field can dilute a small finding. Use skip connections or a
multi-scale head to preserve local detail.

### Step 4. Estimate the output shape through the network

**Input:** Input shape, kernel, stride, padding, and pooling.

**Output:** Feature map shapes at each stage and a memory estimate.

**Rationale:** Shape arithmetic catches architecture errors before training. A mismatch between
the decoder output and the input shape is a common bug that produces a cryptic runtime error or,
worse, a silent crop.

**Branch points:**
- Same padding: shape preserved.
- Valid padding: shape shrinks by `k-1`.
- Strided convolution: shape divided by stride, rounded.
- Transposed convolution: shape multiplied by stride, with output padding.

**Edge case:** Odd input dimensions and even strides produce off-by-one shapes. Prefer input
dimensions divisible by the total stride, or use interpolation for the final resize.

### Step 5. Choose a backbone and pretraining

**Input:** Dataset size, modality, and available pretrained weights.

**Output:** A backbone, a pretraining source, and a fine-tuning policy.

**Rationale:** Pretraining transfers low-level features and reduces the data requirement. The
benefit depends on the similarity of the source and target and on the size of the target dataset.

**Branch points:**
- Small target dataset, similar modality: fine-tune a modality-specific backbone.
- Small target dataset, different modality: fine-tune a general backbone and validate.
- Large target dataset: train from scratch or use self-supervised pretraining.
- Very large model: consider parameter-efficient fine-tuning.

**Edge case:** Pretraining data may overlap the test set. Check the pretraining corpus against the
test identifiers; if overlap is possible, evaluate on a truly external set.

### Step 6. Design the head and the loss

**Input:** Output representation and class balance.

**Output:** Head layers, loss, and weighting.

**Rationale:** The head converts features to the output representation. The loss defines what the
model optimizes, which is not always the metric the clinician cares about. A Dice loss optimizes
overlap and can ignore small lesions; a cross-entropy loss optimizes calibration and can be
dominated by background.

**Branch points:**
- Classification: linear head, cross-entropy, class weights if imbalanced.
- Detection: classification plus box regression, focal or balanced loss.
- Segmentation: per-voxel logits, Dice plus cross-entropy.
- Ordinal: cumulative logits or regression with a monotonic constraint.

**Edge case:** A loss can be correct and the metric still poor. For example, Dice loss on a
dataset with many small lesions optimizes the large lesions because they contribute more voxels.
Use a lesion-level metric to detect the failure.

### Step 7. Handle imbalance without changing the task

**Input:** Class frequencies and the deployment prevalence.

**Output:** A sampling or weighting strategy and a statement of the effective prevalence.

**Rationale:** Imbalance is a property of the data, not the disease. Oversampling positives
changes the effective prevalence and therefore the calibration and PPV. The strategy must be
recorded and the outputs recalibrated if needed.

**Branch points:**
- Mild imbalance: class weights.
- Severe imbalance: foreground oversampling plus hard-negative mining.
- Detection: focal loss and anchor matching.
- Segmentation: patch sampling with a foreground ratio.

**Edge case:** A model trained on 50% positives and deployed at 1% prevalence will have a PPV far
below the training estimate. Recalibrate or reweight before reporting.

### Step 8. Set the training schedule and regularization

**Input:** Dataset size, model size, and compute.

**Output:** Optimizer, learning rate, schedule, weight decay, early stopping, and augmentation.

**Rationale:** Large models need more regularization and longer schedules. Small datasets need
stronger augmentation and earlier stopping. The schedule is part of the reproducibility record.

**Branch points:**
- Small dataset: strong augmentation, early stopping, lower capacity.
- Large dataset: longer schedule, weaker augmentation.
- Transfer learning: lower learning rate for pretrained layers.
- 3D: gradient checkpointing or mixed precision for memory.

**Edge case:** Early stopping on a validation metric that is not the deployment metric can select
the wrong checkpoint. Use the metric that matches the clinical operating point.

### Step 9. Define inference and post-processing

**Input:** Trained model and deployment constraints.

**Output:** Sliding-window or batch inference, threshold, connected components, NMS, and latency.

**Rationale:** A model that segments a patch must be applied to the whole volume without seams.
A detector that outputs overlapping boxes needs NMS. The post-processing is part of the product
and must be evaluated.

**Branch points:**
- Patch inference: overlap-tile with blending.
- Detection: NMS with an IoU threshold and a maximum per scan.
- Segmentation: threshold, largest component, hole filling.
- Triage: a single score per study and a threshold.

**Edge case:** Sliding-window seams can create false positives at patch borders. Use overlapping
tiles and discard border predictions, or use a Gaussian weighting.

### Step 10. Build a baseline and an ablation

**Input:** The proposed architecture and a simple baseline.

**Output:** A comparison table with the same data, splits, and metric.

**Rationale:** An architecture is only justified if it beats a simpler alternative on the task.
The baseline can be a 2D CNN, a threshold rule, or a clinical score. The ablation removes one
component at a time.

**Branch points:**
- Baseline wins: use the baseline; the complexity is not justified.
- Proposed wins: check the confidence interval and the subgroup results.
- Tie: prefer the simpler model for deployment and maintenance.

**Edge case:** A comparison that changes the preprocessing, the split, and the architecture at
once cannot attribute the gain. Change one factor at a time.

### Step 11. Evaluate at the clinical unit and operating point

**Input:** Model outputs and the reference standard.

**Output:** Metrics at the stated unit, with confidence intervals and subgroup results.

**Rationale:** A voxel Dice of 0.85 can coexist with a lesion sensitivity of 0.60 if the model
misses small lesions. The clinical unit is the lesion or the patient, and the operating point is
the threshold or FP rate the workflow can tolerate.

**Branch points:**
- Detection: sensitivity at 0.5, 1, 2, 4 FP/scan.
- Segmentation: Dice per lesion, not pooled voxels.
- Classification: AUROC, sensitivity at a fixed specificity, calibration.
- Triage: time to review, missed critical findings.

**Edge case:** Pooled Dice weights large lesions more than small ones. Report per-lesion Dice and
the size distribution.

### Step 12. Record compute and latency

**Input:** Model, hardware, and batch size.

**Output:** Parameter count, FLOPs, latency, throughput, and memory.

**Rationale:** A model that cannot run in the reading-room latency budget is not deployable. The
compute profile is part of the product specification and the regulatory submission.

**Branch points:**
- Cloud: throughput and network latency.
- On-premises: GPU memory and concurrent users.
- Edge: model size and power.

**Edge case:** Latency measured on a workstation is not latency in the PACS integration. Measure
end-to-end, including transfer and preprocessing.

### 5.1 Worked branch: how many slices for 2.5D?

A 2.5D model takes `k` adjacent slices as input channels. The choice of `k` determines the
through-plane context in physical units:

```text
context (mm) = k × slice spacing
```

For 1.25 mm slices and a 9.4 mm nodule, `k = 5` gives 6.25 mm of context, less than the nodule
itself. `k = 9` gives 11.25 mm, enough for the nodule plus a margin. `k = 15` gives 18.75 mm,
which may include vessels and pleura that help or confuse.

The memory cost scales linearly with `k`. If a 2D model at 96 × 96 uses 1 GB, a 2.5D model with
`k = 9` uses about 9 GB at the same batch size unless the first layer is modified. The common
implementation applies 2D convolutions to each slice and then a 1×1×k fusion, which keeps the
cost closer to 2D.

**Edge case:** At 5 mm spacing, `k = 9` covers 45 mm, far more than the nodule. The same `k` is
not transferable across protocols; the slice count must be derived from the spacing.

### 5.2 Sliding-window inference and seams

A 3D model trained on 96³ patches is applied to a 512 × 512 × 300 volume by sliding a window with
an overlap. A window of 96 with a stride of 48 gives 50% overlap in each axis. Predictions are
blended with a Gaussian weight that is high at the centre and low at the border.

Without overlap, the borders of each patch are predicted with less context and produce seams.
Without border weighting, the seams appear as discontinuities in the mask or as false-positive
blobs at patch corners. The diagnostic is a plot of the prediction along a line crossing a patch
boundary: a valid pipeline is smooth; a broken one has a step.

The cost of overlap is compute. A 50% overlap doubles the number of patches in each axis, so a
3D volume requires about eight times the forward passes of a non-overlapping grid. The overlap is
a product decision: it improves quality and reduces throughput.

### 5.3 Decision table: architecture by task and data

| Task | Data size | Recommended starting point | Alternative | Why |
|---|---|---|---|---|
| Binary classification | Small | 2D CNN with pretrained backbone | Linear probe | Cheap, strong baseline |
| Multi-label classification | Large | 2D CNN or ViT | Ensemble | Per-label calibration |
| Nodule detection | Medium | 3D patch detector or 2.5D | 2D detector plus slice aggregation | Through-plane context |
| Nodule segmentation | Small | nnU-Net | 3D U-Net | Self-configuring baseline |
| Organ segmentation | Medium | nnU-Net | Swin UNETR | Robust preprocessing |
| Haemorrhage triage | Large | 3D CNN or 2.5D | 2D slice ensemble | Volume context |
| Prostate lesion detection | Medium | 3D detector on multiparametric MRI | 2D per-sequence | Sequence fusion |
| Pathology classification | Large | ViT on tiles plus MIL | CNN plus MIL | Tile-level features |
| Whole-slide segmentation | Large | Tile U-Net plus aggregation | Transformer | Memory |

### 5.4 Edge cases that require a written rule

**Mixed slice thickness.** A 3D model trained on 1.25 mm and deployed on 5 mm has a different
through-plane receptive field in physical units. Resample or stratify.

**Missing series.** A multimodal model with a missing sequence needs a missingness indicator or a
separate model. Do not impute a sequence with zeros.

**Inconsistent spacing.** Compute the receptive field in millimetres, not voxels, and resample to
a common grid.

**Rescale absent.** A CT model trained on HU must not receive stored values. Assert the unit before
inference.

**Burned-in annotation.** A model can learn the annotation. Crop or mask it and test the impact.

**Laterality mismatch.** A model with a side-specific output needs a verified side. Do not let the
network infer laterality from a marker.

**Multi-frame objects.** A 3D model must receive a correctly stacked volume, not a frame index
order.

**Private tags carrying labels.** Remove them before training and before inference; an inference
service that reads a private tag is a leak path.

## 6. Worked example

### 6.1 The case

A 3D U-Net is trained to segment solid lung nodules on 1 mm isotropic CT patches of
96 × 96 × 96 voxels. The reference is an expert-edited mask. The model is compared with a 2D
U-Net baseline that sees one slice at a time. The clinical use is a concurrent-read alert, so the
operating point is sensitivity at 0.5 false positives per scan.

### 6.2 Shape arithmetic

The encoder uses 3×3×3 convolutions with stride 1 and padding 1, followed by 2×2×2 max pooling.
The decoder mirrors the encoder with transposed convolutions and skip connections.

| Stage | Operation | Output shape | Channels |
|---|---|---:|---:|
| Input | — | 96 × 96 × 96 | 1 |
| Encoder 1 | Conv 3×3×3, pad 1 | 96 × 96 × 96 | 32 |
| Pool 1 | Max 2×2×2 | 48 × 48 × 48 | 32 |
| Encoder 2 | Conv 3×3×3, pad 1 | 48 × 48 × 48 | 64 |
| Pool 2 | Max 2×2×2 | 24 × 24 × 24 | 64 |
| Encoder 3 | Conv 3×3×3, pad 1 | 24 × 24 × 24 | 128 |
| Pool 3 | Max 2×2×2 | 12 × 12 × 12 | 128 |
| Bottleneck | Conv 3×3×3, pad 1 | 12 × 12 × 12 | 256 |
| Decoder 3 | Upsample plus concat | 24 × 24 × 24 | 256 |
| Decoder 2 | Upsample plus concat | 48 × 48 × 48 | 128 |
| Decoder 1 | Upsample plus concat | 96 × 96 × 96 | 64 |
| Output | Conv 1×1×1 | 96 × 96 × 96 | 1 |

The output shape equals the input shape because every convolution uses same padding and every
pool is reversed by an upsample. A single valid convolution without padding would shrink the
output and break the skip connection.

### 6.3 Receptive field

Track the receptive field `r` and the jump `j`, starting at `r = 1`, `j = 1`:

```text
Conv 3×3×3:   r = 1 + (3-1)×1 = 3,   j = 1
Pool 2×2×2:   r = 3 + (2-1)×1 = 4,   j = 2
Conv 3×3×3:   r = 4 + (3-1)×2 = 8,   j = 2
Pool 2×2×2:   r = 8 + (2-1)×2 = 10,  j = 4
Conv 3×3×3:   r = 10 + (3-1)×4 = 18, j = 4
Pool 2×2×2:   r = 18 + (2-1)×4 = 22, j = 8
Conv 3×3×3:   r = 22 + (3-1)×8 = 38, j = 8
```

The bottleneck receptive field is 38 voxels, or 38 mm at 1 mm spacing. A 9.4 mm nodule occupies
about 25% of the receptive field, leaving context on all sides. At 5 mm spacing, the same network
has a receptive field of 190 mm, which is larger than the thorax in the through-plane direction
and mixes distant anatomy.

### 6.4 Parameter count for one layer

A 3×3×3 convolution with 32 input channels and 64 output channels has:

```text
weights = 32 × 64 × 3 × 3 × 3 = 55,296
biases  = 64
total   = 55,360 parameters
```

The full network in the table has about 1.9 million parameters, dominated by the bottleneck and
the decoder. That is small enough to train on a single GPU with mixed precision and large enough
to overfit a few hundred lesions without augmentation.

### 6.5 Compute per layer

The first convolution operates on 96³ = 884,736 output voxels. Each output voxel requires
55,360 multiply-accumulate operations:

```text
MACs = 884,736 × 55,360 = 4.90 × 10^10
FLOPs ≈ 2 × MACs = 9.8 × 10^10 = 98 GFLOP
```

That single layer costs about 98 GFLOP. A full forward and backward pass is several times larger.
The 2D baseline at 96 × 96 with the same channel widths costs about 1/96 of the compute per
slice, which is why 2D models remain attractive.

### 6.6 Segmentation and detection performance

The 3D model's mask on a representative lesion has 271 voxels, the reference has 303, and the
intersection is 236:

```text
Dice = 2 × 236 / (303 + 271)
     = 472 / 574
     = 0.822
```

On the detection task, 198 of 220 lesions are detected at 0.5 false positives per scan:

```text
sensitivity = 198 / 220 = 0.900
false positives = 0.5 × 220 = 110
```

The 2D baseline detects 185 of 220 at the same operating point:

```text
sensitivity = 185 / 220 = 0.841
```

and its Dice on the same lesion is 0.791. The 3D model is better on both, but the difference in
Dice (0.031) is within the reader-to-reference agreement of 0.777-0.82 measured in Topic 04. The
detection difference (0.059) is larger and is the one that matters for a concurrent-read alert.

### 6.7 Diagnostic 1: patch sampling changes the effective prevalence

The training patches are 80% lesion-centred and 20% random. The model therefore sees a lesion in
80% of patches during training. At inference on full scans, the lesion prevalence is 24%. With
sensitivity 0.90 and specificity 0.95, the PPV at 80% prevalence is:

```text
PPV = (0.90 × 0.80) / (0.90 × 0.80 + 0.05 × 0.20)
    = 0.72 / (0.72 + 0.01)
    = 0.986
```

At 24% prevalence:

```text
PPV = (0.90 × 0.24) / (0.90 × 0.24 + 0.05 × 0.76)
    = 0.216 / (0.216 + 0.038)
    = 0.850
```

The validation PPV of 0.986 is not a deployment estimate. The patch sampler changed the prior, and
the prior changes PPV. The diagnostic is the difference between the patch-level prevalence and the
scan-level prevalence; if they differ by more than a few percentage points, the reported PPV is
not transferable.

### 6.8 Diagnostic 2: adjacent-slice leakage in the 2D baseline

The 2D baseline splits slices randomly. Adjacent slices from the same patient are nearly
identical, so the test set contains near-duplicates of training slices. Test Dice is 0.91. After
a patient-level split, test Dice falls to 0.79. The 0.12 difference is larger than the difference
between the 2D and 3D models, so the comparison was contaminated, not informative.

The diagnostic is a nearest-neighbour check between test slices and training slices: compute a
perceptual or pixel hash and count test slices with a near-duplicate in training. A nonzero count
means the split is contaminated.

### 6.9 Varying one assumption

Assume the input patch is reduced from 96³ to 64³ to save memory. The bottleneck receptive field
remains 38 voxels, but the patch is only 64 voxels wide, so a 9.4 mm nodule near the patch edge
has less than 27 voxels of context on that side. The model's Dice falls from 0.822 to 0.801, and
detection sensitivity at 0.5 FP/scan falls from 0.900 to 0.872. The conclusion changes from "3D
segmentation meets the alert threshold" to "the smaller patch does not meet the same operating
point", and the memory saving is not free.

The same arithmetic applies to deployment: if inference uses 64³ patches but training used 96³,
the model sees a different context distribution. The patch size is part of the model
specification.

### 6.10 The 2.5D intermediate

A 2.5D model with nine input slices was trained on the same data. Its Dice is 0.812 and its
detection sensitivity at 0.5 FP/scan is 0.891. It sits between the 2D baseline (0.791, 0.841) and
the 3D model (0.822, 0.900), at about one-third of the 3D compute. The result is typical: most of
the through-plane benefit comes from a small stack, and the full 3D model adds a modest increment.

The practical conclusion is that 2.5D is a strong default when 3D memory is a constraint. The
choice should be made on the operating point and the latency budget, not on the label "3D".

## 7. Code, query, or calculation

The following code computes the shape table, receptive field, parameter count, FLOPs for one
layer, Dice, and PPV at two prevalences. It uses only the standard library.

```python
def conv_output(size, kernel=3, stride=1, padding=1):
    return (size + 2 * padding - kernel) // stride + 1

def pool_output(size, kernel=2, stride=2):
    return (size - kernel) // stride + 1

sizes = [96]
channels = [1, 32, 64, 128, 256]
for i in range(3):
    sizes.append(conv_output(sizes[-1]))
    sizes.append(pool_output(sizes[-1]))
sizes.append(conv_output(sizes[-1]))
print("shape trace        =", sizes)
print("output shape       =", sizes[-1])

# Receptive field and jump through encoder
r, j = 1, 1
trace = []
for kernel, stride in [(3, 1), (2, 2), (3, 1), (2, 2), (3, 1), (2, 2), (3, 1)]:
    r = r + (kernel - 1) * j
    j = j * stride
    trace.append((kernel, stride, r, j))
print("receptive field    =", r, "voxels")
print("jump               =", j)
for t in trace:
    print("  kernel {} stride {} -> r {} j {}".format(*t))

# Parameters in one 3x3x3 convolution
cin, cout, k = 32, 64, 3
params = cin * cout * k ** 3 + cout
print("layer parameters   =", params)

# FLOPs for that layer at 96^3 output
voxels = 96 ** 3
macs = voxels * params
print("layer MACs         = {:.3e}".format(macs))
print("layer FLOPs        = {:.3e}".format(2 * macs))

# Dice
ref, pred, inter = 303, 271, 236
dice = 2 * inter / (ref + pred)
print("Dice               = {:.3f}".format(dice))

# Detection sensitivity
print("3D sensitivity     = {:.3f}".format(198 / 220))
print("2D sensitivity     = {:.3f}".format(185 / 220))

# PPV at two prevalences
def ppv(prev, sens=0.90, spec=0.95):
    return (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))

print("PPV at 80%         = {:.3f}".format(ppv(0.80)))
print("PPV at 24%         = {:.3f}".format(ppv(0.24)))
```

Expected output:

```text
shape trace        = [96, 96, 48, 48, 24, 24, 12, 12]
output shape       = 12
receptive field    = 38 voxels
jump               = 8
  kernel 3 stride 1 -> r 3 j 1
  kernel 2 stride 2 -> r 4 j 2
  kernel 3 stride 1 -> r 8 j 2
  kernel 2 stride 2 -> r 10 j 4
  kernel 3 stride 1 -> r 18 j 4
  kernel 2 stride 2 -> r 22 j 8
  kernel 3 stride 1 -> r 38 j 8
layer parameters   = 55360
layer MACs         = 4.898e+10
layer FLOPs        = 9.796e+10
Dice               = 0.822
3D sensitivity     = 0.900
2D sensitivity     = 0.841
PPV at 80%         = 0.986
PPV at 24%         = 0.850
```

The shape trace is the encoder only: 96 → 96 → 48 → 48 → 24 → 24 → 12 → 12. The decoder reverses
it. The code assumes same padding, a 2×2×2 pool, and a single convolution per stage. A real
network has two convolutions per stage, which doubles the receptive field growth per stage; the
arithmetic is the same, but the numbers change.

## 8. Data, standards, and regulatory context

### 8.1 DICOM and input geometry

The model's input geometry is defined by the DICOM attributes described in Topic 02. A 3D
architecture assumes a stacked volume with a known affine. A 2D architecture assumes a slice order
and a slice thickness. The architecture must not be separated from the geometry contract: a 3D
model applied to a mis-stacked volume is a different function.

### 8.2 Reporting standards

The Checklist for Artificial Intelligence in Medical Imaging (CLAIM), published in Radiology:
Artificial Intelligence in 2020 and updated in 2024, defines the items a model report should
include: data sources, inclusion and exclusion criteria, reference standard, data partitions,
model architecture, training, evaluation metrics, and error analysis. The 2024 update replaced
"ground truth" with "reference standard" and discourages "validation" in favour of internal and
external testing.

A model description that names the architecture but omits the input geometry, the loss, and the
operating point does not meet the reporting standard.

### 8.3 FDA evidence

The FDA special controls at 21 CFR 892.2090 require a description of the algorithm's inputs and
outputs, pre-specified performance testing, standalone and reader-study evidence where applicable,
and subgroup analyses. The architecture is part of the software description, but the evidence is
about the device's output on a population, not about the block diagram.

A 3D U-Net that segments nodules is a software function. The claim might be "marks solid nodules
of 4 mm or more on thin-section CT for concurrent review". The evidence must show sensitivity and
false positives at the claimed operating point, on the claimed population, with the claimed input
geometry.

### 8.4 Compute and deployment standards

There is no standard for model latency, but there are workflow expectations. A concurrent-read
alert that arrives after the radiologist has finished the study is a second read, not a
concurrent read. The latency budget is a product requirement, and the architecture must meet it
on the deployment hardware.

### 8.5 What the standards do not cover

No standard says which architecture is appropriate for a task. The choice is an engineering
judgment supported by a baseline, an ablation, and an evaluation at the clinical unit. The
standards constrain the evidence, not the design.

### 8.6 Model cards and architecture documentation

A model card should record the architecture family, the input geometry, the output representation,
the training data, the reference standard, the operating point, the intended use, and the known
limitations. The architecture section should be specific enough that another team can reproduce
the input pipeline: patch size, spacing, channels, normalization, and augmentation.

A card that says "3D U-Net" without the patch size and spacing is not reproducible. A card that
says "trained on 10,000 CT scans" without the inclusion criteria is not auditable.

## 9. Industry practice and case evidence

### 9.1 U-Net and nnU-Net

U-Net (Ronneberger, Fischer, Brox, 2015) introduced the encoder-decoder with skip connections that
remains the default for biomedical segmentation. nnU-Net (Isensee et al., Nature Methods 2021)
showed that a self-configuring pipeline, including preprocessing, architecture, and training
rules, surpassed most specialized solutions across 23 public datasets. The practical lesson is
that a well-configured U-Net is a strong baseline and that architecture search often adds less
than data and preprocessing discipline.

### 9.2 Detection architectures in radiology

Detection in radiology is usually evaluated at a fixed false-positive rate per scan, not at
mAP. The Free-Response Receiver Operating Characteristic (FROC) curve and sensitivity at 0.5, 1,
2, or 4 FP/scan are the standard reporting forms. A model with a high mAP can still have
unacceptable sensitivity at the low FP rate a reading room tolerates.

The RSNA intracranial haemorrhage challenge (2019) used more than 25,000 head CT exams annotated
by more than 60 radiologists and required multi-label classification of five haemorrhage subtypes.
The challenge format shaped the architectures: 2.5D and 3D CNNs with slice aggregation dominated,
because the finding spans slices and the label is per study.

### 9.3 Foundation models and promptable segmentation

MedSAM (Ma et al., Nature Communications 2024) fine-tuned the Segment Anything Model on 1,570,263
image-mask pairs across 10 imaging modalities and 30+ cancer types. It reported performance on par
with or better than modality-specific nnU-Net and DeepLabV3+ on many tasks, with better
generalization to unseen targets, and reduced expert annotation time by about 82% in a tumour
annotation user study.

The lesson is not that promptable models replace task-specific models. The lesson is that a
foundation model can be a strong initialization and an annotation accelerator, and that its
zero-shot performance still needs modality-specific validation. A promptable model also changes
the product: it needs a prompt, and the prompt is part of the workflow.

### 9.4 Vision transformers in pathology

Pathology foundation models such as UNI (Chen et al., Nature Medicine 2024) and CONCH (Lu et al.,
Nature Medicine 2024) use vision transformers pretrained on large collections of whole-slide
image tiles. UNI was pretrained on more than 100 million patches from more than 100,000
haematoxylin and eosin slides; CONCH used 1.17 million image-caption pairs. These models are
strong feature extractors for tile-level tasks, but the tile size, magnification, and stain
distribution are part of the model's input contract.

### 9.5 Inference, not fact

The following are inferences: that a specific 3D architecture will beat a 2D baseline on a
specific dataset; that a specific foundation model will transfer to a specific modality; that a
specific patch size will preserve context. Each is plausible and each requires a local ablation.
The documented facts are the published benchmark results and the model specifications.

### 9.6 Compute is a clinical constraint

A model that takes 90 seconds per CT is compatible with a 5-minute read but not with a 30-second
triage window. The compute profile is part of the product, and it should be measured on the
deployment hardware with the deployment batch size. A 3D model that fits in a 24 GB GPU during
training may not fit in an 8 GB GPU at inference, and quantization or distillation may be needed.
Those changes can alter performance and must be re-evaluated.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| 2D model on a 3D finding | Sensitivity plateaus | No through-plane context | Missed lesions | 2.5D or 3D; slice stack |
| 3D model on anisotropic data | Blurred through-plane | Kernel covers 15 mm | False positives | Resample or anisotropic kernel |
| Receptive field too small | Boundary errors | Network cannot see context | Poor segmentation | Deeper encoder or dilation |
| Receptive field too large | Small lesions missed | Signal diluted | Low sensitivity | Skip connections, multi-scale |
| Shape mismatch | Runtime error or crop | Wrong padding or stride | Silent crop | Shape trace test |
| Loss metric mismatch | Dice high, sensitivity low | Dice weights large lesions | Clinical failure | Lesion-level metric |
| Patch prevalence mismatch | PPV inflated | Lesion-centred sampling | Misleading deployment | Reweight and recalibrate |
| Adjacent-slice leakage | Test Dice high | Random slice split | Inflated estimate | Patient-level split |
| Pretraining overlap | External test too good | Test data in pretraining | Invalid claim | Corpus audit |
| Class weights change calibration | PPV shifted | Effective prevalence changed | Miscalibrated output | Recalibrate |
| NMS threshold wrong | Merged or duplicate boxes | Overlap handling | Miscount lesions | Tune on validation |
| Sliding-window seams | Border false positives | Patch boundary artefacts | Alert fatigue | Overlap and discard borders |
| Model too large for latency | Slow inference | Compute exceeds budget | Not deployable | Distill or quantize |
| Overfitting small dataset | Train-test gap | Capacity vs data | Poor generalization | Regularization, augmentation |

### 10.1 Red flags that stop a model comparison

1. The baseline and the proposed model use different splits.
2. The metric is pooled Dice without per-lesion results.
3. The operating point is not stated.
4. The patch prevalence differs from the scan prevalence.
5. The pretraining corpus may contain the test set.
6. The 2D baseline splits slices randomly.
7. The architecture change is confounded with a preprocessing change.

### 10.2 A model comparison checklist

1. Same patients, same splits, same preprocessing.
2. Same input geometry and same patch size.
3. Same training budget or a stated difference.
4. Primary metric at the clinical unit and operating point.
5. Confidence intervals and subgroup results.
6. Compute and latency reported.
7. Ablation changes one factor at a time.
8. Failure cases reviewed, not only aggregate metrics.

A comparison that passes this checklist can support a claim. A comparison that fails it is a
demonstration, not evidence.

### 10.3 What a defensible architecture argument contains

A defensible argument states the task and the clinical unit, computes the required receptive field
in millimetres, compares 2D, 2.5D, and 3D on the same split, reports the primary metric at the
operating point, shows the confidence interval, and explains the compute trade-off. It does not
claim that a larger model is better without an ablation, and it does not compare its own result to
a published number from a different dataset.

The argument should also name what was not tested: the protocol that was absent, the lesion size
that was rare, and the site that was not held out. Those gaps are the first questions a regulator
or a customer will ask, and naming them is stronger than waiting to be caught. A team that can describe the boundary of
its evidence is easier to trust on the part it claims. That is the difference between a
benchmark result and a product claim, and it is the standard the rest of this module applies.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The output representation matches the task | Loss and metric valid | Box used for a boundary task | Clinical review |
| The spacing is uniform | 3D kernels meaningful | Mixed thickness | Delta histogram |
| The receptive field covers the finding | Model can see it | Small patch or shallow net | Compute RF in mm |
| The pretraining distribution is relevant | Transfer works | Different modality | Ablation with and without |
| The patch prevalence is representative | Calibration | Lesion-centred sampling | Compare to scan prevalence |
| The split is patient-level | Independent test | Multiple studies per patient | Identifier join |
| The post-processing is fixed | Reproducible output | Threshold tuned on test | Pre-specify |
| The compute budget is real | Deployable | Workstation-only latency | End-to-end measurement |
| The baseline is fair | Attribution | Different preprocessing | Same pipeline |
| The metric matches the claim | Valid evidence | Pooled voxel metric | Lesion-level analysis |

### 11.1 Limits of this topic

This topic covers the architecture choices that determine whether a model can represent an
imaging task. It does not cover training theory, optimization algorithms, or the full literature
on self-supervised learning. It gives the data scientist the arithmetic and the evaluation
discipline to choose and defend an architecture, not to invent one.

### 11.2 What would change the conclusion

If the deployment scanner used 5 mm slices, the 3D model's physical receptive field would grow to
190 mm and its advantage over 2D would likely disappear. If the lesion size distribution shifted
toward 3 mm nodules, the pooled Dice would rise while lesion sensitivity fell, and the model would
look better on the wrong metric. If the latency budget were 10 seconds per scan, the 3D model
would need distillation or a smaller patch, and the performance comparison would have to be
repeated.

## 12. Exercises

### Exercise 1. Shape arithmetic (calculation, 20 minutes)

**Deliverable:** For an input of 128³ with four stages of 3×3×3 conv (pad 1) and 2×2×2 pool,
compute the bottleneck shape.

**Acceptance criteria:** Correct trace (128, 64, 32, 16, 8), correct bottleneck shape, units
stated.

**Time:** 20 minutes.

### Exercise 2. Receptive field (calculation, 25 minutes)

**Deliverable:** Compute the receptive field for three 3×3 convolutions followed by a 2×2 pool
and two more 3×3 convolutions.

**Acceptance criteria:** Correct value (16 voxels), jump stated, formula shown.

**Time:** 25 minutes.

### Exercise 3. Parameter count (calculation, 20 minutes)

**Deliverable:** Compute the parameters in a 3×3×3 convolution with 64 input and 128 output
channels, including bias.

**Acceptance criteria:** Correct value (221,312), arithmetic shown.

**Time:** 20 minutes.

### Exercise 4. Architecture selection (application, 40 minutes)

**Deliverable:** A one-page recommendation for a 6 mm nodule detector, choosing 2D, 2.5D, or 3D
and justifying with spacing, lesion extent, and compute.

**Acceptance criteria:** Physical units; receptive field; baseline; operating point.

**Time:** 40 minutes.

### Exercise 5. Loss critique (critique, 35 minutes)

**Deliverable:** A 500-word critique of using Dice loss alone for a dataset with many small
lesions.

**Acceptance criteria:** Explains voxel weighting, proposes a lesion-level metric, and suggests a
combined loss.

**Time:** 35 minutes.

### Exercise 6. Prevalence audit (application, 30 minutes)

**Deliverable:** A calculation of PPV at training patch prevalence and deployment prevalence for
a stated sensitivity and specificity.

**Acceptance criteria:** Correct arithmetic; states the direction of the bias; proposes
recalibration.

**Time:** 30 minutes.

### Exercise 7. Ablation design (design, 45 minutes)

**Deliverable:** An ablation plan that isolates the effect of 3D versus 2D while holding data,
preprocessing, and training constant.

**Acceptance criteria:** One factor changed; same splits; confidence intervals; compute reported.

**Time:** 45 minutes.

### Exercise 8. Deployment latency (application, 40 minutes)

**Deliverable:** A latency budget for a concurrent-read alert, including preprocessing, inference,
transfer, and display.

**Acceptance criteria:** End-to-end estimate; hardware stated; bottleneck identified; margin for
peak load.

**Time:** 40 minutes.

### Exercise 9. Patch overlap arithmetic (calculation, 25 minutes)

**Deliverable:** For a 512 × 512 × 300 volume, a 96³ patch, and a stride of 48, compute the number
of patches per axis and the total forward passes.

**Acceptance criteria:** Correct counts (10 × 10 × 6 = 600), overlap stated as 50%, compute
implication stated.

**Time:** 25 minutes.

### Exercise 10. Foundation-model validation plan (design, 45 minutes)

**Deliverable:** A plan to test whether a promptable segmentation model can replace a task-specific
model for one organ, including prompts, reference standard, and failure analysis.

**Acceptance criteria:** Prompt protocol defined; zero-shot and fine-tuned compared; per-lesion
Dice; site-held-out test; annotation-time saving measured.

**Time:** 45 minutes.

## 13. Self-check answers

**A1. Input 128³, four stages of conv 3×3×3 pad 1 and pool 2×2×2. What is the bottleneck shape?**
128 → 64 → 32 → 16 → 8. The bottleneck is 8 × 8 × 8. With 256 channels, that is 131,072
activations per sample.

**A2. Receptive field of three 3×3 convs, one 2×2 pool, two 3×3 convs?**
Start r=1, j=1. Three convs: r=7, j=1. Pool: r=8, j=2. Two convs: r=8+2×2=12, then 16. The
answer is 16 voxels, not 18; the 18 value in the exercise assumes a different order. The formula
is the authority.

**A3. Parameters in a 3×3×3 conv, 64 in, 128 out, with bias?**
64 × 128 × 27 = 221,184 weights plus 128 biases = 221,312.

**A4. Why does a 3D model on 5 mm slices have a different receptive field than on 1 mm?**
The receptive field is measured in voxels by the architecture, but the physical extent is voxels ×
spacing. At 5 mm, 38 voxels is 190 mm; at 1 mm, it is 38 mm. The same network sees a much larger
physical region.

**A5. Why can Dice be high while lesion sensitivity is low?**
Dice pools voxels. A large lesion contributes many voxels, so a model that segments large lesions
well can have a high pooled Dice while missing small lesions entirely. Report per-lesion Dice and
detection sensitivity.

**A6. What is the effect of lesion-centred patch sampling on PPV?**
It raises the effective prevalence. At the same sensitivity and specificity, PPV rises with
prevalence, so the validation PPV overstates deployment PPV.

**A7. Why is a patient-level split necessary for slice-based models?**
Adjacent slices are near-duplicates. A random slice split puts near-duplicates in training and
test, inflating performance. The split must keep all slices from a patient on one side.

**A8. What is the difference between mAP and sensitivity at FP/scan?**
mAP summarizes precision-recall across thresholds and is not tied to a clinical operating point.
Sensitivity at a fixed FP/scan states the detection rate at the alert burden the workflow can
tolerate. The latter is the clinical metric.

**A9. Design rubric for Exercise 4.** A strong recommendation computes the lesion extent in
slices, the required context in mm, the receptive field of each option, and the compute budget.
A weak recommendation says "3D is more advanced".

**A10. Design rubric for Exercise 7.** A strong ablation changes only the dimensionality, keeps
the split and preprocessing fixed, reports confidence intervals, and includes compute. A weak
ablation changes the architecture and the augmentation together.

## 14. Interview questions

**Q1. How do you choose between 2D and 3D?**
A strong answer uses spacing, lesion extent in slices, required context, and compute, and proposes
a 2.5D baseline. A weak answer says 3D is always better.

**Q2. What is a receptive field and why does it matter?**
A strong answer defines it, computes it, and compares it to the lesion and context in millimetres.
A weak answer confuses it with kernel size.

**Q3. Why is Dice not enough for segmentation?**
A strong answer explains voxel weighting, small-lesion misses, and boundary errors, and proposes
per-lesion metrics and Hausdorff distance. A weak answer says Dice is standard.

**Q4. How do you handle class imbalance?**
A strong answer describes weighting, sampling, and focal loss, and notes the effect on calibration
and PPV. A weak answer says "oversample the positives".

**Q5. What does pretraining buy you?**
A strong answer says transfer of low-level features and reduced data requirement, conditional on
domain similarity, and proposes an ablation. A weak answer says pretraining always improves
accuracy.

**Q6. How do you evaluate a detector?**
A strong answer uses sensitivity at a stated FP/scan, a lesion-level reference, and a matching
rule, with confidence intervals. A weak answer reports mAP only.

**Q7. What is the risk of a very large receptive field?**
A strong answer says dilution of small findings and mixing of distant anatomy, and proposes
multi-scale features or skip connections. A weak answer says bigger is better.

**Q8. How do you know your architecture is deployable?**
A strong answer measures end-to-end latency, memory, and throughput on the deployment hardware,
and compares to the workflow budget. A weak answer benchmarks on a workstation.

**Q9. What is the difference between a linear probe and fine-tuning?**
A strong answer says a linear probe freezes the encoder and trains a classifier, measuring feature
quality; fine-tuning updates the encoder and can adapt to the task but risks overfitting. A weak
answer uses the terms interchangeably.

**Q10. How do you attribute a performance gain to the architecture?**
A strong answer holds data, splits, preprocessing, and training constant, changes one factor, and
reports confidence intervals and compute. A weak answer compares two published numbers.

## 15. Key terms

- **2.5D** — 2D network with a stack of slices as channels.
- **3D convolution** — convolution over a volume.
- **Ablation** — removing a component to measure its contribution.
- **Anchor** — reference box in detection.
- **Backbone** — feature extractor.
- **Class imbalance** — unequal class frequencies.
- **CNN** — convolutional neural network.
- **Decoder** — upsampling pathway.
- **Dice loss** — overlap-based loss.
- **Encoder** — downsampling pathway.
- **FLOPs** — floating-point operations.
- **Focal loss** — downweights easy examples.
- **FROC** — free-response ROC.
- **IoU** — intersection over union.
- **Jump** — output-centre distance in input voxels.
- **Latency** — time per inference.
- **Linear probe** — frozen features plus linear classifier.
- **mAP** — mean average precision.
- **NMS** — non-maximum suppression.
- **nnU-Net** — self-configuring segmentation pipeline.
- **Parameter count** — trainable weights.
- **Patch sampling** — extracting sub-volumes.
- **Receptive field** — input region affecting one output.
- **Self-attention** — global token interaction.
- **Self-supervised learning** — pretraining without labels.
- **Skip connection** — direct encoder-to-decoder path.
- **Stride** — kernel step.
- **U-Net** — encoder-decoder with skip connections.
- **ViT** — vision transformer.

## 16. Sources

- Ronneberger, O., Fischer, P., Brox, T., "U-Net: Convolutional Networks for Biomedical Image
  Segmentation," MICCAI 2015. https://doi.org/10.1007/978-3-319-24574-4_28
- Isensee, F. et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation," Nature Methods 18, 203-211 (2021). https://doi.org/10.1038/s41592-020-01008-z
- Ma, J. et al., "Segment anything in medical images," Nature Communications 15, 654 (2024).
  https://doi.org/10.1038/s41467-024-44824-z
- Chen, R. J. et al., "Towards a general-purpose foundation model for computational pathology,"
  Nature Medicine 30, 850-862 (2024). https://doi.org/10.1038/s41591-024-02857-3
- Lu, M. Y. et al., "A visual-language foundation model for computational pathology," Nature
  Medicine 30, 863-874 (2024). https://doi.org/10.1038/s41591-024-02856-4
- Mongan, J., Moy, L., Kahn, C. E. Jr., "Checklist for Artificial Intelligence in Medical Imaging
  (CLAIM)," Radiology: Artificial Intelligence 2(2): e200029 (2020).
  https://doi.org/10.1148/ryai.2020200029
- Tejani, A. S. et al., "Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024
  Update," Radiology: Artificial Intelligence 6: e240300 (2024).
  https://doi.org/10.1148/ryai.240300
- U.S. Food and Drug Administration, 21 CFR 892.2090, final order 90 FR 24969, 13 June 2025.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892.2090
  (accessed 25 October 2026).
- RSNA Intracranial Haemorrhage Detection Challenge (2019).
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019
  (accessed 25 October 2026).
- RSNA Lumbar Spine Degenerative Classification AI Challenge (2024).
  https://www.rsna.org/artificial-intelligence/ai-image-challenge/lumbar-spine-degenerative-classification-ai-challenge
  (accessed 25 October 2026).
- MONAI, Transforms and network documentation. https://docs.monai.io/ (accessed 25 October 2026).

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_labels-and-annotation.md`](04_labels-and-annotation.md)
- Next topic: [`06_leakage-and-shortcuts.md`](06_leakage-and-shortcuts.md)
- Related topic: [`07_evaluation.md`](07_evaluation.md)
- Root survey: [`../../10_medical_imaging.md`](../../10_medical_imaging.md)
- Labs: [`../../labs/`](../../labs/)
- MONAI: https://monai.io/

> **Note:** Architecture choices are evaluated on the clinical unit, not on the benchmark.
