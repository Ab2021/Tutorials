# Foundation Models

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 03, 05, 06, and 07
> Version 2026.3

## 1. Why this topic exists

A foundation model is a model pretrained on a large, broad corpus and adapted to many downstream
tasks. In imaging, the promise is label efficiency: a model that has already learned visual
structure should need fewer annotated cases for a new task. The promise is real in some settings
and oversold in others.

The industry problem is that foundation models change the economics and the evaluation of an
imaging project without changing its fundamentals. The label is still the ceiling. The site is
still a confounder. The test set is still contaminated if the pretraining corpus contains it. The
regulatory claim is still about a specific intended use, not about general intelligence.

Foundation models also introduce new failure modes. A model pretrained on millions of images may
have seen the test set, or a near-duplicate of it. A model trained on one scanner population may
encode that population as a prior. A promptable model's output depends on the prompt, and the
prompt is part of the product. A vision-language model can generate a report that sounds correct
and is not grounded in the image.

This topic teaches how to evaluate and adapt a foundation model for an imaging task. It covers the
pretraining corpus, the adaptation strategies, the input contract, the label-efficiency curve, the
contamination audit, the calibration and subgroup analysis, and the regulatory treatment. It uses
the published foundation models in radiology and pathology as case evidence and separates their
reported results from the claims made about them.

### 1.1 The decisions this topic unblocks

| Decision | What foundation-model knowledge provides |
|---|---|
| Build or adapt | Whether pretraining is justified by data and compute |
| Adaptation strategy | Zero-shot, linear probe, fine-tune, or prompt |
| Label budget | How many labels are needed to reach the target |
| Input contract | Resolution, modality, stain, and prompt format |
| Contamination audit | Whether the test set was in pretraining |
| Baseline | Whether the foundation model beats a task-specific model |
| Regulatory path | Whether the model is locked or adaptive |
| Monitoring | What drift looks like for a pretrained encoder |

### 1.2 What "foundation" does and does not mean

It does not mean general-purpose clinical intelligence. It means a pretrained representation that
transfers to some tasks. The transfer is empirical: it must be measured on the target task, the
target population, and the target acquisition. A foundation model that is excellent on a benchmark
can be mediocre on a new site, and the benchmark result is not a substitute for the local test.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** the pretraining-adaptation-deployment lifecycle of an imaging foundation model.
2. **Choose** an adaptation strategy from the label budget and the task.
3. **Define** the input contract, including resolution, modality, stain, and prompt.
4. **Measure** label efficiency with a curve, not a single point.
5. **Audit** the pretraining corpus for test-set contamination and near-duplicates.
6. **Compare** a foundation model to a task-specific baseline on the same split.
7. **Evaluate** calibration and subgroup performance after adaptation.
8. **Map** an adaptive foundation model to a regulatory change-control strategy.

## 3. The core mental model

A foundation model is a representation learned from a pretraining distribution, adapted to a
target distribution, and evaluated on a test distribution. The three distributions are not the
same, and the gaps between them are the evaluation.

```text
Pretraining corpus          Adaptation                 Deployment
(unlabeled, broad)          (labels, task)             (target population)
       │                         │                          │
       ▼                         ▼                          ▼
  Representation  ────────►  Task model  ────────────►  Clinical output
       │                         │                          │
       └── contamination ────────┴── distribution shift ────┘
```

### 3.1 The three distributions

| Distribution | What it contains | Why it matters |
|---|---|---|
| Pretraining | Images, captions, tiles, modalities | Encodes priors and possible test contamination |
| Adaptation | Labeled target-task data | Determines the task head and the fine-tuning |
| Deployment | Real population, scanners, protocols | Determines whether the model works |

A foundation model's advantage comes from the first distribution; its risk comes from the gap
between the first and the third. The adaptation set is usually small, so it cannot correct a large
prior mismatch.

### 3.2 Adaptation strategies

| Strategy | What is trained | Label need | Risk |
|---|---|---|---|
| Zero-shot | Nothing | None | Prompt sensitivity, no calibration |
| Linear probe | A linear head | Small | Limited adaptation |
| Fine-tune | Encoder and head | Medium | Overfitting, forgetting |
| Parameter-efficient | Adapters or LoRA | Small to medium | Less forgetting, less adaptation |
| Prompt tuning | Prompt tokens | Small | Prompt design |
| Full pretraining | Everything | Very large | Cost, contamination |

The choice is a trade-off between label efficiency and adaptation capacity. A linear probe tests
the representation; a fine-tune adapts it. A zero-shot result tests the pretraining, not the
product.

### 3.3 Label efficiency is a curve

A foundation model is often sold on "10% of the labels". The claim is meaningful only as a curve:
performance at 1%, 5%, 10%, 25%, 50%, and 100% of the labels, compared with a task-specific model
at the same budgets. A single point at 10% can hide a curve that is worse at 100%.

The curve also shows the marginal value of labels. If the curve is flat after 10%, more labels do
not help; if it is steep, the foundation model has not transferred and the task-specific model may
be better.

### 3.4 Contamination is the first audit

A foundation model trained on public data may have seen the test set. The risk is highest for
models trained on the same public benchmarks that are used for evaluation. The audit checks for
exact duplicates, near-duplicates, and patient overlap between the pretraining corpus and the test
set.

Contamination inflates the zero-shot and fine-tuned results. It is hard to detect after the fact
because the pretraining corpus is often not released. The practical response is to evaluate on a
private, site-held-out set that could not have been in the pretraining corpus, and to ask the
vendor for the corpus provenance.

### 3.5 The prompt is part of the product

A promptable segmentation model takes a box, a point, or a mask as input. The output depends on
the prompt: a box around the lesion gives a different mask from a point inside it. The prompt
policy, the user interface, and the correction workflow are part of the device, not a detail.

A product that claims "segments the prostate" must specify who draws the prompt, how, and what
happens when the prompt is wrong. A zero-shot Dice measured with an expert box is not the same as
a Dice measured with a user-drawn box.

### 3.6 The pretraining-compute curve

Foundation models improve with data and compute up to a point, but the curve is not free. The
pretraining cost is amortized over many tasks; the adaptation cost is paid per task. A model that
costs 10 million USD to pretrain is only cheaper than a task-specific model if it serves enough
tasks or enough labels.

The economic test is the same as the statistical one: does the foundation model beat the
task-specific baseline at the label budget that the project can afford? If the answer is yes, the
pretraining cost is someone else's problem (the vendor's). If the answer is no, the foundation
model is an expensive baseline.

### 3.7 The benchmark is not the deployment

A foundation model's published benchmark is a curated dataset with a known reference standard and
a known split. The deployment is a hospital with a different scanner, a different protocol, and a
different case mix. The gap between the two is the evaluation the vendor did not do.

The practical rule is to treat the benchmark as a prior, not as evidence. It tells you the model
can learn something; it does not tell you it will work for you.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Foundation model | Pretrained model adapted to many tasks | Calling any pretrained CNN a foundation model | Model class |
| Pretraining | Learning a representation from broad data | Confusing with fine-tuning | Process |
| Self-supervised learning | Pretraining without labels | Assuming no bias | Method |
| Contrastive learning | Learning by pulling and pushing pairs | Confusing with supervised | Method |
| Masked autoencoder | Reconstruct masked patches | Confusing with denoising | Method |
| Vision transformer | Transformer over image patches | Assuming it needs less data | Architecture |
| Patch | Local image region token | Confusing with a tile | Region |
| Token | Embedded patch or word | Confusing with a pixel | Unit |
| Embedding | Learned feature vector | Treating as interpretable | Vector |
| Linear probe | Frozen encoder plus linear classifier | Confusing with fine-tuning | Evaluation |
| Fine-tuning | Updating pretrained weights | Assuming it always helps | Training |
| LoRA | Low-rank adaptation | Confusing with pruning | Method |
| Adapter | Small trained module | Confusing with a head | Method |
| Prompt | Input cue for a promptable model | Assuming one prompt fits all | Input |
| Zero-shot | No task-specific training | Ignoring prompt sensitivity | Evaluation |
| Few-shot | A few labeled examples | Confusing with zero-shot | Evaluation |
| Label efficiency | Performance per label budget | Reporting one budget | Curve |
| Contamination | Test data in pretraining | Confusing with leakage | Audit finding |
| Near-duplicate | Almost identical image | Ignoring | Audit finding |
| Benchmark | Standard evaluation dataset | Assuming clinical validity | Dataset |
| Distribution shift | Difference between source and target | Confusing with drift | Phenomenon |
| Domain adaptation | Adapting to a new domain | Confusing with fine-tuning | Method |
| Multi-instance learning | Learning from bags of tiles | Confusing with segmentation | Method |
| Vision-language model | Model linking images and text | Assuming grounding | Model class |
| Caption | Text paired with an image | Assuming accuracy | Text |
| Grounding | Linking text to image regions | Confusing with fluency | Property |
| Calibration | Agreement of probabilities | Ignoring after fine-tuning | Property |
| Catastrophic forgetting | Loss of pretrained ability | Ignoring after fine-tuning | Phenomenon |
| Parameter-efficient fine-tuning | Training few parameters | Confusing with full fine-tune | Method |

### 4.1 Published foundation models in imaging

| Model | Modality | Pretraining scale | Reported evaluation | Caveat |
|---|---|---|---|---|
| MedSAM | 10 modalities | 1,570,263 image-mask pairs | 86 internal, 60 external tasks | Promptable; modality imbalance |
| RETFound | Retinal CFP and OCT | 1.6M images | DR, glaucoma, AMD, systemic disease | Retinal only |
| UNI | Pathology H&E | >100M patches, >100k slides | 34 tasks | Tile-level; stain dependence |
| CONCH | Pathology H&E | 1.17M image-caption pairs | 14 benchmarks | Vision-language; grounding limits |
| BiomedCLIP | Biomedical images | 15M image-text pairs | Retrieval and classification | Domain breadth vs depth |

The scale figures are from the published papers. They are not a quality ranking. A smaller model
trained on the target domain can outperform a larger model trained on a different domain.

### 4.3 Prompt types

| Prompt | Input | Strength | Weakness | Typical use |
|---|---|---|---|---|
| Point | One click inside the object | Fast | Ambiguous for irregular shapes | Quick annotation |
| Box | Bounding box | Robust for compact objects | Poor for branching structures | Lesion segmentation |
| Mask | Coarse mask | Most informative | Requires an initial segmentation | Refinement |
| Text | Description | Flexible | Grounding is weak | Zero-shot exploration |
| Scribble | Rough line | Fast for elongated objects | Ambiguous | Interactive editing |

The prompt type is part of the input contract. A model validated with boxes should not be deployed
with points without re-evaluation.

### 4.2 Adaptation by label budget

| Label budget | Recommended strategy | What to compare |
|---|---|---|
| 0 | Zero-shot with a fixed prompt | Prompt variants, task-specific baseline |
| <100 labels | Linear probe or LoRA | Frozen features vs full fine-tune |
| 100-1,000 | Fine-tune with early stopping | Task-specific model at same budget |
| >1,000 | Full fine-tune or from scratch | Cost and marginal gain |

## 5. The mechanics, step by step

This procedure evaluates and adapts a foundation model for an imaging task. It assumes the data
passed ingestion, label, and leakage checks. Each step states input, output, rationale, branches,
and edge cases.

### Step 1. Define the task, unit, and claim

**Input:** The clinical problem and the workflow.

**Output:** A task statement with the unit (patient, lesion, voxel), the reference standard, and
the claim.

**Rationale:** A foundation model does not define the task. The task defines the adaptation, the
metric, and the claim. A segmentation claim and a classification claim use the same encoder
differently.

**Branch points:**
- Classification: study or patient label.
- Detection: lesion boxes.
- Segmentation: voxel mask.
- Report generation: text with grounding.

**Edge case:** A vision-language model can generate a report, but a report-generation claim needs
a grounding and factuality evaluation, not only a text similarity metric.

### Step 2. Audit the pretraining corpus

**Input:** The model card, the paper, and any available corpus information.

**Output:** A provenance statement: data sources, modalities, institutions, dates, and known
exclusions.

**Rationale:** The pretraining distribution determines the priors and the contamination risk. A
model trained on the same public benchmark used for evaluation is not a fair test.

**Branch points:**
- Corpus public and documented: check for test overlap.
- Corpus private: request provenance and exclusion list.
- Corpus unknown: treat as contaminated and use a private test set.

**Edge case:** A model may be trained on data that is public but not the same version as the test
set. Near-duplicates can still exist.

### Step 3. Define the input contract

**Input:** The model's expected resolution, channels, normalization, and prompt.

**Output:** A specification that the deployment pipeline can implement exactly.

**Rationale:** A foundation model is sensitive to resolution and preprocessing. A pathology model
trained at 20x and deployed at 40x sees a different object scale. A CT model trained on 1 mm and
deployed on 5 mm sees a different partial-volume regime.

**Branch points:**
- Fixed resolution: resize and document the interpolation.
- Native resolution: tile and aggregate.
- Multi-resolution: choose the level that matches training.
- Promptable: define the prompt type and the user interaction.

**Edge case:** A model card that says "224 × 224" without the magnification or the physical pixel
size is incomplete for pathology.

### Step 4. Choose the adaptation strategy

**Input:** The label budget, the task, and the compute.

**Output:** A strategy: zero-shot, linear probe, fine-tune, LoRA, or prompt tuning.

**Rationale:** The strategy determines what is learned and what is at risk. A linear probe tests
the representation; a fine-tune adapts it and can overfit. The choice should be justified by the
label budget and the baseline.

**Branch points:**
- No labels: zero-shot, with a prompt sensitivity analysis.
- Few labels: linear probe or LoRA.
- Moderate labels: fine-tune with a held-out validation.
- Large labels: full fine-tune or from scratch.

**Edge case:** A fine-tune on a small dataset can degrade the pretrained features. Use a low
learning rate, early stopping, and a frozen-encoder ablation.

### Step 5. Build the task-specific baseline

**Input:** The same data and split.

**Output:** A baseline model, often a task-specific CNN or a simple classifier.

**Rationale:** A foundation model is only useful if it beats the alternative at the same label
budget. The baseline is the alternative. Without it, the result is a demonstration, not a
comparison.

**Branch points:**
- Small data: baseline may be a linear model on handcrafted features.
- Large data: baseline may be a task-specific network.
- Segmentation: baseline is nnU-Net or a 3D U-Net.

**Edge case:** A baseline trained with a different preprocessing is not a fair comparison. Use the
same pipeline.

### Step 6. Measure the label-efficiency curve

**Input:** The model, the baseline, and label subsets.

**Output:** A curve of the primary metric versus the number of labels, with confidence intervals.

**Rationale:** A single label budget can be chosen to favour the foundation model. The curve shows
the marginal value of labels and the crossover point.

**Branch points:**
- Foundation wins at low labels and ties at high: use it for label efficiency.
- Foundation wins at all budgets: strong evidence.
- Foundation loses at all budgets: the pretraining does not transfer.
- Crossover: choose by the available label budget.

**Edge case:** The subsets must be nested or independently sampled with the same patient-level
split. A subset that excludes hard cases flatters the model.

### Step 7. Test contamination and near-duplicates

**Input:** The test set and any accessible pretraining data.

**Output:** A contamination report with exact and near-duplicate counts.

**Rationale:** Contamination inflates performance and invalidates the comparison. The test must be
independent of the pretraining corpus at the patient level.

**Branch points:**
- Exact duplicate: remove from test or pretraining.
- Near-duplicate: perceptual hash or embedding distance.
- Patient overlap: identifier join if available.
- Unknown corpus: use a private test set and state the limitation.

**Edge case:** A near-duplicate can be a different acquisition of the same patient. The audit must
use patient identifiers when available, not only image hashes.

### Step 8. Evaluate calibration after adaptation

**Input:** The adapted model's probabilities and the reference.

**Output:** Calibration curve, slope, intercept, and Brier score.

**Rationale:** Fine-tuning changes the score distribution. A model that was calibrated before
fine-tuning may not be calibrated after. A clinical threshold needs a calibrated probability.

**Branch points:**
- Calibrated: proceed.
- Miscalibrated: recalibrate on a validation set.
- Calibration varies by subgroup: recalibrate per subgroup or report the limitation.

**Edge case:** A zero-shot model may output a similarity score, not a probability. Converting it
to a probability requires a calibration set.

### Step 9. Analyse subgroups and sites

**Input:** The adapted model and the subgroup fields.

**Output:** Performance by site, scanner, demographic, and acquisition.

**Rationale:** A foundation model can encode the pretraining population. A model trained mostly on
one scanner or one ancestry group may perform differently on others. The subgroup analysis is the
evidence of transportability.

**Branch points:**
- Consistent: proceed.
- Site-specific drop: report and consider adaptation.
- Demographic drop: report and investigate.

**Edge case:** A subgroup with few cases gives a wide interval. Label it descriptive.

### Step 10. Measure compute and latency

**Input:** The model and the deployment hardware.

**Output:** Parameters, FLOPs, latency, and memory.

**Rationale:** A large foundation model may not fit the deployment budget. A model that takes
minutes per study is not compatible with a concurrent workflow.

**Branch points:**
- Fits: deploy.
- Too large: distill, quantize, or use a smaller model.
- Too slow: cache embeddings or batch.

**Edge case:** A quantized model may lose accuracy. Re-evaluate after quantization.

### Step 11. Map the regulatory path

**Input:** The adaptation and the update plan.

**Output:** A regulatory strategy: locked model, PCCP, or new submission.

**Rationale:** A foundation model that is fine-tuned on new data is a changed device unless the
change is within a PCCP. The pretraining corpus and the adaptation data are part of the device's
data management.

**Branch points:**
- Locked model: standard submission.
- Planned updates: PCCP with a modification protocol.
- Continuous learning: likely a new submission or a research-use-only product.

**Edge case:** A foundation model's pretraining corpus may be outside the manufacturer's control.
The labeling and the risk file must state the dependency.

### Step 12. Monitor after deployment

**Input:** The deployed model and the logs.

**Output:** Drift metrics, performance audits, and a re-evaluation trigger.

**Rationale:** The pretraining distribution is fixed; the deployment distribution is not. A new
scanner or protocol can shift the embeddings and the output. Monitoring detects the shift before
the clinical outcome does.

**Branch points:**
- Embedding drift: monitor the distribution.
- Output drift: monitor the score and alert rate.
- Performance drift: audit a labeled sample.

**Edge case:** A drift in embeddings may not change the output, and vice versa. Monitor both.

### 5.1 Pretraining from scratch or adapting

Building a foundation model from scratch requires a corpus, compute, and a pretraining objective.
For most imaging teams, the corpus and the compute are not available, and the adaptation of an
existing model is the realistic path.

The decision is not only cost. A model pretrained on a different modality may have features that
do not transfer, and a model pretrained on a public corpus may be contaminated. The build option
gives control over the corpus and the objective; the adapt option gives speed.

A middle path is self-supervised pretraining on the target site's unlabeled images, followed by
supervised fine-tuning. It requires unlabeled data but no labels, and it can capture the site's
scanner characteristics. The risk is that the site's data are not diverse enough to learn a
general representation.

### 5.2 Embedding reuse and multiple-instance learning

A foundation encoder can be used as a feature extractor for a downstream model. In pathology, a
whole-slide image is tiled, each tile is embedded, and a multiple-instance learning (MIL) model
aggregates the tile embeddings into a slide-level prediction. The encoder is frozen and the MIL
model is trained on slide labels.

The approach is label-efficient because the encoder is shared across tasks. It is also sensitive
to the tile size, the magnification, and the stain distribution. A MIL model trained on 20x tiles
and deployed on 40x tiles sees a different object scale, even though the encoder is the same.

The evaluation must include the tile-level and slide-level metrics, because a slide-level AUC can
hide a tile-level failure.

### 5.3 The adaptation evaluation protocol

The adaptation protocol should be written before the fine-tune. It states the label subsets, the
number of seeds, the learning rate, the early-stopping rule, the primary metric, and the
comparison to the baseline. A fine-tune that is tuned on the test set is not an evaluation.

The protocol should also state the frozen-encoder ablation. If a linear probe matches the
fine-tune, the extra adaptation is not justified, and a simpler product is available. If the
fine-tune is much better, the representation needed adaptation, and the label budget must be
large enough to support it.

### 5.4 The encoder is not the product

A foundation encoder is a component. The product is the encoder plus the adaptation plus the
prompt plus the interface plus the monitoring. A team that licenses an encoder and ships a
threshold has shipped a component, not a device.

The integration work is the same as Topic 08: the input contract, the result object, the failure
path, and the workflow role. A foundation model does not reduce that work; it changes the model
inside it.

### 5.5 Decision table: build, adapt, or buy

| Situation | Option | Why | Risk |
|---|---|---|---|
| Large labeled dataset, narrow task | Task-specific model | Data and task are sufficient | Less reusable |
| Small labeled dataset, common anatomy | Fine-tune a foundation model | Label efficiency | Contamination, domain gap |
| No labels, exploration | Zero-shot | Fast signal | Prompt sensitivity |
| Multiple tasks, one modality | Foundation encoder plus heads | Reuse | Shared bias |
| Regulated product, fixed task | Locked fine-tuned model | Predictable | Update cost |
| Research prototype | Any | Speed | Not a product |

### 5.6 Edge cases that require a written rule

**Mixed slice thickness.** A model pretrained on thin CT and deployed on thick CT has a different
partial-volume regime. Resample or stratify.

**Missing series.** A multimodal foundation model with a missing sequence needs a missingness
indicator or a different model. Do not impute with zeros.

**Inconsistent spacing.** The input contract must state the physical spacing, not only the matrix
size.

**Rescale absent.** A CT foundation model trained on HU must not receive stored values.

**Burned-in annotation.** A model can learn the annotation. Crop and test.

**Laterality mismatch.** A side-specific task needs a verified side.

**Multi-frame objects.** The input contract must define the frame handling.

**Private tags carrying labels.** Remove before adaptation and inference.

## 6. Worked example

### 6.1 The case

A team is adapting MedSAM, a promptable segmentation foundation model, to prostate segmentation on
T2-weighted MRI. The dataset has 60 studies from two scanners, with 80 reference masks drawn by an
expert and edited by a second reader. The clinical use is contouring for radiotherapy planning,
where the endpoint is per-lesion Dice and the editing time.

The comparator is nnU-Net trained on the same data. The team wants to know whether the foundation
model reduces the annotation burden without losing accuracy.

### 6.2 Zero-shot performance

MedSAM is prompted with a bounding box around the prostate, derived from the expert mask. On the 80
lesions, the median Dice is 0.71, with an interquartile range of 0.62 to 0.79. The per-lesion
sensitivity at a Dice threshold of 0.5 is:

```text
68 / 80 = 0.850
```

Zero-shot performance is not sufficient for contouring, but it is a useful initialization. The
result depends on the prompt: with a point prompt instead of a box, the median Dice falls to 0.62
and the sensitivity to 0.71.

### 6.3 Fine-tuned performance

Fine-tuning the decoder on all 80 masks for 50 epochs, with the encoder frozen, raises the median
Dice to 0.863 and the sensitivity to:

```text
76 / 80 = 0.950
```

Wilson 95% confidence interval for 76/80:

```text
p = 0.950, n = 80
centre ≈ 0.929
half-width ≈ 0.051
CI ≈ 0.878 to 0.980
```

nnU-Net trained on the same 80 masks has a median Dice of 0.84 and a sensitivity of 0.925
(74/80). The fine-tuned foundation model is 0.025 Dice better, which is within the reader-to-
reference agreement of about 0.03 Dice. The difference is not clinically material on this sample.

### 6.4 Label efficiency

The team repeats the fine-tune with nested subsets of the labels:

| Label fraction | Labels | Median Dice | Sensitivity |
|---|---:|---:|---:|
| 10% | 8 | 0.780 | 0.800 |
| 25% | 20 | 0.830 | 0.875 |
| 50% | 40 | 0.850 | 0.925 |
| 100% | 80 | 0.865 | 0.950 |

The curve rises steeply between 10% and 25%, then flattens. The marginal gain from 50% to 100% is
0.015 Dice for 40 additional masks. At an annotation cost of 14 minutes per mask, the last 40
masks cost:

```text
40 × 14 = 560 minutes = 9.3 hours
```

for 0.015 Dice. The team decides that 50% of the labels is the efficient operating point for this
task, unless the clinical requirement demands the higher sensitivity.

### 6.5 Annotation time with prompts

Manual contouring takes 14.0 minutes per prostate. With a box prompt and manual correction, the
time falls to 2.6 minutes:

```text
reduction = (14.0 - 2.6) / 14.0 = 0.814 = 81.4%
```

That is consistent with the order of magnitude reported by MedSAM's authors, who reported about an
82% reduction in a tumour-annotation user study. The number is task- and user-dependent and should
be measured locally, not assumed.

### 6.6 Diagnostic 1: pretraining contamination

The test set includes 12 studies from a public prostate MRI collection. The foundation model's
pretraining corpus included that collection. A nearest-neighbour search in the embedding space
finds 9 of the 12 test studies with a cosine similarity above 0.98 to a pretraining image.

The zero-shot Dice on those 9 studies is 0.79, compared with 0.68 on the 71 studies that are not
near-duplicates. The 0.11 difference is contamination, not generalization. The diagnostic is the
near-duplicate count and the performance difference between duplicate and non-duplicate subsets.
The remedy is to remove the overlapping studies and re-evaluate on a private test set.

### 6.7 Diagnostic 2: scanner shift

The two scanners are a 3T Siemens and a 1.5T Philips. The fine-tuned model's median Dice is 0.89
on the Siemens data and 0.79 on the Philips data. The gap is 0.10 Dice, larger than the difference
between the foundation model and nnU-Net.

The diagnostic is a stratified Dice by scanner. The remedy is either to include both scanners in
the fine-tuning set or to report the scanner-specific performance and restrict the claim. A
foundation model does not remove the scanner effect; it inherits it from the pretraining and the
adaptation data.

### 6.8 Varying one assumption

Assume the prompt is a point inside the prostate rather than a box. The zero-shot median Dice
falls from 0.71 to 0.62, and the fine-tuned model's Dice falls from 0.865 to 0.81 because the
fine-tuning was done with box prompts. The conclusion changes from "the foundation model is close
to nnU-Net" to "the foundation model is worse than nnU-Net with a point prompt".

The prompt is part of the input contract. A product that allows a point prompt must be validated
with a point prompt, not with an expert box.

### 6.9 What the label curve implies

The curve rises from 0.780 at 8 masks to 0.830 at 20 masks, a gain of 0.050 for 12 masks. From 20
to 40 masks, the gain is 0.020 for 20 masks. From 40 to 80, the gain is 0.015 for 40 masks.

The marginal Dice per mask is:

```text
8 to 20:   0.050 / 12 = 0.0042 Dice per mask
20 to 40:  0.020 / 20 = 0.0010 Dice per mask
40 to 80:  0.015 / 40 = 0.0004 Dice per mask
```

The first 12 masks are worth ten times the last 40. The efficient budget is where the marginal
gain falls below the clinical value of a mask. That value is a clinical and economic decision, not
a property of the model.

### 6.10 The baseline that was not run

A team that reports only the fine-tuned foundation model has not shown that the foundation model
is better than a task-specific model. In this case, nnU-Net on the same 80 masks reaches 0.84
Dice, within 0.025 of the foundation model. The foundation model's advantage is the label
efficiency, not the ceiling.

The diagnostic is the baseline column in the results table. If it is missing, the comparison is
incomplete. If it is present but trained on different preprocessing, the comparison is unfair.

### 6.11 The cost arithmetic

The fine-tune uses 80 masks at 14 minutes each for the reference, plus 2.6 minutes of correction
per prompt. The annotation cost for the 80 masks is:

```text
80 x (14.0 + 2.6) = 80 x 16.6 = 1,328 minutes = 22.1 hours
```

A fully manual 80-mask set would cost:

```text
80 x 14.0 = 1,120 minutes = 18.7 hours
```

The prompt-assisted workflow is more expensive per mask in this accounting because it includes
both the reference and the correction. The saving appears when the reference is the model output
and the expert only corrects it, which is the intended workflow:

```text
80 x 2.6 = 208 minutes = 3.5 hours
```

The accounting must state which workflow is being compared. A saving that compares correction time
to manual drawing is real; a saving that compares correction plus reference to manual drawing is
not.

### 6.12 The claim the evidence supports

The evidence supports a claim that the fine-tuned model segments the prostate with a median Dice
of 0.86 on two scanners, with a scanner-specific range of 0.79 to 0.89, and that it reduces
correction time relative to manual contouring. It does not support a claim of autonomous
contouring, a claim on a third scanner, or a claim that the model is better than nnU-Net by a
clinically meaningful margin.

The claim is narrower than the model's capability and wider than the zero-shot result. That is the
normal shape of an honest claim.

## 7. Code, query, or calculation

The following code computes the Wilson interval, a bootstrap interval for the median Dice, the
label-efficiency table, and the annotation-time reduction. It uses only the standard library.

```python
import math
import statistics

def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half

lo, hi = wilson(76, 80)
print("Wilson 76/80      = {:.3f} to {:.3f}".format(lo, hi))

dices = [0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.875, 0.865,
         0.86, 0.855, 0.85, 0.84, 0.83, 0.82, 0.80, 0.78, 0.74, 0.68]
print("median Dice       = {:.3f}".format(statistics.median(dices)))
print("mean Dice         = {:.3f}".format(statistics.mean(dices)))

def bootstrap_median(values, reps=2000, seed=20261025):
    state = seed
    n = len(values)
    estimates = []
    for _ in range(reps):
        sample = []
        for _ in range(n):
            state = (1103515245 * state + 12345) % (2 ** 31)
            sample.append(values[state % n])
        estimates.append(statistics.median(sample))
    estimates.sort()
    return estimates[int(0.025 * reps)], estimates[int(0.975 * reps) - 1]

lo_b, hi_b = bootstrap_median(dices)
print("bootstrap median  = {:.3f} to {:.3f}".format(lo_b, hi_b))

manual, assisted = 14.0, 2.6
print("time reduction    = {:.1f}%".format((manual - assisted) / manual * 100))

for frac, dice in [(0.10, 0.780), (0.25, 0.830), (0.50, 0.850), (1.00, 0.865)]:
    print("labels {:>4.0%} -> Dice {:.3f}".format(frac, dice))
```

Expected output:

```text
Wilson 76/80      = 0.878 to 0.980
median Dice       = 0.863
mean Dice         = 0.856
bootstrap median  = 0.835 to 0.895
time reduction    = 81.4%
labels  10% -> Dice 0.780
labels  25% -> Dice 0.830
labels  50% -> Dice 0.850
labels 100% -> Dice 0.865
```

The bootstrap interval contains the point estimate, as it should. The interval is wide because the
sample is small: 20 lesion Dice values, with a skewed lower tail from two difficult lesions. With
the full 80-lesion set, the interval would be narrower. The code illustrates the method; the
width is a property of the sample, not of the bootstrap.

The code assumes: the Dice values are independent, the 20 values are representative, and the
percentile method is appropriate for a median. If lesions are clustered by patient, resample
patients, not lesions.

## 8. Data, standards, and regulatory context

### 8.1 DICOM and the input contract

The input contract is defined by the DICOM attributes from Topic 02: modality, spacing, thickness,
orientation, and rescale. A foundation model that expects a fixed matrix size must document the
resampling, and a model that expects a fixed physical spacing must document the interpolation. The
contract is part of the device's labeling.

### 8.2 Reporting standards

CLAIM (2020, updated 2024) and TRIPOD+AI (2024) apply to foundation-model studies. The report
should state the pretraining corpus, the adaptation data, the label budget, the split, the
contamination audit, the baseline, the calibration, and the subgroup results. A report that says
"we fine-tuned a foundation model" without the corpus and the baseline is not reproducible.

### 8.3 FDA and the adaptive model

A foundation model that is fine-tuned for a specific task is a device software function. The
special controls at 21 CFR 892.2090 apply if the claim is detection or diagnosis. The FDA's PCCP
guidance (final August 2025) allows planned updates within a pre-specified envelope, but a change
in the pretraining corpus or the adaptation population may be outside the envelope.

The FDA's good machine learning practice principles (2021) emphasize representative data,
independent test sets, and monitoring. A foundation model's pretraining corpus is part of the data
management, even if the manufacturer did not collect it.

### 8.4 EU AI Act

Medical devices that require a notified body are high-risk AI systems under the AI Act. The
high-risk obligations for Annex I products were deferred to 2 August 2028 by the Digital Omnibus
regulation adopted in 2026. The MDR obligations apply independently. A foundation model that is
part of a medical device is in scope; a research model is not, until it is placed on the market.

### 8.5 What the standards do not cover

No standard defines the acceptable label efficiency, the minimum pretraining corpus, or the
contamination threshold. Those are design and evidence decisions that must be justified in the
submission and the report.

### 8.6 Model cards for foundation models

A model card for a foundation model should state the pretraining corpus, the modalities, the
inclusion and exclusion criteria, the known gaps, the adaptation data, the evaluation data, the
metrics, the calibration, the subgroups, and the intended use. It should also state the license
and the permitted uses.

A card that says "trained on 1 million medical images" without the modality distribution is not
useful. A card that says "not for clinical use" is useful, because it sets the boundary.

### 8.7 Licensing and data rights

A foundation model comes with a license. Some licenses permit commercial use; some permit
research only; some require attribution or restrict redistribution. The license is a product
constraint, not a legal detail to resolve at launch.

The adaptation data have their own rights. A hospital that provides images for fine-tuning may
retain ownership and restrict the use of the derived model. The contract should state whether the
vendor can use the data to improve the model, whether the model can be used at other sites, and
what happens when the contract ends.

## 9. Industry practice and case evidence

### 9.1 MedSAM

MedSAM (Ma et al., Nature Communications 2024) fine-tuned SAM's image encoder and mask decoder on
1,570,263 image-mask pairs across 10 imaging modalities and 30+ cancer types. It reported
performance on par with or better than modality-specific nnU-Net and DeepLabV3+ on many tasks,
with better generalization to unseen targets, and about an 82% reduction in expert annotation
time in a tumour-annotation user study. The paper notes modality imbalance (CT, MRI, and endoscopy
dominate) and difficulty with vessel-like branching structures.

The documented facts are the dataset size, the modality count, and the reported comparisons. The
inference is that a promptable model can accelerate annotation; it does not replace task-specific
validation.

### 9.2 RETFound

RETFound (Zhou et al., Nature 2023) is a masked-autoencoder model pretrained on 1.6 million
unlabelled retinal images (904,170 colour fundus photographs and 736,442 OCT images). It
outperformed ImageNet-based baselines on diabetic retinopathy and glaucoma diagnosis, wet AMD
prognosis, and three-year prediction of systemic disease, with strong label efficiency.

The case shows that domain-specific pretraining can beat general pretraining. It also shows the
limit: the model is retinal, and the systemic disease predictions are associations, not
diagnoses.

### 9.3 UNI and CONCH

UNI (Chen et al., Nature Medicine 2024) is a self-supervised vision transformer pretrained on more
than 100 million patches from more than 100,000 haematoxylin and eosin slides, evaluated on 34
computational pathology tasks. CONCH (Lu et al., Nature Medicine 2024) is a vision-language model
pretrained on 1.17 million image-caption pairs, evaluated on 14 benchmarks including zero-shot
classification, segmentation, captioning, and retrieval.

The two models illustrate the trade-off between a vision-only encoder and a vision-language model.
CONCH can do zero-shot text-guided tasks; UNI is a stronger feature extractor for supervised
heads. Neither removes the need for a task-specific test.

### 9.4 PathBench and the contamination problem

PathBench (2025) is a benchmark for pathology foundation models built on private multi-centre data:
15,888 whole-slide images from 8,549 patients across 10 hospitals, with 64 diagnosis and prognosis
tasks. The design explicitly excludes the evaluation data from pretraining, addressing the
contamination risk that arises when public benchmarks are used for both pretraining and testing.
It evaluated 19 pathology foundation models and reported consistent internal-external validation
gaps.

The lesson is that a foundation-model benchmark must control the pretraining corpus, not only the
test split. A public benchmark that was scraped into a pretraining corpus is no longer an
independent test.

### 9.5 Inference, not fact

The following are inferences: that a specific foundation model will transfer to a specific
hospital's scanners; that a specific label-efficiency curve will hold for a new task; that a
specific prompt will be usable by radiologists. Each requires a local evaluation. The documented
facts are the published pretraining scales and benchmark results.

### 9.6 BiomedCLIP and the multimodal direction

BiomedCLIP is a vision-language model pretrained on biomedical image-text pairs. It is used for
retrieval, zero-shot classification, and as a feature extractor. The multimodal direction is
promising for report generation and image-text retrieval, but the evaluation is harder: fluency is
not grounding, and a caption that sounds correct can describe a finding that is not in the image.

A multimodal claim needs a grounding evaluation, not only a text similarity metric. The model must
be tested on whether the generated text is supported by the image, not only whether it resembles a
reference report.

### 9.7 Radiology and pathology differ

Radiology foundation models work on volumetric or projection images with a calibrated or
semi-calibrated intensity scale. Pathology foundation models work on gigapixel RGB slides with
stain variation and a pyramid structure. The evaluation differs: radiology reports per-lesion Dice
and detection at FP/scan; pathology reports slide-level AUC and tile-level agreement.

The common error is to transfer a benchmark from one to the other. A model that is strong on
pathology tiles is not evidence for a CT task, and a radiology foundation model is not evidence
for a whole-slide task. The modality and the unit must match.

### 9.8 What the published results do not tell you

The published results tell you that a model can learn a representation from a large corpus and
that the representation transfers to some benchmarks. They do not tell you the label efficiency on
your task, the performance on your scanner, the calibration after your fine-tune, or the
regulatory path for your claim. Those are local measurements.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Test contamination | Zero-shot too good | Test data in pretraining | Inflated result | Private test set |
| Near-duplicate overlap | High embedding similarity | Same patient or image | Inflated result | Hash and patient audit |
| Benchmark overfitting | Gains vanish on new site | Public benchmark reuse | False generalization | Site-held-out test |
| Prompt mismatch | Box vs point difference | Prompt is part of input | Performance drop | Validate prompt policy |
| Resolution mismatch | Object scale differs | 20x vs 40x, 1 mm vs 5 mm | Segmentation error | Match physical scale |
| Stain shift | Colour differs by site | Pathology batch effect | False positives | Stain normalization |
| Calibration loss | Probabilities extreme | Fine-tuning | Threshold errors | Recalibrate |
| Catastrophic forgetting | Base tasks degrade | Full fine-tune | Narrow model | LoRA or low LR |
| Compute too high | Latency exceeds budget | Large model | Not deployable | Distill or quantize |
| Corpus unknown | No provenance | Vendor opacity | Contamination risk | Private test |
| Subgroup drop | Site-specific Dice | Pretraining population | Equity risk | Stratify and adapt |
| Zero-shot claim | No calibration | Similarity score | Threshold invalid | Calibration set |
| Adaptive update | Model changes silently | Continuous learning | Regulatory gap | PCCP or lock |
| Benchmark leakage | Public test reused | Scraped corpus | Invalid comparison | Exclusion list |

### 10.1 Red flags that stop a foundation-model claim

1. The pretraining corpus is unknown and the test set is public.
2. The zero-shot result is reported without a prompt sensitivity analysis.
3. The baseline is missing or trained on different data.
4. The label-efficiency curve has one point.
5. The calibration is not reported after fine-tuning.
6. The model is updated continuously without a PCCP.
7. The compute is measured only on a workstation.

### 10.2 A foundation-model evaluation checklist

1. The pretraining corpus is described, with modalities and institutions.
2. The test set is independent of the corpus at the patient level.
3. The input contract matches deployment.
4. A task-specific baseline is trained on the same data.
5. The label-efficiency curve has at least four points.
6. The prompt policy is specified and tested.
7. Calibration is reported after adaptation.
8. Subgroups and sites are reported.
9. Compute and latency are measured on deployment hardware.
10. The update strategy is locked or covered by a PCCP.

A foundation-model claim that skips any of these is a benchmark result, not a product claim.

### 10.3 The foundation-model red flag that is easy to miss

The easiest red flag to miss is a benchmark that was used for pretraining and for evaluation. The
numbers look excellent, the method is reproducible, and the only problem is that the test is not
independent. The audit is a provenance question, not a modelling question.

### 10.4 The evaluation that changes the decision

The evaluation that most often changes a foundation-model decision is the task-specific baseline
at the same label budget. When the baseline is added, the foundation model's advantage often
shrinks or disappears, and the project shifts from "foundation model" to "well-trained task
model". That is a good outcome, not a failure.

### 10.5 The question to ask a vendor

Ask the vendor for the pretraining corpus provenance, the exclusion list, the label-efficiency
curve, the task-specific baseline, the calibration after adaptation, and the scanner-specific
results. A vendor that can answer those questions is selling evidence. A vendor that can answer
only the benchmark question is selling a model.

### 10.6 What a good foundation-model paper reports

A good paper reports the corpus, the adaptation data, the baseline, the label curve, the
contamination audit, the calibration, the subgroups, and the compute. It also reports the failures,
including the tasks where the model did not transfer. The failures are the most useful part for a
reader who is deciding whether to adopt it.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The pretraining corpus is independent of the test | Valid evaluation | Public benchmark reuse | Provenance audit |
| The input contract matches deployment | Transfer | Resolution or stain shift | Contract test |
| The prompt is reproducible | Product consistency | User-drawn prompts | Prompt study |
| The baseline is fair | Attribution | Different preprocessing | Same pipeline |
| The label budget is realistic | Efficiency claim | Unrepresentative subset | Nested subsets |
| Calibration survives fine-tuning | Threshold validity | Score shift | Calibration curve |
| The subgroups are represented | Equity | Corpus imbalance | Stratified analysis |
| The compute fits deployment | Feasibility | Large model | Latency test |
| The update is controlled | Regulatory | Continuous learning | PCCP review |
| The foundation model is not a diagnosis | Claim validity | Marketing language | Label review |

### 11.1 Limits of this topic

This topic covers the evaluation and adaptation of foundation models for imaging. It does not cover
the full theory of self-supervised learning, the training of a foundation model from scratch, or
the details of vision-language pretraining. It gives the data scientist the evaluation discipline
to use a foundation model without overclaiming, not to build one.

### 11.2 What would change the conclusion

If the test set were private and site-held-out, the zero-shot and fine-tuned results would be
lower and more credible. If the baseline were trained on the same 80 masks with the same
preprocessing, the foundation model's advantage might disappear. If the deployment scanner were
the Philips 1.5T, the relevant Dice would be 0.79, not 0.86.

### 11.3 The limit of the foundation-model literature

The literature is young and publication-biased toward positive results. A model that fails to
transfer is less likely to be published than one that succeeds. The base rate of transfer across
tasks and sites is unknown, so the prior should be modest and the local evaluation should be
mandatory.

## 12. Exercises

### Exercise 1. Adaptation choice (recall, 20 minutes)

**Deliverable:** For three label budgets (0, 50, 5,000), choose an adaptation strategy and justify.

**Acceptance criteria:** Three strategies; label budget rationale; baseline for each.

**Time:** 20 minutes.

### Exercise 2. Label-efficiency curve (application, 35 minutes)

**Deliverable:** A table of Dice at 10%, 25%, 50%, and 100% labels for a foundation model and a
baseline, with the crossover point.

**Acceptance criteria:** Four budgets; both models; crossover stated; confidence intervals.

**Time:** 35 minutes.

### Exercise 3. Contamination audit (application, 40 minutes)

**Deliverable:** A plan to detect exact duplicates, near-duplicates, and patient overlap between a
pretraining corpus and a test set.

**Acceptance criteria:** Three methods; thresholds; action for each finding.

**Time:** 40 minutes.

### Exercise 4. Prompt study (design, 35 minutes)

**Deliverable:** A study comparing box, point, and mask prompts for a segmentation task, with the
metric and the user workflow.

**Acceptance criteria:** Three prompts; per-lesion Dice; user time; failure analysis.

**Time:** 35 minutes.

### Exercise 5. Calibration after fine-tuning (application, 30 minutes)

**Deliverable:** A calibration curve and Brier score before and after fine-tuning, with the
recalibration decision.

**Acceptance criteria:** Both curves; slope and intercept; decision stated.

**Time:** 30 minutes.

### Exercise 6. Regulatory strategy (design, 45 minutes)

**Deliverable:** A strategy for a foundation model that will be fine-tuned annually, including the
PCCP scope and the locked alternative.

**Acceptance criteria:** Scope; data management; evaluation; update procedure; labeling.

**Time:** 45 minutes.

### Exercise 7. Subgroup analysis (application, 40 minutes)

**Deliverable:** A table of Dice by scanner and site for a foundation model, with counts and
intervals.

**Acceptance criteria:** At least three strata; counts; intervals; interpretation.

**Time:** 40 minutes.

### Exercise 8. Build-versus-adapt memo (critique, 45 minutes)

**Deliverable:** A 600-word memo comparing building a task-specific model with adapting a
foundation model for a 500-case segmentation task.

**Acceptance criteria:** Cost, data, compute, regulatory path, and evidence; recommendation.

**Time:** 45 minutes.

### Exercise 9. Pretraining provenance (application, 30 minutes)

**Deliverable:** A provenance table for a foundation model, listing the corpus, the modalities,
the institutions, the dates, and the known exclusions.

**Acceptance criteria:** At least six fields; unknown fields flagged; contamination risk stated.

**Time:** 30 minutes.

### Exercise 10. Adaptation protocol (design, 40 minutes)

**Deliverable:** A pre-specified protocol for fine-tuning a foundation model on 200 labels,
including subsets, seeds, early stopping, and the baseline.

**Acceptance criteria:** Four label budgets; three seeds; primary metric; baseline; test-set
lock.

**Time:** 40 minutes.

## 13. Self-check answers

**A1. What is a foundation model?**
A model pretrained on a large, broad corpus and adapted to downstream tasks. It is not a
general-purpose clinical system; its transfer is empirical.

**A2. Why is zero-shot not a product claim?**
Zero-shot performance depends on the prompt, is often uncalibrated, and is measured without
task-specific training. A product claim needs a defined input, a reference standard, and a
calibrated operating point.

**A3. What is label efficiency?**
Performance as a function of the number of labels, usually a curve. A single budget can be chosen
to favour a model; the curve shows the marginal value.

**A4. Why is contamination a problem?**
If the test set was in the pretraining corpus, the model has seen the answers. The performance is
memorization, not generalization. The test must be independent at the patient level.

**A5. What is the difference between a linear probe and fine-tuning?**
A linear probe freezes the encoder and trains a classifier, testing the representation. Fine-tuning
updates the encoder, adapting it but risking overfitting and forgetting.

**A6. Why does fine-tuning affect calibration?**
It changes the score distribution and the decision boundary. A model calibrated before fine-tuning
may be overconfident or underconfident after, so calibration must be re-measured.

**A7. What is the prompt contract?**
The specification of who provides the prompt, what type (box, point, mask), how it is drawn, and
how corrections are handled. The prompt is part of the device.

**A8. How do you audit a pretraining corpus you cannot access?**
Use a private, site-held-out test set, ask the vendor for provenance and exclusions, and run
near-duplicate detection against any accessible data. If the corpus is unknown, state the
limitation.

**A9. Design rubric for Exercise 4.** A strong study compares three prompts with the same cases,
measures per-lesion Dice and user time, and analyses prompt failures. A weak study uses one expert
box and calls it zero-shot.

**A10. Design rubric for Exercise 6.** A strong strategy defines the PCCP scope, the data
management, the evaluation protocol, and the rollback, or locks the model. A weak strategy says
"we will update when needed".

## 14. Interview questions

**Q1. When would you use a foundation model?**
A strong answer says when labels are scarce, the modality is represented in the pretraining
corpus, and a task-specific baseline is available for comparison. A weak answer says always,
because it is state of the art.

**Q2. How do you evaluate a foundation model?**
A strong answer covers the input contract, the baseline, the label-efficiency curve, contamination,
calibration, subgroups, and compute. A weak answer reports zero-shot Dice.

**Q3. What is contamination and how do you detect it?**
A strong answer explains test data in pretraining, exact and near-duplicate checks, and patient
overlap, and proposes a private test set. A weak answer says it is leakage.

**Q4. Why is the prompt part of the product?**
A strong answer says the output depends on the prompt and the user workflow, so the prompt policy
must be specified and validated. A weak answer says the user can draw a box.

**Q5. What is the difference between zero-shot, few-shot, and fine-tuning?**
A strong answer defines each by the amount of task-specific training and the evidence each
provides. A weak answer uses the terms loosely.

**Q6. How do you compare a foundation model to nnU-Net?**
A strong answer uses the same data, split, preprocessing, and metric, and reports confidence
intervals. A weak answer compares to a published number.

**Q7. What are the regulatory challenges of a foundation model?**
A strong answer covers the pretraining corpus, the adaptation data, the PCCP, and the locked versus
adaptive model. A weak answer says the FDA does not regulate AI.

**Q8. What is catastrophic forgetting?**
A strong answer says fine-tuning can degrade the pretrained representation and other tasks, and
proposes a low learning rate, LoRA, or a frozen encoder. A weak answer says it is overfitting.

**Q9. How do you monitor a foundation model after deployment?**
A strong answer monitors embedding drift, output drift, alert rate, and a labeled audit sample.
A weak answer monitors uptime.

**Q10. When is a task-specific model better?**
A strong answer says when the labeled dataset is large, the task is narrow, the compute is
constrained, or the foundation model's corpus does not represent the deployment population. A weak
answer says never.

## 15. Key terms

- **Adapter** — small trained module for adaptation.
- **Benchmark** — standard evaluation dataset.
- **Calibration** — agreement of probabilities with outcomes.
- **Caption** — text paired with an image.
- **Catastrophic forgetting** — loss of pretrained ability after fine-tuning.
- **Contamination** — test data in pretraining.
- **Contrastive learning** — learning from paired and unpaired examples.
- **Distribution shift** — difference between source and target.
- **Embedding** — learned feature vector.
- **Few-shot** — a few labeled examples.
- **Fine-tuning** — updating pretrained weights.
- **Foundation model** — pretrained model adapted to many tasks.
- **Grounding** — linking text to image regions.
- **Label efficiency** — performance per label budget.
- **Linear probe** — frozen encoder plus linear classifier.
- **LoRA** — low-rank adaptation.
- **Masked autoencoder** — reconstruct masked patches.
- **MedSAM** — promptable medical segmentation model.
- **Multi-instance learning** — learning from bags of tiles.
- **Near-duplicate** — almost identical image.
- **Parameter-efficient fine-tuning** — training few parameters.
- **Patch** — local image region.
- **Pretraining** — learning a representation from broad data.
- **Prompt** — input cue for a promptable model.
- **RETFound** — retinal foundation model.
- **Self-supervised learning** — pretraining without labels.
- **Token** — embedded patch or word.
- **UNI** — pathology vision foundation model.
- **CONCH** — pathology vision-language model.
- **Vision transformer** — transformer over image patches.
- **Zero-shot** — no task-specific training.

## 16. Sources

- Ma, J. et al., "Segment anything in medical images," Nature Communications 15, 654 (2024).
  https://doi.org/10.1038/s41467-024-44824-z
- Zhou, Y. et al., "A foundation model for generalizable disease detection from retinal images,"
  Nature 622, 156-163 (2023). https://doi.org/10.1038/s41586-023-06555-x
- Chen, R. J. et al., "Towards a general-purpose foundation model for computational pathology,"
  Nature Medicine 30, 850-862 (2024). https://doi.org/10.1038/s41591-024-02857-3
- Lu, M. Y. et al., "A visual-language foundation model for computational pathology," Nature
  Medicine 30, 863-874 (2024). https://doi.org/10.1038/s41591-024-02856-4
- PathBench, "A comprehensive comparison benchmark for pathology foundation models towards
  precision oncology," arXiv:2505.20202 (2025). https://doi.org/10.48550/arxiv.2505.20202
- Mongan, J., Moy, L., Kahn, C. E. Jr., "Checklist for Artificial Intelligence in Medical Imaging
  (CLAIM)," Radiology: Artificial Intelligence 2(2): e200029 (2020).
  https://doi.org/10.1148/ryai.2020200029
- Tejani, A. S. et al., "Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024
  Update," Radiology: Artificial Intelligence 6: e240300 (2024).
  https://doi.org/10.1148/ryai.240300
- Collins, G. S. et al., "TRIPOD+AI statement," BMJ 385: e078378 (2024).
  https://doi.org/10.1136/bmj-2023-078378
- U.S. Food and Drug Administration, "Marketing Submission Recommendations for a Predetermined
  Change Control Plan for Artificial Intelligence-Enabled Device Software Functions," final
  guidance, 18 August 2025.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
  (accessed 25 October 2026).
- U.S. Food and Drug Administration, 21 CFR 892.2090, final order 90 FR 24969, 13 June 2025.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892.2090
  (accessed 25 October 2026).
- European Union, Regulation (EU) 2024/1689 (AI Act).
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj (accessed 25 October 2026).
- International Medical Device Regulators Forum, Good Machine Learning Practice principles (2021).
  https://www.imdrf.org/ (accessed 25 October 2026).

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`09_regulatory-path.md`](09_regulatory-path.md)
- Related topic: [`05_architectures.md`](05_architectures.md)
- Related topic: [`06_leakage-and-shortcuts.md`](06_leakage-and-shortcuts.md)
- Root survey: [`../../10_medical_imaging.md`](../../10_medical_imaging.md)
- Labs: [`../../labs/`](../../labs/)

> **Note:** A foundation model is a prior, not a product claim.
> The local evaluation is mandatory.

> **Note:** The pretraining corpus is part of the device evidence.
> Ask for it, test against it, and state the limitation if it is unavailable.
> A benchmark is not an independent test if the benchmark was in the corpus.
