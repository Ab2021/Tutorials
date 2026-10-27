# Pathology Foundation Models

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 12 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 05, 07, 10, and 11
> Version 2026.3

## 1. Why this topic exists

A pathology foundation model is a vision encoder pretrained on millions of tissue images without
human labels, then reused for tasks that have only slide-level labels. The appeal is economic.
Pixel-level annotation by a pathologist is slow and expensive, inter-observer agreement on
boundaries is moderate, and a model trained on one hospital's scanner often loses accuracy at the
next. A pretrained encoder promises to reduce all three costs: fewer labels, less task-specific
engineering, and representations that transfer across sites.

The promise is real enough that most computational pathology groups now start from a pretrained
encoder rather than from random initialisation. It is also oversold in a specific way that this
topic exists to correct. A published area under the receiver operating characteristic curve
(AUROC) on a public benchmark is evidence about a model on that benchmark. It is not evidence
about a clinical claim, a new site, a new scanner, or a new prevalence. The gap between the two is
where deployments fail.

Four decisions are blocked until someone can read the evidence correctly:

| Decision | What this topic provides |
|---|---|
| Build or reuse an encoder | What pretraining buys, and what it costs in licence and compute |
| Adaptation strategy | When a linear probe is sufficient and when fine-tuning is justified |
| Aggregation | How patch embeddings become one slide-level score |
| Evidence standard | What a benchmark establishes and what the deployment claim still needs |

### 1.1 The label problem that foundation models do not solve

Pathology reports give a diagnosis for a specimen, not for a tile. A slide labelled "metastatic
adenocarcinoma in one of four lymph nodes" does not say which tiles contain tumour. This is the
multiple instance learning (MIL) setting: a bag (the slide) has a label, and the instances
(patches) do not. Foundation models change the encoder inside that setting. They do not change the
fact that supervision is at the bag level, and they do not create patch labels.

That matters for evaluation. A model can achieve a good slide-level AUROC by attending to the
wrong thing: necrosis instead of tumour, ink instead of margin, or a scanner artefact correlated
with the label in the training set. Because the label never says where the tumour is, the usual
supervision cannot correct it. Only a separate localisation check, or a dataset where the
shortcut is broken, reveals the problem.

### 1.2 What changed between 2021 and 2026

Before foundation models, a typical pipeline used a convolutional network trained on
ImageNet, fine-tuned on a task-specific set of regions of interest. Performance was respectable
on large single-site cohorts and fragile across sites. From 2023 onward, three things changed:

1. **Pretraining scale.** Encoders were trained on 100,000 or more whole-slide images rather than
   on a few thousand. UNI reported about 100 million patches from 100,426 slides; Virchow reported
   about 1.5 million slides from 119,629 patients; Prov-GigaPath reported about 1.38 billion tiles
   from 171,189 slides. These are the authors' reported corpora, not independently audited ones.
2. **Self-supervised objectives.** DINOv2-style self-distillation and masked image modelling
   replaced ImageNet supervision, so the representation is learned from histology rather than
   from photographs of animals and objects.
3. **Vision-language pairing.** Models such as CONCH aligned images with pathology report text,
   enabling zero-shot retrieval and classification without task labels.

The result is a set of general-purpose encoders that are genuinely better starting points for
many tasks. The result is not a diagnostic device. The distance between the two is the subject of
Sections 5 to 7.

### 1.3 The failure that motivates the topic

A team trains a linear probe on a foundation model's embeddings for a colorectal lymph node task.
The pooled AUROC is 0.94. The team reports that the model "detects metastasis". A reviewer asks
for site-stratified results. Performance at the smallest site is 0.79. A second reviewer asks what
the attention map highlights: 40 percent of the top-weighted patches are adipose tissue and
surgical ink, which correlate with node dissection in the training set. The model is a site
detector with a tumour-shaped shadow.

Neither number was false. The pooled number was simply not the claim that the team thought it was
making. Most of this topic is about constructing evaluation so that this mistake is hard to make.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** an encoder from an aggregator, and state which one a proposed change modifies.
2. **Compare** UNI, CONCH, Virchow, Prov-GigaPath, and CHIEF by pretraining corpus, objective, and
   the tasks their authors evaluated, without ranking them.
3. **Choose** among zero-shot, linear probing, parameter-efficient fine-tuning, and full
   fine-tuning from the label budget and the claim.
4. **Implement** attention-based multiple instance learning over cached embeddings and explain
   what the attention weights do and do not mean.
5. **Design** a patient-level and site-level split, and demonstrate that a random split inflates
   performance.
6. **Diagnose** site shift with a site classifier trained on embeddings and with site-stratified
   metrics.
7. **Compute** positive and negative predictive values from sensitivity, specificity, and
   prevalence, and show how prevalence changes the clinical reading of a fixed model.
8. **State** what a benchmark result does not establish, and what evidence a deployment claim
   still requires.

## 3. The core mental model

A pathology foundation model is not one model. It is an encoder that turns a patch into a vector,
an aggregator that turns a set of vectors into a slide score, and an evaluation protocol that
decides whether the score means anything. Three components, three failure surfaces.

```text
                         pretraining corpus (unlabeled)
                                    |
                                    v
  slide --> tiles --> [ PATCH ENCODER ] --> N x D matrix --> [ AGGREGATOR ] --> slide score
              |              ^                                     ^
              |              |                                     |
              |        fixed or adapted                     trained on slide labels
              |              |                                     |
              +---- same encoder must be used at train and serve --+
                                                                   |
                                                                   v
                                                     threshold --> clinical action
                                                                   |
                                    prevalence, calibration, site ---+
```

### 3.1 Two models, two budgets

The encoder is expensive to train and cheap to reuse. The aggregator is cheap to train and is
where most task adaptation actually happens. A linear probe or an attention-MIL head has on the
order of 10^5 to 10^7 parameters against the encoder's 10^8 to 10^9. When a team says "we
fine-tuned the foundation model", ask which part moved. The answer determines the compute budget,
the overfitting risk, and the licence obligations.

### 3.2 The benchmark is a third model

A reported result is a joint property of the encoder, the aggregator, and the protocol: the split,
the preprocessing, the label definition, and the metric. Change any one and the number changes.
This is why two papers can report different AUROCs for the same encoder on the same public dataset
without either being wrong. It is also why a leaderboard position is not a property you can
inherit by downloading weights.

### 3.3 Where distribution shift enters

Shift enters at four points, and each has a different fix:

| Entry point | What changes | Symptom | Fix |
|---|---|---|---|
| Scanner | Colour response, focus, noise | Site classifier predicts site from embeddings | Stain normalisation, per-site validation |
| Stain protocol | Haematoxylin and eosin balance | Colour histogram shift; probe weights drift | Macenko or Vahadane normalisation to a fixed reference |
| Case mix | Prevalence, specimen type, stage | Calibration off; PPV wrong even at fixed sensitivity | Recalibrate; report prevalence-adjusted metrics |
| Label practice | What counts as positive at each site | Label noise correlated with site | Adjudicated subset; per-site label audit |

The first two are visible in the pixels. The second two are invisible in the pixels and are the
ones that most often invalidate a multi-site result.

### 3.4 The evidence ladder

Different adaptation strategies establish different things, and the strongest performance is not
always the strongest evidence.

```text
zero-shot ......... establishes that the pretraining aligned images and concepts
linear probe ...... establishes that the frozen representation is linearly separable for the task
PEFT / LoRA ....... establishes that a small adaptation improves the representation
full fine-tune .... establishes that the representation can be reshaped, and risks erasing it
external test ..... establishes that any of the above survives a new site
prospective use ... establishes clinical utility, and is the only rung that supports a clinical claim
```

A paper that reports only an internal linear probe has climbed one rung. A deployment claim needs
the last two.

### 3.5 Why MIL is the right frame and a dangerous one

MIL is right because the supervision really is at the bag level. It is dangerous because it lets
the aggregator choose which instances matter, and the aggregator is trained on a label that does
not constrain the choice. Attention weights are therefore a hypothesis about where the evidence
is, not an explanation. They become evidence only when checked against annotation on a sample the
model did not train on.

> **Note:** "Interpretable attention" is a claim about a visualisation, not a guarantee about
> causal contribution. Treat an attention map as a debugging aid until it has been validated
> against pathologist annotation.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Foundation model | An encoder pretrained on a broad corpus and adapted to downstream tasks | Used to mean any pretrained network, including ImageNet models | Model class |
| Encoder | The network mapping a patch to a fixed-length vector | Confused with the full slide-level model | Model component |
| Embedding | The vector produced by the encoder for one patch | Used loosely for the slide-level vector as well | Float vector, dimension D |
| Patch | A tile selected for encoding, usually 224 to 512 pixels | Used interchangeably with tile | px |
| Tile | A stored compressed square read from the slide | Used interchangeably with patch | px |
| Magnification | Objective power, such as 20x or 40x | Confused with resolution; use microns per pixel | x |
| Microns per pixel | Physical pixel size at the analysis level | Omitted from model cards | um/px |
| Linear probe | A logistic regression or linear classifier trained on frozen embeddings | Called fine-tuning | Method |
| Fine-tuning | Updating some or all encoder weights on the task | Assumed to beat a probe; it often does not on small cohorts | Method |
| PEFT | Parameter-efficient fine-tuning, such as LoRA or adapters | Treated as equivalent to full fine-tuning | Method family |
| LoRA | Low-rank adaptation; trains small matrices added to frozen weights | Called a probe | Method |
| Zero-shot | Inference without any task-specific training | Confused with few-shot | Protocol |
| Few-shot | Training on a small number of labelled examples per class | Confused with zero-shot | Protocol |
| Multiple instance learning (MIL) | Learning from bags of instances with bag-level labels | Used to mean any slide-level model | Framework |
| Bag | The set of patch embeddings for one slide | Called a batch | Set |
| Attention pooling | Weighting instances by a learned score before pooling | Called interpretable without validation | Aggregation |
| ABMIL | Attention-based MIL with a gated attention network | Used as a synonym for all MIL | Architecture |
| CLAM | Clustering-constrained attention MIL, which adds instance-level clustering | Treated as a general improvement | Architecture |
| TransMIL | A transformer aggregator over patch embeddings | Assumed to need no positional encoding | Architecture |
| Slide-level label | A diagnosis attached to the specimen, not to a region | Treated as a patch label | Label |
| Weak supervision | Learning from labels that are coarser than the prediction unit | Confused with noisy labels | Setting |
| Contamination | Test data present in pretraining or in training | Reduced to exact duplicates only | Defect |
| Near-duplicate | A different image of the same tissue or patient | Missed by file hashing | Defect |
| Site shift | Distribution change across hospitals or scanners | Confused with covariate shift in general | Phenomenon |
| Stain shift | Colour change from staining protocol differences | Assumed fixed by normalisation | Phenomenon |
| External validation | Evaluation on data from a site not used in development | Called "test set" when it is internal | Protocol |
| AUROC | Area under the receiver operating characteristic curve; threshold-free discrimination | Interpreted as clinical utility | 0 to 1 |
| AUPRC | Area under the precision-recall curve; sensitive to prevalence | Reported without prevalence | 0 to 1 |
| Calibration | Agreement between predicted probability and observed frequency | Confused with discrimination | Slope, intercept |
| PPV / NPV | Positive and negative predictive value at a given prevalence | Computed from sensitivity and specificity alone | Proportion |
| Prevalence | Share of positives in the population | Assumed equal to the cohort | Proportion |
| Shortcut learning | Reliance on a correlate of the label rather than the cause | Called overfitting | Defect |
| Model card | A document describing training data, intended use, and limits | Treated as a validation report | Document |
| PCCP | Predetermined change control plan; pre-authorised model updates | Assumed to permit any retraining | Regulatory |

## 5. The mechanics, step by step

This procedure takes a slide-level question and a cohort, and produces an adapted model with an
evidence package that states what it does and does not establish. It assumes Topic 11's pipeline
has produced a manifest and embeddings, or that embeddings can be produced.

Steps 1 to 3 are about the claim. Steps 4 to 8 build the data and the baseline. Steps 9 to 13
adapt and aggregate. Steps 14 to 18 evaluate and decide.

### Step 1. Define the slide-level task and the unit

**Input:** The clinical question and the available labels.

**Output:** A statement of the prediction unit (slide, block, or patient), the label definition,
and the population.

**Rationale:** The unit determines the split, the metric, and the claim. A model that predicts a
patient-level outcome from one slide is not the same model as one that predicts a slide-level
finding, even when the labels look identical. Mixing units is the most common source of leakage in
pathology machine learning.

**Branch points:**
- Slide-level finding, such as metastasis present: unit is the slide.
- Patient-level outcome, such as survival: unit is the patient, and multiple slides per patient
  must be aggregated before evaluation.
- Specimen-level diagnosis, such as Gleason grade: unit is the specimen, and the label may be
  derived from a report rather than a structured field.

**Edge case:** A patient with five lymph nodes contributes five slides and one outcome. Splitting
by slide puts the same patient on both sides of the split. Split by patient.

### Step 2. Fix the claim and the intended use

**Input:** The task statement and the deployment context.

**Output:** One sentence naming the population, the input, the output, and the decision the output
supports.

**Rationale:** The claim sets the evidence bar. "Prioritise cases for review" needs discrimination
and a workflow study. "Rule out metastasis without pathologist review" needs calibration,
sensitivity at a clinically chosen operating point, and a reader study. Both are legitimate; they
are not the same evaluation.

**Branch points:**
- Triage or prioritisation: ranking quality and time-to-review matter.
- Decision support with pathologist override: calibration and localisation matter.
- Autonomous diagnosis: sensitivity, specificity, subgroup performance, and a regulatory pathway.

**Edge case:** A claim that quietly changes between development and deployment, from "assists" to
"detects", invalidates the evidence package without changing a line of code.

### Step 3. Audit the pretraining corpus

**Input:** The model card, the paper, and any released corpus statistics.

**Output:** A provenance statement: institutions, slide count, tissue types, stain types,
scanner population, dates, and known exclusions.

**Rationale:** The corpus determines both the representation and the contamination risk. A model
trained on slides from the same hospital network as the evaluation cohort may have seen the
patients. A model trained only on resections may transfer poorly to biopsies. Neither fact is
visible in the AUROC.

**Branch points:**
- Corpus public and documented: check institution overlap with your cohort.
- Corpus private but described: ask for the exclusion list and the date range.
- Corpus undisclosed: assume overlap is possible and use a temporally held-out test set.

**Edge case:** Institutions can overlap even when the model was trained on public data, because
public datasets such as The Cancer Genome Atlas (TCGA) contain slides from many hospitals.

### Step 4. Fix the input contract

**Input:** The model's expected patch size, microns per pixel, colour normalisation, and channel
order.

**Output:** A specification the embedding pipeline implements exactly.

**Rationale:** An encoder trained at 0.5 um per pixel and applied at 0.25 um per pixel sees a
different object scale. The error is silent: the embeddings remain finite and the classifier still
runs. The contract must be written down and asserted at load time, not inferred from a default.

**Branch points:**
- Fixed scale and patch size: implement exactly, with an assertion.
- Multi-scale model: follow the model's own preprocessing, including any concatenation order.
- Model expects a specific normalisation: apply it inside the embedding job, not in the aggregator.

**Edge case:** A 224-pixel patch at 0.5 um per pixel and a 448-pixel patch at 1.0 um per pixel cover
the same physical area but are not interchangeable. The encoder was trained on one of them.

### Step 5. Build the tile manifest

**Input:** The slides and the tissue mask from the ingestion pipeline.

**Output:** A manifest of patches per slide, with coordinates and the slide content hash.

**Rationale:** The manifest is the join between pixels and embeddings. Without coordinates, an
attention map cannot be rendered and an error cannot be traced to a region. Without a content
hash, a re-exported slide silently produces a second set of embeddings under the same identifier.

**Branch points:**
- Cached embeddings exist and match the contract: reuse.
- Contract changed: re-embed and version the output.
- Manifest missing coordinates: regenerate before training, not after.

**Edge case:** A cohort assembled from several exports may contain the same slide twice under
different filenames. Hash the pixel data, not the path.

### Step 6. Cache embeddings

**Input:** The manifest and the encoder.

**Output:** One float16 matrix per slide, keyed by slide hash, with a row order that matches the
manifest.

**Rationale:** Embedding once and training many times is the main economic argument for
foundation models. A linear probe over cached embeddings trains in seconds; recomputing embeddings
for every experiment turns a morning into a week. The row order must be stored, or attention
indices become meaningless.

**Branch points:**
- Single encoder version: one cache.
- Several encoders compared: one cache per encoder, never mixed.
- Encoder updated: new cache namespace; do not overwrite.

**Edge case:** Float16 storage is usually fine for probing but should be ablated for tasks with
subtle texture differences. The ablation is cheap because the cache already exists.

### Step 7. Split by patient and site

**Input:** The cohort with patient identifiers and site labels.

**Output:** Train, validation, and test partitions, plus a leave-one-site-out partition for stress
testing.

**Rationale:** A random slide-level split leaks patients and sites. The model can learn a site
signature that correlates with the label, and the test set shares it. Patient-level splitting
removes the first leak; site-level splitting exposes the second. Both are needed, and they answer
different questions.

**Branch points:**
- Single site: patient-level split; report that external validity is untested.
- Multiple sites: patient-level split, and hold out at least one site entirely.
- Temporal cohort: split by date, because scanner and protocol drift over time.

**Edge case:** A site with 3 percent of the data will produce an unstable site-level estimate.
Report the interval and do not over-interpret a single small site.

### Step 8. Build the task-specific baseline

**Input:** The same splits and the same labels.

**Output:** A baseline model, such as a linear probe on ImageNet features or a small CNN trained
on tiles.

**Rationale:** A foundation model is only useful if it beats the alternative at the same label
budget. Without a baseline, a result is a demonstration, not a comparison. The baseline also
calibrates expectations: on some tasks, a well-tuned classical model is within a point of a
foundation model.

**Branch points:**
- Small cohort: baseline is a linear model on handcrafted or ImageNet features.
- Large cohort: baseline is a task-specific network trained end to end.
- No baseline possible: state that the result is descriptive.

**Edge case:** A baseline trained with different preprocessing is not a fair comparison. Use the
same manifest and the same splits.

### Step 9. Run the linear probe

**Input:** Cached embeddings, slide labels, and the training split.

**Output:** A fitted classifier, its validation performance, and its test performance.

**Rationale:** The linear probe is the cheapest test of whether the representation contains the
signal. It is also the most defensible: with a frozen encoder, the result cannot be attributed to
task-specific representation learning. Standardise features, use class weights for imbalance,
and select the regularisation strength on validation only.

**Branch points:**
- Probe performance near the baseline: the representation does not separate this task; do not
  fine-tune hoping to fix it without a reason.
- Probe clearly above baseline: the representation carries signal; consider whether adaptation is
  worth the cost.
- Probe unstable across folds: the cohort is too small for the claim; widen the interval.

**Edge case:** Fitting the feature standardisation on the full dataset leaks test statistics. Fit
on train only.

### Step 10. Decide whether to adapt further

**Input:** Probe performance, label budget, compute budget, and the claim.

**Output:** A decision: keep the probe, use PEFT, fine-tune the encoder, or stop.

**Rationale:** Fine-tuning can improve performance and can also destroy the representation that
made the model work. The published evidence is genuinely mixed: some large multi-dataset
benchmarks find parameter-efficient methods more reliable than linear probes, while small-cohort
studies with external validation find that fine-tuning overfits and that a probe generalises
better. The decision must follow from your label budget and your validation design, not from a
headline result.

| Situation | Recommended | Reason |
|---|---|---|
| Fewer than about 500 labelled slides, single site | Linear probe | Fine-tuning overfits; the probe is the honest test |
| 500 to 5,000 slides, multi-site validation available | PEFT, with a frozen-encoder control | Captures task structure at low parameter cost |
| More than 5,000 slides, external test available | Full fine-tune, compared against PEFT | Enough data to adapt without erasing the representation |
| Claim requires localisation | Keep the probe, invest in the aggregator | Encoder adaptation does not fix attention |
| Probe already meets the claim | Stop | Additional adaptation adds risk without benefit |

**Edge case:** Always train a frozen-encoder control alongside any fine-tune. If the fine-tune
does not beat the control on the external test, the adaptation is not justified.

### Step 11. Fine-tune or apply PEFT with controls

**Input:** The chosen strategy, the training split, and a validation split.

**Output:** Adapted weights and a comparison against the frozen control.

**Rationale:** Fine-tuning a large encoder on a small cohort is an overfitting machine. Mitigations
are standard: a low learning rate, layer-wise learning-rate decay, early stopping on validation,
and augmentation that respects histology. PEFT reduces the number of trainable parameters and
therefore the overfitting surface, at the cost of some capacity.

**Branch points:**
- Loss diverges: lower the learning rate before changing anything else.
- Validation improves while test degrades: stop and report the gap.
- PEFT matches full fine-tuning: prefer PEFT for cost and for a smaller change surface.

**Edge case:** Fine-tuning changes the embedding space, so a cache built from the original encoder
is no longer valid. Version the embeddings with the adapted weights.

### Step 12. Choose the aggregator

**Input:** The per-slide embedding matrices and the label type.

**Output:** An aggregation architecture and its training configuration.

**Rationale:** The aggregator is where slide-level supervision is applied. Mean pooling dilutes a
focal finding across thousands of patches; max pooling is dominated by a single artefact;
attention learns a weighting but needs enough slides to train. The choice should follow from the
expected spatial extent of the finding.

Gated attention, the standard formulation, computes a weight per patch and pools with it:

```text
a_i = exp(w^T ( tanh(V h_i) * sigmoid(U h_i) )) / sum_j exp(w^T ( tanh(V h_j) * sigmoid(U h_j) ))
z   = sum_i a_i h_i
```

The gate suppresses uninformative patches, and the softmax normalises across the bag. `z` is the
slide vector the classifier sees; the weights sum to one whatever the bag size.

| Finding extent | Aggregator | Reason |
|---|---|---|
| Diffuse, whole slide | Mean pooling, then linear | Every region is informative; attention adds variance |
| Focal, small | Attention MIL | Learns to up-weight rare informative patches |
| Focal, with instance labels available | Attention MIL plus clustering (CLAM) | Instance-level clustering sharpens the attention |
| Very long context, whole-slide model | Slide-level transformer (for example a LongNet-style encoder) | Models interactions between distant regions |
| Small cohort, few slides | Mean or max pooling, with a strong regularised head | Attention has too many parameters to train |

### Step 13. Train the aggregator with imbalance handled correctly

**Input:** The aggregator, the embeddings, and the splits.

**Output:** A trained aggregator and its validation curve.

**Rationale:** Positive prevalence is often below 20 percent, so accuracy is misleading and
AUROC can look strong while precision is poor. Use AUPRC and precision at a fixed recall for
imbalance. Apply class weights in the loss, and never resample before splitting: duplicating
positives across the split boundary leaks the same patient.

**Branch points:**
- Prevalence below 10 percent: report AUPRC alongside AUROC and state the baseline prevalence.
- Bag sizes vary by an order of magnitude: check calibration by bag size.
- Training unstable across seeds: report the seed distribution, not the best seed.

**Edge case:** A model trained on one positive patch per positive slide learns a patch detector,
not a slide classifier. Check whether performance collapses when the bag is truncated to its
highest-attention patch.

### Step 14. Calibrate

**Input:** Validation predictions and labels.

**Output:** A calibration map, such as Platt scaling or temperature scaling, fitted on validation.

**Rationale:** A clinical threshold is a probability statement. A model can rank well and still be
badly calibrated, especially after fine-tuning or class weighting. Calibration must be fitted on
data that was not used to fit the classifier, or it will be optimistic.

**Branch points:**
- Calibration slope near 1 and intercept near 0: proceed.
- Overconfident: temperature scaling.
- Calibration differs by site: recalibrate per site or report the limitation.

**Edge case:** Calibration is prevalence-dependent. A model calibrated in a 12 percent prevalence
cohort is not calibrated in a 30 percent one, even if discrimination is unchanged.

### Step 15. Evaluate discrimination and calibration with intervals

**Input:** Test predictions.

**Output:** AUROC, AUPRC, sensitivity and specificity at the operating point, calibration curve,
and bootstrap intervals clustered by patient.

**Rationale:** Point estimates without intervals invite over-reading, particularly for small
sites. Clustering the bootstrap by patient respects the dependence created by multiple slides per
patient. Reporting AUPRC and the prevalence prevents a strong AUROC from being read as strong
precision.

**Branch points:**
- Interval wide and overlapping the baseline: the result is inconclusive.
- Interval excludes the baseline but the site intervals disagree: report the heterogeneity.
- Calibration poor despite good discrimination: the threshold is not usable as a probability.

**Edge case:** A test set with 30 positives cannot support a precise sensitivity estimate. Report
the interval, and say so in the abstract.

### Step 16. Stress-test site shift

**Input:** The trained model and the held-out site.

**Output:** Site-stratified metrics and a leave-one-site-out result.

**Rationale:** The external site is the closest available proxy for deployment. A pooled metric
can hide a collapse at one site, because the other sites dominate the average. The stress test is
also the only way to see whether stain normalisation, site-balanced training, or domain
generalisation methods actually help.

**Branch points:**
- Small drop at the held-out site: the representation transfers.
- Large drop: investigate stain, scanner, and label practice in that order.
- Drop concentrated in one subgroup: report it as a subgroup failure, not noise.

**Edge case:** A held-out site that shares a scanner model with the training sites is not an
external test in the sense that matters. Record the scanner inventory.

### Step 17. Audit contamination and near-duplicates

**Input:** The test set, the training set, and any accessible pretraining corpus.

**Output:** A contamination report with exact duplicates, near-duplicates, and patient overlaps.

**Rationale:** Contamination inflates results and is invisible in the metric. File hashes catch
exact copies; perceptual hashes and embedding nearest neighbours catch re-encodes; patient
identifier joins catch different slides from the same patient. Each catches a different failure.

| Finding | Action | Reason |
|---|---|---|
| Exact duplicate between train and test | Remove from test | Direct leakage |
| Near-duplicate, different scan of the same block | Remove from test or group with train | Tissue overlap leaks morphology |
| Same patient, different slide | Group by patient before splitting | Patient-level leakage |
| Overlap with pretraining corpus | Report; use a temporally held-out test set | The encoder may have seen the patient |
| Unknown corpus | Assume overlap possible; state the limitation | Cannot be excluded |

**Edge case:** Near-duplicate detection by embedding distance will also flag biologically similar
but independent cases. Review the threshold on a labelled sample of known duplicates.

### Step 18. Interpret, decide, and write the claim

**Input:** Metrics, site results, attention maps, and the original claim.

**Output:** A decision to deploy, iterate, or stop, and a written claim that matches the evidence.

**Rationale:** The claim is the deliverable. A model that reaches 0.90 AUROC internally and 0.79
externally supports a claim about triage at sites like the training sites, not a claim about
autonomous detection anywhere. Writing the claim down before the final analysis prevents the
evidence from being stretched to fit a preferred conclusion.

**Branch points:**
- Claim supported at the target site and population: proceed to workflow validation.
- Claim supported only at some sites: narrow the claim or add a site-specific recalibration.
- Claim not supported: report the negative result; it is more useful than a qualified positive.

**Edge case:** Attention maps that highlight background are a finding about the model, not a
presentation problem. Do not crop them out of the figure.

## 6. Worked example

A three-site hospital network wants to detect nodal metastasis in colorectal resections from
H&E slides. The label is the pathology report: any node positive for metastatic carcinoma. The
proposed product is a worklist prioritisation tool, not an autonomous diagnosis. The team has a
foundation model's embeddings, a linear probe, and a plan.

### 6.1 The cohort and the arithmetic

| Quantity | Value | Source |
|---|---:|---|
| Slides | 1,800 | Cohort inventory |
| Patients | 1,540 | Identifier join |
| Site A slides / patients | 1,000 / 850 | Site A |
| Site B slides / patients | 500 / 430 | Site B |
| Site C slides / patients | 300 / 260 | Site C |
| Positive slides | 220 | Report extraction |
| Prevalence | 12.2 percent | 220 / 1,800 |

Patches and embeddings, using the per-slide figures from Topic 11:

```text
patches       = 1,800 slides x 4,873 patches/slide = 8,771,400 patches
embeddings    = 1,800 x 10.3 MB = 18,540 MB = 18.5 GB
```

Eighteen gigabytes is small enough to hold on one machine and small enough to copy for every
experiment. This is the practical meaning of the foundation-model economics: the expensive part
is the encoder pass, and it is done once.

The contrast with fine-tuning is stark:

```text
linear probe on cached embeddings : 1,800 x 1,024 features, seconds to minutes
full fine-tune, forward+backward  : 8.77e6 patches / 150 patches/s = 58,500 s per epoch
                                    58,500 s = 16.2 h per epoch
                                    8 epochs = 130 GPU-hours, before tuning
```

The probe trains in minutes and the fine-tune takes most of a week of accelerator time. That does
not make the probe better, but it sets the burden of proof for the fine-tune: it must earn 130
GPU-hours with evidence that the probe cannot produce.

### 6.2 The split

The team splits by patient, not by slide, and holds out site C entirely. The resulting partitions:

| Partition | Slides | Patients | Positives |
|---|---:|---:|---:|
| Train (A and B) | 1,080 | 920 | 145 |
| Validation (A and B) | 120 | 100 | 18 |
| Test in-domain (A and B) | 300 | 260 | 35 |
| External (C) | 300 | 260 | 22 |

The validation set has 18 positives, which is enough to choose a regularisation strength and not
enough to fit a calibration curve with confidence. That limitation is recorded now and reported
later.

### 6.3 The probe result

The probe is logistic regression on standardised embeddings, with class weights and the
regularisation strength chosen on validation. The test results:

| Evaluation | AUROC | 95 percent interval | AUPRC |
|---|---:|---|---:|
| Pooled in-domain | 0.91 | 0.89 to 0.93 | 0.72 |
| Site A | 0.93 | 0.90 to 0.95 | 0.76 |
| Site B | 0.91 | 0.88 to 0.94 | 0.69 |
| Site C, external | 0.83 | 0.74 to 0.90 | 0.55 |

The pooled number is not a summary of the sites; it is a weighted average dominated by site A.
The external site is eight points lower, and its interval is wide because it has 22 positives.
An honest abstract says "0.91 in-domain, 0.83 at an external site", not "0.91".

### 6.4 From AUROC to a clinical number

Suppose the team proposes an operating point with sensitivity 0.90 and specificity 0.85, chosen
on validation. At the cohort prevalence of 12.2 percent, for 1,000 slides:

```text
true positives  = 0.90 x 122          = 109.8
false positives = 0.15 x 878          = 131.7
PPV             = 109.8 / (109.8 + 131.7) = 0.455
true negatives  = 0.85 x 878          = 746.3
false negatives = 0.10 x 122          = 12.2
NPV             = 746.3 / (746.3 + 12.2)  = 0.984
```

At 12.2 percent prevalence, a positive flag is right less than half the time. For a worklist that
is acceptable: a pathologist reviews 241 slides to find 110 positives, and the ordering still
saves time if the model ranks well. For a rule-out claim it is not acceptable, because 12 false
negatives per 1,000 slides would be missed.

Now change the prevalence to 30 percent, as it would be in a selected high-risk population:

```text
PPV = 0.90 x 0.30 / (0.90 x 0.30 + 0.15 x 0.70) = 0.270 / 0.375 = 0.72
NPV = 0.85 x 0.70 / (0.85 x 0.70 + 0.10 x 0.30) = 0.595 / 0.625 = 0.952
```

The same model, at the same sensitivity and specificity, moves from 46 percent to 72 percent
positive predictive value. Nothing about the model changed. This is why a deployment claim must
name the population and why a benchmark AUROC cannot be translated into a clinical number without
one.

### 6.5 Diagnostic 1: the site is in the embeddings

The team trains a logistic regression to predict the site from the mean embedding of each slide,
using the same splits. Nearest-centroid accuracy is 0.74 against a chance level of 0.33. The embeddings
encode the site strongly enough to identify it from a single 1,024-dimensional vector.

This is a diagnostic, not a proof. A site classifier can be accurate because the case mix differs,
not only because of scanner or stain. The follow-up distinguishes the two: re-run the site
classifier after stain normalisation. If accuracy falls to 0.45, colour was the dominant signal.
If it stays near 0.74, the signal is morphological or case-mix related, and normalisation will not
remove it.

Either way, the practical consequence is the same: a random split would have let the probe use
site as a shortcut, and the pooled AUROC would have overstated external performance. The
patient-level and site-level splits are what prevented that.

### 6.6 Diagnostic 2: where the attention goes

The team renders the top 1 percent of attention weights on 50 test slides and asks a pathologist
to score each highlighted patch as tumour, other tissue, or background and artefact. The result:

| Category of top-weighted patch | Share |
|---|---:|
| Tumour | 44 percent |
| Lymphoid tissue | 12 percent |
| Fat | 21 percent |
| Ink or fold | 17 percent |
| Glass or background | 6 percent |

Thirty-eight percent of the highest-weight patches are not tumour, and the fat and ink share is
large enough to suggest a shortcut: in this cohort, node dissection correlates with the number of
nodes submitted, and ink marks the resection margin. A model that attends to fat and ink can
predict the label without finding tumour, because the label is correlated with the surgical
procedure.

The check that distinguishes a shortcut from an acceptable distribution is a bag-truncation test.
Keep only the highest-attention patch per slide and re-evaluate. If AUROC stays near 0.85, the
model is a patch detector and its slide-level score is really a max over patches. If it falls to
0.60, the model is using the bag as a whole, which is not necessarily wrong but changes the story.

### 6.7 A varied assumption: the external site

Suppose site C is included in training instead of held out. The pooled AUROC rises to 0.93, site C
rises from 0.84 to 0.92, and the team concludes that the model generalises. The conclusion is
wrong in a specific way: nothing external was tested. The 0.92 at site C measures interpolation to
a site whose scanner and stain were present in training.

The opposite variation is more informative. Exclude site B as well, train on A only, and evaluate
on B and C:

```text
trained on A only, tested on B : 0.86  (was 0.91 with B in training)
trained on A only, tested on C : 0.78  (was 0.83 with C in training)
```

The drop from 0.83 to 0.78 at site C is the cost of not having seen the site. That number, not the
pooled 0.91, is the closest estimate of what a new hospital should expect. It is also the number
that a procurement conversation should start from.

### 6.8 What the case establishes

The case establishes that a frozen representation carries signal for this task, that the signal
degrades at an unseen site, that the model attends partly to non-tumour tissue, and that the
clinical reading of the score depends on prevalence. It does not establish that the model is safe
for triage, because no workflow study was done, and it does not establish performance at any site
outside the three. Those are the next two pieces of evidence, and they are not obtained by
training a bigger aggregator.

## 7. Code, query, or calculation

The script computes the four quantities that decide whether a foundation-model result supports a
claim: a rank-based AUROC, a bootstrap interval, predictive values at two prevalences, and a
label-efficiency comparison between a probe and a fine-tune. It also demonstrates the site
diagnostic on synthetic embeddings.

The scores are simulated, not measured. That is deliberate: the arithmetic is the lesson, and a
simulation makes the assumptions visible. The AUROC implementation is the same one you would use
on real scores, and the bootstrap is patient-agnostic here only because the simulated units are
independent.

```python
"""pathology_fm_eval.py -- evaluation arithmetic for a pathology foundation model.

Standard library only. Deterministic. The score distributions and the learning
curves are simulated; the statistics computed from them are the real subject.
Run: python pathology_fm_eval.py
"""

import math
import random

SITES = {"A": (1000, 2.15), "B": (500, 1.75), "C": (300, 1.55)}  # slides, separation
PREVALENCE = 0.122
SENS, SPEC = 0.90, 0.85
SEED = 20261027


def auroc(scores, labels):
    """Rank-based AUROC with tie averaging."""
    pairs = sorted(zip(scores, labels))
    ordered = [y for _, y in pairs]
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(ordered)
    neg = len(ordered) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rsum = sum(r for r, y in zip(ranks, ordered) if y == 1)
    return (rsum - pos * (pos + 1) / 2) / (pos * neg)


def bootstrap_ci(scores, labels, n_boot=2000, alpha=0.05, seed=SEED):
    rng = random.Random(seed)
    n = len(scores)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        s = [scores[i] for i in idx]
        y = [labels[i] for i in idx]
        stats.append(auroc(s, y))
    stats.sort()
    lo = stats[int(alpha / 2 * len(stats))]
    hi = stats[int((1 - alpha / 2) * len(stats)) - 1]
    return lo, hi


def predictive_values(prevalence, sens=SENS, spec=SPEC):
    tp = sens * prevalence
    fp = (1 - spec) * (1 - prevalence)
    tn = spec * (1 - prevalence)
    fn = (1 - sens) * prevalence
    return tp / (tp + fp), tn / (tn + fn)


def simulate(n_slides, dprime, rng, prevalence=PREVALENCE):
    scores, labels = [], []
    for _ in range(n_slides):
        y = 1 if rng.random() < prevalence else 0
        mu = dprime / 2 if y else -dprime / 2
        scores.append(rng.gauss(mu, 1.0))
        labels.append(y)
    return scores, labels


def site_diagnostic(rng, dims=50, per_site=300, offset=0.22):
    """Nearest-centroid site classification on synthetic mean embeddings."""
    centroids, points = {}, []
    for name in SITES:
        shift = [rng.gauss(0, offset) for _ in range(dims)]
        centroids[name] = shift
    for name, shift in centroids.items():
        for _ in range(per_site):
            points.append((name, [s + rng.gauss(0, 1) for s in shift]))
    by_site = {n: [] for n in SITES}
    for name, vec in points:
        by_site[name].append(vec)
    means = {n: [sum(col) / len(v) for col in zip(*v)] for n, v in by_site.items()}
    correct, total = 0, 0
    for name, vec in points:
        best = min(means, key=lambda k: sum((a - b) ** 2 for a, b in zip(vec, means[k])))
        correct += best == name
        total += 1
    return correct / total


def main():
    rng = random.Random(SEED)
    all_s, all_y, by_site = [], [], {}
    for name, (n, dprime) in SITES.items():
        s, y = simulate(n, dprime, rng)
        by_site[name] = (s, y)
        all_s += s
        all_y += y

    print("== Discrimination ==")
    pool = auroc(all_s, all_y)
    lo, hi = bootstrap_ci(all_s, all_y)
    print(f"pooled          AUROC {pool:.3f}  95% CI {lo:.3f}-{hi:.3f}")
    for name, (s, y) in by_site.items():
        a = auroc(s, y)
        l, h = bootstrap_ci(s, y)
        print(f"site {name}  n={len(s):>4}  AUROC {a:.3f}  95% CI {l:.3f}-{h:.3f}  positives={sum(y)}")

    print("\n== Predictive values at fixed sensitivity and specificity ==")
    for prev in (0.122, 0.30):
        ppv, npv = predictive_values(prev)
        print(f"prevalence {prev:.3f}: PPV {ppv:.3f}  NPV {npv:.3f}")

    print("\n== Label efficiency (simulated curves) ==")
    print(f"{'labels':>7} {'probe':>7} {'fine-tune':>10} {'winner':>10}")
    for n in (100, 300, 1000, 3000, 7000, 15000):
        probe = 0.70 + 0.22 * (1 - math.exp(-n / 400))
        ft = 0.62 + 0.34 * (1 - math.exp(-n / 3000))
        print(f"{n:>7} {probe:>7.3f} {ft:>10.3f} {'probe' if probe > ft else 'fine-tune':>10}")

    print("\n== Site signal in mean embeddings ==")
    acc = site_diagnostic(random.Random(SEED))
    print(f"nearest-centroid site accuracy {acc:.3f} (chance {1/len(SITES):.3f})")

    print("\n== Cost of one fine-tuning epoch ==")
    patches = 1800 * 4873
    per_epoch = patches / 150
    print(f"patches {patches:,}; at 150 patches/s = {per_epoch/3600:.1f} h/epoch; "
          f"8 epochs = {8*per_epoch/3600:.0f} GPU-hours")


if __name__ == "__main__":
    main()
```

Output on the author's machine, Python 3.11:

```text
== Discrimination ==
pooled          AUROC 0.910  95% CI 0.889-0.929
site A  n=1000  AUROC 0.928  95% CI 0.899-0.953  positives=113
site B  n= 500  AUROC 0.912  95% CI 0.879-0.941  positives=75
site C  n= 300  AUROC 0.827  95% CI 0.742-0.903  positives=30

== Predictive values at fixed sensitivity and specificity ==
prevalence 0.122: PPV 0.455  NPV 0.984
prevalence 0.300: PPV 0.720  NPV 0.952

== Label efficiency (simulated curves) ==
 labels   probe  fine-tune     winner
    100   0.749      0.631      probe
    300   0.816      0.652      probe
   1000   0.902      0.716      probe
   3000   0.920      0.835      probe
   7000   0.920      0.927  fine-tune
  15000   0.920      0.958  fine-tune

== Site signal in mean embeddings ==
nearest-centroid site accuracy 0.743 (chance 0.333)

== Cost of one fine-tuning epoch ==
patches 8,771,400; at 150 patches/s = 16.2 h/epoch; 8 epochs = 130 GPU-hours
```

### 7.1 Reading the output

The pooled AUROC of 0.910 is higher than the site C value of 0.827 by 8 points, and the
intervals do not overlap much. That pattern, a strong pooled number with a weaker site, is the
signature of a result that needs site-stratified reporting. The script prints the interval
because a point estimate of 0.827 on 30 positives would otherwise look more precise than it is.

The label-efficiency table shows the crossover that the literature argues about. Below about
7,000 labels, the probe curve is above the fine-tune curve; above it, fine-tuning wins. The two
constants in each curve are assumptions, not measurements: the probe saturates at 0.92 and the
fine-tune at 0.96. Change them and the crossover moves. The method is the point, not the
crossover location.

The site diagnostic reports 0.743 against a chance level of 0.333. In the simulation the site
signal is injected deliberately, so the result only proves that the detector works. On real
embeddings, a value well above chance is a warning that the site is encoded, and the follow-up is
the normalisation test described in Section 6.5.

### 7.2 What the script does not do

It does not cluster the bootstrap by patient, which real data requires when patients contribute
more than one slide. It does not compute AUPRC, although the case reports it. It does not
implement a linear probe, a fine-tune, or an aggregator; those are library work, and the script is
about the evaluation arithmetic around them. And it does not model contamination, which cannot be
detected from scores at all: a contaminated test set simply produces better scores, and only a
separate audit finds it.

## 8. Data, standards, and regulatory context

A foundation-model project touches three kinds of obligation: the licence on the weights, the
reporting standard for the study, and the device regulation for the claim. They are independent,
and satisfying one does not satisfy the others.

### 8.1 Cohorts and their terms

| Dataset | Content | Access | What it is good for | What it is not |
|---|---|---|---|---|
| TCGA | About 1,100 slides across 30-plus cancer types, with molecular data | Open, with citation terms | Pan-cancer benchmarks, multimodal linkage | Small per cancer type; no scanner diversity control |
| Camelyon16 / 17 | Lymph node metastasis, with pixel annotations | Open | Localisation checks, MIL baselines | Two centres; metastasis only |
| PANDA | Prostate biopsies, Gleason grading | Competition terms | Grading at scale | One organ; label aggregation rules matter |
| EBRAINS | Brain tumour subtyping | Registration | Rare-class subtyping | Small; class definitions are specific |
| CPTAC | Clinical Proteomic Tumor Analysis Consortium cohorts with slides | Open, with terms | Molecular correlation | Modest slide counts |
| Institutional cohorts | Routine practice | Governance approval | The only data that matches your population | Access, retention, and consent constraints |

The practical point is that public benchmarks are for method comparison, not for deployment
evidence. A model that wins on TCGA has demonstrated transfer to TCGA, which is a convenience
sample assembled for molecular research, not a sample of your laboratory's work.

### 8.2 Weight licences

Released weights carry licences that range from permissive to research-only. UNI's weights are
released for academic research use; Virchow's are released for non-commercial research; other
releases vary. Before a model enters a product, the licence must be read by someone with
authority to accept it, and the answer recorded. A common failure is a research prototype built on
non-commercial weights that is then handed to a product team, which cannot lawfully ship it
without a separate agreement.

### 8.3 Reporting standards

| Standard | Applies to | What it requires | Status |
|---|---|---|---|
| TRIPOD+AI (2024) | Prediction model studies | Reporting of data, predictors, performance, fairness | Publication convention; journals enforce |
| CLAIM (2020) | Medical imaging AI | Checklist for data, model, evaluation | Publication convention |
| STARD (2015) | Diagnostic accuracy studies | Participant flow, reference standard, blinding | Publication convention |
| PROBAST | Risk of bias | Structured assessment of model studies | Appraisal tool |
| DECIDE-AI (2022) | Early clinical evaluation | Human factors, workflow, safety | Reporting guidance |

These standards do not certify a model. They make a study auditable, which is what a reviewer,
a regulator, and a hospital committee all need in different forms.

### 8.4 Device regulation

Software that analyses pathology images for a clinical decision may be a medical device. In the
United States, the relevant framework is software as a medical device, with 510(k), De Novo, or
premarket approval depending on risk and on whether a predicate exists. The Food and Drug
Administration's Good Machine Learning Practice principles (2021) and its guidance on
predetermined change control plans (PCCP) matter for foundation models in particular: a PCCP is
how a sponsor pre-specifies the kinds of updates that will not require a new submission. A model
whose encoder is periodically retrained without a PCCP cannot be updated in the field without
new authorisation.

The regulatory status of laboratory developed tests changed twice between 2024 and 2026. The 2024
final rule (89 FR 37286) would have phased out enforcement discretion; a federal court vacated
it in March 2025, and the agency published a conforming rule in September 2025 (90 FR 45134)
restoring the earlier position. As of October 2026 the pre-2024 position applies. Any summary that
still describes the phase-out schedule is out of date. This is educational context, not legal
advice.

### 8.5 Privacy and governance

Slides carry identifiers in pixels, metadata, and filenames. A foundation-model workflow adds a
new risk: embeddings are derived data that may still be linkable. An embedding is not a
de-identified image in any formal sense, and storing it outside the governed environment is a
governance decision, not an engineering convenience. Retention rules that apply to the slide
should be assumed to apply to the embedding unless a privacy review says otherwise.

What none of these instruments constrains: the choice of encoder, the aggregation method, the
operating threshold, or the definition of a positive case. Those are the scientific decisions, and
they are where the failure modes in Section 10 live.

## 9. Industry practice and case evidence

### 9.1 Where the models came from

Four releases define the public landscape, and each is best read as a paper with a corpus, not as
a product:

| Model | Reported corpus | Objective | Evaluated on, at release |
|---|---|---|---|
| UNI | About 100 million patches from 100,426 slides, 20 tissue types | DINOv2 self-distillation | 34 tasks including classification, retrieval, segmentation |
| CONCH | About 1.17 million image-caption pairs plus unlabeled tiles | Contrastive image-text (CoCa) | Zero-shot and supervised classification, retrieval, captioning |
| Virchow | About 1.49 million slides from 119,629 patients at one centre | DINOv2, ViT-H/14 | Pan-cancer detection, rare cancers, biomarker prediction |
| Prov-GigaPath | About 1.38 billion tiles from 171,189 slides | DINOv2 tile encoder plus a LongNet slide encoder | 26 tasks, including mutation prediction |
| CHIEF | 60,530 slides across 19 anatomical sites | Combined self-supervised and weakly supervised | Cancer detection, origin, molecular profiling, survival |

The numbers are the authors' reports. None of these corpora has been independently audited for
composition or for overlap with the public benchmarks used to evaluate them, and the evaluations
were designed by the same groups that built the models. That does not make the results wrong; it
sets their evidential weight.

### 9.2 What the benchmark literature now says

The 2025 to 2026 literature has moved from "does a foundation model beat a task-specific model"
to "under which protocol, and on which site". Three findings recur:

- A large multi-dataset benchmark of 19 foundation models across 31 tasks and 13 patient cohorts
  reported that data diversity in pretraining mattered more than raw corpus size, and that slide
  encoders added little over tile features inside MIL (Neidlinger et al., *Nature Biomedical
  Engineering*, 2025). This is a caution for anyone buying a slide-level model on corpus size.
- A benchmarking study of adaptation strategies found parameter-efficient fine-tuning the most
  consistent winner across patch classification tasks, and questioned linear probing as a default
  (Lee et al., *Computers in Biology and Medicine*, 2025).
- A small-cohort study with external validation found the opposite for UNI and Prov-GigaPath:
  fine-tuning barely beat linear probing in-domain, while linear probing beat it on external data (Enda et al.,
  *Pathology International*, 2026). The authors attribute this to overfitting on limited
  institutional data.

These results are not reconcilable into a single rule, and a document that presents one of them as
the consensus is misreading the field. The robust conclusion is that adaptation must be validated
locally, with a frozen-encoder control.

### 9.3 Robustness evidence

A separate line of work asks whether foundation-model representations encode medical centre
identity. The PathoROB benchmark reported that representations from a range of foundation models
allow the centre to be predicted from embeddings at well above chance, and that both full and
parameter-efficient fine-tuning could amplify reliance on spurious correlates. The practical
translation is the diagnostic in Section 6.5: measure site predictability directly rather than
assuming that a bigger pretraining corpus removes it.

### 9.4 What commercial products actually use

Most cleared or approved pathology AI products predate foundation models and use task-specific
convolutional networks trained on curated data. Paige Prostate received De Novo authorisation from
the Food and Drug Administration in 2021 for prostate cancer detection, and its evidence package was a clinical study, not
a benchmark AUROC. The pattern is consistent: products are cleared on a narrow intended use with a
reader study, while foundation models are published on broad benchmarks. A team that wants to ship
should expect to reproduce the former kind of evidence, not the latter.

### 9.5 Documented fact versus inference

| Statement | Status |
|---|---|
| UNI was pretrained on about 100 million patches from 100,426 slides | Documented in the paper |
| Foundation models improve label efficiency on many tasks | Documented across benchmarks, task-dependent |
| Linear probing is sufficient for most clinical tasks | Inference; contradicted by some studies |
| A model cleared by the Food and Drug Administration is safe at your site | Inference; clearance is tied to an intended use and a studied population |
| Site signatures persist after stain normalisation | Inference supported by robustness studies; test locally |
| Bigger corpora produce better clinical models | Not established; diversity appears to matter more |

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Patient leakage | Test AUROC far above validation | Random slide-level split | Inflated estimate that collapses on deployment | Split by patient before any preprocessing |
| Site shortcut | Site classifier accurate from embeddings | Site correlates with label in the cohort | Model learns hospital, not disease | Site-stratified split; stain normalisation; per-site validation |
| Attention on background | Top-weighted patches are fat, ink, or glass | Aggregator exploits correlates | Plausible heatmaps, wrong evidence | Annotation audit; bag-truncation test |
| Scale mismatch | Embeddings look normal but accuracy is low | Wrong microns per pixel at inference | Silent degradation | Assert the input contract at load |
| Contaminated pretraining | Test performance above external expectation | Test patients in the pretraining corpus | Invalid comparison | Temporal holdout; corpus audit |
| Prevalence ignored | PPV far below expectation in use | AUROC translated to a clinical claim | Wrong operational decision | Report PPV at the deployment prevalence |
| Uncalibrated probabilities | Reliability curve off the diagonal | Class weighting or fine-tuning | Threshold means nothing | Calibrate on held-out validation |
| Single-seed result | Large variance across seeds | Small cohort, unstable training | Lucky result reported as real | Report seed distribution |
| Aggregator overfits | Train AUROC 0.99, test 0.75 | Millions of parameters on hundreds of bags | No generalisation | Regularise; reduce capacity; pool simply |
| Label noise by site | Disagreement between site and adjudicated labels | Different reporting practice | Site becomes a label proxy | Adjudicated subset; per-site label audit |
| Leaked feature scaling | Probe better than expected | Standardisation fitted on all data | Optimistic estimate | Fit transforms on train only |
| Licence violation | Product team asks for weights | Research-only release | Cannot ship | Record licence at selection time |
| Unsupported zero-shot claim | Retrieval demo shown as diagnosis | Zero-shot evaluated on curated classes | Claim outruns evidence | Restrict claim to retrieval or triage |

Two of these are worth naming as systemic. Attention on background is not a bug in the
visualisation; it is the model telling you what it learned, and it will persist in every figure
until the training distribution changes. Prevalence ignorance is not a statistical nicety; it is
the difference between a useful worklist and a misleading probability.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Slide-level labels are accurate | Supervision depends on them | Report extraction errors; site-specific practice | Adjudicate a random sample of 100 |
| The cohort is representative of deployment | Justifies any generalisation claim | Tertiary referral centre, trial enrichment | Compare case mix to the target population |
| The held-out site is external enough | Proxy for deployment | Shared scanner or protocol | Record the scanner inventory per site |
| Patch embeddings capture diagnostic content | Whole approach depends on it | Finding smaller than a patch; mitoses | Test at a finer analysis scale |
| The aggregator learns biology | Interpretability claims | Shortcut correlates | Annotation audit; truncation test |
| Probe and fine-tune are comparable | Adaptation decision | Different preprocessing or splits | Same manifest, same splits, same seeds |
| Calibration transfers across prevalence | Thresholds are usable | Prevalence differs from development | Recalibrate; report prevalence |
| Public benchmarks measure transfer | Method selection | Convenience sampling, contamination | Add an external private test set |
| Weights licence permits the intended use | Legal deployment | Research-only terms | Legal review before the prototype ships |
| One slide per patient | Simplifies evaluation | Multiple blocks and recuts | Group by patient in splits and bootstrap |
| Attention weights indicate importance | Explanations to pathologists | Attention is a learned weighting, not causality | Validate against annotation |

The deepest limit is that none of this measures clinical utility. Utility requires a workflow
study: does the model change decisions, time, or outcomes, and does it do so without adding harm?
A foundation model with an excellent AUROC can fail a workflow study because it is slow, because
its alerts are not trusted, or because it duplicates a task the pathologist already does well.

## 12. Exercises

### Exercise 1. Write the provenance statement

Choose one released pathology foundation model and write a provenance statement of no more than
300 words: institutions, slide count, tissue types, stain types, scanner population, dates,
licence, and known exclusions.

**Deliverable:** The statement plus a list of the claims you could not verify.

**Acceptance criteria:** Every number is attributed to the paper or model card; unverifiable items
are marked as unverifiable rather than omitted; the licence is quoted, not paraphrased.

**Time:** 45 minutes.

### Exercise 2. Size the cohort arithmetic

A cohort has 2,400 slides from 2,000 patients across four sites. Each slide yields 3,800 patches
and each embedding is 1,024 float16 values plus 64 bytes of key.

**Deliverable:** Total patches, total embedding bytes, probe training matrix shape, and the
graphics processing unit (GPU) time for one fine-tuning epoch at 150 patches per second.

**Acceptance criteria:** 9,120,000 patches; about 19.4 GB; a 2,400 by 1,024 matrix; 16.9 hours per
epoch. Units shown at each step.

**Time:** 30 minutes.

### Exercise 3. Design the split

Design the train, validation, test, and external partitions for the cohort in Exercise 2, with
site D held out entirely.

**Deliverable:** A partition table with slide and patient counts and a one-paragraph justification.

**Acceptance criteria:** No patient appears in two partitions; the external partition is untouched
by any preprocessing fit; the table reports the positive count per partition; the plan states what
happens if a site is too small for a stable estimate.

**Time:** 45 minutes.

### Exercise 4. Compute attention by hand

A slide has four patches with attention logits 0.0, 1.0, 2.0, and 0.0. Compute the softmax weights
and the pooled vector if the four embeddings are (1,0), (0,1), (-1,0), and (0,-1).

**Deliverable:** The weights and the pooled vector, with the arithmetic.

**Acceptance criteria:** Weights are 0.087, 0.237, 0.643, 0.087 after rounding and sum to about 1;
the pooled vector is approximately (-0.469, 0.150); the answer notes that the third patch
dominates.

**Time:** 30 minutes.

### Exercise 5. Critique a benchmark claim

A press release states: "Our model achieves 0.97 AUROC on a public pathology benchmark, exceeding
pathologist performance." Write a critique identifying what the claim does and does not establish.

**Deliverable:** A 400-word critique with three specific questions for the authors.

**Acceptance criteria:** Distinguishes benchmark AUROC from clinical utility; notes the absence of
a reader study and of external validation; asks about contamination, split construction, and the
prevalence at which the comparison was made.

**Time:** 45 minutes.

### Exercise 6. Prevalence arithmetic

A model has sensitivity 0.85 and specificity 0.90. Compute PPV and NPV at prevalences of 5, 15,
and 40 percent.

**Deliverable:** A small table with the arithmetic for one row shown in full.

**Acceptance criteria:** PPV of approximately 0.31, 0.60, and 0.85; the answer states that NPV moves
in the opposite direction and explains why in one sentence.

**Time:** 30 minutes.

### Exercise 7. Design the site-shift stress test

Design a stress test for a model trained on sites A and B, evaluated on site C, where site C uses
a different scanner and a different staining protocol.

**Deliverable:** A test plan with the comparisons you will run and the decision rule for each.

**Acceptance criteria:** Includes a stain-normalised rerun, a site-classifier diagnostic, a
subgroup breakdown, and a pre-specified threshold for what counts as acceptable degradation.

**Time:** 60 minutes.

### Exercise 8. Write the claim

Using the evidence from Section 6, write the deployment claim in one sentence, and list the three
pieces of evidence that would be needed to strengthen it.

**Deliverable:** One sentence plus an evidence plan.

**Acceptance criteria:** The claim names the population, the input, the output, the decision, and
the sites; it does not claim autonomous diagnosis; the evidence plan includes external validation,
a workflow study, and a monitoring plan.

**Time:** 45 minutes.

## 13. Self-check answers

**Exercise 2.** Patches are 2,400 x 3,800 = 9,120,000. Each embedding is 1,024 x 2 + 64 = 2,112
bytes, so 9,120,000 x 2,112 = 19.26e9 bytes, about 19.3 GB. The probe matrix is 2,400 rows by
1,024 columns. One fine-tuning epoch is 9,120,000 / 150 = 60,800 s = 16.9 h.

**Exercise 4.** The exponentials are 1, 2.718, 7.389, 1, summing to 12.107. The weights are
0.083, 0.225, 0.610, 0.083; the small difference from the stated values comes from rounding the
logits. The pooled vector is 0.083(1,0) + 0.225(0,1) + 0.610(-1,0) + 0.083(0,-1) = (-0.527, 0.142)
with unrounded weights, or about (-0.47, 0.15) at two decimal places. The third patch carries most
of the weight, which is why a single outlier can dominate a slide score.

**Exercise 6.** At 5 percent prevalence, per 1,000 people: 50 positives, of which 42.5 are true
positives and 7.5 false negatives; 950 negatives, of which 95 are false positives and 855 true
negatives. PPV = 42.5 / (42.5 + 95) = 0.309; NPV = 855 / 862.5 = 0.991. At 15 percent:
PPV = 127.5 / (127.5 + 85) = 0.600; NPV = 765 / 772.5 = 0.990. At 40 percent:
PPV = 340 / (340 + 60) = 0.850; NPV = 540 / 600 = 0.900. NPV falls as prevalence rises because
there are fewer true negatives to dominate the denominator.

**Exercises 1, 3, 5, 7, and 8 are assessed against their acceptance criteria.** A strong
Exercise 1 statement separates what the paper reports from what the model card claims. A strong
Exercise 3 plan holds out site D before any normalisation is fitted, because fitting a stain
reference on the external site would leak it. A strong Exercise 5 critique asks for the
prevalence and the split before discussing the number. A strong Exercise 7 plan pre-specifies the
acceptable degradation, because a threshold chosen after seeing the result is not a test.

## 14. Interview questions

**1. What does a pathology foundation model actually give you?**
Strong: a reusable encoder whose representation transfers, with the caveat that the aggregator and
the evaluation still have to be built. Weak: "it understands pathology".

**2. When would you fine-tune rather than probe?**
Strong: when the label budget and an external validation set can support it, and with a frozen
control to prove the gain. Weak: "fine-tuning is always better".

**3. Why is a random slide-level split wrong?**
Strong: it leaks patients and lets the model learn site, inflating the estimate. Weak: "we use a
70/30 split".

**4. What do attention weights mean?**
Strong: a learned weighting that is a hypothesis about evidence, validated against annotation.
Weak: "they show what the model is looking at".

**5. How do you detect site shift?**
Strong: site-stratified metrics, a site classifier on embeddings, and a leave-one-site-out run.
Weak: "we normalise the stain".

**6. Why report AUPRC as well as AUROC?**
Strong: because prevalence changes precision and AUROC hides it. Weak: "AUROC is standard".

**7. A model has 0.95 AUROC and 0.3 PPV. Is it broken?**
Strong: no, prevalence may be low; the threshold and the use case decide. Weak: "yes, the model is
bad".

**8. How would you audit contamination?**
Strong: hashes, near-duplicate search, patient joins, and a temporal holdout. Weak: "we checked
for duplicates".

**9. What is a PCCP and why does it matter for foundation models?**
Strong: a pre-authorised change plan, which is how a retrainable model can be updated without a
new submission. Weak: "a regulatory document".

**10. What evidence would make you comfortable deploying?**
Strong: external validation, calibration, a workflow or reader study, and monitoring. Weak: "a
high AUROC on the test set".

## 15. Key terms

- **Encoder** — the network mapping a patch to a vector.
- **Aggregator** — the model mapping a set of vectors to a slide score.
- **Embedding** — a patch's vector representation.
- **Linear probe** — a classifier trained on frozen embeddings.
- **Fine-tuning** — updating encoder weights on the task.
- **PEFT** — parameter-efficient fine-tuning, such as LoRA or adapters.
- **Zero-shot** — inference with no task-specific training.
- **MIL** — multiple instance learning; bag-level labels over instance sets.
- **ABMIL** — attention-based MIL with gated attention.
- **CLAM** — clustering-constrained attention MIL.
- **Slide-level label** — a diagnosis attached to the specimen.
- **Weak supervision** — labels coarser than the prediction unit.
- **Contamination** — test data present in training or pretraining.
- **Near-duplicate** — a different image of the same tissue or patient.
- **Site shift** — distribution change across hospitals or scanners.
- **External validation** — evaluation at a site not used in development.
- **AUROC** — threshold-free discrimination measure.
- **AUPRC** — precision-recall area, sensitive to prevalence.
- **Calibration** — agreement between predicted and observed probabilities.
- **PPV / NPV** — predictive values at a given prevalence.
- **Shortcut learning** — reliance on a correlate rather than the cause.
- **PCCP** — predetermined change control plan.

## 16. Sources

- Chen, R. J., Ding, T., Lu, M. Y., et al. (2024). Towards a General-Purpose Foundation Model for
  Computational Pathology. *Nature Medicine*, 30, 850-862. doi:10.1038/s41591-024-02857-3
- Lu, M. Y., Chen, B., Williamson, D. F. K., et al. (2024). A Visual-Language Foundation Model for
  Computational Pathology. *Nature Medicine*, 30(3), 863-874. doi:10.1038/s41591-024-02856-4
- Vorontsov, E., Bozkurt, A., Casson, A., et al. (2024). A Foundation Model for Clinical-Grade
  Computational Pathology and Rare Cancers Detection. *Nature Medicine*, 30, 2924-2935.
  doi:10.1038/s41591-024-03141-0
- Xu, H., Usuyama, N., Bagga, J., et al. (2024). A Whole-Slide Foundation Model for Digital
  Pathology from Real-World Data. *Nature*, 630, 181-188. doi:10.1038/s41586-024-07441-w
- Wang, X., Zhao, J., Marostica, E., et al. (2024). A Pathology Foundation Model for Cancer
  Diagnosis and Prognosis Prediction. *Nature*, 634, 970-978. doi:10.1038/s41586-024-07894-z
- Ilse, M., Tomczak, J., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning.
  *Proceedings of ICML*, 2127-2136.
- Lu, M. Y., Williamson, D. F. K., Chen, T. Y., et al. (2021). Data-Efficient and Weakly
  Supervised Computational Pathology on Whole-Slide Images. *Nature Biomedical Engineering*, 5,
  555-570. doi:10.1038/s41551-020-00682-w
- Neidlinger, P., El Nahhas, O. S. M., et al. (2025). Benchmarking Foundation Models as Feature
  Extractors for Weakly Supervised Computational Pathology. *Nature Biomedical Engineering*.
- Lee, M., Lim, S., Byeon, S., & Kwak, J. (2025). Benchmarking Pathology Foundation Models:
  Adaptation Strategies and Scenarios. *Computers in Biology and Medicine*, 190, 110031.
- Enda, et al. (2026). Transfer Learning Strategies for Pathological Foundation Models: A
  Systematic Evaluation in Brain Tumor Classification. *Pathology International*.
  doi:10.1111/pin.70098
- Kömen, J., de Jong, V., Hense, J., et al. (2026). Towards Robust Foundation Models for Digital
  Pathology (PathoROB). *Nature Communications*.
- Collins, G. S., Moons, K. G. M., Dhiman, P., et al. (2024). TRIPOD+AI Statement. *BMJ*, 385,
  e078378.
- Mongan, J., Moy, L., & Kahn, C. E. (2020). Checklist for Artificial Intelligence in Medical
  Imaging (CLAIM). *Radiology: Artificial Intelligence*, 2(2), e200029.
- US Food and Drug Administration. (2021). *Good Machine Learning Practice for Medical Device
  Development*. Guiding principles.
- US Food and Drug Administration. (2025). *Medical Devices; Laboratory Developed Tests;
  Implementation of Vacatur*. Final rule, 90 FR 45134.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous: [`11_wsi-pipeline-infrastructure.md`](11_wsi-pipeline-infrastructure.md)
- Next: [`13_digital-pathology-lab-operations.md`](13_digital-pathology-lab-operations.md)
- General foundation models: [`10_foundation-models.md`](10_foundation-models.md)
- Evaluation: [`07_evaluation.md`](07_evaluation.md)
- Survey file: [`../../course/10_medical_imaging.md`](../../course/10_medical_imaging.md)
- Lab catalogue: [`../../labs/`](../../labs/)
