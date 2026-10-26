# Pathology Datasets

> Module 21 — Datasets, Tools, and Benchmarks
> Topic 07 of 12
> Estimated study time: 4 hours
> Prerequisites: Topic 01, Topic 02, Topic 04
> Version 2026.3

## 1. Why this topic exists

Pathology is the modality where the image is largest, the label is scarcest, and the clinical claim is most consequential. A whole-slide image (WSI) can be 100,000 by 100,000 pixels, roughly 10 gigapixels, and a single slide can exceed 4 GB compressed. A dataset of a thousand slides can be terabytes, and the number of slides with exhaustive lesion annotations can be a few dozen. The arithmetic of pixels, tiles, and storage determines the compute plan; the arithmetic of annotations determines the claim.

The failures are specific:

- A team treats a slide-level label as a lesion-level label and reports detection performance it cannot support.
- Tiles from the same slide or patient appear in both training and test, and the model memorises the slide.
- A model learns the scanner or the stain protocol rather than the tumour, because one center dominates training and the test set comes from the same center.
- A dataset released under a non-commercial licence is used in a commercial product.
- A whole-slide pipeline is planned without the storage arithmetic, and the project runs out of disk before it runs out of ideas.
- A metastasis that occupies a few hundred pixels is diluted by a slide-level label, and the model learns the background.
- A foundation-model benchmark is treated as clinical validation, when it measures linear-probe performance on a curated task set.

This topic covers the major resources: TCGA whole-slide images, CAMELYON16 and CAMELYON17, PANDA, MIDOG, and TCGA-derived collections. It treats the annotation bottleneck, the licence constraints, and the pixel arithmetic as the core mechanics.

The scale of a pathology dataset is deceptive in both directions. A thousand slides sounds large until the lesion annotations cover fifty of them, and a single slide sounds small until it contains ten billion pixels. The planning skill is to hold both numbers at once: the clinical sample size and the computational size are different quantities, and a project can fail on either.

A second framing point: the pathologist is part of the measurement. The label is a human judgement recorded under a protocol, and the model is evaluated against that judgement. Inter-pathologist agreement is therefore not a nuisance statistic; it is the ceiling of the claim.

> **Note:** Counts were checked on 26 October 2026. CAMELYON17 contains 1,000 whole-slide images from 200 patients across 5 centers, with 500 training and 500 test slides; the slides are typically about 4 GB each and the dataset is about 2.3 TB. PANDA contains 10,616 prostate needle biopsies, about 383 GB, under CC BY-SA-NC 4.0. Verify the current download page before planning storage.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** pixels, tiles, and storage for a whole-slide dataset at a stated magnification and tissue fraction. Assessed by Exercise 1 and self-check item 4.
2. **Distinguish** slide-level, lesion-level, and patient-level labels and the claims each supports. Assessed by Exercise 2.
3. **Audit** a pathology split for slide, patient, and center leakage. Assessed by Exercise 3.
4. **Estimate** the annotation cost for a lesion-level dataset and the ceiling it imposes. Assessed by Exercise 4.
5. **Map** a pathology dataset licence to a product plan and identify the blocking clause. Assessed by Exercise 5.
6. **Design** a tile-sampling policy that does not leak label information and preserves rare lesions. Assessed by Exercise 6.
7. **Evaluate** scanner and stain shift across centers. Assessed by Exercise 7.
8. **Write** a pathology data card that states magnification, label granularity, licence, and prohibited claims. Assessed by Exercise 8.

## 3. The core mental model

A pathology dataset is a pyramid of images with a label attached at one level. The level of the label determines the claim.

```text
   PATIENT
     |  one patient, many slides
     v
   SLIDE (WSI)
     |  one slide, many tiles at one magnification
     v
   PYRAMID LEVEL
     40x | 20x | 10x | 5x | thumbnail
     |
     v
   TILE
     256 x 256 | 512 x 512 | other
     |
     v
   LABEL LEVEL
     patient | slide | region | lesion | cell
     |
     v
   CLAIM
     diagnosis | detection | grading | prognosis | triage
```

**The label-level rule.** A slide-level label supports a slide-level claim. A lesion-level claim needs lesion coordinates and an annotation protocol. A patient-level claim needs all of a patient's slides grouped together. Most public datasets provide a mix, and the mix is often undocumented.

**The tile rule.** Tiles are not independent samples. Adjacent tiles overlap and share the same tissue, stain, and scanner. A tile-level split leaks the slide. Tile sampling also determines what the model can see: a metastasis that occupies 0.1 percent of a slide will be absent from most random tiles.

**The magnification rule.** A 40x image at 0.25 micrometres per pixel contains four times as many pixels as a 20x image at 0.5 micrometres per pixel. Halving magnification quarters the pixel count and the tile count. The choice changes compute, storage, and the visibility of nuclear detail.

**The annotation rule.** Annotation is the binding constraint. A slide-level label can be extracted from a report in seconds; a lesion-level polygon can take a pathologist tens of minutes. The cost per slide determines the size of the labelled set and the ceiling of the claim.

**The stain and scanner rule.** Haematoxylin and eosin (H&E) staining, scanner model, and compression vary across centers. A model trained on one protocol can learn the protocol. Stain normalisation helps within a scanner and can hurt across scanners.

### Claim-to-evidence map

| Claim | Label needed | Metric | Dataset example |
|---|---|---|---|
| Detect metastasis | Lesion polygon | Lesion sensitivity | CAMELYON16 subset |
| Stage a patient | Patient pN | Patient-level kappa | CAMELYON17 |
| Grade a biopsy | ISUP grade | Quadratic weighted kappa | PANDA |
| Prioritise a slide | Slide label | Sensitivity at capacity | CAMELYON17 |
| Predict recurrence | Patient outcome | Time-dependent AUROC | Linked cohort |
| Segment a region | Region polygon | Dice and boundary | Annotated subset |

The map is the fastest way to see whether a dataset can support a claim. If the label column and the metric column do not match the dataset, the claim is out of scope.

### Dataset scale comparison

| Dataset | Slides | Patients | Centers | Storage | Label level |
|---|---:|---:|---:|---:|---|
| CAMELYON16 | 400 | About 160 | 2 | About 0.7 TB | Lesion subset |
| CAMELYON17 | 1,000 | 200 | 5 | About 2.3 TB | Patient pN stage |
| PANDA | 10,616 biopsies | Not stated as patients | 2 | About 383 GB | ISUP grade |
| TCGA WSI | Thousands | About 11,000 in TCGA | Many | Multi-terabyte | Diagnosis-linked |
| MIDOG | Edition-specific | Edition-specific | Multiple scanners | Edition-specific | Mitosis annotations |

The table is a planning aid, not a comparison of quality. The datasets answer different questions and have different label levels.

| Level | What it supports | What it cannot support | Typical availability |
|---|---|---|---|
| Patient | Patient-level diagnosis or prognosis | Lesion location | Linked slides and reports |
| Slide | Slide-level classification | Lesion detection | Most public datasets |
| Region | Region-level classification | Whole-slide prevalence | Annotated subsets |
| Lesion | Detection and segmentation | Patient-level outcome | Small annotated subsets |
| Cell | Cell classification or counting | Tissue architecture | Research subsets |

## 4. Definitions and vocabulary

### 4.1 Image and slide terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Whole-slide image | A gigapixel image of a tissue slide. | Treated as a normal image. | Pyramid file |
| Pyramid | Multiple resolutions of the same slide. | Assumed to be independent images. | Levels |
| Magnification | Objective power, such as 40x. | Confused with resolution. | x |
| Micrometres per pixel | Physical pixel size. | Not recorded. | µm/px |
| Tile | A patch extracted from a slide. | Treated as independent. | Pixels |
| Stride | The step between tiles. | Ignored in overlap counting. | Pixels |
| Tissue mask | A mask separating tissue from background. | Omitted, inflating tile counts. | Binary mask |
| Stain | The dye protocol, usually H&E. | Assumed constant. | Protocol |
| Scanner | The device that digitised the slide. | Not recorded. | Model |
| Compression | The encoding of the image. | Assumed lossless. | Ratio |
| Pyramid file | A multi-resolution image container. | Treated as a single TIFF. | Format |
| Focus quality | Sharpness of the scanned image. | Ignored as a quality flag. | Score |

### 4.2 Label and annotation terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Slide-level label | One label per slide. | Used for detection. | Label |
| Lesion-level label | Coordinates and class of a lesion. | Assumed present when only slide labels exist. | Polygon |
| Patient-level label | One label per patient. | Computed from slides without grouping. | Label |
| pN stage | Lymph node stage. | Treated as a lesion label. | Ordinal |
| ISUP grade | Prostate cancer grade group. | Treated as a continuous score. | Ordinal |
| Annotation | The process of marking regions. | Assumed complete. | Process |
| Adjudication | Resolution of reader disagreement. | Omitted. | Process |
| Inter-pathologist agreement | Agreement between pathologists. | Reported without the scale. | Kappa |
| Weak label | A label derived from reports or registries. | Treated as expert annotation. | Label type |
| Annotation density | Annotated area per slide. | Ignored. | Proportion |

### 4.3 Evaluation and operations terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Tile-level metric | Performance on tiles. | Reported as slide-level. | Metric |
| Slide-level metric | Performance on slides. | Reported as patient-level. | Metric |
| Patient-level metric | Performance on patients. | Computed without grouping. | Metric |
| Quadratic weighted kappa | Agreement for ordinal grades. | Used for binary tasks. | Statistic |
| Dice | Overlap for segmentation. | Used for detection. | 0 to 1 |
| Free-response ROC | Detection performance with localisation. | Replaced by AUROC. | Curve |
| Inference cost | Compute per slide. | Reported per tile. | Seconds |
| Throughput | Slides per hour. | Ignored in deployment. | Slides/hour |
| Storage cost | Bytes per slide. | Underestimated. | GB |
| Scanner shift | Distribution change across scanners. | Ignored. | Domain |

### 4.4 Quality and workflow terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Focus quality | Sharpness of the scanned image. | Ignored | Score |
| Tissue fold | Folded tissue creating false structures. | Counted as pathology | Flag |
| Air bubble | Occlusion under the coverslip. | Missed | Flag |
| Pen mark | Ink on the slide. | Treated as tissue | Flag |
| Section thickness | Thickness of the tissue slice. | Not recorded | Micrometres |
| Fixation time | Time before processing. | Ignored | Hours |
| Coverslip | Glass covering the tissue. | Not recorded | Protocol |
| Scan time | Time to digitise a slide. | Confused with inference time | Minutes |
| Throughput | Slides processed per hour. | Overestimated | Slides/hour |
| Review capacity | Slides a pathologist can read per session. | Ignored | Slides/session |

### 4.5 Resource terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| TCGA WSI | Diagnostic slides matched to TCGA. | Treated as a uniform cohort. | Dataset |
| CAMELYON16 | Lymph node metastasis detection challenge. | Treated as a patient-level cohort. | Dataset |
| CAMELYON17 | Patient-level lymph node staging challenge. | Treated as a lesion-level dataset. | Dataset |
| PANDA | Prostate biopsy grading challenge. | Treated as a general pathology dataset. | Dataset |
| MIDOG | Mitosis detection across scanners. | Treated as a clinical validation. | Challenge |
| Foundation model | A model pretrained on large image collections. | Treated as validated. | Model |

## 5. The mechanics, step by step

The procedure has 18 steps in four phases: define the claim and the label level, compute the pixel and storage budget, audit the data and annotations, and design the split and evaluation.

### Phase A — Define the claim and the label level

**Step 1. Write the clinical decision.**
Input: the workflow and the intended user.
Output: a decision statement with the actor, action, timing, population, and error costs.
Rationale: the decision determines whether the claim is detection, grading, triage, or prognosis.
Branch: a triage claim needs sensitivity at a fixed review capacity; a grading claim needs ordinal agreement.
Edge case: a diagnosis-support claim may need a reader study, not only a held-out metric.

**Step 2. Fix the label level.**
Input: the decision statement and the available annotations.
Output: patient, slide, region, lesion, or cell.
Rationale: the label level determines the claim and the evaluation.
Branch: if only slide labels exist, the claim is slide-level classification, not lesion detection.
Edge case: a patient-level label can hide a positive slide among negatives; the aggregation rule must be stated.

**Step 3. Choose the magnification and tile size.**
Input: the structure of interest and the compute budget.
Output: micrometres per pixel, tile size, and stride.
Rationale: magnification determines whether nuclei, glands, or architecture are visible.
Branch: if the structure is nuclear, 40x or 20x; if architectural, 10x may suffice.
Edge case: a multi-scale model can combine levels, at higher compute cost.

**Step 4. Estimate the tile count and storage.**
Input: slide dimensions, magnification, tile size, stride, and tissue fraction.
Output: tiles per slide, tiles per dataset, and terabytes.
Rationale: the compute and storage plan depends on these numbers before any model is trained.
Branch: if the dataset exceeds the storage budget, downsample or sample tiles.
Edge case: compressed slides are smaller than raw pixels; use the actual file size for storage and the pixel count for compute.

### Annotation cost arithmetic

| Task | Time per case | Cases for 1,000 annotations | Cost driver |
|---|---|---|---|
| Slide-level label from report | 1 to 3 minutes | 17 to 50 hours | Extraction and review |
| Region annotation | 5 to 10 minutes | 83 to 167 hours | Pathologist time |
| Lesion polygon | 10 to 30 minutes | 167 to 500 hours | Lesion complexity |
| Cell-level annotation | 30 to 120 minutes | 500 to 2,000 hours | Cell density |
| Adjudication | 5 to 20 minutes per disagreement | 20 to 40% extra | Disagreement rate |
| Quality control re-read | 5 to 10% of cases | 50 to 100 cases | Sampling |
| Training and calibration | 4 to 16 hours | Fixed | Protocol complexity |

The arithmetic explains why lesion-level datasets are small. A thousand polygon annotations at 20 minutes each is about 333 pathologist-hours before adjudication. That is the real cost of a detection claim.

### Stain and scanner variation

| Source | Effect | Detectable from metadata | Mitigation |
|---|---|---|---|
| H&E protocol | Colour and contrast | Sometimes | Stain normalisation |
| Scanner model | Sharpness and colour | Often | Scanner holdout |
| Compression | Fine detail | Sometimes | Re-scan or use original |
| Section thickness | Nuclear overlap | Rarely | Protocol documentation |
| Fixation time | Antigen and morphology | Rarely | Site-level stratification |
| Coverslip and mounting | Artefact | Rarely | Quality review |
| Focus quality | Blur | Image metric | Quality exclusion |
| Tissue folds | False structures | Image review | Artefact detection |

Stain normalisation helps within a scanner and can hurt across scanners. The only reliable test is a center or scanner holdout, reported separately.

### Tile sampling policies

| Policy | How it works | Strength | Failure |
|---|---|---|---|
| Full grid | Every tissue tile | No sampling bias | Compute cost |
| Random sample | Random tissue tiles | Cheap | Misses rare lesions |
| Top-k by model | Model scores tiles | Efficient inference | Feedback loop |
| Positive-guided | Sample near annotations | Preserves lesions | Leaks label at inference |
| Hard-negative mining | Sample confusing background | Improves specificity | Overfits artefacts |
| Multi-scale | Tiles at several levels | Context and detail | Higher cost |
| Tissue-mask only | Exclude background | Reduces tiles | Mask errors |

Positive-guided sampling uses the label to choose tiles, which is a form of leakage if the same rule is unavailable at deployment. Use it for training only, with a fixed-grid evaluation.

### Detection evaluation

| Metric | What it measures | Use | Limitation |
|---|---|---|---|
| Slide-level sensitivity | Whether a positive slide is flagged | Triage | Hides lesion location |
| Lesion-level sensitivity | Whether a lesion is found | Detection | Needs lesion annotations |
| False positives per slide | Burden of overcall | Review workload | Depends on slide size |
| Free-response ROC | Detection across thresholds | Research | Complex to communicate |
| Dice | Segmentation overlap | Segmentation | Ignores clinical importance |
| Time per slide | Throughput | Deployment | Hardware dependent |

### Quality-control checks

| Check | Method | Frequency |
|---|---|---|
| Focus | Variance of Laplacian or model score | Every slide |
| Tissue detection | Mask coverage | Every slide |
| Stain intensity | Colour statistics | Batch |
| Scanner metadata | Header review | Batch |
| Duplicate slides | Perceptual hash | Dataset build |
| Annotation scale | Coordinate check against level | Dataset build |
| Label distribution | Prevalence by center | Dataset build |
| Test-set overlap | Patient and slide hash | Before evaluation |

### Storage planning

| Item | Unit | Example | Consequence |
|---|---|---|---|
| Slide file | GB | 4 GB | 1,000 slides = 4 TB |
| Tile cache | GB | 50 GB per 100 slides | Faster training |
| Feature store | MB per slide | 1 to 10 MB | Cheaper than tiles |
| Backups | TB | Duplicate of source | Doubles storage |
| Egress | TB | Transfer to cloud | Cost and time |
| Retention | Months | Project term | Deletion obligation |

The feature store is often the right compromise: store embeddings rather than tiles, and keep the slides in governed storage. The trade-off is that a new model architecture may need the pixels again.

### Slide quality checks

| Check | Method | Threshold | Consequence |
|---|---|---|---|
| Focus | Variance of Laplacian | Study-specific | Blurred regions misclassified |
| Tissue coverage | Mask area | Minimum area | Empty slides in the cohort |
| Folds | Visual or model | Any fold flag | False structures |
| Air bubbles | Visual | Any bubble flag | Occluded tissue |
| Pen marks | Colour detection | Any mark flag | Artefact labels |
| Coverslip edge | Mask | Border region | Exclude border tiles |
| Stain intensity | Colour statistics | Batch range | Normalisation needed |
| Scanner metadata | Header | Required fields | Transport audit blocked |

### Foundation-model evaluation

| Question | What to check | Failure mode |
|---|---|---|
| What was it pretrained on? | Dataset list and dates | Test slides included |
| What is the evaluation task? | Linear probe, fine-tune, or few-shot | Benchmark task differs from claim |
| What is the baseline? | Logistic regression on features | No simple baseline |
| Is the scanner represented? | Center and scanner metadata | Same-scanner test |
| Is the label level correct? | Slide, lesion, or patient | Mismatch with claim |
| Is calibration reported? | Calibration curve | Threshold unusable |
| Is the compute cost reported? | GPU-hours per slide | Deployment infeasible |
| Is the code available? | Repository and weights | Not reproducible |

A foundation model is a feature extractor until it is validated on the target data. The validation is the same as for any other model: patient-level split, center holdout, label-level match, and a simple baseline.

The storage decision is also a governance decision. A tile cache and a feature store are derived artefacts, and the licence may restrict them as it restricts the source. A team that keeps tiles after the project ends can be out of compliance even if the original slides were deleted.

A whole-slide plan without these numbers is a guess. The numbers do not need to be exact to be useful; they need to be stated, sourced, and revised when the tissue mask or the scanner changes. A team that writes them down will catch the storage problem before the download and the compute problem before the training run.

### Phase B — Compute the pixel and storage budget

**Step 5. Convert physical size to pixels.**
Input: tissue dimensions and micrometres per pixel.
Output: width and height in pixels.
Rationale: a 25 mm by 25 mm tissue section at 0.25 µm/px is 100,000 by 100,000 pixels.
Branch: if the section is smaller, the pixel count falls quadratically.
Edge case: scanner resolution can differ from the nominal magnification.

**Step 6. Count tiles including overlap.**
Input: width, height, tile size, and stride.
Output: the number of tiles at the chosen stride.
Rationale: a 50 percent overlap multiplies the tile count by about four.
Branch: use a tissue mask to exclude background tiles.
Edge case: edge tiles may be partial; define whether they are included.

**Step 7. Estimate compute per slide.**
Input: tiles per slide, model cost per tile, and batch size.
Output: seconds or GPU-hours per slide and for the dataset.
Rationale: inference cost determines whether the product can run in the workflow.
Branch: if the cost exceeds the turnaround time, use a lower magnification or a cascade.
Edge case: attention-based multiple-instance models add cost beyond tile inference.

**Step 8. Estimate storage and transfer.**
Input: file size per slide, dataset size, and retention.
Output: terabytes and transfer time.
Rationale: a 2.3 TB dataset over a constrained link is a schedule risk.
Branch: if storage is limited, keep tiles or features rather than full slides.
Edge case: derived tiles and features also consume storage and are governed by the licence.

### Phase C — Audit the data and annotations

**Step 9. Inventory slides, patients, and centers.**
Input: the manifest.
Output: counts by patient, slide, center, scanner, and year.
Rationale: the patient and center counts determine the split and the transport test.
Branch: if a patient has multiple slides, all must stay on one side of the split.
Edge case: a slide can be missing from the manifest but present in the file store.

**Step 10. Audit the label granularity and provenance.**
Input: the annotation files and the dataset paper.
Output: which labels exist at which level, and who produced them.
Rationale: a patient-level pN stage is not a lesion annotation.
Branch: if lesion annotations cover only a subset, the detection claim is limited to that subset.
Edge case: annotation files can use coordinates at a different pyramid level; check the scale.

**Step 11. Estimate annotation cost.**
Input: the annotation protocol and the pathologist time per case.
Output: hours and cost for the target sample size.
Rationale: the cost sets the size of the labelled set and the ceiling.
Branch: if the cost exceeds the budget, use weak labels for pretraining and a small adjudicated test set.
Edge case: adjudication and quality control add 20 to 40 percent.

**Step 12. Audit class balance and lesion size.**
Input: the labels and the lesion coordinates.
Output: prevalence at slide and lesion level, and the size distribution.
Rationale: a rare, small lesion is diluted by a slide-level label.
Branch: if positive tiles are a tiny fraction, use a sampling policy that preserves them.
Edge case: a slide can contain multiple lesions of different classes.

**Step 13. Audit stain and scanner variation.**
Input: the metadata.
Output: a table of center, scanner, stain protocol, and year.
Rationale: scanner and stain are the main sources of domain shift.
Branch: if one center dominates, hold it out or reweight.
Edge case: the scanner field can be missing while the image retains a fingerprint.

### Phase D — Design the split and evaluation

**Step 14. Split by patient and center.**
Input: the patient and center identifiers.
Output: a split with group keys and a hash.
Rationale: tiles and slides from one patient leak identity.
Branch: if the dataset has only one center, state the transport limitation.
Edge case: a patient can appear under different identifiers across centers.

**Step 15. Choose the evaluation level.**
Input: the claim and the label level.
Output: tile, slide, lesion, or patient metric.
Rationale: the metric must match the decision.
Branch: for triage, use slide-level sensitivity at capacity; for detection, use lesion-level sensitivity and false positives per slide.
Edge case: a slide-level metric can hide a missed small lesion.

**Step 16. Fit stain normalisation and preprocessing on training data.**
Input: the training split and the pipeline.
Output: stored parameters applied to all splits.
Rationale: fitting normalisation on the full dataset leaks test information.
Branch: if normalisation is per-slide, document it and test across scanners.
Edge case: aggressive normalisation can remove diagnostic colour differences.

**Step 17. Evaluate by center, scanner, and lesion size.**
Input: the test set and the metadata.
Output: metrics with confidence intervals by group.
Rationale: an average can hide a center that fails.
Branch: if a subgroup has too few lesions, report descriptively.
Edge case: a center with high performance can reflect easier cases.

**Step 18. Score the gates and record the evidence.**
Input: the outputs of steps 1 to 17.
Output: a gate table with evidence and residual uncertainty.
Rationale: the decision must be reviewable.
Branch: a zero on any gate is a veto for the claim as written.
Edge case: a dataset can be admissible for pretraining and inadmissible for a product claim.

### Resource profiles

| Resource | Custodian | Version inspected | Unit | Scale | Access and licence | What it cannot answer |
|---|---|---|---|---|---|---|
| TCGA WSI | NCI and GDC | Current release | Slide, patient | Thousands of diagnostic slides | Open via GDC; check terms | Uniform protocol; lesion-level truth for all slides |
| CAMELYON16 | Radboud and collaborators | 2016 edition | Slide, patient | 400 slides; 159 annotated | Open via Grand Challenge | Patient-level staging; multi-cancer scope |
| CAMELYON17 | Radboud and collaborators | 2017 edition | Patient, slide | 1,000 slides; 200 patients; 5 centers; about 2.3 TB | CC0 | Lesion annotations for all slides; other cancer types |
| PANDA | Radboud and Karolinska | 2020 release | Biopsy | 10,616 biopsies; about 383 GB | CC BY-SA-NC 4.0 | Commercial use; full Gleason pattern detail |
| MIDOG | Challenge consortium | Edition-specific | Slide, region | Multiple scanner domains | Challenge registration | Clinical deployment; broad cancer scope |
| TCGA-derived sets | Various | Collection-specific | Slide, patient | Subsets of TCGA | Per-collection terms | Independence from TCGA |

The scale column mixes slides, patients, and biopsies. A slide count is not a patient count, and a patient count is not a lesion count.

### Whole-slide arithmetic

| Quantity | Formula | Example |
|---|---|---|
| Pixels per side | physical size / micrometres per pixel | 25 mm / 0.25 µm = 100,000 px |
| Total pixels | width x height | 100,000 x 100,000 = 10^10 |
| Tiles, no overlap | ceil(width / tile) x ceil(height / tile) | 391 x 391 ≈ 152,100 |
| Tiles, 50% overlap | roughly 4x the non-overlapping count | about 608,400 |
| Tissue tiles | tiles x tissue fraction | 152,100 x 0.35 ≈ 53,235 |
| Raw storage | pixels x bytes per pixel | 10^10 x 3 = 30 GB |
| Compressed storage | raw / compression ratio | 30 GB / 8 = 3.75 GB |
| Dataset storage | slides x compressed size | 1,000 x 4 GB = 4 TB |
| Inference time | tiles x seconds per tile | 53,235 x 0.02 s ≈ 1,065 s |

The arithmetic is approximate and deliberately so. Its purpose is to bound the plan before compute is purchased. A team that cannot produce these numbers cannot size a cluster or a storage budget.

### Feature storage arithmetic

| Representation | Size per slide | 1,000 slides | Trade-off |
|---|---:|---:|---|
| Full WSI | About 4 GB | About 4 TB | Full fidelity, high storage |
| Tiles at 256 px | About 0.5 to 2 GB | 0.5 to 2 TB | Faster access, still large |
| Features | About 1 to 10 MB | 1 to 10 GB | Cheap, fixed extractor |
| Slide-level summary | About 1 KB | About 1 MB | Very cheap, loses detail |

The feature store is the usual compromise for large cohorts. It fixes the extractor and makes a new architecture expensive to evaluate, which is a scientific cost as well as a storage decision.

### Magnification and tile count

| Magnification | Micrometres per pixel | Pixels across 25 mm | 256-px tiles |
|---:|---:|---:|---:|
| 40x | 0.25 | 100,000 | 152,100 |
| 20x | 0.50 | 50,000 | 38,025 |
| 10x | 1.00 | 25,000 | 9,409 |
| 5x | 2.00 | 12,500 | 2,401 |

Halving magnification quarters the pixel count. The visibility of nuclear detail falls with it, which is why the choice is scientific as well as computational.

### Label-level decision table

| Available label | Claim it supports | Claim it cannot support | Evaluation |
|---|---|---|---|
| Patient pN stage | Patient-level staging | Lesion detection | Patient-level kappa |
| Slide tumour present | Slide classification | Patient-level prognosis | Slide-level AUROC |
| Region annotation | Region classification | Whole-slide prevalence | Region-level metric |
| Lesion polygon | Detection and segmentation | Patient outcome | Lesion-level sensitivity |
| Cell annotation | Cell classification | Tissue architecture | Cell-level agreement |

### Split decision table

| Split | What it estimates | Requirement | Limitation |
|---|---|---|---|
| Random tile | Memorisation resistance | None | Leaks slide and patient |
| Slide-level | Generalisation to new slides | Slide ID | Same patient may recur |
| Patient-level | Generalisation to new patients | Patient ID | Same center and scanner |
| Center-level | Generalisation to a new center | Center ID | Few centers means high variance |
| Scanner-level | Generalisation to a new scanner | Scanner ID | Confounded with center |
| Temporal | Generalisation to a later period | Dates | Staining and protocol change |

### Edge cases that change the procedure

**Licence forbids commercial use.** PANDA is CC BY-SA-NC 4.0 and cannot be used for a commercial product without a separate licence. CAMELYON17 is CC0 and permits commercial use. TCGA slides have their own terms. Check each dataset.

**Dataset withdrawn.** Open challenge data can be removed or moved. Keep a local manifest and hash, and confirm the licence before relying on a mirror.

**Credentialing takes months.** Open challenge data are fast; institutional slides and linked reports can take months. A timeline shorter than the access path fails Gate 3.

**Version drift.** A challenge edition can change the label set, the annotation protocol, and the test distribution. Pin the edition and do not compare across editions.

**Label errors discovered.** Pathologist disagreement is common, and a slide-level label can be wrong. Re-adjudicate a sample, report agreement, and test sensitivity. Do not relabel only the test set.

**Benchmark contamination.** A foundation model may have seen public challenge slides. Check the model's pretraining data and use a private holdout where possible.

## 6. Worked example

### The case

A 40-person digital pathology company wants to sell a triage tool to hospital pathology departments. After a sentinel lymph node is stained with H&E and scanned, the tool flags slides with suspected metastasis so that a pathologist reviews them first. The company has 12 months to a pilot and intends a commercial product.

**Decision statement.** "Within 10 minutes of scanning a sentinel lymph node slide, flag suspected metastasis so that the pathologist reviews it before the rest of the queue; the department can review 30 percent of slides first, a false alert costs about 5 minutes of pathologist time, and a missed metastasis delays staging and may change surgery."

**Unit.** The patient for staging, the slide for triage, and the lesion for detection. The three units need three evaluations.

**Claim.** The triage claim is slide-level sensitivity at a fixed review capacity. The detection claim is lesion-level sensitivity and false positives per slide. The company intends both.

### Candidate long list

| Candidate | Custodian | Version inspected | Unit | Scale | Access and licence | Why it is on the list |
|---|---|---|---|---|---|---|
| CAMELYON17 | Radboud and collaborators | 2017 edition | Patient, slide | 1,000 slides; 200 patients; 5 centers; about 2.3 TB | CC0 | Patient-level staging across centers |
| CAMELYON16 | Radboud and collaborators | 2016 edition | Slide, lesion | 400 slides; 159 annotated | Open via Grand Challenge | Lesion annotations for detection |
| TCGA WSI | NCI and GDC | Current release | Slide, patient | Thousands of diagnostic slides | Open with terms | Morphology diversity for pretraining |
| PANDA | Radboud and Karolinska | 2020 release | Biopsy | 10,616 biopsies; about 383 GB | CC BY-SA-NC 4.0 | Grading methods, wrong organ |
| Institutional partner | Two hospital networks | Not collected | Slide, patient | Site-dependent | Contract and IRB | Target workflow and scanner |

### Gate scores for the commercial claim

| Gate | CAMELYON17 | CAMELYON16 | TCGA WSI | PANDA | Partner |
|---|---:|---:|---:|---:|---:|
| 1 Decision fit | 2 | 2 | 1 | 0 | 2 |
| 2 Population fit | 1 | 1 | 1 | 0 | 2 |
| 3 Access fit | 2 | 2 | 2 | 2 | 0 |
| 4 Licence fit | 2 | 2 | 1 | 0 | 2 |
| 5 Benchmark fit | 1 | 2 | 1 | 1 | 2 |
| Veto | none | none | none | decision, licence | access |

**CAMELYON17, Gate 1 and Gate 5.** The dataset provides patient-level pN stage across five centers, which supports a patient-level staging claim. It does not provide lesion polygons for every slide, so it cannot support a lesion-detection claim across the cohort. Scores 2 and 1.

**CAMELYON17, Gate 4.** Released under CC0, which permits commercial use. Score 2. The attribution and redistribution terms should still be recorded.

**CAMELYON16, Gate 1 and Gate 5.** The dataset provides lesion annotations for a subset, which supports detection evaluation on that subset. The slide count is smaller and the centers are Dutch. Scores 2 and 2.

**TCGA WSI, Gate 1.** The slides are diagnostic cancer slides across tumour types, not sentinel lymph node staging. They are useful for pretraining and morphology, not for the triage claim. Score 1.

**TCGA WSI, Gate 4.** TCGA open-tier slides have their own terms, and some linked data are controlled. Score 1 until the terms are reviewed.

**PANDA, Gate 1 and Gate 4.** The dataset is prostate biopsy grading, not breast lymph node metastasis, and it is CC BY-SA-NC 4.0. Veto on decision fit and licence.

**Partner, Gate 3.** Two hospital networks are interested, but no contract, IRB approval, or data-sharing agreement exists. Twelve months is not credible. Veto for the timeline, retained as the validation path.

### The two diagnostics

**Diagnostic 1: label granularity.** CAMELYON17's primary label is patient-level pN stage, derived from five slides per patient. Only 50 slides across the five centers have exhaustive lesion polygons. A model trained on patient-level labels can rank patients and cannot localise a metastasis. The company's detection claim needs lesion-level truth, and the dataset supplies it for a small subset. The diagnostic is not the dataset size; it is the mismatch between the label level and the claim.

**Diagnostic 2: lesion prevalence and tile dilution.** A micrometastasis can occupy a few hundred pixels in a 10-gigapixel slide. At a 256-pixel tile and a 35 percent tissue fraction, a slide has about 53,000 tissue tiles. A lesion covering 300 by 300 pixels appears in roughly one to four tiles. The positive-tile prevalence is therefore far below the slide-level prevalence, and a model trained on uniformly sampled tiles sees mostly background. The dataset can support slide-level triage and cannot, by itself, teach lesion localisation without a sampling policy and lesion annotations.

A third diagnostic is scanner shift. The five centers used three scanner models. A model trained on four centers and tested on the fifth measures transport across scanner and stain. A random tile split measures neither.

### The selected route

The admissible set is CAMELYON17, CAMELYON16, and TCGA WSI. The plan uses CAMELYON17 for patient-level staging and center holdout, CAMELYON16 for lesion detection on the annotated subset, TCGA for pretraining, and the partner for the target scanner and workflow.

| Track | Steps | Owner | Duration | Planning cost |
|---|---|---|---|---|
| Data acquisition | Download 2.3 TB; verify hashes | Data engineering | 2 to 4 weeks | Storage and egress |
| Tile pipeline | Tissue masks, tiles, features | ML engineering | 6 weeks | Compute, about $10,000 |
| Staging model | Patient-level training and center holdout | ML team | 10 weeks | Compute, about $15,000 |
| Detection model | Lesion-level training on CAMELYON16 | ML team | 10 weeks | Compute, about $12,000 |
| Reader study | Pathologist review with and without the tool | Clinical lead | 8 weeks | About $30,000 |
| Partner validation | Contract, IRB, silent mode | Clinical lead | 6 to 9 months | Six figures |

### Time and cost

| Item | Detail | Calendar time | Planning cost |
|---|---|---|---|
| Storage | 2.3 TB plus tile cache | Ongoing | About $500 per month |
| Transfer | 2.3 TB over the link | 1 to 3 weeks | Egress cost |
| Compute | Training and inference | 20 weeks | About $37,000 |
| Reader study | 6 pathologists, 200 slides | 8 weeks | About $30,000 |
| Partner validation | Contract and pilot | 6 to 9 months | Six figures |
| Total to a pilot decision | Excludes partner contract | About 30 weeks | About $80,000 |

The storage and transfer are not the largest costs, and they are the ones most often omitted from the plan. A 2.3 TB download on a constrained link can delay the project by weeks.

### Substitute plan if the partner route fails

If the hospital networks cannot provide data, the plan changes in three ways:

1. The claim narrows to a research benchmark on public data, with no workflow evidence.
2. The reader study becomes the primary evaluation, using public slides read under a simulated workflow.
3. The product launch is deferred until a partner is secured, because the target scanner and stain cannot be validated otherwise.

The fallback preserves the methods and loses the commercial claim. It is not equivalent.

### Vary one assumption: CAMELYON17 were non-commercial

Suppose CAMELYON17 were released under a non-commercial licence like PANDA. Gate 4 falls to 0 for the product claim. The plan changes:

- CAMELYON16 remains usable only if its terms permit commercial use; otherwise both public datasets are research-only.
- The product model must be trained on partner data or a licensed vendor collection.
- TCGA slides can support pretraining if their terms permit it, but they cannot replace the target cohort.
- The timeline extends by the partner data collection, and the budget shifts from compute to a data contract.

If the variation is size, a smaller slide set would weaken the center holdout but not change the label-level conclusion. If the variation is label quality, a lower inter-pathologist agreement on micrometastases would lower the ceiling and require adjudication.

Three conditions would end the plan: no partner, no lesion truth for the detection claim, and no commercial licence. The first is a schedule risk, the second is a scientific limit, and the third is a product veto.

### Gate evidence log

| Gate | Candidate | Evidence inspected | Verdict | Residual uncertainty |
|---|---|---|---|---|
| 1 | CAMELYON17 | Patient pN stage; 1,000 slides | 2 | Lesion polygons for 50 slides only |
| 1 | CAMELYON16 | Slide labels and 159 annotated slides | 2 | Small cohort |
| 2 | CAMELYON17 | Five Dutch centers; three scanners | 1 | Not the target population |
| 2 | TCGA WSI | Multi-cancer diagnostic slides | 1 | Not lymph node staging |
| 3 | CAMELYON17 | Public download; about 2.3 TB | 2 | Transfer time and storage |
| 3 | Partner | No contract or IRB | 0 | Timeline not credible |
| 4 | CAMELYON17 | CC0 | 2 | Attribution record |
| 4 | PANDA | CC BY-SA-NC 4.0 | 0 | Non-commercial |
| 5 | CAMELYON17 | Patient-level metric | 1 | No lesion-level benchmark for all slides |
| 5 | CAMELYON16 | Lesion annotations for a subset | 2 | Subset selection |
| 5 | Partner | Workflow and scanner match | 2 | Not started |

### Decision timeline

| Month | Data track | Model track | Clinical track | Deliverable |
|---:|---|---|---|---|
| 1 | Download CAMELYON17 and 16 | Tile pipeline design | Define review workflow | Data manifest |
| 2 | Verify hashes and masks | Feature extraction | Pathologist recruitment | Tile cache |
| 3-4 | TCGA pretraining set | Staging model | Reader protocol | Model v0 |
| 5-6 | Center holdout | Detection model | Reader training | Technical report |
| 7-8 | Error analysis | Calibration and thresholds | Reader study | Reader results |
| 9-12 | Partner data | Product model | Silent-mode pilot | Pilot decision |

### What the case demonstrates

- The label level, not the slide count, determined which claim was supportable. Patient-level staging and lesion-level detection needed different evidence from the same collection.
- The positive-tile prevalence was far below the slide prevalence. A model trained on uniform tiles would see mostly background.
- The licence differed across datasets that looked similar. CAMELYON was CC0 and PANDA was non-commercial.
- The storage arithmetic was a schedule risk before it was a cost. A 2.3 TB download can consume weeks.
- The center holdout was the only test that measured scanner and stain transport.
- The fallback lost the workflow claim, not only the dataset.

### Sensitivity of the decision

| Change | Gate affected | Does the source change? | Plan response |
|---|---|---|---|
| A partner signs in month two | 3 and 5 | Yes, becomes primary | Shift training and validation |
| CAMELYON17 adds lesion annotations | 5 | No | Add a detection benchmark |
| The licence becomes non-commercial | 4 | Yes | Use partner data for the product |
| A new scanner is deployed | 2 and 5 | No | Re-run the center holdout |
| Inter-pathologist agreement is low | 1 and 5 | No | Adjudicate and lower the ceiling |

### Conditions under which the case would fail

- **The lesion annotations are too few.** The detection claim is limited to the annotated subset, and the product claim weakens.
- **The partner does not sign.** There is no target-scanner validation, and the pilot cannot start.
- **The inter-pathologist agreement is low.** The label ceiling falls, and the triage threshold must be renegotiated.
- **The storage or transfer budget is insufficient.** The dataset cannot be processed on the timeline.
- **The scanner mix is narrow.** The center holdout becomes a same-scanner test, and transport is unmeasured.
- **The licence changes.** A non-commercial term blocks the product, and the partner route becomes mandatory.

## 7. Code, query, or calculation

The three blocks below use the Python standard library only. They implement the tile arithmetic, the storage arithmetic, and the magnification calculation used above.

### 7.1 Tile arithmetic

```python
W, H = 100_000, 100_000
tile = 256
for stride in (256, 128):
    nx = (W - tile) // stride + 1
    ny = (H - tile) // stride + 1
    print(f"stride={stride:>3} px  tiles={nx*ny:>8,}  overlap={(1-stride/tile):.0%}")
print()
tissue = 0.35
nx = (W - tile) // 256 + 1
ny = (H - tile) // 256 + 1
print(f"Full grid at stride 256: {nx*ny:,} tiles")
print(f"Tissue tiles at 35% tissue: {round(nx*ny*tissue):,}")
```

Expected output:

```text
stride=256 px  tiles= 152,100  overlap=0%
stride=128 px  tiles= 608,400  overlap=50%

Full grid at stride 256: 152,100 tiles
Tissue tiles at 35% tissue: 53,235
```

A 50 percent overlap multiplies the tile count by four. The compute budget follows the tile count, not the slide count.

### 7.2 Storage arithmetic

```python
pixels = 100_000 * 100_000
bytes_rgb = 3
raw_gb = pixels * bytes_rgb / 1e9
print(f"Pixels: {pixels:,} ({pixels/1e9:.0f} gigapixels)")
print(f"Raw RGB: {raw_gb:.1f} GB")
for ratio in (8, 10, 20):
    print(f"Compressed at {ratio}:1 -> {raw_gb/ratio:.2f} GB")
print()
print("A 10-gigapixel slide is 10^10 pixels. Storage planning starts there.")
```

Expected output:

```text
Pixels: 10,000,000,000 (10 gigapixels)
Raw RGB: 30.0 GB
Compressed at 8:1 -> 3.75 GB
Compressed at 10:1 -> 3.00 GB
Compressed at 20:1 -> 1.50 GB

A 10-gigapixel slide is 10^10 pixels. Storage planning starts there.
```

The compressed size explains why a CAMELYON slide is about 4 GB rather than 30 GB. The pixel count still governs compute.

### 7.3 Magnification arithmetic

```python
print(f"{'Magnification':>15}{'um/px':>8}{'Pixels per 25 mm':>18}{'Tiles at 256':>15}")
for mag, mpp in ((40, 0.25), (20, 0.5), (10, 1.0)):
    px = int(25_000 / mpp)
    tiles = (px // 256) ** 2
    print(f"{mag:>13}x{mpp:>8.2f}{px:>18,}{tiles:>15,}")
print()
print("Halving magnification quarters the pixel count and the tile count.")
```

Expected output:

```text
  Magnification   um/px  Pixels per 25 mm   Tiles at 256
           40x    0.25           100,000        152,100
           20x    0.50            50,000         38,025
           10x    1.00            25,000          9,409

Halving magnification quarters the pixel count and the tile count.
```

The calculation assumes a square section and ignores edge effects. It is a planning bound, not a measurement. The actual tile count comes from the tissue mask.

## 8. Data, standards, and regulatory context

Pathology has a mature imaging standard and an immature label standard. The pixels are portable; the truth is not.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| DICOM WSI | Standard, increasingly required | Whole-slide image format, metadata, pyramids | Define a diagnostic label |
| DICOM PS3.15 | Standard, often required by policy | De-identification profiles | Remove all burned-in text |
| IHE pathology profiles | Voluntary, often procured | Workflow and exchange conformance | Validate a model |
| ICC colour profiles | Industry standard | Colour consistency across devices | Guarantee stain consistency |
| CAP protocols | Professional standard | Pathology reporting content | Define image labels |
| FDA device regulation | Binding when applicable | Clearance or approval and labelling | Certify fairness |
| EU MDR and IVDR | Binding in the EU | Device classification and clinical evidence | Address dataset licensing |
| EU AI Act | Binding, phased | Risk management and documentation | Replace device regulation |
| HIPAA | Binding in the US | PHI handling | Grant commercial rights |
| GDPR | Binding in the EU | Personal data processing | Define a label standard |
| TCGA terms | Contractual | Use of TCGA slides and linked data | Provide a uniform licence across collections |

**DICOM WSI is not a quality standard.** It defines how the pyramid and metadata are stored. It does not define tissue quality, stain protocol, or scanner calibration. Two compliant slides can be clinically incomparable.

**Colour is a measurement.** H&E staining, scanner colour calibration, and display settings change the appearance of the tissue. A model can learn the colour profile. ICC profiles help within a controlled chain and do not remove cross-laboratory variation.

**The label standard is the protocol.** There is no DICOM equivalent for "metastasis present." The annotation protocol, the reader credentials, the adjudication rule, and the agreement statistic are the standard. They must be published with the dataset.

### Metadata fields that determine transportability

| Field | Why it matters | Common omission |
|---|---|---|
| Scanner manufacturer and model | Sharpness and colour | Removed during de-identification |
| Magnification and micrometres per pixel | Visibility and tile count | Not recorded |
| Stain protocol | Colour and contrast | Missing |
| Section thickness | Nuclear overlap | Rarely recorded |
| Compression | Fine detail | Assumed lossless |
| Scan date | Protocol and scanner change | Shifted or removed |
| Institution | Site shift | Removed for privacy |
| Tissue type | Lesion prevalence | Missing |
| Annotation protocol | Label meaning | Not published |
| Pyramid levels | Model input | Not documented |

### Slide and tissue standards

| Standard | What it covers | Why it matters | Limit |
|---|---|---|---|
| DICOM WSI | Slide storage and metadata | Interoperability | Not a quality standard |
| DICOM PS3.15 | De-identification | Privacy | Not a stain standard |
| ICC profiles | Colour consistency | Display and scanning | Not cross-laboratory |
| CAP protocols | Pathology reporting | Clinical content | Not image labels |
| SNOMED CT | Clinical terminology | Diagnosis coding | Not a slide label |
| LOINC | Observations | Test identity | Not a tissue label |

A slide can be DICOM-compliant, colour-managed, and clinically uninterpretable. The standards make it portable; the pathologist makes it true.

### Colour and display chain

| Element | Variability | Effect | Control |
|---|---|---|---|
| Stain lot | Batch chemistry | Colour and intensity | Record lot |
| Staining protocol | Laboratory practice | Contrast | Protocol documentation |
| Scanner lamp | Age and model | Colour temperature | Calibration |
| Sensor | Model | Noise and colour | Device metadata |
| Compression | Encoder settings | Fine detail | Use original |
| Display | Monitor profile | Perception | ICC profile |
| Viewer software | Rendering | Appearance | Validated viewer |

The chain is longer than most teams assume. A model trained on one chain can learn the chain, and a deployment on another chain is a domain shift.

### Regulatory artefacts for a pathology product

- **Intended use** names the tissue, stain, scanner, population, and clinical decision.
- **Reference standard** describes the pathologists, the protocol, and the agreement.
- **Representativeness** describes the centers, scanners, and tissue mix.
- **Validation** describes the split, the metric, the operating point, and the subgroups.
- **Human factors** describes the review workflow and the override path.
- **Monitoring** describes scanner and stain drift.

A pathology model that reports tile accuracy and no reader or workflow evidence has a technical result and an incomplete file.

## 9. Industry practice and case evidence

### What organisations actually do

- **Separate slide-level and lesion-level claims.** Mature teams build two evaluation sets and do not substitute one for the other.
- **Hold out a center or scanner.** Transport is tested, not assumed.
- **Store features, not only tiles.** A feature store reduces storage and still supports model iteration.
- **Version the tissue mask.** A mask change changes the tile set and the result.
- **Audit annotation scale.** Coordinates at the wrong pyramid level are a common defect.
- **Check licences per collection.** A repository is not a licence.
- **Run a reader study.** A workflow claim needs a pathologist in the loop.

### Case evidence

| Case | Documented fact | Inference |
|---|---|---|
| CAMELYON17 | The dataset contains 1,000 WSIs from 200 patients across 5 centers, with 500 training and 500 test slides, about 4 GB per slide and about 2.3 TB total, released under CC0. | A public dataset can be large, multi-center, and commercially usable. It still provides lesion annotations for only a small subset. |
| CAMELYON16 | The 2016 challenge released 400 slides with 159 annotated images for lesion-level detection. | Lesion-level truth is a subset, and the detection claim is limited to it. |
| PANDA | The dataset contains 10,616 prostate biopsies, about 383 GB, under CC BY-SA-NC 4.0. | A large, well-curated dataset can be non-commercial, which blocks a product. |
| MIDOG | The challenge focuses on mitosis detection across scanners. | Scanner shift is the point of the benchmark, not an inconvenience. |
| TCGA WSI | Diagnostic slides are linked to TCGA clinical and molecular data. | The collection supports multi-modal research and is not a uniform protocol. |

The first three rows are stable and verified. Challenge editions change; verify the current page before quoting.

### What good looks like

- The label level is stated before the model is chosen.
- The split is by patient and center.
- The tile sampling policy is fixed at evaluation.
- The annotation protocol and agreement are published.
- The scanner and stain are recorded for every slide.
- The storage arithmetic is in the project plan.
- The licence is checked per collection.
- The data card states the prohibited claims.

### Workflow and throughput

| Metric | What it measures | Typical constraint | Failure |
|---|---|---|---|
| Scan time | Digitising a slide | Scanner capacity | Queue |
| Inference time | Model per slide | GPU capacity | Delay |
| Review time | Pathologist per slide | Staffing | Bottleneck |
| Turnaround | Scan to report | Laboratory service level | Missed target |
| Queue length | Slides waiting | Capacity | Backlog |
| Override rate | Pathologist disagreement | Trust | Tool ignored |
| Alert precision | True alerts per 100 | Review burden | Alert fatigue |

A model that is accurate and slower than the existing workflow will not be used. The throughput test belongs in the pilot, not after launch.

### Documented fact versus inference

- **Documented fact:** a challenge data page, a dataset paper, a licence, or a scanner specification.
- **Inference:** a planning consequence drawn for another workflow.
- **Unverified summary:** a secondary account of a count or an edition.

A case table should carry a source and a status. Pathology editions and mirrors change, and a stale download page can misstate both the scale and the terms.

### Procurement questions for a pathology product

- Which scanners, stains, and tissue types were used for training and validation?
- What label level was evaluated, and at what review capacity?
- What is the slide-level sensitivity and the false positives per 100 slides?
- What is the lesion-level sensitivity for micrometastases?
- How does performance change on a new scanner or laboratory?
- What is the inference time per slide and the throughput?
- What happens when the tissue mask or focus quality fails?
- What is the monitoring plan for stain and scanner drift?

A vendor that cannot answer these has a benchmark, not a product.

### Why pathology benchmarks mislead procurement

A buyer asks whether the tool works on their scanner, their stain, and their tissue mix, and whether it saves pathologist time. A leaderboard answers a narrower question: how well does a method rank on a fixed test set from a different mix of centers. The gap is a difference in estimand, and it is closed only by a local validation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Tile-level leakage | Test performance is unusually high | Adjacent tiles across splits | Memorisation | Split by slide and patient |
| Slide-level label used for detection | No lesion coordinates | Label level mismatch | Unsupported detection claim | Lesion annotations or a narrower claim |
| Positive tile dilution | Positive tiles are a tiny fraction | Rare, small lesions | Model learns background | Sampling policy and lesion labels |
| Scanner shortcut | Performance drops at a new center | One scanner dominates | Site-specific model | Center holdout |
| Stain shortcut | Colour predicts the label | Protocol confounded with disease | False association | Stain normalisation and holdout |
| Wrong annotation scale | Coordinates do not align | Pyramid level mismatch | Mislabeled regions | Scale audit |
| Tissue mask error | Background tiles in training | Mask threshold | Noise and compute waste | Mask review |
| Annotation cost underestimated | Fewer labels than planned | Pathologist time | Smaller labelled set | Cost arithmetic |
| Licence assumption | "Open" treated as commercial | Repository conflated with licence | Product blocked | Clause review |
| Storage underestimated | Disk fills during download | Terabyte-scale data | Project delay | Storage arithmetic |
| Compression artefacts | Fine detail lost | Lossy compression | Missed micrometastasis | Re-scan or use original |
| Class imbalance ignored | Accuracy looks high | Rare positives | Useless sensitivity | Sensitivity at capacity |
| Test-set reuse | Same slides across papers | Leaderboard tuning | Inflated results | Private holdout |
| Foundation-model overclaim | Linear probe reported as validation | Benchmark task mismatch | Unsupported clinical claim | Reader and workflow study |
| Pyramid-level mismatch | Coordinates do not align | Annotation at a different level | Wrong regions labelled | Overlay audit |
| Tissue mask drift | Tile counts change after a mask update | Pipeline version change | Results not reproducible | Version the mask |
| Focus failure | Blurred regions misclassified | Scanner or tissue issue | False negatives | Focus filtering |
| Coverslip artefact | Border tiles contain glass | Tile extraction ignores border | Spurious features | Border exclusion |
| Scan-time inflation | Turnaround exceeds the workflow | Large slides and slow scanners | Tool cannot be used | Throughput test |
| Label drift | Grading criteria change over time | Guideline update | Cohort non-comparable | Version the criteria |
| Slide mix drift | Case mix changes by year | Referral patterns | Performance shifts | Temporal holdout |
| Stain lot change | Colour shifts between lots | Reagent variability | Domain shift | Record lot and normalise |
| Viewer difference | Display changes appearance | Rendering | Perception difference | Validated viewer |

Red flags that should stop a pathology project:

- The split unit is a tile.
- The label is slide-level and the claim is lesion-level.
- The scanner and stain are not recorded.
- The licence is "open" with no clause.
- The storage plan is missing.
- The annotation protocol and agreement are missing.
- The evaluation has no center holdout.
- The model is a foundation model with no local validation.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The slide manifest is complete | Counting depends on it | Missing files | Reconcile manifest and store |
| The patient key is correct | Splitting depends on it | Identifier changes | Linkage audit |
| The label level matches the claim | Evaluation depends on it | Slide labels for detection | Label audit |
| The annotation scale is correct | Coordinates depend on it | Pyramid mismatch | Overlay check |
| The tissue mask is accurate | Tile counts depend on it | Threshold error | Mask review |
| The scanner mix matches deployment | Transport depends on it | New scanner | Center holdout |
| The stain protocol matches | Colour depends on it | Different protocol | Stain audit |
| The storage estimate is adequate | The project depends on it | Terabyte data | Capacity plan |
| The licence covers the use | Product depends on it | Non-commercial terms | Clause review |
| The reader agreement is adequate | The ceiling depends on it | Ambiguous lesions | Adjudication study |
| The test set is clean | Validity depends on it | Duplicate slides | Hash and probe |
| The subgroup has enough lesions | Equity analysis depends on it | Small counts | Confidence intervals |
| The pyramid level is consistent | Annotation alignment depends on it | Mixed levels | Overlay audit |
| The tissue mask is versioned | Reproducibility depends on it | Mask update | Version record |
| The scanner throughput is adequate | Deployment depends on it | Large slides | Throughput test |
| The grading criteria are stable | Comparability depends on it | Guideline change | Criteria version |
| The review capacity is real | The metric depends on it | Staffing assumptions | Workflow study |
| The stain lot is recorded | Domain shift depends on it | Reagent variability | Lot metadata |
| The viewer is validated | Display consistency depends on it | Rendering differences | Viewer check |
| The case mix is stable | Comparability depends on it | Referral change | Temporal audit |

The deepest limit is that slide-level performance does not establish clinical benefit. A triage tool can improve review order and fail to improve diagnosis if the pathologist overrides it or the queue is not the bottleneck.

## 12. Exercises

**Exercise 1. Compute the slide arithmetic.**
Deliverable: pixels, tiles, tissue tiles, storage, and inference time for a stated slide size and magnification.
Acceptance criteria: formulas shown; assumptions stated; one sensitivity case at half the magnification.
Time: 40 minutes.

**Exercise 2. Classify label levels.**
Deliverable: a table mapping four claims to patient, slide, region, lesion, or cell labels.
Acceptance criteria: four claims; one mismatch identified; the evaluation level stated.
Time: 30 minutes.

**Exercise 3. Audit a split.**
Deliverable: a script or spreadsheet computing slide and patient overlap and the share of leaked test tiles.
Acceptance criteria: reproduces the logic of section 7; proposes a corrected split; names affected patients.
Time: 50 minutes.

**Exercise 4. Estimate annotation cost.**
Deliverable: hours and cost for 500 lesion annotations, including adjudication and quality control.
Acceptance criteria: time per case sourced; overhead included; the ceiling implication stated.
Time: 35 minutes.

**Exercise 5. Map a licence.**
Deliverable: a clause matrix for CAMELYON17 and PANDA covering training, evaluation, model release, and deployment.
Acceptance criteria: both datasets; each cell marked; the blocking clause named.
Time: 45 minutes.

**Exercise 6. Design tile sampling.**
Deliverable: a sampling policy for training and a separate fixed-grid policy for evaluation.
Acceptance criteria: no label leakage at evaluation; rare lesions preserved; compute estimated.
Time: 45 minutes.

**Exercise 7. Evaluate scanner shift.**
Deliverable: a center-holdout plan with per-center metrics and confidence intervals.
Acceptance criteria: holdout named; metric level stated; a failure threshold proposed.
Time: 40 minutes.

**Assessment note.** Exercises 1, 3, 4, and 5 produce checkable artefacts: an arithmetic sheet, a leakage report, a cost table, and a clause matrix. Exercises 2, 6, 7, and 8 are design tasks and are marked against the rubric. Exercise 8 is the integration task.

**Exercise 8. Write a pathology data card.**
Deliverable: a two-page card with magnification, label level, annotation protocol, licence, storage, and prohibited claims.
Acceptance criteria: every field populated or marked unknown; one prohibited claim; versions recorded.
Time: 60 minutes.

## 13. Self-check answers

**Item 1. Why is a tile not a case?**
Tiles from the same slide share tissue, stain, and scanner. Treating them as independent inflates the sample size and leaks the slide into the test set.

**Item 2. What is the difference between a slide-level and a lesion-level label?**
A slide-level label says whether the slide contains a finding; a lesion-level label gives its coordinates. Only the latter supports detection and localisation claims.

**Item 3. Why is a patient-level split stronger than a slide-level split?**
A patient can have multiple slides, and the same tumour and acquisition can appear in both. Grouping by patient prevents that leakage.

**Item 4. A 25 mm section at 0.25 µm/px is how many pixels per side?**
25 mm is 25,000 µm. Divided by 0.25 µm/px gives 100,000 pixels. The image is 100,000 by 100,000, or 10 gigapixels.

**Item 5. How many 256-pixel tiles in that image without overlap?**
About 391 by 391, or roughly 152,000 tiles. With a 50 percent overlap, about four times as many.

**Item 6. Why does halving magnification quarter the tile count?**
Halving magnification doubles the micrometres per pixel, so each side has half the pixels. The area, and the tile count, falls by a factor of four.

**Item 7. Why is a micrometastasis hard to detect with a slide-level label?**
The lesion occupies a tiny fraction of the slide, so most tiles are negative. A slide-level label does not tell the model where the lesion is, and uniform sampling rarely shows it.

**Item 8. What does CC0 permit that CC BY-SA-NC does not?**
CC0 permits commercial use and redistribution without attribution conditions. CC BY-SA-NC permits non-commercial use with attribution and share-alike, and blocks a commercial product.

**Item 9. Why is stain normalisation not a complete solution to domain shift?**
It reduces colour variation and does not remove scanner sharpness, section thickness, or protocol differences. A center holdout is still required.

**Item 10. When is a foundation model acceptable as a baseline?**
As a feature extractor or pretrained initialization, with a local validation and a simple baseline for comparison. It is not clinical validation.

**Item 11. Why is a slide-level AUROC not a triage metric?**
Triage has a fixed review capacity, so the metric must be sensitivity at the capacity, not discrimination across all thresholds. A model with a high AUROC can perform poorly at the 30 percent review rate.

**Item 12. What is the difference between a micrometastasis and an isolated tumour cell cluster?**
A micrometastasis is a small but established deposit, typically defined by size; isolated tumour cells are single cells or small clusters below the micrometastasis threshold. The distinction changes staging and the label. A model trained on one may not detect the other.

**Item 13. Why is annotation density a ceiling?**
Only annotated regions can be used as positive examples. If annotations cover a small fraction of the slide, the model sees few positives and cannot learn the full appearance of the lesion.

**Item 14. What does a center holdout measure that a patient split does not?**
It measures transport across scanner, stain, and practice. A patient split within one center measures generalization to new patients on the same devices.

**Item 15. Why is a whole-slide feature store a compromise?**
It reduces storage and accelerates iteration, and it fixes the feature extractor. A new architecture may need the pixels again, and the features are derived data governed by the licence.

**Item 16. What is the first check on a pathology manifest?**
The unit and the counts: patients, slides, annotations, and centers. Then the label level and the scanner fields. A strong answer checks the annotation scale before modeling. A weak answer starts with a tile count.

**Item 17. Why can slide-level accuracy be high while lesion sensitivity is zero?**
A slide is positive if any lesion is present. A model that predicts negative for all slides with small lesions can still be correct on most slides if lesions are rare, and it misses every lesion. Accuracy hides the failure.

**Item 18. What is the difference between a scan artefact and a tissue artefact?**
A scan artefact comes from the device or the digitisation, such as blur or compression. A tissue artefact comes from the specimen, such as folds or fixation. Both can create false structures and require different checks.

**Item 19. Why is a private test set useful in pathology challenges?**
It limits repeated tuning against the evaluation and reduces leaderboard overfitting. It does not repair label error or scanner shift. A strong answer separates the two threats.

**Item 20. Why does a change in stain lot matter for a deployed model?**
A new reagent lot can shift colour and intensity, which changes the input distribution. A model that learned the previous lot's appearance can degrade. Recording the lot and testing across lots is part of monitoring.

**Item 21. What is the difference between a temporal holdout and a center holdout?**
A temporal holdout tests generalization to a later period, where protocol and case mix may change. A center holdout tests generalization to a different site, scanner, and practice. They answer different questions and are often combined.

**Rubric for the design exercises (3, 5, 6, 8).** Full marks require a named label level, a patient-level split, a licence clause, and a prohibited-claim statement. Partial marks for a complete structure without the storage arithmetic. No marks for a tile-level evaluation presented as slide-level generalization.

## 14. Interview questions

**1. How would you split a whole-slide dataset?**
By patient, then by center or scanner, with a temporal holdout if dates exist. A strong answer explains what each split estimates. A weak answer says random tiles.

**2. Why is tile-level accuracy misleading?**
Tiles are correlated and imbalanced. A model can achieve high tile accuracy by predicting background and still miss every lesion. A strong answer proposes slide-level and lesion-level metrics. A weak answer reports accuracy.

**3. A dataset has patient-level pN stage. Can you build a detection model?**
Not across the whole dataset. Detection needs lesion coordinates. A strong answer limits the claim to the annotated subset. A weak answer treats the stage as a lesion label.

**4. How do you estimate storage for 1,000 slides?**
Pixels per slide, bytes per pixel, compression ratio, and file size per slide, then multiply and add backups and derived artefacts. A strong answer gives a range. A weak answer guesses.

**5. What does a non-commercial pathology licence prevent?**
Product training, deployment, model release, and possibly internal commercial demos, depending on the clause. A strong answer maps activities to clauses. A weak answer says it prevents selling images.

**6. How do you handle scanner shift?**
Hold out a scanner or center, report per-center metrics, and test stain normalisation. A strong answer treats the drop as the finding. A weak answer pools all centers.

**7. What is the annotation bottleneck?**
The pathologist time required for lesion-level labels. A strong answer quantifies hours per case and the resulting sample size. A weak answer says annotation is expensive.

**8. Why is a foundation model not a validation?**
It is a pretrained feature extractor or initialization. Its benchmark performance does not establish performance on the target scanner, stain, and workflow. A strong answer requires local validation.

**9. How do you audit annotation scale?**
Overlay the coordinates on the slide at the stated pyramid level and check alignment. A strong answer samples several slides. A weak answer trusts the file.

**10. What would you refuse to claim from a slide-level dataset?**
Lesion detection, lesion size, and spatial distribution. A strong answer names the label-level limit. A weak answer reports detection from slide labels.

**11. How do you report a negative result?**
Report the metric, the confidence interval, the cohort, and the label level. A strong answer distinguishes absence of evidence from evidence of absence. A weak answer reports only a p-value.

**12. What belongs in a pathology data card?**
Magnification, micrometres per pixel, scanner, stain, label level, annotation protocol, licence, storage, and prohibited claims. A strong answer records versions. A weak answer lists the slide count.

**13. How do you handle a dataset with no lesion annotations?**
Use it for slide-level classification or pretraining, not detection. A strong answer states the claim that remains. A weak answer treats the slide label as a lesion label.

**14. What would you ask a vendor who reports 0.98 AUROC on CAMELYON17?**
The split unit, the label level, the center holdout, the operating point, the lesion-level sensitivity, and the scanner mix. A strong answer asks for the evaluation script. A weak answer asks for the slide count.

**15. Why is a 2.3 TB download a project risk?**
It consumes storage, transfer time, and egress budget before any model is trained. A strong answer plans the transfer and the cache. A weak answer assumes a fast link.

**16. How do you report a slide-level result to a pathologist?**
Sensitivity and specificity at the review capacity, the false positives per 100 slides, and the missed cases by lesion size. A strong answer includes the reader workflow. A weak answer reports accuracy.

**17. What is the difference between stain normalisation and scanner calibration?**
Stain normalisation adjusts colour across slides; scanner calibration adjusts the device's colour and intensity response. Neither removes section thickness or protocol differences.

**18. When would you refuse a pathology dataset?**
When the label level cannot support the claim, the licence blocks the use, the scanner mix is irrelevant, or the annotations are too sparse. A strong answer treats refusal as a result. A weak answer uses it because it is large.

**19. How do you validate a whole-slide model on a new scanner?**
Collect a sample from the new scanner, run the same pipeline, and report slide-level and lesion-level metrics with intervals. A strong answer treats the scanner as a domain. A weak answer normalises the colours and assumes the problem is solved.

**20. What is the annotation bottleneck in practice?**
Pathologist time. A lesion polygon takes minutes to tens of minutes, and adjudication adds more. A strong answer quantifies the hours and the sample size. A weak answer says annotations are expensive.

**21. Why is a tile cache a derived artefact?**
It is generated from the slides and may be governed by the same licence. A strong answer treats it as data for retention and deletion. A weak answer deletes only the source.

**22. What would you put in a pathology model card?**
Tissue, stain, scanner, magnification, label level, cohort, split, metrics, subgroups, calibration, limitations, and monitoring. A strong answer names the prohibited uses. A weak answer lists the architecture.

**22. How do you handle a dataset with mixed pyramid levels?**
Normalise the annotations to a single level, verify with overlays, and record the conversion. A strong answer treats the scale as metadata. A weak answer assumes the files are aligned.

**23. What is the first arithmetic you do on a new slide dataset?**
Pixels per slide, tiles per slide, tissue tiles, and storage. A strong answer does it before downloading. A weak answer starts the download and discovers the storage cost later.

**24. When is a slide-level model clinically useful?**
When the decision is whether to prioritise or review a slide, and the evaluation is at the review capacity. A strong answer ties the metric to the workflow. A weak answer reports AUROC without a capacity.

## 15. Key terms

- **Annotation** — marking regions or labels on a slide.
- **Adjudication** — resolving reader disagreement.
- **CAMELYON16** — lymph node metastasis detection challenge.
- **CAMELYON17** — patient-level lymph node staging challenge.
- **Compression** — encoding of image data.
- **Dice** — overlap between predicted and reference regions.
- **DICOM WSI** — standard for whole-slide images.
- **Free-response ROC** — detection metric with localisation.
- **H&E** — haematoxylin and eosin stain.
- **Inter-pathologist agreement** — agreement between pathologists.
- **ISUP grade** — prostate cancer grade group.
- **Lesion-level label** — coordinates and class of a lesion.
- **Magnification** — objective power.
- **Micrometres per pixel** — physical pixel size.
- **MIDOG** — mitosis detection challenge.
- **pN stage** — lymph node stage.
- **PANDA** — prostate biopsy grading challenge.
- **Patient-level label** — one label per patient.
- **Pyramid** — multiple resolutions of a slide.
- **Quadratic weighted kappa** — agreement for ordinal grades.
- **Slide-level label** — one label per slide.
- **Stain** — dye protocol.
- **Stain normalisation** — colour correction across slides.
- **Stride** — step between tiles.
- **TCGA WSI** — diagnostic slides linked to TCGA.
- **Tile** — a patch extracted from a slide.
- **Tissue mask** — mask separating tissue from background.
- **Weak label** — label derived from reports or registries.
- **Whole-slide image** — gigapixel image of a tissue slide.
- **WSI** — whole-slide image.
- **Air bubble** — occlusion under the coverslip.
- **Annotation density** — annotated area per slide.
- **Coverslip** — glass covering the tissue.
- **Feature store** — saved embeddings for tiles.
- **Fixation time** — time before tissue processing.
- **Focus quality** — sharpness of a scanned image.
- **Isolated tumour cells** — single cells or small clusters below the micrometastasis threshold.
- **Micrometastasis** — a small established metastatic deposit.
- **Pen mark** — ink on the slide.
- **Pyramid level** — one resolution in a slide pyramid.
- **Review capacity** — slides a pathologist can read per session.
- **Scan time** — time to digitise a slide.
- **Section thickness** — thickness of the tissue slice.
- **Throughput** — slides processed per hour.
- **Tissue fold** — folded tissue creating false structures.
- **Turnaround time** — time from scan to report.
- **Whole-slide feature** — an embedding summarising a slide.
- **Barcode** — a slide identifier.
- **Case** — a patient episode in pathology.
- **Digitisation** — scanning a glass slide.
- **Glass slide** — the physical specimen.
- **Grading** — assigning an ordinal severity.
- **Histology** — tissue structure.
- **Immunohistochemistry** — antibody-based staining.
- **Margin** — the tissue edge relative to a lesion.
- **Metastasis** — a secondary tumour deposit.
- **Normalisation** — colour or intensity correction.
- **Region of interest** — an area selected for analysis.
- **Staging** — assigning disease extent.
- **Tumour grade** — differentiation or aggressiveness.
- **Virtual slide** — a digitised slide.
- **Case mix** — the composition of cases in a cohort.
- **Domain shift** — change in image distribution across sites.
- **Feature extractor** — a model that converts tiles to embeddings.
- **Slide-level summary** — an aggregate representation of a slide.
- **Stain lot** — a batch of stain reagent.
- **Throughput test** — a measurement of slides per hour.
- **Turnaround target** — the service-level time from scan to report.
- **Biopsy** — a tissue sample taken for diagnosis.
- **Carcinoma** — a cancer arising from epithelial tissue.
- **Cytology** — the study of individual cells.
- **Excision** — surgical removal of tissue.
- **Frozen section** — a rapid intraoperative tissue preparation.
- **Grade group** — an ordinal prostate cancer grade category.
- **Histopathology** — microscopic study of diseased tissue.
- **Lymph node** — an immune organ often sampled for staging.
- **Margins** — the tissue edge relative to a tumour.
- **Morphology** — the shape and structure of tissue.
- **Necrosis** — cell death in tissue.
- **Resection** — surgical removal of a tumour.
- **Sentinel node** — the first lymph node draining a tumour.
- **Slide scanner** — a device that digitises glass slides.
- **Stain protocol** — the procedure for applying dyes.
- **Tumour bed** — the tissue area after tumour removal.
- **Whole-mount** — a large tissue section.
- **Annotation tool** — software for marking images.
- **Consensus annotation** — a label agreed by multiple readers.
- **Label noise** — error in the reference label.
- **Reference standard** — the best available truth.
- **Slide manifest** — the list of slides and their metadata.

## 16. Sources

All URLs accessed 26 October 2026. Editions and terms change; verify at the source.

1. Grand Challenge. CAMELYON17 data. https://camelyon17.grand-challenge.org/Data/
2. Bándi P, Geessink O, Manson Q, et al. From detection of individual metastases to classification of lymph node status at the patient level: the CAMELYON17 challenge. *IEEE Transactions on Medical Imaging* 2019;38(2):550-560. https://doi.org/10.1109/TMI.2018.2867350
3. Grand Challenge. CAMELYON16. https://camelyon16.grand-challenge.org/
4. Grand Challenge. PANDA data. https://panda.grand-challenge.org/data/
5. Bulten W, Kartasalo K, Chen PC, et al. Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge. *Nature Medicine* 2022. https://doi.org/10.1038/s41591-021-01620-2
6. National Cancer Institute. TCGA. https://www.cancer.gov/ccg/research/genome-sequencing/tcga
7. Genomic Data Commons. https://portal.gdc.cancer.gov/
8. MIDOG challenge. https://midog2025.deepmicroscopy.org/
9. NEMA. DICOM whole-slide imaging. https://www.dicomstandard.org/
10. DICOM PS3.15 security profiles. https://dicom.nema.org/medical/dicom/current/output/chtml/part15/PS3_15.html
11. College of American Pathologists. Cancer protocols. https://www.cap.org/protocols-and-guidelines
12. International Color Consortium. https://www.color.org/
13. US Food and Drug Administration. AI-enabled device software functions guidance, draft January 2025. https://www.fda.gov/media/184856/download
14. European Parliament and Council. Regulation (EU) 2017/745 (MDR). https://eur-lex.europa.eu/eli/reg/2017/745/oj
15. The Cancer Imaging Archive. Collections. https://www.cancerimagingarchive.net/collections/
16. Grand Challenge. CAMELYON16 evaluation. https://camelyon16.grand-challenge.org/Evaluation/
17. Synapse. MIDOG 2025. https://www.synapse.org/
18. International Electrotechnical Commission. IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/publication/22794
19. Module 21 survey file, sections 6 and 9. [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)

## 17. Links

- Module map and topic table: [`_index.md`](_index.md)
- Previous topic, omics datasets: [`06_omics-datasets.md`](06_omics-datasets.md)
- Imaging datasets: [`04_imaging-datasets.md`](04_imaging-datasets.md)
- Access and licensing: [`02_access-and-licensing.md`](02_access-and-licensing.md)
- Root survey file: [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
- Labs, including the dataset audit: [`../../labs/`](../../labs/)

