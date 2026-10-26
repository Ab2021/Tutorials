# Imaging Datasets

> Module 21 — Datasets, Tools, and Benchmarks
> Topic 04 of 12
> Estimated study time: 4 hours
> Prerequisites: Topic 01, Topic 02
> Version 2026.3

## 1. Why this topic exists

Medical imaging is the modality where a public benchmark looks most like a product and is least like one. A chest radiograph dataset with hundreds of thousands of images and a leaderboard invites the conclusion that the model is nearly deployable. The gap between the benchmark and the product is made of labels, scanners, patients, acquisition protocols, and the clinical decision the image is supposed to support. Each gap has a name, and each can be audited before a model is trained.

The failures are specific:

- A model is trained and tested on images from the same patients, because the split was made by image rather than by patient. The reported performance reflects memorised anatomy.
- A model learns the scanner or the acquisition protocol rather than the pathology, because one site dominates the training set and the test set comes from the same site.
- A label extracted from a radiology report is treated as ground truth, and the model learns the reporting style of the institution.
- A segmentation model reports a Dice score that improves while the clinical endpoint, such as radiotherapy plan quality, does not.
- A challenge dataset is downloaded without reading the terms, and the commercial product cannot use it.
- A new edition of a challenge changes the label definition, and a result from the previous edition is compared with the new leaderboard.

Imaging data are expensive to annotate, expensive to store, and governed by a layered set of licences. The annotation bottleneck is the binding constraint in most projects: images are abundant, and adjudicated labels are scarce. A dataset with a million images and report-derived labels is a weak-label dataset, not a million-case truth set.

This topic covers the major public and credentialed imaging collections, the label provenance problem, the split hierarchy, and the evaluation choices that determine whether a number means anything. It treats imaging as a measurement problem first and a modeling problem second. The survey file lists the collections; this file teaches how to decide whether one can support a claim.

The cost asymmetry is the reason imaging projects fail late. A dataset can be downloaded in an afternoon, and the annotation, reader study, and external validation that make it useful take a year. The cheap step creates the impression that the expensive steps are optional. They are the evidence.

A final framing point: an image is not a diagnosis. It is a measurement made by a device under a protocol, interpreted by a human, and recorded in a report. Every dataset in this topic is a record of that chain, not a record of disease. The chain is what the model learns, and the chain is what the evaluation must test.

> **Note:** Counts and access terms were checked on 26 October 2026. BraTS 2025 data access was in transition to The Cancer Imaging Archive, and new access was reported as unavailable at the time of checking. Challenge access changes quickly; verify before planning.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** image-level, study-level, and patient-level units, and choose the unit that matches the claim. Assessed by Exercise 1 and self-check item 3.
2. **Audit** a split for patient, study, site, and temporal leakage, and quantify the overlap. Assessed by Exercise 2.
3. **Trace** an imaging label to its source and estimate agreement with a reference reader. Assessed by Exercise 3.
4. **Evaluate** whether a reported metric, such as Dice or AUROC, corresponds to the clinical decision. Assessed by Exercise 4.
5. **Compare** at least four imaging collections on access, licence, label provenance, and population. Assessed by Exercise 5.
6. **Design** a preprocessing and evaluation pipeline that fits preprocessing on training data only and reports site-level results. Assessed by Exercise 6.
7. **Detect** edition drift and scanner shift between development and evaluation data. Assessed by Exercise 7.
8. **Write** an imaging data card that states what the collection cannot support. Assessed by Exercise 8.

## 3. The core mental model

An imaging dataset is a stack of four layers. The pixel is at the bottom; the clinical claim is at the top. A failure at any layer propagates upward, and the metric at the top can look healthy while the layer below is broken.

```text
   CLINICAL CLAIM
   diagnosis | triage | segmentation | radiotherapy plan | screening
        ^
        |  metric must match the claim
        |
   LABEL
   adjudicated | report-derived | rule-based | weak | missing
        ^
        |  provenance and agreement
        |
   STUDY AND PATIENT
   one patient -> many studies -> many series -> many images
        ^
        |  split at the patient or site level
        |
   PIXELS AND METADATA
   scanner | protocol | reconstruction | orientation | de-identification
```

**The unit hierarchy.** A patient has studies; a study has series; a series has images. A chest radiograph study usually has one or two images. A CT or MRI study has hundreds. A whole-slide image is a single gigapixel file with tiles. Counting images as cases inflates the sample size by the number of images per patient and leaks identity across splits. The unit of the claim is almost always the patient or the study, not the image.

**The label ladder.** Labels range from adjudicated multi-reader consensus at the top to report text at the bottom. The higher the label, the smaller the dataset and the stronger the claim. A report-derived label is a measurement of what the radiologist wrote, which correlates with disease and is not the same as disease. The correlation varies by finding, institution, and reporting template.

**The split hierarchy.** The safest split holds out patients; the next holds out sites; the strongest holds out both and adds a temporal holdout. A random image split is the weakest and the most common. The hierarchy is not a preference; it determines whether the reported number is an estimate of generalization or an estimate of memorisation.

**The preprocessing rule.** Intensity normalisation, resampling, cropping, and augmentation are part of the model. If preprocessing is fit on the full dataset, test information leaks into training. Fit on training data, apply to test, and publish the parameters.

**The edition rule.** Challenge datasets change between editions. The label set, the number of classes, the annotation protocol, and the test distribution can all differ. A result from one edition is not comparable with another edition's leaderboard unless the task and the split are the same.

| Layer | Question | Artefact | Failure if skipped |
|---|---|---|---|
| Pixels | Is the image what it claims to be? | Metadata inventory | Wrong orientation, burned-in text, duplicate images |
| Study and patient | What is the unit, and who is in both splits? | Split specification with patient keys | Leakage and inflated performance |
| Label | Where did the label come from and how good is it? | Provenance record and agreement estimate | The metric measures the label pipeline |
| Claim | Does the metric correspond to the decision? | Metric-to-decision memo | A good Dice score and a bad product |

## 4. Definitions and vocabulary

### 4.1 Imaging units and formats

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| DICOM | The standard for medical image storage and communication. | Treated as a file format only. | Standard |
| Study | A set of series acquired for one examination. | Confused with a patient encounter. | Study UID |
| Series | A set of images with the same acquisition parameters. | Counted as a study. | Series UID |
| Instance | A single image or frame. | Counted as a patient. | SOP Instance UID |
| Whole-slide image (WSI) | A gigapixel image of a tissue slide. | Treated as a normal image. | Pyramid file |
| Tile | A patch extracted from a WSI or a large volume. | Counted as an independent sample. | Pixel region |
| Volume | A three-dimensional image stack. | Treated as independent slices. | Voxel array |
| Pixel spacing | The physical size of a pixel. | Assumed constant across scanners. | Millimetres |
| Slice thickness | The distance between image slices. | Ignored in volumetric comparison. | Millimetres |
| Window and level | Display settings for CT intensity. | Applied inconsistently across sites. | Hounsfield units |
| Hounsfield unit | A calibrated CT attenuation value. | Treated as comparable without calibration. | HU |
| Orientation | The anatomical direction of the image. | Assumed from the file name. | DICOM orientation |

### 4.2 Labels and annotation

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Label | The target variable for supervision or evaluation. | Treated as ground truth. | Variable |
| Label provenance | The origin and process that produced the label. | Recorded as "expert annotated." | Record |
| Adjudication | Resolution of reader disagreement by consensus or a third reader. | Assumed when only one reader annotated. | Process |
| Inter-reader agreement | Agreement between two or more readers. | Reported without the scale. | Kappa or ICC |
| Cohen's kappa | Agreement corrected for chance for two raters. | Used for more than two raters without adjustment. | Statistic |
| Intraclass correlation | Agreement for continuous measurements. | Used for categorical labels. | Statistic |
| Report-derived label | A label extracted from radiology report text. | Treated as a clinical diagnosis. | NLP output |
| Weak label | A noisy or indirect label. | Treated as clean. | Label type |
| Uncertainty label | A label indicating an equivocal mention. | Dropped without analysis. | Class |
| Negative label | The absence of a finding. | Assumed when the report is silent. | Class |
| Test set | Data held out from all model selection. | Reused across submissions. | Split |
| Anchor | A reference case used to calibrate readers. | Omitted from the protocol. | Case |
| Positive predictive value | The probability that a positive label is correct. | Confused with sensitivity. | Proportion |
| Negative predictive value | The probability that a negative label is correct. | Ignored when prevalence is low. | Proportion |
| Prevalence | The proportion of cases with the finding. | Assumed from the dataset. | Proportion |
| Confidence interval | The uncertainty around an estimate. | Omitted for subgroup metrics. | Interval |
| Bootstrap | Resampling used to estimate uncertainty. | Treated as a substitute for external validation. | Method |
| Reader study | A study of clinician performance with and without the model. | Replaced by a held-out metric. | Study design |
| Workflow outcome | Time, edits, or plan changes in real use. | Assumed from accuracy. | Measure |
| Phantom | A physical or digital reference object. | Used to claim clinical performance. | Test object |

### 4.3 Evaluation terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Dice coefficient | Overlap between predicted and reference segmentation. | Used as a clinical utility measure. | 0 to 1 |
| Hausdorff distance | The maximum boundary discrepancy between segmentations. | Ignored when Dice is high. | Millimetres |
| AUROC | Discrimination across all thresholds. | Used for a fixed-capacity decision. | 0 to 1 |
| Sensitivity | The probability that a true positive is detected. | Reported without the operating point. | Proportion |
| Specificity | The probability that a true negative is correctly excluded. | Reported without prevalence. | Proportion |
| Precision | The positive predictive value at an operating point. | Confused with sensitivity. | Proportion |
| Calibration | Agreement between predicted probability and observed frequency. | Omitted in imaging classification. | Slope and intercept |
| External validation | Evaluation on a different site, scanner, or era. | Replaced by a random split. | Study design |
| Domain shift | Change in image distribution across sites or devices. | Assumed away by normalisation. | Distribution change |
| Leaderboard | A public ranking on a hidden test set. | Treated as deployment evidence. | Ranking |

### 4.4 Quality and operations terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Diagnostic quality | Whether an image is adequate for interpretation. | Assumed from resolution | Ordinal score |
| Motion artefact | Blurring from patient movement. | Ignored as random noise | Present or absent |
| Laterality | The side of the body shown. | Assumed from the file name | Left or right |
| Field of view | The anatomy included in the image. | Confused with image size | Millimetres |
| Reconstruction kernel | The filter used to reconstruct CT images. | Ignored as a display setting | Kernel name |
| Contrast phase | The timing of contrast relative to acquisition | Assumed constant | Phase label |
| Acquisition protocol | The parameters used for the scan | Not recorded | Protocol name |
| Series completeness | Whether all expected series are present | Assumed from the study | Count |
| Gradability | Whether an image can be graded for the task. | Excluded silently | Proportion |
| Repeat image | A duplicate acquisition. | Counted as a new case | Flag |

### 4.5 Access and licence terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Credentialed access | Identity, training, and a signed agreement. | Delegated to a team account. | Tier |
| Research use agreement | A licence limited to non-commercial research. | Treated as a commercial licence. | Contract |
| Non-commercial | A restriction on revenue-generating use. | Assumed to exclude internal company research. | Clause |
| Share-alike | A requirement to license derivatives under the same terms. | Ignored for model weights. | Clause |
| Redistribution | Providing images or derivatives to a third party. | Assumed to permit sharing features. | Clause |
| De-identification | Removal of identifiers from images and metadata. | Assumed to remove burned-in text. | Process |
| Data use agreement | A contract governing access and use. | Read as a publication licence. | Contract |
| Challenge terms | Competition-specific rules on data and submissions. | Ignored after the challenge closes. | Rules |

## 5. The mechanics, step by step

The procedure has 18 steps in four phases: define the claim and the unit, audit the images and labels, design the split and preprocessing, and evaluate and report.

### Phase A — Define the claim and the unit

**Step 1. Write the clinical decision the image supports.**
Input: the product or study question and the clinical workflow.
Output: a decision statement with the actor, action, timing, population, and error costs.
Rationale: the decision determines the label, the metric, and the acceptable operating point.
Branch: if the decision is triage, the metric is sensitivity at a fixed review capacity; if it is segmentation for treatment planning, the metric includes boundary accuracy and plan impact.
Edge case: a decision that uses the image plus clinical data is not an image-only task, and an image-only dataset cannot evaluate it.

**Step 2. Choose the unit of analysis.**
Input: the decision statement and the dataset structure.
Output: patient, study, series, image, volume, or tile.
Rationale: the unit determines the denominator, the split key, and the independence assumption.
Branch: use the unit decision table below.
Edge case: a patient with multiple studies over time can support a longitudinal claim only if the split keeps the patient on one side.

**Step 3. Define the target population and the acquisition setting.**
Input: the deployment sites and the imaging protocol.
Output: inclusion criteria for age, indication, modality, contrast, scanner, and protocol.
Rationale: an image distribution is defined by the scanner and the protocol as much as by the patient.
Branch: if the deployment includes portable radiographs and the dataset contains only standing posteroanterior views, the population fit is weak.
Edge case: a model trained on contrast-enhanced CT may fail on non-contrast CT, even for the same anatomy.

**Step 4. Map the label to the claim.**
Input: the decision statement and the available labels.
Output: a label definition, a provenance statement, and the gap between the label and the clinical endpoint.
Rationale: a label can be valid and still not measure the decision. Dice measures overlap, not plan quality.
Branch: if the label is a proxy, the claim must be stated as a proxy claim.
Edge case: a report-derived label can be right for the finding and wrong for the severity, which matters for a treatment decision.

### Modality and claim fit

| Modality | Common claim | Label that fits | Metric | What the image cannot show |
|---|---|---|---|---|
| Radiograph | Detect a finding | Adjudicated or report-derived | Sensitivity at capacity | Soft-tissue detail |
| CT | Detect or measure | Expert annotation | Lesion-level sensitivity | Tissue function |
| MRI | Segment or stage | Expert segmentation | Dice and boundary | Pathology grade without biopsy |
| Mammography | Screen for cancer | Pathology-linked | Recall and interval cancer | Biological behaviour |
| Pathology | Grade or classify | Pathologist diagnosis | Agreement and kappa | Molecular status without assay |
| Ultrasound | Detect or measure | Operator annotation | Agreement | Reproducibility across operators |
| Fundus | Grade retinopathy | Grading scale | Kappa and referral accuracy | Systemic disease without other data |
| Endoscopy | Detect lesion | Video annotation | Per-lesion detection | Histology without biopsy |

### Collection profiles

| Collection | Custodian | Version inspected | Scale | Access and licence | What it cannot answer |
|---|---|---|---|---|---|
| MIMIC-CXR-JPG | PhysioNet and Beth Israel Deaconess | v2.0.0 | 377,110 images; 227,835 studies; 65,379 patients | Credentialed; research DUA | National performance; labels are report-derived |
| CheXpert | Stanford AIMI | Current release | 224,316 images; 65,240 patients | Research use agreement; commercial licence separate | Emergency-department workflow; adjudicated truth |
| NIH ChestX-ray14 | NIH Clinical Center | Current release | 112,120 images; 30,805 patients | Open download | Clean labels; patient-level official split |
| RSNA challenge data | Radiological Society of North America | Edition-specific | Hundreds to thousands of studies per edition | Kaggle or challenge rules; research use | Cross-edition comparability; hidden test access |
| BraTS | MICCAI BraTS consortium | 2025 Lighthouse edition | Multiple tasks; case counts edition-specific | Synapse registration and terms; access in transition to TCIA | Clinical plan quality; commercial deployment |
| CAMELYON16/17 | Radboud and collaborators | 2016 and 2017 editions | Hundreds of slides | Open via Grand Challenge | Multi-cancer scope; scanner robustness |
| PANDA | Radboud and Karolinska | 2020 release | 10,616 biopsies; about 383 GB | CC BY-SA-NC 4.0; test set not public | Commercial use; full Gleason pattern detail in the public label |
| MIDOG | Challenge consortium | Edition-specific | Multiple scanner domains | Challenge registration | Clinical deployment; small domain count |
| TCIA collections | National Cancer Institute | Collection-specific | Thousands of collections and studies | Per-collection licence | Uniform labels; uniform access |
| SLC-PFM | Challenge organisers | 2025 edition | More than 300 million images; 39 cancers; 23 tasks | Competition terms | Clinical utility; label provenance varies |

The scale column mixes images, studies, patients, slides, and biopsies. A table that compares them without units invites a false sense of sample size. The last column is the one that decides fitness.

### Modality-specific label problems

| Modality | Typical label source | Specific failure | What to check |
|---|---|---|---|
| Chest radiograph | Report text | Negation, uncertainty, reporting style | Negation handling and uncertainty labels |
| CT | Report plus measurement | Slice selection and contrast phase | Contrast phase and reconstruction |
| MRI | Expert segmentation | Boundary ambiguity and protocol variation | Sequence, spacing, and reader agreement |
| Mammography | Pathology or recall | Interval cancers and prevalence | Screening round and follow-up |
| Whole-slide pathology | Pathologist grade | Inter-pathologist disagreement | Grade distribution and adjudication |
| Ultrasound | Operator-dependent acquisition | Probe and settings variation | Operator and device metadata |
| Fundus photography | Grading scale | Camera and image quality | Gradability and camera model |
| Endoscopy | Video annotation | Frame-level labels and temporal context | Frame sampling and procedure type |

### Annotation protocol fields

| Field | Why it matters | Example |
|---|---|---|
| Number of readers | Determines whether agreement is estimable | Two readers plus adjudicator |
| Reader expertise | Determines the reference standard | Board-certified radiologist |
| Blinding | Prevents anchoring | Independent reads |
| Adjudication rule | Resolves disagreement | Third reader or consensus |
| Label scale | Determines the metric | Binary, ordinal, or continuous |
| Uncertainty handling | Changes the effective label set | Excluded, separate class, or probabilistic |
| Annotation unit | Determines the denominator | Image, study, lesion, or slide |
| Software and version | Supports reproducibility | Annotation tool and version |
| Time spent per case | Indicates label quality | Minutes per case |
| Quality control | Detects drift | Random re-read of a sample |

### Preprocessing decisions to record

| Decision | Options | Leakage risk | What to store |
|---|---|---|---|
| Resampling | Original, isotropic, fixed spacing | Fitting spacing on all data | Target spacing |
| Intensity | Raw, windowed, normalised | Dataset-level statistics | Window and normalisation parameters |
| Cropping | Body, lung, lesion-centred | Using test lesions to crop | Crop rule |
| Augmentation | Rotation, flip, intensity | Left-right flip changes anatomy | Augmentation policy |
| Stain normalisation | None, reference-based, colour deconvolution | Reference chosen from test | Reference and method |
| Tiling | Grid, tissue-masked, random | Tile overlap across splits | Tile size and stride |
| Sequence selection | All, fixed subset | Using test availability | Sequence list |
| Label mapping | Raw, harmonised | Mapping fitted on test | Mapping table |

### Image quality audit

| Quality dimension | Question | Method | Effect on the model |
|---|---|---|---|
| Diagnostic quality | Is the image readable? | Reader review or a quality score | Unreadable images add label noise |
| Contrast and exposure | Is the tissue contrast adequate? | Intensity histogram and visual review | Poor contrast mimics pathology |
| Motion | Is there motion artefact? | Visual review and metadata | Artefact correlates with setting |
| Positioning | Is the anatomy in the field of view? | Landmark detection or review | Cropping can remove findings |
| Laterality | Is the side marker correct? | Marker and header check | Left-right errors are dangerous |
| Artefact | Is there a device or implant? | Metadata and review | Support devices confound findings |
| Reconstruction | Is the kernel consistent? | Metadata | Texture differs by kernel |
| Compression | Is the image lossy? | Transfer syntax | Compression can remove detail |
| Duplication | Is the image a repeat? | Hash and perceptual hash | Inflation and leakage |
| Series completeness | Are all expected series present? | Manifest check | Missing sequences change the task |

### Annotation cost model

| Item | Unit | Typical range | Planning implication |
|---|---|---|---|
| Reader time per case | Minutes | 2 to 30 depending on task | Dominates the annotation budget |
| Adjudication time | Minutes per disagreement | 5 to 20 | Add 20 to 40 percent |
| Training and calibration | Hours | 4 to 16 | Required for multi-reader studies |
| Quality control re-read | Share of cases | 5 to 10 percent | Detects drift |
| Software setup | Hours | 8 to 40 | Often omitted from the budget |
| Case selection | Hours | Variable | Stratified sampling takes longer |
| Total for 1,000 cases | Reader-days | Task-dependent | Compute before promising a label set |

The cost model is the reason most imaging projects use report-derived labels for pretraining and reserve adjudicated labels for the evaluation set. The split between weak and strong labels is a budget decision as much as a scientific one.

### Phase B — Audit the images and labels

**Step 5. Inventory the images and metadata.**
Input: the DICOM headers or the dataset manifest.
Output: counts by patient, study, series, modality, scanner, site, and year.
Rationale: the inventory exposes duplicates, missing series, and site imbalance before modeling.
Branch: if the manifest has no scanner or site field, the site audit cannot be done, and the benchmark gate is at most 1.
Edge case: de-identification can remove the site tag while leaving a scanner fingerprint in the pixel data.

**Step 6. Check for duplicates and near-duplicates.**
Input: image hashes and perceptual hashes.
Output: exact duplicate groups and near-duplicate candidates.
Rationale: duplicates across splits inflate performance and are common when images are collected from multiple sources.
Branch: if near-duplicates are found across splits, regroup or remove them.
Edge case: the same study can appear in two datasets with different identifiers, which defeats a simple ID check.

**Step 7. Audit the label provenance.**
Input: the annotation protocol, the reader credentials, and the adjudication rule.
Output: a provenance statement with the number of readers, the agreement statistic, and the adjudication method.
Rationale: the label error bounds the attainable performance and determines the claim.
Branch: if the label is report-derived, estimate agreement on a sample and report the uncertainty labels separately.
Edge case: a single-reader dataset cannot estimate inter-reader agreement without a second read.

**Step 8. Estimate label agreement on a sample.**
Input: a stratified sample and a second reader or a reference standard.
Output: a confusion matrix, kappa or ICC, and a positive predictive value.
Rationale: the agreement estimate converts "expert labelled" into a number.
Branch: if agreement is low, the task may be ambiguous, and the model's ceiling is correspondingly low.
Edge case: the second reader can be biased by seeing the first reader's annotation. Use a blinded read.

**Step 9. Audit class balance and subgroup coverage.**
Input: the label table and the metadata.
Output: prevalence by class, site, scanner, sex, age, and acquisition protocol.
Rationale: a rare finding with a single site contributing most positives produces a site-specific model.
Branch: if a class is present at only one site, a leave-site-out split cannot evaluate it.
Edge case: a negative label can dominate because the finding is rare, and accuracy becomes misleading.

### Phase C — Design the split and preprocessing

**Step 10. Write the split specification before modeling.**
Input: the patient, study, site, and time keys.
Output: a split with group keys, fractions, and a hash of the assigned IDs.
Rationale: a split chosen after seeing results is a tuning parameter.
Branch: if a patient key is absent, the split cannot be patient-level, and the benchmark gate is at most 1.
Edge case: the same patient can have different identifiers across sites; document the linkage and its error.

**Step 11. Choose the split hierarchy.**
Input: the claim and the available metadata.
Output: patient-level, site-level, temporal, or combined.
Rationale: the hierarchy determines what the evaluation estimates.
Branch: use the split decision table below.
Edge case: a site-level split with only two sites gives a high-variance estimate; report the per-site result, not only the average.

**Step 12. Fit preprocessing on training data only.**
Input: the training split and the preprocessing plan.
Output: stored parameters and a transform applied identically to all splits.
Rationale: fitting normalisation or intensity statistics on the full dataset leaks test information.
Branch: if the preprocessing is site-specific, fit per site and report the difference.
Edge case: a public pretrained model may have its own normalisation; document the mismatch.

**Step 13. Check image orientation and geometry.**
Input: the DICOM headers and the resampled volumes.
Output: a geometry report with orientation, spacing, and direction.
Rationale: an orientation error can flip left and right or rotate a volume, producing a model that learns the error.
Branch: if orientation is inconsistent, harmonise to a standard and record the transform.
Edge case: some vendors encode orientation differently; validate on a sample visually.

### Phase D — Evaluate and report

**Step 14. Choose the metric from the decision.**
Input: the decision costs and the capacity.
Output: a primary metric, a secondary metric, and a calibration plan.
Rationale: Dice, AUROC, and sensitivity answer different questions.
Branch: if the decision is a fixed-capacity review, report sensitivity and precision at k; if it is segmentation for planning, report boundary distance and plan impact.
Edge case: a metric required by a challenge is reported alongside, not instead of, the decision metric.

**Step 15. Evaluate by site, scanner, and subgroup.**
Input: the test set and the metadata.
Output: a table of metrics with confidence intervals by group.
Rationale: an average can hide a site that fails.
Branch: if a site has too few cases, report it descriptively and do not rank it.
Edge case: a subgroup with high performance can reflect access bias rather than model quality.

**Step 16. Test for contamination and edition drift.**
Input: the dataset version, the model training data, and the publication dates.
Output: a contamination assessment and a version comparison.
Rationale: a test set that appears in training, or a label set that changed between editions, invalidates the comparison.
Branch: if contamination is found, replace the test set with a clean holdout.
Edge case: near-duplicate images are the common case; use perceptual hashing, not file hashes alone.

**Step 17. Score the gates and record the evidence.**
Input: the outputs of steps 1 to 16.
Output: a gate table with evidence and residual uncertainty.
Rationale: the decision must be reviewable.
Branch: a zero on any gate is a veto for the claim as written.
Edge case: a dataset can be admissible for a methods claim and inadmissible for a product claim.

**Step 18. Re-run on edition, scanner, or licence change.**
Input: the release notes, the challenge rules, and the deployment sites.
Output: an updated gate record with a new date.
Rationale: imaging datasets change with editions, and scanners change with contracts.
Branch: re-run when the dataset version, the label protocol, the scanner fleet, or the licence changes.
Edge case: a challenge can move its data to a new host with new terms.

### Reporting checklist for an imaging benchmark

| Item | What to report | Why a reviewer asks |
|---|---|---|
| Dataset and version | Name, edition, release date, hash | Reproducibility |
| Access date | When the data were obtained | Terms can change |
| Unit | Patient, study, series, image, slide, tile | Denominator and independence |
| Cohort | Inclusion, exclusion, and counts | Population fit |
| Label source | Readers, protocol, adjudication | Label error |
| Agreement | Kappa, ICC, or positive predictive value | Ceiling |
| Split | Keys, fractions, hash, group counts | Leakage |
| Preprocessing | Fit scope and parameters | Test leakage |
| Model | Architecture, pretraining data, checkpoint | Contamination |
| Metric | Definition, operating point, confidence interval | Decision fit |
| Subgroups | Site, scanner, sex, age, protocol | Transportability |
| Calibration | Slope, intercept, curve | Threshold use |
| Contamination | Probes and results | Benchmark validity |
| Code and weights | Availability or a stated reason | Reproducibility |

### Evaluation pitfalls in imaging

- Reporting Dice without a boundary metric, when the clinical task depends on the boundary.
- Reporting AUROC for a fixed-capacity review programme.
- Reporting a single operating point without the threshold or the prevalence.
- Reporting an average across sites without the per-site table.
- Comparing a result with a leaderboard from a different edition.
- Using a public pretrained model without checking whether it saw the test data.
- Splitting by image and describing the result as patient-level generalization.
- Fitting normalisation on the full dataset and calling the result external.
- Reporting a subgroup metric with fewer than ten events.
- Treating a report-derived label as adjudicated truth.

### Collection selection decision table

| Situation | Recommended collection type | Reason | Alternative |
|---|---|---|---|
| Methods development | Open or credentialed public collection | Comparability and cost | Challenge data |
| Commercial product | Licensed or partner data | Deployment rights | Public data for pretraining only |
| Rare finding | Registry or consortium | Case concentration | Multi-site partner network |
| Segmentation | Expert-annotated challenge data | Voxel labels | In-house annotation |
| Screening | Population or programme data | Prevalence and follow-up | Registry linkage |
| Pathology | Whole-slide consortium | Slide-level labels | Single-site archive |
| Device robustness | Multi-scanner collection | Domain coverage | Prospective scanner study |
| Temporal drift | Multi-year collection | Era coverage | Prospective monitoring |

A collection profile is a claim by the custodian until it is checked. The audit in step 5 verifies the manifest, and the licence audit verifies the terms. Neither is complete without the other, and a collection that passes one can fail the other.

### Unit decision table

| Claim | Unit | Split key | Common error |
|---|---|---|---|
| Patient-level diagnosis | Patient | Patient ID | Image-level split |
| Study-level finding | Study | Patient ID plus study date | Splitting images within a study |
| Lesion segmentation | Volume or study | Patient ID | Slicing a volume across splits |
| Whole-slide diagnosis | Slide | Patient ID | Tile-level split |
| Temporal progression | Patient over time | Patient ID plus time | Random time points |
| Scanner robustness | Site or scanner | Site ID | Same scanner on both sides |
| Screening programme | Participant | Participant ID plus episode | Episode-level split |

### Split decision table

| Split | What it estimates | Requirement | Limitation |
|---|---|---|---|
| Random image | Memorisation resistance | None | Leaks patient and study |
| Patient-level | Generalisation to new patients | Patient ID | Same-site scanner remains |
| Site-level | Generalisation to a new site | Site ID | Few sites means high variance |
| Temporal | Generalisation to a later period | Dates | Protocol change confounded with time |
| Combined patient and site | Transportability | Both keys | Smaller test set |
| Leave-one-site-out | Site robustness | Several sites | Expensive and variable |

### Label decision table

| Label source | Strength | Weakness | Claim it supports |
|---|---|---|---|
| Multi-reader adjudicated | Highest agreement estimate | Expensive and small | Clinical performance with a ceiling |
| Single expert | Plausible reference | No agreement estimate | Internal benchmark |
| Report-derived | Large and cheap | Reporting style and negation errors | Weak-label screening and pretraining |
| Registry or pathology | Independent reference | Linkage and selection | Diagnostic accuracy with a reference |
| Rule-based | Reproducible | Encodes assumptions | Pipeline testing |
| Synthetic | Controlled | Not real distributions | Software testing only |

### Edge cases that change the procedure

**Licence forbids commercial use.** BraTS, PANDA, and many challenge datasets are research-use only. A commercial product needs a separate licence or a different data source. Score Gate 4 as 0 for the product claim and keep the dataset for methods.

**Dataset withdrawn or access paused.** BraTS 2025 access was reported as unavailable during a transfer to The Cancer Imaging Archive. If access is paused, the candidate fails Gate 3 for the timeline even if it is scientifically ideal. Check the access page before committing.

**Credentialing takes months.** Imaging credentialing can be quick for an open collection and slow for a controlled one. A clinical partner's data-sharing agreement can take longer than the modeling.

**Version drift.** A new challenge edition can change the label set, the number of classes, and the test distribution. Pin the edition and do not compare across editions without a task audit.

**Label errors discovered.** If the reference labels are wrong for a material fraction of cases, the ceiling falls. Re-adjudicate a sample, report the error, and consider a probabilistic label. Do not relabel only the test set.

**Benchmark contamination.** Pretrained models may have seen public challenge data. Check the model's training data and the dates, and use a private holdout where possible.

## 6. Worked example

### The case

A 30-person medical imaging company wants to sell a glioma segmentation tool to radiotherapy departments. The tool proposes an enhancing-tumour and resection-cavity contour that the radiation oncologist edits, with the promise of reducing contouring time. The company has 14 months to a clinical pilot and intends a commercial product.

**Decision statement.** "At treatment planning for adults with newly diagnosed glioma, propose a tumour segmentation that a radiation oncologist can accept with limited editing, reducing contouring time by at least 25 percent without increasing plan revisions; a false boundary near the brainstem can delay approval, and an under-segmented tumour can leave disease untreated."

**Unit.** The study and the patient. A patient can have a preoperative and a postoperative study, and both must stay on one side of the split.

**Claim.** The workflow claim is time saved and accepted contours, not Dice. Dice is a technical secondary metric.

### Candidate long list

| Candidate | Custodian | Version inspected | Scale | Access and licence | Why it is on the list |
|---|---|---|---|---|---|
| BraTS 2025 | MICCAI BraTS consortium | Lighthouse 2025 | Multiple tasks and editions | Synapse registration and terms; access in transition to TCIA | Expert segmentations and a standard benchmark |
| TCIA collection | National Cancer Institute | Collection-specific | Thousands of studies across collections | Per-collection licence; some open | Multi-institutional images and metadata |
| Institutional partner | Two radiotherapy departments | Not collected | Site-dependent | Contract and IRB | Target workflow and plan outcomes |
| Prior BraTS editions | Consortium | 2023 and 2024 | Edition-specific | Post-challenge terms | Pretraining and method comparison |
| Synthetic phantom | Open tools | Current | Configurable | Open | Pipeline testing only |

### Gate scores for the commercial claim

| Gate | BraTS 2025 | TCIA | Partner | Prior editions | Phantom |
|---|---:|---:|---:|---:|---:|
| 1 Decision fit | 2 | 2 | 2 | 2 | 0 |
| 2 Population fit | 1 | 1 | 2 | 1 | 0 |
| 3 Access fit | 0 | 2 | 0 | 1 | 2 |
| 4 Licence fit | 1 | 1 | 2 | 1 | 2 |
| 5 Benchmark fit | 1 | 1 | 2 | 0 | 0 |
| Veto | access | none | access | benchmark | decision |

**BraTS 2025, Gate 1.** Expert segmentations of tumour subregions are available, and the task matches the segmentation claim. The labels do not include plan quality or editing time, so the workflow claim is not measurable in the dataset. Score 2 for the technical task, with the workflow gap recorded.

**BraTS 2025, Gate 3.** Registration is required, followed by a data access form and acceptance into a data access team. At the time of checking, the organisers reported that data were being transferred to The Cancer Imaging Archive and that no new access was available. For a 14-month plan, the access status is a 0 until a confirmed route exists. Veto.

**BraTS 2025, Gate 4.** The data are distributed under research-use terms, not a commercial licence. A product would need separate terms. Score 1 for research and 0 for the product claim.

**TCIA, Gate 3.** Open access varies by collection. Some collections are downloadable after registration; others have restrictions. The company must audit the licence collection by collection. Score 2 for the collections that are open and permit the intended use.

**TCIA, Gate 4.** Licences vary. A collection released under a Creative Commons Attribution licence permits commercial use with attribution; a collection with a research-only agreement does not. The company cannot treat "TCIA" as one licence. Score 1 until the specific collections are audited.

**Partner, Gate 3.** Two radiotherapy departments are interested, but there is no contract, no IRB approval, and no data-sharing agreement. Fourteen months is not credible for a new multi-site contract plus a prospective pilot. Veto for the current timeline, retained as the validation path.

**Prior editions, Gate 5.** The label set and the task definition changed across editions. A model trained on one edition and evaluated on another is not comparable without a task audit. Score 0 for a cross-edition benchmark claim.

### The two diagnostics

**Diagnostic 1: the access route is closed.** The organisers' notice that data were being transferred and new access was unavailable is decisive for the timeline. The team cannot plan around a dataset that may reopen after the pilot date. The diagnostic is not the transfer itself; it is the absence of a confirmed access date. A candidate with no confirmed route is a zero on Gate 3, not a one.

**Diagnostic 2: the metric does not measure the claim.** The dataset contains voxel labels and no information about contouring time, edits, or plan revisions. A model with a Dice of 0.85 on the enhancing tumour can still require extensive editing at the brainstem, and a model with a slightly lower Dice can be faster to edit if its errors are in less critical regions. The dataset can support a segmentation benchmark and cannot support the workflow claim. This is a Gate 5 failure for the product, not a modeling failure.

A third diagnostic is edition drift. The 2023, 2024, and 2025 editions differ in tasks and label definitions. A pipeline built on one edition must document the version and must not compare its number with a leaderboard from another.

### The selected route

The admissible set is TCIA collections. The plan uses a commercial-permissive TCIA collection for pretraining, a second collection for a site-level test, and the partner sites for the workflow evaluation. BraTS is retained for method comparison only, with the access status checked monthly.

| Track | Steps | Owner | Duration | Planning cost |
|---|---|---|---|---|
| Licence audit | Review each TCIA collection's terms | Legal and data lead | 3 weeks | Staff time |
| Data assembly | Download, harmonise, de-identify check | Data engineering | 6 weeks | Compute and storage |
| Segmentation model | Pretrain, fine-tune, internal test | ML team | 12 weeks | Compute, about $12,000 |
| Reader study | Radiation oncologists edit contours | Clinical lead | 8 weeks | About $25,000 |
| Partner validation | Contract, IRB, silent mode | Clinical lead | 6 to 9 months | Site fees |

The reader study is the only way to test the workflow claim. It is also the most expensive step, and it cannot start until the model is stable.

### Time and cost

| Item | Detail | Calendar time | Planning cost |
|---|---|---|---|
| Licence audit | Collection-by-collection terms | 3 weeks | Legal, 40 hours |
| Data engineering | Harmonise spacing, orientation, intensity | 6 weeks | 0.5 FTE |
| Model development | Two baselines and a strong model | 12 weeks | Compute, about $12,000 |
| Reader study | 6 readers, 100 cases | 8 weeks | About $25,000 |
| Partner validation | Contract, IRB, silent mode | 6 to 9 months | Six figures |
| Total to a pilot decision | Excludes the partner contract | About 30 weeks | About $60,000 before the partner |

The reader study is the gate that converts a Dice number into a workflow claim. A team that skips it has a segmentation model and no evidence that it saves time.

### Substitute plan if the TCIA licence audit fails

If every suitable collection is research-only, the plan changes in three ways:

1. The product claim narrows to a research tool, which is not sellable to radiotherapy departments.
2. The company must obtain commercial data through a partner contract or a licensed vendor, which adds 6 to 12 months.
3. The pretraining can still use research data if the licence permits training and prohibits deployment only for the data, not for the model. That distinction must be confirmed in writing, because some agreements treat weights as derivatives.

The fallback is not equivalent. It changes the commercial route, not only the dataset.

### Vary one assumption: the partner signs in month two

Suppose a partner signs early and provides 400 studies with plan outcomes. Gate 3 for the partner rises to 2, and the plan changes:

- The partner data become the primary development and evaluation set, and TCIA becomes pretraining only.
- The workflow claim becomes measurable, because plan revisions and editing time are recorded.
- The reader study shrinks to a confirmation on partner data rather than a stand-alone study.
- The timeline shortens by about three months, and the cost shifts from a broad reader study to site integration.

The license position also changes: partner data are governed by a contract that can include commercial rights, while TCIA collections may not. The scientific plan improves, and the legal plan becomes simpler, which is unusual and worth recording.

### Gate evidence log

| Gate | Candidate | Evidence inspected | Verdict | Residual uncertainty |
|---|---|---|---|---|
| 1 | BraTS 2025 | Task definition and label schema | 2 | No workflow outcome in the data |
| 1 | TCIA | Collection-level label and image schema | 2 | Label quality varies by collection |
| 2 | BraTS 2025 | Contributing institutions and selection | 1 | Not the target radiotherapy workflow |
| 2 | TCIA | Collection sites and scanners | 1 | Heterogeneous and not protocol-matched |
| 3 | BraTS 2025 | Access page and organiser notice | 0 | Reopening date unknown |
| 3 | TCIA | Collection access pages | 2 | Some collections require registration |
| 4 | BraTS 2025 | Research-use terms | 1 | Commercial terms unavailable |
| 4 | TCIA | Collection licences | 1 | At least one collection must permit commercial use |
| 5 | BraTS 2025 | Dice-based leaderboard; no plan outcome | 1 | Technical benchmark only |
| 5 | TCIA | No common benchmark | 1 | Split must be built by the team |
| 5 | Partner | Plan revisions and editing time measurable | 2 | Contract not started |

### Decision timeline

| Month | Licence track | Data track | Clinical track | Deliverable |
|---:|---|---|---|---|
| 1 | Audit TCIA collection terms | Inventory candidate collections | Identify reader panel | Licence shortlist |
| 2 | Confirm commercial permissions | Download and harmonise | Reader protocol draft | Approved collections |
| 3 | Confirm attribution terms | Build pretraining set | Reader training | Data v1 |
| 4-5 | Review partner contract | Train segmentation model | Reader study setup | Model v1 |
| 6-7 | Partner signature | Internal test and boundary metrics | Reader study | Technical report |
| 8-11 | IRB and data transfer | Site-level test | Silent-mode preparation | External plan |
| 12-14 | Deployment terms | Monitoring setup | Pilot | Pilot decision |

### What the case demonstrates

- The most famous dataset in the field was unavailable on the timeline, and its licence would not have supported the product even if it had been available.
- The technical metric passed and the workflow claim failed. Dice was necessary evidence and not sufficient evidence.
- The patient-level split was a design choice, not a default. The audit found a 40 percent patient overlap in a study-level split.
- The reader study was the bridge between the dataset and the claim, and it was the most expensive part of the plan.
- The partner route changed both the science and the legal position for the better, which is rare and worth designing for.
- The fallback was a different commercial route, not a different model.

### Sensitivity of the decision

| Change | Gate affected | Does the source change? | Plan response |
|---|---|---|---|
| A commercial TCIA collection is confirmed | 4 | No | Proceed; keep partner for workflow |
| BraTS access reopens next month | 3 | Yes, as a comparator | Add as a technical benchmark, not a product source |
| Reader study shows no time saving | 5 | No | Stop the workflow claim; keep the research tool |
| Partner provides plan outcomes | 5 | Yes, becomes primary | Shift development to partner data |
| A new edition changes the label set | 1 and 5 | No | Pin the edition; do not compare across editions |

### Conditions under which the case would fail

- **No TCIA collection permits commercial use.** The product route closes, and the company must rely on a partner contract before it can train a deployable model.
- **The reader study shows no time saving.** The workflow claim fails even if Dice is high, and the product becomes a technical tool without a validated benefit.
- **The partner contract slips past month eight.** There is no external validation in the target workflow, and the pilot is delayed.
- **A new challenge edition changes the label set.** The technical benchmark must be re-specified, and prior results cannot be compared.
- **The scanner mix in the selected collections is narrow.** The model may not transport to the partner's scanners, and the external test becomes a scanner test rather than a workflow test.
- **The de-identification audit finds burned-in identifiers.** The data cannot be used until the issue is resolved, and the access route may need to change.

### What the case demonstrates about evidence

The technical benchmark and the clinical claim are separate evidence streams. Dice answers whether the model overlaps the reference; the reader study answers whether a clinician can use it; the partner pilot answers whether the workflow changes. A dossier that contains only the first is incomplete, and a team that treats the first as the third will over-promise.

## 7. Code, query, or calculation

The three blocks below use the Python standard library only. They implement a split leakage audit, an inter-reader agreement calculation, and a Dice comparison that shows why the metric is not a clinical utility measure.

### 7.1 Split leakage audit

```python
IMAGES = [
    ("p01", "s01", "train"), ("p01", "s01", "train"), ("p01", "s02", "test"),
    ("p02", "s03", "train"), ("p02", "s03", "train"),
    ("p03", "s04", "test"), ("p03", "s04", "test"),
    ("p04", "s05", "train"), ("p05", "s06", "test"),
    ("p06", "s07", "train"), ("p06", "s08", "test"),
    ("p07", "s09", "train"), ("p08", "s10", "test"),
]

train_patients = {p for p, s, split in IMAGES if split == "train"}
test_patients = {p for p, s, split in IMAGES if split == "test"}
train_studies = {s for p, s, split in IMAGES if split == "train"}
test_studies = {s for p, s, split in IMAGES if split == "test"}

print(f"Images: {len(IMAGES)}")
print(f"Train images: {sum(1 for _, _, x in IMAGES if x == 'train')}")
print(f"Test images:  {sum(1 for _, _, x in IMAGES if x == 'test')}")
print(f"Train patients: {len(train_patients)}  Test patients: {len(test_patients)}")
print(f"Train studies:  {len(train_studies)}  Test studies:  {len(test_studies)}")
print()
print(f"Patient overlap: {sorted(train_patients & test_patients)}")
print(f"Study overlap:   {sorted(train_studies & test_studies)}")
print()
n_leak_patients = len(train_patients & test_patients)
n_test_patients = len(test_patients)
print(f"Share of test patients also in train: {n_leak_patients/n_test_patients:.0%}")
print("A patient-level split would move every image for a leaked patient to one side.")
```

Expected output:

```text
Images: 13
Train images: 7
Test images:  6
Train patients: 5  Test patients: 5
Train studies:  5  Test studies:  5

Patient overlap: ['p01', 'p06']
Study overlap:   []

Share of test patients also in train: 40%
A patient-level split would move every image for a leaked patient to one side.
```

The audit shows the typical pattern: the study split is clean and the patient split is not. A patient with a training study and a test study leaks anatomy and acquisition. The fix is to assign the patient, not the study, to a side.

### 7.2 Inter-reader agreement

```python
def cohen_kappa(matrix):
    n = sum(sum(row) for row in matrix)
    po = sum(matrix[i][i] for i in range(len(matrix))) / n
    row_totals = [sum(row) for row in matrix]
    col_totals = [sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix))]
    pe = sum(row_totals[i] * col_totals[i] for i in range(len(matrix))) / (n * n)
    return (po - pe) / (1 - pe), po, pe

# rows: reader A absent/present; columns: reader B absent/present
matrix = [
    [812, 24],
    [38, 126],
]
k, po, pe = cohen_kappa(matrix)
n = sum(sum(row) for row in matrix)
print(f"Cases: {n}")
print(f"Observed agreement: {po:.3f}")
print(f"Chance agreement:   {pe:.3f}")
print(f"Cohen's kappa:      {k:.3f}")
print()
if k < 0.4:
    verdict = "poor"
elif k < 0.6:
    verdict = "moderate"
elif k < 0.8:
    verdict = "substantial"
else:
    verdict = "good"
print(f"Interpretation: {verdict} agreement on this sample.")
```

Expected output:

```text
Cases: 1000
Observed agreement: 0.938
Chance agreement:   0.735
Cohen's kappa:      0.766

Interpretation: substantial agreement on this sample.
```

Observed agreement of 0.938 looks high and kappa is 0.766 because the finding is uncommon. Reporting raw agreement without the chance correction overstates reliability. Report both, and report the prevalence.

### 7.3 Dice and its limits

```python
def dice(pred, truth):
    inter = len(pred & truth)
    return 2 * inter / (len(pred) + len(truth))

cases = [
    ("large lesion, good overlap", set(range(1000)), set(range(50, 1050))),
    ("small lesion, shifted", set(range(100)), set(range(60, 160))),
    ("small lesion, missed edge", set(range(100)), set(range(90, 190))),
]
print(f"{'Case':<32}{'Pred':>6}{'Truth':>7}{'Dice':>8}")
for name, pred, truth in cases:
    print(f"{name:<32}{len(pred):>6}{len(truth):>7}{dice(pred, truth):>8.3f}")
print()
print("Dice measures relative overlap, not absolute error or clinical importance.")
print("A 50-voxel error near the brainstem and a 50-voxel error in the cavity")
print("can produce the same value and very different clinical consequences.")
```

Expected output:

```text
Case                              Pred  Truth    Dice
large lesion, good overlap        1000   1000   0.950
small lesion, shifted              100    100   0.400
small lesion, missed edge          100    100   0.100

Dice measures relative overlap, not absolute error or clinical importance.
A 50-voxel error near the brainstem and a 50-voxel error in the cavity
can produce the same value and very different clinical consequences.
```

The metric is not wrong. It answers a specific question: how much do two masks overlap relative to their size. The clinical question is where the error is, how much editing it causes, and whether the plan changes. A benchmark report should include a boundary metric and, where possible, a reader or plan outcome.

### 7.4 Tile-level versus image-level prevalence

```python
slides = [
    ("slide_01", 2, 1200),
    ("slide_02", 0, 900),
    ("slide_03", 45, 1500),
    ("slide_04", 0, 1100),
    ("slide_05", 1, 800),
]
image_positive = sum(1 for _, pos, total in slides if pos > 0)
n_images = len(slides)
total_tiles = sum(total for _, _, total in slides)
positive_tiles = sum(pos for _, pos, _ in slides)

print(f"{'Slide':<10}{'Positive tiles':>16}{'Total tiles':>13}{'Tile prevalence':>17}")
for name, pos, total in slides:
    print(f"{name:<10}{pos:>16}{total:>13}{pos/total:>17.1%}")
print()
print(f"Image-level prevalence: {image_positive/n_images:.1%} ({image_positive}/{n_images})")
print(f"Tile-level prevalence:  {positive_tiles/total_tiles:.1%} ({positive_tiles}/{total_tiles})")
print()
print("A model trained on uniformly sampled tiles sees a prevalence close to the")
print("tile-level value. A model evaluated at the image level sees the image-level")
print("value. Reporting one as the other overstates or understates the base rate.")
```

Expected output:

```text
Slide       Positive tiles  Total tiles  Tile prevalence
slide_01                 2         1200             0.2%
slide_02                 0          900             0.0%
slide_03                45         1500             3.0%
slide_04                 0         1100             0.0%
slide_05                 1          800             0.1%

Image-level prevalence: 60.0% (3/5)
Tile-level prevalence:  0.9% (48/5500)

A model trained on uniformly sampled tiles sees a prevalence close to the
tile-level value. A model evaluated at the image level sees the image-level
value. Reporting one as the other overstates or understates the base rate.
```

The same issue appears in whole-slide pathology and in lesion detection. The sampling policy is part of the model, and the evaluation unit must match the claim.

## 8. Data, standards, and regulatory context

Imaging has the most developed technical standards of any biomedical data type, and the standards say almost nothing about labels. That asymmetry explains many failed projects: the pixels are interoperable, and the truth is not.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| DICOM | De facto and often contractual | Image format, metadata, transfer syntax, service classes | Guarantee label truth or patient-level grouping |
| DICOM PS3.15 | Standard, often required by policy | Security and de-identification profiles | Guarantee removal of burned-in text |
| DICOMweb | Standard | RESTful access to images and metadata | Solve governance or access |
| IHE profiles | Voluntary, often procured | Workflow and exchange conformance | Validate a model |
| FHIR ImagingStudy | Standard | Resource representation of imaging studies | Replace DICOM for pixel data |
| NIfTI | De facto in research | Volumetric image storage | Carry acquisition metadata reliably |
| TCIA collection terms | Contractual | Access and use of a specific collection | Provide a uniform licence across collections |
| FDA device guidance | Binding when applicable | Validation evidence for a device claim | Certify a model's fairness |
| EU MDR and IVDR | Binding in the EU | Device classification and clinical evidence | Address dataset licensing |
| EU AI Act | Binding, phased | Risk management and documentation for high-risk AI | Replace device regulation |
| HIPAA | Binding in the US | PHI handling and de-identification | Grant research or commercial rights |
| GDPR | Binding in the EU | Personal data processing and transfer | Define a label standard |
| RSNA and challenge rules | Contractual | Data use, external data, and submission limits | Guarantee a clean test set |

**DICOM is not de-identification.** The standard defines profiles, and de-identification is a process applied to pixels and metadata. Burned-in text, secondary capture, and private tags can retain identifiers. A dataset can pass a metadata check and still contain a name in the image. The audit should include a sample of pixel data, not only headers.

**The metadata is part of the data.** Scanner manufacturer, model, acquisition protocol, reconstruction kernel, slice thickness, and contrast phase determine the distribution. A model trained on one reconstruction kernel can fail on another. A dataset that strips these fields removes the ability to audit site shift.

**A label standard does not exist.** There is no DICOM equivalent for "pneumothorax present." Labels are defined by annotation protocols, report extraction rules, or registry linkages. The protocol is the standard, and it must be published with the dataset.

### What the standards do not settle

| Question | Standard answer | Research answer |
|---|---|---|
| What is a case? | Not defined | Patient or study, by claim |
| What is a positive label? | Not defined | Annotation protocol |
| How should readers be trained? | Not defined | Calibration set and anchors |
| How should disagreement be resolved? | Not defined | Adjudication rule |
| What is a clinically meaningful change? | Not defined | Reader or plan outcome |
| How should sites be held out? | Not defined | Site-level split |
| How should uncertainty be reported? | Not defined | Confidence intervals |
| How should drift be monitored? | Partly in device guidance | Site and scanner monitoring |
| How should data be versioned? | Not defined | Release, hash, and edition |
| How should synthetic data be labelled? | Not defined | Explicit synthetic flag |

The standards make the image portable and leave the truth to the study. That is why the label protocol is the most important document in an imaging dataset, and why it is often the shortest.

### Metadata fields that determine transportability

| Field | Why it matters | Common omission |
|---|---|---|
| Manufacturer and model | Scanner-specific appearance | Removed during de-identification |
| Acquisition protocol | Contrast, sequence, kernel | Not recorded in the manifest |
| Slice thickness and spacing | Volumetric comparability | Assumed constant |
| Reconstruction kernel | Texture and noise | Missing for CT |
| Contrast phase | Vessel and lesion appearance | Inconsistent across sites |
| Body part and laterality | Orientation and label scope | Encoded inconsistently |
| Study date | Era and protocol change | Shifted or removed |
| Institution | Site shift | Removed for privacy |
| Software version | Reconstruction change | Rarely recorded |

### DICOM de-identification checklist

| Check | What to look for | Method |
|---|---|---|
| Patient identifiers | Name, ID, birth date, accession | Header audit against a profile |
| Institution identifiers | Site name, address, phone | Header audit |
| Physician identifiers | Referring and reading physician | Header audit |
| Dates | Full dates and times | Date shifting or removal |
| Private tags | Vendor-specific fields | Tag inventory |
| Burned-in text | Names, dates, annotations in pixels | Visual sample and OCR |
| Secondary capture | Screenshots containing identifiers | Image review |
| Frame annotations | Ultrasound and angiography overlays | Image review |
| Device identifiers | Serial numbers and station names | Header audit |
| Study descriptions | Free-text fields with identifiers | Text review |
| Longitudinal consistency | Shifting that preserves intervals | Validation of the shift |
| Re-identification risk | Rare combinations of attributes | Risk assessment |

The checklist is a sample-based process, not a one-time setting. A dataset can pass a header audit and fail a pixel review, and the failure is a privacy incident rather than a model defect.

### Regulatory artefacts for an imaging product

- **Intended use** names the modality, anatomy, population, and clinical decision. A segmentation tool for radiotherapy is not a screening tool.
- **Reference standard** describes who annotated, with what protocol, and with what agreement.
- **Representativeness** describes the sites, scanners, and populations in the development and test data.
- **Validation** describes the split, the metric, the operating point, and the subgroup results.
- **Human factors** describes how the clinician interacts with the output and what happens when it is wrong.
- **Monitoring** describes drift detection and re-validation triggers.

An imaging model that reports Dice and no reader or workflow evidence has a technical result and an incomplete regulatory file.

## 9. Industry practice and case evidence

### What organisations actually do

- **Curate before modeling.** Teams with mature imaging pipelines spend most of their time on manifest construction, duplicate removal, and label audits.
- **Split by patient and site.** A random image split is treated as a defect, not a default.
- **Keep a scanner holdout.** Device robustness is tested on a scanner or site not used in training.
- **Version the preprocessing.** Normalisation, spacing, and cropping are stored with the model.
- **Run a reader study.** A workflow claim needs a human-in-the-loop evaluation, not only a held-out metric.
- **Audit licences collection by collection.** A repository is not a licence.
- **Monitor drift.** Scanner replacement and protocol change are routine, and a model can degrade without any code change.

### Case evidence

| Case | Documented fact | Inference |
|---|---|---|
| BraTS 2025 | The 2025 edition is a cluster of tasks under the MICCAI Lighthouse programme; access requires Synapse registration and acceptance of terms, and organisers reported that data were being transferred to TCIA with no new access at the time of checking. | A leading benchmark can be unavailable on a product timeline. Gate 3 must use the current access status. |
| PANDA | The dataset contains 10,616 prostate needle biopsies from two institutions, is released under CC BY-SA-NC 4.0, and keeps its test sets private to allow late submissions. | A non-commercial licence blocks a product, and a private test set is a deliberate anti-overfitting measure. |
| CheXpert | The Stanford AIMI research use agreement permits personal, non-commercial, non-clinical research; commercial use requires a separate licence. | A widely used public dataset can have a priced commercial route. |
| RSNA 2026 knee challenge | The edition closed in October 2026 with disputes over labels generated at scale and possible leakage, according to the module survey; the organiser report should be verified. | When labels are generated at scale, the label pipeline is part of the benchmark. |
| MIMIC-CXR | Access is credentialed, the labels are report-derived, and the images are from one institution. | A large dataset can be strong for methods and weak for population claims. |

The first four rows are recent and changeable. Verify the current access page, licence, and organiser report before quoting. The structural lesson, that access and labels are the binding constraints, is stable.

### Programme types and their evidence

| Programme | Decision | Primary metric | Required evidence | Common gap |
|---|---|---|---|---|
| Screening | Invite for further testing | Sensitivity at a fixed recall rate | Prospective programme data | Interval cancers |
| Triage | Review sooner | Time to review at fixed capacity | Workflow study | Queue effects |
| Diagnosis | Assist or confirm | Accuracy against a reference | Multi-reader study | Reader behaviour |
| Segmentation | Edit a contour | Boundary metric and editing time | Reader study and plan outcome | Clinical consequence |
| Monitoring | Track change over time | Longitudinal agreement | Serial imaging | Registration error |
| Quality control | Accept or reject an image | Agreement with a technologist | Operational data | Definition of acceptable |

The programme type determines the evidence, not the modality. A chest radiograph screening programme and a chest radiograph triage product need different studies even though they use the same images.

### Documented fact versus inference

- **Documented fact:** a release note, a datasheet, a licence, or an organiser report.
- **Inference:** a planning consequence drawn for a different setting.
- **Unverified summary:** a secondary account of an edition or a metric. Verify before citing.

A case table should carry a source and a status. This matters most for challenge editions, where access and labels change quickly.

### What good looks like

- The manifest states the unit and the counts at every level.
- The split is written before modeling and keyed on the patient.
- The label protocol names the readers, the adjudication, and the agreement.
- Preprocessing is fit on training data and stored with the model.
- The evaluation reports per-site and per-scanner results with intervals.
- The licence is checked collection by collection.
- The model card states the intended use and the prohibited uses.
- A monitoring plan names the drift metrics and the re-validation trigger.

### Why imaging benchmarks mislead procurement

A buyer asks whether the tool works on their scanners, their protocols, and their patients, and whether it saves time. A leaderboard answers a narrower question: how well does a method rank on a fixed test set from a different mix of sites. The gap is not dishonesty; it is a difference in estimand. A vendor that presents a leaderboard as procurement evidence invites a pilot that fails for reasons the benchmark never measured.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Image-level split | Test performance is unusually high | Same patient on both sides | Memorisation reported as generalization | Split by patient |
| Study-level leakage | A patient has two studies in different splits | Study key used instead of patient key | Anatomy and acquisition leak | Patient key and overlap audit |
| Scanner shortcut | Performance drops at a new site | One scanner dominates training | Site-specific model | Scanner holdout and augmentation |
| Report-derived label treated as truth | Label provenance says "from reports" | NLP inherits reporting style | Metric measures language | Audit a sample and report agreement |
| Negation error | A label appears for a finding that was ruled out | Simple keyword extraction | False positives in labels | Negation-aware extraction and review |
| Dice as clinical utility | The paper reports only Dice | Overlap is not plan quality | Product claim unsupported | Reader study and boundary metric |
| Preprocessing leakage | Normalisation uses the full dataset | Test statistics enter training | Inflated performance | Fit on training only |
| Edition drift | Results compared across challenge years | Label set or task changed | Invalid comparison | Pin the edition and audit the task |
| Duplicate images | Near-identical images across sources | Multi-source aggregation | Inflated test performance | Perceptual hashing |
| Orientation error | Left and right are swapped | Header inconsistency | Dangerous model | Geometry validation |
| Burned-in identifiers | Text visible in a sample | De-identification checked metadata only | Privacy breach | Pixel-level review |
| Licence assumption | "Public" is treated as "commercial" | Repository conflated with licence | Product blocked | Collection-by-collection audit |
| Access assumed | The dataset page is bookmarked | No confirmed route or date | Timeline slip | Access owner and date |
| Test-set reuse | The same test set across many papers | Leaderboard tuning | Saturated and noisy ranking | Private holdout |
| Reader anchoring | Agreement is higher than expected | One reader sees the other's mark | Inflated reference quality | Blinded independent reads |
| Cropping by lesion | Crops are centred on the finding | Test lesion location used | Optimistic performance | Crop by body region |
| Label leakage from text | Report text includes the outcome | Text used as a feature | The model reads the report | Exclude report-derived features |
| Phantom validation | A phantom is used to claim clinical accuracy | Controlled object differs from patients | Unsupported clinical claim | Use phantoms for calibration only |

Red flags that should stop an imaging project:

- The split description does not name the unit.
- The label provenance is "expert annotated" with no reader count.
- The dataset has no scanner or site metadata.
- The reported metric is Dice only for a clinical claim.
- The model uses a public pretrained checkpoint with no training-data audit.
- The access page has no confirmed date.
- The licence is described as "open" without a clause.
- The evaluation has no subgroup and no calibration.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The patient key is correct | Splitting depends on it | Identifier changes across sites | Linkage audit |
| The label measures the finding | The metric depends on it | Report extraction errors | Sample audit and agreement |
| The scanner mix matches deployment | Transportability depends on it | A new scanner or protocol | Scanner holdout |
| Preprocessing is reproducible | Reruns depend on it | Library or parameter change | Store parameters and versions |
| The metric matches the decision | Utility depends on it | Dice for a workflow claim | Reader or plan study |
| The edition is comparable | Benchmarking depends on it | Label or task change | Task audit |
| The licence permits the use | Product depends on it | Research-only terms | Clause review |
| The access route is open | Timeline depends on it | Transfer or pause | Confirmed date |
| The test set is clean | Validity depends on it | Duplicates or contamination | Hash and probe |
| The subgroup has enough events | Equity analysis depends on it | Small cell counts | Confidence intervals |
| The annotation is blinded | Agreement estimates depend on it | Readers see each other | Blinded re-read |
| The image is de-identified | Privacy depends on it | Burned-in text | Pixel review |
| The reader study is blinded | Agreement depends on it | Readers see each other | Audit the read order |
| The cropping rule is label-blind | Validity depends on it | Crops centred on the lesion | Re-crop by body region |
| The phantom is representative | Calibration depends on it | Controlled object differs | Use for calibration only |

The deepest limit is that image-level performance does not establish clinical benefit. A model can improve detection and fail to improve outcomes because the workflow does not act on the output, the alert arrives too late, or the clinician overrides it. That is a prospective evaluation question.

## 12. Exercises

**Exercise 1. Choose the unit.**
Deliverable: a table mapping four imaging claims to units and split keys.
Acceptance criteria: four claims; units justified; one leakage mechanism named per row.
Time: 30 minutes.

**Exercise 2. Audit a split.**
Deliverable: a script or spreadsheet that computes patient and study overlap and reports the leak rate.
Acceptance criteria: reproduces the section 7 output; names the affected patients; proposes a corrected split.
Time: 50 minutes.

**Exercise 3. Trace a label.**
Deliverable: a provenance record for one public imaging dataset, with reader count, protocol, and agreement if available.
Acceptance criteria: cites the datasheet or paper; distinguishes what is documented from what is inferred; states the ceiling implication.
Time: 45 minutes.

**Exercise 4. Critique a metric.**
Deliverable: a one-page memo explaining why Dice or AUROC does or does not match a named clinical decision.
Acceptance criteria: names the decision; identifies the missing outcome; proposes a supplementary evaluation.
Time: 35 minutes.

**Exercise 5. Compare four collections.**
Deliverable: a table with access, licence, unit, scale, label source, and what each cannot answer.
Acceptance criteria: four collections; every cell sourced or marked unknown; one rejection justified.
Time: 60 minutes.

**Exercise 6. Design preprocessing.**
Deliverable: a preprocessing specification with fit scope, parameters, and storage.
Acceptance criteria: fit on training only; parameters versioned; one leakage risk addressed.
Time: 45 minutes.

**Exercise 7. Detect edition drift.**
Deliverable: a comparison of two challenge editions on task, label set, and test access.
Acceptance criteria: two editions; at least three differences; a recommendation on comparability.
Time: 40 minutes.

**Assessment note.** Exercises 2, 3, 5, and 7 produce artefacts that can be checked: a leakage report, a provenance record, a comparison table, and a version comparison. Exercises 1, 4, 6, and 8 are design tasks and are marked against the rubric. Exercise 8 is the integration task.

**Exercise 8. Write an imaging data card.**
Deliverable: a two-page card with composition, collection, labels, preprocessing, uses, and prohibited claims.
Acceptance criteria: every field populated or marked unknown; one prohibited claim; no marketing language.
Time: 60 minutes.

## 13. Self-check answers

**Item 1. Why is an image not a case?**
A patient can have many images, studies, and series. Counting images as cases inflates the sample and violates independence. The case is the patient or the study, depending on the claim.

**Item 2. What is the difference between a study and a series?**
A study is an examination; a series is a set of images within it sharing acquisition parameters. A CT study can have several series, such as pre-contrast and post-contrast.

**Item 3. Why is a patient-level split stronger than a study-level split?**
A patient can have multiple studies, and the same anatomy and acquisition can appear in both. A patient-level split keeps all of a patient's images on one side and prevents that leakage.

**Item 4. What does a Dice coefficient of 0.85 tell you, and what does it not?**
It tells you that the predicted and reference masks overlap substantially relative to their combined size. It does not tell you where the error is, whether the error is clinically important, how much editing is required, or whether the plan changes.

**Item 5. Why can a report-derived label be adequate for pretraining and inadequate for a product claim?**
Pretraining benefits from scale and can tolerate label noise because the objective is representation learning. A product claim requires an estimate of clinical performance, and report-derived labels inherit reporting style, negation errors, and institutional vocabulary. The label error bounds the claim.

**Item 6. What is preprocessing leakage?**
Fitting a transformation on the full dataset, including test data, and applying it to all splits. Normalisation statistics, cropping rules, and label mappings are common sources. Fit on training data and store the parameters.

**Item 7. Why is a scanner holdout useful?**
It tests whether the model relies on scanner-specific appearance. A model that performs well on a held-out scanner is more likely to transport than one evaluated only on the same devices.

**Item 8. What is edition drift?**
A change between dataset editions in the label set, task definition, annotation protocol, or test distribution. It makes results across editions non-comparable unless the task is audited and matched.

**Item 9. Why does a private test set reduce leaderboard overfitting?**
It limits the number of times a team can tune against the evaluation. A public leaderboard with repeated submissions becomes a training signal, and the private split preserves a cleaner estimate.

**Item 10. When is synthetic imaging data acceptable?**
For pipeline development, augmentation research, interface testing, and privacy-safe demos. It is not acceptable for clinical performance or fairness claims. Synthetic images have a design distribution, not an observed one.

**Item 11. What is a scanner shortcut, and how do you detect it?**
A model that uses scanner-specific appearance rather than pathology. It can be detected by evaluating on a held-out scanner, by testing whether scanner identity is predictable from the learned features, and by comparing performance across devices. A strong answer proposes a scanner holdout. A weak answer relies on intensity normalisation alone.

**Item 12. Why is a private test set useful even when the labels are imperfect?**
It limits repeated tuning against the evaluation, so the reported result is less inflated by selection. It does not repair label error; it controls a different threat. A strong answer separates overfitting from label noise. A weak answer treats a private set as ground truth.

**Item 13. Why is a phantom not a substitute for clinical validation?**
A phantom is a controlled object with known properties. It can test calibration, geometry, and software behaviour, and it cannot reproduce patient anatomy, pathology variation, or acquisition variability. A phantom result is evidence about the device, not about clinical performance.

**Rubric for the design exercises (2, 5, 6, 8).** Full marks require a named unit, a sourced licence, a leakage check, and a prohibited-claim statement. Partial marks for a complete structure with unsourced claims. No marks for a Dice-only evaluation of a clinical claim.

## 14. Interview questions

**1. How would you split an imaging dataset for a clinical model?**
By patient, and by site or scanner where possible, with a temporal holdout if dates exist. A strong answer explains what each split estimates. A weak answer says "80/20 random."

**2. A paper reports 0.95 AUROC on chest radiographs. What do you ask?**
The split unit, label source, site and scanner mix, external validation, calibration, subgroup results, and whether the test set was reused. A strong answer asks for the data card and code. A weak answer asks only for the sample size.

**3. Why is Dice a poor sole metric for radiotherapy segmentation?**
It ignores boundary location, absolute error, and clinical consequence. A strong answer proposes a boundary metric and a reader or plan study. A weak answer says Dice is standard.

**4. How do you detect patient leakage?**
Build the patient key, compute the intersection of train and test patients, and check study dates. A strong answer also checks near-duplicate images. A weak answer trusts the dataset's official split.

**5. What does a non-commercial licence prevent?**
It can prevent product training, deployment, model release, and internal commercial demonstration, depending on the clause. A strong answer maps activities to clauses. A weak answer says it prevents selling the images.

**6. How do you handle a challenge dataset whose access is paused?**
Treat it as a Gate 3 zero for the timeline, keep a fallback, and monitor the access page. A strong answer does not plan around an unconfirmed reopening. A weak answer assumes the pause is temporary.

**7. What metadata do you need to assess transportability?**
Scanner manufacturer and model, acquisition protocol, slice thickness, reconstruction kernel, contrast phase, site, and date. A strong answer notes that de-identification may remove them. A weak answer asks only for the image count.

**8. How do you evaluate a segmentation model across sites?**
Report per-site Dice and boundary metrics with confidence intervals, and hold out at least one site. A strong answer includes a reader study for the workflow claim. A weak answer reports a pooled average.

**9. What is the difference between external validation and domain generalization?**
External validation is an evaluation on a different site, scanner, or era. Domain generalization is a method aimed at maintaining performance under shift. A strong answer says one external test is evidence, not proof. A weak answer equates the two.

**10. When would you refuse an imaging dataset?**
When the licence blocks the use, the labels cannot support the claim, the patient key is absent, or the scanner mix is irrelevant to deployment. A strong answer treats refusal as a result. A weak answer uses it because it is large.

**11. How do you audit de-identification?**
Check metadata against a profile, inspect pixel data for burned-in text, review private tags, and sample images visually. A strong answer treats it as a process with a sample. A weak answer trusts the dataset's statement.

**12. What would you put in an imaging model card?**
Intended use, modality and anatomy, population, training sites and scanners, label provenance, split, metrics with confidence intervals, subgroup results, calibration, limitations, and monitoring. A strong answer names what the model must not be used for. A weak answer lists architecture and accuracy.

**13. How do you handle a dataset with no patient identifier?**
Treat the benchmark gate as at most 1, document the limitation, and consider linkage or a substitute dataset. A strong answer does not claim patient-level generalization. A weak answer splits by image and describes the result as external.

**14. What is the first thing you check in an imaging manifest?**
The unit and the counts at each level: patients, studies, series, images. Then duplicates and missing fields. A strong answer runs the inventory before any model. A weak answer starts with a class balance table.

**15. How do you report a subgroup result with few cases?**
Report the count, the metric, and a confidence interval, and state that the estimate is imprecise. Do not rank sites on small counts. A strong answer treats the interval as the finding. A weak answer reports a point estimate and calls it a site difference.

## 15. Key terms

- **Adjudication** — resolution of reader disagreement by consensus or a third reader.
- **AUROC** — discrimination across all thresholds.
- **Burned-in text** — identifiers rendered into the image pixels.
- **Calibration** — agreement between predicted probabilities and observed frequencies.
- **Cohen's kappa** — chance-corrected agreement for two raters.
- **Contamination** — test data present in training or pretraining.
- **DICOM** — standard for medical image storage and communication.
- **Dice coefficient** — overlap between predicted and reference masks.
- **Domain shift** — change in image distribution across sites or devices.
- **Edition drift** — change in a dataset between releases.
- **External validation** — evaluation on a different site, scanner, or era.
- **Hausdorff distance** — maximum boundary discrepancy.
- **Hounsfield unit** — calibrated CT attenuation.
- **Instance** — a single image or frame.
- **Inter-reader agreement** — agreement between readers.
- **Label provenance** — the origin and process behind a label.
- **Negative label** — the absence of a finding.
- **NIfTI** — research volumetric image format.
- **Orientation** — anatomical direction of an image.
- **Patient-level split** — a split keeping all of a patient's images on one side.
- **Pixel spacing** — physical size of a pixel.
- **Preprocessing leakage** — fitting a transform on test data.
- **Report-derived label** — a label extracted from report text.
- **Series** — images sharing acquisition parameters.
- **Site-level split** — a split holding out entire sites.
- **Slice thickness** — distance between image slices.
- **Study** — an imaging examination.
- **TCIA** — The Cancer Imaging Archive.
- **Test set** — data held out from model selection.
- **Tile** — a patch extracted from a large image.
- **Volume** — a three-dimensional image stack.
- **Weak label** — a noisy or indirect label.
- **Whole-slide image** — a gigapixel image of a tissue slide.
- **Window and level** — display settings for CT intensity.
- **Acquisition protocol** — the parameters used for a scan.
- **Contrast phase** — timing of contrast relative to acquisition.
- **Diagnostic quality** — whether an image is adequate for interpretation.
- **Field of view** — the anatomy included in an image.
- **Gradability** — whether an image can be graded for the task.
- **Laterality** — the side of the body shown.
- **Motion artefact** — blurring from patient movement.
- **Private tag** — a vendor-specific DICOM field.
- **Reconstruction kernel** — the filter used to reconstruct CT images.
- **Repeat image** — a duplicate acquisition.
- **Series completeness** — whether all expected series are present.
- **SOP Instance UID** — a unique identifier for a DICOM instance.
- **Transfer syntax** — the encoding of DICOM pixel data.
- **Volume rendering** — a display method for three-dimensional data.
- **Anchor case** — a reference case used to calibrate readers.
- **Bootstrap** — resampling used to estimate uncertainty.
- **Gradability** — whether an image can be graded.
- **Reader study** — a study of clinician performance with and without a model.
- **Workflow outcome** — time, edits, or plan changes in real use.
- **Confidence interval** — the uncertainty around an estimate.
- **Prevalence** — the proportion of cases with a finding.
- **Negative predictive value** — the probability that a negative label is correct.
- **Attenuation** — the reduction of beam intensity by tissue.
- **Calibration curve** — observed versus predicted risk across bins.
- **Detection** — localization plus classification of a finding.
- **Localization** — identifying where a finding is.
- **Segmentation** — assigning a label to each voxel or pixel.
- **Classification** — assigning a label to a whole image or study.
- **Annotation tool** — software used to create labels.
- **Consensus read** — a label agreed by multiple readers.
- **Reference standard** — the best available truth for validation.
- **Sampling frame** — the population from which cases were drawn.
- **Selection bias** — distortion from how cases were chosen.
- **Spectrum bias** — distortion from the case mix in a dataset.
- **Verification bias** — distortion when only some patients get the reference test.

## 16. Sources

All URLs accessed 26 October 2026. Access terms and editions change; verify at the custodian page.

1. PhysioNet. MIMIC-CXR-JPG. https://physionet.org/content/mimic-cxr-jpg/
2. Stanford Center for Artificial Intelligence in Medicine and Imaging. CheXpert. https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
3. National Institutes of Health. ChestX-ray14. https://nihcc.app.box.com/v/ChestXray-NIHCC
4. Radiological Society of North America. AI challenges. https://www.rsna.org/artificial-intelligence
5. Synapse. BraTS-Lighthouse 2025 Challenge. https://www.synapse.org/Synapse:syn64153130/
6. Synapse. BraTS data access discussion. https://www.synapse.org/Synapse:syn5637528/discussion/threadId=12356
7. The Cancer Imaging Archive. https://www.cancerimagingarchive.net/
8. Grand Challenge. CAMELYON17. https://camelyon17.grand-challenge.org/
9. Grand Challenge. PANDA data. https://panda.grand-challenge.org/data/
10. Bulten W, Kartasalo K, Chen PC, et al. Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge. *Nature Medicine* 2022. https://doi.org/10.1038/s41591-021-01620-2
11. MIDOG challenge. https://midog2025.deepmicroscopy.org/
12. National Electrical Manufacturers Association. DICOM standard. https://www.dicomstandard.org/
13. DICOM PS3.15, Security and System Management Profiles. https://dicom.nema.org/medical/dicom/current/output/chtml/part15/PS3_15.html
14. HL7. FHIR ImagingStudy resource. https://hl7.org/fhir/imagingstudy.html
15. US Food and Drug Administration. AI-enabled device software functions guidance, draft January 2025. https://www.fda.gov/media/184856/download
16. European Parliament and Council. Regulation (EU) 2017/745 (MDR). https://eur-lex.europa.eu/eli/reg/2017/745/oj
17. European Parliament and Council. Regulation (EU) 2024/1689 (AI Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
18. US Department of Health and Human Services. HIPAA. https://www.hhs.gov/hipaa/
19. National Institutes of Health. The Cancer Imaging Archive collections. https://www.cancerimagingarchive.net/collections/
20. Radiological Society of North America. RSNA 2024 lumbar spine challenge. https://www.rsna.org/artificial-intelligence
21. Synapse. BraTS 2024 post-challenge data terms. https://www.synapse.org/Synapse:syn64952546/
22. NEMA. DICOM PS3.3, Information Object Definitions. https://dicom.nema.org/medical/dicom/current/output/chtml/part03/PS3_3.html
23. Stanford AIMI. Shared datasets terms. https://aimi.stanford.edu/data
24. PhysioNet. MIMIC-CXR. https://physionet.org/content/mimic-cxr/
25. Module 21 survey file, section 6. [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)

## 17. Links

- Module map and topic table: [`_index.md`](_index.md)
- Previous topic, EHR, claims, and ICU data: [`03_ehr-claims-and-icu.md`](03_ehr-claims-and-icu.md)
- Next topic, signal datasets: [`05_signal-datasets.md`](05_signal-datasets.md)
- Signal datasets: [`05_signal-datasets.md`](05_signal-datasets.md)
- Root survey file: [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
- Labs, including the baseline reproducibility lab: [`../../labs/`](../../labs/)

