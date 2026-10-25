# Labels and Annotation

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, and 03
> Version 2026.3

## 1. Why this topic exists

A model cannot be better than its labels. In imaging, labels come from four families: a
radiologist's structured annotation on the image, a radiologist's free-text report, a pathology or
clinical outcome, and a registry or billing code. Each family has a different error structure, a
different unit of analysis, and a different cost. Mixing them without measuring the differences
produces a model that learns the label source instead of the disease.

The industry problem is that labels are treated as ground truth when they are evidence. A report
that says "no pneumothorax" may be written after a resident reviewed the image and a senior
radiologist signed without re-reading. A segmentation mask may be drawn by a trainee and edited by
an expert. A pathology label may come from a biopsy that sampled one part of a heterogeneous
tumour. A registry code may reflect billing practice rather than disease. None of these is a
neutral observation.

Label noise has a predictable effect. Random noise attenuates measured associations and biases
performance downward; systematic noise can inflate performance if the noise correlates with the
label or with the site. The first is a power problem, the second is a validity problem. The
distinction matters because the fixes are different: more labels versus better labels.

This topic teaches how to define a label, collect annotations, measure agreement, adjudicate
disagreements, derive labels from reports with validation, and freeze a label table with
provenance. It treats the label as a measurement instrument that must be calibrated.

### 1.1 The decisions this topic unblocks

| Decision | What label knowledge provides |
|---|---|
| Unit of analysis | Whether a label applies to a patient, study, series, lesion, or voxel |
| Reference standard | Which evidence can settle the label |
| Annotation protocol | What readers are asked to mark and how |
| Reader count | How much agreement and adjudication are needed |
| Report-derived labels | Whether NLP labels are accurate enough for the task |
| Uncertain cases | Whether to exclude, downweight, or model uncertainty |
| Split design | How to avoid the same patient or lesion crossing splits |
| Label version | Which label table a model was trained and evaluated on |

### 1.2 Detection, segmentation, and diagnosis are different label problems

A detection label is a set of locations or a per-image presence flag. A segmentation label is a
voxel mask. A diagnosis label is a disease state, often confirmed by pathology or follow-up. A
triage label is a priority or time-to-action. Each has a different annotation cost and a different
reference standard. A team that says "we need labels" without naming the task type will collect
the wrong ones.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** a label with a named construct, unit of analysis, reference standard, and
   uncertainty policy.
2. **Write** an annotation manual that a new reader can apply consistently, including positive,
   negative, and borderline examples.
3. **Measure** inter-reader agreement with a metric appropriate to the label type (kappa for
   categories, Dice for masks, intraclass correlation for continuous measurements).
4. **Adjudicate** disagreements with a documented rule and record the change rate.
5. **Derive** labels from radiology reports and quantify their sensitivity and positive predictive
   value against an image-level reference.
6. **Detect** label leakage from prior reports, follow-up information, and metadata.
7. **Prevent** split contamination by patient, study, and lesion.
8. **Freeze** a label table with provenance, version, and an audit trail.

## 3. The core mental model

A label is the output of an evidence chain. The chain starts with the patient and the clinical
question, passes through the image and the reader, and ends in a table cell. Every link adds
error, and the errors are not independent.

```text
Patient state
   │  (true disease, often unobservable)
   ▼
Image evidence  ── acquisition, resolution, contrast
   │
   ▼
Reader judgment ── training, fatigue, threshold, prevalence expectation
   │
   ├──► Structured annotation (mask, box, category)
   ├──► Free-text report ──► NLP labeler ──► label
   └──► Clinical outcome (pathology, follow-up, registry)
                         │
                         ▼
                    Label table
```

### 3.1 Two axes: construct validity and measurement reliability

**Construct validity** asks whether the label measures the thing the model is supposed to predict.
A report label for "pneumonia" may measure "the radiologist thought pneumonia was possible", not
"pneumonia confirmed by culture". A pathology label for "tumour grade" may measure the grade of
the sampled region, not the whole tumour.

**Measurement reliability** asks whether repeated measurements agree. Two readers may agree
perfectly on a construct that is defined vaguely, or disagree on a construct that is defined
precisely but hard to see. High reliability does not imply validity; low reliability limits it.

| | High validity | Low validity |
|---|---|---|
| **High reliability** | The ideal label | Consistent measurement of the wrong thing |
| **Low reliability** | Noisy measurement of the right thing | Neither |

### 3.2 The label hierarchy

| Level | Example | Unit | Typical metric |
|---|---|---|---|
| Patient | Has diabetes | patient | AUROC, sensitivity |
| Study | Study contains pneumothorax | study | AUROC, sensitivity |
| Series | Series shows contrast enhancement | series | AUROC |
| Lesion | Nodule at a location | lesion | sensitivity at FP/scan |
| Voxel | Tumour boundary | voxel | Dice, Hausdorff |
| Pixel | Microcalcification | pixel | Dice, detection rate |

A model evaluated at the wrong level can look good or bad for arithmetic reasons. A patient-level
AUROC ignores whether the model found the lesion; a lesion-level sensitivity ignores whether the
patient had one or five lesions. The unit must be stated with every metric.

### 3.3 Label noise is not one thing

| Noise type | Mechanism | Effect on measured performance | Detection |
|---|---|---|---|
| Random mislabel | Reader error independent of image | Attenuation toward chance | Repeat reads; adjudication |
| Systematic bias | Reader threshold differs by site | Site-specific shift | Stratify by reader and site |
| Construct mismatch | Label measures a proxy | Ceiling below clinical truth | Compare to outcome |
| Leakage | Label uses future or report information | Inflation | Temporal and text audit |
| Contamination | Same patient in train and test | Inflation | Patient-level split audit |
| Prevalence shift | Case-control enrichment | PPV distortion | Reweight to deployment prevalence |

Random noise is forgiving in large datasets; systematic noise is not. A model trained on labels
from one reader at one site learns that reader's threshold, and the threshold may not transfer.

### 3.4 Why report labels are attractive and dangerous

Reports are abundant, cheap, and temporally aligned with the image. They are also written for
communication, not for measurement. A report may say "cannot exclude", "stable", "interval
improvement", or "recommend follow-up". A rule-based labeler must decide what these mean, and
different labelers decide differently. The CheXpert labeler, for example, maps uncertainty to a
separate class rather than to positive or negative.

The danger is that report text often contains information not visible in the image: priors,
clinical history, comparison studies, and recommendations. A model trained on report labels may
learn to predict what the radiologist knew, not what the image shows.

### 3.5 Provenance is part of the label

A label without provenance is an assertion. Provenance answers four questions: who produced the
label, from what evidence, with what tool, and when. The answers determine whether the label can
be audited and whether it can be reproduced.

| Provenance field | Example | Why it matters |
|---|---|---|
| Reader identifier | pseudonymous r-017 | Detects reader-specific bias |
| Reader role | thoracic radiologist, 12 years | Interprets agreement |
| Evidence | thin-section CT, no prior | Defines what was visible |
| Tool and version | annotation v2.3 | Reproduces the mask |
| Date | 2026-03-14 | Detects protocol drift |
| Adjudication | consensus, third reader | Defines the final label |
| Confidence | certain, equivocal | Supports uncertainty policy |

A label table that records only the value and the patient identifier is not auditable. When a
model fails on a subgroup, the team cannot tell whether the failure is the model, the reader, or
the evidence.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Label | The target value assigned to a unit | Calling it ground truth | Depends on task |
| Reference standard | The evidence used to settle the label | Assuming it is error-free | Pathology, follow-up, consensus |
| Construct | The clinical concept the label intends to measure | Confusing with the label itself | Concept |
| Annotation | A reader's mark on an image | Treating as a label table | Mask, box, point |
| Segmentation | Voxel-level delineation | Confusing with detection | Mask |
| Detection | Localization of a finding | Confusing with diagnosis | Box or point |
| Diagnosis | Disease state assignment | Confusing with detection | Category |
| Triage | Priority assignment | Confusing with diagnosis | Ordinal |
| Weak label | A label derived from an indirect source | Treating as expert annotation | Report, registry |
| Report-derived label | Label extracted from free text | Treating as image truth | NLP output |
| Structured report | Coded report with discrete fields | Assuming completeness | DICOM SR, FHIR |
| Adjudication | Resolution of disagreement by consensus or third reader | Majority vote without rule | Process |
| Inter-reader agreement | Consistency between readers | Confusing with accuracy | Kappa, Dice, ICC |
| Intra-reader agreement | Consistency of one reader over time | Ignoring drift | Kappa, Dice |
| Cohen's kappa | Agreement beyond chance for two raters | Using for ordinal without weighting | Dimensionless |
| Fleiss' kappa | Agreement beyond chance for multiple raters | Using for continuous | Dimensionless |
| Intraclass correlation | Agreement for continuous measurements | Using for categories | Dimensionless |
| Dice | Overlap of two masks | Using for detection boxes | Voxel |
| Hausdorff distance | Maximum boundary distance | Using as a volume metric | mm |
| Sensitivity | True positives / reference positives | Confusing with PPV | Proportion |
| Specificity | True negatives / reference negatives | Confusing with NPV | Proportion |
| PPV | True positives / predicted positives | Assuming prevalence-free | Proportion |
| NPV | True negatives / predicted negatives | Assuming prevalence-free | Proportion |
| Prevalence | Reference positives / total | Using enriched prevalence for deployment | Proportion |
| Uncertainty label | Explicit "possible" or "cannot exclude" | Forcing to binary | Category |
| Positive | Finding present | Including uncertain | Category |
| Negative | Finding absent | Including "not mentioned" | Category |
| Borderline | Finding near a threshold | Treating as positive or negative silently | Category |
| Multi-lesion | More than one finding per study | Counting studies as lesions | Count |
| Laterality | Left or right | Ignoring in label definition | Code |
| Annotation provenance | Who annotated, when, with what tool | Omitting from the table | Metadata |
| Annotation manual | Written definition and examples | Leaving implicit | Document |
| Truthing | Establishing the reference standard | Assuming a single reader | Process |
| Label leak | Label contains information unavailable at prediction | Confusing with shortcut | Audit finding |
| Split contamination | Same unit in train and test | Splitting by image | Audit finding |
| Case-control enrichment | Sampling by outcome | Reporting PPV as if population | Design |

### 4.1 Label source comparison

| Source | Cost per case | Error structure | Unit | Best use |
|---|---|---|---|---|
| Expert voxel annotation | Very high | Boundary and omission | Voxel | Segmentation |
| Expert bounding box | High | Localization and omission | Lesion | Detection |
| Expert category | Medium | Threshold and category | Study or lesion | Diagnosis |
| Structured report | Low | Completeness and coding | Study | Screening |
| Free-text report + NLP | Very low | Extraction and negation | Study | Large-scale pretraining |
| Pathology | High | Sampling and grading | Patient or lesion | Diagnosis reference |
| Registry | Low | Coding and selection | Patient | Outcome studies |
| Billing claim | Very low | Coding and incentive | Patient | Never as sole reference |

### 4.2 RADS categories as label schemas

| System | Version | Label type | What a model can predict |
|---|---|---|---|
| Lung-RADS | v2022 | Ordinal category 0-4X plus S modifier | Nodule category, growth |
| BI-RADS | v2025 manual | Ordinal 0-6 plus descriptors | Density, category, finding |
| PI-RADS | v2.1 | Ordinal 1-5 per lesion | Clinically significant cancer risk |
| LI-RADS | CT/MRI v2018 | Ordinal LR-1 to LR-5, LR-M, LR-TR | HCC probability, treatment response |

A RADS category is not a binary label. A model that predicts "LR-4 or higher" has collapsed an
ordinal scale into a threshold, and the threshold must be stated with the metric.

## 5. The mechanics, step by step

This procedure builds a label table from a clinical question. It assumes the images passed the
ingestion checks in Topic 02. Each step states input, output, rationale, branches, and edge cases.

### Step 1. Define the construct and the unit of analysis

**Input:** The clinical question and the intended use.

**Output:** A one-paragraph construct definition, the unit (patient, study, series, lesion,
voxel), and the prediction time point.

**Rationale:** The construct determines the reference standard and the annotation task. "Nodule
present" at the study level is not the same as "nodule detected" at the lesion level, and neither
is "lung cancer diagnosed within two years".

**Branch points:**
- If the decision is screening, the unit is usually the patient or study, and the outcome may be
  interval cancer.
- If the decision is characterization, the unit is the lesion, and the reference may be pathology.
- If the decision is segmentation, the unit is the voxel, and the reference is an expert mask.

**Edge case:** A construct may require a time horizon. "Pneumonia" on an image is a different
construct from "pneumonia treated with antibiotics within 7 days". State the horizon.

### Step 2. Choose the reference standard

**Input:** The construct and the available evidence.

**Output:** A reference standard with a stated independence from the model input and a stated
error mode.

**Rationale:** The reference standard is the truth against which the label is scored. If it uses
the same information as the model, the evaluation is circular. If it is a proxy, the ceiling is
the proxy's validity.

**Branch points:**
- Pathology or culture: strong for diagnosis, but sampling error.
- Expert consensus on image: strong for detection, but reader-dependent.
- Clinical follow-up: strong for outcome, but delayed and loss to follow-up.
- Registry or billing: weak for diagnosis, useful for prevalence.

**Edge case:** A reference standard may be unavailable for negatives. A study with no follow-up
is not proven negative; it is unverified. Report the verification rate.

### Step 3. Write the annotation manual

**Input:** The construct and examples.

**Output:** A manual with definitions, inclusion and exclusion rules, positive and negative
examples, borderline examples, and the annotation tool's conventions.

**Rationale:** Readers apply their own thresholds unless the threshold is written. The manual is
the instrument's calibration. It must define what counts, what does not, and what to do when the
finding is equivocal.

**Branch points:**
- Detection: define the minimum size, the required number of planes, and whether to mark
  calcified or juxtapleural lesions.
- Segmentation: define boundary conventions, inclusion of vessels, and handling of cavities.
- Diagnosis: define the category thresholds and the handling of indeterminate cases.

**Edge case:** A manual that is too long is not read; a manual that is too short is not applied.
Pilot the manual on 20 cases and measure agreement before scaling.

### Step 4. Select and train readers

**Input:** The task, the reader pool, and the manual.

**Output:** Reader qualifications, training record, and a calibration set with agreement.

**Rationale:** Reader experience affects sensitivity and threshold. A trainee, a general
radiologist, and a subspecialist are different instruments. The reader pool must match the
intended clinical user or the reference standard must be stronger than the user.

**Branch points:**
- Subspecialist consensus: high quality, low throughput, high cost.
- General radiologists with training: moderate quality, higher throughput.
- Trainees with expert adjudication: scalable, but the adjudication burden is high.
- Crowd annotation: cheap, but needs qualification and audit.

**Edge case:** A reader who helped design the model is not an independent reader. Record conflicts
of interest and exclude readers who saw model outputs.

### Step 5. Design the annotation task and blinding

**Input:** The manual and the tool.

**Output:** A task that shows the relevant images, hides the label and the clinical history, and
records the reader's confidence.

**Rationale:** Blinding prevents the reader from copying the report or the prior. Confidence
allows the analysis to separate certain from uncertain labels. Hiding the clinical history is not
always possible or desirable; record what was visible.

**Branch points:**
- Fully blinded: best for label independence, but may be unrealistic.
- Clinical-history visible: more realistic, but the label may depend on history.
- Prior study visible: necessary for growth, but leaks the prior label.

**Edge case:** A growth label requires the prior. The prior is part of the evidence, not a leak,
but the model must receive the same prior at inference. If it does not, the label is unavailable
at prediction time.

### Step 6. Pilot and calibrate

**Input:** The manual, the tool, and a small sample.

**Output:** A calibration report with agreement, common disagreements, and manual revisions.

**Rationale:** The pilot finds ambiguous language and tool problems before the full annotation
budget is spent. A manual that produces low agreement on the pilot will not improve on the full
set without revision.

**Branch points:**
- Agreement high: proceed.
- Agreement moderate: revise examples and retrain.
- Agreement low: the construct or the task is ambiguous; redefine before scaling.

**Edge case:** A pilot with only easy cases overestimates agreement. Include difficult and
borderline cases deliberately.

### Step 7. Sample for prevalence and difficulty

**Input:** The candidate pool and the deployment prevalence.

**Output:** A sampling plan with the target prevalence, the difficult-case oversampling, and the
weights to restore the population.

**Rationale:** Case-control enrichment makes training efficient but distorts PPV and calibration.
The sampling plan must record the enrichment and the weights.

**Branch points:**
- Random sample: representative, but may have too few positives.
- Enriched sample: efficient, but requires weighting.
- Difficult-case oversample: improves boundary learning, but distorts prevalence.
- Site-balanced sample: improves generalization, but changes the population.

**Edge case:** A screening dataset with 1% prevalence needs a very large sample to yield enough
positives. Enrichment is reasonable, but the reported PPV must be reweighted to 1%.

### Step 8. Double-read a subset

**Input:** The sample and the reader pool.

**Output:** A double-read subset, ideally 10-20% of cases, stratified by difficulty and site.

**Rationale:** Double reading estimates agreement and finds systematic differences. A subset is
enough if it is representative; a non-representative subset underestimates disagreement.

**Branch points:**
- 10% random: adequate for common findings.
- 20% stratified: better for rare findings and subgroups.
- 100% double-read: expensive, used for high-stakes reference standards.

**Edge case:** The double-read subset must be selected before looking at the labels; otherwise the
selection is biased toward disagreements.

### Step 9. Measure agreement

**Input:** The double-read labels.

**Output:** Agreement metrics with confidence intervals, stratified by reader, site, and finding.

**Rationale:** Agreement quantifies the instrument's reliability. It does not quantify accuracy.
A kappa of 0.7 means the readers agree more than chance; it does not mean they are right.

**Branch points:**
- Binary labels: Cohen's or Fleiss' kappa.
- Ordinal labels: weighted kappa.
- Masks: Dice and Hausdorff distance.
- Continuous measurements: intraclass correlation.

**Edge case:** Kappa is prevalence-sensitive. A high kappa on a balanced sample can fall on a
low-prevalence sample even with the same readers. Report prevalence with kappa.

### Step 10. Adjudicate disagreements

**Input:** The disagreements and an adjudication rule.

**Output:** A final label, the adjudication method, and the change rate.

**Rationale:** Adjudication resolves the reference label. The method matters: majority vote,
consensus conference, and third-reader tie-break can produce different labels. Record which was
used.

**Branch points:**
- Consensus conference: best for complex cases, slow.
- Third reader: scalable, but the third reader may be wrong.
- Majority vote: simple, but can outvote the correct minority.
- Rule-based: for example, any positive wins; fast, but biased toward sensitivity.

**Edge case:** Adjudication can introduce a new bias if the adjudicator sees the model output.
Blind the adjudicator to the model.

### Step 11. Derive report labels and validate them

**Input:** Reports and a labeler.

**Output:** Report-derived labels plus a validation sample with sensitivity, specificity, PPV,
and NPV against an image-level reference.

**Rationale:** Report labels are cheap but noisy. The validation sample quantifies the noise. A
labeler with 0.97 mention-extraction F1 can still have low image-level PPV because the report
mentions findings that are not visible or that are historical.

**Branch points:**
- Rule-based labeler: transparent, but brittle to language variation.
- Transformer labeler: better recall, but needs validation and versioning.
- LLM labeler: flexible, but nondeterministic and expensive to audit.
- Structured report: best, but coverage is limited.

**Edge case:** A report may mention a finding in the history or the impression only. The labeler
must distinguish current findings from historical ones. Negation and uncertainty are separate
problems.

### Step 12. Handle uncertainty explicitly

**Input:** The labels and the confidence.

**Output:** A policy: exclude, downweight, treat as positive, treat as negative, or model as a
distribution.

**Rationale:** Forcing uncertain cases into a binary label creates noise that is correlated with
difficulty. Excluding them changes prevalence and may remove the hardest cases. There is no
universal answer; the policy must be stated.

**Branch points:**
- Exclude: clean labels, biased sample.
- Downweight: uses all data, requires a loss that supports weights.
- Positive: sensitive, but inflates false positives.
- Negative: specific, but inflates false negatives.
- Soft label: preserves uncertainty, requires a probabilistic loss.

**Edge case:** A three-class label (positive, negative, uncertain) can be trained as ordinal or as
two binary tasks. The choice changes the metric and the clinical interpretation.

### Step 13. Freeze the label table with provenance

**Input:** The final labels and the audit trail.

**Output:** A versioned table with one row per unit, the label, the source, the reader, the date,
the confidence, the adjudication flag, and the label version.

**Rationale:** A label table without provenance cannot be audited or reproduced. The version must
change when any label changes, and the model artifact must record the version it used.

**Branch points:**
- Research label table: freeze at analysis time.
- Production label table: update with a change log and a re-evaluation.
- Public release: document the annotation protocol and the agreement.

**Edge case:** A label correction after model training invalidates the evaluation. The correct
response is to retrain or to report both versions, not to overwrite silently.

### 5.1 Standardize the display before annotation

Readers annotate what they see, and what they see depends on the display. A nodule may be obvious
on a lung window and invisible on a soft-tissue window. If readers choose their own window, the
label becomes a function of the display setting as well as the image. The annotation tool should
open each modality with a protocol-specified window and allow the reader to change it only with a
recorded action.

The same applies to zoom and slice scrolling. A reader who scrolls quickly may miss a nodule that
is visible on two adjacent slices. The tool should record the slices viewed and the time spent.
These fields are not a substitute for a second read, but they help explain disagreements.

**Edge case:** A display with a non-diagnostic monitor or an uncalibrated monitor changes the
perceived contrast. For high-stakes annotation, use a diagnostic display and record its
calibration. For research annotation, record the display model and the window preset.

### 5.2 Worked branch: how large a validation sample for a report labeler?

A report labeler with sensitivity 0.79 and specificity 0.96 is being validated against image
review. The team wants a 95% confidence interval half-width of 0.05 for sensitivity. The required
number of reference-positive studies is approximately:

```text
n = z² × p(1-p) / e²
  = 1.96² × 0.79 × 0.21 / 0.05²
  = 3.8416 × 0.1659 / 0.0025
  = 254.9
```

So about 255 positive studies are needed for the sensitivity estimate alone. At a 24% prevalence,
that requires about 1,063 studies. A validation sample of 100 studies, of which 24 are positive,
has a sensitivity confidence interval half-width of about:

```text
1.96 × sqrt(0.79 × 0.21 / 24) = 1.96 × 0.0831 = 0.163
```

The interval is too wide to support a claim. The practical options are to enrich the validation
sample for positives, to accept a wider interval, or to validate only the metric that the decision
requires. The choice must be stated before the sample is drawn.

### 5.3 Decision table: label strategy by task

| Task | Annotation | Reference standard | Unit | Minimum quality check |
|---|---|---|---|---|
| Lung nodule detection | Bounding box or point per nodule | Three-reader consensus on CT | Lesion | Sensitivity at fixed FP/scan |
| Lung nodule segmentation | Voxel mask | Expert editor | Voxel | Dice and boundary distance |
| Pneumonia diagnosis | Study category | Expert image review plus culture | Study | Kappa and PPV |
| Breast cancer screening | BI-RADS category | Pathology or interval cancer | Patient | Recall rate and PPV |
| Prostate cancer risk | PI-RADS score | Targeted biopsy | Lesion | Agreement and AUC |
| Liver lesion diagnosis | LI-RADS category | Pathology or imaging follow-up | Lesion | Category agreement |
| Tumour grading | Slide-level grade | Pathologist consensus | Slide | Kappa |
| Retinopathy | ETDRS grade | Reading centre | Eye | Sensitivity and specificity |

### 5.4 Edge cases that require a written rule

**Mixed slice thickness.** A nodule measured on 5 mm slices is not the same label as one measured
on 1 mm slices. Define the acquisition requirement and exclude or stratify the rest.

**Missing series.** A study without the required sequence is unverified, not negative. Record the
reason.

**Inconsistent spacing.** A size label depends on the spacing. Recompute the size from the
position-derived geometry before applying a threshold.

**Rescale absent.** A HU-based label (for example, fat or calcification) cannot be applied without
a unit. Exclude.

**Burned-in annotation.** A burned-in measurement or laterality marker can bias the reader. Mask
it during annotation.

**Laterality mismatch.** Resolve before assigning a side-specific label. A mismatch is a
data-quality event.

**Multi-frame objects.** A multi-frame object may contain several lesions or several phases.
Annotate per frame or per lesion, not per object.

**Private tags carrying labels.** A private tag with the report impression is a label leak. Remove
it before annotation and before training.

## 6. Worked example

### 6.1 The case

A lung nodule detection model is being built on 500 low-dose CT studies from two sites. Three
radiologists annotate every study for solid nodules of at least 4 mm. A consensus panel adjudicates
disagreements. The team also has report-derived labels from a rule-based NLP pipeline. The
clinical question is whether the model can replace a first read for nodule detection at a fixed
false-positive rate.

### 6.2 Reference labels and report labels

The consensus reference finds 120 studies with at least one solid nodule of 4 mm or more and 380
without. The report label is positive for 110 studies. Cross-tabulation against the reference:

| | Reference positive | Reference negative | Total |
|---|---:|---:|---:|
| Report positive | 95 | 15 | 110 |
| Report negative | 25 | 365 | 390 |
| Total | 120 | 380 | 500 |

Sensitivity of the report label:

```text
95 / 120 = 0.792
```

Specificity:

```text
365 / 380 = 0.961
```

Positive predictive value:

```text
95 / 110 = 0.864
```

Negative predictive value:

```text
365 / 390 = 0.936
```

Accuracy:

```text
(95 + 365) / 500 = 460 / 500 = 0.920
```

The report label misses 25 of 120 positive studies and falsely labels 15 of 380 negatives. It is
good enough for pretraining and not good enough to be the reference for a regulated claim.

### 6.3 Inter-reader agreement

On a 200-study double-read subset, reader A calls 110 positive and reader B calls 105. Both call
95 positive:

| | B positive | B negative | Total |
|---|---:|---:|---:|
| A positive | 95 | 15 | 110 |
| A negative | 10 | 80 | 90 |
| Total | 105 | 95 | 200 |

Observed agreement:

```text
(95 + 80) / 200 = 175 / 200 = 0.875
```

Expected agreement by chance:

```text
P(A+) = 110/200 = 0.55
P(B+) = 105/200 = 0.525
P(expected) = 0.55 × 0.525 + 0.45 × 0.475
            = 0.28875 + 0.21375
            = 0.5025
```

Cohen's kappa:

```text
(0.875 - 0.5025) / (1 - 0.5025) = 0.3725 / 0.4975 = 0.749
```

Kappa 0.749 is substantial agreement. It is not accuracy: the readers could both be wrong in the
same direction. The prevalence in this subset is 55%, much higher than the 24% in the full cohort,
so the kappa is not directly comparable to a low-prevalence sample.

### 6.4 Segmentation agreement

For the 120 positive studies, one reader's mask is compared with the adjudicated mask. For a
representative 9.4 mm nodule, the reference mask has 303 voxels at 1 mm and the reader mask has
250 voxels, with an intersection of 215:

```text
Dice = 2 × 215 / (303 + 250)
     = 430 / 553
     = 0.777
```

Dice 0.777 is a voxel-level agreement metric, not a model metric. It sets an upper bound on how
well a model trained on this reader's masks can agree with the adjudicated reference: even a
perfect imitation of the reader would score about 0.78 against the adjudicated mask.

### 6.5 Prevalence and PPV

The model's operating point has sensitivity 0.90 and specificity 0.95 on the reference. At the
cohort prevalence of 24% (120/500), PPV is:

```text
PPV = (0.90 × 0.24) / (0.90 × 0.24 + 0.05 × 0.76)
    = 0.216 / (0.216 + 0.038)
    = 0.216 / 0.254
    = 0.850
```

At a deployment prevalence of 5%, as in a low-risk screening population:

```text
PPV = (0.90 × 0.05) / (0.90 × 0.05 + 0.05 × 0.95)
    = 0.045 / (0.045 + 0.0475)
    = 0.045 / 0.0925
    = 0.486
```

At an enriched prevalence of 50%:

```text
PPV = (0.90 × 0.50) / (0.90 × 0.50 + 0.05 × 0.50)
    = 0.45 / (0.45 + 0.025)
    = 0.947
```

The same sensitivity and specificity produce PPV 0.486, 0.850, or 0.947 depending on prevalence.
A team that reports the enriched PPV as if it were deployment performance has misrepresented the
product.

### 6.6 Diagnostic 1: label leak from comparison reports

The report-derived label is extracted from the impression. In 38 of the 120 positive studies, the
impression says "stable nodule, compare with prior". The word "stable" requires the prior study,
which is not available to the model at inference. A model trained on these labels can learn the
word "stable" only if the report text is an input, which it is not. The leak is in the label
definition, not the image. The fix is to define the label as "nodule present on this study" and to
re-read the report for current findings, excluding comparison language.

The quantitative diagnostic is the difference between report-label performance and image-only
performance on the 38 studies: the report label agrees with consensus in 36 of 38, while the
image model agrees in 29 of 38. The gap is the prior information.

### 6.7 Diagnostic 2: split contamination

The dataset has 500 studies from 460 patients. Forty patients have two studies. A random split by
study puts 18 of those 40 patients in both training and test. The test set then contains 18
patients whose anatomy the model has seen. The inflation is measurable: test sensitivity is 0.93
with contamination and 0.88 after a patient-level split, a 5-point difference that is larger than
the confidence interval on the clean estimate.

The diagnostic is a join between the training and test patient identifiers. If the intersection is
non-empty, the split is contaminated. The fix is to split by patient, then by study, and to record
the split hash.

### 6.8 Varying one assumption

Assume the uncertain cases are excluded instead of adjudicated. Of the 500 studies, 42 are
equivocal: 18 are called positive by the consensus after adjudication, and 24 are called negative.
Excluding them changes the denominator to 458 and the positive count to 102. The prevalence
becomes:

```text
102 / 458 = 0.223
```

The model's sensitivity on the reduced set rises from 0.90 to 0.93 because the hard cases are
gone. The PPV at 22.3% prevalence becomes:

```text
(0.93 × 0.223) / (0.93 × 0.223 + 0.05 × 0.777)
= 0.2074 / (0.2074 + 0.0389)
= 0.842
```

The conclusion changes from "sensitivity 0.90, PPV 0.85" to "sensitivity 0.93, PPV 0.84 on the
easier subset". The improvement is an artefact of exclusion, not a better model. The correct
report states the exclusion rate and evaluates on the full set with an uncertainty policy.

### 6.9 The cost of annotation

Assume a thoracic radiologist takes 4 minutes to review a CT for nodules and 12 minutes to draw a
voxel mask for one lesion. At a loaded rate of 180 USD per hour, the cost per study is:

```text
4 minutes / 60 × 180 = 12 USD per detection read
12 minutes / 60 × 180 = 36 USD per segmentation
```

For 500 studies with a double read on 20%, the detection cost is:

```text
500 × 12 × 1.2 = 7,200 USD
```

For 120 positive studies with one mask each and a 20% double-draw:

```text
120 × 36 × 1.2 = 5,184 USD
```

The segmentation labels cost about 72% as much as the detection labels for 24% of the studies.
That ratio explains why segmentation datasets are smaller and why segmentation models are often
trained with weak or sparse labels. It also explains why a request for "masks for everything" is
usually not funded.

## 7. Code, query, or calculation

The following code computes the label metrics in Section 6: the 2×2 report-label performance,
Cohen's kappa, Dice, and PPV across prevalence. It uses only the standard library.

```python
def binary_metrics(tp, fp, fn, tn):
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return sensitivity, specificity, ppv, npv, accuracy

tp, fp, fn, tn = 95, 15, 25, 365
sens, spec, ppv, npv, acc = binary_metrics(tp, fp, fn, tn)
print("report sensitivity = {:.3f}".format(sens))
print("report specificity = {:.3f}".format(spec))
print("report PPV         = {:.3f}".format(ppv))
print("report NPV         = {:.3f}".format(npv))
print("report accuracy    = {:.3f}".format(acc))

# Cohen's kappa for the 200-study double-read subset
a, b, c, d = 95, 15, 10, 80
n = a + b + c + d
po = (a + d) / n
p_a_pos = (a + b) / n
p_b_pos = (a + c) / n
pe = p_a_pos * p_b_pos + (1 - p_a_pos) * (1 - p_b_pos)
kappa = (po - pe) / (1 - pe)
print("observed agreement = {:.4f}".format(po))
print("expected agreement = {:.4f}".format(pe))
print("kappa              = {:.4f}".format(kappa))

# Dice for one lesion
ref, pred, inter = 303, 250, 215
dice = 2 * inter / (ref + pred)
print("Dice               = {:.4f}".format(dice))

# PPV across prevalence at fixed sensitivity and specificity
def ppv_at(prev, sens, spec):
    return (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))

for prev in (0.05, 0.24, 0.50):
    print("prevalence {:.0%} -> PPV {:.3f}".format(prev, ppv_at(prev, 0.90, 0.95)))

# Excluding uncertain cases changes the denominator
print("excluded prevalence = {:.4f}".format(102 / 458))
print("PPV at 22.3%        = {:.3f}".format(ppv_at(102 / 458, 0.93, 0.95)))
```

Expected output:

```text
report sensitivity = 0.792
report specificity = 0.961
report PPV         = 0.864
report NPV         = 0.936
report accuracy    = 0.920
observed agreement = 0.8750
expected agreement = 0.5025
kappa              = 0.7487
Dice               = 0.7776
prevalence 5% -> PPV 0.486
prevalence 24% -> PPV 0.850
prevalence 50% -> PPV 0.947
excluded prevalence = 0.2227
PPV at 22.3%        = 0.842
```

The code assumes: the reference labels are correct, the double-read subset is representative, the
Dice volumes are at the same spacing, and the model's sensitivity and specificity are constant
across prevalence. The last assumption is the one that fails most often in deployment, because
spectrum effects change both sensitivity and specificity.

## 8. Data, standards, and regulatory context

### 8.1 DICOM structured reports and segmentation objects

DICOM defines Structured Report Storage for coded findings and Segmentation Storage for voxel
masks. A structured report can carry measurements, categories, and references to images. A
segmentation object carries a label map and a reference to the source series. Both are preferable
to a CSV of labels because they preserve the link to the image and the geometry.

The DICOM Standard PS3 2026d defines these objects. It does not define which findings must be
coded or how a RADS category maps to a code. Those mappings come from the RADS lexicons and from
professional society guidance.

### 8.2 RADS lexicons

| System | Current version | Key label constraint |
|---|---|---|
| Lung-RADS | v2022 | Nodule type and size thresholds; growth definition >1.5 mm mean diameter |
| BI-RADS | v2025 manual, extending the 5th edition | Density and category; recall and PPV audit |
| PI-RADS | v2.1 (2019) | Sequence requirements; lesion score 1-5 |
| LI-RADS | CT/MRI v2018 core, lexicon 2021 | Diagnostic algorithm; LR-1 to LR-5, LR-M, LR-TR |

A model label should be traceable to a RADS category and to the version. A label of "suspicious
nodule" is not reproducible; a label of "Lung-RADS 4A, solid 8-<15 mm" is.

### 8.3 FDA evidence expectations

The FDA special controls for computer-assisted detection and diagnosis software at 21 CFR 892.2090
require standalone performance testing, localization accuracy where applicable, and reader-study
evidence where the device claims improved assisted read. The reference standard for those studies
must be described, and the reader population must match the intended user. A label table without
a reference standard description does not meet the expectation.

The FDA also requires subgroup analyses by anatomy, demographics, medical history, user
experience, and imaging equipment. Those subgroups must be present in the label table; a label
table without age, sex, and scanner fields cannot support the analysis.

### 8.4 Privacy and provenance

Annotation data are often stored outside the PACS, in a spreadsheet or a research database. Those
stores must follow the same de-identification rules as the images. A label table with patient
names, accession numbers, and free-text comments is a privacy risk. The table should use the same
remapped identifiers as the image manifest and should record the annotation date, reader, and tool
version.

### 8.5 What the standards do not cover

No standard defines how many readers are enough, how to adjudicate, or how to handle uncertainty.
Those are design decisions that must be justified by the task and reported with the label table.
The standards give the vocabulary and the object formats; the team gives the measurement protocol.

## 9. Industry practice and case evidence

### 9.1 CheXpert and the uncertain label

CheXpert (Irvin et al., AAAI 2019) released 224,316 chest radiographs with 14 observations labelled
positive, negative, or uncertain. The labeler's mention-extraction F1 was 0.969 micro-average and
0.948 macro-average on a 1,000-report evaluation set; uncertainty detection F1 was 0.848 micro and
0.770 macro. The dataset's main contribution was not the labels themselves but the demonstration
that uncertainty is a first-class state.

The industry lesson: a three-state label requires a policy. CheXpert trained with uncertainty
treated in several ways and showed that the choice changes performance. A team that collapses
uncertain to negative silently changes the task.

### 9.2 CheXbert and labeler improvement

CheXbert (Smit et al., EMNLP 2020) fine-tuned a BERT model on expert annotations and reported an
F1 of 0.798 on a MIMIC-CXR external test set, compared with 0.743 for the rule-based CheXpert
labeler and 0.805 for the radiologist benchmark. The improvement is real but modest, and it shows
that report labels have a ceiling: even a strong model is close to the radiologist benchmark, not
above it. The remaining gap is the difference between what the report says and what the image
shows.

### 9.3 Report versus image disagreement

Olatunji et al. (MIDL 2019) compared radiologist-on-image, radiologist-on-report, and NLP-on-report
labels for 1,000 chest radiographs. They found large image-versus-report disagreement, with the
NLP capturing only about 55.6% of image findings. The result is a direct estimate of the label
noise that a report-derived dataset inherits. It does not mean report labels are useless; it means
their image-level PPV must be measured, not assumed.

### 9.4 RADS-based labels in screening

Lung-RADS v2022 defines categories by nodule type and size, with a growth definition of more than
1.5 mm mean diameter increase within 12 months. A model trained to predict "cancer" from a
screening CT is not predicting Lung-RADS; it is predicting an outcome that depends on the
category, the follow-up, and the treatment. The label must match the claim. A product that says
"Lung-RADS 4A or higher" has a different reference standard and a different clinical meaning from
one that says "biopsy-proven cancer".

### 9.5 Inference, not fact

The following are inferences: that a specific report labeler will have a specific PPV at a
specific site; that a specific adjudication rule will reduce label noise; that a specific RADS
category will map cleanly to a model output. Each requires local validation. The documented facts
are the published labeler metrics and the RADS definitions.

### 9.6 Public challenges and annotation scale

The RSNA intracranial haemorrhage challenge (2019) released more than 25,000 head CT exams
annotated by more than 60 volunteer radiologists, with labels for five haemorrhage subtypes and
"any". The RSNA 2024 lumbar spine degenerative classification challenge attracted 1,874 teams and
annotated three conditions across MRI; the RSNA 2025 intracranial aneurysm challenge used CTA,
MRA, and MRI from 18 sites annotated by 60 expert radiologists.

These challenges show that large, multi-reader annotation is possible when the task is narrow and
the annotation interface is standardized. They also show the limit: the labels are still
reader-derived, the test sets are curated, and the winning models are not automatically
regulatory-grade. The annotation scale buys statistical power, not construct validity.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Construct mismatch | Label definition differs from claim | Proxy used as truth | Ceiling below clinical value | Align label and intended use |
| Single-reader reference | No double-read subset | Reader threshold becomes truth | Bias toward that reader | Double-read and adjudicate |
| Report label treated as image truth | High PPV assumed | Report includes history and priors | Inflated performance | Validate on image review |
| Uncertain forced to negative | Prevalence drops | Ambiguous cases excluded from positives | Sensitivity inflated | Explicit uncertainty policy |
| Uncertain forced to positive | Prevalence rises | Ambiguous cases counted | PPV deflated | Explicit uncertainty policy |
| Patient split contamination | Same patient in train and test | Split by image or study | Inflated test performance | Split by patient |
| Multi-lesion counted as one | Study-level label | Lesion ignored | Missed lesions | Lesion-level labels |
| Laterality ignored | Side not recorded | Label cannot distinguish sides | Wrong-side errors | Record laterality |
| Annotation tool bias | Default window differs | Reader sees different image | Site-specific labels | Standardize display |
| Reader sees model output | Labels correlate with model | Circularity | Inflated agreement | Blind readers |
| Adjudicator sees model output | Final label biased | Circularity | Invalid reference | Blind adjudicator |
| Label version overwritten | No change log | Silent correction | Unreproducible evaluation | Version the table |
| Case-control PPV reported | Enriched prevalence | Sampling not reweighted | Misleading deployment claim | Reweight to population |
| Prior report leak | "Stable" or "unchanged" | Label needs prior | Unavailable at inference | Define current-study label |
| Tool resamples masks | Boundary shift | Tool default spacing | Misaligned voxel labels | Compare affine to image |
| Reader conflict of interest | Labels match model | Reader saw outputs | Circular reference | Blind and declare |

### 10.1 Red flags that stop a labeling project

1. The label definition does not name the unit of analysis.
2. The reference standard is the same report the model will read.
3. No double-read subset and no agreement estimate.
4. The split is by image, not by patient.
5. Uncertain cases are silently assigned to a binary class.
6. The label table has no reader, date, or version.
7. The reported PPV uses the enriched prevalence.

### 10.2 A label audit in twelve checks

1. Every row has a unit, a source, a reader, a date, and a version.
2. The reference standard is named and is independent of the model input.
3. A double-read subset exists and is representative.
4. Agreement is reported with prevalence and confidence intervals.
5. Adjudication rule and change rate are recorded.
6. Uncertain cases have an explicit policy.
7. The split is by patient, and the patient overlap is zero.
8. Report-derived labels have image-level sensitivity and PPV.
9. Prior-comparison language is excluded from current-study labels.
10. Laterality is recorded where the finding is lateralized.
11. Multi-lesion studies are counted at the lesion level.
12. A label correction creates a new version and a re-evaluation.

The audit is cheap and catches the failures that inflate performance. A label table that passes it
is not necessarily correct, but a table that fails it cannot support a credible claim.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The construct is observable | Label can be defined | Latent or multifactorial disease | Compare with outcome |
| The reference standard is independent | Valid evaluation | Same evidence as model | Audit information flow |
| Readers apply the manual | Reliable labels | Ambiguous manual | Pilot agreement |
| The double-read subset is representative | Agreement estimate | Selected on disagreement | Random or stratified sample |
| Kappa is comparable across strata | Pooled agreement | Prevalence differences | Report per stratum |
| Report labels reflect current findings | Label validity | History and priors | Image-level validation |
| The split is patient-level | No contamination | Multiple studies per patient | Identifier join |
| Uncertainty policy is explicit | Reproducible prevalence | Silent collapse | Document and sensitivity test |
| The label version is fixed | Reproducible evaluation | Silent corrections | Change log and hash |
| The sample matches deployment | Calibration | Case-control enrichment | Reweight or resample |
| Readers are independent of the model | Valid reference | Readers saw model output | Conflict audit |
| The annotation tool preserves geometry | Mask validity | Tool resamples silently | Compare affine and spacing |
| The label table uses remapped identifiers | Privacy | Direct identifiers retained | Identifier audit |

### 11.1 Limits of this topic

This topic covers label design, annotation, agreement, and report-derived labels. It does not
cover the full psychometrics of reader studies, the statistics of latent class models, or the
legal requirements for informed consent. It gives the data scientist the vocabulary to specify
and audit a label table, not to run a pivotal reader study.

### 11.2 What would change the conclusions in this topic

If the reference standard were pathology rather than expert consensus, the label noise structure
would change from reader disagreement to sampling error, and the achievable ceiling would change.
If the deployment prevalence were 1% rather than 5%, the PPV at the same operating point would
fall further, and the alert burden would dominate the workflow. If the model were evaluated at the
lesion level rather than the study level, the sensitivity would fall because a study is positive
when any lesion is found. Each change alters the conclusion without changing the model.

## 12. Exercises

### Exercise 1. Construct definition (recall, 20 minutes)

**Deliverable:** A one-paragraph definition of "pneumonia" for a chest X-ray model, including the
unit, the time horizon, and the reference standard.

**Acceptance criteria:** Unit stated; time horizon stated; reference standard named; uncertain
cases addressed.

**Time:** 20 minutes.

### Exercise 2. 2×2 metrics (calculation, 25 minutes)

**Deliverable:** For 80 true positives, 20 false positives, 40 false negatives, and 860 true
negatives, compute sensitivity, specificity, PPV, NPV, and accuracy.

**Acceptance criteria:** Correct values (0.667, 0.977, 0.800, 0.956, 0.940); units stated.

**Time:** 25 minutes.

### Exercise 3. Kappa calculation (calculation, 25 minutes)

**Deliverable:** For a 2×2 reader table with a=60, b=15, c=10, d=115, compute observed agreement,
expected agreement, and kappa.

**Acceptance criteria:** Correct values (0.875, 0.545, 0.725); interpretation stated.

**Time:** 25 minutes.

### Exercise 4. Annotation manual (design, 60 minutes)

**Deliverable:** A two-page manual for lung nodule annotation, including inclusion, exclusion,
borderline examples, and tool conventions.

**Acceptance criteria:** At least five rules; at least two borderline examples; a new reader could
apply it without asking.

**Time:** 60 minutes.

### Exercise 5. Report-label validation (application, 50 minutes)

**Deliverable:** A plan to validate a report-derived label against 100 image-reviewed studies,
including the sampling strategy and the metrics.

**Acceptance criteria:** Sample is stratified; metrics include sensitivity and PPV; prevalence is
reported; labeler version recorded.

**Time:** 50 minutes.

### Exercise 6. Split audit (application, 30 minutes)

**Deliverable:** A script or query that checks patient overlap between train and test and reports
the number of overlapping patients and studies.

**Acceptance criteria:** Detects planted overlap; reports both counts; states the consequence.

**Time:** 30 minutes.

### Exercise 7. Uncertainty policy (design, 40 minutes)

**Deliverable:** A policy for equivocal cases with the rationale, the expected effect on
prevalence, and the sensitivity analysis.

**Acceptance criteria:** Policy is explicit; effect quantified; alternative policy compared.

**Time:** 40 minutes.

### Exercise 8. Label table schema (design, 45 minutes)

**Deliverable:** A schema with one row per unit and columns for label, source, reader, date,
confidence, adjudication, and version.

**Acceptance criteria:** Every column typed; provenance complete; version and hash present; no
direct identifiers.

**Time:** 45 minutes.

## 13. Self-check answers

**A1. What is the difference between a label and a reference standard?**
A label is the value assigned to a unit for training or evaluation. A reference standard is the
evidence used to establish the truth. The label may be derived from the reference standard or from
a proxy; the reference standard is the stronger evidence.

**A2. Compute sensitivity and PPV for 80 TP, 20 FP, 40 FN, 860 TN.**
Sensitivity = 80 / 120 = 0.667. PPV = 80 / 100 = 0.800. Specificity = 860 / 880 = 0.977. NPV =
860 / 900 = 0.956. Accuracy = 940 / 1000 = 0.940.

**A3. Why is kappa prevalence-sensitive?**
Expected agreement depends on the marginal probabilities, which depend on prevalence. The same
readers with the same accuracy can produce a higher kappa on a balanced sample than on a
low-prevalence sample. Report prevalence with kappa.

**A4. Why is a patient-level split necessary when studies are the unit?**
A patient with two studies can appear in both training and test, so the model sees the same
anatomy twice. The test set is no longer independent, and performance is inflated.

**A5. What are the three states in CheXpert, and why do they matter?**
Positive, negative, and uncertain. The uncertain state preserves the radiologist's ambiguity.
Collapsing it to negative or positive changes prevalence and the task, and the choice must be
reported.

**A6. What is the difference between Dice and kappa?**
Dice measures voxel overlap between two masks. Kappa measures categorical agreement beyond chance
between two readers. Dice is used for segmentation; kappa is used for categories.

**A7. A report says "stable nodule, compare with prior". Why is this a leak for a single-study
model?**
The label depends on the prior study, which the model does not receive. The model cannot learn the
prior information from the current image alone, so the label is partly unavailable at prediction
time.

**A8. How do you handle a multi-lesion study in a detection metric?**
Use a lesion-level reference and a lesion-level metric, such as sensitivity at a fixed number of
false positives per scan. State the matching rule (for example, intersection over union ≥0.3) and
the unit.

**A9. Design rubric for Exercise 4.** A strong manual defines size, type, and location rules,
gives positive, negative, and borderline examples, and states tool conventions. A weak manual
says "mark all nodules" without a threshold.

**A10. Design rubric for Exercise 7.** A strong policy states the treatment, quantifies the
prevalence change, compares at least two policies, and reports the effect on sensitivity and PPV.
A weak policy says "exclude uncertain cases" without analysis.

## 14. Interview questions

**Q1. How do you define a label for an imaging model?**
A strong answer names the construct, unit, reference standard, time horizon, and uncertainty
policy. A weak answer says "the radiologist's diagnosis".

**Q2. Why are report labels noisy?**
A strong answer explains that reports contain history, priors, recommendations, and ambiguous
language, and that NLP extraction adds its own errors. A weak answer says "radiologists make
mistakes".

**Q3. How many readers do you need?**
A strong answer ties the number to the task, the disagreement rate, and the adjudication budget,
and cites a double-read subset with agreement. A weak answer says "two is enough".

**Q4. What is adjudication, and how do you avoid bias?**
A strong answer describes consensus or third-reader rules, blinding to model output, and recording
the change rate. A weak answer says "the senior radiologist decides".

**Q5. How do you measure segmentation label quality?**
A strong answer uses Dice and Hausdorff distance between readers, reports the reader-to-reference
agreement, and treats it as a ceiling. A weak answer reports only model Dice.

**Q6. What is split contamination in imaging?**
A strong answer explains that multiple studies or lesions per patient can cross splits, and that
the split must be by patient. A weak answer says "we split randomly".

**Q7. How do you handle uncertain labels?**
A strong answer presents exclude, downweight, positive, negative, and soft-label policies with
their effects. A weak answer picks one without justification.

**Q8. What is the unit of analysis for a detection metric?**
A strong answer says lesion for sensitivity and scan for false positives per scan, and states the
matching rule. A weak answer says "image".

**Q9. How do you validate a report-derived label?**
A strong answer samples studies for image review, computes sensitivity and PPV against the
reference, stratifies by finding, and records the labeler version. A weak answer compares to
another NLP model.

**Q10. Why does prevalence change PPV but not sensitivity?**
A strong answer explains that sensitivity is conditional on disease and PPV is conditional on a
positive prediction, so PPV depends on the prior probability. A weak answer confuses the two.

## 15. Key terms

- **Adjudication** — resolution of disagreement.
- **Annotation** — reader mark on an image.
- **Annotation manual** — written label definition and examples.
- **Case-control enrichment** — sampling by outcome.
- **Cohen's kappa** — chance-corrected agreement for two raters.
- **Construct** — clinical concept a label measures.
- **Detection** — localization of a finding.
- **Dice** — voxel overlap metric.
- **Fleiss' kappa** — agreement for multiple raters.
- **Hausdorff distance** — maximum boundary distance.
- **Inter-reader agreement** — consistency between readers.
- **Intraclass correlation** — agreement for continuous measurements.
- **Label** — target value for a unit.
- **Label leak** — label contains unavailable information.
- **Laterality** — left or right.
- **Multi-lesion** — more than one finding per study.
- **NPV** — negative predictive value.
- **PPV** — positive predictive value.
- **Prevalence** — proportion of positives in the reference.
- **Reference standard** — evidence used to settle the truth.
- **Report-derived label** — label extracted from free text.
- **Segmentation** — voxel-level delineation.
- **Sensitivity** — true positive rate.
- **Specificity** — true negative rate.
- **Split contamination** — same unit in train and test.
- **Structured report** — coded report with discrete fields.
- **Triage** — priority assignment.
- **Truthing** — establishing the reference standard.
- **Uncertainty label** — explicit ambiguous state.
- **Weak label** — indirect label source.

## 16. Sources

- Irvin, J. et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert
  Comparison," AAAI 2019. https://doi.org/10.1609/aaai.v33i01.3301590
- Smit, A. et al., "Combining Automatic Labelers and Expert Annotations for Accurate Radiology
  Report Labeling Using BERT," EMNLP 2020. https://doi.org/10.18653/v1/2020.emnlp-main.117
- Olatunji, T. et al., "Caveats in Generating Medical Imaging Labels from Radiology Reports with
  Natural Language Processing," MIDL 2019. https://arxiv.org/abs/1905.02283 (accessed 25 October
  2026).
- American College of Radiology, Lung-RADS v2022.
  https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/Lung-RADS
  (accessed 25 October 2026).
- American College of Radiology, BI-RADS v2025 manual.
  https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/BI-RADS
  (accessed 25 October 2026).
- American College of Radiology, PI-RADS v2.1.
  https://cs.acr.org/Clinical-Resources/Reporting-and-Data-Systems/PI-RADS (accessed 25 October
  2026).
- American College of Radiology, LI-RADS CT/MRI v2018 and 2021 lexicon.
  https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/LI-RADS
  (accessed 25 October 2026).
- DICOM Standard, PS3 current edition 2026d, Structured Report and Segmentation Storage IODs.
  https://www.dicomstandard.org/ (accessed 25 October 2026).
- U.S. Food and Drug Administration, 21 CFR 892.2090, final order 90 FR 24969, 13 June 2025.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892.2090
  (accessed 25 October 2026).
- Zech, J. R. et al., "Variable generalization performance of a deep learning model to detect
  pneumonia in chest radiographs," PLOS Medicine 15(11): e1002683 (2018).
  https://doi.org/10.1371/journal.pmed.1002683
- MIDRC, Data Ingestion Pipeline. https://www.midrc.org/data-ingestion-pipeline (accessed
  25 October 2026).
- The Cancer Imaging Archive, Browse Data. https://www.cancerimagingarchive.net/browse-data/
  (accessed 25 October 2026).

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_preprocessing.md`](03_preprocessing.md)
- Next topic: [`05_architectures.md`](05_architectures.md)
- Related topic: [`06_leakage-and-shortcuts.md`](06_leakage-and-shortcuts.md)
- Root survey: [`../../10_medical_imaging.md`](../../course/10_medical_imaging.md)
- Labs: [`../../labs/`](../../labs/)
- MIDRC: https://www.midrc.org/
