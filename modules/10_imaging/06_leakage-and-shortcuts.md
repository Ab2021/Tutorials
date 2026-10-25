# Leakage and Shortcuts

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 02, 03, and 04
> Version 2026.3

## 1. Why this topic exists

A model that appears to diagnose pneumonia may be reading the hospital's name from the image. A
model that appears to detect pneumothorax may be detecting the chest tube placed to treat it. A
model that appears to predict melanoma may be reading the surgical ink marks drawn before
excision. In each case the model is accurate on the data it was trained on and useless, or
dangerous, on the data it will see.

Leakage is the presence of information about the outcome that is not available at the moment of
prediction. A shortcut is a feature that predicts the outcome in the training distribution but is
not the clinical signal the model is supposed to learn. The two are related but not identical:
leakage is a data-flow error, a shortcut is a learned association. A shortcut can exist without
leakage (a chest tube is visible at prediction time, but it is not the disease), and leakage can
exist without a shortcut (a post-outcome lab value is in the table but the model may not use it).

The industry problem is that both inflate validation performance without raising an error. The
inflation is often larger than the difference between any two legitimate models, so the team
selects the wrong architecture, the wrong preprocessing, and the wrong operating point. The
product then fails at the first external site, and the failure is misdiagnosed as a
generalization problem rather than a measurement problem.

This topic teaches the audit. It covers the information-flow map, the split design, the
metadata-only and pixel-only tests, the site classifier, the prevalence check, the ablation, and
the reporting. It uses the published evidence on shortcut learning in imaging, and it separates
documented findings from inference.

### 1.1 The decisions this topic unblocks

| Decision | What leakage knowledge provides |
|---|---|
| Feature inventory | Which inputs exist at prediction time |
| Split design | Patient, site, and temporal separation |
| Validation claim | Whether performance is honest or inflated |
| Model selection | Whether a gain is signal or shortcut |
| Preprocessing | Whether a marker or annotation must be removed |
| Monitoring | Which distribution shifts predict failure |
| Regulatory evidence | Whether subgroup and site analyses are credible |

### 1.2 The cost of a shortcut

A shortcut does not merely inflate a number. It changes the model's failure mode. A model that
learns a laterality marker fails when the marker is cropped; a model that learns a chest tube
fails when the tube is removed; a model that learns a scanner kernel fails when the scanner is
upgraded. The failure is often sudden and complete, because the feature that carried the
prediction is gone.

The cost is also commercial. A validation package built on a shortcut does not survive an external
site, a regulatory review, or a customer's own validation. The earlier the shortcut is found, the
cheaper the fix.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** the information available at prediction time and identify features that violate it.
2. **Classify** a suspected shortcut into aggregation, timing, or association leakage.
3. **Design** patient-, site-, and temporally-held-out splits and prove that they are disjoint.
4. **Run** a metadata-only model, a site classifier, and a prevalence-only baseline to test for
   non-image signal.
5. **Run** a pixel ablation (crop, mask, or inpaint) to test whether a marker drives performance.
6. **Quantify** the performance change from a shortcut and report it with the unit and reference
   standard.
7. **Distinguish** a legitimate risk factor from a shortcut, and choose the right remedy.
8. **Write** a shortcut audit that a reviewer can reproduce.

## 3. The core mental model

Prediction happens at a point in time. Everything known at that point is legitimate input;
everything learned after it is leakage. The model's input is usually an image, but the pipeline
may also pass metadata, report text, or a prior study. Each channel is a separate opportunity for
leakage.

```text
                    Prediction time
                          │
   ┌──────────────────────┼──────────────────────┐
   │                      │                      │
Image pixels         Metadata               Context
   │                      │                      │
   ├─ anatomy             ├─ scanner             ├─ prior study
   ├─ marker              ├─ protocol            ├─ report text
   ├─ annotation          ├─ site                ├─ clinical history
   └─ compression         └─ description         └─ follow-up
   │                      │                      │
   └──────────────┬───────┴──────────────────────┘
                  ▼
             Model input
                  │
                  ▼
             Prediction
```

### 3.1 Three families of leakage

The taxonomy used here follows the aggregation, timing, and association families described in the
leakage literature (see Module 06, Topic 13); it is useful because the detection test differs by
family.

| Family | Mechanism | Imaging example | Detection |
|---|---|---|---|
| Aggregation | The unit of analysis is mixed across splits | Same patient's studies in train and test | Identifier join |
| Timing | A feature is recorded after the outcome | Follow-up CT used to label the current scan | Timestamp audit |
| Association | A feature is a proxy for the outcome | Chest tube for pneumothorax | Ablation and clinical review |

A fourth family, **site or scanner signature**, is a special case of association: the acquisition
distribution encodes the site, and the site encodes the case mix. It deserves its own test because
it is so common in imaging.

### 3.2 The shortcut triangle

A shortcut requires three things: a feature that is easy to learn, a correlation with the label in
the training distribution, and a distribution shift at deployment that breaks the correlation.

```text
Easy feature  ── correlated with label ──►  learned
      │                                        │
      │                                        ▼
      └──── deployment shift ────────────►  failure
```

If any one of the three is absent, the shortcut does not cause a deployment failure. A marker that
is always present at deployment is not a shortcut in the operational sense, even if it is not the
disease. A feature that is hard to learn may be ignored. A correlation that persists at deployment
may be a legitimate risk factor.

This is why the audit must test the deployment distribution, not only the training set.

### 3.3 Signal, confounder, and shortcut

| Concept | Definition | Example | Remedy |
|---|---|---|---|
| Signal | The clinical feature the model should learn | Nodule margin, consolidation | None |
| Confounder | A feature associated with both input and outcome | Scanner used for sicker patients | Stratify or adjust |
| Shortcut | A feature that predicts the label without the disease mechanism | Laterality marker, chest tube | Remove or ablate |
| Leak | Information unavailable at prediction time | Follow-up report | Exclude from features |

A confounder can be a legitimate predictor if it is available at prediction time and the claim is
about prediction, not mechanism. A shortcut is a confounder that the clinical claim does not
support. The distinction is about the intended use, not only the statistics.

### 3.4 Why image models are especially vulnerable

Images are high-dimensional, so a model has many features to choose from. The clinical signal is
often subtle, while the shortcut is often high-contrast: a bright "L" marker, a dense chest tube,
a scanner's grid pattern. Gradient descent finds the easy feature first. The model is not lazy by
design; it is optimizing the training objective, and the easy feature reduces the loss faster.

The second reason is that image datasets are collected by site. A hospital that serves sicker
patients may use a different scanner, a different protocol, and a different image size. The site
signature and the label become correlated without any clinical relationship.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Leakage | Information about the outcome unavailable at prediction | Confusing with overfitting | Audit finding |
| Shortcut | Feature that predicts the label without the disease mechanism | Calling every confounder a shortcut | Audit finding |
| Confounder | Variable associated with input and outcome | Treating as noise | Statistical |
| Site signature | Acquisition or metadata pattern that identifies a site | Ignoring as nuisance | Site classifier |
| Label leak | Label contains information not in the image | Confusing with text leak | Audit finding |
| Split contamination | Same unit in train and test | Splitting by image | Audit finding |
| Temporal leakage | Using future information | Using follow-up to label current | Audit finding |
| Prevalence confounding | Case mix differs by site | Reporting pooled PPV | Audit finding |
| Laterality marker | "L" or "R" annotation in pixels | Treating as anatomy | Pixel feature |
| Burned-in text | Text rendered into pixel data | Assuming de-id removed it | Pixel feature |
| Private tag | Vendor attribute that may contain text | Ignoring group 0x0019 | Metadata |
| Scanner kernel | Reconstruction filter that changes texture | Treating as fixed | Metadata |
| Chest tube | Device visible on radiograph | Treating as anatomy | Pixel feature |
| Surgical ink | Skin marking visible on dermoscopy | Treating as lesion | Pixel feature |
| Clever Hans | Model that uses an unintended cue | Anthropomorphizing | Concept |
| Saliency map | Attribution over input pixels | Treating as explanation | Visualization |
| Ablation | Remove a feature and retrain or re-test | Changing several factors | Experiment |
| Occlusion test | Mask a region and measure output change | Confusing with retraining | Test |
| Counterfactual | Change one feature and measure output | Confusing with ablation | Test |
| Site-held-out | Test on an unseen site | Testing on a held-out scanner only | Split design |
| Temporal holdout | Test on later time period | Random split by date | Split design |
| Patient-level split | All studies of a patient on one side | Splitting by study | Split design |
| Metadata-only model | Model using only non-pixel fields | Ignoring as trivial | Baseline |
| Prevalence-only model | Model using only base rate | Ignoring | Baseline |
| Shortcut audit | Structured test for non-clinical signal | Ad hoc saliency review | Process |
| Distribution shift | Change in input distribution | Confusing with label shift | Monitoring |
| Domain generalization | Performance on unseen domains | Claiming from one site | Evaluation |
| Spurious correlation | Correlation without causal mechanism | Treating as signal | Statistics |
| Deconfounding | Adjusting for a confounder | Removing a legitimate feature | Method |

### 4.1 Shortcut catalogue in imaging

| Shortcut | Modality | Why it correlates | Test |
|---|---|---|---|
| Laterality marker | Radiography | Marker placement differs by side and by indication | Crop marker, re-evaluate |
| Chest tube | Radiography | Tube placed for pneumothorax | Mask tube, re-evaluate |
| ECG lead | Radiography | Lead placed in monitored patients | Mask lead, re-evaluate |
| Surgical ink | Dermoscopy | Ink marks near lesions selected for excision | Crop ink, re-evaluate |
| Ruler or caliper | Dermoscopy | Ruler used for larger lesions | Crop ruler |
| Scanner kernel | CT | Site uses one kernel for one protocol | Site-held-out, kernel stratification |
| Image size | Radiography | Site-specific detector size | Resize and re-evaluate |
| Text annotation | Any | Report text or label burned in | OCR and crop |
| Prior study | Any | Comparison language in report | Temporal split |
| Prevalence | Any | Case-control sampling | Reweight to population |
| Contrast phase | CT/MRI | Phase correlated with indication | Stratify by phase |
| Patient position | Radiography | Portable patients are sicker | Metadata stratification |
| Image compression | Any | Site-specific codec | Record and stratify |

### 4.2 The three baselines every imaging model needs

| Baseline | Input | What it tests | Expected result |
|---|---|---|---|
| Prevalence-only | Constant score | Whether the metric is driven by base rate | AUC 0.5 |
| Metadata-only | Scanner, protocol, site, demographics | Whether non-image fields predict | AUC near 0.5 |
| Pixel-shortcut-only | Marker region or border | Whether a marker predicts | AUC near 0.5 |

If any baseline has a high AUC, the model's performance is not evidence of image signal. The
baseline is cheap and should run before the deep model.

## 5. The mechanics, step by step

This procedure audits a dataset and a trained model for leakage and shortcuts. It assumes the
data passed ingestion and label checks. Each step states input, output, rationale, branches, and
edge cases.

### Step 1. Draw the prediction-time information map

**Input:** The intended use and the workflow.

**Output:** A list of every input available at prediction time, with its source and timestamp.

**Rationale:** The map defines what is legitimate. A feature recorded after the prediction time is
leakage; a feature recorded before but unavailable in deployment is a shortcut. The map must be
specific: "the image" is not enough; the map must say which series, which metadata, and which
context.

**Branch points:**
- Image only: the strictest and most deployable.
- Image plus metadata: legitimate if the metadata is available at inference.
- Image plus report: usually leakage, because the report is the label source.
- Image plus prior: legitimate only if the prior is available and the claim includes it.

**Edge case:** A model may be deployed in a workflow where the report is already open. If the
radiologist can see the report, a model that uses report text is not adding independent
information, and the claim must say so.

### Step 2. Inventory the features by channel

**Input:** The data manifest and the model input pipeline.

**Output:** A table of channels: pixels, DICOM metadata, report text, prior studies, demographics,
and derived features.

**Rationale:** Leakage hides in channels that the team forgot. A model that reads a private tag, a
file name, or a directory path has a channel that is not in the architecture diagram. The
inventory must include the file path and the dataset version.

**Branch points:**
- Pixel channel: includes burned-in text and markers.
- Metadata channel: includes scanner, protocol, site, and descriptions.
- Text channel: includes report and indication.
- Temporal channel: includes prior and follow-up.

**Edge case:** A file name such as `siteA_positive_001.dcm` is a label leak. The pipeline must not
use file names as features, and the dataset must be renamed before release.

### Step 3. Design patient-, site-, and temporal splits

**Input:** The unit of analysis and the deployment population.

**Output:** Train, validation, and test splits that are disjoint by patient, and optionally by
site and time.

**Rationale:** The split is the first defence against aggregation leakage. A random split by image
or study puts the same patient on both sides. A site-held-out split tests generalization. A
temporal split tests drift.

**Branch points:**
- Random patient-level: standard internal validation.
- Site-held-out: external validation.
- Temporal: prospective-like validation.
- Both site and temporal: strongest, smallest test set.

**Edge case:** A patient may have studies at two sites. The split must be by patient, not by site
alone, or the same patient can cross.

### Step 4. Prove the splits are disjoint

**Input:** The split assignment and the patient, study, and series identifiers.

**Output:** A report of the intersection between splits at each level, and a hash of the split.

**Rationale:** A split is a claim, and the claim must be tested. A patient identifier may be
missing or remapped inconsistently, so the test must use the strongest available key.

**Branch points:**
- Zero intersection at patient level: pass.
- Intersection at study but not patient: fail; same patient.
- Intersection at series but not study: fail; same acquisition.

**Edge case:** De-identified datasets may have different patient identifiers for the same person
across sites. The split cannot detect this without a linkage key; record the limitation.

### Step 5. Run a prevalence-only baseline

**Input:** The label and the test set.

**Output:** The AUC and the PPV of a constant predictor at the test prevalence.

**Rationale:** A constant predictor has AUC 0.5 by construction, but its PPV can be high if the
prevalence is high. A model that appears to have a good PPV may be no better than the base rate.
The baseline also exposes case-control enrichment.

**Branch points:**
- Test prevalence equals deployment: the baseline is directly comparable.
- Test prevalence is enriched: reweight before comparing.
- Prevalence differs by site: report per site.

**Edge case:** A model evaluated with a case-control sample can have a high PPV that is entirely
due to the sampling. The baseline makes the distortion visible.

### Step 6. Run a metadata-only model

**Input:** The metadata channels and the label.

**Output:** AUC and confidence interval for a simple model using only metadata.

**Rationale:** If scanner, protocol, site, or demographics predict the label, the metadata is a
shortcut or a confounder. The model need not be complex; a logistic regression or a gradient
boosted tree is enough.

**Branch points:**
- AUC near 0.5: no metadata signal.
- AUC 0.6-0.7: possible confounder; stratify.
- AUC > 0.7: strong site or protocol signal; investigate.

**Edge case:** A metadata field may be legitimate, such as age for a disease that increases with
age. The question is whether the model's image performance adds anything beyond the metadata. A
combined model that does not beat the metadata-only model is not an image model.

### Step 7. Train a site classifier

**Input:** Image patches or simple image statistics, and the site label.

**Output:** Site classification AUC and the features that drive it.

**Rationale:** If a site can be identified from pixels, the pixels carry a site signature. The
signature can be a marker, a kernel, a compression pattern, or an image size. A high site AUC is a
warning, not a diagnosis; the next step is to find the feature.

**Branch points:**
- Site AUC near 0.5: no strong signature.
- Site AUC 0.7-0.9: signature present; test ablation.
- Site AUC > 0.9: strong signature; likely marker or acquisition difference.

**Edge case:** A site classifier can learn the anatomy distribution rather than the acquisition.
The test is whether the classifier still works on cropped or normalized images.

### Step 8. Run a pixel ablation

**Input:** The trained model and a test set with the suspected feature.

**Output:** Performance with and without the feature, on the same cases.

**Rationale:** Ablation isolates the contribution of the suspected feature. Cropping a marker,
masking a chest tube, or inpainting text changes one thing. If performance falls, the feature
matters; if it does not, the suspicion is wrong.

**Branch points:**
- Crop the marker: tests laterality and text.
- Mask the device: tests chest tube, lead, or catheter.
- Inpaint the annotation: tests burned-in measurement.
- Replace the background: tests border and compression.

**Edge case:** Cropping can remove anatomy as well as the marker. Use a mask that removes only the
marker, and verify the anatomy is preserved.

### Step 9. Run a counterfactual or occlusion test

**Input:** The trained model and a set of cases.

**Output:** The change in output when a region is occluded or a feature is changed.

**Rationale:** A counterfactual asks what the model would predict if the feature were different.
It is weaker than retraining but cheaper, and it can localize the dependency. Occluding the
marker and watching the score move is evidence; a saliency map alone is not.

**Branch points:**
- Occlude the suspected region: score change.
- Occlude a control region: score stability.
- Swap the marker between classes: score reversal.

**Edge case:** Occlusion can create an out-of-distribution image that the model has never seen.
The score change may reflect the artifact, not the feature. Pair occlusion with retraining or
ablation.

### Step 10. Retrain without the shortcut

**Input:** The dataset with the shortcut removed or balanced.

**Output:** A new model and a comparison on the same test set.

**Rationale:** Retraining is the strongest test. If the shortcut is removed and performance holds,
the model was using signal; if performance falls, the shortcut was carrying it. The comparison
must use the same split and the same metric.

**Branch points:**
- Performance holds: the shortcut was not load-bearing.
- Performance falls but remains acceptable: the model had partial signal.
- Performance collapses: the shortcut was the model.

**Edge case:** Removing the shortcut may also remove correlated signal. For example, cropping the
chest tube removes the device but also the pleural line. The interpretation must account for the
confound.

### Step 11. Quantify the impact

**Input:** The performance before and after.

**Output:** A table with the metric, the unit, the reference standard, the confidence interval, and
the change.

**Rationale:** A shortcut audit without a number is an opinion. The number must be at the clinical
unit, not only at the patch level, and it must include the operating point.

**Branch points:**
- Change within the confidence interval: inconclusive.
- Change larger than the interval: material.
- Change reverses the ranking of models: the comparison was invalid.

**Edge case:** A small average change can hide a large subgroup change. Report by site, scanner,
and prevalence.

### Step 12. Report and remediate

**Input:** The audit results.

**Output:** A written report with the suspected feature, the test, the result, the remedy, and the
residual risk.

**Rationale:** The report is the evidence that the team looked. It supports the regulatory
submission, the customer's validation, and the internal decision. The remedy may be removal,
balancing, stratification, or a change in intended use.

**Branch points:**
- Remove the feature: crop, mask, or de-identify.
- Balance the distribution: site-balanced sampling.
- Stratify: report per site and per scanner.
- Change the claim: narrow the intended use.

**Edge case:** A remedy that removes the shortcut may reduce performance below the clinical
threshold. The honest response is to report the lower performance, not to restore the shortcut.

### Step 13. Re-run the audit after any change

**Input:** The new data, model, or pipeline.

**Output:** An updated audit with the version.

**Rationale:** A new site, scanner, or protocol can introduce a new shortcut. The audit is a
lifecycle activity, not a one-time gate.

**Branch points:**
- New site: site-held-out test and site classifier.
- New scanner: kernel and noise stratification.
- New label source: metadata-only and label-leak tests.

**Edge case:** A model that passed the audit in 2024 may fail in 2026 because the data changed,
not because the model changed. The audit must record the data version.

### Step 14. Monitor the shortcut at deployment

**Input:** Production inputs and outputs.

**Output:** Alerts on distribution shift and performance proxy.

**Rationale:** Some shortcuts cannot be removed, only monitored. A scanner signature that is
correlated with disease at one site may be a legitimate predictor at that site and a failure at
another. Monitoring detects the shift before the clinical outcome does.

**Branch points:**
- Input drift: image statistics, metadata distribution.
- Output drift: score distribution, alert rate.
- Outcome drift: delayed labels, audit sample.

**Edge case:** A drop in alert rate can be a model failure or a change in prevalence. Monitor both
the input and the output.

### 5.1 Decision table: suspected shortcut and the test

| Suspicion | First test | Confirmation | Remedy |
|---|---|---|---|
| Laterality marker | Crop marker | Retrain and compare | Remove marker |
| Burned-in text | OCR and crop | Retrain and compare | Inpaint or exclude |
| Chest tube or device | Mask device | Retrain and compare | Mask or stratify |
| Scanner kernel | Site classifier plus kernel stratification | Site-held-out test | Multi-kernel training |
| Image size or compression | Resize and recompress | Site-held-out test | Standardize preprocessing |
| Report text leak | Text-only model | Image-only retrain | Remove text channel |
| Prior study leak | Temporal split | Compare with and without prior | Restrict claim |
| Prevalence confounding | Prevalence-only baseline | Reweight to deployment | Report per prevalence |
| Patient overlap | Identifier join | Re-split by patient | New split |
| Label leak | Metadata-only model | Label audit | Redefine label |

### 5.2 Edge cases that require a written rule

**Mixed slice thickness.** A shortcut may be the thickness itself: thick slices are used for
portable or emergency scans, which correlate with severity. Stratify by thickness.

**Missing series.** A missing series can be a site signature: one site always includes a sequence
and another does not. Do not impute; record the missingness.

**Inconsistent spacing.** A model may learn spacing from the array shape. Resample to a common
grid and test the shape-only baseline.

**Rescale absent.** A missing rescale can produce a distinct intensity distribution that identifies
the site. Assert the unit before training.

**Burned-in annotation.** The annotation may be the label. OCR the pixels, crop, and re-evaluate.

**Laterality mismatch.** A mismatch may be a marker error or a true situs inversus. Resolve before
using laterality as a feature or a label.

**Multi-frame objects.** A multi-frame object may carry a per-frame timestamp or dose that leaks
the protocol. Remove private and acquisition fields that are not available at inference.

**Private tags carrying labels.** Remove all private tags by default; if retained, test for label
association.

### 5.3 Auditing a model that is already deployed

The audit does not end at validation. A deployed model sees new sites, new scanners, and new
protocols. The following quarterly audit uses production data and a small labeled sample.

1. Sample 200 studies stratified by site, scanner, and month.
2. Compute the input statistics that the site classifier used: image size, mean, noise, and
   compression.
3. Run the metadata-only model on the sample. If its AUC rises, the site mix has changed.
4. Compare the alert rate to the validation alert rate. A rise can be prevalence, a shift, or a
   model failure.
5. Label 50 alerts and 50 non-alerts. Estimate PPV with a confidence interval.
6. If PPV falls below the clinical threshold, pause the alert and re-audit.

The audit is not a substitute for prospective monitoring, but it catches the common case: the
model is still running, the inputs have changed, and no one has looked.


### 5.4 How large an audit sample?

A site classifier needs enough images per site to estimate AUC with a useful interval. For an AUC
near 0.9 and a half-width of 0.03, a rough rule of thumb is about 100 images per class per site.
A metadata-only model needs enough events per variable; with 10 predictors, at least 100 positive
cases is a common minimum.

A crop ablation needs paired cases: the same case with and without the marker. The paired design
removes between-case variance, so 100-200 cases are often enough to detect a 0.05 AUC change. An
unpaired design needs several times more.

The sample size should be stated before the audit. An audit that stops when the result is
convenient is not an audit. A pre-registered analysis plan, even a short one, is the difference
between an audit and a search for a result, and it is the first thing a reviewer will ask for.


## 6. Worked example

### 6.1 The case

A pneumonia classifier is trained on 12,000 chest radiographs from two sites. Site A contributes
8,000 images with a pneumonia prevalence of 8% (640 positive). Site B contributes 4,000 images
with a prevalence of 32% (1,280 positive). The pooled prevalence is 16% (1,920/12,000).

The model is a 2D convolutional network at 224 × 224, trained with binary cross-entropy. The
reported pooled test AUC is 0.91. The team plans a deployment at a third site with an expected
prevalence of 12%.

### 6.2 The site signature

Site A images are 1024 × 1024, compressed with JPEG at quality 90, and have a mean pixel value of
0.41 after normalization. Site B images are 2048 × 2048, compressed at quality 75, and have a mean
of 0.55. A logistic regression on two features, image size and mean pixel value, separates the
sites with AUC 0.93 (unit: image; reference: site label).

A site classifier trained on 64 × 64 crops reaches AUC 0.88. The crops do not contain the
laterality marker, so the signature is in the texture and compression, not only the marker.

### 6.3 Pooled versus site-held-out performance

The pooled test AUC is 0.91. When site B is held out and the model is trained on site A only, the
AUC on site B is 0.78. When site A is held out, the AUC on site A is 0.81. The average
site-held-out AUC is 0.795, compared with the pooled 0.91.

The difference, 0.115 AUC, is larger than the difference between most architecture variants. The
pooled number is not a deployment estimate; it is an estimate on a mixture that includes the
site signature.

### 6.4 The metadata-only model

A gradient-boosted model using scanner model, image size, compression quality, and patient age
reaches AUC 0.84 on the pooled test set. The image model's 0.91 is therefore only 0.07 above a
model that never sees the anatomy. That does not prove the image model is a shortcut, but it
establishes that most of the pooled performance is available without pixels.

### 6.5 PPV at deployment prevalence

The model's sensitivity is 0.90 and specificity 0.85 at the chosen threshold. At the pooled
prevalence of 16%:

```text
PPV = (0.90 × 0.16) / (0.90 × 0.16 + 0.15 × 0.84)
    = 0.144 / (0.144 + 0.126)
    = 0.533
```

At the deployment prevalence of 12%:

```text
PPV = (0.90 × 0.12) / (0.90 × 0.12 + 0.15 × 0.88)
    = 0.108 / (0.108 + 0.132)
    = 0.450
```

At site B's prevalence of 32%:

```text
PPV = (0.90 × 0.32) / (0.90 × 0.32 + 0.15 × 0.68)
    = 0.288 / (0.288 + 0.102)
    = 0.738
```

The same model has PPV 0.45, 0.53, or 0.74 depending on prevalence. A deployment plan that uses
the pooled PPV will overestimate the alert burden at the third site.

### 6.6 Diagnostic 1: label leak in the report

The report text is available in a private tag. The word "portable" appears in 62% of site B
positive reports and 18% of site B negative reports. A text-only classifier using the report
reaches AUC 0.79. The image model was trained with a pipeline that reads the private tag as a
categorical feature, so the model has a direct path to the label. The diagnostic is the
metadata-only AUC plus the private-tag audit; the fix is to strip the tag and retrain.

### 6.7 Diagnostic 2: split contamination

The dataset has 12,000 images from 9,400 patients. A random split by image puts 1,850 images from
patients who also appear in training into the test set. The test AUC with contamination is 0.91;
after a patient-level split, it is 0.87. The 0.04 difference is smaller than the site effect but
still material.

The diagnostic is a join on the patient identifier. The split hash and the intersection count are
recorded in the manifest.

### 6.8 Varying one assumption

Assume the sites are balanced to the same prevalence of 16% by downsampling site B positives. The
site-only classifier's AUC falls from 0.93 to 0.51 because the label no longer correlates with the
site. The pooled model's AUC falls from 0.91 to 0.83, and the metadata-only AUC falls from 0.84 to
0.55.

The conclusion changes from "the model has a strong site-driven signal" to "the model has a
moderate image signal once the prevalence confound is removed". The remedy is not a better
architecture; it is a balanced evaluation and a claim that states the prevalence.

### 6.9 What the audit reports

| Item | Value in this case | Interpretation |
|---|---|---|
| Pooled AUC | 0.91 | Not a deployment estimate |
| Site-held-out AUC | 0.78-0.81 | Honest cross-site estimate |
| Metadata-only AUC | 0.84 | Most pooled performance is non-image |
| Site classifier AUC | 0.93 | Strong site signature |
| Crop ablation change | -0.04 | Marker contributes, but not alone |
| Patient overlap | 1,850 images | Split contamination |
| Deployment PPV at 12% | 0.45 | Alert burden higher than pooled |

The table is the audit. A report that lists only the first row has not audited anything.

### 6.10 A second variation: scanner upgrade

Assume site A upgrades its scanner and changes the reconstruction kernel. The image size stays
1024 × 1024, but the noise standard deviation rises from 0.04 to 0.07 and the mean falls from 0.41
to 0.37. The site classifier still identifies the site, but the model's score distribution shifts:
the mean score falls by 0.08 and the alert rate falls by 22%.

The sensitivity on a labeled sample falls from 0.90 to 0.83, while specificity rises from 0.85 to
0.89. The model has not become worse at the disease; it has become worse calibrated for the new
noise level. The remedy is recalibration on a small post-upgrade sample, not retraining on the
whole archive.

This variation shows that a shortcut audit must be repeated after an acquisition change. The
metadata version is the trigger.

### 6.11 Why balancing is not always the remedy

Balancing the prevalence removes the correlation between site and label, but it also removes
information that may be legitimate at a specific site. If site B serves a sicker population, the
site is a real risk factor for that population. A model deployed only at site B may legitimately
use the site, because the site is constant and the claim is site-specific.

The decision depends on the intended use. A product that claims cross-site generalization must be
trained and evaluated without the site cue. A product that claims site-specific performance may
keep it, provided the claim states the site and the population. The audit makes the choice
explicit; it does not make it automatically.

### 6.12 The counterfactual that would settle it

The audit's strongest test is a counterfactual on the same cases: take 100 site B studies, remove
the site signature by resizing to 1024 × 1024 and recompressing at quality 90, and re-run the
model. If the AUC falls from 0.91 to 0.83, the signature was carrying 0.08 AUC. If it stays at
0.90, the signature was not load-bearing and the pooled number is closer to honest than the
site-held-out number suggests.

The counterfactual is not a substitute for a site-held-out test, because it changes the image and
may remove anatomy. It is a cheap diagnostic that localizes the dependency before a retraining
budget is committed.
### 6.13 The decision the audit supports

The audit does not decide whether to ship. It gives the team three numbers: the honest cross-site
performance, the performance attributable to non-image signal, and the performance at deployment
prevalence. The product decision is then a comparison of the honest number to the clinical
threshold, not a comparison of the pooled number to a competitor's pooled number.

The honest number is almost always lower. That is the point: it is the number the customer will
reproduce, and it is the number that should appear in the labeling and the sales conversation.


## 7. Code, query, or calculation

The following code simulates a site-confounded dataset and computes the AUC of a score, the AUC of
a site-only score, PPV at three prevalences, and the effect of balancing. It uses only the
standard library. The AUC is the Mann-Whitney statistic with tie handling.

```python
import math

def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)

def make_site(n_pos, n_neg, base, site_flag, seed_offset=0):
    rows = []
    for i in range(n_pos):
        noise = 0.80 * math.sin((i + seed_offset) * 12.9898)
        rows.append((base + 0.55 + noise, 1, site_flag))
    for i in range(n_neg):
        noise = 0.80 * math.sin((i + n_pos + seed_offset) * 78.233)
        rows.append((base + noise, 0, site_flag))
    return rows

site_a = make_site(64, 736, 0.00, 0)
site_b = make_site(320, 680, 0.35, 1)
data = site_a + site_b
scores = [r[0] for r in data]
labels = [r[1] for r in data]
site_scores = [r[2] for r in data]

print("pooled n           =", len(data))
print("pooled prevalence  = {:.3f}".format(sum(labels) / len(labels)))
print("model AUC          = {:.3f}".format(auc(scores, labels)))
print("site-only AUC      = {:.3f}".format(auc(site_scores, labels)))

def ppv(prev, sens=0.90, spec=0.85):
    return (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))

for prev in (0.12, 0.16, 0.32):
    print("prevalence {:.0%} -> PPV {:.3f}".format(prev, ppv(prev)))

b_pos = [r for r in site_b if r[1] == 1][:64]
b_neg = [r for r in site_b if r[1] == 0][:680]
balanced = site_a + b_pos + b_neg
bs = [r[0] for r in balanced]
bl = [r[1] for r in balanced]
bt = [r[2] for r in balanced]
print("balanced n         =", len(balanced))
print("balanced prev      = {:.3f}".format(sum(bl) / len(bl)))
print("balanced model AUC = {:.3f}".format(auc(bs, bl)))
print("balanced site AUC  = {:.3f}".format(auc(bt, bl)))

```

Expected output:

```text
pooled n           = 1800
pooled prevalence  = 0.213
model AUC          = 0.781
site-only AUC      = 0.677
prevalence 12% -> PPV 0.450
prevalence 16% -> PPV 0.533
prevalence 32% -> PPV 0.738
balanced n         = 1544
balanced prev      = 0.083
balanced model AUC = 0.750
balanced site AUC  = 0.510
```

The simulated numbers depend on the seed and the score parameters; they illustrate the method, not
the published values. The site-only AUC of 0.682 is lower than the 0.93 in the case because the
simulation uses a binary site indicator with overlap, not a rich image signature. The direction is
the point: balancing removes the site signal and lowers the pooled AUC.

The code assumes: independent images, a binary site, and a score that is a mixture of signal and
site. It does not model patient clustering or temporal drift.

## 8. Data, standards, and regulatory context

### 8.1 DICOM and de-identification

DICOM PS3.15 Annex E defines the attributes that must be removed or replaced. The Basic Profile
removes patient identity, dates, and private attributes, but it does not clean pixel data unless
the Clean Pixel Data option is used. Burned-in text and laterality markers therefore survive a
metadata-only de-identification.

The standard is necessary and not sufficient. A dataset can be PS3.15-conformant and still contain
a site signature in the pixel texture, the image size, or the compression. The shortcut audit is a
separate process.

### 8.2 FDA evidence expectations

The special controls at 21 CFR 892.2090 require standalone performance testing, subgroup analyses,
and labeling that states compatible hardware and protocols. A model whose performance depends on a
site signature will fail the subgroup analysis if the subgroups are defined by site or scanner.
The regulation does not use the word "shortcut", but the evidence requirement is designed to
expose one.

The FDA's 2025 request for public comment on measuring and evaluating AI-enabled device
performance in the real world explicitly names data drift, concept drift, and post-deployment
monitoring. A shortcut is a form of concept drift when the correlation between the feature and
the label changes.

### 8.3 Reporting standards

CLAIM (2020, updated 2024) requires a description of data sources, inclusion and exclusion
criteria, data partitions, reference standard, and error analysis. A shortcut audit belongs in the
error analysis and in the discussion of limitations. A paper that reports only pooled performance
and a saliency figure does not meet the standard.

### 8.4 Privacy and governance

A shortcut audit often uses metadata that is sensitive: site, scanner, and demographics. The audit
must follow the same governance as the model. The results should be reported at the level of
aggregates, not individual patients, and the site labels should be pseudonymized where the
contract requires it.

### 8.5 What the standards do not cover

No standard defines a shortcut or requires a specific audit. The vocabulary comes from the
research literature (Zech, Badgeley, Winkler) and from regulatory expectation, not from a
normative document. The team must document its own audit protocol and thresholds.

### 8.6 Contracts and data-use agreements

A shortcut audit often needs site labels, scanner identifiers, and prevalence by site. Those
fields may be restricted by data-use agreements even when the images are de-identified. The audit
plan should state which fields are used, at what aggregation, and who can see the results. An
audit that violates the data-use agreement is not a valid audit, regardless of its statistical
quality.

### 8.7 Ethics of reporting a shortcut

A shortcut audit can reveal that a site's data are systematically different, that a device is
used preferentially for one population, or that a report label is unreliable. Those findings can
be sensitive for the site and for the patients. The report should state the finding at the level
needed for the remedy, not expose individual records, and should give the site a chance to
respond before publication.

## 9. Industry practice and case evidence

### 9.1 Zech et al. and the hospital signature

Zech et al. (PLOS Medicine, 2018) trained pneumonia classifiers on 158,323 chest radiographs from
the National Institutes of Health (112,120), Mount Sinai (42,396), and Indiana University (3,807).
Performance fell on external data in three of five comparisons. A model using only the hospital
system reached AUC 0.861 on the joint Mount Sinai-NIH set, because prevalence differed (34.2% at
Mount Sinai vs about 1% at the other sites). Classifiers could identify the source hospital with
greater than 99.9% accuracy, often from laterality markers and scanner artefacts.

The documented facts are the dataset sizes, the prevalence difference, and the site-classification
accuracy. The inference is that a model trained on such data will use the site signature when the
label is correlated with site.

### 9.2 Badgeley et al. and hip fracture confounders

Badgeley et al. (npj Digital Medicine, 2019) trained convolutional networks on 17,587 hip
radiographs from Mount Sinai, with a fracture prevalence of about 3%. All 20 auxiliary variables
were predictable from pixels alone: scanner model AUC 1.00, scanner brand 0.98, and order priority
0.79. Fracture AUC was 0.78 with image only, 0.86 with patient data, and 0.91 with patient and
hospital-process data. Under case-control matching, performance fell to 0.67 with patient matching
and 0.52 with patient and process matching, which is chance.

The lesson is that hospital-process variables can carry the label and that matching can reveal it.
A model that looks better with metadata may be using the metadata as a proxy for the outcome.

### 9.3 Winkler et al. and surgical ink

Winkler et al. (JAMA Dermatology, 2019) evaluated a CE-marked dermoscopy classifier on 130
melanocytic lesions. Without markings, sensitivity was 95.7%, specificity 84.1%, and AUC 0.969.
With gentian violet surgical ink, sensitivity was 100%, specificity fell to 45.8%, and AUC 0.922.
Cropped images without ink reached sensitivity 100%, specificity 97.2%, and AUC 0.993. Saliency
maps confirmed that the ink drove the predictions.

The case is a clean demonstration that a high-contrast, clinically irrelevant feature can dominate
a model and that removing it can improve performance.

### 9.4 Public challenges and hidden shortcuts

The RSNA intracranial haemorrhage challenge (2019) used more than 25,000 head CT exams annotated
by more than 60 radiologists. Challenge datasets are curated, but they can still contain site
signatures from the contributing institutions. A winning model on a challenge leaderboard is not
evidence of cross-site generalization; the test set is drawn from the same distribution as the
training set.

### 9.5 Inference, not fact

The following are inferences: that a specific hospital's laterality marker will correlate with a
specific label; that a specific scanner kernel will be learned; that a specific model's performance
will fall by a specific amount after cropping. Each is plausible and each requires a local audit.
The documented facts are the three published studies and their reported numbers.

### 9.6 The regulatory consequence of a shortcut

A CADe device that marks a finding must report sensitivity and false positives at a stated
operating point. If the model's performance depends on a laterality marker, the false-positive
rate at a site that crops markers will be higher than the submitted rate. The FDA special controls
require labeling that states compatible imaging protocols and warnings about image quality; a
shortcut is a hidden protocol dependency. The submission may pass, and the deployment may still
fail, which is why post-market performance monitoring is part of the 2025 regulatory discussion.

### 9.7 A note on external validation claims

An external validation on a single held-out site is one sample from the space of sites. It is
stronger than a random split, but it does not establish generalization to all sites. A claim that
says "validated externally" should name the site, the scanner, the protocol, and the prevalence.
A claim that says "generalizes across sites" needs several sites and a report of the variance
across them.
### 9.8 What the evidence does not show

The three landmark studies show that shortcuts exist and can dominate performance in specific
datasets. They do not show that every model on every dataset uses a shortcut, or that a specific
architecture is more prone to shortcuts. The prevalence of shortcut learning across imaging
products is unknown, because most products are not audited publicly.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Laterality marker learned | Crop test drops AUC | Marker correlates with side and label | Fails when cropped | Crop and retrain |
| Chest tube learned | Device mask drops AUC | Tube is treatment for the disease | Fails on treated cases | Mask and stratify |
| Scanner kernel learned | Site classifier AUC high | Texture differs by kernel | Fails after upgrade | Multi-kernel training |
| Image size learned | Shape-only baseline high | Detector size differs by site | Fails on new site | Standardize resize |
| Compression learned | Texture differs by quality | Codec artefacts | Site signature | Record and recompress |
| Report text leak | Text-only AUC high | Report contains label | Inflation | Remove text channel |
| Private tag leak | Metadata-only AUC high | Vendor tag has indication | Inflation | Strip private tags |
| Prior study leak | Temporal split drop | Comparison language | Unavailable at inference | Restrict label |
| Patient overlap | Identifier intersection | Random split by image | Inflation | Patient-level split |
| Prevalence confound | Site-only AUC high | Case mix differs | PPV misreported | Reweight or stratify |
| Label leak in filename | Path predicts label | Ad hoc naming | Inflation | Rename files |
| Burned-in annotation | OCR finds text | Pixels not cleaned | Leak | Inpaint or exclude |
| Saliency treated as proof | Heatmap on marker | Attribution not explanation | False reassurance | Ablation |
| Audit after model selection | Shortcut found late | No pre-specified test | Costly rework | Audit before training |

### 10.1 Red flags that stop a validation claim

1. The test set shares patients with training.
2. A metadata-only model has AUC above 0.75.
3. A site classifier has AUC above 0.85.
4. A crop of a marker changes AUC by more than 0.05.
5. The pooled prevalence differs from the deployment prevalence.
6. The report text is an input and the label comes from the report.
7. A private tag survives de-identification and predicts the label.

### 10.2 A shortcut audit checklist

1. The prediction-time map is written and signed.
2. The split is patient-level and the intersection is zero.
3. A prevalence-only baseline is reported.
4. A metadata-only baseline is reported.
5. A site classifier is reported.
6. A crop or mask ablation is reported for each suspected marker.
7. A retrained model without the shortcut is compared.
8. The impact is quantified at the clinical unit.
9. The remedy and residual risk are written.
10. The audit is versioned and scheduled for repeat.

A model that has not passed this checklist has not been shown to use the clinical signal.

### 10.3 What a failed audit looks like in practice

A failed audit is rarely a single dramatic number. It is a pattern: the metadata-only model is
better than expected, the site classifier is strong, the crop ablation moves the AUC, and the
site-held-out result is much lower than the pooled result. Each finding alone can be explained;
together they describe a model that has learned the acquisition rather than the disease.
### 10.4 The audit as a product feature

A completed audit is a sales asset. A vendor that can show a site-held-out result, a metadata-only
baseline, and a crop ablation is answering the questions a hospital's own validation team will
ask. A vendor that can show only a pooled AUC is asking the hospital to repeat the work.

The audit should be summarized in the model card and the labeling: the sites, the scanners, the
prevalences, the shortcut tests, and the residual risk. That summary is more useful to a customer
than a leaderboard rank, because it describes the conditions under which the product was measured. It also gives the hospital a
starting point for its own audit, which shortens procurement and reduces the chance of a
post-deployment surprise. That is a commercial argument for doing the audit early, not only a
scientific one.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Prediction-time inputs are known | Leakage definition | Workflow changes | Map the workflow |
| Patient identifiers are consistent | Split integrity | Remapping across sites | Identifier audit |
| Site labels are available | Site-held-out test | Missing or wrong site | Metadata check |
| The shortcut is visible in the input | Ablation works | Feature interacts with anatomy | Multiple tests |
| The test set is independent | Valid estimate | Contamination | Split proof |
| Prevalence is known | PPV calculation | Case-control sampling | Reweight |
| The model uses the image | Claim validity | Metadata path | Metadata-only baseline |
| The audit sample is representative | Generalization | Selected cases | Random or stratified |
| The remedy removes only the shortcut | Causal attribution | Correlated anatomy | Check anatomy preserved |
| The audit is repeated | Lifecycle validity | New site or scanner | Versioned audit |

### 11.1 Limits of this topic

This topic covers the audit for leakage and shortcuts in imaging. It does not cover causal
inference, formal domain adaptation, or the full statistics of confounding. It gives the data
scientist the tests and the vocabulary to find a shortcut and quantify its impact, not to prove
causality.

### 11.2 What would change the conclusion

If the deployment population had the same prevalence as the pooled test set and the same scanner
mix, the pooled AUC would be a reasonable estimate, and the site-held-out result would be a
conservative lower bound. If the report text were unavailable at inference, the text leak would
disappear, and the image-only AUC would be the honest number. If the laterality marker were
present in every deployment image and the labeling policy were consistent, the marker would be a
stable feature rather than a shortcut, though it would still not be the disease.

### 11.3 The limit of any audit

An audit can show that a model uses a feature. It cannot show that the feature is the only one, or
that the model will fail at a site that has not been seen. The audit reduces uncertainty; it does
not eliminate it. The residual risk is stated in the report and monitored in deployment.

## 12. Exercises

### Exercise 1. Information map (recall, 20 minutes)

**Deliverable:** A diagram of the inputs available at prediction time for a concurrent-read
pneumonia alert, with each input's source and timestamp.

**Acceptance criteria:** At least six inputs; timestamps; at least one input marked unavailable.

**Time:** 20 minutes.

### Exercise 2. Split audit (application, 30 minutes)

**Deliverable:** A query or script that joins train and test on patient ID and reports the overlap
count.

**Acceptance criteria:** Detects planted overlap; reports both patient and study counts; states
the consequence.

**Time:** 30 minutes.

### Exercise 3. Metadata-only baseline (application, 40 minutes)

**Deliverable:** A logistic regression or simple rule using scanner, site, and image size, with
AUC and confidence interval.

**Acceptance criteria:** AUC reported with unit and reference; features listed; interpretation
stated.

**Time:** 40 minutes.

### Exercise 4. Crop ablation (application, 50 minutes)

**Deliverable:** A plan and a table of AUC before and after cropping a laterality marker, with the
same test cases.

**Acceptance criteria:** One feature changed; confidence intervals; anatomy preserved; conclusion
stated.

**Time:** 50 minutes.

### Exercise 5. PPV and prevalence (calculation, 25 minutes)

**Deliverable:** For sensitivity 0.88 and specificity 0.82, compute PPV at 5%, 15%, and 30%
prevalence.

**Acceptance criteria:** Correct values (0.205, 0.463, 0.677); units and formula stated.

**Time:** 25 minutes.

### Exercise 6. Shortcut report (critique, 40 minutes)

**Deliverable:** A 600-word audit report for a model suspected of learning a chest tube, including
the test, the result, and the remedy.

**Acceptance criteria:** Names the mechanism; quantifies the change; distinguishes confounding from
shortcut; states residual risk.

**Time:** 40 minutes.

### Exercise 7. Site-held-out design (design, 45 minutes)

**Deliverable:** A split plan for three sites with different prevalences, including the training,
validation, and test assignments and the expected limitations.

**Acceptance criteria:** Patient-level; site-held-out; prevalence reported; no site in both train
and test.

**Time:** 45 minutes.

### Exercise 8. Monitoring plan (design, 40 minutes)

**Deliverable:** A monitoring plan with input statistics, output statistics, and a trigger for
re-audit.

**Acceptance criteria:** At least four metrics; thresholds; owner; escalation path.

**Time:** 40 minutes.

## 13. Self-check answers

**A1. What is the difference between leakage and a shortcut?**
Leakage is information about the outcome that is unavailable at prediction time. A shortcut is a
feature that predicts the label without the disease mechanism but may be available at prediction
time. A chest tube is a shortcut, not leakage. A follow-up report is leakage.

**A2. Why does a site-only model with AUC 0.86 matter?**
It shows that the site label alone predicts the outcome, because prevalence or case mix differs by
site. A model trained on the pooled data can learn the site signature instead of the disease.

**A3. How do you prove a split is patient-level?**
Join the training and test sets on the patient identifier and report the intersection. A zero
intersection at the patient level is the proof. If identifiers are remapped, the linkage must be
preserved or the limitation recorded.

**A4. What does cropping a laterality marker test?**
It tests whether the model's performance depends on the marker. If AUC falls materially after
cropping, the marker was carrying signal. If it does not, the marker was not load-bearing.

**A5. Compute PPV at 5% prevalence for sensitivity 0.88 and specificity 0.82.**
PPV = (0.88 × 0.05) / (0.88 × 0.05 + 0.18 × 0.95) = 0.044 / (0.044 + 0.171) = 0.205. The
reported value 0.207 differs by rounding.

**A6. Why is saliency not enough to rule out a shortcut?**
A saliency map shows where the gradient is large, not what the model would do without the feature.
A model can have a diffuse saliency map and still depend on a marker. Ablation or retraining is
the test.

**A7. What is prevalence confounding?**
The case mix differs by site, so the site is correlated with the label. A model can learn the site
and appear to predict the disease. Balancing or stratifying removes the correlation.

**A8. When is a confounder legitimate?**
When it is available at prediction time and the intended use supports using it. Age is a legitimate
predictor for many diseases; a scanner model is usually not, unless the claim is site-specific.

**A9. Design rubric for Exercise 5.** A strong answer shows the formula, the three values, and the
direction of the change. A weak answer reports one value without prevalence.

**A10. Design rubric for Exercise 7.** A strong plan is patient-level, holds out a site, reports
prevalence per site, and states that the test set is small. A weak plan splits randomly and pools
the sites.

## 14. Interview questions

**Q1. What is shortcut learning in imaging?**
A strong answer defines it as a feature that predicts the label without the disease mechanism and
gives a concrete example such as a laterality marker or chest tube. A weak answer calls it
overfitting.

**Q2. How do you detect a site signature?**
A strong answer trains a site classifier, checks metadata-only performance, and stratifies by site.
A weak answer looks at a saliency map.

**Q3. Why is patient-level splitting necessary?**
A strong answer explains that multiple studies per patient create near-duplicates and inflate
performance. A weak answer says it is a convention.

**Q4. What is the difference between a confounder and a shortcut?**
A strong answer says a confounder is associated with input and outcome and may be legitimate; a
shortcut is a confounder the clinical claim does not support. A weak answer uses the terms
interchangeably.

**Q5. How do you test whether a marker drives performance?**
A strong answer crops or masks the marker, re-evaluates on the same cases, and compares with
confidence intervals. A weak answer cites a saliency map.

**Q6. What is the first baseline you run?**
A strong answer runs prevalence-only and metadata-only baselines before the deep model. A weak
answer starts with a ResNet.

**Q7. How does prevalence affect PPV?**
A strong answer gives the formula and an example, and notes that sensitivity and specificity can
also change with spectrum. A weak answer says PPV is a property of the test.

**Q8. How do you report a shortcut audit?**
A strong answer reports the suspected feature, the test, the metric with unit and reference, the
change, the remedy, and the residual risk. A weak answer says "we checked and it was fine".

**Q9. What would you do if removing a shortcut dropped performance below the clinical threshold?**
A strong answer reports the lower performance and narrows the claim or improves the data; it does
not restore the shortcut. A weak answer keeps the shortcut because the number is better.

**Q10. How do you monitor for a new shortcut after deployment?**
A strong answer monitors input statistics, site mix, output rates, and a periodic audit sample,
with triggers for re-audit. A weak answer monitors only accuracy, which is unavailable without
labels.

## 15. Key terms

- **Ablation** — remove a feature and re-evaluate.
- **Burned-in text** — text in pixel data.
- **Chest tube** — device that can act as a shortcut.
- **Clever Hans** — model using an unintended cue.
- **Confounder** — variable associated with input and outcome.
- **Counterfactual** — change one feature and measure output.
- **Distribution shift** — change in input distribution.
- **Label leak** — label contains unavailable information.
- **Laterality marker** — "L" or "R" in pixels.
- **Leakage** — outcome information unavailable at prediction.
- **Metadata-only model** — model using non-pixel fields.
- **Occlusion test** — mask a region and measure output.
- **Patient-level split** — all studies of a patient on one side.
- **Prevalence confounding** — case mix differs by site.
- **Prevalence-only model** — constant predictor baseline.
- **Private tag** — vendor attribute.
- **Saliency map** — attribution over input.
- **Scanner kernel** — reconstruction filter.
- **Shortcut** — feature predicting label without mechanism.
- **Shortcut audit** — structured test for non-clinical signal.
- **Site classifier** — model predicting site from pixels.
- **Site signature** — acquisition pattern identifying a site.
- **Site-held-out** — test on an unseen site.
- **Split contamination** — same unit in train and test.
- **Surgical ink** — skin marking that can act as a shortcut.
- **Temporal leakage** — using future information.
- **Temporal holdout** — test on a later period.
- **Spurious correlation** — correlation without mechanism.

## 16. Sources

- Zech, J. R. et al., "Variable generalization performance of a deep learning model to detect
  pneumonia in chest radiographs: A cross-sectional study," PLOS Medicine 15(11): e1002683
  (2018). https://doi.org/10.1371/journal.pmed.1002683
- Badgeley, M. A. et al., "Deep learning predicts hip fracture using confounding patient and
  healthcare variables," npj Digital Medicine 2, 31 (2019).
  https://doi.org/10.1038/s41746-019-0105-1
- Winkler, J. K. et al., "Association Between Surgical Skin Markings in Dermoscopic Images and
  Diagnostic Performance of a Deep Learning Convolutional Neural Network for Melanoma
  Recognition," JAMA Dermatology 155(10), 1135-1141 (2019).
  https://doi.org/10.1001/jamadermatol.2019.1735
- DICOM Standard, PS3 current edition 2026d, PS3.15 Annex E.
  https://dicom.nema.org/medical/Dicom/current/output/chtml/part15/chapter_E.html (accessed
  25 October 2026).
- U.S. Food and Drug Administration, 21 CFR 892.2090, final order 90 FR 24969, 13 June 2025.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892.2090
  (accessed 25 October 2026).
- U.S. Food and Drug Administration, Request for Public Comment: Measuring and Evaluating
  Artificial Intelligence-enabled Medical Device Performance in the Real-World (2025).
  https://www.fda.gov/medical-devices/digital-health-center-excellence/request-public-comment-measuring-and-evaluating-artificial-intelligence-enabled-medical-device
  (accessed 25 October 2026).
- Mongan, J., Moy, L., Kahn, C. E. Jr., "Checklist for Artificial Intelligence in Medical Imaging
  (CLAIM)," Radiology: Artificial Intelligence 2(2): e200029 (2020).
  https://doi.org/10.1148/ryai.2020200029
- Tejani, A. S. et al., "Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024
  Update," Radiology: Artificial Intelligence 6: e240300 (2024).
  https://doi.org/10.1148/ryai.240300
- MIDRC, Data Ingestion Pipeline. https://www.midrc.org/data-ingestion-pipeline (accessed
  25 October 2026).

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_architectures.md`](05_architectures.md)
- Next topic: [`07_evaluation.md`](07_evaluation.md)
- Related topic: [`02_dicom-and-metadata.md`](02_dicom-and-metadata.md)
- Related topic: [`04_labels-and-annotation.md`](04_labels-and-annotation.md)
- Root survey: [`../../10_medical_imaging.md`](../../10_medical_imaging.md)
- Labs: [`../../labs/`](../../labs/)

> **Note:** The audit is a lifecycle control; the numbers are only valid for the audited version.
> Re-run it when the data, the site mix, or the scanner changes.

> **Note:** Pre-register the audit, including the sample size and the decision rule.
