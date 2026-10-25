# Radiology and Imaging Basics

> Module 01 — Clinical Foundations for Data Scientists
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: 02, 05
> Version 2026.3

---

## 1. Why this topic exists

Medical imaging is the largest source of unstructured clinical data by volume and one of the
most expensive. A single computed tomography (CT) study can contain thousands of images, each
with acquisition parameters, spatial metadata, and a reconstruction history. A magnetic
resonance imaging (MRI) study can contain dozens of sequences. A pathology whole-slide image can
be gigapixels. The data are rich, and they are almost never self-explanatory.

The industry problems are specific.

**Problem 1: the image is not the data.** A Digital Imaging and Communications in Medicine
(DICOM) file contains pixel values, but those values are meaningless without the rescale slope and
intercept that convert them to Hounsfield units (HU) for CT, the windowing that maps them to
display values, and the acquisition parameters that determine what was measured. A pipeline that
reads pixel arrays without metadata is reading a display image, not a measurement.

**Problem 2: the label is a report, not a diagnosis.** Most imaging datasets derive labels from
radiology reports using natural language processing (NLP). Reports contain uncertainty,
negation, comparison to prior studies, and recommendations. "No evidence of pneumonia" and
"cannot exclude pneumonia" are different statements, and both differ from "pneumonia." A label
extracted without negation and uncertainty handling will be noisy.

**Problem 3: prevalence and spectrum change performance.** A model trained on a curated dataset
of biopsy-proven disease will not perform the same way in a screening population where most
patients are healthy. The imaging appearance, the acquisition protocol, and the pretest
probability all differ.

**Problem 4: the workflow determines adoption.** A model that reads a study after the radiologist
has already signed the report has no opportunity to change care. A model that interrupts the
reading workflow with low-value alerts will be turned off. Accuracy is necessary and not
sufficient.

This topic covers the modalities, the acquisition and reporting workflow, the DICOM standard,
the reporting lexicons, radiation dose, and the specific failure modes of imaging data science.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compare** radiography, CT, MRI, ultrasound, positron emission tomography (PET), and
   mammography on physics, strengths, radiation, and data characteristics.
2. **Explain** Hounsfield units, windowing, and the DICOM rescale transformation.
3. **Trace** an imaging study from order to report and **identify** where a model could enter the
   workflow.
4. **Interpret** common structured reporting categories (Lung-RADS, BI-RADS) at a level
   sufficient to build a cohort.
5. **Compute** nodule volume, volume doubling time, and effective dose from acquisition
   parameters.
6. **Critique** an imaging dataset for label noise, acquisition heterogeneity, and leakage.
7. **Design** a DICOM-aware preprocessing pipeline that preserves metadata and patient-level
   splitting.

## 3. The core mental model

Imaging is a chain from physics to pixels to report to decision. Each link has a data
representation and a failure mode.

```text
   PATIENT
      |
      v
   ACQUISITION (x-ray tube, magnet, transducer, radiotracer)
      |
      v
   RAW DATA -> RECONSTRUCTION -> IMAGE SERIES
      |
      v
   DICOM OBJECT (pixels + metadata + UIDs)
      |
      v
   PACS / VNA (storage and retrieval)
      |
      v
   INTERPRETATION (radiologist, CAD, AI)
      |
      v
   REPORT (narrative or structured)
      |
      v
   DECISION (treat, biopsy, follow-up, discharge)
```

### 3.1 The two data layers

Every image has two layers:

```text
   METADATA LAYER                    PIXEL LAYER
   patient, study, series, SOP UIDs   rows x columns x frames
   modality, body part, protocol      bit depth, photometric interpretation
   acquisition parameters             pixel representation (signed/unsigned)
   rescale slope and intercept        transfer syntax (compression)
   window center and width            pixel spacing, slice thickness
   dose indices                       orientation, position
```

The metadata layer is often more informative than the pixel layer for cohort construction. Study
date, modality, body part, and protocol identify the exam. The pixel layer is what a model sees,
but only after the metadata has been used to interpret it correctly.

### 3.2 The report is a derived artifact

A radiology report is written by a human after viewing the images, often with comparison to
prior studies, and often using templates. It contains:

- **Findings:** what is visible.
- **Impression:** the synthesis and recommendation.
- **Comparison:** change from prior studies.
- **Uncertainty:** "likely," "cannot exclude," "indeterminate."
- **Recommendations:** follow-up interval, biopsy, further imaging.

A label extracted from the impression is closer to the clinical decision than a label extracted
from the findings, but it is still a text-derived proxy.

### 3.3 The resolution-sensitivity trade-off

Imaging modalities differ in what they can resolve and what they can detect. A chest radiograph
is a two-dimensional projection with overlapping structures; a CT is a three-dimensional
reconstruction with cross-sectional slices; an MRI provides tissue contrast without ionizing
radiation. None is universally superior. The choice depends on the question, the patient, and the
risk.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or standard |
|---|---|---|---|
| Radiography | Two-dimensional projection using x-rays | Confused with fluoroscopy | DICOM modality CR/DX |
| Fluoroscopy | Real-time x-ray imaging | Confused with radiography | DICOM modality RF |
| Computed tomography (CT) | Cross-sectional reconstruction from multiple x-ray projections | Confused with cone-beam CT | DICOM modality CT |
| Magnetic resonance imaging (MRI) | Imaging using magnetic fields and radiofrequency pulses | Confused with CT | DICOM modality MR |
| Ultrasound (US) | Imaging using high-frequency sound | Confused with echocardiography | DICOM modality US |
| Positron emission tomography (PET) | Functional imaging using radiotracers | Confused with SPECT | DICOM modality PT |
| Mammography | Low-dose x-ray of the breast | Confused with breast MRI | DICOM modality MG |
| Hounsfield unit (HU) | Scaled CT attenuation relative to water and air | Treated as a universal tissue value | HU |
| Window center (WC) | Midpoint of the displayed HU range | Confused with window width | HU |
| Window width (WW) | Range of HU mapped to display | Confused with contrast | HU |
| Rescale slope | Multiplier converting stored pixel values to HU | Ignored in pipelines | Dimensionless |
| Rescale intercept | Offset converting stored pixel values to HU | Ignored in pipelines | HU |
| DICOM | Digital Imaging and Communications in Medicine | Treated as a file format only | PS3.x standard |
| SOP Instance UID | Unique identifier for a DICOM object | Confused with Study UID | UID |
| Study Instance UID | Unique identifier for a study | Confused with accession number | UID |
| Series Instance UID | Unique identifier for a series | Confused with study | UID |
| PACS | Picture Archiving and Communication System | Confused with the EHR | System |
| VNA | Vendor Neutral Archive | Confused with PACS | System |
| DICOMweb | RESTful DICOM services | Confused with DIMSE | Standard |
| Transfer syntax | Encoding rules for DICOM pixel data | Ignored in decompression | UID |
| Photometric interpretation | Meaning of pixel values (monochrome, RGB, palette) | Ignored | DICOM attribute |
| Pixel spacing | Physical size of a pixel | Confused with slice thickness | mm |
| Slice thickness | Thickness of a reconstructed slice | Confused with spacing | mm |
| Reconstruction kernel | Algorithm shaping image texture | Ignored in model transport | DICOM attribute |
| CTDIvol | Volume computed tomography dose index | Confused with effective dose | mGy |
| DLP | Dose-length product | Confused with effective dose | mGy·cm |
| Effective dose | Population-averaged risk-weighted dose | Treated as patient-specific dose | mSv |
| BI-RADS | Breast Imaging Reporting and Data System | Treated as a diagnosis | ACR lexicon |
| Lung-RADS | Lung CT screening reporting system | Treated as a pathology report | ACR lexicon |
| LI-RADS | Liver imaging reporting system | Treated as a diagnosis | ACR lexicon |
| PI-RADS | Prostate imaging reporting system | Treated as a diagnosis | ACR lexicon |
| Fleischner Society | Guidelines for incidental pulmonary nodule follow-up | Confused with Lung-RADS | Society guidance |
| Structured report | Report with coded fields and values | Confused with narrative text | DICOM SR / FHIR |
| CAD | Computer-aided detection or diagnosis | Confused with autonomous AI | Device category |
| SUV | Standardized uptake value in PET | Treated as a diagnosis | Dimensionless |
| T1 / T2 | MRI relaxation time constants | Confused with sequences | ms |
| FLAIR | Fluid-attenuated inversion recovery | Confused with T2 | MRI sequence |
| DWI / ADC | Diffusion-weighted imaging / apparent diffusion coefficient | Confused with perfusion | MRI sequence |
| SUVmax | Maximum standardized uptake value | Confused with mean SUV | Dimensionless |
| Ground-glass opacity | Hazy increased attenuation not obscuring vessels | Confused with consolidation | Descriptive term |
| Consolidation | Increased attenuation obscuring vessels | Confused with ground-glass | Descriptive term |
| Nodule | Rounded opacity | Confused with mass by size | mm |
| Mass | Lesion larger than a nodule threshold | Threshold varies by lexicon | mm |

## 5. The mechanics, step by step

### Acquisition

**Step 1. Confirm the clinical question and the appropriate modality.**

- Input: clinical question, patient history, prior imaging.
- Operation: choose modality, body part, and protocol.
- Output: an order with a protocol.
- Reasoning: the modality determines what can be seen. CT is fast and sensitive for bone and
  acute hemorrhage; MRI is superior for soft tissue and does not use ionizing radiation; ultrasound
  is real-time, portable, and operator-dependent; PET provides metabolic information.
- Branch point: contraindications. MRI is unsafe with certain implants; iodinated contrast is
  risky in severe renal impairment; radiation is cumulative.
- Analytic artifact: `order_id`, `modality`, `body_part`, `protocol`, `indication`.

**Step 2. Acquire the images.**

- Input: patient and protocol.
- Operation: position, scout, acquisition, contrast timing.
- Output: raw data.
- Reasoning: acquisition parameters determine image quality and comparability. Slice thickness,
  pitch, tube voltage, and reconstruction kernel affect the appearance of the same anatomy.
  Contrast phase affects whether a lesion enhances.
- Branch point: motion, poor inspiratory effort, and patient size degrade quality and may
  require repeat acquisition, which adds dose.
- Analytic artifact: `acquisition_ts`, `device`, `protocol`, `contrast_flag`, `phase`.

**Step 3. Reconstruct and post-process.**

- Input: raw data.
- Operation: filtered back projection or iterative reconstruction; multi-planar reformats;
  maximum intensity projections.
- Output: image series.
- Reasoning: reconstruction changes image texture and noise. Two scanners with different kernels
  produce different pixel distributions for the same anatomy. A model trained on one kernel may
  not transport to another.
- Branch point: some reconstructions are designed for display, others for measurement. The
  series description usually indicates which.
- Analytic artifact: `series_description`, `kernel`, `slice_thickness`, `reconstruction_algorithm`.

### DICOM and storage

**Step 4. Parse the DICOM metadata.**

- Input: DICOM object.
- Operation: read patient, study, series, and SOP identifiers; modality; acquisition parameters;
  pixel spacing; rescale values; window values.
- Output: a metadata record linked to the pixel data.
- Reasoning: the identifiers define the hierarchy: patient → study → series → instance. The
  Study Instance UID is the unit of an exam; the Series Instance UID is the unit of a acquisition
  series. Mixing them causes duplicate counting.
- Branch point: anonymized datasets may replace identifiers; the replacement must preserve the
  hierarchy.
- Analytic artifact: a DICOM metadata table with one row per instance.

**Step 5. Convert stored pixel values to physical units.**

- Input: pixel array, rescale slope, rescale intercept.
- Operation: HU = stored value x slope + intercept for CT.
- Output: an array in HU.
- Reasoning: the stored values are device-specific integers. Without the rescale, a value of 500
  could mean 500 HU or -524 HU depending on the intercept. For MRI, the stored values are
  arbitrary units and the rescale is usually identity; absolute quantification requires
  additional calibration.
- Branch point: some modalities use unsigned representations and require a shift; check
  `PixelRepresentation`.
- Analytic artifact: `hu_array`, `rescale_slope`, `rescale_intercept`.

**Step 6. Apply windowing for display, not for measurement.**

- Input: HU array, window center, window width.
- Operation: map HU to display intensity.
- Output: a display image.
- Reasoning: the human eye can distinguish a limited number of gray levels. Windowing selects
  the range of interest: lung window (center approximately -600 HU, width approximately 1,500 HU)
  shows parenchyma; mediastinal window (center approximately 40 HU, width approximately 400 HU)
  shows soft tissue. A model can use either the raw HU or a windowed representation, but the choice
  must be stated.
- Branch point: windowing is lossy. A model trained on windowed images cannot recover the full HU
  range.
- Analytic artifact: `window_center`, `window_width`, `display_array`.

**Step 7. Store and retrieve via PACS or DICOMweb.**

- Input: DICOM objects.
- Operation: archive, query, retrieve.
- Output: a study accessible by identifier.
- Reasoning: PACS stores images; the EHR stores the report and a link. DICOMweb provides
  RESTful access (QIDO-RS for query, WADO-RS for retrieval, STOW-RS for storage). A pipeline that
  retrieves by accession number may fail when accession numbers are reused or missing.
- Branch point: retrieval by Study Instance UID is more reliable than retrieval by patient ID and
  date.
- Analytic artifact: `retrieval_query`, `retrieval_ts`, `object_count`.

### Interpretation and reporting

**Step 8. Review the images in context.**

- Input: images, prior studies, clinical history.
- Operation: systematic search, comparison, measurement.
- Output: findings.
- Reasoning: comparison with priors is central to radiology. A nodule that is stable for two
  years is managed differently from a new nodule. A dataset that does not link priors cannot
  reproduce the radiologist's reasoning.
- Branch point: priors may be at another institution and unavailable.
- Analytic artifact: `prior_study_uid`, `comparison_flag`.

**Step 9. Apply a structured lexicon where one exists.**

- Input: findings.
- Operation: assign a category (for example, Lung-RADS 4B, BI-RADS 4).
- Output: a category with management recommendation.
- Reasoning: lexicons standardize communication and management. They are not diagnoses. A
  Lung-RADS 4B nodule is "very suspicious," not "cancer."
- Branch point: lexicon versions change; a category assigned under v1.1 may differ from v2022.
- Analytic artifact: `lexicon`, `lexicon_version`, `category`.

**Step 10. Write the report.**

- Input: findings and impression.
- Operation: dictate or template a narrative; optionally produce a structured report.
- Output: a signed report.
- Reasoning: the report is the primary communication artifact. It contains the impression, the
  recommendation, and the uncertainty. The findings section may contain measurements that are
  not in the impression.
- Branch point: templates improve completeness but can introduce boilerplate that is not
  patient-specific.
- Analytic artifact: `report_text`, `report_ts`, `author_role`, `structured_fields`.

**Step 11. Communicate critical findings.**

- Input: a finding requiring urgent action.
- Operation: direct communication to the responsible clinician, with documentation.
- Output: a documented communication.
- Reasoning: a critical finding (for example, a large pneumothorax or a misplaced tube) requires
  timely communication, not just a report in the queue.
- Branch point: the communication may occur by phone and be documented in a note, not in the
  report.
- Analytic artifact: `critical_flag`, `communication_ts`, `recipient_role`.

### Measurement and dose

**Step 12. Measure lesions consistently.**

- Input: image and measurement tool.
- Operation: measure diameter, volume, or enhancement.
- Output: a measurement with units and method.
- Reasoning: diameter is the most common measurement and the least reproducible for irregular
  lesions. Volume is more reproducible but depends on segmentation. RECIST (Response Evaluation
  Criteria in Solid Tumors) defines specific rules for target lesions.
- Branch point: the same lesion measured on two slices can differ by several millimeters.
- Analytic artifact: `measurement_value`, `measurement_method`, `slice_uid`.

**Step 13. Compute nodule volume from diameter.**

- Input: mean diameter.
- Operation: assume a sphere; V = (4/3) x pi x r³.
- Output: volume in mm³.
- Reasoning: the sphere assumption is an approximation. Real nodules are irregular. The
  approximation is useful for growth assessment and for converting between diameter and volume
  thresholds.
- Branch point: for a part-solid nodule, the solid component volume is more relevant than the
  total volume.
- Analytic artifact: `volume_mm3`, `diameter_mm`.

**Step 14. Compute volume doubling time.**

- Input: two volumes and the interval between measurements.
- Operation: VDT = (t x ln2) / ln(V2/V1).
- Output: months or days.
- Reasoning: exponential growth implies a constant doubling time. A short VDT suggests faster
  growth and higher suspicion. The calculation assumes exponential growth and accurate volumes.
- Branch point: slow-growing nodules may have a VDT longer than the follow-up interval, so a
  single interval may not establish growth.
- Analytic artifact: `vdt_days`, `interval_days`.

**Step 15. Estimate radiation dose.**

- Input: CTDIvol, scan length, conversion factor.
- Operation: DLP = CTDIvol x length; effective dose = DLP x k.
- Output: mSv.
- Reasoning: CTDIvol is a measure of scanner output; DLP incorporates scan length; the k factor
  converts to effective dose and is body-region-specific. Effective dose is a population average,
  not a patient-specific risk.
- Branch point: pediatric patients have different k factors and higher sensitivity.
- Analytic artifact: `ctdi_vol`, `dlp`, `effective_dose_msv`.

### Data science

**Step 16. Build the imaging cohort.**

- Input: DICOM metadata and reports.
- Operation: select by modality, body part, protocol, date, and indication.
- Output: a cohort with study identifiers.
- Reasoning: the same patient may have multiple studies. The unit of analysis must be stated:
  patient, study, series, or image. Image-level splitting leaks patient information.
- Branch point: a patient with multiple studies in both train and test inflates performance.
- Analytic artifact: `cohort_id`, `unit_of_analysis`, `split_strategy`.

**Step 17. Derive labels from reports.**

- Input: report text.
- Operation: NLP extraction with negation, uncertainty, and comparison handling.
- Output: labels with confidence.
- Reasoning: "no pneumonia," "possible pneumonia," and "pneumonia" are different. A binary label
  loses this distinction. A three-class label or a confidence score preserves it.
- Branch point: report language varies by radiologist and institution.
- Analytic artifact: `label`, `label_confidence`, `extraction_method`.

**Step 18. Preprocess images reproducibly.**

- Input: DICOM series.
- Operation: resample to isotropic spacing, crop or pad, normalize intensity, apply augmentation.
- Output: tensors ready for a model.
- Reasoning: resampling changes anatomy and lesion size; normalization changes contrast.
  Augmentation must respect anatomy (for example, left-right flips are not valid for laterality-
  specific tasks).
- Branch point: cropping around a lesion requires a lesion annotation that may not exist at
  inference.
- Analytic artifact: a preprocessing specification with version.

**Step 19. Evaluate by subgroup and acquisition stratum.**

- Input: model predictions.
- Operation: compute performance by scanner, protocol, site, sex, age, and disease severity.
- Output: a stratified evaluation.
- Reasoning: imaging models often fail on subgroups defined by acquisition parameters or
  prevalence. Overall AUC can hide this.
- Branch point: if a subgroup is too small, report the uncertainty rather than a point estimate.
- Analytic artifact: a stratified performance report.

**Step 20. Plan for deployment.**

- Input: model and workflow.
- Operation: define where the output appears, who sees it, and what action it triggers.
- Output: a deployment specification.
- Reasoning: a model that reads after sign-off cannot change the report. A model that pre-reads
  can prioritize the worklist. A model that flags critical findings must have an escalation path.
- Branch point: regulatory status depends on the intended use and the level of autonomy.
- Analytic artifact: a deployment and monitoring plan.

### 5.1 Modality comparison at a glance

| Modality | Ionizing radiation | Typical effective dose | Strength | Main data limitation |
|---|---|---:|---|---|
| Chest radiograph | Yes | About 0.1 mSv | Fast, portable, cheap | 2D projection, overlap |
| CT chest | Yes | About 6.1 mSv | Cross-sectional, sensitive | Dose, acquisition heterogeneity |
| Low-dose CT chest | Yes | About 1.5 mSv | Screening | Lower sensitivity for small lesions |
| CT abdomen/pelvis | Yes | About 7.7 mSv | Acute abdomen, trauma | Dose, contrast phase |
| MRI brain | No | None | Soft tissue contrast | Motion, implant contraindications |
| Ultrasound | No | None | Real-time, portable | Operator dependence |
| PET/CT | Yes | About 22.7 mSv | Metabolic activity | Cost, availability, resolution |

The dose values are population averages from the ACR reference card and RadiologyInfo. They are
not patient-specific doses and they vary with protocol, patient size, and scanner.

## 6. Worked example

A 64-year-old woman with a 30 pack-year smoking history, who quit five years ago, attends a lung
cancer screening visit. She is asymptomatic. A low-dose chest CT (LDCT) is performed. The
radiologist measures a solid nodule in the right upper lobe at 24 mm in mean diameter. A prior
LDCT 18 months earlier, performed at the same institution, showed the same nodule at 18 mm.

### 6.1 The measurements

| Item | Current | Prior (18 months) | Unit |
|---|---:|---:|---|
| Nodule mean diameter | 24 | 18 | mm |
| Nodule type | Solid | Solid | — |
| Location | Right upper lobe | Right upper lobe | — |
| Emphysema | Mild | Mild | — |
| Mediastinal lymph nodes | Not enlarged | Not enlarged | — |

### 6.2 Step 1: classify with Lung-RADS

The American College of Radiology (ACR) Lung-RADS v2022 classification for a solid nodule at
baseline or annual screening is:

| Category | Solid nodule size | Management |
|---|---|---|
| 2 | Less than 6 mm | Annual LDCT |
| 3 | 6 to less than 8 mm | 6-month LDCT |
| 4A | 8 to less than 15 mm | 3-month LDCT; PET/CT if 8 mm or larger |
| 4B | 15 mm or larger | Diagnostic chest CT, PET/CT, tissue sampling, or referral |

A 24 mm solid nodule is category 4B, "very suspicious." The category is not a diagnosis of
cancer. It is a management recommendation based on size and appearance.

Growth is also a criterion. Lung-RADS v2022 defines growth as an increase of more than 1.5 mm in
mean diameter within a 12-month interval. This nodule grew 6 mm over 18 months, which meets the
growth criterion regardless of the size category.

### 6.3 Step 2: compute the volumes

Assuming a sphere:

```text
V = (4/3) x pi x r³

Current: r = 12 mm
V = (4/3) x 3.14159 x 1728 = 7,238 mm³

Prior: r = 9 mm
V = (4/3) x 3.14159 x 729 = 3,054 mm³
```

The volume ratio is 7,238 / 3,054 = 2.37. The nodule has more than doubled in volume over 18
months.

### 6.4 Step 3: compute the volume doubling time

```text
VDT = (t x ln 2) / ln(V2 / V1)
    = (18 x 0.6931) / ln(2.3704)
    = 12.476 / 0.8630
    = 14.46 months
```

A volume doubling time of about 14.5 months is in the range associated with malignancy, though
there is no single threshold that separates benign from malignant. Rapid growth (for example,
a VDT of less than 100 days) suggests aggressive disease or infection; very slow growth (for
example, more than 400-600 days) suggests a benign process. The cut-offs are heuristics, not
diagnostic criteria.

> **Note:** The sphere assumption overestimates the volume of irregular nodules and is sensitive
> to measurement error. A 1 mm error in diameter changes the volume by about 13% at this size.
> Volumetric segmentation is more reproducible but depends on software and technique.

### 6.5 Step 4: the dose comparison

The LDCT was acquired with a volume computed tomography dose index (CTDIvol) of 3.0 mGy over a
30 cm scan length. Using a chest conversion factor of 0.014 mSv per mGy·cm:

```text
DLP = CTDIvol x scan length = 3.0 x 30 = 90 mGy·cm
Effective dose = DLP x k = 90 x 0.014 = 1.26 mSv
```

The recommended diagnostic chest CT would be acquired with a CTDIvol of 12 mGy over 36 cm:

```text
DLP = 12 x 36 = 432 mGy·cm
Effective dose = 432 x 0.014 = 6.05 mSv
```

The diagnostic CT delivers about 4.8 times the dose of the screening study. For context, the
average annual background radiation in the United States is approximately 3 mSv, a chest
radiograph is approximately 0.1 mSv, and a standard CT chest is approximately 6.1 mSv according
to the ACR dose reference card and RadiologyInfo. The numbers vary with patient size, scanner,
and protocol.

The dose difference is not a reason to avoid the diagnostic CT when it is indicated. It is a
reason to ensure the indication is real and the protocol is optimized.

### 6.6 Step 5: the Hounsfield measurements

The radiologist places a region of interest (ROI) on the nodule and on reference tissues:

| Tissue | Mean HU | Interpretation |
|---|---:|---|
| Nodule | 45 | Soft tissue |
| Lung parenchyma | -850 | Lung |
| Subcutaneous fat | -95 | Fat |
| Muscle | 55 | Soft tissue |
| Rib | 700 | Bone |
| Air outside body | -1,000 | Air |

The nodule is soft-tissue density, not fat and not calcified. A calcified nodule (for example,
above 200 HU) would be more likely benign. A fat-containing nodule (for example, -40 to -120 HU)
suggests a hamartoma. The density measurement is one input to the assessment, not a diagnosis.

### 6.7 Step 6: the display window

The same CT data are displayed with two windows:

| Window | Center (HU) | Width (HU) | Purpose |
|---|---:|---:|---|
| Lung | -600 | 1,500 | Parenchyma and nodules |
| Mediastinal | 40 | 400 | Soft tissue and vessels |

The mapping from HU to display intensity is:

```text
display = ((HU - (center - width/2)) / width) x 255, clamped to 0-255
```

For the nodule at 45 HU:

```text
Lung window: ((45 - (-1350)) / 1500) x 255 = (1395 / 1500) x 255 = 237
Mediastinal window: ((45 - (-160)) / 400) x 255 = (205 / 400) x 255 = 131
```

In the lung window, soft tissue appears near white and the nodule is easy to see against dark
lung. In the mediastinal window, the same nodule is mid-gray. A model trained on one window sees
a different image from a model trained on the other. The choice must be recorded.

### 6.8 Step 7: the DICOM metadata

A single DICOM instance from the study might contain:

| Attribute | Value | Why it matters |
|---|---|---|
| StudyInstanceUID | 1.2.840... (example) | Identifies the study |
| SeriesInstanceUID | 1.2.840... (example) | Identifies the series |
| SOPInstanceUID | 1.2.840... (example) | Identifies the image |
| Modality | CT | Selects the pipeline |
| BodyPartExamined | CHEST | Cohort filter |
| SliceThickness | 1.0 mm | Affects measurement |
| PixelSpacing | 0.7\0.7 mm | Physical scale |
| RescaleSlope | 1 | HU conversion |
| RescaleIntercept | -1024 | HU conversion |
| WindowCenter | -600 | Display |
| WindowWidth | 1500 | Display |
| KVP | 100 | Acquisition |
| CTDIvol | 3.0 mGy | Dose |

A pipeline that reads only the pixel array and ignores `PixelSpacing` cannot measure the nodule
in millimeters. A pipeline that ignores `RescaleIntercept` cannot compute HU. A pipeline that
ignores `StudyInstanceUID` cannot link the image to the report.

### 6.9 Step 8: the report and the label

The report impression might read:

> "24 mm solid right upper lobe nodule, increased from 18 mm on 18 months prior. Suspicious for
> primary lung malignancy. Recommend diagnostic chest CT with contrast and PET/CT, with
> tissue sampling as clinically appropriate. Lung-RADS 4B."

A naive NLP label might extract "malignancy" and label the study positive for cancer. A careful
label would extract "suspicious for" and assign a suspicion category, not a diagnosis. The
difference matters: the study is not biopsy-proven, and the patient may ultimately have a benign
granuloma.

The label for a model should match the decision. If the decision is "who should be referred for
diagnostic workup," the label can be "Lung-RADS 4B or higher." If the decision is "who has
cancer," the label must be biopsy or pathology, not the report.

## 7. Code, query, or calculation

The block computes volumes, doubling time, dose, window mapping, and HU conversion.

```python
"""Imaging calculations for a lung nodule case.

The sphere volume model is an approximation. The dose conversion factor is
body-region-specific and population-averaged.
"""

import math


def sphere_volume_mm3(diameter_mm: float) -> float:
    radius = diameter_mm / 2.0
    return (4.0 / 3.0) * math.pi * radius ** 3


def volume_doubling_time_months(v1_mm3: float, v2_mm3: float,
                                interval_months: float) -> float:
    if v1_mm3 <= 0 or v2_mm3 <= 0:
        raise ValueError("volumes must be positive")
    return interval_months * math.log(2) / math.log(v2_mm3 / v1_mm3)


def effective_dose_msv(ctdi_vol_mgy: float, length_cm: float,
                       k_msv_per_mgy_cm: float = 0.014) -> tuple[float, float]:
    dlp = ctdi_vol_mgy * length_cm
    return dlp, dlp * k_msv_per_mgy_cm


def hu_to_display(hu: float, center: float, width: float) -> int:
    raw = ((hu - (center - width / 2.0)) / width) * 255.0
    return int(round(max(0.0, min(255.0, raw))))


v_current = sphere_volume_mm3(24)
v_prior = sphere_volume_mm3(18)
v_8mm = sphere_volume_mm3(8)
vdt = volume_doubling_time_months(v_prior, v_current, 18)

print(f"volume at 24 mm: {v_current:.1f} mm3")
print(f"volume at 18 mm: {v_prior:.1f} mm3")
print(f"volume at 8 mm: {v_8mm:.1f} mm3")
print(f"volume ratio: {v_current / v_prior:.3f}")
print(f"volume doubling time: {vdt:.2f} months")

dlp_screen, dose_screen = effective_dose_msv(3.0, 30.0)
dlp_diag, dose_diag = effective_dose_msv(12.0, 36.0)
print()
print(f"LDCT DLP: {dlp_screen:.0f} mGy*cm; effective dose: {dose_screen:.2f} mSv")
print(f"Diagnostic CT DLP: {dlp_diag:.0f} mGy*cm; effective dose: {dose_diag:.2f} mSv")
print(f"dose ratio: {dose_diag / dose_screen:.2f}")

print()
print(f"lung window, HU 45: {hu_to_display(45, -600, 1500)}")
print(f"mediastinal window, HU 45: {hu_to_display(45, 40, 400)}")
print(f"lung window, HU -850: {hu_to_display(-850, -600, 1500)}")
print(f"mediastinal window, HU 700: {hu_to_display(700, 40, 400)}")

# DICOM rescale
stored_value = 500
slope = 1.0
intercept = -1024.0
hu = stored_value * slope + intercept
print()
print(f"stored 500 with slope 1, intercept -1024 -> {hu:.0f} HU")

assert round(v_current, 1) == 7238.2
assert round(v_prior, 1) == 3053.6
assert round(vdt, 2) == 14.46
assert round(dose_screen, 2) == 1.26
assert round(dose_diag, 2) == 6.05
```

Expected output:

```text
volume at 24 mm: 7238.2 mm3
volume at 18 mm: 3053.6 mm3
volume at 8 mm: 268.1 mm3
volume ratio: 2.370
volume doubling time: 14.46 months

LDCT DLP: 90 mGy*cm; effective dose: 1.26 mSv
Diagnostic CT DLP: 432 mGy*cm; effective dose: 6.05 mSv
dose ratio: 4.80

lung window, HU 45: 237
mediastinal window, HU 45: 131
lung window, HU -850: 85
mediastinal window, HU 700: 255

stored 500 with slope 1, intercept -1024 -> -524 HU
```

A DICOM metadata query using pydicom:

```python
import pydicom

ds = pydicom.dcmread("instance.dcm", stop_before_pixels=True)
record = {
    "study_uid": ds.StudyInstanceUID,
    "series_uid": ds.SeriesInstanceUID,
    "sop_uid": ds.SOPInstanceUID,
    "modality": ds.Modality,
    "body_part": getattr(ds, "BodyPartExamined", None),
    "slice_thickness_mm": float(getattr(ds, "SliceThickness", "nan")),
    "pixel_spacing_mm": [float(x) for x in getattr(ds, "PixelSpacing", [])],
    "rescale_slope": float(getattr(ds, "RescaleSlope", 1.0)),
    "rescale_intercept": float(getattr(ds, "RescaleIntercept", 0.0)),
    "window_center": float(getattr(ds, "WindowCenter", 0.0)),
    "window_width": float(getattr(ds, "WindowWidth", 0.0)),
}
print(record)
```

The `stop_before_pixels=True` flag matters for large studies: metadata can be read without
loading the pixel data, which is much faster and avoids memory pressure.

## 8. Data, standards, and regulatory context

**DICOM.** DICOM is maintained by the National Electrical Manufacturers Association (NEMA) and
published as the PS3 series. The 2025d release was published on 21 September 2025. The standard
covers image formats, network services, media storage, and structured reporting. PS3.1 is the
introduction and overview; PS3.3 defines the information object definitions; PS3.5 defines the
data structure and encoding; PS3.6 defines the data dictionary; PS3.18 defines web services.

Recent supplements include frame deflate transfer syntax, waveform presentation state, a
structural heart structured report template, patient model gender enhancements aligned with HL7
Gender Harmony, and DICOMweb modality procedure step services. The FDA recognized PS3.1-3.20
2025d in December 2025, with declarations to the older 2024e version accepted until December
2026. These are format and interoperability requirements, not clinical guidance.

**HL7 and FHIR.** FHIR R4 defines ImagingStudy, which describes a study and its series and
instances, and links to endpoints. DICOMweb provides RESTful access to DICOM objects: QIDO-RS for
query, WADO-RS for retrieval, and STOW-RS for storage. Integrating the Healthcare Enterprise
(IHE) publishes profiles that combine these standards for specific workflows, such as scheduled
workflow and cross-enterprise document sharing.

**Structured reporting.** DICOM Structured Reporting (SR) encodes report content as a tree of
coded concepts and values. It is more machine-readable than narrative text but less widely used
than narrative reports. The ACR's RADS lexicons (BI-RADS, Lung-RADS, LI-RADS, PI-RADS) define
categories and management recommendations; they are clinical communication tools, not
regulatory classifications.

**Radiation dose.** The ACR dose reference card and RadiologyInfo publish typical effective doses:
chest radiograph approximately 0.1 mSv; CT head approximately 1.6 mSv; CT chest approximately 6.1
mSv; CT abdomen and pelvis approximately 7.7 mSv; low-dose CT chest for screening approximately
1.5 mSv; PET/CT whole body approximately 22.7 mSv; average annual background in the United States
approximately 3 mSv. These are population averages and vary with patient size, protocol, and
scanner. Effective dose is a risk-weighted quantity, not a patient-specific dose.

**Device regulation.** In the United States, software that analyzes images for clinical decision
support may be regulated as a medical device. The FDA's 510(k) and De Novo pathways are commonly
used for imaging AI. The intended use statement determines the regulatory pathway and the
claims that can be made. A "research use only" label does not automatically exempt a product
from regulation if it is used clinically. The FDA has published guidance on predetermined change
control plans for AI-enabled device software functions, which allows certain modifications to be
pre-authorized.

**Privacy.** Medical images can contain burned-in text with patient identifiers. De-identification
requires removing or replacing identifiers in both the DICOM metadata and the pixel data.
Metadata-only de-identification is insufficient when burned-in annotations exist.

> **Note:** This section is educational and is not medical advice or legal advice. Dose values,
> regulatory pathways, and lexicon recommendations must be verified against current sources
> before clinical or commercial use.

## 9. Industry practice and case evidence

**Lung-RADS changed the management of screening nodules.** The ACR released Lung-RADS v2022,
replacing v1.1, with revised categories and management recommendations. In v2022, a solid nodule
of 15 mm or larger is category 4B, managed with diagnostic chest CT, PET/CT, tissue sampling, or
referral. The category 0 pathway was added for likely infectious or inflammatory findings, and
the risk-of-malignancy column was removed. A dataset that mixes v1.1 and v2022 categories without
a version field is not reproducible.

**DICOM evolves continuously.** The 2025d release and the 2026 supplements show that the standard
is maintained on a rolling basis. New transfer syntaxes and service classes change what a
pipeline can read and how it retrieves studies. A pipeline pinned to an old DICOM library may
fail silently on a new compression type.

**Labels from reports are noisy.** CheXpert, a large public chest radiograph dataset published
by Irvin and colleagues in 2019, used a rule-based labeler to extract 14 observations from
reports, including uncertainty labels. The dataset's design acknowledged that report-derived
labels are imperfect and that uncertainty is a distinct category. The documented fact is the
labeling approach; the inference is that models trained on such labels learn report language as
well as image findings.

**Hidden stratification.** Oakden-Rayner and colleagues described hidden stratification in a
2020 paper: a model can perform well on an overall metric while failing on clinically important
subgroups that are not represented in the label. In imaging, subgroups may be defined by
acquisition device, protocol, body habitus, or disease subtype. The documented phenomenon is
that overall performance can mask subgroup failure. The inference is that stratified evaluation
is mandatory for imaging models.

**AI device clearances.** The FDA has cleared a large number of imaging AI devices, most in
radiology, with intended uses ranging from worklist prioritization to lesion detection. Clearance
is not a claim of clinical benefit; it is a determination that the device is safe and effective
for its intended use under the submitted evidence. A cleared device can still perform poorly at
a site with different acquisition protocols.

**What is documented versus inferred.** The standards, releases, and dataset descriptions above
are documented. The claim that a particular model improves patient outcomes requires a
prospective study and is not established by a clearance or a retrospective AUC.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Pixels read without rescale | HU distribution shifted | Intercept ignored | Wrong density thresholds | Apply slope and intercept |
| Metadata discarded | Cannot filter by protocol | Pixel-only pipeline | Mixed acquisitions | Preserve DICOM metadata |
| Patient-level leakage | Same patient in train and test | Multiple studies per patient | Inflated performance | Split by patient |
| Report label without negation | "No pneumonia" labeled positive | Naive keyword matching | Label noise | Negation and uncertainty handling |
| Prevalence shift | Screening model fails in clinic | Different pretest probability | Low PPV | Recalibrate; report prevalence |
| Acquisition shift | Performance drops at a new site | Kernel, thickness, device | Transport failure | Stratified evaluation |
| Windowing mismatch | Model sees different contrast | Display vs raw HU | Training-serving skew | Fix preprocessing version |
| Laterality flip augmentation | Left-right errors | Augmentation disrespects anatomy | Wrong side prediction | Restrict augmentation |
| Duplicate series counted | Study counts exceed orders | Multiple reconstructions | Overcounted exams | Deduplicate by study UID |
| Burned-in identifiers | PHI in pixels | Annotation overlay | Privacy breach | Pixel-level de-identification |
| Dose metric misused | Effective dose treated as patient dose | Population average | Miscommunicated risk | Label as population estimate |
| Lexicon version mixed | Category distributions shift | v1.1 vs v2022 | Non-reproducible cohort | Version field |
| Label from impression only | Findings contradict impression | Report structure | Mislabeling | Parse both sections |
| Model reads after sign-off | No workflow impact | Deployment point | No clinical value | Integrate before sign-off |
| Unsupported compression | Read failures | New transfer syntax | Missing studies | Pin library; test new releases |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The DICOM metadata is accurate | Measurement and cohort selection depend on it | Entry errors, anonymization | Compare with order and report |
| The rescale is identity for non-CT | MRI and ultrasound quantification | Vendor-specific scaling | Check rescale tags |
| The lesion is spherical | Volume from diameter | Irregular nodules | Compare with segmentation |
| Growth is exponential | Doubling time | Non-exponential growth | Serial measurements |
| The report reflects the image | Labels are derived from reports | Report error, template | Chart or image review |
| The lexicon version is constant | Category comparison | Annual updates | Version field |
| The acquisition protocol is comparable | Transportability | Scanner, kernel, thickness | Stratified analysis |
| Effective dose represents risk | Dose communication | Patient-specific factors | Use as population estimate |
| The model's intended use matches deployment | Regulatory and clinical fit | Different population or workflow | Intended-use review |
| Priors are available | Comparison reasoning | Outside imaging | Linkage audit |

## 12. Exercises

**Exercise 1 — Modality selection (recall and application, 40 minutes).**
For five scenarios — suspected stroke, suspected appendicitis in a child, suspected fracture,
suspected pulmonary embolism in pregnancy, and breast cancer screening — choose a modality and
justify the choice on physics, risk, and urgency. Deliverable: a five-row table. Acceptance
criteria: each choice has a rationale; radiation and contraindications addressed. Time: 40
minutes.

**Exercise 2 — HU and windowing (calculation, 35 minutes).**
A CT pixel has a stored value of 1,024, rescale slope 1, intercept -1024. Compute the HU. Then
compute the display value under a lung window (center -600, width 1,500) and a mediastinal
window (center 40, width 400). Deliverable: calculations with units. Acceptance criteria: HU and
both display values correct; the mapping formula stated. Time: 35 minutes.

**Exercise 3 — Nodule volume and growth (calculation, 45 minutes).**
A nodule measures 10 mm at baseline and 14 mm at 12 months. Compute both volumes, the volume
ratio, and the VDT. State whether it meets the Lung-RADS growth criterion. Deliverable: a
calculation sheet. Acceptance criteria: arithmetic correct; growth criterion applied; sphere
assumption stated. Time: 45 minutes.

**Exercise 4 — Dose comparison (calculation, 35 minutes).**
A CT protocol has CTDIvol 10 mGy and scan length 40 cm. Compute DLP and effective dose using
k = 0.014 mSv/mGy·cm. Compare with a chest radiograph at 0.1 mSv. Deliverable: a short
calculation and a two-sentence interpretation. Acceptance criteria: units correct; comparison
contextualized; population-average caveat stated. Time: 35 minutes.

**Exercise 5 — Report label audit (application, 60 minutes).**
Take 30 radiology reports and label them for pneumonia as positive, negative, uncertain, or
not mentioned. Compute the agreement between two annotators and discuss the sources of
disagreement. Deliverable: a labeled table and a 300-word analysis. Acceptance criteria:
uncertainty category used; agreement quantified; disagreement examples given. Time: 60 minutes.

**Exercise 6 — DICOM pipeline design (design, 75 minutes).**
Design a preprocessing pipeline for chest CT that preserves metadata, converts to HU, resamples
to isotropic spacing, and splits by patient. Specify the failure handling for missing rescale
tags. Deliverable: a pipeline diagram and a specification. Acceptance criteria: metadata
preserved; HU conversion explicit; patient-level split; failure handling defined. Time: 75
minutes.

**Exercise 7 — Deployment critique (critique, 50 minutes).**
A vendor proposes an AI tool that reads chest radiographs after the radiologist signs the report
and sends a summary to the quality dashboard. Critique the deployment and propose an alternative.
Deliverable: a 400-word critique. Acceptance criteria: identifies the workflow mismatch; proposes
a clinically actionable integration; addresses alert burden. Time: 50 minutes.

**Exercise 8 — lexicon version audit (application, 45 minutes).**
A dataset contains Lung-RADS categories from 2019 to 2026 without a version field. Design an
audit to detect version mixing and propose a correction. Deliverable: an audit plan and a
correction strategy. Acceptance criteria: version-sensitive categories identified; correction is
documented; no assumption that categories are stable. Time: 45 minutes.

## 13. Self-check answers

**Answer to Exercise 2.** HU = 1,024 x 1 + (-1,024) = 0 HU. Lung window: ((0 - (-1,350)) /
1,500) x 255 = (1,350/1,500) x 255 = 229.5, rounded to 230. Mediastinal window: ((0 - (-160)) /
400) x 255 = (160/400) x 255 = 102. The same HU value appears bright in the lung window and
mid-gray in the mediastinal window.

**Answer to Exercise 3.** Volume at 10 mm = (4/3) x pi x 125 = 523.6 mm³. Volume at 14 mm =
(4/3) x pi x 343 = 1,436.8 mm³. Ratio = 2.744. VDT = 12 x 0.6931 / ln(2.744) = 8.317 / 1.009 =
8.24 months. The diameter increased by 4 mm in 12 months, which exceeds the 1.5 mm growth
criterion. The VDT is short, which raises suspicion.

**Answer to Exercise 4.** DLP = 10 x 40 = 400 mGy·cm. Effective dose = 400 x 0.014 = 5.6 mSv.
That is about 56 times a chest radiograph at 0.1 mSv, and about 1.9 times the annual background
of 3 mSv. The comparison is population-level and does not predict an individual's risk.

**Answer to self-check on HU.** Hounsfield units are defined relative to water (0 HU) and air
(-1,000 HU). They are computed from the linear attenuation coefficient: HU = 1,000 x
(mu_tissue - mu_water) / mu_water. The rescale slope and intercept convert stored pixel values to
this scale.

**Answer to self-check on windowing.** Windowing maps a range of HU to the display range. It is a
display transformation, not a measurement. A model can use raw HU or a windowed image, but it
must use the same transformation at training and inference.

**Answer to self-check on Lung-RADS.** A 24 mm solid nodule is 4B. The category is a management
recommendation, not a diagnosis. The next step is diagnostic evaluation, which may include
contrast CT, PET/CT, biopsy, or referral.

**Rubric for Exercise 6.** Full credit: metadata preserved; HU conversion with slope and
intercept; isotropic resampling; patient-level split; missing-tag failure handling; versioned
preprocessing. Half credit: pipeline works but metadata or split is underspecified. No credit:
pixel-only pipeline.

**Rubric for Exercise 7.** Full credit: identifies post-sign-off timing as a workflow failure;
proposes pre-read or worklist prioritization; addresses alert burden and escalation; states the
regulatory implication. Half credit: critiques accuracy only. No credit: accepts the dashboard as
sufficient.

## 14. Interview questions

**1. What is a Hounsfield unit, and why does it matter?**
A strong answer defines it relative to water and air and explains that it enables density
measurement. A weak answer says it is a CT number.

**2. What is the difference between window center and window width?**
A strong answer explains the mapping and gives an example. A weak answer confuses them with
contrast and brightness.

**3. Why can two CT scanners produce different images of the same patient?**
A strong answer names kernel, slice thickness, tube voltage, and reconstruction. A weak answer
says "different machines."

**4. Why is a radiology report not a gold-standard label?**
A strong answer explains uncertainty, negation, comparison, and report variation. A weak answer
says reports can be wrong.

**5. What is Lung-RADS, and what does a 4B category mean?**
A strong answer describes the lexicon, the size threshold, and the management recommendation. A
weak answer says it means cancer.

**6. How would you split an imaging dataset?**
A strong answer splits by patient, not by image, and explains leakage. A weak answer splits
randomly by image.

**7. What is hidden stratification?**
A strong answer explains overall performance masking subgroup failure and gives an imaging
example. A weak answer says the model is biased.

**8. How do you compute effective dose from CT parameters?**
A strong answer gives DLP = CTDIvol x length and effective dose = DLP x k, and notes the
population-average limitation. A weak answer quotes a single number.

**9. What is DICOMweb?**
A strong answer describes QIDO-RS, WADO-RS, and STOW-RS and their RESTful nature. A weak answer
says it is "DICOM over the web."

**10. How would you handle a missing rescale intercept?**
A strong answer checks the modality and the manufacturer, and treats the value as unknown rather
than defaulting to zero for CT. A weak answer defaults to zero.

**11. Why is deployment timing important for an imaging model?**
A strong answer explains that a post-sign-off model cannot change the report and gives examples
of actionable integration. A weak answer says deployment is an engineering detail.

**12. What does an FDA clearance for an imaging AI device establish?**
A strong answer says it establishes safety and effectiveness for the intended use under the
submitted evidence, not clinical benefit at every site. A weak answer says it proves the model
works.

## 15. Key terms

- **ACR** — American College of Radiology.
- **ADC** — apparent diffusion coefficient.
- **BI-RADS** — Breast Imaging Reporting and Data System.
- **CAD** — computer-aided detection or diagnosis.
- **CT** — computed tomography.
- **CTDIvol** — volume CT dose index.
- **DICOM** — Digital Imaging and Communications in Medicine.
- **DICOMweb** — RESTful DICOM services.
- **DLP** — dose-length product.
- **DWI** — diffusion-weighted imaging.
- **Effective dose** — population-averaged risk-weighted dose.
- **FLAIR** — fluid-attenuated inversion recovery.
- **Fleischner Society** — pulmonary nodule guidance body.
- **Ground-glass opacity** — hazy increased attenuation.
- **HU** — Hounsfield unit.
- **IHE** — Integrating the Healthcare Enterprise.
- **LDCT** — low-dose computed tomography.
- **LI-RADS** — Liver Imaging Reporting and Data System.
- **Lung-RADS** — lung cancer screening reporting system.
- **MRI** — magnetic resonance imaging.
- **Nodule** — rounded opacity, usually defined by size.
- **PACS** — Picture Archiving and Communication System.
- **PET** — positron emission tomography.
- **PI-RADS** — Prostate Imaging Reporting and Data System.
- **Pixel spacing** — physical size of a pixel.
- **Rescale intercept** — offset for HU conversion.
- **Rescale slope** — multiplier for HU conversion.
- **Series Instance UID** — identifier for a series.
- **SOP Instance UID** — identifier for a DICOM object.
- **Structured report** — coded report content.
- **Study Instance UID** — identifier for a study.
- **SUV** — standardized uptake value.
- **Transfer syntax** — encoding rules for pixel data.
- **VNA** — vendor neutral archive.
- **Volume doubling time** — time for volume to double.
- **Window center** — midpoint of displayed HU range.
- **Window width** — range of HU mapped to display.

## 16. Sources

Accessed 24 October 2026 unless otherwise stated.

**Standards and guidance**

- National Electrical Manufacturers Association. DICOM PS3.1 2025a, Introduction and Overview.
  https://dicom.nema.org/medical/dicom/2025a/output/html/part01.html
- DICOM Standards Committee. DICOM 2025d release, 21 September 2025.
  https://www.dicomstandard.org/current/
- Health Level Seven International. FHIR R4 ImagingStudy.
  https://hl7.org/fhir/R4/imagingstudy.html
- American College of Radiology. Lung-RADS v2022.
  https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Lung-Rads
- American College of Radiology. Radiation Dose to Adults From Common Imaging Examinations,
  reference card, April 2025.
  https://edge.sitecorecloud.io/americancoldf5f-acrorgf92a-productioncb02-3650/media/ACR/Files/Clinical/Radiology-Safety/Dose-Reference-Card.pdf
- RadiologyInfo. Radiation Dose from X-Ray and CT Exams, reviewed 15 April 2025.
  https://www.radiologyinfo.org/en/info/safety-xray
- US Food and Drug Administration. Marketing submission recommendations for a predetermined
  change control plan for AI/ML-enabled device software functions.
  https://www.fda.gov/medical-devices/software-medical-device-samd

**Evidence**

- Irvin J, Rajpurkar P, Ko M, et al. CheXpert: a large chest radiograph dataset with uncertainty
  labels and expert comparison. *Proceedings of the AAAI Conference on Artificial Intelligence*,
  2019. https://doi.org/10.1609/aaai.v33i01.33015901
- Oakden-Rayner L, Dunnmon J, Carneiro G, Re C. Hidden stratification causes clinically
  meaningful harms in machine learning models. *Proceedings of the ACM Conference on Health,
  Inference, and Learning*, 2020. https://doi.org/10.1145/3368555.3384469
- Christensen J, Prosper AE, Wu CC, et al. ACR Lung-RADS v2022: assessment categories and
  management recommendations. *Journal of the American College of Radiology*, 2023.
  https://www.jacr.org/article/S1546-1440(23)00761-5/fulltext

**Background reading**

- Herring W. *Learning Radiology: Recognizing the Basics*, 5th edition.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_disease-taxonomy-and-coding.md`](05_disease-taxonomy-and-coding.md)
- Next topic: [`07_cardiology-and-ecg-basics.md`](07_cardiology-and-ecg-basics.md)
- Related topic: [`03_laboratory-medicine.md`](03_laboratory-medicine.md)
- Related topic: [`10_clinical-plausibility-checks.md`](10_clinical-plausibility-checks.md)
- Survey file: [`../../01_clinical_foundations.md`](../../01_clinical_foundations.md)
- Labs: [`../../labs/`](../../labs/)
- ECG quality lab: [`../../labs/11_ecg_quality.ipynb`](../../labs/11_ecg_quality.ipynb)

