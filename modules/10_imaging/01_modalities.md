# Modalities

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 01 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 07 (deep learning foundations)
> Version 2026.3

## 1. Why this topic exists

A medical image is not a photograph of anatomy. It is a measurement of one physical property,
made by one instrument, under one acquisition protocol, reconstructed by one algorithm, and
stored in one encoding. Every one of those choices changes the numbers in the pixel matrix. A
model that treats all images as "pictures" will learn the instrument and the protocol as often as
it learns the disease.

The practical consequence is that modality is not a cosmetic label on a dataset. It determines
the unit of the signal, the spatial resolution, the contrast mechanism, the noise structure, the
artifacts, and the clinical question the image can answer. Computed tomography (CT) reports
attenuation in Hounsfield units (HU), a calibrated ratio scale with water at 0 HU and air near
-1000 HU. Magnetic resonance imaging (MRI) reports arbitrary intensity units whose meaning
depends on the pulse sequence, coil, and reconstruction. Positron emission tomography (PET)
reports radioactivity concentration in becquerels per millilitre, converted to a body-weight or
lean-body-mass standardized uptake value (SUV or SUL). Ultrasound reports echo amplitude after
log compression and time-gain compensation. Digital pathology reports light transmission through
stained tissue at a stated magnification and micrometres per pixel.

Those units are not interchangeable. A pipeline that concatenates CT and MRI volumes without
per-modality normalization has created a feature whose scale is the scanner, not the patient. A
model trained on 1.25 mm CT slices and deployed on 5 mm slices sees a different partial-volume
regime, a different noise level, and often a different reconstruction kernel. A lung nodule that
is 9 mm on thin-section CT may be invisible on a chest radiograph, not because the model is weak
but because the modality does not carry the contrast.

The industry problem is therefore upstream of modelling. Teams blocked at this layer cannot
answer basic questions: Which images are eligible for the intended use? Which modality does the
label refer to? Which acquisition parameters must be in the inclusion criteria? Which performance
number is comparable to the published literature? Getting modality wrong produces a model that
passes an internal test and fails a site validation, and the failure is often misdiagnosed as a
generalization problem when it is a measurement problem.

This topic builds the physical and data-model vocabulary that the rest of Module 10 assumes. It
does not teach radiology interpretation. It teaches what each modality measures, how that
measurement becomes a pixel, and which properties a data scientist must preserve or stratify.

### 1.1 The decisions this topic unblocks

| Decision | Why modality knowledge is required |
|---|---|
| Eligibility criteria for a dataset | The finding may not be visible, or may be measured differently, on another modality |
| Label definition | A report label for "nodule" means a CT measurement, not an X-ray impression |
| Preprocessing | HU clipping, MRI z-scoring, and pathology stain normalization are not interchangeable |
| Model input | 2D vs 3D, channel count, resolution, and intensity range all follow from the modality |
| Evaluation strata | Scanner, protocol, contrast phase, and slice thickness are modality-specific confounders |
| Regulatory intended use | A claim is tied to modality, anatomy, acquisition, and population |
| Monitoring | Drift in kVp, kernel, tracer, or stain protocol changes the input distribution |

### 1.2 What goes wrong when modality is treated as metadata only

Three failure patterns recur in industry post-mortems. First, a mixed-modality training set
produces a classifier that separates scanners rather than disease; the model has learned that one
scanner is used for sicker patients. Second, a model trained on contrast-enhanced CT is deployed
on non-contrast CT; the lesion-to-background contrast reverses or disappears, and sensitivity
falls without any software error. Third, a pathology model trained at 20x is deployed at 40x;
the object scale in pixels halves, and the model's effective receptive field no longer covers the
diagnostic structure.

None of these failures throws an exception. Each produces plausible outputs on every case. The
only defence is to make modality, acquisition, and units explicit artefacts of the data contract,
and to test performance within each stratum rather than only in aggregate.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** the signal chain from physical property to stored pixel value for CT, MRI,
   ultrasound, PET, mammography, and digital pathology, naming the measured quantity and its unit.
2. **Compute** Hounsfield units from attenuation or from stored values plus rescale slope and
   intercept, and compute SUV from activity concentration, injected dose, and body weight.
3. **Choose** a modality for a stated clinical task and justify the choice by contrast mechanism,
   resolution, and workflow, not by dataset availability.
4. **Characterize** a dataset by modality, acquisition protocol, pixel spacing, slice thickness,
   contrast phase, and reconstruction kernel, and identify which strata are too small to evaluate.
5. **Diagnose** a modality mismatch in a trained model from metadata, image statistics, and
   stratified performance rather than from saliency pictures alone.
6. **Critique** a cross-modality generalization claim by checking whether the units, resolution,
   and reference standard are comparable.
7. **Design** a modality-stratified evaluation table with a stated unit of analysis and reference
   standard for each row.

Each outcome is assessed by at least one exercise in Section 12 or one self-check item in
Section 13.

## 3. The core mental model

The model is a chain with six links. Each link can destroy information, and no downstream model
can recover what a link removed.

```text
Physical property          Acquisition              Reconstruction
(attenuation, proton       (source, detector,       (filtered backprojection,
 density, tracer,           timing, dose,            iterative, deep,
 impedance, stain)          contrast, sequence)      stain deconvolution)
        |                        |                         |
        v                        v                         v
   Measurement  ---------->  Raw signal  ---------->  Pixel matrix
   (line integral,           (counts, volts,           (integers, float,
    RF signal, echo)          photons)                  RGB)
        |                                                    |
        v                                                    v
   Calibration  ------------------------------------->  Display / analysis
   (HU, SUV, dB, µm/pixel,                              (window, LUT, model input)
    optical density)
```

The first link is physics. CT measures the linear attenuation coefficient integrated along a
ray; the reconstruction assigns a local value, and the rescale operation maps stored integers to
HU. MRI measures the response of hydrogen nuclei to radiofrequency pulses in a magnetic field;
the signal depends on proton density, T1 and T2 relaxation, diffusion, flow, and the sequence
timing, so the same tissue can be bright or dark in two acquisitions. Ultrasound measures
reflected sound pressure; the displayed brightness depends on acoustic impedance, angle, depth
gain, and operator technique. PET measures annihilation photon pairs from a radiotracer; the
reconstructed value is activity concentration, and the SUV normalization is a modelling choice.
Mammography measures low-energy X-ray transmission through compressed breast tissue. Digital
pathology measures transmitted light through a stained glass slide; the colour depends on stain
chemistry, thickness, scanner optics, and compression.

The second link is sampling. In-plane pixel spacing and through-plane slice thickness define a
voxel. When spacing is anisotropic, a "3D" convolution has a different physical extent in each
axis. When slice thickness is much larger than in-plane spacing, adjacent slices are correlated
and through-plane resolution is poor. Partial volume averaging mixes tissue classes within a
voxel; a 3 mm nodule in a 5 mm slice is diluted by surrounding lung, and its measured HU and SUV
fall.

The third link is encoding. DICOM stores integer pixel values, usually 12 or 16 bits, with a
rescale slope and intercept that map stored values to a real-world quantity. If the intercept is
missing or ignored, CT values shift by up to 1024 HU. If the photometric interpretation is
MONOCHROME1, high values are displayed dark rather than bright, and a model trained on inverted
images learns an inverted contrast. If the transfer syntax is compressed, decode errors can
produce block artefacts that correlate with the scanner vendor.

The fourth link is calibration. Not every modality has an absolute scale. CT does, approximately,
because the rescale is defined against water and air. PET has a defined unit but several
normalizations. MRI has no absolute scale, so intensity normalization is part of the model, not
a preprocessing afterthought. Ultrasound has a display scale that is not comparable across
machines. Pathology has optical density and stain concentrations that require colour
deconvolution or stain normalization.

The fifth link is display. Window centre and width map a range of HU to the displayable grey
range. The window is a viewing choice, not a property of the data. A model can consume the full
HU range, a fixed window, or a per-image percentile normalization; each choice changes what the
model can see. A radiologist's window is not automatically the right model input.

The sixth link is interpretation. The reference standard is produced by a human or a clinical
outcome, often using a different modality than the one the model sees. A CT model predicting a
diagnosis confirmed on pathology is not predicting the pathology image; it is predicting a
cross-modality inference. The gap between the image evidence and the reference standard is a
source of label noise that is modality-specific.

### 3.1 Contrast, resolution, and noise as the three axes

Any modality can be placed on three axes, and the placement predicts what tasks are feasible.

| Axis | Question | Consequence for modelling |
|---|---|---|
| Contrast | What physical difference separates lesion from background? | Determines whether the finding is visible at all |
| Resolution | What is the smallest object that survives sampling? | Determines the minimum lesion size and the value of upsampling |
| Noise | What variability is unrelated to anatomy? | Determines augmentation, denoising, and calibration strategy |

A task is feasible when the lesion produces a contrast difference larger than the noise at the
resolution of the acquisition. A 4 mm lung nodule on a 5 mm CT slice fails the resolution test
even though the contrast is adequate. A 2 mm breast microcalcification on a 100 µm mammography
pixel passes both. A prostate lesion on T2-weighted MRI may pass resolution but fail contrast
because benign hyperplasia and tumour overlap in signal.

### 3.2 The modality-to-task map

```text
High spatial resolution, high contrast:  mammography, thin-section CT, pathology
High spatial resolution, low contrast:   radiography, ultrasound
Low spatial resolution, high contrast:   PET, functional MRI
No absolute units:                       MRI, ultrasound, pathology (raw RGB)
Absolute or calibrated units:            CT (HU), PET (Bq/mL, SUV), mammography (optical density)
```

This map explains why segmentation is mature in CT and pathology, detection is mature in
mammography and CT, and classification is harder in MRI and ultrasound: the latter have strong
contrast for some tasks but no stable intensity meaning across sites.

### 3.3 Why one model cannot span modalities by default

A convolutional filter learns a local pattern of intensities. If the intensity scale changes,
the same filter responds differently. If the resolution changes, the same filter covers a
different physical area. If the contrast mechanism changes, the pattern that indicates disease
may invert. Multimodal foundation models address this by training on many modalities with
modality-specific tokenization or normalization, but they do not remove the need to validate
each modality separately. The model's shared representation is a hypothesis, not a guarantee.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Modality | The physical acquisition method and its DICOM code | Using "scan" for any image | DICOM codes CT, MR, US, PT, MG, DX, CR, SM, OCT |
| Hounsfield unit (HU) | Scaled linear attenuation relative to water and air | Treating stored integer as HU | HU; water 0, air about -1000 |
| Rescale slope/intercept | Affine map from stored pixel value to HU or other unit | Assuming slope 1 and intercept 0 | Dimensionless slope, intercept in HU |
| Window width (WW) | Range of HU mapped to display | Confusing with image contrast setting | HU |
| Window level (WL) | Centre of the displayed HU range | Calling it "brightness" | HU |
| Pixel spacing | Physical distance between pixel centres in row and column | Confusing with slice thickness | mm |
| Slice thickness | Nominal reconstructed slice width | Using it as the through-plane spacing | mm |
| Voxel | Volume element defined by in-plane spacing and slice spacing | Assuming isotropic | mm³ |
| Anisotropy | Unequal spacing across axes | Ignoring in 3D kernels | Ratio or mm per axis |
| Partial volume | Mixture of tissues within one voxel | Treating voxel value as pure tissue | Dimensionless fraction |
| Reconstruction kernel | Filter used in CT reconstruction | Ignoring kernel as a site signature | Vendor-specific names (B30f, B70f, I30f, etc.) |
| kVp | Peak X-ray tube voltage | Using it as a dose measure | kV |
| mAs | Tube current-time product | Confusing with effective dose | mAs |
| Contrast phase | Timing of contrast relative to acquisition | Pooling phases in one model | Unenhanced, arterial, portal venous, delayed |
| T1/T2 relaxation | MRI time constants | Treating T1 and T2 images as interchangeable | ms |
| TE/TR | Echo time and repetition time | Ignoring sequence timing | ms |
| FLAIR | Fluid-attenuated inversion recovery MRI | Calling it a T2 image | Sequence name |
| DWI/ADC | Diffusion-weighted imaging and apparent diffusion coefficient | Treating DWI bright signal as restricted diffusion without ADC | s/mm² (b-value), mm²/s (ADC) |
| B-mode | Two-dimensional brightness-mode ultrasound | Treating as calibrated intensity | Arbitrary dB |
| Doppler | Frequency-shift ultrasound for flow | Confusing with B-mode | cm/s or kHz shift |
| PET | Positron emission tomography | Using "PET scan" for PET/CT | DICOM PT |
| SUV | Standardized uptake value normalized to body weight | Comparing across tracers or protocols | Dimensionless (g/mL) |
| SUL | SUV normalized to lean body mass | Using body weight in obese patients | Dimensionless |
| Bq/mL | Activity concentration | Mixing with counts per second | Bq/mL |
| Whole slide image (WSI) | Gigapixel digitized pathology slide | Treating as a normal RGB photo | Pyramid TIFF or DICOM SM |
| µm/pixel | Physical size of a pathology pixel | Comparing magnification labels without calibration | µm |
| Optical density | Log-transformed light transmission | Using raw RGB as density | Dimensionless |
| Photometric interpretation | DICOM rule for colour/greyscale | Ignoring MONOCHROME1 inversion | MONOCHROME1, MONOCHROME2, RGB, YBR_FULL |
| Multi-frame | One DICOM object containing many frames | Treating each frame as a separate file | NumberOfFrames |
| Attenuation | Tissue property measured by CT | Confusing with density | cm⁻¹, converted to HU |
| Beam hardening | CT artefact from preferential low-energy absorption | Treating as anatomy | Artefact |
| Scatter | Detected photons not along the primary ray | Ignoring in PET quantification | Fraction |
| Signal-to-noise ratio (SNR) | Signal relative to noise | Reporting without method | Ratio |
| Contrast-to-noise ratio (CNR) | Lesion-background difference relative to noise | Confusing with SNR | Ratio |
| Reference standard | The truth used to score a model | Calling it "ground truth" | Pathology, follow-up, expert consensus |

### 4.1 DICOM modality codes that matter in this module

| Code | Meaning | Typical file content | Common data issue |
|---|---|---|---|
| CT | Computed tomography | 512×512 slices, 12-bit stored | Kernel and slice thickness vary |
| MR | Magnetic resonance | Multi-sequence series | No absolute intensity scale |
| PT | Positron emission tomography | 128×128 or 256×256 slices | Tracer, uptake time, reconstruction |
| US | Ultrasound | RGB or greyscale frames, often multi-frame | Operator, angle, no spatial calibration |
| MG | Mammography | 2D or tomosynthesis, high bit depth | Breast density, positioning, laterality |
| DX | Digital radiography | 2D projection | Portable technique, burned-in markers |
| CR | Computed radiography | 2D projection | Similar to DX, older detectors |
| SM | Slide microscopy | Multi-frame WSI | Stain, magnification, compression |
| OCT | Optical coherence tomography | Retinal cross-sections | Artifacts, segmentation, device-specific |

### 4.2 The unit ledger

Every metric in this module names its unit and reference standard. The same discipline applies
to modality descriptions.

| Quantity | Unit | Reference | Valid comparison? |
|---|---|---|---|
| CT attenuation | HU | Water and air calibration | Within kernel and kVp, approximately |
| PET uptake | SUV or SUL, dimensionless | Injected dose and body composition | Within tracer and uptake protocol |
| MRI intensity | Arbitrary | None | Only after normalization within sequence |
| Ultrasound amplitude | dB or arbitrary | None | Not across machines without phantom calibration |
| Mammographic density | Percent or BI-RADS category | Reader or software | Within acquisition type |
| Pathology stain | Optical density or RGB | Stain protocol | After colour deconvolution |
| Lesion size | mm | Calibrated pixel spacing | Yes if spacing correct |
| Lesion volume | mm³ | Calibrated spacing | Yes if segmentation and spacing correct |

## 5. The mechanics, step by step

This section is a procedure for turning a clinical task into a modality-aware data specification.
It is written for a data scientist who has to decide what to include, how to characterize it, and
what to stratify. Each step states its input, output, and rationale, then the branch points and
edge cases that change the decision.

### Step 1. State the clinical question and the task type

**Input:** A request from a clinical or product stakeholder, in plain language.

**Output:** One sentence naming the population, the finding, the decision, and the task type
(detection, segmentation, triage, diagnosis, or quantification).

**Rationale:** Modality follows from the decision, not from the archive. "Find lung nodules" is a
detection task on CT. "Decide whether a nodule is malignant" is a diagnosis task that may need
PET, prior imaging, and pathology. "Prioritize a head CT for haemorrhage" is a triage task on
non-contrast CT. Conflating these tasks produces an intended use that no single modality can
support.

**Branch points:**
- If the decision is binary and the finding is large, detection on the cheapest adequate modality
  may be sufficient.
- If the decision requires characterization (benign vs malignant), a single modality may not be
  adequate; the reference standard may be pathology or follow-up.
- If the decision is triage, the output is a priority and the tolerated false-negative rate is set
  by the downstream review, not by the model alone.

**Edge case:** A stakeholder asks for "an AI that reads all scans". Reject the scope. A model
without a named modality and finding cannot be evaluated or regulated.

### Step 2. Identify the physical contrast that encodes the finding

**Input:** The task statement and clinical literature.

**Output:** The physical property that differs between the finding and its background.

**Rationale:** A finding is visible only if it changes the measured property. Lung nodules differ
from air in X-ray attenuation, so CT has high contrast. Breast microcalcifications differ from
fat and fibroglandular tissue in attenuation, so mammography has high contrast. Prostate cancer
differs from benign tissue in T2 signal, diffusion, and perfusion, so multiparametric MRI
combines several contrasts because no single one is sufficient.

**Branch points:**
- If the contrast difference is smaller than the noise at the available resolution, the task is
  not feasible on that modality.
- If the contrast depends on a contrast agent, the protocol must include the agent and phase.
- If the contrast is functional rather than structural, the reference standard may be a clinical
  outcome rather than an anatomical measurement.

**Edge case:** A finding may be visible on a modality for the wrong reason. A chest tube is
visible on a radiograph and correlates with pneumothorax severity; a model may learn the tube
rather than the pneumothorax. Contrast is necessary but not sufficient for a valid feature.

### Step 3. Map the task to a modality and acquisition protocol

**Input:** The contrast requirement, the population, and the workflow.

**Output:** A primary modality and an allowed protocol envelope (body part, contrast phase,
slice thickness range, reconstruction kernel family, tracer and uptake time, stain and
magnification).

**Rationale:** The protocol envelope becomes the inclusion criteria and the regulatory intended
use. It is also the first defence against site signatures: if a model is trained only on one
kernel, it may learn the kernel's texture rather than the anatomy.

**Branch points:** see the decision table below.

**Decision table: modality selection for common findings**

| Finding or decision | Primary modality | Alternative | Why the alternative is weaker | Protocol constraint |
|---|---|---|---|---|
| Lung nodule detection | Thin-section CT | Chest radiograph | Nodules under about 1 cm are often obscured by ribs and mediastinum | Slice thickness ≤1.5 mm, lung kernel |
| Acute intracranial haemorrhage | Non-contrast head CT | MRI | Slower, less available in emergency | Axial 5 mm or less, no contrast |
| Large-vessel occlusion | CT angiography | MR angiography | Slower, contraindications | Arterial phase, 1 mm or less |
| Pulmonary embolism | CT pulmonary angiography | V/Q scan | Lower sensitivity for small emboli | Arterial phase, thin collimation |
| Breast cancer screening | Digital mammography or tomosynthesis | Ultrasound, MRI | Lower sensitivity in dense breasts, higher false positives | Standard views, compression |
| Prostate cancer risk | Multiparametric MRI | Systematic biopsy | Invasive, sampling error | T2, DWI, DCE per PI-RADS |
| Liver lesion characterization | Multiphase CT or MRI | Ultrasound | Lower specificity | Arterial, portal venous, delayed |
| Melanoma | Dermoscopy | Clinical photograph | Lower contrast for subsurface structures | Polarized, no ink markings |
| Cancer diagnosis and grading | H&E WSI | Frozen section | Lower quality, sampling | Stain protocol, 20x or 40x |
| Retinopathy | Fundus photography | OCT | Different structure | Non-mydriatic or mydriatic, macula-centred |

**Edge cases at this step:**
- A modality may be appropriate but unavailable at the deployment site; the product then needs a
  different intended use or a fallback pathway.
- A protocol may be standard at one site and exotic at another; "CT chest" is not a protocol.
- Paediatric and adult protocols differ in dose and technique; pooling them creates a site
  signature correlated with age.

### Step 4. Establish the measurement unit and calibration

**Input:** DICOM metadata and vendor documentation.

**Output:** A written statement of the unit the model consumes, and the conversion from stored
values.

**Rationale:** The model's input scale must be stable across sites. CT is the easiest case:
stored values are integers, and `RescaleSlope` and `RescaleIntercept` map them to HU. If the
tags are absent, the data are not guaranteed to be HU and must be flagged. PET requires activity
concentration and a normalization choice. MRI requires a documented normalization because the
scanner does not provide one. Pathology requires colour management and stain normalization.

**Branch points:**
- CT with slope 1 and intercept -1024: stored 0 maps to -1024 HU, stored 1024 to 0 HU.
- CT with slope 1 and intercept 0: stored values are already HU in some exports, but this must be
  verified against a water phantom or a known air region.
- PET with body-weight SUV vs lean-body-mass SUL: choose one and state it; do not mix.
- MRI with bias field: normalization may need N4 or a per-volume z-score before intensity-based
  comparison.

**Edge case:** A CT series may have `RescaleSlope` present but set to 1 and `RescaleIntercept`
absent. The absence is not equivalent to zero; the correct action is to check the
`RescaleType` and the vendor's conformance statement, and to exclude the series if the unit
cannot be established.

### Step 5. Characterize spatial resolution in all three axes

**Input:** `PixelSpacing`, `SliceThickness`, `ImagePositionPatient`, `ImageOrientationPatient`,
`Rows`, `Columns`, and the number of frames.

**Output:** A resolution table: in-plane spacing (mm), through-plane spacing (mm), field of view
(mm), matrix size, and anisotropy ratio.

**Rationale:** Resolution determines the smallest lesion that can be measured and the correct
interpolation strategy. It also determines whether 3D convolution is physically meaningful. A
volume with 0.7 mm in-plane and 5 mm through-plane spacing has an anisotropy ratio of about 7:1;
a 3×3×3 kernel covers 2.1 mm in-plane and 15 mm through-plane, which is not a cube in anatomy.

**Branch points:**
- If anisotropy ratio >3, either resample to isotropic spacing or use anisotropic kernels.
- If slice spacing is derived from `ImagePositionPatient`, use the median of successive
  projections onto the slice normal, not `SliceThickness`.
- If the series is a multi-frame object, spacing may be in
  `PerFrameFunctionalGroupsSequence` rather than at the top level.

**Edge case:** `SliceThickness` may describe the reconstructed slice width while the table moved
by a different amount, producing overlapping or gapped slices. The position delta is the
geometric truth for stacking; the thickness is the partial-volume truth for appearance. Record
both.

### Step 6. Characterize noise, artifacts, and reconstruction

**Input:** Metadata, phantom records if available, and image statistics.

**Output:** A noise and artifact profile: kVp, mAs, kernel, iterative reconstruction strength,
tracer and uptake time, stain batch, scanner model.

**Rationale:** Noise is not random across sites. It is a function of dose, kernel, and
reconstruction. A model trained on low-dose, smooth-kernel images may interpret noise texture as
signal when deployed on high-dose, sharp-kernel images. The same applies to PET reconstruction
(ordered-subset expectation maximization vs point-spread-function modelling) and pathology
scanner optics.

**Branch points:**
- If kernel varies within a site, kernel is a stratification variable, not a nuisance covariate.
- If iterative reconstruction strength changed mid-study, the change date is a drift boundary.
- If the tracer or uptake time varies, SUV is not comparable without harmonization.

**Edge case:** A scanner software upgrade can change noise and contrast without changing the
model name. `SoftwareVersions` and `StationName` are needed to detect it.

### Step 7. Verify contrast timing and phase

**Input:** `ContrastBolusAgent`, `ContrastBolusRoute`, `AcquisitionTime`, `SeriesDescription`,
and the study protocol.

**Output:** A phase label for each series: unenhanced, arterial, portal venous, delayed, or
unknown.

**Rationale:** Lesion contrast changes with phase. A liver lesion may be hypoattenuating on
unenhanced CT, hyperenhancing on arterial phase, and washout on portal venous phase. Pooling
phases teaches the model the phase and the disease together. A model trained on portal venous
CT may fail on unenhanced CT even when the anatomy is identical.

**Branch points:**
- If the phase is unknown, either exclude the series or treat phase as a stratification variable.
- If the protocol uses a fixed delay rather than bolus tracking, the phase label is weaker.
- If contrast was administered but not recorded, the metadata cannot be trusted.

**Edge case:** `ContrastBolusAgent` may be present for a study in which the relevant series was
acquired before injection. Always check `AcquisitionTime` relative to the injection record when
available.

### Step 8. Verify pixel encoding and photometric interpretation

**Input:** `BitsStored`, `BitsAllocated`, `PixelRepresentation`, `PhotometricInterpretation`,
`RescaleSlope`, `RescaleIntercept`, `SamplesPerPixel`, `NumberOfFrames`.

**Output:** A decoded array with a documented dtype, shape, and value range, plus a note on
whether the values are signed.

**Rationale:** Signed vs unsigned decoding changes the interpretation of negative HU. A 12-bit
signed CT stored in 16 bits with an intercept of -1024 can produce values that look like 0-4095
if decoded as unsigned. MONOCHROME1 inverts display and can invert a model's contrast if the
pipeline renders to 8-bit images before training.

**Branch points:**
- `PixelRepresentation` 0 = unsigned, 1 = two's complement signed.
- `PhotometricInterpretation` MONOCHROME2 is the normal greyscale convention; MONOCHROME1
  requires inversion for display.
- RGB or YBR_FULL data need colour conversion before grayscale models.

**Edge case:** A multiframe object may store frames with different rescale values per frame. The
top-level rescale is not sufficient; read the per-frame functional groups.

### Step 9. Verify spatial orientation and laterality

**Input:** `ImageOrientationPatient` (six values: row and column direction cosines),
`ImagePositionPatient`, `PatientPosition`, `ImageLaterality`, `Laterality`.

**Output:** An orientation matrix mapping voxel indices to patient coordinates, and a laterality
check against the report and any burned-in marker.

**Rationale:** Left/right errors are clinically serious and are also a shortcut source. A model
that learns the "L" marker can achieve high accuracy on a dataset where laterality correlates
with the label, then fail when the marker is cropped. The orientation matrix is the only reliable
way to compare voxel coordinates across series.

**Branch points:**
- If the row and column direction cosines are not orthonormal, the acquisition is oblique or the
  metadata are corrupt; recompute the normal and check the determinant.
- If the determinant is negative, the coordinate system is left-handed and the volume may be
  mirrored.
- If `ImageLaterality` disagrees with the report, exclude the series or escalate to a human.

**Edge case:** Burned-in annotations are pixels, not metadata. They survive de-identification
unless the pixels are cleaned. OCR or a visual audit is required.

### Step 10. Assess label visibility and reference standard

**Input:** The label source and the imaging modality.

**Output:** A statement of whether the label is visible on the image, inferred from a report, or
confirmed by another test, and the expected label noise.

**Rationale:** A label derived from a report is a text inference, not an image measurement. A
label derived from pathology is a different-modality reference. A label derived from follow-up
is a longitudinal outcome. The model can only learn what the image shows; if the label depends on
information not in the image, performance has a ceiling and the ceiling is not the model's fault.

**Branch points:**
- If the label is report-derived, quantify agreement with a human image review on a sample.
- If the label is pathology-confirmed but the image is CT, the model is predicting a
  cross-modality inference; report it as such.
- If the label is an outcome (e.g., stroke within 90 days), the image may be only one predictor.

**Edge case:** A report may mention a finding that is not present, or omit a finding that is
present. The first inflates false positives; the second inflates false negatives. Neither is
detectable without a sample of image-level review.

### Step 11. Stratify the dataset by modality and protocol

**Input:** The metadata table for all candidate studies.

**Output:** A stratification table with counts by modality, body part, protocol, scanner,
kernel, slice thickness, contrast phase, and site.

**Rationale:** Aggregate performance hides stratum-specific failure. A model with 0.90 AUROC
overall may have 0.70 on the 5 mm slice stratum and 0.93 on the 1 mm stratum. If the deployment
site uses 5 mm slices, the aggregate number is misleading.

**Branch points:**
- If a stratum has fewer than about 30 positive cases, report it as descriptive, not as an
  estimate with a confidence interval.
- If a stratum is confounded with site, site-held-out validation is required.
- If a protocol is present in training but absent in test, the model's performance on that
  protocol is unknown.

**Edge case:** Stratification by `SeriesDescription` is unreliable because descriptions are
free text. Prefer coded attributes and protocol mapping tables.

### Step 12. Define inclusion and exclusion criteria and version them

**Input:** Steps 1-11.

**Output:** A data specification document with inclusion criteria, exclusion criteria, unit
conversions, and a version identifier.

**Rationale:** The specification is the contract between data engineering, clinical, and
regulatory. Without it, the dataset silently changes when a new site is added. The version
identifier must appear in model cards, validation reports, and monitoring dashboards.

**Branch points:**
- Exclude series with missing spacing, missing rescale for CT, unknown contrast phase, or
  unreadable pixel data.
- Include but flag series with burned-in annotations, laterality mismatch, or private tags.
- Freeze the specification before training; changes after training require a new version and a
  new evaluation.

**Edge case:** A new scanner at an existing site can add a protocol that the specification did
not anticipate. The correct response is a protocol amendment with a new version, not a silent
inclusion.

### 5.1 Decision table: what to do when a modality property is missing

| Missing or inconsistent property | More than two outcomes | Action |
|---|---|---|
| Pixel spacing absent | (a) DICOM DX with implied spacing, (b) ultrasound without calibration, (c) corrupt CT | (a) use the modality-specific implied value and flag, (b) exclude from geometric tasks, (c) exclude |
| Slice thickness absent | (a) single-slice series, (b) multiframe with per-frame spacing, (c) corrupt | (a) treat as 2D, (b) read per-frame groups, (c) exclude |
| Rescale absent | (a) CT with known intercept from vendor, (b) CT with unknown unit, (c) MR/other | (a) apply documented constant and flag, (b) exclude, (c) not applicable |
| Contrast phase unknown | (a) unenhanced by protocol, (b) post-contrast timing unknown, (c) mixed | (a) label unenhanced, (b) exclude or stratify, (c) split series |
| Laterality mismatch | (a) metadata error, (b) marker error, (c) true situs inversus | (a) correct from report, (b) exclude, (c) clinical review |
| Multi-frame object | (a) enhanced CT/MR, (b) WSI, (c) ultrasound cine | (a) parse functional groups, (b) pyramid/tiles, (c) frame timing |

### 5.2 Edge cases that require a written rule

**Mixed slice thickness within one series.** Some reconstructions combine thin and thick slices
in one folder. The series may have a single `SeriesInstanceUID` but non-uniform position deltas.
Rule: compute the position delta distribution; if the coefficient of variation exceeds 5%,
exclude the series or split it into sub-series by delta.

**Missing series.** A study may contain the localizer but not the diagnostic series, or the
venous phase but not the arterial phase. Rule: the inclusion criterion names the required series
by protocol, not by study. A study with a missing required series is incomplete, not negative.

**Inconsistent spacing.** Position deltas may alternate because of gantry tilt or table motion.
Rule: use the median delta for geometry and flag the series if the interquartile range exceeds
0.1 mm.

**Rescale slope/intercept absent.** Covered in Step 4. The rule is: no unit, no inclusion.

**Burned-in annotation.** The pixels contain text, arrows, or measurement calipers. Rule: run OCR
on a sample, crop or inpaint only if the annotation is not diagnostic, and never crop a
laterality marker without verifying that the label does not depend on it.

**Laterality mismatch.** Covered in Step 9. The rule is: a mismatch is a data-quality event, not a
modelling challenge.

**Multi-frame objects.** A single object may contain 300 frames, each with its own position and
orientation. Rule: parse `PerFrameFunctionalGroupsSequence`; do not assume frame order equals
slice order.

**Private tags carrying labels.** A private tag may contain the report impression, the nodule
size, or the indication. Rule: private tags are removed in de-identification, and any tag that
survives must be tested for label leakage.

## 6. Worked example

### 6.1 The case

A 62-year-old former smoker has a 9.4 mm solid pulmonary nodule found on a screening
low-dose CT. The study is a single series: 512 × 512 pixels, `PixelSpacing` 0.703125 mm in both
directions, 180 slices, `SliceThickness` 1.25 mm, `ImagePositionPatient` z from -112.5 mm to
111.25 mm in 1.25 mm steps, `RescaleSlope` 1, `RescaleIntercept` -1024. A PET/CT is performed
for characterization. The clinical question is whether the nodule is metabolically active enough
to warrant biopsy.

This is a detection and characterization case, not a segmentation benchmark. The arithmetic
below is the arithmetic a data scientist must reproduce before trusting any model output on the
case.

### 6.2 Voxel geometry and volume

In-plane pixel area:

```text
0.703125 mm × 0.703125 mm = 0.494384765625 mm²
```

Voxel volume at 1.25 mm slice spacing:

```text
0.494384765625 mm² × 1.25 mm = 0.61798095703125 mm³
```

The nodule is measured by the radiologist on thin-section images with three orthogonal
diameters: 9.4 mm, 8.1 mm, and 7.6 mm. Treating it as an ellipsoid, the semi-axes are 4.7 mm,
4.05 mm, and 3.8 mm:

```text
V = (4/3) × π × 4.7 × 4.05 × 3.8
  = 4.18879 × 72.333
  = 302.99 mm³
```

The number of voxels covered at the acquired spacing is:

```text
302.99 mm³ / 0.61798 mm³ per voxel = 490.3 voxels
```

A segmentation model that outputs a binary mask with 490 voxels is consistent with the
radiologist's calliper measurement, within the error of the ellipsoid assumption. A mask with
2,000 voxels implies a volume of 1,236 mm³, which is a different lesion or a segmentation failure.

### 6.3 Resampling to 1 mm isotropic spacing

The field of view is:

```text
512 × 0.703125 mm = 360.0 mm
```

At 1 mm isotropic spacing, the in-plane matrix becomes 360 × 360. The through-plane extent is:

```text
180 × 1.25 mm = 225.0 mm nominal; the actual position span is 223.75 mm
```

so the resampled volume is 360 × 360 × 224 = 29,030,400 voxels. The original volume is
512 × 512 × 180 = 47,185,920 voxels. Resampling to 1 mm reduces the voxel count by 38.5% while
preserving the physical volume. The nodule occupies about 303 voxels at 1 mm³ each.

If the pipeline instead resamples to 0.5 mm isotropic spacing, the matrix becomes 720 × 720 × 450
= 233,280,000 voxels, an eightfold increase over 1 mm. The added voxels are interpolated, not
measured. The Nyquist limit of the original acquisition is about 1.4 mm in-plane (two pixels per
cycle at 0.7 mm), so 0.5 mm sampling cannot create new detail.

### 6.4 Hounsfield units and windowing

The nodule region of interest (ROI) has a mean stored value of 1062 and a standard deviation of
22 stored units. With slope 1 and intercept -1024:

```text
HU = stored × slope + intercept
   = 1062 × 1 + (-1024)
   = 38 HU
```

The ROI mean is 38 HU, consistent with soft tissue. The lung window used for display is centre
-600 HU and width 1500 HU:

```text
lower = -600 - 1500/2 = -1350 HU
upper = -600 + 1500/2 = 150 HU
```

The nodule at 38 HU maps to:

```text
(38 - (-1350)) / 1500 = 1388 / 1500 = 0.9253
```

which is 236 of 255 display levels, near white. On the soft-tissue window, centre 40 HU and width
400 HU:

```text
lower = 40 - 200 = -160 HU
upper = 40 + 200 = 240 HU
(38 - (-160)) / 400 = 198 / 400 = 0.495
```

which is 126 of 255, mid-grey. The same nodule is near-white on lung window and mid-grey on soft
tissue window. A model trained on 8-bit lung-window images sees a different input than a model
trained on 8-bit soft-tissue-window images, even though both derive from the same HU array. The
window must be part of the preprocessing specification.

### 6.5 PET quantification

The PET ROI has an activity concentration of 25,400 Bq/mL. The injected dose is 370 MBq and the
patient weighs 70 kg. Body-weight SUV:

```text
SUV_bw = activity concentration / (injected dose / body weight)
       = 25,400 Bq/mL / (370,000,000 Bq / 70,000 g)
       = 25,400 / 5,285.7 Bq/g
       = 4.81
```

The lean body mass is estimated at 52 kg. SUL:

```text
SUL = 25,400 / (370,000,000 / 52,000)
    = 25,400 / 7,115.4
    = 3.57
```

The liver SUL mean is 1.9 with a standard deviation of 0.2. The PERCIST measurability threshold is
1.5 × liver mean + 2 × liver SD:

```text
1.5 × 1.9 + 2 × 0.2 = 2.85 + 0.40 = 3.25
```

The lesion SULpeak of 3.57 is above 3.25, so it is measurable by PERCIST criteria. The SUV_bw of
4.81 and the SUL of 3.57 are both true statements about the same lesion under different
normalizations; they are not interchangeable.

### 6.6 Detection performance at a stated operating point

Suppose the CT detection model was evaluated on 400 nodules and 400 negative scans, with a
reference standard of three-radiologist consensus on thin-section CT. At a threshold chosen to
yield 1 false positive per scan, the model detected 376 of 400 nodules:

```text
sensitivity = 376 / 400 = 0.94 (94%)
false positives = 1 per scan
```

On chest radiographs from the same patients, a different model at 1 false positive per image
detected 248 of 400 nodules:

```text
sensitivity = 248 / 400 = 0.62 (62%)
```

The difference is a modality effect, not necessarily a model-quality effect. The radiograph
cannot show most 9 mm nodules because of superimposition. Reporting the two numbers without the
modality and the reference standard would be misleading.

### 6.7 Segmentation agreement

The reference mask and the model mask are compared voxel by voxel at 1 mm isotropic spacing.
Reference volume 303 voxels, prediction 271 voxels, intersection 236 voxels:

```text
Dice = 2 × 236 / (303 + 271)
     = 472 / 574
     = 0.822
```

Dice 0.822 is a voxel-level metric on a single lesion. It says nothing about whether the model
would find a second nodule, and it is sensitive to small-lesion boundary error: a one-voxel
erosion of a 303-voxel object removes roughly 10% of the volume.

### 6.8 Two diagnostics that reveal a broken pipeline

**Diagnostic 1: site signature in the CT values.** The training set contains scans from two
sites. Site A uses a smooth kernel (B30f) and site B a sharp kernel (B70f). The mean HU in a
fixed 20 mm circular ROI in the descending aorta is 42.1 HU at site A and 39.8 HU at site B, and
the noise standard deviation is 9.2 HU vs 18.7 HU. A classifier trained to predict site from
these two features reaches AUC 0.89 (unit: scan; reference: site label). If the disease prevalence
differs by site, the model can learn site instead of disease. The pipeline is broken not because
the model is inaccurate but because the input distribution encodes the site.

**Diagnostic 2: label leak in the series description.** The `SeriesDescription` field contains
"NODULE FOLLOWUP" for 71% of positive cases and "SCREENING" for 84% of negatives. A text-only
model using the description reaches AUC 0.78 (unit: series; reference: nodule label). The image
model's performance is therefore not evidence that the image contains the signal. The pipeline
must strip or ignore the description at inference and re-evaluate.

### 6.9 Varying one assumption

Assume the acquisition used 2.5 mm slices instead of 1.25 mm, with the same in-plane spacing and
the same physical nodule. Through-plane partial volume reduces the measured long axis from 7.6 mm
to about 7.0 mm and the measured mean HU from 38 to 31. The ellipsoid volume becomes:

```text
V = (4/3) × π × 4.7 × 4.05 × 3.5 = 279.0 mm³
```

a 7.9% reduction from 302.99 mm³. The PET partial-volume recovery at 2.5 mm is lower; assume the
measured SULpeak falls from 3.57 to 2.95. The PERCIST threshold remains 3.25. The lesion is now
below the measurability threshold, and the conclusion changes from "metabolically active, biopsy
discussed" to "not measurable by PERCIST, reassess with thin-section imaging". The clinical
conclusion flipped because of a slice-thickness assumption, not because of a model.

The same arithmetic applies to the model: if training used 1.25 mm slices and deployment uses
2.5 mm, the model's input distribution has shifted in partial volume and noise. The correct
response is to include slice thickness in the intended use and to validate the 2.5 mm stratum
separately.

## 7. Code, query, or calculation

The following Python uses only the standard library. It reproduces the arithmetic in Section 6
and can be run without pydicom or numpy. It parses no files; it computes from stated metadata,
which is the point: the geometry must be checkable by hand.

```python
import math

# --- Acquisition metadata (from DICOM, stated here explicitly) ---
rows, cols = 512, 512
pixel_spacing = (0.703125, 0.703125)   # mm, row and column
slice_thickness = 1.25                 # mm
n_slices = 180
z_first, z_last = -112.5, 111.25       # mm, ImagePositionPatient
slope, intercept = 1.0, -1024.0        # RescaleSlope, RescaleIntercept

# 1. Voxel geometry
pixel_area = pixel_spacing[0] * pixel_spacing[1]
voxel_volume = pixel_area * slice_thickness
fov_x = cols * pixel_spacing[1]
fov_y = rows * pixel_spacing[0]
z_span = z_last - z_first
z_step = z_span / (n_slices - 1)

print(f"pixel area          = {pixel_area:.12f} mm^2")
print(f"voxel volume        = {voxel_volume:.12f} mm^3")
print(f"field of view       = {fov_x:.1f} x {fov_y:.1f} mm")
print(f"z step from IPP     = {z_step:.6f} mm")
print(f"z extent            = {z_span:.1f} mm")

# 2. Ellipsoid lesion volume
a, b, c = 9.4 / 2, 8.1 / 2, 7.6 / 2     # semi-axes, mm
lesion_volume = (4 / 3) * math.pi * a * b * c
voxels_covered = lesion_volume / voxel_volume
print(f"lesion volume       = {lesion_volume:.2f} mm^3")
print(f"voxels covered      = {voxels_covered:.1f}")

# 3. Resampling to 1 mm isotropic
iso = 1.0
new_cols = round(fov_x / iso)
new_rows = round(fov_y / iso)
new_slices = round(z_span / iso)
print(f"1 mm shape          = {new_rows} x {new_cols} x {new_slices}")
print(f"1 mm voxel count    = {new_rows * new_cols * new_slices}")
print(f"native voxel count  = {rows * cols * n_slices}")
print(f"voxel reduction     = {1 - (new_rows*new_cols*new_slices)/(rows*cols*n_slices):.1%}")

# 4. HU conversion and window mapping
stored_mean, stored_sd = 1062, 22
hu_mean = stored_mean * slope + intercept
hu_sd = stored_sd * slope
print(f"HU mean / SD        = {hu_mean:.0f} / {hu_sd:.0f} HU")

def window_fraction(value_hu, level, width):
    lower = level - width / 2
    upper = level + width / 2
    clipped = min(max(value_hu, lower), upper)
    return (clipped - lower) / width

lung = window_fraction(hu_mean, -600, 1500)
soft = window_fraction(hu_mean, 40, 400)
print(f"lung window         = {lung:.4f} -> {round(lung*255)}/255")
print(f"soft-tissue window  = {soft:.4f} -> {round(soft*255)}/255")

# 5. SUV and SUL
activity_bq_ml = 25400.0
injected_bq = 370e6
weight_kg = 70.0
lbm_kg = 52.0
suv_bw = activity_bq_ml / (injected_bq / (weight_kg * 1000))
sul = activity_bq_ml / (injected_bq / (lbm_kg * 1000))
liver_mean, liver_sd = 1.9, 0.2
percist_threshold = 1.5 * liver_mean + 2 * liver_sd
print(f"SUV_bw              = {suv_bw:.2f}")
print(f"SUL                 = {sul:.2f}")
print(f"PERCIST threshold   = {percist_threshold:.2f}")
print(f"measurable          = {sul > percist_threshold}")

# 6. Dice at 1 mm
ref, pred, inter = 303, 271, 236
dice = 2 * inter / (ref + pred)
print(f"Dice                = {dice:.3f}")
```

Expected output:

```text
pixel area          = 0.494384765625 mm^2
voxel volume        = 0.617980957031 mm^3
field of view       = 360.0 x 360.0 mm
z step from IPP     = 1.250000 mm
z extent            = 223.8 mm
lesion volume       = 302.99 mm^3
voxels covered      = 490.3
1 mm shape          = 360 x 360 x 224
1 mm voxel count    = 29030400
native voxel count  = 47185920
voxel reduction     = 38.5%
HU mean / SD        = 38 / 22 HU
lung window         = 0.9253 -> 236/255
soft-tissue window  = 0.4950 -> 126/255
SUV_bw              = 4.81
SUL                 = 3.57
PERCIST threshold   = 3.25
measurable          = True
Dice                = 0.822
```

Note the difference from the hand calculation: the z extent is 223.8 mm, not 225.0 mm, because
the positions run from -112.5 to 111.25 inclusive in 180 slices. The 1 mm shape is therefore
224 slices, not 225, and the voxel reduction is 38.5%, not 38.2%. This is exactly the kind of
one-slice discrepancy that a metadata check catches and a visual inspection does not. The code
is the authority; the prose number was the approximation.

The code assumes: the series is a single stack with uniform spacing; the ellipsoid model is an
approximation to the radiologist's calliper measurement; the PET values are decay-corrected and
belong to the same lesion; the Dice volumes are at 1 mm spacing. Each assumption is stated so
that a reader can challenge it.

## 8. Data, standards, and regulatory context

### 8.1 DICOM

The DICOM Standard, PS3 current edition 2026d, defines the information objects and the encoding.
PS3.3 defines modality-specific image IODs, including CT, MR, PET, ultrasound, and slide
microscopy. PS3.5 defines pixel data encoding: `BitsAllocated`, `BitsStored`, `HighBit`,
`PixelRepresentation`, and the rescale attributes. PS3.6 is the data dictionary that maps tags to
names and value representations. PS3.15 defines the security and confidentiality profiles used in
de-identification.

The standard constrains the encoding, not the clinical meaning. It does not tell you that a CT
kernel is a site signature, that a PET reconstruction must be EARL-compliant for multicentre
quantification, or that a pathology scanner's colour profile must be managed. Those are
professional and vendor conventions.

### 8.2 ACR appropriateness and RADS lexicons

The American College of Radiology (ACR) Appropriateness Criteria are evidence-based guidelines
for choosing imaging tests. The ACR reports about 280 diagnostic imaging and interventional documents,
about 4,100 clinical scenarios, and more than 1,350 variants as of the 2026 update. They are
advisory, not binding, but they define the standard of care against which an intended use is
judged.

The ACR Reporting and Data Systems (RADS) turn findings into standardized categories:

| System | Current version | Scope | What it fixes |
|---|---|---|---|
| Lung-RADS | v2022 (November 2022) | Lung cancer screening CT | Nodule size thresholds and management intervals |
| BI-RADS | v2025 manual, extending the 5th edition atlas | Breast imaging | Density, descriptors, assessment categories |
| PI-RADS | v2.1 (2019) | Prostate MRI | Sequence requirements and lesion scores |
| LI-RADS | CT/MRI v2018 core, lexicon 2021 | Liver imaging | Diagnostic algorithm and terminology |

A model's label definition should align with the RADS category it claims to predict. "Nodule
present" is not a Lung-RADS label; "solid nodule 6 to <8 mm, category 3" is. The difference
matters because the management interval, and therefore the clinical value, depends on the
category.

### 8.3 FDA

Imaging AI is generally a medical device in the United States. The FDA's device list recorded
1,614 AI-enabled devices authorized through 29 June 2026, of which about 1,230 (76%) are
radiology devices; the agency's overview page states more than 1,600 as of September 2026. These
counts are from the FDA list and an industry tabulation, and they change quarterly.

Two regulatory documents matter for modality claims:

- The final rule codified at 21 CFR 892.2090 (June 2025) classifies radiological
  computer-assisted detection and diagnosis software as Class II with special controls. The
  special controls require a description of inputs and outputs, pre-specified performance
  testing, standalone and reader-study evidence where applicable, subgroup analyses, and
  labeling that states the compatible imaging hardware and protocols.
- The FDA guidance on technical performance assessment of digital pathology whole slide imaging
  devices (April 2016) specifies component-level and system-level tests for WSI, including
  spatial resolution, colour reproducibility, focusing, stitching, and tissue coverage.

Neither document approves a modality in general. They constrain the evidence a specific device
must provide for a specific intended use.

### 8.4 Other standards and registries

| Standard or body | What it constrains | What it does not |
|---|---|---|
| IEC 60601-2-44 | CT dose and safety | Image quality or AI performance |
| NEMA NU 2 | PET performance measurement | Clinical SUV harmonization |
| EANM/EARL | PET reconstruction harmonization | Regulatory clearance |
| DICOM PS3.18 | DICOMweb services | Workflow semantics |
| IHE Radiology profiles | Integration transactions | Model accuracy |
| ISO 12233 | Spatial resolution measurement for WSI | Diagnostic equivalence |

### 8.5 What the standards do not cover

No standard defines a "correct" normalization for MRI, a universal stain normalization for
pathology, or a minimum slice thickness for a lung nodule model. Those are product decisions that
must be documented and validated. A conformance statement that says "DICOM compatible" says
nothing about whether the pixel values are comparable across sites.

## 9. Industry practice and case evidence

### 9.1 Chest radiography: multi-label, multi-finding

CheXpert (Irvin et al., AAAI 2019) released 224,316 chest radiographs with 14 observations
labelled positive, negative, or uncertain using a rule-based report labeler. The dataset made
multi-label chest X-ray classification a standard benchmark and also exposed the label-noise
problem: report-derived labels are not image ground truth. The CheXpert labeler's mention
extraction F1 was 0.969 micro-average and 0.948 macro-average on a 1,000-report evaluation set,
with uncertainty detection F1 0.848 micro and 0.770 macro. Those are text-level metrics, not
image-level accuracy.

The industry lesson is that per-finding performance must be reported. A model with a good average
AUROC across 14 findings can be unusable on the two findings that matter clinically. CheXpert's
own results varied substantially by finding, and later work showed that report labels and image
labels disagree on a meaningful fraction of cases.

### 9.2 Mammography: screening is not a single accuracy number

McKinney et al. (Nature, 2020) reported an ensemble evaluated on UK and US screening datasets,
with a claimed reduction in false positives and false negatives relative to readers. The study is
a landmark and also a caution: it used a curated retrospective sample, and the endpoint was
reader-level performance on a enriched test set, not interval cancer rate in a population. Later
screening-program evaluations produced mixed results, with some showing improved detection and
others showing increased recall.

The modality lesson is that mammography has high spatial resolution and a standardized
acquisition, which makes detection feasible, but screening benefit depends on prevalence,
recall rate, interval cancers, and overdiagnosis. Those are population metrics, not image
metrics.

### 9.3 Digital pathology: the slide is a multi-resolution image

The first whole slide imaging system was authorized through the De Novo pathway in 2017, and the
FDA's 2016 technical performance guidance defines the tests a WSI system must pass. A WSI is a
pyramid: a 40x scan at 0.25 µm/pixel of a 15 mm × 15 mm tissue section is 60,000 × 60,000 pixels
per level, or 3.6 billion pixels at the highest level. A model rarely consumes the full pyramid;
it consumes tiles at a chosen magnification, and the magnification is part of the model's
definition.

Practical consequence: a model trained on 20x tiles (0.5 µm/pixel) and deployed on 40x tiles
(0.25 µm/pixel) sees structures at twice the pixel size. Unless the deployment pipeline resamples
to the training resolution, the model's effective receptive field halves in physical units.

### 9.4 PET: quantification is protocol-dependent

PERCIST and the EANM/EARL harmonization programme define how to measure metabolic response. The
key industry practice is that a PET SUV is comparable only within a tracer, uptake time,
reconstruction, and normalization. A multicentre trial that pools SUVmax without harmonization
can shift patients across response thresholds. The practical fix is a qualification phantom and a
single reconstruction for quantification, separate from the reconstruction optimized for lesion
detection.

### 9.5 Autonomous diagnosis outside radiology

IDx-DR (De Novo DEN180001, January 2018) was the first FDA-authorized autonomous AI diagnostic
system. Its pivotal study enrolled 900 patients at 10 primary care sites; 819 were analyzable.
Observed sensitivity for more-than-mild diabetic retinopathy was 87.4% (173/198, 95% CI
81.9-92.9%), observed specificity 89.5% (556/621, 95% CI 86.9-93.1%), and imageability 96.1%
(819/852). The reference standard was Wisconsin Fundus Photograph Reading Center grading of
widefield stereoscopic photographs plus optical coherence tomography.

The case shows that a narrow modality, a fixed capture device, a prespecified operating threshold,
and a workflow that routes ungradable images to a human can support an autonomous claim. It does
not show that the same approach transfers to CT or MRI, where acquisition variability is larger.

### 9.6 What is inference, not documented fact

The following statements are inferences from the evidence, not published findings: that a
particular hospital's CT kernel distribution will shift after a scanner upgrade; that a specific
commercial model's performance will fall by a specific amount on 5 mm slices; that a foundation
model trained on many modalities will generalize to a new modality without fine-tuning. Each is
plausible and each requires local validation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Mixed units in one tensor | HU range overlaps MRI arbitrary range | Concatenating modalities without scaling | Model learns modality, not disease | Per-modality normalization and separate input channels |
| Ignored rescale intercept | CT values cluster near 0-4095 | Stored integers treated as HU | Air appears as soft tissue; window wrong | Assert slope/intercept present; unit test |
| Slice thickness treated as spacing | z-spacing differs from thickness | Overlapping or gapped slices | Volume bias; registration error | Compute spacing from ImagePositionPatient |
| Anisotropic 3D kernel | Through-plane blur | 3×3×3 kernel covers 15 mm in z | Missed small lesions | Resample isotropic or use anisotropic kernels |
| Contrast phase pooled | Model fails on unenhanced scans | Phase correlated with indication | Sensitivity drop at deployment | Stratify by phase; restrict intended use |
| Kernel as site signature | Site classifier AUC high | Texture differs by reconstruction | Shortcut learning | Multi-kernel training; kernel stratification |
| MONOCHROME1 inversion | Lesions appear dark | Display inversion applied or not | Inverted contrast | Check PhotometricInterpretation; normalize |
| Laterality marker leak | Model relies on "L"/"R" | Marker correlates with label | Fails when marker cropped | Crop/erase markers; test without |
| Burned-in annotation | OCR finds text in pixels | Annotation encodes finding | Label leak | Pixel cleaning; manual audit |
| Partial volume ignored | Small lesions underestimated | Voxel averages lesion and background | Size and SUV bias | Thin-section protocol; recovery correction |
| Multi-frame misread | Frame count mismatch | Frames treated as files | Missing slices or wrong order | Parse per-frame functional groups |
| Scanner upgrade undetected | Drift in noise and contrast | Software change without metadata change | Silent performance loss | Monitor SoftwareVersions and image statistics |
| Pathology magnification mismatch | Tile scale differs | 20x vs 40x deployment | Receptive field halved | Resample to training µm/pixel |
| PET protocol drift | SUV shifts without disease change | Reconstruction or uptake time change | Response misclassification | EARL qualification; protocol lock |

### 10.1 The three red flags that should stop a project

1. **No unit statement.** If the data specification cannot say whether the model input is HU,
   SUV, normalized intensity, or 8-bit display value, the model is not reproducible.
2. **No protocol envelope.** If the intended use says "CT" without slice thickness, contrast
   phase, and kernel, the deployment population is undefined.
3. **No modality-stratified test set.** If the only test number is pooled across modalities or
   protocols, the team does not know where the model works.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The modality label is correct | Stratification and unit conversion depend on it | Mislabeled series, derived images | Cross-check SOP class UID and pixel statistics |
| Rescale maps stored values to HU | CT windowing and thresholding | Missing or wrong tags | Water/air ROI check; phantom |
| Slice spacing is uniform | Volume and resampling arithmetic | Gantry tilt, mixed reconstruction | Position-delta distribution |
| Pixel spacing is calibrated | Size and volume metrics | Ultrasound, some DX | Compare to known marker or phantom |
| The finding is visible on the modality | Task feasibility | Resolution or contrast below threshold | Reader study on sample |
| The reference standard matches the modality | Label validity | Report label from another test | Image-level review sample |
| Contrast phase is recorded | Avoiding phase confounding | Missing injection record | Protocol audit |
| Scanner metadata are stable | Drift detection | Software upgrade without tag change | Monitor image statistics |
| The training protocol covers deployment | Generalization | New site or protocol | Site- and protocol-held-out validation |
| Normalization is identical at training and inference | Input distribution match | Pipeline version mismatch | Hash and version the transform |

### 11.1 Limits of this topic

This topic does not teach image interpretation, acquisition physics at the level needed to
operate a scanner, or regulatory strategy. It gives the data scientist enough vocabulary to
specify data, detect mismatch, and ask the right expert. It does not replace a medical physicist
for protocol questions or a radiologist for label definition.

## 12. Exercises

### Exercise 1. Unit conversion audit (recall, 20 minutes)

**Deliverable:** A one-page table for five CT series showing stored value range, slope, intercept,
computed HU range, and whether the series is usable for HU-based analysis.

**Acceptance criteria:** All five rows have units; any missing intercept is flagged; at least one
series is correctly excluded.

**Time:** 20 minutes.

### Exercise 2. Modality selection memo (application, 45 minutes)

**Deliverable:** A 400-word memo choosing a modality and protocol for a stated task: detect
6 mm solid lung nodules in a screening population.

**Acceptance criteria:** Names the contrast mechanism, the resolution requirement, the protocol
envelope, one alternative and why it is weaker, and the reference standard.

**Time:** 45 minutes.

### Exercise 3. Resolution arithmetic (calculation, 30 minutes)

**Deliverable:** For a CT with 0.68 mm in-plane spacing, 2.5 mm slices, 400 slices, compute voxel
volume, anisotropy ratio, isotropic 1 mm matrix, and the number of interpolated voxels added.

**Acceptance criteria:** Correct units; correct rounding; a statement of whether 1 mm resampling
adds information.

**Time:** 30 minutes.

### Exercise 4. Window mapping (calculation, 20 minutes)

**Deliverable:** Map a lesion at -450 HU, 35 HU, and 180 HU through lung, soft-tissue, and bone
windows, and state the 8-bit value for each.

**Acceptance criteria:** Correct lower and upper bounds; correct clipping at the ends; units
stated.

**Time:** 20 minutes.

### Exercise 5. PET normalization critique (critique, 40 minutes)

**Deliverable:** A 500-word critique of a paper that reports SUVmax without stating tracer,
uptake time, reconstruction, or normalization.

**Acceptance criteria:** Identifies at least four threats to comparability; proposes what the
authors should have reported; distinguishes SUV from SUL.

**Time:** 40 minutes.

### Exercise 6. Site-signature diagnostic (application, 60 minutes)

**Deliverable:** A script or spreadsheet that computes two image statistics per series (mean and
SD in a fixed ROI) and tests whether they separate sites.

**Acceptance criteria:** Reports AUC or a separability measure with unit (series) and reference
(site); states the confound with disease prevalence.

**Time:** 60 minutes.

### Exercise 7. Data specification (design, 60 minutes)

**Deliverable:** A two-page inclusion/exclusion specification for a CT nodule model, including
unit conversion, protocol envelope, and edge-case rules.

**Acceptance criteria:** Every rule is testable; no rule depends on free-text
`SeriesDescription`; version identifier present.

**Time:** 60 minutes.

### Exercise 8. Cross-modality claim review (critique, 45 minutes)

**Deliverable:** A review of a vendor claim that a model "works across CT and MRI" with a list of
five questions and the evidence each would require.

**Acceptance criteria:** Questions cover units, resolution, contrast, reference standard, and
site validation; at least one question exposes an unfalsifiable claim.

**Time:** 45 minutes.

## 13. Self-check answers

**A1. A CT series has slope 1 and intercept -1024. A stored value of 0, 1024, and 2047 maps to
what HU?**
0 → -1024 HU (air); 1024 → 0 HU (water); 2047 → 1023 HU (dense bone or metal). The arithmetic is
`stored × 1 + (-1024)`. The negative value is valid; unsigned decoding would turn it into 64512
and destroy the scale.

**A2. Why is a 9 mm nodule detected at 94% on CT but 62% on radiograph?**
Contrast and superimposition. CT resolves the nodule in three dimensions against air; the
radiograph projects ribs, mediastinum, and diaphragm over the nodule. The difference is a
modality effect, and the two sensitivities are not comparable without stating modality and
reference standard.

**A3. A volume has 0.7 mm in-plane spacing and 5 mm slices. What is the anisotropy ratio, and
what does it imply for a 3×3×3 kernel?**
Ratio 5 / 0.7 = 7.1. A 3×3×3 kernel covers 2.1 mm in-plane and 15 mm through-plane. It is not a
cube in anatomy; either resample to isotropic spacing or use a kernel sized per axis.

**A4. Convert 25,400 Bq/mL to SUV with 370 MBq injected and 70 kg body weight.**
370 MBq / 70 kg = 5.2857 MBq/kg = 5,285.7 Bq/g. 25,400 / 5,285.7 = 4.81. Unit: dimensionless
(g/mL). The result depends on the body-weight normalization; SUL with 52 kg lean mass is 3.57.

**A5. Why is the PERCIST threshold 1.5 × liver mean + 2 × liver SD rather than a fixed SUV?**
Because PET uptake scales differ by scanner, reconstruction, and patient. The liver is an
internal reference that absorbs global scale differences. With liver mean 1.9 and SD 0.2, the
threshold is 3.25, and the same lesion can be above or below it depending on the reconstruction.

**A6. What is wrong with resampling a 0.7 mm CT to 0.35 mm?**
Nothing is added. The acquisition's Nyquist limit is about 1.4 mm in-plane (two samples per
cycle at 0.7 mm), so 0.35 mm sampling oversamples an already band-limited image. It increases
memory and compute eightfold relative to 0.7 mm and can introduce interpolation artifacts.

**A7. A model trained on 1.25 mm slices is deployed on 2.5 mm slices. Name three mechanisms by
which performance can fall.**
Partial volume reduces lesion contrast and measured size; noise correlation changes because
adjacent slices are more similar; the model's learned through-plane texture is at a different
scale. The correct test is a 2.5 mm held-out stratum, not a visual review.

**A8. Why is "DICOM compatible" not a modality validation?**
DICOM defines encoding, not clinical meaning. A file can be DICOM-compatible and still have
missing rescale, wrong spacing, MONOCHROME1 inversion, or a different kernel. Compatibility is a
syntax claim, not a measurement claim.

**A9. Design rubric for Exercise 7.** A strong specification names the modality, body part,
protocol envelope, unit conversion, spacing source, contrast phase, slice-thickness range,
exclusion rules for missing metadata, and a version identifier. A weak specification says "CT
chest, exclude bad images" and relies on `SeriesDescription`.

**A10. Design rubric for Exercise 8.** A strong review asks for per-modality performance with
confidence intervals, the unit and reference standard, the acquisition envelope, the site-held-out
result, and the failure modes. A weak review accepts a pooled AUROC and a saliency figure.

## 14. Interview questions

**Q1. What does a CT pixel value represent before and after rescale?**
A strong answer: before rescale it is a stored integer whose width and signedness are defined by
`BitsStored` and `PixelRepresentation`; after `stored × slope + intercept` it is HU, a calibrated
attenuation scale with water at 0 and air near -1000. A weak answer says "density" and ignores
the intercept.

**Q2. Why can't you concatenate CT and MRI into one channel?**
A strong answer: the units differ (HU vs arbitrary), the contrast mechanisms differ, and the
resolution and noise differ; concatenation teaches the model modality. A weak answer says
"normalize both to 0-1 and it is fine", which discards the calibrated information and still
leaves resolution and contrast differences.

**Q3. How do you decide whether a finding is visible on a modality?**
A strong answer: compare the lesion-background contrast to the noise at the available resolution,
check the smallest object against the pixel and slice size, and confirm with a reader study on a
sample. A weak answer relies on the dataset label or a single example.

**Q4. What is partial volume and why does it matter for small lesions?**
A strong answer: a voxel averages tissues within its volume; when the lesion is smaller than the
slice thickness, its measured HU or SUV is diluted by background, so size and uptake are
underestimated. A weak answer treats it as blur and does not connect it to quantification.

**Q5. Why is PET SUV not comparable across sites by default?**
A strong answer: it depends on injected dose, uptake time, reconstruction, scanner calibration,
and normalization (body weight vs lean mass); harmonization requires phantom qualification and a
fixed reconstruction. A weak answer says "SUV is standardized" and assumes the name guarantees
comparability.

**Q6. How would you detect a scanner kernel signature in a dataset?**
A strong answer: stratify image statistics by scanner and kernel, train a site classifier on
patches or metadata, and test whether performance falls when the signature is removed or sites
are held out. A weak answer looks at saliency maps and calls them explanations.

**Q7. What is the difference between detection, segmentation, triage, and diagnosis here?**
A strong answer: detection outputs a location or presence; segmentation outputs a voxel mask;
triage outputs a priority for review; diagnosis outputs a disease state against a reference
standard. Each has a different metric and regulatory claim. A weak answer uses "detect" for all
four.

**Q8. When is 2D analysis appropriate for a 3D modality?**
A strong answer: when the finding is defined on a single slice and through-plane context is not
needed, or when data are too anisotropic for 3D; the choice must be validated against a 3D
baseline. A weak answer says 2D is always cheaper and therefore better.

**Q9. What would make you exclude a CT series from a nodule model?**
A strong answer: missing rescale, non-uniform spacing, missing required phase, unreadable pixel
data, or a slice thickness outside the intended use. A weak answer excludes only corrupt files.

**Q10. How do you report a modality-stratified result?**
A strong answer: a table with modality, protocol, n patients, n lesions, metric with unit,
reference standard, and confidence interval per stratum, plus the pooled number labelled as
pooled. A weak answer reports one aggregate AUROC and a subgroup footnote without counts.

## 15. Key terms

- **Attenuation** — reduction in X-ray intensity per unit path; CT's measured property.
- **Anisotropy** — unequal voxel spacing across axes; drives resampling choices.
- **BI-RADS** — ACR breast imaging reporting system; current manual v2025.
- **Bq/mL** — activity concentration unit used by PET.
- **CNR** — contrast-to-noise ratio; lesion-background difference over noise.
- **Contrast phase** — timing of contrast acquisition; a major confounder.
- **DICOM** — Digital Imaging and Communications in Medicine; the encoding standard.
- **Dice** — voxel overlap metric; unit is voxel, reference is a mask.
- **HU** — Hounsfield unit; calibrated CT attenuation.
- **LI-RADS** — ACR liver imaging reporting system; CT/MRI v2018 core.
- **Lung-RADS** — ACR lung screening reporting system; v2022.
- **MONOCHROME1** — photometric interpretation with inverted greyscale.
- **Multi-frame** — one DICOM object containing many frames.
- **Partial volume** — tissue mixing within a voxel.
- **PERCIST** — PET response criteria using SUL and liver reference.
- **PI-RADS** — ACR prostate imaging reporting system; v2.1.
- **Pixel spacing** — physical distance between pixel centres, mm.
- **RADS** — Reporting and Data System; standardized categories.
- **Reference standard** — the truth used to score a model.
- **Rescale intercept** — offset mapping stored values to HU.
- **Rescale slope** — scale mapping stored values to HU.
- **Slice thickness** — nominal reconstructed slice width, mm.
- **SNR** — signal-to-noise ratio.
- **SUL** — SUV normalized to lean body mass.
- **SUV** — standardized uptake value; dimensionless.
- **T1/T2** — MRI relaxation time constants, ms.
- **Voxel** — volume element; mm³.
- **Whole slide image (WSI)** — digitized pathology slide, often a pyramid.
- **Window level/width** — display mapping centre and range in HU.
- **µm/pixel** — physical pixel size in pathology.

## 16. Sources

- DICOM Standard, PS3 current edition 2026d, PS3.3 (Information Object Definitions), PS3.5
  (Data Structures and Encoding), PS3.6 (Data Dictionary), PS3.15 (Security and System
  Management Profiles). https://www.dicomstandard.org/ (accessed 25 October 2026).
- DICOM PS3.5 2026d, Chapter 3, definitions of Pixel Data, Native Format, Encapsulated Format,
  and Multi-frame. https://dicom.nema.org/medical/dicom/current/output/chtml/part05/chapter_3.html
  (accessed 25 October 2026).
- American College of Radiology, Appropriateness Criteria. https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Appropriateness-Criteria
  (accessed 25 October 2026). Program statistics as reported in the 2026 update.
- American College of Radiology, Lung-RADS v2022.
  https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/Lung-RADS
  (accessed 25 October 2026).
- American College of Radiology, BI-RADS v2025 manual and 5th edition atlas.
  https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/BI-RADS
  (accessed 25 October 2026).
- American College of Radiology, PI-RADS v2.1 (2019).
  https://cs.acr.org/Clinical-Resources/Reporting-and-Data-Systems/PI-RADS (accessed 25 October
  2026).
- American College of Radiology, LI-RADS CT/MRI v2018 core and 2021 lexicon.
  https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/LI-RADS
  (accessed 25 October 2026).
- U.S. Food and Drug Administration, "Technical Performance Assessment of Digital Pathology Whole
  Slide Imaging Devices," final guidance, April 2016. https://www.fda.gov/media/90791/download
  (accessed 25 October 2026).
- U.S. Food and Drug Administration, 21 CFR 892.2090, Radiological computer-assisted detection and
  diagnosis software; final order 90 FR 24969, 13 June 2025.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892.2090
  (accessed 25 October 2026).
- U.S. Food and Drug Administration, List of Artificial Intelligence-Enabled Medical Devices.
  https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices
  (accessed 25 October 2026). Counts through 29 June 2026.
- Irvin, J. et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert
  Comparison," AAAI 2019. https://doi.org/10.1609/aaai.v33i01.3301590
- McKinney, S. M. et al., "International evaluation of an AI system for breast cancer screening,"
  Nature 577, 89-94 (2020). https://doi.org/10.1038/s41586-019-1799-6
- Abràmoff, M. D. et al., "Pivotal trial of an autonomous AI-based diagnostic system for detection
  of diabetic retinopathy in primary care offices," npj Digital Medicine 1, 39 (2018).
  https://doi.org/10.1038/s41746-018-0040-6
- U.S. Food and Drug Administration, De Novo DEN180001 (IDx-DR), decision summary.
  https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf (accessed 25 October 2026).
- Aide, N. et al., "EANM/EARL harmonization strategies in PET quantification," European Journal
  of Nuclear Medicine and Molecular Imaging 44, 17-31 (2017).
  https://doi.org/10.1007/s00259-017-3740-2
- O, J. H., Lodge, M. A., Wahl, R. L., "Practical PERCIST," Radiology 280, 576-584 (2016).
  https://doi.org/10.1148/radiol.2016150789
- Ma, J. et al., "Segment anything in medical images," Nature Communications 15, 654 (2024).
  https://doi.org/10.1038/s41467-024-44824-z
- Isensee, F. et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation," Nature Methods 18, 203-211 (2021). https://doi.org/10.1038/s41592-020-01008-z

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_dicom-and-metadata.md`](02_dicom-and-metadata.md)
- Related topic: [`03_preprocessing.md`](03_preprocessing.md)
- Related topic: [`06_leakage-and-shortcuts.md`](06_leakage-and-shortcuts.md)
- Root survey: [`../../10_medical_imaging.md`](../../course/10_medical_imaging.md)
- Labs: [`../../labs/`](../../labs/)
- Module 07 foundations: [`../07_deep_learning/`](../07_deep_learning/)
