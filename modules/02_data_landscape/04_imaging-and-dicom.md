# Imaging and DICOM

> Module 02 — The Biomedical Data Landscape
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 is helpful; no clinical imaging background assumed
> Version 2026.3

---

## 1. Why this topic exists

Medical imaging is the largest data type in the hospital by volume and the most poorly understood by data teams. A single computed tomography (CT) study can contain 800 images and 200 megabytes; a hospital network can accumulate petabytes per year. Yet the analytic work usually begins with someone exporting a folder of files and losing the metadata that made the images comparable.

Digital Imaging and Communications in Medicine (DICOM) is the standard that defines both the pixel data and the metadata, and the metadata is where most analytic value and most privacy risk live. The pixel data tells you what the tissue looked like. The metadata tells you which scanner, which reconstruction kernel, which slice thickness, which contrast phase, and which institution produced the image. Without the metadata, two images of the same anatomy are not comparable, and a model trained on one scanner will fail on another.

The commercial stakes:

- An artificial intelligence (AI) imaging vendor that trains on one scanner's reconstructions and deploys across a health system will see performance drop, often substantially, and will not know why unless the acquisition metadata is in the training set.
- A sponsor that licenses imaging data without auditing burned-in annotations can breach a data use agreement, because the annotation text is inside the pixels and survives metadata de-identification.
- A researcher who converts DICOM to another format for convenience loses the acquisition parameters that would have explained a batch effect.
- A hospital that de-identifies headers but retains full dates and private tags has not de-identified the study in any meaningful sense.

The decision this topic supports: whether an imaging dataset can answer a clinical or AI question, what its acquisition heterogeneity implies for generalization, and what must be removed before it can be shared.

### 1.1 Who is blocked, and by what

| Role | Decision blocked | Failure without this topic |
|---|---|---|
| Imaging AI lead | Whether a model will generalize across sites | Trains on one kernel and one slice thickness; reports optimistic validation |
| Clinical data scientist | Whether an imaging cohort is comparable | Mixes contrast phases and reconstruction settings |
| Regulatory lead | Whether an imaging dataset is de-identified | Misses burned-in text and private tags |
| Research coordinator | Whether a data transfer is permitted | Assumes header de-identification is sufficient |
| PACS administrator | How to satisfy a research query | Exports without a protocol and cannot reproduce the export |
| Radiologist collaborating with data science | How to describe an image-derived label | Label depends on a reconstruction the analyst did not record |
| Procurement lead | What an imaging data license covers | Assumes pixel data and metadata have the same license terms |

### 1.2 Why the metadata is the analysis

Three examples make the point concretely:

1. **Hounsfield units.** CT pixel values are integers that must be converted with a rescale slope and intercept to Hounsfield units (HU). An image analyzed without applying the rescale is not in HU, and a threshold like "fat is below −30 HU" is meaningless.
2. **Reconstruction kernel.** A sharp kernel and a smooth kernel produce different noise texture from the same raw data. A radiomics feature computed on one is not comparable to the same feature on the other.
3. **Contrast phase.** A portal venous phase and an arterial phase show different enhancement. A liver lesion detected on one may be invisible on the other.

Each of these is recorded in the header, and each is lost when images are exported as JPEG or PNG.

---

## 2. Learning outcomes

By the end of this topic you can:

1. **Describe** the DICOM information model (patient, study, series, instance) and use unique identifiers correctly.
2. **Extract** and interpret the acquisition metadata that determines image comparability, including rescale, kernel, slice thickness, and contrast agent.
3. **Convert** stored pixel values to Hounsfield units and explain why the conversion is mandatory for quantitative analysis.
4. **Audit** a DICOM collection for de-identification defects, including burned-in annotations, private tags, and date leakage.
5. **Quantify** scanner-related measurement bias and correct it with a phantom calibration, with correct arithmetic.
6. **Build** an imaging cohort with exclusion accounting and state the comparability consequence of each exclusion.
7. **Critique** an imaging AI validation for acquisition heterogeneity and leakage, and propose the missing analyses.
8. **Design** a DICOM export protocol that a PACS administrator could execute and a reviewer could reproduce.

Mapping to assessment: outcomes 2, 3, and 5 are assessed by exercises 2, 3, and 5. Outcomes 1, 4, and 6 are assessed by the self-check items. Outcomes 7 and 8 are assessed by the design exercises and the module memo.

---

## 3. The core mental model

DICOM is a four-level hierarchy with identifiers at every level, plus a header that carries two very different kinds of information.

```text
  PATIENT  (PatientID, name, birth date, sex)
     |
     | 1..n
     v
  STUDY    (StudyInstanceUID, StudyDate, AccessionNumber, Referrer)
     |
     | 1..n
     v
  SERIES   (SeriesInstanceUID, Modality, SeriesNumber, ProtocolName,
     |      Manufacturer, ManufacturerModelName, DeviceSerialNumber,
     |      SliceThickness, PixelSpacing, ConvolutionKernel)
     |
     | 1..n
     v
  INSTANCE (SOPInstanceUID, SOPClassUID, InstanceNumber,
            RescaleSlope, RescaleIntercept, WindowCenter, WindowWidth,
            Rows, Columns, BitsAllocated, PixelData)

  Two header layers:
  +---------------------------------------------------------------+
  | ACQUISITION METADATA      | IDENTITY METADATA                 |
  | scanner, protocol, kernel | patient name, ID, dates,          |
  | thickness, contrast, dose | institution, device serial,       |
  | -> needed for analysis    | referring physician               |
  | -> must be retained       | -> must be removed or transformed |
  +---------------------------------------------------------------+
```

The key insight is the two-layer split at the bottom. The same header that makes the image analyzable is the header that identifies the patient. De-identification is therefore not a deletion pass; it is a **selective transformation** that must preserve acquisition parameters while removing identity.

In DICOM this is formalized in PS3.15 Annex E, which defines action codes per attribute: remove, zero, replace with a dummy value, clean, keep, or replace with a consistent pseudonym. A de-identification that removes too much makes the data useless; one that removes too little is a breach.

### 3.1 The three comparability axes

Two images are comparable only if they agree on three axes:

| Axis | What varies | Recorded in | Consequence of mismatch |
|---|---|---|---|
| Acquisition | Scanner, kVp, mAs, contrast agent and phase, patient positioning | Study and series header | Different tissue contrast and noise |
| Reconstruction | Kernel, slice thickness, field of view, iterative reconstruction strength | Series header | Different texture and edge sharpness |
| Post-processing | Windowing, cropping, resampling, anonymization | Derivation description, or nowhere | Irreversible loss of quantitative meaning |

The third axis is the dangerous one, because post-processing is often done outside DICOM and leaves no record. A file converted to NIfTI and resampled to 1 mm isotropic has lost the original spacing, and if the conversion script did not write a sidecar, the loss is permanent.

### 3.2 Original versus derived objects

DICOM distinguishes original images from derived objects: segmentations, parametric maps, structured reports, radiation therapy structures, and presentation states. A derived object references its source images through a series of unique identifiers.

For analysis, derived objects are valuable: a segmentation mask is a label, and a structured report is a structured finding. But a derived object is only as good as its source, and the reference chain must be intact. An orphaned mask with no resolvable source series cannot be validated.

### 3.3 The identity leakage surface

Identity leaks from imaging in five places:

1. **Standard identity attributes.** Patient name, identifier, birth date, accession number.
2. **Dates and times.** Study, series, and acquisition dates and times.
3. **Institution and device.** Institution name, station name, device serial number, manufacturer model.
4. **Free-text attributes.** Study description, series description, protocol name, and operator names often contain names, locations, or dates.
5. **Pixels.** Burned-in annotations, ultrasound overlays, and scanned documents contain text rendered into the image.

Items 1 to 3 are handled by standard profiles. Items 4 and 5 require the optional clean profiles and, usually, human review. Item 5 is the one that most often survives an otherwise competent de-identification.

---

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| DICOM | Digital Imaging and Communications in Medicine | Used to mean "an image file" | Standard, PS3.1-PS3.21 |
| Information object definition | The DICOM model of a real-world object | Confused with a file format | IOD |
| SOP class | A service-object pair class defining what an object is and what can be done with it | Ignored in analysis | UID |
| SOP instance | A single DICOM object | Confused with an image | UID |
| Study | A set of series performed for one purpose | Confused with an encounter | UID |
| Series | A set of instances acquired with the same parameters | Confused with a study | UID |
| Instance | One object in a series | Confused with a slice | UID |
| UID | Unique identifier, dot-separated numeric | Treated as a patient identifier | 64 characters maximum |
| StudyInstanceUID | Identifier for a study | Not stable across re-exports if regenerated | UID |
| PatientID | The local patient identifier | Assumed unique across institutions | Local string |
| Accession number | The order number for the study | Confused with a study identifier | Local string |
| Modality | The imaging type (CT, MR, US, PT, MG) | Confused with the body part | DICOM defined term |
| Pixel data | The stored image values | Assumed to be in physical units | Integer array |
| Rescale slope | Multiplier converting stored values to physical units | Ignored | Numeric |
| Rescale intercept | Offset added after the slope | Ignored | Numeric |
| Hounsfield unit (HU) | The CT attenuation scale, water 0, air −1000 | Used for MRI | HU |
| Window center and width | Display mapping for grayscale | Confused with the pixel values | Numeric |
| Photometric interpretation | How pixel values map to colour channels | Ignored in conversion | MONOCHROME2, RGB, YBR_FULL |
| Transfer syntax | The encoding of the pixel data | Ignored, causing read errors | UID |
| Slice thickness | Nominal reconstructed slice thickness | Confused with spacing between slices | Millimetres |
| Pixel spacing | Physical size of a pixel in the image plane | Ignored in measurement | Millimetres |
| Convolution kernel | The reconstruction filter applied | Ignored, producing texture batch effects | Local string |
| kVp | Peak kilovoltage | Ignored | Kilovolts |
| mAs | Tube current-time product | Ignored | Milliampere-seconds |
| CTDIvol | Volume computed tomography dose index | Confused with patient dose | Milligray |
| DLP | Dose-length product | Confused with effective dose | Milligray-centimetre |
| Contrast agent | The administered contrast medium | Absent or free text | Free text or coded |
| Contrast phase | Arterial, portal venous, delayed | Not a standard DICOM field | Convention |
| Protocol name | The scanner protocol label | Free text; often contains identifiers | Local string |
| PACS | Picture archiving and communication system | Confused with the archive standard | System |
| VNA | Vendor neutral archive | Confused with PACS | System |
| DICOMweb | RESTful services for DICOM (QIDO-RS, WADO-RS, STOW-RS) | Assumed universal | Interface |
| DIMSE | The older DICOM network services | Assumed obsolete | Interface |
| Structured report | A DICOM object encoding findings | Confused with a text report | SR |
| SEG | Segmentation object | Confused with a mask file | DICOM SEG |
| RTSTRUCT | Radiation therapy structure set | Confused with SEG | DICOM RT |
| Burned-in annotation | Text rendered into pixel data | Assumed absent | Flag (0028,0301) |
| Private tag | A non-standard vendor attribute | Ignored in de-identification | Odd group number |
| TCIA | The Cancer Imaging Archive | Treated as a single dataset | NCI archive |
| De-identification method | Coded record of what was done | Omitted | CID 7050 |

### 4.1 Terms that are routinely conflated

- **Study versus series versus instance.** A study is the whole exam, a series is a set of images with the same parameters, and an instance is one image. "Number of images" and "number of studies" differ by two orders of magnitude.
- **Slice thickness versus slice spacing.** Thickness is the reconstructed slab; spacing is the distance between slice centres. They differ for overlapping reconstructions, and using one for the other produces wrong volumes.
- **Window width versus pixel range.** Windowing changes display, not the stored values. An analysis on windowed values is analyzing a display setting.
- **Anonymization versus de-identification.** In DICOM practice these are often used interchangeably. The defensible term is de-identification, with the method recorded.

---

## 5. The mechanics, step by step

This section is the working procedure for building a defensible imaging dataset from a PACS or an archive. It has 29 steps in five phases.

### Phase A — Define the query and the export (steps 1-6)

**Step 1. Write the clinical or analytic question and the required series.**
Input: the stakeholder request. Output: a statement naming the modality, body part, contrast phase, and minimum series completeness.
Reason: imaging queries return whatever matches the filter, which is rarely what the analyst meant. Naming the required series prevents a cohort of mixed phases.
Rule: write the requirement in terms of DICOM fields where possible, not clinical descriptions.

**Step 2. Translate the requirement into DICOM query keys.**
Input: the requirement. Output: a query specification with modality, body part, study date range, and series description filters.
Reason: DICOM query services match on attributes. A query for "CT abdomen" is really a query on modality CT plus a body part or protocol string, and the string is site-specific.
Rule: test the query on a small date range and inspect the returned series descriptions before running the full export.

**Step 3. Confirm the archive and the access method.**
Input: the PACS or VNA documentation. Output: a documented access method (DICOMweb, DIMSE, vendor export).
Reason: the access method determines reproducibility. A manual export from a workstation cannot be re-run.
Rule: prefer DICOMweb or a scripted DIMSE query; record the endpoint and the query.

**Step 4. Establish the study-level and series-level inclusion rules.**
Input: the query results. Output: inclusion rules at both levels.
Reason: a study can be included while the required series is absent. Series-level inclusion is what determines analyzability.
Rule: require at least one complete series meeting the acquisition specification; exclude studies with only derived or localizer series.

**Step 5. Record the export date and the archive state.**
Input: the export run. Output: the export date, the archive software version, and the query.
Reason: PACS content changes as studies are added, corrected, or deleted. An export is a snapshot.
Rule: store the query and the run date with the data.

**Step 6. Estimate the data volume before exporting.**
Input: the query result count and the mean study size. Output: an estimated volume in terabytes.
Reason: imaging exports fail on storage more often than on compute. Estimating first avoids a half-complete transfer.
Rule: compute the estimate from the actual mean size of a sample, not a vendor rule of thumb.

### Phase B — Build the cohort (steps 7-12)

**Step 7. Deduplicate studies.**
Input: the export manifest. Output: one record per StudyInstanceUID.
Reason: the same study can appear under two accession numbers after a merge, or be re-sent from a second modality.
Rule: deduplicate on StudyInstanceUID; investigate studies with the same accession and different UIDs.

**Step 8. Verify series completeness.**
Input: the instance counts per series. Output: a completeness flag.
Reason: a series with missing slices is common after a transfer interruption, and it silently breaks volumetric analysis.
Rule: compare the instance count to the expected count from the series header (`NumberOfSeriesRelatedInstances` where present) and to the geometric spacing.

**Step 9. Classify series by acquisition parameters.**
Input: series headers. Output: a series type (for example, portal venous, 1 mm, soft kernel).
Reason: the analysis needs comparable series, and the classification must come from the header, not from the description string.
Rule: classify on contrast agent presence, slice thickness, and kernel, and record the rule.

**Step 10. Apply exclusion criteria with counts.**
Input: the classified series. Output: a flow diagram.
Reason: every exclusion changes comparability. Counting them makes the trade-off visible.
Rule: report the count removed at each step and the reason.

**Step 11. Check patient-level linkage and duplicates.**
Input: PatientID and demographics. Output: a resolved patient key.
Reason: the same patient can have multiple identifiers, and a merge can leave two study sets.
Rule: resolve identity before analysis; check for studies with the same patient but conflicting demographics.

**Step 12. Freeze the manifest.**
Input: the cohort. Output: a versioned manifest listing every StudyInstanceUID, SeriesInstanceUID, and SOPInstanceUID included.
Reason: the manifest is the reproducibility artefact. Without it, the cohort cannot be reconstructed.
Rule: hash the manifest and store it with the analysis code.

### Phase C — Prepare the pixel data (steps 13-18)

**Step 13. Read the transfer syntax and decode correctly.**
Input: the DICOM files. Output: decoded arrays.
Reason: compressed transfer syntaxes require the correct decoder. A wrong decoder produces garbage that looks like an image.
Rule: check the transfer syntax UID and confirm the decoder supports it.

**Step 14. Apply the rescale slope and intercept.**
Input: stored pixel values. Output: values in physical units.
Reason: CT values are stored as integers that must be converted to HU. Without the conversion, all thresholds are wrong.
Rule: compute `HU = stored × slope + intercept` and verify against a known structure (air, water, bone).

**Step 15. Verify the photometric interpretation and channel order.**
Input: the pixel array. Output: a correctly oriented array.
Reason: MONOCHROME1 is inverted relative to MONOCHROME2, and RGB versus YBR ordering changes colours. An inverted image inverts every intensity feature.
Rule: assert the photometric interpretation and apply the correct mapping.

**Step 16. Handle windowing explicitly.**
Input: the pixel array. Output: either the raw values or a documented window.
Reason: windowing is a display transform. If the analysis uses windowed values, the window must be recorded and applied consistently.
Rule: analyze in HU or in raw values, never in an undocumented display window.

**Step 17. Resample only with a recorded target.**
Input: anisotropic volumes. Output: resampled volumes with a recorded spacing.
Reason: resampling changes texture and partial volume. It is sometimes necessary and always consequential.
Rule: record the original spacing, the target spacing, and the interpolation method.

**Step 18. Record the derivation chain for every derived object.**
Input: segmentations, maps, conversions. Output: a record linking derived objects to source series.
Reason: a derived object without its source cannot be validated or reproduced.
Rule: keep the DICOM reference attributes or an explicit mapping table.

### Phase D — De-identify and audit (steps 19-24)

**Step 19. Apply the basic confidentiality profile.**
Input: the DICOM files. Output: de-identified files with a recorded method.
Reason: PS3.15 Annex E defines the baseline profile and the action codes. Applying it is the minimum.
Rule: record the de-identification method code sequence (0012,0064) using CID 7050 values.

**Step 20. Apply the clean descriptors option.**
Input: text attributes. Output: cleaned or removed free text.
Reason: study and series descriptions often contain names, dates, or locations. The clean descriptors option handles them.
Rule: prefer retaining only values known to be safe; document the cleaning method in the conformance statement.

**Step 21. Apply the clean pixel data option and verify burned-in annotations.**
Input: pixel data. Output: pixel data without identifying text, and the burned-in annotation flag set to NO.
Reason: annotations rendered into pixels survive header de-identification. PS3.15 requires removal of the actual pixels, not just an overlay.
Rule: set BurnedInAnnotation (0028,0301) to NO only after confirming no identifying text remains; use OCR or visual review on a sample.

**Step 22. Inspect private tags.**
Input: the full header. Output: a list of private tags and a disposition for each.
Reason: private tags can contain institution names, operator identifiers, and protocol details. They are invisible to standard de-identification tools that only handle standard attributes.
Rule: enumerate odd-group tags and either remove them or justify retention.

**Step 23. Replace identifiers with consistent pseudonyms.**
Input: patient identifiers and UIDs. Output: pseudonymized identifiers that preserve within-patient linkage.
Reason: analysis needs to link a patient's studies; sharing needs the linkage to be non-identifying. A consistent pseudonym achieves both.
Rule: use a keyed hash or a mapping table held separately; never regenerate UIDs randomly per export, which destroys linkage.

**Step 24. Audit a sample and record the results.**
Input: the de-identified collection. Output: an audit report with defect counts.
Reason: de-identification failures are common and are found by sampling, not by assumption.
Rule: sample at least 100 studies, check for dates, names, institutions, device serials, and burned-in text, and report the defect rate.

### Phase E — Analyze and document (steps 25-29)

**Step 25. Test for scanner and protocol effects before modeling.**
Input: the analytic dataset. Output: a comparison of feature distributions by scanner and protocol.
Reason: acquisition heterogeneity is the dominant source of imaging batch effects. A model that ignores it learns the scanner.
Rule: plot the primary feature by scanner and protocol; test for a difference; correct or stratify if present.

**Step 26. Calibrate with a phantom where available.**
Input: phantom scans or a reference standard. Output: an offset and scale per scanner.
Reason: a physical reference converts a relative difference into a measurable bias and allows correction.
Rule: record the phantom, the insert values, and the correction applied.

**Step 27. Report the acquisition distribution.**
Input: the analytic dataset. Output: a table of scanners, protocols, kernels, and thicknesses with counts.
Reason: a validation set from one scanner is not a validation of generalization. The distribution is the evidence.
Rule: report it in the paper, not the supplement.

**Step 28. State the post-processing chain.**
Input: the pipeline. Output: a versioned description of every transform.
Reason: a result that depends on an unrecorded resampling or windowing is not reproducible.
Rule: record the code version, the parameters, and the order of operations.

**Step 29. State what the imaging data cannot answer.**
Input: the question. Output: a limits section.
Reason: images show structure and some function; they do not show symptoms, adherence, or care process.
Rule: name at least three limits and say which threatens the conclusion.

### 5.1 Field-level mechanics: the CT header fields that matter

| Attribute | Tag | Why it matters | Typical failure |
|---|---|---|---|
| Modality | (0008,0060) | Determines the analysis family | Mixed modalities in one cohort |
| Manufacturer | (0008,0070) | Scanner family effect | Not recorded in the analysis |
| ManufacturerModelName | (0008,1090) | Model-level effect | Collapsed into manufacturer |
| DeviceSerialNumber | (0018,1000) | Identifies the physical scanner; also an identifier | Retained in a "de-identified" file |
| StationName | (0008,1010) | Room-level effect; often identifying | Retained |
| InstitutionName | (0008,0080) | Site effect; identifying | Retained |
| StudyDate | (0008,0020) | Temporal trend; identifying | Shifted inconsistently |
| SeriesDescription | (0008,103E) | Free text; often contains protocol and identifiers | Not cleaned |
| ProtocolName | (0018,1030) | Acquisition protocol | Used as a classifier without validation |
| SliceThickness | (0018,0050) | Comparability | Confused with spacing |
| SpacingBetweenSlices | (0018,0088) | Volume computation | Absent for some reconstructions |
| PixelSpacing | (0028,0030) | Physical measurement | Ignored, so measurements are in pixels |
| ConvolutionKernel | (0018,1210) | Texture and noise | Not recorded |
| KVP | (0018,0060) | Contrast and dose | Not recorded |
| Exposure | (0018,1152) | Dose | Not recorded |
| ContrastBolusAgent | (0018,0010) | Phase classification | Free text or absent |
| RescaleSlope | (0028,1053) | HU conversion | Ignored |
| RescaleIntercept | (0028,1052) | HU conversion | Ignored |
| BurnedInAnnotation | (0028,0301) | De-identification status | Set to NO without verification |

Two of these deserve special mention. `DeviceSerialNumber` is both the most useful scanner identifier and a direct identifier under HIPAA Safe Harbor, which lists device identifiers and serial numbers. `ProtocolName` is free text and often contains the technologist's initials or the referring clinic, which is why the clean descriptors option exists.

### 5.2 Field-level mechanics: MR is not CT

Magnetic resonance (MR) images have no Hounsfield scale. Pixel values are arbitrary and vary with coil, sequence, and scanner. Four consequences:

- **No absolute thresholds.** A threshold in MR is scanner-specific and sequence-specific.
- **Intensity normalization is mandatory** for cross-site comparison, and every normalization method introduces its own assumptions.
- **Sequence matters more than in CT.** T1, T2, FLAIR, DWI, and contrast-enhanced T1 are different measurements.
- **Field strength matters.** 1.5 tesla and 3 tesla scanners produce different signal and contrast.

An analyst who treats MR like CT will apply a threshold that means nothing and will report a scanner effect as a biological finding.

### 5.3 Quantitative checks to run before analysis

| Check | Computation | Passing signal | What failure means |
|---|---|---|---|
| Series completeness | Instances versus expected | Equal, or a documented reason | Transfer interruption |
| Geometric consistency | Slice positions evenly spaced | Constant spacing | Mixed acquisitions in one series |
| HU sanity | Air near −1000, water near 0 | Within 20 HU | Wrong rescale or wrong decoder |
| Duplicate studies | Same patient and date | Low rate | Merge artefacts |
| Scanner balance | Studies per scanner per year | Stable | A scanner was replaced |
| Protocol drift | Protocol names over time | Few changes | A protocol change breaks comparability |
| Burned-in flag | Share with flag YES | Decreasing to zero | De-identification not applied |
| Private tag presence | Share with odd-group tags | Zero after de-identification | Incomplete cleaning |

### 5.4 Segmentations and derived labels

A segmentation is the most common label in imaging AI, and it is the object most often handled carelessly.

1. **Confirm the source series.** A segmentation must reference the series it was drawn on. If the reference is missing, the mask cannot be validated against the image.
2. **Confirm the geometry.** Rows, columns, spacing, and origin must match the source. A mask that was resampled independently will be misaligned by a sub-voxel amount that is invisible visually and material for volumetry.
3. **Confirm the label dictionary.** A multi-label segmentation needs a mapping from label value to anatomical structure. Without it, label 3 is whatever the annotator decided.
4. **Confirm the annotator and the adjudication.** Who drew it, with what tool, and whether disagreements were resolved. Inter-annotator agreement is the only estimate of label noise.
5. **Confirm the derivation chain.** If the mask was produced by a model rather than a human, the model version and threshold belong in the record.

A mask without these five items is an array of integers with an unknown relationship to anatomy.

### 5.5 Derivations and the assumptions they hide

| Derived variable | Usual construction | Hidden assumption | Challenge question |
|---|---|---|---|
| Organ volume | Voxel count × voxel volume | Voxel volume is correct and the mask is complete | Was spacing applied per axis? |
| Mean attenuation | Mean of HU in a region | The region is correctly placed | Was the ROI drawn on the same phase? |
| Texture feature | Filter response statistics | The kernel and thickness are fixed | Which kernel, which bin width? |
| Lesion diameter | Maximum in-plane extent | The lesion is measured in the acquisition plane | Was it measured on reformats? |
| Dose estimate | DLP × conversion factor | The conversion factor applies to this body region | Which factor and which region? |
| Scanner effect | Difference in feature by scanner | Scanner is the only difference | Were protocol and patient mix balanced? |

### 5.6 Branch points summarized

| Branch | Condition | Path A | Path B |
|---|---|---|---|
| Series selection | Multiple phases available | Restrict to one phase | Model phase as a covariate |
| Thickness | Range exceeds 1.25-2.5 mm | Restrict to a narrow range | Resample and record the target |
| De-identification | External sharing intended | Full profile plus clean options | Retain options in a controlled enclave |
| UIDs | Longitudinal linkage needed | Consistent pseudonym mapping | Retain UIDs under controlled access |
| Calibration | Phantom available | Apply offsets | Include scanner as a covariate and state the limit |
| Derived objects | SEG or NIfTI | Preserve references | Store a mapping table |

Three of these choices are irreversible once the export is done. Restricting the thickness range discards studies permanently; applying a full de-identification profile cannot be undone; and resampling to a target spacing loses the original grid. Make them deliberately, and record them before the export runs rather than after.

### 5.7 Branch points summarized

### 5.6 Branch points summarized

---

## 6. Worked example

### 6.1 The question

A radiology AI group wants to build a model that predicts hepatic steatosis from unenhanced CT. They have access to a three-scanner archive covering 2022-2024. Before modeling, they must determine whether the cohort is comparable across scanners and whether a scanner-related measurement bias exists.

### 6.2 Cohort construction

| Step | Criterion | Removed | Remaining |
|---|---|---:|---:|
| 0 | CT abdomen studies, 2022-2024 | — | 12,480 |
| 1 | Not portal venous or unenhanced phase (mixed or arterial only) | 3,120 | 9,360 |
| 2 | Slice thickness greater than 2.5 mm | 2,244 | 7,116 |
| 3 | Incomplete series (missing instances) | 426 | 6,690 |
| 4 | Age under 18 at study date | 318 | 6,372 |

Check: 12,480 − 3,120 = 9,360. 9,360 − 2,244 = 7,116. 7,116 − 426 = 6,690. 6,690 − 318 = 6,372.

Scanner distribution: scanner A 2,980 studies, scanner B 2,110, scanner C 1,282.

Check: 2,980 + 2,110 + 1,282 = 6,372.

Storage estimate: mean study size is 220 megabytes (MB).

- 6,372 × 220 MB = 1,401,840 MB = 1.40 terabytes (TB).

Check: 6,372 × 220 = 1,401,840. Divided by 1,024 twice gives 1.337 tebibytes; in decimal units, 1.40 TB. Stating the unit convention matters, because vendors quote decimal and operating systems quote binary.

### 6.3 Liver attenuation by scanner

Mean liver attenuation measured in a 2 cm region of interest in the right lobe, in Hounsfield units:

| Scanner | Studies | Mean HU | SD HU |
|---|---:|---:|---:|
| A | 2,980 | 58.4 | 9.2 |
| B | 2,110 | 52.1 | 8.7 |
| C | 1,282 | 61.7 | 10.1 |

The difference between A and B is 6.3 HU, and between B and C is 9.6 HU. A clinical threshold for steatosis is often around 40 HU, so a 6 to 10 HU scanner offset is material: it moves a meaningful fraction of patients across the threshold.

### 6.4 Threshold prevalence before correction

Using a threshold of less than 40 HU:

| Scanner | Studies | Below 40 HU | Prevalence |
|---|---:|---:|---:|
| A | 2,980 | 244 | 8.2% |
| B | 2,110 | 329 | 15.6% |
| C | 1,282 | 78 | 6.1% |

Check: 244 / 2,980 = 0.0819. 329 / 2,110 = 0.1559. 78 / 1,282 = 0.0608.

The apparent prevalence differs by a factor of 2.6 between scanners. If the scanners serve similar populations, most of that difference is measurement, not disease.

### 6.5 Phantom calibration

A calibration phantom with a liver-equivalent insert of 60 HU was scanned on each scanner. Measured values:

| Scanner | Phantom insert true HU | Measured HU | Offset (measured − true) |
|---|---:|---:|---:|
| A | 60.0 | 58.4 | −1.6 |
| B | 60.0 | 52.1 | −7.9 |
| C | 60.0 | 61.7 | +1.7 |

Corrected value = measured − offset.

- A: 58.4 − (−1.6) = 60.0.
- B: 52.1 − (−7.9) = 60.0.
- C: 61.7 − (+1.7) = 60.0.

All three now agree with the phantom, which is the point of a physical reference: it converts a relative difference into a measurable bias.

### 6.6 Effect of correction on the cohort

Applying the offset to every study raises scanner B's values by 7.9 HU and lowers scanner C's by 1.7 HU. Because the threshold is fixed at 40 HU, a shift of the whole distribution moves a slice of it across the threshold, and the size of that slice depends on the density of the distribution at 40 HU.

A normal approximation gives a first estimate of the direction and magnitude. For scanner B, the z-score of 40 HU moves from (40 − 52.1)/8.7 = −1.391 to (40 − 60.0)/8.7 = −2.299. Under a standard normal, Φ(−1.391) = 8.2% and Φ(−2.299) = 1.1%.

The observed prevalence in scanner B was 15.6%, not 8.2%, which tells you the liver attenuation distribution in that scanner is not normal: it has a heavier lower tail, probably because it includes patients with steatosis. The empirical corrected prevalences are therefore the ones to use.

| Scanner | Corrected mean HU | Corrected below 40 HU | Corrected prevalence | Change |
|---|---:|---:|---:|---:|
| A | 60.0 | 236 | 7.9% | −0.3 |
| B | 60.0 | 156 | 7.4% | −8.2 |
| C | 60.0 | 86 | 6.7% | +0.6 |

Check the directions before checking the arithmetic. Scanner B's values were 7.9 HU too low, so correction raises them and moves patients above the threshold; prevalence must fall, and it falls by 8.2 percentage points. Scanner C's values were 1.7 HU too high, so correction lowers them and moves patients below the threshold; prevalence must rise, and it rises by 0.6 points. Both directions are consistent with the offsets, which is the check that catches most calibration errors.

After correction the three scanners range from 6.7 to 7.9 percent, a spread of 1.2 percentage points instead of 9.5. That is the defensible finding: scanner calibration accounts for most of the apparent difference, and a residual difference of about 1.2 percentage points remains, which may be population or may be residual protocol effect.

A correction table that moves prevalence the wrong way is the most common arithmetic error in calibration work, and it is invisible unless you write the expected direction down before you compute.

### 6.7 De-identification audit

A sample of 500 studies was audited after de-identification.

| Defect | Studies | Rate |
|---|---:|---:|
| Date present in StudyDescription or SeriesDescription | 37 | 7.4% |
| BurnedInAnnotation flag set to YES | 12 | 2.4% |
| Private tag containing institution or operator | 5 | 1.0% |
| Any defect | 49 | 9.8% |

Check: 37 / 500 = 0.074. 12 / 500 = 0.024. 5 / 500 = 0.010. 49 / 500 = 0.098.

A 9.8 percent defect rate is not acceptable for external sharing. The most common defect is date leakage in free-text descriptions, which is exactly what the clean descriptors option is designed to catch and what many pipelines skip.

### 6.8 What the example establishes

- The metadata determines comparability, and the cohort exclusions are comparability decisions as much as clinical ones.
- A phantom converts a relative scanner difference into a measurable offset, which is the only way to correct it defensibly.
- A mean offset near a threshold produces a disproportionate change in prevalence, so the effect must be computed from the empirical distribution.
- De-identification defects concentrate in free text and pixels, not in the standard identity attributes that tools handle well.

### 6.9 What a reviewer would ask

1. **Why exclude studies above 2.5 mm?** Because texture features and partial volume differ materially at thicker slices, and mixing them would confound the scanner effect. The cost is 2,244 studies and a loss of generalizability to thick-slice protocols.
2. **Is one phantom scan per scanner enough?** No. It establishes an offset at one point in time. A monthly phantom scan would detect drift, and the absence of drift data is a limitation.
3. **Could the residual 2.1-point spread be disease?** Yes. The scanners may serve different populations. Testing requires adjusting for age, sex, and body mass index, which the header does not contain.
4. **Is the audit sample large enough?** For a 10 percent defect rate, 500 studies give a standard error of about 1.3 percentage points, which is adequate for a decision. For a 1 percent defect rate, it is not.

---

## 7. Code, query, or calculation

### 7.1 Python: header audit with pydicom

Assumptions: Python 3.11 or later; pydicom 2.4 or later installed; a directory of DICOM files.

```python
"""Audit DICOM headers for acquisition metadata and de-identification defects."""
from pathlib import Path
from collections import Counter
import pydicom

ROOT = Path("dicom_sample")   # change to your directory

REQUIRED = [
    "Modality", "Manufacturer", "ManufacturerModelName", "SliceThickness",
    "PixelSpacing", "ConvolutionKernel", "KVP", "ContrastBolusAgent",
    "RescaleSlope", "RescaleIntercept",
]
IDENTITY = [
    "PatientName", "PatientID", "PatientBirthDate", "InstitutionName",
    "StationName", "DeviceSerialNumber", "ReferringPhysicianName",
]

missing = Counter()
identity_present = Counter()
burned_in = Counter()
private_tags = 0
n = 0

for path in ROOT.rglob("*.dcm"):
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    n += 1
    for attr in REQUIRED:
        if not hasattr(ds, attr) or getattr(ds, attr) in ("", None):
            missing[attr] += 1
    for attr in IDENTITY:
        val = getattr(ds, attr, None)
        if val not in (None, ""):
            identity_present[attr] += 1
    flag = getattr(ds, "BurnedInAnnotation", None)
    burned_in[str(flag)] += 1
    private_tags += sum(1 for elem in ds.iterall() if elem.tag.is_private)

print(f"Files scanned: {n}")
print("\nMissing acquisition attributes (count, pct):")
for attr, c in missing.most_common():
    print(f"  {attr:24s} {c:6d}  {100*c/n:5.1f}%")
print("\nIdentity attributes still present (count, pct):")
for attr, c in identity_present.most_common():
    print(f"  {attr:24s} {c:6d}  {100*c/n:5.1f}%")
print("\nBurnedInAnnotation values:", dict(burned_in))
print(f"Private tag elements: {private_tags}")
```

Expected output shape for an un-cleaned sample:

```text
Files scanned: 500

Missing acquisition attributes (count, pct):
  ContrastBolusAgent        118   23.6%
  ConvolutionKernel          41    8.2%

Identity attributes still present (count, pct):
  PatientID                 500  100.0%
  PatientName               500  100.0%
  InstitutionName           500  100.0%
  DeviceSerialNumber        487   97.4%

BurnedInAnnotation values: {'YES': 12, 'NO': 488}
Private tag elements: 3,214
```

The pattern to note: acquisition attributes are mostly present, identity attributes are entirely present, and the burned-in flag is set to YES for 2.4 percent of files. A pipeline that reports "de-identified" while PatientName is populated is not de-identified.

### 7.2 Python: HU conversion and threshold prevalence

Assumptions: numpy installed; arrays loaded from a DICOM series.

```python
import numpy as np

def to_hu(stored: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Convert stored pixel values to Hounsfield units."""
    return stored.astype(np.float32) * float(slope) + float(intercept)

# Example: one slice with slope 1.0 and intercept -1024.0 (typical CT).
stored = np.array([1024, 1100, 1060, 900], dtype=np.int16)
slope, intercept = 1.0, -1024.0
hu = to_hu(stored, slope, intercept)
print("HU values:", hu.tolist())          # [0.0, 76.0, 36.0, -124.0]

# Threshold prevalence for a cohort summary.
def prevalence(values, threshold=40.0):
    values = np.asarray(values, dtype=np.float64)
    return float((values < threshold).mean())

scanner_means = {"A": 58.4, "B": 52.1, "C": 61.7}
scanner_sd = {"A": 9.2, "B": 8.7, "C": 10.1}
rng = np.random.default_rng(20261024)
for s in scanner_means:
    vals = rng.normal(scanner_means[s], scanner_sd[s], size=50_000)
    print(f"Scanner {s}: mean={scanner_means[s]:.1f} HU  "
          f"prevalence<40HU={prevalence(vals)*100:.1f}%")
```

Expected output:

```text
HU values: [0.0, 76.0, 36.0, -124.0]
Scanner A: mean=58.4 HU  prevalence<40HU=8.0%
Scanner B: mean=52.1 HU  prevalence<40HU=11.6%
Scanner C: mean=61.7 HU  prevalence<40HU=6.5%
```

These simulated prevalences are close to the normal-approximation values and differ from the empirical table in section 6.4, which is the point: a simulated normal distribution is not the real distribution, and a real analysis must use the observed values rather than a mean and SD.

### 7.3 Converting a study size to storage

Given 6,372 studies at 220 MB each:

1. 6,372 × 220 = 1,401,840 MB.
2. Divide by 1,024 to get gigabytes: 1,368.98 GB.
3. Divide by 1,024 again to get tebibytes: 1.336 TiB.
4. In decimal units, 1,401,840 MB = 1.40 TB.

Always say which convention you used. A vendor quoting 1.40 TB and a filesystem reporting 1.34 TiB are both correct, and the 4.7 percent difference has caused more than one capacity incident.

---

## 8. Data, standards, and regulatory context

### 8.1 The DICOM standard

DICOM is published by the National Electrical Manufacturers Association (NEMA) as PS3.1 through PS3.21, with an edition updated several times a year (for example, the 2025a, 2025d, and 2026b editions). The parts that matter most for data work:

| Part | Content | Relevance |
|---|---|---|
| PS3.3 | Information object definitions | Defines the attributes available in each object type |
| PS3.4 | Service class specifications | Defines storage, query, and retrieval services |
| PS3.5 | Data structures and encoding | Explains transfer syntax and value representation |
| PS3.6 | Data dictionary | The authoritative tag and keyword list |
| PS3.10 | Media storage and file format | The file format on disk |
| PS3.14 | Grayscale standard display function | Display calibration |
| PS3.15 | Security and system management profiles | De-identification, including Annex E |
| PS3.18 | Web services | DICOMweb: QIDO-RS, WADO-RS, STOW-RS |
| PS3.20 | Imaging reports using HL7 CDA | Report encoding |

### 8.2 PS3.15 Annex E in detail

Annex E defines the Basic Application Level Confidentiality Profile and a set of options. The mechanism is a table of attributes with an action code for each:

| Action code | Meaning |
|---|---|
| D | Replace with a dummy value of the same type |
| Z | Zero out the value |
| X | Remove the attribute |
| K | Keep the value |
| C | Clean the value so it is non-identifying |
| U | Replace with a consistently mapped pseudonym |

Options that add removal or cleaning:

- **Clean Pixel Data Option (E.3.1).** Requires removal of identifying information burned into the pixel data. The actual stored pixel values must be blacked out; an overlay or shutter is not sufficient. The Burned In Annotation attribute (0028,0301) must be set to NO. Human review may be required, and OCR or image processing is permitted but not mandated.
- **Clean Graphics Option (E.3.3).** Handles identifying information in graphics, text annotations, or overlays, excluding structured reports.
- **Clean Descriptors Option (E.3.5).** Handles unstructured identifying information in text attributes such as Study Description. The standard notes that automated cleaning works best by retaining only values known to be safe, and that the method must be described in the conformance statement.
- **Clean Structured Content Option.** Handles identifying information in structured reports.

Options that retain information:

- **Retain Longitudinal Temporal Information.** Keeps dates, optionally with modified dates, which preserves intervals.
- **Retain Patient Characteristics.** Keeps age, sex, and similar attributes.
- **Retain Device Identity and Retain Institution Identity.** Keeps scanner and site information.
- **Retain Safe Private Option.** Keeps private tags that have been assessed as safe.
- **Retain UIDs.** Keeps unique identifiers, which is necessary for linkage and is only appropriate in controlled settings.

Annex E also states explicitly that attribute-level de-identification does not guarantee full de-identification of the object. That sentence is the most important one in the standard for anyone planning to share data.

The method used must be recorded in De-identification Method Code Sequence (0012,0064), using codes from CID 7050.

### 8.3 What is binding and what is convention

Binding in the US:

- **HIPAA Privacy Rule** (45 CFR 164.514) defines de-identification for protected health information. DICOM objects are PHI when they contain identifiers.
- **HIPAA Security Rule** governs storage and transmission.
- **FDA device regulation** applies to software that analyzes images for a clinical purpose. An AI/ML imaging product with a diagnostic claim is a device and requires a premarket pathway. The FDA has published guidance on predetermined change control plans for AI-enabled devices, which allows specified modifications without a new submission.
- **EU Medical Device Regulation (MDR) 2017/745** applies to imaging software placed on the EU market, with conformity assessment and clinical evaluation requirements.

Convention:

- Using the portal venous phase for liver analysis is a clinical convention, not a standard.
- Reporting a single phantom calibration per scanner per year is common practice and is not sufficient to detect drift.
- Storing derived segmentations as DICOM SEG rather than NIfTI is increasingly common and is not universal.

### 8.4 Licensing of imaging data

Imaging datasets carry two layers of rights: the data use terms and the software terms.

- **TCIA collections** are mostly released under Creative Commons Attribution licenses (CC BY 3.0 or CC BY 4.0), with some non-commercial variants, while a subset requires National Institutes of Health (NIH) controlled access. Check per collection, not per archive.
- **Institutional archives** are governed by the institution's data use agreement and by patient consent, which may not extend to commercial use.
- **Model weights** trained on licensed data may inherit restrictions. Read the license for the training data before distributing weights.

> **Note:** Licensing statements here are educational. A redistribution decision requires reading the specific agreement.

### 8.5 Imaging and FHIR

The `ImagingStudy` resource in HL7 Fast Healthcare Interoperability Resources (FHIR) provides a metadata view of a DICOM study: the study UID, the modality list, the series, and references to endpoints. It does not carry pixel data.

The practical pattern is: use FHIR `ImagingStudy` for cohort discovery and metadata queries, and DICOMweb for pixel retrieval. A pipeline that tries to do both through FHIR will not work, because FHIR is not a pixel transport.

DICOMweb itself defines three services:

- **QIDO-RS** for query (studies, series, instances).
- **WADO-RS** for retrieval (metadata, frames, bulk data).
- **STOW-RS** for storage.

These have largely replaced DIMSE for new integrations, and they make scripted, reproducible exports possible in a way that workstation exports are not.

---

## 9. Industry practice and case evidence

### 9.1 The public imaging commons

The Cancer Imaging Archive (TCIA), operated by the National Cancer Institute, is the largest public cancer imaging resource. Its 2017 data descriptor reported 30.9 million radiology images from about 37,568 subjects; the archive has grown since, and its browse interface now presents more than 200 collections organized by disease, modality, and research focus, with subject counts per collection.

The largest collections give a sense of scale: the National Lung Screening Trial collection has 26,254 subjects, Breast-Cancer-Screening-DBT has 5,060, RADCURE has 3,346, and LIDC-IDRI has 1,010. TCIA reported in March 2024 that it received more than 20,000 visitors per month and that users downloaded 279 TB of data in January 2024.

For an analyst, TCIA's value is not only the images. It is that each collection carries a defined protocol, a data use policy, and often a DOI, which makes a study reproducible in a way that an institutional archive rarely is.

### 9.2 What imaging AI vendors get wrong first

The public record on imaging AI failure is dominated by two themes: site shift and label leakage.

**Site shift.** Models validated on a single institution's scanner and protocol lose performance at other sites. The mechanism is acquisition heterogeneity: different kernels, thicknesses, contrast protocols, and patient populations. The published response is multi-site training and explicit reporting of the acquisition distribution, which is still not universal.

**Label leakage.** A label derived from a report or from a metadata field can encode information not available at deployment. A model that predicts "this study was ordered as a liver protocol" will appear to detect liver disease and will fail when the ordering pattern changes.

Neither failure is a modeling problem in the narrow sense. Both are data provenance problems, and both are prevented by the procedure in section 5.

### 9.3 De-identification in practice

The published evidence on DICOM de-identification defects is consistent across settings: standard identity attributes are handled well by tooling, and free text and pixels are handled poorly. The reasons are structural:

- Tooling operates on standard tags. Private tags and free text require enumeration and judgment.
- Burned-in annotation detection requires either OCR or human review, and many pipelines skip both.
- Date shifting is often done inconsistently across a patient's studies, which breaks longitudinal analysis even when it protects identity.

The practical standard in industry is a documented pipeline plus a sampled audit with a reported defect rate. A pipeline without an audit is an assertion.

### 9.4 PACS, VNA, and the export problem

Most hospitals run a PACS from one of a small number of vendors, increasingly with a vendor neutral archive underneath. The export problem is organizational more than technical:

- The PACS team is measured on uptime and clinical availability, not on research support.
- Research exports compete with clinical traffic for bandwidth and storage.
- Each export is often a manual ticket, which makes reproducibility depend on the person who ran it.

The organizations that do this well build a **research export service**: a scripted DICOMweb query, a documented manifest, a de-identification step, and an audit. That service turns a one-off favour into a reproducible capability, and it is the single highest-leverage investment for an imaging data program.

### 9.5 Phantom and calibration practice

Quantitative imaging groups routinely scan calibration phantoms to detect scanner drift and to harmonize measurements. The practice is more common in clinical trials, where a central imaging core lab specifies the phantom and the acquisition, than in routine care.

For a real-world imaging dataset, the absence of phantom data is a limitation that cannot be fixed retrospectively. The honest statement is that scanner offsets are unknown and that the analysis controls for scanner as a categorical variable, which absorbs the mean offset but not any scanner-by-patient interaction.

### 9.6 What a mature imaging data team does

- **Records the acquisition distribution** in every result.
- **Keeps the manifest** and hashes it.
- **Audits de-identification** on a sample and reports the defect rate.
- **Calibrates or adjusts** for scanner before modeling.
- **Stores derived objects** with their source references.
- **Uses DICOMweb** so exports are scriptable.

None of these is exotic. Each is the difference between a result that can be defended and one that cannot.

### 9.7 The economics of imaging data work

Imaging projects fail on logistics more often than on modeling. Three costs dominate:

| Cost | Typical driver | Order of magnitude | Where teams under-estimate |
|---|---|---|---|
| Storage | Study volume times mean study size | 1-5 TB per 1,000 CT studies | Growth from derived objects and versions |
| Transfer | Bandwidth and de-identification compute | Days per terabyte on hospital networks | Contention with clinical traffic |
| Curation | Selection, QC, and labeling | Weeks per 1,000 studies | Annotation time per case |

The third cost is the one that decides project timelines. A model can be trained in hours; a curated, adjudicated, scanner-balanced dataset takes months. Budgeting for curation, rather than for graphics processing units (GPUs), is the mark of a team that has done this before.

### 9.8 Why single-vendor archives are a procurement risk

Many health systems consolidate imaging into a single vendor's archive. That simplifies operations and creates a dependency: the export interface, the de-identification tooling, and the metadata completeness are all determined by the vendor's implementation of the standard.

DICOM is a standard, and vendors implement it with different degrees of completeness. The practical diligence question is not "is it DICOM compliant" but "which attributes are populated, which are dropped on ingest, and what is the supported export path." A vendor that drops `ConvolutionKernel` on ingest has permanently removed the ability to harmonize texture features, and no amount of downstream work recovers it.

### 9.9 Questions to ask an imaging data vendor

- Which DICOM attributes are retained on ingest, and which are dropped or normalized?
- Is the original transfer syntax preserved, or are objects transcoded?
- How are UIDs handled across exports, and is linkage stable?
- What de-identification profile is applied, and is an audit report available?
- Is there a phantom or calibration record for each scanner?
- What is the export path, and is it scriptable through DICOMweb?

A vendor that can answer all six with documentation is selling an archive. One that cannot is selling a viewer with an export button.

---

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| HU conversion skipped | Liver values in the thousands | Rescale slope and intercept ignored | Every threshold wrong | Apply `value*slope + intercept`; verify air and water |
| Mixed contrast phases | Enhancement varies within cohort | Phase not recorded or not used as a filter | Lesion detection confounded | Classify phase from header; exclude or stratify |
| Kernel heterogeneity | Texture features differ by scanner | Reconstruction filter differs | Model learns kernel not biology | Record kernel; stratify or harmonize |
| Slice thickness mixed | Volume estimates vary | Partial volume differs | Biased volumetry | Restrict to a thickness range |
| Burned-in annotations retained | Flag YES or OCR finds text | Text rendered into pixels | Privacy breach | Clean pixel data option; OCR audit |
| Private tags retained | Odd-group tags present | Vendor attributes not enumerated | Institution or operator leak | Enumerate and remove |
| Dates inconsistently shifted | Longitudinal intervals change | Per-export random shifting | Broken within-patient linkage | Consistent keyed shift per patient |
| Regenerated UIDs | Linkage between exports fails | New UIDs per export | Irreproducible cohorts | Retain or consistently remap UIDs |
| Series incompleteness | Instance count below expected | Transfer interruption | Broken volumes | Compare counts; verify spacing |
| Orphaned derived objects | Mask with no resolvable source | Reference chain broken | Unvalidatable labels | Preserve references |
| Display window used as data | Values bounded by window | Windowed values analyzed | Loss of quantitative meaning | Analyze in HU or raw values |
| Single-site validation | Test set from one scanner | No external data | Overstated generalization | Multi-site test; report acquisition distribution |
| Report-derived label leakage | Label predicts ordering pattern | Label encodes workflow | Model fails at deployment | Define label from image or adjudicated truth |

### 10.1 The de-identification red flags

Three signs that an imaging dataset is not de-identified in any meaningful sense:

1. **`PatientName` or `PatientID` is populated.** This is the most basic check and it still fails in shared datasets.
2. **`BurnedInAnnotation` is YES, or absent.** Absence means nobody checked. YES means there is text in the pixels.
3. **Full dates are present.** Even without names, full dates plus a rare study type plus an institution can identify.

A dataset that passes all three is not guaranteed safe, but one that fails any of them is not de-identified.

### 10.2 The "it looks fine" trap

Imaging data fails visually before it fails numerically. A converted image with the wrong photometric interpretation looks plausible; an image in the wrong window looks plausible; a series with slices out of order reconstructs into something that resembles anatomy. The only defence is programmatic checks on the header and on known reference structures, not visual inspection.

---

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The series is complete | Volumetric analysis | Transfer interruption, missing instances | Instance count versus expected |
| Pixel values are in physical units after rescale | Quantitative thresholds | Wrong slope or intercept | Air near −1000 HU, water near 0 HU |
| The acquisition protocol is stable over time | Trend analysis | Protocol change, scanner replacement | Protocol name distribution by year |
| Scanner is a sufficient proxy for acquisition | Batch adjustment | Protocol varies within scanner | Nested scanner-by-protocol analysis |
| Phantom calibration generalizes to patients | Bias correction | Patient size and positioning differ | Compare corrected cohort distributions |
| De-identification is complete | Data sharing | Free text, pixels, private tags | Sampled audit with defect rate |
| The label is derived from the image | Model validity | Label from report or ordering | Compare label to adjudicated reads |
| The cohort is representative | Generalization | Single-site, single-scanner data | External test set |

### 11.1 Limits that are inherent

- **Images show structure and limited function.** They do not show symptoms, adherence, or care process.
- **Acquisition is not randomized.** Scanner assignment follows geography, time, and workflow, so scanner is confounded with population.
- **Retrospective calibration is impossible without a phantom.** If no phantom was scanned, the offset is unknown.
- **De-identification is risk reduction, not elimination.** Residual risk remains in rare anatomy and unusual findings.
- **Pixel data is large.** Storage and transfer dominate project timelines more often than modeling does.

### 11.2 How to state the acquisition limitation
A defensible sentence: "Studies were acquired on three scanners (2,980, 2,110, and 1,282 studies) with slice thickness 0.625 to 2.5 mm and four reconstruction kernels; scanner and kernel were included as covariates, and a phantom calibration reduced the between-scanner mean difference in liver attenuation from 9.6 to 1.2 HU. Residual confounding by patient body habitus is possible and cannot be assessed from the header."

That sentence states the distribution, the adjustment, the effect of the adjustment, and the residual threat. It is what a reviewer needs.

### 11.3 What the header cannot tell you

The header is rich and it is not complete. Four things it cannot tell you:

- **Patient habitus.** Body mass index affects attenuation, noise, and dose. It is sometimes in the header and usually is not.
- **Why the study was ordered.** The indication is in the order and the report, not the header.
- **Whether the protocol was followed.** ProtocolName records the intent, not the execution.
- **Whether the patient moved.** Motion artifact is visible in the image and not encoded in the header.

Each of these is a real confounder for imaging analysis, and each requires linking to another source or accepting as a stated limitation.

### 11.4 The limit that matters most for AI

For an AI model, the most important limit is that the training distribution is the deployment distribution only by assumption. A model trained on studies that met a research protocol will meet, in deployment, studies that did not: thicker slices, more motion, different contrast timing, and different patient populations.

The defence is not a better model. It is a deployment monitoring plan that measures the acquisition distribution of incoming studies and flags drift before performance is affected.

---

## 12. Exercises

### Exercise 1 — Information model (recall, 20 minutes)

Draw the DICOM hierarchy for a patient with two CT studies, one of which has three series. Label every identifier.

Deliverable: a diagram with four levels and the UIDs at each level.

Acceptance criteria: every level has the correct identifier; the diagram distinguishes study, series, and instance counts.

### Exercise 2 — HU conversion (calculation, 30 minutes)

Given stored values 1024, 1100, 1060, 900 with slope 1.0 and intercept −1024, compute the HU values. Then compute them with slope 1.0 and intercept −1000 and explain which is correct and how you would check.

Deliverable: two sets of values and a verification method.

Acceptance criteria: arithmetic correct; the verification method uses air and water references, not a guess.

### Exercise 3 — Cohort accounting (application, 60 minutes)

Reproduce the cohort flow in section 6.2 and compute the storage estimate in both decimal and binary units. Then recompute the cohort if the slice thickness threshold were 1.25 mm, assuming 38 percent of the 7,116 studies after step 2 are above 1.25 mm.

Deliverable: three tables and the storage estimates.

Acceptance criteria: arithmetic correct; the new cohort count is derived, not asserted; the storage units are labelled.

### Exercise 4 — De-identification audit (application, 60 minutes)

Run the pydicom audit script on a public DICOM sample. Report the defect rates and write a remediation list ordered by risk.

Deliverable: audit output and a remediation list.

Acceptance criteria: no PHI displayed in the deliverable; defects categorized by type; remediation ordered by likelihood and impact.

### Exercise 5 — Phantom correction (calculation, 45 minutes)

Using the offsets in section 6.5, correct the scanner means and recompute the between-scanner spread before and after. Then explain why the threshold prevalence change is larger than the mean change.

Deliverable: a before-and-after table and a 200-word explanation.

Acceptance criteria: arithmetic correct; the explanation invokes the tail of the distribution, not a vague claim about noise.

### Exercise 6 — Imaging AI critique (critique, 60 minutes)

Find a published imaging AI paper with a external validation. Identify the acquisition distribution in training and validation, and whether scanner or protocol was reported. Write a 400-word critique.

Deliverable: a 400-word critique with two proposed analyses.

Acceptance criteria: the critique names the specific acquisition mismatch; each proposal is feasible with header data; no claim about author intent.

### Exercise 7 — Export protocol (design, 75 minutes)

Write a DICOMweb export protocol that a PACS administrator could execute: query, manifest, de-identification, audit, and delivery. Include the reproducibility artefacts.

Deliverable: a protocol document with commands or query parameters.

Acceptance criteria: every step is scriptable; the manifest and hash are specified; the audit has a sample size and a defect threshold.

### Exercise 8 — Compare formats (critique, 45 minutes)

Compare DICOM, NIfTI, and PNG for an imaging AI project on five criteria: metadata retention, quantitative fidelity, storage, tooling, and de-identification.

Deliverable: a five-row comparison table and a recommendation.

Acceptance criteria: the recommendation is use-case specific; the metadata loss of each conversion is named.

### Exercise 9 — Label provenance (design, 45 minutes)

For a task of your choice, define an imaging label from the image rather than the report, and describe how you would validate it against adjudicated reads.

Deliverable: a label definition and a validation plan with sample size and agreement metric.

Acceptance criteria: the label does not depend on workflow artefacts; the validation plan states the reference standard and the agreement statistic.

---

## 13. Self-check answers

**A1. What are the four levels of the DICOM hierarchy, and which identifier belongs to each?**
Patient (PatientID), study (StudyInstanceUID), series (SeriesInstanceUID), instance (SOPInstanceUID). The study is the exam, the series is a set of images with the same parameters, and the instance is one object.

**A2. Convert stored value 1100 with slope 1.0 and intercept −1024 to HU.**
1100 × 1.0 + (−1024) = 76 HU. For comparison, stored 1024 gives 0 HU, which is water, and stored 24 would give −1000 HU, which is air.

**A3. Why is HU conversion mandatory for CT and meaningless for MR?**
CT pixel values are defined relative to water and air through a linear rescale, so the converted value has a physical meaning that is comparable across scanners. MR pixel values are arbitrary and depend on coil, sequence, and scanner, so there is no absolute scale to convert to.

**A4. What is the difference between slice thickness and slice spacing?**
Thickness is the reconstructed slab width; spacing is the distance between slice centres. They differ for overlapping reconstructions, and using spacing as thickness (or the reverse) produces wrong volumes.

**A5. Name the three options in PS3.15 that address non-standard identifying information.**
Clean Pixel Data (burned-in annotations), Clean Graphics (graphics and overlays), and Clean Descriptors (free-text attributes). A fourth is Clean Structured Content, for structured reports.

**A6. What must be done to BurnedInAnnotation after cleaning pixel data?**
It must be set to NO. Setting it to NO without removing the text is a false assertion and is worse than leaving it YES, because it disables downstream checks.

**A7. Compute the storage for 6,372 studies at 220 MB each in TB and TiB.**
6,372 × 220 = 1,401,840 MB = 1.40 TB in decimal units. In binary units, 1,401,840 / 1,024 / 1,024 = 1.336 TiB.

**A8. A scanner measures a 60 HU phantom insert at 52.1 HU. What is the offset, and how do you correct a patient measurement of 48 HU?**
Offset = 52.1 − 60.0 = −7.9 HU. Corrected value = 48 − (−7.9) = 55.9 HU.

**A9. Why does a 7.9 HU offset change prevalence more than 7.9 percentage points near a threshold?**
Because prevalence near a threshold depends on the density of the distribution at that point. Shifting the mean moves a slice of the distribution across the threshold, and the size of that slice is the area under the density curve between the old and new threshold positions, which can be much larger than the mean shift divided by the range.

**A10. Why is burned-in annotation a de-identification problem that header cleaning does not solve?**
Because the text is rendered into the pixel values. Removing the header attributes does not change the pixels, so the name or date remains visible in the image. Only the Clean Pixel Data option, with actual pixel removal, addresses it.

**A11. Why are regenerated UIDs a reproducibility problem?**
Because linkage between exports depends on stable identifiers. If each export assigns new UIDs, a patient's studies cannot be linked across exports, and a cohort defined in one export cannot be reconstructed from another.

**A12. Rubric for exercise 7 (export protocol).**
A strong protocol specifies the DICOMweb query with parameters, the manifest format and hash, the de-identification profile and options, the audit sample size and defect threshold, the delivery format, and the storage of query and run date. A weak protocol says "export the studies and de-identify them" without parameters, artefacts, or audit.

---

## 14. Interview questions

**Q1. What is DICOM and why does it matter beyond the file format?**
A strong answer: it is a standard for both pixel data and metadata, and the metadata is what makes images comparable. Without the header, two images of the same anatomy cannot be compared, and a model trained on one acquisition will fail on another.

**Q2. How would you build an imaging cohort that is comparable across scanners?**
A strong answer: classify series by acquisition parameters from the header, restrict to a defined thickness and phase, record the kernel, test feature distributions by scanner, calibrate with a phantom where available, and report the acquisition distribution.

**Q3. What are the three most common de-identification failures in imaging?**
Burned-in annotations, free-text descriptors containing names or dates, and private tags containing institution or operator information. Standard identity attributes are usually handled by tooling.

**Q4. How do you handle dates in an imaging dataset you want to share?**
A strong answer: shift dates consistently per patient with a keyed offset, so intervals are preserved and identity is not, and record the shift method in the de-identification method code sequence. Weak answers remove dates entirely, which destroys longitudinal analysis.

**Q5. Why is a single-site validation insufficient for an imaging model?**
Because acquisition heterogeneity and population differences are large. A model can learn scanner-specific texture and appear excellent internally. External validation on a different site and scanner is the minimum evidence of generalization.

**Q6. What is a phantom and why would you use one?**
A physical object with known properties, scanned to measure scanner bias and drift. It converts a relative difference between scanners into a measurable offset that can be corrected.

**Q7. What is the difference between a DICOM SEG and a NIfTI mask?**
SEG is a DICOM object that carries references to its source images and DICOM metadata; NIfTI is a standalone array format with minimal metadata. SEG preserves provenance; NIfTI is easier to use and easier to orphan.

**Q8. How would you query a PACS for a research cohort reproducibly?**
A strong answer: use DICOMweb QIDO-RS with explicit query parameters, store the query and the run date, save a manifest of returned UIDs, and hash it. Weak answers describe a manual export.

**Q9. What does the Clean Descriptors option do?**
It handles identifying information in free-text attributes such as Study Description, by retaining only values known to be safe or by cleaning them, with the method documented in the conformance statement.

**Q10. What is label leakage in imaging AI?**
A label that encodes information unavailable at deployment, such as the ordering protocol or a report phrase that reflects workflow rather than disease. The model appears accurate and fails when the workflow changes.

**Q11. Why is MR harder than CT for quantitative analysis?**
Because there is no absolute intensity scale. Pixel values depend on coil, sequence, field strength, and scanner, so cross-site comparison requires normalization, and every normalization method has assumptions.

**Q12. What would you put in an imaging data provenance record?**
Query and endpoint, export date, archive version, manifest hash, acquisition distribution, de-identification profile and options, audit sample and defect rate, and every post-processing step with its parameters.

**Q13. You are handed a folder of PNG images and asked to train a model. What do you ask for?**
A strong answer: the original DICOM series, because PNG conversion loses rescale values, spatial calibration, acquisition parameters, and the ability to audit de-identification. If the DICOM is unavailable, ask what conversion was applied and whether any sidecar metadata exists, and treat the analysis as restricted to display-space features.

**Q14. What is the first check you run on a new imaging dataset?**
A strong answer: a header audit for acquisition attributes, identity attributes, and the burned-in annotation flag, plus an instance count per series to check completeness. Those three checks take minutes and determine whether the dataset is analyzable and shareable.

---

## 15. Key terms

- **Accession number** — the order number for a study.
- **Burned-in annotation** — text rendered into pixel data.
- **Convolution kernel** — the reconstruction filter.
- **CTDIvol** — volume CT dose index.
- **DICOM** — Digital Imaging and Communications in Medicine.
- **DICOMweb** — RESTful DICOM services.
- **DIMSE** — the older DICOM network services.
- **DLP** — dose-length product.
- **HU** — Hounsfield unit.
- **Instance** — one DICOM object.
- **IOD** — information object definition.
- **Manifest** — the list of included objects.
- **PACS** — picture archiving and communication system.
- **Photometric interpretation** — the mapping of pixel values to channels.
- **Pixel spacing** — physical pixel size in the image plane.
- **Private tag** — a non-standard vendor attribute.
- **Protocol name** — the scanner protocol label.
- **QIDO-RS** — DICOMweb query service.
- **Rescale intercept** — offset for physical unit conversion.
- **Rescale slope** — multiplier for physical unit conversion.
- **RTSTRUCT** — radiation therapy structure set.
- **SEG** — DICOM segmentation object.
- **Series** — a set of instances with the same parameters.
- **Slice thickness** — reconstructed slab width.
- **SOP class** — service-object pair class.
- **SOP instance** — a single DICOM object.
- **Structured report** — a DICOM object encoding findings.
- **Study** — a set of series for one purpose.
- **TCIA** — The Cancer Imaging Archive.
- **Transfer syntax** — pixel data encoding.
- **UID** — unique identifier.
- **VNA** — vendor neutral archive.
- **WADO-RS** — DICOMweb retrieval service.
- **Window center and width** — display mapping.

---

## 16. Sources

All URLs accessed 24 October 2026.

**Standards**

- NEMA. *DICOM PS3.15 2025d, Security and System Management Profiles, Annex E: Attribute Confidentiality Profiles.* https://dicom.nema.org/medical/dicom/2025d/output/chtml/part15/chapter_E.html
- NEMA. *DICOM PS3.15, Clean Pixel Data Option (E.3.1).* https://dicom.nema.org/medical/dicom/current/output/chtml/part15/sect_E.3.1.html
- NEMA. *DICOM PS3.15, Clean Graphics Option (E.3.3).* https://dicom.nema.org/medical/dicom/current/output/chtml/part15/sect_E.3.3.html
- NEMA. *DICOM PS3.15, Clean Descriptors Option (E.3.5).* https://dicom.nema.org/medical/dicom/current/output/chtml/part15/sect_e.3.5.html
- NEMA. *DICOM Standard, current edition.* https://www.dicomstandard.org/
- HL7 International. *FHIR Release 5 (5.0.0)*, ImagingStudy resource. https://hl7.org/fhir/

**Regulators**

- HHS Office for Civil Rights. *Guidance Regarding Methods for De-identification of Protected Health Information.* https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
- eCFR. 45 CFR 164.514. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514
- US Food and Drug Administration. *Digital Health Center of Excellence.* https://www.fda.gov/medical-devices/digital-health-center-excellence
- European Union. *Regulation (EU) 2017/745 on medical devices.* https://eur-lex.europa.eu/eli/reg/2017/745/oj

**Datasets and evidence**

- Prior, F., et al. "The public cancer radiology imaging collections of The Cancer Imaging Archive." *Scientific Data*, 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5827108/ (background, pre-2024, for the 30.9 million image figure)
- The Cancer Imaging Archive. *Browse Data.* https://www.cancerimagingarchive.net/browse-data/
- The Cancer Imaging Archive. *Access the Data.* https://www.cancerimagingarchive.net/access-data/
- The Cancer Imaging Archive. *Updates, March 2024.* https://groups.google.com/g/tcia-announcements/c/liRYRDsdoZ8
- The Cancer Imaging Archive. *TCIA: An Information Resource to Enable Open Science.* https://pmc.ncbi.nlm.nih.gov/articles/PMC4257783/ (background)

> **Note:** The live DICOM standard has advanced past the 2025d edition; the "current" pages resolve to a later edition (reported as 2026b at the time of access). The clean descriptor and burned-in annotation requirements are substantively unchanged between editions for the purposes cited here.

---

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_registries-and-public-health.md`](03_registries-and-public-health.md)
- Next topic: [`05_signals-and-wearables.md`](05_signals-and-wearables.md)
- Related sibling: [`01_ehr-data.md`](01_ehr-data.md)
- Related sibling: [`06_omics-data.md`](06_omics-data.md)
- Related sibling: [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md)
- Root survey file: [`../../02_biomedical_data_landscape.md`](../../02_biomedical_data_landscape.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 2.3 in the survey file (DICOM metadata audit) is the practical companion to sections 5 and 7 of this topic.

Related reading: [`05_signals-and-wearables.md`](05_signals-and-wearables.md) covers the other high-frequency data type and shares the provenance problem; [`06_omics-data.md`](06_omics-data.md) covers another domain where batch effects dominate; and [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md) provides the formal vocabulary for the comparability checks in section 5.

The single sentence to remember: the pixel data shows the anatomy, and the header decides whether the anatomy is comparable.

A second sentence, for the privacy side: the header also decides whether the anatomy can be shared, and the pixels can override the header.
