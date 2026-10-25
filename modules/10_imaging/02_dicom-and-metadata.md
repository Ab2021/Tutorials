# DICOM and Metadata

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01
> Version 2026.3

## 1. Why this topic exists

Every imaging AI project begins with a folder of files that someone calls "the data". The folder
usually contains more than images: localizers, scouts, derived reconstructions, dose reports,
structured reports, presentation states, and occasionally a PDF. The files carry patient
identifiers, scanner identifiers, acquisition parameters, geometry, and free-text descriptions.
Some of those fields are needed to build a correct volume. Some are needed to prove the volume is
correct. Some are labels in disguise.

DICOM, the Digital Imaging and Communications in Medicine standard, is the encoding that holds
all of this together. It is not a file format in the way that PNG is a file format. It is an
object model with a data dictionary, a set of information object definitions (IODs), a transfer
syntax negotiation, and a conformance statement. A DICOM file can be a single image, a
multi-frame object containing hundreds of images, a structured report, a segmentation object, or
a presentation state. The pixel data may be uncompressed, JPEG-compressed, JPEG 2000, or
high-throughput JPEG 2000. The metadata may be complete, partially populated, or actively
misleading.

The industry problem is that metadata errors are silent. A series with a missing rescale
intercept produces images that look plausible and are wrong by 1024 Hounsfield units (HU). A
series with non-uniform slice spacing produces a volume that looks smooth and is geometrically
distorted. A duplicated series produces a volume with twice as many slices and a lesion measured
at the wrong size. A private tag carrying the report impression produces a model with excellent
area under the receiver operating characteristic curve (AUROC) and no clinical value. None of
these failures raises an exception in a standard training script.

This topic teaches the data contract. By the end, you should be able to take an arbitrary DICOM
folder, decide which objects belong in a volume, compute the geometry from the tags rather than
from assumptions, decode the pixel values into a known unit, remove identifiers without
destroying the evidence needed for audit, and detect the metadata fields that leak the label.

### 1.1 The decisions this topic unblocks

| Decision | What DICOM knowledge provides |
|---|---|
| Which files form one volume | Study, series, and instance identifiers plus geometry |
| Which series is the diagnostic one | SOP class, series description, slice count, contrast, kernel |
| Whether the volume is geometrically valid | Position, orientation, spacing, and direction cosines |
| Whether the pixel values are in HU | Rescale slope, intercept, and type |
| Whether the object is 2D, 3D, or multi-frame | NumberOfFrames and per-frame functional groups |
| Whether the data are de-identified | PS3.15 profile, private tags, burned-in annotation |
| Whether metadata leak the label | Descriptions, private tags, laterality, acquisition parameters |
| Whether a model can be reproduced | Transfer syntax, decode path, and preprocessing version |

### 1.2 What "metadata" means here

Metadata is every attribute that is not the pixel matrix. It includes identity (names, identifiers,
dates), acquisition (modality, kVp, mAs, contrast, tracer), geometry (pixel spacing, slice
thickness, image position, orientation), encoding (transfer syntax, bit depth, photometric
interpretation), interpretation (series description, protocol name, study description), and
private vendor extensions. Each category has a different reliability and a different de-identification
rule. Treating metadata as one blob is the root cause of most ingestion bugs.

## 2. Learning outcomes

By the end of this topic you can:

1. **Parse** a DICOM object's file meta information and dataset, and identify its transfer syntax
   and SOP class.
2. **Group** objects into patients, studies, and series using the correct unique identifiers.
3. **Compute** slice spacing and orientation from `ImagePositionPatient` and
   `ImageOrientationPatient`, and explain why `SliceThickness` is not the spacing.
4. **Convert** stored pixel values to HU using rescale slope and intercept, and detect when the
   conversion is not possible.
5. **Handle** multi-frame and enhanced objects by reading per-frame functional groups.
6. **De-identify** a dataset against a stated profile and audit the result for residual identifiers
   and burned-in text.
7. **Detect** metadata that leaks the label, including series descriptions, private tags, and
   laterality markers.
8. **Write** an ingestion manifest with one row per series and a QC status that a reviewer can
   audit.

## 3. The core mental model

DICOM has two hierarchies that must be kept separate: the information hierarchy and the encoding
hierarchy.

### 3.1 The information hierarchy

```text
Patient
  └── Study            StudyInstanceUID        one order, one accession
        └── Series     SeriesInstanceUID       one acquisition or reconstruction
              └── Instance   SOPInstanceUID   one object (2D image, multiframe, SR, SEG)
                    └── Frame              one 2D plane inside a multiframe object
```

The identifiers are globally unique and must never be regenerated inconsistently. A study can
contain several series: a scout, an unenhanced series, an arterial series, a portal venous
series, a delayed series, and a dose report. A series can contain one instance or hundreds. A
single instance can contain one frame or thousands.

The practical rule is: **the series is the unit of acquisition and reconstruction; the instance
is the unit of storage; the frame is the unit of image content.** Most ingestion bugs come from
confusing these three.

### 3.2 The encoding hierarchy

```text
File
  ├── 128-byte preamble (optional)
  ├── DICM prefix
  ├── File Meta Information (group 0002, always explicit VR little endian)
  │     ├── TransferSyntaxUID
  │     ├── MediaStorageSOPClassUID
  │     └── MediaStorageSOPInstanceUID
  └── Dataset
        ├── (0008,xxxx) identification
        ├── (0018,xxxx) acquisition
        ├── (0020,xxxx) relationship and geometry
        ├── (0028,xxxx) image presentation and pixel description
        ├── (7FE0,0010) Pixel Data
        └── (gggg,xxxx) private tags
```

The file meta information tells you how to decode the dataset. The dataset tells you what the
object means. If you decode the dataset with the wrong transfer syntax, you get plausible-looking
garbage or an immediate parse failure. If you decode it correctly but ignore the rescale, you get
plausible-looking numbers in the wrong unit.

### 3.3 The geometry model

A voxel index `(i, j, k)` maps to a patient coordinate `(x, y, z)` through an affine matrix:

```text
[x]   [ row_dir_x * Δi   col_dir_x * Δj   normal_x * Δk   origin_x ] [i]
[y] = [ row_dir_y * Δi   col_dir_y * Δj   normal_y * Δk   origin_y ] [j]
[z]   [ row_dir_z * Δi   col_dir_z * Δj   normal_z * Δk   origin_z ] [k]
```

`ImageOrientationPatient` supplies the row and column direction cosines. Their cross product is
the slice normal. `ImagePositionPatient` supplies the origin of each slice. The spacing along the
normal is the projection of the difference between two consecutive positions onto the normal.
`PixelSpacing` supplies the in-plane distances. `SliceThickness` is a nominal reconstruction
width, not the distance the table moved.

### 3.4 The trust hierarchy

Not all metadata is equally trustworthy.

| Layer | Example | Trust | Why |
|---|---|---|---|
| File meta | TransferSyntaxUID | High | Required for decoding; parser checks it |
| Identity | PatientID | Medium | Often replaced or inconsistent across sites |
| Geometry | ImagePositionPatient | High if present | Directly describes the acquisition |
| Geometry | SliceThickness | Medium | Nominal, may not equal spacing |
| Encoding | RescaleSlope/Intercept | High for CT | Required for HU; absence is a defect |
| Description | SeriesDescription | Low | Free text, site-specific, often leaks |
| Private | Vendor tags | Low | Undocumented, may contain identifiers or labels |

The trust hierarchy determines the ingestion policy: derive geometry from positions, derive units
from rescale, never derive labels from descriptions.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| DICOM | Standard for imaging object encoding and communication | Calling any image file "DICOM" | PS3 current edition 2026d |
| SOP class | Service-object pair class defining the object type | Treating all SOP classes as images | UID, e.g. CT Image Storage |
| SOP instance UID | Unique identifier of one object | Regenerating per export | UID |
| Study instance UID | Unique identifier of one study | Reusing across patients | UID |
| Series instance UID | Unique identifier of one series | Merging series by description | UID |
| Transfer syntax | Encoding rules for the dataset | Ignoring compressed vs uncompressed | UID |
| Explicit VR | Value representation is written in the stream | Assuming all files are explicit | Encoding |
| Implicit VR | Value representation comes from the dictionary | Parsing explicit-only | Encoding |
| Little endian | Byte order | Ignoring on big-endian archives | Encoding |
| Pixel data | The stored image values | Treating as display values | Element (7FE0,0010) |
| Pixel representation | Signed or unsigned integer | Decoding signed CT as unsigned | 0 unsigned, 1 signed |
| Bits allocated | Bits per stored sample | Confusing with bits stored | bit |
| Bits stored | Meaningful bits per sample | Using allocated instead | bit |
| High bit | Position of the most significant bit | Ignoring for 12-bit data | bit index |
| Photometric interpretation | Colour or greyscale convention | Ignoring MONOCHROME1 inversion | Code string |
| Rescale slope | Multiplier to physical units | Assuming 1 | Dimensionless |
| Rescale intercept | Offset to physical units | Assuming 0 | HU for CT |
| Rescale type | Unit label for rescaled values | Ignoring | Code string, e.g. HU |
| Pixel spacing | Row and column spacing | Confusing with slice thickness | mm |
| Slice thickness | Nominal slice width | Using as through-plane spacing | mm |
| Image position patient | Centre of the first voxel in patient coordinates | Using as corner | mm, LPS |
| Image orientation patient | Row and column direction cosines | Treating as angles | Unit vector |
| Slice normal | Cross product of row and column directions | Assuming axial | Unit vector |
| Frame of reference UID | Coordinate system shared across series | Ignoring in registration | UID |
| Number of frames | Frames in one object | Treating as one image | count |
| Per-frame functional groups | Per-frame geometry and acquisition metadata | Reading only top-level tags | Sequence |
| Private tag | Vendor-defined attribute | Assuming it is safe or useless | Odd group number |
| Burned-in annotation | Text or graphics in pixel data | Assuming de-id removed it | Pixels |
| De-identification | Removal or replacement of identifiers | Equating with anonymization | Process |
| PS3.15 | DICOM security and confidentiality profiles | Ignoring action codes | Standard part |
| Basic profile | Conservative de-identification profile | Applying without options | Profile |
| Clean pixel data | Option to remove burned-in text | Skipping OCR audit | Option |
| DICOMweb | RESTful services for query, retrieve, store | Confusing with DIMSE | PS3.18 |
| QIDO-RS | Query based on DICOM | Using for bulk export | Service |
| WADO-RS | Retrieve DICOM objects | Assuming rendered output | Service |
| STOW-RS | Store DICOM objects | Sending non-DICOM | Service |
| DIMSE | Traditional DICOM network services | Assuming firewall-friendly | Protocol |
| PACS | Picture archiving and communication system | Treating as a database | System |
| VNA | Vendor-neutral archive | Treating as a PACS replacement | System |
| Accession number | Order identifier | Using as patient identifier | String |
| Study description | Free-text study label | Using as a label source | Free text |
| Series description | Free-text series label | Using for protocol logic | Free text |

### 4.1 The tags that decide whether a volume is valid

| Tag | Name | Why it matters | Failure if wrong |
|---|---|---|---|
| (0008,0016) | SOPClassUID | Object type | Pixel data misread |
| (0008,0018) | SOPInstanceUID | Object identity | Duplicates or collisions |
| (0020,000D) | StudyInstanceUID | Study grouping | Mixed patients |
| (0020,000E) | SeriesInstanceUID | Series grouping | Mixed reconstructions |
| (0020,0052) | FrameOfReferenceUID | Spatial frame | Misregistration |
| (0020,0032) | ImagePositionPatient | Slice origin | Wrong spacing and origin |
| (0020,0037) | ImageOrientationPatient | Row/column directions | Wrong orientation |
| (0028,0030) | PixelSpacing | In-plane geometry | Wrong size |
| (0018,0050) | SliceThickness | Nominal thickness | Wrong partial volume |
| (0018,0088) | SpacingBetweenSlices | Nominal gap | Wrong stacking |
| (0028,0008) | NumberOfFrames | Multi-frame count | Missing frames |
| (0028,0004) | PhotometricInterpretation | Colour convention | Inverted image |
| (0028,0100) | BitsAllocated | Storage width | Decode error |
| (0028,0101) | BitsStored | Meaningful bits | Value range error |
| (0028,0103) | PixelRepresentation | Signedness | Negative HU lost |
| (0028,1052) | RescaleIntercept | HU offset | 1024 HU error |
| (0028,1053) | RescaleSlope | HU scale | Contrast error |
| (0028,1054) | RescaleType | Unit label | Unit ambiguity |
| (0028,0301) | BurnedInAnnotation | Text in pixels | Identifier leak |
| (0018,0015) | BodyPartExamined | Anatomy | Wrong protocol stratum |
| (0018,0080) | RepetitionTime | MRI sequence | Sequence confounding |
| (0018,0081) | EchoTime | MRI sequence | Sequence confounding |
| (0008,1030) | StudyDescription | Free text | Label leak |
| (0008,103E) | SeriesDescription | Free text | Label leak |
| (0018,1030) | ProtocolName | Free text | Label leak |
| (0010,0040) | PatientSex | Demographics | Subgroup analysis |
| (0010,1010) | PatientAge | Demographics | Subgroup analysis |
| (0008,0070) | Manufacturer | Scanner | Site signature |
| (0008,1090) | ManufacturerModelName | Scanner | Site signature |
| (0018,1020) | SoftwareVersions | Software | Drift detection |

### 4.2 Action codes in the de-identification profile

DICOM PS3.15 Annex E uses action codes to say what to do with each attribute.

| Code | Meaning | Typical use |
|---|---|---|
| D | Replace with a non-zero dummy value | Dates, times, identifiers |
| Z | Zero-length | Attributes that must be present but empty |
| X | Remove | Attributes not needed |
| K | Keep | Attributes safe to retain |
| C | Clean | Free text that may contain identifiers |
| U | Replace with a consistently remapped UID | UIDs |
| Z/D | Zero or dummy depending on context | Conditional |
| X/Z | Remove or zero depending on context | Conditional |
| X/D | Remove or dummy depending on context | Conditional |
| X/Z/D | Remove, zero, or dummy | Conditional |
| X/Z/U* | Remove, zero, or remap | Conditional |

The action codes are not suggestions. A de-identification tool that claims PS3.15 Basic Profile
conformance must apply the table, set `PatientIdentityRemoved` (0012,0062) to YES, populate the
de-identification method attributes, and document any options used.

## 5. The mechanics, step by step

This procedure takes a directory of DICOM files and produces a validated volume plus a manifest
row. It assumes no prior knowledge of the folder structure. Each step states input, output, and
the reason it exists.

### Step 1. Inventory the files and classify by SOP class

**Input:** A directory tree of files with unknown extensions.

**Output:** A table with one row per file: path, file size, whether the DICM prefix is present,
MediaStorageSOPClassUID, and parse status.

**Rationale:** Not every file is an image. A study folder may contain Structured Report Storage,
Presentation State Storage, Segmentation Storage, Radiation Dose Structured Report, Key Object
Selection, and Encapsulated PDF. Including a dose report in a pixel array is a type error, not a
minor inconvenience.

**Branch points:**
- If the file has no DICM prefix, it may be a raw dataset or a non-DICOM file. Try a header
  sniff; if it still fails, quarantine it.
- If the SOP class is an image storage class, continue.
- If the SOP class is Segmentation or Structured Report, route it to the label pipeline, not the
  image pipeline.

**Edge case:** Some archives strip the 128-byte preamble and DICM prefix. The dataset may still
parse; do not reject solely on the prefix.

### Step 2. Read the file meta information and transfer syntax

**Input:** The first bytes of each file.

**Output:** TransferSyntaxUID, MediaStorageSOPClassUID, MediaStorageSOPInstanceUID, and a decoder
choice.

**Rationale:** The transfer syntax determines whether pixel data are native, encapsulated, or
compressed. A JPEG 2000 file decoded as uncompressed produces noise. A deflated file decoded
without inflate fails.

**Branch points:**
- Native uncompressed little endian: direct decode.
- Native uncompressed big endian: byte swap.
- Encapsulated JPEG/JPEG-LS/JPEG 2000: use a decoder that supports the specific transfer syntax.
- Deflated: inflate the dataset before parsing.

**Edge case:** A file may declare one transfer syntax in the meta information and contain data
encoded with another. The decoder should validate the first frame and fail loudly rather than
return a plausible array.

### Step 3. Group objects into patients, studies, and series

**Input:** The parsed identifiers.

**Output:** A hierarchy with counts, plus a report of identifier collisions.

**Rationale:** Grouping by folder name is unreliable. Grouping by `SeriesDescription` merges
different reconstructions. The UIDs are the only stable grouping keys.

**Branch points:**
- Same `PatientID` with different `PatientName`: possible identifier collision or name change.
- Same `StudyInstanceUID` with different `PatientID`: corrupt or merged study.
- Same `SeriesInstanceUID` with different geometry: corrupt series or illegal reuse.

**Edge case:** De-identified datasets may have new UIDs generated per export. Two exports of the
same study can then have different UIDs. The manifest must record the source and the export
version.

### Step 4. Select the diagnostic series

**Input:** Series-level summary: SOP class, description, slice count, contrast, kernel, slice
thickness, and body part.

**Output:** One selected series per study, or an exclusion reason.

**Rationale:** A CT study may contain a scout, several reconstructions, and a dose report. The
model's intended use names a reconstruction, not a study. Selecting the wrong series is a
common cause of apparent model failure.

**Branch points:** see the decision table below.

**Decision table: series selection for a CT study**

| Candidate series | Evidence | Decision | Reason |
|---|---|---|---|
| Scout or localizer | 1-2 images, large field of view, low dose | Exclude | Not diagnostic geometry |
| Thin-section soft kernel | 1.25 mm, B30f, 300 slices | Include | Matches nodule protocol |
| Thin-section sharp kernel | 1.25 mm, B70f, 300 slices | Include as separate stratum | Different texture |
| Thick-section | 5 mm, 60 slices | Exclude or stratify | Partial volume differs |
| Contrast arterial | Contrast agent, arterial phase | Include only if intended use allows | Phase confounding |
| Dose report | RDSR SOP class | Exclude | Not an image |
| Derived 3D | MPR or volume rendering | Exclude | Not source data |
| Duplicate export | Same geometry, different UID | Deduplicate | Double counting |

**Edge case:** A study may have two thin-section series with the same description but different
kernels. Treat them as separate strata; do not average them.

### Step 5. Establish the patient coordinate system

**Input:** `ImageOrientationPatient`, `ImagePositionPatient`, `FrameOfReferenceUID`.

**Output:** Row direction, column direction, slice normal, origin, and a handedness check.

**Rationale:** The direction cosines define how voxel indices map to patient space. Without them,
left and right, anterior and posterior, and superior and inferior are guesses. A negative
determinant indicates a left-handed system and a likely mirror.

**Branch points:**
- Orthonormal directions and determinant +1: valid right-handed frame.
- Determinant -1: flag as mirrored; do not silently flip.
- Non-orthonormal directions: oblique or corrupt; recompute and flag.

**Edge case:** `ImageOrientationPatient` may be absent in older or secondary-capture objects.
If the SOP class requires it and it is missing, exclude the series.

### Step 6. Sort slices and compute spacing

**Input:** All positions in the selected series.

**Output:** Sorted slices, a spacing estimate, and a uniformity report.

**Rationale:** File order is not slice order. The position projected onto the normal gives the
true through-plane coordinate. The median of successive deltas is robust to a missing or
duplicated slice.

**Branch points:**
- Uniform deltas within tolerance: accept.
- Deltas alternate or vary: flag mixed spacing.
- Duplicate positions: remove duplicates before computing spacing.
- Negative direction: sort ascending along the normal; do not assume inferior-to-superior.

**Edge case:** Gantry tilt produces an in-plane shift between slices. The positions still give the
correct normal spacing, but a naive stack ignores the shift and shears the volume. The affine
must include the in-plane component.

### Step 7. Decode pixel data and apply rescale

**Input:** Pixel data, bit depth, signedness, photometric interpretation, slope, intercept.

**Output:** A floating-point array in HU (CT) or a documented unit, with shape and dtype recorded.

**Rationale:** The model consumes numbers, not files. The decode must preserve sign and range.
The rescale must be applied before windowing or normalization.

**Branch points:**
- Slope 1, intercept -1024: standard CT.
- Slope 1, intercept 0: verify against a water phantom before treating as HU.
- Slope absent: do not assume 1; flag the series.
- MONOCHROME1: invert for display; document whether the model input is inverted.
- RGB or YBR: convert to the model's expected colour space.

**Edge case:** A CT series may have `RescaleType` equal to "HU" but a slope of 1 and an intercept
that varies per slice. Use the per-slice or per-frame values, not the first one.

### Step 8. Handle multi-frame and enhanced objects

**Input:** `NumberOfFrames`, `PerFrameFunctionalGroupsSequence`, and the shared functional groups.

**Output:** A frame stack with per-frame position, orientation, and rescale where present.

**Rationale:** Enhanced CT and MR store many frames in one object. The top-level geometry may
describe only the first frame. Reading only the top level produces a stack with one position and
no spacing.

**Branch points:**
- Classic single-frame series: one position per object.
- Enhanced multi-frame: per-frame positions in the functional groups.
- Whole slide imaging: frames are tiles at one pyramid level; geometry is in the optical path
  and pixel spacing attributes.
- Ultrasound cine: frames are temporal, not spatial; do not stack as slices.

**Edge case:** Frame order in a multi-frame object is not guaranteed to be spatial order. Sort by
the per-frame position.

### Step 9. De-identify against a stated profile

**Input:** The dataset and a de-identification policy.

**Output:** A de-identified object, a record of the profile and options, and an audit log.

**Rationale:** De-identification is a process with a stated threat model, not a checkbox. The
PS3.15 Basic Profile is conservative but does not clean pixels by default. Dates can be shifted
consistently or removed. UIDs can be remapped or retained.

**Branch points:**
- Research dataset with longitudinal follow-up: retain longitudinal temporal information with
  consistent date shifting.
- Public release: basic profile plus clean pixel data plus clean descriptors.
- Internal training: basic profile plus retain device identity if scanner is a legitimate feature.

**Edge case:** Removing `PatientID` but retaining `AccessionNumber` or a private tag with the
medical record number does not de-identify the object. The audit must search all groups,
including private groups.

### Step 10. Scan for label leaks

**Input:** De-identified metadata and pixel data.

**Output:** A leak report with the field, the prevalence in positive and negative cases, and the
univariate association with the label.

**Rationale:** A field that predicts the label is a shortcut. The most common leaks are series
description, protocol name, study description, laterality, acquisition parameters, and private
tags. Burned-in annotation is a pixel-level leak that survives metadata de-identification.

**Branch points:**
- Categorical field with a strong association: remove or blind it.
- Continuous field with a strong association: stratify or residualize.
- Pixel text: crop, inpaint, or exclude, and verify the label does not depend on the text.

**Edge case:** A leak may be legitimate. Scanner model can be a real risk factor if the scanner
is used preferentially for sicker patients; that is confounding, not label leakage, and the fix
is site-held-out validation, not field removal.

### Step 11. Build the manifest and QC report

**Input:** All prior steps.

**Output:** One row per series with identifiers, geometry, unit, decode path, de-identification
status, leak flags, and a pass/fail QC status.

**Rationale:** The manifest is the reproducible record of what was included. It supports
debugging, audit, and regulatory review. A model without a manifest cannot be re-evaluated on the
same data.

**Branch points:**
- Pass: all required fields present and consistent.
- Pass with flags: usable with documented caveats.
- Fail: missing unit, non-uniform spacing, unreadable pixels, unresolved identifier.

**Edge case:** A series can pass geometry and fail de-identification. The manifest must record
both dimensions separately; a single "valid" flag hides the distinction.

### Step 12. Freeze the ingestion specification

**Input:** The manifest and the code version.

**Output:** A versioned data specification with a hash of the manifest and the code commit.

**Rationale:** Ingestion changes silently when a new site is added or a library is upgraded. The
version identifier makes the change visible and forces re-evaluation.

**Branch points:**
- New site with a new protocol: new specification version and a protocol amendment.
- Library upgrade that changes decode: regression test on a fixed sample.
- De-identification policy change: new version and a re-audit.

**Edge case:** A pydicom upgrade can change default decoding behavior. pydicom 3.0, released
8 September 2024, changed transfer syntax handling, default encoding on save, and YCbCr-to-RGB
conversion. Pin the version and test the decode path.

### 5.1 Decision table: what to do with an inconsistent series

| Condition | More than two outcomes | Action |
|---|---|---|
| Missing `PixelSpacing` | (a) DX with implied spacing, (b) US without calibration, (c) CT | (a) use modality implied value and flag, (b) exclude from size tasks, (c) fail |
| Missing `ImagePositionPatient` | (a) single-frame classic, (b) multiframe enhanced, (c) secondary capture | (a) fail for 3D, (b) read per-frame, (c) exclude |
| Non-uniform position deltas | (a) gantry tilt, (b) mixed thickness, (c) missing slice | (a) model affine shear, (b) split by delta, (c) interpolate only with flag |
| Duplicate positions | (a) duplicate export, (b) repeated acquisition, (c) corrupt | (a) deduplicate, (b) keep both as timepoints, (c) fail |
| Rescale absent | (a) CT with vendor constant, (b) CT unknown, (c) MR | (a) apply documented constant, (b) fail, (c) not applicable |
| `BurnedInAnnotation` = YES | (a) laterality marker, (b) measurement, (c) report text | (a) keep only if label does not depend, (b) crop, (c) exclude or clean |
| Laterality mismatch | (a) metadata error, (b) marker error, (c) situs inversus | (a) correct from report, (b) exclude, (c) clinical review |

### 5.2 Edge cases that require an explicit rule

**Mixed slice thickness.** Two reconstructions may share a series description and a folder. Rule:
group by `SeriesInstanceUID` and by position-delta mode; never concatenate before checking.

**Missing series.** The required contrast phase may be absent. Rule: the study is incomplete, not
negative. Exclude and record the reason.

**Inconsistent spacing.** The table may move by 1.25 mm but the reconstruction may be 1.0 mm with
0.25 mm overlap. Rule: use position delta for geometry and record both delta and thickness.

**Rescale slope/intercept absent.** Rule: no unit, no inclusion for HU-based tasks.

**Burned-in annotation.** Rule: OCR sample, visual audit, and a documented decision. Do not assume
that de-identification removed pixels.

**Laterality mismatch.** Rule: mismatch is a data-quality event that requires human resolution.

**Multi-frame objects.** Rule: parse per-frame groups; sort by position; do not trust frame index.

**Private tags carrying labels.** Rule: remove all private tags by default; if a private tag is
retained for a documented reason, test it for label association.

### 5.3 Branch procedure: mixed slice thickness in one folder

Mixed thickness is common when a scanner exports both a thin and a thick reconstruction into
the same study, and it is the most expensive ingestion bug because the result looks like a
larger, higher-resolution volume.

**Input:** A folder with two or more position-delta modes.

**Output:** Either two validated sub-series or one validated series plus one exclusion.

1. Cluster the slices by `SeriesInstanceUID`. If the UIDs differ, treat them as separate series
   and stop; the folder is not a series.
2. If the UIDs are the same, compute the sorted position deltas along the normal.
3. Build a histogram of deltas rounded to 0.01 mm. A single mode within tolerance is uniform. Two
   modes separated by more than 0.2 mm indicate mixed thickness.
4. Split the slices at the mode boundary. Check that each subset forms a contiguous position
   range; a subset with gaps is not a valid series.
5. For each subset, recompute spacing, slice count, and volume extent. Record the mode, the
   count, and the extent in the manifest.
6. Choose the subset that matches the intended use. If the intended use allows both, keep both as
   separate strata and never concatenate them.
7. If neither subset is contiguous, exclude the folder and record "mixed thickness, non-contiguous".

**Why not interpolate the whole stack?** Interpolation assumes a single sampling grid. Fitting one
grid to two modes either duplicates slices (if the fine grid is used) or discards thin slices (if
the coarse grid is used). Both change the evidence. Splitting preserves the original sampling of
each reconstruction.

**Edge case:** A series may have a single outlier delta caused by one missing slice. The histogram
then has a dominant mode and one large delta. The correct action is to detect the gap, insert a
not-a-number slice if the model supports masking, or exclude the series; do not treat the outlier
as a second reconstruction.

**Edge case:** A gantry tilt produces a constant in-plane shift per slice but a uniform normal
delta. The histogram is unimodal, so the mixed-thickness test passes. The affine must include the
shift; Section 6.10 shows the arithmetic for the uncorrected case.

## 6. Worked example

### 6.1 The case

A hospital sends a CT abdomen study for a liver lesion model. The folder contains 227 files: 180
thin-section slices, 45 thick-section slices from a second reconstruction, one dose report, and
one secondary-capture image. All files have the same `StudyInstanceUID`. The thin and thick
series have different `SeriesInstanceUID` values but the same `SeriesDescription`,
"LIVER LESION LEFT".

The thin series has 512 × 512 pixels, `PixelSpacing` 0.703125 mm, `SliceThickness` 1.25 mm,
positions from -112.5 mm to 111.25 mm in 1.25 mm steps, slope 1, intercept -1024. The lesion is
measured at 22.0 mm × 18.0 mm × 16.0 mm.

### 6.2 Geometry and volume

In-plane pixel area:

```text
0.703125 × 0.703125 = 0.494384765625 mm²
```

Voxel volume at 1.25 mm:

```text
0.494384765625 × 1.25 = 0.61798095703125 mm³
```

Ellipsoid volume with semi-axes 11.0, 9.0, and 8.0 mm:

```text
V = (4/3) × π × 11.0 × 9.0 × 8.0
  = 4.18879 × 792
  = 3317.52 mm³
```

Voxel count at the acquired spacing:

```text
3317.52 / 0.61798 = 5368.3 voxels
```

### 6.3 Slice spacing from positions

The nominal thickness is 1.25 mm. The position delta is:

```text
(111.25 - (-112.5)) / (180 - 1) = 223.75 / 179 = 1.2500 mm
```

The nominal and measured values agree to four decimal places, so the series is internally
consistent. If the delta had been 1.0 mm with a thickness of 1.25 mm, the slices would overlap by
0.25 mm; the volume would be valid only if the overlap were modelled in the affine.

### 6.4 HU and window

The lesion region of interest has a mean stored value of 1100. With slope 1 and intercept -1024:

```text
HU = 1100 × 1 + (-1024) = 76 HU
```

The liver window is centre 60 HU and width 160 HU:

```text
lower = 60 - 80 = -20 HU
upper = 60 + 80 = 140 HU
(76 - (-20)) / 160 = 96 / 160 = 0.600
```

The lesion maps to 0.600 of the display range, or 153 of 255. The soft-tissue window (40/400)
maps it to (76 + 160) / 400 = 0.590, or 150 of 255. The two windows look similar for this lesion;
they differ much more for fat and air.

### 6.5 Segmentation agreement and detection

The reference mask has a volume of 3317.5 mm³. The model mask has 2980 mm³ and the intersection is
2700 mm³:

```text
Dice = 2 × 2700 / (3317.5 + 2980)
     = 5400 / 6297.5
     = 0.858
```

Dice is a voxel-level metric on one lesion; the reference standard is the radiologist's edited
mask. On the detection task, 198 of 220 lesions were detected at an operating point of 0.5 false
positives per scan:

```text
sensitivity = 198 / 220 = 0.900
false positives = 0.5 per scan × 220 scans = 110
```

### 6.6 Diagnostic 1: duplicate series concatenation

A naive ingestion script lists all files, sorts by `InstanceNumber`, and stacks them. The thin
series has 180 slices and the thick series has 45. The combined stack has 225 slices. Sorting by
position produces many duplicate z values: the thick series covers the same 220 mm extent at
5 mm intervals, so its 45 positions coincide with 45 of the thin positions, and the intervening
thin slices fill the gaps. The position-delta distribution becomes bimodal: 1.25 mm for most
steps and 5.0 mm at the transitions.

The pipeline reports a median spacing of 1.25 mm and builds a 225-slice volume. The lesion is
counted twice where the thick slices overlap, and the volume is interpolated across the 5 mm gaps.
The measured lesion volume increases from 3317.5 mm³ to 3710 mm³, a 11.8% inflation that looks
like a segmentation improvement. The QC diagnostic is the duplicate-position count: 45 duplicate
positions and a delta interquartile range of 3.75 mm. A valid single series has neither.

### 6.7 Diagnostic 2: label leak in the series description

The `SeriesDescription` is "LIVER LESION LEFT" for 176 of 180 positive studies (97.8%) and for 34
of 220 negative studies (15.5%). A text-only classifier using the description reaches AUROC 0.91
(unit: study; reference: lesion label). The image model's AUROC of 0.88 is therefore not evidence
that the image contains the signal. The pipeline must remove the description from the model input
and re-evaluate; the honest image-only AUROC is the number that can be deployed.

A second leak is in a private tag from the ultrasound vendor that contains the referring
indication. The tag is in group 0x0019, which is private, and it survives a de-identification
script that only removes group 0x0010. The leak audit must scan private groups.

### 6.8 Varying one assumption

Assume the rescale intercept is missing and the pipeline assumes 0. The stored mean of 1100 is
then interpreted as 1100 HU. The liver window upper bound is 140 HU, so the lesion and the liver
parenchyma both clip to 1.0. The lesion becomes indistinguishable from background on the window,
and any HU threshold above 140 HU behaves identically. A model trained on correctly rescaled HU
sees a different input distribution: the deployment values are shifted by +1024 HU, and the
window is saturated.

The conclusion changes from "76 HU, hypoattenuating lesion, segmentable on liver window" to
"saturated, uninterpretable, exclude the series". The fix is not a better model; it is a unit
assertion in the ingestion pipeline.

### 6.9 What the manifest records for this case

A manifest row is not a log line. It is the evidence that the series was handled correctly. For
the thin series, the row contains:

| Field | Value | Why it is recorded |
|---|---|---|
| StudyInstanceUID | remapped UID | Join key across exports |
| SeriesInstanceUID | remapped UID | Grouping and deduplication |
| SOPClassUID | CT Image Storage | Routing |
| SeriesDescription | blinded | Leak control |
| Rows × Columns | 512 × 512 | Geometry |
| PixelSpacing | 0.703125 × 0.703125 mm | Geometry |
| SliceThickness | 1.25 mm | Partial-volume context |
| PositionDelta | 1.2500 mm | Geometric spacing |
| DeltaIQR | 0.000 mm | Uniformity test |
| Rescale | slope 1, intercept -1024, type HU | Unit assertion |
| TransferSyntax | explicit VR little endian | Decode reproducibility |
| BurnedInAnnotation | NO | Leak audit |
| PrivateTagsRemoved | YES | De-identification audit |
| DuplicatePositions | 0 | QC |
| Status | PASS | Inclusion decision |

The thick series receives a separate row with `Status = EXCLUDED` and `Reason = thickness outside
intended use`. The dose report receives a row with `Status = EXCLUDED` and
`Reason = non-image SOP class`. The row count must equal the file inventory count minus files
that failed to parse, and the difference must be explained.

### 6.10 Varying a second assumption: gantry tilt

Assume the same thin series was acquired with a 15-degree gantry tilt. The normal spacing remains
1.25 mm, but each slice is shifted in-plane by:

```text
shift per slice = 1.25 mm × tan(15°)
                = 1.25 × 0.26795
                = 0.335 mm
```

Over 179 intervals, the total in-plane shift is:

```text
0.335 × 179 = 59.9 mm
```

A naive stack that ignores the shift builds a sheared volume. A lesion that is 16 mm in the
through-plane direction appears elongated by the factor:

```text
sqrt(1 + (0.335 / 1.25)²) = sqrt(1 + 0.0718) = 1.035
```

so the measured long axis becomes 16.0 × 1.035 = 16.56 mm, a 3.5% overestimate. The effect is
small for a single lesion but accumulates for length measurements along the table direction and
for registration to a non-tilted series. The correct affine includes the in-plane shift; the
incorrect one is not detectably wrong from a single slice.

The conclusion changes from "16.0 mm lesion, volume 3317.5 mm³" to "16.6 mm lesion, volume 3434
mm³" if the tilt is ignored. The fix is metadata, not modelling: read the orientation and position
of every slice, and build the affine from them.

## 7. Code, query, or calculation

The following code builds a minimal explicit-VR little-endian DICOM dataset in memory, parses it,
and computes the geometry. It uses only the standard library. It is deliberately small: the point
is to show that the tags, not the file name, carry the geometry.

```python
import struct

SEP = chr(92)  # DICOM multi-value separator
LONG_VRS = {"OB", "OW", "OF", "SQ", "UT", "UN"}

def make_element(group, element, vr, value):
    head = struct.pack("<HH", group, element) + vr.encode("ascii")
    if vr in LONG_VRS:
        return head + b"\x00\x00" + struct.pack("<I", len(value)) + value
    return head + struct.pack("<H", len(value)) + value

def ds(text):
    raw = text.encode("ascii")
    return raw + (b" " if len(raw) % 2 else b"")

def parse_explicit_le(data):
    tags, i = {}, 0
    while i < len(data):
        group, element = struct.unpack_from("<HH", data, i)
        i += 4
        vr = data[i:i + 2].decode("ascii")
        i += 2
        if vr in LONG_VRS:
            i += 2
            length = struct.unpack_from("<I", data, i)[0]
            i += 4
        else:
            length = struct.unpack_from("<H", data, i)[0]
            i += 2
        tags[(group, element)] = (vr, data[i:i + length])
        i += length
    return tags

def text(tags, tag):
    return tags[tag][1].decode("ascii").strip().rstrip("\x00").strip()

def numbers(tags, tag):
    return [float(x) for x in text(tags, tag).split(SEP)]

payload = b"".join([
    make_element(0x0028, 0x0010, "US", struct.pack("<H", 512)),
    make_element(0x0028, 0x0011, "US", struct.pack("<H", 512)),
    make_element(0x0028, 0x0030, "DS", ds(SEP.join(["0.703125", "0.703125"]))),
    make_element(0x0020, 0x0032, "DS", ds(SEP.join(["-140.0", "-150.0", "-112.5"]))),
    make_element(0x0020, 0x0037, "DS", ds(SEP.join(["1", "0", "0", "0", "1", "0"]))),
    make_element(0x0028, 0x1052, "DS", ds("-1024")),
    make_element(0x0028, 0x1053, "DS", ds("1")),
    make_element(0x0028, 0x1054, "LO", ds("HU")),
])

tags = parse_explicit_le(payload)
rows = struct.unpack("<H", tags[(0x0028, 0x0010)][1])[0]
cols = struct.unpack("<H", tags[(0x0028, 0x0011)][1])[0]
row_spacing, col_spacing = numbers(tags, (0x0028, 0x0030))
position = numbers(tags, (0x0020, 0x0032))
row_dir = numbers(tags, (0x0020, 0x0037))[:3]
col_dir = numbers(tags, (0x0020, 0x0037))[3:]
slope = numbers(tags, (0x0028, 0x1053))[0]
intercept = numbers(tags, (0x0028, 0x1052))[0]

def cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

normal = cross(row_dir, col_dir)
determinant = (row_dir[0] * (col_dir[1]*normal[2] - col_dir[2]*normal[1])
               - row_dir[1] * (col_dir[0]*normal[2] - col_dir[2]*normal[0])
               + row_dir[2] * (col_dir[0]*normal[1] - col_dir[1]*normal[0]))

p1 = position
p2 = [-140.0, -150.0, -111.25]
delta = [b - a for a, b in zip(p1, p2)]
z_spacing = abs(sum(d * n for d, n in zip(delta, normal)))

print("tags parsed        =", len(tags))
print("rows x cols        =", rows, "x", cols)
print("pixel spacing      =", row_spacing, "x", col_spacing, "mm")
print("position           =", position, "mm")
print("row direction      =", row_dir)
print("column direction   =", col_dir)
print("slice normal       =", normal)
print("orientation det    =", determinant)
print("rescale            =", slope, intercept, text(tags, (0x0028, 0x1054)))
print("voxel volume       =", round(row_spacing * col_spacing * z_spacing, 6), "mm^3")
print("z spacing from IPP =", round(z_spacing, 6), "mm")
print("HU from stored     =", 1100 * slope + intercept, "HU")
```

Expected output:

```text
tags parsed        = 8
rows x cols        = 512 x 512
pixel spacing      = 0.703125 x 0.703125 mm
position           = [-140.0, -150.0, -112.5] mm
row direction      = [1.0, 0.0, 0.0]
column direction   = [0.0, 1.0, 0.0]
slice normal       = (0.0, 0.0, 1.0)
orientation det    = 1.0
rescale            = 1.0 -1024.0 HU
voxel volume       = 0.617981 mm^3
z spacing from IPP = 1.25 mm
HU from stored     = 76.0 HU
```

The code assumes: explicit VR little endian, single-frame classic CT, one position per object,
orthonormal direction cosines, and a complete dataset. It does not handle sequences, implicit VR,
deflated datasets, or encapsulated pixel data. Those are the cases where a production library
such as pydicom is worth the dependency. The arithmetic is still worth doing by hand once, because
it tells you what the library should return.

## 8. Data, standards, and regulatory context

### 8.1 The DICOM standard

The DICOM Standard is published by the National Electrical Manufacturers Association (NEMA) as
PS3, current edition 2026d. The parts that matter here:

| Part | Content | What it constrains |
|---|---|---|
| PS3.3 | Information Object Definitions | Which attributes an IOD requires and permits |
| PS3.5 | Data Structures and Encoding | Pixel encoding, bit depth, rescale, transfer syntax |
| PS3.6 | Data Dictionary | Tag names, value representations, multiplicity |
| PS3.15 | Security and System Management Profiles | De-identification action codes |
| PS3.18 | Web Services | DICOMweb query, retrieve, store |
| PS3.4 | Service Class Specifications | Storage, query/retrieve, worklist |

The standard is normative for encoding and conformance. It is not normative for clinical
protocol, image quality, or AI performance. A conformance statement says which services and
transfer syntaxes a product supports; it does not say the product produces diagnostically
equivalent images.

### 8.2 Transfer syntax and compression

A transfer syntax UID identifies both the byte order and the compression. Common families:

| Family | Example | Decode requirement |
|---|---|---|
| Implicit VR little endian | 1.2.840.10008.1.2 | Dictionary lookup for VR |
| Explicit VR little endian | 1.2.840.10008.1.2.1 | VR in stream |
| Explicit VR big endian | 1.2.840.10008.1.2.2 | Byte swap |
| JPEG baseline | 1.2.840.10008.1.2.4.50 | Lossy JPEG |
| JPEG lossless | 1.2.840.10008.1.2.4.57/.70 | Lossless JPEG variants |
| JPEG-LS | 1.2.840.10008.1.2.4.80/.81 | JPEG-LS |
| JPEG 2000 | 1.2.840.10008.1.2.4.90/.91 | JPEG 2000 |
| HTJ2K | 1.2.840.10008.1.2.4.201/.202/.203 | High-throughput JPEG 2000 |

Compression is not neutral for AI. Lossy JPEG can remove high-frequency texture that a model uses,
and the compression ratio can differ by site. A model trained on uncompressed data and deployed on
JPEG-compressed data has a distribution shift that is invisible in the metadata unless the
transfer syntax is recorded.

### 8.3 De-identification and privacy law

The PS3.15 Basic Application Level Confidentiality Profile is the technical baseline. It is not a
legal safe harbor by itself. In the United States, the Health Insurance Portability and
Accountability Act (HIPAA) Privacy Rule defines 18 categories of protected health information and
a Safe Harbor method; de-identification must address all of them, including dates, geographic
subdivisions, and device identifiers. In the European Union, the General Data Protection
Regulation (GDPR) treats pseudonymized data as personal data, so UID remapping does not
automatically remove the data from scope.

The practical consequence: a dataset can be PS3.15-conformant and still fail a legal review if
the retention options preserve identifiers that the law treats as identifying. The
de-identification policy must be reviewed by privacy counsel, not only by engineering.

### 8.4 FDA and the intended use

The FDA does not regulate DICOM parsing. It regulates the device that uses the data. The special
controls at 21 CFR 892.2090 require a description of device inputs and outputs and labeling that
states compatible imaging hardware and protocols. That means the DICOM attributes that define the
input, such as modality, slice thickness, kernel, and contrast phase, are part of the regulated
claim. An ingestion pipeline that silently accepts any series is not compatible with a narrow
intended use.

### 8.5 What the standards do not cover

No DICOM attribute defines "diagnostic quality", "adequate de-identification", or "label". Those
are policy decisions. The standard gives you the vocabulary to express them, not the values.

### 8.6 DICOM and FHIR: where the boundary is

Fast Healthcare Interoperability Resources (FHIR) has an `ImagingStudy` resource that describes a
study, its series, and its instances. It references a DICOMweb endpoint through an `Endpoint`
resource, and the pixels are retrieved with WADO-RS. FHIR carries the ordering and workflow
context: accession, modality, body site, number of series and instances, and references to the
encounter and report. It does not carry pixel data, and it does not replace the DICOM geometry.
A pipeline that reads only `ImagingStudy` knows that a CT exists; it does not know the slice
spacing.

The practical division is: FHIR for discovery and workflow, DICOM for content and geometry,
DICOMweb for transport. An integration that conflates them either loses geometry or duplicates
storage.

## 9. Industry practice and case evidence

### 9.1 Public archives and their metadata

The Cancer Imaging Archive (TCIA) reported 287 published datasets and support for more than 3,400
publications in 2026. TCIA collections are curated and usually include a metadata table, but the
underlying DICOM still varies by contributing site. The Medical Imaging and Data Resource Center
(MIDRC) reported 567,397 imaging studies ingested and 193,194 publicly available as of its 2025
data-ingestion documentation. MIDRC was built during the COVID-19 pandemic and includes
contributions from multiple institutions, which makes it a useful testbed for site metadata
heterogeneity.

The practical lesson from both archives: even curated public data require a per-series manifest.
A collection-level description is not a substitute for checking each series.

### 9.2 Metadata leakage in chest radiography

Zech et al. (PLOS Medicine, 2018) trained pneumonia classifiers on 158,323 chest radiographs from
three institutions and found that performance fell on external data in three of five comparisons.
A model using only the hospital system reached AUROC 0.861 on a joint set because prevalence
differed (34.2% at Mount Sinai vs about 1% at the National Institutes of Health and Indiana
University). The authors also reported that classifiers could identify the source hospital with
greater than 99.9% accuracy, often from laterality markers and scanner artefacts.

That result is the canonical evidence that metadata and acquisition signatures are learnable.
DICOM metadata is the structured part of the same problem: scanner model, kernel, series
description, and protocol name can all encode site and, through site, label.

### 9.3 Report-derived labels and text leakage

CheXpert (Irvin et al., AAAI 2019) labelled 224,316 chest radiographs with a rule-based report
labeler. The labeler's mention-extraction F1 was 0.969 micro-average and 0.948 macro-average on a
1,000-report evaluation set. Those are text-level numbers. The dataset also demonstrated that
uncertainty is a first-class label state, not a nuisance. A DICOM pipeline that extracts the
report text from a private tag or a structured report inherits the labeler's errors and the
report's ambiguity.

### 9.4 Library behavior changes

pydicom 3.0 (8 September 2024) made several breaking changes: the JPEGLossless UID value changed,
saving defaults to the transfer syntax UID, `Dataset.pixel_array` converts YCbCr to RGB by
default, and `read_file`/`write_file` were removed. A pipeline pinned to pydicom 2.x and upgraded
without a regression test can change its decoded values. The industry practice is to pin the
library, hash the environment, and run a golden-file test on a fixed DICOM sample.

### 9.5 Inference, not fact

The following are inferences: that a specific hospital's series descriptions will leak a
specific label; that a specific archive's UID remapping will break longitudinal joins; that a
specific compression ratio will reduce a specific model's accuracy. Each is plausible and each
must be tested locally. The documented facts are the standards, the archive counts, and the
published leakage studies.

### 9.6 Operational practice in a health system

A production imaging pipeline at a hospital runs continuously, not once. The recurring tasks are
reconciliation (does the manifest match the archive?), deduplication (did the same study arrive
twice?), identifier mapping (did the enterprise master patient index merge two patients?), and
retention (how long are raw files kept after feature extraction?). These are data-engineering
problems with clinical consequences. A duplicate study in a training set is a leakage event; a
merged patient in a test set is a contamination event.

The common control is a nightly job that compares the PACS query results with the manifest by
`StudyInstanceUID` and reports four counts: new, changed, missing, and duplicated. A change in the
"changed" count after a scanner upgrade is an early warning of protocol drift. The job is boring,
which is why it is often skipped, and its absence is why silent dataset changes are discovered
only when a model's performance is re-evaluated.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong transfer syntax | Pixel values look like noise | Decoder ignores UID | Model trains on garbage | Validate first frame; fail loudly |
| Missing rescale intercept | HU range starts at 0 | Stored values used directly | 1024 HU offset; window wrong | Assert slope/intercept; unit test |
| Series concatenation | 225 slices where 180 expected | Duplicate series merged | Volume and lesion size inflated | Group by SeriesInstanceUID |
| Slice thickness used as spacing | z spacing differs from delta | Nominal vs geometric confusion | Volume bias | Compute spacing from positions |
| Duplicate positions | Repeated z values | Re-export or repeat acquisition | Double counting | Count duplicates; deduplicate |
| Frame order assumed | Position jumps inside object | Multiframe not sorted | Scrambled volume | Sort by per-frame position |
| MONOCHROME1 ignored | Lesions dark on display | Inversion not applied | Inverted contrast | Check photometric interpretation |
| Signed pixel decoded unsigned | Negative HU become large | PixelRepresentation ignored | Air looks dense | Check signedness |
| Private tag leak | Text field predicts label | Undocumented vendor tag | Shortcut learning | Remove private groups; audit |
| Series description leak | Description predicts label | Free text encodes indication | Inflated AUROC | Blind descriptions at inference |
| Burned-in annotation | OCR finds text | Pixels not cleaned | Identifier and label leak | OCR audit; crop or exclude |
| Laterality mismatch | L/R disagrees with report | Marker or metadata error | Wrong-side model | Human resolution; exclude |
| Date shifting inconsistent | Same patient has conflicting dates | Per-file shifting | Longitudinal leakage or loss | Shift per patient |
| UID remapping inconsistent | Series no longer joins | Per-file remapping | Broken splits and joins | Remap per study or patient |
| Compression heterogeneity | Texture differs by site | Lossy codec | Site signature | Record transfer syntax; stratify |
| Library upgrade | Decoded values change | Default behavior change | Silent regression | Pin and golden-test |

### 10.1 Red flags that stop ingestion

1. A series with more than one position-delta mode.
2. A CT series without rescale attributes.
3. A study with two series sharing a description but differing in thickness or kernel.
4. A `BurnedInAnnotation` value of YES without a cleaning record.
5. A private group that survives de-identification.
6. A laterality field that disagrees with the report.
7. A UID that appears under two different patients.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Every image object has a SOP class | Routing and decode | Secondary capture or non-DICOM | Inventory and quarantine |
| Series UID identifies one reconstruction | Grouping | Illegal UID reuse | Geometry consistency check |
| Position delta equals spacing | Volume geometry | Gantry tilt, mixed thickness | Delta distribution |
| Direction cosines are orthonormal | Affine construction | Oblique or corrupt metadata | Dot products and determinant |
| Rescale attributes are present for CT | HU conversion | Vendor export or secondary capture | Presence check and phantom |
| Pixel data are decodable | Model input | Unsupported codec | First-frame decode test |
| De-identification removes identifiers | Privacy | Private tags, burned-in text | Audit and OCR |
| Metadata do not leak label | Valid evaluation | Descriptions and protocol names | Univariate association |
| Frame order is spatial | Volume assembly | Multiframe objects | Sort by position |
| The manifest is complete | Reproducibility | Ad hoc file lists | Hash and row count |

### 11.1 Limits of this topic

This topic does not cover DICOM network protocols in depth, structured report content mapping, or
the full PS3.15 attribute table. It covers the attributes and procedures that determine whether a
volume is valid and whether metadata leak the label. For production ingestion, use a maintained
library and validate it against the cases in Section 5.

## 12. Exercises

### Exercise 1. Tag identification (recall, 15 minutes)

**Deliverable:** A list of the tags needed to compute voxel volume, slice order, HU, and
orientation, with tag numbers and units.

**Acceptance criteria:** At least eight tags; correct group and element; units stated.

**Time:** 15 minutes.

### Exercise 2. Spacing from positions (calculation, 25 minutes)

**Deliverable:** Given positions from -100.0 to 98.75 mm in 1.25 mm steps, compute the number of
slices, the position delta, and the volume extent.

**Acceptance criteria:** Correct count (160 slices), correct extent (198.75 mm), units stated.

**Time:** 25 minutes.

### Exercise 3. Duplicate detection (application, 40 minutes)

**Deliverable:** A script or spreadsheet that flags duplicate positions in a synthetic series and
reports the duplicate count and delta interquartile range.

**Acceptance criteria:** Detects at least two planted duplicates; reports the bimodal delta;
states the consequence for volume.

**Time:** 40 minutes.

### Exercise 4. De-identification audit (critique, 45 minutes)

**Deliverable:** An audit of a sample DICOM header (provided by the instructor) listing residual
identifiers by group and the action code that should have applied.

**Acceptance criteria:** Finds at least three residual identifiers; names the PS3.15 action code;
distinguishes metadata from pixel leakage.

**Time:** 45 minutes.

### Exercise 5. Leak test (application, 50 minutes)

**Deliverable:** A table of univariate associations between candidate metadata fields and the
label, with a recommendation for each field.

**Acceptance criteria:** At least six fields; association metric with unit; recommendation
justified; no field removed without a replacement plan.

**Time:** 50 minutes.

### Exercise 6. Multi-frame handling (application, 45 minutes)

**Deliverable:** A procedure for parsing an enhanced CT object, including where to read
per-frame position and rescale, and how to sort frames.

**Acceptance criteria:** Names the functional group sequence; states the sort key; handles
missing per-frame values.

**Time:** 45 minutes.

### Exercise 7. Manifest schema (design, 60 minutes)

**Deliverable:** A manifest schema with one row per series and columns for identifiers, geometry,
unit, decode path, de-identification status, leak flags, and QC status.

**Acceptance criteria:** Every column has a type and a null rule; QC status is not a single
boolean; schema versioned.

**Time:** 60 minutes.

### Exercise 8. Failure reconstruction (critique, 40 minutes)

**Deliverable:** A 500-word reconstruction of how a 1024 HU offset could pass unnoticed through a
training pipeline and what test would have caught it.

**Acceptance criteria:** Names the tag, the code path, the visual appearance, and a quantitative
test; distinguishes a unit error from a model error.

**Time:** 40 minutes.

## 13. Self-check answers

**A1. Which tag gives the through-plane spacing, and why not `SliceThickness`?**
The spacing is the projection of the difference between consecutive `ImagePositionPatient` values
onto the slice normal. `SliceThickness` is the nominal reconstruction width. They are equal in a
contiguous series but differ when slices overlap or gap, and the position delta is the geometric
truth for stacking.

**A2. A series has 180 slices from -112.5 to 111.25 mm. What is the spacing?**
(111.25 + 112.5) / 179 = 223.75 / 179 = 1.2500 mm. The count is 180 because both endpoints are
included.

**A3. Why is a duplicate series dangerous even if the images look identical?**
It doubles the slice count and creates duplicate positions. A naive stack counts the lesion twice
and interpolates across gaps, inflating volume and changing texture. The fix is grouping by
`SeriesInstanceUID`, not by description.

**A4. What does `RescaleSlope = 1`, `RescaleIntercept = -1024` mean for a stored value of 1024?**
HU = 1024 × 1 - 1024 = 0 HU, water. A stored value of 0 maps to -1024 HU, air. Ignoring the
intercept shifts every value by +1024 HU.

**A5. Why does PS3.15 Basic Profile not guarantee de-identification?**
It removes or replaces attributes but does not clean pixel data by default, does not remove
burned-in text unless the Clean Pixel Data option is used, and does not address legal
re-identification risk. It is a technical profile, not a legal safe harbor.

**A6. What is the difference between Study, Series, and SOP Instance UID?**
Study identifies the order or encounter; Series identifies one acquisition or reconstruction;
SOP Instance identifies one object. A study contains series; a series contains instances; an
instance may contain frames.

**A7. A CT series has `BurnedInAnnotation = YES` and the label is "left-sided lesion". What do you
do?**
Audit the pixels with OCR and visual review. If the text says "LEFT" and the label depends on
laterality, the text is a label leak and must be removed or the series excluded. If the text is a
measurement, crop it and document the crop. Do not assume metadata de-identification removed it.

**A8. Why is `SeriesDescription` a bad grouping key?**
It is free text, site-specific, and often reused across reconstructions. Two series with the same
description can have different kernels, thicknesses, or contrast phases. Group by UID and
stratify by coded attributes.

**A9. Design rubric for Exercise 7.** A strong schema has typed columns, explicit null rules,
separate geometry/unit/de-identification/leak/QC fields, a schema version, and a row count that
matches the file inventory. A weak schema is a single CSV with "valid" and no provenance.

**A10. Design rubric for Exercise 8.** A strong reconstruction traces the stored value through
decode, rescale, window, and normalization, shows the visual consequence (saturation), and
proposes a phantom or ROI assertion test. A weak answer blames the model.

## 14. Interview questions

**Q1. Walk me through what happens between a DICOM file and a HU array.**
A strong answer covers file meta, transfer syntax, dataset parse, pixel decode with signedness and
bit depth, rescale slope/intercept, and shape/orientation. A weak answer says "pydicom
pixel_array" and stops.

**Q2. How do you know which series to use?**
A strong answer groups by UID, checks SOP class, geometry, slice count, contrast, kernel, and
thickness against the intended use, and records the selection rule. A weak answer picks the
largest series or the one with the most slices.

**Q3. Why can `SliceThickness` mislead you?**
A strong answer distinguishes nominal thickness from position delta and gives the overlapping or
gapped case. A weak answer says they are always the same.

**Q4. How do you handle a multi-frame CT object?**
A strong answer reads per-frame functional groups, sorts by position, handles per-frame rescale,
and does not trust frame index. A weak answer treats it as one image or assumes frame order.

**Q5. What metadata would you remove before training, and why?**
A strong answer removes identifiers, private tags, descriptions, and protocol names that leak,
while retaining geometry and acquisition fields needed for stratification. A weak answer removes
everything non-pixel and loses the ability to audit.

**Q6. How do you de-identify while keeping longitudinal follow-up?**
A strong answer uses consistent per-patient date shifting and UID remapping, documents the
options, and audits for residual identifiers. A weak answer removes all dates, which destroys
follow-up.

**Q7. What is the difference between DICOMweb and DIMSE?**
A strong answer says DICOMweb is RESTful (QIDO-RS, WADO-RS, STOW-RS) over HTTP, while DIMSE is the
traditional association-oriented protocol. A weak answer treats them as synonymous.

**Q8. How would you detect a label leak in metadata?**
A strong answer measures univariate association between each candidate field and the label,
checks prevalence in positives and negatives, and tests a metadata-only model. A weak answer looks
at a correlation matrix and stops.

**Q9. What breaks when you upgrade pydicom?**
A strong answer names transfer syntax defaults, YCbCr conversion, save encoding, and removed
APIs, and proposes a golden-file regression test. A weak answer says upgrades are safe.

**Q10. How do you make ingestion reproducible?**
A strong answer versions the specification, hashes the manifest, pins library versions, and runs
the same QC on every batch. A weak answer keeps a script in a shared drive.

## 15. Key terms

- **Accession number** — order identifier; not a patient identifier.
- **Action code** — PS3.15 code (D, Z, X, K, C, U) for de-identification.
- **Burned-in annotation** — text or graphics in pixel data.
- **DICOM** — Digital Imaging and Communications in Medicine standard.
- **DICOMweb** — RESTful DICOM services (QIDO-RS, WADO-RS, STOW-RS).
- **DIMSE** — traditional DICOM network services.
- **Explicit VR** — value representation written in the stream.
- **Frame of reference UID** — shared coordinate system.
- **HU** — Hounsfield unit; calibrated CT attenuation.
- **ImageOrientationPatient** — row and column direction cosines.
- **ImagePositionPatient** — slice origin in patient coordinates.
- **Implicit VR** — value representation from the dictionary.
- **Instance** — one SOP object.
- **Manifest** — one row per series recording provenance and QC.
- **MONOCHROME1** — inverted greyscale photometric interpretation.
- **Multi-frame** — one object containing multiple frames.
- **PACS** — picture archiving and communication system.
- **Per-frame functional groups** — per-frame geometry and acquisition metadata.
- **Photometric interpretation** — colour or greyscale convention.
- **Pixel representation** — signed or unsigned integer encoding.
- **Pixel spacing** — in-plane distance between pixel centres.
- **Private tag** — vendor-defined attribute in an odd group.
- **PS3.15** — DICOM security and confidentiality profiles.
- **Rescale intercept** — offset to physical units.
- **Rescale slope** — multiplier to physical units.
- **Series** — one acquisition or reconstruction.
- **Slice normal** — cross product of row and column directions.
- **Slice thickness** — nominal reconstructed width.
- **SOP class** — object type definition.
- **SOP instance UID** — unique object identifier.
- **Study** — one order or encounter.
- **Transfer syntax** — encoding and compression rules.
- **VNA** — vendor-neutral archive.

## 16. Sources

- DICOM Standard, PS3 current edition 2026d. PS3.3 Information Object Definitions; PS3.5 Data
  Structures and Encoding; PS3.6 Data Dictionary; PS3.15 Security and System Management
  Profiles; PS3.18 Web Services. https://www.dicomstandard.org/ (accessed 25 October 2026).
- DICOM PS3.5 2026d, Chapter 3, definitions of Pixel Data, Native Format, Encapsulated Format,
  Fragment, and Basic Offset Table.
  https://dicom.nema.org/medical/dicom/current/output/chtml/part05/chapter_3.html (accessed
  25 October 2026).
- DICOM PS3.15 2026d, Annex E, Attribute Confidentiality Profiles.
  https://dicom.nema.org/medical/Dicom/current/output/chtml/part15/chapter_E.html (accessed
  25 October 2026).
- pydicom 3.0.0 release notes, 8 September 2024.
  https://pydicom.github.io/pydicom/stable/release_notes/v3.0.0.html (accessed 25 October 2026).
- Zech, J. R. et al., "Variable generalization performance of a deep learning model to detect
  pneumonia in chest radiographs: A cross-sectional study," PLOS Medicine 15(11): e1002683
  (2018). https://doi.org/10.1371/journal.pmed.1002683
- Irvin, J. et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert
  Comparison," AAAI 2019. https://doi.org/10.1609/aaai.v33i01.3301590
- The Cancer Imaging Archive, Browse Data and 2026 program update.
  https://www.cancerimagingarchive.net/browse-data/ (accessed 25 October 2026).
- MIDRC, Data Ingestion Pipeline. https://www.midrc.org/data-ingestion-pipeline (accessed
  25 October 2026).
- U.S. Department of Health and Human Services, HIPAA Privacy Rule, de-identification guidance.
  https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
  (accessed 25 October 2026).
- European Union, Regulation (EU) 2016/679 (General Data Protection Regulation).
  https://eur-lex.europa.eu/eli/reg/2016/679/oj (accessed 25 October 2026).
- U.S. Food and Drug Administration, 21 CFR 892.2090, final order 90 FR 24969, 13 June 2025.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892.2090
  (accessed 25 October 2026).
- IHE Radiology, AI Workflow for Imaging (AIW-I) supplement.
  https://www.ihe.net/uploadedFiles/Documents/Radiology/IHE_RAD_Suppl_AIW-I.pdf (accessed
  25 October 2026).

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_modalities.md`](01_modalities.md)
- Next topic: [`03_preprocessing.md`](03_preprocessing.md)
- Related topic: [`06_leakage-and-shortcuts.md`](06_leakage-and-shortcuts.md)
- Root survey: [`../../10_medical_imaging.md`](../../10_medical_imaging.md)
- Labs: [`../../labs/`](../../labs/)
- DICOM standard: https://www.dicomstandard.org/
