# DICOM Pipelines

> Module 03 — Data Engineering and Interoperability
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 05
> Version 2026.3

## 1. Why this topic exists

Medical imaging is the largest data modality in most health systems and the one most likely to leak
identifiers. A computed tomography (CT) study is a set of hundreds of files, each with a header of two
hundred or more attributes and a pixel matrix. The header contains the patient's name, identifier, birth
date, and often the institution and technologist. The pixels can contain burned-in text: a name, a date,
an accession number rendered into the image itself. A pipeline that cleans the header and ignores the
pixels has de-identified the metadata, not the image.

The failures are specific and recurrent.

- A directory of DICOM files is organized by patient name and study date. A file copy renames the files,
  but the internal `SOPInstanceUID` still encodes the original study. A recipient links the files back to
  the source and re-identifies the patient.
- A de-identification script removes `PatientName` but leaves `StudyDescription` as "MRI brain - John
  Smith follow-up". The description is a free-text field and is not covered by a tag-based rule.
- A study is exported with 1,240 instances, but 8 belong to a different study because the export used a
  filename prefix rather than `StudyInstanceUID`. The series is assembled from two patients, and the
  reconstruction is wrong.
- Dates are shifted per file rather than per patient. The interval between the scan and the surgery
  changes, and a longitudinal analysis of tumor growth becomes meaningless.
- A private tag from the scanner vendor contains the operator's initials and the hospital's internal
  worklist identifier. The de-identification tool removes standard tags and passes the private tag
  through.
- Burned-in annotations are present in 3.4% of images. The pipeline validates headers only, the data are
  released, and the annotation is visible in a screenshot used in a publication.

DICOM is a mature standard with a normative de-identification profile. The profile is detailed and
explicit about which attributes to remove, replace, or retain. The engineering problem is applying it
consistently across an archive, validating the result, and preserving enough metadata for the analysis.

This topic covers series assembly, metadata extraction, de-identification, pixel handling, storage
organization, and the linkage between imaging and clinical data. It follows FHIR ingestion because the
two often meet at the same patient and encounter.

> **Note:** The examples target the DICOM PS3 2026d base edition and PS3.15 Annex E. Conformance in DICOM
> is declared against service classes and profiles, not against an edition number; the edition is cited
> for reproducibility of the rules, not as a compliance claim.

## 2. Learning outcomes

By the end of this topic you can:

1. **Assemble** studies, series, and instances from a file inventory using DICOM unique identifiers
   (UIDs), and detect files that do not belong. Assessed in Exercise 1.
2. **Extract** the metadata needed for cohort construction and quality control without loading pixel
   data. Assessed in Exercise 2.
3. **Apply** the PS3.15 basic confidentiality profile and name the options that change the result.
   Assessed in Exercise 3 and self-check item 3.
4. **Shift** dates consistently per patient and verify that intervals are preserved. Assessed in
   Exercise 4.
5. **Audit** pixel data for burned-in identifiers and describe the limits of automated detection.
   Assessed in Exercise 5.
6. **Design** a storage layout and metadata index that support cohort queries and access control.
   Assessed in Exercise 6.
7. **Link** imaging studies to clinical encounters using identifiers and time, and quantify linkage
   loss. Assessed in Exercise 7.
8. **Critique** a de-identification report for missing tag classes, UID handling, and pixel review.
   Assessed in Exercise 8.

## 3. The core mental model

A DICOM object has three layers, and de-identification must address all three.

```text
+---------------------------------------------------------------+
| 3. PIXELS                                                     |
|    stored pixel data, overlays, burned-in annotations,       |
|    presentation state, structured reports                     |
+---------------------------------------------------------------+
| 2. HEADER                                                     |
|    standard attributes: patient, study, series, equipment,    |
|    acquisition parameters, private tags                       |
+---------------------------------------------------------------+
| 1. FILE / TRANSFER                                            |
|    preamble, file meta information, transfer syntax,          |
|    SOP class, instance UID                                    |
+---------------------------------------------------------------+
```

A tag-based cleaner operates on layer 2. It does not see layer 3, and it may not understand layer 1. The
PS3.15 profile includes options specifically for layer 3 (Clean Pixel Data, Clean Recognizable Visual
Features, Clean Graphics, Clean Structured Content) for that reason.

### 3.1 The hierarchy

```text
Patient
  └── Study (StudyInstanceUID)
        ├── Series (SeriesInstanceUID)
        │     ├── Instance (SOPInstanceUID)
        │     ├── Instance
        │     └── ...
        └── Series
```

Each level has a UID. The UID is the identity; the file name is not. A pipeline that assembles series by
directory name will fail when a directory contains two studies or when a file is renamed. A pipeline that
assembles by `SeriesInstanceUID` is robust to file naming.

The hierarchy is also the unit of analysis. A study is one imaging encounter; a series is one acquisition
protocol; an instance is one image or one object. A radiomics feature is computed per series or per
instance, and a cohort is usually defined at the study or patient level.

### 3.2 Header and pixel are separate evidence

The header says what the acquisition was supposed to be. The pixels are what was acquired. They can
disagree:

- The header says `SliceThickness` 1.25 mm, but the reconstruction used 2.5 mm.
- The header says `ImageOrientationPatient` axial, but the image is coronal.
- The header says no contrast, but the pixel intensities show contrast enhancement.
- The header has no burned-in text flag, but the pixels contain annotations.

Quality control compares the two. A pipeline that trusts the header alone will include corrupted or
mislabeled series.

### 3.3 De-identification is a profile plus options

PS3.15 Annex E defines a Basic Application Level Confidentiality Profile and a set of options. The Basic
Profile is conservative: it removes patient identity, demographics, responsible parties, institution,
and device identity, and it modifies dates and UIDs unless retention options are selected.

The options are not cosmetic. Choosing "Retain Longitudinal Temporal Information with Modified Dates"
keeps the intervals between studies but changes absolute dates. Choosing "Retain UIDs" keeps the original
UIDs, which preserves linkage to the source but also preserves a pseudonymous identifier that can be
resolved by anyone with access to the source. Choosing "Clean Pixel Data" runs a pixel-cleaning algorithm;
it does not guarantee that all burned-in text is removed.

A de-identification report must state the profile, the options, the tool and version, the validation
method, and the residual risk. "De-identified per DICOM" is not a statement.

### 3.4 Linkage is the hard part

Imaging must link to clinical data: the patient, the encounter, the order, the diagnosis. The link can use:

- `PatientID` (source identifier) mapped to an enterprise identifier.
- `StudyInstanceUID` mapped to an accession number, which maps to an order.
- Study date and modality matched to an encounter, which is probabilistic.

Each link has a failure mode. `PatientID` may be local to a facility; an accession number may be reused;
date matching may associate a study with the wrong admission. A pipeline should record the linkage method
and the linkage rate, not just the final join.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| DICOM | Digital Imaging and Communications in Medicine | Treated as a file extension | NEMA PS3 / ISO 12052 |
| SOP class | Service-object pair class defining an object or service | Confused with a transfer syntax | UID |
| IOD | Information object definition | Confused with a SOP class | — |
| UID | Unique identifier for an object or class | Treated as a file name | Dotted decimal, max 64 chars |
| `StudyInstanceUID` | Identifier of a study | Assumed globally unique across copies | UID |
| `SeriesInstanceUID` | Identifier of a series | Confused with study UID | UID |
| `SOPInstanceUID` | Identifier of an instance | Confused with a file name | UID |
| Tag | Attribute identifier as group and element | Confused with a field name | `(gggg,eeee)` |
| VR | Value representation, the data type of an attribute | Ignored by naive parsers | CS, LO, PN, DA, DS, UI |
| Transfer syntax | Encoding of the dataset, including endianness and compression | Assumed always explicit little endian | UID |
| Part 10 file | DICOM file with preamble and file meta information | Confused with a raw dataset | — |
| PACS | Picture archiving and communication system | Confused with a research archive | — |
| VNA | Vendor-neutral archive | Confused with a PACS | — |
| DICOMweb | HTTP-based DICOM services | Confused with the DICOM standard | PS3.18 |
| QIDO-RS | Query service over HTTP | — | PS3.18 §10.6 |
| WADO-RS | Retrieve service over HTTP | — | PS3.18 §10.4 |
| STOW-RS | Store service over HTTP | — | PS3.18 §10.5 |
| Private tag | Vendor-defined attribute with an odd group number | Assumed harmless | `(odd,xxxx)` |
| Burned-in annotation | Text or graphics rendered into pixels | Assumed absent | — |
| Presentation state | Stored display settings, may contain annotations | Ignored | — |
| Structured report | DICOM object containing clinical content as text/codes | Treated as an image | DICOM SR |
| De-identification | Removing or transforming identifiers | Confused with encryption | PS3.15 |
| Confidentiality profile | Normative set of de-identification actions | Confused with a tool setting | PS3.15 Annex E |
| Action code | D, Z, X, K, C, U in PS3.15 tables | Ignored | PS3.15 Table E.1-1a |
| Date shifting | Consistent offset applied to dates | Applied per file | Days |
| UID remapping | Replacing UIDs with new ones consistently | Applied per file | UID |
| Residual risk | Risk remaining after de-identification | Assumed zero | — |
| Linkage rate | Share of studies linked to a clinical encounter | Unreported | Percent |
| Series completeness | Received instances versus expected | Assumed complete | Count |
| Slice thickness | Reconstruction slice spacing | Confused with acquisition thickness | mm |
| Pixel spacing | Physical size of a pixel | Confused with field of view | mm |
| Photometric interpretation | Pixel color model | Ignored in metadata extraction | MONOCHROME2, RGB |

Two terms need expansion.

**Action codes.** PS3.15 Table E.1-1 assigns each attribute an action: D (replace with a dummy value),
Z (zero-length), X (remove), K (keep), C (clean), U (replace with a new UID). The distinction between D
and Z matters: a dummy value such as "ANONYMIZED" preserves the element's presence, while a zero-length
value removes it. Some downstream tools require the element to exist.

**UID remapping.** If UIDs are remapped, they must be remapped consistently across all objects that
reference them. A study UID that changes per file breaks the hierarchy; a series UID that changes per
instance breaks the series. Remapping must be a deterministic function of the original UID and a secret,
applied identically everywhere.

## 5. The mechanics, step by step

The procedure builds an imaging cohort from a PACS export and produces a de-identified research archive.
It assumes CT and magnetic resonance imaging (MRI) studies and a research data use agreement.

### Step 1 — Define the imaging cohort and the clinical question

Input: the research question.
Output: modality, body part, date range, and clinical inclusion criteria.

Example: adults with a first CT of the chest for suspected pulmonary embolism in 2024, linked to an
encounter with a heart-failure diagnosis. The definition determines the query, the metadata fields, and
the linkage keys.

Reason: imaging archives are large. A cohort definition that is not specific will export tens of
terabytes and require a second de-identification review.

### Step 2 — Inventory the source archive

Input: the PACS or VNA inventory.
Output: counts by modality, body part, study date, and storage size.

Use the archive's metadata query, not a file system walk, if available. Record the query, the date of the
query, and the archive's reported counts. File system counts can differ because of duplicates, migrated
objects, or non-DICOM files.

Reason: the inventory is the baseline for reconciliation. Without it, a missing series is invisible.

### Step 3 — Choose the retrieval mechanism

Input: the archive's supported services.
Output: C-FIND/C-MOVE, DICOMweb QIDO-RS/WADO-RS, or a vendor export.

Prefer DICOMweb for HTTP-based automation and cloud archives. Use C-MOVE for older PACS. A vendor bulk
export is acceptable if it includes a manifest and preserves UIDs.

Reason: the mechanism determines the metadata available before retrieval. QIDO-RS can filter by modality
and date, reducing the volume transferred.

### Step 4 — Query metadata before pixels

Input: the cohort criteria.
Output: a candidate list of studies with study UID, patient identifier, modality, study date, and series
count.

Retrieve metadata only first. Validate the candidate list against the clinical cohort before transferring
pixels. A metadata query that returns 4,200 studies can be reviewed; 4.2 TB of pixels cannot.

Reason: pixel transfer is the expensive step. Filtering at the metadata layer avoids transferring studies
that will be excluded.

### Step 5 — Build the instance manifest

Input: the retrieved files or metadata responses.
Output: one row per instance with file path, study UID, series UID, SOP instance UID, SOP class, transfer
syntax, and file size.

Compute a cryptographic hash of each file and include it. The manifest is the contract between retrieval
and assembly; every downstream count reconciles to it.

Reason: a manifest makes the archive auditable and lets a re-run detect changed or missing files.

### Step 6 — Assemble studies and series by UID

Input: the manifest.
Output: a hierarchy with counts per study and series.

Group by `StudyInstanceUID`, then `SeriesInstanceUID`, then `SOPInstanceUID`. Do not group by directory
or file name. Detect duplicate `SOPInstanceUID` values with different hashes, which indicates a
re-encoding or a collision.

Reason: file names are not identities. A directory can contain two studies, and a study can span
directories.

### Step 7 — Validate series completeness

Input: the assembled series and the expected instance count.
Output: a completeness report per series.

Expected count can come from `NumberOfSeriesRelatedInstances` in the study metadata, from the archive's
count, or from the acquisition protocol. A series with 318 of 320 expected instances is incomplete and
may produce a reconstruction artifact.

Reason: an incomplete series can still be readable, which makes the defect silent. The completeness
report is the only signal.

### Step 8 — Inspect transfer syntax and pixel encoding

Input: the file meta information.
Output: a per-instance record of transfer syntax, rows, columns, bits allocated, photometric
interpretation, and whether pixel data are compressed.

A pipeline that assumes explicit VR little endian will fail on implicit VR, big endian, or compressed
transfer syntaxes. Record the syntax and use a library that supports it.

Reason: transfer syntax errors produce garbled pixels or parse failures. They are common in multi-vendor
archives.

### Step 9 — Select the confidentiality profile and options

Input: the data use agreement and the research need.
Output: a written profile selection: Basic Profile plus named options.

Example: Basic Profile, Retain Longitudinal Temporal Information with Modified Dates, Retain UIDs
disabled, Clean Pixel Data enabled, Clean Descriptors enabled. Record why each option was chosen.

Reason: the options change both privacy and utility. A report that says "de-identified" without the
options cannot be reviewed.

### Step 10 — Apply header de-identification

Input: each instance's header.
Output: a transformed header with actions applied per PS3.15.

Apply the action codes: remove patient name and identifiers, zero or dummy demographic fields, remove
institution and referring physician, clean free-text descriptions, and handle dates per the selected
option. Do not rely on a blocklist of ten tags; use the full table.

Reason: a partial blocklist leaves attributes such as `OtherPatientIDs`, `PatientAddress`, and
`StudyDescription` untouched. The standard's table exists because the list is long.

### Step 11 — Remap UIDs consistently

Input: study, series, and SOP instance UIDs.
Output: new UIDs derived deterministically from the originals and a secret.

Use a keyed hash to generate a new UID in a private arc, such as `2.25.<integer>`. Apply the same mapping
to every reference, including `ReferencedSOPInstanceUID` in structured reports and presentation states.
Store the mapping in a secure crosswalk if re-linkage is required.

Reason: inconsistent remapping breaks the hierarchy; retaining original UIDs preserves a link to the
source that may be re-identifiable.

### Step 12 — Shift dates per patient

Input: all date and time attributes for a patient.
Output: shifted values with a single offset per patient.

Compute the offset from a keyed hash of the patient identifier, apply it to study, series, acquisition,
and content dates, and preserve time-of-day where the option requires it. Verify that intervals between
studies are unchanged.

Reason: per-file offsets destroy longitudinal intervals. Per-patient offsets preserve them but still
allow inference at the boundaries, which Section 11 discusses.

### Step 13 — Remove or retain private tags deliberately

Input: all private tags.
Output: a decision per tag: remove, retain as safe, or retain with cleaning.

The default is removal. PS3.15 provides a "Retain Safe Private" option with a table of vendor attributes
that are considered safe. A private tag not on that list should be removed, because its contents are not
standardized and may contain identifiers.

Reason: private tags are where vendor-specific worklist identifiers, operator names, and study notes
hide. A cleaner that ignores them is incomplete.

### Step 14 — Clean pixel data

Input: the pixel matrices.
Output: pixel data with burned-in annotations removed or obscured, plus a record of the method.

Use the Clean Pixel Data option where the tool supports it. Methods include masking regions, replacing
pixel values, or re-rendering from a cleaned presentation state. Validate by visual review of a sample and
by OCR before and after.

Reason: header de-identification cannot remove text that is part of the image. A publication screenshot
can reveal an identifier that no header rule touched.

### Step 15 — Audit with OCR and visual review

Input: the cleaned images.
Output: a detection report with flagged instances and a sample review.

Run OCR on a sample or on all images, depending on risk. OCR has false positives (anatomy that looks like
text) and false negatives (stylized fonts, low contrast). Record the tool, the threshold, and the review
rate.

Reason: OCR is a screening tool, not a guarantee. The report must state its sensitivity limitations.

### Step 16 — Validate header and pixel consistency

Input: the de-identified objects.
Output: a validation report with pass and fail counts.

Checks: no patient name or identifier remains, UIDs are remapped consistently, dates are shifted,
private tags are removed, pixel data hash changed only where expected, and the object still parses. A
de-identified object that no longer opens is a failed de-identification, not a success.

Reason: over-cleaning can break the object. Removing a required attribute or corrupting the pixel data
produces a file that passes a text scan and fails in the viewer.

### Step 17 — Compute derived metadata

Input: the cleaned headers.
Output: a metadata table with modality, body part, study date (shifted), series description, acquisition
parameters, slice thickness, pixel spacing, contrast flag, and manufacturer.

Derive body part from `BodyPartExamined` and the series description, with a controlled vocabulary.
Derive contrast from `ContrastBolusAgent` and the series description. Record missing values rather than
imputing them.

Reason: derived metadata is what cohort queries use. Raw DICOM tags are numerous and inconsistently
populated; a curated table is the usable interface.

### Step 18 — Organize storage

Input: the de-identified objects.
Output: an object store layout and a metadata index.

Use a layout such as `study_uid/series_uid/instance_uid.dcm`, or a hash-based layout with the UIDs in the
index. Store the pixel data in object storage with lifecycle policies, and the metadata in a relational
database for query.

Reason: a layout based on patient name or date leaks identifiers through the path. A UID-based layout
does not.

### Step 19 — Link to clinical data

Input: the imaging metadata and the clinical warehouse.
Output: a linkage table with study, person, encounter, and a linkage method and confidence.

Link on the enterprise patient identifier first, then on accession number, then on date and modality with
a reviewed rule. Report the linkage rate by method and the number of ambiguous matches.

Reason: imaging without clinical context has limited research value. The linkage method determines
whether the cohort is correct.

### Step 20 — Quality control the images

Input: the de-identified series.
Output: a QC report with series completeness, slice spacing consistency, orientation, and whether the
series is likely diagnostic.

Check that slice spacing is uniform, that the number of slices matches the expected coverage, and that
the series is not a scout, localizer, or derived image. Exclude non-diagnostic series from feature
extraction.

Reason: a radiomics feature computed on a scout image is noise. QC is what separates an imaging dataset
from a folder of images.

### Step 21 — Audit and release

Input: the archive, the metadata, and the reports.
Output: a release package with a manifest, a de-identification report, a data dictionary, and a residual
risk statement.

The release must state the profile and options, the tool versions, the validation results, the OCR
method, the date-shift method, the UID policy, and the known limitations. It must be reviewed by a
privacy officer or IRB, not only by the engineering team.

Reason: de-identification is a legal and ethical determination, not only a technical transformation. The
audit package is the evidence for that determination.

### Step 22 — Handle duplicate, superseded, and partial instances

Input: the instance manifest.
Output: one current instance per SOP Instance UID, with duplicates and gaps classified.

| Case | Detection | Rule |
|---|---|---|
| Same SOP UID, same hash | Hash comparison | Keep one; count as duplicate delivery |
| Same SOP UID, different hash | Hash comparison | Quarantine; re-encoding or collision |
| Higher version of an instance | `InstanceNumber`, acquisition time | Keep the later reconstruction, retain both |
| Missing instance in a series | Expected count from metadata | Mark series incomplete |
| Series with non-uniform spacing | Slice location and thickness | Flag for QC; do not compute volumetrics |
| Instance from another study | Study UID mismatch | Quarantine; never release |

Reason: a duplicate instance is harmless if counted, and a missing instance is harmful if not. The
manifest plus the expected count turns both into explicit states.

### Step 23 — Handle clock, timezone, and date boundaries

Input: study, series, acquisition, and content dates.
Output: a normalized time model and a shift rule.

DICOM dates have no timezone. Acquisition times may be local to the scanner. Before shifting, establish
the source timezone from the site configuration or the `TimezoneOffsetFromUTC` attribute; if neither is
available, record the time as unknown-offset and exclude it from interval calculations that cross sites.

| Boundary | Example | Rule |
|---|---|---|
| Midnight study date | Acquisition at 00:05 | Date is the local date, not UTC |
| DST transition | 02:30 on 27 October | Store local time with offset; do not infer |
| Leap day | 29 February 2024 | Use date arithmetic, not day-of-year |
| Date shift across year | 31 December + 30 days | Shift the date, not the year field |
| Birth date year only | `1950` | Store precision; do not impute a day |

Reason: imaging intervals drive longitudinal analyses. A timezone error of a few hours rarely changes a
study date, but it changes acquisition ordering and can swap two series.

### Branch point A — The archive is a clinical PACS

Do not de-identify in place. Copy to a research zone, de-identify there, and keep the clinical archive
untouched. The clinical archive is governed by clinical policy and may be audited; modifying it is a
serious incident.

### Branch point B — The study is a single large series

For whole-slide imaging or large volumetric data, the instance count can be tens of thousands and the
pixel size can be gigabytes. Plan storage and transfer around the object size, not the study count.

### Branch point C — The images are already de-identified

Validate the existing de-identification rather than trusting the label. Check headers, private tags, UIDs,
dates, and pixels. An "already de-identified" archive frequently contains residual identifiers from an
older tool version.

## 6. Worked example

A hospital exports CT chest studies for a pulmonary embolism cohort. The export contains 1,248 files in a
flat directory. The target is a de-identified research archive with a metadata index.

### 6.1 Assembly

The 1,248 files resolve to 1,240 instances in one study and 8 instances from a second study.

| Series | Description | Instances | Slice thickness | Coverage |
|---|---|---:|---:|---:|
| 1 | Scout | 2 | 5.0 mm | 5 mm |
| 2 | CT pulmonary angiogram, arterial | 320 | 1.25 mm | 400.0 mm |
| 3 | CT chest, venous | 318 | 1.25 mm | 397.5 mm |
| 4 | Delayed phase | 600 | 1.25 mm | 750.0 mm |
| **Total** | | **1,240** | | |

Coverage is computed as `(instances - 1) × slice thickness` for a single stack: series 2 is
(320 − 1) × 1.25 = 398.75 mm, not 400.0 mm. If the coverage includes the reconstructed slice extent, the
formula is `instances × spacing = 400.0 mm`. The difference, 1.25 mm, is one slice. The QC report states
which convention it uses; mixing them across series produces inconsistent coverage estimates.

The 8 extra instances belong to `StudyInstanceUID 1.2.840.113619.2.55.3.999`, a different patient. The
export's file naming used a patient-name prefix that collided. Assembly by UID isolates them and the
pipeline quarantines them. Their pixel data are never released.

### 6.2 De-identification counts

| Item | Count | Note |
|---|---:|---|
| Instances | 1,240 | After quarantine |
| Standard attributes per instance | 212 | Varies by SOP class |
| Attributes with an action other than K | 61 | Per PS3.15 table |
| Private tags total | 3,410 | 2.75 per instance |
| Private tags removed | 3,410 | All, no safe-private retention |
| UIDs remapped | 3 per instance | Study, series, SOP |
| Date attributes shifted | 4 per instance | Study, series, acquisition, content |
| Instances with burned-in text flagged | 42 | 3.4% of 1,240 |
| Instances with annotations confirmed | 17 | Visual review of flagged sample |

The burned-in flag rate is 42 / 1,240 = 3.387%, or 3.4%. The confirmed rate is 17 / 42 = 40.5% of flagged
instances, which illustrates the false-positive rate of OCR on chest CT.

### 6.3 Date shift

The patient key `MRN12345` produces a keyed-hash offset of 56 days.

- Study date 2024-03-14 becomes 2024-05-09.
- Birth date 1950-03-14 becomes 1950-05-09.
- Age at study: 2024-03-14 − 1950-03-14 = 74 years. After shift: 2024-05-09 − 1950-05-09 = 74 years.
  The interval is preserved exactly because the same offset applies to both.
- Interval from a prior study on 2023-11-02 to the index study: 133 days. After shift, the prior study
  becomes 2023-12-28 and the index study becomes 2024-05-09; the interval is still 133 days.

If the birth date were shifted by a different offset, the age would change by the difference. That is the
most common date-shift defect: shifting study dates and forgetting birth dates, or applying a per-file
random offset.

### 6.4 Storage

A 512 × 512 image with 16 bits allocated per pixel:

- Bytes per image: 512 × 512 × 2 = 524,288 bytes.
- Study size before compression: 1,240 × 524,288 = 650,117,120 bytes = 620.0 MiB (650,117,120 /
  1,048,576 = 620.0).
- At a 2.5:1 compression ratio for lossless or near-lossless archival, approximately 248 MiB per study.

For 180,000 studies per year at 620 MiB each, the volume is 180,000 × 650,117,120 bytes =
117,021,081,600,000 bytes = 117.0 TB in decimal units (10^12) = 106.4 TiB in binary units (2^40). The
unit distinction matters: an estimate that mixes TB and TiB is off by about 10%, which at this scale is
more than 10 TB. A storage estimate that does not state its unit convention is not an estimate.

### 6.5 Linkage

Of the 1,240 instances, all belong to one patient. The study links to a hospital encounter through the
accession number:

- 1,240 / 1,240 = 100% of instances have an accession number.
- 1,180 / 1,240 = 95.2% link to an encounter by accession number.
- 48 / 1,240 = 3.9% link by patient and study date within ±1 day.
- 12 / 1,240 = 1.0% remain unlinked.

The overall linkage rate is (1,180 + 48) / 1,240 = 99.0%. The unlinked 12 instances are a second
reconstruction from an outside hospital, with no local encounter. They are retained in the archive with a
null encounter and a flag, not dropped.

### 6.6 Release contents

The release package contains the de-identified DICOM objects, a metadata table, a manifest with hashes, a
de-identification report stating the PS3.15 profile and options, the OCR method and its limitations, the
date-shift policy, the UID policy, and the linkage report. The privacy officer reviews the package and
accepts the residual risk in writing.

### 6.7 A second series that exposes a silent failure

Series 2 is the arterial phase and behaves as expected. Series 3 is the venous phase, and it is the series
that exposes a defect the first one hides.

| Metric | Series 2 | Series 3 | Expected |
|---|---:|---:|---|
| Instances received | 320 | 318 | 320 |
| Expected instances | 320 | 320 | 320 |
| Completeness | 100.0% | 99.375% | 100% |
| Slice spacing, median | 1.25 mm | 1.25 mm | 1.25 mm |
| Slice spacing, maximum | 1.25 mm | 2.50 mm | 1.25 mm |
| Coverage by instance count | 400.0 mm | 397.5 mm | 400.0 mm |
| QC result | pass | fail | — |

Series 3 has 318 of 320 instances. The missing two are adjacent, which produces a 2.5 mm gap at slice 190.
The coverage computed as `instances × spacing` is 397.5 mm, only 0.6% below the expected 400.0 mm, so a
coverage check with a 1% tolerance passes. The spacing check fails: the maximum spacing is 2.0 times the
median.

The analytic consequence is concrete. A volumetric measurement that assumes contiguous 1.25 mm slices
under-counts the volume by one slice interval in the gap, and a radiomics feature that uses neighboring
slices crosses a 2.5 mm discontinuity. The series must be excluded from volumetric features or
interpolated with a documented method.

### 6.8 Resolved metadata schema

| Column | Type | Source tag | Rule |
|---|---|---|---|
| `study_uid` | text | `(0020,000D)` | Remapped |
| `series_uid` | text | `(0020,000E)` | Remapped |
| `instance_uid` | text | `(0008,0018)` | Remapped |
| `modality` | text | `(0008,0060)` | Keep |
| `body_part` | text | `(0018,0015)` | Controlled vocabulary |
| `study_date` | date | `(0008,0020)` | Shifted per patient |
| `series_description` | text | `(0008,103E)` | Cleaned |
| `slice_thickness` | numeric | `(0018,0050)` | mm |
| `pixel_spacing` | text | `(0028,0030)` | mm, two values |
| `instances_expected` | integer | `(0020,1206)` or manifest | Null if absent |
| `qc_status` | text | derived | pass, incomplete, spacing |

The schema separates acquisition facts from QC judgments. A consumer can filter on `qc_status` without
recomputing the spacing check, and the QC result is versioned with the release.

### 6.9 Reconciliation from files to series

| Stage | Count | Calculation |
|---|---:|---|
| Files in export directory | 1,248 | — |
| Foreign study instances | 8 | Study UID mismatch |
| Instances in target study | 1,240 | 1,248 − 8 |
| Series | 4 | Distinct series UIDs |
| Instances in series 1 | 2 | Scout |
| Instances in series 2 | 320 | Arterial |
| Instances in series 3 | 318 | Venous, incomplete |
| Instances in series 4 | 600 | Delayed |
| Sum of series | 1,240 | 2 + 320 + 318 + 600 |

The reconciliation matches. A sum of 1,238 would mean two instances were dropped in assembly; a sum of
1,242 would mean two were double-counted across series.

### 6.10 Second run and de-identification digest

| Metric | Run 1 | Run 2 | Expected |
|---|---:|---:|---|
| Instances processed | 1,240 | 1,240 | Equal |
| UIDs remapped | 3,720 | 3,720 | Equal |
| Private tags removed | 3,410 | 3,410 | Equal |
| De-identified digest | `b72f...09de` | `b72f...09de` | Equal |

The UID remapping is deterministic for the same secret, so the digest is stable. If the digest changes,
the usual cause is a random salt or a timestamp embedded in the remapped UID.

### 6.11 Diagnostic checks

**Check A: series completeness.** `318 / 320 = 99.375%`. Threshold 100% for volumetric series. Fail. The
series is usable for slice-level reading and not for volumetrics.

**Check B: spacing uniformity.** Maximum spacing / median spacing = `2.50 / 1.25 = 2.0`. Threshold 1.1.
Fail. The gap is localized at slice 190 and is not a global reconstruction change.

**Check C: date interval preservation.** The interval from the prior study to the index study is 133 days
before and after shifting. Pass. A difference means the birth date or a study date was shifted with a
different offset.

### 6.12 Decision table for a series QC failure

| Symptom | First check | Likely cause | Action |
|---|---|---|---|
| Completeness <100% | Expected count source | Transfer interrupted | Re-retrieve missing instances |
| Spacing max/median >1.1 | Slice locations | Missing slice or mixed reconstruction | Exclude from volumetrics |
| Duplicate SOP UIDs | Hash comparison | Re-encoded export | Keep one; quarantine the other |
| Foreign study UIDs | Study UID mismatch | Filename collision | Quarantine; never release |
| Burned-in text flagged | OCR plus review | Annotation in pixels | Clean pixels; re-audit |
| UID digest changes | Remap function | Random salt | Use deterministic keyed hash |

## 7. Code, query, or calculation

### 7.1 DICOM Part 10 header parser and de-identifier (standard library, executable)

The script constructs a minimal explicit-VR little-endian DICOM file, parses its attributes, applies
header de-identification, remaps UIDs, shifts dates per patient, and verifies the result. It does not
process pixel data, which is the limitation the prose warns about. It was executed on CPython 3.11.15.

```python
"""Minimal DICOM Part 10 header parser and de-identifier. Standard library only."""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import struct

SECRET = b"research-deid-key-2026"


def element(group: int, elem: int, vr: str, value: bytes) -> bytes:
    if len(value) % 2:
        value += b"\x00" if vr == "UI" else b" "
    return (
        struct.pack("<HH", group, elem)
        + vr.encode("ascii")
        + struct.pack("<H", len(value))
        + value
    )


def build_file() -> bytes:
    ts = b"1.2.840.10008.1.2.1\x00"          # explicit VR little endian
    impl = b"1.2.826.0.1.3680043.8.498.1\x00"
    meta_body = (
        element(0x0002, 0x0010, "UI", ts)
        + element(0x0002, 0x0012, "UI", impl)
    )
    meta = element(0x0002, 0x0000, "UL", struct.pack("<I", len(meta_body))) + meta_body

    dataset = b"".join([
        element(0x0008, 0x0018, "UI", b"1.2.3.4.5.6"),
        element(0x0008, 0x0020, "DA", b"20240105"),
        element(0x0008, 0x0060, "CS", b"CT"),
        element(0x0008, 0x0070, "LO", b"ACME"),
        element(0x0008, 0x1030, "LO", b"CHEST CT"),
        element(0x0010, 0x0010, "PN", b"DOE^JOHN"),
        element(0x0010, 0x0020, "LO", b"MRN12345"),
        element(0x0010, 0x0030, "DA", b"19500314"),
        element(0x0011, 0x0010, "LO", b"ACME_PRIVATE"),
        element(0x0011, 0x1010, "LO", b"secret"),
        element(0x0018, 0x0050, "DS", b"1.25"),
        element(0x0020, 0x000D, "UI", b"1.2.3.4"),
        element(0x0020, 0x000E, "UI", b"1.2.3.4.5"),
        element(0x0028, 0x0010, "US", struct.pack("<H", 512)),
        element(0x0028, 0x0011, "US", struct.pack("<H", 512)),
        element(0x0028, 0x0030, "DS", b"0.7\\0.7"),
    ])
    return b"\x00" * 128 + b"DICM" + meta + dataset


def parse(blob: bytes) -> list[tuple[int, int, str, bytes]]:
    if blob[128:132] != b"DICM":
        raise ValueError("not a DICOM Part 10 file")
    pos = 132
    out = []
    while pos < len(blob):
        group, elem = struct.unpack_from("<HH", blob, pos)
        vr = blob[pos + 4:pos + 6].decode("ascii")
        length = struct.unpack_from("<H", blob, pos + 6)[0]
        value = blob[pos + 8:pos + 8 + length]
        out.append((group, elem, vr, value))
        pos += 8 + length
    return out


def text(value: bytes) -> str:
    return value.rstrip(b"\x00 ").decode("ascii", "replace")


def get(elements, group: int, elem: int) -> bytes:
    for g, e, _vr, value in elements:
        if g == group and e == elem:
            return value
    return b""


def shifted(date_text: str, days: int) -> str:
    d = dt.datetime.strptime(date_text, "%Y%m%d").date() + dt.timedelta(days=days)
    return d.strftime("%Y%m%d")


def new_uid(original: str) -> str:
    digest = hmac.new(SECRET, original.encode(), hashlib.sha256).digest()
    number = int.from_bytes(digest[:16], "big")
    return f"2.25.{number}"


def deidentify(elements, patient_key: str, offset_days: int):
    uid_tags = {(0x0008, 0x0018), (0x0020, 0x000D), (0x0020, 0x000E)}
    date_tags = {(0x0008, 0x0020), (0x0010, 0x0030)}
    out = []
    for g, e, vr, value in elements:
        if g % 2 == 1:                      # private tag: remove
            continue
        if (g, e) == (0x0010, 0x0010):      # PatientName: zero length
            out.append((g, e, vr, b""))
            continue
        if (g, e) == (0x0010, 0x0020):      # PatientID: token
            token = hmac.new(SECRET, patient_key.encode(), hashlib.sha256).hexdigest()[:12]
            out.append((g, e, vr, token.encode()))
            continue
        if (g, e) == (0x0008, 0x1030):      # StudyDescription: clean
            out.append((g, e, vr, b"CHEST CT"))
            continue
        if (g, e) in uid_tags:
            out.append((g, e, vr, new_uid(text(value)).encode() + b"\x00"))
            continue
        if (g, e) in date_tags and text(value):
            out.append((g, e, vr, shifted(text(value), offset_days).encode()))
            continue
        out.append((g, e, vr, value))
    return out


def serialize(elements) -> bytes:
    meta_body = b"".join(
        element(g, e, vr, v) for g, e, vr, v in elements if g == 0x0002
    )
    meta = element(0x0002, 0x0000, "UL", struct.pack("<I", len(meta_body))) + meta_body
    dataset = b"".join(
        element(g, e, vr, v) for g, e, vr, v in elements if g != 0x0002
    )
    return b"\x00" * 128 + b"DICM" + meta + dataset


original = build_file()
elements = parse(original)
patient_id = text(get(elements, 0x0010, 0x0020))
offset = int.from_bytes(
    hmac.new(SECRET, patient_id.encode(), hashlib.sha256).digest()[:2], "big"
) % 365 + 1

print("elements", len(elements))
print("patient_name", text(get(elements, 0x0010, 0x0010)))
print("modality", text(get(elements, 0x0008, 0x0060)))
print("rows", struct.unpack("<H", get(elements, 0x0028, 0x0010))[0])
print("columns", struct.unpack("<H", get(elements, 0x0028, 0x0011))[0])
print("pixel_spacing", text(get(elements, 0x0028, 0x0030)))
print("study_uid", text(get(elements, 0x0020, 0x000D)))
print("private_tags", sum(1 for g, _e, _vr, _v in elements if g % 2 == 1))

clean = deidentify(elements, patient_id, offset)
clean_bytes = serialize(clean)
clean_elements = parse(clean_bytes)

print("date_shift_days", offset)
birth_before = text(get(elements, 0x0010, 0x0030))
study_before = text(get(elements, 0x0008, 0x0020))
birth_after = text(get(clean_elements, 0x0010, 0x0030))
study_after = text(get(clean_elements, 0x0008, 0x0020))
interval_before = (
    dt.datetime.strptime(study_before, "%Y%m%d")
    - dt.datetime.strptime(birth_before, "%Y%m%d")
).days
interval_after = (
    dt.datetime.strptime(study_after, "%Y%m%d")
    - dt.datetime.strptime(birth_after, "%Y%m%d")
).days
print("interval_preserved", interval_before == interval_after)
print("study_uid_after", text(get(clean_elements, 0x0020, 0x000D)))
print("private_tags_after", sum(1 for g, _e, _vr, _v in clean_elements if g % 2 == 1))
residual = [
    marker for marker in (b"DOE^JOHN", b"MRN12345", b"19500314", b"20240105")
    if marker in clean_bytes
]
print("residual_identifiers", len(residual))
```

Expected output:

```text
elements 19
patient_name DOE^JOHN
modality CT
rows 512
columns 512
pixel_spacing 0.7\0.7
study_uid 1.2.3.4
private_tags 2
date_shift_days 56
interval_preserved True
study_uid_after 2.25.280083925618065778039680259664266364586
private_tags_after 0
residual_identifiers 0
```

The UID value is deterministic for this secret and input; a different secret produces a different UID.
The date shift is 56 days, and the interval between birth and study is unchanged. Two private tags are
removed. None of the original identifiers appears in the cleaned bytes.

### 7.2 Pixel-burned-in audit query (SQL, PostgreSQL 18)

The following query targets PostgreSQL 18 and a metadata table populated by an OCR audit. It reports the
flag rate by series description and the review status.

```sql
-- PostgreSQL 18
SELECT
    series_description,
    COUNT(*)                                                   AS instances,
    SUM(CASE WHEN ocr_flag THEN 1 ELSE 0 END)                  AS flagged,
    ROUND(100.0 * SUM(CASE WHEN ocr_flag THEN 1 ELSE 0 END)
          / COUNT(*), 2)                                       AS flagged_pct,
    SUM(CASE WHEN reviewed AND confirmed_text THEN 1 ELSE 0 END) AS confirmed,
    SUM(CASE WHEN reviewed AND NOT confirmed_text THEN 1 ELSE 0 END) AS false_positive
FROM imaging.ocr_audit
GROUP BY series_description
HAVING COUNT(*) >= 20
ORDER BY flagged_pct DESC;
```

The `HAVING` clause removes series with too few instances to estimate a rate. A series with two scout
images and one flag has a 50% flag rate, which is not comparable to a 600-image series.

## 8. Data, standards, and regulatory context

### 8.1 The DICOM standard

| Part | Content | Relevance |
|---|---|---|
| PS3.1 | Introduction and structure | Citation conventions |
| PS3.3 | Information object definitions | Attributes per object |
| PS3.5 | Data structures and encoding | Transfer syntax, VR |
| PS3.6 | Data dictionary | Tags and VRs |
| PS3.10 | Media storage and file format | Part 10 files |
| PS3.15 | Security and system management profiles | De-identification |
| PS3.18 | Web services | QIDO-RS, WADO-RS, STOW-RS |

The base standard is updated several times a year. As of September 2026 the current base edition is 2026d,
with Final Text supplements and correction proposals layered on top. The preferred citation is undated:
"NEMA PS3 / ISO 12052, Digital Imaging and Communications in Medicine (DICOM) Standard." ISO 12052:2026
(Edition 3, published 18 February 2026) is the international counterpart.

Conformance is declared against SOP classes, transfer syntaxes, and profiles, not against an edition. A
statement that a tool is "DICOM 2026 compliant" is not meaningful without the service classes and
profiles it supports.

### 8.2 PS3.15 Annex E

The Attribute Confidentiality Profiles are normative. The Basic Application Level Confidentiality Profile
defines actions for each attribute, including patient identity, demographics, responsible parties,
institution, device identity, dates, and UIDs. The options include:

- Clean Pixel Data.
- Clean Recognizable Visual Features.
- Clean Graphics.
- Clean Structured Content.
- Clean Descriptors.
- Retain Longitudinal Temporal Information (Full Dates or Modified Dates).
- Retain Patient Characteristics.
- Retain Device Identity.
- Retain Institution Identity.
- Retain UIDs.
- Retain Safe Private.

The Basic Profile does not address pixel data unless a clean-pixel option is selected. This is stated in
the standard and is the most consequential detail in the profile.

### 8.3 DICOMweb

PS3.18 defines the HTTP services:

- **QIDO-RS** (Query) at §10.6: search studies, series, and instances with attribute matching, paging, and
  `includefield`.
- **WADO-RS** (Retrieve) at §10.4: retrieve metadata, rendered images, thumbnails, bulk data, pixel data,
  and frames.
- **STOW-RS** (Store) at §10.5: store instances with multipart requests and per-instance status.

The services return `application/dicom+json` for metadata. A pipeline can query and validate metadata
without transferring pixels, which is the efficient path described in Step 4.

### 8.4 Regulatory and ethical context

- **HIPAA.** DICOM files are PHI when they contain identifiable attributes or pixels. De-identification
  under 45 CFR 164.514 requires either safe harbor or expert determination; a DICOM profile is a
  technical method, not a legal determination.
- **FDA.** Imaging in a regulated trial may be subject to 21 CFR Part 11 for electronic records. Audit
  trails, version control, and attributable changes apply.
- **IRB and data use agreements.** A research release usually requires an IRB determination and a DUA.
  The de-identification report is part of the submission.
- **GDPR.** If EU data are involved, imaging is personal data and may include special-category health
  data. The lawful basis and transfer mechanism apply.

### 8.5 What the standard leaves open

- Whether to retain UIDs.
- Which date-shift option to use.
- How to detect burned-in text.
- How to validate a de-identified object.
- Whether the residual risk is acceptable.
- How to link to clinical data.

Those are engineering and governance decisions. The standard defines the transformations; it does not
certify the result.

## 9. Industry practice and case evidence

### 9.1 PACS, VNA, and research archives

Clinical imaging lives in a PACS or VNA, governed by clinical policy and optimized for diagnosis. Research
imaging usually lives in a separate archive with de-identified objects and a metadata index. Mixing the
two is a governance failure: a research user with archive access can query clinical identifiers, and a
clinical user can modify research objects.

### 9.2 Public imaging archives

The Cancer Imaging Archive (TCIA) publishes de-identified imaging collections with clinical metadata and
documented curation. Its existence demonstrates that de-identified imaging can support research at scale;
it does not demonstrate that any arbitrary de-identification pipeline is sufficient. TCIA collections are
curated, reviewed, and accompanied by data use agreements.

### 9.3 Burned-in annotation

The DICOM standard includes Clean Pixel Data and Clean Recognizable Visual Features options precisely
because headers are insufficient. Published imaging privacy work has shown that burned-in text and
metadata can re-identify patients. The engineering lesson is that de-identification must cover both
layers and must be validated on the pixels, not only on the tags.

### 9.4 What is inference

- That a particular OCR tool detects most burned-in text. Detection rates vary by modality, font, and
  contrast; the rate must be measured on the archive.
- That UID remapping prevents linkage. It prevents direct linkage to the source UID; other quasi-
  identifiers may still permit linkage.
- That a de-identified archive is safe to share. Safety depends on the DUA, the residual-risk assessment,
  and the recipient's controls, not only on the transformation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Assembly by file name | Series count does not match the archive | Directory prefix collision | Mixed-patient series | Assemble by UID |
| Partial header cleaning | Free-text description contains a name | Blocklist of ten tags | Re-identification | Apply full PS3.15 table |
| Private tags retained | Vendor worklist ID in output | Not in the blocklist | Identifier leak | Remove all private tags by default |
| UID inconsistency | Viewer cannot reconstruct the series | Per-file remapping | Broken hierarchy | Deterministic remap per UID |
| Per-file date shift | Intervals change between studies | Offset recomputed per instance | Invalid longitudinal analysis | One offset per patient |
| Birth date unshifted | Age differs after de-identification | Date tag missed | Age disclosure or error | Shift all date attributes |
| Burned-in text ignored | Annotation visible in a screenshot | Pixels not cleaned | Identifier leak | Clean Pixel Data; OCR audit |
| Over-cleaning | Object fails to open | Required attribute removed | Data loss | Validation parse and viewer test |
| Incomplete series | 318 of 320 instances | Transfer interruption | Reconstruction artifact | Completeness report |
| Transfer syntax assumption | Garbled pixels | Implicit VR or compressed data | Unusable images | Record and support syntaxes |
| No linkage report | Cohort missing imaging | Accession mismatch | Selection bias | Linkage rate by method |
| De-id tool unversioned | Cannot reproduce the release | Unknown configuration | Audit failure | Record tool, version, options |

Two red flags deserve emphasis.

**A de-identification report with no pixel section.** If the report discusses only headers, the pixels were
not assessed. The report should state the pixel method, the OCR tool, the sample reviewed, and the
confirmed findings.

**A storage path containing a patient name or date.** The path leaks the identifier even if the file
content is clean. Paths should use remapped UIDs or hashes.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| UIDs are unique within the export | Assembly and remapping | Collision or re-encoding | Duplicate UID audit with hash comparison |
| The archive's instance count is accurate | Completeness check | Metadata lag | Compare to retrieved manifest |
| Date shift preserves intervals | Longitudinal analysis | Per-file offsets | Compare intervals before and after |
| OCR detects most burned-in text | Pixel audit | Stylized or low-contrast text | Visual review of a sample |
| The de-id tool implements PS3.15 | Profile compliance | Partial implementation | Tag-level validation |
| Remapped UIDs are deterministic | Reproducibility | Random salt per run | Re-run and compare |
| Linkage by accession is reliable | Cohort construction | Accession reuse | Linkage rate and manual review |
| The recipient honors the DUA | Residual risk | Re-sharing | Contract and audit |
| Pixel cleaning does not alter diagnosis | Research utility | Aggressive masking | Radiologist review of a sample |

The largest limit is that de-identification is not anonymization in the mathematical sense. Date shifting
preserves intervals but allows boundary inference; UID remapping preserves linkage within the dataset;
demographic retention can permit linkage with external data. The residual risk must be assessed, not
assumed away.

A second limit is that the code in Section 7 parses headers only. It is a teaching parser for explicit-VR
little-endian files, not a production DICOM library. A production pipeline needs a library that supports
all transfer syntaxes, compressed pixel data, structured reports, and presentation states.

## 12. Exercises

### Exercise 1 — Series assembly (application, 40 minutes)

Given a manifest of 1,248 files with 8 foreign instances, deliverable: a script that groups by UID and
outputs study, series, and instance counts plus a quarantine list.

Acceptance criteria: the foreign instances are isolated by UID, not by file name; counts reconcile to
1,240; the quarantine list includes the reason.

### Exercise 2 — Metadata extraction (application, 45 minutes)

Deliverable: a table of 12 metadata fields per study, including modality, body part, study date, series
count, slice thickness, pixel spacing, manufacturer, and contrast flag.

Acceptance criteria: missing values are null, not imputed; units are stated; the extraction does not read
pixel data.

### Exercise 3 — Profile selection (design, 40 minutes)

Deliverable: a profile and options selection for two studies: a longitudinal tumor-growth analysis and a
one-time algorithm validation.

Acceptance criteria: the longitudinal study retains intervals; the validation study minimizes identifiers;
each option has a stated rationale and a stated cost.

### Exercise 4 — Date shift verification (application, 45 minutes)

Deliverable: a function that shifts all date attributes per patient and a test that verifies three
intervals are preserved.

Acceptance criteria: one offset per patient; birth date included; the test covers a leap year and a
year boundary.

### Exercise 5 — Pixel audit (critique, 45 minutes)

Deliverable: an audit plan for burned-in text with sampling, OCR thresholds, visual review, and a
reporting format.

Acceptance criteria: the plan states the expected false-positive rate; the sample size is justified; the
report distinguishes flagged from confirmed.

### Exercise 6 — Storage and index design (design, 60 minutes)

Deliverable: an object layout and a metadata schema for 180,000 studies per year, with lifecycle tiers and
access controls.

Acceptance criteria: paths contain no identifiers; the index supports cohort queries by modality, body
part, and date; lifecycle rules match the DUA.

### Exercise 7 — Linkage (application, 60 minutes)

Deliverable: a linkage table with method, confidence, and rate for 1,240 instances, plus a plan for the
unlinked 1%.

Acceptance criteria: methods are ordered by reliability; ambiguous matches are flagged; the unlinked
cases are retained, not dropped.

### Exercise 8 — De-identification review (critique, 40 minutes)

Given a report that lists only removed tags, deliverable: a 400-word critique identifying at least five
missing elements of a complete report.

Acceptance criteria: covers profile and options, UID policy, date policy, pixel audit, validation, and
residual risk; no claim that the archive is unsafe without evidence.

## 13. Self-check answers

**1. Why assemble by UID instead of file name?**

Because file names are assigned by exporters and can collide, change, or encode identifiers. UIDs are the
standard's identity mechanism and are preserved across copies. Assembly by file name mixes studies and
loses series.

**2. What are the three layers of a DICOM object?**

File and transfer encoding, header attributes, and pixel data. De-identification must address all three;
a header cleaner alone leaves burned-in text and encoding-level metadata.

**3. What does the Basic Profile do about pixels?**

By default, nothing. Pixel cleaning requires the Clean Pixel Data or Clean Recognizable Visual Features
option. The Basic Profile removes or modifies header attributes and does not inspect pixel values.

**4. Why is per-file date shifting wrong?**

Because it changes the intervals between studies for the same patient. A longitudinal analysis of tumor
growth or disease progression depends on those intervals. One offset per patient preserves them.

**5. Compute the study size.**

1,240 × 512 × 512 × 2 bytes = 1,240 × 524,288 = 650,117,120 bytes = 620.0 MiB. At 2.5:1 compression,
approximately 248 MiB.

**6. Why remap UIDs consistently?**

Because references within and across objects depend on them. Inconsistent remapping breaks series
assembly and structured-report references. Consistent remapping preserves internal linkage while removing
the source identifier.

**7. What is the risk of retaining UIDs?**

A retained UID is a pseudonymous identifier that can be resolved by anyone with access to the source
archive or to a crosswalk. It preserves linkage to the original study, which may be re-identifying in
combination with other data.

**8. Compute the burned-in flag and confirmation rates.**

42 / 1,240 = 3.387%, or 3.4%. Confirmed 17 of 42 flagged = 40.5%. The confirmation rate is not the true
prevalence; it is the yield of the OCR screen after review.

**9. Why validate that a de-identified object still parses?**

Because over-cleaning can remove required attributes or corrupt the encoding. An object that passes a text
scan but fails to open in a viewer is a failed de-identification.

**10. What belongs in a release package?**

De-identified objects, a manifest with hashes, a metadata table, the profile and options, tool versions,
date and UID policies, pixel audit results, linkage rates, validation results, and a residual-risk
statement reviewed by the privacy officer.

## 14. Interview questions

**1. "How would you build an imaging cohort?"**

Strong: define the clinical question, query metadata, assemble by UID, validate completeness, select a
de-identification profile, clean headers and pixels, validate, index, link, and QC. Weak: "export the
folder and zip it."

**2. "What does DICOM de-identification require?"**

Strong: names PS3.15 Annex E, the Basic Profile and options, and the pixel gap. Weak: "remove the patient
name tag."

**3. "How do you handle UIDs?"**

Strong: deterministic keyed remapping, consistent across references, with a secure crosswalk if
re-linkage is needed. Weak: "delete them," which breaks the hierarchy.

**4. "How do you shift dates?"**

Strong: one offset per patient, applied to all date attributes, with interval verification. Weak: "add a
random number of days to each file."

**5. "What is burned-in text, and how do you find it?"**

Strong: text rendered into pixels, detected by OCR plus visual review, with measured false positives and
negatives. Weak: "the header flag tells us."

**6. "How do you know a series is complete?"**

Strong: compare received instances to the expected count from metadata or protocol, and report per
series. Weak: "the viewer opens it."

**7. "What is the difference between a PACS and a research archive?"**

Strong: governance, access, and purpose; the PACS is clinical and optimized for diagnosis, the research
archive is de-identified and optimized for analysis. Weak: "they both store images."

**8. "How do you link imaging to clinical data?"**

Strong: enterprise identifier, then accession, then date and modality, with a reported linkage rate and
ambiguous-match handling. Weak: "join on study date."

**9. "What would you put in the metadata index?"**

Strong: modality, body part, dates, series description, acquisition parameters, manufacturer, linkage
keys, and QC flags. Weak: "the DICOM header."

**10. "When is de-identification not enough?"**

Strong: when residual quasi-identifiers, date shifting, or free text permit linkage, or when the DUA
requires a controlled environment. Weak: "never, the standard guarantees anonymity."

## 15. Key terms

- **Action code** — PS3.15 instruction D, Z, X, K, C, U.
- **Burned-in annotation** — text rendered into pixels.
- **Confidentiality profile** — normative de-identification rule set.
- **DICOM** — imaging communication standard.
- **DICOMweb** — HTTP services for DICOM.
- **Instance** — single DICOM object.
- **IOD** — information object definition.
- **Linkage rate** — share of studies linked to clinical data.
- **PACS** — clinical picture archive.
- **Part 10 file** — DICOM file with preamble and meta information.
- **Photometric interpretation** — pixel color model.
- **Pixel spacing** — physical pixel size.
- **Private tag** — vendor-defined attribute.
- **QIDO-RS** — DICOMweb query.
- **Residual risk** — risk remaining after de-identification.
- **Series** — acquisition group within a study.
- **Series completeness** — received versus expected instances.
- **Slice thickness** — reconstruction spacing.
- **SOP class** — object or service definition.
- **SOPInstanceUID** — instance identifier.
- **STOW-RS** — DICOMweb store.
- **Structured report** — DICOM object with clinical content.
- **Study** — imaging encounter.
- **StudyInstanceUID** — study identifier.
- **Tag** — attribute group and element.
- **Transfer syntax** — dataset encoding.
- **UID** — unique identifier.
- **UID remapping** — replacing UIDs consistently.
- **VNA** — vendor-neutral archive.
- **VR** — value representation.
- **WADO-RS** — DICOMweb retrieve.

## 16. Sources

All URLs accessed 25 October 2026.

1. NEMA. "DICOM Current Edition." https://www.dicomstandard.org/current
2. NEMA. "PS3.1" (2026d). https://dicom.nema.org/medical/dicom/current/output/html/part01.html
3. NEMA. "PS3.15 Annex E, Attribute Confidentiality Profiles."
   https://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_E.html
4. NEMA. "PS3.15 E.2 Basic Application Level Confidentiality Profile."
   https://dicom.nema.org/medical/dicom/current/output/chtml/part15/sect_E.2.html
5. NEMA. "PS3.15 E.3 Basic Application Level Confidentiality Options."
   https://dicom.nema.org/medical/Dicom/current/output/chtml/part15/sect_E.3.html
6. NEMA. "PS3.15 (PDF)." https://dicom.nema.org/medical/dicom/current/output/pdf/part15.pdf
7. NEMA. "PS3.18 Web Services." https://dicom.nema.org/medical/dicom/current/output/html/part18.html
8. DICOM. "DICOMweb." https://www.dicomstandard.org/using/dicomweb
9. DICOM. "STOW-RS." https://www.dicomstandard.org/using/dicomweb/store-stow-rs
10. ISO. "ISO 12052:2026." https://www.iso.org/standard/86414.html
11. The Cancer Imaging Archive. https://www.cancerimagingarchive.net/
12. HHS. "Guidance Regarding Methods for De-identification of PHI."
    https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
13. eCFR. "45 CFR 164.514."
    https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514
14. eCFR. "21 CFR Part 11." https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_fhir-ingestion.md`](05_fhir-ingestion.md)
- Next topic: [`07_data-versioning-and-lineage.md`](07_data-versioning-and-lineage.md)
- Related sibling: [`08_deidentification-engineering.md`](08_deidentification-engineering.md)
- Root survey: [`../../03_data_engineering_interoperability.md`](../../course/03_data_engineering_interoperability.md)
- Labs: [`../../labs/`](../../labs/) — Lab 3.5 audits a synthetic dataset for identifiers, including DICOM if available.
