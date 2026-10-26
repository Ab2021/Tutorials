# Pathology

> Module 23 — Specialty Electives
> Topic 03 of 07
> Estimated study time: 3.5 hours
> Prerequisites: 01, 08
> Version 2026.3

> **Note:** This file is educational. It is not medical advice, not a diagnostic authority,
> and not a substitute for the current World Health Organization (WHO) Classification of
> Tumours, College of American Pathologists protocol, or institutional laboratory procedure. Every threshold below names
> its source and version, and where a rule has changed the superseded and current versions
> are both given.

## 1. Why this topic exists

Pathology is where a piece of tissue becomes a diagnosis, a grade, a margin, and increasingly
a biomarker that selects a drug. It is the rate-limiting step for most cancer treatment, and
it is the only specialty in this module where the primary data object is a physical artifact
that is cut, stained, mounted, scanned, and then interpreted by a human whose judgement is
the reference standard.

Three properties make pathology data unusually easy to break.

**The unit of adoption is the slide and the scanner, not the patient.** A model that performs
well on one scanner model has a small addressable market and a large validation problem. A
laboratory buys a scanner as a capital purchase and an algorithm as a recurring licence, and
the algorithm's value is bounded by the scanners in that laboratory. When a vendor reports a
single number, the buyer's first question should be which instrument produced it.

**The label is a human judgement with measurable, irreducible disagreement.** Two
uropathologists grading the same prostate biopsy agree at a quadratic weighted kappa of
roughly 0.91, and two general pathologists agree at roughly 0.77. Those numbers are the
ceiling, not the floor. A model that reports 0.93 against a reference whose inter-reader
agreement is 0.91 has not been shown to be superhuman; it has been shown to be close to the
reference reader, and the difference may be noise.

**The classification changes faster than the data.** The WHO Classification of Tumours
completed its fifth edition in 2026 and began its sixth edition in July 2026. A grade or an
entity assigned in 2019 may not be expressible in the current scheme, and a molecular
definition introduced in one edition changes the denominator of every histology-specific
analysis that spans the boundary.

### 1.1 The ordinary path of one case

A colonoscopy yields a polyp. The specimen is fixed in formalin, and the fixation time is
recorded on a requisition if the laboratory is well run. A histotech embeds it in a block,
cuts sections four micrometres thick, and stains them with haematoxylin and eosin (H&E) in a
batch with forty other cases. A pathologist reads the slide, assigns a grade, and dictates a
synoptic report. The slide is scanned at 40x, the image is stored as a pyramidal whole-slide
image (WSI), and a research team later extracts tiles for a model. Six artifacts now exist for
one patient, and only one of them is the patient.

### 1.2 The signature failures

- **Slide and scanner leakage.** Tiles from one slide appear in both training and test sets,
  or slides from one patient appear in both, and the model learns the batch rather than the
  biology. The inflation is large and it is invisible in a random split.
- **Fixation and stain batch confounding.** A batch that was stained on a Friday is slightly
  different from one stained on a Monday, and if the case mix differs by day the batch becomes
  a confounder that looks like a feature.
- **Grading disagreement treated as model error.** A model is judged against a single
  pathologist's read, and the disagreement that would exist between two pathologists is
  attributed to the model.
- **Molecular reclassification across WHO editions.** An entity defined by a fusion in the
  fifth edition did not exist as a category in the fourth, so a cohort pooled across the
  boundary contains cases that cannot be compared.
- **Synoptic reporting structure ignored.** The report is a set of required elements with
  controlled responses, and a model that reads free text inherits the dictation habits of one
  institution rather than the protocol's semantics.
- **Specimen hierarchy collapsed.** A patient with two blocks and four slides becomes one row,
  and the lesion size, margin, and grade from different blocks are averaged into a value no
  pathologist reported.

### 1.3 Who is blocked when the measurement is wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Patient | A therapy selected on a misclassified biomarker or grade | Cannot see the slide, the block, or the scoring rule |
| Pathologist | Time re-reviewing cases the model flagged for the wrong reason | No access to the training distribution or the scanner metadata |
| Oncologist | Confidence in a grade that disagrees with the clinical course | Cannot audit the reference standard |
| Laboratory manager | Accreditation exposure from an unvalidated tool in the workflow | The tool's validation was done elsewhere |
| Diagnostics company | A clearance that covers one scanner and one specimen type | Intended use is fixed at submission |
| Pharma sponsor | A trial enriched on a biomarker scored by an unvalidated assay | The assay and the threshold travel together |
| Data scientist | A model with excellent internal numbers and no external market | Batch structure was never recorded |

### 1.4 What this topic covers and what it does not

This topic covers the measurement layer: how tissue becomes a slide, how a slide becomes an
image, how an image becomes a grade or a score, how the reference standard is established,
and how each of those translations fails. It does not teach histology, molecular biology, or
oncology practice. It assumes you have met classification metrics and confidence intervals
elsewhere.

Most computational pathology curricula spend their time on architectures and almost none on
the fact that the same case can be grade 2 in the report, grade 3 in the tumour registry, and
excluded from a research cohort because the block could not be found. The second problem
destroys more projects than the first.

## 2. Learning outcomes

By the end of this topic you can:

| # | Outcome | Evidence of attainment |
|---:|---|---|
| 1 | Draw the specimen-to-pixel hierarchy and state the unit of analysis for a stated question | Exercise 1; self-check 1 |
| 2 | Compute quadratic weighted kappa (QWK) from a confusion matrix and explain why it is not comparable across category counts | Exercise 2; self-check 2 |
| 3 | Detect slide, patient, and scanner leakage and quantify the inflation each produces | Exercise 3; interview question 4 |
| 4 | Measure inter-pathologist agreement and use it as the ceiling for model claims | Exercise 4; self-check 4 |
| 5 | Apply the American Society of Clinical Oncology and College of American Pathologists (ASCO/CAP) HER2 immunohistochemistry (IHC) categories and in situ hybridisation (ISH) groups, including the work-up for groups 2 to 4 | Exercise 5; interview question 6 |
| 6 | Explain how a WHO edition change reclassifies cases and how to version the diagnosis field | Exercise 6; self-check 6 |
| 7 | Write a synoptic report extractor that respects core, conditional, and optional elements | Exercise 7; interview question 9 |
| 8 | Place a pathology product on the US Food and Drug Administration (FDA) device path or the laboratory-developed test route, and state the accreditation constraints | Exercise 8; interview question 10 |

Outcomes 1 and 3 are structural. They are separated because a learner can draw the hierarchy
and still split by tile, which is the error the hierarchy exists to prevent.

Outcomes 2 and 4 are the quantitative core. They are paired because kappa without an
inter-reader ceiling is a number without an interpretation.

Outcomes 5 and 6 are about the rulebook. They are the two places where a pathology model
encodes a version rather than a biology, and where a guideline update silently changes the
label.

Outcomes 7 and 8 connect the measurement layer to the two external systems that decide
whether the work can be used: the reporting protocol and the regulator.

### 2.1 How the outcomes map to the module

Outcomes 1 to 3 are prerequisites for the cross-specialty comparison in topic 07, where the
same leakage logic appears as record-level splitting in cardiology and patient-level splitting
in oncology. Outcome 4 is shared with oncology, where the equivalent object is the RECIST
read. Outcomes 5 and 8 connect to module 12 (omics) and module 14 (device regulation).

If you retain only one outcome, retain outcome 3. A pathology model with leakage is not a
weak model; it is a model whose reported performance is a property of the split, and no amount
of external validation repairs a result that was never measured.

Outcome 2 is the one most often reported without its caveat. A QWK of 0.83 on a five-category
scale and a QWK of 0.76 on a three-category collapse of the same predictions are both correct
and are not comparable, because the quadratic weights are normalised by the square of the
category count. Section 6 shows the size of that difference.

## 3. The core mental model

A pathology analysis is a chain of physical and interpretive transformations. Each arrow loses
information, and each loss is a place where a definition can be wrong.

```text
  Patient
    └── Specimen ......... procedure, site, collection date, laterality
          └── Block ..... cassette, block id, fixation time, tissue type
                └── Slide  section thickness, stain batch, coverslip, barcode
                      └── Image ... scanner, firmware, magnification, ICC profile
                            └── Tile ... coordinates, microns per pixel, tissue mask
                                  └── Pixel
```

Read the diagram as a set of parent keys, not a set of stages. Every analysis unit has a
parent, and every join must name the level. The most common structural defect is a slide
table keyed only on `patient_id`, so a patient with six slides has six rows that can be joined
in any order and the first one becomes the case.

### 3.1 The three clocks

Pathology has three clocks running at once, and confusing them is the root of most temporal
and batch leakage.

| Clock | Origin | Resolution | Used for | Typical error |
|---|---|---|---|---|
| Acquisition clock | Fixation, sectioning, staining, scanning | Minutes to hours | Pre-analytic covariates, batch effects, tissue quality | Using the scan date as if it were the collection date |
| Interpretation clock | Read and sign-out | Days | Turnaround time, diagnosis date, endpoint | Treating sign-out as the date the disease began |
| Classification epoch | WHO edition, CAP protocol version, ASCO/CAP guideline version | Years | Grade and entity definitions, biomarker thresholds | Pooling grades assigned under two schemes |

The acquisition clock is the one that makes pathology different from every other specialty in
this module. A slide carries the physical history of its preparation: how long the tissue sat
in formalin, which stain bath it passed through, how thick the section is, and which scanner
produced the image. Those variables are not in the diagnosis and they are frequently
predictive of it, because laboratories batch work by day and by technologist, and case mix
varies by day.

The interpretation clock is where cohort definitions are made. A "diagnosis date" in a
pathology warehouse is usually the sign-out date, which can be a week after the specimen was
collected and, for a referred consultation, a month after. For a survival analysis the
difference matters at the margins; for a turnaround-time analysis it is the entire measurement.

The classification epoch is the clock most teams forget. It is not a timestamp in the data; it
is a property of the rulebook applied at analysis time. The fifth edition of the WHO
classification was released volume by volume between 2019 and 2026, and the sixth edition
began in July 2026. A case diagnosed in 2021 was classified under whichever volume was current
for its organ, which is not the same date for every organ.

### 3.2 The bag model

A whole-slide image is too large to classify directly, so the standard formulation is
multiple-instance learning: the slide is a bag, the tiles are instances, the slide label is
known, and the tile labels are not.

```text
  slide label: "tumour present"
        │
        ├── tile 1 ..... unknown
        ├── tile 2 ..... unknown
        ├── tile 3 ..... unknown        the model learns which tiles
        └── tile n ..... unknown        support the bag label
```

The bag formulation is correct and it has a consequence that is easy to miss: the model's
output is a slide-level score, and any per-tile heat map is an explanation, not a
measurement. A pathologist who reads a heat map as a segmentation is reading a model artifact.

### 3.3 Where the decision threshold comes from

A pathology threshold is a property of the assay, the specimen, and the clinical action. For a
biomarker the rule is written in a guideline and the categories are defined by staining
intensity and extent, not by a continuous score.

```text
  IHC 3+ ......................... positive, treat
  IHC 2+ ......................... equivocal, reflex to ISH
  IHC 1+ or 0 .................... negative
```

The boundary between 2+ and 3+ is a judgement about "complete, intense circumferential
membrane staining in more than 10 percent of tumour cells", and two readers at the boundary
disagree. That is why the equivocal category exists: it converts a judgement call into a
reflex test with its own rule.

### 3.4 What the model is allowed to see

```text
  admissible ....................... | decision | ............. inadmissible
  H&E morphology at the diagnostic   |          |  the molecular result, the
  slide, clinical history available  |          |  final grade after consensus
  at the time of the read, specimen  |          |  review, the treatment response,
  site and procedure                 |          |  slides cut after the decision
```

The wall is drawn at the moment the report is signed. In a diagnostic aid, the wall is the
pathologist's read. In a biomarker model, it is the assay result that the model is trying to
replace. The common violations are the molecular result used as a feature for a morphology
model, and the consensus grade used as a feature when the model is supposed to predict it.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Histology | Tissue type and subtype | Treated as a free-text field | ICD-O morphology code |
| Haematoxylin and eosin (H&E) | Routine stain showing nuclei and cytoplasm | Assumed comparable across laboratories | Stain protocol identifier |
| Whole-slide image (WSI) | Digitised slide stored as a pyramidal image | Treated as a photograph with one resolution | DICOM (Digital Imaging and Communications in Medicine) Supplement 145; vendor formats |
| Pyramidal image | Multi-resolution image with a base level and downsampled levels | Level index used without microns per pixel | Level 0 to n |
| Microns per pixel (MPP) | Physical size of one pixel | Assumed constant across scanners | µm/pixel |
| Tile | Fixed-size crop extracted from a slide | Called a patch, a tile, or a crop interchangeably | Pixels, at a stated MPP |
| Magnification | Optical enlargement, conventionally 20x or 40x | Confused with resolution | x, with the objective |
| Specimen | Tissue submitted for examination | Used interchangeably with block or slide | Accession number |
| Block | Paraffin cassette containing tissue | Treated as the unit of diagnosis | Block identifier |
| Slide | Section mounted on glass or scanned | Treated as the patient | Slide barcode |
| Section thickness | Microtome setting | Not recorded | µm, usually 4 |
| Fixation time | Interval from excision to formalin penetration | Recorded as "fixation adequate" | Hours |
| Ischaemia time | Interval from devascularisation to fixation | Conflated with fixation time | Minutes |
| Stain batch | Set of slides stained together | Not recorded | Batch or run identifier |
| Coverslip artifact | Bubbles, folds, and mounting medium | Treated as image noise | Visual flag |
| Tissue mask | Region of the slide containing tissue | Confused with the tumour region | Binary mask |
| Multiple-instance learning (MIL) | Model that labels a bag from unlabelled instances | Described as weakly supervised segmentation | Bag and instance |
| Attention weights | Instance weights learned by a MIL model | Read as a segmentation | Unitless weights |
| Tumour proportion | Fraction of the slide that is tumour | Estimated visually without a method | Percent |
| Gleason score | Sum of the two highest-grade patterns | Used after the ISUP grade group replaced it | 6 to 10 |
| ISUP grade group | International Society of Urological Pathology grade groups 1 to 5, derived from Gleason | Called a Gleason score | Ordinal 1 to 5 |
| Nottingham grade | Breast carcinoma grade from tubules, nuclei, and mitoses | Pooled with other sites' grades | 1 to 3 |
| Mitotic figure | Cell in mitosis | Counted per high-power field without an area | Count per mm squared |
| HER2 | Human epidermal growth factor receptor 2 | Reported as positive or negative without the assay | IHC 0 to 3+; ISH group |
| IHC | Immunohistochemistry | Treated as a quantitative assay | Staining intensity and extent |
| ISH | In situ hybridisation | Reported without the probe set | Ratio and copy number |
| Equivocal | Result that requires a reflexive additional test | Treated as negative | Category |
| Synoptic report | Structured report of required elements | Confused with a narrative report | CAP protocol |
| Core element | Element that must be reported whether or not applicable | Treated as optional | CAP protocol |
| Conditional element | Element reported only when applicable | Omitted without a reason | CAP protocol |
| CAP protocol | College of American Pathologists cancer protocol | Assumed static | Versioned release |
| CLIA | Clinical Laboratory Improvement Amendments | Treated as a certification of accuracy | US laboratory regulation |
| Laboratory-developed test (LDT) | Test designed and validated in one laboratory | Assumed to be device-exempt | CLIA framework |
| In vitro diagnostic (IVD) | Test used on specimens from the body | Used as a synonym for any lab test | EU in vitro diagnostic regulation 2017/746 |
| Digital pathology | Primary diagnosis from a WSI | Used for any image analysis | Regulatory category |
| Scanner | Slide digitisation instrument | Treated as interchangeable across models | Make, model, firmware |
| Batch effect | Technical variation between processing groups | Treated as biological signal | Batch identifier |
| Slide-level label | Label assigned to the whole slide | Applied to every tile | Binary or ordinal |
| Quadratic weighted kappa (QWK) | Agreement statistic penalising distance between ordinal categories | Compared across category counts | Unitless, minus 1 to 1 |
| Cohen's kappa | Agreement statistic for nominal categories | Used for ordinal grades | Unitless |
| Wilson interval | Confidence interval for a proportion | Replaced by the normal approximation at small n | Proportion bounds |
| Turnaround time | Order to report | Measured without a start and end definition | Hours or days |

### 4.1 Categories that look like labels but are not

Three distinctions cause more rework than any others.

First, **a grade is a rule applied to a slide, not a property of the tumour**. The ISUP grade
group is derived from the two highest Gleason patterns present, and the rule for which
patterns count has changed with the grading convention. A grade of 3 in a report from 2015 and
a grade of 3 in a report from 2024 may not mean the same thing if the reporting convention
changed in between, and the grade cannot be reconstructed without the pattern detail.

Second, **"positive" is not a category for a biomarker with an equivocal band**. HER2 has five
ISH groups and four IHC categories, and groups 2 to 4 require a defined work-up. A pipeline
that maps everything below 3+ to negative collapses three distinct clinical situations into
one, and the collapse is invisible in the resulting table.

Third, **a slide is not a patient**. A patient can have a biopsy and a resection, a primary and
a metastasis, and a specimen with two blocks that disagree. The unit of analysis determines
the label, and the label determines the metric.

### 4.2 What counts as an event

| Endpoint | Event | Ascertainment source | Common defect |
|---|---|---|---|
| Cancer present | Malignant diagnosis on the slide | Pathologist read | Agreement with the read not measured |
| Grade group | Highest ISUP grade on any slide | Pathologist read | Biopsy and resection grades pooled |
| Positive margin | Tumour at the inked margin | Synoptic report | Ink vs tumour distance conflated |
| Biomarker status | Category from the assay and guideline | IHC and ISH report | Equivocal collapsed to negative |
| Molecular subtype | Result of the molecular assay | Assay report | Assay platform not recorded |
| Turnaround time | Order to sign-out | Laboratory information system | Start and end definitions vary |
| Discordance | Model and pathologist disagree | Adjudicated review | Disagreement treated as model error only |

> **Note:** "Cancer present" is not a property of a patient in the abstract. It is a property
> of a slide, read by a person, under a protocol. Store the reader and the protocol with the
> label, or the label cannot be audited.

## 5. The mechanics, step by step

This section is the procedure for turning specimens and slides into a defensible analysis.
Each of the ten steps states its input, output, rationale, branch points, and edge cases. A
team that executes them in order, and documents the choices, can defend the resulting cohort
to a pathologist, an accrediting body, and a buyer.

### Step 1 — Fix the decision and the unit of analysis

**Input.** A written question naming a decision, a decision-maker, an action, and the specimen
types in scope.

**Output.** A specification naming the unit of analysis, the label, the reference standard,
and the population.

**Rationale.** The unit of analysis determines everything downstream. A model that predicts
"cancer present in the patient" and a model that predicts "cancer present on this slide"
answer different questions, have different prevalences, and produce different metrics.

**Procedure.**

1. Write the decision with an actor and a verb: "the pathologist will use this overlay to
   decide whether to order a deeper section."
2. Name the unit: patient, specimen, block, slide, or tile. Write it in the protocol.
3. Name the label and its reference standard. "Tumour present" on a slide is a different
   label from "tumour present in the patient".
4. State the population and its prevalence at the unit you chose. Slide-level prevalence and
   patient-level prevalence differ whenever a patient contributes more than one slide.
5. State the action that follows a positive output, and whether the pathologist can override
   it without additional work.

**Branch point.** If the action is a triage decision (order this slide first), the label is
about urgency and the reference standard is the eventual diagnosis. If the action is a
detection aid, the label is the lesion and the reference standard is the pathologist's read.

**Edge case.** A patient with a negative biopsy and a positive resection. At patient level the
label is positive, at biopsy level it is negative, and a model trained on the pooled label
learns that biopsies are unreliable rather than that tumours are present.

### Step 2 — Build the specimen hierarchy

**Input.** Accession records, block logs, slide logs, and scanner output.

**Output.** A parent-linked hierarchy with one row per unit and an explicit key at each level.

**Rationale.** Pathology data nest more deeply than most specialties, and the hierarchy is
physical rather than organisational. Every join must respect the level, or a block-level
feature is attached to a patient-level label.

**Procedure.**

1. Establish the key chain: `patient_id`, `accession_id`, `block_id`, `slide_id`, `tile_id`.
2. Record the specimen type at the accession level: biopsy, resection, cytology, or
   consultation.
3. Record the block-level attributes: cassette identifier, tissue type, and number of levels.
4. Record the slide-level attributes: stain, section thickness, barcode, and scan identifier.
5. Verify that every slide resolves to exactly one block and one accession. Orphans are a
   data defect, not a modelling problem.

**Branch point.** If a case has multiple blocks with different grades, the case-level grade is
the highest grade by convention, not the average. Averaging is arithmetically possible and
clinically wrong.

**Edge case.** Recuts and deeper levels. A block can produce several slides at different
depths, and a lesion present at one depth may be absent at another. The block is the unit for
"is the lesion there", and the slide is the unit for "what does this section show".

### Step 3 — Qualify the pre-analytics

**Input.** Fixation records, processing logs, stain batch identifiers, section thickness, and
slide images.

**Output.** A pre-analytic covariate table with a batch identifier and a quality disposition
per slide.

**Rationale.** The slide carries the physical history of its preparation, and that history is
correlated with the diagnosis because laboratories batch work by day and by technologist. An
unrecorded batch becomes a feature the model can learn and a confounder the analyst cannot
adjust for.

**Procedure.**

1. Record ischaemia time and fixation time separately. They are different intervals and only
   one is usually available.
2. Record the stain batch or run identifier for every slide. If the laboratory information
   system does not expose it, the batch structure is unrecoverable.
3. Record section thickness. A four-micrometre and a five-micrometre section differ in nuclei
   per field.
4. Detect and flag folds, bubbles, pen marks, and foreign material.
5. Record the tissue area and the tumour proportion, with the method used to estimate them.

**Decision table for slide disposition.** Five outputs, because "use or discard" loses the
information needed to audit the discard rate.

| Observation | Action | Stored value | Consequence if mishandled |
|---|---|---|---|
| Tissue present, no artifact, stain batch recorded | Use | `usable` | None |
| Focal fold or bubble outside the lesion | Use with a mask over the artifact | `usable_artifact` | Artifact learned as morphology |
| Pen mark or ink overlapping the lesion | Use only with a mask; exclude from colour statistics | `ink_overlap` | Ink learned as a class |
| Tissue present but stain batch unknown | Use, and include a batch-unknown indicator | `batch_unknown` | Batch confounding invisible |
| No tissue, or tissue out of focus at the working magnification | Exclude and count the exclusion | `unusable` | Silent loss of difficult cases |

**Branch point.** If the batch identifier is missing for a subset, the analysis can proceed
only with a sensitivity analysis that treats the missing-batch group as its own level. It
cannot proceed as if the batch did not exist.

**Edge case.** A slide scanned twice. The two images are not independent measurements and the
second is not a replicate. Deduplicate on slide barcode plus scan date, and record which scan
was used.

### Step 4 — Scan and record the instrument

**Input.** Scanner make, model, firmware, objective, illumination, and colour profile.

**Output.** A scan record per image with the instrument fields populated and a colour
reference.

**Rationale.** The scanner is the instrument, and the model is a claim about the instrument as
much as about the tissue. A model validated on one scanner model is a different product on
another, and colour differences alone are enough to change a grade boundary call.

**Procedure.**

1. Record make, model, firmware version, and objective for every scan.
2. Record the microns per pixel at level 0, and derive it for every other level.
3. Record the colour profile and whether a calibration slide was scanned in the session.
4. Record compression. A lossy JPEG-compressed WSI has different high-frequency content from a
   lossless scan, and mitotic figure detection is sensitive to it.
5. Store the scanner fields with the image, not in a separate asset register that can drift.

**Branch point.** If a cohort spans two scanner generations, either validate separately or
include scanner as a stratification variable. Pooling without either is the most common cause
of an unexplained performance drop.

**Edge case.** A vendor format that does not carry the firmware. In that case the scan date
plus the laboratory's scanner inventory is the only reconstruction available, and it should be
labelled as inferred.

### Step 5 — Extract tiles at a fixed physical resolution

**Input.** Pyramidal images, a tissue mask, and a target magnification.

**Output.** A tile table with coordinates, physical resolution, and a tissue fraction per tile.

**Rationale.** Tiles are comparable only if they represent the same physical area. A 224-pixel
tile at 0.5 µm per pixel and a 224-pixel tile at 0.25 µm per pixel are different objects, and
mixing them teaches the model the scanner rather than the tissue.

**Procedure.**

1. Segment tissue from background and store the mask.
2. Choose a target microns per pixel and resample all slides to it.
3. Extract tiles with a fixed stride, and record the stride as a fraction of the tile size.
4. Discard tiles below a tissue-fraction threshold and record the threshold.
5. Store tile coordinates so that any tile can be mapped back to a region of the slide.
6. Record the number of tiles per slide. Tile count is itself a correlate of specimen size and
   therefore of stage.

**Branch point.** If tile count is associated with the label, a model that aggregates by mean
pooling is partly predicting specimen size. Use attention pooling and report the association
between tile count and label as a diagnostic.

**Edge case.** Small biopsies with fewer tiles than the batch size. Pad, or use a variable-size
formulation, and state which. Padding with background teaches the model that small specimens
are a class.

### Step 6 — Split by patient and audit leakage

**Input.** The tile table, the slide table, and the patient identifiers.

**Output.** Train, validation, and test splits with zero patient overlap, plus a leakage audit
report.

**Rationale.** Splitting by tile is the single most common and most damaging error in
computational pathology. Tiles from one slide share stain, scanner, and tissue, so a tile-level
split puts near-duplicates on both sides of the evaluation.

**Procedure.**

1. Split by patient, not by slide, when a patient can contribute more than one specimen.
2. Verify zero patient overlap by set intersection, and report the intersection size even when
   it is zero.
3. Verify zero slide overlap, and report the number of patients with multiple slides in each
   split.
4. Run a near-duplicate check: hash tiles, or compare downsampled slide thumbnails, and report
   the number of pairs above a similarity threshold.
5. Fit any stain normalisation on the training split only. Fitting on all slides leaks
   information across the split.
6. Deliberately run the invalid split as a diagnostic and report the inflation. The gap is the
   quantity that tells a reader how much of the published performance was the split.

**Branch point.** If the dataset provides an official split, check whether it is patient-level
before using it. A challenge split can still be contaminated by patients treated at two
contributing hospitals.

**Edge case.** A patient identifier that changes between institutions. The same patient
referred to a second hospital can appear under two identifiers with two primaries, and no
intersection test will find it. Linkage on date of birth plus sex plus specimen date is
imperfect and should be reported as imperfect.

### Step 7 — Establish the reference standard and measure agreement

**Input.** Pathologist reads, the grading scheme, and a sample for double reading.

**Output.** A label per unit, an agreement statistic, and a documented adjudication rule.

**Rationale.** The label is a human judgement and the disagreement between two humans is the
ceiling for any model. A model cannot be shown to exceed a reference standard whose own
reliability has not been measured.

**Procedure.**

1. Name the reference standard: a single pathologist, a consensus panel, or a registry
   abstract. Each has a different error structure.
2. Have a second pathologist read a random sample, at least 10 percent or 100 cases, under the
   same scheme and blinded to the first read.
3. Compute QWK for ordinal grades and Cohen's kappa for binary calls.
4. Report the agreement as the ceiling, and report the disagreement cases by category.
5. Write the adjudication rule for the cases that disagree, and record which rule was applied.

**Branch point.** If the reference is a consensus panel, the panel's own internal agreement
must be reported. A consensus that is achieved by deferring to the senior member is a single
read with extra steps.

**Edge case.** Grade disagreement is not random. It concentrates at the boundary between
adjacent grade groups and in small biopsies with limited tissue. Report disagreement by grade
and by specimen type, not as one kappa.

### Step 8 — Train, aggregate, and calibrate

**Input.** The tile table, the split, and the labels.

**Output.** A slide-level model with a documented aggregation rule and a calibration report.

**Rationale.** The aggregation rule is a modelling choice with clinical consequences. Mean
pooling dilutes a small tumour across many tiles; attention pooling can focus on it and can
also focus on artifact.

**Procedure.**

1. Choose the aggregation: mean pooling, max pooling, attention MIL, or a transformer over
   tiles. Record the choice and the reason.
2. Train with the label at the level you declared in Step 1. Do not train on slide labels and
   evaluate at patient level without an explicit aggregation.
3. Calibrate on a held-out calibration split, and report the calibration slope, intercept, and
   Brier score.
4. Report discrimination and calibration at the slide level and, if relevant, at the patient
   level.
5. Inspect attention maps for artifact focus, and report the proportion of attention mass on
   flagged tiles.

**Branch point.** If the model is intended as a detection aid, report lesion-level sensitivity
where lesions are annotated, and state the annotation completeness. If it is a slide-level
triage, report slide-level sensitivity at a fixed review budget.

**Edge case.** A slide with multiple lesions of different grades. The slide-level label is
usually the highest grade, and a model trained on that label will learn the highest-grade
region even when it is a small fraction of the slide.

### Step 9 — Apply the grading and biomarker rule under a named edition

**Input.** The model output, the scoring guideline, and the edition in force for the case.

**Output.** A category, the rule version, and the work-up path for equivocal results.

**Rationale.** A pathology category is a rule applied to a measurement, and the rule is
versioned. HER2 has four IHC categories and five ISH groups; a system that maps everything
below 3+ to negative removes the equivocal band and changes who gets treated.

**Procedure.**

1. Record the guideline and version for every case, keyed on the report date.
2. Store the components of the score, not only the final category. A stored category cannot be
   recomputed under a new rule.
3. Implement the equivocal band explicitly, with its reflex test and its own decision rule.
4. Record the assay platform, the probe set, and the control results.
5. If the cohort spans a guideline revision, stratify by version and report the difference.

**Decision table for HER2 under the ASCO/CAP 2018 focused update.** Five outputs, because the
equivocal groups have different work-up paths.

| Result | Category | Action | Output value |
|---|---|---|---|
| IHC 3+ | Positive | Eligible for HER2-directed therapy | `positive` |
| IHC 2+ | Equivocal | Reflex to dual-probe ISH | `equivocal_ihc` |
| ISH group 1, ratio at least 2.0 and copy number at least 4.0 | Positive | Eligible | `ish_positive` |
| ISH group 2, 3, or 4 | Equivocal | IHC review on the same block, recounted by a second observer | `ish_equivocal` |
| ISH group 5, ratio below 2.0 and copy number below 4.0 | Negative | Not eligible on this assay | `ish_negative` |

**Branch point.** If the case is a resection after neoadjuvant therapy, the biomarker may be
scored on residual tumour only, and the category applies to that residual. Pooling
post-treatment scores with diagnostic scores mixes two populations.

**Edge case.** Intratumoural heterogeneity. A clone that is amplified can occupy a minority of
the tumour, and the guideline's counting rule determines whether it is detected. Record the
region scored and the proportion of tumour it represents.

### Step 10 — Report synoptically and version the report

**Input.** The diagnosis, the grade, the stage elements, the biomarker results, and the
protocol version.

**Output.** A structured report extract with core, conditional, and optional elements
distinguished, plus a version record.

**Rationale.** The synoptic report is the data structure that oncology and registries consume.
A model or extractor that ignores its structure produces a table that cannot be joined to the
registry and cannot be audited against the protocol.

**Procedure.**

1. Extract each element as a question and response pair, not as free text.
2. Mark each element as core, conditional, or optional according to the protocol version.
3. Record "not applicable" and "not reported" as distinct values. They are different facts.
4. Record the protocol version and the date the laboratory adopted it. Laboratories have a
   defined window to adopt a new release.
5. Validate the extraction against a manually abstracted sample of at least 50 reports.

**Branch point.** If the protocol changed mid-cohort, the element set changed with it. Align
on the semantic element, not on the position in the report.

**Edge case.** A narrative addendum that contradicts the synoptic section. The addendum is
usually the later and more considered opinion, but it is not structured. Record both and flag
the conflict rather than resolving it silently.

## 6. Worked example

A laboratory has built a prostate biopsy grading model and reports a quadratic weighted kappa
of 0.96 against the reference uropathologist. The buyer's question is whether the model is
ready for a workflow in which it pre-orders cases for review.

The evaluation set is 200 biopsies from 200 patients, graded on the five-category ISUP grade
group scale: group 1 (GG1, Gleason 3+3), group 2 (GG2, 3+4), group 3 (GG3, 4+3), group 4
(GG4, Gleason 8), and group 5 (GG5, Gleason 9 or 10). The reference distribution is 70, 60,
30, 20, and 20 cases.

### 6.1 The patient-level confusion matrix

Rows are the reference read. Columns are the model output for the same case.

| Reference \ model | GG1 | GG2 | GG3 | GG4 | GG5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| GG1 | 53 | 14 | 2 | 1 | 0 | 70 |
| GG2 | 10 | 37 | 11 | 2 | 0 | 60 |
| GG3 | 1 | 9 | 13 | 6 | 1 | 30 |
| GG4 | 0 | 2 | 6 | 8 | 4 | 20 |
| GG5 | 0 | 0 | 2 | 5 | 13 | 20 |
| Total | 64 | 62 | 34 | 22 | 18 | 200 |

The diagonal is exact agreement: 53 + 37 + 13 + 8 + 13 = 124 of 200, or 62 percent. Exact
agreement alone is a poor summary for an ordinal scale, because a one-grade miss and a
four-grade miss are not the same error. Quadratic weighted kappa fixes that by penalising the
squared distance between categories.

### 6.2 The kappa arithmetic

With five categories the normalising denominator is (5 - 1) squared, which is 16. The weighted
observed disagreement is the sum of squared distances times the cell counts, divided by 16:

```text
Distance 1 cells: 14 + 10 + 11 + 9 + 6 + 6 + 4 + 5 = 65
Distance 2 cells:  2 +  2 +  1 + 1 + 2 + 2 = 10
Distance 3 cells:  1
Distance 4 cells:  0

sum of squared-distance counts = 1(65) + 4(10) + 9(1) + 16(0) = 114
weighted observed disagreement   = 114 / 16 = 7.125
```

The expected disagreement is computed from the margins, with expected count
row total times column total divided by 200, and the same squared-distance weighting:

```text
sum over cells of (i-j)^2 x row_i x col_j = 133,440
expected count-weighted total            = 133,440 / 200 = 667.2
weighted expected disagreement           = 667.2 / 16 = 41.7

QWK = 1 - 7.125 / 41.7 = 1 - 0.17086 = 0.8291
```

The reported 0.96 is not this number. The next two sections show where it came from.

### 6.3 Diagnostic one: the split

The reported evaluation split at the tile level, drawing tiles from all 200 slides into both
training and test sets. The same model, evaluated that way, produces the following matrix.

| Reference \ model | GG1 | GG2 | GG3 | GG4 | GG5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| GG1 | 66 | 4 | 0 | 0 | 0 | 70 |
| GG2 | 4 | 52 | 4 | 0 | 0 | 60 |
| GG3 | 1 | 3 | 24 | 2 | 0 | 30 |
| GG4 | 0 | 0 | 3 | 16 | 1 | 20 |
| GG5 | 0 | 0 | 0 | 2 | 18 | 20 |
| Total | 71 | 59 | 31 | 20 | 19 | 200 |

```text
sum of squared-distance counts = 1(23) + 4(1) + 9(0) + 16(0) = 27
weighted observed   = 27 / 16 = 1.6875
weighted expected   = 678.8 / 16 = 42.425
QWK = 1 - 1.6875 / 42.425 = 1 - 0.03978 = 0.9602
```

The gap is 0.9602 minus 0.8291, which is 0.1311. The model did not change. The split changed.
Tiles from the same slide share stain, scanner, focus, and tissue architecture, so a tile-level
split places near-duplicates on both sides and the test set stops being independent.

The diagnostic is to compute both numbers and report the gap. A gap above roughly 0.05 on an
ordinal scale means the split, not the model, is doing the work. Recomputing costs one training
run and no new data.

### 6.4 Diagnostic two: dichotomisation

The second diagnostic asks what happens when the ordinal scale is collapsed for clinical use.
Most prostate decisions turn on "clinically significant cancer", conventionally grade group 2
or higher. Collapse the model output at that boundary and compute the operating point.

```text
True positives  (reference GG2-5, model GG2-5):
  row GG2: 37 + 11 + 2 + 0 = 50
  row GG3:  9 + 13 + 6 + 1 = 29
  row GG4:  2 +  6 + 8 + 4 = 20
  row GG5:  0 +  2 + 5 + 13 = 20
  total = 119
False negatives (reference GG2-5, model GG1) = 10 + 1 + 0 + 0 = 11
False positives (reference GG1, model GG2-5) = 14 + 2 + 1 + 0 = 17
True negatives  (reference GG1, model GG1)   = 53

Sensitivity = 119 / 130 = 0.9154
Specificity =  53 /  70 = 0.7571
```

A team reporting "91.5 percent sensitivity for clinically significant cancer" has told the
truth and has hidden the consequence. Convert the operating point to the positive predictive
value (PPV) at the deployment prevalence, which for a screening biopsy population is about 4
percent:

```text
PPV = (0.04 x 0.9154) / (0.04 x 0.9154 + 0.96 x 0.2429)
    = 0.036616 / (0.036616 + 0.233184)
    = 0.036616 / 0.269800
    = 0.1357 -> 13.6 percent
```

A clinician who reads "91.5 percent sensitivity" and expects nine of ten flagged cases to be
cancer will find one in seven. The sensitivity is real and the expectation is wrong, because
specificity of 0.7571 multiplied by a large disease-free group produces seven false positives
for every true positive at this prevalence.

The diagnostic is to compute PPV at the prevalence of the population the model will serve,
before anyone quotes a sensitivity. It takes the counts already in the matrix.

### 6.5 Vary one assumption: the spectrum

The evaluation set had 35 percent grade group 1, which is a referral population. A screening
biopsy population has more benign-adjacent disease. Reweight the rows to 55, 25, 10, 6, and 4
percent, which gives 110, 50, 20, 12, and 8 cases.

| Reference \ model | GG1 | GG2 | GG3 | GG4 | GG5 | Total |
|---|---:|---:|---:|---:|---:|---:|
| GG1 | 83 | 22 | 3 | 2 | 0 | 110 |
| GG2 | 8 | 31 | 9 | 2 | 0 | 50 |
| GG3 | 1 | 6 | 9 | 4 | 0 | 20 |
| GG4 | 0 | 1 | 4 | 5 | 2 | 12 |
| GG5 | 0 | 0 | 1 | 2 | 5 | 8 |

Recomputing on this matrix gives a QWK of 0.7792, a change of minus 0.0499 from 0.8291. The
model is unchanged and the population is more typical of screening, so the honest expectation
for deployment is 0.78, not 0.83.

This is the varied assumption that changes the conclusion. At 0.83 the model is within 0.03 of
the published inter-pathologist ceiling for general pathologists, which supports a
second-read claim. At 0.78 it is at the ceiling rather than near it, which supports a triage
claim and not an autonomous one. The regulatory and commercial difference between those two
claims is larger than the metric difference.

### 6.6 The category-count effect

There is a third number a vendor can produce without changing the model at all: collapse the
five grade groups into three tiers (GG1, GG2-3, GG4-5) and recompute. The QWK falls to 0.7582,
a change of minus 0.0709.

That direction is counterintuitive, because collapsing categories is often assumed to increase
agreement. It falls here because the quadratic weights are normalised by (k - 1) squared, so
the penalty for a two-category miss grows from 4/16 to 1 when the scale shortens. A five-tier
kappa and a three-tier kappa are different quantities, and neither is wrong. The defect is
reporting one and comparing it to the other.

| Scale | Categories | QWK | Comparable to |
|---|---|---:|---|
| ISUP grade group | 5 | 0.8291 | Five-tier literature only |
| Collapsed tiers | 3 | 0.7582 | Nothing published at five tiers |
| Dichotomised at GG2 | 2 | 0.7571 specificity, 0.9154 sensitivity | Screening studies |

### 6.7 What the case does not establish

The case is a measurement example, not a validation of any product. It does not show that the
model improves a pathologist's accuracy, because the reference is a single read and the
comparison is offline. It does not establish transportability to another scanner, another
laboratory's stain protocol, or a biopsy set with a different grade distribution. It does not
establish that a 0.78 QWK is sufficient for a triage claim, because that depends on the review
budget and the harm of a missed high-grade case.

| Choice | Value used here | Alternative that changes the answer | How to disclose it |
|---|---|---|---|
| Split level | Patient | Tile | Report the leakage gap |
| Category count | Five | Three, or two | Report the scale with the kappa |
| Spectrum | 35 percent GG1 | 55 percent GG1 | Report the grade distribution |
| Reference standard | Single uropathologist | Consensus panel | Report inter-reader kappa |
| Aggregation | Attention MIL | Mean pooling | Report the rule and the tile counts |
| Scanner | One model | Two generations | Report per-scanner performance |
| Equivocal handling | Not modelled | Explicit equivocal class | Report the work-up path |

An analysis that discloses these seven choices can be reproduced and criticised. One that
discloses none produces a number that cannot be checked.

## 7. Code, query, or calculation

The block below is standard-library Python. It implements quadratic weighted kappa, the
three-category collapse that section 6.6 describes, the dichotomisation and prevalence
conversion from section 6.4, Wilson score intervals for slide-level sensitivity and
specificity, and the spectrum reweighting from section 6.5. It is runnable as written with
`python -I`.

**Assumptions.** The confusion matrices are hard-coded so that the arithmetic in section 6 can
be checked line by line. The kappa implementation assumes a square matrix with categories in
ascending order, which is what makes the squared distance meaningful. The Wilson interval
assumes a binomial proportion and independent slides, and it is reported instead of the normal
approximation because the counts are small enough for the difference to matter.

```python
"""Pathology concordance: quadratic weighted kappa and Wilson intervals."""


def qwk(obs):
    """Cohen's kappa with quadratic weights, for a square confusion matrix."""
    k = len(obs)
    n = sum(sum(row) for row in obs)
    rows = [sum(row) for row in obs]
    cols = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * obs[i][j]
            den += w * rows[i] * cols[j] / n
    return 1.0 - num / den


def collapse_to_three(m):
    """GG1 / GG2-3 / GG4-5. Adjacent grades are merged, not re-weighted."""
    idx = [0, 1, 1, 2, 2]
    out = [[0] * 3 for _ in range(3)]
    for i in range(5):
        for j in range(5):
            out[idx[i]][idx[j]] += m[i][j]
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return centre - half, centre + half


# Rows are the reference (uropathologist) ISUP grade group 1 to 5.
# Columns are the model output for the same case.
PATIENT = [
    [53, 14, 2, 1, 0],
    [10, 37, 11, 2, 0],
    [1, 9, 13, 6, 1],
    [0, 2, 6, 8, 4],
    [0, 0, 2, 5, 13],
]
TILE = [
    [66, 4, 0, 0, 0],
    [4, 52, 4, 0, 0],
    [1, 3, 24, 2, 0],
    [0, 0, 3, 16, 1],
    [0, 0, 0, 2, 18],
]

print("cases:", sum(sum(r) for r in PATIENT), sum(sum(r) for r in TILE))
for name, m in (("tile split   ", TILE), ("patient split", PATIENT)):
    print(f"{name} 5-class QWK = {qwk(m):.4f}   3-class QWK = {qwk(collapse_to_three(m)):.4f}")
print(f"leakage inflation in QWK = {qwk(TILE) - qwk(PATIENT):.4f}")
print(f"category effect on QWK    = {qwk(collapse_to_three(PATIENT)) - qwk(PATIENT):.4f}")

tp = sum(PATIENT[i][j] for i in range(1, 5) for j in range(1, 5))
fn = sum(PATIENT[i][0] for i in range(1, 5))
fp = sum(PATIENT[0][j] for j in range(1, 5))
tn = PATIENT[0][0]
sens, spec = tp / (tp + fn), tn / (tn + fp)
prev = 0.04
ppv = (prev * sens) / (prev * sens + (1 - prev) * (1 - spec))
print(f"GG2+ sensitivity {tp}/{tp + fn} = {sens:.4f}; specificity {tn}/{tn + fp} = {spec:.4f}")
print(f"PPV at 4% prevalence = {ppv:.4f}")

lo, hi = wilson(45, 48)
print(f"slide sensitivity 45/48 = {45 / 48:.4f}  95% CI {lo:.4f} to {hi:.4f}")
lo, hi = wilson(76, 82)
print(f"slide specificity 76/82 = {76 / 82:.4f}  95% CI {lo:.4f} to {hi:.4f}")

# Same model, screening spectrum: GG1 55%, GG2 25%, GG3 10%, GG4 6%, GG5 4%.
SCREENING = [
    [83, 22, 3, 2, 0],
    [8, 31, 9, 2, 0],
    [1, 6, 9, 4, 0],
    [0, 1, 4, 5, 2],
    [0, 0, 1, 2, 5],
]
print("screening rows:", [sum(r) for r in SCREENING])
print(f"screening-spectrum QWK = {qwk(SCREENING):.4f} "
      f"(change {qwk(SCREENING) - qwk(PATIENT):+.4f})")
```

Expected output:

```text
cases: 200 200
tile split    5-class QWK = 0.9602   3-class QWK = 0.9335
patient split 5-class QWK = 0.8291   3-class QWK = 0.7582
leakage inflation in QWK = 0.1311
category effect on QWK    = -0.0709
GG2+ sensitivity 119/130 = 0.9154; specificity 53/70 = 0.7571
PPV at 4% prevalence = 0.1357
slide sensitivity 45/48 = 0.9375  95% CI 0.8316 to 0.9785
slide specificity 76/82 = 0.9268  95% CI 0.8494 to 0.9660
screening rows: [110, 50, 20, 12, 8]
screening-spectrum QWK = 0.7792 (change -0.0499)
```

Three details matter more than the arithmetic. First, the three-category kappa is computed by
aggregating cells, not by re-weighting, so the collapse is exactly the one a laboratory would
report if it merged grade groups in its own reporting. Second, the prevalence conversion uses
the counts from the matrix rather than the rounded proportions, so the PPV reproduces to four
decimals. Third, the Wilson interval is reported because 45 of 48 is a small sample and the
normal approximation would give a lower bound that is too tight.

### 7.1 What the code does not do

It does not handle a missing category, which changes the weight denominator and therefore the
kappa. It does not implement a confidence interval for kappa, which requires a variance
estimator or a bootstrap. It does not implement multi-reader agreement statistics such as
Fleiss' kappa or Gwet's AC1, which are appropriate when more than two readers contribute. It
does not correct for the fact that the two matrices in the example are not independent samples
of the same population, which is the point of the comparison rather than a defect in the code.

## 8. Data, standards, and regulatory context

### 8.1 The WHO classification and its editions

The WHO Classification of Tumours defines entity boundaries and is released as a series of
organ-specific volumes rather than as a single dated document. The fifth edition ran from 2019
to 2026 and expanded to fourteen volumes, adding Paediatric Tumours and Genetic Tumour
Syndromes.

| Volume | Published | Volume | Published |
|---|---|---|---|
| Digestive System | 11 July 2019 | Urinary and Male Genital | 27 June 2022 |
| Breast | 21 November 2019 | Paediatric | 18 July 2023 |
| Soft Tissue and Bone | 8 May 2020 | Head and Neck | 20 March 2024 |
| Female Genital | 9 September 2020 | Haematolymphoid | 24 July 2024 |
| Thoracic | 15 April 2021 | Endocrine and Neuroendocrine | 20 March 2025 |
| Central Nervous System | 17 January 2022 | Skin | 26 August 2025 |
| Eye and Orbit | 23 January 2026 | Genetic Tumour Syndromes | 15 September 2026 |

The sixth edition began on 29 July 2026 with Volume 1, Digestive System Tumours, which
introduced revised diagnostic criteria and ICD-O-4 morphology codes. The overlap is the
practical problem: the fifth edition's final volume was published after the sixth edition's
first volume, so for a period a laboratory could be working to the sixth edition for
gastrointestinal cases and the fifth edition for everything else, on the same day.

What this constrains: the diagnosis field must carry the edition and the volume, not only the
year. What it does not constrain: the decision to analyse by molecular subgroup, which is a
design choice.

### 8.2 Grading and biomarker standards

ISUP grade groups replaced the Gleason score as the primary prostate grading output, and the
grade group is derived from the Gleason patterns rather than reported as their sum. A dataset
that stores "Gleason 7" and a dataset that stores "grade group 2 or 3" are not interchangeable,
because Gleason 7 spans two grade groups with different prognoses.

The ASCO/CAP HER2 guideline, focused update 2018 (Wolff et al., Journal of Clinical Oncology
2018;36:2105-2122), defines four IHC categories and five dual-probe ISH groups. The categories
are reproduced in the decision table in section 5, Step 9. Two properties matter for a data
team: the equivocal band is a required output rather than an error state, and the work-up for
ISH groups 2 to 4 depends on a concomitant IHC review on the same block, which means the
result is not a property of the ISH assay alone.

### 8.3 Reporting standards

The CAP cancer protocols define the set of elements a synoptic report must contain, and they
are released on a rolling schedule rather than an annual one. Recent releases have appeared in
December 2024, March, June, and September 2025, and April, June, and September 2026.

| Concept | Definition | Why it matters to a data team |
|---|---|---|
| Core element | Must be reported whether or not applicable | Absence is a protocol violation, not missing data |
| Conditional element | Reported only when applicable | Absence may be correct and is not missingness |
| Optional element | Not required for accreditation | Absence carries no information |
| Synoptic format | Element and response pairs in one location | The structure is the join key to registries |
| Compliance | Measured on the definitive primary resection | Biopsies are not held to the same threshold |
| Adoption window | Laboratories have a defined period to adopt a new release | Cohorts spanning the window mix protocol versions |

CAP publishes the electronic cancer protocols as HL7 Structured Data Capture XML, and the
protocol content is copyright. The CAP terms prohibit incorporating protocol content into
artificial-intelligence or machine-learning systems without written authorisation, which is a
licensing constraint that a data team must check before building an extractor from the
protocol text.

### 8.4 Imaging and laboratory standards

| Standard | Scope | Binding or convention | What it does not constrain |
|---|---|---|---|
| Digital Imaging and Communications in Medicine (DICOM) Supplement 145 | Whole-slide imaging | Convention adopted by scanner vendors | Image quality or stain consistency |
| Vendor pyramidal formats | Aperio SVS, Hamamatsu NDPI, generic TIFF | De facto | Metadata completeness |
| ICC colour profiles | Colour consistency | Convention | Whether the scan is diagnostic |
| CLIA | US laboratory operation | Binding regulation | Algorithm validity |
| CAP accreditation | Laboratory quality system | Binding for accredited laboratories | Model performance |
| HL7 FHIR Observation | Result messaging | Convention unless a purchaser requires it | Whether the value is correct |
| ICD-O-4 | Morphology coding from 2026 | Convention that registries adopt | Entity definition |

### 8.5 Regulatory context

Pathology image analysis is generally a medical device in the United States. The pathways are
510(k), De Novo, and premarket approval. A whole-slide imaging system used for primary
diagnosis is itself a device, which is why the scanner clearance, not the algorithm, was the
event that enabled digital readout. A test that is designed, validated, and performed within
one laboratory may be offered as a laboratory-developed test under CLIA, subject to the
enforcement position in force. In the European Union an in-vitro diagnostic falls under
Regulation (EU) 2017/746 (IVDR), and an AI component may also engage Regulation (EU) 2024/1689.

| Product type | Likely pathway | Evidence expected | Precedent |
|---|---|---|---|
| WSI scanner for primary diagnosis | De Novo or 510(k) | Concordance with glass read | Philips IntelliSite |
| Cancer detection aid on prostate biopsy | De Novo | Reader study with pathologist review | Paige Prostate |
| Autonomous diagnosis | PMA or De Novo with a narrow claim | Prospective, multi-site, adjudicated | None at scale |
| HER2 scoring aid | Device, possibly companion diagnostic | Concordance with the reference assay at thresholds | Assay-specific |

> **Note:** Regulatory status and the LDT enforcement position have changed more than once.
> Confirm the current position with counsel before a submission. Nothing in this section is
> legal advice.

## 9. Industry practice and case evidence

**CAMELYON16 (Ehteshami Bejnordi et al., Journal of the American Medical Association, 2017).**
Documented: about 400 whole-slide images of sentinel lymph nodes from two centres, with a
reader study comparing deep-learning systems against pathologists for metastasis detection.
The systems performed comparably to pathologists under the study's time constraints.
Inference: whole-slide analysis can meet a clinical reference standard on a curated task. The
task was metastasis detection in a challenge set, not full diagnostic workflow, and scanner
diversity was limited.

**PANDA (Bulten et al., Nature Medicine, 2022).** Documented: 10,616 prostate biopsies from
2,113 patients in the development set, 1,010 teams, and external validation on 741 US and 330
European biopsies. The mean algorithm achieved a quadratic weighted kappa of 0.862 on the US
external set and 0.868 on the European set. On a 70-case international comparison the
algorithms scored 0.876 against the reference standard while 13 general pathologists scored
0.765. The main error mode was calling benign biopsies grade group 1. Inference: a large,
multi-centre dataset with an ordinal endpoint can produce a model that matches expert
agreement, and the comparison against general pathologists is a claim about that population
rather than about pathologists in general.

**Histology-based MSI prediction (Kather et al., Nature Medicine, 2019).** Documented: a deep
model predicted microsatellite instability from routine H&E sections in gastrointestinal
cancer. Inference: morphology carries molecular information, and the signal is a surrogate.
Clinical use requires prospective validation against the molecular assay, and performance
varies by tumour type, so a single sensitivity figure across sites is not a deployable claim.

**Philips IntelliSite (FDA, 2017).** Documented: the first whole-slide imaging system
authorised for primary diagnostic use in the United States. Inference: the scanner, not the
algorithm, was the regulatory unlock, and digital readout alone does not validate any model
built on top of it. The commercial consequence is that scanner placement determines the
addressable market for every downstream algorithm.

**Paige Prostate (FDA De Novo DEN200080, 2021).** Documented: an AI tool authorised to help
identify prostate cancer in biopsy images, with a workflow role as a detection aid rather than
an autonomous diagnosis. Inference: a narrow, well-specified detection task can reach market,
and the authorisation does not establish an improvement in patient outcomes. Deployment
requires pathologist review, which makes the human factors of the review the operative
variable.

The pattern across the five cases is consistent. The documented facts are about concordance,
detection, and clearance. The inference that buyers make is about workflow and outcomes.
Between the two sit a validation population, a scanner fleet, and a review protocol.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Tile-level split | Test performance far above external validation | Near-duplicate tiles on both sides | Reported performance is a property of the split | Split by patient; report the gap |
| Slide-level split with repeat patients | Patients with many slides dominate the test set | Same patient in train and test | Inflated performance on prolific patients | Split by patient, verify by intersection |
| Scanner confounding | Performance drops on the second scanner | Colour and focus differ by instrument | Model learns the scanner | Record and stratify by scanner |
| Stain batch confounding | A batch indicator predicts the label | Laboratories batch by day and technologist | Confounded effect estimate | Record batch; sensitivity analysis |
| Fixation time unrecorded | No pre-analytic covariates available | Not captured in the laboratory system | Antigenicity differences look biological | Capture prospectively; flag unknown |
| Reference standard unmeasured | No second read, no kappa | Single pathologist treated as truth | Model judged against noise | Double-read a sample |
| Grade categories pooled | Gleason and grade group in one column | Two schemes stored as one | Cohorts of incomparable disease | Store both; map explicitly |
| Equivocal collapsed to negative | HER2 negative count exceeds the assay's expected rate | Work-up path not implemented | Patients denied therapy | Model the equivocal band |
| WHO edition mixed | Entity counts shift at a release date | Two classifications pooled | Denominators not comparable | Store edition and volume |
| Synoptic structure ignored | Element extraction fails on a new protocol release | Free-text parsing | Missing elements look like absent findings | Extract question-response pairs |
| Tile count correlates with label | Small specimens behave differently | Specimen size is a stage proxy | Model predicts size, not biology | Report the association |
| Attention on artifact | Heat maps cluster on pen and folds | Artifact is predictive in the training set | False confidence in explanations | Mask artifact; report attention mass |
| Batch normalisation fitted on all data | Test performance slightly too good | Statistics leak across the split | Optimistic estimate | Fit on training only |
| Multiple slides per case averaged | Case grade is intermediate and matches nothing | Aggregation by mean | Labels no pathologist would assign | Use the highest grade by rule |

Two red flags deserve emphasis because they are cheap to check.

The first is an external validation that matches the internal one too closely. Real scanner and
stain differences move performance, and an exact match usually means the external set shares a
laboratory, a scanner, or a patient population with the training set.

The second is a class distribution that matches the training set in a deployment population.
Grade distributions differ between referral and screening settings, and a test set that
reproduces the training distribution has probably been selected to do so.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Slides are independent given the patient | Splitting and resampling depend on it | Multiple slides per patient | Count slides per patient; split by patient |
| The reference read is reproducible | The label must mean one thing | Inter-reader disagreement at grade boundaries | Double-read a sample; compute kappa |
| The grade scheme is constant across the cohort | Pooling requires one definition | WHO edition and ISUP convention changes | Cross-tab year by scheme |
| Stain and scanner do not encode the label | The model should learn morphology | Batch and instrument confounded with case mix | Stratify by batch and scanner |
| Tile extraction preserves the lesion | The bag must contain the signal | Small lesions lost at the sampling stride | Compare tile-level and slide-level recall |
| The aggregation matches the clinical unit | The output must answer the question | Mean pooling over a focal lesion | Inspect attention; test max pooling |
| The assay is the assay that was validated | Biomarker categories are assay-specific | Platform or probe change | Record platform and controls |
| Missing elements are missing at random | Complete-case analysis requires it | Protocol version and specimen type drive reporting | Model the reporting process |
| The test population resembles deployment | Metrics are population-dependent | Referral vs screening spectrum | Reweight and recompute |
| Slide-level prevalence equals patient-level prevalence | PPV conversion uses one prevalence | Multiple slides per patient | Compute both |

The largest limit is not in the table. Pathology data are generated by a workflow that selects
which tissue is taken, which block is cut, and which slide is scanned. A lesion that was not
sampled has no slide, a block that was exhausted has no section, and a case sent for
consultation has two records. Each absence is a function of the procedure and the health
system, and a model trained on the resulting slides encodes the selection as if it were
biology.

A second limit is that the reference standard is a person. Every metric in this topic is
conditional on a read, and the read has a reliability that is measurable and usually
unmeasured. Reporting a model's kappa without the human kappa is reporting a numerator without
a denominator.

## 12. Exercises

### Exercise 1 — Draw the hierarchy and pick the unit (recall, 25 minutes)

**Deliverable.** A hierarchy diagram for a supplied case with two specimens, five blocks, and
eleven slides, plus the unit of analysis for three stated questions.

**Acceptance criteria.** Every slide resolves to one block; the unit is justified for each
question; at least one question requires a patient-level unit and one requires a slide-level
unit.

### Exercise 2 — Compute QWK by hand (application, 45 minutes)

**Deliverable.** A worked kappa calculation for a supplied 4x4 matrix, showing the weighted
observed sum, the weighted expected sum, and the final value.

**Acceptance criteria.** The weight denominator is stated; the expected counts are shown as
row times column divided by n; the result matches a programmatic computation to four decimals.

### Exercise 3 — Quantify leakage (critique, 50 minutes)

**Deliverable.** A tile-level split and a patient-level split of a supplied dataset, with both
QWK values, the gap, and a paragraph explaining the mechanism.

**Acceptance criteria.** Patient overlap is verified by intersection and reported; the gap is
stated in kappa units; the write-up distinguishes leakage from overfitting.

### Exercise 4 — Measure the ceiling (application, 40 minutes)

**Deliverable.** A double-read of at least 50 supplied cases, an agreement statistic, and a
revised interpretation of a model's reported performance given that ceiling.

**Acceptance criteria.** Agreement is reported with a confidence interval; disagreement is
broken down by grade; the revised interpretation states whether the model claim survives.

### Exercise 5 — Score HER2 cases (application, 35 minutes)

**Deliverable.** A table of ten supplied cases with IHC category, ISH group where applicable,
the work-up path for equivocal results, and the final category.

**Acceptance criteria.** Groups 2 to 4 are handled through the work-up rather than mapped to
negative; the guideline version is named; the assay platform is recorded.

### Exercise 6 — Version a diagnosis field (design, 40 minutes)

**Deliverable.** A schema for a diagnosis field that carries entity, edition, volume, ICD-O
code, and report date, with a migration note for cases diagnosed before the edition change.

**Acceptance criteria.** The schema distinguishes "not applicable" from "not reported"; it
handles the 2026 overlap between the fifth and sixth editions; it states how a query for a
reclassified entity behaves.

### Exercise 7 — Extract a synoptic report (design, 45 minutes)

**Deliverable.** An extractor specification for a supplied protocol, with core, conditional,
and optional elements labelled and the response vocabulary enumerated.

**Acceptance criteria.** Element-response pairs are extracted rather than sentences; protocol
version is recorded; validation against 50 manually abstracted reports is planned.

### Exercise 8 — Map the regulatory path (design, 30 minutes)

**Deliverable.** A one-page memo placing a described pathology product on the 510(k), De Novo,
PMA, or LDT route, naming the intended use, the evidence, and one precedent.

**Acceptance criteria.** The memo distinguishes scanner, detection aid, and autonomous
diagnosis; it states what the product does not claim; it names the accreditation constraint.

## 13. Self-check answers

**1. The hierarchy and the unit.** The chain is patient, specimen, block, slide, image, tile,
pixel. A question about whether the patient has cancer uses the patient unit and aggregates
across specimens. A question about whether this section shows tumour uses the slide unit. A
question about mitosis detection uses the tile unit at a stated magnification.

**2. QWK arithmetic for the five-category matrix.** The weight denominator is (5 - 1) squared,
which is 16. The squared-distance weighted observed sum is 1 x 65 plus 4 x 10 plus 9 x 1 plus
16 x 0, which is 114, and 114 divided by 16 is 7.125. The expected sum is 133,440 divided by
200 and then by 16, which is 41.7. Kappa is 1 minus 7.125 divided by 41.7, which is 0.8291.

**3. Leakage.** A tile-level split puts tiles from the same slide in both training and test
sets. Those tiles share stain, scanner, focus, and architecture, so the test set is not
independent and the model can memorise the slide. The measured inflation in the worked example
is 0.9602 minus 0.8291, which is 0.1311. Overfitting is a different phenomenon: it is a gap
between training and validation within a valid split.

**4. The ceiling.** Inter-pathologist agreement sets the ceiling. In the PANDA comparison the
reference pathologists agreed pairwise at about 0.91 to 0.93, and general pathologists scored
0.765 against the reference on a 70-case subset while the algorithms scored 0.876. A model
claim of "superhuman" is therefore a claim about a specific comparison population, not about
pathology in general.

**5. HER2 scoring.** IHC 3+ is positive. IHC 2+ is equivocal and reflexes to dual-probe ISH.
ISH group 1 is positive. Groups 2, 3, and 4 require concomitant IHC review on the same block
and a second-observer count before a category is assigned. Group 5 is negative. Collapsing the
equivocal groups to negative removes patients who would have been eligible.

**6. Edition change.** Store entity, edition, volume, and report date. A case diagnosed in 2021
under the fifth edition Thoracic volume is not the same entity set as a 2026 case under the
sixth edition Digestive volume. A query for a reclassified entity must state the edition, and
a pooled analysis must either restrict to one edition or map to a common collapsed category
and report the mapping.

**7. Synoptic extraction.** Extract question and response pairs, label each as core,
conditional, or optional, and keep "not applicable" separate from "not reported". Record the
protocol version and the adoption date. Validate against 50 manually abstracted reports and
report the element-level agreement.

**8. Regulatory mapping.** A scanner used for primary diagnosis is a device with a
concordance claim against glass. A detection aid is a device with a reader-study claim and a
pathologist in the loop. An autonomous diagnosis requires prospective, multi-site, adjudicated
evidence and has no precedent at scale. An LDT route depends on the current enforcement
position and on CLIA validation within the performing laboratory.

**Rubric for Exercises 6 and 8.** A strong answer names the version, the element set, and the
migration path, and states what the product does not claim. A weak answer lists fields without
versioning them, or cites a regulation without its scope.

## 14. Interview questions

**1. Why is a tile-level split invalid?**
Strong answer: tiles from one slide share stain, scanner, and architecture, so the test set is
not independent, and the size of the inflation can be measured by running both splits. Weak
answer: "it leaks", with no mechanism and no measurement.

**2. What is QWK and when is it the wrong metric?**
Strong answer: an ordinal agreement statistic with quadratic distance weights, appropriate for
grades and inappropriate for nominal categories or for comparing across different category
counts. Weak answer: "it is kappa for ordinal data", without the comparability caveat.

**3. Two pathologists disagree on 12 percent of cases. What does that mean for your model?**
Strong answer: the label has irreducible noise, the human kappa is the ceiling, and the model
should be reported against that ceiling. Weak answer: "the labels are noisy, so we need more
data", which does not address the ceiling.

**4. How would you detect scanner leakage?**
Strong answer: scanner metadata, image fingerprints, near-duplicate detection, external
validation on a second scanner, and a per-scanner performance report. Weak answer: "check the
metadata", which is missing precisely when a cohort is pooled.

**5. Your external QWK is 0.78 and internal is 0.93. What changed?**
Strong answer: the split level, the spectrum, the scanner, or the stain protocol, tested one at
a time. Weak answer: "overfitting", which is one hypothesis among four.

**6. Walk me through HER2 scoring.**
Strong answer: IHC categories with the 3+ and 2+ definitions, the reflex to ISH, the five ISH
groups, and the work-up for groups 2 to 4. Weak answer: "3+ is positive, everything else is
negative", which removes the equivocal band.

**7. What is the unit of analysis for a slide-level model?**
Strong answer: the slide, with patient-level aggregation defined explicitly and the label
declared at the same level as the prediction. Weak answer: "the patient", without saying how
multiple slides combine.

**8. How do you handle multiple slides per patient?**
Strong answer: split by patient, aggregate by a stated rule such as the highest grade, and
report the number of slides per patient in each split. Weak answer: average the grades, which
produces a value no pathologist assigned.

**9. What changed in the WHO classification in 2026?**
Strong answer: the sixth edition began with the Digestive volume on 29 July 2026 and introduced
ICD-O-4, while the fifth edition's final volume appeared in September 2026, so editions overlap
by organ. Weak answer: "a new edition came out", without the overlap.

**10. Can a pathology algorithm use a laboratory-developed test route?**
Strong answer: only if it is designed, validated, and performed in one laboratory under CLIA,
and the enforcement position must be confirmed. Weak answer: "yes, it is software in a lab",
which ignores the device question.

**11. What single field would you add to a pathology warehouse first?**
Strong answer: the stain batch or slide identifier, because without it the batch structure is
unrecoverable and no confounder can be adjusted. Weak answer: "more annotations", which does
not fix the provenance.

## 15. Key terms

- **WSI** — whole-slide image, a pyramidal digitisation of a glass slide.
- **MPP** — microns per pixel, the physical size of a pixel.
- **Tile** — a fixed-size crop extracted at a stated physical resolution.
- **Specimen, block, slide** — the three physical levels between a patient and an image.
- **Fixation time** — the interval during which tissue is in fixative.
- **Ischaemia time** — the interval from devascularisation to fixation.
- **Stain batch** — the set of slides processed together, and a latent confounder.
- **Tissue mask** — the region of a slide containing tissue.
- **MIL** — multiple-instance learning, labelling a bag from unlabelled instances.
- **Attention weights** — learned instance weights, an explanation rather than a segmentation.
- **ISUP grade group** — prostate grade groups 1 to 5 derived from Gleason patterns.
- **Gleason score** — the sum of the two highest-grade patterns, now secondary to grade group.
- **Nottingham grade** — breast carcinoma grade from tubules, nuclei, and mitoses.
- **Mitotic figure** — a cell in mitosis, counted per area rather than per field.
- **HER2** — human epidermal growth factor receptor 2.
- **IHC** — immunohistochemistry, scored by intensity and extent.
- **ISH** — in situ hybridisation, scored by ratio and copy number.
- **Equivocal** — a result requiring a defined reflex test, not a negative.
- **Synoptic report** — a structured report of required elements.
- **Core element** — a required element, reported whether or not applicable.
- **Conditional element** — an element required only in defined circumstances.
- **CAP protocol** — the College of American Pathologists cancer reporting protocol.
- **CLIA** — the US laboratory regulation framework.
- **LDT** — laboratory-developed test.
- **IVDR** — EU Regulation 2017/746 for in-vitro diagnostics.
- **QWK** — quadratic weighted kappa, an ordinal agreement statistic.
- **Wilson interval** — a score interval for a proportion, stable at small counts.
- **DICOM Supplement 145** — the whole-slide imaging storage standard.

## 16. Sources

- WHO Classification of Tumours Editorial Board. WHO Classification of Tumours, 5th edition, volumes 1 to 14, 2019 to 2026. https://whobluebooks.iarc.who.int/ (accessed 26 October 2026).
- IARC. Publication of the WHO Classification of Tumours, 6th edition, Volume 1: Digestive System Tumours, 29 July 2026. https://www.iarc.who.int/news-events/publication-of-the-who-classification-of-tumours-6th-edition-volume-1-digestive-system-tumours/ (accessed 26 October 2026).
- College of American Pathologists. Current cancer protocols. https://www.cap.org/protocols-and-guidelines/cancer-protocols/current-cancer-protocols/ (accessed 26 October 2026).
- College of American Pathologists. Cancer protocols frequently asked questions. https://www.cap.org/protocols-and-guidelines/cancer-protocols/cancer-protocols-frequently-asked-questions/ (accessed 26 October 2026).
- College of American Pathologists. CAP electronic cancer protocols. https://www.cap.org/protocols-and-guidelines/cancer-protocols/cap-electronic-cancer-protocols/ (accessed 26 October 2026).
- Wolff AC, et al. Human epidermal growth factor receptor 2 testing in breast cancer: ASCO/CAP clinical practice guideline focused update. J Clin Oncol. 2018;36:2105-2122. https://ascopubs.org/doi/full/10.1200/JCO.2018.77.8738 (accessed 26 October 2026).
- Ehteshami Bejnordi B, et al. Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer. JAMA. 2017. https://doi.org/10.1001/jama.2017.14585 (accessed 26 October 2026).
- Bulten W, et al. Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge. Nat Med. 2022;28:154-163. https://pmc.ncbi.nlm.nih.gov/articles/PMC8799467/ (accessed 26 October 2026).
- Kather JN, et al. Deep learning can predict microsatellite instability directly from histology in gastrointestinal cancer. Nat Med. 2019. https://doi.org/10.1038/s41591-019-0462-y (accessed 26 October 2026).
- US Food and Drug Administration. De Novo classification request DEN200080 (Paige Prostate). https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPMN/denovo.cfm?ID=DEN200080 (accessed 26 October 2026).
- DICOM Standards Committee. DICOM Standard, Supplement 145: whole slide microscopic image IOD and SOP classes. https://www.dicomstandard.org/ (accessed 26 October 2026).
- CAMELYON16 challenge. https://camelyon16.grand-challenge.org/ (accessed 26 October 2026).
- PANDA prostate cancer grade assessment challenge. https://www.kaggle.com/c/prostate-cancer-grade-assessment (accessed 26 October 2026).
- Centers for Medicare and Medicaid Services. Clinical Laboratory Improvement Amendments. https://www.cms.gov/medicare/quality/clinical-laboratory-improvement-amendments (accessed 26 October 2026).
- European Parliament and Council. Regulation (EU) 2017/746 on in vitro diagnostic medical devices. https://eur-lex.europa.eu/eli/reg/2017/746/oj (accessed 26 October 2026).

> **Access note:** All sources were checked on 26 October 2026. WHO volumes, CAP protocols,
> and ASCO/CAP guidelines are revised on independent schedules; confirm the version in force at
> the date of your analysis before citing a category.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/23_specialty_electives.md`](../../course/23_specialty_electives.md)
- Previous topic: [`02_cardiology.md`](02_cardiology.md)
- Related topic: [`01_oncology.md`](01_oncology.md)
- Cross-specialty comparison: [`07_cross-specialty-comparison.md`](07_cross-specialty-comparison.md)
- Labs and datasets: [`../../labs/`](../../labs/)
- Imaging foundations: [`../10_imaging/`](../10_imaging/)
- Omics and biomarkers: [`../12_omics/`](../12_omics/)
