# Pathology

> Module 23 — Specialty Electives
> Topic 03 of 07
> Estimated study time: 3.5 hours
> Prerequisites: 01, 08
> Version 2026.3

> **Note:** This file is educational. It is not medical advice, not a diagnostic authority,
> and not a substitute for the current WHO Classification of Tumours, College of American
> Pathologists protocol, or institutional laboratory procedure. Every threshold below names
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
| 5 | Apply the ASCO/CAP HER2 immunohistochemistry (IHC) categories and in situ hybridisation (ISH) groups, including the work-up for groups 2 to 4 | Exercise 5; interview question 6 |
| 6 | Explain how a WHO edition change reclassifies cases and how to version the diagnosis field | Exercise 6; self-check 6 |
| 7 | Write a synoptic report extractor that respects core, conditional, and optional elements | Exercise 7; interview question 9 |
| 8 | Place a pathology product on the FDA device path or the laboratory-developed test route, and state the accreditation constraints | Exercise 8; interview question 10 |

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
| Whole-slide image (WSI) | Digitised slide stored as a pyramidal image | Treated as a photograph with one resolution | DICOM Supplement 145; vendor formats |
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
| ISUP grade group | Grade groups 1 to 5 derived from Gleason | Called a Gleason score | Ordinal 1 to 5 |
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
| In vitro diagnostic (IVD) | Test used on specimens from the body | Used as a synonym for any lab test | EU IVDR 2017/746 |
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

