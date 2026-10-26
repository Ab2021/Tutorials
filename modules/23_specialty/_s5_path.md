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

