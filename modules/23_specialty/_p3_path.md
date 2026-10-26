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
| DICOM Supplement 145 | Whole-slide imaging | Convention adopted by scanner vendors | Image quality or stain consistency |
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
