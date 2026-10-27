# Spatial Omics and Pathology Integration

> Module 12 — Omics and Precision Medicine
> Topic 13 of 16
> Estimated study time: 4 hours
> Prerequisites: topics 03, 04, and 05
> Version 2026.3

## 1. Why this topic exists

Pathology and molecular biology have spent a century looking at the same tissue through
incompatible instruments. A hematoxylin and eosin (H&E) slide shows architecture, cell morphology,
and the relationship between a tumour and its stroma at sub-micron resolution, but measures no
transcripts. A bulk sequencing assay measures thousands of transcripts but dissolves the
architecture. Single-cell RNA sequencing (scRNA-seq) resolves cell types but requires dissociation,
which destroys the spatial relationships that determine whether a T cell can reach a tumour cell.

Spatial omics closes the gap by measuring transcripts or proteins while retaining coordinates. The
commercial instruments split into two families. Sequencing-based capture, such as 10x Genomics
Visium, places tissue on a grid of barcoded spots and sequences the captured messenger RNA
(mRNA). Imaging-based methods, such as 10x Xenium, NanoString CosMx, and Vizgen MERSCOPE, detect
individual transcripts in situ with fluorescence and localize them to sub-cellular coordinates.
The two families trade resolution against panel size, and the trade-off determines which questions
each can answer.

The industry pull is direct. Pathology is the diagnostic backbone of oncology, and every
immunotherapy decision depends on a spatial question: are the cytotoxic T cells inside the tumour
or trapped in the stroma? A bulk biomarker cannot answer it, and a pathologist reading an H&E
slide can see the pattern but cannot quantify the molecular state. A spatial assay can do both,
in principle.

The technical reality is harder than the marketing.

- A 55 micrometre Visium spot contains several cells, so the measurement is a mixture. Turning the
  mixture into cell-type proportions requires deconvolution, and deconvolution depends on a
  single-cell reference that may not match the tissue.
- An imaging panel measures hundreds to a few thousand genes, not the whole transcriptome, so the
  panel design determines what can be discovered.
- Registration between the histology image and the capture grid is a geometric transformation
  estimated from landmarks, and a small error moves a signal into the wrong tissue compartment.
- The number of independent samples is the number of patients, not the number of spots. A slide
  with 5,000 spots from one patient has an effective sample size of one for any patient-level
  claim.

The most consequential error is interpretive. A deconvolution method will always return
proportions that sum to one, including when the reference is wrong and the tissue contains a cell
type the reference does not include. The output looks like data. The check that reveals the error
is a residual at a marker the model did not use.

### 1.1 The failure pattern

1. A tissue section is registered to the capture grid with a rigid transform and no landmark
   check.
2. A single-cell reference from a different platform is used without adjusting for nuclear versus
   cytoplasmic transcript capture.
3. A deconvolution method returns proportions for every spot, and the proportions are reported
   without a residual diagnostic.
4. A marker gene is interpreted as a cell-type abundance without checking that the marker is
   specific in this tissue.
5. Spots are treated as independent observations in a statistical test.
6. A spatial pattern is reported from a single patient and described as a tumour subtype.
7. The result fails to replicate, and the failure is attributed to cohort differences.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Pathologist | A molecular claim that contradicts the morphology | Did not design the deconvolution |
| Translational lead | A spatial biomarker that does not replicate | Depends on patient-level design |
| Computational biologist | Weeks on a registration error | Inherited the pipeline defaults |
| Oncologist | A treatment hypothesis with no validated assay | Cannot inspect the proportions |
| Program | A slide-level result presented as a cohort finding | No independent sample count |

### 1.3 The industry question

> Is the signal in this spot coming from the cells the model says are there, and does the pattern
> hold across patients rather than across spots?

The question has two parts, and both must be answered. The first is a measurement question:
deconvolution and registration determine whether the spot-level signal is attributed correctly.
The second is an inference question: spots are not replicates, and a pattern within one tumour is
not evidence of a subtype across tumours.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compare** sequencing-based and imaging-based spatial platforms on resolution, panel size,
   throughput, and the questions each can answer.
2. **Estimate** the number of cells in a capture spot from spot geometry and tissue cellularity,
   and state why the estimate is an upper bound.
3. **Perform** an affine registration between spot coordinates and image pixels from landmarks,
   and **report** the residual in micrometres.
4. **Solve** a two-compartment deconvolution by hand and **interpret** the residual at a marker
   excluded from the fit.
5. **Compute** Moran's I for a spatial variable and **interpret** it against its null expectation.
6. **Evaluate** deconvolution methods against benchmark evidence, including the non-negative least
   squares baseline.
7. **Diagnose** reference mismatch, registration error, and pseudoreplication from diagnostics.
8. **Design** a spatial study whose unit of inference is the patient, not the spot.

## 3. The core mental model

A spatial assay produces three linked layers: an image, a coordinate grid, and a molecular matrix.
Every analysis depends on the alignment between them, and every biological claim depends on
turning the molecular matrix into cell-level quantities.

```text
  TISSUE SECTION
      |
      +--> H&E or immunofluorescence image (pixels, microns)
      |
      +--> capture grid or in-situ detection (spots or transcripts)
      |
      v
  REGISTRATION: pixels <-> grid coordinates
      |
      +--> affine or non-rigid transform
      +--> landmark residual in micrometres
      |
      v
  MOLECULAR MATRIX: spot x gene counts
      |
      +--> sequencing capture: multi-cell spots, whole transcriptome
      +--> imaging: single transcripts, targeted panel
      |
      v
  DECONVOLUTION or SEGMENTATION
      |
      +--> proportions per spot (capture-based)
      +--> cell-level assignment (imaging-based)
      |
      v
  SPATIAL STATISTICS
      |
      +--> autocorrelation, neighbourhood enrichment, niche detection
      |
      v
  INFERENCE: patient-level, not spot-level
```

### 3.1 Resolution determines the model

The first decision is whether the measurement unit is a cell or a mixture.

| Platform family | Measurement unit | Typical content | Appropriate model |
|---|---|---|---|
| Visium, original | 55 micrometre spot | Several cells | Deconvolution to proportions |
| Visium HD | 2, 8, or 16 micrometre bin | Sub-cell to a few cells | Segmentation or small-bin models |
| Xenium, CosMx, MERSCOPE | Single transcript with coordinates | One cell after segmentation | Cell-level counts |
| Spatial proteomics | Single cell or small region | Tens of proteins | Cell-level phenotyping |

A model built for mixtures applied to single-cell data is over-parameterized. A model built for
single cells applied to 55 micrometre spots is wrong, because a spot's count is a sum over cells
and the variance structure differs.

### 3.2 Deconvolution is a regression with constraints

For a spot \(s\) and gene \(g\):

```text
  y(s, g) = sum over k of w(s, k) * beta(k, g) + error
```

where \(w(s,k)\) is the abundance of cell type \(k\) in spot \(s\), and \(\beta(k,g)\) is the
reference expression of gene \(g\) in cell type \(k\). The weights are constrained to be
non-negative. With two cell types and two genes, the system can be solved by hand, which is why
the worked example uses that case.

Three properties follow from the formulation:

1. **The reference defines the answer.** If the reference lacks a cell type present in the tissue,
   the model attributes its signal to the types it knows.
2. **Collinearity is fatal.** If two reference profiles are similar, many weight combinations fit
   equally well, and the split between them is arbitrary.
3. **Residuals are the diagnostic.** A gene excluded from the fit but measured in the data
   provides an independent check. A large residual means the model is missing something.

### 3.3 Registration error propagates into biology

The affine transform maps spot coordinates to pixel coordinates. Its residual, measured at
landmarks not used in the fit, is the registration error. A residual of 1.4 micrometres is small
relative to a 55 micrometre spot. A residual of 30 micrometres shifts a signal by half a spot, and
a tumour-stroma boundary becomes blurred.

The biological consequence is specific. If a checkpoint transcript is assigned to the adjacent
stroma rather than to the infiltrating T cell, the analysis concludes that the tumour does not
express the checkpoint, and an immunotherapy hypothesis is built on a registration error.

### 3.4 Spatial autocorrelation is not independent replication

Moran's I measures whether nearby values are more similar than expected under randomness:

```text
  I = (n / W) * (sum_ij w_ij (x_i - mean)(x_j - mean)) / sum_i (x_i - mean)^2
```

where \(w_{ij}\) is 1 for neighbouring units and \(W\) is the sum of weights. Under the null of
spatial randomness, the expected value is approximately \(-1/(n-1)\).

A significant positive Moran's I says the variable clusters in space. It does not say the
clustering is biological rather than an artifact of tissue architecture, and it does not make the
spots independent. A study with 5,000 spots and one patient still has one patient.

### 3.5 Two scales of inference

| Scale | Unit | Valid claim | Invalid claim |
|---|---|---|---|
| Within a section | Spot or cell | This niche exists in this tumour | This niche defines a subtype |
| Across sections | Patient | The niche is more common in responders | The niche causes response |

The move from the first row to the second requires multiple patients, and the number of patients
is the number that enters the statistical test.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Spatial transcriptomics | Measurement of RNA with retained spatial coordinates | Used for any spatially resolved assay, including protein | Assay family |
| Spatial omics | Umbrella term for spatially resolved molecular measurement | Treated as a single technology | Assay family |
| Capture spot | Barcoded region on a sequencing capture array | Called a cell | Micrometres |
| Bin | Aggregated set of capture squares at a chosen resolution | Confused with a cell | Micrometres |
| Visium | Sequencing-based capture platform with 55 micrometre spots in the original format | Used generically for all 10x spatial products | Platform |
| Visium HD | Sequencing platform with 2 micrometre barcoded squares, binned at 2, 8, or 16 micrometres | Assumed to be single-cell without segmentation | Platform |
| Xenium | Imaging-based in-situ hybridization platform | Assumed to be whole transcriptome | Platform |
| CosMx | Imaging-based spatial molecular imager | Assumed to be sequencing-based | Platform |
| MERSCOPE | Imaging-based spatial platform | Assumed to have the same panel as Xenium | Platform |
| Region of interest | Anatomical area selected for profiling | Confused with the whole section | Square millimetres |
| Registration | Geometric alignment between image and capture coordinates | Assumed exact | Transform plus residual |
| Affine transform | Linear transform with translation, rotation, scale, and shear | Sufficient for all tissue deformations | Six parameters in two dimensions |
| Non-rigid registration | Warping transform that allows local deformation | Assumed necessary; often it is not | Deformation field |
| Landmark | Corresponding point in image and array coordinates | Chosen without verification | Coordinate pair |
| Deconvolution | Estimation of cell-type proportions from mixed expression | Confused with cell segmentation | Proportion per spot |
| Segmentation | Assignment of transcripts to cell boundaries | Confused with deconvolution | Cell identifier |
| Reference profile | Per-cell-type expression signature from single-cell data | Assumed platform-independent | Counts per cell |
| Non-negative least squares (NNLS) | Least squares with weights constrained to be non-negative | Treated as obsolete | Solver |
| RCTD | Robust cell type decomposition method | Assumed to handle missing cell types | Software |
| Cell2location | Bayesian deconvolution method | Assumed to require matched platform | Software |
| CARD | Spatial deconvolution using reference and spatial correlation | Assumed to be a segmentation method | Software |
| Tangram | Mapping method that aligns single-cell and spatial data | Used for proportions without checking calibration | Software |
| Moran's I | Spatial autocorrelation statistic | Interpreted as a p-value alone | Dimensionless, roughly -1 to 1 |
| Neighbourhood enrichment | Test of whether cell types co-localize more than expected | Confused with causal interaction | Score or p-value |
| Niche | Recurring spatial combination of cell types | Treated as a cell type | Spatial unit |
| Pseudoreplication | Treating non-independent units as replicates | Common with spots and cells | Study design error |
| Tumour microenvironment | Cellular and extracellular context of a tumour | Used as a catch-all | Tissue compartment |
| Immune excluded | T cells present but confined to stroma | Confused with immune desert | Spatial phenotype |
| Immune desert | Absence of cytotoxic T cells | Confused with immune excluded | Spatial phenotype |
| Formalin-fixed paraffin-embedded (FFPE) | Preservation method for archival tissue | Assumed incompatible with spatial assays | Specimen type |
| Whole-slide image | Digitized histology slide | Confused with a region of interest | Image |
| SpatialData | Open data framework for spatial omics | Assumed to be an analysis method | Software |

## 5. The mechanics, step by step

The procedure runs from a tissue section to a patient-level claim. It assumes the assay has been
run and the raw counts and images exist.

| Step | Question answered | Artifact | Failure if skipped |
|---:|---|---|---|
| 1 | What is the biological question and the unit of inference? | Analysis plan | Spot-level claims about patients |
| 2 | Which platform and resolution? | Assay choice | Model mismatch |
| 3 | How many cells are in a spot? | Cell-count estimate | Proportions interpreted as counts |
| 4 | How are image and grid aligned? | Transform and residual | Signal in the wrong compartment |
| 5 | What preprocessing is needed? | Normalized counts | Technical variation read as biology |
| 6 | Which reference is appropriate? | Reference matrix | Missing cell type absorbed |
| 7 | Which deconvolution method? | Proportion estimates | Method-specific bias |
| 8 | Do the residuals support the model? | Residual diagnostics | Silent model misspecification |
| 9 | Are the proportions spatially structured? | Moran's I and niche maps | Random pattern read as biology |
| 10 | Is the pattern present across patients? | Patient-level test | Pseudoreplication |
| 11 | Does it validate against histology or an orthogonal assay? | Concordance | A molecular artifact |
| 12 | How is it reported? | Report with limits | Overclaimed subtype |

### Step 1. Define the question and the unit of inference

**Input.** A biological or clinical question.

**Output.** A written statement of the unit of inference: cell, spot, region, section, or patient.

**Rationale.** The unit of inference determines the statistical model. A question about whether a
niche exists in a tumour is answered at the section level. A question about whether the niche
predicts response is answered at the patient level. The same dataset can answer the first and not
the second.

The wording of the question determines which unit applies.

| Question wording | Unit of inference | Minimum design |
|---|---|---|
| Does this tumour contain a niche? | Section | One or more sections, descriptive |
| Is the niche associated with a genotype? | Section within patient | Paired sections, patient-level test |
| Does the niche differ between responders and non-responders? | Patient | Multiple patients per group |
| Does the niche predict response? | Patient, with time-to-event | Cohort with follow-up and a pre-specified model |
| Does a cell-cell interaction cause the phenotype? | Cell or niche, with perturbation | Perturbation experiment, not an observational slide |

**If skipped.** The analysis reports a p-value computed over spots and describes it as evidence
about patients.

### Step 2. Choose the platform and resolution

**Input.** Tissue type, preservation, panel requirements, budget, and the question's resolution
requirement.

**Output.** A platform and a binning or segmentation strategy.

**Rationale.** The platform determines what can be measured and at what resolution. Whole
transcriptome capture allows discovery but mixes cells. Targeted imaging resolves cells but only
for the genes on the panel.

The choice has more than two outcomes.

| Requirement | Sequencing capture | Imaging panel |
|---|---|---|
| Discovery of unknown cell states | Whole transcriptome | Not possible outside the panel |
| Single-cell resolution | Requires small bins and segmentation | Native |
| Archival FFPE tissue | Supported | Supported |
| Large section area at low cost | Supported | Cost scales with area |
| Protein co-detection | Limited | Supported on some platforms |
| Sub-cellular localization | Not native | Native |

**If skipped.** A panel that does not contain the marker of interest is used, and the analysis
concludes that the marker is absent.

### Step 3. Estimate the number of cells per spot

**Input.** Spot diameter, tissue cellularity, and typical cell diameter.

**Output.** An estimated cell count per spot, with an explicit upper bound.

**Rationale.** The geometric bound is the spot area divided by the cell cross-section:

```text
  spot area = pi * (27.5 micrometres)^2 = 2,375.8 square micrometres
  cell area = pi * (5 micrometres)^2 = 78.5 square micrometres
  bound = 2,375.8 / 78.5 = 30.3 cells
```

The bound assumes perfect packing. Tissue contains extracellular matrix, vessels, and empty
space, so the realised count is lower, commonly a few cells to about ten. The estimate matters
because a proportion computed from a spot with three cells has a resolution of about 33
percentage points per cell.

**If skipped.** A 10% difference in estimated proportions is interpreted as biology when it is one
cell.

### Step 4. Register the image to the capture grid

**Input.** Spot coordinates in millimetres; landmark pixel coordinates; pixel size.

**Output.** An affine transform, a residual at held-out landmarks in micrometres, and a decision to
accept or re-register.

**Rationale.** Registration is a six-parameter fit in two dimensions. Three landmarks determine it
exactly, and a fourth landmark provides an independent residual. The residual must be compared
with the spot diameter, not with the image resolution.

| Residual relative to spot | Interpretation | Action |
|---|---|---|
| Below 5% of spot diameter | Excellent | Accept |
| 5% to 20% | Acceptable for spot-level analysis | Accept, report |
| 20% to 50% | Boundary signals will blur | Re-register or exclude boundary spots |
| Above 50% | Alignment is wrong | Reject and redo |
| Spatially structured residual | Non-rigid deformation | Use a non-rigid method |

**If skipped.** A rigid transform is applied to a section that stretched during mounting, and the
error is largest where the tissue deformed.

### Step 5. Preprocess counts

**Input.** Raw spot-by-gene counts; tissue image.

**Output.** Normalized, filtered counts with spot-level quality metrics.

**Rationale.** Capture efficiency varies across a slide. Spots under tissue have higher counts
than spots off tissue, and spots at the edge have fewer cells. Normalization must account for
total counts per spot, and spots with very low counts should be excluded rather than normalized
into noise.

Quality control has more than two outcomes per spot.

| Observation | Interpretation | Action |
|---|---|---|
| Counts above the 5th percentile, under tissue | Normal | Keep |
| Counts far below the tissue median | Few cells or poor capture | Exclude, record the reason |
| Counts far above the median | Dense cellular region or a bubble artifact | Inspect the image; keep if under tissue |
| High mitochondrial fraction | Damaged or necrotic tissue | Exclude or analyze as a separate stratum |
| Spot outside the tissue boundary | Off-tissue capture | Exclude |
| Spot at the boundary with intermediate counts | Partial tissue coverage | Keep only if the boundary is not the question |

**If skipped.** A low-count spot at the tissue edge is normalized to look like a high-count spot,
and its noisy profile enters the deconvolution as if it were reliable.

### Step 6. Choose and adapt the reference

**Input.** Single-cell or single-nucleus reference; tissue type; platform.

**Output.** A reference matrix whose genes overlap the spatial panel and whose capture chemistry
matches the spatial assay.

**Rationale.** The reference is the most consequential choice in deconvolution. Three mismatches
matter: platform (nuclei versus whole cells changes the ratio of nuclear to cytoplasmic
transcripts), tissue (a cell line reference lacks the expression states of primary tissue), and
gene set (the intersection of reference and panel genes may be small).

**If skipped.** A reference without fibroblasts is used on a desmoplastic tumour, and the
fibroblast signal is attributed to tumour cells.

### Step 7. Run deconvolution with a stated method and constraints

**Input.** Spatial counts; reference matrix; method and parameters.

**Output.** Proportions per spot, plus a convergence or uncertainty estimate.

**Rationale.** Methods differ in their priors and their assumptions about missing cell types.
RCTD models platform effects and is robust to some reference mismatch. Cell2location uses a
Bayesian hierarchy and reports uncertainty. CARD incorporates spatial correlation. Tangram maps
cells and can produce proportions but tends to flatten variation.

The choice has more than two outcomes.

| Situation | Recommended method family | Reason |
|---|---|---|
| Reference and spatial data from different platforms | RCTD or cell2location | Both model platform effects |
| Need uncertainty intervals | Cell2location | Posterior distribution per spot |
| Strong spatial correlation expected | CARD | Uses spatial information in the prior |
| Very large spot count, limited compute | NNLS baseline or Tangram | Fast and often competitive |
| Rare cell type of interest | Cell2location with a informed prior | Better detection of low-abundance types |
| Panel with few genes | NNLS with a reduced reference | Fewer parameters, more stable |

**If skipped.** A method whose assumptions are violated returns proportions with no warning.

### Step 8. Check residuals on held-out markers

**Input.** Fitted proportions; observed counts; a set of marker genes excluded from the fit.

**Output.** Residual per held-out marker, in counts and in cell equivalents.

**Rationale.** A model that fits its own genes perfectly can still be wrong. Held-out markers
provide an independent test. A positive residual at a T-cell marker means the model has no
compartment for T cells, and the T-cell signal has been absorbed by another compartment.

**If skipped.** The proportions look precise, and the missing compartment is invisible.

### Step 9. Test spatial structure

**Input.** A per-spot or per-cell variable; a spatial weight definition.

**Output.** Moran's I, a permutation p-value, and a map.

**Rationale.** Moran's I distinguishes a structured pattern from a random one. It does not
distinguish biology from tissue architecture, so it must be interpreted with the histology.

The result depends on the weight definition, and the choice must be stated.

| Weight definition | Neighbours | Use when |
|---|---|---|
| Queen contiguity | All eight surrounding spots | A square grid with complete coverage |
| Rook contiguity | Four edge-sharing spots | A grid where diagonal contact is not physical |
| Distance band | All spots within a radius | Irregular spot layouts |
| K nearest neighbours | The k closest spots | Varying spot density |
| Hexagonal | Six neighbours | Hexagonal capture arrays |
| Visium default | Six neighbours in the hex layout | Standard Visium capture |

**If skipped.** A pattern that arises from the tissue boundary is reported as a molecular
gradient.

### Step 10. Aggregate to the patient and test across patients

**Input.** Per-patient summaries of the spatial statistic.

**Output.** A patient-level test with the number of patients as the sample size.

**Rationale.** The spot is not the experimental unit. A study with 20 patients and 100,000 spots
has 20 independent observations for a patient-level claim.

**If skipped.** A p-value computed over spots is reported as evidence of a patient-level
association.

### Step 11. Validate against histology or an orthogonal assay

**Input.** Spatial result; pathologist annotation; immunohistochemistry; an independent assay.

**Output.** A concordance estimate and a list of discordant regions.

**Rationale.** Histology is the ground truth for tissue architecture. If the deconvolution says a
region is 80% tumour and the pathologist calls it necrosis, one of them is wrong, and the
disagreement is the most informative part of the analysis.

| Comparator | What it validates | Limitation |
|---|---|---|
| Pathologist annotation of the same region | Compartment boundaries and major cell types | Observer variability; no molecular resolution |
| Immunohistochemistry for one marker | A single cell-type call | One marker at a time; semiquantitative |
| An independent bulk assay of microdissected regions | Aggregate composition | Loses single-spot resolution |
| A second spatial platform on an adjacent section | Platform-specific artifacts | Section-to-section variation |
| A synthetic mixture with known proportions | The deconvolution solver | Does not test the reference |

**If skipped.** A molecular artifact that contradicts the morphology is reported as a discovery.

### Step 12. Report with the limits attached

**Input.** Results, diagnostics, and validation.

**Output.** A report stating the platform, resolution, method, reference, residual diagnostics,
patient count, and validation.

**Rationale.** A spatial result without its resolution and reference is not reproducible. The
reader cannot tell whether a 10% difference is one cell or a thousand.

A report should also state the number of spots excluded at each quality step, because the
excluded fraction is often related to the tissue region and therefore to the biology. A tumour
with extensive necrosis will lose more spots than a cellular tumour, and a comparison that
ignores the exclusion rate confounds necrosis with composition.

**If skipped.** The result cannot be reproduced, and the failure is attributed to biology.

### 5.13 Benchmarks and what they mean

Four independent benchmarks converge on a small set of methods.

| Benchmark | Scope | Main finding |
|---|---|---|
| Li et al., Nature Methods, 2022 | 16 methods, 45 paired datasets and 32 simulated | Cell2location, SpatialDWLS, and RCTD best for deconvolution; Tangram, gimVI, SpaGE best for expression prediction |
| Li et al., Nature Communications, 2023 | 18 methods, 50 datasets | CARD, Cell2location, and Tangram among the top performers; performance depends on the scenario |
| Yan and Sun, Bioinformatics, 2023 | 14 methods, 4 datasets | Cell2location above RCTD above SpatialDWLS; Tangram fastest |
| Spotless, eLife, 2024 | 11 methods, 63 silver and 3 gold standards | Cell2location and RCTD top; a plain NNLS baseline beats about half of the dedicated methods |

The last finding is the most useful for a data scientist. A simple, transparent baseline is
competitive, which means the burden of proof is on the complex method for a given dataset, not on
the baseline.

### 5.15 Tests that separate a real result from a well-formatted one

| Test | Procedure | Pass criterion | What it catches |
|---|---|---|---|
| Registration residual | Fit on three landmarks, measure at a fourth | Below 20% of spot diameter | Geometric misalignment |
| Held-out marker residual | Exclude one marker per cell type from the fit | Residual below the count noise, no consistent sign | Missing compartment |
| Synthetic spot | Mix known reference profiles in known proportions and deconvolve | Recovered proportions within 5 percentage points | Solver or reference error |
| Null spot | Deconvolve a spot from a tissue with no tumour | Tumour fraction near zero | Prior or constraint forcing a positive answer |
| Residual autocorrelation | Moran's I on residuals | Near the null expectation | Spatially structured model error |
| Histology concordance | Compare regions with pathologist annotation | Agreement on major compartments | Molecular artifact |
| Patient-level replication | Repeat the statistic in an independent patient | Same direction and approximate magnitude | Pseudoreplication |
| Reference swap | Re-run with a second reference | Conclusions stable | Reference dependence |

The synthetic spot test is the highest-value check because it has a known answer. A pipeline that
cannot recover 70% tumour from a mixture constructed to be 70% tumour will not recover it from
tissue.

### 5.16 What the procedure does not decide

The procedure produces proportions, spatial statistics, and a validation record. It does not decide
whether a spatial biomarker is ready for clinical use, whether a panel covers the right genes, or
how to report a result to a pathologist. Those decisions depend on the intended use and on
evidence outside the assay.

## 6. Worked example

### 6.1 The case

A 55 micrometre Visium spot lies in the invasive margin of a breast tumour. The spot has 1,200
counts of EPCAM, 720 counts of COL1A1, and 80 counts of CD8A. A reference from matched single-cell
data gives the following counts per cell:

| Gene | Tumour cell | Fibroblast | T cell |
|---|---:|---:|---:|
| EPCAM | 200 | 5 | 0 |
| COL1A1 | 10 | 400 | 0 |
| CD8A | 0 | 0 | 150 |

The question is whether the spot is mostly tumour or mostly stroma, and whether a T cell
compartment is present.

### 6.2 Step 1: estimate the cell count

```text
  spot area  = pi x (27.5 micrometres)^2 = 2,375.8 square micrometres
  cell area  = pi x (5 micrometres)^2 = 78.5 square micrometres
  upper bound = 2,375.8 / 78.5 = 30.3 cells
```

The realised count is lower because of extracellular matrix and non-cellular space. The
deconvolution below estimates about 8 cells, which is consistent with a partly stromal region.

### 6.3 Step 2: solve the two-compartment model

Use EPCAM and COL1A1, the two genes that distinguish tumour from fibroblast, and ignore CD8A for
the moment. Let \(t\) be tumour cells and \(f\) fibroblasts.

```text
  1,200 = 200t + 5f
    720 = 10t + 400f
```

From the first equation:

```text
  t = (1,200 - 5f) / 200 = 6 - 0.025f
```

Substitute into the second:

```text
  720 = 10(6 - 0.025f) + 400f
  720 = 60 - 0.25f + 400f
  660 = 399.75f
    f = 1.6510
    t = 6 - 0.025 x 1.6510 = 5.9587
```

Total cells:

```text
  t + f = 7.6097
  tumour fraction = 5.9587 / 7.6097 = 78.3%
  fibroblast fraction = 1.6510 / 7.6097 = 21.7%
```

### 6.4 Step 3: check the fit and the excluded marker

The model reproduces EPCAM and COL1A1 exactly, because two equations and two unknowns determine
the solution. The check is the gene that was not used.

```text
  predicted CD8A = 5.9587 x 0 + 1.6510 x 0 = 0
  observed CD8A = 80
  residual = +80 counts
  T cell equivalents = 80 / 150 = 0.533 cells
```

A residual of 80 counts is not noise at this depth. It means the two-compartment model is missing
a compartment. Adding it:

```text
  c = 80 / 150 = 0.5333 T cells
  total = 5.9587 + 1.6510 + 0.5333 = 8.1430
  tumour = 5.9587 / 8.1430 = 73.2%
  fibroblast = 1.6510 / 8.1430 = 20.3%
  T cell = 0.5333 / 8.1430 = 6.5%
```

The tumour fraction falls from 78.3% to 73.2%. That is a 5.1 percentage point difference, or
about 0.53 cells out of 8.1. The correction matters more when the excluded compartment is large.

### 6.5 Step 4: compute the spatial autocorrelation

Suppose a tumour score is measured on a four-by-four grid of neighbouring spots:

```text
  0.9  0.9  0.6  0.3
  0.8  0.9  0.5  0.2
  0.4  0.5  0.4  0.3
  0.2  0.3  0.3  0.1
```

With queen contiguity, the observed Moran's I is +0.4836. The expected value under spatial
randomness for 16 units is:

```text
  E[I] = -1 / (16 - 1) = -0.0667
```

A permutation test with 2,000 shuffles produced no value as large as the observed one, so the
permutation p-value is below 0.0005. The tumour score is spatially clustered, which is expected
for a tumour region and confirms that the grid is not noise.

The clustering does not establish that the pattern is biological rather than architectural. A
tumour-stroma boundary alone produces a strong Moran's I. The next check is whether the pattern
follows the pathologist's boundary.

### 6.6 Diagnostic 1: residual at a held-out marker

The first diagnostic that would reveal a broken analysis is the residual at marker genes excluded
from the fit.

A working model leaves residuals that are small relative to the counts and that do not have a
consistent sign. A broken model leaves one of three patterns:

- **A large positive residual at a marker for a missing cell type.** In this case, an 80-count
  CD8A residual against a predicted zero means T cells are present and unmodeled. The
  two-compartment proportions absorb the signal into the nearest profile, which here is tumour.
- **A systematic negative residual across many markers.** The model is over-fitting, usually
  because the reference contains too many similar cell types and the constraints force a split.
- **Residuals that correlate with total counts.** Normalization is wrong, and the model is
  fitting sequencing depth rather than composition.

For this spot, the diagnostic flags the missing T cell compartment, and the corrected model is the
three-compartment solution.

### 6.7 Diagnostic 2: spatial autocorrelation of residuals

The second diagnostic is Moran's I computed on the residuals of the fitted model across the spot
grid, rather than on the fitted values.

Residuals should be spatially unstructured if the model is correct and the reference matches the
tissue. A positive residual Moran's I means the model is systematically wrong in a region, which
points to one of three causes:

| Residual pattern | Likely cause | Confirmatory check |
|---|---|---|
| Clustered at a tissue boundary | Registration shift | Recompute the transform with boundary landmarks |
| Clustered in a necrotic region | Reference lacks a necrosis or stress state | Add a stress signature to the reference |
| Clustered by array position | Capture efficiency or bubble artifact | Plot residuals against spot coordinates and library quality |
| Clustered by patient | Reference mismatch is patient-specific | Fit per-patient references or add a random effect |

In this case, a residual map would show the CD8A residual concentrated along the invasive margin,
which is consistent with T cells at the tumour boundary rather than inside the tumour.

### 6.8 The varied assumption

Change one assumption: the tumour reference was derived from a cell line in which COL1A1 is 10
counts per cell. Primary breast tumour cells in a desmoplastic region may express more collagen,
say 60 counts per cell. Re-solve:

```text
  1,200 = 200t + 5f
    720 = 60t + 400f
  t = 6 - 0.025f
  720 = 360 - 1.5f + 400f
  360 = 398.5f
    f = 0.9034
    t = 5.9774
  total = 6.8808
  fibroblast fraction = 0.9034 / 6.8808 = 13.1%
```

The fibroblast fraction falls from 21.7% to 13.1%, a relative change of about 40%, from changing
one reference value. The tumour fraction rises from 78.3% to 86.9%. No residual diagnostic
detects this, because both genes used in the fit are reproduced exactly.

The conclusion changes from "the invasive margin contains a substantial stromal compartment" to
"the spot is predominantly tumour with a modest stromal component". The lesson is that the
reference is an assumption, not a measurement, and its provenance must be reported with the
proportions.

## 7. Code, query, or calculation

The script below performs affine registration from landmarks, a two- and three-compartment
deconvolution, and a Moran's I permutation test, using only the Python standard library. The
output was produced by running the script as written.

```python
"""spatial.py - registration, two-compartment deconvolution, and Moran's I.

Standard library only. Run: python spatial.py
"""
import math
import random


# ---------------------------------------------------------------- registration
def solve3(m, v):
    """Solve a 3x3 linear system by Gaussian elimination with partial pivoting."""
    a = [row[:] + [v[i]] for i, row in enumerate(m)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        p = a[col][col]
        for j in range(col, 4):
            a[col][j] /= p
        for r in range(3):
            if r != col:
                factor = a[r][col]
                for j in range(col, 4):
                    a[r][j] -= factor * a[col][j]
    return [a[0][3], a[1][3], a[2][3]]


LANDMARKS = [
    ((0.0, 0.0), (100.0, 120.0)),
    ((6.5, 0.0), (3600.0, 130.0)),
    ((0.0, 6.5), (110.0, 3380.0)),
]

# x_pix = a*x + b*y + c ; y_pix = d*x + e*y + f
xs = [p[0] for p, _ in LANDMARKS]
ys = [p[1] for p, _ in LANDMARKS]
px = [q[0] for _, q in LANDMARKS]
py = [q[1] for _, q in LANDMARKS]

a, b, c = solve3([[xs[i], ys[i], 1.0] for i in range(3)], px)
d, e, f = solve3([[xs[i], ys[i], 1.0] for i in range(3)], py)

CHECK = (6.5, 6.5)
check_px = (3608.0, 3392.0)
pred_x = a * CHECK[0] + b * CHECK[1] + c
pred_y = d * CHECK[0] + e * CHECK[1] + f
resid_px = math.hypot(pred_x - check_px[0], pred_y - check_px[1])
PIXEL_UM = 0.5
print("REGISTRATION")
print(f"  scale x        : {a:.3f} px/mm")
print(f"  scale y        : {e:.3f} px/mm")
print(f"  rotation terms : b={b:.3f}, d={d:.3f} px/mm")
print(f"  predicted px   : ({pred_x:.1f}, {pred_y:.3f})")
print(f"  measured px    : ({check_px[0]:.1f}, {check_px[1]:.1f})")
print(f"  residual       : {resid_px:.2f} px = {resid_px * PIXEL_UM:.2f} um")
print(f"  spot diameter  : 55.00 um")


# --------------------------------------------------------------- deconvolution
# Reference counts per cell for three marker genes, three cell types.
# order: tumour, fibroblast, T cell
REF = {
    "EPCAM":  (200.0,   5.0,   0.0),
    "COL1A1": ( 10.0, 400.0,   0.0),
    "CD8A":   (  0.0,   0.0, 150.0),
}
OBS = {"EPCAM": 1200.0, "COL1A1": 720.0, "CD8A": 80.0}


def deconvolve_two_gene(genes, ref, obs):
    """Solve the 2x2 system for two cell types using two genes."""
    (g1, g2) = genes
    m = [[ref[g1][0], ref[g1][1]], [ref[g2][0], ref[g2][1]]]
    v = [obs[g1], obs[g2]]
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    t = (v[0] * m[1][1] - m[0][1] * v[1]) / det
    fib = (m[0][0] * v[1] - v[0] * m[1][0]) / det
    return t, fib


t, fib = deconvolve_two_gene(("EPCAM", "COL1A1"), REF, OBS)
total = t + fib
print("\nTWO-COMPARTMENT DECONVOLUTION (tumour + fibroblast)")
print(f"  tumour cells      : {t:.3f}")
print(f"  fibroblast cells  : {fib:.3f}")
print(f"  total cells       : {total:.3f}")
print(f"  tumour fraction   : {t / total:.1%}")
print(f"  fibroblast fraction: {fib / total:.1%}")
for g in ("EPCAM", "COL1A1"):
    pred = t * REF[g][0] + fib * REF[g][1]
    print(f"  residual {g:<7}: {OBS[g] - pred:+.1f} counts")
print(f"  residual CD8A     : {OBS['CD8A'] - (t * REF['CD8A'][0] + fib * REF['CD8A'][1]):+.1f} counts")

c_t = OBS["CD8A"] / REF["CD8A"][2]
total3 = t + fib + c_t
print("\nTHREE-COMPARTMENT DECONVOLUTION")
print(f"  T cells           : {c_t:.3f}")
print(f"  total cells       : {total3:.3f}")
print(f"  fractions         : tumour {t/total3:.1%}, fibroblast {fib/total3:.1%}, T {c_t/total3:.1%}")


# ------------------------------------------------------------------- Moran's I
GRID = [
    [0.9, 0.9, 0.6, 0.3],
    [0.8, 0.9, 0.5, 0.2],
    [0.4, 0.5, 0.4, 0.3],
    [0.2, 0.3, 0.3, 0.1],
]
ROWS, COLS = len(GRID), len(GRID[0])


def morans_i(grid):
    n = ROWS * COLS
    vals = [grid[r][j] for r in range(ROWS) for j in range(COLS)]
    mean = sum(vals) / n
    dev = [v - mean for v in vals]
    num = 0.0
    for r in range(ROWS):
        for j in range(COLS):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, j + dc
                    if 0 <= rr < ROWS and 0 <= cc < COLS:
                        num += dev[r * COLS + j] * dev[rr * COLS + cc]
    w = 8 * (ROWS - 2) * (COLS - 2) + 5 * 2 * (ROWS - 2) + 5 * 2 * (COLS - 2) + 3 * 4
    den = sum(x * x for x in dev)
    return (n / w) * (num / den)


I = morans_i(GRID)
expected = -1.0 / (ROWS * COLS - 1)
random.seed(7)
flat = [GRID[r][j] for r in range(ROWS) for j in range(COLS)]
count = 0
TRIALS = 2000
for _ in range(TRIALS):
    random.shuffle(flat)
    shuffled = [flat[r * COLS:(r + 1) * COLS] for r in range(ROWS)]
    if morans_i(shuffled) >= I:
        count += 1

print("\nMORAN'S I (tumour score on a 4x4 grid, queen contiguity)")
print(f"  observed I        : {I:+.4f}")
print(f"  expected under H0 : {expected:+.4f}")
if count == 0:
    print(f"  permutation p     : < {1.0 / TRIALS:.4f} (0 of {TRIALS} permutations)")
else:
    print(f"  permutation p     : {count / TRIALS:.4f} ({count} of {TRIALS} permutations)")
```

Output:

```text
REGISTRATION
  scale x        : 538.462 px/mm
  scale y        : 501.538 px/mm
  rotation terms : b=1.538, d=1.538 px/mm
  predicted px   : (3610.0, 3390.000)
  measured px    : (3608.0, 3392.0)
  residual       : 2.83 px = 1.41 um
  spot diameter  : 55.00 um

TWO-COMPARTMENT DECONVOLUTION (tumour + fibroblast)
  tumour cells      : 5.959
  fibroblast cells  : 1.651
  total cells       : 7.610
  tumour fraction   : 78.3%
  fibroblast fraction: 21.7%
  residual EPCAM  : +0.0 counts
  residual COL1A1 : +0.0 counts
  residual CD8A     : +80.0 counts

THREE-COMPARTMENT DECONVOLUTION
  T cells           : 0.533
  total cells       : 8.143
  fractions         : tumour 73.2%, fibroblast 20.3%, T 6.5%

MORAN'S I (tumour score on a 4x4 grid, queen contiguity)
  observed I        : +0.4836
  expected under H0 : -0.0667
  permutation p     : < 0.0005 (0 of 2000 permutations)
```

Reading the output:

- The registration residual of 1.41 micrometres is 2.6% of the 55 micrometre spot, which is in
  the "excellent" band. The scale factors differ by about 7% between axes (538.5 versus 501.5
  pixels per millimetre), which is the signature of a slightly anisotropic scan or a stretched
  section, and it is absorbed by the affine transform.
- The two-compartment fit reproduces its own genes exactly, which is expected and not evidence of
  correctness. The CD8A residual is the evidence.
- The permutation test cannot report a p-value below 1/2000. Reporting "p = 0" would be wrong;
  reporting "< 0.0005" is correct.
- Moran's I of +0.4836 against an expectation of -0.0667 is strong clustering. It is consistent
  with a tumour region, and it is also consistent with a tissue boundary, which is why the
  histology check in step 11 is not optional.

## 8. Data, standards, and regulatory context

### 8.1 Platform specifications

The specifications below are the ones that determine analysis choices. They change with product
versions, so they should be checked against the current vendor documentation before a study is
designed.

| Platform | Chemistry | Resolution | Panel | Capture area |
|---|---|---|---|---|
| Visium, original | Sequencing capture | 55 micrometre spots | Whole transcriptome | 6.5 by 6.5 millimetres |
| Visium HD | Sequencing capture | 2 micrometre squares, binned at 2, 8, or 16 micrometres | Whole transcriptome, about 20,000 genes | 6.5 by 6.5 millimetres and 11 by 11 millimetres |
| Xenium | In-situ hybridization | Below 50 nanometres in the image plane | Targeted, up to 5,000 genes on the Prime 5K panel | Instrument-dependent |
| CosMx | In-situ hybridization | Sub-cellular | Targeted, panel-dependent | Instrument-dependent |
| MERSCOPE | In-situ hybridization | Sub-cellular | Targeted, panel-dependent | Instrument-dependent |
| PhenoCycler | Spatial proteomics | Single cell | Tens of proteins | Section-dependent |

The Visium HD binning choice is itself an analysis decision: 2 micrometre bins are close to
single-cell resolution but sparse, 8 micrometre bins are the vendor's recommended default, and
16 micrometre bins increase counts at the cost of resolution.

### 8.2 Data formats and frameworks

- **AnnData** stores the spot-by-gene matrix with per-spot and per-gene annotations.
- **SpatialData** is an open framework that stores images, points, shapes, and tables in a single
  container with a common coordinate system. It addresses the problem that image coordinates,
  spot coordinates, and cell coordinates are often stored in incompatible files.
- **OME-NGFF** and **OME-TIFF** are image formats used for whole-slide images.
- **GeoJSON** and **Parquet** are used for geometries and tables in SpatialData.

A study that stores the image, the coordinates, and the counts in three unrelated files will
struggle to reproduce its own registration.

### 8.3 Standards and their limits

There is no binding international standard for spatial transcriptomics data in the way that
DICOM governs radiology. The relevant conventions are:

| Convention | What it covers | Binding status |
|---|---|---|
| Minimum Information about a high-throughput sequencing experiment | Sequencing metadata | Community convention |
| SpatialData specification | Data container and coordinate systems | Community convention |
| Vendor file formats | Assay-specific outputs | Vendor-defined |
| Foundation models and benchmark datasets | Evaluation | Research convention |

The absence of a binding standard means that reproducibility depends on depositing raw counts,
images, coordinates, and the reference used. A paper that reports proportions without the
reference matrix cannot be reproduced.

### 8.4 Clinical and regulatory context

Spatial omics is not, as of this writing, a standard-of-care diagnostic modality. The regulatory
picture:

- **Laboratory developed tests.** A spatial assay used for a clinical decision would be an LDT
  under CLIA, requiring analytical validation for the panel, the resolution, and the
  deconvolution.
- **In vitro diagnostic regulation in Europe.** A spatial test with a clinical claim would require
  conformity assessment under the In Vitro Diagnostic Regulation, with a classification that
  depends on the intended use.
- **Digital pathology precedent.** Whole-slide imaging systems and some computational pathology
  algorithms have been authorized for specific tasks, which establishes a regulatory path for
  image-based diagnostics but not for spatial omics.
- **Companion diagnostics.** No spatial transcriptomics assay is a companion diagnostic for a
  therapy as of this writing. Claims that a spatial signature selects a treatment are research
  claims until validated in a trial with a specified use.

### 8.5 What the standards do not cover

- No standard defines a minimum registration accuracy for spatial assays.
- No standard defines how to report deconvolution uncertainty.
- No standard defines when a spatial biomarker is analytically valid.
- No standard resolves the unit-of-inference problem.

Those gaps are why the diagnostics in sections 6.6 and 6.7 matter: in the absence of a standard,
the analysis must demonstrate its own validity.

## 9. Industry practice and case evidence

### 9.1 How spatial data is used in industry

Four use cases account for most commercial and academic work.

| Use case | Typical platform | Decision it informs | Maturity |
|---|---|---|---|
| Tumour microenvironment characterization | Visium, CosMx, PhenoCycler | Target biology and biomarker hypothesis | Established in research |
| Immunotherapy response stratification | Imaging panels with CD8, PD-L1, FOXP3 | Whether T cells are excluded from the tumour | Research; trial-embedded |
| Spatial validation of a bulk biomarker | Visium or Xenium | Whether a bulk signal comes from a specific compartment | Established in research |
| Drug distribution and target engagement | Imaging or capture with probe panels | Whether the drug reaches the target cells | Early |

The common pattern is that spatial data is used to explain or validate a signal found by another
method, not to discover a clinical biomarker from scratch. That is a realistic use: spatial context
is expensive, and its value is highest when a cheaper assay has produced an ambiguous result.

### 9.2 Documented platform and method performance

The evidence about methods is unusually clear because several groups have run independent
benchmarks.

| Study | Scope | Documented finding |
|---|---|---|
| Li et al., Nature Methods, 2022 | 16 integration methods, 45 paired datasets, 32 simulated | Cell2location, SpatialDWLS, and RCTD ranked highest for deconvolution; Tangram, gimVI, and SpaGE highest for expression prediction |
| Li et al., Nature Communications, 2023 | 18 methods, 50 datasets | CARD, Cell2location, and Tangram among the strongest, with performance depending on the scenario |
| Yan and Sun, Bioinformatics, 2023 | 14 methods, 4 datasets | Cell2location above RCTD above SpatialDWLS; Tangram fastest and most memory-efficient |
| Spotless, eLife, 2024 | 11 methods, 63 silver and 3 gold standards | Cell2location and RCTD top; an NNLS baseline beat about half of the dedicated methods |

Two inferences follow, and both are labelled as inference rather than fact. First, method choice
should be justified per dataset rather than by reputation, because rankings change with the
scenario. Second, a transparent baseline belongs in every comparison, because a complex method
that does not beat NNLS has not earned its complexity.

A caveat that the benchmarks state explicitly: performance degrades when a cell type is dominant,
when a cell type is rare, when the panel has few genes, and when the reference and the spatial data
come from different platforms. Those are exactly the conditions of most clinical tissue.

### 9.3 Histology prediction models

A separate line of work predicts expression from H&E images. ST-Net (He et al., 2020) trained a
convolutional network to predict spatial gene expression from breast cancer histology patches.
HisToGene (Gao et al., 2021) added a transformer architecture with spatial context. The results are
suggestive and should be treated carefully: a model that predicts a gene from morphology is
learning a correlation between morphology and expression, and the correlation can be driven by
tissue type, processing, or the tumour region rather than by a causal relationship.

The practical value is in triage, not replacement. A model that flags regions likely to express a
marker can guide where to run an expensive assay. It cannot substitute for the assay when a
clinical decision depends on the result.

### 9.4 What is not yet standard

- No spatial transcriptomics assay is a companion diagnostic for a therapy as of this writing.
- No consensus standard defines how to report deconvolution uncertainty.
- No health system has published a prospective trial in which a spatial biomarker changed
  treatment and improved an outcome.
- Reimbursement pathways for spatial assays are not established in most jurisdictions.

A program that plans to use spatial data for a clinical decision should budget for the validation
that these gaps imply.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Registration misalignment | Boundary markers appear in the wrong compartment | Rigid transform fitted to poor landmarks | Signal attributed to the wrong tissue | Landmark residual in micrometres; non-rigid fit if structured |
| Reference missing a cell type | Large residual at a held-out marker | Model has no compartment for a present cell | Proportions absorbed by the nearest profile | Held-out marker residuals; expand the reference |
| Reference from a mismatched platform | Systematic bias in one cell type | Nuclear versus cytoplasmic capture differs | Proportions shifted by a constant factor | Platform-matched reference; benchmark on synthetic mixtures |
| Collinear reference profiles | Unstable proportions between runs | Two cell types have similar signatures | Arbitrary split between types | Merge the types or use a hierarchical prior |
| Normalization confounding depth | Proportions correlate with total counts | Counts normalized without depth correction | Composition read from library size | Depth-aware normalization; include depth as a covariate |
| Spot treated as a replicate | p-values far smaller than the patient count implies | Pseudoreplication | False patient-level claims | Aggregate to patient; mixed models |
| Panel missing the marker | Marker absent from all spots | Targeted panel design | False conclusion that the marker is not expressed | Check panel content before the study |
| Segmentation errors | Transcripts assigned to the wrong cell | Nuclei overlap or boundaries are ambiguous | Cell-level counts contaminated | Inspect segmentation masks; compare with markers |
| Low-count spots included | Noisy proportions at the tissue edge | Normalization amplifies noise | Spurious heterogeneity | Filter by counts and tissue coverage |
| Tissue artifact read as biology | A stripe or bubble pattern in the map | Fold, bubble, or drying artifact | A "spatial gradient" that is technical | Inspect the H&E; plot residuals by array position |
| Batch effect across slides | Patients cluster by slide | Staining and capture variation | Confounded comparison | Include slide as a random effect; stain in one batch |
| Overinterpreting Moran's I | Clustering reported as a mechanism | Autocorrelation conflates architecture and biology | A gradient claim with no mechanism | Compare with the histology boundary |
| One patient, many spots | A "cohort" of one | No patient-level replication | Irreproducible subtype claim | Pre-specify the patient count |
| No orthogonal validation | Result never compared with pathology | Model and reference errors go undetected | A confident artifact | Pathologist review; second assay |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The image and grid are aligned | All spatial attribution depends on it | Tissue deformation, landmark error | Residual at held-out landmarks |
| The reference represents the tissue | Deconvolution is a projection onto the reference | Missing cell type, disease-specific state | Held-out marker residuals |
| Reference and spatial data measure the same transcript population | Counts per cell must be comparable | Nuclei versus whole cells, different probes | Synthetic mixture recovery |
| Cell-type signatures are stable across spots | One reference is applied everywhere | Spatial variation in expression state | Residual autocorrelation |
| Spots are independent conditional on covariates | Standard errors require it | Spatial autocorrelation | Moran's I on residuals |
| The panel covers the biology of interest | Only measured genes can be analyzed | Targeted panel design | Panel content review |
| Segmentation is accurate | Cell-level counts depend on boundaries | Overlapping nuclei, low signal | Mask inspection; marker concordance |
| The section represents the tumour | A slide is a sample of a three-dimensional tissue | Intratumour heterogeneity | Multiple sections or blocks |
| The patient is the unit for clinical claims | Replication requires independent patients | One patient per condition | Pre-specified patient count |
| Normalization removes technical variation | Composition should reflect biology | Capture efficiency gradients | Plot proportions against depth |

## 12. Exercises

### Exercise 1. Compute cells per spot

**Task.** A Visium spot has a diameter of 55 micrometres. Assume a cell cross-section of 10
micrometres in diameter and a tissue cellularity of 60%. Estimate the number of cells per spot.

**Deliverable.** The arithmetic and a statement of the assumptions.

**Acceptance criteria.** Spot area 2,375.8 square micrometres; cell area 78.5 square micrometres;
perfect-packing bound 30.3 cells; at 60% cellularity, about 18 cells. The answer states that the
bound ignores extracellular matrix and that observed counts are often lower.

**Time.** 20 minutes.

### Exercise 2. Register a section from landmarks

**Task.** Three landmarks map spot coordinates (0,0), (6.5,0), and (0,6.5) to pixels (100,120),
(3600,130), and (110,3380). Fit an affine transform and compute the residual at (6.5,6.5), which
is measured at (3608,3392). Use a pixel size of 0.5 micrometres.

**Deliverable.** The transform parameters and the residual in micrometres.

**Acceptance criteria.** Scale factors approximately 538.5 and 501.5 pixels per millimetre;
residual 2.83 pixels, or 1.41 micrometres, which is 2.6% of the spot diameter. The answer
classifies the residual as acceptable.

**Time.** 30 minutes.

### Exercise 3. Deconvolve by hand

**Task.** Using the reference and counts in section 6, solve the two-compartment model and report
the proportions. Then compute the residual at CD8A and interpret it.

**Deliverable.** The two equations, the solution, and the interpretation.

**Acceptance criteria.** Tumour 5.96 cells and fibroblast 1.65 cells; proportions 78.3% and 21.7%;
CD8A residual +80 counts, equivalent to 0.53 T cells, which indicates a missing compartment.

**Time.** 30 minutes.

### Exercise 4. Test the reference assumption

**Task.** Recompute Exercise 3 with a tumour reference COL1A1 value of 60 instead of 10, and state
how the conclusion changes.

**Deliverable.** The new solution and a two-sentence interpretation.

**Acceptance criteria.** Fibroblast 0.90 cells and tumour 5.98 cells; fibroblast fraction falls
from 21.7% to 13.1%; the answer states that no residual diagnostic detects the change because the
fit still reproduces its own genes.

**Time.** 25 minutes.

### Exercise 5. Compute Moran's I on a small grid

**Task.** For the four-by-four grid in section 6, compute Moran's I with rook contiguity instead of
queen contiguity, and compare with the value reported in the text.

**Deliverable.** The calculation and the comparison.

**Acceptance criteria.** The value is close to but not identical to +0.4836, and the answer states
that the weight definition changes the statistic while the qualitative conclusion of clustering
does not change. A correct answer notes the different total weight W.

**Time.** 35 minutes.

### Exercise 6. Diagnose a pseudoreplication error

**Task.** A paper reports that a spatial niche is associated with response, using 12 responders and
10 non-responders, with a p-value computed over 84,000 spots. Identify the error, state the correct
test, and describe what the authors should report.

**Deliverable.** A 300-word statistical review.

**Acceptance criteria.** The review identifies pseudoreplication, proposes a patient-level summary
and a test with 22 observations, and asks for the per-patient effect sizes and a confidence
interval. It notes that the spot-level p-value is not interpretable as patient-level evidence.

**Time.** 30 minutes.

### Exercise 7. Design a validation study

**Task.** Design a study to test whether a deconvolution-based fibroblast fraction predicts
resistance to checkpoint blockade. Specify the cohort, the assay, the reference, the endpoint, and
the analysis.

**Deliverable.** A two-page protocol.

**Acceptance criteria.** The protocol specifies a patient-level endpoint, a pre-specified
threshold, an ancestry- and platform-matched reference, a held-out marker diagnostic, an orthogonal
validation in a subset, and a sample size justified by the expected effect.

**Time.** 60 minutes.

### Exercise 8. Critique a spatial claim

**Task.** A press release states that a company's spatial platform "maps the tumour
microenvironment at single-cell resolution across the whole transcriptome". Write a critique.

**Deliverable.** A 300-word critique with at least three specific questions.

**Acceptance criteria.** The critique distinguishes capture-based whole-transcriptome assays at
multi-cell resolution from imaging panels at single-cell resolution with targeted panels, and asks
about bin size, segmentation, panel content, and validation. It does not accept the combination of
"single-cell" and "whole transcriptome" without evidence.

**Time.** 30 minutes.

## 13. Self-check answers

**Exercise 1.** Spot area = pi x 27.5 squared = 2,375.8 square micrometres. Cell area = pi x 5
squared = 78.54 square micrometres. Perfect-packing bound = 2,375.8 / 78.54 = 30.25 cells. At 60%
cellularity, 0.60 x 30.25 = 18.15 cells. The realised count is lower because cells are not
discs, nuclei are smaller than cells, and the capture area includes extracellular matrix. A
reported cell count of 18 should be described as an estimate with an upper bound of about 30.

**Exercise 2.** The transform is x_pix = 538.462 x + 1.538 y + 100 and y_pix = 1.538 x + 501.538 y
+ 120. At (6.5, 6.5), predicted x = 3,500 + 10 + 100 = 3,610, predicted y = 10 + 3,260 + 120 =
3,390. The measured point is (3,608, 3,392), so the residual is the square root of (2 squared plus
2 squared), which is 2.83 pixels. At 0.5 micrometres per pixel, that is 1.41 micrometres, or 2.6%
of the 55 micrometre spot. The residual is acceptable for spot-level analysis.

**Exercise 3.** The equations are 1,200 = 200t + 5f and 720 = 10t + 400f. From the first,
t = 6 - 0.025f. Substituting gives 720 = 60 - 0.25f + 400f, so 660 = 399.75f and f = 1.651. Then
t = 5.959. The proportions are 78.3% tumour and 21.7% fibroblast. The CD8A prediction is zero and
the observation is 80 counts, a residual of 80 counts, which is 80/150 = 0.53 T cells. The
two-compartment model is missing a compartment.

**Exercise 4.** With COL1A1 at 60 for tumour cells, the equations are 1,200 = 200t + 5f and
720 = 60t + 400f. Substituting t = 6 - 0.025f gives 720 = 360 - 1.5f + 400f, so 360 = 398.5f and
f = 0.903. Then t = 5.977. The fibroblast fraction is 0.903 / 6.881 = 13.1%. The tumour fraction
rises to 86.9%. No residual diagnostic detects the change, because both fitted genes are
reproduced exactly. The reference value is an assumption that must be reported.

**Exercise 5.** With rook contiguity, corner cells have two neighbours, edge cells three, and
interior cells four, giving a total weight of 4 x 2 + 8 x 3 + 4 x 4 = 48, compared with 84 for
queen contiguity. The numerator also changes because diagonal pairs are excluded. The resulting I
is close to but not identical to the queen value, and the conclusion of positive spatial
clustering is unchanged. The exact value depends on the grid and the weights, which is why the
weight definition must be reported.

**Exercise 6.** Rubric. Full credit requires identifying pseudoreplication, proposing a
patient-level summary such as the mean niche score per patient, testing with 22 observations, and
asking for per-patient effect sizes with a confidence interval. A review that proposes a mixed
model with a patient random effect but retains the spot-level p-value as the headline fails.

**Exercise 7.** Rubric. Full credit requires a patient-level endpoint, a pre-specified analysis, a
matched reference, a held-out marker diagnostic, orthogonal validation, and a sample size
justified by an expected effect. A protocol that uses spot-level tests or omits the reference
provenance fails.

**Exercise 8.** Rubric. Full credit distinguishes sequencing capture from imaging, asks about bin
size and segmentation, asks whether the panel is whole transcriptome or targeted, and asks for
validation against pathology. A critique that only disputes the word "single-cell" misses the
transcriptome claim.

## 14. Interview questions

**1. Why can a 55 micrometre spot not be treated as a single cell?**
A strong answer gives the geometric estimate, about 10 to 30 cells at the packing bound and fewer
in practice, and states that the measurement is a mixture. A weak answer says the spot is "small".

**2. What does deconvolution estimate, and what does it assume?**
A strong answer describes the linear mixture model with non-negative weights and names the
reference, the platform match, and the missing-cell-type assumption. A weak answer describes it as
"separating cell types" without the model.

**3. How do you check a deconvolution result?**
A strong answer names held-out marker residuals, synthetic mixtures with known proportions, and
comparison with pathology. A weak answer says "look at the heatmap".

**4. What is the difference between Visium and Xenium?**
A strong answer contrasts capture-based whole-transcriptome measurement at multi-cell resolution
with imaging-based targeted panels at sub-cellular resolution. A weak answer treats them as
competing products with the same output.

**5. What does Moran's I tell you, and what does it not tell you?**
A strong answer says it measures spatial clustering against a null of randomness and does not
establish mechanism or independence. A weak answer interprets it as a p-value for biology.

**6. Why is a spot not a replicate?**
A strong answer explains spatial autocorrelation and the patient as the independent unit. A weak
answer says "because they are correlated" without connecting it to inference.

**7. How would you detect a registration error?**
A strong answer uses landmarks held out from the fit and expresses the residual relative to the
spot diameter, and notes that structured residuals suggest non-rigid deformation. A weak answer
says "check the overlay by eye".

**8. What happens if the reference lacks a cell type?**
A strong answer explains that the signal is absorbed by the nearest profile and detected by a
residual at a held-out marker. A weak answer says the method "handles it".

**9. When is an NNLS baseline preferable to a Bayesian method?**
A strong answer cites the benchmark finding that NNLS beats about half of dedicated methods and
argues for a transparent baseline when the panel is small or compute is limited. A weak answer
assumes newer is better.

**10. What would make a spatial biomarker clinically ready?**
A strong answer requires analytical validation, patient-level prospective evidence, a specified
use, and a regulatory path. A weak answer says "a significant association with response".

## 15. Key terms

- **Affine transform** — six-parameter linear mapping between coordinates.
- **Bin** — aggregated capture square at a chosen resolution.
- **Capture spot** — barcoded region on a sequencing array.
- **CARD** — spatial deconvolution method using reference and spatial correlation.
- **Cell2location** — Bayesian deconvolution method with uncertainty estimates.
- **Deconvolution** — estimation of cell-type proportions from mixed expression.
- **FFPE** — formalin-fixed paraffin-embedded tissue.
- **Held-out marker** — marker excluded from a fit and used as an independent check.
- **Immune desert** — absence of cytotoxic T cells from tumour and stroma.
- **Immune excluded** — T cells present but confined to the stroma.
- **Landmark** — corresponding point used to fit a registration.
- **MERSCOPE** — imaging-based spatial platform.
- **Moran's I** — spatial autocorrelation statistic.
- **Neighbourhood enrichment** — test of co-localization beyond expectation.
- **Niche** — recurring spatial combination of cell types.
- **NNLS** — non-negative least squares.
- **Non-rigid registration** — warping transform allowing local deformation.
- **Pseudoreplication** — treating non-independent units as replicates.
- **RCTD** — robust cell type decomposition method.
- **Reference profile** — per-cell-type expression signature.
- **Registration** — alignment of image and capture coordinates.
- **Segmentation** — assignment of transcripts to cell boundaries.
- **Spatial autocorrelation** — similarity of nearby values.
- **SpatialData** — open container for spatial omics data.
- **Spatial transcriptomics** — RNA measurement with retained coordinates.
- **Tangram** — mapping method aligning single-cell and spatial data.
- **Tumour microenvironment** — cellular and extracellular context of a tumour.
- **Visium** — sequencing-based capture platform.
- **Visium HD** — 2 micrometre barcoded capture platform.
- **Whole-slide image** — digitized histology slide.
- **Xenium** — imaging-based in-situ hybridization platform.

## 16. Sources

Accessed 26 October 2026 unless otherwise stated.

- 10x Genomics. *Visium Spatial Platform*. https://www.10xgenomics.com/platforms/visium
- 10x Genomics. *Visium HD Spatial Gene Expression 2.0 User Guide*, 2026.
  https://cdn.10xgenomics.com/image/upload/v1775578153/support-documents/CG001679_VisiumHD_GeneExpression_2.0_UG_Rev_A.pdf
- 10x Genomics. *Xenium Analyzer Specifications*, 2024.
  https://cdn.10xgenomics.com/image/upload/v1712007637/support-documents/CG000630_XeniumAnalyzer_Specifications_RevE.pdf
- Stahl, P. L., et al. *Visualization and Analysis of Gene Expression in Tissue Sections by
  Spatial Transcriptomics*. Science 353, 78-82, 2016. https://doi.org/10.1126/science.aaf2403
- Li, B., et al. *Benchmarking Spatial and Single-Cell Transcriptomics Integration Methods for
  Transcript Distribution Prediction and Cell Type Deconvolution*. Nature Methods, 2022.
  https://doi.org/10.1038/s41592-022-01480-9
- Li, H., et al. *A Comprehensive Benchmarking with Practical Guidelines for Cellular
  Deconvolution of Spatial Transcriptomics*. Nature Communications, 2023.
  https://doi.org/10.1038/s41467-023-37168-7
- Yan, X., and Sun, X. *Benchmarking and Integration of Methods for Deconvoluting Spatial
  Transcriptomic Data*. Bioinformatics 39, 2023. https://doi.org/10.1093/bioinformatics/btac805
- Sang-aram, S., et al. *Spotless: A Reproducible Pipeline for Benchmarking Cell Type
  Deconvolution in Spatial Transcriptomics*. eLife, 2024.
  https://doi.org/10.7554/eLife.88431
- Cable, D. M., et al. *Robust Decomposition of Cell Type Mixtures in Spatial Transcriptomics*.
  Nature Biotechnology 40, 517-526, 2022. https://doi.org/10.1038/s41587-021-00830-w
- Kleshchevnikov, V., et al. *Cell2location Maps Fine-Grained Cell Types in Spatial
  Transcriptomics*. Nature Biotechnology 40, 661-671, 2022.
  https://doi.org/10.1038/s41587-021-01001-7
- He, B., et al. *Integrating Spatial Gene Expression and Breast Morphology via Deep Learning*.
  Nature Biomedical Engineering 4, 827-834, 2020. https://doi.org/10.1038/s41551-020-0577-8
- Moses, L., and Pachter, L. *Museum of Spatial Transcriptomics*. Nature Methods 19, 534-546,
  2022. https://doi.org/10.1038/s41592-022-01409-2
- SpatialData documentation. https://spatialdata.scverse.org/
- Palla, G., et al. *Squidpy: A Scalable Framework for Spatial Omics Analysis*. Nature Methods 19,
  171-178, 2022. https://doi.org/10.1038/s41592-021-01358-2

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`12_polygenic-risk-scores.md`](12_polygenic-risk-scores.md)
- Next topic: [`14_ai-for-protein-engineering.md`](14_ai-for-protein-engineering.md)
- Related topic: [`05_spatial-transcriptomics.md`](05_spatial-transcriptomics.md)
- Related topic: [`04_single-cell.md`](04_single-cell.md)
- Module survey: [`../../course/12_omics_precision_medicine.md`](../../course/12_omics_precision_medicine.md)
- Laboratory exercises and datasets: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)

