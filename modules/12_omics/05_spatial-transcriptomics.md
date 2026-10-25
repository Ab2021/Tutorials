# Spatial Transcriptomics: Spots, Cells, Domains, and Deconvolution

> Module 12 — Omics and Precision Medicine
> Topic 05 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01A, 04, topics 01, 03, 04
> Version 2026.3

## 1. Why this topic exists

Tissue is not a well-mixed tube. A tumour margin, a lymphoid follicle, and an inflamed crypt
differ from their neighbours by cell composition, signalling, and treatment response. Bulk
transcriptomics averages all of it, and single-cell transcriptomics dissolves the architecture
by dissociating the tissue. Spatial transcriptomics is the assay class that keeps the
coordinates.

The commercial reason is that many decisions are architectural. Whether a drug penetrates a
tumour, whether an immune response is organized into a tertiary lymphoid structure, and whether
a target is expressed by malignant cells or by adjacent stroma are all questions about location.
A target expressed in the tumour core and one expressed at the invasive edge may have the same
bulk expression and different therapeutic implications.

The technical reason is that spatial assays are a bundle of trade-offs, and most of them are
invisible in the vendor summary. A capture-based assay with 55 micrometre spots measures a
mixture of roughly five to fifteen cells per spot; it is a bulk assay with coordinates. An
imaging-based assay measures individual transcripts at subcellular resolution but only for a
panel of hundreds to a few thousand genes. Neither is "better." They answer different questions,
and a program that buys one for the other's question will conclude that the technology did not
work.

The analytical reason is that spatial data require deconvolution, and deconvolution requires a
reference. If the reference does not contain the cell types present in the tissue, the
deconvolution does not return "unknown"; it returns a plausible mixture of the types it knows.
A tumour sample deconvolved against a healthy reference will report malignant cells as
fibroblasts, with a good fit and no warning. This is the single most consequential failure in
the field, and it is silent.

A final consideration is cost per claim. A 12-section capture run at roughly USD 3,000 per
section is USD 36,000 before analysis, and it supports a domain-level claim in six patients. A
targeted imaging run at similar cost supports a cell-level claim in fewer patients. The budget
does not decide the assay, but it constrains how many patients can be replicated, and replication
at the patient level is what turns a spatial observation into evidence.

This topic covers the spatial assay landscape, the deconvolution arithmetic, domain and niche
analysis, and the validation path against histology. It depends on the single-cell concepts in
topic 04 and feeds the multi-omics integration in topic 07.

### 1.1 The failure pattern

1. A capture-based assay is chosen for a question about single cells.
2. Spots are treated as cells in a clustering pipeline.
3. Deconvolution is run against a reference from a different tissue or a different condition.
4. The fit is good, so the result is trusted; the reference's missing cell types are absorbed
   into the nearest available type.
5. Spatial domains are detected with a method that ignores coordinates, or with one that
   over-smooths and erases the boundary of interest.
6. Ligand-receptor analysis is run on co-localized cell types and interpreted as communication.
7. Histology is never consulted, so a tissue fold or an acellular region is annotated as a
   domain.

Each step is a default in at least one tutorial. The reference problem in step 3 and 4 is the
one that produces confident, wrong biology.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Pathologist | A molecular claim that contradicts the slide | Not involved in the analysis |
| Target lead | A cell-type attribution that is a deconvolution artifact | Depends on reference quality |
| Translational lead | A spatial biomarker that does not reproduce | Depends on assay choice |
| Computational biologist | Weeks on data that cannot answer the question | Inherited the assay choice |
| Clinical team | A tissue-based test with no validated threshold | Cannot inspect the fit |
| Program | A licensing decision based on a misread niche | No orthogonal validation |

### 1.3 The industry question

> Is this molecular signal a validated biomarker, a causal target, or an artifact of batch,
> ancestry, and tissue composition?

Spatial data add a fourth artifact: the mixture inside the measurement unit. A spot's expression
is a weighted sum of its cells. Any statement about "cells in this region" is a statement about
a deconvolution, and the deconvolution's assumptions must travel with the claim.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** capture-based and imaging-based spatial assays on resolution, panel size,
   throughput, and tissue compatibility.
2. **Compute** deconvolution weights for a spot from marker expression and a reference, and
   state what the residual does and does not tell you.
3. **Choose** a reference that matches the tissue and condition, and diagnose reference
   mismatch.
4. **Detect** spatial domains with a method that uses coordinates, and interpret a domain
   boundary.
5. **Explain** why a spot is not a cell and what follows for clustering and differential
   expression.
6. **Evaluate** ligand-receptor co-localization claims for the assumptions they require.
7. **Validate** a spatial finding against histology or immunohistochemistry.
8. **Design** a spatial study with the right assay, section count, and replication.

## 3. The core mental model

A spatial assay produces a coordinate-indexed matrix. The coordinate is reliable; the
biological unit behind the coordinate depends on the assay.

```text
  Tissue section on a slide
      |
      +--> capture-based: barcoded spots or beads on a grid
      |         |
      |         +--> spot = 1 to 15 cells (55 um) or a 2-8 um bin
      |         +--> whole transcriptome (poly-A) or panel
      |         +--> H&E or immunofluorescence image for registration
      |
      +--> imaging-based: probes in situ, decoded by cycles
                |
                +--> single transcript, subcellular localization
                +--> panel of 100s to 1,000s of genes
                +--> segmentation defines the cell
      v
  Coordinate-indexed matrix: (x, y) or (cell, x, y) by gene
      |
      +--> QC: counts, genes, tissue coverage, artifacts
      +--> normalization
      +--> deconvolution (capture) or segmentation (imaging)
      +--> domain detection (spatial clustering)
      +--> niche and neighbour analysis
      +--> ligand-receptor co-localization
      v
  Claim: "region R is enriched for cell type C" or "cells of type C contact type D"
```

### 3.1 The resolution-panel trade-off

Spatial assays sit on a frontier: you can have many genes or small measurement units, rarely
both at high quality.

| Assay class | Unit | Genes | Throughput | Tissue |
|---|---|---|---|---|
| Capture, low density (Visium) | 55 um spot (~5-15 cells) | Whole transcriptome | 1-4 sections per slide | Fresh-frozen or FFPE |
| Capture, high density (Visium HD) | 2 um or 8 um bins | Whole transcriptome | 1-2 sections | FFPE or fresh-frozen |
| Imaging, targeted (Xenium, CosMx) | Single cell, subcellular | 100s to ~5,000 | 2 slides per run | FFPE or fresh-frozen |
| Sequencing-based single-cell spatial (Slide-seq, Stereo-seq) | 10 um bead or subcellular bin | Whole transcriptome | Limited | Fresh-frozen |

The frontier is not a ranking. A 5,000-plex imaging panel at single-cell resolution is the right
instrument for a cell-contact question. A whole-transcriptome capture array at 55 micrometres is
the right instrument for a discovery question about regional programs, provided the analysis
deconvolves.

### 3.2 Four different objects called "spatial"

The word "spatial" covers four distinct analytical objects, and mixing them is the source of
most confusion.

1. **Spot or bin**: a measurement unit with coordinates and a mixture of cells.
2. **Cell**: a segmented object with coordinates and a transcript count.
3. **Domain**: a contiguous region with a shared expression profile, discovered by spatial
   clustering. A domain is not a cell type; it is a tissue state.
4. **Niche**: the local neighbourhood around a cell or spot, summarized by the composition of
   its neighbours.

A claim about a domain ("the invasive margin") is not a claim about a cell type. A claim about a
niche ("tumour cells contact exhausted T cells") requires cell-level resolution or a
deconvolution with validated weights.

### 3.3 What deconvolution assumes

Deconvolution solves, for each spot, a mixture equation: the observed expression vector is a
non-negative weighted sum of reference profiles, plus error.

```text
y_spot = sum_k w_k * x_k + error
```

The assumptions:

- The reference profiles are correct for this tissue and condition.
- The cell types in the spot are represented in the reference.
- Expression is linear in cell abundance, with no interaction effects.
- The weights are non-negative and sum to a biologically plausible total.

When assumption two fails, the fit can still be excellent, because the model will use the
available types to approximate the missing one. The residual does not diagnose this. Only a
comparison with histology, or a reference built from the same tissue, does.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Spot | Capture area containing multiple cells | Treated as a cell | 55 um diameter typical |
| Bin | Aggregation unit in high-density capture | Confused with a cell | 2-8 um |
| Bead | Capture particle in bead-based spatial assays | Confused with a spot | 10 um typical |
| Segmentation | Assignment of transcripts to cells in imaging assays | Assumed exact | Boundary definition |
| Deconvolution | Estimation of cell-type proportions per spot | Treated as ground truth | Proportions |
| Reference | Cell-type expression profiles used for deconvolution | Assumed representative | Matrix |
| Platform effect | Systematic difference between reference and spatial assay | Ignored | Bias term |
| Domain | Contiguous region with shared expression | Confused with cell type | Region label |
| Niche | Local neighbourhood composition | Confused with domain | Composition vector |
| Moran's I | Spatial autocorrelation statistic | Used without a weight definition | Dimensionless |
| Spatial autocorrelation | Correlation of values with location | Confused with biological similarity | Statistic |
| Ligand-receptor analysis | Co-localization of ligand and receptor | Interpreted as signalling | Score |
| Co-localization | Presence in the same location | Confused with interaction | Proportion |
| Registration | Alignment of the expression grid to the histology image | Assumed exact | Transform |
| Tissue fold | Overlap of tissue on itself | Mistaken for a domain | Artifact |
| Acellular region | Area with no cells | Annotated as low-expression domain | Artifact |
| Panel | Targeted gene set | Assumed to cover the pathway | Gene list |
| Probe dropout | Failure to detect a present transcript | Read as absence | Rate |
| Transcript density | Transcripts per unit area | Confused with expression level | Counts/um2 |
| Capture efficiency | Fraction of transcripts captured | Ignored | Proportion |
| Visium | 10x capture-based spatial assay | Used generically | Platform |
| Visium HD | High-density capture with small bins | Confused with single-cell | Platform |
| Xenium | 10x imaging-based in situ assay | Used generically | Platform |
| CosMx | NanoString imaging-based assay | Platform | Platform |
| MERFISH | Multiplexed error-robust FISH | Platform | Platform |
| Slide-seq | Bead-based sequencing spatial assay | Platform | Platform |
| Stereo-seq | High-density sequencing spatial assay | Platform | Platform |
| H&E | Haematoxylin and eosin stain | Treated as optional | Image |
| ROI | Region of interest | Confused with a domain | Region |
| Orthogonal validation | Confirmation with a different method | Confused with replication | Assay |

Two definitions are worth stating separately.

**A domain is a clustering result, not an anatomical structure.** A domain labelled "tumour"
may correspond to the pathologist's tumour region, or it may correspond to a necrotic area with
similar expression. The correspondence must be checked, not assumed.

**A platform effect is not a batch effect.** A platform effect is the systematic difference in
measured expression between the reference assay (usually single-cell) and the spatial assay. It
arises from chemistry, capture, and quantification, and it biases deconvolution weights unless
modeled. A batch effect is variation between runs of the same assay.

## 5. The mechanics, step by step

### Step 1. State the spatial question and its unit

**Input:** the biological question.

**Output:** one sentence naming the unit (spot, cell, domain, niche) and the comparison.

**Rationale:** "Is the target expressed by malignant cells at the invasive margin?" is a
cell-level question in a defined region. "Does the treated tissue show a different regional
program?" is a domain-level question. The unit determines the assay, the analysis, and the
validation. A cell-level question answered with 55 micrometre spots is a deconvolution claim,
and it must be reported as one.

### Step 2. Choose the assay against three constraints

**Input:** unit, tissue type, panel requirement.

**Output:** assay selection with the trade-off stated.

**Rationale:** Three constraints decide most cases. Resolution: cell-level questions need
imaging or high-density capture. Panel: discovery of an unknown program needs whole
transcriptome; a hypothesis about 300 genes can use a targeted panel. Tissue: FFPE compatibility
and section age rule out some chemistries. A fourth constraint, throughput, decides how many
sections can be run and therefore how many samples can be replicated.

### Step 3. Prepare and register the section

**Input:** tissue block or slide.

**Output:** stained image plus expression coordinates in a common frame.

**Rationale:** Registration maps capture coordinates to histology. A registration error of a few
micrometres is tolerable at spot resolution and fatal at subcellular resolution. Check
registration with landmarks: vessel walls, tissue edges, and folds. A section with a fold should
be excluded or masked, because the fold creates a region with doubled transcript density that
looks like a domain.

### Step 4. QC the section and the spots

**Input:** raw counts and image.

**Output:** per-spot metrics and a mask of usable tissue.

**Rationale:** Metrics include counts per spot, genes per spot, mitochondrial fraction (for
capture assays), transcript density, and the fraction of the array covered by tissue. Spots off
tissue have ambient counts and should be masked, not clustered. The distribution of counts per
spot is often bimodal, with a low mode from acellular or necrotic areas.

### Step 5. Normalize with the assay's composition in mind

**Input:** counts.

**Output:** normalized values.

**Rationale:** Capture assays have variable capture efficiency across the array and across
sections. Library-size normalization is a baseline; a variance-stabilizing transform is usually
needed for visualization. For imaging assays, transcript density varies with segmentation and
cell size, so normalization should account for cell area or volume. A normalization that
ignores area will make large cells look like high expressors.

### Step 6. Choose and validate the reference

**Input:** candidate references: single-cell atlas, matched single-cell run, sorted populations.

**Output:** a reference with a stated match and a validation check.

**Rationale:** The reference must contain the cell types present in the tissue, in the condition
being studied. A healthy reference may lack malignant and reactive states. A matched single-cell
run from the same tissue is best but expensive. Validate by checking that the reference contains
markers observed in the spatial data, and that the deconvolution residuals do not concentrate in
regions where a missing type is expected.

### Step 7. Deconvolve with a stated model and check the fit

**Input:** spatial matrix and reference.

**Output:** per-spot weights, residuals, and diagnostics.

**Rationale:** Methods differ in how they model platform effects, whether they allow
unexplained expression, and whether they estimate uncertainty. Report the method, the version,
and the fit. A good fit is necessary but not sufficient: a misspecified reference can fit well.
Check the weights against histology and against known marker expression.

### Step 8. Detect domains with coordinates

**Input:** normalized expression and coordinates.

**Output:** domain labels.

**Rationale:** Domain detection uses spatial smoothing or a spatial prior. A method that ignores
coordinates will produce domains that are scattered across the section. A method that
over-smooths will erase a thin boundary, which is often the structure of interest. Test at least
two smoothing scales, and compare domains with the histology image.

### Step 9. Segment cells when the assay is imaging-based

**Input:** transcript coordinates and a nuclear or membrane stain.

**Output:** cell boundaries and per-cell counts.

**Rationale:** Segmentation errors are the dominant technical artifact in imaging-based spatial
data. Over-segmentation splits one cell into two; under-segmentation merges neighbouring cells
into a false doublet. Check the distribution of transcripts per cell and the fraction of
transcripts outside any cell. A high unassigned fraction indicates a segmentation problem.

### Step 10. Annotate domains and cells with evidence

**Input:** domain labels and marker expression.

**Output:** labels with positive and negative evidence.

**Rationale:** A domain label should be supported by marker expression, by histology, and by
consistency across sections. A domain called "tumour" should show tumour markers and correspond
to a region the pathologist identifies as tumour. A label without histological support is a
hypothesis.

### Step 11. Analyse niches and neighbours

**Input:** cell or spot labels with coordinates.

**Output:** neighbourhood composition and enrichment statistics.

**Rationale:** A niche is defined by a radius or a k-nearest-neighbour graph. The radius is a
biological assumption: too small and no neighbours are captured, too large and the niche is the
whole tissue. Report the radius and test sensitivity. A niche enrichment claim should be
compared with a null that preserves the overall composition but randomizes locations.

### Step 12. Evaluate ligand-receptor claims carefully

**Input:** cell-type or domain labels and expression.

**Output:** candidate interactions with a stated evidence level.

**Rationale:** Co-localization of a ligand and a receptor is necessary but far from sufficient
for signalling. The analysis assumes that expression implies protein, that the proteins are
secreted or surface-exposed, that the cells are in contact or proximity, and that the receptor
is functional. Report these as candidate interactions, and validate with protein data where the
claim matters.

### Step 13. Validate against histology or immunohistochemistry

**Input:** the spatial finding.

**Output:** orthogonal evidence.

**Rationale:** Histology is the ground truth for tissue architecture. An immunohistochemistry
stain for the candidate protein, on an adjacent section, is the cheapest and most convincing
validation. A spatial finding that contradicts the pathology should be treated as an analysis
problem until resolved.

### Step 14. Report with the assay's limits

**Input:** everything above.

**Output:** a report with assay, resolution, panel, reference, deconvolution method, validation,
and limits.

**Rationale:** The report must state what the measurement unit was and what the deconvolution
assumed. A reader who does not know that a "cell-type proportion" came from a 55 micrometre spot
cannot judge the claim.

### Step 15. Aggregate to the patient before claiming

**Input:** spot-level or cell-level results.

**Output:** patient-level summaries and a test at the patient level.

**Rationale:** Sections and spots are nested within patients. A comparison of margin versus core
across 12 sections from 6 patients has, at most, 6 independent units for a patient-level claim,
and fewer if sections are not balanced. Aggregating to patient-level means (or proportions) and
testing across patients is the honest analysis. A mixed model with patient and section random
effects is an alternative when the design is unbalanced.

The consequence of ignoring nesting is predictable: p-values that fall as sections are added,
without any increase in the number of patients. Reviewers now ask for the patient count, not the
spot count, and the answer changes the interpretation.

### Branch points

**Branch A — the panel does not contain the pathway of interest.** The analysis cannot test that
pathway. Report the panel's coverage before interpreting a negative result; absence of signal
may be absence of probe.

**Branch B — the tissue is necrotic or folded.** Mask the region. Necrotic areas have degraded
RNA and low counts; folds have doubled density. Neither is biology.

**Branch C — the reference lacks a condition-specific state.** Build a reference from matched
data, or add the missing state as an estimated profile and report the uncertainty. Do not
interpret the forced assignment.

**Branch D — the question is about cell contact.** Spot-level data cannot answer it. Use imaging
or high-density capture with segmentation.

### Decision table: question to assay

| Question | Unit | Assay | Analysis | Cannot support |
|---|---|---|---|---|
| Regional expression program | Domain | Capture 55 um, whole transcriptome | Deconvolution + spatial clustering | Cell contact |
| Cell-type proportion in a region | Spot | Capture, any density | Deconvolution | Single-cell expression |
| Target expression by cell type | Cell | Imaging panel or HD capture | Segmentation + annotation | Genes outside panel |
| Cell-cell contact | Cell | Imaging, subcellular | Segmentation + neighbourhood | Whole-transcriptome discovery |
| Rare cell type localization | Cell | Imaging with markers | Segmentation + detection | Unmarked rare types |
| Tissue-wide atlas | Domain | Capture, multiple sections | Integration + domains | Fine-grained niches |
| Tumour margin signature | Domain/spot | Capture FFPE | Deconvolution + boundary analysis | Causal signalling |

### Edge cases

**Low counts per spot.** A spot with 200 counts has unstable deconvolution. Diagnostic: the low
tail of the counts distribution. Consequence: weights dominated by noise. Mitigation: aggregate
neighbouring spots, or exclude with a stated threshold.

**Segmentation errors.** Diagnostic: transcripts per cell bimodal; high unassigned fraction.
Consequence: false cell types and false contacts. Mitigation: improve nuclear segmentation;
report the unassigned fraction.

**Platform effect.** Diagnostic: reference markers systematically lower in spatial data than
expected. Consequence: biased weights. Mitigation: a method that models platform effects, and a
matched reference where possible.

**Reference mismatch.** Diagnostic: a region with high residual and a cell type that histology
shows but the reference lacks. Consequence: missing type absorbed into a neighbour. Mitigation:
matched reference; add the missing profile; validate with IHC.

**Tissue fold.** Diagnostic: a line of doubled transcript density with a matching fold in the
image. Consequence: a false boundary or domain. Mitigation: mask or exclude the section.

**Batch across sections.** Diagnostic: section identity explains a principal component.
Consequence: section effects read as biology. Mitigation: balanced design; include section as a
covariate; integrate only if justified.

**Panel dropout.** Diagnostic: a gene expected from histology is absent across all spots.
Consequence: false negative pathway call. Mitigation: check panel coverage; use orthogonal
staining.

**Acellular spots.** Diagnostic: very low counts and no nuclei in the image. Consequence: a
low-expression domain. Mitigation: mask by tissue coverage.

**Edge effects.** Diagnostic: domains that follow the array boundary. Consequence: boundary
artifacts. Mitigation: exclude spots near the edge or model edge distance.

**Over-smoothing.** Diagnostic: a known thin structure (for example, a crypt layer) disappears.
Consequence: lost biology. Mitigation: test multiple smoothing scales.

**Ambient transcript spread.** In imaging assays, transcripts diffuse from their cell of origin
during fixation. Diagnostic: a gradient of signal away from a high-expressing region.
Consequence: false co-localization. Mitigation: report the diffusion scale; require proximity
within it, not merely within the same field.

**Optical crowding.** In dense tissue, transcripts from adjacent cells overlap and segmentation
assigns them to the wrong cell. Diagnostic: high transcript counts in small cells, and marker
expression that does not match morphology. Consequence: false cell identities. Mitigation:
membrane staining, expanded segmentation, or spot-level analysis with a stated limitation.

**Neighbourhood definition drift.** The niche radius is chosen after seeing which radius gives
a result. Diagnostic: the reported radius is not in the analysis plan and the result is
sensitive to it. Consequence: a selected finding. Mitigation: pre-specify the radius from cell
size and ligand range, and report sensitivity.

**Section sampling bias.** The section analysed is not representative of the block. Diagnostic:
domains differ between serial sections. Consequence: a tissue-level claim from one region.
Mitigation: multiple sections per block, and a stated sampling frame.

**Circular reference.** The reference is built from the same section being deconvolved.
Diagnostic: the reference construction used the spatial data. Consequence: the deconvolution is
self-confirming and cannot detect a missing type. Mitigation: build the reference from an
independent sample, or hold out regions during reference construction.

**Spot-level multiple testing.** Testing 12,000 genes across 3,000 spots with a spot-level model.
Diagnostic: the number of tests is reported as genes x spots, or the model has no patient term.
Consequence: anticonservative inference. Mitigation: aggregate to patient, or use a mixed model
and report the family explicitly.

**Normalization across sections.** Each section has its own capture efficiency, so raw counts
are not comparable across sections. Diagnostic: median counts per spot differ by more than
twofold across sections. Consequence: section differences read as biology. Mitigation: normalize
per section, include section as a covariate, and report the medians.

**Dominated by one patient.** A domain appears in all sections of one patient and nowhere else.
Diagnostic: patient composition per domain. Consequence: a patient-specific state generalized as
a tissue property. Mitigation: report per-patient domain presence and require at least three
patients before a tissue-level claim.

### Which spatial statistic answers which question

| Question | Statistic or method | Null | Reports |
|---|---|---|---|
| Is a gene spatially structured? | Moran's I, Geary's C | Permutation of values | Statistic, p-value |
| Are there contiguous domains? | Spatial clustering with a prior | Stability across scales | Domain labels |
| Is a cell type enriched in a region? | Proportion difference, aggregated | Patient-level permutation | Effect, interval |
| Do cell types co-localize? | Neighbourhood enrichment | Randomization preserving composition | Enrichment ratio |
| Is a boundary sharp or gradual? | Boundary detection, gradient | Smoothed null | Boundary width |
| Does composition differ by condition? | Beta-binomial or quasi-binomial | Patient-level test | Proportion difference |

The table exists to prevent a common mismatch: using Moran's I to claim a domain, or using
domain labels to claim co-localization. Each statistic answers one question, and the claim must
match the statistic.

### What the procedure produces

The output is a coordinate-indexed object plus a record: assay and resolution, panel, reference
and its match, deconvolution method and fit, domain method and scale, segmentation quality,
histological validation, and limits. The record is what makes a spatial claim auditable.

## 6. Worked example

A study asks whether a fibroblast activation program is enriched at the invasive margin of
colorectal tumours. Twelve FFPE sections from six patients are run on a capture assay with
55 micrometre spots, whole-transcriptome chemistry, and an H&E image for registration.

### 6.1 The measurement unit

A 55 micrometre spot has an area of:

```text
area = pi x (55/2)^2 = 2375.8 um^2
```

If a typical cell occupies 200 to 400 square micrometres in section, a spot contains roughly 6
to 12 cells. The spot is therefore a small bulk sample. Any cell-type statement from this assay
is a deconvolution statement, and the analysis must say so.

### 6.2 Deconvolution arithmetic for one spot

A spot has two marker genes measured at 40 and 10 normalized counts. The reference has two
candidate cell types with profiles:

| Cell type | Marker A | Marker B |
|---|---:|---:|
| Fibroblast | 8 | 1 |
| Epithelial | 2 | 6 |

Solve the two equations:

```text
8 w_f + 2 w_e = 40
1 w_f + 6 w_e = 10

determinant = 8 x 6 - 2 x 1 = 46
w_f = (40 x 6 - 2 x 10) / 46 = 220 / 46 = 4.783
w_e = (8 x 10 - 40 x 1) / 46 = 40 / 46 = 0.870
```

The weights sum to 5.652, so the proportions are:

```text
fibroblast = 4.783 / 5.652 = 0.846
epithelial = 0.870 / 5.652 = 0.154
```

The fit is exact because there are two equations and two unknowns. The residual is zero, and it
carries no information. This is the central arithmetic lesson of deconvolution: with as many
cell types as markers, the model is exactly determined, and a perfect fit is guaranteed. Real
deconvolution uses many genes and few types, and the residual becomes informative only then.

### 6.3 The same spot with a matched reference

The reference above came from healthy mucosa. A matched reference from tumour tissue gives a
different fibroblast profile, with marker A at 12 and marker B at 1:

```text
12 w_f + 2 w_e = 40
 1 w_f + 6 w_e = 10

determinant = 12 x 6 - 2 x 1 = 70
w_f = (40 x 6 - 2 x 10) / 70 = 220 / 70 = 3.143
w_e = (12 x 10 - 40 x 1) / 70 = 80 / 70 = 1.143
```

Proportions:

```text
fibroblast = 3.143 / 4.286 = 0.733
epithelial = 1.143 / 4.286 = 0.267
```

The fibroblast proportion falls from 84.6% to 73.3%, and the epithelial proportion rises from
15.4% to 26.7%. The spot did not change; the reference did. A reference that under-represents a
cell type's marker expression will systematically inflate that type's weight elsewhere, because
the model compensates by adding more of it.

### 6.4 Multiple testing across domains

The study tests 12,000 genes for differential expression between margin and core domains. The
Bonferroni threshold is:

```text
0.05 / 12,000 = 4.17 x 10^-6
```

The leading gene has p = 3.0 x 10^-9, so its q-value by BH is:

```text
q = 3.0 x 10^-9 x 12,000 / 1 = 3.6 x 10^-5
```

Suppose 640 genes pass at q < 0.05. The reporting sentence: "640 of 12,000 genes differed
between margin and core (BH q < 0.05); the leading gene had log2FC = 1.4 (95% CI 0.9-1.9) and
q = 3.6 x 10^-5." Note that the test is on spots, which are not independent: neighbouring spots
share cells and technical variation. A spot-level test overstates precision, and the honest
analysis aggregates to patient or uses a mixed model with patient and section effects.

### 6.5 Spatial autocorrelation

Before interpreting a domain, check that expression is spatially structured. A simple binary
adjacency Moran's I on a 4 x 4 grid with two blocks of values:

```text
grid values:
1 1 0 0
1 1 0 0
0 0 2 2
0 0 2 2

n = 16, mean = 0.75
I = 0.394
```

A value near 0.4 indicates moderate positive spatial autocorrelation: neighbours resemble each
other more than distant cells. For a random arrangement, I would be near zero. This check
prevents the common error of running spatial domain detection on data with no spatial structure
and interpreting the resulting domains as anatomy.

### 6.6 Diagnostic 1: the residual map

The deconvolution residual is plotted per spot. Most spots have low residuals, but a band along
one edge of the section has residuals three times the median, and the band corresponds to a
region the pathologist calls "mucinous tumour."

The reference contains no mucinous tumour profile. The model has approximated it with
fibroblasts and epithelial cells, and the residual is concentrated exactly where the missing
type lives. Without the residual map, the analysis would report a "fibroblast-rich margin" in a
region that is mostly mucinous tumour.

The fix is to add a mucinous profile to the reference, or to restrict the claim to regions the
reference covers. The residual map is the diagnostic; a global fit statistic would have hidden
the localized failure.

### 6.7 Diagnostic 2: the fold

One section shows a linear region with twice the transcript density of its surroundings and a
matching bright line in the H&E image. It is a tissue fold, not a domain. Including it creates a
"high-density" domain that correlates with every highly expressed gene, which then dominates the
domain analysis.

Excluding the section, or masking the fold, changes the number of margin-core differentially
expressed genes from 640 to 410. The 230 lost genes were fold artifacts.

### 6.8 Vary one assumption: spots treated as cells

Suppose the analysis skips deconvolution and clusters spots directly, then labels a cluster
"activated fibroblasts" because it expresses fibroblast markers. The cluster is a mixture of
fibroblast-rich and epithelial-rich spots, and its "activation" signature is partly the
epithelial contribution.

Re-running with deconvolution and testing the fibroblast weight per spot, rather than the raw
expression, changes the result: the margin enrichment remains (weight 0.73 versus 0.51 in the
core, a difference of 0.22 in proportion), but the "activation program" genes are no longer
significant after adjusting for the epithelial weight. The conclusion changes from "fibroblasts
at the margin are activated" to "the margin contains more fibroblasts; their activation state is
not established."

One assumption — that a spot's expression is a property of one cell type — changes the
biological claim. It is the most common error in the spatial literature and the easiest to
prevent by stating the unit.

### 6.9 What the report says

> Twelve FFPE sections, six patients, capture assay at 55 um resolution, whole transcriptome.
> Registration checked against H&E; one section excluded for a tissue fold. 3,412 spots on
> tissue, median 8,400 counts per spot. Deconvolution against a matched tumour reference with
> six profiles; residual map reviewed and a mucinous profile added. Margin-core comparison
> aggregated to patient; 12,000 genes; BH correction; 410 genes at q < 0.05. Fibroblast
> proportion higher at margin (0.73 versus 0.51). Activation-state claim not supported after
> adjustment. Limits: spot resolution, panel-wide but capture-limited, no protein validation.

## 7. Code, query, or calculation

The script solves a two-type deconvolution by hand, computes a simple Moran's I, applies BH, and
shows the effect of a reference change.

### 7.1 Two-type deconvolution

```python
def solve2(a, b, c, d, e, f):
    """Solve a*w1 + b*w2 = e; c*w1 + d*w2 = f."""
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("singular system")
    w1 = (e * d - b * f) / det
    w2 = (a * f - e * c) / det
    return w1, w2

for label, ref in [("healthy", (8, 2, 1, 6)), ("matched", (12, 2, 1, 6))]:
    w1, w2 = solve2(*ref, 40, 10)
    total = w1 + w2
    print(label, "weights", round(w1, 3), round(w2, 3),
          "proportions", round(w1 / total, 3), round(w2 / total, 3))
```

Expected output:

```text
healthy weights 4.783 0.87 proportions 0.846 0.154
matched weights 3.143 1.143 proportions 0.733 0.267
```

The arithmetic is intentionally small enough to check by hand. In production, the system has
hundreds of genes and a non-negativity constraint, and is solved by least squares with
regularization.

### 7.2 Simple Moran's I on a grid

```python
grid = [[1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2]]
rows, cols = len(grid), len(grid[0])
vals = [v for row in grid for v in row]
n = len(vals)
mean = sum(vals) / n
z = [v - mean for v in vals]

def neighbours(i, j):
    out = []
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if 0 <= i + di < rows and 0 <= j + dj < cols:
            out.append((i + di, j + dj))
    return out

pos = {(i, j): i * cols + j for i in range(rows) for j in range(cols)}
W = [[0] * n for _ in range(n)]
for i in range(rows):
    for j in range(cols):
        for ni, nj in neighbours(i, j):
            W[pos[(i, j)]][pos[(ni, nj)]] = 1

num = n * sum(W[k][l] * z[k] * z[l] for k in range(n) for l in range(n))
den = sum(sum(r) for r in W) * sum(v * v for v in z)
print("n =", n, "weights =", sum(sum(r) for r in W))
print("Moran's I =", round(num / den, 3))
```

Expected output:

```text
n = 16 weights = 48
Moran's I = 0.394
```

The weight matrix here is binary and not row-standardized, and it is asymmetric at the edges.
That is acceptable for a quick check but not for a formal test. A production analysis uses a
row-standardized weight matrix and a permutation null.

### 7.3 Deconvolution residual under a missing type

```python
# Observed spot has three markers; reference has two types, so the system is overdetermined
obs = [40, 10, 26]
fib = [8, 1, 3]
epi = [2, 6, 4]

# Least squares for two unknowns via the normal equations
s11 = sum(a * a for a in fib)
s12 = sum(a * b for a, b in zip(fib, epi))
s22 = sum(b * b for b in epi)
t1 = sum(a * y for a, y in zip(fib, obs))
t2 = sum(b * y for b, y in zip(epi, obs))
det = s11 * s22 - s12 * s12
w1 = (t1 * s22 - s12 * t2) / det
w2 = (s11 * t2 - s12 * t1) / det

pred = [w1 * a + w2 * b for a, b in zip(fib, epi)]
resid = [round(y - p, 2) for y, p in zip(obs, pred)]
ss = sum((y - p) ** 2 for y, p in zip(obs, pred))
print("weights:", round(w1, 3), round(w2, 3))
print("predicted:", [round(p, 2) for p in pred])
print("residuals:", resid)
print("RMSE:", round((ss / len(obs)) ** 0.5, 3))
```

Expected output:

```text
weights: 4.87 1.4
predicted: [41.76, 13.27, 20.21]
residuals: [-1.76, -3.27, 5.79]
RMSE: 3.971
```

The third marker has a residual of +5.79 while the first two are negative. The residuals are not
random: they are structured by marker, which is what a missing cell type or a platform effect
looks like. Inspecting residuals per gene, not just the global RMSE, is how the failure is
found.

### 7.4 BH across genes

```python
def bh(pvals, m):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    q = [0.0] * len(pvals)
    prev = 1.0
    for rank in range(len(pvals), 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        q[i] = min(prev, 1.0)
    return q

pvals = [3.0e-9, 2.2e-7, 4.8e-6, 3.1e-5, 2.4e-4]
q = bh(pvals, 12000)
for p, qq in zip(pvals, q):
    print(f"p={p:.1e} q={qq:.2e}")
```

Expected output:

```text
p=3.0e-09 q=3.60e-05
p=2.2e-07 q=1.32e-03
p=4.8e-06 q=1.92e-02
p=3.1e-05 q=9.30e-02
p=2.4e-04 q=5.76e-01
```

The fourth gene has p = 3.1 x 10^-5, which looks impressive and is not significant after
correction across 12,000 genes (q = 0.093). The fifth is far from significant.

### 7.5 Assumptions and limits of this code

- The two-type solver has no non-negativity constraint; a real solution must, and a negative
  weight signals a misspecified reference.
- The least-squares example is unweighted; genes have different variances and should be
  weighted.
- Moran's I uses a binary, non-standardized weight matrix and no permutation null.
- The BH function assumes independence or positive dependence, which spatial data violate
  because neighbouring spots are correlated.
- No code here performs segmentation, registration, or domain detection.

## 8. Data, standards, and regulatory context

### 8.1 Formats and metadata

Spatial data lack a single international reporting standard. The de facto conventions are:

- **AnnData/`.h5ad`** with spatial coordinates in `obsm`, widely used in Python.
- **SpatialExperiment** (Bioconductor) for R workflows.
- **OME-TIFF** for images, with pyramid levels and channel metadata.
- **CELLxGENE schema** for cell-level metadata where cells are segmented.
- **HCA metadata schema** for atlas submissions.
- **MINSEQE** for the sequencing layer of capture assays.

A spatial dataset should record: assay and chemistry, resolution, panel, section thickness,
fixation, registration method, and the image used. Without these, the data cannot be compared
across studies, and the resolution is the most important field.

### 8.2 Vendor specifications and what they constrain

Current vendor documentation (2025-2026) gives concrete limits that should be checked against
the question:

- **Xenium pre-designed panels**: human brain 266 genes, breast 280, colon 322, lung 289,
  immuno-oncology 380, multi-tissue and cancer 377, skin 260; a 5,000-gene pan-tissue panel is
  available, with up to 100 add-on genes.
- **Xenium instrument**: subcellular localization (better than 30 nm in XY), 2 slides per run,
  up to 236 square millimetres per slide.
- **Visium**: 55 micrometre spots, 100 micrometre centre-to-centre, 6.5 mm capture area;
  Visium HD offers 2 and 8 micrometre bins.

The inference, not stated by the vendor: a 380-gene panel cannot discover a new pathway, and a
55 micrometre spot cannot resolve cell contact. Both are instrument properties, not analysis
problems.

### 8.3 Pathology and laboratory standards

When spatial data support a clinical claim, the tissue and the assay fall under existing
laboratory frameworks:

- **CAP** accreditation for the pathology laboratory and for immunohistochemistry.
- **CLIA** for any test reported clinically.
- **ISO 15189:2022** for laboratory quality and competence.
- **IVDR 2017/746** in the EU for in vitro diagnostic devices.
- **FDA** device regulation where a spatial assay is marketed for a clinical indication.

None of these specifies a deconvolution method. They require that the test, including its
analysis, be validated and reproducible.

### 8.4 What the standards do not settle

- The reference to use for deconvolution.
- The smoothing scale for domain detection.
- The neighbourhood radius for niche analysis.
- The evidence threshold for a ligand-receptor claim.
- The number of sections needed for a tissue-level claim.

These are scientific choices, and they change results. A spatial paper that reports the software
but not these parameters is not reproducible.

## 9. Industry practice and case evidence

### 9.1 The benchmark evidence

Two benchmark papers define current expectations for deconvolution.

- Li et al., *Nature Methods* 2022, benchmarked 16 integration methods on 45 paired spatial and
  single-cell datasets plus 32 simulated datasets. Tangram, gimVI, and SpaGE performed best for
  predicting the spatial distribution of transcripts not measured in the spatial assay;
  cell2location, SpatialDWLS, and RCTD performed best for cell-type deconvolution of spots.
- Li et al., *Nature Communications* 2023, benchmarked 18 deconvolution methods across 50 real
  and simulated datasets spanning seqFISH+, MERFISH, ST, Visium, Slide-seqV2, and Stereo-seq.
  CARD, cell2location, and Tangram were the best overall; CARD, cell2location, Tangram, and RCTD
  were the most robust.

Neither benchmark evaluates the reference, because both use references constructed from the
same tissue. That is the gap between a benchmark and a study.

A third finding from the same benchmarks is that robustness varies by platform. Methods differ
in how they behave on high-resolution imaging data versus spot data, and a method that wins on
Visium does not automatically win on MERFISH. The assay determines the appropriate method as
much as the tissue does.

The practical reading: method choice matters less than reference quality. The benchmarks use
references matched to the simulated or measured tissue; a real study rarely has that luxury, and
the reference is where the error enters.

### 9.2 What pharma does

Documented practice:

- Target validation uses spatial data to confirm that a target is expressed by the intended cell
  type in the intended region, usually alongside immunohistochemistry.
- Pathology groups review spatial domains against H&E before the domains enter a report.
- Translational groups use spatial data to define immune exclusion or infiltration patterns for
  trial enrichment.
- Assay selection is driven by the question: whole-transcriptome capture for discovery,
  targeted imaging for cell-level validation.

Inference: the most common organizational failure is a computational team delivering domains
without pathology review, and a pathology team unable to interpret the domain labels. The fix is
a joint review session with the image and the domain map side by side, which takes an hour and
prevents most mislabels.

### 9.3 The reference problem in practice

Public references are abundant but tissue-specific. A reference built from blood will not
deconvolve a liver biopsy; a reference from healthy colon will not represent a mucinous tumour.
The practical consequence is that deconvolution accuracy in a new tissue is unknown until it is
checked, and the check is histological, not statistical.

A useful industry heuristic, stated as inference rather than documented fact: if the
deconvolution residual map has a spatial pattern, the reference is missing something. Random
residuals suggest noise; structured residuals suggest a missing profile or a platform effect.

### 9.4 The validation expectation

Regulators and reviewers increasingly expect orthogonal validation of spatial claims. For a
cell-type localization claim, that means immunohistochemistry or immunofluorescence on an
adjacent section. For a domain claim, it means agreement with a pathologist's annotation. For a
ligand-receptor claim, it means protein-level evidence, which spatial transcriptomics alone
cannot provide.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Spot treated as cell | Cluster markers mixed | Mixture in measurement unit | Wrong cell attribution | Deconvolve; state unit |
| Reference mismatch | Structured residual map | Missing cell type | Forced assignment | Matched reference; IHC |
| Platform effect ignored | Reference markers uniformly low | Chemistry difference | Biased weights | Model platform effect |
| Tissue fold included | Doubled density line | Section artifact | False domain | Mask or exclude |
| Necrotic region clustered | Low counts, degraded RNA | RNA quality | False domain | Mask by histology |
| Segmentation error | Bimodal transcripts per cell | Boundary detection | False cells and contacts | Improve segmentation |
| Domain method ignores space | Scattered domain labels | Non-spatial clustering | No architecture | Spatial prior or smoothing |
| Over-smoothing | Thin structure disappears | Kernel too wide | Lost boundary | Multi-scale comparison |
| Panel dropout | Expected gene absent everywhere | Probe failure | False negative | Check panel; stain |
| Ligand-receptor overreach | Many predicted interactions | Co-localization only | False signalling claim | Protein validation |
| Spot-level testing | p-values tiny and uniform | Non-independent spots | Overstated precision | Aggregate to patient |
| Section batch | PC1 by section | Processing | Section as biology | Balance; covariate |
| Acellular spots | Very low counts | No cells | Low-expression domain | Mask |
| Edge artifacts | Domains along array edge | Boundary effects | False structure | Exclude edge spots |
| Registration error | Markers offset from image | Transform error | Wrong region labels | Landmark check |

Four red flags:

**If the residual map has spatial structure, stop.** The reference is missing something.

**If a domain boundary follows the array edge or a fold, stop.** It is an artifact.

**If a cell-contact claim comes from 55 micrometre spots, stop.** The assay cannot support it.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The reference covers the tissue's cell types | Deconvolution | Missing malignant or reactive state | Residual map; IHC |
| Expression is linear in abundance | Mixture model | Cell-cell interaction, ambient | Simulation; marker check |
| Weights are non-negative and sum sensibly | Biological plausibility | Misspecified reference | Constrained solver |
| Spots are independent for testing | Standard errors | Neighbouring spots share cells | Aggregate; mixed model |
| Registration is accurate | Region assignment | Tissue distortion | Landmark check |
| Segmentation recovers cells | Per-cell claims | Crowded nuclei, membrane stains | Unassigned fraction |
| Domains correspond to anatomy | Interpretation | Smoothing, artifacts | Pathologist review |
| Panel covers the pathway | Negative results | Targeted panel | Panel coverage check |
| Transcript density reflects expression | Imaging quantification | Cell size, segmentation | Area normalization |
| Section is representative | Tissue-level claim | Sampling, heterogeneity | Multiple sections |

The reference assumption dominates. A spatially structured residual is the only reliable warning
that it has failed, and it requires looking at a map rather than a number.

## 12. Exercises

**Exercise 1 — Assay selection (application, 30 minutes).**
For each question, choose an assay class and state what it cannot answer: (a) which cell types
contact tumour cells at the margin; (b) which pathways differ between two regions; (c) whether a
target is expressed by malignant cells.
*Deliverable:* three-row table.
*Criteria:* assay matched to unit; a limitation per row; no "best assay" claim.
*Time:* 30 minutes.

**Exercise 2 — Deconvolution by hand (calculation, 35 minutes).**
A spot has markers A = 30 and B = 12; the reference has type 1 (A = 6, B = 2) and type 2
(A = 1, B = 5). Solve for weights and proportions. Then recompute with type 1 at (A = 9, B = 2)
and state the change.
*Deliverable:* arithmetic for both cases.
*Criteria:* weights and proportions correct to three decimals; change quantified.
*Time:* 35 minutes.

**Exercise 3 — Residual interpretation (critique, 30 minutes).**
A deconvolution reports a global RMSE of 0.08 and a residual map with a bright band along one
edge. Write the diagnosis and the next step.
*Deliverable:* 250-word memo.
*Criteria:* localized failure identified; a specific fix proposed; no claim that low RMSE means
correct.
*Time:* 30 minutes.

**Exercise 4 — Domain validation (application, 45 minutes).**
You have a domain labelled "invasive margin." Describe three checks against histology and one
statistical check that would support or refute the label.
*Deliverable:* a validation plan.
*Criteria:* three histological checks; one statistical check with a null; a decision rule.
*Time:* 45 minutes.

**Exercise 5 — Spot versus cell (critique, 40 minutes).**
A paper reports "tumour cells at the margin express ligand X and T cells express receptor Y,
indicating interaction," using 55 micrometre spots. Write a critique.
*Deliverable:* 400-word critique.
*Criteria:* unit mismatch named; three assumptions listed; a validation proposed.
*Time:* 40 minutes.

**Exercise 6 — Niche radius (design, 35 minutes).**
Design a niche analysis for immune exclusion. Specify the radius, the null model, the
sensitivity analysis, and the statistic.
*Deliverable:* a one-page plan.
*Criteria:* radius justified biologically; null preserves composition; sensitivity stated.
*Time:* 35 minutes.

**Exercise 7 — Section QC (application, 30 minutes).**
You have 12 sections; 2 have folds, 1 has 40% necrotic area, and 1 has low counts throughout.
Decide which to exclude and state the consequence for the study's power.
*Deliverable:* a QC decision table.
*Criteria:* each section adjudicated; power consequence quantified; exclusion pre-specified.
*Time:* 30 minutes.

**Exercise 8 — Study design (design, 60 minutes).**
Design a spatial study to test whether a drug changes immune infiltration at the tumour margin.
Specify assay, sections per patient, patients per arm, domains, statistics, and validation.
*Deliverable:* two-page design.
*Criteria:* assay matched to unit; replication at patient level; validation named; limits stated.
*Time:* 60 minutes.

## 13. Self-check answers

**Answer 1 (assay selection).** (a) Cell contact requires imaging-based in situ with subcellular
resolution and segmentation; it cannot survey the whole transcriptome. (b) Regional pathway
differences can use capture-based whole-transcriptome at 55 micrometres with deconvolution; it
cannot resolve individual cells. (c) Target expression by malignant cells requires imaging with
a panel containing the target, or high-density capture with segmentation; a targeted panel
cannot discover whether other genes matter.

**Answer 2 (deconvolution).** For A = 30, B = 12 and reference type 1 (6, 2), type 2 (1, 5):

```text
det = 6 x 5 - 2 x 1 = 28
w1 = (30 x 5 - 2 x 12) / 28 = 126 / 28 = 4.5
w2 = (6 x 12 - 30 x 1) / 28 = 42 / 28 = 1.5
proportions: 0.75 and 0.25
```

With type 1 at (9, 2):

```text
det = 9 x 5 - 2 x 1 = 43
w1 = (30 x 5 - 2 x 12) / 43 = 126 / 43 = 2.93
w2 = (9 x 12 - 30 x 1) / 43 = 78 / 43 = 1.81
proportions: 0.618 and 0.382
```

The first type's proportion falls from 75% to 61.8% when its marker A is corrected upward. The
reference's marker scale directly controls the inferred composition.

**Answer 3 (residual).** A low global RMSE with a spatial residual pattern indicates a localized
model failure, most likely a cell type absent from the reference in that band. The next step is
to inspect the histology of the band, identify the missing population, add its profile to the
reference, and re-run. Reporting the global RMSE as evidence of correctness is invalid because
the error is concentrated, not random.

**Answer 4 (domain validation).** Histological checks: (1) the region corresponds to a boundary
the pathologist identifies between tumour and stroma; (2) a marker stain for the domain's
defining protein is enriched there; (3) the boundary is consistent across serial sections.
Statistical check: the domain's genes have higher spatial autocorrelation than a permutation
null that shuffles labels while preserving the domain size. Decision rule: accept the label if
at least two histological checks and the statistical check agree; otherwise relabel as an
expression domain without anatomical interpretation.

**Answer 5 (critique).** At 55 micrometres, each spot contains multiple cells, so a spot
expressing both ligand and receptor does not show that the same cells or neighbouring cells
express them. The claim requires: (1) cell-level resolution or validated deconvolution, (2)
evidence that the transcripts are translated into protein, (3) proximity at a scale relevant to
the ligand's range, and (4) a functional readout. None is provided. The appropriate claim is
"spots in this region express both ligand and receptor," which is a co-localization observation,
not an interaction.

**Answer 6 (niche).** Radius: 50 to 100 micrometres, roughly two to four cell diameters, because
immune exclusion operates at that scale. Null: randomize cell locations while preserving the
overall cell-type composition and the tissue mask; repeat 1,000 times. Sensitivity: repeat at
25, 50, 100, and 200 micrometres and report whether the enrichment persists. Statistic: the
observed fraction of tumour cells with at least one T cell within the radius, compared to the
null distribution, reported as an enrichment ratio with a permutation p-value.

**Answer 7 (section QC).** Exclude the two folded sections and the low-count section; mask the
necrotic area in the fourth and retain the section. Consequence: 9 of 12 sections remain, which
reduces the effective sample from 6 patients to about 5 if sections are nested in patients. The
power loss should be stated before analysis; if the study needs 6 patients, one patient must be
re-cut or the claim restricted.

**Answer 8 (design).** Assay: capture-based whole transcriptome on FFPE, 2 sections per patient,
one at baseline and one at resection, or 2 regions per tumour. Patients: 10 per arm, balanced
across batches. Domains: margin and core defined by pathology, confirmed by marker expression.
Statistics: deconvolution per spot, aggregation to patient, paired comparison of immune
proportion at the margin, BH across genes. Validation: CD3/CD8 immunohistochemistry on adjacent
sections. Limits: spot resolution, no cell-contact claims, no protein-level signalling claims.

## 14. Interview questions

**1. What is the difference between capture-based and imaging-based spatial assays?**
Strong answer: capture measures mixtures in spots or bins with whole-transcriptome chemistry;
imaging measures single transcripts in cells with a targeted panel; the trade-off is resolution
versus panel size.
Weak answer: "one is newer."

**2. Why is a spot not a cell?**
Strong answer: a 55 micrometre spot contains multiple cells; expression is a mixture; cell-type
claims require deconvolution and are probabilistic.
Weak answer: "spots are small, so they are basically cells."

**3. What does deconvolution assume?**
Strong answer: a reference covering the tissue's cell types, linear mixing, non-negative weights,
and no interaction effects; the residual does not diagnose a missing type.
Weak answer: "it estimates proportions" with no assumptions.

**4. How do you know the reference is adequate?**
Strong answer: matched tissue and condition, marker coverage check, residual map inspection, and
histological validation.
Weak answer: "use a public atlas."

**5. What is a spatial domain?**
Strong answer: a contiguous region with shared expression discovered by spatial clustering; it
is not a cell type and must be validated against anatomy.
Weak answer: "a cluster on a map."

**6. How do you detect a tissue fold?**
Strong answer: doubled transcript density in a line, matching the image; mask or exclude.
Weak answer: "check the QC metrics."

**7. Why are spot-level p-values misleading?**
Strong answer: neighbouring spots are not independent; they share cells and technical variation;
aggregate to patient or use a mixed model.
Weak answer: "because there are many spots."

**8. What can ligand-receptor analysis actually claim?**
Strong answer: co-localization of transcripts, a candidate interaction, requiring protein and
functional validation.
Weak answer: "it shows signalling."

**9. When would you choose a 5,000-plex imaging panel over whole-transcriptome capture?**
Strong answer: when the question is cell-level and the candidate set is large but bounded, and
when tissue architecture matters more than discovering unknown genes.
Weak answer: "more genes is better."

**10. How many sections do you need?**
Strong answer: enough to cover patient-level heterogeneity; sections are nested in patients, and
the patient is the unit for a tissue-level claim.
Weak answer: "one good section."

**11. What is a platform effect?**
Strong answer: systematic expression difference between the single-cell reference and the
spatial assay, which biases weights unless modeled.
Weak answer: confuses it with batch.

**12. What would make you reject a spatial domain result?**
Strong answer: a boundary following an artifact, no histological correspondence, a residual
pattern, or a domain defined by a single marker with no negative evidence.
Weak answer: "if the p-value is not significant."

## 15. Key terms

- **Acellular region** — area with no cells, often misclustered.
- **Bin** — aggregation unit in high-density capture.
- **Capture assay** — spatial method using barcoded spots or beads.
- **Co-localization** — presence in the same location.
- **CosMx** — imaging-based spatial platform.
- **Deconvolution** — estimation of cell-type proportions per spot.
- **Domain** — contiguous region with shared expression.
- **H&E** — haematoxylin and eosin stain.
- **Imaging assay** — in situ method resolving single transcripts.
- **Ligand-receptor analysis** — co-localization-based interaction candidate scoring.
- **MERFISH** — multiplexed error-robust FISH.
- **Moran's I** — spatial autocorrelation statistic.
- **Niche** — local neighbourhood composition.
- **OME-TIFF** — image format with metadata.
- **Panel** — targeted gene set.
- **Platform effect** — systematic difference between reference and spatial assay.
- **Probe dropout** — failure to detect a present transcript.
- **Registration** — alignment of expression grid to image.
- **Reference** — cell-type profiles used in deconvolution.
- **Segmentation** — assignment of transcripts to cells.
- **Slide-seq** — bead-based spatial sequencing.
- **Spatial autocorrelation** — correlation of values with location.
- **SpatialExperiment** — Bioconductor container for spatial data.
- **Spot** — capture area containing multiple cells.
- **Stereo-seq** — high-density spatial sequencing.
- **Tissue fold** — section artifact with doubled density.
- **Transcript density** — transcripts per unit area.
- **Visium** — 10x capture-based spatial platform.
- **Visium HD** — high-density capture with small bins.
- **Xenium** — 10x imaging-based in situ platform.

## 16. Sources

Methods and benchmarks:

- Li, B. et al., "Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type deconvolution," *Nature Methods* 19:662-670, 2022. https://doi.org/10.1038/s41592-022-01480-9 (accessed 25 October 2026).
- Li, H. et al., "A comprehensive benchmarking with practical guidelines for cellular deconvolution of spatial transcriptomics," *Nature Communications* 14:1548, 2023. https://doi.org/10.1038/s41467-023-37168-7 (accessed 25 October 2026).
- Kleshchevnikov, V. et al., "Cell2location maps fine-grained cell types in spatial transcriptomics," *Nature Biotechnology* 40:661-671, 2022. https://doi.org/10.1038/s41587-021-01139-4 (accessed 25 October 2026).
- Cable, D. M. et al., "Robust decomposition of cell type mixtures in spatial transcriptomics," *Nature Biotechnology* 40:517-526, 2022. https://doi.org/10.1038/s41587-021-00830-w (accessed 25 October 2026).
- Stahl, P. L. et al., "Visualization and analysis of gene expression in tissue sections by spatial transcriptomics," *Science* 353:78-82, 2016. https://doi.org/10.1126/science.aaf2403 (accessed 25 October 2026).

Platforms and data:

- 10x Genomics, Xenium panels. https://www.10xgenomics.com/platforms/xenium/panels (accessed 25 October 2026).
- 10x Genomics, Visium spatial gene expression. https://www.10xgenomics.com/platforms/visium (accessed 25 October 2026).
- Human Cell Atlas Data Portal. https://data.humancellatlas.org/ (accessed 25 October 2026).
- CELLxGENE schema. https://github.com/chanzuckerberg/single-cell-curation (accessed 25 October 2026).

Standards and regulation:

- FGED Society, MINSEQE. https://www.fged.org/projects/minseqe (accessed 25 October 2026).
- ISO 15189:2022. https://www.iso.org/standard/76677.html (accessed 25 October 2026).
- Regulation (EU) 2017/746. https://eur-lex.europa.eu/eli/reg/2017/746/oj (accessed 25 October 2026).
- College of American Pathologists, accreditation programmes. https://www.cap.org/laboratory-improvement/accreditation (accessed 25 October 2026).

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`04_single-cell.md`](04_single-cell.md)
- Next topic: [`06_proteomics-and-metabolomics.md`](06_proteomics-and-metabolomics.md)
- Related topic: [`03_transcriptomics.md`](03_transcriptomics.md)
- Survey file: [`../../12_omics_precision_medicine.md`](../../12_omics_precision_medicine.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 12.4, spatial deconvolution, in the survey file: [`../../12_omics_precision_medicine.md`](../../12_omics_precision_medicine.md)
