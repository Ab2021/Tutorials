# Single-Cell Transcriptomics: Capture, QC, Clustering, and Annotation

> Module 12 — Omics and Precision Medicine
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: 01A, 04, topics 01 and 03
> Version 2026.3

## 1. Why this topic exists

Bulk transcriptomics answers "what is the average expression in this tissue?" Single-cell
transcriptomics answers "which cells express it?" That difference is not incremental. It
changes which questions are answerable: whether a disease activates a pathway inside a cell or
recruits more cells of a type, whether a drug response is uniform or confined to a subpopulation,
and whether a bulk signature is a composition change wearing a molecular costume.

The industry pull is strong because the questions are commercially relevant. Target validation
needs to know which cell type expresses the target. Safety assessment needs to know whether a
toxic effect is cell-type specific. Clinical trial enrichment needs a cell state that predicts
response. All three require per-cell attribution, and bulk data cannot provide it.

The technical reality is harder than the marketing. A typical droplet experiment captures
thousands of cells at low depth per cell, with ambient RNA from lysed cells, doublets from co-
encapsulation, and batch effects from separate runs and donors. The resulting matrix is about
95% zeros, and the zeros mean different things: a gene may be absent, or expressed below the
detection limit, or lost to dropout. Clustering algorithms will always return clusters,
including on data with no biological structure, so the existence of a cluster is not evidence
that a cell type exists.

The most consequential error is inferential, not computational. Cells from one donor are not
independent replicates of a condition. A study with 20,000 cells from 4 donors has an effective
sample size of 4 for any donor-level comparison, no matter how small the p-values from a
cell-level test. Treating cells as replicates produces p-values around 10^-40 for effects that
are not reproducible across donors. This single mistake accounts for a large share of
single-cell results that fail to replicate, and it is invisible in the standard output.

A second design error is under-powering the donor level. Programs often budget for cells because
cells are what the vendor quotes, then discover at analysis time that the claim requires donors,
and donors cannot be added retrospectively. A study designed around 100,000 cells from 3 donors
is a case study; a study designed around 20,000 cells from 12 donors is an experiment. The cost
difference is modest and the inferential difference is decisive.

This topic covers the droplet single-cell pipeline from cell calling to annotated clusters and
donor-level inference. Spatial data are topic 05; the deconvolution of bulk data using
single-cell references is covered in topic 05 and in topic 03.

### 1.1 The failure pattern

1. Cells are filtered with thresholds copied from a tutorial (minimum 200 genes, maximum 20%
   mitochondrial reads).
2. Doublets are not detected, and a cluster of "activated" cells is actually two cells in one
   droplet.
3. Ambient RNA is not corrected, so every cluster expresses epithelial markers at low level.
4. Clusters are annotated by a single marker gene, without negative markers.
5. A rare cluster is interpreted as a novel cell state when it is one donor's contamination or
   a technical artifact.
6. Differential expression is run with cells as replicates, producing astronomically small
   p-values.
7. The result fails to replicate in a second cohort, and the failure is attributed to "batch."

Each step is a default in some widely used tutorial. That is what makes the pattern durable.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Target biology lead | A cell-type attribution that does not hold | Depends on annotation quality |
| Translational lead | A cell-state biomarker that does not replicate | Depends on donor-level design |
| Computational biologist | Weeks on a dataset with no identifiable signal | Inherited the design and the thresholds |
| Pathologist | A "novel cell type" that contradicts morphology | Not consulted on annotation |
| Clinical team | A cell-state claim with no validated assay | Cannot inspect clusters |
| Program | A target nomination built on a doublet | No doublet QC in the report |

### 1.3 The industry question

> Is this molecular signal a validated biomarker, a causal target, or an artifact of batch,
> ancestry, and tissue composition?

In single-cell data the artifact list grows: ambient RNA, doublets, dissociation stress, cell
cycle, donor identity, and capture batch. A cluster can be real and still be an artifact of
dissociation. A marker can be real and still be ambient. The analysis must exclude these
explanations one at a time, and the report must show that it did.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** how droplet barcoding and unique molecular identifiers convert molecules into a
   cell-by-gene count matrix, and what each step loses.
2. **Compute** expected doublet rates from loading and apply doublet detection with a stated
   rate.
3. **Set** QC thresholds from the data distribution rather than from a tutorial, and defend
   them.
4. **Distinguish** normalization, feature selection, and integration, and state what each can
   and cannot remove.
5. **Annotate** clusters using positive markers, negative markers, and reference mapping, with
   a confidence statement.
6. **Choose** between cell-level and pseudobulk differential expression based on the unit of
   inference.
7. **Detect** donor imbalance, ambient contamination, and dissociation artifacts from
   diagnostics.
8. **Design** a single-cell study with donors, batches, and hashing that supports the intended
   claim.

## 3. The core mental model

Single-cell data are a sparse sample of transcripts from a biased sample of cells. Every
analysis step is an attempt to recover structure from that sample without inventing it.

```text
  Tissue -> dissociation -> cell suspension (viability, stress response)
      |
      v
  Droplet capture: cell + barcode bead + lysis mix
      |
      +--> empty droplets (ambient RNA only)
      +--> singlets
      +--> doublets / multiplets
      v
  cDNA -> sequencing -> barcode + UMI counting
      |
      v
  Cell x gene UMI matrix (sparse, ~95% zeros)
      |
      +--> cell calling (empty vs occupied)
      +--> ambient RNA correction
      +--> QC filtering (depth, genes, mitochondrial fraction)
      +--> doublet removal
      v
  Normalized matrix
      |
      +--> feature selection (highly variable genes)
      +--> PCA -> neighborhood graph -> clustering
      +--> integration across donors/batches (optional, justified)
      v
  Clusters -> annotation (markers, reference, negative markers)
      |
      v
  Inference: composition, differential expression, trajectory, perturbation
      |
      +--> unit of inference: donor, not cell
```

### 3.1 Four quantities that are constantly confused

**A cluster is not a cell type.** A cluster is a set of cells that are transcriptionally similar
under a chosen representation, resolution, and algorithm. A cell type is a biological category
defined by lineage, function, and markers. The mapping between them is an inference, and it can
be wrong in both directions: one cell type can split into several clusters, and several types
can merge into one.

**A marker is not a definition.** A gene enriched in a cluster is a marker. Whether the cluster
is a T cell depends on multiple markers, negative markers, and often on reference data. A single
marker can be ambient, cell-cycle driven, or shared across types.

**A cell is not a replicate.** Cells from the same donor share genetics, environment, and
dissociation history. Their expression values are correlated. Any test that treats them as
independent overstates precision by a factor that grows with the number of cells per donor.

**A UMI count is not a molecule count.** UMIs collapse PCR duplicates, but they also collapse
multiple transcripts of the same gene in the same cell, and they are subject to capture
efficiency that varies by gene, by cell, and by run. A UMI count is a biased sample of
molecules, not a census.

### 3.2 The three artifact layers

Single-cell data carry three layers of artifact, and each requires a different tool.

| Layer | Source | Signature | Tool class |
|---|---|---|---|
| Ambient | Cell-free mRNA in the droplet | Low-level expression of markers from abundant cell types in every cluster | SoupX, CellBender, DecontX |
| Multiplets | Two cells in one droplet | Cluster with markers from two lineages; intermediate library size | scDblFinder, DoubletFinder, Scrublet |
| Technical batch | Run, lane, chemistry, donor | Cluster or PC associated with a run or donor | Harmony, scVI, BBKNN; design fixes |

None of these tools removes dissociation stress, which is biological-state distortion rather
than contamination. A dissociation-stress signature (immediate early genes, heat-shock genes)
must be handled by design — cold dissociation, transcription inhibition — or reported as a
limitation.

### 3.3 When integration helps and when it hurts

Integration aligns cells across batches or donors so that shared cell types cluster together.
It is necessary when the batch effect is larger than the biological effect. It is harmful when
the biological effect is the batch: integrating across conditions can remove the disease signal
along with the donor difference.

The decision rule is: integrate across donors within a condition if donor is the batch, but
keep condition as a covariate in the model and check that condition-separated populations
survive integration. If a population exists only in cases, integration should not erase it. A
diagnostic is to compare cluster composition before and after integration: a population that
disappears was either an artifact or the biology of interest, and the analyst must decide which
with evidence.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Droplet | Aqueous compartment containing a cell and a barcoded bead | Used loosely for any single-cell method | Count |
| Barcode | Short DNA sequence identifying a cell or droplet | Confused with UMI | 16 bp typical (10x) |
| UMI | Unique molecular identifier, one per captured molecule | Confused with read count | Count |
| Saturation | Fraction of molecules observed more than once | Confused with sequencing depth | Proportion |
| Cell calling | Distinguishing occupied droplets from empty ones | Assumed trivial | Algorithm-dependent |
| Empty droplet | Droplet with ambient RNA but no cell | Filtered by cell calling | Count |
| Ambient RNA | Cell-free transcripts in the suspension | Mistaken for expression | Proportion of counts |
| Soup | The ambient RNA pool | Used interchangeably with ambient | Concentration |
| Doublet | Two cells in one droplet | Confused with a transitional state | Rate per 1,000 cells |
| Homotypic doublet | Doublet of two cells of the same type | Hard to detect | Rate |
| Heterotypic doublet | Doublet of two different types | Detectable by mixed markers | Rate |
| Multiplet | Three or more cells in one droplet | Rarely modeled | Rate |
| Median genes per cell | Median detected features | Confused with depth | Count |
| Mitochondrial fraction | Fraction of UMIs from mitochondrial genes | Used as a viability proxy | Proportion |
| Ribosomal fraction | Fraction of UMIs from ribosomal genes | Ignored in some tissues | Proportion |
| Highly variable gene (HVG) | Gene selected for variance | Assumed to be biologically variable | Selection |
| PCA | Linear dimensionality reduction | Assumed to capture all structure | Components |
| kNN graph | Nearest-neighbor graph over cells | Confused with clustering | Graph |
| Leiden | Graph clustering algorithm | Confused with a biological partition | Algorithm |
| Resolution | Granularity parameter of clustering | Treated as a fixed constant | Dimensionless |
| Marker gene | Gene enriched in a cluster | Treated as a definition | Statistic |
| Annotation | Assignment of cluster identity | Confused with clustering | Label |
| Reference mapping | Projection of cells onto a reference atlas | Confused with annotation from scratch | Label transfer |
| Integration | Alignment across batches or donors | Assumed to remove only technical variation | Method |
| Harmony | Iterative clustering-based integration | Assumed to preserve all biology | Method |
| scVI | Variational autoencoder for single-cell data | Treated as a foundation model | Method |
| Pseudobulk | Sum of counts per sample and cell type | Confused with averaging | Aggregation |
| Unit of inference | The independent sampling unit for a test | Confused with the cell | Design property |
| Differential abundance | Change in cell-type proportions | Confused with differential expression | Composition |
| Pseudotime | Inferred ordering along a trajectory | Treated as real time | Ordinal |
| RNA velocity | Rate of splicing change | Treated as direction of differentiation | Ratio |
| Perturb-seq | Pooled perturbation with single-cell readout | Confused with a screen hit | Assay |
| Hashtag | Oligo tag used to pool samples | Confused with a barcode | Sample label |

Two definitional points deserve emphasis.

**Saturation and depth are different.** Adding reads increases the chance of observing a
molecule that was captured; it does not increase the number of molecules captured. A library at
low saturation benefits from more reads; a library at high saturation does not. Reporting reads
per cell without saturation is reporting effort, not information.

**Pseudobulk is an aggregation, not a transformation.** Summing UMIs across cells of the same
type within a donor produces a count vector that behaves like a bulk sample. It discards
within-donor heterogeneity, which is the point: the donor is the unit of inference, and the
within-donor variation is not independent replication.

## 5. The mechanics, step by step

### Step 1. State the unit of inference before anything else

**Input:** the claim you intend to make.

**Output:** a sentence naming the independent unit (donor, patient, organoid line) and the
comparison.

**Rationale:** "T cells from treated donors express more of gene X than T cells from control
donors" has 8 independent units if there are 8 donors. "Gene X is higher in treated T cells"
has an ambiguous unit and will usually be tested at the cell level, which is wrong. The unit
determines the design, the aggregation, and the test. Choosing it after seeing the data is how
the cell-as-replicate error happens.

### Step 2. Design donors, batches, and pooling

**Input:** the unit of inference and available specimens.

**Output:** a design table with donor, condition, capture batch, and lane.

**Rationale:** Donors must be balanced across conditions and batches. If all cases are processed
in week one and controls in week two, donor and batch are confounded and no integration method
can separate them. Pooling with hashtags reduces batch effects by putting samples in the same
droplet run, at the cost of additional chemistry and demultiplexing.

The minimum defensible design for a donor-level claim is at least 4 donors per condition,
ideally 6 to 10. With 3 donors, a single outlier donor can drive the result, and no test can
distinguish that from biology.

### Step 3. Cell calling

**Input:** raw barcode counts.

**Output:** a cell-by-gene matrix with an empty-droplet boundary.

**Rationale:** Cell calling separates droplets containing a cell from droplets containing only
ambient RNA. The boundary matters: a permissive call admits empty droplets with ambient counts;
a strict call discards low-RNA cells, which are often real and often the population of interest
(plasma cells, for example, have low RNA content). Check the barcode-rank plot and the
distribution of counts per barcode rather than accepting a default.

### Step 4. Ambient RNA correction

**Input:** the cell-by-gene matrix.

**Output:** a corrected matrix or a per-cell ambient estimate.

**Rationale:** Ambient RNA produces a low background of transcripts from abundant cell types in
every droplet. It is not random: the ambient profile reflects the suspension's composition. A
cluster of macrophages will show epithelial markers if epithelial cells lysed during
dissociation. Correction estimates the ambient profile from empty droplets and subtracts a
proportional contribution.

The diagnostic: plot marker expression across clusters. If a marker is present at 1-3% of its
maximal level in every cluster, ambient is present. Correction is not always desirable —
over-correction can remove real low-level expression — so report whether it was applied and how
much was removed.

### Step 5. QC filtering with data-driven thresholds

**Input:** cell-level metrics.

**Output:** a filtered cell set and a record of how many cells were removed for each reason.

**Rationale:** Thresholds should come from the distribution of the data and the biology of the
tissue. A fixed minimum of 200 genes removes low-RNA cells that may be real. A fixed maximum
mitochondrial fraction of 20% removes stressed or dying cells but also removes metabolically
active cells in some tissues. Plot each metric, identify the tails, and state the rule before
applying it.

Report the number removed at each step, in order, so that the final cell count is traceable.

### Step 6. Doublet detection

**Input:** filtered matrix.

**Output:** doublet scores and a removed set.

**Rationale:** Doublets form when two cells enter one droplet. The rate rises with loading: the
vendor-documented multiplet rate for a 10,000-cell target is approximately 7.6%, and lower for
lower loadings. Doublet detection simulates doublets from the observed data and scores cells by
their similarity to simulated doublets. Detection is imperfect, especially for homotypic
doublets, which are transcriptionally similar to singlets.

The rate estimate matters: a detector run without a rate prior will use a default that may not
match the loading.

### Step 7. Normalization

**Input:** filtered counts.

**Output:** normalized values for visualization and feature selection.

**Rationale:** Library size varies by an order of magnitude across cells, driven by capture
efficiency and cell size. Library-size normalization followed by log transformation is the
baseline. Pearson residuals or a negative binomial model are alternatives that better handle
the mean-variance relationship. Normalization is not a substitute for aggregation; it fixes
scale, not dependence.

### Step 8. Feature selection

**Input:** normalized matrix.

**Output:** a set of highly variable genes.

**Rationale:** Using all genes gives low-variance features equal weight and lets technical noise
dominate the graph. Selecting variable genes concentrates the representation on biology, but
the selection can be driven by cell cycle, mitochondrial content, or batch. Check HVGs against
those lists before proceeding.

### Step 9. Dimensionality reduction

**Input:** scaled HVG matrix.

**Output:** principal components and a neighborhood graph.

**Rationale:** PCA denoises and compresses. The number of components should be chosen by
variance explained and by the stability of downstream clusters, not by a default of 50. The
neighborhood graph converts distances into local structure, and its parameters (neighbors,
metric) affect clustering more than most users expect.

### Step 10. Integration, only if justified

**Input:** PCs and batch labels.

**Output:** integrated representation.

**Rationale:** Integration is a modelling decision with a cost. It can remove the biological
effect if condition and batch are entangled. Run the analysis with and without integration,
compare cluster composition and marker stability, and report the comparison. If integration
changes the conclusion, the conclusion is not robust.

### Step 11. Clustering

**Input:** graph.

**Output:** cluster labels at a chosen resolution.

**Rationale:** Graph clustering finds communities; the resolution parameter sets granularity.
There is no correct resolution. Report the resolution, test at least two values, and check that
the clusters of interest are stable across them. A cluster that exists only at one resolution is
a candidate, not a finding.

### Step 12. Annotation with positive and negative markers

**Input:** clusters and marker statistics.

**Output:** labels with evidence and confidence.

**Rationale:** Annotation requires positive markers (present), negative markers (absent), and
context (tissue, species, condition). A cluster expressing CD3D, CD3E, and CD2 but not MS4A1 or
LYZ is a T-cell cluster. A cluster expressing one marker and nothing else is unannotated, and
should be labelled as such rather than forced into a type.

Reference mapping against an atlas provides an independent check, but it inherits the atlas's
errors and its tissue coverage. Agreement between marker-based and reference-based annotation
is stronger evidence than either alone.

### Step 13. Differential expression at the right unit

**Input:** cluster labels, donor labels, counts.

**Output:** per-donor pseudobulk and a donor-level test.

**Rationale:** Sum counts per donor and cell type, then test across donors. With 8 donors, a
paired test has 8 units and honest standard errors. A cell-level test has thousands of units
and standard errors that shrink with cell number, which is not a property of the biology.

The choice between pseudobulk and a mixed model with a donor random effect is a matter of
sample size and balance; both respect the unit of inference. What must not happen is a
cell-level test reported as a donor-level claim.

### Step 14. Composition and trajectory, with skepticism

**Input:** cluster labels per donor.

**Output:** proportion estimates and, if justified, a trajectory.

**Rationale:** Composition is often the real signal: disease changes the proportion of a cell
type more than the expression within it. Test composition at the donor level, with a model that
respects the count nature of the proportions. Trajectory analysis assumes a continuous process;
if the clusters are discrete states, pseudotime imposes an ordering that may not exist.

### Step 15. Report the analysis with the artifact checks

**Input:** everything above.

**Output:** a report with QC counts, doublet rate, ambient correction status, integration
comparison, annotation evidence, and the unit of inference.

**Rationale:** The report must let a reader judge whether the clusters are cells and whether the
claim is donor-level. A methods paragraph that says "standard Seurat pipeline" does not.

### Branch points

**Branch A — a cluster is defined by one donor.** Check donor composition per cluster before
annotation. If one donor contributes more than 80% of a cluster, the cluster is either a
donor-specific biology (possible) or a technical artifact (more likely). Report it as
donor-restricted and do not generalize.

**Branch B — a cluster has high mitochondrial content.** It may be dying cells, or
metabolically active cells. Check whether the cluster expresses stress genes and whether it is
present across donors. If it is donor-specific and stress-high, it is likely degradation.

**Branch C — integration removes a condition-specific population.** Do not accept the
integrated result. Report both, and treat the population as a hypothesis requiring orthogonal
evidence.

**Branch D — cells are few per donor.** Pseudobulk is unstable. Use a mixed model or aggregate
to donor-type means, and report the limitation.

### Decision table: integration method

| Situation | Method | Alternative | Avoid |
|---|---|---|---|
| Several donors, same condition | Harmony or scVI | BBKNN | No integration if donor dominates |
| Case and control, donor nested | Integrate donors, keep condition as covariate | scVI with condition covariate | Integrating across condition |
| Many batches, large data | scVI | Harmony | PCA alone |
| Cross-species or cross-tissue | scVI or scArches | Reference mapping | Forced alignment |
| Small number of cells per batch | Harmony | None | scVI with too few cells |
| Technical replicates only | No integration | Regress run | Over-correction |

### Edge cases

**Low library complexity.** A cell with 300 UMIs has unstable expression for all but the most
abundant genes. Diagnostic: the low-depth tail of the UMI distribution. Consequence: unstable
clustering and false marker enrichment. Mitigation: filter with a tissue-specific floor, or use
a method that models depth.

**Ambient RNA.** Diagnostic: markers of abundant types present at low level everywhere.
Consequence: misannotation and false doublets. Mitigation: ambient correction; compare marker
levels before and after.

**Doublets.** Diagnostic: clusters with two lineage markers; intermediate UMI counts.
Consequence: a spurious transitional state. Mitigation: doublet detection with a loading-based
rate.

**Batch confounding.** Diagnostic: cluster composition by batch, and PC1 by batch.
Consequence: batch interpreted as biology. Mitigation: balanced design; integration with
condition retained.

**Zero inflation.** Diagnostic: zero fraction versus mean; compare to a negative binomial.
Consequence: overdispersion and unstable tests. Mitigation: depth-aware models; avoid
imputing zeros as expression.

**Missing annotation.** Diagnostic: a cluster with no specific markers. Consequence: forced
label or an uninterpretable "unknown." Mitigation: label as unannotated, report markers, and
avoid overclaiming.

**Sample swap.** Diagnostic: hashtag demultiplexing inconsistency, or genotype from expressed
variants. Consequence: donor misattribution. Mitigation: hashtag all samples; verify genotypes
where possible.

**Over-filtering.** Diagnostic: cell count drops sharply at a threshold, and a known population
disappears. Consequence: loss of the population of interest. Mitigation: sensitivity analysis
across thresholds.

**One donor dominating a cluster.** Diagnostic: donor composition table. Consequence:
donor-specific artifact generalized as biology. Mitigation: report per-donor composition;
require replication across donors.

**Cell cycle.** Diagnostic: clusters defined by cell-cycle genes; cycling cells separated from
their lineage. Consequence: misannotation and false DE. Mitigation: score and regress cell
cycle, or annotate cycling states explicitly.

**Dissociation stress.** Diagnostic: immediate early genes (FOS, JUN) high across all cells of
one sample. Consequence: stress signature mistaken for disease. Mitigation: consistent
dissociation protocol; report stress scores by sample.

**Nucleated red cells or platelets.** In blood datasets, red-cell and platelet transcripts
contaminate droplets and create clusters that look like novel progenitors. Diagnostic: high
haemoglobin or platelet factor expression in a cluster with low gene counts. Consequence: a
false haematopoietic progenitor. Mitigation: remove with a marker-based filter, and report it.

**Batch-specific cell types.** A population appears in one batch only. Diagnostic: cluster by
batch cross-tabulation. Consequence: a technical population interpreted as disease-specific.
Mitigation: require presence in at least two batches and two donors.

**Ambiguous parent population.** A cluster sits between two types because the markers are shared
or the cells are transitional. Diagnostic: intermediate marker levels and no unique markers.
Consequence: forced annotation. Mitigation: label as intermediate or unannotated, and test
whether it is a doublet before treating it as biology.

**Hashtag failure.** Hashtag demultiplexing assigns a sample identity to each droplet; a failed
hashtag leaves cells unassigned or misassigned. Diagnostic: the distribution of hashtag counts
is bimodal or has a large unassigned fraction. Consequence: donor labels wrong for a subset, and
donor-level tests contaminated. Mitigation: include a hashing control, report the unassigned
fraction, and exclude ambiguous droplets.

**Multiple testing across clusters and genes.** Testing every gene in every cluster multiplies
the family size. Diagnostic: the reported number of tests is the gene count, but the analysis
tested 9 clusters x 18,000 genes. Consequence: inflated discoveries. Mitigation: test within a
pre-specified cluster, or correct across the full family and report it.

### What the procedure produces

The output is an annotated object plus an analysis record: cells before and after each filter,
doublet rate and method, ambient correction status, integration comparison, cluster resolution,
annotation evidence, and the unit of inference with the donor-level test. The object without the
record is not reusable. A reusable object also names its schema version, because metadata
fields change.

## 6. Worked example

A study compares synovial tissue from 8 patients with rheumatoid arthritis before and after a
therapy. Each patient contributes one pre-treatment and one post-treatment sample, for 16
samples. Cells are pooled with hashtags and run in four batches.

### 6.1 Capture and QC arithmetic

The target is 10,000 cells per run. The vendor-documented multiplet rate at that loading is
approximately 7.6%, so the expected doublets are:

```text
10,000 x 0.076 = 760 cells
```

After cell calling, 9,412 cells are retained per run. After QC (minimum 500 genes, maximum 15%
mitochondrial), 8,730 remain. After doublet removal at the estimated rate, 8,067 remain.

```text
retention = 8,067 / 9,412 = 0.857
```

A retention of 86% is plausible for synovial tissue. A retention below 60% usually indicates a
problem with dissociation or chemistry, not with the cells.

### 6.2 Clustering and annotation

Clustering at resolution 0.8 yields 18 clusters. Marker review groups them into 9 labels:
T cells (4 clusters), NK cells, B cells, plasma cells, monocytes, and fibroblasts. One cluster
has no specific markers and is labelled unannotated.

The T-cell clusters split by CD4 and CD8 expression and by a cytotoxic program. The split is
stable at resolutions 0.6, 0.8, and 1.0, which is evidence that it is not a resolution artifact.

### 6.3 The donor imbalance diagnostic

One cluster, labelled "activated fibroblasts," contains 1,000 cells. The donor composition is:

| Donor | Cells in cluster | Cells total | Proportion |
|---|---:|---:|---:|
| D1 | 820 | 1,000 | 0.82 |
| D2 | 180 | 1,000 | 0.18 |

D1 contributes 82% of the cluster. A chi-square test of independence on the 2 x 2 table of
cluster membership by donor gives:

```text
expected D1 in cluster = 1,000 x 1,000 / 2,000 = 500
chi-square = 4 x (320^2 / 500) = 819.2
```

The cluster is overwhelmingly donor-driven. Two interpretations are possible: D1 has a genuine
donor-specific expansion, or the cluster is a technical artifact of D1's capture. The
discriminator is whether the cluster's markers are consistent across its cells and whether the
same population appears at lower frequency in other donors. Here, the cluster appears at 2-4%
in the other seven donors, so the population is real but its expansion in D1 is donor-specific.
The correct claim is "a fibroblast state present in all donors, expanded in one," not "therapy
induces this state."

### 6.4 Differential expression at two units

The question is whether a cytotoxic gene is higher in post-treatment CD8 T cells. Pseudobulk
counts are formed per donor and cell type. The eight paired log2 differences are:

```text
0.52, 0.41, 0.63, 0.28, 0.47, 0.35, 0.58, 0.44
```

The mean difference is 0.46 log2 units (1.38-fold), with a standard deviation of 0.116 and a
standard error of 0.041. The paired t statistic is:

```text
t = 0.46 / 0.041 = 11.2
```

An exact sign-flip permutation test over 2^8 = 256 assignments gives 2 assignments at least as
extreme:

```text
p = 2 / 256 = 0.0078
```

The reporting sentence: "The gene increased by 0.46 log2 units (1.38-fold) after treatment,
paired by donor (n = 8, exact sign-flip p = 0.0078; 18,000 genes tested, BH q = 0.041)."

Now compute the same comparison at the cell level. With 6,000 cells per condition, a mean
difference of 0.30 log2 units, and a within-condition SD of 1.2:

```text
SE = 1.2 x sqrt(2/6000) = 0.0219
z  = 0.30 / 0.0219 = 13.7
p  = 1.1 x 10^-42
```

The cell-level p-value is 39 orders of magnitude smaller, for a smaller effect, from the same
experiment. It is not a stronger result. It is a result about cells, and the claim is about
patients.

### 6.5 Diagnostic 1: the ambient signature

Every cluster shows low-level expression of epithelial markers, at 1-2% of the level seen in the
one epithelial cluster. The pattern is proportional to the number of epithelial cells in the
suspension, which is the signature of ambient RNA.

If uncorrected, the plasma-cell cluster appears to express EPCAM, and a downstream analysis
might conclude that plasma cells transdifferentiate. Ambient correction removes the signal, and
the epithelial markers fall below detection in the plasma-cell cluster. The correction changes a
biological conclusion, which is why it is not optional.

### 6.6 Diagnostic 2: the mitochondrial cluster

One cluster has a median mitochondrial fraction of 32% and expresses heat-shock genes. It is
present in all donors but at 3-11% of cells, and its proportion correlates with the time between
dissociation and capture.

This is a degradation cluster, not a cell type. Removing it changes the composition estimate for
the remaining populations by a few percent, but leaving it in and annotating it as "stressed
T cells" would create a false biological narrative. The diagnostic is the correlation with
processing time.

### 6.7 Vary one assumption: cells are treated as replicates

Suppose the analyst skips pseudobulk and runs a Wilcoxon test on all CD8 T cells. The p-value is
1.1 x 10^-42, and the effect is 0.30 log2 units. Reported as a donor-level finding, it claims
that the therapy changes the gene in patients. A replication cohort of 8 new donors shows a mean
difference of 0.04 log2 units with a 95% interval from -0.12 to 0.20.

What changed is not the biology but the unit. The cell-level test measured whether the gene
differs between the specific cells captured in the two arms of one experiment, which includes
donor-specific effects that do not generalize. The pseudobulk test measured whether the
difference is consistent across donors, which is the claim of interest. The p-value differs by
39 orders of magnitude; the effect size differs by a factor of 1.5; and the conclusion differs
in kind.

### 6.8 Composition arithmetic

The proportion of cytotoxic CD8 T cells among all CD8 T cells is computed per donor:

```text
pre:  0.18, 0.21, 0.16, 0.19, 0.22, 0.17, 0.20, 0.18
post: 0.24, 0.27, 0.22, 0.26, 0.29, 0.23, 0.27, 0.25
```

The mean proportions are 0.189 and 0.254, a difference of 0.065, or 6.5 percentage points. The
paired differences are:

```text
0.06, 0.06, 0.06, 0.07, 0.07, 0.06, 0.07, 0.07
```

The mean difference is 0.065 with a standard deviation of 0.00535 and a standard error of
0.00189. The paired t statistic is 34.4. This is a composition effect, not an expression effect:
the cells did not change what they express, the tissue contains more of them.

Reporting a composition change as an expression change is a category error that survives
review because both are reported as "increased." The correct sentence names the denominator:
"cytotoxic CD8 T cells rose from 18.9% to 25.4% of CD8 T cells (paired difference 6.5
percentage points, n = 8)."

### 6.9 What the analysis record says

> 16 samples from 8 donors, paired design, 4 capture batches, hashtag pooling. 9,412 cells per
> run called; 8,730 after QC (>= 500 genes, <= 15% mitochondrial); 8,067 after doublet removal
> at 7.6%. Ambient correction applied; marker background reduced from 1-2% to below 0.5%.
> 18 clusters at resolution 0.8, stable at 0.6 and 1.0; 9 labels plus one unannotated.
> Integration across donors with Harmony, condition retained as a covariate; composition
> compared before and after. Differential expression by pseudobulk per donor; 18,000 genes;
> BH correction. Unit of inference: donor.

That record is the difference between a dataset and an analysis.

## 7. Code, query, or calculation

The script implements doublet-rate estimation, QC summarization, a donor-imbalance chi-square,
pseudobulk aggregation, and an exact paired permutation test.

### 7.1 Doublet rate from a loading table

```python
VENDOR_RATES = {500: 0.004, 1000: 0.008, 2000: 0.016, 5000: 0.039, 10000: 0.076}

def rate_for(cells):
    keys = sorted(VENDOR_RATES)
    if cells in VENDOR_RATES:
        return VENDOR_RATES[cells]
    if cells < keys[0] or cells > keys[-1]:
        raise ValueError("outside the tabulated range")
    for lo, hi in zip(keys, keys[1:]):
        if lo < cells < hi:
            f = (cells - lo) / (hi - lo)
            return VENDOR_RATES[lo] + f * (VENDOR_RATES[hi] - VENDOR_RATES[lo])

def expected_doublets(cells):
    return cells * rate_for(cells)

for n in (1000, 6000, 10000):
    r = rate_for(n)
    print(f"{n} cells: rate {r*100:.1f}%, {expected_doublets(n):.0f} doublets")
```

Expected output:

```text
1000 cells: rate 0.8%, 8 doublets
6000 cells: rate 4.6%, 278 doublets
10000 cells: rate 7.6%, 760 doublets
```

The rate is not linear: it rises faster as loading increases, which is why interpolation between
tabulated points is preferable to a single-anchor formula. The same logic applies to any vendor
table: use the table, not a line through one point.

### 7.2 Donor imbalance chi-square

```python
def chi2_2x2(table):
    row = [sum(r) for r in table]
    col = [sum(table[i][j] for i in range(2)) for j in range(2)]
    n = sum(row)
    chi = 0.0
    expected = [[0.0, 0.0], [0.0, 0.0]]
    for i in range(2):
        for j in range(2):
            e = row[i] * col[j] / n
            expected[i][j] = e
            chi += (table[i][j] - e) ** 2 / e
    return chi, expected

table = [[820, 180], [180, 820]]
chi, exp = chi2_2x2(table)
print("expected:", exp)
print("chi-square:", round(chi, 2))
print("donor 1 share of cluster:", round(820 / 1000, 2))
```

Expected output:

```text
expected: [[500.0, 500.0], [500.0, 500.0]]
chi-square: 819.2
donor 1 share of cluster: 0.82
```

The chi-square is enormous because the table is balanced at 1,000 cells per donor. A cluster
with this donor composition is donor-restricted until proven otherwise.

### 7.3 Pseudobulk and paired permutation test

```python
import itertools
import math
import statistics

# log2 pseudobulk difference per donor, post minus pre
diffs = [0.52, 0.41, 0.63, 0.28, 0.47, 0.35, 0.58, 0.44]

mean_diff = statistics.mean(diffs)
sd = statistics.stdev(diffs)
se = sd / math.sqrt(len(diffs))
print("mean log2 difference:", round(mean_diff, 3))
print("SD:", round(sd, 3), "SE:", round(se, 3))
print("paired t:", round(mean_diff / se, 2))

extreme = 0
total = 0
for signs in itertools.product([1, -1], repeat=len(diffs)):
    m = sum(s * x for s, x in zip(signs, diffs)) / len(diffs)
    total += 1
    if abs(m) >= abs(mean_diff) - 1e-12:
        extreme += 1
print("sign-flip p:", extreme, "/", total, "=", round(extreme / total, 4))
print("fold change:", round(2 ** mean_diff, 2))
```

Expected output:

```text
mean log2 difference: 0.460
SD: 0.116 SE: 0.041
paired t: 11.20
sign-flip p: 2 / 256 = 0.0078
fold change: 1.38
```

The exact test is possible only because there are 8 donors. With 20 donors, 2^20 is a million
assignments and still tractable; with 100 donors, use a t-test or a Monte Carlo sample of
permutations.

### 7.4 Cell-level versus donor-level standard error

```python
import math

def cell_level_p(diff, sd, n_per_group):
    se = sd * math.sqrt(2 / n_per_group)
    z = diff / se
    # two-sided normal approximation
    p = math.erfc(abs(z) / math.sqrt(2))
    return se, z, p

for n in (600, 6000, 60000):
    se, z, p = cell_level_p(0.30, 1.2, n)
    print(f"n={n}: SE={se:.4f} z={z:.1f} p={p:.2e}")
```

Expected output:

```text
n=600: SE=0.0693 z=4.3 p=1.49e-05
n=6000: SE=0.0219 z=13.7 p=1.12e-42
n=60000: SE=0.0069 z=43.3 p=0.00e+00
```

The p-value falls by 37 orders of magnitude as cells are added, with no change in the biology.
That is the arithmetic of pseudoreplication: the standard error scales with the square root of
the cell count, and the cell count is a technical choice.

### 7.5 Assumptions and limits of this code

- The doublet model is a linear approximation and is used to show the unit error, not as a
  production calibration.
- The chi-square assumes independent cells, which is false within a donor; it is a diagnostic
  for donor concentration, not a formal test of composition.
- The permutation test assumes exchangeability of paired differences under the null, which is
  reasonable for a paired design but not for an unpaired one.
- The cell-level function ignores donor correlation and is included only to demonstrate the
  error it produces.
- No code here performs clustering, integration, or annotation; those require established
  packages, and hand-rolled versions are not reproducible.

## 8. Data, standards, and regulatory context

### 8.1 Metadata standards for single-cell data

Single-cell metadata is less standardized than bulk metadata, and the gap is a practical
barrier to reuse.

- **Human Cell Atlas metadata schema** defines required fields for donor, tissue, assay, and
  processing, and is the basis for HCA Data Portal submissions.
- **CELLxGENE schema** (CZ CELLxGENE) defines a machine-readable schema for cell-level
  metadata, including ontology terms for cell type, tissue, and disease. It is the most
  widely adopted schema for reusable single-cell datasets.
- **MINSEQE** applies to the sequencing layer: reads, processed data, sample-to-data table,
  protocols.
- **AnnData/`.h5ad`** and **Matrix Market** are the common containers; neither enforces
  metadata content.
- **Cell Ontology (CL)**, **UBERON**, **MONDO** provide terms for cell type, tissue, and
  disease.

A dataset that labels clusters with free-text names ("T cells activated") rather than ontology
terms is not machine-reusable. The practical cost is that meta-analysis requires manual
relabelling, which reintroduces the variability the schema was meant to remove.

### 8.2 The HCA and atlas landscape

The Human Cell Atlas Data Portal reported, in its 2026 snapshot, roughly 70.9 million cells,
532 projects, 11.3 thousand donors, and about 1.0 thousand laboratories. The numbers move
monthly; they are a point-in-time indication of scale, not a fixed specification.

Atlas-based reference mapping has become the default annotation aid. Its limitation is
coverage: an atlas of blood does not annotate a synovial biopsy well, and a label transferred
from an atlas carries the atlas's tissue and ancestry biases.

### 8.3 Analytical validation for clinical single-cell tests

Clinical single-cell tests are rare but not absent (for example, minimal residual disease
assays and some transplant monitoring). When a single-cell assay is used clinically:

- **CLIA** requires performance specifications: accuracy, precision, analytical sensitivity and
  specificity, reportable range, reference intervals.
- **CAP** adds proficiency testing and document control.
- **ISO 15189:2022** provides the international quality framework.
- **IVDR 2017/746** applies to devices placed on the EU market.

The bioinformatics pipeline is part of the test. A cluster annotation that varies between runs
is a precision failure, and it must be measured, not assumed.

### 8.4 What the standards do not settle

- The resolution at which to cluster.
- Whether to integrate, and with which method.
- The minimum number of donors for a claim.
- How to annotate a cluster with no specific markers.
- How to report ambient correction.

These are scientific choices, and they are where analyses diverge. A methods section that names
the software but not the choices is not reproducible.

## 9. Industry practice and case evidence

### 9.1 What the atlases established

The Human Cell Atlas and tissue-specific atlases established that cell-type reference maps are
feasible at scale, and that metadata standardization is the binding constraint on reuse. The
2026 portal snapshot of roughly 70.9 million cells across 532 projects is the visible result.
The practical lesson for a program is that a new dataset should be submitted with schema-
compliant metadata, or it will not be comparable to the atlas.

### 9.2 What pharma does

Documented practice:

- Target validation groups use single-cell data to identify which cell type expresses a target,
  and cross-check against bulk and spatial data.
- Safety groups use single-cell data to detect cell-type-specific toxicity signatures.
- Biomarker groups use pseudobulk and donor-level tests, because regulators and clinicians ask
  about patients, not cells.
- Most companies run internal pipelines with pinned versions, because a re-run must reproduce
  the same clusters.

Inference: the most common internal failure is a target nomination based on a cluster that is
donor-restricted or doublet-driven. The check that would have caught it — donor composition per
cluster — is cheap and often skipped.

A second inference concerns cost. A discovery study of 16 samples at 10,000 cells each generates
about 160,000 cells, which is enough for clustering and composition but not for rare-population
claims in every donor. Programs that need a rare subset typically enrich it by sorting or by
targeted capture before sequencing, because the alternative is sequencing an order of magnitude
more cells to recover a few hundred events.

### 9.3 The foundation-model evidence

Single-cell foundation models (Geneformer, scGPT) were presented as general-purpose
representations learned from large corpora. A 2025 benchmark in *Genome Biology* reported that
zero-shot scGPT and Geneformer underperformed simpler methods such as scVI and Harmony on
cell-type clustering and batch integration, and that larger pretraining did not automatically
improve performance.

The lesson is not that foundation models are useless. It is that a benchmark gain on one task
does not transfer to another, and a model marketed as general requires task-specific
validation. A program adopting one should run it against a simple baseline on its own data
before building a pipeline around it.

> **Note:** benchmark results are task- and dataset-specific. The 2025 finding concerns
> zero-shot clustering and integration; it does not establish that the models are useless for
> perturbation prediction or annotation transfer.

### 9.4 The reproducibility problem in practice

Single-cell analyses are sensitive to analyst choices: QC thresholds, HVG count, PC count,
resolution, and integration method. Studies that ask multiple analysts to analyse the same
dataset report substantial variation in cluster number and marker lists. The implication is that
the analysis plan, the parameters, and the code must be fixed before the data are explored, and
the report should include the sensitivity of the main claim to the parameters that matter.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Cells as replicates | p-values below 10^-20 | Pseudoreplication | Non-replicable claim | Pseudobulk or mixed model |
| Doublets untreated | Cluster with two lineages | Co-encapsulation | Spurious state | Detect at loading-based rate |
| Ambient RNA untreated | Markers everywhere at low level | Cell-free transcripts | Misannotation | Ambient correction |
| Donor-restricted cluster | One donor >80% of cluster | Donor biology or artifact | Overgeneralized claim | Report donor composition |
| Batch confounded | All cases in one run | Design | Batch as biology | Balance design |
| Over-filtering | Cluster disappears at threshold | Strict QC | Lost population | Sensitivity analysis |
| Under-filtering | High mito cluster annotated | Dying cells retained | False cell type | Data-driven thresholds |
| Single-marker annotation | Label from one gene | Marker not specific | Wrong identity | Positive and negative markers |
| Resolution artifact | Cluster only at one resolution | Granularity | Over-split or merged types | Test multiple resolutions |
| Integration removes biology | Condition population vanishes | Over-correction | False negative | Compare integrated and not |
| Cell-cycle confounding | Clusters by phase genes | Proliferation | Misannotation | Score and regress, or annotate |
| Dissociation stress | Immediate early genes high in one sample | Processing | Stress as disease | Consistent protocol; report |
| Sample swap | Hashtag mismatch | Labeling | Misattributed donor | Hashtag; genotype check |
| Zero imputation | Imputed values analyzed as counts | Imputation | Inflated signal | Avoid imputing for tests |
| Trajectory overreach | Pseudotime on discrete states | Assumed continuity | False lineage | Test for discrete structure |

Three red flags:

**If a cluster's cells come mostly from one donor, stop.** It may be real, but it is not yet
generalizable.

**If the p-value depends on the number of cells rather than the number of donors, stop.** You
are testing cells, not patients.

**If every cluster expresses a marker of the most abundant cell type, stop.** That is ambient
RNA.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| A droplet contains one cell | Singlet attribution | Doublets, empty droplets | Doublet detection; barcode rank |
| UMIs count molecules proportionally | Quantification | Capture bias, saturation differences | Saturation curves; spike-ins |
| Ambient RNA is estimable | Correction | Ambient profile differs from empty droplets | Marker background check |
| Clusters correspond to types | Annotation | Continuous states, resolution effects | Multi-resolution stability |
| Donors are exchangeable within condition | Donor-level tests | Unmeasured confounders | Balance table; sensitivity |
| Integration removes only technical variation | Cross-donor comparison | Condition entangled with donor | Compare with and without |
| QC removes only low-quality cells | Data quality | Population-specific RNA content | Threshold sensitivity |
| Markers are specific | Annotation | Ambient, cell cycle, shared programs | Negative markers; reference |
| Dissociation does not alter state | Biological interpretation | Stress response | Stress scores by sample |
| Pseudobulk represents the population | Aggregation | Rare populations diluted | Minimum cells per donor-type |

The first and last assumptions are the ones that most often fail silently. A doublet looks like
a cell, and a pseudobulk of 12 cells looks like a sample.

## 12. Exercises

**Exercise 1 — Capture arithmetic (calculation, 20 minutes).**
A run targets 6,000 cells. Using a vendor-style multiplet table, estimate the doublet count at
4%, and compute the number of singlets after removing doublets. State one reason the estimate
is uncertain.
*Deliverable:* arithmetic and one sentence of uncertainty.
*Criteria:* doublet count correct; singlets computed after removal; uncertainty is specific.
*Time:* 20 minutes.

**Exercise 2 — QC thresholds (application, 40 minutes).**
You are given distributions of genes per cell and mitochondrial fraction for a muscle biopsy.
Propose thresholds, justify each with the distribution, and state which population might be
lost.
*Deliverable:* a threshold memo with two figures described.
*Criteria:* thresholds data-driven; a population at risk named; sensitivity plan stated.
*Time:* 40 minutes.

**Exercise 3 — Pseudoreplication (calculation, 30 minutes).**
A study reports a cell-level p-value of 10^-30 with 12,000 cells from 6 donors. Explain why the
p-value is not evidence about donors, compute the approximate donor-level standard error if the
between-donor SD is 0.4 log2 units, and state what test you would run.
*Deliverable:* one paragraph with arithmetic.
*Criteria:* pseudoreplication named; donor-level SE computed; test named and justified.
*Time:* 30 minutes.

**Exercise 4 — Annotation critique (critique, 45 minutes).**
A cluster is labelled "exhausted T cells" on the basis of high PDCD1 alone. Write a critique
naming the missing evidence and the alternative explanations.
*Deliverable:* 300-word critique.
*Criteria:* at least three alternative explanations; positive and negative markers requested;
reference mapping proposed.
*Time:* 45 minutes.

**Exercise 5 — Donor imbalance (application, 35 minutes).**
A cluster of 800 cells has 700 from donor 3, out of 4,000 total cells from that donor and 12,000
cells overall. Compute the expected count under independence and the chi-square contribution.
State the claim you would permit.
*Deliverable:* arithmetic and a one-sentence claim.
*Criteria:* expected count and contribution correct; claim restricted appropriately.
*Time:* 35 minutes.

**Exercise 6 — Integration decision (design, 45 minutes).**
Design an analysis for 10 donors, 5 per condition, run in 6 batches. State whether to integrate,
which method, how condition is retained, and the diagnostic that would show over-correction.
*Deliverable:* a decision memo.
*Criteria:* integration decision justified; method named; diagnostic specified.
*Time:* 45 minutes.

**Exercise 7 — Ambient RNA (application, 30 minutes).**
Every cluster expresses INS at 0.5-2% of the beta-cell level, and beta cells are 15% of the
suspension. Explain the likely cause, the correction, and the check that correction worked.
*Deliverable:* a 200-word explanation.
*Criteria:* ambient mechanism correct; correction named; before/after check stated.
*Time:* 30 minutes.

**Exercise 8 — Study design (design, 60 minutes).**
Design a single-cell study to test whether a drug changes the proportion of a rare immune
subset. Specify donors, batches, cells per sample, the unit of inference, and the primary
analysis.
*Deliverable:* a two-page design.
*Criteria:* at least 6 donors per arm; power logic for composition; unit of inference explicit;
primary analysis pre-specified.
*Time:* 60 minutes.

## 13. Self-check answers

**Answer 1 (capture).** At 6,000 cells and a 4% multiplet rate, expected doublets are
6,000 x 0.04 = 240. Singlets after removal: 6,000 - 240 = 5,760. The estimate is uncertain
because the rate depends on loading, cell size, and the vendor's calibration, and because
homotypic doublets are under-detected, so the true residual doublet count may be higher than
zero even after removal.

**Answer 2 (QC).** Thresholds should be placed at the tails of the observed distributions: for
example, minimum 500 genes if the distribution has a low-mode below 300, and maximum 20%
mitochondrial if the main population sits below 10% with a tail above 30%. The population at
risk is low-RNA cells such as plasma cells, which may fall below a 500-gene floor. The
sensitivity plan is to re-run with a lower floor and check whether a known marker population
returns.

**Answer 3 (pseudoreplication).** The cell-level p-value treats 12,000 cells as independent,
but cells within a donor are correlated, so the effective sample size is 6. The donor-level
standard error for a paired or two-group comparison with between-donor SD 0.4 and 6 donors is
approximately 0.4 x sqrt(2/6) = 0.23 log2 units for an unpaired design, or 0.4/sqrt(6) = 0.16
for a paired design. The test is a paired t-test or a sign-flip permutation on donor-level
summaries, not a cell-level Wilcoxon test.

**Answer 4 (annotation).** PDCD1 is expressed in activated T cells, not only exhausted ones, and
it can be ambient from a neighbouring population. Missing evidence: negative markers for
exhaustion (for example, absence of stemness markers), functional evidence (cytokine
production, proliferation), and a reference mapping that separates activated from exhausted
states. Alternatives: activation, recent TCR stimulation, ambient contamination, and a doublet
of T cells with another type. The label should be "PDCD1-high T cells, activation versus
exhaustion unresolved" until the evidence exists.

**Answer 5 (donor imbalance).** Total cells 12,000; cluster 800; donor 3 has 4,000 cells.
Expected donor-3 cells in the cluster under independence = 800 x 4,000/12,000 = 266.7. The
chi-square contribution is (700 - 266.7)^2/266.7 = 704. The claim permitted is "a cluster
dominated by donor 3," not "a disease-associated population," unless the same cluster appears
in other donors at lower frequency and the markers are consistent.

**Answer 6 (integration).** Integrate across donors, because donor is the batch and each donor
contributes both conditions only if the design is paired; if donors are nested in condition,
integrate within condition or use condition as a covariate and do not integrate across it.
Method: Harmony for speed, scVI if the dataset is large and GPU is available. Retain condition
as a covariate in the model. Diagnostic for over-correction: a population present only in one
condition disappears after integration, or cluster composition by condition flattens toward
the null.

**Answer 7 (ambient).** Beta cells are 15% of the suspension, so lysis releases insulin mRNA
into the ambient pool, and every droplet captures some. The correction estimates the ambient
profile from empty droplets and subtracts a proportional contribution per cell. The check is
that INS in non-beta clusters falls below the detection threshold while INS in beta cells is
unchanged, and that the total counts removed are reported as a percentage.

**Answer 8 (design).** At least 6 donors per arm, ideally 8, with each donor contributing one
sample and batches balanced across arms. The power calculation is for the proportion, not for
gene expression: with a baseline proportion of 0.5% and a target increase to 1.0%, the variance
of a donor-level proportion is dominated by the binomial component, so more cells per donor
reduce measurement error while more donors reduce biological variance. Both matter, and the
donor count is the one that cannot be fixed after sequencing. Cells per sample: enough that the rare subset has at
least 30 cells per donor, which may require 10,000 to 20,000 cells per sample if the subset is
0.5%. Unit of inference: donor. Primary analysis: proportion of the rare subset among parent
cells, computed per donor, tested with a two-group test on donor proportions or a beta-binomial
model. Secondary: pseudobulk expression within the subset, with BH correction.

## 14. Interview questions

**1. Why is a cell not a replicate?**
Strong answer: cells from a donor are correlated through genetics and processing; the
independent unit is the donor; cell-level tests shrink standard errors with cell number and
produce non-replicable p-values.
Weak answer: "because cells are noisy."

**2. How do you decide QC thresholds?**
Strong answer: from the distribution and the tissue's biology, pre-specified, with a sensitivity
analysis and a named population at risk.
Weak answer: "200 genes and 20% mitochondrial, as in the tutorial."

**3. What is ambient RNA and how do you detect it?**
Strong answer: cell-free transcripts captured in droplets; detected as low-level markers of
abundant types across all clusters; corrected with empty-droplet profiles.
Weak answer: "background noise" with no detection method.

**4. When does integration hurt?**
Strong answer: when condition is entangled with donor or batch, so integration removes the
biology; diagnose by comparing composition before and after.
Weak answer: "integration is always good."

**5. How do you annotate a cluster?**
Strong answer: positive markers, negative markers, cluster stability across resolutions, and
reference mapping; label unannotated clusters as unannotated.
Weak answer: one marker, one label.

**6. What is a doublet and how do you handle it?**
Strong answer: two cells in one droplet; rate rises with loading; detect with simulation-based
methods at a stated rate; homotypic doublets are hard.
Weak answer: "filter cells with too many genes."

**7. What is pseudobulk and why use it?**
Strong answer: sum counts per donor and cell type; test across donors; respects the unit of
inference; loses within-donor heterogeneity by design.
Weak answer: "averaging the cells."

**8. How do you test differential abundance?**
Strong answer: at the donor level, with a model for proportions (beta-binomial or a
quasi-binomial), accounting for total cells per donor.
Weak answer: chi-square on pooled cells.

**9. What does a Leiden resolution mean?**
Strong answer: a granularity parameter with no correct value; stability across resolutions is
the evidence, not the default.
Weak answer: "higher resolution means more accurate."

**10. What would make you distrust a novel cell state?**
Strong answer: donor restriction, doublet markers, ambient contribution, cell-cycle or stress
programs, and absence from reference atlases.
Weak answer: "if it is rare."

**11. How would you validate a single-cell finding?**
Strong answer: replicate in independent donors, confirm with spatial or sorted-population data,
and test the composition or expression claim at the donor level.
Weak answer: "check the markers."

**12. What did the 2025 foundation-model benchmark show?**
Strong answer: zero-shot scGPT and Geneformer underperformed simpler baselines on clustering
and integration, and scale did not guarantee gains; task-specific validation is required.
Weak answer: "foundation models are better" or "foundation models are useless."

## 15. Key terms

- **Ambient RNA** — cell-free transcripts captured in a droplet.
- **Annotation** — assignment of cluster identity from evidence.
- **Barcode** — DNA sequence identifying a droplet or cell.
- **Batch** — group of samples processed together.
- **Cell calling** — separating occupied from empty droplets.
- **Cell Ontology** — controlled vocabulary for cell types.
- **CELLxGENE** — schema and portal for reusable single-cell data.
- **Differential abundance** — change in cell-type proportions.
- **Doublet** — two cells in one droplet.
- **Droplet** — compartment containing a cell and a barcoded bead.
- **Harmony** — clustering-based integration method.
- **HCA** — Human Cell Atlas.
- **Homotypic doublet** — doublet of two cells of the same type.
- **HVG** — highly variable gene.
- **Integration** — alignment of cells across batches or donors.
- **kNN graph** — nearest-neighbor graph over cells.
- **Leiden** — graph clustering algorithm.
- **Marker gene** — gene enriched in a cluster.
- **Multiplet** — three or more cells in one droplet.
- **Multiplet rate** — expected fraction of droplets with more than one cell.
- **PCA** — principal component analysis.
- **Pseudobulk** — per-sample, per-cell-type count aggregation.
- **Pseudotime** — inferred ordering along a trajectory.
- **Reference mapping** — projection onto an atlas.
- **Resolution** — clustering granularity parameter.
- **RNA velocity** — spliced/unspliced ratio as a direction estimate.
- **Saturation** — fraction of molecules observed more than once.
- **scVI** — variational model for single-cell data.
- **Soup** — the ambient RNA pool.
- **Singlet** — a droplet containing exactly one cell.
- **UMI** — unique molecular identifier.
- **Unit of inference** — independent sampling unit for a test.
- **Zero inflation** — excess zeros relative to a count model.

## 16. Sources

Methods and benchmarks:

- Wolf, F. A., Angerer, P., Theis, F. J., "SCANPY: large-scale single-cell gene expression data analysis," *Genome Biology* 19:15, 2018. https://doi.org/10.1186/s13059-017-1382-0 (accessed 25 October 2026).
- Hao, Y. et al., "Integrated analysis of multimodal single-cell data," *Cell* 184:3573-3587, 2021. https://doi.org/10.1016/j.cell.2021.04.048 (accessed 25 October 2026).
- Korsunsky, I. et al., "Fast, sensitive and accurate integration of single-cell data with Harmony," *Nature Methods* 16:1289-1296, 2019. https://doi.org/10.1038/s41592-019-0619-0 (accessed 25 October 2026).
- Lopez, R. et al., "Deep generative modeling for single-cell transcriptomics," *Nature Methods* 15:1053-1058, 2018. https://doi.org/10.1038/s41592-018-0229-2 (accessed 25 October 2026).
- Xi, N. M., Li, J. J., "Benchmarking computational doublet-detection methods for single-cell RNA sequencing data," *Cell Systems* 12:176-194, 2021. https://doi.org/10.1016/j.cels.2020.11.008 (accessed 25 October 2026).
- Theodoris, C. V. et al., "Transfer learning enables predictions in network biology," *Nature* 618:616-624, 2023. https://doi.org/10.1038/s41586-023-06139-9 (accessed 25 October 2026).
- Cui, H. et al., "scGPT: toward building a foundation model for single-cell multi-omics using generative AI," *Nature Methods* 21:1470-1480, 2024. https://doi.org/10.1038/s41592-024-02201-0 (accessed 25 October 2026).
- "Benchmarking foundation cell models," *Genome Biology*, 2025. https://doi.org/10.1186/s13059-025-03574-x (accessed 25 October 2026).

Data and standards:

- Human Cell Atlas Data Portal. https://data.humancellatlas.org/ (accessed 25 October 2026).
- CELLxGENE schema. https://github.com/chanzuckerberg/single-cell-curation (accessed 25 October 2026).
- Cell Ontology. https://obofoundry.org/ontology/cl.html (accessed 25 October 2026).
- FGED Society, MINSEQE. https://www.fged.org/projects/minseqe (accessed 25 October 2026).
- 10x Genomics, Xenium panels and instrument specifications. https://www.10xgenomics.com/platforms/xenium/panels (accessed 25 October 2026).

Regulatory:

- ISO 15189:2022. https://www.iso.org/standard/76677.html (accessed 25 October 2026).
- Regulation (EU) 2017/746. https://eur-lex.europa.eu/eli/reg/2017/746/oj (accessed 25 October 2026).

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`03_transcriptomics.md`](03_transcriptomics.md)
- Next topic: [`05_spatial-transcriptomics.md`](05_spatial-transcriptomics.md)
- Related topic: [`07_multi-omics-integration.md`](07_multi-omics-integration.md)
- Survey file: [`../../12_omics_precision_medicine.md`](../../course/12_omics_precision_medicine.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 12.3, single-cell clustering, in the survey file: [`../../12_omics_precision_medicine.md`](../../course/12_omics_precision_medicine.md)
