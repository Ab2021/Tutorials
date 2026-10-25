# Omics Data Types: What Each Assay Actually Measures

> Module 12 — Omics and Precision Medicine
> Topic 01 of 10
> Estimated study time: 3 hours
> Prerequisites: 01A, 04
> Version 2026.3

## 1. Why this topic exists

Most omics failures are not statistical failures. They are measurement failures that were
already baked in when someone chose an assay, a specimen type, and a processing pipeline, and
then reported the output as if it were the biology. A team that sequences the wrong analyte,
at the wrong resolution, from the wrong specimen, will produce a clean-looking matrix and a
confident conclusion that cannot be reproduced by anyone else.

The immediate industry problem is assay selection under constraints. A translational lead has
120 banked biopsies, half of them formalin-fixed paraffin-embedded (FFPE) and five years old.
She has budget for one discovery assay and one validation assay. The question is whether a
transcriptomic signature predicts response to an anti-TNF therapy. Four vendors will each say
their platform is appropriate. The differences between those platforms — input mass, RNA
integrity tolerance, dynamic range, whether the output is a count or a proportion, whether
the analyte survives fixation — determine whether the study can answer the question at all.
None of those differences appear in the vendor's headline performance claim.

The second problem is inherited data. A data scientist joins a project and is handed a matrix
of 18,432 features by 240 samples, plus a one-line description: "RNA-seq, normalized." The
matrix could be raw counts, counts after size-factor scaling, log-transformed counts,
transcripts per million (TPM), fragments per kilobase million (FPKM), or a batch-corrected
residual matrix. Each of those requires a different downstream model. Applying a negative
binomial differential-expression model to TPM produces p-values that are wrong in a way that
is invisible in the output table. Applying log-transform twice produces effect sizes that are
off by a constant factor. The most expensive errors in omics are scale errors, and scale is
rarely documented.

The third problem is metadata. A matrix without sample annotation is close to useless for any
question involving confounding. Public repositories contain thousands of datasets where the
disease label is present but the batch, the tissue site, the library preparation kit, the
sequencing centre, and the donor identifier are missing or inconsistent. Reanalysis of such
data can still be publishable, but it cannot support a regulatory claim, and it usually cannot
support a clinical one.

This topic exists to make the measurement layer explicit before any modelling begins. It
covers what each major omics assay physically measures, what its output data structure is,
what it systematically loses, and how to decide which assay fits a decision. It is the
foundation for the rest of the module: variant interpretation (topic 02), differential
expression (topic 03), single-cell analysis (topic 04), spatial deconvolution (topic 05),
proteomics and metabolomics (topic 06), and integration (topic 07) all assume that you can
read a data structure and know what produced it.

### 1.1 The failure pattern

The same sequence recurs across academia, biotech, and pharma.

1. A biologist describes a question in molecular terms ("is interferon signalling active in
   these lesions?").
2. A platform is chosen by availability, habit, or a vendor relationship rather than by the
   properties of the analyte.
3. The specimen is fixed, degraded, or too small for the chosen assay, so the protocol is
   modified ad hoc. The modification is not recorded.
4. The output matrix is normalized by a method copied from a paper with a different design.
5. Differential analysis is run on all features at once, with no filter and no batch term.
6. A list of hits is reported with p-values but no effect sizes and no confidence intervals.
7. A validation attempt on an orthogonal platform fails, and the failure is attributed to
   "platform differences" rather than to the original measurement problem.
8. The program loses six to eighteen months and the assay is repeated.

Steps 2 and 3 are the expensive ones, and they are the ones a data scientist is usually
invited to review after they have already happened.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Translational lead | A trial arm that cannot be stratified, and a season of enrolment | Chose the assay before the data team was involved |
| Computational biologist | Months of analysis on a matrix that cannot support the claim | Does not control sample collection or library prep |
| Lab operations | Reagent cost and instrument time for a repeat study | Receives specifications, not the biological question |
| Biobank curator | The scientific value of a consented collection | Inherited a consent scope that limits assay type |
| Clinician | A test result whose limits are not stated | Sees the report, not the validation file |
| Regulator | A submission that cannot be assessed for analytical validity | Receives a summary, not raw data or metadata |
| Patient | A biopsy consumed for an assay that answered a different question | Has no role in assay selection |

The last row matters most: a tissue block is finite, and a library made from the last block
cannot be unmade.

### 1.3 The industry question

The module's framing question applies directly here:

> Is this molecular signal a validated biomarker, a causal target, or an artifact of batch,
> ancestry, and tissue composition?

Before that question can be asked, a prior one must be answered: what does the number in the
matrix actually measure? A TPM value is a proportion of a sequenced library, not a molecule
count. A label-free proteomics intensity is an ion current, not a protein concentration. A
16S operational taxonomic unit (OTU) table is a set of relative abundances, not a census. Each
of these can be useful. None of them is the thing the biologist named.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** a stated biological or clinical question to the analyte that must be measured, and
   justify the choice in writing.
2. **Distinguish** the six output data structures used in omics — raw counts, scaled counts,
   within-sample proportions, log-ratios, continuous intensities, and compositional
   proportions — and state which models each supports.
3. **Compute** library complexity and saturation diagnostics from a count matrix, including
   library size, detected-feature count, and duplication rate.
4. **Audit** a dataset's metadata against the Minimum Information about a high-throughput
   Nucleotide SeQuencing Experiment (MINSEQE) elements and identify which missing fields
   block which analyses.
5. **Critique** an assay choice for a specimen type, including FFPE compatibility, input mass,
   and RNA integrity number (RIN) tolerance.
6. **Calculate** the sample size needed for a stated effect size and dispersion, and the
   multiple-testing burden implied by the feature count.
7. **Identify** the eight canonical data pathologies — low library complexity, zero inflation,
   compositionality, probe dropout, ambient RNA, doublets, batch confounding, and missing
   annotation — from summary statistics alone.
8. **Write** a two-page assay selection memo with a decision table, a cost estimate, and an
   explicit list of what the chosen assay cannot measure.

Each outcome maps to an exercise in section 12 or a self-check item in section 13.

## 3. The core mental model

Every omics assay is a lossy pipeline. The biology is converted into an analyte, the analyte
into a signal, the signal into a number, and the number into a matrix. At each arrow,
something is destroyed and something is added. The skill is knowing which.

```text
  Biological state
        |
        |  (1) sampling: which cells, which region, which time
        v
  Specimen  ------------------------------>  annotation, consent, storage
        |
        |  (2) extraction: lysis, fragmentation, selection
        v
  Analyte  (DNA fragments, mRNA, peptides, metabolites, microbial DNA)
        |
        |  (3) assay chemistry: library prep, labelling, ionization
        v
  Raw signal  (FASTQ reads, probe intensities, m/z spectra, colony counts)
        |
        |  (4) quantification: alignment, peak calling, spectral matching
        v
  Feature x sample matrix  ---------------->  normalization, batch correction
        |
        |  (5) statistical model
        v
  Claim  ("gene X is up in responders")
```

The pipeline is drawn left to right, but the decisions that constrain it run right to left.
If the claim requires an absolute concentration, the assay must have a calibration curve. If
the claim requires a per-cell attribution, the assay must have single-cell resolution. If the
claim requires a protein-level statement, an RNA measurement is a proxy, and the correlation
between the proxy and the target is an assumption that must be measured, not assumed.

### 3.1 The three-axis classification

Any omics assay can be placed on three axes. Most assay-selection arguments are really
disagreements about which axis matters most.

| Axis | Low end | High end | What it determines |
|---|---|---|---|
| Resolution | Bulk tissue | Single cell / subcellular | Whether composition is visible or averaged |
| Specificity | Targeted panel (10^2 features) | Untargeted (10^4-10^6 features) | Multiple-testing burden and interpretability |
| Quantitation | Relative, within-sample | Absolute, calibrated | Whether values are comparable across samples |

A bulk RNA-seq experiment sits at bulk resolution, untargeted specificity, and relative
quantitation. A clinical viral load assay sits at bulk resolution, targeted specificity, and
absolute quantitation. A spatial transcriptomics run sits at near-single-cell resolution,
targeted specificity (a panel of hundreds to thousands of genes), and relative quantitation.
These are different instruments for different jobs, and no amount of analysis converts one
into another.

### 3.2 The information-loss table

The most useful single habit in this topic is to ask, for any assay, "what is lost at each
stage?" The table below is the short version.

| Stage | What is destroyed | What is added |
|---|---|---|
| Sampling | Spatial context, temporal dynamics, rare cells | Selection bias from the sampling frame |
| Extraction | Labile molecules, long transcripts, protein complexes | Lysis bias, fragmentation bias |
| Chemistry | Unselected molecules, absolute abundance | Duplication, amplification bias, batch |
| Quantification | Reads that map ambiguously, isoforms | Reference-genome dependence |
| Normalization | Absolute scale | Assumption of equal total output |

The right-hand column is the part learners forget. Every assay adds structure that was not in
the biology: guanine-cytosine (GC) bias, 3' coverage bias, ion suppression, primer bias,
batch. A differential-expression hit can be a difference in any of these.

### 3.3 Resolution is not free

A single-cell assay resolves composition but pays for it in per-cell depth. A typical
10x Genomics 3' run produces 20,000 to 60,000 reads per cell at targeted recovery, which is
roughly one to two orders of magnitude less than a bulk sample's per-transcript coverage.
The consequence is dropout: a gene expressed at two copies per cell is detected in a minority
of cells. A bulk assay would measure that gene reliably but could not tell you which cells
expressed it. This trade is the single most important design decision in transcriptomics, and
it is a trade, not an improvement.

### 3.4 The matrix is not the data

A count matrix is a summary produced by a pipeline version. Annotation versions change, gene
identifiers are retired and merged, and pipeline defaults change. Two matrices for the same
samples produced eighteen months apart can differ in feature count by thousands. Any claim
that depends on a feature set must pin the annotation, pipeline, and reference build.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Genome | The complete DNA sequence of an organism | Used to mean "exome" | Base pairs (bp); GRCh38 for human |
| Exome | The protein-coding fraction of the genome, about 1-2% | Treated as "all functional variation" | ~30-33 Mb targeted territory |
| Whole-genome sequencing (WGS) | Sequencing of the whole genome, including non-coding | Assumed to detect all structural variants | 30x typical germline depth |
| Whole-exome sequencing (WES) | Capture-based sequencing of coding regions | Compared directly to WGS without coverage caveat | 100x typical mean depth |
| Read | A short sequence produced by an instrument | Used interchangeably with "fragment" | 50-300 bp typical |
| Coverage / depth | Number of reads covering a position | Confused with "number of reads per sample" | x (fold) for genomes; reads for RNA |
| Library complexity | Number of distinct molecules in a library | Confused with library size in reads | Estimated distinct fragments |
| Duplication rate | Fraction of reads that are PCR or optical duplicates | Assumed to indicate bad library only | Proportion (0-1) |
| Unique molecular identifier (UMI) | Random barcode attached to each molecule before amplification | Assumed to remove all bias | Count of distinct UMIs |
| Raw count | Number of reads or UMIs assigned to a feature | Called "expression" without qualification | Integer count |
| Counts per million (CPM) | Counts scaled by library size | Used for differential expression directly | Counts per 10^6 reads |
| Transcripts per million (TPM) | Within-sample proportion after length normalization | Used as input to count-based models | Parts per 10^6 |
| Fragments per kilobase million (FPKM) | Read-pair version of RPKM, length-normalized | Mixed with TPM | Parts per 10^6 (not sum-consistent) |
| Log-ratio | Log of a ratio between two quantities | Taken of zero counts without a pseudocount | log2 or natural log |
| Intensity | Instrument signal proportional to analyte amount | Treated as concentration | Arbitrary units (a.u.) |
| Label-free quantification (LFQ) | MS quantification without isotopic labelling | Treated as absolute abundance | a.u. or normalized intensity |
| Isobaric labelling | TMT/iTRAQ multiplexing by mass tag | Assumed to be ratio-free | Reporter ion intensity |
| Compositional data | Data where only relative proportions are identified | Analyzed with Euclidean distance | Proportions summing to 1 |
| Centered log-ratio (CLR) | Log-ratio transform for compositional data | Applied without zero replacement | Dimensionless |
| Feature | A measured unit: gene, transcript, protein, metabolite, taxon | Used without naming the identifier space | Ensembl, UniProt, HMDB, NCBI taxon |
| Sample | A biological unit that is measured once | Confused with "library" or "run" | Donor or specimen identifier |
| Batch | A group of samples processed together | Ignored when it coincides with group | Categorical processing factor |
| Confounding | A variable associated with both exposure and outcome | Used loosely for any correlation | Design property |
| Metadata | Structured description of samples and processing | Treated as optional | MINSEQE/MIAME fields |
| FASTQ | Text format for reads and base qualities | Assumed to be analysis-ready | Phred+33 quality |
| BAM | Binary alignment format | Edited by hand | SAM/BAM specification |
| Variant call format (VCF) | Text format for variants | Assumed to contain only true variants | VCF 4.4/4.5 |
| Count matrix | Features x samples table of integers | Assumed to be normalized | Matrix Market, HDF5, TSV |
| AnnData | HDF5 container for single-cell data | Assumed to be self-documenting | `.h5ad` |
| SummarizedExperiment | Bioconductor container linking assays and metadata | Bypassed by exporting CSV | R/Bioconductor |
| Ambient RNA | Cell-free transcripts captured in a droplet | Mistaken for low-level expression | Proportion of counts |
| Doublet | Two cells captured in one droplet or well | Mistaken for a novel cell state | Estimated rate per 1,000 cells |
| Probe dropout | Failure of a targeted probe to hybridize or amplify | Mistaken for absent expression | Per-feature detection rate |
| Zero inflation | Excess zeros relative to a count model | Assumed to be biological absence | Proportion of zeros |

Two definitional points deserve emphasis because they cause most of the downstream errors.

First, **normalization is a claim about what is constant**. CPM assumes total output per
sample is proportional to total RNA. Median-of-ratios assumes most genes are not changing.
CLR assumes the geometric mean of the composition is a stable reference. Each assumption can
be false in a specific design, and each failure produces a specific artifact rather than a
random error.

Second, **a feature is defined by an annotation, not by biology**. "Gene X is upregulated"
means "reads assigned to the annotation interval for X increased under pipeline version P
with annotation version A." Change any of those and the statement changes. This is why
reproducibility requires pinned versions, and why a matrix without a pipeline record is a
secondary source at best.

## 5. The mechanics, step by step

This section is a procedure for selecting an assay and then reading its output honestly. It
is written for the person who has to defend the choice in a protocol review and again in a
results meeting six months later. Each step states its input, its output, and why it exists.
Branch points are marked, and a decision table covers the multi-way choices.

### Step 1. State the decision, not the assay

**Input:** a clinical or program question in plain language.

**Output:** one sentence of the form "we will decide X for population Y using a measurement
of Z with an effect size of at least E."

**Rationale:** Without a decision and a minimum effect size, no assay can be judged
appropriate. "Profile the immune response" is not a decision. "Assign patients to an
anti-interferon arm if their lesion interferon signature score exceeds a threshold that
separates responders with an area under the curve (AUC) of at least 0.70" is a decision, and
it implies a minimum detectable effect, a required precision, and a validation path.

A useful test: if the assay returned a null result, would anyone change a decision? If the
answer is no, the study is descriptive and should be labelled as such.

### Step 2. Identify the analyte that carries the decision

**Input:** the decision sentence from step 1.

**Output:** the analyte class — DNA variant, DNA copy number, DNA methylation, chromatin
accessibility, RNA abundance, protein abundance, metabolite abundance, microbial composition.

**Rationale:** The analyte is not always the obvious one. A decision about whether a tumour
will respond to a checkpoint inhibitor may be carried by protein expression of a ligand, by
mismatch-repair deficiency detectable in DNA, by a mutational signature in DNA, or by an RNA
expression signature. These are not interchangeable measurements, and their concordance is
modest.

A second test: does the decision depend on a mechanism that is transcriptional, or on a
mechanism that is post-transcriptional? If the latter, an RNA measurement is a proxy with an
unmeasured error term.

### Step 3. Check the specimen against the assay's input requirements

**Input:** specimen type, age, fixation, storage, available mass or volume.

**Output:** a pass/fail per candidate assay, with the limiting constraint named.

**Rationale:** This is where most studies fail before sequencing. FFPE cross-links nucleic
acids and fragments RNA; a poly-A selection protocol that requires intact mRNA will produce a
3' bias or fail outright. Fresh-frozen tissue is not always available, and a retrospective
cohort is almost always FFPE. Proteomics from FFPE is possible but the yield and peptide
modification profile differ from fresh tissue. Metabolomics from FFPE is generally not
interpretable because metabolites are lost or altered in fixation.

Three constraints to record explicitly:

- Input mass or cell number, with the assay's minimum and the observed value.
- Integrity metric (RIN for RNA, DIN for DNA), with the assay's tolerance.
- Fixation status and time in fixative, which affects fragmentation more than age does.

### Step 4. Decide the resolution the claim requires

**Input:** the decision sentence and the known tissue heterogeneity.

**Output:** bulk, sorted-population, single-cell, or spatial resolution.

**Rationale:** Bulk tissue averages cell types. If a disease changes cell composition — and
most diseases do — a bulk difference can be entirely compositional. A gene expressed only in
infiltrating T cells will appear "upregulated" in bulk tissue when T-cell infiltration rises,
even if its per-cell expression is unchanged. Whether that matters depends on the claim. For
a diagnostic signature, composition may be a perfectly good signal. For a mechanistic claim
about a cell-intrinsic pathway, it is a confounder.

### Step 5. Decide the specificity and the multiple-testing burden

**Input:** prior knowledge about the pathway or candidate set.

**Output:** targeted panel, broad untargeted assay, or a two-stage design.

**Rationale:** The number of features tested sets the correction burden. Testing 380 genes
at a family-wise error rate of 0.05 requires a per-test threshold of 1.3 x 10^-4. Testing
20,000 genes requires 2.5 x 10^-6. At the same true effect size and sample size, the broad
assay may have no power after correction while the panel does. This is not an argument
against untargeted discovery; it is an argument for using untargeted discovery to generate
hypotheses and a targeted assay to test them.

### Step 6. Decide whether relative or absolute quantitation is required

**Input:** the claim's unit. Is it "higher in cases than controls" or "above 15 pg/mL"?

**Output:** relative assay with a comparator, or calibrated assay with a standard curve.

**Rationale:** Most omics assays are relative. Sequencing returns proportions of a library.
Mass spectrometry returns ion currents that depend on ionization efficiency. Absolute
quantification requires either spiked internal standards at known concentrations, a
calibration curve, or a validated clinical platform. A relative measurement can support a
classifier trained on the same assay; it cannot support a clinical threshold in physical
units without a calibration step.

### Step 7. Read the output data structure before modelling

**Input:** the delivered matrix and its documentation.

**Output:** a classification of the matrix into one of six types, with evidence.

**Rationale:** The model must match the data-generating process. Negative binomial models
assume counts with a mean-variance relationship. Linear models on log data assume approximate
homoscedasticity after transformation. Compositional methods assume the data carry only
relative information. Applying the wrong model produces p-values that are not merely
conservative or liberal; they can be anticonservative in one direction for one set of genes
and conservative for another, which makes the error list look plausible.

Six checks that identify the type in practice:

1. Are the values integers? If yes, likely raw or rounded counts.
2. Do columns sum to a constant (10^6, or 1)? If yes, within-sample proportions.
3. Is the maximum value bounded near 10^6 and the distribution heavily skewed with many
   zeros? Consistent with TPM or CPM.
4. Are there negative values? Then it is a log-ratio, a residual, or a centred transform.
5. Is the value range roughly 0-30 with decimals? Consistent with log2(x+1) of normalized
   counts.
6. Are values continuous with a long right tail spanning four or more orders of magnitude,
   with no integer structure? Consistent with intensity data.

### Step 8. Audit the metadata against MINSEQE

**Input:** sample sheet, protocol records, submission metadata.

**Output:** a gap list, each gap mapped to the analyses it blocks.

**Rationale:** MINSEQE specifies five elements: the biological system and experimental
variables; the sequence read data per assay; the final processed data; general experiment
information with sample-to-data relationship tables; and the experimental and data-processing
protocols. A dataset missing element five cannot be reprocessed. A dataset missing the
sample-to-data relationship table cannot be checked for swaps. A dataset missing element one
cannot be adjusted for confounding, because the confounder is not recorded.

### Step 9. Check feature annotation provenance

**Input:** the feature identifiers and the pipeline record.

**Output:** the annotation build, the identifier space, and the mapping loss.

**Rationale:** Gene identifiers are not stable. Ensembl retired the GRCh37 archive; HGNC
symbols are updated monthly and symbols are reused. A matrix keyed on gene symbols from 2016
may silently map two old symbols to one current gene, or map a retired symbol to nothing.
Quantify the mapping loss: of 20,000 features, how many map one-to-one to the current
annotation? A loss above 5% changes pathway enrichment results.

### Step 10. Estimate power and the false-discovery burden

**Input:** expected effect size, dispersion or standard deviation, sample size, feature count.

**Output:** power at the planned sample size, and the number of expected false positives at
an uncorrected threshold.

**Rationale:** A study with 20 samples per arm and a 20,000-feature assay has a predictable
false-positive count: at alpha 0.05, about 1,000 features will pass by chance. If the observed
number of hits is near 1,000, the study has found nothing, and this is knowable before the
data arrive. Computing it in advance prevents the post hoc narrative that "the signal is
subtle."

### Step 11. Define the orthogonal validation assay before generating data

**Input:** the discovery assay and the candidate features.

**Output:** a validation plan naming platform, specimen, sample size, and concordance metric.

**Rationale:** Validation on the same platform with the same protocol tests reproducibility,
not validity. A claim that survives only on one platform is a platform-specific classifier.
Orthogonal validation — quantitative PCR for RNA, immunohistochemistry or targeted mass
spectrometry for protein, a different chemistry for metabolites — tests whether the signal
is in the specimen rather than in the assay.

### Step 12. Write the limits paragraph

**Input:** everything above.

**Output:** a short, specific statement of what the assay cannot measure and which claims it
cannot support.

**Rationale:** Every assay has a claim ceiling. A bulk RNA-seq study cannot support a
cell-intrinsic mechanism claim without deconvolution or single-cell data. A 16S study cannot
support a functional claim about gene content. A label-free proteomics study cannot support
an absolute abundance claim. Writing the ceiling down at design time prevents the ceiling
from being discovered by a reviewer.

### Branch points

Three branch points change the shape of the procedure.

**Branch A — specimen is FFPE and the analyte is RNA.** Move to a 3'-biased or targeted
chemistry, reduce the expected detectable feature count, and increase sample size to
compensate for the loss of dynamic range. If the claim requires full-length isoform
quantification, the assay cannot be run on this specimen, and the design must change.

**Branch B — the assay is targeted and the candidate set is fixed.** Multiple-testing burden
falls by two orders of magnitude, power rises, and the discovery claim becomes a
confirmation claim. The cost is that no novel feature can be found. This is usually the right
trade for a validation study and the wrong trade for a discovery study.

**Branch C — the tissue is heterogeneous and the claim is cell-intrinsic.** Add a
composition measurement: deconvolution from bulk, sorted populations, single-cell, or spatial.
Without it, the cell-intrinsic claim is not identified.

### Decision table: analyte and claim to assay class

| Analyte / claim | Candidate assay | Output structure | Cannot support |
|---|---|---|---|
| Germline coding variant | WES | VCF genotypes | Non-coding regulatory variant claims |
| Somatic mutation and copy number | Tumour-normal WGS or panel | VCF + copy ratio | Clonal architecture without deep sequencing |
| RNA abundance, tissue-level | Bulk poly-A RNA-seq | Raw counts | Cell-type-specific claims |
| RNA abundance, cell-level | scRNA-seq | UMI counts per cell | Absolute abundance; rare transcript detection |
| RNA with location | Spatial capture or in situ | Spot or cell counts | Whole-transcriptome coverage at single-cell level |
| Chromatin accessibility | ATAC-seq | Fragment counts | Transcription factor binding without footprint analysis |
| DNA methylation, array | Infinium array | Beta values (0-1) | Single-cell methylation; repeat regions |
| DNA methylation, sequencing | Bisulfite or EM-seq | Methylation calls per CpG | Allele-specific methylation without reads |
| Protein abundance, discovery | Data-dependent acquisition MS | Intensity, identification score | Absolute concentration; low-abundance cytokines |
| Protein abundance, targeted | Immunoassay or PRM/MS | Calibrated concentration | Novel protein discovery |
| Metabolite abundance | LC-MS untargeted | m/z feature intensity | Identification beyond confidence level 2 without standards |
| Microbial composition | 16S amplicon | Relative abundance table | Functional gene content; absolute load |
| Microbial function | Shotgun metagenomics | Taxonomic and functional profiles | Strain-level claims without assembly |

### Edge cases

**Low library complexity.** A library with few distinct molecules produces duplicate reads
that look like high expression. Diagnostic: duplication rate above 50% combined with a
detected-feature count far below the cohort median. Consequence: count models
underestimate dispersion, and p-values are too small. Mitigation: exclude or model complexity
as a covariate.

**Batch confounding.** Batch is perfectly associated with the group of interest. This happens
when all cases were processed in one week and all controls in another. Diagnostic: cross-
tabulation of batch by group shows an empty cell. Consequence: the batch effect and the
biological effect are not separable; any correction removes one with the other. Mitigation:
none post hoc. Redesign, or restrict the claim to "differences between processing groups."

**Sample swaps.** Two samples exchange labels. Diagnostic: genotype concordance between
sample and its recorded donor, or sex-check from XIST and RPS4Y1 expression. Consequence: one
sample's data are attributed to another patient, and any per-patient decision is wrong.
Mitigation: geno-check every batch; record a fingerprint in the sample sheet.

**Zero inflation.** A count matrix has more zeros than a negative binomial predicts.
Diagnostic: fraction of zeros versus mean count per feature; compare to simulated negative
binomial. Consequence: overdispersion estimates inflate and power falls; or, if zeros are
treated as missing, imputation invents signal. Mitigation: distinguish biological zeros
(undetected because absent) from technical zeros (undetected because of depth), and use a
model that does so — zero-inflated negative binomial, or a detection-probability model.

**Compositionality.** The data are proportions, so an increase in one component forces a
decrease in others. Diagnostic: a small set of features shows coordinated opposite changes
that sum to a constant. Consequence: correlation between features is negative by
construction, and "co-regulated" gene sets appear. Mitigation: log-ratio transforms, or
reference-based normalization.

**Ambient RNA.** Cell-free transcripts from lysed cells contaminate droplets. Diagnostic: a
cell type's marker genes appear at low level in every cluster, in proportion to the
abundance of that cell type in the suspension. Consequence: clusters appear to express
markers they do not express, and doublet detectors underperform. Mitigation: ambient RNA
correction tools, or exclusion of markers with a high ambient fraction.

**Doublets.** Two cells in one droplet. Diagnostic: a cluster with markers from two disjoint
lineages and an intermediate library size; or a bimodal distribution of detected genes.
Consequence: a spurious "transitional" or "novel" cell state. Mitigation: doublet detection
before clustering, with a rate estimate based on loading.

**Probe dropout.** A targeted panel fails to detect a transcript that is present. Diagnostic:
a feature with a detection rate far below its expected expression level across samples, or a
3' bias in coverage. Consequence: false negatives concentrated in specific features, which
biases pathway analysis. Mitigation: per-feature detection QC; exclude features with
unstable detection; do not interpret absence as biological absence.

**Missing annotation.** Sample sheet lacks donor, batch, or tissue site. Diagnostic: the
metadata table has fewer columns than the design requires, or values are inconsistent in
format. Consequence: confounding cannot be assessed, and any batch correction is guesswork.
Mitigation: recover from the original records if possible; otherwise restrict claims and
state the limitation.

**Annotation version drift.** The same feature identifier means different intervals in two
annotation builds. Diagnostic: feature counts differ between matrices that should be
identical. Consequence: meta-analysis joins the wrong features. Mitigation: pin the build and
map identifiers explicitly with a versioned table.

**Pseudo-replication.** Multiple libraries from the same donor are treated as independent
samples. Diagnostic: sample count exceeds donor count, and donor is not in the model.
Consequence: standard errors are too small, sometimes by a factor of the square root of the
replicate number. Mitigation: mixed model with a donor random effect, or aggregate to donor.

### Claim ceilings by assay class

The limits paragraph in step 12 is easier to write if you keep a claim-ceiling table. The
table below is not exhaustive; it covers the claims that most often exceed their assay in
review.

| Assay class | Supports | Does not support |
|---|---|---|
| Bulk RNA-seq | Association between tissue-level expression and outcome | Cell-intrinsic mechanism; absolute abundance |
| Single-cell RNA-seq | Cell-type-resolved expression; composition | Absolute concentration; full transcriptome per cell |
| Spatial capture | Expression with tissue coordinates at spot resolution | Single-cell attribution without deconvolution |
| Targeted panel | Confirmation within a fixed feature set | Novel feature discovery |
| 16S amplicon | Relative taxonomic composition | Absolute load; functional gene content |
| Untargeted metabolomics | Feature-level association with a spectral signature | Named-metabolite identification without standards |
| Label-free proteomics | Relative protein abundance | Absolute concentration; low-abundance cytokines |

### The output of the procedure

The procedure ends with a one-page assay decision record containing: the decision sentence,
the analyte, the specimen constraint, the resolution, the specificity, the quantitation type,
the matrix type, the metadata gaps, the annotation build, the power estimate, the validation
assay, and the limits paragraph. That record is the artifact a reviewer can audit. The
sequence data are not the deliverable; the record plus the data is.

## 6. Worked example

A translational immunology group is planning a 120-patient study of ulcerative colitis. Half
the patients will receive an anti-tumour necrosis factor (anti-TNF) biologic. The question is
whether a mucosal gene-expression signature predicts response at week 12. Biopsies are banked:
72 fresh-frozen, 48 FFPE, collected over six years. The group has budget for one discovery
assay.

### 6.1 Candidate assays and constraints

| Assay | Input | RIN tolerance | Features | Cost per sample (USD) | FFPE usable |
|---|---|---|---:|---:|---|
| Bulk poly-A RNA-seq | 200 ng total RNA | 7.0 | ~20,000 genes | 350 | No |
| Bulk ribo-depleted RNA-seq | 10 ng total RNA | 4.0 | ~25,000 genes | 450 | Partial |
| Targeted expression panel (3' chemistry) | 10 ng total RNA | 2.0 | 380 genes | 120 | Yes |
| Whole-transcriptome array | 100 ng total RNA | 6.0 | ~20,000 probes | 200 | Partial |
| Single-cell 3' RNA-seq | 10,000 live cells | 8.0 | ~20,000 genes | 2,500 | No |
| Spatial capture | one 6.5 mm section | 7.0 | ~18,000 genes | 3,000 | Yes |

Two constraints eliminate most options immediately. The FFPE biopsies have a median RIN of
3.1 (measured on 12 of 48; the rest are assumed similar). Poly-A selection requires intact
mRNA and will show severe 3' bias at that RIN. Single-cell requires viable cells, which
banked biopsies do not provide. What remains is ribo-depleted bulk (partial FFPE), the
targeted 3' panel (FFPE-compatible), the array (partial), and spatial (FFPE-compatible but
expensive and low throughput).

### 6.2 Power arithmetic

Assume the signature's leading gene has a true difference of log2 fold change (log2FC) = 0.8,
which is a 1.74-fold difference. Assume the standard deviation of log2 expression within
group is 1.2 log2 units, typical for inflamed mucosal tissue. For a two-sample comparison
with equal arms, the standard error is:

```text
SE = SD x sqrt(2/n)
n = 20 per arm: SE = 1.2 x sqrt(2/20) = 1.2 x 0.316 = 0.379
z    = 0.8 / 0.379 = 2.11
power (two-sided alpha 0.05, normal approximation) = 0.56
```

Doubling to 40 per arm gives:

```text
SE = 1.2 x sqrt(2/40) = 1.2 x 0.224 = 0.268
z    = 0.8 / 0.268 = 2.98
power = 0.85
```

So the planned 20 per arm has 56% power for the effect the program cares about. That is the
first number to put in the protocol, because it converts "we will look for a signature" into
"we have a 44% chance of missing the effect we care about."

### 6.3 Multiple-testing arithmetic

With the bulk assay, 19,842 genes pass the expression filter. At an uncorrected alpha of 0.05,
the expected false positives are:

```text
19,842 x 0.05 = 992 features
```

The Bonferroni threshold is:

```text
0.05 / 19,842 = 2.52 x 10^-6
```

The smallest observed p-value is 2.4 x 10^-7 for the leading interferon-response gene. It
passes Bonferroni. The 40th smallest p-value is 1.8 x 10^-5. The Benjamini-Hochberg (BH)
threshold at rank 40 is:

```text
40 x 0.05 / 19,842 = 1.008 x 10^-4
```

so it passes, and its q-value is:

```text
q = p x N / rank = 1.8 x 10^-5 x 19,842 / 40 = 8.9 x 10^-3
```

The reporting sentence is therefore: "1 gene passed Bonferroni correction (p = 2.4 x 10^-7,
19,842 tests); at q < 0.05 by BH, 63 genes were significant, with the leading gene at
q = 8.9 x 10^-3." Every element — test, number of tests, correction — is present. A p-value
without those three elements is not reportable.

### 6.4 Effect size and interval

The leading gene's log2FC is 0.83 with a standard error of 0.214 log2 units. The 95%
confidence interval is:

```text
0.83 +/- 1.96 x 0.214 = 0.83 +/- 0.42 = [0.41, 1.25]
```

In fold-change units that is 1.33-fold to 2.38-fold. The interval excludes 1.0, so the
direction is supported. But the lower bound, 1.33-fold, may be below the effect the program
would need for a clinically useful classifier. Significance and usefulness are different
questions, and the interval answers both.

### 6.5 Composition arithmetic

The bulk difference may be compositional. Suppose deconvolution estimates that the epithelial
fraction falls from 0.55 to 0.45 in responders and the T-cell fraction rises from 0.12 to
0.20. If the leading gene is T-cell restricted, its bulk fold change decomposes as:

```text
observed bulk fold change = per-cell fold change x abundance fold change
1.78 = 1.45 x 1.23
```

A 1.45-fold per-cell change and a 1.23-fold abundance change multiply to the observed 1.78.
Either alone would produce a smaller bulk signal. Without deconvolution, the program would
report a 1.78-fold "upregulation" and could not say whether the pathway was activated or the
tissue was more inflamed. The claim ceiling has moved: a bulk assay supports "the signature
differs," not "the pathway is activated per cell."

### 6.6 Diagnostic 1: the sample swap

Before analysis, a sex check is run using XIST and RPS4Y1 expression. Seven of 120 samples
(5.8%) have a genotype-predicted sex that disagrees with the recorded sex. The discordance is
concentrated in one processing batch (six of the seven), which suggests a plate-level
labeling error rather than seven independent mistakes. Two consequences follow. First, the
seven samples must be excluded or relabeled, and relabeling requires genotype matching, not
guessing. Second, the batch containing them is now suspect for other errors, and the batch
term must be retained in the model even after the fix.

If the swap had not been checked, one of the seven might have been the sample driving the
leading association. The published result would be a per-patient claim that is false for two
patients and unverifiable for the rest.

### 6.7 Diagnostic 2: the scale error

The pilot matrix was delivered as "normalized expression." A check of column sums shows they
are all 1,000,000. The values are TPM. The pilot analysis had applied a negative binomial
model directly to TPM, which treats them as counts. Re-running with raw counts and
median-of-ratios normalization changes the dispersion estimates and the hit list: 63 genes at
q < 0.05 by BH becomes 41 genes, and the leading gene's log2FC moves from 0.83 to 0.79.

The two numbers are close, which is the dangerous part. The error did not produce an absurd
result; it produced a plausible one that was 50% too large in its significant-gene count. A
reviewer comparing the two tables would not see an error, only a difference.

### 6.8 Vary one assumption: FFPE contamination of the discovery set

The plan is to include the 48 FFPE samples using ribo-depleted chemistry. Suppose the FFPE
samples are not randomly distributed: they are the earlier-recruited patients, who also have
longer disease duration. Excluding them changes the estimated effect:

```text
all samples (n = 120):      log2FC = 0.83, 95% CI [0.41, 1.25]
fresh-frozen only (n = 72): log2FC = 0.52, 95% CI [-0.06, 1.10]
```

The conclusion changes from "significant at q < 0.05" to "not significant, interval includes
no effect." The FFPE samples were not merely noisier; they carried a different biological
distribution, and the processing method was confounded with disease duration. The fix is not
a better normalization. The fix is to stratify the analysis by processing method, report both
estimates, and pre-specify which population the claim applies to.

This is the single most common pattern in retrospective omics: an apparent effect that is a
difference between specimen eras. It is detectable at design time by cross-tabulating
processing method against every clinical variable, which takes ten minutes and is almost
never done.

### 6.9 What the record says

The assay decision record for this study reads: decision — stratify anti-TNF assignment by a
mucosal interferon signature with AUC target 0.70; analyte — mRNA; specimen — 72 frozen, 48
FFPE with RIN 3.1; resolution — bulk, with deconvolution to adjust for composition;
specificity — untargeted discovery, 19,842 genes after filtering; quantitation — relative;
matrix — raw counts, median-of-ratios; metadata gaps — disease duration missing for 9,
processing batch recorded for all; annotation — Ensembl release pinned, 4.1% mapping loss;
power — 0.56 at 20 per arm for log2FC 0.8, 0.85 at 40; validation — targeted panel on an
independent cohort, with quantitative PCR for the top three genes; limits — cannot support a
per-cell mechanism claim, cannot support an absolute threshold, cannot separate processing
era from disease duration in the FFPE subset.

That paragraph is more valuable than the matrix. It states what the study can and cannot
conclude before anyone is invested in a result.

## 7. Code, query, or calculation

The script below uses only the Python standard library. It reads a small feature-by-sample
table, identifies the matrix type, computes library diagnostics, applies Benjamini-Hochberg
correction to a list of p-values, and bootstraps a confidence interval for a fold change.
Every block is executable as written.

### 7.1 Detect the matrix type and compute library diagnostics

```python
import math
import random

# A tiny counts matrix: 4 genes x 4 samples.
counts = {
    "IFIT1":  [412, 88, 390, 102],
    "CXCL10": [1204, 310, 1180, 288],
    "ACTB":   [9800, 10250, 9910, 10040],
    "MKI67":  [12, 4, 9, 3],
}
samples = ["S1", "S2", "S3", "S4"]

def is_integer_matrix(m):
    return all(float(v).is_integer() for row in m.values() for v in row)

def column_sums(m, cols):
    return [sum(m[g][j] for g in m) for j in range(len(cols))]

def detect_type(m, cols):
    vals = [v for row in m.values() for v in row]
    sums = column_sums(m, cols)
    if is_integer_matrix(m):
        return "raw_counts"
    if all(abs(s - 1_000_000) < 1 for s in sums):
        return "tpm_or_cpm"
    if all(abs(s - 1.0) < 1e-6 for s in sums):
        return "proportions"
    if any(v < 0 for v in vals):
        return "log_ratio_or_residual"
    return "continuous_intensity"

print("matrix type:", detect_type(counts, samples))
print("library sizes:", column_sums(counts, samples))

lib = column_sums(counts, samples)
detected = [sum(1 for g in counts if counts[g][j] > 0) for j in range(len(samples))]
print("features detected:", detected)

# CPM for one gene across samples
gene = "IFIT1"
cpm = [round(counts[gene][j] / lib[j] * 1_000_000, 1) for j in range(len(samples))]
print("IFIT1 CPM:", cpm)
```

Expected output:

```text
matrix type: raw_counts
library sizes: [11428, 10652, 11489, 10433]
features detected: [4, 4, 4, 4]
IFIT1 CPM: [36051.8, 8261.4, 33945.5, 9776.7]
```

The CPM values differ from the raw counts by a factor that is not constant across samples,
which is the point of the normalization.

### 7.2 Benjamini-Hochberg correction from first principles

```python
def bh(pvals):
    """Return (sorted index, p, q) with q = min over k>=i of p_k * m / k."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = pvals[i] * m / rank
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    return [(i, pvals[i], q[i]) for i in order]

pilot = [2.4e-7, 1.8e-5, 4.1e-4, 0.0021, 0.0094, 0.031, 0.048, 0.19]
res = bh(pilot)
print("m =", len(pilot))
for i, p, q in res:
    print(f"rank p={p:.3e}  q={q:.4e}")
print("significant at q<0.05:", sum(1 for _, _, q in res if q < 0.05))
```

Expected output:

```text
m = 8
rank p=2.400e-07  q=1.920e-06
rank p=1.800e-05  q=7.200e-05
rank p=4.100e-04  q=1.093e-03
rank p=2.100e-03  q=4.200e-03
rank p=9.400e-03  q=1.504e-02
rank p=3.100e-02  q=4.133e-02
rank p=4.800e-02  q=5.486e-02
rank p=1.900e-01  q=1.900e-01
significant at q<0.05: 6
```

Note the monotonicity step: the seventh p-value, 0.048, has a raw BH value of 0.0549, above
0.05, so it is not significant even though its p-value is below 0.05.

### 7.3 Bootstrap confidence interval for a fold change

```python
import random

random.seed(7)

# log2 expression for one gene, 12 responders and 12 non-responders
resp = [7.9, 8.4, 7.1, 8.8, 8.0, 7.6, 8.9, 8.2, 7.4, 8.6, 8.1, 7.8]
nonr = [7.2, 7.6, 6.9, 7.4, 7.0, 7.8, 7.1, 6.8, 7.3, 7.5, 6.7, 7.2]

def mean(x):
    return sum(x) / len(x)

observed = mean(resp) - mean(nonr)
print("observed log2FC:", round(observed, 3))

diffs = []
for _ in range(10000):
    a = [random.choice(resp) for _ in resp]
    b = [random.choice(nonr) for _ in nonr]
    diffs.append(mean(a) - mean(b))
diffs.sort()
lo, hi = diffs[250], diffs[9750]
print("95% percentile bootstrap CI:", round(lo, 3), round(hi, 3))
print("fold change:", round(2 ** observed, 2),
      "CI:", round(2 ** lo, 2), round(2 ** hi, 2))
```

Expected output (seed 7, so exactly reproducible):

```text
observed log2FC: 0.858
95% percentile bootstrap CI: 0.5 1.2
fold change: 1.81 CI: 1.41 2.3
```

The bootstrap interval is narrower than the analytic interval in section 6.4 because this
toy example has low variance. The method is: resample within group, recompute, take
percentiles. It assumes independent observations and enough samples for stable tails; with
6 per group, use a t-interval instead.

### 7.4 Assumptions and limits of this code

- The matrix is small and balanced; real matrices require streaming or array libraries.
- `detect_type` uses heuristics. Integer-valued TPM rounded to whole numbers would be
  misclassified as counts, which is why documentation beats detection.
- The BH implementation assumes independent or positively correlated tests. It is robust to
  positive dependence, not to arbitrary dependence.
- The bootstrap assumes exchangeability within group and ignores batch. With batch structure,
  resample within batch.
- No code here performs batch correction. Topic 03 covers that, because batch correction is
  a modelling decision rather than a formatting step.
- Proportions sum to 1 within a sample, so correlations on raw proportions are partly
  structural; use log-ratios for compositional data.

## 8. Data, standards, and regulatory context

Omics data standards divide into three layers: what must be reported, how the data are
encoded, and what a regulated laboratory must demonstrate. Confusing the layers produces
either over-compliance (treating a journal checklist as law) or under-compliance (treating an
accreditation requirement as optional).

### 8.1 Reporting standards

**MINSEQE** (Minimum Information about a high-throughput Nucleotide SeQuencing Experiment)
is a proposal first drafted after a Functional Genomics Data Society (FGED) workshop in 2008
and published as version 1.0 in 2012. It specifies five elements: the biological system and
experimental variables; the sequence reads for each assay; the final processed data; general
experiment information including sample-to-data relationship tables; and the experimental and
data-processing protocols. It is a reporting guideline, not a ratified ISO standard. Journals
and funders require it with varying strictness, and compliance is often partial in practice.

**MIAME** (Minimum Information About a Microarray Experiment) is the older sibling, with six
elements: raw data, final processed data, sample annotation, experimental design and
sample-data relationships, feature annotation, and protocols. Its structure informed MINSEQE.

Neither standard specifies a machine-readable schema with enforced validation. That is why
GEO, ArrayExpress, and SRA accept submissions that are technically compliant and practically
incomplete. A submitter can satisfy the checklist with free-text fields that no downstream
analysis can parse.

### 8.2 Encoding standards

| Layer | Standard | Constrains | Does not constrain |
|---|---|---|---|
| Reads | FASTQ (Phred+33) | Base qualities, record structure | Adapter content, read length |
| Alignment | SAM/BAM/CRAM | Alignment records, flags | Aligner choice, parameters |
| Variants | VCF 4.4 / 4.5 | Variant records, INFO/FORMAT semantics | Calling thresholds, filtering |
| Single-cell matrices | Matrix Market (`.mtx`), HDF5 | Sparse matrix layout | Normalization, feature annotation |
| Single-cell container | AnnData (`.h5ad`) | Array and metadata layout | Analysis provenance |
| R container | SummarizedExperiment | Assay-metadata linkage | Analysis provenance |
| Gene annotation | GFF3/GTF, Ensembl/GENCODE | Feature coordinates | Which release is current |
| Ontologies | EFO, MONDO, CL, UBERON | Term identity | Term granularity in practice |

The Variant Call Format is maintained in the hts-specs repository under the Global Alliance
for Genomics and Health (GA4GH) Large Scale Genomics work stream. Version 4.4 was drafted in
January 2023 and has been superseded by 4.5. The difference matters for structural variants:
4.4 introduced tandem-repeat representation and deprecated `SVTYPE` in favour of `SVTYPE`-free
symbolic alleles and `SVCLAIM`. A pipeline that writes 4.3 conventions into a 4.4 header is
technically malformed and may be silently misread.

### 8.3 Controlled vocabularies and identifiers

Identifiers are the join keys of omics. The main ones:

- **Genes:** HGNC symbols (unstable, reused), Ensembl gene IDs (versioned), Entrez Gene IDs.
- **Transcripts and proteins:** Ensembl transcript IDs, RefSeq accessions, UniProt accessions.
- **Metabolites:** HMDB, ChEBI, InChI, SMILES.
- **Diseases, cells, tissues:** MONDO, EFO, Orphanet, Cell Ontology, UBERON.

A dataset keyed only on gene symbols is not usable for meta-analysis without a mapping table
and a stated release. This is a hard requirement in practice, not a preference.

### 8.4 Repository and access requirements

- **GEO** (Gene Expression Omnibus): series, samples, platforms; requires MIAME/MINSEQE-style
  fields; supports controlled access for human data.
- **SRA/ENA**: raw reads; accession-based; metadata often thin.
- **ArrayExpress/BioStudies**: array and sequencing submissions with MAGE-TAB.
- **dbGaP**: controlled-access human data with data use agreements; institutional review
  board (IRB) and consent scope constrain secondary use.
- **EGA**: controlled-access European repository.
- **Human Cell Atlas Data Portal**: aggregated single-cell data with standardized metadata;
  as of the 2026 portal snapshot it reports roughly 70.9 million cells, 532 projects, and
  11.3 thousand donors. Treat those numbers as a point-in-time snapshot, not a fixed fact.

### 8.5 Regulated laboratory requirements

When an omics assay is used for a clinical claim rather than research, a different regime
applies.

- **Clinical Laboratory Improvement Amendments (CLIA)** in the United States: the laboratory
  must establish performance specifications for accuracy, precision, analytical sensitivity
  and specificity, reportable range, and reference intervals. The requirement attaches to the
  test, not to the platform.
- **College of American Pathologists (CAP)** accreditation: adds proficiency testing,
  document control, and validation requirements that CLIA inspection expects.
- **ISO 15189:2022**: the international standard for medical laboratory quality and
  competence, fourth edition, published December 2022. It merged point-of-care testing
  requirements previously in ISO 22870:2016, which is withdrawn. Accreditation against it is
  a laboratory-level obligation, not a per-assay one.
- **EU In Vitro Diagnostic Regulation (IVDR) 2017/746**: applies to in vitro diagnostic
  medical devices including companion diagnostics, with classification rules and performance
  evaluation requirements. It binds manufacturers placing devices on the EU market.
- **FDA**: in vitro diagnostics are devices. Laboratory-developed tests have had a contested
  regulatory history; see section 9.4.

The distinction that matters for a data scientist: CLIA and ISO 15189 constrain the
laboratory's process and its validation evidence. They do not certify a bioinformatics
pipeline. A pipeline that produces a clinical report is part of the test, and its validation
must be documented as such, but no standard tells you which normalization method to use.

### 8.6 What the standards do not cover

None of these standards specifies:

- Which normalization method is appropriate for a given design.
- How to handle batch effects.
- What effect size is clinically meaningful.
- How to report uncertainty in a classifier.
- How to validate a model that was trained on one platform and deployed on another.

Those are scientific and regulatory judgments, and they are where most of the work lives.

## 9. Industry practice and case evidence

This section separates documented facts from inference; inference is labelled.

### 9.1 Reference atlases

**GTEx.** The Genotype-Tissue Expression project's version 8 release contains 17,382
RNA-seq samples from 54 tissue sites donated by 948 post-mortem donors, with genotype data
for the same donors. The accompanying atlas paper reports cis-expression quantitative trait
loci (eQTLs) for 23,268 eGenes, covering 94.7% of protein-coding genes, in 49 tissues with at
least 70 samples. Donor ancestry was 85.3% European American, 12.3% African American, and
1.4% Asian American. The ancestry skew is not a footnote; it limits the transportability of
eQTL-based variant interpretation, which is the same problem that affects polygenic scores.

**TCGA.** The Cancer Genome Atlas Pan-Cancer Atlas analysed 11,286 tumour samples from 33
cancer types, with 9,759 samples having complete data across four platforms. The integrative
analysis produced 28 molecular subtypes, with copy-number variation contributing 47% of the
clustering signal, transcriptome 42%, and methylation 11%. The practical lesson for data
types is that platform choice determines which structure dominates: a copy-number-driven
clustering would have looked different if only expression had been measured.

**Human Cell Atlas.** The HCA Data Portal aggregates single-cell datasets across tissues.
The 2026 snapshot reports roughly 70.9 million cells from 532 projects and 11.3 thousand
donors, with organ atlases released in stages. The relevant practice point is that HCA
metadata is standardized enough to permit cross-project integration, which most
project-specific submissions are not.

**UK Biobank.** A prospective cohort with genome-wide genotyping, linkage to health records,
and imaging and biomarker subsets. It is the dominant resource for polygenic score
development and for genetic association in European-ancestry populations; its ancestry
composition is the main reason portability is now a required section in any PRS paper.

### 9.2 Platform practice

Vendors publish panel sizes and instrument specifications, and those specifications constrain
what a study can claim. Two examples, current as of 2025-2026 vendor documentation:

- **10x Genomics Xenium**: pre-designed panels range from 247 genes (mouse brain) to 380
  genes (human immuno-oncology), with a 5,000-gene pan-tissue panel and customization of up
  to 100 add-on genes. The instrument localizes transcripts with subcellular resolution and
  supports 2 slides per run with up to 236 mm² per slide.
- **10x Genomics Visium**: capture-area based, with spots that contain multiple cells. The
  spot is the unit of measurement, which is why deconvolution is mandatory rather than
  optional for cell-type claims.

The inference, not stated by the vendor: a 380-gene panel cannot discover a novel pathway.
It can confirm a hypothesis, and it can support a clinical classifier within its feature
space. Programs that buy a panel for discovery and then report "no novel findings" have
mis-specified the instrument.

### 9.3 What pharma and biotech actually do

Documented practice, from publications and regulatory filings:

- Discovery groups run untargeted assays on modest cohorts (tens to low hundreds) and use
  the results to nominate candidates, not to make clinical claims.
- Validation groups run targeted, analytically validated assays on larger cohorts, often
  with a pre-specified analysis plan.
- Translational medicine groups maintain internal reference datasets and re-run public data
  (GTEx, TCGA, HCA) to contextualize internal findings.
- Companion diagnostic development runs in parallel with the therapeutic, on the assay that
  will be used in the trial, because a diagnostic validated on a different platform is a
  different test.

Inference: the most common organizational failure is a handoff in which the discovery group
delivers a feature list and the validation group receives no raw data, no metadata, and no
pipeline record. The validation group then cannot distinguish a platform effect from a
biology effect.

### 9.4 The regulatory case that changed practice

The FDA's May 2024 final rule on laboratory developed tests would have phased out enforcement
discretion and treated LDTs as devices. On 31 March 2025, the US District Court for the
Eastern District of Texas vacated the rule in *American Clinical Laboratory Association v.
FDA*. The FDA then published an "Implementation of Vacatur" final rule on 19 September 2025,
removing the added language from 21 CFR 809.3(a) and restoring the prior text. LDTs are
therefore back under the historical enforcement-discretion approach, and the matter is
remanded to the Department of Health and Human Services.

For assay selection, the practical consequence is that the regulatory pathway for a
laboratory-developed omics test in the US is currently uncertain, and programs should not
assume either device regulation or permanent discretion. The analytical validation evidence
is required under CLIA regardless of which federal pathway applies. A program that defers
validation because "LDTs are not devices" is making a legal bet with its clinical claim.

### 9.5 The reproducibility evidence

Two industry self-audits are routinely cited together and routinely conflated.

- Begley and Ellis (2012) reported that Amgen's oncology group could confirm only 6 of 53
  landmark preclinical cancer papers, a replication rate of 11%.
- Bayer's internal analysis (Prinz, Schlange and Asadullah, 2011) found that of 67
  target-validation projects, only 14 matched published findings, and 43 were highly
  inconsistent.

These are different studies with different denominators. The frequently repeated claim that
"90% of preclinical results cannot be reproduced" comes from the 47/53 figure, not from a
general survey. Both audits are industry self-reports, and neither released the underlying
paper or project list, which limits independent verification. The lesson for omics data types is narrower: platform and protocol differences are a
documented source of non-replication, and a claim tested on one assay is a claim about one
assay.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Sample swap | Sex-check mismatch or genotype discordance in one batch | Labeling error at accessioning or plating | Per-patient results wrong; batch contaminated | Genotype fingerprint every sample; record in sample sheet |
| Scale confusion | Column sums constant; values non-integer | TPM or CPM analysed as counts | Wrong dispersion; p-values and hit counts off | Detect matrix type; use raw counts with count models |
| Double transformation | Effect sizes uniformly compressed | log applied twice | Fold changes understated by a constant | Document transform chain; assert range |
| Metadata loss | Missing batch, donor, or tissue site | Submission checklist satisfied with free text | Confounding unassessable | Enforce a schema at accessioning |
| Batch confounded with group | Empty cell in batch x group table | Cases and controls processed separately | Biology and batch inseparable | Design balanced batches; otherwise limit claim |
| Compositional artifact | Coordinated opposite changes across features | Proportions sum to a constant | Spurious negative correlations | Log-ratio transform; reference-based normalization |
| Zero inflation untreated | Zero fraction far above model expectation | Low depth or true absence conflated | Inflated dispersion or invented signal | Detection-aware model; depth covariates |
| Ambient RNA | Markers appear in every cluster at low level | Cell-free transcripts in droplets | Misleading annotation; false doublets | Ambient correction; marker filtering |
| Doublets | Cluster with two lineage markers, intermediate size | Two cells in one droplet | Spurious transitional state | Doublet detection before clustering; report rate |
| Probe dropout | Feature detection rate far below expectation | Probe failure or 3' degradation | False negatives clustered in pathways | Per-feature QC; exclude unstable features |
| Low library complexity | Duplication rate >50%, low detected features | Insufficient input or over-amplification | Underestimated dispersion; false positives | Exclude or model complexity |
| Annotation drift | Feature counts differ between runs | Annotation release changed | Wrong joins in meta-analysis | Pin release; versioned mapping table |
| Pseudo-replication | Sample count exceeds donor count | Multiple libraries per donor | Standard errors too small | Mixed model with donor effect |
| Over-filtering | Lost rare cell types or low-expression genes | Thresholds copied across datasets | Real biology removed | Dataset-specific thresholds; sensitivity analysis |
| Unrecorded protocol modification | Protocol version mismatch across batches | Ad hoc change mid-study | Batch effect correlated with protocol | Freeze protocol; log deviations |

Three red flags deserve a standing rule.

**If the column sums are constant to six digits, stop.** You are looking at proportions, and
a count model does not apply.

**If every sample in one group comes from one processing week, stop.** No correction can
separate the group from the week. The claim must be re-scoped.

**If the number of significant features is close to the number expected by chance, stop.**
For 20,000 features at alpha 0.05, that number is about 1,000. A hit list of 950 is not a
weak signal; it is noise with a familiar shape.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The assay measures the analyte it names | All interpretation depends on it | Cross-reactivity, off-target probes, isobaric interference | Spike-in recovery; orthogonal assay on a subset |
| Feature annotation is correct and current | Feature-level claims join on it | Annotation release drift, retired identifiers | Mapping-loss audit; pin release |
| Library size reflects total output | CPM and similar normalizations | Large compositional shifts, RNA degradation | Spike-in controls; compare normalization methods |
| Most features are unchanged | Median-of-ratios and TMM | Global pathway activation or arrest | Distribution of ratios; spike-in |
| Missing values are ignorable | Complete-case analysis | MNAR missingness at low abundance | Compare missingness by group; detection-probability model |
| Samples are independent | Standard errors and p-values | Repeated donors, technical replicates | Donor random effect; ICC estimate |
| Batch is exchangeable with group | Batch correction | Perfect confounding | Cross-tabulation; design review |
| Specimen integrity is adequate | Signal-to-noise for the assay | RIN below tolerance; fixation artifacts | RIN/DIN per sample; 3' bias metric |
| A platform-specific classifier transports | Clinical deployment | Different chemistry, different site | Bridge study; concordance on shared samples |
| Relative values can be compared across cohorts | Meta-analysis | Different library prep or pipeline | Recompute from raw data; never pool processed matrices |
| The feature set covers the mechanism | Pathway and mechanism claims | Panel restricted to 380 genes | State panel scope; test outside panel |

The first assumption is almost never tested and almost always load-bearing. A spike-in
control costs a small fraction of a sequencing budget and directly measures recovery.

## 12. Exercises

**Exercise 1 — Matrix identification (recall, 20 minutes).**
You are given six matrices described only by their summary statistics: (a) integers, column
sums unequal; (b) decimals, column sums 1,000,000; (c) decimals, column sums 1.0; (d)
negatives present, range -4 to 6; (e) range 0 to 31 with decimals, many zeros; (f) range
10^2 to 10^7, continuous, long right tail. Classify each and name one model that is
appropriate and one that is not.
*Deliverable:* a six-row table with classification, appropriate model, inappropriate model,
and the check that would confirm the classification.
*Criteria:* all six correct; each check is a computation, not a guess.
*Time:* 20 minutes.

**Exercise 2 — Assay selection memo (application, 90 minutes).**
A program wants to know whether a 400-gene inflammatory signature in synovial biopsy predicts
response to a Janus kinase inhibitor. Specimens: 90 fresh-frozen, 30 FFPE with RIN 2.8.
Budget: USD 60,000 for discovery. Write a two-page memo selecting an assay and stating what
it cannot measure.
*Deliverable:* memo with decision sentence, assay table, power calculation, cost, and limits
paragraph.
*Criteria:* specimen constraint identified; at least three assays compared on input, RIN,
features, and cost; power computed with stated assumptions; limits specific to the claim.
*Time:* 90 minutes.

**Exercise 3 — Power and false-discovery arithmetic (calculation, 40 minutes).**
A study has 24 samples per arm, a 22,000-feature assay, expected log2FC of 0.6, and within-
group SD of 1.1 log2 units. Compute power at alpha 0.05 (normal approximation), the expected
number of false positives at the uncorrected threshold, and the Bonferroni threshold. State
whether the study is adequately powered for the effect.
*Deliverable:* arithmetic with units, and a one-paragraph recommendation.
*Criteria:* SE computed correctly; power within 0.02 of the reference; false-positive count
and Bonferroni threshold correct; recommendation follows from the numbers.
*Time:* 40 minutes.

**Exercise 4 — Metadata audit (critique, 60 minutes).**
Take any GEO series with an RNA-seq count matrix. Extract its sample metadata. Map each field
to the five MINSEQE elements. Identify at least three gaps and name the analysis each gap
blocks.
*Deliverable:* mapping table plus a gap list with blocked analyses.
*Criteria:* all five elements addressed; each gap tied to a specific analysis; one
recommendation for recovering the missing field.
*Time:* 60 minutes.

**Exercise 5 — Compositional data critique (critique, 45 minutes).**
A paper reports that 12 of 15 genes in a pathway increased and 3 decreased, concluding
"coordinated pathway activation," using 16S relative abundances. Write a critique identifying
the compositional problem, the diagnostic that would reveal it, and the reanalysis you would
request.
*Deliverable:* 500-word critique.
*Criteria:* compositionality named; a specific diagnostic proposed; a log-ratio or
reference-based reanalysis specified; no overclaim in the other direction.
*Time:* 45 minutes.

**Exercise 6 — Design a two-stage study (design, 75 minutes).**
Design a discovery-plus-validation study for a protein biomarker of treatment response, given
a 200-patient cohort and a budget that supports one untargeted proteomics run on 60 samples
and one targeted assay on all 200. Specify sample allocation, the discovery assay, the
validation assay, the concordance metric, and the go/no-go rule.
*Deliverable:* two-page design with a sample flow diagram and a decision rule.
*Criteria:* discovery and validation are on different platforms; allocation justified;
go/no-go rule pre-specified with numbers; limits stated.
*Time:* 75 minutes.

**Exercise 7 — Write the limits paragraph (application, 30 minutes).**
For a bulk RNA-seq study of 150 kidney biopsies with no deconvolution and no single-cell data,
write the limits paragraph for the results section, naming three claims the study cannot
support.
*Deliverable:* one paragraph, 150-200 words.
*Criteria:* three specific claims named; each tied to a measurement limitation; no generic
disclaimer language.
*Time:* 30 minutes.

**Exercise 8 — Standards triage (recall/application, 40 minutes).**
For each of the following, name the binding standard or regulation and what it requires:
(a) a CLIA laboratory launching a targeted RNA panel; (b) an EU manufacturer placing a
companion diagnostic on the market; (c) a journal submission of an RNA-seq dataset; (d) a
laboratory seeking accreditation for point-of-care testing.
*Deliverable:* four-row table with standard, requirement, and what it does not cover.
*Criteria:* correct standard for each; binding versus voluntary distinguished; one "does not
cover" per row.
*Time:* 40 minutes.

## 13. Self-check answers

**Answer 1 (matrix identification).** (a) raw counts — use a negative binomial model; do not
use ordinary least squares on the raw scale. (b) TPM or CPM — use for within-sample
comparison and visualization; do not use as input to a count model without conversion.
(c) proportions — use log-ratio methods; do not use Euclidean distance. (d) log-ratio or
residual — use linear models; do not exponentiate without checking the base. (e) log2(x+1) of
normalized counts — use linear models on the log scale; do not re-log. (f) intensity — use
log transformation then robust linear models; do not treat as counts. Confirmation in every
case: check the documented pipeline, then check column sums and integer structure.

**Answer 2 (assay memo).** The FFPE subset with RIN 2.8 eliminates poly-A selection and
single-cell. The reasonable discovery choice is ribo-depleted bulk RNA-seq on the 90 frozen
samples, with the 30 FFPE samples either excluded from discovery or analysed as a separate
stratum using a 3' targeted panel. A 400-gene signature is a hypothesis, so an untargeted
assay is defensible for discovery, but the validation must be targeted. Cost: 90 x 450 =
USD 40,500, leaving USD 19,500 for the FFPE stratum at 30 x 120 = USD 3,600 plus sequencing
and analysis. Limits: no per-cell mechanism claim; no absolute abundance; FFPE stratum cannot
be pooled with frozen without a bridge study.

**Answer 3 (power arithmetic).**

```text
SE = 1.1 x sqrt(2/24) = 1.1 x 0.2887 = 0.3176
z  = 0.6 / 0.3176 = 1.889
power = 0.49 (normal approximation, two-sided alpha 0.05)
expected false positives = 22,000 x 0.05 = 1,100
Bonferroni threshold = 0.05 / 22,000 = 2.27 x 10^-6
```

The study has about 49% power, which is inadequate for the effect the program cares about.
Either the effect assumption is wrong (and should be justified from pilot data) or the sample
size must increase. Reporting a null result from this design would not distinguish absence of
effect from absence of power.

**Answer 4 (metadata audit).** Element 1 (biological system and variables) is usually present
as disease and tissue, often missing age, sex, and donor. Element 2 (reads) is present as
accessions. Element 3 (processed data) is often present as a matrix with no pipeline record.
Element 4 (sample-to-data table) is frequently absent, which blocks swap detection. Element 5
(protocols) is often a one-line kit name, which blocks reprocessing. A missing donor field
blocks any repeated-measures analysis; a missing batch field blocks batch adjustment; a
missing library kit blocks cross-study normalization.

**Answer 5 (compositional critique).** In a 16S relative-abundance table, an increase in 12
taxa mechanically forces a decrease in others, because the total is fixed by the sequencing
and rarefaction procedure. The "coordinated activation" is partly an artifact of the
constraint. Diagnostic: check whether the 3 decreasing taxa are the most abundant taxa, and
whether the sum of the 12 increases equals the sum of the 3 decreases. Reanalysis: centred
log-ratio or additive log-ratio with zero replacement, or a reference-based normalization
using a set of stable taxa. The conclusion may survive, but it must be re-derived on a scale
that is not constrained to a constant sum.

**Answer 6 (two-stage design).** Allocate the 60 discovery samples across response groups and
batches in a balanced design, with at least 20 per group. Run untargeted proteomics (DIA or
TMT) on the 60. Select at most 10 candidates, with a pre-specified effect-size threshold and
an FDR threshold. Validate all 200 on a targeted immunoassay or PRM assay. The concordance
metric is the correlation of effect sizes between platforms and the validation AUC with a
confidence interval. Go/no-go: proceed if at least 3 candidates replicate with the same
direction at FDR < 0.05 and the validation AUC lower bound exceeds 0.65. The design is
defensible because discovery and validation are orthogonal and the decision rule is fixed
before validation data exist.

**Answer 7 (limits paragraph).** A model answer: "These data are bulk tissue averages and
cannot establish that any transcript change is cell-intrinsic; the observed differences may
reflect shifts in epithelial, immune, or stromal composition, which we did not measure. The
assay is relative and cannot support a threshold in copies per cell or per microgram of
tissue. The cohort is a single centre with a single library protocol, so the results cannot
be assumed to transport to another platform without a bridge study. We therefore report
associations between bulk expression and outcome, not mechanisms, thresholds, or
transportable classifiers."

**Answer 8 (standards triage).** (a) CLIA — requires performance specifications for accuracy,
precision, analytical sensitivity and specificity, reportable range, and reference intervals;
does not specify normalization or bioinformatics validation methods. (b) IVDR 2017/746 —
requires classification, performance evaluation, and conformity assessment; does not approve
individual algorithms or specify model metrics. (c) MINSEQE/MIAME via the repository —
requires the five or six reporting elements; does not enforce machine-readable completeness.
(d) ISO 15189:2022 — requires quality management, competence, and point-of-care provisions
(ISO 22870 is withdrawn and merged); does not certify a specific assay's clinical utility.

## 14. Interview questions

**1. How do you decide which omics assay to use?**
Strong answer: starts from the decision and the minimum effect size, then the analyte, then
specimen constraints (input, RIN, fixation), then resolution, specificity, and whether
relative or absolute quantitation is required. Names the validation assay at design time.
Weak answer: names a favourite platform, or says "RNA-seq is the standard."

**2. A matrix arrives labelled "normalized expression." What do you check first?**
Strong answer: column sums, integer structure, value range, presence of negatives, and the
pipeline record. Then maps to raw counts if possible before modelling.
Weak answer: runs a t-test on the values as delivered.

**3. Why is TPM inappropriate for DESeq2?**
Strong answer: TPM is a within-sample proportion after length normalization; the negative
binomial model expects counts with a mean-variance relationship and uses size factors
estimated from counts. TPM also destroys the information needed to estimate dispersion.
Weak answer: "it's normalized, so it's fine" or "TPM is better than FPKM."

**4. What does MINSEQE require, and what does it not?**
Strong answer: five elements — biological system and variables, reads, processed data,
sample-to-data table, protocols. It does not enforce a machine-readable schema, specify
normalization, or guarantee that the metadata is analytically sufficient.
Weak answer: "it's a checklist for sequencing data" with no elements named.

**5. How would you detect a sample swap?**
Strong answer: genotype concordance against a donor sample or a fingerprint, sex check from
XIST/RPS4Y1, and batch-level clustering of discordances. Then relabel only with genotype
evidence, never by guessing.
Weak answer: "check the clustering" without a mechanism for attributing labels.

**6. What is compositionality and why does it matter?**
Strong answer: only relative information is identified; the sum is constrained, so
correlations are partly structural. Uses log-ratios and states the reference assumption.
Weak answer: "it means the data are percentages" with no consequence for inference.

**7. When is batch correction not possible?**
Strong answer: when batch is perfectly confounded with the group of interest, so no model can
separate them. The design must change or the claim must be restricted.
Weak answer: "ComBat always works" or "just add batch as a covariate."

**8. How do you know a targeted panel is the right choice?**
Strong answer: when the hypothesis set is fixed, the multiple-testing burden falls, power
rises, and the claim is confirmation rather than discovery. Acknowledges that novel features
cannot be found.
Weak answer: "panels are cheaper, so they are better."

**9. What is the difference between analytical validation and clinical validation?**
Strong answer: analytical validation establishes that the assay measures the analyte
accurately and reproducibly (accuracy, precision, sensitivity, specificity, range);
clinical validation establishes association with the clinical outcome in the intended
population; clinical utility establishes that using the test improves decisions or outcomes.
Weak answer: conflates the three, or treats a significant association as validation.

**10. A study reports 900 significant genes out of 20,000 at p < 0.05. What do you conclude?**
Strong answer: about 1,000 are expected by chance at that threshold, so the result is
consistent with no signal; ask for effect sizes, QQ plot, and corrected q-values.
Weak answer: "900 genes is a strong signal."

**11. What metadata field most often blocks reanalysis, and why?**
Strong answer: batch or processing group, because without it confounding cannot be assessed
and correction is guesswork; donor identity is second because it determines independence.
Weak answer: "sample name."

**12. Where does an omics claim most often exceed its assay?**
Strong answer: cell-intrinsic mechanism claims from bulk data, absolute thresholds from
relative assays, and functional claims from 16S composition. Each has a specific missing
measurement.
Weak answer: gives a generic "correlation is not causation" answer without naming the assay
limitation.

## 15. Key terms

- **Analyte** — the molecule class an assay physically measures.
- **Annotation** — the reference mapping from genomic coordinates to features.
- **AnnData** — HDF5-based container for single-cell matrices and metadata.
- **Ambient RNA** — cell-free transcripts captured alongside intact cells.
- **Batch** — a group of samples processed together, a source of technical variation.
- **Benjamini-Hochberg (BH)** — FDR-controlling procedure based on ranked p-values.
- **Bulk assay** — measurement averaged over many cells in a specimen.
- **CLIA** — US Clinical Laboratory Improvement Amendments; laboratory testing regulation.
- **Compositional data** — data carrying only relative information, with a constrained sum.
- **Count matrix** — features by samples table of integer counts.
- **Coverage** — reads per genomic position, expressed as fold depth.
- **CPM** — counts per million reads.
- **Doublet** — two cells captured in one droplet or well.
- **FASTQ** — text format for reads and per-base quality scores.
- **FFPE** — formalin-fixed, paraffin-embedded; a specimen type that fragments nucleic acids.
- **FPKM** — fragments per kilobase million; length-normalized, not sum-consistent.
- **GTEx** — Genotype-Tissue Expression project; tissue reference atlas.
- **HCA** — Human Cell Atlas; single-cell reference consortium.
- **Intensity** — instrument signal proportional to analyte amount, in arbitrary units.
- **ISO 15189** — international standard for medical laboratory quality and competence.
- **IVDR** — EU In Vitro Diagnostic Regulation 2017/746.
- **Library complexity** — number of distinct molecules in a sequencing library.
- **Metadata** — structured description of samples, processing, and design.
- **MIAME** — Minimum Information About a Microarray Experiment.
- **MINSEQE** — minimum reporting standard for high-throughput sequencing experiments.
- **Probe dropout** — failure of a targeted probe to detect a present transcript.
- **RIN** — RNA integrity number, a 1-10 degradation score.
- **Single-cell assay** — per-cell measurement with cell-level attribution.
- **Spatial assay** — measurement with retained tissue coordinates.
- **SummarizedExperiment** — Bioconductor container linking assays to sample metadata.
- **TPM** — transcripts per million; within-sample proportion after length normalization.
- **UMI** — unique molecular identifier; a barcode counting distinct molecules.
- **VCF** — variant call format, maintained under GA4GH hts-specs.
- **Zero inflation** — excess zeros relative to a count model's expectation.

## 16. Sources

Standards and specifications:

- FGED Society, MINSEQE project page. https://www.fged.org/projects/minseqe (accessed 25 October 2026).
- MINSEQE v1.0 proposal, Zenodo, 2012. https://doi.org/10.5281/zenodo.5706412 (accessed 25 October 2026).
- NCBI GEO, MIAME and MINSEQE guidelines. https://www.ncbi.nlm.nih.gov/geo/info/MIAME.html (accessed 25 October 2026).
- GA4GH hts-specs, Variant Call Format 4.4 and 4.5. https://samtools.github.io/hts-specs/ (accessed 25 October 2026).
- GA4GH, Genetic Variation Formats (VCF). https://www.ga4gh.org/product/genetic-variation-formats-vcf/ (accessed 25 October 2026).
- ISO 15189:2022, Medical laboratories — Requirements for quality and competence. https://www.iso.org/standard/76677.html (accessed 25 October 2026).
- Regulation (EU) 2017/746 on in vitro diagnostic medical devices. https://eur-lex.europa.eu/eli/reg/2017/746/oj (accessed 25 October 2026).

Datasets and atlases:

- GTEx Portal, V8 release notes. https://gtexportal.org/home/ (accessed 25 October 2026).
- GTEx Consortium, "The GTEx Consortium atlas of genetic regulatory effects across human tissues," *Science*, 2020. https://doi.org/10.1126/science.aaz1776 (accessed 25 October 2026).
- The Cancer Genome Atlas, programme page. https://www.cancer.gov/ccg/research/genome-sequencing/tcga (accessed 25 October 2026).
- Hoadley, K. A. et al., "Cell-of-Origin Patterns Dominate the Molecular Classification of 10,000 Tumors from 33 Types of Cancer," *Cell* 173:291-304, 2018. https://doi.org/10.1016/j.cell.2018.03.022 (accessed 25 October 2026).
- Human Cell Atlas Data Portal. https://data.humancellatlas.org/ (accessed 25 October 2026).
- UK Biobank. https://www.ukbiobank.ac.uk/ (accessed 25 October 2026).

Methods and evidence:

- SEQC/MAQC-III Consortium, "A comprehensive assessment of RNA-seq accuracy, reproducibility and information content," *Nature Biotechnology* 32:903-914, 2014. https://doi.org/10.1038/nbt.2957 (accessed 25 October 2026).
- Love, M. I., Huber, W., Anders, S., "Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2," *Genome Biology* 15:550, 2014. https://doi.org/10.1186/s13059-014-0550-8 (accessed 25 October 2026).
- Begley, C. G., Ellis, L. M., "Raise standards for preclinical cancer research," *Nature* 483:531-533, 2012. https://doi.org/10.1038/483531a (accessed 25 October 2026).
- Prinz, F., Schlange, T., Asadullah, K., "Believe it or not: how much can we rely on published data on potential drug targets?" *Nature Reviews Drug Discovery* 10:712, 2011. https://doi.org/10.1038/nrd3439-c1 (accessed 25 October 2026).

Regulatory:

- FDA, Laboratory Developed Tests. https://www.fda.gov/medical-devices/in-vitro-diagnostics/laboratory-developed-tests (accessed 25 October 2026).
- Federal Register, "Medical Devices; Laboratory Developed Tests; Implementation of Vacatur," 90 FR 45134, 19 September 2025. https://www.federalregister.gov/documents/2025/09/19/2025-18239/ (accessed 25 October 2026).
- CMS, Clinical Laboratory Improvement Amendments. https://www.cms.gov/medicare/quality/clinical-laboratory-improvement-amendments (accessed 25 October 2026).

Vendor documentation:

- 10x Genomics, Xenium panels. https://www.10xgenomics.com/platforms/xenium/panels (accessed 25 October 2026).

## 17. Links

- Module map: [`_index.md`](_index.md)
- Next topic: [`02_genomics-and-variants.md`](02_genomics-and-variants.md)
- Related topic: [`03_transcriptomics.md`](03_transcriptomics.md)
- Related topic: [`06_proteomics-and-metabolomics.md`](06_proteomics-and-metabolomics.md)
- Survey file: [`../../12_omics_precision_medicine.md`](../../course/12_omics_precision_medicine.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 12.2, bulk RNA-seq differential expression, in the survey file: [`../../12_omics_precision_medicine.md`](../../course/12_omics_precision_medicine.md)
