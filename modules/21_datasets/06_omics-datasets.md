# Omics Datasets

> Module 21 — Datasets, Tools, and Benchmarks
> Topic 06 of 12
> Estimated study time: 4 hours
> Prerequisites: Topic 01, Topic 02
> Version 2026.3

## 1. Why this topic exists

Omics data are the clearest case in biomedical data science where the measurement regime, not the file size, determines what can be claimed. A bulk RNA-seq matrix, a single-cell count matrix, a spatial transcriptomics section, and a germline variant call set are all "omics," and they answer different questions with different units of inference. A team that treats them as one data type will make claims the data cannot support.

The recurring failures are specific:

- A differential expression result is computed across cells rather than donors, so the p-value reflects the number of cells and not the number of people.
- A single-cell atlas is used to claim that a cell type is more abundant in a disease, when the atlas has no disease cohort and its composition reflects tissue sampling.
- A germline variant is treated as a drug target because it is associated with a trait, without colocalisation, direction, or replication.
- A model is trained on one batch and tested on another, and the batch effect is reported as biology.
- An open processed matrix is assumed to permit commercial use because it downloaded without a login.
- A TCGA survival model is presented as a biomarker validation, when the cohort is a selected set of tertiary-center cases with heterogeneous treatment.

Omics data are also where the gap between open and controlled access is widest. Processed expression matrices are often open; raw sequence, germline variants, and donor metadata are often controlled. A project can have an open analysis layer and a controlled validation layer, and the controlled layer sets the schedule and the licence.

This topic covers the major resources: TCGA, GTEx, gnomAD, the UK Biobank, CELLxGENE, the Human Cell Atlas, Tabula Sapiens, GEO, PRIDE, MetaboLights, DepMap, LINCS, Open Targets, ChEMBL, and PubChem. It treats the donor as the unit of inference, batch as a confounder, and multiple testing as a design constraint.

The practical consequence is that omics projects need two reviews, not one. The first asks whether the data can answer the biological question; the second asks whether the team may use the data for the intended purpose. A project that passes the first and fails the second has a result it cannot deploy, and a project that passes the second and fails the first has a permission it cannot use.

A second consequence is that the analysis choices are part of the dataset's meaning. A count matrix is not a result until the normalisation, the unit, the batch model, and the test are specified. The dataset card should record those choices alongside the files.

> **Note:** Counts were checked on 26 October 2026. GTEx v8 contains 17,382 RNA-seq samples from 948 donors; the portal also lists later releases with different counts. Tabula Sapiens v1 contains roughly 483,000 to 500,000 cells from 24 tissues and 15 donors, and v2 expands to about 1.1 million cells from 28 tissues and 24 donors. Use the release named in the file you downloaded.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** bulk, single-cell, spatial, and germline measurement regimes by their unit of inference and the claims they support. Assessed by Exercise 1 and self-check item 1.
2. **Identify** the donor, sample, cell, and gene units in an omics table and choose the unit that matches the claim. Assessed by Exercise 2.
3. **Audit** a single-cell analysis for pseudoreplication and compute a donor-level permutation test. Assessed by Exercise 3.
4. **Apply** multiple-testing control and explain why a nominal threshold is not evidence at genome scale. Assessed by Exercise 4.
5. **Map** open and controlled layers for at least four resources and state the obligations of each. Assessed by Exercise 5.
6. **Evaluate** whether an eQTL or GWAS result supports association, colocalisation, or causation. Assessed by Exercise 6.
7. **Detect** batch and composition confounding in an expression dataset. Assessed by Exercise 7.
8. **Write** an omics data card that names the reference build, the annotation version, and the prohibited claims. Assessed by Exercise 8.

## 3. The core mental model

An omics dataset is a stack of transformations from a biological sample to a matrix of numbers. Every transformation is a choice, and the choice determines what the matrix means.

```text
   BIOLOGICAL QUESTION
   association | colocalisation | causal | predictive | descriptive atlas
        |
        v
   MEASUREMENT REGIME
   bulk | single-cell | single-nucleus | spatial | germline | proteomics
        |
        v
   SAMPLE AND DONOR
   tissue | time | treatment | ancestry | batch
        |
        v
   ASSAY
   library prep | sequencing depth | platform | reference build
        |
        v
   MATRIX
   genes x samples | genes x cells | spots | variants
        |
        v
   NORMALISATION AND MODEL
   counts | TPM | log | pseudobulk | integration
        |
        v
   INFERENCE UNIT
   donor | sample | cell | gene | variant
```

**The unit rule.** The unit of inference is the donor for a population claim, the sample for a tissue claim, and the cell for a cell-state description. A p-value computed across cells answers a question about cells in this dataset, not about people. The most common single-cell error is treating cells as independent replicates.

**The regime rule.** Bulk measures an average over a tissue; single-cell measures a distribution over cells; spatial measures a distribution over positions; germline measures inherited variation. A bulk deconvolution is a model output, not an observed cell count. A single-cell atlas is a collection of samples, not a sampling frame for a tissue.

**The batch rule.** Batch can be confounded with biology. If all cases are processed in one batch and all controls in another, the disease effect and the batch effect are not separable. A model that integrates aggressively can remove the disease signal along with the batch. Evaluate on held-out donors and report corrected and uncorrected results.

**The testing rule.** At 20,000 genes and a nominal 5 percent threshold, about 1,000 false positives are expected before any true signal is counted. Multiple-testing control is part of the design, not a post hoc correction.

**The reference rule.** Gene annotations, genome builds, transcript sets, and cell ontologies change. A result computed against one annotation is not directly comparable with a result computed against another. Record the build, the annotation version, and the pipeline version.

### What each resource cannot answer

| Resource | Cannot answer | Why |
|---|---|---|
| TCGA | Population incidence or survival | Tertiary-center case selection |
| GTEx | Disease-specific regulation | No disease cohort |
| gnomAD | Individual-level risk | Aggregate frequencies |
| UK Biobank | General-population prevalence | Volunteer cohort |
| CELLxGENE | Tissue-wide prevalence | Dataset composition, not sampling |
| Tabula Sapiens | Disease comparison | Small donor set, no case-control |
| GEO | A uniform benchmark | Heterogeneous series |
| DepMap | Tumour biology in vivo | Cell lines only |
| Open Targets | Causal proof | Integrated evidence, not experiment |
| ChEMBL | Clinical efficacy | In vitro assay heterogeneity |

| Regime | Unit of observation | Unit of inference | Claim it supports | Claim it cannot support |
|---|---|---|---|---|
| Bulk RNA-seq | Sample | Donor or sample | Tissue-level differential expression | Cell-type-specific effect without deconvolution |
| Single-cell RNA-seq | Cell | Donor for population claims | Cell states and composition | Tissue-wide prevalence |
| Single-nucleus RNA-seq | Nucleus | Donor | Cell states in frozen tissue | Cytoplasmic RNA dynamics |
| Spatial transcriptomics | Spot or cell | Section or donor | Spatial organisation | Whole-organ prevalence |
| Germline variants | Variant | Individual | Association and colocalisation | Causation without design |
| Proteomics | Peptide or protein | Sample | Protein abundance | Transcript-level inference |
| Metabolomics | Feature | Sample | Metabolite association | Pathway causation |

## 4. Definitions and vocabulary

### 4.1 Measurement and assay terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Bulk sequencing | Average signal across a tissue sample. | Treated as cell-type specific. | Sample |
| Single-cell RNA-seq | Transcript counts per cell. | Cells treated as independent donors. | Cell |
| Single-nucleus RNA-seq | Transcript counts per nucleus. | Treated as equivalent to whole cells. | Nucleus |
| Spatial transcriptomics | Transcript measurements with position. | Treated as dissociated single-cell data. | Spot or cell |
| Library preparation | The protocol that converts RNA or DNA into a sequencable library. | Ignored as a batch source. | Protocol |
| Sequencing depth | The number of reads per sample or cell. | Treated as a biological difference. | Reads |
| Coverage | The proportion of a reference covered by reads. | Confused with depth. | Proportion |
| Batch | A group processed together. | Confounded with the outcome. | Categorical |
| Reference build | The genome assembly used for alignment. | Mixed across samples. | GRCh38 or other |
| Annotation version | The gene or transcript annotation used. | Not recorded. | GENCODE release |
| Normalisation | Scaling counts for comparability. | Chosen without stating assumptions. | Method |
| Pseudobulk | Aggregating cells to a sample or donor. | Omitted before differential testing. | Count matrix |

### 4.2 Statistical and causal terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Differential expression | Association between a condition and gene expression. | Interpreted as causation. | Effect size |
| eQTL | A variant associated with expression of a gene. | Treated as a drug target. | Association |
| Colocalisation | Evidence that two traits share a causal variant in a region. | Treated as proof of mediation. | Posterior probability |
| Mendelian randomisation | Causal inference using genetic instruments. | Assumed valid without testing pleiotropy. | Estimate |
| Pleiotropy | A variant affecting multiple traits. | Ignored in instrument selection. | Sensitivity analysis |
| Multiple testing | Testing many hypotheses at once. | Corrected only after seeing results. | Adjusted p-value |
| False discovery rate | Expected proportion of false positives among discoveries. | Confused with family-wise error. | Proportion |
| Bonferroni correction | Alpha divided by the number of tests. | Applied without stating the test count. | Threshold |
| Pseudoreplication | Treating non-independent units as independent. | Common in single-cell testing. | Design error |
| Batch effect | Technical variation between processing groups. | Reported as biology. | Effect |
| Compositional data | Data where proportions sum to a constant. | Analysed as independent absolute values. | Proportion |
| Effect size | The magnitude of an association. | Reported as a p-value only. | Log fold change |

### 4.3 Access and governance terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Open tier | Data available without individual approval. | Assumed to permit commercial use. | Access |
| Controlled tier | Data requiring institutional approval. | Planned as a download. | Access |
| dbGaP | US controlled-access genomic archive. | Treated as a download site. | Repository |
| AnVIL | Cloud platform for genomic analysis. | Treated as a data source. | Platform |
| Data use agreement | Contract governing access and use. | Read as a publication licence. | Contract |
| Germline | Inherited genetic variation. | Treated as somatic. | Variant class |
| Somatic | Acquired variation, often tumour. | Treated as germline. | Variant class |
| Consent scope | What participants agreed to. | Assumed to cover commercial research. | Consent term |
| Data residency | Where data may be processed. | Broken by a default cloud region. | Jurisdiction |

### 4.4 Platform and pipeline terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Pipeline | The sequence of analysis steps. | Treated as a tool name | Configuration |
| Container | A reproducible software environment. | Not pinned | Image digest |
| Quantification | Estimating transcript or protein abundance. | Confused with normalisation | Method |
| Alignment | Mapping reads to a reference. | Assumed to be interchangeable | Aligner |
| Feature counting | Assigning reads to genes or regions. | Ignored as a batch source | Annotation |
| Ambient correction | Removing cell-free transcripts. | Omitted in low-quality data | Method |
| Doublet detection | Identifying droplets with two cells. | Treated as optional | Method |
| Integration | Combining datasets across batches. | Applied without evaluation | Method |
| Clustering | Grouping cells by similarity. | Resolution chosen post hoc | Resolution |
| Embedding | A low-dimensional representation. | Treated as ground truth | Dimensions |

### 4.5 Resource terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| TCGA | Cancer genome atlas across cancer types. | Treated as a population cohort. | Dataset |
| GTEx | Normal-tissue expression and eQTL resource. | Treated as a disease cohort. | Dataset |
| gnomAD | Aggregate variant frequency resource. | Treated as a case-control study. | Dataset |
| UK Biobank | Population cohort with genomics and health records. | Treated as a probability sample. | Cohort |
| CELLxGENE | Single-cell atlas portal with standardized metadata. | Treated as one dataset. | Portal |
| Human Cell Atlas | International single-cell consortium. | Treated as a single consistent dataset. | Consortium |
| Tabula Sapiens | Multi-organ human single-cell atlas. | Treated as a population sample. | Dataset |
| GEO | Gene expression omnibus of functional genomics studies. | Treated as curated and comparable. | Repository |
| DepMap | Cancer cell-line dependency map. | Treated as tumour biology. | Dataset |
| Open Targets | Integrated target-disease evidence platform. | Treated as experimental validation. | Platform |

## 5. The mechanics, step by step

The procedure has 18 steps in four phases: define the claim and the regime, audit samples and batches, analyse at the correct unit, and validate and report.

### Phase A — Define the claim and the regime

**Step 1. State the biological claim and its evidence type.**
Input: the research or product question.
Output: a claim labelled as association, colocalisation, causal, predictive, or descriptive.
Rationale: each evidence type needs a different design and a different dataset. A descriptive atlas cannot support a causal claim.
Branch: if the claim is causal, the design needs genetics, perturbation, or a trial; an atlas alone is insufficient.
Edge case: a target nomination can combine evidence types, but each piece must be labelled with its own strength.

**Step 2. Choose the measurement regime.**
Input: the claim and the tissue.
Output: bulk, single-cell, single-nucleus, spatial, germline, proteomics, or metabolomics.
Rationale: the regime determines the unit of observation and the inference.
Branch: use the regime decision table below.
Edge case: a claim about cell-type composition needs single-cell or deconvolution with validation; a claim about tissue average needs bulk.

**Step 3. Define the tissue, context, and comparison.**
Input: the biological question and the available specimens.
Output: tissue, disease state, treatment, time point, and comparator.
Rationale: an expression difference is only interpretable against a defined comparator.
Branch: if the dataset has no disease cohort, the claim is descriptive or genetic, not a case-control difference.
Edge case: post-mortem tissue differs from biopsy tissue in RNA quality and cell composition.

**Step 4. Map the open and controlled layers.**
Input: the resource documentation.
Output: a table of what is open, what is controlled, and what is not available.
Rationale: the controlled layer sets the schedule and the licence.
Branch: if the claim needs raw sequence or germline data, plan for controlled access.
Edge case: processed data can be open while the underlying raw data are controlled.

### Resource profiles

| Resource | Custodian | Version inspected | Unit | Scale | Access and licence | What it cannot answer |
|---|---|---|---|---|---|---|
| TCGA | National Cancer Institute | Current GDC release | Patient, sample, aliquot | About 11,000 patients; 33 cancer types | Open clinical and expression; germline controlled via dbGaP | Population incidence; causal treatment effect |
| GTEx | Broad Institute and NIH | v8, with later portal releases | Donor, tissue sample | 17,382 samples; 948 donors; 54 tissues | Processed data open; raw and full metadata controlled | Disease-specific effects; clinical outcomes |
| gnomAD | Broad Institute | Current release | Variant, population | Hundreds of thousands of individuals | Aggregate frequencies open; some files registered | Individual-level association; causality |
| UK Biobank | UK Biobank | Current release | Participant | 500,000 participants | Application, fee, UKB-RAP | General-population prevalence; causal effect without design |
| CELLxGENE Discover | Chan Zuckerberg Initiative | Growing portal | Cell, dataset | Tens of millions of cells | Per-collection licence, often CC BY 4.0 | A sampling frame for a tissue |
| Human Cell Atlas | International consortium | Growing | Cell, dataset | Thousands of datasets | Per-dataset terms | One consistent protocol |
| Tabula Sapiens | Tabula Sapiens Consortium | v1 and v2 | Cell, donor | v1 about 483,000 to 500,000 cells, 24 tissues, 15 donors; v2 about 1.1 million cells, 28 tissues, 24 donors | Processed CC BY 4.0; raw under a data use agreement | Population prevalence; disease comparison |
| GEO | NCBI | Continuous | Sample, series | Hundreds of thousands of samples | Per-series terms | Uniform curation and comparable processing |
| PRIDE | EMBL-EBI | Continuous | Dataset, sample | Tens of thousands of datasets | Open | Uniform identification confidence |
| DepMap | Broad Institute | Current release | Cell line | About 1,000 to 1,200 lines | Open with registration | Tumour biology in vivo |
| Open Targets | Open Targets consortium | Periodic releases | Target-disease pair | Counts change by release | Open with attribution | Experimental validation; causal proof |
| ChEMBL | EMBL-EBI | Current release | Compound, assay, activity | Millions of activity records | CC BY-SA 4.0 | Clinical efficacy; assay comparability |

The scale column mixes patients, samples, donors, cells, and cell lines. The last column is the one that decides fitness.

### Assay comparison

| Assay | What it measures | Strength | Limit | Typical claim |
|---|---|---|---|---|
| Bulk RNA-seq | Average transcript abundance | Robust and cheap | Cell-type mixture | Tissue-level expression |
| Single-cell RNA-seq | Transcripts per cell | Cell states and heterogeneity | Sparse and costly | Cell atlas and composition |
| Single-nucleus RNA-seq | Nuclear transcripts | Frozen and difficult tissues | Nuclear-cytoplasmic bias | Cell states in archived tissue |
| Spatial transcriptomics | Expression with position | Tissue architecture | Resolution and depth | Spatial domains |
| Whole-genome sequencing | Germline and somatic variants | Comprehensive variation | Cost and interpretation | Genetic association |
| Whole-exome sequencing | Coding variants | Cost-effective | Misses non-coding | Coding variant discovery |
| Proteomics | Protein abundance | Closer to function | Coverage and quantification | Protein biomarker |
| Metabolomics | Small molecules | Functional readout | Identification confidence | Metabolic signature |

### Quality-control metrics

| Metric | What it detects | Threshold style | Risk if ignored |
|---|---|---|---|
| Reads per sample | Low sequencing depth | Study-specific | Noise and false negatives |
| Mapping rate | Contamination or degradation | Below a study threshold | Misaligned signal |
| Duplication rate | Library complexity | High values flag low complexity | Overstated abundance |
| rRNA fraction | Library preparation failure | High values flag failure | Wasted depth |
| Mitochondrial fraction | Cell stress or damage | High values in single-cell | Dying cells dominate |
| Genes per cell | Low-quality or empty droplets | Study-specific | Empty droplets as cell types |
| Doublet rate | Two cells in one droplet | Estimated by tool | False intermediate states |
| Ambient RNA | Cell-free transcripts | Estimated by tool | False expression |
| Batch distribution | Technical grouping | Balance check | Confounding |
| Sex check | Sample swap | Genotypic sex versus metadata | Mislabeled samples |

### Normalisation choices

| Method | Assumption | Use | Risk |
|---|---|---|---|
| Counts per million | Library size is the main technical factor | Simple comparison | Compositional bias |
| TPM or FPKM | Transcript length matters | Bulk expression | Not comparable across pipelines |
| Median-of-ratios | Most genes are not differential | Bulk differential expression | Violated in global shifts |
| Log transformation | Stabilises variance | Visualisation and linear models | Zero handling |
| Pearson residuals | Models the count distribution | Single-cell | Computationally heavier |
| Scran pooling | Estimates size factors from pools | Single-cell | Pool composition |
| Compositional (CLR) | Data are proportions | Microbiome and cell composition | Zero replacement |
| Integration | Removes batch | Multi-dataset atlas | Can remove biology |

### Single-cell workflow decisions

| Step | Decision | Common failure | Evidence to record |
|---|---|---|---|
| Filtering | Which cells pass QC | Filtering differs by condition | Thresholds and counts |
| Normalisation | Counts to scaled values | Compositional bias | Method |
| Feature selection | Highly variable genes | Batch-driven features | Gene list |
| Dimensionality | PCA or equivalent | Overfitting to batch | Components |
| Integration | Batch correction | Erasing disease signal | Method and parameters |
| Clustering | Resolution | Arbitrary cluster count | Resolution and stability |
| Annotation | Reference and markers | Forced labels | Reference version |
| Differential testing | Cell or donor unit | Pseudoreplication | Unit and test |
| Composition | Proportion test | Denominator ignored | Method |
| Validation | Independent donor set | Same-donor leakage | Split specification |

### Genome build and annotation

| Item | Example | Why it matters | What to record |
|---|---|---|---|
| Genome build | GRCh38 | Coordinates differ between builds | Build and patch |
| Gene annotation | GENCODE release | Gene models change | Release number |
| Transcript set | Ensembl or RefSeq | Quantification differs | Source and version |
| Variant nomenclature | HGVS | Transcript-specific naming | Transcript ID |
| Cell ontology | CL | Cell-type labels need a controlled vocabulary | Ontology version |
| Genome-wide significance | 5e-8 | Conventional threshold for common variants | Threshold used |
| Ancestry reference | Population panel | Imputation and association depend on it | Panel and version |
| Batch metadata | Processing group | Confounding assessment | Field and source |

### Power and replication

| Design | Donors per group | What it can support | What it cannot |
|---|---:|---|---|
| Pilot atlas | 3 | Cell-state description | Population or disease comparison |
| Small case-control | 10 | Large effects with batch balance | Subtle effects and subgroups |
| Moderate case-control | 50 | Common effects | Rare cell types and interactions |
| Consortium | 500 | Robust association and colocalisation | Clinical causality without design |
| Population cohort | 500,000 | Genetic discovery | Deep phenotyping for every participant |
| Replication | Independent | Confirmation | Discovery of new effects by itself |

The donor count, not the cell count, determines the power of a population claim. A million cells from six donors is a six-donor study with a detailed description of those six people.

### Normalisation worked example

| Sample | Gene A counts | Library size | CPM | Rank after CPM |
|---|---:|---:|---:|---:|
| S1 | 100 | 1,000,000 | 100 | 1 |
| S2 | 90 | 900,000 | 100 | 1 |
| S3 | 50 | 1,000,000 | 50 | 2 |
| S4 | 60 | 1,200,000 | 50 | 2 |
| S5 | 20 | 400,000 | 50 | 2 |
| S6 | 10 | 200,000 | 50 | 2 |

The table shows why raw counts are not comparable across samples. S1 and S3 have the same library size and different counts; S3 and S5 have different counts and the same CPM. The normalisation choice changes the ranking, and the choice must be stated.

### Common analysis mistakes and their fixes

| Mistake | Why it happens | Consequence | Fix |
|---|---|---|---|
| Testing cells as replicates | Convenient software default | Pseudoreplication | Pseudobulk by donor |
| Filtering after seeing groups | Exploratory workflow | Selection bias | Pre-specify filters |
| Normalising before splitting | Pipeline convenience | Test leakage | Fit on training |
| Choosing clusters by p-value | Iterative analysis | Overfitting | Pre-specify resolution |
| Annotating with the test reference | Convenience | Circularity | Independent reference |
| Ignoring composition | Standard tools | Spurious proportions | Compositional method |
| Correcting batch by condition | Confounded design | Removes biology | Design fix |
| Reporting only significant genes | Publication habit | Winner's curse | Report effect sizes |

### Phase B — Audit samples and batches

**Step 5. Audit the donor and sample counts.**
Input: the metadata.
Output: donors, samples, cells, and the ratio of cells to donors.
Rationale: the donor count is the effective sample size for a population claim.
Branch: if the donor count is small, the claim is limited regardless of cell count.
Edge case: a donor can contribute multiple tissues and multiple time points; the independence structure must be modelled.

**Step 6. Audit batch and confounding.**
Input: the processing metadata.
Output: a table of batch by condition, site, and time.
Rationale: if batch and condition are collinear, the effect is not identifiable.
Branch: if collinearity is complete, the dataset cannot answer the question; a different dataset or design is needed.
Edge case: a batch variable can be missing from the metadata and recoverable from the expression matrix.

**Step 7. Audit sample quality.**
Input: QC metrics.
Output: the fraction of samples or cells passing quality thresholds.
Rationale: low-quality cells and degraded RNA can drive apparent differences.
Branch: if quality differs by condition, filtering can bias the comparison.
Edge case: ambient RNA and doublets are common in single-cell data and can create false cell types.

**Step 8. Audit the analysis pipeline and reference versions.**
Input: the methods and the pipeline configuration.
Output: aligner, reference build, annotation version, normalisation, and integration method.
Rationale: results are not comparable across pipelines and references.
Branch: if the pipeline is not documented, the result cannot be reproduced.
Edge case: a reference update can change gene identifiers and break a join.

### Phase C — Analyse at the correct unit

**Step 9. Choose the inference unit.**
Input: the claim and the design.
Output: donor, sample, cell, or variant.
Rationale: the unit determines the test and the degrees of freedom.
Branch: for a population claim, aggregate to donor or sample; for a cell-state description, cell-level analysis is acceptable.
Edge case: a donor-level test with three donors per group cannot reach conventional significance.

**Step 10. Aggregate to pseudobulk where needed.**
Input: the cell-level counts and the donor labels.
Output: a donor-by-gene matrix.
Rationale: pseudobulk respects the independence structure and reduces technical noise.
Branch: if the donor count is too small, report effect sizes and confidence intervals rather than p-values.
Edge case: pseudobulk loses cell-type specificity; use it for the donor-level claim and cell-level analysis for description.

**Step 11. Control multiple testing.**
Input: the number of tests and the hypothesis set.
Output: a Bonferroni or false-discovery-rate threshold with the test count stated.
Rationale: genome-scale testing produces false positives at nominal thresholds.
Branch: if the hypothesis set is pre-specified, control within that set; if it is genome-wide, control genome-wide.
Edge case: the number of tests depends on filtering; report the filter and the count.

**Step 12. Handle composition and normalisation.**
Input: the count matrix.
Output: a normalisation method and a composition-aware analysis.
Rationale: proportions sum to a constant, and changes in one cell type affect the apparent abundance of others.
Branch: if the claim is about a proportion, use a compositional method and report the denominator.
Edge case: total-count normalisation can induce spurious correlations.

**Step 13. Annotate cell types with a documented reference.**
Input: the expression matrix and a reference atlas.
Output: cell-type labels with a confidence measure and the reference version.
Rationale: annotation is a model output, not a ground truth.
Branch: if the reference lacks the cell type, the label is forced into the nearest class.
Edge case: a novel cell state can be labelled as a known type and disappear from the analysis.

### Phase D — Validate and report

**Step 14. Test the genetic evidence for association and colocalisation.**
Input: GWAS summary statistics and eQTL data.
Output: association estimates, colocalisation posterior, and a statement of assumptions.
Rationale: association is not colocalisation, and colocalisation is not causation.
Branch: if colocalisation fails, the target hypothesis is weakened.
Edge case: linkage disequilibrium and pleiotropy can produce colocalisation without mediation.

**Step 15. Validate in an independent dataset.**
Input: the discovery result and a second dataset.
Output: replication with the same effect direction and a compatible magnitude.
Rationale: an internal split is not an independent cohort.
Branch: if no independent dataset exists, label the result as discovery only.
Edge case: a replication in the same tissue but a different platform is partial validation.

**Step 16. Evaluate batch-corrected and uncorrected results.**
Input: the analysis with and without integration.
Output: both results with a comparison.
Rationale: aggressive integration can erase disease signal.
Branch: if the corrected result removes the biology, the correction is too strong.
Edge case: a method that improves clustering can worsen effect estimation.

**Step 17. Score the gates and record the evidence.**
Input: the outputs of steps 1 to 16.
Output: a gate table with evidence and residual uncertainty.
Rationale: the decision must be reviewable.
Branch: a zero on any gate is a veto for the claim as written.
Edge case: a resource can be admissible for a descriptive claim and inadmissible for a causal one.

**Step 18. Re-run on release, reference, or annotation change.**
Input: the release notes and the pipeline versions.
Output: an updated gate record with a new date.
Rationale: omics resources change and annotations are revised.
Branch: re-run when the resource release, the reference build, the annotation, or the pipeline changes.
Edge case: a portal can change the default release without changing the URL.

### Regime decision table

| Claim | Regime | Unit of inference | Minimum design |
|---|---|---|---|
| Tissue-level differential expression | Bulk | Donor or sample | Cases and controls with batch balance |
| Cell-type-specific expression | Single-cell or single-nucleus | Donor for population; cell for description | Multiple donors per group |
| Cell-type proportion difference | Single-cell | Donor | Compositional analysis with denominators |
| Spatial organisation | Spatial | Section or donor | Multiple sections per group |
| Genetic association | Germline | Individual | Adequate sample and ancestry control |
| Causal target evidence | Germline plus eQTL | Individual | Colocalisation and sensitivity analysis |
| Drug response | Cell line or organoid | Line | Replication across lines |
| Protein abundance | Proteomics | Sample | Controls and batch balance |

### Open and controlled layers

| Resource | Open layer | Controlled layer | What the controlled layer adds |
|---|---|---|---|
| TCGA | Clinical, expression, somatic variants | Germline variants and some raw sequence | Inherited variation and raw reads |
| GTEx | Processed expression and eQTL | Raw sequence and full donor metadata | Reanalysis and genotype-level work |
| UK Biobank | Summary-level results, some aggregate | Individual-level data in UKB-RAP | Participant-level analysis |
| CELLxGENE | Processed atlas and metadata | Some raw data under separate agreements | Reprocessing from reads |
| Tabula Sapiens | Processed objects under CC BY 4.0 | Raw FASTQ under a data use agreement | Alignment and quantification control |
| GEO | Per-series processed and raw | Some series restricted | Full metadata and controlled samples |
| dbGaP | Study-level metadata | Individual-level data | Participant-level analysis |

### Batch confounding patterns

| Pattern | Identification | Consequence | Action |
|---|---|---|---|
| Batch equals condition | All cases in batch 1, controls in batch 2 | Effect and batch not separable | Do not use for the claim |
| Batch imbalanced | Cases mostly in one batch | Partial confounding | Model batch and report sensitivity |
| Site equals condition | One site contributes all cases | Site and disease confounded | Restrict or use a multi-site dataset |
| Time equals condition | Cases collected in a later period | Temporal and disease confounded | Model time or restrict |
| Depth differs by group | Cases sequenced deeper | Apparent expression difference | Downsample or model depth |
| Tissue quality differs | Cases from biopsies, controls from resections | Quality and disease confounded | Match quality or adjust |

### Claim-to-evidence table

| Claim | Minimum evidence | Common shortcut | Why the shortcut fails |
|---|---|---|---|
| Gene is differentially expressed | Donor-level test with batch control | Cell-level p-value | Pseudoreplication |
| Variant affects expression | eQTL with adequate sample | Nearest-gene annotation | Distance is not mechanism |
| Variant and disease share a signal | Colocalisation | Separate significant variants | Linkage disequilibrium |
| Target is causal | Genetics plus perturbation or trial | Colocalisation alone | Pleiotropy and mediation |
| Cell type is expanded in disease | Donor-level composition | Atlas cell counts | Sampling composition |
| Drug response is predicted | Replication across cell lines | One cell line | Line-specific artefact |
| Protein is a biomarker | Independent cohort | Discovery only | Overfitting |

### Edge cases that change the procedure

**Licence forbids commercial use.** GTEx processed data are released under CC BY 4.0 with attribution; TCGA open-tier data have their own terms; controlled layers are research-use. A commercial target programme must check each layer. A CC BY licence permits commercial use with attribution, while a data use agreement may not.

**Dataset withdrawn.** Controlled access can be terminated for consent or governance reasons. Stop new work, confirm deletion, and re-run the source selection.

**Credentialing takes months.** dbGaP and UK Biobank can take weeks to months, and UK Biobank applications were paused with an intention to reopen in late 2026. A timeline shorter than the access path fails Gate 3.

**Version drift.** GTEx v8 has 17,382 samples and 948 donors; later releases have different counts. A pipeline pinned to one release can break on another. Record the release and the reference build.

**Label errors discovered.** Cell-type annotations and sample labels can be wrong. Re-audit a sample, report the error, and test sensitivity. Do not relabel only the test set.

**Benchmark contamination.** For omics, contamination often means cell-line misidentification, reused datasets, or a model pretrained on the test cohort. Check the cell-line authentication and the dataset overlap.

## 6. Worked example

### The case

A 60-person biotechnology company wants to nominate a first-in-class target for ulcerative colitis (UC). The nomination needs human genetic support, evidence that the target is expressed in the relevant colon cell types, and a plausible causal direction. The company has 18 months to a development candidate and intends to license the programme.

**Decision statement.** "Nominate a target for UC with human genetic evidence that the target's expression or function is causal in the disease, so that the company can commit to a 12-month validation programme; a false nomination costs a year of preclinical work, and a missed target costs the programme."

**Claim type.** The programme needs causal evidence. Association, colocalisation, and expression are supporting evidence, not proof. The final claim will be "genetically supported target," with the evidence grade stated.

**Tissue and context.** Colon mucosa, inflamed and non-inflamed, adult patients with UC, compared with non-IBD controls.

### Candidate long list

| Candidate | Custodian | Version inspected | Unit | Scale | Access and licence | Why it is on the list |
|---|---|---|---|---|---|---|
| GTEx | Broad and NIH | v8, with later releases | Donor, tissue | 17,382 samples; 948 donors; 54 tissues | Processed open; raw controlled | eQTL resource for colon |
| UK Biobank | UK Biobank | Current release | Participant | 500,000 participants | Controlled, fee, UKB-RAP | Genetic association with EHR-derived UC |
| CELLxGENE and Tabula Sapiens | CZI and consortium | Current collections | Cell, donor | Millions of cells | Processed open per collection | Colon cell-type expression |
| TCGA | NCI | Current GDC release | Patient, sample | About 11,000 patients; 33 cancer types | Open and controlled | Multi-omic methods, not UC |
| GEO | NCBI | Continuous | Sample, series | Hundreds of thousands of samples | Per-series terms | Published UC expression studies |
| Open Targets | Consortium | Current release | Target-disease pair | Release-dependent | Open with attribution | Integrated evidence and prioritisation |

### Gate scores for the causal nomination

| Gate | GTEx | UK Biobank | CELLxGENE | TCGA | GEO | Open Targets |
|---|---:|---:|---:|---:|---:|---:|
| 1 Decision fit | 1 | 2 | 1 | 0 | 1 | 1 |
| 2 Population fit | 1 | 2 | 1 | 0 | 1 | 1 |
| 3 Access fit | 2 | 0 | 2 | 2 | 2 | 2 |
| 4 Licence fit | 2 | 1 | 2 | 1 | 1 | 2 |
| 5 Benchmark fit | 1 | 1 | 0 | 1 | 0 | 1 |
| Veto | none | access | benchmark | decision, population | benchmark | none |

**GTEx, Gate 1 and Gate 2.** GTEx provides colon expression and eQTLs in donors without inflammatory bowel disease. It does not contain a UC cohort, so it cannot estimate a disease-specific effect. It supports the expression and regulatory architecture, not the disease association. Scores 1 and 1.

**GTEx, Gate 4.** Processed data are released under CC BY 4.0 with attribution. Raw sequence and full donor metadata are controlled through dbGaP and AnVIL. The processed layer is sufficient for an eQTL look-up and not for reanalysis from reads. Score 2 for the processed layer.

**UK Biobank, Gate 1 and Gate 2.** The cohort has genetic data and EHR-derived diagnoses, which can support a UC case-control association. It is the strongest candidate for the genetic claim. Scores 2 and 2.

**UK Biobank, Gate 3.** Access requires an application, institutional approval, a fee, and analysis inside the approved platform. Applications were paused as of 26 October 2026 with an intention to reopen in late 2026. For an 18-month programme, the access route is not confirmed. Score 0, veto.

**CELLxGENE and Tabula Sapiens, Gate 5.** The atlases support cell-type expression and composition description. They do not contain a UC case-control design, and their composition reflects tissue sampling. They cannot support a disease-comparison claim. Score 0 on benchmark fit for the causal claim.

**TCGA, Gate 1.** The cohort is cancer, not inflammatory bowel disease. Score 0, veto.

**GEO, Gate 5.** Published UC studies exist and are heterogeneous in platform, tissue, and processing. A meta-analysis is possible, and the datasets are not a uniform benchmark. Score 0 for a pre-specified causal benchmark, 1 for supporting evidence.

**Open Targets, Gate 1.** The platform integrates genetic, expression, pathway, and safety evidence. It is an evidence layer, not a dataset, and its scores are not probabilities. Score 1, admissible as a prioritisation tool.

### The two diagnostics

**Diagnostic 1: the eQTL resource has no disease tissue.** GTEx colon samples come from donors without inflammatory bowel disease, and the per-tissue sample size limits the power to detect small eQTL effects. An eQTL that is significant in normal colon may not replicate in inflamed mucosa, because cell composition, treatment, and inflammatory signalling change. The dataset can support "this variant regulates this gene in normal colon" and cannot support "this variant changes the gene's effect in UC." The gap is a disease-context gap, not a sample-size gap alone.

**Diagnostic 2: the atlas is not a sampling frame.** A single-cell atlas of the colon contains cells from a small number of donors, selected by tissue availability. The proportion of a cell type in the atlas reflects the donors and the sampling, not the population. A claim that a cell type is expanded in UC requires a case-control design with donor-level composition and a denominator. The atlas can generate the hypothesis and cannot test it.

A third diagnostic is genetic confounding. A variant associated with both UC and the target's expression may do so through linkage disequilibrium with a third variant, or through pleiotropy. Colocalisation reduces the first risk and does not remove the second. The programme needs a sensitivity analysis, not a single posterior probability.

### The selected route

The admissible set is GTEx and Open Targets, with UK Biobank as a conditional source if access reopens. The plan uses GTEx for expression and eQTL evidence, Open Targets for integrated prioritisation, public GWAS summary statistics for association, and a primary wet-lab validation for causality.

| Track | Steps | Owner | Duration | Planning cost |
|---|---|---|---|---|
| Evidence integration | Open Targets and GTEx look-up | Bioinformatics | 6 weeks | Staff time |
| Genetic association | Public GWAS summary statistics and colocalisation | Statistical genetics | 8 weeks | Staff and compute |
| UK Biobank application | Prepare and submit; monitor reopening | Legal and data lead | 6 to 12 months | Fee plus staff |
| Cell-type expression | CELLxGENE and Tabula Sapiens | Bioinformatics | 4 weeks | Staff time |
| Wet-lab validation | Organoid or primary cell perturbation | Discovery biology | 6 to 9 months | Six figures |
| Nomination review | Evidence grading and go/no-go | Programme board | 4 weeks | Internal |

### Time and cost

| Item | Detail | Calendar time | Planning cost |
|---|---|---|---|
| Open evidence integration | GTEx, Open Targets, GWAS | 8 weeks | Staff time |
| UK Biobank application | Preparation and review | 6 to 12 months | Tier fee and staff |
| Cell-type analysis | Atlas look-up | 4 weeks | Staff time |
| Wet-lab validation | Organoid perturbation | 6 to 9 months | Six figures |
| Total to nomination | Parallel tracks | About 12 months | Dominated by validation |

The access route is not the critical path if the programme relies on public GWAS. It becomes the critical path if the programme requires individual-level UK Biobank analysis.

### Substitute plan if UK Biobank access does not reopen

If the application cannot be submitted or approved in time, the plan changes in three ways:

1. The genetic claim uses published GWAS summary statistics and colocalisation with GTEx eQTLs, with a lower evidence grade.
2. The target nomination is labelled "genetically supported" rather than "causally validated," and the wet-lab programme carries more of the burden.
3. The UK Biobank application remains open as a later validation track, and the programme does not delay the nomination for it.

The fallback is scientifically weaker and faster. The decision record should state the evidence grade explicitly rather than presenting the two routes as equivalent.

### Vary one assumption: the company requires commercial redistribution

Suppose the programme is not a internal target nomination but a licensed target package sold to partners. The licence analysis changes:

- GTEx processed data under CC BY 4.0 can be used commercially with attribution, but the derived target package may need to carry the attribution.
- Open Targets data have their own attribution terms.
- UK Biobank commercial access requires a commercial licence and a fee, and the outputs may be subject to review.
- The package cannot redistribute controlled data; it can redistribute derived evidence with the licences checked.

The science does not change. The commercial route determines which evidence can be included in the package and which must be summarized.

If the variation is size, a smaller eQTL reference would weaken the colocalisation evidence but not change the design. If the variation is label quality, a misclassified UC diagnosis in the EHR would bias the association toward the null and require a validation sample.

### Conditions under which the case would fail

- **No colon eQTL for the target.** The regulatory evidence is absent, and the target falls to a lower grade.
- **The donor count is too small for colocalisation.** The posterior is uninformative, and the genetic claim weakens.
- **The GWAS is underpowered or ancestry-limited.** The association does not replicate, and the nomination is premature.
- **The wet-lab model does not express the target.** The perturbation cannot test the hypothesis.
- **The commercial licence excludes the target package.** The evidence can be generated and not redistributed.

### Gate evidence log

| Gate | Candidate | Evidence inspected | Verdict | Residual uncertainty |
|---|---|---|---|---|
| 1 | GTEx | Colon expression and eQTLs; no IBD cohort | 1 | Disease-context gap |
| 1 | UK Biobank | Genetic data and EHR-derived diagnoses | 2 | Diagnosis coding and cohort selection |
| 2 | GTEx | Post-mortem donors without IBD | 1 | Not the target population |
| 2 | UK Biobank | Population cohort | 2 | Healthy-volunteer and ancestry bias |
| 3 | GTEx | Open processed download | 2 | Raw layer controlled |
| 3 | UK Biobank | Application paused; fee | 0 | Reopening date unknown |
| 4 | GTEx | CC BY 4.0 processed; controlled raw | 2 | Attribution and raw restrictions |
| 4 | UK Biobank | Commercial access terms | 1 | Fee and output review |
| 5 | GTEx | No disease benchmark | 1 | Colocalisation is not causation |
| 5 | CELLxGENE | Atlas composition, no case-control | 0 | Cannot test disease difference |
| 5 | Open Targets | Integrated evidence, not experiment | 1 | Scores are not probabilities |

### Decision timeline

| Month | Evidence track | Access track | Validation track | Deliverable |
|---:|---|---|---|---|
| 1 | Open Targets and GTEx look-up | Draft UK Biobank application | Select organoid model | Candidate list |
| 2 | GWAS summary statistics | Institutional approval | Assay development | Association results |
| 3 | Colocalisation | Submit application if open | Pilot perturbation | Evidence table |
| 4-5 | Cell-type expression | Monitor reopening | Validation batch 1 | Ranked targets |
| 6-8 | Sensitivity analyses | Access or fallback | Validation batch 2 | Shortlist |
| 9-12 | Evidence grading | Data analysis if granted | Validation readout | Nomination |

### What the case demonstrates

- The strongest disease cohort was unavailable, and the open resources could support expression and regulatory evidence but not the disease association on their own.
- The atlas was rich in cells and empty of the comparison the claim needed. Cell count is not cohort design.
- The eQTL resource answered a normal-tissue question. The disease-context gap was the diagnostic, not the sample size.
- Colocalisation reduced one threat and did not remove pleiotropy. The evidence grade had to reflect that.
- The commercial route changed which evidence could be redistributed, not which evidence could be generated.
- The fallback was a lower evidence grade, and the decision record had to say so.

### Sensitivity of the decision

| Change | Gate affected | Does the source change? | Plan response |
|---|---|---|---|
| UK Biobank reopens next month | 3 | Yes | Submit and add individual-level analysis |
| A public UC GWAS with adequate power exists | 1 and 5 | No | Strengthen the genetic evidence |
| The atlas adds a UC case-control collection | 1 and 5 | Possibly | Add a donor-level composition analysis |
| The company requires a licensed package | 4 | No | Restrict redistribution to derived evidence |
| The target has no eQTL in colon | 1 | No | Lower the evidence grade or drop the target |
| A wet-lab validation fails | 5 | No | Do not nominate; record the negative result |

## 7. Code, query, or calculation

The three blocks below use the Python standard library only. They implement an access-layer inventory, a multiple-testing calculation, and a donor-level permutation test.

### 7.1 Open and controlled layers

```python
ACCESS = {
    "GTEx processed":        ("open", "CC BY 4.0 for processed data; attribution", "minutes"),
    "GTEx raw sequence":     ("controlled", "dbGaP and AnVIL; research use", "weeks to months"),
    "TCGA clinical/RNA":     ("open", "GDC open tier; check terms", "hours"),
    "TCGA germline":         ("controlled", "dbGaP; institutional certification", "months"),
    "UK Biobank":            ("controlled", "application, fee, UKB-RAP", "weeks to months"),
    "CELLxGENE atlas":       ("open", "per-collection licence; often CC BY 4.0", "minutes"),
    "Tabula Sapiens raw":    ("controlled", "data use agreement; no redistribution", "weeks"),
    "GEO series":            ("open", "per-series terms", "minutes"),
    "Open Targets":          ("open", "attribution; release version", "minutes"),
}

print(f"{'Resource':<24}{'Tier':>12}  {'Latency':<18}Terms")
for name, (tier, terms, latency) in ACCESS.items():
    print(f"{name:<24}{tier:>12}  {latency:<18}{terms}")
print()
open_count = sum(1 for t, _, _ in ACCESS.values() if t == "open")
print(f"Open resources: {open_count}/{len(ACCESS)}")
print("Open processed data does not imply open raw data or open commercial rights.")
```

Expected output:

```text
Resource                        Tier  Latency           Terms
GTEx processed                  open  minutes           CC BY 4.0 for processed data; attribution
GTEx raw sequence         controlled  weeks to months   dbGaP and AnVIL; research use
TCGA clinical/RNA               open  hours             GDC open tier; check terms
TCGA germline             controlled  months            dbGaP; institutional certification
UK Biobank                controlled  weeks to months   application, fee, UKB-RAP
CELLxGENE atlas                 open  minutes           per-collection licence; often CC BY 4.0
Tabula Sapiens raw        controlled  weeks             data use agreement; no redistribution
GEO series                      open  minutes           per-series terms
Open Targets                    open  minutes           attribution; release version

Open resources: 5/9
Open processed data does not imply open raw data or open commercial rights.
```

The inventory is the first artefact of an omics access review. It separates what can be downloaded from what must be applied for, and it prevents the common assumption that a public portal is a public licence.

### 7.2 Multiple-testing arithmetic

```python
tests = 20000
alpha = 0.05
bonf = alpha / tests
expected_fp = tests * alpha

print(f"Tests: {tests:,}")
print(f"Nominal alpha: {alpha}")
print(f"Bonferroni threshold: {bonf:.2e}")
print(f"Expected false positives at nominal alpha: {expected_fp:,.0f}")
print()
p_values = [0.000001, 0.00001, 0.0004, 0.002, 0.01, 0.03, 0.2, 0.6]
print(f"{'p-value':>12}{'Bonferroni significant':>24}")
for p in p_values:
    print(f"{p:>12.6f}{str(p < bonf):>24}")
print()
print("With 20,000 genes, a nominal 0.05 threshold yields about 1,000 false positives")
print("before any true signal is counted. Multiple-testing control is not optional.")
```

Expected output:

```text
Tests: 20,000
Nominal alpha: 0.05
Bonferroni threshold: 2.50e-06
Expected false positives at nominal alpha: 1,000

     p-value  Bonferroni significant
    0.000001                    True
    0.000010                   False
    0.000400                   False
    0.002000                   False
    0.010000                   False
    0.030000                   False
    0.200000                   False
    0.600000                   False

With 20,000 genes, a nominal 0.05 threshold yields about 1,000 false positives
before any true signal is counted. Multiple-testing control is not optional.
```

The example uses Bonferroni for clarity. A false-discovery-rate procedure is often more powerful, and it still requires the test count and the hypothesis set to be stated. The point is the arithmetic, not the choice of correction.

### 7.3 Donor-level permutation test

```python
from itertools import combinations

group_a = [8.1, 7.4, 8.8]   # donor-level pseudobulk means
group_b = [6.9, 7.2, 6.4]
observed = sum(group_a) / len(group_a) - sum(group_b) / len(group_b)
pooled = group_a + group_b
n = len(group_a)
extreme = 0
total = 0
for idx in combinations(range(len(pooled)), n):
    a = [pooled[i] for i in idx]
    b = [pooled[i] for i in range(len(pooled)) if i not in idx]
    diff = sum(a) / len(a) - sum(b) / len(b)
    total += 1
    if abs(diff) >= abs(observed) - 1e-12:
        extreme += 1

print(f"Observed difference: {observed:.2f}")
print(f"Permutations: {total}")
print(f"Two-sided permutation p-value: {extreme/total:.3f}")
print()
print("With three donors per group, the smallest attainable two-sided p-value is 0.10.")
print("No donor-level claim can reach conventional significance at this sample size.")
```

Expected output:

```text
Observed difference: 1.27
Permutations: 20
Two-sided permutation p-value: 0.100

With three donors per group, the smallest attainable two-sided p-value is 0.10.
No donor-level claim can reach conventional significance at this sample size.
```

The test is exact for this design and shows why a cell-level p-value can be misleading. With three donors per group there are only 20 assignments, and the smallest two-sided p-value is 0.10. No amount of additional cells changes that.

## 8. Data, standards, and regulatory context

Omics data are governed by metadata standards, controlled-access frameworks, and genetic privacy law. The table separates the binding from the conventional.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| FAIR principles | Voluntary, widely adopted | Findability, accessibility, interoperability, reusability | Guarantee quality or licence clarity |
| MIAME | Community standard | Microarray experiment metadata | Cover sequencing-specific fields |
| MINSEQE | Community standard | Sequencing experiment metadata | Enforce a licence |
| GA4GH standards | Community and policy | Data sharing, identifiers, consent codes | Replace institutional review |
| dbGaP data use certification | Binding for controlled data | Access, use, security, renewal, close-out | Grant commercial rights |
| DUOS | Community platform | Consent codes and data use ontology | Automate approval |
| GDPR | Binding in the EU | Personal data, including genetic data | Define a research standard |
| GINA | Binding in the US | Genetic discrimination in insurance and employment | Govern research access |
| HIPAA | Binding in the US | PHI handling | Cover de-identified genomic data uniformly |
| CELLxGENE schema | Community standard | Cell metadata fields | Guarantee biological correctness |
| HCA metadata schema | Consortium standard | Sample and library metadata | Enforce across all datasets |
| HGVS | Community standard | Variant nomenclature | Resolve transcript ambiguity |
| GRCh38 | Reference genome | Coordinates | Cover all species or builds |

**Open data is not open consent.** A processed matrix can be downloadable and still be governed by a consent scope that limits commercial use or disease-specific research. The licence and the consent are separate.

**Genetic data are special in law.** GDPR treats genetic data as a special category, and GINA restricts insurance and employment use in the United States. A commercial programme that uses germline data needs a lawful basis, a transfer mechanism, and a review of the downstream use.

**Metadata standards are not analysis standards.** MINSEQE describes what was sequenced and how; it does not prescribe normalisation, batch correction, or differential testing. Two compliant datasets can be analysed in incompatible ways.

### Data sharing frameworks

| Framework | Purpose | Binding? | Limit |
|---|---|---|---|
| GA4GH consent codes | Standardise data use permissions | Community | Not a law |
| dbGaP data use certification | Bind institutions for controlled access | Contractual | US-focused |
| DUOS | Match consent codes to requests | Community | Depends on depositor |
| Data Access Committee | Review requests | Governance | Variable timelines |
| Material transfer agreement | Govern physical samples | Contractual | Not data-specific |
| Data use agreement | Govern data access | Contractual | Varies by custodian |
| Privacy impact assessment | Assess privacy risk | Legal in some jurisdictions | Not a licence |
| Institutional review board | Review human-subjects research | Legal in some settings | Not commercial permission |

A programme can satisfy one framework and fail another. The review should list each instrument and its status.

### Metadata fields that determine reuse

| Field | Why it matters | Common omission |
|---|---|---|
| Organism and tissue | Determines relevance | Ontology term missing |
| Disease state | Determines the comparison | "Healthy" undefined |
| Treatment | Confounds expression | Not recorded |
| Donor identifier | Enables donor-level inference | Replaced by sample ID |
| Batch | Confounds processing | Not recorded |
| Library protocol | Determines compatibility | Protocol version absent |
| Sequencing platform | Determines batch and bias | Model not recorded |
| Read length | Determines alignment | Missing |
| Reference build | Determines coordinates | Missing |
| Annotation version | Determines gene models | Missing |
| Consent scope | Determines permitted use | Not in the data card |
| Cell ontology term | Determines annotation comparability | Free-text labels only |

### Consent codes and their consequences

| Consent element | What it permits | What it blocks | How to check |
|---|---|---|---|
| General research use | Broad research | Some commercial uses if excluded | Consent text |
| Disease-specific | Research on a named disease | Other diseases | Consent scope |
| Commercial use | Product development | Nothing beyond the scope | Explicit clause |
| Redistribution | Sharing with collaborators | Re-release without approval | Data use agreement |
| Return of results | Reporting findings to participants | Incidental findings if excluded | Consent and protocol |
| Recontact | Asking participants for more data | Follow-up without approval | Consent |
| Linkage | Joining to other data | Linkage without permission | Governance approval |
| Withdrawal | Removing data on request | Continued use after withdrawal | Withdrawal procedure |

Consent and licence answer different questions. A participant can consent to broad research while the custodian licenses the data for non-commercial use only. Both must be checked.

### Regulatory artefacts for an omics programme

- **Data management plan** names the resources, the access tiers, and the retention.
- **Consent and lawful basis** describe what participants agreed to and the legal ground.
- **Data use certification** binds the institution for controlled access.
- **Transfer mechanism** covers cross-border processing.
- **Security plan** covers storage, access, and incident response.
- **Publication plan** covers review, attribution, and small-cell rules.
- **Close-out plan** covers deletion and derived artefacts.

A programme that can download the matrix and cannot document the consent has an incomplete file.

## 9. Industry practice and case evidence

### What organisations actually do

- **Separate discovery from validation.** Open atlases generate hypotheses; controlled cohorts and wet-lab experiments test them.
- **Use donor-level pseudobulk.** Mature single-cell teams aggregate before population claims.
- **Record reference versions.** Build, annotation, and pipeline versions are part of the result.
- **Grade evidence explicitly.** Association, colocalisation, perturbation, and trial evidence are labelled separately.
- **Check cell-line identity.** Misidentified lines are a recurring source of false discovery.
- **Plan controlled access early.** dbGaP and cohort applications sit on the critical path.
- **Report corrected and uncorrected results.** Integration can erase biology.

### Case evidence

| Case | Documented fact | Inference |
|---|---|---|
| GTEx v8 | The release contains 17,382 RNA-seq samples from 948 donors across 54 tissues, with 15,201 samples and 838 donors in the eQTL analysis; processed data are open and raw data are controlled. | A large normal-tissue resource supports regulatory and expression evidence and not a disease-specific effect. |
| GTEx later releases | The portal lists later releases with different sample and donor counts, and a later annotation version without new samples. | A pipeline must pin the release and the annotation; a URL is not a version. |
| Tabula Sapiens | v1 contains roughly 483,000 to 500,000 cells from 24 tissues and 15 donors; v2 expands to about 1.1 million cells from 28 tissues and 24 donors; processed data are CC BY 4.0 and raw data require a data use agreement. | A multi-organ atlas supports cell-state description and not population prevalence. |
| TCGA | The programme spans about 11,000 patients across 33 cancer types, with open clinical and expression layers and controlled germline data. | A cancer cohort supports cancer biology and not a general-population claim. |
| DepMap | The dependency map uses about 1,000 to 1,200 cancer cell lines to nominate dependencies. | Cell lines are models of cancer and not tumours; in vitro artefacts are common. |
| Open Targets | The platform integrates genetic, expression, pathway, and safety evidence and releases periodically. | Integrated evidence supports prioritisation and is not experimental validation. |

The first three rows are release-specific and should be checked against the current portal. The structural conclusions, that tissue context and donor count limit the claim, are stable.

### Platform effects to expect

| Platform | Characteristic | Confounder | Check |
|---|---|---|---|
| Illumina short-read | Dominant sequencing platform | Batch and depth | Run and lane |
| Long-read | Structural variants and isoforms | Error profile | Platform-specific caller |
| Microarray | Older expression studies | Probe annotation | Platform version |
| Proteomics mass spec | Protein abundance | Instrument and run | Batch and calibration |
| Spatial | Position and expression | Section and slide | Section-level batch |
| Single-cell droplet | High throughput | Droplet batch | Chemistry version |
| Single-cell plate | Lower throughput | Plate batch | Plate layout |
| Metabolomics | Small molecules | Instrument drift | Run order |

### Evidence grades for a target

| Grade | Evidence | What it supports | What it does not |
|---|---|---|---|
| A | Human genetics plus perturbation plus clinical direction | Causal target hypothesis | Efficacy in patients |
| B | Human genetics plus colocalisation plus expression | Strong genetic support | Causation without perturbation |
| C | Human genetics association only | Candidate target | Direction or mechanism |
| D | Expression or pathway only | Hypothesis | Genetic support |
| E | Cell-line dependency only | In vitro dependency | Tumour biology |

The grades are a communication device, not a regulatory standard. Their value is that they force the team to say which evidence exists and which is missing.

### What good looks like

- The donor count is reported before the cell count.
- Batches are balanced or the confounding is stated.
- The reference build and annotation are pinned.
- Multiple testing is controlled with a stated test count.
- Cell types are annotated against a versioned reference.
- The licence and consent are checked separately.
- Discovery and validation cohorts are independent.
- The evidence grade is stated in the nomination memo.

### Why target discovery is an evidence-grading problem

A target nomination is not a single result. It is a portfolio of evidence: human genetics, expression in the right cell type, direction of effect, safety, and tractability. Each piece has a different failure mode. A programme that treats a colocalisation posterior as proof, or a DepMap dependency as tumour biology, will over-grade the target. The discipline is to label each piece and to state what would falsify the nomination.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Pseudoreplication | p-values computed across cells | Cells treated as independent | False positives and overstated power | Donor-level pseudobulk |
| Batch confounded with disease | Cases and controls processed separately | Effect and batch not separable | Uninterpretable result | Batch-balanced design |
| Over-integration | Disease signal disappears after correction | Correction removes biology | False negative | Report corrected and uncorrected |
| Atlas composition claim | Cell-type proportion compared across datasets | Sampling determines composition | False disease association | Case-control design with donors |
| eQTL treated as causation | A significant eQTL is called a target | Association is not mechanism | Over-graded target | Colocalisation and perturbation |
| Colocalisation treated as proof | A high posterior is called causal | Pleiotropy and LD remain | False target | Sensitivity analysis |
| Multiple testing ignored | Nominal p-values reported genome-wide | 1,000 false positives expected | Noise reported as discovery | Bonferroni or FDR |
| Reference drift | Gene IDs change after an update | Annotation version changed | Broken joins and changed results | Pin build and annotation |
| Cell-line misidentification | Results do not replicate | Wrong line or contamination | False discovery | Authentication and STR typing |
| GEO heterogeneity | Studies pooled without harmonisation | Different platforms and protocols | Confounded meta-analysis | Harmonise or stratify |
| Open assumed commercial | CC BY and research DUA conflated | Licence not read | Product blocked | Clause review |
| Controlled access ignored | Raw data used without approval | Access tier misunderstood | Compliance breach | Layer mapping |
| Small donor count | Three donors per group | No attainable significance | Unsupported claim | Report effect sizes only |
| Benchmark reuse | Same atlas used for training and test | Dataset overlap | Inflated performance | Independent validation |
| Ambient RNA | False expression in low-quality cells | Cell-free transcripts | Spurious markers | Ambient correction |
| Doublets | Apparent intermediate cell state | Two cells in one droplet | False cell type | Doublet detection |
| Empty droplets | Low-count "cells" | Droplet algorithm | Noise as biology | Empty-droplet filtering |
| Sample swap | Genotype does not match metadata | Labeling error | Wrong group assignment | Genotype-sex and fingerprint checks |
| Ancestry confounding | Association tracks ancestry | Population structure | Spurious association | Principal components and stratified analysis |
| Winner's curse | Effect sizes shrink on replication | Discovery selection | Over-optimistic target | Independent replication |
| Data reuse | Same cohort across papers | Publication pressure | Inflated evidence | Cohort provenance audit |

Red flags that should stop an omics project:

- The unit of inference is a cell and the claim is about people.
- The batch is perfectly confounded with the condition.
- The donor count is not reported.
- The reference build and annotation are not recorded.
- The licence is "open" with no clause.
- The target evidence is a single colocalisation posterior.
- The cell line identity is not authenticated.
- The validation cohort is the discovery cohort renamed.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The donor key is correct | Donor-level inference depends on it | Samples relabeled | Reconcile metadata |
| The tissue is comparable | Expression comparison depends on it | Biopsy versus resection | Match quality and site |
| The batch is balanced | Effect identification depends on it | Cases in one batch | Batch-by-condition table |
| The annotation is correct | Cell-type claims depend on it | Reference lacks the type | Marker validation |
| The reference is pinned | Reproducibility depends on it | Annotation update | Version record |
| The normalisation is appropriate | Comparisons depend on it | Compositional bias | Sensitivity analysis |
| The multiple-testing control is adequate | Discovery depends on it | Test count understated | Report count and method |
| The genetic instrument is valid | Causal inference depends on it | Pleiotropy | Sensitivity analysis |
| The cell line is authenticated | Cell biology depends on it | Misidentification | STR profiling |
| The validation is independent | Generalisation depends on it | Same cohort reused | Cohort provenance |
| The licence covers the use | Product depends on it | Research-only terms | Clause review |
| The consent covers the use | Governance depends on it | Narrow consent | Consent review |
| Ambient RNA is corrected | Cell-type markers depend on it | Contamination | Ambient correction |
| Doublets are removed | Cell states depend on it | False intermediates | Doublet detection |
| Samples are not swapped | Group assignment depends on it | Labeling error | Genotype checks |
| Ancestry is modelled | Association validity depends on it | Population structure | Principal components |
| Effect sizes are replicated | Target grading depends on it | Winner's curse | Independent cohort |

The deepest limit is that omics association is not clinical utility. A target with genetic support can fail in the clinic, and a biomarker with a strong hazard ratio can fail to change decisions. Those are prospective questions.

## 12. Exercises

**Exercise 1. Classify claims by regime.**
Deliverable: a table of six claims mapped to bulk, single-cell, spatial, germline, or proteomics, with the unit of inference.
Acceptance criteria: six claims; each unit justified; one claim rejected as unsupported by the regime.
Time: 35 minutes.

**Exercise 2. Identify the units in a matrix.**
Deliverable: a one-page audit of a public omics dataset naming the donor, sample, cell, and gene units and the count at each level.
Acceptance criteria: four units; counts reconciled; one pseudoreplication risk named.
Time: 45 minutes.

**Exercise 3. Run a donor-level test.**
Deliverable: a pseudobulk matrix and a permutation test with the attainable p-value.
Acceptance criteria: code reproduces the section 7 output; the donor count and the limitation are stated.
Time: 50 minutes.

**Exercise 4. Apply multiple-testing control.**
Deliverable: a table of nominal and adjusted thresholds for a stated test count, with the expected false positives.
Acceptance criteria: arithmetic correct; test count justified; method named.
Time: 30 minutes.

**Exercise 5. Map access layers.**
Deliverable: a table of open, controlled, and unavailable layers for four resources, with the obligations of each.
Acceptance criteria: four resources; each layer sourced; one commercial restriction identified.
Time: 45 minutes.

**Exercise 6. Grade genetic evidence.**
Deliverable: a one-page evidence grade for a named target, separating association, colocalisation, and perturbation.
Acceptance criteria: three evidence types; assumptions stated; a falsification test proposed.
Time: 45 minutes.

**Exercise 7. Detect batch confounding.**
Deliverable: a batch-by-condition table and a sensitivity analysis plan.
Acceptance criteria: table complete; collinearity identified; a correction and an uncorrected comparison proposed.
Time: 40 minutes.

**Assessment note.** Exercises 2, 3, 4, and 5 produce checkable artefacts: a unit audit, a permutation test, a threshold table, and an access map. Exercises 1, 6, 7, and 8 are design tasks and are marked against the rubric. Exercise 8 is the integration task.

**Exercise 8. Write an omics data card.**
Deliverable: a two-page card with resource, release, build, annotation, unit, access, licence, and prohibited claims.
Acceptance criteria: every field populated or marked unknown; one prohibited claim; versions recorded.
Time: 60 minutes.

## 13. Self-check answers

**Item 1. Why are bulk and single-cell different measurement regimes?**
Bulk measures an average over a tissue sample; single-cell measures a distribution over cells. Bulk supports tissue-level inference; single-cell supports cell-state description and, with donor-level aggregation, composition claims. A deconvolved bulk proportion is a model output, not an observed cell count.

**Item 2. What is the unit of inference for a population claim?**
The donor. Cells and samples are observations within a donor, and treating them as independent inflates the effective sample size.

**Item 3. Why is a cell-level p-value misleading?**
It treats thousands of cells from a few donors as independent replicates. The p-value reflects the number of cells, not the number of people, and it can be arbitrarily small with enough cells.

**Item 4. What does a Bonferroni threshold of 2.5e-6 mean at 20,000 tests?**
It is 0.05 divided by 20,000. A p-value below it keeps the family-wise error rate at 5 percent across the 20,000 tests under the null. It is conservative when tests are correlated.

**Item 5. What is the difference between an eQTL and a causal target?**
An eQTL is an association between a variant and expression. A causal target requires evidence that changing the target changes the disease, through colocalisation, perturbation, or a trial. An eQTL is supporting evidence.

**Item 6. Why can colocalisation be misleading?**
Two traits can share a signal because of linkage disequilibrium or pleiotropy, not because one mediates the other. Colocalisation reduces the chance of distinct causal variants and does not establish direction or mechanism.

**Item 7. What is batch confounding?**
A technical grouping that is correlated with the condition. If all cases are processed together and all controls together, the disease effect and the batch effect cannot be separated.

**Item 8. Why can over-integration be harmful?**
Batch correction can remove biological variation along with technical variation. A corrected dataset that no longer separates cases from controls may have removed the disease signal.

**Item 9. Why is an atlas not a sampling frame?**
Its donors and tissues are selected by availability and study design, not by population sampling. Cell-type proportions reflect the samples, not the population.

**Item 10. What does an open processed matrix permit?**
It permits what its licence says. A CC BY 4.0 matrix permits commercial use with attribution; a research-use matrix does not. Open download is not a licence.

**Item 11. Why is a cell-line dependency not a tumour dependency?**
A cell line is an in vitro model selected for growth, with genetic drift, no microenvironment, and culture-specific dependencies. A dependency in a line can be an artefact of the culture or the line's genotype. It is a hypothesis, not tumour biology.

**Item 12. What is the difference between a reference build and an annotation version?**
The build is the genome assembly that defines coordinates; the annotation defines gene and transcript models on that build. Both change, and both must be recorded. A join on gene identifiers can break after an annotation update.

**Item 13. Why is consent separate from licence?**
Consent records what a participant agreed to; a licence records what the custodian permits. A participant may consent to broad research while the custodian licenses only non-commercial use. Both constrain the programme.

**Item 14. What does an enrichment result not tell you?**
It does not tell you which gene in the set is causal, whether the set is coordinately regulated, or whether the direction of change is meaningful. Enrichment is a hypothesis about a set, not a mechanism.

**Item 15. Why is a polygenic score not a diagnostic test?**
It summarises many small genetic effects and typically has modest discrimination. Its performance depends on the ancestry and environment of the training population, and it does not incorporate clinical measurements. It is a risk factor, not a diagnosis.

**Item 16. What is winner's curse, and how do you avoid it?**
The tendency of discovery effect sizes to be inflated because only large effects pass a threshold. It is avoided by independent replication and by reporting effect sizes with confidence intervals rather than only significant hits.

**Item 17. Why does the number of tests depend on the filtering step?**
Filtering removes genes or features before testing, which changes the family of hypotheses. A threshold computed before filtering is not the same as one computed after. Report both the filter and the test count.

**Item 18. What is the difference between a reference panel and a validation cohort?**
A reference panel provides haplotypes for imputation or annotation; a validation cohort provides independent samples for testing a result. Confusing them means the validation is not independent.

**Rubric for the design exercises (2, 3, 5, 8).** Full marks require a named unit, a version record, a licence statement, and a prohibited-claim statement. Partial marks for a complete structure without a version record. No marks for a cell-level claim presented as a population result.

## 14. Interview questions

**1. How do you test differential expression in single-cell data?**
Aggregate to donor or sample, then test at the donor level; use cell-level tests only for description. A strong answer explains pseudoreplication. A weak answer tests each cell.

**2. Why is a million cells from six donors not a million-sample study?**
The donor is the independent unit. Six donors give six observations for a population claim, regardless of cell count. A strong answer states the effective sample size. A weak answer cites the cell count.

**3. What is the difference between an eQTL and a drug target?**
An eQTL is a regulatory association; a target requires causal evidence. A strong answer names colocalisation, perturbation, and direction. A weak answer treats the eQTL as proof.

**4. How do you handle batch effects?**
Balance the design where possible, model batch, and report corrected and uncorrected results. A strong answer notes that correction can remove biology. A weak answer integrates everything by default.

**5. What does GTEx not contain?**
Disease cohorts, clinical outcomes, and inflamed tissue. A strong answer names the normal-tissue limitation. A weak answer treats GTEx as a disease resource.

**6. How do you choose between TCGA and GTEx?**
By the question: cancer biology and tumour multi-omics for TCGA, normal-tissue regulation for GTEx. A strong answer notes that neither is a population cohort. A weak answer chooses on size.

**7. What is the difference between open and controlled omics data?**
Open data are downloadable under a licence; controlled data require institutional approval and a data use agreement. A strong answer notes that processed data can be open while raw data are controlled. A weak answer treats a portal as a single access tier.

**8. How do you validate a target nomination?**
Independent genetic evidence, orthogonal expression, perturbation in a relevant model, and a clear falsification criterion. A strong answer separates evidence types. A weak answer relies on one colocalisation result.

**9. Why is a cell line not a tumour?**
It is an in vitro model selected for growth, often with genetic drift and no microenvironment. A strong answer notes that dependencies can be culture-specific. A weak answer treats DepMap as tumour biology.

**10. What would you refuse to claim from an atlas?**
Population prevalence, disease-specific expansion, and causal effects. A strong answer names the design limitation. A weak answer reports composition as a population fact.

**11. How do you report a negative omics result?**
Report the effect size, the confidence interval, the donor count, and the test count. A strong answer distinguishes absence of evidence from evidence of absence. A weak answer reports only a p-value.

**12. What belongs in an omics data card?**
Resource, release, build, annotation, unit, sample and donor counts, access tier, licence, processing, and prohibited claims. A strong answer records versions. A weak answer lists the file size.

**13. How do you handle a dataset with no donor identifier?**
Treat the donor as unknown, use sample-level inference, state the limitation, and do not claim population generalization. A strong answer seeks a linkage or a substitute. A weak answer treats samples as donors.

**14. What is the first check on an omics matrix?**
The unit and the dimensions: genes by samples or genes by cells, and the metadata behind each column. Then the donor count and the batch field. A strong answer checks the reference build before analysis. A weak answer starts with a volcano plot.

**15. How do you report a target nomination to a scientific committee?**
Evidence grade, genetic association, colocalisation, expression context, perturbation plan, safety risks, and the falsification criterion. A strong answer separates evidence types. A weak answer presents a single score.

**16. When would you stop an omics programme?**
When the donor count cannot support the claim, the batch is confounded, the licence blocks the use, or the target has no independent validation path. A strong answer treats stopping as a result. A weak answer adds more cells.

## 15. Key terms

- **Annotation version** — the gene or transcript model release used.
- **Batch** — a group of samples processed together.
- **Bulk sequencing** — average signal across a tissue sample.
- **Cell ontology** — controlled vocabulary for cell types.
- **Colocalisation** — evidence that two traits share a causal variant.
- **Compositional data** — proportions that sum to a constant.
- **Controlled tier** — data requiring institutional approval.
- **dbGaP** — US controlled-access genomic archive.
- **DepMap** — cancer cell-line dependency map.
- **Differential expression** — association between condition and expression.
- **Donor** — the individual from whom samples were taken.
- **eQTL** — variant associated with gene expression.
- **False discovery rate** — expected proportion of false positives among discoveries.
- **GENCODE** — gene annotation resource.
- **Germline** — inherited genetic variation.
- **gnomAD** — aggregate variant frequency resource.
- **GTEx** — normal-tissue expression and eQTL resource.
- **Human Cell Atlas** — international single-cell consortium.
- **Multiple testing** — testing many hypotheses at once.
- **Open tier** — data available without individual approval.
- **Open Targets** — integrated target-disease evidence platform.
- **Pseudobulk** — aggregating cells to a sample or donor.
- **Pseudoreplication** — treating non-independent units as independent.
- **Reference build** — the genome assembly used for alignment.
- **Single-cell RNA-seq** — transcript counts per cell.
- **Somatic** — acquired genetic variation.
- **Spatial transcriptomics** — expression with position.
- **Tabula Sapiens** — multi-organ human single-cell atlas.
- **TCGA** — cancer genome atlas.
- **UK Biobank** — population cohort with genomics and health records.
- **Association** — a statistical relationship without a causal claim.
- **Cell state** — a transcriptional program within a cell type.
- **Coding variant** — a variant that changes a protein sequence.
- **Coverage** — the proportion of a reference covered by reads.
- **Differential abundance** — a difference in cell-type proportion.
- **Effect size** — the magnitude of an association.
- **Enrichment** — over-representation of a gene set.
- **Expression quantitative trait locus** — see eQTL.
- **Fine mapping** — narrowing a causal variant within a locus.
- **Genome-wide association study** — variant-trait association at genome scale.
- **Genomic inflation** — systematic inflation of association statistics.
- **Haplotype** — a set of variants inherited together.
- **Imputation** — inferring unobserved genotypes from a reference panel.
- **Linkage disequilibrium** — correlation between nearby variants.
- **Mendelian randomisation** — causal inference using genetic instruments.
- **Normalisation** — scaling counts for comparability.
- **Pleiotropy** — a variant affecting multiple traits.
- **Polygenic score** — a weighted sum of variant effects.
- **Principal component** — a dimension of variation, often ancestry.
- **Proteomics** — large-scale protein measurement.
- **Read depth** — the number of reads at a position.
- **Reference panel** — a set of haplotypes used for imputation.
- **Regulatory variant** — a variant affecting gene regulation.
- **Single-nucleus RNA-seq** — nuclear transcript measurement.
- **Splice QTL** — a variant associated with splicing.
- **Transcript** — an RNA product of a gene.
- **Variant call format** — a text format for genetic variants.
- **Whole-exome sequencing** — sequencing of coding regions.
- **Whole-genome sequencing** — sequencing of the whole genome.
- **Ambient RNA** — cell-free transcripts contaminating a library.
- **Doublet** — a droplet containing two cells.
- **Empty droplet** — a droplet with no cell.
- **Feature count** — reads assigned to a gene or region.
- **Library size** — total reads in a sample.
- **Quantification** — estimating abundance from reads.
- **Sample swap** — a mismatch between sample identity and metadata.
- **Splice junction** — a boundary between exons.
- **Transcript abundance** — the amount of a transcript.
- **Winner's curse** — inflated effect sizes in discovery.
- **Count matrix** — genes by samples or cells.
- **Depth** — the number of reads covering a feature.
- **Feature selection** — choosing variables for a model.
- **Latent variable** — an unobserved factor such as batch.
- **Metadata** — descriptive fields about samples.
- **Pipeline version** — the recorded configuration of an analysis.
- **Pseudotime** — an inferred ordering of cell states.
- **Read** — a sequenced fragment.
- **Sparsity** — the proportion of zero counts.
- **UMAP** — a dimensionality-reduction method for visualisation.
- **Variance stabilisation** — transforming counts to reduce mean-variance dependence.

## 16. Sources

All URLs accessed 26 October 2026. Releases and counts change; verify at the portal.

1. GTEx Portal. V8 release announcement. https://gtexportal.org/home/home/news?id=360
2. GTEx Portal. Release history. https://www.gtexportal.org/home/releaseInfoPage
3. GTEx Portal. Datasets and protected access. https://www.gtexportal.org/home/datasets
4. National Human Genome Research Institute. Genotype-Tissue Expression Project. https://www.genome.gov/Funded-Programs-Projects/Genotype-Tissue-Expression-Project
5. National Cancer Institute. TCGA. https://www.cancer.gov/ccg/research/genome-sequencing/tcga
6. Genomic Data Commons. https://portal.gdc.cancer.gov/
7. gnomAD. https://gnomad.broadinstitute.org/
8. UK Biobank. https://www.ukbiobank.ac.uk/
9. CELLxGENE Discover. https://cellxgene.cziscience.com/
10. Human Cell Atlas. https://www.humancellatlas.org/
11. Tabula Sapiens. Data access. https://tabula-sapiens.sf.czbiohub.org/whereisthedata
12. Tabula Sapiens Consortium. The Tabula Sapiens: a multiple-organ, single-cell transcriptomic atlas of humans. *Science* 2022;376:eabl4896. https://doi.org/10.1126/science.abl4896
13. NCBI GEO. https://www.ncbi.nlm.nih.gov/geo/
14. EMBL-EBI PRIDE. https://www.ebi.ac.uk/pride/
15. EMBL-EBI MetaboLights. https://www.ebi.ac.uk/metabolights/
16. DepMap. https://depmap.org/
17. Open Targets. https://www.opentargets.org/
18. ChEMBL. https://www.ebi.ac.uk/chembl/
19. GA4GH. Data use and interoperability standards. https://www.ga4gh.org/
20. EMBL-EBI. ChEMBL licence. https://www.ebi.ac.uk/chembl/
21. NCBI. dbGaP. https://www.ncbi.nlm.nih.gov/gap/
22. Broad Institute. AnVIL. https://anvilproject.org/
23. UK Biobank. Fees. https://www.ukbiobank.ac.uk/use-our-data/fees/
24. Global Alliance for Genomics and Health. https://www.ga4gh.org/
25. Module 21 survey file, section 8. [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)

## 17. Links

- Module map and topic table: [`_index.md`](_index.md)
- Previous topic, signal datasets: [`05_signal-datasets.md`](05_signal-datasets.md)
- Access and licensing: [`02_access-and-licensing.md`](02_access-and-licensing.md)
- Imaging datasets: [`04_imaging-datasets.md`](04_imaging-datasets.md)
- Root survey file: [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
- Labs, including the dataset audit: [`../../labs/`](../../labs/)

