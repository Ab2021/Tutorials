# Module 12 — Omics and Precision Medicine

> **Program:** Biomedical Data Science for Industry (2026)
> **Time:** 32 hours (12 lecture, 4 case clinic, 16 lab)
> **Track fit:** Biopharma & Translational core; elective for others
> **Prerequisites:** Modules 04–07; basic molecular biology helpful
> **Tools:** Python, R/Bioconductor, scanpy, Seurat, DESeq2, PLINK, Open Targets, PyTorch
> **Industry question:** "Is this molecular signal a validated biomarker, a causal target, or an artifact of batch, ancestry, and tissue composition?"

---

## 1. Why industry cares

Omics data promise precision medicine: the right treatment for the right patient at the right time. They also produce some of the highest-dimensional, noisiest, most confounded datasets in biomedicine.

Industry uses omics for:

- Target discovery and validation.
- Biomarker discovery.
- Patient stratification.
- Drug response prediction.
- Safety and toxicity.
- Companion diagnostics.
- Clinical trial enrichment.
- Agricultural and industrial biotechnology.

The gap between a significant association and a validated product is enormous. This module teaches the data science and the evidence needed to cross it.

## 2. Learning outcomes

By the end of this module, you can:

- Explain the central dogma and the major omics data types.
- Process and quality-control bulk RNA-seq, single-cell, spatial, and variant data.
- Identify and correct batch effects without erasing biology.
- Perform differential expression, pathway analysis, and multiple-testing correction.
- Build and evaluate a polygenic risk score with ancestry considerations.
- Annotate variants and interpret them in a clinical context.
- Integrate multi-omics data for discovery.
- Evaluate a target or biomarker for causal and clinical plausibility.
- Design a discovery-to-validation plan.
- Communicate genomic uncertainty and ethical issues.

## 3. Mental model: from molecule to decision

```
Genome → Transcriptome → Proteome → Metabolome → Phenotype → Population
   │          │             │            │           │
Variants   Expression     Protein      Metabolite   Disease
   │          │             │            │           │
   └──────────┴─────────────┴────────────┴───────────┘
                         Integration
                              │
                    Target / biomarker
                              │
                 Analytical validation
                              │
                 Clinical validation
                              │
                    Clinical utility
```

Each arrow is a validation step, not an assumption.

## 4. Data types

| Data type | Measures | Common platforms | Key challenges |
|---|---|---|---|
| Genomics | DNA sequence, variants | Illumina, PacBio, ONT | Coverage, alignment, variant calling |
| Epigenomics | DNA methylation, chromatin | Bisulfite, ATAC-seq | Batch, coverage, cell composition |
| Transcriptomics | RNA abundance | Bulk RNA-seq, microarray | Normalization, isoform, deconvolution |
| Single-cell | Per-cell expression | 10x, Smart-seq | Sparsity, doublets, integration |
| Spatial | Expression + location | Visium, Xenium, CosMx | Resolution, alignment, segmentation |
| Proteomics | Protein abundance | Mass spec, Olink, Somascan | Missingness, normalization, isoforms |
| Metabolomics | Small molecules | LC-MS, GC-MS | Identification confidence, batch |
| Microbiome | Microbial composition | 16S, metagenomics | Compositionality, contamination |
| Multi-omics | Multiple layers | Integrated assays | Alignment, missing modalities |

## 5. Genomics

### 5.1 Sequencing pipeline

```
Sample → Library → Sequencing → FASTQ → QC → Alignment → BAM → Variant calling → VCF → Annotation
```

Key quality metrics:

- Read quality (Q30).
- Coverage depth.
- Duplication rate.
- Mapping rate.
- Contamination.
- Insert size.

### 5.2 Variant annotation

- **HGVS:** nomenclature for variants.
- **VEP / ANNOVAR:** functional annotation.
- **ClinVar:** clinical interpretations.
- **gnomAD:** population allele frequency.
- **COSMIC:** somatic variants.
- **ACMG/AMP:** variant classification framework.

### 5.3 Variant classification

ACMG categories:

- Pathogenic.
- Likely pathogenic.
- Uncertain significance (VUS).
- Likely benign.
- Benign.

**Industry lesson:** a VUS is not a diagnosis. Reporting must include uncertainty and context.

### 5.4 GWAS and PRS

- **GWAS:** association between variants and traits.
- **PRS:** weighted sum of risk alleles.
- **Ancestry:** PRS performance often degrades across ancestries because discovery cohorts are Euro-centric.
- **Clinical use:** PRS may add value for some conditions, but context and counseling are essential.

### 5.5 Cancer genomics

- Somatic mutations.
- Copy-number alterations.
- Structural variants.
- Mutational signatures.
- Tumor mutational burden.
- Microsatellite instability.
- Driver vs passenger mutations.
- Clonal evolution.

## 6. Transcriptomics

### 6.1 Bulk RNA-seq

Pipeline:

```
FASTQ → QC → Alignment/quantification → Count matrix → Normalization → DE → Pathway analysis
```

Normalization methods:

- CPM/TPM/FPKM for expression.
- DESeq2 median-of-ratios for differential expression.
- TMM for cross-sample comparisons.

**Pitfall:** TPM is not appropriate for differential expression without accounting for composition.

### 6.2 Differential expression

- Negative binomial model (DESeq2, edgeR).
- Limma-voom.
- Covariates: batch, sex, age, tissue, cell composition.
- Multiple-testing correction (BH FDR).
- Effect size and confidence interval.
- Pathway enrichment (GSEA, ORA).

### 6.3 Cell-type deconvolution

Bulk tissue is a mixture. Deconvolution estimates cell-type proportions, which can confound disease associations.

Tools: CIBERSORTx, MuSiC, BayesPrism.

## 7. Single-cell

### 7.1 Pipeline

```
FASTQ → Cell Ranger → Count matrix → QC → Normalization → Feature selection → PCA → Integration → Clustering → Annotation → Analysis
```

### 7.2 Quality control

- Cells with low genes/UMIs.
- High mitochondrial fraction.
- Doublets.
- Ambient RNA.
- Empty droplets.

**Caution:** thresholds are dataset-specific. Over-filtering can remove real populations.

### 7.3 Normalization and integration

- Library-size normalization.
- Log transformation.
- Pearson residuals.
- SCTransform.
- Harmony, scVI, BBKNN for integration.
- Batch correction must preserve biological variation.

### 7.4 Clustering and annotation

- Leiden/Louvain clustering.
- Marker genes.
- Reference-based annotation.
- Manual review.
- Differential abundance.

### 7.5 Trajectory and perturbation

- Pseudotime.
- RNA velocity.
- Cell-cell communication.
- Perturb-seq.
- CRISPR screens.

## 8. Spatial transcriptomics

- Spots (Visium) or cells (Xenium, CosMx).
- Spatial domains.
- Cell-type deconvolution.
- Ligand-receptor interactions.
- Niche analysis.
- Alignment with histology.

**Challenges:** resolution, segmentation, batch, tissue orientation, and limited gene panels.

## 9. Proteomics and metabolomics

- Mass spectrometry: identification and quantification.
- Missingness: not missing at random; often below detection limit.
- Normalization: median, quantile, probabilistic.
- Batch effects.
- Identification confidence.
- Pathway and network analysis.

## 10. Multi-omics integration

| Strategy | Description | Strength | Weakness |
|---|---|---|---|
| Early | Concatenate features | Simple | Dimensionality, scaling |
| Late | Combine predictions | Robust | Ignores interactions |
| Intermediate | Shared latent space | Rich | Complex |
| Network | Graph-based | Biological structure | Construction bias |
| Bayesian | Probabilistic | Uncertainty | Compute |

Methods: MOFA, DIABLO, iCluster, similarity network fusion.

**Rule:** integration must be justified by a biological question, not by the availability of data.

## 11. Drug discovery

### 11.1 Target identification

- Genetic evidence (GWAS, Mendelian randomization).
- Expression evidence.
- Pathway evidence.
- Animal models.
- Literature.
- Open Targets.

### 11.2 Target validation

- CRISPR screens.
- Perturbation experiments.
- Patient tissue.
- Replication.
- Safety.

### 11.3 Lead discovery

- Virtual screening.
- Molecular docking.
- Generative chemistry.
- ADMET prediction.
- Protein structure (AlphaFold).
- Assay validation.

### 11.4 Clinical translation

- Biomarker strategy.
- Patient selection.
- Dose selection.
- Safety monitoring.
- Companion diagnostic.

## 12. Industry case studies

### 12.1 TCGA and cancer subtyping

The Cancer Genome Atlas characterized molecular subtypes across cancers, influencing diagnosis and trial design.

**Lesson:** multi-omics discovery requires validation in independent cohorts and clinical trials.

Source: The Cancer Genome Atlas. https://www.cancer.gov/ccg/research/genome-sequencing/tcga

### 12.2 UK Biobank and polygenic risk

UK Biobank enabled large-scale genetic discovery and PRS development. Performance varies across ancestries.

**Lesson:** PRS is not equally transportable; report ancestry-specific performance and avoid clinical use without validation.

Source: UK Biobank. https://www.ukbiobank.ac.uk/

### 12.3 AlphaFold

AlphaFold predicted protein structures with high accuracy, accelerating structural biology.

**Lesson:** structure prediction is not drug discovery; validation and integration with experiments remain essential.

Source: Jumper, J. et al., *Nature*, 2021. https://doi.org/10.1038/s41586-021-03819-2

### 12.4 Single-cell foundation models

Geneformer and scGPT learn representations from large single-cell corpora.

**Lesson:** benchmarks can show gains, but biological validation and reproducibility are essential. A 2025 benchmark found that zero-shot scGPT and Geneformer underperformed simpler methods such as scVI and Harmony on cell-type clustering and batch integration; larger pretraining did not automatically help.

Sources: Theodoris, C. V. et al., *Nature*, 2023. https://doi.org/10.1038/s41586-023-06139-9; Cui, H. et al., *Nature Methods*, 2024. https://doi.org/10.1038/s41592-024-02201-0; *Genome Biology*, 2025. https://doi.org/10.1186/s13059-025-03574-x

### 12.5 CAGI and variant interpretation

CAGI assesses computational methods for genome interpretation against experimental data.

**Lesson:** community benchmarks reveal where methods generalize and where they fail.

Source: https://genomeinterpretation.org/

## 13. Labs

### Lab 12.1 — Variant annotation

**Objective:** Annotate variants and classify evidence.

**Steps:**

1. Load a VCF.
2. Annotate with VEP or a public database.
3. Filter by population frequency.
4. Compare with ClinVar.
5. Classify using ACMG criteria.
6. Write a report with uncertainty.

**Deliverable:** variant report.

**Acceptance criteria:** correct nomenclature; evidence cited; VUS handled appropriately.

### Lab 12.2 — Bulk RNA-seq differential expression

**Objective:** Identify differentially expressed genes.

**Steps:**

1. Load a count matrix.
2. Normalize and inspect QC.
3. Fit a model with covariates.
4. Apply FDR correction.
5. Run pathway enrichment.
6. Visualize results.

**Deliverable:** DE report.

**Acceptance criteria:** normalization appropriate; covariates justified; FDR reported.

### Lab 12.3 — Single-cell clustering

**Objective:** Cluster and annotate PBMCs.

**Steps:**

1. QC and filter.
2. Normalize and select features.
3. PCA and integration.
4. Cluster.
5. Annotate with markers.
6. Check batch effects.

**Deliverable:** annotated dataset.

**Acceptance criteria:** QC documented; integration justified; markers supported.

### Lab 12.4 — Spatial deconvolution

**Objective:** Estimate cell-type composition in spatial data.

**Steps:**

1. Load spatial data and reference.
2. Deconvolve spots.
3. Map spatial domains.
4. Analyze ligand-receptor interactions.
5. Validate with histology if available.

**Deliverable:** spatial analysis.

**Acceptance criteria:** reference appropriate; limitations stated.

### Lab 12.5 — PRS portability

**Objective:** Build and evaluate a PRS across ancestry groups.

**Steps:**

1. Obtain summary statistics.
2. Clump and threshold.
3. Compute scores in a target cohort.
4. Evaluate AUC by ancestry.
5. Discuss clinical and ethical implications.

**Deliverable:** PRS report.

**Acceptance criteria:** ancestry-specific performance; no overclaim.

### Lab 12.6 — Multi-omics integration

**Objective:** Integrate two omics layers.

**Steps:**

1. Preprocess each layer.
2. Try early and late integration.
3. Evaluate predictive performance.
4. Interpret latent factors.
5. Compare with single-omics baselines.

**Deliverable:** integration report.

**Acceptance criteria:** baselines; biological interpretation; limitations.

## 14. Assignment: biomarker evidence plan

Write a 2,000-word plan for a candidate biomarker:

1. Biological rationale.
2. Data and assay.
3. Discovery cohort.
4. Validation cohort.
5. Statistical analysis.
6. Confounders and batch.
7. Clinical utility.
8. Regulatory pathway.
9. Ethical issues.
10. Go/no-go criteria.

## 15. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Biology | 20% | Generic | Correct mechanisms | Connects to clinical decision |
| Data processing | 20% | Default pipeline | QC and normalization justified | Batch and composition handled |
| Statistics | 20% | p-values only | Effect sizes and FDR | Sensitivity and validation |
| Validation | 20% | Discovery only | Independent cohort | Clinical utility |
| Ethics | 10% | Ignores | Identifies issues | Ancestry and privacy plan |
| Communication | 10% | Technical | Clear | Translational |

## 16. Regulatory, ethical, and commercial lens

- **CLIA/CAP:** clinical lab tests require analytical and clinical validation.
- **FDA:** companion diagnostics and some genomic tests are devices; laboratory-developed tests have a complex regulatory history.
- **IVDR:** EU in-vitro diagnostic regulation.
- **Genetic privacy:** GINA in the US; insurance and employment concerns elsewhere.
- **Ancestry:** PRS and variant interpretation can widen disparities.
- **Incidental findings:** genomic data can reveal unexpected risk; consent and counseling are essential.
- **Commercial:** biomarker tests need reimbursement and clinical utility evidence.

## 17. Failure modes and red flags

- No batch correction.
- Over-correction that removes biology.
- p-values without effect sizes.
- Multiple testing ignored.
- Discovery cohort only.
- PRS applied across ancestries without validation.
- VUS reported as pathogenic.
- Single-cell clusters over-interpreted.
- Spatial deconvolution without reference validation.
- Multi-omics integration without a question.

## 18. Interview and discussion questions

1. **What is the central dogma?**
   Hint: DNA → RNA → protein, with regulation and feedback.
2. **Why is batch correction necessary?**
   Hint: technical variation can dominate biology.
3. **How do you validate a biomarker?**
   Hint: analytical, technical, clinical, utility.
4. **What is a VUS?**
   Hint: variant of uncertain significance; not a diagnosis.
5. **Why is PRS transportability a problem?**
   Hint: Euro-centric discovery, differing linkage disequilibrium.
6. **How do you handle single-cell doublets?**
   Hint: detection tools, filtering, reporting.
7. **What is spatial transcriptomics?**
   Hint: expression with spatial location.
8. **How would you prioritize a drug target?**
   Hint: genetic, expression, pathway, safety, tractability.
9. **What is the difference between bulk and single-cell?**
   Hint: tissue average vs per-cell resolution.
10. **How do you correct for multiple testing?**
    Hint: BH FDR, Bonferroni, permutation.
11. **What are the ethical risks of genomic data?**
    Hint: privacy, discrimination, incidental findings, ancestry.
12. **How do you integrate multi-omics data?**
    Hint: early, late, intermediate; question-driven.

## 19. Key terms

| Term | Definition |
|---|---|
| GWAS | Genome-wide association study |
| PRS | Polygenic risk score |
| VCF | Variant call format |
| VUS | Variant of uncertain significance |
| eQTL | Expression quantitative trait locus |
| Bulk RNA-seq | Tissue-level RNA sequencing |
| scRNA-seq | Single-cell RNA sequencing |
| UMI | Unique molecular identifier |
| Doublet | Two cells captured in one droplet |
| Spatial transcriptomics | Expression with location |
| Proteomics | Large-scale protein measurement |
| Metabolomics | Large-scale metabolite measurement |
| Multi-omics | Integration of multiple molecular layers |
| Batch effect | Technical variation across batches |
| Deconvolution | Estimating cell-type composition |

## 20. Sources and further reading

### Data and tools
- TCGA: https://www.cancer.gov/ccg/research/genome-sequencing/tcga
- UK Biobank: https://www.ukbiobank.ac.uk/
- Open Targets: https://www.opentargets.org/
- ChEMBL: https://www.ebi.ac.uk/chembl/
- ClinVar: https://www.ncbi.nlm.nih.gov/clinvar/
- gnomAD: https://gnomad.broadinstitute.org/
- scANVI/scvi-tools: https://scvi-tools.org/
- scanpy: https://scanpy.readthedocs.io/
- Seurat: https://satijalab.org/seurat/
- Bioconductor: https://bioconductor.org/
- Open Problems: https://openproblems.bio/
- CAGI: https://genomeinterpretation.org/

### Methods and evidence
- Jumper, J. et al., "Highly accurate protein structure prediction with AlphaFold," *Nature*, 2021. https://doi.org/10.1038/s41586-021-03819-2
- Theodoris, C. V. et al., "Transfer learning enables predictions in network biology," *Nature*, 2023. https://doi.org/10.1038/s41586-023-06139-9
- Cui, H. et al., "scGPT: toward building a foundation model for single-cell multi-omics using generative AI," *Nature Methods*, 2024. https://doi.org/10.1038/s41592-024-02201-0
- Richards, S. et al., "Standards and guidelines for the interpretation of sequence variants," *Genetics in Medicine*, 2015. https://doi.org/10.1038/gim.2015.30
- Torkamani, A. et al., "The personal and clinical utility of polygenic risk scores," *Nature Reviews Genetics*, 2018. https://doi.org/10.1038/s41576-018-0018-x

### Guidance
- FDA, *In Vitro Diagnostics*. https://www.fda.gov/medical-devices/in-vitro-diagnostics
- CLIA: https://www.cms.gov/medicare/quality/clinical-laboratory-improvement-amendments
- EU IVDR: https://eur-lex.europa.eu/eli/reg/2017/746/oj
- NIH All of Us: https://allofus.nih.gov/

> **Next:** [Module 13](13_biopharma_clinical_development.md) connects these molecular signals to the pharma development pipeline.
