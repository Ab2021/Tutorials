# Omics Data

> Module 02 — The Biomedical Data Landscape
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: none within this module; basic statistics assumed
> Version 2026.3

---

## 1. Why this topic exists

Omics data measures molecules at scale: DNA sequence, RNA abundance, protein abundance, metabolite concentration, chromatin state, and microbial composition. It is the data class most often described as "high-dimensional" and the one where the gap between data volume and inferential value is widest.

The gap exists because omics measures a molecular state in a small number of samples, often with substantial technical variation, and the question of interest is usually about a clinical outcome in a much larger population. A study with 60 tumours and 60 normals can generate 20,000 gene-level tests and still be underpowered to answer a clinical question. The arithmetic of that situation is the core of this topic.

The commercial stakes:

- A biomarker discovery programme that reports 600 differentially expressed genes without multiple-testing control or batch assessment will spend a year validating noise.
- A target identification team that ignores ancestry composition in a genomic dataset will develop a target that may not generalize.
- A diagnostics company that validates a classifier on data processed in one batch will fail at transfer to a clinical laboratory with a different pipeline.
- A data engineer who converts FASTQ to a count matrix without recording the reference genome and pipeline version has created a dataset that cannot be reproduced or compared.

The decision this topic supports: whether an omics dataset can support a discovery, validation, or regulatory claim, what its batch structure and ancestry composition imply, and what must be recorded for the result to be reproducible.

### 1.1 Who is blocked, and by what

| Role | Decision blocked | Failure without this topic |
|---|---|---|
| Biomarker lead | Whether a signature is real | No multiple-testing control; unexamined batch |
| Target ID scientist | Whether a target generalizes | Ancestry and tissue composition ignored |
| Translational lead | Whether an omics finding supports a trial | Effect size and power not assessed |
| Diagnostics developer | Whether a classifier will transfer | Pipeline and reference genome not recorded |
| Data engineer | How to store and version omics data | Reference and pipeline versions lost |
| Regulatory lead | Whether the evidence meets a standard | Analytical validation conflated with clinical validation |
| Privacy officer | Whether genomic data can be shared | Genomic data treated as ordinary PHI |

### 1.2 What makes omics different from other clinical data

Four properties distinguish it:

1. **The measurement is relative.** RNA abundance is a proportion of a library, not an absolute quantity. Protein abundance from mass spectrometry is relative to a reference. Almost every omics measure is compositional or relative, which constrains the statistics.
2. **The feature count exceeds the sample count by orders of magnitude.** Twenty thousand genes and 120 samples is normal. This makes multiple testing the central problem, not a detail.
3. **The technical variation is large and structured.** Batch, lane, flow cell, reagent lot, and operator produce effects that can exceed the biological effect.
4. **The data are identifiable in a way most clinical data are not.** A genome is a unique identifier, and it carries information about relatives who did not consent.

Each of these properties changes the analysis, and each is routinely underweighted in commercial settings.

---

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** an omics dataset by assay, data level (raw reads, alignments, counts, calls), and platform, and state what each level supports.
2. **Compute** the expected number of false positives at a nominal threshold and explain why multiple-testing control is mandatory.
3. **Apply** the Benjamini-Hochberg procedure by hand to a small set of p-values and interpret the resulting q-values.
4. **Diagnose** batch confounding and quantify how much of a signature replicates when the confounded batch is removed.
5. **Explain** why genomic data require controlled access even when clinical identifiers are removed.
6. **Critique** an omics discovery paper for power, batch, composition, and ancestry, and propose two sensitivity analyses.
7. **Choose** between raw, processed, and summary omics data for a stated question, with the access and reproducibility consequences.
8. **Design** a provenance record for an omics dataset that names the reference genome, pipeline, and version of every annotation resource.

Mapping to assessment: outcomes 2, 3, and 4 are assessed by exercises 2, 3, and 4. Outcomes 1, 5, and 6 are assessed by the self-check items. Outcomes 7 and 8 are assessed by the design exercises and the module memo.

---

## 3. The core mental model

Omics analysis is a **pipeline that converts a physical sample into a matrix of relative quantities**, and every stage adds a transformation that must be recorded.

```text
  BIOSPECIMEN
      |
      |  extraction, library preparation, sequencing or mass spec
      v
  RAW READOUT  (FASTQ, mzML, IDAT)      <- level 1: raw
      |
      |  alignment / quantification against a REFERENCE
      v
  ALIGNMENTS or PEAK INTENSITIES        <- level 2: processed
      |
      |  summarization to features (genes, proteins, metabolites)
      v
  FEATURE MATRIX (samples x features)   <- level 3: matrix
      |
      |  normalization, filtering, batch correction
      v
  NORMALIZED MATRIX
      |
      |  statistical testing, effect sizes
      v
  CANDIDATE LIST  ->  validation in an independent cohort
```

Three properties of this chain drive the analysis:

- **The reference is a choice.** For DNA and RNA, alignment requires a reference genome and an annotation set. GRCh38 and T2T-CHM13 produce different coordinates; two annotation releases produce different gene sets. A matrix without its reference is not comparable to any other matrix.
- **Normalization is a model.** Counts must be normalized for library size and composition. Every normalization method assumes something about what most features are doing, and that assumption can be wrong in a way that creates false signals.
- **The pipeline is a batch variable.** Two pipelines on the same FASTQ files produce different matrices. A multi-site study with site-specific pipelines has a batch effect that is confounded with site.

### 3.1 The levels of omics data

| Level | Content | Size per sample | What it supports | What it prevents |
|---|---|---|---|---|
| Raw | FASTQ reads, mzML spectra, IDAT intensities | 1-30 GB | Re-alignment, re-quantification, new methods | Nothing, but storage and compute are significant |
| Processed | BAM/CRAM alignments, peak intensities, variant calls | 1-10 GB | Re-summarization, variant re-analysis | Changing the aligner without re-processing |
| Matrix | Counts or normalized values per feature | 1-50 MB | Standard differential analysis | Recovering reads that failed filtering |
| Summary | Effect sizes, signature scores | Kilobytes | Meta-analysis, comparison | Any re-analysis |

The matrix level is where most collaborations operate, and it is the level at which the most provenance is lost. A matrix that arrives without the reference genome, the annotation version, and the normalization method is a table of numbers with an unknown relationship to biology.

### 3.2 The multiple-testing arithmetic

If you test 20,000 genes at a nominal p-value threshold of 0.05 with no true effects, you expect 1,000 false positives. This is not a defect of the method; it is the definition of the threshold.

Two consequences:

- **A list of "significant" genes at p < 0.05 is mostly noise** unless the true effect count is large. The expected false discovery proportion is 1,000 divided by the number of hits.
- **Control changes the question.** Family-wise error control (Bonferroni) asks how likely any false positive is. False discovery rate (FDR) control asks what proportion of the reported list is false. FDR is usually the right question for discovery.

The Benjamini-Hochberg procedure is the standard. Sort p-values ascending, multiply each by the number of tests divided by its rank, and enforce monotonicity. The resulting q-value is an estimate of the false discovery rate at that threshold.

### 3.3 The three confounders that decide reproducibility

1. **Batch.** Samples processed together share technical artefacts. If batch is correlated with the outcome, the biological effect is confounded and no correction fully separates them.
2. **Composition.** A tumour sample contains tumour, stroma, immune cells, and normal tissue in varying proportions. A "tumour signature" may be a cell-composition signature. Bulk RNA-seq cannot distinguish them without deconvolution.
3. **Ancestry.** Genomic associations and polygenic scores derived in one ancestry group transfer imperfectly to others. A dataset with a single ancestry group cannot estimate transfer.

An omics result that does not address these three is a hypothesis, not a finding.

---

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Omics | Measurement of a class of molecules at scale | Used for any high-dimensional data | None |
| Genomics | Study of DNA sequence and variation | Confused with genetics | Assay class |
| Transcriptomics | Measurement of RNA abundance | Treated as protein abundance | Assay class |
| Proteomics | Measurement of protein abundance | Treated as mRNA abundance | Assay class |
| Metabolomics | Measurement of small molecules | Confused with lipidomics | Assay class |
| Epigenomics | Measurement of chromatin and methylation state | Confused with transcriptomics | Assay class |
| Microbiome | Microbial community composition | Confused with a single pathogen | Assay class |
| WGS | Whole genome sequencing | Confused with WES | Assay |
| WES | Whole exome sequencing | Treated as complete genome coverage | Assay |
| FASTQ | Raw read file with base qualities | Treated as an analysis-ready file | Format |
| BAM | Aligned reads, binary | Confused with CRAM | Format |
| CRAM | Compressed aligned reads | Confused with BAM | Format |
| VCF | Variant call format | Treated as a validated variant list | Format |
| Reference genome | The coordinate system for alignment | Assumed universal | GRCh38, T2T-CHM13 |
| Annotation | The gene model used for summarization | Assumed stable | GTF or GFF version |
| Coverage | Reads covering a position | Confused with depth in all regions | Fold |
| Depth | Number of reads at a locus | Confused with breadth | Reads |
| Breadth | Proportion of target covered | Confused with depth | Percent |
| Variant calling | Identification of differences from the reference | Treated as error-free | Pipeline |
| VAF | Variant allele fraction | Confused with zygosity | Fraction |
| Count matrix | Features by samples of counts | Treated as absolute abundance | Counts |
| TPM | Transcripts per million | Treated as comparable across pipelines | Relative |
| FPKM | Fragments per kilobase per million | Treated as comparable across pipelines | Relative |
| Size factor | A library-size normalization factor | Confused with TPM | Ratio |
| Normalization | Adjustment for technical variation | Confused with batch correction | Method |
| Batch effect | Technical variation associated with processing group | Confused with biology | None |
| Batch correction | Removal of estimated batch effects | Assumed to remove confounding | Method |
| Compositional data | Data where features sum to a whole | Analyzed as independent | Property |
| Deconvolution | Estimation of cell-type proportions | Treated as ground truth | Method |
| Doublet | A droplet containing two cells | Confused with a large cell | Count |
| Ambient RNA | RNA from lysed cells in a droplet | Confused with expression | Count |
| H5AD | AnnData file format for single-cell data | Confused with a count matrix | Format |
| UMI | Unique molecular identifier | Confused with a read | Count |
| FDR | False discovery rate | Confused with p-value | Proportion |
| q-value | An estimate of FDR at a threshold | Treated as a p-value | Proportion |
| BH | Benjamini-Hochberg procedure | Applied without checking assumptions | Method |
| Bonferroni | Family-wise error correction | Used for discovery screens | Method |
| Effect size | Magnitude of a difference | Confused with significance | Varies |
| Power | Probability of detecting an effect | Confused with sample size | Proportion |
| dbGaP | NIH database of genotypes and phenotypes | Treated as open data | Repository |
| EGA | European Genome-phenome Archive | Confused with dbGaP | Repository |
| GDC | Genomic Data Commons | Confused with TCGA | Repository |
| DUO | Data Use Ontology | Ignored in access decisions | Ontology |
| GA4GH | Global Alliance for Genomics and Health | Confused with a repository | Standards body |
| ACMG | American College of Medical Genetics and Genomics | Confused with ClinVar | Guidelines body |
| ClinVar | Public variant-disease database | Treated as curated truth | Database |
| GINA | Genetic Information Nondiscrimination Act | Assumed to cover all insurance | US law |

### 4.1 Terms that are routinely conflated

- **Coverage versus depth.** Coverage is the proportion of the target covered at a threshold; depth is the number of reads at a position. A sample can have 30× mean depth and 85 percent breadth at 20×.
- **Normalization versus batch correction.** Normalization addresses library size and composition; batch correction addresses processing group. Applying one does not fix the other.
- **FDR versus p-value.** A p-value is the probability of the data under the null; a q-value is an estimate of the proportion of false discoveries in the selected set. They answer different questions.
- **Germline versus somatic.** Germline variants are inherited and present in every cell; somatic variants are acquired and may be present at low allele fraction. A pipeline calibrated for germline calling will miss low-fraction somatic variants.

---

## 5. The mechanics, step by step

This section is the working procedure for evaluating and using an omics dataset. It has 28 steps in five phases.

### Phase A — Establish the assay and the reference (steps 1-6)

**Step 1. Identify the assay and what it measures.**
Input: the study documentation. Output: an assay classification and a statement of the measured quantity.
Reason: RNA-seq measures RNA abundance, which is a proxy for transcription and a poor proxy for protein. Confusing the two leads to wrong biological claims.
Rule: state what is measured and what is inferred.

**Step 2. Identify the data level available.**
Input: the data package. Output: a level assignment (raw, processed, matrix, summary).
Reason: the level determines what can be re-analyzed. A matrix cannot be re-aligned.
Rule: if only a matrix is available, record the pipeline that produced it.

**Step 3. Record the reference genome and annotation version.**
Input: the documentation. Output: a reference record.
Reason: coordinates and gene models differ between references and releases. A variant at chr1:100000 in GRCh38 is at a different position in GRCh37.
Rule: state the reference build and the annotation release; if either is unknown, the data cannot be compared to external datasets.

**Step 4. Record the pipeline and its version.**
Input: the documentation or the BAM header. Output: a pipeline record.
Reason: aligners, quantifiers, and variant callers produce different outputs from the same reads.
Rule: record tool names, versions, and key parameters, or cite a workflow identifier.

**Step 5. Determine the sample size and the comparison structure.**
Input: the sample table. Output: counts per group and per batch.
Reason: the sample size determines power, and the batch structure determines whether confounding exists.
Rule: build the sample-by-batch table before looking at any result.

**Step 6. Confirm consent and data use terms.**
Input: the consent and the data use agreement. Output: a statement of permitted uses.
Reason: genomic consent is often broad but not unlimited; commercial use may be excluded.
Rule: check the Data Use Ontology (DUO) terms or the equivalent, and do not assume that de-identification permits any use.

### Phase B — Assess quality and confounding (steps 7-12)

**Step 7. Review per-sample quality metrics.**
Input: the quality report. Output: a per-sample pass or flag.
Reason: low-quality samples add noise and can drive batch effects. Excluding them changes the sample size.
Rule: use assay-appropriate metrics (for RNA-seq: mapping rate, duplication, rRNA fraction; for WES: breadth at 20×, contamination; for proteomics: identification rate).

**Step 8. Check for sample swaps and mislabeling.**
Input: genotype or sex-check data. Output: a concordance check.
Reason: sample swaps are common in large studies and produce spectacular false results.
Rule: compare inferred sex to recorded sex and, where available, compare genotypes across samples that should be related or identical.

**Step 9. Build the sample-by-batch table.**
Input: metadata. Output: a contingency table of group by batch.
Reason: this table is the single most important diagnostic. A batch that contains only cases is confounded with the outcome.
Rule: if any batch is missing a group entirely, confounding is complete and correction cannot separate the effects.

**Step 10. Examine the global structure with dimensionality reduction.**
Input: the normalized matrix. Output: a principal component or uniform manifold projection.
Reason: the first components often reflect batch, composition, or ancestry rather than the outcome. Color the plot by each.
Rule: color by group, batch, sex, and ancestry; a component that separates by batch before group is a warning.

**Step 11. Quantify the variance explained by batch.**
Input: the matrix and the metadata. Output: a variance decomposition.
Reason: a quantitative estimate distinguishes a nuisance from a dominant effect.
Rule: report the proportion of variance attributable to batch and to group.

**Step 12. Decide the correction strategy and record it.**
Input: the confounding assessment. Output: a documented decision: include batch as a covariate, correct the matrix, or restrict the analysis.
Reason: each strategy has different assumptions and different costs. Including batch as a covariate preserves the data; correcting the matrix changes it.
Rule: prefer covariate adjustment when batch is not confounded; use correction only when necessary, and report the sensitivity.

### Phase C — Compute and normalize (steps 13-18)

**Step 13. Filter features with a documented rule.**
Input: the matrix. Output: a filtered matrix.
Reason: features with near-zero counts add noise and multiple-testing burden.
Rule: state the filter (for example, at least 10 counts in at least 20 percent of samples) and report how many features were removed.

**Step 14. Normalize for library size and composition.**
Input: the filtered matrix. Output: a normalized matrix.
Reason: samples have different total reads, and a few highly expressed genes can dominate the total.
Rule: state the method (size factors, trimmed mean of M-values, median-of-ratios), and do not compare normalized values across pipelines.

**Step 15. Transform for stability.**
Input: the normalized matrix. Output: a transformed matrix.
Reason: count data are heteroscedastic; a variance-stabilizing transformation improves distance-based methods and visualization.
Rule: record the transformation and its parameters.

**Step 16. Handle zero and missing values explicitly.**
Input: the matrix. Output: a documented missingness rule.
Reason: single-cell data are sparse by nature, and proteomics missingness is often not random.
Rule: distinguish structural zeros (not expressed) from sampling zeros (not detected) from missing (not measured).

**Step 17. Deconvolve cell composition where relevant.**
Input: bulk tissue data. Output: estimated cell-type proportions.
Reason: bulk tissue signals mix cell types. A difference between tumour and normal may be a difference in immune infiltration.
Rule: use a validated reference and report the deconvolution uncertainty.

**Step 18. Freeze the processed matrix and its metadata.**
Input: the pipeline. Output: a versioned matrix plus a sample sheet.
Reason: the matrix is the input to every downstream analysis and must be reproducible.
Rule: store the code, the parameters, the reference, and the sample sheet together.

### Phase D — Test and interpret (steps 19-24)

**Step 19. State the hypothesis and the test before running it.**
Input: the question. Output: a pre-specified model.
Reason: testing many models and reporting the best is a form of selection that inflates false positives.
Rule: write the model formula, including covariates, before looking at results.

**Step 20. Compute effect sizes with confidence intervals.**
Input: the model. Output: an effect estimate per feature.
Reason: significance without magnitude is not actionable. A gene can be significantly different and biologically trivial.
Rule: report the effect size and its interval, not only the p-value.

**Step 21. Apply multiple-testing control and state the procedure.**
Input: the p-values. Output: q-values or adjusted p-values.
Reason: the expected false positive count at a nominal threshold is large, and the reported list must be interpreted with that in mind.
Rule: state the procedure (Benjamini-Hochberg, Storey, Bonferroni) and the threshold.

**Step 22. Test the sensitivity to batch handling.**
Input: the analysis under two batch strategies. Output: a comparison of the resulting lists.
Reason: if the list changes substantially with the batch strategy, the result depends on an assumption.
Rule: report the overlap between lists and the correlation of effect sizes.

**Step 23. Test the sensitivity to composition.**
Input: the analysis with and without deconvolution covariates. Output: a comparison.
Reason: a signal that disappears after adjusting for immune fraction was a composition signal.
Rule: report both and interpret the difference.

**Step 24. Validate in an independent cohort.**
Input: a second dataset. Output: replication rates and effect-size concordance.
Reason: replication is the only credible evidence that a discovery is not batch or overfitting.
Rule: state the replication cohort, its platform, and its pipeline; a replication in the same pipeline is weaker evidence.

### Phase E — Document and share (steps 25-28)

**Step 25. Record every resource version.**
Input: the analysis. Output: a resource manifest.
Reason: reference genomes, annotations, and databases change. A variant interpretation based on ClinVar is only valid for the ClinVar release used.
Rule: list the reference, annotation, gene ontology, ClinVar, and gnomAD versions with dates.

**Step 26. Record the access tier and the data use terms.**
Input: the repository record. Output: an access statement.
Reason: controlled-access data cannot be redistributed, and derived data may inherit restrictions.
Rule: state the repository, the accession, and the permitted uses.

**Step 27. Write the provenance record.**
Input: everything above. Output: a one-page record.
Reason: omics reproducibility depends on versions more than on code style.
Rule: include assay, level, reference, pipeline, normalization, batch strategy, and resource versions.

**Step 28. State what the data cannot answer.**
Input: the question. Output: a limits section.
Reason: omics measures molecules in a sample, not clinical outcomes in a population.
Rule: name at least three limits and say which threatens the conclusion.

### 5.1 Field-level mechanics: DNA sequencing

1. **Library preparation.** Poly-A selection, exome capture, and whole-genome library prep have different biases. Capture kits differ in coverage of GC-rich regions.
2. **Read length and paired-end status.** Short paired reads resolve structural variants poorly; long reads resolve repeats.
3. **Duplication rate.** Polymerase chain reaction duplicates inflate depth without adding information. Marked duplicates should be excluded from variant calling.
4. **Coverage breadth.** Mean depth is misleading; report the proportion of the target at 20× or higher.
5. **Contamination.** Cross-species or cross-sample contamination produces false variants. Estimate it and report it.
6. **Variant quality filters.** QUAL, depth, strand bias, and allele fraction thresholds are choices, and each changes the call set.

### 5.2 Field-level mechanics: RNA sequencing

1. **Library selection.** Poly-A selection enriches mRNA; ribosomal depletion preserves non-coding RNA. The choice determines what "abundance" means.
2. **Strandedness.** Stranded protocols distinguish sense from antisense; unstranded data cannot.
3. **Read depth.** 20 to 30 million reads per sample is typical for differential expression; rare transcript detection needs more.
4. **Gene versus transcript quantification.** Gene-level counting is robust; transcript-level quantification is uncertain because of isoform ambiguity.
5. **Batch.** Flow cell, lane, and library prep date are batch variables and must be recorded.

### 5.3 Field-level mechanics: single-cell

1. **Droplet versus plate.** Droplet methods profile thousands of cells with lower per-cell depth; plate methods profile hundreds with higher depth.
2. **UMI counts.** UMIs distinguish molecules from amplification duplicates; the count is a molecule count, not a read count.
3. **Doublets.** Two cells in one droplet create a hybrid profile. Detection is statistical and imperfect.
4. **Ambient RNA.** Free RNA in the suspension is captured and appears as low-level expression in every cell.
5. **Batch and donor.** Cells from one donor cluster together. Treating cells as independent replicates inflates significance by orders of magnitude.

The last point is the most consequential. A single-cell study with 10 donors and 50,000 cells has 10 biological replicates, not 50,000.

### 5.4 Quantitative checks before analysis

| Check | Computation | Passing signal | What failure means |
|---|---|---|---|
| Group-by-batch table | Counts per group per batch | Every batch has both groups | Complete confounding |
| Variance explained by batch | Variance decomposition | Batch below group, or adjusted | Batch dominates |
| Sample concordance | Genotype or sex check | Matches metadata | Sample swap |
| Library size distribution | Total counts per sample | Within a factor of 3 | Normalization problem |
| Feature detection | Features with nonzero counts | Stable across samples | Filtering or quality issue |
| Effect-size distribution | Histogram of log fold changes | Centered near zero | Normalization bias |
| Replication rate | Overlap with an independent cohort | Above chance, with concordant direction | Overfitting or batch |
| Resource versions | Reference, annotation, databases | All recorded | Irreproducible |

### 5.5 Derivations and the assumptions they hide

| Derived variable | Usual construction | Hidden assumption | Challenge question |
|---|---|---|---|
| Log fold change | Difference of group means on log scale | The normalization is unbiased | Which normalization, and is it stable? |
| q-value | BH on the p-value distribution | Tests are independent or positively dependent | Are the tests correlated by pathway? |
| Signature score | Weighted sum of gene expression | The weights transfer across platforms | Was it validated on this platform? |
| Cell-type proportion | Deconvolution against a reference | The reference matches this tissue | Which reference, which markers? |
| Polygenic score | Sum of weighted alleles | The weights transfer across ancestry | What is the target ancestry? |
| Variant pathogenicity | ACMG criteria applied to evidence | The evidence is complete and current | Which ClinVar and gnomAD releases? |
| Tumor purity | Estimate from copy number or expression | The estimator is calibrated | Which estimator and version? |

### 5.6 Branch points summarized

| Branch | Condition | Path A | Path B |
|---|---|---|---|
| Data level | Raw reads available | Re-align and re-quantify | Analyze the provided matrix |
| Batch | Not confounded | Include batch as covariate | Correct the matrix, with sensitivity |
| Batch | Completely confounded | Restrict claims; collect new data | Not available |
| Composition | Bulk tissue | Deconvolve and adjust | State the limitation |
| Ancestry | Single group | Report and restrict claims | Validate in a second group |
| Replication | Independent cohort available | Replicate with a different pipeline | Report as a discovery, not a finding |
| Single cell | Many donors | Pseudobulk by donor | State pseudoreplication risk |

### 5.7 Quality metrics by assay

The metrics that matter differ by assay, and using the wrong one gives false reassurance.

- **WES and WGS.** Breadth at 20× or higher, mean depth, duplication rate, contamination estimate, and insert size. A mean depth of 40× with 60 percent breadth is a failed exome.
- **Bulk RNA-seq.** Mapping rate, rRNA fraction, duplication, library size, and the fraction of reads in genes. A high rRNA fraction means the library prep failed.
- **Single-cell RNA-seq.** Median genes per cell, median UMIs per cell, doublet rate, ambient RNA fraction, and the fraction of cells passing quality filters. A high doublet rate inflates apparent cell-type diversity.
- **Proteomics.** Identification rate, missed cleavage rate, precursor intensity distribution, and the fraction of missing values per sample.
- **Methylation arrays.** Detection p-values, bisulfite conversion efficiency, and the distribution of beta values.
- **Microbiome.** Read depth per sample, the fraction of reads classified, and negative-control contamination.

A quality report that lists all six assays' metrics for one assay is not a quality report.

### 5.8 What to do when the pipeline is unknown

If a matrix arrives without a pipeline record, three options remain:

1. **Re-derive from raw data.** Requires raw files, which may not be available.
2. **Compare to a reference matrix processed with a known pipeline.** Concordance gives an estimate of the discrepancy.
3. **Restrict the analysis to within-dataset contrasts** and state that cross-dataset comparison is not supported.

Option 3 is the honest default, and it is usually sufficient for a discovery analysis that will be replicated in a second dataset processed the same way.

A pipeline record does not need to be long. Four lines are enough: the aligner and version, the quantifier and version, the reference and annotation, and the filtering parameters. A matrix with those four lines is reproducible; one without them is not.

If the record is incomplete but the raw data exist, re-deriving is usually cheaper than reconstructing the pipeline from the matrix. The matrix has already lost the reads that failed filtering, and those reads are often where the discrepancy lies.

The same logic applies to annotation. A re-annotation with a newer gene model can move reads between genes, which changes every count for those genes without any change in the data.

### 5.9 Branch points summarized

---

## 6. Worked example

### 6.1 The question

A translational team has bulk RNA sequencing of 120 colorectal tissue samples: 60 tumour and 60 adjacent normal, collected at three sites over 18 months. They have a list of 1,850 genes at p < 0.05 and want to know whether it is a usable discovery set.

### 6.2 The multiple-testing arithmetic

Genes tested: 20,000 (after filtering).

Expected false positives at p < 0.05 under the global null: 20,000 × 0.05 = 1,000.

Observed hits at p < 0.05: 1,850.

Naive false discovery proportion: 1,000 / 1,850 = 0.541, or 54.1%.

That is the number to put in front of the team. More than half of the list is expected to be noise if there were no true effects, and the true proportion is lower but unknown. A list of 1,850 genes is not a discovery; it is a filtered set with an unknown error rate.

Applying Benjamini-Hochberg at q < 0.05 reduces the list to 640 genes. The reduction is the price of controlling the false discovery rate, and the 640 is the defensible discovery set at that threshold.

### 6.3 The batch structure

| Site | Tumour | Normal | Total |
|---|---:|---:|---:|
| Site 1 | 24 | 26 | 50 |
| Site 2 | 22 | 20 | 42 |
| Site 3 | 14 | 14 | 28 |
| Total | 60 | 60 | 120 |

This table looks balanced, and that is the trap. The relevant batch variable is not site but **library preparation date**, which is nested within site and not balanced.

| Prep batch | Tumour | Normal | Total |
|---|---:|---:|---:|
| Batch A (months 1-6) | 12 | 30 | 42 |
| Batch B (months 7-12) | 18 | 22 | 40 |
| Batch C (months 13-18) | 30 | 8 | 38 |
| Total | 60 | 60 | 120 |

Check: 12 + 18 + 30 = 60 tumours; 30 + 22 + 8 = 60 normals; 42 + 40 + 38 = 120.

Batch C is 79 percent tumour (30 of 38), and batch A is 29 percent tumour (12 of 42). Batch is strongly associated with group, which means the tumour-normal difference is partly a batch difference. This is partial confounding, not complete, so the effects are separable in principle, but the separation depends on the overlap and on the model.

### 6.4 Sensitivity to batch handling

Two analyses:

1. **Full cohort with batch as a covariate.** Result: 640 genes at q < 0.05.
2. **Batches A and B only (n = 82), removing the most imbalanced batch.** Result: 410 genes at q < 0.05.

Overlap: 355 genes appear in both lists.

- Proportion of the restricted list reproduced in the full list: 355 / 410 = 86.6%.
- Proportion of the full list reproduced in the restricted list: 355 / 640 = 55.5%.

Check: 355 / 410 = 0.8659. 355 / 640 = 0.5547.

Interpretation: the direction is reassuring for the restricted set, which is mostly contained in the full result. But only 55.5 percent of the full list survives when the imbalanced batch is removed, which means that 285 genes depend on the batch that is most confounded with group. Those 285 are the ones a validation study should test first, because they are the most likely to be batch artefacts.

### 6.5 Effect-size concordance

For the 355 overlapping genes, the correlation between log fold changes in the two analyses is 0.91. For the 285 non-overlapping genes, the correlation is 0.42.

Check the interpretation, not the arithmetic: a high correlation among replicated genes and a low correlation among non-replicated genes is the expected pattern when the non-replicated genes are driven by batch. If the non-replicated genes had a high correlation but different significance, the explanation would be power rather than batch.

### 6.6 A genomic aside: variant arithmetic

The same team runs whole exome sequencing on the 120 samples. Mean 45,000 variants per exome, so 120 × 45,000 = 5,400,000 variant calls.

Quality filtering (QUAL below 30, depth below 10, gnomAD allele frequency above 1 percent) retains 2.1 percent: 5,400,000 × 0.021 = 113,400 variants.

Check: 5,400,000 × 0.021 = 113,400.

Of those, 0.9 percent are classified pathogenic or likely pathogenic in ClinVar: 113,400 × 0.009 = 1,021 variants.

Check: 113,400 × 0.009 = 1,020.6, so about 1,021.

With 120 samples, that is about 8.5 reportable variants per exome, which is in the plausible range for clinical exome reporting and far below the raw call count. The lesson is that the analytic funnel, not the sequencing, determines the number a clinician sees.

Sequencing volume: 30× mean coverage over a 60 megabase target is 1.8 gigabases per sample, or 216 gigabases for 120 samples. At 100 base pair reads, that is 2.16 billion reads.

Check: 30 × 60,000,000 = 1,800,000,000 bases. × 120 = 216,000,000,000 bases = 216 Gb. / 100 = 2,160,000,000 reads.

### 6.7 What the example establishes

- Multiple-testing control changes a 1,850-gene list into a 640-gene list, and the change is the point.
- Sample-level balance does not imply batch balance; the batch table must be built on the actual processing variable.
- A sensitivity analysis that removes the most imbalanced batch quantifies how much of the signature depends on confounding.
- The analytic funnel reduces five million variant calls to about a thousand reportable variants, which is where clinical interpretation begins.

### 6.8 What a reviewer would ask

1. **Why is library prep date the batch variable rather than site?** Because the prep date determines reagent lot, operator, and sequencer run, which are the technical drivers. Site is a proxy that happens to be nested within date here.
2. **Is 86.6 percent replication good?** It is reassuring for direction but does not establish that the genes are biologically meaningful. Replication in an independent cohort with a different pipeline is the stronger test.
3. **Why not correct the matrix with ComBat?** Because correction removes variance that may be biological when batch is confounded with group. Covariate adjustment plus a sensitivity analysis is more transparent.
4. **Is a 0.9 percent pathogenic rate plausible?** It is plausible for a filtered exome set, and it depends heavily on the ClinVar release and the classification pipeline. The number should be reported with both versions.

### 6.9 Cost and timeline arithmetic

A discovery programme's cost is dominated by sample processing and validation, not sequencing.

| Item | Unit cost (illustrative) | Quantity | Total |
|---|---:|---:|---:|
| Bulk RNA-seq library and sequencing | $450 per sample | 120 | $54,000 |
| Exome library and sequencing | $600 per sample | 120 | $72,000 |
| Clinical exome confirmation | $250 per variant | 1,021 | $255,250 |
| Independent validation cohort (RNA-seq) | $450 per sample | 200 | $90,000 |
| Analysis effort | $120 per hour | 600 hours | $72,000 |
| Total | | | $543,250 |

Check: 450 × 120 = 54,000. 600 × 120 = 72,000. 250 × 1,021 = 255,250. 450 × 200 = 90,000. 120 × 600 = 72,000. Sum = 543,250.

The confirmation line is 47 percent of the total (255,250 / 543,250 = 0.4698), which is the counter-intuitive part: the cheapest per-unit step dominates the budget because it is applied to every candidate. A programme that budgets for sequencing and not for confirmation will run out of money before it has a validated result.

The validation cohort is 16.6 percent (90,000 / 543,250 = 0.1657). Cutting it saves the smallest share and removes the only evidence that the discovery replicates, which is the worst trade in the table.

### 6.10 What the example cannot tell you

Three questions the data cannot answer, each with the study that would:

1. **Is the signature clinically actionable?** The cohort has no treatment or outcome data. Answering requires a cohort with outcomes and a pre-specified decision rule.
2. **Does the signature reflect tumour biology or immune infiltration?** The study has no cell-composition measurement. Answering requires deconvolution or single-cell data from the same samples.
3. **Does the signature transfer to other ancestry groups?** The cohort's ancestry is not reported in this example. Answering requires a multi-ancestry cohort and a validation that stratifies by group.

Each of these is a different study, not a different analysis. Recognizing that is the difference between a research plan and a wish list.

### 6.11 The decision the example supports

After the analysis, the team has 640 genes at q < 0.05, 355 of which replicate when the most imbalanced batch is removed, and an estimated 285 that depend on the confounded batch. The defensible next step is not to advance all 640 to validation. It is to advance the 355 replicated genes first, and to treat the remaining 285 as candidates requiring a design that can separate batch from biology.

That is a decision about resource allocation, not about statistics, and it is the kind of decision the data can support. A team that advances 640 genes will spend the confirmation budget three times over and will learn less.

---

## 7. Code, query, or calculation

### 7.1 Python: Benjamini-Hochberg by hand

Assumptions: Python 3.11 or later; numpy installed. The implementation is intentionally explicit so it can be audited.

```python
import numpy as np

def benjamini_hochberg(pvals):
    """Return BH q-values for a list of p-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity from the largest p-value downward.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out

pvals = [0.001, 0.004, 0.008, 0.012, 0.03, 0.04, 0.06, 0.2, 0.5, 0.9]
q = benjamini_hochberg(pvals)

print(f"{'p':>8} {'q':>8} {'q<0.05':>7}")
for pv, qv in zip(pvals, q):
    print(f"{pv:8.3f} {qv:8.4f} {str(qv < 0.05):>7}")
print(f"\nTests: {len(pvals)}")
print(f"p<0.05 count: {sum(p < 0.05 for p in pvals)}")
print(f"q<0.05 count: {int((q < 0.05).sum())}")
```

Expected output:

```text
       p        q  q<0.05
   0.001   0.0100    True
   0.004   0.0200    True
   0.008   0.0267    True
   0.012   0.0300    True
   0.030   0.0600   False
   0.040   0.0667   False
   0.060   0.0857   False
   0.200   0.2500   False
   0.500   0.5556   False
   0.900   0.9000   False

Tests: 10
p<0.05 count: 5
q<0.05 count: 4
```

Note that five p-values are below 0.05 and four q-values are below 0.05. On a small set the correction is mild; on 20,000 tests it is severe, which is why the worked example's list falls from 1,850 to 640.

### 7.2 Python: expected false positives and false discovery proportion

```python
n_tests = 20_000
alpha = 0.05
observed_hits = 1_850
after_bh = 640

expected_fp = n_tests * alpha
naive_fdp = expected_fp / observed_hits

print(f"Tests: {n_tests:,}")
print(f"Expected false positives at alpha={alpha}: {expected_fp:,.0f}")
print(f"Observed hits: {observed_hits:,}")
print(f"Naive false discovery proportion: {naive_fdp:.3f} ({naive_fdp*100:.1f}%)")
print(f"After BH at q<0.05: {after_bh:,} genes")
print(f"Implied enrichment over null: {observed_hits/expected_fp:.2f}x")
```

Expected output:

```text
Tests: 20,000
Expected false positives at alpha=0.05: 1,000
Observed hits: 1,850
Naive false discovery proportion: 0.541 (54.1%)
After BH at q<0.05: 640 genes
Implied enrichment over null: 1.85x
```

The enrichment figure of 1.85 is a useful sanity check: an observed hit count less than twice the null expectation is weak evidence of a real signal across the whole set, even when individual genes are significant.

### 7.3 SQL: the group-by-batch table

Assumptions: PostgreSQL; a `sample` table with `sample_id`, `group_label`, `prep_batch`.

```sql
-- Contingency table of group by prep batch.
SELECT prep_batch,
       COUNT(*) FILTER (WHERE group_label = 'tumour') AS tumour,
       COUNT(*) FILTER (WHERE group_label = 'normal') AS normal,
       COUNT(*) AS total,
       ROUND(100.0 * COUNT(*) FILTER (WHERE group_label = 'tumour') / COUNT(*), 1)
           AS pct_tumour
FROM sample
GROUP BY prep_batch
ORDER BY prep_batch;
```

Expected output (matches section 6.3):

```text
prep_batch | tumour | normal | total | pct_tumour
-----------+--------+--------+-------+------------
Batch A    |     12 |     30 |    42 |       28.6
Batch B    |     18 |     22 |    40 |       45.0
Batch C    |     30 |      8 |    38 |       78.9
```

A batch with `pct_tumour` far from the overall 50 percent is partially confounded. A batch with zero in one column is completely confounded, and no statistical method recovers the biological effect from that batch alone.

### 7.4 A hand calculation for an interview

Given 20,000 tests and 1,850 hits at p < 0.05:

1. Expected false positives = 20,000 × 0.05 = 1,000.
2. Naive FDP = 1,000 / 1,850 = 0.54.
3. If the true number of effects is 900, then false positives are 1,850 − 900 = 950 and FDP = 950 / 1,850 = 0.51.
4. If the true number is 1,700, FDP is 150 / 1,850 = 0.08.

The point of the calculation is that the FDP depends on the unknown number of true effects, which is why an estimate (q-value) is used rather than an exact count.

---

## 8. Data, standards, and regulatory context

### 8.1 Minimum information standards

| Standard | Scope | What it requires | What it does not |
|---|---|---|---|
| MIAME | Microarray experiments | Sample, protocol, and processing metadata | Sequencing-specific detail |
| MINSEQE | Sequencing experiments | Minimum metadata for sequencing | Downstream analysis |
| FAIR principles | Data stewardship | Findable, accessible, interoperable, reusable | Legal access terms |
| DUO | Data use ontology | Machine-readable consent and use terms | Enforcement |
| CONSORT and STROBE | Reporting | Study reporting completeness | Data quality |
| ACMG/AMP 2015 | Variant classification | Evidence categories for pathogenicity | Assay-specific thresholds |

These standards constrain metadata, not conclusions. A dataset can be fully FAIR and biologically wrong.

### 8.2 Access infrastructure

| Repository | Content | Access | Notes |
|---|---|---|---|
| dbGaP | Genotypes and phenotypes, NIH-funded | Controlled, data access committee | Tiered: open aggregate, controlled individual-level |
| EGA | European genotype-phenotype data | Controlled, data access committee | Common for EU studies |
| GDC | NCI genomic data | Open and controlled tiers | TCGA, TARGET, CPTAC, GENIE |
| GEO | Gene expression datasets | Open | Widely used for reanalysis |
| SRA and ENA | Raw sequence reads | Open | Large; accessions are the reproducibility handle |
| ArrayExpress | Functional genomics | Open | European mirror of GEO in function |
| gnomAD | Aggregate allele frequencies | Open, with terms | No individual-level data |

The distinction between open aggregate and controlled individual-level access is fundamental. A variant frequency table is open; a VCF with individual genotypes is controlled, because a genome identifies a person.

### 8.3 GA4GH standards

The Global Alliance for Genomics and Health (GA4GH) publishes standards for interoperable genomic data sharing:

- **Data Use Ontology (DUO)** encodes consent terms as machine-readable classes, so a query can ask for datasets whose terms permit a given use.
- **Authentication and Authorization Infrastructure (AAI)**, version 1.0 published October 2019, defines OpenID Connect-based passports and visas for federated researcher identity and data-access authorization.
- **Data Security Infrastructure Policy (DSIP)**, version 4.0 approved September 2019, sets control objectives referencing NIST 800-53 and ISO 27001.
- **htsget** defines an API for streaming slices of genomic files, so an analyst can retrieve a region without downloading a whole BAM.
- **Crypt4GH** defines encryption for genomic data in transit and at rest.

An implementation detail worth knowing: DSIP v4.0 dates from 2019 and GA4GH has described the work stream as dormant. That does not make it wrong, but it means the policy reflects 2019 threat assumptions and should be read alongside current institutional security requirements rather than instead of them.

### 8.4 What is binding and what is convention

Binding in the US:

- **HIPAA** applies to genomic data held by covered entities. A genome is protected health information when it is linked to an individual and held by a covered entity.
- **NIH Genomic Data Sharing Policy** requires controlled access for most individual-level genomic data generated with NIH funding, and requires a data sharing plan.
- **GINA**, the Genetic Information Nondiscrimination Act of 2008, prohibits discrimination in health insurance and employment on the basis of genetic information. It does not cover life, disability, or long-term care insurance.
- **CLIA and CAP** govern clinical laboratory testing. A genomic test reported to a patient or clinician for a clinical decision must be performed in a certified laboratory.
- **FDA** regulates genomic tests with a clinical claim, including companion diagnostics, which require a premarket pathway.

Convention:

- ACMG/AMP variant classification is a professional convention, widely adopted and not a legal requirement.
- Reporting a polygenic score with a percentile is a convention with no standardized reference population.
- Using GRCh38 rather than GRCh37 is a convention that is now near-universal for new data and is not required for older data.

### 8.5 Privacy: why genomic data is different

Genomic data has three properties that change the privacy calculus:

1. **It is identifying.** A genome is unique to an individual, and a modest number of markers can identify a person from a public database.
2. **It is relational.** It carries information about parents, siblings, and children who did not consent. A privacy breach affects people outside the study.
3. **It is immutable.** A compromised password can be changed; a compromised genome cannot.

The regulatory consequence is that de-identification, in the HIPAA sense, is a weaker protection for genomic data than for clinical data. The operational consequence is that controlled access, data use agreements, and secure processing environments are the norm rather than the exception.

> **Note:** In the EU, genomic data is special-category personal data under Article 9 of the GDPR and requires an explicit legal basis. The European Health Data Space Regulation (EU) 2025/327 adds a secondary-use framework that phases in from 2029, with genetic and genomic data provisions staged to 2031.

### 8.6 Clinical versus research genomics

A research variant call and a clinical variant report are different products:

| Dimension | Research | Clinical |
|---|---|---|
| Laboratory standard | Research-grade | CLIA-certified, CAP-accredited |
| Variant confirmation | Often none | Orthogonal confirmation for reportable variants |
| Classification | Exploratory | ACMG/AMP categories with evidence |
| Reporting | Aggregate or exploratory | Patient-specific report |
| Liability | Research | Clinical and legal |

A pipeline that is adequate for research is not automatically adequate for a clinical report, and the gap is the confirmation and classification step, not the sequencing.

---

## 9. Industry practice and case evidence

### 9.1 The scale of public cancer genomics

The Cancer Genome Atlas (TCGA) characterized about 11,000 patients across 33 cancer types, and remains the most-used public resource for cancer multi-omics. The Genomic Data Commons, which hosts TCGA and other programs, describes 22 programs, 83 projects, 69 primary sites, and 44,637 cases, a figure that includes TARGET, CPTAC, GENIE, and other contributions beyond TCGA.

The distinction matters commercially. "TCGA" is a specific cohort with a specific consent and a specific set of assays; "GDC" is a repository with mixed consent and mixed assay types. A request for "TCGA data" and a request for "GDC data" are different requests, with different access tiers.

### 9.2 gnomAD and the reference frequency problem

gnomAD v4.0, released 1 November 2023, contains 807,162 individuals: 730,947 exomes (including 416,555 from UK Biobank) and 76,215 genomes, all aligned to GRCh38. It includes about 138,000 individuals of non-European genetic ancestry, and the inclusion of UK Biobank raises the overall European proportion.

Two analytic lessons:

1. **Allele frequency is population-specific.** A variant that is rare in one ancestry group can be common in another. Filtering on a global frequency threshold will misclassify variants in under-represented groups, which is a source of health disparity in genetic diagnosis.
2. **Reference databases change.** gnomAD removed TCGA cancer samples on quality grounds and phased out disease-specific subsets. A filter built on one release is not reproducible on the next.

### 9.3 Large cohort programmes

- **UK Biobank** combines questionnaires, physical measures, imaging, genomics, and linked health records in a prospective cohort of about 500,000 participants. Its depth is unmatched and its healthy-volunteer and ancestry composition are well documented limitations.
- **All of Us** reported more than 633,000 participants in its September 2024 data release, including more than 400,000 with whole genome sequences and 429,000 with genotyping arrays. By February 2025 the release included 447,278 genotyping arrays, 414,830 short-read whole genomes, 97,061 structural variant calls, and 2,800 long-read genomes.

The commercial significance of these programmes is that they make large-scale genomic association studies feasible outside a single company, which shifts the competitive advantage from data possession to analysis and validation.

### 9.4 Why omics discoveries fail to replicate

The published record on replication failure in omics is extensive, and the mechanisms are consistent:

- **Underpowered discovery.** Small sample sizes produce effect-size estimates that are inflated by selection, a phenomenon sometimes called the winner's curse.
- **Batch confounding.** Technical variation correlated with the outcome produces signals that do not replicate in a different batch structure.
- **Composition.** Bulk tissue signals driven by cell-type proportions do not replicate when the composition differs.
- **Overfitting.** Classifiers with more features than samples fit noise, and cross-validation within the same dataset does not detect it.
- **Publication and analysis flexibility.** Many analytical choices, one reported result.

The practical response is pre-registration of the analysis plan, an independent replication cohort with a different pipeline, and reporting of effect sizes with intervals rather than only significance.

### 9.5 The single-cell replication problem

Single-cell studies increased the resolution of omics and introduced a new replication problem: cells are not replicates. A study with 50,000 cells from 10 donors has 10 biological replicates, and treating cells as independent inflates significance by orders of magnitude.

The published response is pseudobulk analysis, in which counts are summed per donor per cell type before testing, and mixed models that include a donor random effect. A single-cell paper that reports cell-level p-values without donor-level aggregation should be read with that limitation in mind.

### 9.6 What a mature omics team does

- **Records the reference, annotation, and pipeline versions** for every matrix.
- **Builds the group-by-batch table** before any result.
- **Reports effect sizes with intervals**, not only q-values.
- **Replicates in an independent cohort** with a different pipeline where possible.
- **Reports ancestry composition** and discusses transferability.
- **Aggregates to the biological replicate** in single-cell work.

These six practices are the difference between a discovery and a hypothesis.

### 9.7 How omics data enters a commercial decision

Three commercial uses dominate, and each has a characteristic failure:

| Use | Input | Characteristic failure |
|---|---|---|
| Target identification | Case-control differential expression or genetics | Confusing association with causation; ignoring tissue specificity |
| Companion diagnostic | A classifier on a validated platform | Validating on one platform and deploying on another |
| Market sizing by biomarker prevalence | Population frequency of a marker | Using a reference frequency from a different ancestry group |

The third is the most common and the least discussed. A biomarker-defined market size computed from a European-ancestry reference will be wrong in a population with different allele frequencies, and the error can be large enough to change an investment decision.

### 9.8 The reproducibility dividend

Public omics data has a property that clinical data lacks: a result can often be recomputed from raw files by anyone with access. That makes methodological disputes resolvable, which is why omics methods improve faster than clinical prediction methods.

The commercial implication is that a company's omics advantage rarely comes from possessing public data. It comes from proprietary cohorts, validated pipelines, and the ability to replicate. A strategy built on exclusive access to TCGA would be a strategy built on a public resource.

---

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No multiple-testing control | Thousands of hits at p<0.05 | Nominal threshold on many tests | Mostly false list | BH or Storey FDR; report q |
| Batch confounded with group | One batch contains only cases | Processing order follows enrollment | Biological effect inseparable | Balance the design; sensitivity analysis |
| Reference genome unrecorded | Coordinates cannot be matched | Alignment to an unknown build | Dataset not comparable | Record build and annotation |
| Pipeline unrecorded | Matrix cannot be reproduced | Tool versions differ | Irreproducible result | Version manifest |
| Composition ignored | Signal disappears after deconvolution | Bulk tissue mixes cell types | Wrong biological claim | Deconvolve; adjust |
| Single ancestry | No diversity in the cohort | Recruitment from one population | Poor transferability | Report ancestry; validate elsewhere |
| Cells treated as replicates | Cell-level p-values with few donors | Pseudoreplication | Inflated significance | Pseudobulk; donor random effect |
| Effect size omitted | Only q-values reported | Significance emphasized | Trivial effects presented as findings | Report effect and interval |
| Winner's curse | Effect sizes shrink on replication | Selection on estimated effect | Overstated clinical value | Report replication effect sizes |
| Sample swap | Sex check mismatch | Labeling error | Spectacular false result | Genotype and sex concordance |
| Contamination | Unexpected variants or taxa | Cross-sample contamination | False calls | Estimate and report contamination |
| Resource version drift | Annotation changed mid-project | Database updated | Irreproducible filtering | Pin versions |

### 10.1 The three red flags

1. **A gene list with p-values and no q-values.** Without FDR control, the list is not a discovery set.
2. **A group-by-batch table that is never shown.** If the authors did not show it, either it is balanced and they missed an opportunity, or it is not.
3. **No replication cohort.** A discovery without replication is a hypothesis, and the paper's language should say so.

### 10.2 The "biomarker" inflation

A gene that is differentially expressed at q < 0.001 with a 1.2-fold change is statistically robust and clinically weak. The inflation happens when significance is translated into "biomarker" without a consideration of effect size, measurement stability, and the intended use.

A useful test: ask what decision would change if the biomarker were true. If the answer is "none, because the effect is too small to stratify patients," the biomarker is a research finding, not a product.

---

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The reference and annotation are correct and recorded | Comparability | Wrong build, changed annotation | Re-check coordinates and gene models |
| Batch is not confounded with group | Separating technical from biological | Processing order follows enrollment | Group-by-batch table |
| Normalization assumptions hold | Comparing samples | Compositional shift in a few features | Sensitivity to normalization method |
| Bulk signal reflects the tissue of interest | Interpretation | High stromal or immune content | Deconvolution |
| Missing values are not informative | Complete-case analysis | Proteomics MNAR missingness | Compare missingness by group |
| Ancestry composition supports transfer | Generalization | Single-ancestry cohort | Report composition; external validation |
| Replication is independent | Confirming a discovery | Same pipeline, same site | Different cohort and pipeline |
| Clinical validity implies utility | Product claims | No outcome evidence | Utility study |

### 11.1 Limits that are inherent

- **Omics measures molecules, not outcomes.** A molecular difference is not a clinical benefit.
- **Sample size is usually small.** Discovery cohorts are often underpowered for modest effects.
- **Technical variation is structured.** Batch cannot always be separated from biology.
- **Genomes are identifying and relational.** Privacy protection cannot rely on de-identification alone.
- **Reference resources change.** A result is tied to the versions used.
- **Ancestry representation is unequal.** Transferability is an empirical question, not an assumption.

### 11.2 How to state the omics limitation

"Bulk RNA sequencing of 60 tumour and 60 adjacent normal samples across three preparation batches showed 640 genes at q < 0.05 after Benjamini-Hochberg control. Because batch C was 79 percent tumour, batch was included as a covariate, and a sensitivity analysis restricted to batches A and B reproduced 355 of the 410 genes in that subset (86.6 percent) but only 355 of the 640 in the full set (55.5 percent). Cell composition was not measured; a signal driven by immune infiltration cannot be excluded."

That paragraph gives the design, the control, the confounding, the sensitivity, and the residual threat.

### 11.3 The limit that decides whether a result is a product

For a commercial decision, the decisive limit is usually not statistical but operational:

- **Is the measurement stable enough to be repeated?** A signature that depends on a specific library prep is not a product.
- **Is the effect large enough to change a decision?** A 1.1-fold difference rarely stratifies patients.
- **Is the assay available where the patients are?** A research pipeline in one laboratory is not a distributed test.

A discovery can be statistically impeccable and commercially irrelevant, and the three questions above are how that is detected before the budget is spent.

A fourth question is regulatory: would the intended use require a cleared or approved test? If yes, the analytical validation burden is set by the regulator, not by the discovery study, and the discovery is the beginning of a much longer process.

The practical ordering is: establish the effect and its stability, then the decision relevance, then the assay feasibility, then the regulatory pathway. A programme that starts with the regulatory pathway and works backward usually discovers at the end that the effect was never large enough.

---

## 12. Exercises

### Exercise 1 — Data levels (recall, 20 minutes)

For three assays (WES, bulk RNA-seq, single-cell RNA-seq), list the raw, processed, and matrix levels and one analysis each level supports.

Deliverable: a three-by-three table.

Acceptance criteria: each level is named with its file type; the supported analysis is specific.

### Exercise 2 — Expected false positives (calculation, 30 minutes)

For 15,000 tests at alpha 0.01 and 0.05, compute the expected false positives. Then compute the naive FDP for 400 observed hits at alpha 0.01.

Deliverable: the calculations with intermediate steps.

Acceptance criteria: arithmetic correct; the interpretation states why FDP is not the same as the probability that any single hit is false.

### Exercise 3 — BH by hand (calculation, 40 minutes)

Apply BH to the p-values 0.002, 0.006, 0.015, 0.02, 0.04, 0.05, 0.1, 0.3. Report the q-values and the number of genes at q < 0.05.

Deliverable: a table of p, rank, p×m/rank, and monotone q.

Acceptance criteria: the monotonicity step is shown; the count at q < 0.05 is correct.

### Exercise 4 — Batch confounding (application, 60 minutes)

Using the batch table in section 6.3, compute the expected number of tumours in each batch under independence. Compare to observed and quantify the imbalance.

Deliverable: a table of observed, expected, and difference.

Acceptance criteria: expected counts computed from marginals; the largest imbalance identified; the interpretation states whether correction is possible.

### Exercise 5 — Replication arithmetic (calculation, 30 minutes)

Given 640 genes in the full analysis and 410 in the restricted analysis with 355 overlapping, compute both replication proportions and explain why they differ.

Deliverable: two percentages and a one-paragraph explanation.

Acceptance criteria: arithmetic correct; the explanation distinguishes direction of the comparison (which set is the denominator).

### Exercise 6 — Paper critique (critique, 60 minutes)

Find an omics discovery paper. Identify the sample size, the multiple-testing procedure, the batch handling, and the replication. Write a 400-word critique.

Deliverable: a 400-word critique with two proposed sensitivity analyses.

Acceptance criteria: the critique names the specific missing analysis; each proposal is feasible with the reported data; no claim about author intent.

### Exercise 7 — Provenance record (design, 60 minutes)

Write a provenance record for an omics matrix: assay, level, reference, annotation, pipeline, normalization, batch strategy, and resource versions.

Deliverable: a one-page record.

Acceptance criteria: a second analyst could reproduce the matrix from the record; every version has a date; the access tier is stated.

### Exercise 8 — Access decision (design, 45 minutes)

For a project using dbGaP data, list the access steps, the permitted uses, and the restrictions on derived data. Identify one use that would require a new request.

Deliverable: a step list and a use-restriction table.

Acceptance criteria: the steps reflect controlled access, not open download; the restricted use is specific.

### Exercise 9 — Single-cell replication (application, 45 minutes)

A dataset has 48,000 cells from 12 donors. Explain why the effective sample size is 12, and describe how pseudobulk analysis changes the test.

Deliverable: a 300-word explanation with the pseudobulk procedure.

Acceptance criteria: the explanation distinguishes cells from donors; the procedure states the aggregation unit and the test.

---

## 13. Self-check answers

**A1. What is the expected number of false positives for 20,000 tests at p < 0.05, and what does it imply?**
20,000 × 0.05 = 1,000. It implies that a list of 1,850 hits has a naive false discovery proportion of 1,000/1,850 = 54.1 percent, so more than half the list is expected to be noise if there were no true effects.

**A2. Compute the BH q-value for the third smallest of ten p-values equal to 0.008.**
q = 0.008 × 10 / 3 = 0.0267, subject to monotonicity. Since the next q-value (0.03) is larger, no adjustment is needed, and q = 0.0267.

**A3. Why does BH control FDR rather than family-wise error?**
BH controls the expected proportion of false discoveries among the rejected hypotheses, which is the relevant quantity for a discovery screen. Family-wise error control, such as Bonferroni, controls the probability of any false positive and is far more conservative, which is appropriate for a confirmatory test of a single hypothesis.

**A4. What is the difference between normalization and batch correction?**
Normalization adjusts for library size and composition within a sample; batch correction adjusts for processing group effects across samples. Applying one does not address the other, and batch correction can remove biological signal when batch is confounded with group.

**A5. Compute the tumour percentage in batch C from 30 tumours and 8 normals.**
30 / 38 = 0.789, or 78.9 percent, against an overall 50 percent. The imbalance is 28.9 percentage points.

**A6. Why is a batch with only cases a fatal problem?**
Because the batch effect and the group effect are perfectly collinear. No model can separate them without an assumption, and any correction removes both or neither. The design, not the analysis, is the problem.

**A7. Compute the replication proportion of the full list and of the restricted list.**
Restricted list: 355/410 = 86.6 percent. Full list: 355/640 = 55.5 percent. They differ because the denominators are different sets, and the second answers "how much of the full discovery survives restriction," which is the more conservative question.

**A8. Why does a genome require controlled access when names are removed?**
Because a genome is unique and identifying, and it carries information about relatives. Removing names does not remove the ability to re-identify, and the privacy interest extends to people who did not consent.

**A9. What is pseudoreplication in single-cell data?**
Treating cells as independent replicates when the biological replicate is the donor. Cells from one donor are correlated, so cell-level tests understate variance and inflate significance.

**A10. What is the winner's curse?**
The inflation of estimated effect sizes when a discovery is selected on statistical significance. The selected effects are those that happened to be large, so the estimate is biased upward and shrinks on replication.

**A11. Why is gnomAD's ancestry composition an analytic issue?**
Because allele frequency is population-specific. A global frequency filter will classify variants differently across ancestry groups, which can cause misdiagnosis in under-represented groups. The composition of the reference determines whose variants are well classified.

**A12. Rubric for exercise 7 (provenance record).**
A strong record names the assay, data level, reference build, annotation release, pipeline and versions, normalization method, batch strategy, and every resource version with a date, plus the access tier. A weak record names the assay and the sample count only.

---

## 14. Interview questions

**Q1. Why is multiple-testing control mandatory in omics?**
Because the feature count is orders of magnitude larger than the sample count, and at a nominal 0.05 threshold the expected false positives are proportional to the number of tests. Without control, a gene list is mostly noise.

**Q2. Explain the difference between a p-value and a q-value.**
A p-value is the probability of the observed data under a null hypothesis for one test. A q-value estimates the proportion of false discoveries among all features at or below that threshold. The q-value is a property of the list, not of the individual test.

**Q3. How would you detect batch confounding?**
Build the group-by-batch table, compute variance explained by batch, and color dimensionality-reduction plots by batch and by group. A batch that contains only one group is complete confounding.

**Q4. What would you do if batch were completely confounded with group?**
A strong answer: acknowledge that the biological effect cannot be separated, and either collect new data with a balanced design or restrict claims to what the design supports. Weak answers propose a correction method.

**Q5. Why do omics discoveries fail to replicate?**
Underpowered discovery, winner's curse, batch confounding, composition effects, overfitting, and analytical flexibility. Replication in an independent cohort with a different pipeline addresses most of them.

**Q6. What is the difference between bulk and single-cell replication?**
Bulk has one measurement per sample; single-cell has many cells per sample but the biological replicate is still the donor. Single-cell requires pseudobulk or donor-level mixed models to avoid pseudoreplication.

**Q7. Why is genomic data harder to de-identify than clinical data?**
Because a genome is unique, immutable, and relational. It identifies the individual and carries information about relatives, so removing identifiers does not remove re-identification risk.

**Q8. What does a reference genome version change?**
Coordinates and, depending on the annotation, gene models. A variant at one coordinate in GRCh37 is at a different coordinate in GRCh38, and lifting over is not always exact.

**Q9. How would you decide between raw and matrix data for a reanalysis?**
A strong answer: raw if the question requires a different alignment or quantification method; matrix if the question is a different statistical contrast on the same features. Matrix analysis is cheaper and inherits the original pipeline's choices.

**Q10. What is the clinical versus research genomics distinction?**
Clinical testing requires a certified laboratory, orthogonal confirmation, and ACMG/AMP classification. Research pipelines are not automatically adequate for clinical reporting, and the gap is confirmation and classification.

**Q11. Why is effect size more important than significance for a biomarker?**
Because significance depends on sample size and can be achieved for a trivial effect, while a biomarker must be large enough and stable enough to change a decision. A q-value of 0.001 with a 1.1-fold change is rarely actionable.

**Q12. What would you put in an omics provenance record?**
Assay, data level, reference build, annotation release, pipeline and tool versions, normalization, batch strategy, resource versions with dates, sample sheet, and access tier.

**Q13. A gene list has 1,200 hits at q < 0.05. Is that a discovery?**
A strong answer: not without replication and effect sizes. A large list can reflect a real broad effect, a composition difference, or residual batch. The next question is what the log fold changes are and whether the list replicates in an independent cohort with a different pipeline.

**Q14. Why is a bulk tissue signature often a cell-composition signature?**
Because bulk tissue averages over cell types, and tumour, stroma, and immune cells contribute different expression profiles. A difference between groups can reflect the proportion of immune cells rather than any change within a cell type, which deconvolution or single-cell data can distinguish.

---

## 15. Key terms

- **ACMG** — American College of Medical Genetics and Genomics.
- **Annotation** — the gene model used for summarization.
- **BAM** — aligned reads, binary format.
- **Batch effect** — technical variation associated with processing group.
- **BH** — Benjamini-Hochberg procedure.
- **Bonferroni** — family-wise error correction.
- **Breadth** — proportion of target covered.
- **ClinVar** — public variant-disease database.
- **Compositional data** — data where features sum to a whole.
- **Coverage** — reads covering a position or region.
- **CRAM** — compressed aligned reads.
- **dbGaP** — NIH database of genotypes and phenotypes.
- **Deconvolution** — estimation of cell-type proportions.
- **Depth** — number of reads at a locus.
- **Doublet** — a droplet containing two cells.
- **DUO** — Data Use Ontology.
- **EGA** — European Genome-phenome Archive.
- **FASTQ** — raw read file with base qualities.
- **FDR** — false discovery rate.
- **GDC** — Genomic Data Commons.
- **GINA** — Genetic Information Nondiscrimination Act.
- **gnomAD** — Genome Aggregation Database.
- **H5AD** — AnnData file format for single-cell data.
- **Normalization** — adjustment for technical variation within a sample.
- **Power** — probability of detecting an effect.
- **Pseudobulk** — aggregation of single-cell counts per donor.
- **q-value** — an estimate of FDR at a threshold.
- **Reference genome** — the coordinate system for alignment.
- **Somatic** — acquired variant, not inherited.
- **TCGA** — The Cancer Genome Atlas.
- **TPM** — transcripts per million.
- **UMI** — unique molecular identifier.
- **VAF** — variant allele fraction.
- **VCF** — variant call format.
- **WES** — whole exome sequencing.
- **WGS** — whole genome sequencing.
- **Winner's curse** — inflation of selected effect estimates.

---

## 16. Sources

All URLs accessed 24 October 2026.

**Datasets and repositories**

- National Cancer Institute. *Genomic Data Commons.* https://datacommons.cancer.gov/repository/genomic-data-commons
- National Cancer Institute. *The Cancer Genome Atlas Program.* https://www.cancer.gov/ccg/research/genome-sequencing/tcga
- Genome Aggregation Database. *gnomAD v4.0 release announcement*, 1 November 2023. https://gnomad.broadinstitute.org/news/2023-11-gnomad-v4-0/
- gnomAD. *Dataset selection and downloads.* https://gnomad.broadinstitute.org/downloads
- UK Biobank. https://www.ukbiobank.ac.uk/
- All of Us Research Program. *Data Access Tiers.* https://www.researchallofus.org/data-tools/data-access/
- NIH. *All of Us Research Program year in review: 2024.* https://pmc.ncbi.nlm.nih.gov/articles/PMC12461012/
- NIH. *All of Us Program Update* (September 2024 presentation). https://dpcpsi.nih.gov/sites/default/files/2025-01/Day-1-215PM-All-of-Us-Program-Update-Denny-v3-508.pdf

**Standards and policy**

- GA4GH. *Data Security Infrastructure Policy (DSIP) v4.0*, approved 25 September 2019. https://www.ga4gh.org/product/data-security-infrastructure-policy/
- GA4GH. *Authorisation and Authentication Infrastructure (AAI).* https://www.ga4gh.org/product/aai/
- GA4GH. *Data Use Ontology.* (Cite the ontology; see the GA4GH product pages for current versions.)
- NIH. *Genomic Data Sharing Policy.* (Cite the policy; see the NIH Office of Science Policy pages.)
- US Congress. *Genetic Information Nondiscrimination Act of 2008.* (Cite the statute, Public Law 110-233.)
- European Union. *Regulation (EU) 2016/679 (GDPR).* https://eur-lex.europa.eu/eli/reg/2016/679/oj
- European Union. *Regulation (EU) 2025/327 on the European Health Data Space.* https://eur-lex.europa.eu/eli/reg/2025/327/oj/eng
- US Food and Drug Administration. *Digital Health Center of Excellence.* https://www.fda.gov/medical-devices/digital-health-center-of-excellence

**Methods and background**

- Benjamini, Y., and Hochberg, Y. "Controlling the false discovery rate: a practical and powerful approach to multiple testing." *Journal of the Royal Statistical Society B*, 1995;57:289-300. (Background, pre-2024; the method is unchanged.)
- Storey, J. D. "A direct approach to false discovery rates." *Journal of the Royal Statistical Society B*, 2002;64:479-498. (Background.)
- Richards, S., et al. "Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the American College of Medical Genetics and Genomics and the Association for Molecular Pathology." *Genetics in Medicine*, 2015;17:405-424. (Background for ACMG/AMP classification.)

> **Note:** The TCGA and GDC case counts come from NCI and CRDC descriptions and are quoted as published. The GDC figure includes programs beyond TCGA, and the two figures should not be treated as the same cohort.

---

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_signals-and-wearables.md`](05_signals-and-wearables.md)
- Next topic: [`07_trial-data.md`](07_trial-data.md)
- Related sibling: [`04_imaging-and-dicom.md`](04_imaging-and-dicom.md)
- Related sibling: [`09_licensing-and-access.md`](09_licensing-and-access.md)
- Related sibling: [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md)
- Root survey file: [`../../02_biomedical_data_landscape.md`](../../02_biomedical_data_landscape.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 2.4 in the survey file (source-selection matrix) applies directly to the omics-versus-clinical-data choice, and Lab 2.5 (data-quality profile) covers the batch diagnostics in section 5.

Related reading: [`07_trial-data.md`](07_trial-data.md) for the prospective design that avoids many of the confounding problems described here; [`09_licensing-and-access.md`](09_licensing-and-access.md) for controlled-access mechanics; and [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md) for the formal treatment of completeness and plausibility.

The sentence to remember: omics measures molecules in a sample, and the distance from that to a clinical claim is measured in batches, composition, and replication, not in sequencing depth.

The next topic turns to trial data, where the design is prospective and many of the confounding problems described here are controlled by protocol rather than by analysis.

That contrast is the reason the two topics sit next to each other in this module: same molecular questions, different evidence standards.

A reader who understands why a batch-confounded discovery needs replication will understand why a trial needs randomization, and the logic is the same.