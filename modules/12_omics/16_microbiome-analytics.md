# Microbiome Analytics and Metagenomics

> Module 12 — Omics and Precision Medicine
> Topic 16 of 16
> Estimated study time: 4 hours
> Prerequisites: topics 01, 02, and 06
> Version 2026.3

## 1. Why this topic exists

The human gut contains roughly 10^13 to 10^14 microbial cells and a metagenome with millions of
distinct genes, many times the gene count of the human genome. Those microbes produce metabolites
that enter the circulation, train the immune system, and modify drug metabolism. The microbiome is
therefore both a biological system to be understood and a commercial target: live biotherapeutic
products, microbiome-derived biomarkers, and companion diagnostics.

Two products moved the field from research to regulation. The United States Food and Drug
Administration (FDA) approved Rebyota, a fecal microbiota product, on 30 November 2022 for
recurrent Clostridioides difficile infection, and Vowst, an orally administered fecal microbiota
spore product, on 26 April 2023 for the same indication. Both are defined biological products with
manufacturing controls, and both depend on sequencing to characterize the donor material and the
product.

The analytics are harder than the biology suggests, for four reasons.

**The data are compositional.** A sequencer measures a fixed number of reads per sample. If one
taxon doubles, every other taxon's measured proportion falls, even if its absolute abundance is
unchanged. A differential abundance test that ignores this produces associations that are
arithmetic artifacts.

**The data are sparse and zero-inflated.** A matrix of taxa by samples often has 70% to 90% zeros,
from taxa that are absent, taxa below the detection limit, and taxa lost to technical failure.
Zero-inflation breaks methods that assume a continuous distribution.

**Contamination is structured.** Reagents, extraction kits, and the environment contribute DNA.
The contaminating community is not random: certain genera, such as Halomonas, are common in kits.
A low-biomass sample can be dominated by contamination, and the contamination can look like a
biological signal.

**Associations are not causes.** A taxon that is more abundant in disease may be a cause, a
consequence, or a marker of a third factor such as diet, medication, or inflammation. The
microbiome is unusually susceptible to reverse causation because disease changes the gut
environment.

The consequence is a literature with a reproducibility problem. A 2022 comparison of differential
abundance methods across 38 datasets found that different methods produced different results on
the same data, with little agreement on which taxa were significant. The methods have improved
since, but the lesson holds: the analysis choices determine the answer, and they must be stated.

### 1.1 The failure pattern

1. Samples are collected without a sampling protocol, and the collection method differs between
   cases and controls.
2. DNA is extracted with a kit that contributes its own community, and no negative control is
   sequenced.
3. Reads are assigned to taxa with a classifier whose database does not match the environment.
4. Relative abundances are compared with a t-test or a standard correlation.
5. Alpha diversity is compared without checking that sequencing depth is similar across groups.
6. Beta diversity is tested with PERMANOVA on an unbalanced design with three samples per group.
7. A taxon is declared a biomarker without adjustment for medication or diet.
8. A causal claim is made from a cross-sectional association.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Clinician | A biomarker that does not replicate | Depends on study design and analysis |
| Patient | A dietary or probiotic recommendation with no evidence | Cannot see the data |
| Microbiologist | A result that conflicts with culture | Depends on the bioinformatics |
| Product team | A live biotherapeutic with an uncharacterized product | Depends on strain-level identification |
| Regulator | A submission without a validated characterization | Depends on the sponsor's methods |
| Program | A failed trial | Depends on the whole chain |

### 1.3 The industry question

> Is the difference in this community real, is it compositional or absolute, and does it cause,
> follow, or merely mark the disease?

The question has three clauses. The first is statistical, the second is ecological, and the third
is causal. Most published analyses answer the first and imply the third.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compare** 16S ribosomal RNA amplicon sequencing and whole-genome shotgun metagenomics on
   resolution, cost, bias, and the questions each can answer.
2. **Explain** why relative abundance data are compositional and **apply** a log-ratio
   transformation before correlation or regression.
3. **Compute** alpha diversity metrics and **test** whether a difference is confounded by
   sequencing depth.
4. **Compute** a Bray-Curtis distance matrix and **run** a permutation test with a stated minimum
   attainable p-value.
5. **Choose** a differential abundance method and state its assumptions about zeros, bias, and
   multiple testing.
6. **Detect** reagent contamination from negative controls and **quantify** its plausible
   contribution.
7. **Distinguish** an association from a causal claim, and name the designs that support causation.
8. **Map** a microbiome product onto the FDA live biotherapeutic pathway and the Minimum
   Information about any (x) Sequence (MIxS) metadata standard.

## 3. The core mental model

A microbiome study converts a community into a table of counts, and every downstream claim depends
on choices made before the table exists.

```text
  COMMUNITY (absolute abundances, mostly unknown)
      |
      +--> sampling: site, time, diet, medication, storage
      |
      +--> DNA extraction: kit, lysis, contamination
      |
      +--> sequencing: 16S amplicon or shotgun
      |
      v
  READS
      |
      +--> denoising or classification
      |
      v
  COUNT TABLE (taxa x samples, sparse, compositional)
      |
      +--> filter: prevalence, abundance
      +--> normalize: total-sum, rarefy, CLR
      |
      +--> alpha diversity (within sample)
      +--> beta diversity (between samples)
      +--> differential abundance (between groups)
      |
      v
  INFERENCE
      |
      +--> association
      +--> mechanism (metabolites, strains)
      +--> causation (intervention, Mendelian randomization)
```

### 3.1 The two assay families

| Property | 16S amplicon | Shotgun metagenomics |
|---|---|---|
| Target | Hypervariable regions of the 16S rRNA gene | All DNA in the sample |
| Resolution | Genus, sometimes species | Species and strain |
| Functional information | Inferred, not measured | Measured genes and pathways |
| Host DNA | Not amplified | Present in host-rich samples |
| Cost per sample | Low | Higher |
| Bias | Primer bias, copy number variation | Lysis bias, database dependence |
| Best use | Large cohorts, community structure | Strain tracking, function, products |

The choice is not about which is better. It is about which question the study can answer. A study
of community structure across 2,000 samples may reasonably use 16S. A study that must identify a
strain in a live biotherapeutic requires shotgun sequencing.

### 3.2 The simplex

A composition is a vector of non-negative proportions that sum to one. The space is a simplex, not
a Euclidean space, and standard operations are misleading on it.

Two consequences matter:

1. **Correlations are constrained.** Because the parts sum to a constant, an increase in one part
   forces a decrease somewhere. A negative correlation between two taxa can be entirely
   arithmetic.
2. **Scale is arbitrary.** Only ratios between parts are meaningful. Doubling every count leaves
   the composition unchanged, so absolute abundance cannot be recovered without an external
   measure.

The centered log-ratio (CLR) transformation maps a composition to a vector whose components sum to
zero:

```text
  clr(x)_i = ln(x_i) - (1/D) * sum_j ln(x_j)
```

The denominator is the geometric mean of the composition. CLR makes the analysis scale-invariant:
multiplying all counts by a constant leaves the CLR values unchanged. It does not recover absolute
abundance, and it does not eliminate the constraints of the simplex. It makes the ratios explicit.

### 3.3 Four quantities that are constantly confused

| Quantity | What it is | What it is not |
|---|---|---|
| Relative abundance | Proportion of reads assigned to a taxon | Absolute abundance |
| Absolute abundance | Cells or gene copies per gram | Measurable by sequencing alone |
| Alpha diversity | Within-sample richness or evenness | A measure of health |
| Beta diversity | Between-sample dissimilarity | A distance in physical space |

A fifth confusion is common in press releases: "dysbiosis". The term has no agreed definition and
no diagnostic threshold. It usually means "a community that differs from a reference", which is a
description, not a mechanism.

### 3.4 The inference ladder

| Level | Claim | Design that supports it |
|---|---|---|
| Association | Taxon X is more abundant in disease | Cross-sectional case-control |
| Temporality | Taxon X changes before disease | Longitudinal cohort |
| Mechanism | Taxon X produces metabolite Y that affects the host | Metabolomics plus in vitro or animal work |
| Causation | Changing X changes the outcome | Randomized intervention, germ-free colonization, Mendelian randomization |

Most published claims sit at the first level and are written as if at the fourth. The analysis
cannot climb the ladder; only the design can.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Microbiome | The microorganisms and their genetic material in a habitat | Used for the organisms only | Community |
| Microbiota | The assemblage of microorganisms | Used interchangeably with microbiome | Community |
| Metagenome | The collective genetic material of a community | Confused with a single genome | DNA |
| 16S ribosomal RNA | Conserved gene with hypervariable regions used for taxonomy | Assumed to resolve species | Gene |
| Amplicon sequence variant (ASV) | Exact sequence variant resolved by denoising | Confused with an operational taxonomic unit | Sequence |
| Operational taxonomic unit (OTU) | Cluster of sequences above a similarity threshold | Treated as a species | Cluster |
| Whole-genome shotgun | Sequencing all DNA in a sample | Assumed to be unbiased | Assay |
| Read | A sequenced DNA fragment | Treated as a molecule count | Sequence |
| Depth | Number of reads per sample | Confused with biomass | Reads |
| Relative abundance | Proportion of reads for a taxon | Treated as absolute abundance | Proportion |
| Compositional data | Data whose parts sum to a constant | Analyzed with standard Euclidean methods | Simplex |
| Centered log-ratio (CLR) | Log transform relative to the geometric mean | Treated as a substitute for absolute quantification | Dimensionless |
| Isometric log-ratio (ILR) | Log transform onto an orthonormal basis | Assumed equivalent to CLR for all models | Dimensionless |
| Alpha diversity | Within-sample diversity | Treated as a health metric | Index |
| Shannon index | Entropy of the abundance distribution | Compared across unequal depths without care | Nats or bits |
| Simpson index | Probability that two reads are different taxa | Confused with Shannon | 0 to 1 |
| Chao1 | Estimated richness from singletons and doubletons | Assumed accurate for deep sequencing | Count |
| Beta diversity | Between-sample dissimilarity | Confused with geographic distance | Distance |
| Bray-Curtis dissimilarity | Abundance-based dissimilarity | Assumed to account for phylogeny | 0 to 1 |
| UniFrac | Phylogenetic distance | Assumed to be better in all cases | 0 to 1 |
| Principal coordinates analysis (PCoA) | Ordination of a distance matrix | Treated as a hypothesis test | Coordinates |
| PERMANOVA | Permutation test of group differences in a distance matrix | Reported without the number of permutations | Pseudo-F and p-value |
| Differential abundance | Testing taxa for differences between groups | Confused with differential expression | Test |
| ANCOM-BC2 | Analysis of compositions with bias correction, version 2 | Run without its sensitivity analysis | Software |
| ALDEx2 | Compositional differential abundance using Monte Carlo Dirichlet sampling | Assumed to handle all designs | Software |
| False discovery rate (FDR) | Expected proportion of false positives among discoveries | Confused with family-wise error rate | Proportion |
| Mixed directional FDR | FDR accounting for direction across pairwise comparisons | Omitted in multigroup designs | Proportion |
| Rarefaction | Subsampling to equal depth | Assumed obsolete or mandatory; both are wrong | Reads |
| Negative control | Blank extraction or sequencing control | Omitted | Sample |
| Kit contamination | DNA introduced by reagents and kits | Assumed negligible | Contaminant taxa |
| Live biotherapeutic product (LBP) | Biological product containing live organisms for treating disease | Confused with a probiotic supplement | Regulatory category |
| Fecal microbiota transplantation (FMT) | Transfer of donor stool to a patient | Confused with an LBP | Procedure |
| Dysbiosis | Community difference from a reference | Treated as a diagnosis | None |
| Strain | Genetic variant within a species | Confused with species | Genome |
| Mendelian randomization | Use of genetic variants as instrumental variables | Assumed to prove causation | Design |

## 5. The mechanics, step by step

The procedure runs from an ecological question to an inference, with the assay and the statistics
chosen together.

| Step | Question answered | Artifact | Failure if skipped |
|---:|---|---|---|
| 1 | What is the question and the unit? | Analysis plan | Descriptive fishing |
| 2 | How are samples collected and controlled? | Sampling protocol | Batch confounding |
| 3 | Which assay? | 16S or shotgun | Wrong resolution |
| 4 | How is contamination controlled? | Negative controls | Kit signal read as biology |
| 5 | How are reads turned into taxa? | ASV or taxonomic table | Database or denoising bias |
| 6 | How are counts filtered and normalized? | Analysis-ready matrix | Depth and sparsity artifacts |
| 7 | Is within-sample diversity different? | Alpha diversity | Confounded diversity claim |
| 8 | Is between-sample composition different? | Beta diversity and test | Pseudoreplication |
| 9 | Which taxa differ? | Differential abundance | Compositional false positives |
| 10 | Are batch and confounders controlled? | Adjusted model | Confounded association |
| 11 | What function changes? | Pathway or metabolite analysis | Unsupported functional claim |
| 12 | What level of causal claim is supported? | Inference statement | Overclaiming |

### Step 1. Define the question and the unit of analysis

**Input.** A biological or clinical question.

**Output.** A written plan stating the unit (sample, subject, time point), the comparison, the
primary metric, and the covariates.

**Rationale.** Microbiome studies generate many possible comparisons. Without a pre-specified
primary comparison, the analysis becomes a search across taxa, metrics, and subgroups, and the
resulting p-values are not interpretable.

**If skipped.** The paper reports the comparison that reached significance.

### Step 2. Design sampling and controls

**Input.** Population; sampling site; timing; storage.

**Output.** A sampling protocol with negative controls and metadata.

**Rationale.** The microbiome changes with diet, time of day, medication, and bowel preparation.
A case-control study that collects cases in the clinic and controls at home confounds disease with
sampling. A study without extraction blanks cannot distinguish reagent DNA from a low-biomass
community.

The design decisions have more than two outcomes.

| Design situation | Risk | Action |
|---|---|---|
| Case and control sampled in different settings | Batch and setting confounding | Sample both groups in the same setting and batch |
| Antibiotic use differs between groups | The drug, not the disease, drives the difference | Record and adjust; consider excluding recent use |
| Stool collected at home and stored variably | DNA degradation and composition shift | Standardize collection kits and storage time |
| Low-biomass site such as skin or blood | Kit contamination dominates | Include extraction blanks and sequencing blanks |
| Longitudinal sampling | Within-subject correlation | Use mixed models or subject-level summaries |
| Cross-sectional single time point | Reverse causation | Limit claims to association |

**If skipped.** The group difference is a sampling difference.

### Step 3. Choose the assay

**Input.** Question; resolution requirement; budget; sample type.

**Output.** An assay with a stated resolution and bias profile.

**Rationale.** 16S amplifies a marker gene, which is cheap and genus-level, and is subject to
primer bias and copy number variation. Shotgun sequencing measures all DNA, resolves species and
strain, and is subject to lysis bias and host DNA dilution.

| Question | Assay | Reason |
|---|---|---|
| Community structure across a large cohort | 16S | Cost and throughput |
| Species or strain identification | Shotgun | 16S cannot resolve species reliably |
| Functional gene content | Shotgun | 16S infers function rather than measuring it |
| Response to a live biotherapeutic | Shotgun | Strain tracking is required |
| Archaea, fungi, or viruses | Shotgun or targeted assays | 16S primers target bacteria |
| Low-biomass sample | Shotgun with controls, or targeted assays | Contamination control is critical |

**If skipped.** A genus-level assay is used to support a species-level claim.

### Step 4. Control contamination

**Input.** Extraction and sequencing workflow.

**Output.** Negative controls sequenced alongside samples, plus a contaminant list.

**Rationale.** Kits contribute DNA, and the contaminant community is reproducible across batches.
Sequencing an extraction blank and a no-template control lets the analyst estimate the contaminant
profile and subtract or flag it.

**If skipped.** A genus that is common in kits is reported as a disease-associated taxon.

The blank is not a proof of cleanliness. It is a measurement of the contamination that the
workflow contributes, and the measurement is only valid for the batch in which the blank was
processed. A blank from a different kit lot does not control the current batch.

### Step 5. Convert reads to taxa

**Input.** Reads; reference database; denoising parameters.

**Output.** An ASV table or a taxonomic abundance table.

**Rationale.** For 16S, denoising resolves exact sequence variants and removes sequencing error.
For shotgun, a classifier assigns reads to taxa using a reference database, and the database
determines what can be detected. A database that underrepresents a phylum will undercount it.

| Method | Input | Output | Main limitation |
|---|---|---|---|
| DADA2 | 16S reads | ASVs | Sensitive to error model and truncation choices |
| Deblur | 16S reads | ASVs | Single-base resolution, aggressive filtering |
| QIIME 2 pipeline | Either | End-to-end artifacts and provenance | Complexity; version drift |
| Kraken2 | Shotgun reads | Taxonomic assignments | Database-dependent; memory |
| MetaPhlAn 4 | Shotgun reads | Marker-based species profiles | Limited to species with markers |

**If skipped.** The taxonomy reflects the database, not the community.

### Step 6. Filter and normalize

**Input.** Count table.

**Output.** Analysis-ready matrix with a stated transformation.

**Rationale.** Filtering removes taxa present in very few samples, which are usually noise.
Normalization addresses unequal depth. The choice among total-sum scaling, rarefaction, and
log-ratio transformation determines which statistical models are valid.

| Approach | What it does | When to use | Risk |
|---|---|---|---|
| Total-sum scaling | Divides by library size | Compositional analysis with CLR | Still compositional |
| Rarefaction | Subsamples to a common depth | Alpha diversity comparisons | Discards data; debated |
| CLR | Log-ratio to the geometric mean | Correlation, regression, PCA | Requires zero handling |
| ILR | Orthonormal log-ratio coordinates | Models requiring full-rank covariates | Less interpretable |
| No normalization | Uses raw counts | Count models with offsets | Depth confounding |

**If skipped.** Depth becomes a covariate that is correlated with the group.

### Step 7. Test alpha diversity

**Input.** Normalized counts; group labels.

**Output.** Diversity metrics per sample and a test with a depth check.

**Rationale.** Alpha diversity depends on depth: a sample sequenced more deeply observes more
taxa. If cases and controls differ in depth, a diversity difference may be technical. The check is
to compare depth distributions and to repeat the analysis after rarefaction or with a depth
covariate.

| Metric | What it captures | Sensitivity to depth | Sensitivity to rare taxa |
|---|---|---|---|
| Observed richness | Number of taxa seen | High | High |
| Shannon index | Entropy of the abundance distribution | Moderate | Moderate |
| Simpson index | Probability two reads are different taxa | Low | Low |
| Chao1 | Estimated richness from singletons and doubletons | Moderate | Very high |
| Faith's phylogenetic diversity | Branch length spanned on a tree | High | High |

**If skipped.** A technical depth difference is reported as a loss of diversity.

### Step 8. Test beta diversity

**Input.** Distance matrix; group labels; covariates.

**Output.** Ordination, PERMANOVA or a similar test, and a dispersion check.

**Rationale.** PERMANOVA tests whether the centroids of groups differ, but it is sensitive to
differences in dispersion: a group can be significant because it is more variable, not because its
centre differs. The minimum attainable p-value is determined by the number of distinct partitions,
which depends on group sizes.

| Metric | Uses phylogeny | Weighted by abundance | Best use |
|---|---|---|---|
| Bray-Curtis | No | Yes | General community comparison |
| Jaccard | No | No | Presence and absence |
| Unweighted UniFrac | Yes | No | Rare-taxon and phylogenetic structure |
| Weighted UniFrac | Yes | Yes | Abundant-taxon phylogenetic structure |
| Aitchison distance | No | Yes, on CLR | Compositional analysis |

**If skipped.** A significant result reflects heteroscedasticity, or a small study reports a
p-value that the design could never have produced.

### Step 9. Test differential abundance

**Input.** Count table; group labels; covariates.

**Output.** Effect sizes, adjusted p-values, and a sensitivity analysis.

**Rationale.** Differential abundance methods differ in their handling of compositionality, zeros,
and bias. ANCOM-BC2 estimates absolute abundances with a bias correction and controls the mixed
directional false discovery rate in multigroup designs. ALDEx2 uses Monte Carlo sampling from a
Dirichlet distribution. A method that ignores compositionality will report artifacts.

| Method | Compositional handling | Zero handling | Multiple testing | Best use |
|---|---|---|---|---|
| ANCOM-BC2 | Bias-corrected log-linear model | Structural zero detection | Mixed directional FDR | Multigroup, covariate-adjusted |
| ALDEx2 | Dirichlet Monte Carlo | Pseudocount via Dirichlet | Benjamini-Hochberg | Small studies, robustness |
| DESeq2 | Median-of-ratios | Outliers can dominate | Benjamini-Hochberg | Count data, not compositional |
| LEfSe | Not compositional | Not addressed | Not standard | Exploratory only |
| Wilcoxon on proportions | None | None | Unadjusted | Not recommended |

**If skipped.** The significant taxa are the ones whose proportions were forced to move.

A specific caution applies to ANCOM-BC2. The method includes a sensitivity analysis across
pseudo-count values, and the developers describe it as essential: disabling it raises the false
discovery rate in published comparisons. The version matters as well, because the order of the
sensitivity analysis and the bias correction changed in version 2.10.1. A methods section should
name the version and state whether the sensitivity analysis was run.

### Step 10. Control batch and confounders

**Input.** Metadata: batch, run, extraction date, medication, diet, age, sex.

**Output.** A model with the confounders included, and a batch diagnostic.

**Rationale.** Batch effects in microbiome data are large. A taxon that differs by batch can
appear to differ by group if the batches are not balanced. The diagnostic is to test whether batch
explains more variance than the group, and to check whether the groups are balanced across batches.

**If skipped.** The association is a batch effect.

### Step 11. Analyze function

**Input.** Shotgun reads or 16S profiles.

**Output.** Pathway abundances or predicted functions.

**Rationale.** Functional inference from 16S uses tools such as PICRUSt2, which predict gene
content from taxonomy. The prediction is only as good as the reference genomes and the taxonomic
resolution. Shotgun data can measure genes directly, but pathway assignment still depends on
annotation databases.

| Method | Input | Output | Main limitation |
|---|---|---|---|
| PICRUSt2 | 16S ASVs | Predicted gene families and pathways | Prediction, not measurement |
| HUMAnN | Shotgun reads | Pathways and gene families | Database-dependent |
| MetaCyc | Annotated genes | Pathway definitions | Curation lag for novel pathways |
| KEGG orthology | Annotated genes | Functional categories | Licensing and mapping ambiguity |
| Metabolomics | Mass spectrometry | Measured metabolites | Does not attribute the metabolite to a taxon |

**If skipped.** A predicted function is reported as a measured one.

### Step 12. State the level of causal claim

**Input.** Study design and results.

**Output.** A claim at the level the design supports.

**Rationale.** Cross-sectional associations cannot establish causation. Longitudinal data can
establish temporality. Intervention data, germ-free colonization, and Mendelian randomization can
support causal claims, each with its own assumptions.

**If skipped.** An association is reported as a cause, and a trial is designed on a false premise.

### 5.13 Standards and tools

| Standard or tool | What it covers | Status |
|---|---|---|
| MIxS | Minimum metadata for any sequence, with host-associated extensions | Community standard, version 7 |
| STORMS | Reporting checklist for human microbiome studies | Community standard |
| QIIME 2 | End-to-end analysis with provenance | Open-source software |
| DADA2 | Amplicon denoising | Open-source software |
| MetaPhlAn 4 | Marker-based species profiling | Open-source software |
| Kraken2 | k-mer taxonomic classification | Open-source software |
| ANCOM-BC2 | Compositional differential abundance | Open-source software |

### 5.14 Tests before a claim is made

| Test | Procedure | Pass criterion | What it catches |
|---|---|---|---|
| Negative control | Sequence extraction and no-template blanks | Contaminant profile identified; sample signal above it | Kit contamination |
| Depth balance | Compare read depth distributions by group | No group difference, or depth included as covariate | Technical diversity differences |
| Rarefaction sensitivity | Repeat alpha diversity at equal depth | Conclusion unchanged | Depth confounding |
| Dispersion check | Test homogeneity of multivariate dispersion | Groups equally dispersed | PERMANOVA driven by variance |
| Permutation adequacy | Compute the number of distinct partitions | Minimum attainable p below the reported p | Impossibly small p-values |
| Batch balance | Cross-tabulate group by batch | Balanced, or batch in the model | Batch confounding |
| Compositional sensitivity | Repeat differential abundance with CLR and a second method | Same direction and approximate effect | Method-specific artifacts |
| Functional validation | Compare predicted and measured function where available | Reasonable concordance | Overclaimed function |

### 5.15 Power, replication, and the smallest useful study

Microbiome studies are usually underpowered, and the power calculation is unusual because the
number of tests is large and the effect sizes are small.

| Quantity | Typical value | Consequence |
|---|---|---|
| Samples per group | 10 to 50 | Enough for large effects only |
| Taxa tested | 100 to 1,000 | Multiple testing penalty is severe |
| Median effect size | A log-ratio of 0.5 to 1.5 | Small relative to the variance |
| Within-group variability | High, driven by diet and individual | Requires larger samples |
| Batch count | 2 to 10 | Confounding if unbalanced |

Three practical rules follow. First, a discovery study should report effect sizes with confidence
intervals, not only p-values, because the p-value depends on the number of taxa tested. Second, a
replication cohort is more valuable than a larger discovery cohort, because it tests whether the
association generalizes. Third, the primary comparison should be pre-specified, because a study
that tests every taxon against every subgroup will find something.

### 5.16 What the procedure does not decide

The procedure produces a characterization of a community and a statistical association. It does
not decide whether the association is causal, whether a probiotic will help, or whether a product
is safe. Those require intervention studies and regulatory review.

## 6. Worked example

### 6.1 The case

A study compares the gut microbiome of four healthy volunteers with four patients with active
inflammatory bowel disease, using shotgun metagenomics. Eight genera are reported, with read
counts in thousands. The primary question is whether community composition differs between groups,
and the secondary question is whether diversity is lower in disease.

### 6.2 Step 1: check depth before diversity

The first computation is not diversity. It is the read depth per sample, because depth determines
how many taxa can be observed.

| Sample | Depth (reads) | Richness | Shannon | Simpson |
|---|---:|---:|---:|---:|
| H1 | 13,020 | 8 | 1.730 | 0.793 |
| H2 | 13,105 | 8 | 1.737 | 0.797 |
| H3 | 13,038 | 8 | 1.726 | 0.790 |
| H4 | 13,102 | 8 | 1.736 | 0.795 |
| D1 | 9,230 | 8 | 1.515 | 0.707 |
| D2 | 9,502 | 8 | 1.453 | 0.690 |
| D3 | 9,058 | 8 | 1.547 | 0.717 |
| D4 | 9,336 | 8 | 1.489 | 0.700 |

The disease samples have a mean depth of:

```text
  (9,230 + 9,502 + 9,058 + 9,336) / 4 = 37,126 / 4 = 9,281.5 reads
```

The healthy samples have a mean depth of:

```text
  (13,020 + 13,105 + 13,038 + 13,102) / 4 = 52,265 / 4 = 13,066.25 reads
```

The ratio is 13,066.25 / 9,281.5 = 1.41, so the healthy samples were sequenced about 41% more
deeply. The Shannon difference is:

```text
  healthy mean = (1.730 + 1.737 + 1.726 + 1.736) / 4 = 1.73225
  disease mean = (1.515 + 1.453 + 1.547 + 1.489) / 4 = 1.50100
  difference   = 0.23125 nats
```

A 0.23 nat difference could be real, or it could be a depth artifact. The correct next step is
rarefaction to 9,000 reads per sample and a re-computation. If the difference persists, it is not
a depth artifact. If it disappears, the original claim was technical.

This is the single most common error in microbiome papers, and it is visible before any
differential abundance test.

### 6.3 Step 2: Bray-Curtis distances

The Bray-Curtis dissimilarity between samples \(i\) and \(j\) is:

```text
  BC(i, j) = sum_k |x_ik - x_jk| / sum_k (x_ik + x_jk)
```

The distance matrix shows two blocks:

| | H1 | H2 | H3 | H4 | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | 0.000 | 0.035 | 0.028 | 0.012 | 0.378 | 0.396 | 0.370 | 0.380 |
| H2 | 0.035 | 0.000 | 0.063 | 0.025 | 0.400 | 0.427 | 0.383 | 0.410 |
| H3 | 0.028 | 0.063 | 0.000 | 0.038 | 0.377 | 0.386 | 0.369 | 0.379 |
| H4 | 0.012 | 0.025 | 0.038 | 0.000 | 0.379 | 0.406 | 0.372 | 0.390 |
| D1 | 0.378 | 0.400 | 0.377 | 0.379 | 0.000 | 0.039 | 0.024 | 0.016 |
| D2 | 0.396 | 0.427 | 0.386 | 0.406 | 0.039 | 0.000 | 0.062 | 0.023 |
| D3 | 0.370 | 0.383 | 0.369 | 0.372 | 0.024 | 0.062 | 0.000 | 0.039 |
| D4 | 0.380 | 0.410 | 0.379 | 0.390 | 0.016 | 0.023 | 0.039 | 0.000 |

Within-group distances are 0.012 to 0.063. Between-group distances are 0.369 to 0.427. The ratio
of the means is about 0.393 / 0.035 = 11.2, so the groups are separated by a factor of eleven in
this metric.

### 6.4 Step 3: PERMANOVA and its minimum p-value

A permutation test on the distance matrix gives a pseudo-F of 435.732 and a permutation p of
0.0250 with 999 permutations.

The p-value deserves a check. With four samples in each group, the number of distinct labelled
partitions of eight samples into two groups of four is:

```text
  C(8, 4) = 70
```

Each partition is counted once by choosing which four samples are in the disease group, and the
groups are labelled, so there are 70 distinct assignments. The smallest attainable p-value, if
the observed partition is the most extreme, is 1/70 = 0.0143. A reported p of 0.025 is therefore
attainable but not the minimum, which is consistent with a strong but not perfect separation.

If the study had three per group, there would be C(6,3) = 20 partitions and a minimum p of 0.05,
so no result could reach conventional significance. That is a design limitation, not a finding.

### 6.5 Step 4: the compositional check

Consider two communities with identical absolute counts of taxon B (500 cells) and taxon C (400
cells). Only taxon A differs: 100 cells in group 1 and 900 cells in group 2.

| Group | Total | B relative | B divided by A | CLR of B |
|---|---:|---:|---:|---:|
| Group 1 | 1,000 | 50.0% | 5.000 | +0.611 |
| Group 2 | 1,800 | 27.8% | 0.556 | -0.122 |

The absolute abundance of B did not change. Its relative abundance fell by 22.2 percentage points,
and its ratio to A fell nine-fold. The CLR fell by 0.733 units.

The correct interpretation is relational: B is unchanged in absolute terms and lower relative to
A. A test on relative abundance that reports "B decreased" is reporting an arithmetic consequence
of A's increase, not a change in B. The CLR reports the same relational change without implying an
absolute one, which is why the effect must be described as a log-ratio, not as an abundance change.

### 6.6 Diagnostic 1: depth confounding of alpha diversity

The first diagnostic that would reveal a broken analysis is the relationship between sequencing
depth and the diversity metric, stratified by group.

A working study shows similar depth distributions across groups, or reports the analysis after
rarefaction. A broken study shows one of three patterns:

- **Depth differs by group, and diversity differs in the same direction.** In this dataset,
  healthy samples average 13,066 reads and disease samples 9,282, while Shannon is 1.732 and 1.501.
  The confound is present. Rarefy to 9,000 reads and recompute before claiming a diversity
  difference.
- **Depth correlates with diversity within group.** If Shannon rises with depth within the
  healthy group, the metric is depth-sensitive at this sequencing level, and the between-group
  comparison needs a depth covariate.
- **Richness is identical but Shannon differs.** With eight taxa in every sample, richness cannot
  discriminate; the difference is in evenness, driven by the bloom of Escherichia in disease. That
  is a composition change, not a loss of richness, and the report should say so.

### 6.7 Diagnostic 2: negative control contamination

The second diagnostic is the comparison between the sample signal and the negative control.

Suppose the extraction blank yielded 500 reads, of which 300 are Halomonas. A sample yielded
50,000 reads, of which 150 are Halomonas.

```text
  blank Halomonas fraction  = 300 / 500 = 60.0%
  sample Halomonas fraction = 150 / 50,000 = 0.300%
```

If DNA from the blank entered the sample library in proportion to its read share, the expected
contaminant fraction in a 50,000-read sample would be:

```text
  300 / 50,000 = 0.600%
```

The observed 0.300% is below the expected contamination contribution. The sample's Halomonas
signal is therefore compatible with contamination and should not be reported as a biological
finding. The same logic applies to other genera that are common in kits.

The diagnostic is not "is the blank clean". No blank is clean. The diagnostic is "is the sample
signal above what the blank predicts".

### 6.8 The varied assumption

Change one assumption: the study used 16S amplicon sequencing at genus level rather than shotgun
metagenomics, and the claim is that a specific Escherichia coli strain is associated with disease.

The genus-level result may be unchanged: Escherichia rises from about 300 reads in healthy samples
to about 2,650 in disease samples. But 16S cannot resolve strain, and the species and strain
composition within the genus is unknown. The conclusion changes from "a strain is associated with
disease" to "the genus Escherichia is more abundant in disease", which is a weaker and different
claim.

A second change: if the disease samples were collected after bowel preparation and the healthy
samples were not, the between-group distance of about 0.39 may reflect the preparation rather than
the disease. The conclusion changes from "disease alters the community" to "sampling protocol
differs between groups", and the correct action is to discard the comparison, not to adjust for it.

## 7. Code, query, or calculation

The script below computes alpha diversity, CLR, Bray-Curtis distances, PERMANOVA, a compositional
check, and a contamination comparison, using only the Python standard library. The data are
synthetic and constructed to make the diagnostics visible. The output was produced by running the
script as written.

```python
"""micro.py - alpha diversity, CLR, Bray-Curtis, PERMANOVA, compositionality.

Standard library only. Run: python micro.py
"""
import math
import random

TAXA = ["Bacteroides", "Faecalibacterium", "Akkermansia", "Escherichia",
        "Prevotella", "Roseburia", "Blautia", "Halomonas"]

SAMPLES = {
    "H1": [4200, 3100, 900, 300, 1500, 1800, 1200, 20],
    "H2": [3900, 3400, 850, 280, 1600, 1750, 1300, 25],
    "H3": [4400, 2900, 950, 320, 1400, 1900, 1150, 18],
    "H4": [4100, 3200, 880, 310, 1550, 1820, 1220, 22],
    "D1": [4100, 400, 300, 2600, 500, 600, 700, 30],
    "D2": [4300, 350, 280, 2900, 450, 550, 650, 22],
    "D3": [4000, 450, 320, 2400, 520, 620, 720, 28],
    "D4": [4200, 380, 290, 2700, 480, 580, 680, 26],
}
GROUPS = {s: ("healthy" if s.startswith("H") else "disease") for s in SAMPLES}


def shannon(counts):
    total = sum(counts)
    return -sum((c / total) * math.log(c / total) for c in counts if c > 0)


def simpson(counts):
    total = sum(counts)
    return 1.0 - sum((c / total) ** 2 for c in counts)


print("ALPHA DIVERSITY")
print("  sample  depth  richness  shannon  simpson")
for sid, counts in SAMPLES.items():
    print(f"  {sid:<6} {sum(counts):6d} {sum(1 for c in counts if c>0):9d} "
          f"{shannon(counts):8.3f} {simpson(counts):8.3f}")


# ------------------------------------------------------------------------ CLR
def clr(counts, pseudocount=0.5):
    logs = [math.log(c + pseudocount) for c in counts]
    g = sum(logs) / len(logs)
    return [x - g for x in logs]


CLR = {sid: clr(c) for sid, c in SAMPLES.items()}
print("\nCLR (first three taxa, pseudocount 0.5)")
print("  sample  " + "".join(f"{t[:9]:>11}" for t in TAXA[:3]))
for sid, vals in CLR.items():
    print(f"  {sid:<6}  " + "".join(f"{v:11.2f}" for v in vals[:3]))


# --------------------------------------------------------------- Bray-Curtis
def bray_curtis(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / sum(x + y for x, y in zip(a, b))


ids = list(SAMPLES)
D = {(i, j): bray_curtis(SAMPLES[i], SAMPLES[j]) for i in ids for j in ids}
print("\nBRAY-CURTIS DISSIMILARITY")
print("        " + "".join(f"{j:>7}" for j in ids))
for i in ids:
    print(f"  {i:<6}" + "".join(f"{D[(i, j)]:7.3f}" for j in ids))


# ------------------------------------------------------------------ PERMANOVA
def permanova(labels, dist, permutations=999, seed=11):
    n = len(labels)
    groups = sorted(set(labels))

    def pseudo_f(assign):
        ssw = 0.0
        for g in groups:
            members = [i for i in range(n) if assign[i] == g]
            cnt = len(members)
            if cnt < 2:
                continue
            s = sum(dist[(members[a], members[b])] ** 2
                    for a in range(cnt) for b in range(a + 1, cnt))
            ssw += s / cnt
        total = sum(dist[(a, b)] ** 2 for a in range(n) for b in range(a + 1, n)) / n
        sst = total - ssw
        return (sst / (len(groups) - 1)) / (ssw / (n - len(groups)))

    observed = pseudo_f(labels)
    rng = random.Random(seed)
    perm = labels[:]
    count = 0
    for _ in range(permutations):
        rng.shuffle(perm)
        if pseudo_f(perm) >= observed:
            count += 1
    return observed, (count + 1) / (permutations + 1)


labels = [GROUPS[s] for s in ids]
idx = {s: i for i, s in enumerate(ids)}
dist = {(idx[a], idx[b]): D[(a, b)] for a in ids for b in ids}
F, p = permanova(labels, dist)
print("\nPERMANOVA (Bray-Curtis, 999 permutations, 4 per group)")
print(f"  pseudo-F          : {F:.3f}")
print(f"  permutation p     : {p:.4f}")
print(f"  minimum attainable: {1/70:.4f} (70 balanced partitions of 8 into 4 and 4)")


# ------------------------------------------- compositionality demonstration
# Two communities with identical absolute counts of taxon B (500) and taxon C
# (400). Only taxon A differs: 100 versus 900.
COMM = {"group_1": [100, 500, 400], "group_2": [900, 500, 400]}


def clr_positive(counts):
    logs = [math.log(c) for c in counts]
    g = sum(logs) / len(logs)
    return [x - g for x in logs]


print("\nCOMPOSITIONALITY CHECK (taxa A, B, C; B is constant at 500)")
print("  group    total   B_relative   B_over_A   clr_B")
for name, counts in COMM.items():
    total = sum(counts)
    b_rel = counts[1] / total
    b_over_a = counts[1] / counts[0]
    clr_b = clr_positive(counts)[1]
    print(f"  {name:<8} {total:5d} {b_rel:11.1%} {b_over_a:10.3f} {clr_b:7.3f}")


# --------------------------------------------------- contamination arithmetic
blank_reads, blank_contaminant = 500, 300
sample_reads, sample_contaminant = 50_000, 150
print("\nKIT CONTAMINATION")
print(f"  blank Halomonas fraction          : {blank_contaminant/blank_reads:.1%}")
print(f"  sample Halomonas fraction         : {sample_contaminant/sample_reads:.3%}")
print(f"  blank-to-sample signal ratio      : "
      f"{(blank_contaminant/blank_reads)/(sample_contaminant/sample_reads):.1f}x")
```

Output:

```text
ALPHA DIVERSITY
  sample  depth  richness  shannon  simpson
  H1      13020         8    1.730    0.793
  H2      13105         8    1.737    0.797
  H3      13038         8    1.726    0.790
  H4      13102         8    1.736    0.795
  D1       9230         8    1.515    0.707
  D2       9502         8    1.453    0.690
  D3       9058         8    1.547    0.717
  D4       9336         8    1.489    0.700

CLR (first three taxa, pseudocount 0.5)
  sample    Bacteroid  Faecaliba  Akkermans
  H1             1.62       1.31       0.08
  H2             1.51       1.38      -0.01
  H3             1.67       1.25       0.14
  H4             1.57       1.32       0.03
  D1             2.01      -0.32      -0.60
  D2             2.14      -0.37      -0.59
  D3             1.97      -0.21      -0.55
  D4             2.07      -0.33      -0.60

BRAY-CURTIS DISSIMILARITY
             H1     H2     H3     H4     D1     D2     D3     D4
  H1      0.000  0.035  0.028  0.012  0.378  0.396  0.370  0.380
  H2      0.035  0.000  0.063  0.025  0.400  0.427  0.383  0.410
  H3      0.028  0.063  0.000  0.038  0.377  0.386  0.369  0.379
  H4      0.012  0.025  0.038  0.000  0.379  0.406  0.372  0.390
  D1      0.378  0.400  0.377  0.379  0.000  0.039  0.024  0.016
  D2      0.396  0.427  0.386  0.406  0.039  0.000  0.062  0.023
  D3      0.370  0.383  0.369  0.372  0.024  0.062  0.000  0.039
  D4      0.380  0.410  0.379  0.390  0.016  0.023  0.039  0.000

PERMANOVA (Bray-Curtis, 999 permutations, 4 per group)
  pseudo-F          : 435.732
  permutation p     : 0.0250
  minimum attainable: 0.0143 (70 balanced partitions of 8 into 4 and 4)

COMPOSITIONALITY CHECK (taxa A, B, C; B is constant at 500)
  group    total   B_relative   B_over_A   clr_B
  group_1   1000       50.0%      5.000   0.611
  group_2   1800       27.8%      0.556  -0.122

KIT CONTAMINATION
  blank Halomonas fraction          : 60.0%
  sample Halomonas fraction         : 0.300%
  blank-to-sample signal ratio      : 200.0x
```

Reading the output:

- Richness is 8 in every sample, so the diversity difference is entirely in evenness, driven by
  the Escherichia bloom. The correct description is a composition shift, not a loss of richness.
- The CLR values separate the groups more cleanly than the raw proportions: Faecalibacterium falls
  from about 1.3 in healthy samples to about -0.3 in disease, a shift of about 1.6 log units.
- The PERMANOVA p of 0.025 is attainable given 70 partitions, but it is close to the floor of
  0.0143. A study with this design cannot report a p-value below 0.0143, and a report of
  p < 0.001 would be impossible.
- The compositionality check shows the same absolute count of taxon B in both groups with a
  22.2-point fall in relative abundance. Any method that reports this as a change in B is
  reporting arithmetic.
- The contamination check shows the sample's Halomonas signal is below what the blank predicts,
  so it should not be interpreted biologically.

## 8. Data, standards, and regulatory context

### 8.1 MIxS

The Minimum Information about any (x) Sequence standard, maintained by the Genomic Standards
Consortium, defines metadata fields for marker genes, metagenomes, and environmental samples. The
current release is MIxS 7, which adds a whole-submission container, permanent term identifiers,
and an ancient DNA extension.

The host-associated extension includes fields that matter for analysis: host taxon, body site,
collection time, diet, medication, and storage. A dataset without these fields cannot be adjusted
for the confounders that dominate microbiome associations.

MIxS is a community standard, not a regulation. Journals and repositories increasingly require it,
and a submission without it may be rejected by the repository rather than by a regulator.

### 8.2 QIIME 2 and the analysis stack

QIIME 2 is an open-source pipeline that wraps denoising, taxonomy assignment, diversity, and
ordination in a provenance-tracking framework. Every artifact records the command, the version,
and the inputs, which is the practical answer to the reproducibility problem.

The stack around it includes DADA2 for amplicon denoising, Kraken2 and MetaPhlAn 4 for shotgun
taxonomy, and ANCOM-BC2 and ALDEx2 for differential abundance. Version drift matters: a change in
a reference database or a default parameter can change the result, so the version belongs in the
methods section.

### 8.3 Reporting standards

The STORMS checklist, published in 2021, defines the minimum reporting for human microbiome
studies: study design, sample collection, DNA extraction, sequencing, bioinformatics, and
statistics. It is the microbiome analogue of CONSORT, and it exists because the literature was
unreproducible without it.

A report that omits the extraction kit, the sequencing depth, the reference database, and the
differential abundance method cannot be reproduced or compared.

### 8.4 Regulatory context for live biotherapeutic products

The FDA regulates live biotherapeutic products under the biologics framework. The 2016 guidance on
early clinical trials with LBPs describes the chemistry, manufacturing, and controls information
required, including strain identity, purity, and the absence of transmissible antibiotic
resistance.

Two products have been approved:

| Product | Approval | Composition | Indication |
|---|---|---|---|
| Rebyota | FDA, 30 November 2022 | Fecal microbiota, live | Recurrent Clostridioides difficile infection |
| Vowst | FDA, 26 April 2023 | Fecal microbiota spores, live | Recurrent Clostridioides difficile infection |

Both approvals depend on donor screening, manufacturing controls, and characterization of the
product community. The characterization is a sequencing problem: the product must be consistent
across lots, and consistency is measured by taxonomic and functional profiling.

### 8.5 What the standards do not cover

- No standard defines a healthy microbiome, because none exists.
- No standard defines a dysbiosis threshold.
- No standard defines the differential abundance method to use.
- No standard resolves the compositional interpretation problem.

Those gaps are why the diagnostics in section 6 and the tests in section 5.14 matter.

## 9. Industry practice and case evidence

### 9.1 Approved products and the FMT boundary

Two products and one procedure define the commercial landscape.

| Product or procedure | Status | What it is | Analytics that support it |
|---|---|---|---|
| Rebyota | FDA approved 30 November 2022 | Fecal microbiota, live, administered rectally | Donor screening, community characterization, lot consistency |
| Vowst | FDA approved 26 April 2023 | Fecal microbiota spores, live, oral | Spore fraction characterization, lot consistency |
| Fecal microbiota transplantation | Procedure, regulated as an investigational drug when not under enforcement discretion | Transfer of donor stool | Donor screening, pathogen testing |

The distinction between a product and a procedure is regulatory, not biological. A product has a
defined manufacturing process, release tests, and lot-to-lot consistency data. A procedure has a
donor and a patient. The analytics differ accordingly: a product requires a characterization that
can be repeated across lots, and a procedure requires a donor screen that can be repeated across
donations.

### 9.2 The immunotherapy evidence

Three 2018 papers reported that the gut microbiome composition of patients with melanoma was
associated with response to anti-PD-1 therapy. The studies differed in cohort, sequencing method,
and the taxa they highlighted, and the overlap between their reported signatures was limited.
That divergence is itself informative: it shows how sensitive these associations are to cohort,
method, and analysis choices.

Subsequent work tested whether the association could be manipulated. Small trials of fecal
microbiota transplantation from responders to non-responders reported clinical responses in a
subset of patients. Those trials are early, small, and not definitive, and they are the correct
kind of evidence for a causal question, unlike the original cross-sectional associations.

The pattern is a template for the field: an association is reported, the taxa do not replicate
across cohorts, and the causal question is tested with an intervention. A program that stops at
the first step has a biomarker hypothesis, not a therapy.

### 9.3 Direct-to-consumer testing

Direct-to-consumer microbiome tests report a community profile and, often, a set of
"personalized" recommendations about diet or probiotics. The analytical and clinical validity of
these reports is generally not established. A 2019 bankruptcy of one such company left users
unable to access their data, which illustrates the data-governance risk as much as the scientific
one.

The technical problems are the ones in this topic: compositionality, batch effects, and the
absence of a healthy reference. A consumer report that compares a sample to a company-specific
reference is reporting a rank within that reference, not a health status.

### 9.4 Where the field is going

Three directions are visible in the 2024 to 2026 literature.

1. **Strain-resolved analysis.** Shotgun metagenomics and isolate genomes allow strain tracking,
   which matters for engraftment after FMT and for product characterization.
2. **Metabolites as the mechanistic link.** Short-chain fatty acids, bile acids, and
   trimethylamine N-oxide are measured directly rather than inferred from taxonomy, which moves the
   claim from association to mechanism.
3. **Causal designs.** Germ-free colonization, defined consortia, and Mendelian randomization are
   used to test whether a taxon or a pathway causes the outcome.

None of these removes the compositional problem. They change the question from "which taxon" to
"which function" and "which strain", which are more tractable but still require the same
statistical care.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Compositional analysis ignored | Negative correlations between unrelated taxa | Constant-sum constraint | Spurious associations | CLR or ILR before correlation |
| Depth confounds diversity | Cases sequenced less deeply | More reads observe more taxa | Technical difference read as biology | Rarefy or adjust for depth |
| Batch confounded with group | Cases and controls processed on different dates | Batch effect | Group difference is a batch difference | Balance batches; include batch in the model |
| No negative control | No blank sequenced | Kit contamination | Contaminant genus reported as a taxon | Sequence blanks; compare fractions |
| Database mismatch | Unexpected taxa or missing phyla | Classifier reference does not cover the environment | Wrong taxonomy | Use an environment-appropriate database |
| Genus-level assay, species claim | Species named from 16S data | Resolution limit | Overclaimed identification | Use shotgun for species and strain |
| Pseudoreplication | Many samples from few subjects | Repeated measures treated as independent | Inflated significance | Mixed models or subject-level summaries |
| PERMANOVA without dispersion check | Groups differ in variability | Heteroscedasticity | Significant test with no centroid difference | Test dispersion; report it |
| Impossibly small p-value | p below 1/number of partitions | Permutation test floor ignored | Irreproducible result | Compute the minimum attainable p |
| Unadjusted confounders | No medication or diet data | Disease, diet, and drugs covary | Confounded association | Collect and adjust metadata |
| Functional prediction reported as measurement | PICRUSt2 output in the results | Inference from taxonomy | Unsupported mechanism | Label as predicted; validate where possible |
| Multiple testing ignored | Many taxa, no adjustment | Hundreds of tests | False discoveries | FDR control with a stated method |
| Relative abundance reported as absolute | "Taxon X decreased" | Compositional effect | Wrong biological claim | Report ratios or absolute quantification |
| Causal language from cross-sectional data | "Causes", "drives" in the abstract | Design cannot support it | Failed translation | Match language to the inference ladder |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The sample represents the community of interest | Inference is about the site, not the aliquot | Spatial heterogeneity, sampling bias | Multiple samples per site |
| The assay detects the taxa of interest | Conclusions depend on what is measured | Primer bias, database gaps | Spike-in controls; mock communities |
| Read depth is adequate and comparable | Diversity and detection depend on depth | Low or unequal depth | Depth distribution; rarefaction |
| Contamination is negligible relative to signal | Low-biomass taxa are otherwise uninterpretable | Kit DNA dominates | Negative controls; fraction comparison |
| The composition is comparable across samples | Beta diversity assumes a common feature space | Different pipelines or databases | Single pipeline; version pinning |
| Zero means absent or below detection | Many methods treat zeros as sampling zeros | True absence versus dropout | Zero-inflation models; sensitivity analysis |
| The reference database is appropriate | Taxonomy depends on it | Environment not represented | Mock community; alternative database |
| Covariates capture the confounders | Adjustment requires the variable | Unmeasured diet or medication | Collect metadata; sensitivity analysis |
| The statistical method matches the data type | Tests assume a distribution | Compositional data analyzed as Gaussian | CLR plus a compositional method |
| The study design supports the claim | Causation requires intervention | Cross-sectional data | Match claim to the inference ladder |

## 12. Exercises

### Exercise 1. Compute alpha diversity and check depth

**Task.** For the eight samples in section 6, compute the mean depth by group and the mean Shannon
index by group. State whether the diversity comparison is confounded.

**Deliverable.** The arithmetic and a conclusion.

**Acceptance criteria.** Healthy mean depth 13,066.25 reads and Shannon 1.732; disease mean depth
9,281.5 reads and Shannon 1.501. The answer states that the 41% depth difference confounds the
comparison and that rarefaction is required.

**Time.** 25 minutes.

### Exercise 2. Compute Bray-Curtis by hand

**Task.** For two samples with counts A = (10, 20, 30) and B = (20, 20, 20), compute the
Bray-Curtis dissimilarity.

**Deliverable.** The calculation.

**Acceptance criteria.** Numerator = 10 + 0 + 10 = 20; denominator = 30 + 40 + 50 = 120;
dissimilarity = 20/120 = 0.1667. The answer notes that the metric is bounded between 0 and 1.

**Time.** 15 minutes.

### Exercise 3. Apply CLR

**Task.** For the composition (100, 500, 400), compute the CLR values and verify that they sum to
zero.

**Deliverable.** The calculation.

**Acceptance criteria.** Geometric mean = (20,000,000)^(1/3) = 271.44; CLR values are
ln(100/271.44) = -0.998, ln(500/271.44) = 0.611, and ln(400/271.44) = 0.387; the sum is 0.000
within rounding.

**Time.** 20 minutes.

### Exercise 4. Design a contamination control

**Task.** Design the negative controls for a study of the gut microbiome in infants, where biomass
is low and contamination is a major risk.

**Deliverable.** A control plan with at least four control types and the analysis that uses each.

**Acceptance criteria.** The plan includes an extraction blank, a no-template polymerase chain
reaction control, a mock community, and a sampling blank, and states how each is used to estimate
or bound contamination. A plan with only a no-template control is incomplete.

**Time.** 35 minutes.

### Exercise 5. Choose a differential abundance method

**Task.** A study has 120 samples in three groups, repeated measures on 40 subjects, and 600 taxa.
Recommend a method and justify it, including the multiple testing correction.

**Deliverable.** A one-page methods note.

**Acceptance criteria.** The note recommends a compositional method that handles repeated measures
and multigroup comparisons, such as ANCOM-BC2 with a subject-level random effect and mixed
directional FDR control, or an equivalent. It states the sensitivity analysis and the version. A
note that recommends a Wilcoxon test on proportions fails.

**Time.** 40 minutes.

### Exercise 6. Diagnose an impossible p-value

**Task.** A paper reports a PERMANOVA p-value of 0.0002 with 5 samples per group and two groups.
Is the result possible? Show the calculation.

**Deliverable.** The arithmetic and a conclusion.

**Acceptance criteria.** The number of labelled partitions is C(10,5) = 252, so the minimum
attainable p is 1/252 = 0.0040. A p of 0.0002 is impossible, which indicates an error in the
permutation procedure or the reported value.

**Time.** 20 minutes.

### Exercise 7. Design a causal study

**Task.** Design a study to test whether a candidate taxon causes protection against
inflammation, starting from a cross-sectional association.

**Deliverable.** A two-page design with at least two causal approaches.

**Acceptance criteria.** The design includes an intervention in an animal model with germ-free or
antibiotic-treated animals, a defined consortium rather than a single strain where appropriate, a
metabolite measurement, and a positive and negative control. It states the limitations of each
approach.

**Time.** 60 minutes.

### Exercise 8. Critique a consumer report

**Task.** A consumer report says "Your gut diversity is low, and you have 30% less
Faecalibacterium than the average person. Take our probiotic." Write a critique.

**Deliverable.** A 350-word critique.

**Acceptance criteria.** The critique asks about the reference population, the sequencing method,
depth, the compositional interpretation, the evidence for the probiotic, and the absence of a
clinical claim. It notes that a rank within a company reference is not a health status.

**Time.** 30 minutes.

## 13. Self-check answers

**Exercise 1.** Healthy depths are 13,020, 13,105, 13,038, and 13,102, which sum to 52,265, for a
mean of 13,066.25 reads. Disease depths are 9,230, 9,502, 9,058, and 9,336, which sum to 37,126, for
a mean of 9,281.5 reads. The ratio is 1.41. Healthy Shannon values sum to 6.929, mean 1.73225;
disease values sum to 6.004, mean 1.501. The 0.231 nat difference is confounded by the 41% depth
difference, and the comparison requires rarefaction to a common depth.

**Exercise 2.** The absolute differences are |10 - 20| = 10, |20 - 20| = 0, and |30 - 20| = 10,
which sum to 20. The sums are 30, 40, and 50, which sum to 120. Bray-Curtis is 20/120 = 0.1667. The
communities are similar but not identical.

**Exercise 3.** The product is 100 x 500 x 400 = 20,000,000. The cube root is 271.44. The CLR values
are ln(0.3684) = -0.9985, ln(1.8420) = 0.6109, and ln(1.4736) = 0.3876. Their sum is 0.0000 within
rounding, which is the defining property of CLR.

**Exercise 4.** Rubric. Full credit requires at least four control types, a statement of what each
estimates, and an analysis plan that compares sample fractions against the blank. A plan that
sequences a blank but does not use it in the analysis fails.

**Exercise 5.** Rubric. Full credit recommends a compositional method with repeated-measures
support and a multiple-testing correction appropriate to multigroup designs, names the version, and
requires the sensitivity analysis. A method that ignores the repeated measures or the compositional
structure fails.

**Exercise 6.** With five per group and two groups, the number of ways to choose five of ten is
252. The groups are labelled, so there are 252 distinct assignments. The minimum attainable
p-value is 1/252 = 0.00397. A reported p of 0.0002 is below the floor and is therefore impossible
under a correct permutation procedure. The likely causes are a parametric approximation reported
as a permutation p, a one-sided count error, or a transcription error.

**Exercise 7.** Rubric. Full credit includes at least two causal designs, such as colonization of
germ-free animals with a defined consortium and a metabolite supplementation experiment, with
controls and a stated limitation. A design that repeats the cross-sectional association with more
samples fails.

**Exercise 8.** Rubric. Full credit questions the reference population, the assay and depth, the
compositional interpretation, the probiotic evidence, and the absence of a clinical claim. A
critique that argues only about the probiotic brand fails.

## 14. Interview questions

**1. Why is a microbiome count table compositional?**
A strong answer explains the fixed sequencing budget and the constant-sum constraint, and gives an
example of a spurious negative association. A weak answer says the data are "relative".

**2. What does CLR do, and what does it not do?**
A strong answer describes the log-ratio to the geometric mean, scale invariance, and the fact that
it does not recover absolute abundance. A weak answer says it "normalizes" the data.

**3. Why can alpha diversity differ for technical reasons?**
A strong answer explains depth and its effect on observed taxa, and proposes rarefaction or a depth
covariate. A weak answer says diversity is biological.

**4. What is the minimum attainable p-value in a permutation test, and why does it matter?**
A strong answer computes it from the number of distinct partitions and uses it to flag impossible
results. A weak answer has not considered the question.

**5. What is the difference between Bray-Curtis and UniFrac?**
A strong answer explains that UniFrac uses a phylogenetic tree and comes in weighted and
unweighted forms, and states when each is appropriate. A weak answer says both measure distance.

**6. How do you handle zeros in a compositional analysis?**
A strong answer distinguishes sampling zeros from structural zeros, names pseudocount and
zero-inflation approaches, and requires a sensitivity analysis. A weak answer says "add one".

**7. How would you detect kit contamination?**
A strong answer uses extraction blanks and no-template controls, compares fractions, and names
common contaminant genera. A weak answer says "use a clean kit".

**8. Why do differential abundance methods disagree?**
A strong answer cites the 2022 comparison across 38 datasets and explains differences in
compositional handling, zero models, and multiple testing. A weak answer says one method is simply
better.

**9. What evidence would support a causal claim about a taxon?**
A strong answer names intervention, germ-free colonization, defined consortia, and Mendelian
randomization, with their assumptions. A weak answer says "a strong association".

**10. What does MIxS require, and why does it matter?**
A strong answer describes the minimum metadata fields, the host-associated extension, and the
consequence for adjusting confounders and reproducing an analysis. A weak answer says it is "a
checklist".

## 15. Key terms

- **16S ribosomal RNA** — marker gene used for bacterial taxonomy.
- **Alpha diversity** — within-sample diversity.
- **Amplicon sequence variant** — exact sequence variant from denoising.
- **ANCOM-BC2** — compositional differential abundance method with bias correction.
- **Beta diversity** — between-sample dissimilarity.
- **Bray-Curtis dissimilarity** — abundance-based dissimilarity metric.
- **Centered log-ratio** — log transform relative to the geometric mean.
- **Chao1** — richness estimator based on singletons and doubletons.
- **Compositional data** — data whose parts sum to a constant.
- **Depth** — reads per sample.
- **Differential abundance** — testing taxa for group differences.
- **Dysbiosis** — undefined term for a community differing from a reference.
- **False discovery rate** — expected proportion of false positives among discoveries.
- **Fecal microbiota transplantation** — transfer of donor stool.
- **Isometric log-ratio** — orthonormal log-ratio coordinates.
- **Kit contamination** — DNA introduced by reagents and kits.
- **Live biotherapeutic product** — regulated product containing live organisms.
- **Metagenome** — collective genetic material of a community.
- **MetaPhlAn 4** — marker-based species profiler.
- **MIxS** — minimum metadata standard for sequences.
- **Mixed directional FDR** — FDR accounting for direction across comparisons.
- **Negative control** — blank used to estimate contamination.
- **Operational taxonomic unit** — cluster of similar sequences.
- **PERMANOVA** — permutation test on a distance matrix.
- **Principal coordinates analysis** — ordination of distances.
- **QIIME 2** — provenance-tracking microbiome analysis pipeline.
- **Rarefaction** — subsampling to equal depth.
- **Relative abundance** — proportion of reads for a taxon.
- **Shannon index** — entropy of the abundance distribution.
- **Simpson index** — probability two reads differ in taxonomy.
- **STORMS** — reporting checklist for microbiome studies.
- **Strain** — genetic variant within a species.
- **UniFrac** — phylogenetic distance metric.
- **Whole-genome shotgun** — sequencing all DNA in a sample.

## 16. Sources

Accessed 26 October 2026 unless otherwise stated.

- Aitchison, J. *The Statistical Analysis of Compositional Data*. Journal of the Royal Statistical
  Society, Series B 44, 139-160, 1982. https://doi.org/10.1111/j.2517-6161.1982.tb01195.x
- Bolyen, E., et al. *Reproducible, Interactive, Scalable and Extensible Microbiome Data Science
  Using QIIME 2*. Nature Biotechnology 37, 852-857, 2019. https://doi.org/10.1038/s41587-019-0209-9
- Callahan, B. J., et al. *DADA2: High-Resolution Sample Inference from Illumina Amplicon Data*.
  Nature Methods 13, 581-583, 2016. https://doi.org/10.1038/nmeth.3869
- Lin, H., and Peddada, S. D. *Multigroup Analysis of Compositions of Microbiomes with Covariate
  Adjustments and Repeated Measures*. Nature Methods, 2023.
  https://doi.org/10.1038/s41592-023-02092-7
- Lin, H., and Peddada, S. D. *Analysis of Compositions of Microbiomes with Bias Correction*.
  Nature Communications 11, 3514, 2020. https://doi.org/10.1038/s41467-020-17041-7
- Lozupone, C., and Knight, R. *UniFrac: A New Phylogenetic Method for Comparing Microbial
  Communities*. Applied and Environmental Microbiology 71, 8228-8235, 2005.
  https://doi.org/10.1128/AEM.71.12.8228-8235.2005
- Nearing, J. T., et al. *Microbiome Differential Abundance Methods Produce Different Results
  across 38 Datasets*. Nature Communications 13, 342, 2022.
  https://doi.org/10.1038/s41467-022-28034-z
- Mirzayi, C., et al. *Reporting Guidelines for Human Microbiome Research: The STORMS Checklist*.
  Nature Medicine 27, 1885-1892, 2021. https://doi.org/10.1038/s41591-021-01552-x
- Genomic Standards Consortium. *MIxS: Minimum Information about any (x) Sequence*.
  https://genomicsstandardsconsortium.github.io/mixs/
- Blanco-Miguez, A., et al. *Extending and Improving Metagenomic Taxonomic Profiling with
  Uncharacterized Species Using MetaPhlAn 4*. Nature Biotechnology, 2023.
  https://doi.org/10.1038/s41587-023-01688-w
- Wood, D. E., Lu, J., and Langmead, B. *Improved Metagenomic Analysis with Kraken 2*. Genome
  Biology 20, 257, 2019. https://doi.org/10.1186/s13059-019-1891-0
- United States Food and Drug Administration. *Early Clinical Trials with Live Biotherapeutic
  Products: Chemistry, Manufacturing, and Control Information*, 2016.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/early-clinical-trials-live-biotherapeutic-products-chemistry-manufacturing-and-control-information
- United States Food and Drug Administration. *Rebyota Approval*, 30 November 2022.
  https://www.fda.gov/news-events/press-announcements/fda-approves-first-fecal-microbiota-product
- United States Food and Drug Administration. *Vowst Approval*, 26 April 2023.
  https://www.fda.gov/news-events/press-announcements/fda-approves-first-oral-fecal-microbiota-product
- Routy, B., et al. *Gut Microbiome Influences Efficacy of PD-1-Based Immunotherapy against
  Epithelial Tumors*. Science 359, 91-97, 2018. https://doi.org/10.1126/science.aan3706
- Gopalakrishnan, V., et al. *Gut Microbiome Modulates Response to Anti-PD-1 Immunotherapy in
  Melanoma Patients*. Science 359, 97-103, 2018. https://doi.org/10.1126/science.aan4236

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`15_crispr-and-gene-therapy-pipelines.md`](15_crispr-and-gene-therapy-pipelines.md)
- Related topic: [`07_multi-omics-integration.md`](07_multi-omics-integration.md)
- Related topic: [`08_biomarker-discovery.md`](08_biomarker-discovery.md)
- Module survey: [`../../course/12_omics_precision_medicine.md`](../../course/12_omics_precision_medicine.md)
- Laboratory exercises and datasets: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
- Calibration and decision curve lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)

