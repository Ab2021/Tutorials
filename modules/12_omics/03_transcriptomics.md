# Transcriptomics: Bulk RNA-seq, Normalization, and Differential Expression

> Module 12 — Omics and Precision Medicine
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: 01A, 04, topics 01-02
> Version 2026.3

## 1. Why this topic exists

Differential expression is the most frequently produced analysis in molecular biology and one
of the least reproducible. A typical study reports a list of genes with p-values, a fold
change, and a pathway enrichment result. A second laboratory repeats the experiment and gets a
partially overlapping list, or no overlap at all. The reasons are rarely mysterious: the
normalization assumed something false, the batch was confounded with the condition, the
library was too shallow, the gene filter removed the signal, or the effect size was never
reported so nobody could judge whether the lists disagreed in substance or only in threshold.

The industry stake is direct. A transcriptomic signature that will be used to stratify a trial
must survive three tests: it must be measurable on the platform the trial will use, it must
replicate in an independent cohort, and it must not be a proxy for tissue composition or
processing batch. Most published signatures fail the third test and many fail the second.
Programs that skip the checks spend a phase 2 budget on a biomarker that was a batch effect
with a gene list attached.

The statistical problem is also genuinely hard. RNA-seq counts are overdispersed relative to
Poisson, heteroscedastic across the expression range, and subject to composition shifts when a
few genes dominate the library. The standard solution — a negative binomial generalized linear
model with empirical Bayes shrinkage of dispersion and a size-factor normalization — is good,
but it rests on assumptions that must be checked rather than trusted. When those assumptions
fail, the output is still a tidy table of p-values, which is what makes the failure expensive:
it looks like a result.

This topic covers bulk transcriptomics from count matrix to interpreted result, with emphasis
on normalization, dispersion, batch, composition, and multiple testing. Single-cell
transcriptomics is topic 04; spatial is topic 05; the biomarker validation path is topic 08.

### 1.1 The failure pattern

1. A count matrix is normalized with TPM because "TPM is standard."
2. A negative binomial model is fit to the TPM values, treating them as counts.
3. Batch is added as a covariate, but batch is nearly collinear with condition.
4. Low-count genes are filtered with a threshold chosen by habit, removing the genes that
   would have shown the largest relative changes.
5. Differential expression is run with no independent filtering and no shrinkage.
6. The hit list is reported at p < 0.05 with no effect sizes and no correction.
7. Pathway enrichment is run on the hit list, and the top pathway is whatever was already
   expected.
8. A validation cohort fails to replicate, and the failure is attributed to "cohort
   differences."

Every step is individually defensible and collectively fatal. The diagnostic that would have
caught it — a p-value histogram and a PCA coloured by batch — takes ten minutes.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Translational lead | A stratification hypothesis that does not replicate | Depends on the analysis team's normalization |
| Statistician | Time defending a result that was mis-specified upstream | Did not see the count matrix or the QC |
| Wet-lab lead | Reagent budget on a repeat study | Receives a hit list, not an analysis plan |
| Bioinformatics engineer | Pipeline credibility | Inherited a matrix without provenance |
| Clinician | A test that flags the wrong patients | Cannot inspect the model |
| Payer | Spend on a biomarker with no validated utility | Sees the claim, not the evidence |
| Regulator | A submission that cannot demonstrate analytical validity | Receives summarized results |

### 1.3 The industry question

> Is this molecular signal a validated biomarker, a causal target, or an artifact of batch,
> ancestry, and tissue composition?

For a differential-expression result, the question decomposes into three: is the effect larger
than technical noise, is it larger than composition change, and does it replicate outside the
discovery cohort? A result that answers only the first is a candidate. A result that answers
all three is a biomarker.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** raw counts, size-factor-scaled counts, CPM, TPM, and log-transformed values,
   and choose the model each supports.
2. **Compute** size factors by the median-of-ratios method and explain what the method assumes.
3. **Fit** a negative binomial model with covariates and interpret dispersion shrinkage.
4. **Apply** Benjamini-Hochberg correction and independent filtering, and report the number of
   tests.
5. **Detect** batch confounding, composition shifts, and RNA-quality effects from diagnostics.
6. **Estimate** a composition-adjusted effect and state what changes when composition is
   ignored.
7. **Critique** a published differential-expression analysis against a checklist.
8. **Design** a bulk RNA-seq study with power, blocking, and a pre-specified analysis plan.

## 3. The core mental model

Bulk RNA-seq measures the relative abundance of transcripts in a library. Every downstream
number is a ratio to something: a library total, a set of reference genes, or a geometric mean.
The choice of denominator is the analysis.

```text
  Tissue (mixed cell types, varying RNA quality)
      |
      v
  mRNA selection -> fragmentation -> cDNA -> sequencing
      |                                        |
      |                                        +--> library size, GC bias, 3' bias
      v
  Reads per gene (counts)
      |
      +--> normalization: what is held constant?
      |         |
      |         +--> total library (CPM, TPM)
      |         +--> median ratio (DESeq2 size factors)
      |         +--> trimmed mean (TMM)
      |         +--> geometric mean of composition (CLR)
      v
  Normalized expression
      |
      +--> model: negative binomial, quasi-likelihood, or linear on log scale
      |         |
      |         +--> dispersion (biological variability)
      |         +--> size factor (technical depth)
      |         +--> covariates (batch, sex, age, RIN, composition)
      v
  Per-gene test statistic -> p-value
      |
      +--> independent filtering -> BH correction -> q-value
      |
      v
  Effect size (shrunken log2FC) + confidence interval
      |
      v
  Interpretation: gene, pathway, signature, or classifier
```

### 3.1 The three confounders that dominate

Most failed replications trace to one of three confounders, and each has a distinct signature.

**Batch.** Samples processed in different weeks, by different operators, with different reagent
lots. Signature: a principal component that separates batches rather than conditions, and a
p-value histogram that is skewed even for genes with no biological reason to differ.

**Composition.** The proportion of cell types in the tissue changes with disease. Signature: a
large number of genes change in a coordinated direction, and the changes disappear after
deconvolution or in sorted populations.

**RNA quality.** Degraded RNA produces 3' bias and loss of long transcripts. Signature: RIN
correlates with a principal component, and the genes most affected are long and
GC-rich.

A study can survive one confounder with a good design. It cannot survive two that are
collinear with the condition, because no model identifies the separate effects.

### 3.2 What normalization assumes

Every normalization is a statement that something is constant across samples. The statement is
either true enough or it produces a specific artifact.

| Method | What it holds constant | Fails when | Artifact |
|---|---|---|---|
| Total count (CPM) | Total RNA output per sample | A few genes dominate | Spurious fold changes in all other genes |
| TPM | Total transcript output after length correction | Composition shifts | Not usable for count models |
| Median-of-ratios | Most genes unchanged | Global activation or arrest | Size factors biased; fold changes compressed |
| TMM | Trimmed mean of log ratios | Asymmetric differential expression | Bias toward the majority direction |
| Spike-in (ERCC) | Known external RNA quantity | Spike-in ratio differs from endogenous | Corrects only if spike-in behaves like endogenous |
| Reference genes | Stable housekeeping set | Reference genes regulated | Systematic bias |

The practical rule: use median-of-ratios for count-based differential expression, check it with
a spike-in or an orthogonal estimate, and report the check. A normalization that was never
checked is an assumption presented as a method.

### 3.3 Why dispersion matters more than the mean

The negative binomial model has two parameters per gene: a mean and a dispersion. The mean is
estimated from the data in that gene; the dispersion is estimated from the variance across
replicates and then shrunk toward a trend because per-gene estimates are unstable with three
to five replicates. The shrinkage is what makes small experiments analyzable, and it is also
what makes them fragile: if the dispersion trend is estimated from a set of genes that includes
a global shift, the trend is wrong and every p-value moves.

A useful diagnostic is the dispersion plot: empirical dispersion against mean expression, with
the fitted trend. Genes far above the trend are either genuinely variable or outliers. Genes far
below are suspiciously consistent, which can indicate that the counts were normalized twice.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Count | Reads assigned to a feature | Called "expression" | Integer |
| Library size | Total reads assigned to features in a sample | Confused with sequencing depth | Reads |
| Size factor | Multiplicative scaling for a sample's depth and composition | Confused with CPM scaling | Dimensionless, centered near 1 |
| CPM | Counts per million mapped reads | Used for DE directly | Counts per 10^6 |
| TPM | Transcripts per million after length normalization | Used in count models | Parts per 10^6 |
| FPKM | Fragments per kilobase million | Mixed with TPM | Parts per 10^6, not sum-consistent |
| Dispersion | Variance parameter of the negative binomial | Confused with standard deviation | Dimensionless |
| Overdispersion | Variance exceeding the Poisson mean | Ignored in Poisson models | Ratio |
| Negative binomial | Count distribution with mean and dispersion | Applied without checking fit | Model family |
| Quasi-likelihood | Model with a dispersion estimated separately | Confused with full NB | Model family |
| limma-voom | Linear model on log-CPM with precision weights | Treated as equivalent to NB in all cases | Model family |
| Design matrix | Sample-by-covariate matrix | Missing batch or donor terms | Numeric matrix |
| Contrast | A linear combination of coefficients | Confused with the coefficient itself | Vector |
| Blocking | Design that groups similar samples | Confused with batch correction | Design property |
| Batch | Group processed together | Ignored when collinear | Categorical |
| RIN | RNA integrity number | Used as a QC after the fact | 1-10 |
| 3' bias | Over-representation of 3' ends | Missed in degraded samples | Ratio 5'/3' |
| ERCC | External RNA spike-in controls | Assumed to track endogenous RNA | Known concentration |
| Housekeeping gene | Gene assumed stably expressed | Assumed without testing | Normalization reference |
| Composition | Proportion of cell types in a specimen | Ignored in bulk analysis | Proportions summing to 1 |
| Deconvolution | Estimation of cell-type proportions from bulk | Treated as ground truth | Proportions |
| log2 fold change (log2FC) | Difference in mean log2 expression | Reported without a confidence interval | log2 units |
| Shrinkage | Pulling gene-wise estimates toward a prior | Confused with normalization | Statistical procedure |
| Independent filtering | Removing low-count genes before correction | Confused with p-hacking | Filtering rule |
| FDR | Expected proportion of false discoveries | Confused with family-wise error | Proportion |
| q-value | Adjusted p-value controlling FDR | Reported as "p-value" | Probability |
| BH | Benjamini-Hochberg procedure | Applied to dependent tests without comment | Procedure |
| IHW | Independent hypothesis weighting | Applied without covariate validity | Procedure |
| GSEA | Gene set enrichment on ranked statistics | Run on a filtered gene list | Rank-based test |
| ORA | Over-representation analysis on a hit list | Threshold-dependent | Hypergeometric test |
| sva | Surrogate variable analysis | Used to remove biology | Batch method |
| RUV | Remove unwanted variation using controls | Requires valid control genes | Batch method |
| TMM | Trimmed mean of M-values | Applied to very asymmetric designs | Normalization |
| Precision weight | Weight from mean-variance relationship | Ignored in downstream models | Weight |

Two distinctions that are routinely lost.

**A q-value is not a probability that the gene is null.** It is an estimate of the expected
proportion of false discoveries among all genes at least as significant. A q-value of 0.05 on a
gene list of 100 means about five genes in that list are expected to be false, not that this
gene has a 5% chance of being false.

**Shrinkage changes effect sizes, not just p-values.** Empirical Bayes shrinkage pulls extreme
log2 fold changes toward zero, especially for low-count genes. The shrunken estimate is a
better predictor of replication than the raw estimate. Reporting the raw fold change alongside
a shrunken p-value is inconsistent and inflates apparent effect sizes.

## 5. The mechanics, step by step

### Step 1. Write the contrast before looking at the data

**Input:** the biological question and the sample sheet.

**Output:** one sentence naming the comparison, the unit of replication, and the covariates.

**Rationale:** "Response versus non-response in pre-treatment biopsies, with donor as the
experimental unit, adjusted for batch and RIN" is a contrast. "Analyze the data" is not. The
contrast determines the design matrix, the dispersion estimate, and the interpretation. Writing
it first prevents the post hoc search for a comparison that produces a result.

### Step 2. Audit the sample sheet

**Input:** sample sheet and laboratory records.

**Output:** a design table with every covariate coded, plus a missing-data map.

**Rationale:** The sample sheet is the analysis. If batch, donor, tissue site, RIN, or
processing date is missing, no downstream step can recover it. Cross-tabulate every covariate
against condition; an empty cell is a confounding problem, not a modelling problem.

### Step 3. Run read-level and library-level QC

**Input:** FASTQ and alignment summaries.

**Output:** per-sample QC table.

**Rationale:** Library size, detected genes, duplication rate, mapping rate, and 3' bias
identify samples that will distort normalization. A sample with 2 million reads in a 30-million
cohort will have unstable size factors and noisy counts. Flag it, then decide whether to
exclude with a pre-specified rule.

### Step 4. Quantify to gene level with a declared annotation

**Input:** BAM files or FASTA/FASTQ for pseudo-alignment.

**Output:** count matrix, features by samples, with annotation version.

**Rationale:** Transcript-level estimates are uncertain; gene-level counts are more stable and
sufficient for most questions. If transcript-level inference is required, use a method that
propagates ambiguity and report the uncertainty. Record the annotation release, because feature
definitions change.

### Step 5. Filter features with a documented rule

**Input:** count matrix.

**Output:** filtered matrix and a record of how many features were removed.

**Rationale:** Genes with near-zero counts carry no information and inflate the multiple-testing
burden. A common rule is to retain genes with counts in at least a minimum number of samples,
but the rule must be chosen before seeing the result. Removing genes that are low in one
condition only is a form of selection on the outcome.

### Step 6. Estimate size factors and inspect them

**Input:** filtered counts.

**Output:** size factors per sample, plus a plot against library size.

**Rationale:** Median-of-ratios computes, for each gene, the ratio of a sample's count to the
gene's geometric mean across samples, then takes the median across genes. It assumes most genes
are unchanged. If a large fraction of the transcriptome shifts — immune activation, for example
— the median ratio is biased and fold changes are compressed. Comparing size factors to library
sizes detects this: a systematic divergence means composition is driving normalization.

### Step 7. Build the design matrix with the confounders

**Input:** design table and contrast.

**Output:** design matrix.

**Rationale:** The design matrix encodes what the model will attribute to the condition and
what it will attribute to nuisance variables. Include batch, donor (for paired designs), sex if
relevant, and RIN if it varies with condition. Do not include a variable that is a consequence
of the condition; adjusting for a mediator removes the effect you are trying to measure.

### Step 8. Estimate dispersions and check the trend

**Input:** counts, size factors, design.

**Output:** gene-wise and shrunken dispersion estimates, plus the trend plot.

**Rationale:** Dispersion estimation is where small experiments gain power and lose robustness.
The trend is fit across genes; a global shift distorts it. Check that the trend is smooth and
that the shrunken estimates are not systematically below the empirical ones.

### Step 9. Fit the model and extract the contrast

**Input:** counts, design, dispersions.

**Output:** per-gene coefficients, standard errors, test statistics, p-values.

**Rationale:** The contrast, not the coefficient, is the quantity of interest. In a model with
batch, the condition coefficient is already adjusted; in an interaction model, the contrast is
a difference of differences. Extracting the wrong coefficient is a common and silent error.

### Step 10. Apply independent filtering

**Input:** per-gene p-values and mean expression.

**Output:** filtered set with a stated threshold.

**Rationale:** Low-count genes have low power and contribute mostly noise to the multiple-testing
correction. Independent filtering removes them using a statistic that is independent of the
p-value under the null, which increases the number of discoveries without inflating the FDR.
The filter must not depend on the observed effect size.

### Step 11. Correct for multiple testing

**Input:** p-values from the filtered set.

**Output:** q-values and the number of tests.

**Rationale:** Benjamini-Hochberg controls the false discovery rate under independence or
positive dependence. Report the number of tests, the procedure, and the threshold. A q-value
without the number of tests is not interpretable.

The number of tests is not always the number of genes in the matrix. If isoforms are tested
separately, or if several contrasts are tested, the effective family is larger, and the
correction must match the family that the claim covers. Reporting a q-value computed over genes
while the claim is about isoforms is a family mismatch.

### Step 12. Shrink effect sizes and report intervals

**Input:** model fit.

**Output:** shrunken log2FC with standard error or confidence interval.

**Rationale:** Raw fold changes for low-count genes are unstable and biased away from zero.
Shrinkage improves ranking and calibration. The interval communicates precision, which a
p-value does not.

### Step 13. Run diagnostics before interpreting

**Input:** fitted model.

**Output:** diagnostic set: p-value histogram, MA plot, PCA, dispersion plot, size-factor plot.

**Rationale:** The p-value histogram is the single most informative diagnostic. Under the null,
it should be flat with a spike near 1. A U-shape indicates a misspecified model or confounding.
A histogram skewed toward small p-values across all genes indicates a global shift. Neither
pattern is visible in a hit list.

The MA plot adds a second view: log fold change against mean expression. A cloud that widens at
low expression indicates that shrinkage is needed; a cloud that is asymmetric around zero
indicates a normalization bias. The PCA adds a third: sample separation by condition is
necessary but not sufficient, because a condition effect on PC2 with batch on PC1 is still
identifiable, while the reverse may not be.

### Step 14. Interpret with a pathway method matched to the design

**Input:** ranked statistics or a hit list.

**Output:** enriched sets with a correction and a stated background.

**Rationale:** GSEA uses the full ranked list and is less threshold-dependent than
over-representation analysis. ORA requires a hit list and a background, and the background
choice changes the result. Either way, the gene sets, the database version, and the correction
must be reported.

### Step 15. Run a normalization sensitivity analysis

**Input:** the filtered counts and at least two defensible normalizations.

**Output:** a table of hit counts and leading effect sizes under each normalization.

**Rationale:** If the conclusion survives median-of-ratios, TMM, and a spike-in or reference
normalization, it is robust to the normalization assumption. If it does not, the conclusion is
a property of the method, and the report must say so. The sensitivity analysis takes an hour and
prevents the most common reviewer objection.

A useful form is a small table:

| Normalization | Genes at q < 0.05 | Leading log2FC | Leading q |
|---|---:|---:|---:|
| Median-of-ratios | 812 | 2.10 | 4.8 x 10^-4 |
| TMM | 764 | 2.04 | 6.1 x 10^-4 |
| Spike-in reference | 590 | 1.82 | 2.2 x 10^-3 |

The spread is the uncertainty attributable to normalization. A conclusion that holds across all
three is stronger than any single analysis; a conclusion that exists only under one is a
method artifact until proven otherwise.

### Branch points

**Branch A — the design is paired.** Use donor as a blocking factor. Ignoring pairing inflates
the residual variance and loses power, sometimes dramatically.

**Branch B — the design has no replicates in one condition.** No dispersion estimate is
possible for that condition, and no inferential test is valid. Report descriptive statistics
only.

**Branch C — the library is dominated by a few genes.** Median-of-ratios may be biased. Use a
spike-in or a reference-based normalization, and state the assumption.

### Decision table: model choice

| Design | Recommended model | Alternative | Avoid |
|---|---|---|---|
| Two groups, n >= 3, counts | Negative binomial (DESeq2, edgeR) | limma-voom | t-test on CPM |
| Paired samples | NB with donor term | limma with blocking | Unpaired model |
| Multi-group with batch | NB with batch term | RUV then NB | Batch correction on residuals only |
| Very small n (2 per group) | NB with shrinkage, descriptive caution | limma-voom | Permutation with 6 permutations |
| Large n, many covariates | NB or limma-voom | Mixed model for repeated measures | Stepwise selection |
| Global shift suspected | Spike-in normalization | Reference genes | Median-of-ratios alone |
| Outlier sample present | Robust NB or exclusion with rule | Winsorization | Silent removal |

### Edge cases

**Low library complexity.** A sample with few distinct molecules has unstable counts.
Diagnostic: duplication rate above 50% and detected genes well below the cohort median.
Consequence: size factor unstable, dispersion inflated. Mitigation: exclude with a
pre-specified rule, or model library complexity as a covariate.

**Batch confounding.** Batch is collinear with condition. Diagnostic: cross-tabulation shows an
empty cell, and PC1 separates batch. Consequence: condition and batch effects are not
separable. Mitigation: none post hoc; redesign or restrict the claim.

**Sample swaps.** Diagnostic: sex check, genotype concordance, and expression-based
relatedness. Consequence: misattributed samples and inflated within-group variance. Mitigation:
fingerprint at accession, check before analysis.

**Zero inflation.** More zeros than the negative binomial predicts. Diagnostic: zero fraction
versus mean count, compared to a simulated NB. Consequence: dispersion overestimated and power
lost, or zeros treated as missing and imputed. Mitigation: check whether zeros are biological
or depth-driven; consider a hurdle model.

**Compositionality.** A global shift makes all genes appear to change. Diagnostic: the median
log ratio deviates from zero across most genes. Consequence: compressed or inflated fold
changes. Mitigation: spike-in normalization or a reference set; report the check.

**Probe dropout (arrays).** A probe fails to hybridize for a transcript that is present.
Diagnostic: a feature with a detection p-value above threshold across samples that should
express it. Consequence: false negatives. Mitigation: per-probe QC, exclusion of unreliable
probes.

**Missing annotation.** A gene has no annotated transcript in the release used, or maps to a
retired identifier. Diagnostic: unmapped reads in a region with known expression. Consequence:
missing features in the matrix. Mitigation: pin the annotation and quantify mapping loss.

**RIN confounding.** RNA quality differs between groups. Diagnostic: RIN differs by condition
and loads on a principal component. Consequence: degradation effects attributed to biology.
Mitigation: include RIN as a covariate, stratify, or exclude with a stated rule.

**Global activation.** An immune or stress response changes thousands of genes at once.
Diagnostic: the p-value histogram is skewed across the whole range, and size factors diverge
from library sizes. Consequence: normalization biased; fold changes compressed. Mitigation:
spike-in controls or reference-based normalization; interpret with composition in mind.

**Isoform ambiguity.** Reads are assigned to transcripts that share sequence, so
transcript-level fold changes are uncertain even when gene-level changes are clear. Diagnostic:
a large fraction of reads with ambiguous assignment, and transcript estimates that shift with
the annotation. Consequence: false isoform-switching claims. Mitigation: gene-level analysis
for most questions, or a probabilistic assignment method with uncertainty reported.

**Spike-in ratio mismatch.** The spike-in was added at a different ratio than assumed, or
degrades differently from endogenous RNA. Diagnostic: spike-in recovery varies systematically
across samples and does not track library size. Consequence: normalization to spike-ins is
biased. Mitigation: verify spike-in linearity across a dilution series before relying on it.

**Batch correction on residuals.** Batch is regressed out of the expression matrix before the
model, and the condition effect is then estimated from corrected values. Diagnostic: the
correction was fit without the condition in the model. Consequence: the correction removes part
of the condition effect, and power falls unpredictably. Mitigation: include batch in the model
rather than correcting the matrix.

**Unbalanced groups.** One condition has 8 samples and the other 40. Diagnostic: the design
table shows the imbalance, and the smaller group has unstable dispersion estimates.
Consequence: power is much lower than the total sample size suggests, and the smaller group
dominates the dispersion trend. Mitigation: report power by group, and consider a paired or
blocked design that recovers precision.

**Zeros that are biological.** A gene is absent in one condition because the cell type is
absent, not because of low depth. Diagnostic: zeros cluster by group and by cell-type markers.
Consequence: a hurdle model may fit better than a negative binomial, and treating zeros as
missing invents expression. Mitigation: model the zero process explicitly, or analyse at the
cell-composition level.

**Filters applied after viewing results.** A gene filter is chosen because it produces a
cleaner hit list. Diagnostic: the filter threshold is not in the analysis plan, and changing it
changes the headline. Consequence: selection bias that no correction accounts for. Mitigation:
pre-register the filter; report sensitivity to it.

**Repeated measures without a donor term.** Multiple biopsies per patient treated as
independent. Diagnostic: sample count exceeds patient count. Consequence: standard errors too
small. Mitigation: donor random effect or aggregation.

### What the procedure produces

The output is a results table with one row per tested gene: log2FC, standard error, shrunken
log2FC, p-value, q-value, mean expression, and the number of tests. The table should be
accompanied by a version stamp: software versions, annotation release, and the date of the
analysis, so that a later reanalysis can be compared rather than merely repeated. Alongside it belong the
diagnostic figures and a short analysis record naming the normalization, the design, the
filter, the correction, and the software versions. Without the record, the table is not
reproducible.

## 6. Worked example

A study compares pre-treatment colonic biopsies from 48 responders and 48 non-responders to an
anti-tumour necrosis factor therapy. Samples were processed in three batches. The count matrix
has 24,310 genes. The analysis plan specifies a negative binomial model with batch as a
covariate, a minimum count filter, independent filtering, and BH correction.

### 6.1 Size factors from a small table

Consider four genes across three samples.

| Gene | S1 | S2 | S3 | Geometric mean |
|---|---:|---:|---:|---:|
| G1 | 120 | 60 | 240 | 120.0 |
| G2 | 80 | 40 | 160 | 80.0 |
| G3 | 200 | 100 | 400 | 200.0 |
| G4 | 40 | 20 | 80 | 40.0 |

The ratios to the geometric mean are 1.0, 0.5, and 2.0 for every gene, so the median ratios are
1.0, 0.5, and 2.0. Sample S2 has half the depth of S1, and S3 has twice.

```text
size factors: S1 = 1.0, S2 = 0.5, S3 = 2.0
normalized counts for G1: 120/1.0 = 120; 60/0.5 = 120; 240/2.0 = 120
```

The normalization removes depth and leaves the gene constant. That is the method working as
intended. It fails when the ratios are not consistent across genes, which is what a global
shift produces.

### 6.2 One gene, end to end

A candidate gene has raw counts in five responders and five non-responders:

```text
responders:     210, 190, 240, 205, 195
non-responders: 120, 140, 110, 130, 125
```

The means are 208 and 125 counts. On the log2 scale, the means are 7.696 and 6.961, so:

```text
log2FC = 7.696 - 6.961 = 0.734   (1.66-fold)
```

The within-group standard deviations of log2 counts are 0.131 and 0.130. An exact permutation
test over all 252 assignments of five samples to each group gives 2 assignments with a
difference at least as extreme as the observed one:

```text
p = 2 / 252 = 0.0079
```

The permutation test is exact for this design but has limited resolution: with 252
permutations, the smallest attainable p-value is 1/252 = 0.0040. A p-value of 0.0079 is
therefore "two permutations out of 252," not a precise probability.

### 6.3 Multiple testing across the transcriptome

With 24,310 genes and a filter retaining 19,842, the Bonferroni threshold is:

```text
0.05 / 19,842 = 2.52 x 10^-6
```

The smallest observed p-value is 2.4 x 10^-8. Its q-value by BH is:

```text
q = 2.4 x 10^-8 x 19,842 / 1 = 4.8 x 10^-4
```

Suppose 812 genes pass at q < 0.05. The 812th smallest p-value is 1.9 x 10^-3, and the BH
threshold at that rank is:

```text
812 x 0.05 / 19,842 = 2.05 x 10^-3
```

The p-value is below the threshold, so the gene is included. Its q-value is:

```text
q = 1.9 x 10^-3 x 19,842 / 812 = 0.046
```

The reporting sentence: "812 of 19,842 genes were differentially expressed at q < 0.05
(Benjamini-Hochberg); the leading gene had log2FC = 2.1 (95% CI 1.6-2.6) and q = 4.8 x 10^-4."
Test, number of tests, correction, effect size, unit, and interval are all present.

### 6.4 Batch adjustment arithmetic

Before adjustment, the leading principal component separates the three processing batches and
explains 31% of variance. Condition explains 9%. After including batch in the model, the
condition coefficient for the candidate gene changes:

```text
unadjusted log2FC = 0.734
batch-adjusted log2FC = 0.612, SE = 0.118
95% CI = 0.612 +/- 1.96 x 0.118 = [0.381, 0.843]
```

The effect shrinks by 17% but remains clearly positive. That is the benign case: batch added
noise but did not create the effect.

### 6.5 Composition arithmetic

The candidate gene is expressed mainly in plasma cells. Deconvolution estimates that the plasma
cell fraction rises from 0.08 to 0.14 between non-responders and responders, a 1.75-fold
increase. If the per-cell expression is unchanged, the bulk fold change should be 1.75. The
observed bulk fold change is 1.66, which is close to the composition prediction.

```text
observed bulk FC = per-cell FC x abundance FC
1.66 = 0.95 x 1.75
```

A per-cell fold change of 0.95 is essentially no change. The gene is not regulated; the tissue
contains more plasma cells. Without deconvolution, the result would be reported as
"upregulation of a plasma-cell gene," which is technically true of the tissue and false of the
cell.

### 6.6 Diagnostic 1: the p-value histogram

The p-value histogram for all 19,842 genes is bimodal: a spike near 0 and a second bump near
0.5, with a dip near 1. A well-specified null produces a flat histogram with a spike at 1. A
bump in the middle is the signature of a misspecified model, often a covariate that is
correlated with the outcome or a normalization that does not fit.

The diagnosis in this case: RIN was not in the model, and RIN differs between batches. Adding
RIN as a covariate flattens the middle bump and changes the hit count from 812 to 640. The
analysis was not wrong in direction; it was overconfident in count.

### 6.7 Diagnostic 2: the size-factor plot

Size factors are plotted against library size. For most samples the relationship is linear, but
12 samples deviate by more than 20%. Those 12 are the samples with the highest immune
infiltration, where a large fraction of the transcriptome shifts and the median ratio is
pulled.

The consequence is a compressed fold change in those samples. The fix is not to remove them; it
is to use a normalization that is robust to composition, such as a spike-in or a reference gene
set, and to report the sensitivity of the result to the normalization choice.

### 6.8 Vary one assumption: composition is ignored

Suppose the analysis skips deconvolution and reports the bulk fold change as the biological
effect. The conclusion is "the plasma-cell gene is upregulated 1.66-fold in responders." A
follow-up study measures the gene in sorted plasma cells and finds no difference, and the
program concludes the biomarker failed to replicate.

If the analysis had adjusted for composition, the conclusion would be "plasma cell abundance is
higher in responders; the gene itself is not regulated." That is a different biomarker — a
compositional one — and it might still be useful, but it requires a different validation:
measuring cell abundance, not gene expression.

The single assumption — that bulk expression differences reflect per-cell regulation — changes
the biomarker class, the validation assay, and the clinical interpretation. It is worth stating
explicitly in every bulk study.

### 6.9 What the analysis record says

> 96 samples, 2 conditions, 3 batches. 24,310 genes quantified against a pinned annotation;
> 19,842 retained after a filter of at least 10 counts in at least 24 samples. Median-of-ratios
> size factors; negative binomial model with batch and RIN; dispersion trended and shrunken;
> independent filtering by mean expression; BH correction. 812 genes at q < 0.05. Effect sizes
> shrunken; 95% intervals reported. Deconvolution indicates the leading signal is
> compositional. Diagnostics: p-value histogram flat after RIN adjustment; size factors deviate
> in 12 high-infiltration samples.

That record is what a reviewer needs to decide whether to believe the 812.

## 7. Code, query, or calculation

The script implements size factors, a permutation test, Benjamini-Hochberg correction, and a
simple composition decomposition, using only the standard library.

### 7.1 Median-of-ratios size factors

```python
import math
import statistics

counts = {
    "G1": [120, 60, 240],
    "G2": [80, 40, 160],
    "G3": [200, 100, 400],
    "G4": [40, 20, 80],
}

def size_factors(counts):
    genes = list(counts)
    geo = {}
    for g in genes:
        vals = [c for c in counts[g] if c > 0]
        geo[g] = math.exp(sum(math.log(v) for v in vals) / len(vals))
    n_samples = len(next(iter(counts.values())))
    factors = []
    for j in range(n_samples):
        ratios = [counts[g][j] / geo[g] for g in genes if counts[g][j] > 0]
        factors.append(statistics.median(ratios))
    return factors

sf = size_factors(counts)
print("size factors:", [round(x, 3) for x in sf])
for g in counts:
    norm = [round(counts[g][j] / sf[j], 1) for j in range(3)]
    print(g, "raw", counts[g], "normalized", norm)
```

Expected output:

```text
size factors: [1.0, 0.5, 2.0]
G1 raw [120, 60, 240] normalized [120.0, 120.0, 120.0]
G2 raw [80, 40, 160] normalized [80.0, 80.0, 80.0]
G3 raw [200, 100, 400] normalized [200.0, 200.0, 200.0]
G4 raw [40, 20, 80] normalized [40.0, 40.0, 40.0]
```

The example is deliberately exact so that the arithmetic is visible. With real data the ratios
are not consistent, and the median is what makes the estimate robust to a minority of changing
genes.

### 7.2 Exact permutation test

```python
import itertools
import math
import statistics

resp = [210, 190, 240, 205, 195]
nonr = [120, 140, 110, 130, 125]

def log2(x):
    return math.log(x, 2)

a = [log2(x) for x in resp]
b = [log2(x) for x in nonr]
observed = statistics.mean(a) - statistics.mean(b)

pooled = a + b
extreme = 0
total = 0
for idx in itertools.combinations(range(10), 5):
    g1 = [pooled[i] for i in idx]
    g2 = [pooled[i] for i in range(10) if i not in idx]
    diff = statistics.mean(g1) - statistics.mean(g2)
    total += 1
    if abs(diff) >= abs(observed) - 1e-12:
        extreme += 1

print("observed log2FC:", round(observed, 3))
print("permutations:", total)
print("as or more extreme:", extreme)
print("p =", extreme, "/", total, "=", round(extreme / total, 4))
```

Expected output:

```text
observed log2FC: 0.734
permutations: 252
as or more extreme: 2
p = 2 / 252 = 0.0079
```

Note the resolution limit. With 252 permutations the smallest non-zero p-value is 0.004, so a
p-value below 0.004 is not attainable. If the decision threshold is 0.001, this design cannot
support it.

### 7.3 Benjamini-Hochberg with a realistic test count

```python
def bh(pvals, m=None):
    m = m if m is not None else len(pvals)
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    q = [0.0] * len(pvals)
    prev = 1.0
    for rank in range(len(pvals), 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvals[i] * m / rank)
        q[i] = min(prev, 1.0)
    return q

pvals = [2.4e-8, 3.1e-7, 1.2e-6, 8.4e-6, 4.1e-5, 1.9e-4]
q = bh(pvals, m=19842)
for p, qq in zip(pvals, q):
    print(f"p={p:.2e}  q={qq:.3e}")
print("significant at q<0.05:", sum(1 for x in q if x < 0.05))
```

Expected output:

```text
p=2.40e-08  q=4.762e-04
p=3.10e-07  q=3.076e-03
p=1.20e-06  q=7.937e-03
p=8.40e-06  q=4.167e-02
p=4.10e-05  q=1.627e-01
p=1.90e-04  q=6.283e-01
significant at q<0.05: 4
```

The fifth gene has p = 4.1 x 10^-5, which would be significant under a naive threshold of
0.05 and even under 10^-4, but is not significant after correction across 19,842 tests. This
is the arithmetic that changes conclusions.

### 7.4 Composition decomposition

```python
def decompose(bulk_fc, abundance_fc):
    return bulk_fc / abundance_fc

cases = [(1.66, 1.75), (1.66, 1.10), (1.66, 1.00)]
for bulk, abundance in cases:
    per_cell = decompose(bulk, abundance)
    print(f"bulk {bulk} / abundance {abundance} = per-cell {per_cell:.2f}")
```

Expected output:

```text
bulk 1.66 / abundance 1.75 = per-cell 0.95
bulk 1.66 / abundance 1.1 = per-cell 1.51
bulk 1.66 / abundance 1.0 = per-cell 1.66
```

The same bulk number supports three different biological conclusions depending on the
composition estimate. The estimate is not a detail; it is the interpretation.

### 7.5 Assumptions and limits of this code

- The size-factor function ignores genes with zero counts in a sample, which biases the median
  when zeros are common.
- The permutation test assumes exchangeability under the null, which fails if samples are
  paired or batched. Resample within blocks instead.
- The BH function assumes independence or positive dependence; it is not valid under arbitrary
  dependence.
- The composition decomposition assumes the abundance estimate is unbiased, which deconvolution
  is not.
- Nothing here fits a negative binomial model. A full model requires dispersion estimation,
  which is why production analyses use established packages rather than hand-rolled code.

## 8. Data, standards, and regulatory context

### 8.1 Reporting and data standards

- **MINSEQE** requires the biological system and variables, the reads, the processed data, the
  sample-to-data table, and the protocols. For a differential-expression study, the
  sample-to-data table is what permits batch and pairing to be recovered.
- **MIAME** applies to array data, with the same six elements.
- **GEO** and **ArrayExpress/BioStudies** accept submissions with these fields. Compliance is a
  precondition for reuse, not a guarantee of quality.
- **Sequence Ontology** and **GENCODE/Ensembl** define feature and consequence vocabularies.
- **MAGE-TAB** is the tabular format for array and sequencing metadata in ArrayExpress.

### 8.2 Analytical standards for clinical use

When a transcriptomic signature is used clinically, the analytical performance of the assay is
regulated, not the bioinformatics method.

- **CLIA** requires the laboratory to establish accuracy, precision, analytical sensitivity and
  specificity, reportable range, and reference intervals for the test.
- **CAP** accreditation adds proficiency testing and document control.
- **ISO 15189:2022** is the international laboratory quality standard, including
  point-of-care provisions.
- **IVDR 2017/746** applies in the EU to in vitro diagnostic devices, including
  transcriptomic classifiers placed on the market.
- **FDA** regulates in vitro diagnostics as devices. Laboratory-developed tests are currently
  under enforcement discretion following the 2025 vacatur of the 2024 LDT rule.

None of these specifies a normalization method, a dispersion model, or a multiple-testing
procedure. They require that whatever method is used be validated and documented. That is a
different obligation from statistical correctness, and a laboratory can satisfy one while
failing the other.

### 8.3 Reference data and consortia

- **GTEx v8**: 17,382 RNA-seq samples from 54 tissues and 948 donors, used as a reference for
  tissue expression and eQTLs.
- **TCGA**: 11,286 tumours across 33 cancer types, used for cancer expression reference.
- **SEQC/MAQC-III**: the FDA-coordinated cross-platform benchmark that established relative
  quantification as reproducible and absolute quantification as unreliable across platforms.
- **recount3**: uniformly processed public RNA-seq, useful for reanalysis with a single pipeline.
- **Expression Atlas**: curated expression across conditions and tissues.

### 8.4 What the standards do not constrain

- The choice of count model.
- The handling of outliers.
- The definition of the gene filter.
- The use of independent filtering.
- The choice between GSEA and ORA.
- The decision to adjust for composition.

These are scientific choices, and they are where published analyses diverge. A study that
reports them is reproducible; a study that reports only the software package is not.

## 9. Industry practice and case evidence

### 9.1 The benchmark evidence

The SEQC/MAQC-III consortium, coordinated by the FDA, ran reference RNA samples across sites
and platforms with ERCC spike-ins. Key documented findings:

- Relative expression is accurate and reproducible across sites and platforms; absolute
  quantification is not reliable for any platform tested, including quantitative PCR.
- Without filtering, both RNA-seq and microarrays produce unacceptably high false-positive
  rates for differential expression between identical samples processed at different sites.
- Applying fold-change and expression filters reduced the estimated false discovery rate to
  below 1.5% and raised inter-site concordance of differential-expression calls to about 95%.
- Inter-site variation was traced to library preparation rather than sequencing site, and
  surrogate variable methods removed more than 85% of inter-site false positives in the
  companion analysis.

The practical translation: a differential-expression result without a filter and without batch
handling is not a result. The benchmark quantified what practitioners already suspected.

### 9.2 What pharma does

Documented practice:

- Discovery transcriptomics uses bulk RNA-seq on cohorts of tens to low hundreds, with
  pre-specified contrasts and batch-balanced designs.
- Validation moves to a targeted assay (quantitative PCR, Nanostring-style counting, or a
  panel) on an independent cohort.
- Clinical-grade signatures are locked to a platform and a versioned algorithm, because a
  signature re-fit on a new platform is a new test.
- Reanalysis of public data (GTEx, TCGA, recount3) is standard for contextualizing internal
  findings.

Inference: the most common organizational gap is that discovery and validation use different
normalization and different feature filters, so the validation tests a different quantity than
the discovery. A signature should be locked as a fixed algorithm — normalization, features,
weights, threshold — before validation begins.

### 9.3 The reproducibility record

The Begley and Ellis audit (2012) and the Bayer analysis (2011) are about preclinical target
validation, not transcriptomics specifically, but the mechanism they describe applies:
unblinded selection, single-platform confirmation, and selective reporting. In
transcriptomics, the analogue is a hit list selected at p < 0.05 with no correction and no
replication.

A more specific body of evidence comes from reproducibility projects in which the same RNA
samples are analysed by multiple teams. The consistent finding is that analyst choices —
filtering, normalization, and model — account for substantial variation in results, sometimes
more than the biological effect under study. The implication is that the analysis plan must be
pre-specified, not discovered.

### 9.4 The composition lesson from single-cell

Single-cell atlases (topic 04) showed that many bulk expression differences between conditions
are explained by cell composition. The Human Cell Atlas and tissue-specific atlases provide
reference profiles that make deconvolution practical. The industry consequence is that a bulk
signature claiming a cell-intrinsic mechanism is now expected to show deconvolution or sorted-
population evidence. A program that does not is behind current practice.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| TPM analysed as counts | Column sums 10^6; non-integer values | Wrong model family | Invalid p-values | Check matrix type; use raw counts |
| No batch adjustment | PC1 separates processing groups | Technical variation dominates | False positives and negatives | Balance design; include batch |
| Batch fully confounded | Empty cell in batch x condition | Cases and controls processed apart | Effects inseparable | Redesign; restrict claim |
| Over-correction | Biological signal disappears after correction | Removing condition along with batch | False negatives | Include condition in the model; check |
| No filtering | 24,000 genes tested with many zeros | Noise inflates correction burden | Few discoveries; unstable estimates | Pre-specified count filter |
| Filtering on outcome | Genes removed because low in one group | Selection on the result | Inflated FDR | Filter on total count only |
| Uncorrected p-values | Hit list much larger than expected | Multiple testing ignored | Mostly false discoveries | BH with reported m |
| No effect sizes | Only p-values in the table | Significance without magnitude | Uninterpretable biology | Report log2FC and interval |
| Raw fold changes | Extreme values in low-count genes | Sampling noise | Overstated effects | Shrinkage |
| Composition ignored | Coordinated changes across cell-type markers | Cell abundance change | Mechanism claim false | Deconvolution |
| RIN confounding | RIN differs by group; loads on PC | Degradation effects | Misattributed biology | Include RIN; stratify |
| Sample swap | Sex-check mismatch | Labeling error | Misattributed samples | Fingerprint samples |
| P-value histogram U-shape | Dip near 1, bump in middle | Misspecified model | Unreliable inference | Add covariates; refit |
| Double normalization | Dispersion implausibly low | Normalized twice | p-values too small | Check dispersion plot |
| Pathway over-interpretation | Same pathway in every study | Database and background bias | Confirmatory illusion | Report set version; use GSEA |

Three red flags:

**If the p-value histogram is not flat near 1, stop.** The model is wrong, and no amount of
downstream interpretation repairs it.

**If batch and condition are collinear, stop.** You are measuring the batch.

**If the effect size is reported without a unit or an interval, stop.** You cannot tell whether
the result matters.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Most genes are unchanged | Median-of-ratios normalization | Global activation or arrest | Size factor vs library size plot |
| Counts follow a negative binomial | Standard DE model | Excess zeros, extreme outliers | Dispersion plot; goodness of fit |
| Samples are independent | Standard errors | Repeated donors, technical replicates | Donor term; ICC |
| Batch is measured | Adjustment | Missing batch metadata | Sample sheet audit |
| Batch is not collinear with condition | Identifiability | Design | Cross-tabulation |
| RNA quality is comparable | Expression comparability | RIN differences | RIN by group; 3' bias |
| Annotation is stable | Feature identity | Release change | Pin version; mapping audit |
| Effect sizes are exchangeable across genes | Shrinkage | Gene-specific biology | Check shrinkage calibration |
| Composition is stable or measured | Biological interpretation | Disease changes cell types | Deconvolution |
| The platform transports | Validation and clinical use | Different chemistry | Bridge study |

The composition assumption is the one most often violated and least often tested. A disease
that changes tissue architecture changes bulk expression by construction, and a bulk study
without composition data cannot distinguish regulation from infiltration.

## 12. Exercises

**Exercise 1 — Normalization identification (recall, 20 minutes).**
For each method — CPM, TPM, median-of-ratios, TMM — state what is held constant, one design in
which it fails, and the diagnostic that reveals the failure.
*Deliverable:* four-row table.
*Criteria:* all four correct; each diagnostic is a computation or plot, not a judgment.
*Time:* 20 minutes.

**Exercise 2 — Size factors by hand (calculation, 30 minutes).**
Given a 3-gene by 4-sample count table, compute median-of-ratios size factors and normalized
counts. State whether the normalization is appropriate if two of the three genes are known to
be strongly induced in one condition.
*Deliverable:* arithmetic and a one-paragraph judgment.
*Criteria:* size factors correct to two decimals; judgment follows from the method's assumption.
*Time:* 30 minutes.

**Exercise 3 — BH and interpretation (calculation, 35 minutes).**
A study tests 18,000 genes and reports 240 at q < 0.05. The 240th p-value is 1.4 x 10^-3.
Verify the BH threshold at that rank. Then state what happens to the count if the number of
tests is 36,000 because isoforms were counted separately.
*Deliverable:* arithmetic and a two-sentence consequence.
*Criteria:* threshold correct; consequence quantified, not asserted.
*Time:* 35 minutes.

**Exercise 4 — Critique a DE paper (critique, 60 minutes).**
Select a published bulk RNA-seq paper. Extract its normalization, design, filter, correction,
and effect-size reporting. Identify three choices that are unjustified or unreported and state
the consequence of each.
*Deliverable:* a one-page critique table.
*Criteria:* three specific findings; each tied to a consequence; no generic complaints.
*Time:* 60 minutes.

**Exercise 5 — Composition adjustment (application, 45 minutes).**
A bulk study reports a 1.9-fold increase in a gene expressed only in macrophages. Deconvolution
estimates macrophage fraction rising from 0.10 to 0.16. Compute the implied per-cell fold
change and state whether the claim "the gene is upregulated" is supported.
*Deliverable:* arithmetic and a two-sentence conclusion.
*Criteria:* decomposition correct; conclusion distinguishes composition from regulation.
*Time:* 45 minutes.

**Exercise 6 — Power and design (design, 60 minutes).**
Design a two-group bulk RNA-seq study to detect a log2FC of 0.7 with 80% power, given a
within-group SD of 1.0 log2 units, three batches, and a budget of 90 samples. Specify
allocation, blocking, and the analysis model.
*Deliverable:* design table and power calculation.
*Criteria:* power computed; batch balanced; model matches design; sample size justified.
*Time:* 60 minutes.

**Exercise 7 — Diagnostics (application, 40 minutes).**
You are shown a p-value histogram with a spike at 0 and a broad bump near 0.4, and a PCA where
PC1 separates sequencing runs. Write the diagnosis and the two changes you would make before
reporting any result.
*Deliverable:* a 300-word diagnostic memo.
*Criteria:* both diagnostics interpreted; changes are specific; no claim is made before the
fix.
*Time:* 40 minutes.

**Exercise 8 — Analysis plan (design, 45 minutes).**
Write a pre-specified analysis plan for a 120-sample bulk RNA-seq study with a paired design
and two batches. Include the contrast, normalization, filter, model, correction, effect-size
reporting, and the go/no-go rule for validation.
*Deliverable:* two-page plan.
*Criteria:* every choice named before data; go/no-go rule numeric; limitations stated.
*Time:* 45 minutes.

## 13. Self-check answers

**Answer 1 (normalization).** CPM holds total mapped reads constant; it fails when a few genes
dominate, and the diagnostic is a size-factor versus library-size divergence. TPM holds total
length-normalized transcript output constant; it fails for count models and under composition
shift, and the diagnostic is non-integer values with column sums of 10^6. Median-of-ratios
holds the median gene ratio constant; it fails under global activation, and the diagnostic is
the same size-factor plot. TMM holds a trimmed mean of log ratios constant; it fails in highly
asymmetric expression, and the diagnostic is a comparison with spike-in or reference
normalization.

**Answer 2 (size factors).** Compute the geometric mean per gene, the ratio per sample, and the
median across genes. If two of three genes are induced in one condition, the median ratio is
computed from a set that is mostly changing, so the size factor for that condition is biased
upward and the fold changes are compressed. The normalization is not appropriate; use a
spike-in or a larger reference set.

**Answer 3 (BH).** At rank 240 with m = 18,000, the threshold is 240 x 0.05 / 18,000 =
6.67 x 10^-4. The observed p-value of 1.4 x 10^-3 is above the threshold, so the 240th gene
would not be included; the reported count is inconsistent with the stated m and rank and should
be checked. If isoforms are counted separately and m doubles to 36,000, the threshold at rank
240 becomes 3.33 x 10^-4, and the number of significant features falls further. Counting
isoforms as independent tests inflates the burden without adding independent biological
information.

**Answer 4 (critique).** A good critique names the normalization (for example, "TPM, described
as normalized counts"), the design (for example, "batch not reported"), the filter (for example,
"genes with zero counts removed after viewing the groups"), and the correction (for example,
"Bonferroni, which is conservative given correlated genes"). Each is tied to a consequence:
invalid model, unassessable confounding, selection bias, and lost power. The critique should
also note whether effect sizes and intervals were reported.

**Answer 5 (composition).** Macrophage abundance rises from 0.10 to 0.16, a 1.6-fold increase.
Bulk fold change 1.9 divided by abundance fold change 1.6 gives a per-cell fold change of 1.19.
A 1.19-fold change is small and may not be statistically distinguishable from 1.0. The claim
"the gene is upregulated" is not supported as a cell-intrinsic statement; the supported claim is
"the gene's bulk signal rises, largely because macrophage content rises."

**Answer 6 (power and design).** For a two-sample comparison with SD 1.0 and log2FC 0.7:

```text
SE = 1.0 x sqrt(2/n); z = 0.7 / SE
n = 34 per arm gives SE = 0.2425, z = 2.89, power = 0.81
```

Allocate 34 per arm, 68 total, leaving 22 for replacement or a third group. Balance batches
within each arm (roughly 11-12 per batch per arm). Model: negative binomial with batch, or
limma-voom with batch; paired if the design is paired, which would reduce the required n.

**Answer 7 (diagnostics).** The histogram shape indicates a misspecified model with a
confounded covariate, and the PCA indicates a run effect. The two changes: include the
sequencing run as a batch term (or a surrogate variable if run is unknown for some samples),
and re-examine whether the condition is balanced across runs. If it is not, the analysis cannot
separate them, and the result must be restricted or the design repeated. No result should be
reported before these changes.

**Answer 8 (analysis plan).** Contrast: response versus non-response at baseline, paired by
donor where repeat biopsies exist. Normalization: median-of-ratios, with a spike-in check.
Filter: at least 10 counts in at least 20% of samples, fixed before analysis. Model: negative
binomial with batch and donor; dispersion trended and shrunken. Correction: independent
filtering then BH, with m reported. Effect sizes: shrunken log2FC with 95% intervals.
Go/no-go: validate if at least 20 genes replicate in the same direction at q < 0.05 and the
leading gene's interval excludes a 1.2-fold change. Limitations: bulk tissue, composition
unmeasured unless deconvolution is added.

## 14. Interview questions

**1. Why not use TPM for differential expression?**
Strong answer: TPM is a within-sample proportion after length normalization; count models need
counts and size factors, and TPM removes the information needed to estimate dispersion. TPM is
fine for visualization and within-sample comparison.
Weak answer: "TPM is normalized, so it is fine."

**2. What does median-of-ratios assume?**
Strong answer: most genes are unchanged, so the median ratio across genes estimates depth and
composition. Fails under global shifts; check size factors against library size.
Weak answer: "it normalizes for library size" with no assumption.

**3. How do you detect batch confounding?**
Strong answer: cross-tabulate batch by condition; if a cell is empty, the design is
non-identifiable. PCA and p-value histograms confirm.
Weak answer: "add batch to the model" without checking collinearity.

**4. What does a U-shaped p-value histogram mean?**
Strong answer: model misspecification or confounding, often a covariate correlated with the
outcome or a normalization that does not fit; it is not evidence of many true effects.
Weak answer: "many genes are significant."

**5. What is independent filtering and why does it help?**
Strong answer: filtering on a statistic independent of the p-value under the null (mean
expression) removes low-power tests and increases discoveries at the same FDR.
Weak answer: "removing genes with low counts" with no independence argument.

**6. Why shrink fold changes?**
Strong answer: low-count genes have noisy, biased-away-from-zero estimates; shrinkage improves
ranking and calibration and better predicts replication.
Weak answer: "to make the plot look better."

**7. When is limma-voom preferable to DESeq2?**
Strong answer: large designs with many covariates, or when speed matters; both are valid for
count data, and the choice should be justified rather than habitual.
Weak answer: names a favourite package.

**8. How would you tell regulation from composition?**
Strong answer: deconvolution, sorted populations, or single-cell data; the bulk fold change
decomposes into per-cell and abundance components.
Weak answer: "check if the gene is a marker" without quantifying.

**9. What do you report with a q-value?**
Strong answer: the test, the number of tests, the correction procedure, the effect size with a
unit and interval, and the threshold.
Weak answer: just the q-value.

**10. How do you handle an outlier sample?**
Strong answer: pre-specify an exclusion rule based on QC (library size, detected genes,
outlier distance), document it, and show the result with and without it.
Weak answer: remove it silently, or keep it because "all data are valid."

**11. What is the difference between GSEA and ORA?**
Strong answer: GSEA uses the full ranked list and is threshold-free; ORA uses a hit list and a
background, and its result depends on both.
Weak answer: treats them as interchangeable.

**12. What would make you distrust a 2,000-gene hit list?**
Strong answer: a p-value histogram skewed across all genes, size factors diverging from library
size, batch collinear with condition, or a hit list dominated by cell-type markers.
Weak answer: "2,000 is too many" without a mechanism.

## 15. Key terms

- **Benjamini-Hochberg** — FDR-controlling procedure on ranked p-values.
- **Batch** — group of samples processed together.
- **Blocking** — design that groups similar samples to reduce variance.
- **Composition** — proportion of cell types in a specimen.
- **Contrast** — linear combination of model coefficients.
- **CPM** — counts per million.
- **Deconvolution** — estimation of cell-type proportions from bulk data.
- **Design matrix** — numeric encoding of covariates.
- **Dispersion** — negative binomial variance parameter.
- **ERCC** — external RNA spike-in controls.
- **FDR** — expected proportion of false discoveries.
- **FPKM** — fragments per kilobase million.
- **GSEA** — gene set enrichment analysis on a ranked list.
- **Independent filtering** — removal of low-power tests using a null-independent statistic.
- **limma-voom** — linear modelling of log-CPM with precision weights.
- **log2FC** — log2 fold change, in log2 units.
- **Negative binomial** — count distribution with mean and dispersion.
- **ORA** — over-representation analysis.
- **Overdispersion** — variance above the Poisson mean.
- **Permutation test** — exact test by relabeling.
- **Precision weight** — weight from a mean-variance model.
- **q-value** — FDR-adjusted significance measure.
- **RIN** — RNA integrity number.
- **RUV** — remove unwanted variation using control genes.
- **Shrinkage** — pulling estimates toward a prior.
- **Size factor** — per-sample depth and composition scaling.
- **sva** — surrogate variable analysis.
- **TMM** — trimmed mean of M-values normalization.
- **TPM** — transcripts per million.
- **3' bias** — over-representation of 3' transcript ends.

## 16. Sources

Methods:

- Love, M. I., Huber, W., Anders, S., "Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2," *Genome Biology* 15:550, 2014. https://doi.org/10.1186/s13059-014-0550-8 (accessed 25 October 2026).
- Robinson, M. D., McCarthy, D. J., Smyth, G. K., "edgeR: a Bioconductor package for differential expression analysis of digital gene expression data," *Bioinformatics* 26:139-140, 2010. https://doi.org/10.1093/bioinformatics/btp616 (accessed 25 October 2026).
- Law, C. W. et al., "voom: precision weights unlock linear model analysis tools for RNA-seq read counts," *Genome Biology* 15:R29, 2014. https://doi.org/10.1186/gb-2014-15-2-r29 (accessed 25 October 2026).
- Benjamini, Y., Hochberg, Y., "Controlling the false discovery rate: a practical and powerful approach to multiple testing," *Journal of the Royal Statistical Society B* 57:289-300, 1995. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x (accessed 25 October 2026).

Benchmarks and evidence:

- SEQC/MAQC-III Consortium, "A comprehensive assessment of RNA-seq accuracy, reproducibility and information content by the Sequencing Quality Control Consortium," *Nature Biotechnology* 32:903-914, 2014. https://doi.org/10.1038/nbt.2957 (accessed 25 October 2026).
- Li, J. et al., "Detecting and correcting systematic variation in large-scale RNA sequencing data," *Nature Biotechnology* 32:888-895, 2014. https://doi.org/10.1038/nbt.3000 (accessed 25 October 2026).
- Begley, C. G., Ellis, L. M., "Raise standards for preclinical cancer research," *Nature* 483:531-533, 2012. https://doi.org/10.1038/483531a (accessed 25 October 2026).

Data resources:

- GTEx Portal. https://gtexportal.org/ (accessed 25 October 2026).
- recount3. https://rna.recount.bio/ (accessed 25 October 2026).
- Expression Atlas. https://www.ebi.ac.uk/gxa/ (accessed 25 October 2026).

Standards:

- FGED Society, MINSEQE. https://www.fged.org/projects/minseqe (accessed 25 October 2026).
- NCBI GEO, MIAME and MINSEQE. https://www.ncbi.nlm.nih.gov/geo/info/MIAME.html (accessed 25 October 2026).
- ISO 15189:2022. https://www.iso.org/standard/76677.html (accessed 25 October 2026).
- Regulation (EU) 2017/746. https://eur-lex.europa.eu/eli/reg/2017/746/oj (accessed 25 October 2026).

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_genomics-and-variants.md`](02_genomics-and-variants.md)
- Next topic: [`04_single-cell.md`](04_single-cell.md)
- Related topic: [`05_spatial-transcriptomics.md`](05_spatial-transcriptomics.md)
- Related topic: [`08_biomarker-discovery.md`](08_biomarker-discovery.md)
- Survey file: [`../../12_omics_precision_medicine.md`](../../12_omics_precision_medicine.md)
- Labs: [`../../labs/`](../../labs/)
