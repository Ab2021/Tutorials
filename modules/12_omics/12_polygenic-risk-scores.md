# Polygenic Risk Scores in Clinical Practice

> Module 12 — Omics and Precision Medicine
> Topic 12 of 16
> Estimated study time: 4 hours
> Prerequisites: topics 01, 02, and 08
> Version 2026.3

## 1. Why this topic exists

Most common disease is not caused by one variant. Coronary artery disease, type 2 diabetes,
breast cancer, and schizophrenia each involve hundreds to thousands of common variants, each
contributing a small increment of risk. A polygenic risk score (PRS) combines those increments
into one number: a weighted sum of risk allele dosages, with weights estimated from a genome-wide
association study (GWAS).

The arithmetic is simple. The interpretation is not. A PRS is meaningful only relative to a
reference distribution in a defined ancestry group, at a defined age, against a defined baseline
risk. Remove any of those four and the number becomes a percentile that patients read as a
probability, or an odds ratio that clinicians read as a diagnosis.

The commercial pull is real. Polygenic scores are sold directly to consumers, offered by
laboratories as laboratory developed tests (LDTs), and returned by research programs such as the
Electronic Medical Records and Genomics (eMERGE) network, which in 2024 reported selecting and
validating ten chronic disease scores for return to 25,000 diverse participants. The PGS Catalog,
the public repository for published scores, listed more than 7,000 scores across more than 800
traits when accessed on 26 October 2026. That volume creates a selection problem: a clinician
cannot tell which of the 7,000 is valid for a given patient, and most have never been evaluated
for clinical utility.

Three structural problems dominate the field.

**Ancestry portability.** Most discovery GWAS have been performed in European-ancestry
populations. A score trained in one population loses accuracy in another because linkage
disequilibrium (LD) patterns, allele frequencies, and environmental contexts differ. In a 2022
analysis of 245 scores trained in UK Biobank Europeans and applied to nine ancestry groups,
relative predictive performance was 93.8% in Poland, 64.7% in India, 48.6% in China, 25.2% in the
Caribbean, and 18.0% in Nigeria, measured as partial R-squared relative to the training
population. A 2019 meta-analysis of 26 studies found median performance in African-ancestry
samples at 42% of matched European samples. A score that is well calibrated for one group is
miscalibrated for another, and the miscalibration is in the direction of false reassurance.

**The gap between relative and absolute risk.** A score in the top 5% of the distribution may
carry a relative risk of 2 or 3 for a disease. If the baseline ten-year risk is 2%, the absolute
risk is still 4% to 6%, and the decision may not change. Reporting a relative risk without a
baseline is the most common error in consumer-facing PRS reports.

**The gap between discrimination and utility.** A model can discriminate well (a high area under
the receiver operating characteristic curve) and still add nothing to a decision, because the
decision depends on the absolute risk near a threshold. Net reclassification improvement (NRI) and
decision curve analysis (DCA) answer the utility question; discrimination metrics do not.

### 1.1 The failure pattern

1. A score is selected because it is the most recent publication, not because it was validated in
   the target population.
2. The scoring file is applied without harmonizing effect alleles, so some variants are counted on
   the wrong strand and the score is noise.
3. The raw score is converted to a percentile using a reference panel from a different ancestry.
4. The percentile is reported to a patient as a probability.
5. The relative risk is reported without a baseline absolute risk.
6. The score is added to a clinical model without checking calibration or net benefit.
7. The model is deployed, and no one measures whether it changed decisions or outcomes.

Each step is a default in at least one commercial or research pipeline.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Clinician | A number with no decision attached | Depends on the report's calibration and baseline |
| Patient | A percentile misread as a probability | Cannot see the reference population |
| Laboratory | A validated assay with an unvalidated interpretation | Does not control the report template |
| Payer | Spend on testing with no demonstrated utility | Depends on evidence the vendor did not produce |
| Researcher | A score that fails to replicate | Inherited a discovery cohort with the wrong ancestry |
| Regulator | No basis to assess a claim | No standard for PRS clinical validity |

### 1.3 The industry question

> Does this score change a decision for this patient, at this age, in this ancestry group, given
> the risk we can already estimate without it?

The question has four qualifiers, and a PRS program must answer all four. A score that changes a
decision in a 60-year-old smoker may not change one in a 35-year-old non-smoker, because the
baseline risks differ. A score validated in Europeans may not be interpretable in a patient of
West African ancestry without recalibration. A score with an odds ratio of 1.4 per standard
deviation may still fail a decision curve at the threshold that matters.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** a polygenic score from a scoring file and a genotype matrix, including effect
   allele harmonization and missing-variant handling.
2. **Standardize** a raw score against an ancestry-matched reference distribution and **explain**
   what the resulting z-score and percentile do and do not mean.
3. **Convert** an odds ratio per standard deviation into an absolute risk at a stated baseline,
   using the odds-based method, and show why the naive multiplicative formula diverges.
4. **Compare** clumping and thresholding, LDpred2, PRS-CS, and multi-ancestry methods on their
   assumptions, data requirements, and failure modes.
5. **Evaluate** clinical utility with net reclassification improvement and decision curve
   analysis, and state why discrimination alone is insufficient.
6. **Diagnose** ancestry miscalibration, allele harmonization errors, and winner's curse inflation
   from score distributions and cohort-level checks.
7. **Write** a PRS report that presents absolute risk, a continuous percentile, ancestry
   limitations, and the evidence grade, following the Polygenic Risk Score Reporting Standards.
8. **Decide** when a PRS should not be reported at all.

## 3. The core mental model

A polygenic score is a weighted sum whose meaning depends entirely on the population and the
baseline against which it is compared. The model below separates four layers that are often
collapsed into one number.

```text
  LAYER 1: DISCOVERY
    GWAS summary statistics (beta, se, p, effect allele, allele frequency)
         |
         +--> ancestry of cohort, sample size, phenotype definition
         |
         v
  LAYER 2: SCORE CONSTRUCTION
    variant selection + LD adjustment + shrinkage
         |
         +--> scoring file: rsID, effect allele, weight
         |
         v
  LAYER 3: APPLICATION
    target genotypes --> harmonize alleles --> sum beta x dosage
         |
         +--> raw score (arbitrary units)
         |
         v
  LAYER 4: INTERPRETATION
    standardize against reference --> z-score --> percentile
         |
         +--> convert to absolute risk with a baseline
         |
         v
    DECISION: does the absolute risk cross a threshold, and by how much?
```

### 3.1 A PRS is not a probability

The raw score is in units of log odds weighted by allele count. It has no intrinsic scale. Two
scoring files for the same trait will produce different raw values for the same person. Only the
position of the score within a reference distribution is interpretable, and only if the reference
distribution was measured in a comparable population with a comparable genotyping array and a
comparable variant set.

The z-score is:

```text
  z = (PRS_raw - mean_reference) / sd_reference
```

The percentile is the normal cumulative distribution function of z. Both are statements about
rank, not risk. A person at the 95th percentile is higher than 95% of the reference population;
their probability of disease depends on the baseline.

### 3.2 The three populations in every PRS

Every score involves three populations, and confusing them is the source of most errors:

| Population | Role | What goes wrong if confused |
|---|---|---|
| Discovery population | Where the variant weights were estimated | Weights reflect that population's LD and environment |
| Reference population | Where the mean and standard deviation were measured | Percentile is computed against the wrong distribution |
| Target population | The patient in front of you | Calibration and transportability are untested |

A score can be valid in the discovery population, standardized against a reference from a third
population, and applied to a fourth. Nothing in the arithmetic detects that chain.

### 3.3 Why ancestry changes the score

Three mechanisms operate at once.

1. **LD differences.** A score uses tag variants that correlate with causal variants. If the
   correlation differs in the target population, the tag no longer proxies the causal variant.
   African populations have older genomes, smaller LD blocks, and more diverse haplotypes, so
   European tag variants transfer least well there.
2. **Allele frequency differences.** A variant common in Europe may be rare or absent elsewhere,
   so its contribution is missing or disproportionately weighted.
3. **Gene-environment and phenotype differences.** The measured phenotype may have different
   ascertainment, treatment, or environmental context across populations.

The practical consequence is not that PRS should never be used across ancestries. It is that a
score must be recalibrated, and its performance reported, for each group in which it is used.

### 3.4 Relative risk, absolute risk, and the baseline

The relationship between a z-score and absolute risk requires a baseline. Using the odds scale:

```text
  odds_baseline   = p_baseline / (1 - p_baseline)
  odds_individual = odds_baseline x OR_per_SD ^ z
  p_individual    = odds_individual / (1 + odds_individual)
```

At low baseline risk this is close to the simpler multiplicative form:

```text
  p_individual = 1 - (1 - p_baseline) ^ RR
```

The two diverge as risk rises. At a baseline of 3.2% and a z-score of 1.75 with an odds ratio of
1.45 per standard deviation, the odds method gives 5.96% and the multiplicative form gives 6.04%,
a difference of 0.08 percentage points. At a baseline of 8.0%, the same z-score gives 14.28% and
14.77%, a difference of 0.49 percentage points. The divergence matters when a decision threshold
sits between the two estimates.

### 3.5 Discrimination, calibration, and utility are three different questions

| Question | Metric | What a good value means | What it does not mean |
|---|---|---|---|
| Can the score rank people? | Area under the curve, or R-squared | Higher-risk people tend to have events first | The ranking is useful at a decision threshold |
| Are the predicted risks correct? | Calibration slope and intercept, observed versus expected | Predicted 5% means about 5% observed | The model changes any decision |
| Does the score improve decisions? | Net reclassification improvement, decision curve net benefit | More benefit than the alternatives across thresholds | The improvement is large enough to justify cost |

A score can be well calibrated and useless, or poorly calibrated and useful after recalibration.
The three questions must be answered separately.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Polygenic risk score (PRS) | Weighted sum of risk allele dosages across many variants | Called a genetic test result, as if it were diagnostic | Arbitrary units until standardized |
| Polygenic score (PGS) | Synonym for PRS, used by the PGS Catalog | Used for quantitative traits without a risk framing | Arbitrary units |
| Genome-wide association study (GWAS) | Study testing associations between variants and a trait | Treated as a causal inference | Study design |
| Summary statistics | Per-variant effect size, standard error, p-value, allele frequency | Assumed to be on a common build and strand | Tabular file |
| Effect allele | Allele to which the reported effect size refers | Assumed to be the minor allele; it need not be | Nucleotide code |
| Other allele | The alternate allele at the locus | Dropped during harmonization, making strand flips invisible | Nucleotide code |
| Dosage | Number of effect alleles carried, 0, 1, or 2 | Confused with imputation dosage, a continuous value | Allele count, or 0 to 2 |
| Linkage disequilibrium (LD) | Non-random association of alleles at different loci | Treated as a fixed property of a variant | r-squared or D-prime, 0 to 1 |
| Clumping and thresholding (C+T) | Method that prunes correlated variants and applies a p-value cutoff | Treated as equivalent to Bayesian methods | Method |
| LDpred2 | Bayesian method that shrinks weights using an LD matrix | Applied with a mismatched LD panel | Software |
| PRS-CS | Bayesian continuous shrinkage method using external LD | Assumed to handle ancestry automatically | Software |
| PRS-CSx | Multi-ancestry extension of PRS-CS | Assumed to require individual-level data | Software |
| Winner's curse | Inflation of effect sizes selected for statistical significance | Ignored when using published weights | Statistical bias |
| Shrinkage | Reduction of effect estimates toward zero | Confused with standardization | Statistical operation |
| Standardization | Expressing a raw score relative to a reference mean and SD | Confused with recalibration | z-score |
| z-score | Number of reference standard deviations above the mean | Reported to patients as a risk | Standard deviations |
| Percentile | Rank of a score within a reference distribution | Read as a probability of disease | Percent, 0 to 100 |
| Reference population | Group whose score distribution defines the z-score | Assumed to match the patient | Population definition |
| Calibration | Agreement between predicted and observed risk | Confused with discrimination | Slope, intercept, observed-to-expected ratio |
| Discrimination | Ability to rank cases above non-cases | Confused with clinical utility | Area under the curve, 0.5 to 1.0 |
| Net reclassification improvement (NRI) | Change in the proportion correctly reclassified across risk categories | Reported without its confidence interval or its known biases | Proportion |
| Decision curve analysis (DCA) | Net benefit across decision thresholds | Reported without the treat-all and treat-none references | Net benefit |
| Net benefit | True positives minus false positives weighted by the threshold odds | Confused with monetary benefit | Proportion |
| Number needed to treat (NNT) | Patients treated to prevent one event | Computed from relative risk alone | Patients |
| Absolute risk | Probability of an event in a defined period | Omitted from consumer reports | Percent over a stated period |
| Relative risk | Ratio of risks between groups | Reported without a baseline | Ratio |
| Odds ratio (OR) | Ratio of odds between groups | Interpreted as a risk ratio when the outcome is common | Ratio |
| Liability scale | Underlying continuous liability assumed by a threshold model | Ignored when comparing R-squared across traits | Variance explained |
| Ancestry | Genetic similarity to a reference population | Treated as a social or geographic category | Principal components, self-report |
| Portability | Performance of a score in a population other than the discovery population | Assumed to be uniform within a continental group | Relative performance |
| PGS Catalog | Public repository of polygenic scores and scoring files | Treated as a validation authority | Database |
| PRS-RS | Polygenic Risk Score Reporting Standards, 22 items | Confused with a clinical guideline | Reporting standard |
| LDT | Laboratory developed test | Assumed to be unregulated | Regulatory category |

## 5. The mechanics, step by step

The procedure builds a score from summary statistics, applies it to a target cohort, and converts
the result into a decision-relevant absolute risk. Each step states its input, output, and reason.

| Step | Question answered | Artifact | Failure if skipped |
|---:|---|---|---|
| 1 | What trait, population, and decision? | Analysis plan | Score with no use case |
| 2 | Which discovery study? | Selected summary statistics | Weights from an incompatible phenotype |
| 3 | Are the variants comparable? | Harmonized summary statistics | Strand and build errors |
| 4 | Which weighting method? | Scoring file | Wrong architecture assumption |
| 5 | Which LD reference? | LD matrices | Shrinkage against the wrong population |
| 6 | How to handle selection bias? | Shrunk weights | Inflated scores |
| 7 | How to compute the score? | Raw scores | Missing and mismatched variants |
| 8 | Against whom to standardize? | z-scores | Percentile against the wrong group |
| 9 | Does it discriminate and calibrate? | Validation metrics | Deployment of an untested score |
| 10 | Does it improve decisions? | NRI and net benefit | Utility claimed from discrimination |
| 11 | How to report it? | Report template | Percentile read as probability |
| 12 | How to monitor it? | Monitoring plan | Silent drift after deployment |

### Step 1. Define the trait, the population, and the decision

**Input.** A clinical question, such as whether to start a statin in a 45-year-old woman with a
borderline ten-year risk.

**Output.** A written analysis plan naming the trait and its definition, the target population,
the decision threshold, and the comparator model.

**Rationale.** The same score can be evaluated against different thresholds and produce different
conclusions. Fixing the decision first determines which metric matters. If the decision is
"statin or not at a 7.5% ten-year threshold", then calibration near 7.5% and net benefit at that
threshold matter more than the area under the curve.

**If skipped.** The evaluation reports discrimination because it is easy to compute, and nobody
can say whether the score helps.

### Step 2. Select the discovery study

**Input.** Candidate GWAS publications and summary statistics.

**Output.** One primary discovery dataset, plus a sensitivity dataset if available.

**Rationale.** The discovery study fixes the ancestry, the phenotype definition, the age range,
and the variant set. A score for "coronary artery disease" built from a study that included
prevalent cases has different properties from one built from incident cases. Sample size
determines how much winner's curse correction is needed.

Study selection has more than two outcomes, and the choice constrains everything downstream.

| Situation | Evidence available | Action |
|---|---|---|
| Large GWAS in the target ancestry, matching phenotype | Summary statistics and replication | Use directly; check build and allele frequency |
| Large GWAS in a different ancestry, validated in target | Published cross-ancestry validation with calibration | Use with recalibration; report the performance loss |
| Large GWAS in a different ancestry, no validation | No target-population performance data | Do not report a percentile; consider a multi-ancestry score |
| Small GWAS, target ancestry | Sample size below a few thousand | Expect heavy winner's curse; prefer a meta-analysis |
| Meta-analysis with overlapping cohorts | Possible sample overlap | Check for duplicate cohorts before treating it as independent |
| Case-control with extreme ascertainment | Odds ratios inflated relative to population risk | Convert to liability scale before comparing with population risk |
| Phenotype defined by treatment or code | Cases may be misclassified or prevalent | Check the phenotype definition against the intended use |

**If skipped.** Weights from a study with a different phenotype definition are applied to a
target where the phenotype means something else, and the miscalibration is invisible.

### Step 3. Harmonize summary statistics

**Input.** Summary statistics with rsID, chromosome, position, effect allele, other allele,
effect size, standard error, p-value, and allele frequency.

**Output.** A table with a consistent genome build, consistent strand, and effect alleles aligned
to the target genotype data.

**Rationale.** Harmonization is where most silent errors occur. Four cases must be handled
explicitly: the effect allele matches the genotype's effect allele; the effect allele is the
complement; the alleles are swapped; or the alleles do not match at all and the variant must be
dropped. A pipeline that matches on rsID alone without checking alleles will count some variants
backwards.

The decision has more than two outcomes.

| Genotype alleles | Summary statistic alleles | Action |
|---|---|---|
| Same pair, effect allele present | Effect A, other B | Count A |
| Complement pair | Effect A, other B, genotype T/C | Flip strand, then count |
| Swapped | Genotype lists B/A | Count A after swap |
| Ambiguous A/T or C/G | Either | Drop unless allele frequency resolves it |
| No match | Different variant or multiallelic | Drop and log |
| Duplicate rsID | Multiple rows | Resolve by position and allele, or drop |

**If skipped.** A strand flip at a common variant biases the score in a direction that depends on
the genotype distribution, and the error survives every downstream check that does not compare
score distributions.

### Step 4. Choose the weighting method

**Input.** Harmonized summary statistics; trait architecture assumptions; compute budget.

**Output.** A scoring file of variant, effect allele, and weight.

**Rationale.** Methods differ in how they handle correlated variants. Clumping and thresholding
discards variants in LD with a lead variant, which is simple and loses signal. Bayesian methods
model all variants jointly with a prior on effect sizes, which recovers more signal but requires
an LD matrix and more compute. The right choice depends on sample size, heritability architecture,
and available reference panels.

| Method | Data required | Assumption | Strength | Weakness |
|---|---|---|---|---|
| C+T | Summary statistics | One lead variant per LD block captures the signal | Fast, interpretable | Discards correlated signal; sensitive to thresholds |
| LDpred2 | Summary statistics plus LD matrix | Spike-and-slab prior on effects | Good performance, handles LD | Requires matched LD panel; compute |
| PRS-CS | Summary statistics plus external LD | Continuous shrinkage prior | Robust across architectures | Assumes LD panel matches target ancestry |
| PRS-CSx | Multi-ancestry summary statistics | Shared and population-specific effects | Improves cross-ancestry performance | Needs multiple ancestry GWAS |
| Lassosum | Summary statistics plus LD | Penalized regression | Fast, stable | Less flexible prior |

### Step 5. Match the LD reference panel

**Input.** Target ancestry; candidate LD panels such as 1000 Genomes or UK Biobank.

**Output.** An LD matrix per chromosome, matched to the target population and the variant set.

**Rationale.** LD is a property of a population, not of a variant. Using a European LD panel to
shrink weights for an African-ancestry target applies the wrong correlation structure. The
mismatch does not produce an error message; it produces a score with reduced accuracy.

**If skipped.** The score is technically valid and empirically worse, and the cause is invisible
without a comparison against an ancestry-matched panel.

### Step 6. Correct for winner's curse and overfitting

**Input.** Selected variants and their effect sizes.

**Output.** Shrunk weights.

**Rationale.** Variants selected because they reached a significance threshold have effect sizes
that are inflated on average, because selection favors large estimates. The inflation is larger
for small studies and for variants near the threshold. Bayesian shrinkage and cross-validation
both reduce it. A simple illustration: an observed effect of 0.28 log-odds per allele with a
standard error of 0.050 has a z statistic of 5.60, and a crude shrinkage factor of 1 minus 1 over
z squared equals 0.968, giving a shrunk effect of 0.271.

**If skipped.** The score overweights the variants that happened to look strongest, and its
out-of-sample performance is worse than its discovery performance suggests.

### Step 7. Compute raw scores in the target cohort

**Input.** Scoring file; target genotypes or imputed dosages.

**Output.** One raw score per person, plus a count of variants used and variants missing.

**Rationale.** The score is the sum over matched variants of weight times dosage. Missing variants
are usually imputed or the variant is dropped. Dropping variants changes the score's scale,
especially when the missingness is related to ancestry or array design, so the count must be
reported and the score standardized only after the same variants are used for the reference.

The handling of a missing variant is a decision with more than two outcomes.

| Situation | Effect on the score | Action |
|---|---|---|
| Genotype absent, variant on the array | Zero contribution, scale shifted | Impute; otherwise drop and record |
| Variant absent from the array but tagged by a proxy | Partial signal recoverable | Use an \(r^2\)-weighted proxy if \(r^2\) is above 0.8 |
| Variant monomorphic in the target | Contributes nothing | Drop; note the portability loss |
| Dosage below the imputation quality threshold | Noisy contribution | Drop if imputation \(r^2\) is below 0.3 |
| Alleles incompatible after harmonization | Cannot be scored | Drop and log the reason |
| Duplicate variant in the scoring file | Double counting | Resolve by position; take the larger \(r^2\) match |

**If skipped.** Two patients scored with different variant sets are compared on different scales.

### Step 8. Standardize against a matched reference

**Input.** Raw scores; a reference distribution measured in a comparable population.

**Output.** z-scores and percentiles.

**Rationale.** The reference must match the target in ancestry, genotyping platform, and variant
set. eMERGE, for example, used ancestry-calibrated z-scores derived from All of Us participants
rather than a single European reference.

The choice of reference has more than two outcomes.

| Reference situation | Evidence | Action |
|---|---|---|
| Matched ancestry and platform | Reference from the same group and array | Standardize directly |
| Matched ancestry, different platform | Same group, different variant coverage | Recompute reference on the common variant set |
| Different ancestry, score validated there | Published validation in the target group | Recalibrate, then standardize |
| Different ancestry, no validation | No performance data | Do not report a percentile |
| Admixed individual | Continuous ancestry, not a category | Use a model that accounts for ancestry proportions |

**If skipped.** A European-referenced percentile applied to a West African patient places them at
a rank that does not correspond to their risk.

### Step 9. Validate discrimination and calibration

**Input.** Scores and outcomes in an independent target cohort.

**Output.** Area under the curve or R-squared, calibration slope and intercept, and observed versus
expected risk by decile.

**Rationale.** Discrimination and calibration answer different questions. A score can rank well
and be systematically wrong. Calibration must be assessed in the group where the score will be
used, because that is where the baseline differs.

A validation report needs four numbers, not one.

| Metric | How it is computed | Passing pattern | Failing pattern |
|---|---|---|---|
| Area under the curve | Probability a case ranks above a non-case | Above the comparator, with a confidence interval | Overlapping confidence intervals |
| Calibration slope | Regression of observed outcome on predicted log-odds | Near 1.0 | Below 1.0 indicates overfitting; above 1.0 indicates underfitting |
| Calibration intercept | Intercept of the same regression | Near 0.0 | Non-zero indicates systematic over- or under-prediction |
| Observed-to-expected ratio | Observed events divided by expected events, by decile | Near 1.0 across deciles | A gradient across deciles indicates miscalibration |

**If skipped.** A score that is miscalibrated by a factor of two is deployed, and every absolute
risk it produces is wrong by that factor.

### Step 10. Evaluate clinical utility

**Input.** Predicted risks from the comparator model and from the model with the score.

**Output.** Net reclassification improvement, decision curve net benefit, and the range of
thresholds where the score adds value.

**Rationale.** Utility depends on the threshold. A score can add net benefit at a 5% threshold and
subtract it at a 20% threshold, because the harm of a false positive grows as the threshold rises.
Reporting a single number hides this.

Reading a decision curve requires three comparisons, not one.

| Comparison | Interpretation | What it rules out |
|---|---|---|
| Model curve above treat-all | The model beats treating everyone at that threshold | A model that is no better than universal treatment |
| Model curve above treat-none | The model beats treating nobody | A model that harms more than it helps |
| Model curve above the comparator model | The score adds value beyond existing predictors | A score that only reproduces known risk factors |
| Curves cross within the relevant threshold range | The score helps only in part of the range | A claim of uniform benefit |
| Confidence band includes zero net benefit | The increment is not established | A claim of utility from a point estimate |

**If skipped.** The program claims utility from an improvement in area under the curve that does
not translate into better decisions.

### Step 11. Report absolute risk, not just a percentile

**Input.** Calibrated absolute risk, percentile, ancestry, reference population, and evidence
grade.

**Output.** A report that a clinician and a patient can act on.

**Rationale.** Studies of report comprehension find that absolute risk is the most acceptable
format, that percentiles are often read as probabilities, and that odds ratios are poorly
understood. A report that leads with "92nd percentile" invites the reading "92% chance of
disease". A report that leads with "6.0% ten-year risk, compared with 3.2% for an average woman
of the same age" invites the correct comparison.

**If skipped.** The patient receives a number that is technically correct and systematically
misinterpreted.

### Step 12. Monitor performance after deployment

**Input.** Post-deployment scores, outcomes, and ancestry composition.

**Output.** A monitoring report with calibration drift, subgroup performance, and a trigger for
recalibration.

**Rationale.** A score's calibration depends on the population and on the clinical context. As a
health system's population changes, or as treatment improves and baseline risks fall, calibration
drifts. Monitoring is the only way to detect it.

**If skipped.** The score silently becomes miscalibrated, and the program cannot tell whether a
change in outcomes is due to the score or to secular trends.

### 5.13 Four tests every score should pass before reporting

| Test | Procedure | Pass criterion | What it catches |
|---|---|---|---|
| Harmonization audit | Compare effect allele frequencies with the discovery study for every variant | At least 95% of variants within 0.05, no systematic direction | Strand flips and build errors |
| Reference match | Standardize a held-out cohort against the reference and compute the mean z-score | Mean between -0.2 and 0.2, standard deviation between 0.8 and 1.2 | Wrong reference population |
| Calibration-in-the-large | Compare observed and expected events in the target cohort | Observed-to-expected ratio between 0.8 and 1.25 | Baseline mismatch and miscalibration |
| Utility at the threshold | Decision curve at the guideline threshold with a confidence interval | Net benefit above both default strategies and above the comparator model | A score that discriminates but does not help |

A score that passes the first two and fails the last two can still be reported as a research
result. It cannot be reported as a clinical result with an action attached.

### 5.14 What a report must contain

1. The trait, the outcome definition, and the time horizon.
2. The score identifier and version, ideally a PGS Catalog identifier.
3. The discovery population and its ancestry.
4. The reference population used for the percentile.
5. The absolute risk, with the baseline and the method used to combine them.
6. The percentile, clearly labeled as a rank.
7. The validation population and the performance in the patient's ancestry group.
8. The evidence grade and the guideline, if any, that supports the intended use.
9. The limits: what the score does not measure, and what it should not be used for.

An item missing from this list means the report cannot be interpreted by a clinician who has not
read the underlying paper.

### 5.15 What the procedure does not decide

The procedure produces a calibrated risk and a utility estimate. It does not decide whether the
health system should offer the test, who pays, how to consent, or how to communicate a result that
may worry a patient for decades. Those are program and policy decisions. The technical obligation
is to make the score's provenance and limits visible: discovery population, reference population,
variant count, method, calibration group, and the threshold at which utility was demonstrated.

## 6. Worked example

### 6.1 The case

A 45-year-old woman of European ancestry has a ten-year atherosclerotic cardiovascular disease
(ASCVD) risk of 3.2% from the Pooled Cohort Equations. She does not smoke, has a systolic blood
pressure of 124 mmHg, total cholesterol of 196 mg/dL, high-density lipoprotein cholesterol of
58 mg/dL, and no diabetes. The 2018 American College of Cardiology and American Heart Association
guideline calls 3.2% a borderline risk, where a risk enhancer can justify a statin.

She purchases a coronary artery disease PRS. The report says "96th percentile". The question is
whether that changes the decision.

### 6.2 Step 1: convert the percentile to a z-score

A percentile of 96 corresponds to a z-score of 1.751, from the inverse normal distribution. Check:
the normal cumulative distribution at 1.751 is 0.960.

### 6.3 Step 2: find the odds ratio per standard deviation

Assume the score has a published odds ratio of 1.45 per standard deviation for coronary artery
disease, estimated in a European-ancestry cohort. This is an assumption for the example, chosen
to be plausible for a modern coronary score; the exact value depends on the score and the
outcome definition.

### 6.4 Step 3: compute the odds ratio at this z-score

```text
  OR(z) = OR_per_SD ^ z
        = 1.45 ^ 1.751
        = exp(1.751 x ln 1.45)
        = exp(1.751 x 0.37156)
        = exp(0.65060)
        = 1.9168
```

### 6.5 Step 4: convert to absolute risk

```text
  baseline odds  = 0.032 / (1 - 0.032)
                 = 0.032 / 0.968
                 = 0.033058
  individual odds = 0.033058 x 1.9168
                  = 0.063363
  risk            = 0.063363 / (1 + 0.063363)
                  = 0.059585
                  = 5.96%
```

The ten-year absolute risk rises from 3.2% to 6.0%, an absolute increase of 2.76 percentage
points. The relative risk is 5.96 / 3.2 = 1.86.

### 6.6 Step 5: check against the naive formula

```text
  naive risk = 1 - (1 - 0.032) ^ 1.9168
             = 1 - 0.968 ^ 1.9168
             = 1 - exp(1.9168 x ln 0.968)
             = 1 - exp(1.9168 x -0.032523)
             = 1 - exp(-0.062340)
             = 1 - 0.939560
             = 0.060440
             = 6.04%
```

The two methods differ by 0.08 percentage points at this baseline. Recompute at a baseline of
8.0%, for a patient with the same z-score:

```text
  odds method  : odds = 0.08/0.92 = 0.086957; x 1.9168 = 0.166680
                 risk = 0.166680 / 1.166680 = 14.29%
  naive method : 1 - 0.92 ^ 1.9168 = 1 - exp(-0.15981) = 14.77%
  difference   : 0.48 percentage points
```

The divergence grows with baseline risk. Neither method is a substitute for a cohort-specific
calibration model; both assume the odds ratio is constant across the baseline range, which is an
approximation.

### 6.7 Step 6: does the decision change

At a 7.5% ten-year threshold, 5.96% does not cross the threshold. At a 5% threshold, used in some
guidelines for intermediate risk, it does. The answer therefore depends on which guideline the
clinician follows, and the PRS report should state the threshold it assumes rather than imply
that 96th percentile means "treat".

The 2018 guideline allows a risk enhancer to support a statin decision in the 5% to 7.5% range.
Whether a PRS counts as a risk enhancer is a guideline-interpretation question, not an arithmetic
one. A defensible position is that a validated, ancestry-matched score can be presented as one
piece of evidence in a shared decision, not as an automatic trigger.

### 6.8 Diagnostic 1: the score distribution in the target cohort

The first diagnostic that would reveal a broken analysis is the distribution of raw scores and
z-scores in the target cohort.

A working pipeline produces a z-score distribution with mean near 0 and standard deviation near 1
when the reference matches the cohort. Three patterns indicate a defect:

- **Mean z well away from 0.** A cohort mean z of +0.32 with a standard deviation of 1.33, as in
  the code output below, means the reference distribution does not match the cohort. The
  percentile for every patient is shifted. The cause may be a different array, a different
  variant set, or a different ancestry composition.
- **Standard deviation far from 1.** A standard deviation of 1.33 means the cohort is more
  variable than the reference. If the cohort is a high-risk clinic, this can be real; if it is a
  general population sample, it indicates a scoring or harmonization error.
- **A bimodal distribution.** Two modes usually indicate that two genotyping platforms or two
  ancestry groups were combined without stratification.

For this patient, a single score cannot be diagnosed from one value. The diagnostic is run on the
cohort, not the individual.

### 6.9 Diagnostic 2: allele frequency and effect allele concordance

The second diagnostic is a comparison of the effect allele frequency in the target cohort against
the frequency reported in the discovery study, per variant.

```text
  Expected for a common variant: |f_target - f_discovery| < 0.05
  Flag: |difference| between 0.05 and 0.15
  Reject: |difference| > 0.15, or the variant is monomorphic in the target
```

A systematic shift across many variants indicates a strand or build problem. A shift at a subset
of variants indicates ancestry differences or a genotyping error. A variant that is monomorphic
in the target but common in the discovery cohort contributes nothing to the score in this
population, which is a portability problem rather than a coding error.

The two diagnostics are complementary. The first detects a distribution-level mismatch; the
second localizes it to specific variants.

### 6.10 The varied assumption

Change one assumption: the patient is of West African ancestry, and the score was discovered and
validated in Europeans.

Published estimates put relative predictive performance in African-ancestry samples at roughly
18% to 42% of European performance, depending on the score and the cohort. Take 25% as an
illustrative value for this score. The effective odds ratio per standard deviation is not 1.45;
the score explains about a quarter of the variance it explains in Europeans, which corresponds to
a much smaller odds ratio per standard deviation, approximately:

```text
  log OR_target = sqrt(0.25) x log OR_discovery
                = 0.5 x 0.37156
                = 0.18578
  OR_target     = exp(0.18578) = 1.204
```

The variance-scaling step assumes the score's correlation with the outcome scales with the square
root of the relative variance explained, which is an approximation. With an odds ratio of 1.204
per standard deviation at z = 1.751:

```text
  OR(z) = 1.204 ^ 1.751 = exp(1.751 x 0.18578) = exp(0.32530) = 1.3845
  odds  = 0.033058 x 1.3845 = 0.045768
  risk  = 0.045768 / 1.045768 = 4.38%
```

The ten-year risk is 4.4% rather than 6.0%. More importantly, the z-score of 1.751 was computed
against a European reference, so even the percentile is not interpretable. The correct action is
not to report 4.4%; it is to state that the score has not been validated in this population, to
use a multi-ancestry score if one exists, or to omit the PRS from the decision.

The conclusion changes in kind, not just in magnitude: from "the score raises estimated risk by
2.8 percentage points and may support a statin discussion" to "the score cannot be interpreted
without ancestry-specific calibration".

## 7. Code, query, or calculation

The script below performs harmonization, scoring, standardization, absolute-risk conversion,
winner's curse shrinkage, NRI, DCA, and a cohort calibration check, using only the Python standard
library. The weights are illustrative. The output was produced by running the script as written.

```python
"""prs.py - score construction, calibration, and clinical utility arithmetic.

Standard library only. Run: python prs.py
Weights and genotypes are illustrative, not a validated score.
"""
from statistics import NormalDist

# ---------------------------------------------------------------- scoring file
# rsid, effect allele, other allele, effect weight (log odds per allele)
WEIGHTS = [
    ("rs10757274", "G", "A", 0.28),
    ("rs1333049",  "C", "G", 0.22),
    ("rs9349379",  "A", "G", 0.18),
    ("rs20455",    "G", "A", 0.15),
    ("rs662",      "T", "C", 0.09),
    ("rs4380028",  "C", "T", 0.07),
    ("rs9818870",  "T", "C", 0.06),
    ("rs3184504",  "T", "C", 0.11),
]

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def dosage(genotype, effect_allele, other_allele):
    """Return 0/1/2 effect-allele dosage, or None if the alleles do not match."""
    a, b = genotype
    pair = {a, b}
    if pair <= {effect_allele, other_allele}:
        return (a == effect_allele) + (b == effect_allele)
    flipped_effect = COMPLEMENT[effect_allele]
    flipped_other = COMPLEMENT[other_allele]
    if pair <= {flipped_effect, flipped_other}:
        return (a == flipped_effect) + (b == flipped_effect)
    return None


def raw_prs(genotypes):
    total = 0.0
    used = 0
    for rsid, ea, oa, beta in WEIGHTS:
        gt = genotypes.get(rsid)
        if gt is None:
            continue
        d = dosage(gt, ea, oa)
        if d is None:
            continue
        total += beta * d
        used += 1
    return total, used


PATIENTS = {
    "PT-A": {"rs10757274": ("G", "G"), "rs1333049": ("C", "C"), "rs9349379": ("A", "G"),
             "rs20455": ("G", "A"), "rs662": ("T", "C"), "rs4380028": ("C", "C"),
             "rs9818870": ("T", "T"), "rs3184504": ("T", "C")},
    "PT-B": {"rs10757274": ("A", "A"), "rs1333049": ("G", "G"), "rs9349379": ("G", "G"),
             "rs20455": ("A", "A"), "rs662": ("C", "C"), "rs4380028": ("T", "T"),
             "rs9818870": ("C", "C"), "rs3184504": ("C", "C")},
    "PT-C": {"rs10757274": ("G", "A"), "rs1333049": ("C", "G"), "rs9349379": ("A", "A"),
             "rs20455": ("G", "G"), "rs662": ("T", "T"), "rs4380028": ("C", "T"),
             "rs9818870": ("T", "C"), "rs3184504": ("T", "T")},
}

REFERENCE_MEAN = 0.83
REFERENCE_SD = 0.35
OR_PER_SD = 1.45


def absolute_risk(baseline_risk, z, or_per_sd):
    """Convert a z-score to absolute risk through the baseline odds."""
    baseline_odds = baseline_risk / (1.0 - baseline_risk)
    individual_odds = baseline_odds * (or_per_sd ** z)
    return individual_odds / (1.0 + individual_odds)


def nri(events_up, events_down, n_events, non_up, non_down, n_non):
    return (events_up - events_down) / n_events + (non_down - non_up) / n_non


def net_benefit(tp, fp, n, threshold):
    return tp / n - (fp / n) * (threshold / (1 - threshold))


def main():
    nd = NormalDist()
    print("PATIENT  RAW_PRS  SNPS  Z      PERCENTILE  RISK_3.2%  RISK_8.0%")
    for pid, gts in PATIENTS.items():
        score, used = raw_prs(gts)
        z = (score - REFERENCE_MEAN) / REFERENCE_SD
        pct = 100.0 * nd.cdf(z)
        r_low = absolute_risk(0.032, z, OR_PER_SD)
        r_high = absolute_risk(0.080, z, OR_PER_SD)
        print(f"{pid:<8} {score:7.3f} {used:5d} {z:6.2f} {pct:11.1f} {r_low:10.3%} {r_high:10.3%}")

    # Winner's curse illustration: a discovery beta significant at p=5e-8 in
    # a study of n=10,000 has a large standard error; shrinking it changes the score.
    beta_obs = 0.28
    se = 0.050
    z_stat = beta_obs / se
    shrinkage = max(0.0, 1.0 - 1.0 / (z_stat ** 2))
    print(f"\nWINNER'S CURSE")
    print(f"  observed beta      : {beta_obs:.3f} log-odds per allele")
    print(f"  standard error     : {se:.3f}")
    print(f"  z statistic        : {z_stat:.2f}")
    print(f"  simple shrinkage   : {shrinkage:.3f} (multiply beta by this)")
    print(f"  shrunk beta        : {beta_obs * shrinkage:.3f}")

    # Reclassification: 10,000 women, 400 events.
    print(f"\nNET RECLASSIFICATION IMPROVEMENT")
    val = nri(events_up=40, events_down=10, n_events=400,
              non_up=500, non_down=400, n_non=9600)
    print(f"  events up / down   : 40 / 10 of 400")
    print(f"  non-events up/down : 500 / 400 of 9600")
    print(f"  NRI                : {val:+.4f}")

    # Decision curve analysis at a 5% threshold.
    print(f"\nDECISION CURVE AT A 5% THRESHOLD (n=10,000, 400 events)")
    for label, tp, fp in [("treat all", 400, 9600),
                          ("clinical model", 300, 1200),
                          ("clinical + PRS", 340, 1460)]:
        nb = net_benefit(tp, fp, 10000, 0.05)
        print(f"  {label:<16} TP={tp:4d} FP={fp:5d} net benefit={nb:+.4f}")

    # Ancestry check: mean z-score of a small target cohort should be near 0
    # when the reference distribution matches the cohort.
    cohort_z = [-0.9, 0.4, 1.2, -1.1, 2.0]
    mean_z = sum(cohort_z) / len(cohort_z)
    print(f"\nCALIBRATION CHECK")
    print(f"  cohort mean z      : {mean_z:+.3f} (flag if |mean| > 0.20)")
    print(f"  cohort SD of z     : {(sum((x - mean_z) ** 2 for x in cohort_z) / (len(cohort_z) - 1)) ** 0.5:.3f}")


if __name__ == "__main__":
    main()
```

Output:

```text
PATIENT  RAW_PRS  SNPS  Z      PERCENTILE  RISK_3.2%  RISK_8.0%
PT-A       1.790     8   2.74        99.7     8.391%    19.416%
PT-B       0.000     8  -2.37         0.9     1.351%     3.477%
PT-C       1.690     8   2.46        99.3     7.610%    17.809%

WINNER'S CURSE
  observed beta      : 0.280 log-odds per allele
  standard error     : 0.050
  z statistic        : 5.60
  simple shrinkage   : 0.968 (multiply beta by this)
  shrunk beta        : 0.271

NET RECLASSIFICATION IMPROVEMENT
  events up / down   : 40 / 10 of 400
  non-events up/down : 500 / 400 of 9600
  NRI                : +0.0646

DECISION CURVE AT A 5% THRESHOLD (n=10,000, 400 events)
  treat all        TP= 400 FP= 9600 net benefit=-0.0105
  clinical model   TP= 300 FP= 1200 net benefit=+0.0237
  clinical + PRS   TP= 340 FP= 1460 net benefit=+0.0263

CALIBRATION CHECK
  cohort mean z      : +0.320 (flag if |mean| > 0.20)
  cohort SD of z     : 1.333
```

Reading the output:

- PT-B has a raw score of exactly 0.000 because it carries no effect alleles at any of the eight
  variants. Its z-score of -2.37 corresponds to the 0.9th percentile, and its ten-year risk at a
  3.2% baseline is 1.35%. This is the correct behavior, not a bug: a low score lowers risk.
- The winner's curse illustration uses a crude shrinkage factor of 1 minus 1 over z squared. It is
  a teaching device, not a recommended estimator; LDpred2 and PRS-CS use proper priors.
- The NRI of +0.0646 combines an events component of +0.075 and a non-events component of
  -0.0104. Reporting only the events component would overstate the benefit.
- The decision curve shows the clinical model at +0.0237 and the clinical plus PRS model at
  +0.0263 at a 5% threshold, an increment of 0.0026. The increment is small and its confidence
  interval is not computed here.
- The calibration check flags a cohort mean z of +0.320. In a real program this would trigger an
  investigation before any patient report is issued.

## 8. Data, standards, and regulatory context

### 8.1 The PGS Catalog

The PGS Catalog is maintained by the European Molecular Biology Laboratory's European
Bioinformatics Institute and the University of Cambridge, and is the standard public repository
for published scores. When accessed on 26 October 2026 it listed more than 7,000 scores across
more than 800 traits.

Each score has a scoring file with a header and a variant table:

| Field | Meaning |
|---|---|
| `pgs_id` | Stable score identifier, for example PGS000001 |
| `trait_reported` | Trait as described by the authors |
| `trait_mapped` | Mapped trait label |
| `trait_efo` | Experimental Factor Ontology term |
| `genome_build` | Build of the positions in the file |
| `variants_number` | Number of variants in the score |
| `weight_type` | Whether weights are beta, odds ratio, or another type |
| `effect_allele` | Allele to which the weight refers |
| `other_allele` | Alternate allele |
| `effect_weight` | Weight per effect allele |

Harmonized versions are provided for GRCh37 and GRCh38, with added columns recording the
harmonization source and match status. The `pgsc_calc` tool applies scores to target genotypes and
handles many of the harmonization cases described in step 3.

What the catalog does: makes scores reproducible and comparable. What it does not do: validate
them. Inclusion means the authors deposited a scoring file, not that the score has clinical
utility, calibration in any population, or regulatory clearance.

### 8.2 The PRS-RS reporting standard

The Polygenic Risk Score Reporting Standards (PRS-RS), published in Nature in 2021, define 22
items that a PRS study or report should contain. They update the earlier Genetic Risk Prediction
Studies statement. The items cover study purpose, populations and ancestry, genetic data,
outcome definitions, score construction, model fitting, evaluation, subgroup analyses,
limitations, and data availability.

The standard is a reporting standard, not a clinical guideline. It constrains what a paper or a
report must disclose. It does not say when a score should be used.

### 8.3 Clinical implementation evidence

The eMERGE network's 2024 report described the selection, optimization, and validation of ten
chronic disease scores for return to 25,000 participants across diverse United States
populations. Two design choices are instructive:

- **Ancestry-calibrated z-scores.** Rather than standardizing against a single European
  reference, the program derived z-scores from an ancestry-diverse reference cohort of about
  13,475 All of Us participants.
- **Absolute risk in the report.** For breast cancer and coronary heart disease, the score fed an
  integrated absolute risk calculation rather than being returned as a percentile alone.

The program also illustrates the regulatory path: scores returned as clinical results required
CLIA-compliant laboratory processes, a validation package, and a report reviewed by clinical
leadership.

### 8.4 Laboratory and regulatory status

In the United States, most clinical PRS are offered as laboratory developed tests under CLIA
certification. Key constraints:

- **Analytical validation.** The assay must be validated for the variants it reports, including
  imputation accuracy if genotypes are imputed.
- **Clinical validation.** The score's association with the outcome must be established in the
  population where it is used, with calibration reported.
- **No companion diagnostic role.** As of this writing, no polygenic score functions as an FDA
  companion diagnostic for a specific drug. PRS are risk stratification tools, not treatment
  selection tests.
- **LDT oversight.** The FDA's 2024 rule on laboratory developed tests was contested in
  litigation, and the effective status should be checked before relying on it.

The European approach differs: the In Vitro Diagnostic Regulation classifies tests by risk, and a
PRS used to inform prevention in an asymptomatic person is likely to require conformity
assessment. The details depend on the intended use statement.

### 8.5 What the standards do not cover

- They do not define a minimum performance threshold for clinical use.
- They do not specify how to communicate a percentile without implying a probability.
- They do not resolve how to report a score in an admixed individual.
- They do not say whether a score should be offered at all when the evidence is limited to one
  ancestry group.

Those decisions belong to the program, and they should be written down before the first report is
issued.

<!--PART2-->
## 9. Industry practice and case evidence

### 9.1 How scores reach patients

There are four routes, and they differ in validation and in what the patient receives.

| Route | Who orders | Validation expected | What the patient sees |
|---|---|---|---|
| Research return (eMERGE, biobank) | Research program | Program-defined; often CLIA-confirmed before return | A report with absolute risk for selected conditions |
| Clinical laboratory LDT | Clinician | Analytical and clinical validation under CLIA | A percentile and, increasingly, an absolute risk |
| Direct-to-consumer | Consumer | Consumer genomics standards; not a clinical result in most jurisdictions | A percentile, often with educational text |
| Integrated clinical model (BOADICEA, iCARE) | Clinician | Published model validation | A combined absolute risk with the score as one input |

The eMERGE network's 2024 report is the most detailed public account of the research-return route.
It selected ten chronic disease scores, optimized and validated them, and returned results to
25,000 participants. Two design decisions are worth copying: ancestry-calibrated z-scores from a
diverse reference, and integration with absolute risk for the conditions where a decision depends
on it.

### 9.2 What the evidence shows about reporting

A 2025 systematized review of PRS reports found consistent problems across 13 studies: absolute
risk is the most acceptable format, percentiles are often confused with probabilities, odds
ratios are poorly understood, and icon arrays or bell curves improve comprehension. The review
also found no evidence-based report template.

This is a documented gap, not a solved problem. A program that reports a percentile without an
absolute risk is not following a best practice; it is repeating a known error.

### 9.3 The ancestry evidence in numbers

| Study | Design | Reported result |
|---|---|---|
| Privé et al., 2022 | 245 scores trained in UK Biobank Europeans, applied to nine ancestry groups | Relative partial R-squared of 18.0% in Nigeria, 25.2% in the Caribbean, 48.6% in China, 64.7% in India, 85.6% in Italy, 93.8% in Poland |
| Duncan et al., 2019 | Meta-analysis of 26 studies | Median performance in African-ancestry samples at 42% of matched European samples; South Asian 60%; East Asian 95% |
| Majara et al., 2023 | 34 traits, UK Biobank Europeans to other groups | Relative accuracy 0.219 in African, 0.477 in East Asian, 0.621 in Central and South Asian groups |
| Cavazos and Witte, 2020 | Simulation with admixed individuals | Accuracy declines roughly linearly with the proportion of African ancestry |

The numbers differ because the scores, traits, and target cohorts differ. The direction does not.
A single continental label also hides within-group variation: Majara et al. reported relative
accuracy ranging from 0.083 in South African to 0.511 in Ethiopian groups, a spread wider than
some between-continent differences.

### 9.4 Commercial and clinical reality

A 2025 landscape analysis of 28 professional-group position statements found no group that
recommended a complete implementation framework for PRS. The statements converge on a few points:
PRS should not be used alone, ancestry must be reported, and clinical utility must be demonstrated
before reimbursement.

That is the honest state of the field. The arithmetic is settled; the evidence for clinical
utility is strong for a small number of conditions and weak for most; and the reporting standards
are ahead of the practice.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Effect allele not harmonized | Score distribution shifted; cohort mean z far from 0 | Strand flip or allele swap at some variants | Score is partly noise; direction unpredictable | Allele frequency concordance check per variant |
| Wrong genome build | Many variants unmatched after position join | GRCh37 weights joined to GRCh38 genotypes | Large fraction of score missing | Harmonized scoring files; assert the build |
| Reference population mismatch | Percentile implausible for the patient's ancestry | Standardization against a different group | Every percentile wrong in the same direction | Ancestry-matched or ancestry-calibrated reference |
| Percentile reported as probability | Patient believes "96th percentile" means 96% risk | Rank and probability conflated in the report | Unnecessary anxiety or false reassurance | Lead with absolute risk; label the percentile |
| Relative risk without baseline | Report gives an odds ratio only | Baseline omitted | Clinician cannot judge absolute benefit | Compute and display absolute risk |
| Winner's curse uncorrected | Score performs worse out of sample than in discovery | Selected effect sizes inflated | Overconfident risk estimates | Bayesian shrinkage or cross-validation |
| LD panel mismatch | Accuracy lower than published, especially in non-European groups | Shrinkage uses the wrong correlation structure | Systematic underperformance | Ancestry-matched LD panel |
| Calibration never checked | No observed-versus-expected report | Validation measured discrimination only | Miscalibrated absolute risks | Calibration slope, intercept, and observed-to-expected ratio |
| Utility claimed from AUC | Report shows only an AUC improvement | Discrimination substituted for utility | No decision benefit demonstrated | NRI and decision curve at the relevant threshold |
| Multiple testing across scores | The best of 20 scores looks good | Selection without correction | Inflated apparent performance | Pre-specify one score; correct for selection |
| Case-control ascertained sample used as population baseline | Absolute risks too high | Odds ratios from an enriched sample applied to population risk | Over-treatment | Convert to liability scale or use population cohort baselines |
| Imputation quality ignored | Many low-quality dosages included | Noisy genotypes treated as observed | Regression dilution; attenuated score | Filter on imputation \(r^2\) |
| Score applied outside its age range | Risk estimates implausible at extremes of age | Weights estimated in a narrow age band | Wrong absolute risk | State and enforce the validated age range |
| No post-deployment monitoring | Calibration never rechecked | Population and treatment change over time | Silent drift | Annual recalibration audit by ancestry |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The scoring file and the genotypes use the same build and strand | Weights are position- and allele-specific | Build mismatch, strand flip, multiallelic site | Allele frequency concordance per variant |
| The discovery population is relevant to the target | Effect sizes are population-specific | Different ancestry or environment | Cross-ancestry validation |
| The reference distribution matches the patient | Percentile is a rank within the reference | Different ancestry, array, or variant set | Cohort mean and SD of z |
| The odds ratio per SD is constant across the risk range | Conversion to absolute risk assumes a constant effect | Strong interaction with baseline risk | Calibration across risk deciles |
| The baseline risk is appropriate for the patient | Absolute risk requires a baseline | Baseline from a different era or population | Compare with local cohort incidence |
| The outcome definition in discovery matches the intended use | A score for prevalent disease may not predict incident disease | Different ascertainment or case definition | Check phenotype definitions |
| Missing variants are missing at random | Dropping variants assumes no informative missingness | Ancestry- or array-related missingness | Compare variant counts by group |
| The score adds information beyond the comparator | Utility requires incremental value | Score correlates with existing predictors | Decision curve against the comparator |
| Decision thresholds are stable | Net benefit depends on the threshold | Guideline change | Re-run DCA at the current threshold |
| Reported performance generalizes to the clinic | Trial cohorts differ from practice | Spectrum effects, comorbidity, adherence | Local calibration audit |

## 12. Exercises

### Exercise 1. Score two patients by hand

**Task.** Using the eight-variant scoring file in section 7, compute the raw score for a patient
who is homozygous for the effect allele at rs10757274, heterozygous at rs1333049, and homozygous
for the other allele at the remaining six variants. Then compute the z-score against a mean of
0.83 and a standard deviation of 0.35.

**Deliverable.** The arithmetic and the resulting percentile.

**Acceptance criteria.** Raw score is 0.28 x 2 + 0.22 x 1 = 0.78. z = (0.78 - 0.83) / 0.35 =
-0.143. Percentile is approximately 44.3. The answer states that the percentile is a rank, not a
risk.

**Time.** 20 minutes.

### Exercise 2. Convert an odds ratio to absolute risk

**Task.** A score has an odds ratio of 1.35 per standard deviation. A patient is at the 90th
percentile (z = 1.282) and has a baseline ten-year risk of 6.0%. Compute the absolute risk using
the odds method and using the naive multiplicative method, and report the difference.

**Deliverable.** Both calculations with units.

**Acceptance criteria.** Odds method: OR(z) = 1.35^1.282 = 1.475; baseline odds = 0.06383;
individual odds = 0.09415; risk = 8.61%. Naive: 1 - 0.94^1.475 = 8.72%. Difference 0.11
percentage points. The answer notes that the two agree closely at this baseline.

**Time.** 25 minutes.

### Exercise 3. Diagnose a harmonization error

**Task.** A cohort of 2,000 patients has a mean z-score of -0.41 and a standard deviation of 1.6
after scoring. The effect allele frequency at rs1333049 is 0.31 in the cohort and 0.47 in the
discovery study. Propose the most likely cause, a confirmatory test, and a fix.

**Deliverable.** A one-page diagnosis.

**Acceptance criteria.** The diagnosis names allele harmonization or ancestry mismatch as the two
candidates, proposes a per-variant frequency comparison to distinguish them, and states that a
systematic shift across many variants favors harmonization while a shift at a subset favors
ancestry or genotyping error.

**Time.** 30 minutes.

### Exercise 4. Compute NRI

**Task.** In 5,000 patients with 250 events, adding a PRS moves 25 events up and 5 down, and moves
300 non-events up and 200 down. Compute the NRI and its two components.

**Deliverable.** The arithmetic and one sentence on interpretation.

**Acceptance criteria.** Events component (25 - 5) / 250 = 0.080. Non-events component
(200 - 300) / 4,750 = -0.0211. NRI = +0.0589. The interpretation notes that most of the gain
comes from events and that the non-event component is negative.

**Time.** 25 minutes.

### Exercise 5. Build a decision curve

**Task.** At a 10% threshold, a clinical model treats 900 of 10,000 patients with 250 events and
650 false positives; the model plus PRS treats 1,050 with 290 events and 760 false positives.
Compute net benefit for both and for treat-all, and state whether the PRS adds value.

**Deliverable.** A table of net benefit at 10%.

**Acceptance criteria.** Treat-all: 0.025 - 0.975 x (0.10/0.90) = -0.0833. Clinical:
0.025 - 0.065 x 0.1111 = 0.0178. Clinical plus PRS: 0.029 - 0.076 x 0.1111 = 0.0206. The
increment is 0.0028, and the answer states that it is small and needs a confidence interval.

**Time.** 30 minutes.

### Exercise 6. Audit a consumer report

**Task.** Critique a report that says "Your coronary artery disease risk: 94th percentile. OR
1.8 per SD. This is a high-risk result." Rewrite it in six sentences.

**Deliverable.** The critique and the rewrite.

**Acceptance criteria.** The critique identifies the missing absolute risk, the missing reference
population, the ambiguous odds ratio, and the undefined "high risk". The rewrite includes a
ten-year absolute risk with a baseline, a percentile labeled as a rank, the ancestry of the
reference, and a statement of what the result does not mean.

**Time.** 40 minutes.

### Exercise 7. Choose a method

**Task.** You have European summary statistics with 400,000 cases and controls, an African
summary statistics set with 15,000, and a target cohort that is 60% African-ancestry and 40%
European-ancestry. Recommend a scoring method and justify it.

**Deliverable.** A two-page method memo.

**Acceptance criteria.** The memo considers PRS-CSx or a multi-ancestry approach, explains why a
single-ancestry LD panel is insufficient, addresses the sample-size imbalance, and states how
performance will be reported separately for the two groups.

**Time.** 50 minutes.

### Exercise 8. Design the monitoring plan

**Task.** Specify a post-deployment monitoring plan for a coronary PRS returned to 25,000
patients over three years.

**Deliverable.** A monitoring specification with metrics, frequency, and thresholds.

**Acceptance criteria.** The plan measures calibration by ancestry group at least annually,
tracks the observed-to-expected ratio with an action threshold, monitors the score distribution
for drift, records whether the score changed a decision, and defines who reviews a threshold
breach.

**Time.** 45 minutes.

## 13. Self-check answers

**Exercise 1.** Effect allele dosages are 2 at rs10757274, 1 at rs1333049, and 0 at the other six.
Raw score = 0.28 x 2 + 0.22 x 1 = 0.56 + 0.22 = 0.78. z = (0.78 - 0.83) / 0.35 = -0.05 / 0.35 =
-0.1429. The percentile is the normal CDF at -0.1429, which is 0.4432, or 44.3. A patient at the
44th percentile is slightly below the reference median. The score says nothing about absolute
risk without a baseline.

**Exercise 2.** OR(z) = 1.35^1.282. ln 1.35 = 0.30010; multiplied by 1.282 = 0.38473; exp =
1.4691. Baseline odds = 0.06 / 0.94 = 0.06383. Individual odds = 0.06383 x 1.4691 = 0.09377. Risk
= 0.09377 / 1.09377 = 8.57%. Naive: 1 - 0.94^1.4691 = 1 - exp(1.4691 x -0.061875) = 1 -
exp(-0.09090) = 8.69%. Difference 0.12 percentage points. Both are approximations; a local
calibration model is preferable.

**Exercise 3.** Two candidates fit the evidence. A mean z of -0.41 indicates a systematic shift,
and a standard deviation of 1.6 indicates a scale mismatch. The allele frequency difference at
rs1333049 of 0.16 exceeds the 0.15 rejection threshold, which points to a strand or build problem
at that variant, but one variant cannot explain the cohort mean. The confirmatory test is the
per-variant frequency difference across all variants: if most variants differ in the same
direction, the cause is harmonization or a build mismatch; if only a subset differs, the cause is
ancestry or a genotyping error. The fix is to re-harmonize against the correct build, re-run the
reference standardization, and re-check the mean and standard deviation.

**Exercise 4.** Events component = (25 - 5) / 250 = 20 / 250 = 0.080. Non-events component =
(200 - 300) / 4,750 = -100 / 4,750 = -0.02105. NRI = 0.080 - 0.021 = 0.0589, or 5.9%. The NRI is
positive, but it is dominated by the events component and the non-events component is negative,
which means some non-events were moved into higher-risk categories. NRI is sensitive to the
choice of category boundaries, so it should be reported with the boundaries and a confidence
interval.

**Exercise 5.** At a 10% threshold, the threshold odds are 0.10 / 0.90 = 0.1111. Treat-all: 250 /
10,000 - (9,750 / 10,000) x 0.1111 = 0.025 - 0.1083 = -0.0833. Clinical model: true positives 250,
false positives 650, so net benefit = 0.025 - 0.065 x 0.1111 = 0.025 - 0.00722 = 0.01778.
Clinical plus PRS: 290 / 10,000 - (760 / 10,000) x 0.1111 = 0.029 - 0.00844 = 0.02056. The
increment is 0.00278. The PRS adds a small amount of net benefit at this threshold, and the
increment needs a confidence interval before it supports a claim.

**Exercise 6.** Rubric. Full credit requires all four missing elements identified and a rewrite
that leads with an absolute risk, labels the percentile as a rank, names the reference ancestry,
and states a limit. A rewrite that repeats "94th percentile" without a baseline fails.

**Exercise 7.** Rubric. Full credit requires a multi-ancestry method such as PRS-CSx, an
explanation that LD and effect sizes differ by ancestry, a plan for the sample-size imbalance such
as weighting or shrinkage toward the shared component, and separate performance reporting. A
single European score applied to all participants fails.

**Exercise 8.** Rubric. Full credit requires calibration by ancestry, a defined metric and
threshold, a named reviewer, a decision-impact measure, and a drift check on the score
distribution. A plan that monitors only the number of tests ordered fails.

## 14. Interview questions

**1. What is a polygenic risk score, in one sentence, and what does it not tell you?**
A strong answer describes a weighted sum of risk allele dosages and states that it does not
provide an absolute risk without a baseline and a reference population. A weak answer calls it a
genetic test for disease.

**2. Why is a percentile not a probability?**
A strong answer explains that a percentile is a rank within a reference distribution and that the
probability depends on baseline risk. A weak answer treats the two as interchangeable.

**3. How would you detect a strand flip in a scoring pipeline?**
A strong answer compares effect allele frequencies per variant against the discovery study and
looks for a systematic shift, then checks the score distribution's mean and standard deviation.
A weak answer says "check the rsIDs".

**4. Why does a European-trained score lose accuracy in African-ancestry populations?**
A strong answer covers LD differences, allele frequency differences, and environment, and cites a
quantitative range such as 18% to 42% relative performance. A weak answer says "genetics differ".

**5. What is the difference between discrimination and calibration?**
A strong answer separates ranking from agreement between predicted and observed risk, and names
the metrics for each. A weak answer uses the two terms interchangeably.

**6. When would you not report a PRS at all?**
A strong answer names an unvalidated ancestry, an absent reference distribution, a baseline that
cannot be estimated, and a decision threshold where the score has no net benefit. A weak answer
says "when the score is low".

**7. How does winner's curse affect a PRS?**
A strong answer explains that selection on significance inflates effect sizes, that the inflation
is larger in small studies and near the threshold, and that shrinkage or cross-validation
corrects it. A weak answer has not heard of the problem.

**8. What does decision curve analysis add over an AUC?**
A strong answer explains that net benefit incorporates the threshold and the harms of false
positives, and that a model can improve AUC without improving net benefit. A weak answer says DCA
is "another accuracy metric".

**9. How would you validate a score for a new population?**
A strong answer requires an independent cohort, calibration slope and observed-to-expected ratio,
subgroup reporting, and a utility analysis at the local threshold. A weak answer proposes
recomputing the AUC only.

**10. What belongs in a PRS report?**
A strong answer lists absolute risk, percentile labeled as a rank, reference population ancestry,
score identifier, validation performance, and limits. A weak answer lists only the score and a
risk category.

## 15. Key terms

- **Absolute risk** — probability of an event over a stated period.
- **Ancestry** — genetic similarity to a reference population, not a social category.
- **Calibration** — agreement between predicted and observed risk.
- **Clumping and thresholding** — pruning correlated variants and applying a p-value cutoff.
- **Decision curve analysis** — net benefit across decision thresholds.
- **Discrimination** — ability to rank cases above non-cases.
- **Dosage** — number of effect alleles, 0, 1, or 2.
- **Effect allele** — allele to which a reported effect size refers.
- **Genome-wide association study** — variant-by-variant association testing across the genome.
- **Harmonization** — aligning alleles, strand, and build between summary statistics and genotypes.
- **LDpred2** — Bayesian polygenic scoring method using an LD matrix.
- **Liability scale** — underlying continuous risk scale in a threshold model.
- **Linkage disequilibrium** — non-random association of alleles at different loci.
- **Net benefit** — true positives minus weighted false positives.
- **Net reclassification improvement** — change in correct classification across risk categories.
- **Number needed to treat** — patients treated to prevent one event.
- **Odds ratio** — ratio of odds between groups.
- **PGS Catalog** — public repository of polygenic scores and scoring files.
- **Polygenic risk score** — weighted sum of risk allele dosages.
- **Portability** — performance of a score in a population other than the discovery one.
- **PRS-CS** — Bayesian continuous shrinkage scoring method.
- **PRS-CSx** — multi-ancestry extension of PRS-CS.
- **PRS-RS** — 22-item reporting standard for polygenic scores.
- **Percentile** — rank within a reference distribution.
- **Reference population** — group whose score distribution defines the z-score.
- **Relative risk** — ratio of risks between groups.
- **Standardization** — expressing a raw score relative to a reference mean and standard deviation.
- **Summary statistics** — per-variant effect sizes, standard errors, and allele frequencies.
- **Winner's curse** — inflation of effect sizes selected for statistical significance.
- **z-score** — number of reference standard deviations from the mean.

## 16. Sources

Accessed 26 October 2026 unless otherwise stated.

- PGS Catalog. https://www.pgscatalog.org/
- PGS Catalog download documentation and scoring file format.
  https://www.pgscatalog.org/downloads/
- Wand, H., et al. *Improving Reporting Standards for Polygenic Scores in Risk Prediction
  Studies*. Nature 591, 211-219, 2021. https://doi.org/10.1038/s41586-021-03243-6
- Choi, S. W., Mak, T. S. H., and O'Reilly, P. F. *Tutorial: A Guide to Performing Polygenic Risk
  Score Analyses*. Nature Protocols 15, 2759-2772, 2020. https://doi.org/10.1038/s41596-020-0353-1
- Khera, A. V., et al. *Genome-wide Polygenic Scores for Common Diseases Identify Individuals with
  Risk Equivalent to Monogenic Mutations*. Nature Genetics 50, 1219-1224, 2018.
  https://doi.org/10.1038/s41588-018-0183-z
- Martin, A. R., et al. *Clinical Use of Current Polygenic Risk Scores May Exacerbate Health
  Disparities*. Nature Genetics 51, 584-591, 2019. https://doi.org/10.1038/s41588-019-0379-x
- Privé, F., et al. *Portability of 245 Polygenic Scores When Derived from the UK Biobank and
  Applied to 9 Ancestry Groups from the Same Cohort*. American Journal of Human Genetics, 2022.
  https://doi.org/10.1016/j.ajhg.2022.01.012
- Duncan, L., et al. *Analysis of Polygenic Risk Score Usage and Performance in Diverse Human
  Populations*. Nature Communications 10, 3328, 2019.
  https://doi.org/10.1038/s41467-019-11112-0
- Majara, L., et al. *Low and Differential Polygenic Score Generalizability among African
  Populations Due Largely to Genetic Diversity*. Human Genetics and Genomics Advances, 2023.
  https://doi.org/10.1016/j.xhgg.2023.100190
- Ge, T., et al. *Polygenic Prediction via Bayesian Regression and Continuous Shrinkage Priors*.
  Nature Communications 10, 1776, 2019. https://doi.org/10.1038/s41467-019-09718-5
- Privé, F., et al. *LDpred2: Better, Faster, Stronger*. Bioinformatics 36, 5424-5431, 2020.
  https://doi.org/10.1093/bioinformatics/btaa1029
- Lennon, N. J., et al. *Selection, Optimization and Validation of Ten Chronic Disease Polygenic
  Risk Scores for Clinical Implementation in Diverse US Populations*. Nature Medicine 30,
  480-487, 2024. https://doi.org/10.1038/s41591-024-02796-z
- Hanley, et al. *The Development and Evaluation of Polygenic Risk Score Reports: A Systematized
  Review of the Literature*. Genetics in Medicine, 2025.
  https://www.gimjournal.org/article/S1098-3600(25)00073-5/fulltext
- Purvis, et al. *Defining Next Steps in the Clinical Implementation of Polygenic Scores: A
  Landscape Analysis of Professional Groups' Perspectives*. Genetics in Medicine 27, 101414,
  2025. https://doi.org/10.1016/j.gim.2025.101414

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`11_pharmacogenomics-cds-and-cpic.md`](11_pharmacogenomics-cds-and-cpic.md)
- Next topic: [`13_spatial-omics-and-pathology-integration.md`](13_spatial-omics-and-pathology-integration.md)
- Related topic: [`02_genomics-and-variants.md`](02_genomics-and-variants.md)
- Related topic: [`08_biomarker-discovery.md`](08_biomarker-discovery.md)
- Module survey: [`../../course/12_omics_precision_medicine.md`](../../course/12_omics_precision_medicine.md)
- Laboratory exercises and datasets: [`../../labs/`](../../labs/)
- Calibration and decision curve lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)

