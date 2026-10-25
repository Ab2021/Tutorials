# Multi-Omics Integration: When Combining Layers Earns Its Complexity

> Module 12 — Omics and Precision Medicine
> Topic 07 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01A, 04, topics 01, 03, 04, 06
> Version 2026.3

## 1. Why this topic exists

Every omics platform produces a matrix. Once a program has three matrices for the same samples,
someone proposes integrating them, and the proposal is almost never argued against. Integration
sounds like more evidence, and more evidence sounds better than less. In practice, integration
adds parameters, assumptions, and failure modes, and it often produces a result that is harder
to interpret and no better at prediction than the best single layer.

The question that should precede any integration is not "can we combine these data?" but "what
does combining them estimate that no single layer estimates?" There are good answers. A
regulatory network is only visible when transcriptome and chromatin accessibility are considered
together. A disease subtype may be driven by copy number in one patient and methylation in
another, and only a joint model finds a shared structure. A protein biomarker may be predicted
better from RNA plus protein than from either alone. There are also bad answers: concatenating
matrices because they exist, fitting a latent-factor model and interpreting the factors
post hoc, and reporting an integrated classifier without a single-omics baseline.

The industry cost of bad integration is concrete. Multi-omics studies are expensive, and a
program that spends a year on an integrated model that beats the best single layer by 0.01 AUC
has spent a year on nothing. A program that reports an integrated classifier without
cross-validated baselines cannot tell whether the integration helped, and a reviewer will ask.

The economics reinforce the point. A three-layer study on 200 patients can cost several million
dollars once sequencing, proteomics, methylation arrays, and analysis are counted. The marginal
cost of the second and third layers is high, and the marginal information is often low because
the layers are correlated: RNA and protein for the same gene are related, and methylation near a
promoter is related to expression. The layers are not independent measurements of independent
processes; they are partially redundant measurements of one process. Integration is therefore a
bet on the non-redundant part, and the size of that part is an empirical question that the
baseline comparison answers.

This topic covers the integration strategies, the conditions under which each is justified, the
evaluation that separates a real gain from an artifact, and the interpretation of latent
factors. It assumes the data types in topics 01 through 06 and feeds target and biomarker work
in topics 08 and 09.

### 1.1 The failure pattern

1. Three matrices are available for 80 samples: RNA, protein, and methylation.
2. They are concatenated after z-scoring each feature.
3. A classifier is trained on the concatenated matrix with 5-fold cross-validation.
4. The integrated classifier reaches AUC 0.86, versus 0.71 for RNA alone.
5. The gain is reported as evidence that multi-omics integration improves prediction.
6. Nobody checks that feature selection was done inside the cross-validation folds, or that the
   layers were normalized using all samples.
7. An external cohort gives AUC 0.62 for the integrated model and 0.68 for RNA alone.

The leak in step 6 is the mechanism. Feature selection on the full dataset, or normalization
using all samples, lets information from the validation fold enter training, and the leak grows
with the number of features. Integration adds features, so it amplifies the leak. The integrated
model looks better in cross-validation precisely because it has more opportunities to leak.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Program lead | A year of a multi-omics budget | Depends on the analysis team's evaluation |
| Statistician | Credibility of the integrated result | Did not design the cross-validation |
| Biologist | An interpretable mechanism | Receives latent factors with no labels |
| Translational lead | A classifier that does not transport | Depends on the validation cohort |
| Reviewer or regulator | A claim that cannot be assessed | Receives only the integrated performance |

A related cost is interpretive. A latent-factor model produces factors, and factors require
interpretation, and interpretation requires biological knowledge that the modelling team may not
have. The result is a set of factors with names chosen after the fact: "the interferon factor,"
"the proliferation factor," "the stromal factor." Some of those names survive validation; many
do not. The naming is not the analysis, but it is what the program remembers.

### 1.3 The industry question

> Is this molecular signal a validated biomarker, a causal target, or an artifact of batch,
> ancestry, and tissue composition?

Integration adds a fourth possibility: the signal is an artifact of the integration method. A
latent factor can be driven by a batch, a sample-quality gradient, or the correlation structure
of the concatenated matrix rather than by biology.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** what an integration estimates that no single layer estimates, in one sentence.
2. **Distinguish** early, intermediate, late, network, and Bayesian integration and the
   assumptions of each.
3. **Check** sample alignment and per-layer preprocessing before any joint model.
4. **Evaluate** an integrated model against single-omics baselines with leakage-free
   cross-validation.
5. **Interpret** latent factors using loadings, variance explained, and external association.
6. **Diagnose** a factor that is a batch or quality gradient.
7. **Handle** missing modalities without imputing them into existence.
8. **Report** an integration result with the comparison that justifies it.
9. **Decide** when not to integrate, and defend the decision with the same evidence standard.
10. **Size** an integration study for the smallest gain that would change a decision.

## 3. The core mental model

Integration is a bet that the shared structure across layers is larger than the layer-specific
noise. The bet is sometimes right and always testable.

```text
  Layer 1 (RNA)      Layer 2 (protein)      Layer 3 (methylation)
      |                    |                       |
      +--------------------+-----------------------+
                           |
                 sample alignment check
                           |
                 per-layer normalization
                           |
        +------------------+-------------------+
        |                  |                   |
   Early fusion      Intermediate         Late fusion
   concatenate       shared latent        combine predictions
        |                  |                   |
        +------------------+-------------------+
                           |
                 evaluation against baselines
                           |
        +------------------+-------------------+
        |                                      |
   better than best layer?              not better
        |                                      |
   interpret and validate              report the negative result
```

### 3.1 The three alignment questions

Before any model, three questions decide whether integration is even defined.

**Same samples?** The matrices must be indexed by the same biological units. A RNA matrix with
80 samples and a protein matrix with 74 samples, where the missing six are not random, is not
aligned. The intersection is 74 samples, and the six excluded may be the most diseased.

**Same features?** Layers measure different features, which is the point, but they must refer to
the same annotation build and the same identifier space where they overlap. A gene symbol from
one pipeline and an Ensembl ID from another cannot be joined without a mapping table.

**Same question?** The layers must inform a common outcome or a common latent process. If RNA
answers a mechanism question and protein answers a diagnostic question, a joint model estimates
neither well.

### 3.2 Five strategies and what each assumes

| Strategy | What it does | Assumes | Typical method |
|---|---|---|---|
| Early fusion | Concatenates features | Layers are commensurable after scaling; relevant signal is sparse | Penalized regression, random forest |
| Intermediate | Learns a shared latent space | Shared factors explain covariance across layers | MOFA, iCluster, CCA, PLS |
| Late fusion | Trains per-layer models, combines predictions | Layers are conditionally independent given the outcome | Stacking, averaging |
| Network | Builds a graph across features or layers | Similarity is biologically meaningful | Similarity network fusion |
| Bayesian | Joint generative model with priors | Prior structure and likelihoods are correct | Hierarchical models |

Early fusion is the simplest and the most prone to scale and dimensionality problems: if one
layer has 20,000 features and another has 200, the larger layer dominates unless weights are
tuned. Intermediate methods are the most interpretable when they work, because the factors can
be examined. Late fusion is the most robust to layer-specific noise and the least able to
capture interactions.

The three questions are ordered: alignment, then question, then strategy. A team that starts
with the strategy has usually skipped the first two.

### 3.3 What a factor is and is not

A latent factor is a direction in the joint space that explains covariance. It is not a pathway,
a cell type, or a mechanism. Interpreting it requires loadings (which features define it),
variance explained per layer (how much of each layer it captures), and external association
(what it correlates with).

A factor that explains 18% of RNA variance, 12% of protein variance, and 4% of methylation
variance is a transcriptome-led factor with a protein echo. Calling it "a shared regulatory
program" is an interpretation that needs evidence beyond the variance numbers.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Layer / modality / view | One assay's matrix | Used interchangeably without definition | Matrix |
| Early fusion | Feature concatenation | Assumed to preserve layer structure | Method |
| Intermediate fusion | Shared latent representation | Assumed to be causal | Method |
| Late fusion | Combination of per-layer predictions | Assumed to capture interactions | Method |
| Latent factor | Direction explaining covariance | Called a pathway | Vector |
| Loading | Feature weight on a factor | Interpreted causally | Weight |
| Variance explained | Share of a layer's variance captured | Compared across layers without scaling | Proportion |
| CCA | Canonical correlation analysis | Assumed to find causal links | Method |
| PLS | Partial least squares | Confused with PCA | Method |
| MOFA | Multi-omics factor analysis | Treated as a classifier | Method |
| DIABLO | Multi-block discriminant analysis | Treated as a discovery method | Method |
| iCluster | Joint latent clustering | Assumed to find subtypes | Method |
| SNF | Similarity network fusion | Assumed to preserve all structure | Method |
| Stacking | Meta-model over base predictions | Leaks if fit on training folds | Method |
| Missing modality | A layer absent for some samples | Imputed without flagging | Design |
| Block weighting | Relative weight of a layer | Ignored in early fusion | Weight |
| Leakage | Information from validation entering training | Invisible in the metric | Error |
| Baseline | Best single-layer model | Omitted | Comparison |
| Nested cross-validation | Selection inside the training fold | Confused with plain CV | Procedure |
| AUC | Area under the ROC curve | Compared without intervals | 0.5-1.0 |
| Calibration | Agreement of predicted and observed risk | Confused with discrimination | Slope, intercept |
| Permutation null | Null from shuffling labels | Used without preserving structure | Procedure |
| Factor rotation | Reorientation of factors | Treated as a different model | Transform |
| Confounded factor | Factor aligned with batch or quality | Interpreted as biology | Error |
| Multi-block | Several matrices for the same samples | Confused with multi-omics | Design |
| Integration gain | Improvement over best single layer | Reported without interval | Delta AUC |

Two definitions are load-bearing.

**Integration gain is a difference, and differences need intervals.** An AUC of 0.74 versus
0.68 is a gain of 0.06, but with 200 samples the confidence interval on each AUC is roughly
±0.07, so the gain is not distinguishable from zero. Reporting the point difference without an
interval is the most common overclaim in the integration literature.

**A missing modality is a design fact, not a data-cleaning problem.** If proteomics was run on
only half the cohort, the integrated analysis has two options: restrict to the intersection, or
model the missingness. Imputing an entire layer for samples that never had it is fabrication
with a statistical veneer.

## 5. The mechanics, step by step

### Step 1. Write the integration hypothesis

**Input:** the program question.

**Output:** one sentence stating what the joint model estimates that single layers do not.

**Rationale:** "A shared latent factor across RNA and protein predicts response better than
either layer" is a testable hypothesis with a comparison built in. "Integrate the omics data" is
not. The hypothesis determines the strategy, the evaluation, and the interpretation. If it
cannot be written, the integration should not be done.

### Step 2. Verify sample alignment

**Input:** sample identifiers from each layer.

**Output:** an alignment table with the intersection and the reasons for exclusion.

**Rationale:** Sample identifiers differ across systems (barcode, donor ID, pathology number).
A join on a formatted string can silently drop samples or, worse, mis-join them. Verify with a
second key, such as sex or genotype, and report the intersection size and the characteristics of
excluded samples.

### Step 3. Preprocess each layer independently

**Input:** raw matrices.

**Output:** normalized, filtered matrices with per-layer QC.

**Rationale:** Each layer has its own normalization, transformation, and filtering requirements.
RNA counts need a variance-stabilizing transform; methylation beta values need an arc-sine or
logit transform; proteomics needs log transformation and missingness handling. Applying one
transformation across layers because it is convenient produces a matrix whose scales are
arbitrary and whose factors are driven by scale differences.

Critically, preprocessing must be fit inside cross-validation when the analysis is predictive.
Normalizing on the full dataset leaks information.

### Step 4. Choose the strategy from the hypothesis

**Input:** hypothesis, sample size, layer count.

**Output:** strategy with a stated rationale.

**Rationale:** A prediction hypothesis with many features and few samples favors a penalized
early-fusion model or late fusion. A structure hypothesis favors an intermediate latent-factor
model. A subtype hypothesis favors joint clustering. The choice should follow from what is being
estimated, not from what software is available.

### Step 5. Handle missing modalities explicitly

**Input:** alignment table with missingness.

**Output:** an analysis population and a missingness plan.

**Rationale:** Three options are defensible: restrict to complete cases and report the
characteristics of excluded samples; model the missingness with a latent indicator; or use a
method that handles missing views. Imputing a whole layer is not defensible unless the
imputation is validated, which requires samples with the layer measured.

### Step 6. Fit the model with the tuning inside the folds

**Input:** preprocessed matrices.

**Output:** fitted model and out-of-fold predictions.

**Rationale:** Every data-dependent choice — feature selection, factor number, penalty strength,
layer weighting — must be made inside the training fold. Otherwise the out-of-fold estimate is
optimistic, and the optimism grows with the number of choices and the number of features.
Integration multiplies both.

### Step 7. Compare against the best single-layer baseline

**Input:** out-of-fold predictions from integrated and single-layer models.

**Output:** performance with intervals and a paired comparison.

**Rationale:** The baseline is the strongest single layer, not the average or the weakest. A
gain over the average is not evidence of integration value. Use a paired comparison on the same
samples (for example, a bootstrap of the AUC difference) and report the interval. If the
interval includes zero, the integration has not demonstrated a gain.

### Step 8. Interpret factors with three pieces of evidence

**Input:** fitted latent model.

**Output:** factor interpretation with loadings, variance, and association.

**Rationale:** Loadings say which features define the factor; variance explained says how much
of each layer it captures; external association says what it correlates with. A factor with
strong loadings but no external association is a description, not a finding. A factor with a
strong association but loadings dominated by a known technical gradient is an artifact.

### Step 9. Test association at the factor level

**Input:** factor scores and outcome.

**Output:** effect size per factor with an interval and a correction.

**Rationale:** With five factors, five tests are performed, and the correction is over five, not
over the thousands of features. The effect size should be reported in interpretable units, such
as odds ratio per standard deviation of the factor score.

### Step 10. Correct for the tests actually performed

**Input:** all tested associations.

**Output:** corrected significance with the family stated.

**Rationale:** The family includes every factor-outcome association tested, and every model
compared. Selecting the best factor and then testing it is a selection, and the p-value must
account for it, usually by permuting the outcome and repeating the whole selection procedure.

### Step 11. Validate externally or temporally

**Input:** an independent cohort, or a held-out time period.

**Output:** performance in the validation set.

**Rationale:** Cross-validation estimates transport to the same population and pipeline. An
external cohort tests transport across sites, ancestries, and platform versions. For a
multi-omics classifier, the external cohort often lacks one layer, which is itself a finding
about deployability.

### Step 12. Report the negative result when there is one

**Input:** the comparison.

**Output:** a statement that integration did not improve performance, with the interval.

**Rationale:** A well-conducted integration that finds no gain is a useful result, because it
saves the program from building on a more complex model. Reporting only the positive results
distorts the literature and leads other programs to repeat the work.

### Step 13. Assess factor stability

**Input:** fitted model and repeated fits under resampling or different seeds.

**Output:** stability statistics per factor.

**Rationale:** Latent-factor models have rotational and initialization ambiguity. Two fits on the
same data can produce the same variance explained with different loadings. A factor that is not
stable across bootstrap samples cannot be interpreted, because its loadings are not a property of
the data. The practical test is to refit on bootstrap samples, match factors by loading
correlation, and report the proportion of resamples in which a factor with correlated loadings
appears.

A factor stable in 95% of resamples is a candidate for interpretation. A factor stable in 40% is
a fitting artifact, and the honest report says so.

### Step 14. Check deployability

**Input:** the intended use and the layers available at that point.

**Output:** a deployment specification and a model restricted to available layers.

**Rationale:** A model that requires three layers cannot be deployed where only one is measured.
The check is not technical but procedural: write down which layers are available at the moment
the decision is made, and re-run the comparison restricted to those. If the restricted model
does not beat the single-layer baseline, the integration has produced a research result, not a
product.

### Step 15. Replicate

**Input:** an independent cohort or a held-out site.

**Output:** performance in the replication set.

**Rationale:** Cross-validation is internal. Replication tests transport across sites,
ancestries, and platform versions. For multi-omics models, replication often fails for a
structural reason: one layer was measured on a different platform, or the cohort's case mix
differs. Discovering that in replication is better than discovering it at launch.

### Sample size arithmetic

Multi-omics models have many parameters and often few samples. Three rules of thumb help
before fitting.

**Events per variable.** For a logistic model, a common guideline is at least 10 outcome events
per estimated parameter. A model with 50 selected features and 20 factors has far more
parameters than 10 events per parameter allows at n = 200.

**AUC precision.** The standard error of an AUC is roughly sqrt(A(1-A)(1 + n_pos/2 + n_neg/2) /
(n_pos n_neg)) in the Hanley-McNeil approximation. With 100 cases and 100 controls and AUC 0.7,
the standard error is about 0.04, so a 95% interval spans roughly 0.08. A gain smaller than that
cannot be demonstrated.

**Effective number of tests.** Correlated features reduce the effective number of independent
tests, which is why permutation-based correction is often more powerful than Bonferroni for
omics data. It does not reduce the number of parameters in the model, which is a separate
problem.

### Cross-validation schemes and what they test

| Scheme | What it estimates | When to use | Failure |
|---|---|---|---|
| Random k-fold | Performance on a random held-out sample | Large, homogeneous cohort | Leakage across related samples |
| Stratified k-fold | Performance preserving class balance | Imbalanced outcome | Still leaks across donors |
| Grouped k-fold | Performance on unseen donors or sites | Repeated samples per donor | Few groups |
| Nested CV | Performance with tuning included | Any tuned model | Computationally expensive |
| Leave-one-site-out | Transport across sites | Multi-site cohort | Site confounded with case mix |
| Temporal split | Transport across time | Registry or EHR data | Distribution drift |

The scheme must match the deployment. A model deployed to new patients needs grouped or
site-based validation, not random folds over samples that share donors.

### Step 16. Record the negative result

**Input:** the comparison outcome.

**Output:** a documented statement, positive or negative.

**Rationale:** A negative integration result is a finding about the data and the design, and it
should be recorded with the same care as a positive one. The record prevents a later team from
repeating the analysis with the same data, and it documents that the program considered
integration and rejected it for a stated reason.

### Branch points

**Branch A — one layer is missing for half the cohort.** Restrict to complete cases and report
the selection, or model the missingness. Do not impute an entire layer.

**Branch B — one layer has 100 times more features.** Early fusion will be dominated by that
layer. Use block scaling, a group penalty, or late fusion.

**Branch C — the factor correlates with a technical variable.** The factor is a batch or quality
gradient. Remove the technical variable from the layer before fitting, or include it as a
covariate, and re-fit.

### Decision table: hypothesis to strategy

| Hypothesis | Strategy | Evaluation | Risk |
|---|---|---|---|
| Joint model predicts better | Early fusion or late fusion | Paired AUC difference with interval | Leakage, overfitting |
| Shared latent structure exists | Intermediate (MOFA, CCA) | Variance explained, factor stability | Uninterpretable factors |
| Molecular subtypes are shared | Joint clustering (iCluster, SNF) | Cluster stability, external validation | Cluster number ambiguity |
| One layer mediates another | Network or causal model | Conditional independence tests | Confounding |
| A layer adds incremental value | Late fusion with and without | Nested model comparison | Correlated layers |
| Missing modality at deployment | Model with missing-view handling | Performance by view availability | Silent imputation |

### Edge cases

**Sample mismatch.** Diagnostic: intersection smaller than either layer, and excluded samples
differ by group. Consequence: a selected population. Mitigation: report the intersection and the
selection.

**Layer-specific batch.** Diagnostic: each layer has its own batch structure, and the joint
factor aligns with one of them. Consequence: a factor that is a processing artifact.
Mitigation: batch-correct within layer before integration, and check factors against batch.

**Scale dominance.** Diagnostic: the first factor's loadings are almost entirely from the
largest layer. Consequence: integration is that layer plus noise. Mitigation: block scaling,
group penalties, or late fusion.

**Missing modality.** Diagnostic: a layer exists for a non-random subset. Consequence:
integration is estimated on a biased sample. Mitigation: restrict, model, or report.

**Leakage in cross-validation.** Diagnostic: performance drops sharply on external validation,
and the gap grows with feature count. Consequence: an over-optimistic estimate. Mitigation:
nested cross-validation with all steps inside the fold.

**Overfitting with small n.** Diagnostic: the number of parameters exceeds the sample size, and
the model fits perfectly in training. Consequence: no transport. Mitigation: penalization,
fewer factors, or a simpler strategy.

**Factor rotation ambiguity.** Diagnostic: two fits give different loadings for the same
variance. Consequence: unstable interpretation. Mitigation: report stability across
initializations; interpret only stable factors.

**Confounded factor.** Diagnostic: the factor correlates with age, sex, or a quality metric more
strongly than with the outcome. Consequence: a biological story about a technical variable.
Mitigation: include covariates, or regress them out and re-fit.

**Block weighting arbitrariness.** Diagnostic: results change when block weights change.
Consequence: a finding that is a tuning choice. Mitigation: report sensitivity to weights.

**Post hoc factor selection.** Diagnostic: the factor was chosen because it was associated.
Consequence: inflated significance. Mitigation: permutation of the full selection procedure.

**Bootstrap instability.** Diagnostic: loadings for the "same" factor differ across resamples.
Consequence: unreproducible interpretation. Mitigation: match factors by loading correlation and
report stability.

**Tuning on the test set.** Diagnostic: the number of factors was chosen to maximize the reported
performance. Consequence: an optimistic estimate. Mitigation: nested selection.

**Outcome leakage through preprocessing.** Diagnostic: a transformation used the outcome, such as
selecting features by t-test on all samples. Consequence: inflated performance even with
cross-validation. Mitigation: unsupervised preprocessing inside folds only.

**Site confounding.** Diagnostic: the cohort's sites differ by case mix. Consequence: site
effects read as biology. Mitigation: leave-one-site-out validation; site as a covariate.

### What the procedure produces

The output is an integration record: the hypothesis, the alignment, the per-layer preprocessing,
the strategy and its rationale, the missingness plan, the out-of-fold comparison against
baselines with intervals, the factor interpretation with three pieces of evidence, the
correction family, and the external validation. Without the comparison, the record does not
establish that integration was worth doing.

## 6. Worked example

A study has 200 patients with RNA-seq, proteomics, and DNA methylation from the same biopsy. The
hypothesis is that a shared latent factor across the three layers predicts response to therapy
better than any single layer.

### 6.1 Alignment and preprocessing

The RNA matrix has 200 samples, the protein matrix 186, and the methylation matrix 192. The
intersection is 178 samples.

```text
RNA only:        200 - 178 = 22 samples lost
protein only:    186 - 178 = 8
methylation only: 192 - 178 = 14
```

The 22 samples without protein data are not random: they are the earlier-recruited patients,
who also have more advanced disease. Restricting to 178 therefore selects a less advanced
population, and the restriction must be reported as a limitation.

### 6.2 The factor model

An intermediate integration fits five factors. Factor 1 explains:

```text
RNA:        18% of variance
protein:    12% of variance
methylation: 4% of variance
```

The factor is transcriptome-led. Its RNA loadings are dominated by interferon-response genes,
and its protein loadings include the corresponding proteins at lower weights. The methylation
loading is weak and diffuse, which is consistent with a transcriptional program rather than an
epigenetic one.

### 6.3 Association arithmetic

Factor 1 is associated with response. A logistic model gives an odds ratio of 2.10 per standard
deviation of the factor score, with a 95% confidence interval:

```text
log OR = 0.74, SE = 0.19
95% CI = 0.74 +/- 1.96 x 0.19 = [0.37, 1.11]
OR CI = [exp(0.37), exp(1.11)] = [1.44, 3.04]
```

The p-value from a Wald test is 9.8 x 10^-5. With five factors tested, the Bonferroni threshold
is:

```text
0.05 / 5 = 0.01
```

so the association survives correction. The reporting sentence: "Factor 1 was associated with
response (OR 2.10 per SD, 95% CI 1.44-3.04, Wald p = 9.8 x 10^-5, Bonferroni-corrected across
5 factors)."

### 6.4 The comparison that decides the question

The integrated model's out-of-fold AUC is 0.74. The single-layer baselines are:

| Model | AUC | 95% bootstrap CI |
|---|---:|---|
| RNA only | 0.68 | 0.60-0.76 |
| Protein only | 0.66 | 0.58-0.74 |
| Methylation only | 0.59 | 0.51-0.67 |
| Integrated | 0.74 | 0.66-0.82 |

The integrated model is 0.06 AUC above the best single layer. The intervals overlap
substantially. A paired bootstrap of the difference gives a 95% interval of -0.01 to 0.13,
which includes zero.

The honest conclusion is: "the integrated model is not demonstrated to outperform RNA alone in
this cohort; the point estimate favors integration but the interval includes no difference."
That is a negative result, and it is the correct one.

### 6.5 Diagnostic 1: the leak

The first analysis reported an integrated AUC of 0.86. The difference from 0.74 came from
feature selection: the top 500 features per layer were chosen using all 200 samples, before
cross-validation. Re-running with selection inside the folds removes the gain.

The magnitude of the leak is the point:

```text
AUC with leaky selection:  0.86
AUC with nested selection: 0.74
inflation:                 0.12 AUC
```

An inflation of 0.12 is enough to turn a null result into a publishable one. The diagnostic is
to check whether any step touched the full dataset before the split, and the number of such
steps is usually larger than the analyst expects.

### 6.6 Diagnostic 2: the batch factor

Factor 3 explains 9% of RNA variance and 7% of protein variance, and it is associated with
response at p = 0.003. Its loadings, however, are dominated by ribosomal genes and by proteins
that correlate with sample storage time.

A check shows factor 3 correlates with storage time at r = 0.61, and storage time differs by
recruitment site. The factor is a quality gradient, not biology. Including storage time as a
covariate reduces the factor-outcome association to p = 0.21.

Without the check, factor 3 would have been reported as a second predictive program. The
diagnostic is to correlate every factor with every technical variable before interpreting it.

### 6.7 Vary one assumption: protein is unavailable at deployment

The integrated model requires three layers. Suppose the intended deployment setting has RNA
only, because proteomics is too slow and too expensive for clinical use.

The integration has therefore optimized a model that cannot be deployed. A RNA-only model
trained on the same 178 samples gives AUC 0.68, and a model trained on RNA plus a protein-
derived score gives 0.71. The deployable gain is 0.03 AUC, with an interval that includes zero.

Changing the assumption from "all layers available" to "RNA available at deployment" changes the
question from "does integration improve prediction?" to "does integration produce a model that
can be used?" Those are different questions, and the second is the one that matters
commercially.

### 6.8 The precision of the comparison

The integrated and single-layer AUCs come from the same 178 patients, so the comparison must be
paired. An unpaired comparison of two AUCs with overlapping intervals is conservative and wastes
information; a paired bootstrap uses the correlation between the two models' predictions.

In this cohort, the two models rank patients similarly (their predictions correlate at r = 0.82), which also
means they would make similar decisions, so the practical value of the integrated model is smaller
than the AUC difference suggests,
so the paired difference is more precisely estimated than either AUC. The paired interval,
-0.01 to 0.13, is narrower than the difference of the individual intervals would suggest, and it
still includes zero. The conclusion is unchanged, but the precision statement is more honest.

The paired comparison also changes what counts as a failure. An integrated model that is 0.06
better but highly correlated with the baseline adds little clinical value even if the difference
were significant, because it would change few decisions. A model that is 0.02 better but
uncorrelated with the baseline might change more decisions, because it gets different patients
right. The comparison should therefore report both the difference and the correlation between
predictions, not only the AUCs.

A useful mental arithmetic: with 178 patients and roughly 89 events, the standard error of an
AUC near 0.70 is about 0.045. Two such models compared unpaired have a difference standard error
near 0.064, so a difference of 0.06 is one standard error. No design with this sample size can
establish a gain of that size, which is why the negative result is a statement about the study as
much as about the biology.

### 6.9 What the record says

> Hypothesis: a shared factor across RNA, protein, and methylation predicts response better than
> any single layer. 178 of 200 samples with all three layers; excluded samples are earlier and
> more advanced. Per-layer normalization with variance-stabilizing transforms; batch corrected
> within layer. Five factors from an intermediate model; factor 1 transcriptome-led (18/12/4%
> variance), OR 2.10 per SD (1.44-3.04), p = 9.8 x 10^-5, Bonferroni across 5. Out-of-fold AUC
> 0.74 (0.66-0.82) versus RNA 0.68 (0.60-0.76); paired difference interval includes zero. No
> demonstrated gain. Factor 3 identified as a storage-time gradient.

That record is the deliverable. The model is secondary.

## 7. Code, query, or calculation

The script computes a factor score from loadings, tests its association, bootstraps an AUC
difference, and applies BH.

### 7.1 Factor score and association

```python
import math

# Loadings for three features on factor 1, and one sample's standardized values
loadings = {"IFIT1": 0.62, "CXCL10": 0.55, "STAT1": 0.48}
sample = {"IFIT1": 1.8, "CXCL10": 1.2, "STAT1": 0.9}

score = sum(loadings[g] * sample[g] for g in loadings)
norm = math.sqrt(sum(v * v for v in loadings.values()))
print("raw score:", round(score, 3))
print("normalized score:", round(score / norm, 3))
print("loading norm:", round(norm, 3))
```

Expected output:

```text
raw score: 2.208
normalized score: 2.305
loading norm: 0.958
```

The normalization makes scores comparable across factors with different loading magnitudes. The
score is a projection, not a measurement, and its scale depends on the standardization of the
input features.

### 7.2 Wald test for an odds ratio

```python
import math

def wald_or(log_or, se):
    lo = log_or - 1.959963985 * se
    hi = log_or + 1.959963985 * se
    z = log_or / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return math.exp(log_or), math.exp(lo), math.exp(hi), p

or_, lo, hi, p = wald_or(0.74, 0.19)
print(f"OR {or_:.2f} 95% CI [{lo:.2f}, {hi:.2f}] p={p:.1e}")
print("Bonferroni threshold for 5 factors:", 0.05 / 5)
```

Expected output:

```text
OR 2.10 95% CI [1.44, 3.04] p=9.8e-05
Bonferroni threshold for 5 factors: 0.01
```

The simulated AUC difference is larger than the worked example because the toy data were
generated with a strong separation. The interval still includes zero, which is the point: even
a 0.14 point difference is not established at this sample size.

### 7.3 AUC and a paired bootstrap

```python
import random

def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / (len(pos) * len(neg))

random.seed(11)
n = 60
labels = [1] * 30 + [0] * 30
rna = [random.gauss(0.55 * y, 1.0) for y in labels]
integrated = [random.gauss(0.72 * y, 1.0) for y in labels]

print("RNA AUC:", round(auc(rna, labels), 3))
print("Integrated AUC:", round(auc(integrated, labels), 3))

diffs = []
idx = list(range(n))
for _ in range(2000):
    sample = [random.choice(idx) for _ in idx]
    s_r = [rna[i] for i in sample]
    s_i = [integrated[i] for i in sample]
    y = [labels[i] for i in sample]
    if len(set(y)) < 2:
        continue
    diffs.append(auc(s_i, y) - auc(s_r, y))
diffs.sort()
print("paired difference 95% CI:",
      round(diffs[int(0.025 * len(diffs))], 3),
      round(diffs[int(0.975 * len(diffs))], 3))
```

Expected output (seed 11):

```text
RNA AUC: 0.671
Integrated AUC: 0.808
paired difference 95% CI: -0.019 0.311
```

The point difference is 0.137, and the interval includes zero. The simulated data were generated
with a true difference, and the study is still too small to demonstrate it. That is the
realistic situation for most integration studies: the effect may be real and the evidence
insufficient.

### 7.4 BH across factors

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

pvals = [9.8e-5, 0.003, 0.21, 0.44, 0.71]
q = bh(pvals, 5)
for p, qq in zip(pvals, q):
    print(f"p={p:.2e} q={qq:.3e}")
```

Expected output:

```text
p=9.80e-05 q=4.900e-04
p=3.00e-03 q=7.500e-03
p=2.10e-01 q=3.500e-01
p=4.40e-01 q=5.500e-01
p=7.10e-01 q=7.100e-01
```

Two factors pass at q < 0.05. The second is the storage-time gradient from section 6.6, which
illustrates why a significant factor is not necessarily a biological one.

### 7.5 Assumptions and limits of this code

- The factor score is a simple projection; a real factor model estimates loadings jointly.
- The Wald test assumes a large sample and a correctly specified logistic model.
- The AUC function is the Mann-Whitney statistic and is slow for large cohorts.
- The bootstrap resamples patients, which is correct only if patients are independent; with
  repeated samples, resample donors.
- The BH function assumes independence or positive dependence across factors, which is
  questionable because factors are estimated jointly.

## 8. Data, standards, and regulatory context

### 8.1 There is no multi-omics standard

No ISO, CLSI, or FDA standard specifies how to integrate omics layers. The applicable
requirements are inherited from each layer and from the intended use:

- **MINSEQE/MIAME** for the sequencing and array layers.
- **MIAPE** for proteomics.
- **MSI** reporting levels for metabolomics.
- **CELLxGENE** and **HCA** schemas for single-cell layers.
- **ISO 15189:2022** and **CLIA** if the integrated result is reported clinically.
- **IVDR 2017/746** for a device placed on the EU market.

Each of these constrains one layer. None constrains the joint model, which is the gap a program
must fill with its own validation plan.

### 8.2 Model reporting standards

For a predictive model, **TRIPOD** (Transparent Reporting of a multivariable prediction model
for Individual Prognosis Or Diagnosis) provides a checklist: population, predictors, outcome,
sample size, missing data, model building, performance, and validation. TRIPOD+AI extends it to
machine-learning models. A multi-omics classifier should be reported against it, because the
checklist forces the comparison and the calibration that integration papers often omit.

### 8.3 Reproducibility

A joint model is reproducible only if the input matrices, the per-layer preprocessing, the
alignment table, the model version, and the random seed are recorded. Containerization and
workflow engines (Docker, Singularity, Nextflow, Snakemake) are the practical mechanisms.
Without them, a factor model cannot be re-fit, and a re-fit can produce different factors
because of initialization and rotation.

### 8.4 What a validation plan must contain

No standard prescribes the contents, but a defensible plan for a multi-omics model contains:

1. The intended use and the decision the model changes.
2. The layers, platforms, and versions used at deployment.
3. The locked model: preprocessing, features, weights, and threshold.
4. Analytical validation of each layer's measurement.
5. Discrimination and calibration in an independent cohort, with intervals.
6. Subgroup performance by age, sex, and ancestry where sample size permits.
7. The comparison with the best single-layer model.
8. A plan for monitoring drift and revalidation.

Items 3 and 7 are the ones most often missing. A model that is not locked cannot be validated,
because the validation tests a different model each time it is re-fit. A model without a baseline
comparison cannot demonstrate that its complexity was earned.

### 8.5 Documentation that makes a joint model auditable

Four artifacts make a joint model auditable, and each is cheap to produce at the time and
expensive to reconstruct later:

- The alignment table, listing every sample and which layers it has.
- The per-layer preprocessing record, with parameters and the data they were fit on.
- The model card: version, hyperparameters, random seed, and the folds used.
- The evaluation table, with every model compared and every interval.

A program that keeps these can answer a reviewer's question in an hour. A program that does not
will re-run the analysis to answer it, and the re-run will produce a slightly different number.

### 8.6 Regulatory pathway for a multi-omics biomarker

If an integrated model is intended for clinical use, the regulatory question is about the test,
not the integration. A device pathway requires analytical validation of each measurement, of the
combined algorithm, and of the clinical performance in the intended population. The FDA
Biomarker Qualification Program can qualify a biomarker for a context of use, but the
qualification attaches to the biomarker, not to a specific model.

Nothing in the current guidance says that a multi-omics model is treated differently from a
single-omics model. The validation burden is the same or higher, because there are more
components to validate and more ways for the model to fail at deployment.

## 9. Industry practice and case evidence

### 9.1 The TCGA integrative precedent

The TCGA Pan-Cancer Atlas integrated copy number, methylation, mRNA, miRNA, and protein across
11,286 tumours from 33 cancer types. The integrative clustering produced 28 molecular subtypes,
with copy-number variation contributing 47% of the clustering signal, transcriptome 42%, and
methylation 11%.

The result is often cited as evidence that integration works. A more careful reading is that it
shows integration is informative when the layers have strong, distinct structure and the sample
size is large. Copy number dominated because it carried the strongest signal, not because the
method was neutral across layers. A smaller study with weaker structure would not reproduce the
result.

### 9.2 The method landscape

MOFA and MOFA+ provide a Bayesian group factor analysis with automatic relevance determination,
which prunes irrelevant factors and handles missing views. The 2020 MOFA+ paper added
GPU-accelerated inference and group-wise priors, scaling to millions of cells.

DIABLO performs multi-block discriminant analysis for classification. iCluster performs joint
latent clustering for subtype discovery. Similarity network fusion builds a graph from
per-layer similarity matrices.

The documented pattern across benchmarks is that no method dominates on all tasks, and that
performance depends on the signal-to-noise ratio per layer and on the sample size. A method
choice justified only by popularity is not justified.

### 9.3 What pharma does

Documented practice:

- Target discovery integrates genetics, expression, and pathway evidence, usually in a
  prioritization framework rather than a single joint model (topic 09).
- Biomarker discovery uses multi-omics for hypothesis generation and single-layer assays for
  validation.
- Translational groups increasingly require single-omics baselines in internal review, because
  integrated models that do not beat the baseline are common.

Inference: the most common organizational failure is a multi-omics result reported to a program
committee without the baseline comparison, because the baseline was never run. The committee
then approves a validation budget for a model that has no demonstrated advantage.

### 9.4 The negative-result literature

The published record is biased toward positive integration results, but the negative results are
informative when they appear. The common pattern is a small cohort, many features, an integrated
model with a modest cross-validated gain, and no external validation. When an external cohort is
added, the gain disappears and the single-layer model performs as well or better.

This is not evidence that integration never works. It is evidence that the typical integration
study is underpowered for the gain it hopes to detect, and that the gain, when present, is
smaller than the study's precision. The practical implication for a program is to size the study
for the smallest gain that would change a decision, and to decide in advance that a smaller gain
will not be pursued.

A practical way to read the literature is to count how many integration papers report a paired
interval on the difference. Most report two AUCs and a p-value from an unpaired comparison, which
is the weakest form of evidence for a difference between models on the same samples. The absence
of the paired interval is a red flag about the analysis, not only about the reporting.

### 9.6 The benchmark evidence

Open Problems in Single-Cell Analysis runs community benchmarks on tasks including multi-modal
integration. The consistent finding is that simple baselines are strong, and that method ranking
depends on the task and the dataset. A program should run the baseline before adopting a complex
method, and should not assume that a method's published ranking transfers to its data.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No baseline | Integrated AUC reported alone | Missing comparison | Unknown value | Best single layer |
| Leaky selection | CV much better than external | Selection outside folds | Inflated performance | Nested CV |
| Sample misalignment | Intersection smaller than expected | ID mismatch | Wrong samples joined | Two-key verification |
| Scale dominance | Factor loadings from one layer | Feature count imbalance | Integration is one layer | Block scaling |
| Whole-layer imputation | Layer filled for absent samples | Fabricated data | Biased model | Restrict or model |
| Confounded factor | Factor correlates with batch | Technical gradient | False biology | Covariate check |
| Post hoc factor choice | Best factor tested | Selection | Inflated p-value | Permute selection |
| Overfitting | Parameters exceed samples | Small n | No transport | Penalization |
| Unstable factors | Loadings differ across seeds | Rotation, initialization | Unreproducible interpretation | Stability analysis |
| Ignoring missing modality at deployment | Model needs unavailable layer | Design mismatch | Undeployable | Deployment-aware design |
| Arbitrary block weights | Results change with weights | Tuning choice | Fragile finding | Sensitivity analysis |
| No interval on gain | Point AUC difference only | Overclaim | False superiority | Bootstrap interval |
| Correlated layers | RNA and protein nearly redundant | Biology | No gain possible | Check correlation |
| Wrong correction family | Corrected across features not factors | Family mismatch | Wrong significance | State family |
| Tuning on test data | Factor number chosen for performance | Selection | Optimistic estimate | Nested selection |
| Outcome-aware preprocessing | Features selected by outcome | Leakage | Inflated performance | Unsupervised inside folds |

Four red flags:

**If there is no single-layer baseline, stop.** The integration has not been evaluated.

**If cross-validation performance collapses externally, suspect leakage.** The gap grows with
feature count.

**If a factor correlates with a technical variable more than with the outcome, stop.** It is a
quality gradient.

**If the model requires a layer that deployment will not have, stop.** It is a research model,
not a product.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Samples are aligned across layers | Joint modelling | ID mismatch | Two-key verification |
| Layers share biological structure | Integration premise | Independent processes | Cross-layer correlation |
| Per-layer normalization is correct | Commensurability | Different distributions | Per-layer QC |
| Missingness is ignorable or modeled | Complete-case analysis | Non-random missing layer | Compare included/excluded |
| Factors are stable | Interpretation | Rotation, initialization | Multiple seeds |
| Factors are biological | Claims | Technical gradients | Covariate correlation |
| Cross-validation is leak-free | Performance estimate | Selection outside folds | Nested CV audit |
| Gain is estimable | Comparison | Small n, wide intervals | Paired bootstrap |
| Model is deployable | Clinical use | Missing layer at deployment | Deployment simulation |
| Validation cohort is comparable | Transport | Site, ancestry, platform | External validation |
| Factors are stable | Interpretation | Initialization, rotation | Bootstrap matching |
| The gain is large enough to matter | Decision relevance | Small effect, wide interval | Pre-specified minimum gain |
| Deployment has all layers | Product viability | Missing layer at use | Deployment simulation |

The assumption that layers share structure is the one that must be tested first, because it is
cheap to test: correlate the strongest signal from each layer. If the correlation is near zero,
the integration is combining unrelated measurements, and the joint model will find noise.

Before the exercises, a note on the standard they enforce. Every exercise asks for a comparison
or an interval, because the central skill in this topic is not fitting a joint model but deciding
whether the joint model earned its complexity. A learner who can fit MOFA and cannot answer that
question has learned the easy half.

## 12. Exercises

**Exercise 1 — Integration hypothesis (design, 25 minutes).**
Write an integration hypothesis for a study with RNA and protein data, specifying what the joint
model estimates that neither layer does, and the comparison that would test it.
*Deliverable:* one paragraph with hypothesis and comparison.
*Criteria:* hypothesis falsifiable; comparison names the baseline; no vague "better
understanding."
*Time:* 25 minutes.

**Exercise 2 — Strategy selection (application, 30 minutes).**
For each goal, choose a strategy and state its main risk: (a) predict response; (b) find shared
subtypes; (c) test whether protein adds to RNA.
*Deliverable:* three-row table.
*Criteria:* strategy matched to goal; risk specific.
*Time:* 30 minutes.

**Exercise 3 — Leakage audit (critique, 35 minutes).**
A pipeline selects the top 1,000 features per layer, normalizes with all samples, fits a model,
and reports 5-fold CV AUC. List every leak and its direction of bias.
*Deliverable:* a leak list with bias direction.
*Criteria:* at least three leaks; each with a mechanism; a fix per leak.
*Time:* 35 minutes.

**Exercise 4 — AUC comparison (calculation, 30 minutes).**
Integrated AUC 0.74 (0.66-0.82), RNA AUC 0.68 (0.60-0.76). Compute the point difference, state
whether the intervals support superiority, and write the reporting sentence.
*Deliverable:* arithmetic and one sentence.
*Criteria:* difference correct; interpretation correct; sentence includes intervals.
*Time:* 30 minutes.

**Exercise 5 — Factor interpretation (application, 40 minutes).**
A factor explains 20% RNA, 3% protein, and 1% methylation variance, and correlates with storage
time at r = 0.58. Bootstrap refits correlate with the original loadings at 0.88 in 72% of
resamples. Write the interpretation, the stability judgment, and the next step.
*Deliverable:* 250 words.
*Criteria:* technical explanation considered first; a test proposed; no biological story without
evidence.
*Time:* 40 minutes.

**Exercise 6 — Missing modality (design, 35 minutes).**
Protein data exist for 60% of a cohort, and the missing samples are the more severe cases.
Propose two defensible analyses and state the limitation of each.
*Deliverable:* two analysis plans.
*Criteria:* both avoid whole-layer imputation; limitations specific.
*Time:* 35 minutes.

**Exercise 7 — Baseline design (design, 30 minutes).**
Design the baseline comparison for a three-layer classifier, including which baseline, which
metric, and which interval.
*Deliverable:* a comparison protocol.
*Criteria:* baseline is the best single layer; paired comparison; interval method named.
*Time:* 30 minutes.

**Exercise 8 — Report a negative result (application, 40 minutes).**
Write the results paragraph for an integration that did not beat the best single layer, without
overstating or apologizing.
*Deliverable:* one paragraph, 200 words.
*Criteria:* point estimate and interval reported; conclusion clear; no spin.
*Time:* 40 minutes.

## 13. Self-check answers

**Answer 1 (hypothesis).** "A shared latent factor across RNA and protein predicts response with
an AUC at least 0.05 higher than the best single layer, evaluated by out-of-fold predictions
with a paired bootstrap interval." The comparison is explicit, the metric is defined, and the
claim is falsifiable. A hypothesis without a comparison is a description of an activity, not a
testable statement.

**Answer 2 (strategy).** (a) Prediction: early fusion with penalization or late fusion; risk is
overfitting and leakage. (b) Shared subtypes: joint clustering such as iCluster or similarity
network fusion; risk is an arbitrary cluster number and unstable labels. (c) Incremental value
of protein: late fusion with and without the protein block, compared by a paired interval; risk
is that RNA and protein are correlated and the added value is small.

**Answer 3 (leakage).** Feature selection on all samples leaks outcome-correlated features into
training, biasing performance upward. Normalization on all samples leaks the distribution of the
validation set, biasing upward, usually less. Choosing the number of factors on the full data
leaks, biasing upward. Reporting the best of several models without correction leaks, biasing
upward. Every leak in this list biases the estimate in the same direction, which is why the
error is systematically optimistic.

**Answer 4 (AUC).** The difference is 0.06. The intervals overlap substantially, so the
comparison does not establish superiority; a paired interval is needed. The standard error of
an AUC near 0.70 with about 90 events is roughly 0.045, so a difference of 0.06 is about one
standard error and cannot be distinguished from zero at this sample size. Reporting sentence:
"the integrated model achieved an AUC of 0.74 (95% CI 0.66-0.82) versus 0.68 (0.60-0.76) for
RNA alone; the paired difference was 0.06 with a 95% interval that included zero."

**Answer 5 (factor).** The factor is transcriptome-led and weakly represented in protein and
methylation. Its correlation with storage time (r = 0.58) is stronger than most biological
associations, so the technical explanation should be tested first: include storage time as a
covariate, re-fit, and check whether the factor's association with the outcome survives. If it
does not, the factor is a quality gradient and should not be interpreted biologically.

**Answer 6 (missing modality).** Analysis one: restrict to the 60% with all layers, and report
that the excluded 40% are more severe, so the result applies to a less severe population.
Analysis two: model the missingness with an indicator and use a method that handles missing
views, reporting the assumption that missingness is related to observed severity. Neither
imputes the protein layer for samples that never had it.

**Answer 7 (baseline).** The baseline is the best single layer under the same cross-validation
scheme, not the average. It must use the same folds, preprocessing discipline, and tuning budget;
otherwise the comparison confounds model class with effort. A baseline that received less tuning
is a straw man, and a baseline that received more argues against integration for the wrong
reason. The metric is AUC with a paired bootstrap interval on the difference,
computed on the same out-of-fold predictions. If the outcome is time-to-event, use the
concordance index with the same paired approach. Report calibration as well, because a
discrimination gain without calibration can still be clinically useless.

**Answer 8 (negative result).** "The integrated model achieved an out-of-fold AUC of 0.74 (95%
CI 0.66-0.82), compared with 0.68 (0.60-0.76) for RNA alone. The paired difference was 0.06
(95% CI -0.01 to 0.13), which does not exclude no difference. We therefore did not find evidence
that integration improves prediction in this cohort, and we report the single-layer model as the
primary result." No apology, no spin, and the interval does the work.

The questions below are ordered from definition to judgment. A candidate who can define the
strategies but cannot say when integration fails has read the literature without doing the work.

## 14. Interview questions

**1. When does multi-omics integration help?**
Strong answer: when layers carry distinct, correlated structure and the joint model estimates
something a single layer cannot; requires a baseline comparison.
Weak answer: "more data is always better."

**2. What is the difference between early and late fusion?**
Strong answer: early concatenates features and can capture interactions but suffers from scale
and dimensionality; late combines predictions, is robust, and misses interactions.
Weak answer: "they are both integration."

**3. How do you evaluate an integrated model?**
Strong answer: out-of-fold predictions, best single-layer baseline, paired interval on the
difference, external validation, calibration.
Weak answer: "cross-validation AUC."

**4. What is leakage and how does integration amplify it?**
Strong answer: information from validation entering training; integration adds features and
choices, so leaks inflate more.
Weak answer: "using the test set."

**5. What does a latent factor mean?**
Strong answer: a direction explaining covariance; interpretation needs loadings, variance per
layer, and external association; it is not a pathway.
Weak answer: "a biological program."

**6. How do you handle a missing modality?**
Strong answer: restrict and report, or model the missingness; never impute a whole layer.
Weak answer: "impute with the mean."

**7. How do you correct for testing five factors?**
Strong answer: the family is the five factor-outcome tests; if the best factor was selected,
permute the selection procedure.
Weak answer: "Bonferroni across genes."

**8. What would make you distrust an integrated AUC of 0.86?**
Strong answer: no baseline, selection outside folds, small n, no external validation, and a
large gap to external performance.
Weak answer: "0.86 is good."

**9. What is the difference between a factor and a cluster?**
Strong answer: a factor is a continuous direction; a cluster is a discrete partition; both
require stability evidence.
Weak answer: treats them as the same.

**10. When is integration not worth doing?**
Strong answer: when layers are redundant, when the sample size cannot support the parameters,
or when deployment will not have all layers.
Weak answer: "never, integration is always better."

**11. How would you report a negative integration result?**
Strong answer: point estimates, intervals, the baseline, and a clear conclusion that the gain
was not demonstrated.
Weak answer: buries it or spins it.

**12. How do you decide the number of factors?**
Strong answer: by stability and by out-of-fold performance, not by variance explained alone; a
factor that is unstable or adds nothing out of fold is not retained.
Weak answer: "the scree plot."

**13. What is the difference between internal and external validation here?**
Strong answer: internal cross-validation estimates performance in the same population and
pipeline; external validation tests transport across sites, platforms, and case mix.
Weak answer: treats them as interchangeable.

**14. What does TRIPOD require for a multi-omics model?**
Strong answer: population, predictors, outcome, sample size, missing data, model building,
performance, and validation, with calibration.
Weak answer: "a checklist for papers."

**15. What would you do if the integrated model needed a layer unavailable at deployment?**
Strong answer: re-run the comparison restricted to deployable layers, report the deployable
gain, and treat the full model as a research result.
Weak answer: "deploy it anyway and collect the missing layer later."

## 15. Key terms

- **AUC** — area under the receiver operating characteristic curve.
- **Baseline** — the best single-layer comparator.
- **Block weighting** — relative weight assigned to a layer.
- **Calibration** — agreement between predicted and observed risk.
- **CCA** — canonical correlation analysis.
- **Confounded factor** — factor aligned with a technical variable.
- **DIABLO** — multi-block discriminant analysis.
- **Early fusion** — feature concatenation.
- **Factor rotation** — reorientation of latent factors.
- **iCluster** — joint latent clustering method.
- **Integration gain** — improvement over the best single layer.
- **Intermediate fusion** — shared latent representation.
- **Late fusion** — combination of per-layer predictions.
- **Latent factor** — direction explaining covariance across layers.
- **Layer** — one assay's matrix.
- **Leakage** — validation information entering training.
- **Loading** — feature weight on a factor.
- **Missing modality** — a layer absent for some samples.
- **MOFA** — multi-omics factor analysis.
- **Multi-block** — several matrices for the same samples.
- **Nested cross-validation** — selection performed inside training folds.
- **Permutation null** — null distribution from shuffling labels.
- **PLS** — partial least squares.
- **SNF** — similarity network fusion.
- **Stacking** — meta-model over base predictions.
- **TRIPOD** — reporting guideline for prediction models.
- **Variance explained** — share of a layer's variance captured by a factor.
- **View** — synonym for layer or modality, used in factor models.

## 16. Sources

Methods:

- Argelaguet, R. et al., "MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data," *Genome Biology* 21:111, 2020. https://doi.org/10.1186/s13059-020-02015-1 (accessed 25 October 2026).
- Argelaguet, R. et al., "Multi-Omics Factor Analysis — a framework for unsupervised integration of multi-omics data sets," *Molecular Systems Biology* 14:e8124, 2018. https://doi.org/10.15252/msb.20178124 (accessed 25 October 2026).
- Singh, A. et al., "DIABLO: an integrative approach for identifying key molecular drivers from multi-omics assays," *Bioinformatics* 35:3055-3062, 2019. https://doi.org/10.1093/bioinformatics/bty1054 (accessed 25 October 2026).
- Wang, B. et al., "Similarity network fusion for aggregating data types on a genomic scale," *Nature Methods* 11:333-337, 2014. https://doi.org/10.1038/nmeth.2810 (accessed 25 October 2026).

Evidence and reporting:

- Hoadley, K. A. et al., "Cell-of-Origin Patterns Dominate the Molecular Classification of 10,000 Tumors from 33 Types of Cancer," *Cell* 173:291-304, 2018. https://doi.org/10.1016/j.cell.2018.03.022 (accessed 25 October 2026).
- Collins, G. S. et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods," *BMJ* 385:e078378, 2024. https://doi.org/10.1136/bmj-2023-078378 (accessed 25 October 2026).
- Open Problems in Single-Cell Analysis. https://openproblems.bio/ (accessed 25 October 2026).

Standards:

- FGED Society, MINSEQE. https://www.fged.org/projects/minseqe (accessed 25 October 2026).
- HUPO Proteomics Standards Initiative. https://www.psidev.info/ (accessed 25 October 2026).
- ISO 15189:2022. https://www.iso.org/standard/76677.html (accessed 25 October 2026).
- Regulation (EU) 2017/746. https://eur-lex.europa.eu/eli/reg/2017/746/oj (accessed 25 October 2026).

The survey file's Lab 12.6 is the starting point for the practical work; the comparison against
single-omics baselines is the part learners most often skip, and it is the part this topic
weights most heavily.

The module index lists this topic between proteomics and biomarker discovery, which is the order
in which a program usually encounters it: measurements first, then the question of whether
combining them adds anything.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`06_proteomics-and-metabolomics.md`](06_proteomics-and-metabolomics.md)
- Next topic: [`08_biomarker-discovery.md`](08_biomarker-discovery.md)
- Related topic: [`04_single-cell.md`](04_single-cell.md)
- Survey file: [`../../12_omics_precision_medicine.md`](../../12_omics_precision_medicine.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 12.6, multi-omics integration, in the survey file: [`../../12_omics_precision_medicine.md`](../../12_omics_precision_medicine.md)
