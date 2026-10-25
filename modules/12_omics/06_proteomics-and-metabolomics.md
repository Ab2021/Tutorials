# Proteomics and Metabolomics: Mass Spectrometry, Missingness, and Identification

> Module 12 — Omics and Precision Medicine
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: 01A, 04, topics 01 and 03
> Version 2026.3

## 1. Why this topic exists

Proteins are the molecules most drugs act on, and metabolites are the molecules closest to
phenotype. Yet both are measured with assays that are less complete, less precise, and less
standardized than DNA sequencing, and the gap between a mass spectrometry result and a clinical
measurement is wider than most data scientists expect.

Three properties make these data hard. First, the measurement is indirect: a mass spectrometer
does not see a protein, it sees peptides with a mass-to-charge ratio and a retention time, and
the assignment of those features to proteins is an inference from a database search. Second,
the data are incomplete by construction: a protein present at low abundance in some samples is
missing in others, and the missingness is usually related to abundance, which means it is not
ignorable. Third, identification confidence is graded: a metabolite can be "identified" at four
different levels of evidence, and a level-2 annotation is not the same claim as a level-1
identification against an authentic standard.

The industry consequence is that proteomic and metabolomic biomarkers fail translation at a
higher rate than genomic ones, and the failures are often predictable from the data. A plasma
protein that is "significant" in a discovery cohort of 60 samples, with 30% missing values and
a batch-correlated abundance, is not a biomarker. It is a candidate that needs a targeted assay.
Programs that treat the discovery table as a result spend years on validation without
understanding why the effect shrinks.

This topic covers mass-spectrometry proteomics and metabolomics, the missingness mechanisms that
dominate both, identification confidence, normalization, and the path from a discovery table to
a validated assay. It builds on the study-design material in topic 03 and feeds biomarker
discovery in topic 08.

### 1.1 The failure pattern

1. A plasma proteomics discovery run quantifies 1,200 proteins across 60 samples.
2. Twenty percent of values are missing, and the analyst imputes them with a minimum value.
3. Differential expression is run on the imputed matrix with a t-test.
4. A protein with a 1.5-fold difference and p = 4 x 10^-4 is declared a candidate.
5. The multiple-testing correction across 1,200 proteins is omitted, or applied only to the
   shortlist after selection.
6. A targeted assay is developed and the effect is 1.1-fold, not significant.
7. The program concludes that the discovery platform was unreliable.

The platform was not the problem. The imputation created the effect, the missingness was
informative, and the correction was applied after selection. All three are diagnosable before
the validation assay is built.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Biomarker lead | Two years and a validation budget | Depends on the discovery analysis |
| Mass spectrometrist | Trust in a platform that was misused | Receives a matrix, not the question |
| Assay development team | A target that does not translate | Inherited candidate without QC |
| Clinician | A test with no measurable range | Sees the validated assay, not the discovery |
| Payer | Spend on a test with no utility | Sees the claim |
| Regulator | A submission with uninterpretable analytical validity | Receives summarized discovery data |

### 1.3 The industry question

> Is this molecular signal a validated biomarker, a causal target, or an artifact of batch,
> ancestry, and tissue composition?

Proteomics adds two artifact classes to the list: identification error (the peptide was assigned
to the wrong protein) and missingness (the value is absent because of abundance, not because it
is zero). Metabolomics adds a third: annotation error (the feature is not the metabolite it is
named after).

## 2. Learning outcomes

By the end of this topic you can:

1. **Trace** a peptide-spectrum match from raw spectra to protein quantification, naming the
   inference at each step.
2. **Distinguish** data-dependent, data-independent, and targeted acquisition, and state what
   each can and cannot quantify.
3. **Diagnose** the missingness mechanism and choose an analysis that does not invent signal.
4. **Apply** peptide- and protein-level false discovery control and explain the difference.
5. **Normalize** proteomic and metabolomic data with a stated assumption, and test it.
6. **Assign** metabolite identification confidence and state what each level supports.
7. **Compute** a differential result with correction across the correct family.
8. **Design** a discovery-to-validation path with an orthogonal, calibrated assay.

## 3. The core mental model

A mass spectrometer measures ions. Everything above the ion is inference.

```text
  Sample (plasma, tissue, cells)
      |
      +--> lysis, depletion, digestion (proteins -> peptides)
      +--> extraction, derivatization (metabolites)
      v
  LC separation -> ionization -> MS1 (precursor m/z) -> MS2 (fragments)
      |
      +--> DDA: fragment the top N precursors per cycle
      +--> DIA: fragment all precursors in windows
      +--> PRM/MRM: fragment pre-specified precursors
      v
  Spectra
      |
      +--> identification: database search or spectral library
      |         |
      |         +--> peptide-spectrum match score
      |         +--> target-decoy FDR
      |         +--> protein inference (shared peptides)
      |         +--> PTM localization confidence
      v
  Quantification
      |
      +--> label-free: precursor intensity, LFQ/MaxLFQ
      +--> isobaric: reporter ion ratios (TMT, iTRAQ)
      +--> targeted: calibration curve, absolute concentration
      v
  Protein x sample matrix
      |
      +--> missingness diagnosis -> filtering or model
      +--> normalization -> batch adjustment
      +--> statistical test -> correction -> effect size
      v
  Claim
```

### 3.1 Identification is inference, not observation

A peptide-spectrum match is a hypothesis that a spectrum came from a peptide sequence. The
search engine scores it, and the target-decoy procedure estimates the false discovery rate of
the set. A 1% peptide FDR means that about 1% of the accepted matches are expected to be wrong;
it does not mean that any particular match is 99% likely to be correct.

Protein inference adds another layer. Peptides shared between isoforms or homologous proteins
cannot assign the protein uniquely. A "protein" in the output is often a protein group, and the
group can change when the database changes.

### 3.2 Missingness is a mechanism, not a nuisance

Three mechanisms are usually distinguished:

| Mechanism | Meaning | Example | Consequence of naive imputation |
|---|---|---|---|
| MCAR | Missing independent of value | Random injection failure | Little bias, lost power |
| MAR | Missing depends on observed values | Lower abundance in a batch | Bias if the model ignores the predictor |
| MNAR | Missing depends on the missing value itself | Below limit of detection | Strong bias toward false effects |

In label-free proteomics, most missingness is MNAR, because low-abundance peptides fall below
detection. Imputing a minimum value asserts that the protein was near zero, which is often
false: it may have been present at a level just below detection. Imputing a mean asserts that
the missing values resemble the observed ones, which is also often false. Both choices change
the differential result, and the direction of the change depends on which group has more
missing values.

A 2023 analysis argued that detection probability is neither purely random nor purely censored,
but a logit-linear function of the underlying intensity. That framing matters because it implies
that the right analysis models detection jointly with abundance, rather than imputing and then
testing.

### 3.3 Two quantification families

**Label-free** quantification uses precursor intensity across runs. It requires consistent
chromatography and normalization, and it is sensitive to missingness. MaxLFQ-style approaches
use pairwise intensity ratios between runs to build a consistent protein intensity, which
reduces the effect of missing peptides.

**Isobaric labelling** (TMT, iTRAQ) multiplexes samples and quantifies reporter ions. It reduces
missingness and run-to-run variation, at the cost of ratio compression: co-isolated peptides
contribute background to the reporter channels, shrinking observed ratios toward one.

**Targeted** assays (PRM, MRM) measure pre-specified peptides with a calibration curve. They
are the bridge to clinical use, because they can be validated for accuracy, precision, and
range in the way a clinical assay requires.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Peptide | Short amino acid chain produced by digestion | Treated as a protein | Sequence |
| Protein group | Set of proteins supported by shared peptides | Reported as one protein | Group ID |
| Precursor | Ion selected for fragmentation | Confused with peptide | m/z |
| Fragment ion | Product ion from fragmentation | Used loosely for any peak | m/z |
| Retention time | Elution time in chromatography | Assumed stable across runs | Minutes |
| m/z | Mass-to-charge ratio | Treated as mass | Th (Thomson) |
| DDA | Data-dependent acquisition | Assumed unbiased | Acquisition mode |
| DIA | Data-independent acquisition | Assumed complete | Acquisition mode |
| PRM/MRM | Targeted acquisition of specified precursors | Confused with DIA | Acquisition mode |
| TMT | Tandem mass tag isobaric labelling | Assumed ratio-free | Reagent |
| iTRAQ | Isobaric tag for relative quantification | Same as TMT | Reagent |
| Label-free quantification (LFQ) | Quantification from precursor intensities | Treated as absolute | Intensity |
| MaxLFQ | Ratio-based label-free normalization | Treated as a calibration | Algorithm |
| MS1 | Survey scan of intact ions | Confused with MS2 | Scan |
| MS2 | Fragmentation scan | Confused with identification | Scan |
| PSM | Peptide-spectrum match | Treated as identification | Match |
| Decoy | Reversed or shuffled sequence for FDR estimation | Assumed perfect | Sequence |
| Peptide FDR | Expected false fraction among accepted PSMs | Confused with protein FDR | Proportion |
| Protein FDR | Expected false fraction among reported proteins | Confused with peptide FDR | Proportion |
| PEP | Posterior error probability for a PSM | Confused with q-value | Probability |
| Protein inference | Assignment of peptides to proteins | Assumed unique | Procedure |
| PTM | Post-translational modification | Assumed localized | Modification |
| Localization probability | Confidence in a PTM site | Ignored in site claims | Probability |
| Ratio compression | Shrinkage of isobaric ratios | Mistaken for biology | Proportion |
| Ion suppression | Reduced ionization in a matrix | Ignored | Effect |
| Carryover | Contamination from a previous sample | Mistaken for signal | Proportion |
| LOD | Limit of detection | Confused with LLOQ | Concentration |
| LLOQ | Lower limit of quantification | Confused with LOD | Concentration |
| Calibration curve | Standard curve for absolute quantification | Assumed linear at the extremes | Concentration |
| Metabolite feature | A peak with m/z and retention time | Called a metabolite | Feature |
| Adduct | Ion formed with a salt or solvent | Mistaken for another metabolite | m/z shift |
| Isomer | Same mass, different structure | Unresolved by MS1 | Structure |
| MSI level | Metabolite identification confidence level (1-4) | Treated as binary identified | Level |
| HMDB | Human Metabolome Database | Treated as complete | Database |
| InChI | Structure identifier | Confused with a name | String |
| MIAPE | Minimum information about a proteomics experiment | Assumed enforced | Reporting guideline |
| mzML/mzIdentML | Standard mass spectrometry formats | Assumed lossless | XML |

Three definitions matter most.

**Peptide FDR and protein FDR are different quantities.** A 1% peptide FDR can correspond to a
much higher protein FDR, because one false peptide can support a false protein, and because
proteins are inferred from multiple peptides with shared evidence. Reporting a protein list at
"1% FDR" without saying which level was controlled is ambiguous.

**LOD and LLOQ are different.** The limit of detection is the concentration at which a signal
can be distinguished from noise; the lower limit of quantification is the concentration at
which it can be measured with acceptable precision. A value between them is detected but not
quantifiable, and reporting it as a number is a category error.

**A metabolite feature is not a metabolite.** An m/z and retention time define a feature. The
assignment to a chemical structure is an annotation with a confidence level, and many features
remain unannotated even in well-studied matrices.

## 5. The mechanics, step by step

### Step 1. Define the analyte and the required sensitivity

**Input:** the biological question and the expected concentration range.

**Output:** an assay specification with a target sensitivity and a matrix.

**Rationale:** Plasma proteins span more than ten orders of magnitude in concentration, and a
handful of abundant proteins dominate the mass spectrum. A discovery experiment without
depletion may quantify the abundant proteins well and miss the low-abundance ones entirely. The
question determines whether depletion, fractionation, or a targeted assay is required.

### Step 2. Prepare the sample with a documented protocol

**Input:** specimen.

**Output:** peptides or metabolite extract with a processing record.

**Rationale:** Digestion efficiency, depletion reproducibility, and extraction recovery are the
largest sources of technical variation. A protocol change mid-study creates a batch effect that
is confounded with the sample order. Record lysis, depletion, digestion time, and cleanup for
every batch.

### Step 3. Choose the acquisition mode

**Input:** the question and the sample count.

**Output:** DDA, DIA, or targeted acquisition.

**Rationale:** DDA selects the most intense precursors and is biased toward abundant peptides;
it is good for identification and poor for consistent quantification across many samples. DIA
fragments all precursors in windows and gives more consistent quantification, at the cost of
more complex analysis. Targeted acquisition measures specified peptides with high sensitivity
and is the appropriate mode for validation.

### Step 4. Identify peptides with a controlled FDR

**Input:** spectra.

**Output:** peptide-spectrum matches with q-values and a stated FDR.

**Rationale:** The target-decoy approach estimates the false discovery rate by counting decoy
matches. A 1% FDR at the peptide level is a common threshold, but the choice should reflect the
downstream claim: a biomarker discovery may use 1%, while a clinical assay needs a validated
limit of detection rather than an FDR.

### Step 5. Perform protein inference

**Input:** peptide matches.

**Output:** protein groups with supporting peptides.

**Rationale:** Shared peptides make unique assignment impossible for some proteins. The
parsimonious approach reports the smallest set of proteins explaining the peptides observed.
This is a modelling choice, and the group composition can change with the database. Report
whether the analysis is at the protein or protein-group level.

### Step 6. Quantify with a method matched to the acquisition

**Input:** identified features.

**Output:** protein by sample intensity matrix.

**Rationale:** Label-free quantification uses precursor intensities and requires normalization
for loading and instrument drift. Isobaric quantification uses reporter ion intensities and
requires correction for ratio compression. Either way, the matrix should carry a record of how
missing values arose: not identified, identified but not quantified, or quantified below the
limit.

### Step 7. Diagnose missingness before choosing a treatment

**Input:** protein by sample matrix with missingness flags.

**Output:** a missingness profile by group and a mechanism assessment.

**Rationale:** Compare the missing fraction between groups, and compare the observed intensity
distribution of proteins with high and low missingness. If missingness is higher in the group
with lower abundance, it is MNAR. A protein detected in 40 of 60 cases and 55 of 60 controls is
a detection difference, which may be the biological signal rather than a nuisance.

### Step 8. Filter features with a pre-specified rule

**Input:** matrix and missingness profile.

**Output:** analysis set.

**Rationale:** A common rule is to retain proteins detected in at least 70% of samples in at
least one group. The rule must be pre-specified, because filtering on differential missingness
selects on the outcome. Report the number removed and the reason.

### Step 9. Normalize with a stated assumption

**Input:** filtered matrix.

**Output:** normalized matrix.

**Rationale:** Median or quantile normalization assumes that the global intensity distribution
is comparable across samples. That assumption fails when the sample composition differs, or
when depletion efficiency varies. Compare normalization methods and report the sensitivity of
the result.

### Step 10. Model batch and covariates

**Input:** normalized matrix and design.

**Output:** model with batch and relevant covariates.

**Rationale:** Mass spectrometry batches are strong: column condition, instrument tuning, and
reagent lot all change intensities. Include batch in the model, and check that it is not
collinear with the condition. For isobaric designs, the labelling batch is a blocking factor.

### Step 11. Test and correct across the right family

**Input:** model fit.

**Output:** effect sizes, p-values, q-values, and the number of tests.

**Rationale:** The family is the set of features tested, which must match the claim. Testing
1,200 proteins and correcting across 1,200 is different from testing a pre-specified panel of
60 and correcting across 60. The second is legitimate if the panel was pre-specified; the first
is not legitimate if the panel was chosen after seeing the p-values.

### Step 12. Assign identification confidence for metabolites

**Input:** annotated features.

**Output:** confidence levels with evidence.

**Rationale:** The Metabolomics Standards Initiative defines four levels: identified compounds
(level 1) validated against an authentic standard under identical conditions with at least two
orthogonal properties; putatively annotated compounds (level 2) based on spectral similarity
without a standard; putatively characterized compound classes (level 3); and unknown compounds
(level 4). A level-2 annotation supports a hypothesis, not a clinical claim.

### Step 13. Validate orthogonally

**Input:** candidate features.

**Output:** confirmation with a different method.

**Rationale:** For proteins, a targeted PRM assay or an immunoassay. For metabolites, an
authentic standard, a different chromatography, or NMR. Orthogonal validation tests whether the
signal is in the sample rather than in the pipeline.

### Step 14. Report the analytical limits

**Input:** everything above.

**Output:** a report with acquisition, FDR, missingness, normalization, batch, and limits.

**Rationale:** The report must state what the assay cannot measure: the dynamic range covered,
the LOD and LLOQ, the identification confidence, and the missingness mechanism. A reader who
does not know these cannot judge whether a negative result is absence or insensitivity.

### Step 15. Separate measurement error from biological variance

**Input:** replicate and sample-level data.

**Output:** variance components.

**Rationale:** A discovery result is only as good as its ability to distinguish biological from
technical variation. With triplicate injections, the technical CV can be estimated directly;
without replicates, it must be inferred from the platform's published performance. A protein
with a 30% technical CV cannot support a 1.2-fold claim, regardless of its p-value.

The arithmetic is simple and often omitted: the observed variance is the sum of biological and
technical variance, so the biological component is the difference, and it can be negative when
sampling noise dominates. A negative estimate is a signal that the study cannot resolve the
effect it is chasing.

### Quantification arithmetic

Three calculations come up repeatedly and are worth writing down.

**Ratio to a reference sample.** If a pooled reference is run in every batch, each sample's
intensity can be divided by the reference in its batch. This removes batch-level scale
differences but assumes the reference is stable and that the analyte is measurable in it.

**Median normalization.** Each sample is divided by its median intensity across features, then
multiplied by the cohort median. This assumes that the median feature is unchanged, which fails
when the sample composition differs systematically — for example, when depletion efficiency
varies between batches.

**Isolation purity correction for isobaric data.** If a precursor is co-isolated with
interfering peptides at a known proportion, the observed reporter ratios can be corrected
toward the true ratios. The correction requires an estimate of interference, which is often
unavailable, so the honest alternative is to report that ratios are compressed.

### Branch points

**Branch A — the candidate is a low-abundance plasma protein.** Discovery is likely
insensitive. Move to depletion plus targeted assay, or accept that the discovery cannot see it.

**Branch B — missingness differs by group.** Treat detection as a separate outcome. Test the
detection difference, and do not impute the value as if it were observed.

**Branch C — the metabolite is level 2.** Do not report a concentration or a clinical
association as if the identity were confirmed. Confirm with a standard before the claim moves
forward.

### Decision table: question to platform

| Question | Platform | Quantification | Limitation |
|---|---|---|---|
| Discover candidate proteins | DDA or DIA label-free | Relative | Missingness, dynamic range |
| Compare many samples consistently | DIA | Relative | Library dependence |
| Multiplex 10-16 samples | TMT | Relative ratios | Ratio compression |
| Validate a candidate protein | PRM or immunoassay | Absolute with calibration | Requires reagent |
| Measure low-abundance plasma protein | Depletion + targeted | Absolute | Depletion bias |
| Discover metabolites | LC-MS untargeted | Relative feature intensity | Annotation confidence |
| Confirm a metabolite | Authentic standard, LC-MS/MS or NMR | Absolute | Standard availability |
| Clinical measurement | Validated immunoassay or MS with IVD | Absolute | Regulatory validation |

### Edge cases

**MNAR missingness.** Diagnostic: missing fraction higher in the low-abundance group.
Consequence: imputation creates or erases effects. Mitigation: model detection; report the
detection test separately.

**Low-abundance proteins.** Diagnostic: a protein near the detection limit has high variability
and high missingness. Consequence: unstable fold changes. Mitigation: exclude from discovery
claims; validate with a targeted assay.

**Ion suppression.** Diagnostic: a sample with high total ion current shows reduced signal for
specific peptides. Consequence: false negatives correlated with matrix. Mitigation: cleanup,
internal standards, dilution.

**Chimeric spectra.** Diagnostic: a spectrum matches two peptides with similar scores.
Consequence: misassignment. Mitigation: stricter scoring, MS3, or DIA.

**Shared peptides.** Diagnostic: a peptide maps to several protein groups. Consequence:
ambiguous protein inference. Mitigation: report groups; do not claim a specific isoform.

**PTM localization.** Diagnostic: the modification could sit on either of two residues.
Consequence: a site-specific claim that is wrong. Mitigation: report localization probability;
require high confidence for site claims.

**Ratio compression (isobaric).** Diagnostic: ratios cluster near one across known spikes.
Consequence: effect sizes understated. Mitigation: MS3-based quantification, or report the
compression.

**Carryover.** Diagnostic: a blank run after a high-abundance sample shows the same peak.
Consequence: false positives in the next sample. Mitigation: wash runs, randomized order,
blank subtraction.

**Batch confounding.** Diagnostic: all cases in one instrument block. Consequence: batch and
condition inseparable. Mitigation: randomization at design; otherwise restrict the claim.

**Isomer ambiguity.** Diagnostic: two features with the same m/z but different retention times.
Consequence: the wrong isomer is named. Mitigation: authentic standards; report the ambiguity.

**Adduct misassignment.** Diagnostic: a feature at a mass consistent with a sodium adduct of
another feature. Consequence: double counting one metabolite. Mitigation: adduct deconvolution.

**Detergent contamination.** Diagnostic: polymers dominating the spectrum. Consequence: reduced
peptide identification. Mitigation: cleanup and QC.

**Under-filtration.** Diagnostic: proteins detected in 2 of 60 samples included in the test.
Consequence: unstable estimates. Mitigation: pre-specified detection filter.

**Detergent and polymer contamination.** Diagnostic: a repeating mass series dominating the
spectrum, with few peptide identifications. Consequence: wasted instrument time and low
identification. Mitigation: cleanup, blank runs, and a contamination check before the batch.

**Sample ordering effect.** Diagnostic: signal drifts monotonically with injection order.
Consequence: a trend confounded with any group that was run in sequence. Mitigation: randomize
order, insert washes, and include a pooled reference at intervals.

**Depletion batch effect.** Diagnostic: the abundant-protein profile differs by depletion lot.
Consequence: all downstream intensities shift. Mitigation: single lot, or record lot and include
it as a covariate.

**Peptide-level selection bias.** Diagnostic: only peptides with high ion intensity are used for
quantification, and the selection differs by group. Consequence: a protein-level difference
driven by peptide selection. Mitigation: use all identified peptides, or a consistent
top-N rule across samples.

**Database version drift.** Diagnostic: protein groups change when the search database is
updated. Consequence: incomparable results across time. Mitigation: pin the database release and
report it.

**Overfitting the spectral library.** Diagnostic: a DIA library built from the same samples it is
used to quantify. Consequence: circular identification and optimistic quantification.
Mitigation: build the library from an independent sample or a public resource, and report the
library source.

**Isobaric channel imbalance.** Diagnostic: one reporter channel has systematically lower
intensity across all peptides. Consequence: the sample appears globally lower. Mitigation: check
labelling efficiency, re-label, or normalize by channel median and report the imbalance.

### What the procedure produces

The output is a feature matrix with identification confidence, a missingness profile, a
normalized and batch-adjusted analysis set, effect sizes with intervals, corrected
significance, and an orthogonal validation plan. The identification and missingness records are
as important as the p-values. A matrix without them cannot be reanalysed, only re-reported. The record is the analysis.
Without it, a reviewer can check arithmetic but not provenance, and cannot decide whether to believe either.
The provenance record is short: instrument, method, database, search engine, and the thresholds used.

## 6. Worked example

A discovery study measures plasma proteins in 60 patients: 30 with a condition and 30 controls.
The platform quantifies 1,200 proteins, with an average missingness of 20%. Samples were run in
three instrument batches.

### 6.1 Identification arithmetic

The search accepts peptide-spectrum matches at a 1% target-decoy FDR. Of 84,000 spectra, 21,400
PSMs pass, and the decoy count among accepted matches implies:

```text
estimated false PSMs = 0.01 x 21,400 = 214
```

Protein inference collapses the peptides into 1,240 protein groups. If protein-level FDR is
controlled separately at 1%, the reported list is 1,200 proteins, implying about 12 false
proteins. The two numbers are not interchangeable: the 214 false peptides do not map one-to-one
to false proteins, and a protein supported by one peptide carries more risk than one supported
by ten.

### 6.2 Missingness as a signal

One candidate protein, call it P1, is detected in only 40 of the 60 samples overall, and the
missingness is not evenly distributed:

| Group | Detected | Not detected | Total |
|---|---:|---:|---:|
| Cases | 15 | 15 | 30 |
| Controls | 25 | 5 | 30 |

A two-sided Fisher exact test on this table gives:

```text
p = 0.0127
```

The detection itself differs by group. That is a result in its own right: the protein is more
often detectable in controls. It is not a missing-data nuisance, and imputing the missing cases
would erase the signal by assigning them a value they did not have.

The arithmetic of the table is worth stating: the odds ratio is (15 x 5) / (15 x 25) = 0.20, so
the odds of detection in cases are one fifth of the odds in controls.

### 6.3 Differential expression and correction

Among samples where P1 is quantified, the mean difference is 0.58 log2 units (1.49-fold) with a
standard error of 0.16:

```text
95% CI = 0.58 +/- 1.96 x 0.16 = [0.27, 0.89]
```

The t-test p-value is 4.0 x 10^-4. With 1,200 proteins tested, the BH q-value for the smallest
p-value is:

```text
q = 4.0 x 10^-4 x 1,200 / 1 = 0.48, and the BH minimum over all ranks is 0.36
```

No protein is significant at q < 0.05. The study has no corrected discoveries. That is the
result, and it must be reported as such.

If the analysis had instead been restricted to a pre-specified panel of 60 proteins, the same
p-value would give:

```text
q = 4.0 x 10^-4 x 60 / 1 = 0.024, with a BH minimum of 0.018
```

which is significant. The difference is not the data; it is the family. A pre-specified panel is
a legitimate family if it was fixed before the data were seen. A panel assembled from the
top hits is not a family; it is a selection, and correcting within it is circular.

### 6.4 The imputation experiment

The analyst imputes missing values with the minimum observed intensity and reruns the test with
all 60 samples. The imputed cases receive a value below every observed value, so the case mean
falls and the effect grows:

```text
imputed log2FC = 0.91 (1.88-fold), p = 8.0 x 10^-7
```

The effect nearly doubles, and the p-value falls by three orders of magnitude. Neither change is
biological. The imputation asserted that the 15 missing cases had values below the detection
limit, which is exactly the assertion the detection test was designed to evaluate. The correct
analysis treats detection and abundance separately, not by imputing.

### 6.5 Diagnostic 1: the batch effect

Samples were run in three batches, and the case-control ratio differs by batch: batch 1 is 18
cases and 2 controls, batch 2 is 8 and 14, batch 3 is 4 and 14. Batch and condition are
correlated, though not perfectly.

After adding batch to the model, the P1 effect changes:

```text
without batch: log2FC = 0.58, p = 4.0 x 10^-4
with batch:    log2FC = 0.21, 95% CI [-0.09, 0.51], p = 0.17
```

The effect is no longer distinguishable from zero. The batch distribution was the diagnostic:
a condition that is 90% in one batch cannot be separated from that batch's instrument state.

This is the most common way a plasma proteomics candidate is born and dies. The design made the
question unanswerable, and no statistical method recovers it.

### 6.6 Diagnostic 2: the level-2 metabolite

The same study includes an untargeted metabolomics arm. A feature is annotated as a specific
acylcarnitine by spectral library match, without an authentic standard. That is MSI level 2.

A level-2 annotation supports "a feature with this spectrum differs between groups." It does not
support "this acylcarnitine is elevated," because the library match does not exclude an isomer.
The distinction matters clinically: an isomer with a different biological role would change the
interpretation.

Confirming the annotation requires an authentic standard run under the same conditions, giving
level 1. Until then, the claim is about a feature, not a molecule.

### 6.7 Normalization arithmetic

The instrument drifts over the run. A pooled reference sample is injected every ten samples, and
its median intensity is used to rescale each block. Suppose the reference medians for the three
blocks are 1.00, 1.18, and 0.92 in arbitrary units relative to the first block. Sample
intensities in block 2 are divided by 1.18, and block 3 by 0.92, before any comparison.

```text
block 1: scale 1.00
block 2: scale 1.18
block 3: scale 0.92
```

If the normalization is omitted, a protein measured at 1,000 units in block 1 and 1,180 units in
block 2 appears to rise by 18%, which is exactly the drift. The reference corrects it.

The assumption is that the reference is stable and measurable. If the pooled sample degrades
over the run, the correction removes a real trend and imposes an artificial one. Checking the
reference's own intensity over time is the diagnostic: a monotonic decline in the reference
means the normalization is correcting for degradation, not drift.

### 6.8 Vary one assumption: complete-case analysis

Suppose the analysis restricts to the 40 samples where P1 is quantified, ignoring the detection
difference. The complete-case comparison has 15 cases and 25 controls, and it estimates the
effect among detectable samples only.

```text
complete-case log2FC = 0.58 (same estimate)
interpretation: the effect among samples where the protein was detectable
```

The estimate is unchanged, but the population is different. The complete-case claim is about
detectable samples, not about all patients. If detectability itself is the biomarker, the
complete-case analysis discards the strongest part of the signal.

Changing one assumption — that the analysis population is all patients rather than detectable
patients — changes the claim from "the protein is higher in cases" to "the protein is more often
detectable in controls, and among detectable samples it is higher in cases." Those are different
biomarkers with different assays: one needs a presence/absence test, the other a quantitative
one.

### 6.9 What the report says

> Plasma proteomics, 60 samples, three instrument batches, DDA label-free. 21,400 PSMs at 1%
> peptide FDR; 1,200 protein groups at 1% protein FDR. Average missingness 20%. P1 detected in
> 15/30 cases and 25/30 controls (Fisher exact p = 0.0127); among detected samples, log2FC 0.58
> (95% CI 0.27-0.89), p = 4.0 x 10^-4, BH q = 0.36 across 1,200 proteins. After batch
> adjustment, log2FC 0.21 (95% CI -0.09 to 0.51), p = 0.17. No corrected discoveries. A
> pre-specified 60-protein panel analysis gives q = 0.024 for P1 but is reported as hypothesis-
> generating. Metabolomics: one level-2 annotation, not confirmed.

That report is honest, and it is more useful than a table of 40 "significant" proteins.

## 7. Code, query, or calculation

The script implements Fisher's exact test, BH correction, a detection-versus-abundance
decomposition, and an identification-confidence classifier.

### 7.1 Fisher's exact test from first principles

```python
import math

def hypergeom(a, b, c, d):
    n = a + b + c + d
    return math.comb(a + b, a) * math.comb(c + d, c) / math.comb(n, a + c)

def fisher_two_sided(a, b, c, d):
    obs = hypergeom(a, b, c, d)
    r1, r2, c1 = a + b, c + d, a + c
    total = 0.0
    for x in range(max(0, c1 - r2), min(r1, c1) + 1):
        y = r1 - x
        z = c1 - x
        w = r2 - z
        p = hypergeom(x, y, z, w)
        if p <= obs + 1e-12:
            total += p
    return total

# 2x2 table: cases detected/not, controls detected/not
a, b, c, d = 15, 15, 25, 5
print("table:", [[a, b], [c, d]])
print("p =", round(fisher_two_sided(a, b, c, d), 4))
print("odds ratio =", round((a * d) / (b * c), 3))
```

Expected output:

```text
table: [[15, 15], [25, 5]]
p = 0.0127
odds ratio = 0.2
```

The test is exact and assumes fixed margins. It answers whether detection is associated with
group, which is a different question from whether abundance differs.

### 7.2 BH across the protein family

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

pvals = [4.0e-4, 6.0e-4, 1.2e-3, 3.0e-3]
for family in (1200, 60):
    q = bh(pvals, family)
    print("family", family, [round(x, 4) for x in q])
```

Expected output:

```text
family 1200 [0.36, 0.36, 0.48, 0.9]
family 60 [0.018, 0.018, 0.024, 0.045]
```

The same four p-values are a null result in one family and a significant result in the other.
The family is a design decision made before the data, not a choice made after.

### 7.3 Detection versus abundance

```python
def summarize(detected_cases, total_cases, detected_controls, total_controls):
    pc = detected_cases / total_cases
    pk = detected_controls / total_controls
    return pc, pk, pc - pk

pc, pk, diff = summarize(15, 30, 25, 30)
print(f"detection cases {pc:.2f}, controls {pk:.2f}, difference {diff:.2f}")

# Among detected samples only
n_cases_det, n_ctrl_det = 15, 25
mean_cases, mean_ctrl = 8.42, 7.84
print("among detected: log2 difference", round(mean_cases - mean_ctrl, 2),
      "fold", round(2 ** (mean_cases - mean_ctrl), 2))
```

Expected output:

```text
detection cases 0.50, controls 0.83, difference -0.33
among detected: log2 difference 0.58 fold 1.5
```

Two numbers, two claims. The detection difference is 33 percentage points; the abundance
difference among detected samples is 1.5-fold. Reporting only the second omits the larger
signal.

### 7.4 Identification confidence classifier

```python
def msi_level(has_standard, orthogonal_properties, library_match, class_match):
    if has_standard and orthogonal_properties >= 2:
        return 1
    if library_match:
        return 2
    if class_match:
        return 3
    return 4

cases = [
    ("standard + RT + MS2", True, 2, False, False),
    ("library match only", False, 0, True, False),
    ("class match only", False, 0, False, True),
    ("unmatched feature", False, 0, False, False),
]
for label, std, orth, lib, cls in cases:
    print(label, "-> level", msi_level(std, orth, lib, cls))
```

Expected output:

```text
standard + RT + MS2 -> level 1
library match only -> level 2
class match only -> level 3
unmatched feature -> level 4
```

The classifier encodes the definition, not a judgment about how confident the analyst feels. A
level-2 annotation is a hypothesis; the code makes that explicit.

### 7.5 Coefficient of variation and replicate precision

```python
import statistics

replicates = [7.82, 8.10, 7.95]
mean = statistics.mean(replicates)
sd = statistics.stdev(replicates)
cv = sd / mean
print("mean", round(mean, 3), "SD", round(sd, 3), "CV", round(cv, 4))
print("CV of the mean of 3:", round(cv / (3 ** 0.5), 4))
```

Expected output:

```text
mean 7.957 SD 0.14 CV 0.0176
CV of the mean of 3: 0.0102
```

A per-replicate CV of 1.8% becomes 1.0% for a triplicate mean. This is the arithmetic behind
"technical replicates reduce measurement error but not biological variance": the CV falls with
the square root of the replicate count, and biological between-sample variance does not.

### 7.6 Assumptions and limits of this code

- Fisher's exact test assumes fixed margins and independent observations; repeated samples from
  one patient violate independence.
- The BH function assumes independence or positive dependence; correlated proteins violate this
  mildly, and the correction is approximate.
- The MSI classifier implements the published levels schematically; real annotation requires
  spectral evidence review.
- The CV example uses log-transformed values; a CV on raw intensities would differ.
- No code here performs identification, quantification, or normalization.

## 8. Data, standards, and regulatory context

### 8.1 Proteomics standards

The Human Proteome Organisation Proteomics Standards Initiative (HUPO-PSI) maintains the
formats and reporting guidelines:

- **mzML** for mass spectra, **mzIdentML** for identification results, **mzTab** for
  quantification summaries.
- **MIAPE** (Minimum Information About a Proteomics Experiment) defines the metadata modules:
  sample, separation, mass spectrometry, and informatics.
- **Controlled vocabularies** for modifications and instrument parameters.

These are reporting conventions, not enforced schemas. A submission can be MIAPE-compliant in
free text and still be unanalyzable.

### 8.2 Metabolomics standards

The Metabolomics Standards Initiative (MSI) published minimum reporting standards in 2007,
including the four identification confidence levels. The levels are:

| Level | Evidence | Claim supported |
|---|---|---|
| 1 | Authentic standard, same conditions, at least two orthogonal properties | Identified compound |
| 2 | Library or literature spectral match, no standard | Putatively annotated compound |
| 3 | Class-level spectral or physicochemical evidence | Putatively characterized class |
| 4 | Unmatched but differentiable feature | Unknown compound |

The levels are a reporting convention. Journals increasingly require them; regulators do not
yet enforce them uniformly.

### 8.3 Clinical laboratory standards

When a protein or metabolite measurement is used clinically, the assay is regulated as a
clinical test:

- **CLIA** requires accuracy, precision, analytical sensitivity and specificity, reportable
  range, and reference intervals.
- **CAP** accreditation adds proficiency testing.
- **ISO 15189:2022** provides the international quality framework.
- **IVDR 2017/746** applies in the EU to in vitro diagnostic devices.
- **FDA** regulates devices and maintains the biomarker qualification pathway.

Mass spectrometry assays can be clinical tests, and several are cleared or approved, but they
require calibration, internal standards, and validation that discovery workflows do not
provide.

### 8.4 Biomarker qualification

The FDA Biomarker Qualification Program accepts a biomarker for a defined context of use through
a three-stage process: Letter of Intent, Qualification Plan, and Full Qualification Package. The
qualification attaches to the biomarker, not to a specific assay; any analytically validated
assay can be used within the context of use.

ICH E16 defines the context, structure, and format of qualification submissions for genomic
biomarkers, with principles that apply to proteomic and metabolomic biomarkers as well. The
context of use must specify the biomarker category, the intended drug development use, and the
critical parameters such as sample type and assay.

### 8.5 What the standards do not settle

- Which missingness treatment is appropriate.
- How to normalize across batches.
- What fold change is meaningful.
- How to handle ratio compression.
- When a level-2 annotation is sufficient.

These are scientific choices, and they decide whether a result replicates.

## 9. Industry practice and case evidence

### 9.1 The missingness evidence

Lazar et al. (2016) compared imputation strategies across label-free proteomics datasets and
found that methods designed for MNAR (minimum-value and probabilistic minimum) outperform
MCAR/MAR methods when at least 70% of missingness is MNAR, while k-nearest-neighbour and
singular-value methods perform better when missingness is mostly MCAR/MAR. Their practical
recommendation was to impute at the peptide level before protein aggregation and to diagnose the
mechanism first.

Harris et al. (2024) evaluated imputation with downstream-centric criteria and found that
imputation often does not improve differential-expression detection relative to no imputation,
and that single-value methods can perform well on error metrics while performing poorly
downstream.

Li and Smyth (2023) argued that missingness in label-free proteomics is neither random nor
purely censored but depends on intensity through a logit-linear detection function, and that
modelling detection likelihood is preferable to imputing.

The practical convergence: diagnose first, prefer models over imputation, and never impute a
minimum without stating that the value is a bound.

### 9.2 What industry does

Documented practice:

- Discovery proteomics uses DDA or DIA on cohorts of tens to low hundreds, with batch-balanced
  designs and a pre-specified analysis plan.
- Validation moves to PRM or an immunoassay with a calibration curve.
- Large cohort studies use affinity-based platforms (Olink, SomaScan) rather than mass
  spectrometry when throughput dominates; these are not mass spectrometry and have their own
  specificity and missingness profiles.
- Metabolomics discovery reports MSI levels and confirms key annotations with standards.

Inference: the most common organizational failure is treating an affinity-platform association
as a mass-spectrometry measurement. The two have different specificity, different dynamic range,
and different failure modes, and a candidate that replicates on one may not replicate on the
other.

### 9.3 The cohort scale evidence

The UK Biobank Pharma Proteomics Project measured a few thousand plasma proteins in tens of
thousands of participants using an affinity-based platform, and has been used for
protein-quantitative trait locus mapping and biomarker discovery. The scale is the point: at
that sample size, small effects become statistically detectable, which makes effect-size
reporting and replication more important, not less.

> **Note:** platform, protein count, and participant numbers should be checked against the
> current publication and data release before citation; they change with each release.

### 9.4 The clinical translation record

Clinical mass spectrometry assays exist for newborn screening, therapeutic drug monitoring, and
some hormone measurements, and they are validated with calibration curves and internal
standards. Discovery proteomics has produced relatively few clinically validated protein
biomarkers, and the reasons are consistent: insufficient sample size, missingness treated as
noise, batch confounding, and no orthogonal validation. The biomarker pipeline problem is
covered in topic 08.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Minimum-value imputation | Many values at the floor | MNAR treated as zero | Inflated effects | Model detection; report bound |
| No multiple-testing correction | Long hit list at p < 0.05 | Family ignored | False discoveries | BH with stated m |
| Correction after selection | Shortlist corrected only | Selection bias | Invalid q-values | Pre-specify family |
| Batch confounded | Condition concentrated in one batch | Design | Effect is batch | Balance design |
| Peptide/protein FDR confusion | "1% FDR" without level | Different quantities | Overstated protein confidence | Report both levels |
| Shared-peptide overclaim | Protein group reported as one protein | Inference ambiguity | Wrong isoform claim | Report groups |
| Ratio compression | Isobaric ratios near one | Co-isolation | Understated effects | MS3; report compression |
| Carryover | Blank run shows the peak | Contamination | False positive | Wash runs; randomize |
| Ion suppression | Sample-specific signal loss | Matrix effect | False negatives | Cleanup; internal standard |
| Isomer misassignment | Same m/z, different RT | MS1 cannot separate | Wrong metabolite named | Standards; report ambiguity |
| Level-2 overclaim | No standard used | Annotation only | Clinical claim unsupported | Confirm to level 1 |
| LOD reported as value | Values near noise | Detection not quantification | False precision | Report LLOQ |
| Under-filtering | Proteins in 2 of 60 samples | Unstable estimates | Noise in results | Pre-specified filter |
| Over-normalization | Biological global shift removed | Median/quantile assumption | Signal erased | Compare methods |
| No orthogonal validation | Single-platform claim | Platform-specific signal | Non-replication | PRM or immunoassay |

Four red flags:

**If the fold change grows after imputation, stop.** The imputation created it.

**If a protein is "significant" with p = 4 x 10^-4 among 1,200 tests, stop.** It is not
significant after correction.

**If the condition is concentrated in one instrument batch, stop.** You are measuring the
instrument.

**If a metabolite is named without a standard, stop.** You have a feature, not a molecule.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Peptides identify their source protein | Protein inference | Shared peptides, isoforms | Peptide uniqueness count |
| FDR estimates are calibrated | Identification confidence | Decoy model mismatch | Decoy ratio check |
| Missingness is ignorable | Complete-case analysis | MNAR | Detection by group |
| Imputation preserves distribution | Imputed analysis | MNAR, group differences | Compare with/without |
| Intensities are proportional to abundance | Quantification | Ion suppression, saturation | Spike-in linearity |
| Normalization removes loading effects | Cross-sample comparison | Composition differences | Compare methods |
| Batch effects are additive | Batch adjustment | Interaction with condition | Check collinearity |
| Ratios are not compressed | Isobaric quantification | Co-isolation | Spike-in ratios |
| Annotation identifies the metabolite | Metabolite claims | Isomers, adducts | Authentic standard |
| Discovery generalizes to validation | Translation | Platform and cohort differences | Orthogonal validation |

The missingness assumption is the load-bearing one. Every imputation method assumes something
about why values are absent, and none of those assumptions is testable within the imputed
dataset. It is testable only by comparing detection across groups, which is a separate analysis
that should be reported.

## 12. Exercises

**Exercise 1 — Identification arithmetic (calculation, 25 minutes).**
A search accepts 30,000 PSMs at 1% peptide FDR and reports 1,500 protein groups at 1% protein
FDR. Compute the expected false PSMs and false proteins. Explain why the two numbers are not
proportional.
*Deliverable:* arithmetic and a three-sentence explanation.
*Criteria:* both counts correct; the explanation names shared peptides and inference.
*Time:* 25 minutes.

**Exercise 2 — Missingness diagnosis (application, 35 minutes).**
A protein is detected in 18/40 cases and 32/40 controls. Compute the detection difference, run
a Fisher exact test by hand or with the code from section 7, and state whether the analysis
should impute.
*Deliverable:* table, p-value, and a recommendation.
*Criteria:* p-value correct to three decimals; recommendation follows from the mechanism.
*Time:* 35 minutes.

**Exercise 3 — Correction family (calculation, 30 minutes).**
A study tests 900 proteins and finds a candidate at p = 2 x 10^-4. Compute the BH q-value at
rank 1. Then compute it for a pre-specified 40-protein panel. State which analysis is
legitimate and why.
*Deliverable:* two q-values and a justification.
*Criteria:* both correct; the justification distinguishes pre-specification from selection.
*Time:* 30 minutes.

**Exercise 4 — Metabolite confidence (application, 25 minutes).**
Classify each as MSI level 1-4 and state the claim each supports: (a) spectrum matches a library
entry, no standard; (b) standard run in the same laboratory with retention time and MS2 match;
(c) feature with no match; (d) spectrum consistent with a lipid class.
*Deliverable:* four-row table.
*Criteria:* levels correct; claims match the levels.
*Time:* 25 minutes.

**Exercise 5 — Critique a proteomics paper (critique, 60 minutes).**
Select a published plasma proteomics study. Extract its missingness treatment, correction
family, batch handling, and validation. Identify three defects and their consequences.
*Deliverable:* one-page critique.
*Criteria:* three specific defects; consequences quantified where possible.
*Time:* 60 minutes.

**Exercise 6 — Design a validation (design, 60 minutes).**
Design a targeted PRM assay to validate three candidate proteins from a discovery study.
Specify peptides, internal standards, calibration, acceptance criteria, and sample size.
*Deliverable:* a validation protocol outline.
*Criteria:* two peptides per protein where possible; calibration range; LLOQ; acceptance
criteria numeric.
*Time:* 60 minutes.

**Exercise 7 — Ratio compression (application, 30 minutes).**
An isobaric experiment shows a known 2-fold spike at 1.6-fold measured. Correct the observed
ratio using a compression factor and state the assumption.
*Deliverable:* arithmetic and one sentence on the assumption.
*Criteria:* correction method stated; assumption named; no overcorrection.
*Time:* 30 minutes.

**Exercise 8 — Analysis plan (design, 45 minutes).**
Write a pre-specified analysis plan for a 120-sample plasma proteomics study with three batches.
Include missingness handling, normalization, batch, family, correction, and validation.
*Deliverable:* two-page plan.
*Criteria:* every choice pre-specified; detection analysis included; go/no-go rule numeric.
*Time:* 45 minutes.

## 13. Self-check answers

**Answer 1 (identification).** Expected false PSMs: 30,000 x 0.01 = 300. Expected false
proteins: 1,500 x 0.01 = 15. The numbers are not proportional because a false protein can be
supported by several peptides, and a single false peptide does not necessarily create a false
protein. Protein inference aggregates evidence, so the protein-level error rate depends on the
peptide-level error rate, the number of peptides per protein, and the sharing structure.

**Answer 2 (missingness).** Detection is 45% in cases and 80% in controls, a difference of 35
percentage points. The 2 x 2 table is [[18, 22], [32, 8]]. Fisher's exact test gives a small
p-value, about 0.0024 depending on the two-sided definition. The analysis
should not impute the missing case values as a low number, because the detection difference is
itself the signal; it should report detection and abundance separately.

**Answer 3 (family).** At rank 1 with 900 tests, q = 2 x 10^-4 x 900 = 0.18, not significant.
With 40 tests, q = 2 x 10^-4 x 40 = 0.008, significant. The 40-protein panel is legitimate only
if it was fixed before the data were examined. If it was assembled from the top hits, the
correction is circular and the q-value is invalid.

**Answer 4 (metabolite confidence).** (a) Level 2, putatively annotated compound. (b) Level 1,
identified compound. (c) Level 4, unknown feature. (d) Level 3, putatively characterized class.
The claim for level 2 is "a feature with this spectrum"; level 1 supports "this compound";
level 3 supports "this class"; level 4 supports only that the feature is distinguishable and
quantifiable.

**Answer 5 (critique).** Typical defects: minimum-value imputation with MNAR missingness, which
inflates effects; correction across a shortlist rather than the tested family, which invalidates
q-values; and batch confounded with group, which makes the effect unidentifiable. A fourth is
the absence of orthogonal validation. Each defect has a specific consequence: an inflated effect
size, an invalid significance claim, an uninterpretable comparison, and a non-replicating
candidate.

**Answer 6 (validation).** Two peptides per protein, chosen for uniqueness and detectability;
stable-isotope-labelled internal standards for each peptide; a calibration curve spanning the
expected range with at least six points; acceptance criteria of CV below 15% at each level and
accuracy within 20%; LLOQ defined as the lowest level meeting those criteria; sample size
sufficient to detect the discovery effect with 80% power at the validation precision.

**Answer 7 (ratio compression).** If a known 2-fold ratio is measured as 1.6, the compression
factor on the log scale is log2(1.6)/log2(2) = 0.678, so the observed log ratio is 68% of the
true log ratio. Correcting by dividing the observed log ratio by 0.678 assumes the compression
factor is constant across the dynamic range, which is an approximation; compression varies with
abundance and co-isolation.

**Answer 8 (analysis plan).** Missingness: diagnose by group; report detection as a separate
outcome; retain proteins detected in at least 70% of samples in at least one group; do not
impute for the primary test. Normalization: median, with sensitivity to quantile. Batch: three
batches as a covariate, with a check for collinearity. Family: all 1,200 proteins, BH. Go/no-go:
validate if at least three proteins pass q < 0.05 with a consistent direction and the effect is
at least 1.3-fold.

## 14. Interview questions

**1. Why is proteomics missingness different from a random missing value?**
Strong answer: it is usually MNAR, driven by abundance below detection; imputation changes the
effect, and detection itself can be the signal.
Weak answer: "missing values are common, so impute."

**2. What is the difference between peptide and protein FDR?**
Strong answer: peptide FDR is the expected false fraction among PSMs; protein FDR is among
inferred proteins; they differ because of shared peptides and aggregation.
Weak answer: treats them as the same number.

**3. When would you use DIA over DDA?**
Strong answer: when consistent quantification across many samples matters more than deep
identification; DIA fragments everything in windows.
Weak answer: "DIA is newer."

**4. What is ratio compression?**
Strong answer: co-isolated peptides contribute background to isobaric reporter channels,
shrinking ratios toward one; it understates effects.
Weak answer: "the ratios are noisy."

**5. How do you correct multiple testing in proteomics?**
Strong answer: BH across the tested family, with the family stated and pre-specified; a
pre-specified panel is a legitimate family, a post hoc shortlist is not.
Weak answer: "Bonferroni" without a family.

**6. What are the MSI identification levels?**
Strong answer: 1 standard plus orthogonal evidence, 2 library annotation, 3 class, 4 unknown;
each supports a different claim.
Weak answer: "identified or not identified."

**7. Why can a protein be detected in one group and not another?**
Strong answer: abundance near the detection limit, matrix effects, or true presence/absence;
the detection difference may be the biological signal.
Weak answer: "technical failure."

**8. What is protein inference?**
Strong answer: assigning shared peptides to protein groups; the parsimonious set is a model, and
group membership can change with the database.
Weak answer: "matching peptides to proteins" with no ambiguity.

**9. How would you validate a discovery hit?**
Strong answer: targeted PRM or immunoassay with calibration, on an independent cohort, with
pre-specified acceptance criteria.
Weak answer: "run it again on the same platform."

**10. What is the difference between LOD and LLOQ?**
Strong answer: detection versus quantification with acceptable precision; values between them
are not reportable numbers.
Weak answer: treats them as synonyms.

**11. Why do many protein biomarkers fail translation?**
Strong answer: small discovery cohorts, missingness treated as noise, batch confounding,
platform-specific effects, and no orthogonal validation.
Weak answer: "biology is complex."

**12. What would make you distrust a plasma proteomics hit list?**
Strong answer: imputation-driven effects, correction after selection, condition concentrated in
one batch, no detection analysis, and no validation plan.
Weak answer: "too many hits."

## 15. Key terms

- **Adduct** — ion formed with a salt or solvent.
- **Carryover** — contamination from a previous sample.
- **Chimeric spectrum** — spectrum containing fragments from multiple peptides.
- **DDA** — data-dependent acquisition.
- **DIA** — data-independent acquisition.
- **Decoy** — shuffled or reversed sequence for FDR estimation.
- **Fisher's exact test** — exact test for association in a contingency table.
- **HMDB** — Human Metabolome Database.
- **HUPO-PSI** — proteomics standards body.
- **InChI** — structure identifier for metabolites.
- **Ion suppression** — reduced ionization due to matrix.
- **Isobaric labelling** — TMT/iTRAQ multiplexing.
- **Isomer** — same mass, different structure.
- **LFQ** — label-free quantification.
- **LLOQ** — lower limit of quantification.
- **LOD** — limit of detection.
- **MaxLFQ** — ratio-based label-free normalization.
- **MIAPE** — minimum information about a proteomics experiment.
- **MNAR** — missing not at random.
- **MSI level** — metabolite identification confidence level.
- **m/z** — mass-to-charge ratio.
- **mzML** — standard spectra format.
- **PEP** — posterior error probability.
- **Peptide FDR** — false discovery rate among peptide-spectrum matches.
- **PRM** — parallel reaction monitoring.
- **Protein group** — proteins supported by shared peptides.
- **Protein inference** — assignment of peptides to proteins.
- **PSM** — peptide-spectrum match.
- **q-value** — FDR-adjusted significance for a feature.
- **PTM** — post-translational modification.
- **Ratio compression** — shrinkage of isobaric ratios.
- **Retention time** — elution time.
- **TMT** — tandem mass tag.
- **Target-decoy** — FDR estimation method.
- **XIC** — extracted ion chromatogram, the intensity trace for an ion.

## 16. Sources

Methods and evidence:

- Lazar, C. et al., "Accounting for the multiple natures of missing values in label-free quantitative proteomics data sets to compare imputation strategies," *Journal of Proteome Research* 15:1116-1125, 2016. https://doi.org/10.1021/acs.jproteome.5b00981 (accessed 25 October 2026).
- Harris, L. et al., "Evaluating proteomics imputation methods with improved criteria," *Journal of Proteome Research*, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC10949645/ (accessed 25 October 2026).
- Li, M., Smyth, G. K., "Neither random nor censored: estimating intensity-dependent probabilities for missing values in label-free proteomics," *Bioinformatics* 39:btad200, 2023. https://doi.org/10.1093/bioinformatics/btad200 (accessed 25 October 2026).
- Sumner, L. W. et al., "Proposed minimum reporting standards for chemical analysis," *Metabolomics* 3:211-221, 2007. https://doi.org/10.1007/s11306-007-0082-2 (accessed 25 October 2026).
- Cox, J., Mann, M., "MaxQuant enables high peptide identification rates, individualized p.p.b.-range mass accuracies and proteome-wide protein quantification," *Nature Biotechnology* 26:1367-1372, 2008. https://doi.org/10.1038/nbt.1511 (accessed 25 October 2026).

Standards and regulation:

- HUPO Proteomics Standards Initiative. https://www.psidev.info/ (accessed 25 October 2026).
- Metabolomics Standards Initiative. https://github.com/MSI-Metabolomics-Standards-Initiative (accessed 25 October 2026).
- FDA, Biomarker Qualification Program. https://www.fda.gov/drugs/biomarker-qualification-program/about-biomarkers-and-qualification (accessed 25 October 2026).
- ICH E16, Biomarkers Related to Drug or Biotechnology Product Development. https://database.ich.org/sites/default/files/E16_Guideline.pdf (accessed 25 October 2026).
- ISO 15189:2022. https://www.iso.org/standard/76677.html (accessed 25 October 2026).
- Regulation (EU) 2017/746. https://eur-lex.europa.eu/eli/reg/2017/746/oj (accessed 25 October 2026).

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`05_spatial-transcriptomics.md`](05_spatial-transcriptomics.md)
- Next topic: [`07_multi-omics-integration.md`](07_multi-omics-integration.md)
- Related topic: [`08_biomarker-discovery.md`](08_biomarker-discovery.md)
- Survey file: [`../../12_omics_precision_medicine.md`](../../12_omics_precision_medicine.md)
- Labs: [`../../labs/`](../../labs/)
