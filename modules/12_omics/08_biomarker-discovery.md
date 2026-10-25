# Biomarker Discovery: Discovery, Analytical Validation, Clinical Validation, Utility

> Module 12 — Omics and Precision Medicine
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: 01A, 04, 05, topics 03, 06, 07
> Version 2026.3

## 1. Why this topic exists

The word "biomarker" is used for four different claims, and the difference between them is the
difference between a research paper and a product. A discovery biomarker is a feature associated
with an outcome in one cohort. An analytically validated biomarker is a measurement that is
accurate, precise, and reproducible in a defined specimen on a defined platform. A clinically
validated biomarker is associated with the outcome in the intended population with a stated
performance. A biomarker with demonstrated clinical utility changes a decision and improves an
outcome. Most published biomarkers reach the first stage and never reach the second.

The attrition is not a mystery. Discovery studies are small, flexible, and analysed after the
data arrive; validation studies are large, pre-specified, and analysed blind. The transition from
one to the other eliminates the degrees of freedom that produced the discovery, and many
discoveries are degrees of freedom in disguise. A signature selected from 20,000 features in 80
samples has an enormous opportunity to fit noise, and cross-validation within the same cohort
does not remove that opportunity if the selection happened before the split.

The industry cost is a pipeline of candidates that consume validation budgets and fail. A
biomarker program that does not distinguish the four claims will treat a discovery association as
evidence of clinical value, fund a validation study on that basis, and discover at the end that
the effect was a batch, a spectrum difference, or an optimism artifact.

This topic covers the evidence ladder, the design of each stage, the statistical arithmetic that
determines whether a biomarker can work at a given prevalence, and the reporting standards that
make the evidence assessable. It follows the omics measurement topics and precedes drug
discovery (topic 09) and clinical genomics (topic 10).

### 1.1 The failure pattern

1. A candidate is discovered in a case-control cohort of 80 samples with 20,000 features.
2. Internal cross-validation gives an AUC of 0.86.
3. The model is locked and tested in a validation cohort of 120 samples.
4. The validation AUC is 0.71, and the effect is smaller but still significant.
5. A clinical champion asks whether the test should be used.
6. Nobody has calculated the positive predictive value at the intended prevalence, which is 3%.
7. At that prevalence, the PPV is 12%, and the test generates eight false positives for every
   true one.
8. The program discovers this after building a laboratory assay.

Steps 6 and 7 take an hour and should precede step 1. The prevalence arithmetic determines
whether any test with this sensitivity and specificity can be useful, independent of how good the
AUC looks.

### 1.2 Who is blocked when this goes wrong

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Biomarker lead | A program year and a validation budget | Depends on the discovery evidence |
| Clinical champion | Credibility after advocating a test | Cannot see the prevalence arithmetic |
| Laboratory director | A validated assay with no clinical demand | Receives a candidate, not a use case |
| Payer | Spend on a test without utility evidence | Sees the claim, not the decision impact |
| Regulator | A submission that cannot demonstrate utility | Receives a device, not an outcome study |
| Patient | A false positive and the follow-up it triggers | Has no role in the design |

### 1.3 The industry question

> Is this molecular signal a validated biomarker, a causal target, or an artifact of batch,
> ancestry, and tissue composition?

The question is the module's organizing question, and this topic is where it is answered most
directly. The answer requires four separate pieces of evidence, and a program that has three of
them has a candidate, not a biomarker.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** discovery, analytical validation, clinical validation, and utility, and name
   the study design that establishes each.
2. **Compute** sensitivity, specificity, predictive values, and net benefit at a stated
   prevalence.
3. **Explain** why AUC is prevalence-independent and PPV is not, and what follows for a business
   case.
4. **Design** a discovery study with pre-specification, blinding, and a locked model.
5. **Quantify** optimism and correct for it with internal validation.
6. **Write** an analytical validation plan with accuracy, precision, range, and limit of
   detection.
7. **Assess** a published biomarker claim against REMARK, STARD, and TRIPOD.
8. **Decide** whether a candidate should advance, using pre-specified go/no-go criteria.

## 3. The core mental model

A biomarker is a chain of four evidence links. Each link has its own design, its own failure
modes, and its own reviewers.

```text
  Biological hypothesis
        |
        v
  Discovery: association in a cohort
        |     design: case-control or cohort, pre-specified, internally validated
        |     failure: overfitting, batch, spectrum, multiplicity
        v
  Analytical validation: the assay measures reliably
        |     design: replicates, calibration, interference, stability
        |     failure: imprecision, matrix effects, drift
        v
  Clinical validation: the association holds in the intended population
        |     design: independent prospective cohort, blinded, pre-specified cutoff
        |     failure: spectrum bias, population differences, recalibration
        v
  Clinical utility: using it improves decisions or outcomes
        |     design: randomised or decision-impact study
        |     failure: no change in management, harm from false positives
        v
  Adoption: reimbursement, guidelines, workflow
```

### 3.1 The four claims and their evidence

| Claim | Question | Design | Typical metric |
|---|---|---|---|
| Discovery | Is there an association? | Case-control or cohort, exploratory | Effect size, q-value |
| Analytical validation | Does the assay measure reliably? | Replicates, dilution series, interference | CV, bias, LoD, LoQ |
| Clinical validation | Does it discriminate in the intended population? | Independent cohort, blinded | AUC, sensitivity, specificity, calibration |
| Clinical utility | Does using it improve outcomes? | Randomised or decision-impact study | Net benefit, decision change, outcome |

A program can hold a strong discovery and a weak assay, or a strong assay and no clinical
validation. The four claims are independent, and the weakest link determines what can be said.

### 3.2 The prevalence arithmetic

AUC is a property of the score's ranking and is independent of prevalence. Sensitivity and
specificity are properties of the cutoff and are also prevalence-independent. Predictive values
are not: they depend on how common the outcome is in the tested population.

```text
PPV = sensitivity x prevalence / (sensitivity x prevalence + (1-specificity) x (1-prevalence))
NPV = specificity x (1-prevalence) / (specificity x (1-prevalence) + (1-sensitivity) x prevalence)
```

The consequence is that the same test can be useful in a high-prevalence setting and harmful in
a low-prevalence setting. A sepsis biomarker tested in an intensive care unit and in an
outpatient clinic is two different products with the same assay.

### 3.3 Optimism and the lock

Every data-dependent choice inflates apparent performance. The choices include feature selection,
cutoff selection, model form, and the decision to report the model at all. Internal
cross-validation estimates performance for a fixed modelling procedure, not for the procedure
plus the analyst's choices.

The remedy is a lock: pre-specify the procedure, fit it once, and evaluate it once on data that
were not used for any choice. The lock is procedural, not statistical. A model that is re-fitted
after seeing validation results is no longer the locked model.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Biomarker | A measured characteristic indicating a biological process or outcome | Used for any correlated feature | Definition |
| Discovery | First identification of an association | Treated as validation | Stage |
| Analytical validation | Establishing assay performance | Confused with clinical validation | Stage |
| Clinical validation | Association with outcome in the intended population | Confused with utility | Stage |
| Clinical utility | Evidence that use improves decisions or outcomes | Assumed from validation | Stage |
| Context of use (COU) | The specified use of a biomarker in drug development | Left vague | Statement |
| BEST category | Biomarker category in the FDA framework | Confused with COU | Category |
| Sensitivity | True positive fraction at a cutoff | Confused with PPV | Proportion |
| Specificity | True negative fraction at a cutoff | Confused with NPV | Proportion |
| PPV | Probability of disease given a positive test | Assumed constant | Proportion |
| NPV | Probability of no disease given a negative test | Assumed constant | Proportion |
| AUC | Area under the ROC curve | Treated as clinical utility | 0.5-1.0 |
| ROC curve | Sensitivity against 1-specificity across cutoffs | Used without prevalence context | Curve |
| Calibration | Agreement of predicted and observed risk | Confused with discrimination | Slope, intercept |
| Calibration slope | Regression slope of observed on predicted | Ignored | Dimensionless |
| Net benefit | Benefit minus harm in decision terms | Confused with AUC | Probability units |
| Decision curve | Net benefit across thresholds | Rarely reported | Curve |
| Spectrum bias | Performance differs by case mix | Ignored | Bias |
| Optimism | Inflation from data-dependent choices | Confused with overfitting only | AUC units |
| Internal validation | Performance within the discovery cohort | Treated as external | Stage |
| External validation | Performance in an independent cohort | Confused with replication | Stage |
| Replication | Same finding in new data | Confused with validation of a locked model | Stage |
| Locked model | Fully specified model evaluated once | Re-fitted after validation | Artifact |
| Pre-specification | Analysis fixed before data | Confused with a plan written after | Practice |
| Cutoff | Threshold converting a score to a decision | Selected on the validation set | Value |
| LoD | Limit of detection | Confused with LoQ | Concentration |
| LoQ | Limit of quantification | Confused with LoD | Concentration |
| Precision | Closeness of repeated measurements | Confused with accuracy | CV |
| Accuracy | Closeness to the true value | Confused with precision | Bias |
| Interference | Effect of another substance on the assay | Ignored | Effect |
| Matrix effect | Specimen-specific assay behaviour | Ignored | Effect |
| REMARK | Reporting guideline for tumour marker prognostic studies | Assumed enforced | Guideline |
| STARD | Reporting guideline for diagnostic accuracy studies | Assumed enforced | Guideline |
| TRIPOD | Reporting guideline for prediction models | Assumed enforced | Guideline |
| IVD | In vitro diagnostic device | Confused with LDT | Regulatory class |
| LDT | Laboratory-developed test | Assumed unregulated | Regulatory class |
| Companion diagnostic | Test required for a therapeutic | Confused with any biomarker | Regulatory class |

Three distinctions carry most of the weight.

**Discrimination is not calibration.** A model can rank patients well (high AUC) and predict
risks that are systematically too high. Calibration matters wherever the score is used as a
probability, which is most clinical uses.

**A cutoff chosen on the validation set is not validated.** Selecting the cutoff that maximizes
validation performance converts the validation set into a training set. The cutoff must be
pre-specified, or the performance must be reported across all cutoffs with the selection
accounted for.

**Analytical validation is not a formality.** An assay with a 25% coefficient of variation cannot
support a 1.2-fold decision rule, regardless of the discovery effect. The assay's precision sets
the smallest detectable difference, and the smallest detectable difference sets the clinical
claim.

## 5. The mechanics, step by step

### Step 1. Define the decision the biomarker changes

**Input:** the clinical or program decision.

**Output:** a statement of who decides what, when, with what alternatives, and what would change.

**Rationale:** A biomarker that does not change a decision is a measurement, not a tool. The
decision determines the intended population, the prevalence, the acceptable false-positive rate,
and the evidence needed. "Predict response" is not a decision; "decide whether to start
immunotherapy in patients with no targetable mutation" is.

### Step 2. Define the endpoint and the comparator

**Input:** the decision and the clinical pathway.

**Output:** a primary endpoint, a comparator, and a time horizon.

**Rationale:** A biomarker that predicts a surrogate endpoint (response rate) may not predict the
outcome that matters (survival). The comparator is the current decision rule, which may be
clinical judgement rather than a test. Without a comparator, the biomarker's incremental value
cannot be estimated.

### Step 3. Specify the specimen, assay, and measurement procedure

**Input:** the decision context.

**Output:** an assay specification: specimen type, collection, processing, platform, and storage.

**Rationale:** A biomarker measured in fresh tissue may not be measurable in the FFPE specimens
available at the point of decision. Specimen feasibility is a design constraint, not an
implementation detail. The assay specification must match what is available in the intended use.

### Step 4. Design the discovery cohort

**Input:** the endpoint and available samples.

**Output:** a discovery cohort with a stated sampling frame.

**Rationale:** Case-control sampling is efficient for discovery and biased for absolute risk. A
cohort sampled by outcome cannot estimate prevalence, so predictive values cannot be computed
from it. The sampling frame must be recorded, because it determines which claims are supported.

The discovery cohort should be large enough that the expected number of events per candidate
feature is defensible. A common, rough guideline is at least 10 events per parameter in the final
model; with 20,000 features screened, the screening step must be separated from the modelling
step and validated.

### Step 5. Pre-specify the analysis

**Input:** the design.

**Output:** a written analysis plan: feature handling, model, validation, metric, cutoff rule.

**Rationale:** The plan is the difference between a discovery and a fishing expedition. It does
not need to specify every detail, but it must specify the choices that would otherwise be made
after seeing the data: normalization, filtering, model class, number of features, and how the
cutoff will be chosen.

### Step 6. Run the discovery analysis with the selection inside the folds

**Input:** discovery data.

**Output:** out-of-fold predictions and a candidate model.

**Rationale:** Feature selection, tuning, and cutoff selection must occur inside the training
fold. The out-of-fold predictions are the only honest estimate of performance for the procedure.
In-sample performance is a description of the fit, not a prediction.

### Step 7. Quantify optimism explicitly

**Input:** discovery model and resampling.

**Output:** an optimism-corrected performance estimate.

**Rationale:** Bootstrap or cross-validation gives an estimate of optimism, the average
difference between in-sample and out-of-sample performance. Reporting the corrected estimate
rather than the apparent one is a basic requirement of a credible discovery. An optimism of 0.05
to 0.10 AUC is common in omics studies and is often larger than the effect being claimed.

### Step 8. Lock the model

**Input:** the fitted procedure.

**Output:** a versioned artifact: preprocessing parameters, features, weights, cutoff, and code.

**Rationale:** The lock is what makes the validation meaningful. A locked model can be shipped to
a validation cohort and evaluated once. An unlocked model is re-fitted at each evaluation, and
each evaluation is a new analysis with a new opportunity to overfit.

### Step 9. Perform analytical validation

**Input:** the locked assay.

**Output:** a validation report with accuracy, precision, range, LoD, LoQ, interference, and
stability.

**Rationale:** Analytical validation establishes that the measurement is trustworthy. For an
immunoassay, that means replicates at multiple concentrations, a dilution linearity study, an
interference study (haemolysis, lipaemia, biotin), and stability studies. For a sequencing-based
assay, it means reproducibility across runs, input amounts, and operators, with a stated limit of
detection in the relevant specimen.

### Step 10. Perform clinical validation in the intended population

**Input:** the locked model and an independent cohort.

**Output:** discrimination, calibration, and predictive values at the intended prevalence.

**Rationale:** The validation cohort must resemble the intended-use population in spectrum,
prevalence, and case mix. A validation cohort of healthy controls and severe cases will
overestimate performance. Report AUC with an interval, calibration slope and intercept, and
sensitivity and specificity at the pre-specified cutoff.

### Step 11. Establish utility

**Input:** the validated model.

**Output:** evidence that use changes decisions or outcomes.

**Rationale:** Utility is a causal question, and the strongest design is a randomised trial or a
stepped-wedge implementation study. A decision-impact study, which measures whether clinicians
change management when the test is available, is weaker but often feasible. Without utility
evidence, a validated test is a test that could be used, not one that should be.

### Step 12. Map the regulatory pathway

**Input:** the intended use and the jurisdiction.

**Output:** a regulatory classification and a plan.

**Rationale:** A test that informs a therapeutic decision may be a companion diagnostic; a
laboratory-developed test has a different pathway from a marketed kit; a biomarker used in drug
development may qualify through the FDA Biomarker Qualification Program. The pathway determines
the evidence package and the timeline.

### Step 13. Build the economic case

**Input:** the validated performance and the decision.

**Output:** a cost-consequence or cost-effectiveness analysis.

**Rationale:** A test that costs less than the intervention it avoids can be cost-saving even at
modest performance; a test that adds a confirmatory step can increase cost. The economics depend
on prevalence, PPV, and the cost of the downstream pathway, not on AUC alone.

### Step 14. Report against a standard

**Input:** everything above.

**Output:** a report that a reviewer can assess.

**Rationale:** REMARK covers tumour marker prognostic studies; STARD covers diagnostic accuracy;
TRIPOD covers prediction models. Each requires the population, the assay, the missing data, the
performance with intervals, and the limitations. Reporting against a standard is not
bureaucracy; it is the checklist a reviewer uses to find the gaps.

### Step 15. Define the decision threshold before the cutoff

**Input:** the clinical decision and its harm-to-benefit ratio.

**Output:** a threshold probability and a net-benefit analysis.

**Rationale:** The cutoff is a property of the test; the threshold is a property of the decision.
A clinician who treats when the predicted risk exceeds 20% has a threshold of 0.20, and the test
should be evaluated at that threshold, not at the Youden-optimal cutoff. Decision curve analysis
shows whether the test adds net benefit across the range of thresholds a reasonable clinician
might use.

### Step 16. Plan specimen and data management

**Input:** the intended workflow.

**Output:** a specimen handling plan and a data management plan.

**Rationale:** A biomarker measured in a research biobank may not survive the clinical workflow:
collection tubes, time to processing, freeze-thaw cycles, and storage duration all change
measurements. The plan should specify the collection device, the processing window, the storage
conditions, and the acceptable number of freeze-thaw cycles, and the analytical validation should
test them.

Data management includes the sample identity chain, the assay batch record, and the versioned
analysis code. A biomarker result that cannot be traced from specimen to report is not a clinical
measurement.

### Step 17. Arrange independent review

**Input:** the evidence dossier.

**Output:** a review by someone who did not build the model.

**Rationale:** The team that built a model knows its assumptions and unconsciously reads results
in their light. An independent reviewer with access to the locked model and the validation data
can check the pre-specification, the cutoff, the calibration, and the arithmetic. The review is
cheap and catches errors that internal review misses.

### Analytical validation parameters

| Parameter | Question | Typical design | Acceptance criterion |
|---|---|---|---|
| Precision | How repeatable is the measurement? | 20 replicates, 3 levels, 5 days | CV below 15-20% |
| Accuracy | How close to truth? | Reference material or recovery | Bias within 15-20% |
| Linearity | Does dilution behave? | 7 levels across the range | Slope 0.9-1.1, R-squared above 0.98 |
| LoD | What can be detected? | Blank plus 3 SD, 20 replicates | Stated concentration |
| LoQ | What can be quantified? | Lowest level meeting precision | Stated concentration |
| Interference | What confounds it? | Haemolysis, lipaemia, biotin, drugs | Recovery within 10% |
| Carryover | Does it contaminate? | High followed by blank | Below LoD |
| Stability | Does it survive handling? | Time, temperature, freeze-thaw | Within acceptance at each condition |
| Lot-to-lot | Does it change with reagents? | Multiple lots | Within precision claim |

### Reading the analytic validation table

The parameters are not independent. A test with excellent precision at high concentration and
poor precision at the decision threshold has a precision problem where it matters, and the
overall CV hides it. Report precision at the clinical decision point, not only across the range.
Similarly, a LoD below the threshold is not useful if the LoQ is above it: the test can detect
what it cannot quantify.

### Branch points

**Branch A — the discovery cohort is case-control.** Prevalence cannot be estimated, so
predictive values cannot be computed from it. Clinical validation must use a cohort or a
representative sample.

**Branch B — the assay cannot be run on the intended specimen.** The biomarker is not deployable.
Either change the specimen or change the intended use.

**Branch C — the validation performance is lower but the interval excludes the discovery
estimate.** The model does not transport. Recalibration may help; re-fitting requires a new
validation.

### Decision table: stage to design

| Stage | Design | Primary metric | Minimum evidence |
|---|---|---|---|
| Discovery | Case-control or cohort, pre-specified | Out-of-fold AUC, effect size | Optimism-corrected performance |
| Analytical validation | Replicates, dilutions, interference | CV, bias, LoD, LoQ | Pre-specified acceptance criteria |
| Clinical validation | Independent cohort, blinded | AUC, calibration, sens/spec | Interval on each metric |
| Utility | Randomised or decision-impact | Net benefit, decision change, outcome | Comparator and time horizon |
| Regulatory | Device or qualification pathway | Analytical and clinical performance | Submission package |
| Post-market | Surveillance | Drift, subgroup performance | Monitoring plan |

### Edge cases

**Spectrum bias.** Diagnostic: the validation cohort has a different case mix from the intended
population, with more severe cases and healthier controls. Consequence: overestimated
performance. Mitigation: a representative cohort; report performance by subgroup.

**Prevalence mismatch.** Diagnostic: the validation prevalence differs from the intended setting.
Consequence: predictive values differ even with identical sensitivity and specificity.
Mitigation: recompute predictive values at the intended prevalence, with intervals.

**Batch effects.** Diagnostic: the assay signal differs by processing batch. Consequence: a
biomarker that is partly a batch marker. Mitigation: balanced batches; batch as a covariate;
reproducibility across reagent lots.

**Missing data.** Diagnostic: the biomarker is missing more often in one group. Consequence:
complete-case analysis selects a population. Mitigation: report missingness by group; model it.

**Class imbalance.** Diagnostic: few events relative to features. Consequence: overfitting and
unstable selection. Mitigation: penalization, fewer features, or a larger cohort.

**Optimism.** Diagnostic: in-sample performance far exceeds cross-validated performance.
Consequence: an inflated claim. Mitigation: report the corrected estimate and the correction
method.

**Cutoff selection on validation.** Diagnostic: the reported cutoff differs from the
pre-specified one. Consequence: the validation is partly a training set. Mitigation: pre-specify
the cutoff, or report the full ROC.

**Assay drift.** Diagnostic: control materials trend over time. Consequence: a moving decision
threshold. Mitigation: external quality assessment, lot-to-lot studies, monitoring.

**Matrix change.** Diagnostic: the assay performs differently in serum versus plasma.
Consequence: validation does not transfer to the other specimen. Mitigation: validate the matrix
actually used.

**Competing risks.** Diagnostic: the outcome can be pre-empted by death from another cause.
Consequence: a model that predicts the wrong event. Mitigation: cause-specific or
sub-distribution models.

**Specimen timing.** Diagnostic: the biomarker is measured at a different time point from the
decision. Consequence: the test cannot inform the decision. Mitigation: align collection with the
decision point and validate at that time.

**Overlapping endpoints.** Diagnostic: multiple endpoints tested and the best reported.
Consequence: multiplicity. Mitigation: one primary endpoint; correct the rest.

The four stages also have different reviewers. Discovery is reviewed by scientists, analytical
validation by laboratory accreditors, clinical validation by clinical journals and regulators,
and utility by payers and guideline committees. A program that has satisfied the first reviewer
has not satisfied the fourth, and the evidence packages are not interchangeable.

### What the procedure produces

The output is a biomarker evidence dossier: the decision and intended use, the discovery design
and corrected performance, the locked model, the analytical validation report, the clinical
validation with intervals and calibration, the utility evidence, the regulatory classification,
and the economic case. The dossier is what distinguishes a candidate from a product.

## 6. Worked example

A program develops a blood biomarker to identify patients with suspected infection who will
progress to septic shock. The intended use is triage in an emergency department, where the
prevalence of progression is 5%.

### 6.1 Discovery

The discovery cohort has 300 patients: 100 progressors and 200 non-progressors, sampled
case-control. A 20-protein panel is measured. The pre-specified analysis selects features inside
cross-validation and fits a logistic model.

Apparent AUC in the discovery cohort is 0.85. Ten-fold cross-validation gives 0.78, so the
optimism estimate is:

```text
optimism = 0.85 - 0.78 = 0.07 AUC
```

The program reports 0.78 as the discovery performance, not 0.85. The 0.07 is the average
inflation from selection and fitting, and it is larger than many published biomarker effects.

### 6.2 The prevalence arithmetic

The locked model, at its pre-specified cutoff, has sensitivity 0.85 and specificity 0.80 in the
discovery cohort. At the intended prevalence of 5%:

```text
PPV = (0.85 x 0.05) / (0.85 x 0.05 + 0.20 x 0.95)
    = 0.0425 / (0.0425 + 0.190)
    = 0.183

NPV = (0.80 x 0.95) / (0.80 x 0.95 + 0.15 x 0.05)
    = 0.760 / (0.760 + 0.0075)
    = 0.990
```

At 5% prevalence, 18.3% of positive tests are true positives, and 99.0% of negative tests are
true negatives. The test is good at ruling out and poor at ruling in.

At a 30% prevalence, as in a high-risk subgroup:

```text
PPV = (0.85 x 0.30) / (0.85 x 0.30 + 0.20 x 0.70)
    = 0.255 / (0.255 + 0.140)
    = 0.646

NPV = (0.80 x 0.70) / (0.80 x 0.70 + 0.15 x 0.30)
    = 0.560 / (0.560 + 0.045)
    = 0.926
```

The same assay has a PPV of 64.6% in the high-risk subgroup. The product is different, and the
regulatory claim must specify the population.

### 6.3 Net benefit

Net benefit at a decision threshold p_t is:

```text
NB = sensitivity x prevalence - (1 - specificity) x (1 - prevalence) x p_t / (1 - p_t)
```

At a threshold of 0.20 (treat if predicted risk is at least 20%) and 5% prevalence:

```text
NB = 0.85 x 0.05 - 0.20 x 0.95 x 0.25
   = 0.0425 - 0.0475
   = -0.005
```

The net benefit is negative. At this prevalence and threshold, using the test is worse than
treating everyone or treating no one. At 30% prevalence:

```text
NB = 0.85 x 0.30 - 0.20 x 0.70 x 0.25
   = 0.255 - 0.035
   = 0.220
```

The net benefit is strongly positive. The same test, the same cutoff, and the same threshold
produce opposite decisions depending on prevalence. This is the arithmetic that a business case
must contain and usually omits.

### 6.4 Clinical validation

The validation cohort is a prospective emergency department sample of 400 patients with
suspected infection, prevalence 12%. The locked model gives:

```text
AUC = 0.71 (95% CI 0.64-0.78)
sensitivity = 0.79, specificity = 0.74 at the pre-specified cutoff
```

The AUC is lower than the discovery estimate of 0.78, and the interval includes values from 0.64
to 0.78, so the discovery estimate is at the edge of the interval. The performance drop is
consistent with a modest optimism and a different case mix.

At 12% prevalence:

```text
PPV = (0.79 x 0.12) / (0.79 x 0.12 + 0.26 x 0.88) = 0.0948 / (0.0948 + 0.2288) = 0.293
NPV = (0.74 x 0.88) / (0.74 x 0.88 + 0.21 x 0.12) = 0.6512 / (0.6512 + 0.0252) = 0.963
```

The PPV is 29.3%. Three of four positives are false, and each triggers a follow-up pathway.

### 6.5 Diagnostic 1: calibration

The calibration slope of the locked model in validation is 0.72, with an intercept of 0.31. A
slope below 1 means the predictions are too extreme: high risks are too high and low risks are
too low. The intercept of 0.31 means the predictions are systematically low on the log-odds
scale.

Recalibrating the model in the validation cohort raises the slope to 0.95 and the intercept to
0.02, and changes the operating characteristics:

```text
before recalibration: sensitivity 0.79, specificity 0.74
after recalibration:  sensitivity 0.81, specificity 0.76
```

The discrimination is unchanged, but the probabilities are now usable. A program that reports
only AUC would have missed this entirely.

### 6.6 Diagnostic 2: the cutoff

The validation report used a cutoff chosen to maximize the Youden index in the validation cohort,
which produced a sensitivity of 0.86 and a specificity of 0.68. The pre-specified cutoff gave
0.79 and 0.74.

The difference is not large, but the practice is invalid: the cutoff was selected on the data used
to report performance, so the reported performance is optimistic by an amount that depends on the
number of cutoffs considered. With 400 patients, the optimism from cutoff selection is small but
nonzero, and with 100 patients it can be substantial.

The correct report uses the pre-specified cutoff and shows the full ROC so a reader can see the
alternatives.

### 6.7 Vary one assumption: prevalence doubles

Suppose the emergency department changes its triage criteria and the prevalence of progression
rises from 5% to 10%. Sensitivity and specificity are unchanged. The PPV becomes:

```text
PPV = (0.85 x 0.10) / (0.85 x 0.10 + 0.20 x 0.90) = 0.085 / (0.085 + 0.18) = 0.321
```

The PPV rises from 18.3% to 32.1%, and the net benefit at a 0.20 threshold becomes:

```text
NB = 0.85 x 0.10 - 0.20 x 0.90 x 0.25 = 0.085 - 0.045 = 0.040
```

The sign flips from negative to positive. A change in the population, not the assay, turns a
harmful test into a useful one. This is why the intended-use population is part of the
regulatory claim and not a marketing detail.

### 6.8 What the dossier says

> Intended use: triage of suspected infection in the emergency department, prevalence 5%.
> Discovery: 300 case-control, 20-protein panel, out-of-fold AUC 0.78 (apparent 0.85, optimism
> 0.07). Model locked with preprocessing, features, weights, and cutoff. Analytical validation:
> CV 8% at 50 pg/mL, LoD 12 pg/mL, no interference from haemolysis at 1 g/L. Clinical
> validation: 400 prospective patients, AUC 0.71 (0.64-0.78), calibration slope 0.72, recalibrated
> to 0.95. At 5% prevalence, PPV 18.3%, NPV 99.0%, net benefit -0.005 at a 0.20 threshold.
> Conclusion: the assay is analytically valid and clinically associated, but utility at the
> intended prevalence is not demonstrated. A high-prevalence subgroup (30%) shows PPV 64.6% and
> net benefit 0.22, and would be the appropriate intended use for a revised program.

That dossier supports a decision: do not launch in the low-prevalence setting; consider a
restricted claim. It is more valuable than a headline AUC.

## 7. Code, query, or calculation

The script computes diagnostic metrics, net benefit, a bootstrap AUC interval, and an
optimism-corrected estimate.

### 7.1 Sensitivity, specificity, predictive values

```python
def diagnostic_metrics(sens, spec, prevalence):
    tp = sens * prevalence
    fn = (1 - sens) * prevalence
    fp = (1 - spec) * (1 - prevalence)
    tn = spec * (1 - prevalence)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "ppv": ppv, "npv": npv}

for prev in (0.05, 0.12, 0.30):
    m = diagnostic_metrics(0.85, 0.80, prev)
    print(f"prev={prev:.2f} PPV={m['ppv']:.3f} NPV={m['npv']:.3f}")
```

Expected output:

```text
prev=0.05 PPV=0.183 NPV=0.990
prev=0.12 PPV=0.367 NPV=0.975
prev=0.30 PPV=0.646 NPV=0.926
```

The PPV rises from 18.3% to 64.6% across prevalences, with identical sensitivity and
specificity. Nothing about the test changed.

### 7.2 Net benefit

```python
def net_benefit(sens, spec, prevalence, threshold):
    return sens * prevalence - (1 - spec) * (1 - prevalence) * threshold / (1 - threshold)

for prev in (0.05, 0.10, 0.30):
    nb = net_benefit(0.85, 0.80, prev, 0.20)
    print(f"prev={prev:.2f} threshold=0.20 net benefit={nb:.4f}")
print("treat all:", round(0.05 - (1 - 0.05) * 0.25, 4))
print("treat none:", 0.0)
```

Expected output:

```text
prev=0.05 threshold=0.20 net benefit=-0.0050
prev=0.10 threshold=0.20 net benefit=0.0400
prev=0.30 threshold=0.20 net benefit=0.2200
treat all: -0.1875
treat none: 0.0
```

At 5% prevalence the test has negative net benefit, but it is still better than treating
everyone. The comparison that matters is against the best default strategy, not against zero.

### 7.3 Bootstrap AUC interval

```python
import random

random.seed(3)
pos = [random.gauss(1.0, 1.0) for _ in range(40)]
neg = [random.gauss(0.0, 1.0) for _ in range(60)]

def auc(pos, neg):
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / (len(pos) * len(neg))

point = auc(pos, neg)
print("AUC:", round(point, 3))

vals = []
for _ in range(2000):
    p = [random.choice(pos) for _ in pos]
    n = [random.choice(neg) for _ in neg]
    vals.append(auc(p, n))
vals.sort()
print("95% CI:", round(vals[50], 3), round(vals[1950], 3))
```

Expected output:

```text
AUC: 0.72
95% CI: 0.609 0.82
```

The interval spans 0.21 AUC. A validation cohort of 100 patients cannot distinguish a good test
from a moderate one, which is why reported AUCs without intervals are not evidence of
performance.

### 7.4 Optimism correction

```python
apparent = 0.85
cv_values = [0.76, 0.79, 0.77, 0.80, 0.78, 0.77, 0.79, 0.78, 0.76, 0.80]
mean_cv = sum(cv_values) / len(cv_values)
optimism = apparent - mean_cv
corrected = apparent - optimism
print("apparent:", apparent)
print("mean CV:", round(mean_cv, 3))
print("optimism:", round(optimism, 3))
print("corrected:", round(corrected, 3))
```

Expected output:

```text
apparent: 0.85
mean CV: 0.78
optimism: 0.07
corrected: 0.78
```

The corrected estimate equals the mean cross-validated value, which is the definition used here.
Other definitions exist, and the method must be stated.

### 7.5 BH for the discovery screen

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

pvals = [1.2e-7, 4.5e-6, 3.3e-5, 2.1e-4, 9.0e-4]
q = bh(pvals, 20000)
for p, qq in zip(pvals, q):
    print(f"p={p:.1e} q={qq:.3e}")
print("significant at q<0.05:", sum(1 for x in q if x < 0.05))
```

Expected output:

```text
p=1.2e-07 q=2.400e-03
p=4.5e-06 q=4.500e-02
p=3.3e-05 q=2.200e-01
p=2.1e-04 q=1.000e+00
p=9.0e-04 q=1.000e+00
significant at q<0.05: 2
```

Only two of the five features survive correction across 20,000 tests. A discovery screen that
reports the top five without correction would carry three expected false leads into validation.

### 7.6 Assumptions and limits of this code

- The metrics assume a binary outcome and a single cutoff; time-to-event outcomes need
  time-dependent definitions.
- The bootstrap assumes independent patients; clustered sampling requires a cluster bootstrap.
- The optimism correction uses one definition; others (for example, Harrell's) give different
  values.
- Net benefit assumes the threshold reflects the harm-to-benefit ratio, which is a clinical
  judgment.
- Nothing here performs the modelling that produces the scores.

## 8. Data, standards, and regulatory context

### 8.1 Reporting standards

- **REMARK** (REporting recommendations for tumour MARKer prognostic studies) specifies the
  specimen, assay, patient selection, and analysis reporting for prognostic markers.
- **STARD** (STAndards for the Reporting of Diagnostic accuracy studies) specifies the flow of
  participants, the index test, the reference standard, and the results, including indeterminate
  results.
- **TRIPOD** specifies the reporting of prediction models, including calibration and validation;
  TRIPOD+AI extends it to machine learning.
- **CONSORT** applies when the biomarker is evaluated in a randomised trial.

These are reporting guidelines, not regulations. Journals enforce them unevenly, and a paper can
be technically compliant and analytically weak.

### 8.2 Analytical validation standards

- **CLIA** requires performance specifications for accuracy, precision, analytical sensitivity
  and specificity, reportable range, and reference intervals.
- **CAP** accreditation adds proficiency testing and document control.
- **ISO 15189:2022** provides the international laboratory quality framework.
- **CLSI** publishes consensus protocols for precision, linearity, detection limits, and
  interference testing.
- **IVDR 2017/746** requires analytical and clinical performance evaluation for CE marking.

The analytical validation report should state the acceptance criteria before the study, not after.

### 8.3 Regulatory pathways

**FDA device pathway.** A biomarker test marketed as an IVD requires a premarket submission. A
companion diagnostic, required for a therapeutic, is reviewed with the therapeutic and appears on
the FDA's authorized companion diagnostic list. In 2025, authorizations included FoundationOne
CDx for tovorafenib in pediatric low-grade glioma and Guardant360 CDx for imlunestrant in
ESR1-mutant breast cancer, among others.

**Laboratory-developed tests.** The 2024 FDA rule that would have ended enforcement discretion
was vacated by the Eastern District of Texas on 31 March 2025, and the FDA's September 2025
"Implementation of Vacatur" rule restored the prior text. LDTs are currently under enforcement
discretion, with the matter remanded. A program should not assume either device regulation or
permanent discretion.

**Biomarker qualification.** The FDA Biomarker Qualification Program qualifies a biomarker for a
defined context of use through a three-stage process (Letter of Intent, Qualification Plan, Full
Qualification Package). Qualification attaches to the biomarker, not to a specific assay, and a
qualified biomarker is publicly available for use within its context of use.

**ICH E16** defines the context, structure, and format of qualification submissions for genomic
biomarkers. The same principles apply to proteomic and metabolomic biomarkers: the context of use
must specify the biomarker category, the intended use, and the critical parameters.

### 8.4 What the standards do not settle

- What effect size is clinically meaningful.
- Which prevalence applies to the intended setting.
- Whether a surrogate endpoint is acceptable.
- How much uncertainty is tolerable in a clinical decision.
- Whether a test's benefits justify its harms.

These are clinical and policy judgments, and they determine whether a statistically valid
biomarker is a useful one.

## 9. Industry practice and case evidence

### 9.1 The attrition evidence

The published record on biomarker attrition is consistent across reviews: the majority of
candidate biomarkers reported in discovery studies are not validated in independent cohorts, and
a smaller fraction demonstrate clinical utility. The reasons cited are small discovery cohorts,
lack of pre-specification, publication bias, and the absence of analytical validation before
clinical testing.

Ioannidis (2005) formalized the arithmetic: for a research field with many tested associations,
small studies, small effect sizes, and flexible designs, the probability that a reported positive
finding is true can be low even when the p-value is below 0.05. Biomarker discovery is close to
the worst case on every dimension.

### 9.2 The reproducibility evidence

Begley and Ellis (2012) reported that Amgen could confirm only 6 of 53 landmark preclinical
cancer papers. Prinz, Schlange and Asadullah (2011) reported that of 67 target-validation
projects at Bayer, only 14 matched published findings. These are preclinical target studies, not
biomarker validation studies, but the mechanisms — unblinded selection, single-platform
confirmation, and selective reporting — are the same ones that produce non-replicating
biomarkers.

The frequently cited claim that "90% of preclinical results cannot be reproduced" comes from the
47/53 figure in the first study, not from a general survey. Both are self-audits with undisclosed
study lists, so they indicate a problem rather than quantify it precisely.

### 9.3 What industry does

Documented practice:

- Discovery is run in academic or internal cohorts with pre-specified analysis plans, increasingly
  with a locked model before validation.
- Analytical validation is run before clinical validation for any assay intended for use, because
  an assay with poor precision cannot support a clinical claim.
- Clinical validation uses independent, often prospective cohorts with the intended case mix.
- Utility evidence is collected through decision-impact studies or pragmatic trials, and is the
  stage most often skipped.
- Regulatory strategy is scoped at the start, because a companion diagnostic pathway changes the
  trial design.

Inference: the most common program failure is sequencing these stages by cost rather than by
logic — discovery first, then a large clinical validation, then analytical validation when a
regulator asks. The correct order is discovery, analytical validation, clinical validation,
utility, because an assay that fails analytical validation cannot produce interpretable clinical
validation.

### 9.4 The economics of prevalence

The prevalence arithmetic has a direct commercial consequence. A test with a PPV of 18% at 5%
prevalence generates about 4.5 false positives per true positive. If each false positive leads to
a confirmatory procedure costing USD 2,000, the confirmatory cost per true case detected is about
USD 9,000, before the cost of the test itself. At 30% prevalence, the PPV is 65% and the
confirmatory cost per true case falls to about USD 1,100.

The same assay, the same price, and a different population produce an eightfold difference in the
cost per case detected. A reimbursement argument that ignores prevalence is not an argument.

### 9.5 The companion diagnostic precedent

Companion diagnostics show what the full chain looks like. A therapeutic is developed with a
test that selects the population, the test is analytically validated, the clinical validation is
the trial itself, and the utility is established by the trial's result. The FDA list of
authorized companion diagnostics is the public record of tests that completed the chain.

The lesson for a non-companion biomarker is that the chain is the same, even when the regulatory
pathway is lighter. A test that selects patients for a therapy is held to the therapeutic's
evidence standard, because the test's error becomes the therapy's error.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Discovery-only claim | No independent cohort | Overfitting and selection | Non-replicating biomarker | External validation |
| Case-control prevalence | Sampling by outcome | Cannot estimate PPV | Wrong business case | Cohort validation |
| Cutoff chosen post hoc | Cutoff not in the plan | Selection on validation | Optimistic performance | Pre-specify cutoff |
| No calibration | AUC only | Discrimination without probabilities | Unusable risk estimates | Calibration slope |
| Spectrum bias | Cases severe, controls healthy | Case-mix difference | Overestimated accuracy | Representative cohort |
| Batch as biomarker | Signal tracks processing | Technical variation | Non-transportable | Balanced batches |
| No analytical validation | Assay precision unknown | Measurement error | Uninterpretable clinical result | Validation before clinical |
| Missing data by group | Biomarker missing in cases | Informative missingness | Selected population | Report and model |
| Multiplicity ignored | Many features, no correction | Selection | False leads | BH with family |
| Optimism unreported | In-sample performance | Data-dependent choices | Inflated claim | Corrected estimate |
| Assay drift | Control trend over time | Reagent or instrument change | Moving threshold | EQA, lot studies |
| Matrix change | Serum validated, plasma used | Specimen difference | Performance loss | Validate the used matrix |
| Surrogate endpoint | Response rate not survival | Endpoint mismatch | Utility not shown | Outcome endpoint |
| No comparator | Test vs nothing | Missing incremental value | Unknown benefit | Current decision rule |
| Utility assumed | Validation called utility | Stage conflation | Adoption without evidence | Decision-impact study |

Four red flags:

**If the PPV at the intended prevalence is below 50%, stop and ask what the false positives cost.**
The test may still be useful for rule-out, but the claim must change.

**If the cutoff is not in the pre-specified plan, stop.** The validation is partly training.

**If the assay's CV is larger than the effect, stop.** The measurement cannot support the
decision.

**If there is no calibration report, stop.** A probability that is not calibrated cannot be used
in a threshold decision.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The discovery cohort represents the intended population | Transport | Case-control sampling | Compare distributions |
| The assay measures reliably | Clinical use | Imprecision, interference | Analytical validation |
| The model is locked | Validation meaning | Re-fitting | Versioned artifact |
| Prevalence is known | Predictive values | Case-control design | Cohort estimate |
| The cutoff is pre-specified | Unbiased performance | Post hoc selection | Plan audit |
| Calibration holds | Threshold decisions | Overfitting, drift | Calibration plot |
| The validation cohort is independent | Unbiased estimate | Shared samples or sites | Provenance check |
| The endpoint matters | Utility | Surrogate mismatch | Clinical rationale |
| The comparator is current practice | Incremental value | Wrong comparator | Pathway mapping |
| Benefits exceed harms | Utility | False-positive burden | Net benefit, trial |

The first and last assumptions are the ones that decide whether a biomarker is a product. A
perfectly measured, well-calibrated test for an outcome nobody acts on is a research instrument.

## 12. Exercises

**Exercise 1 — Stage classification (recall, 20 minutes).**
Classify each claim into discovery, analytical validation, clinical validation, or utility, and
name the missing evidence: (a) "the protein differs between cases and controls in our cohort";
(b) "the assay's CV is 6% across three sites"; (c) "the score predicts response in an
independent cohort"; (d) "using the score increased the proportion of patients receiving the
right drug."
*Deliverable:* four-row table.
*Criteria:* correct stage and missing evidence for each.
*Time:* 20 minutes.

**Exercise 2 — PPV arithmetic (calculation, 30 minutes).**
Sensitivity 0.90, specificity 0.85. Compute PPV and NPV at prevalence 2%, 10%, and 40%. State at
which prevalence the test is suitable for ruling in.
*Deliverable:* six numbers and a one-sentence recommendation.
*Criteria:* arithmetic correct; recommendation follows from PPV.
*Time:* 30 minutes.

**Exercise 3 — Net benefit (calculation, 35 minutes).**
Using the same test, compute net benefit at a 0.25 threshold for prevalence 5% and 25%. Compare
with treat-all and treat-none. State the decision.
*Deliverable:* arithmetic and a decision table.
*Criteria:* net benefit correct; comparison includes both defaults.
*Time:* 35 minutes.

**Exercise 4 — Optimism (application, 30 minutes).**
Apparent AUC 0.88, cross-validated 0.79. Compute optimism and the corrected estimate. State how
the correction would change a go/no-go decision with a 0.75 threshold.
*Deliverable:* arithmetic and a decision.
*Criteria:* corrected estimate correct; decision follows from the threshold.
*Time:* 30 minutes.

**Exercise 5 — Critique a biomarker paper (critique, 60 minutes).**
Select a published biomarker study. Extract the discovery design, validation, cutoff selection,
calibration, and prevalence. Identify three defects and their consequences.
*Deliverable:* one-page critique.
*Criteria:* three specific defects; consequences quantified where possible.
*Time:* 60 minutes.

**Exercise 6 — Analytical validation plan (design, 50 minutes).**
Design an analytical validation for an immunoassay intended for emergency triage. Specify
precision, linearity, LoD, LoQ, interference, and stability studies with acceptance criteria.
*Deliverable:* a validation protocol.
*Criteria:* each study has a design and a numeric criterion; specimen matrix stated.
*Time:* 50 minutes.

**Exercise 7 — Go/no-go criteria (design, 40 minutes).**
Write pre-specified go/no-go criteria for advancing a biomarker from discovery to analytical
validation, and from analytical validation to clinical validation.
*Deliverable:* two sets of criteria.
*Criteria:* numeric thresholds; includes assay precision and effect size; includes a stop rule.
*Time:* 40 minutes.

**Exercise 8 — Utility design (design, 60 minutes).**
Design a decision-impact study for a biomarker that predicts response. Specify the population,
the comparator, the primary endpoint, the sample size logic, and the analysis.
*Deliverable:* two-page design.
*Criteria:* comparator is current practice; endpoint is a decision or outcome; sample size
justified.
*Time:* 60 minutes.

## 13. Self-check answers

**Answer 1 (stages).** (a) Discovery; missing independent validation and analytical validation.
(b) Analytical validation; missing clinical validation and utility. (c) Clinical validation;
missing utility and, unless reported, calibration. (d) Utility; the missing evidence is the
analytical and clinical validation that should precede it, and the effect size of the decision
change.

**Answer 2 (PPV).** With sensitivity 0.90 and specificity 0.85:

```text
prevalence 0.02: PPV = 0.018 / (0.018 + 0.147) = 0.109; NPV = 0.833 / (0.833 + 0.002) = 0.998
prevalence 0.10: PPV = 0.090 / (0.090 + 0.135) = 0.400; NPV = 0.765 / (0.765 + 0.010) = 0.987
prevalence 0.40: PPV = 0.360 / (0.360 + 0.090) = 0.800; NPV = 0.510 / (0.510 + 0.040) = 0.927
```

The test is suitable for ruling in only at high prevalence: at 40%, PPV is 80%. At 2%, nine of
ten positives are false.

**Answer 3 (net benefit).** At threshold 0.25:

```text
prevalence 0.05: NB = 0.90 x 0.05 - 0.15 x 0.95 x (0.25/0.75) = 0.045 - 0.0475 = -0.0025
prevalence 0.25: NB = 0.90 x 0.25 - 0.15 x 0.75 x 0.333 = 0.225 - 0.0375 = 0.1875
treat all at 0.05: 0.05 - 0.95 x 0.333 = -0.267
treat all at 0.25: 0.25 - 0.75 x 0.333 = 0.0
treat none: 0
```

At 5% prevalence the test's net benefit is slightly negative but far better than treat-all; the
best default is treat-none, and the test is worse than that default. At 25% prevalence the test
is better than both defaults. The decision is to restrict use to the higher-prevalence setting.

**Answer 4 (optimism).** Optimism is 0.88 - 0.79 = 0.09, and the corrected estimate is 0.79. If
the go/no-go threshold is 0.75, the corrected estimate passes, but narrowly, and the interval
should be checked before advancing. A program that used the apparent 0.88 would have more
confidence than the data support.

**Answer 5 (critique).** Typical defects: a case-control discovery cohort without a prevalence
estimate, so PPV is uncomputable; a cutoff selected on the validation set, so the reported
sensitivity and specificity are optimistic; no calibration report, so the score cannot support a
threshold decision; and no analytical validation, so the measurement error is unknown. Each has a
specific consequence: an unusable business case, inflated accuracy, unusable probabilities, and
an uninterpretable clinical result.

**Answer 6 (analytical validation).** Precision: 20 replicates per level across three
concentrations on five days, CV below 15% at each level. Linearity: seven dilution levels,
regression slope 0.9 to 1.1, R-squared above 0.98. LoD: blank plus three standard deviations
across 20 replicates. LoQ: lowest level with CV below 20% and bias within 20%. Interference:
haemoglobin, bilirubin, triglycerides, and biotin at defined concentrations, recovery within
10%. Stability: specimen at room temperature, 4 degrees Celsius, and frozen, at 0, 24, and 72
hours and after a freeze-thaw cycle.

**Answer 7 (go/no-go).** Discovery to analytical validation: out-of-fold AUC at least 0.75 with a
lower confidence bound above 0.65, at least two independent feature-selection runs agreeing on
the top features, and a plausible biological mechanism. Analytical validation to clinical
validation: CV below 15%, LoD below the decision threshold, no clinically relevant interference,
and specimen stability sufficient for the workflow. Clinical validation to utility: AUC at least
0.70 with a lower bound above 0.60, calibration slope 0.8 to 1.2, and a PPV at intended
prevalence that the program judges actionable.

**Answer 8 (utility design).** Population: patients for whom the treatment decision is uncertain.
Comparator: current practice without the test. Primary endpoint: proportion of patients receiving
guideline-concordant therapy, or the clinical outcome if the therapy is effective. Sample size:
powered to detect a 10 percentage point absolute increase in the primary endpoint, with the
intracluster correlation accounted for if clinicians are randomised. Analysis: intention-to-treat
at the clinician or site level, with a pre-specified subgroup by test result.

## 14. Interview questions

**1. What is the difference between clinical validation and clinical utility?**
Strong answer: validation shows association and discrimination in the intended population;
utility shows that using the test changes decisions or improves outcomes.
Weak answer: treats a validated AUC as utility.

**2. Why is PPV not a property of the test alone?**
Strong answer: it depends on prevalence; sensitivity and specificity are test properties, PPV is
a property of the test in a population.
Weak answer: "PPV is the same as sensitivity."

**3. What does an AUC of 0.8 tell you, and what does it not?**
Strong answer: discrimination, not calibration, not utility, and not performance at a specific
cutoff or prevalence.
Weak answer: "80% accuracy."

**4. How do you correct for optimism?**
Strong answer: bootstrap or cross-validation to estimate the gap between apparent and
out-of-sample performance, with the method stated.
Weak answer: "use cross-validation" without an estimate.

**5. What must an analytical validation contain?**
Strong answer: accuracy, precision, linearity, LoD, LoQ, interference, and stability, with
pre-specified acceptance criteria.
Weak answer: "reproducibility."

**6. Why does a case-control discovery cohort limit the claims?**
Strong answer: sampling by outcome prevents prevalence estimation, so predictive values cannot
be computed and the cohort's spectrum may not match the intended population.
Weak answer: "it is a weaker design" without the consequence.

**7. What is spectrum bias?**
Strong answer: performance changes when the case mix changes, because the test discriminates
severe cases from healthy controls better than mild cases from comorbid controls.
Weak answer: "the sample is biased."

**8. When would you lock a model?**
Strong answer: after discovery and before validation, with preprocessing, features, weights, and
cutoff fixed and versioned.
Weak answer: "at the end."

**9. What is net benefit and why use it?**
Strong answer: a decision-analytic metric that weights true and false positives by the threshold,
compared against treat-all and treat-none.
Weak answer: "another accuracy metric."

**10. What regulatory pathway applies to a biomarker used in drug development?**
Strong answer: the FDA Biomarker Qualification Program, with a defined context of use, or an IVD
pathway if it is a test; ICH E16 for the submission format.
Weak answer: "FDA approval."

**11. What would make you stop a biomarker program?**
Strong answer: PPV too low at intended prevalence, assay precision worse than the effect, no
calibration, or no decision change.
Weak answer: "if the AUC is below 0.7."

**12. How do you choose the intended population?**
Strong answer: from the decision and the prevalence at which the test is useful, not from the
easiest cohort to recruit.
Weak answer: "whoever is available."

## 15. Key terms

- **Analytical validation** — establishing assay performance.
- **AUC** — area under the ROC curve.
- **BEST category** — FDA biomarker category.
- **Biomarker** — measured characteristic indicating a process or outcome.
- **Calibration** — agreement of predicted and observed risk.
- **Clinical utility** — evidence that use improves decisions or outcomes.
- **Clinical validation** — association in the intended population.
- **Companion diagnostic** — test required for a therapeutic.
- **Context of use** — specified use of a biomarker.
- **Cutoff** — threshold converting score to decision.
- **Decision curve** — net benefit across thresholds.
- **Discovery** — first identification of an association.
- **External validation** — performance in an independent cohort.
- **Interference** — effect of another substance on the assay.
- **IVD** — in vitro diagnostic device.
- **LDT** — laboratory-developed test.
- **LoD** — limit of detection.
- **LoQ** — limit of quantification.
- **Matrix effect** — specimen-specific assay behaviour.
- **Net benefit** — decision-analytic benefit minus harm.
- **NPV** — negative predictive value.
- **Optimism** — inflation from data-dependent choices.
- **PPV** — positive predictive value.
- **Pre-specification** — analysis fixed before data.
- **Precision** — closeness of repeated measurements.
- **REMARK** — reporting guideline for prognostic marker studies.
- **Sensitivity** — true positive fraction.
- **Specificity** — true negative fraction.
- **Spectrum bias** — performance change with case mix.
- **STARD** — reporting guideline for diagnostic accuracy.
- **TRIPOD** — reporting guideline for prediction models.
- **Utility study** — a study that measures whether test use changes decisions or outcomes.
- **Youden index** — sensitivity plus specificity minus one, a cutoff criterion.

## 16. Sources

Standards and guidance:

- FDA, Biomarker Qualification Program. https://www.fda.gov/drugs/biomarker-qualification-program/about-biomarkers-and-qualification (accessed 25 October 2026).
- FDA, Context of Use. https://www.fda.gov/drugs/biomarker-qualification-program/context-use (accessed 25 October 2026).
- ICH E16, Biomarkers Related to Drug or Biotechnology Product Development. https://database.ich.org/sites/default/files/E16_Guideline.pdf (accessed 25 October 2026).
- FDA, List of FDA-Authorized Companion Diagnostic Devices. https://www.fda.gov/medical-devices/in-vitro-diagnostics/list-fda-authorized-companion-diagnostic-devices-in-vitro-and-imaging-tools (accessed 25 October 2026).
- FDA, Laboratory Developed Tests. https://www.fda.gov/medical-devices/in-vitro-diagnostics/laboratory-developed-tests (accessed 25 October 2026).
- ISO 15189:2022. https://www.iso.org/standard/76677.html (accessed 25 October 2026).
- Regulation (EU) 2017/746. https://eur-lex.europa.eu/eli/reg/2017/746/oj (accessed 25 October 2026).

Methods and evidence:

- Ioannidis, J. P. A., "Why most published research findings are false," *PLoS Medicine* 2:e124, 2005. https://doi.org/10.1371/journal.pmed.0020124 (accessed 25 October 2026).
- Begley, C. G., Ellis, L. M., "Raise standards for preclinical cancer research," *Nature* 483:531-533, 2012. https://doi.org/10.1038/483531a (accessed 25 October 2026).
- Prinz, F., Schlange, T., Asadullah, K., "Believe it or not: how much can we rely on published data on potential drug targets?" *Nature Reviews Drug Discovery* 10:712, 2011. https://doi.org/10.1038/nrd3439-c1 (accessed 25 October 2026).
- Collins, G. S. et al., "TRIPOD+AI statement," *BMJ* 385:e078378, 2024. https://doi.org/10.1136/bmj-2023-078378 (accessed 25 October 2026).
- Vickers, A. J., Elkin, E. B., "Decision curve analysis: a novel method for evaluating prediction models," *Medical Decision Making* 26:565-574, 2006. https://doi.org/10.1177/0272989X06295361 (accessed 25 October 2026).

Reporting guidelines:

- REMARK. https://www.equator-network.org/reporting-guidelines/remark/ (accessed 25 October 2026).
- STARD. https://www.equator-network.org/reporting-guidelines/stard/ (accessed 25 October 2026).

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`07_multi-omics-integration.md`](07_multi-omics-integration.md)
- Next topic: [`09_drug-discovery.md`](09_drug-discovery.md)
- Related topic: [`03_transcriptomics.md`](03_transcriptomics.md)
- Related topic: [`10_clinical-genomics.md`](10_clinical-genomics.md)
- Survey file: [`../../12_omics_precision_medicine.md`](../../12_omics_precision_medicine.md)
- Labs: [`../../labs/`](../../labs/)
