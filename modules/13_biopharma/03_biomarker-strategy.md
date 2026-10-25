# Biomarker Strategy: Prognostic, Predictive, and the Arithmetic of Enrichment

> Module 13 — Biopharma and Clinical Development
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 04 (statistics); Module 12 topic 08 (biomarker discovery); topic 01
> Version 2026.3

## 1. Why this topic exists

A biomarker is a measurement that is used to make a decision. The decision may be which patients to
enrol, which dose to give, whether a drug is engaging its target, or whether a patient is responding.
Each of those decisions requires a different kind of evidence, and the word "biomarker" conceals the
difference. A marker that predicts outcome regardless of treatment is prognostic. A marker that
predicts who responds to a treatment is predictive. A marker that changes when the drug hits its
target is pharmacodynamic. A marker that predicts toxicity is a safety marker. These are not
interchangeable, and a programme that treats one as another will make a confident wrong decision.

The commercial consequence is large. A predictive biomarker can halve a trial, shorten enrolment,
and produce a label that supports a premium price. A prognostic biomarker used as if it were
predictive shrinks the trial population without increasing the treatment effect, which raises cost
per patient and lowers the chance of success. The two look identical in a single-arm study, because
both produce a correlation between marker status and outcome. Only a comparison of treated and
untreated patients can distinguish them.

There is a second, subtler problem: the arithmetic of testing. A biomarker assay is never perfect.
At low prevalence, even a highly sensitive and specific assay produces a majority of false positives
among the patients it labels positive. Those false-positive patients are enrolled in the trial,
receive the drug, and do not respond, which dilutes the measured treatment effect. The trial then
fails, and the post-mortem concludes that the biomarker "did not work" when the correct conclusion
is that the assay's positive predictive value was too low for the prevalence.

This topic teaches the evidence ladder for a biomarker, the arithmetic that determines whether a
marker can support a decision at a given prevalence, and the design choices that follow: enrichment,
assay cut-off, companion diagnostic co-development, and regulatory qualification. It builds on the
biomarker discovery material in Module 12 and precedes the feasibility (topic 04) and standards
(topic 05) topics, both of which depend on the biomarker definition being fixed.

### 1.1 The failure pattern

1. A retrospective analysis finds that patients with high expression of a protein respond better to
   the drug. The hazard ratio in the high group is 0.55; in the low group it is 0.90.
2. The programme decides to enrich for high expression in Phase III.
3. The assay cut-off is chosen from the same retrospective data that produced the finding.
4. At the chosen cut-off, 35% of screened patients are "high".
5. The assay has 85% sensitivity and 82% specificity for the true high state, and true high
   prevalence is 20%. The positive predictive value is therefore 54%, so nearly half of enrolled
   patients are not truly high.
6. The Phase III is powered on the 0.55 hazard ratio observed in the retrospective high group.
7. The observed hazard ratio is 0.74, the trial misses its endpoint, and the programme is stopped.

The failure occurred at step 5, and it was computable at step 3. The observed effect in the trial is
approximately what the dilution arithmetic predicts. No amount of additional biology would have
rescued a design that enrolled the wrong patients and powered on the wrong effect.

### 1.2 Who is blocked when biomarker strategy is weak

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Translational lead | A validated hypothesis and a programme | Does not control assay validation resources |
| Clinical team | A trial population that can show the effect | Receives a cut-off without its performance |
| Biostatistics | Power, because the effect is diluted | Receives the effect estimate after design lock |
| Regulatory affairs | A label claim that the data can support | Receives the assay and the trial as fixed inputs |
| Diagnostic partner | A co-development timeline | Is engaged after the Phase III design is fixed |
| Payer | A test-and-treat pathway with evidence of utility | Sees the drug evidence, not the testing pathway |
| Patient | Exposure to a drug unlikely to help | Is told they are biomarker-positive |

### 1.3 The industry question

> Is this marker a cause, a consequence, or a correlate — and does the assay, at this prevalence,
> identify the patients who benefit?

The question separates the biology (is the marker on the causal pathway?) from the measurement (does
the assay identify the state?) from the decision (does identifying the state change the outcome?).
A programme needs all three, and most biomarker failures are a failure of the second or third, not
the first.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a biomarker as prognostic, predictive, pharmacodynamic, safety, monitoring,
   diagnostic, or susceptibility, and name the study design that supports each claim.
2. **Compute** sensitivity, specificity, positive predictive value, and negative predictive value at
   a stated prevalence, and explain why predictive values change with prevalence while sensitivity
   and specificity do not.
3. **Quantify** the dilution of a treatment effect caused by enrolling assay-positive but
   biomarker-negative patients, and translate the diluted effect into a required event count.
4. **Design** an enrichment strategy with a pre-specified cut-off, an assay performance
   specification, and a screening plan.
5. **Distinguish** a predictive from a prognostic marker using a treatment-by-marker interaction,
   and state why a single-arm study cannot make the distinction.
6. **Map** the companion diagnostic co-development path and the regulatory qualification routes in
   the US and EU.
7. **Diagnose** a biomarker programme whose cut-off was selected on the same data used to estimate
   the effect, and state the consequence for the trial's power.

## 3. The core mental model

### 3.1 The evidence ladder

```text
 Discovery            Analytical validation        Clinical validation          Utility            Qualification
 ---------            ---------------------        -------------------          -------            -------------
 association in       assay measures the           marker associated with      using the marker   regulator accepts
 a cohort             state accurately and          outcome in the intended      changes the        the marker for a
                      reproducibly                  population, pre-specified    decision and       stated context
                                                                                improves outcome   of use
     |                        |                            |                        |                  |
  sensitivity to          precision, accuracy,        discrimination,          randomised or      COU statement,
  confounding,            LoD, specimen               calibration, PPV,        decision-impact    package review
  overfitting             stability, cut-off          interaction test         evidence
```

Each rung is a different claim. A marker can be analytically perfect and clinically useless; it can
be clinically associated and have no utility; it can have utility in one context of use and none in
another. The ladder is not a sequence of increasing effort on the same claim; it is a sequence of
different claims, each requiring its own evidence.

### 3.2 Prognostic versus predictive

| Question | Design that answers it | What a single-arm study shows |
|---|---|---|
| Does the marker predict outcome regardless of treatment? | Cohort or control arm, marker measured at baseline | Yes, if marker and outcome are associated |
| Does the marker predict who benefits from treatment? | Randomised comparison with a treatment-by-marker interaction | Nothing; there is no untreated comparison |
| Does the marker change with treatment? | Serial measurement in treated and control | Change, but not whether the change mediates benefit |
| Does the marker predict toxicity? | Randomised comparison of toxicity by marker status | Association, confounded by exposure |

The distinction is not academic. A prognostic marker that also enriches a trial will produce a
population with a different event rate, which changes the sample size and the interpretation of the
control arm, without changing the relative benefit of treatment. A predictive marker changes the
relative benefit itself.

### 3.3 The testing funnel

```text
 Intended-use population
 1,000 patients
     |
     | true biomarker-positive prevalence 8%  ->  80 true positive, 920 true negative
     |
     v
 Assay with sensitivity 92%, specificity 88%
     |
     +--> detected positive: 0.92 x 80  + 0.12 x 920 = 73.6 + 110.4 = 184
     |
     v
 Assay-positive cohort (184)
     +--> true positive 73.6  (40.0%)
     +--> false positive 110.4 (60.0%)
```

The funnel shows why "the assay is 92% sensitive" is not a statement about the patient in front of
you. At 8% prevalence, 60% of assay-positive patients are negative for the true state. Those
patients enter the trial, receive the drug, and dilute the effect.

### 3.4 Dilution of the treatment effect

If the true effect is a hazard ratio of 0.55 in biomarker-positive patients and 0.95 in
biomarker-negative patients, and the enrolled population is a mixture with a true-positive fraction
w, the expected hazard ratio in the enrolled population is:

```text
S_treat = w x S_control^HR_pos + (1 - w) x S_control^HR_neg
HR_mixture = ln(S_treat) / ln(S_control)
```

where S is the survival or non-event probability at the analysis time. The formula is an
approximation that assumes the mixture is stable over time and that the marker does not change. It
is accurate enough to show the magnitude of the problem: at w = 0.40, the example above produces a
mixture hazard ratio of about 0.78 rather than 0.55.

The consequence for sample size is severe. The required number of events is approximately:

```text
E = 4 x (z_alpha/2 + z_beta)^2 / (ln HR)^2
```

For 90% power at two-sided alpha 0.05, an HR of 0.55 requires about 118 events; an HR of 0.78
requires about 657. The false positives have multiplied the trial by a factor of 5.6.

### 3.5 The decision the model supports

The model does not decide whether to enrich. It prices the alternatives:

### 3.6 Three questions to ask of any biomarker

Before any arithmetic, three questions determine whether the programme has a biomarker at all.

1. **Is the marker on the causal pathway, or a correlate?** A marker that is a consequence of tumour
   burden will change with effective treatment without identifying who responds. Causal evidence
   comes from genetics, mechanism, or experiments, not from association.
2. **Can the marker be measured reliably in the patients who need the decision?** An assay that
   requires fresh tissue, a research-only platform, or a specialised laboratory may be analytically
   perfect and operationally useless.
3. **Does the decision change the outcome?** A marker that reclassifies risk without changing
   treatment is a prognostic description, not a clinical tool.

A programme that cannot answer all three has a research hypothesis. That is a legitimate starting
point, but it is not a basis for a Phase III design.

- Enrich with a better assay: fewer patients screened per enrolled patient, smaller trial, but a
  narrower label and a diagnostic development cost.
- Enrich with a worse assay: larger trial, more screen failures, and a diluted effect that may not
  reach significance.
- Do not enrich: the largest population, the smallest effect, and a label that may not support a
  companion diagnostic claim.
- Develop a companion diagnostic in parallel: the highest evidence bar and the strongest label.

Each option has a different cost, timeline, and evidence package. The arithmetic in Sections 5 and
6 is how the choice is made explicit.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Biomarker | A defined characteristic measured as an indicator of normal biological processes, pathogenic processes, or responses to an intervention | Used for any measured variable | Defined characteristic |
| Prognostic biomarker | Identifies likelihood of a clinical event or progression in patients with the disease, independent of treatment | Used as a synonym for predictive | Association with outcome |
| Predictive biomarker | Identifies individuals more likely to experience a favourable or unfavourable effect from a treatment | Inferred from a single-arm association | Treatment-by-marker interaction |
| Pharmacodynamic (PD) biomarker | Indicates biological activity of a product, such as target engagement | Treated as evidence of efficacy | Change with exposure |
| Safety biomarker | Indicates likelihood of toxicity | Used as a substitute for clinical safety data | Association with adverse event |
| Monitoring biomarker | Measured serially to track disease or response | Confused with a PD marker | Serial measurement |
| Diagnostic biomarker | Detects or confirms a disease or condition | Confused with a predictive marker | Diagnostic accuracy |
| Susceptibility/risk biomarker | Indicates potential for developing a disease | Used as a diagnostic | Risk association |
| Response biomarker | Indicates biological response to treatment | Treated as a surrogate endpoint | Change from baseline |
| Surrogate endpoint | An endpoint used as a substitute for a direct measure of clinical benefit | Assumed validated | Endpoint classification |
| Context of use (COU) | The explicit statement of how a biomarker will be used, in what population, for what decision | Omitted from qualification packages | Statement |
| Analytical validation | Demonstration that an assay measures the intended analyte accurately and reproducibly | Confused with clinical validation | Performance study |
| Clinical validation | Association of the biomarker with the clinical outcome in the intended population | Confused with utility | Association study |
| Clinical utility | Evidence that using the biomarker changes decisions and improves outcomes | Assumed from association | Decision-impact study |
| Sensitivity (analytical) | Ability to detect the analyte when present | Confused with diagnostic sensitivity | Proportion |
| Specificity (analytical) | Ability to avoid detecting the analyte when absent | Confused with diagnostic specificity | Proportion |
| Diagnostic sensitivity | Proportion of true positives classified positive by the test | Quoted without the reference standard | Proportion |
| Diagnostic specificity | Proportion of true negatives classified negative | Quoted without the reference standard | Proportion |
| Positive predictive value (PPV) | Proportion of test positives who are true positives | Quoted as if prevalence-independent | Proportion, given prevalence |
| Negative predictive value (NPV) | Proportion of test negatives who are true negatives | Quoted as if prevalence-independent | Proportion, given prevalence |
| Prevalence | Proportion of the population with the condition or biomarker state | Taken from a trial screen rather than the intended-use population | Proportion |
| Cut-off | The threshold that defines a positive result | Selected after seeing outcome data | Value with units |
| Limit of detection (LoD) | Lowest analyte concentration reliably detected | Confused with limit of quantification | Concentration |
| Analytical range | Range over which the assay is validated | Assumed to cover all samples | Concentration interval |
| Pre-analytical variables | Specimen collection, handling, and storage factors that affect measurement | Ignored in validation | Protocol |
| Tumour heterogeneity | Variation in marker expression within and between lesions | Ignored in single-biopsy designs | Biological property |
| Companion diagnostic (CDx) | An in vitro diagnostic whose information is essential for the safe and effective use of a corresponding therapeutic | Used for any exploratory assay | Regulated device |
| Laboratory-developed test (LDT) | A test developed and performed within a single laboratory | Assumed equivalent to an approved IVD | Regulatory category |
| In vitro diagnostic (IVD) | A diagnostic test performed on a specimen outside the body | Confused with CDx | Device category |
| Biomarker qualification | Regulatory acceptance of a biomarker for a stated context of use | Confused with assay approval | Regulatory status |
| Enrichment | Selecting patients by a marker to increase the expected treatment effect or reduce heterogeneity | Confused with arbitrary restriction | Design strategy |
| All-comers design | Enrolling without biomarker selection | Assumed to be more generalisable | Design strategy |
| Treatment-by-marker interaction | Statistical test that the treatment effect differs by marker status | Confused with a subgroup analysis | Interaction term |
| Effect dilution | Reduction in the observed treatment effect from including non-responders | Ignored in power calculations | Hazard ratio or risk difference |
| Standard of care (SoC) | The treatment a patient would receive absent the trial | Assumed static | Clinical practice |

### 4.1 The two claims that are most often confused

**Prognostic:** "patients with high marker have worse outcomes." The evidence is an association
between marker and outcome in a defined population. It does not require a treatment comparison, and
it does not imply that treating marker-high patients works better.

**Predictive:** "patients with high marker benefit more from this treatment than patients without
it." The evidence is a difference in treatment effect between marker strata, estimated from a
randomised comparison. It requires an interaction, not two separate significant results, and it
requires the interaction to be pre-specified or replicated.

A marker can be both. A marker can be prognostic in the control arm and predictive because the
treatment effect is larger in marker-positive patients. The two properties are estimated from
different contrasts, and one does not imply the other.

## 5. The mechanics, step by step

This procedure takes a candidate biomarker from a research hypothesis to a decision-supporting
asset. It is written for a translational or clinical team preparing a development programme, and it
assumes the discovery evidence exists in at least one cohort.

### Step 1. Write the context of use

**Input:** the candidate marker and the programme decision it is meant to inform.

**Output:** a context-of-use statement naming the population, the decision, the assay, the specimen,
the timing, and the claimed consequence.

**Rationale:** The context of use is the anchor for every subsequent validation. "Predicts response"
is not a context of use. "Selects patients with advanced disease and the activating alteration for
first-line treatment with the drug, using the assay on formalin-fixed tissue, to decide whether to
start the drug" is. Without it, analytical validation has no performance target and clinical
validation has no population.

**Branch point:** if the biomarker is intended for a regulated claim (patient selection in a label),
the context of use must be written to the standard of a qualification or device submission. If it
is intended for internal dose selection, a lighter statement is acceptable, but it must still name
the decision.

### Step 2. Classify the biomarker and the claim

**Input:** the context of use and the biology.

**Output:** a category (prognostic, predictive, PD, safety, monitoring, diagnostic, susceptibility)
and the specific claim.

**Rationale:** The category determines the design. A PD marker needs serial sampling and a
demonstrated exposure-response; a predictive marker needs a randomised interaction; a prognostic
marker needs a cohort with outcome data. Misclassifying the marker selects the wrong design, and the
resulting evidence cannot support the claim even if the study is well conducted.

**Branch point — what evidence each claim requires:**

| Claim | Minimum design | Key analysis | What would falsify it |
|---|---|---|---|
| Prognostic | Cohort with marker measured at baseline | Marker-outcome association adjusted for confounders | No association after adjustment |
| Predictive | Randomised comparison with marker strata | Treatment-by-marker interaction | Interaction consistent with no difference |
| PD | Dose-ranging with serial sampling | Exposure-response for the marker | No change with exposure |
| Safety | Randomised comparison by marker status | Toxicity rate difference | No difference after adjustment |
| Monitoring | Serial measurement with clinical anchors | Change precedes or tracks outcome | Change unrelated to outcome |
| Diagnostic | Cross-sectional with reference standard | Sensitivity, specificity, predictive values | Accuracy below the clinical requirement |

### Step 3. Specify the assay and the specimen

**Input:** the context of use and the available sample types.

**Output:** an assay specification: platform, reagent, specimen type, collection and processing,
storage, analytical range, and control materials.

**Rationale:** The assay must be usable at the point of decision in the intended population. A marker
measured in fresh frozen tissue cannot select patients in a community oncology setting where only
archival formalin-fixed tissue exists. The specimen requirement is a design constraint that
determines feasibility and, often, the commercial viability of the test.

**Edge case:** tumour heterogeneity. A single biopsy may not represent the marker status of
metastatic lesions. The protocol must state how many samples are required, what constitutes a
positive result when samples disagree, and whether circulating tumour material is acceptable.

### Step 4. Run analytical validation

**Input:** the assay specification and the intended specimen types.

**Output:** a validation report covering precision, accuracy, sensitivity, specificity, LoD,
analytical range, interference, specimen stability, and cut-off reproducibility.

**Rationale:** Clinical validation of a marker measured by an unvalidated assay is a validation of
the assay's error, not of the biology. Analytical validation establishes the measurement before the
measurement is used to make a claim. The report must include the pre-analytical variables, because
specimen handling often explains more variance than the assay itself.

**Branch point:** if the assay is a laboratory-developed test intended for trial use only, the
validation standard is fit-for-purpose and should be agreed with the regulator in advance. If the
assay will become a companion diagnostic, the validation must meet the device standard, which is
substantially more demanding.

### Step 5. Pre-specify the cut-off and lock it

**Input:** the analytical validation and the discovery data.

**Output:** a cut-off with its derivation, its performance at that cut-off, and a lock date.

**Rationale:** A cut-off chosen after seeing outcome data is a fitted parameter, and its performance
in the same data is optimistic. The cut-off must be derived on one dataset and tested on another, or
pre-specified from biology and tested prospectively. The lock date is what makes the later
validation meaningful.

**Edge case:** continuous markers. A cut-off discards information. If the marker is continuous, the
programme should consider a pre-specified functional form (for example, a linear predictor or a
restricted cubic spline) and report the cut-off as a decision rule derived from it, not as an
independent discovery.

### Step 6. Design the clinical validation

**Input:** the locked assay and cut-off, and the intended population.

**Output:** a validation study with a pre-specified analysis, a reference standard, and a sample
size justified by the expected performance.

**Rationale:** The design must estimate the performance that matters for the decision: predictive
values at the intended prevalence, and the treatment effect in marker-positive and marker-negative
strata. A study powered only for association may be too small to estimate the interaction.

**Branch point:** prospective versus prospective-retrospective. A prospective-retrospective design
uses stored specimens from a completed randomised trial and is acceptable when the analysis is
pre-specified and the specimens are representative. It is faster and cheaper than a new prospective
study, and it is the standard approach for many predictive markers.

### Step 7. Estimate discrimination and calibration

**Input:** the validation data.

**Output:** AUC or equivalent discrimination, calibration, and predictive values at stated
prevalences.

**Rationale:** Discrimination (who is higher risk) and calibration (how high the risk is) answer
different questions. A marker can discriminate well and be badly calibrated, which matters when the
marker is used to set a threshold. For a binary decision rule, predictive values at the intended
prevalence are more useful than AUC.

### Step 8. Test the predictive claim with an interaction

**Input:** the randomised data with marker strata.

**Output:** a treatment-by-marker interaction estimate with a confidence interval, and the
stratum-specific effects.

**Rationale:** Two significant results in opposite directions are not required, and two
non-significant results do not exclude an interaction. The interaction estimate is the quantity that
supports a predictive claim. A common error is to report the marker-positive effect and omit the
marker-negative effect, which hides the dilution.

**Edge case:** a qualitative interaction (benefit in one stratum, harm in another) is rare and
requires strong evidence before it is used to restrict a label.

### Step 9. Model enrichment and dilution

**Input:** prevalence, assay performance, and stratum-specific effects.

**Output:** the expected effect in the enrolled population, the required event count, the screen
failure burden, and the timeline.

**Rationale:** This is the step that converts a biological hypothesis into a feasible trial. It is
also the step most often skipped, because it requires the assay performance and the prevalence to be
stated explicitly, and both are uncertain. The uncertainty should be carried into a sensitivity
analysis, not avoided.

### Step 10. Decide the companion diagnostic path

**Input:** the regulatory claim, the assay, and the commercial context.

**Output:** a co-development plan with the device partner, the submission sequence, and the
labeling strategy.

**Rationale:** A companion diagnostic is a regulated device whose information is essential for the
safe and effective use of the drug. The drug and device are expected to be developed together, and
the labeling of each references the other. Engaging the device partner after the Phase III design is
fixed usually produces a mismatch between the trial assay and the commercial assay.

**Branch point:** class labeling. In oncology, a diagnostic may be labeled for a class of
therapeutics rather than a single drug, which changes the commercial calculus for both partners.

### Step 11. Choose the qualification route

**Input:** the context of use and the evidence package.

**Output:** a decision among internal use, scientific advice, FDA biomarker qualification, EMA
qualification of novel methodologies, or device authorization.

**Rationale:** Qualification is voluntary and confers regulatory acceptance for a stated context of
use, but it is slow and requires a substantial package. Many biomarkers are used internally without
qualification. The choice depends on whether the claim needs regulatory acceptance beyond the
sponsor's own submission.

### Step 12. Manage the lifecycle

**Input:** the approved or qualified marker and its assay.

**Output:** version control for the assay, the cut-off, the reagent, and the analysis; a plan for
revalidation when any changes; and a post-approval testing-access plan.

**Rationale:** Assays change. Reagents are reformulated, platforms are upgraded, and cut-offs are
re-derived. Each change can alter the patient population. A marker that was validated on one
platform and is marketed on another requires a bridging study, and a label that does not specify the
assay invites off-label testing with unvalidated methods.

### 5.1 Decision table for the biomarker evidence package

| Evidence pattern | Interpretation | Recommended path | Condition |
|---|---|---|---|
| Analytical validation complete, no clinical association | The assay works; the hypothesis does not | Stop, or test another hypothesis | Do not proceed to trial enrichment |
| Association in one cohort, no replication | Candidate marker | Replicate in an independent cohort | Pre-specify the analysis |
| Association replicated, no interaction | Prognostic marker | Use for stratification or prognosis, not selection | Do not claim predictive benefit |
| Interaction with wide confidence interval | Suggestive predictive marker | Increase sample or pool data | Pre-specify the interaction |
| Interaction replicated, assay performance adequate | Predictive marker | Enrich and co-develop the assay | Lock cut-off and assay |
| Interaction replicated, assay PPV low at prevalence | Predictive biology, poor test | Improve assay or restrict population | Recompute power |
| Qualified biomarker with accepted COU | Regulatory-grade tool | Use within the COU only | Outside COU requires new evidence |

### 5.2 Edge cases that change the arithmetic

**Prevalence estimated from the trial screen.** Screening prevalence in a trial is not the same as
prevalence in the intended-use population, because eligibility criteria select patients. Using the
screen prevalence to compute PPV overstates the performance that will be seen in practice.

**Missing biomarker status.** Patients without a valid sample cannot be classified. If 15% of
screened patients have insufficient tissue, the effective prevalence among testable patients is
higher, and the enrolled population is selected for tissue availability as well as marker status.

**Assay drift.** A change in reagent lot or platform can shift the cut-off. Without a bridging
study, the labelled population changes silently.

**Concomitant standard of care.** A marker's predictive value can depend on the background regimen.
A marker validated with one chemotherapy backbone may not predict with another.

**Regulatory divergence.** The FDA and EMA may accept different evidence for the same marker. A
global programme needs the stricter package or a region-specific strategy.

### 5.3 Design options for a biomarker-driven trial

The arithmetic in Step 9 prices four designs. The choice among them is a portfolio decision, not a
statistical one.

| Design | Who is enrolled | Effect expected | Evidence for a predictive label | Main risk |
|---|---|---|---|---|
| All-comers | Everyone eligible | All-comers effect, usually smaller | None unless an interaction is pre-specified | Large trial, diluted effect, broad label |
| Stratified | Everyone, analysed by marker stratum | Stratum-specific effects | Interaction, if powered | Requires a large sample in each stratum |
| Enrichment | Marker-positive by the assay only | Diluted by assay error | Predictive claim requires the assay to define the population | Screening burden; label restricted to assay-positive |
| Adaptive enrichment | Initially all-comers, then restrict | Interim decision | Interaction at the interim | Complexity; type I error control; operational speed |

Two design properties are often conflated. **Enrichment** changes who is enrolled. **Stratification**
changes how the enrolled are analysed. A trial can enrich without stratifying (and usually should
still stratify by the marker used for enrolment, to check assay performance), and it can stratify
without enriching. The evidence each produces is different, and a protocol that says "biomarker
trial" without specifying which is not a design.

### 5.4 Writing the assay performance specification

The performance specification converts the clinical requirement into acceptance criteria for the
assay. It should be written before analytical validation begins, because a validation performed
against no target can pass every test and still be unfit for the decision.

| Parameter | What it means | Typical acceptance question | Consequence if inadequate |
|---|---|---|---|
| Sensitivity (analytical) | Detection of the analyte at low levels | Does it detect the states that matter? | False negatives, missed eligible patients |
| Specificity (analytical) | Discrimination from related analytes | Does it cross-react? | False positives, diluted effect |
| Precision | Repeatability and reproducibility | Is the result stable across runs and sites? | Misclassification near the cut-off |
| Accuracy | Agreement with a reference method | Is the measured value correct? | Systematic shift in the cut-off |
| Limit of detection | Lowest reliably detected level | Is the low-expression state measurable? | Unreliable classification at the low end |
| Analytical range | Validated interval | Are all clinical samples inside it? | Extrapolation beyond validation |
| Specimen stability | Tolerance to collection and storage | Do archival samples give the same result? | Trial samples not representative |
| Cut-off reproducibility | Consistency of the threshold | Does the same sample classify the same way? | Population changes between sites |

The specification should state the clinical consequence of each failure, not only the acceptance
number. A precision study that shows 10% variability is acceptable if the cut-off is far from the
measured values and unacceptable if most patients sit near it.

### 5.5 Failure patterns that recur across programmes

| Pattern | How it presents | Why it survives review | What it costs |
|---|---|---|---|
| Cut-off fitted to outcome | The cut-off is described as "pre-specified" but its date is after the data cut | The document is dated and the analysis is not | Inflated effect; underpowered trial |
| Prognostic marker used as predictive | Stratum-specific effects reported without an interaction | Two significant subgroup results look convincing | Restricted population with no added benefit |
| Assay validated on the trial population | Performance quoted from the enriched cohort | The cohort is the only one with samples | PPV overstated; label population differs |
| Prevalence borrowed from screening | PPV computed on trial-eligible patients | The number is available and looks clinical | Generalisation failure |
| Single biopsy assumed representative | One archival sample per patient | Heterogeneity is invisible without multiple samples | Misclassification; diluted effect |
| Assay change without bridging | New reagent or platform after validation | The change is described as technical | Silent shift in the labelled population |
| Qualification treated as approval | "The biomarker is qualified" in a label discussion | Qualification and device approval are conflated | Claim outside the context of use |

The first pattern is the most damaging because it is invisible in the final document. A cut-off can
be honestly pre-specified and still be fitted, if the specification was written after the discovery
analysis that selected it. The date that matters is the date of the analysis that generated the
hypothesis, not the date of the protocol.

### 5.6 When the biomarker is unavailable

A programme can proceed without a validated biomarker, but it must then accept a different evidence
package. Three options exist:

- **All-comers with a pre-specified subgroup.** The trial enrols broadly and tests the interaction
  as a secondary objective. The label is broad unless the interaction is convincing.
- **All-comers with a prognostic stratification.** The marker is used to balance or stratify, not to
  select. It improves precision without restricting the label.
- **Deferred selection.** The trial enrols all-comers, collects samples, and uses a validated assay
  retrospectively to define a subgroup for a follow-on study.

Each option requires the assay to be validated before the retrospective analysis is used for a
regulatory claim. A marker that is unavailable at the point of treatment cannot support a
point-of-care selection claim, regardless of its biology.

The fourth option is to delay the biomarker claim to a post-approval study. That keeps the initial
label broad and defers the selection evidence, at the cost of a later label update and a period in
which the drug is used in patients who may not benefit. It is a legitimate strategy only when the
all-comers effect is clinically meaningful on its own.

Whichever option is chosen, the programme should record the population in which the effect is
expected to hold. The population is a design decision, not an inference from whoever happened to
enrol.

## 6. Worked example

> **Note:** The drug, marker, and assay in this example are fictional. The arithmetic follows the
> formulas in Section 3 and is reproducible with the code in Section 7.

### 6.1 The programme

A sponsor is developing **velmatinib** (fictional) for advanced solid tumours. A retrospective
analysis of a randomised Phase II trial found:

- In patients with high expression of the marker (by immunohistochemistry), the progression-free
  survival hazard ratio versus chemotherapy was 0.55 (95% CI 0.38 to 0.80), n = 96.
- In patients with low expression, the hazard ratio was 0.95 (95% CI 0.72 to 1.25), n = 184.
- The interaction p-value was 0.04.

The programme proposes a Phase III in marker-high patients. The assay is a central
immunohistochemistry test with sensitivity 92% and specificity 88% for the true high state,
measured against a research-grade reference assay. The prevalence of true high expression in the
intended second-line population is 8%.

### 6.2 Step 9: the testing funnel

At 8% prevalence, in 1,000 patients:

```text
true high           = 0.08 x 1,000 = 80
true low            = 920
detected high       = 0.92 x 80 + (1 - 0.88) x 920
                    = 73.6 + 110.4 = 184
true positive share = 73.6 / 184 = 40.0%
```

So 40% of assay-positive patients are truly high; 60% are false positives. The negative predictive
value is 0.92 × 0.88 / (0.92 × 0.88 + 0.08 × 0.08) = 0.8096 / 0.8160 = 99.2%. The assay is excellent
at ruling out, and mediocre at ruling in.

To enrol 400 assay-positive patients, the programme must screen 400/0.184 = 2,174 patients. Of the
400 enrolled, 160 are truly high and 240 are not.

### 6.3 Step 9b: the diluted effect

Using the mixture formula with a 24-month control non-progression probability of 0.55:

```text
S_treat(true high)  = 0.55^0.55  = 0.720
S_treat(true low)   = 0.55^0.95  = 0.566
S_treat(mixture)    = 0.40 x 0.720 + 0.60 x 0.566 = 0.288 + 0.340 = 0.628
HR_mixture          = ln(0.628) / ln(0.55) = -0.465 / -0.598 = 0.778
```

The trial will observe approximately 0.78, not 0.55. The biomarker has not failed; the test has
diluted it.

### 6.4 Step 9c: events and sample size

Required events for 90% power, two-sided alpha 0.05:

```text
E = 4 x (1.960 + 1.282)^2 / (ln 0.778)^2
  = 4 x 10.507 / 0.0636
  = 669 events
```

Compare with the event count if every enrolled patient were truly high:

```text
E(HR 0.55) = 4 x 10.507 / (ln 0.55)^2 = 42.03 / 0.3567 = 117.8 events
```

The false positives multiply the required events by 5.7.

Converting events to patients: with 24-month follow-up and a mixture event probability of about
0.46 per patient, the trial needs 669/0.46 = 1,455 assay-positive patients. At 18.4% assay
positivity, that requires screening 7,910 patients. The programme's feasibility estimate assumed
2,200 screens. The trial is not feasible as designed.

### 6.5 Two diagnostics that reveal a broken design

**Diagnostic 1 — the PPV was computed at the wrong prevalence.** A common version of this analysis
uses the prevalence among trial-screened patients, which may be 15-20% because eligibility criteria
enrich for the marker. At 20% prevalence, the same assay gives PPV = 0.92 × 0.20 / (0.92 × 0.20 +
0.12 × 0.80) = 0.184/0.280 = 65.7%, and the dilution is much smaller. If the intended label
population has 8% prevalence, the trial result will not generalise to it. The diagnostic is to
compare the prevalence used in the power calculation with the prevalence in the label population.

**Diagnostic 2 — the interaction was estimated in the same data used to choose the cut-off.** The
Phase II cut-off was selected to maximise the difference between strata, so the 0.55 estimate is
optimistic. An honest power calculation shrinks it toward the all-comers effect. If the all-comers
effect in the same trial was 0.80, the shrunken marker-positive estimate might be 0.65, and the
diluted effect at 40% true-positive share would be worse than 0.78. The diagnostic is to ask whether
the cut-off, the strata, and the effect estimate came from the same dataset; if they did, the
estimate is a fitted maximum, not a prediction.

### 6.6 Vary one assumption: assay specificity

The diagnostic partner proposes a technical improvement that raises specificity from 88% to 95%
without changing sensitivity.

```text
detected high = 0.92 x 80 + 0.05 x 920 = 73.6 + 46.0 = 119.6
PPV           = 73.6 / 119.6 = 61.5%
S_treat       = 0.615 x 0.720 + 0.385 x 0.566 = 0.443 + 0.218 = 0.661
HR_mixture    = ln(0.661) / ln(0.55) = -0.414 / -0.598 = 0.693
E             = 4 x 10.507 / (ln 0.693)^2 = 42.03 / 0.1348 = 312 events
```

At 312 events and 0.46 events per patient, the trial needs about 678 assay-positive patients.
Assay positivity is now 12.0%, so screening is 5,670 patients. The trial moves from infeasible
(7,910 screens) to demanding but feasible (5,670 screens), and the required events fall from 669 to
312.

The decision changes from "do not proceed with the current assay" to "proceed only if the improved
assay is validated and the screening capacity is confirmed". Note that the improved assay also
shrinks the eligible population: fewer patients test positive, so the screening burden remains
high even though the trial is smaller. A better test is not free.

### 6.8 The four designs side by side

The table below uses the same assumptions as the example and adds the all-comers and ideal-assay
options for comparison. Event counts use the formula in Section 3.4; patient counts assume 0.46
events per patient for the enriched designs and 0.47 for all-comers.

| Design | Assumed HR | Events | Patients | Screens | Predictive label supported |
|---|---:|---:|---:|---:|---|
| All-comers | 0.80 | 844 | 1,796 | 0 | No, unless interaction is powered |
| Enrichment, current assay (PPV 40%) | 0.778 | 669 | 1,455 | 7,910 | Yes, for assay-positive patients |
| Enrichment, improved assay (PPV 61.5%) | 0.693 | 312 | 678 | 5,670 | Yes, for assay-positive patients |
| Ideal assay, true-high only | 0.550 | 118 | 256 | 3,200 | Yes, but no such assay exists |

The all-comers design is the smallest in screening burden and the largest in patients. The ideal
assay is the smallest overall but is unattainable. The realistic choice is between the two middle
rows, and the decision depends on whether the improved assay can be validated in time and whether
5,670 screens are available.

A frequent governance error is to compare the all-comers patient number (1,796) with the enriched
patient number (1,455) and conclude that enrichment saves little. The comparison ignores the
screening burden, the probability of success, and the label. A like-for-like comparison states the
events, the screens, the expected effect, and the label claim together.

### 6.7 What the example establishes

1. Sensitivity and specificity describe the assay; PPV describes the patient in front of you, at a
   prevalence.
2. Enrolling assay-positive patients dilutes the treatment effect, and the dilution is computable
   before the trial.
3. The required event count is a fifth-power problem: small changes in the effect produce large
   changes in sample size.
4. The cut-off and the effect estimate must not come from the same data without shrinkage.
5. Assay improvement is a design decision with a feasibility consequence, not a technical detail.

### 6.9 The decision record

The programme's decision record should contain the prevalence, its source and uncertainty; the assay
performance with its validation report identifier; the cut-off and its lock date; the stratum-specific
effects with their confidence intervals; the dilution calculation; the required events and screens;
and the sensitivity of the decision to assay specificity and prevalence. A record missing any of
these cannot be reviewed, and a reviewer who cannot reconstruct the arithmetic will assume the worst.

The record should also state what would change the decision: which assay performance, prevalence, or
effect size would move the programme from go to no-go. That statement is the difference between a
decision and a preference.

## 7. Code, query, or calculation

The script reproduces the worked example and adds the design comparison. It uses only the standard
library and prints every intermediate quantity, so each number can be checked by hand.

```python
import math

def predictive_values(prevalence, sensitivity, specificity):
    """PPV and NPV for a binary test at a stated prevalence."""
    tp = prevalence * sensitivity
    fp = (1 - prevalence) * (1 - specificity)
    tn = (1 - prevalence) * specificity
    fn = prevalence * (1 - sensitivity)
    return tp / (tp + fp), tn / (tn + fn), tp, fp

def diluted_hr(w_true_positive, hr_pos, hr_neg, s_control=0.55):
    """Hazard ratio in a mixture, using survival at one time point."""
    s_treat = w_true_positive * s_control ** hr_pos + \
              (1 - w_true_positive) * s_control ** hr_neg
    return math.log(s_treat) / math.log(s_control)

def events_required(hr, alpha=0.05, power=0.90):
    """Events for a two-sided log-rank test, equal allocation."""
    z_alpha = 1.959964
    z_beta = 1.281552
    return 4 * (z_alpha + z_beta) ** 2 / math.log(hr) ** 2

prevalence = 0.08
for spec in (0.88, 0.95):
    ppv, npv, tp, fp = predictive_values(prevalence, 0.92, spec)
    hr = diluted_hr(ppv, 0.55, 0.95)
    events = events_required(hr)
    positivity = prevalence * 0.92 + (1 - prevalence) * (1 - spec)
    patients = events / 0.46
    print(f"spec {spec:.2f}: PPV {ppv:.3f}  NPV {npv:.3f}  "
          f"HR {hr:.3f}  events {events:.0f}  patients {patients:.0f}  "
          f"screens {patients / positivity:.0f}")

print(f"events at HR 0.55: {events_required(0.55):.0f}")
print(f"events at HR 0.80: {events_required(0.80):.0f}")
```

Expected output (Python 3.12, executed):

```text
spec 0.88: PPV 0.400  NPV 0.992  HR 0.778  events 669  patients 1455  screens 7910
spec 0.95: PPV 0.615  NPV 0.993  HR 0.693  events 312  patients 678  screens 5670
events at HR 0.55: 118
events at HR 0.80: 844
```

Reading the output:

- The PPV moves from 40.0% to 61.5% when specificity improves from 88% to 95%, while sensitivity is
  unchanged. The NPV is above 99% in both cases, which is why a negative result is informative even
  when a positive result is not.
- The diluted hazard ratio moves from 0.778 to 0.693, and the required events fall from 669 to 312.
  The relationship is steep because the event count depends on the inverse square of the log hazard
  ratio.
- The screening burden falls less dramatically, from 7,910 to 5,670, because the improved assay also
  reduces the proportion of patients who test positive.

Assumptions: the mixture formula uses a single time point and a constant hazard ratio within each
stratum; the events-per-patient figure of 0.46 is an assumption, not a derived quantity; and the
design comparison assumes equal allocation and no interim analysis. Changing any of these changes
the numbers but not the direction of the conclusion.

The calculation deliberately does not model assay misclassification of the *effect* (for example,
patients whose true status differs from the assay result but who respond like true positives). That
refinement requires individual-level simulation and is covered in the exercises.

## 8. Data, standards, and regulatory context

### 8.1 The vocabulary standard

The FDA-NIH BEST (Biomarkers, EndpointS, and other Tools) Resource is the reference glossary. It
defines a biomarker as a defined characteristic measured as an indicator of normal biological
processes, pathogenic processes, or responses to an exposure or intervention, and it sets out the
categories used in this topic: susceptibility/risk, diagnostic, monitoring, prognostic, predictive,
response, and safety. It also defines endpoint, surrogate endpoint, clinical benefit, context of
use, and qualification.

BEST is a glossary, not a regulation. Its value is that it forces a programme to use the same word
for the same claim. A protocol that says "predictive biomarker" when it means "prognostic marker" is
not making a terminology error; it is making an evidence error that will surface at the regulatory
review.

### 8.2 Analytical and clinical evidence standards

| Instrument | Scope | Bindingness | What it does not cover |
|---|---|---|---|
| FDA *In Vitro Companion Diagnostic Devices* guidance (final, August 2014) | Co-development, labeling, submission pathway for CDx | FDA guidance; describes expectations, not statute | Biomarker qualification outside a device |
| FDA *List of Authorized Companion Diagnostic Devices* | Authorized CDx and their indications | Informational list | Assays not submitted as devices |
| EU In Vitro Diagnostic Regulation 2017/746 (IVDR) | Device classification, conformity assessment, performance evaluation | EU law, with phased transition | Trial-only laboratory tests used for research |
| CLIA (US) | Laboratory testing quality | US law | Clinical validity or utility |
| ICH E16 | Context, structure, and format of biomarker qualification submissions | ICH guideline | The evidence standard for a specific marker |
| ICH E15 | Definitions for genomic biomarkers, pharmacogenomics, and pharmacogenetics | ICH guideline | Non-genomic biomarkers |
| FDA Biomarker Qualification Program | Qualification of a biomarker for a stated context of use | Voluntary programme under the 21st Century Cures Act | Assay approval |
| EMA Qualification of Novel Methodologies (QoNM) | CHMP qualification opinions for biomarkers and methods | EU scientific opinion | Marketing authorisation |

### 8.3 Qualification versus approval

These are different regulatory acts and are routinely conflated.

- **Biomarker qualification** is acceptance that a biomarker is fit for a stated context of use in
  drug development. It is voluntary, it applies to the biomarker rather than to a specific assay,
  and it does not authorise a test to be marketed. A qualified biomarker may be measured with any
  analytically validated assay within the qualified context.
- **Device authorization** (FDA approval or clearance, or EU IVDR conformity) authorises a specific
  test, with a specific manufacturer, intended use, and performance. A companion diagnostic requires
  this authorisation, and the drug label references the authorised test.
- **Marketing authorisation** of the drug is a separate decision that may reference a biomarker
  without either of the above, if the evidence in the submission supports the claim.

A programme that says "our biomarker is qualified" when it means "our assay is approved" will
misstate what the label can claim. A programme that says "our assay is approved" when it means
"our biomarker is qualified" will misstate what the commercial test can do.

### 8.4 What the standards do not decide

No standard specifies the prevalence at which predictive values must be computed, the acceptable
PPV for a given context, or the size of a treatment-by-marker interaction that justifies a
restricted label. Those are judgements made by the sponsor and reviewed by the regulator. The
standards constrain the form of the evidence, not its sufficiency.

### 8.5 Bridging the trial assay to the commercial assay

A trial is usually run with a laboratory-developed test or a research-use assay, and the marketed
product requires an authorised device. The bridge between them is a study, not an assumption.

| Bridging element | Question | Evidence required | Consequence if absent |
|---|---|---|---|
| Analytical concordance | Do the two assays classify the same samples the same way? | Agreement study on a representative sample set, with confidence intervals | The labelled population differs from the trial population |
| Cut-off transfer | Does the trial cut-off correspond to a defined cut-off on the commercial platform? | Calibration study across the analytical range | Patients near the threshold are reclassified |
| Specimen equivalence | Do archival and fresh samples give the same result? | Stability and concordance study by specimen type | Sites without the required specimen cannot test |
| Reproducibility | Do results agree across sites and operators? | Multi-site precision study | Site-to-site variation changes enrolment |
| Software and readout | Do image analysis or scoring algorithms agree? | Algorithm comparison, including version | A software update changes the result |

The bridging study should be designed with the same rigour as the clinical validation, because it
determines whether the trial's result applies to the patients who will actually be tested. A
concordance of 85% between trial and commercial assays is not a technical detail; it means 15% of
patients in the labelled population would have been classified differently in the trial.

### 8.6 What the guidance does not require

The FDA companion diagnostic guidance expects contemporaneous development and cross-referencing in
labeling, but it does not require the trial assay and the commercial assay to be identical at the
start of the trial. It requires the evidence to support the claim. That gap is where most bridging
problems live: the programme meets the letter of the guidance and still cannot demonstrate that the
marketed test identifies the trial population.

### 8.7 Where the performance data come from

A programme needs four kinds of data before it can run the arithmetic in Section 6, and each comes
from a different source with a different limitation.

| Input | Typical source | Limitation |
|---|---|---|
| Prevalence in the intended population | Registry, epidemiology literature, trial screening data | Rarely measured in the exact label population |
| Assay sensitivity and specificity | Analytical validation against a reference method | The reference method is itself imperfect |
| Stratum-specific treatment effects | Randomised Phase II, prospective-retrospective analysis | Often selected on the same data that defined the strata |
| Expected event rate in the control arm | Historical trials, registry, current standard of care | Changes with standard of care and line of therapy |

The weakest of the four is usually prevalence. A programme that states a single prevalence without a
source is not making an estimate; it is making an assumption. The report should give the source, the
year, the population, and a range.

The most consequential gap is the assay-trial interface. A trial can be designed around a
laboratory-developed test and later transition to a commercial in vitro diagnostic. The standards
require the commercial device to be validated, but they do not require the trial assay and the
commercial assay to be identical. The bridging evidence is the sponsor's responsibility, and its
absence is a common review issue.

## 9. Industry practice and case evidence

### 9.1 How biomarker programmes are organised

Three organisational models are visible across the industry:

- **Translational-led.** A translational medicine group owns the biomarker hypothesis and the assay
  relationships, and hands a validated package to clinical development. Strong on biology, often
  weak on assay lifecycle and diagnostic partnerships.
- **Diagnostics-led.** A companion diagnostic group owns the assay and the device pathway, and
  clinical development configures around it. Strong on regulatory device evidence, sometimes slow to
  accommodate a changing clinical hypothesis.
- **Integrated.** A joint team owns the context of use, the assay, and the trial design, with a
  single decision-maker. The model most likely to catch the PPV problem before Phase III, and the
  hardest to staff.

These descriptions are inferences from public job structures and pipeline disclosures, not findings
from a controlled comparison.

### 9.2 Public qualification opinions

The EMA publishes CHMP qualification opinions for novel methodologies, which are among the few
public documents that show how a regulator reasons about a biomarker's context of use.

- **Centiloid for amyloid PET.** The CHMP qualification opinion was adopted on 30 May 2024 and
  published on 25 June 2024. It accepts Centiloid units as a validated measure of global brain
  amyloid load for enrichment in clinical trials, as an adjunct to visual reads and with quality
  control requirements. Thresholds below 10 Centiloid rule out amyloid pathology, above 30 indicate
  pathology, and 10-30 is a grey zone. Prognostic and predictive uses are outside the opinion.
- **AIM-NASH for MASH histology.** The opinion was adopted on 27 February 2025 and published on 20
  March 2025. It qualifies an artificial-intelligence-based measurement as an aid to a single central
  pathologist for patient enrolment and histology-based endpoint evaluation in phase 2 and 3 trials,
  with a specified scanner requirement and caveats about model lifecycle.
- **GFR slope in chronic kidney disease.** A qualification opinion published on 21 December 2023
  accepted GFR slope as a validated surrogate endpoint in a defined context, which is a stronger
  claim than an enrichment biomarker.

The pattern is instructive. Each opinion names a narrow context of use, specifies what the marker may
and may not be used for, and imposes quality conditions. None of them says "this biomarker is
validated" without qualification. That precision is the standard a programme should imitate in its
own context-of-use statement.

### 9.3 Companion diagnostics in practice

The HER2/trastuzumab pair is the canonical example of co-development: the drug label requires
assessment of HER2 status, and the diagnostic is authorised for that purpose. In oncology, the FDA
maintains a list of authorised companion diagnostics that includes both drug-specific tests and
tests with class labeling for groups of therapeutics.

**Documented:** the existence of the list, the requirement that a CDx be stipulated in the labeling
of the therapeutic, and the expectation of contemporaneous development in the 2014 guidance.
**Inference:** that programmes which engage the diagnostic partner before the Phase III design lock
have fewer bridging problems than those that engage afterwards. The inference is widely held and
plausible, but no public controlled comparison establishes it.

### 9.4 What the evidence does not show

| Claim | Status |
|---|---|
| Predictive biomarkers increase the probability of Phase III success | Plausible and widely believed; not established as an average effect across programmes |
| Most published biomarker associations fail to replicate | Documented in the reproducibility literature; the exact fraction depends on the field and the definition |
| Companion diagnostics shorten development timelines | Inference; the co-development requirement adds work even as it de-risks |
| Assay PPV is routinely computed before Phase III | No public data; the failure pattern in Section 1 suggests it is often omitted |

The last row is the one that matters for practice. There is no registry of biomarker programme
decisions, so the frequency of the PPV omission cannot be measured. The arithmetic is cheap, which
is the argument for doing it regardless.

### 9.5 The economics of screening

Screening burden is the cost that biomarker strategies most often understate. Using the worked
example, 7,910 screens at an illustrative USD 3,000 per screen (assay, pathology review, and site
time) is USD 23.7 million, before any drug is administered. The improved assay reduces this to 5,670
screens, or USD 17.0 million. The difference, USD 6.7 million, is comparable to the cost of the
bridging study that the improved assay requires.

The arithmetic is illustrative, not a benchmark; screening costs vary by assay, country, and
whether an existing test can be used. The point is structural: in a low-prevalence population, the
cost of finding eligible patients can exceed the cost of treating them, and a biomarker strategy
that ignores screening cost will mis-rank the design options.

A second economic effect is the screen-failure rate's impact on sites. A site that screens 20
patients to randomise one may decline the next study. Screening burden is therefore a feasibility
constraint as well as a cost.

The budget comparison should therefore include screening, assay validation, bridging, and device
submission alongside the trial cost. A biomarker strategy that is justified only by a smaller trial
is usually under-costed.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Assay used without analytical validation | Validation report predates the protocol or does not exist | Measurement error treated as biology | Unreliable strata; irreproducible result | Validate before clinical use; agree standard with regulator |
| Cut-off fitted to outcome | Cut-off date after the discovery analysis | Selection on the same data | Inflated effect; failed trial | Derive on one dataset, test on another; lock |
| Prognostic marker used as predictive | No interaction reported | Single-arm association | Restricted population, no added benefit | Require a randomised interaction |
| PPV computed at the wrong prevalence | Prevalence equals the trial screen rate | Convenience sample | Generalisation failure | Use the intended-use prevalence; sensitivity analysis |
| False positives dilute the effect | Power calculation uses the stratum-specific HR | No mixture adjustment | Underpowered trial | Model the mixture before design lock |
| Missing biomarker data ignored | Denominator excludes untested patients | Selection for tissue availability | Biased population; overstated prevalence | Report untested patients; pre-specify handling |
| Single biopsy assumed representative | One sample per patient, no heterogeneity plan | Tumour heterogeneity | Misclassification | Multiple samples or explicit limitation |
| Diagnostic partner engaged late | No device plan at Phase III design | Assay and trial diverge | Bridging study; label delay | Engage before design lock |
| Qualification treated as device approval | "Qualified assay" in labeling discussions | Conflating regulatory acts | Claim not supported | Separate the two tracks explicitly |
| Assay changed without bridging | New reagent lot or platform | Silent shift in cut-off | Labelled population changes | Bridging study; version control |
| Continuous marker dichotomised post hoc | Cut-off chosen to maximise separation | Loss of information and overfitting | Optimistic effect | Pre-specify functional form |
| Regulatory divergence unmanaged | FDA and EMA packages differ | Different evidentiary standards | Duplicate work; delayed filing | Global package or region-specific plan |

The red flag that is easiest to check and hardest to argue with is the date of the cut-off relative
to the date of the analysis that produced the hypothesis. If the cut-off was chosen after the
outcome data were seen, the effect estimate is a maximum, not an estimate, and the trial is powered
on an optimistic number.

### 10.1 Red flags by stage

| Stage | Red flag | What it usually means |
|---|---|---|
| Discovery | Association reported without adjustment for confounders | The marker may be a proxy for stage or histology |
| Analytical validation | Validation uses only samples from the discovery cohort | Performance is estimated on a selected specimen set |
| Cut-off selection | Cut-off described as clinical rather than data-derived | The derivation is undocumented |
| Phase II design | Single-arm with marker strata | No predictive claim is possible |
| Power calculation | Uses the stratum-specific effect without dilution | The trial is underpowered |
| Diagnostic planning | No device partner named at Phase III design | Bridging will be late and expensive |
| Labeling | Claim broader than the assay's validated use | The test cannot identify the labelled population |
| Post-approval | No plan for assay changes | The labelled population may drift |

The stage column matters because a red flag that is tolerable early is disqualifying late. An
unvalidated assay in discovery is normal; an unvalidated assay at Phase III design is a programme
risk.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The marker state is binary and stable | Simplifies the decision rule | Continuous markers, changing expression | Check stability over time and sample |
| The reference standard is correct | Sensitivity and specificity depend on it | Imperfect reference assay | Report reference standard and its error |
| Prevalence is known in the intended population | PPV depends on it | Rare populations, screening effects | Range the prevalence; use registry data |
| Assay performance is constant across sites | Central and local testing assumed equivalent | Site variability, specimen handling | Reproducibility study across sites |
| The stratum-specific effects are estimated without bias | Power calculation uses them | Cut-off selection, subgroup selection | Shrink estimates; pre-specify |
| The mixture is stable over time | The dilution formula assumes it | Differential dropout, crossover | Sensitivity analysis by time |
| The marker does not change with treatment | Classification at baseline applies throughout | Treatment-induced marker change | Serial sampling in a subset |
| The trial population resembles the label population | Generalisation | Eligibility criteria select healthier patients | Compare baseline characteristics |
| The commercial assay matches the trial assay | Label references the device | Bridging absent | Bridging study; concordance analysis |
| The diagnostic is available where patients are treated | Utility requires access | Centralised testing, reimbursement | Testing-access plan |

The central limit is that all of these are estimates with uncertainty, and the decision model treats
them as point values. A programme should present the PPV and the required events as ranges, and
should state which input drives the range. In most cases it is the prevalence, because it is the
least well measured of the inputs.

Two further assumptions determine whether the marker can be used at all, as distinct from whether it
works:

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| A confirmatory method is available for discordant results | Some patients need a second test | No orthogonal assay exists | Concordance study with a second platform |
| Testing capacity is available at launch | Utility requires access | Central laboratory only; reimbursement absent | Testing-access plan before launch |

A marker with a perfect evidence package and no testing pathway has no clinical utility.

A related limit is that the model treats the assay result as fixed at baseline. In practice, a
patient's marker status can change between the archival sample and the treatment decision, and a
retest can move a patient across the cut-off. The programme should state whether the enrolment
result or a confirmatory result defines eligibility, and should report the discordance rate if both
are collected.

## 12. Exercises

**Exercise 1 — Classify the claim (recall, 20 minutes).**
Deliverable: for six marker-outcome statements, label each as prognostic, predictive, pharmacodynamic,
safety, monitoring, or diagnostic, and name the design that would support it.
Acceptance criteria: each label justified in one sentence; at least two statements correctly
identified as requiring a randomised comparison.
Time: 20 minutes.

**Exercise 2 — Predictive values (application, 25 minutes).**
Deliverable: compute PPV and NPV for prevalence 5%, sensitivity 90%, specificity 95%; then recompute
at prevalence 30%.
Acceptance criteria: both prevalence scenarios shown; explanation of why sensitivity is unchanged
and PPV is not; arithmetic correct to two decimal places.
Time: 25 minutes.

**Exercise 3 — Dilution calculation (application, 35 minutes).**
Deliverable: given HR 0.60 in true positives, 0.95 in true negatives, and PPV 45%, compute the
mixture HR at a 24-month control event-free probability of 0.50, and the events required for 90%
power.
Acceptance criteria: intermediate survival probabilities shown; mixture HR within 0.02 of the
reference; event count stated with the formula.
Time: 35 minutes.

**Exercise 4 — Design comparison (design, 45 minutes).**
Deliverable: a table comparing all-comers, enrichment, and stratified designs for a marker with 12%
prevalence and assay PPV 55%, including events, patients, screens, and label claim.
Acceptance criteria: assumptions stated; screening burden included; recommendation justified by more
than sample size.
Time: 45 minutes.

**Exercise 5 — Cut-off audit (critique, 40 minutes).**
Deliverable: a 400-word audit of a programme whose cut-off was selected by maximising the
log-rank statistic in the same trial used to estimate the effect, with the correction required.
Acceptance criteria: identifies the selection mechanism; states the direction of bias; proposes a
replication or shrinkage approach; states the effect on power.
Time: 40 minutes.

**Exercise 6 — Assay specification (design, 40 minutes).**
Deliverable: an assay performance specification for a predictive immunohistochemistry marker, with
acceptance criteria for precision, accuracy, specimen stability, and cut-off reproducibility.
Acceptance criteria: each criterion linked to a clinical consequence; specimen types named; a
bridging plan for assay changes.
Time: 40 minutes.

**Exercise 7 — Regulatory pathway (application, 35 minutes).**
Deliverable: a one-page map of the US and EU pathways for a drug and its companion diagnostic,
showing which submissions are independent and which are linked.
Acceptance criteria: distinguishes qualification from device authorisation; names the relevant
instruments; identifies the sequencing risk.
Time: 35 minutes.

**Exercise 8 — Red-team a biomarker plan (critique, 50 minutes).**
Deliverable: a red-team memo on a plan that proposes enrichment with an unvalidated assay, a
data-derived cut-off, and a Phase III powered on the Phase II subgroup effect.
Acceptance criteria: at least six defects; at least three quantified; a corrected design proposed.
Time: 50 minutes.

## 13. Self-check answers

**Exercise 1.** Examples: "high marker expression is associated with shorter survival" is prognostic;
"patients with the alteration respond better to the drug than those without" is predictive and
requires a randomised interaction; "marker levels fall within 24 hours of dosing" is pharmacodynamic;
"marker-positive patients have more hepatotoxicity" is a safety claim requiring a randomised
comparison; "marker levels track relapse" is monitoring; "the marker detects the disease" is
diagnostic. A statement that reports a single-arm association with response cannot establish a
predictive claim.

**Exercise 2.** At 5%: PPV = 0.05 × 0.90 / (0.05 × 0.90 + 0.95 × 0.05) = 0.045/0.0925 = 48.6%;
NPV = 0.95 × 0.95 / (0.95 × 0.95 + 0.05 × 0.10) = 0.9025/0.9075 = 99.4%. At 30%: PPV = 0.27 /
(0.27 + 0.70 × 0.05) = 0.27/0.305 = 88.5%; NPV = 0.665/(0.665 + 0.03) = 95.7%. Sensitivity and
specificity are properties of the test; predictive values depend on prevalence.

**Exercise 3.** S_treat(pos) = 0.50^0.60 = 0.660; S_treat(neg) = 0.50^0.95 = 0.518;
mixture = 0.45 × 0.660 + 0.55 × 0.518 = 0.297 + 0.285 = 0.582. HR = ln(0.582)/ln(0.50) =
−0.541/−0.693 = 0.781. Events = 4 × 10.507/(ln 0.781)^2 = 42.03/0.0611 = 688. Accept 675-700
depending on rounding.

**Exercise 4.** With 12% prevalence and PPV 55%, the true-positive fraction is 0.55. If HR_true is
0.55 and HR_false is 0.95, the mixture HR is approximately 0.75. Events for 90% power are about 780.
At 0.46 events per patient, patients are about 1,700 and screens about 1,700/0.115 = 14,800 if assay
positivity is 11.5%. An all-comers design assumes HR 0.80, requires 844 events and about 1,796
patients with no screens. A stratified design requires the interaction to be powered, which usually
means several hundred patients per stratum. The recommendation depends on whether the label requires
a predictive claim and whether 14,800 screens are available.

**Exercise 5.** The bias is optimistic: selecting the cut-off that maximises the test statistic
inflates the estimated effect and understates the standard error. The correction is to derive the
cut-off in one dataset and test it in an independent one, or to shrink the effect toward the
all-comers estimate before powering. The effect on power is large: a true HR of 0.70 rather than the
observed 0.55 raises the required events from 118 to 330 at 90% power.

**Exercise 6 rubric.** Full marks require acceptance criteria for precision, accuracy, specimen
stability, and cut-off reproducibility, each linked to a clinical consequence; specimen types and
handling specified; and a bridging plan for reagent or platform changes. Deduct marks for criteria
without a consequence, or for a specification that omits archival tissue.

**Exercise 7.** In the US, the drug follows an NDA or BLA and the diagnostic follows a PMA or 510(k),
with the 2014 guidance expecting contemporaneous development and cross-referencing in labeling. In
the EU, the drug follows a marketing authorisation and the diagnostic follows IVDR conformity
assessment; EMA qualification of a biomarker is a separate, voluntary opinion that does not
authorise either. The sequencing risk is that the device submission lags the drug, leaving the drug
label referencing a test that is not yet available.

**Exercise 8.** Defects: unvalidated assay; data-derived cut-off; Phase III powered on a selected
subgroup effect; no PPV calculation; no prevalence estimate; no diagnostic partner; no plan for
missing biomarker data; no bridging plan. Quantify at least three: PPV at the intended prevalence;
the diluted HR; the event count at the diluted HR versus the planned count.

## 14. Interview questions

**1. What is the difference between a prognostic and a predictive biomarker?**
Strong: prognostic predicts outcome independent of treatment; predictive predicts differential
treatment effect and requires a randomised interaction. Gives an example of a marker that is both.
Weak: says prognostic is "about the disease" and predictive is "about the drug" without the design
consequence.

**2. Why does PPV change with prevalence when sensitivity does not?**
Strong: sensitivity and specificity are conditional on the true state; PPV is conditional on the
test result and therefore depends on how many true positives exist in the population. Gives the
arithmetic. Weak: says "because prevalence affects the base rate" without being able to compute it.

**3. How would you decide whether to enrich a trial?**
Strong: model prevalence, assay performance, dilution, events, screens, label, and diagnostic
timeline; compare designs on all dimensions. Weak: enrich because the subgroup looked better.

**4. What is the risk of choosing the cut-off from the same data that showed the effect?**
Strong: the effect is a selected maximum, optimistic, and the trial will be underpowered; the fix is
independent derivation or replication. Weak: says it is "not ideal" without quantifying.

**5. What does a companion diagnostic require that a laboratory-developed test does not?**
Strong: device authorisation, analytical and clinical validation to the device standard, labeling
linked to the drug, and availability. Weak: says "FDA approval" without the linkage.

**6. How do you handle patients with missing biomarker results?**
Strong: report them, pre-specify the analysis population, consider the selection mechanism, and test
sensitivity to their exclusion. Weak: excludes them silently.

**7. What is the difference between biomarker qualification and assay approval?**
Strong: qualification accepts the biomarker for a context of use and is assay-independent;
authorisation approves a specific test. Weak: uses the terms interchangeably.

**8. How would you assess whether a biomarker programme is ready for Phase III?**
Strong: checks assay validation, cut-off lock, prevalence estimate, PPV, interaction evidence,
diagnostic partner, and the power calculation. Weak: checks that the Phase II result was significant.

**9. What would make you abandon a biomarker?**
Strong: failed replication, no interaction, assay performance too poor at the intended prevalence, or
no feasible testing pathway. Weak: a single negative study.

**10. How do regulatory expectations differ between the US and the EU for biomarkers?**
Strong: distinguishes FDA qualification and device pathways from EMA qualification opinions and
IVDR, and notes that a global programme needs the stricter package. Weak: assumes the two are
harmonised.

## 15. Key terms

- **Analytical validation** — demonstration that an assay measures the analyte accurately and
  reproducibly.
- **All-comers design** — enrolment without biomarker selection.
- **Biomarker** — a defined characteristic measured as an indicator of biology or response.
- **Biomarker qualification** — regulatory acceptance of a biomarker for a stated context of use.
- **Clinical utility** — evidence that using the marker changes decisions and improves outcomes.
- **Clinical validation** — association of the marker with outcome in the intended population.
- **Companion diagnostic (CDx)** — an in vitro diagnostic essential for the safe and effective use
  of a therapeutic.
- **Context of use (COU)** — the explicit statement of how a biomarker will be used.
- **Cut-off** — the threshold defining a positive result.
- **Dilution** — reduction in observed effect from including non-responders.
- **Enrichment** — selecting patients by marker to increase the expected effect.
- **Interaction** — a difference in treatment effect between marker strata.
- **Laboratory-developed test (LDT)** — a test developed and performed in one laboratory.
- **Negative predictive value (NPV)** — proportion of test negatives who are true negatives.
- **Pharmacodynamic biomarker** — indicates biological activity of a treatment.
- **Positive predictive value (PPV)** — proportion of test positives who are true positives.
- **Prevalence** — proportion of the population with the condition or marker state.
- **Prognostic biomarker** — predicts outcome independent of treatment.
- **Predictive biomarker** — predicts differential response to treatment.
- **Sensitivity** — proportion of true positives detected.
- **Specificity** — proportion of true negatives correctly classified.
- **Surrogate endpoint** — an endpoint substituted for clinical benefit.
- **Tumour heterogeneity** — variation in marker expression within and between lesions.

## 16. Sources

- FDA-NIH Biomarker Working Group. *BEST (Biomarkers, EndpointS, and other Tools) Resource*.
  https://www.ncbi.nlm.nih.gov/books/NBK326791/ (accessed 25 October 2026).
- BEST Glossary. https://www.ncbi.nlm.nih.gov/books/NBK338448/ (accessed 25 October 2026).
- BEST, Understanding Prognostic versus Predictive Biomarkers.
  https://www.ncbi.nlm.nih.gov/books/NBK402284/ (accessed 25 October 2026).
- US Food and Drug Administration. *Biomarker Qualification Program*.
  https://www.fda.gov/drugs/biomarker-qualification-program/biomarker-qualification-program
  (accessed 25 October 2026).
- US Food and Drug Administration. *Qualification Process for Drug Development Tools*. January 2014.
  https://www.fda.gov/files/drugs/published/Qualification-Process-for-Drug-Development-Tools.pdf
  (accessed 25 October 2026).
- US Food and Drug Administration. *In Vitro Companion Diagnostic Devices: Guidance for Industry
  and Food and Drug Administration Staff*. August 2014.
  https://www.fda.gov/media/81309/download (accessed 25 October 2026).
- US Food and Drug Administration. *List of FDA-Authorized Companion Diagnostic Devices*.
  https://www.fda.gov/medical-devices/in-vitro-diagnostics/list-fda-authorized-companion-diagnostic-devices-in-vitro-and-imaging-tools
  (accessed 25 October 2026).
- European Medicines Agency. *Qualification of novel methodologies for medicine development: CHMP
  qualification opinions*.
  https://www.ema.europa.eu/en/human-regulatory-overview/research-development/scientific-advice-protocol-assistance/qualification-novel-methodologies-medicine-development/qualification-novel-methodologies-medicine-development-chmp-qualification-opinions
  (accessed 25 October 2026).
- European Medicines Agency. *Qualification opinion for Centiloid measure of amyloid PET*. Adopted
  30 May 2024, published 25 June 2024.
  https://www.ema.europa.eu/en/documents/other/qualification-opinion-centiloid-measure-amyloid-pet-quantify-brain-amyloid-deposition_en.pdf
  (accessed 25 October 2026).
- European Medicines Agency. *Qualification opinion for AI-based measurement of NASH histology*.
  Adopted 27 February 2025, published 20 March 2025.
  https://www.ema.europa.eu/en/documents/other/qualification-opinion-artificial-intelligence-based-measurement-non-alcoholic-steatohepatitis-histology-liver-biopsies-determine-disease-activity-nash-mash-clinical-trials_en.pdf
  (accessed 25 October 2026).
- ICH. *E16 Biomarkers Related to Drug or Biotechnology Product Development*.
  https://www.ich.org/page/efficacy-guidelines (accessed 25 October 2026).
- European Union. *Regulation (EU) 2017/746 on in vitro diagnostic medical devices*.
  https://eur-lex.europa.eu/eli/reg/2017/746/oj (accessed 25 October 2026).

> **Note:** The drug, marker, and assay in Section 6 are fictional. The EMA opinions cited in
> Section 9 are real and dated; the interpretation of them is the course's own.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_data-in-pharma.md`](02_data-in-pharma.md)
- Next topic: [`04_feasibility-and-recruitment.md`](04_feasibility-and-recruitment.md)
- Related: [`05_cdisc-and-trial-standards.md`](05_cdisc-and-trial-standards.md)
- Related: [`07_rwe-in-pharma.md`](07_rwe-in-pharma.md)
- Survey file: [`../../13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md)
- Labs: [`../../labs/`](../../labs/)
