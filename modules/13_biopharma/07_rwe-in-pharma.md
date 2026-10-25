# Real-World Evidence: Target Trial Emulation and the Cost of Confounding

> Module 13 — Biopharma and Clinical Development
> Topic 07 of 10
> Estimated study time: 5 hours
> Prerequisites: Module 08 (EHR and claims); Module 04 (statistics); topics 01 and 06
> Version 2026.3

## 1. Why this topic exists

Real-world data (RWD) are collected during routine care rather than in a trial: claims, electronic
health records, registries, prescriptions, and wearables. Real-world evidence (RWE) is the evidence
derived from analysing them. The commercial appeal is obvious. A trial costs tens of millions and
takes years; a database analysis costs a fraction and can be completed in months. The evidentiary
appeal is less obvious, and it is where programmes get into trouble.

The central problem is that treatment in routine care is not assigned at random. Clinicians choose
treatments based on prognosis, comorbidity, prior response, cost, and preference. Those choices
create associations between treatment and outcome that reflect the choice, not the treatment. A drug
given to healthier patients will appear to reduce mortality; a drug given to sicker patients will
appear to increase it. The direction of the bias depends on the prescribing pattern, and the size can
exceed the true effect.

A second problem is time. Real-world analyses routinely introduce immortal time, where patients are
classified as treated during a period in which they could not have died, or where follow-up starts
before treatment initiation. These errors are mechanical and severe: they can produce a large
apparent benefit from a drug that does nothing.

This topic teaches the target trial emulation framework, which forces the analyst to specify the
randomised trial that the observational analysis is trying to imitate, and then to align the
observational analysis to it. It teaches the design choices that determine whether the analysis is
credible: eligibility, treatment definition, time zero, outcome ascertainment, confounding control,
and sensitivity analysis. It follows the safety topic, because post-marketing surveillance is a
real-world problem, and precedes the AI topic, because many AI claims in development rest on
real-world data.

### 1.1 The failure pattern

1. A sponsor wants to support a label expansion with RWE.
2. A claims database is queried for patients receiving the drug and patients receiving an older
   therapy.
3. The crude analysis shows a 33% reduction in the outcome (rate ratio 0.67).
4. The result is presented to a regulator as supportive evidence.
5. The regulator asks how time zero was defined and how confounding was addressed.
6. The analysis had started follow-up at the first prescription and classified patients as treated
   from the first prescription, which is reasonable, but had not adjusted for the fact that treated
   patients were younger and healthier.
7. After adjustment, the rate ratio is 0.96 (95% CI 0.88 to 1.05), and the apparent benefit
   disappears.

Steps 3 and 7 use the same data. The difference is the design and the confounding control, and the
regulator's questions in step 5 are designed to detect exactly that difference.

### 1.2 Who is blocked when RWE is weak

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Market access team | A payer argument built on a biased estimate | Receives an analysis it did not design |
| Regulatory affairs | A submission that fails the evidentiary bar | Cannot add confounding control after the fact |
| Epidemiology team | Credibility of the whole RWE programme | Depends on data access and clinical input |
| Clinical team | A label expansion that would help patients | Cannot control the database |
| Payer | Budget allocated on a wrong effect estimate | Sees the result, not the design |
| Patient | Access delayed or denied | Has no role in the analysis |

### 1.3 The industry question

> What randomised trial is this analysis trying to emulate, and how closely does it succeed?

The question is the target trial framework in one sentence. An analysis that cannot name the trial it
emulates cannot be assessed, because there is no benchmark for its design choices.

### 1.4 Four questions a reviewer will ask

1. **What is the target trial, and is it the right one for the decision?** The comparator and the
   population must match the decision, not merely the data.
2. **When does follow-up start, and why?** Time zero is where most bias enters.
3. **What was done about confounding, and how do you know it worked?** Balance diagnostics are the
   evidence, not the method name.
4. **How fragile is the result?** An E-value and a negative control answer this better than a
   p-value.

A package that answers these four in the first two pages is easier to review than one that answers
them in an appendix.

## 2. Learning outcomes

By the end of this topic you can:

1. **Specify** a target trial with its eligibility criteria, treatment strategies, assignment,
   time zero, outcomes, follow-up, and estimand.
2. **Align** an observational analysis to a target trial, including time zero and treatment
   definitions.
3. **Identify** the principal confounders for a given question and choose a control method
   (stratification, matching, weighting, or adjustment).
4. **Compute** crude and adjusted rates and rate ratios, and explain the direction of the change.
5. **Compute** an E-value and interpret it as the strength of unmeasured confounding needed to
   explain away the result.
6. **Diagnose** immortal time bias, healthy-user bias, and outcome misclassification in a supplied
   analysis.
7. **Assess** whether an RWE package meets the FDA's evidentiary expectations and state what
   additional evidence would be required.

## 3. The core mental model

### 3.1 The target trial

```text
 Component            Target trial question                      Observational analogue
 ---------            ---------------------                      ----------------------
 Eligibility          Who would be enrolled?                     Inclusion/exclusion at index date
 Treatment strategies What treatments are compared?              Exposure definitions with grace period
 Assignment           How would treatment be assigned?           Observed initiation (not random)
 Time zero            When does follow-up start?                 Date of eligibility + treatment decision
 Outcomes             What is measured, how, over what window?   Claims/EHR algorithms with validation
 Follow-up            Until when?                                Until outcome, disenrolment, death, or end
 Causal contrast      ITT, per-protocol, or as-treated?          Emulated with cloning/weights or naively
```

The framework works by making the observational analysis answer the same question as the trial. If
the trial would start follow-up at randomisation, the observational analysis starts at the equivalent
moment: when a patient becomes eligible and a treatment decision is made. If the trial would compare
initiation strategies, the observational analysis compares initiators, not prevalent users.

### 3.2 Why time zero is the most consequential choice

```text
 Wrong:  eligibility ----(wait)----> first prescription ----(follow-up)----> outcome
                                                              ^
                                              patients who die or improve before
                                              this point are excluded, creating
                                              immortal time and selection

 Right:  eligibility + treatment decision = time zero ----(follow-up)----> outcome
```

If follow-up starts at the first prescription but eligibility was established earlier, the period
between eligibility and prescription is immortal: the patient had to survive to be treated. The
result is a spurious benefit proportional to the length of that period.

The same problem appears in reverse when prevalent users are included: patients who tolerated the
drug for a year are selected for survival, and their subsequent outcomes are better than those of
new users.

### 3.3 Confounding: the diagram

```text
                 Age, comorbidity, severity
                    /                 \
                   v                   v
              Treatment -----------> Outcome
                   ^                   ^
                   |                   |
            Prescribing preference   Health system access
```

A confounder is a cause of both treatment and outcome. Adjusting for it removes the path through the
confounder. Adjusting for a variable caused by treatment (a mediator) removes part of the effect and
is a different error. Adjusting for a variable caused by both treatment and outcome (a collider)
introduces bias where none existed.

The practical rule: adjust for variables that are causes of both, measured before treatment, and not
affected by treatment.

### 3.4 The estimand determines the method

| Estimand | Question | Observational method | Common error |
|---|---|---|---|
| Intention-to-treat | What is the effect of being assigned to start treatment? | Clone-censor-weight or ITT-like follow-up from time zero | Starting follow-up at first dispensing |
| Per-protocol | What is the effect of adhering to the strategy? | Weighting for adherence, cloning | Ignoring discontinuation |
| As-treated | What is the effect of the treatment actually received? | Time-varying exposure analysis | Immortal time from future exposure |

Most regulatory RWE questions are about the effect of a treatment strategy, which corresponds to the
per-protocol or ITT estimand, not to the as-treated effect. An as-treated analysis answers a
different question and is often the source of the largest biases.

### 3.5 What the data can and cannot support

| Question | Claims | EHR | Registry | Trial |
|---|---|---|---|---|
| Diagnosis | Coded, incomplete | Coded plus notes | Validated | Adjudicated |
| Treatment | Dispensed, not taken | Prescribed, not taken | Recorded | Administered |
| Outcome | Coded, variable validity | Coded plus notes | Validated | Adjudicated |
| Severity | Poorly captured | Partly captured | Captured | Measured |
| Confounders | Limited to claims | Better | Disease-specific | Randomised |
| Lab values | Absent | Present | Variable | Present |

The table explains why the same question can be answered well in one data source and poorly in
another. A claims database may be adequate for a hard outcome such as hospitalisation and inadequate
for a question that requires disease severity.

### 3.6 The four biases that explain most failures

| Bias | Mechanism | Diagnostic | Remedy |
|---|---|---|---|
| Immortal time | Follow-up before exposure | Timeline audit | Align time zero |
| Healthy user | Healthier patients treated | Baseline balance table | Adjust, active comparator |
| Confounding by indication | Sicker patients treated | Baseline balance, negative control | Adjust, restriction |
| Outcome misclassification | Algorithm error | Validation study | Better algorithm, sensitivity |

The first three produce a spurious benefit; the fourth usually attenuates an effect but can produce
either direction if the misclassification differs by exposure. A report that checks these four has
covered most of the methodological risk.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Real-world data (RWD) | Data collected during routine care | Treated as synonymous with RWE | Data category |
| Real-world evidence (RWE) | Evidence derived from RWD analysis | Treated as any observational study | Evidence category |
| Target trial | The randomised trial the observational analysis emulates | Omitted from the protocol | Design specification |
| Target trial emulation | Aligning an observational analysis to a target trial | Used for any observational study | Method |
| Time zero | The start of follow-up, aligned to the treatment decision | Set at first prescription without checking eligibility | Date |
| Immortal time | A period during which the outcome cannot occur, misclassified as exposed | Ignored | Bias |
| Prevalent user | A patient already on treatment at cohort entry | Included with new users | Population |
| New-user design | Cohort entry at treatment initiation | Confused with prevalent-user design | Design |
| Grace period | The window after time zero in which treatment must start | Omitted | Duration |
| Confounder | A cause of both treatment and outcome | Confused with a mediator | Variable |
| Mediator | A variable on the causal path from treatment to outcome | Adjusted as if a confounder | Variable |
| Collider | A variable caused by both treatment and outcome | Adjusted, inducing bias | Variable |
| Propensity score | Probability of treatment given covariates | Treated as a causal estimate | Probability |
| Inverse probability of treatment weighting (IPTW) | Weighting to create a pseudo-population balanced on covariates | Used without checking balance | Weight |
| Standardised mean difference | Balance metric between groups | Confused with a significance test | Proportion |
| Effective sample size | Sample size after weighting | Ignored | Count |
| Negative control | An outcome that should not be affected by treatment | Omitted | Design check |
| Positive control | A comparison with a known answer | Omitted | Design check |
| E-value | Minimum strength of unmeasured confounding needed to explain away an association | Treated as a probability | Ratio |
| Rate ratio | Ratio of incidence rates | Confused with a risk ratio | Ratio |
| Hazard ratio | Ratio of hazards | Interpreted as a risk ratio | Ratio |
| Misclassification | Error in measuring exposure, outcome, or confounder | Assumed non-differential | Bias |
| Healthy-user bias | Healthier patients are more likely to receive a new treatment | Ignored | Bias |
| Channeling bias | Patients with a specific prognosis are channeled to a drug | Ignored | Bias |
| External control arm | A comparator group drawn from outside the trial | Confused with a historical control | Design |
| Data lock | The date at which the RWD extract is fixed | Confused with trial database lock | Date |
| Feasibility count | Number of patients meeting computable criteria | Treated as the analysis cohort | Count |
| Algorithm validation | Comparison of a computable phenotype with a reference | Omitted | Study |
| Pre-registration | Public registration of the protocol before analysis | Treated as optional | Practice |

### 4.1 Emulation is a design discipline, not a statistical technique

The framework does not require a particular estimator. It requires that every design choice be
specified before the analysis and aligned to the trial being emulated. Two analyses of the same data
can use different estimators and reach the same conclusion if both align to the same target trial.
Two analyses can use the same estimator and reach different conclusions if they align to different
target trials.

## 5. The mechanics, step by step

This procedure specifies and executes a target trial emulation for a comparative effectiveness or
safety question. It assumes access to a patient-level dataset with treatment, outcome, and covariate
information, and to clinical input on the question.

### Step 1. Write the policy question and the causal contrast

**Input:** the decision the evidence is meant to inform.

**Output:** a question in causal language, naming the population, the two strategies, the outcome,
the time horizon, and the contrast.

**Rationale:** "Is drug A effective?" is not answerable. "In adults with the indication who are
initiating a first-line therapy, what is the effect of initiating A versus B on hospitalisation over
24 months?" is. The contrast determines the estimand, which determines the method.

### Step 2. Specify the target trial

**Input:** the question and clinical input.

**Output:** a table with the seven components: eligibility, treatment strategies, assignment, time
zero, outcomes, follow-up, and estimand.

**Rationale:** The table is the benchmark for every subsequent choice. It also exposes questions that
the data cannot answer: if the trial would measure severity and the database does not, the emulation
has a limitation that must be stated before the analysis, not after.

### Step 3. Select and characterise the data source

**Input:** candidate databases.

**Output:** a data source assessment covering coverage, enrolment, coding, outcome validation, and
the availability of confounders.

**Rationale:** A database that does not capture the outcome or the main confounders cannot support
the question, regardless of its size. The assessment should include the date range, the population
covered, and the coding systems, and it should be written before the analysis.

**Branch point:** claims versus EHR versus registry. Claims have broad coverage and hard outcomes but
limited clinical detail; EHR have clinical detail but fragmented coverage; registries have disease
specificity but selection. The choice determines which confounders can be addressed.

### Step 4. Define eligibility at time zero

**Input:** the target trial's criteria.

**Output:** computable criteria applied at the index date, with the washout period and the lookback
window.

**Rationale:** Eligibility must be assessed at the same moment as the treatment decision. A criterion
assessed after time zero can be affected by treatment and introduces bias. The washout period ensures
new users rather than prevalent users.

### Step 5. Define exposure and the grace period

**Input:** prescription or dispensing data.

**Output:** an exposure definition with the agent, dose, and route; a grace period; and a rule for
discontinuation and switching.

**Rationale:** The grace period allows the treatment to start after the decision without creating
immortal time. A common approach is to assign exposure at time zero based on initiation within a
defined window, and to analyse both initiation and adherence.

### Step 6. Define the outcome and its ascertainment

**Input:** the target trial's outcome and the database's coding.

**Output:** an outcome algorithm with codes, settings, and a validation estimate.

**Rationale:** Outcome misclassification biases toward the null if non-differential and in either
direction if differential. The algorithm's positive predictive value should be stated, ideally from a
validation study in the same database.

### Step 7. Identify confounders and draw the diagram

**Input:** clinical knowledge and the data dictionary.

**Output:** a list of confounders measured before time zero, with a causal diagram and a rationale
for inclusion or exclusion.

**Rationale:** The list should include the variables that cause both treatment and outcome, and
exclude mediators and colliders. A diagram makes the assumptions visible and allows a reviewer to
disagree with a specific arrow rather than with the whole analysis.

### Step 8. Choose the confounding control method

**Input:** the confounder list, the sample size, and the estimand.

**Output:** a method: stratification, matching, IPTW, standardisation, or multivariable outcome
modelling, with diagnostics.

**Rationale:** The methods differ in what they assume and what they can balance. IPTW balances
measured confounders and provides a marginal estimate; matching balances and can be easier to explain;
outcome modelling requires a correctly specified model. The choice should be justified, not defaulted.

**Branch point — method selection:**

| Situation | Suitable method | Why | Diagnostic |
|---|---|---|---|
| Many confounders, moderate sample | IPTW or overlap weighting | Balances many variables at once | Standardised mean differences, effective sample size |
| Small sample, strong confounding | Matching with caliper | Explicit comparability | Balance table, matched sample size |
| Time-varying treatment | Marginal structural model | Handles time-varying confounding | Weight distribution, positivity |
| Rare outcome | Outcome regression with few covariates | Stable estimation | Calibration, influence diagnostics |
| Policy question about a population | Standardisation | Marginal effect in the population | Covariate distribution comparison |

### Step 9. Estimate and report

**Input:** the analysis dataset.

**Output:** crude and adjusted estimates with confidence intervals, and the distribution of weights.

**Rationale:** The crude estimate is not a nuisance; it shows the direction and size of confounding.
A report that shows only the adjusted estimate hides the most important information about the data.

### Step 10. Run sensitivity analyses

**Input:** the primary analysis.

**Output:** analyses under alternative designs: different grace periods, negative control outcomes,
positive control comparisons, E-values, and quantitative bias analysis.

**Rationale:** Sensitivity analyses test whether the conclusion depends on a design choice. A result
that survives several plausible alternatives is stronger than one that does not, and a result that
reverses under one alternative is a finding about the fragility of the evidence.

### Step 11. Engage the regulator before the analysis

**Input:** the protocol and the target trial table.

**Output:** a pre-submission meeting record or scientific advice, with the regulator's comments.

**Rationale:** The FDA's RWE guidance emphasizes early engagement and pre-specification. An analysis
designed without regulatory input may answer a question the regulator did not ask, and the cost of
redesign after the analysis is the whole analysis.

### Step 12. Report transparently and register the protocol

**Input:** the completed analysis.

**Output:** a report with the protocol, the code, the data source version, the design table, the
results, and the limitations.

**Rationale:** RWE is assessed on its design as much as its result. A report that cannot be
reproduced cannot be assessed, and a protocol registered before the analysis is the strongest
evidence that the design was not chosen after seeing the result.

### 5.1 The design table in practice

A target trial table is useful only if each row is specific enough to be implemented. The table below
shows the difference between a specification and a wish.

| Component | Weak specification | Implementable specification |
|---|---|---|
| Eligibility | Adults with the disease | Age ≥ 18 at index; ≥ 2 diagnosis codes 30 days apart; no prior treatment in 12 months |
| Treatment | Drug A versus drug B | First dispensing of A or B after a 12-month clean period; grace period 30 days |
| Time zero | First prescription | Date of first dispensing after eligibility, with eligibility confirmed on that date |
| Outcome | Hospitalisation for the event | Primary diagnosis code for the event in the first position, inpatient setting |
| Follow-up | 24 months | Until outcome, death, disenrolment, or 24 months after time zero |
| Estimand | Effect of treatment | Per-protocol effect of initiating and remaining on treatment, with discontinuation handled by weighting |

The right-hand column is what a programmer can build and a reviewer can audit. The left-hand column
is what most protocols contain.

### 5.2 Confounder selection without a fishing expedition

A confounder list assembled from every available variable produces unstable estimates and hides the
assumptions. The list should be built from three sources:

1. **Clinical knowledge** of what determines the treatment choice and the outcome.
2. **The published literature** on the question, including prior observational studies.
3. **A causal diagram** that makes the proposed relationships explicit.

| Variable class | Include? | Reason |
|---|---|---|
| Age, sex | Yes | Causes of both treatment and outcome |
| Comorbidity index | Yes | Causes of both |
| Prior event of interest | Yes | Strong predictor and prescribing driver |
| Concomitant medication | Yes, if measured before time zero | Proxy for severity |
| Post-treatment lab value | No | Mediator or collider |
| Health-system access | Yes, if available | Causes of both |
| Variable measured after outcome | No | Reverse causation |

A confounder list that includes a post-treatment variable will remove part of the effect and produce
a biased estimate in the direction of the null. A list that omits a strong confounder produces an
estimate in the direction of the prescribing pattern.

### 5.3 Weighting diagnostics that must be reported

| Diagnostic | What it shows | Acceptable range | Consequence if failed |
|---|---|---|---|
| Standardised mean difference | Balance after weighting | Below 0.10, ideally below 0.05 | Residual confounding |
| Weight distribution | Extreme weights | Maximum weight below about 10 | Unstable estimate |
| Effective sample size | Information retained | A substantial fraction of the original | Wide intervals, fragile result |
| Positivity | Overlap of covariate distributions | No structural zeros | Untestable region; trim or restrict |
| Negative control outcome | Residual bias | Null association | Confounding remains |

The diagnostics are not optional. An IPTW analysis that reports a point estimate without the weight
distribution and the effective sample size has hidden the information a reviewer needs to judge
whether the estimate is stable.

### 5.4 The sensitivity analyses that answer the obvious objections

| Objection | Sensitivity analysis | What it shows |
|---|---|---|
| The grace period drove the result | Re-run with 15, 30, and 60 days | Whether the effect depends on the window |
| The outcome algorithm is wrong | Re-run with a stricter algorithm | Direction and size of misclassification |
| Unmeasured confounding explains it | E-value and quantitative bias analysis | Strength needed to explain it away |
| The comparator is inappropriate | Active comparator and negative control | Whether the result is comparator-specific |
| Prevalent users were included | Restrict to new users | Whether selection explains the result |
| Discontinuation was ignored | Per-protocol weighting | Whether adherence changes the estimate |

A package that reports one primary analysis and no sensitivity analyses is not an RWE package; it is
a result. The FDA's guidance expects pre-specified sensitivity analyses, and the reviewers ask for
them.

### 5.5 Data-source pitfalls that recur

- **Coding changes over time.** A new code or a coding guideline can create an apparent trend.
- **Coverage changes.** Enrolment and disenrolment create gaps that look like discontinuation.
- **Carry-over.** A dispensing record can cover a period after disenrolment, creating exposure that
  did not occur.
- **Death reporting lag.** Deaths are reported late, which can misclassify follow-up time.
- **Out-of-network care.** Claims do not capture care paid by another insurer.
- **Lab values.** Present in EHR but not in claims; their absence may proxy for severity.

Each pitfall has a diagnostic, and each diagnostic belongs in the data-source assessment in Step 3.

### 5.6 Precision: how much data is enough

RWE datasets are large, but the number of events is often modest. A study with 100,000 patients and a
1% event rate has 1,000 events, which detects a rate ratio of about 0.85 with 80% power. The same
study with a 0.2% event rate has 200 events, which detects only a ratio of about 0.70.

The arithmetic is the same as for a trial:

```text
Events required (80% power, two-sided alpha 0.05) = 4 x (1.96 + 0.84)^2 / (ln RR)^2
```

For RR 0.85: 4 × 7.84 / 0.0264 = 1,188 events. For RR 0.70: 4 × 7.84 / 0.1272 = 246 events.

A programme that asks an RWE study to detect a 15% relative difference needs more than a thousand
events, and should check the event count before committing. Large databases do not guarantee
precision.

### 5.7 External control arms

An external control arm is a comparator group drawn from outside the trial, often from a registry or
a prior cohort. It is used when randomisation to control is unethical or infeasible, typically in
rare disease or oncology.

The design questions are the same as for any emulation, with one addition: the external control must
be comparable on the trial's eligibility and on the standard of care at the time of the trial. A
registry cohort treated five years earlier, or in a different health system, differs in both.

| Issue | Question | Mitigation |
|---|---|---|
| Eligibility | Does the external cohort meet the trial's criteria? | Apply the criteria; report the attrition |
| Standard of care | Was the comparator treatment the same? | Restrict to the same era; adjust for background therapy |
| Ascertainment | Were outcomes measured the same way? | Validate the algorithm; compare with trial adjudication |
| Time zero | Does the external cohort start at the same point? | Align definitions |
| Confounding | Are the same covariates available? | Restrict to shared covariates; state the gap |

The regulatory status is important: an externally controlled trial is generally an interventional
study and is subject to the rules for investigational drugs, even though the control data are
external. An observational comparison using external data is a different regulatory object.

### 5.8 When RWE cannot answer the question

Four situations defeat the method:

- **The confounders are unmeasured and strong.** If severity drives both treatment and outcome and is
  not in the data, no method recovers the effect.
- **The outcome is not ascertainable.** A symptomatic endpoint recorded only in free text cannot be
  measured reliably at scale.
- **The exposure is not recorded.** Over-the-counter or cash-pay use is invisible.
- **The question is about a rare subgroup.** The data may contain too few patients to estimate
  anything.

Recognising these before the analysis saves the cost of an analysis that cannot be used. The
data-source assessment in Step 3 is where the recognition belongs.

### 5.9 Pre-registration and transparency

The FDA's RWE guidance emphasizes pre-specification and transparency. In practice this means a
protocol with the target trial table, the analysis plan, and the sensitivity analyses, dated and
registered before the data are analysed.

Three transparency practices strengthen a package:

1. **Register the protocol** on a public registry, as the RCT-DUPLICATE initiative did for its 32
   emulations.
2. **Publish the code** or make it available to the regulator, so the analysis can be reproduced.
3. **Report the design choices that were not pre-specified**, with their rationale.

An analysis that was designed after seeing the data can still be valid, but it must be reported as
exploratory, and the distinction is what the reviewer is looking for.

The registry entry should also name the data source version, because a later extract can produce a
different cohort and the protocol must be tied to the data it planned for.

## 6. Worked example

> **Note:** The drug, database, and counts are fictional and constructed for teaching. The
> methodological references and regulatory guidance cited are real and referenced in Section 16.

### 6.1 The question and the target trial

A sponsor wants to claim that a new oral anticoagulant reduces major bleeding compared with warfarin
in patients with atrial fibrillation and moderate kidney impairment. The target trial:

| Component | Specification |
|---|---|
| Eligibility | Adults with atrial fibrillation, eGFR 30-59, no prior anticoagulant in 12 months |
| Treatment | Initiate the new drug versus initiate warfarin |
| Assignment | By clinician choice (not randomised) |
| Time zero | Date of the first prescription after eligibility |
| Outcome | Hospitalisation with major bleeding, 24 months |
| Follow-up | Until outcome, disenrolment, death, or 24 months |
| Estimand | Per-protocol effect of initiating and remaining on treatment |

### 6.2 The crude analysis

The database yields 12,000 new users of the drug (36,000 person-years, 720 events) and 24,000 new
users of warfarin (72,000 person-years, 2,160 events).

```text
Rate, new drug   = 720 / 36,000 x 1,000 = 20.0 per 1,000 person-years
Rate, warfarin   = 2,160 / 72,000 x 1,000 = 30.0 per 1,000 person-years
Crude rate ratio = 20.0 / 30.0 = 0.667
```

The crude analysis suggests a 33% reduction in major bleeding. That is a large effect for an
anticoagulant, and the first response should be scepticism rather than enthusiasm.

### 6.3 The confounding

New users of the new drug are younger (mean 71 versus 78 years), have fewer prior bleeds (2.1% versus
5.4%), and are less likely to have prior falls or dementia. These are exactly the factors a clinician
considers when choosing an anticoagulant, and all of them predict bleeding.

The standardised mean differences before weighting are large: age 0.42, prior bleed 0.17, falls 0.21.
After IPTW, all are below 0.05, and the effective sample size is 9,600 for the new drug and 18,900 for
warfarin, from 12,000 and 24,000.

### 6.4 The adjusted analysis

```text
Weighted rate, new drug  = 26.0 per 1,000 person-years
Weighted rate, warfarin  = 27.0 per 1,000 person-years
Adjusted rate ratio      = 26.0 / 27.0 = 0.963
95% CI                   = 0.884 to 1.049
```

The apparent 33% benefit becomes a 4% difference with a confidence interval that includes no effect.
The confounding accounted for almost the entire association.

### 6.5 The E-value

```text
E-value for the crude RR of 0.667:
  1/RR = 1.50
  E = 1.50 + sqrt(1.50 x 0.50) = 1.50 + 0.866 = 2.37

E-value for the adjusted RR of 0.963:
  1/RR = 1.039
  E = 1.039 + sqrt(1.039 x 0.039) = 1.039 + 0.201 = 1.24
```

The E-value is the minimum strength of association that an unmeasured confounder would need with both
treatment and outcome to explain away the observed association. For the crude result, an unmeasured
confounder with a risk ratio of 2.37 for both would be needed; given the known prescribing patterns,
that is plausible. For the adjusted result, a confounder with a risk ratio of 1.24 would suffice,
which is very plausible, so the adjusted null is fragile in the other direction: a small unmeasured
confounder could hide a real benefit.

### 6.6 Two diagnostics that reveal a broken analysis

**Diagnostic 1 — immortal time.** Check whether follow-up begins before treatment can start. If the
cohort is defined by "any prescription in the year" and follow-up starts at the first eligibility
date, patients who never receive the drug contribute unexposed person-time while waiting, and
patients who die before treatment are misclassified. In this example, starting follow-up at the
eligibility date rather than the prescription date would add 3,400 unexposed person-years to the
warfarin arm and inflate the apparent benefit.

**Diagnostic 2 — balance is not tested.** An adjusted analysis that does not report standardised mean
differences before and after weighting cannot be assessed. In this example, the unadjusted age
difference is 0.42, which is far above the 0.10 threshold usually used for concern. An analysis that
reports only the adjusted estimate hides the fact that the groups were not comparable before
adjustment.

A third check is the negative control outcome. An outcome that should not be affected by the choice of
anticoagulant (for example, appendicitis) should show a null association after adjustment. If it does
not, residual confounding is present.

### 6.7 Vary one assumption: the comparator

The payer asks for a comparison with a direct oral anticoagulant already on the market, rather than
warfarin. The new drug and the comparator have similar prescribing patterns, and the age difference
falls from 7 years to 2 years.

```text
Weighted rate, new drug      = 26.0 per 1,000 person-years
Weighted rate, active comp.  = 24.5 per 1,000 person-years
Adjusted rate ratio          = 1.06 (95% CI 0.97 to 1.16)
```

The conclusion changes from "no difference versus warfarin" to "a possible small increase versus the
active comparator". Neither result supports the original claim of a 33% reduction, and the active
comparator result is the one the payer will use, because it reflects the choice the payer faces.

### 6.8 What the example establishes

1. The crude estimate is a description of prescribing, not of treatment effect.
2. Time zero and the new-user design determine whether immortal time is present.
3. Balance diagnostics are part of the result, not an appendix.
4. The E-value quantifies how fragile a conclusion is in either direction.
5. The comparator defines the decision the evidence informs.

### 6.9 How a regulator would assess this package

The assessment would follow the FDA's RWE considerations: whether the study was pre-specified, whether
the data source was fit for the question, whether confounding was adequately controlled, and whether
the result is robust.

| Question | Evidence in this example | Assessment |
|---|---|---|
| Was the protocol pre-specified? | Target trial table and registered protocol | Adequate if dated before analysis |
| Was the data source fit for purpose? | Coding, coverage, outcome validation | Needs an outcome validation estimate |
| Was time zero aligned? | First dispensing after eligibility | Adequate |
| Was confounding controlled? | IPTW with balance diagnostics | Adequate for measured confounders |
| Is the result robust? | E-value 1.24; negative control | Fragile; residual confounding plausible |
| Is the comparator relevant? | Warfarin versus active comparator | The active comparator is the decision-relevant one |

The package would support a conclusion of "no clinically important difference versus warfarin", not a
claim of a 33% reduction.

### 6.10 What would change the conclusion

The adjusted result would become a credible benefit if a second database reproduced it with the same
design, if the negative control outcome were null, and if the E-value were large enough that plausible
confounding could not explain it. None of those conditions holds in this example.

The result would become a credible harm if the active comparator analysis were replicated and the
confidence interval excluded one. That is the analysis the payer will ask for, and it is the one the
programme should run before making a claim.

### 6.11 The cost of the wrong answer

If the programme submits the crude estimate, the regulator will identify the confounding, and the
review will lose time and credibility. If the programme submits the adjusted estimate, it has a
defensible null result and can choose whether to invest in a trial that could detect a smaller effect.

The second path is cheaper. An RWE analysis that concludes "no meaningful difference" is a successful
analysis, because it prevents an expensive programme built on a false premise. The failure is not a
negative result; it is a positive result produced by prescribing patterns.

The programme's decision should therefore be recorded as a decision about the evidence, not about the
drug: "the available RWE does not support a comparative claim; a trial would be required to detect an
effect smaller than 15%."

That statement is more useful to a portfolio committee than an ambiguous positive result, because it
converts an evidence question into a cost question.

## 7. Code, query, or calculation

The script computes the crude and adjusted rates, the rate ratio, and the E-value. It uses only the
standard library, so the arithmetic can be checked anywhere.

```python
import math

def rate_per_1000(events, person_years):
    return events / person_years * 1000

def e_value(rr):
    """Minimum confounder association with both treatment and outcome
    that would explain away an observed rate ratio."""
    if rr < 1:
        rr = 1 / rr
    return rr + math.sqrt(rr * (rr - 1))

def rr_ci(rr, lo, hi):
    se = (math.log(hi) - math.log(lo)) / (2 * 1.959964)
    return math.exp(math.log(rr) - 1.959964 * se), math.exp(math.log(rr) + 1.959964 * se)

# Crude rates
t_ev, t_py = 720, 36_000
c_ev, c_py = 2_160, 72_000
crude = rate_per_1000(t_ev, t_py) / rate_per_1000(c_ev, c_py)
print(f"crude rates {rate_per_1000(t_ev, t_py):.1f} vs {rate_per_1000(c_ev, c_py):.1f} "
      f"per 1000 PY; RR {crude:.3f}; E-value {e_value(crude):.2f}")

# Adjusted (weighted) rates and CI from the fitted model
t_adj, c_adj = 26.0, 27.0
adj = t_adj / c_adj
lo, hi = 0.884, 1.049
print(f"weighted rates {t_adj:.1f} vs {c_adj:.1f}; RR {adj:.3f} "
      f"({lo:.3f}-{hi:.3f}); E-value {e_value(adj):.2f}")

# Events needed for a given rate ratio at 80% power
def events_needed(rr, alpha=0.05, power=0.80):
    z_a, z_b = 1.959964, 0.841621
    return 4 * (z_a + z_b) ** 2 / math.log(rr) ** 2

for rr in (0.90, 0.85, 0.80, 0.70):
    print(f"RR {rr}: {events_needed(rr):.0f} events")
```

Expected output (Python 3.12, executed):

```text
crude rates 20.0 vs 30.0 per 1000 PY; RR 0.667; E-value 2.37
weighted rates 26.0 vs 27.0; RR 0.963 (0.884-1.049); E-value 1.24
RR 0.9: 2828 events
RR 0.85: 1189 events
RR 0.8: 631 events
RR 0.7: 247 events
```

Reading the output:

- The crude rate ratio of 0.667 is associated with an E-value of 2.37, meaning a confounder with a
  risk ratio of 2.37 for both treatment and outcome could explain it away. Given the age and prior
  bleeding differences, that is plausible.
- The adjusted rate ratio of 0.963 has an E-value of 1.24, so even a weak unmeasured confounder
  could explain it. A null result with a small E-value is also fragile, in the opposite direction.
- The events-needed table shows why an RWE study aiming to detect a 10% relative difference needs
  about 2,800 events. Many claims analyses have far fewer, which is why they can only exclude large
  effects.

The E-value is computed on the ratio scale and assumes a risk ratio interpretation. For a rate ratio
from person-time data, the approximation is reasonable when the outcome is not very common; for a
common outcome it overstates the required confounding. The code states the assumption rather than
hiding it.

A production analysis would replace the hard-coded weighted rates with estimates from a fitted
propensity model. The structure of the calculation, and the reporting of crude and adjusted side by
side, would remain the same.

A count query illustrates the feasibility step that precedes any analysis. The number of patients who
meet the computable eligibility criteria is an upper bound on the cohort, not the cohort:

```sql
SELECT COUNT(DISTINCT patient_id) AS eligible
FROM diagnosis d
WHERE d.code BETWEEN 'I48.0' AND 'I48.91'
  AND d.service_date BETWEEN '2022-01-01' AND '2024-12-31'
  AND NOT EXISTS (
      SELECT 1 FROM dispensing x
      WHERE x.patient_id = d.patient_id
        AND x.drug_class IN ('DOAC','WARFARIN')
        AND x.service_date BETWEEN d.service_date - 365 AND d.service_date);
```

The query uses ICD-10-CM codes for atrial fibrillation and a 12-month washout. It answers "how many
patients could be in the cohort", not "what is the effect". The distinction matters because
feasibility counts are often presented as if they were the analysis, and the cohort shrinks
substantially once exposure, outcome, and covariate completeness are applied.

## 8. Data, standards, and regulatory context

### 8.1 The FDA guidance landscape

| Document | Status | What it covers |
|---|---|---|
| Considerations for the Use of Real-World Data and Real-World Evidence to Support Regulatory Decision-Making for Drug and Biological Products | Final, August 2023 | General framework: data relevance and reliability, study design, regulatory engagement |
| Real-World Evidence: Considerations Regarding Non-Interventional Studies | Draft, March 2024 | Design and analysis of observational studies submitted as evidence |
| Considerations for the Design and Conduct of Externally Controlled Trials | Draft, February 2023 | Externally controlled trials, including historical controls |
| Data Standards for Submissions Containing Real-World Data | Guidance, 2023 | Data standards for RWD submissions |
| Assessing Registries to Support Regulatory Decision-Making | Guidance, 2023 | Registry fitness |
| Assessing Electronic Health Record and Medical Claims Data | Guidance series | Data source fitness |

The status distinction matters. Only the general framework is final. A programme that cites a draft
guidance as a requirement is overstating it; a programme that ignores a draft because it is not final
is understating the review risk.

### 8.2 What the guidance expects

Across the documents, the recurring expectations are:

- **Fit-for-purpose data.** The data must be relevant to the question and reliable for the analysis,
  with the limitations stated.
- **Pre-specified design.** A protocol and analysis plan before the analysis, with the design choices
  justified.
- **Confounding control.** Adjustment for measured confounders, with diagnostics, and an assessment of
  unmeasured confounding.
- **Transparency.** Registration, data sharing with the regulator, and reporting of design choices.
- **Early engagement.** Pre-submission or Type V DMF discussions for third-party data.

### 8.3 What is a requirement and what is a convention

- **Requirement.** The statutory standard for substantial evidence of effectiveness is adequate and
  well-controlled investigations under 21 CFR 314.126. An RWE study can contribute to that standard,
  but the statute does not name RWE as sufficient on its own.
- **Regulatory expectation.** Pre-specification, confounding control, and transparency are expressed
  in guidance. They are not statutory language, but a submission that lacks them will not be
  accepted as substantial evidence.
- **Convention.** The choice of IPTW versus matching, the E-value threshold for concern, and the
  specific grace period are analytical conventions. They should be justified, not treated as rules.

### 8.4 The EU context

The EMA has invested in a coordinated RWE infrastructure through DARWIN EU, a network that provides
access to European healthcare databases for regulatory decision-making. The EMA's framework
emphasizes data quality, methodological standards, and transparency, and it is developing guidance on
the use of RWE in regulatory decisions.

A global programme should not assume that a US-designed emulation will satisfy an EU assessment. The
data sources, coding systems, and standard of care differ, and the EU may require a European data
source or a multi-database replication.

### 8.5 What the guidance does not provide

No guidance specifies an acceptable E-value, a minimum balance threshold, or a list of required
sensitivity analyses. Those are scientific judgements. The guidance describes the process by which
the judgements should be made and documented, not the thresholds at which a result becomes credible.

### 8.6 Data standards for RWD submissions

The FDA's data standards guidance for submissions containing RWD describes how to structure and
document the data. Two requirements are operationally significant:

- **Traceability to the source.** The submission must allow a reviewer to trace an analytic variable
  to the source data and the transformation that produced it.
- **Documentation of data quality.** The submission must describe the data source, the coding, the
  validation, and the limitations.

A programme that treats RWD as a dataset to be converted at the end, without documenting the source
and the transformations, will fail these requirements. The documentation is built during the
analysis, not after it.

### 8.7 Data quality checks specific to RWD

| Check | What it detects | How to run it |
|---|---|---|
| Enrolment continuity | Gaps in coverage that look like discontinuation | Plot enrolment by month for the cohort |
| Coding consistency | Code changes over time | Count codes by year |
| Outcome validation | Misclassification | Compare with records or a validated subset |
| Death ascertainment | Incomplete mortality | Compare with a death registry |
| Exposure capture | Out-of-network or cash-pay use | Compare dispensing with prescribing |
| Duplicate records | Same patient, multiple identifiers | Probabilistic linkage audit |

### 8.8 The difference between regulatory and payer evidence

| Dimension | Regulatory | Payer |
|---|---|---|
| Question | Efficacy and safety in the label population | Comparative value in the covered population |
| Comparator | Usually placebo or standard of care | The treatment the payer would otherwise fund |
| Outcome | Endpoints accepted for the claim | Cost per outcome, budget impact |
| Evidence | Trials plus RWE | RWE, economic models, real-world cost |
| Uncertainty | Confidence intervals and sensitivity | Scenario and budget impact analysis |

A single RWE analysis rarely serves both. The regulatory analysis asks whether the drug works; the
payer analysis asks whether it is worth the price compared with the alternative. The comparator and
the outcome differ, and a programme that reuses the regulatory analysis for the payer will be asked
for a different one.

## 9. Industry practice and case evidence

### 9.1 What RCT-DUPLICATE showed

The RCT-DUPLICATE initiative emulated 32 randomised trials using claims data, with protocols
registered before analysis. The results, published in JAMA in 2023, are the best public evidence on
when emulation works.

| Measure | Overall (32 trials) | Closely emulated (16) | Poorly emulated (16) |
|---|---:|---:|---:|
| Correlation of effect estimates | 0.82 | 0.93 | 0.53 |
| Agreement on statistical significance | 75% | 94% | 50% |
| Agreement on estimate | 66% | 88% | 50% |
| Agreement on standardised difference | 75% | 88% | 69% |

The pattern is that emulation reproduces the trial when the design and measurement can be closely
matched, and fails when they cannot. The lesson is not "RWE works" or "RWE does not work"; it is that
the quality of the emulation determines the answer, and the quality is assessable before the analysis.

A 2026 systematic review of 106 target trial emulation and randomised trial pairs reported a lower
overall correlation of 0.58, with closer emulations reaching 0.83, and found that claims-based
emulations tended to underestimate effects (ratio of ratios 0.89). The difference between the two
estimates is itself informative: the result depends on the set of trials selected and the quality of
the emulation.

### 9.2 How the industry uses RWE

| Use | Evidence standard | Typical data source |
|---|---|---|
| Label expansion | Substantial evidence, often with a trial | Claims plus EHR, or registry |
| Post-authorisation safety | Defined population, measured exposure | Claims, EHR, registry |
| Payer value demonstration | Decision-analytic model plus RWE | Claims, registries |
| Trial feasibility | Counts and eligibility | Claims, EHR |
| External control | Comparability with the trial | Registry, prior cohort |
| Clinical guideline input | Systematic review plus RWE | Multiple |

The evidentiary bar falls as the decision moves from regulatory to payer to internal. A programme
that uses the same analysis for all four will over-claim at the regulatory end and under-invest at
the payer end.

### 9.3 Case: external controls in rare disease

External control arms are used most often in rare disease, where randomisation to control may be
unethical or infeasible. Regulators have accepted them in specific circumstances, typically with a
natural history cohort, a pre-specified comparability analysis, and a sensitivity analysis against
alternative external controls.

**Documented:** the regulatory pathway and the guidance on externally controlled trials.
**Inference:** that the acceptance depends more on the comparability of the external cohort than on
the statistical method. A well-matched external control from the same era and the same centres is
more convincing than a large but distant cohort.

### 9.5 Reproducibility as the new credibility test

A reviewer can no longer verify an RWE analysis by reading the manuscript. The data are too large, the
code too complex, and the design choices too many. Reproducibility has therefore become the practical
test: can an independent analyst, given the protocol, the code, and the data access, obtain the same
result?

The RCT-DUPLICATE initiative demonstrated the standard by registering its protocols before analysis.
Its protocols, code lists, and results are public, which is why the discrepancy between closely and
poorly emulated trials could be studied at all.
A programme that cannot provide the code and the extract version is asking the reviewer to trust the
result rather than verify it, and trust is not an evidentiary standard.

The practical implication is that the analysis code, the data extract date, and the protocol should be
treated as submission artefacts, not as internal working files. A reviewer who cannot rerun the code
cannot distinguish a pre-specified analysis from one designed after the result.

A good RWE report therefore opens with the target trial table and the data-source assessment, presents
the crude and adjusted estimates together, reports the balance diagnostics and the negative control,
and closes with the E-value and the conditions under which the conclusion would change.

### 9.4 What is documented and what is inferred

| Claim | Status |
|---|---|
| Emulation reproduces trials when design and measurement match closely | Documented in RCT-DUPLICATE |
| Claims-based emulations tend to underestimate effects | Documented in the 2026 systematic review |
| External controls are accepted in some rare disease approvals | Documented in guidance and approvals |
| Pre-specification improves regulatory acceptance | Inference from guidance emphasis; not measured |
| Most RWE label expansions fail | Not established publicly; approval counts exist but the denominator does not |
| E-values are routinely reported | Inference; the practice is growing but not universal |

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Immortal time | Follow-up starts before treatment | Exposure defined after time zero | Spurious benefit | Align time zero to treatment decision |
| Prevalent users included | Cohort includes patients already on treatment | Selection for survival | Spurious benefit | New-user design with washout |
| Healthy-user bias | Treated patients younger and healthier | Prescribing pattern | Spurious benefit | Adjust for health status; active comparator |
| Confounder measured after treatment | Lab value at follow-up in the model | Mediator adjustment | Understated effect | Restrict to baseline covariates |
| Collider adjustment | Adjusting for a post-treatment event | Collider stratification | Bias in either direction | Causal diagram |
| Outcome misclassification | Algorithm not validated | Coding error | Attenuated or biased estimate | Validation study |
| Comparator irrelevant | All-comers comparator | Different decision | Wrong answer for the decision | Active comparator |
| No balance diagnostics | Adjusted estimate only | Hidden residual confounding | Unassessable result | Report SMDs and weights |
| Extreme weights | Max weight very large | Positivity violation | Unstable estimate | Trim, weight truncation, overlap weights |
| Multiple testing of subgroups | Many subgroup analyses | Data mining | False positive | Pre-specify subgroups |
| Data cut not stated | Extract date absent | Unreproducible | Unverifiable | Version the extract |
| Negative control failed but ignored | Control outcome shows association | Residual confounding | Invalid conclusion | Report and act on the control |

The most damaging red flag is a large crude effect that disappears after adjustment, presented without
the crude estimate. The crude estimate is the evidence that confounding was present; hiding it hides
the reason for scepticism.

### 10.1 A diagnostic sequence for any RWE result

1. **Draw the timeline.** Plot eligibility, time zero, first exposure, and outcome for a sample of
   patients. If exposure precedes time zero for some, the design is broken.
2. **Compute the crude estimate.** It shows the size and direction of confounding.
3. **Check balance.** Standardised mean differences before and after adjustment.
4. **Run a negative control.** An outcome that should not be affected.
5. **Compute the E-value.** How fragile is the result?
6. **Change the comparator.** Does the conclusion survive?

The sequence takes a day and catches most of the errors that would otherwise be found in review.

### 10.2 Red flags by design stage

| Stage | Red flag | What it usually means |
|---|---|---|
| Question | No comparator named | The analysis cannot define a contrast |
| Data source | Outcome validation absent | The outcome may not measure what it claims |
| Eligibility | Criteria applied after time zero | Selection bias |
| Exposure | Prevalent users included | Survival selection |
| Time zero | Follow-up starts at diagnosis or first code | Immortal time |
| Confounding | Covariates measured after treatment | Mediator adjustment |
| Analysis | No balance diagnostics | Residual confounding unassessed |
| Sensitivity | No negative control | Residual bias unexamined |
| Reporting | Crude estimate absent | Confounding hidden |

A package that fails at the question stage cannot be rescued by the analysis; a package that fails at
the reporting stage can be fixed in a day.

The staging also determines the cost of correction and the credibility of the fix. A question-stage error requires a new protocol; a
data-source error requires a new database; an analysis-stage error requires a re-run; a reporting
error requires an edit. Reviewing in this order is cheaper than reviewing the result first.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| No unmeasured confounding | Causal interpretation | Severity not captured | E-value, negative control |
| Positivity | Weighting and matching | Structural zeros | Check covariate overlap |
| Consistency | Treatment definition | Multiple versions of the treatment | Define the exposure precisely |
| Correct outcome ascertainment | Outcome measurement | Coding error | Validation study |
| Correct model specification | Regression or propensity model | Non-linearity, interactions | Alternative specifications |
| Time zero aligned | Follow-up validity | Eligibility before treatment | Design audit |
| No informative censoring | Follow-up completeness | Disenrolment related to outcome | Sensitivity analysis |
| Stable standard of care | Comparability over time | Practice change | Restrict to an era |
| Data coverage is complete | Confounder and outcome capture | Out-of-network care | Coverage assessment |
| The target trial is the right one | Relevance of the answer | Wrong comparator or population | Clinical input || The data extract is reproducible | Re-analysis | Extract not versioned | Date and version in the report |
| The code is available | Verification | Proprietary black box | Code sharing or regulator access || The target trial table is documented | Design audit | Missing | Include in the protocol |
| The negative control result is reported | Residual bias | Not run | Pre-specify a control outcome |

The deepest limit is that the target trial is a construct. The analysis can align to it, but the
alignment is only as good as the data. A perfect emulation of the wrong trial is still the wrong
answer.

A second limit is that the analysis is observational regardless of how well it is designed. The
estimate is conditional on the assumption of no unmeasured confounding, and that assumption is not
testable. The E-value and the negative control are the best available checks, and they bound the
fragility rather than proving the assumption.

A third limit is the data's purpose. Claims data are collected for payment, EHR data for care, and
registries for reporting. None was designed for research, and each carries a structure that reflects
its purpose. The analyst who forgets this will mistake a billing rule for a clinical fact.

A fourth limit is that the analysis answers only the question it was designed for. The target trial
table is therefore also the boundary statement for what the evidence supports, and a claim that goes
beyond it requires a different study. A null result in a
population with good adherence does not rule out benefit in a population with poor adherence, and a
result in one health system may not transfer to another.

## 12. Exercises


Each exercise uses a hypothetical database; where a real extract is unavailable, the worked example's
structure can be reused.

**Exercise 1 — Target trial table (design, 35 minutes).**
Deliverable: a seven-component target trial table for a comparative safety question of your choice.
Acceptance criteria: each component implementable; time zero explicit; estimand named.
Time: 35 minutes.

**Exercise 2 — Crude and adjusted rates (application, 30 minutes).**
Deliverable: crude rates, rate ratio, and an interpretation of the direction of confounding for a
supplied two-cohort table.
Acceptance criteria: person-time denominators used; direction of bias explained; no causal claim
from the crude estimate.
Time: 30 minutes.

**Exercise 3 — Confounder list (application, 30 minutes).**
Deliverable: a confounder list with a causal diagram for an anticoagulant comparison, marking
confounders, mediators, and colliders.
Acceptance criteria: at least eight variables classified; post-treatment variables excluded;
rationale for each inclusion.
Time: 30 minutes.

**Exercise 4 — E-value (application, 20 minutes).**
Deliverable: E-values for rate ratios of 0.70, 0.85, and 0.95, with an interpretation of fragility.
Acceptance criteria: formula shown; values correct to two decimals; interpretation in plain language.
Time: 20 minutes.

**Exercise 5 — Immortal time audit (critique, 40 minutes).**
Deliverable: an audit of a supplied cohort definition that starts follow-up at diagnosis and defines
exposure by any prescription in the following year, quantifying the immortal period.
Acceptance criteria: the mechanism explained; the direction of bias stated; a corrected design
proposed.
Time: 40 minutes.

**Exercise 6 — Sensitivity plan (design, 35 minutes).**
Deliverable: a sensitivity analysis plan with at least five analyses and what each would show.
Acceptance criteria: each analysis tied to a specific objection; pre-specified; interpretation
stated for each outcome.
Time: 35 minutes.

**Exercise 7 — Regulatory assessment (critique, 45 minutes).**
Deliverable: a 500-word assessment of an RWE package against the FDA's expectations, with a
recommendation.
Acceptance criteria: covers pre-specification, data fitness, confounding, and robustness; identifies
the weakest element; states what would strengthen it.
Time: 45 minutes.

**Exercise 8 — Emulation versus trial (application, 40 minutes).**
Deliverable: a comparison of an emulation result with a published trial result, explaining the
difference using the RCT-DUPLICATE framework.
Acceptance criteria: design differences identified; direction of the expected difference stated;
conclusion about emulation quality.
Time: 40 minutes.

## 13. Self-check answers

**Exercise 1.** A complete table names the population with computable criteria, the two treatment
strategies with a grace period, the assignment mechanism (clinician choice), time zero (first
dispensing after eligibility), the outcome algorithm, the follow-up rule, and the estimand
(per-protocol or ITT). A table that omits time zero or the estimand cannot be implemented.

**Exercise 2.** For 400 events in 20,000 person-years (20 per 1,000) and 900 events in 30,000
person-years (30 per 1,000), the crude rate ratio is 0.667. If treated patients are younger and
healthier, the crude estimate is biased away from the null, and adjustment will move it toward one.

**Exercise 3.** Confounders: age, sex, comorbidity, prior bleeding, renal function, concomitant
antiplatelet use. Mediator: time in therapeutic range on treatment (do not adjust). Collider: a
post-treatment hospitalisation that affects both subsequent treatment and outcome (do not adjust).
The diagram should show arrows from each confounder to both treatment and outcome.

**Exercise 4.** E-value for 0.70 = 1.43 + sqrt(1.43 × 0.43) = 2.22. For 0.85 = 1.18 + sqrt(1.18 ×
0.18) = 1.64. For 0.95 = 1.05 + sqrt(1.05 × 0.05) = 1.28. The closer the estimate to the null, the
smaller the confounding needed to explain it, so a null result is fragile in the direction of a
hidden benefit.

**Exercise 5.** Starting follow-up at diagnosis and defining exposure by a prescription in the next
year creates a period between diagnosis and prescription in which the patient must survive to be
classified as exposed. Patients who die in that period are classified as unexposed, which biases the
estimate in favour of the drug. The correction is to start follow-up at the prescription date and
require eligibility on that date.

**Exercise 6.** Analyses: alternative grace periods (15, 30, 60 days); active comparator; negative
control outcome; E-value; restriction to patients with a validated outcome; per-protocol weighting for
discontinuation. Each addresses a specific objection and has a pre-specified interpretation.

**Exercise 7 rubric.** Full marks require an assessment of pre-specification, data fitness,
confounding control with diagnostics, robustness, and comparator relevance, with the weakest element
named and a specific strengthening step. Deduct marks for a verdict without a specific gap.

**Exercise 8.** Differences in design (eligibility, time zero, exposure definition, outcome
ascertainment) explain the divergence.**Exercise 8.** The first step is to compare the emulation's design table with the trial's protocol,
component by component: eligibility, time zero, exposure, outcome, and follow-up.

Differences in design (eligibility, time zero, exposure definition, outcome
ascertainment) explain the divergence. In RCT-DUPLICATE, closely emulated trials reached a correlation
of 0.93 and poorly emulated ones 0.53. The conclusion should state which design differences are
present and whether the emulation is close or poor.

## 14. Interview questions

**1. What is a target trial, and why specify one?**
Strong: a framework that defines the randomised trial being emulated, aligning design choices and
making them auditable. Weak: says it is a protocol.

**2. Why is time zero the most important design choice?**
Strong: it determines whether immortal time and selection are present; must align with the treatment
decision. Weak: says it is the start date.

**3. What is immortal time bias?**
Strong: a period in which the outcome cannot occur is misclassified as exposed, producing a spurious
benefit. Gives an example. Weak: says it is "a bias in follow-up".

**4. How would you control confounding?**
Strong: identify confounders from a causal diagram, measure before time zero, use IPTW or matching,
report balance and effective sample size, and assess unmeasured confounding. Weak: adjusts for
everything available.

**5. What does an E-value tell you?**
Strong: the minimum strength of association an unmeasured confounder would need with both treatment
and outcome to explain away the result; a fragility measure, not a probability. Weak: calls it a
p-value.

**6. When would you use an external control arm?**
Strong: when randomisation is unethical or infeasible, with comparability on eligibility, era, and
ascertainment, and preferably multiple external controls. Weak: whenever a trial is expensive.

**7. Why do claims-based emulations underestimate effects?**
Strong: misclassification of exposure and outcome, incomplete confounder capture, and inclusion of
patients who would not be in the trial. Weak: says the data are noisy.

**8. How would you assess whether an RWE package is credible?**
Strong: target trial alignment, data fitness, pre-specification, confounding diagnostics, negative
controls, and robustness. Weak: looks at the p-value.

**9. What is the difference between a rate ratio and a risk ratio?**
Strong: rate uses person-time; risk uses persons at risk over a period; they diverge when the outcome
is common or follow-up differs. Weak: treats them as interchangeable.

**10. What would make you abandon an RWE analysis?**
Strong: unmeasured confounding that cannot be addressed, an outcome that cannot be ascertained, or a
comparator that does not answer the decision. Weak: a non-significant result.**10. What would make you abandon an RWE analysis?**
Strong: unmeasured confounding that cannot be addressed, an outcome that cannot be ascertained, or a
comparator that does not answer the decision. Weak: a non-significant result.

**11. How do you know an RWE analysis is reproducible?**
Strong: the protocol is registered, the extract has a version and date, the code is available, and an
independent analyst can obtain the same numbers. Weak: the report describes the method in prose.**11. How do you know an RWE analysis is reproducible?**
Strong: the protocol is registered, the extract has a version and date, the code is available, and an
independent analyst can obtain the same numbers. Weak: the report describes the method in prose.

**12. What would you do if the crude and adjusted estimates disagree?**
Strong: treat the difference as evidence about confounding, report both, diagnose the source, and
assess whether the adjustment is adequate. Never report only the adjusted estimate. Weak: reports the
adjusted estimate because it is "the real one".

## 15. Key terms

- **Active comparator** — a comparator treatment used in the same indication.
- **Channeling bias** — patients with a specific prognosis are channeled to a treatment.
- **Collider** — a variable caused by both treatment and outcome.
- **Confounder** — a cause of both treatment and outcome.
- **Effective sample size** — sample size after weighting.
- **E-value** — minimum unmeasured confounding needed to explain away an association.
- **External control arm** — a comparator group drawn from outside the trial.
- **Grace period** — the window after time zero in which treatment must start.
- **Healthy-user bias** — healthier patients are more likely to receive a new treatment.
- **Immortal time** — a period in which the outcome cannot occur, misclassified as exposed.
- **Inverse probability of treatment weighting (IPTW)** — weighting to balance covariates.
- **Mediator** — a variable on the causal path from treatment to outcome.
- **Negative control outcome** — an outcome that should not be affected by treatment.
- **New-user design** — cohort entry at treatment initiation.
- **Positivity** — overlap of covariate distributions across groups.
- **Prevalent user** — a patient already on treatment at cohort entry.
- **Propensity score** — probability of treatment given covariates.
- **Rate ratio** — ratio of incidence rates per person-time.
- **Real-world data (RWD)** — data collected during routine care.
- **Real-world evidence (RWE)** — evidence derived from RWD analysis.
- **Standardised mean difference** — balance metric between groups.
- **Target trial** — the randomised trial an observational analysis emulates.
- **Time zero** — the start of follow-up, aligned to the treatment decision.

## 16. Sources

- US Food and Drug Administration. *Considerations for the Use of Real-World Data and Real-World
  Evidence to Support Regulatory Decision-Making for Drug and Biological Products*. Final guidance,
  August 2023. https://www.fda.gov/media/171667/download (accessed 25 October 2026).
- US Food and Drug Administration. *Real-World Evidence: Considerations Regarding Non-Interventional
  Studies for Drug and Biological Products*. Draft guidance, March 2024.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/real-world-evidence-considerations-regarding-non-interventional-studies-drug-and-biological-products
  (accessed 25 October 2026).
- US Food and Drug Administration. *Considerations for the Design and Conduct of Externally
  Controlled Trials for Drug and Biological Products*. Draft guidance, February 2023.
  https://www.fda.gov/media/164960/download (accessed 25 October 2026).
- Wang SV, Schneeweiss S, RCT-DUPLICATE Initiative. Emulation of randomized clinical trials with
  nonrandomized database analyses: results of 32 clinical trials. *JAMA*. 2023;329(16):1376-1385.
  https://doi.org/10.1001/jama.2023.4221 (accessed 25 October 2026).
- Heyard R, Held L, Schneeweiss S, Wang SV. Design differences and variation in results between
  randomised trials and non-randomised emulations: meta-analysis of RCT-DUPLICATE data. *BMJ
  Medicine*. 2024;3:e000709. https://doi.org/10.1136/bmjmed-2023-000709 (accessed 25 October 2026).
- Wang CH, et al. Concordance between target trial emulation and randomised controlled trials:
  systematic review and meta-analysis. *BMJ*. 2026. https://doi.org/10.1136/bmj-2025-086810
  (accessed 25 October 2026).
- European Medicines Agency. *DARWIN EU*.
  https://www.ema.europa.eu/en/about-us/how-we-work/big-data/darwin-eu (accessed 25 October 2026).
- US Government. *21 CFR 314.126, Adequate and Well-Controlled Studies*.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-D/part-314/subpart-D (accessed 25 October 2026).
- Hernán MA, Robins JM. *Causal Inference: What If*. Chapman and Hall/CRC, 2020.
  https://miguelhernan.org/whatifbook (accessed 25 October 2026).

> **Note:** The drug, database, and counts in Section 6 are fictional. The RCT-DUPLICATE results and
> the regulatory documents cited are real; the interpretation of the example is the course's own.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`06_safety-and-pharmacovigilance.md`](06_safety-and-pharmacovigilance.md)
- Next topic: [`08_ai-in-discovery-and-development.md`](08_ai-in-discovery-and-development.md)
- Related: [`04_feasibility-and-recruitment.md`](04_feasibility-and-recruitment.md)
- Related: [`10_submission-data.md`](10_submission-data.md)
- Survey file: [`../../13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md)
- Labs: [`../../labs/`](../../labs/)
