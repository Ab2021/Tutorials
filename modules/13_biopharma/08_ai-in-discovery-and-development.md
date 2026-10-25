# AI in Discovery and Development: Where Models Are Used, Validated, and Rejected

> Module 13 — Biopharma and Clinical Development
> Topic 08 of 10
> Estimated study time: 5 hours
> Prerequisites: Module 06 (machine learning); topics 01, 03, and 07
> Version 2026.3

## 1. Why this topic exists

Artificial intelligence (AI) in drug development is unusual among data science applications because
the output is not a prediction that a person can check. A model that ranks compounds, segments a
pathology image, or forecasts a clinical outcome produces a number that enters a decision with
regulatory consequences, and the people receiving the number usually cannot inspect the reasoning
that produced it. That asymmetry is the central governance problem of the field.

The industry has two failure modes, and they are opposites. The first is overclaiming: a model
reports an impressive internal validation metric, the programme treats it as evidence of biological
or clinical validity, and the model fails when it meets a new site, a new population, or a new batch.
The second is underuse: a model that could reduce a search space or prioritise site monitoring is
never adopted because the organisation cannot validate it to its own satisfaction and has no process
for deciding what "good enough" means.

Both failures come from the same missing step: a specified context of use with a risk-proportionate
validation plan. The FDA's 2025 draft guidance on AI for regulatory decision-making makes this
explicit through a seven-step credibility assessment framework. The EMA's 2024 reflection paper makes
the same point across the product lifecycle. A model is not validated in general; it is validated for
a question, a population, and a decision consequence.

This topic covers where AI is actually used across discovery and development, how to define a context
of use, how to assess model risk, how to validate internally and externally, how to detect drift, and
how to document the evidence so a regulator or an internal governance board can assess it. It follows
the RWE topic, because many AI claims rest on real-world data, and precedes the decentralised trials
and submission topics, both of which involve model-generated data.

### 1.1 The failure pattern

1. A sponsor develops a deep learning model that scores pathology slides for a predictive biomarker.
2. Internal validation on 1,200 slides gives an area under the receiver operating characteristic curve
   (AUC) of 0.88.
3. The model is used to select patients for a Phase II trial, and the trial reports a promising
   response rate.
4. A larger external validation on 600 slides from other hospitals gives an AUC of 0.74.
5. At the locked threshold, sensitivity is 0.82 and specificity 0.78, so at 15% prevalence the
   positive predictive value is 40%: 60% of selected patients do not have the biomarker state.
6. The trial's effect is diluted, and the programme cannot tell whether the drug or the model failed.
7. The model is retired, and the programme returns to conventional testing.

Step 4 is not a surprise; it is the expected result of validating on data from the same institutions
that produced the training set. The failure is that the programme did not plan for it, and therefore
did not collect the external data until after the trial had started.

### 1.2 Who is blocked when AI validation is weak

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Data scientist | Adoption of a working model | Cannot define the decision consequence |
| Translational lead | A biomarker programme | Depends on assay and model performance |
| Clinical team | A trial population that reflects the hypothesis | Receives a score without its error rate |
| Regulatory affairs | A defensible credibility package | Receives a model without a context of use |
| Quality function | Confidence in model lifecycle control | Depends on engineering practices |
| Patient | Exposure to a misclassified decision | Has no visibility of the model |

### 1.3 The three claims that are routinely conflated

| Claim | What it asserts | Evidence required |
|---|---|---|
| The model works | It performs on a defined dataset | Validation results |
| The model is useful | Using it improves a decision | Prospective or comparative study |
| The model is approved | A regulator has accepted it for a use | Regulatory decision |

Most AI communication conflates the first two. A model that works on a dataset may not improve a
decision, because the decision depends on the workflow, the human response, and the cost of errors.
The third claim is rare and should be stated only with a specific regulatory decision.

### 1.4 Why the regulatory consequence matters more than the model quality

Two models with the same AUC can have different regulatory consequences. A model that ranks compounds
for synthesis is a research tool; a model that selects patients is part of the evidence for a
marketing application. The second must be documented, validated, and monitored to a standard that the
first does not need.

The consequence determines the evidence burden, and the evidence burden determines the cost. A
programme that applies the same validation standard to every model will over-invest in low-consequence
tools and under-invest in the ones that matter.

### 1.5 The industry question

> For this decision, at this consequence, is this model credible enough, and how would we know if it
> stopped being credible?

The question has three parts: the decision, the consequence, and the monitoring. A model that is
credible for triaging compounds is not credible for selecting patients for a pivotal trial, and a
model that was credible at launch may not be credible after a year of drift.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** AI use cases across discovery, preclinical, clinical, and post-marketing to their decision
   consequence.
2. **Write** a context-of-use statement that specifies the question, population, output, and decision.
3. **Assess** model risk as the product of model influence and decision consequence, and choose a
   proportionate validation plan.
4. **Design** data splits that prevent leakage between training, tuning, and test sets.
5. **Evaluate** a model at the decision threshold, reporting discrimination, calibration, predictive
   values, and subgroup performance.
6. **Detect** drift and plan lifecycle monitoring, including thresholds for retraining or withdrawal.
7. **Document** a credibility assessment that a regulator or governance board can review.

## 3. The core mental model

### 3.1 Where models are used

```text
 Discovery            Preclinical          Clinical                 Post-marketing
 ---------            -----------          --------                 --------------
 Target identification  ADMET prediction    Patient selection        Safety signal detection
 Molecule generation    Toxicity prediction   Digital endpoints        RWE confounding control
 Virtual screening      Biomarker discovery   Trial feasibility        Commercial forecasting
 Property prediction    Assay optimisation    Site selection           Label optimisation
        |                    |                     |                        |
        v                    v                     v                        v
 Low regulatory        Low-moderate          High regulatory          Moderate regulatory
 consequence           consequence           consequence              consequence
```

The regulatory consequence rises as the model moves toward the patient. A model that ranks compounds
can be wrong cheaply; a model that decides who receives a drug cannot.

### 3.2 The credibility framework

```text
 1. Question of interest
        |
 2. Context of use (population, output, decision, timing)
        |
 3. Model risk = model influence x decision consequence
        |
 4. Credibility assessment plan (data, development, evaluation, documentation)
        |
 5. Execute the plan
        |
 6. Credibility assessment report
        |
 7. Adequacy determination for the context of use
        |
        +--> life cycle monitoring and change control
```

The framework is a loop, not a gate. The adequacy determination feeds monitoring, and a change in the
model, the data, or the population returns the process to step 3.

### 3.3 Model risk matrix

| Model influence | Low consequence | Medium consequence | High consequence |
|---|---|---|---|
| Low (advisory, human decides) | Low | Low-medium | Medium |
| Medium (ranks or prioritises) | Low-medium | Medium | High |
| High (determines eligibility or dose) | Medium | High | High |

The matrix determines the evidence. A high-risk model needs external validation, subgroup performance,
drift monitoring, and a documented change-control process. A low-risk model may need only internal
validation and a periodic review.

### 3.4 The validation ladder

| Rung | Question | Evidence | Typical failure |
|---|---|---|---|
| Analytical | Does the model compute what it claims? | Code review, unit tests, reproducibility | Silent preprocessing differences |
| Internal | Does it perform on held-out data from the same source? | Cross-validation or a held-out set | Leakage between train and test |
| External | Does it perform on data from other sites or eras? | Independent cohort | Spectrum and site shift |
| Prospective | Does it perform when used in the workflow? | Silent or shadow mode | Workflow and input differences |
| Impact | Does using it improve the decision? | Randomised or quasi-experimental comparison | No benefit over the incumbent |

Most programmes stop at rung two and claim rung four. The gap is where the failure pattern lives.

### 3.5 Why external performance drops

| Cause | Mechanism | Diagnostic |
|---|---|---|
| Site shift | Staining, scanners, and protocols differ | Performance by site |
| Spectrum bias | Case mix differs (stage, severity) | Performance by subgroup |
| Label leakage | Preprocessing used information from the test set | Audit the pipeline |
| Prevalence shift | PPV changes even when sensitivity and specificity do not | Recompute at local prevalence |
| Temporal drift | Practice or population changed | Performance by year |
| Upstream changes | Coding, assay, or EHR vendor changed | Input distribution check |

The first two are the most common in imaging and biomarker models; the fourth is the most often
forgotten, because sensitivity and specificity look stable while the clinical meaning of a positive
changes.

### 3.6 What the model does not see

A model sees the data it was trained on and the inputs it is given. It does not see the reason a
patient was tested, the clinician's suspicion, the hospital's protocol, or the fact that a variable
was measured only because another was abnormal.

Those omissions are not neutral. A variable measured because of suspicion is a proxy for suspicion,
and a model that learns it will reproduce the clinician's decision rather than predict the biology.
The result is a model that appears accurate because it predicts the decision, not the outcome.

The diagnostic is to ask, for each input, why it was measured. An input that exists only because of
the outcome is a leak, and the model will fail when the measurement practice changes.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Artificial intelligence (AI) | Systems that perform tasks associated with human intelligence | Used as a synonym for machine learning | Umbrella term |
| Machine learning (ML) | Models that learn patterns from data | Used for rule-based systems | Method |
| Context of use (COU) | The specified question, population, output, and decision a model supports | Omitted or written vaguely | Statement |
| Model influence | The degree to which the model affects the decision | Confused with model accuracy | Low, medium, high |
| Decision consequence | The impact of a wrong decision | Confused with model risk | Low, medium, high |
| Model risk | Influence multiplied by consequence | Treated as accuracy | Category |
| Credibility | Evidence that a model is trustworthy for a specific COU | Treated as a property of the model | Assessment |
| Credibility assessment plan | The pre-specified plan for generating evidence | Written after validation | Document |
| Credibility assessment report | The record of the plan, execution, and deviations | Confused with a validation report | Document |
| Training set | Data used to fit the model | Includes tuning data | Dataset |
| Validation set | Data used to tune hyperparameters | Used for final evaluation | Dataset |
| Test set | Data held out for final evaluation | Reused after seeing results | Dataset |
| Leakage | Information from the test set influencing training | Ignored | Error |
| Discrimination | Ability to rank positives above negatives | Confused with calibration | AUC or c-statistic |
| Calibration | Agreement between predicted and observed risk | Ignored | Calibration slope, intercept |
| AUC | Area under the receiver operating characteristic curve | Interpreted as accuracy | 0.5-1.0 |
| Sensitivity | Proportion of positives detected | Confused with PPV | Proportion |
| Specificity | Proportion of negatives correctly classified | Confused with NPV | Proportion |
| Positive predictive value (PPV) | Proportion of positives who are true positives | Assumed constant across populations | Proportion |
| Threshold | The score cut-off for a decision | Chosen after seeing outcomes | Value |
| Decision curve analysis | Net benefit across thresholds | Omitted | Net benefit |
| External validation | Evaluation on independent data | Confused with cross-validation | Study |
| Prospective validation | Evaluation in the intended workflow | Confused with external validation | Study |
| Drift | Change in input or outcome distribution over time | Ignored | Metric |
| Population stability index (PSI) | Measure of distribution shift | Treated as a performance metric | Index |
| Explainability | Ability to describe why a model produced an output | Confused with interpretability | Property |
| Interpretability | Degree to which a human can understand the model | Confused with explainability | Property |
| Model card | A structured summary of a model's purpose and performance | Treated as documentation only | Document |
| Change control | The process for approving and validating model changes | Omitted | Process |
| Silent mode | Running a model without using its output | Confused with deployment | Mode |

### 4.1 The difference between accuracy and credibility

Accuracy is a property of a model on a dataset. Credibility is a judgement about whether the evidence
supports a specific use. A model with 95% accuracy on a dataset that does not resemble the intended
population is not credible for that population, and a model with 75% accuracy on a representative
population may be credible for a triage decision.

The distinction is why the framework begins with the context of use rather than with the model. The
same model can be credible for one decision and not for another.

## 5. The mechanics, step by step

This procedure follows the FDA's seven-step credibility framework and adds the engineering practices
that make it executable.

### Step 1. Define the question of interest

**Input:** the decision the model is meant to support.

**Output:** a question in decision terms, naming the population, the outcome, the action, and the
consequence of error.

**Rationale:** A model developed to answer "can we predict response?" will be evaluated on
discrimination, while a model developed to answer "should this patient be enrolled?" must be evaluated
on the positive predictive value at the enrolment threshold. The question determines the metric.

### Step 2. Write the context of use

**Input:** the question and the workflow.

**Output:** a statement naming the population, the input data, the output, the timing, the decision,
and the human role.

**Rationale:** The context of use is the contract for validation. It must be specific enough that a
reviewer can decide whether the evidence supports it. "Assists pathologists" is not a context of use;
"provides a score used to select patients with the biomarker state for a Phase II trial, with a
pathologist confirming every positive" is.

### Step 3. Assess model risk

**Input:** the context of use.

**Output:** a risk category from the influence-by-consequence matrix, with the rationale.

**Rationale:** Risk determines the evidence burden. A high-risk model requires external validation,
subgroup analysis, drift monitoring, and change control; a low-risk model may require only internal
validation. The assessment should be made before development, because it determines the data
collection plan.

**Branch point — evidence by risk category:**

| Risk | Minimum validation | Monitoring | Documentation |
|---|---|---|---|
| Low | Internal held-out set | Annual review | Model card |
| Medium | Internal plus one external set | Quarterly performance review | Credibility report |
| High | External multi-site plus prospective or silent-mode evaluation | Continuous drift and performance monitoring | Full credibility assessment with change control |

### Step 4. Plan the data and the splits

**Input:** the available data and the context of use.

**Output:** a data plan with the source, the sampling frame, the labelling procedure, and the split
strategy by site and time.

**Rationale:** Random splits leak information when the data contain repeated patients, sites, or
time periods. A split by site or by time answers the question the model will face in deployment. The
labelling procedure must be described, because the label is often the weakest part of the dataset.

**Edge case:** multiple images per patient. A random split puts images from the same patient in
training and test, and the model learns the patient rather than the biology.

### Step 5. Develop and tune the model

**Input:** the training and validation sets.

**Output:** a fitted model with the preprocessing pipeline, the hyperparameters, and the selection
rationale.

**Rationale:** Every preprocessing step is part of the model. A normalisation computed on the full
dataset, including the test set, is leakage. The pipeline must be fitted on training data only and
applied unchanged to validation and test data.

### Step 6. Internal validation

**Input:** the held-out test set.

**Output:** discrimination, calibration, and performance at the decision threshold, with confidence
intervals.

**Rationale:** Internal validation establishes that the model works on data from the same source. It
does not establish that it will work elsewhere. Reporting only AUC hides the calibration and the
threshold performance that the decision requires.

### Step 7. External validation

**Input:** independent data from other sites, eras, or populations.

**Output:** performance by site and subgroup, with a comparison to internal performance.

**Rationale:** External validation is the strongest evidence short of prospective use. The expected
drop is the difference between the development setting and the deployment setting, and it should be
quantified, not discovered later.

### Step 8. Evaluate at the decision threshold

**Input:** the validated model and the intended threshold.

**Output:** sensitivity, specificity, predictive values at the deployment prevalence, and net benefit.

**Rationale:** A model's AUC is threshold-independent; the decision is not. The threshold should be
chosen on the validation set for a stated objective (for example, a minimum PPV), locked, and then
evaluated on the test set. Choosing the threshold on the test set is a form of leakage.

### Step 9. Assess subgroup and failure-mode performance

**Input:** the validation data with subgroup labels.

**Output:** performance by age, sex, site, scanner, disease severity, and any group with a plausible
distribution shift.

**Rationale:** An average performance can hide a subgroup where the model fails. For a clinical
decision, a subgroup failure is a safety issue, not a statistical curiosity.

### Step 10. Plan lifecycle monitoring

**Input:** the deployment context and the drift risks.

**Output:** monitored inputs, metrics, thresholds, and the actions they trigger.

**Rationale:** Models decay. Inputs change, populations change, and labels change. Monitoring must
cover both the input distribution (drift) and the output performance (where labels are available).
A monitoring plan with no action threshold is a dashboard, not a control.

### Step 11. Write the credibility assessment report

**Input:** the plan and the results.

**Output:** a report with the context of use, the risk assessment, the data, the development, the
evaluation, the deviations, and the adequacy conclusion.

**Rationale:** The report is the evidence a reviewer assesses. It should state what was planned, what
was done, what changed, and what the evidence supports. Deviations are not failures; undocumented
deviations are.

### Step 12. Engage the regulator and plan change control

**Input:** the report and the development plan.

**Output:** a meeting record or advice, and a change-control procedure for model updates.

**Rationale:** Early engagement clarifies the evidence the regulator expects. Change control ensures
that a model update is assessed before it affects a decision, rather than after.

### 5.1 What the credibility assessment plan contains

The plan is the document that distinguishes a validation exercise from a credible one. It should
contain, at minimum:

| Section | Content | Common omission |
|---|---|---|
| Question and COU | The decision, population, input, output, and human role | The human role |
| Risk assessment | Influence, consequence, and the matrix result | The rationale for the rating |
| Data | Source, sampling frame, labelling, exclusions | The labelling procedure |
| Splits | Training, tuning, test, external, with grouping | The grouping variable |
| Development | Model class, features, preprocessing, tuning | Preprocessing fitted on all data |
| Evaluation | Metrics, thresholds, subgroups, intervals | Calibration and PPV |
| Acceptance criteria | What performance is adequate, stated in advance | Criteria set after results |
| Deviations | Any change from the plan, with rationale | Silent changes |
| Monitoring | Inputs, outputs, thresholds, actions | Action thresholds |

The acceptance criteria are the most important and most often missing element. A plan without them
cannot fail, and a validation that cannot fail is not a validation.

### 5.2 Splitting strategies and what each answers

| Split | What it estimates | When to use | Limitation |
|---|---|---|---|
| Random by record | Performance on similar records | Early development only | Leakage if records cluster |
| Grouped by patient | Performance on new patients from the same sites | Internal validation | Still same-site |
| Grouped by site | Performance at new sites | External validation | Few sites, wide intervals |
| Temporal | Performance on future patients | Deployment realism | Confounded by practice change |
| Prospective | Performance in the workflow | Pre-deployment | Expensive, slow |

A validation report should name the split and its grouping variable. "Held-out test set" without the
grouping is uninterpretable, because a random split and a site split answer different questions.

### 5.3 Metrics that match the decision

| Decision | Primary metric | Supporting metrics | Metric to avoid |
|---|---|---|---|
| Rank compounds for synthesis | Enrichment factor at the top k | Recall, precision at k | AUC alone |
| Exclude compounds for toxicity | Negative predictive value | Sensitivity, specificity | Accuracy |
| Select patients for a trial | PPV at the threshold | Sensitivity, calibration | AUC alone |
| Triage images for review | Sensitivity at fixed workload | PPV, review rate | Overall accuracy |
| Predict a continuous outcome | Mean absolute error | Calibration, R² | AUC |

The table is a discipline against reporting the metric that is easiest to compute rather than the one
the decision needs. A model that ranks well but is badly calibrated may be useless for a threshold
decision, and a model that is well calibrated but poorly discriminating may be useful only for
triage.

### 5.4 Documentation: model card and credibility report

The two documents serve different audiences. The model card is a short, structured summary for users:
purpose, inputs, outputs, performance, limitations, and contact. The credibility report is the
evidence package for a reviewer: the plan, the execution, the results, the deviations, and the
adequacy conclusion.

A model card without the report is marketing; a report without the card is inaccessible. Both should
be versioned with the model, and the version should appear in any output the model produces.

### 5.5 Change control for a living model

A model that is retrained, recalibrated, or re-featured is a new model for regulatory purposes. The
change-control process should classify the change:

| Change | Example | Assessment |
|---|---|---|
| Retraining on new data, same pipeline | Annual refresh | Performance revalidation; drift check |
| Recalibration | Intercept update for a new era | Calibration validation |
| New features or model class | Architecture change | Full revalidation |
| New population or use | Expanded label | New context of use |
| Upstream data change | New scanner or assay | Bridging study |

A change that is not classified cannot be validated, and a model that changes silently cannot be
monitored. The classification is a governance decision, not an engineering one.

### 5.6 Bias and fairness assessment

A model that performs differently across subgroups can produce inequitable decisions even when the
overall performance is good. The assessment should cover:

- **Data representation.** Are the subgroups present in training and test data in sufficient numbers?
- **Label bias.** Were the labels assigned consistently across groups, or does one group receive more
  or less testing?
- **Performance by group.** Sensitivity, specificity, and PPV by subgroup, with confidence intervals.
- **Consequence by group.** A false negative may be more harmful in one group because of access to
  alternative care.

The assessment is not a statistical formality. A model that selects patients for a trial and performs
poorly in a group that is already underrepresented will worsen the evidence gap that the diversity
plan is meant to address.

### 5.7 Reproducibility requirements

A model that cannot be reproduced cannot be reviewed. Four artefacts are necessary:

1. **The code**, with the version and dependencies pinned.
2. **The data version**, with the extract date and the inclusion criteria.
3. **The trained parameters**, with a checksum.
4. **The random seeds**, for training and for any stochastic preprocessing.

A validation report that lacks these cannot be replicated by a reviewer or by the sponsor's own team
after staff turnover. Reproducibility is a precondition for credibility, not a separate virtue.

### 5.8 Human factors and automation bias

A model used by a person changes that person's behaviour. Automation bias is the tendency to accept a
model's output even when other evidence suggests otherwise, and it is strongest when the output is
presented as a score rather than as information.

The mitigations are design choices: present the model's uncertainty, require the human to record the
basis for overriding it, and audit override rates. A model that is never overridden is either perfect
or ignored, and the second is more likely.

### 5.9 When not to use a model

A model is not always the right tool. The alternatives should be considered before development:

- **A deterministic rule.** If the decision can be expressed as a threshold on a measured variable, a
  rule is easier to validate, explain, and monitor.
- **A simpler model.** If a logistic regression performs within the confidence interval of the complex
  model, the simpler model is the better regulatory object.
- **A prospective study.** If the question is whether an intervention works, a model that predicts the
  outcome is not a substitute for measuring it.
- **No change.** If the current process performs adequately, the burden of validation may exceed the
  benefit.

The last option is the most often skipped. A model that improves a metric by two points but requires
a credibility package, monitoring, and change control may be a net cost to the programme.

### 5.10 The questions to ask before regulatory engagement

1. Is the model's output used to support a claim of safety, effectiveness, or quality?
2. What is the context of use, and is the human role part of it?
3. What is the model risk, and what evidence does the risk require?
4. What external data exist, and can they be accessed?
5. How will the model be monitored after approval or deployment?
6. What change-control process will govern updates?

The answers determine whether the engagement is a scientific advice question, a qualification question,
or a pre-submission discussion.

### 5.11 The one-page model summary

For governance, the credibility report should be accompanied by a one-page summary containing: the
decision the model supports; the risk category; the external performance at the decision threshold;
the main limitation; the monitoring plan; and the name of the accountable owner. A governance board
that cannot see these six items cannot make a decision, and a report that hides them behind technical
detail will be approved without being understood.

### 5.12 The review conversation

The credibility discussion is easier when the sponsor and the reviewer agree on the question before
the evidence. A useful structure is: state the decision; state the error that matters; state the
evidence that would detect it; state what was found; state what remains uncertain. That structure
keeps the conversation on the adequacy of the evidence rather than on the sophistication of the model,
and it makes the reviewer's objection specific: the disagreement is about an error, not about a
preference for simpler methods.

The structure also records what was not assessed, which is often the most useful part of the record
for the next programme.

## 6. Worked example

> **Note:** The model, sponsor, and data are fictional. The framework and the regulatory documents
> cited are real and referenced in Section 16.

### 6.1 The model and the decision

A sponsor has developed a convolutional neural network that scores haematoxylin and eosin stained
slides for a predictive biomarker. The intended use is to select patients for a Phase II trial in a
solid tumour. A pathologist reviews every positive score, but a negative score excludes the patient
from screening.

- Development set: 1,200 slides from three hospitals in one country, 18% biomarker-positive.
- Internal test set: 240 slides from the same hospitals, held out by patient.
- External test set: 600 slides from four hospitals in two other countries, 15% biomarker-positive.

### 6.2 The results

| Metric | Internal | External |
|---|---:|---:|
| AUC | 0.88 | 0.74 |
| Sensitivity at locked threshold | 0.86 | 0.82 |
| Specificity at locked threshold | 0.84 | 0.78 |
| Calibration slope | 0.95 | 0.71 |

The drop from 0.88 to 0.74 is 0.14 AUC. It is the expected consequence of site shift: different
staining protocols, scanners, and case mix. The calibration slope of 0.71 means the model's predicted
probabilities are too extreme in the external data, which matters because the threshold is applied to
the probability.

### 6.3 The decision-threshold arithmetic

At 15% prevalence, with sensitivity 0.82 and specificity 0.78:

```text
True positives detected   = 0.15 x 0.82 = 0.1230
False positives           = 0.85 x 0.22 = 0.1870
PPV                       = 0.1230 / (0.1230 + 0.1870) = 0.397
NPV                       = 0.85 x 0.78 / (0.85 x 0.78 + 0.15 x 0.18) = 0.663/0.690 = 0.961
```

So 39.7% of model-positive patients are truly biomarker-positive, and 96.1% of model-negative
patients are truly negative. The model is good at excluding and mediocre at selecting.

### 6.4 What the error does to the trial

Assume that truly positive patients have a 48% response rate on the drug and truly negative patients
have 18%.

```text
Enriched (model-positive) ORR = 0.397 x 0.48 + 0.603 x 0.18 = 0.191 + 0.109 = 29.9%
All-comers ORR                = 0.15 x 0.48 + 0.85 x 0.18 = 0.072 + 0.153 = 22.5%
```

The model improves the expected response rate by 7.4 percentage points, not the 30 points that the
true biomarker would deliver. The trial is smaller than an all-comers trial, but much larger than a
trial in a perfectly selected population.

### 6.5 Two diagnostics that reveal a broken validation

**Diagnostic 1 — the split leaked by patient or site.** If the internal test set contains slides from
patients in the training set, the internal AUC is inflated. The diagnostic is to group the split by
patient identifier and to report the number of patients, not only slides, in each set. In this
example, the internal set was split by patient but not by hospital, which is why the external drop is
so large: the model learned hospital-specific staining.

**Diagnostic 2 — the threshold was tuned on the test set.** If the locked threshold differs from the
threshold specified in the protocol, it was probably chosen after seeing test performance. The
diagnostic is to compare the protocol's threshold with the analysis code's threshold and the dates of
both.

A third check is calibration. A model with good discrimination and poor calibration can still rank
patients correctly while producing probabilities that are wrong, which matters when the probability
is compared with a threshold.

### 6.6 Vary one assumption: prevalence

The programme considers restricting screening to a biomarker-enriched population with 30% prevalence,
for example by pre-selecting patients with a related clinical feature.

```text
PPV = (0.30 x 0.82) / (0.30 x 0.82 + 0.70 x 0.22) = 0.246 / (0.246 + 0.154) = 0.615
Enriched ORR = 0.615 x 0.48 + 0.385 x 0.18 = 0.295 + 0.069 = 36.4%
```

At 30% prevalence the PPV rises to 61.5% and the enriched response rate to 36.4%. The same model, with
the same sensitivity and specificity, becomes a much better selection tool. The decision changes from
"the model is not fit for selection" to "the model is fit for selection in the enriched population,
provided the enrichment rule is validated".

The lesson is that model performance is not a property of the model alone. It depends on the
population in which it is applied, and the population can sometimes be changed more cheaply than the
model.

### 6.7 What the example establishes

1. Internal validation is a ceiling, not an estimate of deployment performance.
2. The AUC drop is expected and should be planned for with external data.
3. Predictive values, not AUC, determine whether a selection model works.
4. Calibration matters when probabilities are compared with thresholds.
5. Prevalence can change the decision more than model improvement.

### 6.8 The credibility determination

Applying the framework: the model's influence is high (a negative score excludes a patient from
screening) and the decision consequence is high (patients are denied a potentially effective therapy,
and the trial's result is affected). The risk is therefore high, and the evidence required is external
multi-site validation plus prospective or silent-mode evaluation.

The external AUC of 0.74 and the calibration slope of 0.71 do not meet the pre-specified acceptance
criterion of 0.80 AUC and 0.85 calibration slope. The determination is "not adequate for the context
of use as specified". The model may be adequate for a narrower context: selection within an enriched
population at 30% prevalence, or triage with a confirmatory test for every patient.

### 6.9 What would change the decision

Three changes would make the model credible for the original context:

1. **Retraining with site-balanced data and staining normalisation**, followed by external validation
   at 0.80 AUC or above with calibration slope above 0.85.
2. **A change in the workflow** so that the model triages rather than excludes, with a confirmatory
   test for all model-negative patients. This lowers the consequence of a false negative.
3. **A change in the population** to 30% prevalence, where the PPV is 61.5% and the enriched response
   rate is 36.4%.

Each change addresses a different part of the risk equation. The first improves the model; the second
reduces the consequence; the third changes the population. A programme should compare their costs
before choosing.

### 6.10 The cost of the wrong model

If the model is deployed as specified, 60% of selected patients do not have the biomarker state, and
the trial's response rate is 29.9% instead of the 48% expected in a truly selected population. The
trial may still succeed, but it will be larger and slower than planned, and the label will be
ambiguous.

If the model is rejected without an alternative, the programme returns to conventional testing, which
may be slower and more expensive per patient but has known performance. The decision is not between a
model and perfection; it is between a model with quantified error and a process with different
error.

### 6.11 Sensitivity to the threshold

The locked threshold determines the trade-off between false positives and false negatives. Lowering
the threshold increases sensitivity and PPV falls; raising it increases PPV and misses more true
positives.

At 15% prevalence, a threshold that gives sensitivity 0.90 and specificity 0.65 produces PPV = 0.135/
(0.135 + 0.2975) = 0.312, with a higher detection of true positives. A threshold that gives
sensitivity 0.70 and specificity 0.88 produces PPV = 0.105/(0.105 + 0.102) = 0.507, with more missed
cases.

The choice depends on the consequence of each error. For a trial enrolment decision, a false negative
denies a patient the trial and loses a potential responder; a false positive enrols a non-responder
and dilutes the effect. The programme should state which error it prefers and why, rather than
defaulting to the threshold that maximises accuracy.

### 6.12 Confidence intervals for the external estimate

The external AUC of 0.74 is a point estimate from 600 slides. A bootstrap interval clustered by
hospital would be wide, because the effective sample size is closer to the number of hospitals than to
the number of slides when the site effect is large.

With four hospitals and substantial between-site variation, the interval may span 0.66 to 0.81. The
lower bound matters for the decision: if it falls below the acceptance criterion, the model cannot be
declared adequate even if the point estimate is close.

A validation report that presents a point estimate without an interval overstates the precision of the
evidence, and a governance board that sees only the point estimate will under-weight the risk.

## 7. Code, query, or calculation

The script computes the AUC from paired comparisons, the predictive values at a stated prevalence,
the enriched response rate, and a population stability index. It uses NumPy only.

```python
import numpy as np

def auc(y, score):
    """Mann-Whitney estimate of AUC, ties counted as 0.5."""
    y = np.asarray(y)
    score = np.asarray(score)
    pos, neg = score[y == 1], score[y == 0]
    diff = pos[:, None] - neg[None, :]
    return ((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg))

y = [1, 1, 1, 1, 0, 0, 0, 0]
s = [0.9, 0.8, 0.6, 0.4, 0.7, 0.5, 0.3, 0.2]
print(f"AUC worked example: {auc(y, s):.4f}")

def predictive_values(prevalence, sensitivity, specificity):
    tp = prevalence * sensitivity
    fp = (1 - prevalence) * (1 - specificity)
    tn = (1 - prevalence) * specificity
    fn = prevalence * (1 - sensitivity)
    return tp / (tp + fp), tn / (tn + fn)

for prev in (0.15, 0.30):
    ppv, npv = predictive_values(prev, 0.82, 0.78)
    enriched = ppv * 0.48 + (1 - ppv) * 0.18
    allcomers = prev * 0.48 + (1 - prev) * 0.18
    print(f"prevalence {prev:.2f}: PPV {ppv:.3f} NPV {npv:.3f} "
          f"enriched ORR {enriched:.1%} all-comers {allcomers:.1%}")

def psi(expected, actual, bins=10):
    e = np.histogram(expected, bins=bins, range=(0, 1))[0].astype(float)
    a = np.histogram(actual, bins=bins, range=(0, 1))[0].astype(float)
    e = np.clip(e / e.sum(), 1e-4, None)
    a = np.clip(a / a.sum(), 1e-4, None)
    return np.sum((a - e) * np.log(a / e))

rng = np.random.default_rng(7)
print(f"PSI, stable: {psi(rng.beta(2, 5, 2000), rng.beta(2, 5, 2000)):.3f}")
print(f"PSI, shifted: {psi(rng.beta(2, 5, 2000), rng.beta(2.6, 4.4, 2000)):.3f}")
```

Expected output (Python 3.12, NumPy 2.4, executed):

```text
AUC worked example: 0.8125
prevalence 0.15: PPV 0.397 NPV 0.961 enriched ORR 29.9% all-comers 22.5%
prevalence 0.30: PPV 0.615 NPV 0.910 enriched ORR 36.4% all-comers 27.0%
PSI, stable: 0.007
PSI, shifted: 0.270
```

Reading the output:

- The worked example's AUC is 0.8125, computed from 13 correctly ordered pairs out of 16.
- The PPV doubles from 39.7% to 61.5% when prevalence rises from 15% to 30%, with the same
  sensitivity and specificity.
- The PSI is 0.007 for a stable population and 0.270 for a shifted one. A common convention treats a
  PSI below 0.10 as no meaningful shift, 0.10-0.25 as moderate, and above 0.25 as major.

The PSI thresholds are a convention, not a regulatory standard. The appropriate threshold depends on
how the score is used and how much drift the decision can tolerate.

The code does not compute confidence intervals for the AUC or the PPV. A production validation would
use bootstrap intervals stratified by site, which is the subject of Exercise 5.

## 8. Data, standards, and regulatory context

### 8.1 The FDA framework

The FDA's draft guidance *Considerations for the Use of Artificial Intelligence to Support Regulatory
Decision-Making for Drug and Biological Products*, issued on 6 January 2025, proposes a risk-based
credibility assessment framework for AI models used to produce information supporting regulatory
decisions. Its seven steps are the structure of Section 5:

1. Define the question of interest.
2. Define the context of use.
3. Assess model risk as model influence multiplied by decision consequence.
4. Develop a credibility assessment plan.
5. Execute the plan.
6. Document the results and deviations in a credibility assessment report.
7. Determine adequacy for the context of use.

The guidance covers nonclinical, clinical, post-marketing, and manufacturing uses and excludes drug
discovery and operational efficiency. It also addresses lifecycle maintenance and early engagement.

### 8.2 The EMA reflection paper

The EMA's *Reflection Paper on the Use of Artificial Intelligence in the Medicinal Product Lifecycle*
was published on 30 September 2024. It covers discovery, non-clinical development, clinical trials,
precision medicine, product information, manufacturing, and post-authorisation, and sets out
principles for a risk-based approach, data quality, model development, performance assessment,
interpretability, governance, data protection, and trustworthy AI.

The reflection paper is not a guideline; it describes principles and points to the regulatory
interactions available (innovation task force, scientific advice, qualification of novel
methodologies). A programme should treat it as the EMA's statement of expectations, not as a
checklist.

### 8.3 Other instruments

| Instrument | Scope | Status |
|---|---|---|
| EU Artificial Intelligence Act (Regulation 2024/1689) | Risk classification of AI systems, obligations for high-risk systems | EU law, phased application |
| GDPR | Personal data processing, automated decision-making | EU law |
| ICH Q12 | Lifecycle management of manufacturing changes | ICH guideline |
| FDA guidance on AI-enabled device software functions | Device AI, lifecycle and submissions | Draft, January 2025 |
| Model risk management guidance from banking regulators | Model governance analogies | Not applicable to medicines directly |

The EU AI Act classifies some medical AI as high-risk, which brings obligations for risk management,
data governance, technical documentation, and human oversight. A model used in a clinical trial may
fall within scope depending on its function and the jurisdiction.

### 8.4 Requirement versus expectation

- **Requirement.** The EU AI Act imposes legal obligations on providers of high-risk AI systems,
  including those used in medical contexts. The GDPR governs personal data and automated decisions.
- **Expectation.** The FDA's credibility framework is a draft guidance, so it is not binding, but it
  describes the evidence the agency will expect. The EMA reflection paper is not binding either, but
  it signals the review approach.
- **Convention.** The choice of AUC as a metric, the PSI thresholds, and the use of a model card are
  conventions. They may be good practice; they are not requirements.

### 8.5 What the guidance expects in a submission

A submission that relies on an AI-derived result should be able to show:

- the question and context of use, with the human role;
- the model risk assessment and the evidence proportionate to it;
- the data sources, labelling, and splits;
- the model development, including preprocessing and tuning;
- the evaluation, including external validation and subgroup performance;
- the credibility assessment report, with deviations;
- the lifecycle plan, including monitoring and change control.

The list is an inference from the framework, not a checklist published by the agency. It is,
however, the set of artefacts that the framework's seven steps produce when they are followed. A programme that
provides these artefacts is in a strong position to answer the questions a reviewer will ask.

### 8.6 The data standard question

AI models consume data that are often not standardised for regulatory use: free-text notes, images,
waveforms, and sensor streams. The submission of such data raises questions that tabular standards do
not answer: how the raw data are represented, how the preprocessing is documented, and how a reviewer
can reproduce the model's input.

A programme should define a data representation for the model inputs, document the preprocessing as
part of the model, and retain the raw data or a traceable derivative. The standard for the model input
is as important as the standard for the clinical dataset.

### 8.7 What the frameworks do not cover

Neither the FDA framework nor the EMA reflection paper specifies a performance threshold for a given
context of use. They require the sponsor to justify the threshold and to assess adequacy. That is a
scientific judgement, and it is the part of the package most likely to be challenged.

## 9. Industry practice and case evidence

### 9.1 Where the field actually is

AI-discovered molecules have entered clinical trials. A 2025 review estimated that more than 75
AI-designed molecules had entered human trials by mid-2025, up from none in 2020; the figure is a
review estimate, not a registry count. Insilico Medicine reported Phase IIa results for rentosertib
in idiopathic pulmonary fibrosis, with 71 patients and a change in forced vital capacity that
favoured the 60 mg dose over placebo, and with safety as the primary endpoint met. No AI-discovered
drug had been approved as of the source cutoff.

The pattern is that AI compresses the discovery phase and the binding constraint moves to target
validation, clinical translation, and regulatory evidence. A molecule that was designed quickly still
requires the same trials.

**Documented:** the trial and the review estimate. **Inference:** that the number of AI-designed
molecules in trials overstates the number that will reach approval, because the base rate of clinical
success applies to them as it does to any molecule.

### 9.2 What companies validate and what they do not

| Use case | Common validation | Common gap |
|---|---|---|
| Virtual screening | Retrospective enrichment on known actives | Prospective hit rate |
| ADMET prediction | Held-out test set | Prospective synthesis and assay |
| Pathology imaging | Internal and external AUC | Calibration and workflow impact |
| Trial feasibility | Comparison with observed enrolment | Prospective forecasting |
| Safety signal detection | Retrospective comparison with known signals | Prospective detection |
| Digital endpoints | Analytical validation | Regulatory qualification |

The gap in the right-hand column is where the credibility framework adds value. A retrospective
enrichment factor is evidence that the model can rank known actives; it is not evidence that it will
find new ones.

### 9.5 AI in trial operations

The least regulated and most rapidly adopted uses are operational: site selection, enrolment
forecasting, monitoring visit targeting, and query prediction. These models have low decision
consequence individually, but they can have high aggregate influence because they allocate attention
and resources across a portfolio.

The validation standard is different from a clinical model: the question is whether the model improves
the operational decision compared with the current process, and that question is best answered by a
prospective comparison rather than by retrospective accuracy. A monitoring model that predicts queries
well but does not reduce them has no operational value.

### 9.6 Vendor and third-party models

Many programmes use vendor models rather than building their own. The credibility question then
becomes: what evidence does the vendor provide, and can the sponsor reproduce it?

| Evidence | What it supports | What it does not |
|---|---|---|
| Vendor validation report | Performance on the vendor's data | Performance on the sponsor's population |
| Model card | Intended use and limitations | Independent verification |
| API documentation | Input and output specification | Stability of the model over time |
| Version history | Change tracking | Validation of each change |
| Audit rights | Ability to inspect | Access to training data |

A sponsor that cannot inspect the model or reproduce its validation is relying on the vendor's
assurance. That may be acceptable for a low-consequence use and is rarely acceptable for a high-
consequence one.

### 9.7 Adoption is a separate problem

A validated model is not an adopted model. Adoption depends on workflow fit, trust, training, and the
incentives of the people who use it. A model that adds five minutes to a pathologist's day will be
ignored unless the benefit is visible to the pathologist, not only to the sponsor.

The evidence for adoption is operational: usage rates, override rates, time saved, and the
characteristics of non-users. A programme that measures only model performance has measured half of
the intervention.

### 9.8 What a good AI programme looks like

| Property | Evidence | Why it matters |
|---|---|---|
| A named context of use | Approved document | Validation has a target |
| A risk assessment | Matrix result with rationale | Evidence is proportionate |
| External data | Independent cohort | Performance generalises |
| Threshold performance | PPV at deployment prevalence | The decision is assessed |
| Monitoring | Drift and performance metrics | Decay is detected |
| Change control | Classified updates | The model stays validated |
| Reproducibility | Code, data version, seeds | The evidence can be checked |

A programme with these seven properties can answer the regulator's questions. A programme without them
will answer them retrospectively, under time pressure, which is the most expensive way to discover a
gap.

### 9.3 Case: a model that was rejected

A programme developed a model to predict which compounds would pass a toxicity assay, trained on
historical assay data. The model performed well retrospectively but failed prospectively because the
historical assay had changed: a reagent change altered the assay's sensitivity, and the model had
learned the old reagent's behaviour.

**Documented:** the phenomenon of assay drift and its effect on predictive models is well described in
the analytical literature. **Inference:** the specific failure, which is constructed here, illustrates
why lifecycle monitoring must cover the assay and not only the model.

The remedy is a change-control process that treats an assay change as a model change, requiring
revalidation before the model is used for decisions.

### 9.4 What is documented and what is inferred

| Claim | Status |
|---|---|
| The FDA issued a draft AI credibility guidance in January 2025 | Documented |
| The EMA published a reflection paper in September 2024 | Documented |
| AI-designed molecules are in clinical trials | Documented by review estimates |
| No AI-discovered drug had been approved as of the source cutoff | Documented in the cited review |
| Most AI models in development lack external validation | Inference; not systematically measured |
| Prospective validation improves adoption | Inference; plausible but not quantified |

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No context of use | The model is described as "predictive" | Validation has no target | Evidence cannot be assessed | Write the COU first |
| Leakage by patient | Test set shares patients with training | Memorisation | Inflated internal performance | Group splits by patient |
| Leakage by site | All sites in both sets | Site-specific features | External drop | Split by site |
| Preprocessing leakage | Normalisation on the full dataset | Test information in training | Inflated performance | Fit pipeline on training only |
| Threshold tuned on test | Threshold differs from protocol | Selection on test | Optimistic decision performance | Lock threshold on validation |
| AUC reported alone | No calibration or PPV | Metric mismatch | Model looks better than it is | Report threshold metrics |
| Prevalence ignored | PPV quoted from the development set | Base rate shift | Wrong selection performance | Recompute at deployment prevalence |
| No subgroup analysis | Average performance only | Hidden subgroup failure | Inequitable or unsafe use | Report by subgroup |
| No drift monitoring | Model performance reviewed annually | Silent decay | Wrong decisions over time | Monitor inputs and outputs |
| Assay change not bridged | Reagent or platform change | Model learned old behaviour | Prospective failure | Change control including the assay |
| Undocumented deviations | Report differs from plan | Post-hoc changes | Credibility lost | Record deviations |
| Overclaiming discovery | "AI-designed drug" in a press release | Discovery conflated with approval | Reputational and regulatory risk | Separate the claims |

The most common red flag in internal review is an AUC reported without the threshold performance and
without the prevalence. It signals that the team is reporting what the model can do rather than what
the decision needs.

### 10.1 Red flags by development stage

| Stage | Red flag | What it usually means |
|---|---|---|
| Question | No decision named | The model has no use case |
| Data | Labels from a proxy with no validation | The target is not what the model learns |
| Splits | Random split with repeated patients | Leakage |
| Development | Preprocessing fitted on all data | Leakage |
| Validation | Only AUC reported | Metric mismatch |
| External | No external data | Performance unknown outside development |
| Deployment | No monitoring plan | Drift will be invisible |
| Lifecycle | Updates without change control | Untracked model versions |

A programme that catches the first four in design review avoids most of the cost of the last four.

### 10.3 The organisational red flags

Technical red flags are easier to see than organisational ones, and the organisational ones are more
predictive of failure.

| Red flag | What it signals |
|---|---|
| The model owner is not the decision owner | No one is accountable for the consequence |
| Validation is performed by the development team | No independent challenge |
| The model card is written by communications | Performance claims are not technical |
| Monitoring has no budget | The model will decay unobserved |
| Change control is an IT ticket | Model changes bypass validation |
| The regulator is engaged after the analysis | The evidence may not match the question |

An organisation that has these red flags will produce a technically competent model with a weak
credibility package, and the weakness will surface at review.

### 10.2 The failure that looks like success

The most dangerous outcome is a model that improves a metric the team tracks while degrading the
decision. A model that increases the number of patients screened, or the number of compounds tested,
may look productive while selecting the wrong patients or compounds.

The defence is to measure the decision, not the activity. For a selection model, the metric is the
response rate in the selected population, not the number selected. For a screening model, it is the
hit rate, not the number of screens.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The label is correct | Supervised learning depends on it | Labelling error, proxy labels | Label audit, adjudication |
| The development data resemble deployment | Performance transfers | Site, era, population shift | External validation |
| The threshold is stable | Decisions use a fixed cut-off | Prevalence or calibration change | Re-derive at local prevalence |
| The model's inputs are available at decision time | The workflow depends on it | Late lab or imaging | Workflow analysis |
| The human role is defined | Accountability | Automation bias | Usability study |
| Drift is detectable | Monitoring depends on it | No labels, no input data | Shadow mode, input monitoring |
| The model is reproducible | Regulatory review | Non-deterministic training | Seeds, environment capture |
| Subgroup performance is adequate | Equity and safety | Small subgroups | Stratified analysis |
| The change-control process works | Lifecycle safety | Model updates bypass review | Audit trail |
| The claimed benefit is causal | Impact claims | Confounding by workflow | Prospective or randomised evaluation || Users will follow the model | Adoption | Automation bias, workflow friction | Usability study, override audit |

The deepest limit is that a model's performance is a property of the model, the data, and the
population together. Changing any one of them changes the performance, and only monitoring detects
the change.

A second limit is that credibility is a judgement, not a measurement. The framework structures the
judgement, but it does not remove the need for an accountable person to decide that the evidence is
sufficient for the decision. That person should be named in the report.

### 11.1 When to re-validate

Re-validation is required when the model, the data, the population, or the workflow changes. The
common trigger that is missed is the upstream data change: a new scanner, a new assay, or a new EHR
vendor changes the inputs without changing the model, and the model's performance changes without
anyone editing a line of code.

A practical rule is to re-validate when any input variable's distribution moves beyond the monitoring
threshold, when the model is retrained, or when the context of use changes. Waiting for an annual
review is a choice to be late.

A third limit is that the regulatory guidance is recent and evolving. The FDA framework is a draft,
the EMA paper is a reflection, and the AI Act is still being implemented. A programme should design to
the principles and check the current texts rather than assume the detail is settled.

## 12. Exercises

**Exercise 1 — Context of use (design, 30 minutes).**
Deliverable: a context-of-use statement for an AI model in a development decision, naming population,
input, output, timing, decision, and human role.
Acceptance criteria: specific enough to validate; human role explicit; consequence of error stated.
Time: 30 minutes.

**Exercise 2 — Risk assessment (application, 25 minutes).**
Deliverable: a model risk classification for four use cases using the influence-by-consequence matrix,
with rationale.
Acceptance criteria: each classification justified; evidence burden stated for each.
Time: 25 minutes.

**Exercise 3 — Split design (design, 30 minutes).**
Deliverable: a data split plan for an imaging model with multiple images per patient across five
sites.
Acceptance criteria: grouping by patient and site; test set held out by site; labelling procedure
described.
Time: 30 minutes.

**Exercise 4 — Threshold arithmetic (application, 30 minutes).**
Deliverable: PPV and NPV at 10%, 20%, and 30% prevalence for sensitivity 0.80 and specificity 0.85,
with the implication for selection.
Acceptance criteria: arithmetic correct; the prevalence dependence explained; a recommendation on
whether the model is fit for selection at each prevalence.
Time: 30 minutes.

**Exercise 5 — External validation plan (design, 40 minutes).**
Deliverable: a plan for external validation of a pathology model, including sites, sample size,
subgroup analyses, and bootstrap confidence intervals.
Acceptance criteria: sample size justified; subgroups pre-specified; performance metrics include
calibration and PPV.
Time: 40 minutes.

**Exercise 6 — Drift monitoring (design, 35 minutes).**
Deliverable: a monitoring plan with input metrics, output metrics, thresholds, and actions.
Acceptance criteria: at least four metrics; thresholds justified; an action for each threshold; an
owner.
Time: 35 minutes.

**Exercise 7 — Credibility report (application, 50 minutes).**
Deliverable: a credibility assessment report outline for a model used in patient selection, following
the seven steps.
Acceptance criteria: all seven steps represented; deviations section included; adequacy conclusion
stated with its limits.
Time: 50 minutes.

**Exercise 8 — Red-team an AI claim (critique, 45 minutes).**
Deliverable: a critique of a press release claiming that an AI-designed drug "has been validated in
clinical trials", separating what is documented from what is inferred.
Acceptance criteria: at least five claims classified; the base rate of clinical success applied; no
invented results.
Time: 45 minutes.

## 13. Self-check answers

**Exercise 1.** A complete statement names the population (for example, adults with the tumour type
and adequate tissue), the input (slides scanned on a specified platform), the output (a probability
score), the timing (before enrolment), the decision (screen or do not screen), and the human role
(pathologist confirms positives). It should also state the consequence of a false positive and a false
negative.

**Exercise 2.** A compound-ranking model is low influence and low consequence; an ADMET filter is
medium influence and low consequence; a patient-selection model is high influence and high
consequence; a safety signal detector is medium influence and medium consequence. The high-risk case
requires external and prospective evidence; the low-risk case requires internal validation.

**Exercise 3.** Split by patient first, then by site: training sites A, B, C; test site D; and a
further held-out site E for a second external check. The labelling procedure should describe who
labeled the slides, with what reference, and how disagreements were resolved.

**Exercise 4.** At 10%: PPV = 0.08/(0.08+0.135) = 0.372; NPV = 0.765/(0.765+0.02) = 0.974. At 20%:
PPV = 0.16/(0.16+0.12) = 0.571; NPV = 0.68/(0.68+0.04) = 0.944. At 30%: PPV = 0.24/(0.24+0.105) =
0.696; NPV = 0.595/(0.595+0.06) = 0.908. The model is suitable for selection only at the higher
prevalences, unless the false-positive cost is low.

**Exercise 5.** The plan should specify at least three external sites with different scanners and
staining protocols, a sample size justified by the expected confidence interval width, pre-specified
subgroups (stage, grade, site), and bootstrap intervals clustered by patient. It should report AUC,
calibration slope, sensitivity, specificity, and PPV at the deployment prevalence.

**Exercise 6.** Input metrics: score distribution (PSI), missingness, scanner mix. Output metrics:
positive rate, PPV where labels exist, referral rate. Thresholds: PSI above 0.25 triggers review; a
positive rate change of more than 20% triggers review; PPV below a pre-specified floor triggers
suspension. Actions have owners and timelines.

**Exercise 7 rubric.** Full marks require all seven steps, a deviations section, and an adequacy
conclusion that states the limits of the evidence. Deduct marks for a report that presents only
performance metrics.

**Exercise 8.** Claims to classify: the molecule was designed with AI (documented if disclosed); it
entered clinical trials (documented by registration); it was validated in trials (depends on the
endpoint and the phase); it is effective (not established by a Phase IIa); it will be approved (a
forecast). The base rate of Phase I to approval is roughly 8%, which is the appropriate prior for the
last claim.**Exercise 8.** Claims to classify: the molecule was designed with AI (documented if disclosed); it
entered clinical trials (documented by registration); it was validated in trials (depends on the
endpoint and the phase); it is effective (not established by a Phase IIa); it will be approved (a
forecast). The base rate of Phase I to approval is roughly 8%, which is the appropriate prior for the
last claim.

A strong answer also distinguishes the claim types: a design claim is about the process, a trial entry
claim is about a milestone, an efficacy claim is about an endpoint, and an approval claim is about a
regulatory decision. Conflating them is how a Phase IIa result becomes a headline.

## 14. Interview questions

**1. What is a context of use, and why does it come before model development?**
Strong: it specifies the population, input, output, decision, and human role, and determines the
validation evidence. Weak: says it is the intended use statement.

**2. How do you assess model risk?**
Strong: model influence multiplied by decision consequence, with evidence proportionate to the
product. Weak: uses accuracy as a proxy for risk.

**3. Why does external performance drop?**
Strong: site shift, spectrum bias, prevalence change, temporal drift, and upstream changes. Weak: says
the model overfits.

**4. What is leakage, and how do you prevent it?**
Strong: information from test data influencing training, through patient, site, time, or
preprocessing; prevent with grouped splits and pipelines fitted on training only. Weak: says it is
overfitting.

**5. Why is AUC not enough?**
Strong: it is threshold-independent and ignores calibration and prevalence, while the decision needs
PPV at a threshold. Weak: says AUC is a good metric.

**6. How would you choose a threshold?**
Strong: on validation data for a stated objective, locked, then evaluated on test data; consider net
benefit. Weak: chooses the threshold that maximises accuracy.

**7. What would you monitor after deployment?**
Strong: input drift, output rates, performance where labels exist, and subgroup performance, with
thresholds and actions. Weak: says "monitor accuracy".

**8. How do you handle a model update?**
Strong: change control, revalidation, documentation, and regulatory assessment if the context of use
changes. Weak: deploys the update and monitors.

**9. What does the FDA credibility framework require?**
Strong: seven steps from question to adequacy, with a plan and a report. Weak: says it requires
validation.

**10. When would you reject a model?**
Strong: when external performance at the decision threshold is inadequate for the consequence, when
subgroup failures are unsafe, or when drift cannot be monitored. Weak: when the AUC is below 0.8.

## 15. Key terms

- **AUC** — area under the receiver operating characteristic curve.
- **Calibration** — agreement between predicted and observed risk.
- **Change control** — the process for approving and validating model changes.
- **Context of use (COU)** — the specified question, population, output, and decision.
- **Credibility** — evidence that a model is trustworthy for a specific context of use.
- **Credibility assessment plan** — the pre-specified plan for generating evidence.
- **Decision consequence** — the impact of a wrong decision.
- **Discrimination** — ability to rank positives above negatives.
- **Drift** — change in input or outcome distribution over time.
- **External validation** — evaluation on independent data.
- **Leakage** — test information influencing training.
- **Model influence** — the degree to which the model affects the decision.
- **Model risk** — influence multiplied by consequence.
- **Population stability index (PSI)** — measure of distribution shift.
- **Positive predictive value (PPV)** — proportion of positives who are true positives.
- **Prospective validation** — evaluation in the intended workflow.
- **Silent mode** — running a model without using its output.
- **Threshold** — the score cut-off for a decision.
- **Training set** — data used to fit the model.
- **Validation set** — data used to tune hyperparameters.
- **Test set** — data held out for final evaluation.

## 16. Sources

- US Food and Drug Administration. *Considerations for the Use of Artificial Intelligence to Support
  Regulatory Decision-Making for Drug and Biological Products*. Draft guidance, January 2025.
  https://www.fda.gov/media/184830/download (accessed 25 October 2026).
- US Food and Drug Administration. *FDA Proposes Framework to Advance Credibility of AI Models Used
  for Drug and Biological Product Submissions*. January 2025.
  https://www.fda.gov/news-events/press-announcements/fda-proposes-framework-advance-credibility-ai-models-used-drug-and-biological-product-submissions
  (accessed 25 October 2026).
- European Medicines Agency. *Reflection Paper on the Use of Artificial Intelligence in the Medicinal
  Product Lifecycle*. 30 September 2024.
  https://www.ema.europa.eu/en/documents/scientific-guideline/reflection-paper-use-artificial-intelligence-ai-medicinal-product-lifecycle_en.pdf
  (accessed 25 October 2026).
- European Union. *Regulation (EU) 2024/1689, Artificial Intelligence Act*.
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj (accessed 25 October 2026).
- ICH. *Q12 Technical and Regulatory Considerations for Pharmaceutical Product Lifecycle Management*.
  https://www.ich.org/page/quality-guidelines (accessed 25 October 2026).
- US Food and Drug Administration. *AI-Enabled Device Software Functions: Lifecycle Management and
  Marketing Submission Recommendations*. Draft guidance, January 2025.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents (accessed 25 October 2026).

> **Note:** The model and its performance figures in Section 6 are fictional. The regulatory
> documents and the AI-drug trial figures in Section 9 are real and cited; the interpretation is the
> course's own.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`07_rwe-in-pharma.md`](07_rwe-in-pharma.md)
- Next topic: [`09_decentralized-trials.md`](09_decentralized-trials.md)
- Related: [`03_biomarker-strategy.md`](03_biomarker-strategy.md)
- Related: [`10_submission-data.md`](10_submission-data.md)
- Survey file: [`../../13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md)
- Labs: [`../../labs/`](../../labs/)
