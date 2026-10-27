# Patient-Level Digital Twins and In Silico Trials

> Module 13 — Biopharma and Clinical Development
> Topic 11 of 13
> Estimated study time: 5 hours
> Prerequisites: Topics 01, 02, 04, 07, and 10
> Version 2026.3

## 1. Why this topic exists

A Phase III trial is an expensive way to answer a question that is often narrower than the one the
protocol states. The sponsor wants to know whether a drug changes the course of a disease. The trial
answers a different question: whether the average difference between two randomised groups, measured
on one endpoint at one time point, is larger than sampling error would produce. Everything else —
who responds, how the disease would have progressed without treatment, what a different population
would have shown — is left outside the randomisation.

The digital twin is an attempt to bring that outside information back in. A patient-level digital twin
is a model that takes a patient's baseline characteristics and produces a prediction of that
patient's outcome under a specified condition, usually the control condition. Build enough of them
and you have a virtual cohort. Use that cohort in the design or analysis of a trial and you have an
in silico trial, or at least an in silico component of one.

The commercial pressure behind this is real and quantifiable. A sponsor running a 380-patient Phase
III trial at an assumed $85,000 per patient spends $32.3 million on patient-related costs alone.
If a validated prognostic model could cut the required sample by 27 percent, the saving is 103
patients, or $8.8 million, before the cost of building and validating the model. For a mid-sized
biotechnology company, that difference decides whether a trial happens at all.

The scientific pressure is equally real. Some diseases cannot be studied with a large placebo arm.
Amyotrophic lateral sclerosis progresses predictably enough that a long placebo exposure is hard to
justify. Rare paediatric diseases may have twenty eligible patients worldwide. In these settings the
question is not whether a virtual cohort is as good as a randomised control arm; it is whether a
weaker but defensible design is better than no trial.

### 1.1 The failure pattern this topic is designed to prevent

1. A sponsor builds a prognostic model on historical control data and reports an R-squared of 0.42.
2. The model is used to size a Phase III trial at 40 percent fewer patients.
3. The trial reads out negative, and the sponsor concludes the drug failed.
4. A post-hoc review finds that the 0.42 came from within-sample fit, that the external validation
   R-squared was 0.16, and that the trial was therefore underpowered from the start.
5. The programme is discontinued, and the model is blamed for a decision it never supported.

Step 4 is the analytical failure. The number that matters for sample size is the out-of-sample
partial correlation between the prognostic score and the outcome, measured in a population like the
one about to be enrolled. An in-sample R-squared is a statement about the model's flexibility, not
about its predictive value. This topic is largely about the distance between those two numbers.

### 1.2 Three different products are called a digital twin

The word is used for at least three things that carry different evidence requirements. Keeping them
apart is the single most useful discipline in this topic.

| What is built | What it predicts | Typical use | Evidence standard |
|---|---|---|---|
| Mechanistic model | A physiological or pharmacological trajectory from first principles | Dose selection, safety margin, device performance | Verification, calibration, validation against bench and animal data |
| Statistical prognostic model | A control outcome from baseline covariates, learned from historical patients | Covariate adjustment, sample size, enrichment | Out-of-sample prediction error, calibration, prospective validation |
| Generative virtual cohort | A synthetic population with joint distributions of covariates and outcomes | Trial design, sensitivity analysis, external control | All of the above plus population representativeness |

A mechanistic model that predicts drug concentration from a physiologically based pharmacokinetic
(PBPK) structure is not the same object as a gradient-boosted model that predicts a clinical endpoint
from a hundred baseline variables, and neither is the same as a generative model that samples whole
synthetic patients. They share a vocabulary and almost nothing else. A team that says "we have a
digital twin" has told you the marketing category, not the evidence.

### 1.3 The distinction that carries the regulatory weight

There is a difference between a simulation that supports a design decision and a simulation that
supports a regulatory claim. The first is internal. It changes how many patients you enrol or which
dose you take forward, and if it is wrong you lose money. The second is external. It changes what a
regulator will accept as evidence of efficacy or safety, and if it is wrong a product is approved on
a false premise.

The first is governed by professional judgement. The second requires a credibility argument a reviewer
can audit: a defined context of use, a model risk assessment, a verification and validation record, an
uncertainty quantification, and a statement of what the model would have to get wrong for the
conclusion to reverse. They are different regulatory objects, not points on one scale.

> **Note:** This topic is educational. It describes regulatory frameworks as published at the access
> date and does not constitute legal, regulatory, or medical advice. Any submission strategy requires
> advice from qualified regulatory professionals in the relevant jurisdiction.

### 1.4 Who is blocked when the evidence is weak

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Trial statistician | A defensible sample size | Depends on a model built by another team |
| Clinical team | A feasible design in a rare disease | Cannot enlarge the eligible population |
| Regulator | A reviewable basis for accepting fewer patients | Receives a model without its validation record |
| Patient | Access to a trial that did not happen | Has no view of the design trade-off |
| Sponsor board | Capital allocated to the right programme | Sees a single R-squared number |
| Data engineer | A stable model in production | Owns neither the clinical question nor the decision |

The statistician is the pivotal role. A digital twin enters a trial through the statistical analysis
plan, and the statistician is the person who must certify that the plan controls type I error. If the
statistician cannot explain the model's provenance, the model does not enter the plan, regardless of
its performance on a benchmark.

### 1.5 The industry question

> Given a model that predicts control outcomes, what is the largest claim it can support, and what
> evidence would a reviewer need before accepting that claim?

The question has two parts because the answer is not a property of the model in isolation. The same
model can support a design decision and cannot support a label claim. The claim determines the
evidence, and the evidence determines whether the model is ready.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a mechanistic, a statistical, and a generative twin, and state the evidence each
   requires before it can enter a trial protocol.
2. **Write** a context-of-use statement that names the decision, the population, the outcome, the
   timing, and the consequence of error.
3. **Compute** the sample size and power of a two-arm continuous-endpoint trial with and without a
   prognostic covariate, including a deflation factor for validation uncertainty.
4. **Diagnose** an optimistic prognostic model by comparing in-sample and out-of-sample performance
   and by checking calibration slope on a held-out cohort.
5. **Explain** why adjustment for a baseline prognostic score preserves type I error while replacing
   a control arm with synthetic patients does not.
6. **Apply** the ASME V&V 40 risk-informed credibility logic to a trial-level model and identify the
   credibility evidence that the context of use demands.
7. **Critique** a twin-augmented trial design for leakage, cohort drift, post-randomisation inputs,
   and unstated assumptions.
8. **Choose** between a design use, an analysis use, and a substitution use of a twin, and justify
   the choice on evidence grounds rather than on expected savings.

## 3. The core mental model

### 3.1 The information flow

```text
   Historical control data                     Trial population
   (completed trials, registries,      ┌──>    (screened, consented,
    claims, EHR)                      │        randomised)
          │                           │              │
          v                           │              v
   ┌──────────────────┐               │      ┌──────────────────┐
   │ Model development │               │      │ Baseline capture  │
   │  - eligibility    │               │      │  - same variables │
   │  - feature set    │               │      │  - same units     │
   │  - model class    │               │      │  - same timing    │
   └────────┬─────────┘               │      └────────┬─────────┘
            │                         │               │
            v                         │               v
   ┌──────────────────┐               │      ┌──────────────────┐
   │ Internal validation│             │      │ Prognostic score  │
   │  - held-out split │               │      │  f(x_i) per patient│
   │  - cross-validation│              │      └────────┬─────────┘
   └────────┬─────────┘               │               │
            │                         │               │
            v                         │               │
   ┌──────────────────┐               │               │
   │ External validation│──────────────┘               │
   │  - different site │                               │
   │  - different era  │                               │
   │  - calibration    │                               │
   └────────┬─────────┘                               │
            │                                         │
            v                                         v
   ┌──────────────────────────────────────────────────────────┐
   │                    Context of use                         │
   │  design only  |  covariate adjustment  |  substitution     │
   └────────────┬───────────────────┬──────────────┬───────────┘
                │                   │              │
                v                   v              v
        Sample size /       Pre-specified      External control
        enrichment /        ANCOVA in SAP      arm or synthetic
        dose choice                             control
                │                   │              │
                v                   v              v
        Internal decision    Type I error        Bias assessment
        (no regulatory       preserved by       and sensitivity
         claim)              randomisation      analysis required
```

The diagram has one structural feature worth memorising: the model is developed on data that are
separate from the trial, and it enters the trial only as a function evaluated at baseline. Every
failure mode in Section 10 is a violation of one of those two separations. Data leakage violates the
first. Post-randomisation inputs violate the second. Substitution violates both, because the
synthetic patients stand in for the trial data rather than being evaluated against it.

### 3.2 The two arithmetic facts

Almost everything quantitative in this topic follows from two facts.

**Fact one: variance reduction is proportional to the squared partial correlation.** If a prognostic
score explains a fraction R-squared of the residual variance of the outcome after treatment is
accounted for, the variance of the treatment-effect estimate falls by that fraction. The standard
error falls by the square root of one minus R-squared, and the sample size required for fixed power
falls by one minus R-squared.

```text
   SE_ratio      = sqrt(1 - R^2)
   N_required    = N_unadjusted x (1 - R^2)
   SE reduction  = 1 - sqrt(1 - R^2)

   R^2 = 0.10  ->  SE -5.1%   ->  N -10%
   R^2 = 0.20  ->  SE -10.6%  ->  N -20%
   R^2 = 0.30  ->  SE -16.3%  ->  N -30%
   R^2 = 0.50  ->  SE -29.3%  ->  N -50%
```

**Fact two: randomisation makes the score independent of treatment.** Because the score is computed
from baseline data before assignment, it is independent of the randomised arm in expectation. An
analysis that adjusts for an independent baseline covariate cannot bias the treatment estimate in
expectation, whatever the covariate's predictive value. A useless score adds a degree of freedom and
costs a small amount of precision. A useful score removes variance. Neither outcome inflates the
type I error rate, provided the adjustment is pre-specified and the score is genuinely baseline.

Fact two is why the covariate-adjustment route is defensible and the substitution route is not. When
the synthetic patients replace real controls, the comparison is no longer protected by
randomisation. The synthetic arm differs from the treated arm in every way that the model failed to
capture, and no amount of model performance removes the possibility that the difference is
systematic.

### 3.3 Why the R-squared that matters is out of sample

A model's in-sample R-squared measures how much of the variance the fitted function can explain in
the data it was fitted to. It rises with model flexibility and with the number of covariates, and it
can be made arbitrarily large by adding parameters. The quantity in the sample-size formula is the
correlation between the score and the outcome in a new, independent sample from the target
population. Call it the out-of-sample R-squared.

| Quantity | What it measures | Typical value | Use |
|---|---|---|---|
| In-sample R-squared | Fit to the development data | 0.35 to 0.50 | Never use for sizing |
| Cross-validated R-squared | Fit to held-out folds of the same data | 0.20 to 0.35 | Model selection |
| External R-squared | Fit to a different cohort, site, or era | 0.10 to 0.25 | Sample size, with deflation |
| Partial R-squared given treatment | Increment over treatment alone | Slightly below external R-squared | The number in the formula |

The gap between rows one and three is the single most common source of over-optimistic design. In
the published reanalysis of a Phase III Alzheimer's trial discussed in Section 9, the reported
sample-size reduction was in the region of 20 to 23 percent, which corresponds to an R-squared near
0.20 to 0.23 rather than the higher values the model achieved in development.

### 3.4 The credibility ladder

Borrowing the logic of ASME V&V 40-2018, credibility is not a property of a model. It is a property
of a model in a context of use. The same model is credible for a go/no-go decision and not credible
for a label claim, because the consequence of being wrong differs.

```text
   Model      Label change,                    Regulatory
   influence  withdrawal                       efficacy claim
      ^            |                                |
      |            |     credibility evidence       |
      |            |     scales with both axes      |
      |       Internal design               Primary endpoint
      |       decision                      adjustment
      +-------------------------------------------------->
                   Consequence of a wrong decision
```

The vertical axis is how much the decision depends on the model. The horizontal axis is how bad a
wrong decision is. A model that determines the primary analysis of a registrational trial sits in
the top right, and no amount of benchmark performance substitutes for a prospective validation and a
pre-specified analysis plan. A model that helps choose between two doses in an internal governance
meeting sits in the bottom left, and a well-documented internal validation is enough.

### 3.5 What the twin is not

Three misconceptions recur in vendor material and in internal proposals.

First, the twin is not a substitute for a trial when the endpoint is subjective or the effect is
small. A model trained on historical controls encodes the standard of care of the era in which those
controls were treated. If the standard of care has changed, the model predicts a counterfactual that
no longer exists.

Second, the twin is not a source of causal evidence about the drug. It predicts the control
trajectory. The treatment effect is still estimated from randomised patients. A model that predicts
both arms is making a causal claim that its training data cannot support unless the data include
randomised comparisons of the drug, which would make the trial redundant.

Third, a high area under the receiver operating characteristic curve (AUC) is not evidence of
usefulness for sample size. AUC measures discrimination across the whole range of risk. Sample size
depends on the squared correlation with the continuous outcome in the enrolled population. A model
can have an AUC of 0.78 and add almost nothing to a trial that already adjusts for age and baseline
severity, because the information it adds is already in the covariates.

## 4. Definitions and vocabulary

The table defines each term as it is used in this file. Where a term has a regulatory meaning, the
definition follows the published source and the source is named. Where a term is used loosely in
industry, the common misuse is stated.

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Patient-level digital twin | A model that predicts an individual patient's outcome trajectory under a specified condition from that patient's baseline data | Used for any predictive model, including ones that predict nothing patient-specific | Model |
| Virtual patient | A synthetic individual generated by a model, with covariates and outcomes drawn from a joint distribution | Used interchangeably with digital twin; a virtual patient need not correspond to a real person | Simulated record |
| Virtual cohort | A set of virtual patients used as a comparison group | Presented as if it were a randomised control arm | Simulated sample |
| In silico trial | A trial in which some or all patients, or some part of the analysis, are simulated | Used for any simulation, including a single dose-response curve | Study |
| Mechanistic model | A model whose structure encodes physiological, pharmacological, or physical mechanisms | Used for any model with a biological narrative | Model |
| PBPK model | Physiologically based pharmacokinetic model: compartments representing organs, linked by blood flow and partition coefficients | Treated as a clinical outcome model | Model |
| QSP model | Quantitative systems pharmacology model: mechanistic model of drug and disease biology at cell and pathway level | Treated as validated for a new indication without re-validation | Model |
| Statistical twin | A model learned from data that predicts a control outcome from baseline covariates | Assumed to be causal because it predicts well | Model |
| Generative model | A model that learns a joint distribution and can sample new records | Assumed to reproduce the target population without a representativeness check | Model |
| Conditional variational autoencoder (cVAE) | A generative model that samples outcomes conditional on covariates | Assumed to preserve marginal distributions it was never trained on | Model |
| Diffusion model | A generative model that learns to reverse a noising process | Assumed to be calibrated without a check | Model |
| Neural ordinary differential equation (neural ODE) | A model that represents a trajectory as the solution of a learned differential equation | Assumed to extrapolate beyond the observed time range | Model |
| Gaussian process | A distribution over functions defined by a kernel, giving predictive mean and variance | Assumed to scale to large cohorts without approximation | Model |
| Counterfactual | The outcome a patient would have had under a condition they did not experience | Confused with a prediction; a counterfactual is unobservable and model-dependent | Concept |
| Prognostic score | A scalar computed from baseline covariates that predicts outcome under control | Confused with a predictive biomarker of treatment effect | Score |
| Prognostic versus predictive | Prognostic: predicts outcome regardless of treatment. Predictive: predicts differential treatment effect | Used interchangeably, which changes the design | Concept |
| PROCOVA | Prognostic Covariate Adjustment: a pre-specified ANCOVA using an externally trained prognostic score | Treated as a licence to shrink any trial by any amount | Method |
| ANCOVA | Analysis of covariance: linear model with treatment and baseline covariates | Applied without checking linearity or interaction | Method |
| Estimand | The precise description of the quantity to be estimated, including population, variable, treatment, intercurrent events, and summary | Treated as a synonym for endpoint | ICH E9(R1) |
| Intercurrent event | An event after treatment initiation that affects the interpretation of the outcome | Ignored in the twin analysis, which then does not match the estimand | ICH E9(R1) |
| Deflation factor | A multiplier applied to the estimated R-squared to account for validation uncertainty | Omitted, producing optimistic power | Dimensionless, 0 to 1 |
| Partial correlation | Correlation between score and outcome after removing the effect of other covariates, including treatment | Replaced by marginal correlation, which overstates the gain | Dimensionless |
| Calibration | Agreement between predicted and observed outcome levels | Confused with discrimination | Slope and intercept |
| Discrimination | Ability to rank patients by outcome | Confused with calibration | AUC, c-statistic |
| Calibration slope | Slope of observed on predicted in a validation sample; 1 is perfect | Reported without its confidence interval | Dimensionless |
| Verification | Confirmation that the model is implemented as intended | Confused with validation | Process |
| Validation | Confirmation that the model represents the intended reality for a context of use | Confused with verification | Process |
| Context of use | The specific decision the model informs, with population, outcome, timing, and consequence | Stated so broadly that no evidence could satisfy it | Statement |
| Model risk | The combination of the model's influence on a decision and the consequence of an incorrect decision | Reduced to a single severity score | ASME V&V 40 |
| Credibility | The trust that the model is adequate for a context of use, supported by evidence | Treated as a property of the model alone | ASME V&V 40 |
| Uncertainty quantification (UQ) | The characterisation and propagation of uncertainty in inputs, structure, and parameters | Reported as a single interval without its sources | Analysis |
| Data leakage | Use of information in model development that would not be available at prediction time | Detected only by accident | Defect |
| Cohort drift | Change in the patient population or standard of care between development and use | Assumed away | Defect |
| External control arm | A comparison group drawn from outside the trial, usually historical or registry data | Described as a control arm without the word external | Design |
| Synthetic control arm | A comparison group generated by a model rather than observed | Presented as data | Design |
| Positivity | The assumption that every patient has a non-zero probability of receiving each treatment | Ignored when the comparator is a different era | Assumption |
| Effective sample size | The number of real patients to which a synthetic cohort is equivalent in information | Computed as if one synthetic patient equalled one real patient | Count |
| Type I error rate | The probability of rejecting a true null hypothesis | Quoted without the design that controls it | Probability |
| Power | The probability of rejecting a false null hypothesis at a given effect size | Quoted without the effect size and variance | Probability |
| Twin-augmented trial | A randomised trial whose analysis uses a prognostic score derived from a twin | Used for trials where the twin replaced patients | Design |

## 5. The mechanics, step by step

This procedure takes a model from a decision question to a defensible use in a trial. It is written
for the statistician and data scientist who must produce the evidence, and for the clinical lead who
must decide whether to rely on it. The order matters: the context of use determines the model class,
which determines the validation design, which determines the number that enters the sample-size
calculation. Teams that start with the model reverse the dependency and end up validating the wrong
thing.

### Step 1. Write the decision question and the context of use

**Input:** a proposed trial design, a clinical development plan, and a statement of what the sponsor
wants to do differently if the model works.

**Output:** a context-of-use statement of no more than 150 words naming six elements: the decision,
the population, the outcome, the time horizon, the comparator, and the consequence of an incorrect
decision.

**Rationale:** The context of use is the anchor for every later choice. Without it, "is the model
good enough" has no answer, because adequacy is relative to a decision. The consequence element is
the one teams omit, and it is the one that determines the credibility evidence. A model that informs
a $2 million internal dose decision and a model that informs a $300 million registrational programme
require different evidence even if the mathematics is identical.

**Example:** "To decide whether the Phase III trial in relapsing-remitting multiple sclerosis can be
sized at 280 rather than 380 patients by adjusting for a baseline prognostic score predicting
12-month confirmed disability progression under standard care. An incorrect decision would either
underpower the trial or add 100 patients and roughly $8.5 million in cost; it would not by itself
support a label claim."

**Edge case:** the sponsor wants the model to support a label claim and a design decision at the same
time. Split the context of use into two statements with two evidence packages. A single statement
that spans both will be assessed at the higher standard and will fail on evidence gathered for the
lower one.

### Step 2. Choose the role of the twin

**Input:** the context-of-use statement and the available data.

**Output:** one of five roles, with the evidence obligation that follows from it.

**Rationale:** The role determines whether randomisation protects the analysis. In the first two
roles it does. In the third it protects only part of the comparison, and in the fourth it does not
apply at all. Teams that blur the roles produce designs in which nobody can say what controls the
type I error rate.

| Role | What the model does | What protects the inference | Minimum evidence | Typical regulatory posture |
|---|---|---|---|---|
| Design only | Informs sample size, enrichment, or dose choice; not used in the analysis | Nothing needs to; the trial analysis is conventional | Internal validation, documented | No submission needed; describe in protocol |
| Baseline covariate | Enters a pre-specified ANCOVA as a prognostic score | Randomisation plus pre-specification | External validation, calibration, locked model, analysis plan | Acceptable in principle; EMA qualified the method in 2022 |
| Partial substitution | Replaces or augments part of the control arm | Randomisation of the remaining patients plus a bias assessment | All of the above plus comparability analysis and sensitivity analyses | Case-by-case; no general acceptance |
| Full substitution | The control arm is synthetic | Nothing external; the model must carry the inference | A full causal credibility argument, including unmeasured confounding | Not accepted as pivotal evidence on its own |
| Mechanistic support | Explains dose or safety margin alongside empirical data | The empirical data; the model is supportive | Verification and validation for the mechanism | Accepted as supportive in some contexts |

The table has five rows because the middle of the spectrum is where most proposals actually sit. A
design that keeps 70 percent of the control arm and replaces 30 percent is not a randomised trial
with a covariate; it is a partial substitution, and it needs the comparability analysis.

> **Note:** The regulatory posture column describes published positions at the access date. It is
> not a prediction of what any agency will accept for a specific product.

### Step 3. Assemble and harmonise the development data

**Input:** historical control records from completed trials, registries, claims, or electronic health
records (EHR), plus the protocol of the planned trial.

**Output:** an analysis-ready development dataset in which every variable used by the model has the
same definition, unit, and timing as it will have in the trial.

**Rationale:** Most twin failures begin here. A model trained on a variable recorded at screening and
applied to a variable recorded at randomisation is not the same model. A model trained on a
population that excluded patients with renal impairment and applied to a trial that admits them is
extrapolating outside its support.

The harmonisation work has four parts:

1. **Variable mapping.** Map each source variable to the trial's case report form (CRF) variable.
   Record the mapping in a data dictionary with the source system, the source code, and the
   transformation. Do not map a derived variable to a collected variable without stating the
   derivation.
2. **Unit and scale alignment.** Convert laboratory values to a single unit, and record the conversion
   factor. A model trained on creatinine in micromoles per litre and applied in milligrams per
   decilitre will produce nonsense without an error, because both are plausible numbers.
3. **Timing alignment.** Record the day relative to a reference date, and require the same reference
   date in the trial. A baseline score computed from a value taken at day minus 14 is not the same
   score when the trial takes it at day minus 1.
4. **Eligibility alignment.** Compare the development cohort's eligibility criteria with the trial's.
   Quantify the overlap, for example by computing the proportion of development patients who would
   have met the trial's criteria. If the proportion is below about 80 percent, the model is being
   applied to a materially different population.

**Edge case:** the development data include patients treated after the trial's standard-of-care
date. Either restrict the development cohort to the era that matches the trial, or include era as a
covariate and validate separately by era. Mixing eras without a term for era produces a model that
predicts an average of practices that never coexisted.

### Step 4. Specify the estimand and the outcome definition

**Input:** the trial protocol and the context of use.

**Output:** an estimand statement following ICH E9(R1) with population, variable, treatment
conditions, intercurrent events and their strategy, and population-level summary, plus the model's
prediction target expressed in the same terms.

**Rationale:** The model predicts an outcome under a condition. The trial estimates a treatment
effect under an estimand. If the two definitions differ, the covariate is adjusting for a different
quantity than the one being analysed, and the variance reduction is not the one computed.

The commonest mismatch is the handling of intercurrent events. A model trained on patients who
completed twelve months of follow-up predicts the twelve-month completer outcome. A trial using a
treatment-policy strategy analyses all randomised patients regardless of discontinuation. The
prognostic score is then predicting a mixture that the analysis does not estimate.

**Output check:** state in one sentence what the model predicts, for whom, at what time, under what
treatment condition, and with what handling of discontinuation. If that sentence cannot be written,
the model is not ready for the analysis.

### Step 5. Select the model class

**Input:** the prediction target, the development sample size, the number of candidate covariates,
and the intended regulatory role.

**Output:** a model class with a documented justification, and a pre-specified procedure for any
hyperparameter tuning.

**Rationale:** Model class determines the validation burden and the interpretability available to a
reviewer. A deep sequence model may fit better and will be harder to defend than a penalised
regression on fifteen pre-specified variables, particularly when the development sample is small.
The choice should follow the sample size and the claim, not the reverse.

| Situation | Reasonable class | Why | Main risk |
|---|---|---|---|
| Continuous outcome, n below 500, few covariates | Penalised linear regression or a pre-specified score | Stable, auditable, closed-form variance | Misses non-linearity |
| Continuous outcome, n from 500 to 5,000, many covariates | Gradient boosting or random forest with pre-specified tuning | Captures interactions | Overfits without nested validation |
| Longitudinal trajectory, irregular visits | Gaussian process or mixed-effects model | Handles unequal follow-up | Kernel choice drives extrapolation |
| Whole-trajectory generation | Conditional variational autoencoder, diffusion model, or neural ODE | Samples realistic paths | Hard to calibrate, hard to audit |
| Mechanistic dose or safety question | PBPK or QSP | Encodes known biology | Parameters not identifiable from available data |
| Very small development sample, strong prior knowledge | Bayesian hierarchical model with informative priors | Borrows strength explicitly | Prior sensitivity must be shown |

### Step 6. Separate development from evaluation

**Input:** the harmonised dataset.

**Output:** a development set, a held-out internal validation set, and at least one external
validation set that is separate by site, era, or data source.

**Rationale:** Every estimate of performance is optimistic to the extent that the evaluation data
influenced the model. The separation must be temporal and institutional, not merely random. A random
split within one trial shares the same investigators, the same era, and the same standard of care,
so it measures reproducibility rather than transportability.

The separation rules that matter:

- The trial's own patients never enter development, including through an unblinded interim analysis.
- Feature selection uses the development set only. Selecting variables on the full dataset and then
  splitting leaks information and inflates every subsequent estimate.
- Hyperparameters are tuned inside the development set with nested cross-validation.
- The external set is touched once, after the model is frozen. Repeated looks at the external set
  convert it into a tuning set.
- The final model is refitted on development plus internal validation only if the refit is
  pre-specified; otherwise the frozen model is used.

**Edge case:** the only available external set is small. A 200-patient external cohort gives an
R-squared estimate with a wide confidence interval, which is why the deflation factor in Step 9
exists. A wide interval is information, not a reason to skip the check.

### Step 7. Calibrate the model

**Input:** the fitted model and a calibration dataset.

**Output:** a calibration intercept and slope, a calibration plot by risk decile, and a decision on
whether recalibration is needed.

**Rationale:** Calibration is agreement between predicted and observed values. A model can rank
patients well and still predict the wrong level. For a covariate adjustment the level matters less
than the correlation, but a miscalibrated model is evidence that the development and target
populations differ, which undermines the correlation estimate too.

| Calibration finding | Interpretation | Action |
|---|---|---|
| Slope near 1, intercept near 0 | Predictions transfer | Proceed |
| Slope below 1 | Predictions too extreme; overfitting or population shift | Recalibrate on external data or deflate R-squared |
| Slope above 1 | Predictions too conservative | Check for range restriction in the calibration set |
| Intercept away from 0, slope near 1 | Systematic shift in outcome level | Check standard-of-care change; consider recalibration |
| Slope unstable across deciles | Non-linear miscalibration | Refit with the misspecified region represented |

Recalibration must be pre-specified and performed on data that are not the trial data. Recalibrating
on the trial's own blinded data is acceptable only if the recalibration uses no outcome information
and the procedure is documented before unblinding.

### Step 8. Validate internally and externally

**Input:** the frozen model and the validation sets.

**Output:** a validation report with discrimination, calibration, and the out-of-sample R-squared,
each with a confidence interval, stratified by site, era, and key subgroups.

**Rationale:** Validation is the evidence that the model represents reality in the context of use.
It is not a single number. A model that performs well overall and poorly in the elderly has a
subgroup problem that matters if the trial enrols elderly patients. A model validated in one country
and applied in another has a transportability problem that no overall statistic reveals.

The report should contain, at minimum:

- the out-of-sample R-squared with a confidence interval, computed on the scale used in the
  sample-size formula;
- the calibration slope and intercept with intervals;
- discrimination, reported for completeness but not used for sizing;
- performance by prespecified subgroups, including age, sex, baseline severity, and region;
- a comparison against a simple baseline model using age and the single strongest clinical predictor,
  because the incremental value over that baseline is what the trial will actually gain.

The last item is the one that most often changes the decision. A complex model with an R-squared of
0.28 may add only 0.04 over a two-variable regression that any statistician could specify, and the
regulatory and operational cost of the complex model is not justified by 0.04.

### Step 9. Quantify and deflate the expected R-squared

**Input:** the external validation R-squared, its confidence interval, and the number and independence
of the validation cohorts.

**Output:** a single deflated R-squared for use in design, with the deflation rule stated.

**Rationale:** The point estimate from one external cohort is itself uncertain, and the cohort was
not a random sample of the trial's target population. Using the point estimate produces a design
that is powered at 50 percent conditional on the true value being at the estimate. Deflation is a
crude but transparent correction.

A workable rule, similar in spirit to the sponsor handbook summarised in the EMA qualification
opinion:

| Validation evidence | Deflation factor applied to R-squared |
|---|---|
| Two or more independent external validations with consistent results | 0.95 |
| One independent external validation | 0.90 |
| External validation on a related but different population | 0.75 |
| Internal validation only | 0.50 or lower; not eligible for a regulatory-facing role |
| Development and validation on the same data | Not eligible for any sizing claim |

**Arithmetic check:** if the external R-squared is 0.30 and the factor is 0.90, the design R-squared
is 0.27, and the sample size multiplier is 0.73. A trial that needed 380 patients unadjusted needs
278. The difference between using 0.30 and 0.27 is 12 patients, which is small; the difference
between 0.30 and an in-sample 0.42 is 80 patients, which is not.

### Step 10. Size the trial

**Input:** the deflated R-squared, the target difference, the outcome standard deviation, the
allocation ratio, the two-sided alpha, and the target power.

**Output:** the number of evaluable patients per arm, the total, and the inflation for dropout.

**Rationale:** The sample-size formula for a two-arm continuous outcome with a prognostic covariate
is the standard formula multiplied by one minus the design R-squared.

```text
   n per arm = 2 x (z_(1-alpha/2) + z_(1-beta))^2 x sigma^2 / Delta^2 x (1 - R^2_design)

   alpha = 0.05 two-sided:  z = 1.959964
   power = 0.90:            z = 1.281552
   sum = 3.241516, squared = 10.50743
```

Three checks belong in the same document as the calculation:

1. The safety database is not reduced by the same factor. A trial sized for efficacy with 278
   patients still exposes 278 patients to the drug, and the safety evaluation may require a larger
   exposure than the efficacy comparison requires. Reducing the sample for efficacy and then
   claiming the reduced exposure is adequate for safety is a common and serious error.
2. The subgroup analyses are not powered by the reduced sample. A trial sized for the main effect
   will be underpowered for every interaction it plans.
3. The dropout inflation is applied after the variance reduction, not before, and the dropout rate
   is estimated for the same population and duration.

**Edge case:** the allocation is not 1:1. The variance reduction applies to the residual variance,
so the formula generalises, but the arithmetic must be redone rather than scaled. A 2:1 allocation
gains less from a prognostic covariate than a 1:1 allocation at the same total sample, because the
smaller arm dominates the variance.

### Step 11. Pre-specify the analysis and lock the model

**Input:** the frozen model, the validation report, and the trial protocol.

**Output:** a statistical analysis plan section that names the model version, the score computation,
the ANCOVA specification, the primary estimand, the sensitivity analyses, and the handling of
protocol deviations.

**Rationale:** The protection against bias in the covariate role comes from pre-specification and
randomisation together. A model chosen after unblinding, or a score recomputed with a different
version, is no longer a baseline covariate in the sense that preserves the guarantee.

The lock should include:

- a cryptographic hash of the model artefact and of the scoring code;
- the exact variable list, transformations, unit conversions, and development dataset version;
- the ANCOVA formula, including whether treatment-by-score interaction is included (usually it is
  not, unless a differential effect is hypothesised);
- the handling of missing baseline covariates, pre-specified, with a sensitivity analysis;
- the sensitivity analysis without the covariate, which shows how much the conclusion depends on it.

**Edge case:** a baseline variable is missing for some randomised patients. Mean imputation with a
missing indicator is common and is not always wrong, but it changes the score's distribution. The
analysis plan should state the rule and the expected proportion missing, and the sensitivity
analysis should repeat the primary analysis without the score.

### Step 12. Run blinded checks and analyse

**Input:** the locked model, the randomised trial data, and the analysis plan.

**Output:** the primary estimate with its confidence interval, the unadjusted estimate for
comparison, and the pre-specified sensitivity analyses.

**Rationale:** Two checks belong before unblinding and are cheap. First, verify that the score is
balanced across arms; under randomisation the standardised mean difference should be close to zero,
and a large value indicates either a scoring error or a post-randomisation input. Second, verify
that the score's distribution in the trial matches the development population; a shift indicates
that the model is being applied outside its support.

After unblinding, the primary comparison is the pre-specified ANCOVA. The unadjusted estimate is
reported alongside, not to choose between them, but to show the variance reduction actually achieved.
If the observed reduction is far from the design assumption, the trial's power is not what the
protocol claimed, and that fact belongs in the clinical study report.

### Step 13. Assess credibility for the intended claim

**Input:** the context of use, the model risk assessment, the verification and validation record, and
the uncertainty analysis.

**Output:** a credibility assessment with an explicit statement of what the evidence supports and
what it does not.

**Rationale:** Credibility is an argument, not a score. The ASME V&V 40-2018 framework asks the
assessor to identify the question of interest, the context of use, the model risk, the credibility
factors, and the evidence gathered for each. The output should let a reviewer who disagrees with the
conclusion see exactly which link in the argument they are rejecting.

| Model risk | Decision consequence | Credibility evidence expected |
|---|---|---|
| Low | Internal, reversible, small cost | Internal validation, documented assumptions |
| Moderate | Internal, costly, or affects a programme milestone | External validation, calibration, sensitivity analysis |
| High | Regulatory, irreversible, or affects patient safety | Prospective validation, pre-specified analysis, independent replicate, full uncertainty quantification |
| Very high | Pivotal efficacy or safety claim | All of the above plus a qualification opinion or equivalent agency agreement |

### Step 14. Verify after the trial and update the record

**Input:** the trial results and the model's predictions.

**Output:** a post-trial verification note comparing predicted and observed control outcomes, and an
updated validation record for future use.

**Rationale:** Every trial that used a twin is a validation opportunity. Comparing the predicted
control trajectory with the observed control arm is the most direct test the model will ever
receive, and it is nearly free. Organisations that skip this step accumulate models whose
performance is asserted and never checked, and they repeat the same design error across programmes.

The note should report the observed control-arm mean against the predicted mean, the calibration
slope in the trial population, and any protocol deviation that affected scoring. It should be filed
with the model's version record, not with the trial's clinical study report, because the next
programme will look for it in the model record.

### 5.13 The governance questions that decide whether any of this survives audit

Four questions determine whether the procedure above produces a defensible artefact or a folder of
slides.

1. **Who owns the model?** A named person, not a team. The owner signs the lock, approves changes, and
   answers the reviewer.
2. **What is the change-control rule?** Any change to the variable list, the training data, or the
   model class creates a new version that requires revalidation. Silent updates are the most common
   audit finding.
3. **Where is the evidence stored?** The validation report, the lock hash, and the analysis plan
   should be retrievable together, by version, for the life of the programme.
4. **What would make you abandon the model?** A stated falsification condition, agreed before the
   trial, prevents the post-hoc rationalisation that turns a failed assumption into a reinterpreted
   success.

### 5.14 Closing rule

If the model cannot be described in a context-of-use statement, validated on data it never saw, and
locked before the first patient is randomised, it is a research prototype, not evidence.

## 6. Worked example

This example follows one trial from protocol draft to a revised design. The setting is a Phase III
trial in amyotrophic lateral sclerosis (ALS), a disease chosen because progression is measurable, the
placebo arm is ethically constrained, and pooled historical control data exist. All numbers are
constructed for teaching and are internally consistent; they are not taken from a real programme.

### 6.1 The setting

A sponsor is planning a randomised, double-blind, placebo-controlled trial of a disease-modifying
compound in ALS. The primary endpoint is the change in the ALS Functional Rating Scale-Revised
(ALSFRS-R) from baseline to 12 months, analysed as a continuous variable. A higher score means better
function, so a slowing of decline appears as a positive treatment difference.

| Design input | Value | Source of the value |
|---|---|---|
| Target difference | 3.0 ALSFRS-R points at 12 months | Clinical judgement; the smallest effect the programme would develop |
| Standard deviation of change | 9.0 points | Pooled from historical control arms |
| Two-sided alpha | 0.05 | Convention |
| Power | 90 percent | Programme standard |
| Allocation | 1:1 | Protocol |
| Cost per randomised patient | $85,000 | Internal planning assumption, all trial costs attributed per patient |
| Site count and rate | 40 sites, 1.2 patients per site per month | Feasibility assessment |

The unadjusted sample size follows the standard formula.

```text
n per arm = 2 x (1.959964 + 1.281552)^2 x 9.0^2 / 3.0^2
          = 2 x 10.50743 x 81 / 9
          = 2 x 10.50743 x 9
          = 189.13

Rounded up: 190 per arm, 380 in total.
```

At $85,000 per patient, the patient-related cost of the trial is 380 x $85,000 = $32,300,000. At 48
patients per month across 40 sites, enrolment takes 380 / 48 = 7.92 months.

### 6.2 What the sponsor proposes

The sponsor has access to 2,400 historical control patients from seven completed trials and proposes
to build a prognostic score that predicts the 12-month ALSFRS-R change from baseline characteristics.
The score would enter the primary analysis as a covariate, and the trial would be sized using the
expected variance reduction.

The context-of-use statement is written first, as Step 1 requires:

> To decide whether the Phase III trial can be sized below 380 patients by adjusting for a baseline
> prognostic score that predicts 12-month ALSFRS-R change under placebo plus standard care, in
> patients meeting the trial's eligibility criteria. An incorrect decision would underpower the trial
> or add patients and cost. The model is not intended to support a label claim, and it will not
> replace any randomised patients.

The role is therefore "baseline covariate", the second row of the table in Step 2. The evidence
obligation is external validation, calibration, a locked model, and a pre-specified analysis. Nothing
in the design requires the model to be a faithful simulation of ALS biology. It requires the score to
correlate with the control outcome in the trial population.

### 6.3 Building and validating the score

The 2,400 historical controls are split by trial, not by patient: five trials (1,800 patients) form
the development set and two trials (600 patients) form the internal validation set. Splitting by
trial prevents patients from the same investigator group appearing on both sides.

The model is a penalised linear regression on eleven pre-specified covariates: age, sex, site of
onset, symptom duration, baseline ALSFRS-R, forced vital capacity, riluzole use, body mass index, and
three interactions fixed in advance. At 1,800 patients and eleven covariates there are about 160
patients per parameter, well above the conventional threshold of ten.

An external cohort of 500 controls from a national ALS registry, treated between 2015 and 2019, is
held back and touched once, after the model is frozen. The results are below.

| Quantity | Internal validation (600 patients) | External validation (500 patients) |
|---|---|---|
| Pearson correlation, predicted versus observed | 0.583 | 0.548 |
| Squared correlation (R-squared) | 0.340 | 0.300 |
| 95 percent confidence interval for R-squared | 0.271 to 0.406 | 0.240 to 0.362 |
| Calibration slope | 0.91 | 0.94 |
| Calibration intercept | 0.4 points | 0.6 points |
| Mean absolute prediction error | 6.1 points | 7.2 points |

The external R-squared of 0.30 is the number that enters the design. It is lower than the internal
0.34, which is expected: the external cohort differs in era, in riluzole use, and in the mix of
clinical sites. The calibration slope of 0.94 with an interval from 0.86 to 1.02 is consistent with
predictions that transfer without systematic distortion.

The deflation rule from Step 9 applies. One independent external validation gives a factor of 0.90.

```text
Design R-squared = 0.90 x 0.30 = 0.27
Sample-size multiplier = 1 - 0.27 = 0.73
Required total = 380 x 0.73 = 277.4, rounded up to 278
Per arm = 139
```

The standard error of the treatment effect falls by 1 - sqrt(0.73) = 1 - 0.8544 = 0.1456, or 14.6
percent. The power at 278 patients, assuming the design R-squared is correct, is:

```text
SE = 9.0 x sqrt(2 / 139) x sqrt(0.73)
   = 9.0 x 0.11995 x 0.85440
   = 0.9224

z = 3.0 / 0.9224 = 3.252
Power = Phi(3.252 - 1.960) = Phi(1.292) = 0.902
```

So the design is powered at 90.2 percent, as intended.

### 6.4 The business case as first presented

| Item | Unadjusted design | Twin-adjusted design | Difference |
|---|---|---|---|
| Randomised patients | 380 | 278 | 102 fewer |
| Patient cost at $85,000 | $32,300,000 | $23,630,000 | $8,670,000 saved |
| Enrolment months at 48 per month | 7.92 | 5.79 | 2.13 months faster |
| Model development and validation | $0 | $3,900,000 | $3,900,000 added |
| Net financial position | baseline | $4,770,000 better | |

The committee approves, subject to a blinded data review before the first patient is randomised.

### 6.5 Diagnostic 1: the prognostic score is not balanced across arms

The blinded review computes the standardised mean difference (SMD) of the prognostic score between
the two arms, using the first 120 randomised patients who have complete baseline data. Under
randomisation the expected SMD is zero, and its standard error is approximately the square root of
1/60 plus 1/60, which is 0.183.

The observed SMD is 0.31, which is 1.7 standard errors from zero: not decisive, but large enough to
investigate. The cause is the score's largest coefficient, on baseline ALSFRS-R. For 38 of the 120
patients the value used was the "most recent" assessment before first dose, recorded between day 3
and day 7 after randomisation, because the scoring code selected the latest timestamp rather than the
row flagged as the screening visit.

The score was therefore computed partly from post-randomisation data, with two consequences. The
balance check loses its usual interpretation, because the score is not a baseline variable. And if
early treatment response affects the post-randomisation values, the covariate sits on the causal
pathway, so adjusting for it can bias the estimate rather than merely reduce variance.

The fix is to score from the screening visit, verify the SMD on the recomputed score, and lock the
corrected code; the recomputed SMD is 0.04. The lesson is not that the model was wrong, but that an
hour-long balance check detected a data-plumbing defect that would otherwise have entered the primary
analysis unremarked.

### 6.6 Diagnostic 2: the external R-squared does not reproduce

The second diagnostic compares the external R-squared with the value observed when the frozen model
is applied to the trial's own blinded control patients, at a pre-planned cut with 90 control
outcomes. This uses only the control arm, makes no between-arm comparison, and was pre-specified.

The result is an R-squared of 0.18 in the contemporaneous trial controls, against 0.30 in the
registry cohort, with a calibration slope of 0.62 (95 percent interval 0.51 to 0.73). A slope below 1
means the predictions are too extreme: patients predicted to do badly do worse than predicted, and
those predicted to do well do better. That is what overfitting looks like when the validation sample
is more homogeneous than the development sample. Two checks separate the explanations.

| Check | Result | Interpretation |
|---|---|---|
| R-squared in registry cohort, by era | 0.31 for 2015-2016, 0.19 for 2017-2019 | Performance degrades over calendar time |
| Riluzole use, development versus trial | 58 percent versus 84 percent | Standard of care has shifted |
| Calibration slope, development-era holdout | 0.89 | The model is adequate in its own era |
| Calibration slope, current-era holdout | 0.62 | The model is miscalibrated in the current era |

The explanation is cohort drift, not a coding error. The development trials ran between 2005 and
2014, when riluzole use was partial and the patient mix differed. The trial will enrol in 2027, when
riluzole is near-universal and the population is older. The model is predicting the counterfactual
of a patient who no longer exists.

### 6.7 The varied assumption and the revised decision

The design assumption of R-squared 0.27 is replaced by the contemporaneous estimate. The external
validation in the current era gives 0.18, and the trial's own blinded controls give 0.18 as well.
Applying the same deflation factor of 0.90:

```text
Design R-squared = 0.90 x 0.18 = 0.162
Multiplier = 1 - 0.162 = 0.838
Required total = 380 x 0.838 = 318.4, rounding to 159 per arm and 318 in total
```

The revised business case is materially different.

| Item | Unadjusted | Original assumption (R-squared 0.27) | Revised assumption (R-squared 0.162) |
|---|---|---|---|
| Randomised patients | 380 | 278 | 318 |
| Patient cost | $32,300,000 | $23,630,000 | $27,030,000 |
| Gross saving | — | $8,670,000 | $5,270,000 |
| Net of $3,900,000 model cost | — | $4,770,000 | $1,370,000 |
| Enrolment months | 7.92 | 5.79 | 6.63 |
| Power at the assumed R-squared | 90 percent | 90.2 percent | 90.1 percent |

The conclusion changes from "clear business case" to "marginal". A net saving of $1.37 million on a
$32 million trial is within the uncertainty of the cost assumptions, and it does not compensate for
the added operational complexity of a model lock, a scoring pipeline, and a regulatory conversation.
The committee's decision rule, agreed in advance, was that the twin would be used only if the net
saving exceeded $3 million. Under the revised assumption it does not.

Three options follow, and the choice is a judgement rather than a calculation. The sponsor can run
the unadjusted 380-patient trial and avoid the model entirely; run 318 patients with a pre-specified
recalibration on current-era data and accept a modest contribution; or rebuild on current-era controls
and re-validate, at the cost of another development cycle with unknown performance.

The point is the arithmetic, not the decision. The number that changed it was not the model's
complexity or its internal performance but the difference between 0.30 and 0.18: a well-validated
prediction against a prediction of a population that has moved on.

### 6.8 What the example teaches

Four points generalise beyond ALS.

- **Transportability dominates precision.** Moving from 0.30 to 0.27 cost 12 patients; moving from
  0.30 to 0.18 cost 42. The dominant uncertainty is cohort drift, not the precision of the estimate.
- **The balance check is cheap and finds real defects.** It caught a post-randomisation input that no
  amount of model performance reporting would have revealed.
- **The relevant validation cohort resembles the trial, not the development data.** A registry cohort
  from the same era as the trials is an internal validation in disguise.
- **The decision rule is agreed before the numbers are known.** A committee that sees only the 0.27
  case will approve; one that sees both cases and has a rule will decide defensibly either way.

## 7. Code, query, or calculation

The script below reproduces the worked example's sizing, then checks it two ways: by simulating the
trial at the reduced sample size, and by demonstrating how in-sample R-squared behaves when
covariates are unrelated to the outcome. It uses only the Python standard library, so it runs in any
Python 3.8 or later environment without installation.

Assumptions: a continuous outcome, 1:1 allocation, a two-sided alpha of 0.05, a prognostic score with
a known squared correlation to the outcome, and normally distributed errors. The simulation draws
the score from a standard normal and scales the outcome so that the score explains exactly the
requested fraction of variance.

```python
"""Prognostic covariate adjustment: sizing, simulated power, and a leakage check."""
import math, random, statistics

Z = statistics.NormalDist()
Z975, Z90 = Z.inv_cdf(0.975), Z.inv_cdf(0.90)


def solve(a, b):
    """Gauss-Jordan elimination with partial pivoting."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        pv = m[c][c]
        for j in range(c, n + 1):
            m[c][j] /= pv
        for r in range(n):
            if r != c and m[r][c]:
                f = m[r][c]
                for j in range(c, n + 1):
                    m[r][j] -= f * m[c][j]
    return [m[i][n] for i in range(n)]


def inverse(a):
    n = len(a)
    cols = [solve(a, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def ols(y, X):
    """Return coefficients, standard errors, and residual variance."""
    n, k = len(X), len(X[0])
    xtx = [[sum(X[i][p] * X[i][q] for i in range(n)) for q in range(k)] for p in range(k)]
    xty = [sum(X[i][p] * y[i] for i in range(n)) for p in range(k)]
    beta = solve(xtx, xty)
    rss = sum((y[i] - sum(beta[p] * X[i][p] for p in range(k))) ** 2 for i in range(n))
    s2 = rss / (n - k)
    inv = inverse(xtx)
    return beta, [math.sqrt(max(s2 * inv[p][p], 0.0)) for p in range(k)], s2


def total_n(delta, sd, r2=0.0, power=0.90, alpha=0.05):
    """Total patients, 1:1 allocation, continuous endpoint."""
    per_arm = 2 * (Z.inv_cdf(1 - alpha / 2) + Z.inv_cdf(power)) ** 2 * sd ** 2 / delta ** 2 * (1 - r2)
    return 2 * math.ceil(per_arm)


def analytic_power(n_per_arm, delta, sd, r2=0.0, alpha=0.05):
    se = sd * math.sqrt(2.0 / n_per_arm) * math.sqrt(1 - r2)
    return Z.cdf(delta / se - Z.inv_cdf(1 - alpha / 2))


def simulate(n_per_arm, r2, delta=3.0, sd=9.0, reps=4000, seed=20261027):
    """Empirical power with and without adjustment for the prognostic score."""
    rng = random.Random(seed)
    b, s_e = sd * math.sqrt(r2), sd * math.sqrt(1 - r2)
    reject_un = reject_adj = 0
    estimates = []
    for _ in range(reps):
        rows = []
        for arm in (0, 1):
            for _ in range(n_per_arm):
                g = rng.gauss(0.0, 1.0)
                rows.append((delta * arm + b * g + rng.gauss(0.0, s_e), float(arm), g))
        y = [r[0] for r in rows]
        b_un, se_un, _ = ols(y, [[1.0, r[1]] for r in rows])
        b_adj, se_adj, _ = ols(y, [[1.0, r[1], r[2]] for r in rows])
        reject_un += abs(b_un[1] / se_un[1]) > Z975
        reject_adj += abs(b_adj[1] / se_adj[1]) > Z975
        estimates.append(b_adj[1])
    return reject_un / reps, reject_adj / reps, statistics.fmean(estimates)


def r_squared(y, pred):
    mean = statistics.fmean(y)
    return 1 - sum((y[i] - pred[i]) ** 2 for i in range(len(y))) / sum((v - mean) ** 2 for v in y)


def leakage_demo(n_dev=60, n_hold=500, n_noise=20, seed=7):
    """20 covariates unrelated to the outcome, fitted on 60 patients."""
    rng = random.Random(seed)

    def make(n):
        X = [[1.0] + [rng.gauss(0.0, 1.0) for _ in range(n_noise)] for _ in range(n)]
        return X, [rng.gauss(0.0, 1.0) for _ in range(n)]

    Xd, yd = make(n_dev)
    Xh, yh = make(n_hold)
    beta, _, _ = ols(yd, Xd)
    fit = lambda X: [sum(beta[p] * X[i][p] for p in range(len(beta))) for i in range(len(X))]
    return r_squared(yd, fit(Xd)), r_squared(yh, fit(Xh)), n_noise, n_dev


def main():
    delta, sd, r2_ext, deflation = 3.0, 9.0, 0.30, 0.90
    r2_design = r2_ext * deflation
    n0, n1 = total_n(delta, sd), total_n(delta, sd, r2_design)
    print("SAMPLE SIZE (alpha 0.05 two-sided, power 0.90)")
    print(f"  unadjusted                  {n0}")
    print(f"  external R2 / deflation     {r2_ext:.2f} / {deflation:.2f}")
    print(f"  design R2                   {r2_design:.3f}")
    print(f"  adjusted                    {n1}")
    print(f"  patients saved              {n0 - n1}")
    print(f"  standard error reduction    {100 * (1 - math.sqrt(1 - r2_design)):.1f} percent")
    print("\nSENSITIVITY OF THE TOTAL TO THE DESIGN R2")
    for r2 in (0.000, 0.100, 0.162, 0.200, 0.270, 0.300):
        print(f"  R2 {r2:>5.3f}  total {total_n(delta, sd, r2):>4d}"
              f"  power at {n1} = {analytic_power(n1 // 2, delta, sd, r2):.3f}")
    p_un, p_adj, mean_est = simulate(n1 // 2, r2_design)
    print(f"\nSIMULATION AT {n1} PATIENTS ({n1 // 2} PER ARM, 4000 REPLICATES)")
    print(f"  empirical power, unadjusted {p_un:.3f}")
    print(f"  empirical power, adjusted   {p_adj:.3f}")
    print(f"  analytic power, adjusted    {analytic_power(n1 // 2, delta, sd, r2_design):.3f}")
    print(f"  mean adjusted estimate      {mean_est:.3f} (true {delta:.1f})")
    r2_in, r2_out, p, n = leakage_demo()
    print(f"\nLEAKAGE CHECK: {p} NOISE COVARIATES, {n} DEVELOPMENT PATIENTS")
    print(f"  in-sample R2                {r2_in:.3f}")
    print(f"  out-of-sample R2            {r2_out:.3f}")


if __name__ == "__main__":
    main()
```

Output on the authoring machine, Python 3.12, seed 20261027:

```text
SAMPLE SIZE (alpha 0.05 two-sided, power 0.90)
  unadjusted                  380
  external R2 / deflation     0.30 / 0.90
  design R2                   0.270
  adjusted                    278
  patients saved              102
  standard error reduction    14.6 percent

SENSITIVITY OF THE TOTAL TO THE DESIGN R2
  R2 0.000  total  380  power at 278 = 0.794
  R2 0.100  total  342  power at 278 = 0.834
  R2 0.162  total  318  power at 278 = 0.859
  R2 0.200  total  304  power at 278 = 0.874
  R2 0.270  total  278  power at 278 = 0.902
  R2 0.300  total  266  power at 278 = 0.913

SIMULATION AT 278 PATIENTS (139 PER ARM, 4000 REPLICATES)
  empirical power, unadjusted 0.786
  empirical power, adjusted   0.897
  analytic power, adjusted    0.902
  mean adjusted estimate      2.999 (true 3.0)

LEAKAGE CHECK: 20 NOISE COVARIATES, 60 DEVELOPMENT PATIENTS
  in-sample R2                0.407
  out-of-sample R2            -0.779
```

### 7.1 What the output shows

Three results carry the teaching.

First, the analytic and simulated power agree. The formula gives 0.902 and the simulation gives
0.897, a difference of 0.005 that is within simulation error at 4,000 replicates. If a design assumes
an R-squared of 0.27 and the true value is 0.27, the trial is powered as planned.

Second, the sensitivity block shows why the design R-squared must be honest. At 278 patients, power
is 0.902 if the true R-squared is 0.27, 0.859 if it is 0.162, and 0.794 if the score is useless. A
trial sized on 0.27 that delivers 0.162 is not catastrophic; a trial sized on 0.30 and delivering
0.10 has roughly 80 percent power, and the risk of a false negative is the dominant risk in the
programme.

Third, the leakage demonstration is the arithmetic behind Section 6's second diagnostic. Twenty
covariates generated independently of the outcome produce an in-sample R-squared of 0.407 on 60
patients, because the expected in-sample value under the null is approximately the number of
parameters divided by the number of observations, here 20/59 = 0.34. On 500 new patients the same
model has an R-squared of -0.779, which is worse than predicting the mean. A reported R-squared
without a statement of where it was measured is not evidence of anything.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and describes published frameworks at the access date, 27
> October 2026. It is not legal or regulatory advice, and it does not predict any agency's decision on
> a specific submission.

### 8.1 Data standards that the model depends on

A twin is only as portable as the variables it consumes. Four standards determine whether the same
variable means the same thing in development and in the trial.

| Standard | What it covers | What it constrains for a twin | What it does not constrain |
|---|---|---|---|
| CDISC SDTM | Tabulation of trial data, with domains such as demographics, laboratory, and vital signs | Variable names, controlled terminology, and the structure of the historical trial data | Derived analysis variables, which live in ADaM |
| CDISC ADaM | Analysis datasets and their metadata | The definition of the analysis population and the derived endpoint | Model inputs outside the trial, such as registry or claims data |
| OMOP CDM | Observational data mapped to a common structure, with standard vocabularies | Cohort definitions and covariate extraction from EHR and claims | Trial-specific assessments and PRO instruments |
| HL7 FHIR | Exchange of healthcare data between systems | The interface by which a score might be computed at the point of care | The statistical properties of the score |

A model trained on SDTM and applied to OMOP-mapped data requires an explicit mapping layer, and that
layer is a source of error: an SDTM laboratory value carries a controlled unit, while the same value in
claims data may carry none. The mapping belongs in the versioned model record.

### 8.2 The regulatory frameworks

| Instrument | Status | What it says | Relevance to a twin |
|---|---|---|---|
| ASME V&V 40-2018 | Consensus standard, recognised by the US Food and Drug Administration (FDA) | Credibility of a computational model should be commensurate with model risk and the context of use | The framework for arguing that a model is adequate for a decision |
| FDA guidance on computational modeling and simulation, final November 2023 | Guidance for industry | Risk-informed credibility assessment for medical device submissions; explicitly excludes standalone statistical and machine-learning models | Applies to mechanistic components, not to a prognostic score |
| FDA complex innovative trial designs guidance, 2020 | Guidance for industry | Process for discussing novel designs, including simulation-based ones, before submission | The route for a pre-specified conversation about a twin-augmented design |
| EMA qualification of novel methodologies | Formal EU procedure | A qualification opinion states whether a method is acceptable for a specified use | The 2022 opinion on prognostic covariate adjustment is the relevant precedent |
| ICH E9(R1) | International Council for Harmonisation guideline, adopted 2019 | Defines the estimand and the handling of intercurrent events | The twin's prediction target must match the estimand |
| ICH E6(R2) | Good clinical practice guideline, 2016 | Responsibilities for data integrity and computerised systems | Model versioning and audit trails fall under this |
| FDA model-informed drug development (MIDD) paired meeting programme | Process, established under PDUFA VI | A forum to discuss model-informed decisions across the product lifecycle | A venue for design-stage models, not a qualification |

Three distinctions carry most of the weight.

**Binding requirement versus convention.** A regulation that requires an expedited safety report
within fifteen days is binding. A convention that a disproportionality threshold is a lower
confidence bound above one is not. The ASME V&V 40 framework is a consensus standard, and the FDA
guidance that adopts its logic is guidance rather than regulation; a sponsor can propose an
alternative argument, but must make the argument.

**Qualified method versus qualified model.** The EMA qualification opinion addresses a method, not a
particular sponsor's model. The opinion states that the development of the prognostic model itself
was not qualified and must be assessed case by case. A team that reads the opinion as an approval of
its own model has misread it.

**Model risk versus product risk.** The credibility evidence is scaled to the consequence of a wrong
decision about the model, not to the severity of the disease. A model used to choose a dose in a
phase 2 trial for a severe disease may carry lower model risk than a model used to support a
secondary endpoint in a common condition, because the decision it informs is smaller.

### 8.3 What the standards do not settle

None of the frameworks specifies an acceptable R-squared, a minimum external validation sample size,
or a deflation factor. Those are scientific judgements that the sponsor must make and defend. The
frameworks specify the structure of the argument, not its contents. This is why two competent teams
can reach different conclusions about the same model and both be defensible, and why the written
argument matters as much as the number.

## 9. Industry practice and case evidence

### 9.1 The documented case: prognostic covariate adjustment

The most concrete public evidence is the European Medicines Agency's qualification opinion on
Prognostic Covariate Adjustment, adopted by the Committee for Medicinal Products for Human Use on 15
September 2022 and published on 20 September 2022. The opinion followed a submission by Unlearn.AI
and addresses randomised controlled trials with continuous outcomes.

| Element | What the opinion states | What it does not state |
|---|---|---|
| Method | Prognostic score adjustment is a special case of analysis of covariance and is acceptable as a primary analysis | That any particular model is qualified |
| Model development | The development step was not qualified and is assessed case by case | That a model trained on historical controls is automatically acceptable |
| Power gain | The reviewed examples showed up to about 8 percent narrower confidence intervals and up to about 15 percent sample-size reduction | That a 30 percent reduction is supported |
| Sample size | Reduction may be used only if conservative about uncertainty and consistent with other trial objectives | That the safety database may be reduced |
| Analysis | Use t-distributions, robust variance, and avoid subgroup analyses on the primary model | That a single specification fits all trials |

The underlying methods paper is Schuler, Walsh, Hall, Walsh and Fisher, published in the
International Journal of Biostatistics in 2022. It reports fractional sample-size reductions
approximately equal to the squared correlation between the score and the outcome, giving 10 to 30
percent for R-squared values of 0.1 to 0.3, and reports a reanalysis of a 402-patient Phase III
docosahexaenoic acid trial in Alzheimer's disease with a prospective power analysis showing a
sample-size reduction in the low twenties percent. The difference between that figure and the
"up to about 15 percent" in the opinion is itself instructive: the regulator summarised the evidence
more conservatively than the sponsor's paper did, which is the normal direction of travel.

### 9.2 Documented fact and inference

| Claim | Status | Basis |
|---|---|---|
| EMA qualified prognostic covariate adjustment as a method in 2022 | Documented | Published qualification opinion |
| The qualification excludes model development | Documented | Text of the opinion |
| Unlearn.AI markets a twin-augmented trial design under the PROCOVA name | Documented | Company publications and the opinion's procedural record |
| The FDA has no equivalent published qualification decision for a patient-level twin in a registrational trial | Inference from a search of published decisions at the access date | Absence of a published opinion; not a statement that none exists |
| Several sponsors are using prognostic scores in design | Inference | Conference presentations and job descriptions; not auditable |
| In silico trials are accepted for some medical devices | Documented | FDA guidance and published device examples, including the ENRICHMENT project with Dassault Systèmes |
| Digital twins will reduce pivotal trial sizes by a third | Not supported | The published evidence supports smaller gains, and only for continuous outcomes with strong prognostic scores |

The last row is the claim most often made in commercial material and the one least supported by the
public record. The published range is roughly 10 to 25 percent, in specific settings, with the
regulator's own summary at the low end.

### 9.3 Where the practice is ahead of the evidence

Three patterns appear across sponsors that have built twin programmes.

First, the model is built by a data science team and handed to a statistics function that had no role
in its development. The statistician then has to certify a model they cannot audit, and the
predictable outcome is that the model is used for design and not for analysis. This is a governance
failure rather than a scientific one, and it is the commonest reason a twin never reaches a protocol.

Second, the validation is done on a random split of the development data. The reported R-squared is
then a reproducibility measure presented as a transportability measure. The number looks respectable
and means something different from what the reader assumes.

Third, the model is evaluated on its ability to predict the mean trajectory of a cohort rather than
the outcome of an individual. A model can reproduce the average decline of a population and add
almost nothing to a trial, because the trial's power depends on explaining between-patient variance,
not on getting the average right.

None requires a bad model, only a process that never asked what the number would be used for.

## 10. Failure modes and red flags

The table lists failures that recur across twin programmes. The early warning column is the one to
use in review: it is observable before the failure reaches a protocol.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Leakage from the trial into development | R-squared in development is close to the trial's own control R-squared | Trial patients or unblinded interim data enter training | Inflated design R-squared, underpowered trial | Physical separation of data, access log, hash of training set |
| Post-randomisation inputs | Score is imbalanced across arms (standardised difference above 0.2) | Scoring code picks the latest value rather than the baseline visit | Covariate on the causal pathway; biased estimate | Blinded balance check before first patient in |
| In-sample R-squared used for sizing | The validation report has no external cohort | Fit statistic reported as if predictive | Trial sized on a number that will not reproduce | Require external validation in the design document |
| Cohort drift | Calibration slope below 0.8 in a recent cohort | Standard of care or case mix changed between eras | Prediction of a population that no longer exists | Validate by era; restrict development window |
| Calibration never checked | Only discrimination is reported | Team equates AUC with validity | Predictions systematically too extreme or too flat | Calibration slope and intercept with intervals |
| Model never locked | No version hash in the analysis plan | Model updated during trial conduct | Analysis no longer pre-specified | Lock artefact and scoring code before randomisation |
| Substitution without bias assessment | Proposal describes synthetic controls as "controls" | Model carries the comparison without randomisation | Unmeasured confounding presented as effect | Comparability analysis, negative control outcomes, sensitivity analysis |
| Twin used to shrink the safety database | Sample size reduction applied to all objectives | Efficacy variance confused with safety exposure | Inadequate characterization of rare risks | Separate sizing for efficacy and safety |
| Extrapolation beyond support | Eligibility overlap below 80 percent | Development cohort excluded the patients now enrolled | Predictions outside the model's range | Overlap analysis and restriction |
| Model shopping | Many candidates compared on the same validation set | Selection on noise | Winner's curse, optimistic performance | Pre-specify one model and one validation set |
| Outcome definition mismatch | Model predicts completer outcome; estimand is treatment policy | Discontinuation handled differently | Adjustment for a different quantity | Write model target beside the estimand |
| Useless incremental value | No comparison with a simple baseline model | Complexity rewarded without a benchmark | Cost and risk without gain | Report incremental R-squared over age and severity |
| Vendor black box | Scoring code not provided | Proprietary implementation | No audit, no reproduction, no regulatory defence | Contract for code, model artefact, and validation data |

Two patterns connect the rows. The first six are all failures of separation: between development and
evaluation data, between baseline and post-baseline information, between the model that was validated
and the model that is used. The remaining rows are failures of claim discipline: the team asks the
model to support a conclusion larger than the evidence.

## 11. Assumptions and limits

Every quantitative claim in this topic rests on assumptions. The ledger below states them, says why
they are needed, what breaks them, and how to test them. A design document that does not contain a
version of this table is incomplete.

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Historical controls represent the trial's control condition | The score is trained on them | Standard of care, background therapy, or case mix changed | Compare era, riluzole or equivalent use, and baseline severity distributions |
| The score is independent of randomised treatment | Preserves type I error under adjustment | Post-randomisation inputs or a broken randomisation | Standardised mean difference across arms, blinded |
| The score is measured without material error | Attenuation otherwise reduces the gain | Noisy or missing baseline variables | Repeat with a missing-data sensitivity analysis |
| The R-squared is constant across populations | The variance reduction is population-specific | Case mix or endpoint variance differs | External validation in the target population |
| Linear adjustment is adequate | ANCOVA assumes a linear score-outcome relation | Strong non-linearity or effect modification | Compare with a spline or stratified analysis |
| Intercurrent events are handled identically in model and analysis | Otherwise the covariate predicts a different quantity | Discontinuation rates differ between development and trial | Map the estimand to the prediction target explicitly |
| No unmeasured confounding in a substitution design | The synthetic arm must be exchangeable with the treated arm | Any unmeasured prognostic factor that differs | Negative control outcomes, E-value style sensitivity analysis |
| The development and trial cohorts are exchangeable | Transportability of the score | Eligibility, region, or era differences | Eligibility overlap and subgroup performance |
| The validation sample is independent of development | Unbiased performance estimate | Same sites, same era, or feature selection on the full data | Audit the split; require site or era separation |
| The model class is stable under retraining | The frozen model represents the class | Small development sample, unstable feature selection | Refit on bootstrap samples and compare predictions |
| The endpoint variance used in sizing is correct | The formula is sensitive to it | Variance estimated from a different population | Use the upper confidence bound of the variance |

The most fragile assumption is the first. Almost every model that fails in practice fails because the
control condition moved, not because the algorithm was wrong. The most neglected is the seventh: a
substitution design inherits every unmeasured confounder that an observational study inherits, and
the modelling effort does nothing to remove them.

## 12. Exercises

The exercises use the worked example's numbers where possible, so that answers can be checked
against Section 6 and the script in Section 7.

**Exercise 1 — Context of use (application, 20 minutes).**
Deliverable: a 150-word context-of-use statement for a twin in a disease of your choice, naming the
decision, population, outcome, horizon, comparator, and consequence of error.
Acceptance criteria: all six elements present; the consequence quantified in patients or currency;
no claim larger than a design decision unless the evidence is described.
Time: 20 minutes.

**Exercise 2 — Sizing with deflation (application, 25 minutes).**
Deliverable: the total sample size for a two-arm trial with standard deviation 9, difference 3,
90 percent power, external R-squared 0.24, and a deflation factor of 0.95, with the arithmetic shown.
Acceptance criteria: design R-squared stated; multiplier applied to the unadjusted total of 380;
result rounded consistently; standard error reduction stated.
Time: 25 minutes.

**Exercise 3 — Diagnose the validation table (critique, 30 minutes).**
Deliverable: a one-page assessment of a validation report showing internal R-squared 0.44 on 600
patients and external R-squared 0.12 on 300 patients, with a recommendation.
Acceptance criteria: the gap explained by at least two mechanisms; the design consequence computed;
a recommendation that distinguishes rejecting the model from rejecting the use.
Time: 30 minutes.

**Exercise 4 — Calibration (application, 25 minutes).**
Deliverable: an interpretation of a calibration slope of 0.55 with a 95 percent interval of 0.44 to
0.66, and the action it implies.
Acceptance criteria: direction of miscalibration stated correctly; effect on the R-squared estimate
explained; a specific recalibration or restriction proposed.
Time: 25 minutes.

**Exercise 5 — Validation plan (design, 40 minutes).**
Deliverable: a validation plan for applying an existing oncology prognostic score to a new tumour
type, including data sources, split strategy, metrics, and stopping rules.
Acceptance criteria: at least one external cohort separated by site or era; primary metric
pre-specified; sample size justified; a rule for what result would stop the programme.
Time: 40 minutes.

**Exercise 6 — Vendor claim (critique, 35 minutes).**
Deliverable: a 400-word review of a proposal claiming a 35 percent sample-size reduction from a
digital twin, with the questions you would ask before signing.
Acceptance criteria: the claim compared with the published range; the distinction between method
qualification and model qualification made; at least five due-diligence questions, including
access to scoring code and validation data.
Time: 35 minutes.

**Exercise 7 — Substitution bias (design, 45 minutes).**
Deliverable: a bias analysis plan for a design that replaces 30 percent of the control arm with
synthetic patients.
Acceptance criteria: at least three quantitative bias analyses; negative control outcomes named;
a tipping-point analysis stated; the effective sample size of the synthetic portion estimated
rather than assumed.
Time: 45 minutes.

**Exercise 8 — Model lock (application, 30 minutes).**
Deliverable: a change-control procedure for a model used in two ongoing trials, including what
constitutes a new version.
Acceptance criteria: version triggers listed; approval authority named; revalidation requirement
stated per change type; audit trail described.
Time: 30 minutes.

## 13. Self-check answers

**Exercise 1.** A complete statement names a decision ("size the trial at 280 rather than 380"), a
population (eligibility criteria), an outcome and horizon (12-month ALSFRS-R change), a comparator
(placebo plus standard care), and a consequence ("$8.5 million and 100 patients"). The common failure
is a statement that names only the outcome, which cannot determine the evidence.

**Exercise 2.** Design R-squared = 0.95 x 0.24 = 0.228. Multiplier = 0.772. Total = 380 x 0.772 =
293.4, so 294 patients, 147 per arm. Standard error reduction = 1 - sqrt(0.772) = 12.1 percent. The
saving is 86 patients, or $7.31 million at $85,000 per patient.

**Exercise 3.** The gap of 0.32 is too large for transportability alone. Two mechanisms are plausible:
overfitting, since 600 patients can fit noise, and feature selection before the split, which leaks
outcome information. A trial sized on 0.44 with a true 0.12 would be powered near 70 percent, so the
recommendation is to reject this model for sizing and rebuild with pre-specified features and a
site-separated validation.

**Exercise 4.** A slope of 0.55 means predicted differences are about 1.8 times too large. Patients
predicted to do badly do worse than predicted and those predicted to do well do better. The R-squared
measured in that cohort is therefore an overstatement of the score's usable correlation, and the
design R-squared should be taken from the recalibrated model or from a cohort where the slope is
near one. The action is recalibration on current-era data, pre-specified, or restriction of the
model to the population in which it calibrates.

**Exercise 5 rubric.** Full marks require an external cohort separated by site or era, a pre-specified
primary metric with its confidence interval method, a sample-size justification for the validation
itself, and a stopping rule stated before the analysis. Deduct marks for a random split presented as
external validation, for reporting discrimination without calibration, and for a plan that cannot
fail.

**Exercise 6.** The published range is roughly 10 to 25 percent, with the EMA opinion summarising
about 15 percent in the examples it reviewed. A 35 percent claim implies a squared correlation near
0.35 that is sustained out of sample, which is unusual. The method may be qualified while the model
is not; the questions should establish the external validation cohort, the calibration slope, the
deflation rule, access to the scoring code, the lock procedure, and who bears the regulatory risk if
the agency disagrees.

**Exercise 7 rubric.** Full marks require at least three bias analyses with explicit assumptions,
negative control outcomes that should show no effect, a tipping-point statement of how much
unmeasured confounding would reverse the conclusion, and an effective sample size for the synthetic
portion that is below its nominal count. Deduct marks for treating synthetic patients as exchangeable
with randomised controls.

**Exercise 8 rubric.** Full marks require a list of version triggers (variable list, training data,
model class, scoring code, unit conversion), a named approval authority, a rule that any trigger
requires revalidation before use in an ongoing trial, and an audit trail linking each trial analysis
to a specific version. Deduct marks for a procedure that allows "minor" updates without review.

## 14. Interview questions

**1. What is a patient-level digital twin, and what does it predict?** Strong: a model predicting an
individual's outcome under a specified condition from baseline data, usually the control trajectory;
distinguishes mechanistic, statistical, and generative forms. Weak: "simulating the patient" without
saying what is predicted or under what condition.

**2. Why does adjusting for a prognostic score preserve type I error, while replacing controls does not?**
Strong: the score is measured before randomisation and is therefore independent of arm, so adjustment
cannot bias the estimate in expectation; substitution removes the randomisation that guarantees
exchangeability. Weak: says adjustment "controls for confounders", confusing variance reduction with
confounding control.

**3. Your model has an R-squared of 0.45. What else do you need before sizing a trial?** Strong: where
it was measured, the external cohort, the calibration slope, the deflation rule, and the incremental
value over a simple baseline. Weak: asks for a bigger validation set and accepts the number.

**4. Discrimination versus calibration, and which matters here?** Strong: discrimination ranks,
calibration agrees in level; correlation drives the variance reduction, but miscalibration is evidence
that the population has shifted. Weak: treats AUC as a measure of usefulness for sample size.

**5. How would you detect data leakage?** Strong: compare in-sample, internal, and external R-squared;
audit feature selection and the split; confirm that trial patients and unblinded interim data never
entered training; check the score's balance across arms. Weak: relies on the team's assurance.

**6. A vendor claims a 35 percent sample-size reduction. What do you ask?** Strong: the external cohort
and its era, the calibration slope, the published comparison range, whether the method or the model was
qualified, and access to scoring code and validation data. Weak: negotiates the price.

**7. When would you accept a synthetic control arm as pivotal evidence?** Strong: almost never without
randomised support; requires a pre-specified causal argument, comparability analysis, negative control
outcomes, and prior agency agreement. Weak: says yes if the model performs well, confusing prediction
with causation.

**8. What would make you abandon a twin programme?** Strong: a falsification condition agreed before the
trial, such as external R-squared below a threshold or calibration slope outside a range. Weak: no
condition; the model is defended after the fact.

**9. How does the estimand interact with the model?** Strong: the prediction target must match the
estimand's population, variable, and intercurrent-event strategy; a mismatch adjusts for a different
quantity. Weak: treats the estimand as a formality unrelated to the model.

## 15. Key terms

- **ANCOVA** — analysis of covariance; the linear model used to adjust for a baseline score.
- **Calibration slope** — slope of observed on predicted outcome; 1 indicates agreement.
- **Context of use** — the decision, population, outcome, and consequence that the model informs.
- **Counterfactual** — the outcome a patient would have had under a condition not experienced.
- **Credibility** — the evidence that a model is adequate for a specific context of use.
- **Deflation factor** — multiplier on the estimated R-squared to account for validation uncertainty.
- **Digital twin** — a model predicting an individual's trajectory from that individual's data.
- **Effective sample size** — the number of real patients to which a synthetic cohort is equivalent.
- **Estimand** — the precise definition of the quantity to be estimated, per ICH E9(R1).
- **External control arm** — a comparison group drawn from outside the trial.
- **In silico trial** — a trial in which some patients or some analysis are simulated.
- **Leakage** — use of information in development that would not be available at prediction time.
- **Model risk** — the combination of a model's influence on a decision and the consequence of error.
- **OMOP CDM** — a common data model for observational health data.
- **PBPK** — physiologically based pharmacokinetic model.
- **Prognostic score** — a scalar predicting outcome under control.
- **PROCOVA** — prognostic covariate adjustment; a pre-specified ANCOVA on an external score.
- **QSP** — quantitative systems pharmacology model.
- **Synthetic control** — a comparison group generated by a model rather than observed.
- **Transportability** — whether a model validated in one population predicts in another.
- **Virtual patient** — a synthetic individual sampled from a model.

## 16. Sources

- ASME. *Assessing Credibility of Computational Modeling through Verification and Validation:
  Application to Medical Devices*. V&V 40-2018. American Society of Mechanical Engineers, 2018.
- US Food and Drug Administration. *Assessing the Credibility of Computational Modeling and
  Simulation in Medical Device Submissions*. Final guidance, November 2023. Docket FDA-2021-D-0980.
- US Food and Drug Administration. *Interacting with the FDA on Complex Innovative Trial Designs for
  Drugs and Biological Products*. Guidance for industry, 2020.
- European Medicines Agency. *Qualification Opinion for Prognostic Covariate Adjustment (PROCOVA)*.
  EMA/CHMP, adopted 15 September 2022, published 20 September 2022.
- Schuler A, Walsh D, Hall D, Walsh J, Fisher C. Increasing the efficiency of randomized trial
  estimates via linear adjustment for a prognostic score. *International Journal of Biostatistics*,
  2022. doi:10.1515/ijb-2021-0072.
- International Council for Harmonisation. *E9(R1) Addendum on Estimands and Sensitivity Analysis in
  Clinical Trials*. 2019.
- International Council for Harmonisation. *E6(R2) Good Clinical Practice*. 2016.
- CDISC. *Study Data Tabulation Model* and *Analysis Data Model* implementation guides.
- OHDSI. *Observational Medical Outcomes Partnership Common Data Model*, current version.
- Pathmanathan P, et al. Credibility assessment of in silico clinical trials for medical devices.
  *PLOS Computational Biology*, 2024. doi:10.1371/journal.pcbi.1012289.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`10_submission-data.md`](10_submission-data.md)
- Next topic: [`12_statistical-pharmacovigilance.md`](12_statistical-pharmacovigilance.md)
- Related topic: [`13_pass-and-active-safety-surveillance.md`](13_pass-and-active-safety-surveillance.md)
- Related topic: [`07_rwe-in-pharma.md`](07_rwe-in-pharma.md)
- Survey file: [`../../course/13_biopharma_clinical_development.md`](../../course/13_biopharma_clinical_development.md)
- Labs: [`../../labs/`](../../labs/)
