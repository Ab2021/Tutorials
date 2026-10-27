# In-Silico Trials and Computational Modeling and Simulation

> **Module 14 — Medical Devices, SaMD, and Regulation**
> Topic 11 of 12
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 03, 05, and 06
> Version 2026.3

## 1. Why this topic exists

A computational model is cheap to run and expensive to trust. That asymmetry is the whole
problem. A finite element model of a stent can run overnight; a bench fatigue campaign for the
same stent takes months and costs six figures; an animal study costs more and carries an ethics
burden; a human trial costs more again and may be impossible for the population you care about.
So the model gets built, and then someone has to decide how much weight it can carry.

Three people are blocked by that decision, and they are blocked in different ways.

The first is an engineer who wants to change a design. A fixation plate is being revised: one
screw hole moves four millimetres, the plate thickness drops from 3.5 mm to 3.2 mm. The existing
bench fatigue data do not cover the new geometry. Re-running the whole bench campaign would add
five months to the programme. A finite element model already exists and predicts a peak stress
four percent lower than the predicate. Can the model substitute for the bench test? The answer
depends on what the model will be used to decide, not on how good the model is in the abstract.

The second is a regulator. A 510(k) submission arrives with a 400-page computational appendix and
a claim that the device is safe across an annulus range wider than the predicate's. The reviewer
has weeks, not months. The reviewer needs to know which parts of the model were verified, which
were calibrated, which were validated against independent data, and whether the cohort of virtual
patients actually resembles the population in the label. A submission that does not answer those
questions in the first twenty pages will be sent back, and the delay is measured in quarters.

The third is a clinical team. A device works in adults. A paediatric version is needed and no
trial is feasible: the population is small, the anatomy is different, and the risk of a trial is
not justified by the size of the market. Simulation is the only route to evidence. That makes the
credibility question harder, not easier, because there is no clinical fallback to catch a model
that is wrong.

The regulatory position has changed in the last decade. The United States Food and Drug
Administration (FDA) published final guidance in November 2023 on assessing the credibility of
computational modeling and simulation in medical device submissions, following a 2021 draft and
building on the consensus standard ASME V&V 40-2018. Both documents say the same thing in
different words: credibility is not a property of a model. It is a property of a model used for a
particular purpose to inform a particular decision. The same solver can be credible for ranking
five design options and not credible for supporting a labelling claim, because the consequences of
being wrong are different.

That distinction is the subject of this topic. It is also the distinction most often collapsed in
practice, because the model, the code, and the report are the same in both cases. Only the
decision changes.

There is a second reason this topic exists. Computational modeling and simulation (CM&S) is where
the data scientist's habits and the engineer's habits collide. A data scientist reaches for
cross-validation, held-out sets, and calibration curves. A computational engineer reaches for mesh
convergence, manufactured solutions, and bench comparators. Both are right, and each is blind to
the other's failure modes. A model can pass every statistical check and still solve the wrong
equations. It can converge beautifully on a boundary condition nobody validated. An in-silico
trial needs both sets of checks, applied in the right order, and most teams are fluent in only one.

What this topic does not do is teach computational fluid dynamics (CFD) or finite element analysis
(FEA). Those are engineering disciplines with their own literatures. What it teaches is how to
decide whether a simulation is fit for a stated purpose, how to show that in a document a reviewer
can audit, and how to avoid the specific errors that turn a useful design tool into a false claim.

> **Note:** Regulatory material in this file is educational. It is not legal advice, and it does
> not substitute for a regulatory professional who knows your device, your markets, and the
> current state of the guidance.

## 2. Learning outcomes

By the end of this topic you will be able to:

1. **Write** a context-of-use statement that names the decision the model informs, the output
   quantity, the population, and the boundary of the claim, and **critique** a vague one.
2. **Classify** a model as mechanistic, statistical, or hybrid, and **state** what each class can
   and cannot support.
3. **Apply** the FDA nine-step credibility workflow and ASME V&V 40-2018 to a concrete model,
   including a model-risk rating from influence and decision consequence.
4. **Distinguish** code verification, calculation verification, calibration, validation, and
   applicability, and **identify** which has been skipped from the evidence presented.
5. **Compute** a cohort coverage statistic comparing simulated and observed populations, and
   **decompose** the resulting bias into geometry and covariate components.
6. **Detect** leakage between calibration and validation, including the arithmetic signature of a
   validation root-mean-square error that falls below bench repeatability.
7. **Decide** whether a given simulation can support a claim or only a design decision, and
   **justify** the decision in writing.
8. **Design** a credibility assessment report outline that a reviewer can audit without reading
   the code.

## 3. The core mental model

The mental model has one claim and two ledgers.

The claim: **a simulation is an argument, not a measurement.** It moves from a question, through a
model, to a statement about the world. Every step in that movement is an inference that can fail,
and credibility is the accumulated strength of the chain, not the strength of any link.

The two ledgers are the reason the same model can be adequate in one room and inadequate in
another.

```text
                     DESIGN DECISION LEDGER          CLAIM LEDGER
   Question          which of five designs is best?  is the device safe across the label?
   Cost of error     rework, weeks, internal money    false public statement, injury, recall
   Audience          the team                         regulator, clinician, plaintiff
   What you need     a correct RANKING                a defensible BOUND, including the tail
   Falsified by      a better design appearing        one patient outside the envelope
   Standard of proof better than the alternatives    preponderance, documented in advance
```

A ranking only has to be right about the ordering. A bound has to be right about the magnitude,
including the tail. These are different mathematical requirements, and a model tuned for the first
routinely fails the second.

### The chain

```text
  question of interest -> context of use -> model -> claim
                             |                 |
                             |                 +-- code verification       "solved correctly?"
                             |                 +-- calculation verification "numerical error small?"
                             |                 +-- calibration              "tuned on what?"
                             |                 +-- validation               "agrees with independent data?"
                             |                 +-- applicability            "envelope covers the claim?"
                             |                 +-- uncertainty                "how wide, and why?"
                             v
                  who, what, when, which population, which output
```

A reviewer walks this chain in order. Most failed submissions break at calibration or applicability,
not at the solver: the solver is the part engineers enjoy checking, and it is usually the part that
is fine.

### Three model classes and what each can bear

| | Mechanistic | Statistical | Hybrid |
|---|---|---|---|
| Built from | Conservation laws, constitutive models, boundary conditions | Regression, trees, neural networks, survival models | Physics model with learned closures, or a surrogate of a physics model |
| Extrapolates because | It encodes mechanism | It does not; it interpolates within the covariate support | Only as far as the physics term constrains the answer |
| Fails when | The physics or the boundary condition is wrong | The population shifts, or the label is a proxy for the outcome | The two halves disagree and nobody notices |
| Verification means | Manufactured solutions, mesh convergence | Leakage checks, calibration, drift monitoring | Both, plus a check that the surrogate is used inside its training envelope |

The FDA 2023 guidance covers first-principles models and the first-principles components of hybrid
models. Standalone statistical and machine-learning models are outside its scope; they are handled
by other guidance, including the good machine learning practice material covered in Topic 04. A claim
stitched together from a computational fluid dynamics model and a logistic regression therefore has
two credibility regimes in one document, and a reviewer will ask about both.

### Model risk

ASME V&V 40-2018 and the FDA guidance both rate model risk on two axes: **model influence** (how
much the model drives the decision) and **decision consequence** (what happens if the decision is
wrong). The product sets the credibility effort.

```text
                 decision consequence
                 low          medium         high
            +------------+------------+------------+
  influence |            |            |            |
  high      |  medium    |   high     |  maximum   |
            |            |            |            |
            +------------+------------+------------+
  medium    |  low       |  medium    |   high     |
            +------------+------------+------------+
  low       |  low       |  low       |  medium    |
            +------------+------------+------------+
```

A model that ranks five screw-hole positions is low influence and low consequence. A model that
replaces a clinical arm for a paediatric indication is high influence and high consequence. The
first needs a plausible answer and a sanity check. The second needs verification, independent
validation, uncertainty quantification, and a documented prospective rationale agreed with the
regulator before the study is run.

The trap is that the model does not change when its influence changes. A team builds a model for
the left-hand cell, then a programme manager notices it could support the right-hand cell, and the
evidence package is promoted without the work. The rest of this topic is largely about detecting
that promotion.

### Where the chain actually breaks

In submissions and in internal reviews the failures cluster in four places, in this order of
frequency: **applicability** (validated on bench specimens, applied to anatomy never seen),
**calibration contamination** (parameters tuned on data later described as validation), **cohort
coverage** (the virtual population is narrower than the labelled one), and **uncertainty** (a single
deterministic run reported with no interval). Solver errors are real but rarer, and they are the ones
the existing engineering culture is best at catching. The data scientist's contribution is mostly to
the third and fourth; the engineer's is to the first and second.

## 4. Definitions and vocabulary

The table gives the working definition, the misuse that causes trouble, and the unit or coding
system where one applies. Terms are ordered so that each one builds on the ones above it.

| Term | Working definition | Common misuse | Unit or system |
|---|---|---|---|
| In-silico trial (ISCT) | A study in which a device's performance is evaluated in a computational model of patients rather than, or in addition to, living subjects | Used for any simulation, including a single run on one anatomy | None; a study design label |
| Computational modeling and simulation (CM&S) | The family of methods that represent a device, anatomy, or physiology in a numerical model and compute its behaviour | Treated as synonymous with CFD; it includes FEA, lumped-parameter, agent-based, and pharmacokinetic models | None |
| Verification | Establishing that the model is solved correctly | Confused with validation; a verified model can still be wrong about the world | Dimensionless error metrics |
| Code verification | Showing that the implementation solves the intended equations, usually by comparing against an exact solution | Replaced by "the software is commercial and widely used" | Residual norms, order of accuracy |
| Calculation verification | Showing that numerical error (mesh, time step, iteration) is small relative to the quantity of interest | Reported as a single mesh-refinement table with no extrapolation | Error in the output unit, for example mmHg or MPa |
| Validation | Comparing model output with independent physical measurement in the context of use | Applied to a comparison against data used to build the model | Error metric plus its uncertainty |
| Calibration | Adjusting model parameters so that output matches a reference dataset | Reported as validation; the same data then appear twice | Parameter values and their fitted ranges |
| Applicability | The argument that a validated model may be used outside the exact conditions of its validation | Assumed whenever the physics "should be the same" | A stated envelope, not a number |
| Context of use (COU) | The specific role and scope of the model in the regulatory decision: question, output, population, conditions | Written as a restatement of the model description | Free text, but auditable |
| Question of interest | The clinical or engineering question the model exists to answer | Omitted, so the model answers a question nobody asked | Free text |
| Model influence | How much the model's output drives the decision, relative to other evidence | Rated high by the modelling team and low by the regulatory team, with no reconciliation | Ordinal: low, medium, high |
| Decision consequence | The severity of harm if the decision informed by the model is wrong | Rated by the device's risk class rather than the specific decision | Ordinal: low, medium, high |
| Model risk | The combination of influence and consequence that sets the required credibility effort | Treated as a property of the code's complexity | Ordinal matrix |
| Credibility | The trust that a model, in a specific context of use, supports a specific decision | Treated as a single score for the model as a whole | Ordinal, with a documented rationale |
| Credibility factor | One of the categories of evidence against which credibility is judged, such as code verification or population validation | Treated as a checklist to complete rather than a judgement to make | Named categories |
| Credibility evidence | The specific studies, comparisons, and analyses that populate a credibility factor | Supplied as appendices with no mapping to the factors | Documents and datasets |
| Uncertainty quantification (UQ) | Propagating input and model uncertainty to an interval on the output | Replaced by a sensitivity analysis, which answers a different question | Interval in the output unit |
| Sensitivity analysis | Ranking which inputs most affect the output | Reported as if it bounded the output | Ranked indices, for example Sobol indices, dimensionless |
| Virtual patient | A computational representation of one patient's anatomy or physiology, used as a test case | Used for a synthetic average that represents nobody | Model instance |
| Virtual cohort | A set of virtual patients used to represent a population | Assumed representative because it is large | Set of model instances plus a sampling description |
| Statistical shape model (SSM) | A parameterised description of anatomical shape built by registering a training set and decomposing the variation | Treated as a generative model of patients, including their material properties | Modes, eigenvalues, shape coefficients |
| Subject-specific model | A model built from one real patient's imaging | Called a virtual patient, blurring whether the anatomy is observed or generated | Model instance with a source image identifier |
| Digital twin | A model kept synchronised with a specific physical asset or patient through data | Used as a synonym for any personalised model | Model plus a data link |
| Surrogate model | A fast approximation trained on runs of an expensive model | Used outside the input range where it was trained | Regression or emulator with a stated training envelope |
| Boundary condition | The constraint applied at the edge of the computational domain, such as inlet flow or fixed displacement | Taken from a textbook case rather than the anatomy in the model | Physical units: mmHg, mL/s, mm |
| Constitutive model | The equation relating stress to strain, or flux to gradient, for a material | Selected for convenience and never varied in sensitivity analysis | Material parameters with units |
| Validation metric | The scalar used to compare model and measurement, with its acceptance criterion | Chosen after seeing the result | Unit of the output quantity |
| Coverage | The degree to which a virtual cohort spans the covariate distribution of the target population | Reported as a single pooled number that hides the tails | Proportion, dimensionless |
| Extrapolation | Using the model outside the range of conditions represented in validation | Described as "application" to avoid the word | Distance beyond the validated envelope, in covariate units |
| Grid convergence index (GCI) | A Richardson-extrapolation-based estimate of discretisation error from successive mesh refinements | Computed from two meshes, which cannot separate error from noise | Percent of the extrapolated value |
| Method of manufactured solutions (MMS) | Code verification by substituting a chosen analytic solution into the governing equations and adding the resulting source term | Omitted because the code is commercial | Observed order of accuracy versus formal order |
| Repeatability | Agreement between repeated measurements of the same specimen by the same method | Confused with model accuracy; a model cannot beat the reference | Standard deviation in the output unit |
| Envelope of validation | The set of conditions, geometries, and parameter ranges actually covered by validation evidence | Assumed to extend to any anatomy of the same organ | A bounding box in covariate space |
| Prospective adequacy rationale | A pre-study argument, sometimes agreed with the regulator through a Q-Submission, that the planned evidence will be sufficient | Written after the study, which converts it into a justification | Document with a date |
| Credibility assessment report (CAR) | The document that maps evidence to credibility factors and states the resulting judgement | Treated as a cover letter for the appendices | Structured report |

Three conventions matter when reading the rest of the file.

**Units are part of the claim.** A predicted stress of "120" means nothing; 120 MPa at a stated
load case means something. Every number in a credibility argument carries a unit and, where the
unit is conventional, the convention. Blood pressure in mmHg and stress in MPa are both
conventional and both need stating.

**Ordinal scales are not scores.** Low, medium, and high are judgements with a rationale, not
points to be summed. A model that scores "medium" on four factors is not thereby credible; the
factors are not exchangeable.

**The context of use is a sentence, not a section.** If you cannot write the COU as two or three
sentences naming the decision, the output quantity, the population, and the conditions, the model
is not ready to be assessed. Everything downstream inherits the ambiguity.
## 5. The mechanics, step by step

The procedure below takes a model from a decision to a documented credibility judgement. It follows
the nine-step structure of the FDA 2023 guidance and the risk-informed logic of ASME V&V 40-2018,
with the data-science steps inserted where they belong. Each step states its input, its output, the
reason it exists, the procedure, the branch points, and the edge cases.

The steps are ordered because later steps consume earlier outputs. In particular, the context of
use from step 2 is the object against which every later choice is judged, and the risk rating from
step 3 sets how much work steps 5 to 10 require.

### Step 1 — State the decision and the question of interest

**Input.** A programme decision that is actually pending, the device description, and the evidence
that already exists.

**Output.** One sentence naming the decision, and one sentence naming the question the model must
answer to inform it.

**Rationale.** A model built without a pending decision drifts toward answering the most
interesting technical question rather than the decision-relevant one. Naming the decision also
names the cost of being wrong, which drives everything in step 3. If the decision is "which of
three hinge geometries has the lowest peak stress", the model needs to rank. If the decision is
"can the label cover annuli from 19 to 31 mm", the model needs to bound an error across a
population.

**Procedure.** (1) Write the decision as a question with a named decision-maker. (2) Write the
answer the decision-maker needs, in the form it will be consumed: a ranking, a number with an
interval, or a yes or no against a threshold. (3) Write what would change the decision. (4) Check
that the model's output can actually change it. (5) Date the statement.

**Branch points.** If the decision is a ranking, the model needs ordinal accuracy and steps 8 to 10
can be lighter. If the decision is a threshold, the model needs a calibrated boundary and a false
negative rate. If the decision is a population claim, the model needs cohort coverage and the
population validation of step 9.

### Step 2 — Write the context of use

**Input.** The decision and question from step 1, the intended-use statement, and the label.

**Output.** A context-of-use statement of two to four sentences, plus an explicit out-of-scope list.

**Rationale.** The context of use (COU) is the contract between the model and the claim. It fixes
the output quantity, the population, the conditions, and the boundary. Every later judgement about
adequacy is made against it, so an ambiguous COU makes the credibility argument unfalsifiable.
Reviewers read the COU first because everything else is judged relative to it.

**Procedure.** (1) Name the output quantity and its unit. (2) Name the population, with the
covariates that matter: age, sex, anatomy range, comorbidity, device size. (3) Name the conditions:
loading, flow, temperature, duration, use environment. (4) Name what the model is not being used
for. (5) Name the model version and software version. (6) Have a second person check that the
sentence would exclude a plausible misuse.

**Branch points.** A COU that covers a range of anatomies needs a cohort argument. A COU that
covers a single nominal anatomy needs a subject-specific argument. A COU that covers a time span,
such as ten years of implant life, needs an argument about how the short-term validation maps onto
the long-term claim.

### Step 3 — Rate the model risk

**Input.** The decision from step 1, the COU from step 2, and the risk analysis from the device's
ISO 14971 file.

**Output.** A model-risk rating, an influence rating, a consequence rating, and a written
rationale for each.

**Rationale.** Model risk sets the effort. ASME V&V 40-2018 combines how much the model influences
the decision with how bad the consequence is if the decision is wrong. The combination determines
which credibility factors need strong evidence and which can be satisfied by argument. Rating it
explicitly also exposes promotions: a model built at low influence that is later used at high
influence has to be re-rated, and the re-rating usually triggers work.

**Procedure.** (1) List the evidence the decision will rest on. (2) Estimate what fraction of the
decision the model carries. (3) Rate influence low, medium, or high. (4) Take the hazard and harm
from the ISO 14971 analysis for the failure the model could mask. (5) Rate consequence low,
medium, or high. (6) Read the cell from the matrix. (7) Record who rated it and when.

**Decision table — model risk from influence and consequence**

| Influence | Consequence | Model risk | What the evidence package needs |
|---|---|---|---|
| Low | Low | Low | Plausibility argument, code verification, a sanity comparison |
| Low | Medium | Low | As above, plus a documented sensitivity check |
| Low | High | Medium | Calculation verification, bench validation, stated limits |
| Medium | Low | Low | Code verification, calculation verification, one validation comparison |
| Medium | Medium | Medium | Full verification, independent validation, sensitivity analysis |
| Medium | High | High | As above, plus uncertainty quantification and an applicability argument |
| High | Low | Medium | Verification, validation, and a stated envelope |
| High | Medium | High | As above, plus population validation and a prospective rationale |
| High | High | Maximum | All factors strong, prospective agreement with the regulator, full UQ, audited data provenance |

**Branch points.** A model that informs a design freeze is usually medium influence. A model that
replaces a bench test in a submission is high influence. A model that replaces a clinical endpoint
is high influence and high consequence, and the burden is then comparable to a clinical study.

### Step 4 — Choose the model class and justify it

**Input.** The COU, the model-risk rating, and the available data and computing budget.

**Output.** A model architecture, a written justification, and a list of the assumptions the
architecture embeds.

**Rationale.** The model class determines what kind of evidence can establish credibility. A
mechanistic model can extrapolate but must have its constitutive laws and boundary conditions
defended. A statistical model interpolates well but cannot be trusted outside its training
support. A hybrid inherits both obligations. Choosing the class after seeing which one gives a
convenient answer is a form of result-driven analysis.

**Procedure.** (1) Decide whether the governing physics is known well enough to encode. (2) Decide
whether representative data exist to fit a statistical model. (3) If both, decide how the two
halves will be coupled and which one dominates the output. (4) Write the assumptions the choice
embeds: material homogeneity, rigid walls, quasi-static loading, independent observations. (5)
Decide the fidelity level: three-dimensional versus reduced-order, transient versus steady. (6)
Record the alternatives considered and why they were rejected.

**Branch points.** A model whose output is dominated by geometry favours a mechanistic or hybrid
approach with a shape model. A model whose output is dominated by population variability favours a
statistical or hybrid approach with an explicit cohort. A model that must run thousands of times
for a cohort needs a surrogate, which adds its own validation obligation.

### Step 5 — Verify the code

**Input.** The solver, its documentation, and the governing equations in the COU.

**Output.** A code verification record: test problems, observed order of accuracy, and any
discrepancies.

**Rationale.** Code verification answers whether the equations the code solves are the equations
you think it solves. It is cheap relative to validation and catches errors that no amount of
physical data will reveal, because a coding error and a wrong constitutive law can produce
similarly wrong numbers. For commercial solvers, verification is partly inherited but the
inheritance must be documented for the specific physics and element types used.

**Procedure.** (1) Identify the governing equations and their expected order of accuracy. (2)
Select a manufactured solution or an analytic benchmark with the same mathematical character. (3)
Refine the mesh or time step across at least three levels. (4) Compute the observed order of
accuracy. (5) Compare with the formal order. (6) Record the test case, the input files, and the
result. (7) Repeat for each physics module used in the COU.

**Branch points.** A code with adaptive meshing or limiters will show a lower observed order than
the formal one, which is expected and must be explained rather than hidden. A coupled multiphysics
solve needs a verification case that exercises the coupling.

### Step 6 — Verify the calculation

**Input.** The verified code, the specific geometry and load case in the COU, and a refinement
strategy.

**Output.** A discretisation error estimate for the quantity of interest, and a chosen mesh or time
step.

**Rationale.** Calculation verification answers how much of the answer is numerical artefact. It is
distinct from code verification: a correct code on an inadequate mesh gives a wrong number. The
estimate must be for the quantity of interest, not for a global norm, because a mesh that is
adequate for displacement can be inadequate for peak stress.

**Procedure.** (1) Choose the output quantity from the COU. (2) Run at three refinement levels with
a constant refinement ratio, typically 1.5 to 2. (3) Compute the grid convergence index (GCI) by
Richardson extrapolation. (4) Check that the observed order is plausible. (5) Check that the
extrapolated value is not sensitive to which three levels are used. (6) Set the production mesh so
that the discretisation error is small relative to the effect size the claim depends on. (7) Record
the runs.

**Branch points.** If the three levels do not give a consistent order, the asymptotic range has not
been reached and finer meshes are needed. If the quantity of interest is noisy, as with contact
stress, the GCI may be dominated by non-smoothness and a different approach is needed. If the
effect size is smaller than the discretisation error, the model cannot resolve the claim at all.

### Step 7 — Calibrate, and firewall the calibration data

**Input.** The model, a calibration dataset, and a list of parameters to tune.

**Output.** Fitted parameter values, a calibration record, and a frozen validation dataset that
was never touched.

**Rationale.** Calibration is legitimate and often unavoidable: material properties and boundary
conditions are rarely known exactly. The credibility problem is not that calibration happened but
that the same data are then used to claim agreement. Once a dataset has influenced a parameter, it
cannot provide independent evidence. The firewall has to be built before the fitting starts,
because afterwards every choice is suspect.

**Procedure.** (1) List the parameters to be calibrated and their physically plausible ranges. (2)
Split the available data into calibration and validation sets before looking at either, stratifying
by the covariate that matters. (3) Fit within the plausible ranges, recording the objective
function and the optimiser. (4) Report the fitted values with their uncertainty. (5) Report the
calibration residual and its structure, not only its magnitude. (6) Freeze and hash the validation
set. (7) Record which data influenced which parameter.

**Branch points.** If the calibration dataset is the only data that exist, there is no independent
validation and the credibility argument must rest on other factors, stated as such. If the fit
drives a parameter to the edge of its plausible range, the model form is probably wrong and the
fit should not be accepted.

### Step 8 — Validate against independent measurement

**Input.** The frozen validation dataset, the calibrated model, and a pre-specified validation
metric.

**Output.** A validation result with an uncertainty interval, and a judgement against the
pre-specified acceptance criterion.

**Rationale.** Validation asks whether the model agrees with reality closely enough for the COU. The
metric must be chosen before the comparison, because any set of numbers can be summarised
favourably after the fact. The comparison must also account for measurement uncertainty: a model
cannot be expected to agree more closely than the reference method can resolve, and a claimed
agreement tighter than repeatability is evidence of leakage, not of accuracy.

**Procedure.** (1) State the validation metric and its acceptance criterion, with a date. (2) State
the uncertainty of the reference measurement, including repeatability. (3) Compare on the frozen
data. (4) Compute the interval on the agreement, not only the point estimate. (5) Check the
residual structure against the covariates. (6) Check that the agreement is not tighter than the
reference uncertainty. (7) Record the comparison.

**Decision table — interpreting a validation result**

| Observation | Interpretation | Action |
|---|---|---|
| Agreement within criterion, interval excludes a harmful discrepancy, residual structure flat | Validation supports the COU | Proceed to applicability |
| Agreement within criterion, but the interval includes a harmful discrepancy | Underpowered, not validated | Extend the validation set or widen the claim interval |
| Agreement within criterion, residual structure shows a trend with a covariate | Agreement is coincidental over the sampled range | Investigate the missing term; do not extrapolate |
| Disagreement outside criterion | Model or measurement is wrong | Diagnose; if the model is wrong, return to step 4 |
| Agreement tighter than reference repeatability | Impossible without leakage | Audit the data split immediately |

**Branch points.** A validation that fails on the whole range but passes on a sub-range supports a
narrower COU, not a failed project. A validation that passes only after removing outliers needs the
outlier rule to have been pre-specified.

### Step 9 — Build the virtual cohort and check its coverage

**Input.** The validated model, a source of anatomical or physiological variation, and the
population named in the COU.

**Output.** A virtual cohort with a documented sampling scheme, and a coverage assessment against
an observed population.

**Rationale.** An in-silico trial replaces patients with virtual patients, so the cohort is the
study population. If the cohort is narrower than the labelled population, the trial systematically
avoids the patients most likely to fail. This is the mechanism by which a technically excellent
model produces an optimistic result: the model is fine and the population is wrong.

**Procedure.** (1) Choose the cohort type: subject-specific, statistical shape model, or
registry-derived. (2) Document the training population of the shape model. (3) Choose the sampling
scheme and the truncation of each mode. (4) Generate the cohort and record the seed. (5) Compare
marginal distributions with an observed population using standardised mean differences, variance
ratios, and a distributional distance. (6) Compute coverage of the observed distribution by the
simulated support. (7) Check coverage by subgroup and by tail, not only pooled. (8) Re-weight or
regenerate if coverage fails.

**Decision table — cohort coverage outcomes**

| Coverage finding | Interpretation | Action |
|---|---|---|
| Means and variances match, tails covered, no subgroup gap | Cohort represents the population | Proceed to uncertainty quantification |
| Means match, variances too small, tails outside support | Cohort is truncated, usually by mode clipping | Widen the sampling or model the tails explicitly |
| Means differ in a subgroup only | Cohort misses a subpopulation | Stratify and regenerate, or narrow the COU |
| A covariate is absent above a threshold | The generative model cannot represent that region | State the limit and restrict the claim, or source that region elsewhere |
| Cohort matches the training set but not the target population | The shape model inherited its training bias | Rebuild the shape model on a representative sample |

**Branch points.** A subject-specific cohort, built from real scans, has coverage by construction
but a small sample. A statistical cohort can be arbitrarily large but only as representative as its
training data. A registry-derived cohort has real covariates but often lacks the geometry.

### Step 10 — Quantify uncertainty

**Input.** The calibrated, validated model, the cohort, and the uncertainty in inputs and model
form.

**Output.** An interval on the output quantity, decomposed by source.

**Rationale.** A point estimate from a deterministic run is not a prediction; it is one draw from an
unknown distribution. The interval is what a claim can be built on. Decomposition matters because it
tells you where to spend the next dollar: if model-form uncertainty dominates, more patient data
will not help.

**Procedure.** (1) List the uncertainty sources: input variability, parameter uncertainty from
calibration, model-form error, numerical error, and cohort sampling. (2) Propagate input and
parameter uncertainty by sampling. (3) Estimate model-form error from the validation residuals, not
from expert judgement alone. (4) Carry discretisation error from step 6. (5) Combine, stating
whether the sources are treated as independent. (6) Report the interval and the contribution of each
source.

**Branch points.** If model-form error dominates, the claim interval cannot be narrowed by adding
patients. If input variability dominates, the claim is about a distribution and should be phrased
as a rate, not a limit. If numerical error dominates, return to step 6.

### Step 11 — Assess applicability and write the credibility argument

**Input.** Everything from steps 1 to 10.

**Output.** A credibility assessment report that maps evidence to factors and states a judgement.

**Rationale.** Applicability is the step most often skipped and most often fatal: the model was
validated somewhere, and the claim is somewhere else. The report has to make the mapping from
evidence to claim explicit, because the reviewer will otherwise do it themselves and reach a
different conclusion.

**Procedure.** (1) Tabulate each credibility factor against the evidence that supports it. (2) For
each, state the strength: strong, moderate, weak, or absent. (3) Write the applicability argument
from the validated envelope to the claimed envelope, naming every extrapolation. (4) State the
residual risk that the model is wrong in a way the evidence would not detect. (5) State the
conclusion in the form the decision needs. (6) Have a reviewer who did not build the model attempt
to falsify the argument.

**Branch points.** An applicability argument that relies on "the physics is the same" needs a
demonstration, not an assertion. An argument that relies on a published model of the same organ
needs the differences between the two models stated.

### Step 12 — Decide what the model can support, and document it

**Input.** The credibility assessment and the decision from step 1.

**Output.** A decision record: what the model supports, what it does not, and what would have to
change.

**Rationale.** The same evidence supports different conclusions depending on the decision. Making
the conclusion explicit prevents the promotion described in section 3, where a design tool becomes a
regulatory claim by drift.

**Procedure.** (1) Compare the achieved credibility with the requirement from step 3. (2) If
achieved exceeds required, state the supported claim. (3) If not, state the narrower claim that is
supported. (4) Record the gap and the work needed to close it. (5) Record the date and the model
version. (6) Re-rate if the intended use changes.

**Decision table — what the evidence supports**

| Achieved versus required credibility | Permitted use | Documentation |
|---|---|---|
| Exceeds requirement, all factors strong | Support a labelling or safety claim | Full credibility assessment report in the submission |
| Meets requirement, some factors moderate | Support a submission argument alongside other evidence | Report with the weak factors named |
| Meets requirement only within a sub-range | Support a narrowed claim | Report plus a revised COU and label |
| Below requirement, but verification and plausibility are sound | Support internal design decisions only | Internal memo; not for submission |
| Below requirement and verification is incomplete | Support nothing | Do not use the output in any decision |

**Branch points.** A model that fails for a claim may still be useful for design. A model that
passes for a claim in one jurisdiction may not satisfy another, because the guidance and the
evidentiary standard differ.

The device, sponsor, cohort, and numbers in this case are constructed for teaching. The arithmetic
is real and is reproduced by the script in section 7. The case is not a real submission and the
device does not exist.

## 6. Worked example

### 6.1 The decision and the claim

A sponsor makes a self-expanding transcatheter aortic valve in six sizes: 20, 22, 24, 26, 28, and
30 mm. The current label covers annulus diameters from 21.0 to 27.0 mm. The sponsor wants to widen
the label to 19.0 to 31.0 mm, which would cover a substantially larger fraction of the referral
population, and wants to do it without a new clinical study.

The device has an existing finite element model of deployment: the valve frame is expanded inside
a patient-specific aortic root, contact and oversizing are simulated, and the output is a predicted
probability of procedural success. Procedural success is defined in the clinical study as device
implanted in the intended position with no more than mild paravalvular leak and no annulus rupture.

The sponsor's regulatory argument is: "Simulation predicts procedural success in 94.6 percent of
the intended-use population, which exceeds the 93 percent threshold the clinical study met."

That sentence contains a context of use, a number, and a population. The rest of this case is about
whether the number belongs to the population it is attached to.

### 6.2 What was built

**The model.** A finite element model of the frame, calibrated against bench radial force
measurements, with the aortic root represented as a hyperelastic material. The model was verified
by mesh convergence on peak frame strain and validated against bench radial force and against
bench deployment in silicone annuli.

**The cohort.** A statistical shape model built from 240 contrast computed tomography (CT) scans of
patients who received the device in the clinical study. The shape model retained five modes
explaining 87 percent of the variance. For the in-silico trial, mode amplitudes were sampled
uniformly between minus two and plus two standard deviations and 1,000 virtual patients were
generated. Annulus calcification was assigned from a regression on diameter, with a cap at a
calcification index of 0.72 because the segmentation pipeline could not reliably grade heavier
calcification.

**The comparison cohort.** A registry of 512 patients treated with the predicate device across
seven centres, with measured annulus diameter and a calcification index graded on the same scale.
This registry was not used to build the shape model.

### 6.3 The coverage arithmetic

The first check is whether the virtual cohort resembles the registry on the two covariates that
drive the result.

| Quantity | Observed registry, n = 512 | Simulated cohort, n = 1,000 |
|---|---:|---:|
| Annulus diameter, mean (mm) | 24.538 | 24.449 |
| Annulus diameter, standard deviation (mm) | 1.949 | 1.648 |
| Calcification index, mean | 0.336 | 0.325 |
| Calcification index, standard deviation | 0.203 | 0.160 |
| Diameter, 1st to 99th percentile (mm) | 21.04 to 30.55 | 21.98 to 27.49 |
| Calcification, 1st to 99th percentile | 0.020 to 0.799 | 0.020 to 0.707 |

The standardised mean difference for diameter is the difference in means divided by the pooled
standard deviation:

```text
SMD = (24.449 - 24.538) / sqrt((1.648^2 + 1.949^2) / 2)
    = -0.089 / sqrt((2.716 + 3.799) / 2)
    = -0.089 / sqrt(3.258)
    = -0.089 / 1.805
    = -0.049
```

An absolute standardised mean difference below 0.1 is conventionally read as negligible. On that
statistic the cohort looks excellent. The variance ratio is another matter:

```text
variance ratio = 1.648 / 1.949 = 0.846
```

The simulated cohort has 85 percent of the observed spread. The means agree because the shape
model was built on the same population; the spread does not, because the modes were clipped at two
standard deviations. A normal variable clipped at plus or minus two standard deviations loses
roughly five percent of its mass at each tail and about 15 percent of its variance, which is what
the 0.846 shows.

The distributional distance confirms it, but only just. The two-sample Kolmogorov-Smirnov statistic
is 0.0703 against a 5 percent critical value of 0.0739 for these sample sizes. It does not reject.
That is the trap: a test designed to detect a general distributional difference has low power
against a pure scale difference at these sample sizes, so a non-significant result is not evidence
of coverage.

The coverage calculation is more direct. It asks what fraction of observed patients fall inside the
support the simulated cohort actually spans, defined as its 1st to 99th percentile:

```text
diameter coverage      = 0.867   (68 of 512 patients outside)
calcification coverage = 0.963   (19 of 512 patients outside)
```

Thirteen percent of the real population lies outside the range the virtual population was allowed
to explore. Those patients are not missing at random. They are the large annuli, and they are
exactly the patients the label expansion is meant to cover.

### 6.4 The bias arithmetic

The quantity of interest is the mean predicted probability of procedural success, evaluated on the
cohort. The model is the same in both columns; only the population differs.

```text
mean predicted success, simulated cohort = 0.9461
mean predicted success, observed  cohort = 0.8889
optimism                                 = 0.9461 - 0.8889 = +0.0572
```

Five point seven percentage points. It is worth decomposing, because the two components have
different fixes. The method is a one-at-a-time substitution: evaluate the model on simulated
geometry with calcification held at a reference value, then on reference geometry with simulated
calcification, and difference each against the observed counterpart.

```text
geometry component, calcification fixed at 0.30 : +0.0523
calcification component, diameter fixed at 24.5 : +0.0068
sum of the two parts                             : +0.0592
actual joint difference                         : +0.0572
interaction (actual minus sum)                  : -0.0020
```

Geometry carries 0.0523 of the 0.0572, or 91 percent. The calcification cap, which the team worried
about and documented, contributes 0.0068. The decomposition redirects the fix: widening the mode
truncation addresses almost all of the bias, while improving the calcification grading addresses
almost none of it, because only 3.7 percent of observed patients exceed the cap.

Against the 0.93 threshold, the two cohorts reach opposite conclusions:

```text
simulated: 0.9461, Wilson 95% interval 0.9302 to 0.9584  -> passes 0.93
observed : 0.8889, Wilson 95% interval 0.8585 to 0.9131  -> fails 0.93
```

The intervals do not overlap. This is not a marginal call that more virtual patients would settle;
it is a difference in population, and adding virtual patients would narrow the simulated interval
around the wrong number.

### 6.5 Diagnostic 1: the pooled coverage number hides the tail

A team that reports only the pooled coverage of 0.867 will conclude that the cohort is roughly
adequate and move on. The diagnostic is to compute the predicted success rate by decile of the
observed population rather than in aggregate.

| Decile of observed diameter | Range (mm) | Mean calcification | Predicted success |
|---:|---|---:|---:|
| 1 | 20.21 to 22.27 | 0.179 | 0.974 |
| 2 | 22.31 to 22.81 | 0.254 | 0.966 |
| 3 | 22.81 to 23.33 | 0.286 | 0.958 |
| 4 | 23.34 to 23.74 | 0.306 | 0.955 |
| 5 | 23.74 to 24.27 | 0.346 | 0.949 |
| 6 | 24.31 to 24.85 | 0.332 | 0.953 |
| 7 | 24.85 to 25.38 | 0.353 | 0.941 |
| 8 | 25.38 to 26.09 | 0.384 | 0.941 |
| 9 | 26.09 to 27.04 | 0.426 | 0.931 |
| 10 | 27.05 to 31.07 | 0.489 | 0.342 |

Nine deciles run between 0.93 and 0.97. The tenth runs at 0.342. The aggregate of 0.889 is a
weighted average of a device that works and a device that does not, and the average conceals both.

The mechanism is visible in the size table. The largest available size is 30 mm. For an annulus of
31.07 mm the achievable oversizing with a 30 mm device is negative, so no size is valid and the
predicted success is zero. For an annulus of 29 mm the oversizing with a 30 mm device is 3.4
percent, far below the 13 percent target, and the predicted success falls steeply. The label
expansion to 31 mm asks the device to treat annuli for which the commercial size range has no
adequate answer, and the virtual cohort, clipped at 27.49 mm, never tested one.

The lesson generalises: a coverage statistic computed on the pooled distribution answers a question
about the average patient. A claim about a labelled range is a question about the boundary.

### 6.6 Diagnostic 2: the validation set leaked into calibration

The second diagnostic concerns the bench validation of the frame model. The bench method measures
radial force at controlled compression, with three repeats per specimen, giving a repeatability
standard deviation of 0.34 N. Forty-eight specimens were tested and split into 24 for calibration
and 24 for validation, stratified by compression to keep the ranges comparable.

Four model forms were fitted to the calibration set and compared:

| Form | Calibration RMSE (N) | Validation RMSE (N) |
|---|---:|---:|
| Linear | 0.3820 | 0.4139 |
| Quadratic | 0.3818 | 0.4170 |
| Cubic | 0.3448 | 0.4328 |
| Quartic | 0.3373 | 0.4497 |

The report presented the quartic's calibration RMSE of 0.3373 N as its validation result. The
honest validation RMSE for the same form is 0.4497 N, and the pre-specified linear form gives
0.4139 N.

The arithmetic proof that 0.3373 cannot be a validation result is that it is below the reference
method's repeatability of 0.34 N. A model cannot agree with a measurement more closely than the
measurement agrees with itself, except by fitting the noise. And the size of the effect is
predictable: for a model with p parameters fitted to n points, the root-mean-square residual of the
fit is biased low by approximately the factor

```text
sqrt((n - p) / n) = sqrt((24 - 5) / 24) = sqrt(0.7917) = 0.890
```

so a correct five-parameter model fitted to a reference with 0.34 N repeatability should show a
calibration RMSE near 0.34 x 0.890 = 0.302 N and a validation RMSE near or above 0.34 N. The
reported 0.3373 N sits between the two, which is the signature of a number that is partly fit and
partly validated.

The diagnostic is cheap and does not require the raw data: ask for the repeatability of the
reference method and compare it with the reported agreement. If the reported agreement is tighter,
the set leaked. The follow-up question is which dataset influenced which decision, and the answer
is usually model-form selection rather than parameter fitting.

### 6.7 Vary one assumption

The cohort construction contained one discretionary choice: the shape-mode amplitudes were sampled
uniformly between minus two and plus two standard deviations. That choice was made to avoid
"non-physical shapes". It is the assumption that produces the whole result.

Re-running the cohort generation at truncations of 1.5, 2.0, 2.5 and 3.0 standard deviations, with
everything else fixed, gives simulated success rates of 0.9569, 0.9461, 0.8789 and 0.8100 against the
unchanged observed 0.8889. The optimism runs from +0.068 to -0.079 and reverses sign at 2.5 standard
deviations. The model, the device and the threshold did not change; only the sampling range did, and
it moved the answer by 13.7 percentage points between the narrowest and widest settings. That is why
the truncation must be justified against the target population rather than against physical
plausibility: a shape that is implausible for a living patient is not implausible as a stress case,
and the claim is about patients.

There is a second, more useful variation. If the claim is restricted to the currently labelled
range of 21.0 to 27.0 mm, the comparison changes:

```text
observed patients retained in range : 453 of 512
simulated patients retained in range: 922 of 1,000
mean predicted success, observed    : 0.9524
mean predicted success, simulated   : 0.9572
optimism in range                   : +0.0048
```

The optimism falls from 5.7 points to 0.5 points. Inside the labelled range the cohort is
adequate and the simulation is defensible; outside it, it is not. The conclusion therefore is not
that the model is unfit. The conclusion is that the model supports the existing label and does not
support the expansion, and the gap is concentrated in the population the expansion was meant to
serve: 11.5 percent of observed patients lie outside 21.0 to 27.0 mm.

### 6.8 What the case does not show

The case does not show that the model is wrong. Every verification and validation result in it is
respectable. It shows that a correct model applied to a population it does not represent produces a
confident and incorrect answer, and that the error is invisible in the statistics a modelling team
typically reports: the mean difference, the goodness of fit, and the pooled coverage. It also does
not show how to fix the device; it shows which of the available fixes the evidence supports.

## 7. Code, query, or calculation

The script below reproduces every number in section 6. It uses only the Python standard library,
runs in under two seconds, and is deterministic because every random draw is seeded. It is a
teaching model, not a validated device model: the cohort generator and the success function are
constructed to have the properties the case discusses.

Save it as `cohort_audit.py` and run `python cohort_audit.py`.

```python
import math, random, statistics

SEED, SIZES = 20261027, (20.0, 22.0, 24.0, 26.0, 28.0, 30.0)


def q(xs, p):
    ys = sorted(xs); pos = p * (len(ys) - 1); lo = int(pos)
    return ys[lo] + (pos - lo) * (ys[min(lo + 1, len(ys) - 1)] - ys[lo])


def ks(a, b):
    sa, sb = sorted(a), sorted(b); ia = ib = 0; best = 0.0
    for x in sorted(set(a) | set(b)):
        while ia < len(sa) and sa[ia] <= x: ia += 1
        while ib < len(sb) and sb[ib] <= x: ib += 1
        best = max(best, abs(ia / len(sa) - ib / len(sb)))
    return best


def smd(ma, sa, mb, sb):
    return (ma - mb) / math.sqrt((sa * sa + sb * sb) / 2)


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cohorts(lim=2.0, cap=0.72):
    rng = random.Random(SEED); obs = []
    while len(obs) < 512:
        z = rng.gauss(0, 1); d = 24.6 + 1.95 * z + 0.22 * (z * z - 1)
        if 19.5 <= d <= 31.5:
            c = 0.30 + 0.045 * (d - 24.0) + rng.gauss(0, 0.21)
            obs.append((round(d, 2), round(min(0.97, max(0.02, c)), 3)))
    sim = []
    while len(sim) < 1000:
        b = rng.uniform(-lim, lim); d = 24.5 + 1.42 * b + 0.10 * (b * b - 1.33)
        if 19.5 <= d <= 31.5:
            c = 0.30 + 0.045 * (d - 24.0) + rng.gauss(0, 0.15)
            sim.append((round(d, 2), round(min(cap, max(0.02, c)), 3)))
    return obs, sim


def ps(size, d, c, t=0.13, klo=1200.0, khi=25.0, kc=4.0, b0=4.6):
    o = (size - d) / d
    if o < 0 or o > 0.45: return 0.0
    pen = klo * (t - o) ** 2 if o < t else khi * (o - t) ** 2
    return 1 / (1 + math.exp(-(b0 - pen - kc * c)))


def psucc(d, c): return max(ps(s, d, c) for s in SIZES)
def meanp(co): return sum(psucc(d, c) for d, c in co) / len(co)


obs, sim = cohorts()
do = [d for d, _ in obs]; ds = [d for d, _ in sim]
co = [c for _, c in obs]; cs = [c for _, c in sim]

print("== 1. Cohort coverage ==")
for n, g in (("observed", obs), ("simulated", sim)):
    dd = [d for d, _ in g]; cc = [c for _, c in g]
    print(f"{n:<10} n={len(g):>4}  diameter mean={statistics.mean(dd):6.3f} "
          f"sd={statistics.stdev(dd):5.3f}  calc mean={statistics.mean(cc):5.3f} "
          f"sd={statistics.stdev(cc):5.3f}")
print(f"SMD diameter                : {smd(statistics.mean(ds), statistics.stdev(ds), statistics.mean(do), statistics.stdev(do)):+.3f}")
print(f"SD ratio diameter           : {statistics.stdev(ds) / statistics.stdev(do):.3f}")
print(f"SMD calcification           : {smd(statistics.mean(cs), statistics.stdev(cs), statistics.mean(co), statistics.stdev(co)):+.3f}")
print(f"SD ratio calcification      : {statistics.stdev(cs) / statistics.stdev(co):.3f}")
crit = 1.36 * math.sqrt((len(obs) + len(sim)) / (len(obs) * len(sim)))
print(f"KS diameter                 : {ks(do, ds):.4f}   (5% critical value {crit:.4f})")
print(f"KS calcification            : {ks(co, cs):.4f}   (5% critical value {crit:.4f})")
for lab, o, s in (("diameter", do, ds), ("calcification", co, cs)):
    print(f"{lab:<14} observed P1-P99 {q(o, 0.01):6.3f}-{q(o, 0.99):6.3f}  "
          f"simulated P1-P99 {q(s, 0.01):6.3f}-{q(s, 0.99):6.3f}  "
          f"coverage {sum(1 for x in o if q(s, 0.01) <= x <= q(s, 0.99)) / len(o):.3f}")
print(f"observed calcification > 0.72 : {sum(1 for c in co if c > 0.72) / len(co):.3f}")

print("\n== 2. Predicted success ==")
a, b = meanp(sim), meanp(obs)
print(f"simulated {a:.4f}   observed {b:.4f}   optimism {a - b:+.4f} ({100 * (a - b):+.2f} pp)")
geom = sum(psucc(d, 0.30) for d in ds) / len(ds) - sum(psucc(d, 0.30) for d in do) / len(do)
calc = sum(psucc(24.5, c) for c in cs) / len(cs) - sum(psucc(24.5, c) for c in co) / len(co)
print(f"geometry component        : {geom:+.4f}")
print(f"calcification component   : {calc:+.4f}")
print(f"interaction               : {a - b - geom - calc:+.4f}")
ys = sorted(obs, key=lambda t: t[0]); st = len(ys) // 10
for i in range(10):
    ch = ys[i * st:(i + 1) * st] if i < 9 else ys[9 * st:]
    print(f"decile {i + 1:>2}  D {ch[0][0]:5.2f}-{ch[-1][0]:5.2f}  "
          f"C {statistics.mean(c for _, c in ch):.3f}  success {meanp(ch):.3f}")
print(f"Wilson simulated : {wilson(int(a * 1000), 1000)[0]:.4f} to {wilson(int(a * 1000), 1000)[1]:.4f}")
print(f"Wilson observed  : {wilson(int(b * 512), 512)[0]:.4f} to {wilson(int(b * 512), 512)[1]:.4f}")

print("\n== 3. Sensitivity to truncation ==")
for lim in (1.5, 2.0, 2.5, 3.0):
    _, s2 = cohorts(lim=lim)
    p = meanp(s2)
    print(f"+/-{lim}  sim sd {statistics.stdev(d for d, _ in s2):.3f}  "
          f"p_sim {p:.4f}  p_obs {b:.4f}  optimism {p - b:+.4f}")

print("\n== 4. Restricted to 21.0-27.0 mm ==")
o2 = [t for t in obs if 21.0 <= t[0] <= 27.0]; s2 = [t for t in sim if 21.0 <= t[0] <= 27.0]
print(f"retained {len(o2)}/{len(obs)} observed, {len(s2)}/{len(sim)} simulated")
print(f"observed {meanp(o2):.4f}  simulated {meanp(s2):.4f}  optimism {meanp(s2) - meanp(o2):+.4f}")
print(f"observed outside range: {1 - len(o2) / len(obs):.3f}")

print("\n== 5. Calibration and validation ==")
rng = random.Random(SEED + 7); sd = 0.34; rows = []
for _ in range(48):
    d = rng.uniform(4, 12); rows.append((d, 0.62 * d + 0.9 + rng.gauss(0, sd)))
cal, val = rows[0::2], rows[1::2]


def poly(dat, deg):
    m = deg + 1
    A = [[sum(x ** (i + j) for x, _ in dat) for j in range(m)] for i in range(m)]
    v = [sum(x ** i * y for x, y in dat) for i in range(m)]
    for i in range(m):
        pv = A[i][i]; A[i] = [z / pv for z in A[i]]; v[i] /= pv
        for k in range(m):
            if k != i:
                f = A[k][i]; A[k] = [A[k][j] - f * A[i][j] for j in range(m)]; v[k] -= f * v[i]
    return v


def rmse(dat, cf): return math.sqrt(sum((y - sum(c * x ** i for i, c in enumerate(cf))) ** 2 for x, y in dat) / len(dat))


for name, deg in (("linear", 1), ("quadratic", 2), ("cubic", 3), ("quartic", 4)):
    cf = poly(cal, deg)
    print(f"{name:>10}  calib {rmse(cal, cf):.4f}  valid {rmse(val, cf):.4f}")
q4 = poly(cal, 4)
print(f"repeatability 0.34 N; reported 'validation' {rmse(cal, q4):.4f} N; honest validation {rmse(val, q4):.4f} N")
print(f"dof factor sqrt((24-5)/24) = {math.sqrt(19 / 24):.3f}; expected fit RMSE {0.34 * math.sqrt(19 / 24):.4f} N")
```

Output on a machine running Python 3.11.15:

```text
== 1. Cohort coverage ==
observed   n= 512  diameter mean=24.538 sd=1.949  calc mean=0.336 sd=0.203
simulated  n=1000  diameter mean=24.449 sd=1.648  calc mean=0.325 sd=0.160
SMD diameter                : -0.049
SD ratio diameter           : 0.846
SMD calcification           : -0.059
SD ratio calcification      : 0.789
KS diameter                 : 0.0703   (5% critical value 0.0739)
KS calcification            : 0.0787   (5% critical value 0.0739)
diameter       observed P1-P99 21.044-30.549  simulated P1-P99 21.980-27.490  coverage 0.867
calcification  observed P1-P99  0.020- 0.799  simulated P1-P99  0.020- 0.707  coverage 0.963
observed calcification > 0.72 : 0.037

== 2. Predicted success ==
simulated 0.9461   observed 0.8889   optimism +0.0572 (+5.72 pp)
geometry component        : +0.0523
calcification component   : +0.0068
interaction               : -0.0020
decile  1  D 20.21-22.27  C 0.179  success 0.974
decile  2  D 22.31-22.81  C 0.254  success 0.966
decile  3  D 22.81-23.33  C 0.286  success 0.958
decile  4  D 23.34-23.74  C 0.306  success 0.955
decile  5  D 23.74-24.27  C 0.346  success 0.949
decile  6  D 24.31-24.85  C 0.332  success 0.953
decile  7  D 24.85-25.38  C 0.353  success 0.941
decile  8  D 25.38-26.09  C 0.384  success 0.941
decile  9  D 26.09-27.04  C 0.426  success 0.931
decile 10  D 27.05-31.07  C 0.489  success 0.342
Wilson simulated : 0.9302 to 0.9584
Wilson observed  : 0.8585 to 0.9131

== 3. Sensitivity to truncation ==
+/-1.5  sim sd 1.235  p_sim 0.9569  p_obs 0.8889  optimism +0.0680
+/-2.0  sim sd 1.648  p_sim 0.9461  p_obs 0.8889  optimism +0.0572
+/-2.5  sim sd 2.062  p_sim 0.8789  p_obs 0.8889  optimism -0.0101
+/-3.0  sim sd 2.477  p_sim 0.8100  p_obs 0.8889  optimism -0.0790

== 4. Restricted to 21.0-27.0 mm ==
retained 453/512 observed, 922/1000 simulated
observed 0.9524  simulated 0.9572  optimism +0.0048
observed outside range: 0.115

== 5. Calibration and validation ==
    linear  calib 0.3820  valid 0.4139
 quadratic  calib 0.3818  valid 0.4170
     cubic  calib 0.3448  valid 0.4328
   quartic  calib 0.3373  valid 0.4497
repeatability 0.34 N; reported 'validation' 0.3373 N; honest validation 0.4497 N
dof factor sqrt((24-5)/24) = 0.890; expected fit RMSE 0.3025 N
```

Three things to notice. The seed fixes the numbers so a reader can reproduce them, but they are
one realisation: a different seed moves each figure in the third decimal and does not move the
conclusion. The Wilson interval is computed on integer counts, `int(a * 1000)`, not on the
proportion, which matters at small n. The decomposition uses reference values of 0.30 for
calcification and 24.5 mm for diameter; changing the reference changes the split between components
but not their sum, and a stronger interaction would require a Shapley-style allocation instead of the
one-at-a-time split.

## 8. Data, standards, and regulatory context

Two documents dominate the field, and neither is a recipe.

**ASME V&V 40-2018**, *Assessing Credibility of Computational Modeling through Verification and
Validation: Application to Medical Devices*, is a consensus standard published by the American
Society of Mechanical Engineers and recognised by the FDA. It supplies the risk-informed framework:
context of use, model risk from influence and consequence, credibility factors, and a judgement of
adequacy. It deliberately does not specify how much verification or validation is enough, because
that depends on the model and the decision. As of 2026 it is listed by ASME on stabilized
maintenance, meaning no revision is in progress.

**The FDA final guidance**, *Assessing the Credibility of Computational Modeling and Simulation in
Medical Device Submissions* (November 2023, docket FDA-2021-D-0980), sets out a nine-step process:
state the question of interest; state the context of use; assess model risk; identify credibility
evidence; define credibility factors and their adequacy goals; provide a prospective adequacy
rationale; execute the plan; perform a post-study adequacy assessment; and document the result in a
credibility assessment report. It applies to first-principles models and to the first-principles
components of hybrid models. Standalone statistical, machine-learning, and artificial-intelligence
models are outside its scope.

> **Note:** There is no ISO technical report on in-silico testing that can be cited for conformity
> as of October 2026. The work in progress is the ISO/IEC 63719 series on in-silico technologies:
> Part 1 on terminology and context, and Part 2 on credibility assessment of first-principles
> methods. Both are at committee-draft or working-draft stage and neither is available as a
> published standard. Any document that cites an ISO technical report numbered 23809 for in-silico
> testing is citing something that does not exist.

Sector standards do reference the verification and validation framework, and they are the ones a
reviewer will expect to see named:

| Standard | Scope | What it constrains | What it does not |
|---|---|---|---|
| ASME V&V 40-2018 | Credibility framework for device models | The structure of the credibility argument | The numerical acceptance thresholds |
| ASME V&V 10-2019 | Verification and validation in solid mechanics | Vocabulary and procedure for FEA credibility | Device-specific evidence |
| ASME V&V 20-2009 (R2021) | Verification and validation in CFD | Procedure for CFD verification and validation | Structural models |
| ISO 5840-3:2021 | Heart valve substitutes, transcatheter | Testing requirements, with computational methods referenced to V&V 40 | Non-valve devices |
| ASTM F2514 | Finite element analysis of metallic vascular stents | Analysis practice and reporting | Polymer or bioresorbable behaviour |
| ASTM F2996 | FEA of non-metallic total hip replacements | Analysis practice and reporting | Knee, spine, trauma |
| ASTM F3161 | FEA of metallic knee and hip replacements | Analysis practice and reporting | Soft tissue and fixation |
| ISO 14971:2019 | Risk management for medical devices | The hazard and harm language used in the consequence rating | Model credibility itself |
| ISO 13485:2016 and the FDA QMSR | Quality management | Design verification and validation records, document control | Technical adequacy of a model |

The distinction that matters for a submission is binding requirement versus convention versus
guidance. In the United States, the Quality Management System Regulation, which replaced the
Quality System Regulation and took effect on 2 February 2026, makes design verification and
validation a requirement and incorporates ISO 13485:2016 by reference. It does not say that a
computational model is acceptable evidence. The FDA guidance is guidance: a sponsor may follow
another approach, but must then justify it. ASME V&V 40 is a recognised consensus standard, which
means a submission that follows it can cite it as evidence of a accepted practice but does not
create a legal obligation.

```text
  BINDING                         CONVENTION                    GUIDANCE
  -------                         ----------                    --------
  21 CFR 820 / QMSR:              ASME V&V 40-2018:             FDA CM&S guidance 2023:
  design V&V records              structure of the              nine steps, eight
  must exist and be               credibility argument          credibility factors
  controlled

  ISO 13485:2016 (via QMSR)       Sector ASTM and ISO           EU: no device-specific
  clause 7.3: design              standards: analysis           CM&S guidance; notified
  verification and                practice for a device         body assesses case by case
  validation planning             family

  ISO 14971:2019: risk             Reporting convention:         Pre-submission (Q-Sub)
  file must identify              state COU, model,             to agree the credibility
  hazards                         evidence, limits              plan before the study
```

For data, there is no standard interchange format for a device model, and this is a real gap. The
inputs are usually Digital Imaging and Communications in Medicine (DICOM) images; the geometry is
usually a surface mesh in STL or a volume mesh in a solver-specific format; the results are usually
fields in VTK or HDF5. None of these carries the provenance a reviewer needs: which segmentation,
which material assignment, which solver version, which parameter set. Provenance is therefore
maintained by document control, not by the file format, and a submission that cannot trace a result
back to a versioned input set will be treated as unreproducible regardless of its technical merit.

For the European Union, Regulation (EU) 2017/745 requires safety and performance under the general
safety and performance requirements of Annex I, and there is no device-specific guidance on
computational modeling equivalent to the FDA document. A notified body assesses a simulation as
part of the technical documentation, and in practice applies the same framework, often naming ASME
V&V 40. A sponsor planning an EU submission should not assume that an FDA-shaped credibility report
will be accepted unchanged, and should not assume it will be rejected either; the assessment is
case by case.
## 9. Industry practice and case evidence

### The clearest public case: HeartFlow FFRct

**Documented.** HeartFlow received a De Novo classification for its FFRct software on 26 November
2014, under DEN130045, creating regulation 21 CFR 870.1415, "Coronary Physiologic Simulation
Software Device", product code PJA, Class II. The device computes fractional flow reserve from a
computational fluid dynamics model built on a standard coronary computed tomography angiogram,
replacing an invasive pressure-wire measurement. The clinical basis was the HeartFlowNXT study,
with 254 patients analysed on an intention-to-diagnose basis and 484 vessels, reporting per-vessel
sensitivity of 83.5 percent and specificity of 85.8 percent against invasive fractional flow reserve
of 0.80 or less as the reference. The De Novo summary lists limitations including acute coronary
syndrome, recent myocardial infarction, prior bypass surgery, body mass index above 35, metallic
stents, pacemaker leads, prosthetic valves, significant arrhythmia, and heavy calcification, and it
lists non-clinical computational model validation among the special controls.

**Inference.** Three things follow, and they are inference rather than documented fact. First, the
device is a case where the model is the product rather than a design aid, so the highest
credibility burden applies by construction: there is no bench campaign to fall back on, because the
output is a physiological quantity in a living patient. Second, the validation that mattered was
against the clinical reference standard, not against the solver: the geometry extraction, the
boundary conditions, and the outlet model were validated jointly through their effect on a number
that could be measured invasively. Third, the limitation list is an applicability argument written
as a label: the excluded populations are those outside the envelope in which the model was
validated, and the exclusions are specific because a general statement would not have been
enforceable.

The case is often cited as evidence that simulation "replaces" trials. It does not. It shows that a
model can replace one specific measurement, in one specified population, with a documented
reference standard and a labelled envelope.

### Orthopaedic practice

**Documented.** The ASTM standards for finite element analysis of orthopaedic implants, F2996 for
non-metallic hips and F3161 for metallic knee and hip replacements, specify analysis practice and
what a report must contain. They exist because orthopaedic submissions routinely use analysis in
place of, or alongside, physical testing, and reviewers needed a common expectation.

**Inference.** In practice the argument in an orthopaedic 510(k) is usually a worst-case argument
rather than a population argument: the sponsor identifies the worst-case size and loading
combination, analyses that, and argues that other configurations are bounded by it. That is a
weaker claim than a population claim and it is usually the right one, because the decision is
whether the design is at least as safe as the predicate, not what the failure rate will be in
service. The failure mode of the worst-case argument is a misidentified worst case: the
configuration with the highest stress is not always the one with the lowest safety factor, because
the allowable also varies with geometry.

### What teams actually do

**Inference from practice, not from a survey.** Three patterns recur.

The credibility matrix is filled in at the end. A team builds a model for design, then the
regulatory team asks for a credibility assessment, and the matrix is completed retrospectively to
match the evidence that exists. The resulting document is internally consistent and describes a
plan nobody followed. Reviewers have learned to ask for the dates.

The cohort is generated once and reused. A shape model is built for one programme, then reused for
a label expansion it was never designed to cover, because regenerating it is expensive and the
existing one "looks representative". Section 6 is that failure with numbers.

The sensitivity analysis is one-at-a-time. Varying one input at a time is cheap, intuitive, and
wrong when inputs interact. It also tends to vary the inputs that are easy to vary, which are
rarely the ones that matter.

### The research literature

Three 2024 papers frame the current state, and all three are about the population rather than the
solver. Pathmanathan and colleagues, in *PLOS Computational Biology*, set out a credibility approach
for in-silico clinical trials that separates subject-specific from synthetic cohorts, adds
population-based validation as a credibility factor in its own right, and names the risk that a
virtual cohort shows less variability than the real population. Aycock and colleagues, in
*Frontiers in Medicine*, propose a hierarchical framework in which each submodel is assessed before
the assembled trial. Gevertz and Wares, in the *Bulletin of Mathematical Biology*, quantify how the
choice of patient generation technique changes cohort heterogeneity and therefore the predicted
trial outcome. The common finding: the modelling choice that most affects the answer is the
population, and the population is the part least often validated.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Validation set used for calibration | Reported agreement tighter than reference repeatability | Parameters or model form tuned on data later called validation | Inflated credibility; a claim that will not replicate | Freeze and hash the validation set before fitting; report repeatability |
| Model form selected on validation error | Validation set size is small and several forms were tried | Each comparison is a noisy estimate; the minimum is biased low | Optimistic error estimate | Pre-specify the form; use a third block if selection is unavoidable |
| Cohort truncated at plus or minus two SD | Simulated variance below observed variance | Mode clipping removes the tails | The patients most likely to fail are absent | Compare marginals with an observed population; sample the tails explicitly |
| Shape model trained on device recipients | Cohort matches the trial population but not the referral population | Selection into treatment is baked into the training set | Extrapolation to a wider label fails | Train on a representative sample or restrict the label |
| Pooled coverage reported | One number for the whole cohort | Tails diluted by the bulk | Boundary failures invisible | Report coverage and outcomes by decile and subgroup |
| Boundary conditions borrowed | Inlet flow or fixation taken from a textbook case | The condition belongs to a different anatomy | Correct physics, wrong problem | Validate the boundary condition or vary it in sensitivity analysis |
| Material properties without variance | Single literature value, no range | Population variability in tissue ignored | Interval on the answer too narrow | Propagate a distribution; test the extremes |
| Right answer for the wrong reason | Agreement across the range but residuals trend with a covariate | Two errors cancelling | Fails on extrapolation | Inspect residual structure, not only magnitude |
| Surrogate used outside its training envelope | Cohort covariates beyond the training range | Regression extrapolates silently | Unbounded error with no warning | Enforce a domain check before evaluation |
| One-at-a-time sensitivity | Inputs varied over convenient rather than plausible ranges | Interactions missed | Importance ranking wrong | Use a global method with interaction terms |
| Deterministic single run | No interval reported | Numerical and parameter uncertainty unquantified | False precision | Propagate uncertainty; carry discretisation error |
| Mesh convergence from two meshes | Refinement table with two rows | Cannot separate convergence from coincidence | Discretisation error unknown | Three levels with a stated refinement ratio |
| Claim promotion | Same model, new slide deck, higher-stakes decision | Influence increased without new evidence | Unsupported claim | Re-rate model risk whenever the intended use changes |
| Unarchived inputs | Results cannot be regenerated from version control | Code, mesh, or parameters overwritten | Submission not reproducible | Version and hash inputs; archive the run |
| Unit or scale error | Stress or flow off by a round factor | Millimetres against metres, pascals against mmHg | Gross error that still looks plausible | Dimensional check against an analytic case |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The context of use captures the decision | Every later adequacy judgement is relative to it | The decision changes after the model is built | Re-read the COU at each stage gate; re-rate if it changes |
| The model class matches the dominant physics | Mechanistic models are chosen when mechanism is known | A statistical model used where mechanism matters, or the reverse | Compare a second class on a subset and check the ranking |
| The calibration parameters are identifiable | Fitting requires the data to constrain the parameters | More parameters than independent measurements | Check the sensitivity matrix; report parameter uncertainty |
| The validation data are independent of calibration | Independence is what makes validation informative | Shared specimens, shared patients, shared site | Audit the split; compare agreement with repeatability |
| The reference measurement is accurate enough | A model cannot be validated beyond the reference | Reference bias larger than the model error | Report reference uncertainty and bias separately |
| The virtual cohort represents the labelled population | The trial population is the cohort | Truncation, training-set selection, covariate shift | Coverage statistics against an observed population, by tail |
| The validation envelope covers the claim | Applicability is an argument, not an assumption | Extrapolation in geometry, size, or population | Map claim covariates onto the envelope and name the extrapolations |
| Sources of uncertainty are independent | Combining in quadrature requires it | Correlated inputs, shared model error | Test with a correlated sampling scheme |
| Numerical error is small relative to the effect | Otherwise the answer is an artefact | Coarse mesh, large time step, under-converged solve | Grid convergence index at the production setting |
| The model form is adequate | Every model is wrong in some respect | Structural error larger than the claimed effect | Residual analysis against covariates; alternative forms |

Two limits apply to everything above.

The framework tells you how to argue, not what answer to reach. Two competent teams can apply ASME
V&V 40 to the same model and reach different adequacy judgements, because influence and consequence
are judgements. That is a feature: the framework forces the judgement into the open where it can be
contested.

Nothing here establishes that a simulation is safe. A credible model supports a decision; it does
not replace the decision-maker, and it does not remove the residual risk that the model is wrong in
a way the evidence cannot detect. That residual belongs in the risk file.
## 12. Exercises

**Exercise 1 — Write a context of use (recall and application).**
Take the fixation plate from section 1, in which one screw hole moves 4 mm and the thickness drops
from 3.5 to 3.2 mm. Write a context-of-use statement for a finite element model intended to support
the design change in a 510(k).
*Deliverable:* one page containing the COU in three sentences, an out-of-scope list of at least
five items, and the decision the model informs.
*Acceptance:* the COU names the output quantity with units, the population, the loading conditions,
and the model version; a reader can name one plausible use that the COU excludes; the decision is
stated as a question with a named decision-maker.
*Time:* 45 minutes.

**Exercise 2 — Compute a cohort coverage statistic (calculation).**
You are given two cohorts on one covariate: observed n = 200, mean 24.0 mm, SD 2.0 mm; simulated
n = 400, mean 23.9 mm, SD 1.6 mm. Compute the standardised mean difference and the variance ratio,
and state whether the cohort passes a coverage check at an SMD threshold of 0.10.
*Deliverable:* the two calculations, with the formula written out, and a three-sentence
interpretation.
*Acceptance:* SMD within 0.005 of -0.055; variance ratio within 0.005 of 0.80; the interpretation
states that the mean check passes while the variance check fails, and names one consequence for a
tail claim.
*Time:* 30 minutes.

**Exercise 3 — Compute a rate with an interval (calculation).**
A simulated cohort of 1,000 virtual patients yields 918 successes. Compute the Wilson 95 percent
interval and state the smallest number of virtual patients needed to halve the interval width.
*Deliverable:* the interval, the half-width, and the required n.
*Acceptance:* interval within 0.002 of 0.899 to 0.933; recognition that shrinking the interval by
half requires about four times the sample, so roughly 4,000.
*Time:* 25 minutes.

**Exercise 4 — Find the leak (critique).**
A report states: "The model was validated against bench measurements with a root-mean-square error
of 0.18 N. The bench method has a repeatability of 0.25 N." The report also says four model forms
were compared and the best was selected.
*Deliverable:* a 300-word critique naming the leak, the arithmetic evidence, and the two questions
you would ask the author.
*Acceptance:* identifies that 0.18 N is below the 0.25 N repeatability and therefore cannot be an
independent validation result; identifies model-form selection as the likely mechanism; proposes
asking for the pre-specified form and the untouched hold-out set.
*Time:* 40 minutes.

**Exercise 5 — Estimate discretisation error (application).**
Peak strain at three mesh levels is 1.42 percent, 1.38 percent, and 1.36 percent, with a refinement
ratio of 2. Estimate the grid convergence index and state whether the finest mesh is adequate for a
claim that depends on a 5 percent difference.
*Deliverable:* the observed order, the Richardson extrapolate, the GCI, and a one-paragraph
decision.
*Acceptance:* observed order computed as approximately 2 (values within 0.3); GCI below 2 percent;
the decision states that a 5 percent effect is resolvable but a 1 percent effect is not.
*Time:* 60 minutes.

**Exercise 6 — Design a paediatric cohort (design).**
A device is cleared for adults with annuli from 21 to 27 mm. A paediatric indication is proposed for
annuli from 14 to 19 mm. No paediatric trial is feasible. Design the cohort and the evidence plan.
*Deliverable:* a two-page plan covering cohort source, sampling, coverage comparison, validation
strategy without clinical data, and the claim you would and would not make.
*Acceptance:* states how paediatric anatomy is sourced and why it is representative; names a
non-clinical validation route and its limits; proposes a narrower initial claim; identifies the
residual risk in writing.
*Time:* 90 minutes.

**Exercise 7 — Red-team a credibility report (critique).**
Take any published computational study of a device, or the credibility section of a public 510(k)
summary that includes modeling, and attempt to falsify its credibility argument.
*Deliverable:* a one-page red-team memo with five specific objections, each tied to a credibility
factor and to a step in section 5.
*Acceptance:* each objection names the missing evidence rather than expressing doubt; at least two
objections concern population or applicability rather than the solver; the memo states which
objections the authors could answer cheaply.
*Time:* 60 minutes.

**Exercise 8 — Write the decision memo (design).**
For the case in section 6, write the memo that step 12 requires: what the model supports, what it
does not, and what would have to change.
*Deliverable:* a one-page memo addressed to the programme decision-maker.
*Acceptance:* distinguishes the supported claim (existing label) from the unsupported one
(expansion); cites the optimism and the coverage figure with their derivations; proposes at least
two routes to support the expansion and states what evidence each requires; contains no claim not
traceable to a number in the case.
*Time:* 45 minutes.

## 13. Self-check answers

**Exercise 1.** A strong COU reads: "The model predicts peak von Mises stress and fatigue safety
factor in the revised plate (3.2 mm, hole repositioned 4 mm) under the ISO 7206-4 loading case for a
patient population of 55 to 75 years and body mass index 18 to 35, to support a determination that
the revised design is at least as safe as the predicate. It is not used for wear prediction, for
screw purchase in osteoporotic bone, or for any configuration other than the two plate lengths
modelled." The out-of-scope list should include at least: other plate lengths, bone quality outside
the stated range, cyclic loading other than the specified profile, and any claim about clinical
outcomes.

**Exercise 2.** The pooled standard deviation is the square root of (2.0 squared plus 1.6 squared)
divided by 2, which is the square root of 3.28, or 1.811 mm. The standardised mean difference is
(23.9 - 24.0) divided by 1.811, which is -0.055. The variance ratio is 1.6 divided by 2.0, or 0.80.
The mean check passes; a variance ratio of 0.80 means the simulated cohort has 20 percent less
spread, so a claim about the 95th percentile is being made on a population that under-represents
the upper tail.

**Exercise 3.** With 918 of 1,000, the proportion is 0.918. The Wilson interval is 0.8994 to 0.9334,
a half-width of 1.70 percentage points. Halving the half-width requires roughly four times the
sample, because the half-width falls with the square root of n, so about 4,000 virtual patients.
Note that this buys precision, not representativeness: a biased cohort estimated precisely is still
biased.

**Exercise 4.** The leak is that 0.18 N is below the 0.25 N repeatability of the reference method.
A model cannot agree with a measurement more closely than the measurement agrees with itself unless
it has fitted the noise. The stated comparison of four model forms supplies the mechanism: selecting
the best of four on the same data that is then reported as validation converts that set into a
second calibration set, and the minimum of four noisy estimates is biased low. The questions to ask
are which form was pre-specified before the comparison, and whether any data remained untouched
after selection.

**Exercise 5.** With values 1.42, 1.38, and 1.36 at a refinement ratio of 2, the differences are
0.04 and 0.02, giving an observed order of about 2.0, consistent with a second-order element. The
Richardson extrapolate is approximately 1.353 percent. The grid convergence index on the finest pair
is about 1.5 times the relative difference of 0.015 divided by (2 squared minus 1), or roughly 0.7
percent. A 5 percent effect is resolvable; a 1 percent effect is marginal and would need a finer
mesh.

**Exercises 6 to 8** are assessed against a rubric rather than an answer.

| Criterion | Weight | What earns full credit |
|---|---:|---|
| Names the decision and the claim boundary | 25% | A claim that a reviewer could falsify, and a stated narrower fallback |
| Separates evidence types correctly | 25% | Verification, calibration, validation, and applicability are distinguished, not merged |
| Quantifies the population problem | 20% | Coverage or bias computed, not asserted, with the denominator stated |
| States residual risk and limits | 20% | Names what the evidence would not detect |
| Isauditable | 10% | Dates, versions, data provenance, and pre-specified criteria |

## 14. Interview questions

**1. What makes a simulation credible?** Strong: credibility is relative to a context of use and a
decision, established by evidence mapped to credibility factors, with effort proportional to model
risk. Weak: treats credibility as model quality or as the solver's reputation.

**2. Verification versus validation?** Strong: verification asks whether the equations are solved
correctly; validation asks whether the right equations were solved, judged against independent
measurement in the context of use. Weak: uses the terms interchangeably.

**3. You are told the model was validated. What do you ask first?** Strong: which data, collected
when, and whether any of it was used in calibration or model selection. Weak: asks for R-squared.

**4. How would you know a validation set leaked?** Strong: agreement tighter than the reference
method's repeatability, a validation set smaller than the number of model choices tried, or residual
structure that vanishes when a covariate is added. Weak: says leakage is undetectable after the fact.

**5. What is wrong with 10,000 virtual patients generated from 200 scans?** Strong: the effective
information content is 200 scans; the sample size buys precision in the estimated rate, not
representativeness, and the tails are whatever the shape model allowed. Weak: says 10,000 is large
and therefore reliable.

**6. How much verification is enough?** Strong: enough that numerical error is below the effect size
the claim depends on, set from model risk. Weak: says "until it converges".

**7. When can a model support a labelling claim?** Strong: when evidence meets the requirement set by
high influence and consequence, validation covers the labelled population, and applicability is
documented in advance. Weak: says when it agrees with bench data.

**8. What if the model passes on the labelled range but fails outside it?** Strong: narrow the claim,
document the boundary, state what evidence would support the wider claim. Weak: reports the pooled
result and says nothing about the boundary.

**9. Why is a one-at-a-time sensitivity analysis often inadequate?** Strong: it misses interactions
and usually varies convenient rather than plausible ranges, so the ranking can invert under a global
method. Weak: says it is standard practice.

**10. Design decision versus regulatory claim?** Strong: the decision needs a correct ranking with
internal cost of error; the claim needs a defensible bound including the tail, with public cost of
error. Weak: sees one model with different paperwork.

## 15. Key terms

- **Applicability** — the argument that a validated model may be used beyond its validation conditions.
- **ASME V&V 40-2018** — the consensus credibility framework for device models.
- **Calculation verification** — estimating numerical error in the quantity of interest.
- **Calibration** — fitting model parameters to a reference dataset.
- **Code verification** — showing the implementation solves the intended equations.
- **Context of use (COU)** — the decision, output, population, and conditions the model serves.
- **Coverage** — how well a virtual cohort spans the target population's covariate distribution.
- **Credibility** — trust that a model supports a specific decision in a specific context.
- **Credibility factor** — a category of evidence against which credibility is judged.
- **Decision consequence** — the severity of harm if the model-informed decision is wrong.
- **Envelope of validation** — the conditions actually covered by validation evidence.
- **Grid convergence index (GCI)** — a Richardson-based estimate of discretisation error.
- **In-silico trial (ISCT)** — a device study conducted in a computational patient population.
- **Mechanistic model** — a model built from governing equations and constitutive laws.
- **Model influence** — how much the model drives the decision.
- **Model risk** — influence combined with decision consequence.
- **Repeatability** — agreement between repeated measurements by the same method; the floor on validation.
- **Statistical shape model (SSM)** — a parameterised description of anatomical variation.
- **Surrogate model** — a fast approximation of an expensive model, valid only in its training range.
- **Uncertainty quantification (UQ)** — propagating input and model uncertainty to an output interval.
- **Validation** — comparison with independent measurement in the context of use.
- **Verification** — establishing that the model is solved correctly.
- **Virtual cohort** — a set of virtual patients used to represent a population.
- **Virtual patient** — a computational representation of one patient used as a test case.

## 16. Sources

1. US Food and Drug Administration. *Assessing the Credibility of Computational Modeling and
   Simulation in Medical Device Submissions*. Final guidance, November 2023. Docket
   FDA-2021-D-0980. https://www.fda.gov/regulatory-information/search-fda-guidance-documents
2. ASME. *Assessing Credibility of Computational Modeling through Verification and Validation:
   Application to Medical Devices*. ASME V&V 40-2018. American Society of Mechanical Engineers.
3. ASME. *Standard for Verification and Validation in Computational Solid Mechanics*,
   ASME V&V 10-2019; and *Standard for Verification and Validation in Computational Fluid Dynamics
   and Heat Transfer*, ASME V&V 20-2009 (R2021).
5. US Food and Drug Administration. De Novo classification DEN130045, HeartFlow FFRct, 26 November
   2014, 21 CFR 870.1415, product code PJA.
6. Pathmanathan P, Aycock K, Badal A, Bighamian R, Bodner J, Craven BA, Niederer S. Credibility
   assessment of in silico clinical trials for medical devices. *PLOS Computational Biology*.
   2024;20(8):e1012289.
7. Aycock KI, Battisti AJ, Peterson A, et al. Toward trustworthy medical device in silico clinical
   trials: a hierarchical framework for establishing credibility and strategies for overcoming key
   challenges. *Frontiers in Medicine*. 2024;11:1433372.
8. Gevertz JL, Wares JR. Assessing the role of patient generation techniques in virtual clinical
   trial outcomes. *Bulletin of Mathematical Biology*. 2024;86:119.
9. International Organization for Standardization. *ISO 5840-3:2021, Cardiovascular implants —
   Cardiac valve prostheses — Part 3: Heart valve substitutes implanted by transcatheter
   techniques*.
10. ASTM International. *F2996, Standard Practice for Finite Element Analysis of Non-metallic
    Orthopaedic Hip Replacements*; *F3161, Standard Test Method for Finite Element Analysis of
    Metallic Orthopaedic Hip and Knee Replacements*; *F2514, Standard Guide for Finite Element
    Analysis of Metallic Vascular Stents*.
11. International Organization for Standardization. *ISO 14971:2019, Medical devices — Application
    of risk management to medical devices*.
12. International Electrotechnical Commission. *IEC 63719-1 (in development), Medical devices — In
    silico technologies — Part 1: Terminology and context*; *IEC 63719-2 (in development), Part 2:
    Credibility assessment of first-principles based methods*. Status checked October 2026; neither
    is published.
13. Oberkampf WL, Roy CJ. *Verification and Validation in Scientific Computing*. Cambridge
    University Press, 2010. See also Roache PJ, *Verification and Validation in Computational Science
    and Engineering*, Hermosa Publishers, 1998.
14. Taylor CA, Fonte TA, Min JK. Computational fluid dynamics applied to cardiac computed
    tomography for noninvasive quantification of fractional flow reserve. *Journal of the American
    College of Cardiology*. 2013;61(22):2233-2241.

Sources 1 to 5 and 9 to 12 were checked against the issuing body's page on 27 October 2026. Sources
6 to 8 and 13 to 14 are cited by title and venue; confirm the version before relying on a figure.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`10_post-market-surveillance.md`](10_post-market-surveillance.md)
- Next topic: [`12_udi-and-pms-analytics.md`](12_udi-and-pms-analytics.md)
- Related: [`06_iso-14971-risk-management.md`](06_iso-14971-risk-management.md) — the hazard and harm
  language used for the decision-consequence rating.
- Related: [`09_the-evidence-package.md`](09_the-evidence-package.md) — where a credibility
  assessment report sits in a submission.
- Root survey file: [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
- Labs and datasets: [`../../labs/`](../../labs/)
- Calibration and decision-curve lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
