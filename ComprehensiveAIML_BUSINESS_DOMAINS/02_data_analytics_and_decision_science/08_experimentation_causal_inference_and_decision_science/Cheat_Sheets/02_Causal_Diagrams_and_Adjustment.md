---
collection: "08_experimentation_causal_inference_and_decision_science"
file_id: "02"
topic: "Causal Diagrams and Adjustment"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "A change to the estimand, the assumed graph, the target population, or the adjustment-set policy invalidates every number in this sheet"
notation_style: "plain-text"
prerequisites: ["01_Experiment_Design_and_Guardrails.md", "conditional expectation and probability", "ordinary least squares with one binary regressor"]
pattern_tags: ["causal-inference", "statistics-experimentation", "observational-study"]
source_status: "OPEN"
owner_role: "causal inference and experimentation lead"
source_index: "INDEX.md"
---

# Causal Diagrams and Adjustment

## 1. File Header

> **Purpose:** Decide *which variables to condition on* when a randomised experiment is unavailable, using a graph you must defend rather than a list of covariates chosen by habit. The sheet turns a written causal claim into a named estimand, a drawn graph, an adjustment set with a validity argument, and an estimate whose uncertainty and sensitivity are reported alongside it.
> **Prerequisites:** [Experiment Design and Guardrails](01_Experiment_Design_and_Guardrails.md) for estimand and guardrail vocabulary; comfort with conditional expectation, total probability, and a single-regressor OLS slope; ability to read an arrow diagram as a statement about data-generating processes rather than about correlation strength.
> **Key outcome:** Given a decision, a treatment definition, an outcome definition, and a defensible graph, you can derive the adjustment set, prove it satisfies the backdoor criterion, compute a standardised effect with worked arithmetic, detect over-adjustment and collider conditioning, and state the sensitivity of the conclusion to an unmeasured confounder.
> **This file does NOT cover:** how to randomise, size, or monitor a live experiment (that is file 01); matching estimators, difference-in-differences, and synthetic control construction (file 03); heterogeneous treatment effect estimation and uplift modelling (file 04). Instrument construction, regression discontinuity implementation details, and structural equation fitting are named here only to the depth needed to choose between them.
> **Level:** 2 — Core Methods. Section 4.3 and Section 10 are the parts a practitioner returns to under pressure; Sections 3 and 4.1 are the parts that must be understood first.
> **Version / reviewed:** 2.0 / not independently reviewed (`reviewed: null`). The review trigger is any change to the estimand, graph, or target population.
> **Evidence:** Derivations are DERIVED from the stated graph and the Markov/faithfulness assumptions. All arithmetic in Sections 5, 8, and 9 was checked by hand twice and is labelled VERIFIED for arithmetic only; every data table is ILLUSTRATIVE and describes no real company, product, or population. Claims about published methods are SOURCED to the primary literature named in Section 10 and Section 13 of the collection index. Where a quantity depends on an assumption nobody has measured, it is labelled OPEN.
> **Retrieval cue:** If a colleague says "we controlled for everything we had", go to Section 8 failure F-03 (over-adjustment) and Section 8 failure F-05 (collider conditioning) before reading any estimate.

Every observational comparison in a business setting answers a question of the form "what would the outcome have been under the other action, holding the population fixed?" The observed data never contain that counterfactual, so the answer is produced by a chain of assumptions: a graph that says which variables cause which, a criterion that says which variables may be conditioned on, an estimator that implements the conditioning, and a sensitivity statement that says how much the conclusion depends on the parts of the graph nobody can verify. This sheet teaches that chain as a single auditable artefact. The graph is not decoration added after the regression; it is the thing that decides whether the regression coefficient has a causal reading at all.

The practical failure this sheet prevents is expensive and quiet. A team conditions on a post-treatment variable because it "improves the model fit", the coefficient moves in the desired direction, the deck ships, and the decision is wrong. Nothing in the output announces the error: standard errors shrink, R-squared rises, and the point estimate becomes more decisive precisely because the adjustment destroyed the variation that identified the effect. Causal diagram reasoning is the only routinely available check that catches this before the decision, and it catches it by inspecting the graph rather than the fit.

A second failure is the mirror image. A team refuses to adjust for anything because "correlation is not causation", runs a raw comparison between adopters and non-adopters, and reports a number inflated by three to ten times the real effect because the adopters were already the high-intent segment. The first failure over-controls; the second under-controls. Both are graph errors, both are invisible in the regression output, and both are cheap to detect with the procedures in Sections 4 and 8.

| Decision the reader must make | Wrong default people use | What this sheet substitutes |
|---|---|---|
| Which covariates enter the model | Everything available, or a stepwise search on p-values | A backdoor-valid set, derived from the graph and stated by name |
| Whether a coefficient is causal | It is, because the model is "controlled" | It is causal only if the graph plus criterion plus estimator hold |
| Whether to condition on a post-treatment variable | Yes, it reduces residual variance | Only if the graph shows it is not a mediator, collider, or proxy |
| Which population the effect applies to | Whoever is in the data | An explicitly named target population and matching weights |
| How to report an unverifiable assumption | Omit it | Direction and magnitude of the bias it would produce, plus an E-value |
| What to do when a stratum is empty | Drop it and continue | Declare a positivity violation and restrict the estimand |

The unit of analysis in this sheet is the *decision unit*: one row per entity for which the action would be taken, at the time the action would be taken. That sentence is doing real work. If the action is a checkout redesign, the decision unit is a session, and a customer appearing in four sessions contributes four decision opportunities — but the confounding structure operates at the customer level as well as the session level, which means the graph has to distinguish these grains. A graph drawn at the wrong grain produces an adjustment set that is invalid at the grain actually analysed, and no amount of statistical care repairs it afterwards.

**Takeaway:** the adjustment set is a graph-derived claim with a proof obligation, not a modelling preference, and it must be derived at the same grain as the decision.

## 2. Concept Map / Mental Model

The whole method fits on one screen. Read the map top to bottom: the graph is written first, the criterion is applied to the graph, the estimator is chosen to implement the criterion, and the sensitivity analysis states what survives if the graph is wrong.

```text
[Decision + estimand: which contrast, for which population, at which grain]
                              |
                              v
[Draw the graph G]  <-- domain knowledge, timing, mechanisms, prior evidence
                              |
                              v
[Add the action node X and outcome node Y] --(X and Y not in G?)--> [Redraw: the
                              |                                      graph must
                              v                                      contain them]
[Apply a criterion to G]  ---- backdoor / front-door / IV / do-calculus
        |                                    |
        | no admissible set                  | admissible set Z found
        v                                    v
[Change design: instrument,   [Estimate: standardisation, IPW, PS matching,
 front-door design, panel,     doubly robust AIPW, outcome regression]
 or run an experiment]                        |
        |                                     v
        |                          [Diagnostics: positivity, balance,
        |                           overlap, effective sample size]
        v                                     |
[Report "not identified under                 v
 the stated graph" + what      [Sensitivity: E-value, OVB bias bound,
 evidence would identify it]    negative control, placebo outcome]
                                              |
                                              v
                                   [Decision, with the assumption
                                    it depends on named in the deck]
```

Three ideas carry the structure, and they must be separated cleanly because they fail independently.

*The first idea is that a graph encodes conditional independences.* A directed acyclic graph over the variables, with arrows meaning "is a direct cause of", implies a set of statements of the form "these two variables are independent given those". The implication runs from graph to data, one way: a graph that implies an independence the data violate is wrong, but a graph that fits the data is not thereby correct, because many different graphs imply the same independences. This asymmetry is the reason causal discovery from observational data alone cannot, in general, recover the graph, and it is why the graph in a serious analysis is written from domain knowledge and defended in prose rather than estimated.

*The second idea is that correlation flows along paths, and conditioning opens and closes valves on those paths.* Association between X and Y travels along every unblocked path between them. A path is blocked if it contains a chain or fork whose middle node is conditioned on, and a path is open if it contains a collider whose middle node is conditioned on. Adjustment works by choosing a set that closes every path that enters X through an arrowhead, while leaving the directed path from X to Y untouched. Once the reader internalises "colliders are valves that open when you condition, chains and forks are valves that close when you condition", most of the criterion becomes obvious rather than memorised.

*The third idea is that identification is a property of the graph, not of the sample size.* No amount of data identifies an effect when the graph contains a structure that leaves the counterfactual entangled with the observed distribution — the canonical case being an unmeasured variable causing both treatment and outcome with no other route to break the entanglement. The honest output in that case is "not identified", plus a description of the design change that would identify it. Reporting a wide confidence interval instead is a category error: the interval describes sampling noise around a quantity that is measuring the wrong thing entirely.

The map below organises the six question types a reader actually brings to this material, so that a question can be routed without reading the whole sheet.

| Reader's question | Where the answer lives | The one-line answer |
|---|---|---|
| "Which variables should I control for?" | Section 4.2, criterion BD-1 | The set that blocks every backdoor path and contains no descendant of the treatment |
| "Is controlling for more always safer?" | Section 8, failures F-03 to F-05 | No: mediators, colliders, and their descendants remove or invert the effect |
| "I cannot measure the confounder — now what?" | Section 4.3, blocks CD-09 to CD-11 | Front-door, instrument, or panel design; otherwise report non-identification |
| "How do I compute the adjusted number?" | Section 4.2 and Section 5 | Standardise stratum-specific outcomes to one common weight distribution |
| "How wrong could I be?" | Section 4.3 block CD-13, Section 8 | E-value plus an omitted-variable bias bound in outcome units |
| "Which method for this dataset?" | Section 7 | Match the method to the assumption you can defend, not to the trend |

A useful discipline when starting: write the graph before touching the data. Teams that extract the analytic table first tend to draw a graph containing exactly the columns that exist, which converts a scientific question into an artefact of the warehouse schema. Draw the graph from the mechanism — who is exposed, why, when, and what else moves at the same time — then check which nodes are measurable. Every node you cannot measure becomes a sensitivity question rather than a silent omission.

A second discipline: give every node an explicit time stamp. Confounding is a statement about what is determined before the action. A covariate recorded after the action cannot be a pre-treatment confounder by definition, whatever its correlation with the treatment looks like; it is at best a mediator and at worst a collider. The single most common error in production analyses is a "control for everything" query that pulls a wide feature table with no as-of-date, silently sweeping post-treatment columns into the model. The as-of-date is not hygiene; it is the mechanism that makes the backdoor criterion applicable at all.

**Takeaway:** route the question to a criterion first — the estimator choice is a downstream detail that cannot repair an invalid set.

## 3. Foundational Definitions

Each definition below is written to be executable: it names the object, its grain, its timing, and a nonexample that shows what the definition excludes. Where a symbol is used later in a formula, it appears in the notation table at the end of this section.

**D-1. Unit (decision unit).** The entity to which one action is applied at one point in time: a session, an account, a store-week, a member-month. Its grain must equal the grain of the outcome measurement. *Nonexample:* a customer-level action with a session-level outcome count, where one customer's treatment status is duplicated across four outcome rows without any weighting statement.

**D-2. Treatment (action) variable X.** A variable whose value the analyst contemplates setting, with at least two levels and a stated assignment mechanism. For a binary action, X=1 means the action was applied. *Nonexample:* a continuous engagement score used as "treatment", where no intervention could set the score to a chosen value; that is a predictor, not an action.

**D-3. Outcome Y.** The measured consequence, with numerator, denominator, window, and censoring rule fixed before analysis. *Nonexample:* "revenue" without a definition of returns, refunds, and the recognition window, which makes two analyses incomparable while both are labelled revenue.

**D-4. Potential outcome Y(x).** The value Y would take for a given unit if X were set to x, possibly contrary to fact. It is a property of the unit and the intervention, not of the sample. *Nonexample:* the observed outcome among units that happened to receive x; that is E[Y | X=x], a different quantity whenever treatment is not randomised.

**D-5. Causal estimand.** A functional of the joint distribution of potential outcomes, such as ATE = E[Y(1) − Y(0)], ATT = E[Y(1) − Y(0) | X=1], or a quantile contrast. Chosen before estimation. *Nonexample:* "the effect of the programme", which names no contrast, no population, and no scale.

**D-6. Structural causal model (SCM).** A set of assignments X_i := f_i(Pa_i, U_i) for each node, where Pa_i are the observed parents and U_i are exogenous disturbances assumed mutually independent. The SCM induces the graph and the observational distribution simultaneously. *Nonexample:* a regression equation fitted to data and then relabelled "structural" without any claim about which variables would change if another were intervened on.

**D-7. Directed acyclic graph (DAG).** A graph whose edges are directed and which contains no directed cycle. Acyclicity encodes that no variable is its own cause through any chain. *Nonexample:* a feedback diagram with a cycle, drawn as an illustration but used as if it were a DAG for path analysis.

**D-8. Path.** A sequence of adjacent edges between two nodes with no node repeated, ignoring edge direction. A *directed path* follows arrows consistently. Path counting matters because association flows along paths, not along edges. *Nonexample:* calling a single edge a "path" and concluding that the number of paths equals the number of arrows.

**D-9. Collider.** A node on a path with two incoming arrows (→ C ←). Conditioning on a collider, or on a descendant of one, *opens* the path. *Nonexample:* treating a collider like any other covariate and adding it to the model because it correlates with the outcome.

**D-10. Chain and fork.** A chain is A → M → B; a fork is A ← M → B. Conditioning on the middle node of either *closes* the path. *Nonexample:* assuming that a fork's middle node is "the same kind of variable" as a collider, which reverses every subsequent decision.

**D-11. d-separation.** Node sets A and B are d-separated by Z if every path between them is blocked given Z. Blocking is defined locally by the chain/fork/collider rules. d-separation in the graph implies conditional independence in every distribution Markov and faithful to it. *Nonexample:* checking only the shortest path and declaring separation.

**D-12. Backdoor path.** A path from X to Y whose first edge points into X (an arrowhead at X). Such a path carries non-causal association and must be blocked. *Nonexample:* counting the directed path X → M → Y as a backdoor path and blocking it, which removes the effect of interest.

**D-13. Adjustment set Z.** A set of observed, pre-treatment variables conditioned on in the estimator. It is *admissible* if it satisfies a stated criterion relative to (X, Y, G). *Nonexample:* calling any collection of covariates an "adjustment set" without a criterion check.

**D-14. Confounder.** Informally, a variable on an open backdoor path. Formally, the definition is criterion-relative: what matters is whether the set blocks the backdoor paths, not whether an individual variable is "a confounder". This distinction matters because a variable can be a confounder in one graph and a neutral bystander in another, and because some valid sets contain no variable that would individually be called a confounder. *Nonexample:* searching for "the confounder" as a single variable when the confounding is distributed across a set.

**D-15. Over-adjustment (bad control).** Conditioning on a descendant of X, including mediators, colliders on a path from X to Y, and proxies of X. It changes the estimand, usually toward zero for mediators and away from zero for colliders. *Nonexample:* "controlling for engagement" when engagement is a consequence of the feature being evaluated.

**D-16. Positivity (overlap).** For every covariate stratum that receives positive weight in the estimand, both treatment levels occur with positive probability. *Nonexample:* a stratum where all units were treated, handled by deleting the stratum and reporting the remainder as if it were the whole population.

**D-17. Consistency.** The observed outcome for a unit with X=x equals Y(x). It fails under versions of treatment: two "treated" units receiving materially different interventions. *Nonexample:* pooling a soft-launch variant and a full-rollout variant under one binary treatment flag.

**D-18. SUTVA (no interference, no hidden versions).** One unit's outcome does not depend on other units' treatment assignment, and there is a single well-defined version of each treatment level. *Nonexample:* a referral incentive evaluated at member level when treated members recruit their untreated neighbours, so the control group is partly treated.

**D-19. Faithfulness.** The distribution exhibits exactly the independences implied by the graph, with no accidental cancellations. Needed for converse directions of d-separation reasoning, not for the forward direction used in adjustment. *Nonexample:* two paths of exactly equal magnitude with opposite sign producing independence in the data although both paths exist.

**D-20. Identifiability.** A causal quantity is identifiable from the observed distribution plus the graph if it can be written as a functional of that distribution alone. *Nonexample:* asserting a point estimate is "the effect" when the graph leaves the counterfactual entangled; the correct status is "not identified", which is a finding rather than a failure.

**D-21. Target population and standardisation weights.** The population to which the reported effect refers, implemented as a weight distribution over covariate strata w(z). Different weightings of the same stratum-specific effects are different estimands, not different estimates of one estimand. *Nonexample:* averaging stratum effects with the treated group's covariate distribution and labelling the result "the population effect".

**D-22. Instrument.** A variable that affects the outcome only through the treatment and is independent of the unmeasured confounders. Validity is a graph claim, not a testable property in general. *Nonexample:* a variable correlated with the treatment but plausibly affecting the outcome directly, reported as an instrument because a first-stage F statistic was large.

The notation table is the contract for the rest of the sheet. Every symbol used in Section 4 appears here with its type and range, so a formula can be read without hunting.

| Symbol | Meaning | Type / units | Domain | First substantive use |
|---|---|---|---|---|
| X | Treatment, binary unless stated | scalar in {0,1} | two levels | 4.1 |
| Y | Outcome, binary or continuous | scalar | as defined | 4.1 |
| Z, z | Adjustment set and an index over its strata | vector; strata indexed | observed, pre-treatment | 4.2 |
| U | Unmeasured disturbance / latent confounder | scalar or vector | unobserved | 4.1 |
| G | DAG over observed and latent nodes | graph | acyclic | 4.1 |
| Pa_i | Observed parents of node i in G | set of nodes | — | 4.1 |
| P(·) | Observational probability or density | probability | sums to 1 | 4.1 |
| P(· \| do(X=x)) | Interventional distribution | probability | sums to 1 | 4.1 |
| E[·] | Expectation | outcome units | — | 4.1 |
| e(z) | Propensity score P(X=1 \| Z=z) | probability | [0,1] | 4.2 |
| w | Weight (IPW or standardisation) | dimensionless | positive | 4.2 |
| ρ | Risk ratio | dimensionless | > 0 | 4.3 |
| β | Regression coefficient / structural coefficient | outcome units per treatment unit | — | 4.2 |
| γ | Coefficient of the unmeasured confounder in the outcome equation | outcome units per U unit | — | 4.4 |
| λ | Attenuation / reliability factor | dimensionless | [0,1] | 4.4 |
| n, k | Sample size, number of strata | counts | positive integers | 5 |

**Takeaway:** the definitions that decide the analysis are the grain (D-1), the timing (nodes must precede the action), and the estimand (D-5); everything else is machinery in service of those three.

## 4. Core Formulas with Full Derivation

The section builds in the required ladder. **4.1** derives the base adjustment identity for a single discrete confounder from first principles, so that every later formula is a generalisation of something already proved. **4.2** generalises to an arbitrary adjustment set and to the two standard estimators that implement it. **4.3** covers the variants used when no backdoor set exists: front-door, instrumental variables, do-calculus, and the sensitivity instruments that bound an unmeasured confounder. **4.4** is the limit section: what happens at positivity failure, what collider conditioning does to a zero effect, and where identification genuinely stops.

### 4.1 Base case: adjustment for one discrete confounder

*Validity before use.* The derivation below requires four things and they are not negotiable: (i) the graph G is correct in the sense that it contains every common cause of the variables it connects; (ii) consistency, so that a unit observed with X=x has observed outcome equal to its own Y(x); (iii) conditional exchangeability, X ⊥⊥ Y(x) | Z, which the backdoor criterion will supply; (iv) positivity, 0 < P(X=x | Z=z) < 1 for every stratum z that carries weight in the target population. Violating (iv) makes the stratum-specific term undefined rather than merely noisy, which is a different kind of problem from a small sample.

*Derivation.* Start from the definition of the interventional distribution. Under an SCM, setting X to x by intervention removes the assignment mechanism of X and replaces it with the constant x, leaving every other assignment untouched. Summing over the confounder Z by total probability:

```text
step 1  P(y | do(X=x)) = Σ_z P(y | do(X=x), Z=z) · P(Z=z | do(X=x))
step 2  P(Z=z | do(X=x)) = P(Z=z)                  [Z is not a descendant of X]
step 3  P(y | do(X=x), Z=z) = P(y | X=x, Z=z)      [conditional exchangeability]
step 4  P(y | do(X=x)) = Σ_z P(y | X=x, Z=z) · P(Z=z)
```

Step 2 is the part readers skip. It holds because an intervention on X cannot change the distribution of a variable that no arrow from X reaches; the backdoor criterion's "no descendant of X" clause is exactly what licenses it. Step 3 is the substantive assumption: within a stratum of Z, treatment is as good as randomly assigned with respect to potential outcomes. It is implied by the graph when Z blocks every backdoor path, via d-separation: blocking all backdoor paths makes X and Y(x) d-separated given Z, and d-separation implies conditional independence.

Taking expectations of a difference gives the mean-difference form actually estimated.

```text
FORMULA [A-01]  (base standardisation / g-formula, one discrete confounder)
  ATE = Σ_z { E[Y | X=1, Z=z] − E[Y | X=0, Z=z] } · P(Z=z)

WHERE:
  ATE    average treatment effect on the additive scale, in outcome units
  Y      outcome, here binary retention so E[Y|·] is a risk in [0,1]
  X      treatment indicator, two levels
  Z      the single discrete confounder, strata indexed by z
  P(Z=z) standardisation weights over the TARGET population, summing to 1

CONSTRAINT:
  (i) Z blocks every backdoor path from X to Y; (ii) Z is not a descendant of X;
  (iii) 0 < P(X=1|Z=z) < 1 for every z with P(Z=z) > 0; (iv) consistency and no
  interference; (v) the graph is correct. Weights must come from the population
  you intend to speak about, not automatically from the analytic sample.

NUMERICAL EXAMPLE (ILLUSTRATIVE inputs, VERIFIED arithmetic):
  z=low:  risk(1)=0.300, risk(0)=0.200, weight 0.60
  z=high: risk(1)=0.800, risk(0)=0.720, weight 0.40
  stratum contrasts: low = +0.100, high = +0.080
  ATE = 0.60(0.100) + 0.40(0.080) = 0.0600 + 0.0320 = 0.092  →  +9.2 pp

INTERPRETATION:
  A 9.2 percentage-point retention gain, referred to the stated target population.
  The two stratum contrasts differ (+10.0 vs +8.0), so the 9.2 is a weighted
  average of genuinely heterogeneous effects, not a single homogeneous effect
  plus noise; Section 7 decides whether reporting the strata separately is
  required by the decision.
```

*Boundary check.* If Z has a stratum where nobody was treated, P(Z=z) > 0 while the conditional term is undefined, and the sum has no value. That is a positivity violation, not a missing data problem — Section 4.4 treats it. If instead Z is a descendant of X, step 2 fails and the formula estimates something that is neither the ATE nor a clean contrast; that is the over-adjustment failure in Section 8.

*Why the naive difference is so much larger.* In the same illustrative data the crude comparison gives 0.6571 − 0.3200 = +33.7 pp against the adjusted +9.2 pp. The gap is not sampling noise; it is composition. Treated units are 28.6% low-activity (1000 of 3500) whereas controls are 76.9% low-activity (5000 of 6500), and low-activity units retain far less often. The crude number answers "how do the treated and untreated differ", which is an associational question, while the adjusted number answers "what would change if the action were set", which is the decision-relevant one. The size of the gap is a diagnostic: when adjustment moves an estimate by a factor of three, the treatment assignment is strongly confounded and every remaining assumption deserves explicit attack.

### 4.2 General case: arbitrary adjustment set, and the two estimators that implement it

The base case generalises without changing the logic. Z becomes a vector, stratification becomes conditioning, and the sum becomes an integral or a sum over realised covariate values.

*The criterion.* Define the backdoor criterion formally, because the definition is what you cite when defending a model.

**Criterion BD-1 (backdoor).** A set Z satisfies the backdoor criterion relative to (X, Y) in G if: **(a)** no element of Z is a descendant of X; and **(b)** Z blocks every path between X and Y that contains an arrow into X.

When BD-1 holds, X ⊥⊥ Y(x) | Z in every distribution Markov and faithful to G, so the adjustment formula is valid. Two consequences that surprise practitioners: the criterion says nothing about which variables are *associated* with Y, and it permits sets that include variables with no marginal association with either X or Y — precision variables. A variable that predicts Y but is unrelated to X reduces variance without introducing bias, and including it is good practice. Conversely a variable strongly associated with X may be inadmissible.

*Derivation of the general adjustment formula.* The steps are identical to 4.1 with Z a vector; the only change is that the sum runs over the joint support of Z. Writing the result as an expectation over the target covariate distribution:

```text
FORMULA [A-02]  (general adjustment / g-formula)
  E[Y(x)] = ∫ E[Y | X=x, Z=z] dF_Z(z)
  ATE     = E[Y(1)] − E[Y(0)]

WHERE:
  F_Z    covariate distribution of the TARGET population
  E[Y|X=x,Z=z]  outcome regression, a function of (x,z)
  x ∈ {0,1} for a binary action

CONSTRAINT:
  BD-1 holds w.r.t. (X,Y,G); positivity for all z in the support of F_Z;
  the outcome regression must be correctly specified OR replaced by a
  nonparametric estimator; consistency; no interference.

NUMERICAL EXAMPLE: see Section 5, which implements A-02 with two strata and
  then repeats it with a continuous-style weighting to show the equivalence.

INTERPRETATION:
  The estimand is a weighted average of conditional outcome means. All the
  difficulty lives in getting F_Z right: standardising to the treated
  distribution gives the ATT, to the full population gives the ATE, to a
  future population gives a forecast effect. Same regression, three estimands.
```

```text
FORMULA [A-03]  (inverse probability weighting, Horvitz–Thompson form)
  Ê[Y(1)] = (1/n) Σ_i  X_i Y_i / e(Z_i)
  Ê[Y(0)] = (1/n) Σ_i  (1−X_i) Y_i / (1 − e(Z_i))
  where e(z) = P(X=1 | Z=z) is the propensity score.

WHERE:
  n      sample size [count]
  e(z)   propensity score, dimensionless in (0,1)
  1/e    weight applied to treated units; 1/(1−e) to controls

CONSTRAINT:
  Requires positivity (e bounded away from 0 and 1 in the sample), BD-1,
  consistency, and a correctly specified propensity model. Unstable when
  e is near 0 or 1; always report the weight distribution, not just the mean.

DERIVATION SKETCH:
  E[ X Y / e(Z) ] = E[ E[X|Z] · E[Y(1)|Z] / e(Z) ]   (conditional exchangeability)
                  = E[ e(Z) · E[Y(1)|Z] / e(Z) ]      (definition of e, cancellation)
                  = E[ E[Y(1)|Z] ] = E[Y(1)]          (tower property)
  The cancellation of e(Z) is the whole trick and it is why exact propensity
  values matter: an error in e does not cancel.

NUMERICAL EXAMPLE (continuing 4.1, saturated model):
  e(low)  = 1000/6000 = 0.1667 → treated weight 6.000, control weight 1.200
  e(high) = 2500/4000 = 0.6250 → treated weight 1.600, control weight 2.667
  treated pseudo-population: 1000(6.000) + 2500(1.600) = 6000 + 4000 = 10000
  treated retained:           300(6.000) + 2000(1.600) = 1800 + 3200 = 5000
  risk(1) = 5000/10000 = 0.500   (matches A-01's 0.500 exactly)
  control pseudo-population:  5000(1.200) + 1500(2.667) = 6000 + 4000 = 10000
  control retained:           1000(1.200)+ 1080(2.667) = 1200 + 2880 = 4080
  risk(0) = 4080/10000 = 0.408   (matches A-01's 0.408 exactly)
  ATE = 0.500 − 0.408 = 0.092

INTERPRETATION:
  With a saturated propensity model the two estimators agree exactly, which is
  a useful self-check: if they disagree materially with discrete Z and a
  saturated model, the code has a bug. With a smoothed model they diverge, and
  the divergence measures model dependence rather than sampling error.
```

The equality in the numerical example is exact rather than approximate, and that is worth stating as a property: when Z has k strata and the propensity model is saturated (k free parameters, no smoothing), IPW and standardisation target the same pseudo-population and coincide. Practitioners use this as a unit test. If a pipeline reports 9.2 pp from standardisation and 11.4 pp from IPW with the same two strata, one implementation is wrong or a weight is being trimmed silently.

*AIPW, the doubly robust combination.* Neither A-02 nor A-03 is protected against its own model being wrong. AIPW uses both and is consistent if either is right.

```text
FORMULA [A-04]  (augmented IPW / doubly robust)
  ψ = (1/n) Σ_i [ m1(Z_i) − m0(Z_i)
                  + X_i (Y_i − m1(Z_i)) / e(Z_i)
                  − (1−X_i)(Y_i − m0(Z_i)) / (1−e(Z_i)) ]
  where m_x(z) = Ê[Y | X=x, Z=z]

WHERE: m1, m0 outcome regressions; e propensity; ψ the ATE estimate.
CONSTRAINT: BD-1, positivity, consistency; at least one of {e, (m0,m1)} correctly
  specified; the correction terms must not be trimmed inconsistently with e.
NUMERICAL EXAMPLE: with correctly specified saturated m and e, the correction
  terms are exactly zero in each stratum and ψ reduces to A-01's 0.092.
INTERPRETATION: the augmentation term is a residual correction; when the outcome
  model is right the correction vanishes, when the propensity model is right the
  correction repairs the outcome model's bias. It is insurance, not a licence
  to skip diagnostics on both models.
```

*Effective sample size.* Weighting creates a pseudo-population larger than the data, so nominal standard errors are too small. The Kish effective sample size is the honest denominator.

```text
FORMULA [A-05]  (Kish effective sample size)
  ESS = (Σ_i w_i)² / Σ_i w_i²

WHERE: w_i the weight actually applied to unit i (1/e or 1/(1−e)); ESS in units of
  observations, ≤ n with equality iff all weights are equal.
CONSTRAINT: descriptive diagnostic, not a variance estimator; it summarises
  weight concentration and must be reported per arm.
NUMERICAL EXAMPLE (treated arm of 4.1):
  Σw = 10000; Σw² = 1000(36.000) + 2500(2.560) = 36000 + 6400 = 42400
  ESS = 10000² / 42400 = 100,000,000 / 42400 = 2358.49
  control arm: Σw² = 5000(1.4400) + 1500(7.1111) = 7200 + 10666.67 = 17866.67
  ESS = 100,000,000 / 17866.67 = 5597.01
  total ESS = 2358.49 + 5597.01 = 7955.51 from n = 10000 → 79.6% efficiency
INTERPRETATION: the design loses about a fifth of its information to weighting.
  If that loss is unacceptable the answer is a better design (more overlap), not
  a smaller weight cap chosen after seeing the result.
```

### 4.3 Variants: what to do when no backdoor set exists

Three families of escape exist. Each buys identification with a different, testable-in-part assumption, and choosing between them is the substance of Section 7.

**Variant 1 — front-door.** When an unmeasured confounder U sits between X and Y and a measured mediator M carries the entire causal path, the front-door formula identifies the effect without measuring U.

```text
FORMULA [A-06]  (front-door)
  P(y | do(X=x)) = Σ_m P(M=m | X=x) · Σ_{x'} P(y | M=m, X=x') · P(X=x')

WHERE: M the mediator on every directed path X→Y; x' dummy index over treatment levels.
CONSTRAINT (all three required):
  (F1) M intercepts ALL directed paths from X to Y — no unmediated path may exist;
  (F2) no unblocked backdoor path from X to M;
  (F3) every backdoor path from M to Y is blocked by X.
NUMERICAL EXAMPLE (binary M, ILLUSTRATIVE):
  P(M=1|X=1)=0.70, P(M=1|X=0)=0.30, P(X=1)=0.50
  P(Y=1|M=1,X=1)=0.60, P(Y=1|M=1,X=0)=0.60   (M fully determines the path)
  P(Y=1|M=0,X=1)=0.10, P(Y=1|M=0,X=0)=0.10
  inner sums: m=1 → 0.60(0.50)+0.60(0.50)=0.60 ; m=0 → 0.10(0.50)+0.10(0.50)=0.10
  P(y|do(X=1)) = 0.70(0.60) + 0.30(0.10) = 0.420 + 0.030 = 0.450
  P(y|do(X=0)) = 0.30(0.60) + 0.70(0.10) = 0.180 + 0.070 = 0.250
  front-door ATE = 0.450 − 0.250 = 0.200
INTERPRETATION: the mediator is used twice — first to describe how treatment
  moves it, then to describe how it moves the outcome — and the confounding
  between X and M is differenced away by the inner sum over x'. F1 is the
  assumption that usually fails: any direct effect of X on Y invalidates it.
```

Note what F1 costs in practice. A feature that changes retention partly through setup completion and partly through a support call has a direct path, so front-door on setup completion alone is invalid. The honest response is either to measure the second mediator and extend the chain, or to use a design that does not need the mediation assumption.

**Variant 2 — instrumental variables.** An instrument shifts X but reaches Y only through X.

```text
FORMULA [A-07]  (Wald / two-stage least squares estimand, binary instrument)
  β_IV = [ E[Y | Z=1] − E[Y | Z=0] ] / [ E[X | Z=1] − E[X | Z=0] ]
       = ITT_Y / ITT_X

WHERE: Z instrument; ITT_X first-stage effect (the "compliance" numerator);
  ITT_Y reduced-form effect on the outcome. β_IV in outcome units per treatment unit.
CONSTRAINT:
  (IV1) relevance: ITT_X ≠ 0, and in practice a strong first stage;
  (IV2) independence: Z ⊥⊥ (Y(x), X(z), U) — instruments are as-if randomised;
  (IV3) exclusion: Z affects Y only through X;
  (IV4) monotonicity (for the LATE reading): no defiers.
  Under IV1–IV3, β_IV identifies the effect for units whose treatment is moved
  by Z (the compliers), NOT the ATE. IV4 names that subpopulation.
NUMERICAL EXAMPLE (ILLUSTRATIVE):
  ITT_X = 0.12 (12% of the target group takes up the offer because of the nudge)
  ITT_Y = 0.024 (2.4 pp retention difference between nudged and not-nudged)
  β_IV = 0.024 / 0.12 = 0.20 → +20 pp per complier
  Compare: the naive per-treated difference in the same data is +9.2 pp.
INTERPRETATION: the per-complier effect is more than twice the population average
  because the population average averages over 88% who were never moved. Neither
  number is wrong; they answer different questions, and the decision owner must
  be told which one the ROI calculation is using.
```

The arithmetic here is the single most misused calculation in applied work. A 12% first stage with a 2.4 pp reduced form is entirely plausible in a voluntary programme, and dividing produces a number that is large, uncertain, and applies to a subpopulation nobody can list. If ITT_X were 0.02, the same 2.4 pp gives 120 pp, an impossible value that reveals the first stage is too weak for the ratio to be meaningful. That is why weak-instrument diagnostics precede interpretation rather than accompany it.

**Variant 3 — do-calculus and truncated factorisation.** For graphs too complex for a named criterion, the do-calculus supplies three rewrite rules and an algorithm can search for a sequence that eliminates the do-operator. The result is a truncated factorisation: an expression in observed conditional distributions only.

```text
FORMULA [A-08]  (do-calculus rules, stated for reference)
  R1  P(y | do(x), z, w) = P(y | do(x), w)     if Y ⊥⊥ Z | X, W in G_X̄
  R2  P(y | do(x), do(z), w) = P(y | do(x), z, w)  if Y ⊥⊥ Z | X, W in G_X̄Z̲
  R3  P(y | do(x), do(z), w) = P(y | do(x), w) if Y ⊥⊥ Z | X, W in G_X̄Z̄(W)

WHERE: G_X̄ deletes arrows into X; G_X̄Z̲ additionally deletes arrows out of Z;
  G_X̄Z̄(W) deletes arrows into and out of Z except those into W.
CONSTRAINT: graphical conditions are evaluated on the mutilated graphs, not on G.
  R2 is the rule that turns an intervention into an observation; R3 removes an
  intervention entirely, which is why it needs the strongest condition.
NUMERICAL EXAMPLE: R2 applied to Z in the backdoor case is exactly step 3 of the
  4.1 derivation, so A-01 is a one-rule do-calculus application.
INTERPRETATION: do-calculus is the completeness result, not a daily tool; if a
  quantity is identifiable at all, some sequence of these rules identifies it,
  and if no sequence exists the quantity is genuinely non-identified.
```

**Variant 4 — sensitivity instruments.** These do not identify anything; they bound how wrong the estimate could be. They belong in every observational report, because the graph assumption is the one thing never verified.

```text
FORMULA [A-09]  (E-value for a risk ratio)
  E = RR + sqrt( RR · (RR − 1) )

WHERE: RR the observed risk ratio (> 1; invert if < 1 then invert back);
  E the minimum strength of association, on the risk-ratio scale, that an
  unmeasured confounder would need with BOTH treatment and outcome to
  reduce the observed association to the null.
CONSTRAINT: RR must be a genuine ratio of risks in comparable arms; the bound
  concerns a single unmeasured confounder (or a set treated as one), and it is
  computed on the ratio scale while the estimate may be reported additively.
DERIVATION:
  Bias factor for a confounder with associations RR_UD (with treatment) and
  RR_AU (with outcome): B = RR_UD·RR_AU / (RR_UD + RR_AU − 1).
  Set the two associations equal to E and require RR_obs / B = 1:
      E² / (2E − 1) = RR_obs  →  E² − 2E·RR_obs + RR_obs = 0
      E = RR_obs ± sqrt(RR_obs² − RR_obs)
  Taking the root above 1 gives E = RR_obs + sqrt(RR_obs(RR_obs − 1)).
NUMERICAL EXAMPLE (Section 4.1 result):
  risk(1) = 0.500, risk(0) = 0.408 → RR = 1.2255
  E = 1.2255 + sqrt(1.2255 × 0.2255) = 1.2255 + sqrt(0.27635) = 1.2255 + 0.5257 = 1.7512
INTERPRETATION: an unmeasured confounder would need associations of at least
  RR ≈ 1.75 with both the offer and retention — beyond the measured covariates —
  to explain the whole effect. Compare that to the strongest measured confounder
  (activity band, RR ≈ 2.7 for retention): the bound is not comfortably large,
  which is a finding the deck must carry.
```

```text
FORMULA [A-10]  (omitted-variable bias, linear, single binary omitted confounder)
  plim(β̂) = β + γ · δ,  where δ = E[U | X=1] − E[U | X=0]

WHERE: β true effect; γ the omitted variable's coefficient in the outcome equation
  (outcome units per unit of U); δ the treatment–confounder association.
  Product γ·δ is the bias, in the same outcome units as β.
CONSTRAINT: linear outcome equation with additivity; U binary or standardised;
  the formula bounds one omitted variable at a time, not the joint effect of all.
DERIVATION: partialling X out of U gives the residual U − δX̃; regressing Y on X
  alone therefore attributes γ·δ of the omitted term to X.
NUMERICAL EXAMPLE:
  unmeasured "internal champion" U raises retention by γ = 0.15 and raises the
  probability of being offered the feature by δ = 0.10 → bias = 0.015 = 1.5 pp.
  Reported 9.2 pp therefore lies in [7.7, 10.7] under this single-confounder model.
INTERPRETATION: a 1.5 pp bias on a 9.2 pp effect changes the sign of no decision
  but does change an ROI ranking against a 8 pp threshold — so the bound, not the
  point estimate, is what the decision rule should consume.
```

**Variant 5 — negative controls.** A placebo outcome that the treatment cannot plausibly affect, and a placebo exposure that cannot plausibly affect the outcome, both probe residual confounding. If the placebo outcome shows an "effect" of similar size, the adjustment set is failing to block a path. This is a diagnostic, not an estimator, and it has no formula block because its value is entirely in the design of the placebo.

### 4.4 Limits and counterexamples

Four boundaries define the end of the method's reach. Each is a case where a formula above is *inapplicable*, not merely imprecise.

**Limit 1 — positivity failure makes the estimand undefined.** Suppose the 4.1 example had a third stratum, z=enterprise, with 900 treated and 0 controls. The term E[Y|X=0, Z=enterprise] does not exist. Three responses are legitimate: (a) restrict the estimand to the population with overlap and say so; (b) obtain controls in that stratum; (c) treat the enterprise effect as a separate, non-identified question. One response is illegitimate: impute the missing arm from a model fitted on other strata and report the result as if it were observed, because the extrapolation is exactly where the confounded mechanism bites hardest.

**Limit 2 — conditioning on a collider manufactures an effect from nothing.**

*Validity before use.* Linear-Gaussian SCM: X ~ N(0,1); Y = βX + ε with ε ~ N(0,1) and β = 0; C = aX + bY + η with a = b = 1 and η ~ N(0,1). The graph is X → C ← Y with no X → Y edge, so the true effect is exactly zero. Conditioning on C is a pure collider condition.

```text
FORMULA [A-11]  (collider-induced partial covariance and slope)
  Cov(X,Y | C) = Cov(X,Y) − Cov(X,C)·Cov(Y,C) / Var(C)
  β_{Y·X|C}    = Cov(X,Y | C) / Var(X | C),  Var(X|C) = Var(X) − Cov(X,C)²/Var(C)

WHERE: all quantities are population (or large-sample) moments of the linear-
  Gaussian system; β_{Y·X|C} is the coefficient on X in a regression of Y on X and C.
CONSTRAINT: linear-Gaussian; the formula describes the induced association, and it
  applies whether or not an X→Y edge also exists (then it is added to β).
NUMERICAL EXAMPLE (β = 0, i.e. no true effect):
  Var(X)=1, Var(Y)=1, Cov(X,Y)=0
  Var(C) = 1 + 1 + 0 + 1 = 3
  Cov(X,C) = 1 + 0 = 1 ;  Cov(Y,C) = 0 + 1 = 1
  Cov(X,Y|C) = 0 − (1)(1)/3 = −0.3333
  Var(X|C)   = 1 − 1/3 = 0.6667
  β_{Y·X|C}  = −0.3333 / 0.6667 = −0.5000
INTERPRETATION: adjusting for a collider turns a true zero into a precisely
  estimated −0.5. The standard error shrinks as the sample grows, so the wrong
  answer becomes more confidently wrong with more data. Significance testing
  cannot detect this; only the graph can.
```

**Limit 3 — M-bias: a collider that is not on any causal path still poisons the set.** In the graph X ← U₁ → M ← U₂ → Y, the node M is a collider with no role in the X→Y effect, and with no backdoor path through it either. The unadjusted estimate is unbiased. Conditioning on M opens the path X ← U₁ → M ← U₂ → Y and induces bias. This is the honest counterexample to "adjust for everything": there exists a variable that is neither a confounder nor a mediator, whose inclusion strictly hurts. The practical rule is not "never adjust for M" — if M were also needed to block another path, the trade-off would have to be resolved explicitly — it is that every variable in Z needs a stated reason referencing a specific path.

**Limit 4 — non-identification is a legitimate answer.** Consider X ← U → Y with no other structure. No observed set blocks the path, no mediator intercepts the effect, and no instrument exists. The ATE is not identified. Reporting a confidence interval here is worse than reporting nothing, because it implies a quantity has been estimated. The correct deliverable is the graph, the demonstration that no admissible set exists, the direction of the resulting bias (upward if U raises both X and Y), a bound from A-10 if a plausible γ is available, and the design change — randomisation, an instrument, a panel — that would identify it.

*A fifth limit worth naming: the graph itself is unfalsifiable in the direction that matters.* Agreement between implied and observed independences can reject a graph but cannot confirm it, because distinct graphs are Markov-equivalent when they share skeleton and v-structures. Two analysts can hold graphs that fit the data equally well and imply different adjustment sets. The resolution is not statistical: it is to write down the mechanism, date the assumptions, and record which disagreements would change the decision.

**Takeaway:** the formulas in 4.1–4.3 are all the same identity under different identification strategies; the limits in 4.4 are where you stop and report non-identification rather than substitute a number.

## 5. Worked Numerical Example

The example is deliberately small enough to check by hand and rich enough to expose every decision the method forces. **All counts are ILLUSTRATIVE** — a constructed teaching dataset about a hypothetical "guided setup" offer in a subscription product, with no company, product, or population behind it. **All arithmetic is VERIFIED** by two independent hand computations and by re-derivation from the raw counts.

### 5.1 The decision and the estimand

A product team can offer a guided setup flow to new accounts. Rollout is not randomised: accounts are offered the flow when an onboarding agent is available, and agent availability correlates with account size and with the season. Leadership asks whether to make the flow mandatory for everyone. The decision is binary, it is expensive (about 20 minutes of agent time per account), and it must be made before the next quarter.

The estimand is fixed before any table is opened:

| Element | Specification |
|---|---|
| Treatment X | X=1 if the account was offered guided setup within 7 days of signup |
| Outcome Y | Y=1 if the account was still active at day 90 |
| Unit | account, one row per account, grain = account |
| Window | signup to day 90, both arms, same calendar quarters |
| Target population | all new accounts in the two quarters analysed, weighted by their actual covariate distribution |
| Contrast | additive risk difference, percentage points |
| Decision rule | mandate if the lower 95% bound exceeds +5.0 pp, given the 20-minute cost |

The graph, written from the mechanism before any data extraction:

```text
account size (A) ──────────────► outcome (Y)
        │                            ▲
        │                            │
        └──► agent availability ──► offer (X)
                     │
                     │
        season (S) ──┴───────────────► Y
```

Read the graph: agent availability is a cause of the offer (larger accounts get more agent time), season drives both availability and retention (a promotional quarter lifts both), and account size drives retention directly. Every path from X to Y that starts with an arrowhead at X runs through agent availability. Conditioning on availability and season blocks those paths. Account size is not needed to block a path — it enters only if we want to speak about the size mix, and including it is harmless because it is a pre-treatment cause of Y that is not a descendant of X. So the candidate set is Z = {agent availability, season}, with account size optional.

### 5.2 The data

Two strata on agent availability (high, low) and two quarters collapsed into one season indicator. Counts:

| Stratum z | Treated n₁ | Treated retained | Control n₀ | Control retained | Risk(1) | Risk(0) | Contrast |
|---|---:|---:|---:|---:|---:|---:|---:|
| low availability | 1,000 | 300 | 5,000 | 1,000 | 0.300 | 0.200 | +0.100 |
| high availability | 2,500 | 2,000 | 1,500 | 1,080 | 0.800 | 0.720 | +0.080 |
| **total** | **3,500** | **2,300** | **6,500** | **2,080** | **0.6571** | **0.3200** | **+0.3371** |

Every cell is an integer count, and every risk below is recomputed from the counts rather than carried over:

```text
treated,  low :  300/1000 = 0.3000
treated,  high: 2000/2500 = 0.8000
control,  low : 1000/5000 = 0.2000
control,  high: 1080/1500 = 0.7200
treated pooled: 2300/3500 = 0.657143
control pooled: 2080/6500 = 0.320000
crude difference: 0.657143 − 0.320000 = 0.337143  → +33.7 pp
```

The crude +33.7 pp is the number that would appear in a dashboard comparing "offered" with "not offered". It is also the number that produces a disastrous mandate decision. Note the composition: 71.4% of treated accounts are in the high-availability stratum (2500/3500) against 23.1% of controls (1500/6500). Agent availability is doing two things at once — it selects who gets the offer and it marks accounts that are on a better trajectory — so the crude comparison is mostly measuring availability.

### 5.3 Stratified adjustment, step by step

Standardise to the full population's stratum weights. The weights are the marginal distribution of the confounder, which is the correct target for a "mandate everyone" decision.

```text
step 1  marginal weights
        w(low)  = (1000 + 5000) / 10000 = 6000/10000 = 0.60
        w(high) = (2500 + 1500) / 10000 = 4000/10000 = 0.40
        check: 0.60 + 0.40 = 1.00

step 2  stratum-specific risk differences
        Δ(low)  = 0.3000 − 0.2000 = 0.1000
        Δ(high) = 0.8000 − 0.7200 = 0.0800

step 3  weighted sum
        ATE = 0.60(0.1000) + 0.40(0.0800)
            = 0.0600 + 0.0320
            = 0.0920  →  +9.2 pp

step 4  independent cross-check, standardised risks
        risk*(1) = 0.60(0.300) + 0.40(0.800) = 0.180 + 0.320 = 0.500
        risk*(0) = 0.60(0.200) + 0.40(0.720) = 0.120 + 0.288 = 0.408
        difference = 0.500 − 0.408 = 0.092   ✓ matches step 3
```

Step 4 is the check worth institutionalising: standardise each arm separately and take the difference, then confirm it equals the weighted average of stratum contrasts. The identity holds exactly because both routes are linear in the same weights, so any disagreement is an implementation bug, not statistical variation.

### 5.4 The same estimand three other ways

**ATT, for a different decision.** If the question were "should we keep offering the flow to the accounts that currently get it", the weights would be the treated distribution: w(low) = 1000/3500 = 0.285714, w(high) = 2500/3500 = 0.714286.

```text
ATT = 0.285714(0.1000) + 0.714286(0.0800)
    = 0.0285714 + 0.0571429
    = 0.0857143  →  +8.57 pp
```

The ATT (+8.57 pp) is *smaller* than the ATE (+9.20 pp). That is a real feature of this graph, not a rounding artefact: the stratum with the larger effect (low availability, +10.0 pp) is under-represented among the treated. A mandate expands the offer to the low-availability population where the effect happens to be larger, so the mandate's per-account value exceeds the current programme's per-account value. Reporting the ATT while pricing a mandate would understate the return by 0.63 pp per account. This is the most common estimand mismatch in practice and it is invisible unless both numbers are printed.

**Outcome regression as a check.** A linear probability model with an interaction recovers the stratum-specific risks exactly when it is saturated on Z:

```text
fit  Y = a + b·X + c·H + d·(X·H),  H = 1 for high availability
  a = risk(0,low)                      = 0.200
  b = risk(1,low) − risk(0,low)        = 0.100
  c = risk(0,high) − risk(0,low)       = 0.720 − 0.200 = 0.520
  d = [risk(1,high) − risk(0,high)] − [risk(1,low) − risk(0,low)]
    = 0.080 − 0.100 = −0.020
check: risk(1,high) = 0.200 + 0.100 + 0.520 − 0.020 = 0.800  ✓
population-average effect = a + b + ... no: average the fitted contrasts with
  weights (0.60, 0.40): 0.60(0.100) + 0.40(0.080) = 0.092  ✓
```

A *main-effects-only* model, Y = a + bX + cH, forces d = 0 and therefore reports a single effect equal to a variance-weighted compromise between +10.0 and +8.0 that is neither. The compromise is not wrong arithmetic; it is a different, unstated estimand. If the decision differs by stratum — and here the cost is identical while the benefit is not — the interaction is the honest specification.

**IPW, already derived as A-03.** With the saturated propensity model the estimates coincide exactly at 0.092, which is the self-check described in 4.2. Using unsmoothed propensities:

```text
e(low)  = 1000/6000 = 0.166667  →  1/e = 6.0000,  1/(1−e) = 1.2000
e(high) = 2500/4000 = 0.625000  →  1/e = 1.6000,  1/(1−e) = 2.6667
treated: 1000(6.0000) + 2500(1.6000) = 6000 + 4000 = 10000  (pseudo-n)
retained: 300(6.0000) + 2000(1.6000) = 1800 + 3200 = 5000
risk(1) = 5000/10000 = 0.500
control: 5000(1.2000) + 1500(2.6667) = 6000 + 4000 = 10000
retained: 1000(1.2000) + 1080(2.6667) = 1200 + 2880 = 4080
risk(0) = 4080/10000 = 0.408
ATE = 0.092  ✓
```

The pseudo-population arithmetic is worth internalising: low-availability treated accounts are counted six times each, high-availability treated accounts 1.6 times each, and the resulting treated population has the same 60/40 availability mix as the full sample. That is what "removing confounding by weighting" means concretely.

**AIPW.** Because the outcome model is saturated on the same single covariate, m₁ and m₀ fit the cell means exactly, so the residuals Y − m_x(Z) are zero within every cell and the augmentation terms vanish identically. AIPW returns 0.092 as well. Reporting all four as agreeing is a legitimate quality signal *for this dataset only*; with a continuous Z and flexible models the four would differ, and the spread would be informative.

### 5.5 Uncertainty, effective sample size, and the decision

```text
nominal n: 3500 + 6500 = 10000
Kish ESS treated:  Σw = 10000, Σw² = 1000(36.0000) + 2500(2.5600) = 42400
                   ESS = 10000²/42400 = 100000000/42400 = 2358.49
Kish ESS control:  Σw² = 5000(1.4400) + 1500(7.1111) = 7200 + 10666.67 = 17866.67
                   ESS = 100000000/17866.67 = 5597.01
total ESS = 2358.49 + 5597.01 = 7955.51  (79.6% of nominal)

standard error of the risk difference, using ESS as the effective n:
  SE = sqrt( 0.500(1−0.500)/2358.49 + 0.408(1−0.408)/5597.01 )
     = sqrt( 0.2500/2358.49 + 0.241536/5597.01 )
     = sqrt( 0.00010600 + 0.00004316 )
     = sqrt( 0.00014916 )
     = 0.012213
95% CI: 0.0920 ± 1.9600(0.012213) = 0.0920 ± 0.023938
     → [0.0681, 0.1159]  →  [+6.8 pp, +11.6 pp]
```

Now apply the pre-registered decision rule: mandate if the lower bound exceeds +5.0 pp. The lower bound is +6.8 pp, which clears the threshold by 1.8 pp. But the threshold comparison is only valid if the assumptions hold, and the sensitivity analysis decides whether it does.

```text
E-value for the point estimate:
  RR = 0.500/0.408 = 1.225490
  E  = 1.225490 + sqrt(1.225490 × 0.225490)
     = 1.225490 + sqrt(0.276337)
     = 1.225490 + 0.525678
     = 1.751168   ≈ 1.75

E-value for the CI bound nearest the null:
  RR_lower = (0.408 + 0.0681)/0.408 = 0.4761/0.408 = 1.166912
  E_lower  = 1.166912 + sqrt(1.166912 × 0.166912)
           = 1.166912 + sqrt(0.194771)
           = 1.166912 + 0.441329
           = 1.608241   ≈ 1.61
```

Reading: an unmeasured confounder would need roughly a 1.75-fold association with both the offer and retention to explain away the point estimate, and 1.61-fold to explain away the CI bound. The measured activity variable has a much larger association with retention than 1.75, so the bound is *not* comfortable — a variable of that strength could exist and be unmeasured. The honest report therefore states the estimate, the interval, the E-values, and the specific unmeasured candidate (internal champion / procurement momentum) that would have to be ruled out.

**Decision consequence.** The point estimate clears the threshold; the sensitivity analysis shows the clearance is not robust. The recommendation is to mandate for the high-availability-adjacent segment where the effect is directly measured, and to run a randomised pilot in the low-availability segment — not to mandate globally on a bound that a single plausible unmeasured variable could erase.

### 5.6 Two wrong analyses on the same data, with numbers

**Wrong analysis 1: conditioning on a mediator.** Setup-completion S is caused by the offer and causes retention. In a separate illustrative sub-population:

```text
                   S=1 share   risk|S=1   S=0 share  risk|S=0   pooled risk
treated (X=1)        0.70        0.80       0.30       0.20      0.620
control (X=0)        0.30        0.80       0.70       0.20      0.380

unadjusted difference: 0.620 − 0.380 = 0.240  → +24 pp   (this is the true total effect)
adjusted within S:     S=1: 0.80 − 0.80 = 0.00
                       S=0: 0.20 − 0.20 = 0.00
                       weighted: 0.00  →  0 pp
```

The entire effect flows through S, so conditioning on S eliminates it exactly. A team that "controls for setup completion because it is the strongest predictor of retention" reports a zero effect for a programme that works. The model fit is excellent, the standard errors are small, and the conclusion is the opposite of the truth.

**Wrong analysis 2: conditioning on a collider.** Escalation E is caused by the offer (offered accounts contact support more) and by retention (accounts that stay raise more tickets over 90 days). In the linear-Gaussian illustration of A-11 with a true effect of zero, conditioning on E produces a coefficient of −0.500. In the offer setting this shows up as an adjusted effect that is smaller than the crude one by an implausible margin, often changing sign:

```text
crude:              +33.7 pp   (biased upward by availability)
backdoor-valid:      +9.2 pp   (the estimate to report)
adjusting for E:     −4.0 pp   (ILLUSTRATIVE collider-induced sign flip; not
                                computed from the table above, which does not
                                contain E)
```

The lesson is not that the middle number is "the true one" — it is that the middle number is the only one whose bias direction and source are stated. The crude number has an identified, quantified bias; the collider number has an identified bias of the *opposite* sign; and the valid number has a sensitivity bound. Only the third supports a decision.

### 5.7 What this example establishes, and what it does not

Established: the estimand can be fixed before analysis; the graph determines the set; two estimators agree exactly under a saturated model; the ATE and ATT differ materially and one of them is wrong for the stated decision; weighting costs about 20% of effective sample size; the E-value converts an unverifiable assumption into a number a decision owner can argue with.

Not established: that the graph is correct (no observational analysis can establish that); that agent availability is fully measured (it is a proxy for internal priority); that the effect is constant within strata; that the 90-day window is the decision-relevant horizon. Each of these is an OPEN item that belongs in the report next to the estimate, not in an appendix.

**Takeaway:** with a single binary confounder and four cell counts, the entire method is hand-checkable — which means every production analysis of this shape can be audited by re-deriving it from the counts.

## 6. Visual / Geometric Intuition

Geometry is the fastest way to see why adjustment does what it does, and the fastest way to see when it does the wrong thing. Three pictures carry almost all of the intuition: the *flow* picture for paths, the *valve* picture for conditioning, and the *geometry of conditional independence* for partialling.

### 6.1 Association flows along paths; conditioning operates valves

Picture the graph as a network of pipes with X at one end and Y at the other. Correlation is fluid. It flows along every path that is not blocked. The three-node motifs are valves with a fixed mechanical rule:

```text
CHAIN            FORK             COLLIDER
X → M → Y        X ← M → Y        X → C ← Y

fluid flows      fluid flows      fluid CANNOT flow
   X→M→Y            X←M→Y            X→C←Y
close by         close by         OPEN by
conditioning     conditioning     conditioning
on M             on M             on C  (or a descendant of C)

intuition:       intuition:       intuition: two effects
M carries the    M is the common  collide at C; selecting
effect: block    cause. Blocking  on a value of C forces
it and you       it removes the   an inverse relation
remove the       shared driver.   between the inflows.
effect you want.
```

The collider rule is the one that needs a physical picture, because it inverts ordinary intuition about "controlling for confounders". Consider a company where promotion (P) depends on either exceptional sales (S) or exceptional relationships (R), and everyone promoted has at least one of the two. Among promoted employees, S and R are negatively associated: if you learn that a promoted person is weak at sales, you can infer they must be strong at relationships. In the full population S and R are independent. Restricting to the promoted subpopulation *creates* an association. Conditioning on a collider is exactly this restriction, done statistically.

Translated to outcomes: if a feature is used more by high-intent accounts, and also used more by accounts that hit a specific pain point, then among feature users those two drivers become negatively associated. Adjusting for "user" status in that setting manufactures a relationship between intent and pain that does not exist in the population, and the resulting coefficient is contaminated by it.

### 6.2 The block/close ledger

The practical skill is to walk a path and decide, node by node, whether the currently conditioned set blocks it. The rule compresses to a ledger:

| Node on the path | Role in the path | Conditioned? | Path status |
|---|---|---|---|
| M | chain middle (→ M →) | no | open |
| M | chain middle (→ M →) | yes | **blocked** |
| M | fork middle (← M →) | no | open |
| M | fork middle (← M →) | yes | **blocked** |
| C | collider (→ C ←) | no | **blocked** |
| C | collider (→ C ←) | yes | **open** |
| D | descendant of a collider | yes | **open** (descendants behave like the collider) |
| D | descendant of a chain/fork middle | yes | partially blocked, not fully |

The last two rows are where implementations go wrong. Conditioning on a *descendant* of a collider opens the path just as conditioning on the collider does — so a "days since escalation" feature, a downstream consequence of escalation, opens the same path. Teams that carefully avoid the collider itself often reintroduce it through a derived feature built from it. The safe practice is to tag the analytic table with a `causal_role` column populated at design time (confounder / mediator / collider / proxy / neutral), so a downstream query cannot silently recruit a forbidden column.

Walking the graph in Section 5 with this ledger:

```text
path 1: X ← availability → Y            fork at availability
        availability unconditioned → path OPEN, must block → include availability ✓
path 2: X ← availability ← season → Y   chain through availability, fork at season
        conditioning on availability blocks it ✓ (season not needed here)
path 3: X ← availability → ... → Y      any further route through availability
        all blocked by availability ✓
directed path: X → Y (the effect)
        contains no conditioned middle node → remains OPEN ✓ (which is what we want)

conclusion: {availability} alone suffices; {availability, season} also valid;
            adding account size is valid and improves precision.
```

Note the asymmetry in the conclusion. Availability alone satisfies BD-1, yet including season is standard practice because season is a cause of Y and improves precision. Both sets are valid; only one is efficient. The criterion governs validity; efficiency is a separate, secondary choice — and confusing the two is how "adjustment set" arguments become endless.

### 6.3 Geometric picture of what partialling does

In the linear-Gaussian case there is an exact geometric interpretation that transfers to every estimator in Section 4.2. Treat each variable as a vector in n-dimensional space, centred. The regression coefficient of Y on X is the projection of Y onto X scaled by the length of X. Conditioning on Z means projecting Z out of both X and Y and regressing the residuals:

```text
              Y
              |\
              | \
    Y - proj_Z(Y)   <- the residual vector: the part of Y Z cannot explain
              |   \
              |    \
    ----------+-----+------ X
             proj_Z(Y)   X - proj_Z(X)

β_adj = <resid(Y), resid(X)> / <resid(X), resid(X)>
```

Three geometric consequences follow immediately, and each maps to a failure in Section 8.

*Orthogonality is not causation.* The residual of Y is orthogonal to Z by construction, so any variable you condition on becomes uncorrelated with the residual by arithmetic, regardless of the causal structure. The fit cannot tell you whether Z should have been there. This is the geometric statement of "R-squared is not evidence of a valid adjustment set".

*Conditioning removes variance, and that is both the benefit and the danger.* When Z is a genuine confounder, the residual of Y is shorter but the *ratio* of covariance to variance is unbiased. When Z is a mediator, the residual of X has had the mediated variation removed, so the numerator loses the effect itself — geometrically the residual X vector becomes nearly orthogonal to the outcome change the treatment causes. The estimator is well-defined and answers a question about a filtered population that no intervention corresponds to.

*Collider conditioning shortens both residuals in a correlated way, inducing apparent signal.* Conditioning on C = aX + bY + η projects out a vector containing both X and Y, so the residuals acquire a negative covariance proportional to ab/Var(C) — exactly the −1/3 in A-11. As η shrinks, C becomes a near-deterministic function of X and Y, the residuals become nearly perfectly anti-correlated, and the induced coefficient approaches −a/b. A "clean" collider makes the artefact larger, not smaller, which is why data quality does not protect against this failure.

### 6.4 The positivity picture

Positivity is a statement about the joint support of (X, Z), and it is best seen as a picture of the covariate space split by treatment.

```text
        Z (covariates, 2-D projection)
   ┌──────────────────────────────────────────┐
   │  ░░░░░░░░░░░░  treated only              │  ← no controls exist here
   │  ░░░░░░░░░░░░                            │    effect NOT identified
   │  ░░░░░░░░░░░░        ██████              │
   │        ░░░░░░░░░░   ████████             │  ← both arms present
   │           ░░░░░░░░░░░░░███               │    effect identified locally
   │              ░░░░░░░░░░░░░               │
   │  control only ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓         │  ← no treated exist here
   └──────────────────────────────────────────┘
        X: ░ treated        ▓ control        █ overlap
```

Reading the picture: the estimand is defined only over █. The ░ region has treated units whose counterfactual control outcome must be extrapolated from a model fitted where? Nowhere local — the nearest controls are far away in covariate space. The ▓ region has controls whose treated counterfactual is equally unavailable. An estimate that reports a single number for the whole covariate space is silently asserting a model for the ░ and ▓ regions.

Three visual diagnostics follow. First, plot the propensity distribution by arm; separated distributions mean the overlap region is small even if it is non-empty. Second, plot the distribution of weights; a long right tail is the numeric signature of near-violation. Third, check whether the *target population* definition actually lives inside █ — if the decision is about the ░ region, the honest conclusion is that this design cannot inform it, regardless of sample size.

### 6.5 Why the graph precedes the data

The final visual intuition is about workflow. Two orders of operation produce two different artefacts:

```text
ORDER A (graph first)                 ORDER B (data first)
──────────────────────                ────────────────────
mechanism → graph → measure →         warehouse → wide table →
check which nodes exist →             graph drawn from the columns →
adjustment set → estimate             adjustment set = all columns →
                                      stepwise selection → estimate

result: identifiable set, with        result: a set with unknown validity,
named unmeasured nodes and a          where unmeasured confounders are
sensitivity analysis for each         invisible because they are absent
                                      from the schema
```

Order B is not laziness; it is the path of least resistance in every organisation, because the warehouse is where the ticket starts. The corrective is procedural: require the graph, with node timestamps, as an artefact reviewed before the query is written. Once the graph exists, an absent column becomes a visible hole rather than an invisible one, and the sensitivity analysis has something specific to bound.

**Takeaway:** paths are pipes, conditioning is a valve, and the collider rule is an instruction to leave a particular valve alone — geometry explains why "control for everything" is a mathematical error rather than a stylistic one.

## 7. Comparison / Selection Table

Selection is driven by which assumption you can defend, not by which method is fashionable. The table below is ordered from strongest to weakest identification, and the "avoid when" column carries the disqualifiers that matter most in practice.

### 7.1 Identification strategies compared

| Strategy | What must be true | What it estimates | Data requirement | Use when | Avoid when |
|---|---|---|---|---|---|
| Randomised experiment | assignment independent of potential outcomes; compliance understood | ATE (or LATE if imperfect compliance) | ability to randomise at the decision grain | the decision can be delayed and assignment is controllable | randomising is unethical, illegal, or would leak across units |
| Backdoor adjustment (A-01/A-02) | BD-1 holds; all confounders measured | ATE or ATT, per standardisation weights | rich pre-treatment covariates with reliable as-of dates | the mechanism is well understood and confounders are instrumented in the warehouse | a plausible confounder is unmeasured and unbounded |
| IPW / propensity (A-03) | BD-1 holds; propensity model adequate; positivity | ATE or ATT depending on weighting | same covariates, plus overlap diagnostics | many covariates, few strata, and overlap is good | propensities pile up near 0 or 1 |
| AIPW / TMLE (A-04) | BD-1; one of two models correct | ATE | both a propensity and outcome model | both models can be specified plausibly | nobody will diagnose either model |
| Front-door (A-06) | F1–F3 all hold, especially no direct path | ATE | a mediator that carries the whole effect | a mechanism is genuinely serial and measured | any direct effect exists — F1 fails and the estimate is biased |
| Instrumental variables (A-07) | relevance, independence, exclusion, monotonicity | LATE for compliers | an instrument with a strong first stage | a lottery, cutoff, or policy discontinuity exists | the first stage is weak, or the effect is needed for the whole population |
| Difference-in-differences (file 03) | parallel trends absent treatment | ATT | pre-period panel, same units | a clean untreated comparison group exists with a long pre-period | trends differ, or treatment timing is endogenous |
| Regression discontinuity (file 03) | continuity of potential outcomes at the cutoff | local effect at the cutoff | a running variable with a threshold | eligibility is rule-based with an exact cutoff | units manipulate the running variable |
| Synthetic control (file 03) | donor pool spans the treated unit's pre-period | unit-specific counterfactual | long pre-period, many donors | one or few treated units with aggregate outcomes | few donors, short pre-period, structural break |
| Structural / SCM fitting | functional form correct | counterfactuals under the model | strong parametric assumptions | parameters are policy-invariant and identified | the model is fitted for prediction and relabelled |

The ordering is not a quality ranking of researchers; it is a ranking of how much of the identification burden is carried by design versus assumption. An experiment fails when it is badly run (leakage, attrition, novelty); a backdoor analysis fails when the graph is incomplete. Both failures are common; only one is detectable from the data alone, which is why experiments are preferred when feasible — not because observational methods are unscientific.

### 7.2 Estimator choice inside the backdoor family

Given that a valid Z exists, the choice among implementations is a matter of diagnostics rather than principle.

| Diagnostic reading | Recommended estimator | Reason |
|---|---|---|
| Few discrete strata, all cells populated | Standardisation (A-01/A-02) | exact, transparent, hand-checkable |
| Many covariates, good overlap, propensities mid-range | IPW with ESS reported | avoids outcome-model extrapolation |
| Overlap decent but propensities reach 0.02/0.98 | AIPW with trimming at a pre-declared rule | augmentation reduces reliance on extreme weights |
| Outcome model trusted, overlap poor in some region | Outcome regression, restricted estimand | avoids weighting into empty space |
| Both models uncertain, sample large | AIPW/TMLE with cross-fitting | double robustness plus honest inference |
| Strata with zero cells | Restrict estimand; report the restriction | no estimator repairs undefined terms |
| Effect clearly heterogeneous across Z | Report stratum contrasts, not one number | a single average hides the decision |

The last row deserves emphasis: the choice is not only about which estimator, but whether an average is the right object at all. In the Section 5 example the two strata differ by 2 pp, which is small enough that an average is defensible. At a 10 pp difference between strata with different costs, the average would obscure a decision that should be made per stratum.

### 7.3 Which adjustment set is "the" set

Any set satisfying BD-1 is valid, and there may be many. Selection among valid sets is governed by precision and by defensibility.

| Set | Valid? | Precision | Defensibility | Choose when |
|---|---|---|---|---|
| {availability} | yes | good | weak — omits a known cause of Y | never, if a stronger cause of Y is available |
| {availability, season} | yes | better | strong — both are mechanism-derived | the default |
| {availability, season, size} | yes | best | strong | size is measured with an as-of date and predicts Y |
| {availability, season, size, plan tier} | yes if tier precedes X | marginal gain | requires a timing argument | tier is recorded at signup |
| {availability, season, setup completion} | no — mediator | artificially high | fatal | never |
| {availability, season, escalation} | no — collider | artificially high | fatal | never |
| {availability, season, post-hoc segment flag} | no if flag is derived from Y | artificially high | fatal — outcome leakage | never |

The pattern in the rejected rows is identical: fit improves, bias increases. Any covariate that raises R² while carrying information *about the outcome* that was generated after the treatment is a candidate for exactly this error. The as-of date is the only reliable guard, applied at the column level.

### 7.4 Reporting choice by audience

| Audience | Report | Do not report |
|---|---|---|
| Decision owner | effect, interval, decision rule outcome, the one assumption it hinges on | estimator internals, balance tables |
| Analytical peer | estimand, graph, set, estimator, diagnostics, sensitivity, code | the deck's rounded headline alone |
| Auditor | all of the above plus as-of dates, cell counts, exclusion log | any number without its denominator |
| Regulator / external | estimand and population in plain language; limitations section | causal language beyond what the design supports |

Each audience needs a different slice of the same artefact. The failure mode is producing one artefact and handing it to all four, which means the decision owner gets balance tables and the auditor gets a rounded percentage.

**Takeaway:** pick the strategy by the assumption you can defend, then pick the estimator by the diagnostic you can measure; never pick either by what the previous analyst used.

## 8. Failure Modes & Edge Cases

Each entry gives the symptom as it appears in a real workflow, the competing causes that produce the same symptom, the diagnostic that discriminates them, a bounded fix, and the check that verifies the fix. Nine failures, ordered roughly by how often they occur.

### F-01 Positivity violation in a decisive stratum

**Symptom.** One covariate stratum has treated units and no controls (or the reverse), and the effect is largest there. The pipeline drops the stratum and reports the remainder.

**Competing causes.** (a) Genuine structural non-overlap — eligibility rules made the arms impossible in that stratum; (b) a measurement artefact where a covariate is only recorded for treated units, so controls look empty; (c) a join error that silently restricted controls to a subset.

**Discriminating check.** Count units by (X, Z) cell and compare to the eligibility rules in the source system. If the source rules explain the empty cell, it is (a). If Z is recorded in a different table per arm, it is (b). If the counts contradict known business volumes, it is (c).

**Bounded fix.** For (a), restrict the estimand to the overlap population and state the restriction in the headline; optionally report the non-overlap region as a separate, explicitly non-identified question. For (b), harmonise the covariate source. For (c), repair the join and re-run.

**Verification.** The reported estimand names its population; the cell counts sum to the analytic n; a re-run with the repaired join changes the empty cell to a small but non-zero count.

### F-02 Conditioning on a mediator

**Symptom.** The adjusted effect is near zero or much smaller than the crude effect, model fit is excellent, and the coefficient on the mediator is highly significant.

**Competing causes.** (a) The effect genuinely operates entirely through the mediator; (b) the mediator was included as a "strong predictor" without any causal-role review; (c) the outcome is measured so late that a second, unmodelled path dominates and the mediator is absorbing it.

**Discriminating check.** Ask whether an intervention on X changes the mediator. If yes and the mediator is not the decision target, it is over-adjustment by construction. Then compare the crude and adjusted estimates: a near-total collapse with a highly significant mediator coefficient is the signature of (a) or (b). To separate them, fit the model without the mediator and inspect whether the total effect is decision-relevant — if it is, the mediation question is a secondary, mechanism question.

**Bounded fix.** Remove the mediator from the adjustment set; if the mechanism matters for the decision, report the total effect as the headline and decompose it as a secondary analysis using a mediation method with its own stated assumptions.

**Verification.** The set no longer contains any descendant of X; the effect returns to the crude-plus-confounder-adjustment value; the decomposition is labelled as mechanism, not as the decision estimand.

### F-03 Conditioning on a collider, or on its descendant

**Symptom.** An adjusted effect that is smaller than expected, sometimes with the opposite sign, in a model containing a variable that "should not matter". Or an effect that grows in magnitude as more "quality" covariates are added.

**Competing causes.** (a) A true collider; (b) a descendant of a collider built from a derived feature; (c) selection on the outcome — the analytic sample was restricted to units that survived to a later date, which is collider conditioning via sampling.

**Discriminating check.** Draw the graph including the suspect variable and check the arrowheads on the path it lies on. If both arrows point into it, it is (a). Check the feature lineage for the derived case (b). Compare the analysed sample's inclusion rule against its survival requirement for (c). The signature of all three: adding the variable moves the estimate in a direction the mechanism does not predict.

**Bounded fix.** Drop the variable (and every feature derived from it) from the set; for (c), reconstruct the cohort with a fixed, pre-treatment inclusion rule and accept a shorter follow-up window rather than restricting on survival.

**Verification.** Re-run the "add one covariate at a time" sequence and confirm that no single addition moves the estimate by more than the pre-declared stability tolerance without a stated causal reason. Record the tolerance before the sequence.

### F-04 Unmeasured confounding, reported as if it were noise

**Symptom.** An estimate with a tight confidence interval and no sensitivity analysis; a reviewer asks "what if there is something you did not measure" and the answer is "we controlled for everything available".

**Competing causes.** (a) A genuine unmeasured confounder; (b) a measured confounder that is a poor proxy, so adjustment is incomplete; (c) a violation of consistency (multiple versions of treatment) that looks like confounding.

**Discriminating check.** Compute an E-value for the point estimate and for the CI bound. Then list candidate unmeasured variables and estimate their plausible γ·δ product using A-10 from adjacent data. For (b), compute an attenuation factor by comparing the estimate using the proxy against a subsample where a better measure exists. For (c), check whether "treated" is homogeneous in the intervention delivered.

**Bounded fix.** Report the E-values; run a negative-control outcome; if a candidate's plausible bias exceeds the effect, downgrade the claim to "consistent with, not established". For (c), split the treatment definition into its real versions.

**Verification.** The report contains a sensitivity subsection with a number, not a disclaimer; the negative control shows an effect near zero; the treatment definition lists its versions.

### F-05 M-bias introduced by an "obviously good" control

**Symptom.** A variable with no plausible relation to the treatment is included because it predicts the outcome strongly, and the estimate shifts.

**Competing causes.** (a) Genuine M-structure (X ← U₁ → M ← U₂ → Y); (b) the variable is a proxy for an unmeasured confounder and the shift reflects partial adjustment of a real bias; (c) sampling noise in a small stratum.

**Discriminating check.** In (a), the variable is uncorrelated with X marginally but correlated with Y; conditioning opens a path that was closed and the shift is towards the null-with-noise pattern. In (b), the shift is towards a value that better matches an external benchmark. Distinguish by examining whether the variable's marginal association with X is near zero while its conditional association after adjusting is non-zero — the M-pattern.

**Bounded fix.** Prefer the smaller set when the bias from a collider could exceed the bias from an omitted precision variable; report both estimates and the mechanism argument. Do not adopt the larger set merely because it is larger.

**Verification.** Each variable in Z maps to a named path it blocks; variables that block nothing are either removed or justified as precision-only with a stated M-structure argument.

### F-06 Estimand mismatch between analysis and decision

**Symptom.** The decision is "expand the programme to everyone" but the reported number is the ATT computed on current participants; or the reverse.

**Competing causes.** (a) The estimator naturally produces the other estimand; (b) weights were taken from the analytic sample by habit; (c) the target population changed after the analysis was scoped.

**Discriminating check.** Write the decision rule as a sentence with a population in it, then check whether the standardisation weights match that population's covariate distribution. In the Section 5 example, ATE = 9.20 pp against ATT = 8.57 pp — a 0.63 pp gap that flips an ROI ranking.

**Bounded fix.** Recompute with the weights of the decision population; if that population has no overlap, restrict the decision to the overlap and state it.

**Verification.** The weights sum to 1 over the target population; the reported number changes when the weights are changed to the other population (if it does not, the covariate does not matter and the point is moot).

### F-07 Weak instrument with confident reporting

**Symptom.** A per-complier estimate far outside any plausible range, presented without the first-stage magnitude.

**Competing causes.** (a) Genuinely small compliance; (b) the instrument affects the outcome through a second channel (exclusion violation); (c) the instrument is a proxy for a confounder.

**Discriminating check.** Report ITT_X and ITT_Y separately. A first stage of 0.02 with a 2.4 pp reduced form produces 120 pp, which is impossible for a binary outcome and reveals (a) or a violation. Test exclusion by checking whether the instrument predicts pre-treatment covariates it should not.

**Bounded fix.** Report the ITT as the primary result — it is a valid causal effect of the *offer* and is decision-relevant when the decision is whether to make the offer. Report LATE as secondary with its complier subpopulation described.

**Verification.** ITT_X and ITT_Y appear in the same table as β_IV; the estimate is inside the outcome's logical range; the subpopulation is described in words.

### F-08 Graph drawn from available columns

**Symptom.** The adjustment set equals the feature table. Unmeasured confounders never appear in the analysis narrative.

**Competing causes.** (a) Workflow starting from the warehouse; (b) a genuine belief that "more control is safer"; (c) pressure to ship an estimate by a date.

**Discriminating check.** Ask which variables the analyst believes cause both treatment and outcome but were not measured. A blank answer for a well-understood mechanism indicates (a). Compare the graph's node list with the table's column list: identity between them is the tell.

**Bounded fix.** Redraw the graph from the mechanism, with unmeasured nodes included and marked. For each unmeasured node, either find a proxy or add a sensitivity bound.

**Verification.** The graph contains at least one node not in the analytic table, or an explicit statement that none exists with the argument for it.

### F-09 Causal language unsupported by the design

**Symptom.** A slide reading "the feature drove a 12% lift" produced by a cross-sectional regression.

**Competing causes.** (a) Estimand confusion; (b) a valid analysis described carelessly in the deck; (c) deliberate framing.

**Discriminating check.** Compare the deck's verb against the design: "drove", "caused", "led to" require an identification argument; "was associated with", "co-moved with", "predicted" do not. If the underlying analysis is valid, the fix is editorial. If not, the fix is analytical.

**Bounded fix.** Replace causal verbs with associational ones unless the identification argument is documented, and add a one-line limitation naming the key assumption the causal reading would require.

**Verification.** Every causal verb in the deliverable traces to a documented design; the limitations section names the unverifiable assumption.

### Edge cases worth separate handling

**Binary outcomes with rare events.** Risk differences near 0 or 1 make the additive scale unstable and the odds-ratio scale misleading; report both scales and check which one the decision rule was written in. A rule of "+5 pp" cannot be evaluated from an odds ratio without conversion.

**Time-varying treatment.** If treatment status changes during follow-up, the simple backdoor set is insufficient; adjusting for a time-varying confounder that is also affected by past treatment requires g-methods (marginal structural models with inverse-probability-of-treatment weights). Treating such a variable as a baseline confounder is F-02 with a time dimension.

**Clustered assignment.** When treatment is assigned at store or region level but outcomes are measured per customer, the effective n is the number of clusters. Ignoring this understates standard errors by a large factor; the diagnostic is to compute the design effect and report it.

**Interference.** If treated units change the environment for controls (shared inventory, shared support queue, social contagion), SUTVA fails and the "control" outcome is contaminated. Detect by comparing control outcomes across randomisation ratios; a control group whose outcome moves with the treated share indicates interference.

**Multiple versions of treatment.** A soft launch and a full launch pooled under one flag violate consistency. Split them, or define the estimand as the effect of the *policy* of offering whichever version was live, and accept the wider interpretation.

**Truncated or censored outcomes.** Follow-up ending before the outcome is observable creates differential censoring when censoring is related to treatment. Adjusting for "still active at last observation" is conditioning on a post-treatment variable; use a survival or censoring-aware estimator instead.

**Takeaway:** every failure in this section leaves the model fitting well — the diagnostic that catches them is structural, so it must be built into the workflow before any p-value is computed.

## 9. Complexity & Cost Analysis

Causal adjustment has real computational and organisational costs. The computational costs are usually trivial next to the organisational ones, but both need sizing because the choice of method is often decided by cost rather than by validity.

### 9.1 Computational cost of each estimator

Let n be rows, p the number of covariates in Z, k the number of strata for a discrete Z, and T the number of iterations for a numeric optimiser.

| Operation | Time | Space | Dominant term | Notes |
|---|---|---|---|---|
| Stratified standardisation, k cells | O(n + k) | O(k) | one pass to count, one to accumulate | exact; no iteration |
| Outcome regression (OLS, p covariates) | O(n p² + p³) | O(p²) | forming and solving XᵀX | p³ matters only if p is large |
| Logistic propensity, Newton/IRLS, T iters | O(T n p²) | O(p²) | repeated weighted least squares | T ~ 5–25 in practice |
| IPW weighting pass | O(n) | O(n) | building the weight vector | one pass |
| Kish ESS | O(n) | O(1) | sum and sum-of-squares | streaming-friendly |
| AIPW / doubly robust | O(T n p² + n) | O(n + p²) | same as propensity plus predictions | two outcome models |
| Nearest-neighbour propensity matching | O(n log n) to O(n²) | O(n) | the search | exact matching is the n² case |
| d-separation check on a DAG with V nodes, E edges | O(V + E) per pair, O(V³) for all pairs | O(V²) | path enumeration, or moralisation | graph is tiny; cost irrelevant |
| Front-door estimation | O(n + k_m k_x) | O(k_m k_x) | two strata-crossed passes | exact if all variables discrete |
| 2SLS | O(n p² + p³) | O(p²) | two OLS fits | same order as one regression |
| Bootstrap interval, B replicates | O(B × estimator) | O(n) if streaming | linear in B | B = 1000 typical |

The practical reading: for n up to about 10⁷ and p under a few hundred, every estimator in the table runs in seconds to minutes on one machine, and the difference between methods is irrelevant to the compute budget. Any argument for choosing a method "because it is faster" is almost always an argument about analyst time or tooling, not about the machine.

### 9.2 Worked sizing: one analysis, end to end

```text
Scenario: 40 million account-rows, p = 60 covariates, binary treatment,
          outcome measured at day 90, quarterly refresh.

Data extraction and as-of-dated join
  scan 40e6 rows, ~120 bytes/row  = 4.8e9 bytes ≈ 4.47 GiB (1 GiB = 1073741824 B)
  warehouse scan at 500 MB/s       = 4.8e9/500e6 = 9.6 s
  with 3x overhead for joins/sort  ≈ 29 s

Propensity model (logistic, IRLS, T = 10, p = 60)
  per iteration: O(n p²) = 4.0e7 × 3600 = 1.44e11 flops
  ten iterations:                  = 1.44e12 flops
  at 1e10 flops/s single core:     144 s
  measured-type caveat: flops counting is an upper-bound model, not a benchmark

IPW + ESS + AIPW
  O(n) passes: 4 passes × 4.0e7 = 1.6e8 row-ops → seconds
  two outcome regressions at O(n p²) each ≈ 2 × 14.4 s of the above per-iteration
                                           cost ≈ 29 s

Bootstrap, B = 1000, refitting propensity each time
  = 1000 × 144 s ≈ 144000 s ≈ 40 h single core
  → reduce to B = 200 (28.8 ks ≈ 8 h) or use the influence-function
    variance, which is O(n) after the fit and avoids refitting entirely.

Total, without bootstrap:       ≈ 1–2 minutes compute
Total, with 1000 bootstraps:    ≈ 40 hours compute
Analyst time to specify graph:  ≈ 2–6 hours of domain interviews
Review/sign-off:                ≈ 1–3 days elapsed
```

The sizing teaches two things. First, the compute is not the constraint; the graph specification and the review cycle are. Second, the bootstrap is the only component that can blow the budget, and it is usually avoidable: for AIPW the influence function gives an analytic variance in O(n), and for standardisation the stratified variance formula does the same. Reserving the bootstrap for cases where no analytic variance exists is a genuine cost decision.

### 9.3 Cost of getting it wrong

The cost asymmetry is the argument for the method. Consider a mandate decision on a population of 100,000 accounts with a 20-minute agent cost per account.

```text
cost of a false positive (mandate when true effect ≈ 0):
  100000 accounts × (20/60) h × 45 currency/h = 100000 × 15 = 1,500,000 currency
  plus opportunity cost of agent time displaced from other work: not measured

cost of a false negative (skip when the true effect is +9 pp):
  foregone retention: 100000 × 0.09 = 9000 retained accounts
  at a stated contribution margin of 120 currency/account/year
    = 9000 × 120 = 1,080,000 currency/year
  (both figures ILLUSTRATIVE; the point is the ratio, not the level)

cost of doing the graph work first:
  ~2–6 analyst hours + 1–3 days review ≈ small relative to either error
```

The ratio is what matters: a single avoided wrong mandate costs more than a year of careful adjustment work across the whole team. This is the honest business case for the discipline, and it does not require any claim about how often analyses are wrong.

### 9.4 Organisational cost and latency

| Cost centre | Unit | Typical size | Where it bites |
|---|---|---|---|
| Domain interviews to draw the graph | hours per mechanism | 2–6 | schedule, not budget |
| As-of-dated feature construction | engineering days | 2–10 per feature family | the real bottleneck |
| Review by a second analyst | hours | 2–4 | catches F-01 to F-05 cheaply |
| Re-run after a set change | minutes to hours | 0.5–4 h | cheap if the pipeline is parameterised by the set |
| Decision delay | calendar days | 3–10 | often the binding constraint |

The last row is why teams skip the work. The mitigation is to make the graph a reusable artefact per mechanism rather than a per-analysis deliverable: a documented graph for "onboarding offers" serves every future analysis of that mechanism, so the cost amortises across quarters.

### 9.5 Cost of diagnostics

| Diagnostic | Cost | What it catches | Skip risk |
|---|---|---|---|
| Cell counts by (X,Z) | seconds | positivity, F-01 | estimate undefined in a stratum |
| Propensity overlap plot | minutes | near-violation | unstable weights, wide CI |
| Weight distribution + ESS | seconds | weight concentration | false precision |
| Covariate balance table | minutes | model misspecification | residual confounding |
| Negative control outcome | one extra query | residual confounding | overconfident claim |
| E-value | a calculator | unmeasured confounding | unsupported causal language |
| Add-one-covariate stability | one pipeline run per covariate | F-02, F-03 | mediator/collider inclusion |

Every row is cheap. The total is under a day of analyst time for a standard analysis, against a decision whose cost of error runs to seven figures. The economics are not close.

**Takeaway:** compute is never the constraint — the graph, the as-of-dated features, and the review cycle are, and each of them is cheap relative to one wrong mandate.

## 10. Key Theorems / Results Reference

Each result below is stated with the hypotheses it actually requires, because the commonest misuse of this literature is to cite a theorem whose conditions are not met. Statements follow Pearl's graph-based formulation and the surrounding causal-inference literature; page-level citations are named in the collection index's source register rather than invented here.

### T-1 Markov property and d-separation

**Statement.** If a distribution P is Markov relative to a DAG G, then d-separation in G implies conditional independence in P: for disjoint sets A, B, Z, if A and B are d-separated given Z in G, then A ⊥⊥ B | Z in every P Markov to G.

**Hypotheses.** G is acyclic; P satisfies the Markov condition with respect to G (each variable is independent of its non-descendants given its parents). No faithfulness is needed for this direction.

**Use.** This is what licenses "Z blocks the backdoor path, therefore conditional exchangeability holds". It is the forward direction and it is the one relied on by every estimator in Section 4.

**Limitation.** The converse (independence in P implies d-separation) requires faithfulness, which can fail through exact cancellations. Since only the forward direction is used for adjustment, faithfulness is not needed to justify A-01 — a point that is frequently and incorrectly asserted to the contrary.

### T-2 Backdoor criterion and the adjustment theorem

**Statement.** If Z satisfies criterion BD-1 relative to (X, Y) in G, then the interventional distribution is identified by the adjustment formula P(y | do(X=x)) = Σ_z P(y | X=x, Z=z) P(Z=z), and more generally P(y | do(X=x)) = Σ_z P(y | X=x, Z=z) P(z) for any distribution P(z) matching the target population.

**Hypotheses.** G contains all common causes of the variables in the analysis; consistency; positivity.

**Use.** Justifies A-01 and A-02 and, by extension, the weighting versions in A-03.

**Limitation.** The criterion is sufficient, not necessary: there exist graphs where the effect is identified but no set satisfies BD-1, requiring the do-calculus (T-3) or the generalised adjustment criterion (T-4).

### T-3 Do-calculus and its completeness

**Statement.** The three rules R1–R3 of A-08 are sound for deriving interventional expressions from observational ones. They are also complete: if a causal effect is identifiable in a causal DAG, some sequence of these rules derives it.

**Hypotheses.** The graph is a DAG (or is converted to one by representing latent structure with bidirected edges); the Markov and, for the completeness direction, faithfulness assumptions.

**Use.** Decides identifiability in graphs too complex for a named criterion; the basis of the ID algorithm implemented in the `causaleffect`/`dowhy`-style libraries.

**Limitation.** Deciding identifiability is distinct from estimating well. An identified expression containing a high-dimensional conditional density may be practically unestimable at the available n. Identifiability is necessary, not sufficient, for a usable estimate.

### T-4 Generalised adjustment criterion

**Statement.** A set Z is an adjustment set if (i) Z contains no descendant of any node on a proper causal path from X to Y (other than X itself), and (ii) Z blocks every non-causal path between X and Y. This generalises BD-1 beyond the case where X has no causal ancestors.

**Hypotheses.** Same graph assumptions as T-2.

**Use.** Handles the case where treatment has parents on the causal path — for example a sequential decision where an earlier treatment stage is a parent of a later one. BD-1 can fail to find a set that T-4 finds.

**Limitation.** Requires care in identifying proper causal paths; a mis-specified path set produces a set that blocks the effect. Implementations exist but must be checked against a hand-drawn graph on a small example.

### T-5 Front-door criterion

**Statement.** If M satisfies (F1) M intercepts every directed path from X to Y, (F2) no unblocked backdoor path from X to M, and (F3) every backdoor path from M to Y is blocked by X, then P(y | do(X=x)) = Σ_m P(m | X=x) Σ_{x'} P(y | M=m, X=x') P(x').

**Hypotheses.** F1–F3 as stated; consistency; positivity in both the X→M and M→Y relations.

**Use.** Identifies an effect in the presence of an unmeasured X–Y confounder, at the cost of a strong mediation assumption.

**Limitation.** F1 is falsified by any direct effect of X on Y. Because the formula still returns a number when F1 fails, front-door is dangerous precisely because it never announces its own violation. It is not a general escape from unmeasured confounding.

### T-6 Propensity score balancing property

**Statement.** If X ⊥⊥ Z | e(Z), where e(Z) = P(X=1 | Z), then X ⊥⊥ Z given e(Z) for every Z: conditioning on the scalar propensity score balances the full covariate distribution between arms.

**Hypotheses.** e(Z) is the true propensity; the treatment is binary and unconfounded given Z.

**Use.** Justifies dimension reduction — adjusting for a scalar instead of a vector — which underlies matching and IPW (A-03).

**Limitation.** The property is about the *true* score. With an estimated score, balance holds only approximately, and never exactly in finite samples. "The propensity model has good AUC" is not the balancing property; balance must be checked directly, covariate by covariate, in the weighted or matched sample.

### T-7 Double robustness

**Statement.** The AIPW estimator (A-04) is consistent for the ATE if either the propensity model or the outcome model is correctly specified (not necessarily both), given BD-1 and positivity.

**Hypotheses.** Positivity; the correct model's specification; both models estimated from the same data (cross-fitting is needed for valid inference, not for consistency, in the modern formulation).

**Use.** Justifies the default recommendation to report AIPW when two models can be plausibly specified.

**Limitation.** Double robustness protects against one misspecification, not two, and it does not protect against a wrong adjustment set. An AIPW estimate built on a set containing a collider is confidently wrong. Also, with heavy trimming the estimator no longer targets the original estimand.

### T-8 Instrumental variable identification and the LATE theorem

**Statement.** Under IV1–IV3 (relevance, independence, exclusion), β_IV = ITT_Y / ITT_X identifies a weighted average of treatment effects; under IV4 (monotonicity, no defiers) it equals the average effect for compliers.

**Hypotheses.** IV1–IV4 plus SUTVA. IV2 and IV3 are not testable in general.

**Use.** Identifies an effect when treatment is confounded but an as-good-as-random shifter exists.

**Limitation.** The estimand is local to compliers, a subpopulation that is not enumerated in the data and may not be the decision-relevant population. A strong first stage is required for the ratio to be informative, and the Wald ratio is highly sensitive to small ITT_X.

### T-9 Non-identification: the bow-arc graph

**Statement.** In the graph X ← U → Y with U unobserved and no other structure, no observed set satisfies any adjustment criterion, no mediator intercepts the path, and no instrument exists. The ATE is not identified.

**Hypotheses.** Nothing beyond the graph.

**Use.** Grounds the requirement to report non-identification rather than a wide interval, and motivates each of the design escapes.

**Limitation.** None as a negative result — but it is a statement about *this* graph. Adding a measured proxy for U, or a variable that shifts X, changes the answer, and the analysis should search for such variables before concluding.

### T-10 Collider-induced bias (selection bias)

**Statement.** If C is a collider on a path between X and Y, or a descendant of such a collider, then conditioning on C induces association between X and Y along that path, in the direction opposite to the collider's generating structure. Quantitatively, in the linear-Gaussian case, the induced partial covariance is −Cov(X,C)Cov(Y,C)/Var(C) (A-11).

**Hypotheses.** The graph structure; in the quantitative form, linearity and Gaussianity.

**Use.** Explains sign reversals after "adding controls", and explains why restricting the sample to survivors, completers, or escalators is a conditioning operation with causal consequences.

**Limitation.** The quantitative form is linear-Gaussian; the qualitative form (bias exists and is signed) holds generally, but the magnitude is model-specific.

### T-11 Markov equivalence and the limits of data-driven discovery

**Statement.** Two DAGs are Markov equivalent if and only if they have the same skeleton and the same v-structures (unshielded colliders). An equivalence class is represented by a CPDAG, in which some edges remain undirected.

**Hypotheses.** Faithfulness, for the observational equivalence to be detectable.

**Use.** Explains why constraint-based discovery (PC algorithm) returns partially oriented graphs, and why "the algorithm found the graph" is never a valid defence of an adjustment set.

**Limitation.** Any adjustment set that requires an orientation not fixed by the equivalence class is not determined by the data. In practice, the ambiguous edges are usually exactly the ones the decision depends on.

### T-12 Bias formulas for unmeasured confounding

**Statement.** For a single binary unmeasured confounder with associations RR_UD and RR_AU, the bias factor on the risk-ratio scale is B = RR_UD·RR_AU/(RR_UD + RR_AU − 1); the E-value is the common strength E at which B equals RR_obs, giving E = RR_obs + sqrt(RR_obs(RR_obs − 1)).

**Hypotheses.** A single confounder (or a set collapsing to one); the bias formula's own modelling assumptions, which differ slightly between the risk-ratio and risk-difference versions.

**Use.** Converts "we may have missed something" into a number that can be compared against measured covariates' strengths.

**Limitation.** The E-value is a *minimum* strength for the observed association to be fully explained; it is not a probability, not a confidence statement about the truth, and it does not handle multiple confounders jointly or differential measurement error.

### T-13 Positivity and the limits of the g-formula

**Statement.** The g-formula requires P(X=x | Z=z) > 0 for all z in the support of the target covariate distribution. Where this fails, the corresponding term in the sum is undefined, and the estimand must be restricted.

**Hypotheses.** None beyond the definition; this is a support condition, not an assumption about a model.

**Use.** Grounds the pre-analysis check of cell counts and the honest restriction of the estimand.

**Limitation.** Practical positivity is a continuum: probabilities of 0.001 are formally positive and statistically useless. Finite-sample overlap diagnostics (ESS of A-05, weight distributions) are needed, and there is no universal threshold — the tolerance must be declared before the analysis.

### T-14 What cannot be tested from data alone

**Statement.** The assumptions IV2, IV3, F1, and the completeness of the confounder set in an observational study are not testable from the observed distribution; they are claims about the data-generating process. Only their implications (over-identifying restrictions, placebo tests, negative controls) are testable, and those tests have power only against specific alternatives.

**Hypotheses.** Not applicable — this is a statement about the limits of inference.

**Use.** The honest framing for any report: name the untestable assumptions, say what evidence supports each (mechanism, timing, prior study), and bound the consequence if each is wrong.

**Limitation.** None; but the practical corollary is that a second analyst reviewing the *graph* adds more value than a second analyst re-checking the *fit*.

### Results at a glance

| Result | Gives you | Requires | Fails silently when |
|---|---|---|---|
| T-1 d-separation | a validity proof for Z | Markov, correct graph | graph omits a common cause |
| T-2 backdoor | identification by adjustment | BD-1, positivity | Z contains a descendant of X |
| T-3 do-calculus | identification in general graphs | acyclicity | the identified expression is unestimable |
| T-5 front-door | identification with an unmeasured confounder | no direct effect | a direct effect exists |
| T-6 propensity balance | dimension reduction | true propensity | balance never checked |
| T-7 double robustness | protection against one misspecification | correct set | the set is wrong |
| T-8 IV/LATE | identification from a shifter | exclusion, monotonicity | the first stage is weak |
| T-9 non-identification | a defensible "no" | nothing | replaced by a wide interval |
| T-10 collider bias | explains sign flips | correct graph | the extra variable looks predictive |
| T-11 Markov equivalence | why discovery is partial | faithfulness | the ambiguous edge is the decisive one |
| T-12 E-value | a sensitivity number | one-confounder model | reported as a probability |

**Takeaway:** the theorems are permission slips with conditions attached — the permission is void, not weakened, when a condition fails.

## 11. Interview / Exam Questions

Fourteen questions across recall, application, and judgment. At least eight require reasoning, calculation, selection, or diagnosis. Each answer is checked arithmetic where a number appears, and each carries the red flag that should end an interview answer.

**Q1 (recall, 30 seconds).** State the two clauses of the backdoor criterion.

*Answer.* A set Z satisfies the backdoor criterion relative to (X, Y) if no element of Z is a descendant of X, and Z blocks every path between X and Y that contains an arrow into X. The first clause preserves the treatment's assignment mechanism; the second removes non-causal association.

*Reasoning.* The clauses are not symmetric: the first is a prohibition (do not touch descendants), the second is a requirement (must block these paths). Reciting only the second clause produces over-adjustment.

*Red flag.* Saying "Z must block all paths between X and Y" — that would block the effect itself.

**Q2 (application, calculation).** Two strata: low, risk(1)=0.30, risk(0)=0.20, weight 0.6; high, risk(1)=0.80, risk(0)=0.72, weight 0.4. Compute the ATE and explain why it is not the crude difference of 33.7 pp.

*Answer.* Stratum contrasts are +0.10 and +0.08; ATE = 0.6(0.10)+0.4(0.08) = 0.06+0.032 = 0.092, i.e. +9.2 pp. The crude difference is inflated because treated units concentrate in the high-availability stratum (71.4% versus 23.1% of controls), and that stratum has a much higher baseline risk.

*Reasoning.* Crude difference = Σ_z [risk(1,z)P(z|X=1) − risk(0,z)P(z|X=0)], which uses different weights per arm; the ATE uses one weight vector for both. The 24.5 pp gap is entirely composition.

*Follow-up.* What if the decision is to keep the programme only for current participants? Then the ATT, +8.57 pp, is the right estimand.

*Red flag.* Reporting the crude number as "the effect" while describing it as causal.

**Q3 (diagnosis).** After adding "setup completion" to the model, the effect drops from +9.2 pp to approximately 0 and the mediator coefficient is highly significant. Diagnose and fix.

*Answer.* Setup completion is a descendant of the treatment and lies on the causal path, so it is a mediator; conditioning on it removes the indirect effect. Because essentially the whole effect flows through it, the adjusted estimate is ~0. Fix: remove it from the adjustment set, report the total effect as the headline, and treat the mediation decomposition as a separate secondary analysis with its own assumptions.

*Reasoning.* Conditioning on a mediator changes the estimand from the total effect to a controlled direct effect — a different quantity, not a better estimate of the same one.

*Red flag.* "The model fit improved, so the smaller number is more accurate."

**Q4 (diagnosis, hard).** A variable uncorrelated with treatment but strongly predictive of the outcome is added, and the estimate moves towards zero. What are the two competing explanations and how do you tell them apart?

*Answer.* Either (a) M-bias: the variable is a collider on a path X ← U₁ → C ← U₂ → Y, so conditioning opens a path that was closed and creates bias towards the null; or (b) the variable is a proxy for a genuine confounder, and the movement is partial correction of real bias. Discriminate by (i) checking whether the variable has near-zero marginal association with X but non-zero conditional association — the M-signature; (ii) checking whether the moved estimate agrees better with an external benchmark; (iii) examining the domain mechanism for a plausible U₂ that causes both C and Y without causing X.

*Reasoning.* Both produce a null-ward shift from a variable that "should not matter"; only the graph distinguishes them.

*Red flag.* Assuming all movement is correction, or assuming none is.

**Q5 (calculation).** A trial-style nudge has ITT_X = 0.12 and ITT_Y = 0.024. Give the per-complier effect and state the population it applies to. What if ITT_X were 0.02?

*Answer.* β_IV = 0.024/0.12 = 0.20, i.e. +20 pp for compliers — those whose take-up was moved by the nudge. With ITT_X = 0.02, β_IV = 0.024/0.02 = 1.20, i.e. 120 pp, which is impossible for a bounded outcome and signals that the ratio is uninformative at that first-stage strength. Report the ITT of 2.4 pp as primary.

*Reasoning.* The ratio blows up as the denominator approaches zero; the estimand also becomes a tiny, unrepresentative complier set.

*Red flag.* Quoting 120 pp, or quoting 20 pp as "the effect of the programme" for the whole population.

**Q6 (judgment).** When would you prefer reporting "not identified" over an estimate with a wide confidence interval?

*Answer.* Whenever the graph shows an open backdoor path with an unmeasured confounder and no admissible set exists. An interval asserts that a quantity has been estimated with sampling uncertainty; when the point estimate is measuring an association contaminated by an unmeasured cause, the interval describes the precision of the wrong number. Reporting non-identification plus the design change that would identify the effect is more informative and more honest.

*Reasoning.* Identification precedes estimation; a confidence interval is conditional on identification.

*Follow-up.* What would you put in the deck instead? The graph, the direction of the bias, an E-value, and the pilot design.

*Red flag.* "With enough data the interval will shrink and we will know" — data does not fix identification.

**Q7 (application).** An A/B test is impossible because the rollout cannot be held back, but a regional eligibility rule assigns the feature by a threshold on account age. What design does the threshold enable and what is its estimand?

*Answer.* A regression discontinuity if accounts cannot manipulate their age at the cutoff: the estimand is the effect at the cutoff for accounts just above versus just below, a local effect. Alternatively a fuzzy RD if eligibility changes take-up without determining it, in which case the estimand is a local complier effect at the cutoff. Either way the estimand is local, not the population average.

*Reasoning.* Continuity of potential outcomes at the cutoff is the identifying assumption; it is plausible for age thresholds precisely because age is not manipulable by the account.

*Red flag.* Reporting the RD estimate as "the effect of the feature for all accounts".

**Q8 (recall + reasoning).** Why does the propensity score allow adjustment on a scalar instead of a vector, and what does the property *not* say?

*Answer.* Because if treatment is independent of covariates given the true score, then conditioning on the score balances the entire covariate distribution. It does not say that a well-fitting propensity model (high AUC) produces balance: an estimated score balances only approximately, so balance must be checked covariate by covariate in the weighted or matched sample. It also does not say the score is a sufficient adjustment set if the covariates themselves were chosen invalidly.

*Reasoning.* The theorem is about the true score under unconfoundedness; finite-sample estimation error is a separate matter.

*Red flag.* "AUC 0.85 means the propensity model is good, so the weights are fine."

**Q9 (calculation, sensitivity).** An estimate gives risk 0.500 in the treated and 0.408 in the control. Compute the E-value and say what it means for a report.

*Answer.* RR = 0.500/0.408 = 1.2255; E = 1.2255 + sqrt(1.2255 × 0.2255) = 1.2255 + 0.5257 = 1.751. An unmeasured confounder would need associations of at least RR ≈ 1.75 with both treatment and outcome to explain the whole association. Since the strongest measured confounder has a larger association with the outcome, the claim is not robust and must be reported with the bound.

*Reasoning.* The bound compares against the strength of measured covariates; a bound below the strongest measured association is a warning, not a reassurance.

*Red flag.* Presenting the E-value as a probability, or as "the chance the result is due to confounding".

**Q10 (diagnosis).** A pipeline reports an effect of +9.2 pp and an ESS of 800 from n = 100,000. What do you check first?

*Answer.* The weight distribution. A 0.8% efficiency means a small number of units carry enormous weight, which is the numeric signature of near-positivity-violation. Check the propensity distribution by arm for separation, identify which covariates drive the extreme scores, and either restrict the estimand to the overlap region or trim at a pre-declared rule and re-report the changed estimand.

*Reasoning.* ESS is the honest denominator for precision; a 100x loss means the estimate rests on a handful of units, and the nominal interval is far too narrow.

*Red flag.* Reporting the nominal n-based interval, or capping weights after seeing that the result was unstable.

**Q11 (selection).** You have a valid backdoor set, good overlap, 60 covariates, and a moderate sample. Which estimator and why?

*Answer.* AIPW with cross-fitting, reporting the propensity-weight ESS and a balance table. Reasons: 60 covariates make exact stratification impossible; AIPW protects against misspecification of either model; cross-fitting keeps inference honest when both models are estimated on the same data. If overlap were marginal, restrict the estimand rather than rely on augmentation.

*Reasoning.* The choice is driven by diagnostics (many covariates → dimension reduction; uncertain specification → double robustness), not fashion.

*Red flag.* Choosing matching because it looks more rigorous, or choosing IPW without ESS.

**Q12 (judgment, ambiguity).** A stakeholder says: "Our analysis controls for account size, agent assignment, setup completion, support tickets in the first 90 days, and renewal status. It is the most rigorous model we have." How do you respond?

*Answer.* Two of those five are inadmissible: setup completion is a mediator, and support tickets in the first 90 days is either a descendant of treatment or a collider (or both); renewal status is the outcome itself in another guise. Agent assignment is the treatment mechanism and conditioning on it may block the effect. The set must be rebuilt from a graph with timestamps: pre-treatment causes of both treatment and outcome, and nothing downstream.

*Reasoning.* "Most rigorous" has been operationalised as "most covariates", which is the over-adjustment error. The response is to redraw the graph rather than to argue about fit.

*Red flag.* Accepting the set because the model has many controls and a good R².

**Q13 (calculation, estimand).** Given the Section 5 numbers, compute the ATT and explain when the two estimands would differ most.

*Answer.* ATT weights: 1000/3500 = 0.2857 and 2500/3500 = 0.7143, so ATT = 0.2857(0.10) + 0.7143(0.08) = 0.02857 + 0.05714 = 0.0857, i.e. +8.57 pp. The gap between ATE and ATT is largest when the effect is heterogeneous across strata and the treated group's stratum mix is far from the population's — exactly this case, where the low-availability stratum has the larger effect but is under-represented among the treated.

*Reasoning.* Both are weighted averages of the same stratum contrasts with different weights; the difference is zero iff the weights coincide or the contrasts are homogeneous.

*Red flag.* Reporting one number without saying which population it refers to.

**Q14 (judgment, changed constraint).** The graph is right, the set is valid, but leadership needs an answer in 48 hours and the as-of-dated covariate table takes two weeks to build. What do you do?

*Answer.* Answer with a design whose validity does not depend on the missing covariates if one exists — for instance an instrument already present in the data, or a threshold rule, or a pre-period comparison. If none exists, report the crude association with its quantified bias direction, the specific missing covariates and their plausible bias magnitude, and the decision rule revised for that uncertainty; do not silently substitute a convenient post-treatment covariate because it is available today.

*Reasoning.* Deadlines change the design choice, not the validity requirement. A clearly labelled weaker answer with a stated bound is usable; an unlabelled over-adjusted answer is not.

*Red flag.* "We will ship the estimate now and fix the covariates later" without a statement of which direction the bias runs.

### Scoring guide

| Dimension | Passing answer | Failing answer |
|---|---|---|
| Criterion recall | states both clauses of BD-1 | says "block all paths" |
| Calculation | substitution shown, units named, cross-checked | number only |
| Diagnosis | names competing causes and the discriminating check | jumps to a fix |
| Selection | conditions on diagnostics and estimand | names a favourite method |
| Judgment | names the unverifiable assumption and its consequence | asserts robustness |

**Takeaway:** the interviews that fail are almost never about arithmetic; they fail on stating the estimand, naming the assumption, and refusing to report a number that identification does not support.

## 12. Quick Reference Card

| Field | Content |
|---|---|
| **Core criterion** | Z is admissible iff no element is a descendant of X and Z blocks every path with an arrow into X (BD-1); then ATE = Σ_z [E(Y\|X=1,Z=z) − E(Y\|X=0,Z=z)]·P(Z=z), and it holds only under positivity, consistency, no interference, and a correct graph |
| **Valve rules** | Chain (→M→) and fork (←M→): conditioning *blocks*. Collider (→C←) and any descendant of one: conditioning *opens*. So mediators and colliders must be left out, even when they predict the outcome strongly |
| **Three estimators, one estimand** | Standardisation, IPW, and AIPW agree exactly under a saturated model — use that as a unit test. Report the Kish ESS (Σw)²/Σw²; an ESS far below n means the nominal interval is too narrow |
| **No backdoor set?** | Front-door only if no direct X→Y path exists; IV gives a complier-local effect requiring a strong first stage (report ITT_X and ITT_Y); otherwise report *not identified* with the bias direction and the design change that would identify it |
| **Unverifiable assumption** | Report an E-value: E = RR + sqrt(RR(RR−1)). Compare it with the strongest measured confounder's association; and add an omitted-variable bound γ·δ in outcome units for a named candidate |
| **Fastest diagnostics** | Cell counts by (X,Z) for positivity; weight distribution and ESS for precision; add one covariate at a time for over-adjustment; a negative-control outcome for residual confounding. All are cheap; none is optional |
| **Do not** | Choose covariates by R² or stepwise search; take standardisation weights from the analytic sample by habit; divide by a small first stage; report a confidence interval for a non-identified quantity; use causal verbs without a documented identification argument |
| **Estimand check** | Write the decision as a sentence with a population in it, then confirm the weights match that population. ATE and ATT differ materially whenever effects are heterogeneous and the treated mix differs from the population — in the worked case, 9.20 pp versus 8.57 pp |
| **Evidence standard** | Graph, estimand, set, estimator, diagnostics, and sensitivity are all artefacts of the analysis. Arithmetic can be VERIFIED; the completeness of the graph is OPEN by construction and must be labelled so |
