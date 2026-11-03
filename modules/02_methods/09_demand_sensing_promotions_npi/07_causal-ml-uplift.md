# Causal Machine Learning and Uplift — heterogeneous treatment effects and who responds to a promotion

> Module 09 — Demand Sensing, Promotions, and New Product Forecasting
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 05; Module 09 topics 04–06
> Version 2026.5

---

## 1. Why this topic exists

A promotion review has one number and one question. The number is the average lift: the event
sold 13.5 more units per store per week than the counterfactual. The question is the one the trade
team must answer on Monday: **which stores do we include next time, and which do we leave out?**

Topic 04 taught you to build the counterfactual and estimate the average. Topic 05 taught you to
adjust that average into profit. Neither answers the allocation question, because the answer is
not an average. It is a function: the effect of the promotion as a function of store, customer,
item, and timing. The statistical name is the **conditional average treatment effect** (CATE); the
marketing name for the same object is **uplift**.

The default answer in most retail organisations is a response model — a propensity score, a churn
score, a "likelihood to buy" — repackaged as a targeting tool. It is the wrong estimand, and it
fails in a direction rather than at random. A response model asks *who buys when promoted?* and
the answer is dominated by who buys anyway: the largest stores, the most loyal customers, the most
popular items. An uplift model asks *who buys more when promoted?* and the answer is dominated by
who changes behaviour, which is frequently small, mid-volume, and unremarkable in every
descriptive report the company already has.

The gap is not a rounding error. In the worked example in section 6, targeting the top 120 stores
by baseline volume loses €23,232 per event, while targeting the 120 stores with the highest
estimated uplift earns €16,680. Same promotion, same data, same budget; a different estimand.

The money at stake is the trade budget. Trade spend is routinely reported at 10–20% of gross sales,
and a large share of individual events do not return their cost (Bain & Company / Grocery
Manufacturers Association, as reported in Topic 05; Grade C). Allocating that budget by response
scores and category-manager intuition allocates it by the wrong function.

**Why machine learning entered, and why it is not enough.** Heterogeneity in retail is
high-dimensional and interactive: the effect of a display depends on format, competitive set, price
position, and week. Tree ensembles find those interactions without being told where to look, and
the causal-forest family attaches an inferential theory to that flexibility. But a learner
minimises prediction error on the outcome, while the object you need is a difference between two
predictions, one of which is never observed. Nothing in the supervised machinery notices whether
that difference is identified. A gradient-boosted model can rank stores beautifully by expected
sales and carry no information about who responds to the promotion. The algorithm is the easy
part; the identification condition is the topic.

**What this topic is and is not.** It is the heterogeneity layer over Topics 04–06: given a lift you
believe, estimate how it varies and decide who gets treated. It does not re-derive experimental
design — Module 05 owns randomisation, power, clustering, and interference, and this file points
there rather than repeating it. It does not estimate price elasticity (Topic 06) or build the
promotion profit and loss (Topic 05); section 6 reuses that arithmetic rather than rebuilding it.

> **Note:** The vocabulary is split across three communities that mean the same thing. Marketing
> scientists say *uplift*, *incremental response*, or *net lift*; econometricians say
> *heterogeneous treatment effects* or *CATE*; causal-inference practitioners reserve *ITE* for the
> unobservable individual effect. Section 9.3 explains why the split happened and why it causes real
> misunderstandings in meetings.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** the estimand of an uplift model as a CATE with an explicit conditioning set, unit,
   and time window, and **distinguish** it from a response probability or a churn score without
   consulting a reference.
2. **Explain** why a high-response unit can have zero or negative uplift, and **identify** that
   pattern in a segment table or a targeting list.
3. **Choose** between an S-learner, a T-learner, and an X-learner (single, two, and cross-model
   meta-learners) for a given arm-size imbalance, and **justify** the choice in terms of bias and
   variance rather than fashion.
4. **Describe** what "honest" means in a causal forest, why split selection and leaf estimation
   must use disjoint samples, and what goes wrong with confidence intervals when they do not.
5. **Compute** a Qini curve, a cumulative gain, and a gain-at-depth figure by hand for a small
   evaluation table, and **state** the normalisation convention used.
6. **Evaluate** an uplift model on held-out data containing both treated and control units, and
   **explain** why accuracy, AUC (area under the receiver operating characteristic curve), and
   response-lift deciles are not valid substitutes.
7. **Diagnose** confounding in observational promotion data, naming the selection mechanism and
   the design change that would address it.
8. **Specify** a treatment definition — any promotion, a depth band, feature and display, or a
   customer coupon — and **explain** why each definition requires a different design and yields a
   different answer.
9. **Write** a deployment protocol: the targeting threshold, the permanent holdout, the drift
   monitor, and the condition under which the model is withdrawn.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: two rankings of the same population that disagree. The disagreement is not a bug
in either model. It is the definition of the problem.

### 3.1 The estimand, stated once and precisely

For a unit `i` with covariates `X_i`, a binary treatment `W_i ∈ {0,1}`, and potential outcomes
`Y_i(1)` and `Y_i(0)`:

```text
   CATE(x) = E[ Y(1) - Y(0) | X = x ]

   Y(1)  outcome if treated      units per store-week, or € per customer
   Y(0)  outcome if not treated  same unit
   x     the covariate vector at which the effect is evaluated
   E[.]  average over units sharing the same covariate value
```

Three things about that definition carry most of the weight. **The unit is attached to the
outcome.** "+60 units per store-week" is meaningless without the unit and the 12-week window.
**The conditioning set is part of the claim.** `CATE(store format)` is a different estimand from
`CATE(format, competition, week)`, and a CATE at a point in 200-dimensional space is a ranking
score, not a segment average. **The expectation does the work that "ITE" hides.** The individual
effect `Y_i(1) − Y_i(0)` is never observed; only a conditional mean at covariate values with data
in both arms is estimable. Any per-customer "individual uplift" is a prediction of such a mean.

### 3.2 Response probability and uplift are different functions

A response model estimates `P(Y = 1 | X = x, W = 1)`: a level. An uplift model estimates
`E[Y(1) − Y(0) | X = x]`: a difference. When between-unit variance in baseline response exceeds
between-unit variance in effect — the normal state of retail — the response ranking is
approximately the baseline ranking, and the uplift signal is buried inside it.

```text
        THE SAME 240 STORES, RANKED TWO WAYS  (schematic, not data)

  PANEL A — RESPONSE PROBABILITY          PANEL B — UPLIFT (CATE)
  P(buy | promoted)                       E[Y(1) - Y(0) | store]

  60% | A A A A                          +50 |              C C
  50% | A A A A A                        +40 |          C C C
  40% | A A B B B                        +30 |      C C C
  30% | B B B B B                        +20 |  B B C C
  20% | B C C C                          +10 |  B B
  10% | C C                              0   |--A A A A A A A A----
   0% +------------------>              -10 |  A A A
        ranked by response                  +-------------------->
        probability                          ranked by uplift

  A = anchor stores, 38-58% response, uplift -8 units.
      HIGH RESPONSE, LOW UPLIFT.  A response model places them
      in the top decile; an uplift model places them at the bottom.

  B = mid-format stores, 22-35% response, uplift +6 units.

  C = small-format stores, 9-18% response, uplift +42 units.
      LOW RESPONSE, HIGH UPLIFT.  Almost never in the top response
      decile, and the only group whose promotion pays for itself.
```

Panel A is what the CRM team already has. Panel B is what the allocation decision needs. They are
computed from the same 240 stores, and they invert the ranking of the two most important groups.

### 3.3 The four segments, and why the negative one matters

| Segment | `Y(1)` | `Y(0)` | Effect | Retail reading |
|---|---|---|---|---|
| Persuadable | buy | not buy | positive | The only group promotion should target |
| Sure thing | buy | buy | zero | Buys anyway; the discount is a pure transfer |
| Lost cause | not buy | not buy | zero | No response; contact is wasted |
| Sleeping dog | not buy | buy | negative | Would have bought; the promotion suppresses the sale |

The first three are familiar to anyone who has run a campaign. The fourth changes decisions and
response models cannot see it: a loyal shopper who learns the cadence of the discount calendar, or
a lapsed customer reminded that the purchase is optional. In subscription settings the mechanism is
documented — a retention contact can increase cancellation among low-risk customers by surfacing a
decision they had not been considering. In retail the same pattern appears as a full-price buyer who
waits for the next promotion.

> **Note:** Negative effects are not a modeling artifact and should not be clipped to zero. If your
> estimated CATE distribution has no mass below zero on a category with deep, frequent discounts,
> suspect the model before celebrating the finding.

### 3.4 Two clocks, and what the machine learning buys

- **The identification clock** is slow and external. It is set by the design: randomiser,
  switchback, matched pair, or a category manager's judgement. No algorithm moves it. If assignment
  was confounded, more data does not fix it; it makes the wrong answer more precise.
- **The estimation clock** is fast and internal. It is set by the learner: trees, regularisation,
  honesty, smoothing. This clock can be moved by engineering.

Most causal-machine-learning projects spend 90% of their effort on the fast clock and inherit what
the slow clock already said. The machine learning buys flexible estimation, automatic interaction
discovery, and — in the forest family — an interval with an asymptotic justification. It does not
buy identification, overlap, a counterfactual, a treatment definition, a cost model, or a holdout
that survives deployment. Every one of those is a design decision that can destroy the model after
the algorithm has done its job perfectly.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Treatment | The intervention whose causal effect is being estimated | Used to mean "the campaign" as a whole, including its targeting | Binary `W ∈ {0,1}`, with a written definition |
| Control | Units assigned to no treatment | Called "baseline", which is a different object | `W = 0` |
| Potential outcome | The outcome a unit would show under a specified treatment assignment | Confused with the observed outcome | `Y(1)`, `Y(0)`; units per store-week |
| Counterfactual | The potential outcome not observed for a given unit | Used as a synonym for "forecast" | Same unit as `Y` |
| Fundamental problem of causal inference | Only one potential outcome is observed per unit | Treated as a data-cleaning defect | None |
| ATE | Average treatment effect over a stated population | Reported without naming the population | Units, e.g. units per store-week |
| ATT | Average effect on the units that were actually treated | Used interchangeably with ATE | Units |
| CATE | Conditional average treatment effect at covariate value `x` | Called "the uplift score" without naming the conditioning set | Units per unit of `X` |
| ITE | Individual treatment effect, `Y_i(1) − Y_i(0)` | Reported per customer as if it were estimated | Units; not identified |
| Uplift | Marketing term for the incremental outcome caused by treatment | Applied to any model that predicts response | Units, or probability points |
| Response probability | `P(Y = 1 | X, W = 1)` | Mistaken for uplift | Probability, 0–1 |
| Incremental response | Direct-marketing synonym for uplift | Assumed to equal observed response | Units |
| Net lift | Synonym for uplift; sometimes lift net of cannibalisation | Ambiguous without a stated adjustment | Units |
| Baseline | The counterfactual level absent treatment | Confused with the control-group mean | Units per period |
| Incrementality | The share of observed outcome that is causal | Used loosely as "lift" | Proportion |
| Confounder | A variable that affects both assignment and outcome | Called "a feature" and fed to the model | Covariate |
| Unconfoundedness | Treatment independent of potential outcomes given `X` | Asserted because "we control for a lot" | Assumption, untestable |
| Overlap / positivity | Every unit has positive probability of either arm | Checked by looking at the propensity range | `0 < e(x) < 1` |
| Propensity score | `e(x) = P(W = 1 | X = x)` | Used as a causal estimate instead of a weight | Probability |
| SUTVA | A unit's outcome depends only on its own treatment | Ignored in store networks with shared catchment | Assumption |
| Interference / spillover | One unit's treatment changes another unit's outcome | Called "halo" and left uncorrected | Cross-unit effect, units |
| Cluster randomisation | Randomising groups rather than individual units | Analysed at the individual level | Cluster identifier |
| Intention-to-treat (ITT) | Effect of assignment, regardless of compliance | Confused with the effect of receipt | Units |
| Meta-learner | A procedure that reduces CATE estimation to supervised learning | Called "a model" rather than an algorithm wrapper | Algorithm |
| S-learner | One outcome model with treatment as a feature | Assumed to capture interactions automatically | Single model |
| T-learner | Separate outcome models fitted in each arm | Used with a tiny treatment arm | Two models |
| X-learner | Two-stage imputation plus propensity-weighted combination | Treated as automatically superior to T | Three stages |
| R-learner | Residual-on-residual estimation of the effect | Confused with the doubly robust learner | Two nuisance models |
| DR-learner | Doubly robust pseudo-outcome regression | Assumed to repair a confounded design | Pseudo-outcome |
| Causal forest | An ensemble of trees estimating a local treatment effect | Called "a random forest" or "a black box" | Forest object |
| Honest splitting | Split selection and leaf estimation use disjoint subsamples | Confused with cross-validation | Split flag per tree |
| Subsampling | Each tree sees a random subset without replacement | Confused with bootstrapping | Fraction of `n` |
| Cross-fitting | Nuisance models fit on one fold and scored on another | Confused with tuning by cross-validation | Fold identifier |
| Local centering | Residualising outcome and treatment on `X` before growing the forest | Omitted, which leaves baseline variation in the split criterion | Residual |
| Qini curve | Cumulative incremental outcome against depth, sorted by score | Called "the uplift curve" without qualification | Units against fraction |
| Uplift curve | Per-prefix average uplift against depth | Used interchangeably with Qini | Units per treated unit |
| Cumulative gain | Cumulative incremental outcome at a given depth | Normalised inconsistently across libraries | Units, or index |
| AUUC | Area under the uplift curve | Quoted without its curve definition | Units squared |
| Qini coefficient | Normalised area between the model and random curves | Compared across datasets | Dimensionless, 0–1 |
| Policy value | Expected outcome under a specified decision rule | Confused with the ATE | Units |
| Targeting rule | The mapping from estimated score to action | Reduced to "top decile" with no cost model | Threshold, or budget |
| Breakeven uplift | The effect at which treatment profit is zero | Set to zero | Units |
| Persuadable | Responds only when treated | Used for "high score" | Segment label |
| Sure thing | Responds regardless of treatment | The majority of high-response units | Segment label |
| Lost cause | Never responds | — | Segment label |
| Sleeping dog | Responds less when treated | Clipped to zero as "noise" | Segment label |
| Holdout | A randomly withheld group never treated | Called "the test set" and then targeted | Proportion of population |
| Drift | Change in the joint distribution of `X`, `W`, and `Y` | Monitored only on features | Population stability index, effect change |
| Feedback loop | The policy changes the data that trains the next model | Ignored | None |
| Treatment definition | The exact rule mapping an observed event to `W = 1` | Left implicit in a dashboard | Codebook entry |
| Dose-response | Effect as a function of treatment intensity | Treated as binary | Depth bands, percent |
| Multi-valued treatment | More than two treatment levels | Collapsed to binary | Level code |
| Calibration | Agreement between predicted and realised uplift | Confused with discrimination | Slope, intercept |
| Rank invariance | A metric unchanged by monotone score transforms | The reason Qini cannot choose a threshold | Property |
| Leakage | Post-treatment information included as a feature | Invisible in accuracy metrics | Feature list |
| Winner's curse | Selected extremes overstate their true effect | Visible in segment tables | Bias, units |


## 5. The mechanics, step by step

This section runs from the decision backwards: question, identification, estimator, evaluation,
deployment. The order matters. Most failed uplift projects run it the other way round —
estimator first, question discovered later.

### Phase A — Define the estimand and the decision

**Step 1. Write the decision rule before writing the model.**

- Produce one sentence: "treat unit `u` in period `t` if `f(x) > c`", with `c` in money.
- Branch: **ranking under a fixed budget, or threshold against a cost?** A ranking ("we can fund
  120 stores") needs only a monotone score; a threshold ("promote where expected profit is
  positive") needs a calibrated effect.
- Failure if skipped: a model tuned for ranking is deployed with a threshold set at zero because
  nobody computed the cost.

**Step 2. Define the treatment precisely.**

"Promotion" covers at least four treatments, and each has a different estimand, design, and
answer.

| Treatment definition | `W = 1` means | Estimation requirement | What it can answer |
|---|---|---|---|
| Any promotion | A temporary price reduction above a stated threshold | Binary design with a clean control | Whether to run an event at all |
| Depth band | A specific discount band, e.g. 15–20% | Multi-valued treatment; generalised propensity or binned design | How deep, and whether returns diminish |
| Feature and display | In the circular or on a display, price held | Separate arms per mechanism; compliance data | Whether to buy attention or buy price |
| Customer coupon | An individually addressed offer | Customer-level randomisation; household interference check | Which customers, at what cost |

- Output: a rule mapping every store-week to exactly one level, plus counts per level.
- Branch: **does the data support the definition?** If the calendar records only "promo yes/no",
  depth effects need a new test. If feature and display always run with a price cut, they cannot
  be separated, and no model will do it.
- Failure if skipped: a model trained on a mixture is deployed as one of its components. The CATE
  is a weighted average of price, attention, and depth effects, with weights that differ by store.

> **Note:** In most retail data the treatment is neither binary nor cleanly recorded. Feature and
> display are often unlogged, depth is computed from a regular price that itself moves, and "no
> promotion" weeks sometimes contain unlogged markdowns. The definition is a data-quality claim,
> and it should be tested against the execution audit.

**Step 3. Define the outcome, its unit, and its window.**

- Output: an outcome column with a unit (units, revenue, margin), an aggregation (store-week,
  customer-trip, item-week), and a window (the event plus any carry-over).
- Branch: **is the outcome at the same unit as the treatment?** A store-level treatment with a
  customer-level outcome is a cluster-randomised design whether or not you intended it, and the
  standard errors must reflect that.
- Failure if skipped: intervals are too narrow by the design effect, and the targeting list is
  longer than the evidence supports.

**Step 4. State the conditioning set and the population.**

- Output: the covariate vector `X`, the population over which the CATE is averaged, and the period.
- Branch: **are covariates measured before treatment?** Display compliance, in-event sell-through,
  and promo-week traffic are post-treatment variables. Including them is leakage: the model
  learns the consequence of treatment and calls it a cause.
- Failure if skipped: accuracy looks excellent, the effect estimate is biased, and the error is
  invisible in every predictive metric.

### Phase B — Identification

**Step 5. Establish the design, and name it.**

Five designs support uplift estimation in retail, in descending order of credibility:

1. **Store-level randomised controlled trial (RCT).** Stores assigned by a randomiser, stratified
   by format and region. The strongest design; Module 05 owns the mechanics.
2. **Cluster or matched-pair randomisation.** Stores paired on baseline volume, format, and
   competitive set, then one of each pair randomised. Better balance, fewer degrees of freedom.
3. **Switchback or geo-test.** Treatment alternates across time within a market, or across matched
   markets. Handles store-level confounding and introduces time confounding.
4. **Natural experiment.** A funding change, a supplier-driven rollout with a sharp cutoff, a
   regression-discontinuity rule on store size. Credible if the cutoff is exogenous and the
   running variable is not manipulable.
5. **Observational with covariates only.** Treatment chosen by category managers, adjusted by a
   propensity score. The weakest design, and the one most uplift projects actually have.

- Branch: **did assignment use information in `X`?** If managers promoted where they expected a
  response, and the expectation rested on local knowledge not recorded in `X`, unconfoundedness
  fails and the model reproduces the managers' selection.
- Failure if skipped: the model learns where the company already promotes and recommends
  promoting there — a machine-learning restatement of the status quo.

**Step 6. Check overlap and balance.**

- Output: a propensity distribution by arm, standardised mean differences, and a count of units
  outside common support.
- Branch: **are there units with propensity near 0 or 1?** Those units have no counterfactual.
  Estimates for them are extrapolation. Trim them, or state that the CATE is not identified there.
- Failure if skipped: the top of the targeting list is populated by units whose effect exists
  only in the model's functional form.

**Step 7. Check interference and spillover.**

- Output: an assessment of which control stores share customers with treated stores.
- Branch: **do treated and control units share a market?** If a promoted store draws shoppers from
  a nearby control store, the control mean is depressed and the effect is overstated. Module 05's
  interference designs apply directly.
- Failure if skipped: the effect is inflated in proportion to store density, so the model
  recommends promoting in dense markets for the wrong reason.

### Phase C — Estimation

**Step 9. Choose the learner family, and know what each one assumes.**

The meta-learner family reduces CATE estimation to supervised learning. The trade-off is between
sharing information across arms and protecting each arm's response surface.

```text
   S-LEARNER                  T-LEARNER                 X-LEARNER
   one model                  two models                three stages

   mu(x, w)                   mu1(x)   mu0(x)           stage 1: mu1, mu0
      |                          |        |                       |
   tau = mu(x,1) - mu(x,0)    tau = mu1 - mu0          stage 2: impute
                                                          D1 = Y1 - mu0(X1)
                                                          D0 = mu1(X0) - Y0
                                                             |
                                                          regress D on X
                                                             |
                                                          stage 3: combine
                                                          with weights g(x),
                                                          often g = e(x)

   shares strength            protects each arm's      borrows the well-
   across arms; may shrink    surface; noisy when      estimated arm to
   tau toward zero            one arm is small         impute the small arm
```

- Branch: **what is the arm ratio?** At 50/50 the T-learner is defensible. At 20/80 the control
  model is much better estimated, and the difference inherits the weaker arm's error. At 5/95 the
  treated arm may not support a flexible model at all.

**Step 10. The S-learner: one model, treatment as a feature.**

Fit `mu(x, w)` on pooled data with the treatment indicator as one column, then
`tau(x) = mu(x, 1) - mu(x, 0)`.

- Strength: all data inform one surface, so variance is often lower when the arms are similar.
- Weakness: treatment is one column among dozens. Regularisation may never find it worthwhile to
  split on `W`, and the model silently returns `tau = 0`. This is the mechanism the causal-machine-
  learning literature calls regularisation-induced confounding.
- Reasonable when the treatment strongly predicts the outcome, the arms are balanced, and the
  effect is believed smooth in the same covariates that drive the level. Its failure mode is
  visible: a flat CATE distribution.

**Step 11. The T-learner: two models, one per arm.**

Fit `mu1(x)` on treated units and `mu0(x)` on control units, then subtract.

- Strength: each arm gets its own surface, so the model can represent a promotion that changes
  the shape of demand, not just its level — common when a display creates a threshold effect.
- Weakness under imbalance: the control model sees far more data; the treated model is noisy; the
  difference inherits the noisier arm's error, and independent fits can disagree for reasons that
  have nothing to do with the effect. That disagreement is indistinguishable from heterogeneity.
- Reasonable with balanced arms, a large sample, and a belief that the surfaces genuinely differ.

**Step 12. The X-learner: impute the missing arm, then combine.**

Introduced by Künzel, Sekhon, Bickel, and Yu (2019):

1. Fit `mu1` and `mu0` as in the T-learner.
2. Impute effects: for treated units `D1 = Y1 - mu0(X1)`; for control units `D0 = mu1(X0) - Y0`.
   Each imputation uses the *other* arm's model, which is the well-estimated one under imbalance.
3. Regress `D1` on `X` among treated to get `tau1`, regress `D0` on `X` among controls to get
   `tau0`, and combine: `tau(x) = g(x) tau0(x) + (1 - g(x)) tau1(x)`, with `g` usually the
   propensity score.

- Strength: designed for many controls and few treated. The imputation lets a large control
  sample inform the effect for a small treated group.
- Weakness: depends on the propensity model in stage 3, and weights near 0 or 1 amplify noise. It
  also assumes unbiased imputation, which returns you to unconfoundedness — there is no free
  lunch, only a different place to pay.

**Step 13. The causal forest: a local moment, not a black box.**

A causal forest grows many trees that partition the covariate space and averages the within-leaf
effect estimates. Three mechanical facts decide whether the output is an estimate or decoration.

- **It solves a local moment equation, not a prediction problem.** Each split maximises the
  heterogeneity of the estimated effect between child leaves, not the fit to `Y`. In the
  generalized-random-forest formulation of Athey, Tibshirani, and Wager (2019) the criterion is a
  local estimating equation, and the causal forest is the case where the moment is the treatment
  effect. That is why it can rank uplift while a random forest on `Y` cannot: different objectives.
- **Local centering removes baseline signal before splitting.** In the standard implementation,
  `Y` and `W` are residualised on `X` using regression forests — the propensity `e(x)` and the
  outcome mean `m(x)` — and the forest grows on the residuals. Without this, the split criterion
  is dominated by "which stores are big", and the forest partitions on size rather than on
  responsiveness.
- **Honesty is what makes the interval meaningful.** Honest splitting, formalised by Wager and
  Athey (2018), uses disjoint subsamples for split selection and leaf estimation. The estimate in
  a leaf is computed on data that did not influence which leaf it is in.

### 5.5 What "honest" means, and why it is necessary

Suppose you run a promotion in 40 stores, find the store with the largest measured lift, and
report that store's lift as the expected effect of promoting a similar store. The number is biased
upward because you selected it for being extreme: selection and estimate used the same data.

A non-honest causal forest does this at every leaf, with noise working through the split
criterion. Two consequences follow. The point estimate is biased away from zero, so the apparent
heterogeneity is larger than the real heterogeneity. And the variance estimate is too small,
because the same data chose the leaf and estimated its mean, so intervals under-cover and a
"significant" segment can be an artifact of the split search.

With honesty, the leaf's estimation sample is independent of the split, so the leaf mean is
conditionally unbiased and its variance is estimable. Wager and Athey prove pointwise consistency
and asymptotic normality under conditions on the subsample size, and the infinitesimal jackknife
gives a variance estimate that tracks the truth. That is the sense in which honesty is necessary
for valid inference: without it you have a number and no interval, and a number without an
interval cannot support a decision under uncertainty.

The price is real. Honest forests need more data for the same precision, because each tree uses
part of the sample twice. In a 480-store panel with 160 treated, an honest forest with default
leaf sizes will produce intervals so wide that every segment overlaps every other. That is not a
failure of the method; it is the method telling you the truth about the sample.

> **Note:** Honest splitting is not cross-validation, and the two are confused in code review.
> Cross-validation estimates out-of-sample prediction error by holding out folds. Honesty holds
> out a subsample within the effect estimation, after the split is chosen, per tree. A pipeline
> can have impeccable cross-validation and a dishonest forest.

### 5.6 Cross-fitting, and why nuisance models need it too

Doubly robust and residual-based approaches need the propensity `e(x)` and the outcome mean
`m(x)`. If those are fitted and scored on the same rows they overfit, the residuals fed into the
effect equation are too small, and the effect estimate is biased — usually toward zero — with an
understated variance.

Cross-fitting is the fix: split into `K` folds, fit the nuisance models on `K-1`, predict on the
held-out fold, repeat. The effect is computed on residuals that are genuinely out-of-sample. The
general principle is worth stating plainly, because it is the prerequisite for everything in this
phase: **a learner is a function fitted to data, and a flexible learner fitted and evaluated on
the same rows will look better than it is. Cross-fitting is fitting on one part and scoring on
another.**

### Phase D — Evaluation

**Step 14. Evaluate on held-out data containing both arms.**

- Output: every metric below computed once, on data not used for fitting, tuning, segment
  selection, or threshold selection.
- Branch: **does the holdout contain treated and control units?** A treatment-only holdout
  supports no uplift metric. A holdout used to choose segment boundaries is no longer a holdout.
- Failure if skipped: the reported Qini is a training-set Qini, optimistic by an amount that grows
  with model flexibility.

**Step 15. Understand why accuracy fails.**

Accuracy, AUC, precision, recall, and log-loss all score `P(Y | X, W)`. Uplift is a contrast
between two such functions. A model can predict `Y` perfectly in both arms and carry no
information about `tau`: if treatment shifts every outcome by the same amount, the surfaces are
parallel, prediction is excellent, and the effect is constant. Conversely, a model with mediocre
accuracy can rank effects well, because ranking depends on differences, not levels. The metric
must target the same object as the decision.

**Step 16. Compute the Qini curve.**

The Qini curve, named by analogy with the Lorenz curve and popularised by Nicholas Radcliffe in
2007, sorts units by predicted uplift and plots cumulative incremental outcome against depth. For
a prefix of `k` units containing `N_t(k)` treated and `N_c(k)` control units with outcome sums
`Y_t(k)` and `Y_c(k)`:

```text
   Q(k) = Y_t(k) - Y_c(k) * ( N_t(k) / N_c(k) )
```

The ratio rescales the control outcome to the size of the treated group. Three practical points:
the random baseline is not zero but a line from the origin to `Q(N)`; the normalisation is not
standardised across libraries; and unbalanced arms make `N_t(k)/N_c(k)` swing across the prefix,
producing noise that has nothing to do with model quality. Corrections exist and they change the
number — state which one you used.

**Step 17. Distinguish the three curve names.**

| Name | Vertical axis | Horizontal axis | Sensitive to |
|---|---|---|---|
| Cumulative gain | Cumulative incremental outcome, units | Number or fraction targeted | Outcome scale |
| Qini | Cumulative incremental outcome, rescaled by arm ratio | Fraction targeted | Arm ratio and outcome scale |
| Uplift curve | Average uplift per treated unit in the prefix | Fraction targeted | Depth profile, not total |

A team can report "the uplift curve" and mean any of the three. Ask for the axis labels before
comparing anything, including a comparison to a published number.

**Step 18. Compute policy value, and understand the disagreement with Qini.**

Policy value is the expected outcome if a rule `pi` is deployed: `V(pi) = E[Y(pi(X))]`, estimated on
held-out data by inverse propensity weighting (IPW, which reweights observed outcomes by the
inverse probability of the assignment received) or a doubly robust estimator.

- **For Qini.** Cheap, no cost model, compares candidate models on a common scale within a
  dataset, and is the language of the direct-marketing literature.
- **Against Qini.** It is rank-invariant: any monotone transform of the score gives the same
  curve. Deployment is not rank-invariant — a threshold depends on where the score sits relative
  to cost. A well-ranked, badly calibrated model produces a bad policy, and Qini will not notice.
- **For policy value.** Evaluates the rule actually deployed, including threshold and budget.
- **Against policy value.** Higher variance, dependent on the propensity model and the policy
  class; two estimates with overlapping intervals are often inconclusive.

A 2026 preprint benchmark (UpliftBench, not peer-reviewed) reports that for a sign-threshold
policy, selecting models by Qini produced 14–15% more regret than selecting on policy risk, and
that calibrating the threshold removed about 81% of the gap; under a budgeted objective the gap
largely disappeared, because ranking is all a budgeted rule needs. Treat that as a well-argued
preprint finding, not a settled result.

The practical synthesis: **Qini to compare candidate models in development, policy value to make
the go/no-go and threshold decision, and both reported on the same held-out set.**


**Step 20. Set the threshold against cost, not zero.**

The breakeven uplift is `u* = c / m`, where `c` is the per-unit treatment cost and `m` the margin
per incremental unit. With heterogeneous margins the rule is `tau(x) * m(x) > c(x)`.

- Branch: **point estimate or lower confidence bound?** The lower bound is conservative and
  protects against sleeping dogs; the point estimate maximises expected value if the model is
  unbiased. State which you chose.
- Failure if skipped: the rule promotes units whose interval spans zero, and the programme loses
  money on the margin.

**Step 21. Maintain a permanent holdout.**

- Output: 5–10% of units, randomly selected, never targeted, audited on a stated schedule.
- Branch: **is the holdout truly never treated?** A holdout store promoted through another channel
  is contaminated, and the next model's counterfactual is corrupted.
- Failure if skipped: after the first deployment the training data contains no controls for the
  targeted population, and the next model can learn only from residual variation the policy left
  behind. This is the feedback loop, and it is the most common way a working uplift model decays
  into an unfalsifiable dashboard.

**Step 22. Monitor drift in the effect, not only in the features.**

Four monitors: feature drift (population stability index by covariate), score drift (distribution
of predicted uplift), effect drift (holdout ATE by period), and calibration drift (realised
against predicted, by decile). Feature drift is common and often harmless; effect drift is rarer
and always material. A plan that watches only features will miss a change in the promotion's
economics.

**Step 23. Write the withdrawal condition.**

Pre-register what would switch the model off — for example, "holdout ATE below 5 units for two
consecutive quarters, or realised uplift in the targeted decile below breakeven for one quarter".
A condition that requires an experiment the company will not run is decoration. Without one, the
model is never withdrawn, because nobody agreed in advance what failure would look like.

## 6. Worked example

A grocery retailer runs a 12-week promotion on one category across 480 stores: 15% off shelf
price with feature and display. The retailer's net cost of the discount, after the manufacturer's
trade allowance, is €0.30 per unit sold; margin per unit is €5.00; fixed cost per promoted store is
€50. All figures are illustrative and constructed to be internally consistent.

### 6.1 The design and the estimates

Randomisation is at store level, stratified by format, 1:2 treatment to control: 160 promoted, 320
not. The unit is the store-week, averaged over the 12 weeks; the outcome is units sold per store
per week. Three formats were pre-specified before the data were seen: urban convenience (120
stores), large suburban supercentre (240), and rural discount (120). Pre-specification matters: if
segments were chosen after seeing results, they would be selected extremes and the intervals below
would be invalid.

| Segment | Stores | Treated | Control | Mean units, T | Mean units, C | Difference | 95% interval |
|---|---:|---:|---:|---:|---:|---:|---|
| A — urban convenience | 120 | 40 | 80 | 370 | 310 | +60 | +37.2 to +82.8 |
| B — suburban supercentre | 240 | 80 | 160 | 512 | 510 | +2 | −14.1 to +18.1 |
| C — rural discount | 120 | 40 | 80 | 270 | 280 | −10 | −32.8 to +12.8 |
| All stores | 480 | 160 | 320 | 416.0 | 402.5 | +13.5 | +2.1 to +24.9 |

The intervals assume a within-segment standard deviation of 60 units per store-week: the standard
error is `60 × sqrt(1/40 + 1/80) = 11.6` for A and C, `60 × sqrt(1/80 + 1/160) = 8.2` for B, and
`60 × sqrt(1/160 + 1/320) = 5.8` overall.

Read the table carefully; it contains the argument of this topic. The average effect is +13.5
units, 3.4% of the control mean, positive with an interval excluding zero, so the promotion
"worked" by the standard test. Yet only segment A is distinguishable from zero. The spread of
point estimates, from −10 to +60, is 70 units against standard errors of 8 to 12 — larger than the
noise, which is the precondition for heterogeneity to be worth modelling, but only just, and only
for one segment.

> **Note:** The overall interval, +2.1 to +24.9, comes from a sample underpowered for a precise
> average, let alone fine segmentation. An 80%-powered test of a 13.5-unit effect at this variance
> would need roughly 316 stores per arm at equal allocation — far more than the 480 available. The
> example is realistic precisely because it is underpowered. Most retail promotion tests are.

### 6.2 The economics

Profit per store for uplift `u` on baseline `b`:

```text
   profit(u) = 5.00 * u  -  0.30 * (b + u)  -  50
   breakeven:  4.70 * u = 0.30 * b + 50
```

At the overall baseline of 402.5 units, `u* = (120.75 + 50) / 4.70 = 36.3` units, or 9.0% of
baseline. Below that, the promotion loses money even though it "lifted sales".

| Segment | Baseline | Uplift | Incremental margin | Discount giveaway | Fixed cost | Profit per store |
|---|---:|---:|---:|---:|---:|---:|
| A — urban convenience | 310 | +60 | €300.00 | €111.00 | €50 | **+€139.00** |
| B — suburban supercentre | 510 | +2 | €10.00 | €153.60 | €50 | **−€193.60** |
| C — rural discount | 280 | −10 | −€50.00 | €81.00 | €50 | **−€181.00** |
| Weighted average | 402.5 | +13.5 | €67.50 | €124.80 | €50 | **−€107.30** |

The average store loses €107. That is not a contradiction of the positive average lift; it is the
arithmetic of a discount paid on the baseline. The promotion gives away €0.30 on every unit sold,
including the 400 that would have sold anyway, and the incremental margin does not cover it.
Topic 05 made this point; here it becomes the reason heterogeneity matters. The average is
negative and a subgroup is strongly positive, so allocation is worth more than the average
decision.

### 6.3 Three allocation rules

| Rule | Stores selected | Mean true uplift | Profit per store | Total profit |
|---|---:|---:|---:|---:|
| Promote everyone | 480 | +13.5 | −€107.30 | −€51,504 |
| Top 120 by baseline volume (response proxy) | 120 | +2 | −€193.60 | −€23,232 |
| Top 120 by estimated uplift (CATE) | 120 | +60 | +€139.00 | **+€16,680** |

The response rule selects the largest stores, all in segment B, where the true uplift is +2 and
the promotion loses €194 per store. The uplift rule selects segment A: smaller, less impressive in
every descriptive report, and the only group where the promotion pays. The difference between the
two targeted rules is €39,912 per event; the difference between the response rule and doing
nothing is a loss of €23,232.

### 6.4 The conservative version

Segment A's interval is +37.2 to +82.8. Using the lower bound rather than the point estimate:

```text
   profit(37.2) = 5.00 * 37.2 - 0.30 * (310 + 37.2) - 50
                = 186.00 - 104.16 - 50 = +€31.84 per store
   total        = 120 * 31.84 = +€3,821
```

Still positive, but €12,859 less than the point-estimate programme. That gap is the price of
honesty about uncertainty, and it is the number to show a finance committee that asks why the
analyst is being pessimistic.

### 6.5 What the observational version would have said

Without randomisation, with category managers choosing stores where they expected a strong
response, a naive comparison shows promoted stores averaging 512 units against 402 for the rest —
a 27% difference that is almost entirely selection, because the promoted stores are the
supercentres. Conditioning on format collapses it to +13.5, and conditioning further on competitive
intensity and region moves it again.

The lesson is not that observational data are useless. The observational estimate is a statement
about *which stores were promoted*; the causal estimate is a statement about *what promotion does*.
When the allocation rule is what you are trying to improve, the first is actively misleading,
because it encodes the rule.

## 7. Code, query, or calculation

The calculation below is the Qini curve for eight stores in a matched-pair evaluation. Each store
contributes one promoted week and one matched non-promoted week, so both potential outcomes are
observed and `delta` is a direct estimate of that store's effect. In a parallel-arm trial you
would not have per-store deltas; you would use the group-scaled estimator of section 5, and the
same cautions would apply.

```text
store  score   Y(0)  Y(1)  delta  cum_delta  random_line
S1     0.95     400   520    120        120        51.25
S2     0.90     380   470     90        210       102.50
S3     0.80     150   300    150        360       153.75
S4     0.70     620   640     20        380       205.00
S5     0.60     250   290     40        420       256.25
S6     0.50     200   180    -20        400       307.50
S7     0.40     900   880    -20        380       358.75
S8     0.30     120   150     30        410       410.00

total delta = 410    mean = 51.25
area(model)   = 2475   trapezoid, origin to k = 8
area(random)  = 1640   triangle, 0.5 * 8 * 410
area(perfect) = 2695   sorted by true delta: 150,120,90,40,30,20,-20,-20
qini          = (2475 - 1640) / (2695 - 1640) = 0.791
gain at 50%   = 380 units, against a random expectation of 205
peak of curve = k = 5 (420 units); the last three stores destroy 10 units
```

Read the table for the two facts that matter.

**Fact one: the highest-response store is not the highest-uplift store.** S7 has the largest
baseline, 900 units, and a negative effect of −20. It would be first on any response or volume
ranking and last on any uplift ranking. S3 has a baseline of 150 units and the largest effect,
+150. A model trained to predict the outcome would learn that S7 is the best store in the
estate; a model trained to predict the effect would learn that promoting S7 is a mistake.

**Fact two: the curve turns down.** Cumulative gain peaks at k = 5 with 420 units and falls to
410 by k = 8. The bottom three stores, taken together, contribute −10 units. A targeting rule
that promotes everyone destroys 10 units relative to a rule that stops at the fifth store. This
is the empirical form of the sleeping-dog segment: at the bottom of the ranking, promotion is
worse than nothing.

The Qini coefficient of 0.791 is implausibly high for a real retail model. Real values on noisy
store data are typically far lower and are not comparable across datasets, because the
normalisation depends on the perfect-model area, which depends on the outcome distribution and
the sample size. The number is useful as an arithmetic check and useless as a benchmark.

Two checks before you trust any Qini you compute:

- Confirm that the evaluation set contains both arms, and that it was not used to choose the
  score, the segment boundaries, or the threshold.
- Confirm that the random line is drawn from the origin to the full-depth value. A curve that
  starts at a non-zero value is usually plotting a per-prefix average and calling it a cumulative
  sum.

## 8. Data, standards, and regulatory context

Uplift work is constrained less by statistical standards than by the identity, event, and
personal-data standards that determine whether the treatment and the outcome can be joined at
all. A model built on a broken join is not a weak model; it is a model of the join error.

### 8.1 The minimum data requirement

| Field | Unit | Typical source | Why the design needs it |
|---|---|---|---|
| Store identifier | GLN | Store master | Defines the unit of randomisation and the join key |
| Item identifier | GTIN | Product master | Joins the promotion calendar to sales |
| Week | ISO week | POS extract | Defines the time bucket; must match across sources |
| Units sold | Units per store-week | POS or syndicated data | The outcome, or the input to revenue and margin |
| Regular price | € per unit | Price file | The denominator of the discount depth |
| Promoted price | € per unit | Promotion calendar | The numerator of the depth |
| Feature flag | 0/1 | Circular or retailer portal | Separates the attention mechanism from the price mechanism |
| Display flag | 0/1 plus audit | Store execution audit | The treatment actually delivered, not the one planned |
| Trade allowance | € per unit | Trade promotion management system | Converts gross discount to retailer net cost |
| Unit cost and margin | € per unit | Finance | The breakeven threshold |
| Competitive price index | Index | Syndicated data | A confounder and a CATE covariate |
| Store format | Category | Store master | A stratifier and a segment definition |

Three of these are routinely missing and each one breaks a different part of the design.

**Execution data.** The promotion calendar records intent; the display audit records delivery. A
store assigned to treatment that never installed the display is a compliance failure, and the
correct analysis is by intention-to-treat with the compliance rate reported separately. If you
analyse by delivered treatment, you have broken the randomisation and reintroduced selection.

**Trade allowance.** Without it, the discount cost is overstated and every store looks
unprofitable. With it, the retailer's true cost is visible and the breakeven threshold moves. The
allowance is frequently recorded at category level rather than item level, which forces an
allocation assumption that should be stated.

**Competitive price.** Competitor pricing during the event is a confounder and a genuine effect
modifier: a 15% cut against a competitor who also cut is a different treatment from a 15% cut
against a competitor who did not. If it is unavailable, say so; do not assume it away.

### 8.2 Standards that constrain the join

| Standard | Governs | What it constrains for uplift work | Grade |
|---|---|---|---|
| GS1 GTIN | Trade item identification | Whether the promoted item and the sold item are the same object | B |
| GS1 GLN | Location identification | Whether the store in the calendar is the store in the POS feed | B |
| GS1 EPCIS 2.0 | Event data | Whether display installation and removal can be verified | B |
| ANSI ASC X12 852 | Product activity data | The weekly store-level sales and inventory feed used by many retailers | B |
| GS1 General Specifications | Identification rules | The barcode-level identity that makes the join possible | B |

None of these standards defines a treatment, an estimand, or an evaluation metric. They define
identity and event records. That distinction matters when a vendor claims "standards-compliant
uplift": the compliance claim is about the data plumbing, not about the causal claim.

> **Note:** The most consequential data standard in this topic is not a formal one. It is the
> promotion calendar's definition of an event. Two retailers can both be "GS1 compliant" and
> still disagree about whether a 3% temporary price reduction is a promotion. That definition
> determines the treatment, and the treatment determines the answer.

### 8.3 Personal data and automated decisions

Store-level uplift analysis uses aggregated sales and does not process personal data. Customer-
level uplift does, and the obligations change accordingly.

| Requirement | Jurisdiction | What it constrains | Grade |
|---|---|---|---|
| GDPR (Regulation (EU) 2016/679), including the provisions on profiling and automated decision-making | European Union | Lawful basis, transparency, and the right not to be subject to solely automated decisions with significant effects | B |
| Regulation (EU) 2024/1689 (AI Act) | European Union | Risk classification, transparency, and prohibitions on manipulative techniques; applicability to a given targeting system requires legal review | B |
| Sectoral and national consumer-protection rules on promotional pricing claims | Various | What may be advertised as a discount, and against which reference price | B |

Two practical consequences follow. First, the choice of unit of analysis is also a choice of
regulatory regime: a store-level model avoids most personal-data questions that a customer-level
model must answer. Second, a targeting model that decides which individuals receive an offer is
closer to an automated decision than a model that produces a report, and the transparency
obligations attach to the decision, not to the algorithm's sophistication.

> **Note:** This section is educational and not legal advice. The applicability of the AI Act to
> a retail targeting system depends on its purpose, its effect on the person, and the member
> state's implementation. Obtain a legal review before deployment, not after.

## 9. Industry practice and case evidence

### 9.1 Origin, and why two literatures use different words

Uplift modelling was built by direct marketers solving a mailing-list problem: a fixed budget, a
randomised campaign flag, and a need to rank customers. Nicholas Radcliffe's 2007 article "Using
control groups to target on predicted lift" is the usual starting point; it introduced the Qini
curve and the four-segment vocabulary used in this file. (Grade C.) Econometricians approached the
same function from the other side, needing a parameter with a confidence interval and later a
policy rule. The vocabulary split is not arbitrary; it reflects what each field needed the estimate
for.

| Community | Typical design | What it needed | Canonical metrics |
|---|---|---|---|
| Direct marketing | Randomised mailing or email | A ranking under a fixed budget | Qini, cumulative gain |
| Econometrics | Experimental or observational, with inference | A parameter and an interval | CATE, standard errors, policy value |
| Machine learning | Large observational data | A scalable estimator with a guarantee | Honest forests, doubly robust scores |

The reconciliation is one sentence: **Qini is a ranking metric for a fixed-budget decision; a
confidence interval is an uncertainty statement for a threshold decision; a project with a
threshold decision that reports only Qini has not finished.**

### 9.2 What the retail evidence shows

- **Customer-level uplift works, with caveats.** Gubela, Lessmann, and Stöcker (2022) benchmark
  multiple-treatment uplift across eight retailer campaign datasets and find specialised methods
  outperform causal forests and separate-model approaches on Qini-type metrics. The setting is
  customer targeting, not store-level trade promotion. (Grade A.)
- **Profit uplift beats revenue uplift.** Baier and Stöcker (2021), on 155,000 German fashion
  customers, find profit uplift — netting margin and discount — substantially outperforms response
  and revenue uplift for target selection. (Grade A.)
- **Uplift is not always worth the machinery.** Schoder and Rößler (2025) find it pays most in
  contractual settings, high-cost offline campaigns, and frequent campaigns; it is often not worth
  it when the average effect is tiny, costs are near zero, or there are few customers. They also
  note that response modelling often targets sure things. (Grade A.)
- **Selection effects are the norm.** Pacella and colleagues (2026) report that in confounded
  observational regimes uplift models can underperform a response baseline, and that the
  highest-yield causal policy is often the least stable across runs. (Grade A.)

The published retail evidence is thinner than the vendor market implies, and most of it is
customer-level direct marketing rather than store-level trade promotion. Named cases are worth
grading explicitly: the 40% contact-volume reduction at 5× return attributed to U.S. Bank in a
Portrait Software case study is a vendor claim, widely repeated and not independently audited
(Grade E); the German fashion and eight-campaign benchmarks are peer-reviewed (Grade A). One
widely cited practitioner comparison reports the same model scoring 0.8873 under one library's
AUUC and 0.0212 under another's because of normalisation differences (Grade E; illustrative, not a
benchmark).

### 9.3 The Qini-versus-policy-value argument

The argument is live and worth understanding in both directions. Qini is cheap, needs no cost
model, and compares candidates on a common scale — but it is rank-invariant, while deployment is
not, so a well-ranked and badly calibrated model produces a bad threshold that Qini cannot detect.
Policy value evaluates the rule you will deploy, including threshold and budget, at the cost of
higher variance and dependence on the propensity model. UpliftBench (2026, preprint, Grade C)
reports 14–15% more policy regret when models are selected by Qini rather than policy risk under a
sign-threshold rule, with threshold calibration removing about 81% of the gap; under a budgeted
objective the gap largely disappears. The practical synthesis is to use Qini for model comparison
in development, policy value for the go/no-go and threshold decision, and both on the same held-out
set.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Response probability sold as uplift | The targeting list correlates almost perfectly with baseline sales or past spend | The model predicts the level, not the difference | Budget spent on sure things; sleeping dogs promoted | Ask for the estimand and for the control-group evaluation |
| No held-out treated and control set | The evaluation set is described as "last year's campaign data" and was used for feature selection | Metrics computed on data that informed the model | Qini inflated; deployment underperforms | Freeze a holdout before modelling |
| Qini computed on unbalanced arms without adjustment | Treated share under 20%; no mention of reweighting | The `N_t/N_c` ratio swings across the prefix | Curve noise read as model quality | Use an adjusted Qini or reweight; report arm sizes |
| Treatment definition drift | The promo flag changes meaning between years; depth is recomputed from a moving regular price | The model learns a mixture of treatments | CATE is an average of mechanisms with shifting weights | Publish a treatment codebook with version dates |
| Observational confounding | Treated stores were "chosen because they respond well" | Assignment uses local information absent from `X` | Model reproduces the current allocation rule | Randomise, or find a design with exogenous assignment |
| Post-treatment leakage | Features include display compliance, in-event traffic, or promo-week sell-through | The feature is a consequence of treatment | Effect estimate biased; accuracy looks excellent | Audit the feature list against treatment timing |
| Post-hoc segmentation | Segments named after the results are seen; no pre-registration | Selected extremes overstate effects | Confidence intervals invalid; targeting overtreats | Pre-specify segments; report all of them |
| Winner's curse in the top decile | Predicted uplift in the top decile is far above the realised uplift on holdout | Selection on noisy estimates | Overtreatment at the top of the list | Calibration table by decile; shrinkage |
| Threshold set at zero | "We promote where uplift is positive" | The cost of the discount and the fixed cost are omitted | Programme loses money despite positive lift | Compute `u* = c/m` and apply it |
| No permanent holdout | After two campaigns, no untreated stores remain in the data | The policy removed the counterfactual | The next model cannot be validated; drift is invisible | Reserve 5–10% permanently |
| Feedback loop | The treated share rises each cycle and the model's confidence rises with it | The model trains on data its own policy generated | Self-confirming model; decay is silent | Holdout, plus propensity logging of assignment |
| Metric-name ambiguity | Two teams report different Qini values for the same model | Different curve definitions and normalisations | Decisions made on incomparable numbers | Publish the axis definitions and the normalisation |
| Per-customer "ITE" reported as fact | A CRM export has a precise uplift score per customer with no interval | `CATE(x)` presented as an individual effect | False precision; regulatory and ethical exposure | Report intervals, or report segment means |

Three of these deserve a note.

**Response probability sold as uplift** is the most common and the hardest to see, because the
model's output looks the same in a dashboard. The diagnostic is the correlation between the
score and baseline volume. If it is above roughly 0.8, you are almost certainly looking at a
response model. A genuine uplift score is usually close to uncorrelated with baseline volume,
and sometimes negatively correlated, as in section 7 where the largest store had a negative
effect.

**No permanent holdout** is the failure with the longest lag. It causes no harm in the first
campaign and destroys the ability to learn by the third. The organisational fix is to make the
holdout a budget line item — a stated percentage of stores that sales cannot promote — rather
than a modelling choice that sales can override.

**Metric-name ambiguity** is not a statistical error at all. It is a governance error, and it is
fixed with a metric dictionary, the same instrument Topic 01 recommended for service metrics.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| Unconfoundedness given `X` | Identifies the CATE from observational data | The model estimates selection, not effect | Compare with a randomised subset; test a negative control outcome |
| Overlap / positivity | Every unit has a counterfactual in the data | Estimates at the extremes are extrapolation | Plot propensity by arm; count units outside common support |
| SUTVA, no interference | Each unit's outcome depends only on its own treatment | Effect inflated or deflated by cross-unit contamination | Map catchments; compare effects by store density |
| Consistency | The treatment version is the same in training and deployment | The estimand changes without the model changing | Version the treatment codebook; audit execution |
| No anticipation | Pre-treatment outcomes are unaffected by the coming promotion | Baseline is contaminated; effect understated | Check pre-event weeks for pantry loading |
| Treatment defined stably | The CATE is attached to a definition | A moving definition makes the model uninterpretable | Version the calendar; test the definition against audits |
| Outcome measured at the treatment unit | Standard errors and effects are at the right level | Intervals too narrow; over-targeting | Check the cluster structure; compare clustered and unclustered SEs |
| Covariates measured pre-treatment | Prevents leakage of the treatment's consequences | Spurious effect; inflated accuracy | Audit feature timestamps against event timestamps |
| Cost and margin known per unit | Converts an effect into a decision | Threshold is wrong; profitable units are excluded | Reconcile the cost model with finance |
| Holdout remains random and untreated | Keeps the counterfactual alive over time | Drift undetectable; next model unvalidatable | Audit holdout treatment status each cycle |
| Effect heterogeneity is learnable from `X` | Justifies a CATE model over a segment average | The forest fits noise; segments do not replicate | Test on a second period or a second market |
| Stationarity over the deployment horizon | The estimated CATE remains valid later | Model decays silently | Monitor effect drift on the holdout |

Four of these carry more weight than the rest.

**Unconfoundedness is the load-bearing assumption, and it is untestable.** Everything in Phase C
is machinery for estimating a function that only exists if this assumption holds. When a
stakeholder asks "how accurate is the model?", the honest first answer is "the accuracy is not
the binding constraint; the design is."

**Overlap is testable and frequently ignored.** In retail promotion data, the propensity to be
promoted is often near 1 for large stores in certain formats, because that is where the company
always promotes. Those stores have no counterfactual, and the model's estimate for them is
functional-form extrapolation. Trimming them changes the population and therefore the estimand,
which is why the decision must be made explicitly rather than by a default in the software.

**Consistency sounds trivial and is not.** If the 2024 promotion was a 15% cut with display and
the 2025 promotion is a 15% cut with feature only, they are different treatments with the same
label. A model trained across both estimates an average of two mechanisms, and the deployment
chooses one of them. Versioning the treatment definition is the fix, and it is a data-governance
task, not a modelling task.

**Stationarity is the assumption that deployment breaks.** The moment the policy changes, the
data-generating process for future training data changes. Section 5's feedback loop is the
mechanism; the permanent holdout is the only defence that survives contact with a sales team.

Two limits of the whole approach are worth stating plainly.

**Uplift cannot create identification.** If the promotion was assigned by people using
information you do not have, the best causal-forest implementation in the world will give you a
precise estimate of the wrong quantity. The method's value is proportional to the credibility of
the design, and no amount of tuning changes that ratio.

**Uplift is a decision tool, not an explanation.** A CATE surface tells you where to act. It does
not tell you why the effect is larger in one store than another, and the variable importances
from a causal forest are not a mechanistic explanation. If the business needs a mechanism —
"small stores respond because they have less shelf space and display is more salient" — that
hypothesis needs a separate test, and it may not be identifiable from the same data.

## 12. Exercises

### Exercise 1 — Estimand audit (critique, 30 minutes)

A model card reads: "Predicts each store's likelihood to respond to the spring promotion. AUC
0.81. The top decile is targeted."

**Deliverable:** a rewritten estimand as `E[Y(1) − Y(0) | X = x]`, the missing elements, and three
questions for the model owner.
**Acceptance criteria:** the rewrite names outcome unit and window; at least three missing
elements are identified; the questions are answerable from documentation.

### Exercise 2 — Response versus uplift ranking (calculation, 45 minutes)

| Segment | Customers | Treated | Control | Response, treated | Response, control |
|---|---:|---:|---:|---:|---:|
| P | 1,200 | 600 | 600 | 30% | 28% |
| Q | 300 | 150 | 150 | 9% | 3% |

Contact cost €1.50 per customer; margin per response €40.

**Deliverable:** uplift per segment, profit per contact under each ranking for a budget of 150
contacts, and one sentence on which ranking the campaign manager will prefer.
**Acceptance criteria:** both rankings computed; profit stated per contact and in total; the
sentence names the incentive that produces the wrong choice.

### Exercise 3 — Meta-learner under imbalance (design, 40 minutes)

A retailer has 90 treated stores, 9,000 control stores, 120 covariates, and a decision that needs
a store-level effect.

**Deliverable:** a choice among S-, T-, and X-learner with the bias and variance argument, plus one
alternative that avoids the learner question.
**Acceptance criteria:** the arm imbalance is named as the driver; the variance consequence is
stated in words; the alternative is a real design option, not "collect more data".

### Exercise 4 — Honest splitting review (critique, 40 minutes)

A colleague reports leaf-level intervals from a causal forest with `honesty = FALSE` and calls
them "conservative because we used 2,000 trees".

**Deliverable:** a one-page review of the inferential problem, the direction of the bias in the
point estimate and the variance, and the specific change you would make.
**Acceptance criteria:** distinguishes honesty from cross-validation; states that the point
estimate is biased away from zero and the variance too small; names a concrete setting change.

### Exercise 5 — Qini by hand (calculation, 50 minutes)

Use the eight-store table in section 7. Recompute the curve, the Qini coefficient, and the gain at
50% depth without looking at the answers.

**Deliverable:** the cumulative-gain column, the random line, the three areas, the Qini
coefficient, and the highest-response and highest-uplift stores.
**Acceptance criteria:** arithmetic reproduces 2,475, 1,640, 2,695, and 0.791; the answer names the
two stores and explains why 0.79 is not a realistic benchmark.

### Exercise 6 — Treatment definition memo (design, 60 minutes)

A grocery retailer wants to know "whether promotions work". Its calendar records a binary flag, a
depth percentage, and feature and display flags, but feature and display almost always run
together.

**Deliverable:** a one-page memo choosing one treatment definition, the design required, the data
needed, and the question the definition cannot answer.
**Acceptance criteria:** the definition is operational, with a threshold and a window; the design
names randomisation or identification; the limitation is specific to the definition chosen.

### Exercise 7 — Confounding and design (critique, 50 minutes)

You have 3,000 observational store-weeks with a binary promo flag and 40 covariates. Category
managers chose which stores to promote.

**Deliverable:** three named confounders with the selection mechanism for each, one design change
per confounder, and a statement of what the observational estimate measures.
**Acceptance criteria:** each confounder is a variable, not a slogan; each design change is
feasible in retail; the statement distinguishes "the effect of promotion" from "the effect of
being promoted".

### Exercise 8 — Deployment protocol (design, 60 minutes)

Using the worked example in section 6, write the protocol for the next cycle.

**Deliverable:** the targeting rule with a threshold in units and euros, the holdout design, four
monitors with trigger levels, and the withdrawal condition.
**Acceptance criteria:** the threshold uses the cost model and states whether it is applied to the
point estimate or a lower bound; the holdout is permanent and auditable; the withdrawal condition
is falsifiable.

## 13. Self-check answers

**Exercise 1.** The estimand is `E[Y(1) − Y(0) | X = x]` for a stated outcome, unit, and window. The
card describes `P(Y = 1 | X, W = 1)`, a level. Missing: treatment definition, control group,
conditioning set, cost model, evaluation metric. Questions: what was the control group and was
assignment random; what does "respond" mean in units; how was the top decile chosen and on what
data.

**Exercise 2.** P: 30% − 28% = +2 points. Q: 9% − 3% = +6 points. Profit per contact is
`uplift × €40 − €1.50`: P is `0.02 × 40 − 1.50 = −€0.70`; Q is `0.06 × 40 − 1.50 = +€0.90`. A
response ranking puts P first; 150 contacts there lose €105. An uplift ranking puts Q first; 150
contacts earn €135. The manager prefers P because it has the higher response rate and the larger
population, and because the dashboard reports response, not uplift.

**Exercise 3.** With 90 treated and 9,000 control, the control surface is estimated well and the
treated surface badly. The T-learner subtracts a noisy `mu1` from a precise `mu0`, so the effect
inherits the treated arm's variance, and independent fits can disagree for reasons unrelated to
the effect. The S-learner shares strength but may never split on treatment. The X-learner is
designed for this shape, imputing treated effects from the well-estimated control model, but its
stage-2 regression still sees 90 units and must be heavily regularised. The alternative is to
pre-specify three to five segment effects and estimate difference in means with intervals.

**Exercise 4.** Without honesty the split-selection and leaf-estimation samples are the same, so a
leaf is selected partly because its noise is favourable and then estimated on that same noise.
The point estimate is biased away from zero, so apparent heterogeneity exceeds real heterogeneity,
and the variance is understated, so intervals under-cover. More trees reduce Monte Carlo error,
not the bias. The fix is honest splitting with a subsample small enough to leave an estimation
sample and a leaf size large enough for that sample to be informative. Cross-validation is a
different procedure.

**Exercise 5.** Cumulative gains: 120, 210, 360, 380, 420, 400, 380, 410; random line `51.25k`.
Model area 2,475; random 1,640; perfect 2,695; Qini `(2,475 − 1,640)/(2,695 − 1,640) = 0.791`. Gain
at 50% is 380 against a random 205. S7 ranks first on baseline (900) and last on effect (−20); S3
ranks third on score and first on effect (+150). A Qini of 0.79 is implausible because the perfect
curve comes from the same eight units, the sample is tiny, and the normalisation is
dataset-specific.

**Exercise 6.** A defensible choice for "does the event pay" is binary: any temporary price
reduction of at least 5% with a feature or display, in a named week, against a randomised
no-promotion control, stratified by format. Data needed: the calendar with depth, the execution
audit, trade allowances, and margin. It cannot answer the depth question, because depth varies
within the treated group and the estimate is an average over depths.

**Exercise 7.** Store size and format (large stores are both promoted and high-volume);
category-manager expectation (managers promote where they expect response, using local knowledge
absent from the 40 covariates); competitive activity (a competitor's price cut both invites a
response and depresses sales). Design changes: stratify or match on size and format; randomise
within manager and region; measure competitor price, or use a design where competitor shocks are
plausibly exogenous. The observational estimate measures the difference between promoted and
unpromoted stores, which is the allocation rule plus its effect.

**Exercise 8.** Rule: promote store `s` if
`tau_hat(s) × €5.00 > €0.30 × (b(s) + tau_hat(s)) + €50`, using the lower bound of the interval;
at the average baseline this is `tau_hat > 36.3` units. Holdout: 7% of stores, randomly selected
within format, never promoted through any channel, audited quarterly. Monitors: feature drift
(population stability index above 0.2), score drift (mean predicted uplift moving more than one
standard error), effect drift (holdout ATE below 5 units for two quarters), calibration drift
(realised top-decile uplift below 70% of predicted for one quarter). Withdrawal: effect or
calibration drift sustained for two quarters.

## 14. Interview questions

**1. "Response model versus uplift model?"** A strong answer says the response model estimates
`P(Y | X, W = 1)`, a level, while uplift estimates `E[Y(1) − Y(0) | X]`, a difference, and notes
that response ranking approximates baseline ranking, so it targets sure things. A weak answer
calls uplift "a better response model".

**2. "Define CATE. Why not ITE?"** A strong answer writes the conditional mean, names the
conditioning set and unit, and explains that only one potential outcome is observed per unit, so
the individual difference is never identified; `CATE(x)` is estimable because it averages over
units sharing `x`. A weak answer says the model estimates individual effects.

**3. "Five percent treated, 95% control. Which learner?"** A strong answer names the imbalance as
the driver, explains the T-learner's noisy treated model and the S-learner's risk of ignoring
treatment, chooses the X-learner with a regularised stage 2, and asks whether per-unit estimation
is needed. A weak answer names a favourite library.

**4. "What does honest mean in a causal forest?"** A strong answer says split selection and leaf
estimation use disjoint subsamples, removing selection bias and making the variance valid, and
distinguishes honesty from cross-validation. A weak answer says "it doesn't overfit".

**5. "Why not evaluate uplift with AUC?"** A strong answer says AUC scores `P(Y | X, W)`, and a
model can predict both arms perfectly while the effect is constant; ranking effects depends on
differences, not levels. A weak answer says accuracy is "not the right metric" without saying why.

**6. "What is a Qini curve, and what are its limits?"** A strong answer gives cumulative gain, the
arm-ratio rescaling, the random baseline, and the normalisation, then names rank invariance,
inconsistent normalisation, and sensitivity to unbalanced arms. A weak answer describes a lift
chart.

**7. "When policy value over Qini?"** A strong answer says when deployment is a threshold rather
than a budget, because Qini is rank-invariant and cannot choose where to cut; policy value
evaluates the rule, at the cost of higher variance and dependence on the propensity model. A weak answer says policy value is "more business-focused".

**8. "Promote where uplift is positive — what is wrong?"** A strong answer says the threshold is
`c/m`, not zero, and probably a lower bound; zero ignores the discount on baseline units and the
fixed cost. A weak answer says the model needs more data.

**9. "How would you define the treatment?"** A strong answer distinguishes any promotion, depth
band, feature and display, and customer coupon, says each needs a different design, and insists
on a written rule plus an execution audit. A weak answer says "the promo flag".

**10. "Why is observational promotion uplift confounded?"** A strong answer names the selection
mechanism — managers promote where they expect response using unrecorded local knowledge — and
says the estimate reproduces the allocation rule, the same problem as in elasticity estimation. A weak answer says "correlation is not causation".

**11. "How do you keep a deployed model honest?"** A strong answer describes a permanent
randomised holdout, propensity logging, effect-drift and calibration monitoring on that holdout,
and a pre-registered withdrawal condition. A weak answer says "retrain quarterly".

## 15. Key terms

- **Treatment** — the intervention, defined by a written rule.
- **Potential outcomes** — `Y(1)` and `Y(0)`; one is observed per unit.
- **ATE / ATT** — average effect over a population / over the treated.
- **CATE** — conditional average treatment effect; the estimand of uplift.
- **ITE** — individual treatment effect; not identified.
- **Uplift** — marketing term for the incremental outcome caused by treatment.
- **Response probability** — `P(Y = 1 | X, W = 1)`; a level, not an effect.
- **Persuadable / sure thing / lost cause / sleeping dog** — the four response segments.
- **Confounder** — affects both assignment and outcome.
- **Unconfoundedness** — treatment independent of potential outcomes given covariates.
- **Overlap** — positive probability of either arm at every covariate value.
- **SUTVA** — no interference between units.
- **Propensity score** — probability of treatment given covariates.
- **Meta-learner** — reduces CATE estimation to supervised learning.
- **S- / T- / X-learner** — one model / one per arm / impute then combine.
- **Causal forest** — tree ensemble estimating a local treatment effect.
- **Honest splitting** — disjoint samples for split selection and leaf estimation.
- **Cross-fitting** — fitting nuisance models on one fold, scoring on another.
- **Qini curve / AUUC** — cumulative incremental outcome by depth / its area.
- **Policy value** — expected outcome under a decision rule.
- **Breakeven uplift** — the effect at which treatment profit is zero.
- **Holdout** — never-treated units that preserve the counterfactual.
- **Feedback loop** — the policy changes the data that trains the next model.

## 16. Sources

**Peer-reviewed and academic**

- Künzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). "Metalearners for Estimating
  Heterogeneous Treatment Effects using Machine Learning." *PNAS*, 116(10), 4156–4165.
  DOI: 10.1073/pnas.1804597116. Grade A.
- Wager, S. and Athey, S. (2018). "Estimation and Inference of Heterogeneous Treatment Effects
  using Random Forests." *JASA*, 113(523), 1228–1242. DOI: 10.1080/01621459.2017.1319839. Grade A.
- Athey, S., Tibshirani, J., and Wager, S. (2019). "Generalized Random Forests." *Annals of
  Statistics*, 47(2), 1148–1178. arXiv:1610.01271. Grade A.
- Gubela, R. M., Lessmann, S., and Stöcker, B. (2022). "Multiple Treatment Modeling for Target
  Marketing Campaigns: A Large-Scale Benchmark Study." *Information Systems Frontiers*.
  DOI: 10.1007/s10796-022-10283-4. Grade A.
- Baier, D. and Stöcker, B. (2021). "Profit uplift modeling for direct marketing campaigns."
  *Journal of Business Economics*. DOI: 10.1007/s11573-021-01068-3. Grade A.
- Schoder, D. and Rößler, J. (2025). "To Treat or Not To Treat? Five Lessons Learned from Using
  Uplift Modeling to Optimize Marketing Campaigns." *California Management Review*, November 2025. Grade A.
- Pacella, M. et al. (2026). "Stability-Aware Uplift Policy Selection for Customer Retention."
  *Applied Sciences*, 16, 4918. Grade A.
- Verbeken, B. et al. (2025). "Uplift Model Evaluation with Ordinal Dominance Graphs." *Journal of
  Machine Learning Research*, 26. Grade A.
- Devriendt, F., Guns, T., and Verbeke, W. (2020). "Learning to Rank for Uplift Modeling." Venue
  not reproduced in the retrieved results; verify before citation. Grade C.
- Osborne, M. (2018). "Frequency versus Depth." *Japanese Economic Review*, 69(3), 258–283. Grade A.

**Preprints and working papers (not peer-reviewed)**

- UpliftBench (2026). "Revealing Outcome-Regime and Objective Mismatch in Uplift Evaluation."
  arXiv:2608.00915. Grade C.
- Renaudin, C. and Martin, D. (2021). "About Evaluation Metrics for Contextual Uplift Modeling."
  arXiv:2107.00537. Grade C.
- Bokelmann, S. and Lessmann, S. (2024). Work on the variance of Qini-based evaluation,
  *European Journal of Operational Research*. Full title not reproduced in the retrieved results. Grade C.
- Gubela, R. M., Bequé, A., Gebert, K., and Lessmann, S. (2018). "Conversion uplift in
  e-commerce." IRTG 1792 Discussion Paper 2018-062. Grade C.
- Haupt, J. and Lessmann, S. "Targeting Customers under Response-Dependent Costs."
  arXiv:2003.06271. Grade C.
- Elberg, J., Gardete, P. M., Macera, R., and Noton, C. (2018). "Dynamic Effects of Price
  Promotions." Working paper. Grade C.
- Javadi, A. (2026). "Diminishing Returns to Promotion Depth in Grocery." Preprints.org.
  DOI: 10.20944/preprints202603.1966.v1. Grade C.
- Wager, S. (2014). "Asymptotic Theory for Random Forests." arXiv:1405.0352. Grade C.

**Standards and official**

- GS1 General Specifications, including GTIN, GLN, and EPCIS 2.0. Grade B.
- ANSI ASC X12 852, Product Activity Data. Grade B.
- Regulation (EU) 2016/679 (GDPR), including profiling and automated decision-making. Grade B.
- Regulation (EU) 2024/1689 (AI Act). Applicability requires legal review. Grade B.

**Industry, practitioner, and vendor**

- Radcliffe, N. (2007). "Using control groups to target on predicted lift." *Direct Marketing
  Analytics Journal*. Origin of Qini and the four-segment vocabulary. Grade C.
- Radcliffe, N. (2007). "The Fundamental Campaign Segmentation." *The Scientific Marketer*.
  https://www.scientificmarketer.com/2007/02/fundamental-campaign-segmentation.html. Grade E.
- Insightful Data Lab (2025). "AUUC: Which One Are You Quoting?"
  https://insightful-data-lab.com/2025/08/19/auuc-area-under-the-uplift-curve/. Grade E.
- CausalML documentation, https://causalml.readthedocs.io/en/stable/methodology.html. Grade D.
- EconML documentation, https://www.pywhy.org/EconML/spec/estimation/metalearners.html. Grade D.
- Yi, R. and Frost, W. (2020). pylift white paper, https://ryi.me/papers/uplift_eval.pdf.
  Grade D.
- Portrait Software case study on U.S. Bank, as reported by *Chief Marketer*. Vendor claim, not
  independently audited. Grade E.

> **Note on citations.** Sources are reported as the searches returned them. Where a full title,
> venue, or DOI was not returned, the entry says so rather than supplying one. Preprints are
> graded C regardless of quality.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Earlier in this module: [`04_baseline-incrementality.md`](04_baseline-incrementality.md)
- Earlier in this module: [`05_cannibalization-halo.md`](05_cannibalization-halo.md)
- Promotion mechanics and the treatment definition: [`03_promotion-mechanics.md`](03_promotion-mechanics.md)
- Signals and their lead times: [`01_demand-sensing-signals.md`](01_demand-sensing-signals.md)
- Decomposition into baseline and event: [`02_sales-decomposition.md`](02_sales-decomposition.md)
- Experimental design, power, and interference: [`../../01_foundations/05_experimentation_pilots_evidence/_index.md`](../../01_foundations/05_experimentation_pilots_evidence/_index.md)
- Spillover between units: [`../../01_foundations/05_experimentation_pilots_evidence/07_interference-spillover.md`](../../01_foundations/05_experimentation_pilots_evidence/07_interference-spillover.md)
- Scoring rules for probabilistic forecasts: [`../08_hierarchical_intermittent_probabilistic_forecasting/09_scoring-rules.md`](../08_hierarchical_intermittent_probabilistic_forecasting/09_scoring-rules.md)
- Module survey: [`../../../course/09_demand_sensing_promotions_npi.md`](../../../course/09_demand_sensing_promotions_npi.md)
- Part map: [`../_index.md`](../_index.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Next topic: `08_npi-forecasting.md`
- Later topic: [`09_distribution-assortment.md`](09_distribution-assortment.md)
- Final topic: [`10_post-launch-learning.md`](10_post-launch-learning.md)
- Price and display effects: `06_price-display-effects.md`
