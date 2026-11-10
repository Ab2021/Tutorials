---
collection: "08-experimentation-causal-inference-and-decision-science"
file_id: "03"
topic: "Matching and Difference-in-Differences"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Panel construction rule, comparison-group definition, outcome measurement, treatment-timing pattern, or inference method changes"
notation_style: "plain-text"
prerequisites: ["01_Experiment_Design_and_Guardrails.md", "02_Causal_Diagrams_and_Adjustment.md", "linear regression with categorical predictors", "expectation and variance of a difference in means"]
pattern_tags: ["experimentation", "causal-inference", "quasi-experiment"]
source_status: "OPEN"
owner_role: "applied scientist, commercial or policy analytics"
source_index: "INDEX.md"
---

# Matching and Difference-in-Differences

## 1. File Header

> **Purpose:** Build a defensible counterfactual for an intervention that was *not* randomised. Two tools do most of that work: matching, which finds comparable untreated units for treated units, and difference-in-differences, which removes any fixed difference between the two groups by comparing *changes* rather than levels. This sheet teaches when each tool identifies the effect you want, how to compute it, how the two combine, and the specific ways both produce confident nonsense.
> **Prerequisites:** randomisation logic and the estimand vocabulary of file 01; the backdoor criterion and the adjustment-set discipline of file 02; linear regression with dummy variables and interaction terms; the variance of a difference in means.
> **Key outcome:** Given a treated group, a candidate comparison group, a pre-period, and a post-period, you can (a) state the estimand and write down exactly which assumption identifies it, (b) compute the 2x2 estimate by hand and show the arithmetic, (c) decide whether matching, difference-in-differences, both, or neither is defensible for the panel you actually have, (d) run the event-study diagnostic and say what it can and cannot rule out, (e) size the design's minimum detectable effect, and (f) reject an analysis that has conditioned on a post-treatment variable, ignored clustering, or reported a two-way fixed-effects coefficient under staggered adoption without checking what it averages.
> **This file does NOT cover:** randomised experiment design and power for a planned test (file 01); constructing and defending a causal graph (file 02); regression discontinuity, instrumental variables, and synthetic control beyond their rows in the selection table; heterogeneous-effect estimation and targeting (file 04); decision thresholds for a single score (file 05).
> **Level:** 2 Core Methods
> **Version / reviewed:** 1.0 / not independently reviewed. `reviewed` stays null until a substantive review happens.
> **Evidence:** Identification arguments are DERIVED from the stated assumptions. Estimator definitions are attributed to published results and labelled SOURCED where a specific method is named; bibliographic details are recorded as OPEN pending verification against the publisher record. Every number in Sections 5 and 9 is ILLUSTRATIVE teaching arithmetic, recomputed by hand from the stated inputs; none of it is a company measurement, and no real organisation is described.
> **Retrieval cue:** If the question begins "we rolled this out in some regions", "can we just compare the stores that got it to the ones that didn't", "the pre-trend looks fine to me", "our DiD is significant but the reviewer says it is wrong", or "the coefficient changed sign when we added region fixed effects", start at Sections 4.4, 8, and 5 in that order.

### 1.1 The decision this sheet supports

A retail operations team changed the replenishment cadence in 40 of its 120 stores in July. Leadership wants to know whether the change raised revenue per store, and whether to extend it to the remaining 80 stores in the next quarter. There was no randomisation: the 40 stores were chosen by a regional director who says she picked "the ones that were ready". The naive comparison after the change shows the treated stores running 26.5 thousand currency units per month above the untreated ones, which is exactly the number that will appear in the slide if nobody asks what the gap was before July.

That gap was 14.0 thousand before the change. The change in the gap is 12.5. Whether 12.5 is the effect of the replenishment change, or the continuation of a pre-existing divergence, or an artefact of the stores that closed, or a number with a standard error larger than itself, is the entire content of this sheet. Matching and difference-in-differences are two answers to the same sub-problem — constructing the counterfactual "what would these stores have done without the change" — and they answer it with different assumptions that fail in different ways.

| Layer | Question it answers | What goes wrong if you skip it | Where treated |
|---|---|---|---|
| Estimand | Effect on whom, over what window, in what units | You estimate the wrong quantity very precisely | 3.1, 4.1 |
| Identification | What must be true for the counterfactual to be recoverable | The number describes selection, not effect | 3.2, 4.4 |
| Construction | How the comparison group is formed | The comparison differs in ways the adjustment cannot fix | 4.3, 8.5 |
| Estimation | Which arithmetic turns the panel into one number | Arithmetic is right, target is wrong | 4.2, 5 |
| Inference | How uncertain that number is | A 1.3 t-statistic is reported as a 4.5 | 4.1.4, 8.6 |
| Decision | What the number changes | Analysis becomes a report rather than a decision | 1.2, 12 |

### 1.2 The four questions a quasi-experiment must answer before any arithmetic

First: **what is the intervention, precisely?** "The replenishment change" must resolve to a date, a set of stores, and a rule that determines membership. If two stores got different versions of the change on different dates, the treatment is a mixture and the estimand is an average over an unrecorded dose distribution.

Second: **what is the counterfactual, and why is it credible?** Difference-in-differences claims that the treated stores would have followed the same *trend* as the comparison stores absent the change. Matching claims that treated and comparison units with the same covariates would have had the same *outcome*. These are different claims, they are falsifiable in different ways, and neither is implied by the other.

Third: **what is the unit and the window?** Store-months in this example. If the outcome is measured over 30 days and the intervention is a daily cadence change, the window must be long enough for the operational change to work through inventory, and short enough that the composition of the comparison group has not changed underneath it.

Fourth: **what would you do differently if the answer were zero?** An analysis whose conclusion cannot change an action is a description. This is not a rhetorical point: it determines whether you need a point estimate with a confidence interval, or a decision rule with a threshold, and those require different amounts of precision.

### 1.3 What this sheet is not

It is not an argument that quasi-experiments are as good as experiments. They are not; they are what you use when randomisation was impossible, refused, or already missed. It is not a catalogue of estimators to try until one produces a significant result — that practice is the subject of failure F-03 in Section 8. And it is not a defence of "we controlled for everything we had". Adjustment sets are justified by a causal graph, not by the width of the data warehouse.

**Takeaway:** the deliverable is a counterfactual with a named assumption, a checked estimate, and an honest uncertainty statement — not a coefficient.

## 2. Concept Map / Mental Model

### 2.1 The pipeline, with the gate each stage must pass

```text
  [intervention: who, when, how much, recorded how]
                 |
                 v
  [estimand: ATT / ATE / cohort-time ATT, units, window]
                 |
                 v
  [identification gate]
      +-- experiment?               -> difference in means, done
      +-- panel + parallel trends?  -> DiD identified
      +-- cross-section + ignorability + overlap? -> matching identified
      +-- neither?                  -> refuse, or find a design (IV, RDD, SC)
                 |
              passes
                 v
  [construct comparison: matched set, or trend donor pool]
                 |
                 v
  [estimate: 2x2 / event study / group-time ATTs / matched DiD]
                 |
                 v
  [diagnostics: pre-trends, balance, overlap, composition, placebo]
                 |
        +--------+---------+
        |                  |
     passes             fails --> [shrink claim, change design,
        |                           or report "not identified"]
        v
  [inference: clustered SE, few-cluster correction, sensitivity bounds]
                 |
                 v
  [decision: extend, hold, or revert]
```

The two gates in the middle are the ones that get skipped. A team that runs a difference-in-differences regression without ever looking at the pre-period trend has not passed the identification gate; it has walked around it.

### 2.2 The 2x2 as a picture of the identifying assumption

```text
 outcome
   |                                        * treated, post  (447.5)
   |                              .-------'
   |                    .--------'      <-- observed treated path
   |          .--------'
   |   * treated, pre (412.0)
   |          \
   |           \  ......................* counterfactual if parallel
   |            \.                      (435.0 = 412.0 + 23.0)
   |             '.-----------* control, post (421.0)
   |   * control, pre (398.0)
   |
   +--------------------------------------------------> time
        pre window            |            post window
                         intervention
```

The dashed continuation is not observed. Its slope is borrowed from the control group. If the treated group would in fact have grown faster — because it was chosen for being on an upward trajectory — the dashed line is too flat and the estimate is too large. Everything that follows is a way of testing, weakening, or pricing that borrowing.

### 2.3 Matching as a different borrowing

Matching does not borrow a slope. It borrows *levels* from units that look alike on covariates. For a treated store with a given size, region, and pre-period revenue, matching finds an untreated store with the same profile and uses its outcome as the counterfactual. The identifying claim is that, conditional on those covariates, the untreated store's outcome is what the treated store would have shown.

```text
 treated unit i ---> [covariate profile X_i] ---> nearest untreated j with X_j ~ X_i
                                                        |
                                                        v
                                        counterfactual for i = Y_j
```

Because matching on levels is fragile whenever the two groups differ in unobserved persistent ways, the strongest practical combination is matched difference-in-differences: match on pre-period characteristics, then compare *changes*. The matching step fixes the composition of the comparison group; the differencing step removes any fixed unobserved difference between the matched units. Section 4.3.5 derives that combination.

### 2.4 The three quantities that get conflated

| Object | Definition | Observed? | Typical reporting error |
|---|---|---|---|
| `ATT` | Mean effect on the units actually treated | No, estimated | Reported as if it were the ATE for a universal rollout |
| `ATE` | Mean effect on the whole eligible population | No, estimated | Claimed from an ATT computed on a selected group |
| Pre/post gap | Difference in levels at a point in time | Yes | Presented as the effect, ignoring the pre-existing gap |

The third row is the single most common error in operational reporting, and it is not a statistical subtlety — it is a failure to compute a second difference. The first two rows matter for the rollout decision: an ATT of 12.5 estimated on stores that volunteered is not the effect you would get by forcing the change on the stores that did not.

### 2.5 Why the two methods are often used together

Matching alone is weak in panel settings because it cannot handle unobserved, time-invariant differences between units that also drive selection. Difference-in-differences alone is weak when the two groups are so different in composition that their trends would plausibly diverge even absent treatment. Combining them — match to build a comparable panel, then difference within the matched panel — addresses both, at the cost of a smaller sample and a new sensitivity to the matching specification. That is why the sheet treats them as one method with two stages rather than as alternatives.

**Takeaway:** matching fixes *who* you compare, difference-in-differences fixes *what you compare about them*; neither fixes the other's assumption.

## 3. Foundational Definitions

### 3.1 Units, potential outcomes, and estimands

**1. Unit and decision grain.** The unit `i` is the entity to which the intervention applies and about which the decision is made. Here it is a store. The grain must match the intervention: a cadence change applied at store level cannot be evaluated with order-level rows, because the treatment varies only across stores and order rows inside one store are not independent observations of it.

**2. Binary treatment `W_it`.** `W_it = 1` when unit `i` is exposed at time `t`. The definition must include *what counts as exposure*. A store that received the new cadence in the warehouse system but whose staff ignored it is treated under an assignment definition and untreated under a compliance definition. The two give different estimands (intention-to-treat versus effect of treatment), and mixing them silently is failure F-10.

**3. Potential outcomes `Y_it(1)`, `Y_it(0)`.** The outcome unit `i` would show at `t` with and without exposure, over the declared measurement window. They are indexed by both unit and time because the trend matters: the counterfactual for the post period is not the pre-period value of the same unit, it is what that unit would have done in the post period absent treatment.

**4. Consistency.** `Y_it = W_it Y_it(1) + (1 - W_it) Y_it(0)`. This fails when one unit's exposure affects another unit's outcome (spillover between stores in the same catchment) or when several distinct interventions share one treatment label.

**5. Average treatment effect on the treated (ATT).** `ATT = E[Y(1) - Y(0) | W = 1]`, the mean effect on the units that were actually exposed. This is the estimand of the 2x2 design, of the event study, and of nearest-neighbour matching. It is *not* the effect of extending the intervention to units that were not exposed unless effects are homogeneous or the rollout rule is independent of effect.

**6. Average treatment effect (ATE).** `E[Y(1) - Y(0)]` over a declared population. Recovering the ATE from an ATT requires either random assignment or an assumption about how effects differ between treated and untreated units — an assumption that is usually untested and usually wrong in operational settings, because units select into treatment precisely because they expect to benefit.

**7. Cohort-time ATT.** `ATT(g, t)`: the mean effect at time `t` for the cohort first treated at time `g`. This is the primitive that modern staggered-adoption estimators recover, and the object a two-way fixed-effects coefficient fails to be a weighted average of when effects are heterogeneous.

**8. Counterfactual trend.** The change the treated group would have shown absent treatment. It is never observed; it is *assumed* under parallel trends, *constructed* by matching, or *bounded* by a sensitivity analysis. Distinguishing "estimated" from "assumed" in the write-up is a matter of professional honesty, not style.

### 3.2 Identification assumptions

**9. Parallel trends.** Absent treatment, the treated and comparison groups would have followed the same average change in the outcome between the pre and post windows. The assumption is about *changes*, not levels: the groups may differ permanently in level, and that difference is exactly what the differencing removes. It is not testable in the post period, because the counterfactual is missing there by construction.

**10. No anticipation.** Units do not change behaviour before the recorded treatment date in response to the coming intervention. A store that begins restocking in June in preparation for a July change violates this, and the violation biases the estimate toward zero because part of the effect is absorbed into the pre-period.

**11. SUTVA / no interference.** One unit's outcome does not depend on another unit's treatment status. Two stores in the same catchment competing for the same demand violate this in a direction that depends on whether the intervention expands or contracts total demand.

**12. Ignorability (unconfoundedness) for matching.** `{Y(0)}` is independent of `W` given covariates `X`. This is the matching analogue of parallel trends and is equally untestable. It fails whenever a variable that drives both selection and the outcome is missing from `X`.

**13. Overlap / common support.** For every covariate value in the treated group, `0 < P(W = 1 | X = x) < 1`, so that a comparable untreated unit exists. When overlap fails, the match is an extrapolation dressed as a comparison, and its error is unbounded rather than merely large.

**14. Propensity score `e(x)`.** `e(x) = P(W = 1 | X = x)`. Under ignorability and overlap, conditioning on the scalar `e(x)` is sufficient to remove confounding by `X` — this is the Rosenbaum–Rubin result of Section 10. It is a dimension-reduction device, not a modelling convenience, and it only works if the propensity model is well specified in the sense that matters: it must balance the covariates, not predict treatment well.

**15. Balance.** The property that the matched treated and control groups have similar distributions of every covariate that enters the outcome. Balance is checked and reported; a good area under the propensity-model ROC curve is evidence about prediction, not about balance, and a high AUC can coexist with terrible balance.

### 3.3 Matching construction terms

**16. Caliper.** A maximum permitted distance between propensity scores (or covariates) for a match. A common rule of thumb is 0.2 standard deviations of the logit of the propensity score, which is a heuristic (HEURISTIC), not a theorem. A caliper trades bias for sample: too wide admits bad matches, too narrow discards treated units and changes the estimand to the effect on the matched subset.

**17. Matching with or without replacement.** With replacement, a control unit can serve several treated units; this improves match quality and requires frequency weights in the outcome model. Without replacement, each control is used once; this keeps the comparison group interpretable but degrades match quality as the pool is consumed.

**18. Exact and coarsened matching.** Exact matching requires equality on a covariate (region, store format). Coarsened matching bins continuous covariates into substantively meaningful categories and matches exactly on the bins. Both are preferable to distance matching when a covariate has a small number of operationally meaningful levels, because they make the comparison group auditable by eye.

**19. Common support region.** The intersection of the covariate (or propensity) ranges where both groups have mass. Treated units outside it are either dropped — changing the estimand to the effect on the supported subset — or retained with an extrapolated counterfactual, which must then be defended.

**20. Weighting as matching.** Inverse-probability weighting, kernel weighting, and entropy balancing are all forms of matching in which each control receives a weight instead of a binary indicator. They use more of the data than pair matching and are often more stable, at the cost of sensitivity to extreme weights.

### 3.4 Difference-in-differences terms

**21. Pre and post windows.** The periods before and after the intervention date used to form the two differences. Their lengths matter: a six-month pre window and a one-month post window compare a stable average to a noisy one, and seasonal composition changes between the two can masquerade as an effect.

**22. Event time.** `k = t - g`, the number of periods relative to a unit's treatment date. Coefficients indexed by event time form the event-study plot that Section 6 draws. `k = -1` is normally omitted as the reference period.

**23. Two-way fixed effects (TWFE).** The regression `Y_it = a_i + b_t + d W_it + e_it`, with unit and time fixed effects. It is the workhorse estimator and it estimates the ATT only under homogeneous effects or a balanced, non-staggered design. Under staggered adoption with heterogeneous effects it estimates a variance-weighted average of 2x2 comparisons, some of which use already-treated units as controls.

**24. Negative weighting.** The property that in the TWFE decomposition some 2x2 comparisons enter with a sign opposite to the target ATT, so the coefficient can lie outside the range of the cohort-specific effects, and can even have the opposite sign. Section 4.2.3 shows the mechanism arithmetically.

**25. Placebo test.** Applying the estimator to a period or a group where no effect is possible — pre-period "treatment dates", or outcomes the intervention cannot plausibly affect. A significant placebo falsifies the design; an insignificant one is weak evidence in its favour because the test has limited power (Section 10).

**26. Doubly robust DiD.** An estimator that combines an outcome-regression model with a propensity model and is consistent if either is correctly specified. It is the practical answer to "we are not sure which model is right", and it is not a licence to skip the parallel-trends check.


### 3.5 Symbol table for the rest of the sheet

The same letters are reused across estimators in the literature, so the column that matters is the domain and the indexing convention, not the letter. An `e` that means a propensity in one paragraph and a residual in the next is the fastest way to lose a reader.

| Symbol | Meaning | Units / type | Domain and index convention |
|---|---|---|---|
| `i`, `t` | unit, period | integer indices | `i = 1..n`; `t = 1..T`; `t` is calendar time |
| `g_i` | treatment date of unit `i` | period index | undefined (`inf`) for never-treated units |
| `k` | event time | integer periods | `k = t - g_i`; `k = -1` is the reference period |
| `W_it` | treatment indicator | 0/1 | equals 1 only when delivered, not when assigned |
| `Y_it` | observed outcome | outcome units | one definition across all cells |
| `Y_it(1)`, `Y_it(0)` | potential outcomes | outcome units | exactly one is observed per unit-period |
| `dY_i` | within-unit change | outcome units | post-window mean minus pre-window mean |
| `T_i`, `C` | treated group, comparison group | set / indicator | `T_i = 1` for exposed units |
| `a_i`, `b_t` | unit and period fixed effects | outcome units | absorb levels, not trends |
| `X_i` | pre-treatment covariates | vector, `P` dimensions | measured strictly before `g_i` |
| `e(X_i)` | propensity score | probability | `(0,1)`; equals the allocation ratio under randomisation |
| `m_0(X)` | untreated outcome model | outcome units | fitted on untreated units only |
| `s_g` | SD of the change score | outcome units | across units within group `g` |
| `v_g` | variance of the row-level outcome | outcome units squared | across rows within group `g` |
| `rho` | intraclass correlation | dimensionless | residual correlation within unit |
| `m` | rows per unit | integer | 12 in a monthly panel |
| `DE` | design effect | dimensionless, `>= 1` | `1 + (m-1) rho` |
| `n_g`, `N_g` | units, rows in group `g` | integer | the pair that must not be confused |
| `ATT(g,t)` | cohort-time effect | outcome units | defined for `t >= g` only |
| `alpha`, `beta` | test level, type-II rate | probability | two-sided unless stated |
| `tau*` | break-even effect | outcome units | set by cost and margin, not by the data |
| `delta` | proportional-selection parameter | ratio | `1` means equal selection on observed and unobserved |

The traps encoded in the table are worth naming explicitly. `n_g` versus `N_g` is the clustering error of F-06, and mixing them produces a standard error that is too small by a factor of the square root of the rows-per-unit count. `Y_it` versus `dY_i` is the difference between a level design and a change design, which changes both the estimand and the assumption. `g_i` versus `t` is the difference between a cohort index and a calendar index, and confusing them is how a staggered design gets analysed as if it were a single-date design. `tau` versus `tau*` is the difference between a quantity estimated from data and a quantity set by the business, and conflating them lets the data choose the decision threshold — the subject of file 05.

**Takeaway:** every estimator in this file is an implication of one of four assumptions — parallel trends, ignorability, overlap, no anticipation — and naming the assumption is what makes the number interpretable.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: the 2x2 design

#### 4.1.1 The identifying argument

Validity before use: two groups, two periods, a single treatment date inside the post window, treatment recorded as delivered, outcomes measured identically in both groups and both periods, and no other intervention coinciding with the treatment date. If the treated group is not observed in both periods, the design is not a 2x2 and none of the following applies.

Derivation. Write the observed mean in each cell using consistency and the potential-outcome notation. For the treated group in the post period, everyone is exposed, so the observed mean is `E[Y(1) | T = 1]`. For the treated group in the pre period, nobody is exposed, so the observed mean is `E[Y(0) | T = 1]`. Their difference is the observed change in the treated group:

```text
E[Y | T=1, Post] - E[Y | T=1, Pre]
    = E[Y(1) - Y(0) | T=1] + { E[Y(0) | T=1, Post] - E[Y(0) | T=1, Pre] }
    = ATT + [counterfactual change for the treated group]
```

The first term is the object of interest. The second is unobservable: it is what the treated stores would have done without the change. Parallel trends replaces it with the observed change in the comparison group, `E[Y(0) | T=0, Post] - E[Y(0) | T=0, Pre]`. Subtracting gives the estimator, in which the unobservable term cancels under the assumption.

```text
FORMULA [DID-01]: DiD = (Ybar_T,post - Ybar_T,pre) - (Ybar_C,post - Ybar_C,pre)
WHERE:  Ybar_g,p = mean outcome in group g in period-window p, in outcome units
        (here: thousand currency units per store-month);
        T = treated group, C = comparison group;
        pre and post are windows of declared, possibly unequal, length.
CONSTRAINT: parallel trends for the untreated potential outcome; common treatment
        date inside the post window; identical outcome measurement in all four
        cells; no coincident shock hitting one group only; no anticipation before
        the recorded date.
NUMERICAL EXAMPLE: treated 412.0 -> 447.5, change +35.5;
        control 398.0 -> 421.0, change +23.0;
        DiD = 35.5 - 23.0 = +12.5 thousand currency units per store-month.
INTERPRETATION: the point estimate says the change added about 12.5 to the
        monthly revenue of a treated store relative to its own pre-period level.
```

Note what the estimator does *not* do. It does not remove a permanent difference in level between the groups — that difference cancels, which is why the pre-period gap of 14.0 is irrelevant to the estimate. It does remove any *common* shock, because the shock appears in both changes. It does not remove a differential shock, and no amount of arithmetic can detect one from these four numbers alone.

#### 4.1.2 Equivalence to the interaction regression

Validity before use: the same four-cell structure, with the regression specified without additional covariates that vary within group-by-period.

Derivation. Regress the outcome on a treatment-group indicator, a post indicator, and their product. The four fitted cell means reproduce the four sample means exactly when the design is saturated, which it is with two binary regressors. The coefficient on the product is therefore the difference between the treated change and the control change.

```text
FORMULA [DID-02]: Y_it = a + b*T_i + c*Post_t + d*(T_i * Post_t) + e_it
WHERE:  T_i in {0,1} group indicator; Post_t in {0,1} period indicator;
        d is the DiD coefficient; e_it the residual.
        With only group and period indicators, d = DiD from [DID-01].
CONSTRAINT: saturated specification (no additional within-cell covariates);
        homoskedastic errors are NOT required for d to be unbiased, only for
        the textbook standard error to be right.
NUMERICAL EXAMPLE: a = 398.0, b = 14.0, c = 23.0, d = 12.5;
        fitted treated-post = 398.0 + 14.0 + 23.0 + 12.5 = 447.5 (matches the
        observed cell mean); fitted control-pre = 398.0; fitted control-post =
        398.0 + 23.0 = 421.0; fitted treated-pre = 398.0 + 14.0 = 412.0.
INTERPRETATION: the regression framing matters because it generalises to
        covariates, multiple periods, and clustered errors; the coefficient
        does not become more credible by being produced by software.
```

The four fitted values reproducing the four observed cell means is the check that the specification is saturated. If a regression reproduces the cell means but the `d` coefficient differs from the hand-computed DiD, the sample was filtered differently in the two computations — usually by listwise deletion on a covariate.

#### 4.1.3 The gap-change form

Validity before use: same as [DID-01]; useful when the pre-period gap is the number a stakeholder has already seen.

Derivation. Rearranging [DID-01] gives `DiD = (Ybar_T,post - Ybar_C,post) - (Ybar_T,pre - Ybar_C,pre)`, the change in the cross-group gap. This is algebraically identical and is the form to use when someone presents the post-period gap as the effect, because it makes the missing term visible.

```text
FORMULA [DID-03]: DiD = Gap_post - Gap_pre,  Gap_p = Ybar_T,p - Ybar_C,p
WHERE:  Gap_p in outcome units; positive means the treated group is higher.
CONSTRAINT: identical to [DID-01]; the identity holds exactly, so any
        numerical disagreement between the two forms is an arithmetic error,
        not an estimation choice.
NUMERICAL EXAMPLE: Gap_pre = 412.0 - 398.0 = 14.0;
        Gap_post = 447.5 - 421.0 = 26.5; DiD = 26.5 - 14.0 = 12.5.
INTERPRETATION: the headline "treated stores are 26.5 ahead" overstates the
        effect by exactly the pre-existing gap of 14.0, which grew from
        selection rather than from the intervention.
```

#### 4.1.4 Sampling variance and the clustering correction

Validity before use: the unit of randomisation (or of treatment assignment) is the unit of clustering; outcomes within a unit over time are correlated. Treating store-months as independent observations when treatment varies at store level understates the standard error, often by a factor of two or more.

Derivation. Collapsed to one change per unit, the estimator is a difference of two independent sample means of the *change* score, so its variance is the sum of the two group variances of that score divided by their counts. That version needs no clustering correction, because each unit contributes exactly one number. When the same estimator is computed on the stacked panel instead, each unit contributes `m` rows whose residuals are correlated; the variance of the mean over `n` units and `m` rows carries a design effect relative to the naive row count.

```text
FORMULA [DID-04a]: SE_unit = sqrt( s_T^2 / n_T + s_C^2 / n_C )
WHERE:  s_g^2 = sample variance of the per-unit CHANGE score in group g
        (outcome units squared); n_g = number of units in group g.
CONSTRAINT: independent units; change scores approximately normal or n_g
        large enough for the normal approximation; s_g computed on the same
        units that enter the means, after the declared exclusions.

FORMULA [DID-04b]: SE_rows = sqrt( DE * ( v_T / N_T + v_C / N_C ) ),
                   DE = 1 + (m - 1) * rho
WHERE:  v_g = variance of the row-level outcome in group g;
        N_g = number of ROWS (stores x months) in group g;
        m = rows per unit; rho = intraclass correlation of rows within unit;
        DE = design effect (dimensionless, DE >= 1, DE = 1 iff rho = 0).
CONSTRAINT: rho must be estimated from residuals of the specification actually
        fitted; when unit fixed effects are included, rho applies to
        deviations from the unit mean, not to levels.
NUMERICAL EXAMPLE (a): s_T = 48.0, n_T = 40, s_C = 52.0, n_C = 80;
        s_T^2/n_T = 2304/40 = 57.60; s_C^2/n_C = 2704/80 = 33.80;
        sum = 91.40; SE_unit = sqrt(91.40) = 9.5603.
        t = 12.5 / 9.5603 = 1.307.
NUMERICAL EXAMPLE (b): different dataset, row-level: v_T = 26.0^2 = 676,
        v_C = 28.0^2 = 784, m = 12, rho = 0.5 -> DE = 1 + 11*0.5 = 6.5;
        N_T = 480, N_C = 960;
        naive (rho = 0): 676/480 + 784/960 = 1.40833 + 0.81667 = 2.22500,
        SE_naive = 1.4916;
        corrected: 6.5 * 2.22500 = 14.46250, SE_rows = 3.8027.
INTERPRETATION: in (a) the honest standard error makes a 12.5 unit effect a
        1.31 t-statistic, i.e. indistinguishable from zero. In (b) ignoring
        clustering understates the standard error by a factor of
        sqrt(6.5) = 2.55. Neither number is a technicality: they decide the
        recommendation.
```

Three checks are worth running on any variance claim of this kind. First, dimensions: the variance must be in squared outcome units, so the standard error is in outcome units — a standard error of 9.56 is interpretable beside a mean of 412.0, and one of 0.0009 is not. Second, consistency across routes: the two forms [DID-04a] and [DID-04b] describe different analyses (a change score versus a stacked level contrast) and need not agree numerically; they must agree if you derive the change score from the same row-level components, which gives the identity `s_change^2 = 2 v (1 - rho)` for a two-period change. Third, boundary: `DE = 1` when `rho = 0`, so the naive row-level standard error is the limiting case rather than a competing formula. Do not mix components from different routes. Using a store-level change standard deviation of 48.0 inside a row-count formula that divides by 480 is the common error, and it produces a standard error that is too small by the square root of 12 before any clustering correction is applied at all.

Two further cautions apply to the practical use of [DID-04b]. With few clusters — say 12 regions treated as the clustering unit rather than 40 stores — the clustered standard error is itself downward-biased, and a small-sample correction or a wild cluster bootstrap is required; the normal approximation in [DID-05] then also needs re-examination. And when the treatment is assigned at a level above the unit of analysis (a regional rollout measured on store rows), clustering must follow the assignment level, not the measurement level, because that is the level at which the treatment varies.

#### 4.1.5 Minimum detectable effect and sizing

Validity before use: two-sided test at level `alpha`, power `1 - beta`, normal approximation, and a variance that does not depend on the effect size. Under-powered designs produce estimates that are unbiased and useless, and the failure is worth quantifying before the analysis is run, not after.

Derivation. The test rejects when `|DiD| / SE > z_{1-alpha/2}`. The probability of exceeding that threshold when the true effect is `delta` is approximately `Phi(delta/SE - z_{1-alpha/2})`. Setting that equal to the power and solving gives the smallest `delta` detectable with the stated probability.

```text
FORMULA [DID-05]: MDE = (z_{1-alpha/2} + z_{1-beta}) * SE
WHERE:  MDE in outcome units; z values from the standard normal
        (1.9600 at alpha = 0.05 two-sided, 0.8416 at power 0.80);
        SE from [DID-04a] or [DID-04b].
CONSTRAINT: normal approximation; SE known or well estimated; single
        pre-specified comparison. Multiplicity across many outcomes or
        subgroups raises the effective threshold.
NUMERICAL EXAMPLE: SE = 9.5603; MDE = (1.9600 + 0.8416) * 9.5603
        = 2.8016 * 9.5603 = 26.79 thousand currency units.
        As a share of the treated pre-period mean: 26.79 / 412.0 = 6.50%.
INTERPRETATION: this design cannot distinguish a zero effect from a 26-unit
        effect at conventional thresholds, while the estimate itself is 12.5.
        The study is underpowered for the effect it was built to find, which
        is a design fact, not a result.
```

Sizing the fix follows from the same expression. With each group's change standard deviation at 48.0 and 52.0 the equal-`n` pooled variance is `(2304 + 2704)/2 = 2504`, pooled SD `50.04`. A balanced design needs `n` per arm satisfying `2504 * (2/n) = (12.5/2.8016)^2 = 19.9077`, hence `n = 5008/19.9077 = 251.56`, so 252 stores per arm and 504 in total. With the 1:2 allocation actually available, the variance factor is `1/n_T + 1/(2 n_T) = 1.5/n_T`, giving `n_T = 2504 * 1.5 / 19.9077 = 188.67`, so 189 treated and 378 comparison stores, 567 in total. The unequal design costs `567/504 - 1 = 12.5%` more stores at the same power, which is the number to quote when someone proposes an unequal rollout for operational reasons.

### 4.2 General case: panels, covariates, and staggered timing

#### 4.2.1 The event study

Validity before use: at least three pre-periods (so that a trend can be distinguished from noise), a single treatment date per unit, and a reference period omitted from the regression. With two pre-periods the "pre-trend test" has one degree of freedom and almost no power.

Derivation. Replace the single `Post` indicator with a set of event-time indicators, each equal to one when the observation sits `k` periods from the unit's treatment date. Omitting `k = -1` makes every coefficient a difference relative to the last pre-period, and the coefficient at `k = 0` is the immediate effect while `k > 0` traces how it evolves.

```text
FORMULA [DID-06]: Y_it = a_i + b_t + SUM_{k != -1} d_k * 1[t - g_i = k] + e_it
WHERE:  g_i = treatment date of unit i (undefined for never-treated units,
        which contribute only to a_i and b_t);
        k = event time in periods; d_k in outcome units;
        a_i unit fixed effects, b_t period fixed effects.
CONSTRAINT: single treatment date per unit; no other policy coinciding with
        the event window; d_k identified from units observed at event time k.
NUMERICAL EXAMPLE (ILLUSTRATIVE, six pre and six post months, reference -1):
        k=-6: -2.4 (SE 3.3)   k=-5: +1.1 (3.2)   k=-4: +0.4 (3.1)
        k=-3: +2.2 (3.0)      k=-2: -1.5 (2.9)   k=-1: 0 (reference)
        k= 0: +6.0 (3.4)      k=+1: +9.0 (3.5)   k=+2: +11.5 (3.6)
        k=+3: +13.5 (3.7)     k=+4: +16.0 (3.8)  k=+5: +19.0 (3.9)
        Mean of the six post coefficients = 75.0 / 6 = 12.5 = the 2x2 estimate.
INTERPRETATION: the pre-period coefficients are individually and jointly
        indistinguishable from zero, and the post coefficients rise
        monotonically, which is the pattern a real, ramping effect produces.
        The equality of the post-average with the 2x2 estimate is a check, not
        a coincidence: with equal post-period weights the two are the same
        linear combination.
```

The joint test of the five pre-period coefficients here would be reported as `F = 0.9` on 5 and 118 degrees of freedom, `p = 0.48` (ILLUSTRATIVE). That test failing to reject is weak evidence: as Section 10 records, pre-trend tests have low power against the trend violations that actually matter, so "the pre-trends are fine" should never be written as if it established parallel trends.

#### 4.2.2 Conditional parallel trends

Validity before use: parallel trends is credible only within strata defined by covariates that predict the outcome trend; the covariates must be measured before treatment and must not be affected by it.

Derivation. Replace the unconditional assumption with `E[Y_it(0) - Y_i,t-1(0) | X_i, W_i = 1] = E[Y_it(0) - Y_i,t-1(0) | X_i, W_i = 0]`: the untreated *change* is mean-independent of treatment given `X`. Then the ATT is identified by averaging the within-stratum DiD over the treated distribution of `X`.

```text
FORMULA [DID-07]: ATT = E_{X|W=1}[ E[dY | X, W=1] - E[dY | X, W=0] ]
WHERE:  dY = within-unit change over the comparison window;
        X = pre-treatment covariate vector with overlap in both groups.
CONSTRAINT: X measured before treatment and unaffected by it; overlap for
        every X value with treated mass; the outcome model must be
        correctly specified in the dimension where trends differ, or the
        estimator must be doubly robust.
NUMERICAL EXAMPLE: within region A (60% of treated stores) the DiD is +18.0;
        within region B (40%) it is +4.0; weighted ATT = 0.6*18.0 + 0.4*4.0
        = 10.8 + 1.6 = 12.4 (close to the pooled 12.5, the small gap coming
        from unequal group shares inside regions).
INTERPRETATION: the pooled estimate hides a five-fold difference across
        regions. The weighted average is the same number with a defensible
        variance, and the regional split is the actionable finding.
```

#### 4.2.3 The staggered-adoption decomposition

Validity before use: units adopt at different dates and effects may differ across cohorts or evolve with exposure. Under these conditions the TWFE coefficient is not the ATT, and the decomposition below shows exactly what it is.

Derivation. With unit and time fixed effects, the estimator is a weighted average of all 2x2 comparisons available in the panel: each timing group compared with each later-treated or never-treated group using the window in which the latter is still untreated, plus a comparison of later-treated units against already-treated ones. Weights are proportional to group sizes and to the variance of treatment within the pair. The already-treated-as-control comparisons subtract the *continuing change* of the earlier cohort, so their estimand is a difference between cohort effects rather than an effect level; those weights can be negative relative to the target.

```text
FORMULA [DID-08]: b_TWFE = SUM_{k,l} s_kl * DiD_2x2(k,l),  SUM_{k,l} s_kl = 1
WHERE:  k, l index timing groups; s_kl weights from group sizes and the
        within-pair variance of treatment; DiD_2x2 the pairwise estimate;
        some s_kl may be negative when k is treated before l.
CONSTRAINT: the identity holds exactly for the TWFE estimator on a balanced
        panel; unbalanced panels change the weights and are computed
        numerically rather than from closed forms.
NUMERICAL EXAMPLE (ILLUSTRATIVE, three groups of 30 units, three periods,
        untreated path 100 / 102 / 104 for every group):
        E (early, treated from t=2): effect 0 at t=2, +20 at t=3
            observed 100, 102, 124
        L (late, treated from t=3): effect -5 at t=3
            observed 100, 102, 99
        N (never treated): observed 100, 102, 104
        True unit-period-weighted ATT = (0 + 20 - 5)/3 = +5.0.
        Two-way within transformation on the nine cell means gives
        SUM(Ytilde * Dtilde) = -1.6666 and SUM(Dtilde^2) = 0.6667,
        hence b_TWFE = -1.6666 / 0.6667 = -2.50.
        Pairwise 2x2s: E vs N at t1->t2 = 0; L vs N at t2->t3 = -5;
        E vs L at t1->t2 = 0; L vs E at t2->t3 = (99-102) - (124-102) = -25.
INTERPRETATION: every cohort effect is non-negative except L's -5, the
        average effect is +5.0, and the regression reports -2.50. The sign
        is wrong. The contaminant is the last comparison: using treated
        group E as the control forces L's estimate to include the negative of
        E's continued ramp (+22 change), producing -25 where the true
        L effect is -5.
```

The practical consequence is that in a staggered rollout the TWFE coefficient is not a conservative or noisy version of the ATT — it is a different functional of the cohort effects, and the difference does not vanish as the sample grows, because it is a property of the weighting, not of sampling error. The remedy is to compute `ATT(g, t)` cells under an explicit comparison group and aggregate them with stated weights, as in 4.2.4. A useful review question to ask of any staggered DiD table is simply: "which units serve as the control group for the 2024 cohort in 2025, and are they still untreated then?" If the answer is "the 2023 cohort", the coefficient needs the decomposition before it can be reported.

#### 4.2.4 Group-time ATT and aggregation

Validity before use: the panel is balanced in group-time cells where the estimator is computed, and there exists a valid comparison group for each cohort — either never-treated units or not-yet-treated units, with the choice affecting the estimand and the sample.

Derivation. For each cohort `g` and period `t >= g`, compute a 2x2 using period `g - 1` (or a stable pre-period average) against period `t`, with the comparison group drawn from units untreated at both dates. Each `ATT(g, t)` is then a clean 2x2 under parallel trends within the cohort. Aggregation over `t` and `g` is a separate, explicit choice.

```text
FORMULA [DID-09]: ATT(g,t) = { E[Y_t - Y_{g-1} | G = g] }
                          - { E[Y_t - Y_{g-1} | C] }
WHERE:  G = g is the cohort first treated in period g;
        C is the comparison set (never-treated, or not-yet-treated at t);
        ATT(g,t) in outcome units.
CONSTRAINT: parallel trends holds for each cohort against its comparison
        set; comparison set non-empty at every (g,t) used in the aggregation;
        if not-yet-treated units are used, their own future treatment must not
        affect them before their date (no anticipation).
NUMERICAL EXAMPLE: cohorts g=3 and g=5, periods 3..6.
        ATT(3,3)=8, ATT(3,4)=12, ATT(3,5)=18, ATT(3,6)=24;
        ATT(5,5)=28, ATT(5,6)=32.
        Simple average over all six cells = 122/6 = 20.33;
        cohort-balanced average = (mean for g=3 + mean for g=5)/2
        = (15.5 + 30.0)/2 = 22.75.
INTERPRETATION: the two aggregations are both legitimate and differ by 2.4
        units, because cohort 5 is observed for fewer periods. State which
        one the decision uses; a "the effect is about 20" headline hides a
        choice that a reviewer will ask about.
```

### 4.3 Variants

#### 4.3.1 Propensity-score matching

Validity before use: ignorability given `X`, overlap, and a propensity model that achieves balance. Matching on the score rather than on `X` is justified by the balancing property, so the score's predictive performance is irrelevant to validity.

```text
FORMULA [MAT-01]: e(x) = P(W = 1 | X = x);  match i to j(i) minimising |e(X_i) - e(X_j)|
WHERE:  e(x) in (0,1); j(i) drawn from the untreated pool;
        matching may be 1:1, 1:M, with or without replacement, with a caliper.
CONSTRAINT: ignorability given X; overlap; the propensity model must balance
        every covariate that enters the outcome (check standardised
        differences, not AUC); caliper chosen ex ante and reported.
NUMERICAL EXAMPLE (six units, logit propensity):
        treated T1 0.62, T2 0.48, T3 0.35;
        controls C1 0.60, C2 0.50, C3 0.44, C4 0.33, C5 0.20.
        1:1 without replacement, ties broken by smallest index:
        T1->C1 (0.02), T2->C2 (0.02), T3->C4 (0.02); C3 and C5 discarded.
        Common support after matching: [0.33, 0.62].
INTERPRETATION: C5 at 0.20 has no comparable treated unit; using it as a
        control is extrapolation. Dropping it changes the estimand from "all
        treated" to "treated with support", which must be stated.
```

#### 4.3.2 The matched ATT

Validity before use: as 4.3.1, plus a decision about whether the outcome is compared in levels or in changes. Levels require ignorability of the *level*; changes require ignorability of the *change*, which is the weaker and usually more credible condition in panel data.

```text
FORMULA [MAT-02]: ATT_match = (1/n1) SUM_i [ Y_i - Y_j(i) ]
WHERE:  n1 = number of matched treated units; j(i) the match for i;
        Y measured post-treatment (levels version).
CONSTRAINT: matches formed on pre-treatment covariates only; no match may be
        formed using information from the post period; frequency weights if
        controls are reused.
NUMERICAL EXAMPLE: treated post levels 452, 421, 396 (mean 423.0);
        matched controls 438, 415, 398 (mean 417.0);
        ATT_match = 423.0 - 417.0 = +6.0.
INTERPRETATION: the level comparison gives 6.0, but the treated stores were
        already larger before the intervention; the change version below is
        the one to report.
```

#### 4.3.3 Bias-corrected matching

Validity before use: a regression model for the untreated outcome is available and approximately correct in the region of the matches. The correction removes the first-order bias from imperfect matches without requiring exact matching.

```text
FORMULA [MAT-03]: ATT_bc = ATT_match - (1/n1) SUM_i [ m0(X_i) - m0(X_j(i)) ]
WHERE:  m0(.) = fitted untreated outcome model (here, fitted change);
        the correction term is the average predicted gap between each treated
        unit and its match.
CONSTRAINT: m0 estimated on the untreated sample only; no post-treatment
        covariates; the correction is first-order, so severe extrapolation
        remains uncorrected.
NUMERICAL EXAMPLE: ATT_match = 3.0 (change version, see Section 5);
        mean predicted change for treated profiles 7.4, for their matches 6.5;
        correction = 0.9; ATT_bc = 3.0 - 0.9 = 2.1.
INTERPRETATION: the raw match overstated the effect by 0.9 because the
        matched controls had slightly worse baseline trajectories. Reporting
        3.0 without the correction overstates the effect by 43%.
```

#### 4.3.4 Inverse-probability weighting and doubly robust DiD

Validity before use: for IPW, a correct propensity model; for the doubly robust version, a correct propensity model *or* correct outcome regressions. Overlap is required in both cases, and the estimator degrades badly when propensities approach 0 or 1.

```text
FORMULA [DID-10]: ATT_DR = E[ (W/e - (1-W)/(1-e)) * (dY - m0(X)) ] / E[W/e]
                    with e = e(X), m0 = E[dY | X, W = 0]
WHERE:  dY = within-unit change; e(X) the propensity for treatment;
        m0 the untreated change model; expectation over the pooled sample.
CONSTRAINT: overlap; either e(.) or m0(.) correctly specified; the two models
        estimated on separate samples or with cross-fitting to avoid
        own-observation bias.
NUMERICAL EXAMPLE: with e = 0.5 for all units (a randomised-like design) the
        weights are 2 and 2, and ATT_DR collapses to the difference in mean
        changes, 12.5, regardless of m0 — the double robustness shows up as
        invariance to a deliberately wrong outcome model.
INTERPRETATION: the invariance is the property to demonstrate in a review:
        change m0 drastically, and the estimate should barely move if the
        propensity model is right. If it moves a lot, neither model is
        trustworthy and the sensitivity analysis is mandatory.
```

#### 4.3.5 Matched difference-in-differences

Validity before use: the matched groups are comparable on pre-period levels *and* on pre-period trends, and the differencing removes fixed unobserved differences. This is the recommended default in panel settings because it requires only that the matched units have parallel trends, not that they have equal levels.

```text
FORMULA [DID-11]: ATT_mDiD = (1/n1) SUM_i [ dY_i - dY_j(i) ]
WHERE:  dY = post-window mean minus pre-window mean;
        j(i) matched on pre-treatment covariates, optionally including the
        pre-period level and the pre-period slope.
CONSTRAINT: parallel trends within matched pairs; matches formed without
        post-period information; balance checked on the change, not only on
        the level.
NUMERICAL EXAMPLE: treated changes 22, 9, -4 (mean 9.0);
        matched control changes 14, 6, -2 (mean 6.0);
        ATT_mDiD = 9.0 - 6.0 = +3.0.
        Unmatched control changes 14, 6, 8, 3, -6 (mean 5.0);
        unmatched estimate = 9.0 - 5.0 = +4.0.
INTERPRETATION: matching removes 1.0 of the 4.0, and the removed portion is
        attributable to the two controls with no comparable treated unit
        (one of them falling 6.0). The matched estimate is smaller and more
        defensible; the difference between them is a bias diagnostic.
```

#### 4.3.6 Triple difference

Validity before use: a second dimension in which one subgroup should be unaffected by the intervention, used to difference out a differential trend that would otherwise confound the DiD.

```text
FORMULA [DID-12]: DDD = [DiD_group1] - [DiD_group2]
WHERE:  group1 is exposed to the intervention, group2 is plausibly not,
        both observed before and after in the same markets/regions.
CONSTRAINT: the third difference requires that the differential trend between
        group1 and group2 be the same in treated and comparison regions absent
        treatment; a new assumption replaces the old one rather than removing
        the need for assumptions.
NUMERICAL EXAMPLE: DiD for the affected category = +12.5; DiD for an
        unaffected category in the same stores = +4.0; DDD = 8.5.
INTERPRETATION: if the unaffected category also moved, part of the 12.5 is a
        store-level shock (traffic, staffing) rather than the intervention.
```

### 4.4 Limits and counterexamples

**4.4.1 The pre-trend test cannot establish parallel trends.** A joint test on pre-period coefficients has power only against large, sustained divergences. With the standard errors in 4.2.1 (about 3.0 per coefficient), a drift of 0.4 units per month is undetectable over six months yet accumulates to 2.4 units by the end of the post window — a fifth of the estimated effect. Passing the test is therefore consistent with a material violation, and the correct statement is "we did not detect a differential pre-trend", not "parallel trends holds".

**4.4.2 Parallel trends can hold in levels and fail in logs.** An effect that is constant in absolute units implies a declining effect in percentage terms when the base grows. Reporting "the effect is 12.5 thousand units, or 3.0%" is fine; reporting a log-scale coefficient as if it were scale-free is not, because the percentage changes with the baseline. The choice of scale is a choice about which constant-effect assumption you are willing to defend.

**4.4.3 Composition change mimics an effect.** If treated stores are more likely to close, or if the panel is unbalanced because high-performing stores drop out of the reporting system, the post-period mean is computed on a different set of stores than the pre-period mean. The fix is a balanced panel of units observed in every period used, or an explicit statement of the composition and a re-estimate on the balanced subsample. Section 5 shows the arithmetic of the check.

**4.4.4 Matching cannot fix an unobserved confounder that also drives trends.** Matching balances what is measured. If the selection rule was "the director picked stores she expected to improve", the unmeasured variable is the expectation, and it drives the post-period trend directly. In that case the matched DiD fails, and the honest response is a different design, not a richer covariate set.

**4.4.5 Overlap failure is not a small-sample problem.** When a region has no untreated stores, the matched comparison for treated stores in that region is drawn from elsewhere. The estimate is then a comparison across regions, and its bias has no bound that shrinks with sample size — more data in the wrong place does not help.

**4.4.6 Bad controls.** Conditioning on a variable measured after treatment, or on one that the treatment affects, changes the estimand and can reverse the sign. Store staffing in the post period is a classic: the intervention may have *caused* a staffing change that then mediates part of the effect, and controlling for it removes the mechanism while leaving the confounding attribution intact. Only pre-treatment variables belong on the right-hand side.

**4.4.7 Negative weights in staggered designs.** As 4.2.3 shows, the TWFE coefficient can lie outside the range of cohort effects. A reviewer who asks "is this a weighted average of effects" deserves the decomposition, not a defence of the sample size.

**Takeaway:** each formula is a conditional statement, and the conditions — not the algebra — decide whether the number means anything.


## 5. Worked Numerical Example

### 5.1 The setting and the evidence label

**Evidence:** ILLUSTRATIVE. Every store count, revenue figure, and covariate below is constructed for teaching. The arithmetic has been recomputed by hand and cross-checked three ways (cell means, regression identity, and gap-change identity), but the population is not a company, an industry panel, or a published dataset. No claim about any real retailer is made or implied.

**Workflow:** 120 stores. On 1 July a replenishment-cadence change is applied to 40 of them. The outcome is revenue per store per month, in thousands of currency units, measured for the six months before and six months after. Monthly figures are converted to pre-window and post-window means so that the 2x2 arithmetic is visible; the monthly path matters for the trend check and appears in 5.4.

**Decision:** whether to extend the cadence change to the remaining 80 stores at the start of the next quarter. The extension costs a fixed 60 thousand currency units of systems work plus 0.9 thousand per store per month in training and audit, and the margin rate on incremental revenue is 22%. The break-even incremental revenue is therefore computable, and Section 5.7 computes it.

**Valid assumptions:** the 40 stores were selected before the change date on information observable at that time; the outcome definition did not change in either group during the window; no other programme was rolled out to a subset of stores in the same months; stores in both groups are measured over the same calendar months, so seasonality is common and cancels in the second difference.

**Intentionally simplified:** no store openings or closures in the headline table (5.4 handles the closure case explicitly); no weighting by store size; no covariate adjustment in the headline (5.5 adds it).

### 5.2 The input table

| Group | Stores | Pre mean | Post mean | Change | SD of change | Pre-gap to control |
|---|---:|---:|---:|---:|---:|---:|
| Treated | 40 | 412.0 | 447.5 | +35.5 | 48.0 | +14.0 |
| Comparison | 80 | 398.0 | 421.0 | +23.0 | 52.0 | 0 (reference) |

The group means are unweighted store averages; SD is the standard deviation of each store's post-minus-pre change across stores in that group. Both groups are observed in all 12 months, so the table is balanced.

### 5.3 Step-by-step arithmetic

**Step 1 — treated change.** `447.5 - 412.0 = +35.5`. In thousands per store-month, the treated stores gained 35.5.

**Step 2 — comparison change.** `421.0 - 398.0 = +23.0`. The untreated stores gained 23.0 over the same calendar months, which is the common component: seasonality, market growth, price changes affecting everyone.

**Step 3 — first difference of the differences.** `35.5 - 23.0 = +12.5`. This is [DID-01].

**Step 4 — the gap-change cross-check.** Pre-gap `412.0 - 398.0 = 14.0`; post-gap `447.5 - 421.0 = 26.5`; `26.5 - 14.0 = 12.5`. The identity in [DID-03] holds exactly, which confirms that no arithmetic slip occurred in steps 1–3.

**Step 5 — the regression cross-check.** With `a = 398.0`, `b = 14.0`, `c = 23.0`, `d = 12.5`, the four fitted cell values are control-pre 398.0, treated-pre 412.0, control-post 421.0, treated-post 447.5. All four reproduce the observed means, which confirms the specification is saturated and that the coefficient is the 2x2 estimate rather than some partialled variant.

**Step 6 — naive standard error and t-statistic.** `SE = sqrt(48.0^2/40 + 52.0^2/80) = sqrt(57.60 + 33.80) = sqrt(91.40) = 9.5603`; `t = 12.5/9.5603 = 1.307` on 118 degrees of freedom, two-sided `p` approximately 0.19. Under a normal approximation `p = 0.19`; the exact `t` distribution gives a slightly larger value because the degrees of freedom are large enough that the difference is immaterial here.

**Step 7 — minimum detectable effect.** `MDE = 2.8016 * 9.5603 = 26.79`, or `6.50%` of the treated pre-period mean. The estimate of 12.5 sits at 0.47 of the design's detection threshold.

**Step 8 — the decision-relevant translation.** Incremental revenue of 12.5 per store-month at a 22% margin is `12.5 * 0.22 = 2.75` of gross margin per store-month. Against a running cost of 0.9 per store-month, contribution is `2.75 - 0.9 = 1.85` per store-month. Over 12 months across 80 stores in the extension, that is `1.85 * 12 * 80 = 1776` thousands, against 60 thousands of fixed cost: a favourable ratio, but built entirely on a point estimate whose confidence interval spans zero.

**Step 9 — the interval.** A 95% interval for the effect is `12.5 +/- 1.9600 * 9.5603 = 12.5 +/- 18.74`, i.e. `[-6.24, +31.24]`. Translated to contribution per store-month: `[-6.24 * 0.22 - 0.9, 31.24 * 0.22 - 0.9] = [-2.27, +5.97]`. The lower end is a loss. A decision rule that extends the programme on the point estimate is betting that the interval's lower half is not where the truth lives.

### 5.4 The trend and composition checks

**Trend check.** The monthly treated means are 401, 405, 411, 414, 418, 423 (pre) and 430, 438, 444, 452, 458, 463 (post); the comparison means are 390, 392, 395, 398, 403, 410 (pre) and 411, 414, 418, 422, 428, 433 (post), which averages to 421.0. Treated monthly change in the pre window averages `(423 - 401)/5 = 4.40`; comparison `(410 - 390)/5 = 4.00`. The differential pre-trend is 0.40 per month against a standard error of about 0.9 per monthly coefficient, so it is not detectable — and yet 0.40 per month over the six post months accumulates to 2.4 units, which is 19% of the estimate. This is the concrete illustration of 4.4.1: the test does not reject, and a material violation is still possible.

**Composition check.** Suppose two treated stores closed at the end of the pre-period and are absent from the post means. If those two stores had pre-period revenue of 505 and 512 against the treated average of 412, then the balanced-panel post mean must be recomputed on the 38 surviving stores. The reported post mean of 447.5 includes only survivors, so the pre mean on the same 38 stores is `(412.0 * 40 - 505 - 512)/38 = (16480 - 1017)/38 = 15463/38 = 406.92`. The treated change on the balanced panel is `447.5 - 406.92 = 40.58` rather than 35.5; the naive mean had *understated* the change because it carried two high-revenue stores in the pre-period mean and dropped them afterwards. Here the direction is favourable to the programme; in the general case the sign is determined by whether exiting units were above or below their group mean, which is why the check must be run rather than assumed.

**The closure bias quantified.** With `n_pre = 40`, `n_post = 38`, drop-outs with mean `m_d = 508.5`, survivors with pre mean `m_s = 406.92`, the composition bias in the estimated change is

```text
bias = (n_pre/n_post - 1) * (m_pre - m_d)
     = (40/38 - 1) * (412.0 - 508.5) = 0.052632 * (-96.5) = -5.08
```

so the naive change understates the balanced one by 5.08, which matches `40.58 - 35.50 = 5.08` exactly. The corresponding DiD moves from 12.5 to `40.58 - 23.0 = 17.58`, a 41% increase produced entirely by who was counted.

### 5.5 Covariate adjustment

The regional split from [DID-07]: 24 treated stores in region A with a within-region DiD of +18.0, 16 in region B with +4.0. The weighted average with shares 0.6 and 0.4 is `0.6 * 18.0 + 0.4 * 4.0 = 12.4`. The 0.1 difference from the pooled 12.5 is a rounding artefact of the illustrative inputs, not a second estimate; when the shares inside each region are unequal, the pooled and stratified estimates differ by the covariance between the regional effect and the regional treatment share, and that difference is informative rather than erroneous.

What the split changes is the recommendation. A programme with an average of 12.5 and a range of 4.0 to 18.0 across two regions does not have one decision; it has two. Extending to region A is favourable on any plausible interval; extending to region B is not. Reporting only the pooled number forces a single yes/no onto a population where the answer differs, which is the analytical version of a category error.

### 5.6 Matched difference-in-differences on the same data

Matching on pre-period revenue band, region, and store format produces 31 usable treated stores with at least one comparable control inside a caliper of 0.2 standard deviations of the logit propensity score. The matched comparison group has 47 stores (some reused). Within the matched sample the treated change is 22.4 and the matched control change is 19.4, giving `ATT_mDiD = +3.0`.

The collapse from 12.5 to 3.0 is the most important number in this section. The unmatched comparison group contains stores that are systematically smaller and in regions with slower growth; once the comparison is restricted to stores of similar size, region, and format, most of the apparent effect disappears. Which estimate is right? Neither is "right" without the assumption, but the matched one is the one whose comparison group is defensible on observable characteristics, and its standard error is larger because the sample is smaller: `SE_matched = 11.9`, giving `t = 3.0/11.9 = 0.25`. The honest reading is that this design cannot distinguish the matched estimate from zero, and the unmatched one from zero either once clustering and matching are taken seriously.

The 9.5-unit gap between the two estimates is itself a diagnostic. It measures how much of the raw DiD is attributable to differences in the composition of the comparison group. When that gap is large relative to the standard error — here `9.5/11.9 = 0.80` standard errors, which is not large — there is evidence that the unmatched design is confounded by observables. A gap of three standard errors would be a red flag requiring the matched design as the primary specification.

### 5.7 Break-even and the recommendation

The extension's economics: fixed 60 thousand, running 0.9 per store-month, margin 22%, 80 stores, 12 months. Total running cost `0.9 * 80 * 12 = 864`, so the total cost is `60 + 864 = 924` thousand. Required incremental revenue to break even is `924 / 0.22 = 4200` thousand across 80 stores over 12 months, i.e. `4200 / (80 * 12) = 4.375` per store-month. The matched estimate of 3.0 is below break-even; the unmatched estimate of 12.5 is well above it. The decision therefore hinges entirely on which comparison group is credible, which is why Section 5.6 precedes this paragraph rather than the other way round.

**Independent checks performed.** (a) Both DiD forms agree at 12.5. (b) The regression's four fitted cells reproduce the four observed means. (c) The composition bias formula reproduces the difference between naive and balanced changes to two decimals. (d) Contributions in step 8 recompute from margin times revenue minus cost, and the 12-month total `1.85 * 960 = 1776` matches `(12.5 * 0.22 * 960) - (0.9 * 960) = 2640 - 864 = 1776`. (e) Units are consistent: thousands of currency units per store-month throughout, except where a total over stores and months is explicitly labelled.

**Wrong path (plausible mistake).** Report the post-period gap of 26.5 as the effect: "treated stores are 26.5 ahead". This overstates the estimate by the pre-existing gap of 14.0, and the source of that gap is selection. A second wrong path is to control for post-period staffing in the regression; staffing rose because of the intervention, so the control absorbs part of the effect and the coefficient drops without any gain in validity.

**Production-scale change.** At 4,000 stores and 36 months the sample is large enough that the standard error shrinks to roughly `9.56 * sqrt(120/4000) = 1.66` under proportional scaling, and a 3.0 effect becomes detectable — but the confounding does not shrink, so the matched design becomes more valuable, not less, at scale.

**What the example does not establish.** It does not establish that the cadence change had an effect of 12.5, of 3.0, or of anything. It establishes the arithmetic of the estimator, the size of the composition and matching gaps in a constructed case, and the fact that a decision can reverse when the comparison group is changed. That is a teaching outcome, not a business result.

**Takeaway:** the arithmetic takes ten minutes; deciding which comparison group is credible takes the rest of the day, and only the second part changes the recommendation.

## 6. Visual / Geometric Intuition

### 6.1 The parallel-trends picture and its two failure directions

```text
 outcome per store-month (thousand currency units)
 460 |                                        *  T post (447.5)
     |                                   .-'
 445 |                              .-'
     |                         .-'
 430 |  .....................-'..............*  T counterfactual (435.0)
     |  .              .-'''                    (parallel to control)
 415 | .        .-'''      *  C post (421.0)
     |.   .-'''
 400 |* T pre (412.0)
     |* C pre (398.0)
     +---------------------------------------------> time
       Jan   Feb   Mar   Apr   May   Jun | Jul ... Dec
                                        ^
                                  intervention (1 July)
```

Schematic, not to scale. The vertical distance between the observed treated point and the dashed counterfactual is the estimate of 12.5. Two distortions of the dashed line are worth naming. If the true counterfactual rises faster — because treated stores were selected for momentum — the dashed line should be steeper, the gap smaller, and 12.5 overstates the effect. If a common shock raised every store's revenue in the post period but the shock was stronger in the treated group's region, the control change of 23.0 understates the counterfactual change and the estimate again overstates. The diagram cannot distinguish these; only a design that supplies another comparison dimension can.

### 6.2 The event-study plot as the diagnostic instrument

```text
 coefficient (thousand currency units), 95% CI shown as [---]
  20 |                                        [---]
     |                                   [---]
  15 |                              [---]
     |                         [---]
  10 |                    [---]
     |               [---]
   5 |          [---]
     |     [---]
   0 +-[---]-+-----+-----+-----+-----+-----+-----> event time k
     |-6 -5 -4 -3 -2 |-1  0  +1  +2  +3  +4  +5
     |               reference
  -5 |  pre-period coefficients straddle zero
```

Read the pre-period coefficients first: five of them near zero with intervals crossing the axis is the pattern consistent with parallel trends, though as 4.4.1 insists, it is not proof. Read the post-period shape second. A jump at `k = 0` followed by a flat plateau is an immediate, permanent effect. A ramp from zero with no jump is a diffusion story and argues for a longer measurement window. A jump that decays within three months argues for a novelty or announcement effect and against a structural change. A negative pre-period coefficient at `k = -2` followed by a jump to a positive one at `k = 0` is a classic anticipation signature: behaviour changed before the recorded date.

### 6.3 Matching geometry: the support region in two dimensions

```text
 pre-period revenue (thousand)
 520 |                                  o  o        o = untreated
     |                            o   o              * = treated
 480 |                       o  o                     shaded = overlap
     |                  o  o        *
 440 |             o  o        *  *  *
     |        o  o       *  *  *  *
 400 |   o  o      *  *  *  *
     |  o     *  *  *  *               treated store with no
 360 |     *  *                        nearby control (extrapolation)
     |  o
 320 +------------------------------------------> store size (sqm)
        200   400   600   800   1000  1200
```

The treated store at the upper right sits at 480 revenue and 1,150 square metres, where the untreated points are sparse. Its nearest neighbour is far away in covariate space, and its matched difference is dominated by that distance. The honest options are to discard it, to report the estimate on the supported subset, or to fit an outcome model that is validated in that region. What is not an option is to keep it and report the average as if it were supported everywhere.

### 6.4 The bias-variance trade of the matching caliper

```text
 bias (bias direction)                variance
    ^                                    ^
    |*                                   |*
    | *                                  |  *
    |  *                                 |    *
    |   *                                |      *
    |     *                              |        *
    |        *                           |            *
    +------------------> caliper width   +------------------> caliper width
      narrow        wide                   narrow        wide
```

Narrowing the caliper removes bad matches (lower bias) but discards treated units and reduces the effective sample (higher variance), and it changes the estimand to the effect on a better-supported subset. There is no single correct width; there is a width you chose, a balance table you reported, and a sensitivity check showing how the estimate moves as the width changes. A caliper presented without that sensitivity is a hidden modelling decision.

### 6.5 The four-cell square

```text
                    pre              post
              +--------------+--------------+
   treated    |  412.0       |  447.5       |   change +35.5
              |  (Y(0) obs)  |  (Y(1) obs)  |
              +--------------+--------------+
   comparison |  398.0       |  421.0       |   change +23.0
              |  (Y(0) obs)  |  (Y(0) obs)  |
              +--------------+--------------+
                   DiD = 35.5 - 23.0 = 12.5
```

The square is the whole design. Everything else — regression, matching, event study, doubly robust weighting — is machinery for the case where the square's cells are not directly comparable because they are composed of different units or weighted differently. Drawing the square with its four cell means before running any estimator is the cheapest way to catch a filter that dropped one cell's rows.

**Takeaway:** the trend plot answers "could the counterfactual be right", the event study answers "when did the effect start", and the support plot answers "for whom is the estimate defined".

## 7. Comparison / Selection Table

### 7.1 The method table

| Method | Identification assumption | Estimand recovered | Data required | Cost / risk | Use when | Avoid when | Evidence |
|---|---|---|---|---|---|---|---|
| Randomised experiment | Random assignment, SUTVA | ATE (or ATT under partial rollout) | Assignment mechanism under your control | Highest design cost; may be infeasible or unethical | You can randomise before rollout | Rollout already happened, or assignment is legally fixed | DERIVED |
| Raw pre/post on treated only | Nothing beyond "no other change" | None in general | Treated series only | Very cheap; almost always biased | Never for causal claims; fine for monitoring | Any causal question | DERIVED |
| 2x2 DiD | Parallel trends, no anticipation, SUTVA | ATT | Two groups, two windows, balanced panel | Moderate; fragile to differential shocks | A clean single-date rollout with a plausible donor group | Groups chosen on expected trend; coincident regional shock | DERIVED |
| Event study / TWFE | Parallel trends per event time | ATT path over event time | Panel with 3+ pre periods | Moderate; pre-trend test has low power | You want dynamics and a visible pre-trend check | Staggered adoption with heterogeneous effects (use 4.2.4) | DERIVED |
| Group-time ATT + explicit aggregation | Parallel trends per cohort vs stated comparison | `ATT(g,t)`, aggregate with stated weights | Panel, cohort structure, valid comparison set | More computation, more decisions to disclose | Staggered adoption, any effect heterogeneity | No never- or not-yet-treated units available | SOURCED (Callaway–Sant'Anna lineage; record as OPEN for exact citation) |
| PSM on cross-section | Ignorability given X, overlap | ATT on the supported subset | Cross-section plus rich pre-treatment covariates | Moderate; sensitive to propensity specification and caliper | No panel, strong covariate set, credible ignorability | Unmeasured selection; poor overlap; panel available (use matched DiD) | DERIVED |
| PSM + DiD | Ignorability of the change given matched X, parallel trends within matched pairs | ATT on matched treated | Panel plus pre-treatment covariates | Moderate; two decisions (match, difference) to defend | Panel available and groups differ in composition | Post-treatment covariates needed to match; overlap failure | DERIVED |
| IPW / doubly robust DiD | Correct propensity or correct outcome model | ATT (population-weighted variants possible) | Panel plus models | Higher; extreme weights can dominate | Model uncertainty is the main worry, overlap is good | Propensities near 0/1; tiny effective sample after weighting | DERIVED |
| Synthetic control | A weighted combination of untreated units reproduces the treated unit's pre-period path | Effect for one (or few) treated units | Long, stable pre-period; donor pool with no spillover | High per-unit effort; inference by placebo | One region, city, or unit treated; long history | Many treated units; short pre-period; donors affected by the intervention | SOURCED (record as OPEN for exact citation) |
| Regression discontinuity | Continuity of potential outcomes at a threshold | Effect at the cutoff (local) | Running variable with a known threshold and enough mass near it | High; sensitive to bandwidth and manipulation | Treatment assigned by a score threshold | Threshold manipulable; no mass near cutoff; effect wanted away from cutoff | SOURCED (record as OPEN for exact citation) |
| Instrumental variables | Relevance, exclusion, monotonicity (for LATE) | LATE for compliers | A credible instrument | High; exclusion untestable | An exogenous shifter of treatment exists | Instrument correlated with the outcome except through treatment | SOURCED (record as OPEN for exact citation) |
| Triple difference | Differential trend between subgroups is stable across regions | ATT net of the subgroup differential trend | A second, plausibly unaffected dimension | Moderate; one more assumption to defend | A within-unit control category exists | The "unaffected" group is in fact affected | DERIVED |

The table's honest reading is that the assumptions differ in kind, not in strength. Parallel trends and ignorability are not ordered; a matched cross-section with excellent covariates can be more credible than a DiD across two structurally different regions, and vice versa.

### 7.2 The selection guide

```text
IF assignment was random and recorded            -> difference in means; stop.
ELSE IF a clean threshold determines treatment   -> regression discontinuity.
ELSE IF a credible instrument exists             -> IV/LATE; state the complier population.
ELSE IF one or few units are treated with a long
        stable pre-period and clean donors       -> synthetic control.
ELSE IF panel data and a single treatment date   -> 2x2 DiD, then event study.
ELSE IF panel data and staggered adoption        -> group-time ATT with explicit
                                                    aggregation; never raw TWFE.
ELSE IF cross-section with rich pre-treatment
        covariates and defensible ignorability   -> PSM or IPW on the supported set.
ELSE IF panel AND groups differ in composition   -> matched DiD, with balance table.
ELSE                                             -> report "not identified"; propose
                                                    a design or a bounded claim.
If conditions conflict: validity beats precision. A wide interval on an identified
estimand is information; a narrow interval on an unidentified one is noise with
a decimal point.
```

### 7.3 Choosing between matching and DiD when both are feasible

The question is which assumption you are more willing to defend. Matching requires that, conditional on covariates, the untreated outcome is independent of treatment — an assumption about *levels* and about unobserved variables. DiD requires that the untreated *change* is independent of treatment — an assumption about trends, which is often more credible because persistent differences between units are differenced away.

Panel data therefore usually favours DiD, and the matched variant when composition differs. Cross-sectional data with a rich pre-treatment covariate set, where the outcome is measured once after the intervention, favours matching. When both are feasible, the informative move is to compute both and report the gap, as 5.6 does: the gap is a diagnostic on composition, and a large gap argues for the matched design as the primary specification.

### 7.4 What each method cannot do

Matching cannot handle a confounder that was never measured, and cannot fix a comparison group drawn from a different population. DiD cannot handle a differential shock that coincides with the treatment date, and cannot recover anything when the treated group has no plausible donor pool. Synthetic control cannot handle many treated units or spillovers into the donor pool. Regression discontinuity cannot answer a question about units far from the threshold. IV cannot answer a question about the average unit when the complier population is selected. Triple difference cannot substitute for a valid third dimension; it moves the assumption rather than removing it.

**Takeaway:** select on the assumption you can defend and falsify, not on the estimator you can run.

## 8. Failure Modes & Edge Cases

### 8.1 F-01 — Differential pre-trend read as noise

**Trigger/symptom.** The event-study pre-period coefficients are individually insignificant, the analysis proceeds, and the post-period estimate is later found to shrink when a longer pre-period is used.

**Competing causes.** (a) A genuine differential trend that the test lacks power to detect, because pre-period coefficients have standard errors around 3.0 while the drift is 0.4 per month. (b) A level shift at the treatment date that the pre-period cannot see by construction. (c) A composition change that begins before the date, for example stores dropping out ahead of a planned closure.

**Discriminating check.** Extend the pre-period by as much history as the data permits and re-estimate the event study; a coefficient pattern that grows in magnitude as you go back is a trend, not noise. Compare the *linear* pre-trend coefficient, which pools the pre-period information and has a much smaller standard error than any single event-time coefficient, against zero. And run the placebo-date test: assign the treatment to a date inside the pre-period and see whether a comparable "effect" appears.

**Bounded fix.** If a differential pre-trend is detected, three options remain: include a group-specific linear trend (which absorbs the trend and requires the trend to be linear, a strong assumption), re-weight the comparison group on pre-period trend (the same operation as matching on trend), or report a bound rather than a point estimate by extrapolating the pre-trend into the post window and subtracting it.

**Verification.** Re-run the event study on the extended window and confirm the pre-period coefficients are stable; confirm that the trend-adjusted estimate is inside the bound computed from the pre-trend extrapolation. If the bound includes zero, the claim must be softened to "not distinguishable from zero under a differential trend of the observed size".

**Provenance.** HYPOTHETICAL, mechanism DERIVED from [DID-06] and the power argument in 4.4.1.

### 8.2 F-02 — Composition change inside the panel

**Trigger/symptom.** The panel is unbalanced: treated stores are more likely to exit, or new stores enter the comparison group after the treatment date. The 2x2 means are computed on different store sets in the pre and post windows.

**Competing causes.** (a) Differential attrition driven by the treatment itself — the intervention raised costs and marginal stores closed. (b) Attrition driven by the outcome — poor performers stopped reporting. (c) A purely administrative change, such as a reporting-system migration that dropped some stores from the extract.

**Discriminating check.** Count stores per group per period and compare with the extract's source system. Recompute the estimate on the balanced panel and compare with the unbalanced one; the 5.4 example shows a 5.08-unit bias from two closings, which is 41% of the estimate. For administrative attrition, check whether the missing stores are present in an independent source (a finance ledger) with different values.

**Bounded fix.** Use a balanced panel of stores observed in all periods as the primary specification. If the balanced panel changes the estimand materially, report both and state which population each refers to. If attrition is treatment-induced, the estimand becomes the effect on survivors and must be labelled as such — it is a different and usually less useful quantity.

**Verification.** Store counts reconcile across periods within the extract; balanced and unbalanced estimates are both reported; the difference is explained by the exit pattern rather than left as an unexplained discrepancy.

**Provenance.** HYPOTHETICAL, arithmetic VERIFIED in 5.4.

### 8.3 F-03 — Searching estimators until one is significant

**Trigger/symptom.** The analysis plan lists six estimators, and the reported one is whichever gave `p < 0.05`. The write-up says "robustness checks confirm the result".

**Competing causes.** (a) Genuine model dependence, where the estimand really does vary by specification and the variation is informative. (b) Specification search, where the analyst tried several and reported the favourable one. (c) A single coding error in one variant that happened to move the estimate across the threshold.

**Discriminating check.** Ask for all specifications, pre-registered or not, with their estimates side by side. A spread of estimates with correlated errors is expected; the question is whether the reported one is at the extreme of the distribution and whether the choice rule was stated before seeing the results. Check whether the "robustness" variants differ from the main specification on a dimension that should matter for the estimand — if they do, they are not robustness checks but different questions.

**Bounded fix.** Declare the primary specification before estimation, on identification grounds rather than on the estimate. Report the full specification curve with the primary marked. Treat the spread across defensible specifications as part of the uncertainty, not as a menu.

**Verification.** The specification curve is reproducible from a script; every variant's code differs only in the declared dimension; the reported estimate is the pre-declared primary; the conclusion is stable under the plausible range.

**Provenance.** HYPOTHETICAL, mechanism DERIVED from the multiple-comparison logic in file 01.

### 8.4 F-04 — Conditioning on a post-treatment variable

**Trigger/symptom.** The regression includes store staffing, inventory turns, or promotion intensity measured after the treatment date. The coefficient shrinks toward zero or changes sign, and someone argues that the effect "runs through" the control.

**Competing causes.** (a) The control is a mediator: the intervention changed it, and it changed the outcome, so controlling removes part of the effect. (b) The control is a collider: both treatment and outcome influence it, so controlling induces a spurious association. (c) The control is genuinely pre-treatment but time-varying, and its inclusion is legitimate.

**Discriminating check.** Establish the timing of each covariate relative to the treatment date from the data dictionary, not from memory. For each suspect covariate, run the estimate with and without it: a large movement is evidence of mediation or collider bias, not of confounding removal. Examine whether the covariate's own treatment effect is nonzero; if it is, the covariate is downstream.

**Bounded fix.** Remove post-treatment covariates from the primary specification. If their role matters for interpretation, report a mediation analysis separately with its own assumptions, and never label the mediated estimate as "the effect net of staffing".

**Verification.** Every covariate on the right-hand side has a recorded measurement date earlier than the treatment date; the primary estimate is invariant to dropping any post-treatment variable by construction, because none is included.

**Provenance.** HYPOTHETICAL, mechanism DERIVED from the backdoor criterion in file 02.

### 8.5 F-05 — Overlap failure disguised as a match

**Trigger/symptom.** Every treated unit gets a match, the balance table looks acceptable on average, and the estimate is precise. But some treated units are matched to controls with propensity scores far away, and a trimmed analysis moves the estimate substantially.

**Competing causes.** (a) Genuine lack of common support in a region, format, or size band. (b) A propensity model that is misspecified in the tails, producing scores that are comparable numerically but not substantively. (c) A nearest-neighbour algorithm forced to return a match when none exists — a default in several packages.

**Discriminating check.** Plot the propensity distributions of both groups and compute the share of treated units outside the control range. Re-estimate with a caliper and report how many treated units are discarded and how the estimate moves. Inspection of the worst matches — the largest propensity distance — usually reveals whether the issue is systematic or a tail artefact.

**Bounded fix.** Impose a caliper declared in advance, report the discarded share, and redefine the estimand as the effect on the supported subset. If the discarded share is large, the honest conclusion is that the comparison group cannot support the question, and a different design is needed.

**Verification.** Propensity overlap plot included; discarded share reported with the estimate; the estimate is stable across caliper widths inside the reported sensitivity band.

**Provenance.** HYPOTHETICAL, mechanism DERIVED from the overlap definition in 3.2.

### 8.6 F-06 — Standard errors that ignore the assignment level

**Trigger/symptom.** The estimate is computed on 480 treated rows and 960 comparison rows, the standard error is small, and the result is significant. The treatment was assigned to 40 stores.

**Competing causes.** (a) Clustering ignored: rows within a store are correlated. (b) Clustering applied at the measurement level rather than the assignment level. (c) Few clusters, so even a clustered standard error is too small.

**Discriminating check.** Compare the row-level standard error with the standard error from the collapsed unit-level data. They should agree when the specification is saturated in the unit dimension and the design effect is correctly applied; a factor of two or more of disagreement indicates that the correction was applied to the wrong correlation. Recompute with clustering at the assignment level and, if clusters are fewer than about 40, with a wild cluster bootstrap.

**Bounded fix.** Report clustered standard errors at the assignment level as the default, with the number of clusters stated. For few clusters, use the bootstrap or a small-sample correction and state the method.

**Verification.** Standard error from the row-level model with clustering equals the collapsed-data standard error to within Monte Carlo error; the number of clusters appears in the table; the inference does not depend on the choice between the two routes.

**Provenance.** HYPOTHETICAL, arithmetic VERIFIED in 4.1.4.

### 8.7 F-07 — Staggered adoption reported as a single ATT

**Trigger/symptom.** The TWFE coefficient is reported as "the effect of the programme", the rollout happened in four waves, and the effects differ by wave.

**Competing causes.** (a) Negative weighting from already-treated controls, as in 4.2.3, so the coefficient is not an ATT. (b) Genuine effect heterogeneity across cohorts, where the TWFE coefficient is a legitimate weighted average of a different set of estimands. (c) Effect dynamics, where early cohorts' effects grow and contaminate later comparisons.

**Discriminating check.** Decompose the coefficient into its 2x2 components and inspect the weights, including their signs. Estimate `ATT(g, t)` and check whether the TWFE coefficient lies inside the range of the cohort effects; if it lies outside, negative weighting is operating.

**Bounded fix.** Report group-time ATTs with an explicit aggregation and its weights. If the decision needs a single number, say which aggregation produced it — cohort-balanced, exposure-balanced, or calendar-weighted — and show the alternatives.

**Verification.** The decomposition reproduces the TWFE coefficient exactly; the aggregated estimate is inside the cohort range; the aggregation weights sum to one and are reported.

**Provenance.** HYPOTHETICAL, arithmetic VERIFIED in 4.2.3.

### 8.8 F-08 — Anticipation contaminating the pre-period

**Trigger/symptom.** A negative pre-period coefficient immediately before the treatment date, then a positive one after. The estimate is smaller than the operational team expected.

**Competing causes.** (a) Genuine anticipation: stores prepared before the date, so part of the effect is already in the pre-period. (b) A calendar artefact: the reference period was unusual, making the next period look negative by comparison. (c) A data-timing artefact: the treatment date in the extract is earlier than the operational date.

**Discriminating check.** Compare the recorded date with the operational communication date, the systems-change date, and the first date on which treated stores' behaviour differs. If behaviour diverges before the recorded date, the date is wrong or anticipation is real. Re-estimate with the reference period moved earlier.

**Bounded fix.** Move the treatment date to the first behavioural change, or drop the contaminated pre-period from the window and state the shortened pre-period. If the intervention was announced in advance, the estimand becomes the effect of announcement plus implementation, which must be labelled.

**Verification.** The event-study plot shows no divergence before the chosen date; the operational and analytical dates are reconciled and recorded; the estimate is stable to the reference-period choice.

**Provenance.** HYPOTHETICAL, mechanism DERIVED from the no-anticipation condition in 3.2.

### 8.9 F-09 — Spillover into the comparison group

**Trigger/symptom.** The effect is smaller in markets where treated and comparison stores are geographically close, and larger where they are separated.

**Competing causes.** (a) Demand substitution: customers shift between treated and comparison stores in the same catchment, so the comparison group loses revenue the treated group gains. (b) Supply-side interference: shared distribution capacity means a treated store's gains come at a comparison store's expense. (c) Genuine market heterogeneity that happens to correlate with geography.

**Discriminating check.** Estimate the effect separately for comparison stores near treated stores and far from them. If near comparison stores underperform far ones, spillover is present. Check whether the total across both groups is unchanged — pure substitution leaves the sum flat.

**Bounded fix.** Define the comparison group to exclude stores within a stated catchment radius, accepting a smaller sample. Report the estimand as the effect under the observed spillover environment; a universal rollout would have different spillovers and a different effect.

**Verification.** The distance-stratified estimates are reported; the aggregate effect of a universal rollout is discussed separately rather than extrapolated from the partial one; the exclusion radius is documented.

**Provenance.** HYPOTHETICAL, mechanism DERIVED from SUTVA in 3.2.

### 8.10 F-10 — Treatment misclassification

**Trigger/symptom.** The treatment flag records intention rather than delivery, or records delivery for stores where the new cadence was never actually adopted. The estimated effect is smaller than the operational team's own measurement.

**Competing causes.** (a) Intention-to-treat versus effect-of-treatment: the ITT is attenuated by non-compliance by construction. (b) A label applied at the wrong grain: the flag is set on the warehouse, but several stores share a warehouse. (c) A mixture of doses under one label.

**Discriminating check.** Cross-tabulate the treatment flag against an independent adoption signal — a system log of the new cadence, an audit sample, or a manager confirmation. Compute the compliance rate. The ratio of ITT to the compliance rate bounds the effect-of-treatment under monotonicity.

**Bounded fix.** Report the ITT as the primary estimand, which is the correct one for a rollout decision, and report the compliance-adjusted estimate as secondary with its assumption stated. Fix the flag at the grain of delivery, not of intention.

**Verification.** Compliance rate reported; the flag reconciles to an independent source for a random sample of units; the two estimands are both labelled and neither is presented as the other.

**Provenance.** HYPOTHETICAL, mechanism DERIVED from consistency in 3.1.

### 8.11 F-11 — Scale and functional form

**Trigger/symptom.** The effect is significant in levels and not in logs, or vice versa. The two analyses are presented as a robustness check that "confirms" the result.

**Competing causes.** (a) A constant absolute effect implies a declining percentage effect as the base grows; a constant percentage effect implies a growing absolute effect. The two cannot both hold. (b) Heteroskedasticity: the log transform stabilises variance when the outcome is proportional to size. (c) Genuine nonlinearity in the treatment response.

**Discriminating check.** State which constant-effect assumption the decision requires. If the business cares about total margin, levels are appropriate; if about percentage lift across differently sized stores, logs are. Check whether the conclusion differs between the two scales; if it does, the decision rule must be stated in the scale the business uses.

**Bounded fix.** Choose the scale from the decision, not from the result, and report the estimate in both with the conversion shown. Do not describe a scale-dependent sign change as a robustness success.

**Verification.** The scale choice is justified in the analysis plan; both estimates are reported; the recommended action is the same under both, or the divergence is disclosed.

**Provenance.** HYPOTHETICAL, mechanism DERIVED from 4.4.2.

### 8.12 The edge-case checklist

Beyond the eleven failures, five boundary conditions recur in review. **A single pre-period** makes the event study impossible and leaves parallel trends entirely untested; the honest statement is that the design rests on an untestable assumption. **A treatment date in the first observed period** leaves no pre-period at all and the design collapses to a cross-sectional comparison. **A comparison group of one unit** makes any variance estimate meaningless; this is the domain of synthetic control with placebo inference instead. **Outcome measurement changed in one group** — a new revenue definition applied only to the pilot stores — produces an artefact indistinguishable from an effect, and must be checked against the finance ledger rather than the operational dashboard. **A simultaneously rolled-out national campaign** raises both groups' outcomes and cancels in the second difference only if its effect is proportional in both groups; if it interacts with store size and the groups differ in size, it does not cancel.

**Takeaway:** the failures that matter are not arithmetic errors; they are identifications that were assumed rather than checked, and the check is usually a second cut of the same data.

## 9. Complexity & Cost Analysis

### 9.1 Computational cost of the estimator family

The sizes that matter are `n1` treated units, `n0` comparison units, `T` periods, `P` covariates, and `R` bootstrap or placebo replications. Matching cost is dominated by the distance computation, DiD cost by the linear solve, and inference cost by the resampling. Each is stated in operation counts, not seconds, because seconds are a property of the machine and the implementation.

| Operation | Input sizes | Time | Space | Dominant term / assumption |
|---|---|---|---|---|
| Propensity model fit (logit, IRLS) | `n = n1+n0` rows, `P` covariates, `I` iterations | `O(I * n * P^2)` | `O(n * P)` | Design matrix in memory; `I` typically 5–25 until convergence |
| Full distance matrix, 1:1 matching | `n1`, `n0` | `O(n1 * n0 * P)` | `O(n1 * n0)` | Space is the binding constraint: 40 x 80 is trivial, 50,000 x 200,000 is 10^10 cells |
| Greedy nearest-neighbour with k-d tree | `n1`, `n0`, `P` | `O((n1+n0) log n0)` expected in low `P` | `O(n0)` | Expected cost degrades toward `O(n1 * n0)` as `P` grows past about 10 |
| Optimal (Hungarian-style) 1:1 matching | `n1`, `n0` | `O(n1^2 * n0)` | `O(n1 * n0)` | Prohibitive above a few thousand treated units |
| 2x2 DiD, closed form | 4 cell aggregates | `O(n)` to aggregate | `O(1)` | No matrix inversion |
| TWFE OLS with two-way FE | `N = n * T` rows, `K` regressors | `O(N * K^2)` per iteration after within-transformation | `O(N * K)` | With `n` unit dummies, absorbing the intercepts first is essential |
| Event study | `N` rows, `2T - 1` event dummies | `O(N * T^2)` dense, `O(N * T)` with sparse algebra | `O(N * T)` | Wide event windows make the design matrix dense |
| Group-time ATT (`ATT(g,t)` cells) | `G` cohorts, `T` periods | `O(G * T * n)` | `O(G * T)` | One 2x2 per cell; the cost is in bookkeeping, not algebra |
| Wild cluster bootstrap | `R` replications, `C` clusters | `O(R * N * K^2)` | `O(N * K)` | `R = 999` conventional; cost linear in `R` |
| Placebo / permutation inference | `R` replications | `O(R * cost of one estimate)` | `O(R)` for stored statistics | The estimate must be cheap enough to run 500–1,000 times |

Two practical consequences follow. First, never build the full distance matrix when either group exceeds a few thousand units; use a tree or a blocked design in which matching happens within strata (region, format, size band), which also makes the match auditable. Second, the group-time estimator is expensive only in bookkeeping: the arithmetic per cell is a 2x2, so the compute budget is dominated by data manipulation rather than by any linear algebra.

### 9.2 Worked sizing: analysing a 4,000-store panel

**Workload.** 4,000 stores, 48 months, 192,000 store-month rows, 22 covariates, 12 event-time coefficients, four bootstrap replications of the whole pipeline for a specification curve.

**Environment.** A single analysis machine with 32 GiB of memory and a columnar data engine; no distributed cluster. Figures below are operation and byte counts from the schema, not measured runtimes.

**Calculation — memory.** At 8 bytes per numeric cell, the design matrix for the event study is `192000 * (22 + 12 + 1) * 8 = 192000 * 35 * 8 = 53,760,000 bytes = 51.27 MiB`. With the unit fixed effects stored as a 4,000-level categorical (integer codes, 4 bytes per row) the additional working set is `192000 * 4 = 768,000 bytes = 0.73 MiB`, so the dense matrix dominates. Absorbing the unit intercepts by within-transformation avoids materialising 4,000 dummy columns, which would otherwise be `192000 * 4000 * 8 = 6.144e9` bytes, about 5.72 GiB — large but not fatal, and it becomes fatal at 10,000 stores.

**Calculation — matching.** A full 800 x 3,200 distance matrix on 22 covariates is `800 * 3200 * 22 * 8 = 450,560,000` bytes, about 429.7 MiB for the distances alone, plus the index arrays. Blocking into 8 regions reduces the largest block to roughly 120 x 480, which is `120 * 480 * 22 * 8 = 10,137,600` bytes, about 9.7 MiB. Blocking is a 44-fold reduction in the working set and a much larger reduction in the number of distance evaluations.

**Calculation — bootstrap.** 999 replications of an estimator that takes 2 seconds each is `999 * 2 = 1,998` seconds, about 33 minutes. At 4,000 stores and 48 months the same estimator's linear algebra is larger, so the practical choice is either fewer replications, parallel workers, or a closed-form variance with the bootstrap reserved for the few-cluster case where it is actually needed.

**Bottleneck.** Data movement, not arithmetic. The 192,000-row panel is small enough that the whole analysis fits in memory; the binding constraints are (a) the correctness of the join that attaches treatment dates, (b) the group-time bookkeeping, and (c) the number of bootstrap replications. Measurement that would confirm this is a profile of the pipeline rather than of the estimator.

**Economics.** The analysis itself costs analyst-days, not compute: at a loading of 800 currency units per analyst-day, a two-day specification-curve run costs 1,600, against a rollout whose twelve-month contribution is measured in millions. The asymmetry is the point: spending two more analyst-days on the pre-trend check and the balance table is cheap relative to a wrong rollout decision, and the compute cost is not the constraint that should ration rigour.

### 9.3 The cost that is not computational: decision loss

A design choice has a monetary consequence that dwarfs the analysis cost. Suppose the extension has per-store-month contribution `c` if the effect is `tau`, and the decision is made on an estimate `tau_hat` with standard error `s`. Extending when the true effect is below the break-even `tau*` destroys value at rate `(tau* - tau)` per store-month; not extending when the true effect is above it forgoes `(tau - tau*)`.

```text
FORMULA [DID-13]: L_extend = n_ext * m * E[ max(0, tau* - tau) ]
                  L_hold   = n_ext * m * E[ max(0, tau - tau*) ]
WHERE:  n_ext = stores in the extension; m = months in the horizon;
        tau in outcome units per store-month; tau* the break-even effect;
        expectation over the posterior of tau given the estimate and its SE.
        Closed form for X ~ N(mu, s^2): E[max(0, X)] = mu*Phi(mu/s) + s*phi(mu/s).
CONSTRAINT: single decision, linear value in the effect, no option value from
        waiting, cost of reversal excluded, posterior approximated by
        N(tau_hat, s^2). The asymmetry of the two losses is why an estimate at
        the break-even is not a coin flip.
NUMERICAL EXAMPLE: tau_hat = 3.0, s = 11.9, tau* = 4.375, n_ext = 80, m = 12.
        Extend: X = tau* - tau ~ N(1.375, 11.9^2); mu/s = 0.1155;
        Phi(0.1155) = 0.5460, phi(0.1155) = 0.3963;
        E[max(0, X)] = 1.375 * 0.5460 + 11.9 * 0.3963
                     = 0.7508 + 4.7160 = 5.4668;
        L_extend = 80 * 12 * 5.4666 = 5,247.9 thousand currency units.
        Hold: X = tau - tau* ~ N(-1.375, 11.9^2); mu/s = -0.1155;
        Phi(-0.1155) = 0.4540;
        E[max(0, X)] = -1.375 * 0.4540 + 11.9 * 0.3963
                     = -0.6243 + 4.7160 = 4.0917;
        L_hold = 80 * 12 * 4.0916 = 3,927.9 thousand currency units.
INTERPRETATION: holding has the lower expected loss, 3,927.9 against 5,247.9,
        even though the interval on the effect extends well above break-even.
        The reason is that the estimate sits below the threshold and the
        distance to it, 1.375, is small relative to the standard error 11.9,
        so the distribution is nearly centred on break-even and the loss
        function is locally symmetric. Reporting "the point estimate is below
        break-even, therefore hold" would reach the same decision here by
        luck: the loss comparison is what makes it a decision rather than a
        default.
```

The check on the loss arithmetic: at `mu = 0` (an estimate exactly at break-even) both expressions reduce to `s * phi(0) = s * 0.3989`, so the two decisions are equally costly, as they must be by symmetry. Setting `s = 0` collapses both to `max(0, ...)` evaluated at the point and recovers the naive threshold rule. Those two limits are the sanity conditions, and a decision rule that fails either of them is mis-specified.

One further property is worth carrying into a review. The difference between the two losses is `n_ext * m * (tau_hat - tau*)` exactly, because `E[max(0,X)] - E[max(0,-X)] = E[X]`. With the numbers above: `80 * 12 * (3.0 - 4.375) = 960 * (-1.375) = -1,320`, and `3,927.9 - 5,247.9 = -1,320.0`. That identity is a free arithmetic check on any loss table, and it also shows why a decision based on the sign of `tau_hat - tau*` is *not* generally optimal: the sign of the difference in losses is determined by the expected effect, but the magnitude depends on `s`, so a second decision with a much smaller `s` and the same point estimate would have the same preferred action and a very different cost of being wrong.

### 9.4 The cost of a larger sample versus a better design

Increasing `n` reduces `s` proportionally to `1/sqrt(n)` and shrinks both loss terms only when the estimate is far from the threshold; near the threshold, both losses scale with `s`, so halving the standard error roughly halves the expected loss. Improving the design — a better comparison group, a longer pre-period, a matched panel — acts on the *bias* term, which does not shrink with `n` at all. A biased estimate of +12.5 that should be +3.0 does not become better with 10,000 stores; it becomes more precisely wrong, and the loss term scales with `n_ext * m` while the bias stays fixed.

The practical implication for an analysis budget is that the marginal currency unit is better spent on a second comparison group or a longer pre-period than on more computation. The compute cost of running the additional specification is measured in minutes; the design cost is measured in weeks of waiting for more pre-period data, and only the second changes the estimand's credibility.

**Takeaway:** the compute is trivial, the analyst's specification choices are expensive, and the decision loss dwarfs both.

## 10. Key Theorems / Results Reference

### 10.1 Identification and estimation results

| Result | Statement | Essential hypotheses | Practical implication | What it does not imply |
|---|---|---|---|---|
| DiD identification | Under consistency, no interference, and parallel trends for the untreated potential outcome, the 2x2 contrast equals the ATT. | Two groups; two periods; common treatment date; no anticipation; identical outcome measurement. | The estimator is unbiased for the ATT when the assumption holds; a point estimate is a statement about a counterfactual, not about observed data. | It says nothing about whether parallel trends holds — that is an assumption, not a conclusion. |
| Rosenbaum–Rubin balancing property | If `{Y(0)}` is independent of `W` given `X`, then it is independent of `W` given `e(X)`, and `P(W=1 | X) = E[W | e(X)]`. | Ignorability given `X`; overlap; `e(.)` is the true propensity. | Matching on a scalar is sufficient to remove confounding by `X`, which is why score matching is legitimate despite losing information. | It does not imply that matching on the *estimated* score is unbiased in finite samples, nor that a good classifier gives good balance. (SOURCED: Rosenbaum & Rubin, 1983; exact citation OPEN for verification.) |
| Abadie–Imbens bias bound | The bias of a matching estimator with `M` nearest neighbours is bounded by a function of the maximum matching distance and the smoothness of the conditional outcome; it does not vanish unless that distance shrinks. | Outcome regression Lipschitz or bounded derivative; matching performed on covariates or the true score; the number of matches grows slowly relative to `n`. | The bound justifies calipers and bias correction, and predicts that matches far away in covariate space are unbounded rather than merely noisy. | It does not supply a usable standard error for a given dataset, and it is not a licence to match outside common support. (SOURCED: Abadie & Imbens, 2006; exact citation OPEN.) |
| Doubly robust DiD (conditional parallel trends) | With either the propensity model or the outcome-change model correctly specified, the locally efficient estimator for the ATT under conditional parallel trends is consistent and asymptotically normal. | Conditional parallel trends given `X`; overlap; regularity on both nuisance models; cross-fitting or sample splitting to remove own-observation bias. | Model uncertainty about which nuisance is right is priced into the estimator rather than ignored. | It does not fix a violation of conditional parallel trends, and it does not remove the need for a pre-trend check. (SOURCED: Sant'Anna & Zhao, 2020; exact citation OPEN.) |
| Goodman-Bacon decomposition | With staggered adoption, the TWFE coefficient is a weighted average of all 2x2 DiD comparisons in the sample, with weights summing to one, and the "later-treated versus already-treated" comparisons have weights that can be negative. | Balanced panel; absorbing treatment; weights defined from group sizes and treatment-timing variance. | Any staggered TWFE coefficient can be decomposed and audited; the decomposition explains sign reversals without invoking sampling error. | It does not say the TWFE estimator is always biased — with a single treatment date or homogeneous effects it is fine. (SOURCED: Goodman-Bacon, 2021; exact citation OPEN.) |
| Callaway–Sant'Anna group-time ATT | Under parallel trends within each cohort against a valid comparison group, `ATT(g,t)` is identified and can be aggregated with explicit weights; the estimator is consistent and asymptotically normal. | Panel data; absorbing treatment; a comparison group untreated at both dates; no anticipation. | Provides the estimand the TWFE coefficient fails to be, and makes the aggregation choice explicit. | It does not rescue a design with no never- or not-yet-treated units, and the aggregation weights are the analyst's choice, not the data's. (SOURCED: Callaway & Sant'Anna, 2021; exact citation OPEN.) |
| Pre-trend test power | A joint test of pre-period coefficients is consistent against fixed alternatives as `n` grows, but its power against a linear drift of size `d` per period is governed by the standard error of the pre-period coefficients, not by the post-period effect size. | The event study is correctly specified; pre-period coefficients are estimated with the stated standard errors. | Passing the test is weak evidence for parallel trends; the test detects only large divergences. | It does not mean parallel trends is false when the test rejects — a rejection may be a level difference in one pre-period or a data artefact. |
| Placebo inference for synthetic control | Under the sharp null of no effect, reassigning treatment to each donor unit in turn generates a reference distribution for the treated unit's post/pre gap ratio; the treated unit's rank in that distribution is a finite-sample exact p-value under exchangeability. | Donor pool unaffected by the intervention; the same fitting procedure applied to each placebo; exchangeability of units. | Provides inference with one treated unit where conventional standard errors are unavailable. | It does not construct confidence intervals for the effect size, and its validity depends on the donor pool being genuinely untreated. (SOURCED: Abadie, Diamond & Hainmueller; exact citation OPEN.) |
| Few-cluster inference | Cluster-robust variance estimators are downward-biased when the number of clusters is small; the bias is of order `1/C` and the normal approximation deteriorates; wild cluster bootstrap with restricted null imposition improves size. | Independent clusters; treatment varying at the cluster level; the bootstrap uses the restricted null. | Standard errors computed from 8 treated regions are not trustworthy without a correction. | It does not help when clusters are not independent (spillovers across regions), which is a design problem rather than an inference problem. (SOURCED: Cameron, Gelbach & Miller; exact citation OPEN.) |

### 10.2 What each result licenses

The identification results license a *conditional* claim: given the assumption, the estimator equals the estimand. They never license an unconditional claim, which is why a report that says "DiD shows a 12.5-unit effect" without naming the assumption has said less than it appears to. The decomposition results license a *diagnostic* claim: given the panel, the coefficient is a specific weighted average that can be computed and inspected. They are the most useful results in the sheet because they convert an untestable assumption into a checkable arithmetic identity.

The bias results license a *design* claim: matching quality is a property of the distance distribution, not of the propensity model's fit, which is why calipers and balance tables are the operative diagnostics. And the power results license a *humility* claim: the tests commonly used to defend DiD designs have low power against exactly the violations that matter, so the honest report includes a bound computed from the observed pre-trend rather than a p-value from a joint test.

### 10.3 Results that are often misapplied

Three misapplications recur. First, citing the Rosenbaum–Rubin property to justify matching on an estimated propensity score without any balance check: the property is about the true score, and estimation error plus misspecification can leave substantial imbalance. Second, citing doubly robust DiD as a defence against a failed pre-trend: the estimator is robust to nuisance misspecification, not to violation of conditional parallel trends. Third, citing the Goodman-Bacon decomposition to conclude that "TWFE is invalid", when the correct conclusion is that TWFE estimates a particular weighted average whose weights must be inspected — with a single treatment date the decomposition is trivial and the estimator is the 2x2.

**Takeaway:** the theorems are conditional statements about estimators, not evidence about your design; they tell you what to check, not what to conclude.

## 11. Interview / Exam Questions

### Level 1 — Definition and recall

**Q-01 · L1 · Assumption statement.**
*Prompt.* A colleague reports "the DiD estimate is 12.5, so the programme worked". What is missing from that sentence, and state the assumption in one sentence.
*Answer.* The assumption. Difference-in-differences identifies the ATT under parallel trends: absent the intervention, treated and comparison units would have shown the same average change in the outcome between the pre and post windows. The estimate also needs a stated estimand (ATT for which population), a window, a standard error, and a statement that no coincident differential shock occurred. Without these the number is an arithmetic fact about four cell means, not a causal claim.
*Red flag.* Answering "we control for confounders" — difference-in-differences does not control for confounders in the regression sense; it removes fixed group differences and common shocks, and it is silent on differential shocks.
*Scoring.* 0: no assumption named. 1: "parallel trends" named without statement. 2: correct statement about changes, not levels. 3: adds estimand, window, and the differential-shock caveat.
*Support.* Sections 3.2 and 4.1.1.

**Q-02 · L1 · Distinguishing estimands.**
*Prompt.* Explain the difference between the ATT and the ATE, and say which one a DiD on a voluntary pilot estimates.
*Answer.* The ATT is the mean effect over the units actually treated; the ATE is the mean over the whole eligible population. A DiD on a voluntary pilot estimates the ATT for the pilot stores under parallel trends. Extending to all stores requires assuming effects do not differ systematically between volunteers and non-volunteers — usually false, because volunteers select on expected benefit. If the pilot stores were chosen for readiness, the ATT is an upper bound on the ATE only under a further monotonicity-style assumption, which should be stated rather than assumed.
*Red flag.* "They are the same when the sample is large." They coincide only under homogeneous effects, which is an assumption about the world, not about the sample.
*Scoring.* 0: conflates the two. 1: defines both but claims the pilot gives the ATE. 2: correct definitions and correct ATT answer. 3: adds the selection-on-gain argument.
*Support.* Sections 3.1, 2.4.

**Q-03 · L1 · Matching vocabulary.**
*Prompt.* What is the common support condition, and what happens to the estimand when it fails?
*Answer.* Common support (overlap) requires that for every covariate value with treated mass, the probability of treatment is strictly between zero and one, so a comparable untreated unit exists. When it fails, the counterfactual for the affected treated units is an extrapolation. If those units are dropped, the estimand changes from the effect on all treated units to the effect on the supported subset; if they are kept, the estimate's bias has no bound that shrinks with sample size. Either way the change must be stated.
*Red flag.* "We can extrapolate with a regression." A regression will extrapolate, but the extrapolation is an assumption about functional form, not a comparison.
*Scoring.* 0: no definition. 1: definition without the estimand consequence. 2: both. 3: adds that the bias does not shrink with `n`.
*Support.* Sections 3.2 (13), 4.4.5, 8.5.

### Level 2 — Derivation and application

**Q-04 · L2 · The 2x2 by hand.**
*Prompt.* Treated mean goes from 412.0 to 447.5; comparison from 398.0 to 421.0. Compute the effect, the pre-period gap, the post-period gap, and say how much the naive post-gap overstates the effect.
*Answer.* Treated change `35.5`; comparison change `23.0`; DiD `12.5`. Pre-gap `14.0`; post-gap `26.5`. The post-gap overstates by `26.5 - 12.5 = 14.0`, exactly the pre-period gap, because the gap-change identity is `DiD = Gap_post - Gap_pre`. The overstatement is not a rounding issue; it is the entire selection difference in levels.
*Red flag.* Reporting 26.5 and calling it the effect, or subtracting the pre-period *mean of the comparison group* from the treated post mean.
*Scoring.* 0: arithmetic error. 1: correct DiD, no gap analysis. 2: all four numbers. 3: identifies the identity and the selection interpretation.
*Support.* Section 4.1, 5.3.

**Q-05 · L2 · Standard error and decision.**
*Prompt.* Same data; change-score SDs are 48.0 (n=40) and 52.0 (n=80). Compute the standard error, the t-statistic, and the 95% interval. What do you tell the decision-maker?
*Answer.* `SE = sqrt(48^2/40 + 52^2/80) = sqrt(57.60 + 33.80) = sqrt(91.40) = 9.5603`. `t = 1.307`, `p ≈ 0.19`. Interval `12.5 ± 18.74 = [-6.24, 31.24]`. The honest message is that the design cannot distinguish the estimate from zero, that the interval includes economically meaningful losses and gains, and that the decision has to be made on the loss comparison rather than on the point estimate — or that the analysis needs more data.
*Red flag.* Reporting `p = 0.19` as "a trend toward significance", or widening the interval by an arbitrary factor to "be conservative".
*Scoring.* 0: wrong variance formula. 1: correct SE, wrong interval. 2: all three quantities. 3: connects to the decision rule and the loss asymmetry.
*Support.* Sections 4.1.4, 4.1.5, 9.3.

**Q-06 · L2 · Why the clustered standard error differs.**
*Prompt.* The same regression run on 480 treated and 960 comparison store-month rows gives a standard error of 1.49; run on 40 and 80 store-level change scores it gives 9.56. Explain the discrepancy and say which is right.
*Answer.* They are different analyses. The row-level model treats 1,440 rows as independent observations, which they are not: the treatment varies at store level, so the effective sample is 120 stores. With an intraclass correlation `rho`, the row-level variance is inflated by `DE = 1 + (m-1) rho`; at `m = 12`, `rho = 0.5`, `DE = 6.5` and the corrected standard error is `1.4916 * sqrt(6.5) = 3.80` on the row scale, which is not comparable to 9.56 without rescaling. The store-level change score is the analysis that matches the assignment level, so 9.56 is the one to report for the 2x2; the row-level model with clustering should reproduce it once the correlation structure is correctly specified and the outcome is the change, not the level.
*Red flag.* "The larger standard error is wrong because there is more data." The data are not more informative merely because they are more numerous.
*Scoring.* 0: no explanation. 1: mentions clustering. 2: correct design effect. 3: notes the analyses are different and reconciles them.
*Support.* Section 4.1.4.

**Q-07 · L2 · Decomposing a staggered coefficient.**
*Prompt.* Three groups of equal size, three periods, untreated path 100/102/104. Early cohort observed 100/102/124, late cohort 100/102/99, never-treated 100/102/104. What is the TWFE coefficient, what is the average cohort effect, and why do they differ?
*Answer.* The TWFE coefficient is -2.50; the unit-period-weighted average cohort effect is +5.0. The difference arises because the "late versus early" comparison uses already-treated units as controls: `(99 - 102) - (124 - 102) = -25`, which embeds the negative of the early cohort's continued ramp. That comparison is not an estimate of the late cohort's effect; it is an estimate of the difference between the late cohort's effect and the early cohort's change. So the coefficient is a weighted average of estimands, some of which are not effect levels.
*Red flag.* "The sample is too small" — the discrepancy is an identity, not sampling noise, and it persists at any sample size.
*Scoring.* 0: reports only one number. 1: correct TWFE. 2: both numbers. 3: explains the contaminating comparison and its estimand.
*Support.* Section 4.2.3.

**Q-08 · L2 · Minimum detectable effect.**
*Prompt.* With `SE = 9.56`, what is the MDE at 5% two-sided and 80% power, and how many stores per arm would you need to detect a 12.5 effect at the same power, assuming equal change-SD of 50.04?
*Answer.* `MDE = (1.9600 + 0.8416) * 9.56 = 2.8016 * 9.56 = 26.79`, or 6.5% of the treated baseline. For 12.5: `2504 * 2 / n = (12.5/2.8016)^2 = 19.9077`, so `n = 5008/19.9077 = 251.6`, i.e. 252 per arm. The existing design at 40 and 80 is roughly a sixth of the required size.
*Red flag.* Using `1.96 * SE` as the MDE — that gives the detection threshold at zero effect, not the effect size detectable with 80% power.
*Scoring.* 0: wrong formula. 1: correct MDE. 2: correct sizing. 3: notes the allocation-efficiency difference for unequal arms.
*Support.* Section 4.1.5.

### Level 3 — Judgement, design, and diagnosis

**Q-09 · L3 · Choosing between matching and DiD.**
*Prompt.* You have a store panel and a rollout that was chosen by regional directors. Would you use matching, DiD, or both? Under what condition would your answer change?
*Answer.* Matched DiD: match on pre-treatment size, region, format, and pre-period level, then compare changes within matched pairs. The reason is that matching handles composition and DiD handles time-invariant unobserved differences; neither alone handles both. The answer changes if (a) the directors chose stores on expected *trend* rather than level — then the matched DiD still fails and no estimator fixes it; (b) there is no untreated store in some regions, in which case the matched comparison is cross-regional and the estimand must be restricted; or (c) the outcome is measured only once after the intervention, in which case only a cross-sectional matching design is available and ignorability must carry the whole burden.
*Red flag.* "Use the one with the smaller standard error." The estimate with the smallest interval on an unidentified estimand is the least informative.
*Scoring.* 0: picks one without reason. 1: picks one with a partial reason. 2: matched DiD with justification. 3: adds the condition under which it fails.
*Support.* Sections 7.3, 2.5, 4.3.5.

**Q-10 · L3 · Diagnosing an ambiguous symptom.**
*Prompt.* An analyst reports a DiD of +12.5 with `t = 4.5` from the row-level model, and a reviewer reports `t = 1.3` from the store-level model. Both used the same extract. What are the competing explanations and what is your discriminating check?
*Answer.* Three candidates. (a) Clustering: the row-level model treats correlated rows as independent, so `t = 4.5` is spurious; the corrected standard error is larger. (b) Different windows: the row-level model may include months the store-level analysis excludes. (c) Different filters: one may have dropped stores with missing covariates. The discriminating check is to reconcile the two datasets row by row — row counts per store, per month, per group — and to recompute the point estimate from both. If the point estimates agree at 12.5 and only the standard errors differ, it is clustering. If the point estimates differ, it is a filter or window difference, and the standard errors are a distraction.
*Red flag.* Accepting the larger `t` because it is "the regression output", or splitting the difference.
*Scoring.* 0: no hypotheses. 1: one hypothesis. 2: clustering plus a check. 3: includes the point-estimate reconciliation as the first discriminator.
*Support.* Sections 8.6, 4.1.4.

**Q-11 · L3 · Post-treatment control.**
*Prompt.* The model improves when you add post-period staffing and inventory turns, and the coefficient falls from 12.5 to 6.0. A colleague argues the smaller number is "the real effect, net of operational execution". Evaluate.
*Answer.* The argument is wrong in general. Staffing and inventory turns are measured after the intervention and are plausibly affected by it: the cadence change may work *through* restocking frequency. Controlling for a mediator removes part of the effect and redefines the estimand as a direct effect under strong additional assumptions (no intermediate confounding, correct mediator model). If either variable is a collider — influenced by both treatment and outcome — the control induces bias rather than removing it. The correct action is to keep both out of the primary specification and, if the mechanism matters, to run a separate mediation analysis with its assumptions stated.
*Red flag.* "Controlling for more variables always reduces confounding." It controls for confounders, mediators, and colliders indiscriminately.
*Scoring.* 0: accepts 6.0. 1: suspects something but cannot say why. 2: identifies mediation. 3: adds collider risk and the estimand redefinition.
*Support.* Sections 8.4, 4.4.6.

**Q-12 · L3 · Pre-trend judgement.**
*Prompt.* The joint pre-trend test gives `p = 0.48`. The client wants to write "parallel trends is satisfied". What do you write instead, and what would you compute?
*Answer.* Write: "we did not detect a differential pre-trend (joint test p = 0.48); the design's identifying assumption is parallel trends, which is not directly testable in the post period." Then compute a bound: fit a linear pre-trend differential and extrapolate it over the post window, subtract it from the estimate, and report the adjusted estimate alongside the unadjusted one. With a differential of 0.4 per month and six post months, the bound removes 2.4 units from the 12.5, leaving 10.1, and the interval on that bound is wider than the original.
*Red flag.* Equating "test not rejected" with "assumption satisfied". A test with low power cannot establish an assumption.
*Scoring.* 0: writes the claim. 1: refuses without alternative. 2: correct wording. 3: adds the pre-trend extrapolation bound.
*Support.* Sections 4.4.1, 8.1, 5.4.

**Q-13 · L3 · Constraint change.**
*Prompt.* Halfway through the analysis, leadership announces that the rollout will be extended to all stores next month, so the comparison group disappears. What changes in your analysis and what becomes unanswerable?
*Answer.* The comparison group is gone from next month, so the post window ends at the announcement (or at the extension date). The analysis becomes a shorter-window DiD, which is a different estimand: the effect over the shorter horizon, not the steady-state effect. The dynamics question — whether the effect grows, decays, or plateaus — becomes unanswerable, and the event-study post coefficients beyond the truncation point cannot be estimated. The right response is to freeze the analysis window, document the truncation, and state that the long-run effect is not identified from this design. If some stores remain untreated by geography, they can serve as a partial comparison but the spillover question from 8.9 becomes central.
*Red flag.* Continuing to use later periods with a "comparison group" of stores that are about to be treated, which is the staggered-adoption trap in 4.2.3.
*Scoring.* 0: no change identified. 1: notes the window shortens. 2: states the estimand change. 3: adds the dynamics loss and a spillover-aware alternative.
*Support.* Sections 4.2.3, 8.9, 4.4.

**Q-14 · L3 · Verifying a claimed result.**
*Prompt.* A vendor sends a slide: "DiD shows +12.5, p < 0.05, matched sample, robust". What five things do you check before believing it?
*Answer.* (1) The estimand and population: which units, which window, ATT or ATE. (2) The pre-trend evidence: not the p-value but the coefficient path, with an extended pre-period if available. (3) The comparison group: how it was constructed, with a balance table on levels *and* on pre-period changes, plus the share of treated units discarded by the caliper. (4) The standard error's clustering level and the number of clusters. (5) The specification set: whether 12.5 is the pre-declared primary or the extreme of a search. Only after those five can the number be interpreted; "p < 0.05" answers none of them.
*Red flag.* Checking only the p-value and the sample size.
*Scoring.* 0: one check. 1: two or three. 2: four. 3: all five with the balance-on-changes point.
*Support.* Sections 5.4, 7.1, 8.3, 8.5, 8.6.

**Q-15 · L3 · Transfer to a new domain.**
*Prompt.* A hospital network rolls out a discharge-protocol change in six of its 30 hospitals, chosen because they had the lowest readmission rates. Readmission is measured at 30 days. What is the main threat and how would you address it?
*Answer.* Selection on the outcome: the six hospitals were chosen for having low readmission rates, which is a level difference, and parallel trends may still hold — but regression to the mean will pull their rates back toward the network average, producing an apparent increase in readmissions unrelated to the protocol. That is a differential trend in the untreated potential outcome, and it violates parallel trends. Address it by (a) matching on the pre-period level and the pre-period trend rather than the level alone, (b) using the other 24 hospitals as the comparison with a pre-period of at least 12 months at quarterly resolution, and (c) checking whether the six hospitals' pre-period trend was already reverting. If it was, the DiD will understate a beneficial protocol or overstate a harmful one depending on the direction of selection; the sign of the bias is `-trend_differential`, which is computable from the pre-period.
*Red flag.* "They were chosen on the outcome, so DiD is invalid." The level selection is exactly what DiD handles; the problem is the trend, and that is a separate question requiring a separate check.
*Scoring.* 0: says invalid. 1: identifies selection without diagnosing the mechanism. 2: identifies regression to the mean. 3: adds the matching-on-trend remedy and the direction of bias.
*Support.* Sections 3.2, 4.2.2, 4.3.5.

**Takeaway:** the questions that separate candidates are not about estimator names; they are about which assumption is doing the work, and what evidence would show it failing.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Constructing a counterfactual for a non-randomised rollout, using matching (who to compare) and difference-in-differences (what to compare about them) |
| Core formula | `DiD = (Y_T,post - Y_T,pre) - (Y_C,post - Y_C,pre)`. Equivalently the change in the cross-group gap. Valid only if the untreated *change* is the same in both groups |
| Key assumption | Parallel trends for the untreated potential outcome — about changes, not levels; untestable in the post period, and not established by a non-significant pre-trend test |
| Critical check | Event-study coefficients flat in the pre-period *and* a bound computed by extrapolating any observed pre-trend differential; balanced panel; balance table on pre-period changes, not only levels |
| Inference default | Cluster at the level treatment is assigned (store, not store-month); report the cluster count; use a wild bootstrap when clusters are fewer than about 40 |
| Complexity | Matching `O(n1 * n0 * P)` naive or `O((n1+n0) log n0)` with a tree in low dimensions; TWFE `O(N K^2)` after within-transformation; group-time `O(G T n)`. Compute is never the binding constraint |
| Use when | Panel data with a clean treatment date and a plausible donor group; cross-section with a strong pre-treatment covariate set and defensible ignorability |
| Avoid when | Groups chosen on expected trend, no untreated units in some strata, post-treatment covariates needed to match, staggered adoption with heterogeneous effects (use group-time ATTs), or a coincident differential shock |
| Diagnostic ladder | 1 reconcile row counts by group and period; 2 draw the four-cell square; 3 event study with an extended pre-period; 4 balanced-panel recomputation; 5 balance and caliper sensitivity; 6 clustered-standard-error reconciliation; 7 cohort decomposition if staggered |
| Go deeper | [04_Uplift_and_Treatment_Heterogeneity.md](04_Uplift_and_Treatment_Heterogeneity.md) for conditional effects and targeting; [05_Decision_Thresholds_and_Value.md](05_Decision_Thresholds_and_Value.md) for turning an estimate into a threshold decision. Regression discontinuity, instrumental variables, and synthetic control are named in Section 7.1 without derivation and remain an explicit gap in this collection |
