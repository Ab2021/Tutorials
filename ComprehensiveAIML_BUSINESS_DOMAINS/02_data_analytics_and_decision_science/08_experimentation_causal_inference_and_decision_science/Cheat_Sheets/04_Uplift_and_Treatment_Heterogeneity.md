---
collection: "08-experimentation-causal-inference-and-decision-science"
file_id: "04"
topic: "Uplift and Treatment Heterogeneity"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Estimator family, targeting metric, treatment cost, or experiment eligibility rule changes"
notation_style: "plain-text"
prerequisites: ["01_Experiment_Design_and_Guardrails.md", "02_Causal_Diagrams_and_Adjustment.md", "03_Matching_and_Difference_in_Differences.md", "binary outcome modelling", "conditional expectation"]
pattern_tags: ["experimentation", "causal-inference", "targeting"]
source_status: "OPEN"
owner_role: "applied scientist, growth or CRM analytics"
source_index: "INDEX.md"
---

# Uplift and Treatment Heterogeneity

## 1. File Header

> **Purpose:** Estimate *who* responds differently to a treatment, rank the population by that conditional effect, and convert the ranking into a costed targeting decision that survives an out-of-sample test. The sheet is about the difference between "does the treatment work" and "for whom does the treatment work", and about the fact that the second question is materially harder, noisier, and easier to get flatteringly wrong.
> **Prerequisites:** a randomized experiment with a clean control arm (file 01), a causal graph that justifies the adjustment set (file 02), and comfort with the identification logic of difference-in-differences and matching (file 03). On the modelling side: conditional expectation, logistic or gradient-boosted binary classifiers, and the sampling distribution of a difference in proportions.
> **Key outcome:** Given an experiment with an outcome, a treatment flag, and covariates, plus a treatment cost and an outcome value, you can (a) state the estimand and its identification conditions, (b) choose between a T-learner, S-learner, X-learner, R-learner, DR-learner, and a causal forest with reasons, (c) compute the Qini curve and its coefficient from a decile table with the arithmetic shown, (d) select a targeting depth from a break-even calculation rather than from the largest significant number, and (e) reject a submitted uplift analysis that has evaluated its ranking on the data that produced it.
> **This file does NOT cover:** experiment assignment mechanics and power for the main effect (file 01), covariate adjustment and backdoor reasoning (file 02), panel estimators (file 03), threshold selection for a single global score without effect heterogeneity (file 05), and production feature-store engineering for scoring pipelines. It also does not cover multi-treatment or continuous-treatment generalisations beyond a short note in 4.4.
> **Level:** 2 Core Methods
> **Version / reviewed:** 1.0 / not independently reviewed; `reviewed` stays null until a substantive review happens.
> **Evidence:** Identification results and estimator definitions are DERIVED from stated assumptions or attributed to published results (SOURCED, cited in Section 10). Every number in Sections 5 and 9 is ILLUSTRATIVE teaching arithmetic that has been recomputed by hand from the stated inputs; none of it is a company measurement. Claims about tooling defaults and metric variants are OPEN and flagged where they appear.
> **Retrieval cue:** If the question starts with "who should we target", "the lift is positive but the campaign lost money", "our uplift model looked great offline and did nothing in market", or "is the top decile real", start at Sections 8, 9, and 5 in that order.

### 1.1 The decision this sheet supports

A retention team can send an incentive to 100,000 eligible customers. The incentive costs 1.50 currency units per customer to deliver. A retained customer is worth 60 units of margin over the horizon of interest. A previous campaign had a positive average effect and a negative contribution, because it was sent to everyone. The team now has one experiment, one holdout, and a budget that permits treating a subset of the base. The question is not "what is the average effect" — that number already exists and it is not enough to fix the economics. The question is which subset, at what depth, and how much of the apparent advantage of the chosen subset is real.

Everything in this file is machinery for that question. The machinery has three parts that are often confused with each other: the estimand (a property of the world), the estimator (a procedure applied to finite data), and the decision rule (a comparison of expected value against cost). A submission that reports a Qini coefficient and stops has produced neither a decision rule nor a check on whether the ranking will hold.

| Layer | Question answered | Failure if skipped | Where treated here |
|---|---|---|---|
| Estimand | Which conditional contrast, in which population, over which window | You estimate the wrong quantity precisely | 3.1, 4.2 |
| Identification | What must be true for that contrast to be recoverable from observed data | The estimate is a description of selection, not of effect | 3.2, 8.2 |
| Estimation | Which learner, with what nuisance handling and sample splitting | Variance swamps the heterogeneity; ranking is noise | 4.3, 7.1 |
| Evaluation | How the ranking is scored out of sample | The winner's curse is reported as insight | 5.4, 8.1 |
| Decision | Which depth is profitable net of cost | Positive average effect, negative contribution | 4.2.5, 5.5 |

### 1.2 What "heterogeneity" is not

It is not a subgroup table. A subgroup table reports the effect in slices that someone chose in advance; a heterogeneity model ranks units on a score that was fitted to explain variation. The first is a small number of pre-specified contrasts with known multiplicity; the second is a high-dimensional object whose apparent quality is easy to inflate. It is also not interaction hunting in a regression: a significant interaction term establishes that the slope differs across a *specified* moderator, while uplift modelling asks for the best ordering of an entire population, including regions of covariate space where the experiment provides almost no information.

It is also not "personalisation". Personalisation changes the *content* of an intervention; uplift modelling changes *whether* and *whom*. A model that predicts who will convert is a response model; a model that predicts who converts *because of* the intervention is an uplift model. These two order the population differently, and the difference is the whole point of the file.

**Takeaway:** the deliverable is a costed, out-of-sample-validated ordering of units by conditional effect, not a chart of a coefficient.

## 2. Concept Map / Mental Model

### 2.1 The pipeline, with the gate that each stage must pass

```text
   [eligible population U, decision grain i, window T]
                     |
                     v
   [randomised or adjusted assignment W in {0,1}]
                     |
        +------------+-------------+
        |                          |
        v                          v
  [identification gate]      [fails] --> observational adjustment,
  ignorability? positivity?             stronger design, or refuse to
  SUTVA? consistency?                   report a CATE
        |
     passes
        v
   [nuisance fits: mu_0(x), mu_1(x), e(x)]  ---- cross-fitting ----+
        |                                                          |
        v                                                          |
   [CATE estimator tau_hat(x)]  <---- orthogonal score / residual ---+
        |
        v
   [rank all units by tau_hat]  --> deciles / percentiles / policy
        |
        v
   [out-of-sample evaluation: Qini, AUUC, policy value, calibration]
        |
        +--> [passes] --> [break-even depth k*: v*tau(x) > c_t(x)]
        |                        |
        |                        v
        |                 [rollout + holdout + guardrails]
        |                        |
        |                        v
        |                 [measured incremental value vs forecast]
        |
        +--> [fails] --> [shrink, re-specify, or report "no usable
                          heterogeneity" and fall back to ATE policy]
```

### 2.2 The four response classes

The mental model that makes uplift intuitive is a two-by-two on potential outcomes. Each unit has a pair `(Y(0), Y(1))`, one of which is observed.

```text
                       Y(1) = 1        Y(1) = 0
                    +--------------+--------------+
      Y(0) = 1      |  SURE THING  | SLEEPING DOG |
                    |  tau = 0     |  tau = -1    |
                    +--------------+--------------+
      Y(0) = 0      | PERSUADABLE  | LOST CAUSE   |
                    |  tau = +1    |  tau = 0     |
                    +--------------+--------------+
```

Persuadables are the only class that rewards treatment. Sure things convert anyway, so the treatment is a pure cost. Lost causes do not convert either way, so the treatment is also a pure cost. Sleeping dogs convert *less* when treated — the incentive reminds them to cancel, the discount trains them to wait, the notification creates a support contact. The classical marketing labels are due to the uplift-modelling literature; the structure is simply the sign of the individual effect crossed with the baseline level.

The important consequence: the *level* of conversion probability and the *sign of the effect* are close to orthogonal in real populations. High-baseline segments are often full of sure things. That single fact explains most failed targeting campaigns, and it is why Section 5 is built around a table where the two orderings disagree.

### 2.3 Level versus effect, drawn from the Section 5 table

```text
  conv. rate | d1    d2    d3    d4    d5    d6    d7    d8    d9    d10
  -----------+---------------------------------------------------------
    12%      |
    11%      |      T
    10%      |
     9%      |T                                   C+T
   8.3%      |                  T
     8%      |                                          C
     7%      |      C
   6.8%      |                                          T
   6.5%      |                                                      C
     6%      |            T     C
   5.9%      |                              T
     5%      |                              C
     4%      |C
   3.6%      |                        T
   3.5%      |                                                C     T
     3%      |            C
     2%      |                        C
   1.5%      |                                                T
  -----------+---------------------------------------------------------
    C = control conversion rate, T = treated conversion rate,
    plotted for each decile; where the two coincide the cell reads C+T.
```

Read the columns vertically, not the points diagonally. Decile 7 has the highest control rate in the table (9.0%) and the highest *pair* — both arms at 9.0%, so a 400-customer lift over 10,000 customers buys nothing. Decile 2 has the highest treated rate (11.0%) and a 4.0 pp effect. Decile 1 has a lower treated rate (9.0%, tying decile 7) but a larger effect (5.0 pp) because its control rate is only 4.0%. Decile 10 has the second-highest control rate (6.5%) and the worst effect (−3.0 pp).

So the two orderings genuinely disagree. Ranking by treated rate puts decile 2 first and decile 7 third; ranking by difference puts decile 1 first and decile 7 last. A team that reports "the top three segments by response" and a team that reports "the top three by uplift" will deploy to different customers at the same cost, and Section 5.8 prices the gap.

### 2.4 The three quantities that are constantly confused

| Symbol | Name | What varies | Observability |
|---|---|---|---|
| `tau_i` | individual treatment effect | unit | never observed; the fundamental problem |
| `tau(x)` | conditional average treatment effect | covariate vector | identified under assumptions, estimated with error |
| `ATE` | average treatment effect | nothing (a scalar) | identified under weaker assumptions, estimated precisely |

A model can rank `tau(x)` well while estimating the ATE badly, and vice versa. This is not a paradox: ranking only needs the ordering to be right, while the ATE needs the level to be right. It is why AUC-style metrics (rank-based) and Qini-style metrics (rank-based, weighted by effect) can look healthy while the reported average is biased, and why a program can be "statistically successful and financially negative".

### 2.5 Where the difficulty actually enters

Three sources dominate, in this order. First, the signal is a difference of two noisy quantities, so its variance is roughly double the variance of either component; heterogeneity of a couple of percentage points on a five-percent base rate is at the edge of what a normal-sized experiment can resolve. Second, the ranking is chosen using the same data that score it, which produces a positive measured Qini even when the true effect is constant everywhere. Third, the treatment effect is only identified inside the region where both arms have support, and a model will happily extrapolate a confident-looking ranking into regions where one arm is nearly empty.

**Takeaway:** the pipeline is identification, then estimation, then honest ranking, then a costed threshold — and each stage can fail while the previous one looked fine.

## 3. Foundational Definitions

Definitions are grouped by the layer they belong to. Each entry gives the formal object, the unit or type, the thing it is *not*, and the practical consequence of getting it wrong.

### 3.1 Estimands and potential outcomes

**1. Unit and decision grain.** The unit `i` is the entity to which the treatment is assigned and about whom the decision is made — a customer, a household, a store, a session. The decision grain must match the assignment grain; if the incentive is delivered to a household but the model scores customers, two customers in one household can receive contradictory actions. *Non-example:* a model trained on order-level rows applied to customer-level decisions, where a single customer contributes fifteen rows and therefore fifteen votes.

**2. Treatment indicator `W_i`.** `W_i = 1` if unit `i` received the intervention, `0` otherwise. It must record *delivery*, not *assignment*, when the two differ; an email that bounced is not treated. *Non-example:* an "assigned" flag that is set at queue time and never updated on send failure — this attenuates the estimated effect by exactly the non-delivery rate.

**3. Potential outcomes `Y_i(1)`, `Y_i(0)`.** The outcome unit `i` would show under treatment and under control respectively, over the declared window, measured the same way in both arms. They are properties of the unit and the experimental protocol, not of the data extract. *Non-example:* `Y(1)` defined on a 30-day window and `Y(0)` on a 14-day window because the campaign ended early.

**4. Observed outcome and consistency.** `Y_i = W_i * Y_i(1) + (1 - W_i) * Y_i(0)`. This identity is the consistency assumption, and it fails when one unit's treatment status affects another unit's outcome (interference) or when multiple versions of the treatment exist under one label. *Non-example:* three different discount depths all recorded as `W = 1`; the "effect" is then an average over an unrecorded mixture.

**5. Individual treatment effect (ITE).** `tau_i = Y_i(1) - Y_i(0)`, in outcome units. Never observed for any unit, because exactly one potential outcome is realised. *Non-example:* the difference between a treated converter and a control non-converter — that difference mixes the effect with the baseline.

**6. Average treatment effect (ATE).** `E[tau_i]` over the population that the experiment represents. A scalar in outcome units. Identified in a randomised trial by a difference in means under randomisation, consistency, and no interference. *Non-example:* the difference in means computed on a self-selected subset of the experiment.

**7. Conditional average treatment effect (CATE).** `tau(x) = E[Y(1) - Y(0) | X = x]`. A function of covariates, in outcome units. This is the object every uplift model estimates, whether or not its documentation says so. *Non-example:* a fitted value from a model of `Y` on `(X, W)` evaluated at `W = 1` and `W = 0` when the model contains no interaction between `W` and `X`; that difference is a constant, not a CATE.

**8. Uplift.** In marketing usage, the CATE on a binary outcome, reported in percentage points of conversion. It is not a distinct statistical object; it is `100 * tau(x)` when `Y` is an indicator. *Non-example:* "uplift" used to mean the treated conversion rate, which is a level, not a difference.

**9. ATT and ATC.** `ATT = E[tau | W = 1]` and `ATC = E[tau | W = 0]`. They differ from the ATE whenever effects are heterogeneous and assignment is not random. A targeting decision usually wants a conditional effect on the *eligible* base, which is neither of these unless assignment was random.

**10. Policy and policy value.** A policy `pi` maps covariates to an action in `{0, 1}`. Its value is `V(pi) = E[Y(pi(X))]`, the outcome the population would show if the policy decided everyone's treatment. The decision problem is to choose `pi` maximising `V(pi)` net of cost, not to estimate `tau(x)` for its own sake.

**11. Regret.** `V(pi*) - V(pi_hat)`, the value lost relative to the best policy in the considered class. Reported in outcome units or currency. It is the honest way to describe a targeting model's worth, and it is bounded above by the quality of the class, not by the model's fit.

### 3.2 Identification objects

**12. Ignorability (unconfoundedness).** `{Y(0), Y(1)}` independent of `W` given `X`. In a randomised experiment it holds unconditionally; in observational data it is an assumption that no unmeasured variable drives both assignment and outcome. *Non-example:* a "control group" assembled from customers who were eligible but not contacted by a sales team, where the team's choice of whom to skip is itself informative.

**13. Positivity (overlap).** For every `x` in the target population, `0 < e(x) < 1`. Without it, some units have no comparable counterpart and the CATE at those `x` is not identified. *Non-example:* a platinum tier that is always treated; the model will still emit a score there, derived entirely from extrapolation.

**14. Propensity score `e(x)`.** `e(x) = P(W = 1 | X = x)`. In a balanced experiment it is a constant (the allocation ratio); in observational data it is the gatekeeper of comparability. Its *distribution*, not its mean, is the useful diagnostic: mass near 0 or 1 is where CATE estimates become unstable.

**15. Outcome regressions `mu_1(x)`, `mu_0(x)`.** `mu_w(x) = E[Y | X = x, W = w]`. These are nuisance functions: they are needed to estimate `tau(x)` but are not themselves the target. A model can fit them badly in absolute terms and still deliver a usable `tau` if the errors are similar across arms.

**16. Nuisance function and orthogonal score.** A nuisance is any function estimated only to be plugged into a target estimand. A score is orthogonal (Neyman-orthogonal) when its derivative with respect to the nuisance is zero at the truth, so small nuisance errors move the target estimate only to second order. This is what makes doubly robust and residualised estimators more stable than naive plug-ins.

**17. Cross-fitting.** Estimating nuisances on one part of the data and evaluating the target score on another, then rotating. It removes the own-observation bias that makes in-sample pseudo-outcomes look better than they are. *Non-example:* fitting `mu_1`, `mu_0`, and `e` on all rows and then computing the AIPW score on the same rows.

### 3.3 Estimators

**18. Meta-learner.** Any procedure that reduces CATE estimation to one or more standard supervised fits. The family includes S-, T-, X-, R-, and DR-learners. The name describes the wrapper, not the base model.

**19. S-learner.** One model of `Y` on `(X, W)`, with the CATE read off as `f(x, 1) - f(x, 0)`. Cheap, statistically efficient when the effect is small relative to the baseline, and structurally unable to find heterogeneity if the learner regularises the treatment interaction away.

**20. T-learner.** Two separate models, one per arm, CATE as the difference of predictions. Simple and robust to effect/baseline scale differences, but each model sees only half the data, and the two error surfaces are independent, so the difference inherits both variances.

**21. X-learner.** Two outcome models, then imputed individual effects in each arm using the *other* arm's model, then a weighted combination of the two effect models with weights from `e(x)`. Designed for unbalanced arms and for cases where one arm is much larger.

**22. R-learner.** Residualises `Y` on `X` and `W` on `X`, then regresses the outcome residual on the treatment residual times a candidate `tau(x)`. Built on the Robinson decomposition, which is the cleanest algebraic statement of why partialling out helps.

**23. DR-learner.** Constructs a doubly robust pseudo-outcome for each unit (AIPW form) and regresses it on `X`. Unbiased for `tau(x)` if *either* the propensity model *or* the outcome models are correct, hence "doubly robust".

**24. Causal forest.** An ensemble of trees whose splits maximise heterogeneity in the estimated effect rather than accuracy in the outcome, with honesty (separate samples for split selection and leaf estimation). Provides an asymptotic normality result under regularity conditions, which is what makes confidence intervals on `tau(x)` possible at all.

**25. Transformed outcome / class transformation.** A single supervised problem constructed so that its conditional expectation is an affine function of `tau(x)`. Elegant, requires known or estimated propensity, and breaks when the propensity varies strongly with `X`.

**26. CATE calibration.** The property that units with predicted effect `t` have an average realised effect near `t`. Distinct from ranking quality: a score can order units perfectly and be off by a factor of two in level. Checked by binning predicted effect and comparing with an out-of-sample doubly robust estimate per bin.

### 3.4 Evaluation objects

**27. Uplift (Qini) curve.** For each population fraction `p` from 0 to 1, the cumulative incremental outcome obtained by treating the top fraction `p` according to the score: `G(p)`. Units are conversions in Section 5, currency where a value is attached.

**28. AUUC and Qini coefficient.** AUUC is the area under `G(p)`. The Qini coefficient is the area between `G(p)` and the random-targeting line `p * G(1)`. Definitions are not standardised across libraries — some weight by group sizes, some normalise by the perfect-model area — so a number is meaningless without the definition and the population size attached. *Evidence:* SOURCED that variants differ (Section 10, R-12); the specific arithmetic used here is DERIVED and stated explicitly in 5.4.

**29. Uplift by decile.** A ten-row table of treated rate, control rate, difference, and incremental count per decile. The most useful diagnostic in the file, because it exposes non-monotonicity, sign flips, and the deciles where the ranking is guessing.

**30. Permutation null.** The distribution of the Qini coefficient when the score is randomly permuted across units, or when the experiment is re-analysed with a synthetic constant effect. A ranking whose Qini is inside this band has demonstrated nothing.

**31. Reliability (shrinkage) coefficient.** In a normal model where the estimated effect equals the true effect plus independent noise, `rho = Var(tau) / (Var(tau) + Var(noise))`. It is the factor by which a selected extreme estimate should be shrunk toward the mean; the top decile of a noisy score is biased upward by roughly `(1 - rho)` times its distance from the mean.

### 3.5 Decision objects

**32. Targeting depth `k`.** The fraction (or count) of the eligible base that receives the treatment under the policy. Chosen by economics, not by where the Qini curve happens to peak.

**33. Break-even uplift.** `u* = c_t / v`, where `c_t` is treatment cost per unit and `v` is the value of one outcome unit. Units with `tau(x) < u*` destroy value when treated, even if `tau(x) > 0`. In Section 5, `1.50 / 60 = 0.025`, i.e. 2.5 percentage points.

**34. Incremental value per treated unit.** `v * tau(x) - c_t(x)`. Summing this over the targeted set gives the program's expected contribution; it is the quantity the business actually experiences. A significance test on `tau` is not a test on this quantity.

**35. Guardrail.** A secondary metric that is allowed to veto a rollout even when the primary target improves — complaint rate, unsubscribe rate, support contact rate, margin per order. Defined before the analysis, with a threshold, because a post-hoc guardrail is a rationalisation.

**36. Do-no-harm holdout.** A small random sample withheld from the treatment *after* the policy is deployed, used to measure the realised incremental effect of the deployed policy rather than of the original experiment. Without it, the rollout has no counterfactual and the forecast is unfalsifiable.

**Takeaway:** keep the estimand, the estimator, the evaluation statistic, and the decision threshold as four separate nouns; conflating them is the origin of most uplift reporting failures.

## 4. Core Formulas with Full Derivation

### 4.1 Base results

#### 4.1.1 The average effect in a randomised two-arm trial

Validity before use: randomisation with a known allocation ratio, consistency, no interference between units, and outcomes measured identically in both arms over the same window. If any of these fails, the expression below estimates something that is not the ATE.

Derivation. By consistency, `Y_i = W_i Y_i(1) + (1 - W_i) Y_i(0)`. Taking expectations within arm and using independence of `W` from the potential-outcome pair:

```text
E[Y | W = 1] = E[W Y(1) + (1 - W) Y(0) | W = 1] = E[Y(1) | W = 1] = E[Y(1)]
E[Y | W = 0] = E[Y(0) | W = 0] = E[Y(0)]
```

so the difference of arm means equals `E[Y(1)] - E[Y(0)] = E[tau_i]`. The second equality in each line is exactly where randomisation is used: it licenses dropping the conditioning on `W`.

```text
FORMULA [U-01]: ATE_hat = mean(Y | W = 1) - mean(Y | W = 0)
WHERE: Y is the outcome in its natural unit (indicator, revenue, minutes);
       W in {0,1} is delivered treatment; the means are over the analysis
       population after the declared exclusions.
CONSTRAINT: randomisation, consistency, no interference, identical measurement
       windows, no differential attrition correlated with the outcome.
NUMERICAL EXAMPLE: treated 3230 conversions in 50000, control 2700 in 50000;
       ATE_hat = 0.0646 - 0.0540 = 0.0106 = +1.06 percentage points.
INTERPRETATION: the campaign is worth about 1.06 conversions per 100 treated
       customers on average; this says nothing about whether any subgroup
       responds more or less.
```

Boundary/counterexample: if 20% of "treated" units never received the message but are counted as treated, the estimate becomes roughly `0.8 * tau` under one-sided non-compliance, and the reported average understates the effect on compliers by the same factor while saying nothing about the effect on never-takers.

#### 4.1.2 Identification of the CATE and the bias in the naive contrast

Validity before use: conditional ignorability `{Y(0), Y(1)} ⊥ W | X`, positivity `0 < e(x) < 1`, consistency, and SUTVA. The derivation shows precisely which term vanishes under these assumptions, which is more useful than memorising the assumption list.

Derivation. Write the observed contrast at a covariate value and split the observed outcome into the potential outcome plus the effect:

```text
E[Y | X=x, W=1] - E[Y | X=x, W=0]
  = E[Y(1) | X=x, W=1] - E[Y(0) | X=x, W=0]
  = { E[Y(1) | X=x, W=1] - E[Y(0) | X=x, W=1] }     <-- effect on the treated at x
    + { E[Y(0) | X=x, W=1] - E[Y(0) | X=x, W=0] }   <-- selection bias b(x)
```

Under conditional ignorability the second brace is zero, so the observed contrast equals `tau(x)`. Under randomisation it is zero even without conditioning on `X`. Nothing else makes it zero.

```text
FORMULA [U-02]: tau(x) = mu_1(x) - mu_0(x),  with mu_w(x) = E[Y | X=x, W=w]
WHERE: tau in outcome units; x is a covariate vector in the support of the
       target population; mu_w are conditional means within arm.
CONSTRAINT: conditional ignorability, positivity, consistency, SUTVA. The
       equality is an identification statement, not an estimation recipe.
NUMERICAL EXAMPLE: at a covariate profile with mu_1 = 0.090 and mu_0 = 0.040,
       tau(x) = 0.050, i.e. +5.0 percentage points for that profile.
INTERPRETATION: the effect is a function of x; reporting only its average
       discards the information the targeting decision needs.
```

Boundary/counterexample: `b(x)` is not a nuisance that averages out. If motivated customers are both more likely to be selected for the incentive and more likely to convert anyway, `b(x) > 0` and every CATE is overstated by roughly the size of that motivational gap — in the illustrative arithmetic of Section 5, a 0.5 percentage-point gap would turn a 5.0 point estimate into a 4.5 point one.

#### 4.1.3 Why a marginal difference is not an average of CATEs you can invert

The ATE is a weighted average of `tau(x)`, but the weights are the covariate distribution, not the treatment-assignment distribution. The two coincide only under randomisation. This matters because a common shortcut — estimating `tau` in a subgroup by differencing arm means inside that subgroup — is valid only if randomisation holds *within* the subgroup, which it does in a factorial or stratified design and does not in an observational one.

```text
FORMULA [U-03]: ATE = sum_x tau(x) * P(X = x)
WHERE: the sum runs over the covariate cells of the target population; P(X=x)
       is the population share, not the treated share.
CONSTRAINT: the same `tau(x)` must be identified on the target population;
       weights come from the deployment base, not from the experiment sample.
NUMERICAL EXAMPLE: ten equally sized deciles with effects
       5.0, 4.0, 3.0, 2.3, 1.6, 0.9, 0.0, -1.2, -2.0, -3.0 (in pp)
       give ATE = 10.6 / 10 = 1.06 pp.
INTERPRETATION: a positive average can hide a large negative mass; the ten
       numbers above sum to a positive total while three of them harm.
```

Boundary/counterexample: re-weighting the ten effects by the *experiment's* arm sizes rather than by the deployment population is the standard population-shift error; it produces a number that is correct for the trial and wrong for the rollout.

### 4.2 General results

#### 4.2.1 The Robinson decomposition and the R-learner

Validity before use: `e(x) = P(W=1|X=x)` bounded away from 0 and 1, `m(x) = E[Y|X=x]` integrable, and the linear-in-`W` representation `E[Y|X=x,W] = mu_0(x) + W * tau(x)`, which follows from consistency plus the definition of `tau`.

Derivation.

```text
mu_0(x) + W*tau(x) = mu_0(x) + e(x)*tau(x) + (W - e(x))*tau(x)
                   = m(x) + (W - e(x))*tau(x)
```

because `m(x) = mu_0(x) + e(x) tau(x)` by the tower property. Therefore

```text
Y - m(x) = (W - e(x)) * tau(x) + noise,   E[noise | X, W] = 0
```

which is a regression of the outcome residual on the *product* of the treatment residual and the candidate effect function. It can be solved by any weighted least-squares or boosting procedure treating `(W - e(x))` as a feature multiplier.

```text
FORMULA [U-04]: minimise over tau in F  sum_i [ (Y_i - m_hat(X_i))
                - (W_i - e_hat(X_i)) * tau(X_i) ]^2
WHERE: m_hat and e_hat are nuisance fits from cross-fitting folds; F is the
       function class of the effect model (linear, tree ensemble, spline).
CONSTRAINT: overlap, correct specification of at least the product structure,
       nuisance estimation at o(n^{-1/4}) rate for the quasi-oracle property.
NUMERICAL EXAMPLE: one unit with Y = 1, m_hat = 0.054, W = 1, e_hat = 0.5:
       outcome residual = 0.946; treatment residual = 0.5; implied tau
       contribution = 0.946 / 0.5 = 1.892 for that single observation, which
       is only meaningful after averaging over many units with the same x.
INTERPRETATION: the residualised form removes the part of Y explained by
       baseline risk, so the effect model is not forced to re-learn the level.
```

Boundary/counterexample: if `e_hat` is badly estimated the treatment residual is wrong and the objective is misspecified; with a constant `e_hat = 0.5` in a trial this is harmless, but in an observational study a mis-specified propensity reintroduces bias of the same order as the confounding it was meant to remove.

#### 4.2.2 The doubly robust pseudo-outcome

Validity before use: positivity, consistency, and either `e(x)` or the pair `(mu_1, mu_0)` correctly specified. The estimator is unbiased if *either* nuisance block is right, which is the meaning of "doubly".

Derivation. Consider the candidate

```text
psi_i = mu_1(X_i) - mu_0(X_i)
      + W_i (Y_i - mu_1(X_i)) / e(X_i)
      - (1 - W_i) (Y_i - mu_0(X_i)) / (1 - e(X_i))
```

and take its conditional expectation given `X = x`. The first term is deterministic, equal to `mu_1_hat(x) - mu_0_hat(x)`. For the second, using `E[W | X=x] = e(x)` and `E[W Y | X=x] = e(x) mu_1(x)`:

```text
E[ W (Y - mu_1_hat(x)) / e(x) | X = x ]
   = (1/e(x)) * ( E[W Y | X=x] - mu_1_hat(x) E[W | X=x] )
   = (1/e(x)) * ( e(x) mu_1(x) - e(x) mu_1_hat(x) )
   = mu_1(x) - mu_1_hat(x)
```

so the residual term contributes exactly the error of the treated-arm outcome model — zero when that model is correct, and otherwise a term that cancels the corresponding error in the deterministic part. The control side is the mirror image:

```text
E[ -(1-W)(Y - mu_0_hat(x))/(1-e(x)) | X = x ] = -( mu_0(x) - mu_0_hat(x) )
```

Summing the three pieces gives `mu_1(x) - mu_0(x) = tau(x)` identically, whatever the values of `mu_1_hat` and `mu_0_hat`. Only `e(x)` must be correct for that cancellation to be exact; conversely, if `e` is wrong but both outcome models are right, the residual terms are zero and `psi` still equals `tau(x)`. That symmetry is double robustness.

```text
FORMULA [U-05]: psi_i as above; tau_hat(x) = regression of psi on X
WHERE: e(X) in (0,1); mu_w are arm-specific outcome predictions; psi is in
       outcome units and can fall outside [0,1] for binary Y.
CONSTRAINT: cross-fitting for the nuisances; if e is estimated, its errors
       must be small relative to the residual variance; extreme e inflates
       the variance even when the mean is right.
NUMERICAL EXAMPLE: treated unit with Y=1, mu_1=0.10, mu_0=0.04, e=0.5:
       psi = 0.06 + 1*(0.90)/0.5 - 0 = 0.06 + 1.80 = 1.86.
       Control unit with Y=0, mu_1=0.10, mu_0=0.04, e=0.5:
       psi = 0.06 + 0 - 1*(0.04)/0.5 = 0.06 - 0.08 = -0.02.
INTERPRETATION: individual pseudo-outcomes are extremely noisy (1.86 for one
       unit), which is why they must be smoothed over X rather than averaged
       naively or read one at a time.
```

Boundary/counterexample: with `e = 0.02` the treated pseudo-outcome contribution is multiplied by 50; a single mislabelled unit then moves the local estimate by more than the effect being estimated. This is the mechanism behind "the DR-learner exploded in the low-propensity region".

#### 4.2.3 The class transformation for binary outcomes

Validity before use: binary outcome, binary treatment, and a propensity that is constant across `X` (or known and used in the generalised form). The simple version below requires `e(x) = p` for all `x`, which is what a balanced randomised trial provides.

Derivation. Define the transformed label

```text
Z = 1 if (W = 1 and Y = 1) or (W = 0 and Y = 0), else Z = 0
```

Then, using independence of `W` from `(Y(0), Y(1))` and `P(W=1) = p`:

```text
P(Z=1 | X=x) = p * mu_1(x) + (1-p) * (1 - mu_0(x))
```

Set `p = 1/2` and substitute `mu_1 = mu_0 + tau`:

```text
P(Z=1 | X=x) = 0.5*(mu_0 + tau) + 0.5*(1 - mu_0) = 0.5 + tau(x)/2
```

so `tau(x) = 2 P(Z=1 | X=x) - 1`.

```text
FORMULA [U-06]: tau(x) = 2 * g(x) - 1, where g(x) = P(Z=1 | X=x)
WHERE: Z is the transformed binary label; g is fitted by any probabilistic
       classifier; tau is on the probability scale.
CONSTRAINT: binary Y, binary W, constant propensity p = 0.5. For p != 0.5 the
       affine map changes and must be re-derived rather than reused.
NUMERICAL EXAMPLE: at a profile with mu_1=0.09 and mu_0=0.04,
       g = 0.5*0.09 + 0.5*0.96 = 0.525, so tau = 2*0.525 - 1 = 0.05.
INTERPRETATION: one classifier replaces two, so the effect estimate is
       available from any library that fits a binary classifier.
```

Boundary/counterexample: the transformation silently assumes the propensity does not vary with `X`. In an observational study with `e(x)` ranging over 0.05 to 0.95, using the constant-`p` formula mis-scales `tau` and can flip its sign in regions where `e(x)` departs from `p`.

#### 4.2.4 The Qini curve and its area

Validity before use: a score that orders units, an outcome, and an experiment from which both arm rates are estimable *within* each score bin. The curve is a property of the score on a specific population; it is not transportable across populations without re-estimation.

Derivation. Partition the population into `K` bins ordered by decreasing score, with bin `k` containing `n_k` units. Within a bin the incremental outcome per unit is estimated as the difference in arm means, `d_k`. The incremental outcome from treating the first `k` bins is `G_k = sum_{j<=k} n_j d_j`. Normalising the population axis gives a curve through `(0, 0)` and `(1, G_K)`. Random targeting would deliver proportional gain, so its curve is the straight line from `(0,0)` to `(1, G_K)`. The area between the two curves is the Qini coefficient.

```text
FORMULA [U-07]: Q = sum_{k=1..K} (p_k - p_{k-1}) * (G_k - p_k * G_K)
WHERE: p_k = cumulative population fraction through bin k; G_k cumulative
       incremental outcome; G_K total incremental outcome; Q in
       (outcome units) x (population fraction).
CONSTRAINT: bins must be ordered by the score and computed on data not used
       to fit the score; equal-width bins in population, not in score value.
NUMERICAL EXAMPLE: G = (0, 500, 900, 1200, 1430, 1590, 1680, 1680, 1560,
       1360, 1060) at p = 0, .1, ..., 1.0; trapezoid area = 1243.0; random
       area = 1060/2 = 530.0; Q = 1243.0 - 530.0 = 713.0 conversions.
INTERPRETATION: targeting by this score captures 713 more conversions than
       random targeting of the same total volume, on this population.
```

Boundary/counterexample: if `G_K < 0` the random line slopes downward and a model that concentrates harm early scores *positively* by the same formula; always report the sign of `G_K` next to `Q`.

#### 4.2.5 The cost-aware treatment rule

Validity before use: a per-unit treatment cost that is known or estimable, a value per outcome unit, and a CATE estimate with quantified error. The rule is optimal only within the class of rules that treat on `tau(x)` alone; if cost varies with `x`, it must enter the rule explicitly.

Derivation. Treating unit `x` yields expected value `v * tau(x)` and costs `c_t(x)`, so the net contribution is `v tau(x) - c_t(x)`. Treating is preferred whenever this is positive. Dividing by `v > 0` gives the threshold form.

```text
FORMULA [U-08]: treat iff tau(x) > c_t(x) / v
WHERE: v is margin per outcome unit in currency; c_t is delivery cost per
       treated unit in the same currency; tau on the probability scale.
CONSTRAINT: v and c_t constant over the decision horizon; no capacity
       constraint; the CATE estimate is unbiased enough that the sign of
       tau(x) - u* is right more often than chance.
NUMERICAL EXAMPLE: c_t = 1.50, v = 60.00 gives u* = 0.025. Decile 3 has
       tau = 0.030 > 0.025, so it is treated; decile 4 has tau = 0.023 <
       0.025 and is not, despite being statistically distinguishable from zero.
INTERPRETATION: statistical significance is not the decision boundary; the
       cost-to-value ratio is.
```

Boundary/counterexample: with a capacity constraint the rule becomes a knapsack — rank by `v tau(x) - c_t(x)` per unit of scarce capacity, not by `tau(x)`. With a fixed budget rather than a fixed population, the threshold becomes the shadow price of the budget and must be solved jointly.

### 4.3 Variants

#### 4.3.1 S-learner versus T-learner: the variance argument

Validity before use: both learners require a correctly specified function class, and the comparison below assumes independent arm samples of size `n/2` each, as in a balanced trial.

Derivation. A T-learner estimates `tau_hat = mu_1_hat - mu_0_hat` where the two hats are fitted on disjoint samples. If each has variance `sigma^2 / (n/2)` at a given `x` (the usual rate for a smooth conditional mean), then `Var(tau_hat) = 2 sigma^2 / (n/2) = 4 sigma^2 / n`. An S-learner with a correctly specified shared structure concentrates its variance on the level and can estimate a small effect at variance close to `sigma^2 / n` per unit of information, but only if the interaction is not shrunk away. The trade is bias (S-learner misses heterogeneity when regularised) against variance (T-learner doubles the error surface).

```text
FORMULA [U-09]: Var(tau_hat_T) ~= 4 sigma^2 / n   vs   Var(tau_hat_S) ~= sigma^2 / n
WHERE: n total sample size, sigma^2 conditional outcome variance at x,
       balanced arms, equally smooth learners.
CONSTRAINT: the S-learner bound requires that the interaction is estimable
       without regularisation removing it; the T-learner bound assumes the
       two fits are independent, true for disjoint samples.
NUMERICAL EXAMPLE: sigma^2 = 0.05, n = 100000: T-learner SE = sqrt(4*0.05/1e5)
       = sqrt(2e-6) = 0.00141 = 0.14 pp; S-learner SE = 0.00071 = 0.07 pp.
INTERPRETATION: at a 1 pp effect, the T-learner's noise is 14% of the signal
       and the S-learner's is 7%; the T-learner needs about four times the
       sample to match the S-learner's precision on a constant effect.
```

Boundary/counterexample: the S-learner's advantage disappears entirely when the base learner cannot represent the interaction — a regularised linear model with a single additive treatment term produces a rank-degenerate CATE (everyone identical), which some evaluation scripts will report as a Qini of exactly zero and others as an error.

#### 4.3.2 The X-learner and unbalanced arms

Validity before use: overlap, and at least one arm large enough to fit a stable outcome model. The construction is designed for the case where, say, 90% of units are treated and 10% are controls.

Derivation. Fit `mu_1` on treated units and `mu_0` on controls. Impute individual effects in each arm using the opposite model:

```text
D1_i = Y_i - mu_0_hat(X_i)   for treated i     (observed minus counterfactual)
D0_i = mu_1_hat(X_i) - Y_i   for control i
```

Then fit `tau_1` to `D1` on the treated sample and `tau_0` to `D0` on the control sample, and combine with weights `g(x)` that favour the better-supported side: `tau_hat(x) = g(x) tau_1_hat(x) + (1 - g(x)) tau_0_hat(x)`, typically with `g = e_hat`.

```text
FORMULA [U-10]: tau_hat(x) = g(x) tau_1_hat(x) + (1-g(x)) tau_0_hat(x)
WHERE: D1, D0 are imputed effects in outcome units; g in [0,1] is a weight
       function, commonly the estimated propensity.
CONSTRAINT: overlap; the imputation is only as good as the opposite-arm model;
       with a tiny minority arm the minority imputations dominate the noise.
NUMERICAL EXAMPLE: treated unit with Y=1 and mu_0_hat=0.04 gives D1=0.96;
       with only 5000 treated units the regression of D1 on X has variance
       driven by the 0/1 spread of Y, so the effect model is fit to a
       near-binary target and needs strong smoothing.
INTERPRETATION: X-learner buys stability in unbalanced designs at the cost of
       an extra modelling layer and a weight choice that must be defended.
```

Boundary/counterexample: if the propensity is used as `g` and is near 1 everywhere, `tau_hat` collapses to `tau_1_hat` and the control-side information is discarded; this is the intended behaviour only when the control arm genuinely cannot support estimation.

#### 4.3.3 Split criteria in uplift trees

Validity before use: node-level arm rates estimable (both arms present in the node), and a splitting objective that targets the difference rather than the level.

Derivation. For a candidate split into children `L` and `R`, the standard significance-based criterion compares the parent's effect with the weighted child effects:

```text
Delta = (n_L/n) * (p1_L - p0_L)^2 + (n_R/n) * (p1_R - p0_R)^2
      - (p1 - p0)^2
```

evaluated with a variance normalisation so that nodes with few observations do not dominate. Maximising `Delta` prefers splits that separate high-effect from low-effect regions, even when the two regions have identical baseline rates — which is exactly what a level-based criterion such as Gini gain will miss.

```text
FORMULA [U-11]: Delta as above, with a variance penalty
                Delta / sqrt(Var_hat(Delta))
WHERE: p1, p0 are within-node treated and control rates; n counts units;
       the penalty uses the binomial variance of each difference.
CONSTRAINT: both arms must be represented in every candidate child; splits
       producing an empty arm are inadmissible rather than infinite-valued.
NUMERICAL EXAMPLE: parent with p1=0.0609, p0=0.0490, effect 0.0119;
       L with p1=0.09, p0=0.04, effect 0.05, n_L/n=0.5;
       R with p1=0.032, p0=0.058, effect -0.026, n_R/n=0.5.
       Delta = 0.5*(0.05)^2 + 0.5*(0.026)^2 - (0.0119)^2
             = 0.00125 + 0.000338 - 0.000142 = 0.001446.
INTERPRETATION: the split is rewarded for pushing half the population to a
       negative effect, which a level-based criterion would not detect.
```

Boundary/counterexample: without a variance penalty and without honesty, a deep tree reaches leaves with one treated unit and one control unit and reports enormous effects; the Delta value grows with node count and the model memorises the experiment.

#### 4.3.4 Honesty and the causal forest

Validity before use: subsampling, honesty (split selection and leaf estimation use disjoint subsamples), and sufficient leaves per tree. The normality result requires regularity conditions on the subsample rate and the tree depth, stated in the source.

Derivation sketch. Each tree partitions the covariate space, so the forest induces weights `alpha_i(x)` that are nonzero only for units sharing a leaf with `x`, averaged over trees. Because honesty makes `alpha_i(x)` depend on the subsample used for splitting and not on the outcomes used for leaf estimation, the weights are independent of the residuals that enter the local estimating equation. The estimate is the solution of a local moment condition that residualises both `Y` on `mu_hat(x)` and `W` on `e_hat(x)` inside the leaf neighbourhood, i.e. a locally weighted R-learner. That is what makes the same weights usable for a variance estimate.

```text
FORMULA [U-12]: tau_hat(x) = sum_i alpha_i(x) * (W_i - e_hat(x)) * (Y_i - mu_hat(x))
                            / sum_i alpha_i(x) * (W_i - e_hat(x))^2
WHERE: alpha_i(x) are honest leaf-sharing weights summing to 1; e_hat, mu_hat
       are propensity and outcome estimates from the same forest or external
       fits; both sums run over the training units; tau is in outcome units.
CONSTRAINT: honest splitting, subsampling, and enough effective units per
       leaf that both sums are stable. Normality is pointwise in x under
       regularity on the subsample rate and tree depth; intervals are not
       simultaneous over x.
NUMERICAL EXAMPLE: a leaf with 12 units, 6 treated and 6 control, e_hat = 0.5,
       mu_hat = 0.05; treated conversions 4, control conversions 1.
       Numerator, treated side: (1/12)*0.5*(4 - 6*0.05) = 0.15417;
       control side: (1/12)*(-0.5)*(1 - 6*0.05) = -0.02917; total 0.125.
       Denominator: (1/12)*12*(0.25) = 0.25.
       tau_hat = 0.125/0.25 = 0.500, i.e. +50 pp in this leaf, which equals
       the raw difference in means 4/6 - 1/6 as it must when the arms are
       balanced and the propensity is constant.
INTERPRETATION: with balanced arms the residualised leaf estimate coincides
       with the difference in means; the residualisation earns its keep only
       when the propensity varies inside the leaf.
```

Boundary/counterexample: the pointwise intervals are not simultaneous. Scanning 100,000 units for the "most significant" effect and reporting the winner's interval is a multiple-comparison error of exactly the kind Section 8.5 describes.

### 4.4 Limits and counterexamples

#### 4.4.1 Overlap failure inflates variance long before it biases the mean

Validity before use: the expression below is the variance of an inverse-propensity weighted arm mean; it is a lower bound on the variance of any estimator that uses weights of this form.

Derivation. Write the weighted arm mean with fixed weights `w_i = 1/e(x_i)` as `sum w_i Y_i / sum w_i`. For independent units with conditional variance `sigma^2(x_i)`, `Var = sum w_i^2 sigma^2(x_i) / (sum w_i)^2`. When one unit's weight dominates, its contribution to the variance is proportional to `w_i^2`, i.e. to `1/e(x_i)^2`. A unit with `e = 0.05` carries weight 20 against weight 2 for a unit with `e = 0.5`, so its squared weight — and therefore its variance contribution per unit of outcome variance — is 100 times larger.

```text
FORMULA [U-13]: Var(tau_hat_w) = sum_i w_i^2 sigma^2(x_i) / (sum_i w_i)^2,
                w_i = 1/e(x_i) for the treated arm, 1/(1-e(x_i)) for control
WHERE: sigma^2(x) conditional outcome variance at x; w are fixed inverse
       propensity weights; the sum runs over the arm's units.
CONSTRAINT: independent units, correct propensity, untruncated weights.
       Truncation changes the estimand to a trimmed population.
NUMERICAL EXAMPLE: two covariate regions of 100 units each, sigma^2 = 0.05.
       Region A has e = 0.5 (w = 2, sum w = 200, sum w^2 = 400);
       region B has e = 0.05 (w = 20, sum w = 2000, sum w^2 = 40000).
       Combined: sum w = 2200, sum w^2 = 40400;
       Var = 0.05 * 40400 / 2200^2 = 2020/4,840,000 = 4.17e-4, SE = 0.0204.
       Region A alone: Var = 0.05 * 400/40000 = 5.0e-4, SE = 0.0224; region
       B alone: Var = 0.05 * 40000/4,000,000 = 5.0e-4, SE = 0.0224. The
       pooled SE is 0.0204, only 9% below either region alone, because B
       contributes 99% of the squared weight while delivering half the units.
INTERPRETATION: a CATE map will be smooth where overlap is good and violently
       unstable at the edges; the instability is a property of the data, not
       of the learner.
```

Boundary/counterexample: truncating weights at, say, 20 removes the explosion but changes the estimand to a trimmed population — report the trimming rule and the population it defines, or the number is not interpretable.

#### 4.4.2 Constant effect: the Qini is exactly zero, and yet measured Qini is positive

Validity before use: the exactly-zero statement requires a *fixed* score that is independent of the outcomes and a truly constant `tau`. The optimism statement requires that the score was selected using the same data that evaluate it.

Derivation, part one. If `tau(x) = tau` for all `x`, then `d_k = tau` in every bin, so `G_k = tau * (cumulative n)` and `G_k = p_k * G_K` exactly. The curve coincides with the random line for every `p`, giving `Q = 0`.

Derivation, part two. If the score is the in-sample maximiser of the measured gain, then the measured gain is a maximum over many candidate rankings of a noisy statistic. Its expectation under the null is strictly positive and grows with the number of candidates and with the noise. This is the mechanism by which "our uplift model found a great top decile" is reported for data with no heterogeneity at all.

```text
FORMULA [U-14]: E[Q_hat | selected in-sample] > 0 even when Q = 0
WHERE: Q_hat is the measured coefficient on the fitting data; selection is
       over scores or hyperparameters.
CONSTRAINT: the inequality is strict whenever the score has any freedom to
       track noise; the size depends on the search space, not on the topic.
NUMERICAL EXAMPLE: with 10 deciles, 5000 units per arm per decile, and no
       true heterogeneity, decile effects have SE ~0.44 pp; the maximum of
       ten such estimates is expected near 1.5 SE, about 0.66 pp, and the
       resulting apparent Qini is positive by construction.
INTERPRETATION: a positive in-sample Qini is not evidence; only a held-out
       or cross-fitted Qini, compared with a permutation band, is evidence.
```

Boundary/counterexample: re-using the same data with cross-fitting reduces but does not eliminate this — if hyperparameters were tuned on the held-out fold, the reported Qini is again optimistic.

#### 4.4.3 Without ignorability, only bounds are available

Validity before use: the bounds below require only bounded outcomes `Y` in `[0,1]` and no other assumption. They are therefore always valid and usually useless, which is the point.

Derivation. The worst case assigns every treated unit the highest possible outcome under control and every control unit the lowest; the best case does the reverse. With treated share `p`, observed arm means `y1` and `y0`, and `Y` in `[0,1]`, the lower bound puts all unobserved control mass at 1 for treated units and all unobserved treated mass at 0 for control units:

```text
lower = -( p * y1 + (1-p) * (1 - y0) )
upper =   p * (1 - y1) + (1-p) * y0
```

```text
FORMULA [U-15]: ATE in [ -(p*y1 + (1-p)*(1-y0)),  p*(1-y1) + (1-p)*y0 ]
WHERE: y1, y0 are observed arm means in [0,1]; p is the treated share of the
       population; both bounds are on the probability scale.
CONSTRAINT: bounded outcome only; no confounding, overlap, or modelling
       assumptions of any kind.
NUMERICAL EXAMPLE: p = 0.5, y1 = 0.0646, y0 = 0.0540:
       lower = -(0.5*0.0646 + 0.5*0.946) = -(0.0323 + 0.4730) = -0.5053
       upper =  0.5*0.9354 + 0.5*0.0540 = 0.4677 + 0.0270 = 0.4947
       so the identified set is [-50.53 pp, +49.47 pp], width 99.99 pp.
INTERPRETATION: the bound is nearly the whole outcome range, three orders of
       magnitude wider than the 1.06 pp effect being studied; it cannot
       support a targeting decision.
```

Boundary/counterexample: quoting a Manski-style bound as if it were a confidence interval is a category error; the bound is about identification, the interval is about sampling, and the bound stays wide no matter how large `n` grows.

#### 4.4.4 The winner's curse in the top decile, quantified

Validity before use: joint normality of `(tau, tau_hat)` at the unit level, which holds approximately for a linear score built on many weak features. The formula is a shrinkage rule, not a correction for arbitrary models.

Derivation. Write `tau_hat = tau + eps` with `eps` independent of `tau`, `Var(tau) = s_t^2`, `Var(eps) = s_e^2`. For jointly normal variables, `E[tau | tau_hat] = mu + rho (tau_hat - mu)` with `rho = s_t^2/(s_t^2+s_e^2)`. Selecting on `tau_hat` therefore selects on a partly-noise quantity, and the expected true effect in the selected set is pulled toward the mean by `rho`.

```text
FORMULA [U-16]: E[tau | tau_hat in top set] = mu + rho * (mean(tau_hat in set) - mu)
WHERE: mu population mean effect; rho reliability in [0,1]; the expectation
       is over repeated experiments.
CONSTRAINT: approximate joint normality; rho estimable from repeated splits
       or from the variance of the score under permutation.
NUMERICAL EXAMPLE: mu = 1.06 pp, top-decile mean tau_hat = 5.00 pp,
       rho = 0.60 gives expected true effect 1.06 + 0.6*3.94 = 3.42 pp,
       a 32% haircut on the headline.
INTERPRETATION: forecast the program on the shrunk number; if the go/no-go
       decision flips between 5.0 and 3.4, the decision is not robust.
```

Boundary/counterexample: `rho` estimated on the training data is itself optimistic; estimate it by comparing scores fitted on disjoint halves, or by the correlation between two independently fitted effect models.

**Takeaway:** every estimator in this section is a smoothness assumption wearing a different costume; the choice is about which nuisance you can model well and how much variance the decision can tolerate.

## 5. Worked Numerical Example

### 5.1 The setting

An illustrative retention program. The eligible base is 100,000 customers, partitioned into ten deciles by a fitted uplift score (decile 1 = highest predicted effect). Inside the experiment each decile contained 5,000 treated and 5,000 control customers, so each decile provides an independent difference in proportions. The outcome is a binary retention event over a 30-day window. Treatment cost is 1.50 per delivered incentive; one retained customer is worth 60.00 of margin. All numbers are constructed for teaching; the arithmetic is checked line by line.

### 5.2 The decile table

| Decile | Ctrl conv | Ctrl rate | Treated conv | Treated rate | Uplift (pp) | Incremental conv per 10k |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 4.00% | 450 | 9.00% | +5.00 | +500 |
| 2 | 350 | 7.00% | 550 | 11.00% | +4.00 | +400 |
| 3 | 150 | 3.00% | 300 | 6.00% | +3.00 | +300 |
| 4 | 300 | 6.00% | 415 | 8.30% | +2.30 | +230 |
| 5 | 100 | 2.00% | 180 | 3.60% | +1.60 | +160 |
| 6 | 250 | 5.00% | 295 | 5.90% | +0.90 | +90 |
| 7 | 450 | 9.00% | 450 | 9.00% | 0.00 | 0 |
| 8 | 400 | 8.00% | 340 | 6.80% | -1.20 | -120 |
| 9 | 175 | 3.50% | 75 | 1.50% | -2.00 | -200 |
| 10 | 325 | 6.50% | 175 | 3.50% | -3.00 | -300 |
| **Total** | **2700** | **5.40%** | **3230** | **6.46%** | **+1.06** | **+1060** |

### 5.3 Arithmetic, decile by decile

Decile 1: control rate 200/5000 = 0.0400; treated rate 450/5000 = 0.0900; difference 0.0500. Incremental over the 10,000-customer decile = 0.0500 × 10,000 = 500 conversions. This scaling step is where many analyses go wrong: the uplift is estimated on 5,000 per arm but applied to 10,000 customers, so the multiplier is the decile population, not the arm size.

Decile 2: 350/5000 = 0.0700 versus 550/5000 = 0.1100; difference 0.0400; 400 conversions.

Decile 3: 150/5000 = 0.0300 versus 300/5000 = 0.0600; difference 0.0300; 300 conversions.

Decile 4: 300/5000 = 0.0600 versus 415/5000 = 0.0830; difference 0.0230; 230 conversions.

Decile 5: 100/5000 = 0.0200 versus 180/5000 = 0.0360; difference 0.0160; 160 conversions.

Decile 6: 250/5000 = 0.0500 versus 295/5000 = 0.0590; difference 0.0090; 90 conversions.

Decile 7: 450/5000 = 0.0900 versus 450/5000 = 0.0900; difference 0.0000; 0 conversions. This decile has the joint-highest baseline in the table and contributes nothing — the purest example of sure things.

Decile 8: 400/5000 = 0.0800 versus 340/5000 = 0.0680; difference -0.0120; -120 conversions.

Decile 9: 175/5000 = 0.0350 versus 75/5000 = 0.0150; difference -0.0200; -200 conversions.

Decile 10: 325/5000 = 0.0650 versus 175/5000 = 0.0350; difference -0.0300; -300 conversions.

### 5.4 Three independent checks on the table

Check one, column sums. Control conversions: 200+350+150+300+100+250+450+400+175+325 = 2700. Treated: 450+550+300+415+180+295+450+340+75+175 = 3230. Both reproduce the totals row.

Check two, aggregate versus average-of-effects. Aggregate: 3230/50,000 − 2700/50,000 = 0.0646 − 0.0540 = 0.0106. Mean of the ten decile effects: (5.0+4.0+3.0+2.3+1.6+0.9+0.0−1.2−2.0−3.0)/10 = 10.6/10 = 1.06 pp. They agree because the deciles are equally sized; with unequal deciles the mean would need re-weighting and the naive average would be wrong.

Check three, incremental total. Sum of the tenth column: 500+400+300+230+160+90+0−120−200−300 = 1060. And `0.0106 × 100,000 = 1060`. The two routes to total incremental conversions agree, which validates both the per-decile scaling and the aggregate rate.

### 5.5 The Qini curve and coefficient

Cumulative gain at each depth, in conversions: `G = (0, 500, 900, 1200, 1430, 1590, 1680, 1680, 1560, 1360, 1060)` at population fractions `p = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)`.

Trapezoid area, term by term with width 0.1:

```text
[0.0,0.1]: (0+500)/2      = 250
[0.1,0.2]: (500+900)/2    = 700
[0.2,0.3]: (900+1200)/2   = 1050
[0.3,0.4]: (1200+1430)/2  = 1315
[0.4,0.5]: (1430+1590)/2  = 1510
[0.5,0.6]: (1590+1680)/2  = 1635
[0.6,0.7]: (1680+1680)/2  = 1680
[0.7,0.8]: (1680+1560)/2  = 1620
[0.8,0.9]: (1560+1360)/2  = 1460
[0.9,1.0]: (1360+1060)/2  = 1210
sum = 12430;  AUUC = 0.1 * 12430 = 1243.0 conversions
```

Random-targeting area: `0.5 × G(1.0) = 0.5 × 1060 = 530.0`. Qini coefficient: `1243.0 − 530.0 = 713.0` conversions. Ratio of areas: `1243.0/530.0 = 2.345`, meaning the score's cumulative gain integrates to about 2.3 times what random targeting achieves at the same average depth.

Note what the coefficient does *not* say. It does not say 713 conversions will be gained in the rollout, because the total gain in the rollout depends on the depth actually chosen and on whether the ranking transports. It also does not normalise against a perfect model, so it cannot be compared with a published Qini from a different population or a differently scaled population size.

### 5.6 Decile-level standard errors

For a difference in independent proportions, `SE = sqrt(p1(1-p1)/n1 + p0(1-p0)/n0)` with 5,000 per arm.

Decile 1: `0.09×0.91/5000 = 1.638e-5`; `0.04×0.96/5000 = 7.680e-6`; sum `2.406e-5`; `SE = 0.004905` = 0.49 pp; `t = 5.00/0.49 = 10.2`.

Decile 2: `0.11×0.89/5000 = 1.958e-5`; `0.07×0.93/5000 = 1.302e-5`; sum `3.260e-5`; `SE = 0.005710` = 0.57 pp; `t = 4.00/0.57 = 7.0`.

Decile 3: `0.06×0.94/5000 = 1.128e-5`; `0.03×0.97/5000 = 5.820e-6`; sum `1.710e-5`; `SE = 0.004136` = 0.41 pp; `t = 3.00/0.41 = 7.3`.

Decile 4: `0.083×0.917/5000 = 1.522e-5`; `0.06×0.94/5000 = 1.128e-5`; sum `2.650e-5`; `SE = 0.005148` = 0.51 pp; `t = 2.30/0.51 = 4.5`.

Decile 5: `0.036×0.964/5000 = 6.941e-6`; `0.02×0.98/5000 = 3.920e-6`; sum `1.086e-5`; `SE = 0.003296` = 0.33 pp; `t = 1.60/0.33 = 4.9`.

Decile 6: `0.059×0.941/5000 = 1.110e-5`; `0.05×0.95/5000 = 9.500e-6`; sum `2.060e-5`; `SE = 0.004539` = 0.45 pp; `t = 0.90/0.45 = 2.0`.

Decile 6 is the interesting row: nominally significant at the 5% level on a two-sided test with a `t` of about 2.0, and utterly unprofitable, as the next table shows. Decile 4 is strongly significant and also unprofitable. The significance test and the decision agree only by coincidence, which is the lesson of 4.2.5.

### 5.7 The costed decision

Break-even uplift: `u* = 1.50/60.00 = 0.025` = 2.5 pp. Per-decile net contribution, taking the 10,000-customer decile as the unit:

| Decile | Uplift | Incr conv | Revenue at 60 | Cost at 1.50 | Net |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.00 | 500 | 30,000 | 15,000 | **+15,000** |
| 2 | 4.00 | 400 | 24,000 | 15,000 | **+9,000** |
| 3 | 3.00 | 300 | 18,000 | 15,000 | **+3,000** |
| 4 | 2.30 | 230 | 13,800 | 15,000 | −1,200 |
| 5 | 1.60 | 160 | 9,600 | 15,000 | −5,400 |
| 6 | 0.90 | 90 | 5,400 | 15,000 | −9,600 |
| 7 | 0.00 | 0 | 0 | 15,000 | −15,000 |
| 8 | −1.20 | −120 | −7,200 | 15,000 | −22,200 |
| 9 | −2.00 | −200 | −12,000 | 15,000 | −27,000 |
| 10 | −3.00 | −300 | −18,000 | 15,000 | −33,000 |
| **All** | 1.06 | 1060 | 63,600 | 150,000 | **−86,400** |

The policy implied by 4.2.5 treats deciles 1 through 3: 30,000 customers, 1,200 incremental conversions, revenue 72,000, cost 45,000, net **+27,000**. The column sum confirms the "treat all" figure: +15,000+9,000+3,000−1,200−5,400−9,600−15,000−22,200−27,000−33,000 = −86,400.

This is the canonical failure the sheet exists to prevent: the average effect is positive and significant, and the full-base program loses 86,400 per wave.

### 5.8 Comparison with level-based targeting at equal spend

Suppose the team instead ranks by predicted *conversion under treatment* — the response-model approach — and still sends to 30,000 customers. The top three deciles by treated rate are decile 2 (11.00%), decile 1 (9.00%), and decile 7 (9.00%). Their effects are 4.00, 5.00, and 0.00 pp, so incremental conversions are 400 + 500 + 0 = 900, revenue 54,000, cost 45,000, net **+9,000**.

Ranking by *baseline* rate instead picks decile 7 (9.00%), decile 8 (8.00%), decile 2 (7.00%), with effects 0.00, −1.20, and +4.00, giving 0 − 120 + 400 = 280 conversions, revenue 16,800, cost 45,000, net **−28,200**.

Three policies, identical spend of 45,000: uplift targeting +27,000, treated-level targeting +9,000, baseline targeting −28,200. The spread is 55,200 per wave, which is twice the profit of the best uplift policy. Same data, same model class, different ranking target.

### 5.9 Sensitivity: what if the score is less reliable than it looks?

Apply the shrinkage rule of 4.4.4 with population mean 1.06 pp and reliability 0.60:

```text
decile 1: 1.06 + 0.60 * (5.00 - 1.06) = 1.06 + 2.364 = 3.42 pp
decile 2: 1.06 + 0.60 * (4.00 - 1.06) = 1.06 + 1.764 = 2.82 pp
decile 3: 1.06 + 0.60 * (3.00 - 1.06) = 1.06 + 1.164 = 2.22 pp
decile 4: 1.06 + 0.60 * (2.30 - 1.06) = 1.06 + 0.744 = 1.80 pp
```

Under this scenario only deciles 1 and 2 clear the 2.50 pp break-even. Their net contributions become `10,000×(0.0342×60 − 1.50) = 10,000×0.552 = 5,520` and `10,000×(0.0282×60 − 1.50) = 10,000×0.192 = 1,920`, for a program total of 7,440 instead of 27,000. The program is still positive, but a 72% haircut on the forecast is the difference between a strategic investment and a marginal test.

The decision that survives both scenarios: run deciles 1 and 2, hold decile 3 behind a confirmatory holdout, and do not treat deciles 4 through 10. That decision is robust to the reliability question, which is the property to aim for.

### 5.10 A second, smaller worked example: the class transformation

Take a single covariate profile with `mu_1 = 0.09`, `mu_0 = 0.04`, balanced arms. Applying 4.2.3: `g = 0.5×0.09 + 0.5×(1−0.04) = 0.045 + 0.480 = 0.525`; `tau = 2×0.525 − 1 = 0.050`. The transformation has converted a two-model problem into one classifier whose output is centred at 0.5 for a zero effect. That centring is a useful debugging property: a fitted `g` that averages 0.50 across the population while the true mean effect is 1.06 pp indicates a mis-scaled or mis-specified transformation, because the average of `g` should be `0.5 + ATE/2 = 0.5053` when the transformation is applied consistently.

**Takeaway:** the arithmetic that decides the program is a subtraction and a division — effect minus break-even — not the area under a curve.

## 6. Visual / Geometric Intuition

### 6.1 The Qini curve against the random line

```text
 cumulative incremental conversions
 1680 |              ####o####
      |          ####         ####
 1400 |      ####                 ####
      |   ###                        ###
 1120 | ##                              ###
      |#                                   ###
  840 |                                      ###
      |                                        ##
  560 |             .............................###  <-- random line ends
      |        .....                                #      at (1.0, 1060)
  280 |    .....
      |....                                        #
    0 +----+----+----+----+----+----+----+----+----+----
      0   .1   .2   .3   .4   .5   .6   .7   .8   .9  1.0
                        population fraction

  '#' = score-ordered gain curve;  '.' = random targeting
  shaded area between them = Qini coefficient = 713 conversions
```

Geometric reading: the curve rises steeply while the score is picking out genuinely high-effect units, flattens at the point where it starts including zero-effect units (around `p = 0.7` where the gain stops rising at 1680), and declines beyond that as harmful units enter. The peak location is the *maximum total* depth, not the *optimal* depth: the optimal depth is set by the cost line, which in this example stops at 0.3.

### 6.2 Uplift by decile, including the negative tail

```text
 uplift (pp)
  +6 |  ##
  +5 |  ##   ##
  +4 |  ##   ##   ##
  +3 |  ##   ##   ##   ##
  +2 |  ##   ##   ##   ##   ##
  +1 |  ##   ##   ##   ##   ##   ##
   0 +--##---##---##---##---##---##---##----------------  <-- zero line
  -1 |                                  ##   ##   ##   ##
  -2 |                                       ##   ##   ##
  -3 |                                            ##   ##
  -4 |                                                 ##
       d1   d2   d3   d4   d5   d6   d7   d8   d9   d10

  Note: d7 sits on the zero line while having the joint-highest baseline
  rate in the table (9.00%). Baseline and effect are orthogonal here.
```

The shape that should worry a reviewer is not a negative tail — a negative tail is informative and actionable — but a *non-monotone middle*. If decile 5 is larger than decile 2, the score is not ordering well and the top-of-list numbers are unstable. Section 8.1 gives the diagnostic.

### 6.3 The response-class scatter

```text
  P(convert | treated)
  1.0 |  SURE THINGS          |  PERSUADABLES
      |  (high, flat)         |  (low baseline,
      |                       |   large vertical gap)
  0.5 |.......................+.......................
      |                       |
      |  LOST CAUSES          |  SLEEPING DOGS
  0.0 |  (low, flat)          |  (high baseline,
      |                       |   large NEGATIVE gap)
      +-----------------------------------------------
       0.0                    0.5                   1.0
                    P(convert | control)

  The diagonal y = x is the zero-effect line. Distance from the diagonal
  is |tau|; side of the diagonal is the sign. A response model ranks by
  height; an uplift model ranks by distance from the diagonal.
```

This is the single most useful picture in the file, because it explains why predicted-probability ranking fails. A high-conversion segment sits at the top of the plot but can be anywhere horizontally: at the top-left it is sure things, at the top-right it may be sleeping dogs. Only the horizontal displacement from the diagonal measures the effect.

### 6.4 What the R-learner is doing, geometrically

```text
  Y - m(x)
      ^
      |                       * slope = tau(x)
      |                     *
      |                   *
      |                 *
      |               *
      |             *
      |           *
  ----+------------------------> W - e(x)
      |      0
      |
```

Each point is a unit placed by its treatment residual (horizontal) and outcome residual (vertical). The cloud has been tilted so that its principal direction has slope equal to the effect at that `x`. Before residualisation, the cloud would be dominated by the level of the outcome — the intercept would be huge and the slope unreadable. Residualising by `m(x)` removes the intercept; residualising by `e(x)` puts the horizontal axis in units of "surprise in assignment", so a unit's leverage is exactly how unexpected its treatment was.

### 6.5 Calibration versus ranking

```text
  observed effect (out-of-sample DR estimate)
    6% |            *
       |        *
    3% |    *
       | *
    0% +---*----*----*----*----
       0%  1.5% 3%   4.5% 6%   predicted effect
              (45-degree line = perfect calibration)

  Ranking quality = are the points ordered left to right?
  Calibration     = are they near the diagonal?
  A model can pass the first test and fail the second by a factor of 2.
```

Ranking is what a targeting decision needs; calibration is what a forecast needs. Section 5.7 is a ranking decision (which deciles to treat) and Section 5.9 is a calibration question (how much money to expect). Mixing them produces the familiar "the model works but the business case did not materialise".

### 6.6 The score-versus-depth curve for the decision

```text
  net contribution per decile (thousands)
  +15 | ##
  +12 | ##
   +9 | ##   ##
   +6 | ##   ##
   +3 | ##   ##   ##
    0 +--##---##---##---##-------------------  break-even
   -3 |                  ##   ##
   -6 |                       ##   ##
   -9 |                            ##   ##   ##
  -12 |                                     ##   ##
  -15 |                                          ##
       d1   d2   d3   d4   d5   d6   d7   d8   d9   d10
```

The decision-relevant picture is this one, not the Qini curve. It is a per-decile subtraction of a constant cost from a decile-specific revenue, so it crosses zero at a well-defined point (between deciles 3 and 4) and the crossing is the depth to deploy.

**Takeaway:** draw the curve, draw the random line, draw the cost line — the answer is where the third one crosses, not where the first one peaks.

## 7. Comparison / Selection Table

### 7.1 Estimator selection

| Estimator | Identification needed beyond trial | Sample efficiency | Handles unbalanced arms | Main failure | Use when |
|---|---|---|---|---|---|
| T-learner | none in a trial | low (variance doubles on the difference) | poorly | Noisy where arms are small; two error surfaces | Effects large relative to baseline; simple auditable baseline |
| S-learner | none in a trial | high if interaction survives | well | Regularisation shrinks interaction to zero; rank-degenerate output | Small effects, strong shared structure, linear/GAM base |
| X-learner | overlap | medium | well (by design) | Minority-arm imputations dominate noise | Arms very unbalanced, e.g. 90/10 delivery |
| R-learner | overlap, `e(x)` estimable | high | well | Nuisance misspecification reintroduces bias | Observational data with a good propensity model |
| DR-learner | overlap, `e(x)` or outcome models | high | well | Variance explosion at extreme `e(x)` | Observational data, one nuisance unreliable |
| Causal forest | overlap, honest splits | medium | medium | Too few units per leaf; pointwise intervals misused | Need per-unit uncertainty; moderate `n`, many covariates |
| Transformed outcome | constant or known `e(x)` | high | requires `p` handling | Mis-scaling when `e(x)` varies | Balanced trial, fastest possible baseline |

Decision rules that follow from the table. If the data are from a clean 50/50 trial and the effects are small, start with an S-learner with an explicit treatment-by-covariate structure and compare against a T-learner; if the S-learner's CATE has near-zero variance, it has failed to learn heterogeneity rather than discovered homogeneity — check the variance of the score, not just its Qini. If arms are unbalanced by design, use the X-learner. If the data are observational, do not use a T-learner without a propensity adjustment inside each arm; prefer the DR-learner and report the overlap diagnostics. If the deliverable includes "how confident are we per customer", only the forest-family gives you that, with the caveat about pointwise intervals.

### 7.2 Evaluation metric selection

| Metric | Answers | Units | Sensitive to | Do not use for |
|---|---|---|---|---|
| Qini coefficient | does the ranking beat random? | outcome × population fraction | population size, bin count, definition variant | comparing across populations of different sizes |
| AUUC | total area under the gain curve | outcome × population fraction | scale of outcome and of population | a normalised quality claim |
| Uplift by decile | where does the ranking work? | pp per decile | bin boundaries, arm sizes | a single headline number |
| Top-k policy value | what does the chosen policy earn? | currency | `k`, cost assumptions | evaluating depths other than `k` |
| CATE calibration | is the level right? | pp | binning of predicted effect | ranking decisions |
| Permutation-band test | is any of this distinguishable from noise? | p-value or band | number of permutations, search space | effect size estimation |

The practical rule: report a decile table, a held-out Qini with a permutation band, and a policy value at the chosen depth. A single Qini number without the other two is not reviewable.

### 7.3 Targeting-policy comparison

| Policy | Rule | Requires | Strength | Failure |
|---|---|---|---|---|
| Treat all | `pi(x) = 1` | nothing | simple, no model risk | loses money whenever mean effect < break-even |
| Treat none | `pi(x) = 0` | nothing | zero cost, honest baseline | leaves value on the table when effects are large |
| Response model | rank by `mu_1(x)` | outcome model | cheap, familiar | targets sure things; can select sleeping dogs |
| Uplift threshold | treat iff `tau(x) > c/v` | CATE + cost | directly optimises contribution | sensitive to `rho` and to cost mis-estimation |
| Capacity knapsack | rank by `v tau(x) - c(x)` per unit capacity | CATE + capacity | handles constrained operations | needs a shadow price; unstable when capacity is small |
| Segment policy | treat pre-specified segments | prior evidence | auditable, low variance | coarse; may miss the profitable middle |

The honest baseline comparison in Section 5.8 is the standard to hold submissions to: any uplift policy must be compared against treat-all, treat-none, and a response-model ranking at equal spend. A policy that beats random but loses to treat-none is a net-negative contribution dressed as sophistication.

### 7.4 When not to use uplift modelling at all

Four disqualifiers, each sufficient on its own. First, the mean effect is far below the break-even uplift and there is no plausible subgroup above it — then the answer is "do not run the campaign", and no model improves that. Second, the experiment is too small to resolve heterogeneity: if the per-decile standard error is comparable to the effects, the ranking is noise and the correct output is "no usable heterogeneity detected". Third, the treatment cannot be withheld from a subset for legal, contractual, or fairness reasons — then the targeting question is moot and the decision is about content, not selection. Fourth, the outcome window extends beyond the decision deadline — then any ranking is based on a proxy that has not been validated against the decision-relevant outcome.

**Takeaway:** pick the estimator by which nuisance you can model, and pick the policy by comparing it to treat-all and treat-none at equal spend.

## 8. Failure Modes & Edge Cases

Each entry gives the observable symptom, the competing causes that produce it, the check that discriminates between them, the smallest fix that resolves it, and how to verify the fix.

### 8.1 Failure: the offline Qini is strong and the rollout gains nothing

**Symptom.** Held-out Qini of 700+ conversions, a clean top-decile story, and a post-launch measurement within noise of zero.

**Competing causes.** (a) The ranking was evaluated on data used to choose it, so the reported Qini is selection-optimistic. (b) The deployment population differs from the experimental population — different tenure mix, different channel, different season. (c) The rollout implemented a level-based list rather than the uplift list, or capped delivery so that the intended deciles were not the delivered deciles. (d) The effect decayed between experiment and rollout (novelty, competitive response, saturation). (e) The outcome was measured on a different window or definition.

**Discriminating check.** Recompute the Qini on a genuinely untouched holdout that was never used for fitting, tuning, or model selection, and compare with a permutation band. Then compare the covariate distribution of the deployment base with the experiment base using standardised mean differences per feature; a difference above about 0.1 on features that drive the score points to (b). Then compare the realised delivery file against the intended deciles — a match rate below 95% points to (c). Cause (d) requires the post-launch do-no-harm holdout, which is the only way to distinguish it from (e).

**Bounded fix.** Freeze the score, evaluate on the untouched holdout, and if the Qini survives, ship with a 5% do-no-harm holdout. If it does not survive, shrink the depth to the top two deciles and re-test.

**Verification.** The held-out Qini must exceed the 95th percentile of the permutation band, and the top-two-decile uplift must clear break-even on the holdout. Both conditions, not either.

### 8.2 Failure: the CATE is really a selection effect

**Symptom.** Large positive effects in the segment that was easy to contact; a plausible business story; no experiment.

**Competing causes.** (a) Unmeasured confounding — motivation drives both contact and conversion. (b) The "control" group is a different population by construction (for example, customers who were eligible but unreachable). (c) The model captured a post-treatment variable — a feature that is itself caused by the treatment, such as "number of app sessions in the 7 days after the offer". (d) Reverse causation in a feature measured after the outcome window opens.

**Discriminating check.** For (c) and (d), audit every feature's availability timestamp against the treatment timestamp; any feature whose first observable value postdates assignment is disqualified, no exceptions. For (a), run a negative-control outcome that the treatment cannot plausibly affect (a pre-period behaviour) and check whether the model still "finds" an effect there — a nonzero CATE on a pre-period outcome is a confounding signature. For (b), compare covariate distributions of the two groups before adjustment; overlap collapse is diagnostic.

**Bounded fix.** Drop post-treatment features; re-estimate with the DR-learner; if a negative-control effect appears, do not report a CATE from this data — run a randomised holdout instead.

**Verification.** Negative-control CATE statistically indistinguishable from zero, all features timestamped before assignment, and overlap diagnostics within the reported range.

### 8.3 Failure: extreme propensity produces a jagged CATE map

**Symptom.** Smooth effects in the middle of the covariate space and wild swings at the edges; a handful of units with effects of 40 percentage points.

**Competing causes.** (a) Genuine non-linearity at the boundary. (b) Inverse-propensity weights of 1/0.02. (c) The model extrapolating a tree ensemble past its training support. (d) A coding error in the propensity (for example, fitting it on the full data including the outcome window).

**Discriminating check.** Plot the propensity distribution; compute the effective sample size `(sum w)^2 / sum w^2` for the weighted estimate. If the effective sample is under about 10% of the nominal count, the variance is the explanation. Then check whether the extreme units are inside the convex hull of the training covariates.

**Bounded fix.** Trim or stabilise the weights, declare the trimmed population, and re-report the estimand on that population.

**Verification.** Recompute the CATE map with the trimmed weights; the edge swings should shrink by roughly the square root of the weight reduction, and the interior estimates should barely move. If the interior moves a lot, the problem was not the weights.

### 8.4 Failure: the top decile is the smallest and noisiest bin

**Symptom.** The headline effect comes from a decile with 300 units and a confidence interval that includes zero two deciles down.

**Competing causes.** (a) A genuine effect concentrated in a rare segment. (b) A scoring artefact that puts the most extreme score values on the units with the least data (common with tree ensembles near the boundary). (c) Bin construction: equal-count deciles over a population with a heavily skewed score produce a top bin that is a different size from the rest.

**Discriminating check.** Report `n` per bin and the per-bin standard error alongside the effect. If the top-bin `t` statistic is below about 3, the headline is not established. Then re-bin by score value (not by count) and see whether the top effect persists.

**Bounded fix.** Merge the top bins to a minimum sample size, or report the effect with an interval and choose the depth from the interval's lower bound rather than its point estimate.

**Verification.** The chosen depth must remain profitable when the effect is set to the lower confidence bound. In Section 5, using the lower bound of decile 3's interval (3.0 − 1.96×0.41 = 2.20 pp) puts it below the 2.50 pp break-even, which is exactly the kind of fragility the check exposes.

### 8.5 Failure: many segments, one "discovery"

**Symptom.** A report with fifteen subgroup effects, one of which is significant at 5%, presented as the finding.

**Competing causes.** (a) A real moderator. (b) Multiplicity: fifteen tests at 5% produce a false positive with probability about `1 − 0.95^15 = 0.54`. (c) The subgroups were chosen after seeing the data.

**Discriminating check.** Count the tests. Compare the smallest p-value with a Bonferroni or Benjamini–Hochberg threshold for the number actually run. Then check whether the moderator was pre-specified in the analysis plan.

**Bounded fix.** Report all subgroups with a multiplicity-adjusted threshold, and treat any post-hoc subgroup as a hypothesis for the next experiment, not as a finding.

**Verification.** The next experiment tests the moderator as a pre-registered contrast, and the effect replicates with the same sign and a comparable magnitude.

### 8.6 Failure: heterogeneity that is really a measurement artefact

**Symptom.** Effects differ sharply by a variable that is itself derived from the outcome pipeline — for example, "customers with at least one support ticket in the window".

**Competing causes.** (a) Genuine moderation. (b) The variable is a collider or a downstream consequence: the treatment changes ticket volume, and ticket volume is correlated with the outcome. (c) Differential measurement: the treated arm has more opportunities to be observed (more app opens, hence more chances to record a conversion).

**Discriminating check.** Ask whether the variable is measured identically in both arms and before assignment. Then test whether the treatment shifts the variable's distribution — a large shift means it is post-treatment and cannot be a moderator for a decision, only a mediator.

**Bounded fix.** Remove the variable, or move it into a mediation analysis with explicit assumptions, and never into a targeting rule.

**Verification.** Re-estimate without the variable; the surviving heterogeneity must come from pre-treatment covariates only.

### 8.7 Failure: interference between units

**Symptom.** The uplift model shows strong positive effects in dense urban cells and negative effects in sparse ones, and the aggregate effect is smaller than the sum of the parts.

**Competing causes.** (a) Genuine geographic heterogeneity. (b) Spillover: treated customers tell untreated neighbours, or a limited-inventory offer is consumed by treated customers leaving less for controls. (c) Saturation: the campaign saturates the channel in dense areas, so the marginal treated customer is worth less.

**Discriminating check.** Compare the control-arm outcome in high-density areas where the treated share was high against low-density areas where it was low. Under no interference, control outcomes should not depend on the treated share; a dependence is direct evidence of spillover.

**Bounded fix.** Cluster-randomise at the geographic or household level, or use a saturation design with several treated-share levels, and estimate the effect as a function of the share.

**Verification.** The control outcome is flat in the treated share within cluster-randomised data, and the estimated effect at the deployed saturation level is the one used in the forecast.

### 8.8 Failure: the cost model, not the effect model, is wrong

**Symptom.** The model's ranking is validated, the depth is chosen correctly, and the program still misses its financial target by 40%.

**Competing causes.** (a) Treatment cost per delivered unit understated — fulfilment, contact-centre load, discount leakage to customers who would have bought anyway. (b) Value per retained customer overstated — margin, not revenue; net of future service cost. (c) Cannibalisation: the retained customer would have stayed with a smaller incentive. (d) Double counting between the uplift effect and the incentive's price effect.

**Discriminating check.** Rebuild the cost from the invoice and the value from the margin line, not from the business case. Then recompute the break-even `c/v`; a 40% miss usually implies `c` is 40% higher or `v` is 40% lower than assumed, and the sensitivity of the depth to that change is the diagnostic.

**Bounded fix.** Recompute the threshold with the audited cost and value, and re-derive the depth. Report the depth as a function of `c/v` so that a change in either input is a one-line update.

**Verification.** The forecast uses margin, and a reconciliation of predicted versus realised contribution at the decile level closes within a stated tolerance.

### 8.9 Failure: the score is fine and the deployment is not

**Symptom.** A validated score, an approved depth, and a delivered campaign whose treated group has a different composition than intended.

**Competing causes.** (a) Eligibility filters applied downstream (suppression lists, contact-frequency caps) removed units non-randomly. (b) The score was computed on a stale feature snapshot. (c) The delivery system sampled for capacity and lost the ordering. (d) Two campaigns ran simultaneously on overlapping populations.

**Discriminating check.** Join the intended list to the delivery log at the unit level. Compute the decile distribution of *delivered* units and compare with the intended 10% per decile. A chi-square test on that table localises the loss quickly.

**Bounded fix.** Move the eligibility filter upstream of the scoring, freeze the feature snapshot timestamp, and enforce ordering at the delivery layer.

**Verification.** Delivered decile shares within 1 percentage point of intended, and the feature snapshot timestamp precedes the delivery timestamp for 100% of units.

**Takeaway:** the failures that cost money are rarely estimation failures; they are evaluation, deployment, and cost-model failures that a decile table exposes in one page.

## 9. Complexity & Cost Analysis

### 9.1 Computational cost of the estimator families

Let `n` be the number of experimental units, `p` the number of covariates, `K` the number of cross-fitting folds, and `B` the number of trees.

| Procedure | Fits | Approximate cost | Memory | Notes |
|---|---|---|---|---|
| S-learner | 1 | `C(n, p)` for the base learner | `O(n p)` | one model, interaction must be explicit |
| T-learner | 2 | `2 C(n/2, p)` | `2 O(n p)` | two disjoint fits |
| X-learner | 4 | `2 C(n/2,p) + 2 C(n_arm,p)` | `O(n p)` | two extra effect models on imputed labels |
| R-learner | `2K + K` | `3K C(n(K-1)/K, p)` | `O(n p)` | nuisances for `m`, `e`, then `tau` |
| DR-learner | `3K` | `3K C(...)` | `O(n p)` | plus pseudo-outcome storage `O(n)` |
| Causal forest | `B` trees | `B * O(n_sub p log n_sub)` | `O(B n_sub)` | subsample per tree, honesty halves usable rows |

With `n = 200,000`, `p = 60`, `K = 5`, `B = 500`, and `n_sub = 100,000`, the DR-learner performs 15 nuisance fits — the dominant cost — while the forest performs 500 subsampled fits. On commodity hardware with a gradient-boosting base learner, the DR-learner is usually minutes and the forest tens of minutes; both are one-off costs relative to a campaign that runs for weeks. The computational cost is therefore almost never the binding constraint, which is worth saying explicitly because teams routinely trade statistical quality for speed in a setting where speed is free.

### 9.2 Scoring and serving cost

```text
FORMULA [U-17]: node evaluations = N_units * B * D
WHERE: N_units is the number to score, B trees per model, D average depth.
CONSTRAINT: a plain (non-vectorised) tree traversal visits one node per level;
       vectorised implementations are faster by a constant factor only.
NUMERICAL EXAMPLE: 10,000,000 units * 500 trees * 8 levels = 4.0e10 node
       evaluations. At a deliberately conservative 1e8 evaluations per
       second per core, that is 400 core-seconds, i.e. under 7 minutes on
       one core and seconds on a small cluster.
INTERPRETATION: daily scoring of a ten-million base is a batch job, not an
       architectural problem; the cost that matters is the campaign.
```

Storage is equally modest: 10 million scores at 8 bytes is 80 MB, and a 60-feature float32 matrix for the same base is `1e7 × 60 × 4 = 2.4 GB`, which is the number that actually constrains the pipeline. The point of the calculation is to move the conversation from "can we compute it" to "is it right".

### 9.3 Sample size for the average effect

```text
FORMULA [U-18]: n_per_arm = (z_{1-alpha/2} + z_{1-beta})^2 *
                            (p1(1-p1) + p0(1-p0)) / Delta^2
WHERE: Delta is the detectable difference in proportions; z values for the
       two-sided level and the desired power.
CONSTRAINT: independent units, two equal arms, one comparison, no clustering
       and no attrition. Multiplicity and clustering both reduce power.
NUMERICAL EXAMPLE: Delta = 0.01, p1 = 0.0646, p0 = 0.0540,
       alpha = 0.05, power = 0.80: (1.960 + 0.8416)^2 = 7.849;
       p1(1-p1) = 0.060427, p0(1-p0) = 0.051084, sum = 0.111511;
       n = 7.849 * 0.111511 / 0.0001 = 8752.5, so 8,753 per arm
       and 17,506 units in total.
INTERPRETATION: the experiment in Section 5 (100,000 units) has roughly 5.7
       times this per-arm requirement, which is why its decile effects are
       estimable at all.
```

### 9.4 Sample size for heterogeneity: why the multiplier is four

Detecting that two segments respond differently is a difference of two differences, so its variance is the sum of four independent cell variances rather than two. With equal cells of size `m`, `Var = 4 sigma^2 / m` against `2 sigma^2 / m` for a single difference of the same size, so the per-cell sample must be twice as large and the total four times as large.

```text
FORMULA [U-19]: m_per_cell = (z_{1-alpha/2}+z_{1-beta})^2 * 4 * p(1-p) / Delta^2
WHERE: m is the per-cell sample in a 2x2 (segment x arm) layout; Delta is the
       difference in uplift between the two segments.
CONSTRAINT: equal cells, independent units, a single pre-specified contrast.
NUMERICAL EXAMPLE: p = 0.05, p(1-p) = 0.0475, Delta = 0.02:
       m = 7.849 * 4 * 0.0475 / 0.0004 = 3728.3, so 3,729 per cell and
       14,916 total. The single-difference version gives 1,864 per arm and
       3,728 total: exactly a factor of four.
INTERPRETATION: an experiment powered for the average effect is almost never
       powered for a moderator; plan the heterogeneity question separately.
```

The practical consequence for Section 5: with 5,000 per arm per decile, a decile effect of 1.0 pp has `SE` near 0.44 pp, so only effects above roughly 1.3 pp are distinguishable from zero at conventional levels. The deciles with effects of 0.9 and 0.0 pp are therefore *estimates of small numbers*, not *demonstrations of small numbers*, and the ranking between decile 6 and decile 7 is not resolvable at this sample size.

### 9.5 Economic sizing of the program

At `c_t = 1.50` and `v = 60.00`, the costed decision in Section 5.7 has these dimensions:

| Quantity | Targeted (d1–d3) | Treat all | Level-targeted |
|---|---:|---:|---:|
| Customers treated | 30,000 | 100,000 | 30,000 |
| Delivery cost | 45,000 | 150,000 | 45,000 |
| Incremental conversions | 1,200 | 1,060 | 900 |
| Revenue at 60 | 72,000 | 63,600 | 54,000 |
| Net contribution | **+27,000** | **−86,400** | **+9,000** |
| Net per treated customer | +0.90 | −0.86 | +0.30 |
| Net per eligible customer | +0.27 | −0.86 | +0.09 |

Four campaign waves a year at this base make the difference between the targeted and the treat-all policy `4 × (27,000 + 86,400) = 453,600` per year. That is the number to put in front of the decision-maker, and it is derived entirely from the decile table plus two unit costs.

### 9.6 Cost of getting the evaluation wrong

A reliability of 0.60 instead of 1.0 reduces the forecast from 27,000 to 7,440 per wave (Section 5.9), an error of 19,560 per wave or 78,240 a year at four waves. The cost of the extra evaluation work — one untouched holdout, a permutation band, and a repeated-split reliability estimate — is measured in analyst days. The asymmetry is stark and is the economic argument for the evaluation budget.

**Takeaway:** computation is cheap, campaign cost is large, and evaluation error is the largest term in the budget; spend accordingly.

## 10. Key Theorems / Results

Each result is stated with its conditions, because the conditions are the part that gets dropped in practice. Citations are to the standard literature; page numbers are omitted where this authoring pass did not re-verify them, which is marked OPEN.

**R-1. The fundamental problem of causal inference.** For any unit, at most one of `Y_i(1)`, `Y_i(0)` is observed, so `tau_i` is never identified from data alone without additional structure. Consequence: every "individual effect" is a model-based extrapolation, and any product that reports one per customer is reporting a prediction, not a measurement. (Holland 1986; SOURCED.)

**R-2. Identification of the CATE.** Under consistency, SUTVA, conditional ignorability, and positivity, `tau(x) = E[Y|X=x,W=1] - E[Y|X=x,W=0]`. Consequence: the assumption list is short and each item is separately falsifiable in a designed study; in an observational study none of them is testable without a design argument. (Rosenbaum & Rubin 1983 for the propensity formulation; Imbens & Rubin 2015 for the potential-outcomes statement; SOURCED.)

**R-3. The propensity score as a balancing score.** If ignorability holds given `X`, it holds given `e(X)`. Consequence: adjustment can be one-dimensional, which is what makes weighting and matching feasible with many covariates; it does not make the adjustment assumption true. (Rosenbaum & Rubin 1983; SOURCED.)

**R-4. Double robustness.** The AIPW estimator of `tau(x)` is consistent if either `e(x)` or the outcome regressions are correctly specified. Consequence: two chances to be right, which is why DR-learner and AIPW variants dominate in observational settings — but the variance can still be unbounded when overlap fails, so robustness of the mean is not robustness of the interval. (Robins, Rotnitzky & Zhao 1994; SOURCED.)

**R-5. Robinson partialling-out.** Regressing `Y - m(X)` on `(W - e(X)) tau(X)` identifies `tau` when `m` and `e` are known; with estimated nuisances, `n^{-1/4}`-rate nuisance errors still allow `sqrt(n)`-consistent and asymptotically normal estimation of `tau` under regularity. Consequence: machine-learned nuisances are admissible, which is the entire justification for the modern meta-learner family. (Robinson 1988; Chernozhukov et al. 2018; SOURCED.)

**R-6. Quasi-oracle property of the R-learner.** Under a sparsity or smoothness condition on `tau` and cross-fitted nuisances, the R-learner attains the same convergence rate as an oracle that knew `m` and `e`. Consequence: the effect model can be fitted with a flexible learner without paying a first-order price for nuisance estimation — provided cross-fitting is actually used. (Nie & Wager 2021; SOURCED.)

**R-7. Meta-learner equivalence and difference.** S-, T-, X-, and R-learners target the same estimand; they differ in bias and variance under finite samples and in their behaviour when one arm is small. Consequence: "which meta-learner" is an empirical question answered by held-out evaluation, not a matter of principle. (Künzel et al. 2019; SOURCED.)

**R-8. Honest splitting and forest inference.** A causal forest with honest splitting yields pointwise asymptotic normality of `tau_hat(x)` with a variance estimable from the leaf weights. Consequence: per-unit uncertainty is available, but the intervals are pointwise, not simultaneous, so scanning many units for significance requires a multiplicity correction. (Wager & Athey 2018; SOURCED.)

**R-9. The constant-effect null for the Qini coefficient.** If `tau(x)` is constant and the score is fixed and independent of the outcomes, the Qini coefficient is exactly zero for every depth, because `G(p) = p G(1)`. Consequence: any positive Qini is either heterogeneity or noise, and only an out-of-sample comparison against a permutation band distinguishes them. (DERIVED in 4.4.2.)

**R-10. Selection optimism.** If the score is chosen as the maximiser of a measured gain, the measured gain has strictly positive expectation under the null, with the excess growing in the size of the search space. Consequence: in-sample Qini is not a test statistic; the fix is held-out evaluation with the fitting and the evaluation separated by construction. (DERIVED; the general phenomenon is the winner's curse, standard in multiple-comparison theory.)

**R-11. Shrinkage of selected extremes.** Under joint normality of `(tau, tau_hat)`, `E[tau|tau_hat] = mu + rho(tau_hat - mu)` with `rho` the reliability ratio. Consequence: the headline effect of the selected set is biased upward by `(1-rho)` times its distance from the mean, and the forecast must use the shrunk value. (DERIVED in 4.4.4.)

**R-12. Metric non-standardisation.** Published uplift studies use materially different definitions of the uplift curve, AUUC, and Qini coefficient — including different weightings by arm size and different normalisations. Consequence: a Qini number is comparable only to a number computed with the identical definition on a comparable population; cross-paper comparisons are unsound without the code. (Devriendt, Moldovan & Verbeke 2018; Gutierrez & Gérardy 2017; SOURCED for the existence of variants; OPEN for any specific library's current default, which must be read from the source at the version in use.)

**R-13. Policy regret and empirical welfare maximisation.** Choosing a policy by maximising estimated value over a class `Pi` achieves regret of order `sqrt(V/n)` relative to the best policy in the class, under bounded outcomes and a complexity condition on `Pi`; the rate degrades with the complexity of the class. Consequence: a richer policy class is not free — the regret bound loosens as the class grows, which is the formal version of "do not target on fifteen segments". (Kitagawa & Tetenov 2018; Athey & Wager 2021; SOURCED.)

**R-14. Bounds without assumptions.** With bounded outcomes and no confounding assumptions, the ATE is bounded but the interval typically has width of order the outcome range, so it cannot support a decision. Consequence: bounds are a honesty device for the limitations section, not an analysis. (Manski 1990; SOURCED.)

**Takeaway:** four assumptions carry the whole edifice — consistency, no interference, ignorability, positivity — and every estimator in the file is a different way of paying for them.

## 11. Interview / Exam Questions

Fifteen questions, each with the reasoning that earns credit and the answer shape that should raise a red flag.

**Q1. Define the CATE and explain why the individual effect is not identified.**
The CATE is `E[Y(1) - Y(0) | X = x]`, an average over units sharing covariates. The individual effect is not identified because consistency gives each unit exactly one realised potential outcome, so the other is counterfactual by construction; no amount of data on one unit reveals it. Identification of the CATE comes from comparing *different* units that share `x`, which is why ignorability and positivity are needed. *Red flag:* answering "we identify it with a strong model" — model class is not identification.

**Q2. Why can't you compute uplift as treated-converters minus control-converters?**
That comparison conditions on the outcome, which is post-treatment. Among converters, the treated are a different mixture of persuadables and sure things than the control converters, so the difference measures selection into conversion, not the effect. The correct contrast conditions on covariates or on randomisation, not on `Y`. *Red flag:* proposing a t-test on the subset where `Y = 1`.

**Q3. Using the decile table in Section 5, compute the ATE and verify it two ways.**
Aggregate: `3230/50,000 - 2700/50,000 = 0.0646 - 0.0540 = 0.0106`. Average of decile effects: `(5.0+4.0+3.0+2.3+1.6+0.9+0.0-1.2-2.0-3.0)/10 = 10.6/10 = 1.06` pp. They agree because deciles are equal-sized. Total incremental conversions `1060` also equals `0.0106 × 100,000`. *Red flag:* reporting 1.06 pp without noticing that three deciles are negative, i.e. treating the average as a description of every unit.

**Q4. With cost 1.50 and value 60, which deciles do you treat, and why not decile 4?**
Break-even is `1.50/60 = 0.025`, so deciles 1, 2, and 3 (effects 5.0, 4.0, 3.0 pp) are treated and decile 4 (2.3 pp) is not. Decile 4 is statistically distinguishable from zero (`t ≈ 4.5`) but its expected net contribution is `10,000 × (0.023×60 − 1.50) = −1,200`. Statistical significance is not the decision boundary. Treating all ten gives `−86,400`; treating three gives `+27,000`. *Red flag:* treating everything with a positive point estimate, or treating everything because the average is significant.

**Q5. Interpret a Qini coefficient of 713 conversions with a total gain of 1060.**
The area between the score curve and the random line is 713 conversions on a population fraction axis, against a random area of 530. The ratio 2.345 says the score concentrates gain about 2.3 times better than random at the same average depth. It does **not** forecast 713 extra conversions: the realised gain depends on the depth chosen (1,200 conversions at 30% in Section 5.7) and on transport. *Red flag:* quoting the Qini as the expected business gain.

**Q6. A T-learner and an S-learner disagree. Which do you trust?**
Neither on principle; the comparison is empirical. The T-learner has roughly four times the variance on the difference (Section 4.3.1), so it is noisier; the S-learner can shrink the interaction to zero and produce a constant, rank-degenerate CATE. Check the variance of the S-learner's score first: near-zero variance means the interaction was lost, not that heterogeneity is absent. Then compare held-out Qini and, decisively, held-out policy value at the chosen depth. *Red flag:* choosing by in-sample fit or by which one has the larger top-decile effect.

**Q7. When is a DR-learner preferred to a T-learner?**
When assignment is observational, so the T-learner's within-arm comparison is confounded; when overlap is adequate so the inverse-propensity terms are stable; and when at least one nuisance (propensity or outcome model) is defensible. The DR-learner is consistent if either is right, and it uses all units rather than half per arm. It is *not* preferred when overlap fails badly, because the variance is then unbounded regardless of double robustness. *Red flag:* "DR is always better because it is doubly robust."

**Q8. How do you validate an uplift model out of sample, concretely?**
Split units, not rows; fit the score on the training portion including any hyperparameter selection; compute the Qini on the untouched portion; build a permutation band by shuffling the score within the holdout (200–1,000 permutations) and require the observed Qini above the 95th percentile; check decile monotonicity on the holdout; and check CATE calibration by regressing the out-of-sample doubly robust pseudo-outcome on the predicted effect, where the slope should be near 1. *Red flag:* cross-validated Qini where the folds were also used for tuning.

**Q9. The holdout Qini is negative. What does that mean and what do you do?**
Negative means the ordering is worse than random on that population: either the model is overfit to a non-transporting sample, or the score's sign is flipped somewhere (a common bug when a difference is computed as control minus treated), or the population shifted. Check the decile table on the holdout for a monotone decline of the effect; check the sign convention against a hand-computed decile; check covariate drift. Do not "fix" it by re-fitting on the holdout. *Red flag:* reporting the training Qini after seeing a negative holdout Qini.

**Q10. The experiment showed +1.06 pp and the rollout shows zero. Diagnose.**
Five candidates, checked in order: (1) the rollout targeted on level rather than effect (Section 5.8 shows a 3× profit difference at equal spend); (2) population shift between experiment and deployment; (3) delivery shortfall — suppression lists, frequency caps, stale features (check delivered decile shares); (4) effect decay from novelty or competition (only the do-no-harm holdout distinguishes this); (5) outcome definition or window changed between the two measurements. *Red flag:* attributing the gap to "the model overfit" without checking the delivered list.

**Q11. Explain sleeping dogs and what action follows.**
Units with `Y(1) < Y(0)`: the intervention reduces the outcome. In the illustrative table, deciles 8–10 have effects of −1.2, −2.0, and −3.0 pp; treating decile 10 costs `10,000 × (−0.03×60 + 1.50) = −3,000` in lost value on top of the 15,000 delivery cost, i.e. −33,000 net. Suppression is the action, but it must be operationally possible: "do not treat" has to be a maintainable state, and if the control condition requires an active choice (a holdout that customers notice), the negative effect may not persist. *Red flag:* treating negative-uplift segments because "everyone gets the campaign".

**Q12. How many units per arm to detect a 1 pp effect at 5% and 80% power, with base rates near 5.4% and 6.5%?**
`n = (1.960 + 0.8416)^2 × (0.0646×0.9354 + 0.0540×0.9460) / 0.01^2 = 7.849 × 0.111511 / 0.0001 ≈ 8,753` per arm, 17,506 total. Note the answer is not "a few thousand": a 1 pp effect on a 5% base is a 20% relative change but a small absolute one, and the sample follows the absolute scale. *Red flag:* quoting 2,000 per arm from a rule of thumb for a 30% base rate.

**Q13. How do you handle treatment cost that varies by segment?**
Generalise the rule: treat iff `tau(x) > c_t(x)/v`, and with capacity constraints rank by `v tau(x) − c_t(x)` per unit of scarce capacity. If cost is a step function (free in-app message versus paid voucher), the policy has two thresholds and the cheaper channel should be considered first for medium-effect units. Estimate the cost from the invoice, not the business case: a 40% understatement of `c` is common and moves the break-even proportionally. *Red flag:* using a single average cost when the two channels differ by an order of magnitude.

**Q14. What breaks if the experiment randomised sessions but the model scores users?**
Two things. Contamination: a user in the control group can be treated in another session, so the "control" is partly treated and the estimated effect is attenuated by the contamination share. And grain mismatch: the model's decision applies to a user, while the design identifies an effect per session, so the aggregation from session to user requires an assumption about how sessions combine. Check the contamination rate by measuring treatment exposure in the control arm; a nonzero rate is the diagnostic. *Red flag:* reporting a per-user CATE from a session-randomised test without measuring exposure.

**Q15. Why is the 5.0 pp top-decile estimate probably too high, and by how much?**
Because the decile was selected for having the largest measured effect among ten noisy estimates, so it is the maximum of ten draws and is biased upward. Under the normal model of 4.4.4 with mean 1.06 pp and reliability 0.60, the expected true effect is `1.06 + 0.60 × (5.00 − 1.06) = 3.42` pp. The check is a repeated-split reliability estimate; the consequence is that the forecast should use 3.42, which still clears 2.50 but with a third of the headroom. *Red flag:* defending the 5.0 by pointing at its t-statistic, which addresses sampling error of the estimate and not selection of the maximum.

**Takeaway:** the answers that distinguish candidates are the ones that separate the estimand from the estimator and the statistic from the decision.

## 12. Quick Reference Card

| Field | Rule |
|---|---|
| Estimand | `tau(x) = E[Y(1) - Y(0) | X=x]`; state population, window, outcome unit. The ITE is never identified. |
| Identification | Consistency, SUTVA, ignorability, positivity. Randomisation gives the first three plus ignorability unconditionally; observational data gives none of them for free. |
| Estimator | Balanced trial + small effect → S-learner with explicit interaction; unbalanced arms → X-learner; observational → DR-learner with overlap checks; per-unit intervals → causal forest. Always cross-fit nuisances. |
| Evaluation | Untouched holdout, Qini with permutation band, monotone decile table with `n` and SE per bin, CATE calibration slope near 1. |
| Decision | Treat iff `tau(x) > c_t/v`. In the worked case `1.50/60 = 0.025`, so deciles 1–3 at `+27,000` versus `−86,400` for treat-all. |
| Optimism | In-sample Qini is positive under a constant effect. Shrink selected extremes by `rho`: `1.06 + 0.60 × 3.94 = 3.42` pp. |
| Failure to check first | Level-based targeting: same spend, `+9,000` instead of `+27,000`; baseline-based targeting, `−28,200`. |
| Cost | Scoring is minutes; the campaign is the budget; evaluation error of 0.4 in `rho` costs 78,240 a year on this base. |
| Evidence | DERIVED identification and arithmetic; ILLUSTRATIVE decile inputs; SOURCED literature results; OPEN library metric defaults. |
