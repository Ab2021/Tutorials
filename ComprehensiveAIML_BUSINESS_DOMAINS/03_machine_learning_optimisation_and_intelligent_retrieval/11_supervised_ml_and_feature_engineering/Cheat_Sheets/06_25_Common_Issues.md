---
title: "Common issues: supervised ML"
topic: "Defect catalogue for supervised tabular models: leakage, metric misuse, threshold errors, and feature-engineering traps"
collection: "supervised-ml-and-feature-engineering"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Common issues: supervised ML

> **Evidence:** every formula in Sections 4, 5, 9 and 10 is DERIVED and every arithmetic step is shown so it can be recomputed by hand. Every population, count, cost and score below is a constructed teaching value and is labelled ILLUSTRATIVE; no portfolio, queue, vendor benchmark or organizational result is described or claimed. The catalogue in Section 8 names mechanisms, not incidents: none of the twenty-eight entries reports something that happened at a named firm.

> **Scope boundary.** This file assumes you already know how to fit a model. It is about the class of failures that survive a green test suite: the model trains, the metric prints a number, the number is wrong, and the wrongness comes from the pipeline around the estimator rather than from the estimator. Model-family selection lives in `03_Linear_Trees_and_Boosting.md`; split construction in `01_Model_Validation_and_Error_Analysis.md`; skew and calibration mathematics in `05_Imbalance_and_Calibration.md`; formulation in `02_Problem_Formulation_and_Losses.md`. Those sheets teach the correct construction. This one teaches the recognition of the incorrect one, which is a different skill and is examined separately.

## 1. File Header

### 1.1 The decision this sheet changes

A supervised model that underperforms is almost never fixed by a better estimator. In the reference population used throughout this sheet — 2,000 applicants, 224 observed defaults, 22 candidate features — the gap between a competent gradient-boosted model and a well-tuned logistic regression is roughly 0.01 to 0.02 AUC. The gap between a model evaluated with a random split on time-ordered rows and the same model evaluated with a forward split is routinely 0.05 to 0.15 AUC. The second gap is four to ten times the first, and it is produced entirely by the evaluation harness. Time spent choosing an estimator before the harness is clean is therefore spent on the smaller term.

The concrete decisions this sheet is written to settle:

1. When a validation number is not reproducible in production, which of the four defect families is responsible, and which single computation separates them.
2. When two models differ by 0.013 AUC, whether that difference exceeds the selection noise you have already paid for.
3. When a score must be turned into an action, whether the threshold is in the right space — the trained space, the resampled space, or the deployed prior.
4. When a feature engineering step looks like it improved the model, whether the improvement is information or artefact.

Each of those is a different owner's decision. The 1,000:200 cost ratio used in the worked example implies a break-even probability of 0.1667, and a team that cannot state that number cannot defend any threshold it ships.

### 1.2 Audience and assumed background

Two readers. The first owns a model in production and has a dashboard that disagrees with reality. The second is preparing an evaluation for review and needs to know which of their own numbers will be challenged first. Both need a mechanism plus one discriminating computation; neither needs a list of best practices, because best practices without a diagnostic cannot be applied to a defect you have not yet named.

Assumed competence: you can compute a mean, a standard deviation and a weighted average by hand; you can read a contingency table and derive precision from it; you can explain why precision depends on prevalence; you can distinguish event time from processing time; you can state the units of every column you use. If precision-versus-prevalence is not yet solid, read `05_Imbalance_and_Calibration.md` sections 3 and 4 first — six of the twenty-eight entries here are corollaries of it.

### 1.3 Category vocabulary

Four categories exhaust the failure space observed in tabular supervised work, and the category determines who fixes the defect and which wrong first move is most likely.

| Category | Meaning | Usual owner | Characteristic wrong first move |
|---|---|---|---|
| `VIOLATED_ASSUMPTION` | the estimator's premises, or the split's premises, do not hold for this data | data owner plus method owner | add regularisation |
| `NUMERICAL_EDGE` | arithmetic is undefined, saturated, clipped or discretised at a boundary | engineer | collect more data |
| `SCALE_FAILURE` | mechanism correct, volume or cardinality breaks it | platform owner | buy compute |
| `CONCEPT_CONFUSION` | the reported quantity is not the quantity the decision needs | reviewer | rename the metric |

The taxonomy is not decoration. It predicts the failure of the first fix. Adding L2 regularisation to a model whose training rows contain the outcome-derived feature will shrink the leaked coefficient and *improve* the validation number, which makes the wrong fix look like a success for one cycle. Nothing in the `NUMERICAL_EDGE` family is repaired by more rows, because the edge case appears in the extra rows too.

### 1.4 Severity rubric

Severity here is not "how bad the model is" but "how long the error survives undetected", because a quiet error that persists for a quarter costs more than a loud one a human catches in an hour.

| Severity | Definition | Response window |
|---|---|---|
| S1 | produces confidently wrong decisions for a whole population, or invalidates a shipped threshold | same day |
| S2 | biases a reported number by more than its own confidence interval | this sprint |
| S3 | degrades a diagnostic without changing the decision | next review |
| S4 | documentation or reproducibility defect only | next edit |

The 0.0159 standard error of a paired AUC difference calibrates S2: a bias larger than that is S2, because it moves the reported number by more than the noise band the number is quoted with.

### 1.5 How to use the catalogue

Find the symptom in the Section 8.1 summary, read the full entry, run the discriminating check *before* changing any parameter. Changing a threshold is the single most common wrong first move, because it reframes every failure as a calibration failure; in this catalogue only five of twenty-eight entries (CI-009 through CI-013) are genuinely in that family.

Report a defect by identifier and by the check that was run. "The model is bad" is not a defect report; "CI-004, preprocessing fitted before the split, reproduced by refitting the scaler inside each fold and observing AUC fall from 0.71 to 0.66" is.

### 1.6 What this sheet does not cover

Online experimentation and uplift measurement; causal attribution of a decision to an outcome; fairness certification under a legal standard; infrastructure failures of the scoring service (timeouts, partial batches, schema drift at the transport layer); and deep models on unstructured input. Those are separate sheets in Parts 03 and 04. The catalogue here concerns the *inference* being wrong while the *system* is up and answering.

### 1.7 On the number twenty-eight

Twenty-eight is not completeness. It is the number of distinct mechanisms that survived merging: any two entries whose discriminating check was identical were collapsed, and any entry without a check that changes a decision was deleted. Two entries that were merged in the final pass are worth naming so a reader does not look for them: "imputer fitted on all data" and "target encoder fitted on all data" were folded into CI-004, because the check is the same computation (refit inside the fold and remeasure) even though the mechanism differs in severity. A longer list would have been easier to write and less useful.

## 2. Concept Map / Mental Model

### 2.1 Five layers, and the rule that a defect cannot be repaired below its layer

Every supervised pipeline is five layers. A defect introduced at layer `i` cannot be corrected by any change at layer `j > i`; it can only be corrected at `i` or masked at `j`. Arguments between team members about a bad model are usually two people describing defects at different layers with the same vocabulary.

```text
LAYER 0  POPULATION & LABEL        LAYER 1  SPLIT              LAYER 2  TRANSFORM
who is in the table, when the       which rows train, which     impute, scale, encode,
label became true, censoring,       validate, grouping,         bin, aggregate, lag
label definition version            time direction              selection, target enc.
        |                                   |                           |
        v                                   v                           v
LAYER 3  ESTIMATOR                  LAYER 4  DECISION
loss, regularisation, capacity,     threshold in the deployed
class weighting, early stop         prior space, capacity,
                                    cost ratio, review queue
```

The layers are ordered by how silently they corrupt everything downstream. A label that becomes true ninety days after the feature snapshot (layer 0) produces a model that looks weak at layer 3 and gets regularised harder; the regularisation shrinks the coefficients, the AUC falls further, and the team concludes that the signal is thin. The signal was never thin; the label horizon was mismatched to the feature horizon.

### 2.2 The four questions in order

When a validation number fails to reproduce, ask in this order and stop at the first yes:

1. *Can this model, in principle, be right?* If any feature is computed after the decision time, no metric can rescue it (CI-002, CI-008).
2. *Is the validation set independent of the training set?* Rows, groups, and preprocessing state all count as dependence (CI-003, CI-004, CI-005).
3. *Is the label mature and stably defined across the window?* If not, the denominator of every rate is wrong (CI-006, CI-007).
4. *Is the reported metric the one the decision needs, at the deployed prior and the deployed operating point?* (CI-023 through CI-028).

Steps 1 and 2 explain the majority of large offline-online gaps. Steps 3 and 4 explain most disputes in review meetings, because they are invisible in the offline table and visible in the business result.

### 2.3 The gap decomposition identity

The single most useful mental device in this sheet is that the offline-to-online gap decomposes into terms you can measure separately rather than one mystery:

```text
gap  =  leakage term        (feature available offline but not at decision time, or label reused)
      + dependence term     (test rows not independent of train rows)
      + prior term          (pi_train != pi_deploy, and the metric is prior-sensitive)
      + definition term     (label or feature computed by a different rule in each environment)
      + noise term          (the difference is inside the metric's standard error)
```

Each term has a different fix and only the last is benign. A team that treats the whole gap as "the model does not generalise" will try to fix a prior term by adding data, which adds leakage instead: more rows means more duplicated groups straddling the split.

### 2.4 Why metrics lie in a specific order

Metrics fail in a predictable sequence as prevalence falls, and knowing the sequence tells you which metric to distrust first.

Accuracy is the first to fail, because it is dominated by the majority class; at `pi = 0.0112` a constant-negative model scores 0.9888. Precision is next, because it is prior-sensitive by construction; a model with `tpr = 0.66` and `fpr = 0.37` has PPV 0.185 at `pi = 0.112` and PPV 0.020 at `pi = 0.011`. Recall survives prior shift (it is computed within positives) but collapses under label immaturity. AUC survives prior shift and label immaturity, which is exactly why it is over-trusted: it is the last metric standing and therefore gets asked questions it cannot answer, such as "how many true positives at the capacity we have".

### 2.5 The attribution diagram

```text
                    offline number good, online bad
                                 |
             +-------------------+-------------------+
             |                   |                   |
      feature not available   test dependence     prior differs
      at decision time        (rows/groups/       (metric is
             |                transform)          prior-sensitive)
             v                   v                   v
      compare feature        refit transform     recompute metric
      timestamps to the      inside folds;       at deploy pi;
      decision timestamp     group split         if it matches, no
             |                   |               model change needed
             v                   v                   v
        CI-002, CI-008      CI-003/4/5           CI-023..CI-026
```

If all three branches come back clean, the remaining explanation is usually the noise term, and the honest response is to widen the reported interval rather than to hunt for a cause.

## 3. Foundational Definitions

Each definition states what the quantity is, its units or type, and the specific way it is misused in this catalogue. Misuse is included because a definition without its failure mode does not help a reviewer.

**Decision time (`t_d`).** The instant at which the prediction must exist, as a timestamp with a timezone. Type: instant. Misuse: defining it as the *row creation* time in the warehouse, which is later than the business event; every feature computed between those two instants is then invisible in the offline table and absent in production. CI-008 is this defect.

**Label maturity lag (`L`).** The elapsed time between `t_d` and the instant at which the outcome becomes observable and recorded, in days. Type: duration. A window shorter than `L` contains rows whose label is still `unknown` but is stored as `0`. Misuse: treating `0` as "did not default" when it means "not yet knowable". CI-006.

**Outcome horizon (`H`).** The forward window over which the label is defined, in days, measured from `t_d`. A model with `H = 365` and `L = 90` cannot be evaluated on the most recent 90 days at all; the usable evaluation set ends at `today - L` and the label is only fully observed for rows before `today - L`.

**Base rate (`pi`).** Fraction of rows in a declared population and window whose label is 1. Type: dimensionless, `[0,1]`. In the reference population `pi = 224/2000 = 0.112`. Misuse: quoting a base rate without its window, which makes two rates incomparable across a change in the label definition.

**True positive rate (`tpr`), false positive rate (`fpr`).** `tpr = TP/(TP+FN)` and `fpr = FP/(FP+TN)`, both dimensionless, both conditional on the true class and therefore invariant to prior shift. Misuse: quoting them from a resampled test set, where the ratio `n_neg/n_pos` no longer matches deployment and any cost or capacity computation built on them is scaled wrongly.

**Positive predictive value (`PPV`, precision).** `TP/(TP+FP)`. Prior-sensitive. The identity in 4.2 is the most-used result in this sheet, because it lets a reviewer reconstruct a production precision from an offline `tpr`, `fpr` and the deployed `pi` without touching the model.

**Negative predictive value (`NPV`).** `TN/(TN+FN)`. At `pi = 0.112` with `tpr = 0.66` and `fpr = 0.37`, NPV is 0.937 — which is why a reviewer who samples only approved applications concludes the model is excellent.

**Calibration.** For each predicted probability `p`, the observed rate among rows receiving `p` equals `p`. Formally `E[y | s(x) = p] = p`. Type: a property of the score-to-probability map, not of the ranking. Misuse: inferring calibration from a high AUC, which says nothing about it; a monotone transform of a calibrated score keeps AUC and destroys calibration.

**Discrimination.** The ability to rank positives above negatives, summarised by AUC. Invariant to any strictly increasing transform. Misuse: reading discrimination as a business quantity; two models with equal AUC can have very different expected cost at the operating point, because AUC integrates over all thresholds and the decision uses one.

**Threshold (`t`).** The score value at which the action changes. Not a property of the model; a property of the cost ratio and capacity. Misuse: carrying a threshold across a change of prior, class weighting, or score transform — CI-011.

**Break-even probability (`p*`).** `C_FP/(C_FP + C_FN)`, dimensionless. The probability at which approving and rejecting have equal expected cost. With `C_FN = 1000` and `C_FP = 200`, `p* = 0.1667`.

**Events per variable (`EPV`).** `n_pos / d`, where `d` is the number of estimated parameters. Type: dimensionless count. The conventional floor is 10. In the reference population, 224 events over 22 features gives 10.2, so the model sits exactly at the boundary and any added interaction term crosses it.

**Leakage.** Any path by which information unavailable at `t_d`, or information from the evaluated row's own label, reaches the fitted model. Not a synonym for "the model memorised"; there are five distinct mechanisms (Section 8.2) and they require different checks.

**Dependence.** Correlation between training and validation rows sufficient to make the validation error estimate optimistic. Distinct from leakage: no future information crosses, but duplicate or same-entity rows make the two sets statistically non-independent.

**Score band.** A contiguous range of scores treated as one operating decision. In the reference table, five bands of 400 rows each, with observed default rates 3%, 6%, 10%, 15%, 22%. Bands are how a calibration table is actually checked, because a per-score check has too few rows.

**Population stability index (`PSI`).** `sum_i (a_i - e_i) * ln(a_i/e_i)` over bins, where `e_i` is the expected (reference) share and `a_i` the actual (current) share. Unitless. Conventional reading: below 0.10 stable, 0.10 to 0.25 moderate, above 0.25 major.

**Weight of evidence (`WOE`), information value (`IV`).** `WOE_i = ln((bad_i/n_bad)/(good_i/n_good))`; `IV = sum_i (bad_i/n_bad - good_i/n_good) * WOE_i`. Both unitless. IV above 0.3 is "strong" by convention and is also the level at which a feature deserves a leakage audit rather than a celebration.

**Selection noise.** The inflation of a reported metric caused by choosing the maximum over `k` candidates. Its expected size is `sigma * E[max Z_k]`, not zero, and it is the reason CI-017 exists.

**Reproducibility key.** The tuple of (data snapshot id, split seed, transform version, library version, hyperparameters) that determines a reported number. A number without its key is an anecdote; this is what CI-027 in the sibling catalogue calls a measurement change.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: the cost-optimal threshold

```text
FORMULA
        p*  =  C_FP / (C_FP + C_FN)

WHERE   p*    break-even probability, dimensionless, in (0,1)
        C_FP  cost of a false positive, currency units per row (approve a bad applicant)
        C_FN  cost of a false negative, currency units per row (reject a good applicant)
        The score s(x) is a calibrated estimate P(y=1 | x), and the action is approve for
        s < p*, reject for s >= p*.

DERIVATION
        Expected cost of approving a row with probability p:
            E[C | approve] = p * C_FN + (1 - p) * 0
        Expected cost of rejecting:
            E[C | reject]  = (1 - p) * C_FP + p * 0
        Set equal and solve:
            p*C_FN = (1 - p*)C_FP
            p*(C_FN + C_FP) = C_FP
            p* = C_FP / (C_FP + C_FN)

CONSTRAINT
        Requires (i) a calibrated score, not merely a ranking; (ii) costs that are constant
        per row, i.e. no capacity limit and no interaction between decisions; (iii) the cost
        of a false negative independent of which good applicant is rejected.

NUMERICAL EXAMPLE
        C_FN = 1000 (a defaulted exposure of 1,000 currency units)
        C_FP = 200  (a declined good applicant costs 200 in margin)
        p* = 200/(200 + 1000) = 200/1200 = 0.16666...
        A row with s = 0.15 is approved: expected cost 0.15*1000 = 150 against
        (1-0.15)*200 = 170 for rejecting. Approving saves 20.
        A row with s = 0.22 is rejected: 0.22*1000 = 220 against 0.78*200 = 156.
        The threshold where the two costs meet is exactly 0.1667, at which
        0.1667*1000 = 166.7 = 0.8333*200.

INTERPRETATION
        The threshold is a ratio of costs, not a property of the model. Moving C_FP from
        200 to 282.05 (= 0.22*1000/0.78) makes a row at s = 0.22 exactly indifferent, so a
        41% error in the cost estimate does not change the ranking of decisions but does
        change which band is marginal. Publishing the threshold without the cost ratio
        publishes an unfalsifiable number.

VALIDITY CONDITION
        Holds iff the score is calibrated on the deployed population and the decision is
        independent of other decisions. Fails under a capacity constraint (fix the count and
        sort by score instead) and under a per-row cost that varies (use a per-row p*).
```

### 4.2 General case: reconstructing precision from the prior

```text
FORMULA
        PPV  =  tpr * pi / ( tpr * pi + fpr * (1 - pi) )
        NPV  =  (1 - fpr)(1 - pi) / ( (1 - fpr)(1 - pi) + (1 - tpr) * pi )

WHERE   tpr = P(flag | y=1), fpr = P(flag | y=0), pi = P(y=1), all dimensionless.
        The identity holds for any threshold and any population, and requires only that
        tpr and fpr were estimated on a population whose *conditional* rates transfer.

DERIVATION
        By Bayes: PPV = P(y=1 | flag) = P(flag | y=1)P(y=1) / P(flag)
        and P(flag) = P(flag|y=1)P(y=1) + P(flag|y=0)P(y=0)
                   = tpr*pi + fpr*(1-pi).
        Substituting gives the first line; the NPV line is the same expansion on the
        complement event.

CONSTRAINT
        tpr and fpr must be measured at the *deployed* operating point. A model evaluated at
        its default 0.5 cut has different rates from one deployed at 0.1667, and substituting
        the wrong pair is the most common arithmetic error in a model review.

NUMERICAL EXAMPLE
        From the reference evaluation at the KS-optimal cut, which flags the top 800 rows:
        TP = 88 + 60 = 148, FP = 800 - 148 = 652, so tpr = 148/224 = 0.6607,
        fpr = 652/1776 = 0.3671, pi = 0.112.
        PPV = (0.6607*0.112) / (0.6607*0.112 + 0.3671*0.888)
            = 0.0740 / (0.0740 + 0.3260) = 0.0740/0.4000 = 0.185
        Now move the same model to a population with pi = 0.011 (a 10x rarer event):
        PPV = 0.00727/(0.00727 + 0.3632) = 0.0196
        NPV = (0.6329*0.989)/(0.6329*0.989 + 0.3393*0.011) = 0.9941

INTERPRETATION
        A tenfold fall in prevalence cuts PPV by a factor of 9.4 with no change to the model.
        Any review that compares a pilot precision to a production precision without
        recomputing pi is comparing two different quantities that share a name. This single
        identity resolves the majority of "the model degraded in production" claims.

VALIDITY CONDITION
        Holds iff the model's conditional rates transfer to the new population. It fails when
        the population change is a change in the *covariate* distribution rather than the
        prior alone, because then tpr and fpr move too; the check is to re-estimate tpr and
        fpr on a labelled sample of the new population and see whether they moved.
```

### 4.3 Variants

**4.3.1 Threshold after class weighting.** Training with positive class weight `w` multiplies the positive term of the loss; the fitted score then approximates `P(y=1|x)` under an effective prior shifted by `w`. If the trained threshold is `t_w` in the weighted space, the equivalent threshold in the unweighted space is

```text
        t  =  t_w / ( w - (w - 1) * t_w )
```

At `w = 9` and `t_w = 0.1667`: `t = 0.1667/(9 - 8*0.1667) = 0.1667/7.6667 = 0.02174`. Check: `0.02174*9 = 0.1957`, and `0.1957/(0.1957 + 0.9783) = 0.1667` recovers the weighted threshold. So a team that trains with 9:1 weighting and then applies 0.1667 to the *weighted* output is rejecting at an implied unweighted probability of 0.0217, roughly eight times too aggressively. This is CI-011, and it is the single highest-frequency defect in resampled pipelines because no error is raised.

**4.3.2 Prior-shift correction of the intercept.** For a logistic model, changing the prior from `pi_0` to `pi_1` while holding the covariate-conditional likelihood fixed requires only an intercept offset:

```text
        delta  =  logit(pi_1) - logit(pi_0)
```

From `pi_0 = 0.112` to `pi_1 = 0.30`: `logit(0.112) = ln(0.112/0.888) = -2.0705`; `logit(0.30) = ln(0.30/0.70) = -0.8473`; `delta = +1.2232`. Verify: the row previously at `p = 0.112` now has logit `-2.0705 + 1.2232 = -0.8473`, so `p = 0.30`. This is why a deployment that changes the eligible population must re-solve the threshold, not just re-fit.

**4.3.3 Brier decomposition.** For `K` score bands with share `w_k`, mean predicted `p_k` and observed `o_k`:

```text
        BS  =  REL - RES + UNC
        REL =  sum_k w_k (p_k - o_k)^2        reliability, lower is better
        RES =  sum_k w_k (o_k - o_bar)^2      resolution, higher is better
        UNC =  o_bar (1 - o_bar)              irreducible
```

This is the only decomposition here that separates "the probabilities are wrong" (REL) from "the model separates the classes" (RES). Two models can have identical Brier scores with opposite defects: one sharp and miscalibrated, one flat and well calibrated.

**4.3.4 Selection-noise inflation.** For `k` candidate configurations, each evaluated with metric standard error `sigma`, the expected value of the maximum of `k` independent standard normals is `E[max Z_k]`, so the reported best metric is inflated by approximately `sigma * E[max Z_k]`. For `k = 30`, `E[max Z] = 3.96`; at `sigma = 0.008` the inflation is 0.0317 AUC — larger than the 0.013 difference used as a comparison in Section 7.

### 4.4 Limits and counterexamples

*Counterexample 1 — capacity instead of cost.* Suppose the review team can process 400 applications per week regardless of cost. Then `p*` is irrelevant: the correct policy is "flag the top 400 by score", and the relevant metric is precision at 400, `88/400 = 0.22` at the reference point. Optimising a cost-weighted threshold under a hard capacity produces a queue that violates the capacity.

*Counterexample 2 — variable cost per row.* If a default costs 1,000 for one product and 40,000 for another, a single threshold is wrong for both. The correct object is a per-row `p*_i = C_FP/(C_FP + C_FN,i)`, and the cost curve is no longer monotone in score. The failure looks like "the threshold is miscalibrated" and is actually a heterogeneous-cost defect (CI-026).

*Counterexample 3 — non-probabilistic score.* For a margin-based score (SVM distance, tree leaf value), `p*` has no meaning until a calibration map is fitted. Applying 0.1667 to a raw margin is a category error, not a numerical one.

*Counterexample 4 — the asymmetric extremes.* As `C_FP -> 0`, `p* -> 0` and the optimal action is to approve everything, at which point the model has no value; the expected cost equals the unconditional default cost `pi*C_FN = 112` per applicant. As `C_FN -> 0`, `p* -> 1` and the optimal action is to reject everything, costing `(1-pi)*C_FP`. A cost model that drives the policy to an extreme is a signal that the cost estimate is wrong, not that the model is good.

*Counterexample 5 — the identity fails under covariate shift.* Section 4.2 assumed conditional rates transfer. If the new population has a different joint distribution of features and label, `tpr` and `fpr` move and the reconstructed PPV is wrong in either direction. The discriminating check is a small labelled sample from the new population: if the empirical `tpr` differs from the training `tpr` by more than two standard errors, covariate shift is present and the identity is invalid.

### 4.5 Variants continued: more than two actions

Most teaching examples are binary, and most production decisions are not. When the action set has `K > 2` elements, the break-even argument generalises to a comparison of costs, and the error it produces is different in kind.

```text
FORMULA   choose action a  =  argmin_a  sum_c  p_c * C(a, c)
WHERE     p_c  calibrated probability of class c, sum_c p_c = 1
          C(a, c)  cost of taking action a when the true class is c, currency per row
CONSTRAINT the costs must be stated per (action, class) pair, not per class alone; a matrix
          with a dominant row makes the decision trivial and the model worthless
NUMERICAL EXAMPLE
        Three classes: clear (0.70), watch (0.20), decline (0.10) on one row.
        Action costs: approve-if-clear C = 0 for correct, 1000 for a default;
                      manual review costs 40 always and prevents nothing;
                      decline costs 200 if the customer was good, 0 if bad.
        approve: 0.70*0 + 0.20*1000 + 0.10*1000 = 300
        review:  0.70*40 + 0.20*40 + 0.10*40 = 40    (review resolves, so no loss)
        decline: 0.70*200 + 0.20*200 + 0.10*0 = 180
        Minimum is review at 40.
INTERPRETATION
        The optimal action is neither of the extremes and is not monotone in the probability
        of the worst class: at p_decline = 0.10 review wins, at 0.30 decline wins
        (0.70*200 + 0.30*0 = 140 against 40), and at 0.02 approve wins
        (0.98*0 + 0.02*1000 = 20 against 40). A binary threshold cannot express this, which
        is why collapsing a three-action decision into a score plus a cut loses money in a
        way no AUC can show.
VALIDITY CONDITION
        Holds iff p is calibrated and the cost matrix is complete. Incomplete matrices are
        the norm in practice; the check is to name the missing pair and see whether its
        omission changes the argmin on a sample of 100 rows.
```

The failure that this variant explains is CI-023's cousin: a team reports "accuracy 0.94" on a three-way decision where two of the three actions are almost never chosen, and the model is effectively a binary classifier with a decorative third class. The cost matrix is the artefact that forces the third action to justify itself.

## 5. Worked Numerical Example

The reference population is ILLUSTRATIVE: 2,000 applicants at a single decision date, 224 observed defaults (`pi = 0.112`), 22 candidate features, a score grouped into five bands of 400 rows. Costs `C_FN = 1000`, `C_FP = 200`.

### 5.1 The band table

| Band | Score range | Rows | Defaults | Observed rate | Mean score |
|---|---|---:|---:|---:|---:|
| 5 (worst) | 0.18 – 0.25 | 400 | 88 | 0.220 | 0.22 |
| 4 | 0.13 – 0.18 | 400 | 60 | 0.150 | 0.15 |
| 3 | 0.10 – 0.13 | 400 | 40 | 0.100 | 0.10 |
| 2 | 0.06 – 0.10 | 400 | 24 | 0.060 | 0.06 |
| 1 (best) | 0.02 – 0.06 | 400 | 12 | 0.030 | 0.03 |

Totals: 2,000 rows, 224 defaults, 1,776 goods.

### 5.2 Derivation 1 — cumulative cost of each approval policy

```text
FORMULA   cost(k) = (defaults among the k worst bands, approved) * C_FN
                  + (goods among the remaining bands, wrongly rejected) * C_FP
WHERE     approve bands 1..(5-k) and reject the worst k bands; k = number of bands rejected
CONSTRAINT all 2,000 rows are decided; costs are constant per row
NUMERICAL EXAMPLE
        k = 0 (approve all):  224*1000 + 0 = 224,000
        k = 1 (reject band 5): 136*1000 + 312*200 = 136,000 + 62,400 = 198,400
        k = 2 (reject 5 and 4): 76*1000 + 652*200 = 76,000 + 130,400 = 206,400
        k = 3: 36*1000 + 1012*200 = 36,000 + 202,400 = 238,400
        k = 4: 12*1000 + 1388*200 = 12,000 + 277,600 = 289,600
        k = 5 (reject all): 0 + 1776*200 = 355,200
INTERPRETATION
        The minimum is at k = 1: 198,400, i.e. 99.2 per applicant, against 112 per applicant
        for approving everyone. The saving is 25,600 over 2,000 applicants = 12.80 per
        applicant. Compare that with a manual review cost of 40 per application: the model is
        worth 32% of one manual review per case, which is the number to put in the business
        case, not AUC.
```

The check on this derivation is Section 4.1: `p* = 0.1667` lies between band 5's rate (0.22) and band 4's rate (0.15), so the optimum must reject band 5 only. The two computations agree, which is the point of doing both.

### 5.3 Derivation 2 — threshold sensitivity

The threshold band is where the two neighbouring policies trade places. Band 4 becomes indifferent to rejection when `C_FP` satisfies `0.15*C_FN = (1-0.15)*C_FP`, i.e. `C_FP = 0.15*1000/0.85 = 176.47`. Band 5 becomes indifferent at `C_FP = 0.22*1000/0.78 = 282.05`.

```text
FORMULA   C_FP*  =  p_band * C_FN / (1 - p_band)
WHERE     p_band is the band's observed default rate; C_FN held at 1000
CONSTRAINT the band is treated as homogeneous at its observed rate
NUMERICAL EXAMPLE
        band 4: 0.15 * 1000 / 0.85 = 176.47
        band 5: 0.22 * 1000 / 0.78 = 282.05
        cost of the k=1 policy at C_FP = 176: 136,000 + 312*176 = 190,912
        cost of the k=2 policy at C_FP = 176:  76,000 + 652*176 = 190,752
INTERPRETATION
        At C_FP = 176 the k=2 policy is cheaper by 160, so the decision flips. The shipped
        policy tolerates a 12% downward error in C_FP and a 41% upward error before flipping
        the other way. A cost estimate with wider-than-40% uncertainty on a 200-unit figure
        does not support a band-level threshold at all: it supports a range of admissible
        thresholds, published as a range.
```

### 5.4 Derivation 3 — calibration and Brier decomposition

```text
FORMULA   BS = sum_k w_k (p_k - o_k)^2 + sum_k w_k o_k(1-o_k) - sum_k w_k (o_k - o_bar)^2
WHERE     w_k = 0.2 for each of five equal bands; o_bar = 0.112
CONSTRAINT bands are equal-sized and the score is the band mean
NUMERICAL EXAMPLE
        REL: band 1 (0.03 vs 0.03) 0; band 2 (0.06 vs 0.06) 0; band 3 (0.10 vs 0.10) 0;
             band 4 (0.15 vs 0.15) 0; band 5 (0.22 vs 0.22) 0  -> REL = 0
        RES: 0.2*[(0.03-0.112)^2 + (0.06-0.112)^2 + (0.10-0.112)^2
                  + (0.15-0.112)^2 + (0.22-0.112)^2]
             = 0.2*[0.006724 + 0.002704 + 0.000144 + 0.001444 + 0.011664]
             = 0.2*0.022680 = 0.0045360
        UNC: 0.112*(1 - 0.112) = 0.112*0.888 = 0.0994560
        BS  = 0 - 0.0045360 + 0.0994560 = 0.0949200
        Cross-check by the direct route: 0.2*[0.03*0.97 + 0.06*0.94 + 0.10*0.90
                  + 0.15*0.85 + 0.22*0.78] = 0.2*[0.0291+0.0564+0.09+0.1275+0.1716]
                  = 0.2*0.4746 = 0.09492.   Agrees to five decimals.
        Brier skill score = 1 - 0.09492/0.099456 = 0.0454
INTERPRETATION
        The model removes 4.5% of the Brier score available from always predicting the base
        rate. That is a small number and it is the honest one: strong discrimination at
        0.689 AUC coexists with weak squared-error skill when prevalence is 11.2%, because
        most of the variance is irreducible.
```

### 5.5 Independent re-check of the band table

A second route to the same numbers, using only the totals: the five bands must sum to 224. `88+60+40+24+12 = 224`. The goods must sum to 1,776: `(400-88)+(400-60)+(400-40)+(400-24)+(400-12) = 312+340+360+376+388 = 1776`. Sum of rows `5*400 = 2000`. The base rate `224/2000 = 0.112`. Every reported quantity in Section 5.2 can be reconstructed from the band column alone, which is the property a reader should demand before accepting a cost table in a review.

### 5.6 Sensitivity note

Three parameters move the answer, and their plausible ranges are asymmetric.

First, `C_FN`. If the true recovery rate on a default is 40%, the effective `C_FN` is 600, so `p* = 200/800 = 0.25` and the optimum becomes `k = 0` (approve everything), because band 5's 0.22 is below 0.25. The entire model becomes worthless under a recovery assumption nobody wrote down.

Second, `C_FP`. A margin of 200 on an application with a 2-year life and a 12% attach rate is closer to 48 in present value, which gives `p* = 48/1048 = 0.0458`, and the optimum becomes `k = 4` (reject bands 4 and 5) at cost `12,000 + 1,388*48 = 78,624` — a much larger apparent saving that exists mostly because the cost estimate shrank.

Third, the band rates themselves. Each band has 400 rows, so the standard error of a band rate at 0.22 is `sqrt(0.22*0.78/400) = 0.0207`; band 5's 0.22 is therefore `0.22 ± 0.041` at 95%. The band ordering is secure (0.22 versus 0.15 is 2.6 standard errors of the difference, since the gap 0.07 is measured against `sqrt(0.0207^2 + 0.0179^2) = 0.0274`) but the exact threshold is not, and the honest artefact is a threshold range of roughly 0.15 to 0.18 rather than a single value.

### 5.7 Derivation 4 — average precision from the same bands

```text
FORMULA   AP = sum_k (r_k - r_{k-1}) * precision_k
WHERE     rows are added worst-band-first; r_k is cumulative recall, precision_k is
          cumulative precision after adding band k
CONSTRAINT this is the step-wise definition, which equals the area under the interpolated
          PR curve only when the curve is monotone decreasing, which it is here
NUMERICAL EXAMPLE
        after band 5: 88/400 rows, precision 0.2200, recall 88/224 = 0.3929
        after band 4: 148/800, precision 0.1850, recall 148/224 = 0.6607
        after band 3: 188/1200, precision 0.1567, recall 0.8393
        after band 2: 212/1600, precision 0.1325, recall 0.9464
        after band 1: 224/2000, precision 0.1120, recall 1.0000
        AP = 0.3929*0.2200
           + (0.6607-0.3929)*0.1850 = 0.2679*0.1850 = 0.0496
           + (0.8393-0.6607)*0.1567 = 0.1786*0.1567 = 0.0280
           + (0.9464-0.8393)*0.1325 = 0.1071*0.1325 = 0.0142
           + (1.0000-0.9464)*0.1120 = 0.0536*0.1120 = 0.0060
        AP = 0.0864 + 0.0496 + 0.0280 + 0.0142 + 0.0060 = 0.1842
INTERPRETATION
        Average precision is 0.1842 against a prevalence of 0.112, a lift of 1.64. AP is the
        metric to quote for a screening decision and the reason is visible in the arithmetic:
        every term is weighted by the recall it buys, so the first band contributes 47% of the
        total. A model that is excellent in the first decile and useless afterwards can have
        high AP and modest AUC; the reverse also happens, and the pair of numbers says which.
```

Check: the five recall increments must sum to 1. `0.3929 + 0.2679 + 0.1786 + 0.1071 + 0.0536 = 1.0001`, the excess being rounding at the fourth decimal. The precision values are the cumulative `cumulative_defaults / cumulative_rows`, each reproducible from the band column: `148/800 = 0.185` and `188/1200 = 0.15667`.

Two consequences follow for a reviewer. First, AP is not comparable across populations with different prevalence, because the baseline moves; the useful comparison is AP divided by prevalence, here 1.64. Second, AP computed on a resampled evaluation set is meaningless, because the recall increments depend on `n_pos`, which resampling changed — this is the AP version of CI-011.

### 5.8 Derivation 5 — how much a doubling of data is worth

```text
FORMULA   m(n) = A - B * n^(-p)      power-law learning curve
WHERE     n = training rows; m = validation AUC; A = asymptotic ceiling; B = gap at n = 1;
          p = learning exponent, typically 0.3 to 0.7 for tabular supervised work
CONSTRAINT the functional form is a heuristic, not a law; it fits observed curves well over
          one to two orders of magnitude of n and fails when a regime change (a new feature
          source, a definition change) alters A
NUMERICAL EXAMPLE
        Fit A = 0.80, B = 6.36, p = 0.5 (ILLUSTRATIVE parameters chosen to reproduce a
        curve of the shape commonly observed at this sample size).
        m(2000)  = 0.80 - 6.36/44.72  = 0.80 - 0.1422 = 0.6578
        m(5000)  = 0.80 - 6.36/70.71  = 0.80 - 0.0900 = 0.7101
        m(20000) = 0.80 - 6.36/141.42 = 0.80 - 0.0450 = 0.7550
        m(80000) = 0.80 - 6.36/282.84 = 0.80 - 0.0225 = 0.7775
        m(320000)= 0.80 - 6.36/565.69 = 0.80 - 0.0112 = 0.7888
INTERPRETATION
        Going from 2,000 to 5,000 rows (+3,000 labels) buys 0.052 AUC. Going from 20,000 to
        80,000 (+60,000 labels) buys 0.0225. At an annotation cost of 3.50 per label, the first
        increment costs 10,500 for 0.052 and the second costs 210,000 for 0.0225 — 20 times
        the cost for less than half the gain. The break-even against a feature-engineering
        week is decided by that ratio, and a team that asks "should we get more data" without
        fitting a curve is guessing at a factor of 20.
```

The re-check is a limit case: as `n -> infinity`, `m -> A = 0.80`, so no quantity of data closes the remaining 0.02 to a perfect model. If a proposed data programme claims to reach 0.90 on this problem, the claim contradicts the fitted ceiling and one of the two is wrong. The counterexample where the form breaks is a new data *source* rather than more rows of the same source: adding a genuinely new feature moves `A`, and no amount of the old rows does.


## 6. Visual / Geometric Intuition

### 6.1 The ROC curve as a threshold sweep

```text
 fpr ->
 1.0 |                            *  (1,1) reject everything
     |                       *
     |                  *
 0.37| - - - - - - -* - - - - - -  deployed cut: tpr 0.6607, fpr 0.3671
     |            *
     |        *
     |    *
 0.0 *----+----+----+----+----+--  (0,0) approve everything
     0   0.2  0.4  0.66 0.8  1.0   tpr ->

 Area under this curve = 0.6890.  The dashed cross marks the shipped operating point,
 which is *not* the point that maximises any geometric criterion; it is the point where
 the cost slope equals the ROC tangent slope.
```

The geometric reading that matters: the optimal operating point on an ROC curve given a cost ratio is where the tangent has slope `(1-pi)/pi * C_FP/C_FN`. At `pi = 0.112` that factor is `7.93 * 0.2 = 1.586`, so the tangent is fairly steep and the optimum sits left of the 45-degree line's natural crossing. A team that picks the point closest to the top-left corner is optimising a different objective (Youden's J) and will be too aggressive.

### 6.2 The lift chart and the capacity argument

```text
 cumulative % of defaults captured
 100% |                                         *
      |                                  *
  75% |                          *
      |                  *
  50% |          *
      |     *
  39% |  *  <- 20% of the population captures 39.3% of defaults (88 of 224)
   0% +----+----+----+----+----+----+----+----+
      0   10   20   30   40   50   60   70  100   % of population, worst first
```

Lift at 20% is `0.393/0.20 = 1.96`. Lift is the right chart for a capacity-constrained decision and the wrong chart for a cost-constrained one, because it is invariant to the cost ratio; two models with identical lift curves can have different cumulative-cost curves once the band rates differ inside the first 20%.

### 6.3 The calibration curve and what ECE hides

Plot observed rate against mean predicted score per band. The reference table lies exactly on the diagonal, so REL is 0 and ECE is 0. Now consider the *same ranking* with compressed scores 0.06, 0.09, 0.12, 0.15, 0.18 instead of 0.03 to 0.22. REL rises to 0.00076, ECE to 0.024, Brier to 0.09568 — while AUC is unchanged at 0.6890, because AUC only sees the order.

The geometric reason is that compression is a monotone map that squeezes the score axis toward the middle; the *ranks* survive, the *heights* do not. A model can therefore pass every discrimination gate and fail calibration, and the failure is invisible on an ROC chart because the ROC chart has no vertical scale for probability.

### 6.4 The threshold-crossing picture

```text
 cost per applicant
  240 |*
      |  *
  200 |     *        <-- approve-all, 112 per applicant (horizontal)
      |        *
  176 |           *
      |              *
  150 |                 *     *
      |                    *      *
  112 +-----------------------+-----+-----+-----+-----+
        0.02 0.05 0.10 0.1667 0.22 0.30 0.40 0.50   predicted p

 The two lines are E[cost | approve] = p*1000 and E[cost | reject] = (1-p)*200.
 They cross at p = 0.1667. Left of the crossing, approving is cheaper; right, rejecting.
```

Every threshold defect in the catalogue is one of three things relative to this picture: the score axis is not a probability (so the crossing is at the wrong horizontal position), the cost lines have the wrong slopes, or there are ties at the crossing (so the vertical line is not a line but a step).

### 6.5 Feature-space intuition for leakage

A leaked feature is one whose value carries information only available after the outcome. Geometrically, the training points of the two classes become separable along that axis by construction, and the fitted decision boundary is a line in a direction that does not exist at scoring time. The signature in the fitted model is a coefficient so large that its standard error is comparable to it, and a training AUC above 0.95 on a problem where the base rate suggests 0.75 is the ceiling. In the reference population, a training AUC of 0.97 against a forward-split 0.66 is not a good model — it is a timestamp defect, and the ratio 0.97/0.66 tells you the leaked axis is doing most of the work.

### 6.6 The precision-recall plane, and why ROC flatters at low prevalence

ROC has a fixed frame: `fpr` runs 0 to 1 regardless of `pi`. The precision axis does not have a fixed frame — its floor is the prevalence itself. The same model, at the same `tpr = 0.6607` and `fpr = 0.3671`, traces a PR curve whose height depends entirely on the population it is drawn on:

```text
 pi        PPV at the deployed cut     PR floor
 0.011            0.0196                  0.011
 0.030            0.0527                  0.030
 0.050            0.0865                  0.050
 0.112            0.1850                  0.112
 0.300            0.4354                  0.300
 0.500            0.6428                  0.500
```

Two readings matter. First, the *vertical distance above the floor* is roughly constant — about 0.0085 at `pi = 0.011`, 0.073 at 0.112, 0.14 at 0.5 — so the model's marginal value is similar in all six populations while the PR curve looks catastrophically worse in the first. A team that compares a PR curve from a 1.1% population with one from a 30% population and concludes the model degraded has compared two charts with different frames. Second, the ROC curve is identical in all six rows, which is the precise sense in which ROC is prior-invariant and therefore unable to answer the question "how many true positives per hundred flags".

The practical consequence: quote AP normalised by prevalence (the lift, 1.64 here) when comparing across populations, and quote raw precision when sizing a review queue, never the reverse.

### 6.7 The tie-group picture

```text
 score
 0.24 |  *
 0.22 |  *******   <-- 400 rows in band 5, mean 0.22
 0.20 |  ********
 0.18 |  _________  <- threshold candidate A
 0.17 |  .........  <- threshold candidate B is 0.003 away
 0.15 |  *******
 0.13 |  _________
```

If the score were continuous, moving the threshold from A to B would change the flagged count by a handful of rows. With 400 rows sharing a band mean and a discretised score, the same move changes it by hundreds. The picture to hold: the flag count is a step function of the threshold, and the step height is the mass at that score value. Every "the count jumped and nothing changed" report is a step-height question, and the answer is the count at exactly the threshold divided by the total flagged — CI-012.

## 7. Comparison / Selection Table

### 7.1 Routing table: symptom to check

| Observed symptom | Most likely entry | First computation | What a positive result proves |
|---|---|---|---|
| Training AUC 0.95, forward-split 0.66 | CI-001, CI-002 | compare feature max timestamp to `t_d` | a feature exists after the decision |
| Validation AUC falls 0.05 when the seed changes | CI-003, CI-005 | dedupe on the natural key, re-split | dependence between folds |
| AUC drops 0.04 when preprocessing moves inside the fold | CI-004 | refit scaler per fold | transform leakage |
| Recall collapses on the last 60 days | CI-006 | label coverage by day | immature negatives |
| Metric changes the week the definition changed | CI-007 | version the label, re-score both | definition drift |
| Offline 0.71, production 0.58, same population | CI-008 | replay the feature job at `t_d` | train/serve skew |
| Loss `inf` or `nan` in one fold | CI-009, CI-013 | print min/max of the logit | clipping or null propagation |
| Coefficients diverge, no convergence | CI-010 | tabulate the feature against the label | separation |
| Precision 0.22 offline, 0.05 in the queue | CI-011, CI-023 | re-derive the threshold in the deployed space | wrong threshold space |
| Alert count jumps 3x on the same threshold | CI-012 | count score ties at the cut | tie mass |
| Drift dashboard shows `inf` for one feature | CI-014 | print per-bucket masses | empty reference bucket |
| Reported gain 0.03 AUC from a 30-config search | CI-017 | compute `sigma * E[max Z_30]` | selection noise |
| Latency 8 ms to 90 ms after adding a field | CI-019 | count distinct values | cardinality explosion |

The table is a router, not a summary. Each row sends you to one computation whose result changes a decision; if that computation cannot be run because an input is missing, the missing input *is* the defect, and the corresponding entry names it.

### 7.2 Choosing the metric by decision type

| Decision type | Primary metric | Complement | Metric that misleads here | Reason |
|---|---|---|---|---|
| Fixed capacity queue | precision at k | recall at k | accuracy | k, not cost, sets the operating point |
| Cost-weighted approve/reject | expected cost per row | calibration (REL) | AUC | AUC integrates over unused thresholds |
| Ranking for human review | AUC or average precision | lift at 10% | F1 at 0.5 | the human sets the cut |
| Probability used downstream | Brier / log loss | REL, ECE | macro F1 | downstream arithmetic needs calibrated numbers |
| Rare-event screening | recall at fixed PPV | PR-AUC | accuracy, ROC-AUC | ROC looks good at 1% prevalence |
| Multi-class with a dominant class | macro F1 | per-class recall | micro F1 | micro is dominated by the large class |

The last row is worth a number. With three classes and confusion counts (90/10/30), (8/2/2), (2/10/0) as (TP, FP, FN), per-class F1 is 0.8182, 0.8000, 0.2857, so macro F1 is 0.6346. Micro F1 from the pooled counts is `2*100/(2*100 + 22 + 32) = 200/254 = 0.7874`. The 0.153 gap is entirely the third class, and a team that reports 0.787 has hidden the only class that is failing.

### 7.3 Choosing the split by data structure

| Structure of the data | Correct split | What breaks otherwise | Entry |
|---|---|---|---|
| Time-ordered rows, decision at `t` | forward (train `< T`, test `>= T`) | random split leaks the future | CI-001 |
| Multiple rows per customer | grouped by customer | same customer in both sets | CI-005 |
| Duplicated or near-duplicated rows | group by content hash | memorisation counted as skill | CI-003 |
| Small `n`, single population | stratified k-fold with fixed seed | unstable estimate | CI-004 |
| Multiple sites or cohorts | leave-one-site-out | site-specific quirks as signal | CI-005 |
| Very rare label, `n_pos < 50` | repeated stratified k-fold, report mean ± SE | one split is noise | CI-017 |

### 7.4 Choosing what to do about imbalance

| Situation | Resampling | Threshold move | Weighting | Note |
|---|---|---|---|---|
| Need calibrated probabilities | none | no | no | resampling breaks the probability scale |
| Need ranking only | none | not needed | optional | AUC is invariant to prior |
| Cost-sensitive action, calibration not needed | optional | yes | yes | must convert the threshold (4.3.1) |
| Capacity-constrained | none | top-k | no | sort, do not threshold |
| Label noise high | none | no | no | resampling amplifies noisy positives |

The rule that prevents most of this table: resampling is a training-time convenience, never a deployment-time truth. Any probability that leaves a resampled model must be mapped back before it is compared to a cost ratio.

### 7.5 Choosing a diagnostic for a feature that looks too good

| Suspicion | Check | Confirmatory number |
|---|---|---|
| Computed after `t_d` | max(feature timestamp) vs `t_d` | any row with `timestamp > t_d` |
| Encodes the label | IV and per-category rate | IV > 0.3 with a monotone rate |
| Target-encoded with own label | recompute with leave-one-out | encoding shifts by `1/n_cat` |
| Aggregated over a future window | window bounds vs `t_d` | any window end after `t_d` |
| Duplicated across rows | hash of the feature vector | duplicate rate above 1% |
| Cardinality proxy | distinct count | distinct count near `n` |

## 8. Failure Modes & Edge Cases

The catalogue. Each entry names the mechanism, the symptom as it appears in a dashboard or a review, at least two competing causes, the one computation that separates them, the correction, the prevention, and the verification that the correction worked. Identifiers are stable within this file.

### 8.1 Summary table

| ID | Category | Symptom in one line | Severity |
|---|---|---|---|
| CI-001 | VIOLATED_ASSUMPTION | random split on time-ordered rows inflates validation | S1 |
| CI-002 | VIOLATED_ASSUMPTION | a feature is only knowable after the outcome | S1 |
| CI-003 | VIOLATED_ASSUMPTION | duplicated rows straddle the split | S1 |
| CI-004 | VIOLATED_ASSUMPTION | preprocessing fitted before the split | S1 |
| CI-005 | VIOLATED_ASSUMPTION | the same entity appears in train and test | S1 |
| CI-006 | VIOLATED_ASSUMPTION | immature labels enter the evaluation window | S1 |
| CI-007 | VIOLATED_ASSUMPTION | label definition changes inside the window | S2 |
| CI-008 | VIOLATED_ASSUMPTION | training feature is unavailable or different at scoring | S1 |
| CI-009 | NUMERICAL_EDGE | log loss explodes on one confidently wrong row | S2 |
| CI-010 | NUMERICAL_EDGE | perfect separation drives coefficients to infinity | S2 |
| CI-011 | NUMERICAL_EDGE | threshold applied in the weighted score space | S1 |
| CI-012 | NUMERICAL_EDGE | the cut lands inside a large tie group | S2 |
| CI-013 | NUMERICAL_EDGE | nulls propagate to `nan` and silently drop rows | S1 |
| CI-014 | NUMERICAL_EDGE | drift statistic returns infinity | S3 |
| CI-015 | NUMERICAL_EDGE | float32 accumulation loses small gradients | S3 |
| CI-016 | NUMERICAL_EDGE | bin edges recomputed at scoring time | S2 |
| CI-017 | SCALE_FAILURE | a large search reports selection noise as gain | S2 |
| CI-018 | SCALE_FAILURE | a feature backfill rewrites history for every model | S1 |
| CI-019 | SCALE_FAILURE | high-cardinality encoding breaks latency | S2 |
| CI-020 | SCALE_FAILURE | monitoring cost grows with history, not with rate | S3 |
| CI-021 | SCALE_FAILURE | small segments have no usable model | S2 |
| CI-022 | SCALE_FAILURE | retrain cadence is shorter than label latency | S2 |
| CI-023 | CONCEPT_CONFUSION | accuracy quoted at a 1% base rate | S1 |
| CI-024 | CONCEPT_CONFUSION | AUC compared across populations with different priors | S1 |
| CI-025 | CONCEPT_CONFUSION | macro and micro averages conflated | S2 |
| CI-026 | CONCEPT_CONFUSION | impurity importance read as effect size or cause | S2 |
| CI-027 | CONCEPT_CONFUSION | calibration asserted from discrimination | S1 |
| CI-028 | CONCEPT_CONFUSION | offline metric gain equated with business saving | S1 |

### 8.2 VIOLATED_ASSUMPTION

**CI-001 — Random split on time-ordered rows.** `Category:` VIOLATED_ASSUMPTION. `Symptom:` validation AUC 0.79 with a random 80/20 split, 0.66 with a forward split on the same rows, and the production number settles near 0.66. `Competing causes:` (a) genuine temporal drift in the relationship, so the older rows really are a different population; (b) the random split lets a row's near-neighbour from an adjacent day sit in training, so the model sees a slightly perturbed copy of each test row; (c) a monotone trend in a feature (balance, tenure) that the model exploits as a proxy for the period rather than for risk. `Discriminating check:` refit with three splits — random, forward by date, and random-but-within-block (permute whole months) — and compare. If random-within-block recovers most of the 0.79, the gain came from cross-block adjacency and (b) is the cause; if forward and within-block agree at 0.66 while random is higher, (c) is operating. `Correction:` forward split with an embargo equal to the label horizon `H`; report the forward number as the only one that predicts deployment. `Prevention:` enforce the split strategy in the evaluation harness, not in a notebook, so it cannot be forgotten on the next experiment. `Verification:` the random-split number must be *recorded* next to the forward number; a gap above 0.03 is a harness defect, and the review sign-off requires both.

**CI-002 — A feature knowable only after the outcome.** `Category:` VIOLATED_ASSUMPTION. `Symptom:` training AUC 0.97, forward split 0.66, and the top-importance feature is a collections flag or a settlement code. `Competing causes:` (a) a post-outcome field was joined into the feature table (direct label proxy); (b) a field is legitimate but its *value* is updated after the decision, so the snapshot is post-hoc even though the column existed at `t_d`; (c) an aggregate was computed over a window that extends past `t_d`, e.g. a 12-month mean centred on the decision date. `Discriminating check:` for the suspect column, compute `max(event_timestamp)` and `count(event_timestamp > t_d)` over the training table. Any nonzero count proves post-outcome information; then check whether the *value* rather than the existence is post-hoc by comparing the row's value in two snapshots taken at `t_d` and `t_d + 1 day` — a difference proves (b). `Correction:` rebuild the feature as of `t_d` from the event log with a strict `event_timestamp < t_d` filter, and drop the column if the rebuild loses more than 20% of rows. `Prevention:` a feature registry that stores `as_of` for every column and blocks registration of any column whose `as_of` is not derivable from the event log. `Verification:` after the rebuild, training AUC must fall toward the forward number; if it does not, a second leaked column exists and the audit continues with the next-highest importance.

**CI-003 — Duplicated rows straddling the split.** `Category:` VIOLATED_ASSUMPTION. `Symptom:` the validation set contains rows that are identical, or identical up to one field, to training rows; reported AUC is stable but 3 to 8% too high. `Competing causes:` (a) exact duplicates from an upstream join fan-out, where one application produced several rows; (b) near-duplicates, the same applicant re-applying, differing only in a request id and a date; (c) legitimate multiple events per entity, which is a grouping problem rather than a duplication problem (CI-005). `Discriminating check:` hash the feature vector and count rows per hash. On 2,000 rows with a 3% duplication rate there are 60 duplicate rows; a random split places 20% of them in test, so about 12 test rows share a hash with a training row, and the probability that a given duplicate pair straddles the split is `2*0.2*0.8 = 0.32`, giving `60*0.32 = 19.2` contaminated test rows — 4.8% of a 400-row test set. `Correction:` dedupe on the natural business key before splitting, keeping the row nearest `t_d`; if duplicates are legitimate events, group them. `Prevention:` a duplicate-rate assertion in the data contract (`distinct_keys / rows > 0.99`). `Verification:` recompute AUC after deduping; a fall below 0.01 confirms the duplicates were contributing almost nothing, and a fall above 0.03 means the model was largely memorising and the feature set needs review.

**CI-004 — Preprocessing fitted before the split.** `Category:` VIOLATED_ASSUMPTION. `Symptom:` validation AUC drops 0.03 to 0.06 when the pipeline is wrapped correctly, and the drop is larger for the imputer than for the scaler. `Competing causes:` (a) the imputer's fill value (mean or median) was computed on all rows including validation, so the validation rows influenced their own fill; (b) the scaler's mean and standard deviation were computed on all rows, a smaller effect; (c) a target encoder computed on all rows, which is the largest effect because it carries label information. `Discriminating check:` refit each transform separately inside the fold and measure the delta for each. On a scalar feature where the test mean is 40 and the train mean is 30 with `sd = 10`, the all-rows scaler gives the test point `z = (40-30)/10 = 1.0` while a train-only scaler with mean 40 gives `z = 0`; the test row is moved a full standard deviation by the defect. `Correction:` wrap all fitted transforms in a pipeline that receives only the training fold, including imputation, scaling, binning, target encoding and feature selection. `Prevention:` make the pipeline the only entry point — a function that takes raw data and a fold index — so the leaky path is not expressible. `Verification:` the fitted parameters must differ between folds; identical scaler means across folds is proof that the transform saw the whole dataset.

**CI-005 — The same entity on both sides of the split.** `Category:` VIOLATED_ASSUMPTION. `Symptom:` AUC 0.74 on rows, 0.68 when grouped by customer, and the review cannot explain where the 0.06 went because there are no duplicate rows. `Competing causes:` (a) multiple rows per entity and a random row split, so the model learns entity identity; (b) entity-level attributes that are constant within entity (industry, region) act as an identifier when cardinality is high; (c) genuine entity-level signal that grouping removes because a new entity is genuinely harder. `Discriminating check:` compute the number of entities and the rows-per-entity distribution, then compare the grouped and ungrouped splits. If grouped AUC falls by more than twice the standard error, membership is doing work; then fit a model *without* entity-constant columns and repeat — if the gap closes, (b) is the mechanism and those columns were identifiers. `Correction:` `GroupKFold` on the entity key, or a forward split on the entity's first-seen date to mimic deployment. `Prevention:` every table declares its grain; the split function refuses to run without a group key when grain is entity-level. `Verification:` report rows-per-entity and the grouped AUC in the same table; the second number is the one that forecasts.

**CI-006 — Immature labels inside the evaluation window.** `Category:` VIOLATED_ASSUMPTION. `Symptom:` recall measured on the last 60 days is 0.84 and falls to 0.66 when the same week is re-scored a month later; or precision rises every week as adjudication catches up. `Competing causes:` (a) unadjudicated outcomes stored as `0`, so recent positives are counted as negatives; (b) genuine change in the process over the last 60 days; (c) a partial rollout that changed the decision for a subset, so the recent window contains a different policy. `Discriminating check:` compute label coverage per day as (adjudicated outcomes) / (outcomes discoverable by any route, including later adjudication). Coverage below 0.9 for the recent days proves (a), and the recall figure is a lower bound on a partial denominator. Then compare the recent window's `pi` to the mature window's: a materially lower `pi` with unchanged features is the signature of missing positives rather than of a better world. `Correction:` truncate the evaluation at `today - L`; never evaluate inside the maturity window. `Prevention:` store the label with a `known_at` timestamp and filter on it, rather than relying on a refresh schedule. `Verification:` re-run the evaluation after the truncation and check that the reported metric is stable when the same window is re-scored a month later — a drifting number means the filter is still wrong.

**CI-007 — Label definition changes inside the window.** `Category:` VIOLATED_ASSUMPTION. `Symptom:` the event rate steps from 11.2% to 8.9% on a date that coincides with a policy or taxonomy change, and the model's precision appears to improve. `Competing causes:` (a) a definition change — a new disposition code, a longer grace period, a narrower event class; (b) a genuine process improvement; (c) a population change, where a new segment was added and it happens to have a lower rate. `Discriminating check:` restrict to the entities present in both periods and recompute the rate. A step that disappears under the fixed-entity restriction is (c); a step that persists is (a) or (b). Then examine the disposition-code distribution: a code that appears only after the change date, or a code whose share jumps, proves (a). `Correction:` version the label and either rebuild the old period under the new definition (if the raw events exist) or train and evaluate entirely on one side of the change. `Prevention:` the label has a version number in its name, and the training job asserts a single version across the window. `Verification:` the rate curve must be continuous across the boundary after the correction, or the definition still differs.

**CI-008 — Training feature unavailable or different at scoring.** `Category:` VIOLATED_ASSUMPTION. `Symptom:` offline AUC 0.71, production 0.58 on the same population and the same labels, with no code change to the model. `Competing causes:` (a) the feature exists only in the warehouse and the serving path fills it with a default, so the model scores a distribution it never saw; (b) the feature exists in both paths but is computed by different code — one uses a 30-day window, the other a 90-day window; (c) the feature exists but arrives late, so at scoring time the value is stale by up to a day. `Discriminating check:` log the full feature vector at scoring for 1,000 requests, join it back to the warehouse vector for the same `(entity, t_d)`, and compute per-column the fraction of exact matches and the mean absolute difference. A column matching on 0% of rows is (a); a column matching 60% with a systematic difference is (b); a column matching 95% with differences only for rows near a batch boundary is (c). `Correction:` make the serving path read the same materialised table the training path reads, or compute both from one feature definition artifact. `Prevention:` a parity test that runs nightly on a sample and fails the build when any column's match rate falls below 0.99. `Verification:` the parity dashboard must be green for a full window *before* the offline-online gap is re-measured, otherwise the gap measurement is contaminated by the fix itself.

### 8.3 NUMERICAL_EDGE

**CI-009 — Log loss explodes on one confidently wrong row.** `Category:` NUMERICAL_EDGE. `Symptom:` mean log loss jumps from 0.31 to 1.42 in a fold while accuracy is unchanged, and the fold's model is then discarded as unstable. `Competing causes:` (a) an unclipped probability, where a single positive row scored at `1e-15` contributes `-ln(1e-15) = 34.54`, which over 2,000 rows adds 0.0173 to the mean but over 20 rows adds 1.73; (b) a genuine hard case the model gets badly wrong and should learn from; (c) a label error, where the row is a positive recorded as a negative. `Discriminating check:` sort rows by loss and print the top ten with their labels and a few features. If one row contributes more than 20% of the total loss and its probability is at the clip floor, (a); if several rows cluster in an identifiable slice, (b); if the row's feature vector contradicts its label, (c). `Correction:` clip probabilities at `1e-7` and `1 - 1e-7` (contributing at most 16.1 and 0.0000001 respectively), report the clipped-loss value, and quarantine the top-loss rows for label review rather than deleting them. `Prevention:` the loss function asserts a floor before the logarithm; the evaluation reports clipped loss and the count of clipped rows. `Verification:` the fold's loss should be stable across seeds once clipping is in place; an unstable loss with stable accuracy is the signature of a tail row, not of a bad model.

**CI-010 — Perfect separation drives coefficients to infinity.** `Category:` NUMERICAL_EDGE. `Symptom:` the logistic fit does not converge, the coefficient standard error prints as `1e6`, or one coefficient grows with every iteration. `Competing causes:` (a) a feature perfectly aligned with the label on a small subset, typically a rare category where all rows are positive; (b) collinearity so strong that the design is numerically singular, with VIF above 50; (c) the optimiser's tolerance is too tight for the scale of the features. `Discriminating check:` cross-tabulate the suspect feature against the label and find any level with zero variance in the outcome. A level with, say, 14 rows all positive and no negatives proves (a). Then compute VIF: at a pairwise correlation of 0.95, `VIF = 1/(1-0.9025) = 10.26`; at 0.99, `1/(1-0.9801) = 50.25`. A VIF above 10 alongside a finite but huge coefficient proves (b) before (a). `Correction:` for (a), add a small L2 penalty (which bounds the coefficient) or merge the rare level into "other"; for (b), drop the collinear column or use the difference of the pair. `Prevention:` a rare-level guard that merges any category with fewer than 20 rows, and a VIF gate at 10. `Verification:` the fit must converge with a finite gradient norm, and the coefficient's standard error must be smaller than the coefficient itself; a bounded L2 solution that still needs 5,000 iterations is a sign the penalty is too small relative to the feature scale.

**CI-011 — Threshold applied in the weighted score space.** `Category:` NUMERICAL_EDGE. `Symptom:` the review approves 62% of rows while the cost model says it should approve 84%, and nobody changed the threshold. `Competing causes:` (a) positive class weight `w` was used in training and the unweighted threshold was applied to the weighted output; (b) the model was trained on a resampled set with a different positive share and the threshold was not mapped back; (c) the deployed prior differs from the training prior and the threshold was not re-solved. `Discriminating check:` compute the implied unweighted threshold `t = t_w/(w - (w-1)t_w)`. At `w = 9` and `t_w = 0.1667` this gives 0.0217, an eightfold difference in the flag rate; if the observed approval share matches the 0.0217 cut rather than the 0.1667 cut, (a) is proved. For (b), compute the positive share of the training sample and the deployment prior; a factor `c` between them implies the same mapping. For (c), compute `logit(pi_deploy) - logit(pi_train)` and check whether adding it to the logit reproduces the observed rate. `Correction:` either train without weights and calibrate after, or convert the threshold with the formula above and record the conversion in the model card. `Prevention:` the threshold is stored with the space it belongs to (`space: weighted|calibrated`) and the serving code refuses to apply a threshold whose space does not match the score's declared space. `Verification:` the flag rate on a held-out sample must equal the design rate within sampling error; at 2,000 rows and a 0.28 flag rate, the standard error is `sqrt(0.28*0.72/2000) = 0.0100`, so a rate outside 0.26 to 0.30 indicates the conversion is still wrong.

**CI-012 — The cut lands inside a large tie group.** `Category:` NUMERICAL_EDGE. `Symptom:` the flagged count is 512 one day and 1,340 the next on an unchanged threshold and a nearly unchanged score distribution. `Competing causes:` (a) the score is discretised (integer points, integer tree outputs) and a large mass sits exactly at the threshold, so `>=` versus `>` decides thousands of rows; (b) the flag count is capped downstream and the cap interacts with tie ordering; (c) the score is a percentile within a batch, so its value depends on the batch's composition. `Discriminating check:` count rows with score exactly equal to `t` and divide by the flagged count. Above 20% proves (a), and the count becomes a step function of `t`. Then check whether the ordering within the tie group is deterministic: hash two runs and compare the flagged sets; a difference proves (b) or (c). `Correction:` add a deterministic secondary sort (a stable hash of the row key) so the count at a tie is reproducible, and publish the count as a range `[count(>t), count(>=t)]`. `Prevention:` thresholds are chosen at gaps in the score distribution, or the model is scored on a continuous scale (logit) rather than a discretised one. `Verification:` two identical runs must produce identical flagged sets; the tie-share metric must be printed on every threshold change.

**CI-013 — Nulls propagate and silently drop rows.** `Category:` NUMERICAL_EDGE. `Symptom:` the training job reports 2,000 rows in and 1,940 rows fitted, with no warning; the 60 dropped rows have a 25% default rate against 10% for the rest. `Competing causes:` (a) a division by a null denominator yields `nan`, and the estimator's default behaviour is to drop the row; (b) a `where` clause implemented as an inner join to a lookup table that lacks 3% of keys; (c) an explicit filter intended for a different purpose, left in from an earlier experiment. `Discriminating check:` count rows at each stage of the pipeline. A drop between the join and the fit proves (b) if the join is the stage; a drop after the arithmetic stage proves (a). Then quantify the consequence: with 160 rows (8%) having a 25% rate and 1,840 having 10%, the overall rate is `(40 + 184)/2000 = 0.112`, but the *fitted* rate is `184/1840 = 0.100`, a 10.7% relative understatement of prevalence; a chi-square on the 2x2 table (40/120 versus 184/1656) gives 33.3 on one degree of freedom, so the dropped rows are not missing at random. `Correction:` replace the drop with an explicit imputation plus a missingness indicator, and keep the row. `Prevention:` a row-count assertion at every stage with the expected drop budget stated as zero, plus a missingness report attached to the model artefact. `Verification:` the fitted `pi` must match the population `pi` within sampling error; a fitted rate of 0.100 against a population of 0.112 is the residual signature of the drop.

**CI-014 — Drift statistic returns infinity.** `Category:` NUMERICAL_EDGE. `Symptom:` the monitoring report shows `inf` for one feature and the whole report is suppressed by the dashboard's error handler, so a real drift in another feature goes unnoticed for a week. `Competing causes:` (a) a bucket that has mass in the reference and zero in the current window, so `ln(a/e)` diverges; (b) a bucket with an empty reference created by fixed bin edges beyond the data range, so both sides are zero; (c) the feature went constant in the current window, putting 100% of mass in one bucket. `Discriminating check:` print per-bucket reference and current masses. If a current mass is zero with a reference mass of 0.004, (a) holds and the drift is real but extreme; if both are zero, (b) is an artefact of the edge rule; if one bucket holds 1.0, (c) is a data defect, not drift. `Correction:` add a floor of `1e-4` to both masses (documented), drop zero-mass-on-both-sides buckets from the sum, and report the unfloored contribution alongside. `Prevention:` quantile-based bins fitted on the reference with guards for constant features, and a `nan`/`inf` count in the report rather than an exception. `Verification:` the corrected PSI for the reference against itself must be exactly 0 and finite; on the moderate-shift example (shares 0.10/0.15/0.20/0.25/0.30 against equal reference) the value is 0.1354, and on the sharp example (0.05/0.10/0.20/0.30/0.35) it is 0.4017 — both finite and consistent with the 0.10 and 0.25 conventions.

**CI-015 — Float accumulation loses small gradients.** `Category:` NUMERICAL_EDGE. `Symptom:` the loss curve is flat for the first 200 iterations and then falls, or the model trained in one framework does not match the same model trained in another despite identical hyperparameters. `Competing causes:` (a) float32 accumulation of a sum over millions of rows, where the running total is large relative to each increment; (b) a learning rate too small relative to the gradient scale; (c) different default regularisation between frameworks. `Discriminating check:` repeat the training with float64 accumulation (or Kahan summation) and compare the loss after 50 iterations. A material difference proves (a). Then compare the loss at iteration 1 on identical data — a difference at iteration 1 is (c), because no update has occurred yet. `Correction:` accumulate in float64, or centre the target, or use a loss that is scale-invariant. `Prevention:` a two-framework parity test on a 5,000-row sample at fixed seed, with a tolerance stated in the test. `Verification:` the parity test must agree to within the stated tolerance for 20 iterations; agreement at iteration 1 but divergence at 50 is an accumulation problem, not a specification problem.

**CI-016 — Bin edges recomputed at scoring time.** `Category:` NUMERICAL_EDGE. `Symptom:` a feature's contribution changes for a row whose raw value did not change; the same applicant scores 0.14 in the replay and 0.18 in production. `Competing causes:` (a) quantile edges are recomputed on the current batch, so a row's bin depends on who else is in the batch; (b) the binning is on a different scale (raw versus log) in the two paths; (c) the batch is smaller than the number of quantiles, so several edges are equal and the bin assignment is arbitrary. `Discriminating check:` score a fixed 100-row fixture in batches of 100, 500 and 5,000 and compare scores. A change proves (a). Then print the edge arrays from both paths and compare element-wise; unequal edges at the same index prove (b) or (a) depending on whether they are constants. `Correction:` freeze edges at fit time, store them in the artefact, and use them verbatim at scoring; fall back to an open-ended tail bin rather than recomputing. `Prevention:` the binning transform is part of the serialised pipeline, never a parameter recomputed at serve time. `Verification:` batch-size invariance — the fixture's scores must be identical across batch sizes to the last printed digit.

### 8.4 SCALE_FAILURE

**CI-017 — A large search reports selection noise as a gain.** `Category:` SCALE_FAILURE. `Symptom:` a 30-configuration search improves validation AUC by 0.031 over the baseline, the winning configuration is not reproducible on a second seed, and a re-run picks a different winner. `Competing causes:` (a) selection noise, where the maximum over 30 candidates on a noisy metric is biased upward even when every candidate is equal; (b) genuinely different configurations with different variance, so the winner is the high-variance one; (c) the validation set was used both to select and to report. `Discriminating check:` compute the expected inflation `sigma * E[max Z_k]`. At `sigma = 0.008` (a fold-level AUC standard error) and `k = 30`, `E[max Z_30] = 3.96`, so the expected inflation is 0.0317 — which is the *entire* observed gain. Then re-run the search with a different seed and count how often the winner changes; above half is proof of (a). `Correction:` nest the search inside an inner cross-validation and report on an outer holdout never touched by selection; state the search size and the inflation estimate next to the result. `Prevention:` a fixed protocol: one holdout, sealed; all selection on the inner loop; the search size recorded in the model card. `Verification:` the outer-holdout gain must be smaller than the inner gain by roughly the inflation; a gain that *grows* on the holdout is evidence the holdout leaked.

**CI-018 — A feature backfill rewrites history for every model.** `Category:` SCALE_FAILURE. `Symptom:` after a backfill, every model's offline metrics improve by 2 to 4 points on the same period, and the improvement appears for models that were not retrained. `Competing causes:` (a) the backfill corrected a genuine data defect, so the improvement is real but the historical metrics were computed on a different table than the current one; (b) the backfill introduced post-hoc information (a late-arriving event now visible in the historical window); (c) the models are re-scored on new snapshots, so the numbers were never comparable. `Discriminating check:` for one feature, compare the value at `t_d` in the pre-backfill snapshot to the value in the post-backfill snapshot, restricted to rows whose source events arrived before `t_d`. Differences on that restriction prove (a) or (b); then restrict to events arriving after `t_d` — a difference there proves (b) and the feature is now leaked. `Correction:` version the feature table, score all models against the version they were trained on, and rebuild the leaked feature from the as-of snapshot. `Prevention:` immutability of historical snapshots; corrections are published as a new version, never as an in-place update. `Verification:` a reconciliation report showing row counts and value distributions for old and new versions, signed off before any model is retrained.

**CI-019 — High-cardinality encoding breaks latency.** `Category:` SCALE_FAILURE. `Symptom:` median scoring latency rises from 8 ms to 90 ms after adding a merchant-identifier field, with no change in model complexity. `Competing causes:` (a) one-hot expansion of a 50,000-level field, so the vector grows from 22 to 50,022 columns and the model's inner product becomes memory-bound; (b) target encoding with a lookup that misses on new levels, causing a fallback path that is slow; (c) the feature store call is now on the critical path and dominates the latency. `Discriminating check:` print the feature vector width and the model's parameter count before and after. A width above 1,000 with a sparse model proves (a); a nonzero miss rate on the lookup with latency concentrated in the encoding step proves (b); a latency profile showing the model at 5 ms and the fetch at 80 ms proves (c). `Correction:` for (a), hash the field to 256 buckets or use a target-encoded scalar; for (b), precompute a default for unseen levels at fit time; for (c), move the fetch off the critical path or cache it. `Prevention:` a cardinality budget in the feature contract (any field above 1,000 levels requires an encoding decision recorded in the model card) and a latency budget asserted in CI. `Verification:` p99 latency at the design throughput must return within 10% of the pre-change baseline, and the score must be unchanged to within `1e-6` on a fixture of rows that do not contain the new field.

**CI-020 — Monitoring cost grows with history, not with rate.** `Category:` SCALE_FAILURE. `Symptom:` the weekly performance report takes 40 minutes and grows by 3 minutes a week, while the weekly scored volume is constant. `Competing causes:` (a) the report recomputes the metric over all history each week rather than over the window; (b) the report joins the label table without an index, so the join is a full scan that grows with the label table; (c) the metric is computed per entity and the entity count grows even though the row count does not. `Discriminating check:` compute the number of rows scanned per report; growth proportional to the total history proves (a), growth proportional to the label table proves (b), and growth proportional to distinct entities with a constant row count proves (c). `Correction:` materialise the metric per period once and aggregate over periods; the corrected report should be `O(window)` not `O(history)`. `Prevention:` the report takes a window parameter and refuses a window larger than 12 periods without an explicit override. `Verification:` the report's runtime must be independent of how far back the table goes; a runtime that still grows with the table size means the query is not truly windowed.

**CI-021 — Small segments have no usable model.** `Category:` SCALE_FAILURE. `Symptom:` the model is fitted per segment; 9 of 40 segments hold 60% of the volume and the remaining 31 hold 4% each, and their per-segment AUC ranges from 0.51 to 0.88 with overlapping intervals. `Competing causes:` (a) each small segment has too few positives for the parameters fitted, so its estimate is noise; (b) the segments genuinely differ and pooling loses real signal; (c) the segment definition changed recently, so history per segment is short. `Discriminating check:` compute events per variable per segment. At 22 parameters and 5% of 224 events, a small segment has 11 events, an EPV of 0.5 — two orders below the floor of 10. Then fit a pooled model with a segment indicator and compare: if the pooled model matches the large segment's AUC and beats the small segments' average, (a) holds. `Correction:` pool with segment-specific intercepts (a partially pooled model), or fall back to the pooled model for segments below the EPV floor. `Prevention:` a minimum-volume gate in the model registry, with the fallback declared in the model card so the consumer knows which model scored a given row. `Verification:` the coverage report must state, per segment, which model scored it; a segment scored by a fallback whose false-positive rate is unmeasured is a coverage gap, not a result.

**CI-022 — Retrain cadence shorter than label latency.** `Category:` SCALE_FAILURE. `Symptom:` the model is retrained monthly; each retrain uses the most recent 12 months, but the last 3 months of labels are incomplete, so every retrain trains on a window whose most recent quarter is 40% negatives-by-default. `Competing causes:` (a) the maturity lag `L` exceeds the retrain interval, so the newest data is unusable but included; (b) the retrain window is too short and the model overfits a seasonal period; (c) the retrain is triggered by a calendar rather than by a measured need. `Discriminating check:` compute the positive rate by month of decision for the last 12 months. A monotone decline in the most recent months that reverses when the same months are re-read a quarter later proves (a). Then compare the model's thresholded flag rate on the mature window with its rate on the immature window; a systematic difference is (a) again, not (b). `Correction:` truncate the training window at `today - L` and state the effective training end date in the model card; if that leaves too little data, lengthen the window rather than shortening the lag. `Prevention:` the training job computes `L` from the label table and refuses to include rows newer than `max(t_d) - L`. `Verification:` the positive rate in the final month of the training window must be within sampling error of the preceding months; a dip is the signature of truncation failure.

### 8.5 CONCEPT_CONFUSION

**CI-023 — Accuracy quoted at a low base rate.** `Category:` CONCEPT_CONFUSION. `Symptom:` the model card says "accuracy 0.94" on a problem with a 1.12% positive rate, and the reviewer approves it; a constant-negative model would score 0.9888 on the same data. `Competing causes:` (a) accuracy was chosen because it is familiar, not because it matches the decision; (b) the evaluation set was resampled to balance classes, so accuracy is computed on a population that does not exist; (c) the reported number came from a different window with a different prevalence. `Discriminating check:` compute the majority-class baseline on the same rows. If the model's accuracy is below or within 0.01 of that baseline, the metric carries no information. Then check the positive share of the evaluation set: a share near 0.5 means (b). `Correction:` replace accuracy with recall at a fixed precision, or the expected cost per row, and state the base rate next to every rate. `Prevention:` the metric registry pairs every rate with its base rate and refuses to register an unpaired one. `Verification:` the reported metric must separate the model from the baseline by more than the metric's standard error; at 2,000 rows and accuracy 0.94, the standard error is `sqrt(0.94*0.06/2000) = 0.0053`, so a 0.01 margin is under two standard errors and is not evidence.

**CI-024 — AUC compared across populations with different priors.** `Category:` CONCEPT_CONFUSION. `Symptom:` the model's AUC is 0.74 on the pilot population and 0.66 on the full population, and the review concludes the model degraded; the pilot population had a 34% positive rate against 11.2% overall. `Competing causes:` (a) the populations differ in case mix, so the conditional distributions differ and AUC is legitimately different; (b) the pilot was a selected sample and the selection was on a variable correlated with the label; (c) the model changed between the two measurements. `Discriminating check:` recompute AUC on the pilot rows *using the full-population model* and on a random sample of the full population matched to the pilot's case mix. If the first reproduces 0.74, the model is unchanged and (a) or (b) explains the gap; if it reproduces 0.66, the pilot's number came from a different model, i.e. (c). `Correction:` report AUC per population with the population defined, and never as a single global figure; where the decision is per population, evaluate per population. `Prevention:` every reported metric carries a population identifier and a window, in the same way a rate carries a denominator. `Verification:` the two AUCs must be recomputable from the stored scores and population definitions by a third party; a metric that cannot be reproduced from stored artefacts is not a measurement.

**CI-025 — Macro and micro averages conflated.** `Category:` CONCEPT_CONFUSION. `Symptom:` the model card reports F1 0.787; the class that matters has F1 0.286 and the report does not show it. `Competing causes:` (a) micro averaging was used on a multi-class problem, so the large class dominates; (b) the per-class threshold was tuned on the training set only; (c) the class ordering was permuted between the label encoder and the report, so the reported per-class number belongs to a different class. `Discriminating check:` recompute both. With counts (90/10/30), (8/2/2), (2/10/0) as (TP, FP, FN), per-class F1 is 0.8182, 0.8000, 0.2857 and macro F1 is 0.6346, while the pooled counts give micro F1 = 200/254 = 0.7874. The 0.153 gap localises the problem to the third class; then print the label encoder's class order to test (c). `Correction:` report per-class precision and recall, plus one aggregate whose definition is stated; for a decision that concerns one class, report that class alone. `Prevention:` the metric function returns a per-class table and a named aggregate, never an unnamed scalar. `Verification:` the per-class counts must sum to the pooled counts; if they do not, two different subsets were evaluated.

**CI-026 — Impurity importance read as effect size or cause.** `Category:` CONCEPT_CONFUSION. `Symptom:` a feature with 12 levels and no relationship to the label ranks fourth in gain importance, and the business is told to collect more of it. `Competing causes:` (a) impurity gain rewards cardinality, because a 12-level split has 11 candidate cut points and will find a spurious one; (b) the feature is a proxy for a genuinely predictive one; (c) the importance was computed on training data where the feature memorised rows. `Discriminating check:` compute permutation importance on held-out data and compare rankings. A feature that ranks high on gain and near zero on permutation importance is (a) or (c). Quantify the expectation: with 12 levels and 400 rows there are 33.3 rows per level, and a random feature's expected share of the total gain is about `1/12 = 0.083`, so a reported 8% share is exactly what noise produces. `Correction:` report permutation importance with its standard error, computed on held-out data, and drop gain-based rankings from any decision-facing artefact. `Prevention:` importance is computed by one function that returns held-out permutation values with intervals; the impurity version is not exposed. `Verification:` the top features' permutation importance must exceed the noise band (`1/n_features` of total, roughly 4.5% at 22 features) by a margin; a feature inside the band is not evidence of anything.

**CI-027 — Calibration asserted from discrimination.** `Category:` CONCEPT_CONFUSION. `Symptom:` the review states "the model is well calibrated, AUC is 0.71", and the deployed threshold then flags 62% of rows where the cost model intended 36%. `Competing causes:` (a) the score is a monotone transform of a probability (rank-based, or a margin), so the ordering is right and the heights are arbitrary; (b) the model is calibrated on average but not within bands, so the mean matches and the tails do not; (c) the score is calibrated on the training prior and the deployment prior differs. `Discriminating check:` build a 5-band calibration table with predicted and observed rates. On the reference table the deviations are all zero (REL = 0) and Brier is 0.09492; on a compressed version of the same ranking, REL rises to 0.00076, ECE to 0.024, Brier to 0.09568 with AUC unchanged at 0.6890. That pair of numbers is the proof that discrimination and calibration are independent. For (c), check the mean predicted score against the observed base rate: a mean of 0.30 against a rate of 0.112 is a prior offset of `logit(0.30) - logit(0.112) = 1.223`, exactly the shift in 4.3.2. `Correction:` fit a calibration map (Platt scaling or isotonic on a held-out set), report REL, ECE and the band table, and re-solve the threshold after calibration. On the compressed example, a two-point Platt map has slope 1.7896 and intercept 1.4479 and recovers 0.03/0.0634/0.1074/0.1603/0.22 against true 0.03/0.06/0.10/0.15/0.22. `Prevention:` no threshold may be applied to a score without a stored calibration artefact and a band table generated from held-out data. `Verification:` the calibrated score's mean must equal the observed base rate within sampling error, and the band deviations must be within two binomial standard errors per band.

**CI-028 — Offline metric gain equated with business saving.** `Category:` CONCEPT_CONFUSION. `Symptom:` a 0.02 AUC improvement is presented as worth 4 million in avoided losses, and the derivation is a single multiplication nobody can reproduce. `Competing causes:` (a) the AUC-to-cost translation ignored the operating point, where the actual saving is `O(100k)`; (b) the assumed intervention effect is 100%, so every correctly ranked default is assumed prevented; (c) the baseline cost was computed on a different population than the improvement. `Discriminating check:` rebuild the number from the band table at the deployed threshold. The k=1 policy costs 198,400 against 224,000 for approving everyone, a saving of 25,600 over 2,000 applicants = 12.80 per applicant; a 0.02 AUC gain that moves one band boundary might change that by one band's worth of cost, which is `8,000` on the same population — three orders below the claim. Then multiply by the intervention's measured effectiveness; at 30% the saving falls to 2,400. `Correction:` express the benefit as `(cost_baseline - cost_new) * effectiveness` on a stated population and window, with each factor sourced. `Prevention:` a benefit template with four required inputs (population, window, cost ratio, intervention effectiveness) and a reviewer who recomputes one line by hand. `Verification:` the recomputed saving must match the claimed saving within 10%; the reference computation is `(224,000 - 198,400) = 25,600` for the population, or 12.80 per applicant.

### 8.6 Masking map: which entry imitates which

Entries are not independent, and the dependence has a direction. An upstream defect produces the symptom of a downstream entry, so the downstream check passes while the real cause survives. Read the arrow as "produces the symptoms of".

```text
CI-002  post-outcome feature        ->  CI-026  "this feature is highly important"
CI-001  random split on time rows   ->  CI-017  "search found a better configuration"
CI-004  preprocessing before split  ->  CI-017  "regularisation improved the model"
CI-003  duplicates straddling       ->  CI-024  "pilot population scores higher"
CI-006  immature labels             ->  CI-007  "the definition changed"
CI-008  train/serve skew            ->  CI-024  "the model degraded in production"
CI-011  wrong threshold space       ->  CI-028  "the business case did not materialise"
CI-012  tie mass at the cut         ->  CI-023  "accuracy is unstable week to week"
CI-013  null rows dropped           ->  CI-027  "the score is miscalibrated"
CI-018  backfill rewrote history    ->  CI-002  "the new feature is suspiciously strong"
```

Two mechanisms make the map concrete. First: a post-outcome feature (`CI-002`) produces the largest gain importance in the model, so the importance review (`CI-026`) flags it as a cardinality artefact, the team replaces the raw field with a hashed version of the same field, and the leak survives with a new name. The checks disagreed about the mechanism and agreed about nothing else; the timestamp audit is the only computation that can settle it.

Second, arithmetic rather than logical: dropped null rows (`CI-013`) lower the fitted prevalence from 0.112 to 0.100, so the fitted intercept sits at `logit(0.100) = -2.197` instead of `logit(0.112) = -2.070`. The score is then systematically low by 0.127 in logit space, the calibration band table shows every band slightly under-predicting, and the review files it as a calibration problem (`CI-027`), fits a Platt map, and ships a model whose intercept is now correct for a population that excludes the 8% of rows with the highest default rate. Both fixes were locally correct; the composition is wrong, and only a row-count assertion finds it.

A third case where the masking runs upward. A large search (`CI-017`) selects a configuration that happens to include an aggressive imputation, which happens to reduce the effect of the dropped nulls, so the search "discovers" a fix for a defect nobody diagnosed. The pipeline is now better and the defect remains, ready to reappear when the data's missingness pattern changes. Recording both entries is the honest resolution, which is why the map is a graph rather than a priority order.

## 9. Complexity & Cost Analysis

### 9.1 Fit cost

```text
FORMULA   cost_fit  =  n_rows * n_features * n_trees * c_split
WHERE     n_rows = 2,000; n_features = 22; n_trees = 300; c_split = histogram build per node,
          approximately O(n_features) per node visit with a small constant.
CONSTRAINT gradient boosting evaluates each feature once per node over the rows in the node;
          total work is O(n_rows * n_features * n_trees * depth) in the worst case.
NUMERICAL EXAMPLE
          Feature evaluations per model = 2,000 * 22 * 300 = 13,200,000
          Node visits per model          = 2,000 * 300 * (depth 6 levels) = 3,600,000
          At 1e7 feature evaluations per second on one core: 1.32 s per model.
INTERPRETATION
          One model on 2,000 rows costs about a second; the model is never the cost centre at
          this size. The cost centre is the search around it (9.2) and the human time spent
          auditing the result (9.4).
```

The memory figure is equally modest and worth stating because it is often misremembered: `2,000 * 22 * 8 = 352,000` bytes = 344 KiB for a dense float64 matrix. A team that reports "out of memory" at this scale is reporting a bug in the pipeline (a cartesian join, a one-hot expansion, or an accidental copy per fold), not a capacity limit. At 5 million rows the same matrix is 880 MB, which is where the sparse representation starts to matter and where CI-019 becomes a real constraint.

### 9.2 Search cost, and why the search is the expensive object

```text
FORMULA   cost_search  =  k_configs * k_folds * cost_fit
WHERE     k_configs = 30, k_folds = 5, cost_fit = 1.32 s
CONSTRAINT the search is embarrassingly parallel, so wall-clock is cost_search / workers;
           the statistical cost does not parallelise.
NUMERICAL EXAMPLE
          Total fits = 30 * 5 * 150 (300 trees, counted at 0.5 s avg) = 45,000 model-fits
          CPU time   = 30 * 5 * 1.32 s = 198 s single-core
          Feature evaluations = 45,000 * 13,200,000 / 300 * ... = 1.98e9
INTERPRETATION
          Three minutes of CPU buys a 0.0317 AUC inflation (Section 4.3.4) that will not
          replicate. The marginal value of the 30th configuration is negative: it costs a
          seed of selection noise and returns nothing, which is why the correction in CI-017
          is procedural rather than computational. A 6-configuration search with a sealed
          holdout costs 40 s and yields a number that survives review.
```

### 9.3 Evaluation and monitoring cost

```text
FORMULA   cost_eval  =  n_eval * (1 + k_metrics) * c_metric + cost_join
WHERE     n_eval = rows in the evaluation window; cost_join dominates when the label table
          is large and unindexed.
CONSTRAINT recomputing over all history makes cost_eval grow with the table, not the window
          (CI-020); the corrected form is windowed.
NUMERICAL EXAMPLE
          Weekly report at n_eval = 2,000 rows and 12 metrics: 24,000 metric evaluations,
          negligible. With a full-scan join over 40 million label rows: 40e6 * 40 bytes =
          1.6 GB read per report, which at 100 MB/s is 16 s — and it is 16 s every week,
          forever, because the scan grows with history while the window does not.
INTERPRETATION
          The monitoring cost is not proportional to the modelling work; it is proportional
          to the join. Materialising the metric per period converts an O(history) job into an
          O(window) job and removes the most common reason a monitoring suite is quietly
          abandoned.
```

### 9.4 Human review cost, the term that dominates

```text
FORMULA   cost_review  =  n_flagged * minutes_per_review / 60 * hourly_rate
WHERE     n_flagged = 400 (one band), minutes_per_review = 6, rate = 40 per hour
CONSTRAINT reviewers are a fixed capacity; the queue does not shorten by adding model quality
NUMERICAL EXAMPLE
          400 * 6/60 = 40 reviewer-hours per decision round
          at 40 per hour = 1,600 per round
          model saving per round = 25,600 (Section 5.2) -- of which the review already
          consumes 1,600, so the incremental value is 24,000 only if the review is *new*
          work rather than work that already happened.
INTERPRETATION
          The 12.80 per applicant saving from Section 5.2 must be compared against the
          marginal review cost per applicant, not against zero. If the existing process
          already reviews every application, the model's contribution is the reduction in
          that review, and the honest comparison is 12.80 against the per-application review
          cost of 40 * 6/60 = 4.00, a ratio of 3.2. Stating the ratio this way is what makes
          the case auditable; stating an absolute saving invites the CI-028 defect.
```

### 9.5 The cost of a false alarm in the evaluation itself

An evaluation harness that is wrong costs more than any model it evaluates, because every downstream decision inherits the error. A leakage defect that inflates AUC by 0.10 will cause a model to be shipped that should not have been, and the cost of the resulting bad decisions is unbounded by the modelling budget. This is why the harness belongs in version control with tests (batch-size invariance, fold-parameter divergence, timestamp monotonicity, duplicate rate, label coverage), and why those tests are cheap: each is a few seconds on a fixture, against a search that costs minutes.

### 9.6 Where the budget should go

| Activity | Cost | Expected effect on decisions | Verdict |
|---|---|---|---|
| Dedupe and group split | minutes | removes 0.03 to 0.08 of inflation | first |
| Timestamp audit of features | hours | removes 0.10+ of inflation when present | first |
| Label maturity filter | minutes | corrects every rate | first |
| Forward split with embargo | hours | makes the reported number predictive | first |
| Threshold derivation from costs | hours | changes the action | second |
| Calibration map | hours | makes the probability usable | second |
| 30-config search | minutes of CPU, days of review | negative at this `n` | last |
| Extra trees, extra depth | minutes | 0.005 or less | last |

## 10. Key Theorems / Results Reference

**10.1 Bayes threshold (Section 4.1).** For a calibrated score and constant per-row costs, the cost-minimising action threshold is `p* = C_FP/(C_FP + C_FN)`. This is a decision-theoretic result, not a statistical one; it holds for any model, and its only requirement is calibration. It is the origin of the 0.1667 used throughout.

**10.2 Bayes' identity for precision (Section 4.2).** `PPV = tpr*pi/(tpr*pi + fpr(1-pi))`. The consequence that matters operationally is that a tenfold fall in `pi` cuts PPV by roughly a factor of nine when `fpr` dominates the denominator. Every "the model degraded in production" claim must be checked against this identity before any retraining is authorised.

**10.3 Hanley–McNeil standard error for AUC.** For two classes with `n_1` positives and `n_0` negatives,

```text
        Q1 = A/(2-A)          Q2 = 2A^2/(1+A)
        SE(A) = sqrt( ( A(1-A) + (n1-1)(Q1 - A^2) + (n0-1)(Q2 - A^2) ) / (n1*n0) )
```

At `A = 0.6890`, `n_1 = 224`, `n_0 = 1,776`: `Q1 = 0.6890/1.3110 = 0.5256`, `Q2 = 2*0.4747/1.6890 = 0.5622`, giving `SE = 0.0205` and a 95% interval of 0.649 to 0.729. The practical reading is that any AUC difference below about 0.04 on this sample is not distinguishable from noise.

**10.4 Mann–Whitney equivalence.** `AUC = (concordant pairs + 0.5*ties) / (n_1 * n_0)`. With the band table, concordant pairs number 236,144 and ties 75,936, giving `(236,144 + 37,968)/397,824 = 0.6890`, matching the trapezoidal area exactly. The equivalence matters because it gives a second, independent computation to check a library's AUC against — and because it explains why ties halve the credit: a discretised score with large tie groups cannot exceed the area its resolution permits.

**10.5 Brier decomposition (Murphy).** `BS = REL - RES + UNC`, with the three terms defined in 4.3.3 and verified to five decimals in 5.4. The result that matters is that REL and RES can move in opposite directions: a sharper model with worse calibration can have a worse Brier score than a flat calibrated one.

**10.6 Proper scoring rules.** Log loss and Brier are strictly proper: their expectation is minimised by the true probability, so they cannot be gamed by reporting a distorted probability. Accuracy, F1, and AUC are not proper; each can be improved by a transformation that changes nothing about the underlying belief. This is the formal reason a threshold-tuned model may not be used where a probability is needed.

**10.7 Events-per-variable heuristic.** The conventional floor is 10 events per estimated parameter. With `d = 22` and 224 events the ratio is 10.2, so the reference model is at the boundary; adding one interaction term moves it below. The heuristic is a rule of thumb rather than a theorem, and its empirical basis is in the logistic-regression simulation literature; treat 10 as a warning line, not a guarantee.

**10.8 Prior-shift intercept correction (Saerens et al., logistic case).** Under a prior change with unchanged class-conditional covariate distributions, the log-odds shift by a constant `logit(pi_1) - logit(pi_0)`, so only the intercept changes. At 0.112 to 0.30 the shift is +1.2232. The theorem's precondition — unchanged conditionals — is exactly what CI-024 tests, which is why the correction is not automatic.

**10.9 Selection-maximum inflation.** For `k` independent candidates with metric noise `sigma`, `E[max] ≈ sigma * E[max Z_k]`, with `E[max Z_30] = 3.96`. This is a result about order statistics, and it is why a search size belongs in a model card: the reported best is not an unbiased estimate of that configuration's performance.

**10.10 Log-loss floor from clipping.** Clipping at `1e-15` bounds a single row's contribution at 34.54 nats; at `1e-7` the bound is 16.12 nats. With 2,000 rows, one clipped row at the tighter bound moves the mean by 0.0081, which is the scale at which clipping decisions become visible in a reported number.

## 11. Interview / Exam Questions

**Q1. A model's training AUC is 0.96 and its forward-split AUC is 0.66. You have one hour. What do you compute first?**
Answer: the maximum event timestamp of every feature against the decision timestamp, restricted to the training window. Reasoning: the 0.30 gap is too large for drift and too large for regularisation to explain; the two mechanisms that produce it are a post-outcome feature and a random split on time-ordered rows, and both are settled by timestamps in minutes. Follow-up: if every timestamp is before `t_d`, what is the next check? Red flag: "add regularisation and see if the gap closes" — that treats a provenance defect as a capacity defect and will shrink the leaked coefficient into a plausible size. Scoring: full credit for naming the timestamp audit and the two candidate mechanisms.

**Q2. With `C_FN = 1000` and `C_FP = 200`, a row scores 0.18. Approve or reject, and by how much?**
Answer: reject; expected cost of approving is `0.18*1000 = 180` against `0.82*200 = 164` for rejecting, so rejecting is cheaper by 16 per row. Reasoning: the break-even is `200/1200 = 0.1667` and 0.18 exceeds it; the comparison is linear in `p` because both actions have one cost term. Follow-up: at what `C_FP` does the decision flip? Red flag: comparing 0.18 to 0.5, which is the unweighted default and has no cost meaning. Scoring: full credit for 164 versus 180 and the 0.1667 break-even.

**Q3. A pilot reports precision 0.34 and production reports 0.05 on the same model. Is the model broken?**
Answer: not necessarily. Recompute `PPV = tpr*pi/(tpr*pi + fpr(1-pi))` for both populations. If the pilot's `pi` was 0.34 and production's is 0.05, the identity alone produces the difference with no model change. Reasoning: precision is prior-sensitive by construction, and the identity separates a prior effect from a model effect. Follow-up: what would prove the model *did* change? Red flag: "retrain on production data" without checking `pi` first, which destroys the evidence. Scoring: full credit for the identity and the two-population comparison.

**Q4. You trained with positive class weight 9 and applied a threshold of 0.1667. What flag rate do you expect relative to the intended one?**
Answer: roughly eight times too high. The weighted threshold maps to `t = 0.1667/(9 - 8*0.1667) = 0.0217` in unweighted probability, so the cut is at 0.0217 rather than 0.1667. Reasoning: weighting multiplies the positive odds by `w`, so the same decision boundary corresponds to a much smaller unweighted probability. Follow-up: how would you detect this from a flag-rate dashboard alone? Red flag: "class weights do not change the threshold", which is true only if the threshold is also converted. Scoring: full credit for the conversion formula and the factor of about 8.

**Q5. Validation AUC falls from 0.74 to 0.68 when you switch to `GroupKFold`. Which number do you ship?**
Answer: the grouped number, and you investigate the 0.06 before shipping either. Reasoning: the grouped split estimates performance on unseen entities, which is the deployment condition when new customers arrive; the row split lets the model learn entity identity. Follow-up: what if the business genuinely scores existing customers only? Red flag: "0.74, because it is higher" — that is selection on the metric. Scoring: full credit for the grouped number plus the conditional on the deployment population.

**Q6. Compute macro and micro F1 for classes with (TP, FP, FN) of (90,10,30), (8,2,2), (2,10,0) and say which to report.**
Answer: per-class F1 is 0.8182, 0.8000, 0.2857; macro F1 is 0.6346; micro F1 is `200/254 = 0.7874`. Report per-class values plus one named aggregate. Reasoning: the 0.153 gap is entirely the third class, whose recall is 1.0 on two positives and whose precision is 0.167; a single micro number hides a class that is failing. Follow-up: what does recall 1.0 on `n = 2` tell you? Red flag: quoting 0.787 without the per-class table. Scoring: full credit for both aggregates and the localisation.

**Q7. A drift report shows `inf` for one feature. What are the three possible causes and which do you check first?**
Answer: a zero-mass bucket in the current window with nonzero reference mass; a bucket with zero mass on both sides from fixed edges; or a constant feature putting all mass in one bucket. Check the per-bucket masses first. Reasoning: the three produce the same displayed symptom and different corrections, and printing the masses distinguishes them in one command. Follow-up: what floor do you apply and what do you report alongside it? Red flag: "drop the feature", which discards the only evidence that the feature changed. Scoring: full credit for the three causes and the mass print.

**Q8. You run a 30-configuration search and the best validation AUC is 0.031 above the baseline. What do you do?**
Answer: treat the 0.031 as noise until an outer holdout confirms it, because at `sigma = 0.008` the expected maximum over 30 candidates is `0.008 * 3.96 = 0.0317`. Reasoning: the maximum of 30 noisy estimates is biased upward by construction; the observed gain equals the expected inflation exactly. Follow-up: how large must the holdout be to resolve a true 0.01 difference? Red flag: "the search found a better model", which mistakes order statistics for evidence. Scoring: full credit for the `sigma * E[max Z]` computation.

**Q9. Explain why a compressed score can have the same AUC and a worse Brier score, with numbers.**
Answer: AUC depends only on the ordering, and compression is monotone, so AUC stays 0.6890. Brier depends on the heights: compressing band means from 0.03–0.22 to 0.06–0.18 raises REL from 0 to 0.00076 and Brier from 0.09492 to 0.09568, with ECE rising from 0 to 0.024. Reasoning: `REL = sum w_k (p_k - o_k)^2` is zero only when the predicted height equals the observed rate. Follow-up: what map would repair it, and how many parameters does it need? Red flag: "AUC and Brier measure the same thing". Scoring: full credit for the monotone-invariance argument plus one concrete pair of numbers.

**Q10. Your fitted prevalence is 0.10 but the population prevalence is 0.112. What happened and what is the consequence?**
Answer: 8% of rows were dropped, and their default rate is higher than average — with 160 rows at 25% and 1,840 at 10%, the fitted rate is exactly 0.100 while the population rate is 0.112. The consequence is a systematically low intercept (`logit(0.10) = -2.197` versus `logit(0.112) = -2.070`, a 0.127 logit error) and a model that under-predicts risk for the excluded group. Reasoning: the fitted rate is a weighted average over the retained rows only. Follow-up: how do you test whether the drop is missing-at-random? Red flag: "the difference is sampling noise", since 160 rows at a 2.5x rate ratio is a chi-square of 33.3 on one degree of freedom. Scoring: full credit for the drop inference, the intercept consequence, and the MAR test.

**Q11. A feature has 12 levels and ranks fourth in gain importance with an 8% share. What is your prior?**
Answer: scepticism. A random 12-level feature's expected share of total gain is about `1/12 = 0.083`, so an 8% share is exactly what noise produces. Require held-out permutation importance above the `1/n_features` noise band, about 4.5% at 22 features, before treating the feature as informative. Reasoning: impurity gain rewards the number of candidate cut points and evaluates on training rows. Follow-up: how do you choose between dropping the feature and re-encoding it? Red flag: accepting gain importance as evidence of a relationship. Scoring: full credit for the 1/12 expectation and the permutation cross-check.

**Q12. Production latency rose 8 ms to 90 ms after adding one field. Give three mechanisms and the measurement that separates them.**
Answer: one-hot expansion of a high-cardinality field (measure the vector width and parameter count), a lookup miss falling through a slow default path (measure the miss rate and the per-stage latency), or a feature fetch now on the critical path (profile: model 5 ms plus fetch 80 ms). The separating measurement is a per-stage latency profile plus the vector width. Reasoning: three mechanisms with the same aggregate symptom and different fixes; the profile localises the time and the width localises the memory. Follow-up: what budget would you assert in CI? Red flag: "add more replicas", which scales a per-request cost that should be removed. Scoring: full credit for all three mechanisms and the profile.

**Q13. State the difference between leakage and dependence, and give one check for each.**
Answer: leakage is information unavailable at `t_d`, or from the row's own label, reaching the fit; dependence is training and validation rows being statistically non-independent. Check leakage by comparing every feature's event timestamp to `t_d` and by refitting transforms per fold; check dependence by deduplicating on the natural key and grouping the split, then comparing metrics. Reasoning: the corrections differ — leakage requires rebuilding the feature, dependence requires changing the split — so collapsing them sends the fix to the wrong layer. Follow-up: can a pipeline have dependence without leakage? Red flag: using the two words interchangeably. Scoring: full credit for both definitions and one executable check each.

**Q14. You must defend a threshold of 0.1667 in review. What four numbers do you bring?**
Answer: the cost ratio (1000:200) and the derivation `200/1200`; the policy comparison `(224,000 - 198,400) = 25,600` saved over 2,000 applicants, i.e. 12.80 each; the sensitivity range where the decision flips (`C_FP` between 176.47 and 282.05); and the calibration evidence — the band table with REL 0 and Brier 0.09492, plus the AUC interval 0.649 to 0.729 so the reviewer knows the precision of the ranking claim. Reasoning: a threshold is defensible only with its cost basis, its quantified benefit, its robustness range, and evidence that the score is a probability. Follow-up: what do you say if the reviewer disputes `C_FP`? Red flag: presenting the threshold as a tuned hyperparameter. Scoring: full credit for all four, with the sensitivity range as the discriminating item.

## 12. Quick Reference Card

| If you see | Suspect | Compute | Then |
|---|---|---|---|
| training 0.96, forward 0.66 | CI-002, CI-001 | `max(event_ts)` vs `t_d` | rebuild the feature as-of `t_d` |
| AUC falls when the seed changes | CI-003, CI-005 | dedupe on the key, group split | `GroupKFold` on the entity key |
| AUC falls when preprocessing moves inside the fold | CI-004 | refit each transform, measure delta | pipeline takes raw data plus fold |
| precision 0.34 pilot, 0.05 live | CI-024, CI-023 | `tpr*pi/(tpr*pi + fpr(1-pi))` | re-solve the threshold at deploy `pi` |
| flag rate 8x the design rate | CI-011 | `t = t_w/(w - (w-1)t_w)` | convert, or drop the weights |
| fitted prevalence 0.100 vs population 0.112 | CI-013 | row count per stage, chi-square on missingness | impute plus indicator, keep the row |
| best-of-30 search gains 0.03 | CI-017 | `sigma * E[max Z_30] = 0.008*3.96` | seal an outer holdout |
| `inf` in a drift report | CI-014 | per-bucket masses | floor at `1e-4`, drop empty-both buckets |
| macro 0.63, micro 0.79 | CI-025 | per-class precision and recall | report per class plus one named aggregate |
| "AUC 0.71, so it is calibrated" | CI-027 | 5-band predicted versus observed, REL and ECE | fit Platt or isotonic, re-solve `t` |

Three constants to carry: `p* = C_FP/(C_FP + C_FN) = 0.1667` at 200:1000; `SE(AUC) = 0.0205` on 224 positives and 1,776 negatives, so ignore differences below about 0.04; and expected selection inflation of `0.008 * 3.96 = 0.0317` for any 30-configuration search at that noise level.
