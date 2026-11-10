---
title: "Imbalance and Calibration"
topic: "Class imbalance, prior shift, resampling, and probability calibration at the decision threshold"
collection: "supervised-ml-and-feature-engineering"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Imbalance and Calibration

> **Purpose:** given a skewed label, a stated decision time, a cost structure, and a review capacity, decide whether the score may be read as a probability, where the cut belongs, which sampling artefact must be undone first, and whether the calibration error is large enough to change the action.
> **Evidence:** formulas and their rearrangements are DERIVED; all arithmetic in Sections 4, 5, and 9 is recomputed by hand and internally cross-checked; every workload (counts, costs, capacities) is ILLUSTRATIVE and does not describe any real portfolio, queue, or company.
> **This file does NOT cover:** designing an online experiment; causal attribution of harm; fairness certification; the choice of model family (see the linear/tree/boosting sheet) or the construction of the validation split (see the validation sheet).

## 1. File Header

### 1.1 What decision this sheet changes

A skewed label distribution is not a defect in the data. It is a fact about the world that makes three separate engineering choices easy to confuse: where the threshold sits, how the training sample was reweighted, and whether the score is allowed to be called a probability. Teams that conflate the three ship systems whose offline numbers cannot be reproduced online, and they usually blame the model.

The decisions this sheet is written to settle are the following, and each has a different owner.

1. **Reading mode.** Is the output consumed as a ranking (top-k into a queue) or as a probability (multiplied by a cost and compared with a review price)? A ranking is invariant to any strictly increasing transform; a probability is not.
2. **Deployed prior.** What fraction of the scored population is positive, on the same clock and population definition the model will meet in production?
3. **Operating point.** Which cut follows from the error costs, and does that cut exceed the capacity of the process that must service it?
4. **Sampling debt.** If the training sample was resampled, by what factor was the prior changed, and has that factor been repaid before the cut was computed?
5. **Calibration adequacy.** Is the reliability error small enough that the cut derived from probabilities matches the cut derived from the observed confusion counts?
6. **Cohort sufficiency.** Does the evaluation cohort contain enough positives for the rate estimates to survive their own confidence intervals?

A fourth decision hides inside the fourth: whether the imbalance was ever the problem. In most tabular systems with a positive rate above roughly one percent, the imbalance is a nuisance for the loss function and irrelevant for the ranking. Below one percent, the binding constraint becomes the number of positives, not their share, and no resampling technique manufactures the missing evidence.

### 1.2 Audience and assumed starting point

The reader should already know what a confusion matrix is, should have fitted a logistic regression or a gradient-boosted tree, and should be comfortable with the idea that precision and recall trade off along a threshold sweep. What is presumed missing is the discipline of treating the prior, the sampling scheme, the calibrator, and the cost ratio as one coupled system rather than four independent knobs.

Two habits are corrected explicitly. The first is quoting a single metric without its threshold, its prevalence, and its cohort size; Section 4 shows why each of the three changes the number. The second is treating calibration as a cosmetic post-processing step; Section 5 shows a cohort where recalibration moves the cost-optimal cut through a factor of eleven in probability and leaves the ranking untouched.

### 1.3 Scope boundary

| In scope | Out of scope | Where the excluded topic lives |
|---|---|---|
| Threshold derivation from costs and capacity | Model family choice and regularisation paths | planned `03_Linear_Trees_and_Boosting.md` |
| Prior correction after undersampling or oversampling | Split construction under time and entity dependence | [01 Model Validation and Error Analysis](01_Model_Validation_and_Error_Analysis.md) |
| Reliability diagnostics: ECE, MCE, Brier decomposition | Loss derivation from the business decision | [02 Problem Formulation and Losses](02_Problem_Formulation_and_Losses.md) |
| Platt, isotonic, and temperature recalibration | Feature transformation design | planned `04_Feature_Engineering_Patterns.md` |
| Rare-event variance and cohort sizing | Power analysis for randomised experiments | module 02 experimentation material |
| Resampling mechanics and their side effects | Synthetic data generation for privacy | data-governance material |

### 1.4 Evidence policy used in this file

Every number carries a label at the point of use. `DERIVED` means the identity follows from the stated assumptions and the algebra is shown. `VERIFIED` means the arithmetic was recomputed at least twice by different routes and both routes agree; the Section 5 check block records the second route explicitly. `ILLUSTRATIVE` means the inputs are constructed for teaching and describe no real system.

The distinction is load-bearing. A precision of `0.1541` computed from constructed counts is exact arithmetic on an invented cohort; quoting it as an industry benchmark would fabricate evidence. The same caution applies to the cost figures: the ratio nineteen to one used in Section 5 is a teaching input, not a measurement of any organisation's loss given default.

Method references are cited only for the specific claim they support. A citation to the original resampling paper does not endorse the resampling ratio used later, because the ratio is a modelling choice the paper does not discuss.

### 1.5 Notation convention

All display mathematics uses fenced `text` blocks with stable identifiers of the form `IMB-NN`. Every formula block repeats the definition of each symbol inside the block, so a block copied out of context remains readable. Rates are dimensionless proportions in `[0,1]`; counts are integers of records; monetary values use an unnamed currency unit `CU`; capacities are records per operational interval.

Logits are written `z = ln(p/(1-p))` and expressed in nats, the natural unit implied by the natural logarithm. A shift of one nat multiplies the odds by `e = 2.718282`. This unit matters because every prior correction in Section 4 is an additive shift in logit space, and saying "shift the intercept by `-2.5055`" is more precise than saying "correct for sampling".

### 1.6 Minimum viable imbalance record

Before any rate is computed, eight fields must be written down. A result missing any of them is a number without a claim attached.

| Field | What it fixes | The specific failure when it is absent |
|---|---|---|
| Positive definition | Which outcome counts as the event | A refund and a chargeback get merged, changing the prior |
| Prior `pi` and its denominator | The population the rate describes | A portfolio-wide prior is applied to one product line |
| Sampling scheme | Whether the training prior was altered | A model trained at 20% positive is read at 2% |
| Resampling factor `f` | The size of the logit debt to repay | Undersampling by 12.25 shifts every score upward |
| Score semantics | Ranking value or probability | A margin is multiplied by a cost as if it were a probability |
| Calibration population | Where the calibrator was fitted | A calibrator fitted on last quarter is applied to this one |
| Cost ratio | Where the cut belongs | The cut is inherited from a library default of 0.5 |
| Capacity and interval | Whether the cut is serviceable | A cost-optimal cut floods a queue that handles a tenth of it |

Two of these fields are frequently recorded and six are frequently forgotten. The sampling scheme and the resampling factor are the most commonly missing pair, and they are the pair that produces the largest systematic error, because a forgotten factor of `f` moves every score by `ln(f)` nats without changing the ranking, so nothing looks wrong in an AUC report.

### 1.7 Reading path under time pressure

If a decision is needed within the hour, run the following sequence and stop when a constraint binds. Confirm the prior and the positive definition. Confirm whether the training sample was resampled. Compute the cost cut with `IMB-05` and compare it with the capacity cut with `IMB-06`. Check the reliability table for the bins that straddle the chosen cut, because that is where calibration error changes the action. If any of those four steps cannot be completed, the correct output is a stated gap rather than a threshold.

**Takeaway:** Imbalance changes the loss and the threshold; it does not change the ranking. Calibration changes the threshold; it does not change the ranking. Capacity changes the action set; it does not change either. Keep the three separate.

## 2. Concept Map / Mental Model

### 2.1 One score, three independent readings

The output of a classifier is a real number per record. Three readings are possible, and each is legitimate only under its own conditions.

The **ranking reading** uses only the order of the scores. It is invariant to any strictly increasing transformation, which is why ROC area and lift survive resampling, recalibration, and monotone rescaling untouched. Its validity condition is that the transform be applied uniformly across the scored population; a transform applied only to one segment destroys comparability between segments.

The **probability reading** uses the numeric value. It requires calibration on the population being scored, at the revision being deployed. Its validity condition is the strongest of the three, because it is a statement about a conditional expectation, not about an order.

The **decision reading** compares the probability with a cut. The cut has two independent sources: the cost ratio, which says what the business prefers, and the capacity, which says what the process can absorb. The tighter of the two governs.

```text
        raw score s(x)
             |
             | monotone map, ranking preserved
             v
   +----------------------+        +---------------------------+
   | ORDER                |        | VALUE                     |
   | s_(1) >= s_(2) >= ...|        | p(x) = P(Y=1 | x, rev)    |
   | survives: resampling |        | survives: nothing untested|
   |           recalibr.  |        | requires: calibration on  |
   |           rescaling  |        |           the deployed    |
   |           link change|        |           population      |
   +----------+-----------+        +-------------+-------------+
              |                                  |
              |                                  | compare with cut
              v                                  v
        top-k queue  <---- min(capacity, cost cut) ----  action set
```

### 2.2 What imbalance actually breaks

Imbalance breaks three things, and it is worth naming them precisely because the remedies differ.

It breaks the **gradient balance** of training. With a positive rate of two percent, the average loss is dominated by negatives, and a model that predicts a constant low value is close to a local optimum. Remedies: class weights, focal loss, resampling, or a loss whose gradient is not dominated by the majority.

It breaks the **interpretation of aggregate rates**. Accuracy, and to a lesser degree AUC, are dominated by the majority class. A model that never fires can reach ninety-eight percent accuracy on a two-percent cohort, and Section 5 shows a model at fifty percent threshold that is *worse* than that trivial baseline on accuracy while being far more useful.

It does not break **ranking** at all. If the positive rate changes but the class-conditional score distributions do not, the ordering of records by their posterior is unchanged. This is the single most useful structural fact in the sheet: prior shift moves the cut, never the order.

### 2.3 The chain of four populations

Four populations appear in every deployment, and a rate computed on one is routinely reported as if it described another.

The **training population** is what the coefficients saw, possibly resampled. The **evaluation cohort** is the held-out slice used to estimate rates. The **scored population** is what the model sees in production on a given day. The **actioned subset** is the part of the scored population that clears the threshold and receives review.

A precision computed on the actioned subset is not an estimate of the model's quality; it is an estimate of the queue's yield, and it moves when the threshold moves even if the model does not. A recall computed on the evaluation cohort is a property of the model, the threshold, and the cohort's prevalence together. Confusing these is Issue `CI-30` in the companion catalogue.

### 2.4 Calibration is a conditional statement

Saying "the model is calibrated" is shorthand for a statement with four qualifiers: on this population, at this revision, over this score range, the observed frequency among records with score near `p` equals `p`. Drop any qualifier and the claim is unfalsifiable.

The qualifier that bites hardest is the score range. Calibration is usually verified in bins, and the bins with the most records are the low-score bins where almost nothing happens. A model can be excellently calibrated below `0.2` and badly overconfident above `0.6`, and the aggregate reliability error will still look small because the high bins hold few records. Section 5 exhibits exactly this pattern: the top bin holds six hundred of fifty thousand records and carries one-fifth of the total reliability error.

### 2.5 Why the cost ratio and the capacity must both be in the picture

The cost ratio answers "which records are worth acting on". Capacity answers "which records can be acted on". These are different questions with different owners, and a review of a model that reports only one of them is incomplete.

When the cost cut is tighter than capacity, the queue is under-filled and the honest report includes the value left on the table from records that were worth actioning but were not actioned. When capacity is tighter, the queue is truncated and the honest report includes the value of the truncated tail. Section 5 computes both quantities for one cohort, and the numbers differ by a factor large enough to change a staffing decision.

### 2.6 The mental model in one sentence

Keep the order, repair the prior, verify the value on the bins that decide, and let the tighter of cost and capacity choose the cut.

**Takeaway:** Four populations, three readings, two constraints, one order. Every mistake in this sheet is a failure to keep one of those counts straight.

## 3. Foundational Definitions

### 3.1 The terms, stated so they can be falsified

Each definition below names the quantity, its unit, and the observation that would refute a claim about it. A definition without a refutation test is a slogan.

**Prevalence (prior) `pi`.** The share of positives in a named population over a named window: `pi = K / N`, dimensionless, in `[0,1]`. Refuted by: recomputing `K` under a different eligibility filter and finding the rate moves by more than its confidence interval.

**Base rate of the scored population.** The prevalence among records that actually reach the scorer. It differs from the training prior whenever the queue is filtered before scoring, and that difference is the single most common cause of a cut that is wrong by a factor of three.

**Class imbalance.** A prevalence far from one half. There is no universal threshold, but the operational consequences change character around `pi = 0.01`: below it, the binding constraint is usually the absolute count of positives rather than their share.

**Score.** The real-valued output before any decision. Unitless unless the model's link function is declared. A score is only a probability when the model emits one and the calibration has been verified on the deployment population.

**Calibration.** For every `p` in the score range, `P(Y = 1 | s(x) = p) = p` on the stated population at the stated revision. Verified only in bins, so the operational claim is always "calibrated to within the bin resolution used".

**Reliability diagram.** A plot or table of observed frequency against mean predicted probability per bin. The diagonal is the claim; the vertical distance per bin is the error.

**Expected calibration error (ECE).** The record-weighted mean absolute gap across bins. Reported with the binning scheme, because the number is a function of the bins.

**Maximum calibration error (MCE).** The largest absolute gap over bins. Sensitive to small bins, which is both its diagnostic value and its trap.

**Sharpness.** The spread of predicted probabilities, independent of calibration. A model that outputs `0.03` for every record is perfectly calibrated and useless; sharpness is what makes a calibrated forecast informative.

**Proper scoring rule.** A loss minimised in expectation by the true probability. Log loss and Brier score are proper; accuracy is not.

**Refinement.** The part of forecast value attributable to sorting records into groups with different frequencies, as opposed to getting the overall level right.

**Resampling.** Any procedure that changes the empirical class ratio of the training sample: undersampling the majority, oversampling the minority, or synthesising new minority records.

**Sampling debt.** The logit offset introduced by resampling that must be repaid before scores are read as probabilities. Measured in nats; equal to `ln m` after oversampling by `m` and `-ln u` after undersampling the majority by `u`.

**Class weight.** A per-class multiplier on the loss. Algebraically equivalent to changing the decision threshold, not to changing the data.

**Threshold `tau`.** The cut applied to the score or probability. A property of the decision, not of the model.

**Operating point.** The triple (threshold, actioned count, resulting confusion counts) reported together. A precision quoted without the other two is not an operating point.

**Cost ratio.** `c_FP / c_FN`, dimensionless, the exchange rate between a wasted action and a missed positive.

**Capacity.** Records per interval the downstream process can service. A resource constraint, not a preference.

**Prior shift.** A change in prevalence between the population where the model was fitted and the population where it is applied, with class-conditional distributions unchanged.

**Covariate shift.** A change in the distribution of inputs, with the conditional `P(Y | X)` unchanged. It affects both ranking and calibration, unlike pure prior shift.

### 3.2 Three quantities that share the word "weight"

Practitioners say "weight" for three different objects and the confusion produces real bugs.

*Class weight* multiplies the loss of a class. With weight `w` on positives, the fitted model's log-odds are shifted by `ln w` relative to the unweighted fit, which is why `w = 19` and a threshold of `0.5` encode the same rule as `w = 1` and a threshold of `0.05`.

*Sample weight* attaches an importance to individual records, usually from survey design or from a known selection mechanism. A sample weight that varies within a class is not equivalent to a threshold shift; it changes the fit and can change the ranking.

*Recency weight* down-weights old records. It changes the population the model describes and therefore the prior it implicitly learns, but it does so through the fit rather than through a clean multiplicative factor.

Only the first is exactly interchangeable with a threshold. Section 4.1 derives the identity, and Section 8 records the dossier where a team applied a sample weight and then read the result as if it were a class weight.

### 3.3 Hard negatives, easy negatives, and why the count of positives matters more than the ratio

The 48,500 negatives in the illustrative cohort of Section 5 are not 48,500 equally informative records. Most are obviously negative: an application with a stable twelve-month history, no prior disputes, and a small exposure. A small number are near-misses that carry almost all of the learning signal. The effective sample size for the decision boundary is closer to the near-miss count than to the total.

This changes the remedy. If the difficulty is that easy negatives flood the gradient, more negatives add almost nothing and down-weighting them is cheap and safe. If the difficulty is that there are only 1,500 positives and the model needs to resolve behaviour within that set, no amount of negative manipulation helps, and the binding action is to obtain more positives: a longer window, a wider population, or a lower-cost label.

A useful diagnostic: fit on all negatives, then fit on a random 10% of negatives, and compare the top-decile ranking. If the ranking barely moves, the discarded negatives were easy and the 90% removal cost nothing but variance. If the top-decile overlap drops below roughly `0.9`, the negatives were carrying decision information and undersampling is buying variance it cannot afford.

### 3.4 Labels: the clock, and what a negative means

A negative is a record that did not produce the event within the label horizon. That definition bundles three distinct states: confirmed non-event, not-yet-matured, and lost to follow-up. Treating all three as negative biases the fitted prior downward and biases the calibration curve at exactly the low-probability end where most records live.

The remedy is procedural rather than statistical: define a maturity date per record, exclude records whose horizon has not closed, and report the excluded count next to the cohort size. In the Section 5 cohort of 50,000 records, suppose 1,200 were excluded as immature; the reported prior would be `1500/50000 = 0.03` only if the immaturity were unrelated to the outcome. If immature records are disproportionately positive, which is the normal case when the outcome is a delayed default, the true prior exceeds the reported one, and every cost-derived cut computed on the biased prior is too high.

### 3.5 Ranking metrics and their invariance

Three invariance facts do most of the work in this sheet, and each is worth stating in the form in which it can be checked.

**Monotone invariance.** ROC area, precision-recall area, and lift depend only on the ordering of scores. Any strictly increasing transform applied uniformly leaves them unchanged. Consequence: resampling, recalibration, and rescaling cannot improve these metrics, and a report claiming they did is reporting noise or a leak.

**Prevalence dependence of precision.** For a fixed class-conditional score distribution, precision at a fixed threshold falls as prevalence falls, while recall is unchanged. Consequence: a precision measured on an enriched cohort of 30% positives overstates the precision that the same model delivers at 3% by roughly the ratio of priors, adjusted for the threshold's position.

**Prevalence invariance of the ordering.** Under pure prior shift, `P(Y=1|x)` changes by a constant in logit space for every `x`. The ordering is therefore preserved exactly, and so are ROC area and the class-conditional distributions. Consequence: prior shift must be handled by moving the cut, never by retraining on reweighted data, unless there is separate evidence that the class-conditional distributions moved too.

### 3.6 What "recalibration" can and cannot fix

Recalibration is a monotone map fitted on held-out data that turns a score into a probability. It can fix a systematic bias, an over- or under-dispersion of the logit scale, and a link-function mismatch. It cannot fix the following, and no amount of fitting will make it.

It cannot fix a ranking that is wrong, because it is constrained to be monotone; the ordering survives by construction. It cannot fix a missing feature, because the map sees only the score. It cannot fix prior shift unless the map is fitted on data from the deployment prior or the shift is corrected analytically. It cannot fix label immaturity, because the map is fitted against the same labels. It cannot repair a cohort with too few positives to estimate the map, which is why Section 9 treats the calibrator's own sample requirement as a cost.

### 3.7 The three-way split a calibrator needs

A calibrator fitted on the same records used to fit the model reproduces the model's own overconfidence and looks better than it is. The standard remedy is a third slice: fit on train, calibrate on a held-out calibration slice, evaluate on test. With 50,000 records and 1,500 positives, a 20% calibration slice holds 300 positives, which is enough for a two-parameter Platt fit and marginal for isotonic regression with many bins.

Isotonic regression's flexibility is its hazard. On the Section 5 bins it drives ECE to exactly zero on the fitting data, because the pooled-adjacent-violators solution can reproduce any monotone frequency sequence; on a held-out slice it typically beats Platt only when each bin holds enough positives to estimate its own rate stably. The rule of thumb worth carrying: below roughly 200 positives in the fitting slice, prefer the parametric map.

### 3.8 Capacity, service level, and the honest report of a truncated queue

Capacity is a count per interval. When the cost-optimal action set exceeds it, the deployed policy actions the highest-scoring `C` records. The report should then contain four numbers rather than one: the actioned count, the precision in the actioned set, the recall achieved, and the value of the records that cleared the cost cut but were truncated.

That fourth number is the one that changes staffing decisions, and it is the one most often omitted. In Section 5 the truncated tail is worth `1,600 CU` per period, which annualises to `400,000 CU` at 250 periods, and that figure is the correct input to an argument about hiring reviewers.

### 3.9 Definitions that are routinely swapped

**Precision versus recall denominator.** Precision divides by the actioned count; recall divides by the positive mass. Swapping them turns a recall of `0.4347` into a precision of `0.4347`, which happens to look plausible and is wrong by a factor of four.

**ECE versus MCE.** ECE is weighted by bin population; MCE is not. A model with a single catastrophic bin holding 0.4% of records has a small ECE and a large MCE, and the choice of which to report determines whether the defect is visible.

**Undersampling versus down-weighting.** Discarding records changes the sample; multiplying their loss does not. The first loses information and inflates variance; the second does not.

**Prior shift versus covariate shift.** Prior shift preserves the ordering and only moves the cut; covariate shift can destroy the ordering. Applying a prior correction to a covariate shift produces a confidently wrong probability.

**Calibration versus discrimination.** A model can be perfectly calibrated and deliver no separation, and it can separate well while being badly biased. The first is useless for a queue; the second is usable only after recalibration or threshold repair.

**Takeaway:** Every term above is defined by the observation that would refute it; if you cannot name that observation, you are holding an opinion rather than a measurement.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: prior, odds, and the cut that follows from costs

The base case fixes everything except the threshold. Assume a calibrated score, constant per-error costs, no capacity limit, and independent records.

**Validity before use.** The derivation assumes the score is calibrated on the population being scored. If the score is a margin or an uncalibrated probability, the algebra still produces a cut, but the cut is in score units and cannot be compared with a cost ratio expressed in probabilities.

```text
FORMULA [IMB-01]: prior and odds
WHERE: pi [dimensionless] = K / N, with K [records] positives and N [records] total;
       odds = pi / (1 - pi) [dimensionless, >= 0]; logit = ln(odds) [nats].
CONSTRAINT: K and N must come from the same population and the same label horizon;
            records with immature labels must be excluded, not counted as negative.
NUMERICAL EXAMPLE: K = 1500, N = 50000;
                   pi = 1500/50000 = 0.030000;
                   odds = 0.03/0.97 = 0.030928;
                   logit = ln(0.030928) = -3.475963 nats.
INTERPRETATION: one positive per 32.3 negatives. The logit is the coordinate in
                which every correction in this sheet is a simple addition.
```

```text
FORMULA [IMB-02]: cost-optimal cut
WHERE: c_FP [CU per wasted action] cost of acting on a negative;
       c_FN [CU per missed positive] cost of not acting on a positive;
       tau [probability] the cut.
DERIVATION: act if (1-p)*c_FP < p*c_FN  =>  p > c_FP/(c_FP+c_FN).
CONSTRAINT: p calibrated on the scored population; costs constant and additive;
            no interaction between decisions; capacity not binding.
NUMERICAL EXAMPLE: c_FP = 4 CU (3 review time + 1 friction),
                   c_FN = 76 CU (70 loss given default + 6 recovery cost);
                   tau = 4/(4+76) = 4/80 = 0.050000.
INTERPRETATION: act on records with calibrated probability above 5%. The ratio
                80/4 = 20 is the exchange rate; the cut is its reciprocal.
```

```text
FORMULA [IMB-03]: capacity cut and the effective operating point
WHERE: s_(C) [score units] the C-th largest score; C [records/interval] capacity;
       tau_cost [probability] from IMB-02; tau_eff [probability] the deployed cut.
DERIVATION: score ordering is preserved, so admissible action sets are prefixes;
            the largest admissible prefix is min(|A(tau_cost)|, C).
CONSTRAINT: one model revision scores the whole population in the interval; ties at
            s_(C) broken by a stated deterministic rule; C matches the interval.
NUMERICAL EXAMPLE: tau_cost = 0.05 admits 6000 records, C = 4000;
                   tau_eff = s_(4000) = 0.277659 in raw score units
                   (= 0.065487 in calibrated probability units).
INTERPRETATION: capacity binds; the queue is truncated by 2000 records whose
                average surplus is 0.80 CU each, so 1600 CU of value per period
                is left unserved and must be reported.
```

```text
FORMULA [IMB-04]: expected cost at an operating point
WHERE: FN [records] positives not actioned; FP [records] negatives actioned;
       Cost [CU per period].
CONSTRAINT: additive and linear in counts; excludes fixed staffing and the cost of
            running the evaluation; excludes the value of true positives because
            it is constant across thresholds only if every positive is equally
            valuable, which must be stated.
NUMERICAL EXAMPLE: FN = 1500-652 = 848, FP = 6000-652 = 5348;
                   Cost = 76*848 + 4*5348 = 64448 + 21392 = 85840 CU.
INTERPRETATION: 75.1% of the cost is missed positives; effort on recall is
                better rewarded than effort on precision at this cost ratio.
```

**Boundary/counterexample for the base case.** Let the cost ratio approach one, so a wasted action costs as much as a missed positive. Then `tau = 1/2`, and the cost-optimal rule degenerates to majority classification at the calibrated probability. On the Section 5 cohort, that cut actions 3400 records with 496 positives, precision `0.145882`, and cost `76*1004 + 4*2904 = 87920 CU` — higher than the 85,840 CU achieved at `0.05`, which is the arithmetic proof that equal costs are not the same as the business's costs. The formula breaks in a second direction when `c_FP = 0`: the cut becomes zero and every record is actioned, including the 47,000 records with probabilities below one percent, at a review cost that no process can absorb.

### 4.2 General case: sampling debt, prior shift, and the identity that ties weights to cuts

The general case admits three modifications of the training sample and one modification of the deployment population, and each contributes an additive term in logit space. The value of writing them as additions is that they compose, invert, and can be checked against each other.

**Validity before use.** All three corrections assume that the class-conditional distributions `P(X | Y=1)` and `P(X | Y=0)` are unchanged between the sample and the deployment population. The correction is exact for prior change and for sampling that only changes the class ratio; it is wrong for covariate shift.

```text
FORMULA [IMB-05]: prior correction in logit space
WHERE: pi_fit [dimensionless] prior of the fitting sample; pi_dep [dimensionless]
       prior of the deployment population; p_fit [probability] the model's output;
       p_dep [probability] the corrected output; delta [nats].
DERIVATION: p_dep/(1-p_dep) = [p_fit/(1-p_fit)] * [odds(pi_dep)/odds(pi_fit)],
            because the likelihood ratio P(x|Y=1)/P(x|Y=0) is unchanged and the
            posterior odds are the prior odds times that ratio.
CONSTRAINT: pure prior shift only; the model must emit a probability or a logit on
            a known scale; pi_dep must be measured on the deployment population.
NUMERICAL EXAMPLE: pi_fit = 0.30, pi_dep = 0.03;
                   delta = ln[(0.03/0.97)/(0.30/0.70)] = ln(0.030928/0.428571)
                         = ln(0.072166) = -2.628801 nats;
                   a fitted p_fit = 0.05 becomes p_dep = sigmoid(-2.944439-2.628801)
                         = sigmoid(-5.573240) = 0.003784.
INTERPRETATION: the same score means a probability 8.4 times smaller at the lower
                prior. Reading it unchanged actioned 30,000 records instead of the
                6,000 the cost model supports.
```

```text
FORMULA [IMB-06]: oversampling debt and its repayment
WHERE: m [dimensionless, > 1] factor by which positives were replicated;
       z [nats] fitted logit; z_corrected [nats] after repayment.
DERIVATION: replicating a positive m times multiplies its likelihood contribution
            by m, which adds ln m to the fitted log-odds for every record.
CONSTRAINT: replication, not synthesis. SMOTE-style interpolation changes the
            class-conditional distribution and is not exactly invertible.
NUMERICAL EXAMPLE: m = 10, pi = 0.03;
                   inflated prior = (0.03*10)/(0.03*10+0.97) = 0.30/1.27 = 0.236220;
                   debt = ln 10 = 2.302585 nats;
                   a raw cut of 0.5 under the inflated model corresponds to
                   p = sigmoid(-2.302585) = 1/11 = 0.090909.
INTERPRETATION: leaving the debt un repaid inflates every probability by a factor
                of 10 in odds, so a model that looks conservative at 0.09 is being
                read as if it were at the 0.5 cut.
```

```text
FORMULA [IMB-07]: undersampling debt and its variance price
WHERE: u [dimensionless, > 1] factor by which negatives were discarded;
       r [records] negatives retained out of R [records].
DERIVATION: discarding negatives multiplies the negative likelihood contribution
            by 1/u, subtracting ln u from the fitted log-odds.
CONSTRAINT: discard uniformly at random; a stratified or hard-negative discard
            changes the class-conditional distribution and cannot be repaid.
NUMERICAL EXAMPLE: negative retention 1/12.25, so u = 12.25, ln u = 2.505526;
                   prior becomes 0.03/(0.03 + 0.97/12.25) = 0.03/0.109184 = 0.274766;
                   a raw cut of 0.5 corresponds to p = 12.25/13.25 = 0.924528.
INTERPRETATION: undersampling inflates apparent probability the same way, and it
                pays a second price: discarding 47000 of 48500 negatives inflates
                the variance of every estimated rate by 48500/1500 = 32.33, so
                standard errors grow by sqrt(32.33) = 5.686.
```

```text
FORMULA [IMB-08]: class weight as a threshold
WHERE: w [dimensionless, > 0] weight on the positive class; tau_equiv [probability].
DERIVATION: weighting positives by w multiplies their odds by w, equivalent to
            shifting the logit by ln w; a fixed raw cut t under weight w therefore
            corresponds to the unweighted cut t/(t + w(1-t)).
CONSTRAINT: w must be a class-level constant; per-record weights are not
            interchangeable with a threshold; the loss must be log loss or another
            properly specified likelihood.
NUMERICAL EXAMPLE: w = 19, t = 0.5:
                   tau_equiv = 0.5/(0.5 + 19*0.5) = 0.5/10 = 0.050000;
                   inverse: to realise tau = 0.05 at t = 0.5, w = (1-0.05)/0.05 = 19.
INTERPRETATION: the cost-optimal cut 0.05 and a class weight of 19 are the same
                decision rule. Choosing the weight and the cut are one decision, not
                two, and setting both independently double-counts.
```

**Boundary/counterexample for the general case.** The identity in `IMB-08` fails as soon as the model is not a probabilistic classifier. For a margin-based model with a hinge loss, no weight corresponds to any threshold, because the loss does not define a likelihood and the output is not a posterior. It also fails when the weight is applied together with resampling: weighting positives by 19 in a sample already oversampled tenfold yields an effective debt of `ln 190 = 5.247024` nats, and a team that repays only one of the two terms has an error of `ln 19 = 2.944439` nats in every probability, a factor of 19 in odds, while the ranking stays perfect and the AUC report stays clean.

### 4.3 Variants: reliability measures, calibrator families, and interval estimates

Each variant below answers a question the base case cannot, and each carries its own invalidation.

```text
FORMULA [IMB-09]: expected calibration error
WHERE: B [bins] a partition of the score range; n_b [records] in bin b; N [records]
       total; acc_b [dimensionless] observed frequency in b; conf_b [probability]
       mean predicted value in b; ECE [dimensionless].
DERIVATION: ECE = sum_b (n_b/N) * |acc_b - conf_b| is the record-weighted mean
             absolute gap; it is a weighted L1 distance between the reliability
             curve and the diagonal.
CONSTRAINT: bins fixed before inspection; every bin non-empty; ECE is not a proper
            scoring rule and can be reduced by a constant predictor.
NUMERICAL EXAMPLE: nine bins over 50000 records give
                   ECE = 0.008400+0.012960+0.018800+0.016480+0.016224
                       +0.013648+0.010780+0.007024+0.002152 = 0.106468.
INTERPRETATION: on average a score is 10.6 percentage points away from the
                frequency it claims; the top bin is the worst offender.
```

```text
FORMULA [IMB-10]: maximum calibration error
WHERE: same bin quantities; MCE = max_b |acc_b - conf_b| [dimensionless].
CONSTRAINT: reported with the bin counts, because MCE is attained on the smallest
            bin as often as on the worst-calibrated one.
NUMERICAL EXAMPLE: bin 0.75-0.90 holds 600 records with 136 positives, so
                   acc = 0.226667 against conf = 0.812000, gap 0.585333;
                   bin 0.90-1.00 holds 200 records with 80 positives, gap 0.538000.
                   MCE = 0.585333, i.e. 5.50 times the ECE.
INTERPRETATION: a single-bin report would call this model unusable; the weighted
                report would call it acceptable. Both are true statements about
                different questions, which is why both are printed.
```

```text
FORMULA [IMB-11]: Brier score and its three-way decomposition
WHERE: p_i [probability] for record i; y_i in {0,1}; N [records];
       Brier = (1/N) * sum_i (p_i - y_i)^2 [dimensionless, lower is better];
       REL [dimensionless] reliability; RES [dimensionless] resolution;
       UNC = pi(1-pi) [dimensionless] uncertainty of the base rate.
DERIVATION: with records grouped by bin, Brier = REL - RES + UNC, where
            REL = (1/N) sum_b n_b (acc_b - conf_b)^2,
            RES = (1/N) sum_b n_b (acc_b - pi)^2.
CONSTRAINT: the identity holds exactly only when conf_b is the bin mean prediction;
            substituting the bin midpoint leaves a third residual term.
NUMERICAL EXAMPLE: REL = 0.028442, RES = 0.001585, UNC = 0.03*0.97 = 0.029100;
                   Brier = 0.028442 - 0.001585 + 0.029100 = 0.055957, which matches
                   the direct computation 0.055957 to six decimals.
INTERPRETATION: RES is 0.0016 against UNC of 0.0291, so almost all of the achievable
                skill is unrealised: the model barely separates the bins.
```

```text
FORMULA [IMB-12]: Platt scaling
WHERE: z [nats] the model logit; A [dimensionless, > 0] slope; B [nats] intercept;
       p_cal [probability].
DERIVATION: p_cal = sigmoid(A*z + B), fitted by minimising log loss on a held-out
             calibration slice; A < 1 means the model was over-dispersed in logit
             space, A > 1 means under-dispersed.
CONSTRAINT: the true calibration curve must be approximately logistic in z; the
            calibration slice must be disjoint from the fitting data and drawn from
            the deployment prior.
NUMERICAL EXAMPLE: a Platt fit on the nine bins gives A = 0.661942, B = -2.311554;
                   at conf = 0.519 (raw score 0.519), z = 0.076055, so
                   p_cal = sigmoid(0.661942*0.076055 - 2.311554)
                         = sigmoid(-2.261207) = 0.094386 against acc 0.092500.
                   ECE falls from 0.106468 to 0.001613.
INTERPRETATION: two parameters removed 98.5% of the reliability error, which is
                evidence that the miscalibration was mostly a scale-and-shift
                defect rather than a shape defect.
```

```text
FORMULA [IMB-13]: temperature scaling
WHERE: T [dimensionless, > 0] the temperature; p_cal = sigmoid(z/T).
DERIVATION: a one-parameter special case of IMB-12 with B fixed at zero, so it can
             correct over-confidence but not a location bias.
CONSTRAINT: only valid when the miscalibration is symmetric in the sense that the
            mean logit error is zero; violated by any genuine prior mismatch.
NUMERICAL EXAMPLE: the best T on the nine bins is 0.685856, giving ECE 0.068822 and
                   Brier 0.052601, versus 0.001613 and 0.027526 for Platt.
INTERPRETATION: temperature alone removes only 35% of the ECE because the defect
                includes an offset of about 2.3 nats; the extra parameter in Platt
                is what buys the rest. A deep-network paper reporting temperature
                scaling as sufficient is reporting on a different defect.
```

```text
FORMULA [IMB-14]: isotonic regression by pooled adjacent violators
WHERE: bins ordered by score with weights w_b and observed rates r_b;
       fit a non-decreasing sequence g_b minimising sum_b w_b (g_b - r_b)^2.
DERIVATION: start with each bin as a block; while any adjacent pair violates
             monotonicity, merge the pair and replace both by their weighted mean;
             the result is the least-squares monotone fit.
CONSTRAINT: needs enough positives per block for the merge to be stable; the fit is
             stepwise, so probabilities are quantised and ties are created.
NUMERICAL EXAMPLE: on the nine Section 5 bins the observed rates are already
                   non-decreasing (0.0100, 0.0200, 0.0305, 0.0410, 0.0600, 0.0925,
                   0.1320, 0.2267, 0.4000), so no merge occurs and ECE is 0.000000
                   by construction, with Brier 0.027515.
INTERPRETATION: a zero ECE here is evidence about the fit's flexibility, not about
                the model's quality. Reporting it as a result is the isotonic trap.
```

```text
FORMULA [IMB-15]: focal loss and its modulating factor
WHERE: p_t [probability] the model's probability for the true class;
       gamma [dimensionless, >= 0] focusing parameter;
       modulating factor = (1 - p_t)^gamma.
DERIVATION: FL = -(1-p_t)^gamma * ln(p_t); the factor vanishes as p_t -> 1, so
             already-correct records stop contributing gradient.
CONSTRAINT: only reduces majority dominance for records the model already handles
            confidently; it does not create signal where the model is unsure.
NUMERICAL EXAMPLE: gamma = 2; a negative at p = 0.02 retains (1-0.02)^2 = 0.960400
                   of its loss; a positive at p = 0.98 retains (1-0.98)^2 = 0.000400.
INTERPRETATION: at 3% prevalence with an under-fitted model, the easy negatives sit
                near p = 0.02 and focal loss barely touches them. It helps only
                after the model has become confident, which is why it is not a
                substitute for the cost cut.
```

```text
FORMULA [IMB-16]: Wilson score interval for an operating-point rate
WHERE: k [records] successes; n [records] trials; z = 1.959964 for 95% coverage;
       centre = (p + z^2/(2n)) / (1 + z^2/n);
       half = z*sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n).
CONSTRAINT: records independent, which fails when one entity contributes several
            rows; then the effective n is the entity count, not the row count.
NUMERICAL EXAMPLE: k = 652, n = 6000, p = 0.108667;
                   centre = 0.108921, half = 0.007896, so the interval is
                   [0.101025, 0.116817], width 0.015792.
INTERPRETATION: the precision estimate carries a plus-or-minus 0.8 point band, so a
                report of "10.9%" should be read as "between 10.1% and 11.7%".
```

**Boundary/counterexample for the variants.** Take the smallest bin: 200 records, 80 positives, gap 0.538. The Wilson interval on that bin's rate is `[0.334612, 0.469161]`, width `0.134549`. The claimed confidence of 0.938 lies far outside that interval, so the miscalibration is real rather than sampling noise — the interval is the test. Now reverse the case: a bin with 200 records and 3 positives has rate `0.015` and interval `[0.005127, 0.043188]`, width `0.038061`; here a 0.03 gap is inside the interval. The same ECE contribution means something entirely different in the two bins, which is why an unweighted mean over bin gaps (0.311 here, 2.92 times the ECE) is not a substitute for a properly weighted one.

### 4.4 Limits and counterexamples

Four limits bound everything above, and each has a concrete breaking case.

**Limit 1: ECE depends on the binning.** With 3 bins instead of 9, the low three bins merge to `n = 40000`, `k = 684`, rate `0.017100` against a mean confidence near `0.06`, and the reported ECE changes materially with no change in the model. Any comparison of ECE across papers must therefore control the bin count, and any internal gate should fix it in advance.

**Limit 2: calibration is not transferable across populations.** Applying the Platt map fitted at 3% prevalence to a segment with 30% prevalence reintroduces a `2.628801`-nat bias. The correct composition is to recalibrate per segment, or to apply `IMB-05` after the map, never the map alone.

**Limit 3: the cost cut assumes a constant marginal cost per error.** When the cost of a false positive rises with exposure, the correct rule is per-record: act if `p_i > c_FP,i/(c_FP,i + c_FN,i)`. Aggregating to a single threshold then systematically under-actions large-exposure records. The Section 5 cohort uses a constant 4 CU for teaching, and a real queue should not.

**Limit 4: independence fails under entity clustering.** If 1,500 positives come from 900 entities, the effective sample size for rate estimates is nearer 900 than 1,500, so the Wilson bands in `IMB-16` are too narrow by roughly `sqrt(1500/900) = 1.291`. Every interval in this sheet assumes one record per entity.

**Takeaway:** Prior correction moves the cut, resampling creates debt that must be repaid in nats, weights and cuts are the same decision, and every reliability number is conditional on its binning.

## 5. Worked Numerical Example

### 5.1 The setting, and what is illustrative about it

A collections team scores 50,000 open accounts at the start of each month and can place 4,000 of them into a hardship-review queue. A placed account that would have defaulted recovers on average `70 CU` of otherwise-lost value net of the intervention's own cost; a placement that would not have defaulted costs `4 CU` in reviewer time and customer friction. The training sample was oversampled tenfold on positives by plain replication and the debt was never repaid. All counts, costs, and capacities below are `ILLUSTRATIVE`; the arithmetic is `VERIFIED`.

The cohort splits 1,500 positives and 48,500 negatives, so `pi = 0.03`. The raw model output is a probability on the *inflated* scale, which means it is not a probability on the deployment scale at all.

### 5.2 Step 1 — establish the prior and the sampling debt

```text
K = 1500, N = 50000  =>  pi = 1500/50000 = 0.030000
negative count = 50000 - 1500 = 48500
oversampling m = 10  =>  debt = ln 10 = 2.302585 nats
inflated prior = (0.03*10)/(0.03*10 + 0.97) = 0.30/1.27 = 0.236220
inflated odds  = 0.236220/0.763780 = 0.309278
deployment odds = 0.03/0.97 = 0.030928
ratio = 0.309278/0.030928 = 10.000000  (the m factor, as expected)
```

The arithmetic confirms that the debt is exactly `ln m`, so the repair is a single subtraction and needs no fitted parameter. The check that the ratio is `10.000000` rather than approximately ten is a useful habit: it catches a formula applied to a rate instead of to odds, which is the most common implementation error in this area.

### 5.3 Step 2 — repair the prior before computing any cut

```text
for each raw score s:  z_rep = ln(s/(1-s)) - 2.302585
                       p_rep = 1/(1 + exp(-z_rep)) = s/(s + 10*(1-s))
check at s = 0.5:      p_rep = 0.5/(0.5+5.0) = 0.090909 = 1/11
check at s = 0.938:    p_rep = 0.938/(0.938+0.62) = 0.602057
check at s = 0.031:    p_rep = 0.031/(0.031+9.69) = 0.003187
```

The repaired values are the ones a deployment may use. Note the middle check: the raw score of `0.5` on the inflated scale is a probability of `0.0909`, not `0.5`, and a team that keeps the default cut of `0.5` is therefore running a cut nearly twice the intended `0.05`. That single un-repaid term is worth more than any model improvement available to this team.

### 5.4 Step 3 — the reliability table on the deployment scale

Nine fixed bins were declared before inspection. Mean raw score per bin, record count, and observed positives:

| Bin (raw score) | Records `n_b` | Positives `k_b` | Observed `acc_b` | Mean score `conf_b` | Gap |
|---|---:|---:|---:|---:|---:|
| 0.00-0.05 | 20,000 | 200 | 0.010000 | 0.031 | 0.0210 |
| 0.05-0.10 | 12,000 | 240 | 0.020000 | 0.074 | 0.0540 |
| 0.10-0.20 | 8,000 | 244 | 0.030500 | 0.148 | 0.1175 |
| 0.20-0.30 | 4,000 | 164 | 0.041000 | 0.247 | 0.2060 |
| 0.30-0.45 | 2,600 | 156 | 0.060000 | 0.372 | 0.3120 |
| 0.45-0.60 | 1,600 | 148 | 0.092500 | 0.519 | 0.4265 |
| 0.60-0.75 | 1,000 | 132 | 0.132000 | 0.671 | 0.5390 |
| 0.75-0.90 | 600 | 136 | 0.226667 | 0.812 | 0.5853 |
| 0.90-1.00 | 200 | 80 | 0.400000 | 0.938 | 0.5380 |
| **Total** | **50,000** | **1,500** | **0.030000** | — | — |

The column totals reconcile to the cohort: `200+240+244+164+156+148+132+136+80 = 1500` and `20000+12000+8000+4000+2600+1600+1000+600+200 = 50000`. This reconciliation is the cheapest possible check and it catches bin-boundary errors, which are otherwise invisible because each row looks plausible in isolation.

### 5.5 Step 4 — ECE, MCE, and where the error concentrates

```text
ECE = sum_b (n_b/N)*|acc_b - conf_b|
    = 0.008400 + 0.012960 + 0.018800 + 0.016480 + 0.016224
    + 0.013648 + 0.010780 + 0.007024 + 0.002152
    = 0.106468
MCE = 0.585333  (bin 0.75-0.90)
unweighted mean gap = 0.0210+0.0540+0.1175+0.2060+0.3120+0.4265+0.5390+0.5853+0.5380
                    = 2.7993 / 9 = 0.311033  (2.92x the ECE)
```

Three facts follow. The model is monotonically *under*-confident in the direction that matters: every gap has `conf_b > acc_b`, which is the signature of unrepaid oversampling rather than of a shape defect. The top three bins hold `6.8%` of records and `31.6%` of the ECE, so a report that plots only the top of the curve sees a different model than a report that plots the bottom. And the unweighted mean is nearly three times the ECE, so the two summary statistics disagree by enough that quoting one without naming it is a communication failure.

### 5.6 Step 5 — fit the calibrator and re-measure

Fitting Platt scaling on these nine bin summaries by minimising weighted log loss gives `A = 0.661942` and `B = -2.311554`, both in nats after the logit transform of the mean score.

```text
p_cal(0.519) = sigmoid(0.661942*ln(0.519/0.481) - 2.311554)
             = sigmoid(0.661942*0.076055 - 2.311554)
             = sigmoid(0.050345 - 2.311554) = sigmoid(-2.261209) = 0.094386
observed in that bin = 148/1600 = 0.092500; residual = 0.001886
p_cal(0.938) = sigmoid(0.661942*2.716660 - 2.311554)
             = sigmoid(1.798216 - 2.311554) = sigmoid(-0.513338) = 0.374417
observed = 80/200 = 0.400000; residual = -0.025583
p_cal(0.031) = sigmoid(0.661942*(-3.442019) - 2.311554) = sigmoid(-4.590218) = 0.010049
observed = 200/20000 = 0.010000; residual = 0.000049
ECE_after = 0.001613;  MCE_after = 0.025583;  Brier 0.055957 -> 0.027526
```

The residual pattern matters more than the aggregate. The small bins now carry the error and the large bins are nearly exact, which is the correct trade for a deployment where almost all traffic sits in the low bins. Note also that the slope `A = 0.661942` is *below* one, meaning the raw model was over-dispersed in logit space even before the level correction; a temperature-only fit (`T = 0.685856`, which is `1/1.458030`) captures the dispersion but not the offset and leaves `ECE = 0.068822`.

### 5.7 Step 6 — cost-optimal cut on the corrected scale

```text
tau_cost = c_FP/(c_FP + c_FN) = 4/(4+76) = 4/80 = 0.050000
compare each bin's calibrated value against 0.05:
  bin 0.90-1.00 -> 0.374417  PASSES
  bin 0.75-0.90 -> 0.207002  PASSES
  bin 0.60-0.75 -> 0.137077  PASSES
  bin 0.45-0.60 -> 0.094386  PASSES
  bin 0.30-0.45 -> 0.065487  PASSES
  bin 0.20-0.30 -> 0.045243  FAILS
admitted set: n = 200+600+1000+1600+2600 = 6000;  k = 80+136+132+148+156 = 652
precision = 652/6000 = 0.108667
recall    = 652/1500 = 0.434667
cost = 76*(1500-652) + 4*(6000-652) = 76*848 + 4*5348 = 64448+21392 = 85840 CU
```

The bin at `0.30-0.45` is the boundary case and is worth dwelling on: its calibrated value `0.065487` clears `0.05` while the *raw* mean `0.372` would have cleared any naive cut. Getting this bin wrong changes the actioned count by `2,600` records and the cost by `2,600*(0.065487*76 - 0.934513*4) = 2,600*(4.977 - 3.738) = 2,600*1.239 = 3221 CU`, so the boundary bin alone accounts for more than half the difference between a good and a mediocre policy.

### 5.8 Step 7 — capacity binds, and the truncation is priced

Capacity is 4,000 records, but the cost cut admits 6,000. The deployed policy takes the top 4,000 by score:

```text
top bin  0.90-1.00:  200 records, 80 positives   cumulative n = 200,  k = 80
next     0.75-0.90:  600 records, 136 positives  cumulative n = 800,  k = 216
next     0.60-0.75: 1000 records, 132 positives  cumulative n = 1800, k = 348
next     0.45-0.60: 1600 records, 148 positives  cumulative n = 3400, k = 496
next     0.30-0.45: 2600 records, 156 positives  need 600 of 2600 => 138.46 positives
                    round: 600*(156/2600) = 36.00 positives
                    cumulative n = 4000, k = 532
precision = 532/4000 = 0.133000
recall    = 532/1500 = 0.354667
cost = 76*(1500-532) + 4*(4000-532) = 76*968 + 4*3468 = 73568+13872 = 87440 CU
```

The truncation costs `87440 - 85840 = 1600 CU` per period. That equals `2000 * 0.80`, where `0.80 = 0.065487*76 - 0.934513*4` is the average surplus of a record in the truncated tail — the two routes agree, which is the independent check on the marginal-value arithmetic. Annualised at 250 periods the gap is `400,000 CU`, and that is the number to put in front of a staffing decision rather than a precision figure.

### 5.9 Step 8 — the naive readings, priced for comparison

Three wrong readings are common enough to price explicitly.

```text
(a) un-repaired raw cut at 0.05:
    bins 0.05 and above: n = 30000, k = 1300
    precision 0.043333; cost = 76*200 + 4*29800 = 15200+119200 = 134400 CU

(b) un-repaired raw cut at 0.50:
    bins 0.45 and above: n = 3400, k = 496  (bin 0.30-0.45 mean 0.372 < 0.5 fails)
    precision 0.145882; cost = 76*1004 + 4*2904 = 76304+11616 = 87920 CU

(c) repaired but default cut at 0.50:
    p_cal >= 0.5 never occurs (max is 0.374417) => zero records actioned,
    cost = 76*1500 = 114000 CU, recall 0.000000
```

Reading (a) is the most expensive: `134400 CU`, which is `56.6%` above the correctly repaired policy at `85840 CU`. Reading (b) is deceptively close (`87920` versus `85840`, a `2.4%` gap) and is the one that survives review, because its precision of `0.145882` is actually *higher* than the correct policy's `0.108667` — it actions fewer records, so it wastes less per action while missing more. Reading (c) is the silent failure: no exception, no empty report, just an actioned count of zero and a recall of zero.

### 5.10 Step 9 — uncertainty on the actioned set

```text
Wilson 95%, k = 652 in n = 6000:
  p_hat = 0.108667, z = 1.959964
  centre = (0.108667 + 3.841459/12000)/(1 + 3.841459/6000)
         = (0.108667 + 0.000320)/(1.000640) = 0.108921
  half   = 1.959964*sqrt(0.108667*0.891333/6000 + 3.841459/(4*36e6))
         = 1.959964*sqrt(0.000016144 + 0.000000027) = 1.959964*0.004020 = 0.007880
  interval = [0.101041, 0.116801]  width 0.015760
```

Two cautions. First, 1,500 positives spread over 900 entities would inflate this band by `sqrt(1500/900) = 1.291`, to a width of about `0.020345`, and the sheet's assumption of one record per entity is doing real work. Second, the interval covers sampling noise only: it says nothing about the risk that next month's population has a different prior, and that risk dominates. A one-point prior shift from `0.03` to `0.01` moves the calibrated probability of the `0.05`-cut record from `0.050` to `0.016`, which is a far larger effect than the half-width above.

### 5.11 Step 10 — the decision this sheet produces

The output of the whole exercise is four sentences a risk owner can act on, and they are worth writing in this form.

1. At the declared costs, records above `0.05` calibrated probability are worth reviewing; that is 6,000 records per month.
2. Capacity is 4,000, so 2,000 worth-reviewing records are truncated, costing `1,600 CU` per month and `400,000 CU` per year.
3. The model's ranking is good enough that the top 8% of the population captures 35.5% of positives, which is `11.8` times the base rate.
4. The score may not be read as a probability until the tenfold oversampling debt of `2.302585` nats is repaid; until then every report is on an inflated scale and the default cut of `0.5` is silently a cut of `0.090909`.

### 5.12 Independent check block

Every material number above was recomputed by a second route. The second route is recorded here so a reader can reproduce the check rather than trust it.

```text
CHECK 1  prior:         1500/50000 = 0.03;  sum_b k_b = 1500;  sum_b n_b = 50000.
CHECK 2  debt:          0.309278/0.030928 = 10.000000;  ln(10) = 2.302585.
CHECK 3  repair at 0.5: s/(s+10(1-s)) = 0.5/5.5 = 0.090909;  sigmoid(-2.302585) = 0.090909.
CHECK 4  cost optimum:  additive form 76*848 + 4*5348 = 64448+21392 = 85840.
                        algebraic form c_FP*n + c_FN*K - TP*(c_FN+c_FP):
                        4*6000 + 76*1500 - 652*80 = 24000+114000-52160 = 85840.
                        two structurally different routes, same total. VERIFIED.
CHECK 5  truncation:    additive form for the truncated policy:
                        76*(1500-532) + 4*(4000-532) = 73568+13872 = 87440;
                        algebraic form: 4*4000 + 76*1500 - 532*80 = 16000+114000-42560
                        = 87440. Difference 87440-85840 = 1600. VERIFIED.
CHECK 6  ECE:           0.008400+0.012960 = 0.021360; +0.018800 = 0.040160;
                        +0.016480 = 0.056640; +0.016224 = 0.072864;
                        +0.013648 = 0.086512; +0.010780 = 0.097292;
                        +0.007024 = 0.104316; +0.002152 = 0.106468. VERIFIED.
CHECK 7  Brier identity: 0.028442-0.001585+0.029100 = 0.055957 = direct Brier. VERIFIED.
CHECK 8  naive (a):     76*200 = 15200; 4*29800 = 119200; total 134400.
                        134400/85840 = 1.5657, i.e. 56.6% higher. VERIFIED.
CHECK 9  lift:          0.354667/0.030000 = 11.822; precision lift over base rate
                        0.133000/0.030000 = 4.433. Two different lift definitions;
                        both are reported because readers assume different ones.
CHECK 10 class weight:  w = (1-0.05)/0.05 = 19;  0.5/(0.5+19*0.5) = 0.05. VERIFIED.
```

Check 4 and Check 5 use two algebraic routes that must agree, and the agreement is the test: the additive form charges every negative error at `c_FP` and every positive error at `c_FN`, while the algebraic form starts from the all-negative baseline and credits each true positive with the `c_FN + c_FP` it avoids paying. A single route can be wrong in a way that looks self-consistent; two routes that share no terms cannot.

### 5.13 Sensitivity: what would change the decision

| Parameter | Base | Change | Effect on actioned count and cost |
|---|---:|---|---|
| Miss cost `c_FN` | 76 CU | 40 CU | `tau = 4/44 = 0.090909`; actioned 3,400; cost `51,776 CU` — a `39.7%` reduction |
| Miss cost `c_FN` | 76 CU | 200 CU | `tau = 4/204 = 0.019608`; actioned 18,000; cost `155,760 CU` on the 200-CU basis |
| False-action cost `c_FP` | 4 CU | 1 CU | `tau = 1/77 = 0.012987`; actioned 30,000; cost `43,900 CU` on the 1-CU basis |
| Capacity | 4,000 | 6,000 | truncation gap vanishes; cost equals the cost-optimal `85,840 CU` exactly |
| Prior | 0.03 | 0.01 | every calibrated probability falls by `1.119021` nats; the 0.05 cut admits only 800 records |
| Calibrator | Platt | none (raw) | cost rises to `134,400 CU`; recall at the naive 0.05 cut falls to `1300/1500 = 0.866667` |

The capacity row is the most useful: raising capacity from 4,000 to 6,000 recovers exactly the `1,600 CU` gap and no more, because beyond 6,000 the next record's expected surplus turns negative. A manager who wants to know whether to hire reviewers now has the shape of the answer: the first 2,000 additional slots are worth `1,600 CU` per month in total, and the next 1,000 are worth nothing.

The prior row deserves a warning. Moving from 0.03 to 0.01 does not change the model's ranking — lift stays at `11.822` — but it removes five of the six admitted bins, because every probability falls by `1.119021` nats and the `0.065487` bin becomes `0.022375`. A team that monitors only ranking metrics would see an unchanged AUC and a collapsed queue and would spend a week looking for a pipeline bug that does not exist.

**Takeaway:** The worked example's whole value is in the three numbers that a naive report would have produced instead — `134,400`, `87,920`, and `0` — and in the fact that only one of the three looks wrong.

## 6. Visual / Geometric Intuition

### 6.1 The reliability diagram as a distance from the diagonal

Plot predicted probability on the horizontal axis and observed frequency on the vertical. The diagonal is the calibration claim. Each bin is a point at `(conf_b, acc_b)` with radius proportional to `sqrt(n_b)`.

In the Section 5 cohort every point sits below and to the right of the diagonal, and the vertical distance grows with the horizontal coordinate: `0.0210` in the first bin, `0.5853` in the eighth. That fan shape is a specific signature. A pure additive offset in logit space produces a *converging* pattern near the extremes and a maximum gap in the middle; a pure multiplicative scale error in logit space produces exactly the monotone widening seen here, because a slope `A = 0.661942` stretches the gap proportionally to `z`. Reading the shape tells you which parameter is missing before you fit anything, and that is worth more than the aggregate number.

The same nine points under the Platt map lie almost on the diagonal, with the residual `0.025583` concentrated in the 200-record top bin. The geometry has changed from a systematic fan to a small-bin scatter, which is the correct outcome: what remains is sampling noise, not a defect.

### 6.2 Logit space and why prior correction is a translation

In logit space the reliability diagram is a straight-line question. If `z = ln(p/(1-p))` and the observed log-odds of a bin are `ln(acc_b/(1-acc_b))`, calibration means the two agree line for line. A slope below one means the model's logits are stretched; an intercept below zero means they are shifted down.

The Section 5 bins give observed log-odds from `ln(0.01/0.99) = -4.595120` at the bottom to `ln(0.4/0.6) = -0.405465` at the top, against predicted logits from `ln(0.031/0.969) = -3.442277` to `ln(0.938/0.062) = 2.716616`. The span of predictions is `6.158893` nats; the span of observations is `4.189655` nats. Their ratio `0.680261` is close to the fitted slope `0.661942`, which is the geometric statement that the model is over-dispersed: it spreads scores wider than the truth spreads.

Prior correction is a pure translation in this space and therefore cannot change the slope at all. A shift of `-2.302585` nats moves every point left by the same amount, which is exactly why resampling is invertible and why a slope defect survives it untouched. This single geometric fact explains the most common confusion in the area: two people argue about "fixing the imbalance", one is translating, the other is rescaling, and neither is listening.

### 6.3 ROC area versus precision-recall area, under prevalence change

Both curves are generated by sweeping a threshold over the same ordering, so their shapes encode the same information conditioned differently.

ROC plots `FPR = FP/(N-K)` against `TPR = TP/K`. Both denominators are class-conditional, so changing `K` and `N-K` proportionally leaves the curve approximately unchanged. On the Section 5 bins the trapezoidal area is `0.742562`.

Precision-recall plots `precision = TP/(TP+FP)` against `recall = TP/K`. The precision denominator mixes classes, so a change in prevalence moves the whole curve. The Section 5 trapezoidal average precision is `0.115626` against a base rate of `0.03`, so the curve sits `3.85` times above the no-skill line at its left edge and converges to `0.03` at recall 1.

```text
recall    precision
0.0533    0.4000
0.1440    0.2700
0.2320    0.1933
0.3307    0.1459
0.4347    0.1087   <- the cost-optimal point
0.5440    0.0816
0.7067    0.0589
0.8667    0.0433
1.0000    0.0300   <- precision equals prevalence at full recall
```

The last row is an identity, not a coincidence: actioning every record makes precision equal the base rate. Any PR curve whose rightmost point is not the prevalence has been computed on a filtered population, and that is a fast check for a cohort error.

### 6.4 The threshold sweep as a cost landscape

Plot total cost against the threshold and the landscape is a shallow bowl whose minimum is at `0.05` and whose walls are asymmetric. On the cost-optimal scale:

```text
tau      actioned   cost CU
0.0100   50000      194000
0.0200   18000      101200
0.0300   18000      101200
0.0500   6000        85840   <- minimum
0.1000   1800        93360
0.2000    800        99920
```

The flat region between `0.02` and `0.03` is an artefact of the bin resolution: no bin boundary falls in that interval, so the actioned set does not change. With record-level scores the bowl would be smooth. Learning to read that flatness as resolution rather than as robustness is the practical skill here, because a team that trusts a flat region will be surprised when a finer threshold grid produces a different optimum.

The asymmetry is real, not an artefact. Moving the cut from `0.05` to `0.02` adds `101200 - 85840 = 15360 CU`; moving it from `0.05` to `0.10` adds `93360 - 85840 = 7520 CU`. Under-actioning is roughly twice as expensive as over-actioning at this cost ratio (ratio `2.04`), because `c_FN/c_FP = 19` and the loss grows with the ratio on the under-actioning side while it is capped by the finite number of records on the over-actioning side.

### 6.5 Gain and lift as the geometry of a queue

Order records by score and plot the cumulative share of positives captured against the cumulative share of records reviewed. The Section 5 curve rises steeply then flattens.

At `8%` of the population (4,000 records) it captures `35.5%` of positives. At `12%` (6,000) it captures `43.5%`. At `36%` (18,000) it captures `70.7%`. The slope of the curve at any point is the local precision, so the point where the slope drops below `c_FP/(c_FN+c_FP) = 0.05` is exactly the cost-optimal cut — the gain chart and the cost model are the same object viewed from different axes.

That equivalence is worth using as a cross-check. If the cost-optimal threshold computed from `IMB-02` does not coincide with the point where the gain-curve slope crosses the cost ratio, then one of the two has a population mismatch, and finding which is faster than auditing either alone.

### 6.6 The queue as a truncated area

Draw the actioned set as the area under the score-density curve to the right of the cut. Capacity truncates that area from the left, removing the lowest-scoring admitted records — which are exactly the records with the smallest surplus per record.

In Section 5 the cost cut at `0.05` admits `6,000` records; capacity `4,000` truncates the bottom `2,000` of them, whose mean surplus is `0.80 CU`. Because surplus rises with score, the truncated tail's average is well below the admitted set's average of `(0.108667*76 - 0.891333*4) = 8.258692 - 3.565332 = 4.693360 CU`, and the total lost value `1,600 CU` is much smaller than `2,000 * 4.693360 = 9,386.72 CU` would suggest. Reading the queue as an area with a *gradient* rather than as a uniform block is what makes that distinction visible.

### 6.7 The resampling picture: two distributions and a moved boundary

Before resampling, the score distributions of positives and negatives overlap heavily because negatives outnumber positives 32 to 1; the fitted boundary sits where the two weighted densities cross. Oversampling positives by ten moves that crossing left, because the positive density is multiplied by ten everywhere. Undersampling negatives by twelve moves it right, because the negative density is divided.

The geometry of the confusion is that the *shapes* never change; only the weights do. Hence the decision boundary moves but the class-conditional densities stay put, which is the visual form of the statement that resampling is a threshold operation in disguise. When someone reports that resampling "improved the model", the correct question is which curve moved — and the answer, under pure resampling, is always the boundary, never the curve.

### 6.8 Where the eye fails: the 0.4% bin

The top bin holds 200 of 50,000 records, `0.4%` of the population, and it carries `0.538` of vertical gap. On a scatter plot with radius proportional to `sqrt(n)`, its marker is invisible next to the 20,000-record first bin, yet it is the bin that decides whether high scores are trustworthy.

The standard fix is to plot two panels: one weighted by population, one unweighted, with bin counts annotated. The two panels answer different questions — "how wrong is the model on average" and "how wrong is it anywhere" — and a single panel cannot answer both. Section 8's third dossier is precisely a case where the single panel hid the failure.

**Takeaway:** Slope in logit space is a scaling defect, intercept is a level defect, prior repair is a translation, and the only panel that reveals a small-bin disaster is the unweighted one.

## 7. Comparison / Selection Table

### 7.1 Choosing the imbalance remedy from the mechanism

| Observed mechanism | Remedy that addresses it | Remedy that does not | Why the distinction matters |
|---|---|---|---|
| Gradient dominated by easy negatives | Class weight, focal loss, or random undersampling of easy negatives | Oversampling positives | The negatives are the problem; adding positives multiplies the same gradient imbalance |
| Too few positives to resolve behaviour | Longer window, wider population, cheaper label | Any resampling | Resampling cannot create signal that is not in the sample |
| Model over-confident after resampling | Repay the logit debt (`IMB-05`, `IMB-06`) | Retrain at the deployment prior | The defect is arithmetic; retraining adds variance for no gain |
| Threshold inherited from a default | Derive from `IMB-02` | Resample to make 0.5 correct | Resampling to fix a threshold is a costly way to move an intercept |
| Prior shifted between fit and deployment | Add `ln[odds(pi_dep)/odds(pi_fit)]` | Recalibrate on old data | The old data carries the old prior |
| Calibration curve has wrong slope | Platt or temperature | Isotonic on small bins | Isotonic overfits the noise in sparse bins |

### 7.2 Choosing a calibrator

| Calibrator | Parameters | Minimum positives to fit | Fixes level bias | Fixes slope | Preserves ranking | Failure mode |
|---|---:|---:|---|---|---|---|
| None | 0 | 0 | no | no | yes | Reports inflated probabilities |
| Prior correction only | 0 (analytic) | prior measurement | yes | no | yes | Wrong if `P(X\|Y)` moved |
| Temperature `T` | 1 | ~100 | no | yes | yes | Cannot fix an offset |
| Platt `A, B` | 2 | ~200 | yes | yes | yes | Fails if the curve is not logistic |
| Isotonic | `O(bins)` | ~50 per bin | yes | yes (piecewise) | yes, with ties created | Overfits, quantises probabilities |
| Beta calibration | 3 | ~300 | yes | yes, asymmetric | yes | Needs a richer slice |

The Section 5 evidence is directly relevant to the third and fourth rows: `T = 0.685856` alone leaves `ECE = 0.068822` while Platt's two parameters reach `0.001613`. On these bins isotonic reaches `0.000000` on the fitting data — a result that is a warning, not an achievement.

### 7.3 Choosing the metric to report

| Question being asked | Correct metric | Wrong but common substitute | What the substitute hides |
|---|---|---|---|
| Does the model order records usefully? | ROC area, or lift at the operating depth | Accuracy | The trivial all-negative policy wins on accuracy |
| How much of the queue is wasted? | Precision at the deployed cut | Average precision | AP integrates over cuts nobody runs |
| What fraction of positives do we reach? | Recall at the deployed cut | ROC area | Area counts performance at unusable cuts |
| Is the probability usable in a cost model? | ECE, MCE, and a reliability table | ROC area | Rank metrics are blind to level |
| Is the forecast better than the base rate? | Brier skill score against `pi(1-pi)` | Brier score alone | A raw Brier of 0.056 looks good until compared with 0.029 |
| Will the queue be serviceable? | Actioned count versus capacity | Precision | A high-precision queue of 200 records serves nobody |
| Did the model degrade? | Slice-level rates with intervals | One global number | A single slice can collapse inside a stable mean |

### 7.4 Choosing the source of the threshold

| Source | Formula or rule | When it is correct | When it is wrong |
|---|---|---|---|
| Cost ratio | `c_FP/(c_FP+c_FN)` | Costs are measurable and roughly constant | Costs vary with exposure per record |
| Capacity | score of the `C`-th record | Service limit is hard and costs are vague | Capacity is elastic and the marginal record has negative surplus |
| Precision target | smallest cut reaching target precision | A contractual or audit standard exists | The target was set without reference to what the model can do |
| Recall target | largest cut reaching target recall | Missing a positive is unacceptable at any cost | The resulting queue exceeds capacity by an order of magnitude |
| Historic volume | last period's actioned count | The process, not the model, is the constraint | The model improved and the volume should have grown |
| Library default 0.5 | none | The score is a calibrated probability and costs are equal | Almost always, in an imbalanced problem |

### 7.5 Choosing between repairing the sample and repairing the threshold

| Situation | Repair the sample | Repair the threshold | Reason |
|---|---|---|---|
| Positives are 2% of 50,000 | No | Yes | Enough positives to estimate; the cut is the only defect |
| Positives are 40 in 200,000 | No | No, first collect more | Neither repair creates evidence |
| Training took 9 hours unweighted | No | Yes | Threshold repair costs nothing at inference |
| Deployment prior differs from fit prior | No | Yes, analytically | Prior shift is a translation |
| The loss is a likelihood and you need standard errors | No | Careful | Resampling distorts the information matrix |
| A downstream team consumes probabilities | No, repair the logit | Yes, and repay the debt | Probabilities must be on the deployment scale |

**Takeaway:** Resample to change what the model learns, shift to change what the score means, and cut to change what the business does — three different levers, and using one to fix another is the whole failure catalogue.

## 8. Failure Modes & Edge Cases

### 8.1 Failure index

| ID | Failure | Observable symptom | First discriminating check |
|---|---|---|---|
| `IMB-F01` | Oversampling debt unrepaid | Every probability inflated, `conf > acc` in all bins | Compare `conf/acc` per bin; a constant ratio near `m` is the tell |
| `IMB-F02` | Undersampling variance unaccounted | Rate estimates move sharply between seeds | Refit on five seeds; spread of precision far above the Wilson band |
| `IMB-F03` | Prior mismatch between fit and deploy | Queue fills or collapses after a population change | Compare deployment prior with fit prior; `IMB-05` predicts the shift |
| `IMB-F04` | Calibration fitted on training data | ECE good in-sample, poor out-of-sample | Refit calibrator on the held-out slice and re-measure |
| `IMB-F05` | Threshold from a library default | Actioned count far from capacity | Recompute `IMB-02`; compare with the actioned count |
| `IMB-F06` | Cost double-counted as weight plus cut | Recall lower than either remedy alone predicts | Check whether `w` and `tau` both encode the ratio |
| `IMB-F07` | SMOTE treated as invertible | Corrected probabilities still biased after `ln m` repair | Check whether synthetic points lie off the class manifold |
| `IMB-F08` | ECE reported without the binning | ECE changes when bins change | Recompute at 5, 10, 20 bins; report the sensitivity |
| `IMB-F09` | Entity clustering ignored | Intervals too narrow, slices look significant | Recompute with entity-level bootstrap |
| `IMB-F10` | Immature labels counted as negatives | Prior too low, top-bin calibration biased | Exclude unmatured rows and recompute the prior |

### 8.2 Dossier `IMB-D01` — the tenfold oversampling that survived three reviews

**Symptom.** A queue reported precision `0.43` in the top bin while the observed default rate among those accounts was `0.09`. Nothing failed: no exception, no empty output, no drift alert. The AUC was `0.74` and stable.

**Competing causes.** (a) The positives were oversampled tenfold and the debt never repaid. (b) The deployment population genuinely had a lower default rate than the cohort. (c) The model was over-fitted and the top bin was memorised.

**Discriminating check.** Compute `conf_b/acc_b` per bin. Under (a) the ratio is approximately constant across bins and equals `m`; under (b) it grows with the score coordinate because a prior shift is additive in logits and therefore multiplicative in odds, not in probability; under (c) the gap is concentrated in the highest bins and vanishes in the middle. The observed ratios here were `3.1, 3.7, 4.9, 6.0, 6.2, 5.6, 5.1, 3.6, 2.3` — not constant, which ruled out the pure-debt story and pointed at a combination: a debt of ten plus a slope defect. Fitting Platt after repaying `ln 10` gave `A = 0.661942`, confirming the slope defect existed independently.

**Correction.** Subtract `2.302585` nats from every logit, then fit Platt on a held-out slice. Cost fell from `134,400 CU` to `85,840 CU` per period.

**Prevention.** Store the sampling factor as a field on the model artefact and require any consumer of probabilities to declare it. A model whose artefact lacks that field cannot be promoted.

**Verification.** Recompute the bin table from raw scores and confirm every `conf_b <= acc_b*m` within the Wilson band of `acc_b`.

### 8.3 Dossier `IMB-D02` — undersampling that bought variance it could not afford

**Symptom.** Precision at the deployed cut moved between `0.081` and `0.171` across five random seeds, a spread wider than any plausible population change.

**Competing causes.** (a) The negatives were undersampled 12.25 to 1, inflating the variance of every fitted quantity. (b) The positives were too few for the model complexity. (c) The evaluation cohort was resampled rather than only the training set, so the estimate itself was noisy.

**Discriminating check.** Refit on five seeds with the same undersampling and compute the standard deviation of precision at a fixed cut. Then refit with all negatives and recompute. The undersampled spread here was `0.031` in absolute precision; the full-sample spread was `0.009`. The ratio `3.44` is close to `sqrt(12.25) = 3.500`, which identifies (a) precisely: the variance inflation factor of `u` predicts a standard-error ratio of `sqrt(u)`. Cause (c) was ruled out by rebuilding the evaluation cohort from the full held-out set, which did not change the spread.

**Correction.** Drop undersampling; use a class weight of `19` instead, which moves the same boundary with no records discarded. Precision at the cut settled at `0.109` with a spread of `0.009`.

**Prevention.** Never resample the evaluation cohort. If undersampling is used for training cost reasons, report the seed spread of every rate at the deployed cut.

**Verification.** Confirm the standard-error ratio against `sqrt(u)` within a factor of 1.2; a larger ratio means something other than sampling is unstable.

### 8.4 Dossier `IMB-D03` — the aggregate that hid a segment collapse

**Symptom.** Overall ECE `0.019`, well inside the gate. A month later, a segment holding `7%` of records showed an actioned-set precision of `0.02` against a claimed `0.31`.

**Competing causes.** (a) The calibrator was fitted globally and did not transfer to a segment whose prevalence differed. (b) The segment's features had shifted. (c) The segment's labels matured more slowly, so its recent positives were counted as negatives.

**Discriminating check.** Recompute ECE per segment. Under (a) the segment's gap is a level shift and disappears after applying `IMB-05` with the segment prior; under (b) the segment's ranking degrades and the segment AUC falls; under (c) the gap grows monotonically as the label window shortens. The segment prior was `0.006` against a global `0.03`, a shift of `ln[(0.006/0.994)/(0.03/0.97)] = ln(0.006036/0.030928) = -1.633879` nats, and applying it closed `0.29` of the `0.31` gap — identifying (a) with `93.5%` of the defect explained. Segment AUC was unchanged at `0.71`, ruling out (b).

**Correction.** Per-segment prior correction, with a floor: segments below 400 records inherit the global prior plus a conservative offset.

**Prevention.** Every calibration gate is computed per pre-declared segment as well as globally, and the global figure is never the only figure reported.

**Verification.** Confirm the closed gap is at least `0.8` of the observed gap and that the residual is inside the segment's Wilson band.

### 8.5 Dossier `IMB-D04` — the weight and the cut that cancelled

**Symptom.** Recall `0.11` at what the team believed was the cost-optimal operating point, against a predicted `0.43`.

**Competing causes.** (a) The class weight of `19` and a cut of `0.05` were both applied, cancelling to an effective cut of `0.00263`. (b) The model's probabilities were on the inflated scale, so the cut was applied in the wrong units. (c) The cost ratio was mis-stated as `4:76` when it was `76:4`.

**Discriminating check.** Compose the two effects: with `w = 19` and cut `t = 0.05`, the effective unweighted cut is `t/(t + w(1-t)) = 0.05/(0.05 + 19*0.95) = 0.05/18.10 = 0.002762`. That is *below* the intended `0.05`, so the policy should over-action and over-recall, not under-recall — which rules out (a) as the cause of an under-recall. Checking (c), the stated ratio in the model card was `c_FP = 76, c_FN = 4`, giving `tau = 0.95`, and actioning almost nothing. That explained recall `0.11`.

**Correction.** Fix the cost direction in the card and re-derive: `tau = 4/(4+76) = 0.05`. Remove the class weight, since the cut already encodes the ratio.

**Prevention.** Store costs with a stated direction ("cost of a wasted action" versus "cost of a miss") rather than as a bare pair. A unit test asserts that the derived cut lies in `(0, 0.5)` when misses cost more than false actions.

**Verification.** Assert `tau = c_FP/(c_FP + c_FN)` and that the actioned count is within a factor of two of capacity; a violation in either direction is an error, not a tuning result.

### 8.6 Dossier `IMB-D05` — isotonic on bins that held three positives

**Symptom.** Calibration ECE of `0.000` in the calibration report, and a deployed over-confidence of `0.14` in the highest score band within one month.

**Competing causes.** (a) Isotonic regression overfitted a slice with too few positives per bin. (b) The deployment prior moved. (c) The calibrator was fitted on the fitting data, not a held-out slice.

**Discriminating check.** Hold out a further slice and measure ECE there. Under (a) the held-out ECE explodes while the in-sample stays at zero; that is the signature of a flexible monotone fit. Under (b) the held-out gap is a pure level shift; under (c) the in-sample figure would already be near zero and the held-out figure would grow with the number of bins. Here the held-out ECE was `0.131` against an in-sample `0.000`, and restricting isotonic to five bins with at least 50 positives each brought held-out ECE to `0.026`. Prior was stable, ruling out (b).

**Correction.** Replace isotonic with Platt where bin positives fall below 200; keep isotonic only for the low-score region where records are abundant.

**Prevention.** A calibration report prints in-sample and held-out ECE side by side; a gap above `0.05` blocks promotion.

**Verification.** Confirm the held-out ECE is within `0.05` of the in-sample value, or that the difference is explained by the documented bin counts.

### 8.7 Edge case: the degenerate cohort

If a cohort has zero positives, precision is undefined, recall is zero, and every formula in Section 4 divides by an empty set. The correct report is "not estimable, `K = 0`", never a zero. If a cohort has zero actioned records, precision is undefined rather than one. Both are cheap to handle and both are frequently handled wrongly by a library that returns `0.0`.

### 8.8 Edge case: ties at the capacity boundary

When `s_(C) = s_(C+1)`, the capacity cut is ambiguous. Breaking ties by record identifier introduces an arbitrary but reproducible policy; breaking them randomly introduces noise across reruns. The defensible choice is to action all tied records and report the capacity overshoot, because a partial action on a tie is unauditable.

### 8.9 Edge case: several records per entity

With 1,500 positives across 900 entities, both the rate intervals and the calibrator's effective sample size are over-stated. The correction is to compute intervals at entity level and to fit the calibrator on entity-level aggregates, or to accept a band inflated by `sqrt(records/entities) = 1.291`.

### 8.10 Edge case: costs that vary per record

When `c_FP` scales with exposure, the single-threshold rule under-actions large records. The fix is to compute per-record surplus `p_i*c_FN,i - (1-p_i)*c_FP,i` and rank by surplus rather than by probability. That change alone typically moves the composition of the queue more than any model improvement.

**Takeaway:** Every dossier above has a check that costs less than an hour and that would have found the defect before deployment; the discipline is running the check when the aggregate looks fine.

## 9. Complexity & Cost Analysis

### 9.1 Compute cost of the remedies

| Remedy | Training cost relative to baseline | Inference cost | Memory | Main hidden cost |
|---|---|---|---|---|
| None | 1.0 | 1.0 | 1.0 | Every probability wrong; no compute saved |
| Class weight | ~1.0 (loss multiply) | 1.0 | 1.0 | None material |
| Oversample `m` | `~1 + (m-1)*pi` | 1.0 | `~1 + (m-1)*pi` | Repeated rows inflate fit time and duplicate memory |
| Undersample `u` | `~1/u` | 1.0 | `~1/u` | Variance inflation of `u` |
| SMOTE | `~1 + pi*(k+1)` neighbour searches | 1.0 | `~1 + generated rows` | Neighbour search dominates at low `pi` |
| Focal loss | ~1.05 | 1.0 | 1.0 | Tuning `gamma` costs fits |
| Platt calibration | ~0.01 of a training run | `1` sigmoid per record | 2 parameters | Needs a held-out slice |
| Isotonic | ~0.005 | 1 binary search per record | `O(bins)` | Overfit risk, not compute |

For the Section 5 cohort the arithmetic is concrete. Oversampling positives tenfold adds `1500*9 = 13,500` rows to `50,000`, a `27%` increase in rows and therefore roughly a `27%` increase in fit time: `20.3 MB` of feature matrix against `16.0 MB` at 40 float64 features per record. Undersampling to `1/u = 1/12.25` retention keeps `48500/12.25 = 3,959` negatives plus `1,500` positives, a total of `5,459` rows — an `89%` reduction in rows and a matching reduction in fit time, bought with a variance inflation of `12.25`.

### 9.2 Compute cost of the calibration pipeline

A Platt fit on nine bin summaries is a two-parameter optimisation over `50000` records; at roughly `5` passes of gradient descent over the logits it is `250,000` sigmoid evaluations, which is milliseconds. The real cost is the *slice*, not the fit: reserving `20%` of the cohort for calibration gives `10,000` records and `300` positives, which is adequate for two parameters but marginal for isotonic with more than six bins.

Isotonic with `B` bins costs `O(B log B)` for the sort plus `O(B)` for the pooling, so `B = 9` is trivial. The cost is statistical: each bin's rate has a standard error of `sqrt(acc(1-acc)/k_b)`, which for the top bin with `k_b = 80` is `sqrt(0.4*0.6/80) = 0.054772`. A calibrator that must resolve a `0.05` difference cannot be built from bins with `0.055` standard errors, and that is a budget statement rather than an algorithmic one.

### 9.3 Cost of a threshold sweep

A sweep over `100` candidate cuts costs `100 * 50,000 = 5,000,000` comparisons, which is a fraction of a second in a vectorised implementation. It is cheap enough that there is no excuse for using a default threshold. What is *not* cheap is the population discipline: each candidate cut must be evaluated on the same population, the same label maturity, and the same deduplication, and a sweep that silently re-derives the population per cut produces a spurious optimum.

### 9.4 Cost of the uncertainty estimate

A Wilson interval is closed-form, so it is free. An entity-level bootstrap with `1,000` resamples over `50,000` records costs `1,000 * 50,000 = 5 * 10^7` record draws plus a refit-free recomputation of precision, which is seconds. That is worth paying whenever `records/entities > 1.1`, because the closed-form interval is then wrong by a known factor.

### 9.5 Operational cost of the three mistakes

| Mistake | Per-period cost | Basis |
|---|---:|---|
| Unrepaid oversampling with a raw 0.05 cut | `48,560 CU` | `134,400 - 85,840` |
| Unrepaid oversampling with the default 0.5 cut | `0 CU` visible, `114,000 CU` of missed value | recall zero, no queue |
| Capacity left unexpanded at 4,000 | `1,600 CU` per period | truncation of 2,000 records |
| Undersampling variance | unquantified in CU; `0.031` precision spread | seed-to-seed range |
| Missing calibrator entirely | `48,560 CU` | same as the first row |

The middle row is the one that matters for governance: a policy that actions nothing has no false positives, so every precision-style report reads well or is empty, and the loss appears only as unbudgeted defaults.

### 9.6 Sizing the calibration slice properly

The number of positives needed to estimate a rate to a target half-width is `n = z^2 * p(1-p) / h^2`. For `p = 0.1`, `z = 1.959964`:

```text
h = 0.05  -> n = 3.841459*0.09/0.0025 = 138.29  -> 139 positives
h = 0.02  -> n = 3.841459*0.09/0.0004 = 864.33  -> 865 positives
h = 0.01  -> n = 3.841459*0.09/0.0001 = 3457.31 -> 3458 positives
```

For the rare end, `p = 0.02`:

```text
h = 0.05  -> n = 3.841459*0.0196/0.0025 = 30.12  -> 31 positives
h = 0.01  -> n = 3.841459*0.0196/0.0001 = 753.05 -> 754 positives
```

The first table says a `0.02` half-width on a `0.1` rate needs `865` positives in the slice, which at `3%` prevalence is `865/0.03 = 28,833` records — `57.7%` of the cohort. The second says a `0.01` half-width on a `0.02` rate needs `754` positives, or `37,700` records, which exceeds the cohort entirely and therefore cannot be bought at this population. That is the quantitative form of the statement that fine calibration at the rare end is expensive, and it explains why the top bin in Section 5, with `80` positives, cannot support a claim finer than about `0.05`.

### 9.6a The variance price of undersampling, stated exactly

Undersampling negatives by `u` multiplies the variance of any estimate that depends on the negative distribution by approximately `u`, so standard errors rise by `sqrt(u)`. At `u = 12.25` that factor is `3.500`. The Dossier `IMB-D02` spread ratio of `3.44` against a predicted `3.500` is a `1.7%` match, which is as close as a five-seed estimate can be expected to land.

The row count falls from `50,000` to `1,500 + 48,500/12.25 = 1,500 + 3,959 = 5,459`, an `89.1%` reduction, and the feature matrix falls from `16.0 MB` to `5,459*40*8/10^6 = 1.75 MB`. So the trade is a `9.1x` reduction in fit rows for a `12.25x` inflation of variance, and that exchange is only worth making when fit time is the binding constraint — which, for a tabular cohort of this size, it never is.

### 9.7 Reproducibility cost

A calibration result is reproducible only if the binning, the slice boundaries, the sampling factor, and the costs are recorded with it. Recording four extra fields costs nothing at write time and saves an entire investigation. The minimum manifest is: cohort identifier and row count, positive count, prior, sampling scheme and factor, bin edges, calibrator type and parameters, deployment costs with direction, capacity and interval, and the software versions of the fitting library.

**Takeaway:** The compute is trivial and the evidence is expensive; spend on the slice and the manifest, not on the resampling.

## 10. Key Theorems / Results Reference

### 10.1 Results table

| ID | Result | What it buys | Where it fails |
|---|---|---|---|
| `IMB-T01` | Bayes rule under asymmetric costs | The cut, from first principles | Costs not constant per record |
| `IMB-T02` | Prior-shift invariance of the ordering | Justifies cut-shift instead of retraining | Covariate shift also present |
| `IMB-T03` | Prior-shift additivity in logit space | One subtraction corrects a sample | Likelihood ratio changed by resampling type |
| `IMB-T04` | Class-weight and threshold equivalence | One decision, not two | Non-likelihood losses |
| `IMB-T05` | Propriety of log loss and Brier | Justifies fitting calibration by those losses | Accuracy-optimised training |
| `IMB-T06` | Murphy decomposition | Separates reliability from resolution | Bins must use bin-mean predictions |
| `IMB-T07` | Calibration-refinement decomposition | Shows a constant predictor is calibrated | Discrete forecasts only |
| `IMB-T08` | Ranking interpretation of ROC area | Explains invariance to monotone maps | Prevalence-dependent readings |
| `IMB-T09` | Prevalence dependence of precision at fixed threshold | Predicts queue yield after a prior change | Threshold re-derived per prevalence |
| `IMB-T10` | Consistency of isotonic regression | Justifies the nonparametric calibrator | Finite-sample overfit |
| `IMB-T11` | Wilson interval coverage | Closed-form band for a rate | Clustered records |
| `IMB-T12` | Variance inflation under subsampling | Predicts seed spread | Stratified or hard-negative sampling |

### 10.2 `IMB-T01` — the cost-weighted Bayes rule

For a record with calibrated `p = P(Y=1|x)`, the expected loss of acting is `(1-p)c_FP` and of abstaining is `p*c_FN`. Acting is optimal iff `(1-p)c_FP < p*c_FN`, i.e. `p > c_FP/(c_FP+c_FN)`. The result is exact and needs no distributional assumption beyond calibration, because it compares two conditional expectations over the same binary outcome.

It fails when `c_FP` depends on the record. Then the rule is per-record and no global threshold is optimal; the failure is not in the theorem but in its aggregation. A second failure is when `p` is not calibrated: the inequality then compares a monotone transform of a probability with a ratio of costs, and the ordering of records by surplus is destroyed.

### 10.3 `IMB-T02` and `IMB-T03` — prior shift

Write the posterior odds as `p/(1-p) = [P(x|Y=1)/P(x|Y=0)] * [pi/(1-pi)]`. If only `pi` changes, the likelihood ratio is a record-specific constant independent of the prior, so the posterior odds change by the same multiplicative factor for every record. Two consequences follow immediately: the *ordering* is preserved (T02), and the *logit* shifts by the constant `ln[odds(pi_new)/odds(pi_old)]` (T03).

The practical value of T03 is that a prior change never requires refitting. The practical limit is that the factor must be applied to odds; applying it to probabilities gives an error that grows with `p`. At `p = 0.05` and a factor of `0.072166`, the correct result is `0.003784` while the naive multiplication gives `0.003608` — a `4.9%` relative error, small here and much larger near the top of the range.

### 10.4 `IMB-T04` — weights are thresholds

Weighting positives by `w` multiplies their likelihood contribution by `w`, so the fitted parameter vector satisfies the same stationarity condition as an unweighted fit on data where each positive appears `w` times. The posterior therefore has log-odds shifted by `ln w`. Consequently a fixed cut `t` under weight `w` acts exactly as the cut `t/(t+w(1-t))` under no weight.

The result fails for losses without a likelihood interpretation. A support vector machine with hinge loss and a class weight does not produce a posterior, so no cut is equivalent; the weight changes the margin and therefore the *ranking*, which a threshold cannot do. Any team that treats SVM class weights and thresholds as interchangeable is relying on a theorem that does not apply to their model.

### 10.5 `IMB-T05` — propriety

Log loss and Brier score are strictly proper: their expected value is minimised uniquely at the true conditional probability. This is why fitting a calibrator by minimising either loss produces a map whose output is calibrated in the limit, and why accuracy is unsuitable — accuracy is minimised by a deterministic `0` or `1` and therefore rewards an over-confident model.

The practical corollary is that a calibration pipeline fitted by accuracy or by AUC has no theoretical claim to produce probabilities. Teams do this routinely when they reuse a training harness, and the result is a "calibrator" that only rescales.

### 10.6 `IMB-T06` — the Murphy decomposition

With records grouped into bins and `conf_b` the bin-mean prediction, `Brier = REL - RES + UNC` exactly, where `REL = (1/N)sum n_b(acc_b - conf_b)^2`, `RES = (1/N)sum n_b(acc_b - pi)^2`, `UNC = pi(1-pi)`. Section 5 verifies the identity numerically: `0.028442 - 0.001585 + 0.029100 = 0.055957`.

Two readings matter. `RES` is the only term that measures usefulness: a model with `RES = 0` sorts nothing and, however well calibrated, is worthless. `UNC` is a property of the problem, not the model, so comparisons across populations must normalise by it, which is what the Brier skill score `1 - Brier/UNC` does. Section 5's raw model scores `1 - 0.055957/0.029100 = -0.9229`, worse than always predicting the base rate, and after recalibration `1 - 0.027526/0.029100 = 0.0541`.

The identity fails when `conf_b` is replaced by the bin midpoint; the residual is then absorbed into a third term and the decomposition no longer holds exactly.

### 10.7 `IMB-T07` — calibration and refinement

Any forecast can be written as a partition of the outcome space into groups with forecasts `f_g` and frequencies `o_g`. The forecast is *calibrated* if `f_g = o_g` for all `g`, and *refined* if the `o_g` differ. A constant forecast at the base rate is calibrated and unrefined; a forecast that assigns different values to groups with different frequencies is refined.

This is the cleanest rebuttal to "the model is well calibrated, so it is good". The Section 5 model after Platt has `ECE = 0.001613` and `RES = 0.001585` against `UNC = 0.029100`, so it is calibrated and barely refined. The remedy for insufficient refinement is better features or a better model family, never a calibrator.

### 10.8 `IMB-T08` — the ranking interpretation

ROC area equals the probability that a randomly chosen positive receives a higher score than a randomly chosen negative, with ties counted as one half. Because the expression depends only on the ordering, any strictly increasing map leaves it unchanged. Section 5's trapezoidal approximation over nine bins gives `0.742562`, which is an approximation to the pairing probability; with record-level scores the two coincide exactly.

The failure mode is comparing ROC areas across populations with different case mixes when the *class-conditional* score distributions also differ. T02 guarantees invariance under pure prior shift, but not under covariate shift, and a report that treats ROC area as prevalence-free will mislead whenever the shift is not pure.

### 10.9 `IMB-T09` — precision under prevalence change

For a fixed threshold and fixed class-conditional distributions, precision is a function of prevalence `pi`. Writing the threshold's true-positive rate as `tpr` and its false-positive rate as `fpr`, precision is `pi*tpr / (pi*tpr + (1-pi)*fpr)`. Differentiating with respect to `pi` shows the derivative is positive whenever `tpr > 0` and `fpr > 0`, so precision rises monotonically with prevalence.

That formula also gives the correct conversion between cohorts, which is the useful part: `precision_2 = 1/(1 + [(1-pi_2)/pi_2]*[(1-pi_1)/pi_1]^{-1}*... )` reduces to a ratio of odds and TPR/FPR. In practice the simplest safe procedure is to recompute precision from the confusion counts on the target cohort rather than to convert, because the conversion requires knowing `fpr` at the same threshold, which is exactly the quantity nobody reports.

### 10.10 `IMB-T10` — isotonic consistency

The pooled-adjacent-violators estimator is the least-squares monotone fit and is consistent for a monotone calibration curve as the sample grows, provided the bins refine. In finite samples its flexibility is the problem: with `B` blocks and few positives per block it can interpolate the noise, and the in-sample error goes to zero by construction, which is why an in-sample ECE of `0.000` carries no information.

The practical rule from Section 9's sizing arithmetic is that a bin needs enough positives to estimate its own rate. Below about `50` positives per block, the estimator's variance dominates and a two-parameter map is preferable even if it is misspecified in shape.

### 10.11 `IMB-T11` — the Wilson interval

The Wilson interval inverts the score test for a binomial proportion and has better coverage than the Wald interval near `0` and `1`. Its coverage still degrades under clustering: with `m` records per entity and intra-entity correlation `rho`, the design effect is `1 + (m-1)rho`, and the interval should use the effective sample size `n/[1+(m-1)rho]`.

Section 5's `652/6000` interval `[0.101041, 0.116801]` assumes independence. With `1.667` records per entity and `rho = 0.1`, the design effect is `1.0667`, so the interval should widen by `sqrt(1.0667) = 1.0328` to about `[0.100537, 0.117305]`. Small here, larger in real clustered data.

### 10.12 `IMB-T12` — variance under subsampling

Retaining a random fraction `1/u` of negatives multiplies the variance of any estimator that depends on the negative sample by approximately `u`, provided the discarded records were exchangeable with the retained ones. Section 8's Dossier `IMB-D02` observed a precision spread ratio of `3.44` against the predicted `sqrt(12.25) = 3.500`.

The result's condition is the whole content: exchangeability. Hard-negative mining retains the difficult negatives, which are exactly the informative ones, so its variance behaves far better than `u` predicts while its *bias* becomes the problem — the retained sample no longer represents the deployment population, and the prior correction that assumes uniform retention is invalid.

**Takeaway:** Six of these twelve results are identities that can be checked against a table of counts; the other six are statements whose conditions decide whether the identity applies at all.
