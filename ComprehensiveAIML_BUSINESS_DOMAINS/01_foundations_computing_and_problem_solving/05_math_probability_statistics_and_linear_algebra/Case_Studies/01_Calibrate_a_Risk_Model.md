---
case_id: "CASE-05-001"
case_title: "Calibrate a Risk Model"
case_version: "2.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Foundations and computing"
primary_pattern: "calibration-under-capacity"
secondary_patterns: ["expected-value-decision-rule", "label-maturity"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "risk decision lead"
source_index: "05_math_probability_statistics_and_linear_algebra/Case_Studies/INDEX.md"
review_trigger: "Any change to review capacity, loss per fraud, or score version"
---

# Calibrate a Risk Model

**Evidence declaration:** This is an ILLUSTRATIVE teaching case. The order volumes, reliability table, labels, cost inputs, and dollar outcomes are constructed so that every step of the argument can be re-derived by hand; none is a company deployment or a measured result. Where a number could be mistaken for a measurement it is labelled ILLUSTRATIVE at first use. **Implementation stage:** proposed. **Editorial status:** draft; source verification and publication review remain OPEN. **What is verified:** all arithmetic in Section 6 and Section 11, recomputed independently and cross-checked through three different routes (bin table, threshold formula, daily profit-and-loss). **What is not verified:** any claim that a real review queue behaved this way.

## 1. The One-Line Framing

A fraud score can rank orders correctly and still be unusable as a probability, so the discipline is to separate **ordering** (which survives any monotone distortion) from **magnitude** (which does not), then rebuild every threshold and every dollar figure from the calibrated magnitude rather than from the raw score's decimal places.

The one-sentence version a business sponsor can carry: *the model's ranking was worth keeping; the model's numbers were not, and every threshold and savings claim derived from those numbers was wrong by a factor between two and five.*

## 2. Context & Background

### 2.1 The decision that consumes the score

A high-risk order segment is scored at checkout. Each scored order enters a review queue where an analyst either clears it (releases the order) or holds it for verification (a call to the customer, a document request, or a cancellation). The score exists to order that queue, because the queue is short and the segment is long.

The consequence of a wrong clear decision is a chargeback plus lost goods. The consequence of a wrong hold decision is a delayed good order plus analyst minutes. Those two consequences are not symmetric, and the asymmetry is the whole reason a threshold must be derived rather than chosen by feel.

### 2.2 Why the segment is scored at all

The segment is defined by three conjunctive conditions at checkout: account age under 30 days, order value at or above 150 USD, and shipping country different from billing country. That definition is a policy artifact, not a model output, and it is stable across the window studied here. Membership is decided before scoring, so the population denominator cannot drift because of a model change.

Inside that segment the underlying fraud rate is far above the site-wide rate. A site-wide rate near 0.4 percent is uninteresting: no threshold can make reviewing 0.4 percent of traffic pay for itself. The segment concentrates risk by roughly twentyfold, which is what makes a per-order review economically defensible in the first place.

### 2.3 The model and the claim about its output

The scoring model is a gradient-boosted tree ensemble over 240 features: account tenure, device fingerprint stability, payment instrument history, address-to-IP distance, velocity counters at four time scales, and a small block of carrier and geography flags. It was trained on a calendar year of matured labels and it outputs a sigmoid of the ensemble margin.

The sigmoid makes the output *look like* a probability. It sits in the open interval, it is monotone in the margin, and the training loss was a logistic loss, so the fitting procedure had every opportunity to make it a probability. None of that is a guarantee. A logistic loss optimises a strictly proper scoring rule, but the fitted function is only calibrated if the model class contains the true conditional probability function and the fitting procedure reaches the optimum on the right population. Boosted trees with early stopping, class weighting, and a shifted label definition violate at least two of those conditions at once.

### 2.4 The operations side

Six analysts work the queue. Each handles roughly twenty orders per hour at the segment's document-verification rate, which gives 120 reviews per day against a segment volume of 1,000 scored orders per day. That capacity figure is the binding constraint on the entire analysis and it was never written into the model documentation.

The queue has a hard operational property: it must drain each day. An order held past its ship deadline is worse than an order released and later charged back, because the hold is visible to a good customer while the chargeback is absorbed by a reserve. So the practical rule has always been "review the top 120 by score, every day," and that rule was never connected to the cost model that the savings claim used.

### 2.5 The economics as they were stated

Two numbers were in the business case. A handled chargeback plus lost goods nets 420 USD per fraud, averaged across the segment's order-value mix. An analyst review costs 26 USD in loaded labour at 18 minutes per review. Both numbers were supplied by finance in a spreadsheet, both were treated as constants, and neither was ever revisited after the first quarter.

A third cost was omitted entirely: the friction imposed on a good customer whose order is held. Nobody had priced it, so nobody subtracted it, and the omission silently assumed it was zero. Section 6.10 shows that a friction cost of 18 USD per false positive moves the economically optimal review volume by a third.

### 2.6 What triggered the work

A quarterly review reported approximately 2.60 million USD of fraud prevented in the segment. Finance could trace only about 0.60 million USD of net benefit through the chargeback ledger. The gap was not fraud and not accounting error in the ledger; it was the model's output being multiplied by 420 in a spreadsheet as though the decimal places were probabilities.

Two separate mistakes produced the gap, and they pushed in opposite directions, which is why the reconciliation took a month rather than an afternoon. The raw probabilities were too high by roughly a factor of two. The claimed volume was five times larger than the volume the team could actually review. Correcting the first error alone would have overstated the answer after correction; correcting both required recomputing the precision at the real capacity point, which turned out to be far higher than the flagged-set average.

### 2.7 The baseline that had to be beaten

Before the model, the queue was ordered by a rules list: velocity above a fixed count, geo mismatch, first-order high value. The rules list had a precision of 21.0 percent at 120 reviews per day, and it was fully explainable to an auditor. Any replacement must be compared at the same capacity, not at the same threshold, because the rules list and the model produce different score distributions and therefore different volumes at any common cut point.

That baseline matters more than it looks. A model that achieves 36 percent precision at 120 reviews per day beats the rules list by 15 percentage points, which is worth 43.2 minus 25.2 caught frauds per day, or 18 frauds per day, or 7,560 USD per day at 420 USD. The entire business case lives in that difference, and the difference is measurable only if both systems are evaluated at the same review count.

## 3. The Business Question

| Field | Contract |
|---|---|
| Question | May the calibrated probability replace the raw score in the review threshold and in the reported benefit? |
| Owner | risk decision lead, accountable for the queue and the claim |
| Trigger | Quarterly benefit reconciliation disagrees with the chargeback ledger by 4.3x |
| Options | keep raw score as probability; recalibrate and re-derive; retrain the model; revert to the rules list |
| Objective | maximise net fraud prevented per day subject to 120 reviews per day and a hard daily drain |
| Hard constraint | no money figure may be computed from an uncalibrated score |
| Secondary constraint | every threshold must trace to a stated cost ratio and a stated capacity |
| Scope | 40,000 scored orders in the high-risk segment over 40 consecutive days |
| Out of scope | model architecture, feature engineering, chargeback recovery, site-wide traffic |
| Decision deadline | before the next quarterly benefit statement is published |
| Evidence required | reliability table on matured labels, Brier decomposition, threshold derivation, capacity recomputation |
| Acceptable outcome | either a calibration artifact adopted with a monitoring guardrail, or an honest statement that the score is not a probability |

The question is deliberately narrow. It is not "is the model good." A model can be good at ranking, bad at magnitude, and still be the right model to deploy once the magnitude is repaired by a monotone map. Conflating those two questions is what produced the original overstatement: the team had evidence for ranking and used it to support a claim about magnitude.

The distinction has a precise statement. Ranking quality is invariant under any strictly increasing transformation of the score. Magnitude is not invariant under any transformation except the identity. So a monotone recalibrator preserves every ranking metric exactly and changes every magnitude-derived number completely. That asymmetry is why calibration work looks like bookkeeping while being the difference between a defensible claim and an indefensible one.

## 4. Assumptions & Constraints

| ID | Assumption | Test | If violated |
|---|---|---|---|
| A-01 | Labels are mature at extraction time | compare the 30-day and 90-day label rate on an overlapping cohort | restrict the window to cohorts with 90 matured days |
| A-02 | Segment membership is decided pre-score | inspect the segment filter in the scoring log, not the model artifact | rebuild the denominator from the filter, not the scores |
| A-03 | Volume is stable at 1,000 orders per day | daily count coefficient of variation over the window | weight days, report the day-level distribution |
| A-04 | Reviewer capacity is 120 per day | queue ledger: reviews completed, not reviews requested | use the ledger figure; capacity claims are never evidence |
| A-05 | Loss per fraud is 420 USD | ledger: chargeback plus goods minus recovery | recompute the threshold; it is linear in this number |
| A-06 | Review cost is 26 USD | time study on 200 reviews | re-derive; the threshold moves slowly in this input |
| A-07 | No order is scored twice | distinct order key count equals scored event count | deduplicate on the order key before binning |
| A-08 | The score is monotone in the model margin | recompute the sigmoid on stored margins | if not, the score log is corrupt; stop |
| A-09 | The 40-day window is long enough for 10 bins | at least 3,000 positives overall | pool bins until each has at least 200 positives |
| A-10 | A monotone recalibrator is acceptable to policy | policy owner signs off on monotone-only transforms | restrict to two-parameter Platt if not |
| A-11 | Friction cost of a false positive is not zero | estimate from cancellation rate after hold | include it; see Section 6.10 |
| A-12 | The rules baseline is still reproducible | rerun the rules list on the same window | compare against the constant-prior rule instead |

Each assumption carries a test that a second person can run without asking the first. That property matters because the failure mode this case studies is precisely an assumption that everyone believed and nobody tested: that a sigmoid output is a probability.

Three assumptions are load-bearing for the headline numbers and deserve their own sentence. A-01 determines whether the label rate itself is trustworthy; an immature label window understates the fraud rate and therefore makes the model look *better* calibrated than it is. A-04 determines the volume used in every dollar figure; capacity claimed by the team rather than counted in the ledger is a recurring source of a factor-of-two error. A-05 determines the threshold; because the threshold is a ratio, a doubled loss per fraud halves the threshold and roughly doubles the review volume, which is the largest sensitivity in the whole analysis.

Two assumptions are explicitly *not* allowed to be assumed away. A-11 says friction is non-zero and unknown; the correct treatment is a sensitivity band, not a point estimate. A-09 says the window must support the binning; if it does not, the honest move is coarser bins with wider intervals, not ten bins with three positives each.

## 5. Data, Tools & Method

### 5a. Data Specification

| Element | Specification |
|---|---|
| Grain | one row per scored order decision event, not per order |
| Key | `order_id` (business key) plus `scoring_event_id` (idempotency key) |
| Population | high-risk segment as defined in 2.2; site-wide traffic excluded |
| Window | 40 consecutive days, 2026-06-01 00:00:00 UTC through 2026-07-10 23:59:59 UTC |
| Cut-off | decision timestamp; every feature must have an event time at or before it |
| Label | confirmed fraud adjudication, matured within 90 days of order creation |
| Label maturity | extraction on 2026-10-08, at least 90 days after the last order in the window |
| Extraction rows | 41,240 raw rows |
| Exclusions | 1,240 rows (3.0 percent): 780 duplicate scoring events, 460 with unmatured or missing adjudication |
| Analysis rows | 40,000 |
| Unit of the target | indicator in {0,1}; no partial credit, no refund-state ambiguity |
| Unit of the score | raw model output in [0,1], treated as an ordinal score |
| Unit of the calibrated output | probability of confirmed fraud for the order |
| Currency | USD, nominal, no inflation adjustment across 40 days |
| Time basis | event time for features, processing time for ledger entries; never mixed |

What is not available at decision time, and must never appear in a feature or in the evaluation: the adjudication outcome; the chargeback date; the number of contacts the customer required; any counter derived from post-decision activity; and the reviewer's own note. The reviewer note is the most dangerous of these because it is textually informative and trivially leaky.

The grain choice is not cosmetic. Analysing per order instead of per decision event silently drops re-scored orders (a customer who edits the address mid-checkout is scored twice) and inflates the denominator, which lowers the observed rate and makes the model look conservative. The 780 duplicate events excluded above are 2.0 percent of the raw extract, which is enough to move a rate of 8.0 percent by 16 basis points.

The label unit is binary and adjudicated, so the denominator of every rate in this case is a count of decisions with a resolved outcome, not a count of decisions with a predicted outcome. This sounds obvious and is violated constantly in practice by joining predictions to outcomes on a key that is present only for reviewed orders.

### 5b. Tools & Stack

| Component | Role | State |
|---|---|---|
| Scoring log | append-only record of raw score, margin, feature hash, version | exists |
| Label join | adjudication table keyed by decision event | exists |
| Reliability report | equal-count binning, ECE, Brier decomposition | constructed for this case |
| Recalibration fitter | isotonic regression with PAVA, Platt fallback | proposed |
| Calibration registry | versioned artifact: coefficients, fit window, fit hash | proposed |
| Policy config | cost inputs, capacity, friction band | proposed |
| Serving map | raw score to calibrated probability at request time | proposed |
| Benefit ledger | per-review outcome record feeding the quarterly claim | proposed |

The stack is deliberately boring. Every component above can be built from a dataframe library plus a small service; the intellectual content is in the acceptance criteria, not in the infrastructure. A calibration artifact is a few hundred bytes of coefficients or a step-function table, which means it can be reviewed line by line by a human, which is the property that makes it auditable.

The one component that deserves real engineering is the calibration registry. If the calibration artifact and the score artifact are versioned independently, then a score rollback silently leaves a mismatched map in production, and the failure is invisible because both artifacts are individually valid. The registry must therefore treat the pair (score version, calibration version) as a single addressable object.

### 5c. Method Summary

The pipeline is: deduplicate decision events, join matured labels, bin the raw score into ten equal-count bins, compute the reliability table, compute ECE and the Brier decomposition, fit a monotone recalibrator on a disjoint half of the window, evaluate on the other half, re-derive the threshold from stated costs, and recompute the benefit at the capacity point.

Every step before the recalibration fit is descriptive and cannot overfit. The fit is the only learned step, and it is constrained to be monotone, which caps its capacity at the number of blocks minus the number of merges. For ten blocks the cap is ten parameters, which against 40,000 rows is a favourable ratio; the risk is not overfitting in the statistical sense but *fitting to a single cohort's noise*, which is why the odd-day/even-day split exists.

The alternative methods, and why they are not the primary path:

- Platt scaling (two-parameter logistic fit on the score's logit). Cheapest, most stable, and wrong when the distortion is not logistic. Section 6.7 shows a two-point fit that misses an interior bin by a factor of 1.9.
- Beta calibration (three parameters, handles the asymmetric shape that raw sigmoids often have). Worth trying when Platt residuals show curvature by score level; not needed here because isotonic is affordable at this sample size.
- Histogram binning with a fixed number of bins. Equivalent to isotonic with no pooling constraint; simple to explain to an auditor, but it can be non-monotone and that breaks ranking preservation.
- Temperature scaling on the margin (one parameter). Appropriate for a neural network whose calibration defect is a single global sharpness; the reliability table here shows the defect varies by level, so one temperature cannot fix it.

The method ledger records which of these was tried, on which half of the data, and with what result. A method that was tried and lost is evidence, and omitting it invites the next analyst to repeat the experiment.

### 5d. Provenance, Quality, Exclusions, and Leakage Controls

Rejected records and the reason code assigned to each:

| Rejection | Count | Reason code |
|---|---|---|
| Duplicate scoring event for the same order within 24h | 780 | DUP_EVENT |
| No adjudication row at extraction | 310 | LABEL_MISSING |
| Adjudication still open at extraction | 150 | LABEL_IMMATURE |
| Decision timestamp after the window close | 0 | WINDOW_EDGE |
| Score present but feature hash absent | 0 | FEATURE_HASH |
| Negative or null order value | 0 | VALUE_INVALID |

The duplicate rule is "same order within 24 hours, keep the last event before the ship deadline." Keeping the last event rather than the first matters: the first event may be scored on a pre-edit address, which is precisely the case where the model's input was wrong through no fault of the model. Keeping the last event aligns the decision with the moment of consequence.

Three leakage controls were enforced mechanically rather than by review:

1. Every feature carries an event timestamp, and the loader rejects any feature row whose timestamp exceeds the decision timestamp. This catches the classic case where a velocity counter is materialised at extraction time rather than decision time.
2. The label table is joined only after binning is complete, so no bin edge can be chosen with knowledge of the outcome.
3. The recalibration fit and the evaluation use disjoint day sets, odd days versus even days, so within-day correlation does not leak across the split.

The day-level split was chosen over a random row-level split for a specific reason. Fraud arrives in coordinated bursts; two orders in the same burst share a device, an IP range, and often an adjudication decision made by one analyst in one sitting. A random split puts members of the same burst on both sides of the boundary, and the evaluation then measures the model's ability to recall a burst it has already seen. Splitting by day keeps bursts intact on one side.

### 5e. Reproducibility and Method-Selection Ledger

| ID | Artifact | Purpose | Accessibility | Limitation |
|---|---|---|---|---|
| ILL-01 | 10-bin reliability table | worked arithmetic for Sections 6.3 and 6.4 | constructed | not a production measurement |
| ILL-02 | cost grid in 6.10 | threshold sensitivity to friction | constructed | friction cost is unmeasured |
| CFG-01 | policy config with cost inputs | reproduce every threshold | proposed | needs change control |
| RPT-01 | benefit bridge in 11.3 | reconcile claim to ledger | constructed | uses the same labels |
| ORC-01 | constant-prior rule | floor comparator | trivial | ignores all features |

Reproduction recipe, in the order a second analyst would run it: (1) rebuild the decision-event table from the scoring log filtered by the segment definition; (2) apply the four rejection rules and confirm the counts in 5d; (3) bin on the raw score with equal counts, tie-broken by decision event id; (4) recompute the reliability table and confirm each bin's positive count; (5) recompute ECE and the Brier decomposition and confirm they satisfy the identity in 6.4; (6) fit isotonic on odd days, evaluate on even days; (7) recompute the threshold from CFG-01; (8) recompute the benefit at 120 reviews per day.

Step (5) is the one worth enforcing as a gate. If the reliability, resolution, and uncertainty terms do not add up to the Brier score to within rounding, then one of the three was computed on a different denominator, and every downstream comparison is suspect. The identity is cheap to check and it catches denominator drift, which is otherwise invisible.

## 6. Analysis Walkthrough

### 6.1 What the walkthrough must establish

Three things have to be shown, in this order, and each depends on the previous one. First, that the raw score's total predicted mass exceeds the observed number of positives, which establishes that the score is not a probability. Second, that this excess is concentrated in specific score ranges rather than spread evenly, which determines whether a monotone recalibrator can repair it. Third, that the repair changes the decision rule and the money, not merely the reported calibration metric.

A walkthrough that stops after the first step produces a report saying "the model is overconfident" and changes nothing. A walkthrough that stops after the second produces a new artifact with no decision attached. Only the third step converts a statistical finding into an operational one, and it is the step most often skipped.

### 6.2 The reliability table

Ten equal-count bins over 40,000 decisions, 4,000 rows each. `p̄` is the mean raw score in the bin; `k` is the count of confirmed frauds; `q` is the observed rate in the bin.

| Bin | Raw range | p̄ | k | q | q − p̄ (pp) |
|---|---|---|---|---|---|
| 1 | 0.002–0.015 | 0.010 | 8 | 0.0020 | −0.8 |
| 2 | 0.015–0.025 | 0.020 | 16 | 0.0040 | −1.6 |
| 3 | 0.025–0.040 | 0.030 | 32 | 0.0080 | −2.2 |
| 4 | 0.040–0.065 | 0.050 | 56 | 0.0140 | −3.6 |
| 5 | 0.065–0.100 | 0.080 | 88 | 0.0220 | −5.8 |
| 6 | 0.100–0.150 | 0.120 | 140 | 0.0350 | −8.5 |
| 7 | 0.150–0.220 | 0.180 | 220 | 0.0550 | −12.5 |
| 8 | 0.220–0.320 | 0.260 | 320 | 0.0800 | −18.0 |
| 9 | 0.320–0.480 | 0.360 | 640 | 0.1600 | −20.0 |
| 10 | 0.480–0.900 | 0.550 | 1600 | 0.4000 | −15.0 |

The observed sequence `q` is strictly increasing across all ten bins, which is the crucial structural fact: the raw score's *ordering* is correct everywhere. A monotone recalibrator can therefore be fitted without any risk of degrading ranking, and every ranking metric will be identical before and after.

The defect is uniform in sign and increasing in magnitude up to bin 9, then slightly decreasing. That shape is the signature of a model that compresses its estimates toward the middle on the probability scale, which is what happens when the tree ensemble is trained against a label stream whose base rate is higher than the serving population's. The training window had a segment fraud rate near 16 percent; the serving window has 7.8 percent. The model learned a prior it no longer faces.

### 6.3 Derivation C-01: predicted mass versus observed positives

```text
FORMULA [C-01]:  M = sum_b n_b * p_bar_b ,   and   bias = M/N - pi ,   pi = sum_b k_b / N
WHERE:  M = total predicted probability mass [frauds], N = decisions [count],
        n_b = bin count [count], p_bar_b = mean raw score in bin [probability],
        pi = observed prevalence [probability].
CONSTRAINT: bins partition the population; every p_bar_b is in [0,1];
            labels are matured and binary.
NUMERICAL EXAMPLE:  M = 4000*(0.010+0.020+0.030+0.050+0.080+0.120+0.180+0.260+0.360+0.550)
                    = 4000 * 1.660 = 6,640 predicted frauds.
                    observed positives = 8+16+32+56+88+140+220+320+640+1600 = 3,120.
                    pi = 3,120/40,000 = 0.0780.
                    M/N = 6,640/40,000 = 0.1660.
                    bias = 0.1660 - 0.0780 = +0.0880.
INTERPRETATION: the score asserts 6,640 frauds where 3,120 exist; its total
                predicted mass is 2.128x the truth. Any money figure computed
                as sum(p * loss) is inflated by at least this factor before
                volume effects are considered.
```

The 2.13 factor is the floor of the overstatement, not the whole of it. A savings claim multiplies this mass by a loss amount and then subtracts nothing, so the floor is a factor of 2.13 and the realised factor depends on which subset the claim covers. Section 11.3 shows the claim's realised inflation at 4.33.

An independent check that does not reuse the bin table: compute `M` directly from the per-row scores. Sum of 40,000 raw scores must equal 6,640 to rounding. That check costs one pass over the data and it catches the most common error in this analysis, which is binning on a different column (a rescaled score, or a percentile) than the one used for the mass computation.

### 6.4 Derivation C-02: ECE and the Brier decomposition

The Brier score decomposes into three terms whose sum is exact. This is the Murphyy decomposition, and it holds for any partition into bins where `p̄_b` is the bin mean forecast.

```text
FORMULA [C-02]:  BS = REL - RES + UNC
  REL = (1/N) * sum_b n_b * (q_b - p_bar_b)^2
  RES = (1/N) * sum_b n_b * (q_b - pi)^2
  UNC = pi * (1 - pi)
WHERE:  BS = mean squared error of the forecast [probability^2],
        REL = reliability, penalty for miscalibration,
        RES = resolution, reward for separating bins from the base rate,
        UNC = uncertainty, the irreducible variance of a binary outcome.
CONSTRAINT: exact identity when p_bar_b is the arithmetic mean forecast within
            each bin and the bins partition the population.
NUMERICAL EXAMPLE:
  REL = 0.1 * (0.008^2 + 0.016^2 + 0.022^2 + 0.036^2 + 0.058^2
             + 0.085^2 + 0.125^2 + 0.180^2 + 0.200^2 + 0.150^2)
      = 0.1 * 0.123214 = 0.0123214
  RES = 0.1 * (0.076^2 + 0.074^2 + 0.070^2 + 0.064^2 + 0.056^2
             + 0.043^2 + 0.023^2 + 0.002^2 + 0.082^2 + 0.322^2)
      = 0.1 * 0.136174 = 0.0136174
  UNC = 0.0780 * 0.9220 = 0.0719160
  BS  = 0.0123214 - 0.0136174 + 0.0719160 = 0.0706200
INTERPRETATION: resolution 0.0136 is almost entirely cancelled by reliability
                0.0123. The model separates the population well and then throws
                the separation away with a monotone distortion.
```

Direct evaluation confirms the decomposition: `BS = (1/N) * sum_b [k_b(1-p̄_b)^2 + (n_b-k_b)p̄_b^2]`. For bin 10 that term is `1600*0.2025 + 2400*0.3025 = 324 + 726 = 1050`; summing all ten bins gives `2824.80`, and `2824.80/40,000 = 0.07062`. The two routes agree to five decimal places, which is the check that licenses every subsequent use of `REL` and `RES`.

The ratios are the actionable part. `REL/RES = 0.0123214/0.0136174 = 0.905`, so 90.5 percent of the model's discriminative gain is spent on its miscalibration. Compare with the constant-prior forecast, whose Brier score is `UNC = 0.071916`: the model beats a forecast that ignores every feature by `0.071916 - 0.070620 = 0.001296`, a relative improvement of 1.8 percent. That is the honest headline for the raw score, and it is a long way from the story the business case told.

After monotone recalibration the reliability term collapses to zero *on the fitting data by construction*, so the in-sample Brier becomes `UNC - RES = 0.0719160 - 0.0136174 = 0.0582986`. The skill score against the constant-prior forecast is then `1 - 0.0582986/0.0719160 = 0.1893`, or 18.9 percent. The 17-point jump from 1.8 percent to 18.9 percent costs nothing in modelling effort and changes no ranking; it is entirely a repair of the output scale.

That in-sample figure is optimistic and must be reported as such. On held-out even days the reliability term does not vanish; a realistic post-recalibration `REL` is a few tenths of a percentage point rather than zero, because the per-block rates are estimated from finite counts. The correct claim is "the reliability term drops by about 95 percent," not "the model is perfectly calibrated."

### 6.5 Derivation C-03: the cost-optimal threshold

A review is an action with a fixed cost that is paid whether or not the order is fraudulent. Not reviewing an order that is fraudulent costs the net loss. Indifference between the two actions gives the threshold.

```text
FORMULA [C-03]:  p* = C_FP / C_FN
WHERE:  p* = review threshold [probability]; review iff calibrated p >= p*,
        C_FP = cost of a review, paid on every reviewed order [USD],
        C_FN = net loss if a fraud is released [USD].
CONSTRAINT: review is assumed to prevent the loss with certainty; C_FP and
            C_FN are constants independent of the order; no capacity limit.
NUMERICAL EXAMPLE:  C_FP = 26.00, C_FN = 420.00
                    p* = 26.00/420.00 = 0.061905 -> 6.1905 percent.
                    check: 0.061905 * 420 = 26.0004 USD, equal to C_FP.
INTERPRETATION: at p = p* the expected loss avoided equals the review cost, so
                the firm is indifferent. Below p* a review destroys value.
```

The derivation is one line of expected value and it is worth writing out because the common alternative is not a derivation at all. If the firm reviews, expected cost is `C_FP`. If the firm releases, expected cost is `p*C_FN`. Setting them equal gives `p* = C_FP/C_FN`. The frequently quoted form `p* = C_FP/(C_FP + C_FN)` answers a different question, namely the threshold when a review both costs `C_FP` and *fails* to recover the loss, which double-counts the loss. Using the wrong form here would give `26/446 = 0.0583`, a threshold 6 percent too low and a volume 8 percent too high.

Applying `p* = 0.0619` to the *calibrated* score selects bins whose observed rate is at or above 6.19 percent. From the table that is bins 8, 9, and 10, because bin 7 sits at 0.0550 and bin 8 at 0.0800. Volume is 12,000 decisions, 30 percent, or 300 per day. Precision on that set is `2,560/12,000 = 0.213333`.

Applying the same number to the *raw* score selects bins 5 through 10, because the raw value 0.065 is the first bin edge above 0.0619. Volume is 24,000 decisions, 60 percent, or 600 per day. Precision is `3,008/24,000 = 0.125333`.

The two rules differ by a factor of two in volume and 8.8 percentage points in precision, and neither is feasible at 120 reviews per day. That last observation is the pivot of the case: the cost-optimal rule and the raw rule are both wrong, for different reasons, and the actually implementable rule is a third thing.

### 6.6 Derivation C-04: capacity-constrained selection and the shadow price

With a hard capacity `K`, the problem acquires a constraint and a multiplier. The solution is unchanged in form; only the threshold's interpretation changes.

```text
FORMULA [C-04]:  max_a  sum_i [ a_i*(p_i*C_FN - C_FP) ]   s.t.  sum_i a_i <= K
                 => a_i = 1  iff  p_i >= lambda,  with lambda the multiplier
                 => lambda in [p_(K), p_(K+1)]  (order statistics of p)
WHERE:  a_i in {0,1} is the review decision, K = reviews available [count/day],
        lambda = shadow price of one review [USD of expected loss],
        p_(K) = the K-th largest calibrated probability.
CONSTRAINT: p_i treated as the true probability; no interaction between reviews;
            the constraint binds, i.e. more items pass the unconstrained rule
            than capacity allows.
NUMERICAL EXAMPLE: segment 1,000 orders/day, calibrated rates by bin
   bin 10: 100/day at 0.400  -> 40.0 frauds/day
   bin 9 : 100/day at 0.160  -> 16.0 frauds/day
   bin 8 : 100/day at 0.080  ->  8.0 frauds/day
   K = 120 => 100 from bin 10 plus 20 from bin 9
   caught = 40.0 + 20*0.160 = 40.0 + 3.2 = 43.2 frauds/day
   precision = 43.2/120 = 0.360
   marginal item sits inside bin 9, so lambda = 0.160
   implied C_FP = lambda * C_FN = 0.160*420 = 67.20 USD
INTERPRETATION: the queue behaves as if a false positive costs 67.20 USD,
                2.585x the accounting figure of 26.00. Capacity, not
                economics, is setting the threshold.
```

The shadow price is the single most useful number in this case for a business audience, because it is a price the organisation is already paying without knowing it. Every day at 120 reviews, the marginal reviewed order is one whose expected fraud loss is 67.20 USD against a review cost of 26.00. The gap of 41.20 USD per review is the value destroyed by not having a 121st analyst. Over 120 reviews that is 4,944 USD per day of forgone value, and it is forgone silently because no line item records it.

The gap also has a second reading, which matters for the model team: if the marginal item's true probability is 0.16 and the accounting threshold is 0.0619, then the model is not the constraint. Hiring to 300 reviews per day would move the marginal item to bin 8 at 0.080, still above 0.0619, and would raise daily net from 15,024 to 19,080 USD. After that, the next increment would cross into bin 7 at 0.055, which is below the threshold, and the expansion stops paying. The economically optimal queue length is 300 per day, and the firm runs at 120.

### 6.7 Derivation C-05: Platt scaling, and where a global fit fails

Platt scaling fits two parameters on the logit scale: `p = sigma(a*z + b)` where `z = logit(raw)`. A two-point solve is enough to see the shape, and enough to see its limitation.

```text
FORMULA [C-05]:  p = 1/(1+exp(-(a*z+b))),  z = ln(s/(1-s))
WHERE:  s = raw score (0,1); z = logit of raw score; a = slope, a>0 preserves
        ordering; b = intercept, b<0 shifts probabilities down.
CONSTRAINT: distortion must be logistic in z; a single (a,b) pair must serve
            the whole score range.
NUMERICAL EXAMPLE (two-point solve on bins 8 and 10):
  bin 8 : s=0.260 -> z = ln(0.260/0.740) = ln(0.351351) = -1.0460
          target q = 0.080 -> logit = ln(0.080/0.920) = -2.4418
  bin 10: s=0.550 -> z = ln(0.550/0.450) = ln(1.222222) = +0.2007
          target q = 0.400 -> logit = ln(0.400/0.600) = -0.4055
  a = (-0.4055 - (-2.4418)) / (0.2007 - (-1.0460)) = 2.0363/1.2467 = 1.6334
  b = -2.4418 - 1.6334*(-1.0460) = -2.4418 + 1.7085 = -0.7333
  refit check at bin 10: 1.6334*0.2007 - 0.7333 = -0.4055  (matches target)
INTERPRETATION: a = 1.633 > 1 means the recalibration is steeper than identity
                on the logit scale: the raw score spreads probabilities too
                little at the top while sitting too high overall, so the fix
                must both shift down and stretch.
```

The interior check exposes the limitation. Apply the fitted pair to bin 6: `z = ln(0.120/0.880) = -1.9924`, so `1.6334*(-1.9924) - 0.7333 = -3.2544 - 0.7333 = -3.9877`, and `sigma(-3.9877) = 1/54.93 = 0.0182`. The observed rate in bin 6 is 0.0350. The global two-parameter fit under-predicts that bin by a factor of 1.92.

That is not a bug in the fit; it is the fit telling the truth about its own capacity. Two parameters can match two points exactly and must compromise everywhere else. When the distortion varies by score level, as this reliability table shows it does, the honest options are a richer parametric family (beta calibration), a non-parametric monotone fit (isotonic), or an explicit decision to accept the interior error because the operating region is at the top. The last option is defensible here, since the decision lives in bins 8 through 10, but it must be written down rather than assumed.

`a > 0` is the property that preserves ordering and it is the only property that must be enforced. If a fit ever returns `a <= 0`, the fitting set contains a reversed relationship, which means the score direction or the label polarity was flipped somewhere upstream. That is a stop-the-line condition, not a modelling choice.

### 6.8 Derivation C-06: isotonic regression by pool-adjacent-violators

Isotonic regression finds the non-decreasing step function minimising weighted squared error. The algorithm is PAVA: scan left to right, and whenever a block's value exceeds the next block's value, merge them and replace both with their pooled weighted mean.

```text
FORMULA [C-06]:  minimise  sum_b w_b (y_b - q_b)^2   s.t.  q_1 <= q_2 <= ... <= q_B
                 merge rule: q_pool = (w_b*y_b + w_c*y_c) / (w_b + w_c)
WHERE:  y_b = observed rate in block b [probability]; w_b = block weight,
        normally the block count [count]; q_b = fitted value [probability].
CONSTRAINT: weights equal the block counts for the fit to be the least-squares
            solution; merging is repeated until no violation remains.
NUMERICAL EXAMPLE (violation, equal weights 1,000 each):
  observed: 0.010, 0.030, 0.052, 0.049, 0.120, 0.210
  block 3 (0.052) > block 4 (0.049) -> merge
  q_pool = (1000*0.052 + 1000*0.049)/2000 = 101/2000 = 0.0505
  after merge: 0.010, 0.030, 0.0505, 0.120, 0.210  -> monotone, stop.
  weighted variant: weights 2,000 and 1,000
  q_pool = (2000*0.052 + 1000*0.049)/3000 = (104+49)/3000 = 0.0510
INTERPRETATION: pooling is not smoothing for its own sake; it is the exact
                constrained least-squares answer, and it destroys the block
                boundary because the data no longer support one.
```

On the ten-bin table of 6.2, no violation occurs and PAVA returns the observed rates unchanged. That is the correct outcome and it is worth stating explicitly, because a reader who expects the recalibrator to "do something" will suspect a broken implementation. The recalibrator's job is to enforce monotonicity, and if the empirical rates are already monotone there is nothing to enforce.

At a finer 20-block resolution the same population does produce a violation of exactly the kind above, at the 3rd and 4th block, with rates 0.052 and 0.049. So the choice of resolution changes whether pooling happens. That is a real property of the method and a real decision: coarser blocks give more stable rates and fewer merges but coarser resolution near the threshold; finer blocks resolve the threshold region and pay for it with wider intervals.

Pooling has an operational consequence that is easy to miss. After a merge, every item in the pooled block receives the same calibrated probability. Two orders that the model ranked differently now carry identical numbers. If the serving path sorts on calibrated probability, it must break ties on the raw score — otherwise the pool destroys ordering inside the block and precision at a capacity cut falls to the block's average. In this population the pooled block spans 10 percent of volume, so a sort that respects only the calibrated value loses whatever separation existed inside that block. The rule is: sort by `(q, s)` lexicographically, never by `q` alone.

### 6.9 Derivation C-07: how many rows are needed to resolve the threshold

A threshold is only actionable if the data can distinguish the rates on either side of it. The Wilson interval gives an asymmetric interval appropriate for small counts, and the interval width sets the resolution of the decision.

```text
FORMULA [C-07]:  Wilson interval for a proportion
  centre = (q + z^2/(2n)) / (1 + z^2/n)
  half   = (z/(1+z^2/n)) * sqrt( q(1-q)/n + z^2/(4n^2) )
WHERE:  q = observed rate, n = bin count, z = 1.96 for 95 percent,
        output = [centre-half, centre+half] on the probability scale.
CONSTRAINT: independent observations within the bin; z from the normal
            approximation, adequate when n*q >= 5.
NUMERICAL EXAMPLE (bin 7, q = 0.055, n = 4,000):
  z^2/n = 3.8416/4000 = 0.0009604
  centre = (0.055 + 0.00048)/1.0009604 = 0.055428
  half   = (1.96/1.0009604)*sqrt(0.051975/4000 + 3.8416/6.4e7)
         = 1.95812*sqrt(1.299375e-5 + 6.0025e-8)
         = 1.95812*0.00361299 = 0.0070745
  interval = [0.04835, 0.06250]
INTERPRETATION: the upper bound 0.0625 sits above the threshold 0.0619, so the
                status of bin 7 is not resolved by 4,000 rows. Coarser statement:
                the data cannot tell us whether bin 7 should be reviewed.
```

The consequence is concrete. Bin 7 contains 100 orders per day. If its true rate is 0.048, reviewing it destroys `100*(26 - 0.048*420) = 100*(26 - 20.16) = 584` USD per day. If its true rate is 0.062, reviewing it gains `100*(0.062*420 - 26) = 100*(26.04 - 26) = 4` USD per day. The decision is worth roughly 588 USD per day in expectation and the data cannot resolve it, which means the correct action is neither review nor skip but *collect more data on that stratum*.

Required sample size to settle it, using the normal approximation for a target half-width `h = 0.005`: `n = q(1-q)z^2/h^2 = 0.055*0.945*3.8416/0.000025 = 0.19965/0.000025 = 7,986`. At 100 orders per day in that stratum, that is 80 days of data collection, or pooling two adjacent bins into a 200-per-day stratum with a coarser question.

The same arithmetic applied to bin 8, `q = 0.080`, `n = 4,000`: `half = 1.95812*sqrt(0.0736/4000 + 6.0025e-8) = 1.95812*0.0042895 = 0.00840`, giving `[0.0716, 0.0884]`. That interval's lower bound is 0.0716, comfortably above 0.0619, so bin 8's status *is* resolved. This is the structural reason the operating decision is robust even though the exact threshold is not: the threshold falls in a 2.5 percentage-point gap between bin 7's upper bound region and bin 8's lower bound, and gaps that wide survive label noise.

### 6.10 Boundary and counterexample: when the cost rule breaks

The threshold `p* = C_FP/C_FN` assumed that a review prevents the loss with certainty. Suppose the reviewer catches the fraud with probability `c < 1` — a real property of document review, where a determined fraudster produces convincing documents.

```text
FORMULA [C-03']: review: C_FP + (1-c)*p*C_FN ;  release: p*C_FN
                 => p* = C_FP / (c * C_FN)
WHERE:  c = probability the review catches a genuine fraud [dimensionless],
        0 < c <= 1.
CONSTRAINT: c is constant across orders and independent of p; the reviewer's
            effort does not vary with the score.
NUMERICAL EXAMPLE:  C_FP = 26.00, C_FN = 420.00
  c = 1.00 -> p* = 26.00/420.00 = 0.0619  -> bins 8,9,10 -> 300/day
  c = 0.85 -> p* = 26.00/357.00 = 0.0728  -> bins 8,9,10 -> 300/day
  c = 0.60 -> p* = 26.00/252.00 = 0.1032  -> bins 9,10    -> 200/day
INTERPRETATION: the threshold moves continuously but the operating set moves
                discontinuously, because the score distribution is sparse
                between 0.080 and 0.160. A 15-point drop in catch rate changes
                nothing operationally; a 40-point drop removes a whole bin.
```

This is the general lesson about thresholds on empirical score distributions: they are step functions of their inputs. A threshold is a smooth function of the cost ratio, but the *action set* is a step function because the population's rates are discrete lumps. Between steps, changes in the cost inputs are absorbed with no effect; at a step, a small change flips 100 orders per day. Any sensitivity analysis that reports the threshold's movement without reporting the action set's movement has reported the wrong quantity.

A second boundary, distinct from the first: the rule assumes `C_FN` is constant across orders. It is not. `C_FN` is the order value plus chargeback fee minus recovery, so it varies by an order of magnitude inside the segment. With a heterogeneous `C_FN_i`, the correct rule is per-order, `review iff p_i*C_FN_i >= C_FP`, not `p_i >= C_FP/C_FN_bar`. Using the mean loss as though it were every order's loss systematically under-reviews low-value orders with high fraud probability and over-reviews high-value orders with low fraud probability. The magnitude is easy to bound: for an order at half the mean value, the correct threshold is `2*C_FP/C_FN = 0.1238` rather than 0.0619.

A third boundary, and the one that matters most for the business case: the capacity constraint means the unconstrained rule is not what is implemented. Optimising a rule that cannot be executed, and then reporting the optimised value, is the mechanism that produced the original 4.33x overstatement. The rule that runs is the capacity rule; the reported benefit must be computed from the capacity rule; and the unconstrained optimum belongs in the business case as an argument for capacity, not in the benefit statement as an achievement.

### 6.11 What the walkthrough establishes

Four results are now established by computation rather than assertion. The raw score's predicted mass is 6,640 against 3,120 observed, a factor of 2.128. The reliability term consumes 90.5 percent of the resolution term, so the model's ranking is nearly all of its value and the raw magnitude is nearly none of it. The cost-optimal threshold is 6.19 percent, which selects 300 orders per day, twice the available capacity. The capacity rule actually running has a shadow price of 0.16 in probability units, or 67.20 USD in cost units, 2.58 times the accounting cost of a review.

None of those four numbers depends on the recalibration artifact that Section 8 proposes. They are descriptive facts about the score and the queue. The recalibration artifact exists only to make the serving path able to quote a probability that a money calculation can use, which is a narrower purpose than "improve the model."

## 7. The Finding

The supported conclusion is conditional and specific. The raw score is a valid ordering of the segment and an invalid probability for it; the two properties must be separated before any threshold or benefit figure is computed. The magnitude defect is a uniform over-prediction of 8.8 percentage points at the population level, concentrated in bins 6 through 10, and it is repairable by a strictly monotone map with no loss of ranking.

Quantitatively: mean predicted probability 0.166 against observed prevalence 0.078; ECE 0.088 on ten equal-count bins; Brier score 0.07062 against a constant-prior floor of 0.07192, a relative skill of 1.8 percent that rises to 18.9 percent after recalibration with no change to any ranking metric.

Operationally: the review threshold that the firm's own cost inputs imply is 6.19 percent, which selects 30 percent of volume or 300 orders per day, and the firm can review 120. The rule that actually runs therefore sits at a shadow price of 0.16, not 0.0619, and the marginal reviewed order is worth 67.20 USD of expected loss against a 26.00 USD review cost. Capacity, not calibration, is the binding constraint on captured value, and calibration is what makes that statement visible.

For the benefit claim: the reported 2.60 million USD was computed by multiplying raw scores by the loss amount over a flagged set the team could not review. The same computation on calibrated probabilities over the reviewable set yields 725,760 USD gross, and 600,960 USD net of review cost, which is the figure finance could trace. The gap is not a modelling failure; it is three arithmetic decisions stacked on top of each other, two of which inflated the number and one of which deflated it.

What the finding does not support: any claim that recalibration improves the model's ability to find fraud. It does not. Precision at 120 reviews per day is 36.0 percent before and after, because the transform is monotone. Recalibration improves the *decisions derived from the score* and the *truthfulness of the reported money*, and those are different goods from accuracy.

## 8. System/Solution Design

### 8.1 Design goal and non-goals

The goal is a serving path that cannot emit an uncalibrated probability into a money calculation, and cannot emit a calibrated probability from a calibration artifact that does not match the score artifact in production. The non-goals are: improving ranking, changing the feature set, retraining the ensemble, and building a general experimentation platform. Those are all worthwhile and none of them is this.

The design is deliberately small. A calibration artifact is a step table of ten to twenty rows, or two floats. The engineering is not in computing it but in pinning it, versioning it with its score, monitoring it as the population drifts, and refusing to serve when the pairing is broken.

### 8.2 Component layout

```text
[checkout] -> [score service] -> (raw score, score_version, feature_hash)
                    |
                    v
          [decision service]
             |            \
             |             +--> [calibration registry] --(map for score_version)
             |                        |
             v                        v
      [policy evaluator] <--- (calibrated p, calibration_version)
             |
             +--> review queue (sorted by p desc, then raw desc)
             +--> release
             +--> [decision ledger]  (immutable: p, versions, cost inputs, action)
             |
             +--> [benefit ledger] --> [quarterly claim]
```

The decision service holds no durable state. It reads the calibration map, applies the policy, writes one ledger row, and returns. Every number used in the decision is recorded in the ledger row, including the cost inputs and their version, because the commonest audit failure is being unable to reconstruct which cost ratio was in force on a given day.

### 8.3 The four invariants

Invariant 1: **no money figure without a calibration version**. The benefit ledger's schema requires a non-null `calibration_version`; a row without one is rejected at write time. This converts a policy statement into a database constraint, which is the only form of policy that survives a deadline.

Invariant 2: **the pair is atomic**. The registry stores `(score_version, calibration_version, fit_window, fit_hash)` as one record. Deploying a score without its calibration is impossible because the decision service resolves them together; a score rollback rolls back the map with it.

Invariant 3: **ties break on the raw score**. Sorting is lexicographic on `(calibrated_p, raw_score)`. Isotonic pooling creates exact ties, and a tie broken arbitrarily at a capacity cut makes the daily selection non-reproducible, which is exactly the property an auditor tests first.

Invariant 4: **a broken map degrades to the prior, not to the raw score**. If the registry cannot be read or the fit hash does not match, the decision service uses the last known matured prevalence as the probability and flags every row. Falling back to the raw score would reintroduce the defect silently; falling back to a constant is visibly wrong and therefore safe.

### 8.4 The calibration artifact lifecycle

The fit runs on a schedule tied to label maturity, not to a calendar: it runs when a new 90-day-matured cohort of at least 20,000 decisions is available. In this segment that is roughly monthly. Each fit produces a candidate artifact, an evaluation report on the held-out days, and a decision record naming the evaluator.

Promotion requires four conditions: the reliability term on held-out days falls by at least 80 percent relative to the incumbent; the ranking metric at the capacity cut is unchanged within 0.5 percentage points; every block in the step table has at least 200 positives; and the fitted map is non-decreasing. A fit that fails any condition is retained as evidence but not promoted.

Retirement is by supersession rather than deletion. An artifact stays readable forever because ledger rows reference it. That is a storage cost of a few kilobytes per month and it is the difference between an auditable history and a set of numbers nobody can reconstruct.

### 8.5 Serving-path cost and latency

The map is a sorted array of at most twenty breakpoints; applying it is a binary search, then a table lookup, then a lexicographic sort key. That is tens of nanoseconds and it does not appear in any latency budget worth measuring. The only cost that matters is the registry read, which is cached and refreshed on version change.

The decision service is on the checkout critical path, so the fallback must be fast and total. A registry timeout after 50 milliseconds returns the constant-prior probability and sets a flag; it never throws, and it never blocks the order. An order released because a calibration service was slow is a much smaller problem than an order stuck because a probability was unavailable.

### 8.6 What the ledger must contain per decision

The raw score and the calibrated probability, both explicitly labelled; the score version and the calibration version; the cost inputs in force (`C_FP`, `C_FN`, `capacity`); the action taken and the reason code; the ordering key used, so the capacity cut can be replayed; the decision timestamp in UTC; and the segment filter version. Twelve fields, all of which exist to answer one question later: *why did this order get this decision on this day.*

The ordering key deserves emphasis because it is the field most often omitted. Without it, a replay cannot distinguish "this order was reviewed because it was above the cut" from "this order was reviewed because the queue had room," and those two have completely different implications for whether the rule is being followed.

## 9. Evaluation & Validation

### 9.1 Metric definitions

| ID | Metric | Numerator | Denominator | Window | Comparator | Status |
|---|---|---|---|---|---|---|
| M-01 | calibration bias | sum of (p − y) | decisions with matured labels | held-out even days | 0, tolerance ±1.0 pp | executed on ILL-01 |
| M-02 | ECE, 10 equal-count bins | sum of n_b*abs(q_b − p̄_b) | N | same as fit | < 1.0 pp | executed on ILL-01 |
| M-03 | Brier skill score | 1 − BS/UNC | unitless | held-out | > 0 | executed on ILL-01 |
| M-04 | precision at 120/day | frauds in top 120 per day | 120 | 40 days | rules baseline 21.0 pp | executed on ILL-01 |
| M-05 | recall at 120/day | frauds in top 120 per day | all frauds that day | 40 days | 43.2/78.0 = 55.4 pp | executed on ILL-01 |
| M-06 | expected-cost regret | cost of policy − cost of oracle | decisions | 40 days | < 5 pp of oracle cost | not executed |
| M-07 | monotonicity of the map | violations in the step table | breakpoints | artifact | exactly 0 | executed on fit |
| M-08 | tie rate after pooling | decisions in pooled blocks | N | artifact | < 10 pp | executed on fit |
| M-09 | block positive count | positives in the smallest block | — | artifact | >= 200 | executed on fit |
| M-10 | queue drain | reviews completed | reviews required that day | daily | 100 pp | not executed |

`M-06` is the most informative metric and the hardest to compute, because the oracle requires true per-order probabilities that no one observes. The workable substitute is the cost of the policy under the recalibrated probabilities versus the cost of the best achievable policy under the same probabilities, which measures *policy* regret rather than *model* regret. It is worth stating which of the two is being reported.

### 9.2 Guardrails

Four guardrails run continuously and none of them is a model-quality metric:

- Queue drain rate by day. A drop below 100 percent means the capacity constraint binds harder than assumed, and the shadow price computed in 6.6 is then an underestimate.
- Share of held orders that are later verified as legitimate. This is the friction signal; a rise of 5 percentage points triggers a re-derivation of `C_FP` because the omitted friction term has become material.
- Population mix drift, measured on the segment filter's components separately (account age, order value, country pair). Mix drift changes prevalence without changing the model, so it degrades calibration while leaving ranking intact.
- Label arrival lag. If adjudications start arriving later, the next fit silently uses a thinner matured cohort. The guardrail is the count of matured labels at fit time, and it fires before the fit runs.

### 9.3 Uncertainty treatment

Every rate in this case is reported with a Wilson interval rather than a standard error, because the low bins have small counts and the symmetric interval goes negative. The population-level bias of 0.088 is not accompanied by a simple interval, since it is a sum of correlated bin errors; the honest treatment is a day-level block bootstrap, resampling whole days and recomputing the bias each time.

Three specific uncertainties are material and each is handled differently. The `C_FN` input is a ledger average with a wide item-level spread; it is handled by sensitivity (Section 6.10), not by an interval, because the question is "does the action change," not "what is the mean." The friction cost is unmeasured; it is handled by a band from 0 to 60 USD. The bin rates carry sampling error; they are handled by Wilson intervals and by the explicit statement in 6.9 that bin 7's status is unresolved.

### 9.4 What was executed and what was not

Executed: the reliability table, the Brier decomposition with its cross-check, the threshold derivations, the capacity recomputation, the PAVA trace on a constructed violation, the Wilson intervals, and the benefit bridge. All of these are arithmetic on a constructed table and all of them were recomputed independently.

Not executed: any live serving test, any A/B experiment, any latency measurement, any queue simulation with real arrival processes, and any evaluation on a population other than the constructed one. The absence of an experiment is the largest gap in the evidence and it is not repairable offline, because the decision variable is which orders get reviewed and the counterfactual is unobservable without randomisation.

A future randomised evaluation would assign a small random share of orders below the threshold to review, measure their true fraud rate, and use that to validate the calibrated map on the region where the map is extrapolating. That design costs review capacity, which is the scarce resource, so it needs a sponsor willing to buy evidence with throughput.

## 10. Action Taken

No production change was made and no claim was restated publicly. The actions taken were analytical and organisational, in this order.

First, the benefit bridge in Section 11.3 was written and circulated to finance and to the model team as a single page. The bridge's purpose was to make the three errors separable, because a single "we were wrong by 4.3x" message invites a defensive argument about which part is whose fault, while a four-step bridge assigns each step to a specific decision.

Second, the cost inputs were moved out of the spreadsheet and into a versioned configuration with an owner and a monthly review. `C_FN` had not been revisited in four quarters while average order value had risen; `C_FP` had never included friction at all.

Third, the raw score was removed from the benefit computation. The reporting job now refuses to run without a calibration version, and the refusal is a hard error rather than a warning, because a warning in a monthly job is a warning nobody reads.

Fourth, capacity was put into the model documentation as a first-class input alongside the cost ratio, with the shadow-price computation attached. The consequence was an unplanned but useful outcome: the capacity expansion request moved from "we need more analysts" to "the marginal review is worth 67.20 USD against a 26.00 USD cost, so the 121st analyst returns 41.20 USD per review," which is a different conversation.

Fifth, the recalibration artifact was specified, built for the window studied, and left unpromoted pending the four promotion conditions in 8.4. It is scheduled to be promoted with the next matured cohort so that the fit and the evaluation use independent label vintages.

Deliberately not done: retraining the ensemble. The reliability table shows the ordering is sound to the resolution the data supports, and a retrain would confound two changes. The sequencing rule is repair the output scale first, measure, then consider the model, because otherwise a retrain that fixes calibration and breaks ranking looks like an improvement.

## 11. Outcome & Results

### 11.1 Result status

**Not measured.** No order was reviewed differently because of this work, and no production system changed behaviour. The results below are arithmetic on a constructed 40,000-row table (ILL-01) and a constructed cost configuration (CFG-01). They are reported because they are fully checkable, not because they are measurements.

### 11.2 Calibration results on ILL-01

| Quantity | Raw score | After monotone recalibration | Denomination |
|---|---|---|---|
| mean predicted probability | 0.1660 | 0.0780 | probability |
| observed prevalence | 0.0780 | 0.0780 | probability |
| signed bias | +0.0880 | 0.0000 in-sample | probability |
| ECE, 10 equal-count bins | 0.0880 | 0.0000 in-sample | probability |
| reliability term REL | 0.0123214 | 0.0000000 in-sample | probability squared |
| resolution term RES | 0.0136174 | 0.0136174 | probability squared |
| uncertainty term UNC | 0.0719160 | 0.0719160 | probability squared |
| Brier score | 0.0706200 | 0.0582986 | probability squared |
| Brier skill vs constant prior | 0.0180 | 0.1893 | ratio |
| precision at 120 reviews/day | 0.3600 | 0.3600 | ratio |
| recall at 120 reviews/day | 0.5538 | 0.5538 | ratio |

The precision and recall rows are identical by construction, and the identity is the point. A monotone transform cannot change which orders are in the top 120. Recall is `43.2/78.0 = 0.553846`, computed from 43.2 frauds caught per day against 78 frauds present per day.

The in-sample zeros in the recalibration column are not a claim of perfect calibration. They are what least-squares isotonic regression produces on its own fitting data, and reporting them as an achieved result would be the same error as reporting training accuracy. On held-out days the reliability term is positive; the honest summary is that it falls by roughly an order of magnitude.

### 11.3 The benefit bridge

Four steps, each isolating one arithmetic decision. All amounts USD, all ILLUSTRATIVE, all computed from the same table.

```text
STEP 0  as claimed:      24,000 flagged * 0.258333 * 420 = 2,604,000
        flagged mean is the average raw score over bins 5..10
        = (0.080+0.120+0.180+0.260+0.360+0.550)/6 = 1.550/6 = 0.258333
STEP 1  calibrate the same flagged set, keep the volume:
        24,000 * 0.125333 * 420 = 1,263,360        delta = -1,340,640
        precision 0.125333 = 3,008 positives / 24,000 flagged
STEP 2  restrict to the reviewable volume, keep the flagged-set precision:
        4,800 * 0.125333 * 420 = 252,672           delta = -1,010,688
        4,800 = 120 reviews/day * 40 days
STEP 3  recompute precision at the capacity cut instead of the flagged average:
        4,800 * 0.360000 * 420 = 725,760           delta = +473,088
        top-120/day precision = 1,728/4,800 = 0.36
STEP 4  subtract review cost:
        725,760 - 4,800*26 = 725,760 - 124,800 = 600,960
                                                   delta = -124,800
FINAL   600,960 USD over 40 days = 15,024 USD/day
        per day check: 43.2*420 - 120*26 = 18,144 - 3,120 = 15,024
```

The bridge closes on 600,960 against a traced 600,000, a difference of 960 USD or 0.16 percent, which is within the rounding of a ledger reconciliation. The closure is the evidence that the three errors fully explain the gap and that no fourth error is hiding in it.

The shape of the bridge is the instructive part. Step 1 removes 1.34 million and step 2 removes another 1.01 million, taking the number from 2.60 to 0.25 million, a factor of ten. Step 3 then adds back 0.47 million by moving the operating point to where precision is genuinely 36 percent rather than the flagged average of 12.5 percent. A team that stopped after step 2 would have reported a benefit ten times too *low* and concluded the programme was worthless. The two errors do not cancel; they are sequential and they reverse the sign of the conclusion.

### 11.4 Per-day operating picture

| Policy | Reviews/day | Precision | Frauds caught/day | Gross value/day | Cost/day | Net/day |
|---|---|---|---|---|---|---|
| top 100 (bin 10 only) | 100 | 0.400 | 40.0 | 16,800 | 2,600 | 14,200 |
| top 200 (bins 9–10) | 200 | 0.280 | 56.0 | 23,520 | 5,200 | 18,320 |
| top 300 (bins 8–10) | 300 | 0.213 | 64.0 | 26,880 | 7,800 | 19,080 |
| capacity 120 | 120 | 0.360 | 43.2 | 18,144 | 3,120 | 15,024 |
| rules baseline | 120 | 0.210 | 25.2 | 10,584 | 3,120 | 7,464 |

The table makes the capacity argument in one glance. At 120 reviews the firm nets 15,024 USD per day. At 300 it would net 19,080, and the increment costs 180 reviews of analyst time which the `C_FP` column already prices at 26 USD each. The expansion is not a cost-saving argument; it is a value-capture argument, and it needs the fourth row to be true, which requires the calibration to be trusted.

The rules baseline row is included to keep the comparison honest. It catches 25.2 frauds per day against the model's 43.2, a difference of 18 frauds or 7,560 USD of gross value, and the rules list needs no calibration artifact at all. The model's advantage is real and it is 7.5 thousand USD per day, not the 60 thousand per day that the pre-correction claim implied.

### 11.5 What remains unknown

Four things are unknown and each has a name and an owner. The friction cost of a false positive is unmeasured; it belongs to the customer-experience owner. The catch probability `c` of a document review is unmeasured for this segment; it belongs to the operations lead, and it changes the operating set at `c = 0.60`. The stability of the calibrated map under the next cohort's mix is unknown; it is what the monthly refit measures. And the true fraud rate among orders *below* the threshold is unobserved by construction, which is why the randomised holdout in 9.4 is the only design that can validate the map where it extrapolates.

## 12. Failure Modes & What Went Wrong

### FM-01 Immature labels in the calibration cohort

**Mechanism.** Adjudications arrive 30 to 90 days after the order. A cohort extracted at 30 days has a *lower* observed fraud rate than the same cohort at 90 days, because opened cases are not yet resolved. Fitting on the immature cohort therefore produces a map that under-predicts, and the map looks like an improvement because it moves the score toward zero, which is the direction the raw score needed to move anyway.

**Competing causes.** A genuine decline in fraud in the recent period; a change in adjudication policy that closes weak cases as legitimate; a scoring-log gap that dropped high-risk orders.

**Discriminating check.** Compare the same cohort at two extraction dates 60 days apart. If the rate rises between extractions, the labels are immature. If it is flat, the decline is real.

**Containment, repair, verification.** Containment: freeze the map at the previous vintage. Repair: require at least 90 matured days for every row in the fit window. Verification: the fit report must print the minimum label age in days, and a value below 90 fails the report.

### FM-02 Prevalence shift with a stable model

**Mechanism.** The model is unchanged, the population mix changes, prevalence moves from 7.8 percent to 11 percent, and every calibrated probability is now too low. Ranking is unaffected, so the ranking dashboard stays green while every money calculation is wrong by 40 percent.

**Competing causes.** A real change in attack behaviour; a change in the segment filter; a change in the score distribution caused by a feature pipeline update.

**Discriminating check.** Recompute mean predicted probability against observed prevalence on matured data. Then decompose the prevalence change by segment component: if account age distribution shifted, it is mix; if not, it is behaviour.

**Containment, repair, verification.** Containment: recompute the decision threshold from the *observed* prevalence rather than the modeled one, which restores the operating point without changing the map. Repair: refit on the new cohort. Verification: a monthly chart of mean predicted versus observed with a two-standard-error band; two consecutive points outside the band trigger a refit.

### FM-03 ECE that is stable because the binning is fixed

**Mechanism.** ECE is not a proper scoring rule and it depends on the binning. Ten equal-count bins on a population of 40,000 hides a defect concentrated in 1 percent of volume, because that 1 percent shares a bin with 9 percent of well-behaved orders. The metric reads low, the monitor stays quiet, and the operating region at the top is badly calibrated.

**Competing causes.** Genuine calibration at the population level with a defect only in the tail; a defect that the metric family cannot see; a binning error that mixed rows.

**Discriminating check.** Recompute ECE at three resolutions, 10, 20, and 50 equal-count bins, and look at the *operating* bins specifically. If tail ECE moves by more than 30 percent across resolutions, the headline ECE is not a control.

**Containment, repair, verification.** Containment: report tail-conditional calibration alongside the aggregate. Repair: adopt Brier skill on the operating region as the primary metric, with ECE as a diagnostic. Verification: the metric specification names the bin count, the binning rule, and the region, and any change to those is a version change.

### FM-04 Isotonic pooling destroys ordering inside a block

**Mechanism.** PAVA merges adjacent blocks that violate monotonicity, assigning every item in the merged block the same probability. If the serving path sorts on the calibrated value alone, the block's internal ordering is lost and precision at a capacity cut falls to the block average.

**Competing causes.** A duplicate-key join that genuinely collapsed distinct rows; a scoring-log truncation that zeroed the low bits of the score; a sort implemented on a rounded display value.

**Discriminating check.** Count tied calibrated values at the cut point. If more than a handful of orders share the cut value and the queue picks arbitrarily among them, pooling is the cause. Compare precision with a `(calibrated, raw)` sort: if it improves, ordering was lost.

**Containment, repair, verification.** Containment: re-sort with the raw score as tiebreak. Repair: make the lexicographic key part of the serving contract. Verification: a unit test that constructs a pooled block and asserts the sort order within it.

### FM-05 The claimed volume is not the reviewable volume

**Mechanism.** The benefit computation multiplies precision by the number of *flagged* orders while the reviewed set is bounded by capacity. The two numbers differ by a factor of five in this case, and nothing in the reporting job compares them, because flagging and reviewing are owned by different teams and the join between them was never built.

**Competing causes.** A genuine capacity increase mid-quarter; a definitional mismatch where "flagged" meant "above threshold" for one team and "reviewed" for the other; a double-counted day.

**Discriminating check.** Reconcile the count of decisions in the benefit numerator against the count of rows in the review ledger for the same days. Any gap above 2 percent is a volume reconciliation failure, not a rounding issue.

**Containment, repair, verification.** Containment: restate the quarter's benefit at the reviewable volume. Repair: compute the benefit from the review ledger, one row per review, with the outcome joined. Verification: the reporting job's row count equals the ledger's row count, asserted at run time.

### FM-06 Threshold drift from stale cost inputs

**Mechanism.** `C_FN` moves with average order value and with recovery rates. Held constant across four quarters while order value rose, it makes the threshold too low, so the queue reviews too many marginal orders. The error is invisible because the threshold is recomputed from the same stale constant.

**Competing causes.** A deliberate policy decision to review more conservatively; a capacity increase that lowered the effective threshold; a change in the loss definition (gross versus net of recovery).

**Discriminating check.** Recompute `p*` from current ledger averages and compare with the configured value. A difference above 20 percent is drift; below that it is inside the input noise.

**Containment, repair, verification.** Containment: keep the current threshold and flag every decision made with the stale value. Repair: version the cost inputs and require a monthly review with a named owner. Verification: the policy config's `last_reviewed` date is asserted to be within 35 days at service start.

### FM-07 A score rollback without its map

**Mechanism.** Score version and calibration version are deployed independently. A rollback of the score leaves the newer map in place, so a different score is mapped through coefficients fitted for the old one. Both artifacts are individually valid, no error is raised, and the output is plausible and wrong.

**Competing causes.** A legitimate deliberate decoupling during an experiment; a map that was fitted on both score versions; a registry read that returned a stale cached entry.

**Discriminating check.** Log the pair on every decision and count distinct pairs per day. More than one pair in production is either an experiment or a bug, and the two are distinguished by whether an experiment record exists.

**Containment, repair, verification.** Containment: pin both versions to the last known-good pair. Repair: store the pair as one registry record with a composite key, so the mismatch is unrepresentable. Verification: a startup assertion that the resolved pair exists in the registry and that its fit hash matches.

### FM-08 Reporting gross value as net benefit

**Mechanism.** Gross value is `frauds caught * loss`; net is gross minus review cost. Both are legitimate numbers, and only one of them is a benefit. Reporting gross inflates by the review cost over the prevented loss, which here is `124,800/725,760 = 17.2` percent. On a smaller segment with lower loss per fraud, the inflation can exceed the benefit itself.

**Competing causes.** A cost that is accounted in a different budget line and therefore "not attributable"; a deliberate gross-versus-net distinction that was documented but not carried into the headline; a review cost that is genuinely near zero because analysts are salaried and idle.

**Discriminating check.** Divide the reported benefit by the review count and compare with the per-review gross value. If the reported per-review value equals the gross, the cost was omitted.

**Containment, repair, verification.** Containment: restate as net with the cost itemised. Repair: the benefit schema requires gross, cost, and net as three fields, and the headline is net. Verification: gross minus cost equals net, asserted in the reporting tests.

### FM-09 A calibrated probability used as a ranking score after pooling

**Mechanism.** A team adopts the calibrated value everywhere, including as the queue's sort key, and loses the within-block resolution. The failure is subtle: calibration metrics improve, ranking metrics at capacity degrade by 2 to 3 percentage points, and the two reports are read by different people.

**Competing causes.** A genuine change in the population that reduced separation; a feature pipeline regression; a change in the capacity cut.

**Discriminating check.** Recompute precision at the cut with both sort keys on the same day's data. If the calibrated-only sort is worse, pooling is the cause. A paired comparison removes the population effect entirely.

**Containment, repair, verification.** Containment: revert the sort key. Repair: define two named outputs, `p_calibrated` for money and `rank_key` for ordering, and forbid one from substituting for the other. Verification: a test that asserts the rank key is strictly ordered on a constructed pooled block.

## 13. Transferable Lesson

The transferable content is not the reliability table. It is the separation of three questions that a single number appears to answer: *is this item more likely than that one* (ordering), *how likely is it in absolute terms* (magnitude), and *what should we do given costs and capacity* (decision). A score can answer the first well, the second badly, and the third not at all, and the three answers have different owners, different metrics, and different failure modes.

Four practices transfer to any domain where a model output feeds a money calculation.

First, compute the predicted mass before anything else. Sum the probabilities and compare with the observed count of positives. It is one line of code, it needs no binning, and it detects the factor-of-two class of error immediately. In this case it would have caught the overstatement in the first week rather than the fourth month.

Second, state the decision rule with its inputs and its constraint. A threshold without a cost ratio is a preference; a cost ratio without a capacity is a fantasy. The pair `(p*, K)` is the minimum viable specification of a review policy, and it belongs in the model card, not in a spreadsheet owned by one analyst.

Third, compute the shadow price of the binding constraint. The number the organisation actually behaves as if it believes is the multiplier, not the configured threshold, and the gap between the two is a finding in its own right. Here the gap was 2.58x, which reframed the conversation from model quality to staffing.

Fourth, evaluate everything at the operating point. A metric computed over the whole population describes a population, not a decision. Precision at the capacity cut is the number that determines whether the queue creates value, and it is the number that was absent from the original business case.

The lesson has a corollary about honesty that is easy to state and hard to practice: a monotone recalibration changes no ranking metric, so any claim that recalibration "improved the model" must be a claim about magnitude, money, or decision quality, and it must say which. Conversely, a claim about magnitude that is supported only by ranking evidence is unsupported, however good the ranking is. The original business case was exactly that claim.

## 14. Connections & Cross-References

### 14.1 Within this unit

- `../Cheat_Sheets/01_Matrix_and_Vector_Identities.md` carries the linear-algebra identities used when the score is a linear predictor and the reliability problem reduces to a projection; the isotonic fit in 6.8 is a constrained quadratic program whose normal equations use those identities.
- `02_Derive_Gradient_Descent_for_Regression.md` is the companion case: the regression there is fitted by an iterative solver whose stopping rule is a convergence claim, which is the same kind of claim as a calibration claim, and it fails in the same way when the diagnostic is computed on the fitting data.
- `../Cheat_Sheets/99_25_Common_Issues.md` catalogues the numerical-edge and concept-confusion entries that this case instantiates: denominator drift, metric dependence on binning, and the confusion between a proper and an improper scoring rule.
- `INDEX.md` in this directory records the case's stage and its relation to the other cases in the unit.

### 14.2 Outside this unit

The pattern generalises to any scored queue with a capacity constraint: credit-line review, clinical triage, content moderation, insurance claim audit, and email deliverability. In each, the same three questions separate, and in each the same failure appears: the queue is ordered well, the score is quoted as a probability, and the benefit is computed over the flagged set rather than the reviewed set.

The specific transferable artifact is the four-step benefit bridge. Its value is that it assigns each correction to a decision-maker: step 1 to the model owner, step 2 to the operations owner, step 3 to the analyst who chose the operating point, step 4 to finance. A single-factor error message produces an argument; a bridge produces four small agreements.

### 14.3 Interview pack

#### Q1. From the ten-bin table, what is the model's mean predicted probability and what is the observed prevalence, and what does the difference mean?

**Answer:** Mean predicted is 0.166 and observed prevalence is 0.078, a bias of +0.088. The score asserts 6,640 frauds where 3,120 exist, a factor of 2.128.

**Reasoning:** Mean predicted is the weighted average of the bin means, `4000*1.660/40000`. Observed prevalence is total positives over total rows, `3120/40000`. Since all ten bins over-predict, the aggregate bias equals the ECE, `0.1*0.880 = 0.088`. The factor 2.128 is the quantity that inflates every `sum(p*loss)` computation before volume effects.

**Follow-up:** Ask what the bias would be if the population were reweighted to the training window's 16 percent prevalence. The answer is that it would shrink toward zero, which identifies the cause as a prior shift rather than a fitting bug.

**Red flag:** Computing the bias as a mean of per-bin *differences* without weighting by count. With equal bins it happens to give the right answer, and it silently breaks the moment bins differ in size.

**Scoring:** Full credit names both numbers, the factor, and the reason the aggregate bias equals ECE here. Partial credit for the two numbers without the mechanism.

#### Q2. Compute ECE with ten equal-count bins from the table and state what it does and does not tell you.

**Answer:** ECE is 0.088, the count-weighted mean absolute gap between bin rate and bin mean forecast. It says the score is overconfident by 8.8 points on average. It does not say where, and it is not a proper scoring rule, so it can be reduced by binning choices rather than by improving the forecast.

**Reasoning:** With 4,000 per bin the weights are 0.1, so ECE is `0.1*(0.008+0.016+0.022+0.036+0.058+0.085+0.125+0.180+0.200+0.150) = 0.1*0.880 = 0.088`. Because every gap has the same sign, ECE coincides with the signed bias here; that coincidence is a property of this table, not of ECE in general.

**Follow-up:** Recompute at 20 bins. If the value rises above 0.10, the defect is concentrated in a region the 10-bin version smears; if it falls, the 10-bin version was splitting a homogeneous region.

**Red flag:** Claiming ECE is a proper scoring rule, or comparing ECE values computed with different bin counts.

**Scoring:** Full credit gives 0.088 with the arithmetic, names the improper-scoring-rule caveat, and proposes the resolution sensitivity check.

#### Q3. A colleague sets the raw-score cut at 0.10 to hold review volume near 20 percent. What volume and precision result, and what is the true expected value per review?

**Answer:** The cut at 0.10 selects bins 6 through 10, which is 50 percent of volume, not 20. Precision there is `2,920/20,000 = 0.146`. Expected value per review is `0.146*420 - 26 = 35.32` USD, so each review still pays, but the queue is 400 orders per day over a 200-order target.

**Reasoning:** Bin 6 starts at 0.100, so the cut includes it. Positives in bins 6-10 are `140+220+320+640+1600 = 2,920` out of 20,000 rows. The intended 20 percent corresponds to a raw cut near 0.22, which is the bin 8 edge. The colleague reasoned about the score as a probability and therefore about volume as a probability mass.

**Follow-up:** Ask what calibrated cut gives 20 percent. The answer is the calibrated value at the 80th percentile, which is bin 9's lower edge, `q = 0.16` — and that is a cut chosen by volume, not by cost, which is a different and legitimate rule if capacity is the binding constraint.

**Red flag:** Accepting the 20 percent claim because "the score looked like a probability." The volume check costs one histogram.

**Scoring:** Full credit computes 50 percent and 0.146, notes the EV is still positive, and distinguishes a volume rule from a cost rule.

#### Q4. The claim was 2.60 million USD and finance traced 0.60 million. Name the three errors and say which direction each pushes.

**Answer:** Raw probability used at face value, overstated by 2.06x (pushes up). Flagged volume used instead of reviewed volume, overstated by 5x (pushes up). Gross reported instead of net, overstated by 17.2 percent of gross (pushes up). Recomputing precision at the capacity cut pushes down, which is the fourth correction and the reason a naive fix would understate.

**Reasoning:** Step 0 to step 1 applies the measured precision 0.125333 instead of the flagged mean 0.258333, a ratio of 2.061. Step 1 to step 2 applies the volume ratio `4,800/24,000 = 0.2`. Step 3 replaces 0.125333 with 0.36 at the capacity cut, a gain of 473,088. Step 4 removes 124,800 of review cost. The bridge closes at 600,960 against 600,000 traced.

**Follow-up:** Ask which error a reviewer would find first. The volume error, because the review ledger's row count is one query away, whereas the probability error requires the reliability table.

**Red flag:** Asserting that the errors "cancel out." They are sequential and multiplicative on different quantities; the net is a factor of 4.33, not a factor near one.

**Scoring:** Full credit names all three, gives the direction of each, and volunteers the capacity-precision reversal.

#### Q5. Which Brier decomposition term dominates, and what does that imply for where to spend effort?

**Answer:** Resolution is 0.01362 and reliability is 0.01232, so 90.5 percent of the discriminative gain is cancelled by miscalibration. Effort should go to the output map, not the features, because the features already separate the population to the resolution the data supports.

**Reasoning:** `REL/RES = 0.0123214/0.0136174 = 0.905`. The identity `BS = REL - RES + UNC` with `UNC = 0.071916` gives `BS = 0.07062`, confirmed independently by the direct `k(1-p)^2 + (n-k)p^2` sum, which totals 2,824.80 over 40,000. Recalibration drives REL toward zero and leaves RES untouched because RES depends only on the observed rates.

**Follow-up:** If RES were the small term and REL the large one, the answer would reverse: the features would be the problem and no monotone map would help. Ask the candidate to state which table row would tell them which regime they are in.

**Red flag:** Concluding that because the Brier score is close to the constant-prior score, the model is worthless. The near-tie is evidence about the *output scale*, not about the information content.

**Scoring:** Full credit computes both terms, the ratio, and states the implication for effort allocation.

#### Q6. With C_FP of 26 USD and C_FN of 420 USD, what is the threshold, and what happens at C_FN of 250 USD?

**Answer:** `26/420 = 0.061905`, so 6.19 percent. At `C_FN = 250` the threshold is `26/250 = 0.104`, so 10.4 percent, which moves the operating set from bins 8-10 to bins 9-10 and cuts volume from 300 to 200 per day.

**Reasoning:** The threshold is a ratio, so it is linear in `C_FP` and inverse in `C_FN`. A 40 percent fall in loss per fraud raises the threshold by 68 percent. The action set is a step function of the threshold because bin 8 sits at 0.080, above 0.0619 but below 0.104, so the step happens exactly at `C_FN ≈ 325 USD` where the threshold crosses 0.080.

**Follow-up:** Ask for the `C_FN` at which bin 7, at 0.055, enters the set: `26/0.055 = 472.7` USD. That is a 12.5 percent increase over the current input, which is well inside a year of order-value growth.

**Red flag:** Reporting the threshold without the resulting volume. A threshold change of 68 percent that changes nothing operationally and one that removes a third of the queue are very different findings.

**Scoring:** Full credit gives both thresholds, the volume consequence, and the step-function insight or the crossing value.

#### Q7. Why does isotonic pooling matter operationally, and what is the fix?

**Answer:** Pooled blocks receive one probability per block, so sorting on the calibrated value alone loses the ordering inside the block. At a capacity cut that picks part of a block, selection becomes arbitrary among tied items and precision falls to the block average. The fix is a lexicographic sort on `(calibrated, raw)`.

**Reasoning:** PAVA merges adjacent violators and assigns the weighted mean. In the constructed six-block trace, blocks at 0.052 and 0.049 pool to 0.0505 with equal weights, or 0.0510 with weights 2,000 and 1,000. Every item in that block now has the same number. A 10 percent-of-volume block at a cut inside it means 10 percent of volume is ordered by insertion order.

**Follow-up:** Ask how to detect the loss. Compute precision at the cut under both sort keys on the same day; a paired comparison isolates the sorting effect from any population change.

**Red flag:** Concluding that pooling is a bug to be avoided. It is the exact constrained least-squares solution; the bug is in the serving sort, not in the fit.

**Scoring:** Full credit identifies the tie, the arbitrariness at the cut, and the lexicographic fix. Partial credit for "pooling loses information" without the operational consequence.

#### Q8. Bin 7 has 100 orders per day at an observed 5.5 percent. Is it above or below the 6.19 percent threshold?

**Answer:** Unresolved. The Wilson interval at `n = 4,000` is `[0.0484, 0.0625]`, and its upper bound is above 6.19 percent. The data cannot settle the question at 4,000 rows.

**Reasoning:** `z^2/n = 0.0009604`, centre `0.055428`, half-width `1.95812*sqrt(1.299375e-5 + 6.0025e-8) = 0.00707`. Requiring a half-width of 0.005 needs `n = 0.055*0.945*3.8416/0.000025 = 7,986` rows, which is 80 days at 100 per day. The decision is worth about 588 USD per day across the two resolutions, and 80 days of ambiguity costs more than the information is worth in the short run.

**Follow-up:** Ask what to do with an unresolved stratum. The defensible answers are: pool it with the adjacent bin to get a coarser but resolvable question, or hold the conservative branch (review) while collecting, and record the choice.

**Red flag:** Reporting a point estimate of 5.5 percent and a binary verdict. The interval is the finding.

**Scoring:** Full credit gives the interval, the sample-size requirement, and an explicit choice under ambiguity.

#### Q9. Would you recalibrate or retrain?

**Answer:** Recalibrate first, measure, then decide about retraining. The reliability table shows the ordering is sound and the defect is a monotone distortion with a plausible cause — a prior shift from a 16 percent training population to a 7.8 percent serving population. A retrain would confound two changes and could fix calibration while degrading ranking.

**Reasoning:** `RES = 0.01362` is large relative to `UNC = 0.07192` for a 7.8 percent base rate, so the features carry real signal. The bias is uniform in sign and monotone in size across bins, which is the signature of a prior shift rather than a fitting failure. A retrain on the serving prior would probably fix it too, but it would also change the ordering, and then no metric could attribute the change.

**Follow-up:** Ask what evidence would reverse the decision. A resolution term near zero, or a non-monotone reliability curve, or a bias whose sign flips by segment — any of those points at the model rather than the map.

**Red flag:** "Retrain with more data," which is an action rather than a diagnosis and which does not state what would be measured afterward.

**Scoring:** Full credit sequences the two interventions, names the diagnostic that selects between them, and states the confound.

#### Q10. Labels mature at 90 days. How do you calibrate the most recent 90 days?

**Answer:** You do not. You calibrate on the newest cohort with at least 90 matured days and you *monitor* the immature period with a proxy. The honest options for the recent window are: hold the map fixed and watch the bias monitor, or use an outcome proxy that matures faster and validate the proxy against labels where both exist.

**Reasoning:** A cohort extracted at 30 days has a lower observed rate because open cases are unresolved, so the fitted map under-predicts. The 460 excluded rows in 5d are exactly this population. Fitting on them biases the map toward zero, which happens to be the direction the raw score needed to move, so the error hides inside an apparent improvement.

**Follow-up:** Ask for the fastest valid proxy. Reasonable candidates are chargeback *initiation* rather than adjudication, or a reviewer's hold decision, both validated against matured labels on the overlap. Each has its own bias and the validation is what licenses it.

**Red flag:** Fitting on immature labels and calling the result "recent calibration." It is the most common way this analysis goes wrong and it produces a plausible, wrong, improving-looking number.

**Scoring:** Full credit refuses the immature fit, names the 30-day versus 90-day rate comparison as the diagnostic, and proposes a validated proxy.

#### Q11. A two-point Platt fit gives a = 1.633 and b = -0.733. What does a > 1 mean, and where does this fit fail?

**Answer:** `a > 1` means the recalibration is steeper than identity on the logit scale: the raw score both sits too high overall and is too compressed at the top, so the fix must shift down and stretch. It fails in the interior: at bin 6 it predicts 0.0182 against an observed 0.0350, a factor of 1.92.

**Reasoning:** The fit solves `1.6334*(-1.046) + b = -2.4418` at bin 8 and `1.6334*0.2007 + b = -0.4055` at bin 10, giving `b = -0.7333` and reproducing the bin 10 target exactly. At bin 6, `z = -1.9924`, so `1.6334*(-1.9924) - 0.7333 = -3.9877` and `sigma(-3.9877) = 0.0182`. Two parameters match two points and must compromise elsewhere; when the distortion is level-dependent, a two-parameter family cannot be right everywhere.

**Follow-up:** Ask when the interior failure is acceptable. It is acceptable when the decision lives in the region that was matched, and unacceptable when the queue cuts through the interior. Here the cut is in bins 8-10, so it is acceptable if written down.

**Red flag:** Reporting `a` and `b` as a validated calibration without an interior check. The check is one row of the reliability table.

**Scoring:** Full credit explains `a > 1`, computes the interior prediction, and states the acceptance condition.

#### Q12. What is the shadow price of a review at 120 per day, and how do you compute it?

**Answer:** 0.16 in probability units, or 67.20 USD against a 420 USD loss. It is the calibrated probability of the marginal reviewed order — the 120th, which sits 20 orders into bin 9 at 0.16.

**Reasoning:** With capacity binding, the constrained optimum reviews in descending probability until capacity is exhausted, so the multiplier equals the marginal item's probability. Bin 10 supplies 100 orders per day at 0.40; the remaining 20 come from bin 9 at 0.16. `0.16*420 = 67.20`. The gap against the accounting cost of 26.00 is 41.20 per review, or 4,944 per day across 120 reviews.

**Follow-up:** Ask what the firm should do with a shadow price 2.58x the accounting cost. The answer is either buy capacity, which returns 41.20 per marginal review, or improve the model so that the 120th item is worth more — and those are different investments with different owners.

**Red flag:** Computing the shadow price as an average over the reviewed set. The average is 36 percent precision, 151 USD, which is not a price of anything.

**Scoring:** Full credit identifies the marginal item, computes 0.16 and 67.20, and states the capacity implication.

#### Q13. Calibration looks fine but the decisions are wrong. What are the competing causes?

**Answer:** Four candidates: the threshold was derived from stale cost inputs; the operating point is capacity-driven and the reported rule is the unconstrained one; the ranking degrades inside pooled blocks; or the population mix shifted so the map is calibrated for a population that no longer exists.

**Reasoning:** Each has a distinct signature. Stale costs show up as a threshold that does not match the current ledger ratio. A capacity-versus-rule mismatch shows up as a review count that differs from the rule's implied volume. Pooling shows up as ties at the cut. Mix shift shows up as a bias monitor excursion concentrated in one segment component.

**Follow-up:** Ask which to check first and why. The volume reconciliation, because it is one query and it distinguishes the first two candidates immediately.

**Red flag:** "Monitor the model more closely." That is not a diagnosis and it names no discriminating check.

**Scoring:** Full credit lists at least three candidates with distinct signatures and orders them by cost of checking.

#### Q14. How many rows do you need to pin an 8 percent rate to within half a percentage point?

**Answer:** About 11,310, using `n = q(1-q)z^2/h^2 = 0.08*0.92*3.8416/0.000025`. With 4,000 rows the half-width is 0.0086, more than 1.7 times the target.

**Reasoning:** `0.08*0.92 = 0.0736`, times 3.8416 gives 0.28274, divided by `0.000025` gives 11,309.7. The Wilson half-width at 4,000 is `1.95812*sqrt(1.84e-5 + 6.0e-8) = 0.00840`, so the interval is roughly `[0.072, 0.088]`.

**Follow-up:** Ask what to do when the budget is 4,000 rows. The answers are: widen the tolerance, pool adjacent bins to raise `n` at the cost of resolution, or accept the interval and make the decision robust to it rather than precise within it.

**Red flag:** Using `sqrt(p(1-p)/n)` and calling the result a 95 percent interval. At these counts the symmetry assumption matters and Wilson is the defensible choice.

**Scoring:** Full credit gives the sample size, the existing half-width, and a plan under a fixed budget.

#### Q15. Name the one invariant you would enforce in the serving path.

**Answer:** A decision row cannot be written without both a score version and a calibration version, and the pair must exist as one registry record. Everything else — thresholds, tie-breaks, fallbacks — is derived from that pair and can be reconstructed if the pair is known.

**Reasoning:** The failure that produces silently wrong money is a mismatched pair, and it is unrepresentable if the pair is the key. A threshold can be recomputed from the ledger's cost inputs; a probability cannot be recomputed from a raw score without the map. So the map's identity, not the threshold's value, is the thing that must be durable.

**Follow-up:** Ask what the fallback should be when the registry is unavailable. Constant prior, flagged, never the raw score — because the raw score is the defect.

**Red flag:** "Log everything." Logging without a schema that makes the bad state unrepresentable is a monitoring plan, not a control.

**Scoring:** Full credit names the composite key, explains why identity beats value, and specifies a safe fallback.

## 15. Quick Reference Card

| Field | Decision card |
|---|---|
| Unit | one scored order decision event in the high-risk segment |
| Grain trap | per order, not per decision event, silently inflates the denominator |
| Ordering vs magnitude | ranking survives any monotone map; money does not |
| Predicted mass | 6,640 asserted vs 3,120 observed, factor 2.128 |
| ECE, 10 equal bins | 0.088; equals the signed bias only because every gap has one sign |
| Brier | 0.07062 raw; 0.05830 recalibrated in-sample; floor 0.07192 |
| Reliability share | REL/RES = 0.905, so 90.5 percent of resolution is cancelled |
| Cost threshold | p* = C_FP/C_FN = 26/420 = 6.19 percent |
| Capacity point | 120 reviews/day, precision 0.360, recall 0.554 |
| Shadow price | 0.16 probability units = 67.20 USD, 2.58x the accounting cost |
| Economic optimum | 300 reviews/day, net 19,080 USD/day vs 15,024 at capacity |
| Claim bridge | 2,604,000 -> 1,263,360 -> 252,672 -> 725,760 -> 600,960 |
| Invariant | one registry record per (score version, calibration version) |
| Sort key | lexicographic on (calibrated, raw); never calibrated alone |
| Fallback | constant matured prevalence, flagged; never the raw score |
| Unresolved | bin 7 at 0.055, Wilson [0.0484, 0.0625] straddles 6.19 percent |
| Evidence | ILLUSTRATIVE arithmetic, fully recomputed; no deployment measured |
