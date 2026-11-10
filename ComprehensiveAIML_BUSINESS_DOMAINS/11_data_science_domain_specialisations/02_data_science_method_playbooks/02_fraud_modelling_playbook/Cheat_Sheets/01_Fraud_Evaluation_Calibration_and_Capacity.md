---
collection: "fraud_modelling_playbook"
file_id: "01"
topic: "Fraud evaluation, calibration, and capacity-constrained decisions"
level: 3
status: "draft"
version: "1.0"
created: "2026-11-07"
reviewed: null
review_trigger: "Changes to label maturity, action policy, reviewer capacity, or evaluation population"
notation_style: "plain-text"
prerequisites: ["Binary classification, conditional probability, expected value, and basic SQL"]
pattern_tags: ["evaluation", "calibration", "review-capacity"]
source_status: "OPEN"
owner_role: "Fraud analytics lead"
source_index: "INDEX.md"
---
# Fraud Evaluation, Calibration, and Capacity-Constrained Decisions

## 1. File Header

> **Purpose:** Translate point-in-time fraud scores into defensible evaluation and review decisions.
> **Prerequisites:** Binary classification, conditional probability, expected value and basic SQL; see [scikit-learn evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html).
> **Key outcome:** Given a mature evaluation cohort, calibrated risk, action effects and capacity, calculate and defend an operational policy.
> **This file does NOT cover:** Model training architecture, legal advice, or AML reporting obligations.
> **Level:** 3 — advanced / production.
> **Version / reviewed:** 1.0 / not reviewed; editorial status draft.
> **Evidence:** DERIVED for stated mathematics, ILLUSTRATIVE for all constructed inputs; empirical production outcomes are OPEN.
> **Retrieval cue:** If precision looks good while losses rise, start with Sections 3, 5 and 8.

A payment screening team must decide which events receive scarce investigation.
A score is not itself an intervention and an observed label is not automatically ground truth.
We distinguish ranking, probability estimation and action value throughout.
The source scope is the [fraud modelling crosswalk](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-methods-02) and [directory inventory](../../../../../Templates/directory.txt).
The crosswalk lists evaluation metrics, imbalanced learning and production monitoring as planned topics.
The examples below do not describe an actual institution, incident or outcome.
**Takeaway:** Evaluate the decision population and its available actions, not merely the classifier.

## 2. Concept Map / Mental Model

```text
Event at decision time -> eligible cohort -> historical score and action snapshot
                                  |                    |
                                  v                    v
                        maturity / label audit    intervention and feedback
                                  |                    |
                                  +-------> evaluation cohort
                                                  |
                          +-----------------------+-------------------+
                          v                       v                   v
                     ranking metrics       calibration         action effects
                          |                       |                   |
                          +-----------------------+-------------------+
                                                  v
                                 capacity-constrained policy selection
                                                  |
                                                  v
                                  temporal holdout and controlled rollout
                                                  |
                                                  v
                                   monitoring, audit, appeal, revision
```
The event cohort is defined before the outcome is known.
Maturity determines whether the outcome is observable by the analysis cutoff.
Selection and intervention determine whether observed outcomes represent untreated risk.
The score ranking answers an ordering question.
Calibration answers a probability question conditional on the score.
Utility requires the consequences of each available action.
Capacity converts individual choices into a coupled allocation problem.
An experiment can test incremental policy effects when ethical and operational constraints permit.
**Takeaway:** Maturity, selection and action effects intervene between a score and a valid policy estimate.

## 3. Foundational Definitions

The table of operational definitions below uses transactions as the unit unless specified otherwise.
A case, customer or account can contain multiple transactions, so changing grain changes denominators.
The [collection index](INDEX.md) records the shared symbols.

### Decision cohort

- **Definition:** All transactions eligible at the same decision point and inside a specified event-time window.
- **Type and domain:** event set; fixed by eligibility.
- **Example:** A month of authorization attempts.
- **Nonexample:** Only investigated transactions.
- **Operational check:** Freeze eligibility and retain events with missing scores.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Outcome and label

- **Definition:** Outcome Y is the target fraud event; observed label L is the recorded evidence about it.
- **Type and domain:** binary variables 0 or 1.
- **Example:** Confirmed fraud after a defined horizon.
- **Nonexample:** A review flag treated as confirmed fraud.
- **Operational check:** Store label source, certainty, availability time and revision history.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Maturity horizon

- **Definition:** The elapsed time after event time at which label ascertainment is considered adequate.
- **Type and domain:** duration, nonnegative.
- **Example:** A specified 90-day window.
- **Nonexample:** Days since data extraction without event timestamps.
- **Operational check:** Estimate delay distribution and compare cohorts at equal age.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Selection indicator

- **Definition:** R indicates whether the specified outcome is observed under the sampling and review process.
- **Type and domain:** binary 0 or 1.
- **Example:** Random audit inclusion.
- **Nonexample:** Whether the model score exceeds a threshold.
- **Operational check:** Log inclusion probabilities and preserve a nonzero audit probability.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Risk score

- **Definition:** S is a scalar used for ordering or probability estimation.
- **Type and domain:** real scalar per eligible event.
- **Example:** A classifier output of 0.3.
- **Nonexample:** A guaranteed 30% risk without validation.
- **Operational check:** Persist model version, timestamp, missingness and score precision.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Calibration

- **Definition:** Among events assigned probability p, the event frequency is p in the stated population.
- **Type and domain:** conditional probability, dimensionless.
- **Example:** Among a large representative set at 0.2, approximately one fifth positive.
- **Nonexample:** High discrimination or a single correct overall rate.
- **Operational check:** Inspect reliability by time and operationally meaningful slices.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### True positive

- **Definition:** TP counts positive outcomes selected by the policy.
- **Type and domain:** nonnegative event count.
- **Example:** Fraudulent event sent to review.
- **Nonexample:** Loss prevented by the review.
- **Operational check:** Link prediction to mature, appropriately observed outcomes.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### False positive

- **Definition:** FP counts negative outcomes selected by the policy.
- **Type and domain:** nonnegative event count.
- **Example:** Legitimate event sent to review.
- **Nonexample:** Every investigation that did not recover money.
- **Operational check:** Audit labels and distinguish review from customer decline.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### False negative

- **Definition:** FN counts positive outcomes not selected by the policy.
- **Type and domain:** nonnegative event count.
- **Example:** Mature fraud below the policy cutoff.
- **Nonexample:** All reported losses regardless of cohort.
- **Operational check:** Use unbiased ascertainment or qualify the estimate as selected.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### True negative

- **Definition:** TN counts negative outcomes not selected.
- **Type and domain:** nonnegative event count.
- **Example:** Legitimate unreviewed event.
- **Nonexample:** Every event without an investigation record.
- **Operational check:** Account for unknown labels rather than defaulting them to negative.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Prevalence

- **Definition:** pi is the fraction or probability of positive outcomes in the evaluation population.
- **Type and domain:** dimensionless in [0,1].
- **Example:** 100 positives among 10000 events gives 0.01.
- **Nonexample:** Precision among alerts.
- **Operational check:** Report the population, time window and label definition.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Recall

- **Definition:** Sensitivity or true-positive rate is TP divided by TP plus FN.
- **Type and domain:** dimensionless when positives exist.
- **Example:** 80 captured among 100 known frauds.
- **Nonexample:** Fraction of reviewed alerts that are fraud.
- **Operational check:** Report the positive denominator and label coverage.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Precision

- **Definition:** Positive predictive value is TP divided by TP plus FP.
- **Type and domain:** dimensionless when selected events exist.
- **Example:** 80 frauds among 278 reviewed.
- **Nonexample:** Fraud capture as a fraction of all fraud.
- **Operational check:** Report alert count, cohort and uncertainty.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### False-positive rate

- **Definition:** FPR is FP divided by FP plus TN.
- **Type and domain:** dimensionless when negatives exist.
- **Example:** 198 flagged among 9900 negatives.
- **Nonexample:** False discoveries among the selected set.
- **Operational check:** Use the negative denominator from the same population.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Capacity

- **Definition:** K is the maximum number of reviews deliverable over a specified period.
- **Type and domain:** integer cases per period.
- **Example:** 200 reviews per day.
- **Nonexample:** A score threshold or a queue arrival rate.
- **Operational check:** Adjust for staffing, working time, priority and review duration.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Action benefit

- **Definition:** Delta_i is expected utility of review minus the utility of the alternative.
- **Type and domain:** currency per event or explicitly stated utility.
- **Example:** Avoided expected loss minus review and friction costs.
- **Nonexample:** The raw risk score.
- **Operational check:** Separate probability, intervention effect, loss severity and cost.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Calibration sample

- **Definition:** A held-out dataset used to fit or assess mapping from score to probability.
- **Type and domain:** events with outcome and score.
- **Example:** Later out-of-training calibration cohort.
- **Nonexample:** The model-training data.
- **Operational check:** Keep policy selection and final evaluation separate.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Temporal holdout

- **Definition:** A later event-time cohort excluded from model and policy selection.
- **Type and domain:** time-indexed set of events.
- **Example:** June evaluation after training through April.
- **Nonexample:** Random rows from the same day as training.
- **Operational check:** Apply point-in-time feature joins and wait for label maturity.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Counterfactual outcome

- **Definition:** Y(a) is the outcome that would occur for an event under action a.
- **Type and domain:** potential outcome under a defined action.
- **Example:** Loss under review compared with loss under no review.
- **Nonexample:** The observed outcome under both actions.
- **Operational check:** Identify effects using a defensible design, not prediction alone.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Tie policy

- **Definition:** A reproducible rule for events with equal priority at the capacity boundary.
- **Type and domain:** ordering rule on eligible events.
- **Example:** Stable hashed ID lottery within a score tie.
- **Nonexample:** Rounding ties away after selection.
- **Operational check:** Log eligibility, random seed and inclusion probabilities.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Uncertainty interval

- **Definition:** A range produced by a specified uncertainty procedure with stated coverage interpretation.
- **Type and domain:** same units as estimated metric.
- **Example:** A cluster-bootstrap interval for review precision.
- **Nonexample:** Guaranteed interval for every future day.
- **Operational check:** Resample independent units or time blocks appropriate to dependence.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

### Support and positivity

- **Definition:** Each relevant group has a nonzero probability of receiving the action or observation needed for comparison.
- **Type and domain:** probability in (0,1] for relevant strata.
- **Example:** A random audit samples each eligible score band.
- **Nonexample:** Observing only high-score reviews.
- **Operational check:** Restrict claims to supported regions or collect new evidence.
- **Interpretation:** Apply this definition to the stated cohort, not an implicitly different population.

**Takeaway:** A metric has meaning only after its event grain, observation process and denominator are fixed.

## 4. Core Formulas with Full Derivation

All formulas below are DERIVED from stated assumptions.
Numerical inputs are ILLUSTRATIVE, not operational measurements.
A rate is dimensionless; a cost is in the stated currency per decision.

### 4.1 Simplest / Base Case

**Confusion-matrix rates [E1].**
Validity: the same eligible cohort has complete, correct binary labels and a fixed decision rule.
Derivation: a selected event is either positive or negative; a positive event is either selected or not.
Dividing the appropriate count by the corresponding partition total gives the conditional fractions.
```text
FORMULA: precision = TP / (TP + FP); recall = TP / (TP + FN)
WHERE: TP, FP and FN are event counts defined in Section 3.
CONSTRAINT: Each respective denominator must be strictly positive.
NUMERICAL EXAMPLE: TP=80, FP=198, FN=20 gives precision=80/278=0.28777;
                   recall=80/100=0.80.
INTERPRETATION: About 28.8% of reviewed events are positive and 80% of positives are reviewed.
```
Boundary: with no selected events, precision is undefined, not automatically zero or one.
Boundary: with no positives, recall is undefined; do not interpret a library default as an estimate.

**Prevalence conversion [E2].**
Validity: the class-conditional TPR and FPR apply to the target population with prevalence pi.
Derivation: by the law of total probability, selected mass is pi*TPR+(1-pi)*FPR.
The positive selected mass is pi*TPR; their ratio gives positive predictive value.
```text
FORMULA: PPV = pi*TPR / (pi*TPR + (1-pi)*FPR)
WHERE: pi=P(Y=1); TPR=P(selected|Y=1); FPR=P(selected|Y=0).
CONSTRAINT: Selected probability is positive; rates are target-population rates.
NUMERICAL EXAMPLE: pi=0.01, TPR=0.8, FPR=0.02;
                   PPV=0.008/(0.008+0.0198)=0.28777.
INTERPRETATION: Strong sensitivity can coexist with many false alerts at low prevalence.
```
Boundary: at pi=0 and FPR>0, PPV is zero; an absent denominator is undefined.
A shift in prevalence need not leave class-conditional rates unchanged.

### 4.2 General Case

**Expected incremental review benefit [E3].**
Validity: the probability and conditional action outcomes are applicable to each eligible event.
Define p_i=P(Y_i(no review)=1|information at decision), a counterfactual risk.
Let L_i be expected loss conditional on fraud without review, and r_i the fraction of that loss review prevents.
Let c_i be review cost and h_i expected incremental customer or operational harm in currency.
Derivation: conditional on fraud, expected avoided loss is L_i*r_i; average by p_i and subtract costs.
```text
FORMULA: Delta_i = p_i * L_i * r_i - c_i - h_i
WHERE: Delta_i,c_i,h_i,L_i are currency/event; p_i,r_i are dimensionless.
CONSTRAINT: The defined action has estimated conditional effects; r_i in [0,1]; additive harms.
NUMERICAL EXAMPLE: p_i=0.10, L_i=200, r_i=0.5, c_i=4, h_i=1;
                   Delta_i=0.10*200*0.5-4-1=5 currency units/event.
INTERPRETATION: Review has positive expected incremental value under these assumptions.
```
Boundary: if review effectiveness is zero, predicted risk alone cannot justify this benefit model.
If h_i differs by outcome or review causes additional fraud loss, use conditional expected utilities directly.
A historical post-intervention label is not automatically p_i for the no-review outcome.

**Unconstrained action cutoff [E4].**
Validity: L*r is positive and constant across events, costs c and h are constant, and actions are independent.
Derivation: review when E3 is positive; divide both sides by the positive constant L*r.
```text
FORMULA: review iff p > (c+h)/(L*r)
WHERE: p is counterfactual fraud probability; c,h,L are currency; r is fraction prevented.
CONSTRAINT: No capacity limit; nonnegative cost; positive L*r; consistent action effects.
NUMERICAL EXAMPLE: c=4, h=1, L=200, r=0.5 -> cutoff=5/100=0.05.
INTERPRETATION: Review above 5%; at equality either action has equal expected utility.
```
Boundary: if the cutoff exceeds one, no event has strictly positive estimated review benefit.
For unequal losses or effects, a single probability cutoff generally fails to maximize benefit.

**Constrained allocation [E5].**
Validity: each eligible review consumes one interchangeable slot, benefits are additive and known for this decision.
Let x_i be 1 for review and 0 otherwise; N is eligible event count and K is nonnegative integer capacity.
The feasible policy maximizes the sum of Delta_i*x_i with at most K selected events.
An exchange argument proves that replacing a selected lower-benefit event with an unselected higher-benefit event cannot reduce value.
```text
FORMULA: maximize sum_i Delta_i*x_i subject to sum_i x_i <= K and x_i in {0,1}
WHERE: i indexes N eligible events; Delta_i is currency/event; K is slots/period.
CONSTRAINT: Independent additive benefit, identical slot usage, and no minimum-fill requirement.
NUMERICAL EXAMPLE: benefits [9, 5, -1] currency and K=2 -> select 9 and 5; total=14.
INTERPRETATION: Take up to K positive benefits, ordered by benefit, not necessarily by risk.
```
Boundary: if all benefits are negative, the optimal at-most-K policy selects none.
If decisions have unequal review durations, selection is a knapsack or scheduling problem; benefit per minute is not always exact for indivisible cases.

### 4.3 Key Special Cases / Variants

**Prior shift correction [E6].**
Validity: class-conditional feature distributions remain unchanged, training posterior is calibrated, and both class priors are strictly between zero and one.
From Bayes' rule, posterior odds equal likelihood ratio times prior odds.
The likelihood ratio cancels between training and target populations under the stated prior-shift assumption.
```text
FORMULA: odds_target = odds_train * [pi_target/(1-pi_target)] / [pi_train/(1-pi_train)]
WHERE: odds=p/(1-p); pi_train and pi_target are source and target prevalence.
CONSTRAINT: Stable class-conditional distributions and calibrated source posterior.
NUMERICAL EXAMPLE: p_train=0.5, pi_train=0.5, pi_target=0.01;
                   odds_target=1*(0.01/0.99)/1=1/99; p_target=0.01.
INTERPRETATION: A balanced-sample posterior is not automatically a deployment probability.
```
Boundary: features can drift within a class, breaking the correction even if prevalence is known.
The training prior must describe the sampling scheme producing the posterior, not necessarily the raw historical prior.

**Inverse-probability audit estimate [E7].**
Validity: every target event has known strictly positive observation probability q_i, and its label is observed when sampled.
For a fixed finite cohort, inclusion indicator R_i has expectation q_i.
Therefore the expected contribution R_i*Y_i/q_i equals Y_i; summing gives an unbiased total.
```text
FORMULA: estimated fraud total = sum_i R_i*Y_i/q_i
WHERE: R_i is audit inclusion, Y_i is the observed binary outcome, q_i is inclusion probability.
CONSTRAINT: q_i>0 for all target events; correct recorded probabilities and ascertainment.
NUMERICAL EXAMPLE: two sampled positives each have q_i=0.1 -> estimated total=20 events.
INTERPRETATION: The estimate scales observed positives to the target cohort under the sampling design.
```
Boundary: a ratio of estimated totals is not generally exactly unbiased, and small q_i can produce high variance.
If q_i=0, the corresponding target region is not identified by this estimator.

| Variant | Change | Use when | Avoid when | Reference |
|---|---|---|---|---|
| Fixed risk cutoff | Applies E4 to every event | Constant benefit parameters and enough capacity | Variable amounts or hard capacity | E4 |
| Top K risk | Orders probability only | Benefits increase identically with risk | Heterogeneous losses or effects | E5 |
| Top K benefit | Orders incremental value | Equal slot cost and reliable effect estimates | Unequal duration or interacting cases | E5 |
| Duration-aware allocation | Adds time consumption | Review durations differ materially | Duration estimates or feasibility are unavailable | E5 boundary |
| Prior correction | Adjusts posterior odds | Credible pure prior shift | Class-conditional drift | E6 |
| Random audit weighting | Accounts for unequal inclusion | Known positive sampling probability | Missing support or unknown selection | E7 |

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

At lower prevalence, fixed nonzero FPR can dominate the alert queue.
At K=0, no event is reviewed and precision has no denominator.
At K greater than the number of positive-benefit cases, E5 does not require filling the unused slots.
At probability zero or one, odds require limiting treatment instead of direct division.
At extreme sample weights, a few audited events dominate the estimated total.
A rank-preserving recalibration preserves top-K membership in the absence of newly introduced ties.
It can nevertheless change fixed-probability threshold decisions and expected-value calculations.
If two transactions have the same risk but different loss exposure, their optimal review priorities can differ.
If an investigator must review all alerts, the model must impose an equality constraint rather than E5's upper bound.
**Takeaway:** An optimal threshold depends on an identified action effect and feasible capacity, not on discrimination alone.

## 5. Worked Numerical Example

### 5.1 Base-rate exercise

Evidence: ILLUSTRATIVE inputs and DERIVED arithmetic, not a measured fraud program.
Workflow: one authorization decision per transaction; assume all outcomes are mature and correctly observed.
Population: 10,000 events, 1% fraud prevalence, 80% sensitivity and 2% false-positive rate.
There are 100 positive and 9,900 negative events.
TP = 100 * 0.80 = 80 events.
FN = 100 - 80 = 20 events.
FP = 9,900 * 0.02 = 198 events.
TN = 9,900 - 198 = 9,702 events.
Selected volume = 80 + 198 = 278 events.
Precision = 80 / 278 = 0.2877698, or about 28.78%.
Recall = 80 / 100 = 80%.
FPR = 198 / 9,900 = 2%.
Independent check: 80+20+198+9,702=10,000 events.
Independent check: E2 gives 0.008/0.0278 = 0.2877698.
The 278 selected events exceed a daily 200-slot review limit if all belong to that day.
The confusion matrix alone does not reveal which 200 should be reviewed.
A score distribution and tie rule are needed to compute recall at 200.
Wrong path: counting the remaining 78 alerts as false positives without their labels invents their outcomes.
Wrong path: equating 80 captured frauds with 80 prevented losses assumes every review succeeds.

### 5.2 Cost-sensitive allocation exercise

All five candidates occur in the same period and require one review slot each.
Assume a correctly estimated counterfactual fraud probability and known review efficacy.
L is conditional loss, r is the prevented fraction, and c+h is combined incremental cost.
The values are hypothetical currency units per event.

| Candidate | p | L | r | c+h | E3 benefit |
|---|---:|---:|---:|---:|---:|
| A | 0.40 | 100 | 0.50 | 5 | 15 |
| B | 0.20 | 300 | 0.50 | 5 | 25 |
| C | 0.10 | 500 | 0.50 | 5 | 20 |
| D | 0.50 | 20 | 0.50 | 6 | -1 |
| E | 0.05 | 400 | 0.50 | 5 | 5 |

For A: 0.40*100*0.50-5=15.
For B: 0.20*300*0.50-5=25.
For C: 0.10*500*0.50-5=20.
For D: 0.50*20*0.50-6=-1.
For E: 0.05*400*0.50-5=5.
With K=2, E5 selects B and C for 45 expected currency units of incremental benefit.
Risk-only top two would select D and A for 14, a difference of 31.
Independent check: the remaining benefits are 15, 5 and -1, all below the selected two.
If K=3, add A and total 60; the incremental value of the third slot is 15.
If K=5 with an at-most constraint, select B, C, A and E; leave D unreviewed.
The resulting total is 65, not 64: an optional negative-benefit review is omitted.
If C's prevented fraction falls from 0.50 to 0.10, its benefit falls to zero.
At K=2 with other assumptions unchanged, B and A then yield 40.
If effectiveness is uncertain, this ranking is only as reliable as the action-effect estimates.
The calculation does not identify those effects from historical scores.

### 5.3 Calibrated bin exercise

Consider a held-out audit containing 100 mature events all assigned probability 0.20.
Suppose 15 are positive and 85 are negative; this is illustrative, not a real sample.
Observed event rate is 15/100=0.15 versus predicted 0.20.
The signed observed-minus-predicted gap is -0.05, or -5 percentage points.
The aggregate expected count is 100*0.20=20 versus 15 observed.
This bin alone does not establish persistent overprediction because the observed count is noisy.
Under independent Bernoulli sampling with common probability 0.20, count standard deviation is sqrt(100*0.2*0.8)=4.
The 5-event difference is 1.25 such standard deviations; this is not a calibrated significance test.
Account clustering, varying probabilities and policy selection invalidate that simple standard deviation.
A single bin also cannot establish calibration at other scores or in other groups.

### 5.4 Capacity and time-scale exercise

Assume 10 reviewers work 6 productive hours per day each, with 15 minutes per case.
Available time is 10*6*60=3,600 reviewer-minutes per day.
Nominal capacity is 3,600/15=240 cases per day.
Independent check: each reviewer can complete 24 cases at this fixed average duration, and 10*24=240.
If only 80% of productive minutes remain after unplanned interruptions, capacity is 192 cases per day.
This is a fluid arithmetic estimate, not a staffing guarantee or an observed service rate.
Variability, skill restrictions, shifts, urgent escalations and rework need measured queue data.
A 278-case daily arrival at 192 completions adds 86 cases per day under the simplified constant-rate assumptions.
If the rate imbalance persists, a finite queue cannot resolve it.
A smaller batch may meet capacity but also change recall and loss, which must be re-estimated.
**Takeaway:** A numerical threshold decision requires a complete population, capacity period and action-effect model.

## 6. Visual / Geometric Intuition

```text
Schematic priority ordering; horizontal position is order, not score distance.
Higher expected incremental benefit                            lower
B:25  --->  C:20  | K=2 boundary |  A:15  --->  E:5  --->  D:-1
                  ^
                  last available equal-duration review slot
```
Caption: for the Section 5.2 inputs, B and C are selected at K=2.
The zero-benefit boundary is distinct from the capacity boundary.
When capacity is tight, a positive-benefit event can remain unreviewed.
When capacity is ample, negative-benefit events remain unreviewed under an at-most constraint.
A risk-only line would order D first and would not follow this picture.

```text
Reliability diagram, schematic only
observed fraud fraction
1.0 |                              / ideal diagonal
    |                         .   /
    |                    .   /
    |               .   /
    |          .   /
0.0 +------------------------------ predicted probability
    0.0                            1.0
```
A point below the ideal diagonal indicates lower observed frequency than predicted for that bin.
Bin counts and uncertainty must accompany points; the dots are not measured data.
A smooth-looking diagonal in large bins can hide offsetting subgroup errors.
**Takeaway:** A capacity cutoff is a boundary on ordered action value, while a calibration diagram compares probabilities with frequencies.

## 7. Comparison / Selection Table

| Method | Preconditions | Useful result | Cost / limitation | Avoid when | Evidence |
|---|---|---|---|---|---|
| Existing fixed rule | Stable rule and logged eligibility | Operational baseline | May miss emerging patterns | Rule history or population is unavailable | HEURISTIC baseline |
| ROC AUC | Mature outcomes and comparable cohort | Pairwise ranking across thresholds | May obscure low-FPR operating behavior | Used alone to pick a queue policy | DERIVED |
| Precision-recall curve | Mature outcomes; positive class defined | Precision across recall levels | Sensitive to prevalence | Comparing different populations without context | DERIVED |
| Reliability bins | Representative labeled data | Conditional score-frequency comparison | Sparse bins and binning hide detail | Outcomes are selected without correction | HEURISTIC diagnostic |
| Platt-style logistic calibration | Held-out calibration data | Parametric score-to-probability map | Model form may be misspecified | Severe nonmonotonic relationship | SOURCED [S2] |
| Isotonic calibration | Held-out data and enough support | Flexible monotone mapping | Ties and overfit with small samples | Small or highly shifted calibration sample | SOURCED [S2] |
| Capacity-constrained benefit | Reliable risk, effect, cost and feasibility | Highest additive expected benefit under E5 | More inputs and estimation risk | Action effects or support are absent | DERIVED |

Selection order:
1. Reject any analysis that relies on leaked features or unsupported outcome labels.
2. Define the decision and compare with the currently deployed rule on the same eligible cohort.
3. If outcomes are selectively observed, design an audit or a defensible identification strategy before declaring population recall.
4. If probability estimates are needed, calibrate on a representative held-out period and validate separately.
5. If action effects and costs are credible, use E5 with the actual capacity and constraints.
6. If effects are unknown, report ranking metrics and run a bounded evidence-gathering policy rather than asserting utility.
7. If constraints conflict, safety and legal requirements are hard constraints; escalate rather than silently violating them.
8. Evaluate with a later mature cohort, uncertainty and a comparable current-policy baseline.

Benchmark evidence for these alternatives in this workload is NOT MEASURED.
A valid comparison uses the same event cohort, score time, label horizon, review capacity, intervention policy and currency definition.
Report sample size, clusters, time window, software version, baseline, uncertainty and guardrail definitions.
Do not treat model or calibration selection on the final holdout as independent validation.
**Takeaway:** Select an evaluation method for the decision question and reject it when its observation assumptions fail.

## 8. Failure Modes & Edge Cases

The following diagnostic cases are HYPOTHETICAL teaching scenarios, not observed incidents.
The proposed checks discriminate mechanisms; none alone proves a causal explanation.

### F-01 — Immature labels

- **Trigger:** Recent cohorts show a lower fraud rate.
- **Competing explanations:** Genuine fraud reduction versus delayed chargebacks.
- **Discriminating check:** Plot positive rate by event age and label source.
- **Expected signal:** Rates rising with cohort age support delay.
- **Interpretation:** Cohort-age growth makes maturity mismatch the leading explanation; it does not estimate fraud that never becomes visible.
- **Immediate response:** Hold comparisons until comparable maturity; mark unresolved outcomes unknown.
- **Verification:** Recompute same-age cohorts after the horizon.
- **Limitation:** A stable mature cohort can still have unreported fraud.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-02 — Selective review

- **Trigger:** Precision is high only in reviewed events.
- **Competing explanations:** Strong model versus selective ascertainment.
- **Discriminating check:** Compare reviewed events with randomly audited nonreviews by score band.
- **Expected signal:** Positive findings in the audit indicate missing support in ordinary labels.
- **Interpretation:** Audit discoveries quantify ordinary-label blind spots only to the extent the audit design covers the target population.
- **Immediate response:** Maintain a randomized audit with logged inclusion probabilities.
- **Verification:** Check weighted estimates and audit coverage.
- **Limitation:** Review effects and imperfect adjudication can remain.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-03 — Post-action outcome

- **Trigger:** Declined events show no realized chargeback.
- **Competing explanations:** Intervention prevented losses versus no underlying fraud.
- **Discriminating check:** Compare randomized or otherwise justified action alternatives with comparable follow-up.
- **Expected signal:** A difference in outcome under comparable actions supports an action effect.
- **Interpretation:** A controlled action contrast supports incremental value, whereas a score/outcome association does not.
- **Immediate response:** Define counterfactual estimand and collect intervention evidence.
- **Verification:** Audit treatment assignment and outcomes.
- **Limitation:** An unobserved counterfactual is never individually verified.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-04 — Leakage in feature join

- **Trigger:** Offline ranking is much better than live ranking.
- **Competing explanations:** Future data leakage versus genuine distribution shift.
- **Discriminating check:** Replay feature timestamps relative to the decision timestamp.
- **Expected signal:** Features first available after scoring support leakage.
- **Interpretation:** Post-decision availability invalidates the evaluation even if the feature is predictive in retrospect.
- **Immediate response:** Rebuild snapshots with availability-time constraints.
- **Verification:** Evaluate a later untouched cohort.
- **Limitation:** A correct timestamp does not prove every upstream feature is causal.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-05 — Prevalence shift

- **Trigger:** Precision falls while class-conditional rates appear stable.
- **Competing explanations:** Lower base rate versus increased FPR.
- **Discriminating check:** Compare mature prevalence, TPR and FPR at the same policy.
- **Expected signal:** Lower prevalence with stable rates predicts lower PPV by E2.
- **Interpretation:** The direction follows the stated base-rate relationship; verify that operating conditions have not also changed.
- **Immediate response:** Reforecast workload and review value for the new cohort.
- **Verification:** Compare predicted with realized mature alert composition.
- **Limitation:** Class-conditional stability itself is an assumption to test.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-06 — Within-class drift

- **Trigger:** Prior correction still miscalibrates.
- **Competing explanations:** Pure prior shift versus changing fraud behavior.
- **Discriminating check:** Compare feature and score distributions separately within mature classes.
- **Expected signal:** Material within-class changes contradict pure prior shift.
- **Interpretation:** Within-class movement points beyond a simple prevalence shift and makes prior-only correction unsafe.
- **Immediate response:** Refit or recalibrate with representative recent data.
- **Verification:** Validate on later held-out cohorts.
- **Limitation:** Small fraud samples may not reveal a real change.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-07 — Capacity overload

- **Trigger:** Backlog grows despite a stable threshold.
- **Competing explanations:** Arrival surge versus slower case handling.
- **Discriminating check:** Separate incoming alerts, completions, available reviewer minutes and age.
- **Expected signal:** Arrivals exceeding completions explain positive backlog growth.
- **Interpretation:** Net positive arrivals identify a capacity imbalance; they do not identify whether staffing, routing, or case duration is responsible.
- **Immediate response:** Apply approved prioritization and protect urgent queues.
- **Verification:** Track arrivals, completion and oldest case age.
- **Limitation:** Average rates conceal bursts and deadline violations.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-08 — Tied cutoff

- **Trigger:** Daily selected volume exceeds K.
- **Competing explanations:** Threshold ties versus duplicate events.
- **Discriminating check:** Count scores equal to the boundary and deduplicate event IDs.
- **Expected signal:** A large boundary mass or duplicate IDs distinguishes the causes.
- **Interpretation:** Boundary concentration implicates selection mechanics; distinguish an intentional policy tie from accidental duplication.
- **Immediate response:** Apply a reproducible tie lottery or deterministic secondary key.
- **Verification:** Replay selection and reconcile exactly K unique events.
- **Limitation:** Tie policy may create unequal access or missed priority.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-09 — No alerts

- **Trigger:** Precision dashboard reads 100 percent.
- **Competing explanations:** Zero denominator default versus genuine perfect precision.
- **Discriminating check:** Inspect selected count and metric null handling.
- **Expected signal:** Zero selected events makes precision undefined.
- **Interpretation:** An empty selected set makes the conditional metric undefined, rather than evidence of perfect or failed precision.
- **Immediate response:** Display undefined and alert on unexpected empty queues.
- **Verification:** Test a zero-selection cohort.
- **Limitation:** A deliberate no-review policy can be appropriate.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-10 — No positives

- **Trigger:** Recall displays zero for a tiny segment.
- **Competing explanations:** No observed positive denominator versus missed positive events.
- **Discriminating check:** Show TP+FN and label maturity for the segment.
- **Expected signal:** A zero positive denominator makes recall undefined.
- **Interpretation:** No mature positives means sensitivity cannot be learned for that cohort; it is not zero recall.
- **Immediate response:** Aggregate cautiously or collect more audited outcomes.
- **Verification:** Check that dashboard labels denominator and uncertainty.
- **Limitation:** Combining segments can hide localized harm.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-11 — Calibration bin masking

- **Trigger:** Overall reliability looks acceptable but a segment complains.
- **Competing explanations:** Offsetting subgroup miscalibration versus small-sample noise.
- **Discriminating check:** Compare calibration by prespecified group with counts and intervals.
- **Expected signal:** Opposite signed subgroup gaps can cancel in aggregate.
- **Interpretation:** Offset subgroup errors can conceal material slice harm, so the aggregate is insufficient for a calibration claim.
- **Immediate response:** Investigate data and recalibrate only with adequate support.
- **Verification:** Evaluate subgroup calibration on independent later data.
- **Limitation:** Subgroup estimates can be unstable and sensitive.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-12 — Score rounding

- **Trigger:** Top-K membership changes after a calibration update.
- **Competing explanations:** Real ordering change versus increased ties.
- **Discriminating check:** Compare full-precision rank and rounded displayed values.
- **Expected signal:** Identical raw ordering with extra rounded ties points to quantization.
- **Interpretation:** Quantization can change who receives the last slots without changing the underlying score order.
- **Immediate response:** Select using documented full precision and explicit tie rule.
- **Verification:** Replay the same cohort with pinned versions.
- **Limitation:** A non-strict isotonic mapping can legitimately create ties.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-13 — Review-cost omission

- **Trigger:** Fraud capture rises while net value falls.
- **Competing explanations:** Improved capture with costly friction versus higher losses.
- **Discriminating check:** Attribute review hours and customer harm by action.
- **Expected signal:** Costs rising faster than avoided loss contradict capture-only optimization.
- **Interpretation:** Capture is a proxy; the decision changes when review, friction, and avoided-loss terms are included.
- **Immediate response:** Include incremental review and friction costs in E3.
- **Verification:** Compare total utility and guardrails against the baseline.
- **Limitation:** Intangible harm may not have an agreed monetary value.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-14 — Amount heterogeneity

- **Trigger:** Highest-risk reviews recover less expected value.
- **Competing explanations:** Different loss amounts versus miscalibrated probabilities.
- **Discriminating check:** Compare p, conditional loss and action efficacy for ranked cases.
- **Expected signal:** High-risk low-loss cases can have lower E3 benefit.
- **Interpretation:** Risk is not value: a lower-probability event can deserve priority when its credible prevented loss is larger.
- **Immediate response:** Rank by credible incremental benefit under E5 conditions.
- **Verification:** Backtest benefit estimates by exposure segment.
- **Limitation:** Expected loss is not necessarily collectible loss.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-15 — Variable handling time

- **Trigger:** A top-benefit queue misses service deadlines.
- **Competing explanations:** Long cases versus unexpected staffing loss.
- **Discriminating check:** Measure per-case handling-time distribution and available skills.
- **Expected signal:** A few long investigations consuming time support duration mismatch.
- **Interpretation:** Long cases break the equal-slot assumption; case counts cannot certify a feasible time schedule.
- **Immediate response:** Use a time-budgeted scheduling formulation.
- **Verification:** Compare workload feasibility and deadline misses.
- **Limitation:** A simple benefit-per-minute greedy rule can be suboptimal.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-16 — Feedback-loop contamination

- **Trigger:** New model appears worse on investigator labels.
- **Competing explanations:** Model error versus policy-dependent investigation.
- **Discriminating check:** Compare random audit label rates across score bands and policy periods.
- **Expected signal:** Different observation probability can explain label-rate changes.
- **Interpretation:** A label-rate shift may be observation selection rather than risk shift, requiring a shared observed population.
- **Immediate response:** Record action, observation and sampling probability.
- **Verification:** Repeat evaluation on a common audited population.
- **Limitation:** Audit coverage may still miss some fraud types.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-17 — Calibration training reuse

- **Trigger:** Reliability looks perfect in the calibration sample.
- **Competing explanations:** Overfit mapping versus genuine reliable probabilities.
- **Discriminating check:** Compare calibration fit with a later untouched validation cohort.
- **Expected signal:** Much worse held-out reliability supports overfit.
- **Interpretation:** Calibration deterioration on untouched data supports fitted-map overflexibility, subject to separating temporal drift.
- **Immediate response:** Reduce flexibility or gather more calibration data.
- **Verification:** Reassess held-out reliability and discrimination.
- **Limitation:** Temporal shift can mimic overfit.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-18 — Case-control posterior

- **Trigger:** Balanced-training output predicts far too much fraud.
- **Competing explanations:** Sampling-distorted prior versus incorrect preprocessing.
- **Discriminating check:** Compare training sampling prior, deployment prevalence and class-conditional distributions.
- **Expected signal:** A balanced training prior with low deployment prevalence signals prior mismatch.
- **Interpretation:** Training prevalence is not a deployment probability; correction remains conditional on stable class-conditional behavior.
- **Immediate response:** Apply E6 only if its class-conditional assumptions hold.
- **Verification:** Validate corrected probabilities on representative data.
- **Limitation:** Prior adjustment cannot fix arbitrary covariate or concept drift.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-19 — Low audit support

- **Trigger:** Weighted recall estimate is unstable.
- **Competing explanations:** Very small inclusion probabilities versus random outcome variability.
- **Discriminating check:** Inspect weights, effective sample concentration and unsampled bands.
- **Expected signal:** A few large weights dominating totals indicates fragility.
- **Interpretation:** Dominant inverse-probability weights signal weak support and unstable population extrapolation.
- **Immediate response:** Increase audit sampling in underrepresented regions.
- **Verification:** Report uncertainty and predeclared weight diagnostics.
- **Limitation:** Weight clipping changes the estimand and introduces bias.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-20 — Duplicate events

- **Trigger:** Alert counts inflate after a replay.
- **Competing explanations:** Repeated delivery versus genuine repeated attempts.
- **Discriminating check:** Compare immutable event IDs, authorization IDs and replay markers.
- **Expected signal:** Repeated event IDs under the same decision support duplication.
- **Interpretation:** Repeated decision IDs inflate event-level counts; confirm whether retries are distinct decision opportunities.
- **Immediate response:** Deduplicate at the correct decision grain.
- **Verification:** Reconcile unique and raw counts after replay.
- **Limitation:** Distinct retries can be legitimate separate decisions.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-21 — Label revisions

- **Trigger:** Historic metrics change without a model release.
- **Competing explanations:** Late adjudication versus pipeline corruption.
- **Discriminating check:** Inspect label version, effective time and correction history.
- **Expected signal:** A rise in late corrected labels supports revision.
- **Interpretation:** Label revisions require an explicit vintage; neither the early nor late view is automatically the sole truth.
- **Immediate response:** Use as-of snapshots and report both vintage and mature views.
- **Verification:** Recompute a frozen cohort against versioned labels.
- **Limitation:** A clean revision history does not guarantee label correctness.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-22 — Inconsistent denominators

- **Trigger:** Two teams report incompatible precision.
- **Competing explanations:** Different eligibility or action definitions versus arithmetic error.
- **Discriminating check:** Reconcile cohort IDs, time window, policy and label definition.
- **Expected signal:** Set differences isolate denominator mismatch.
- **Interpretation:** Reconciled sets localize a denominator disagreement before comparing model quality.
- **Immediate response:** Publish a versioned metric contract and shared cohort.
- **Verification:** Recompute both values from the same event set.
- **Limitation:** Common definitions still require reliable underlying data.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-23 — Temporal split contamination

- **Trigger:** Holdout results are implausibly good.
- **Competing explanations:** Shared identities or future information versus stable repeat behavior.
- **Discriminating check:** Inspect entity overlap and feature availability across the split.
- **Expected signal:** Future-derived aggregates or duplicate decisions support leakage.
- **Interpretation:** A future aggregate can leak through joins even when the raw event timestamp appears acceptable.
- **Immediate response:** Use point-in-time joins and an entity-aware sensitivity split.
- **Verification:** Evaluate both realistic future traffic and new-entity cohorts.
- **Limitation:** A strict entity split can misrepresent recurring real customers.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-24 — Threshold instability

- **Trigger:** Small daily changes cause large queue-volume swings.
- **Competing explanations:** Score mass near cutoff versus upstream distribution shift.
- **Discriminating check:** Plot score histogram around the boundary and daily counts.
- **Expected signal:** A large spike at the threshold indicates tie or quantization sensitivity.
- **Interpretation:** Threshold spikes make the marginal selected population policy-sensitive and warrant a tie-rule audit.
- **Immediate response:** Use exact top-K selection and stable tie handling when allowed.
- **Verification:** Monitor selection rate and threshold movement.
- **Limitation:** Top-K can force low-value cases unless positivity is enforced.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-25 — Uncertainty underestimation

- **Trigger:** Confidence interval is too narrow.
- **Competing explanations:** Correlated customer events versus independent observations.
- **Discriminating check:** Measure repeated customer or merchant contributions to outcomes.
- **Expected signal:** Strong within-entity dependence reduces effective independent information.
- **Interpretation:** Dependent entities provide fewer independent observations than raw event counts suggest.
- **Immediate response:** Bootstrap entities or appropriate time blocks.
- **Verification:** Compare interval sensitivity to block definitions.
- **Limitation:** Few independent clusters limit interval reliability.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-26 — Score missingness

- **Trigger:** Alert volume unexpectedly drops.
- **Competing explanations:** Pipeline outage versus truly lower risk.
- **Discriminating check:** Measure missing-score rate by input path and compare raw eligibility counts.
- **Expected signal:** A simultaneous score-coverage drop points to a pipeline issue.
- **Interpretation:** Concurrent coverage loss points first to pipeline availability, not necessarily risk or discrimination change.
- **Immediate response:** Apply an approved safe fallback and escalate.
- **Verification:** Reconcile scored, fallback and eligible counts.
- **Limitation:** A fallback can impose different customer harms.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-27 — Objective mismatch

- **Trigger:** ROC AUC improves but business performance worsens.
- **Competing explanations:** Improvement outside the operating region versus cost shift.
- **Discriminating check:** Compare recall and benefit at the same actual capacity.
- **Expected signal:** AUC gain with lower top-K capture indicates a local ranking loss.
- **Interpretation:** Global ranking improvement can coexist with worse operational capture at the actual review cutoff.
- **Immediate response:** Select using prespecified capacity-aligned metrics.
- **Verification:** Repeat on later comparable cohorts with uncertainty.
- **Limitation:** No single offline metric identifies prevented loss.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-28 — Delayed review

- **Trigger:** Nominal K is met but fraud losses continue.
- **Competing explanations:** Review occurs after loss deadline versus ineffective actions.
- **Discriminating check:** Measure event-to-review completion relative to actionable deadline.
- **Expected signal:** Many reviews after the deadline support timing mismatch.
- **Interpretation:** A late decision cannot prevent a time-sensitive loss; latency is part of the intervention definition.
- **Immediate response:** Prioritize actionable deadlines or redesign intervention timing.
- **Verification:** Track on-time completion and experimental loss effect.
- **Limitation:** A fast review alone does not establish prevention.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-29 — Model-version mismatch

- **Trigger:** Calibrated probabilities suddenly shift.
- **Competing explanations:** Wrong calibrator paired with a score version versus population change.
- **Discriminating check:** Join scored events to exact model and calibrator artifact IDs.
- **Expected signal:** Unexpected version pairs indicate deployment mismatch.
- **Interpretation:** Unexpected artifact pairs isolate compatibility risk before attributing a change to the model itself.
- **Immediate response:** Roll back to an approved compatible pair.
- **Verification:** Enforce version compatibility at deployment.
- **Limitation:** Correct artifact pairing does not guarantee calibration.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-30 — Confounded action effect

- **Trigger:** Reviewed cases have larger losses than unreviewed cases.
- **Competing explanations:** Review harm versus investigators targeting high-risk cases.
- **Discriminating check:** Inspect pre-action risk distributions and action assignment mechanism.
- **Expected signal:** Large assignment imbalance makes raw outcome differences confounded.
- **Interpretation:** Different treatment exposure confounds raw outcomes, so the comparison needs a causal design or qualified conclusion.
- **Immediate response:** Use randomization or a justified causal design.
- **Verification:** Check covariate balance and sensitivity to unmeasured confounding.
- **Limitation:** Observational adjustment cannot guarantee identification.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-31 — Simpson reversal

- **Trigger:** Aggregate precision improves while key cohorts deteriorate.
- **Competing explanations:** Mix shift versus true within-cohort gains.
- **Discriminating check:** Standardize by a fixed cohort mix and report each prespecified slice.
- **Expected signal:** Aggregate change with declines in each slice indicates composition effects.
- **Interpretation:** Composition can reverse aggregate trends; slice-standardized reporting separates mix change from performance change.
- **Immediate response:** Monitor segment and standardized performance together.
- **Verification:** Reconcile segment weights with overall metrics.
- **Limitation:** Small slices may be too noisy for a confident conclusion.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-32 — Fraud labels as truth

- **Trigger:** Chargeback precision differs from investigator precision.
- **Competing explanations:** Different target definitions versus wrong joins.
- **Discriminating check:** Match a sample to adjudication source and define each outcome.
- **Expected signal:** Disagreements concentrated by label source support definition differences.
- **Interpretation:** Source-specific disagreement means the outcome definition, not just the model, differs across reports.
- **Immediate response:** Publish separate outcome-specific measures.
- **Verification:** Audit a blinded adjudication sample.
- **Limitation:** Even expert adjudication can disagree.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-33 — Customer friction

- **Trigger:** Avoided loss rises but complaints increase.
- **Competing explanations:** More legitimate reviews versus unrelated service incident.
- **Discriminating check:** Measure action-specific complaint and abandonment rates by cohort.
- **Expected signal:** Concentrated increases after review support a friction hypothesis.
- **Interpretation:** Concentration after review is compatible with friction but needs an appropriate comparison group before attribution.
- **Immediate response:** Set explicit guardrails and consider lower-friction action.
- **Verification:** Compare guardrails with a suitable control group.
- **Limitation:** Complaint reporting is incomplete and selective.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-34 — Data privacy leakage

- **Trigger:** Evaluation extracts expose sensitive identifiers.
- **Competing explanations:** Overbroad analyst access versus necessary linkage.
- **Discriminating check:** Review fields, permissions and access logs against the metric contract.
- **Expected signal:** Unused raw identifiers or excess access support overcollection.
- **Interpretation:** Unused identifiers create avoidable exposure; utility must be demonstrated for each retained field.
- **Immediate response:** Minimize fields and restrict access with approved controls.
- **Verification:** Review access and retention against policy.
- **Limitation:** Pseudonymization alone is not anonymity.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-35 — Optimization interaction

- **Trigger:** Two highly related alerts both consume review capacity.
- **Competing explanations:** Independent cases versus one investigation resolving both.
- **Discriminating check:** Trace case linkage and investigator disposition.
- **Expected signal:** Shared investigation outcomes show nonadditive benefits or costs.
- **Interpretation:** Linked investigations violate additive item value, so independent top-K optimization may double count benefit.
- **Immediate response:** Allocate at case level or explicitly model interactions.
- **Verification:** Compare predicted and realized workloads by linked group.
- **Limitation:** Entity resolution errors can merge unrelated customers.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-36 — Policy drift

- **Trigger:** Threshold remains fixed but review yield changes.
- **Competing explanations:** Changing incoming mix versus new reviewer policy.
- **Discriminating check:** Stratify by model version, eligibility, action and reviewer workflow.
- **Expected signal:** Change aligned to a workflow release indicates a policy component.
- **Interpretation:** A release-aligned shift identifies a policy-change candidate, not a uniquely identified policy effect.
- **Immediate response:** Version policies and evaluate on a shared frozen cohort.
- **Verification:** Reconcile policy-specific outcomes after maturity.
- **Limitation:** Several simultaneous changes may not be separable.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-37 — Capacity definition

- **Trigger:** A weekly top-K policy violates daily commitments.
- **Competing explanations:** Weekly average capacity versus daily shift constraints.
- **Discriminating check:** Compare arrivals and staffing by shift and deadline.
- **Expected signal:** Days with overload despite weekly spare capacity support granularity mismatch.
- **Interpretation:** Weekly spare capacity cannot repair intraday deadline failures; the constraint must match operational granularity.
- **Immediate response:** Allocate within daily or shift-level constraints.
- **Verification:** Simulate the historical arrival sequence with timestamps.
- **Limitation:** Retrospective simulation does not identify intervention benefit.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-38 — False certainty

- **Trigger:** A model with 80 percent recall is declared superior.
- **Competing explanations:** Sampling variation versus real improvement.
- **Discriminating check:** Compute paired differences on the same events and resample dependence units.
- **Expected signal:** An interval including meaningful losses prevents a confident superiority claim.
- **Interpretation:** A wide interval preserves plausible harm, so it cannot justify a superiority assertion at the chosen decision threshold.
- **Immediate response:** Use the prespecified uncertainty and practical-effect criteria.
- **Verification:** Evaluate a later independent cohort.
- **Limitation:** A nonsignificant difference is not proof of equivalence.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-39 — Audit nonresponse

- **Trigger:** Random audit still finds too little fraud.
- **Competing explanations:** Selection after sampling versus truly low fraud prevalence.
- **Discriminating check:** Compare completion rates and characteristics of audited cases.
- **Expected signal:** Unequal adjudication completion suggests second-stage selection.
- **Interpretation:** Unequal completion creates second-stage selection; observed adjudications need not represent reviewed cases.
- **Immediate response:** Record completion probabilities and improve follow-up.
- **Verification:** Report missingness and sensitivity bounds.
- **Limitation:** Unknown missing-not-at-random effects remain open.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

### F-40 — Threshold tuned on holdout

- **Trigger:** A threshold looks optimal but fails next month.
- **Competing explanations:** Selection overfit versus genuine regime shift.
- **Discriminating check:** Separate threshold-search period from untouched test period.
- **Expected signal:** Large development-to-test deterioration is consistent with selection overfit.
- **Interpretation:** Development-to-test decay is consistent with selection overfit, while repeated temporal splits test its persistence.
- **Immediate response:** Use nested temporal selection and final holdout.
- **Verification:** Repeat across rolling temporal splits.
- **Limitation:** Even proper separation cannot eliminate future shift.
- **Provenance:** HYPOTHETICAL; no incident or actual measurement is claimed.
- **Escalation:** If the necessary evidence is unavailable, mark the conclusion OPEN and involve the responsible owner.

The separate collection-wide Common Issues artifact is planned and has no authored issue links yet.
**Takeaway:** Diagnose the observation and policy mechanism before interpreting a changed metric.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic Complexity

Let N be events in an evaluation period, K the integer review capacity, and B the number of reliability bins.
Let I be the number of iterations used by a chosen iterative calibrator.
All bounds describe operations, not measured execution times.

| Operation | Time | Auxiliary space | I/O and assumption |
|---|---|---|---|
| Confusion counts at one threshold | O(N) | O(1) | One pass after scores and labels are joined |
| Sort for full ranked curve | O(N log N) | O(N) typical | Includes materializing sortable scores; implementation dependent |
| Select top K with heap | O(N log K) | O(K) | K>0; final ordered output can cost O(K log K) |
| Reliability bins | O(N+B) | O(B) | Fixed bins, after joining mature labels |
| Logistic calibration | O(I*N) for scalar score and fixed-dimensional solver passes | O(N) or streaming-dependent | Excludes convergence guarantees and feature preparation |
| Audit weighting | O(N) | O(1) for totals | Requires inclusion probabilities and correct label joins |
| Replay | O(N log N) if sorting is required | O(N) typical | Includes ordering but excludes upstream feature reconstruction |

Feature building, joining and storage scans can dominate the score-level computations.
A distributed sort adds network shuffle and synchronization not described by a single-machine bound.
Never present a complexity expression as a measured latency or service-level guarantee.

### 9b. Back-of-Envelope Sizing

Use the 240 nominal cases/day and 192 interruption-adjusted cases/day from Section 5.4.
Input assumptions are illustrative; software, hardware and runtime are unspecified and not measured.
The relevant bottleneck in this example is reviewer time, not computed model throughput.
A 278-case/day inflow adds 86/day at constant 192/day completions, assuming an initially empty queue and no abandonment.
After five such days, the idealized backlog is 5*86=430 cases.
Independent check: 5*278=1,390 arrivals and 5*192=960 completions; 1,390-960=430.
This backlog estimate excludes case priority, burstiness, duplicate alerts and variable duration.
If the inflow falls to 150/day, spare capacity is 192-150=42/day.
A fluid drain of 430/42 is approximately 10.24 days, assuming constant rates and all cases remain actionable.
The real estimate needs shift calendars, escalation overhead, actual durations, case expiry and staffing measurements.
For economics, use incremental costs and benefits per decision at a documented currency and date.
No currency conversion, production price or vendor charge is assumed here.
**Takeaway:** Compute pipeline complexity and investigator capacity separately; neither implies the other.

## 10. Key Theorems / Results Reference

| Result | Statement and hypotheses | Practical implication | Does not imply | Source |
|---|---|---|---|---|
| Bayes' rule | Conditional probabilities relate through joint probability when conditioning events have positive probability | Derive E2 and E6 | Prior shift holds in a real system | [S1](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35.htm) |
| Probability calibration | A calibrated predictor's conditional event frequency matches its predicted probability | Reliability checks assess probability meaning | High ranking quality or useful actions | [S2](https://scikit-learn.org/stable/modules/calibration.html) |
| Proper scoring | Brier score measures squared probability error against binary outcomes | Compare probability quality on the same target cohort | Calibration alone determines the score | [S3](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2) |
| Exchange argument | With additive values and identical unit resource, any selected lower value can be exchanged for a higher unselected value | E5 chooses the positive top K | Greedy ratio solves arbitrary time-budgeted integer allocation | E5 derivation above |
| ROC / PR relationship | Precision depends on prevalence even at fixed TPR and FPR | Compare PR only with population context | AUC alone fixes expected utility | [S4](https://doi.org/10.1145/1143844.1143874) |
| Inverse-probability total | Known positive inclusion probabilities make the weighted sampled total unbiased for a fixed finite population | Random audits can estimate missed positives | The ratio is exactly unbiased or low variance | [S5](https://doi.org/10.1080/01621459.1952.10483446) |

### Source and maintenance ledger

- **S1:** NIST/SEMATECH e-Handbook of Statistical Methods, probability rules, online edition; supports conditional-probability reasoning, not the fraud workload.
- **S2:** scikit-learn probability calibration guide, stable documentation; supports the description of calibration and its methods, not a deployment guarantee.
- **S3:** Brier, 1950, *Verification of forecasts expressed in terms of probability*; original scoring-rule source, no fraud benchmark claimed.
- **S4:** Davis and Goadrich, 2006, *The Relationship Between Precision-Recall and ROC Curves*; population dependence and curve concepts, not a claimed experiment here.
- **S5:** Horvitz and Thompson, 1952, *A Generalization of Sampling Without Replacement from a Finite Universe*; unequal-probability sampling theory, not proof of this audit's label quality.
- **Version/date:** linked historical papers carry their publication years; web documentation is unpinned and should be rechecked before publication.
- **Review status:** source and link review is OPEN; no external empirical results have been reproduced.
- **Trigger:** recheck when source versions, outcome definitions, audit design or operating policy change.
- **Application:** The [illustrative payment-card case](../../../01_production_ml_case_library/01_fraud_modelling/Case_Studies/01_real_time_payment_card_fraud_detection.md) applies these distinctions to authorization decisions; its modeled savings are not measured outcomes.
**Takeaway:** Named results transfer only with their hypotheses; no cited source certifies this illustrative policy.

## 11. Interview / Exam Questions

Each answer uses the definitions and assumptions in this sheet.
Each follow-up tests whether the candidate can identify a limitation, not just recall a formula.

### Q01 — Recall

**Question:** What is precision and why is it not recall?

**Answer:** Precision is TP/(TP+FP), the fraud fraction among selected events; recall is TP/(TP+FN), the selected fraction among positive events.

**Reasoning:** The two condition on different sets and therefore answer different operational questions.

**Follow-up:** What if there are no selected events?

**Follow-up answer:** Precision is undefined; display the denominator instead of silently assigning a score.

**Acceptable alternative:** A correct confusion matrix or equivalent conditional probability is acceptable.

**Red flag:** Saying precision is captured fraud divided by all fraud confuses it with recall.

**Scoring:** Give credit for both denominators and the undefined boundary.

**Answer support:** E1 and Section 3.

### Q02 — Recall

**Question:** What does calibrated probability mean?

**Answer:** Within the target population, conditional event frequency equals the predicted probability, subject to estimation uncertainty.

**Reasoning:** It concerns probabilities, not whether high-risk cases outrank low-risk cases.

**Follow-up:** Can a constant prevalence predictor be calibrated?

**Follow-up answer:** Yes, in the population if its constant equals prevalence, but it has no useful ranking discrimination.

**Acceptable alternative:** A reliability interpretation with correct conditioning is acceptable.

**Red flag:** Calling a high AUC proof of calibration is incorrect.

**Scoring:** Require a stated population and the distinction from ranking.

**Answer support:** Section 3 and S2.

### Q03 — Application

**Question:** Compute precision for 100 frauds, 9900 legitimate events, TPR 80% and FPR 2%.

**Answer:** TP=80 and FP=198, so precision=80/278 approximately 28.78%.

**Reasoning:** False positives arise from 9900 negatives, not 100 positives.

**Follow-up:** What is recall?

**Follow-up answer:** 80/100=80%; the 20 missed positives are false negatives.

**Acceptable alternative:** The E2 probability calculation is equivalent.

**Red flag:** Using 2% of the entire population as false positives gives the wrong denominator.

**Scoring:** Check counts, denominator and rounding.

**Answer support:** Section 5.1 and E2.

### Q04 — Application

**Question:** How should two identical-capacity slots be allocated in Section 5.2?

**Answer:** Select B and C, whose expected incremental benefits are 25 and 20, totaling 45 currency units.

**Reasoning:** E5 sorts positive incremental benefits rather than probabilities when amounts differ.

**Follow-up:** What if C effectiveness falls to 0.10?

**Follow-up answer:** C benefit becomes zero; select B and A for 25+15=40.

**Acceptable alternative:** An explicit feasible enumeration yields the same answer.

**Red flag:** Selecting D solely because its risk is 0.50 ignores its negative benefit.

**Scoring:** Require arithmetic and the identical-slot assumption.

**Answer support:** Section 5.2 and E5.

### Q05 — Judgment

**Question:** Why can precision increase after random audits are stopped even when the model is unchanged?

**Answer:** The evaluated population can become more selectively reviewed, so observed labels no longer represent all eligible transactions.

**Reasoning:** Review policy determines which outcomes are discoverable and can change the apparent positive fraction.

**Follow-up:** How can you investigate missed fraud?

**Follow-up answer:** Audit a known random sample of nonreviews, preserve positive inclusion probabilities, and use design-aware estimates.

**Acceptable alternative:** A justified causal or sampling design with support is acceptable.

**Red flag:** Treating unreviewed cases as confirmed negatives makes recall misleading.

**Scoring:** Require explicit selection and support reasoning.

**Answer support:** F-02 and E7.

### Q06 — Application

**Question:** What is the unconstrained risk cutoff if L=200, r=0.5 and combined cost is 5?

**Answer:** The cutoff is 5/(200*0.5)=0.05, with equal expected value at exactly 0.05.

**Reasoning:** Subtracting the cost from p*L*r and solving for positive benefit gives E4.

**Follow-up:** Does a 200-slot cap change the policy?

**Follow-up answer:** Yes; among positive-benefit candidates select the highest incremental benefits up to available capacity.

**Acceptable alternative:** Direct calculation of both action utilities is equivalent.

**Red flag:** Applying the same cutoff with varying loss and effectiveness is invalid.

**Scoring:** Check units, strict inequality and capacity boundary.

**Answer support:** E3 through E5.

### Q07 — Judgment

**Question:** What changes if review times vary?

**Answer:** Equal-slot top-K selection may be infeasible or suboptimal; use the relevant time and scheduling constraints.

**Reasoning:** A 30-minute case consumes twice the resource of a 15-minute case.

**Follow-up:** Is greedy benefit per minute always optimal?

**Follow-up answer:** No; indivisibility can make a combination of lower-ratio cases better.

**Acceptable alternative:** Knapsack or scheduling formulations are acceptable when they match the workflow.

**Red flag:** Claiming a ratio greedy algorithm is universally exact is a red flag.

**Scoring:** Require a feasible formulation and an explicit indivisibility limitation.

**Answer support:** E5 boundary and F-15.

### Q08 — Application

**Question:** A balanced training sample yields posterior 0.5; deployment prevalence is 1%. What is the corrected posterior under pure prior shift?

**Answer:** Training odds are 1 and target prior odds are 1/99, giving posterior 0.01.

**Reasoning:** E6 holds the class-conditional likelihood ratio fixed and replaces prior odds.

**Follow-up:** When is this correction unsafe?

**Follow-up answer:** When class-conditional score or feature distributions change, source calibration fails, or priors are wrong.

**Acceptable alternative:** A direct Bayes calculation with the same assumptions is acceptable.

**Red flag:** Claiming all kinds of drift are corrected by prior odds is wrong.

**Scoring:** Check odds conversion and assumptions.

**Answer support:** E6.

### Q09 — Judgment

**Question:** AUC rises but precision at capacity falls. Is that contradictory?

**Answer:** No; AUC averages pairwise ranking over the full score range while the queue operates at one restricted rank region.

**Reasoning:** The model can improve many nonoperational pairs while worsening the top of the list.

**Follow-up:** What must be held fixed in comparison?

**Follow-up answer:** Eligibility, evaluation window, label maturity, capacity, score availability and policy should be comparable.

**Acceptable alternative:** Recall at K or value at K can complement precision at K.

**Red flag:** Choosing solely by global AUC ignores the actual decision.

**Scoring:** Require a local-ranking explanation and a comparable baseline.

**Answer support:** Section 7 and F-27.

### Q10 — Judgment

**Question:** Why is observed loss after a decline not enough to estimate no-decline fraud risk?

**Answer:** The decline changes the outcome, so the observed post-decline loss is not the untreated counterfactual.

**Reasoning:** No single event reveals outcomes under both mutually exclusive actions.

**Follow-up:** What evidence could identify the effect?

**Follow-up answer:** A feasible randomized policy or a carefully justified observational design with support and confounding controls.

**Acceptable alternative:** Natural experiments can be acceptable if assumptions are defended.

**Red flag:** Equating zero observed loss with nonfraud is incorrect.

**Scoring:** Require an estimand, intervention and identification caveat.

**Answer support:** Section 3 and F-03.

### Q11 — Application

**Question:** A bin has 100 predictions of 0.20 and 15 positives. What can be concluded?

**Answer:** Observed frequency is 0.15, five percentage points below the predicted 0.20.

**Reasoning:** The expected count under the prediction is 20 while the observed count is 15.

**Follow-up:** Does this prove systematic overprediction?

**Follow-up answer:** No; sampling variability, dependence and selection must be assessed and other bins can differ.

**Acceptable alternative:** Reporting observed and expected counts is equivalent.

**Red flag:** Treating one bin as a definitive population calibration failure overstates evidence.

**Scoring:** Require the arithmetic and uncertainty qualification.

**Answer support:** Section 5.3.

### Q12 — Judgment

**Question:** What happens if K exceeds the number of positive-value cases?

**Answer:** Under the at-most-K formulation, leave the extra slots unused rather than take negative expected value.

**Reasoning:** The feasible set permits fewer than K selections, and adding a negative value decreases the objective.

**Follow-up:** When would filling K be appropriate?

**Follow-up answer:** Only when an explicit minimum or equality constraint, or omitted benefit, changes the decision model.

**Acceptable alternative:** An equivalent comparison of objective values is acceptable.

**Red flag:** Confusing a maximum capacity with a mandatory quota is a red flag.

**Scoring:** Require a clear constraint and explicit distinction between zero and negative value.

**Answer support:** E5 and Section 5.2.

### Q13 — Application

**Question:** What is the daily capacity of 10 reviewers with 6 productive hours and 15 minutes per case?

**Answer:** The nominal capacity is 10*6*60/15=240 cases per day.

**Reasoning:** Each reviewer contributes 360 minutes, sufficient for 24 fixed-duration reviews.

**Follow-up:** If only 80% of that time remains available?

**Follow-up answer:** The adjusted fluid estimate is 240*0.8=192 cases per day.

**Acceptable alternative:** Computing total reviewer-minutes first gives the same result.

**Red flag:** Treating this arithmetic as a measured throughput guarantee is incorrect.

**Scoring:** Check units and limitations from variable handling times.

**Answer support:** Section 5.4.

### Q14 — Judgment

**Question:** What would you do when the audit inclusion probability is zero in a score band?

**Answer:** Do not claim design-based identification for that band; collect an approved sample with positive probability or bound the claim.

**Reasoning:** E7 divides by inclusion probability, and an unobserved region has no sample support.

**Follow-up:** Can inverse weighting recover an unobserved band?

**Follow-up answer:** No; infinite or undefined weights cannot manufacture observations.

**Acceptable alternative:** Restricting the estimand to supported regions is acceptable if stated.

**Red flag:** Assigning an arbitrary huge weight to unsampled events is invalid.

**Scoring:** Require positivity, an evidence plan and a scope restriction.

**Answer support:** E7 and F-19.

### Q15 — Judgment

**Question:** How would you interpret a metric improvement with a narrow transaction-level interval when many transactions share customers?

**Answer:** The interval may understate uncertainty because transactions within the same customer are dependent.

**Reasoning:** The effective number of independent units can be much smaller than the event count.

**Follow-up:** What would you do?

**Follow-up answer:** Use customer-level resampling or appropriate time blocks and report sensitivity to the dependence structure.

**Acceptable alternative:** A justified clustered variance method is also acceptable.

**Red flag:** Treating every event as independent without examining clustering is a red flag.

**Scoring:** Require a dependence-aware method and its limitations.

**Answer support:** F-25 and Section 3.

**Takeaway:** Defensible interview answers specify the population, assumptions, arithmetic, decision and failure boundary.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Fraud evaluation, calibration and review-capacity allocation |
| Core formula | Precision = TP/(TP+FP); recall = TP/(TP+FN); benefit = p*L*r-c-h under E5 assumptions |
| Key assumption | Mature or appropriately sampled labels represent the defined cohort; action effects are identified |
| Complexity | Sort n candidates in O(n log n) time for equal-slot top-K; other constraints may require a different allocator |
| Use when | Compare review policies with a defined population, capacity, action effect and costs |
| Avoid when | Labels lack support or intervention effects are unknown but are treated as established |
| Go deeper | [Collection index](INDEX.md); [probability calibration](https://scikit-learn.org/stable/modules/calibration.html) |
| Diagnostic | Check label delay, selective review, leakage, calibration shift and customer friction |
| Evidence | Illustrative calculations; production results not measured |
