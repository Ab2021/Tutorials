---
collection: fraud-modelling-playbook
file_id: "03"
topic: Imbalanced fraud learning and thresholds
level: 3
status: draft
version: "1.0"
created: "2026-11-07"
reviewed: null
review_trigger: Change in label maturity, sampling, capacity, costs or deployed policy
notation_style: plain-text
prerequisites: ["01_Fraud_Evaluation_Calibration_and_Capacity.md", "02_Calibration_and_Probability_Scoring.md"]
source_status: OPEN
owner_role: fraud decision science
source_index: INDEX.md
---

# Imbalanced Fraud Learning and Thresholds

## 1. File Header

> **Purpose:** Evaluate rare-event scores and select a defensible action policy under sampling, delayed outcomes and finite capacity.
> **Prerequisites:** Binary outcomes, conditional probability, expected value, temporal validation, and [evaluation](01_Fraud_Evaluation_Calibration_and_Capacity.md) and [calibration](02_Calibration_and_Probability_Scoring.md).
> **Key outcome:** Given a cohort, confusion counts, costs and review limit, compute rates and choose a feasible action while stating what cannot be identified.
> **This file does NOT cover:** Detailed fraud feature engineering, legal decisions, or a universal optimal model architecture.
> **Level:** 3 — advanced / production.
> **Version / reviewed:** 1.0 / not independently reviewed.
> **Evidence:** DERIVED and ILLUSTRATIVE calculations; sources are linked below; production results NOT MEASURED.
> **Retrieval cue:** Start at Section 5 for a numerical decision or Section 8 for incident diagnosis.

The source inventory names `DS_Domain/Fraud_Modelling_Cheat_Sheets/04_Imbalanced_Learning.md` and `06_Evaluation_Metrics_Fraud.md`.
Only their names are available; their original contents were not supplied and are not represented as reviewed.
The sheet independently teaches those source topics, not a reconstruction of their text.
A score orders eligible transactions; an action changes customer experience and sometimes the outcome itself.
A threshold is a policy decision, not a property of the algorithm alone.
Every number below is constructed unless explicitly marked otherwise.
**Takeaway:** A high-ranking model needs representative outcomes, decision costs and operational capacity before it becomes a defensible policy.

## 2. Concept Map / Mental Model

```text
Event at decision time -> point-in-time features -> score/version
                                 |                    |
                                 v                    v
                       temporal train sample     rank or probability
                                 |                    |
                      sampling / class weights        v
                                 |              decision policy + capacity
                                 v                    |
                       held-out calibration      action -> outcome availability
                                 |                    |
                                 +-----------> mature cohort / audit
                                                      |
                                       rates, costs, uncertainty, feedback
```

The diagram separates learning, scoring, action, and label observation.
Training sampling changes the model's input distribution; it does not redefine production prevalence.
The policy may prevent the target event or determine whether the label can be observed.
A label arriving later is not equivalent to a negative label now.
**Takeaway:** Preserve the timeline and target population through every transformation.

## 3. Foundational Definitions

| Term | Formal meaning and units | Boundary / nonexample |
|---|---|---|
| Event | One eligible transaction at a specified decision point | A customer with several transactions is several events. |
| Y | Binary fraud outcome under a specified action and horizon | A chargeback is a proxy, not necessarily all fraud. |
| Score S | Real scalar used for ranking | S need not be a probability. |
| Probability p | Conditional probability of the specified Y given available information | A calibrated observed-action outcome need not be the no-action risk. |
| Prevalence pi | P(Y=1) in the target cohort | Training prevalence after sampling may differ. |
| Threshold t | Act when score is at least t | A top-K policy can induce a different threshold each batch. |
| TP, FP | Acted-on fraud and nonfraud counts | Require observable outcomes for the target estimand. |
| FN, TN | Unacted-on fraud and nonfraud counts | A pending outcome is neither confirmed negative nor necessarily positive. |
| TPR / recall | TP/(TP+FN) | Requires at least one positive. |
| FPR | FP/(FP+TN) | Requires at least one negative. |
| Precision / PPV | TP/(TP+FP) | Requires at least one selected event. |
| Capacity K | Available review slots in the specified time window | Slots are not interchangeable when review times differ. |
| Calibration | P(Y=1 given p approximately u) approximately u | Calibration alone does not guarantee good ranking. |
| Maturity | Sufficient observation time for the chosen outcome definition | A mature cohort may still have selectively missing labels. |
| Sampling inclusion q | Probability an eligible record enters a sample | Inverse-probability estimation needs positive known inclusion. |
| Action effect | Outcome change caused by acting versus not acting | Observed fraud risk alone does not identify preventable loss. |

Operational denominators must specify event eligibility, action, observation horizon, time zone and label version.
A model version is paired with its feature pipeline and calibration map.
Threshold ties require a deterministic documented rule or a randomized allocation.
False negatives count only against the specified actual outcome, which may be counterfactual when interventions prevent fraud.
**Takeaway:** Every metric is conditional on an explicitly defined event population and outcome.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

**E1: Confusion counts and rates.**
Validity before use: each event has one binary prediction and one known binary outcome on the same cohort.
Derivation: partition the cohort by predicted action and actual outcome; the four cells are disjoint and exhaustive.
- FORMULA: N = TP + FP + FN + TN; recall = TP/(TP+FN); FPR = FP/(FP+TN); precision = TP/(TP+FP).
- WHERE: N and the four cells are event counts; each rate is dimensionless.
- CONSTRAINT: each denominator must be nonzero; missing labels are excluded only if that produces the intended estimand.
- NUMERICAL EXAMPLE: TP=80, FP=999, FN=20, TN=98,901 gives N=100,000, recall=0.8, FPR=0.01 and precision=80/1,079≈0.074143.
- INTERPRETATION: about 7.41% of selected transactions are fraud despite 80% recall.
Boundary: precision is undefined, not zero by mathematics, when no events are selected; software conventions must be documented.
Accuracy is (TP+TN)/N=0.98981 here; an all-negative baseline achieves 0.999 and misses all fraud.

**E2: Precision from base rate.**
Validity before use: TPR and FPR correspond to the same target population and threshold as prevalence.
Derivation: Bayes' identity gives P(Y=1|selected)=P(selected|Y=1)P(Y=1)/P(selected).
The law of total probability gives the denominator as pi*TPR+(1-pi)*FPR.
- FORMULA: PPV = pi*TPR / (pi*TPR + (1-pi)*FPR).
- WHERE: pi is prevalence, TPR recall, and FPR false-positive rate.
- CONSTRAINT: selected probability must be positive; rates must transfer if used for a new population.
- NUMERICAL EXAMPLE: pi=0.001, TPR=0.8, FPR=0.01 gives 0.0008/(0.0008+0.00999)=0.074143 approximately.
- INTERPRETATION: even a low FPR can generate many false alerts when negatives dominate.
Boundary: as pi approaches zero with positive FPR and fixed TPR, precision approaches zero.

### 4.2 General Case

**E3: Expected cost threshold.**
Validity before use: calibrated probability of the same outcome, two actions, constant nonnegative error costs, no capacity constraint and no additional action effects.
Derivation: act incurs expected cost (1-p)*C_FP; do not act incurs p*C_FN.
Act when the first cost is no greater than the second; rearrange.
- FORMULA: t = C_FP/(C_FP+C_FN); act if p >= t.
- WHERE: p is fraud probability; C_FP and C_FN are currency per false positive and false negative respectively.
- CONSTRAINT: C_FP+C_FN > 0; ties may be resolved consistently; cost model must include the relevant counterfactual consequences.
- NUMERICAL EXAMPLE: C_FP=10 and C_FN=90 currency/event yield t=0.1.
- INTERPRETATION: a transaction with p=0.12 has action cost 8.8 versus no-action cost 10.8 under this model.
Boundary: customer harm, heterogeneous amounts, imperfect review and limited slots invalidate a single universal threshold.

**E4: Constrained incremental utility.**
Validity before use: incremental values are comparable and additive; one action consumes one identical slot; at most K actions can be taken.
Derivation: compare each action with no action; an exchange replacing a smaller selected positive value by a larger unselected value cannot reduce utility.
- FORMULA: maximize sum_i x_i*Delta_i subject to sum_i x_i <= K and x_i in {0,1}; choose up to K largest positive Delta_i.
- WHERE: Delta_i is expected incremental benefit in currency/event, K is slots, and x_i is the action indicator.
- CONSTRAINT: utility must reflect causal action effects, review costs and harm; no interactions or differing slot durations.
- NUMERICAL EXAMPLE: values 8, 5, -1 with K=2 select 8 and 5, yielding 13 currency.
- INTERPRETATION: ranking by raw risk is optimal only when it preserves the ordering of incremental value.
Boundary: a review lasting two hours and one lasting ten minutes require a time-budget allocation, not naive top K.

**E5: Case-control prior correction.**
Validity before use: selection depends only on Y, class-conditional feature distributions are stable, and sampled posterior is calibrated for the sampled population.
Derivation: posterior odds equal likelihood ratio times prior odds; replace sampled prior odds by target prior odds.
- FORMULA: odds_target = odds_sample * [pi_target/(1-pi_target)] / [pi_sample/(1-pi_sample)].
- WHERE: odds=p/(1-p), pi_sample and pi_target are sampled and deployment fraud prevalence.
- CONSTRAINT: both priors and sampled probabilities lie strictly between zero and one; support must overlap.
- NUMERICAL EXAMPLE: p_sample=0.5, pi_sample=0.5, pi_target=0.01 gives target odds=1/99 and target probability=0.01.
- INTERPRETATION: balanced sampling can make raw scores overstate deployment risk.
Boundary: covariate, concept or class-conditional score shift requires more than an intercept correction.

**E6: Inverse inclusion weighting.**
Validity before use: each target event has known positive label inclusion probability q_i and observed target labels are accurate.
Derivation: for inclusion indicator R_i, E[R_i/q_i]=1; linearity gives unbiased totals under the sampling design.
- FORMULA: estimated total = sum over observed i of y_i/q_i; estimate a rate using weighted numerator and denominator.
- WHERE: y_i is the event contribution, q_i is its inclusion probability, and R_i indicates observation.
- CONSTRAINT: positivity and correct inclusion probabilities; the ratio estimator is not generally exactly unbiased in finite samples.
- NUMERICAL EXAMPLE: one observed positive sampled with q=0.1 contributes 10 estimated positives.
- INTERPRETATION: a randomized audit can represent otherwise under-observed traffic, at the cost of variance.
Boundary: q=0 makes design-based recovery of that stratum impossible without additional assumptions.

### 4.3 Key Special Cases / Variants

| Variant | Change | Assumption | Use | Avoid | Reference |
|---|---|---|---|---|---|
| Fixed threshold | Constant score cutoff | Stable semantics | Predictable action rule | Variable arrival/capacity | E3 |
| Top K | Relative batch ordering | Equal unit slots | Hard batch capacity | Time-varying review duration | E4 |
| Cost-sensitive training | Changes training loss | Weights express intended optimization | Rare positive learning | Treating raw score as calibrated | E5 caution |
| Downsampling | Fewer negatives in fit | Selection known | Lower training cost | Applying sampled prevalence to deployment | E5 |
| Random audit | Sample outside queue | Known positive inclusion | Estimate broader outcomes | Extrapolating into zero-support regions | E6 |
| Variable-duration review | Optimize time, not count | Estimated durations | Heterogeneous workload | Assuming risk rank is optimal | E4 boundary |

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

| Invariant | Assumption | Diagnostic | Violation | Response |
|---|---|---|---|---|
| Precision falls as prevalence falls | Fixed positive FPR and TPR | Stratified rates | Class-conditional shift | Measure both new rates |
| Top K maximizes additive utility | Identical resource use | Review duration distribution | Different handling times | Optimize time budget |
| Prior correction preserves score order | Positive fixed odds multiplier | Compare scores | Class-dependent feature shift | Refit or recalibrate with representative data |
| Audit weighting recovers totals in expectation | Known positive inclusion | Audit design records | Unobserved stratum | Expand audit support |
| Time split avoids future training data | Features also point-in-time | Availability-time replay | Backfilled post-decision value | Rebuild as-of features |

**Takeaway:** Simple formulas are useful only when their population, cost and observability assumptions hold.

## 5. Worked Numerical Example

**ILLUSTRATIVE, DERIVED:** a payment team evaluates 100,000 eligible transactions from one mature cohort.
Assume all outcomes are known under a stable no-intervention evaluation setting.
There are 100 frauds and 99,900 legitimate transactions.
At a chosen threshold recall is 80% and FPR is 1%.
| Cell | Calculation | Events |
|---|---|---:|
| TP | 100*0.8 | 80 |
| FN | 100-80 | 20 |
| FP | 99,900*0.01 | 999 |
| TN | 99,900-999 | 98,901 |
| Total selected | 80+999 | 1,079 |

Precision is 80/1,079≈0.074143, or 7.41%.
The independent check is 80+20+999+98,901=100,000.
If prevalence instead rises to 1% while class-conditional rates remain fixed, expected counts at the same N are TP=800, FN=200, FP=990 and TN=98,010.
Precision then becomes 800/1,790≈44.69%; the model's TPR and FPR did not change.
With K=500 review slots the 1,079-alert policy is infeasible; no precise top-500 recall follows without score-level ordering and labels.
If the individual incremental values are known and equal-duration assumptions hold, E4 ranks them rather than assuming all 1,079 have equal value.
A wrong path treats 80/100,000 as precision; that is the cohort true-positive fraction, not the selected-set fraud fraction.
Even the calculated values do not identify prevented loss because the example assumes known outcomes and no intervention.
Production implementation needs delayed-label handling, action-effect evidence, staffing, score distributions and uncertainty.
### Additional decision laboratory

All examples in this laboratory are ILLUSTRATIVE and DERIVED, not measured production results.
Currency amounts and traffic volumes are hypothetical.

#### A. Capacity changes the winning threshold

A mature cohort contains 10,000 transactions and 100 frauds.
Three nested candidate sets represent increasingly permissive thresholds.
| Policy | TP | FP | FN | TN | Reviews | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 30 | 70 | 70 | 9,830 | 100 | 30% | 30% |
| B | 55 | 245 | 45 | 9,655 | 300 | 18.33% | 55% |
| C | 70 | 630 | 30 | 9,270 | 700 | 10% | 70% |
Every row sums to 10,000 and includes the same 100 positives.
Suppose each detected fraud yields 200 currency of genuinely incremental value.
Suppose each review costs 5 currency, including all relevant intervention costs.
Policy A yields 30*200-100*5=5,500 currency.
Policy B yields 55*200-300*5=9,500 currency.
Policy C yields 70*200-700*5=10,500 currency.
If K=300, C is infeasible and B wins among these whole-threshold policies.
If K=700, C wins despite having the lowest precision.
From A to B the extra 200 reviews yield 25 additional positives and incremental value 4,000.
From B to C the extra 400 reviews yield 15 additional positives and incremental value 1,000.
With K=500, the table does not reveal which 200 of C's 400 additional cases to choose.
Individual expected benefits or finer score-level data are necessary for that decision.
The example assumes detection causes the stated benefit; confusion counts alone cannot establish prevention.

#### B. Same recall, different customer burden

Two comparable 10,000-event cohorts each contain 100 actual frauds.
Policy D detects 80 frauds and flags 100 of the 9,900 legitimate events.
Policy E detects 80 frauds and flags 500 of the 9,900 legitimate events.
Both recall values are 80%, but their FPRs are approximately 1.01% and 5.05% respectively.
D produces 180 reviews and precision 80/180≈44.44%.
E produces 580 reviews and precision 80/580≈13.79%.
E requires 400 more reviews if each alert consumes one review.
If each false alert imposes 3 currency of expected friction, its extra friction is 1,200 currency.
That estimate assumes identical friction per false alert and excludes labor costs.
Different cohorts can have different composition, so an observed comparison is not automatically causal.

#### C. Risk rank and value rank can disagree

Assume review perfectly prevents the stated fraud loss and costs 4 currency.
Event X has calibrated fraud probability 0.10 and loss 100 conditional on fraud.
Event Z has calibrated fraud probability 0.05 and loss 400 conditional on fraud.
Under the stated assumptions, X has expected incremental benefit 0.10*100-4=6 currency.
Z has expected incremental benefit 0.05*400-4=16 currency.
One available slot should go to Z despite its lower probability.
If review prevents half of each loss, the benefits become 1 and 6 currency respectively.
If Z's review efficacy is zero instead, its incremental value is -4 currency.
Review efficacy is a counterfactual quantity, not a direct consequence of the observed fraud rate.
Conditional loss, efficacy, review cost and customer harm may each vary by transaction.

#### D. Unequal handling times change the optimization

Assume 60 reviewer-minutes and independent actions with known incremental benefit.
Case A uses 60 minutes and yields 30 currency.
Case B uses 30 minutes and yields 20 currency.
Case C uses 30 minutes and yields 19 currency.
The largest individual value is A, which consumes the whole budget and yields 30.
Selecting B and C consumes the same 60 minutes and yields 39.
The equal-slot top-K rule is therefore not an optimizer for this time budget.
A benefit-per-minute ranking happens to select B and C here.
That ratio heuristic is not guaranteed to solve every indivisible 0–1 knapsack problem.
Uncertain durations and deadlines require additional risk or scheduling considerations.

#### E. The marginal value of capacity is not constant

Five independent cases have incremental values 12, 9, 4, 1 and -3 currency.
Every case consumes exactly one identical slot.
With K=2, choose 12 and 9 for total value 21.
With K=3, choose the first three for total value 25.
With K=4, total value rises to 26.
With K=5, leave one slot unused because the last case has negative value.
The third slot adds 4 currency and the fourth adds 1.
A third slot costing more than 4 currency is not worthwhile in this fixed example.
A fourth slot costing less than 1 currency is worthwhile if that cost was not already included.
These are discrete local comparisons, not a universal capacity price.
Noisy estimated benefits can make the apparent best cases overly optimistic.

#### F. A tenfold prevalence change

Assume TPR=0.8 and FPR=0.01 at the same threshold in both cohorts.
Cohort L contains 100,000 transactions including 100 frauds.
The expected alert counts are 80 fraud and 999 legitimate, for precision 80/1,079≈7.41%.
Cohort H contains 100,000 transactions including 1,000 frauds.
Its expected alert counts are 800 fraud and 990 legitimate, for precision 800/1,790≈44.69%.
Selected shares are 1.079% and 1.790% respectively.
A 1,500-slot limit accommodates the first queue but not the second.
The example assumes the class-conditional score distributions transfer.
Attack patterns or feature changes may instead alter both rates and prevalence.
Observed precision alone cannot identify which component changed.

#### G. Cost thresholds differ by action context

False-negative cost is 90 currency in both hypothetical channels.
A false positive costs 10 currency in channel 1 and 30 in channel 2.
The unconstrained E3 cutoffs are 10/(10+90)=0.10 and 30/(30+90)=0.25.
At p=0.20, acting in channel 1 costs 0.8*10=8 in expectation.
Not acting in channel 1 costs 0.2*90=18.
At the same p, acting in channel 2 costs 0.8*30=24.
Not acting in channel 2 still costs 18.
The simple model therefore chooses different actions for the same risk.
The calculation does not establish the legality or fairness of channel-specific treatment.
A shared capacity constraint can alter the unconstrained decisions.

#### H. Selective labels and a randomized audit

A population has 10,000 eligible events, of which a model selects 1,000.
Every selected event receives an accurate mature label, revealing 100 frauds.
A simple random audit of 900 of the 9,000 unselected events finds 9 frauds.
The audited unselected fraud rate is 9/900=1%.
Scaling to the unselected population gives an estimated 90 additional frauds.
The estimated total is 190 and the estimated recall is 100/190≈52.63%.
Incorrectly treating every unselected event as legitimate would give a spurious 100% recall.
Selected precision is 100/1,000=10%.
The unselected inclusion probability is 900/9,000=0.1.
Each audited positive contributes 1/0.1=10 to the estimated positive total.
The total is an estimate, not the known number of frauds.
The recall ratio is not generally exactly unbiased even when the total estimator is design-unbiased.
The calculation needs accurate labels and positive inclusion in the target population.

#### I. Unequal inclusion probabilities

An unselected population contains 1,000 events in stratum A and 9,000 in stratum B.
An audit samples 500 events from each stratum, with probabilities 0.5 and 1/18.
Suppose the audit finds 50 frauds in A and 5 frauds in B.
The unweighted pooled audit rate is 55/1,000=5.5%.
The estimated stratum rates are 10% in A and 1% in B.
Estimated population fraud counts are 100 in A and 90 in B.
The population fraud-rate estimate is 190/10,000=1.9%.
Equivalently, the inverse-inclusion weights are 2 for A and 18 for B.
Oversampling the smaller high-risk stratum makes the unweighted pooled estimate misleading.
Only five B positives were observed, so the weighted estimate remains uncertain.
If B was sampled opportunistically, its nominal rate of 1/18 may not be its actual inclusion probability.

#### J. Label delay changes apparent recall

Suppose a historical cohort ultimately contains 100 frauds.
At day 7, 30 have been confirmed; at day 30, 70; at day 90, all 100.
These are hypothetical cumulative counts, not a universal reporting-delay curve.
At day 7, the model has selected 24 of the 30 confirmed frauds.
A naive analysis of observed positives reports recall 24/30=80%.
By day 90, suppose another 16 selected frauds have been confirmed.
The mature recall is 40/100=40%, not 80%.
The change is possible because selected and unselected cases can have different reporting delays.
A common maturity horizon reduces this bias only if labels are sufficiently mature for both groups.
A delay model must be validated against older cohorts and changes in reporting processes.
Pending labels must not silently become confirmed legitimate transactions.

#### K. Prevention requires a counterfactual

Imagine a randomized experiment with 100 eligible events per arm and comparable outcome ascertainment.
The control arm experiences 20 fraud losses and the review arm experiences 12.
The observed difference is 8 losses per 100 events, or 8 percentage points.
If each prevented loss is worth exactly 100 currency, the sample difference represents 800 currency.
At 5 currency for each of 100 reviews, observed net incremental value is 800-500=300.
This estimate has sampling uncertainty and assumes identical loss amounts.
The eight prevented frauds cannot be identified individually from the aggregate result.
A real experiment requires safety, ethics and applicable legal review.
If blocking changes outcome observability, group comparisons need an ascertainment analysis.
An observational comparison does not automatically have randomization's protection against selection confounding.

#### L. A defensible temporal protocol

Use January through June for training, July for selection, and August for a locked final test.
Every feature must be available by the corresponding transaction's decision time.
Suppose the outcome definition requires 90 days of observation.
The August test cannot be fully evaluated under that rule until at least the end of November.
Training on June cases in early July would also use immature labels unless censoring is explicitly handled.
A March training cutoff at the beginning of July allows more time for the chosen horizon.
Exact timestamps and late revisions still need checking.
Fit imputation, encoders and resampling within each training fold only.
Lock model, calibration and threshold choices before opening August test outcomes.
If August informs another threshold change, it is no longer an untouched final test for that change.
A future-transaction split and an entity-disjoint split answer different deployment questions.

#### M. Concentrated inverse weights

Consider four audited observations with inverse-inclusion weights 1, 1, 1 and 9.
The sum of weights is 12, though only four labels were collected.
A weight-concentration diagnostic is (sum weights)^2 divided by sum of squared weights.
Here it is 12^2/(1+1+1+81)=144/84≈1.71.
The diagnostic is smaller than four because one observation dominates.
It is not a universal substitute for a design-specific variance estimator.
If the weight-9 observation alone is positive, the weighted positive rate is 9/12=75%.
If instead only the three weight-1 observations are positive, the weighted rate is 3/12=25%.
A single rare audited record can therefore strongly influence the estimated rate.
Clipping the weight changes bias or the estimand unless supported by additional assumptions.
Improved audit design and transparent uncertainty are preferable to hiding the concentration.

#### N. Comparable precision-at-K

Model M selects 100 alerts containing 20 known frauds from 10,000 eligible events.
Its precision-at-100 is 20% if all selected outcomes are complete.
Model N selects 100 alerts containing 25 known frauds from the 1,000 cases it can score.
Its apparent precision-at-100 is 25%, but the candidate universes differ.
The comparison ignores the 9,000 cases N cannot score and the fallback behavior for those cases.
Evaluate both systems end to end on the same eligible event population.
Count any fallback alerts against the same review capacity.
Specify the time window, score version and tie-breaking rule before observing labels.
A daily top-100 queue is not comparable with a weekly top-100 queue without further context.
Precision-at-K is not sufficient to estimate full-population recall or prevented loss.

#### O. Diagnosing an alert-volume jump

Suppose a fixed threshold produced 1,000 alerts yesterday and 1,400 today.
First check whether eligible traffic also rose by 40%.
If traffic rose from 100,000 to 140,000, the selected share stayed at 1%.
A raw alert increase alone does not establish a change in selection rate.
Compare model version, missing-feature rate and score distributions by channel.
Inspect the number of events tied at the threshold boundary.
Compare arrivals with actual reviewer service capacity and outstanding backlog.
Do not compute today's recall by treating tomorrow's pending fraud labels as negatives.
For mature cohorts, compare prevalence, TPR and FPR separately.
If only the channel mixture changed, inspect within-channel behavior before blaming the model.
If the feature pipeline changed, replay decision-time feature values against the previous implementation.
Log the timing of overrides or rollbacks to avoid mixing incompatible policies in later evaluation.

#### P. Training weights and probability semantics

Suppose a binary classifier is trained with positive-class weight 10 and negative-class weight 1.
The positive examples contribute more to the fitted loss.
A score of 0.7 does not thereby become a validated 70% deployment fraud probability.
The training weight also does not justify a deployment threshold of 0.5.
Under ideal population optimization of weighted binary cross entropy, the weighting changes posterior odds by the class-weight ratio.
Finite samples, regularization, model misspecification and other losses can break that idealized relationship.
Use a representative calibration set and verify probability quality on a separate cohort.
Compare weighted and unweighted models using the same temporal split and decision capacity.
Report workload, capture, calibration and value rather than only the default-cutoff F1 score.
Do not fit the calibration map on a final test presented as independent evidence.

#### Q. Numerical prior correction

A deployment population has 1,000 frauds and 99,000 legitimate transactions.
Keep all frauds and sample 1,000 legitimate transactions uniformly for fitting.
The resulting training sample has 50% fraud prevalence while deployment prevalence is 1%.
Suppose a sampled-population calibrated posterior equals 0.8 for one event.
The sampled posterior odds are 0.8/0.2=4.
The target-to-sample prior-odds ratio is (1/99)/1=1/99.
The corrected odds are 4/99 and the target probability is 4/(99+4)=4/103≈3.88%.
A sample posterior of 0.5 maps to 1% under the same assumptions.
The correction is monotone, so it preserves rank in exact arithmetic.
It can still change decisions at a fixed probability threshold.
If sampling depends on score or channel within a class, the class-only assumption fails.
Changes in class-conditional distributions also invalidate a prior-only correction.

#### R. Different reporting delays by channel

Assume channel A reports 90% of eventual frauds within 30 days.
Assume channel B reports 50% of eventual frauds within the same period.
If each truly has 100 frauds, the expected observed counts are 90 and 50.
Suppose each channel has 10,000 eligible events and true prevalence is 1%.
The day-30 observed label rates are then 0.9% and 0.5%.
They differ even though the eventual fraud prevalence is the same.
Training or threshold selection on incomplete labels can underprioritize slower-reporting channels.
A longer common horizon helps only if both channels are sufficiently mature by that time.
Adjudication quality and outcome definitions may still differ after waiting.
Any delay adjustment needs an explicit transport assumption and uncertainty assessment.
Record observed labels, unresolved counts and the chosen horizon separately.

#### S. Precision does not equal customer harm

Suppose 2,000 reviewed transactions include 100 frauds and 1,900 legitimate events.
Precision is 100/2,000=5% and the false-discovery proportion is 95%.
Those proportions sum to one because every outcome is assumed complete and binary.
If 200 investigations remain pending, the full-queue proportions are unknown.
Treating the pending cases as legitimate creates an unsupported estimate.
Computing only on resolved cases can also be biased if resolution speed depends on outcome.
A false alert does not necessarily mean a blocked purchase.
The action may instead be a silent review, a customer contact or a decline.
Count alerts, review effort, interventions and customer harms as separate outcomes.
The model's net benefit also depends on incremental prevented loss and alternatives.

#### T. Time-dependent review benefit

A case is worth 20 currency if reviewed within one hour and 5 if reviewed later.
A second case is worth 12 currency whether reviewed in the first or second hour.
Both arrive simultaneously and require exactly one hour of work.
Reviewing the urgent case first yields 20+12=32 currency.
Reviewing the stable case first yields 12+5=17 currency.
The same two cases are reviewed in either ordering, but total value differs by 15.
A static selected set does not specify scheduling and misses this difference.
The example assumes deterministic durations and known benefit decay.
Uncertain processing times and specialist requirements require richer scheduling decisions.
A fixed score cutoff cannot itself guarantee a review deadline.

#### U. Threshold selection uncertainty

A validation set reports net value 10,000 at threshold 0.10 and 10,050 at 0.11.
A 50-currency observed difference is not proof that the second threshold is truly better.
Inspect the utility curve and workload across neighboring thresholds.
A sharp peak based on a few frauds may be unstable in a later period.
Use resampling or later cohorts appropriate to the dependence and timing of the data.
If a fraud ring creates many related transactions, independent-event resampling can understate uncertainty.
An operationally robust threshold range can be preferable to a noisy apparent maximum.
Freeze both threshold and selection procedure before evaluating a final test.
Report capacity breaches and uncertainty rather than only the best point estimate.
Repeated use of the test data to tune the cutoff undermines its independence.

#### V. Precision rises while recall falls

Two fully observed cohorts each contain 10,000 events.
Cohort 1 has 100 frauds, of which 80 are detected, and 99 false alerts.
Its precision is 80/179≈44.69%, recall is 80%, and FPR is 99/9,900=1%.
Cohort 2 has 200 frauds, of which 140 are detected, and 98 false alerts.
Its precision is 140/238≈58.82%, recall is 70%, and FPR is 98/9,800=1%.
Precision rose while recall fell by 10 percentage points.
The number of detected frauds rose from 80 to 140 despite the lower capture fraction.
The higher prevalence changes the composition of the selected set.
Reporting precision or the true-positive count alone hides the recall deterioration.
Compare confusion cells, cohort prevalence, uncertainty and preventable financial loss together.

#### W. Daily totals conceal burst overload

Suppose 600 events arrive uniformly over six hours and service capacity is 100 per hour.
The average arrival rate matches the service rate, leaving no planned headroom for variability.
Now suppose all 600 arrive at the start of the first hour instead.
After one hour, 500 remain waiting under deterministic fluid accounting.
After two hours, 400 remain; after six hours, the original backlog has cleared.
Both scenarios contain 600 arrivals and 600 nominal service slots.
Their waiting experiences differ because arrival timing differs.
If the intervention expires after two hours, late completion may have little value.
This example assumes an initially empty queue and no further arrivals.
Measured burst patterns and service-time tails are needed for percentile wait estimates.

#### X. Audit allocation with a fixed budget

A team can audit 100 events drawn from two disjoint population strata.
Stratum A contains 1,000 events and stratum B contains 9,000.
A proportional allocation samples 10 from A and 90 from B.
An equal allocation samples 50 from each, oversampling the smaller stratum.
Suppose 10 of the 50 audited A cases and 5 of the 50 audited B cases are fraud.
The estimated stratum rates are 20% and 10% respectively.
The population-weighted estimate is 0.1*20%+0.9*10%=11%.
The naive pooled audit rate is 15/100=15%, four percentage points higher.
Equal allocation may improve information for the small stratum but requires population weighting.
The small sample does not establish the optimal allocation or a precise confidence interval.
Choose the design before outcomes and ensure positive inclusion in target strata.

#### Y. Ranking can survive a score transformation

Four events have scores 0.1, 0.2, 0.3 and 0.4 in increasing risk order.
Multiplying every score by two produces 0.2, 0.4, 0.6 and 0.8.
The order is unchanged, so a top-two policy selects the same two events.
At numeric cutoff 0.5 the original scores select none and the transformed scores select two.
Neither score set is necessarily calibrated, and four events cannot validate calibration.
A cost threshold expressed in probability units cannot be applied to arbitrary scores.
A strictly increasing calibration map preserves ranking apart from numerical ties.
A non-strict map can introduce ties even if the original scores differ.
Tie-breaking needs explicit policy and capacity semantics.
Unchanged rank does not imply unchanged expected cost under a fixed numeric threshold.

#### Z. What an observed precision can and cannot establish

Suppose a review queue has 100 alerts and 30 are confirmed fraud.
With complete and correct labels, observed precision is 30%.
That fact does not reveal fraud among the unreviewed population.
It does not show whether review prevented loss or merely confirmed an already stopped event.
It does not quantify review labor, customer friction or intervention error.
It does not prove another policy with equal capacity would find fewer preventable frauds.
Estimate recall using credible information about fraud outside the queue.
Estimate action value using an appropriate no-action or alternative-action counterfactual.
Choose a policy using incremental benefit, capacity and deadlines.
Keep cohort composition and label timing alongside the precision numerator.

**Takeaway:** Base rate drives alert precision and capacity feasibility cannot be inferred from recall alone.

## 6. Visual / Geometric Intuition

```text
Illustrative 100,000-event cohort, exact 1% FPR
actual fraud        100:  [TP 80]  [FN 20]
actual legitimate 99,900: [FP 999] [TN 98,901]
                             ^ selected ^
```

Caption: rows are actual outcomes; selected cells are TP and FP; widths are not to scale.
The selected column contains 1,079 events, of which 80 are fraud.
Increasing the threshold generally shrinks both selected cells, but does not guarantee a better business outcome.

```text
Time -> event ---- decision ---- investigation ---- horizon ---- late correction
         features  policy         partial label     mature cohort  versioned update
```

Caption: data recorded after the decision cannot be a feature available at the decision.
A mature cohort has passed its defined horizon, but a late correction can still revise a label.
**Takeaway:** Matrix orientation explains denominators; timeline orientation exposes leakage and censoring.

## 7. Comparison / Selection Table

| Method | Preconditions | Quality / guarantee | Time and cost | Use | Avoid | Evidence |
|---|---|---|---|---|---|---|
| No-action baseline | Defined outcome | Simple comparator, not a safety recommendation | Minimal action cost | Establish missed-loss baseline | Policy requires mandatory action | DERIVED |
| Fixed probability threshold | Calibrated target probability, costs | E3 minimizes stated per-event expected cost | O(N) scoring plus actions | Stable costs and ample capacity | Uncalibrated or constrained queue | DERIVED |
| Top K benefit | Credible additive benefit and equal slot use | E4 exchange argument | Sort O(N log N) | Hard slot limit | Variable times or interactions | DERIVED |
| Class weighting | Training weights specified | Alters fitting objective, no universal improvement | Training-dependent | Compare on representative holdout | Assuming resulting scores calibrated | HEURISTIC |
| Negative downsampling | Known sampling design | Efficient training; prior correction conditional | Lower sampled training volume | Extremely abundant negatives | Ignoring selection mechanism | DERIVED under E5 |
| Oversampling | Split before resampling | May emphasize rare positives | More fitting work | A training-only experiment | Duplicating records across split | HEURISTIC |
| Random audit | Positive known inclusion | Supports weighted totals | Label acquisition cost | Selective labels | Zero inclusion regions | DERIVED under E6 |

First disqualify options lacking an observable or identifiable target.
Then satisfy mandatory safety, fairness and capacity constraints before optimizing optional utility.
If equal review slots and valid incremental values exist, choose up to K largest positive values.
If review durations vary, use a resource-constrained allocation and measure durations.
If probabilities are unreliable, retain a validated baseline while collecting representative labels.
Do not compare training techniques on their resampled training distribution.
Comparable production benchmarks for these alternatives are NOT MEASURED here.
Use the same time-separated test cohort, feature availability, outcomes, compute budget and action simulation for a fair comparison.
**Takeaway:** Select the policy by identification and constraints first, then measure incremental value against a baseline.

## 8. Failure Modes & Edge Cases

All cases below are HYPOTHETICAL teaching diagnostics, not incident reports.
The collection-wide Common Issues file is planned and has no existing issue IDs.
Each case separates a signal from evidence needed to distinguish alternative explanations.

### F-01 — Prevalence collapse

- Symptom: Precision falls despite steady scores.
- Diagnostic: Compare mature prevalence and class-conditional rates by cohort.
- Discriminating evidence: A lower base rate alone predicts lower precision using E2.
- Correction: Re-estimate queue volume and reconsider threshold.
- Boundary: Do not assume the ranking model degraded.

### F-02 — FPR creep

- Symptom: Legitimate alerts rise faster than traffic.
- Diagnostic: Compare FPR on mature negatives at a fixed score threshold.
- Discriminating evidence: A higher FPR with stable prevalence implicates score or traffic shift.
- Correction: Trace feature and model changes by segment.
- Boundary: Do not infer causality from one aggregate.

### F-03 — Fraud recall falls

- Symptom: More confirmed fraud appears below threshold.
- Diagnostic: Compare mature positive score distributions and capture rate.
- Discriminating evidence: A shift of positive scores downward supports ranking drift.
- Correction: Investigate features and retrain only after validating labels.
- Boundary: Do not treat pending positives as negatives.

### F-04 — Review selection bias

- Symptom: Reviewed precision looks excellent.
- Diagnostic: Compare random-audit outcomes inside and outside the queue.
- Discriminating evidence: A nonzero fraud rate outside review exposes missing positives.
- Correction: Expand randomized audits with recorded inclusion probabilities.
- Boundary: Do not extrapolate to zero-inclusion strata.

### F-05 — Delayed labels

- Symptom: Recent recall is implausibly low.
- Diagnostic: Compare outcome age curves for identical cohort definitions.
- Discriminating evidence: Older cohorts gaining positives indicates delayed reporting.
- Correction: Evaluate mature cohorts and track label development.
- Boundary: Do not force a recent pending case into TN.

### F-06 — Post-decision features

- Symptom: Offline metrics greatly exceed online results.
- Diagnostic: Replay feature values using availability rather than event timestamps.
- Discriminating evidence: A feature populated after decision indicates leakage.
- Correction: Rebuild point-in-time joins and retrain.
- Boundary: Do not assume a temporal split fixes backfills.

### F-07 — Duplicate entities

- Symptom: Validation is unexpectedly easy.
- Diagnostic: Check entity and transaction overlap across partitions.
- Discriminating evidence: Repeated events across splits permit memorization.
- Correction: Choose a split matching intended generalization and deduplicate.
- Boundary: Do not impose an entity split without defining deployment use.

### F-08 — Oversampling before split

- Symptom: Validation contains duplicates of training positives.
- Diagnostic: Match exact IDs and synthetic-neighbor ancestry across splits.
- Discriminating evidence: Shared source records establish contamination.
- Correction: Split first and resample training only.
- Boundary: Do not compare against the contaminated validation score.

### F-09 — Downsampling correction

- Symptom: Raw scores exceed observed fraud prevalence.
- Diagnostic: Compare sampled and deployment priors plus calibration bins.
- Discriminating evidence: A prior mismatch can produce a probability intercept shift.
- Correction: Correct odds only if class-conditional stability holds.
- Boundary: Do not apply prior-only correction to arbitrary selection.

### F-10 — Class weighting

- Symptom: Weighted model scores look like probabilities.
- Diagnostic: Compare raw scores with mature representative outcomes.
- Discriminating evidence: Systematic miscalibration shows score semantics changed.
- Correction: Fit a separate calibration map on an untouched representative set.
- Boundary: Do not assume weighting always improves ranking.

### F-11 — Threshold ties

- Symptom: Queue exceeds capacity at a cutoff.
- Diagnostic: Count records equal to the boundary score.
- Discriminating evidence: Boundary ties can produce more than K selections.
- Correction: Use a documented deterministic or randomized tie-break.
- Boundary: Do not silently select different customers on replay.

### F-12 — Time-varying capacity

- Symptom: Alerts accumulate after staffing changes.
- Diagnostic: Compare arrival rate and available handling time by shift.
- Discriminating evidence: Sustained arrivals above service capacity predict growing backlog.
- Correction: Adjust routing, staffing or risk-prioritized capacity policy.
- Boundary: Do not assume daily totals control intra-day peaks.

### F-13 — Variable handling time

- Symptom: Top-risk queue consumes all reviewer hours.
- Diagnostic: Compare observed review durations and incremental values.
- Discriminating evidence: Large duration variation invalidates equal-slot assumptions.
- Correction: Optimize value subject to review-time budget.
- Boundary: Do not equate review count with workload.

### F-14 — Blocked transaction outcome

- Symptom: Declines appear to eliminate fraud labels.
- Diagnostic: Check action-specific observability and downstream outcome pathways.
- Discriminating evidence: Missing counterfactual outcomes prevent direct no-action recall.
- Correction: Use an ethically approved randomized or quasi-experimental design.
- Boundary: Do not label all blocked transactions as fraud.

### F-15 — Investigator confirmation bias

- Symptom: Reviewed cases have unusually high positive rates.
- Diagnostic: Compare independent adjudication with the operational labels.
- Discriminating evidence: Disagreement correlated with score suggests potential confirmation bias.
- Correction: Blind adjudicators where feasible and record disagreements.
- Boundary: Do not assume a second reviewer is automatically independent.

### F-16 — Proxy-label drift

- Symptom: Chargeback capture changes after process change.
- Diagnostic: Compare label-source mix, reporting lag and adjudication rules.
- Discriminating evidence: A workflow change can alter the proxy without true fraud changing.
- Correction: Version the target and separately monitor validated fraud outcomes.
- Boundary: Do not compare incompatible label definitions.

### F-17 — Calibration drift

- Symptom: Observed fraud rates differ from predicted bins.
- Diagnostic: Use mature representative labels and bin uncertainty intervals.
- Discriminating evidence: Persistent bin gaps show probability mismatch.
- Correction: Recalibrate on fresh data and validate out of time.
- Boundary: Do not assume recalibration fixes selection bias.

### F-18 — Ranking and calibration confusion

- Symptom: Brier improves while queue capture declines.
- Diagnostic: Compare precision-at-K and calibration on the same cohort.
- Discriminating evidence: The metrics can move in different directions.
- Correction: Evaluate the decision objective and probability quality separately.
- Boundary: Do not choose solely by one aggregate score.

### F-19 — Zero positive slice

- Symptom: A segment has undefined recall.
- Diagnostic: Inspect TP+FN and the number of mature labeled positives.
- Discriminating evidence: A zero denominator makes recall mathematically undefined.
- Correction: Report support and pool or collect more data when justified.
- Boundary: Do not silently claim zero or perfect recall.

### F-20 — Zero selected slice

- Symptom: Precision appears as a perfect score.
- Diagnostic: Inspect TP+FP and library zero-division behavior.
- Discriminating evidence: No selected events means precision has no defined denominator.
- Correction: Report not defined and the selection count.
- Boundary: Do not compare it numerically to populated slices.

### F-21 — Rare event uncertainty

- Symptom: A segment metric jumps after one fraud.
- Diagnostic: Show counts and uncertainty with entity dependence considered.
- Discriminating evidence: A tiny positive denominator makes estimates unstable.
- Correction: Accumulate evidence or use a justified hierarchical analysis.
- Boundary: Do not call every swing model drift.

### F-22 — Data join multiplication

- Symptom: Confusion totals exceed eligible volume.
- Diagnostic: Compare unique event IDs before and after label joins.
- Discriminating evidence: Multiple labels per event duplicate counts.
- Correction: Choose an as-of label version and enforce one row per event.
- Boundary: Do not divide duplicated numerator by unduplicated denominator.

### F-23 — Score version mismatch

- Symptom: Calibration deteriorates immediately after deployment.
- Diagnostic: Join scores to model and calibration artifact versions.
- Discriminating evidence: Mixed mappings indicate an incompatible deployment.
- Correction: Rollback or restore a matched model-calibrator pair.
- Boundary: Do not pool incompatible versions into one calibration curve.

### F-24 — Label revision

- Symptom: Historical dashboards disagree across runs.
- Diagnostic: Compare event IDs, label versions and as-of extraction timestamps.
- Discriminating evidence: Late corrections change earlier cohort totals.
- Correction: Publish frozen as-of views alongside revised views.
- Boundary: Do not overwrite history without lineage.

### F-25 — Audit probability error

- Symptom: Weighted fraud totals are unstable.
- Diagnostic: Reconcile assigned versus realized inclusion probabilities.
- Discriminating evidence: Wrong probabilities or extreme weights distort estimates.
- Correction: Fix design logging and examine weight distributions.
- Boundary: Do not clip weights without disclosing changed bias.

### F-26 — Audit support gap

- Symptom: A subgroup has no labeled outcomes.
- Diagnostic: Inspect inclusion probabilities by eligibility and action.
- Discriminating evidence: Zero selection probability makes that subgroup unidentified.
- Correction: Extend audit coverage where lawful and safe.
- Boundary: Do not present a model-based extrapolation as design-unbiased.

### F-27 — Action-effect confusion

- Symptom: High-risk alerts have low recovered value.
- Diagnostic: Measure review benefit against a credible no-review comparator.
- Discriminating evidence: Risk may be high while intervention efficacy is low.
- Correction: Estimate incremental value including preventable loss and friction.
- Boundary: Do not substitute correlation for causal effect.

### F-28 — Threshold transfer

- Symptom: A cutoff creates excessive friction in a new market.
- Diagnostic: Compare prevalence, calibration, costs and score distribution by market.
- Discriminating evidence: Changed costs or base rates invalidate the transferred policy.
- Correction: Re-estimate on local representative evidence and constraints.
- Boundary: Do not assume the original numeric cutoff is universal.

### F-29 — Policy feedback loop

- Symptom: Training labels concentrate on the previous model queue.
- Diagnostic: Plot label probability against previous score and action.
- Discriminating evidence: Concentrated labels expose policy-dependent observation.
- Correction: Maintain a representative audit and log propensity.
- Boundary: Do not train as if missing outcomes were random.

### F-30 — Wrong decision grain

- Symptom: Customer-level recall conflicts with transaction recall.
- Diagnostic: Reconcile event-to-customer mapping and aggregation rules.
- Discriminating evidence: Repeated events change weighting between the two metrics.
- Correction: Report each estimand with its denominator.
- Boundary: Do not treat transaction and customer rates as interchangeable.

### F-31 — Monetary versus count recall

- Symptom: Fraud count recall increases while losses rise.
- Diagnostic: Compare fraud amount distribution and prevented loss by cohort.
- Discriminating evidence: Missed high-value cases can dominate monetary impact.
- Correction: Evaluate expected preventable loss with action costs.
- Boundary: Do not replace count recall without describing tradeoffs.

### F-32 — Temporal aggregation

- Symptom: Daily capacity fits but peak queue breaches SLA.
- Diagnostic: Compare hourly arrivals, service rates and waiting times.
- Discriminating evidence: A burst can overwhelm the system despite daily average headroom.
- Correction: Plan shifts and buffer policies at the relevant time resolution.
- Boundary: Do not infer tail latency from average rate alone.

### F-33 — Data freshness

- Symptom: New scores abruptly collapse toward one value.
- Diagnostic: Inspect feature freshness, missingness and fallback usage.
- Discriminating evidence: Simultaneous missing features can flatten the score distribution.
- Correction: Restore the pipeline or switch to a validated fallback.
- Boundary: Do not retrain before diagnosing input integrity.

### F-34 — Sampling leakage

- Symptom: Synthetic examples leak validation information.
- Diagnostic: Audit resampling lineage and preprocessing fit boundaries.
- Discriminating evidence: Transformers fit on held-out records contaminate evaluation.
- Correction: Fit all learned transformations on each training fold only.
- Boundary: Do not rely solely on a final holdout if it guided repeated tuning.

### F-35 — Simpson aggregation

- Symptom: Overall precision improves while all segments worsen.
- Diagnostic: Compare segment-specific precision and traffic weights over time.
- Discriminating evidence: Composition shifts can reverse an aggregate comparison.
- Correction: Report standardized and segment-specific views.
- Boundary: Do not interpret the aggregate as a within-segment improvement.

### F-36 — Drifting costs

- Symptom: Previously optimal threshold becomes expensive.
- Diagnostic: Update review cost, fraud loss and friction estimates.
- Discriminating evidence: Changed relative costs change the E3 threshold.
- Correction: Recompute the policy with current credible costs.
- Boundary: Do not update cost assumptions without versioning.

### F-37 — Label contamination

- Symptom: Training target includes the model decision.
- Diagnostic: Inspect label generation logic for circular use of scores.
- Discriminating evidence: A label assigned from the decision creates self-confirmation.
- Correction: Separate independent evidence from policy-generated status.
- Boundary: Do not evaluate a model against its own action.

### F-38 — Improper test reuse

- Symptom: Validation gains vanish in the final cohort.
- Diagnostic: Track the number of model and threshold choices informed by each set.
- Discriminating evidence: Repeated tuning can overfit even a time holdout.
- Correction: Reserve a genuinely untouched final temporal test.
- Boundary: Do not relabel a tuned validation set as final test.

### F-39 — Score comparator bug

- Symptom: Exactly-at-threshold events behave inconsistently.
- Diagnostic: Replay stored scores and compare >= versus >.
- Discriminating evidence: Differences only at the cutoff reveal comparator mismatch.
- Correction: Version the threshold rule and tie behavior.
- Boundary: Do not infer expected alert counts without comparator semantics.

### F-40 — Maturity differs by group

- Symptom: One channel seems safer in recent data.
- Diagnostic: Compare reporting-delay distributions across channels.
- Discriminating evidence: Different delay distributions bias a common short horizon.
- Correction: Use a suitable common mature horizon or explicit delay modeling.
- Boundary: Do not equate missing recent claims with clean events.

### F-41 — Fraud ring dependence

- Symptom: Confidence intervals are implausibly narrow.
- Diagnostic: Inspect shared customers, devices and campaigns.
- Discriminating evidence: Correlated events violate independent-event resampling.
- Correction: Use a justified cluster or temporal bootstrap.
- Boundary: Do not claim independence solely from unique transaction IDs.

### F-42 — Review queue expiry

- Symptom: High-risk events wait until intervention is ineffective.
- Diagnostic: Compare decision-to-review time with intervention deadlines.
- Discriminating evidence: Expiring cases lose value as they wait.
- Correction: Include time-decaying benefit and route urgent work.
- Boundary: Do not optimize static end-of-day capture alone.

### F-43 — Reversed score orientation

- Symptom: Alerts focus on lowest-risk records.
- Diagnostic: Check score distribution against known mature positives.
- Discriminating evidence: A reversed comparator selects the wrong tail.
- Correction: Correct the score contract and replay a controlled cohort.
- Boundary: Do not assume larger scores always mean higher fraud risk.

### F-44 — Training deployment skew

- Symptom: Offline features are complete but serving features are missing.
- Diagnostic: Compare point-in-time offline and online feature snapshots.
- Discriminating evidence: Systematic differences reveal training-serving skew.
- Correction: Unify definitions and test event-level parity.
- Boundary: Do not treat matching column names as matching semantics.

### F-45 — Subgroup capacity disparity

- Symptom: One population experiences excessive false alerts.
- Diagnostic: Compare rates and workloads by relevant lawful subgroup.
- Discriminating evidence: Different FPR or base rates may yield unequal burden.
- Correction: Investigate causes and evaluate policy under applicable constraints.
- Boundary: Do not infer legal compliance from one fairness metric.

### F-46 — Investigation backlog censoring

- Symptom: Recent reviews show surprisingly low confirmation.
- Diagnostic: Track pending investigations and outcome age by queue.
- Discriminating evidence: Unresolved high-risk cases bias early precision estimates.
- Correction: Wait for maturity or use a justified observation model.
- Boundary: Do not count unresolved cases as cleared.

### F-47 — Manual override

- Symptom: Observed actions diverge from model recommendations.
- Diagnostic: Compare recommendation, override reason and actual action.
- Discriminating evidence: A high override rate breaks simulated-policy assumptions.
- Correction: Measure override effects and include operational behavior.
- Boundary: Do not claim model policy impact from recommended actions alone.

### F-48 — Queue duplicates

- Symptom: Reviewers receive repeated cases for one transaction.
- Diagnostic: Compare unique event and case IDs with retry logs.
- Discriminating evidence: Duplicate case creation consumes capacity without new coverage.
- Correction: Make queue writes idempotent and reconcile outstanding work.
- Boundary: Do not interpret raw case count as unique events.

### F-49 — Expensive false positives

- Symptom: Precision improves but customer harm grows.
- Diagnostic: Measure friction, appeals and legitimate transaction value.
- Discriminating evidence: A smaller queue can still concentrate high-friction interventions.
- Correction: Include incremental customer harm in the objective.
- Boundary: Do not equate fraud precision with customer welfare.

### F-50 — Threshold overfit

- Symptom: Best validation cutoff fails in the next month.
- Diagnostic: Compare utility across neighboring thresholds and later cohorts.
- Discriminating evidence: A narrow validation optimum indicates sensitivity to noise.
- Correction: Use temporal validation and a robust operating range.
- Boundary: Do not report the selected maximum as an unbiased estimate.

### F-51 — Positive-unlabeled labels

- Symptom: Unlabeled examples are treated as confirmed negative.
- Diagnostic: Compare label mechanisms and audit outcomes among unlabeled events.
- Discriminating evidence: Fraud in the unlabeled pool violates binary-complete assumptions.
- Correction: Use audited outcomes or an explicit positive-unlabeled design.
- Boundary: Do not apply ordinary confusion counts to unknown labels.

### F-52 — Interacting investigations

- Symptom: Two alerts refer to the same fraud ring.
- Diagnostic: Check overlap in prevented loss and investigator actions.
- Discriminating evidence: Summed individual benefits can double-count shared loss.
- Correction: Allocate at the relevant entity or campaign level.
- Boundary: Do not apply independent additive top K without checking interactions.

### F-53 — High-variance weighting

- Symptom: Audit estimate changes sharply across runs.
- Diagnostic: Inspect effective support and concentration of inverse weights.
- Discriminating evidence: A few low-inclusion records can dominate estimates.
- Correction: Improve sampling design and report uncertainty.
- Boundary: Do not hide unstable estimates behind point precision.

### F-54 — Queue latency

- Symptom: Cases reach analysts after their actionable window.
- Diagnostic: Measure event-to-enqueue and enqueue-to-review latency separately.
- Discriminating evidence: The larger component locates the delay mechanism.
- Correction: Repair the bottleneck and monitor expiry-adjusted benefit.
- Boundary: Do not equate processing latency with review waiting time.

### F-55 — Model staleness

- Symptom: Feature patterns shift after a campaign change.
- Diagnostic: Compare fixed-cohort and rolling-cohort performance with label maturity.
- Discriminating evidence: A persistent shift after data integrity checks warrants investigation.
- Correction: Reassess features and retraining on temporally valid evidence.
- Boundary: Do not assume every score shift means fraud drift.

### F-56 — Precision-recall baseline

- Symptom: A PR curve looks impressive at high prevalence.
- Diagnostic: Record prevalence and the exact PR metric definition.
- Discriminating evidence: Random-ranking precision tracks prevalence in expectation.
- Correction: Compare models on the same deployment-relevant cohort.
- Boundary: Do not compare unadjusted PR curves from different populations.

### F-57 — ROC optimism

- Symptom: ROC-AUC remains strong while review load is unacceptable.
- Diagnostic: Measure FPR and precision at the actual operating capacity.
- Discriminating evidence: A small FPR applied to many negatives can overwhelm reviewers.
- Correction: Use operational precision, recall and utility at feasible load.
- Boundary: Do not use AUC alone as a deployment criterion.

### F-58 — Score-only capacity rule

- Symptom: Reviewing the top risk cases loses more money than expected.
- Diagnostic: Compare risk rank with estimated preventable value rank.
- Discriminating evidence: Loss size and review efficacy can reverse the ordering.
- Correction: Prioritize validated incremental value where defensible.
- Boundary: Do not multiply risk by loss without action-effect assumptions.

### F-59 — Population eligibility mismatch

- Symptom: Offline fraud rate differs from production.
- Diagnostic: Reconcile filters, event time and eligibility at each stage.
- Discriminating evidence: Different inclusion rules change prevalence and all metrics.
- Correction: Rebuild the cohort to match the deployed decision.
- Boundary: Do not compare metrics across incompatible populations.

### F-60 — Calibration extrapolation

- Symptom: Predictions near one lack supporting data.
- Diagnostic: Count mature labeled examples in the score tail.
- Discriminating evidence: Sparse support makes tail calibration uncertain.
- Correction: Collect targeted randomized evidence or avoid extrapolating.
- Boundary: Do not equate a smooth curve with validated tail probabilities.

**Takeaway:** Diagnose cohort, label, score, action and capacity changes separately before assigning a cause.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic Complexity

Let N be the number of events, K the maximum selected events, and d the feature count.
These are comparison/operation bounds, not measured latency or a guarantee about a particular library.
| Operation | Time | Additional space | I/O / qualification |
|---|---|---|---|
| Scan fixed threshold | O(N) | O(1) excluding output | Read N scores; output up to N IDs. |
| Full sort for top K | O(N log N) | O(N) for a conventional materialized sort | External sort adds disk I/O. |
| Heap top K | O(N log K) for K>=2 | O(K) | Streaming; stable tie-break included in comparator. |
| Confusion totals | O(N) | O(1) per fixed cohort | Requires an as-of label join; join cost is additional. |
| Dense linear scoring | O(Nd) | O(d) model plus input/output | Sparse representation can reduce arithmetic. |
| Calibration fit | Method-dependent | Method-dependent | Iterations, score count and implementation matter. |
| Recovery / replay | O(N) for a simple replay | Output and state dependent | Idempotent sink and versioned features required. |

These bounds omit data ingestion, distributed shuffle, model fitting and case-management work.
A distributed top-K additionally requires local candidate exchange and merging.
Unequal partitions and slow workers can determine end-to-end completion.

### 9b. Back-of-Envelope Sizing

**ILLUSTRATIVE:** 100,000 eligible transactions/day arrive uniformly over eight staffed hours.
The selected-alert rate is 1.079%, producing 1,079 alerts/day under Section 5.
Assume a reviewer completes six cases/hour with no interruptions.
A reviewer supplies 6*8=48 cases/day.
The mathematical minimum is ceil(1,079/48)=23 reviewers.
At 23 reviewers, nominal daily throughput is 1,104 cases, only 25 above demand.
At 25 reviewers, nominal throughput is 1,200 cases, leaving 121 cases/day of nominal headroom.
This is not a staffing recommendation: arrival bursts, breaks, rework, skill mix and service-time tails are excluded.
If each reviewer costs 200 currency/day including the stated labor scope, 25 reviewers cost 5,000 currency/day.
At 1,079 completed reviews the illustrative labor cost is approximately 4.63 currency/review.
Fraud loss, tooling, customer friction and benefits are excluded; currency and date are hypothetical.
Measure actual arrival and handling-time distributions before deciding staffing or service levels.
An eight-hour uniform-arrival approximation cannot establish a percentile wait-time guarantee.
**Takeaway:** Count-level allocation can be computationally cheap while human review and delayed outcomes dominate operating cost.

## 10. Key Theorems / Results Reference

| Result | Statement and hypotheses | Implication | Does not imply | Source |
|---|---|---|---|---|
| Bayes rule | Conditional probabilities with nonzero conditioning probability obey the identity in E2. | Prevalence changes precision even with fixed TPR and FPR. | TPR and FPR remain fixed after a population shift. | [NIST probability handbook](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35.htm) |
| Neyman–Pearson lemma | For simple hypotheses, a likelihood-ratio test is most powerful at a specified size, allowing randomization where needed. | A useful idealized fixed-FPR benchmark. | A learned score is a true likelihood ratio or that utility is optimized. | [Neyman and Pearson, 1933](https://doi.org/10.1098/rsta.1933.0009) |
| Inverse-probability total | With known positive design inclusion probabilities, the Horvitz–Thompson total is design-unbiased. | Audits can estimate population totals. | Ratio estimates are exactly unbiased or missing labels are correct. | [Horvitz and Thompson, 1952](https://doi.org/10.1080/01621459.1952.10483446) |
| Precision–recall relationship | PR and ROC views are related for a fixed class distribution. | PR exposes workload under imbalance. | Areas are comparable across different prevalence without qualification. | [Davis and Goadrich, 2006](https://doi.org/10.1145/1143844.1143874) |

**Source ledger and limits:**
- The linked primary papers support the stated theoretical results; they do not validate the hypothetical payment workload.
- [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) documents fit/transform leakage precautions; software behavior is version-dependent.
- [scikit-learn precision–recall documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) distinguishes precision, recall and average precision.
- [Federal Reserve Regulation E, 12 CFR 1005.11](https://www.ecfr.gov/current/title-12/chapter-X/part-1005/section-1005.11) illustrates that dispute handling can be legally constrained; applicability requires qualified counsel and is not established for this scenario.
- Source links and current versions have not been independently reviewed in this draft; review when publishing.
- The source inventory lists filenames, not the unavailable original source contents; coverage is topic-level only.
- No comparative model experiment, operational benefit or live production benchmark has been measured.
**Takeaway:** Formal guarantees are conditional and cannot fill an unobserved counterfactual or justify an invented benchmark.

## 11. Interview / Exam Questions

### Level 1 — Definition / Recall

1. **What is precision?**
   Precision is TP/(TP+FP) on the defined selected set when the denominator is positive.
   For 8 frauds among 100 alerts, precision is 8%; this is not recall.
2. **What is recall?**
   Recall is TP/(TP+FN) among all actual frauds with observable target outcomes.
   Eight found out of ten total frauds gives 80%, regardless of the number of false alerts.
3. **Why is an uncalibrated score not necessarily a probability?**
   A score can rank transactions without matching observed outcome frequencies.
   Scores 2 and 1 may order risk correctly but are not probabilities at all.

### Level 2 — Derivation / Apply

4. **Derive the base-rate precision identity.**
   Apply Bayes' rule to selected and fraud events.
   Expand selection probability by the fraud and nonfraud partition.
   The result is E2, provided selection probability is nonzero.
5. **Compute the confusion matrix for Section 5.**
   There are 100 positives and 99,900 negatives.
   At recall 0.8 and FPR 0.01 the cells are TP 80, FN 20, FP 999 and TN 98,901.
   Precision is 80/1,079≈7.41%, and counts sum to 100,000.
6. **Compute a cost threshold for false-positive cost 10 and false-negative cost 90.**
   Compare (1-p)*10 against p*90.
   Action is weakly preferred at p>=0.1, assuming the E3 cost model.
7. **What happens to a 0.5 posterior learned from a balanced case-control sample if target prevalence is 1%?**
   Under pure class-dependent sampling and stable class-conditional distributions, the corrected odds are 1/99.
   The resulting probability is 0.01; other forms of shift invalidate the shortcut.

### Level 3 — Judgment / Design / Debug

8. **Why can ROC-AUC be high while an alert queue is unusable?**
   AUC summarizes ranking over thresholds; a small FPR can yield many alerts among abundant negatives.
   Compute precision, recall and expected action value at the actual capacity.
9. **How do you evaluate a model when only investigated events have labels?**
   State the target and missingness mechanism; start a safe, representative randomized audit where possible.
   Record inclusion probabilities, use design-aware estimation, and acknowledge unsupported strata.
10. **Does a temporal split prevent leakage?**
    Not necessarily: a backfilled feature can have an old event timestamp but become available only after the decision.
    Replay features using availability time and test versioned joins.
11. **How do you select 500 reviews?**
    If each review takes one equivalent slot and benefits are additive, choose up to 500 highest positive incremental benefits.
    If durations or interactions differ, solve the corresponding resource-constrained allocation instead.
12. **What is wrong with treating a pending chargeback as a negative?**
    The label may arrive after the extraction date, creating right censoring and downward-biased recent fraud counts.
    Use mature cohorts or a defensible delay model and report unresolved cases.
13. **How would you compare weighted and unweighted training?**
    Tune only on training/validation data, keep a time-separated representative test, and compare decision metrics at equal capacity.
    Evaluate calibration separately and quantify uncertainty, including campaign dependence.
14. **Does better calibration prove higher prevented loss?**
    No: calibration describes probability agreement for a specified outcome, not the effect of intervention.
    Estimate action effects and costs and evaluate a feasible policy against a comparator.
15. **Why can group precision improve while each group's precision declines?**
    Group proportions may shift toward a historically higher-precision group.
    Show group counts, within-group rates and a standardized comparison before attributing improvement.

### Additional applied interview variations

Each answer states an assumption or failure boundary; all amounts and events are hypothetical.

**16. How can a model with higher recall produce lower total benefit?**
Suppose policy A finds 40 frauds using 100 reviews and policy B finds 50 using 400 reviews.
Each prevented fraud is worth 100 currency and every review costs 5 currency.
A's net benefit is 40*100-100*5=3,500 currency.
B's net benefit is 50*100-400*5=3,000 currency.
B captures ten extra frauds but spends 1,500 more on reviews for 1,000 more gross value.
The comparison assumes review causes the entire stated benefit, which must be separately supported.
If benefits or friction vary by case, a single constant unit value is insufficient.

**17. Can a classifier with no positive predictions have perfect accuracy?**
Yes, in a cohort with zero actual frauds an all-negative classifier has 100% accuracy.
Its recall is undefined because TP+FN=0.
Its precision is also undefined because TP+FP=0.
If there is one fraud in 10,000 events, accuracy is 99.99% but recall is zero.
Neither high accuracy nor a library's zero-division convention establishes useful fraud detection.
Show the denominator and identify whether no frauds or no predictions caused the undefined metric.

**18. How would you estimate loss-weighted recall?**
Define each fraud's eligible loss amount and a common observation horizon first.
Sum the loss amounts associated with detected frauds and divide by the total fraud loss amount.
For detected fraud losses of 100 and 300, with an additional missed loss of 600, the ratio is 400/1,000=40%.
The event-count recall for those same three frauds is 2/3≈66.67%.
Detected loss is not necessarily prevented loss because the intervention may not change the outcome.
Missing or censored loss amounts can bias the monetary ratio just as missing labels bias count recall.

**19. What if calibration is good overall but poor in a subgroup?**
Suppose two equally sized groups each receive probability 0.10 for every transaction.
Their actual fraud rates are 0.05 and 0.15, so the pooled rate is 0.10.
The aggregate prediction matches the aggregate rate even though both groups are miscalibrated.
Examine group-specific reliability with adequate support and mature outcomes.
Avoid treating a tiny subgroup's noisy observed fraction as definitive evidence of miscalibration.
Whether subgroup-specific action is permissible requires policy and applicable legal review.

**20. Does inverse weighting remove every audit bias?**
No: it addresses the specified sampling mechanism under correct inclusion probabilities and positive support.
If audited labels are systematically wrong, weighting reproduces a distorted target.
If unreviewed blocked transactions have unknowable counterfactual outcomes, sampling their records alone does not reveal those outcomes.
If some eligible cases have zero audit probability, a purely design-based estimate cannot represent them.
Unmeasured selection within the supposed random sample can also invalidate the weights.
Use independent adjudication, design logs and a clearly stated estimand in addition to weighting.

**21. How does a top-K threshold vary with volume?**
Assume K=100 and one event per review slot.
When 1,000 eligible events arrive, the selected fraction is 10%.
When 10,000 arrive, the selected fraction is 1%.
The numeric score boundary depends on each batch's score distribution and tie rule.
A top-K rule maintains a count limit but does not guarantee the same expected fraud risk.
A daily batch may also defer urgent events until after their actionable window.

**22. How would you handle a fixed review cost plus a fixed setup cost?**
Let each selected case consume one slot and have known benefit after per-case costs.
Suppose positive benefits are 8, 5 and 2 currency and opening a review shift costs 10.
Selecting all three gives 8+5+2-10=5 currency; selecting none gives zero.
Selecting only the top one gives 8-10=-2, so the action set interacts through setup cost.
A per-case rule that simply accepts all positive benefits misses the decision to open the shift.
Once the shift is opened, additive top-K ordering may still solve the remaining allocation under its assumptions.

**23. When does an intercept-only recalibration make sense?**
It can be appropriate when sampled and target class priors differ but class-conditional score distributions remain stable.
Under that pure prior-shift assumption, the posterior log-odds change by a constant.
A changing fraud strategy may alter the score distribution within the fraud class, violating the assumption.
Compare mature conditional score distributions and check calibration on representative target data.
A positive constant odds multiplier preserves rank but changes probability-based decisions.
The method does not solve intervention-induced missing counterfactual labels.

**24. How do you distinguish review latency from inference latency?**
Inference latency runs from scoring request to returned model score under a defined clock and request population.
Queue wait runs from actionable enqueue time to review start, and handling time ends at review completion.
A fast score can still sit in a long analyst backlog.
An alert routed after its expiry may have zero operational benefit despite low inference latency.
Measure each interval from versioned event timestamps and inspect their distributions by shift.
Do not add component p99 values and call the result the end-to-end p99.

**25. How can entity dependence alter uncertainty?**
Suppose 100 fraudulent transactions all belong to the same coordinated campaign.
They are not necessarily 100 independent examples of attacker behavior.
A random transaction bootstrap may repeatedly treat correlated events as independent evidence.
Cluster resampling by campaign or customer may better match the target dependence if those units are observed.
Time-block resampling may be more relevant when campaigns are not reliably identified.
Neither choice is automatic: explain the independence assumptions and the target deployment horizon.

**26. Is a 1% false-positive rate always acceptable?**
No: with one million legitimate transactions it implies 10,000 false alerts at that operating point.
At five minutes each, those alerts alone require 50,000 minutes, or about 833.33 reviewer-hours.
That excludes true alerts, rework, breaks and other interventions.
At 100,000 legitimate transactions the same rate implies 1,000 false alerts.
The applicable constraint therefore depends on eligible volume, review duration and the costs of mistakes.
FPR also needs reliable actual negatives from the target population.

**27. How would you compare a fixed cutoff with a top-K queue?**
Evaluate both on the same time-separated events and the same action and outcome definitions.
The fixed cutoff can produce variable workload as volume or score distributions shift.
Top K fixes a count within its window but can admit low-value events when all risks are low.
Permit fewer than K actions when incremental benefit is negative and no other constraint requires action.
Measure prevented value, alert burden, late cases and uncertainty, not only fraud precision.
If review times differ, replace count K with the relevant time or staffing constraints.

**28. Why may a mature cohort still have missing-not-at-random labels?**
The maturity horizon addresses when an outcome could arrive, not whether every event is investigated.
A model-driven queue can preferentially reveal labels for high-score events even years later.
Selection may also differ by geography, action type or customer response.
Randomized audit coverage and recorded inclusion probabilities can address some observation gaps.
No weighting strategy recovers outcomes for a zero-inclusion stratum without extra assumptions.
Intervention can additionally alter the outcome itself, which is a different problem from missing observation.

**29. What is the error in adding average costs twice?**
Suppose expected benefit is already defined as prevented loss minus 5 currency of review labor.
If that benefit is 8 currency, subtracting the same 5 again reports an incorrect value of 3.
Alternatively, starting from gross prevented loss of 13 and subtracting labor once gives 8.
State the baseline and cost scope for every utility term.
Separate fixed staffing commitments from marginal per-review labor where economically relevant.
Sensitivity analysis can vary uncertain costs without silently changing which expenses are included.

**30. What should be logged for a replayable decision?**
Record event and decision timestamps with time-zone and late-arrival semantics.
Retain the eligible population definition and the point-in-time feature versions.
Record model, calibration map, threshold, comparison operator and tie-break version.
Record the recommendation, actual action, override reason and review completion time separately.
Version label sources, outcome horizons, corrections and audit inclusion probabilities.
Protect access to sensitive data and apply retention and governance rules appropriate to the organization.
Replay parity supports debugging, but it does not establish that the policy was causally beneficial.

**31. Can average precision be read directly as precision at the deployed threshold?**
No: average precision summarizes precision across recall increments under a specified computation.
Precision at the deployed threshold uses one particular selected set and its own denominator.
Two models can exchange their relative ordering at different workload limits.
A threshold-free summary cannot substitute for the actual review budget and action utility.
Compare score versions on the same eligible cohort and report prevalence with the summary.
Implementations may differ in interpolation conventions, so name the precise metric definition.

**32. When is a cost threshold not enough even with perfect calibration?**
A calibrated fraud probability is not sufficient when loss amount or intervention efficacy differs across events.
It is also insufficient when review capacity, customer constraints or deadlines bind.
For each action compare its expected incremental effect with the alternatives under the relevant constraints.
For example, a low-probability high-loss transaction can have greater preventable value than a higher-probability low-loss one.
Calibration concerns predicted outcome frequencies; causal efficacy concerns the difference between action outcomes.
Both can be uncertain even when the score model has excellent rank discrimination.

**33. How should a team react when the audit finds no fraud?**
Zero observed positives do not prove the population fraud rate is zero.
A sample of 100 events can miss rare fraud even when fraud exists outside the sample.
The sample design, population size and dependence determine an appropriate uncertainty interval.
Report the sample count, inclusion design and observation horizon rather than a false certainty.
Increasing the sample can reduce uncertainty but cannot correct a systematically biased selection mechanism.
If a subgroup has no audit coverage, its rate remains unidentified by that audit.

**34. Why does a blocked transaction complicate ordinary confusion counts?**
A successful block may prevent the fraud outcome that would otherwise have occurred.
Calling the resulting observed nonfraud a false positive assumes the unobserved no-block outcome is known.
Calling every blocked transaction a true positive makes the opposite unsupported assumption.
Define whether the target is attempted fraud, realized loss or loss under a no-action policy.
Each target has a different measurement process and potentially different observable labels.
Estimate counterfactual loss with an appropriately designed causal study where feasible and safe.
Document which cases are excluded from metrics and how exclusions affect the target population.

**35. How should a team report uncertainty at a business threshold?**
Start with the number of selected events, observed positives and fully eligible events.
Distinguish uncertainty from sampled labels from uncertainty due to future distribution shift.
For a representative simple random audit, a binomial interval may be suitable for a stratum rate under its assumptions.
Complex stratified or clustered designs need variance calculations appropriate to that design.
For a ratio such as estimated recall, propagate uncertainty from numerator and denominator jointly.
Confidence intervals do not include unknown systematic error from a misclassified target unless modeled explicitly.
A time-based holdout gives evidence about that particular future period, not a universal guarantee.
Report economic sensitivity to both uncertain effects and uncertain costs.
Avoid presenting a single decimal place as precision beyond the evidence.
Decision owners should know whether the policy choice changes across plausible estimates.

**36. How can a randomized audit be safe and operationally useful?**
Define the target population and record the probability with which each eligible event is sampled.
The audit need not expose customers to a risky action merely to acquire a label.
For instance, a separate, authorized retrospective investigation can inspect existing evidence.
Whether that investigation can identify the target outcome depends on what evidence remains observable.
Keep audit assignment independent of unknown outcomes within the design strata.
Protect sensitive information and limit access according to organizational rules.
Record nonresponse, investigation failures and disagreements rather than silently excluding them.
Use a prespecified plan for resolving difficult or disputed cases.
If response probability depends on outcome, audit selection weights alone do not resolve the resulting missingness.
Escalate potential customer harm or legal uncertainty before executing the audit.

**37. Why should the test set reflect deployment prevalence?**
A representative test set makes direct estimates of precision and alert volume interpretable for the target population.
A case-control test can estimate TPR and FPR when sampled representatively within each class.
Its raw precision instead reflects the artificial sample prevalence.
A target-population precision estimate can be reconstructed using E2 if target prevalence and transferable rates are known.
For instance, TPR 0.8 and FPR 0.01 imply precision about 7.41% at 0.1% prevalence.
At 1% prevalence the same rates imply about 44.69% precision.
Neither reconstructed value is trustworthy if the sampled positives or negatives differ from their target classes.
Record selection probabilities and report sample counts along with adjusted metrics.
A balanced test cohort does not by itself establish a balanced production workload.

**38. What is the difference between a rule override and a model error?**
A model recommendation is an input to the policy, while the final action may incorporate a rule or human judgment.
A rule can be correct under a legal or safety requirement even if it reduces a model metric.
An override can also reflect an implementation bug, outdated guidance or information unavailable to the model.
Log the proposed action, final action and reason independently.
Evaluate the complete decision system as deployed as well as the underlying score.
Use action-specific outcomes rather than attributing every downstream result to the model.
A large override rate can invalidate an offline simulation that assumes all recommendations are followed.
Review override rates by version, shift and segment for operational investigation.
Do not remove mandatory controls simply to improve offline precision.

**39. What can go wrong with negative downsampling across time?**
If sampling probability changes by month, the training sample's apparent trends may reflect the sampling design.
If the probability differs by channel within the negative class, a single global prior correction may be inadequate.
If a random seed is reused without versioning the source cohort, the exact sample may still change.
Preserve event identifiers, inclusion probabilities and sampling logic with each training artifact.
Avoid sampling the held-out evaluation cohort unless its design is explicitly accounted for.
A temporal model can otherwise learn artifacts of the collection policy rather than fraud behavior.
Inspect distributions both before and after sampling, including small but operationally important groups.
A less aggressive or stratified sample may preserve information needed for rare legitimate behavior.
The choice should be assessed using the same mature deployment-relevant test set.

**Takeaway:** A strong answer states the estimand, carries the calculation, identifies a failure boundary and proposes evidence.

## 12. Quick Reference Card

| Required field | Reference |
|---|---|
| Topic | Imbalanced fraud learning, evaluation and capacity-constrained action. |
| Core formula | Precision = pi×TPR / [pi×TPR + (1−pi)×FPR]; the denominator must be positive. |
| Key assumption | Prevalence and class-conditional rates describe the same target cohort; label selection and maturity must be understood. |
| Complexity | Sorting N candidate scores costs O(N log N) time and O(N) storage for a materialized batch; selection and streaming alternatives have different costs. |
| Use when | Designing a fraud policy with an explicit target population, costs, action effects and review capacity. |
| Avoid when | Labels, probabilities, action effects or policy constraints are unsupported; a high score alone does not justify intervention. |
| Go deeper | Consult the cited class-imbalance and evaluation sources and the linked collection index. |

| Decision / signal | Rule | Preconditions / critical warning |
|---|---|---|
| Confusion counts | N=TP+FP+FN+TN | Same eligible, mature, observable target cohort. |
| Recall | TP/(TP+FN) | Undefined with no actual positives; selective labels can bias it. |
| Precision | TP/(TP+FP) | Undefined with no selected events; prevalence matters. |
| Base rate | PPV=pi*TPR/[pi*TPR+(1-pi)*FPR] | New-population calculation requires transferable class-conditional rates. |
| Two-action cost | Act at p>=C_FP/(C_FP+C_FN) | Calibrated target probability, specified constant costs, no capacity constraint. |
| Capacity | Choose up to K largest positive incremental benefits. | Identical slots, additive values and credible action effects. |
| Sampled posterior | Multiply sampled posterior odds by target prior odds / sampled prior odds. | Class-only sampling and stable class-conditional distributions; otherwise recalibrate with representative evidence. |
| Selective labels | Audit with known positive inclusion probabilities and use design-aware estimates. | Zero inclusion prevents nonparametric recovery; weights may have high variance. |
| Delayed outcomes | Use defined mature cohorts and versioned label-as-of dates. | Maturity alone does not cure selective observation or intervention bias. |
| Deployment | Version score, features, calibration, threshold, queue and label policy together. | Monitor actual actions, capacity, fairness, drift and evidence gaps. |
