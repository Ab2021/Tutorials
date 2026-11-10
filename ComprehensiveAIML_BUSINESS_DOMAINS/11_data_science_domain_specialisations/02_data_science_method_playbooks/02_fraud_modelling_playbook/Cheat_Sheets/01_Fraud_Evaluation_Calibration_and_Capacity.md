---
collection: "fraud_modelling_playbook"
file_id: "01"
topic: "Fraud evaluation, calibration, and capacity-constrained decisions"
level: 3
status: "draft"
version: "2.0"
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

**Purpose:** Convert a fraud score into a defensible review decision when investigators are scarce, labels arrive late, and every false alert has a real cost.
**Reader:** A fraud or risk data scientist who owns the evaluation that justifies a production policy, plus the analyst lead who must staff it.
**Outcome:** You can compute the expected net value of an alert policy, locate the capacity point where the next alert stops paying for itself, and show whether a calibration gap or a ranking gap explains a disappointing quarter.
**Scope:** Binary fraud alerts with a human review queue. Payment screening, account takeover triage, application fraud, and claims triage share the same arithmetic.
**Not covered:** Model architecture selection, feature engineering, legal or regulatory advice, and suspicious-activity reporting duties. Those are separate documents in this unit and outside it.
**Evidence:** Every formula in Sections 4 and 10 is DERIVED from stated assumptions; every numeric input in Sections 5, 9, and 11 is ILLUSTRATIVE and constructed for teaching, not measured at any institution.
**Retrieval cue:** If losses are rising while precision is stable, go to Sections 4.3, 8, and 9. If the queue is drowning, go to Sections 4.4 and 7.
**Retrieval cue:** If a new geography or channel produces many alerts and few confirmed cases, go to Section 4.2 and the prevalence-transfer entry in Section 8.
**Version:** 2.0 replaces an earlier draft that repeated boilerplate definitions; the topic, title, and role in `Cheat_Sheets/INDEX.md` are unchanged.

The organising claim of this file is narrow and testable: a fraud model is never evaluated alone.
It is evaluated as a *policy* — a score plus a threshold plus an action plus a queue — against a population at a point in time.

Three quantities get conflated in practice and must be separated:

1. **Ranking quality.** Can the model put frauds above non-frauds regardless of any probability scale?
2. **Probability quality.** Does a score of 0.24 mean roughly 24 confirmed frauds per 100 alerts of that kind?
3. **Action value.** Does acting on this alert at this moment produce more expected benefit than the review it consumes?

A model can be first-rate on (1), mediocre on (2), and still produce a policy with positive value on (3) — because thresholds can absorb scale error.
The reverse also happens: a perfectly calibrated score with poor ranking forces an unaffordable queue, because calibration tells you the mix but not where the frauds sit.
Only (3) connects to money, and (3) is where capacity enters.

```text
Weekly scored events          2,000,000
Confirmed frauds (mature)         8,000     prevalence 0.400%
Alerts sent to review             8,800     alert rate 0.440%
Alerts that are true fraud        2,112     precision 24.0%
Frauds caught                      2,112     recall 26.4%
Review hours consumed           1,349.3     at 9.2 minutes per alert
```

Those five numbers are the spine of every calculation that follows.
Each is defined in Section 3, derived in Section 4, and priced in Section 5.

The honest summary of the illustrative policy: it catches about a quarter of confirmed fraud, each alert costs 14.00 currency units of review, and the policy is worth roughly 282 thousand per week *if* the recovery assumption holds.
That conditional is doing heavy work, and Section 8 lists the six ways it fails.

**Takeaway:** Evaluate the decision population and the action, never the classifier in isolation; state prevalence, capacity, and cost assumptions before quoting any metric.

## 2. Concept Map / Mental Model

The pipeline below is the mental model for the whole file. Read it left to right: each arrow is a place where the evaluation population silently changes.

```text
scored event ---> eligible cohort ---> alert set ---> investigated set ---> labelled set
   (2.0M)          (policy filter)      (8,800)        (human action)        (mature only)
      |                 |                  |                |                     |
      |                 |                  |                |                     |
   score at t0      exclusions         threshold +        disposition         label after lag
   features at t0   (test, dup,        capacity cap       codes, notes         (mean 55 days)
                     policy-blocked)
      |                 |                  |                |                     |
      +-----------------+------------------+----------------+---------------------+
                                          |
                                          v
                              +---------------------------+
                              |   evaluation ledger       |
                              |  ranking | calibration |  |
                              |  value   | capacity    |  |
                              +---------------------------+
                                          |
                          +---------------+---------------+
                          v               v               v
                    AUC / PR curve   reliability      net-value curve
                    (ordering)       (probability)    (money vs capacity)
```

Three nested loops govern this system, and confusing them is the most common analytical error in fraud work.

**Loop 0 — the world.** Fraudsters adapt. Prevalence moves. A model trained eighteen months ago ranks a population that no longer exists.
**Loop 1 — the queue.** Reviewers work what they are given. If you send 8,800 alerts, they process 8,800 alerts with a fixed-quality disposition, and the resulting labels are a biased sample of the truth.
**Loop 2 — the reporting.** The metric you publish determines what gets optimised next quarter. Publish precision alone and the next iteration will shrink the queue to inflate it.

A useful discipline: draw the funnel with the counts written on it before touching a model.
If you cannot fill in the five numbers in Section 1 for your own programme, you are not ready to compare models — you are comparing samples of unknown composition.

### 2.1 The three-layer separation

Think of the evaluation as three layers stacked vertically, each answering a different question.

| Layer | Question | Primary diagnostic | Failure signature |
|---|---|---|---|
| Ranking | Are frauds above non-frauds? | PR-AUC at fixed capacity, precision@k | High AUC, unusable queue |
| Probability | Does 0.24 mean 24%? | Reliability curve, ECE, Brier decomposition | Threshold tuned repeatedly, drifting |
| Value | Is the action worth its cost? | Net value curve, marginal band value | Profitable-looking precision, negative cash |

The layer you report should match the layer you are deciding. A capacity decision is a value decision, and value decisions need cost inputs, not AUC.

### 2.2 Why capacity is a first-class input

Capacity is not an implementation detail appended to the analysis; it is part of the objective.
Selecting the top *k* alerts changes which errors you commit. At k = 8,800 you commit 6,688 false positives per week; at k = 17,600 you commit 14,400.
The false-positive count is not a property of the model alone — it is a property of the model *and* the cut.

This is why precision at a fixed threshold is a weak summary and precision at capacity is the strong one.
Two models with identical AUC can differ materially in precision at k = 8,800, and that difference is the only one the operations team can feel.

### 2.3 The evaluation ledger

Keep one row per evaluation run with these fields, because arguments about numbers are usually arguments about cohorts.

```text
run_id | population | window | score_version | threshold | alerts | tp | fp | fn
week-41 | live events | 2026-10-05..2026-10-11 | m7 | 0.081 | 8800 | 2112 | 6688 | 5888
label_state | maturity_days | cost_version | value_currency | net_value
partial     | 90            | costs-2026-09 | USD-equivalent | 281,776
```

The `label_state` column is the one people omit, and it is the one that invalidates comparisons.
A run labelled `partial` at 90 days is missing roughly 19.5% of the frauds that will eventually be confirmed, so its precision estimate is biased low by construction.

## 3. Foundational Definitions

Definitions are grouped so that a reader can stop after the group they need. Each entry states the term, what it measures, and the nearest thing it is confused with.

### 3.1 Population and timing

**Scored event.** A single unit presented to the model at a decision time t0: one authorisation, one application, one claim, one session.
Its identity is the event key plus t0. Aggregating events into customers before scoring changes the population, so it must be declared.

**Decision-time snapshot.** The set of features legally and practically available at t0.
Anything written after t0 — a chargeback, a case note, a device blacklist update — is unavailable at decision time and contaminating if used.
The boundary is the timestamp on the source system, not the timestamp on your extract.

**Eligible cohort.** Scored events that the policy is allowed to act on, after exclusions.
Exclusions typically include test traffic, internal accounts, duplicates from retries, and events already blocked by a deterministic rule.
The eligible cohort is the denominator of every prevalence statement; changing it changes prevalence without changing the world.

**Mature label.** A confirmed outcome that has passed its observation window: the chargeback was received, the case was closed as fraud, or the dispute window expired without a dispute.
Until maturity, the event is *unlabelled*, not *negative*. Treating unlabelled as negative is the single most common evaluation defect in this domain.

**Maturity curve.** The cumulative share of eventual positives that are observable by day t after the event.
With a mean confirmation lag of 55 days assumed exponential, maturity is 42.0% at 30 days, 66.4% at 60 days, and 80.5% at 90 days.
The exponential form is an assumption to be checked against the observed lag distribution, not a law.

**Observation window.** The fixed number of days after t0 during which an outcome is counted.
A 90-day window and a 120-day window produce different recall for the same model on the same events, so the window belongs in every metric's name.

**Event time versus processing time.** Event time is when the transaction happened; processing time is when the warehouse row was written.
Mixing them produces features that look available at t0 but were backfilled later. Always join on event time and filter on processing time separately.

### 3.2 Scores, ranks, and thresholds

**Score.** The model output. It may be a raw margin, a rank, or a probability; only the third supports statements about rates.
Write down which one you have before computing anything else.

**Threshold.** The cut value that maps a score to an alert. A threshold of 0.081 on a calibrated score is interpretable as an expected-value cut; a threshold of 0.61 on a raw margin is not.
Thresholds are policy, not model parameters, and they should be owned by the same people who own the queue.

**Alert.** An event that crossed the threshold and was routed to a human or an automated action.
Alerts, not predictions, consume capacity. An event the model scores 0.99 but that never enters the queue because of a deduplication rule is not an alert.

**Alert rate.** Alerts divided by eligible events. Here 8,800 / 2,000,000 = 0.440%.
Alert rate is the operational lever: it is what a reviewer feels, and it is capped by staffing, not by model quality.

**Capacity.** The maximum number of alerts the review function can process to its standard quality within the period.
Capacity is measured in alerts per week and, more honestly, in *review minutes per week* divided by *minutes per alert*.

**Precision at capacity.** True positives divided by alerts actually worked, evaluated at the capacity-limited cut rather than at a globally optimal cut.
This is the headline number for a queue, and it is 2,112 / 8,800 = 0.240 here.

**Recall at capacity.** True positives divided by all mature frauds in the cohort. Here 2,112 / 8,000 = 0.264.
Recall and precision at capacity trade off against each other along the same curve; quoting one without the other describes a point as if it were a curve.

**False positive rate (FPR).** False alerts divided by legitimate events: 6,688 / 1,992,000 = 0.003357.
FPR is small in absolute terms and large in count terms precisely because the legitimate class is 249 times bigger than the fraud class.

**Lift.** Precision divided by prevalence. At 0.240 against 0.004, lift is 60.0 on the alerted set.
Lift is useful for communication and dangerous for decisions: a high lift on a tiny alert set can still miss most of the money.

### 3.3 Probability quality

**Calibration.** The agreement between predicted probability and observed frequency within score bands.
A score of 0.24 is calibrated if events scored near 0.24 contain 24% confirmed frauds, within sampling error.

**Reliability curve.** Observed fraud rate plotted against mean predicted probability per bin.
The 45-degree line is perfect calibration; deviations above the line mean under-prediction, below mean over-prediction.

**Expected calibration error (ECE).** Weighted mean absolute deviation between bin predicted and bin observed, weighted by bin population.
For the six-bin illustrative audit in Section 5, ECE is 0.000470 against a base rate of 0.003600 — a 13% relative gap concentrated in the top bin.

**Brier score.** Mean squared error of the probability. It decomposes into reliability, resolution, and uncertainty, which is why it can rank models that ECE cannot.
The illustrative audit yields 0.003491 against an uncertainty term of 0.003587: the model adds very little beyond the base rate in squared-error terms.

**Log loss.** Mean negative log likelihood. It punishes confident errors harder than Brier and is the natural loss for probability training.
Here 0.019072 against a base-rate benchmark of 0.023850: skill of 20.0%.

**Proper scoring rule.** A loss that is minimised in expectation by reporting the true probability. Brier and log loss are proper; accuracy at a fixed threshold is not.
Any metric you optimise directly will be gamed; this is why threshold-free evaluation matters.

**Discrimination.** The ability to separate classes, independent of scale: AUC, PR-AUC, and the shape of the precision-recall curve.
Discrimination is invariant to monotone transforms, so it cannot detect a calibration break at all.

**Sharpness.** How far predictions spread from the base rate. A model that predicts 0.004 for everything is perfectly calibrated and useless; sharpness is the counterweight to calibration.

### 3.4 Action and economics

**Action.** What happens after an alert: block, hold, contact the customer, request documents, open a case, or close with no action.
The action determines the cost and the benefit, so a metric that ignores the action cannot rank policies.

**Loss given fraud (L).** The amount lost when a fraud succeeds without intervention, in currency. Here 320 per event.
L is a distribution, not a constant; using the mean is an approximation that fails when the tail dominates, which Section 4.4 examines.

**Recovery rate (rho).** The share of L that is actually avoided when the alert is raised in time. Here 0.62.
Recovery below 1 is the norm: some losses are already irreversible at decision time, and some cases are recovered only partially.

**Value per true positive.** rho times L = 198.40. This is the benefit side of every threshold calculation and the number most often overstated in business cases.

**Review cost (c_a).** Fully loaded cost of working one alert: analyst minutes, tooling, supervision, and queue overhead. Here 14.00 per alert.
At 9.2 minutes per alert and 52 per hour, labour alone is 7.97, so 14.00 implies roughly 6.03 of tooling and supervision per alert.

**Marginal review cost (c_m).** The cost of the *next* alert when capacity is already staffed: usually the variable portion only. Here 11.07.
Average cost and marginal cost diverge because staffing is lumpy; Section 4.3 shows why using the wrong one moves the optimal threshold.

**Friction cost (c_f).** Cost imposed on a legitimate customer who is wrongly alerted: contact handling, delay, attrition risk. Here 2.10.
Friction is the cost that compliance reviews forget and customers remember.

**Expected value of an alert.** p times rho times L, minus c_a, minus (1 - p) times c_f, where p is the calibrated probability that this alert is fraud.
Setting this to zero solves for the break-even probability, which is the only threshold with an economic meaning.

**Break-even probability (p-star).** (c_a + c_f) / (rho L + c_f) = 16.10 / 200.50 = 0.0803.
Above p-star an alert is worth working; below it, closing without review is cheaper. The margin is thin, which is why the assumptions matter.

**Net value.** Sum over alerts of expected value, expressed per period. Here 281,776 per week under the stated assumptions.
Net value is the only metric that combines ranking, calibration, capacity, and cost; it is also the most assumption-sensitive.

**Marginal band value.** The net value contributed by the next tranche of capacity, computed from the precision *within that tranche* rather than cumulatively.
Cumulative precision can stay healthy while the marginal band is already negative; the illustrative frontier turns negative in the 21,000-alert band.

### 3.5 Evaluation design

**Temporal holdout.** A test period strictly after the training period, with a gap at least as long as the label lag.
Random splits leak future information through lagged labels and through fraud-ring overlap; temporal splits approximate deployment.

**Cohort contamination.** The presence of post-decision information in features, or of unresolved events coded as negatives.
Contamination inflates offline metrics and produces the classic gap between backtest and production.

**Label selection bias.** The labels you observe depend on the actions you took. Alerts get investigated and resolved; non-alerts rarely do.
A model evaluated only on alerted events measures agreement with the previous policy, not accuracy against truth.

**Feedback loop.** The policy changes the data that trains the next model. Blocking a segment removes its outcomes from the labelled set, so the next model learns less about it.
Loops are manageable only if you reserve a small random hold-out that is never actioned.

**Guardrail metric.** A metric that must not degrade even if the primary metric improves: complaint rate, contact rate, time-to-decision, share of alerts closed as "no issue".
Guardrails are how a value-optimised policy stays inside its operational and regulatory envelope.

**Decision-relevant metric set.** The minimum set that fully determines the policy: prevalence, TPR at the cut, FPR at the cut, value per true positive, and costs.
Any dashboard missing one of these is describing a policy it cannot defend.
## 4. Core Formulas with Full Derivation

Section 4 is deliberately arithmetic-first. Each result is presented in five parts — FORMULA, WHERE, CONSTRAINT, NUMERICAL EXAMPLE, INTERPRETATION — and closes with an explicit validity condition. If you cannot state the validity condition, you do not yet own the result.

### 4.1 Base case

**Result 1 — posterior precision from a confusion matrix (Bayes).**

```text
FORMULA
  PPV = (pi * TPR) / (pi * TPR + (1 - pi) * FPR)

WHERE
  pi  = prevalence of fraud among eligible events, dimensionless, 0 < pi < 1
  TPR = P(alert | fraud), dimensionless, 0..1        (recall, sensitivity)
  FPR = P(alert | legitimate), dimensionless, 0..1   (1 - specificity)
  PPV = P(fraud | alert), dimensionless, 0..1        (precision)

CONSTRAINT
  The three inputs must be measured on the same eligible cohort and the same
  observation window. Mixing a 90-day TPR with a 120-day FPR breaks the identity.

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  pi = 0.004, TPR = 0.55, FPR = 0.01
  numerator   = 0.004 * 0.55          = 0.002200
  denominator = 0.002200 + 0.996*0.010 = 0.002200 + 0.009960 = 0.012160
  PPV         = 0.002200 / 0.012160    = 0.18095
  lift        = 0.18095 / 0.004        = 45.2

INTERPRETATION
  A model that catches 55% of fraud while flagging 1% of legitimate traffic still
  produces a queue that is 82% false. The false positives number 9,960 per million
  legitimate events against 2,200 per million frauds; the ratio, not the rate, sets
  precision. Precision is a property of prevalence as much as of the model.
```

Validity condition: the identity holds exactly for any cohort with a well-defined TPR and FPR, but it is only *useful* when FPR is estimated on at least a few hundred false alerts, because FPR's own sampling error propagates multiplicatively into PPV.

**Result 2 — expected value of a single alert.**

```text
FORMULA
  EV(p) = p * rho * L - c_a - (1 - p) * c_f

WHERE
  p    = calibrated P(fraud | alert), dimensionless
  rho  = recovery rate given timely action, 0..1
  L    = loss given undetected fraud, currency per fraud event
  c_a  = review cost per alert, currency
  c_f  = friction cost per false alert, currency
  EV   = expected net value per alert, currency

CONSTRAINT
  p must be a probability, not a rank or margin. If p is monotone-but-uncalibrated,
  EV(p) is a monotone transform of nothing economic; only its ordering survives.

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  p = 0.24, rho = 0.62, L = 320, c_a = 14.00, c_f = 2.10
  benefit  = 0.24 * 0.62 * 320 = 47.616
  review   = 14.000
  friction = 0.76 * 2.10       =  1.596
  EV       = 47.616 - 14.000 - 1.596 = 32.02 per alert

INTERPRETATION
  At 24% precision an alert is worth 32 currency units in expectation. Multiply by
  8,800 alerts and you get 281,776 per week — the same figure the whole file builds
  toward. Note the friction term is only 5% of the benefit here; in low-value
  segments it dominates and flips the sign (see Section 8, failure 3).
```

Validity condition: EV is linear in p, which requires the benefit and cost terms to be independent of p. If high-risk alerts also take longer to review — true in practice — the linear form overstates value at the top of the queue.

**Result 3 — break-even probability.**

```text
FORMULA
  p* = (c_a + c_f) / (rho * L + c_f)

DERIVATION
  Set EV(p) = 0 and solve:
    p*rho*L - c_a - (1 - p)c_f = 0
    p*(rho*L + c_f) = c_a + c_f
    p* = (c_a + c_f) / (rho*L + c_f)

WHERE
  all symbols as in Result 2; p* in 0..1, currency units cancel

CONSTRAINT
  Requires rho*L + c_f > 0, always true for positive economics. Requires the
  friction to be paid on false alerts only; if friction also accrues to true
  positives (customer contacted, card replaced) the denominator changes.

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  c_a = 14.00, c_f = 2.10, rho = 0.62, L = 320
  rho*L + c_f = 198.40 + 2.10 = 200.50
  p* = 16.10 / 200.50 = 0.08030

  Check by substitution at p = 0.0803:
    0.0803 * 198.40 = 15.931
    15.931 - 14.00 - 0.9197*2.10 = 15.931 - 14.00 - 1.931 = 0.000  (rounding)

INTERPRETATION
  Only alerts with a calibrated probability above 8.03% are worth investigating.
  Anything below should be closed without review — not because the model is wrong,
  but because the review costs more than the expected recovery. The threshold is a
  business number that happens to be expressed in probability units.
```

Validity condition: p* assumes a binary review decision with a fixed review cost. With tiered actions (Result 7) the single threshold is replaced by a set of thresholds, one per action.

**Result 4 — net value of a capacity-limited policy.**

```text
FORMULA
  V(a) = TP(a) * rho * L - a * c_a - (a - TP(a)) * c_f
  TP(a) = a * precision(a) = a * PPV at the a-th ranked alert

WHERE
  a      = alerts worked per period (capacity used), integer >= 0
  TP(a)  = true positives among them
  others as above

CONSTRAINT
  V(a) is defined only along the model's own ranking; comparing V across models
  requires each model to be evaluated at its own best a, not at a common a.

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  a = 8,800, TP = 2,112, rho = 0.62, L = 320
  benefit  = 2,112 * 198.40        = 419,020.8
  review   = 8,800 * 14.00         = 123,200.0
  friction = (8,800 - 2,112) * 2.10 = 14,044.8
  V        = 419,020.8 - 123,200.0 - 14,044.8 = 281,776.0

INTERPRETATION
  V is the objective. Precision and recall are inputs to it. A model with better
  precision at the same a always improves V by the difference in TP times 200.50,
  minus the friction saved at 2.10 per displaced false alert.
```

Validity condition: V assumes independent alerts. When 6% of alerts are duplicates of the same underlying case (Section 8, failure 4), the benefit is overcounted because one fraud is credited multiple times while its review cost is paid multiple times.

### 4.2 General case

The general case replaces "one threshold" with "one capacity plus one ranking", which is what operations actually constrains.

**Result 5 — marginal band value and the optimal capacity.**

```text
FORMULA
  For a band of alerts between cumulative counts a1 and a2 with true positives
  TP1 and TP2, define the band marginal precision:
      mp = (TP2 - TP1) / (a2 - a1)
  The band contributes:
      V_band = (a2 - a1) * (mp * rho * L - c_m - (1 - mp) * c_f)
  Optimal capacity a* is the largest a such that V_band > 0 for the marginal band.

WHERE
  c_m = marginal review cost (variable portion), currency per alert
  mp  in 0..1; bands are contiguous segments of the ranked alert list

CONSTRAINT
  Requires c_m <= c_a. Using average cost in the marginal formula systematically
  under-expands capacity, because it charges fixed supervision to each extra alert.

NUMERICAL EXAMPLE  (ILLUSTRATIVE, cumulative frontier)
  a=8,800  TP=2,112  precision 0.2400   V=281,776
  a=11,000 TP=2,450  precision 0.2227   V=314,125
  a=14,000 TP=2,800  precision 0.2000   V=336,000
  a=17,600 TP=3,200  precision 0.1818   V=358,240
  a=21,000 TP=3,360  precision 0.1600   V=335,580

  Band 8,800 -> 11,000: n=2,200, dTP=338, mp=0.15364
     0.15364*198.40 = 30.482
     - 11.07 - 0.84636*2.10 = -11.07 - 1.777 = 30.482 - 12.847 = 17.63 > 0
     band value = 2,200 * 17.63 = 38,795  (matches 314,125 - 281,776 = 32,349
     plus rounding in the cumulative TP figures; recompute from raw counts when
     reconciling, never from rounded precision)

  Band 17,600 -> 21,000: n=3,400, dTP=160, mp=0.04706
     0.04706*198.40 = 9.337
     - 11.07 - 0.95294*2.10 = 9.337 - 13.071 = -3.73 < 0

INTERPRETATION
  Cumulative value peaks at 17,600 alerts and declines after. The band arithmetic
  explains why: beyond 17,600, each extra alert brings 4.7% precision, worth 9.34,
  against 13.07 of marginal cost. Expanding from 8,800 to 11,000 is worth about
  17.63 per alert; expanding from 21,000 to 26,000 destroys about 1.94 each.
  A programme that reports only cumulative precision would expand past the peak.
```

Reconciliation note: cumulative differences computed from rounded precision (32,349) differ from band sums computed from raw counts (38,795) because the cumulative TP figures were themselves rounded in the table. In production always store integer counts and derive precision from them.

Validity condition: the band test assumes bands are processed in rank order and that capacity is divisible. With hiring in whole FTEs (195.65 alerts per FTE-week) the practical a* is the largest multiple of the staffing quantum below the true optimum.

**Result 6 — Lagrangian form: one ranking, many action costs.**

```text
FORMULA
  Choose the cut tau to maximise  sum_i [ p_i * rho * L - c_a - (1-p_i)*c_f ] * 1[alert_i]
  subject to                     sum_i 1[alert_i] <= K
  Lagrangian:  alert i iff  p_i * (rho*L + c_f) > c_a + c_f + lambda
  giving the capacity-adjusted threshold
      p*(lambda) = (c_a + c_f + lambda) / (rho*L + c_f)

WHERE
  K        = capacity in alerts per period
  lambda   = shadow price of capacity, currency per alert, >= 0
  p*(lambda) rises monotonically in lambda

CONSTRAINT
  Requires the constraint to bind. If K is generous relative to the score
  distribution, lambda = 0 and the unconstrained threshold p* applies.

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  Unconstrained p* = 0.0803. At a = 8,800 the observed cut is 0.081, so the
  implied lambda is small:
      lambda = 0.081 * 200.50 - 16.10 = 16.24 - 16.10 = 0.14 per alert
  If capacity were halved to 4,400, the cut would move up to roughly 0.145 on
  this ranking, and lambda would rise to 0.145*200.50 - 16.10 = 12.97.

INTERPRETATION
  The shadow price is a useful management number: it says what one more unit of
  reviewer capacity is worth at the current margin. Here 0.14 per alert means the
  queue is almost exactly right-sized — capacity is marginally binding, not
  painfully so. A lambda of 13 would justify hiring.
```

Validity condition: the Lagrangian is exact for a single linear constraint and separable objective; it is approximate when review cost depends on alert complexity (see Result 7), because then the objective is no longer separable in the same way.

**Result 7 — tiered actions.**

```text
FORMULA
  For actions k = 1..K with cost c_k and avoided-loss fraction rho_k, choose for
  alert i the action maximising
      argmax_k  [ p_i * rho_k * L - c_k - (1 - p_i) * c_f,k ]
  and choose "no action" (k=0, all terms zero) if every k gives a negative value.

WHERE
  rho_k is non-decreasing in k, c_k and c_f,k are non-decreasing in k
  boundaries between actions occur where two value lines cross:
      p_k = (c_{k+1} - c_k) / (L*(rho_{k+1} - rho_k))

CONSTRAINT
  Requires the action set to be ordered by intensity. Unordered actions
  (block vs. contact) can have crossing lines twice and need explicit enumeration.

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  Action A: auto-decline, c=0.30, rho=0.20, c_f=1.10
  Action B: analyst review, c=14.00, rho=0.62, c_f=2.10
  Action C: review + customer contact, c=26.50, rho=0.78, c_f=6.40

  A vs B boundary: (14.00-0.30) / (320*(0.62-0.20)) = 13.70 / 134.40 = 0.1019
  B vs C boundary: (26.50-14.00) / (320*(0.78-0.62)) = 12.50 / 51.20 = 0.2441

  So: p < 0.0215 -> no action; 0.0215..0.1019 -> auto-decline;
      0.1019..0.2441 -> analyst review; p > 0.2441 -> review plus contact.
  (0.0215 is p* for action A alone: (0.30+1.10)/(0.20*320+1.10) = 1.40/65.10.)

INTERPRETATION
  Tiering raises value in both tails: the very low band is handled by automation
  at 0.30 instead of 14.00, and the very high band buys the extra recovery that
  only customer contact achieves. The queue is then sized on the middle band only,
  which is a fraction of total alerts — the capacity problem shrinks without
  losing coverage.
```

Validity condition: tier boundaries are computed from mean L. With heavy-tailed L the crossing points shift per alert, and a segment-level threshold is the practical compromise.

### 4.3 Variants

**Variant A — average versus marginal cost in the threshold.**
With average cost 14.00 the threshold is 0.0803; with marginal cost 11.07 it is 0.0657.

```text
  p*_avg = (14.00 + 2.10) / 200.50 = 0.08030
  p*_mrg = (11.07 + 2.10) / 200.50 = 0.06569
  logit difference = ln(0.0803/0.9197) - ln(0.06569/0.93431)
                   = -2.4387 - (-2.6552) = +0.2165
```

A logit shift of 0.2165 is small in probability terms at the bottom of the scale and large at the top: at p = 0.05 it moves the score by about 0.010 in probability, at p = 0.60 by about 0.043.
The practical rule: use p*_avg when deciding whether the *function* should exist, p*_mrg when deciding whether to work *one more* alert this week.

**Variant B — expected-loss minimisation instead of value maximisation.**
Minimising expected loss per event gives the same threshold only under a symmetric-accounting assumption: loss if missed = rho L, loss if falsely flagged = c_f plus the review cost allocated to that alert. The two formulations coincide when review cost is charged per alert regardless of outcome. When review cost is charged per *hour* and fraud alerts take longer, the value form is the correct one and the loss form silently under-weights high-risk alerts.

**Variant C — precision at a fixed alert rate instead of a fixed threshold.**
If the queue is capped, express everything as a function of alert rate r: precision(r), recall(r), V(r). The threshold then floats with score distribution drift, which is usually the desired behaviour — the queue stays constant and the model quality shows up as varying precision. The failure mode is that a drifting threshold hides model degradation: precision stays flat because the threshold silently loosens. Always publish both the threshold and the alert rate.

**Variant D — cost-weighted recall.**
For regulatory coverage targets the objective is often "catch at least X% of value", not "maximise net value". Value-weighted recall uses L_i per event rather than a count:

```text
  R_value(a) = sum_{i in caught, ranked <= a} L_i / sum_{all frauds} L_i
```

With a heavy tail, R_value can be much lower than count recall at the same a — a programme reporting 26.4% count recall may be catching 18% of value if the largest cases sit below the cut. Section 8, failure 5 develops this.

**Variant E — threshold on the logit for model portability.**
Thresholds on probability are not portable across prevalence shifts; thresholds on the logit are portable *if* the shift is a constant offset. If a new geography has half the prevalence, the logit shifts by ln(0.002/0.998) - ln(0.004/0.996) = -6.215 - (-5.517) = -0.698. Recomputing the operating point means shifting the logit threshold by +0.698, not halving the probability.

```text
  p1 = 0.004  -> logit = -5.5175
  p2 = 0.002  -> logit = -6.2146
  offset = -0.6971
```

Applying a naive "half the threshold" rule (0.0803 -> 0.0402) instead of the logit shift (which gives 0.0403 at the same relative economics) happens to be close here because both p are small; at p above 0.2 the two rules diverge sharply and the naive rule is wrong.

### 4.4 Limits and counterexamples

**Limit 1 — heavy-tailed loss breaks the mean.**
If L is lognormal with median 90 and 1% of cases above 3,000, the mean may be 320 while the median case is worth far less than the threshold implies. Then p* computed from mean L under-alerts on the tail and over-alerts on the body: a case at median loss has true p* of (16.10)/(0.62*90+2.10) = 16.10/57.90 = 0.278, not 0.0803.

```text
  Counterexample: two alerts, both p = 0.12
    Alert X: L = 90    EV = 0.12*0.62*90  - 14.00 - 0.88*2.10 = -9.15
    Alert Y: L = 3,000 EV = 0.12*0.62*3000 - 14.00 - 0.88*2.10 = 207.34
  Mean-L arithmetic says both are worth 0.12*198.40 - 16.10 = +7.71, which is
  wrong for X by 16.86 and wrong for Y by 199.63.
```

The correction is to segment by expected loss band, or to use a per-alert EV and rank by EV rather than by p. Ranking by p when L varies is a known and common error.

**Limit 2 — prevalence non-transportability.**
PPV is not a property of the model. Using TPR = 0.264 and FPR = 0.003357 (the illustrative operating point) on populations with different prevalence:

```text
  pi = 0.0005 -> PPV 0.0378, EV  -8.51 per alert
  pi = 0.0010 -> PPV 0.0730, EV  -1.47
  pi = 0.0020 -> PPV 0.1361, EV +11.19
  pi = 0.0040 -> PPV 0.2400, EV +32.02
  pi = 0.00667-> PPV 0.3455, EV +53.18
  pi = 0.0200 -> PPV 0.6161, EV +107.42
```

At pi = 0.001 the same model, same threshold, same reviewers produce a *negative*-value queue. A model validated on one channel and deployed on another without re-deriving p* is not a validated model.

**Limit 3 — calibration is not monotone-invariant, discrimination is.**
Any strictly increasing transform of the score leaves AUC and PR-AUC unchanged while changing every probability-based threshold. Therefore: an AUC regression test cannot catch a calibration break, and a calibration test cannot catch a ranking break. A monitoring plan with only one of the two is blind to half the failure surface.

**Limit 4 — the break-even threshold is undefined when friction is paid on true positives.**
If every alert triggers a customer contact costing 2.10 regardless of outcome, then c_f applies to all alerts and cancels from both sides:

```text
  EV = p*rho*L - c_a - c_f
  p* = (c_a + c_f) / (rho*L) = 16.10 / 198.40 = 0.0811
```

versus 0.0803 in the base case — a 1% shift. Small, but the *structure* changes: the threshold no longer depends on c_f at all, and the sensitivity analysis must be redone. The general lesson: write down which costs attach to which outcome before differentiating.

**Limit 5 — capacity constraints make the objective non-separable across segments.**
If a large case takes 3 hours and a small one 5 minutes, a single alert-count capacity is wrong. Replace it with minutes: capacity 1,349.3 minutes per week becomes a knapsack, and the greedy value-per-minute ordering (not value-per-alert) is optimal only under the fractional relaxation. The integer version is NP-hard and in practice solved by value-per-minute sorting with a correction for the last few large cases.

```text
  value per minute:  EV_i / minutes_i
  small case: 32.02 / 5.0  = 6.40 per minute
  large case: 240.00 / 180 = 1.33 per minute
  A thousand small alerts beat sixty large ones on the same clock, even though
  the large ones carry more value apiece. Capacity in alerts hides this.
```

The counterexample that breaks value-per-minute: when a large case's loss is recoverable only if worked *today*, its option value exceeds the static EV, and strict value-per-minute ordering liquidates the tail. Treat time-critical cases as a separate queue with reserved capacity.
## 5. Worked Numerical Example

A complete pass through one week of a payment-screening programme, from raw counts to a defensible policy recommendation. All inputs are ILLUSTRATIVE; every arithmetic step is shown so a reader can substitute their own numbers.

### 5.1 The input set

```text
INPUT BLOCK (ILLUSTRATIVE)
  window                2026-10-05 .. 2026-10-11 (7 days, event time)
  eligible events       2,000,000      test/internal/duplicate traffic removed
  mature frauds           8,000        labels observed at >= 90 days
  alert threshold         0.081         calibrated probability cut
  alerts raised           8,800
  alerts confirmed fraud  2,112
  loss given fraud L        320.00      currency per confirmed fraud
  recovery rate rho          0.62       share avoided given timely action
  review cost c_a           14.00       currency per alert, fully loaded
  friction cost c_f          2.10       currency per false alert
  marginal cost c_m         11.07       currency per additional alert
  minutes per alert          9.20       mean, all alert types
  reviewer minutes/week    1,800.00     30 hours x 60, per FTE
```

### 5.2 Step 1 — build the confusion matrix before any ratio

```text
  TP = 2,112                 alerts that were confirmed fraud
  FP = 8,800 - 2,112 = 6,688 alerts that were not
  FN = 8,000 - 2,112 = 5,888 frauds never alerted
  TN = 1,992,000 - 6,688 = 1,985,312 legitimate events not alerted
  
  check: TP + FP + FN + TN = 2,112 + 6,688 + 5,888 + 1,985,312 = 2,000,000 ✓
```

The row check is not ceremony. In practice one of the four cells is derived from a different extract than the others, and the mismatch is how you discover it.

```text
  TPR = TP / (TP + FN) = 2,112 / 8,000      = 0.26400
  FPR = FP / (FP + TN) = 6,688 / 1,985,312  = 0.003369
  PPV = TP / (TP + FP) = 2,112 / 8,800      = 0.24000
  NPV = TN / (TN + FN) = 1,985,312/1,991,200 = 0.997044
```

Note the FPR recomputed against TN rather than against all legitimate events. The two differ by 6,688 in the denominator and give 0.003357 versus 0.003369 — a 0.36% relative difference that becomes visible when you multiply by millions of events. Always state which denominator you used.

### 5.3 Step 2 — price one alert

```text
  value per TP  = rho * L        = 0.62 * 320.00 = 198.40
  EV per alert  = PPV*198.40 - 14.00 - (1-PPV)*2.10
                = 0.24*198.40 - 14.00 - 0.76*2.10
                = 47.616 - 14.000 - 1.596
                = 32.020
  weekly value  = 8,800 * 32.020 = 281,776.0
```

Independent re-check by the other route:

```text
  benefit  = 2,112 * 198.40 = 419,020.80
  review   = 8,800 *  14.00 = 123,200.00
  friction = 6,688 *   2.10 =  14,044.80
  net      = 419,020.80 - 123,200.00 - 14,044.80 = 281,776.00 ✓
```

Two independent arithmetic paths agreeing to the cent is the minimum standard for a number that will appear in a business case.

### 5.4 Step 3 — express value per unit of the scarce resource

Capacity is not alerts; it is minutes. The conversion matters because it changes which segment looks attractive.

```text
  minutes consumed = 8,800 * 9.20          = 80,960 minutes
  hours            = 80,960 / 60           = 1,349.3 hours
  FTE equivalent   = 1,349.3 / 30          = 44.98 FTE
  value per hour   = 281,776 / 1,349.3     = 208.83
  value per minute = 208.83 / 60           = 3.48
  value per FTE-week = 281,776 / 44.98     = 6,264.5
```

Sanity check against the marginal economics: an FTE costs 30 hours at 52 per hour = 1,560 per week.
Value per FTE-week of 6,264.5 against 1,560 of cost is a 4.0x ratio — consistent with the earlier 3.05x return on total review spend, since the FTE cost line excludes tooling. If the two ratios disagreed by more than 30%, one of the inputs would be wrong.

```text
  3.05 = 419,020.80 / 137,244.80   (benefit over full review spend)
  4.02 = 6,264.5   / 1,560.0       (value per FTE over wage cost)
```

### 5.5 Step 4 — locate the threshold that the current policy implies

The published cut is 0.081. What break-even probability does that correspond to, and how much slack does it leave?

```text
  p* implied by costs   = (14.00 + 2.10) / (198.40 + 2.10) = 16.10/200.50 = 0.08030
  actual cut            = 0.08100
  slack in probability  = 0.00070  (0.87% above break-even)
  slack in value        = 0.00070 * 200.50 = 0.140 per alert
```

A cut 0.14 currency units above break-even means the marginal alert is barely worth working. That is a defensible place to sit — it means capacity and value are balanced — but it also means any of the six cost assumptions moving by more than a fraction flips the sign. Section 8 treats each assumption as a failure mode for exactly this reason.

### 5.6 Step 5 — value the current policy against two references

```text
  Reference A: no screening at all
      loss = 8,000 * 320.00 = 2,560,000 per week
      value of screening relative to nothing = 281,776 (net) versus -2,560,000
  
  Reference B: oracle ranking, same capacity
      the best 8,800 alerts contain 8,000 frauds and 800 non-frauds
      benefit  = 8,000 * 198.40 = 1,587,200
      review   = 8,800 *  14.00 =   123,200
      friction =   800 *   2.10 =     1,680
      net      = 1,462,320
  
  gap to oracle = 1,462,320 - 281,776 = 1,180,544 per week
```

The gap to oracle is the honest measure of headroom. It says that better *ranking* is worth up to 1.18 million per week at this capacity — an order of magnitude more than any plausible cost saving. Conversely, better *calibration* alone is worth almost nothing here, because the threshold can absorb a monotone scale error. This single comparison should determine where the modelling budget goes.

### 5.7 Step 6 — audit calibration on a stratified sample

A 200,000-event random sample, stratified across score bands, with mature labels:

```text
BAND       n        mean pred p   observed frauds   observed rate   deviation
0-0.001   120,000     0.00050            40           0.000333      -0.000167
0.001-005  50,000     0.00200           130           0.002600      +0.000600
0.005-015  20,000     0.00800           180           0.009000      +0.001000
0.015-040   8,000     0.02500           200           0.025000       0.000000
0.040-120   1,800     0.07000           140           0.077778      +0.007778
0.120+        200     0.20000            30           0.150000      -0.050000
TOTAL     200,000                      720           0.003600
```

```text
  ECE = sum_b (n_b/N) * |p_b - o_b|
      = 0.600*0.000167 + 0.250*0.000600 + 0.100*0.001000
      + 0.040*0.000000 + 0.009*0.007778 + 0.001*0.050000
      = 0.000100 + 0.000150 + 0.000100 + 0 + 0.000070 + 0.000050
      = 0.000470
  relative ECE = 0.000470 / 0.003600 = 13.1%
```

Brier decomposition on the same table:

```text
  reliability  = sum (n_b/N)(p_b - o_b)^2        = 0.0000033
  resolution   = sum (n_b/N)(o_b - o_bar)^2      = 0.0000988
  uncertainty  = o_bar(1 - o_bar)                = 0.0035870
  Brier        = reliability - resolution + uncertainty
               = 0.0000033 - 0.0000988 + 0.0035870 = 0.0034915
  Brier skill  = 1 - Brier/uncertainty = (resolution - reliability)/uncertainty
               = 0.0000955 / 0.0035870 = 0.0266   (2.66%)
```

Log loss on the same bins:

```text
  log loss = sum (n_b/N) * [ -o_b ln p_b - (1-o_b) ln(1-p_b) ]
           = 0.019072
  base-rate benchmark = -o_bar ln o_bar - (1-o_bar) ln(1-o_bar) = 0.023850
  log-loss skill = 1 - 0.019072/0.023850 = 0.2004   (20.0%)
```

The two skills disagree by a factor of 7.5, and the disagreement is informative rather than a mistake. Brier skill is dominated by the high-probability band where the model over-predicts (0.20 predicted, 0.15 observed): squared error there is 0.0025 per case in that bin, which swamps everything the low bands contribute. Log loss is dominated by confidence in the low bands: predicting 0.0005 where the truth is 0.0036 earns a large per-case reward on 60% of the population. Neither score is "right"; they answer different loss questions. Report both, and say which one the decision depends on.

The top bin is the one to fix: 200 events, 30 confirmed, observed 0.150 against predicted 0.200. Its binomial standard error is sqrt(0.15*0.85/200) = 0.0252, so the 0.050 gap is about 2.0 standard errors — mildly significant, and consistent with a model that is over-confident where data is scarcest. A monotone recalibration fitted on 200 events would overfit; the correct action is to wait for more labelled volume in that band or to pool it with the 0.040-0.120 band.

### 5.8 Step 7 — sensitivity, because every input is a guess

Net value under joint perturbation of rho and L, holding counts, capacity, and costs fixed:

```text
                L=200        L=320        L=400
  rho=0.45     52,835      166,883      242,915
  rho=0.55     95,075      234,467      327,395
  rho=0.62    124,643      281,776      386,531
  rho=0.75    179,555      369,635      496,355
```

Break-even probability across the same grid:

```text
                L=200        L=320        L=400
  rho=0.45     0.1748       0.1102       0.0884
  rho=0.55     0.1436       0.0904       0.0725
  rho=0.62     0.1277       0.0803       0.0644
  rho=0.75     0.1059       0.0665       0.0533
```

Reading the grid: the policy stays positive across every cell, which is reassuring, but the *threshold* moves by a factor of 3.3 (0.0533 to 0.1748). The value conclusion is robust; the operating point is not. That asymmetry is typical and it dictates the monitoring design — watch the threshold's economic justification, not just the total value.

The worst cell (rho = 0.45, L = 200) yields 52,835 per week, which is still positive but only 1.05x the 50,000 weekly fixed cost of the review function implied by 44.98 FTE at 1,560 plus tooling. In that cell the programme is marginal, and the recommendation would shift from "expand" to "hold capacity and improve ranking".

```text
  worst-case margin check: 52,835 / (44.98 * 1,560) = 52,835 / 70,169 = 0.75
```

A ratio below 1 means the function does not cover its wage bill in that scenario. This is the calculation that should appear in the quarterly review, not the point estimate.

### 5.9 Step 8 — the recommendation the numbers support

1. Hold capacity at 8,800 alerts per week; the marginal band beyond 17,600 is negative and the band from 8,800 to 11,000 is worth only 17.63 per alert, which does not justify a hiring round at 45 FTE scale.
2. Spend modelling effort on ranking, not calibration: the oracle gap (1,180,544) exceeds the entire current net value (281,776) by 4.2x.
3. Recalibrate only the top band, and only after it accumulates 400+ labelled cases; do not fit a monotone map on 200.
4. Publish value per FTE-week (6,264.5) alongside precision, so that staffing decisions use the same units as the benefit.
5. Fix the loss segmentation: with L varying by an order of magnitude, ranking by p rather than by p*L_i leaves value on the table that no threshold change recovers.
## 6. Visual / Geometric Intuition

Four pictures carry most of the intuition. Each is a text diagram with a stated reading order, because a diagram you cannot read in one pass is decoration.

### 6.1 The precision-prevalence surface

Precision is not a point on the ROC curve; it is a point on a surface indexed by prevalence. Hold TPR at 0.55 and slide FPR:

```text
                       precision
   1.0 |                                     ........  FPR=0.0001
       |                            .......            FPR=0.001
   0.8 |                     ......                    FPR=0.005
       |                 ....                          FPR=0.010
   0.6 |              ...                              FPR=0.020
       |           ...
   0.4 |        ..
       |      ..
   0.2 |   ..                                            <- 0.181 at pi=.004, FPR=.01
       | ..
   0.0 +---|------|------|------|------|------|-------> prevalence
       0.0005 0.001 0.002 0.004 0.01  0.02   0.05
```

Reading order: find your prevalence on the horizontal axis, then find where it intersects the FPR curve you actually have. The vertical distance between two curves is what a model improvement buys you; the horizontal distance is what a population change costs you. The second is usually larger than the first, which is why a model validated on a high-prevalence channel and deployed on a low-prevalence one *looks* broken while being unchanged.

At the illustrative operating point (pi = 0.004, FPR = 0.01, TPR = 0.55), precision is 0.181. To reach 0.50 precision on the same prevalence, FPR must fall to 0.00221:

```text
  FPR needed for PPV = 0.5:
    pi*TPR = 0.004*0.55 = 0.0022
    0.5 = 0.0022 / (0.0022 + 0.996*FPR)
    0.0022 + 0.996*FPR = 0.0044
    0.996*FPR = 0.0022 -> FPR = 0.00221
    equivalent false positives per 1.992M legitimate events: 4,400
```

Halving FPR buys 0.32 of precision at this prevalence. Halving the false-positive *count* while keeping TPR fixed is therefore the single highest-leverage modelling change available — and it is invisible on an ROC chart, where 0.01 and 0.005 FPR are a few pixels apart.

### 6.2 The confusion-matrix budget

Think of the week as a fixed budget of 2,000,000 events and a fixed budget of 8,800 reviews. Both constraints bind simultaneously, so the diagram is a rectangle divided twice.

```text
                 total events 2,000,000
  +-------------------------------------------+-------------------+
  |          TN 1,985,312                      |  FP 6,688         |
  |          (no alert, legitimate)            |  (alert, no fraud)|
  |                                            |  cost 14,044.8    |
  +--------------------------------------------+-------------------+
  |          FN 5,888          |  TP 2,112                          |
  |          (no alert, fraud) |  (alert, fraud)                    |
  |          loss 1,884,160    |  avoided 419,020.8                 |
  +----------------------------+------------------------------------+
  <---------------- 1,992,000 legitimate ----> <---- 8,000 fraud --->
```

Reading order: the horizontal split is truth, the vertical split is action. The two right-hand cells are the entire alert budget (8,800 wide). Shrinking that column raises precision and lowers recall; growing it does the reverse, and neither changes the area of the fraud column.

The lost-loss figure: 5,888 missed frauds at 320 each = 1,884,160 of loss that screening did not prevent. Compare with the 419,020.8 avoided — the programme currently prevents 18.2% of the loss it could prevent.

```text
  419,020.8 / (419,020.8 + 1,884,160) = 419,020.8 / 2,303,180.8 = 0.1820
```

### 6.3 The value-capacity curve

Net value as capacity grows, the curve that decides hiring:

```text
  V (currency/week)
  360k |                          * 358,240 @ 17,600
  336k |                     * 336,000 @ 14,000
  314k |          * 314,125 @ 11,000
  282k |    * 281,776 @ 8,800
  204k |* 204,150 @ 6,000
       |
       |                                     * 335,580 @ 21,000
       |                                          * 311,220 @ 26,000
       +----|--------|--------|--------|--------|--------|------
           6k       9k       12k      15k      18k      21k   26k
                           alerts per week
```

Reading order: the curve rises while the marginal band's precision times 198.40 exceeds the marginal cost of 13.17 (c_m plus friction at the band's precision). It peaks at 17,600 and turns. Three properties matter:

1. The peak is flat. Between 14,000 and 17,600 the curve gains only 2.2% of value (358,240/336,000 = 1.066) for 26% more capacity. A peak this flat means the capacity decision should be made on risk appetite and staffing feasibility, not on the value maximum.
2. The decline after the peak is gentle too (335,580 at 21,000 is still 94% of peak), so over-expansion costs precision less than intuition suggests — but it costs *reputation*, because 21,000 alerts means 17,640 customers contacted unnecessarily per week.
3. The vertical distance from zero is not the story; the vertical distance to oracle (1,462,320) is. The curve is a shallow hill next to a mountain the programme cannot see from the current operating point.

### 6.4 The logit scale as a portable axis

Probability thresholds do not transport; logit thresholds transport under constant prevalence shift. The mapping:

```text
  p      0.03    0.05   0.0803   0.12    0.24    0.40
  logit -3.476  -2.944 -2.4387  -1.992  -1.153  -0.405
```

Equal steps in probability are unequal steps in logit, and model evidence accumulates additively in logit space. Two consequences:

- A model improvement worth a fixed amount of *evidence* (one more useful feature with log-likelihood ratio 0.5) shifts the logit by 0.5 at every probability — which is huge at p = 0.004 and trivial at p = 0.40.
- A threshold move stated in probability units means different things at different operating points. Moving from 0.08 to 0.12 removes 0.34 of logit; moving from 0.24 to 0.28 removes only 0.19.

The practical version: state policy changes in logit units when discussing model revisions, and in probability units when discussing business thresholds. Mixing the two produces conversations where both parties are right and neither is understood.

### 6.5 The capacity-as-knapsack picture

When alert cost varies, capacity is a 1-dimensional bin and alerts are items with value and size.

```text
  capacity 1,349.3 minutes
  +--------------------------------------------------+
  | small alerts: 5.0 min each, 32.02 value           |
  |   fill: 8,800 * 5 min = 44,000 min -> overflows   |
  |   efficient packing: value per minute = 6.40      |
  +--------------------------------------------------+
  | large alerts: 180 min each, ~240 value            |
  |   value per minute = 1.33                         |
  +--------------------------------------------------+
```

Reading order: value per minute, not value per alert, decides who gets the clock. The diagram also shows why the knapsack is not the whole story — a small alert can be *deferred* a day while a large one cannot, so the true problem is scheduling with deadlines rather than a static packing problem. Section 4.4's limit 5 covers the counterexample.

## 7. Comparison / Selection Table

### 7.1 Choosing the evaluation metric

| Metric | What it answers | Invariant to | Blind to | Use when | Do not use when |
|---|---|---|---|---|---|
| ROC-AUC | Can the score order events? | Monotone transforms, prevalence | Precision at capacity, calibration | Comparing models on the same cohort | The alert budget is tiny (AUC is optimistic at low FPR) |
| PR-AUC | How does precision behave across recall? | Monotone transforms | Absolute probability, capacity | Prevalence is low and alerts are scarce | You need a specific operating point |
| Precision at capacity | What fraction of the queue is real? | Nothing (uses the cut) | Missed fraud | Sizing and defending a queue | Model selection across cohorts with different capacity |
| Recall at capacity | What share of fraud is caught? | Nothing | Queue cost | Regulatory coverage questions | Capacity is unconstrained |
| Lift | How much better than chance? | Prevalence scaling | Everything at the operating point | Communicating to non-specialists | Any monetary decision |
| ECE | Does the probability mean what it says? | Nothing | Ordering, sharpness | Threshold setting and probability reporting | Comparing models with different sharpness |
| Brier | Squared-error accuracy of probability | Nothing | Where the error sits (before decomposition) | Model selection on probability quality | Rare events, where it is dominated by uncertainty |
| Log loss | Likelihood accuracy, punishes confidence | Nothing | The low-probability band's practical relevance | Training and probability model selection | Reporting to business stakeholders |
| Net value | Is the policy worth running? | Nothing | Breakdown by segment or failure cause | Any decision involving money or capacity | Cost inputs are unreliable (then use a range) |
| Cost per prevented fraud | Unit economics of the queue | Nothing | Scale, composition | Budget negotiation | Comparing models that catch different segments |

Three selection rules that follow from the table:

1. For a capacity decision, use **net value** and report precision and recall as its inputs. Any other headline invites optimisation of the wrong thing.
2. For a model-selection decision inside a fixed capacity, use **precision at capacity**, because it is the only ranking metric aligned with the constraint you actually face.
3. For a threshold-setting decision, use **calibration plus p\***, because a threshold is an economic quantity expressed in probability units.

### 7.2 Choosing the capacity point

| Capacity (alerts/wk) | Precision | Recall | Net value | Value/alert | Marginal band value | Recommendation |
|---|---|---|---|---|---|---|
| 6,000 | 0.2500 | 0.1875 | 204,150 | 34.02 | 30.65 (first band) | Under-uses staffed capacity |
| 8,800 | 0.2400 | 0.2640 | 281,776 | 32.02 | 30.65 (6k->8.8k) | Current operating point |
| 11,000 | 0.2227 | 0.3063 | 314,125 | 28.56 | 17.63 | Expand if hiring is approved |
| 14,000 | 0.2000 | 0.3500 | 336,000 | 24.00 | 10.22 | Expansion still pays |
| 17,600 | 0.1818 | 0.4000 | 358,240 | 20.35 | 9.11 | Value maximum; flat region |
| 21,000 | 0.1600 | 0.4200 | 335,580 | 15.98 | -3.73 | Over-expanded; loses value |
| 26,000 | 0.1400 | 0.4550 | 311,220 | 11.97 | -1.94 | Requires re-ranking before use |

The marginal column is the one to argue about. It is computed from integer counts within each band, then priced with marginal cost 11.07 rather than average 14.00. Notice the discontinuity pattern: the band from 8,800 to 11,000 carries 15.36% marginal precision and 17.63 of value; the band from 17,600 to 21,000 carries 4.71% and loses 3.73. The drop between those two bands is 12.92 per alert, and it is where the recommendation changes.

### 7.3 Choosing the calibration correction

| Situation | Correction | Data needed | Risk | Verification |
|---|---|---|---|---|
| Monotone distortion, plenty of labels | Isotonic on a temporal holdout | 2,000+ positives in the calibration window | Overfits sparse top bins | Reliability on a third, untouched period |
| Mild distortion, stable prevalence | Platt scaling (logistic on score) | 500+ positives | Cannot fix non-monotone error | Log loss vs uncalibrated baseline |
| Prevalence shift only | Prior correction on the logit: shift by ln(pi_new/pi_old) odds | Population base rate | Assumes constant likelihood ratios | Precision at capacity on the new cohort |
| Sharpness too low (predictions bunched) | Retrain with a proper loss, not post-hoc | Full training set | Post-hoc maps cannot add resolution | Brier decomposition: resolution term rises |
| Sparse top band only | Pool adjacent bins or abstain | Depends on band | Pooling hides the segment you care about | Per-band coverage report |
| No mature labels | Do nothing; report uncalibrated | — | Every downstream threshold is unjustified | Document the label gap and the window |

The prior-correction row is worth an explicit formula because it is the most common production fix and the most commonly mis-applied:

```text
  logit_new = logit_old + ln( odds(pi_new) / odds(pi_old) )
  example: pi_old = 0.004, pi_new = 0.002
    odds_old = 0.004/0.996 = 0.0040161 -> ln = -5.5175
    odds_new = 0.002/0.998 = 0.0020040 -> ln = -6.2146
    shift    = -0.6971
  so a score of 0.0803 becomes sigmoid(-2.4387 - 0.6971) = sigmoid(-3.1358) = 0.0417
```

The correction is valid only if the likelihood ratio P(x|fraud)/P(x|legitimate) is unchanged. If the new population also has different fraud *behaviour* — different attack types, different devices — the likelihood ratios have moved too, and the shift is a coincidence rather than a calibration.

### 7.4 Choosing between reviewing and automating

| Criterion | Human review | Rules / automation | Hybrid (auto + review on exception) |
|---|---|---|---|
| Fixed cost | High (FTE, training, supervision) | High build, low run | Medium |
| Marginal cost | 11.07 per alert | near zero per alert | 11.07 only on exceptions |
| Recovery rate achievable | High (0.62 in the example) | Low (0.20 typical for auto-decline) | Mixed by band |
| Friction imposed | Medium (2.10) | Higher when wrong (blocked good customer) | Lowest, if exceptions are routed well |
| Latency | Hours to days | Milliseconds | Milliseconds for the clear cases |
| Adaptability to new attack | Fast (analyst notices patterns) | Slow (rule authoring cycle) | Fast at the boundary |
| Precision needed to be viable | p > 0.0803 | p > 0.0215 at the auto-decline cost structure | band-dependent |
| Auditability | Case notes, reviewable | Rule versions, replayable | Both, split by band |

The precision thresholds in that table are the key row and they are computed, not asserted: the auto-decline action from Section 4.2 (c = 0.30, rho = 0.20, c_f = 1.10) breaks even at 0.0215, while analyst review breaks even at 0.0803. Automation is viable at lower precision precisely because it is cheap — which is why the correct architecture sends the *low* band to automation and the *middle* band to humans, not the reverse.
## 8. Failure Modes & Edge Cases

Each entry states the mechanism, what it looks like on a dashboard, the *competing* causes that produce the same symptom, the check that separates them, and the corrective action. A failure mode without a discriminating check is a worry, not a diagnosis.

### 8.1 Failure 1 — label immaturity read as model degradation

**Mechanism.** Labels arrive with a lag. A cohort evaluated at 30 days is missing roughly 58% of the positives that will eventually be confirmed, and it is missing them *non-randomly*: slow-confirming frauds skew toward complex cases, cross-border disputes, and small amounts. Precision computed at 30 days is therefore biased low and biased toward the easy cases.

**Symptom.** Precision steps down sharply right after a model refresh and recovers over the following quarter without any code change.

**Competing causes.**
1. Immature labels (recovery without intervention is the tell).
2. Genuine concept drift (attack mix changed; recovery does not occur without retraining).
3. A scoring pipeline bug in a specific segment (recovery occurs only when the segment is fixed).
4. Threshold moved (check the published threshold and alert rate first — cheapest test).

**Discriminating check.** Recompute precision on the *same* event cohort at several maturities and fit the maturity curve:

```text
  t=30  maturity 0.4204  TP observed 888   precision observed 0.1009
  t=45  maturity 0.5588  TP observed 1180  precision observed 0.1341
  t=60  maturity 0.6641  TP observed 1403  precision observed 0.1594
  t=90  maturity 0.8053  TP observed 1701  precision observed 0.1933
```

If precision rises monotonically with maturity on a *fixed* cohort, the "degradation" is measurement. If it is flat while an older cohort's matured precision is higher, the model genuinely worsened.

**Correction.** Evaluate only cohorts at or beyond the maturity threshold needed for 80% coverage (here about 90 days), or apply an inverse-maturity correction with an explicit uncertainty band. Publish the maturity of every reported number.

**Verification.** The maturity curve on a fixed cohort should approach the 90-day plateau from below, and the implied eventual precision at maturity 1.0 should equal the full-cohort figure: 0.24 * 0.8053 ≈ 0.1933 observed, so 0.24 at full maturity — the arithmetic closes.

### 8.2 Failure 2 — selection bias from evaluating on alerted events only

**Mechanism.** Only alerted events are investigated; only investigated events get dispositions. Precision on that subset is well measured, recall is not measured at all, and any model trained on it learns to reproduce the previous policy's decisions.

**Symptom.** A new model's offline recall is suspiciously similar to the incumbent's; the "improvement" appears only in precision.

**Competing causes.**
1. Selection bias (a random audit sample shows the unalerted population contains fraud).
2. The new model is genuinely similar (feature sets nearly identical; check the correlation of scores).
3. Labels genuinely absent for unalerted events (the unalerted population has no dispositions at all and none can be inferred).

**Discriminating check.** Take a 2,000-event random sample from the unalerted population and investigate it to completion. If more than a handful are fraud, the recall estimate is biased. Even a 500-event audit bounds the miss rate to within about ±2 percentage points at 95% confidence when the observed rate is under 2%.

**Correction.** Maintain a permanent random hold-out (0.1% of events, never actioned) and include it in every recall calculation. Accept its cost as an evaluation expense: 2,000 events per week at 4.3% investigation rate is 86 alerts — small against 8,800 and decisive for measurement.

**Verification.** Recall computed with the audit sample should be *lower* than recall computed on alerted-only data, and the gap should shrink as the audit sample grows. A gap that grows is a sign the hold-out is contaminated by indirect action (for example, a fraud ring blocked by one alert suppresses the others).

### 8.3 Failure 3 — friction cost omitted from low-value segments

**Mechanism.** c_f is small in the base case (2.10 against 198.40 of value per true positive, 1.06%), so it is easy to drop. In a low-loss segment — a 12-unit micro-transaction, a low-value claim — the same 2.10 dominates and flips the sign.

**Symptom.** A segment shows acceptable precision (say 0.14) but complaints per thousand contacted customers rise 4x.

**Competing causes.**
1. Friction genuinely exceeds benefit (the threshold is below p* for that segment).
2. Complaint coding changed (check the coding manual version date).
3. Contact channel changed (email to phone calls raises friction 3-5x with identical alerts).

**Discriminating check.** Recompute p* per segment with segment-specific L and c_f:

```text
  segment A: L=320, c_f=2.10  -> p* = 16.10/200.50 = 0.0803
  segment B: L=12,  c_f=2.10  -> p* = 16.10/(7.44+2.10) = 16.10/9.54 = 1.688
```

A p* of 1.688 is unattainable: no alert in segment B can ever be worth reviewing. The correct policy for that segment is no review at all, regardless of model quality. This is the arithmetic that ends arguments about "why don't we catch more micro-fraud".

**Correction.** Never apply a global threshold to segments whose L varies by more than 3x. Compute p* per loss band and route by band. Alternatively auto-decline the low band (p* = 0.0215 under the automation cost structure) instead of reviewing it.

**Verification.** After segmentation, complaints per thousand in each band should be within a factor of 1.5 of each other; a wider spread means a threshold is still mispriced.

### 8.4 Failure 4 — duplicate alerts inflate every metric

**Mechanism.** One underlying case generates several events: retries, card-testing bursts, multiple claims on one incident. Each becomes an alert, each consumes review time, and each can be confirmed — so TP counts one fraud several times while recall's denominator counts it once.

**Symptom.** Precision looks strong, recall looks weak, and the ratio of alerts to distinct cases drifts upward.

**Competing causes.**
1. Genuine duplicates (a case-level dedup shows 6% collapse).
2. Legitimate repeat events (subscription retries that are genuinely separate decisions).
3. Scoring granularity mismatch (scoring at event level while operations works at case level).

**Discriminating check.** Count distinct case keys per alert and per confirmed fraud:

```text
  alerts 8,800 -> distinct cases 8,272 (6.0% duplicate)
  cost per distinct case = 8,800*14.00 / 8,272 = 123,200/8,272 = 14.89
  vs 14.00 nominal: 6.4% cost understatement
```

If the duplicate share differs between the alert set and the confirmed set, precision is misstated in the direction of the difference.

**Correction.** Decide the unit of account once — event or case — and convert every metric to it. Deduplicate the queue by (customer, instrument, 60-minute window) before dispatch, and recompute p* against the deduplicated review cost.

**Verification.** Sum of per-case review minutes should reconcile with total analyst hours within 2%. If not, the dedup key is wrong, not the timekeeping.

### 8.5 Failure 5 — value-weighted recall diverges from count recall

**Mechanism.** Count recall treats a 12-unit fraud and a 4,000-unit fraud identically. With a heavy-tailed loss distribution, the alerts you cut are the small ones at the bottom of the ranking, so count recall falls slowly while value recall collapses.

**Symptom.** "We catch 26% of frauds" while recovered value falls quarter over quarter.

**Competing causes.**
1. Genuine tail migration (fraudsters moved to larger transactions; the loss distribution shifted).
2. Ranking by p instead of p*L (the tail sits below the cut and is never alerted).
3. Recovery rate declining on large cases (slower action on high-value alerts because they route to a slower review tier).

**Discriminating check.** Compute both recalls on the same cohort and compare the gap:

```text
  count recall    = 2,112 / 8,000                        = 0.2640
  value recall    = sum L over caught / sum L over all    (needs per-event L)
  if value recall < 0.75 * count recall, the tail is under-alerted
```

Then check the score distribution of the top-1% loss events: if their mean score is below the threshold, ranking is the cause, not the world.

**Correction.** Rank by expected loss p*L_i rather than by p, and reserve capacity for a tail queue. Report both recalls in every dashboard.

**Verification.** After switching the ranking key, value recall should rise while count recall stays flat or falls slightly. A simultaneous rise in both means the model improved, not just the ranking.

### 8.6 Failure 6 — threshold drift hidden by a fixed alert rate

**Mechanism.** When the queue is capped at 8,800, the effective threshold floats with the score distribution. As the model's scores drift downward (or prevalence falls), the threshold falls to keep the queue full, and precision decays quietly.

**Symptom.** Alert count constant at 8,800, precision falling, and *no change in any threshold-monitoring alert* because the threshold was never the monitored quantity.

**Competing causes.**
1. Score drift with a floating cut (the threshold moves; check its history).
2. Genuine model degradation at a fixed threshold (precision falls and the threshold is unchanged).
3. Population mix change (a segment with different base rate grew; check segment shares before and after).
4. Label lag interacting with a recent policy change (compare only matured cohorts).

**Discriminating check.** Plot threshold and precision on the same time axis. If the threshold moved by more than 10% while precision fell, drift is the cause. If the threshold is flat and precision fell, the model is the cause. If segment shares moved by more than 5 percentage points, mix is the cause.

**Correction.** Monitor the threshold, the alert rate, and the score distribution's 50th/90th/99th percentiles together. Set an alert on threshold movement independent of precision, because precision alone cannot distinguish the causes.

**Verification.** After pinning the threshold, the alert count should float — and that floating count is itself the drift signal, now visible instead of absorbed.

### 8.7 Failure 7 — capacity measured in alerts when review time varies

**Mechanism.** Alerts differ in complexity by a factor of 35 in the illustrative case (5 minutes versus 180 minutes). Planning capacity in alert counts assumes homogeneity and produces systematic overcommitment whenever the mix shifts toward complex cases.

**Symptom.** Queue backlog grows while "capacity utilisation" reports 100%; overtime rises; time-to-first-review rises.

**Competing causes.**
1. Mix shift toward complex alerts (mean handle time rises; the count is unchanged).
2. Staffing shortfall (mean handle time constant, hours available fell).
3. Tooling slowdown (mean handle time rises across all complexity bands uniformly).
4. Routing defect (complex alerts sent to junior reviewers who take longer).

**Discriminating check.** Track mean handle time by complexity band. Uniform increase across bands points to tooling or staffing; increase concentrated in the complex band points to mix; increase concentrated in one team points to routing or skill.

```text
  capacity in minutes: 1,349.3 * 60 = 80,958 minutes available
  demand at 8,800 alerts, 9.2 min mean = 80,960 minutes
  utilisation = 80,960 / 80,958 = 100.002%
```

A utilisation of 100.002% is the mathematical signature of a queue with no slack: any variance produces a backlog that never clears. Plan to 85-90% and treat the remainder as the buffer that absorbs mix shocks.

**Correction.** Convert capacity to minutes, staff to 88% utilisation, and keep a separate fast lane for the 5-minute alerts so that they are not blocked behind 180-minute cases.

**Verification.** Backlog should be stationary week over week at 88% utilisation. A trending backlog at that utilisation means the mean handle time estimate is stale.

### 8.8 Failure 8 — the counterfactual is not what the alert prevented

**Mechanism.** Value is computed as TP * rho * L, which assumes that absent the alert, the full loss would occur. In reality some alerted frauds would have been declined anyway by downstream controls, some would have been reversed by the customer, and some were never going to complete.

**Symptom.** Net value is stable and healthy while chargeback volumes fall for reasons unrelated to the model — the "prevented" loss is being double-counted against other controls.

**Competing causes.**
1. Double counting with downstream controls (the alert and a rule both claim the same case).
2. rho overstated (recovery measured on cases where recovery was inevitable).
3. Genuine improvement in prevention (chargebacks fall *and* the model's contribution is measured by uplift, not by attribution).
4. Reporting window mismatch (losses booked in a different period than prevention).

**Discriminating check.** Run a small randomised hold-out where a fraction of alerts is suppressed (subject to policy and legal constraints) and measure the *difference* in loss between actioned and suppressed cohorts. Attribution without randomisation cannot separate the causes.

```text
  hold-out 5% of alerts (440 per week) suppressed
  loss in actioned cohort:     L_a
  loss in suppressed cohort:   L_s
  incremental benefit per alert = (L_s - L_a) / 440
  Compare with the modelled 32.02. If the realised figure is 18, rho is overstated
  by a factor of 1.78 and every downstream threshold should be recomputed.
```

**Correction.** Replace attribution with measurement where permitted; where it is not, apply a conservative rho and state it as an assumption with a range. Publish the assumption's source and date.

**Verification.** Modelled and measured incremental benefit should converge within 25% after two quarters of hold-out data. Persistent divergence means the attribution model, not the estimator, is wrong.

## 9. Complexity & Cost Analysis

### 9.1 Computational complexity of the evaluation itself

Fraud evaluation is cheap relative to scoring; the cost that matters is human review. Still, the evaluation pipeline has its own complexity, and getting it wrong produces slow, expensive, and eventually stale numbers.

| Operation | Complexity | Cost driver | Practical scale | Risk at scale |
|---|---|---|---|---|
| Sort scores for a threshold | O(n log n) | n = eligible events | 2M events, sub-second in memory | None |
| Exact PR curve | O(n log n) | sorting dominates | 2M events | Memory if materialised per segment |
| Top-k precision per segment | O(n) with a heap of size k | k per segment, segments | 2M events x 40 segments | 40 passes if implemented naively |
| Reliability curve | O(n) | n plus bin count | 2M events, 20 bins | Sparse top bins |
| Brier / log loss | O(n) | n | 2M events | Numerical underflow on extreme logits |
| Isotonic calibration | O(n log n) | n positives in the window | 2,000-20,000 positives | Overfits below ~200 per bin |
| Net value curve | O(n log n) | ranking plus prefix sums | 2M events | Recomputing per scenario multiplies by scenario count |
| Per-segment p* | O(s) | s = segments | 40 segments | Fine, but inputs diverge per segment |

Two engineering notes that save real time:

1. Compute the confusion matrix once at the ranked cut and derive all ratios from integers. Recomputing precision from rounded displayed values is how tables stop reconciling (Section 4.2 showed a 6,446 discrepancy from exactly this).
2. Store prefix sums of TP and FP along the ranking so that any capacity's value is an O(1) lookup. Without prefix sums, a 40-scenario sensitivity sweep on 2M events re-sorts 40 times.

```text
  naive sweep: 40 sorts * O(2e6 log 2e6) ~ 40 * 42e6 comparisons ~ 1.7e9
  prefix-sum sweep: 1 sort + 40 * O(2e6) lookups ~ 42e6 + 80e6 ~ 1.2e8
  ratio ~ 14x, before counting the 40x reduction in memory traffic
```

### 9.2 Human cost model

The real cost structure, in the illustrative programme:

```text
  alerts per week                8,800
  minutes per alert               9.20
  minutes per week           80,960
  hours per week              1,349.3
  productive hours per FTE      30.00   (of 40 paid; 25% non-review work)
  FTE required                 44.98
  wage cost per FTE per week 1,560.00   (30 h * 52.00)
  wage cost total           70,169.0
  other review cost         53,031.0    (tooling, supervision, QA, training)
  total review cost        123,200.0    matches 8,800 * 14.00
```

The decomposition matters because the two components behave differently. Wage cost scales linearly with FTE; tooling and supervision have a large fixed component. That is exactly why marginal cost (11.07) is below average cost (14.00): the fixed 2.93 per alert is already paid.

```text
  marginal cost = 14.00 - 2.93 = 11.07
  break-even with marginal: 0.0657
  break-even with average:  0.0803
```

### 9.3 Cost per outcome, and why it is the wrong headline

```text
  cost per alert worked       14.00
  cost per true positive      123,200 / 2,112  = 58.33
  cost per prevented fraud    123,200 / (2,112*0.62) = 123,200 / 1,309.4 = 94.08
  benefit per prevented fraud 320.00
  return on review spend      419,020.8 / 137,244.8 = 3.05
```

"Cost per prevented fraud of 94.08 against 320 of loss" is a good sentence for a budget meeting and a bad optimisation target. Minimising cost per prevented fraud is achieved by alerting only on the very top of the ranking, which reduces prevention volume and total value. The objective is net value, and cost per outcome is a *diagnostic* on it.

The distinction becomes concrete at two operating points:

```text
  capacity 6,000:  cost/prevented = (6,000*14.00 + 4,500*2.10)/(1,500*0.62) = 93,450/930 = 100.5
  capacity 8,800:  cost/prevented = 137,244.8/1,309.4                        = 104.8
  capacity 17,600: cost/prevented = (17,600*14.00 + 14,400*2.10)/(3,200*0.62) = 276,640/1,984 = 139.4
```

Cost per prevented fraud rises with capacity while net value also rises (up to 17,600). A team optimising the ratio stops at 6,000 alerts and leaves 154,090 per week of value on the table. A team optimising net value expands to 17,600 and accepts a 33% worse unit cost. The second team is correct.

### 9.4 Cost of a false alert, decomposed

The 2.10 friction figure is an average that hides a wide distribution. A defensible decomposition:

```text
  component                      currency per false alert
  contact handling (email/SMS)              0.55
  inbound call probability 0.09 * 8.50      0.77
  analyst time on the false alert           0.00   (already in c_a)
  customer attrition risk 0.0004 * 900      0.36
  goodwill credit issuance 0.02 * 21        0.42
  total                                     2.10
```

Two implications follow. First, the largest single component is customer contact, so friction is reducible by changing the contact channel rather than by improving the model. Second, attrition risk is a probability times a large number (900 lifetime value): doubling the attrition probability to 0.0008 raises c_f to 2.46 and p* to 0.0824 — a 2.6% threshold move from a change no model can influence.

```text
  p* sensitivity to c_f:  (16.10+0.36)/200.50 = 0.0821 at c_f = 2.46
  recomputed exactly:     (14.00+2.46)/(198.40+2.46) = 16.46/200.86 = 0.08195
```

### 9.5 Complexity of maintaining the evaluation

The ongoing cost that nobody budgets:

| Activity | Frequency | Effort | Consequence if skipped |
|---|---|---|---|
| Maturity curve refresh | Monthly | 2 hours | Metrics compared across incomparable cohorts |
| Reliability curve per segment | Monthly | 3 hours | Calibration break discovered by an auditor |
| Threshold and alert-rate reconciliation | Weekly | 30 minutes | Silent drift, precision decay |
| Capacity and handle-time audit | Monthly | 4 hours | Backlog or idle staff |
| Cost input refresh (L, c_a, c_f, rho) | Quarterly | 1 day | Threshold justified by stale economics |
| Hold-out audit sample | Weekly | 6 hours | Recall unmeasurable; selection bias undetected |
| Cost-weighted (value) recall | Monthly | 2 hours | Tail under-alerting invisible |
| Re-derivation of p* per segment | Quarterly | 4 hours | Global threshold applied to incompatible segments |

Total steady-state effort is roughly 1.5 FTE-days per month excluding the audit sample. That is the price of a defensible number, and it is small compared with the cost of a policy built on an undefended one.

### 9.6 Where the next unit of effort pays

Ranked by expected value per unit of effort, using the illustrative numbers:

```text
  1. Ranking improvement         gap to oracle 1,180,544/wk   -> largest by far
  2. Tail segmentation (p*L)     partial, order 10^5/wk        -> cheap, mechanical
  3. Duplicate dedup             6% of review cost = 7,392/wk  -> one query
  4. Tiered actions              low-band automation, order 10^4/wk
  5. Top-band recalibration      small in value, large in trust
  6. Threshold micro-tuning      bounded by 0.14/alert = 1,232/wk
```

The ordering is the useful output of this section. Threshold tuning — the activity that consumes the most meeting time — is second to last, and bounded at about 1,232 per week by the existing slack of 0.14 per alert across 8,800 alerts. Ranking work is worth roughly a thousand times more.
## 10. Key Theorems / Results Reference

Each entry states the result, the conditions under which it holds, and the operational consequence. Proofs are omitted where the derivation is in Section 4; what matters here is the *scope* of each result, because scope errors are how correct theorems produce wrong policies.

### 10.1 Bayes posterior for precision

**Statement.** For a cohort with prevalence pi, detection rate TPR, and false-positive rate FPR, precision PPV = pi·TPR / (pi·TPR + (1−pi)·FPR).

**Conditions.** TPR and FPR measured on the same cohort, same window, same eligibility rules. Event independence is *not* required for the identity (it is an average), but it is required for the confidence interval around it.

**Consequence.** Precision is jointly determined by the model and the population. A model cannot be certified as "24% precise" without naming a prevalence. Corollary: any precision figure quoted cross-channel requires the prevalence to be quoted with it.

### 10.2 Neyman-Pearson and the shape of the optimal frontier

**Statement.** For a fixed FPR budget, the most powerful test is a likelihood-ratio threshold: alert when P(x|fraud)/P(x|legitimate) exceeds a constant. The ROC frontier attained by such tests is concave, and any point inside the concave hull is dominated.

**Conditions.** Both class-conditional densities known (or well estimated); a single FPR constraint.

**Consequence.** If your measured ROC has points inside its own convex hull, those operating points are unattainable-or-dominated and should not be chosen. In practice, hull violations are a sign of score quantisation or of two populations mixed together — often the finding that reveals a segment boundary.

### 10.3 Bayes-optimal threshold equals the cost ratio

**Statement.** Under a binary action with loss λ_FN per missed fraud and λ_FP per false alert, the expected-loss-minimising rule alerts when p ≥ λ_FP/(λ_FP + λ_FN).

**Conditions.** Losses constant across events; actions binary; probabilities calibrated.

**Consequence.** With λ_FN = rho·L = 198.40 and λ_FP = c_a + c_f = 16.10, the threshold is 16.10/214.50 = 0.0751 — *not* the 0.0803 derived in Section 4.1. The difference is structural: the loss-ratio form charges review cost to the false-alert branch, while the value form charges it to every alert. Both are correct under their own accounting; mixing them produces a threshold that is right for neither.

```text
  loss-ratio form:     0.0751  (review cost inside the FP branch)
  value form:          0.0803  (review cost charged to all alerts)
  difference:          0.0052 probability = 1.04 currency per alert at this margin
```

This is the most common silent error in threshold derivation, and the fix is to write down which branch pays c_a before dividing.

### 10.4 Proper scoring rules and the Murphy decomposition

**Statement.** Brier score decomposes exactly as Brier = reliability − resolution + uncertainty, where reliability = Σ (n_b/N)(p_b − o_b)², resolution = Σ (n_b/N)(o_b − ō)², uncertainty = ō(1 − ō). Both Brier and log loss are strictly proper: expected loss is minimised by reporting the true probability.

**Conditions.** Decomposition uses discrete bins; the identity is exact for the binned estimate and approximate for the continuous score. Resolution and reliability trade off: a model can improve reliability by predicting closer to the base rate, which destroys resolution.

**Consequence.** Calibration work that raises reliability while lowering resolution is a *net loss* in Brier terms. The illustrative audit shows resolution of 0.0000988 against reliability of 0.0000033 — the model is already nearly perfectly reliable and its entire skill (2.66%) comes from the small resolution term. Flattening predictions "to improve calibration" would remove the only skill present.

```text
  Brier          = 0.0000033 - 0.0000988 + 0.0035870 = 0.0034915
  skill          = (resolution - reliability)/uncertainty = 2.66%
  flatten to ō   -> reliability 0, resolution 0, skill 0   (a strictly worse model)
```

### 10.5 Monotone invariance of ranking metrics

**Statement.** For any strictly increasing transform g, AUC(g(s)) = AUC(s) and PR-AUC(g(s)) = PR-AUC(s). Precision at a fixed *quantile* is also invariant; precision at a fixed *threshold* is not.

**Conditions.** g strictly increasing and independent of the label. A transform fitted on labels (isotonic, Platt) is not label-independent and can change ranking on held-out data through overfitting.

**Consequence.** AUC cannot detect calibration breakage; calibration cannot detect ranking breakage. A monitoring suite needs both, and it needs precision at capacity — which is invariant to monotone transforms only when capacity is expressed as a count, not a threshold.

### 10.6 Logit additivity and prior correction

**Statement.** If class-conditional densities belong to an exponential family with equal dispersion, the posterior logit is an additive function of features plus a prior term. Changing prevalence from pi₁ to pi₂ shifts every logit by a constant ln(odds(pi₂)/odds(pi₁)), leaving the ranking and every likelihood ratio unchanged.

**Conditions.** Equal-dispersion exponential family (logistic regression, naive Bayes with shared variance). Violated by models with prevalence-dependent structure (tree ensembles fit to imbalanced data, reweighted training).

**Consequence.** Prior correction is legitimate for logistic models and *not* automatically for gradient-boosted trees trained with class weights. Applying a constant logit shift to a GBM can leave it mis-calibrated because the weighting changed the fitted function, not just the intercept.

```text
  pi 0.004 -> 0.002:  shift = -6.2146 - (-5.5175) = -0.6971
  score 0.0803 -> sigmoid(-3.1358) = 0.0417
```

### 10.7 Value of information: why ranking beats calibration here

**Statement.** The value of improving a score's ordering is bounded below by zero and above by the oracle gap at the operating capacity; the value of correcting a monotone calibration error at a fixed capacity is zero, because the threshold absorbs it.

**Conditions.** Capacity fixed in alerts; threshold free to move; costs unchanged.

**Consequence.** At 8,800 alerts the oracle gap is 1,180,544 per week while the threshold slack is 0.14 per alert (1,232 per week total). Ranking improvement therefore dominates calibration improvement by roughly a factor of 958 on this programme. The theorem is not deep; the failure to apply it is common, because calibration projects are easier to scope and easier to declare finished.

```text
  oracle gap            = 1,462,320 - 281,776 = 1,180,544
  threshold slack value = 0.14 * 8,800        = 1,232
  ratio                 = 958
```

### 10.8 Precision-recall dominance and the iso-lift curve

**Statement.** On a precision-recall plane, the line precision = lift × prevalence is an iso-lift contour. A model's PR curve crosses each contour at the alert rate where it achieves that lift. Comparing two models by maximum lift is equivalent to comparing them at possibly different alert rates, which is meaningless under a capacity constraint.

**Conditions.** Single prevalence; the same recall definition.

**Consequence.** Compare models by precision at the *same* alert count, or by the area between their curves over the capacity-relevant range only. The illustrative model achieves lift 60.0 at 8,800 alerts; a competitor achieving lift 80 at 2,000 alerts is not comparable and may well be worse at 8,800.

### 10.9 Knapsack optimality and its failure

**Statement.** Maximising total value subject to a single capacity constraint is a 0/1 knapsack. The fractional relaxation is solved exactly by value-per-unit-capacity sorting; the integer version is NP-hard and the greedy solution is within a factor of 2 of optimal in the worst case, usually much closer.

**Conditions.** Values and sizes independent across items; single constraint. With deadlines (time-critical cases) the problem becomes scheduling, and greedy value-density can be arbitrarily bad.

**Consequence.** Use value-per-minute sorting for a static queue, and carve out a reserved lane for time-critical alerts whose option value is not captured by static EV. Section 4.4's limit 5 gives the numbers (6.40 versus 1.33 per minute) and the exception.

### 10.10 Summary table of results and their use

| Result | Gives you | Requires | Typical misuse |
|---|---|---|---|
| Bayes PPV | Precision from prevalence | Same-cohort TPR/FPR | Quoting precision without prevalence |
| Neyman-Pearson | Shape of the attainable frontier | Known densities | Choosing a hull-dominated point |
| Cost-ratio threshold | Optimal cut under binary loss | Constant losses | Mixing the two cost accountings |
| Brier decomposition | Where probability skill lives | Binned scores | "Improving calibration" while destroying resolution |
| Monotone invariance | Which metrics can detect what | Strictly increasing transform | Using AUC to certify calibration |
| Prior correction | Portability across prevalence | Exponential family | Applying it to a reweighted GBM |
| Value of information | Where modelling effort pays | Fixed capacity | Spending on calibration at a floating threshold |
| Iso-lift contour | Comparable model comparison | Fixed alert rate | Comparing maximum lift |
| Knapsack | Capacity allocation with variable size | Independent items | Ignoring deadlines |
## 11. Interview / Exam Questions

Fourteen questions in ascending difficulty. Each has its own worked reasoning; the computations are reproducible from the Section 5 input block.

### Q1. Define precision, recall, and false-positive rate for the illustrative cohort, and state which one is invariant to the alert threshold.

**Answer.** Precision = TP/(TP+FP) = 2,112/8,800 = 0.240. Recall = TP/(TP+FN) = 2,112/8,000 = 0.264. FPR = FP/(FP+TN) = 6,688/1,985,312 = 0.003369. None of the three is invariant to the threshold: all move as the cut moves. What *is* threshold-invariant is the underlying score distribution — the ROC and PR curves, and AUC/PR-AUC derived from them.

**Reasoning.** The question tests whether the candidate confuses "property of the model" with "property of the operating point". Precision, recall, and FPR are all evaluated at a specific cut; only the curves are cut-free. A candidate who says "precision is a model property" will mis-design every comparison.

**Follow-up.** Which of the three is invariant to prevalence? Answer: none, and recall is the least sensitive (it depends on the score distribution within the fraud class), while precision is highly sensitive.

**Red flag.** Reciting the formulas but claiming AUC is "precision averaged over thresholds" — it is not; it is the probability that a random fraud outranks a random legitimate event.

**Scoring.** 1 point formulas, 1 point naming the curves as the invariant object, 1 point noting prevalence sensitivity of precision.

### Q2. Your model has AUC 0.91 and the queue is unusable. What are the three most likely explanations, and how do you test each?

**Answer.** (1) *Low prevalence*: at pi = 0.004 an excellent ranker still yields low precision — compute PPV from the observed TPR/FPR at your cut; if it matches the observed 0.24, nothing is wrong with the model, the population is the constraint. (2) *Capacity too high*: you are cutting deep into the ranking; compute precision at 4,400 instead of 8,800 and check the curve's slope. (3) *Calibration-independent threshold mismatch*: the cut was set on a probability scale that no longer matches the population, so it alerts far more than intended; check the alert rate against the intended 0.44%.

**Reasoning.** AUC is prevalence-insensitive by construction, so a high AUC can coexist with any precision. The three causes are distinguished by three different checks, and each implies a different remedy: accept (1), shrink (2), recalibrate (3).

**Follow-up.** If precision at 4,400 is 0.25 and at 8,800 is 0.24, is capacity the cause? No — the curve is flat, so the ranking is the constraint, and cutting capacity buys almost nothing.

**Red flag.** "Retrain the model" as the first answer, with no diagnostic.

**Scoring.** 1 point each cause, 1 point for matching each to a check, bonus for noting the flat-curve case.

### Q3. Compute the break-even probability under two cost accountings and explain why they differ.

**Answer.** Value form: (c_a + c_f)/(rho·L + c_f) = (14.00+2.10)/(198.40+2.10) = 16.10/200.50 = 0.0803. Loss-ratio form: (c_a + c_f)/(c_a + c_f + rho·L) = 16.10/(16.10+198.40) = 16.10/214.50 = 0.0751. They differ because the value form charges review cost to every alert and the loss form charges it only to the false-alert branch; the denominators differ by c_a + c_f versus rho·L.

**Reasoning.** Both are correct under their own accounting. The value form is the right one when the review budget is real and finite, because the cost is paid whatever the outcome. The loss form is right when you are choosing between two classifiers with review capacity assumed available.

**Follow-up.** Which one should be used to set a production threshold? The value form, because production review capacity is the binding constraint and every alert consumes it.

**Red flag.** Getting one number and asserting the other is a mistake, rather than a different accounting.

**Scoring.** 1 point each formula, 1 point for the structural explanation, 1 point for choosing correctly with a reason.

### Q4. Prevalence in a new market is 0.10%. Recompute precision and expected value at the same operating point, and state the policy implication.

**Answer.** With TPR = 0.264 and FPR = 0.003369: numerator = 0.001×0.264 = 0.000264; denominator = 0.000264 + 0.999×0.003369 = 0.000264 + 0.003366 = 0.003630; PPV = 0.0727. EV = 0.0727×198.40 − 14.00 − 0.9273×2.10 = 14.42 − 14.00 − 1.947 = −1.53 per alert.

**Reasoning.** The model is unchanged; the population is quartered. Below roughly 8% precision the queue destroys value, and the new market sits at 7.3%. The implication is either a higher threshold (accept much lower recall), a cheaper action (auto-decline at p* = 0.0215), or no programme in that market until ranking improves.

**Follow-up.** At what prevalence does this operating point break even? Solve pi×0.264/(pi×0.264+(1−pi)×0.003369) = 0.0803 → pi ≈ 0.00110, i.e. about 11 basis points.

**Red flag.** Quoting precision without recomputing, or claiming the model "transfers" because AUC was stable.

**Scoring.** 1 point PPV, 1 point EV, 1 point for a concrete policy choice, 1 point for the break-even prevalence.

### Q5. You have 1,349.3 review hours per week. Two alert types take 5 and 180 minutes. How do you allocate, and where does the greedy rule fail?

**Answer.** Rank by value per minute: small alerts give 32.02/5.0 = 6.40 per minute, large give about 240/180 = 1.33 per minute. Greedy value-density fills the week with small alerts. It fails when a large alert is time-critical — if its loss is recoverable only today, the static EV understates its option value, and strict greedy allocation liquidates the tail that carries most of the money.

**Reasoning.** The problem is a knapsack with deadlines, not a static knapsack. The correct design is two lanes: a value-density lane for the bulk queue, and a reserved-capacity lane for time-critical cases sized from their historical arrival rate plus a variance buffer.

**Follow-up.** How much capacity to reserve? Enough that the probability of exceeding the lane in a week is under 5% — with a Poisson arrival rate of, say, 12 large cases per week at 180 minutes each, reserve 12×180 plus about 1.65×sqrt(12)×180 ≈ 2,160 + 1,029 ≈ 3,190 minutes.

**Red flag.** Allocating purely by alert count, or by value per alert.

**Scoring.** 1 point for the two densities, 1 point for the deadline failure, 1 point for a reserved-lane design.

### Q6. Precision is stable at 0.24 but the threshold has fallen from 0.095 to 0.081 over two quarters. What is happening, and what do you monitor?

**Answer.** The queue is capacity-capped, so the effective cut floats. A falling threshold with stable precision means the score distribution shifted downward and the cut followed to keep 8,800 alerts flowing — the model is producing weaker evidence per event, and the flat precision is an artefact of the floating cut, not evidence of stability. Monitor threshold, alert rate, and score percentiles (p50/p90/p99) jointly.

**Reasoning.** With a floating cut, precision is held roughly constant by construction while the quality of the alerted set declines. The tell is that the threshold moved 15% while nothing else did. If instead the threshold were pinned, precision would have fallen and the drift would have been obvious.

**Follow-up.** Which single chart distinguishes drift from degradation? Threshold and precision on one time axis: same-direction movement means drift; precision alone falling means degradation.

**Red flag.** "Precision is flat, so the model is fine."

**Scoring.** 1 point identifying the floating cut, 1 point interpreting it correctly, 1 point for the monitoring set.

### Q7. Given the six-bin reliability table in Section 5, compute ECE and identify which band you would fix first.

**Answer.** ECE = 0.600×0.000167 + 0.250×0.000600 + 0.100×0.001000 + 0.040×0 + 0.009×0.007778 + 0.001×0.050000 = 0.000100+0.000150+0.000100+0+0.000070+0.000050 = 0.000470. Fix the top band first for credibility, not for ECE: it contributes only 0.000050 (10.6%) of ECE but is where every high-value decision happens and where 0.20 predicted versus 0.15 observed is a 25% relative over-prediction.

**Reasoning.** ECE weights by population, so the sparsest band is nearly invisible. But the top band governs the largest losses and the most expensive actions, and its error is 2.0 standard errors from zero (SE = sqrt(0.15×0.85/200) = 0.0252). Fixing it needs more labels, not a fitted map on 200 cases.

**Follow-up.** What would isotonic calibration on those 200 cases do? Overfit: with 30 positives it would produce a step function with huge variance and degrade log loss on hold-out.

**Red flag.** Choosing the band with the largest absolute deviation without considering population weight or decision importance.

**Scoring.** 1 point ECE, 1 point identifying the top band with a reason, 1 point for the SE argument.

### Q8. A colleague proposes optimising "cost per prevented fraud". Show numerically why this is the wrong objective.

**Answer.** Cost per prevented fraud is 100.5 at 6,000 alerts, 104.8 at 8,800, and 139.4 at 17,600. Minimising it selects 6,000 alerts, yielding net value 204,150. The value-maximising point is 17,600 with net value 358,240. The ratio objective therefore forgoes 154,090 per week.

**Reasoning.** The ratio has no term for volume: it can always be improved by shrinking the numerator's denominator, i.e. by catching fewer frauds. Any objective that improves when you do less of the useful thing is mis-specified. Net value internalises both magnitude and efficiency.

**Follow-up.** When *is* a ratio legitimate? As a constraint — "cost per prevented fraud must stay below 150" — which bounds the expansion without driving the choice.

**Red flag.** Agreeing that the ratio is a reasonable KPI.

**Scoring.** 1 point the three ratio values, 1 point the forgone value, 1 point for constraint-versus-objective framing.

### Q9. Your audit finds 15% of alerts have no disposition after 90 days. Quantify the effect on precision and on the decision.

**Answer.** Observed precision on disposed alerts only: if 15% of the 8,800 alerts (1,320) are undisposed and the 2,112 confirmed come from the 7,480 disposed, precision on disposed is 2,112/7,480 = 0.2824 versus 0.240 on all alerts. If the undisposed alerts contain fraud at the population rate, true precision is 0.240; if they contain fraud at a lower rate (likely, since easy cases get closed fast), precision is between 0.240 and 0.2824.

**Reasoning.** The direction of bias depends on whether non-disposition correlates with fraud. Undisposed alerts are typically the ones the analyst could not resolve — often genuinely ambiguous, sometimes fraud that aged out. Reporting 0.2824 without disclosing the 15% is an overstatement of 17.7% relative.

**Follow-up.** What is the minimum fix? A stratified sample of 200 undisposed alerts investigated to completion bounds the missing rate to about ±4 percentage points, enough to state a defensible interval.

**Red flag.** Dropping undisposed alerts from the denominator without comment.

**Scoring.** 1 point both precisions, 1 point the bias-direction reasoning, 1 point the sampling fix.

### Q10. Design the smallest experiment that would tell you whether a new model is better, given 8,800 alerts per week.

**Answer.** A/B on the alert stream with precision as the primary metric, 4 weeks per arm. Required sample per arm for detecting 0.24 versus 0.20 at 80% power, α = 0.05: n = 2(z_{0.975}+z_{0.80})²·p̄(1−p̄)/Δ² with p̄ = 0.22, Δ = 0.04, giving n ≈ 1,684 per arm — about 0.19 weeks at 8,800 per week, so 4 weeks per arm is far more than sufficient; the binding constraint is not sample size but the number of distinct fraud campaigns observed. Minimum detectable difference at 4 weeks per arm (n = 35,200): MDE = 2.802·sqrt(2·0.22·0.78/35,200) = 0.0087.

**Reasoning.** Statistical power is not the limiter here; temporal coverage is. Four weeks per arm spans at least four weekly attack cycles and two pay cycles. The correct design also randomises at the *event* level within each arm's window and pre-registers the guardrails (complaint rate, time-to-decision).

**Follow-up.** Why not randomise at the analyst level? Because analyst skill dominates handling time and confounds the precision comparison; event-level randomisation keeps the workload mix identical.

**Red flag.** Proposing a 12-month test, or testing on a single week.

**Scoring.** 1 point the sample-size calculation, 1 point the MDE, 1 point for separating statistical from temporal adequacy.

### Q11. The loss distribution has median 90 and mean 320. Where does ranking by p go wrong, and what do you rank by instead?

**Answer.** Ranking by p ignores L, so a case with p = 0.30 and L = 40 outranks a case with p = 0.12 and L = 3,000, even though the second is worth roughly 207 versus −1.5 in expected value. Rank by p·rho·L − c_a − (1−p)·c_f, i.e. by expected value per alert, which is monotone in p·L when costs are constant.

**Reasoning.** With L constant, ranking by p and by EV coincide, which is why the error survives testing on homogeneous data. With a heavy tail, the ordering diverges precisely in the tail that carries the loss. The threshold p* also moves per loss band: at L = 90, p* = 16.10/57.90 = 0.278 rather than 0.0803.

**Follow-up.** What breaks if you rank by p·L? Nothing economically, but the queue becomes dominated by a few large cases, so capacity planning must move from alert counts to minutes (Q5).

**Red flag.** "The model outputs a probability, so rank by probability."

**Scoring.** 1 point the counterexample, 1 point ranking by EV, 1 point per-band p*.

### Q12. Estimate the value of a 1 percentage point precision improvement at fixed capacity, and compare it with the value of halving FPR.

**Answer.** At 8,800 alerts, 1 point of precision is 88 additional true positives, worth 88×198.40 = 17,459 per week, plus 88 fewer false alerts saving 88×2.10 = 184.8, total 17,644. Halving FPR from 0.003369 to 0.001685 at fixed TPR raises precision to 0.002200/(0.002200+0.996×0.001685) = 0.002200/0.003878 = 0.5673 — a 32.7 point improvement, worth about 32.7×88×200.50 ≈ 577,000 per week.

**Reasoning.** The two framings describe the same curve from different ends. Precision improvements are cheap-sounding and small; FPR improvements at low prevalence are the dominant lever because the denominator is 249 times larger. Halving FPR is equivalent to roughly 33 precision points here, which is why it deserves the modelling budget.

**Follow-up.** Is halving FPR achievable? At FPR 0.0017 you would be flagging 3,380 legitimate events instead of 6,688, so the queue would have to be filled with higher-ranked alerts — check whether the score distribution has the mass to fill 8,800 at that precision.

**Red flag.** Treating precision points and FPR reductions as independent knobs.

**Scoring.** 1 point the precision-point value, 1 point the recomputed precision, 1 point the equivalence insight.

### Q13. An analyst says "we should lower the threshold to catch more fraud". Give the three-number rebuttal.

**Answer.** (1) The marginal band from 8,800 to 11,000 has marginal precision 0.1536, worth 0.1536×198.40 − 11.07 − 0.8464×2.10 = 30.48 − 12.85 = 17.63 per alert — positive, so expansion there is defensible; (2) the band from 17,600 to 21,000 has marginal precision 0.0471, worth 9.34 − 11.07 − 0.9529×2.10 = 9.34 − 13.07 = −3.73 per alert, so expansion past 17,600 destroys value; (3) each additional alert costs 9.2 minutes, so 2,200 more alerts is 2,200×9.2/60/30 = 11.2 FTE-weeks of work every week — 11 people, not a configuration change.

**Reasoning.** The rebuttal is not "no" but "up to a point, and here is its price in people". This reframes the conversation from model quality to staffing, which is where the decision actually lives.

**Follow-up.** What if the analyst offers to accept lower precision? Then the answer is the marginal band test: value per alert, not aggregate precision, decides, and the answer changes sign at 17,600, not at 8,800.

**Red flag.** Refusing on principle, or accepting without the marginal arithmetic.

**Scoring.** 1 point band values, 1 point FTE conversion, 1 point identifying the turning point.

### Q14. You must report one number to the board. What is it, and what three caveats go with it?

**Answer.** Report net value per week (281,776) with its assumption set, not precision. Caveats: (1) it depends on rho = 0.62, unverified without a randomised hold-out; at rho = 0.45 the figure is 166,883 and at L = 200 it is 124,643; (2) it is computed on a cohort at 90-day maturity with about 19.5% of positives still unconfirmed, so it is a lower bound on benefit; (3) it excludes the fixed cost of 70,169 per week of wages, so the margin over the wage bill is 211,607, not 281,776.

**Reasoning.** The board needs a figure that survives a challenge. Net value survives because it is constructed from auditable counts and stated costs; precision does not, because it invites the question "24% of what?" with no economic answer. Disclosing the caveats pre-empts the challenge and converts a number into a defensible position.

**Follow-up.** What if the board asks for a single ratio instead? Offer benefit-to-review-spend (3.05) and immediately note that it improves when you review less.

**Red flag.** Reporting AUC or precision as the headline, or reporting net value without the assumption set.

**Scoring.** 1 point the headline number, 1 point the sensitivity figures, 1 point the maturity caveat, 1 point the fixed-cost caveat.
## 12. Quick Reference Card

| Item | Formula or rule | Illustrative value | Watch out for |
|---|---|---|---|
| Precision from rates | PPV = pi·TPR / (pi·TPR + (1−pi)·FPR) | 0.181 at pi=0.004, TPR=0.55, FPR=0.01 | Precision without prevalence is meaningless |
| Precision at capacity | TP / alerts at the capacity cut | 2,112/8,800 = 0.240 | Rounded inputs break reconciliation; use integer counts |
| Recall at capacity | TP / all mature frauds | 2,112/8,000 = 0.264 | Immature labels bias it down 19.5% at 60 days |
| Value per true positive | rho · L | 0.62 × 320 = 198.40 | rho unverified without a randomised hold-out |
| Break-even probability | (c_a + c_f) / (rho·L + c_f) | 16.10/200.50 = 0.0803 | Loss-ratio form gives 0.0751; state which you used |
| Net value per period | TP·rho·L − a·c_a − (a−TP)·c_f | 281,776 per week | Excludes fixed wage cost (70,169) |
| Marginal band value | n_band·(mp·rho·L − c_m − (1−mp)·c_f) | +17.63 at 8.8k→11k; −3.73 at 17.6k→21k | Use marginal cost 11.07, not average 14.00 |
| Capacity in minutes | alerts × minutes per alert ÷ FTE minutes | 8,800×9.2/60 = 1,349.3 h; 44.98 FTE | Alert counts hide a 35x spread in handle time |
| Calibration error | ECE = Σ (n_b/N)·\|p_b − o_b\| | 0.000470 (13.1% of base rate) | Population weighting hides the top band |
| Threshold portability | logit shift = ln(odds(pi_new)/odds(pi_old)) | −0.697 for 0.004→0.002 | Valid only for equal-dispersion models, not reweighted GBMs |
