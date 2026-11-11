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
