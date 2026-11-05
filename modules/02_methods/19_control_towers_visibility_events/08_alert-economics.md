# Alert Precision, Recall, and the Economics of Attention

> Module 19 — Control Towers, Visibility, and Event Intelligence
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: 03, 04, 06, and 07 of this module
> Version 2026.5

## 1. Why this topic exists

A control tower can identify thousands of events and still fail the business.
The bottleneck is often the few minutes during which an operator can intervene.
A planner who investigates a harmless shipment cannot use the same time on a shipment about to miss a cutoff.
A missed exception may consume expensive recovery capacity or cause a customer commitment to fail.
The comparison is not between perfect information and no information.
It is between feasible allocation policies with different costs, timeliness, and outcomes.

An alert is a request for attention, not a resolution.
A risk score is a forecast, not an instruction to expedite.
A plan describes an intended action; a target states a desired result.
A customer commitment is a promise made to a counterparty, not the model's predicted arrival.
This distinction matters when the model predicts a late delivery but the customer has agreed to a later promise.

For example, a supervisor has 100 reviews available in a shift and 1,000 eligible shipments.
A threshold that generates 600 alerts does not produce 600 reviews.
It produces a queue in which work will be selected, delayed, abandoned, or escalated.
Measure the queue as operated, not only the classifier as designed.
The supervisor needs to know which cases to inspect, which actions are authorized, and when to stop.
Finance needs an incremental cost estimate that does not count avoided loss twice.
Customer service needs to know which claims remain uncertain.

The same numeric precision can support different decisions at different base rates, capacities, and severities.
A ten-minute investigation may be worthwhile for a disrupted production line and wasteful for a low-value delivery.
A missed cold-chain excursion may have safety consequences that cannot be reduced to average freight cost.
Where a safety or legal duty requires escalation, do not replace that duty with a profit threshold.
This topic gives a reproducible economic frame without claiming that all harms are interchangeable.

## 2. Learning outcomes

After this topic you can:

1. Construct a confusion matrix at a defined shipment-decision grain and time horizon (Exercise 1).
2. Calculate precision, recall, false-positive rate, and miss rate from the same cohort (Exercises 1–2).
3. Explain how prevalence and selection change the meaning of a metric (Exercise 3).
4. Derive a cost-sensitive action threshold and state its assumptions (Exercise 4).
5. Choose a queue under an attention constraint and calculate marginal value (Exercises 5–6).
6. Critique selectively observed labels and identify a defensible audit design (Exercise 7).
7. Specify decision rights, measurement, and safeguards for deployment (Exercise 8).
8. Reproduce the worked arithmetic with the standard-library Python artifact in section 7.

These outcomes concern decisions per cohort, not only a global model score.
Every metric must specify the eligible population, evaluation horizon, and actual handling policy.
A score can be well ranked and poorly calibrated, or calibrated and useless for ranking.
The exercise answers show which conclusion each calculation supports.

## 3. The core mental model

```text
eligible shipments at decision time
       |
       v
risk forecast + action deadline + potential loss
       |
       v
mandatory escalation? ---- yes ---> authorized safety/compliance process
       | no
       v
expected avoidable loss - action and review cost
       |
       v
rank feasible cases --> queue capacity and time --> actual review --> action
       |                                                 |             |
       +--> unreviewed / expired ------------------------+             v
                                                              observed outcome
       ^                                                               |
       +----------- randomized audit and delayed label -----------------+
```

The diagram contains two distinct selection points: the alert rule and the human queue.
An alert may be generated but never reviewed before the actionable deadline.
Counting it as successfully handled would confuse detection with intervention.
Observed outcomes come after decisions; action can prevent the very event the alert predicted.
A successful intervention can therefore look like a false positive if the label is the realized outcome.
Use an untreated-outcome estimand when evaluating risk, and use a policy experiment to assess avoided harm.
The counterfactual event is not directly observed for any individual shipment.

At the decision time, define the eligible cohort and freeze the information available then.
A later arrival scan can establish a label but must not leak into the earlier forecast.
An event may arrive late; event time, ingestion time, and processing time must be kept separate.
A correct alert that arrives after the only intervention deadline has little operational value.
A clean model metric must therefore be accompanied by lead time and deadline coverage.

In binary classification, precision asks: among alerts, what share were true cases?
Recall asks: among all true cases, what share were alerted?
False-positive rate asks: among noncases, what share triggered?
Miss rate asks: among cases, what share did not trigger?
Neither precision nor recall alone expresses dollars, minutes, rights, or customer harm.
A system can improve precision by alerting only on one obvious case while missing almost everything else.
It can improve recall by alerting on every shipment while exhausting the queue.

A decision rule needs a common unit of value and a feasible budget.
Expected value depends on probability, severity, action effectiveness, expense, and the time required.
When cases take different amounts of time, selecting the highest risk alone is not necessarily optimal.
When interventions depend on a shared carrier or replenishment constraint, independent case values may not add.
When a required escalation is non-discretionary, remove it from the optimization and reserve its capacity first.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Eligible cohort | Shipments satisfying a frozen rule at a decision time, counted in shipments | Using only alerted shipments as the denominator |
| Decision grain | One shipment and one decision opportunity in this example | Treating each repeated ping as an independent shipment |
| Horizon | Interval over which an outcome is assessed, in hours or days | Comparing labels from different horizons |
| Positive case | Shipment meeting the specified outcome definition | Equating a risk score with an observed event |
| Alert | Triggered request for attention, counted in shipment opportunities | Assuming every alert received a review |
| Review | Operator investigation completed before the deadline | Counting a queue entry as a completed review |
| Action | Authorized intervention, such as rebooking | Counting inspection as proof that action was taken |
| True positive (TP) | Alert and positive label in the same cohort | Treating prevented outcomes as observable positives |
| False positive (FP) | Alert and negative label | Assuming every negative observed after action was harmless untreated |
| False negative (FN) | No alert and positive label | Ignoring cases excluded from the eligible population |
| True negative (TN) | No alert and negative label | Including unknown or censored outcomes as negatives |
| Precision | TP / (TP + FP), dimensionless | Interpreting it as the fraction of all cases found |
| Recall | TP / (TP + FN), dimensionless | Interpreting it as the fraction of alerts that are useful |
| False-positive rate | FP / (FP + TN), dimensionless | Calling one minus precision the false-positive rate |
| Miss rate | FN / (TP + FN), dimensionless | Calling 1 − precision the miss rate |
| Prevalence | Positive cases / eligible cases in a defined cohort | Assuming the historical rate holds on a new lane |
| Calibration | Agreement of predicted probabilities with observed frequencies in comparable cases | Assuming ranking implies calibrated probabilities |
| Ranking | Ordering by a score or expected value | Assuming scores are probabilities |
| Precision at k | Positives in the first k positions divided by k | Ignoring whether those k cases were actually reviewed |
| Capacity | Available review time or case slots during a window | Equating theoretical staffing with usable review time |
| Opportunity cost | Value forgone by using scarce attention elsewhere, USD or minutes | Adding an arbitrary penalty without a baseline |
| Decision threshold | Score or probability cutoff for a defined action | Reusing a cutoff after costs or calibration change |
| Avoidable loss | Harm an available intervention can prevent, USD per case | Treating total exposure as fully preventable |
| Incremental cost | Additional expense relative to the comparator, USD per case | Counting already-sunk fixed platform fees per case |
| Intervention effect | Difference between outcomes under action and no action | Assuming an action always succeeds |
| Counterfactual | Outcome under an alternative action not observed for the same case | Treating a model prediction as a verified outcome |
| Selective labels | Outcomes observed or validated mainly for selected cases | Scoring only the reviewed subset as the whole population |
| Censoring | Outcome not yet known by the measurement cutoff | Assigning a negative label to an open shipment |
| Workload | Total expected review minutes over a defined period | Counting alerts without accounting for investigation time |
| Lead time | Decision deadline minus alert availability, in minutes or hours | Using event occurrence time when the tower received it later |
| Queue expiry | Time after which a review cannot support the intended decision | Letting a stale alert occupy active capacity |
| Shadow price | Incremental objective value of an additional unit of scarce capacity | Treating an estimated value as the employee's wage |
| Policy value | Expected outcomes and costs under a specified operational policy | Claiming predictive accuracy alone proves impact |

## 5. The mechanics, step by step

The following procedure is a decision design, not a claim that a classifier has causal effect.
Each step names its input, output, and the branch that changes the next decision.
### Step 1: Name the business decision

- **Input:** A shipment may miss a receiving appointment; a planner may request a rebooking.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Write one decision sentence, the owner, and the last useful action time.
- **Branch:** If no owner can act, do not launch an attention-consuming alert.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 2: Choose the denominator

- **Input:** Start from all eligible shipment opportunities at the shift opening, including unalerted cases.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Create a frozen cohort with stable shipment identifiers and eligibility reasons.
- **Branch:** If a shipment can appear in several shifts, define whether opportunities or unique shipments are the unit.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 3: Define the positive outcome

- **Input:** Use arrival after the contracted receiving cutoff, for example, with a 24-hour outcome horizon.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Record the cutoff, time zone, evidence source, and label maturity date.
- **Branch:** If the action changes the cutoff, retain the original commitment and separately record the new agreement.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 4: Separate available information from hindsight

- **Input:** Collect events actually received before the decision timestamp.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Store event time, ingestion time, source, and model version alongside each score.
- **Branch:** If data arrives late, replay it as late rather than injecting it into the earlier forecast.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 5: Specify the action and comparator

- **Input:** Compare review followed by an authorized intervention with business-as-usual handling.
- **Work:** Record the decision context before examining downstream results.
- **Output:** List review time, possible remedies, costs, and the no-alert workflow.
- **Branch:** If the no-alert team already catches cases, estimate incremental rather than gross avoided loss.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 6: Classify mandatory cases first

- **Input:** Apply mandatory safety, contractual, or regulatory escalation rules as interpreted by the responsible owner.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Reserve staff and document the rule and its source.
- **Branch:** If mandatory work consumes capacity, optimize the remaining discretionary capacity, not the original total.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 7: Estimate untreated risk

- **Input:** For each eligible case estimate the chance of the defined outcome without the proposed intervention.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Keep risk predictions separate from action-effect estimates and display uncertainty.
- **Branch:** If predictions are uncalibrated, use ranking cautiously and do not interpret score cutoffs as probabilities.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 8: Estimate preventable severity

- **Input:** For each case estimate loss under the comparator and the fraction that an intervention can avoid.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Use a documented unit such as USD per eligible shipment, with a loss distribution when possible.
- **Branch:** If the harm includes non-monetizable safety or rights consequences, escalate under the separate rule.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 9: Include real incremental costs

- **Input:** Count paid review time, intervention charges, and any extra operational consequences versus the comparator.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Produce a cost basis per case and note which cost occurs only after a positive investigation.
- **Branch:** If review duration varies, use minutes per case rather than a uniform case count.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 10: Build the confusion matrix

- **Input:** Join alerts at the chosen cutoff to mature, consistently defined outcomes.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Count TP, FP, FN, and TN with unknown outcomes reported separately.
- **Branch:** If action prevented events, the observed matrix is not an untreated-risk matrix.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 11: Compute diagnostic rates

- **Input:** Divide TP by alerts for precision and TP by positives for recall.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Also report FP divided by all negatives and FN divided by all positives.
- **Branch:** If a denominator is zero, report undefined rather than assigning a comforting zero.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 12: Account for the base rate

- **Input:** Measure prevalence in the target population and compare it with historical training data.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Calculate expected alert counts and positive yield at relevant prevalence values.
- **Branch:** If the population mix changes, test calibration and workload before retaining the old cutoff.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 13: Derive a simple decision cutoff

- **Input:** For calibrated untreated probability p, preventable positive-case benefit B, and unconditional review cost c, compare pB with c.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Under those special assumptions, review when p exceeds c/B.
- **Branch:** If false-alarm costs or positive-action costs differ, use the full incremental payoff equation instead.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 14: Model capacity explicitly

- **Input:** Subtract breaks, mandatory work, and other duties from shift minutes.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Select a feasible set whose expected review minutes fit the remaining capacity.
- **Branch:** If work has unequal durations, deadlines, and coupled actions, solve or approximate a scheduling problem.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 15: Rank by expected incremental value

- **Input:** Estimate benefit minus cost for each feasible case, not merely raw risk.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Use value per scarce minute as a useful heuristic for divisible work with one capacity constraint.
- **Branch:** For indivisible cases the ratio heuristic can be suboptimal; compare with a knapsack solution.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 16: Define the queue policy

- **Input:** Specify alert threshold, tie break, deduplication, expiry, reassignment, and escalation.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Log the generated list and the actual start and completion of every review.
- **Branch:** If the queue overflows, report the unreviewed population and whether urgent cases expired.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 17: Measure timely handling

- **Input:** Join review timestamps to individual action deadlines.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Calculate reviewed-before-deadline share and late or expired backlog.
- **Branch:** If a case is reviewed too late, do not count it as a timely operational success.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 18: Audit what was not reviewed

- **Input:** Sample unalerted and unreviewed cases independently of the current queue where feasible.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Record sampling probabilities and obtain mature labels through the same adjudication process.
- **Branch:** If labels require intervention to exist, use an alternative outcome source or explicitly state nonidentification.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 19: Evaluate impact, not only prediction

- **Input:** Compare an authorized pilot policy against a credible business-as-usual comparator.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Report incremental harm, review effort, and uncertainty at an appropriate randomization unit.
- **Branch:** If action spillovers connect shipments, randomize compatible clusters or use another defensible design.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Step 20: Govern and revise

- **Input:** Assign a model owner, queue owner, decision authority, and review cadence.
- **Work:** Record the decision context before examining downstream results.
- **Output:** Freeze a versioned policy and monitor workload, performance, timeliness, and subgroup effects.
- **Branch:** If the process drifts or staffing changes, reassess rather than silently changing the threshold.
- **Control:** Retain the decision timestamp, owner, and version for audit.
### Threshold arithmetic beyond the simple case

Let p be the calibrated probability of the untreated event.
Let B be the avoided loss if the event would occur and the action succeeds, multiplied by the probability of success.
Let c be a review cost paid for every reviewed case.
With no other effects, expected incremental value is p × B − c.
When B is positive, the break-even probability is c / B.
At equality, the operator is indifferent under this simplified model.
For B = USD 500 per positive case and c = USD 10 per review, break-even is 0.02, or 2%.
A 2% threshold is not automatically deployable if the resulting workload exceeds capacity.

A different model may have cost c1 if the event would occur and c0 otherwise.
The expected incremental value is p × (B − c1) − (1 − p) × c0.
When the denominator is positive, the break-even probability is c0 / (B − c1 + c0).
Do not add c again if c0 and c1 already include review.
If the benefit or cost depends on time, calculate it at the expected handling time, not alert creation.
When p is a score without calibration, a numerical break-even probability cannot be applied directly.

### Attention as a scarce input

For a single constrained resource, a positive case value can still be displaced by a higher-value case.
A capacity shadow price represents the marginal value of another minute of that resource.
Review an additional case only when its incremental value exceeds the opportunity cost of its minutes.
This is a conceptual rule; the exact discrete optimum can differ from greedy ratio ordering.
A queue with urgent deadlines can also require a lower-value case to be handled earlier.
A threshold may be chosen on historical data, but prospective arrivals and staffing make the realized queue stochastic.
Track the work actually completed and the cases left waiting as separate populations.

### Step 21: Reconcile the cost ledger

Start with a ledger rather than an undifferentiated estimate of the cost of a miss.
For each cost component, name the party paying it and the period in which it occurs.
Mark whether the amount is incurred under the proposed policy, the comparator, or both.
Only the difference between policies belongs in incremental net value.
The underlying amounts should remain available for a finance reconciliation.

| Component | Possible comparator cost | Possible proposed-policy cost | Treatment |
|---|---|---|---|
| Planner investigation | Existing manual checks | Alert-driven review | Difference in paid time, not all staff salary |
| Carrier rebooking | Reactive rebooking | Earlier rebooking | Difference in charges, including cancellation terms |
| Dock labor | Overtime after a late arrival | Rescheduled receiving labor | Difference in actually avoidable labor |
| Inventory | Stockout or emergency inventory | Earlier replenishment | Avoid double-counting the same lost margin |
| Customer service | Reactive calls | Proactive communication | Net change in effort and service outcome |
| Platform subscription | Existing contract | Incremental license or usage | Include incremental fee over the evaluation horizon |
| Claims or penalties | Applicable contractual charge | Charge remaining after action | Include only expected paid amounts under the contract |
| Safety response | Required response under either policy | Required response under new policy | Fund and report separately; do not waive the duty |

A fixed cost can matter for a launch decision even when it is irrelevant to a per-case threshold.
For example, a USD 6,000 monthly license is not an incremental cost of reviewing one more alert under an existing contract.
It is relevant when comparing adoption against a comparator that avoids the license altogether.
Use consistent time buckets when combining a monthly license with a daily benefit.
Do not multiply a gross avoided disruption loss by the number of alerts.
Multiply the relevant conditional benefit by the probability of the event and the probability and magnitude of prevention.

The finance owner should also distinguish accounting savings from released capacity.
Fifteen minutes returned to an analyst has operational value only if the organization can use it productively.
It does not automatically become a cash saving.
Likewise, avoiding an overtime shift can create a cash saving only if it is actually canceled.
Report avoided expense, released capacity, and customer outcome as separate lines.
Assign a source and confidence range to each important ledger entry.

### Step 22: Compare decisions with imperfect interventions

Review is not necessarily the intervention that prevents the miss.
Review may reveal whether a remedy is available; the operator then decides whether to exercise it.
Suppose a review costs USD 8 regardless of outcome.
On a shipment that would miss its appointment, the review enables a rebooking 50% of the time.
Each successful rebooking avoids USD 300 of loss and costs USD 40.
Assume no rebooking occurs for an untreated noncase and that unsuccessful review has no extra action cost.
The conditional net benefit on an untreated positive is 0.50 × (USD 300 − USD 40) = USD 130.
The expected net value of review is p × USD 130 − USD 8.
The break-even probability is USD 8 / USD 130, approximately 6.15%.
This differs from dividing USD 8 by the entire USD 300 exposure.

If the operator sometimes rebooks a noncase, add the corresponding expected cost.
For example, if the unnecessary rebooking costs USD 40 and happens for 10% of untreated noncases,
the additional expected cost is (1 − p) × 0.10 × USD 40.
The complete expression is p × USD 130 − USD 8 − (1 − p) × USD 4.
It simplifies to p × USD 134 − USD 12.
The break-even probability rises to USD 12 / USD 134, approximately 8.96%.
This calculation assumes the noncase rebooking produces no compensating benefit.
If it also causes disruption, add that incremental harm; if it produces value, include that value.
The probabilities must describe the proposed policy relative to its comparator, not generic historical averages.

### Step 23: Establish a prospective scoring and handling contract

Before launching a shift, document the sequence at which information becomes available.
An example contract is:

1. Freeze eligible open shipments at 06:00 local site time.
2. Convert source timestamps to UTC while retaining their original time zone and offset.
3. Use only events ingested by the 06:00 scoring cutoff.
4. Apply the separately approved mandatory-escalation rules.
5. Estimate remaining analyst minutes after scheduled duties.
6. Rank discretionary cases using the approved expected-value model.
7. Assign cases to named operators and retain each assignment timestamp.
8. Recompute or expire a case when new information changes its actionability.
9. Log completion and action time separately from the initial alert time.
10. Join to mature outcomes after the measurement horizon.

The cutoff is a policy choice, not a statement that every site should score once per day.
An event arriving at 06:01 may deserve an incremental rescoring process.
That process needs its own capacity and preemption rules.
Do not retrospectively insert the 06:01 event into the 06:00 prediction.
If a shift spans a daylight-saving transition, retain the offset used to interpret each local time.
A scheduled review at an ambiguous local time needs an unambiguous actual instant.

### Step 24: Decide how a queue competes with incoming work

A daily top-k list assumes the relevant cases are known when ranking occurs.
Live events challenge that assumption.
An arrival at 09:00 can be more valuable than the case an analyst is currently investigating.
The policy must say whether interruption is permitted and what it costs.
It must also say whether an urgent case can preempt a less urgent one.
The cost of switching includes partially completed investigation and the time needed to regain context.

For a simple operational policy, use three fields for every case:

| Field | Purpose | Example |
|---|---|---|
| Latest useful start | Protect the intervention window | 10:20 local time |
| Estimated duration | Estimate capacity consumption | 12 analyst minutes |
| Expected net value | Compare benefits and costs | USD 55 per case |

At each assignment decision, first remove cases that can no longer be completed in time.
Among remaining discretionary cases, choose using a documented rule that accounts for deadlines.
Where values are comparable and deadlines are not binding, value per minute is a useful starting heuristic.
When deadlines bind, an exact schedule or a tested dispatch heuristic may perform better.
After an interruption, update remaining work rather than charging the whole duration again without evidence.
Keep a log of preemptions so that service teams can inspect disruption to their work.

The practical comparator is the current dispatch policy, not a hindsight-optimal schedule.
A hindsight schedule can be a useful upper bound but knows the future.
An online policy cannot rank an alert before the data needed to generate it has arrived.
Backtests should replay arrival order and the actual information available at each time.
They should include queue aging, staffing changes, and the time needed for each action.

### Step 25: Make the audit sample support population estimates

Construct strata before drawing cases, for example alerted and nonalerted shipments.
Sampling more nonalerts can make missed cases easier to find, but the sample will not be self-weighting.
Keep each case's selection probability and adjudication result.
If a shipment in a stratum has inclusion probability q, its inverse-probability weight is 1 / q.
Use those weights to estimate counts in the full stratum, provided selection probabilities are known and nonzero.
For example, auditing 10% of 900 nonalerts gives each sampled case weight 10.
If the random audit finds 6 positives among 90 audited nonalerts, the estimated nonalert positives are 60.
The estimate is uncertain; it is not a statement that precisely 60 misses occurred.

If 100 alerted cases are all adjudicated and 30 are positive, estimated recall is 30 / (30 + 60), or 33.3%.
This is a ratio of estimated population totals.
The result depends on representative random sampling within the nonalert stratum and consistent labels.
It can be unstable when only a few positives are found.
Report the 6 observed positives, 90 audits, and the sampling design alongside the estimate.
Use a design-appropriate uncertainty interval rather than treating 60 as a census count.
Nonresponse or missing outcome evidence within the audit can still bias the estimate.
Track such cases explicitly and investigate whether their missingness differs by lane or carrier.

Auditing identifies an observed outcome only if the evidence exists and can be interpreted.
It does not recover an individual untreated outcome after an effective intervention.
For that distinction, the evaluation design needs a suitable comparator.
Randomization is not appropriate when withholding an action would violate an obligation.
In that situation, keep the mandatory process intact and evaluate only the discretionary portion.

### Step 26: Design a pilot decision before observing its results

Write down the comparator, allocation unit, intended duration, and decision rule before the pilot starts.
The comparator should specify what staff actually do without the new queue.
If manual escalation already catches a late shipment, the new system can claim only incremental impact.
Define whether the policy changes selection, treatment, staffing, or all three.
Otherwise an observed difference cannot be attributed to a particular operational change.

Where feasible and authorized, randomize comparable shifts, lanes, or sites to the new policy and business as usual.
Choose a unit large enough to limit contamination from shared analysts and carrier capacity.
For each assigned unit, record all eligible shipments, not just handled cases.
Analyze outcomes according to the assigned policy as the primary comparison.
That preserves the policy contrast when operators do not follow every recommendation.
Also describe adherence and reasons for overrides without treating adherent cases as a randomized group.

Specify at least one business outcome, one workload outcome, and one safety or service guardrail.
An example business outcome is total incremental cost in USD per eligible shipment.
An example workload outcome is analyst minutes per 100 eligible shipments.
An example guardrail is the number of mandatory escalations completed after their deadlines.
These examples are design choices, not universal standards.
Define outcome maturity and how cancellations, missing shipments, and changed appointments are handled.
Estimate uncertainty at the randomization unit, not by pretending correlated shipments are independent.

Before rollout, agree who can stop the pilot for a safety or operational concern.
Agree who approves expanded use if benefits remain uncertain.
Avoid repeatedly changing the primary success criterion after inspecting interim outcomes.
If the sample is too small to distinguish a useful benefit from zero, report that uncertainty.
An inconclusive pilot is not proof that the policy has no effect or that it is safe to scale.

### Step 27: Maintain a decision log after deployment

Keep a compact record for each threshold or policy revision.
The record should include:

- The effective timestamp and time zone of the change.
- The responsible operator and approver.
- The previous and replacement policy versions.
- The forecast workload and the available staffed minutes.
- The comparator and evidence used to estimate effects.
- The mandatory-rule inventory and reserved capacity.
- The impacted lanes, customers, and shifts.
- The reason for the change and the alternatives considered.
- The measurement window and expected label maturity date.
- The rollback trigger and the person authorized to act.

The log makes a future difference-in-performance analysis interpretable.
It also helps an operator explain why an apparently similar shipment received a different priority.
Do not infer that a policy revision caused an outcome merely because it preceded it.
Weather, carrier capacity, and customer mix can change at the same time.
Where those factors matter, design the evaluation to address them rather than relying on the log alone.

## 6. Worked example

### Business setup

A regional distributor manages 1,000 eligible inbound shipment opportunities in one day.
The decision is whether a planner should review an arriving shipment before the appointment cutoff.
A risk model supplies a forecast of missing the original receiving appointment without the proposed review.
The operating plan schedules reviews; it is not a commitment that every alert will be handled.
The target is fewer missed appointments, while the commitment is the original agreed receiving cutoff.
The appointment and intervention deadlines are known at scoring time in this simplified case.
All examples in this section are invented for instruction, not reported company results.

The outcome horizon ends when each appointment is observed.
Suppose an independent mature label for the untreated outcome is available in this numerical illustration.
That assumption is deliberately strong; operationally, prevention makes the untreated outcome counterfactual.
Each positive case would create USD 500 of loss under the comparator.
A timely review prevents that full loss for each positive case, with no further intervention charge.
Each review, including a false alarm, costs USD 10 in additional labor.
Each review consumes the same amount of the scarce shift capacity.
The team has capacity for exactly 100 reviews in this day, with all reviews completed before deadlines.
These assumptions are for arithmetic, not a claim about actual intervention effectiveness.

### Frozen cohort and score bands

| Ordered score band | Eligible shipments | Untreated positives | Untreated negatives | Observed band prevalence |
|---|---:|---:|---:|---:|
| A: highest | 100 | 60 | 40 | 60% |
| B | 200 | 60 | 140 | 30% |
| C | 300 | 45 | 255 | 15% |
| D: lowest | 400 | 20 | 380 | 5% |
| Total | 1,000 | 185 | 815 | 18.5% |

The ordering is a ranking, and band prevalences are retrospective observations for this invented cohort.
They are not proof that individual model probabilities are calibrated.
The unreviewed comparator has 185 positive shipments and loss of 185 × USD 500 = USD 92,500.

### Compare feasible and infeasible cutoffs

| Included bands | Alerts | TP | FP | FN | TN | Precision | Recall | Review cost | Residual loss plus review cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| None | 0 | 0 | 0 | 185 | 815 | undefined | 0% | USD 0 | USD 92,500 |
| A | 100 | 60 | 40 | 125 | 775 | 60.0% | 32.4% | USD 1,000 | USD 63,500 |
| A–B | 300 | 120 | 180 | 65 | 635 | 40.0% | 64.9% | USD 3,000 | USD 35,500 |
| A–C | 600 | 165 | 435 | 20 | 380 | 27.5% | 89.2% | USD 6,000 | USD 16,000 |
| A–D | 1,000 | 185 | 815 | 0 | 0 | 18.5% | 100% | USD 10,000 | USD 10,000 |

For A, precision is 60 / (60 + 40) = 60%.
Recall is 60 / (60 + 125) = 32.43%, rounded to 32.4%.
The false-positive rate is 40 / 815 = 4.91%.
The miss rate is 125 / 185 = 67.57%.
The 40 false alerts consume USD 400 of the USD 1,000 review cost.
The remaining 60 reviews consume USD 600; their cost is not zero merely because they find cases.
The 125 misses correspond to USD 62,500 in residual loss under the stated assumptions.
Thus residual loss plus all review cost is USD 62,500 + USD 1,000 = USD 63,500.
The incremental net saving against no reviews is USD 92,500 − USD 63,500 = USD 29,000.

For A–B, recall rises to 120 / 185 = 64.86%, but 300 reviews exceed the 100-review budget.
Its USD 35,500 figure assumes all 300 reviews are completed, so it is not a feasible daily policy here.
Generating 300 alerts while handling only the top 100 yields the A outcome, not the A–B outcome.
If the actual queue handles a different 100, measure that actual selection rather than assuming A.
A lower alert threshold can increase the backlog without increasing realized recall.
A perfect review-all outcome is likewise infeasible at the given capacity.

### Value of extra capacity

Band B adds 60 positives in 200 reviews.
At full effectiveness, its gross avoidable loss is 60 × USD 500 = USD 30,000.
Its incremental labor cost is 200 × USD 10 = USD 2,000.
Its incremental net value is USD 28,000 for 200 extra review slots, or USD 140 per slot on average.
A proposal to add capacity should compare that net value with capacity-specific costs not already included.
Do not subtract the same labor cost again as if it were an additional capacity cost.
Average band value is not proof that each individual slot has the same marginal value.
The next slot depends on within-band ranking, time, and staff availability.

Band C has net value of 45 × USD 500 − 300 × USD 10 = USD 19,500.
Band D has net value of 20 × USD 500 − 400 × USD 10 = USD 6,000.
Under this retrospective arithmetic all bands have positive average value, even though precision falls.
The ranking makes A the best of these equal-duration bands, not a universal threshold for every network.
An audited prospective pilot is needed to estimate whether the assumed full prevention is realistic.

### Decision and handoff

The supervisor approves the top 100 timely reviews for this shift, subject to mandatory escalations.
The dashboard shows eligible count, alerts, completed reviews, expiry, and mature outcomes separately.
An owner reviews severity, staffing, and calibration each week before changing the policy.
Customer communication follows decision rights; the model does not automatically change a promise.

### Sensitivity to intervention effectiveness

The base example assumes every timely review prevents the loss on an untreated positive.
That is deliberately optimistic.
Suppose instead that a timely review prevents 40% of the USD 500 loss on average for an untreated positive.
Assume the same 100 band-A shipments are reviewed and the USD 10 review cost is unchanged.
This is a scenario calculation, not a measured treatment effect.

| Quantity | Full-prevention scenario | 40%-prevention scenario |
|---|---:|---:|
| Untreated positives in A | 60 shipments | 60 shipments |
| Gross loss exposed in A | USD 30,000 | USD 30,000 |
| Avoided loss in A | USD 30,000 | USD 12,000 |
| Review cost in A | USD 1,000 | USD 1,000 |
| Incremental net value | USD 29,000 | USD 11,000 |
| Total cost including review | USD 63,500 | USD 81,500 |

The 40% scenario computes avoided loss as 60 × USD 500 × 0.40 = USD 12,000.
The residual loss over the entire cohort is USD 92,500 − USD 12,000 = USD 80,500.
Adding USD 1,000 in review cost gives USD 81,500.
The resulting USD 11,000 net saving is lower than the USD 29,000 base-case saving.
Neither scenario estimates the effect of the actual operational policy without supporting evidence.
The causal effect may vary by lane, deadline, and available carrier alternative.

The band-A precision stays at 60% in both calculations because the hypothetical untreated labels do not change.
Economic value changes because the remedy's effectiveness changes.
This is the essential distinction between identifying a likely miss and being able to prevent its loss.
Even perfect prediction does not create benefit if no feasible action can change the outcome.

### Sensitivity to capacity and backlog

Now suppose the team receives only 70 discretionary review slots after mandatory work.
The model still generates 100 alerts for band A.
Without within-band information, assume for illustration that 70 cases are randomly selected from the band.
The expected number of untreated positives reviewed is 70 × 60 / 100 = 42 shipments.
At full prevention, expected avoided loss is 42 × USD 500 = USD 21,000.
Review costs are 70 × USD 10 = USD 700.
The expected net saving is USD 20,300 per day relative to the no-review comparator.
Expected remaining loss plus review cost is USD 92,500 − USD 21,000 + USD 700 = USD 72,200.

These are expectations under random selection, not guaranteed results for the 70 actual shipments.
The alert-level precision remains 60% for the full 100-case band.
The expected timely-handling share is 70 / 100 = 70% of alerts.
The expected timely detection of untreated positives is 42 / 185, approximately 22.7%.
The 30 unreviewed alerts must remain visible as unreviewed or expired.
They are not automatically false alarms and are not automatically treated positives.

If the 70 cases are chosen using additional valid information, their positive share may differ from 60%.
Estimate that share from evidence available at selection time.
If selection prioritizes shipments close to deadline, test the resulting effectiveness as well as precision.
Do not apply the full-band result to a nonrandom subset without justification.
If mandatory cases are included in band A, specify whether those cases consume the 70 discretionary slots.
The scenario assumes mandatory work was already removed from the available capacity.

### Reconcile the recommendation to the operational plan

The recommended action is not to lower the cutoff until the dashboard reports better recall.
The immediate plan is to assign the highest-value feasible cases to the available operators.
The forecast is the model's estimated chance of an untreated missed appointment.
The target is an operational objective for missed appointments and timely handling.
The customer commitment remains the appointment agreed with the customer.
Only an authorized party can change that commitment.

Before the next shift, the supervisor should answer:

1. How many net discretionary minutes remain after mandatory work?
2. Which cases can still be reviewed before their latest useful start?
3. Is the likely intervention actually available for each case?
4. Which alerts are duplicates or refer to an already resolved shipment?
5. Which partner needs to approve an action or a changed appointment?
6. How will unreviewed cases be audited after their outcomes mature?
7. What effect and cost assumptions require prospective evidence?

The daily handoff should list counts and minutes by queue state.
The weekly handoff should add mature outcomes, estimates of incremental effects, and uncertainty.
Finance should reconcile labor, carrier fees, and service costs against the comparator.
The service owner should independently inspect missed commitments and overdue escalations.
This division prevents a single attractive model metric from becoming the entire decision.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the band table and calculates the cost if every alerted case is reviewed on time.
It does not train a model or infer counterfactual outcomes.
The final feasibility flag uses a capacity of 100 reviews per day.
The precision field is `undefined` when no cases are alerted.
Run it as `python alert_economics.py` after saving the block in that file.

```python
bands = [
    ("A", 100, 60),
    ("B", 200, 60),
    ("C", 300, 45),
    ("D", 400, 20),
]
capacity = 100
loss_per_positive = 500
review_cost = 10
population = sum(size for _, size, _ in bands)
positives = sum(positive for _, _, positive in bands)
negatives = population - positives


def report(label, alerted, tp):
    fp = alerted - tp
    fn = positives - tp
    tn = negatives - fp
    precision = "undefined" if alerted == 0 else f"{tp / alerted:.1%}"
    recall = f"{tp / positives:.1%}" if positives else "undefined"
    total_cost = fn * loss_per_positive + alerted * review_cost
    feasible = alerted <= capacity
    print(
        f"{label:4} alerts={alerted:4} TP={tp:3} FP={fp:3} "
        f"FN={fn:3} TN={tn:3} precision={precision:>9} "
        f"recall={recall:>6} cost=USD {total_cost:>6,} "
        f"feasible={feasible}"
    )


report("none", 0, 0)
alerted = 0
tp = 0
for label, size, positive in bands:
    alerted += size
    tp += positive
    report(label, alerted, tp)
```

Expected output, with alignment dependent only on the formatting shown:

```text
none alerts=   0 TP=  0 FP=  0 FN=185 TN=815 precision=undefined recall=  0.0% cost=USD 92,500 feasible=True
A    alerts= 100 TP= 60 FP= 40 FN=125 TN=775 precision=    60.0% recall= 32.4% cost=USD 63,500 feasible=True
B    alerts= 300 TP=120 FP=180 FN= 65 TN=635 precision=    40.0% recall= 64.9% cost=USD 35,500 feasible=False
C    alerts= 600 TP=165 FP=435 FN= 20 TN=380 precision=    27.5% recall= 89.2% cost=USD 16,000 feasible=False
D    alerts=1000 TP=185 FP=815 FN=  0 TN=  0 precision=    18.5% recall=100.0% cost=USD 10,000 feasible=False
```

The code reports cutoff labels A, B, C, and D as cumulative inclusion through that band.
Review-all costs are hypothetical when `feasible=False`.
Costs are USD per day for the frozen cohort; no discount rate is relevant to this one-day example.
Unknown labels and action-dependent labels are excluded by construction, not solved by the code.
A production implementation needs joins, timestamp controls, audit sampling, and authorization.

## 8. Data, standards, and regulatory context

A threshold calculation inherits the meaning and quality of its inputs.
Retain the shipment identifier, order or purchase-order association, site, carrier, and decision timestamp.
Keep both the original appointment commitment and subsequent authorized changes.
Store event occurrence time and time of availability to the tower.
Store the score, model version, rules version, alert timestamp, and queue state history.
Record assignment, review start, review end, action, approver, and action time.
Keep the label definition, source, maturity date, and evidence of later corrections.
Unknown, disputed, and censored outcomes must remain distinguishable from negatives.
Record operational capacity and mandatory work in minutes per shift.
A measured review duration and a planning assumption should be distinguishable.

GS1 EPCIS 2.0 is an event-data standard describing what, when, where, why, and how of events.
Its identifiers and event structures can support traceability across parties.
It does not define the business cost of a false alarm or guarantee timely, complete partner data.
An EPCIS event can be structurally valid and still refer to the wrong shipment or arrive too late.
Use the partner agreement to define data sharing, quality obligations, and permitted uses.

The NIST AI Risk Management Framework 1.0 is voluntary guidance, not a universal legal mandate.
Its govern, map, measure, and manage functions support risk ownership and documentation.
The framework does not prescribe one precision target or establish that a given threshold is lawful.
Depending on use, personal-data or sector-specific obligations may apply to workforce and customer records.
The EU General Data Protection Regulation (GDPR) specifies principles including data minimization and storage limitation.
GDPR applicability depends on whether personal data is processed and the relevant legal and territorial context.
A shipment identifier can become personal data if it identifies or is linkable to a person.
Do not infer a universal retention period from this teaching example.

The EU AI Act is a risk-based regulation with provisions applying on different dates and to different roles.
A control-tower alert is not automatically a high-risk AI system merely because it uses a model.
Assess the actual purpose, applicable classification, role, and current law with qualified counsel.
No section here substitutes for a legal determination or overrides mandatory safety procedures.
The sources section links primary texts; legal requirements must be checked as of deployment.

### Minimum analytical records

Use a shipment-opportunity table as the denominator.
An event stream alone is not sufficient because shipments with no events may disappear from the analysis.
The table should identify both included and excluded opportunities and their eligibility reasons.
Keep source-system identifiers alongside a stable internal identifier.
If partners disagree on an identifier, retain the mapping decision and its effective dates.

| Record | Required fields for this analysis | Control question |
|---|---|---|
| Opportunity | Shipment ID, decision time, eligibility, original commitment | Does every eligible shipment appear exactly once per defined opportunity? |
| Event | Event ID, shipment ID, source, event time, ingestion time | Was the event available before the forecast? |
| Forecast | Opportunity ID, probability or score, model version, forecast time | Is the score a calibrated probability for this horizon? |
| Alert | Opportunity ID, rule version, creation time, priority | Can an alert be distinguished from a duplicate notification? |
| Queue history | State, owner, entry time, exit time, expiry | Was the case actually reviewed while action was still possible? |
| Intervention | Action type, action time, approver, expense | What was done, by whom, and at what incremental cost? |
| Outcome | Original cutoff, arrival evidence, maturity and dispute status | Is this an observed outcome or an untreated counterfactual? |
| Sampling | Stratum, inclusion probability, selection and response | Can the audit be generalized to the population? |
| Capacity | Shift, available minutes, mandatory workload, absences | Does the reported capacity reflect usable attention? |
| Economics | Cost component, currency, period, comparator, source | Is the same cost counted once and on the correct side? |

The queue history is an event log: a shipment can enter, be assigned, be requeued, and expire.
Do not overwrite the original assignment when responsibility moves.
A single current-state snapshot cannot show how long the case waited at each stage.
Store corrections as traceable revisions rather than silently changing the old analytical record.

If an appointment is amended, store both the original and the amended commitment.
The new appointment may be valid for customer service, while the original is needed for the original-miss measure.
Separately record whether the customer accepted the revision.
Do not call a missed original appointment on time merely because a later timestamp replaced it.
On the other hand, do not label a shipment late against a superseded promise without explaining the measure.

### Quality controls at the data boundary

Reconcile the opportunity count to a source such as the shipment management system's eligible manifest.
Report unmatched identifiers and late-arriving records rather than silently dropping them.
For every source, measure the lag from event occurrence to ingestion.
Use percentiles of that lag as well as the mean when late events are operationally important.
Check whether the lag differs for particular carriers, sites, or shifts.
A single overall average can hide a partner that supplies nearly all updates after the action deadline.

Test timestamp ordering while allowing for documented source-clock errors.
An arrival event with an occurrence time before dispatch may indicate a mapping or clock problem.
An event created before the score but ingested afterward was unavailable to the model at scoring.
An ingestion timestamp after the alert does not by itself prove the underlying event happened afterward.
Record time zone and offset when a source supplies local time.
Flag missing or ambiguous time-zone information for remediation.

Test the relationship between expected and observed review duration.
If the queue assumes 10 minutes per case but difficult cases take 35 minutes, the capacity plan will fail.
Compare estimated duration with actual start and completion timestamps by case type.
Account for pauses, handoffs, and simultaneous work before calling the elapsed interval pure labor time.
Have the operating team review outliers so that system logging artifacts do not become policy inputs.

### Access and retention decisions

Use the least detail necessary for the stated operational and analytical purpose.
For example, a quality analyst may need a shipment-level pseudonymous key but not a driver's personal phone number.
Pseudonymization alone does not establish that data is anonymous under applicable law.
Access permissions, retention periods, and deletion procedures should follow the organization's applicable policy.
Check obligations arising from contracts and relevant jurisdictions with the responsible privacy and legal owners.
Avoid placing personal information in free-text alert explanations if a structured reason code will suffice.

The fact that an event is exchangeable under a standard does not create permission to share it.
Document who supplies the event, who may receive it, and which purposes are authorized.
Record correction and dispute routes when a partner challenges an event.
If a source is removed or a sharing agreement changes, assess the effect on monitoring coverage.
Loss of one carrier's data can change both the population and the measured precision.

## 9. Industry practice and case evidence

The following sources support mechanisms and governance, not the numerical example's impact claims.
The public evidence cited here does not establish this topic's USD values or review productivity.

**UPS ORION.** UPS describes its On-Road Integrated Optimization and Navigation system as a route optimization technology.
The company's description illustrates that a recommendation must meet real operating constraints.
[INFERENCE] A tower allocating analyst attention similarly needs to account for feasible capacity and deadlines.
This analogy does not establish an alert-precision improvement or a causal estimate for the worked case.

**FDA adverse-event surveillance.** The U.S. Food and Drug Administration describes the FDA Adverse Event Reporting System and warns about interpretation of its reports.
The reporting system illustrates why spontaneously reported events are not a representative denominator for population incidence.
[INFERENCE] A tower that labels only investigated alerts faces a related selection problem.
These contexts differ: adverse-event reports are not shipment-level counterfactual labels.
The example supports the need for denominator discipline, not equivalence of their datasets.

**Alarm fatigue in clinical practice.** The Joint Commission's Sentinel Event Alert on medical device alarm safety describes the danger of excessive alarms and fatigue.
That is documented in a healthcare setting, not evidence of a specific freight loss rate.
[INFERENCE] Competing demands on human attention make alert volume and prioritization relevant to control towers too.
A safety-critical alarm may require action even when a commercial expected-value calculation is unfavorable.

**Precision-recall under imbalance.** Saito and Rehmsmeier analyze precision-recall plots for imbalanced classification.
Their finding is about statistical interpretation, not the financial value of a logistics intervention.
Use it to explain why base rate and positive predictive value deserve explicit reporting.
Neither an area under a curve nor a supplier's dashboard establishes causal value on its own.

A responsible business case distinguishes public evidence, local measured baseline, and forecast benefits.
Name the comparator and say whether figures are observed, simulated, or assumed.
Require the operator's time and the financial impact to be attributable to the same change.
Where no public logistics case discloses the required confusion matrix, do not invent one.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Alert flood | Queue grows each shift | Threshold ignores capacity | Valuable cases expire | Cap and rank reviews, then audit overflow |
| High-precision tunnel vision | Very few alerts | Cutoff excludes many real cases | Missed disruptions | Report recall and severity of misses |
| Untreated-label illusion | Actions precede labels | Intervention prevents outcomes | Good alerts look false | Use an appropriate causal evaluation |
| Selective labels | Unreviewed cases lack outcomes | Investigation determines observability | Biased precision and recall | Randomly audit both selected and unselected cases |
| Leakage | Backtest uses later scans | Future data enters forecast | Inflated performance | Replay as-of ingestion timestamps |
| Duplicate alerts | Multiple rows per shipment | Event retries create extra cases | Wasted attention | Deduplicate at decision grain |
| Stale threshold | Queue grows after lane mix shifts | Prevalence and calibration change | Budget breach | Monitor cohort mix and recalibrate |
| False capacity | Reviews exceed usable minutes | Breaks and other duties omitted | Work is delayed | Measure net available shift time |
| Mistaken denominator | Precision reported as FP rate | Different populations are mixed | Wrong diagnosis | Publish the full matrix |
| Double-counted labor | Benefit excludes no-alert labor inconsistently | Same cost charged twice | Misleading business case | Reconcile all incremental costs |
| Late alert | High model recall but expired cases | No lead-time requirement | No feasible intervention | Measure timely review and action |
| Wrong objective | Highest risk always wins | Loss and effectiveness differ | Lower preventable value | Rank expected incremental value |
| Hidden safety obligation | Mandatory case enters discretionary queue | Profit objective overrides policy | Regulatory or physical harm | Separate mandatory escalation first |
| Label immaturity | Open shipments marked negative | Outcome window incomplete | Apparent precision decline | Delay scoring until maturity |
| Gaming | Staff close easy cases first | Metric rewards counts only | Severe cases remain unresolved | Audit case mix and outcomes |
| Shared-resource interaction | Multiple cases request one slot | Case benefits are not additive | Infeasible plans | Optimize joint actions and constraints |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Cohort is complete | Recall needs all eligible positives | Missing carriers or shipments | Reconcile against independent manifests |
| Label is mature | Outcome must be observable | Open or disputed shipments | Track maturity lag and censoring |
| Positive definition is stable | Rates need comparable outcomes | Promise dates are revised | Preserve the original commitment |
| Risk is calibrated | Probability cutoff needs meaningful p | Lane or season shifts | Calibration plots by cohort and time |
| Benefit is preventable | Gross loss is not action benefit | Review cannot change the outcome | Compare action and comparator in a pilot |
| Review cost is incremental | Net value requires a common baseline | Existing staff already perform the work | Map actual workflow and time |
| Equal review time in worked case | Case-slot capacity is interpretable | Complex cases take longer | Measure duration distribution |
| Actions are independent | Case values can be summed | Shared dock or vehicle capacity | Model interactions and joint constraints |
| Reviews are timely | Preventive action must remain possible | Late queue completion | Compare completion and deadline timestamps |
| Outcome is untreated | Matrix describes risk before intervention | Successful action changes outcome | Use experimental or defensible causal design |
| Unit costs apply equally | Band cost model is simple | Different customers have different losses | Segment sensitivity analysis |
| Single-day horizon suffices | No downstream cost is omitted | Future backlog or cascading delay | Expand the model and report horizon |

A small number of observed positives can produce a volatile precision estimate.
Report uncertainty intervals or cohort counts rather than presenting a percentage alone.
Monitoring by lane or partner can reveal heterogeneity but also creates small-sample uncertainty.
Never infer individual causality from a before-and-after aggregate alone.
The example's hypothetical labels and full prevention assumption are strong teaching simplifications.

## 12. Exercises

Use USD and the same decision-grain conventions unless a question says otherwise.
State all assumptions and distinguish forecasts, plans, targets, and customer commitments.
### Exercise 1: Confusion matrix

- **Scenario:** A cohort has 200 shipments: 30 positives, 50 alerts, and 18 alerted positives.
- **Deliverable:** Calculate TP, FP, FN, TN, precision, recall, false-positive rate, and miss rate.
- **Acceptance criteria:** All four counts sum to 200 and every denominator is identified.
- **Time:** 12 minutes.
### Exercise 2: Deadline coverage

- **Scenario:** A queue generates 80 alerts; 60 are reviewed, 45 before their decision deadline.
- **Deliverable:** Calculate reviewed share of alerts and timely share of alerts; explain why these differ from recall.
- **Acceptance criteria:** Give both percentages and identify the missing positive-case denominator.
- **Time:** 8 minutes.
### Exercise 3: Base-rate shift

- **Scenario:** A detector has 80% sensitivity and 10% false-positive rate in a cohort of 1,000 shipments with 5% prevalence.
- **Deliverable:** Compute the expected confusion matrix and precision, assuming the stated rates hold.
- **Acceptance criteria:** Show all four expected counts and explain the assumption behind transport to another cohort.
- **Time:** 12 minutes.
### Exercise 4: Threshold

- **Scenario:** A timely review costs USD 12 and prevents an expected USD 400 of loss conditional on an untreated positive.
- **Deliverable:** Compute the break-even probability with no other effects; state two limitations.
- **Acceptance criteria:** Give 3% and identify capacity and calibration as limitations or equivalent justified caveats.
- **Time:** 10 minutes.
### Exercise 5: Capacity tradeoff

- **Scenario:** Two cases have expected gross avoidable loss of USD 70 and USD 60, respectively; review costs are USD 10 and USD 5, and each takes one slot.
- **Deliverable:** Choose one case if only one slot is available and compute the net value.
- **Acceptance criteria:** Choose the first and show USD 60 versus USD 55.
- **Time:** 8 minutes.
### Exercise 6: Unequal duration

- **Scenario:** Case A has net value USD 100 and requires 60 minutes; case B has net value USD 80 and requires 30 minutes; case C has net value USD 75 and requires 30 minutes. Capacity is 60 minutes.
- **Deliverable:** Choose the best feasible set by enumeration and compare it with risk-only ranking.
- **Acceptance criteria:** Choose B plus C with USD 155 and explain that values, not probabilities alone, drive this choice.
- **Time:** 12 minutes.
### Exercise 7: Selective labels

- **Scenario:** Only alerted shipments receive manual outcome adjudication; some are rebooked and their original delivery is never observed.
- **Deliverable:** Design a measurement plan addressing both missing labels and intervention effects.
- **Acceptance criteria:** Specify a sampling frame, audit of nonalerts, maturity, and a causal comparator or state why identification fails.
- **Time:** 20 minutes.
### Exercise 8: Operating policy

- **Scenario:** A new warehouse offers one analyst, variable deadlines, and non-discretionary safety escalations.
- **Deliverable:** Draft an operating policy and a pilot scorecard.
- **Acceptance criteria:** Name owners, capacity reservation, ranking, expiry, evidence, safeguards, and evaluation comparator.
- **Time:** 25 minutes.

### Exercise 9: Cost of an imperfect remedy

- **Scenario:** Review costs USD 6 for every shipment.
- On an untreated positive, review enables an action with 25% probability.
- The action avoids USD 240 of loss and costs USD 40 when used.
- Assume the action is never used on a noncase and there are no other effects.
- **Deliverable:** Calculate conditional net benefit and break-even untreated probability.
- **Acceptance criteria:** Show the USD 50 conditional benefit and the probability threshold.
- Explain whether a case with a 15% calibrated probability is positive-value before capacity constraints.
- **Time:** 15 minutes.

### Exercise 10: Weighted audit and interpretation

- **Scenario:** A cohort has 200 alerts and 800 nonalerts.
- All alerts are audited and 40 have positive observed outcomes.
- A simple random sample of 80 nonalerts is audited; 12 have positive observed outcomes.
- All labels are mature and measured consistently for this exercise.
- **Deliverable:** Estimate the number of nonalert positives and the observed-outcome recall.
- **Acceptance criteria:** Show the sampling weight and estimated counts.
- Explain the uncertainty and why observed outcomes need not identify untreated risk.
- **Time:** 15 minutes.

In both exercises, state the comparator before interpreting the number.
Treat all monetary quantities as USD per shipment unless the calculation states a cohort total.
Neither a positive expected value nor an estimated recall is sufficient to approve a deployment.
The proposal also needs feasible capacity and required safeguards.

## 13. Self-check answers

### Exercise 1

TP = 18 shipments.
FP = 50 − 18 = 32 shipments.
FN = 30 − 18 = 12 shipments.
TN = 200 − 18 − 32 − 12 = 138 shipments.
Precision = 18 / 50 = 36%.
Recall = 18 / 30 = 60%.
False-positive rate = 32 / 170 = 18.82%.
Miss rate = 12 / 30 = 40%.
The 32 false alerts are 64% of alerts but only 18.82% of negatives.
Those two denominators answer different questions.

### Exercise 2

Reviewed share = 60 / 80 = 75% of alerts.
Timely share = 45 / 80 = 56.25% of alerts.
The difference is 15 reviews completed after the deadline.
Neither figure identifies the number of all positive cases, so neither is recall.
A timely review is not necessarily an effective action.

### Exercise 3

Five percent of 1,000 shipments is 50 positives; 950 are negatives.
TP = 0.8 × 50 = 40; FN = 10.
FP = 0.1 × 950 = 95; TN = 855.
Precision = 40 / 135 = 29.63%.
Even with 80% recall, most alerts are false at this prevalence and false-positive rate.
The transport calculation assumes both rates remain unchanged in the new cohort.

### Exercise 4

The review is worthwhile when p × USD 400 exceeds USD 12.
The break-even probability is USD 12 / USD 400 = 0.03, or 3%.
A staffing cap can require a higher effective cutoff or value-based ranking.
Uncalibrated scores cannot be compared directly to this probability.
USD 400 must be preventable expected loss, not total loss if action can fail.

### Exercise 5

Case A has USD 70 − USD 10 = USD 60 of net expected value.
Case B has USD 60 − USD 5 = USD 55 of net expected value.
One equal-size slot goes to A under the stated assumptions.
The choice could change if the cases have different deadlines, mandatory rights, or dependencies.

### Exercise 6

Feasible nonempty sets are A, B, C, and B plus C; A cannot fit with another case.
Their values are USD 100, USD 80, USD 75, and USD 155 respectively.
B plus C is the best feasible set, consuming 60 minutes.
A raw risk ranking cannot determine that choice without severity, effects, costs, and duration.

### Exercise 7 rubric

A strong design enumerates all eligible shipments before alerting and samples nonalerts as well as alerts.
It records inclusion probabilities and obtains labels using a common mature outcome definition.
It distinguishes missing observed labels from unobservable untreated counterfactuals.
An authorized randomized policy comparison can estimate average incremental outcomes under suitable design.
If no defensible comparator or outcome source is available, the design states that the causal effect is unidentified.
It does not claim that auditing alone reveals each treated shipment's untreated outcome.

### Exercise 8 rubric

A strong policy identifies the queue owner and the person authorized to change an appointment.
It reserves capacity for required escalations and defines a feasible priority and deadline rule.
It logs generated, reviewed, expired, acted-on, and mature-outcome populations separately.
It specifies a baseline, pilot allocation, costs, timing, and decision criteria.
It includes safety, privacy, and drift review without claiming universal legal applicability.
A weak policy reports only model accuracy or treats an alert as an executed action.

### Exercise 9

Conditional on an untreated positive, an enabled action has a net benefit of USD 240 − USD 40 = USD 200.
It occurs with probability 0.25, giving expected conditional benefit of USD 50.
For untreated probability p, expected net value of review is p × USD 50 − USD 6.
Break-even occurs at p = USD 6 / USD 50 = 0.12, or 12%.
At p = 15%, expected net value is 0.15 × USD 50 − USD 6 = USD 1.50 per review.
This value is positive before considering competition for the same analyst's time.
It assumes the USD 40 action cost occurs only in the stated enabled positive cases.
It also assumes no cost or benefit from action on a noncase.
If another case offers higher net value for the same scarce time, capacity may go there instead.

### Exercise 10

The nonalert audit probability is 80 / 800 = 0.10.
Each audited nonalert therefore has weight 1 / 0.10 = 10.
The estimated number of positive nonalerts is 12 × 10 = 120 shipments.
All 200 alerts are audited, so the observed positive-alert count is 40 shipments.
Estimated observed-outcome recall is 40 / (40 + 120) = 25%.
Estimated observed-outcome precision among alerts is 40 / 200 = 20%.
The recall denominator is an estimate, not a census of all positive cases.
The sample's 12 observed nonalert positives are subject to sampling variation.
An uncertainty interval should account for the sampling design.
Nonresponse, inconsistent adjudication, or a nonrandom audit would weaken the estimate.
If reviewed alerts received effective interventions, their observed outcomes can differ from untreated outcomes.
The weighted audit does not reconstruct each treated shipment's counterfactual outcome.
A policy comparison is needed to estimate incremental impact.

### Connecting the answers to decisions

The first eight exercises distinguish metric calculation from operational and causal design.
Exercise 9 adds a remedy that works only some of the time.
Exercise 10 shows how a known audit sampling rate changes an estimated denominator.
Together, they show why neither a score nor an alert count alone determines economic value.
The operator must still check actionability, deadlines, capacity, and the required escalation process.

## 14. Interview questions
### Question 1: Why can 90% precision still be a poor system?

- **Strong answer:** Explain recall, severity of missed cases, timeliness, and capacity.
- **Weak answer:** Equates a high precision figure with complete coverage.
### Question 2: How do precision and false-positive rate differ?

- **Strong answer:** Names alerts versus all actual negatives as denominators.
- **Weak answer:** Calls both false alarms divided by alerts.
### Question 3: What happens when prevalence falls?

- **Strong answer:** Discusses declining precision if sensitivity and false-positive rate remain fixed.
- **Weak answer:** Assumes precision is independent of population mix.
### Question 4: How do you choose a threshold?

- **Strong answer:** Uses incremental benefit, cost, calibration, capacity, and mandatory rules.
- **Weak answer:** Chooses 0.5 without a decision rationale.
### Question 5: What is precision at k?

- **Strong answer:** Defines positives in the first k ranked cases and connects k to actual capacity.
- **Weak answer:** Computes it on an unordered alert set.
### Question 6: How can an effective action look like a false alert?

- **Strong answer:** Explains prevented outcomes and the untreated counterfactual.
- **Weak answer:** Labels every post-action non-event a model error.
### Question 7: How would you evaluate an overloaded queue?

- **Strong answer:** Separates generated alerts, actual reviews, deadlines, and expired cases.
- **Weak answer:** Treats all alerts as handled.
### Question 8: What if case durations differ?

- **Strong answer:** Uses minute capacity, expected net value, and possible scheduling or knapsack constraints.
- **Weak answer:** Ranks only by model probability.
### Question 9: How do you correct selective labels?

- **Strong answer:** Audits nonselected cases and distinguishes missing data from causal identification.
- **Weak answer:** Assumes reviewed cases represent all shipments.
### Question 10: When is a cost threshold inappropriate?

- **Strong answer:** Identifies mandatory safety or legal escalation and non-monetizable harms.
- **Weak answer:** Claims the highest estimated financial return always overrides obligations.
## 15. Key terms

The terms below point to their operational definitions in [section 4](#4-definitions-and-vocabulary).

- **Action:** authorized intervention following investigation.
- **Alert:** request for attention, not an intervention.
- **Calibration:** agreement between probabilities and frequencies.
- **Capacity:** usable attention in a defined period.
- **Censoring:** incomplete outcome observation at cutoff.
- **Cohort:** eligible decision opportunities.
- **Commitment:** promise made to a counterparty.
- **Counterfactual:** outcome under an unobserved alternative.
- **Decision grain:** unit at which the rule acts.
- **Decision threshold:** boundary used to select an action.
- **False negative:** positive outcome with no alert.
- **False positive:** alert with a negative outcome.
- **False-positive rate:** false alerts divided by all negatives.
- **Horizon:** period over which the outcome is measured.
- **Lead time:** available time before the decision deadline.
- **Miss rate:** misses divided by all positives.
- **Opportunity cost:** value forgone by a resource use.
- **Plan:** intended sequence of actions.
- **Policy value:** expected outcomes and costs of a policy.
- **Precision:** true cases divided by all alerts.
- **Precision at k:** positive share of the top k cases.
- **Prevalence:** positive share of the eligible cohort.
- **Ranking:** ordering of cases by a score or value.
- **Recall:** alerted cases divided by all positives.
- **Review:** completed investigation by an operator.
- **Selective labels:** outcomes disproportionately observed for selected cases.
- **Shadow price:** marginal value of relaxing a constraint.
- **Target:** desired result, distinct from a forecast.
- **True negative:** negative outcome with no alert.
- **True positive:** positive outcome with an alert.

## 16. Sources

1. GS1, [EPCIS standard](https://www.gs1.org/standards/epcis), official standard and documentation.
   Use for event vocabulary and exchange, not for an economic threshold.
2. NIST, [AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework), January 2023.
   Voluntary risk-management guidance; applicability of law is a separate question.
3. European Union, [Regulation (EU) 2016/679, GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj), official legal text.
   Check applicability and current consolidated provisions with counsel.
4. European Union, [Regulation (EU) 2024/1689, AI Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), official legal text.
   Classification, obligations, and applicable dates depend on the particular system and role.
5. Saito, T. and Rehmsmeier, M. (2015), [The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets](https://doi.org/10.1371/journal.pone.0118432), *PLOS ONE*.
   Peer-reviewed discussion of metrics under class imbalance.
6. Davis, J. and Goadrich, M. (2006), [The Relationship Between Precision-Recall and ROC Curves](https://doi.org/10.1145/1143844.1143874), *Proceedings of ICML*.
   A mathematical treatment of the relationship between curve types.
7. The Joint Commission, [Medical device alarm safety in hospitals](https://www.jointcommission.org/en-us/knowledge-library/newsletters/sentinel-event-alert/issue-50), Sentinel Event Alert 50, 2013.
   Clinical alarm-fatigue context, not a logistics impact estimate.
8. U.S. FDA, [FDA Adverse Event Reporting System (FAERS)](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers), official explanation and limitations.
   The reporting setting illustrates selection limits, not shipment economics.
9. UPS, [ORION](https://about.ups.com/us/en/our-company/innovation-driven/orion.html), company description of route optimization.
   A company source, not independent evidence for this topic's hypothetical results.

## 17. Links

- [Module 19 topic index](_index.md).
- [Module 19 survey and its labs](../../../course/19_control_towers_visibility_events.md).
- [Event time versus processing time](03_event-time-processing-time.md).
- [ETA prediction](04_eta-prediction.md).
- [Delay propagation](05_delay-propagation.md).
- The index also lists planned anomaly detection, exception queue, and decision-rights topics.
- Use the survey's lab material to turn the case into a queue-and-decision exercise.
