---
case_id: "CASE-03-01-005"
case_title: "Triage a shifted delivery-risk model without confusing drift with harm"
case_version: "2.0"
created: "2026-11-07"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "incident"
domain: "production ML monitoring"
primary_pattern: "model-monitoring"
secondary_patterns: ["distribution-shift", "label-delay"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "ML on-call lead"
source_index: "INDEX.md"
review_trigger: "Change in outcome maturity, serving contract, or intervention policy"
---

# Triage a shifted delivery-risk model without confusing drift with harm

Every entity, timestamp, count, threshold, cost, and architectural fact in this case is a constructed teaching input, labelled ILLUSTRATIVE wherever it appears.
No company deployed this system, no operator received this page, and no outcome in Sections 10 and 11 was observed or measured.
The arithmetic is real: each number printed below was recomputed from the stated parameters, and the derivation blocks show the intermediate values so a reader can contradict them.
The monitoring statistics are derived from a stated null distribution; the one empirical check in this file is a seeded simulation of that null, described in Section 6.2 and reproducible from the seed stated there.
Where a claim depends on something that would only exist in a live deployment — the true late-delivery rate of a shifted population, the actual effect of a carrier call — the case says so and does not fill the gap.

## 1. The One-Line Framing

At 09:30 a drift monitor fires on the distance feature of a late-delivery risk model, and the on-call lead must choose between continuing, containing, and rolling back, knowing that the alarm fired twenty-five minutes after the damage had already stopped growing.
The decision is hour-scale incident triage over a bounded proactive-contact queue, not a decision to withhold service from any order.
A good response keeps the contact queue pointed at orders a call can still rescue, separates a broken feature contract from a genuinely different delivery population, and refuses to describe either as "the model got worse" without a matured-label comparison.
The decisive quantity is not the drift statistic; it is the precision of the thirty contacts per hour that the operator can actually place.
Drift measures whether inputs moved. Harm measures whether the ranked queue still concentrates rescuable lateness. The two numbers diverge so far in this case that a monitor built on the first would have missed the entire economically relevant part of the incident.

## 2. Context & Background

A delivery operations group promises customers a delivery window and pays a service credit when an order arrives more than thirty minutes past the promise.
The group cannot rescue every late order, so it runs a proactive-assistance queue: an agent calls the carrier or the customer and attempts to re-plan the delivery.
The queue is small relative to demand. In the constructed slice used throughout this case, 900 orders are accepted per hour and the team can place 30 contacts per hour, each taking six minutes of agent time.
Thirty contacts at six minutes each is 180 agent-minutes per hour, or exactly three full-time agents. That is the binding constraint of the whole case: not model quality, not data volume, not latency.
A gradient-boosted model scores each accepted order once with the probability that it will arrive more than thirty minutes late.
Scores are converted to a priority order, eligibility filters remove orders that cannot be helped, and the top thirty eligible orders per hour receive calls.
The production model at the start of the incident is version v43, promoted eleven days earlier, and the previous version v42 is retained as a rollback artifact.
The distance feature is produced by an upstream team and carries a documented contract: kilometres, non-negative, clipped at 100, refreshed within five minutes of acceptance.
At 09:05 the upstream team enables a new merchant-campaign routing path that emits the distance field in metres for a subset of orders.
The subset is not random. It is the long-haul campaign cohort, which happens to be the cohort with the highest late-delivery risk, so the corruption lands preferentially on the orders the model is trying to catch.
By 09:17 the affected share reaches 22 percent of accepted orders in the slice and stays there.
Two competing explanations are live at 09:30. Either a real campaign changed the delivery geography, or a producer bug changed the units. Both produce a distribution shift in the same feature. Only one of them produces a feature contract violation.
The ML on-call lead owns diagnosis and the containment recommendation. The operations duty manager owns the queue. The data platform team owns the producer and the change that caused it. The incident commander owns external communication and closure.
No one in this case has a mature label for any order accepted after 08:00, because a delivery outcome cannot be known before the delivery happens. That single sentence explains most of the confusion in Sections 6 and 12.

## 3. The Business Question

The question put to the on-call lead is narrow and time-boxed.

> Within the next hour, should the queue keep scoring with v43 on the current feature feed, switch to a contained configuration that routes the suspected-bad cohort to a fallback scorer, or roll back to v42 — and what evidence justifies that choice before any delivery outcome for the affected orders is known?

Three sub-questions have to be answered to make that choice honestly.

1. Is the observed input shift evidence of degraded decisions, or only evidence of changed inputs?
2. If decisions are degraded, by how much per hour, in the currency the business actually spends, and how does that compare with the cost of each candidate action?
3. What monitoring change would have shortened the exposure, given that the existing statistic did fire and the fire was late rather than absent?

Each sub-question has a different failure mode if answered sloppily.
Answering the first with "PSI is above 0.2, so the model is broken" conflates a diagnostic with an outcome.
Answering the second with accuracy or AUC on a holdout conflates an offline discrimination measure with the precision of the thirty contacts actually placed.
Answering the third with "lower the threshold" trades a real reduction in detection latency against a false-page rate that must be computed, not asserted.

The decision has asymmetric errors, and naming them is part of framing the problem.

- Continuing on a broken feed spends agent hours on orders whose scores are close to arbitrary, while the queue is the only channel that can rescue lateness.
- Containing wrongly removes coverage from a cohort that may be exactly the cohort needing help, and the loss is invisible because excluded orders never appear in the queue's own metrics.
- Rolling back to a model that is eleven days older buys a known-small improvement while leaving the data bug in place, so the next order from the corrupted cohort still gets a corrupted score.

The framing that survives all three errors is: separate the data-contract verdict from the model-quality verdict, act on the first immediately, and quantify the second only when labels mature.

## 4. Assumptions & Constraints

The case is only as good as its stated assumptions, so they are listed with their basis and the consequence of being wrong.

| ID | Assumption | Basis | If wrong |
|---|---|---|---|
| A-01 | 900 accepted orders per hour in the slice | constructed slice definition | every rate scales linearly; conclusions about shares do not change |
| A-02 | 30 contacts per hour can be placed | staffing plan for the shift | optimal capacity argument in Section 6.7 moves |
| A-03 | 6 minutes of agent time per contact | time-and-motion sample, treated as fixed | agent-hour cost and capacity both move proportionally |
| A-04 | Base late-delivery rate is 0.09 | prior quarter, treated as stable | reference precision moves; saturation share does not |
| A-05 | Reference precision at capacity is 0.375 | offline replay of v43 at the same cutoff | break-even comparison shifts; ranking conclusions hold |
| A-06 | A successful contact prevents 18 percent of late deliveries | assumed, not measured | value per contact and break-even precision scale linearly |
| A-07 | Mean cost of a late delivery is 180 currency units | service credit plus support cost | break-even precision scales inversely |
| A-08 | Loaded cost per contact is 5.80 | wage plus overhead divided by contacts per hour | break-even precision scales linearly |
| A-09 | The corrupted cohort is 22 percent of orders and 23 percent late | constructed incident shape | saturation share and harm magnitude change |
| A-10 | v42 remains deployable and its artifact is unchanged | release policy retains two versions | rollback option disappears; containment becomes the only lever |
| A-11 | Containment needs 15 minutes of change approval | change-management policy | exposure window lengthens or shortens |
| A-12 | Delivery outcomes are known within 2 hours for 85 percent of orders | fulfilment telemetry, treated as fixed | label-maturity correction in Section 6.5 is mis-specified |

Two of these assumptions do the heavy lifting and deserve separate comment.

A-06 is the weakest link in the whole economic argument. An eighteen-percent rescue rate is plausible for a carrier call that re-plans a route before dispatch, and implausible for a call placed after the van has left. If the true rate is nine percent, every value figure in Section 6.6 halves and the break-even precision doubles to 0.358; the ranking of the three actions does not change, because all three are valued through the same multiplier.
A-09 is the assumption the incident is built on, and it is deliberately benign-looking: a 22-percent share sounds like a minority problem. Section 6.3 shows that anything above 3.33 percent saturates the queue, so 22 percent and 4 percent are economically identical here. That insensitivity is the finding, not an artefact.

Constraints that are not assumptions, because they cannot be varied within the incident:

- The queue cannot expand during the incident. Three agents are rostered; a fourth requires a shift change that takes longer than the incident.
- No ground-truth late flag exists for any order accepted after 08:00 at the time of the decision. Predictions can be compared with predictions; they cannot be compared with outcomes.
- The score is computed once at acceptance. Rescoring requires a versioned replay event, so a "the score will fix itself later" plan is not available.
- Every contact displaces another contact. There is no free capacity to test an alternative ranking in parallel.
- Rolling back changes the model artifact and the feature contract stays broken, because the feature is fetched before scoring.
- Orders that are filtered out for ineligibility never get a score and never appear in queue metrics, so any evaluation conditioned on "scored" quietly drops them.

## 5. Data, Tools & Method

### 5a. Data Specification

The decision-time table is one row per accepted order per scoring event.

| Field | Type | Unit | Available at cutoff | Note |
|---|---|---|---|---|
| order_id | string | - | yes | primary key |
| accepted_at | timestamp, UTC | second | yes | event time, not ingest time |
| region, channel | enum | - | yes | slice keys; one region, one channel |
| promised_window_min | integer | minutes | yes | 120 or 240 in this slice |
| distance_km | float | kilometres | yes | the feature under suspicion |
| merchant_campaign_flag | boolean | - | yes | set at acceptance by the campaign service |
| courier_load_ratio | float | dimensionless | yes | assigned-courier load at acceptance |
| score_v43 | float | probability | yes | model output, one per scoring event |
| queue_slot | integer or null | rank | yes | 1 to 30 for contacted orders |
| contact_started_at | timestamp or null | second | no | written after the contact |
| late_flag_30m | boolean or null | - | no | written at delivery, matures over hours |
| feature_source_version | string | - | yes | producer release identifier |

Grain, keys, and the cutoff are the parts that make this table usable.

- Grain is one accepted order. An order can appear twice only if it is rescored; rescoring writes a new row with a new feature_source_version, and analyses must pin one version per order.
- The cut-off timestamp is accepted_at. Any field written after accepted_at is forbidden in the scoring path, which in this table excludes contact_started_at and late_flag_30m by construction rather than by discipline.
- Population for this incident is one region and one channel, orders accepted between 08:00 and 11:00 UTC. The reference window is the seven preceding days for the same slice, 151,200 orders.
- Units are explicit per column because the incident is a unit bug. A distance column with no unit in its name is how this happens; the fix in Section 8 includes renaming the field.
- The event-time versus processing-time distinction matters: a backfill can rewrite distance_km for rows already scored, which changes history without changing any decision that was made. Section 5d treats that as a first-class hazard.

Not available at decision time, and therefore not usable in any justification for the 09:30 call:

- Whether a contacted order would have been late without the contact. The counterfactual does not exist; Section 6.6 discusses what a holdout would have to look like to produce it.
- The late rate of the affected cohort. It is knowable in principle and unknowable at 09:30.
- The volume of the campaign cohort after the campaign ends, which determines whether this is a transient or a permanent population change.
- Whether customers in the long-haul cohort respond differently to a call, which would change the rescue rate and therefore the value of a slot.

### 5b. Tools & Stack

The stack is stated as an assumption, because the incident is about reasoning rather than about vendors.

| Layer | Assumed tool class | Role in the incident |
|---|---|---|
| Event log | columnar warehouse table | source of accepted orders and scores |
| Feature store | offline store plus low-latency online reader | serves distance_km at acceptance |
| Producer | streaming job, schema-registry validated | emits distance_km, changed at 09:05 |
| Scoring service | stateless model server, two artifacts resident | serves v43 and v42 side by side |
| Drift job | hourly and 15-minute windows, per-feature PSI | produced the alert |
| Quality job | matured-label join and calibration report | the only source of truth about precision |
| Incident tooling | paging, runbook, change-approval queue | governs containment latency |

Three tool facts matter more than the vendor names.

1. The drift job compares against a stored reference distribution snapshot, not against a live recomputation. If the snapshot is regenerated by a backfill, the reference silently changes and every subsequent PSI value is measured against a moving baseline.
2. The scoring service keeps both artifacts resident, so rollback is a routing change measured in seconds, not a redeploy. That asymmetry — fast rollback, slow containment — shapes the action ranking in Section 6.6.
3. The quality dashboard joins labels at whatever maturity they happen to have, without a completeness gate, which is why its precision number sits at 0.238 while the true value is 0.375. Section 6.5 derives that gap rather than asserting it.

### 5c. Method Summary

The walkthrough in Section 6 is built from seven steps, in this order, because the order is what prevents the common error of starting from the model.

1. Decompose the drift statistic into per-bin contributions to see which bin carries it, and check whether the bin that carries it is the bin the corruption would produce.
2. Calibrate the drift statistic against its null distribution, so that "0.15 is large" becomes "0.15 is this many null standard deviations at this sample size".
3. Convert a corrupted input distribution into a decision-relevant quantity: the share of queue slots that the corrupted cohort consumes.
4. Convert that slot share into the realised precision of the queue, then into value per hour, using the break-even precision as the anchor.
5. Correct the observed precision for label maturity, because the dashboard number is computed on a skewed subset of outcomes.
6. Compare the three candidate actions over the same clock interval, so that fast-but-worse and slow-but-better are priced on equal terms.
7. State which parts of the conclusion are arithmetic and which are assumptions, and size the evaluation that would replace the assumptions with measurements.

### 5d. Provenance, Quality, Exclusions, Leakage

Provenance for this case is deliberately thin, and saying so is part of the method.

- The incident shape, the parameter values, and the timeline are constructed. There is no source system, no ticket number, and no operator to interview.
- The PSI arithmetic and the null-distribution calibration are derived from first principles and checked by simulation; those results would hold for any system with the same bin structure and sample sizes.
- The label-maturity correction is derived from a stated exponential maturity curve, not fitted to data.
- The value model is a three-parameter construct: rescue rate, cost of lateness, cost per contact. It is transparent rather than accurate.

Data-quality checks that the incident response actually used, each with the defect it can expose:

| Check | What it catches | What it cannot catch |
|---|---|---|
| Unit-name and range assertion on distance_km | metres in a kilometre field | a metres value that happens to sit inside the kilometre range |
| Producer version pinned per row | silent mid-window producer change | a change that keeps the version string |
| Null-count and default-value rate | missing values defaulted to zero | a wrong-but-present value |
| Join cardinality per order | fan-out duplicating orders | a missing join that drops orders |
| Reference-window immutability hash | backfills moving the baseline | a legitimate population change |
| Score distribution versus artifact version | artifact mismatch in serving | a model that is wrong for its population |

Exclusions, and why each is not a silent deletion:

- Orders accepted after 09:05 from the corrupted cohort are excluded from any retraining set until the producer is fixed. Excluding them is correct for training and dangerous for evaluation, because the excluded cohort is exactly the harmed one. Section 9 keeps them in the evaluation population and out of the training population, and states both.
- Orders with a null contact outcome are excluded from contact-effectiveness calculations, not from precision. A contact that was never attempted is a policy event, not a model event.
- The eleven-day-old v42 holdout period is excluded from v43 comparisons because the population changed between them; the campaign is the reason and the reason is recorded.

Leakage risks specific to this incident, all of which are realistic rather than hypothetical:

- Joining contact_started_at or any post-contact field into a training set teaches the model which orders operations chose to call. That is a policy label wearing a feature's clothes.
- Conditioning evaluation on "orders that received a score" drops ineligible orders, and ineligibility correlates with the outcome; the drop is small in volume and large in bias.
- Backfilling corrected distance values into the history used to build the reference distribution moves the baseline toward the observed, which mechanically suppresses future PSI. The suppression looks like recovery.
- Using a post-incident snapshot of the campaign flag as an input would leak the routing decision into a feature that was supposed to describe the order.

### 5e. Reproducibility & Method-Selection Ledger

Each methodological choice is recorded with the alternative that was rejected and the reason, so a reviewer can see the decision rather than only the outcome.

| Choice | Alternative rejected | Reason |
|---|---|---|
| Three fixed bins for distance | decile bins recomputed on live data | recomputed edges make the statistic incomparable across windows and hide bin migration |
| PSI on the input | KS on continuous distance | KS is fine for a continuous feature but loses the tail concentration that matters here |
| Precision at capacity | AUC of the scoring model | AUC is invariant to the cutoff the operator cannot move |
| Per-slot value model | model-accuracy delta | accuracy is not denominated in agent hours |
| Exponential maturity correction | wait six hours before any comparison | waiting is honest but the incident is decided in ninety-five minutes |
| Two-window alert rule | single-window alert | single-window has a defensible false-page rate, but the runbook in force required two |
| Containment over rollback | rollback to v42 | v42 also consumes the corrupted feature, so rollback does not remove the cause |

Reproducibility requirements for anyone re-deriving the numbers:

- Parameters are the twelve assumptions in Section 4 plus the four constants in Section 6.1. Nothing else enters.
- All logarithms are natural. PSI with base-10 logs is a different number; reporting a threshold without the base is a defect.
- All rates are per hour unless a per-day or per-year figure is labelled as annualised.
- All shares are fractions of a stated denominator, and the denominator is named in the same sentence.
- Random numbers appear once, in the null simulation of Section 6.2, with seed 20261110 and 3,000 trials per sample size.

