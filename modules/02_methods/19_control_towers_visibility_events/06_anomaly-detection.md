# Anomaly detection in operational streams

> Module 19 — Control Towers, Visibility, and Event Intelligence  
> Topic 06 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Topics 02–05; basic descriptive statistics  
> Version 2026.1

## 1. Why this topic exists

A tower operator cannot inspect every shipment update, inventory movement, and sensor reading.
The operational question is not simply whether a measurement is unusual.
It is whether an unusual observation is sufficiently credible, consequential, and actionable to deserve scarce attention now.

Consider a cross-dock manager monitoring inbound dwell.
A trailer that has been waiting 180 minutes may be ordinary at a congested port but exceptional at a small regional transfer point.
The same delay can be harmless for replenishment with two days of cover and critical for a line-side part required in one hour.
A global threshold can therefore both miss costly cases and flood the team with alerts.

An anomaly detector compresses many observations into a smaller set of candidates for investigation.
A decision policy then determines whether a person should investigate, whether an automated response is authorized, and when to escalate.
The detector does not establish causation, prove that an error occurred, or guarantee that intervention will help.
It is one component in a larger event-to-decision system.

The business owner needs a decision before choosing an algorithm:
- Which unit of work can a person affect: an event, shipment, order, lane, or site?
- What action can still change the outcome?
- How long does that action take?
- What does a miss cost, and what does an investigation cost?
- How many investigations can the shift actually complete?
- Who decides that the evidence is sufficient to act?

Bad detection design has costs beyond compute.
Repeated alerts consume the same limited operator capacity needed for genuine disruptions.
Alerts based on incomplete data can trigger unnecessary expediting, partner disputes, and customer promises that cannot be kept.
Suppressing an alert because a value looks statistically familiar can also miss a severe, recurring service failure.
A recurring failure can be normal in the data and unacceptable to the business.

This lesson builds statistical, distance-based, and model-based detectors around a concrete dwell-time decision.
It separates an unusualness score from the threshold and from an operational action.
It also shows how to evaluate alerts when outcomes are delayed, labels are incomplete, and the underlying operation changes.

> **Note:** The numerical cases in this lesson are instructional and fictional unless explicitly identified as public evidence.
> Currency amounts are illustrative US dollars (USD), not measured carrier costs.
> References to standards and law are educational, not legal advice.

## 2. Learning outcomes

After completing this topic, you can:

1. Define an alerting decision in terms of an entity, decision time, action window, owner, and loss.
   Exercise 1 tests this definition against a live inbound operation.
2. Compute and interpret a robust statistical score, including its behavior when scale is zero.
   Exercise 2 requires both the calculation and the fallback decision.
3. Compare a distance-based detector with a statistical rule and explain feature scaling and peer selection.
   Exercise 3 asks for a nearest-neighbor calculation.
4. Describe a model-based residual or isolation score without treating it as a probability of disruption.
   Exercise 4 asks you to critique a model claim.
5. Select an operating threshold using a temporal validation set, measured costs, and shift capacity.
   Exercise 5 requires a threshold recommendation and a capacity check.
6. Calculate alert precision, recall, workload, and lead time while accounting for delayed or missing labels.
   Exercises 5 and 6 test the arithmetic and label interpretation.
7. Design a drift response that distinguishes data-pipeline breakage from a genuine operational change.
   Exercise 7 asks for a controlled response to a regime change.
8. Specify safe deployment, ownership, monitoring, and rollback for a human-in-the-loop detector.
   Exercise 8 asks for a deployment decision record.

The goal is not to memorize one universally best detector.
The goal is to make a defensible choice for a named decision under stated evidence and capacity constraints.

## 3. The core mental model

```text
Physical shipment and partner activity
            |
            v
Captured events + status snapshots ----> freshness / quality checks
            |                                  |
            |                                  +--> data-quality incident
            v
As-of features and comparable peer group
            |
            v
Detector: unusualness score, not a verdict
            |
            v
Threshold + impact + time-to-act + queue capacity
            |
            v
Case: owner, evidence, action rights, expiry
            |
            v
Intervention and observed outcome
            |
            v
Delayed labels, unbiased audit, monitoring, recalibration
            +-------------------------------------> detector and policy
```

There are four separable judgments.
First, a data-quality gate asks whether inputs are fit for the intended decision.
Second, a detector asks how unusual the observed pattern is relative to a reference.
Third, an operational policy asks whether to create or update a case.
Fourth, an authorized person or system chooses an intervention.
Conflating these judgments makes a score appear more certain and more useful than it is.

A statistical detector compares a measurement with a distribution or control limit.
A distance detector compares a vector with other points in a suitably scaled feature space.
A model-based detector compares observations with a learned representation, predicted value, or learned partitioning of normal behavior.
The same shipment can receive different scores under each method because each defines normality differently.

A business rule can operate independently of unusualness.
For instance, a shipment within 45 minutes of a contractual cutoff might require review even if many similar shipments are also late.
Conversely, a statistically unusual early arrival need not prompt action if there is no cost or useful intervention.
The queue can combine a hard safety rule, an anomaly score, estimated impact, and explicit capacity limits without pretending they are interchangeable.

The reference population is part of the model.
A detector trained on all lanes may treat a slow international lane as anomalous every day.
A detector trained separately for every tiny lane may have too little history to estimate a stable baseline.
The business must balance comparability against sample size and revisit that trade-off after network changes.

Time is also part of the model.
A shipment's current dwell is known only after it has arrived and before it departs.
Completed dwell is unavailable at that decision time.
Late-arriving events can alter a reconstruction of the past, but they must not silently rewrite what the live system knew then.
The distinction is essential for honest backtesting.

Finally, a detector changes the system it observes.
An investigated shipment may arrive on time precisely because someone intervened.
Calling that case a false positive merely because the bad outcome did not occur can punish useful alerts.
Log the intervention and evaluate both outcomes and the process that generated them.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Entity | The object scored, such as one shipment-leg visit. | Treating each repeated scan as an independent shipment. |
| Event | A recorded occurrence with a subject and time. | Assuming that record creation proves physical occurrence at that moment. |
| Event time | Claimed time of physical or business occurrence, in a named time zone. | Equating it with receipt time. |
| Processing time | Time the platform receives or processes the event. | Using a later correction as if it had been available earlier. |
| Decision time | Instant at which the alert policy could act. | Calculating features with information from after that instant. |
| As-of feature | Value computable using records available by decision time. | Reconstructing a perfect history with later-arriving records. |
| Dwell | Time between arrival and departure at a location, usually minutes. | Treating incomplete current dwell as completed dwell. |
| Exposure | Eligible entity-time or entity count used as a denominator. | Comparing raw alert counts across differently sized sites. |
| Cohort | A comparable group selected by features such as lane and facility. | Assuming every lane has enough independent history. |
| Baseline | Reference data or distribution defining typical behavior. | Calling a baseline automatically acceptable performance. |
| Contamination | Unusual cases or errors present in a training reference. | Assuming an unlabeled training set is entirely normal. |
| Point anomaly | One observation unusual relative to its reference. | Assuming one extreme value explains an operational incident. |
| Contextual anomaly | Observation unusual in its time, route, site, or other context. | Using a single global limit despite known context. |
| Collective anomaly | A sequence or group unusual even if each element is ordinary. | Ignoring an abnormal sequence of individually plausible scans. |
| Median | Middle ordered observation, or mean of the middle pair. | Treating it as a mean. |
| Median absolute deviation (MAD) | Median of absolute distances from the median, in the original unit. | Using a zero MAD as a denominator. |
| Robust z-score | Difference from median divided by a robust scale estimate. | Calling its magnitude a calibrated disruption probability. |
| Quantile | Value at or below which a specified fraction of a reference lies, subject to an estimator. | Assuming a sample quantile gives a stable future false-alert rate. |
| Distance | Dissimilarity between feature vectors under a specified metric. | Mixing kilometers and minutes without scaling. |
| Nearest neighbor | Reference observation with the smallest chosen distance. | Including the point itself in its own evaluation reference. |
| Residual | Observed minus predicted value in the target's unit. | Using an actual outcome unavailable at the decision time. |
| Isolation forest | Ensemble that scores observations using random partition path lengths. | Treating its score as a calibrated probability or explanation. |
| Threshold | Score or rule boundary at which an alert policy takes a branch. | Treating a default algorithm parameter as a business optimum. |
| Alert | Notification that a condition was met at a decision time. | Counting repeated updates as separate independent incidents. |
| Case | Managed unit of investigation that can aggregate related alerts. | Confusing a case with an observation or physical event. |
| Deduplication | Linking repeated evidence about the same case within a defined horizon. | Suppressing genuinely new, independent disruptions. |
| Cooldown | Defined interval during which repeat notifications are suppressed or consolidated. | Assuming silence means the underlying risk disappeared. |
| True positive (TP) | Alerted eligible case meeting the defined positive outcome label. | Treating all investigated, successfully rescued cases as false positives. |
| False positive (FP) | Alerted eligible case with a mature negative label under the evaluation definition. | Assuming an unobserved outcome is negative. |
| False negative (FN) | Eligible positive case not alerted in the specified window. | Counting only reviewed shipments and ignoring unalerted cases. |
| Precision | TP divided by TP plus FP, among mature labeled alerts. | Calling it recall or estimating it from a biased review sample. |
| Recall | TP divided by TP plus FN, among eligible positives. | Reporting it when unalerted cases have no labels. |
| Lead time | Time from alert to outcome or last useful action, in minutes. | Crediting alerts that occur after an action is no longer possible. |
| Capacity | Available investigations per shift, adjusted for effort and coverage. | Equating theoretical alert rate with cases actually completed. |
| Drift | Change in input, relationship, score, label, or operation over time. | Automatically retraining without diagnosing the cause. |
| Label maturity | Time by which an outcome can be fairly observed and adjudicated. | Scoring unresolved shipments as negatives. |
| Censoring | Incomplete observation of a time-to-event or outcome. | Treating an open shipment's elapsed time as its final duration. |
| Data leakage | Use of unavailable or outcome-derived information in evaluation or training. | Calling a random split of correlated shipment events a realistic replay. |
| Calibration | Agreement between predicted probabilities and observed frequencies. | Assuming an anomaly score is a probability. |
| Watermark | A processing convention for expected event-time completeness. | Assuming a watermark proves no later event can arrive. |
| Commitment | A promise or contractual obligation to a customer or partner. | Treating a model forecast as an agreed promise. |
| Target | Desired performance level, not necessarily a commitment. | Treating a target as a prediction. |
| Plan | Intended operational arrangement, such as a scheduled appointment. | Treating the plan as a guarantee or observed event. |

## 5. The mechanics, step by step

### 5.1 Define the decision before the score

1. Name the decision and its owner.
   Input: a business process and a person or authorized system able to act.
   Output: a decision statement such as “review inbound trailers before the cross-dock cutoff.”
   State what the owner can actually do: call a carrier, reserve a dock, request a revised appointment, or escalate.
   Do not prescribe an expedite unless that owner has the right and budget to authorize it.

2. Specify the eligible unit and decision horizon.
   A useful unit may be a shipment-leg visit at a facility, not a scan or an entire order.
   Define when a unit becomes eligible and when it exits.
   Document whether one alert per visit, one case per incident, or multiple escalations are allowed.
   This definition controls both workload and metric denominators.

3. Separate prediction, plan, target, and commitment.
   A forecast estimates what might happen, with uncertainty.
   A plan describes what operators intend to do.
   A target describes desired performance.
   A commitment is an externally or internally agreed obligation.
   They may use the same clock but are not the same number.

4. State the positive outcome and action window.
   For a dwell alert, the outcome could be “departure after the receiving cutoff,” measured against the cutoff known at the decision time.
   Define whether a case counts if the alert arrives 5 minutes before cutoff when dispatch requires 30 minutes.
   A useful definition is a positive outcome with at least the minimum actionable lead time.
   Retain a separate measure for late-but-correct alerts rather than silently discarding them.

5. Record costs and constraints in explicit units.
   Estimate investigation minutes per case and shift capacity in cases or minutes.
   Distinguish the cost of an unnecessary review from the loss due to an unmitigated miss.
   If an intervention has its own cost and success probability, do not assign the full avoidable loss to every alert.
   Use ranges when stakeholders cannot defend a point estimate.

### 5.2 Build a time-correct observation

6. Define event semantics before computing features.
   Identify the arrival, departure, appointment, and correction events.
   Specify the source of truth, location identifier, event time, receipt time, and revision semantics for each.
   Decide how to handle duplicate identifiers, conflicting timestamps, and partner-specific status codes.
   A null departure can mean still present, a missing message, or a canceled movement.
   These situations should not be silently treated as equivalent.

7. Assemble an as-of snapshot.
   At decision time t, include only records received by t.
   Use the latest valid version available by t rather than the final corrected version.
   Current dwell is t minus the best available arrival time, after validating the time zone and chronology.
   Do not use final dwell, final departure time, or a later carrier explanation as a live feature.
   Preserve the original score and snapshot identifiers for replay.

8. Handle latency and incomplete observations explicitly.
   Compare receipt time with event time by partner, site, and event type.
   If arrival is stale or missing, produce a data-quality case or abstain rather than score a fabricated zero.
   A watermark can trigger a provisional calculation; it is not proof that a missing event never happened.
   Define how a correction updates a case, including when to retract or annotate a prior alert.

9. Define a comparable reference population.
   Candidate context includes facility, direction, shipment type, time of day, weekday, and appointment class.
   Restrict the reference to historical information available before t.
   Exclude canceled or invalid visits according to documented business rules.
   If context becomes too fine, use a broader parent cohort or abstain until the minimum sample criterion is met.
   Record which fallback was used; scores from different baselines may not be comparable.

10. Separate live eligibility from completed reference observations.
    A live trailer has elapsed dwell, while historical completed trailers have final dwell.
    Comparing these directly answers “has elapsed dwell already exceeded a typical completed duration?”
    It does not estimate the probability that the trailer will eventually miss cutoff.
    For a risk estimate, build a time-to-event or classification model that handles open visits and is validated for the decision horizon.
    Alternatively compare elapsed dwell against appropriate at-risk snapshots at the same age and decision context.

### 5.3 Construct statistical detectors

11. Start with a transparent baseline.
    A high quantile of completed dwell can be a pragmatic candidate threshold for a stable cohort.
    A median and MAD describe center and dispersion more robustly than a mean and standard deviation under extreme values.
    For observations x, let m be the median and MAD be median(|x − m|).
    An often-used normal-consistent scale estimate is 1.4826 × MAD.
    The score z = (x − m) / (1.4826 × MAD) is dimensionless.
    The constant gives normal-distribution consistency under suitable assumptions; it does not make skewed dwell normally distributed.

12. Choose direction deliberately.
    A one-sided high-dwell score targets long waits.
    An absolute score would also flag unusually short waits, which could be useful for missing departure or process-bypass checks but creates a different queue.
    State the comparison operator and equality behavior at the threshold.
    For example, “alert when z ≥ 2.0” includes a score exactly equal to 2.0.

13. Define zero-scale and small-sample behavior.
    If MAD is zero, division is undefined even when some values differ from the median.
    Do not add an arbitrary tiny constant and call the resulting huge score meaningful.
    Use a pre-approved broader cohort, another justified scale estimate, a fixed business rule, or abstention.
    Label the method actually used on each case.
    A sparse cohort can make a sample quantile unstable even when MAD is nonzero.

14. Consider distribution shape and seasonal context.
    Dwell is usually nonnegative and may have a long right tail.
    A fixed symmetric z threshold can be poorly matched to it.
    Empirical quantiles, transformations, or separate context-specific limits may be better, subject to validation.
    A holiday peak or new operating shift may need its own comparison group if sufficient data exist.
    Do not build so many small segments that random noise becomes apparent precision.

15. Use control charts only with their intended logic.
    A control limit concerns stability of a process under a stated sampling and dependence model.
    It is not automatically a customer service specification or an optimal alert threshold.
    A sustained shift can be detected by sequential summaries such as an exponentially weighted moving average, but its limits and run rules need validation under actual autocorrelation.
    A process can be statistically stable while routinely missing a business target.

### 5.4 Construct distance detectors

16. Choose an interpretable feature vector available at t.
    For example, use current dwell in minutes, remaining slack to cutoff in minutes, and the count of missing required scans.
    Do not include a post-departure field or a feature that encodes the realized label.
    Consider whether a negative slack value deserves a deterministic overdue rule rather than an anomaly score.

17. Fit scaling on the historical reference only.
    Without scaling, a feature measured in hundreds of minutes can dominate one measured in small counts.
    Robust centering and scaling can limit the effect of extremes, but zero scales need explicit fallback.
    Missing values require a documented imputation or abstention policy.
    An imputed value should not conceal a data-quality issue.
    Store the fitted scaler with the detector so scoring uses the same transformation.

18. Choose a distance and neighborhood.
    Euclidean distance treats standardized axes as orthogonal and is sensitive to correlated features and outliers.
    Other metrics can represent categorical similarity or correlation, but require their own assumptions and sufficient data.
    A k-nearest-neighbor score might be the distance to the kth nearest reference point.
    Large distance indicates isolation under that metric, not necessarily high business impact.
    During validation, prevent an observation or duplicate of the same case from becoming its own nearest neighbor.

19. Check whether the geometry matches operations.
    A rare but legitimate temperature-controlled shipment should be compared with appropriate peers, not a majority of ambient freight.
    A newly opened lane may be distant because it has no historical peers.
    Adding many weak or correlated features can make distances less discriminating.
    Inspect nearest neighbors and reason codes with operators before using the score in a queue.

### 5.5 Construct model-based detectors

20. Decide whether the model predicts a normal value or learns a representation.
    A residual detector predicts an expected value from known context and flags a large deviation.
    Its residual must use an observation available at the scoring moment.
    For example, a forecast of cumulative scan count at the current hour can be compared with the count observed by that hour.
    A model predicting final dwell cannot yield a realized final-dwell residual before departure.
    Use a model of remaining dwell or probability of missing cutoff if that is the actual decision, and evaluate it as such.

21. Consider an isolation forest as an unsupervised alternative.
    It builds random partitions; observations isolated by relatively short paths tend to be scored as unusual.
    It can capture nonlinear combinations without a labeled disruption for every training row.
    Training data, feature representation, missingness treatment, and hyperparameters still determine what is unusual.
    In common implementations the sign and offset of scoring methods differ: document whether higher or lower means more anomalous.
    A contamination setting used to produce a cutoff is not an estimate of the true disruption rate.

22. Treat reconstruction models with particular caution.
    An autoencoder may flag a large reconstruction error, but a flexible model can also reconstruct unusual cases well.
    Aggregate reconstruction error depends on scaling and which features are included.
    There is no guarantee that a large reconstruction error identifies an actionable operational problem.
    Use the simplest validated method that supports the decision and its operating constraints.

23. Keep the statistical rule as a benchmark.
    Compare models on the same entities, as-of features, periods, action horizon, and queue budget.
    A complex model is not better merely because it generates a more varied score.
    If it adds no measurable value at the available capacity, prefer the easier-to-maintain policy.
    Compare a business-rule-only policy as well: some urgent risks are common rather than anomalous.

### 5.6 Convert scores into an operating policy

24. Use development data to fit and a later period to choose the threshold.
    Freeze feature definitions and detector parameters before examining the final test period.
    Replay time in order and preserve receipt-time availability.
    Split by time and, where needed, shipment or incident so repeated records do not leak across folds.
    Use a final untouched period to estimate the selected policy's performance.

25. Enumerate candidate thresholds and queue rules.
    For each threshold, compute case-level alerts, mature outcomes, actionable lead times, and total investigation minutes.
    Apply the same deduplication and cooldown that production will use.
    Include a no-alert and a current-rule baseline.
    The candidate set can include a top-k-per-shift policy, provided scores are comparable and a minimum risk gate prevents low-value filler alerts.
    Top-k alone can bury urgent cases when an entire shift is abnormal.

26. Select based on decision value and capacity, not a fashionable percentile.
    For a simple retrospective loss table, add false-alert review costs and missed-case losses.
    Include true-positive review costs and intervention costs if they are not already included.
    A case's preventable loss is not necessarily the full loss from the incident.
    Check the expected minutes and the busiest shifts; an average that fits may still overwhelm a peak shift.
    If no threshold fits safely, improve capacity, narrow scope, or redesign the workflow rather than hide alerts.

27. Assign an owner and explicit action.
    Show the entity, decision time, baseline, feature evidence, cutoff, remaining slack, and data freshness.
    State whether a case is advisory or authorizes an action.
    Provide a way to acknowledge, merge, dismiss with a reason, escalate, or reopen.
    Make repeated scores update the same case unless a documented state change justifies another notification.

28. Protect deterministic obligations.
    A required temperature excursion response, safety rule, or binding service commitment should not disappear because a statistical detector assigned a low anomaly score.
    Separate mandatory rules from discretionary prioritization.
    Route data-quality failures to a data owner rather than presenting them as confirmed logistics failures.
    Establish a human override and a logging path for emergency action.

### 5.7 Evaluate and monitor after launch

29. Define a denominator and label policy before launch.
    Count eligible shipment-leg visits and cases, not just messages.
    Define outcome cutoff, lead-time requirement, and maturity lag.
    Measure TP, FP, and FN on the same unit and window.
    Record unknown or censored outcomes separately, not as negatives.
    A useful alert can prevent its own positive outcome; an outcome-only label requires special interpretation.

30. Obtain evidence for unalerted cases.
    If only alert recipients are reviewed, precision is subject to selection and recall cannot be estimated from that review alone.
    Use independently available outcome records where possible.
    Otherwise audit a probability sample of alerted and unalerted eligible cases, record inclusion probabilities, and use an appropriate weighted estimator.
    Report sample sizes and uncertainty, especially for rare disruptions.
    Confirm the reference standard and adjudication process before drawing a causal conclusion.

31. Monitor at multiple layers.
    Track source freshness, missingness, duplicates, and schema changes before score distributions.
    Track eligible volume, alert rate per 100 eligible visits, case handling time, queue age, and capacity breach.
    Track score quantiles, feature distributions, and cohort mix against a named reference period.
    Once labels mature, track precision, recall where measurable, timeliness, and realized business outcomes.
    Segment by site and partner but suppress misleading conclusions from tiny denominators.

32. Distinguish drift types and diagnose before changing a threshold.
    Input drift means a feature distribution changed.
    Concept drift means the relationship between available features and an outcome changed.
    Prior or prevalence drift means the outcome rate changed.
    Measurement drift means capture, units, or semantics changed.
    These can occur together; a score shift alone does not identify which occurred.
    Compare source telemetry, operational changes, seasonality, and matured outcomes.

33. Use a documented change-control response.
    A broken unit conversion requires fixing and replaying the data pipeline, not retraining on corrupt measurements.
    A new service schedule may require a revised cohort and a newly validated threshold.
    A genuine disruption may require temporary surge staffing and a lower threshold rather than teaching the baseline that the disruption is normal.
    Shadow a candidate model, compare with the current policy on the same eligible units, and record approval before promotion.
    Specify rollback triggers, including missing inputs, alert storms, degraded lead time, or unexplained partner disparities.

34. Learn from interventions without confusing correlation with impact.
    Log action time, owner, intervention type, and cost.
    Compare outcomes using a defensible design, such as a randomized rollout where ethical and feasible, or a carefully specified observational comparison.
    Random assignment must respect safety and contractual obligations.
    A before-and-after improvement alone can reflect seasonality, changing freight mix, or the disruption ending.
    Use outcome evidence to improve both the detector and the operations that make alerts actionable.

### 5.8 Replay an event stream without hindsight

35. Establish the replay clock and the scoring schedule.
    Input: the event ledger, historical eligibility rules, and scoring cadence.
    Output: an ordered list of decision instants for each eligible visit.
    A 15-minute cadence means decisions at the designated quarter-hour boundaries, not at whichever instant produces the most favorable retrospective score.
    Align the clock to a named time zone and preserve offsets through daylight-saving transitions.
    If a cutoff moves, retain the version known at each decision instant.

36. Reconstruct the visible ledger at each instant.
    Include an event only when its receipt timestamp is no later than the decision timestamp.
    Apply only corrections received by that instant.
    If an arrival message is received after the trailer has physically departed, it cannot justify a historical alert before receipt.
    Conversely, a correction received later does not erase an alert that was reasonable on the original evidence.
    Preserve the initial decision and append the revision, with both receipt times.

37. Resolve entity and state as they were known then.
    Match the source event to the shipment-leg visit using the crosswalk valid at that instant.
    Reject impossible chronology rather than calculating a negative elapsed dwell.
    A canceled visit should leave the eligible population when the cancellation becomes known.
    An unreceived cancellation cannot be used to remove the visit from an earlier replay snapshot.
    Mark uncertain entity matches for investigation rather than copying the event into multiple visits.

38. Calculate features and apply freshness gates.
    Use the most recent valid arrival and cutoff known at the decision instant.
    Store input event IDs, reference cohort version, score version, and decision timestamp alongside the score.
    If a required input fails its freshness or validity rule, record abstention with its reason.
    Count abstentions in the coverage denominator so a low alert rate cannot conceal a broken feed.
    Keep mandatory obligations on their independent path even when the detector abstains.

39. Reproduce the case lifecycle, not just individual scores.
    A threshold crossing can create a case only when the case policy says a new case is permitted.
    Later qualifying scores normally update the existing case.
    Define whether a material worsening, approaching deadline, or new independent failure can reopen or escalate it.
    Model operator availability and dispatch delay if the historic record supports them.
    A case that waited in the queue beyond the action deadline is not an actionable alert merely because the score was timely.

40. Freeze the replay output before inspecting labels.
    The output is a case ledger with creation time, state transitions, owner, data-quality status, and deadline.
    Join mature outcomes only after the replay policy is fixed.
    Report the fraction of visits scored, abstained, alerted, reviewed on time, and unresolved.
    A retrospective that scores only visits with complete final data estimates performance on a selected population.
    Compare that population with all visits that would have entered the live eligibility rule.

### 5.9 Estimate value when labels and interventions are incomplete

41. Separate three questions before calculating a metric.
    Detection asks whether the policy identified an outcome under a stated observational label.
    Execution asks whether an operator reviewed the case before the last useful action time.
    Impact asks whether an intervention changed the outcome compared with what would otherwise have happened.
    A confusion matrix answers the first question only under its label definition.
    It cannot by itself answer the causal impact question.

42. Design the outcome review around the full eligible population.
    When independent departure records are available, use them to identify mature outcomes among alerted and unalerted visits.
    When review is necessary, sample both groups with known, nonzero inclusion probabilities.
    Retain the sampling frame, probability, adjudicator decision, and reason for missing review.
    Reviewers should use a consistent positive-outcome definition.
    If reviewers know an alert was raised, consider whether that knowledge can bias adjudication.

43. Weight audited cases according to the sampling design.
    For a simple stratified random sample, each reviewed case represents its stratum size divided by the number reviewed.
    Estimate total positives by adding weighted positive cases across alert and no-alert strata.
    Estimate detected positives from the weighted positive cases in the alert stratum.
    Their ratio estimates recall for the defined population and outcome under the sampling assumptions.
    The estimate is not exact: report the audit sizes and uncertainty appropriate to the sampling design.

44. Distinguish a prevented outcome from an unneeded alert.
    A visit that leaves on time after a dock reassignment might have left late without it, or might have left on time anyway.
    The observed record alone generally cannot identify which counterfactual occurred.
    A reviewer may label the action operationally justified based on contemporaneous evidence, but that is a different outcome from measured delay prevention.
    Store both the observed outcome and the adjudicated action rationale rather than replacing one with the other.
    Avoid using the phrase “false alert” without naming the evaluation label.

45. Choose an impact design that respects operational obligations.
    Where safe and authorized, randomize an advisory workflow among otherwise eligible discretionary cases.
    Define the unit of randomization, allocation, outcome window, and analysis before observing results.
    If cases share a dock or operator, actions on one visit can affect other visits.
    Consider shift- or site-level assignment to reduce this interference, subject to enough independent units.
    Never randomize away a mandatory safety response or a contractual action.

46. Compare the incremental economics of the intervention.
    Include staff review time for all cases, intervention cost for acted cases, and downstream loss in the same time period and currency.
    Compare with the actual current workflow, not an imaginary operation with no existing controls.
    Report effects on service, queue time, and total operating cost separately before combining them.
    If the causal effect is unknown, present a break-even intervention effect rather than claiming savings.
    Reassess the decision under low, central, and high estimates of loss and effect.

47. Close the learning loop with explicit ownership.
    Operations owns action feasibility and case handling.
    Data engineering owns event contracts and freshness.
    Analytics owns replay, score validity, uncertainty, and threshold evidence.
    The business sponsor approves capacity, cost assumptions, and the release decision.
    Quality, privacy, and legal owners review relevant obligations rather than delegating them to the model.
    A change request records the evidence, decision, affected cohorts, owner, and rollback condition.

## 6. Worked example

### Business situation and decision

A fictional regional cross-dock receives appliance parts from a carrier.
At 14:00 local time, an inbound shipment has been at the yard for 162 minutes.
The receiving team can call the carrier and rearrange a dock assignment in about 25 minutes.
The planned dock departure is 15:00 local time.
The target is to complete 95% of eligible visits within the site's service window, measured monthly.
The customer commitment is a 16:00 local-time delivery to a nearby assembly site; the delivery leg normally takes 50 minutes.
A separate ETA forecast, available at 14:00, gives a 15:45 point estimate, not a promise.
The operational cutoff for useful dock intervention is 15:00, not the customer commitment itself.
The team has a 30-minute action lead requirement, so an alert at 14:00 remains potentially actionable.

The eligible unit is one inbound shipment-leg visit.
A positive evaluation outcome is departure after 15:00 for that visit, provided a score was available by 14:30.
This simplified label is an observed late departure, not a measurement of what would have happened without intervention.
Every case in this worked retrospective has a mature label and one score at the same decision age.
The values below are instructional; they are not performance claims.

### Reference and robust score

A comparable cohort has nine completed reference dwell times, in minutes:

`76, 82, 89, 94, 101, 107, 116, 128, 141`.

The median is the fifth ordered value, 101 minutes.
The absolute deviations from 101 minutes are:

`25, 19, 12, 7, 0, 6, 15, 27, 40` minutes.

Sorted deviations are `0, 6, 7, 12, 15, 19, 25, 27, 40` minutes.
The MAD is therefore 15 minutes.
The normal-consistent robust scale is 1.4826 × 15 = 22.239 minutes.
At 162 minutes, the robust score is:

`(162 − 101) / 22.239 = 2.743` approximately.

The score is dimensionless.
At a candidate threshold of 2.0, the visit is flagged because 2.743 ≥ 2.0.
This calculation says elapsed dwell has already exceeded a typical completed dwell by a large amount relative to this small reference.
It does not say the chance of a missed cutoff is 99% or any other calibrated probability.
Nine observations are used here for hand arithmetic, not as a recommended production minimum.

The threshold in elapsed minutes is `101 + 2.0 × 22.239 = 145.478` minutes.
At 14:00, the visit is 16.522 minutes beyond that threshold.
If the arrival event was actually delayed or misidentified, the calculation can be wrong even though the arithmetic is correct.
The operator should check arrival source and event freshness before calling the carrier.

### Distance comparison

Suppose two standardized, as-of features are available: current dwell and remaining cutoff slack.
The current visit has coordinates `(2.0, -1.0)`, where each coordinate is dimensionless after historical scaling.
The three nearest eligible reference vectors are `(1.0, -0.5)`, `(0.5, 0.0)`, and `(0.0, -0.5)`.
Their Euclidean distances are:

- First: `sqrt((2.0 − 1.0)^2 + (-1.0 + 0.5)^2) = sqrt(1.25) = 1.118`.
- Second: `sqrt((2.0 − 0.5)^2 + (-1.0 − 0.0)^2) = sqrt(3.25) = 1.803`.
- Third: `sqrt((2.0 − 0.0)^2 + (-1.0 + 0.5)^2) = sqrt(4.25) = 2.062`.

With k = 2, the distance score is 1.803.
If the separately validated k = 2 threshold is 1.7, this detector also flags the visit.
The distance score and robust score are on different scales and must not be compared numerically.
The example assumes the scaler and reference were fitted before the decision and that no reference is a duplicate of this visit.

### Threshold choice on a later validation period

Suppose a subsequent validation period contains 100 eligible visits and 20 mature late-departure outcomes.
Each visit is scored once before the action deadline.
The results below are nested candidate thresholds of the same detector.

| Robust-score threshold | TP, visits | FP, visits | FN, visits | Alerts, cases | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| ≥ 1.5 | 16 | 24 | 4 | 40 | 16/40 = 40.0% | 16/20 = 80.0% |
| ≥ 2.0 | 14 | 11 | 6 | 25 | 14/25 = 56.0% | 14/20 = 70.0% |
| ≥ 2.5 | 9 | 5 | 11 | 14 | 9/14 = 64.3% | 9/20 = 45.0% |

Assume each investigated case costs USD 8 in staff time regardless of outcome.
For this simplified threshold comparison, each missed positive incurs an assumed incremental loss of USD 40.
Assume no additional intervention costs and do not count a loss for an alerted positive.
That is a decision model assumption, not proof that every intervention succeeds.
The resulting comparison is:

- Threshold 1.5: 40 × USD 8 + 4 × USD 40 = USD 480.
- Threshold 2.0: 25 × USD 8 + 6 × USD 40 = USD 440.
- Threshold 2.5: 14 × USD 8 + 11 × USD 40 = USD 552.
- No alerts: 0 × USD 8 + 20 × USD 40 = USD 800.

If capacity is at most 30 cases across the same validation period, threshold 1.5 is infeasible.
Threshold 2.0 is the lowest-cost feasible candidate under these assumptions.
A threshold of 2.5 has higher precision but misses seven more positives than threshold 1.5.
It is not automatically the best business policy.

The conclusion changes when loss assumptions change.
For example, with missed-case loss of USD 10 and the same USD 8 review cost, threshold 2.5 costs USD 222, threshold 2.0 costs USD 260, and no alerts cost USD 200.
Under that simplified model the no-alert policy is cheaper, absent mandatory obligations and other benefits.
Perform sensitivity analysis before treating one cost estimate as authoritative.

The aggregate capacity limit is not a guarantee that each shift fits.
Suppose all 25 alerts at threshold 2.0 arrive during one busy shift with capacity of 8 cases: the policy is not operationally feasible without a surge plan or different routing.
Inspect shift-level peaks, handling time, and unresolved backlog before launch.
An alternative ranking policy must still guarantee attention to mandatory and time-critical cases.

### Evaluation and action

A final later test period must be held out because the threshold was selected on validation data.
The test replay must reconstruct what was known at each decision time, including late messages and revisions.
Report uncertainty around 14 true positives out of 20 positives: small denominators make the 70% validation recall unstable.
Measure how many of the 14 arrived at least 30 minutes before the relevant deadline; do not infer timeliness from the score alone.

For the current 162-minute visit, the policy opens one case for the inbound supervisor.
The case includes the as-of arrival event, 162-minute elapsed dwell, 2.743 robust score, 15:00 cutoff, and an explanation that the baseline contains nine illustrative visits.
The supervisor checks freshness, calls the carrier, and decides whether to move a dock appointment under existing authority.
The system does not automatically promise the customer a new arrival time or spend money on an expedite.
If the carrier later corrects the arrival timestamp, retain the original decision record and update the case with the correction.

### A separate audit of unalerted visits

The following fictional audit illustrates how to estimate recall when complete outcome records are not available.
It is separate from the 100-visit threshold comparison above.
During one month, 500 eligible visits reach the decision point.
The frozen policy alerts 100 visits and does not alert 400 visits.
The analyst samples 20 alerted visits and 40 unalerted visits uniformly within their respective strata.
All 60 selected visits receive a mature, independently adjudicated outcome.
The review finds 12 positive outcomes among the 20 alerted visits and 4 among the 40 unalerted visits.

| Stratum | Population, visits | Reviewed, visits | Positive reviews | Weight per review |
|---|---:|---:|---:|---:|
| Alerted | 100 | 20 | 12 | 100 / 20 = 5 |
| Unalerted | 400 | 40 | 4 | 400 / 40 = 10 |

The estimated number of alerted positive visits is `12 × 5 = 60` visits.
The estimated number of unalerted positive visits is `4 × 10 = 40` visits.
Estimated total positives are therefore `60 + 40 = 100` visits.
Estimated recall is `60 / 100 = 60%`.
Estimated precision is `60 / 100 = 60%`, equivalently `12 / 20` within the alerted stratum.
The identical percentages in this example are coincidental.
The sample's raw positive fraction, `(12 + 4) / 60 = 26.7%`, is not an estimate of population prevalence because the alert stratum was sampled at a higher rate.
The weighted prevalence estimate is `100 / 500 = 20%`.

This calculation assumes that the stratum populations and sample probabilities are correct.
It also assumes that every selected case was adjudicated and that selection within each stratum was random.
If four difficult unalerted cases could not be reviewed, the original weights alone would not repair the missingness.
Investigate nonresponse and use an appropriate design-based adjustment or report the resulting limitation.
The estimate is uncertain because only four unalerted positives were observed.
Use a confidence interval or another uncertainty assessment appropriate to stratified sampling before comparing a small change in recall.
The audit estimates association with the observed outcome; it does not establish whether alerting prevented late departures.

### Break-even value of the alert workflow

Return to the 100-visit validation example and compare the 2.0 threshold with no alerts.
The threshold generates 25 reviews and identifies 14 observed positives.
At USD 8 per review, incremental review cost is `25 × USD 8 = USD 200`.
Suppose each reviewed case also incurs an average additional intervention cost of USD 3.
Total incremental workflow cost is `25 × (USD 8 + USD 3) = USD 275`.
This intervention-cost assumption is new and is not included in the earlier simplified loss table.
Assume, only for this sensitivity calculation, that all 14 identified positives are eligible for an effective intervention.
Let `p` be the fraction of those 14 losses actually prevented by the workflow.
Let each prevented loss be valued at USD 40.

Expected avoided loss is `14 × p × USD 40 = USD 560p`.
The workflow breaks even when `USD 560p = USD 275`.
Thus `p = 275 / 560 = 49.1%` approximately.
If the true preventable fraction were 30%, expected avoided loss would be USD 168 and net value would be minus USD 107.
If the true preventable fraction were 70%, expected avoided loss would be USD 392 and net value would be USD 117.
These are conditional scenarios, not estimates of actual causal impact.
The 14 retrospective positives may not represent the cases affected under a live intervention policy.
Some late departures may be impossible to prevent and some alerted on-time visits may have been rescued.
Intervention cost may apply only to acted cases rather than every reviewed case.
Replace the simplified quantities with prospective evidence and the actual action mix before approving a business case.

The decision owner should also check whether a delay avoided at the dock shifts congestion to another dock.
If so, the financial boundary must include that displaced work rather than crediting one shipment in isolation.
The customer commitment remains an obligation regardless of the break-even calculation.
The break-even result informs discretionary investment; it does not cancel a required response.

## 7. Code, query, or calculation

The following standalone Python 3 program uses only the standard library.
It reproduces the worked robust score and candidate threshold arithmetic.
It scores a single current elapsed dwell against completed reference dwell solely as an unusually long wait check.
It does not estimate the probability of missing the cutoff.
The small reference and cost model are illustrative and not production settings.

```python
from statistics import median


def robust_high_score(value, reference):
    """Return score, center, MAD and scale; reject an undefined scale."""
    if not reference:
        raise ValueError("reference must not be empty")
    center = median(reference)
    mad = median(abs(x - center) for x in reference)
    if mad == 0:
        raise ValueError("zero MAD: use an approved fallback or abstain")
    scale = 1.4826 * mad
    return (value - center) / scale, center, mad, scale


def evaluate(tp, fp, fn, review_cost, miss_cost):
    """Costs are USD; counts are case-level and labels are mature."""
    alerts = tp + fp
    precision = tp / alerts if alerts else None
    positives = tp + fn
    recall = tp / positives if positives else None
    loss = alerts * review_cost + fn * miss_cost
    return alerts, precision, recall, loss


reference_minutes = [76, 82, 89, 94, 101, 107, 116, 128, 141]
current_minutes = 162
score, center, mad, scale = robust_high_score(current_minutes, reference_minutes)
print(f"median={center:.0f} min, MAD={mad:.0f} min, scale={scale:.3f} min")
print(f"score={score:.3f}, threshold=2.0, alert={score >= 2.0}")

# Validation outcomes are explicitly supplied, not learned from the test set.
policies = [
    (1.5, 16, 24, 4),
    (2.0, 14, 11, 6),
    (2.5, 9, 5, 11),
]
capacity = 30  # cases over this specific validation period
for threshold, tp, fp, fn in policies:
    alerts, precision, recall, loss = evaluate(tp, fp, fn, 8, 40)
    print(
        f"threshold={threshold:.1f}, alerts={alerts}, "
        f"precision={precision:.3f}, recall={recall:.3f}, "
        f"loss=USD {loss}, feasible={alerts <= capacity}"
    )
```

Expected output:

```text
median=101 min, MAD=15 min, scale=22.239 min
score=2.743, threshold=2.0, alert=True
threshold=1.5, alerts=40, precision=0.400, recall=0.800, loss=USD 480, feasible=False
threshold=2.0, alerts=25, precision=0.560, recall=0.700, loss=USD 440, feasible=True
threshold=2.5, alerts=14, precision=0.643, recall=0.450, loss=USD 552, feasible=True
```

Production code also needs an event-time reconstruction, as-of joins, validated reference selection, case identity, freshness checks, and intervention logging.
Those are not simulated by the small program.
A zero MAD raises an error rather than quietly creating an infinite or arbitrary score.
An operational implementation must catch that condition and execute the approved fallback or abstention route.
The function does not enforce a minimum cohort size; the production policy must set and monitor one.

## 8. Data, standards, and regulatory context

### Data contract for an operational detector

A useful input contract names the entity, source, event semantics, timestamps, units, and allowed corrections.
For a yard-dwell detector it should include:

- A stable shipment and shipment-leg visit identifier, with a documented reassignment rule.
- A facility identifier and the role of that location, not merely a free-text name.
- Arrival and departure event types and their source systems.
- Event time, receipt time, time zone or offset, and timestamp precision.
- A unique event identifier or source-specific deduplication key.
- A revision or correction relationship when an earlier event is amended.
- Appointment and cutoff values with effective times and a source of authority.
- Shipment type and other context required for peer selection.
- Explicit missing, unknown, canceled, and invalid states.
- Data-access rights, retention periods, and accountable owners.

A message can be syntactically valid yet operationally misleading.
An arrival scan at a gate is not necessarily an arrival at a dock.
A time stored as local clock time without a time-zone offset can be ambiguous during daylight-saving transitions.
A late correction can be valuable for reporting while remaining unavailable for a historical live decision.
Keep both original and corrected records or otherwise preserve an auditable as-of history.

### GS1 EPCIS and event interoperability

GS1 Electronic Product Code Information Services (EPCIS) 2.0 is a standard for sharing visibility-event data.
It provides a common event model, including what, when, where, why, and how-related information, and supports sensor-related data.
A conforming representation can improve interpretation across organizations, but it does not certify that an event is true, timely, complete, or useful for a particular detector.
A trading-partner agreement still needs to define identifiers, business steps, timing expectations, correction behavior, and access.

GS1 identifiers and location vocabularies help distinguish the same shipment or site across parties when partners apply them consistently.
A local identifier may still be necessary, with a maintained crosswalk.
Do not infer that every logistics update in a control tower is an EPCIS event; telematics, appointment systems, and proprietary APIs can have different semantics.
A detector's data contract should name the actual source and mapping rather than assuming a standard eliminates integration work.

### Privacy and commercial constraints

Shipment events can contain personal data when they identify a driver or are linkable to a person's location or work patterns.
The European Union General Data Protection Regulation (GDPR) applies to qualifying processing and includes principles such as purpose limitation, data minimization, accuracy, and storage limitation; applicability depends on the facts.
Article 22 addresses certain solely automated decisions with legal or similarly significant effects, subject to its conditions and exceptions.
It does not mean that every automated logistics alert is prohibited or that a human click automatically resolves every legal issue.
Consult qualified counsel and the organization's privacy team about the specific processing and jurisdiction.

Use the least identifying data needed for a shipment-level decision.
Define role-based access and retention for raw events, scores, case notes, and audit logs.
A case note can introduce sensitive free text that was absent from structured telemetry.
Sharing partner-specific anomaly scores can also expose commercially sensitive service patterns.
Specify permitted uses, recipients, correction procedures, and dispute handling in the relevant agreement.

### Operational and model governance

The NIST AI Risk Management Framework is voluntary guidance, not a universal legal certification for an anomaly detector.
Its govern, map, measure, and manage functions provide a useful way to assign accountability and evaluate risks throughout a system's life cycle.
A detector may be a simple rule rather than an AI system; the same practical controls around ownership, data quality, and change approval still help.
Document what is measured, who approves a threshold, what triggers review, and how an affected operator can contest an incorrect case.

Regulatory requirements may be sector-specific.
For example, a temperature-sensitive pharmaceutical shipment may have obligations or contractual quality procedures independent of an anomaly score.
Do not use a statistical baseline to waive a validated excursion procedure.
The responsible quality and legal functions must identify applicable requirements for the product, route, and jurisdiction.

## 9. Industry practice and case evidence

### Public evidence: supply-chain monitoring as a business capability

The U.S. Federal Highway Administration's Freight Analysis Framework describes public freight flow data and forecasts used for transportation planning.
It illustrates that freight networks are heterogeneous by origin, destination, commodity, and mode.
It does not provide evidence that a particular private carrier uses the robust-score policy in this lesson.
**Inference:** this heterogeneity supports testing peer groups by relevant operational context rather than imposing one universal dwell distribution.

The U.S. Food and Drug Administration's Food Safety Modernization Act (FSMA) traceability rule describes recordkeeping requirements for certain foods and entities, subject to the rule's scope and applicable dates.
Its traceability focus illustrates the importance of being able to locate and interpret event records across a supply chain.
It is not a mandate to deploy an anomaly detector, nor evidence that an anomaly score satisfies traceability obligations.
Check the regulator's current implementation and compliance information before making a legal or project decision.
**Inference:** a traceability event can support investigation, but detector quality still depends on timely capture and correct entity linkage.

### Public evidence: detection methods

The original isolation forest research by Liu, Ting, and Zhou describes isolation as a method for anomaly detection.
It supports the methodological description that anomalous points can be isolated with shorter random-tree paths.
It does not establish the accuracy of a particular supply-chain implementation or demonstrate business savings.
A control tower must validate the features and policy on its own historical stream and current decision process.

NIST's Engineering Statistics Handbook discusses control charts and their role in distinguishing process variation from signals that may merit investigation.
It does not claim that a control limit is equivalent to a contractual service limit.
**Inference:** a control chart can help a site investigate sustained process change, while a separate business rule can protect individual urgent shipments.

### Three illustrative operating cases

**Case A: inbound dwell at a cross-dock.**
A supervisor can move a dock assignment before the departure cutoff.
The detector compares elapsed dwell with an appropriate historical peer group, while a separate rule checks remaining slack.
The useful outcome is not merely finding long dwell; it is avoiding a late departure through an available intervention.
If arrival scans are often delayed, a data-quality indicator must accompany the score.

**Case B: temperature excursions in refrigerated transport.**
A quality team receives readings from a sensor and applies a product-specific validated range and excursion procedure.
An anomaly detector can identify unusual sensor patterns or missing telemetry for investigation.
It cannot replace a mandatory product-quality response simply because the event is statistically common.
A repeated sensor reading may indicate either an unusually stable environment or a stuck device; battery and communication evidence help distinguish them.

**Case C: inventory movements at a distribution center.**
An inventory controller investigates unusual negative adjustments by location, item class, and shift.
A point detector flags an extreme adjustment; a collective detector might flag a run of modest adjustments.
Neither proves theft or employee misconduct.
The controller checks cycle-count procedures, unit-of-measure changes, and system migrations before escalating.
Privacy and employment considerations require careful access and review.

These are fictional applications, not claimed implementations by named companies.
Across all three, documented actions, timing, and outcomes are necessary to judge operational value.
A model can detect unusual data while the business remains unable to correct the underlying issue.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Mixed peers | One lane dominates alerts. | Global baseline combines different service patterns. | Routine work crowds out rare events elsewhere. | Segment on meaningful context; validate fallback cohorts. |
| Tiny cohort | Threshold changes sharply after one visit. | Sample quantile or MAD is unstable. | Inconsistent case creation. | Minimum sample, pooled parent, and uncertainty reporting. |
| Zero MAD | Many identical reference values. | Robust scale is zero. | Undefined or artificially extreme scores. | Explicit fallback or abstention with a quality signal. |
| Unit mismatch | Values jump by a factor near 60. | Minutes and hours are mixed. | Alert storm or missed delays. | Validate units and ranges at ingestion; repair and replay. |
| Clock ambiguity | Negative elapsed dwell near clock changes. | Local times lack offsets or clocks differ. | Invalid scores and disputes. | Preserve offsets, validate chronology, quarantine uncertainty. |
| Late events | Backfill materially changes past features. | Replay uses eventual rather than received-by-time data. | Overstated backtest performance. | Preserve receipt time and as-of revisions. |
| Duplicate scans | Alert counts track message retries. | Multiple events map to one physical occurrence. | Repeated cases and inflated workload. | Stable deduplication keys and case-level aggregation. |
| Wrong label | Rescued cases are labeled negative. | Intervention changes observed outcome. | Useful alerts appear harmful. | Log actions; use adjudication and causal evaluation where feasible. |
| Selective review | Labels exist only for alerts. | Unalerted cases never enter review. | Recall is unidentifiable. | Independent outcomes or probability-sampled audits. |
| Late label | Recent precision looks unusually high or low. | Unresolved outcomes are treated as negatives. | Premature threshold changes. | Enforce maturity lag and report unknowns. |
| Feature leakage | Test accuracy collapses live. | Final dwell or later corrections enter features. | False confidence and missed incidents. | As-of replay and feature availability checks. |
| Training contamination | Threshold rises during a long disruption. | Failure becomes part of baseline. | Detector normalizes a bad process. | Review baseline windows and retain business limits. |
| Score sign reversal | Worst cases disappear after upgrade. | Library score orientation differs. | Silent inversion of ranking. | Version contracts and known-example checks. |
| Alert overload | Queue age rises above action window. | Capacity is exceeded at peaks. | Important cases expire unattended. | Shift-level capacity, escalation, surge plans, and deduplication. |
| Overaggressive cooldown | New deterioration is hidden. | Suppression ignores severity changes. | Escalation arrives too late. | Define reopen and severity-change rules. |
| Threshold gaming | Score improves but service does not. | Staff alter inputs or targets to avoid alerts. | Misleading metrics and unresolved harm. | Audit source changes and independent outcomes. |
| New network regime | Score distribution changes after lane launch. | Old peers no longer represent operations. | False alarms or blind spots. | Assess regime, shadow alternatives, approve change. |
| Missing-data masking | Imputed values look ordinary. | Nulls become typical values. | Pipeline failure is concealed. | Separate missingness flags and quality routing. |
| Automation overreach | System reroutes without authority. | Alert is treated as an instruction. | Cost, safety, or contractual harm. | Explicit decision rights and human override. |
| Unsafe suppression | Common excursion no longer alerts. | Statistical normality overrides hard rule. | Required response is missed. | Independent mandatory controls and periodic audit. |
| Correlated features | Distance scores cluster despite clear problems. | Redundant dimensions distort geometry. | Poor ranking and hard-to-explain cases. | Inspect scaling and neighbors; validate simpler features. |
| Intervention confounding | Alerted visits look worse than unalerted ones. | Operators select harder cases for action. | Incorrect conclusion that intervention caused delay. | Record selection; use suitable experimental or observational design. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Arrival is correctly identified. | Elapsed dwell requires a valid start. | Gate and dock events are confused. | Compare source semantics and audit a sample against yard records. |
| Receipt history is retained. | Honest replay needs available-by-time data. | A warehouse stores only the final corrected state. | Trace revisions and build an immutable or versioned history. |
| Reference visits are comparable. | A score needs a meaningful notion of normal. | A new lane, mode, or appointment process changes the mix. | Segment diagnostics and operator review; use fallback cohorts. |
| Reference size is adequate. | Estimates need enough information to be stable. | Sparse lane or rare event. | Bootstrap or examine rolling stability; pool or abstain. |
| Robust scale is nonzero. | The score divides by scale. | Repeated identical observations. | Detect zero MAD and execute a named fallback. |
| Reference is not dominated by prolonged failures. | A baseline should represent the intended comparison. | Persistent congestion contaminates training. | Compare multiple periods and operational incident logs. |
| The action window remains open. | An alert has value only if someone can act. | Alert arrives after dispatch cutoff. | Measure lead-time distribution against action duration. |
| Review capacity is available. | An unhandled alert cannot trigger the intended response. | Peak volume exceeds staffing. | Monitor shift-level minutes and backlog, not only average volume. |
| Outcome labels mature reliably. | Precision and recall require comparable outcomes. | Cancellations or unreported departures remain unresolved. | Track unknowns and adjudicate or sample. |
| Cost estimates reflect the decision. | Threshold optimization depends on relative loss. | Expedite costs or unsuccessful interventions are omitted. | Use ranges, sensitivity analysis, and action-level records. |
| Scores preserve their meaning. | A threshold presumes a stable scoring contract. | Library update reverses sign or changes scaling. | Version and compare known examples before promotion. |
| Historical intervention policy is understood. | Outcomes may be changed by actions. | Operators selectively rescue the worst cases. | Log intervention; use a defensible causal design. |
| A statistical anomaly is relevant. | Unusualness must connect to an action. | Rare benign shipments dominate top scores. | Audit cases with operators and measure decision value. |
| Mandatory rules are separate. | Legal and safety obligations may not be statistical. | A combined score suppresses a required response. | Independent rule path and policy review. |

None of these assumptions guarantees external validity.
A detector validated at one cross-dock, during one season, and with one carrier may fail elsewhere.
When a case is outside the validated population, the correct answer may be a business rule, broader cohort, or no model score.
Abstention should produce an owned workflow when the underlying situation can still be dangerous.

## 12. Exercises

### Exercise 1 — Frame the decision

Time: 15 minutes.
A distribution center receives 400 inbound visits per week.
The yard team can call a carrier or change a dock assignment but cannot approve premium freight.
A customer cutoff is 17:00 local time and a useful intervention requires 45 minutes.
A late scan can arrive two hours after the physical event.

Deliverable: a one-page decision statement defining entity, owner, eligible period, actionable deadline, action, and one data-quality branch.
Acceptance criteria: distinguish the 16:15 action deadline from the 17:00 cutoff; do not use the late scan before receipt; do not give the yard team unauthorized expedite rights.

### Exercise 2 — Robust score and zero scale

Time: 15 minutes.
A reference contains dwell values `40, 45, 50, 55, 60` minutes.
The current elapsed dwell is 80 minutes.
Use the normal-consistent robust scale from this lesson and a high-side threshold of 2.0.
Now replace the reference with `50, 50, 50, 50, 70` minutes.

Deliverable: median, MAD, scale, score, threshold decision, and a zero-scale response for the second reference.
Acceptance criteria: show deviations and units; do not divide by zero or claim a calibrated probability.

### Exercise 3 — Nearest neighbors

Time: 15 minutes.
A current standardized vector is `(2, 1)`.
Reference vectors are `(1, 1)`, `(2, 3)`, and `(5, 1)`.
Use Euclidean distance and k = 2; flag a score strictly greater than 1.8.

Deliverable: all three distances, kth-neighbor score, decision, and one condition that would invalidate the comparison.
Acceptance criteria: sort the distances; use a fitted historical scaler; exclude self-matches.

### Exercise 4 — Critique a detector claim

Time: 20 minutes.
A vendor says, “Our isolation forest marks the top 5% of scores, so each alert has a 95% chance of being a genuine late shipment.”
The evaluation uses final dwell and a random split of repeated scans from the same shipment.

Deliverable: a short critique and three corrective design changes.
Acceptance criteria: distinguish a score percentile from probability; identify as-of leakage and correlated split leakage; request mature case-level evaluation.

### Exercise 5 — Choose a threshold

Time: 20 minutes.
In a validation period, policy A has 12 TP, 18 FP, and 8 FN; policy B has 10 TP, 6 FP, and 10 FN.
Each alert costs USD 5 to review; each missed positive costs USD 20 in the simplified model.
Capacity is 20 alerts in that period.

Deliverable: precision, recall, modeled cost, feasibility, and chosen policy.
Acceptance criteria: count all alerts in review cost, apply capacity, and state that a later test period is still needed.

### Exercise 6 — Labels and timeliness

Time: 20 minutes.
A team alerts 20 cases; 8 have mature positive outcomes, 6 mature negative outcomes, and 6 are unresolved.
Among 30 independently identified mature positive cases in the eligible population, the 8 alerted positives are included.
Only 5 of those 8 were alerted with at least 30 minutes of lead time.

Deliverable: labeled-alert precision, recall for the specified mature cohort, timely recall, and a statement about unresolved cases.
Acceptance criteria: do not label unresolved cases negative; use a denominator of 30 only for the explicitly defined mature cohort; note possible selection bias.

### Exercise 7 — Diagnose drift

Time: 25 minutes.
An update changes the arrival source from gate scans to dock scans.
After release, dwell decreases and alert volume drops by 60% at one site.
At the same time, the site introduces a new appointment schedule.
Outcome labels will mature in two weeks.

Deliverable: an investigation sequence and temporary operating decision.
Acceptance criteria: distinguish a semantic measurement change from genuine operational change; preserve mandatory controls; do not automatically retrain before understanding the source.

### Exercise 8 — Deployment record

Time: 30 minutes.
A proposed detector scores shipment-leg visits every 15 minutes and creates a new notification each time the score crosses its threshold.
There are three shifts, a 20-case daily average capacity, and no recorded intervention outcomes.

Deliverable: a one-page deployment decision record with baseline, threshold selection, deduplication, capacity, labels, monitoring, authority, and rollback.
Acceptance criteria: specify case-level aggregation, shift peaks, as-of evaluation, action rights, and a measurable rollback trigger; identify unknown costs and label gaps.

### Exercise 9 — Weighted audit and value

Time: 25 minutes.
A policy alerts 80 of 400 eligible visits.
An auditor randomly reviews 20 alerted visits and 40 unalerted visits, with complete mature outcomes.
Ten reviewed alerted visits and five reviewed unalerted visits are positive.
The 80 reviews in the live workflow cost USD 6 each.
Suppose the workflow prevents a loss worth USD 30 for each prevented positive outcome.

Deliverable: estimate positive counts in both strata and recall, then calculate the number of losses that must be prevented to cover review cost.
Acceptance criteria: use separate stratum weights, show the denominator, and explain why observed positives do not establish the number prevented.

## 13. Self-check answers

### Exercise 1

A defensible entity is one shipment-leg visit at the distribution center.
The yard supervisor owns the review, carrier call, and dock-assignment decision.
At a 17:00 cutoff with a 45-minute intervention duration, a useful alert must arrive by 16:15, assuming the full 45 minutes is needed and no other buffer is required.
Use only arrival information received by the decision time.
If the physical arrival is known only from a scan received two hours later, route the missing or stale evidence to a data-quality workflow rather than reconstructing it into the original live snapshot.
Premium freight requires a separate authorized decision maker.
A strong answer also states how cancellations, visits already departed, and conflicting timestamps are handled.

### Exercise 2

The median is 50 minutes.
The absolute deviations are `10, 5, 0, 5, 10` minutes; their median is 5 minutes.
The scale is `1.4826 × 5 = 7.413` minutes.
The score is `(80 − 50) / 7.413 = 4.047` approximately, so it exceeds 2.0.
For `50, 50, 50, 50, 70`, the median is 50 minutes and the MAD is 0 minutes.
The robust score is undefined.
Use a documented parent cohort, another validated scale, a business rule, or abstention; do not replace zero with an arbitrary epsilon.

### Exercise 3

Distances are 1, 2, and 3 dimensionless units respectively.
The second-nearest distance is 2.
Because 2 > 1.8, the point is flagged.
A valid comparison assumes the same historical scaling and comparable reference population.
Including the current visit or a duplicate of it in the reference can create a misleadingly small distance.
The score is not a probability of an operational loss.

### Exercise 4

The top 5% describes the fraction selected by a threshold under a particular score distribution; it does not establish 95% precision or a per-case probability.
Final dwell is unavailable before departure and creates temporal leakage if used for a live score.
Repeated scans from the same shipment can leak near-duplicates across random splits.
Rebuild features as of decision time, split chronologically and by shipment or incident as appropriate, and evaluate on mature case-level labels.
Include action lead time, capacity, and an unbiased way to observe unalerted positives.
Review score direction and the role of the implementation's contamination parameter.

### Exercise 5

Policy A generates 30 alerts, precision `12/30 = 40%`, recall `12/20 = 60%`, and modeled cost `30 × USD 5 + 8 × USD 20 = USD 310`.
It exceeds capacity of 20 alerts.
Policy B generates 16 alerts, precision `10/16 = 62.5%`, recall `10/20 = 50%`, and modeled cost `16 × USD 5 + 10 × USD 20 = USD 280`.
It is feasible and has lower modeled cost among the two choices.
Select B provisionally, subject to shift-level capacity, cost sensitivity, mandatory rules, and evaluation on a later untouched test period.
The data do not establish that B is optimal among all possible policies.

### Exercise 6

Mature labeled-alert precision is `8 / (8 + 6) = 57.1%` approximately.
Six unresolved cases remain unknown and are excluded from that denominator.
Recall for the stated mature cohort is `8 / 30 = 26.7%` approximately.
Timely recall with a 30-minute lead requirement is `5 / 30 = 16.7%` approximately.
These are conditional on the stated mature cohort and its ascertainment.
If unresolved outcomes or unalerted cases mature differently, neither metric necessarily represents the full eligible population.

### Exercise 7 rubric

A strong investigation freezes an audit trail of the release and checks source definitions, timestamp mappings, units, and receipt latency first.
It compares paired gate and dock events where both exist and quantifies the timing difference.
It separately documents appointment changes and compares appropriate cohorts before attributing causation.
Until the issue is understood, keep independent mandatory rules and route questionable data to an owner.
If the source change invalidates the detector, pause or fall back under the approved policy rather than quietly lowering the threshold.
When labels mature, compare timeliness and outcomes, then validate any revised baseline on a later period.

### Exercise 8 rubric

A strong record names a responsible owner, eligible shipment-leg visits, an as-of baseline, and a later temporal validation and test.
It aggregates repeated 15-minute scores into a case, with explicit escalation and reopening rules.
It checks each shift's investigation minutes and backlog, not only 20 cases per day on average.
It defines mature outcomes, intervention logs, and an audit of unalerted cases or an independent source of outcomes.
It keeps the model advisory unless a named authority permits action.
It sets measurable rollback conditions, such as a sustained input freshness failure or queue age beyond the action window, with a named person empowered to act.
A weak record claims a precision target without specifying labels, denominators, or how missing outcomes are handled.

### Exercise 9

The alerted sampling weight is `80 / 20 = 4` and the unalerted weight is `320 / 40 = 8`.
Estimated alerted positives are `10 × 4 = 40` visits.
Estimated unalerted positives are `5 × 8 = 40` visits.
Estimated recall is `40 / (40 + 40) = 50%`.
The review cost is `80 × USD 6 = USD 480`.
At USD 30 per prevented loss, break-even requires `480 / 30 = 16` prevented losses.
Observed positive outcomes do not reveal the outcomes that would have occurred without the workflow.
The estimate also has sampling uncertainty and assumes complete, unbiased adjudication.

## 14. Interview questions

1. **How would you define an anomaly for a control tower?**
   Strong: names the entity, reference, decision, time, action, and distinction between unusualness and impact.
   Weak: treats any large value as an incident.
2. **When would you use a robust statistical rule?**
   Strong: discusses a transparent baseline, skew, reference selection, sample stability, and zero MAD.
   Weak: assumes robust statistics eliminate the need for data-quality checks.
3. **Why can a stable process still require an alert?**
   Strong: separates statistical control from business specifications and commitments.
   Weak: assumes normal variation is automatically acceptable.
4. **What makes distance meaningful?**
   Strong: explains feature availability, scaling, metric, cohort, and duplicate/self-neighbor handling.
   Weak: computes Euclidean distance over raw mixed units without qualification.
5. **What does isolation forest contamination mean?**
   Strong: distinguishes an implementation's threshold setting from known prevalence and calibrated probability.
   Weak: calls a 5% contamination setting 95% alert precision.
6. **How do you backtest a live detector?**
   Strong: reconstructs received-by-time features, splits chronologically, groups related cases, and reserves a final test period.
   Weak: uses final corrected records and randomly splits scans.
7. **How would you select a threshold with limited staffing?**
   Strong: compares cost and value at case level, accounts for shift peaks, lead time, and mandatory rules.
   Weak: chooses the highest precision without checking missed cases or workload.
8. **Can you measure recall by reviewing only alerts?**
   Strong: explains missing unalerted positives and proposes independent outcomes or probability sampling.
   Weak: divides correct alerts by all alerts and calls that recall.
9. **What changes when an intervention prevents a miss?**
   Strong: notes outcome labels become treatment-dependent and calls for action logs and defensible causal evaluation.
   Weak: automatically marks every on-time alerted shipment false positive.
10. **What would you do after an alert-rate shift?**
    Strong: checks data semantics and freshness, mix, seasonality, operations, and mature outcomes before recalibration.
    Weak: immediately retrains on the latest data or suppresses alerts.
11. **How do you prevent an alert storm from hiding a critical case?**
    Strong: combines case aggregation, capacity monitoring, escalation, hard-rule protection, and surge plans.
    Weak: silences every repeated alert for a fixed period regardless of severity.
12. **How would you explain the model to an operator?**
    Strong: shows the baseline, data freshness, evidence, uncertainty, available action, and owner.
    Weak: presents a score as a probability without validation or context.

## 15. Key terms

- [Action window](#51-define-the-decision-before-the-score): interval in which intervention can still affect the outcome.
- [Alert](#4-definitions-and-vocabulary): notification that an alert policy condition was met.
- [As-of feature](#52-build-a-time-correct-observation): feature available using information received by decision time.
- [Baseline](#4-definitions-and-vocabulary): reference defining typical observations for a comparison.
- [Calibration](#4-definitions-and-vocabulary): agreement between predicted probabilities and observed frequencies.
- [Case](#4-definitions-and-vocabulary): managed unit of work that can combine alerts.
- [Censoring](#4-definitions-and-vocabulary): incomplete observation of a duration or outcome.
- [Cohort](#4-definitions-and-vocabulary): comparable population used for a reference or evaluation.
- [Commitment](#51-define-the-decision-before-the-score): agreed obligation, distinct from a forecast.
- [Contamination](#54-construct-distance-detectors): anomalous or erroneous observations in a reference set.
- [Control limit](#53-construct-statistical-detectors): statistical boundary used within a specified process-monitoring method.
- [Cooldown](#56-convert-scores-into-an-operating-policy): interval for consolidating repeated notifications.
- [Decision time](#52-build-a-time-correct-observation): instant at which a score and action could be produced.
- [Distance](#54-construct-distance-detectors): defined dissimilarity between feature vectors.
- [Drift](#57-evaluate-and-monitor-after-launch): change in data, measurement, outcomes, or relationships.
- [Dwell](#4-definitions-and-vocabulary): elapsed or completed time at a location, depending on definition.
- [False negative](#4-definitions-and-vocabulary): eligible positive case not alerted within the specified window.
- [False positive](#4-definitions-and-vocabulary): alerted case with a mature negative evaluation label.
- [Isolation forest](#55-construct-model-based-detectors): random-partition ensemble producing an anomaly score.
- [Label maturity](#57-evaluate-and-monitor-after-launch): time by which an outcome is sufficiently observed for evaluation.
- [Lead time](#4-definitions-and-vocabulary): time between alert and outcome or action deadline.
- [Leakage](#52-build-a-time-correct-observation): use of information unavailable at the intended decision.
- [MAD](#53-construct-statistical-detectors): median absolute deviation from the median.
- [Nearest neighbor](#54-construct-distance-detectors): closest reference point under a chosen metric.
- [Precision](#57-evaluate-and-monitor-after-launch): true positives divided by mature labeled alerts.
- [Recall](#57-evaluate-and-monitor-after-launch): alerted positives divided by eligible positives.
- [Residual](#55-construct-model-based-detectors): observed value minus the model's prediction.
- [Robust score](#reference-and-robust-score): deviation from median divided by robust scale.
- [Threshold](#56-convert-scores-into-an-operating-policy): boundary used by a policy to create or prioritize cases.
- [Watermark](#52-build-a-time-correct-observation): stream-processing convention for expected event-time completeness.

## 16. Sources

The following sources support definitions, methods, or legal context, not the fictional numerical results.
Access dates and operational applicability should be checked when implementing a policy.

1. GS1, [EPCIS Standard](https://www.gs1.org/standards/epcis), including EPCIS 2.0 specifications and supporting materials.
   Primary standards body; describes interoperable visibility-event structures, not event truth or detector performance.
2. Liu, F. T., Ting, K. M., and Zhou, Z.-H., “Isolation Forest,” *2008 Eighth IEEE International Conference on Data Mining*, pp. 413–422, [DOI: 10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17).
   Original method paper; does not validate this lesson's business policy.
3. NIST/SEMATECH, [Engineering Statistics Handbook: Control Charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc3.htm).
   Official statistical guidance on process monitoring and control charts.
4. NIST, [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework), AI RMF 1.0 and related resources.
   Voluntary risk-management guidance, not a universal legal certification.
5. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Articles 5 and 22.
   Binding legal text subject to its material and territorial scope; obtain jurisdiction-specific advice.
6. U.S. Food and Drug Administration, [Food Traceability Rule](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods).
   Official description of the rule; scope and compliance dates must be checked against current authority.
7. U.S. Federal Highway Administration, [Freight Analysis Framework](https://ops.fhwa.dot.gov/freight/freight_analysis/faf/).
   Official freight flow data and planning resource; not evidence of a particular detector deployment.
8. scikit-learn, [Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html).
   Implementation guidance; check the installed version for score orientation and API behavior.
9. Chandola, V., Banerjee, A., and Kumar, V., “Anomaly Detection: A Survey,” *ACM Computing Surveys* 41(3), 2009, [DOI: 10.1145/1541880.1541882](https://doi.org/10.1145/1541880.1541882).
   Survey of anomaly definitions and methods; not a claim of supply-chain business impact.

## 17. Links

- [Module 19 topic index](_index.md).
- [Visibility layers](01_visibility-layers.md): identify the business object and available view.
- [Event streams and EPCIS](02_event-streams-epcis.md): understand event semantics and partner data.
- [Event time versus processing time](03_event-time-processing-time.md): reconstruct what was known at a decision.
- [ETA prediction](04_eta-prediction.md): distinguish predicted arrival from a service commitment.
- [Delay propagation](05_delay-propagation.md): connect a local delay to downstream exposure.
- [Module 19 survey and lab exercises](../../../course/19_control_towers_visibility_events.md): module overview and applied work.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md): related applied modeling context.
